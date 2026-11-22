//! The 2-bit head shortlist's top-k pick and its kept-row argmax without a full sort.
//!
//! `topk` returns the same k indices, in the same order, as `argpartition(-y)[..k]` does on the GPU: MLX runs
//! argpartition as its stable merge sort (`mlx/backend/metal/sort.cpp` `ArgPartition::eval_gpu`, `sort.h` `LessThan`),
//! so the first k are the k largest scores, equal scores in ascending index order, NaN after every number. Two
//! dispatches: each threadgroup sorts one chunk of the row (bitonic, in threadgroup memory) and keeps its first k; one
//! threadgroup then merges the chunks' sorted lists k times.
//!
//! `headTail64` returns the token `headTailArgmaxMap` returns for the kept row the shortlist builds (the k rescored
//! scores at their positions, softcap(-1e30) everywhere else), from the k scores and their positions alone.
const std = @import("std");
const mlx = @import("mlx.zig");

/// Rows above this many chunks fall back (the merge runs one thread a chunk).
/// 256-score chunks (36 bitonic stages instead of 66 at 2,048); the served head row of 162,489 scores is 635
/// chunks, under the merge's 1,024.
const CH: c_int = 256;
const P: c_int = 256;
const TG1: c_int = 128;
pub const MAX_K: c_int = 64;

const HEADER =
    \\// a ranks before b: numbers before NaN, larger first, then the lower index (MLX's stable ascending sort of -y)
    \\inline bool shortlist_select_before(float av, uint ai, float bv, uint bi) {
    \\  bool an = metal::isnan(av);
    \\  bool bn = metal::isnan(bv);
    \\  if (an != bn) return bn;
    \\  if (!an && av != bv) return av > bv;
    \\  return ai < bi;
    \\}
    \\// headTailArgmaxMap's rule: NaN first (lowest index), else the larger, ties to the lower index
    \\inline bool shortlist_select_better(float v, uint i, float best, uint bidx) {
    \\  return (metal::isnan(v) && (!metal::isnan(best) || i < bidx)) || (!metal::isnan(best) && (best < v || (best == v && i < bidx)));
    \\}
    \\
;

const PART_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint g = threadgroup_position_in_grid.x;
    \\const float qnan = as_type<float>(0x7fc00000u);
    \\threadgroup float sv[P];
    \\threadgroup uint si[P];
    \\const uint base = g * uint(CH);
    \\for (uint t = tid; t < uint(P); t += uint(TG)) {
    \\  const uint i = base + t;
    \\  if (t < uint(CH) && i < uint(N)) { sv[t] = float(y[i]); si[t] = i; }
    \\  else { sv[t] = qnan; si[t] = 0xFFFFFFFFu; }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (uint k = 2; k <= uint(P); k <<= 1) {
    \\  for (uint j = k >> 1; j > 0; j >>= 1) {
    \\    for (uint t = tid; t < uint(P) / 2; t += uint(TG)) {
    \\      const uint i = 2 * j * (t / j) + (t % j);
    \\      const uint l = i + j;
    \\      const float av = sv[i];
    \\      const float bv = sv[l];
    \\      const uint ai = si[i];
    \\      const uint bi = si[l];
    \\      const bool sw = ((i & k) == 0) ? shortlist_select_before(bv, bi, av, ai) : shortlist_select_before(av, ai, bv, bi);
    \\      if (sw) { sv[i] = bv; sv[l] = av; si[i] = bi; si[l] = ai; }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\}
    \\for (uint t = tid; t < uint(K); t += uint(TG)) { cv[g * uint(K) + t] = sv[t]; ci[g * uint(K) + t] = si[t]; }
;

const MERGE_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sgi = simdgroup_index_in_threadgroup;
    \\const float qnan = as_type<float>(0x7fc00000u);
    \\threadgroup float tv[32];
    \\threadgroup uint ti[32];
    \\threadgroup uint tw[32];
    \\threadgroup uint win;
    \\uint head = 0;
    \\for (uint t = 0; t < uint(K); ++t) {
    \\  float v = qnan;
    \\  uint i = 0xFFFFFFFFu;
    \\  uint w = tid;
    \\  if (tid < uint(G) && head < uint(K)) { v = cv[tid * uint(K) + head]; i = ci[tid * uint(K) + head]; }
    \\  for (ushort off = 16; off > 0; off /= 2) {
    \\    const float ov = simd_shuffle_down(v, off);
    \\    const uint oi = simd_shuffle_down(i, off);
    \\    const uint ow = simd_shuffle_down(w, off);
    \\    if (shortlist_select_before(ov, oi, v, i)) { v = ov; i = oi; w = ow; }
    \\  }
    \\  if (lane == 0) { tv[sgi] = v; ti[sgi] = i; tw[sgi] = w; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sgi == 0) {
    \\    v = (lane < uint(TG) / 32) ? tv[lane] : qnan;
    \\    i = (lane < uint(TG) / 32) ? ti[lane] : 0xFFFFFFFFu;
    \\    w = (lane < uint(TG) / 32) ? tw[lane] : 0xFFFFFFFFu;
    \\    for (ushort off = 16; off > 0; off /= 2) {
    \\      const float ov = simd_shuffle_down(v, off);
    \\      const uint oi = simd_shuffle_down(i, off);
    \\      const uint ow = simd_shuffle_down(w, off);
    \\      if (shortlist_select_before(ov, oi, v, i)) { v = ov; i = oi; w = ow; }
    \\    }
    \\    if (lane == 0) { out[t] = i; win = w; }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (tid == win) head += 1;
    \\}
;

// One threadgroup. The kept row the shortlist builds holds the k rescored scores `v` at kept positions `sel` and
// fv = softcap(-1e30) at every other kept position (suppressed positions read -inf, as in headTailArgmaxMap). Its
// argmax is the best rescored entry unless that is not above fv; only then are the other positions scanned, in
// ascending order per thread, for the lowest unsuppressed one (or the lowest suppressed one when all are).
const HT64_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sgi = simdgroup_index_in_threadgroup;
    \\threadgroup float tv[32];
    \\threadgroup uint ti[32];
    \\threadgroup uint sk[K];
    \\threadgroup float rb;
    \\threadgroup uint rbi;
    \\float best = -INFINITY;
    \\uint bidx = 0xFFFFFFFFu;
    \\for (uint t = tid; t < uint(K); t += uint(TG)) {
    \\  const uint i = sel[t];
    \\  sk[t] = i;
    \\  float x = float(v[t]);
    \\  if (MASKED && bool(cmask[i])) x = -INFINITY;
    \\  if (shortlist_select_better(x, i, best, bidx)) { best = x; bidx = i; }
    \\}
    \\for (ushort off = 16; off > 0; off /= 2) {
    \\  const float ov = simd_shuffle_down(best, off);
    \\  const uint oi = simd_shuffle_down(bidx, off);
    \\  if (shortlist_select_better(ov, oi, best, bidx)) { best = ov; bidx = oi; }
    \\}
    \\if (lane == 0) { tv[sgi] = best; ti[sgi] = bidx; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sgi == 0) {
    \\  best = (lane < uint(TG) / 32) ? tv[lane] : -INFINITY;
    \\  bidx = (lane < uint(TG) / 32) ? ti[lane] : 0xFFFFFFFFu;
    \\  for (ushort off = 16; off > 0; off /= 2) {
    \\    const float ov = simd_shuffle_down(best, off);
    \\    const uint oi = simd_shuffle_down(bidx, off);
    \\    if (shortlist_select_better(ov, oi, best, bidx)) { best = ov; bidx = oi; }
    \\  }
    \\  if (lane == 0) { rb = best; rbi = bidx; }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\best = rb;
    \\bidx = rbi;
    \\const float f = float(fv[0]);
    \\if (!metal::isnan(best) && !(best > f)) {
    \\  float jb = -INFINITY;
    \\  uint jbi = 0xFFFFFFFFu;
    \\  for (uint i = tid; i < uint(NK); i += uint(TG)) {
    \\    bool in = false;
    \\    for (uint t = 0; t < uint(K); ++t) in = in || (sk[t] == i);
    \\    if (in) continue;
    \\    const float x = (MASKED && bool(cmask[i])) ? -INFINITY : f;
    \\    if (shortlist_select_better(x, i, jb, jbi)) { jb = x; jbi = i; }
    \\    if (x == f) break;
    \\  }
    \\  for (ushort off = 16; off > 0; off /= 2) {
    \\    const float ov = simd_shuffle_down(jb, off);
    \\    const uint oi = simd_shuffle_down(jbi, off);
    \\    if (shortlist_select_better(ov, oi, jb, jbi)) { jb = ov; jbi = oi; }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (lane == 0) { tv[sgi] = jb; ti[sgi] = jbi; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sgi == 0) {
    \\    jb = (lane < uint(TG) / 32) ? tv[lane] : -INFINITY;
    \\    jbi = (lane < uint(TG) / 32) ? ti[lane] : 0xFFFFFFFFu;
    \\    for (ushort off = 16; off > 0; off /= 2) {
    \\      const float ov = simd_shuffle_down(jb, off);
    \\      const uint oi = simd_shuffle_down(jbi, off);
    \\      if (shortlist_select_better(ov, oi, jb, jbi)) { jb = ov; jbi = oi; }
    \\    }
    \\    if (lane == 0 && jbi != 0xFFFFFFFFu && shortlist_select_better(jb, jbi, best, bidx)) { best = jb; bidx = jbi; }
    \\  }
    \\}
    \\if (tid == 0) {
    \\  uint tok = idx[bidx];
    \\  const float d = float(dv[0]);
    \\  if (!metal::isnan(best)) {
    \\    if (best == d) tok = metal::min(tok, di[0]);
    \\    else if (best < d) tok = di[0];
    \\  }
    \\  out[0] = tok;
    \\}
;
const HT64_TG: c_int = 1024;

// The top-k pick as a threshold select, generated from a reference
// implementation. A bf16 score's 16-bit key orders as
// shortlist_select_before does; two 256-bin histograms find each chunk's exact k-th key, a stable compaction keeps its k in
// index order, one threadgroup repeats it on the candidates and ranks the final k by key, then index.
const THRESHOLD_SELECT_HEADER =
    \\// A bf16 score's 16-bit key; unsigned order is the pick's order: larger first, +0 and -0 one key, every NaN key 1
    \\// (below every number, whose keys start at 0x007F for -inf), padding key 0
    \\inline uint threshold_select_key(float v) {
    \\  if (metal::isnan(v)) return 1u;
    \\  if (v == 0.0f) v = 0.0f;
    \\  const uint u = as_type<uint>(v);
    \\  const uint k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\  return k >> 16;
    \\}
    \\// exclusive prefix sum of x over the threadgroup's threads in thread order; total gets the sum
    \\inline uint threshold_select_exscan(uint x, threadgroup uint* st, uint lane, uint sgi, uint nsg, thread uint& total) {
    \\  const uint ex = simd_prefix_exclusive_sum(x);
    \\  const uint s = simd_sum(x);
    \\  if (lane == 0) st[sgi] = s;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  uint pre = 0, tot = 0;
    \\  for (uint j = 0; j < nsg; ++j) { const uint v = st[j]; if (j < sgi) pre += v; tot += v; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  total = tot;
    \\  return pre + ex;
    \\}
    \\// the bin b (of 256, summed over hc copies) with count(bins above b) < need <= count(bins at or above b): res[0] = b,
    \\// res[1] = the count above it; needs at least 256 threads
    \\inline void threshold_select_pick_bin(threadgroup atomic_uint* h, uint hc, uint need, uint tid, uint lane, uint sgi, uint nsg,
    \\                          threadgroup uint* st, threadgroup uint* res) {
    \\  uint c = 0;
    \\  const uint b = 255u - tid;
    \\  if (tid < 256u) for (uint s = 0; s < hc; ++s) c += atomic_load_explicit(&h[s * 256u + b], memory_order_relaxed);
    \\  uint tot;
    \\  const uint ex = threshold_select_exscan(c, st, lane, sgi, nsg, tot);
    \\  if (tid < 256u && ex < need && ex + c >= need) { res[0] = b; res[1] = ex; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\inline void threshold_select_clear(threadgroup atomic_uint* h, uint n, uint tid, uint tg) {
    \\  for (uint j = tid; j < n; j += tg) atomic_store_explicit(&h[j], 0u, memory_order_relaxed);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;
const THRESHOLD_SELECT_PART_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sgi = simdgroup_index_in_threadgroup;
    \\const uint g = threadgroup_position_in_grid.x;
    \\const uint nsg = uint(TG) / 32u;
    \\threadgroup atomic_uint h[HC * 256];
    \\threadgroup uint st[32];
    \\threadgroup uint res[2];
    \\uint key[E];
    \\uint idx[E];
    \\const uint base = g * uint(C) + tid * uint(E);
    \\for (uint e = 0; e < uint(E); ++e) {
    \\  const uint i = base + e;
    \\  if (i < uint(N)) { key[e] = threshold_select_key(float(y[i])); idx[e] = i; } else { key[e] = 0u; idx[e] = 0xFFFFFFFFu; }
    \\}
    \\device uint* ok_ = ck + g * uint(K);
    \\device uint* oi_ = ci + g * uint(K);
    \\
    \\  threshold_select_clear(h, uint(HC) * 256u, tid, uint(TG));
    \\  for (uint e = 0; e < uint(E); ++e) atomic_fetch_add_explicit(&h[(sgi % uint(HC)) * 256u + (key[e] >> 8)], 1u, memory_order_relaxed);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  threshold_select_pick_bin(h, uint(HC), uint(K), tid, lane, sgi, nsg, st, res);
    \\  const uint b1 = res[0];
    \\  const uint a1 = res[1];
    \\  threshold_select_clear(h, uint(HC) * 256u, tid, uint(TG));
    \\  for (uint e = 0; e < uint(E); ++e) {
    \\    if ((key[e] >> 8) == b1) atomic_fetch_add_explicit(&h[(sgi % uint(HC)) * 256u + (key[e] & 255u)], 1u, memory_order_relaxed);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  threshold_select_pick_bin(h, uint(HC), uint(K) - a1, tid, lane, sgi, nsg, st, res);
    \\  const uint thr = (b1 << 8) | res[0];
    \\  const uint ties = uint(K) - a1 - res[1];
    \\  uint tl = 0;
    \\  for (uint e = 0; e < uint(E); ++e) tl += (key[e] == thr) ? 1u : 0u;
    \\  uint tot;
    \\  uint r = threshold_select_exscan(tl, st, lane, sgi, nsg, tot);
    \\  uint sl = 0;
    \\  bool keep[E];
    \\  for (uint e = 0; e < uint(E); ++e) {
    \\    bool s = key[e] > thr;
    \\    if (key[e] == thr) { s = r < ties; r += 1u; }
    \\    keep[e] = s;
    \\    sl += s ? 1u : 0u;
    \\  }
    \\  uint pos = threshold_select_exscan(sl, st, lane, sgi, nsg, tot);
    \\  for (uint e = 0; e < uint(E); ++e) {
    \\    if (keep[e]) { ok_[pos] = key[e]; oi_[pos] = idx[e]; pos += 1u; }
    \\  }
;
const THRESHOLD_SELECT_MERGE_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sgi = simdgroup_index_in_threadgroup;
    \\const uint nsg = uint(TG) / 32u;
    \\threadgroup atomic_uint h[HC * 256];
    \\threadgroup uint st[32];
    \\threadgroup uint res[2];
    \\threadgroup uint sk[K];
    \\threadgroup uint si[K];
    \\uint key[E];
    \\uint idx[E];
    \\for (uint e = 0; e < uint(E); ++e) {
    \\  const uint j = tid * uint(E) + e;
    \\  if (j < uint(M)) { key[e] = ck[j]; idx[e] = ci[j]; } else { key[e] = 0u; idx[e] = 0xFFFFFFFFu; }
    \\}
    \\
    \\  threshold_select_clear(h, uint(HC) * 256u, tid, uint(TG));
    \\  for (uint e = 0; e < uint(E); ++e) atomic_fetch_add_explicit(&h[(sgi % uint(HC)) * 256u + (key[e] >> 8)], 1u, memory_order_relaxed);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  threshold_select_pick_bin(h, uint(HC), uint(K), tid, lane, sgi, nsg, st, res);
    \\  const uint b1 = res[0];
    \\  const uint a1 = res[1];
    \\  threshold_select_clear(h, uint(HC) * 256u, tid, uint(TG));
    \\  for (uint e = 0; e < uint(E); ++e) {
    \\    if ((key[e] >> 8) == b1) atomic_fetch_add_explicit(&h[(sgi % uint(HC)) * 256u + (key[e] & 255u)], 1u, memory_order_relaxed);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  threshold_select_pick_bin(h, uint(HC), uint(K) - a1, tid, lane, sgi, nsg, st, res);
    \\  const uint thr = (b1 << 8) | res[0];
    \\  const uint ties = uint(K) - a1 - res[1];
    \\  uint tl = 0;
    \\  for (uint e = 0; e < uint(E); ++e) tl += (key[e] == thr) ? 1u : 0u;
    \\  uint tot;
    \\  uint r = threshold_select_exscan(tl, st, lane, sgi, nsg, tot);
    \\  uint sl = 0;
    \\  bool keep[E];
    \\  for (uint e = 0; e < uint(E); ++e) {
    \\    bool s = key[e] > thr;
    \\    if (key[e] == thr) { s = r < ties; r += 1u; }
    \\    keep[e] = s;
    \\    sl += s ? 1u : 0u;
    \\  }
    \\  uint pos = threshold_select_exscan(sl, st, lane, sgi, nsg, tot);
    \\  for (uint e = 0; e < uint(E); ++e) {
    \\    if (keep[e]) { sk[pos] = key[e]; si[pos] = idx[e]; pos += 1u; }
    \\  }
    \\
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < uint(K)) {
    \\  const uint mk = sk[tid];
    \\  const uint mi = si[tid];
    \\  uint rank = 0;
    \\  for (uint j = 0; j < uint(K); ++j) rank += (sk[j] > mk || (sk[j] == mk && si[j] < mi)) ? 1u : 0u;
    \\  out[rank] = mi;
    \\}
;

var part_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;
var ht64_kernel: ?mlx.mlx_fast_metal_kernel = null;
var threshold_select_part_kernel: ?mlx.mlx_fast_metal_kernel = null;
var threshold_select_merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The threshold select's chunk (4,096 scores, 256 threads, 16 a thread), its histogram copies (one per
/// simdgroup in the part, 8 in the merge) and the merge's threadgroup (1,024 threads).
const THRESHOLD_SELECT_C: c_int = 4096;
const THRESHOLD_SELECT_TG: c_int = 256;
const THRESHOLD_SELECT_HC: c_int = 8;
const THRESHOLD_SELECT_TGM: c_int = 1024;
const THRESHOLD_SELECT_HCM: c_int = 8;
var threshold_select_off: ?bool = null;

/// MLX_SERVE_HEAD_SHORTLIST_THRESHOLD_OFF=1 keeps the sorted chunks for every dtype.
fn thresholdSelectOff() bool {
    if (threshold_select_off) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_HEAD_SHORTLIST_THRESHOLD_OFF")) |p| std.mem.eql(u8, std.mem.span(p), "1") else false;
    threshold_select_off = v;
    if (!v) std.debug.print("[threshold-select] top-k pick: threshold select on bf16 rows (chunk {d})\n", .{THRESHOLD_SELECT_C});
    return v;
}

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    return kernelWith(slot, name, ins, outs, src, HEADER);
}

fn kernelWith(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn apply(k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream, outs: []mlx.mlx_array) !void {
    const in_vec = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(out_vec) != outs.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (outs[0..got]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, out_vec, i)) catch |err| {
            _ = mlx.mlx_array_free(o.*);
            return err;
        };
        got = i + 1;
    }
}

/// The first `k` of `argpartition(-y)` along the last axis of the one-row `y`, as uint32 with `y`'s shape less its last
/// axis plus k. Null when the shape is not one row, k is outside 1 to MAX_K, or the row is longer than 1024 chunks.
pub fn topk(s: mlx.mlx_stream, y: mlx.mlx_array, k: c_int) !?mlx.mlx_array {
    const sh = mlx.getShape(y);
    if (sh.len < 1 or sh.len > 8) return null;
    const n = sh[sh.len - 1];
    if (n < 1 or mlx.mlx_array_size(y) != @as(usize, @intCast(n))) return null;
    if (k < 1 or k > MAX_K or k > n) return null;
    // the kernels are Metal: a CPU stream keeps argpartition; longer rows than 1024 chunks fall back (checked before the
    // chunk count, which would overflow near the c_int limit)
    if (!mlx.streamIsGpu(s) or n > 1024 * CH) return null;
    switch (mlx.mlx_array_dtype(y)) {
        .bfloat16, .float16, .float32 => {},
        else => return null,
    }
    // A bf16 row (the served 2-bit product's scores) takes the threshold select; its 16-bit key is the bf16 bits
    if (mlx.mlx_array_dtype(y) == .bfloat16 and !thresholdSelectOff()) return try thresholdTopk(s, y, k, sh, n);
    const groups = @divFloor(n - 1, CH) + 1;
    const pk = try kernel(&part_kernel, "mlxserve_shortlist_select_topk_part", &.{"y"}, &.{ "cv", "ci" }, PART_SOURCE);
    const mk = try kernel(&merge_kernel, "mlxserve_shortlist_select_topk_merge", &.{ "cv", "ci" }, &.{"out"}, MERGE_SOURCE);

    const c1 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c1);
    const cshape = [_]c_int{groups * k};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &cshape, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &cshape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c1, groups * TG1, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c1, TG1, 1, 1));
    inline for (.{ .{ "N", n }, .{ "K", k }, .{ "CH", CH }, .{ "P", P }, .{ "TG", TG1 } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c1, kv[0], kv[1]));
    }
    var cand: [2]mlx.mlx_array = undefined;
    try apply(pk, c1, &.{y}, s, &cand);
    defer {
        for (cand) |c| _ = mlx.mlx_array_free(c);
    }

    const tg2: c_int = @max(32, @divFloor(groups + 31, 32) * 32);
    const c2 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c2);
    var oshape: [8]c_int = undefined;
    @memcpy(oshape[0..sh.len], sh);
    oshape[sh.len - 1] = k;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c2, &oshape, sh.len, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c2, tg2, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c2, tg2, 1, 1));
    inline for (.{ .{ "G", groups }, .{ "K", k }, .{ "TG", tg2 } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c2, kv[0], kv[1]));
    }
    var out: [1]mlx.mlx_array = undefined;
    try apply(mk, c2, &.{ cand[0], cand[1] }, s, &out);
    return out[0];
}

/// `topk` for a bf16 row by the threshold select. The caller checked the shape, k (1 to MAX_K, at most n), the
/// GPU stream and n up to 1024 x CH (262,144), so there are at most 64 chunks and 4,096 candidates (4 a merge thread).
fn thresholdTopk(s: mlx.mlx_stream, y: mlx.mlx_array, k: c_int, sh: []const c_int, n: c_int) !mlx.mlx_array {
    const groups = @divFloor(n - 1, THRESHOLD_SELECT_C) + 1;
    const pk = try kernelWith(&threshold_select_part_kernel, "mlxserve_threshold_select_part", &.{"y"}, &.{ "ck", "ci" }, THRESHOLD_SELECT_PART_SOURCE, THRESHOLD_SELECT_HEADER);
    const mk = try kernelWith(&threshold_select_merge_kernel, "mlxserve_threshold_select_merge", &.{ "ck", "ci" }, &.{"out"}, THRESHOLD_SELECT_MERGE_SOURCE, THRESHOLD_SELECT_HEADER);

    const c1 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c1);
    const cshape = [_]c_int{groups * k};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &cshape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &cshape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c1, groups * THRESHOLD_SELECT_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c1, THRESHOLD_SELECT_TG, 1, 1));
    inline for (.{ .{ "N", n }, .{ "K", k }, .{ "C", THRESHOLD_SELECT_C }, .{ "TG", THRESHOLD_SELECT_TG }, .{ "E", @divExact(THRESHOLD_SELECT_C, THRESHOLD_SELECT_TG) }, .{ "HC", THRESHOLD_SELECT_HC } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c1, kv[0], kv[1]));
    }
    var cand: [2]mlx.mlx_array = undefined;
    try apply(pk, c1, &.{y}, s, &cand);
    defer {
        for (cand) |c| _ = mlx.mlx_array_free(c);
    }

    const m = groups * k;
    const c2 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c2);
    var oshape: [8]c_int = undefined;
    @memcpy(oshape[0..sh.len], sh);
    oshape[sh.len - 1] = k;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c2, &oshape, sh.len, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c2, THRESHOLD_SELECT_TGM, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c2, THRESHOLD_SELECT_TGM, 1, 1));
    inline for (.{ .{ "M", m }, .{ "K", k }, .{ "TG", THRESHOLD_SELECT_TGM }, .{ "E", @divFloor(m + THRESHOLD_SELECT_TGM - 1, THRESHOLD_SELECT_TGM) }, .{ "HC", THRESHOLD_SELECT_HCM } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c2, kv[0], kv[1]));
    }
    var out: [1]mlx.mlx_array = undefined;
    try apply(mk, c2, &.{ cand[0], cand[1] }, s, &out);
    return out[0];
}

/// headTailArgmaxMap's token for the kept row of `nk` positions that holds the softcapped rescored scores `v` (last
/// axis k) at the kept positions `sel` (uint32, k) and `fv` (softcap of -1e30 in the row's dtype, [1]) everywhere else;
/// `idx`, `cmask`, `dv`, `di` and `masked` as for headTailArgmaxMap. The output has `v`'s shape less its last axis.
pub fn headTail64(s: mlx.mlx_stream, v: mlx.mlx_array, sel: mlx.mlx_array, idx: mlx.mlx_array, cmask: mlx.mlx_array, fv: mlx.mlx_array, dv: mlx.mlx_array, di: mlx.mlx_array, masked: bool) !mlx.mlx_array {
    const vs = mlx.getShape(v);
    if (vs.len < 1 or vs.len > 4) return error.HeadTail64Shape;
    const k = vs[vs.len - 1];
    if (k < 1 or k > MAX_K or mlx.mlx_array_size(v) != @as(usize, @intCast(k)) or mlx.mlx_array_size(sel) != @as(usize, @intCast(k))) return error.HeadTail64Shape;
    const nk = mlx.getShape(idx)[0];
    if (nk < k) return error.HeadTail64Shape;
    const hk = try kernel(&ht64_kernel, "mlxserve_shortlist_select_head_tail64", &.{ "v", "sel", "idx", "cmask", "fv", "dv", "di" }, &.{"out"}, HT64_SOURCE);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const one = [_]c_int{1};
    if (vs.len > 1) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, vs.ptr, vs.len - 1, .uint32));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &one, 0, .uint32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, HT64_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, HT64_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NK", nk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TG", HT64_TG));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "MASKED", masked));
    var out: [1]mlx.mlx_array = undefined;
    try apply(hk, c, &.{ v, sel, idx, cmask, fv, dv, di }, s, &out);
    return out[0];
}

// ── Tests: the pick against MLX's own argpartition, ties, NaN and signed zeros included ──

const testing = std.testing;

fn testRow(data: []const f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = [_]c_int{ 1, 1, @intCast(data.len) };
    const f = mlx.mlx_array_new_data(data.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

fn argpartFirst(s: mlx.mlx_stream, y: mlx.mlx_array, k: c_int) !mlx.mlx_array {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, y, s));
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, neg, k - 1, -1, s));
    const sh = mlx.getShape(y);
    var stop: [8]c_int = undefined;
    @memcpy(stop[0..sh.len], sh);
    stop[sh.len - 1] = k;
    const zeros: [8]c_int = @splat(0);
    const ones: [8]c_int = @splat(1);
    var idx = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&idx, part, &zeros, sh.len, &stop, sh.len, &ones, sh.len, s));
    return idx;
}

fn expectSamePick(s: mlx.mlx_stream, data: []const f32, dt: mlx.mlx_dtype, k: c_int) !void {
    const y = try testRow(data, dt, s);
    defer _ = mlx.mlx_array_free(y);
    const want = try argpartFirst(s, y, k);
    defer _ = mlx.mlx_array_free(want);
    const got = (try topk(s, y, k)) orelse return error.TestUnexpectedResult;
    defer _ = mlx.mlx_array_free(got);
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, want, got, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    if (!same) {
        try mlx.check(mlx.mlx_array_eval(want));
        try mlx.check(mlx.mlx_array_eval(got));
        const w = mlx.mlx_array_data_uint32(want).?;
        const g = mlx.mlx_array_data_uint32(got).?;
        std.debug.print("shortlist-select pick differs (n {d}, k {d}): want {any}\n got {any}\n", .{ data.len, k, w[0..@intCast(k)], g[0..@intCast(k)] });
    }
    try testing.expect(same);
}

test "shortlist-select topk: the same k indices in the same order as argpartition(-y), with ties, NaN and signed zeros" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x167);
    const rnd = prng.random();
    // The served row's length (Gemma 4's Latin-script head), chunk-multiple, short and odd lengths.
    const lens = [_]usize{ 162489, 4096, 2048, 2049, 1000, 512, 257, 256, 64, 65, 130 };
    for (lens) |n| {
        const data = try a.alloc(f32, n);
        defer a.free(data);
        // 1: continuous scores, few ties after the bf16 cast
        for (data) |*v| v.* = 8.0 * rnd.floatNorm(f32);
        for ([_]c_int{ 64, 16, 1 }) |k| if (@as(usize, @intCast(k)) <= n) try expectSamePick(s, data, .bfloat16, k);
        // 2: a coarse grid, so the k-th value is tied with many rows across chunks
        for (data) |*v| v.* = @floatFromInt(rnd.intRangeAtMost(i32, -3, 3));
        for ([_]c_int{ 64, 16, 1 }) |k| if (@as(usize, @intCast(k)) <= n) try expectSamePick(s, data, .bfloat16, k);
        // 3: every value equal: the pick is the first k indices
        @memset(data, 1.5);
        if (n >= 64) try expectSamePick(s, data, .bfloat16, 64);
        // 4: signed zeros and NaN: -0 ties with +0, NaN ranks after every number
        for (data, 0..) |*v, i| v.* = switch (i % 7) {
            0 => -0.0,
            1 => 0.0,
            2 => std.math.nan(f32),
            else => -1.0 - @as(f32, @floatFromInt(i % 5)),
        };
        if (n >= 64) try expectSamePick(s, data, .bfloat16, 64);
        // 5: fewer numbers than k: the NaN rows fill the pick in index order
        for (data, 0..) |*v, i| v.* = if (i % 97 == 5) @as(f32, @floatFromInt(i % 11)) else std.math.nan(f32);
        if (n >= 64) try expectSamePick(s, data, .bfloat16, 64);
    }
    // float32 rows (the row-exact 2-bit product) take the same path.
    const d32 = try a.alloc(f32, 162489);
    defer a.free(d32);
    for (d32) |*v| v.* = rnd.floatNorm(f32);
    try expectSamePick(s, d32, .float32, 64);
}

test "threshold-select topk: the threshold select returns the sorted chunks' pick on bf16 rows" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x172);
    const rnd = prng.random();
    const saved = threshold_select_off;
    defer threshold_select_off = saved;
    for ([_]usize{ 162489, 4096, 4097, 8192, 300 }) |n| {
        const data = try a.alloc(f32, n);
        defer a.free(data);
        for (0..3) |shape| {
            for (data, 0..) |*v, i| v.* = switch (shape) {
                0 => 8.0 * rnd.floatNorm(f32),
                1 => @floatFromInt(rnd.intRangeAtMost(i32, -2, 2)),
                // ties at the k-th value spread over every chunk, with signed zeros among them
                else => if (i % 509 == 3) 9.0 else if (i % 4099 == 1) (if (i % 2 == 0) -0.0 else 0.0) else -5.0,
            };
            const y = try testRow(data, .bfloat16, s);
            defer _ = mlx.mlx_array_free(y);
            for ([_]c_int{ 64, 7, 1 }) |k| {
                threshold_select_off = true;
                const want = (try topk(s, y, k)) orelse return error.TestUnexpectedResult;
                defer _ = mlx.mlx_array_free(want);
                // the threshold kernels called directly (Codex on 50b88510): the test must not depend on topk's dispatch
                const ysh = mlx.getShape(y);
                const got = try thresholdTopk(s, y, k, ysh, ysh[ysh.len - 1]);
                defer _ = mlx.mlx_array_free(got);
                var eq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(eq);
                try mlx.check(mlx.mlx_array_equal(&eq, want, got, false, s));
                var same = false;
                try mlx.check(mlx.mlx_array_item_bool(&same, eq));
                if (!same) std.debug.print("threshold-select pick differs (n {d}, shape {d}, k {d})\n", .{ n, shape, k });
                try testing.expect(same);
            }
        }
    }
}

test "shortlist-select topk refuses what it does not serve" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1, 2, 3, 4 };
    const y = try testRow(&data, .bfloat16, s);
    defer _ = mlx.mlx_array_free(y);
    try testing.expect((try topk(s, y, 5)) == null);
    try testing.expect((try topk(s, y, 0)) == null);
    try testing.expect((try topk(s, y, MAX_K + 1)) == null);
}
