//! The MoE prefill expert path without NAX. Where MLX would run its sorted row-block kernel (affine_gather_qmm_rhs at
//! 16-row tiles, quantized.cpp gather_qmm_rhs) for gate, up and down, with the table GeGLU between them, this runs
//!   1. the expert offsets (MLX's gather_mm_offsets: one lower bound per expert over the sorted expert ids),
//!   2. gate and up of the same routed rows in one pass at 32-row tiles, token rows read through the row map (no
//!      gathered copy of x), GeGLU in the epilogue (moe_prefill::fused_gate_up in kernels/moe_prefill_fused.metal),
//!   3. down with MLX's own kernel body at 32-row tiles (moe_prefill::gather_qmm_rhs).
//! Every output keeps MLX's K order and float accumulation, so the act and down bits equal the stock path's
//! (MLX_SERVE_MOE_PREFILL_FUSED_CHECK=1 compares both in the engine). The kernels are built with MLX's custom-kernel
//! compiler (MathMode::Safe), not the offline metallib; the engine check covers that difference.
//! MLX_SERVE_MOE_PREFILL_FUSED_OFF=1 restores the stock path.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const steel = @import("mlx_steel_sources");

pub const BM: c_int = 32;
const BN: c_int = 32;
const BK: c_int = 32;
const WM: c_int = 1;
const WN: c_int = 2;
const GROUP: c_int = 64;
/// Engage only when an expert averages this many rows or more (a 32-row tile then pads little); below it the stock
/// 16-row path runs. MLX's own sorted-kernel rule (rows >= 16 and rows / experts >= 4) is implied.
pub const MIN_ROWS_PER_EXPERT: c_int = 32;

var off_cache: ?bool = null;
var check_cache: ?bool = null;
var engaged_logged = false;
var declined_logged = false;
/// Set while `warm` runs, so its tiny dispatch does not print the engaged line.
var warming = false;
pub var check_calls: u32 = 0;

pub fn off() bool {
    if (off_cache) |v| return v;
    off_cache = std.c.getenv("MLX_SERVE_MOE_PREFILL_FUSED_OFF") != null;
    return off_cache.?;
}

/// A kernel that failed to build at load stays off: the stock path gives the same bits.
pub fn disable() void {
    off_cache = true;
}

pub fn checkOn() bool {
    if (check_cache) |v| return v;
    check_cache = std.c.getenv("MLX_SERVE_MOE_PREFILL_FUSED_CHECK") != null;
    return check_cache.?;
}

const OFFSETS_SOURCE: [:0]const u8 =
    \\  // MLX's gather_mm_offsets (kernels/gather_mm_offsets.metal): the first sorted row of each expert.
    \\  const uint g = thread_position_in_grid.x;
    \\  if (int(g) >= dims[1]) return;
    \\  int lo = 0;
    \\  int hi = dims[0];
    \\  while (lo < hi) {
    \\    const int mid = (lo + hi) / 2;
    \\    if (idx[mid] < g) {
    \\      lo = mid + 1;
    \\    } else {
    \\      hi = mid;
    \\    }
    \\  }
    \\  offs[g] = lo;
;

const GATE_UP_SOURCE: [:0]const u8 =
    \\  using T = bfloat16_t;
    \\  constexpr int BKP = 32 + 16 / sizeof(T);
    \\  threadgroup T Xs[32 * BKP];
    \\  threadgroup T Wgs[32 * BKP];
    \\  threadgroup T Wus[32 * BKP];
    \\  moe_prefill::fused_gate_up<T, 64, 4, 32, 32, 32, 1, 2>(xt, wg, sg, bg, offs, y, dims[0], dims[1], dims[2], dims[3],
    \\      rmap, wu, su, bu, table, Xs, Wgs, Wus, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
    \\      thread_index_in_simdgroup);
;

const DOWN_SOURCE: [:0]const u8 =
    \\  using T = bfloat16_t;
    \\  constexpr int BKP = 32 + 16 / sizeof(T);
    \\  threadgroup T Xs[32 * BKP];
    \\  threadgroup T Ws[32 * BKP];
    \\  moe_prefill::gather_qmm_rhs<T, 64, 4, 32, 32, 32, 1, 2>(act, wd, sd, bd, offs, y, dims[0], dims[1], dims[2], dims[3],
    \\      Xs, Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
;

var header_cache: ?[:0]const u8 = null;

/// macOS 14's runtime Metal compiler has no `metal::remove_addrspace_t`, which MLX's steel/utils/integral_constant.h
/// names, so the whole JIT source failed to build there (M3 Pro, macOS 14.4, 2026-11-12). The source gets this trait
/// instead, defined as the macOS 26 SDK's metal_type_traits defines remove_addrspace (plain, device, constant and
/// threadgroup), so the compiled kernel is the same on every macOS.
const addrspace_shim =
    \\template <typename T> struct lesto_remove_addrspace { typedef T type; };
    \\template <typename T> struct lesto_remove_addrspace<device T> { typedef T type; };
    \\template <typename T> struct lesto_remove_addrspace<constant T> { typedef T type; };
    \\template <typename T> struct lesto_remove_addrspace<threadgroup T> { typedef T type; };
    \\template <typename T> using lesto_remove_addrspace_t = typename lesto_remove_addrspace<T>::type;
    \\
;
const addrspace_from = "metal::remove_addrspace_t";
const addrspace_to = "lesto_remove_addrspace_t";

/// Appends `line` with every `metal::remove_addrspace_t` replaced by the shim's trait.
fn appendAddrspaceFixed(a: std.mem.Allocator, out: *std.ArrayList(u8), line: []const u8) !void {
    var rest = line;
    while (std.mem.indexOf(u8, rest, addrspace_from)) |i| {
        try out.appendSlice(a, rest[0..i]);
        try out.appendSlice(a, addrspace_to);
        rest = rest[i + addrspace_from.len ..];
    }
    try out.appendSlice(a, rest);
}

test "the JIT header names no metal::remove_addrspace_t" {
    const h = try header();
    try std.testing.expect(std.mem.indexOf(u8, h, addrspace_from) == null);
    try std.testing.expect(std.mem.indexOf(u8, h, "using res_t = " ++ addrspace_to) != null);
}

/// MLX's steel and quantized headers in include order, their own `#include "mlx/` and `#pragma once` lines dropped
/// (as lane_qmm.zig does), then the fused functions. MLX prepends its utils preamble itself.
fn header() ![:0]const u8 {
    if (header_cache) |h| return h;
    const a = std.heap.c_allocator;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n");
    try out.appendSlice(a, addrspace_shim);
    for ([_][]const u8{ steel.defines, steel.type_traits, steel.integral_constant, steel.steel_utils, steel.transforms, steel.mma, steel.loader, steel.params, steel.gemm, steel.quantized_utils, steel.quantized, @embedFile("kernels/moe_prefill_fused.metal") }) |src| {
        var lines = std.mem.splitScalar(u8, src, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "#include \"mlx/") or std.mem.startsWith(u8, line, "#pragma once")) continue;
            try appendAddrspaceFixed(a, &out, line);
            try out.append(a, '\n');
        }
    }
    const h = try out.toOwnedSliceSentinel(a, 0);
    header_cache = h;
    return h;
}

const Kind = enum(u2) { offsets, gate_up, down };
var kernels: [3]?mlx.mlx_fast_metal_kernel = .{ null, null, null };

fn kernel(kind: Kind) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[@intFromEnum(kind)];
    if (slot.*) |k| return k;
    const names_off = [_][*:0]const u8{ "idx", "dims" };
    const names_gu = [_][*:0]const u8{ "xt", "rmap", "wg", "sg", "bg", "wu", "su", "bu", "offs", "table", "dims" };
    const names_dn = [_][*:0]const u8{ "act", "wd", "sd", "bd", "offs", "dims" };
    const in_names: []const [*:0]const u8 = switch (kind) {
        .offsets => &names_off,
        .gate_up => &names_gu,
        .down => &names_dn,
    };
    const out_off = [_][*:0]const u8{"offs"};
    const out_y = [_][*:0]const u8{"y"};
    const out_names: []const [*:0]const u8 = if (kind == .offsets) &out_off else &out_y;
    const in_vec = mlx.mlx_vector_string_new_data(in_names.ptr, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(out_names.ptr, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = switch (kind) {
        .offsets => mlx.mlx_fast_metal_kernel_new("msv_moe_prefill_offsets", in_vec, out_vec, OFFSETS_SOURCE, "", true, false),
        .gate_up => mlx.mlx_fast_metal_kernel_new("msv_moe_prefill_gate_up", in_vec, out_vec, GATE_UP_SOURCE, (try header()).ptr, true, false),
        .down => mlx.mlx_fast_metal_kernel_new("msv_moe_prefill_down", in_vec, out_vec, DOWN_SOURCE, (try header()).ptr, true, false),
    };
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn apply(kind: Kind, inputs: []const mlx.mlx_array, out_shape: []const c_int, out_dtype: mlx.mlx_dtype, grid: [3]c_int, tg: [3]c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, out_shape.ptr, out_shape.len, out_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg[0], tg[1], tg[2]));
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(kind), in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outs) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

/// A [4] int32 array; under 8 elements MLX binds it in `constant` (custom_kernel's max_constant_array_size), which
/// the kernel sources expect.
fn dims4(a: c_int, b: c_int, c: c_int, d: c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(&[_]c_int{ a, b, c, d }, &[_]c_int{4}, 1, .int32);
}

/// One quantized bank [E, N, K / 8] uint32 (4 bits), scales and biases [E, N, K / 64] bf16.
fn bankOk(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, e: c_int, n: c_int, k: c_int) bool {
    if (w.ctx == null or sc.ctx == null or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    if (mlx.mlx_array_ndim(w) != 3 or mlx.mlx_array_ndim(sc) != 3 or mlx.mlx_array_ndim(bi) != 3) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    if (ws[0] != e or ws[1] != n or @as(i64, ws[2]) * 8 != k) return false;
    return ss[0] == e and ss[1] == n and ss[2] == @divExact(k, GROUP) and bs[0] == e and bs[1] == n and bs[2] == ss[2];
}

/// The shape rule for the whole path: rows M = idx's length, E experts (8 to 2048: an offsets array under 8
/// elements would be bound in `constant`), N and K multiples of the tiles and K of the group, at least
/// MIN_ROWS_PER_EXPERT rows per expert on average.
fn shapeOk(m: c_int, e: c_int, n: c_int, k: c_int) bool {
    if (e < 8 or e > 2048 or m < 16) return false;
    if (@divTrunc(m, e) < MIN_ROWS_PER_EXPERT) return false;
    return n > 0 and k > 0 and @rem(n, BN) == 0 and @rem(k, BK) == 0 and @rem(k, GROUP) == 0;
}

/// Expert offsets [E] int32 over sorted expert ids [M] uint32 (MLX's gather_mm_offsets).
pub fn offsets(idx: mlx.mlx_array, e: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const m = mlx.getShape(idx)[0];
    const d = dims4(m, e, 0, 0);
    defer _ = mlx.mlx_array_free(d);
    const grid_x = @divTrunc(e + 31, 32) * 32;
    return apply(.offsets, &.{ idx, d }, &.{e}, .int32, .{ grid_x, 1, 1 }, .{ 32, 1, 1 }, s);
}

/// gelu_tanh(x W_gate^T) * (x W_up^T) for the sorted routed rows, [M, N] bf16 in sorted order, or null when the
/// shapes are outside the rule. `xf`: [tokens, K] bf16; `rmap`: [M] uint32 token row per sorted row; `idx`: [M]
/// uint32 sorted expert ids; `offs`: from `offsets`; `table`: the bf16 gelu_tanh table of all 65536 patterns.
pub fn gateUp(xf: mlx.mlx_array, rmap: mlx.mlx_array, idx: mlx.mlx_array, wg: mlx.mlx_array, sg: mlx.mlx_array, bg: mlx.mlx_array, wu: mlx.mlx_array, su: mlx.mlx_array, bu: mlx.mlx_array, offs: mlx.mlx_array, table: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or off()) return null;
    if (xf.ctx == null or mlx.mlx_array_dtype(xf) != .bfloat16 or mlx.mlx_array_ndim(xf) != 2) return null;
    if (idx.ctx == null or mlx.mlx_array_dtype(idx) != .uint32 or mlx.mlx_array_ndim(idx) != 1) return null;
    if (rmap.ctx == null or mlx.mlx_array_dtype(rmap) != .uint32 or mlx.mlx_array_ndim(rmap) != 1) return null;
    if (table.ctx == null or mlx.mlx_array_dtype(table) != .bfloat16 or mlx.mlx_array_size(table) != 65536) return null;
    const m = mlx.getShape(idx)[0];
    if (mlx.getShape(rmap)[0] != m) return null;
    const k = mlx.getShape(xf)[1];
    if (wg.ctx == null or mlx.mlx_array_ndim(wg) != 3) return null;
    const e = mlx.getShape(wg)[0];
    const n = mlx.getShape(wg)[1];
    if (!shapeOk(m, e, n, k) or !bankOk(wg, sg, bg, e, n, k) or !bankOk(wu, su, bu, e, n, k)) return null;
    if (offs.ctx == null or mlx.mlx_array_dtype(offs) != .int32 or mlx.mlx_array_size(offs) != @as(usize, @intCast(e))) return null;
    const d = dims4(m, n, k, e);
    defer _ = mlx.mlx_array_free(d);
    const gy = @min(m, @divTrunc(m + BM - 1, BM) + e - 1);
    const out = try apply(.gate_up, &.{ xf, rmap, wg, sg, bg, wu, su, bu, offs, table, d }, &.{ m, n }, .bfloat16, .{ @divExact(n, BN) * 32, gy * WN, WM }, .{ 32, WN, WM }, s);
    if (!engaged_logged and !warming) {
        engaged_logged = true;
        log.info("[moe-prefill-fused] MoE prefill gate+up+GeGLU fused through the row map and down, both at {d}-row tiles: M={d} E={d} inter={d} hidden={d} (MLX_SERVE_MOE_PREFILL_FUSED_OFF=1 restores the stock gathers)\n", .{ BM, m, e, n, k });
    }
    return out;
}

/// act W_down^T for the sorted rows, [M, 1, N] bf16 (the stock gather's shape), or null outside the rule.
/// `act`: [M, K] bf16 in sorted order.
pub fn down(act: mlx.mlx_array, wd: mlx.mlx_array, sd: mlx.mlx_array, bd: mlx.mlx_array, offs: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or off()) return null;
    if (act.ctx == null or mlx.mlx_array_dtype(act) != .bfloat16 or mlx.mlx_array_ndim(act) != 2) return null;
    if (wd.ctx == null or mlx.mlx_array_ndim(wd) != 3) return null;
    const m = mlx.getShape(act)[0];
    const k = mlx.getShape(act)[1];
    const e = mlx.getShape(wd)[0];
    const n = mlx.getShape(wd)[1];
    if (!shapeOk(m, e, n, k) or !bankOk(wd, sd, bd, e, n, k)) return null;
    if (offs.ctx == null or mlx.mlx_array_dtype(offs) != .int32 or mlx.mlx_array_size(offs) != @as(usize, @intCast(e))) return null;
    const d = dims4(m, n, k, e);
    defer _ = mlx.mlx_array_free(d);
    const gy = @min(m, @divTrunc(m + BM - 1, BM) + e - 1);
    return try apply(.down, &.{ act, wd, sd, bd, offs, d }, &.{ m, 1, n }, .bfloat16, .{ @divExact(n, BN) * 32, gy * WN, WM }, .{ 32, WN, WM }, s);
}

/// Logs once why the path declined although the caller's model-level rule held (shape or dtype outside the rule).
pub fn noteDeclined(m: c_int, e: c_int) void {
    if (declined_logged) return;
    declined_logged = true;
    log.info("[moe-prefill-fused] declined at M={d} E={d}: shapes or dtypes outside the rule, stock path\n", .{ m, e });
}

/// `warm` at load, or the path switched off when it fails. A failed build (macOS 14's Metal compiler rejected the
/// header once) latches MLX's error; left latched, the first request's prefill `checkError` consumed it and failed
/// with MlxFailure although the stock path was on. Only a latch the warm-up raised is dropped: one that was already
/// pending belongs to someone else and stays.
pub fn warmOrDisable(s: mlx.mlx_stream) void {
    const had_error = mlx.errorPending();
    warm(s) catch |err| {
        log.warn("[moe-prefill-fused] warm-up build failed ({s}); the stock path stays on\n", .{@errorName(err)});
        disable();
        mlx.dropLatchedErrorUnless(had_error);
    };
}

test "a failed warm-up switches the path off and drops only its own latch" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const saved = off_cache;
    defer off_cache = saved;
    defer mlx.armLatchingFaultForTest(0);

    // An error latched before the warm-up stays for its owner.
    off_cache = false;
    mlx.latchErrorForTest("earlier op before the warm-up");
    mlx.armLatchingFaultForTest(1);
    warmOrDisable(s);
    try std.testing.expect(mlx.latchingFaultFiredForTest());
    try std.testing.expect(off());
    var buf: [512]u8 = undefined;
    const msg = mlx.takeError(&buf) orelse return error.EarlierLatchLost;
    try std.testing.expect(std.mem.indexOf(u8, msg, "earlier op") != null);

    // The warm-up's own latch is dropped, so the next prefill checkpoint passes.
    off_cache = false;
    mlx.armLatchingFaultForTest(1);
    warmOrDisable(s);
    try std.testing.expect(mlx.latchingFaultFiredForTest());
    try std.testing.expect(off());
    try std.testing.expect(!mlx.errorPending());
    try mlx.checkError();
}

/// Builds the three kernels' libraries at load, so no request pays the compile: one tiny dispatch of each with the
/// production signature (the same dtypes, and every array of 8 or more elements, so `device` binding, except dims).
/// MLX keys a custom kernel's library on its full source, signature included, so a different binding would not warm.
pub fn warm(s: mlx.mlx_stream) !void {
    if (!mlx.streamIsGpu(s) or off()) return;
    const m: c_int = 256;
    const e: c_int = 8;
    const n: c_int = 32;
    const k: c_int = 64;
    const shapes = [_]struct { sh: []const c_int, dt: mlx.mlx_dtype }{
        .{ .sh = &.{ m, k }, .dt = .bfloat16 }, // 0: x [tokens, K], also down's input [M, K]
        .{ .sh = &.{m}, .dt = .uint32 }, // 1: row map and sorted ids
        .{ .sh = &.{ e, n, @divExact(k, 8) }, .dt = .uint32 }, // 2: bank [E, N, K / 8]
        .{ .sh = &.{ e, n, @divExact(k, GROUP) }, .dt = .bfloat16 }, // 3: scales and biases
        .{ .sh = &.{65536}, .dt = .bfloat16 }, // 4: table
    };
    var arrs: [8]mlx.mlx_array = undefined;
    var made: usize = 0;
    defer for (arrs[0..made]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    for (shapes) |sp| {
        arrs[made] = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_zeros(&arrs[made - 1], sp.sh.ptr, sp.sh.len, sp.dt, s));
    }
    warming = true;
    defer warming = false;
    const offs = try offsets(arrs[1], e, s);
    arrs[made] = offs;
    made += 1;
    const act = (try gateUp(arrs[0], arrs[1], arrs[1], arrs[2], arrs[3], arrs[3], arrs[2], arrs[3], arrs[3], offs, arrs[4], s)) orelse return error.MoePrefillFusedWarmDeclined;
    arrs[made] = act;
    made += 1;
    const y = (try down(arrs[0], arrs[2], arrs[3], arrs[3], offs, s)) orelse return error.MoePrefillFusedWarmDeclined;
    arrs[made] = y;
    made += 1;
    const ev = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ act, y }, 2);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
}
