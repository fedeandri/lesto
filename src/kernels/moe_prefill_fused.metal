// The MoE prefill expert kernels without NAX, as `metal_kernel` header functions for src/moe_prefill_fused.zig. This file
// is the tail of the header; moe_prefill_fused.zig puts MLX's own steel and quantized headers in front of it
// (lib/mlx_steel_sources.zig, their `#include "mlx/` and `#pragma once` lines dropped), and MLX's custom-kernel builder
// prepends its utils preamble (backend/metal/custom_kernel.cpp:53, `metal::utils() + source_`).
//
// 1. moe_prefill::fused_gate_up: gate and up of the same routed rows in one pass, token rows read through the sorted row
//    map (no gathered copy of x), MLX's QuantizedBlockLoader and BlockMMA K loop, the float -> T cast of
//    BlockMMA::store_result, then the engine's table GeGLU (tableGateMul .gelu_tanh: act = table[bits(gate)], y = act * up,
//    in T). Rows past a tile's valid count read a clamped valid row; MMA rows are independent and those rows are never
//    stored (store_result_safe), so no stored value depends on them.
// 2. moe_prefill::gather_qmm_rhs: MLX's affine_gather_qmm_rhs (kernels/quantized.h:2436-2574) with transpose = true and
//    the align_N and align_K function constants fixed to true (the host declines any N % BN or K % BK), so only the two
//    branches MLX takes when both are true remain, line for line.
// Both are instantiated by moe_prefill_fused.zig at BM 32, BN 32, BK 32, WM 1, WN 2, bf16, group 64, 4 bits.
namespace moe_prefill {

// BlockLoader's thread layout (steel/gemm/loader.h), but each of a thread's rows has its own source pointer taken
// from the row map, so the token rows need not be contiguous. Same ReadVector copy as BlockLoader::load_unsafe.
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size,
          short n_reads = (BCOLS * BROWS) / (tgp_size), short TCOLS = BCOLS / n_reads,
          short TROWS = tgp_size / TCOLS>
struct RowLoader {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;
  static_assert(BROWS % TROWS == 0, "rows must tile evenly");
  struct alignas(sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src[n_rows];

  METAL_FUNC RowLoader(const device T* x, const device uint32_t* rows, int row0, short nvalid, int ld,
                       threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id) thread
      : bi((simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(vec_size * ((simd_group_id * 32 + simd_lane_id) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      const short r = min(short(bi + i * TROWS), short(nvalid - 1));
      src[i] = x + size_t(rows[row0 + r]) * size_t(ld) + bj;
    }
  }
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      *((threadgroup ReadVector*)(&dst[i * TROWS * dst_ld])) = *((const device ReadVector*)(src[i]));
    }
  }
  METAL_FUNC void next() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) src[i] += BCOLS;
  }
};

template <typename T, int group_size, int bits, int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void fused_gate_up(
    const device T* x,          // x_flat [tokens, K]
    const device uint32_t* wg,  // gate bank [E, N, K / 8]
    const device T* sg,
    const device T* bg,
    const device int32_t* offsets,
    device T* y,                // act [M, N], sorted order
    const int M,
    const int N,
    const int K,
    const int num_groups,
    const device uint32_t* lhs, // sorted row -> token row
    const device uint32_t* wu,  // up bank
    const device T* su,
    const device T* bu,
    const device T* table,      // gelu_tanh of every 16-bit pattern
    threadgroup T* Xs,
    threadgroup T* Wgs,
    threadgroup T* Wus,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = RowLoader<T, BM, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  y += size_t(y_row) * N + size_t(y_col);
  const device uint8_t* wgl = (const device uint8_t*)wg + size_t(y_col) * K_w + group * stride_w;
  const device uint8_t* wul = (const device uint8_t*)wu + size_t(y_col) * K_w + group * stride_w;
  const size_t s_off = size_t(y_col) * K_g + group * stride_s;

  thread mma_t mma_g(simd_group_id, simd_lane_id);
  thread mma_t mma_u(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(x, lhs, y_row, tgp_bm, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_g(wgl, sg + s_off, bg + s_off, K, Wgs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_u(wul, su + s_off, bu + s_off, K, Wus, simd_group_id, simd_lane_id);

  for (int k = 0; k < K_it; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_g.load_unsafe();
    loader_u.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_g.mma(Xs, Wgs);
    mma_u.mma(Xs, Wus);
    loader_x.next();
    loader_g.next();
    loader_u.next();
  }

  // Epilogue: the stock path stores gate and up as T (static_cast, mma.h store_result), then tableGateMul computes
  // table[bits(gate)] * up in T. The product is a T value, so the float round trip through Ctile is exact.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_g.Ctile)::kElemsPerTile; i++) {
    const T g = static_cast<T>(mma_g.Ctile.elems()[i]);
    const T u = static_cast<T>(mma_u.Ctile.elems()[i]);
    const T act = table[as_type<ushort>(g)];
    const T r = act * u;
    mma_g.Ctile.elems()[i] = static_cast<float>(r);
  }
  if (tgp_bm == BM) {
    mma_g.store_result(y, N);
  } else {
    mma_g.store_result_safe(y, N, short2(BN, tgp_bm));
  }
}

// MLX's affine_gather_qmm_rhs<T, group_size, bits, BM, BN, BK, WM, WN, transpose = true> with align_N = align_K =
// true. The host guarantees N % BN == 0 and K % BK == 0.
template <typename T, int group_size, int bits, int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void gather_qmm_rhs(
    const device T* x,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device int32_t* offsets,
    device T* y,
    const int M,
    const int N,
    const int K,
    const int num_groups,
    threadgroup T* Xs,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, true, WM * WN * SIMD_SIZE, group_size, bits>;

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = N * K_w;
  const size_t stride_s = N * K_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds (align_N)
  const short tgp_bn = BN;

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += y_col_long * K_w;
  scales += y_col_long * K_g;
  biases += y_col_long * K_g;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(
      wl + group * stride_w,
      scales + group * stride_s,
      biases + group * stride_s,
      K,
      Ws,
      simd_group_id,
      simd_lane_id);

  // align_N and align_K: MLX's first two branches without the K finalize.
  if (tgp_bm == BM) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    mma_op.store_result(y, N);
  } else {
    gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    mma_op.store_result_safe(y, N, short2(BN, tgp_bm));
  }
}

}  // namespace moe_prefill
