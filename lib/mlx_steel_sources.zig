// MLX's steel NAX tile headers, embedded for the kernels that reuse its tile
// matmul in their own `metal_kernel` source (src/lane_qmm.zig). Order matters:
// each header only uses the ones before it; their `#include "mlx/...` lines are
// stripped by the consumer.

pub const defines: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/defines.h");
pub const type_traits: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils/type_traits.h");
pub const integral_constant: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils/integral_constant.h");
pub const nax: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/nax.h");

// The non-NAX steel GEMM and the quantized kernels' headers, for src/moe_prefill_fused.zig, in include order after
// defines, type_traits and integral_constant (utils.h comes from MLX's own custom-kernel preamble).
pub const steel_utils: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/utils.h");
pub const transforms: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/transforms.h");
pub const mma: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/mma.h");
pub const loader: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/loader.h");
pub const params: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/params.h");
pub const gemm: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/steel/gemm/gemm.h");
pub const quantized_utils: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/quantized_utils.h");
pub const quantized: []const u8 = @embedFile("mlx-src/mlx/backend/metal/kernels/quantized.h");
