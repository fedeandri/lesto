//! Quantized linear weights and chunked row requantization (used by the DFlash draft head).

const std = @import("std");
const mlx = @import("mlx.zig");

/// One linear: quantized (w packed u32, s/b bf16) when `s.ctx != null`,
/// otherwise a pre-transposed bf16 weight `[in, out]` for plain matmul.
pub const QuantLinear = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,

    pub fn deinit(self: *QuantLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.s);
        _ = mlx.mlx_array_free(self.b);
    }
};

/// Requantize a row-quantized affine weight `(w, scales, biases)` from
/// `(from_gs, from_bits)` to `(to_gs, to_bits)`, chunk-wise over rows so the
/// dequantized bf16 transient stays bounded (~chunk_rows × in_features × 2 B
/// instead of the whole matrix - a 248K×5120 lm_head would otherwise
/// materialize 2.5 GB). Rows quantize independently in MLX's affine packing,
/// so per-chunk triples concatenate along axis 0 into a valid whole.
///
/// A DENSE source (`scales.ctx == null`, e.g. a bf16 lm_head) skips the
/// dequantize and quantizes the row chunk directly - same chunking, one
/// requantizer.
pub fn requantizeRows(
    s: mlx.mlx_stream,
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    from_gs: u32,
    from_bits: u32,
    from_mode: [*:0]const u8,
    to_gs: u32,
    to_bits: u32,
    chunk_rows: c_int,
) !QuantLinear {
    const w_shape = mlx.getShape(w);
    if (w_shape.len != 2) return error.UnsupportedDraftHeadShape;
    const rows: c_int = w_shape[0];

    const wv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(wv);
    const sv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(sv);
    const bv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(bv);

    var r0: c_int = 0;
    while (r0 < rows) : (r0 += chunk_rows) {
        const r1: c_int = @min(rows, r0 + chunk_rows);

        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        if (scales.ctx == null) {
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try sliceRows(&raw, w, r0, r1, s);
            try mlx.check(mlx.mlx_astype(&dense, raw, .bfloat16, s));
        } else {
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sq);
            var bq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bq);
            try sliceRows(&wq, w, r0, r1, s);
            try sliceRows(&sq, scales, r0, r1, s);
            if (biases.ctx != null) try sliceRows(&bq, biases, r0, r1, s);
            try mlx.check(mlx.mlx_dequantize(
                &dense,
                wq,
                sq,
                bq,
                mlx.mlx_optional_int.some(@intCast(from_gs)),
                mlx.mlx_optional_int.some(@intCast(from_bits)),
                from_mode,
                .{}, // global_scale
                .{ .value = .bfloat16, .has_value = true },
                s,
            ));
        }

        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            dense,
            mlx.mlx_optional_int.some(@intCast(to_gs)),
            mlx.mlx_optional_int.some(@intCast(to_bits)),
            "affine",
            .{}, // global_scale
            s,
        ));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
        var part = [3]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_vector_array_get(&part[0], triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&part[1], triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&part[2], triple, 2));
        // Realize the chunk so its dense transient can be reclaimed before
        // the next chunk builds (lazy eval would otherwise stack them all).
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for (part) |p| _ = mlx.mlx_vector_array_append_value(ev, p);
            try mlx.check(mlx.mlx_eval(ev));
        }
        _ = mlx.mlx_vector_array_append_value(wv, part[0]);
        _ = mlx.mlx_vector_array_append_value(sv, part[1]);
        _ = mlx.mlx_vector_array_append_value(bv, part[2]);
        for (part) |p| _ = mlx.mlx_array_free(p);
    }

    var out = QuantLinear{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_concatenate_axis(&out.w, wv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.s, sv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.b, bv, 0, s));
    return out;
}

fn sliceRows(out: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(src);
    const start = [_]c_int{ r0, 0 };
    const stop = [_]c_int{ r1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(out, src, &start, 2, &stop, 2, &strides, 2, s));
}

const testing = std.testing;

test "quant_rows: requantizeRows round-trips through a finer re-encode (chunked)" {
    const s = mlx.gpuStream();
    const rows: usize = 64;
    const cols: usize = 256;

    var prng = std.Random.DefaultPrng.init(42);
    const buf = try testing.allocator.alloc(f32, rows * cols);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const dense_f32 = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense_f32);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f32, .bfloat16, s));

    // "Trunk" 4-bit/gs64 triple.
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var q4 = QuantLinear{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    defer q4.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&q4.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q4.s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q4.b, triple, 2));

    // Requantize to 8-bit/gs64, chunked at 16 rows (4 chunks → exercises concat).
    var q8 = try requantizeRows(s, q4.w, q4.s, q4.b, 64, 4, "affine", 64, 8, 16);
    defer q8.deinit();

    // Dequantize both and compare - an 8-bit re-encode of 4-bit-quantized
    // values is near-lossless, so cosine must be ~1.
    var deq = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (deq) |d| {
        _ = mlx.mlx_array_free(d);
    };
    try mlx.check(mlx.mlx_dequantize(&deq[0], q4.w, q4.s, q4.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_dequantize(&deq[1], q8.w, q8.s, q8.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    for (deq) |d| try mlx.check(mlx.mlx_array_eval(d));

    const a = mlx.mlx_array_data_float32(deq[0]).?;
    const b = mlx.mlx_array_data_float32(deq[1]).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..rows * cols) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    try testing.expect(cos > 0.999);

    // Shape sanity: 8-bit packs 4 in-features per u32 → cols/4 packed cols.
    const w8_shape = mlx.getShape(q8.w);
    try testing.expectEqual(@as(c_int, @intCast(rows)), w8_shape[0]);
    try testing.expectEqual(@as(c_int, @intCast(cols / 4)), w8_shape[1]);
}

test "quant_rows: requantizeRows accepts a non-affine (mxfp8) source - the issue-#81 draft-head path" {
    // With headQuantParams fixed, buildDraftHead on an mxfp8 trunk passes
    // mode="mxfp8" and the load path's null-ctx biases into requantizeRows.
    // The source dequantize must ride that without demanding affine biases
    // (else the crash just moves from the first forward to bind time), and
    // the 3-bit affine draft re-encode must stay correlated.
    const s = mlx.gpuStream();
    const rows: usize = 64;
    const cols: usize = 256;

    var prng = std.Random.DefaultPrng.init(7);
    const buf = try testing.allocator.alloc(f32, rows * cols);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const dense_f32 = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense_f32);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f32, .bfloat16, s));

    // mxfp8 "trunk head": a (w, scales) pair - no biases tensor, by design.
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, dense, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, s));
    var qmx = QuantLinear{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    defer qmx.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&qmx.w, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qmx.s, pair, 1));
    // qmx.b stays null-ctx - exactly what the load path hands buildDraftHead.

    // The live buildDraftHead call shape: 3-bit/gs64 draft re-encode, chunked.
    var q3 = try requantizeRows(s, qmx.w, qmx.s, qmx.b, 32, 8, "mxfp8", 64, 3, 16);
    defer q3.deinit();

    var deq = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (deq) |d| {
        _ = mlx.mlx_array_free(d);
    };
    try mlx.check(mlx.mlx_dequantize(&deq[0], qmx.w, qmx.s, qmx.b, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_dequantize(&deq[1], q3.w, q3.s, q3.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(3), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    for (deq) |d| try mlx.check(mlx.mlx_array_eval(d));

    const a = mlx.mlx_array_data_float32(deq[0]).?;
    const b = mlx.mlx_array_data_float32(deq[1]).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..rows * cols) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    try testing.expect(cos > 0.95);
}
