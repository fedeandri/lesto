//! The GPU side of `lesto tune` (spec 4.6, 5 and 6): the `Worker` the scheduler (src/tune_sched.zig) drives. A window's
//! timing banks, the paired timing batches through the engine's own MoE rows dispatch (`gatherQmvGateUpRowsAct`,
//! `gatherQmvDownReduceRows`), a priming call before every timed batch so no timed batch rebuilds a kernel config, the
//! per-call bit comparison after each pair, and the identity pass at the ten cases of spec 6.1.
//!
//! The banks, the expert tables and the pair order are ported (not imported) from a standalone micro-benchmark
//! (`affineBank`, `expertTable`, `timePair`). The engine's test seams
//! (`moe_rows_layout_override`, `moe_rows_layout_plan_override` and the three fused overrides) are the tune's seams too: plain globals, safe
//! because one tune thread dispatches GPU work and the monitor thread never touches them.
const std = @import("std");
const mlx = @import("mlx.zig");
const transformer = @import("transformer.zig");
const tune_stats = @import("tune_stats.zig");
const tune_sched = @import("tune_sched.zig");
const tune_signals = @import("tune_signals.zig");
const kernel_table = @import("kernel_table.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const Allocator = std.mem.Allocator;
const Kernel = tune_stats.Kernel;
const Regime = tune_stats.Regime;
const PairTimes = tune_stats.PairTimes;
const Cand = tune_sched.Cand;
const Worker = tune_sched.Worker;
const IdentityOutcome = tune_sched.IdentityOutcome;
const BatchOutcome = tune_sched.BatchOutcome;
const PairHook = tune_sched.PairHook;
const MoeRowsLayoutPlan = transformer.MoeRowsLayoutPlan;
const Bank = transformer.MoeRowsGeluBank;

// The served verify's shapes (Gemma 4 26B-A4B: hidden 2,816, 128 experts of width 704, top-6, 4-bit affine group 64,
// bf16, GELU-tanh table, 3 rows), the probe's.
const NROWS: c_int = 3;
const E_TIMING: c_int = 128;
const TOPK: c_int = 6;
const H: c_int = 2816;
const I: c_int = 704;
const BITS: u32 = 4;
const GS: u32 = 64;
/// The identity banks' experts (the GPU test's `moeRowsLayoutBank` at 32 experts).
const E_IDENTITY: c_int = 32;

const NCOPY = tune_stats.NCOPY;
const NIND = tune_stats.NIND;
const MAXC: usize = @intCast(tune_stats.MAX_CALLS);
const SMALLC: usize = @intCast(tune_stats.CALLS_SMALL);
const SERVED_PLAN = MoeRowsLayoutPlan{ .gateup = null, .down = null };

// ── Seams ──

/// Test seam: `identity` returns `refused: "injected"` for this candidate without dispatching.
pub var moe_rows_layout_identity_refuse_override: ?Cand = null;
/// Timed batches started (four a pair) and priming calls started, over the process, each counted when it starts: the
/// stop-latency test reads them from the monitor's thread, so they are read and written atomically.
pub var tune_batches_started: u64 = 0;
pub var tune_priming_calls: u64 = 0;
/// Pairs started, over the process (atomic as the two above).
pub var tune_pairs_started: u64 = 0;
/// Test seam: called right after a pair's entry check, before its first batch, so a test can raise the stop there.
pub var tune_pair_entry_observer: ?*const fn () void = null;
/// Test seam: the comparisons of this pair (0-based, warm-up included) count as a mismatch.
pub var tune_force_mismatch_pair: ?usize = null;
/// Test seam: counted pairs a batch (the smoke sets 2); null is `PAIRS`. Only the first `PAIRS` enter the statistics.
pub var tune_counted_pairs_override: ?usize = null;
/// Test seam: called right after every dispatch call of an arm (priming and timed), before the next call, so a test can
/// read `moe_rows_layout_last_gateup` / `moe_rows_layout_last_down` while that arm's layout is the last one launched.
pub var tune_arm_observer: ?*const fn (kernel: Kernel, arm_is_reference: bool, batch_is_timed: bool) void = null;
/// Test seam: called after every identity case with its result (1-based case number).
pub var tune_identity_observer: ?*const fn (kernel: Kernel, cand: Cand, case: u8, identical: bool) void = null;

/// Spec 6.1's ten cases, in order: rows, top-k, how many of the first row's experts each later row keeps, GELU table (false: SiLU,
/// the SwiGLU path without a table). Cases 1 to 9 are the GPU test's; case 10 is the 8-row maximum of the batched tick.
pub const IdentityCase = struct { rows: c_int, topk: c_int, keep: usize, gelu: bool };
pub const IDENTITY_CASES = [10]IdentityCase{
    .{ .rows = 3, .topk = 6, .keep = 0, .gelu = true },
    .{ .rows = 3, .topk = 6, .keep = 3, .gelu = true },
    .{ .rows = 3, .topk = 6, .keep = 6, .gelu = true },
    .{ .rows = 3, .topk = 6, .keep = 2, .gelu = false },
    .{ .rows = 1, .topk = 6, .keep = 0, .gelu = true },
    .{ .rows = 2, .topk = 6, .keep = 3, .gelu = true },
    .{ .rows = 4, .topk = 6, .keep = 3, .gelu = true },
    .{ .rows = 3, .topk = 8, .keep = 3, .gelu = true },
    .{ .rows = 3, .topk = 4, .keep = 2, .gelu = true },
    .{ .rows = 8, .topk = 6, .keep = 0, .gelu = true },
};
const NCASE = IDENTITY_CASES.len;

/// The five engine globals the worker sets, saved on entry and restored on every exit.
const Seams = struct {
    on: ?bool,
    plan: ?MoeRowsLayoutPlan,
    fused: ?bool,
    gateup: ?bool,
    down: ?bool,

    fn save() Seams {
        return .{
            .on = transformer.moe_rows_layout_override,
            .plan = transformer.moe_rows_layout_plan_override,
            .fused = transformer.moe_rows_fused_override,
            .gateup = transformer.gqmv_gateup_override,
            .down = transformer.downred_override,
        };
    }

    fn restore(self: Seams) void {
        transformer.moe_rows_layout_override = self.on;
        transformer.moe_rows_layout_plan_override = self.plan;
        transformer.moe_rows_fused_override = self.fused;
        transformer.gqmv_gateup_override = self.gateup;
        transformer.downred_override = self.down;
    }

    /// The switch on and the three fused paths forced on, so a user's `MLX_SERVE_MOE_ROWS_FUSED=0` cannot make every
    /// call decline; the plan is set per arm.
    fn engage() void {
        transformer.moe_rows_layout_override = true;
        transformer.moe_rows_layout_plan_override = SERVED_PLAN;
        transformer.moe_rows_fused_override = true;
        transformer.gqmv_gateup_override = true;
        transformer.downred_override = true;
    }
};

/// The MLX diagnostics that reject a launch's configuration itself, the only failures the identity pass records as a
/// refused layout. MLX's custom-kernel dispatch has one such check, the threadgroup size against the pipeline's maximum
/// (mlx/backend/metal/custom_kernel.cpp, `Thread group size (<n>) is greater than the maximum allowed threads per
/// threadgroup (<max>).`). Other `[metal_kernel]` errors there concern inputs and outputs, which no layout changes.
const LAUNCH_REFUSAL_DIAGNOSTICS = [_][]const u8{
    "maximum allowed threads per threadgroup",
};

pub fn isLaunchRefusal(msg: []const u8) bool {
    for (LAUNCH_REFUSAL_DIAGNOSTICS) |d| {
        if (std.mem.indexOf(u8, msg, d) != null) return true;
    }
    return false;
}

fn planOf(c: Cand) MoeRowsLayoutPlan {
    return switch (c) {
        .reference => SERVED_PLAN,
        .gateup => |g| .{ .gateup = g, .down = null },
        .down => |d| .{ .gateup = null, .down = d },
    };
}

/// `sg=32 ax=1`, `sps=6 tpt=1 ax=1`, or `served`.
pub fn candText(buf: []u8, c: Cand) []const u8 {
    return switch (c) {
        .reference => "served",
        .gateup => |g| kernel_table.gateUpText(buf, g),
        .down => |d| kernel_table.downText(buf, d),
    };
}

// ── MLX helpers ──

fn emptyBank() Bank {
    return .{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new(), .bi = mlx.mlx_array_new() };
}

fn freeArr(a: *mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a.*);
    a.* = mlx.mlx_array_new();
}

fn freeBank(b: *Bank) void {
    freeArr(&b.w);
    freeArr(&b.sc);
    freeArr(&b.bi);
}

fn evalAll(arrs: []const mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| try mlx.check(mlx.mlx_vector_array_append_value(vec, a));
    try mlx.check(mlx.mlx_eval(vec));
}

fn randUniformBf16(rnd: std.Random, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return out;
}

/// One [NROWS, TOPK] expert table at 128 experts, every row's experts distinct (the probe's `expertTable`): `indep`
/// draws each row alone; `shared` makes the second row keep 3 of the first row's experts and the third keep 2 of the first two rows', the rest
/// fresh to every row, so the union is 13 of 18.
fn expertTable(rnd: std.Random, regime: Regime) mlx.mlx_array {
    const R: usize = @intCast(NROWS);
    const K: usize = @intCast(TOPK);
    const EN: usize = @intCast(E_TIMING);
    var table: [R * K]u32 = undefined;
    var used: [EN]bool = @splat(false);
    var pool_n: usize = 0;
    var pool: [R * K]u32 = undefined;
    const keep = [_]usize{ 0, 3, 2 };
    for (0..R) |r| {
        var in_row: [EN]bool = @splat(false);
        var k: usize = 0;
        if (regime == .shared and r > 0) {
            var cand: [R * K]u32 = undefined;
            @memcpy(cand[0..pool_n], pool[0..pool_n]);
            var left = pool_n;
            while (k < keep[r] and left > 0) {
                const j = rnd.uintLessThan(usize, left);
                const pick = cand[j];
                cand[j] = cand[left - 1];
                left -= 1;
                table[r * K + k] = pick;
                in_row[pick] = true;
                k += 1;
            }
        }
        while (k < K) {
            const pick: u32 = @intCast(rnd.uintLessThan(usize, EN));
            if (in_row[pick]) continue;
            // `shared` keeps the union at 13: a fresh expert there must be new to every row
            if (regime == .shared and used[pick]) continue;
            table[r * K + k] = pick;
            in_row[pick] = true;
            k += 1;
        }
        for (0..K) |q| {
            const v = table[r * K + q];
            if (!used[v]) {
                used[v] = true;
                pool[pool_n] = v;
                pool_n += 1;
            }
        }
        // slot order is the router's, not sorted: kept experts do not always sit first
        rnd.shuffle(u32, table[r * K .. (r + 1) * K]);
    }
    return mlx.mlx_array_new_data(&table, &[_]c_int{ NROWS, TOPK }, 2, .uint32);
}

fn gateUpCall(s: mlx.mlx_stream, x: mlx.mlx_array, g: Bank, u: Bank, inds: mlx.mlx_array, act: ?mlx.mlx_array) !mlx.mlx_array {
    return (try transformer.gatherQmvGateUpRowsAct(s, x, g.w, g.sc, g.bi, u.w, u.sc, u.bi, inds, BITS, GS, .affine, null, act)) orelse error.GateUpRowsDeclined;
}

fn downCall(s: mlx.mlx_stream, x: mlx.mlx_array, d: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array) !mlx.mlx_array {
    return (try transformer.gatherQmvDownReduceRows(s, x, d.w, d.sc, d.bi, inds, scores, BITS, GS, .affine)) orelse error.DownRowsDeclined;
}

// ── The worker ──

const Copy = struct { gate: Bank, up: Bank, down: Bank };

/// A window's timing inputs: `NCOPY` bank copies at 128 experts (about 430 MB each), `NIND` expert tables per regime.
const Timing = struct {
    copies: [NCOPY]Copy,
    tables: [2][NIND]mlx.mlx_array,
    x: mlx.mlx_array, // [NROWS, H]
    x_act: mlx.mlx_array, // [NROWS, TOPK, I], the down kernel's input
    scores: mlx.mlx_array, // [NROWS, TOPK]
};

/// The identity pass's inputs and the served outputs of every case, made when the window opens (before any test seam on
/// the launch is set), so a candidate's case is one dispatch and one comparison.
const IdInputs = struct {
    gate: Bank,
    up: Bank,
    down: Bank,
    x: [NCASE]mlx.mlx_array,
    scores: [NCASE]mlx.mlx_array,
    inds: [NCASE]mlx.mlx_array,
    gu_ref: [NCASE]mlx.mlx_array,
    dn_ref: [NCASE]mlx.mlx_array,
};

/// What the worker had counted when it first saw the stop flag set inside a batch.
pub const StopSnap = struct { batches: u64, priming: u64, pairs: u64 };

const Leg = struct { ref: bool, large: bool };
/// The probe's `timePair` order: even pairs ref small, ref large, arm small, arm large; odd pairs the reverse.
const EVEN_ORDER = [4]Leg{ .{ .ref = true, .large = false }, .{ .ref = true, .large = true }, .{ .ref = false, .large = false }, .{ .ref = false, .large = true } };
const ODD_ORDER = [4]Leg{ .{ .ref = false, .large = true }, .{ .ref = false, .large = false }, .{ .ref = true, .large = true }, .{ .ref = true, .large = false } };

fn bump(counter: *u64) void {
    _ = @atomicRmw(u64, counter, .Add, 1, .acq_rel);
}

pub fn readCounter(counter: *const u64) u64 {
    return @atomicLoad(u64, counter, .acquire);
}

pub const GpuWorker = struct {
    alloc: Allocator,
    seed: u64,
    /// The window's stream handle, owned from `openWindow` to `closeWindow` (or the failed open's cleanup).
    s: mlx.mlx_stream = .{},
    has_stream: bool = false,
    /// Stream handles taken and freed over the worker's life: equal whenever no window is open.
    streams_taken: u32 = 0,
    streams_freed: u32 = 0,
    /// Engine dispatch calls made (identity and timing, priming included).
    dispatches: u64 = 0,
    is_open: bool = false,
    t: Timing,
    id: IdInputs,
    /// The GELU-tanh table over every bf16 pattern (as the verify serves Gemma 4).
    act_table: mlx.mlx_array,
    /// The last window's build time of every bank, table and served identity output.
    bank_build_ms: u64 = 0,
    refused_buf: [512]u8 = undefined,
    fail_buf: [640]u8 = undefined,
    /// Set by `timeBatch` the first time it sees the stop flag; reset at each batch's start.
    stop_seen: ?StopSnap = null,
    /// The running batch's mismatch verdict and comparison count, kept where an early exit can read them.
    batch_differ: bool = false,
    batch_compared: u64 = 0,

    pub fn init(alloc: Allocator, seed: u64) GpuWorker {
        var t: Timing = undefined;
        for (&t.copies) |*c| c.* = .{ .gate = emptyBank(), .up = emptyBank(), .down = emptyBank() };
        for (&t.tables) |*row| {
            for (row) |*a| a.* = mlx.mlx_array_new();
        }
        t.x = mlx.mlx_array_new();
        t.x_act = mlx.mlx_array_new();
        t.scores = mlx.mlx_array_new();
        var id: IdInputs = undefined;
        id.gate = emptyBank();
        id.up = emptyBank();
        id.down = emptyBank();
        inline for (.{ &id.x, &id.scores, &id.inds, &id.gu_ref, &id.dn_ref }) |arr| {
            for (arr) |*a| a.* = mlx.mlx_array_new();
        }
        return .{ .alloc = alloc, .seed = seed, .t = t, .id = id, .act_table = mlx.mlx_array_new() };
    }

    pub fn deinit(self: *GpuWorker) void {
        if (self.is_open) self.closeImpl();
        self.freeAll();
        self.releaseStream();
    }

    fn releaseStream(self: *GpuWorker) void {
        if (!self.has_stream) return;
        _ = mlx.mlx_stream_free(self.s);
        self.s = .{};
        self.has_stream = false;
        self.streams_freed += 1;
    }

    pub fn worker(self: *GpuWorker) Worker {
        return .{ .ctx = self, .vt = &vtable };
    }

    const vtable: Worker.VTable = .{ .openWindow = vOpen, .closeWindow = vClose, .identity = vIdentity, .timeBatch = vTime };

    fn at(p: *anyopaque) *GpuWorker {
        return @ptrCast(@alignCast(p));
    }
    fn vOpen(p: *anyopaque) anyerror!void {
        return at(p).openImpl();
    }
    fn vClose(p: *anyopaque) void {
        at(p).closeImpl();
    }
    fn vIdentity(p: *anyopaque, k: Kernel, c: Cand, stop: *const std.atomic.Value(bool), hook: PairHook) IdentityOutcome {
        return at(p).identityImpl(k, c, stop, hook);
    }
    fn vTime(p: *anyopaque, k: Kernel, c: Cand, r: Regime, seed: u64, stop: *const std.atomic.Value(bool), hook: PairHook) BatchOutcome {
        return at(p).timeImpl(k, c, r, seed, stop, hook);
    }

    fn freeAll(self: *GpuWorker) void {
        for (&self.t.copies) |*c| {
            freeBank(&c.gate);
            freeBank(&c.up);
            freeBank(&c.down);
        }
        for (&self.t.tables) |*row| {
            for (row) |*a| freeArr(a);
        }
        freeArr(&self.t.x);
        freeArr(&self.t.x_act);
        freeArr(&self.t.scores);
        freeBank(&self.id.gate);
        freeBank(&self.id.up);
        freeBank(&self.id.down);
        inline for (.{ &self.id.x, &self.id.scores, &self.id.inds, &self.id.gu_ref, &self.id.dn_ref }) |arr| {
            for (arr) |*a| freeArr(a);
        }
        freeArr(&self.act_table);
    }

    // ── Windows ──

    /// Builds the window's banks from the fixed seed (every window rebuilds identical ones) and logs the build time.
    fn openImpl(self: *GpuWorker) !void {
        if (self.is_open) self.closeImpl();
        const io = std.Io.Threaded.global_single_threaded.io();
        const sw = io_util.Stopwatch.init(io);
        self.s = mlx.gpuStream();
        self.has_stream = true;
        self.streams_taken += 1;
        errdefer self.releaseStream();
        errdefer {
            self.freeAll();
            _ = mlx.mlx_clear_cache();
        }
        try self.buildActTable();
        try self.buildIdentity();
        try self.buildTiming();
        self.bank_build_ms = sw.read() / std.time.ns_per_ms;
        self.is_open = true;
        log.info("[tune] window banks built: bank_build_ms={d}\n", .{self.bank_build_ms});
    }

    /// Frees the window's buffers and stream handle and clears MLX's cache.
    fn closeImpl(self: *GpuWorker) void {
        self.freeAll();
        _ = mlx.mlx_clear_cache();
        self.releaseStream();
        self.is_open = false;
    }

    fn buildActTable(self: *GpuWorker) !void {
        const NT: usize = 1 << 16;
        const tb = try self.alloc.alloc(f32, NT);
        defer self.alloc.free(tb);
        for (tb, 0..) |*v, i| {
            const xv: f64 = @floatCast(@as(f32, @bitCast(@as(u32, @intCast(i)) << 16)));
            const gv = 0.5 * xv * (1.0 + std.math.tanh(0.7978845608028654 * (xv + 0.044715 * xv * xv * xv)));
            v.* = if (std.math.isFinite(gv)) @floatCast(gv) else @floatCast(xv);
        }
        const t32 = mlx.mlx_array_new_data(tb.ptr, &[_]c_int{@intCast(NT)}, 1, .float32);
        defer _ = mlx.mlx_array_free(t32);
        freeArr(&self.act_table);
        try mlx.check(mlx.mlx_astype(&self.act_table, t32, .bfloat16, self.s));
        try mlx.check(mlx.mlx_array_eval(self.act_table));
    }

    fn buildIdentity(self: *GpuWorker) !void {
        const s = self.s;
        var prng = std.Random.DefaultPrng.init(self.seed ^ 0x1D3A_7171);
        const rnd = prng.random();
        const d = &self.id;
        d.gate = try transformer.moeRowsLayoutBank(s, rnd, E_IDENTITY, I, H);
        d.up = try transformer.moeRowsLayoutBank(s, rnd, E_IDENTITY, I, H);
        d.down = try transformer.moeRowsLayoutBank(s, rnd, E_IDENTITY, H, I);
        try evalAll(&.{ d.gate.w, d.gate.sc, d.gate.bi, d.up.w, d.up.sc, d.up.bi, d.down.w, d.down.sc, d.down.bi });
        for (IDENTITY_CASES, 0..) |cs, ci| {
            d.x[ci] = try randUniformBf16(rnd, &[_]c_int{ cs.rows, H }, -1.0, 1.0, s);
            d.scores[ci] = try randUniformBf16(rnd, &[_]c_int{ cs.rows, cs.topk }, 0.05, 0.3, s);
            d.inds[ci] = transformer.moeRowsLayoutTable(rnd, @intCast(E_IDENTITY), @intCast(cs.rows), @intCast(cs.topk), cs.keep);
        }
        // the served outputs, through the same seams as every arm (the switch on, the plan's two kernels null)
        const saved = Seams.save();
        defer saved.restore();
        Seams.engage();
        for (IDENTITY_CASES, 0..) |cs, ci| {
            const act: ?mlx.mlx_array = if (cs.gelu) self.act_table else null;
            d.gu_ref[ci] = try gateUpCall(s, d.x[ci], d.gate, d.up, d.inds[ci], act);
            d.dn_ref[ci] = try downCall(s, d.gu_ref[ci], d.down, d.inds[ci], d.scores[ci]);
            try evalAll(&.{ d.gu_ref[ci], d.dn_ref[ci] });
        }
    }

    fn buildTiming(self: *GpuWorker) !void {
        const s = self.s;
        var prng = std.Random.DefaultPrng.init(self.seed);
        const rnd = prng.random();
        for (&self.t.copies) |*c| {
            c.gate = try transformer.moeRowsLayoutBank(s, rnd, E_TIMING, I, H);
            c.up = try transformer.moeRowsLayoutBank(s, rnd, E_TIMING, I, H);
            c.down = try transformer.moeRowsLayoutBank(s, rnd, E_TIMING, H, I);
            // evaluated copy by copy, so no graph spans the whole 1.7 GB
            try evalAll(&.{ c.gate.w, c.gate.sc, c.gate.bi, c.up.w, c.up.sc, c.up.bi, c.down.w, c.down.sc, c.down.bi });
        }
        for ([_]Regime{ .indep, .shared }) |rg| {
            for (&self.t.tables[@intFromEnum(rg)]) |*a| a.* = expertTable(rnd, rg);
        }
        self.t.x = try randUniformBf16(rnd, &[_]c_int{ NROWS, H }, -1.0, 1.0, s);
        self.t.x_act = try randUniformBf16(rnd, &[_]c_int{ NROWS, TOPK, I }, -1.0, 1.0, s);
        self.t.scores = try randUniformBf16(rnd, &[_]c_int{ NROWS, TOPK }, 0.05, 0.3, s);
        try evalAll(&.{ self.t.x, self.t.x_act, self.t.scores });
    }

    // ── Identity (spec 6.1) ──

    fn identityImpl(self: *GpuWorker, k: Kernel, c: Cand, stop: *const std.atomic.Value(bool), hook: PairHook) IdentityOutcome {
        // a stop already set: no GPU work at all
        if (stop.load(.acquire)) return .stopped;
        if (moe_rows_layout_identity_refuse_override) |r| {
            if (std.meta.eql(r, c)) return .{ .refused = "injected" };
        }
        if (!self.is_open) return .{ .failed = "no window open" };
        const saved = Seams.save();
        defer saved.restore();
        Seams.engage();
        transformer.moe_rows_layout_plan_override = planOf(c);
        // a message latched before this check is not this candidate's
        if (mlx.errorPending()) {
            var trash: [512]u8 = undefined;
            _ = mlx.takeError(&trash);
        }
        var compared: u64 = 0;
        for (0..NCASE) |ci| {
            var out = self.identityLaunch(k, ci) catch |e| return self.launchFailure(e);
            defer freeArr(&out);
            const want = if (k == .gateup) self.id.gu_ref[ci] else self.id.dn_ref[ci];
            const same = transformer.moeRowsLayoutBitsEqual(out, want, self.s) catch |e| return .{ .failed = self.failText("identity comparison", e) };
            compared += 1;
            const case_no: u8 = @intCast(ci + 1);
            if (tune_identity_observer) |f| f(k, c, case_no, same);
            hook.call();
            if (!same) return .{ .differs = case_no };
            if (stop.load(.acquire)) return .stopped;
        }
        return .{ .identical = .{ .calls_compared = compared } };
    }

    /// One case's launch: the candidate's dispatch (the plan seam already set), evaluated. A down candidate reads the
    /// served gate/up output, so each kernel is compared on its own.
    fn identityLaunch(self: *GpuWorker, k: Kernel, ci: usize) !mlx.mlx_array {
        const d = &self.id;
        const cs = IDENTITY_CASES[ci];
        const act: ?mlx.mlx_array = if (cs.gelu) self.act_table else null;
        self.dispatches += 1;
        var out = switch (k) {
            .gateup => try gateUpCall(self.s, d.x[ci], d.gate, d.up, d.inds[ci], act),
            .down => try downCall(self.s, d.gu_ref[ci], d.down, d.inds[ci], d.scores[ci]),
        };
        errdefer freeArr(&out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// `refused` (a verdict that excludes the layout for good and holds on resume) only when MLX's latched message is a
    /// launch-configuration rejection of this layout (`isLaunchRefusal`). Every other failure (out of memory, a command
    /// buffer error, a compile error, unknown text, no message) is `failed`, which ends the run with the message. The
    /// latch is cleared either way, so the process stays usable.
    fn launchFailure(self: *GpuWorker, e: anyerror) IdentityOutcome {
        const msg = mlx.takeError(&self.refused_buf) orelse "";
        if (isLaunchRefusal(msg)) return .{ .refused = msg };
        if (msg.len == 0) return .{ .failed = std.fmt.bufPrint(&self.fail_buf, "identity launch: {s}", .{@errorName(e)}) catch "identity launch" };
        return .{ .failed = std.fmt.bufPrint(&self.fail_buf, "identity launch: {s}: {s}", .{ @errorName(e), msg }) catch "identity launch" };
    }

    /// `<what>: <error name>[: <MLX message>]` in the worker's buffer, the latch cleared.
    fn failText(self: *GpuWorker, what: []const u8, e: anyerror) []const u8 {
        var mbuf: [512]u8 = undefined;
        const msg = mlx.takeError(&mbuf) orelse "";
        if (msg.len == 0) return std.fmt.bufPrint(&self.fail_buf, "{s}: {s}", .{ what, @errorName(e) }) catch what;
        return std.fmt.bufPrint(&self.fail_buf, "{s}: {s}: {s}", .{ what, @errorName(e), msg }) catch what;
    }

    // ── Timing (spec 5) ──

    fn timeImpl(self: *GpuWorker, k: Kernel, c: Cand, r: Regime, seed: u64, stop: *const std.atomic.Value(bool), hook: PairHook) BatchOutcome {
        self.stop_seen = null;
        self.batch_differ = false;
        self.batch_compared = 0;
        if (!self.is_open) return .{ .failed = "no window open" };
        const saved = Seams.save();
        defer saved.restore();
        Seams.engage();
        return self.timeLoop(k, c, r, seed, stop, hook) catch |e| {
            // an engine or comparison failure is not a stop: the scheduler ends the run with it. The latch is cleared.
            // The verdict order is failure, then mismatch, then stop: a failure after an observed mismatch is still
            // `failed` (the run ends), and its text says the mismatch had been seen, so neither is lost.
            const cause = self.failText("timing batch", e);
            const text = if (self.batch_differ) self.withMismatchNote(cause) else cause;
            log.err("[tune] {s}\n", .{text});
            return .{ .failed = text };
        };
    }

    const MISMATCH_NOTE = " (a bit mismatch had already been seen in this batch)";

    /// `text` with `MISMATCH_NOTE` after it, in the worker's buffer; a long `text` is cut so the note always fits.
    /// `text` may lie in that buffer, so it goes through a copy first.
    fn withMismatchNote(self: *GpuWorker, text: []const u8) []const u8 {
        var tmp: [@typeInfo(@TypeOf(self.fail_buf)).array.len]u8 = undefined;
        const keep = @min(text.len, tmp.len - MISMATCH_NOTE.len);
        @memcpy(tmp[0..keep], text[0..keep]);
        @memcpy(tmp[keep..][0..MISMATCH_NOTE.len], MISMATCH_NOTE);
        const n = keep + MISMATCH_NOTE.len;
        @memcpy(self.fail_buf[0..n], tmp[0..n]);
        return self.fail_buf[0..n];
    }

    /// The outcome of a batch that ends early (a stop): `stopped`, unless a mismatch was already observed, which is never
    /// lost to the stop: then `done` with `bits_differ`, its statistics from the pairs counted so far.
    fn stopOutcome(differ: bool, times: []const PairTimes, seed: u64, compared: u64) BatchOutcome {
        if (!differ) return .stopped;
        return .{ .done = .{ .stats = tune_stats.batchStats(times, seed), .calls_compared = compared, .bits_differ = true } };
    }

    /// True when the stop flag is set; the first time in a batch it records what had been started by then.
    fn sawStop(self: *GpuWorker, stop: *const std.atomic.Value(bool)) bool {
        if (!stop.load(.acquire)) return false;
        if (self.stop_seen == null) self.stop_seen = .{ .batches = readCounter(&tune_batches_started), .priming = readCounter(&tune_priming_calls), .pairs = readCounter(&tune_pairs_started) };
        return true;
    }

    fn timingCall(self: *GpuWorker, k: Kernel, r: Regime, i: usize) !mlx.mlx_array {
        const cp = &self.t.copies[i % NCOPY];
        const inds = self.t.tables[@intFromEnum(r)][i % NIND];
        self.dispatches += 1;
        return switch (k) {
            .gateup => gateUpCall(self.s, self.t.x, cp.gate, cp.up, inds, self.act_table),
            .down => downCall(self.s, self.t.x_act, cp.down, inds, self.t.scores),
        };
    }

    /// One timed batch of `n` calls of one arm: a priming call first (bank copy 0, expert table 0, evaluated and freed,
    /// outside the stopwatch) so the dispatch's config cache already holds this arm's layout, then the stopwatch over
    /// the `n` calls and their evaluation. The outputs stay in `outs` for the comparison after the pair. The stop is
    /// checked before the priming call and again before the timed batch: null when it is seen, and nothing more starts.
    fn leg(self: *GpuWorker, k: Kernel, c: Cand, r: Regime, is_ref: bool, n: usize, outs: *[MAXC]mlx.mlx_array, nout: *usize, stop: *const std.atomic.Value(bool)) !?u64 {
        if (self.sawStop(stop)) return null;
        transformer.moe_rows_layout_plan_override = if (is_ref) SERVED_PLAN else planOf(c);
        {
            bump(&tune_priming_calls);
            var y = try self.timingCall(k, r, 0);
            defer freeArr(&y);
            if (tune_arm_observer) |f| f(k, is_ref, false);
            try mlx.check(mlx.mlx_array_eval(y));
        }
        if (self.sawStop(stop)) return null;
        bump(&tune_batches_started);
        const io = std.Io.Threaded.global_single_threaded.io();
        const sw = io_util.Stopwatch.init(io);
        for (0..n) |i| {
            outs[i] = try self.timingCall(k, r, i);
            nout.* = i + 1;
            if (tune_arm_observer) |f| f(k, is_ref, true);
        }
        try evalAll(outs[0..n]);
        return sw.read();
    }

    fn timeLoop(self: *GpuWorker, k: Kernel, c: Cand, r: Regime, seed: u64, stop: *const std.atomic.Value(bool), hook: PairHook) !BatchOutcome {
        const counted: usize = tune_counted_pairs_override orelse tune_stats.PAIRS;
        const total: usize = tune_stats.WARM + counted;
        var times: [tune_stats.PAIRS]PairTimes = undefined;
        var kept: usize = 0;
        var compared: u64 = 0;
        var differ = false;
        // [arm: 0 reference, 1 candidate][size: 0 small, 1 large]
        var outs: [2][2][MAXC]mlx.mlx_array = undefined;
        var nout: [2][2]usize = .{ .{ 0, 0 }, .{ 0, 0 } };
        defer freeOuts(&outs, &nout);
        for (0..total) |p| {
            if (self.sawStop(stop)) return stopOutcome(differ, times[0..kept], seed, compared);
            bump(&tune_pairs_started);
            if (tune_pair_entry_observer) |f| f();
            var ns: [2][2]u64 = .{ .{ 0, 0 }, .{ 0, 0 } };
            const order = if (p % 2 == 0) EVEN_ORDER else ODD_ORDER;
            for (order) |lg| {
                const a: usize = if (lg.ref) 0 else 1;
                const z: usize = if (lg.large) 1 else 0;
                ns[a][z] = (try self.leg(k, c, r, lg.ref, if (lg.large) MAXC else SMALLC, &outs[a][z], &nout[a][z], stop)) orelse return stopOutcome(differ, times[0..kept], seed, compared);
            }
            if (self.sawStop(stop)) return stopOutcome(differ, times[0..kept], seed, compared);
            // outside the stopwatch: each candidate call against the reference call on the same copy and table (same i)
            for (0..2) |z| {
                for (0..nout[1][z]) |i| {
                    if (!try transformer.moeRowsLayoutBitsEqual(outs[1][z][i], outs[0][z][i], self.s)) {
                        differ = true;
                        self.batch_differ = true;
                    }
                    compared += 1;
                    self.batch_compared = compared;
                }
            }
            if (tune_force_mismatch_pair) |fp| {
                if (fp == p) differ = true;
            }
            self.batch_differ = differ;
            self.batch_compared = compared;
            freeOuts(&outs, &nout);
            if (p >= tune_stats.WARM and kept < tune_stats.PAIRS) {
                times[kept] = .{ .ref_small_ns = ns[0][0], .ref_large_ns = ns[0][1], .arm_small_ns = ns[1][0], .arm_large_ns = ns[1][1] };
                kept += 1;
            }
            hook.call();
        }
        if (self.sawStop(stop)) return stopOutcome(differ, times[0..kept], seed, compared);
        const stats = tune_stats.batchStats(times[0..kept], seed);
        return .{ .done = .{ .stats = stats, .calls_compared = compared, .bits_differ = differ } };
    }
};

fn freeOuts(outs: *[2][2][MAXC]mlx.mlx_array, nout: *[2][2]usize) void {
    for (0..2) |a| {
        for (0..2) |z| {
            for (outs[a][z][0..nout[a][z]]) |o| _ = mlx.mlx_array_free(o);
            nout[a][z] = 0;
        }
    }
}

// ── Tests (spec 10.2) ──

const testing = std.testing;
const ScriptedSignals = tune_signals.ScriptedSignals;
const TEST_SEED: u64 = 0x7E57_176A;
const NS_PER_MS: u64 = std.time.ns_per_ms;

extern "c" fn getloadavg(loadavg: [*]f64, nelem: c_int) c_int;

fn printIdentityCase(k: Kernel, c: Cand, case: u8, same: bool) void {
    var buf: [48]u8 = undefined;
    // a leading newline: the test runner's own output may share the line before the first print
    std.debug.print("\n[tune identity] kernel={s} cand={s} case={d} identical={}\n", .{ @tagName(k), candText(&buf, c), case, same });
}

fn identicalCalls(k: Kernel, c: Cand, out: IdentityOutcome) !u64 {
    var buf: [48]u8 = undefined;
    switch (out) {
        .identical => |i| return i.calls_compared,
        .differs => |case| std.debug.print("\n[tune identity fail] kernel={s} cand={s} differs in case {d}\n", .{ @tagName(k), candText(&buf, c), case }),
        .refused => |msg| std.debug.print("\n[tune identity fail] kernel={s} cand={s} refused: {s}\n", .{ @tagName(k), candText(&buf, c), msg }),
        .failed => |msg| std.debug.print("\n[tune identity fail] kernel={s} cand={s} failed: {s}\n", .{ @tagName(k), candText(&buf, c), msg }),
        .stopped => std.debug.print("\n[tune identity fail] kernel={s} cand={s} stopped\n", .{ @tagName(k), candText(&buf, c) }),
    }
    return error.TestIdentityNotIdentical;
}

test "moe-rows-layout tune: every candidate gives the served bits" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var gw = GpuWorker.init(testing.allocator, TEST_SEED);
    defer gw.deinit();
    const wk = gw.worker();
    try wk.openWindow();
    defer wk.closeWindow();
    tune_identity_observer = printIdentityCase;
    defer tune_identity_observer = null;
    var stop: std.atomic.Value(bool) = .init(false);
    var cands: usize = 0;
    // every candidate compares all ten cases, one comparison each
    for (kernel_table.gateUpCandidates()) |g| {
        const c: Cand = .{ .gateup = g };
        try testing.expectEqual(@as(u64, NCASE), try identicalCalls(.gateup, c, wk.identity(.gateup, c, &stop, .{})));
        cands += 1;
    }
    for (kernel_table.downCandidates()) |d| {
        const c: Cand = .{ .down = d };
        try testing.expectEqual(@as(u64, NCASE), try identicalCalls(.down, c, wk.identity(.down, c, &stop, .{})));
        cands += 1;
    }
    try testing.expectEqual(@as(usize, 58), cands);
    try testing.expectEqual(@as(u64, 58 * NCASE), gw.dispatches);

    // the opposing pair: a stop already set returns `stopped` with no dispatch at all
    stop.store(true, .release);
    const first: Cand = .{ .gateup = kernel_table.gateUpCandidates()[0] };
    const before = gw.dispatches;
    const rebuilds = transformer.moe_rows_layout_cfg_rebuilds_gateup;
    tune_identity_observer = null;
    try testing.expect(wk.identity(.gateup, first, &stop, .{}) == .stopped);
    try testing.expectEqual(before, gw.dispatches);
    try testing.expectEqual(rebuilds, transformer.moe_rows_layout_cfg_rebuilds_gateup);
}

test "moe-rows-layout tune: a refused launch is recorded" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    var gw = GpuWorker.init(testing.allocator, TEST_SEED);
    defer gw.deinit();
    const wk = gw.worker();
    try wk.openWindow();
    defer wk.closeWindow();
    var stop: std.atomic.Value(bool) = .init(false);

    // part 1, real: a 2,048-thread threadgroup, past every pipeline's 1,024 maximum
    const big: Cand = .{ .gateup = .{ .sg = 32, .ax = 1 } };
    transformer.moe_rows_layout_tg_threads_override = 2048;
    const out = wk.identity(.gateup, big, &stop, .{});
    transformer.moe_rows_layout_tg_threads_override = null;
    var msg_buf: [512]u8 = undefined;
    const msg: []const u8 = switch (out) {
        .refused => |m| blk: {
            const n = @min(m.len, msg_buf.len);
            @memcpy(msg_buf[0..n], m[0..n]);
            break :blk msg_buf[0..n];
        },
        .failed => |m| {
            std.debug.print("\n[tune refused] kernel=gateup cand=sg=32 ax=1 not refused, failed: {s}\n", .{m});
            return error.TestLaunchNotRefused;
        },
        else => {
            std.debug.print("\n[tune refused] kernel=gateup cand=sg=32 ax=1 not refused: {s}\n", .{@tagName(out)});
            return error.TestLaunchNotRefused;
        },
    };
    std.debug.print("\n[tune refused] kernel=gateup cand=sg=32 ax=1 launch refused: {s}\n", .{msg});
    // MLX's own diagnostic, not a Zig error name: the threadgroup limit and the size asked for
    try testing.expect(std.mem.indexOf(u8, msg, "maximum allowed threads per threadgroup") != null);
    try testing.expect(std.mem.indexOf(u8, msg, "2048") != null);
    if (mlx.errorPending()) {
        std.debug.print("\n[tune refused next] the latch stayed set: the identity pass needs a child process per candidate (spec 6.1)\n", .{});
        return error.TestIdentityNeedsChildProcess;
    }
    const next: Cand = .{ .gateup = .{ .sg = 16, .ax = 0 } };
    const out2 = wk.identity(.gateup, next, &stop, .{});
    if (out2 != .identical) {
        std.debug.print("\n[tune refused next] cand=sg=16 ax=0 outcome={s}: the identity pass needs a child process per candidate (spec 6.1)\n", .{@tagName(out2)});
        return error.TestIdentityNeedsChildProcess;
    }
    std.debug.print("\n[tune refused next] cand=sg=16 ax=0 identical=true\n", .{});

    // the opposing case: a generic MLX error latched during a launch is not a refusal of the layout
    mlx.armLatchingFaultForTest(2);
    const out_gen = wk.identity(.gateup, next, &stop, .{});
    const gen_fired = mlx.latchingFaultFiredForTest();
    mlx.armLatchingFaultForTest(0);
    try testing.expect(gen_fired);
    switch (out_gen) {
        .failed => |m| {
            std.debug.print("\n[tune generic failure] cand=sg=16 ax=0 failed: {s}\n", .{m});
            try testing.expect(std.mem.indexOf(u8, m, "injected latching mlx-c raise") != null);
        },
        else => {
            std.debug.print("\n[tune generic failure] cand=sg=16 ax=0 outcome={s}, not failed\n", .{@tagName(out_gen)});
            return error.TestGenericErrorNotFailed;
        },
    }
    try testing.expect(!mlx.errorPending());
    try testing.expect(wk.identity(.gateup, next, &stop, .{}) == .identical);

    // part 2, injected: no dispatch at all
    const inj: Cand = .{ .down = .{ .sps = 6, .tpt = 1, .ax = 1 } };
    moe_rows_layout_identity_refuse_override = inj;
    defer moe_rows_layout_identity_refuse_override = null;
    const rebuilds = transformer.moe_rows_layout_cfg_rebuilds_down;
    const before = gw.dispatches;
    const out3 = wk.identity(.down, inj, &stop, .{});
    try testing.expectEqual(rebuilds, transformer.moe_rows_layout_cfg_rebuilds_down);
    try testing.expectEqual(before, gw.dispatches);
    switch (out3) {
        .refused => |m| try testing.expectEqualStrings("injected", m),
        else => return error.TestInjectedNotRefused,
    }
    std.debug.print("\n[tune injected refusal] kernel=down cand=sps=6 tpt=1 ax=1 refused={s} rebuilds_down_unchanged=true\n", .{out3.refused});
}

/// What the timing smoke's observer saw.
const SmokeSeen = struct {
    cand: Cand = .reference,
    mismatches: u32 = 0,
    cand_obs: u32 = 0,
    ref_obs: u32 = 0,
    /// Observations made inside the timed loop, by arm (priming calls excluded).
    cand_timed: u64 = 0,
    ref_timed: u64 = 0,
    after_prime: u64 = 0,
    rebuilds_in_timed: u32 = 0,
};
var smoke_seen: SmokeSeen = .{};

fn rebuildsOf(k: Kernel) u64 {
    return if (k == .gateup) transformer.moe_rows_layout_cfg_rebuilds_gateup else transformer.moe_rows_layout_cfg_rebuilds_down;
}

fn smokeObserver(k: Kernel, is_ref: bool, timed: bool) void {
    const rb = rebuildsOf(k);
    if (!timed) {
        smoke_seen.after_prime = rb;
    } else if (rb != smoke_seen.after_prime) {
        smoke_seen.rebuilds_in_timed += 1;
    }
    const ok = switch (k) {
        .gateup => std.meta.eql(transformer.moe_rows_layout_last_gateup, if (is_ref) transformer.MOE_ROWS_LAYOUT_GATEUP_SERVED else smoke_seen.cand.gateup),
        .down => blk: {
            const want = if (is_ref) transformer.MOE_ROWS_LAYOUT_DOWN_SERVED else smoke_seen.cand.down;
            // the layout the call took and the kernel it launched: the array body must not run the scalar kernel
            const want_kernel: transformer.MoeRowsLayoutDownKernel = if (std.meta.eql(want, transformer.MOE_ROWS_LAYOUT_DOWN_SERVED)) .served else if (want.body == .array) .array else .scalar_body;
            break :blk std.meta.eql(transformer.moe_rows_layout_last_down, want) and transformer.moe_rows_layout_last_down_kernel == want_kernel;
        },
    };
    if (!ok) smoke_seen.mismatches += 1;
    if (is_ref) smoke_seen.ref_obs += 1 else smoke_seen.cand_obs += 1;
    if (timed) {
        if (is_ref) smoke_seen.ref_timed += 1 else smoke_seen.cand_timed += 1;
    }
}

/// The pair-entry stop: raised at the entry of the `entry_stop_pair`-th pair, with the counters at that moment.
var entry_stop_flag: ?*std.atomic.Value(bool) = null;
var entry_stop_pair: u32 = 0;
var entry_pairs_seen: u32 = 0;
var entry_snap: ?StopSnap = null;

fn entryStopObserver() void {
    entry_pairs_seen += 1;
    if (entry_pairs_seen != entry_stop_pair) return;
    entry_snap = .{ .batches = readCounter(&tune_batches_started), .priming = readCounter(&tune_priming_calls), .pairs = readCounter(&tune_pairs_started) };
    if (entry_stop_flag) |f| f.store(true, .release);
}

test "moe-rows-layout tune: timing smoke" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var gw = GpuWorker.init(testing.allocator, TEST_SEED);
    defer gw.deinit();
    const wk = gw.worker();
    try wk.openWindow();
    defer wk.closeWindow();
    tune_counted_pairs_override = 2;
    defer tune_counted_pairs_override = null;
    tune_arm_observer = smokeObserver;
    defer tune_arm_observer = null;
    var sc = try ScriptedSignals.parse(testing.allocator, "{\"at_ms\":0,\"hid_idle_s\":600}\n", .real);
    defer sc.deinit();
    // every candidate call of every pair (warm-up included) is compared: 8 small and 32 large a pair
    const want_compared: u64 = (tune_stats.WARM + 2) * (SMALLC + MAXC);

    // gate/up, down with the scalar body, and down with the array body (its kernel is checked through moe_rows_layout_last_down_kernel)
    const cands = [_]Cand{ .{ .gateup = .{ .sg = 32, .ax = 1 } }, .{ .down = .{ .sps = 6, .tpt = 1, .ax = 1 } }, .{ .down = .{ .sps = 6, .tpt = 1, .ax = 1, .body = .array } } };
    const kernels = [_]Kernel{ .gateup, .down, .down };
    for (cands, kernels) |c, k| {
        for ([_]Regime{ .indep, .shared }) |rg| {
            smoke_seen = .{ .cand = c, .after_prime = rebuildsOf(k) };
            var stop: std.atomic.Value(bool) = .init(false);
            const res = tune_sched.runOneBatch(sc.signals(), wk, k, c, rg, 1, &stop);
            try testing.expectEqual(@as(?tune_sched.DiscardReason, null), res.discard);
            const done = switch (res.outcome) {
                .done => |d| d,
                .stopped => return error.TestSmokeStopped,
                .failed => |m| {
                    std.debug.print("\n[tune smoke] kernel={s} regime={s} failed: {s}\n", .{ @tagName(k), @tagName(rg), m });
                    return error.TestSmokeFailed;
                },
            };
            std.debug.print("\n[tune smoke] kernel={s} regime={s} calls_compared={d} rebuilds_in_timed={d} bank_build_ms={d}\n", .{
                @tagName(k), @tagName(rg), done.calls_compared, smoke_seen.rebuilds_in_timed, gw.bank_build_ms,
            });
            std.debug.print("\n[tune smoke detail] kernel={s} regime={s} cand_obs={d} ref_obs={d} mismatches={d} duration_ms={d} ref_ns={d} diff_ns={d}\n", .{
                @tagName(k), @tagName(rg), smoke_seen.cand_obs, smoke_seen.ref_obs, smoke_seen.mismatches, res.duration_ns / NS_PER_MS, done.stats.ref_ns, done.stats.diff_ns,
            });
            try testing.expectEqual(@as(u32, 0), smoke_seen.mismatches);
            // the timed loop itself was observed at every call of both arms
            try testing.expectEqual(want_compared, smoke_seen.cand_timed);
            try testing.expectEqual(want_compared, smoke_seen.ref_timed);
            try testing.expectEqual(@as(u32, 0), smoke_seen.rebuilds_in_timed);
            try testing.expectEqual(want_compared, done.calls_compared);
            try testing.expect(!done.bits_differ);
        }
    }
    tune_arm_observer = null;

    // a stop raised between a pair's entry check and its first batch starts no priming call, batch or pair
    {
        var stop: std.atomic.Value(bool) = .init(false);
        entry_stop_flag = &stop;
        entry_stop_pair = 3;
        entry_pairs_seen = 0;
        entry_snap = null;
        tune_pair_entry_observer = entryStopObserver;
        defer {
            tune_pair_entry_observer = null;
            entry_stop_flag = null;
        }
        const res = tune_sched.runOneBatch(sc.signals(), wk, .gateup, cands[0], .indep, 1, &stop);
        try testing.expect(res.outcome == .stopped);
        const snap = entry_snap orelse return error.TestPairEntryNotReached;
        try testing.expectEqual(snap.batches, readCounter(&tune_batches_started));
        try testing.expectEqual(snap.priming, readCounter(&tune_priming_calls));
        try testing.expectEqual(snap.pairs, readCounter(&tune_pairs_started));
        std.debug.print("\n[tune smoke entry stop] pair=3 batches_after=0 priming_after=0 pairs_after=0\n", .{});
    }

    // the opposing case: a mismatch seen in pair 1 (0-based) and then the same stop at pair 3's entry is still the
    // batch's verdict, `done` with bits_differ, never `stopped`
    {
        var stop: std.atomic.Value(bool) = .init(false);
        entry_stop_flag = &stop;
        entry_stop_pair = 3;
        entry_pairs_seen = 0;
        entry_snap = null;
        tune_pair_entry_observer = entryStopObserver;
        tune_force_mismatch_pair = 1;
        defer {
            tune_pair_entry_observer = null;
            entry_stop_flag = null;
            tune_force_mismatch_pair = null;
        }
        const res = tune_sched.runOneBatch(sc.signals(), wk, .gateup, cands[0], .indep, 1, &stop);
        try testing.expect(entry_snap != null);
        switch (res.outcome) {
            .done => |d| {
                try testing.expect(d.bits_differ);
                try testing.expectEqual(2 * (SMALLC + MAXC), d.calls_compared);
                std.debug.print("\n[tune smoke mismatch then stop] outcome=done bits_differ=true calls_compared={d}\n", .{d.calls_compared});
            },
            else => {
                std.debug.print("\n[tune smoke mismatch then stop] outcome={s}\n", .{@tagName(res.outcome)});
                return error.TestMismatchLostToStop;
            },
        }
    }

    // an engine failure inside a batch is `failed` with its message, never a stop, and the latch is left clear
    {
        var stop: std.atomic.Value(bool) = .init(false);
        mlx.fault.arm(40);
        defer mlx.fault.disarm();
        const res = tune_sched.runOneBatch(sc.signals(), wk, .gateup, cands[0], .indep, 1, &stop);
        try testing.expect(mlx.fault.didFire());
        try testing.expectEqual(@as(?tune_sched.DiscardReason, null), res.discard);
        switch (res.outcome) {
            .failed => |m| {
                std.debug.print("\n[tune smoke failure] failed: {s}\n", .{m});
                try testing.expect(std.mem.startsWith(u8, m, "timing batch: MlxError"));
            },
            else => return error.TestFailureNotReported,
        }
        try testing.expect(!mlx.errorPending());
    }

    // a mismatch seen in pair 1 and then an engine failure in pair 2: `failed` (the failure outranks the mismatch, the
    // run ends), its text naming the mismatch too; the opposing case, mismatch then stop with no failure, is `done`
    // with bits_differ above
    {
        var stop: std.atomic.Value(bool) = .init(false);
        entry_fault_pair = 3;
        entry_pairs_seen = 0;
        tune_pair_entry_observer = entryFaultObserver;
        tune_force_mismatch_pair = 1;
        defer {
            tune_pair_entry_observer = null;
            tune_force_mismatch_pair = null;
            mlx.fault.disarm();
        }
        const res = tune_sched.runOneBatch(sc.signals(), wk, .gateup, cands[0], .indep, 1, &stop);
        try testing.expect(mlx.fault.didFire());
        switch (res.outcome) {
            .failed => |m| {
                std.debug.print("\n[tune smoke mismatch then failure] failed: {s}\n", .{m});
                try testing.expect(std.mem.startsWith(u8, m, "timing batch: MlxError"));
                try testing.expect(std.mem.endsWith(u8, m, GpuWorker.MISMATCH_NOTE));
            },
            else => {
                std.debug.print("\n[tune smoke mismatch then failure] outcome={s}\n", .{@tagName(res.outcome)});
                return error.TestFailureLostToMismatch;
            },
        }
        try testing.expect(!mlx.errorPending());
    }
}

test "moe-rows-layout tune: the mismatch note always fits after a long failure text" {
    var gw = GpuWorker.init(testing.allocator, TEST_SEED);
    defer gw.deinit();
    const short = gw.withMismatchNote("timing batch: MlxError");
    try testing.expectEqualStrings("timing batch: MlxError" ++ GpuWorker.MISMATCH_NOTE, short);
    // a text filling the whole buffer (as `failText` can leave it) is cut, the note kept
    @memset(&gw.fail_buf, 'x');
    const long = gw.withMismatchNote(gw.fail_buf[0..]);
    try testing.expectEqual(gw.fail_buf.len, long.len);
    try testing.expect(std.mem.endsWith(u8, long, GpuWorker.MISMATCH_NOTE));
    // the stop-latency hang guard: the measured ~430 ms passes, a hang does not
    try testing.expect(stopFreedWithinHangGuard(430));
    try testing.expect(stopFreedWithinHangGuard(STOP_HANG_GUARD_MS));
    try testing.expect(!stopFreedWithinHangGuard(STOP_HANG_GUARD_MS + 1));
    try testing.expect(!stopFreedWithinHangGuard(60_000));
}

/// The pair-entry fault: arms the MLX fault seam on the next checked call at the entry of the `entry_fault_pair`-th pair.
var entry_fault_pair: u32 = 0;

fn entryFaultObserver() void {
    entry_pairs_seen += 1;
    if (entry_pairs_seen == entry_fault_pair) mlx.fault.arm(1);
}

/// The stop-latency test's hang guard, 10 x TUNE_STOP_LATENCY_MAX_MS. task-8.md rules that this test asserts nothing
/// against TUNE_STOP_LATENCY_MAX_MS itself (the test machine can run beside another GPU server; deploy.md step 1 asserts the
/// bound on a quiet GPU), so only a value no busy GPU explains fails it.
const STOP_HANG_GUARD_MS: u64 = 10 * tune_sched.TUNE_STOP_LATENCY_MAX_MS;

fn stopFreedWithinHangGuard(input_to_freed_ms: u64) bool {
    return input_to_freed_ms <= STOP_HANG_GUARD_MS;
}

/// A `Signals` over a scripted source that records the worker's counters at the reading where the idle time drops,
/// the reading on which the monitor sets the stop flag (just before it sets it): the start of the window that ends when
/// the worker first sees the flag.
const FlagWatch = struct {
    inner: *ScriptedSignals,
    last_idle: ?u64 = null,
    snap: ?StopSnap = null,

    fn at(p: *anyopaque) *FlagWatch {
        return @ptrCast(@alignCast(p));
    }
    fn hid(p: *anyopaque) ?u64 {
        const self = at(p);
        const v = self.inner.hidIdleNs();
        if (v) |x| {
            if (self.snap == null and self.last_idle != null and x < self.last_idle.?) {
                self.snap = .{ .batches = readCounter(&tune_batches_started), .priming = readCounter(&tune_priming_calls), .pairs = readCounter(&tune_pairs_started) };
            }
            self.last_idle = x;
        }
        return v;
    }
    fn ac(p: *anyopaque) ?bool {
        return at(p).inner.onAc();
    }
    fn lowPow(p: *anyopaque) ?bool {
        return at(p).inner.lowPower();
    }
    fn therm(p: *anyopaque) ?u32 {
        return at(p).inner.thermalLevel();
    }
    fn gpuP(p: *anyopaque) ?u32 {
        return at(p).inner.gpuPct();
    }
    fn cpuT(p: *anyopaque) ?[4]u64 {
        return at(p).inner.cpuTicks();
    }
    fn memP(p: *anyopaque) u32 {
        return at(p).inner.memPressure();
    }
    fn absT(p: *anyopaque) u64 {
        return at(p).inner.absTimeNs();
    }
    fn contT(p: *anyopaque) u64 {
        return at(p).inner.contTimeNs();
    }
    fn waitT(p: *anyopaque, ns: u64) void {
        at(p).inner.waitNs(ns);
    }
    const vt: tune_signals.VTable = .{ .hidIdleNs = hid, .onAc = ac, .lowPower = lowPow, .thermalLevel = therm, .gpuPct = gpuP, .cpuTicks = cpuT, .memPressure = memP, .absTimeNs = absT, .contTimeNs = contT, .waitNs = waitT };

    fn signals(self: *FlagWatch) tune_signals.Signals {
        return .{ .ctx = self, .vt = &vt };
    }
};

fn activeMemory() !usize {
    var v: usize = 0;
    if (mlx.mlx_get_active_memory(&v) != 0) return error.TestActiveMemoryQueryFailed;
    return v;
}

test "moe-rows-layout tune: stop latency" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var gw = GpuWorker.init(testing.allocator, TEST_SEED);
    defer gw.deinit();
    const wk = gw.worker();

    // every window's stream handle is freed: two opens and closes, then an open that fails after taking it
    for (0..2) |_| {
        try wk.openWindow();
        wk.closeWindow();
    }
    mlx.fault.arm(1);
    const failed_open = wk.openWindow();
    const fired = mlx.fault.didFire();
    mlx.fault.disarm();
    try testing.expect(fired);
    try testing.expectError(error.MlxError, failed_open);
    try testing.expect(!gw.is_open and !gw.has_stream);
    try testing.expectEqual(@as(u32, 3), gw.streams_taken);
    try testing.expectEqual(gw.streams_taken, gw.streams_freed);

    const active_before = try activeMemory();
    try wk.openWindow();
    var closed = false;
    defer if (!closed) wk.closeWindow();
    // a batch long enough that the scripted input lands inside it on any GPU (only the first PAIRS pairs would count)
    tune_counted_pairs_override = 400;
    defer tune_counted_pairs_override = null;
    const c: Cand = .{ .gateup = .{ .sg = 32, .ax = 1 } };
    const bound_ns = (tune_sched.TUNE_MONITOR_MS + 100) * NS_PER_MS;
    var attempt: u32 = 0;
    while (true) : (attempt += 1) {
        // `at_ms` counts from the script's parse, right before the batch starts: input 100 ms in
        var sc = try ScriptedSignals.parse(testing.allocator, "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":100,\"hid_idle_s\":0}\n", .real);
        defer sc.deinit();
        var watch: FlagWatch = .{ .inner = &sc };
        const input_ns = sc.start_ns + 100 * NS_PER_MS;
        var stop: std.atomic.Value(bool) = .init(false);
        const res = tune_sched.runOneBatch(watch.signals(), wk, .gateup, c, .indep, 1, &stop);
        try testing.expectEqual(@as(?tune_sched.DiscardReason, .input), res.discard);
        try testing.expect(res.outcome == .stopped);
        const st = res.stop_times orelse return error.TestNoStopTimes;
        const flag_after_ns = st.flag_ns -| input_ns;
        if (flag_after_ns > bound_ns) {
            var la: [3]f64 = .{ 0, 0, 0 };
            _ = getloadavg(&la, 3);
            std.debug.print("\n[tune stop] flag bound missed by {d} ms, load {d:.2} {d:.2} {d:.2}\n", .{ (flag_after_ns - bound_ns) / NS_PER_MS, la[0], la[1], la[2] });
            if (attempt == 0) continue;
            return error.TestStopFlagLate;
        }
        // after the worker's first check that saw the flag: nothing new starts at all
        const seen = gw.stop_seen orelse return error.TestWorkerNeverSawTheFlag;
        const batches_after = readCounter(&tune_batches_started) - seen.batches;
        const priming_after = readCounter(&tune_priming_calls) - seen.priming;
        const pairs_after = readCounter(&tune_pairs_started) - seen.pairs;
        try testing.expectEqual(@as(u64, 0), pairs_after);
        try testing.expectEqual(@as(u64, 0), priming_after);
        try testing.expectEqual(@as(u64, 0), batches_after);
        // between the monitor's input reading (just before it sets the flag) and the worker seeing the flag: at most the
        // in-flight pair's work, one pair and 3 batches
        const read = watch.snap orelse return error.TestMonitorNeverSawTheInput;
        const pre_pairs = seen.pairs - read.pairs;
        const pre_batches = seen.batches - read.batches;
        const pre_priming = seen.priming - read.priming;
        try testing.expect(pre_pairs <= 1);
        try testing.expect(pre_batches <= 3);
        try testing.expect(pre_priming <= 3);
        wk.closeWindow();
        closed = true;
        const freed_ns = watch.signals().absTimeNs();
        const active_after = try activeMemory();
        try testing.expectEqual(gw.streams_taken, gw.streams_freed);
        const freed_ms = (freed_ns -| input_ns) / NS_PER_MS;
        std.debug.print("\n[tune stop] input_to_freed_ms={d} hang_guard_ms={d}\n", .{ freed_ms, STOP_HANG_GUARD_MS });
        std.debug.print("\n[tune stop detail] flag_ms={d} abandoned_ms={d} after_seen: batches={d} priming={d} pairs={d}; read_to_seen: batches={d} priming={d} pairs={d}; active_before_mb={d} active_after_mb={d}\n", .{
            flag_after_ns / NS_PER_MS, (st.abandoned_ns -| input_ns) / NS_PER_MS, batches_after, priming_after, pairs_after, pre_batches, pre_priming, pre_pairs, active_before >> 20, active_after >> 20,
        });
        // a hang guard only, never TUNE_STOP_LATENCY_MAX_MS itself (task-8.md; see STOP_HANG_GUARD_MS)
        try testing.expect(stopFreedWithinHangGuard(freed_ms));
        try testing.expect(active_after <= active_before + (64 << 20));
        break;
    }
}
