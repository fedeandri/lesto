//! The scheduler of `lesto tune` (spec 4.3 to 6.2): idle windows, the monitor that stops a batch when the Mac is used,
//! the discard rule, the identity pass, calibration, the screen, the final stage, `confirm2`, and `progress.jsonl` with
//! its resume rules. It sees the machine only through `Signals` (src/tune_signals.zig) and the GPU only through `Worker`
//! (src/tune_gpu.zig in the engine, a fake in the tests), and takes its numbers and rules from src/tune_stats.zig.
//!
//! Memory: a run allocates its tables, strings and the `Outcome` from the allocator it is given and frees none of it, so
//! the caller passes an arena (the CLI) or a test arena.
const std = @import("std");
const tune_signals = @import("tune_signals.zig");
const tune_stats = @import("tune_stats.zig");
const kernel_table = @import("kernel_table.zig");
const transformer = @import("transformer.zig");

const Signals = tune_signals.Signals;
const Kernel = tune_stats.Kernel;
const Regime = tune_stats.Regime;
const Phase = tune_stats.Phase;
const BatchStats = tune_stats.BatchStats;
const ServedReason = tune_stats.ServedReason;
const CandState = tune_stats.CandState;
const MoeRowsLayoutGateUp = transformer.MoeRowsLayoutGateUp;
const MoeRowsLayoutDown = transformer.MoeRowsLayoutDown;
const MoeRowsLayoutPlan = transformer.MoeRowsLayoutPlan;
const Allocator = std.mem.Allocator;

pub const TUNE_WAIT_POLL_S: u64 = 5;
pub const TUNE_MONITOR_MS: u64 = 250;
pub const TUNE_SLEEP_GAP_MS: u64 = 1000;
pub const TUNE_STOP_LATENCY_MAX_MS: u64 = 2000;

const NS_PER_MS: u64 = std.time.ns_per_ms;
const NS_PER_S: u64 = std.time.ns_per_s;
const PROGRESS_NAME = "progress.jsonl";
const LOG_NAME = "tune.log";
const MAX_CANDS = kernel_table.DOWN_CANDIDATES;
const IDENTITY_CASES: u32 = 10;
const BITS_DIFFER_TIMING = "bits differ during timing";
/// The longest worker-failure cause a reported line (384-byte buffer) and a `tune.log` line (768) carry; the rest of
/// each line takes under 130 bytes.
const FAIL_REPORT_CAUSE_MAX = 240;
const FAIL_LOG_CAUSE_MAX = 600;

const Clipped = struct { s: []const u8, more: []const u8 };

/// The first `n` bytes of `s`, and `...` when it was longer.
fn clipped(s: []const u8, n: usize) Clipped {
    return if (s.len > n) .{ .s = s[0..n], .more = "..." } else .{ .s = s, .more = "" };
}
/// Window and attempt ids a progress record may carry: far beyond any real run, and far from u32's end.
const MAX_ID: i64 = 1 << 30;

// ── The worker (the GPU side) ──

pub const Cand = union(enum) { reference, gateup: MoeRowsLayoutGateUp, down: MoeRowsLayoutDown };

pub const DiscardReason = enum { input, power, low_power, thermal, sleep, stopped, gate };

pub const IdentityOutcome = union(enum) {
    identical: struct { calls_compared: u64 },
    /// The first differing case, 1 to 10.
    differs: u8,
    /// MLX refused the launch; the message.
    refused: []const u8,
    stopped,
    /// The check itself failed (an engine or comparison error that is not a launch refusal): the run ends with it.
    failed: []const u8,
};

pub const BatchOutcome = union(enum) {
    done: struct { stats: BatchStats, calls_compared: u64, bits_differ: bool },
    stopped,
    /// The batch failed for a reason that is not a stop (an engine or comparison error): the run ends with it, it is
    /// never retried as an abandoned window.
    failed: []const u8,
};

pub const PairHook = struct {
    ctx: ?*anyopaque = null,
    f: ?*const fn (?*anyopaque) void = null,

    pub fn call(self: PairHook) void {
        if (self.f) |f| f(self.ctx);
    }
};

pub const Worker = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        /// Builds the window's banks (and logs the build time).
        openWindow: *const fn (*anyopaque) anyerror!void,
        /// Frees the window's buffers and clears MLX's cache.
        closeWindow: *const fn (*anyopaque) void,
        /// The identity check of one layout. The worker calls `between_cases` after every case, then checks `stop`, so a
        /// monitor without its thread still ticks during the check.
        identity: *const fn (*anyopaque, Kernel, Cand, *const std.atomic.Value(bool), PairHook) IdentityOutcome,
        /// One timing batch. The worker calls `between_pairs` after every pair, then checks `stop`.
        timeBatch: *const fn (*anyopaque, Kernel, Cand, Regime, u64, *const std.atomic.Value(bool), PairHook) BatchOutcome,
    };

    pub fn openWindow(self: Worker) !void {
        return self.vt.openWindow(self.ctx);
    }
    pub fn closeWindow(self: Worker) void {
        self.vt.closeWindow(self.ctx);
    }
    pub fn identity(self: Worker, k: Kernel, c: Cand, stop: *const std.atomic.Value(bool), hook: PairHook) IdentityOutcome {
        return self.vt.identity(self.ctx, k, c, stop, hook);
    }
    pub fn timeBatch(self: Worker, k: Kernel, c: Cand, r: Regime, seed: u64, stop: *const std.atomic.Value(bool), hook: PairHook) BatchOutcome {
        return self.vt.timeBatch(self.ctx, k, c, r, seed, stop, hook);
    }
};

// ── The monitor ──

/// Test seam: false turns the monitor off, so the after-batch reading alone catches input.
pub var monitor_enabled: bool = true;

pub const MonitorMode = enum { thread, manual };

pub const Monitor = struct {
    sig: Signals,
    mode: MonitorMode,
    /// Set when one of the four conditions fails, and when the external stop is set (a signal).
    win_stop: *std.atomic.Value(bool),
    /// The external stop (SIGINT, SIGTERM); null when there is none.
    user_stop: ?*const std.atomic.Value(bool) = null,
    last_idle_ns: u64 = 0,
    next_tick_ns: u64 = 0,
    /// 0 none, else 1 + the `MonitorFailure` that was first.
    failed: std.atomic.Value(u8) = .init(0),
    input_ns: u64 = 0,
    flag_ns: u64 = 0,
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn failure(self: *const Monitor) ?tune_signals.MonitorFailure {
        const v = self.failed.load(.acquire);
        return if (v == 0) null else @enumFromInt(v - 1);
    }

    /// One reading of the four in-window conditions (`monitorHolds`) at `sig.absTimeNs()`; a failure sets the stop flag
    /// and the flag time. The input time is the reading's time minus the idle time it read.
    pub fn tick(self: *Monitor) void {
        if (!monitor_enabled) return;
        if (self.user_stop) |u| {
            if (u.load(.acquire)) self.win_stop.store(true, .release);
        }
        const now = self.sig.absTimeNs();
        const m = tune_signals.monitorHolds(self.sig, self.last_idle_ns);
        if (m.idle_ns) |i| self.last_idle_ns = i;
        const f = m.failed orelse return;
        if (self.failed.load(.acquire) == 0) {
            self.input_ns = if (f == .input) now -| (m.idle_ns orelse 0) else now;
            self.flag_ns = now;
            self.failed.store(@as(u8, @intFromEnum(f)) + 1, .release);
        }
        self.win_stop.store(true, .release);
    }

    fn threadMain(self: *Monitor) void {
        while (!self.done.load(.acquire)) {
            self.tick();
            self.sig.waitNs(TUNE_MONITOR_MS * NS_PER_MS);
        }
    }

    pub fn start(self: *Monitor) !void {
        self.last_idle_ns = self.sig.hidIdleNs() orelse 0;
        self.next_tick_ns = self.sig.absTimeNs() + TUNE_MONITOR_MS * NS_PER_MS;
        self.done.store(false, .release);
        if (self.mode == .thread and monitor_enabled) {
            // no thread: the pair hook ticks the monitor instead, so the in-batch checks are never silently off
            self.thread = std.Thread.spawn(.{}, threadMain, .{self}) catch blk: {
                self.mode = .manual;
                break :blk null;
            };
        }
    }

    pub fn join(self: *Monitor) void {
        self.done.store(true, .release);
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    /// Manual mode: one `tick` for each `TUNE_MONITOR_MS` boundary the clock passed since the last one. The scheduler
    /// hands this to the worker as its `PairHook`, so a fake batch that advances the clock a pair at a time is monitored
    /// at the points a thread would see.
    pub fn catchUp(self: *Monitor) void {
        if (self.mode != .manual or !monitor_enabled) return;
        const now = self.sig.absTimeNs();
        var n: u32 = 0;
        while (now >= self.next_tick_ns and n < 8) : (n += 1) {
            self.tick();
            self.next_tick_ns += TUNE_MONITOR_MS * NS_PER_MS;
        }
        if (now >= self.next_tick_ns) self.next_tick_ns = now + TUNE_MONITOR_MS * NS_PER_MS;
    }

    /// The hook a worker calls between pairs or identity cases: `catchUp` in manual mode, nothing when the thread runs.
    pub fn hook(self: *Monitor) PairHook {
        return if (self.mode == .manual) .{ .ctx = self, .f = hookFn } else .{};
    }
};

fn hookFn(ctx: ?*anyopaque) void {
    const m: *Monitor = @ptrCast(@alignCast(ctx.?));
    m.catchUp();
}

fn discardOfFailure(f: tune_signals.MonitorFailure) DiscardReason {
    return switch (f) {
        .input => .input,
        .ac => .power,
        .low_power => .low_power,
        .thermal => .thermal,
    };
}

/// The four monotonic times of one stop (`absTimeNs`): the input, the flag being set, the batch abandoned, the buffers freed.
pub const StopTimes = struct { input_ns: u64, flag_ns: u64, abandoned_ns: u64, freed_ns: u64 };

pub const BatchResult = struct {
    outcome: BatchOutcome,
    discard: ?DiscardReason,
    stop_times: ?StopTimes,
    duration_ns: u64,
    /// True when the stop came from the external flag and no monitor condition failed.
    user_stop: bool = false,
};

/// One batch with the monitor running and the after-batch discard rule applied (spec 4.4): a monitor failure, a gap
/// between the two clocks above `TUNE_SLEEP_GAP_MS`, or an idle time shorter than the batch's duration plus 250 ms
/// discards it. `stop` is the flag the worker watches; `user_stop`, when given, is copied into it by the monitor.
pub fn runBatchWith(sig: Signals, w: Worker, kernel: Kernel, cand: Cand, regime: Regime, boot_seed: u64, stop: *std.atomic.Value(bool), user_stop: ?*const std.atomic.Value(bool), mode: MonitorMode) BatchResult {
    const abs0 = sig.absTimeNs();
    const cont0 = sig.contTimeNs();
    var mon: Monitor = .{ .sig = sig, .mode = mode, .win_stop = stop, .user_stop = user_stop };
    mon.start() catch {};
    const out = w.timeBatch(kernel, cand, regime, boot_seed, stop, mon.hook());
    // the batch ends when the worker returns: joining the monitor (up to one sleep of `TUNE_MONITOR_MS`) is not GPU time
    const abs1 = sig.absTimeNs();
    const cont1 = sig.contTimeNs();
    const abandoned = abs1;
    mon.join();
    const duration = abs1 - abs0;
    // a failed batch is neither kept nor discarded: the caller ends the run with it
    if (out == .failed) return .{ .outcome = out, .discard = null, .stop_times = null, .duration_ns = duration };

    var discard: ?DiscardReason = null;
    var times: ?StopTimes = null;
    var user = false;
    if (mon.failure()) |f| {
        discard = discardOfFailure(f);
        times = .{ .input_ns = mon.input_ns, .flag_ns = mon.flag_ns, .abandoned_ns = abandoned, .freed_ns = abandoned };
    } else if (out == .stopped) {
        discard = .stopped;
        user = true;
    } else if ((cont1 - cont0) -| duration > TUNE_SLEEP_GAP_MS * NS_PER_MS) {
        discard = .sleep;
    } else {
        const read_ns = sig.absTimeNs();
        const idle = sig.hidIdleNs();
        if (idle == null or idle.? < duration + TUNE_MONITOR_MS * NS_PER_MS) {
            discard = .input;
            times = .{ .input_ns = read_ns -| (idle orelse 0), .flag_ns = read_ns, .abandoned_ns = read_ns, .freed_ns = read_ns };
        }
    }
    return .{ .outcome = out, .discard = discard, .stop_times = times, .duration_ns = duration, .user_stop = user };
}

/// `runBatchWith` with the monitor in thread mode and no external stop: the form the GPU tests call.
pub fn runOneBatch(sig: Signals, w: Worker, kernel: Kernel, cand: Cand, regime: Regime, boot_seed: u64, stop: *std.atomic.Value(bool)) BatchResult {
    return runBatchWith(sig, w, kernel, cand, regime, boot_seed, stop, null, .thread);
}

// ── The run's inputs and outputs ──

pub const Reporter = struct {
    ctx: ?*anyopaque = null,
    onLine: ?*const fn (?*anyopaque, []const u8) void = null,

    fn line(self: Reporter, text: []const u8) void {
        if (self.onLine) |f| f(self.ctx, text);
    }
};

pub const RunConfig = struct {
    io: std.Io,
    /// The tune folder, opened by the caller.
    root: std.Io.Dir,
    monitor: MonitorMode,
    key: kernel_table.Key,
    idle_minutes: u32,
    mode: enum { tune, verify } = .tune,
    verify_plan: MoeRowsLayoutPlan = .{ .gateup = null, .down = null },
    boot_seed_base: u64 = 0x6c657374,
    reporter: Reporter = .{},
};

pub const Outcome = struct {
    plan: MoeRowsLayoutPlan,
    served_reasons: [2]?ServedReason,
    evidence: [2]?kernel_table.KernelEvidence,
    identity: struct { plans_checked: u32, cases: u32, calls_compared: u64, excluded: []const kernel_table.Excluded },
    not_measurable: []const kernel_table.NotMeasurable,
    windows_used: u32,
    batches_kept: u32,
    discarded: [7]u32,
};

pub const SchedResult = union(enum) { finished: Outcome, stopped };

// ── Progress file ──

pub const BatchRec = struct {
    phase: Phase,
    kernel: Kernel,
    cand: []const u8,
    regime: ?Regime,
    window: u32,
    kept: bool,
    reason: ?[]const u8,
    stats: ?BatchStats,
    calls_compared: u64,
    attempt: ?u32,
};

pub const Rec = union(enum) {
    batch: BatchRec,
    marker: struct { attempt: u32, window: u32 },
};

pub const Progress = struct {
    /// The file was absent, or its header named another key or plan space (`restart_reason` says which of the two).
    restarted: bool,
    restart_reason: []const u8,
    /// True when a progress file existed (and was replaced, when `restarted`).
    had_file: bool,
    recs: []const Rec,
    /// Bytes of the file that are whole, parsed lines: anything after is a torn tail, truncated before the next append.
    valid_len: u64,
    file_len: u64,
};

fn jsonStr(list: *std.ArrayList(u8), alloc: Allocator, s: []const u8) !void {
    try list.append(alloc, '"');
    for (s) |c| switch (c) {
        '"' => try list.appendSlice(alloc, "\\\""),
        '\\' => try list.appendSlice(alloc, "\\\\"),
        '\n' => try list.appendSlice(alloc, "\\n"),
        '\r' => try list.appendSlice(alloc, "\\r"),
        '\t' => try list.appendSlice(alloc, "\\t"),
        0...8, 11, 12, 14...31, 127 => {
            var b: [6]u8 = undefined;
            try list.appendSlice(alloc, std.fmt.bufPrint(&b, "\\u{x:0>4}", .{c}) catch unreachable);
        },
        else => try list.append(alloc, c),
    };
    try list.append(alloc, '"');
}

fn headerLine(alloc: Allocator, key: kernel_table.Key) ![]u8 {
    var l: std.ArrayList(u8) = .empty;
    try l.appendSlice(alloc, "{\"progress\":1,\"key\":{\"gpu_arch\":");
    try jsonStr(&l, alloc, key.gpu_arch);
    try l.print(alloc, ",\"gpu_cores\":{d},\"hw_model\":", .{key.gpu_cores});
    try jsonStr(&l, alloc, key.hw_model);
    try l.appendSlice(alloc, ",\"os_build\":");
    try jsonStr(&l, alloc, key.os_build);
    try l.appendSlice(alloc, ",\"kernel_hash\":");
    try jsonStr(&l, alloc, key.kernel_hash);
    try l.appendSlice(alloc, ",\"mlx_version\":");
    try jsonStr(&l, alloc, key.mlx_version);
    try l.print(alloc, "}},\"space\":{d}}}", .{kernel_table.TUNE_PLAN_SPACE_VERSION});
    return l.toOwnedSlice(alloc);
}

fn objStr(o: std.json.ObjectMap, k: []const u8) ?[]const u8 {
    const v = o.get(k) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn objInt(o: std.json.ObjectMap, k: []const u8) ?i64 {
    const v = o.get(k) orelse return null;
    return switch (v) {
        .integer => |i| i,
        else => null,
    };
}

/// A field the writer always writes as a string or `null`: `.absent` when it is missing or of another type, which makes
/// the record bad (a missing field never reads as an explicit `null`).
const NullableStr = union(enum) { absent, null_value, string: []const u8 };

fn objNullableStr(o: std.json.ObjectMap, k: []const u8) NullableStr {
    const v = o.get(k) orelse return .absent;
    return switch (v) {
        .null => .null_value,
        .string => |s| .{ .string = s },
        else => .absent,
    };
}

/// An optional integer field: `.absent` when missing, `.bad` when of another type or outside `[lo, hi]`.
const OptInt = union(enum) { absent, bad, value: i64 };

fn objIntIn(o: std.json.ObjectMap, k: []const u8, lo: i64, hi: i64) OptInt {
    const v = o.get(k) orelse return .absent;
    return switch (v) {
        .integer => |i| if (i >= lo and i <= hi) .{ .value = i } else .bad,
        else => .bad,
    };
}

/// The range a progress record's timing fields may hold: far beyond any real batch, and far enough from i64's ends that
/// the gates and `judgeConfirm` (differences of two fields, times 100) cannot overflow.
const TIME_BOUND: i64 = 1 << 50;

fn headerMatches(o: std.json.ObjectMap, key: kernel_table.Key) bool {
    if ((objInt(o, "progress") orelse 0) != 1) return false;
    if ((objInt(o, "space") orelse -1) != kernel_table.TUNE_PLAN_SPACE_VERSION) return false;
    const kv = o.get("key") orelse return false;
    const ko = switch (kv) {
        .object => |x| x,
        else => return false,
    };
    return std.mem.eql(u8, objStr(ko, "gpu_arch") orelse return false, key.gpu_arch) and
        (objInt(ko, "gpu_cores") orelse -1) == key.gpu_cores and
        std.mem.eql(u8, objStr(ko, "hw_model") orelse return false, key.hw_model) and
        std.mem.eql(u8, objStr(ko, "os_build") orelse return false, key.os_build) and
        std.mem.eql(u8, objStr(ko, "kernel_hash") orelse return false, key.kernel_hash) and
        std.mem.eql(u8, objStr(ko, "mlx_version") orelse return false, key.mlx_version);
}

/// One record line, or null when the line is not a record; running out of memory is an error, never a bad line.
fn parseRec(alloc: Allocator, text: []const u8) !?Rec {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, text, .{}) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    const o = switch (parsed.value) {
        .object => |x| x,
        else => return null,
    };
    if (o.get("stage_done") != null) {
        if (!std.mem.eql(u8, objStr(o, "stage_done") orelse return null, "final")) return null;
        const att = objInt(o, "attempt") orelse return null;
        const win = objInt(o, "window") orelse return null;
        if (att < 0 or win < 0 or att >= MAX_ID or win >= MAX_ID) return null;
        return .{ .marker = .{ .attempt = @intCast(att), .window = @intCast(win) } };
    }
    const phase = std.meta.stringToEnum(Phase, objStr(o, "phase") orelse return null) orelse return null;
    const kernel = std.meta.stringToEnum(Kernel, objStr(o, "kernel") orelse return null) orelse return null;
    const cand = objStr(o, "cand") orelse return null;
    // `regime` and `reason` are always written, as a string or null: missing or another type is a bad record
    const regime: ?Regime = switch (objNullableStr(o, "regime")) {
        .absent => return null,
        .null_value => null,
        .string => |r| std.meta.stringToEnum(Regime, r) orelse return null,
    };
    const reason: ?[]const u8 = switch (objNullableStr(o, "reason")) {
        .absent => return null,
        .null_value => null,
        .string => |s| s,
    };
    const win = objInt(o, "window") orelse return null;
    const kept = switch (o.get("kept") orelse return null) {
        .bool => |b| b,
        else => return null,
    };
    // the timing fields come all together or not at all, each within `TIME_BOUND`
    var stats: ?BatchStats = null;
    const time_names = [_][]const u8{ "ref_ns", "cand_ns", "diff_ns", "q1_ns", "q3_ns", "lo_ns", "hi_ns" };
    var times: [time_names.len]i64 = undefined;
    var present: usize = 0;
    for (time_names, 0..) |name, i| switch (objIntIn(o, name, -TIME_BOUND, TIME_BOUND)) {
        .absent => {},
        .bad => return null,
        .value => |x| {
            times[i] = x;
            present += 1;
        },
    };
    if (present == time_names.len) {
        stats = .{ .ref_ns = times[0], .cand_ns = times[1], .diff_ns = times[2], .q1_ns = times[3], .q3_ns = times[4], .lo_ns = times[5], .hi_ns = times[6] };
    } else if (present != 0) return null;
    // a count is never negative; its sums saturate, so its size is not bounded here
    const calls: u64 = switch (objIntIn(o, "calls_compared", 0, std.math.maxInt(i64))) {
        .absent => 0,
        .bad => return null,
        .value => |x| @intCast(x),
    };
    const att: ?u32 = switch (objIntIn(o, "attempt", 0, MAX_ID - 1)) {
        .absent => null,
        .bad => return null,
        .value => |a| @intCast(a),
    };
    if (win < 0 or win >= MAX_ID) return null;
    return .{ .batch = .{
        .phase = phase,
        .kernel = kernel,
        .cand = cand,
        .regime = regime,
        .window = @intCast(win),
        .kept = kept,
        .reason = reason,
        .stats = stats,
        .calls_compared = calls,
        .attempt = att,
    } };
}

fn isRegularMode(mode: anytype) bool {
    const S_IFMT: u32 = 0o170000;
    const S_IFREG: u32 = 0o100000;
    return @as(u32, @intCast(mode)) & S_IFMT == S_IFREG;
}

/// True when `fd` is a regular file (the reader and both writers of the tune folder refuse anything else).
fn isRegularFd(fd: c_int) bool {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return false;
    return isRegularMode(st.mode);
}

/// The progress file's bytes, or null when there is none. Opened without following a link and without blocking, and only
/// when it is a regular file, so a link or a FIFO at its name is an error and not a hang or a read of something else.
fn readProgressBytes(alloc: Allocator, root: std.Io.Dir) !?[]u8 {
    const fd = std.c.openat(root.handle, PROGRESS_NAME, .{ .ACCMODE = .RDONLY, .NONBLOCK = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c.errno(fd) == .NOENT) null else error.TuneFolderUnwritable;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.TuneFolderUnwritable;
    if (!isRegularMode(st.mode)) return error.TuneFolderUnwritable;
    const size: usize = @intCast(st.size);
    if (size > (256 << 20)) return error.TuneFolderUnwritable;
    const buf = try alloc.alloc(u8, size);
    var got: usize = 0;
    while (got < size) {
        const n = std.c.read(fd, buf[got..].ptr, size - got);
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.TuneFolderUnwritable;
        got += @intCast(n);
    }
    return buf;
}

/// Reads `progress.jsonl` and applies the resume rules of spec 4.5: a missing or mismatched header replaces the file, a
/// last line without its newline or not parseable is ignored (and truncated by the next writer).
pub fn loadProgress(alloc: Allocator, io: std.Io, root: std.Io.Dir, key: kernel_table.Key) !Progress {
    const absent: Progress = .{ .restarted = false, .restart_reason = "", .had_file = false, .recs = &.{}, .valid_len = 0, .file_len = 0 };
    _ = io;
    const bytes = (try readProgressBytes(alloc, root)) orelse return absent;
    if (bytes.len == 0) return absent; // an empty file is the same as none: the header is written next
    var recs: std.ArrayList(Rec) = .empty;
    var off: usize = 0;
    var valid: usize = 0;
    var header_ok = false;
    var first = true;
    while (off < bytes.len) {
        const nl = std.mem.indexOfScalarPos(u8, bytes, off, '\n') orelse break; // a line without its newline is torn
        const text = bytes[off..nl];
        if (first) {
            first = false;
            // only a line that is not JSON is a malformed header: running out of memory is an error, never a restart
            const parsed = std.json.parseFromSlice(std.json.Value, alloc, text, .{}) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => break,
            };
            const o = switch (parsed.value) {
                .object => |x| x,
                else => break,
            };
            header_ok = headerMatches(o, key);
            if (!header_ok) break;
        } else {
            // a bad line followed by good ones is not a torn tail: it is skipped and stays in the file; only trailing bad
            // lines are cut (`valid` moves only past a good line)
            if (try parseRec(alloc, text)) |rec| {
                try recs.append(alloc, rec);
                off = nl + 1;
                valid = off;
            } else {
                off = nl + 1;
            }
            continue;
        }
        off = nl + 1;
        valid = off;
    }
    if (!header_ok) {
        return .{
            .restarted = true,
            .restart_reason = "the progress file's header names another Mac, build or plan space",
            .had_file = true,
            .recs = &.{},
            .valid_len = 0,
            .file_len = bytes.len,
        };
    }
    return .{ .restarted = false, .restart_reason = "", .had_file = true, .recs = try recs.toOwnedSlice(alloc), .valid_len = valid, .file_len = bytes.len };
}

pub const ProgressSummary = struct {
    kept: u32,
    discarded: [7]u32,
    windows_used: u32,
    /// Kept `final` and `confirm1` lines of an attempt with no `stage_done` marker: they count for `--show` only.
    void_lines: u32,
    excluded: []const kernel_table.Excluded,
    not_measurable: []const kernel_table.NotMeasurable,
};

/// What `--show` prints of an unfinished run (served reasons are decisions of the run, not lines, so they are not here).
pub fn summarize(alloc: Allocator, p: Progress) !ProgressSummary {
    var s: ProgressSummary = .{ .kept = 0, .discarded = @splat(0), .windows_used = 0, .void_lines = 0, .excluded = &.{}, .not_measurable = &.{} };
    var excluded: std.ArrayList(kernel_table.Excluded) = .empty;
    var fails: std.StringHashMapUnmanaged(u32) = .empty;
    var nm: std.ArrayList(kernel_table.NotMeasurable) = .empty;
    var markers: std.AutoHashMapUnmanaged(u32, void) = .empty;
    // one exclusion per candidate, the last line's reason winning, as replay keeps it
    var excluded_at: std.StringHashMapUnmanaged(usize) = .empty;
    for (p.recs) |r| switch (r) {
        .marker => |m| try markers.put(alloc, m.attempt, {}),
        .batch => {},
    };
    for (p.recs) |r| switch (r) {
        .marker => {},
        .batch => |b| {
            if (b.kept and b.phase != .identity) {
                s.kept +|= 1;
                if ((b.phase == .final or b.phase == .confirm1) and b.attempt != null and !markers.contains(b.attempt.?)) s.void_lines +|= 1;
            }
            if (!b.kept) if (b.reason) |why| {
                for (kernel_table.DISCARD_REASONS, 0..) |name, i| {
                    if (std.mem.eql(u8, name, why)) s.discarded[i] +|= 1;
                }
            };
            // the candidate decisions below are replay's: a line naming no candidate of the plan space decides nothing
            if (b.phase == .cal or !knownCand(b.kernel, b.cand)) continue;
            const key = try std.fmt.allocPrint(alloc, "{s} {s}", .{ @tagName(b.kernel), b.cand });
            var why_excluded: ?[]const u8 = identityExclusion(b);
            if (!b.kept) if (b.reason) |why| {
                if (b.phase != .identity and std.mem.eql(u8, why, "gate")) {
                    const n = (fails.get(key) orelse 0) +| 1;
                    try fails.put(alloc, key, n);
                    if (n == tune_stats.TUNE_MAX_GATE_FAILS) try nm.append(alloc, .{ .kernel = @tagName(b.kernel), .cand = b.cand });
                }
                if (b.phase != .identity and std.mem.eql(u8, why, BITS_DIFFER_TIMING)) why_excluded = BITS_DIFFER_TIMING;
            };
            if (why_excluded) |why| {
                if (excluded_at.get(key)) |at| {
                    excluded.items[at].reason = why;
                } else {
                    try excluded_at.put(alloc, key, excluded.items.len);
                    try excluded.append(alloc, .{ .kernel = @tagName(b.kernel), .cand = b.cand, .reason = why });
                }
            }
        },
    };
    s.windows_used = try distinctWindows(alloc, p.recs);
    s.excluded = try excluded.toOwnedSlice(alloc);
    s.not_measurable = try nm.toOwnedSlice(alloc);
    return s;
}

/// The windows the records name (batch lines and markers), each id once: replay's `windows_used` and `--show`'s.
fn distinctWindows(alloc: Allocator, recs: []const Rec) !u32 {
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(alloc);
    for (recs) |r| try seen.put(alloc, switch (r) {
        .marker => |m| m.window,
        .batch => |b| b.window,
    }, {});
    return seen.count();
}

/// `bits differ in case <1 to 10>` or `launch refused: <message>`: the two reasons an identity pass writes.
fn identityReasonValid(why: []const u8) bool {
    const bits = "bits differ in case ";
    if (std.mem.startsWith(u8, why, bits)) {
        const n = std.fmt.parseInt(u32, why[bits.len..], 10) catch return false;
        return n >= 1 and n <= IDENTITY_CASES;
    }
    return std.mem.startsWith(u8, why, "launch refused: ");
}

/// The exclusion a kept identity line records, when its reason is one an identity pass writes. Replay and `summarize`
/// both read identity lines through this, so `--show` never lists a candidate the scheduler would check again.
fn identityExclusion(b: BatchRec) ?[]const u8 {
    if (b.phase != .identity or !b.kept) return null;
    const why = b.reason orelse return null;
    return if (identityReasonValid(why)) why else null;
}

/// A kept identity line with no exclusion reason proves the layout identical only when it compared every case: fewer
/// than `IDENTITY_CASES` comparisons is an incomplete line, and the layout is checked again.
fn identityComplete(b: BatchRec) bool {
    return b.phase == .identity and b.kept and b.reason == null and b.calls_compared >= IDENTITY_CASES;
}

/// True when `text` is a candidate of the plan space for `k` (replay looks lines up the same way).
fn knownCand(k: Kernel, text: []const u8) bool {
    var b: [48]u8 = undefined;
    const gu = kernel_table.gateUpCandidates();
    const dn = kernel_table.downCandidates();
    switch (k) {
        .gateup => for (gu) |c| {
            if (std.mem.eql(u8, kernel_table.gateUpText(&b, c), text)) return true;
        },
        .down => for (dn) |c| {
            if (std.mem.eql(u8, kernel_table.downText(&b, c), text)) return true;
        },
    }
    return false;
}

/// An append-only file in the tune folder: one `write` a line, optionally `fsync`ed.
const AppendFile = struct {
    fd: c_int,

    /// Opened without following a link and without blocking, and kept only when it is a regular file: a FIFO with no
    /// reader is refused by the open itself, one with a reader (or a device) by the file-type check. The descriptor then
    /// goes back to blocking writes.
    fn open(root: std.Io.Dir, name: [*:0]const u8, truncate: bool) !AppendFile {
        const fd = std.c.openat(root.handle, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .NONBLOCK = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.TuneFolderUnwritable;
        errdefer _ = std.c.close(fd);
        if (!isRegularFd(fd)) return error.TuneFolderUnwritable;
        const fl = std.c.fcntl(fd, std.c.F.GETFL);
        if (fl < 0) return error.TuneFolderUnwritable;
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        if (std.c.fcntl(fd, std.c.F.SETFL, fl & ~nonblock) < 0) return error.TuneFolderUnwritable;
        // truncated only once it is known to be a regular file
        if (truncate and std.c.ftruncate(fd, 0) != 0) return error.TuneFolderUnwritable;
        return .{ .fd = fd };
    }

    fn close(self: *AppendFile) void {
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
    }

    fn truncateTo(self: *AppendFile, len: u64) !void {
        if (std.c.ftruncate(self.fd, @intCast(len)) != 0) return error.TuneFolderUnwritable;
    }

    fn appendLine(self: *AppendFile, alloc: Allocator, text: []const u8, sync: bool) !void {
        const buf = try alloc.alloc(u8, text.len + 1);
        defer alloc.free(buf);
        @memcpy(buf[0..text.len], text);
        buf[text.len] = '\n';
        var off: usize = 0;
        while (off < buf.len) {
            const n = std.c.write(self.fd, buf[off..].ptr, buf.len - off);
            if (n < 0 and std.c.errno(n) == .INTR) continue;
            if (n <= 0) return error.TuneFolderUnwritable;
            off += @intCast(n);
        }
        if (sync and std.c.fsync(self.fd) != 0) return error.TuneFolderUnwritable;
    }
};

// ── The state machine ──

const Item = struct { k: u8, idx: u8, regime: u8 };

const KS = struct {
    n: usize,
    cands: [MAX_CANDS]CandState = @splat(.{ .identical = null, .bits_excluded = false, .gate_fails = 0, .screen = .{ null, null } }),
    reason: [MAX_CANDS]?[]const u8 = @splat(null),
    id_done: [MAX_CANDS]bool = @splat(false),
    served: ?ServedReason = null,
    pick_final: ?usize = null,
    had_finalists: bool = false,
    /// The pick of the finished final stage: the candidate `confirm1` timed and `confirm2` times.
    pick: usize = 0,
    /// `confirm1` judged ok: the kernel waits for `confirm2`.
    awaiting: bool = false,
    c1: [2]?BatchStats = .{ null, null },
    c2: [2]?BatchStats = .{ null, null },
    c1_window: u32 = 0,
    c2_window: [2]u32 = .{ 0, 0 },
    evidence: ?kernel_table.KernelEvidence = null,

    fn decided(self: *const KS) bool {
        return self.served != null or self.pick_final != null;
    }
};

/// `bits_differ_closed`: the bits differ and the batch was also discarded for a timing reason, so the verdict stands and
/// the window closes.
const TimedResult = enum { kept, gate, bits_differ, bits_differ_closed, closed };
const AttemptResult = enum { finished, abandoned, closed };

const Sched = struct {
    alloc: Allocator,
    cfg: RunConfig,
    sig: Signals,
    w: Worker,
    user_stop: *std.atomic.Value(bool),
    win_stop: std.atomic.Value(bool) = .init(false),
    prog: AppendFile,
    log: AppendFile,
    gu: [kernel_table.GATEUP_CANDIDATES]MoeRowsLayoutGateUp,
    dn: [kernel_table.DOWN_CANDIDATES]MoeRowsLayoutDown,
    texts: [2][MAX_CANDS][]const u8,
    ks: [2]KS,
    window_id: u32 = 0,
    final_window: u32 = 0,
    final_done: bool = false,
    cal_ref: [2][2]i64 = .{ .{ 0, 0 }, .{ 0, 0 } },
    next_attempt: u32 = 1,
    gate_abandons: u32 = 0,
    queue: std.ArrayList(Item) = .empty,
    qpos: usize = 0,
    queue_built: bool = false,
    pending_stop: ?StopTimes = null,
    last: BatchStats = undefined,
    last_calls: u64 = 0,
    batches_kept: u32 = 0,
    discarded: [7]u32 = @splat(0),
    windows_used: u32 = 0,
    gpu_ns: u64 = 0,
    boot_counter: u64 = 0,
    calls_total: u64 = 0,
    window_discards: u32 = 0,

    fn kname(k: usize) []const u8 {
        return if (k == 0) "gateup" else "down";
    }

    fn candOf(self: *const Sched, k: usize, idx: usize) Cand {
        return if (k == 0) .{ .gateup = self.gu[idx] } else .{ .down = self.dn[idx] };
    }

    fn candIndex(self: *const Sched, k: usize, text: []const u8) ?usize {
        for (0..self.ks[k].n) |i| {
            if (std.mem.eql(u8, self.texts[k][i], text)) return i;
        }
        return null;
    }

    fn logf(self: *Sched, comptime fmt: []const u8, args: anytype) void {
        var buf: [768]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..0];
        self.log.appendLine(self.alloc, text, false) catch {}; // the log is a convenience: a failed write never stops the run
    }

    /// A worker failure that is not a stop: logged and reported once, and the run ends with `error.TuneWorkerFailed`
    /// instead of reopening a window to retry the same failing work.
    /// A long cause is cut to fit the line (the kernel, candidate and the cause's start always fit), never dropped.
    fn workerFailed(self: *Sched, phase: Phase, k: usize, text: []const u8, msg: []const u8) error{TuneWorkerFailed} {
        const log_cut = clipped(msg, FAIL_LOG_CAUSE_MAX);
        self.logf("window {d}: {s} {s} {s}: the GPU worker failed: {s}{s}; the run ends", .{ self.window_id, @tagName(phase), kname(k), clipped(text, 32).s, log_cut.s, log_cut.more });
        const rep_cut = clipped(msg, FAIL_REPORT_CAUSE_MAX);
        self.report("[tune] window {d}: {s} {s} {s}: the GPU worker failed: {s}{s}; the run ends", .{ self.window_id, @tagName(phase), kname(k), clipped(text, 32).s, rep_cut.s, rep_cut.more });
        return error.TuneWorkerFailed;
    }

    fn report(self: *Sched, comptime fmt: []const u8, args: anytype) void {
        var buf: [384]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..0];
        self.cfg.reporter.line(text);
    }

    fn init(alloc: Allocator, cfg: RunConfig, sig: Signals, w: Worker, stop: *std.atomic.Value(bool)) !*Sched {
        const s = try alloc.create(Sched);
        s.* = .{
            .alloc = alloc,
            .cfg = cfg,
            .sig = sig,
            .w = w,
            .user_stop = stop,
            .prog = undefined,
            .log = undefined,
            .gu = kernel_table.gateUpCandidates(),
            .dn = kernel_table.downCandidates(),
            .texts = undefined,
            .ks = .{ .{ .n = kernel_table.GATEUP_CANDIDATES }, .{ .n = kernel_table.DOWN_CANDIDATES } },
        };
        for (0..kernel_table.GATEUP_CANDIDATES) |i| {
            var b: [48]u8 = undefined;
            s.texts[0][i] = try alloc.dupe(u8, kernel_table.gateUpText(&b, s.gu[i]));
        }
        for (0..kernel_table.DOWN_CANDIDATES) |i| {
            var b: [48]u8 = undefined;
            s.texts[1][i] = try alloc.dupe(u8, kernel_table.downText(&b, s.dn[i]));
        }
        const p = try loadProgress(alloc, cfg.io, cfg.root, cfg.key);
        const fresh = p.restarted or !p.had_file;
        s.prog = try AppendFile.open(cfg.root, PROGRESS_NAME, fresh);
        // registered before anything else can fail, so every later error closes the descriptor
        errdefer s.prog.close();
        if (fresh) {
            try s.prog.appendLine(alloc, try headerLine(alloc, cfg.key), true);
            if (p.restarted) s.report("[tune] progress file replaced ({s}): the run starts over", .{p.restart_reason});
        } else if (p.file_len > p.valid_len) try s.prog.truncateTo(p.valid_len);
        s.log = try AppendFile.open(cfg.root, LOG_NAME, false);
        errdefer s.log.close();
        try s.applyProgress(p.recs);
        return s;
    }

    fn deinit(self: *Sched) void {
        self.prog.close();
        self.log.close();
    }

    /// Rebuilds the run's state from the kept lines (spec 4.5): identity results, screen batches, excluded and
    /// not-measurable candidates, window ids and the windows used so far, and a finished final stage (the lines of an
    /// attempt with no marker are void). Every counter saturates, so a hand-edited file cannot overflow one.
    fn applyProgress(self: *Sched, recs: []const Rec) !void {
        var markers: std.AutoHashMapUnmanaged(u32, void) = .empty; // the attempts a marker closed
        var last_marker: u32 = 0;
        // the line of the last attempt's marker, and the candidates' state when it was written
        var marker_at: usize = 0;
        var snap: CandSnapshot = undefined;
        var ever_final: [2]bool = .{ false, false };
        for (recs, 0..) |r, ri| switch (r) {
            .marker => |m| {
                try markers.put(self.alloc, m.attempt, {});
                if (m.attempt != 0 and m.attempt >= last_marker) {
                    last_marker = m.attempt;
                    marker_at = ri;
                }
                self.next_attempt = @max(self.next_attempt, m.attempt +| 1);
            },
            .batch => {},
        };
        var gate_attempts: std.AutoHashMapUnmanaged(u32, void) = .empty;
        for (recs, 0..) |r, ri| {
            if (last_marker != 0 and ri == marker_at) snap = .{
                .cands = .{ self.ks[0].cands, self.ks[1].cands },
                .id_done = .{ self.ks[0].id_done, self.ks[1].id_done },
            };
            switch (r) {
                .marker => {},
                .batch => |b| {
                    self.window_id = @max(self.window_id, b.window);
                    if (b.attempt) |a| self.next_attempt = @max(self.next_attempt, a +| 1);
                    if (b.kept and b.phase != .identity) {
                        self.batches_kept +|= 1;
                        self.calls_total +|= b.calls_compared;
                    }
                    if (!b.kept) if (b.reason) |why| {
                        for (kernel_table.DISCARD_REASONS, 0..) |name, i| {
                            if (std.mem.eql(u8, name, why)) self.discarded[i] +|= 1;
                        }
                    };
                    if (b.phase == .cal) continue;
                    const k: usize = @intFromEnum(b.kernel);
                    if (b.phase == .final or b.phase == .confirm1) ever_final[k] = true;
                    const idx = self.candIndex(k, b.cand) orelse continue;
                    const c = &self.ks[k].cands[idx];
                    if (b.phase == .identity) {
                        if (!b.kept) continue;
                        // a kept identity line says what the pass found: a comparison count, or an exclusion reason
                        if (b.reason != null) {
                            const why = identityExclusion(b) orelse continue;
                            c.identical = false;
                            c.bits_excluded = true;
                            self.ks[k].reason[idx] = why;
                        } else {
                            if (!identityComplete(b)) continue; // an incomplete line: the candidate is checked again
                            c.identical = true;
                            self.calls_total +|= b.calls_compared;
                        }
                        self.ks[k].id_done[idx] = true;
                        continue;
                    }
                    if (!b.kept) if (b.reason) |why| {
                        if (std.mem.eql(u8, why, "gate")) {
                            c.gate_fails +|= 1;
                            if (b.attempt) |a| if (!markers.contains(a)) try gate_attempts.put(self.alloc, a, {});
                        } else if (std.mem.eql(u8, why, BITS_DIFFER_TIMING)) {
                            c.bits_excluded = true;
                            c.screen = .{ null, null }; // its earlier kept batches are void
                            self.ks[k].reason[idx] = BITS_DIFFER_TIMING;
                        }
                        continue;
                    };
                    if (b.kept and b.phase == .screen) if (b.stats) |st| if (b.regime) |rg| {
                        c.screen[@intFromEnum(rg)] = st;
                    };
                },
            }
        }
        self.gate_abandons = gate_attempts.count();
        self.windows_used = try distinctWindows(self.alloc, recs);
        for (0..2) |k| self.ks[k].had_finalists = ever_final[k];
        // a finished final stage: its finalist lines and its confirm1 lines are the marker's attempt's
        if (last_marker != 0) try self.restoreFinal(recs, last_marker, recs[marker_at].marker.window, marker_at, &snap);
        // a candidate excluded from timing keeps no screen result
        for (&self.ks) |*ks| for (0..ks.n) |i| {
            if (ks.cands[i].bits_excluded) ks.cands[i].screen = .{ null, null };
        };
        if (self.gate_abandons >= tune_stats.TUNE_FINAL_GATE_ABANDONS) self.noisyFinal();
    }

    const C2Rec = struct { k: usize, idx: usize, rg: usize, window: u32, stats: BatchStats };

    /// The candidates' state at one line of the replay: what the live run knew when it wrote that line.
    const CandSnapshot = struct { cands: [2][MAX_CANDS]CandState, id_done: [2][MAX_CANDS]bool };

    /// True when the screen of kernel `k` had finished in `snap`, as the live run requires before a final stage: every
    /// candidate identity-checked, and every one not excluded and still measurable screened in both regimes.
    fn screenFinished(snap: *const CandSnapshot, k: usize, n: usize) bool {
        for (0..n) |i| {
            if (!snap.id_done[k][i]) return false;
            const c = snap.cands[k][i];
            if (c.bits_excluded or c.notMeasurable()) continue;
            if (c.screen[0] == null or c.screen[1] == null) return false;
        }
        return true;
    }

    /// Restores the final stage of the attempt a marker closed, re-deriving what the live run decided from the candidates'
    /// state at the marker (`snap`, line `marker_at`). The marker is trusted only when the records bear out that decision:
    /// the screen had finished for both kernels; every finalist the screen selects has one kept final batch in each regime
    /// in the marker's attempt and window, and no other candidate has one; the pick, recomputed with the live rule over
    /// those batches, has its confirm1 pair there, and no other candidate has a confirm1 line; no line of the attempt is
    /// discarded or comes after the marker. Otherwise the final stage runs again as a new attempt.
    fn restoreFinal(self: *Sched, recs: []const Rec, attempt: u32, window: u32, marker_at: usize, snap: *const CandSnapshot) !void {
        var fin: [2][3]usize = undefined;
        var nfin: [2]usize = .{ 0, 0 };
        for (0..2) |k| {
            const n = self.ks[k].n;
            if (!screenFinished(snap, k, n)) return;
            nfin[k] = tune_stats.finalists(snap.cands[k][0..n], &fin[k]);
        }
        var fs: [2][3][2]?BatchStats = @splat(@splat(.{ null, null }));
        var c1: [2][MAX_CANDS][2]?BatchStats = @splat(@splat(.{ null, null }));
        var c2s: std.ArrayList(C2Rec) = .empty;
        for (recs, 0..) |r, ri| switch (r) {
            .marker => {},
            .batch => |b| {
                const k: usize = @intFromEnum(b.kernel);
                const of_attempt = (b.phase == .final or b.phase == .confirm1) and b.attempt != null and b.attempt.? == attempt;
                if (of_attempt) {
                    if (ri > marker_at or !b.kept or b.window != window) return;
                    const st = b.stats orelse return;
                    const rg: usize = @intFromEnum(b.regime orelse return);
                    const idx = self.candIndex(k, b.cand) orelse return;
                    const slot: *?BatchStats = if (b.phase == .final) blk: {
                        const pos = std.mem.indexOfScalar(usize, fin[k][0..nfin[k]], idx) orelse return; // not a finalist
                        break :blk &fs[k][pos][rg];
                    } else &c1[k][idx][rg];
                    if (slot.* != null) return; // the live run times each once
                    slot.* = st;
                } else if (b.phase == .confirm2 and b.kept) {
                    const st = b.stats orelse continue;
                    const rgv = b.regime orelse continue;
                    const idx = self.candIndex(k, b.cand) orelse continue;
                    try c2s.append(self.alloc, .{ .k = k, .idx = idx, .rg = @intFromEnum(rgv), .window = b.window, .stats = st });
                }
            },
        };
        var pick: [2]usize = .{ 0, 0 };
        for (0..2) |k| {
            if (nfin[k] > 0) {
                var by: [3][2]BatchStats = undefined;
                for (0..nfin[k]) |pos| by[pos] = .{ fs[k][pos][0] orelse return, fs[k][pos][1] orelse return };
                pick[k] = fin[k][tune_stats.pickOfAttempt(by[0..nfin[k]])];
            }
            // confirm1 timed the pick alone, in both regimes; a kernel with no finalist has none
            for (0..self.ks[k].n) |i| {
                const want = nfin[k] > 0 and i == pick[k];
                const both = c1[k][i][0] != null and c1[k][i][1] != null;
                const any = c1[k][i][0] != null or c1[k][i][1] != null;
                if (want and !both) return;
                if (!want and any) return;
            }
        }
        self.final_done = true;
        self.final_window = window;
        for (0..2) |k| {
            const ks = &self.ks[k];
            if (nfin[k] == 0) {
                // no finalist at the finished attempt: an earlier attempt's finalists were all lost, or there never were any
                ks.served = if (ks.had_finalists) .no_measurable_finalist else tune_stats.noFinalistReason(snap.cands[k][0..ks.n], true);
                continue;
            }
            ks.had_finalists = true;
            ks.pick = pick[k];
            ks.c1 = .{ c1[k][pick[k]][0].?, c1[k][pick[k]][1].? };
            ks.c1_window = window;
            switch (tune_stats.judgeConfirm(.{ ks.c1[0].?, ks.c1[1].? })) {
                .not_confirmed => ks.served = .confirm1_not_confirmed,
                .below_floor => ks.served = .below_floor,
                .ok => {
                    ks.awaiting = true;
                    const pc = &ks.cands[ks.pick];
                    if (pc.bits_excluded) {
                        ks.served = .pick_bits_differ; // recorded in confirm2 before the run stopped
                    } else if (pc.notMeasurable()) {
                        ks.served = .confirm2_not_measurable;
                    } else {
                        // only the pick's own batches from a window after the final stage's count
                        for (c2s.items) |c| {
                            if (c.k == k and c.idx == ks.pick and c.window > window) {
                                ks.c2[c.rg] = c.stats;
                                ks.c2_window[c.rg] = c.window;
                            }
                        }
                        if (ks.c2[0] != null and ks.c2[1] != null) self.judgeConfirm2(k);
                    }
                },
            }
        }
    }

    fn allDecided(self: *const Sched) bool {
        return self.ks[0].decided() and self.ks[1].decided();
    }

    fn noisyFinal(self: *Sched) void {
        for (&self.ks) |*ks| {
            if (!ks.decided() and ks.had_finalists and !ks.awaiting) ks.served = .final_stage_noisy;
        }
        self.final_done = true;
    }

    // ── writing ──

    fn writeBatch(self: *Sched, phase: Phase, k: usize, cand_text: []const u8, regime: ?usize, kept: bool, reason: ?[]const u8, stats: ?BatchStats, calls: u64, attempt: ?u32) !void {
        var l: std.ArrayList(u8) = .empty;
        defer l.deinit(self.alloc);
        try l.print(self.alloc, "{{\"phase\":\"{s}\",\"kernel\":\"{s}\",\"cand\":", .{ @tagName(phase), kname(k) });
        try jsonStr(&l, self.alloc, cand_text);
        if (regime) |r| {
            try l.print(self.alloc, ",\"regime\":\"{s}\"", .{@tagName(@as(Regime, @enumFromInt(r)))});
        } else try l.appendSlice(self.alloc, ",\"regime\":null");
        try l.print(self.alloc, ",\"window\":{d},\"kept\":{}", .{ self.window_id, kept });
        if (reason) |why| {
            try l.appendSlice(self.alloc, ",\"reason\":");
            try jsonStr(&l, self.alloc, why);
        } else try l.appendSlice(self.alloc, ",\"reason\":null");
        if (stats) |st| {
            try l.print(self.alloc, ",\"ref_ns\":{d},\"cand_ns\":{d},\"diff_ns\":{d},\"q1_ns\":{d},\"q3_ns\":{d},\"lo_ns\":{d},\"hi_ns\":{d},\"calls_compared\":{d}", .{ st.ref_ns, st.cand_ns, st.diff_ns, st.q1_ns, st.q3_ns, st.lo_ns, st.hi_ns, calls });
        } else if (calls > 0) {
            // an identity line: no timing fields, but its comparisons survive a restart
            try l.print(self.alloc, ",\"calls_compared\":{d}", .{calls});
        }
        if (attempt) |a| try l.print(self.alloc, ",\"attempt\":{d}", .{a});
        try l.append(self.alloc, '}');
        try self.prog.appendLine(self.alloc, l.items, true);
        const rname = if (regime) |r| @tagName(@as(Regime, @enumFromInt(r))) else "-";
        self.logf("{s} {s} {s} {s} w{d} {s} ref={d} cand={d} diff={d}", .{
            @tagName(phase),                kname(k),                         cand_text,
            rname,                          self.window_id,                   if (kept) "kept" else (reason orelse "discarded"),
            if (stats) |s| s.ref_ns else 0, if (stats) |s| s.cand_ns else 0, if (stats) |s| s.diff_ns else 0,
        });
    }

    fn writeMarker(self: *Sched, attempt: u32) !void {
        var buf: [96]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "{{\"stage_done\":\"final\",\"attempt\":{d},\"window\":{d}}}", .{ attempt, self.window_id });
        try self.prog.appendLine(self.alloc, text, true);
        self.logf("decision final stage attempt {d} finished in window {d}", .{ attempt, self.window_id });
    }

    // ── windows ──

    fn waitForWindow(self: *Sched) !void {
        while (true) {
            if (self.user_stop.load(.acquire)) return error.UserStop;
            const chk = tune_signals.checkOpen(self.sig, self.cfg.idle_minutes);
            if (chk.allHold()) return;
            var buf: [320]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            w.writeAll("[tune] waiting: ") catch {};
            if (self.sig.hidIdleNs()) |ns| {
                const s = ns / NS_PER_S;
                w.print("no input for {d}m{d:0>2}s of {d}m", .{ s / 60, s % 60, self.cfg.idle_minutes }) catch {};
            } else w.writeAll("input unreadable") catch {};
            w.writeAll(switch (chk.ac) {
                .holds => ", on AC",
                .fails => ", on battery",
                .unknown => ", power source unreadable",
            }) catch {};
            if (chk.low_power != .holds) w.writeAll(", Low Power Mode") catch {};
            if (chk.thermal != .holds) w.writeAll(", thermal pressure") catch {};
            if (chk.gpu == .fails) w.writeAll(", GPU busy") catch {};
            if (chk.cpu == .fails) w.writeAll(", CPU busy") catch {};
            if (chk.mem == .fails) w.writeAll(", memory pressure") catch {};
            self.cfg.reporter.line(w.buffered());
            self.sig.waitNs(TUNE_WAIT_POLL_S * NS_PER_S);
        }
    }

    fn openWin(self: *Sched) !void {
        self.window_id +|= 1;
        self.windows_used +|= 1;
        self.window_discards = 0;
        self.cal_ref = .{ .{ 0, 0 }, .{ 0, 0 } };
        self.win_stop.store(false, .release);
        try self.w.openWindow();
        self.logf("window {d} open", .{self.window_id});
    }

    fn closeWin(self: *Sched) void {
        self.w.closeWindow();
        if (self.pending_stop) |*st| {
            st.freed_ns = self.sig.absTimeNs();
            const latency = (st.freed_ns -| st.input_ns) / NS_PER_MS;
            self.logf("stop input={d} flag={d} abandoned={d} freed={d} latency_ms={d}", .{ st.input_ns, st.flag_ns, st.abandoned_ns, st.freed_ns, latency });
            self.pending_stop = null;
        }
        self.logf("window {d} closed", .{self.window_id});
    }

    // ── one timed batch ──

    fn timedBatch(self: *Sched, phase: Phase, k: usize, cand: Cand, idx: usize, regime: usize, attempt: ?u32) !TimedResult {
        if (self.user_stop.load(.acquire)) return error.UserStop;
        const mem = self.sig.memPressure();
        if (mem >= 2) {
            self.report("[tune] window {d}: memory pressure, the window closes", .{self.window_id});
            return .closed;
        }
        self.win_stop.store(false, .release);
        self.boot_counter += 1;
        const text = if (cand == .reference) "reference" else self.texts[k][idx];
        const res = runBatchWith(self.sig, self.w, @enumFromInt(k), cand, @enumFromInt(regime), self.cfg.boot_seed_base +% self.boot_counter, &self.win_stop, self.user_stop, self.cfg.monitor);
        self.gpu_ns += res.duration_ns;
        const stats: ?BatchStats = switch (res.outcome) {
            .done => |d| d.stats,
            .stopped => null,
            .failed => |msg| return self.workerFailed(phase, k, text, msg),
        };
        // the bits verdict outranks every timing discard: a completed batch whose bits differ excludes its candidate (or
        // voids an A/A batch) whatever the monitor or the after-batch reading says; the window still closes on a discard
        if (res.outcome == .done and res.outcome.done.bits_differ) {
            const done = res.outcome.done;
            const counted = phase == .cal or res.discard != null; // a candidate's exclusion alone keeps today's counts
            if (counted) self.window_discards += 1;
            if (phase != .cal) {
                const c = &self.ks[k].cands[idx];
                c.bits_excluded = true;
                self.ks[k].reason[idx] = BITS_DIFFER_TIMING;
                c.screen = .{ null, null };
            }
            try self.writeBatch(phase, k, text, regime, false, BITS_DIFFER_TIMING, done.stats, done.calls_compared, attempt);
            if (counted) self.reportWindow(phase, k, "discarded ({s})", .{BITS_DIFFER_TIMING});
            if (res.discard != null) {
                self.pending_stop = res.stop_times;
                return .bits_differ_closed;
            }
            return .bits_differ;
        }
        if (res.discard) |d| {
            self.discarded[@intFromEnum(d)] += 1;
            self.window_discards += 1;
            const calls: u64 = if (res.outcome == .done) res.outcome.done.calls_compared else 0;
            try self.writeBatch(phase, k, text, regime, false, @tagName(d), stats, calls, attempt);
            if (res.user_stop and self.user_stop.load(.acquire)) return error.UserStop;
            self.pending_stop = res.stop_times;
            self.reportWindow(phase, k, "discarded ({s})", .{@tagName(d)});
            return .closed;
        }
        const done = res.outcome.done;
        const c = &self.ks[k].cands[idx];
        if (phase != .cal and !tune_stats.gatePasses(done.stats, self.cal_ref[k][regime])) {
            c.gate_fails +|= 1;
            self.discarded[@intFromEnum(DiscardReason.gate)] += 1;
            try self.writeBatch(phase, k, text, regime, false, "gate", done.stats, done.calls_compared, attempt);
            return .gate;
        }
        self.batches_kept += 1;
        self.calls_total +|= done.calls_compared;
        self.last = done.stats;
        self.last_calls = done.calls_compared;
        try self.writeBatch(phase, k, text, regime, true, null, done.stats, done.calls_compared, attempt);
        self.reportWindow(phase, k, "kept", .{});
        return .kept;
    }

    fn reportWindow(self: *Sched, phase: Phase, k: usize, comptime what: []const u8, args: anytype) void {
        var buf: [160]u8 = undefined;
        const tail = std.fmt.bufPrint(&buf, what, args) catch "";
        var screened: usize = 0;
        for (0..self.ks[k].n) |i| {
            if (self.ks[k].cands[i].screen[0] != null and self.ks[k].cands[i].screen[1] != null) screened += 1;
        }
        self.report("[tune] window {d}: {s} {s} {d} of {d} candidates, {d} batches discarded, {d} s of GPU so far; last batch {s}", .{
            self.window_id, @tagName(phase), kname(k), screened, self.ks[k].n, self.window_discards, self.gpu_ns / NS_PER_S, tail,
        });
    }

    // ── identity ──

    const IdResult = enum { done, closed };

    fn identityPass(self: *Sched) !IdResult {
        for (0..2) |k| {
            for (0..self.ks[k].n) |idx| {
                if (self.ks[k].id_done[idx]) continue;
                if (self.user_stop.load(.acquire)) return error.UserStop;
                self.win_stop.store(false, .release);
                var mon: Monitor = .{ .sig = self.sig, .mode = self.cfg.monitor, .win_stop = &self.win_stop, .user_stop = self.user_stop };
                mon.start() catch {};
                const out = self.w.identity(@enumFromInt(k), self.candOf(k, idx), &self.win_stop, mon.hook());
                const abandoned = self.sig.absTimeNs();
                mon.join();
                mon.tick(); // a last reading: input or a power change after the worker's last look at the flag
                const c = &self.ks[k].cands[idx];
                switch (out) {
                    .identical => |i| {
                        c.identical = true;
                        self.ks[k].id_done[idx] = true;
                        self.calls_total +|= i.calls_compared;
                        try self.writeBatch(.identity, k, self.texts[k][idx], null, true, null, null, i.calls_compared, null);
                    },
                    .differs => |case| {
                        c.identical = false;
                        c.bits_excluded = true;
                        self.ks[k].id_done[idx] = true;
                        const why = try std.fmt.allocPrint(self.alloc, "bits differ in case {d}", .{case});
                        self.ks[k].reason[idx] = why;
                        try self.writeBatch(.identity, k, self.texts[k][idx], null, true, why, null, 0, null);
                    },
                    .refused => |msg| {
                        c.identical = false;
                        c.bits_excluded = true;
                        self.ks[k].id_done[idx] = true;
                        const why = try std.fmt.allocPrint(self.alloc, "launch refused: {s}", .{msg});
                        self.ks[k].reason[idx] = why;
                        try self.writeBatch(.identity, k, self.texts[k][idx], null, true, why, null, 0, null);
                    },
                    .stopped => {
                        const user = self.user_stop.load(.acquire) and mon.failure() == null;
                        const why: DiscardReason = if (mon.failure()) |f| discardOfFailure(f) else .stopped;
                        self.discarded[@intFromEnum(why)] += 1;
                        try self.writeBatch(.identity, k, self.texts[k][idx], null, false, @tagName(why), null, 0, null);
                        if (user or self.user_stop.load(.acquire)) return error.UserStop;
                        if (mon.failure() != null) self.pending_stop = .{ .input_ns = mon.input_ns, .flag_ns = mon.flag_ns, .abandoned_ns = abandoned, .freed_ns = abandoned };
                        return .closed;
                    },
                    .failed => |msg| return self.workerFailed(.identity, k, self.texts[k][idx], msg),
                }
                // the result of a completed check stands (it does not depend on the Mac being idle), but the window ends
                if (mon.failure() != null) {
                    self.pending_stop = .{ .input_ns = mon.input_ns, .flag_ns = mon.flag_ns, .abandoned_ns = abandoned, .freed_ns = abandoned };
                    return .closed;
                }
            }
        }
        return .done;
    }

    // ── calibration ──

    const CalResult = enum { ok, closed };

    fn calibrate(self: *Sched) !CalResult {
        var round: u32 = 0;
        while (round < tune_stats.TUNE_CAL_ROUNDS) : (round += 1) {
            var res: [2][2]BatchStats = undefined;
            for (0..2) |k| for (0..2) |rg| {
                switch (try self.timedBatch(.cal, k, .reference, 0, rg, null)) {
                    .kept => res[k][rg] = self.last,
                    else => return .closed,
                }
            };
            if (tune_stats.calRoundPasses(res)) {
                for (0..2) |k| for (0..2) |rg| {
                    self.cal_ref[k][rg] = res[k][rg].ref_ns;
                };
                self.logf("window {d} calibrated: refs {d} {d} {d} {d}", .{ self.window_id, self.cal_ref[0][0], self.cal_ref[0][1], self.cal_ref[1][0], self.cal_ref[1][1] });
                return .ok;
            }
            if (round + 1 < tune_stats.TUNE_CAL_ROUNDS) {
                self.sig.waitNs(@as(u64, tune_stats.TUNE_CAL_RETRY_S) * NS_PER_S);
                if (!tune_signals.checkOpen(self.sig, self.cfg.idle_minutes).allHold()) return .closed;
            }
        }
        self.logf("window {d} closed as noisy", .{self.window_id});
        return .closed;
    }

    // ── screen ──

    fn buildQueue(self: *Sched) !void {
        self.queue.clearRetainingCapacity();
        self.qpos = 0;
        for (0..2) |k| for (0..self.ks[k].n) |i| for (0..2) |rg| {
            const c = self.ks[k].cands[i];
            if (c.bits_excluded or c.notMeasurable() or c.screen[rg] != null) continue;
            try self.queue.append(self.alloc, .{ .k = @intCast(k), .idx = @intCast(i), .regime = @intCast(rg) });
        };
        self.queue_built = true;
    }

    fn screenPhase(self: *Sched) !enum { done, closed } {
        if (!self.queue_built) try self.buildQueue();
        while (self.qpos < self.queue.items.len) {
            const it = self.queue.items[self.qpos];
            const c = &self.ks[it.k].cands[it.idx];
            if (c.bits_excluded or c.notMeasurable() or c.screen[it.regime] != null) {
                self.qpos += 1;
                continue;
            }
            switch (try self.timedBatch(.screen, it.k, self.candOf(it.k, it.idx), it.idx, it.regime, null)) {
                .kept => {
                    c.screen[it.regime] = self.last;
                    self.qpos += 1;
                },
                .gate => {
                    self.qpos += 1;
                    if (!c.notMeasurable()) try self.queue.append(self.alloc, it); // re-queued at the end
                },
                .bits_differ => self.qpos += 1,
                .bits_differ_closed => {
                    self.qpos += 1;
                    return .closed;
                },
                .closed => return .closed,
            }
        }
        return .done;
    }

    // ── the final stage ──

    fn finalAttempt(self: *Sched) !AttemptResult {
        const attempt = self.next_attempt;
        self.next_attempt +|= 1;
        var fin: [2][3]usize = undefined;
        var nfin: [2]usize = .{ 0, 0 };
        for (0..2) |k| {
            const ks = &self.ks[k];
            if (ks.decided()) continue;
            nfin[k] = tune_stats.finalists(ks.cands[0..ks.n], &fin[k]);
            if (nfin[k] == 0) {
                ks.served = if (ks.had_finalists) .no_measurable_finalist else tune_stats.noFinalistReason(ks.cands[0..ks.n], true);
                self.logf("decision {s} served: {s}", .{ kname(k), @tagName(ks.served.?) });
            } else ks.had_finalists = true;
        }
        if (nfin[0] == 0 and nfin[1] == 0) {
            self.final_done = true;
            return .finished;
        }
        var fs: [2][3][2]BatchStats = undefined;
        for (0..2) |k| for (0..nfin[k]) |fi| for (0..2) |rg| {
            switch (try self.timedBatch(.final, k, self.candOf(k, fin[k][fi]), fin[k][fi], rg, attempt)) {
                .kept => fs[k][fi][rg] = self.last,
                .gate => return self.abandon(true),
                .bits_differ => return self.abandon(false),
                .bits_differ_closed => {
                    _ = self.abandon(false);
                    return .closed;
                },
                .closed => return .closed,
            }
        };
        var c1: [2][2]BatchStats = undefined;
        var pick_at: [2]usize = .{ 0, 0 };
        for (0..2) |k| {
            if (nfin[k] == 0) continue;
            const fi = tune_stats.pickOfAttempt(fs[k][0..nfin[k]]);
            pick_at[k] = fin[k][fi];
            for (0..2) |rg| {
                switch (try self.timedBatch(.confirm1, k, self.candOf(k, pick_at[k]), pick_at[k], rg, attempt)) {
                    .kept => c1[k][rg] = self.last,
                    .gate => return self.abandon(true),
                    .bits_differ => return self.abandon(false),
                    .bits_differ_closed => {
                        _ = self.abandon(false);
                        return .closed;
                    },
                    .closed => return .closed,
                }
            }
        }
        try self.writeMarker(attempt);
        self.final_done = true;
        self.final_window = self.window_id;
        for (0..2) |k| {
            if (nfin[k] == 0) continue;
            const ks = &self.ks[k];
            ks.pick = pick_at[k];
            ks.c1 = .{ c1[k][0], c1[k][1] };
            ks.c1_window = self.window_id;
            switch (tune_stats.judgeConfirm(c1[k])) {
                .ok => ks.awaiting = true,
                .not_confirmed => ks.served = .confirm1_not_confirmed,
                .below_floor => ks.served = .below_floor,
            }
            if (ks.served) |sv| self.logf("decision {s} served: {s}", .{ kname(k), @tagName(sv) });
        }
        return .finished;
    }

    /// An attempt abandoned by the gate counts toward the cap; any other abandonment (a bits difference) does not.
    fn abandon(self: *Sched, by_gate: bool) AttemptResult {
        if (by_gate) {
            self.gate_abandons += 1;
            self.logf("final stage attempt abandoned by the gate ({d} of {d})", .{ self.gate_abandons, tune_stats.TUNE_FINAL_GATE_ABANDONS });
            if (self.gate_abandons >= tune_stats.TUNE_FINAL_GATE_ABANDONS) self.noisyFinal();
        }
        return .abandoned;
    }

    // ── confirm2 ──

    fn confirm2Phase(self: *Sched) !enum { done, closed } {
        var q: std.ArrayList(Item) = .empty;
        defer q.deinit(self.alloc);
        for (0..2) |k| {
            const ks = &self.ks[k];
            if (!ks.awaiting or ks.decided()) continue;
            for (0..2) |rg| {
                if (ks.c2[rg] == null) try q.append(self.alloc, .{ .k = @intCast(k), .idx = @intCast(ks.pick), .regime = @intCast(rg) });
            }
        }
        var pos: usize = 0;
        while (pos < q.items.len) {
            const it = q.items[pos];
            const ks = &self.ks[it.k];
            if (ks.decided()) {
                pos += 1;
                continue;
            }
            switch (try self.timedBatch(.confirm2, it.k, self.candOf(it.k, it.idx), it.idx, it.regime, null)) {
                .kept => {
                    ks.c2[it.regime] = self.last;
                    ks.c2_window[it.regime] = self.window_id;
                    pos += 1;
                    if (ks.c2[0] != null and ks.c2[1] != null) self.judgeConfirm2(it.k);
                },
                .gate => {
                    pos += 1;
                    if (ks.cands[it.idx].notMeasurable()) {
                        ks.served = .confirm2_not_measurable;
                        self.logf("decision {s} served: confirm2_not_measurable", .{kname(it.k)});
                    } else try q.append(self.alloc, it);
                },
                .bits_differ => {
                    pos += 1;
                    ks.served = .pick_bits_differ;
                    self.logf("decision {s} served: pick_bits_differ", .{kname(it.k)});
                },
                .bits_differ_closed => {
                    ks.served = .pick_bits_differ;
                    self.logf("decision {s} served: pick_bits_differ", .{kname(it.k)});
                    return .closed;
                },
                .closed => return .closed,
            }
        }
        return .done;
    }

    fn judgeConfirm2(self: *Sched, k: usize) void {
        const ks = &self.ks[k];
        const c2: [2]BatchStats = .{ ks.c2[0].?, ks.c2[1].? };
        switch (tune_stats.judgeConfirm(c2)) {
            .ok => {
                ks.pick_final = ks.pick;
                const c1 = [2]BatchStats{ ks.c1[0].?, ks.c1[1].? };
                const worse = if (tune_stats.share(c1[0].diff_ns, c1[0].ref_ns) >= tune_stats.share(c1[1].diff_ns, c1[1].ref_ns)) c1[0] else c1[1];
                const cs = [4]BatchStats{ c1[0], c1[1], c2[0], c2[1] };
                const wins = [4]u32{ ks.c1_window, ks.c1_window, ks.c2_window[0], ks.c2_window[1] };
                var ev: kernel_table.KernelEvidence = .{ .ref_ns = worse.ref_ns, .pick_ns = worse.cand_ns, .confirm = undefined };
                for (cs, wins, 0..) |s, wn, i| ev.confirm[i] = .{ .median_ns = s.diff_ns, .lo_ns = s.lo_ns, .hi_ns = s.hi_ns, .window = wn };
                ks.evidence = ev;
                self.logf("decision {s} pick {s}", .{ kname(k), self.texts[k][ks.pick] });
            },
            .not_confirmed => ks.served = .confirm2_not_confirmed,
            .below_floor => ks.served = .below_floor,
        }
        if (ks.served) |sv| self.logf("decision {s} served: {s}", .{ kname(k), @tagName(sv) });
    }

    // ── one window, and the run ──

    const WinResult = enum { closed, finished };

    fn windowWork(self: *Sched) !WinResult {
        if (try self.identityPass() == .closed) return .closed;
        if (self.allDecided()) return .finished;
        if (try self.calibrate() == .closed) return .closed;
        while (true) {
            if (self.allDecided()) return .finished;
            if (!self.queue_built or self.qpos < self.queue.items.len) {
                if (try self.screenPhase() == .closed) return .closed;
                continue;
            }
            if (!self.final_done) {
                switch (try self.finalAttempt()) {
                    .closed => return .closed,
                    .finished, .abandoned => continue,
                }
            }
            // confirm2 never runs in the window that finished the final stage
            if (self.window_id == self.final_window) return .closed;
            if (try self.confirm2Phase() == .closed) return .closed;
        }
    }

    fn drive(self: *Sched) !void {
        while (!self.allDecided()) {
            try self.waitForWindow();
            try self.openWin();
            const r = self.windowWork() catch |e| {
                self.closeWin();
                return e;
            };
            self.closeWin();
            _ = r;
        }
    }

    fn outcome(self: *Sched) !Outcome {
        var plan: MoeRowsLayoutPlan = .{ .gateup = null, .down = null };
        if (self.ks[0].pick_final) |i| plan.gateup = self.gu[i];
        if (self.ks[1].pick_final) |i| plan.down = self.dn[i];
        var excluded: std.ArrayList(kernel_table.Excluded) = .empty;
        var nm: std.ArrayList(kernel_table.NotMeasurable) = .empty;
        var checked: u32 = 0;
        for (0..2) |k| for (0..self.ks[k].n) |i| {
            if (self.ks[k].id_done[i]) checked += 1;
            if (self.ks[k].reason[i]) |why| try excluded.append(self.alloc, .{ .kernel = kname(k), .cand = self.texts[k][i], .reason = why });
            if (self.ks[k].cands[i].notMeasurable()) try nm.append(self.alloc, .{ .kernel = kname(k), .cand = self.texts[k][i] });
        };
        return .{
            .plan = plan,
            .served_reasons = .{ self.ks[0].served, self.ks[1].served },
            .evidence = .{ self.ks[0].evidence, self.ks[1].evidence },
            .identity = .{ .plans_checked = checked, .cases = IDENTITY_CASES, .calls_compared = self.calls_total, .excluded = try excluded.toOwnedSlice(self.alloc) },
            .not_measurable = try nm.toOwnedSlice(self.alloc),
            .windows_used = self.windows_used,
            .batches_kept = self.batches_kept,
            .discarded = self.discarded,
        };
    }
};

/// The whole state machine of spec 4.3 to 6.2. `stop` is the external flag (SIGINT, SIGTERM): the monitor copies it into
/// the flag the worker watches, and a worker that returns `stopped` with `stop` set ends the run as `.stopped`; a monitor
/// stop only closes the window, and the run waits for the next one. Allocations go to `alloc` (pass an arena).
pub fn run(alloc: Allocator, cfg: RunConfig, sig: Signals, w: Worker, stop: *std.atomic.Value(bool)) !SchedResult {
    const s = try Sched.init(alloc, cfg, sig, w, stop);
    defer s.deinit();
    s.drive() catch |e| switch (e) {
        error.UserStop => return .stopped,
        else => return e,
    };
    return .{ .finished = try s.outcome() };
}

pub const VerifyResult = union(enum) {
    identical: struct { kernels: u8, calls_compared: u64 },
    failed: struct { kernel: Kernel, reason: []const u8 },
    stopped,
};

/// Identity only (spec 7.2 `--verify`): waits for an idle window and checks the layouts of `cfg.verify_plan`, one worker
/// call each. A monitor stop closes the window and the check waits for the next one.
pub fn verify(alloc: Allocator, cfg: RunConfig, sig: Signals, w: Worker, stop: *std.atomic.Value(bool)) !VerifyResult {
    var win_stop: std.atomic.Value(bool) = .init(false);
    var todo: [2]?Cand = .{
        if (cfg.verify_plan.gateup) |g| Cand{ .gateup = g } else null,
        if (cfg.verify_plan.down) |d| Cand{ .down = d } else null,
    };
    var kernels: u8 = 0;
    var calls: u64 = 0;
    while (true) {
        var left = false;
        for (todo) |t| left = left or t != null;
        if (!left) return .{ .identical = .{ .kernels = kernels, .calls_compared = calls } };
        // wait for an idle window, then open it
        while (true) {
            if (stop.load(.acquire)) return .stopped;
            if (tune_signals.checkOpen(sig, cfg.idle_minutes).allHold()) break;
            sig.waitNs(TUNE_WAIT_POLL_S * NS_PER_S);
        }
        try w.openWindow();
        defer w.closeWindow();
        for (0..2) |k| {
            const c = todo[k] orelse continue;
            win_stop.store(false, .release);
            var mon: Monitor = .{ .sig = sig, .mode = cfg.monitor, .win_stop = &win_stop, .user_stop = stop };
            mon.start() catch {};
            const out = w.identity(@enumFromInt(k), c, &win_stop, mon.hook());
            mon.join();
            mon.tick();
            switch (out) {
                .identical => |i| {
                    calls += i.calls_compared;
                    kernels += 1;
                    todo[k] = null;
                },
                .differs => |case| return .{ .failed = .{ .kernel = @enumFromInt(k), .reason = try std.fmt.allocPrint(alloc, "bits differ in case {d}", .{case}) } },
                .refused => |msg| return .{ .failed = .{ .kernel = @enumFromInt(k), .reason = try std.fmt.allocPrint(alloc, "launch refused: {s}", .{msg}) } },
                .failed => |msg| {
                    // not the layout's verdict: the check could not run, and the verify ends with the message
                    var buf: [384]u8 = undefined;
                    const cut = clipped(msg, FAIL_REPORT_CAUSE_MAX);
                    cfg.reporter.line(std.fmt.bufPrint(&buf, "[tune] verify {s}: the GPU worker failed: {s}{s}", .{ Sched.kname(k), cut.s, cut.more }) catch msg);
                    return error.TuneWorkerFailed;
                },
                .stopped => {
                    if (stop.load(.acquire)) return .stopped;
                    break; // the window closes; the unfinished layouts wait for the next one
                },
            }
            if (mon.failure() != null) break; // the Mac is in use again: the window closes, the rest waits for the next one
        }
    }
}

// ── Tests ──

const testing = std.testing;
const ScriptedSignals = tune_signals.ScriptedSignals;

const TEST_KEY = kernel_table.Key{ .gpu_arch = "applegpu_g16g", .gpu_cores = 20, .hw_model = "Mac16,11", .os_build = "25B78", .kernel_hash = "aa11", .mlx_version = "0.31.0" };
const IDLE_SCRIPT = "{\"at_ms\":0,\"hid_idle_s\":600}\n";

/// What a fake batch looks like to the scheduler: percent of the reference times 100 (-500 is a 5% gain), the reference
/// median, the paired-difference IQR, the interval's upper end, and what the worker does.
const Spec = struct {
    diff_pct100: i64 = 100,
    ref: ?i64 = null,
    iqr_pct100: i64 = 20,
    hi: ?i64 = null,
    bits: bool = false,
    /// Return `.stopped` without the external flag (a window-closing abandonment).
    stop: bool = false,
    /// Set the external stop flag and return `.stopped` (SIGINT).
    user_stop: bool = false,
    /// Return `.failed` (an engine error that is not a stop).
    fail: bool = false,
};

const SpecIn = struct {
    kernel: Kernel,
    /// The candidate's index in the plan space, null for the reference (an A/A batch).
    idx: ?usize,
    regime: Regime,
    /// The call's number among those of the same kernel, candidate (or reference) and regime, counted from 1.
    nth: u32,
    /// The call's number among all batches, counted from 1.
    call: u32,
};

fn baseSpec(in: SpecIn) Spec {
    return if (in.idx == null) .{ .diff_pct100 = 0 } else .{};
}

const SpecFn = *const fn (SpecIn) Spec;
const IdFn = *const fn (Kernel, usize, u32) IdentityOutcome;

fn idAllIdentical(_: Kernel, _: usize, _: u32) IdentityOutcome {
    return .{ .identical = .{ .calls_compared = 10 } };
}

const CallRec = struct { identity: bool, kernel: Kernel, idx: ?usize, regime: Regime };

const FakeWorker = struct {
    sc: *ScriptedSignals,
    spec_fn: SpecFn = baseSpec,
    id_fn: IdFn = idAllIdentical,
    ext_stop: *std.atomic.Value(bool),
    gu: [kernel_table.GATEUP_CANDIDATES]MoeRowsLayoutGateUp,
    dn: [kernel_table.DOWN_CANDIDATES]MoeRowsLayoutDown,
    nth: [2][kernel_table.DOWN_CANDIDATES + 1][2]u32 = @splat(@splat(.{ 0, 0 })),
    calls: std.ArrayList(CallRec) = .empty,
    alloc: Allocator,
    opens: u32 = 0,
    closes: u32 = 0,
    first_open_ms: u64 = 0,
    identity_calls: u32 = 0,
    batch_calls: u32 = 0,
    /// Set the external stop on the nth identity call (1-based; 0 never).
    stop_at_identity: u32 = 0,
    /// Set the external stop when the nth window opens (0 never).
    stop_at_open: u32 = 0,
    pair_ns: u64 = 14 * NS_PER_MS,
    /// Fake-clock time of one identity case (0: an identity call takes no time).
    id_case_ms: u64 = 0,
    /// Identity cases that ran to the end, over all calls.
    id_cases: u32 = 0,

    fn idxOf(self: *const FakeWorker, k: Kernel, c: Cand) ?usize {
        switch (c) {
            .reference => return null,
            .gateup => |g| {
                for (self.gu, 0..) |x, i| if (std.meta.eql(x, g)) return i;
            },
            .down => |d| {
                for (self.dn, 0..) |x, i| if (std.meta.eql(x, d)) return i;
            },
        }
        _ = k;
        return null;
    }

    fn at(p: *anyopaque) *FakeWorker {
        return @ptrCast(@alignCast(p));
    }

    fn openWindow(p: *anyopaque) anyerror!void {
        const self = at(p);
        self.opens += 1;
        if (self.opens == 1) self.first_open_ms = self.sc.now_ms;
        if (self.stop_at_open != 0 and self.opens >= self.stop_at_open) self.ext_stop.store(true, .release);
    }

    fn closeWindow(p: *anyopaque) void {
        at(p).closes += 1;
    }

    fn identity(p: *anyopaque, k: Kernel, c: Cand, stop: *const std.atomic.Value(bool), hook: PairHook) IdentityOutcome {
        const self = at(p);
        self.identity_calls += 1;
        const idx = self.idxOf(k, c).?;
        self.calls.append(self.alloc, .{ .identity = true, .kernel = k, .idx = idx, .regime = .indep }) catch {};
        if (self.stop_at_identity != 0 and self.identity_calls == self.stop_at_identity) {
            self.ext_stop.store(true, .release);
            return .stopped;
        }
        // the ten cases, each `id_case_ms` of the fake clock, the hook after each and then the stop flag
        for (0..IDENTITY_CASES) |_| {
            self.sc.advanceMs(self.id_case_ms);
            hook.call();
            if (stop.load(.acquire)) return .stopped;
            self.id_cases += 1;
        }
        return self.id_fn(k, idx, self.identity_calls);
    }

    fn timeBatch(p: *anyopaque, k: Kernel, c: Cand, r: Regime, seed: u64, stop: *const std.atomic.Value(bool), hook: PairHook) BatchOutcome {
        _ = seed;
        const self = at(p);
        self.batch_calls += 1;
        const idx = self.idxOf(k, c);
        const slot = idx orelse kernel_table.DOWN_CANDIDATES;
        const ki = @intFromEnum(k);
        const ri = @intFromEnum(r);
        self.nth[ki][slot][ri] += 1;
        self.calls.append(self.alloc, .{ .identity = false, .kernel = k, .idx = idx, .regime = r }) catch {};
        const spec = self.spec_fn(.{ .kernel = k, .idx = idx, .regime = r, .nth = self.nth[ki][slot][ri], .call = self.batch_calls });
        if (spec.user_stop) {
            self.ext_stop.store(true, .release);
            return .stopped;
        }
        if (spec.stop) return .stopped;
        if (spec.fail) return .{ .failed = "injected engine failure" };
        for (0..25) |_| {
            if (self.sc.clock == .real) self.sc.signals().waitNs(self.pair_ns) else self.sc.advanceMs(self.pair_ns / NS_PER_MS);
            hook.call();
            if (stop.load(.acquire)) return .stopped;
        }
        const ref: i64 = spec.ref orelse (if (r == .indep) @as(i64, 10_000) else 9_000);
        const diff = @divTrunc(ref * spec.diff_pct100, 10_000);
        const iqr = @divTrunc(ref * spec.iqr_pct100, 10_000);
        const stats: BatchStats = .{ .ref_ns = ref, .cand_ns = ref + diff, .diff_ns = diff, .q1_ns = diff - @divTrunc(iqr, 2), .q3_ns = diff + @divTrunc(iqr, 2), .lo_ns = diff - 20, .hi_ns = spec.hi orelse (diff + 20) };
        return .{ .done = .{ .stats = stats, .calls_compared = 40, .bits_differ = spec.bits } };
    }

    const vt: Worker.VTable = .{ .openWindow = openWindow, .closeWindow = closeWindow, .identity = identity, .timeBatch = timeBatch };

    fn worker(self: *FakeWorker) Worker {
        return .{ .ctx = self, .vt = &vt };
    }
};

const Harness = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    stop: std.atomic.Value(bool) = .init(false),
    sc: ScriptedSignals = undefined,
    fw: FakeWorker = undefined,
    key: kernel_table.Key = TEST_KEY,
    last_line: []const u8 = "",
    all_lines: std.ArrayList([]const u8) = .empty,
    /// Set the stop flag when the manual clock passes this many ms (0 never).
    stop_clock_ms: u64 = 0,

    fn create() !*Harness {
        const h = try testing.allocator.create(Harness);
        h.* = .{ .arena = std.heap.ArenaAllocator.init(testing.allocator), .tmp = std.testing.tmpDir(.{ .iterate = true }) };
        try h.tmp.dir.setPermissions(testing.io, .fromMode(0o700));
        return h;
    }

    fn destroy(self: *Harness) void {
        self.tmp.cleanup();
        self.arena.deinit();
        testing.allocator.destroy(self);
    }

    /// A fresh signal source and worker over the same tune folder (a second `start` is a second process).
    fn start(self: *Harness, script: []const u8, spec_fn: SpecFn, id_fn: IdFn) !void {
        const a = self.arena.allocator();
        self.stop.store(false, .release);
        self.sc = try ScriptedSignals.parse(a, script, .manual);
        self.fw = .{ .gu = kernel_table.gateUpCandidates(), .dn = kernel_table.downCandidates(), .sc = &self.sc, .spec_fn = spec_fn, .id_fn = id_fn, .ext_stop = &self.stop, .alloc = a };
    }

    fn onLine(ctx: ?*anyopaque, text: []const u8) void {
        const self: *Harness = @ptrCast(@alignCast(ctx.?));
        self.last_line = self.arena.allocator().dupe(u8, text) catch "";
        self.all_lines.append(self.arena.allocator(), self.last_line) catch {};
        if (self.stop_clock_ms != 0 and self.sc.now_ms >= self.stop_clock_ms) self.stop.store(true, .release);
    }

    fn config(self: *Harness) RunConfig {
        return .{
            .io = testing.io,
            .root = self.tmp.dir,
            .monitor = .manual,
            .key = self.key,
            .idle_minutes = 5,
            .reporter = .{ .ctx = self, .onLine = onLine },
        };
    }

    fn run(self: *Harness) !SchedResult {
        return tune_sched_run(self);
    }

    fn file(self: *Harness, name: []const u8) ![]u8 {
        return self.tmp.dir.readFileAlloc(testing.io, name, self.arena.allocator(), .limited(1 << 22)) catch |e| switch (e) {
            error.FileNotFound => return try self.arena.allocator().dupe(u8, ""),
            else => return e,
        };
    }
};

fn tune_sched_run(h: *Harness) !SchedResult {
    return run(h.arena.allocator(), h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
}

fn countLines(text: []const u8, a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| {
        if (std.mem.indexOf(u8, l, a) != null and std.mem.indexOf(u8, l, b) != null) n += 1;
    }
    return n;
}

fn finishedOutcome(r: SchedResult) !Outcome {
    return switch (r) {
        .finished => |o| o,
        .stopped => error.TestExpectedFinished,
    };
}

fn candText(buf: []u8, k: Kernel, idx: usize) []const u8 {
    return if (k == .gateup) kernel_table.gateUpText(buf, kernel_table.gateUpCandidates()[idx]) else kernel_table.downText(buf, kernel_table.downCandidates()[idx]);
}

// ── Scenarios shared by several tests ──

/// Gate/up candidate 0 is the one candidate that gains (-5%); every other candidate, and all of down, gains nothing.
fn gainSpec(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx != null and in.kernel == .gateup and in.idx.? == 0) s.diff_pct100 = -500;
    return s;
}

test "tune sched: a window opens only when every condition holds" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start("{\"at_ms\":0,\"hid_idle_s\":600,\"ac\":false,\"low_power\":true,\"thermal\":2}\n{\"at_ms\":30000,\"ac\":true}\n{\"at_ms\":60000,\"low_power\":false}\n{\"at_ms\":90000,\"thermal\":0}\n", baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    try testing.expect(h.fw.first_open_ms >= 90_000);
    try testing.expect(h.fw.first_open_ms < 110_000); // polls start 8.75 s apart: the one at 87.5 s still sees thermal pressure
    // the waiting lines named what was missing
    var saw_battery = false;
    for (h.all_lines.items) |l| saw_battery = saw_battery or std.mem.indexOf(u8, l, "on battery") != null;
    try testing.expect(saw_battery);
}

test "tune sched: an unreadable idle time never opens a window" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start("{\"at_ms\":0,\"hid_idle_s\":null}\n", baseSpec, idAllIdentical);
    h.stop_clock_ms = 3_600_000; // one hour of fake clock
    const r = try h.run();
    try testing.expect(r == .stopped);
    try testing.expectEqual(@as(u32, 0), h.fw.opens);
    try testing.expect(std.mem.indexOf(u8, h.last_line, "input unreadable") != null);
}

/// Input 100 ms into the first calibration batch (the first window opens at 3,750 ms of the fake clock, after the open
/// check's 8 GPU readings and 2 s CPU span), and the idle time then grows again until the next window.
const INPUT_IN_BATCH = "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":3850,\"hid_idle_s\":0}\n";

test "tune sched: input during a batch discards it through the monitor flag" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(INPUT_IN_BATCH, baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expect(countLines(prog, "\"phase\":\"cal\"", "\"kept\":false,\"reason\":\"input\"") >= 1);
    const log = try h.file(LOG_NAME);
    const at = std.mem.indexOf(u8, log, "stop input=") orelse return error.TestExpectedStopLine;
    const rest = log[at..];
    const lat = std.mem.indexOf(u8, rest, "latency_ms=") orelse return error.TestExpectedLatency;
    const ms = try std.fmt.parseInt(u64, std.mem.sliceTo(rest[lat + "latency_ms=".len ..], '\n'), 10);
    try testing.expect(ms < TUNE_STOP_LATENCY_MAX_MS);
    for ([_][]const u8{ "input=", "flag=", "abandoned=", "freed=" }) |f| try testing.expect(std.mem.indexOf(u8, rest, f) != null);
}

test "tune sched: input the monitor missed discards the batch after it" {
    const h = try Harness.create();
    defer h.destroy();
    monitor_enabled = false;
    defer monitor_enabled = true;
    try h.start(INPUT_IN_BATCH, baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expect(countLines(prog, "\"phase\":\"cal\"", "\"kept\":false,\"reason\":\"input\"") >= 1);
}

test "tune sched: a sleep gap discards the batch" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start("{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":3850,\"sleep_ms\":5000}\n", baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"kept\":false", "\"reason\":\"sleep\""));
}

test "tune sched: AC loss, Low Power Mode and thermal pressure discard the batch" {
    const cases = [_]struct { script: []const u8, reason: []const u8 }{
        .{ .script = "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":3850,\"ac\":false}\n{\"at_ms\":9000,\"ac\":true}\n", .reason = "power" },
        .{ .script = "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":3850,\"low_power\":true}\n{\"at_ms\":9000,\"low_power\":false}\n", .reason = "low_power" },
        .{ .script = "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":3850,\"thermal\":2}\n{\"at_ms\":9000,\"thermal\":0}\n", .reason = "thermal" },
    };
    for (cases) |c| {
        const h = try Harness.create();
        defer h.destroy();
        try h.start(c.script, baseSpec, idAllIdentical);
        _ = try finishedOutcome(try h.run());
        const prog = try h.file(PROGRESS_NAME);
        var needle_buf: [64]u8 = undefined;
        const needle = try std.fmt.bufPrint(&needle_buf, "\"reason\":\"{s}\"", .{c.reason});
        try testing.expectEqual(@as(usize, 1), countLines(prog, "\"kept\":false", needle));
    }
}

test "tune sched: memory pressure closes the window before a batch" {
    const h = try Harness.create();
    defer h.destroy();
    // memory turns bad exactly when the third batch would start (calibration batches take 350 ms from 3,750 ms)
    try h.start("{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":4450,\"mem_level\":2}\n{\"at_ms\":30000,\"mem_level\":1}\n", baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 2), countLines(prog, "\"window\":1,", "\"phase\":\"cal\""));
    try testing.expect(countLines(prog, "\"window\":2,", "\"phase\":\"cal\"") >= 4);
}

test "tune sched: the identity pass runs before any timing" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expect(h.fw.calls.items.len > 58);
    for (h.fw.calls.items[0..58]) |c| try testing.expect(c.identity);
    const first = h.fw.calls.items[58];
    try testing.expect(!first.identity and first.idx == null); // an A/A calibration batch
    try testing.expectEqual(@as(u32, 58), out.identity.plans_checked);
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 58), countLines(prog, "\"phase\":\"identity\"", "\"kept\":true"));
}

fn stopAtFifthIdentity(h: *Harness) void {
    h.fw.stop_at_identity = 5;
}

test "tune sched: a stop inside the identity pass redoes the cut candidate and skips the kept ones" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    stopAtFifthIdentity(h);
    try testing.expect((try h.run()) == .stopped);
    var prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 4), countLines(prog, "\"phase\":\"identity\"", "\"kept\":true"));
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"phase\":\"identity\"", "\"kept\":false,\"reason\":\"stopped\""));
    // the second run: candidates 1 to 4 never run again, candidate 5 first, then 6 to 58 (54 checks)
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(u32, 54), h.fw.identity_calls);
    const first = h.fw.calls.items[0];
    try testing.expect(first.identity and first.kernel == .gateup and first.idx.? == 4);
    prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 58), countLines(prog, "\"phase\":\"identity\"", "\"kept\":true"));
}

fn anyLineHas(h: *Harness, needle: []const u8) bool {
    for (h.all_lines.items) |l| {
        if (std.mem.indexOf(u8, l, needle) != null) return true;
    }
    return false;
}

/// The first screen batch fails with an engine error.
fn firstScreenFails(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx != null) s.fail = true;
    return s;
}

/// The third identity check fails with an engine error.
fn idThirdFails(_: Kernel, _: usize, n: u32) IdentityOutcome {
    if (n == 3) return .{ .failed = "injected identity failure" };
    return .{ .identical = .{ .calls_compared = 10 } };
}

test "tune sched: a worker failure ends the run with the message, never as a retried window" {
    {
        const h = try Harness.create();
        defer h.destroy();
        try h.start(IDLE_SCRIPT, firstScreenFails, idAllIdentical);
        try testing.expectError(error.TuneWorkerFailed, h.run());
        try testing.expectEqual(@as(u32, 1), h.fw.opens);
        try testing.expectEqual(@as(u32, 1), h.fw.closes);
        // one failing batch: calibration's 4 A/A batches, then the first screen batch, never a retry
        try testing.expectEqual(@as(u32, 5), h.fw.batch_calls);
        try testing.expect(anyLineHas(h, "the GPU worker failed: injected engine failure"));
        const log = try h.file(LOG_NAME);
        try testing.expect(std.mem.indexOf(u8, log, "the GPU worker failed: injected engine failure; the run ends") != null);
    }
    {
        const h = try Harness.create();
        defer h.destroy();
        try h.start(IDLE_SCRIPT, baseSpec, idThirdFails);
        try testing.expectError(error.TuneWorkerFailed, h.run());
        try testing.expectEqual(@as(u32, 1), h.fw.opens);
        try testing.expectEqual(@as(u32, 3), h.fw.identity_calls);
        try testing.expectEqual(@as(u32, 0), h.fw.batch_calls);
        try testing.expect(anyLineHas(h, "the GPU worker failed: injected identity failure"));
    }
    {
        // a 2,000-byte cause: the reported line and the log line keep the candidate and the cause's start
        const h = try Harness.create();
        defer h.destroy();
        try h.start(IDLE_SCRIPT, baseSpec, idThirdFailsLong);
        try testing.expectError(error.TuneWorkerFailed, h.run());
        var cbuf: [32]u8 = undefined;
        var nbuf: [160]u8 = undefined;
        const needle = try std.fmt.bufPrint(&nbuf, "identity gateup {s}: the GPU worker failed: injected long identity failure: xxxx", .{candText(&cbuf, .gateup, 2)});
        try testing.expect(anyLineHas(h, needle));
        try testing.expect(anyLineHas(h, "...; the run ends"));
        const log = try h.file(LOG_NAME);
        try testing.expect(std.mem.indexOf(u8, log, needle) != null);
    }
}

const LONG_FAILURE_TAIL: [1968]u8 = @splat('x');
const LONG_FAILURE: []const u8 = "injected long identity failure: " ++ &LONG_FAILURE_TAIL;

fn idThirdFailsLong(_: Kernel, _: usize, n: u32) IdentityOutcome {
    if (n == 3) return .{ .failed = LONG_FAILURE };
    return .{ .identical = .{ .calls_compared = 10 } };
}

/// Gate/up candidate 0's first screen batch (indep) gives differing bits.
fn bitsFirstScreen(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx != null and in.kernel == .gateup and in.idx.? == 0 and in.regime == .indep and in.nth == 1) s.bits = true;
    return s;
}

/// Input inside the first screen batch: the window opens at 3,750 ms, the four calibration batches take 350 ms each, so
/// the screen's first batch runs from 5,150 to 5,500 ms; input at 5,400 ms leaves 100 ms of idle time at its end.
const INPUT_IN_FIRST_SCREEN = "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":5400,\"hid_idle_s\":0}\n";

test "tune sched: differing bits outrank a timing discard" {
    monitor_enabled = false; // the after-batch reading alone catches the input, so the batch completes
    defer monitor_enabled = true;
    const cand0 = "\"cand\":\"sg=1 ax=0\"";
    {
        // bits differ and the batch is discarded for input: the candidate is excluded, never timed again
        const h = try Harness.create();
        defer h.destroy();
        try h.start(INPUT_IN_FIRST_SCREEN, bitsFirstScreen, idAllIdentical);
        const out = try finishedOutcome(try h.run());
        const prog = try h.file(PROGRESS_NAME);
        try testing.expectEqual(@as(usize, 1), countLines(prog, cand0, "\"reason\":\"bits differ during timing\""));
        try testing.expectEqual(@as(usize, 0), countLines(prog, cand0, "\"reason\":\"input\""));
        try testing.expectEqual(@as(usize, 0), countLines(prog, cand0, "\"kept\":true,\"reason\":null,\"ref_ns\""));
        var found = false;
        for (out.identity.excluded) |e| {
            if (std.mem.eql(u8, e.kernel, "gateup") and std.mem.eql(u8, e.cand, "sg=1 ax=0")) found = std.mem.eql(u8, e.reason, BITS_DIFFER_TIMING);
        }
        try testing.expect(found);
        var timed: u32 = 0;
        for (h.fw.calls.items) |c| {
            if (!c.identity and c.kernel == .gateup and c.idx != null and c.idx.? == 0) timed += 1;
        }
        try testing.expectEqual(@as(u32, 1), timed);
        try testing.expect(h.fw.opens >= 2); // the discard still closed the window
    }
    {
        // the input discard alone: the candidate stays, its batch is timed again and kept
        const h = try Harness.create();
        defer h.destroy();
        try h.start(INPUT_IN_FIRST_SCREEN, baseSpec, idAllIdentical);
        const out = try finishedOutcome(try h.run());
        const prog = try h.file(PROGRESS_NAME);
        try testing.expectEqual(@as(usize, 1), countLines(prog, cand0, "\"reason\":\"input\""));
        try testing.expectEqual(@as(usize, 0), countLines(prog, cand0, "\"reason\":\"bits differ during timing\""));
        try testing.expectEqual(@as(usize, 2), countLines(prog, cand0, "\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\""));
        for (out.identity.excluded) |e| try testing.expect(!(std.mem.eql(u8, e.kernel, "gateup") and std.mem.eql(u8, e.cand, "sg=1 ax=0")));
    }
}

fn aaNoisy(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx == null) s.iqr_pct100 = 900; // the A/A batches' IQR is 9%
    return s;
}

test "tune sched: calibration per kernel and regime, three rounds then noisy" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, aaNoisy, idAllIdentical);
    h.fw.stop_at_open = 2; // stop when the second window opens: the first one ended as noisy
    try testing.expect((try h.run()) == .stopped);
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 12), countLines(prog, "\"window\":1,", "\"phase\":\"cal\""));
    try testing.expectEqual(@as(usize, 0), countLines(prog, "\"phase\":\"screen\"", ""));
    const log = try h.file(LOG_NAME);
    try testing.expect(std.mem.indexOf(u8, log, "closed as noisy") != null);
    // two waits of 60 s between the three rounds
    try testing.expect(h.sc.now_ms >= 3_750 + 12 * 350 + 2 * 60_000);
}

fn sharedRef9100(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx != null and in.regime == .shared) s.ref = 9_100;
    return s;
}

test "tune sched: the scheduler gates a shared batch against the shared reference" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, sharedRef9100, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    // calibration gave indep 10,000 and shared 9,000; a shared screen batch at 9,100 is kept (against 10,000 it would fail)
    try testing.expect(countLines(prog, "\"phase\":\"screen\"", "\"regime\":\"shared\",\"window\":1,\"kept\":true,\"reason\":null,\"ref_ns\":9100") > 0);
    try testing.expectEqual(@as(usize, 0), countLines(prog, "\"reason\":\"gate\"", ""));
}

// ── More scenarios ──

fn isX(in: SpecIn) bool {
    return in.kernel == .gateup and in.idx != null and in.idx.? == 0;
}

fn isXIndep(in: SpecIn) bool {
    return isX(in) and in.regime == .indep;
}

/// A reference median 20% above the calibration one: the batch fails the 8% drift gate.
const GATE_BAD_REF: i64 = 12_000;

/// Both kernels' candidate 0 gain 5%.
fn bothGain(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx != null and in.idx.? == 0) s.diff_pct100 = -500;
    return s;
}

/// X fails the gate twice in the screen and three times in the final stage (one batch per attempt).
fn failAcrossPhases(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in) and (in.nth <= 2 or (in.nth >= 4 and in.nth <= 6))) s.ref = GATE_BAD_REF;
    return s;
}

fn gateFinalOnce(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in) and in.nth == 2) s.ref = GATE_BAD_REF;
    return s;
}

/// Gate/up candidates 0 to 3 gain 6, 4, 3 and 2%; candidate 0 fails like `failAcrossPhases`' X.
fn replacedFinalist(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.kernel == .gateup and in.idx != null and in.idx.? < 4) {
        const gains = [4]i64{ -600, -400, -300, -200 };
        s.diff_pct100 = gains[in.idx.?];
        if (in.idx.? == 0 and in.regime == .indep and (in.nth <= 2 or (in.nth >= 4 and in.nth <= 6))) s.ref = GATE_BAD_REF;
    }
    return s;
}

/// Four gate abandonments of X's final-stage batches (calls 2, 4, 5, 7) and three window closings between (3, 6, 8), then
/// the fifth gate abandonment (9).
fn fiveGateAbandons(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in)) switch (in.nth) {
        2, 4, 5, 7, 9 => s.ref = GATE_BAD_REF,
        3, 6, 8 => s.stop = true,
        else => {},
    };
    return s;
}

fn c2Requeue(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in) and in.nth == 4) s.ref = GATE_BAD_REF;
    return s;
}

/// X carries 3 gate failures into `confirm2` (screen 1 and 2, final 4) and fails it twice there (7 and 8); down's pick is clean.
fn c2NotMeasurable(in: SpecIn) Spec {
    var s = bothGain(in);
    if (isXIndep(in)) switch (in.nth) {
        1, 2, 4, 7, 8 => s.ref = GATE_BAD_REF,
        else => {},
    };
    return s;
}

fn c2Bits(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in) and in.nth == 4) s.bits = true;
    return s;
}

fn c1NotConfirmed(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in) and in.nth == 3) s.hi = 10;
    return s;
}

fn c2NotConfirmed(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isXIndep(in) and in.nth == 4) s.hi = 10;
    return s;
}

fn belowFloor(in: SpecIn) Spec {
    var s = gainSpec(in);
    if (isX(in) and in.nth == 3) s.diff_pct100 = -100;
    return s;
}

fn noMeasurableCandidate(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.kernel == .gateup and in.idx != null) s.ref = GATE_BAD_REF;
    return s;
}

fn idGateupDiffers(k: Kernel, _: usize, _: u32) IdentityOutcome {
    return if (k == .gateup) .{ .differs = 3 } else .{ .identical = .{ .calls_compared = 10 } };
}

test "tune sched: the five-failure rule counts across phases" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, failAcrossPhases, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(usize, 1), out.not_measurable.len);
    try testing.expectEqualStrings("gateup", out.not_measurable[0].kernel);
    try testing.expectEqualStrings("sg=1 ax=0", out.not_measurable[0].cand);
    // never timed again after its fifth failure: 2 screen failures, 1 kept, 3 final-stage failures
    try testing.expectEqual(@as(u32, 6), h.fw.nth[0][0][0]);
    try testing.expectEqual(@as(?ServedReason, .no_measurable_finalist), out.served_reasons[0]);
}

test "tune sched: a gate failure in the final stage abandons the attempt and charges the candidate" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, gateFinalOnce, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"phase\":\"final\"", "\"reason\":\"gate\",\"ref_ns\":12000"));
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"phase\":\"final\"", "\"attempt\":1}"));
    // the abandoned attempt wrote no marker; the next one did
    try testing.expectEqual(@as(usize, 0), countLines(prog, "stage_done", "\"attempt\":1,"));
    try testing.expectEqual(@as(usize, 1), countLines(prog, "stage_done", "\"attempt\":2,"));
}

test "tune sched: a finalist that becomes not measurable is replaced, and none left records no_measurable_finalist" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, replacedFinalist, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(usize, 1), out.not_measurable.len);
    const prog = try h.file(PROGRESS_NAME);
    // attempt 4 holds candidates 1, 2 and 3 (sg=1 ax=1, sg=2 ax=0, sg=2 ax=1), not 0
    try testing.expect(countLines(prog, "\"phase\":\"final\",\"kernel\":\"gateup\",\"cand\":\"sg=2 ax=1\"", "\"attempt\":4}") > 0);
    try testing.expectEqual(@as(usize, 0), countLines(prog, "\"cand\":\"sg=1 ax=0\"", "\"attempt\":4}"));
    // the other branch of the rule is `the five-failure rule counts across phases`: the only finalist is lost, none is left
    const h2 = try Harness.create();
    defer h2.destroy();
    try h2.start(IDLE_SCRIPT, failAcrossPhases, idAllIdentical);
    const out2 = try finishedOutcome(try h2.run());
    try testing.expectEqual(@as(?ServedReason, .no_measurable_finalist), out2.served_reasons[0]);
}

test "tune sched: five gate abandonments record final_stage_noisy, abandonments by input do not count" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, fiveGateAbandons, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(?ServedReason, .final_stage_noisy), out.served_reasons[0]);
    try testing.expect(out.plan.gateup == null and out.evidence[0] == null);
    // the three closings did not count: the fifth gate abandonment came on X's ninth indep batch
    try testing.expectEqual(@as(u32, 9), h.fw.nth[0][0][0]);
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 0), countLines(prog, "\"phase\":\"confirm2\"", ""));
    try testing.expectEqual(@as(usize, 0), countLines(prog, "stage_done", ""));
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), out.served_reasons[1]);
}

test "tune sched: a closed window during the finalists or before confirm1 leaves no marker, and resume starts a new attempt" {
    const Cut = struct {
        fn atFinalist(in: SpecIn) Spec {
            var s = gainSpec(in);
            if (isXIndep(in) and in.nth == 2) s.user_stop = true;
            return s;
        }
        fn atConfirm1(in: SpecIn) Spec {
            var s = gainSpec(in);
            if (isXIndep(in) and in.nth == 3) s.user_stop = true;
            return s;
        }
    };
    const fns = [_]SpecFn{ Cut.atFinalist, Cut.atConfirm1 };
    for (fns) |f| {
        const h = try Harness.create();
        defer h.destroy();
        try h.start(IDLE_SCRIPT, f, idAllIdentical);
        try testing.expect((try h.run()) == .stopped);
        const prog = try h.file(PROGRESS_NAME);
        try testing.expectEqual(@as(usize, 0), countLines(prog, "stage_done", ""));
        // a second process: the void lines are not reused, the final stage starts again with the first finalist
        try h.start(IDLE_SCRIPT, gainSpec, idAllIdentical);
        _ = try finishedOutcome(try h.run());
        try testing.expectEqual(@as(u32, 0), h.fw.identity_calls);
        const first_timed = h.fw.calls.items[4]; // after the 4 calibration batches
        try testing.expect(!first_timed.identity and first_timed.kernel == .gateup and first_timed.idx.? == 0 and first_timed.regime == .indep);
    }
}

test "tune sched: confirm2 never shares the final stage's window" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, gainSpec, idAllIdentical); // the Mac stays idle the whole time
    const out = try finishedOutcome(try h.run());
    const ev = out.evidence[0].?;
    try testing.expect(ev.confirm[2].window > ev.confirm[0].window);
    try testing.expect(ev.confirm[3].window > ev.confirm[1].window);
    try testing.expectEqual(ev.confirm[0].window, ev.confirm[1].window);
    try testing.expect(out.windows_used >= 2);
}

test "tune sched: a monitor stop returns within one pair" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start("{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":100,\"hid_idle_s\":0}\n", baseSpec, idAllIdentical);
    var stop: std.atomic.Value(bool) = .init(false);
    const res = runBatchWith(h.sc.signals(), h.fw.worker(), .gateup, .reference, .indep, 1, &stop, null, .manual);
    try testing.expectEqual(@as(?DiscardReason, .input), res.discard);
    const st = res.stop_times.?;
    try testing.expect(st.abandoned_ns - st.flag_ns <= 14 * NS_PER_MS);
    // `freed_ns` of a direct call is a placeholder (the abandonment time): the freed time is read after the window closes
    h.fw.worker().closeWindow();
    const freed = h.sc.signals().absTimeNs();
    try testing.expect((freed - st.input_ns) / NS_PER_MS < TUNE_STOP_LATENCY_MAX_MS);
    try testing.expect(stop.load(.acquire));
}

test "tune sched: the monitor thread sets the stop flag" {
    var ext: std.atomic.Value(bool) = .init(false);
    var sc = try ScriptedSignals.parse(testing.allocator, "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":150,\"hid_idle_s\":0}\n", .real);
    defer sc.deinit();
    var fw: FakeWorker = .{ .gu = kernel_table.gateUpCandidates(), .dn = kernel_table.downCandidates(), .sc = &sc, .ext_stop = &ext, .alloc = testing.allocator, .pair_ns = 25 * NS_PER_MS };
    defer fw.calls.deinit(testing.allocator);
    var stop: std.atomic.Value(bool) = .init(false);
    const res = runBatchWith(sc.signals(), fw.worker(), .gateup, .reference, .indep, 1, &stop, null, .thread);
    try testing.expectEqual(@as(?DiscardReason, .input), res.discard);
    try testing.expect(stop.load(.acquire));
    try testing.expect(res.outcome == .stopped);
}

test "tune sched: a torn last line is ignored and truncated on resume" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const header = try headerLine(a, TEST_KEY);
    const line = "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\",\"window\":1,\"kept\":true,\"reason\":null,\"ref_ns\":10000,\"cand_ns\":10100,\"diff_ns\":100,\"q1_ns\":90,\"q3_ns\":110,\"lo_ns\":80,\"hi_ns\":120,\"calls_compared\":40}";
    const torn = "{\"phase\":\"scr";
    const content = try std.fmt.allocPrint(a, "{s}\n{s}\n{s}", .{ header, line, torn });
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = content });
    const p = try loadProgress(a, testing.io, h.tmp.dir, TEST_KEY);
    try testing.expect(!p.restarted);
    try testing.expectEqual(@as(usize, 1), p.recs.len);
    try testing.expectEqual(@as(u64, header.len + 1 + line.len + 1), p.valid_len);
    try testing.expectEqual(@as(u64, content.len), p.file_len);
    // the next append lands on a clean line
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    try s.prog.appendLine(a, "{\"x\":1}", true);
    s.deinit();
    const after = try h.file(PROGRESS_NAME);
    const want = try std.fmt.allocPrint(a, "{s}\n{s}\n{{\"x\":1}}\n", .{ header, line });
    try testing.expectEqualStrings(want, after);
}

fn stopAtScreen21(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.call == 4 + 21) s.user_stop = true; // 4 calibration batches, 20 kept screen batches, then the cut one
    return s;
}

test "tune sched: a resumed run skips kept batches" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, stopAtScreen21, idAllIdentical);
    try testing.expect((try h.run()) == .stopped);
    var prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 20), countLines(prog, "\"phase\":\"screen\"", "\"kept\":true"));
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(u32, 0), h.fw.identity_calls);
    try testing.expectEqual(@as(u32, 4 + 96), h.fw.batch_calls); // a calibration round and the remaining 96 screen batches (116 - 20)
    prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 116), countLines(prog, "\"phase\":\"screen\"", "\"kept\":true"));
}

test "tune sched: a header with another key starts over" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    h.fw.stop_at_identity = 10;
    try testing.expect((try h.run()) == .stopped);
    h.key.kernel_hash = "bb22";
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    var said = false;
    for (h.all_lines.items) |l| said = said or std.mem.indexOf(u8, l, "starts over") != null;
    try testing.expect(said);
    try testing.expectEqual(@as(u32, 58), h.fw.identity_calls);
}

test "tune sched: a space 1 progress file restarts the run" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    h.fw.stop_at_identity = 10;
    try testing.expect((try h.run()) == .stopped);
    // a 0.26.0 file: the same header with the plan space it named
    const old = try h.file(PROGRESS_NAME);
    const want = try std.fmt.allocPrint(a, "\"space\":{d}}}", .{kernel_table.TUNE_PLAN_SPACE_VERSION});
    try testing.expect(std.mem.indexOf(u8, old, want) != null);
    const v1 = try std.mem.replaceOwned(u8, a, old, want, "\"space\":1}");
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = v1 });
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    _ = try finishedOutcome(try h.run());
    var said = false;
    for (h.all_lines.items) |l| said = said or std.mem.indexOf(u8, l, "starts over") != null;
    try testing.expect(said);
    // nothing of the old file was replayed: the identity pass ran every one of the 58 candidates again
    try testing.expectEqual(@as(u32, 58), h.fw.identity_calls);
    try testing.expectEqual(@as(usize, 58), countLines(try h.file(PROGRESS_NAME), "\"phase\":\"identity\"", "\"kept\":true"));
}

fn bitsInShared(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.kernel == .gateup and in.idx != null and in.idx.? == 2 and in.regime == .shared and in.nth == 1) s.bits = true;
    return s;
}

test "tune sched: bits differing during timing exclude the candidate" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, bitsInShared, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"kept\":false", "\"reason\":\"bits differ during timing\""));
    var found = false;
    for (out.identity.excluded) |e| found = found or (std.mem.eql(u8, e.cand, "sg=2 ax=0") and std.mem.eql(u8, e.reason, "bits differ during timing"));
    try testing.expect(found);
    // never timed again: one indep batch (kept before the exclusion) and the one shared batch that differed
    try testing.expectEqual(@as(u32, 1), h.fw.nth[0][2][0]);
    try testing.expectEqual(@as(u32, 1), h.fw.nth[0][2][1]);
    // a resumed run keeps it excluded and its earlier kept batch void
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(h.arena.allocator(), h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(s.ks[0].cands[2].bits_excluded);
    try testing.expect(s.ks[0].cands[2].screen[0] == null);
    try testing.expectEqualStrings("bits differ during timing", s.ks[0].reason[2].?);
}

test "tune sched: a run with picks needs two windows" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, bothGain, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expect(out.plan.gateup != null and out.plan.down != null);
    try testing.expect(out.windows_used >= 2);
    try testing.expect(out.evidence[0] != null and out.evidence[1] != null);
    try testing.expect(out.served_reasons[0] == null and out.served_reasons[1] == null);
    // the pick is the candidate that gained
    try testing.expectEqual(kernel_table.gateUpCandidates()[0], out.plan.gateup.?);
    try testing.expectEqual(kernel_table.downCandidates()[0], out.plan.down.?);
    try testing.expectEqual(@as(i64, 10_000), out.evidence[0].?.ref_ns);
}

test "tune sched: a run with no gain records served" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expect(out.plan.gateup == null and out.plan.down == null);
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), out.served_reasons[0]);
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), out.served_reasons[1]);
    try testing.expectEqual(@as(u32, 1), out.windows_used);
    try testing.expectEqual(@as(u32, 4 + 116), h.fw.batch_calls);
}

test "tune sched: verify checks only the plan's kernels" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    var cfg = h.config();
    cfg.mode = .verify;
    cfg.verify_plan = .{ .gateup = kernel_table.gateUpCandidates()[3], .down = null };
    const r = try verify(h.arena.allocator(), cfg, h.sc.signals(), h.fw.worker(), &h.stop);
    try testing.expectEqual(@as(u8, 1), r.identical.kernels);
    try testing.expectEqual(@as(u32, 1), h.fw.identity_calls);
    try testing.expect(h.fw.calls.items[0].kernel == .gateup and h.fw.calls.items[0].idx.? == 3);
    // a differing layout fails the check
    try h.start(IDLE_SCRIPT, baseSpec, idGateupDiffers);
    const bad = try verify(h.arena.allocator(), cfg, h.sc.signals(), h.fw.worker(), &h.stop);
    try testing.expect(bad == .failed);
    try testing.expectEqualStrings("bits differ in case 3", bad.failed.reason);
}

test "tune sched: a confirm2 gate failure re-queues the batch" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, c2Requeue, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"phase\":\"confirm2\"", "\"kept\":false,\"reason\":\"gate\""));
    // the failed batch came before a kept one for the same kernel and regime
    const failed_at = std.mem.indexOf(u8, prog, "\"phase\":\"confirm2\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\",\"window\":2,\"kept\":false,\"reason\":\"gate\"") orelse return error.TestExpectedGateLine;
    const kept_at = std.mem.indexOfPos(u8, prog, failed_at, "\"phase\":\"confirm2\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\",\"window\":2,\"kept\":true") orelse return error.TestExpectedKeptLine;
    try testing.expect(kept_at > failed_at);
    try testing.expect(out.plan.gateup != null);
    try testing.expectEqual(@as(usize, 0), out.not_measurable.len);
}

test "tune sched: a pick not measurable in confirm2 records confirm2_not_measurable" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, c2NotMeasurable, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(?ServedReason, .confirm2_not_measurable), out.served_reasons[0]);
    try testing.expect(out.plan.gateup == null);
    try testing.expect(out.plan.down != null); // the other kernel's pick stands
    try testing.expectEqual(@as(usize, 1), out.not_measurable.len);
}

test "tune sched: bits differing in confirm2 record pick_bits_differ" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, c2Bits, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(?ServedReason, .pick_bits_differ), out.served_reasons[0]);
    var found = false;
    for (out.identity.excluded) |e| found = found or (std.mem.eql(u8, e.cand, "sg=1 ax=0") and std.mem.eql(u8, e.reason, "bits differ during timing"));
    try testing.expect(found);
    // no new final-stage attempt
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 1), countLines(prog, "stage_done", ""));
}

test "tune sched: every served reason reaches the outcome" {
    const Row = struct { reason: ServedReason, spec: SpecFn, id: IdFn };
    const rows = [_]Row{
        .{ .reason = .no_identical_candidate, .spec = baseSpec, .id = idGateupDiffers },
        .{ .reason = .no_measurable_candidate, .spec = noMeasurableCandidate, .id = idAllIdentical },
        .{ .reason = .no_gain_in_screen, .spec = baseSpec, .id = idAllIdentical },
        .{ .reason = .no_measurable_finalist, .spec = failAcrossPhases, .id = idAllIdentical },
        .{ .reason = .final_stage_noisy, .spec = fiveGateAbandons, .id = idAllIdentical },
        .{ .reason = .confirm1_not_confirmed, .spec = c1NotConfirmed, .id = idAllIdentical },
        .{ .reason = .confirm2_not_confirmed, .spec = c2NotConfirmed, .id = idAllIdentical },
        .{ .reason = .below_floor, .spec = belowFloor, .id = idAllIdentical },
        .{ .reason = .pick_bits_differ, .spec = c2Bits, .id = idAllIdentical },
        .{ .reason = .confirm2_not_measurable, .spec = c2NotMeasurable, .id = idAllIdentical },
    };
    for (rows) |row| {
        const h = try Harness.create();
        defer h.destroy();
        try h.start(IDLE_SCRIPT, row.spec, row.id);
        const out = try finishedOutcome(try h.run());
        try testing.expectEqual(@as(?ServedReason, row.reason), out.served_reasons[0]);
        try testing.expect(out.plan.gateup == null);
    }
}

// ── Fix round 1 tests ──

test "tune sched: bad progress records are skipped one by one and only a bad tail is cut" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const header = try headerLine(a, TEST_KEY);
    const good1 = "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\",\"window\":1,\"kept\":true,\"reason\":null,\"ref_ns\":10000,\"cand_ns\":10100,\"diff_ns\":100,\"q1_ns\":90,\"q3_ns\":110,\"lo_ns\":80,\"hi_ns\":120,\"calls_compared\":40}";
    const good2 = "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=1\",\"regime\":\"indep\",\"window\":1,\"kept\":true,\"reason\":null,\"ref_ns\":10000,\"cand_ns\":10100,\"diff_ns\":100,\"q1_ns\":90,\"q3_ns\":110,\"lo_ns\":80,\"hi_ns\":120,\"calls_compared\":40}";
    const huge_window = "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=2 ax=0\",\"regime\":\"indep\",\"window\":4294967296,\"kept\":true,\"reason\":null}";
    const marker_false = "{\"stage_done\":false,\"attempt\":1,\"window\":1}";
    const marker_huge = "{\"stage_done\":\"final\",\"attempt\":4294967296,\"window\":1}";
    const marker_ok = "{\"stage_done\":\"final\",\"attempt\":1,\"window\":1}";
    // interior bad records between good ones: all three skipped, both good lines and the marker kept, nothing cut
    const content = try std.fmt.allocPrint(a, "{s}\n{s}\n{s}\n{s}\n{s}\n{s}\n{s}\n", .{ header, good1, huge_window, marker_false, marker_huge, good2, marker_ok });
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = content });
    const p = try loadProgress(a, testing.io, h.tmp.dir, TEST_KEY);
    try testing.expectEqual(@as(usize, 3), p.recs.len);
    try testing.expectEqual(@as(u64, content.len), p.valid_len);
    // a bad record at the tail is cut, the good ones before it stay
    const tail_bad = try std.fmt.allocPrint(a, "{s}\n{s}\n{s}\n", .{ header, good1, huge_window });
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = tail_bad });
    const q = try loadProgress(a, testing.io, h.tmp.dir, TEST_KEY);
    try testing.expectEqual(@as(usize, 1), q.recs.len);
    try testing.expectEqual(@as(u64, header.len + 1 + good1.len + 1), q.valid_len);
}

test "tune sched: a resumed run restores the comparison count and counts no identity line as a kept batch" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, stopAtScreen21, idAllIdentical);
    try testing.expect((try h.run()) == .stopped);
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    // two calibration rounds (4 + 4) and 116 screen batches: identity lines are not timing batches
    try testing.expectEqual(@as(u32, 4 + 20 + 4 + 96), out.batches_kept);
    // 58 identity checks of 10 comparisons, and 40 for every kept batch of both runs (4 + 20 + 4 + 96 = 124)
    try testing.expectEqual(@as(u64, 58 * 10 + 124 * 40), out.identity.calls_compared);
}

// ── Fix round 2 tests ──

fn rawLine(a: Allocator, phase: []const u8, cand: []const u8, regime: ?[]const u8, kept: bool, reason: ?[]const u8, calls: u64) ![]u8 {
    const rg = if (regime) |r| try std.fmt.allocPrint(a, "\"{s}\"", .{r}) else "null";
    const why = if (reason) |w| try std.fmt.allocPrint(a, "\"{s}\"", .{w}) else "null";
    const timing = if (regime != null) ",\"ref_ns\":10000,\"cand_ns\":10100,\"diff_ns\":100,\"q1_ns\":90,\"q3_ns\":110,\"lo_ns\":80,\"hi_ns\":120" else "";
    return std.fmt.allocPrint(a, "{{\"phase\":\"{s}\",\"kernel\":\"gateup\",\"cand\":\"{s}\",\"regime\":{s},\"window\":1,\"kept\":{},\"reason\":{s}{s},\"calls_compared\":{d}}}", .{ phase, cand, rg, kept, why, timing, calls });
}

test "tune sched: replay validates identity lines and saturates its counters" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const header = try headerLine(a, TEST_KEY);
    const big: u64 = std.math.maxInt(i64);
    var content: std.ArrayList(u8) = .empty;
    try content.print(a, "{s}\n", .{header});
    const lines = [_][]u8{
        try rawLine(a, "identity", "sg=1 ax=0", null, true, null, 0), // no count and no reason: incomplete
        try rawLine(a, "identity", "sg=1 ax=1", null, true, null, 10), // a complete identical line
        try rawLine(a, "identity", "sg=2 ax=0", null, true, "garbage", 0), // a reason no identity pass writes
        try rawLine(a, "identity", "sg=2 ax=1", null, true, "bits differ in case 4", 0), // a real exclusion
        try rawLine(a, "screen", "sg=4 ax=0", "indep", true, null, big),
        try rawLine(a, "screen", "sg=4 ax=0", "shared", true, null, big),
        try rawLine(a, "screen", "sg=4 ax=1", "indep", true, null, big),
    };
    for (lines) |l| try content.print(a, "{s}\n", .{l});
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = content.items });
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.ks[0].id_done[0]); // checked again
    try testing.expect(s.ks[0].id_done[1] and s.ks[0].cands[1].identical == true);
    try testing.expect(!s.ks[0].id_done[2]);
    try testing.expect(s.ks[0].id_done[3] and s.ks[0].cands[3].bits_excluded);
    // three kept batches of the largest count a record can hold: the total saturates, it does not overflow
    try testing.expectEqual(@as(u64, std.math.maxInt(u64)), s.calls_total);
    try testing.expectEqual(@as(u32, 3), s.batches_kept);
}

test "tune sched: the writers do not follow a symbolic link" {
    const h = try Harness.create();
    defer h.destroy();
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = "victim.txt", .data = "keep" });
    try testing.expectEqual(@as(c_int, 0), std.c.symlinkat("victim.txt", h.tmp.dir.handle, PROGRESS_NAME));
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    try testing.expectError(error.TuneFolderUnwritable, Sched.init(h.arena.allocator(), h.config(), h.sc.signals(), h.fw.worker(), &h.stop));
    try testing.expectEqualStrings("keep", try h.file("victim.txt"));
}

fn bothGainGateupBitsDownCut(in: SpecIn) Spec {
    var s = bothGain(in);
    if (isXIndep(in) and in.nth == 4) s.bits = true; // gate/up's pick differs in confirm2
    if (in.kernel == .down and in.idx != null and in.idx.? == 0 and in.regime == .indep and in.nth == 4) s.user_stop = true; // then the run is cut
    return s;
}

test "tune sched: a resume keeps a pick that confirm2 disqualified out" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, bothGainGateupBitsDownCut, idAllIdentical);
    try testing.expect((try h.run()) == .stopped);
    try h.start(IDLE_SCRIPT, bothGain, idAllIdentical);
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(?ServedReason, .pick_bits_differ), out.served_reasons[0]);
    try testing.expect(out.plan.gateup == null);
    try testing.expect(out.plan.down != null); // the other kernel finished its confirm2 in the second run
    // the disqualified pick was not timed again
    try testing.expectEqual(@as(u32, 0), h.fw.nth[0][0][0]);
}

test "tune sched: a finished run resumes as finished, and a lost finalist stays lost" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, gainSpec, idAllIdentical);
    const first = try finishedOutcome(try h.run());
    try h.start(IDLE_SCRIPT, gainSpec, idAllIdentical);
    const again = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(u32, 0), h.fw.identity_calls);
    try testing.expectEqual(@as(u32, 0), h.fw.batch_calls);
    try testing.expectEqual(first.plan, again.plan);
    try testing.expectEqual(first.windows_used, again.windows_used); // the windows of the first run still count
    try testing.expectEqual(first.evidence[0].?, again.evidence[0].?);
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), again.served_reasons[1]);
    // a finalist lost to the five-failure rule is not replaced by a screen reason after a restart
    const h2 = try Harness.create();
    defer h2.destroy();
    try h2.start(IDLE_SCRIPT, failAcrossPhases, idAllIdentical);
    _ = try finishedOutcome(try h2.run());
    try h2.start(IDLE_SCRIPT, failAcrossPhases, idAllIdentical);
    const out = try finishedOutcome(try h2.run());
    try testing.expectEqual(@as(?ServedReason, .no_measurable_finalist), out.served_reasons[0]);
    try testing.expectEqual(@as(usize, 1), out.not_measurable.len);
}

// ── Fix round 3 tests ──

fn rawAttempt(a: Allocator, phase: []const u8, cand: []const u8, regime: []const u8, window: u32, attempt: u32) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"phase\":\"{s}\",\"kernel\":\"gateup\",\"cand\":\"{s}\",\"regime\":\"{s}\",\"window\":{d},\"attempt\":{d},\"kept\":true,\"reason\":null,\"ref_ns\":10000,\"cand_ns\":9500,\"diff_ns\":-500,\"q1_ns\":-600,\"q3_ns\":-400,\"lo_ns\":-700,\"hi_ns\":-300,\"calls_compared\":40}}", .{ phase, cand, regime, window, attempt });
}

test "tune sched: replay rejects a case number that does not exist and an id near the end of u32" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const header = try headerLine(a, TEST_KEY);
    const bad_case = try rawLine(a, "identity", "sg=1 ax=0", null, true, "bits differ in case 99", 0);
    const good_case = try rawLine(a, "identity", "sg=1 ax=1", null, true, "bits differ in case 10", 0);
    const content = try std.fmt.allocPrint(a, "{s}\n{s}\n{s}\n", .{ header, bad_case, good_case });
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = content });
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.ks[0].id_done[0] and !s.ks[0].cands[0].bits_excluded);
    try testing.expect(s.ks[0].id_done[1] and s.ks[0].cands[1].bits_excluded);
    // ids: one below the bound parses, one at the old u32 end does not
    try testing.expect((try parseRec(a, "{\"stage_done\":\"final\",\"attempt\":4294967294,\"window\":1}")) == null);
    try testing.expect((try parseRec(a, "{\"stage_done\":\"final\",\"attempt\":7,\"window\":1}")) != null);
}

test "tune sched: the progress file is never read through a link" {
    const h = try Harness.create();
    defer h.destroy();
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = "real.jsonl", .data = "x\n" });
    try testing.expectEqual(@as(c_int, 0), std.c.symlinkat("real.jsonl", h.tmp.dir.handle, PROGRESS_NAME));
    try testing.expectError(error.TuneFolderUnwritable, loadProgress(h.arena.allocator(), testing.io, h.tmp.dir, TEST_KEY));
    // the complement: the same bytes as a regular file are read
    const h2 = try Harness.create();
    defer h2.destroy();
    try h2.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = "x\n" });
    const p = try loadProgress(h2.arena.allocator(), testing.io, h2.tmp.dir, TEST_KEY);
    try testing.expect(p.restarted and p.had_file);
}

test "tune sched: a marker whose confirm1 records name two candidates is not trusted" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    // a finished screen with two gate/up finalists (A and B), then their final stage
    var lines = try screenedLines(a, &.{ 0, 1 }, false);
    try lines.appendSlice(a, &[_][]const u8{
        try rawAttempt(a, "final", "sg=1 ax=0", "indep", 2, 1),
        try rawAttempt(a, "final", "sg=1 ax=0", "shared", 2, 1),
        try rawAttempt(a, "final", "sg=1 ax=1", "indep", 2, 1),
        try rawAttempt(a, "final", "sg=1 ax=1", "shared", 2, 1),
        try rawAttempt(a, "confirm1", "sg=1 ax=0", "indep", 2, 1), // independent of A ...
        try rawAttempt(a, "confirm1", "sg=1 ax=1", "shared", 2, 1), // ... shared of B
        "{\"stage_done\":\"final\",\"attempt\":1,\"window\":2}",
    });
    try writeProgress(h, lines.items);
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.final_done);
    // the complement: the pick (A: the finals tie, the first finalist wins) has both regimes in the marker's window
    lines.items[lines.items.len - 2] = try rawAttempt(a, "confirm1", "sg=1 ax=0", "shared", 2, 1);
    try writeProgress(h, lines.items);
    const s2 = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s2.deinit();
    try testing.expect(s2.final_done);
    try testing.expectEqual(@as(usize, 0), s2.ks[0].pick);
}

// ── Fix round 4 tests ──

extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;

fn writeProgress(h: *Harness, lines: []const []const u8) !void {
    const a = h.arena.allocator();
    var content: std.ArrayList(u8) = .empty;
    try content.print(a, "{s}\n", .{try headerLine(a, TEST_KEY)});
    for (lines) |l| try content.print(a, "{s}\n", .{l});
    try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = content.items });
}

test "tune sched: replay skips an identity check only when the line compared every case" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    // gate/up candidate 0 compared 1 case (incomplete: checked again), candidate 1 compared all 10 (skipped)
    try writeProgress(h, &.{
        "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":null,\"window\":1,\"kept\":true,\"reason\":null,\"calls_compared\":1}",
        "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=1\",\"regime\":null,\"window\":1,\"kept\":true,\"reason\":null,\"calls_compared\":10}",
    });
    _ = try finishedOutcome(try h.run());
    var checked = [2]u32{ 0, 0 };
    for (h.fw.calls.items) |c| {
        if (c.identity and c.kernel == .gateup and c.idx != null and c.idx.? < 2) checked[c.idx.?] += 1;
    }
    try testing.expectEqual(@as(u32, 1), checked[0]);
    try testing.expectEqual(@as(u32, 0), checked[1]);
    try testing.expectEqual(@as(u32, 57), h.fw.identity_calls); // 58 candidates, one skipped
}

test "tune sched: the writers refuse a FIFO and append to a regular file" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try h.tmp.dir.realPath(testing.io, &dir_buf)];
    const fifo = try std.fs.path.joinZ(a, &.{ dir, LOG_NAME });
    try testing.expectEqual(@as(c_int, 0), mkfifo(fifo.ptr, 0o600));
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    // no reader: the open is refused at once instead of blocking
    try testing.expectError(error.TuneFolderUnwritable, Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop));
    // a reader present: the open succeeds, and the file-type check refuses it
    const reader = std.c.open(fifo.ptr, .{ .ACCMODE = .RDONLY, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    try testing.expect(reader >= 0);
    defer _ = std.c.close(reader);
    try testing.expectError(error.TuneFolderUnwritable, Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop));
    // the complement: a regular file at the same name is opened and appended to, its earlier line kept
    const h2 = try Harness.create();
    defer h2.destroy();
    try h2.tmp.dir.writeFile(testing.io, .{ .sub_path = LOG_NAME, .data = "earlier\n" });
    try h2.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(h2.arena.allocator(), h2.config(), h2.sc.signals(), h2.fw.worker(), &h2.stop);
    s.logf("later", .{});
    s.deinit();
    const log = try h2.file(LOG_NAME);
    try testing.expect(std.mem.startsWith(u8, log, "earlier\n"));
    try testing.expect(std.mem.indexOf(u8, log, "later\n") != null);
}

test "tune sched: a missing or mistyped reason or regime is a bad record, an explicit null is not" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const base = "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":null,\"window\":1,\"kept\":true";
    try testing.expect((try parseRec(a, base ++ ",\"reason\":5,\"calls_compared\":10}")) == null);
    try testing.expect((try parseRec(a, base ++ ",\"reason\":false,\"calls_compared\":10}")) == null);
    try testing.expect((try parseRec(a, base ++ ",\"calls_compared\":10}")) == null);
    try testing.expect((try parseRec(a, "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"window\":1,\"kept\":true,\"reason\":null,\"calls_compared\":10}")) == null);
    const ok = (try parseRec(a, base ++ ",\"reason\":null,\"calls_compared\":10}")) orelse return error.TestExpectedRecord;
    try testing.expect(ok.batch.reason == null and ok.batch.regime == null and ok.batch.calls_compared == 10);
    // through replay: the numeric-reason line leaves its candidate unchecked, the explicit null marks it identical
    const numeric = "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":null,\"window\":1,\"kept\":true,\"reason\":7,\"calls_compared\":10}";
    const explicit = "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=1\",\"regime\":null,\"window\":1,\"kept\":true,\"reason\":null,\"calls_compared\":10}";
    try writeProgress(h, &.{ numeric, explicit });
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.ks[0].id_done[0] and s.ks[0].cands[0].identical == null);
    try testing.expect(s.ks[0].id_done[1] and s.ks[0].cands[1].identical == true);
}

fn rawTimed(a: Allocator, diff: i64, hi: i64) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"phase\":\"confirm1\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\",\"window\":2,\"attempt\":1,\"kept\":true,\"reason\":null,\"ref_ns\":10000,\"cand_ns\":9500,\"diff_ns\":{d},\"q1_ns\":-600,\"q3_ns\":-400,\"lo_ns\":-700,\"hi_ns\":{d},\"calls_compared\":40}}", .{ diff, hi });
}

test "tune sched: timing fields outside the safe range are a bad record, the range's ends are not" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    try testing.expect((try parseRec(a, try rawTimed(a, std.math.minInt(i64), -300))) == null);
    try testing.expect((try parseRec(a, try rawTimed(a, -TIME_BOUND - 1, -300))) == null);
    try testing.expect((try parseRec(a, try rawTimed(a, -500, TIME_BOUND + 1))) == null);
    // a partial set of timing fields, a negative count and a mistyped attempt are bad too
    try testing.expect((try parseRec(a, "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":\"indep\",\"window\":1,\"kept\":true,\"reason\":null,\"ref_ns\":10000}")) == null);
    try testing.expect((try parseRec(a, "{\"phase\":\"identity\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":null,\"window\":1,\"kept\":true,\"reason\":null,\"calls_compared\":-1}")) == null);
    try testing.expect((try parseRec(a, "{\"phase\":\"final\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=0\",\"regime\":null,\"window\":1,\"kept\":true,\"reason\":null,\"attempt\":\"1\"}")) == null);
    // the complement: the range's end parses, and judging it does the arithmetic without overflow
    const edge = (try parseRec(a, try rawTimed(a, -TIME_BOUND, -TIME_BOUND))) orelse return error.TestExpectedRecord;
    const st = edge.batch.stats.?;
    try testing.expectEqual(-TIME_BOUND, st.diff_ns);
    try testing.expectEqual(tune_stats.Confirm.ok, tune_stats.judgeConfirm(.{ st, st }));
}

test "tune sched: a marker whose finalist misses a regime's final batch is not trusted" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const marker = "{\"stage_done\":\"final\",\"attempt\":1,\"window\":2}";
    // a finished screen with two gate/up finalists, then their final stage
    var lines = try screenedLines(a, &.{ 0, 1 }, false);
    const prefix = lines.items.len;
    try lines.appendSlice(a, &[_][]const u8{
        try rawAttempt(a, "final", "sg=1 ax=0", "indep", 2, 1),
        try rawAttempt(a, "final", "sg=1 ax=0", "shared", 2, 1),
        try rawAttempt(a, "final", "sg=1 ax=1", "indep", 2, 1), // the second finalist's shared batch is missing
        try rawAttempt(a, "confirm1", "sg=1 ax=0", "indep", 2, 1),
        try rawAttempt(a, "confirm1", "sg=1 ax=0", "shared", 2, 1),
        marker,
    });
    try writeProgress(h, lines.items);
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.final_done);
    // the complement: with that batch present (before the marker) the marker is restored, and the pick is the one
    // confirm1 timed
    try lines.insert(a, prefix + 3, try rawAttempt(a, "final", "sg=1 ax=1", "shared", 2, 1));
    try writeProgress(h, lines.items);
    const s2 = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s2.deinit();
    try testing.expect(s2.final_done);
    try testing.expectEqual(@as(usize, 0), s2.ks[0].pick);
}

test "tune sched: the summary and replay agree on identity exclusions" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    try writeProgress(h, &.{
        try rawLine(a, "identity", "sg=1 ax=0", null, true, "bits differ in case 99", 0), // a reason no pass writes
        try rawLine(a, "identity", "sg=1 ax=1", null, true, "bits differ in case 3", 0),
        try rawLine(a, "identity", "sg=99 ax=99", null, true, "bits differ in case 3", 0), // no such candidate
        try rawLine(a, "identity", "sg=1 ax=1", null, true, "launch refused: no pipeline", 0), // the later line wins
    });
    const p = try loadProgress(a, testing.io, h.tmp.dir, TEST_KEY);
    const sum = try summarize(a, p);
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    // the bad reason: neither lists the candidate, and replay checks it again
    try testing.expect(!s.ks[0].id_done[0] and s.ks[0].reason[0] == null);
    for (sum.excluded) |e| try testing.expect(!std.mem.eql(u8, e.cand, "sg=1 ax=0"));
    // the good one: both list it once, with the same reason
    try testing.expectEqual(@as(usize, 1), sum.excluded.len);
    try testing.expectEqualStrings("sg=1 ax=1", sum.excluded[0].cand);
    try testing.expectEqualStrings(s.ks[0].reason[1].?, sum.excluded[0].reason);
    try testing.expectEqualStrings("launch refused: no pipeline", sum.excluded[0].reason);
}

/// Input 250 ms into the first window, while the first identity check (ten cases of 100 ms) runs.
const INPUT_IN_IDENTITY = "{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":4000,\"hid_idle_s\":0}\n";

test "tune sched: without its thread the monitor ticks during an identity check" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(INPUT_IN_IDENTITY, baseSpec, idAllIdentical);
    h.fw.id_case_ms = 100;
    _ = try finishedOutcome(try h.run());
    // the check was cut mid-way by the monitor, not finished and judged afterwards
    const prog = try h.file(PROGRESS_NAME);
    try testing.expectEqual(@as(usize, 1), countLines(prog, "\"phase\":\"identity\"", "\"kept\":false,\"reason\":\"input\""));
    try testing.expect(h.fw.identity_calls == 59);
    // the complement: an idle Mac lets every check run its ten cases
    const h2 = try Harness.create();
    defer h2.destroy();
    try h2.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    h2.fw.id_case_ms = 100;
    _ = try finishedOutcome(try h2.run());
    try testing.expectEqual(@as(usize, 0), countLines(try h2.file(PROGRESS_NAME), "\"phase\":\"identity\"", "\"kept\":false"));
    try testing.expectEqual(@as(u32, 58 * IDENTITY_CASES), h2.fw.id_cases);
}

test "tune sched: without its thread the monitor ticks during a verify check" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(INPUT_IN_IDENTITY, baseSpec, idAllIdentical);
    h.fw.id_case_ms = 100;
    var cfg = h.config();
    cfg.mode = .verify;
    cfg.verify_plan = .{ .gateup = kernel_table.gateUpCandidates()[3], .down = null };
    const r = try verify(h.arena.allocator(), cfg, h.sc.signals(), h.fw.worker(), &h.stop);
    try testing.expectEqual(@as(u8, 1), r.identical.kernels);
    // the first check was cut by the input and ran again in the next window
    try testing.expectEqual(@as(u32, 2), h.fw.identity_calls);
    try testing.expect(h.fw.id_cases < 2 * IDENTITY_CASES);
    // the complement: an idle Mac checks the layout once
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    h.fw.id_case_ms = 100;
    const r2 = try verify(h.arena.allocator(), cfg, h.sc.signals(), h.fw.worker(), &h.stop);
    try testing.expectEqual(@as(u8, 1), r2.identical.kernels);
    try testing.expectEqual(@as(u32, 1), h.fw.identity_calls);
}

/// A `Signals` over another whose absolute clock jumps `JOIN_MARK_NS` when the monitor thread's sleep ends, a sleep that
/// ends only once the calling thread has read the clock after the worker returned. A duration that took in the monitor's
/// join holds the jump; one that ends with the worker cannot, however long the worker's real sleeps took.
const JoinMarkSignals = struct {
    inner: Signals,
    main_id: std.Thread.Id,
    worker_returned: std.atomic.Value(bool) = .init(false),
    end_read: std.atomic.Value(bool) = .init(false),
    offset_ns: std.atomic.Value(u64) = .init(0),

    const JOIN_MARK_NS: u64 = 1000 * NS_PER_S;

    fn at(p: *anyopaque) *JoinMarkSignals {
        return @ptrCast(@alignCast(p));
    }
    fn hid(p: *anyopaque) ?u64 {
        return at(p).inner.hidIdleNs();
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
        const self = at(p);
        const t = self.inner.absTimeNs() + self.offset_ns.load(.acquire);
        if (std.Thread.getCurrentId() == self.main_id and self.worker_returned.load(.acquire)) self.end_read.store(true, .release);
        return t;
    }
    fn contT(p: *anyopaque) u64 {
        return at(p).inner.contTimeNs();
    }
    fn waitT(p: *anyopaque, ns: u64) void {
        const self = at(p);
        if (std.Thread.getCurrentId() == self.main_id) return self.inner.waitNs(ns);
        // the monitor thread: its sleep lasts until the batch's end was read, and then counts as the jump
        while (!self.end_read.load(.acquire)) self.inner.waitNs(NS_PER_MS);
        _ = self.offset_ns.fetchAdd(JOIN_MARK_NS, .acq_rel);
    }

    const vt: tune_signals.VTable = .{ .hidIdleNs = hid, .onAc = ac, .lowPower = lowPow, .thermalLevel = therm, .gpuPct = gpuP, .cpuTicks = cpuT, .memPressure = memP, .absTimeNs = absT, .contTimeNs = contT, .waitNs = waitT };

    fn signals(self: *JoinMarkSignals) Signals {
        return .{ .ctx = self, .vt = &vt };
    }
};

/// A worker over another that sets `flag` when a timing batch returns.
const ReturnFlagWorker = struct {
    inner: Worker,
    flag: *std.atomic.Value(bool),

    fn at(p: *anyopaque) *ReturnFlagWorker {
        return @ptrCast(@alignCast(p));
    }
    fn openW(p: *anyopaque) anyerror!void {
        return at(p).inner.openWindow();
    }
    fn closeW(p: *anyopaque) void {
        at(p).inner.closeWindow();
    }
    fn ident(p: *anyopaque, k: Kernel, c: Cand, flag_stop: *const std.atomic.Value(bool), h: PairHook) IdentityOutcome {
        return at(p).inner.identity(k, c, flag_stop, h);
    }
    fn timeB(p: *anyopaque, k: Kernel, c: Cand, r: Regime, seed: u64, flag_stop: *const std.atomic.Value(bool), h: PairHook) BatchOutcome {
        const self = at(p);
        const out = self.inner.timeBatch(k, c, r, seed, flag_stop, h);
        self.flag.store(true, .release);
        return out;
    }

    const vt: Worker.VTable = .{ .openWindow = openW, .closeWindow = closeW, .identity = ident, .timeBatch = timeB };

    fn worker(self: *ReturnFlagWorker) Worker {
        return .{ .ctx = self, .vt = &vt };
    }
};

test "tune sched: a batch's duration ends with its worker, not with the monitor's join, however long the pairs take" {
    // 25 pairs of 2 ms, and of 12 ms (past the monitor's first sleep and any fixed deadline)
    for ([_]u64{ 2, 12 }) |pair_ms| {
        var ext: std.atomic.Value(bool) = .init(false);
        var sc = try ScriptedSignals.parse(testing.allocator, IDLE_SCRIPT, .real);
        defer sc.deinit();
        var fw: FakeWorker = .{ .gu = kernel_table.gateUpCandidates(), .dn = kernel_table.downCandidates(), .sc = &sc, .ext_stop = &ext, .alloc = testing.allocator, .pair_ns = pair_ms * NS_PER_MS };
        defer fw.calls.deinit(testing.allocator);
        var js: JoinMarkSignals = .{ .inner = sc.signals(), .main_id = std.Thread.getCurrentId() };
        var rw: ReturnFlagWorker = .{ .inner = fw.worker(), .flag = &js.worker_returned };
        var stop: std.atomic.Value(bool) = .init(false);
        const sig = js.signals();
        const t0 = sig.absTimeNs();
        const res = runBatchWith(sig, rw.worker(), .gateup, .reference, .indep, 1, &stop, null, .thread);
        const elapsed = sig.absTimeNs() - t0;
        try testing.expect(res.outcome == .done and res.discard == null);
        // the call waited for the monitor's join: the jump at the end of its sleep is inside the call's time ...
        try testing.expect(elapsed >= JoinMarkSignals.JOIN_MARK_NS);
        // ... and outside the batch's duration, which still holds the worker's 25 pairs
        try testing.expect(res.duration_ns < JoinMarkSignals.JOIN_MARK_NS);
        try testing.expect(res.duration_ns >= 25 * pair_ms * NS_PER_MS);
    }
}

// ── Fix round 5 tests ──

/// One kept progress line of `k`'s candidate `idx`: an identity line (`regime` null, 10 comparisons) or a timing line
/// whose difference is `diff` over a reference of 10,000 ns, with `attempt` when given.
fn rawRec(a: Allocator, phase: []const u8, k: Kernel, idx: usize, regime: ?[]const u8, window: u32, attempt: ?u32, diff: i64) ![]u8 {
    var b: [48]u8 = undefined;
    const cand = candText(&b, k, idx);
    var l: std.ArrayList(u8) = .empty;
    try l.print(a, "{{\"phase\":\"{s}\",\"kernel\":\"{s}\",\"cand\":\"{s}\",", .{ phase, @tagName(k), cand });
    if (regime) |rg| {
        try l.print(a, "\"regime\":\"{s}\",\"window\":{d},\"kept\":true,\"reason\":null,\"ref_ns\":10000,\"cand_ns\":{d},\"diff_ns\":{d},\"q1_ns\":{d},\"q3_ns\":{d},\"lo_ns\":{d},\"hi_ns\":{d},\"calls_compared\":40", .{ rg, window, 10_000 + diff, diff, diff - 100, diff + 100, diff - 200, diff + 200 });
    } else try l.print(a, "\"regime\":null,\"window\":{d},\"kept\":true,\"reason\":null,\"calls_compared\":10", .{window});
    if (attempt) |att| try l.print(a, ",\"attempt\":{d}", .{att});
    try l.append(a, '}');
    return l.toOwnedSlice(a);
}

/// A finished screen in window 1: the 58 identity lines, then the 116 screen lines, gate/up's candidates in `gains` at -5%
/// and every other at +1%. `drop_last` leaves out the last screen line (down's last candidate, shared).
fn screenedLines(a: Allocator, gains: []const usize, drop_last: bool) !std.ArrayList([]const u8) {
    var out: std.ArrayList([]const u8) = .empty;
    const counts = [2]usize{ kernel_table.GATEUP_CANDIDATES, kernel_table.DOWN_CANDIDATES };
    for (0..2) |k| {
        for (0..counts[k]) |i| try out.append(a, try rawRec(a, "identity", @enumFromInt(k), i, null, 1, null, 0));
    }
    for (0..2) |k| {
        for (0..counts[k]) |i| {
            var d: i64 = 100;
            if (k == 0 and std.mem.indexOfScalar(usize, gains, i) != null) d = -500;
            for ([_][]const u8{ "indep", "shared" }) |rg| try out.append(a, try rawRec(a, "screen", @enumFromInt(k), i, rg, 1, null, d));
        }
    }
    if (drop_last) _ = out.pop();
    return out;
}

const MARKER_W2 = "{\"stage_done\":\"final\",\"attempt\":1,\"window\":2}";

test "tune sched: a marker before the screen finished is not trusted" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const marker = "{\"stage_done\":\"final\",\"attempt\":1,\"window\":1}";
    // a header and a lone marker: nothing is decided, and the run does every check
    try writeProgress(h, &.{marker});
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    {
        const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
        defer s.deinit();
        try testing.expect(!s.final_done and !s.ks[0].decided() and !s.ks[1].decided());
    }
    const out = try finishedOutcome(try h.run());
    try testing.expectEqual(@as(u32, 58), h.fw.identity_calls);
    try testing.expectEqual(@as(u32, 4 + 116), h.fw.batch_calls);
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), out.served_reasons[0]);
    // one screen line missing: not trusted either
    var cut = try screenedLines(a, &.{}, true);
    try cut.append(a, marker);
    try writeProgress(h, cut.items);
    const s2 = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s2.deinit();
    try testing.expect(!s2.final_done and !s2.ks[1].decided());
    // the complement: the whole screen, no gain anywhere, and the marker is restored with the screen's reasons
    var whole = try screenedLines(a, &.{}, false);
    try whole.append(a, marker);
    try writeProgress(h, whole.items);
    const s3 = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s3.deinit();
    try testing.expect(s3.final_done);
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), s3.ks[0].served);
    try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), s3.ks[1].served);
}

test "tune sched: a marker missing a finalist's final batches is not trusted" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    // the screen selects A and B; only A's final batches, A's confirm1 pair and the marker are in the file
    var lines = try screenedLines(a, &.{ 0, 1 }, false);
    const prefix = lines.items.len;
    try lines.appendSlice(a, &[_][]const u8{
        try rawRec(a, "final", .gateup, 0, "indep", 2, 1, -500),
        try rawRec(a, "final", .gateup, 0, "shared", 2, 1, -500),
        try rawRec(a, "confirm1", .gateup, 0, "indep", 2, 1, -500),
        try rawRec(a, "confirm1", .gateup, 0, "shared", 2, 1, -500),
        MARKER_W2,
    });
    try writeProgress(h, lines.items);
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.final_done and !s.ks[0].decided());
    // the complement: B's two final batches, slower than A's, and the marker is restored with A as the pick
    try lines.insertSlice(a, prefix + 2, &[_][]const u8{
        try rawRec(a, "final", .gateup, 1, "indep", 2, 1, -300),
        try rawRec(a, "final", .gateup, 1, "shared", 2, 1, -300),
    });
    try writeProgress(h, lines.items);
    const s2 = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s2.deinit();
    try testing.expect(s2.final_done);
    try testing.expectEqual(@as(usize, 0), s2.ks[0].pick);
}

test "tune sched: a marker whose confirm1 names another candidate than the pick rule is not trusted" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    // B's finals are faster than A's, but confirm1 timed A
    var lines = try screenedLines(a, &.{ 0, 1 }, false);
    const at_c1 = lines.items.len + 4;
    try lines.appendSlice(a, &[_][]const u8{
        try rawRec(a, "final", .gateup, 0, "indep", 2, 1, -300),
        try rawRec(a, "final", .gateup, 0, "shared", 2, 1, -300),
        try rawRec(a, "final", .gateup, 1, "indep", 2, 1, -800),
        try rawRec(a, "final", .gateup, 1, "shared", 2, 1, -800),
        try rawRec(a, "confirm1", .gateup, 0, "indep", 2, 1, -500),
        try rawRec(a, "confirm1", .gateup, 0, "shared", 2, 1, -500),
        MARKER_W2,
    });
    try writeProgress(h, lines.items);
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s.deinit();
    try testing.expect(!s.final_done and !s.ks[0].decided());
    // the complement: confirm1 timed B, the pick the rule gives, and the marker is restored with B
    lines.items[at_c1] = try rawRec(a, "confirm1", .gateup, 1, "indep", 2, 1, -500);
    lines.items[at_c1 + 1] = try rawRec(a, "confirm1", .gateup, 1, "shared", 2, 1, -500);
    try writeProgress(h, lines.items);
    const s2 = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
    defer s2.deinit();
    try testing.expect(s2.final_done);
    try testing.expectEqual(@as(usize, 1), s2.ks[0].pick);
}

test "tune sched: the summary counts the windows replay counts" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const Case = struct { windows: []const u32, want: u32 };
    // windows 1 and 3 only: two windows, not the largest id; the complement 1, 2 and 3: three
    for ([_]Case{ .{ .windows = &.{ 1, 3 }, .want = 2 }, .{ .windows = &.{ 1, 2, 3 }, .want = 3 } }) |c| {
        var lines: std.ArrayList([]const u8) = .empty;
        for (c.windows) |w| try lines.append(a, try rawRec(a, "screen", .gateup, 0, "indep", w, null, 100));
        try writeProgress(h, lines.items);
        const sum = try summarize(a, try loadProgress(a, testing.io, h.tmp.dir, TEST_KEY));
        const s = try Sched.init(a, h.config(), h.sc.signals(), h.fw.worker(), &h.stop);
        defer s.deinit();
        try testing.expectEqual(c.want, sum.windows_used);
        try testing.expectEqual(s.windows_used, sum.windows_used);
    }
}

fn calBitsOnce(in: SpecIn) Spec {
    var s = baseSpec(in);
    if (in.idx == null and in.kernel == .gateup and in.regime == .indep and in.nth == 1) s.bits = true;
    return s;
}

test "tune sched: a calibration batch whose bits differ is discarded, never a calibration result" {
    // the first A/A batch differs: discarded, the window closes uncalibrated; the complement: an A/A batch that agrees
    for ([_]SpecFn{ calBitsOnce, baseSpec }, [_]bool{ true, false }) |f, differs| {
        const h = try Harness.create();
        defer h.destroy();
        try h.start(IDLE_SCRIPT, f, idAllIdentical);
        const out = try finishedOutcome(try h.run());
        const prog = try h.file(PROGRESS_NAME);
        try testing.expectEqual(@as(usize, if (differs) 1 else 0), countLines(prog, "\"phase\":\"cal\"", "\"kept\":false,\"reason\":\"bits differ during timing\""));
        try testing.expectEqual(@as(usize, if (differs) 1 else 4), countLines(prog, "\"window\":1,", "\"phase\":\"cal\""));
        try testing.expectEqual(@as(usize, if (differs) 0 else 116), countLines(prog, "\"window\":1,", "\"phase\":\"screen\""));
        // the reference's difference excludes no candidate
        try testing.expectEqual(@as(usize, 0), out.identity.excluded.len);
        try testing.expectEqual(@as(?ServedReason, .no_gain_in_screen), out.served_reasons[0]);
    }
}

/// The lowest descriptor number free in this process (the next `open` gets it).
fn lowestFreeFd() !c_int {
    const fd = std.c.open("/dev/null", .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestCannotOpen;
    _ = std.c.close(fd);
    return fd;
}

test "tune sched: a start that fails after opening the progress file closes it, one that succeeds keeps it to deinit" {
    const h = try Harness.create();
    defer h.destroy();
    try h.start(IDLE_SCRIPT, baseSpec, idAllIdentical);
    const free0 = try lowestFreeFd();
    var after_open: u32 = 0;
    var fail_at: usize = 0;
    while (true) : (fail_at += 1) {
        var failing = std.testing.FailingAllocator.init(h.arena.allocator(), .{ .fail_index = fail_at });
        if (Sched.init(failing.allocator(), h.config(), h.sc.signals(), h.fw.worker(), &h.stop)) |s| {
            // the complement: a start that succeeds holds its descriptors until deinit
            try testing.expect((try lowestFreeFd()) != free0);
            s.deinit();
            break;
        } else |e| {
            try testing.expectEqual(error.OutOfMemory, e);
            try testing.expectEqual(free0, try lowestFreeFd());
            // the progress file exists only once a start opened it: this failure came after the open
            const probe = std.c.openat(h.tmp.dir.handle, PROGRESS_NAME, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
            if (probe >= 0) {
                _ = std.c.close(probe);
                after_open += 1;
            }
        }
    }
    try testing.expect(after_open > 0);
    try testing.expectEqual(free0, try lowestFreeFd());
}

test "tune sched: running out of memory while reading progress is an error, a malformed header a restart" {
    const h = try Harness.create();
    defer h.destroy();
    const a = h.arena.allocator();
    const line = try rawRec(a, "screen", .gateup, 0, "indep", 1, null, 100);
    try writeProgress(h, &.{line});
    // every allocation failing in turn: an error each time, never a restart or a dropped record
    var fail_at: usize = 0;
    while (true) : (fail_at += 1) {
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_at });
        if (loadProgress(failing.allocator(), testing.io, h.tmp.dir, TEST_KEY)) |p| {
            try testing.expect(!p.restarted);
            try testing.expectEqual(@as(usize, 1), p.recs.len);
            break;
        } else |e| try testing.expectEqual(error.OutOfMemory, e);
    }
    try testing.expect(fail_at > 1);
    // the complement: a header that is not JSON, or JSON of another shape, restarts the run
    for ([_][]const u8{ "{\"progress\":1,", "[1]" }) |bad| {
        try h.tmp.dir.writeFile(testing.io, .{ .sub_path = PROGRESS_NAME, .data = try std.fmt.allocPrint(a, "{s}\n{s}\n", .{ bad, line }) });
        const p = try loadProgress(a, testing.io, h.tmp.dir, TEST_KEY);
        try testing.expect(p.restarted);
    }
}
