//! The idle conditions of `lesto tune` (spec 4.1 and 4.2): what the Mac is doing, read through `Signals`, a struct of
//! function pointers with two implementations: `RealSignals` (IOKit, notify, Foundation, Mach) and `ScriptedSignals`,
//! which replays a JSON-lines script on a manual or a real clock. The scheduler and its tests see only `Signals`.
const std = @import("std");
const builtin = @import("builtin");
const status = @import("status.zig");

pub const TUNE_GPU_PCT_MAX: u32 = 10;
pub const TUNE_CPU_PCT_MAX: u32 = 25;
pub const TUNE_SIGNAL_SPAN_S: u32 = 2;
/// GPU readings of the open check, and the gap between two of them.
const GPU_READINGS: u32 = 8;
const GPU_READING_GAP_NS: u64 = 250 * std.time.ns_per_ms;

const is_macos = builtin.os.tag == .macos;

// ── The interface ──

pub const VTable = struct {
    hidIdleNs: *const fn (*anyopaque) ?u64,
    onAc: *const fn (*anyopaque) ?bool,
    lowPower: *const fn (*anyopaque) ?bool,
    thermalLevel: *const fn (*anyopaque) ?u32,
    gpuPct: *const fn (*anyopaque) ?u32,
    cpuTicks: *const fn (*anyopaque) ?[4]u64,
    memPressure: *const fn (*anyopaque) u32,
    absTimeNs: *const fn (*anyopaque) u64,
    contTimeNs: *const fn (*anyopaque) u64,
    /// Waits `ns` on this source's clock (a sleep for the real one, an advance of the manual clock for a scripted one), so
    /// scheduler tests on a fake clock run in milliseconds.
    waitNs: *const fn (*anyopaque, u64) void,
};

pub const Signals = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub fn hidIdleNs(self: Signals) ?u64 {
        return self.vt.hidIdleNs(self.ctx);
    }
    pub fn onAc(self: Signals) ?bool {
        return self.vt.onAc(self.ctx);
    }
    pub fn lowPower(self: Signals) ?bool {
        return self.vt.lowPower(self.ctx);
    }
    pub fn thermalLevel(self: Signals) ?u32 {
        return self.vt.thermalLevel(self.ctx);
    }
    pub fn gpuPct(self: Signals) ?u32 {
        return self.vt.gpuPct(self.ctx);
    }
    pub fn cpuTicks(self: Signals) ?[4]u64 {
        return self.vt.cpuTicks(self.ctx);
    }
    /// 0 for unknown, as `status.memoryPressureLevel()`.
    pub fn memPressure(self: Signals) u32 {
        return self.vt.memPressure(self.ctx);
    }
    pub fn absTimeNs(self: Signals) u64 {
        return self.vt.absTimeNs(self.ctx);
    }
    pub fn contTimeNs(self: Signals) u64 {
        return self.vt.contTimeNs(self.ctx);
    }
    pub fn waitNs(self: Signals, ns: u64) void {
        self.vt.waitNs(self.ctx, ns);
    }
};

/// One wrapper per method, so a type with the ten methods is a `Signals` source.
fn vtableFor(comptime T: type) VTable {
    const W = struct {
        fn at(p: *anyopaque) *T {
            return @ptrCast(@alignCast(p));
        }
        fn hid(p: *anyopaque) ?u64 {
            return at(p).hidIdleNs();
        }
        fn ac(p: *anyopaque) ?bool {
            return at(p).onAc();
        }
        fn lp(p: *anyopaque) ?bool {
            return at(p).lowPower();
        }
        fn th(p: *anyopaque) ?u32 {
            return at(p).thermalLevel();
        }
        fn gpu(p: *anyopaque) ?u32 {
            return at(p).gpuPct();
        }
        fn cpu(p: *anyopaque) ?[4]u64 {
            return at(p).cpuTicks();
        }
        fn mem(p: *anyopaque) u32 {
            return at(p).memPressure();
        }
        fn abs(p: *anyopaque) u64 {
            return at(p).absTimeNs();
        }
        fn cont(p: *anyopaque) u64 {
            return at(p).contTimeNs();
        }
        fn wait(p: *anyopaque, ns: u64) void {
            at(p).waitNs(ns);
        }
    };
    return .{
        .hidIdleNs = W.hid,
        .onAc = W.ac,
        .lowPower = W.lp,
        .thermalLevel = W.th,
        .gpuPct = W.gpu,
        .cpuTicks = W.cpu,
        .memPressure = W.mem,
        .absTimeNs = W.abs,
        .contTimeNs = W.cont,
        .waitNs = W.wait,
    };
}

// ── The real source ──

extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn IORegistryEntryCreateCFProperty(entry: u32, key: ?*const anyopaque, alloc: ?*anyopaque, opts: u32) ?*const anyopaque;
extern "c" fn IOPSGetProvidingPowerSourceType(snapshot: ?*const anyopaque) ?*const anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
extern "c" fn CFStringGetCString(s: ?*const anyopaque, buf: [*]u8, len: isize, enc: u32) u8;
extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;
extern "c" fn notify_register_check(name: [*:0]const u8, out_token: *c_int) u32;
extern "c" fn notify_get_state(token: c_int, state: *u64) u32;
extern "c" fn notify_cancel(token: c_int) u32;
extern "c" fn mach_absolute_time() u64;
extern "c" fn mach_continuous_time() u64;
extern "c" fn mach_timebase_info(info: *MachTimebase) c_int;
extern "c" fn lesto_low_power_mode() c_int;

const MachTimebase = extern struct { numer: u32, denom: u32 };
const CF_UTF8: u32 = 0x08000100;

var timebase: MachTimebase = .{ .numer = 0, .denom = 0 };

fn machToNs(t: u64) u64 {
    if (timebase.denom == 0) {
        var tb: MachTimebase = .{ .numer = 1, .denom = 1 };
        if (mach_timebase_info(&tb) != 0 or tb.denom == 0) tb = .{ .numer = 1, .denom = 1 };
        timebase = tb; // a benign race: every thread computes the same pair
    }
    return @intCast(@as(u128, t) * timebase.numer / timebase.denom);
}

fn sleepNs(ns: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

pub const RealSignals = struct {
    thermal_token: ?c_int = null,

    pub fn deinit(self: *RealSignals) void {
        if (comptime !is_macos) return;
        if (self.thermal_token) |t| _ = notify_cancel(t);
        self.thermal_token = null;
    }

    /// `HIDIdleTime` of `IOHIDSystem`: nanoseconds since the last keyboard, mouse or trackpad input. Null when the service,
    /// the property or the number is not readable (a headless Mac, an agent context that cannot read it).
    pub fn hidIdleNs(_: *RealSignals) ?u64 {
        if (comptime !is_macos) return null;
        const matching = IOServiceMatching("IOHIDSystem") orelse return null;
        var iter: u32 = 0;
        if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return null;
        defer _ = IOObjectRelease(iter);
        const entry = IOIteratorNext(iter);
        if (entry == 0) return null;
        defer _ = IOObjectRelease(entry);
        const key = CFStringCreateWithCString(null, "HIDIdleTime", CF_UTF8) orelse return null;
        defer CFRelease(key);
        const num = IORegistryEntryCreateCFProperty(entry, key, null, 0) orelse return null;
        defer CFRelease(num);
        var v: i64 = 0;
        if (CFNumberGetValue(num, 4, @ptrCast(&v)) == 0 or v < 0) return null;
        return @intCast(v);
    }

    pub fn onAc(_: *RealSignals) ?bool {
        if (comptime !is_macos) return null;
        const src = IOPSGetProvidingPowerSourceType(null) orelse return null;
        defer CFRelease(src);
        var buf: [64]u8 = undefined;
        if (CFStringGetCString(src, &buf, buf.len, CF_UTF8) == 0) return null;
        return std.mem.eql(u8, std.mem.sliceTo(&buf, 0), "AC Power");
    }

    pub fn lowPower(_: *RealSignals) ?bool {
        if (comptime !is_macos) return null;
        return switch (lesto_low_power_mode()) {
            0 => false,
            1 => true,
            else => null,
        };
    }

    pub fn thermalLevel(self: *RealSignals) ?u32 {
        if (comptime !is_macos) return null;
        if (self.thermal_token == null) {
            var tok: c_int = 0;
            if (notify_register_check("com.apple.system.thermalpressurelevel", &tok) != 0) return null;
            self.thermal_token = tok;
        }
        var state: u64 = 0;
        if (notify_get_state(self.thermal_token.?, &state) != 0) return null;
        return @intCast(@min(state, std.math.maxInt(u32)));
    }

    pub fn gpuPct(_: *RealSignals) ?u32 {
        return status.gpuUtilPct();
    }

    pub fn cpuTicks(_: *RealSignals) ?[4]u64 {
        return status.cpuTicks();
    }

    pub fn memPressure(_: *RealSignals) u32 {
        return status.memoryPressureLevel();
    }

    pub fn absTimeNs(_: *RealSignals) u64 {
        if (comptime !is_macos) return 0;
        return machToNs(mach_absolute_time());
    }

    pub fn contTimeNs(_: *RealSignals) u64 {
        if (comptime !is_macos) return 0;
        return machToNs(mach_continuous_time());
    }

    pub fn waitNs(_: *RealSignals, ns: u64) void {
        sleepNs(ns);
    }

    pub fn signals(self: *RealSignals) Signals {
        const Holder = struct {
            const vt = vtableFor(RealSignals);
        };
        return .{ .ctx = self, .vt = &Holder.vt };
    }
};

// ── The scripted source ──

pub const Clock = enum { manual, real };

pub const ParseErrorKind = enum { not_json, not_an_object, unknown_field, missing_at_ms, bad_value, time_goes_back };

pub const ParseError = struct {
    line: usize = 0,
    kind: ParseErrorKind = .not_json,
    /// The field the error is about (unknown_field and bad_value), cut to 32 bytes.
    field: [32]u8 = @splat(0),

    pub fn fieldName(self: *const ParseError) []const u8 {
        return std.mem.sliceTo(&self.field, 0);
    }
};

/// One script line. A field that is `null` in the line (as opposed to absent) reads as unknown from that line on.
const Line = struct {
    at_ms: u64,
    hid_idle_s: ??f64 = null,
    ac: ??bool = null,
    low_power: ??bool = null,
    thermal: ??u32 = null,
    gpu_pct: ??u32 = null,
    cpu_pct: ??u32 = null,
    mem_level: ?u32 = null,
    sleep_ms: ?u64 = null,
};

/// CPU ticks are counted at 1,000 a second, offset so a first reading is not zero.
const CPU_TICK_BASE: u64 = 1_000_000;

pub const ScriptedSignals = struct {
    lines: []Line,
    clock: Clock,
    alloc: std.mem.Allocator,
    now_ms: u64 = 0, // manual clock
    start_ns: u64 = 0, // real clock: the tune's start

    pub fn deinit(self: *ScriptedSignals) void {
        self.alloc.free(self.lines);
        self.lines = &.{};
    }

    pub fn parse(alloc: std.mem.Allocator, text: []const u8, clock: Clock) !ScriptedSignals {
        var perr: ParseError = .{};
        return parseWithError(alloc, text, clock, &perr);
    }

    /// `err` names the first bad line (1-based) and what is wrong with it when the result is `error.BadScript`.
    pub fn parseWithError(alloc: std.mem.Allocator, text: []const u8, clock: Clock, err: *ParseError) !ScriptedSignals {
        var out: std.ArrayList(Line) = .empty;
        errdefer out.deinit(alloc);
        var line_no: usize = 0;
        var last_at: u64 = 0;
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |raw| {
            line_no += 1;
            const t = std.mem.trim(u8, raw, " \t\r");
            if (t.len == 0) continue;
            const line = parseLine(alloc, t, line_no, err) catch |e| switch (e) {
                error.BadScript => return error.BadScript,
                else => |other| return other,
            };
            if (line.at_ms < last_at) {
                err.* = .{ .line = line_no, .kind = .time_goes_back };
                return error.BadScript;
            }
            last_at = line.at_ms;
            try out.append(alloc, line);
        }
        var s: ScriptedSignals = .{ .lines = try out.toOwnedSlice(alloc), .clock = clock, .alloc = alloc };
        if (clock == .real) s.start_ns = if (is_macos) machToNs(mach_absolute_time()) else 0;
        return s;
    }

    pub fn advanceMs(self: *ScriptedSignals, ms: u64) void {
        self.now_ms += ms;
    }

    fn tMs(self: *const ScriptedSignals) u64 {
        return switch (self.clock) {
            .manual => self.now_ms,
            .real => if (is_macos) (machToNs(mach_absolute_time()) -| self.start_ns) / std.time.ns_per_ms else 0,
        };
    }

    const State = struct {
        hid: ?f64 = 0,
        hid_at: u64 = 0,
        ac: ?bool = true,
        low_power: ?bool = false,
        thermal: ?u32 = 0,
        gpu: ?u32 = 0,
        mem: u32 = 1,
        gap_ms: u64 = 0,
        cpu_share: ?u32 = 0,
        busy_x100: u128 = 0, // busy ticks times 100, summed over the segments
    };

    fn stateAt(self: *const ScriptedSignals, t: u64) State {
        var s: State = .{};
        var seg_from: u64 = 0;
        for (self.lines) |ln| {
            if (ln.at_ms > t) break;
            s.busy_x100 += @as(u128, ln.at_ms - seg_from) * (s.cpu_share orelse 0);
            seg_from = ln.at_ms;
            if (ln.hid_idle_s) |v| {
                s.hid = v;
                s.hid_at = ln.at_ms;
            }
            if (ln.ac) |v| s.ac = v;
            if (ln.low_power) |v| s.low_power = v;
            if (ln.thermal) |v| s.thermal = v;
            if (ln.gpu_pct) |v| s.gpu = v;
            if (ln.cpu_pct) |v| s.cpu_share = v;
            if (ln.mem_level) |v| s.mem = v;
            if (ln.sleep_ms) |v| s.gap_ms += v;
        }
        s.busy_x100 += @as(u128, t - seg_from) * (s.cpu_share orelse 0);
        return s;
    }

    pub fn hidIdleNs(self: *ScriptedSignals) ?u64 {
        const t = self.tMs();
        const s = self.stateAt(t);
        const v = s.hid orelse return null;
        const base: u64 = @intFromFloat(v * 1e9);
        return base + (t - s.hid_at) * std.time.ns_per_ms;
    }

    pub fn onAc(self: *ScriptedSignals) ?bool {
        return self.stateAt(self.tMs()).ac;
    }

    pub fn lowPower(self: *ScriptedSignals) ?bool {
        return self.stateAt(self.tMs()).low_power;
    }

    pub fn thermalLevel(self: *ScriptedSignals) ?u32 {
        return self.stateAt(self.tMs()).thermal;
    }

    pub fn gpuPct(self: *ScriptedSignals) ?u32 {
        return self.stateAt(self.tMs()).gpu;
    }

    /// A tick pair consistent with the script's busy share: 1,000 ticks a second, `cpu_pct` of them user ticks.
    pub fn cpuTicks(self: *ScriptedSignals) ?[4]u64 {
        const t = self.tMs();
        const s = self.stateAt(t);
        if (s.cpu_share == null) return null;
        const busy: u64 = @intCast(s.busy_x100 / 100);
        return .{ CPU_TICK_BASE + busy, 0, CPU_TICK_BASE + (t -| busy), 0 };
    }

    pub fn memPressure(self: *ScriptedSignals) u32 {
        return self.stateAt(self.tMs()).mem;
    }

    pub fn absTimeNs(self: *ScriptedSignals) u64 {
        return switch (self.clock) {
            .manual => self.now_ms * std.time.ns_per_ms,
            .real => if (is_macos) machToNs(mach_absolute_time()) else 0,
        };
    }

    /// The continuous clock runs ahead of the absolute one by every `sleep_ms` reached so far.
    pub fn contTimeNs(self: *ScriptedSignals) u64 {
        const s = self.stateAt(self.tMs());
        return self.absTimeNs() + s.gap_ms * std.time.ns_per_ms;
    }

    pub fn waitNs(self: *ScriptedSignals, ns: u64) void {
        switch (self.clock) {
            .manual => self.now_ms += (ns + std.time.ns_per_ms - 1) / std.time.ns_per_ms,
            .real => sleepNs(ns),
        }
    }

    pub fn signals(self: *ScriptedSignals) Signals {
        const Holder = struct {
            const vt = vtableFor(ScriptedSignals);
        };
        return .{ .ctx = self, .vt = &Holder.vt };
    }
};

fn setField(err: *ParseError, line: usize, kind: ParseErrorKind, name: []const u8) error{BadScript} {
    err.* = .{ .line = line, .kind = kind };
    const n = @min(name.len, err.field.len - 1);
    @memcpy(err.field[0..n], name[0..n]);
    return error.BadScript;
}

fn jsonUint(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn parseLine(alloc: std.mem.Allocator, text: []const u8, line_no: usize, err: *ParseError) !Line {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, text, .{}) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            err.* = .{ .line = line_no, .kind = .not_json };
            return error.BadScript;
        },
    };
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => {
            err.* = .{ .line = line_no, .kind = .not_an_object };
            return error.BadScript;
        },
    };
    var line: Line = .{ .at_ms = 0 };
    var have_at = false;
    var it = obj.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const v = e.value_ptr.*;
        if (std.mem.eql(u8, k, "at_ms")) {
            line.at_ms = jsonUint(v) orelse return setField(err, line_no, .bad_value, k);
            have_at = true;
        } else if (std.mem.eql(u8, k, "hid_idle_s")) {
            line.hid_idle_s = switch (v) {
                .null => @as(?f64, null),
                .integer => |i| if (i >= 0 and i <= 1_000_000_000) @as(?f64, @floatFromInt(i)) else return setField(err, line_no, .bad_value, k),
                .float => |f| if (f >= 0 and f <= 1e9) @as(?f64, f) else return setField(err, line_no, .bad_value, k),
                else => return setField(err, line_no, .bad_value, k),
            };
        } else if (std.mem.eql(u8, k, "ac")) {
            line.ac = switch (v) {
                .null => @as(?bool, null),
                .bool => |b| @as(?bool, b),
                else => return setField(err, line_no, .bad_value, k),
            };
        } else if (std.mem.eql(u8, k, "low_power")) {
            line.low_power = switch (v) {
                .null => @as(?bool, null),
                .bool => |b| @as(?bool, b),
                else => return setField(err, line_no, .bad_value, k),
            };
        } else if (std.mem.eql(u8, k, "thermal")) {
            line.thermal = try optU32(v, err, line_no, k);
        } else if (std.mem.eql(u8, k, "gpu_pct")) {
            line.gpu_pct = try optU32(v, err, line_no, k);
        } else if (std.mem.eql(u8, k, "cpu_pct")) {
            const o = try optU32(v, err, line_no, k);
            if (o) |inner| if (inner) |p| if (p > 100) return setField(err, line_no, .bad_value, k);
            line.cpu_pct = o;
        } else if (std.mem.eql(u8, k, "mem_level")) {
            const n = jsonUint(v) orelse return setField(err, line_no, .bad_value, k);
            if (n > std.math.maxInt(u32)) return setField(err, line_no, .bad_value, k);
            line.mem_level = @intCast(n);
        } else if (std.mem.eql(u8, k, "sleep_ms")) {
            line.sleep_ms = jsonUint(v) orelse return setField(err, line_no, .bad_value, k);
        } else {
            return setField(err, line_no, .unknown_field, k);
        }
    }
    if (!have_at) {
        err.* = .{ .line = line_no, .kind = .missing_at_ms };
        return error.BadScript;
    }
    return line;
}

fn optU32(v: std.json.Value, err: *ParseError, line_no: usize, k: []const u8) error{BadScript}!??u32 {
    return switch (v) {
        .null => @as(?u32, null),
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @as(?u32, @intCast(i)) else setField(err, line_no, .bad_value, k),
        else => setField(err, line_no, .bad_value, k),
    };
}

// ── Choosing the source ──

pub const Choice = union(enum) {
    real: RealSignals,
    scripted: struct { s: ScriptedSignals, path: []const u8 },
};

/// `null` gives the real source; a path reads that script on the real clock (a tune's GPU batches take real time).
pub fn fromValue(alloc: std.mem.Allocator, io: std.Io, fake_path: ?[]const u8) !Choice {
    const path = fake_path orelse return .{ .real = .{} };
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20));
    defer alloc.free(text);
    const s = try ScriptedSignals.parse(alloc, text, .real);
    return .{ .scripted = .{ .s = s, .path = try alloc.dupe(u8, path) } };
}

/// Reads only `MLX_SERVE_TUNE_FAKE_SIGNALS` and calls `fromValue` (an empty value means unset).
pub fn fromEnv(alloc: std.mem.Allocator, io: std.Io) !Choice {
    const raw = std.c.getenv("MLX_SERVE_TUNE_FAKE_SIGNALS");
    const v: ?[]const u8 = if (raw) |r| std.mem.sliceTo(r, 0) else null;
    return fromValue(alloc, io, if (v) |s| (if (s.len == 0) null else s) else null);
}

// ── The conditions ──

pub const Cond = enum { holds, fails, unknown };

pub const OpenCheck = struct {
    input: Cond,
    ac: Cond,
    low_power: Cond,
    thermal: Cond,
    gpu: Cond,
    cpu: Cond,
    mem: Cond,

    /// Input, power, Low Power Mode and thermal must hold (unreadable fails: it never reads as idle); a GPU, CPU or memory
    /// reading that does not exist passes, the statistical gate covers it (spec 4.1).
    pub fn allHold(self: OpenCheck) bool {
        return self.input == .holds and self.ac == .holds and self.low_power == .holds and self.thermal == .holds and
            self.gpu != .fails and self.cpu != .fails and self.mem != .fails;
    }
};

fn condOfBool(b: ?bool, want: bool) Cond {
    const v = b orelse return .unknown;
    return if (v == want) .holds else .fails;
}

/// The open-time rules of spec 4.1. The GPU is the max of 8 readings 250 ms apart and the CPU the busy share over
/// `TUNE_SIGNAL_SPAN_S`, both through `waitNs`, so this call takes about 4 s on the real clock.
pub fn checkOpen(sig: Signals, idle_minutes: u32) OpenCheck {
    var r: OpenCheck = undefined;
    // saturating: no idle threshold wraps around to a short one, whatever the caller passes (the CLI caps it at a week)
    r.input = if (sig.hidIdleNs()) |ns| (if (ns >= @as(u64, idle_minutes) *| 60 *| std.time.ns_per_s) Cond.holds else Cond.fails) else Cond.unknown;
    r.ac = condOfBool(sig.onAc(), true);
    r.low_power = condOfBool(sig.lowPower(), false);
    r.thermal = if (sig.thermalLevel()) |l| (if (l == 0) Cond.holds else Cond.fails) else Cond.unknown;
    const m = sig.memPressure();
    r.mem = if (m == 0) .unknown else if (m == 1) .holds else .fails;

    var gpu_max: ?u32 = null;
    for (0..GPU_READINGS) |i| {
        if (sig.gpuPct()) |g| gpu_max = @max(gpu_max orelse 0, g);
        if (i + 1 < GPU_READINGS) sig.waitNs(GPU_READING_GAP_NS);
    }
    r.gpu = if (gpu_max) |g| (if (g < TUNE_GPU_PCT_MAX) Cond.holds else Cond.fails) else Cond.unknown;

    const a = sig.cpuTicks();
    sig.waitNs(@as(u64, TUNE_SIGNAL_SPAN_S) * std.time.ns_per_s);
    const b = sig.cpuTicks();
    r.cpu = .unknown;
    if (a != null and b != null) {
        var total: u64 = 0;
        for (0..4) |i| total += b.?[i] -| a.?[i];
        const idle = b.?[2] -| a.?[2];
        if (total > 0) {
            const share = (total -| idle) * 100 / total;
            r.cpu = if (share < TUNE_CPU_PCT_MAX) .holds else .fails;
        }
    }
    return r;
}

pub const MonitorFailure = enum { input, ac, low_power, thermal };

pub const MonitorCheck = struct {
    /// The first condition that failed, null when all four hold.
    failed: ?MonitorFailure,
    /// This reading's idle time, to pass as `last_idle_ns` at the next call (null when unreadable).
    idle_ns: ?u64,

    pub fn ok(self: MonitorCheck) bool {
        return self.failed == null;
    }
};

/// The four in-window conditions (spec 4.3). Input has happened when the idle time is below what it was at the window's
/// previous reading; unreadable input, AC, Low Power Mode or thermal state counts as failed.
pub fn monitorHolds(sig: Signals, last_idle_ns: u64) MonitorCheck {
    const idle = sig.hidIdleNs();
    var failed: ?MonitorFailure = null;
    if (idle == null or idle.? < last_idle_ns) {
        failed = .input;
    } else if (condOfBool(sig.onAc(), true) != .holds) {
        failed = .ac;
    } else if (condOfBool(sig.lowPower(), false) != .holds) {
        failed = .low_power;
    } else if ((sig.thermalLevel() orelse 1) != 0) {
        failed = .thermal;
    }
    return .{ .failed = failed, .idle_ns = idle };
}

/// `[tune] agent signals: hid_idle=<s or unreadable> ac=<..> low_power=<..> thermal=<..>`: the line `--agent` logs first.
pub fn agentSignalsLine(buf: []u8, sig: Signals) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.writeAll("[tune] agent signals: hid_idle=") catch return w.buffered();
    if (sig.hidIdleNs()) |ns| w.print("{d}", .{ns / std.time.ns_per_s}) catch {} else w.writeAll("unreadable") catch {};
    w.writeAll(" ac=") catch {};
    if (sig.onAc()) |b| w.print("{}", .{b}) catch {} else w.writeAll("unreadable") catch {};
    w.writeAll(" low_power=") catch {};
    if (sig.lowPower()) |b| w.print("{}", .{b}) catch {} else w.writeAll("unreadable") catch {};
    w.writeAll(" thermal=") catch {};
    if (sig.thermalLevel()) |l| w.print("{d}", .{l}) catch {} else w.writeAll("unreadable") catch {};
    return w.buffered();
}

// ── Tests ──

const testing = std.testing;

fn scripted(text: []const u8) !ScriptedSignals {
    return ScriptedSignals.parse(testing.allocator, text, .manual);
}

test "tune signals: scripted fields keep their last value and the idle time grows with the clock" {
    var s = try scripted("{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":10000,\"ac\":false}\n");
    defer s.deinit();
    const sig = s.signals();
    s.advanceMs(5000);
    try testing.expectEqual(@as(?u64, 605 * std.time.ns_per_s), sig.hidIdleNs());
    try testing.expectEqual(@as(?bool, true), sig.onAc());
    s.advanceMs(5000);
    try testing.expectEqual(@as(?bool, false), sig.onAc());
    try testing.expectEqual(@as(?u64, 610 * std.time.ns_per_s), sig.hidIdleNs());
    // the fields no line set keep their defaults
    try testing.expectEqual(@as(?bool, false), sig.lowPower());
    try testing.expectEqual(@as(?u32, 0), sig.thermalLevel());
    try testing.expectEqual(@as(?u32, 0), sig.gpuPct());
    try testing.expectEqual(@as(u32, 1), sig.memPressure());
}

test "tune signals: a scripted input resets the idle time" {
    var s = try scripted("{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":3000,\"hid_idle_s\":0}\n");
    defer s.deinit();
    const sig = s.signals();
    s.advanceMs(2000);
    try testing.expectEqual(@as(?u64, 602 * std.time.ns_per_s), sig.hidIdleNs());
    s.advanceMs(1500);
    try testing.expectEqual(@as(?u64, 500 * std.time.ns_per_ms), sig.hidIdleNs());
}

test "tune signals: a scripted sleep opens a gap between the two clocks" {
    var s = try scripted("{\"at_ms\":1000,\"sleep_ms\":5000}\n");
    defer s.deinit();
    const sig = s.signals();
    s.advanceMs(500);
    try testing.expectEqual(sig.absTimeNs(), sig.contTimeNs());
    s.advanceMs(500);
    try testing.expectEqual(@as(u64, 5 * std.time.ns_per_s), sig.contTimeNs() - sig.absTimeNs());
    s.advanceMs(3000);
    try testing.expectEqual(@as(u64, 5 * std.time.ns_per_s), sig.contTimeNs() - sig.absTimeNs());
}

test "tune signals: a script with an unknown field fails with its line number" {
    var perr: ParseError = .{};
    try testing.expectError(error.BadScript, ScriptedSignals.parseWithError(testing.allocator, "{\"at_ms\":0}\n\n{\"at_ms\":10,\"bogus\":1}\n", .manual, &perr));
    try testing.expectEqual(@as(usize, 3), perr.line);
    try testing.expectEqual(ParseErrorKind.unknown_field, perr.kind);
    try testing.expectEqualStrings("bogus", perr.fieldName());
    // the other malformed lines, each with its own line number
    try testing.expectError(error.BadScript, ScriptedSignals.parseWithError(testing.allocator, "{\"at_ms\":5}\n{\"at_ms\":4}\n", .manual, &perr));
    try testing.expectEqual(ParseErrorKind.time_goes_back, perr.kind);
    try testing.expectEqual(@as(usize, 2), perr.line);
    try testing.expectError(error.BadScript, ScriptedSignals.parseWithError(testing.allocator, "{\"ac\":true}\n", .manual, &perr));
    try testing.expectEqual(ParseErrorKind.missing_at_ms, perr.kind);
    try testing.expectError(error.BadScript, ScriptedSignals.parseWithError(testing.allocator, "not json\n", .manual, &perr));
    try testing.expectEqual(ParseErrorKind.not_json, perr.kind);
    try testing.expectError(error.BadScript, ScriptedSignals.parseWithError(testing.allocator, "{\"at_ms\":0,\"ac\":\"yes\"}\n", .manual, &perr));
    try testing.expectEqual(ParseErrorKind.bad_value, perr.kind);
    // the opposing case: a valid script parses
    var ok = try scripted("{\"at_ms\":0,\"gpu_pct\":null,\"cpu_pct\":null}\n");
    ok.deinit();
}

/// A one-line script at time 0 with the given values (`gpu` and `cpu` are JSON text, so `null` means unknown).
fn openScript(idle_s: u32, ac: bool, lp: bool, thermal: u32, gpu: []const u8, cpu: []const u8, mem: u32) ![]u8 {
    return std.fmt.allocPrint(testing.allocator, "{{\"at_ms\":0,\"hid_idle_s\":{d},\"ac\":{},\"low_power\":{},\"thermal\":{d},\"gpu_pct\":{s},\"cpu_pct\":{s},\"mem_level\":{d}}}\n", .{ idle_s, ac, lp, thermal, gpu, cpu, mem });
}

fn openWith(idle_s: u32, ac: bool, lp: bool, thermal: u32, gpu: []const u8, cpu: []const u8, mem: u32) !OpenCheck {
    const text = try openScript(idle_s, ac, lp, thermal, gpu, cpu, mem);
    defer testing.allocator.free(text);
    var s = try scripted(text);
    defer s.deinit();
    return checkOpen(s.signals(), 5);
}

test "tune signals: the open check needs every condition" {
    const base = try openWith(360, true, false, 0, "5", "10", 1);
    try testing.expect(base.allHold());
    try testing.expectEqual(Cond.holds, base.input);
    try testing.expectEqual(Cond.holds, base.gpu);
    try testing.expectEqual(Cond.holds, base.cpu);

    const idle = try openWith(240, true, false, 0, "5", "10", 1);
    try testing.expect(!idle.allHold());
    try testing.expectEqual(Cond.fails, idle.input);
    const ac = try openWith(360, false, false, 0, "5", "10", 1);
    try testing.expect(!ac.allHold());
    try testing.expectEqual(Cond.fails, ac.ac);
    const lp = try openWith(360, true, true, 0, "5", "10", 1);
    try testing.expect(!lp.allHold());
    try testing.expectEqual(Cond.fails, lp.low_power);
    const th = try openWith(360, true, false, 1, "5", "10", 1);
    try testing.expect(!th.allHold());
    try testing.expectEqual(Cond.fails, th.thermal);
    const gpu = try openWith(360, true, false, 0, "10", "10", 1);
    try testing.expect(!gpu.allHold());
    try testing.expectEqual(Cond.fails, gpu.gpu);
    const cpu = try openWith(360, true, false, 0, "5", "25", 1);
    try testing.expect(!cpu.allHold());
    try testing.expectEqual(Cond.fails, cpu.cpu);
    const mem = try openWith(360, true, false, 0, "5", "10", 2);
    try testing.expect(!mem.allHold());
    try testing.expectEqual(Cond.fails, mem.mem);
    // each flipped condition fails alone: the others still hold
    try testing.expectEqual(Cond.holds, mem.gpu);
    try testing.expectEqual(Cond.holds, gpu.cpu);

    // unknown GPU, CPU and memory pass; the values just under their limits hold
    const unknown = try openWith(360, true, false, 0, "null", "null", 0);
    try testing.expect(unknown.allHold());
    try testing.expectEqual(Cond.unknown, unknown.gpu);
    try testing.expectEqual(Cond.unknown, unknown.cpu);
    try testing.expectEqual(Cond.unknown, unknown.mem);
    const edge = try openWith(300, true, false, 0, "9", "24", 1);
    try testing.expect(edge.allHold());

    // an unreadable idle time fails: it never reads as idle
    var s = try scripted("{\"at_ms\":0,\"hid_idle_s\":null}\n");
    defer s.deinit();
    const unreadable = checkOpen(s.signals(), 5);
    try testing.expectEqual(Cond.unknown, unreadable.input);
    try testing.expect(!unreadable.allHold());
}

test "tune signals: the GPU pre-filter takes the max of 8 readings" {
    var s = try scripted("{\"at_ms\":0,\"hid_idle_s\":600,\"gpu_pct\":2}\n{\"at_ms\":1000,\"gpu_pct\":40}\n{\"at_ms\":1250,\"gpu_pct\":2}\n");
    defer s.deinit();
    const r = checkOpen(s.signals(), 5);
    try testing.expectEqual(Cond.fails, r.gpu);
    try testing.expect(!r.allHold());
    // the same script without the spike holds
    var calm = try scripted("{\"at_ms\":0,\"hid_idle_s\":600,\"gpu_pct\":2}\n");
    defer calm.deinit();
    try testing.expect(checkOpen(calm.signals(), 5).allHold());
}

test "tune signals: the monitor names the first condition that failed" {
    var s = try scripted("{\"at_ms\":0,\"hid_idle_s\":600}\n{\"at_ms\":1000,\"hid_idle_s\":0}\n{\"at_ms\":2000,\"ac\":false}\n{\"at_ms\":3000,\"ac\":true,\"low_power\":true}\n{\"at_ms\":4000,\"low_power\":false,\"thermal\":2}\n");
    defer s.deinit();
    const sig = s.signals();
    var last: u64 = 0;
    var m = monitorHolds(sig, last);
    try testing.expect(m.ok());
    last = m.idle_ns.?;
    s.advanceMs(500);
    m = monitorHolds(sig, last);
    try testing.expect(m.ok()); // the idle time grew
    last = m.idle_ns.?;
    s.advanceMs(600); // input at 1,000 ms
    m = monitorHolds(sig, last);
    try testing.expectEqual(@as(?MonitorFailure, .input), m.failed);
    last = m.idle_ns.?;
    s.advanceMs(1000); // AC off at 2,000 ms
    try testing.expectEqual(@as(?MonitorFailure, .ac), monitorHolds(sig, last).failed);
    s.advanceMs(1000); // low power at 3,000 ms
    try testing.expectEqual(@as(?MonitorFailure, .low_power), monitorHolds(sig, last).failed);
    s.advanceMs(1000); // thermal at 4,000 ms
    try testing.expectEqual(@as(?MonitorFailure, .thermal), monitorHolds(sig, last).failed);
}

test "tune signals: the agent signals line" {
    var buf: [200]u8 = undefined;
    var s = try scripted("{\"at_ms\":0,\"hid_idle_s\":360,\"ac\":true,\"low_power\":false,\"thermal\":0}\n");
    defer s.deinit();
    try testing.expectEqualStrings("[tune] agent signals: hid_idle=360 ac=true low_power=false thermal=0", agentSignalsLine(&buf, s.signals()));
    var u = try scripted("{\"at_ms\":0,\"hid_idle_s\":null,\"ac\":true,\"low_power\":false,\"thermal\":0}\n");
    defer u.deinit();
    try testing.expectEqualStrings("[tune] agent signals: hid_idle=unreadable ac=true low_power=false thermal=0", agentSignalsLine(&buf, u.signals()));
}

test "tune signals: real signals read on this Mac" {
    if (comptime !is_macos) return error.SkipZigTest;
    var real: RealSignals = .{};
    defer real.deinit();
    const sig = real.signals();
    const idle = sig.hidIdleNs();
    const ac = sig.onAc();
    const lp = sig.lowPower();
    const th = sig.thermalLevel();
    const gpu = sig.gpuPct();
    const ticks = sig.cpuTicks();
    const mem = sig.memPressure();
    var ibuf: [32]u8 = undefined;
    const idle_text = if (idle) |ns| std.fmt.bufPrint(&ibuf, "{d}", .{ns / std.time.ns_per_s}) catch "?" else "unreadable";
    std.debug.print("tune signals: real signals hid_idle={s} ac={any} low_power={any} thermal={any} gpu={any} cpu_ticks={any} mem={d}\n", .{ idle_text, ac, lp, th, gpu, ticks, mem });
    try testing.expect(ticks != null);
    try testing.expect(th != null);
    try testing.expect(mem != 0);
    const a = sig.absTimeNs();
    sig.waitNs(20 * std.time.ns_per_ms);
    const b = sig.absTimeNs();
    try testing.expect(b > a);
    try testing.expect(b - a < std.time.ns_per_s);
    const abs_first = sig.absTimeNs(); // the absolute clock is read first, so the later continuous reading cannot be behind it
    try testing.expect(sig.contTimeNs() >= abs_first);
}

test "tune signals: a source is chosen from a value" {
    const real = try fromValue(testing.allocator, testing.io, null);
    try testing.expect(real == .real);
    // a missing script is an error, not a silent fall back to the real source
    try testing.expectError(error.FileNotFound, fromValue(testing.allocator, testing.io, "/nonexistent-lesto-dir/signals.jsonl"));
}
