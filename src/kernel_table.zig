//! The local kernel table: the layouts `lesto tune` found on this Mac for the two MoE rows kernels, written to one JSON
//! file the engine reads at the first MoE load. This file holds the table's types, its canonical serialization and
//! checksum, the hash that ties a table to the kernels it was tuned for, the refusals, the permission check and the
//! atomic write. It never decides which layout to serve (that is `transformer.moeRowsLayoutDecideSources`).
//!
//! Files go through libc directly (open, fstat, stat, read, fsync, rename): the reader must see the same file it
//! checked, so the permission check runs on the opened descriptor, as the refusal rules require.
const std = @import("std");
const mlx = @import("mlx.zig");
const status = @import("status.zig");
const os_mutex = @import("os_mutex.zig");
const transformer = @import("transformer.zig");

const MoeRowsLayoutGateUp = transformer.MoeRowsLayoutGateUp;
const MoeRowsLayoutDown = transformer.MoeRowsLayoutDown;

/// The plan space this build searches (spec 3.1). Part of the kernel hash, so a changed space refuses old tables.
/// Version 2 adds the down body (scalar or array) to the down plans.
pub const TUNE_PLAN_SPACE_VERSION: u32 = 2;

/// The table file's schema. Schema 2 adds `body` to a down layout object; a 0.26.0 (schema 1) table is refused
/// `unknown_schema`, and `lesto tune` writes a new one.
pub const TABLE_SCHEMA: u32 = 2;

pub const KernelTableRefusal = enum {
    unreadable,
    unsafe_permissions,
    unparseable,
    unknown_schema,
    checksum_mismatch,
    shape_unknown,
    plan_value_out_of_range,
    plan_outside_space,
    identity_not_true,
    key_mismatch,

    pub fn text(self: KernelTableRefusal) []const u8 {
        return @tagName(self);
    }
};

/// The key fields, in the order they are checked.
pub const KeyField = enum { gpu_arch, gpu_cores, hw_model, kernel_hash, mlx_version, os_build };

pub const RefusalDetail = struct {
    reason: KernelTableRefusal,
    field: ?KeyField = null,
    /// The two values of a text field (they point into the table and the Mac key); empty for `gpu_cores`.
    table_value: []const u8 = "",
    mac_value: []const u8 = "",
    /// The two values of `gpu_cores`, kept as numbers so a detail owns them; `text` prints them.
    table_cores: u32 = 0,
    mac_cores: u32 = 0,

    /// The reason text the refusal line prints: the tag name, and for `key_mismatch` the field, both values and the fix.
    pub fn text(self: RefusalDetail, buf: []u8) []const u8 {
        const f = self.field orelse return self.reason.text();
        const fix: []const u8 = if (f == .os_build) "lesto tune --verify" else "lesto tune";
        if (f == .gpu_cores) {
            return std.fmt.bufPrint(buf, "key_mismatch: gpu_cores is {d} in the table and {d} on this Mac; run `{s}`", .{ self.table_cores, self.mac_cores, fix }) catch self.reason.text();
        }
        return std.fmt.bufPrint(buf, "key_mismatch: {s} is {s} in the table and {s} on this Mac; run `{s}`", .{ @tagName(f), self.table_value, self.mac_value, fix }) catch self.reason.text();
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Table types (spec 3.3, field order is the file's order)

pub const Key = struct {
    gpu_arch: []const u8,
    gpu_cores: u32,
    hw_model: []const u8,
    os_build: []const u8,
    kernel_hash: []const u8,
    mlx_version: []const u8,
};

pub const Shape = struct { hidden: u32, inter: u32, bits: u32, group: u32, dtype: []const u8 };
pub const TUNED_SHAPE = Shape{ .hidden = 2816, .inter = 704, .bits = 4, .group = 64, .dtype = "bfloat16" };

/// A plan value outside the parsers' ranges, kept as the file wrote it so the checksum still re-serializes it.
const RawGateUp = struct { sg: i64, ax: i64 };
const RawDown = struct { sps: i64, tpt: i64, ax: i64, body: i64 };

/// `"served"` or `{"sg":..,"ax":..}`. `out_of_range` is what a parse of a value the range check refuses gives.
pub const GateUpEntry = union(enum) {
    served,
    layout: MoeRowsLayoutGateUp,
    out_of_range: RawGateUp,

    pub fn jsonStringify(self: GateUpEntry, jw: anytype) !void {
        switch (self) {
            .served => try jw.write("served"),
            .layout => |l| try writeGateUp(jw, l.sg, l.ax),
            .out_of_range => |r| try writeGateUp(jw, r.sg, r.ax),
        }
    }

    pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !GateUpEntry {
        switch (try source.peekNextTokenType()) {
            .string => {
                const s = try std.json.innerParse([]const u8, alloc, source, options);
                if (!std.mem.eql(u8, s, "served")) return error.UnexpectedToken;
                return .served;
            },
            .object_begin => {
                const raw = try std.json.innerParse(RawGateUp, alloc, source, options);
                transformer.moeRowsLayoutCheckGateUp(raw.sg, raw.ax) catch return .{ .out_of_range = raw };
                return .{ .layout = .{ .sg = @intCast(raw.sg), .ax = @intCast(raw.ax) } };
            },
            else => return error.UnexpectedToken,
        }
    }
};

fn writeGateUp(jw: anytype, sg: anytype, ax: anytype) !void {
    try jw.beginObject();
    try jw.objectField("sg");
    try jw.write(sg);
    try jw.objectField("ax");
    try jw.write(ax);
    try jw.endObject();
}

/// `"served"` or `{"sps":..,"tpt":..,"ax":..,"body":0|1}`.
pub const DownEntry = union(enum) {
    served,
    layout: MoeRowsLayoutDown,
    out_of_range: RawDown,

    pub fn jsonStringify(self: DownEntry, jw: anytype) !void {
        switch (self) {
            .served => try jw.write("served"),
            .layout => |l| try writeDown(jw, l.sps, l.tpt, l.ax, @intFromEnum(l.body)),
            .out_of_range => |r| try writeDown(jw, r.sps, r.tpt, r.ax, r.body),
        }
    }

    pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !DownEntry {
        switch (try source.peekNextTokenType()) {
            .string => {
                const s = try std.json.innerParse([]const u8, alloc, source, options);
                if (!std.mem.eql(u8, s, "served")) return error.UnexpectedToken;
                return .served;
            },
            .object_begin => {
                const raw = try std.json.innerParse(RawDown, alloc, source, options);
                transformer.moeRowsLayoutCheckDown(raw.sps, raw.tpt, raw.ax) catch return .{ .out_of_range = raw };
                const body = transformer.moeRowsLayoutCheckDownBody(raw.body) catch return .{ .out_of_range = raw };
                return .{ .layout = .{ .sps = @intCast(raw.sps), .tpt = @intCast(raw.tpt), .ax = @intCast(raw.ax), .body = body } };
            },
            else => return error.UnexpectedToken,
        }
    }
};

fn writeDown(jw: anytype, sps: anytype, tpt: anytype, ax: anytype, body: anytype) !void {
    try jw.beginObject();
    try jw.objectField("sps");
    try jw.write(sps);
    try jw.objectField("tpt");
    try jw.write(tpt);
    try jw.objectField("ax");
    try jw.write(ax);
    try jw.objectField("body");
    try jw.write(body);
    try jw.endObject();
}

pub const Plan = struct { gateup: GateUpEntry, down: DownEntry };

/// One confirmation in one regime: the candidate-minus-reference median and its 95% interval, in nanoseconds a call.
pub const ConfirmStat = struct { median_ns: i64, lo_ns: i64, hi_ns: i64, window: u32 };
/// `confirm` order: confirm 1 independent, confirm 1 shared, confirm 2 independent, confirm 2 shared (index Decision 5).
pub const KernelEvidence = struct { ref_ns: i64, pick_ns: i64, confirm: [4]ConfirmStat };
pub const Evidence = struct { gateup: ?KernelEvidence, down: ?KernelEvidence };

pub const Excluded = struct { kernel: []const u8, cand: []const u8, reason: []const u8 };
pub const Identity = struct {
    plans_checked: u32,
    cases: u32,
    calls_compared: u64,
    os_build: []const u8,
    all_identical: bool,
    excluded: []const Excluded,
};

pub const Discarded = struct { reason: []const u8, n: u32 };
pub const NotMeasurable = struct { kernel: []const u8, cand: []const u8 };
pub const ServedReasons = struct { gateup: ?[]const u8, down: ?[]const u8 };
/// The reasons a batch is discarded, in the order the table lists them.
pub const DISCARD_REASONS = [7][]const u8{ "input", "power", "low_power", "thermal", "sleep", "stopped", "gate" };
pub const Info = struct {
    lesto_version: []const u8,
    tuned_at: []const u8,
    signals: []const u8,
    windows_used: u32,
    batches_kept: u32,
    batches_discarded: [7]Discarded,
    not_measurable: []const NotMeasurable,
    served_reasons: ServedReasons,
};

pub const Table = struct {
    schema: u32,
    key: Key,
    shape: Shape,
    plan: Plan,
    evidence: Evidence,
    identity: Identity,
    info: Info,
    checksum: []const u8,
};

// ---------------------------------------------------------------------------------------------------------------
// Canonical serialization and checksum (spec 3.5)

/// Compact `std.json`; every order is the struct's, so the same table always gives the same bytes.
pub fn serialize(alloc: std.mem.Allocator, t: Table) ![]u8 {
    return std.json.Stringify.valueAlloc(alloc, t, .{});
}

/// SHA-256, lowercase hex, of `serialize` with `checksum` set to the empty string.
pub fn checksumHex(alloc: std.mem.Allocator, t: Table) ![64]u8 {
    var c = t;
    c.checksum = "";
    const bytes = try serialize(alloc, c);
    defer alloc.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// Sets `t.checksum` (allocated with `alloc`, which the caller owns, best an arena).
pub fn finalize(alloc: std.mem.Allocator, t: *Table) !void {
    const hex = try checksumHex(alloc, t.*);
    t.checksum = try alloc.dupe(u8, &hex);
}

// ---------------------------------------------------------------------------------------------------------------
// kernel_hash (spec 3.4)

/// SHA-256 over the parts in order, each prefixed by its byte length as a little-endian u64.
pub fn kernelHash(parts: []const []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (parts) |p| {
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, p.len, .little);
        h.update(&len);
        h.update(p);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

const RULE_NAMES = [3][]const u8{ "downredLanesFor", "moeRowsLayoutGateUp", "moeRowsLayoutDown" };

/// The text of one rule function: from the line that starts `fn <name>(` through the first line that is exactly `}`.
fn cutRule(src: []const u8, name: []const u8) []const u8 {
    var needle_buf: [64]u8 = undefined;
    const needle = std.fmt.bufPrint(&needle_buf, "fn {s}(", .{name}) catch return "";
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, src, from, needle)) |at| {
        if (at == 0 or src[at - 1] == '\n') {
            const end = std.mem.indexOfPos(u8, src, at, "\n}\n") orelse return "";
            return src[at .. end + 2];
        }
        from = at + 1;
    }
    return "";
}

/// The texts of `downredLanesFor`, `moeRowsLayoutGateUp` and `moeRowsLayoutDown`, cut from this module's own source file.
pub fn ruleTexts() [3][]const u8 {
    const src = @embedFile("transformer.zig");
    return .{ cutRule(src, RULE_NAMES[0]), cutRule(src, RULE_NAMES[1]), cutRule(src, RULE_NAMES[2]) };
}

const RULES_BUF_LEN = 16 * 1024;
var parts_mu: os_mutex.Mutex = .{};
var parts_ready = std.atomic.Value(bool).init(false);
var parts_rules: [RULES_BUF_LEN]u8 = undefined;
var parts_rules_len: usize = 0;
var parts_version: [4]u8 = undefined;
var parts_hex: [64]u8 = @splat(0);
var parts_hex_ready: bool = false; // guarded by parts_mu

fn engineSources() [6][]const u8 {
    return .{
        transformer.GQMV_GATEUP_ROWS_SOURCES[0],
        transformer.GQMV_DOWNRED_ROWS_SOURCES[0],
        transformer.GQMV_GATEUP_ROWS_LAYOUT_SOURCE,
        transformer.GQMV_DOWNRED_ROWS_LAYOUT_SOURCE,
        transformer.GQMV_DOWNRED_ROWS_ARRAY_SOURCE,
        transformer.GQMV_AFFINE_HEADER,
    };
}

/// The eight parts of spec 3.4 in order (the array body's source is part 5, before the header). The rule texts (part 7)
/// and the version (part 8) live in static buffers filled once; the call is safe from any thread.
pub fn engineParts() [8][]const u8 {
    ensureParts();
    const s = engineSources();
    return .{ s[0], s[1], s[2], s[3], s[4], s[5], parts_rules[0..parts_rules_len], &parts_version };
}

fn ensureParts() void {
    if (parts_ready.load(.acquire)) return;
    parts_mu.lock();
    defer parts_mu.unlock();
    if (parts_ready.load(.acquire)) return;
    var n: usize = 0;
    for (ruleTexts()) |r| {
        if (n + r.len > RULES_BUF_LEN) @panic("kernel_table: the rule texts outgrew their buffer");
        @memcpy(parts_rules[n .. n + r.len], r);
        n += r.len;
    }
    parts_rules_len = n;
    std.mem.writeInt(u32, &parts_version, TUNE_PLAN_SPACE_VERSION, .little);
    parts_ready.store(true, .release);
}

/// The running engine's kernel hash as 64 lowercase hex characters, computed once.
pub fn engineKernelHashHex() [64]u8 {
    ensureParts();
    parts_mu.lock();
    defer parts_mu.unlock();
    if (!parts_hex_ready) {
        parts_hex = std.fmt.bytesToHex(kernelHash(&engineParts()), .lower);
        parts_hex_ready = true;
    }
    return parts_hex;
}

// ---------------------------------------------------------------------------------------------------------------
// The plan space (spec 3.1)

const GATEUP_SGS = [_]c_int{ 1, 2, 4, 8, 16, 32 };
const DOWN_SPS = [_]c_int{ 1, 2, 3, 6 };
const DOWN_TPT = [_]c_int{ 1, 2, 4 };

pub fn gateUpInSpace(g: MoeRowsLayoutGateUp) bool {
    for (GATEUP_SGS) |sg| if (g.sg == sg) return true;
    return false;
}

/// The down bodies the space holds. An exhaustive switch: a third body added later must be placed here explicitly.
fn downBodyInSpace(b: transformer.MoeRowsLayoutDownBody) bool {
    return switch (b) {
        .scalar, .array => true,
    };
}

pub fn downInSpace(d: MoeRowsLayoutDown) bool {
    var sps_ok = false;
    for (DOWN_SPS) |v| sps_ok = sps_ok or d.sps == v;
    var tpt_ok = false;
    for (DOWN_TPT) |v| tpt_ok = tpt_ok or d.tpt == v;
    return sps_ok and tpt_ok and downBodyInSpace(d.body);
}

/// Candidates a kernel searches: every point of its space but the served one.
pub const GATEUP_CANDIDATES: usize = 11;
pub const DOWN_CANDIDATES: usize = 47;

/// Every gate/up layout but the served one: SG ascending, then AX 0 before 1.
pub fn gateUpCandidates() [GATEUP_CANDIDATES]MoeRowsLayoutGateUp {
    var out: [GATEUP_CANDIDATES]MoeRowsLayoutGateUp = undefined;
    var n: usize = 0;
    for (GATEUP_SGS) |sg| {
        for ([_]u1{ 0, 1 }) |ax| {
            const g = MoeRowsLayoutGateUp{ .sg = sg, .ax = ax };
            if (std.meta.eql(g, transformer.MOE_ROWS_LAYOUT_GATEUP_SERVED)) continue;
            out[n] = g;
            n += 1;
        }
    }
    std.debug.assert(n == out.len);
    return out;
}

/// Every down plan but the served one: SPS ascending, then TPT ascending, then AX 0 before 1, then the body (scalar
/// before array). So the first candidate is (1, 1, 0, array).
pub fn downCandidates() [DOWN_CANDIDATES]MoeRowsLayoutDown {
    var out: [DOWN_CANDIDATES]MoeRowsLayoutDown = undefined;
    var n: usize = 0;
    for (DOWN_SPS) |sps| {
        for (DOWN_TPT) |tpt| {
            for ([_]u1{ 0, 1 }) |ax| {
                for ([_]transformer.MoeRowsLayoutDownBody{ .scalar, .array }) |body| {
                    const d = MoeRowsLayoutDown{ .sps = sps, .tpt = tpt, .ax = ax, .body = body };
                    if (std.meta.eql(d, transformer.MOE_ROWS_LAYOUT_DOWN_SERVED)) continue;
                    out[n] = d;
                    n += 1;
                }
            }
        }
    }
    std.debug.assert(n == out.len);
    return out;
}

/// The `moeRowsLayoutParseGateUp` text form of a layout: `sg=32 ax=1`.
pub fn gateUpText(buf: []u8, g: MoeRowsLayoutGateUp) []const u8 {
    return std.fmt.bufPrint(buf, "sg={d} ax={d}", .{ g.sg, g.ax }) catch "";
}

/// The `moeRowsLayoutParseDown` text form: `sps=6 tpt=1 ax=1 body=0`.
pub fn downText(buf: []u8, d: MoeRowsLayoutDown) []const u8 {
    return std.fmt.bufPrint(buf, "sps={d} tpt={d} ax={d} body={d}", .{ d.sps, d.tpt, d.ax, @intFromEnum(d.body) }) catch "";
}

// ---------------------------------------------------------------------------------------------------------------
// Reading and refusing (spec 3.6)

pub const StatLite = struct { uid: u32, mode: u32 };

/// `unsafe_permissions` when the file is group- or world-writable or not ours, or its folder is group- or
/// world-writable; null otherwise.
pub fn tablePermsSafe(uid: u32, file: StatLite, dir: StatLite) ?KernelTableRefusal {
    if (file.uid != uid) return .unsafe_permissions;
    if (file.mode & 0o022 != 0) return .unsafe_permissions;
    if (dir.mode & 0o022 != 0) return .unsafe_permissions;
    return null;
}

pub const ReadResult = union(enum) {
    absent,
    refused: RefusalDetail,
    /// Passed checks 1 to 9 and failed only the key check: `--verify` refreshes this very table.
    key_mismatch: struct { detail: RefusalDetail, table: std.json.Parsed(Table) },
    valid: std.json.Parsed(Table),

    /// The refusal detail for `refused` and `key_mismatch`, null otherwise.
    pub fn refusal(self: ReadResult) ?RefusalDetail {
        return switch (self) {
            .refused => |d| d,
            .key_mismatch => |k| k.detail,
            else => null,
        };
    }

    pub fn deinit(self: ReadResult) void {
        switch (self) {
            .key_mismatch => |k| k.table.deinit(),
            .valid => |p| p.deinit(),
            else => {},
        }
    }
};

fn refuse(r: KernelTableRefusal) ReadResult {
    return .{ .refused = .{ .reason = r } };
}

/// The first key field that differs from the running Mac and binary, in the order of `KeyField`.
pub fn checkKey(t: Table, mac: Key) ?RefusalDetail {
    if (!std.mem.eql(u8, t.key.gpu_arch, mac.gpu_arch)) return .{ .reason = .key_mismatch, .field = .gpu_arch, .table_value = t.key.gpu_arch, .mac_value = mac.gpu_arch };
    if (t.key.gpu_cores != mac.gpu_cores) return .{ .reason = .key_mismatch, .field = .gpu_cores, .table_cores = t.key.gpu_cores, .mac_cores = mac.gpu_cores };
    if (!std.mem.eql(u8, t.key.hw_model, mac.hw_model)) return .{ .reason = .key_mismatch, .field = .hw_model, .table_value = t.key.hw_model, .mac_value = mac.hw_model };
    if (!std.mem.eql(u8, t.key.kernel_hash, mac.kernel_hash)) return .{ .reason = .key_mismatch, .field = .kernel_hash, .table_value = t.key.kernel_hash, .mac_value = mac.kernel_hash };
    if (!std.mem.eql(u8, t.key.mlx_version, mac.mlx_version)) return .{ .reason = .key_mismatch, .field = .mlx_version, .table_value = t.key.mlx_version, .mac_value = mac.mlx_version };
    if (!std.mem.eql(u8, t.key.os_build, mac.os_build)) return .{ .reason = .key_mismatch, .field = .os_build, .table_value = t.key.os_build, .mac_value = mac.os_build };
    return null;
}

const MAX_TABLE_BYTES: usize = 1 << 20;

fn cPath(buf: *[std.fs.max_path_bytes]u8, path: []const u8) ?[*:0]const u8 {
    // an embedded NUL would cut the path short for libc, so the file operated on would not be the one named
    if (path.len >= buf.len or std.mem.indexOfScalar(u8, path, 0) != null) return null;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return @ptrCast(buf);
}

fn statLite(st: std.c.Stat) StatLite {
    return .{ .uid = @intCast(st.uid), .mode = @intCast(st.mode) };
}

const S_IFMT: u32 = 0o170000;
const S_IFREG: u32 = 0o100000;
const S_IFDIR: u32 = 0o040000;

/// Checks spec 3.6 in order; `key_mismatch` last. The returned value owns its parse (`deinit`). `io` is unused: the
/// reader opens, fstats and reads one descriptor with libc so the permission check covers the bytes it parses.
pub fn readTable(alloc: std.mem.Allocator, io: std.Io, path: []const u8, mac: Key) ReadResult {
    _ = io;
    // The folder is opened once and the file is opened inside that descriptor, so the folder whose mode is checked is the
    // one the bytes come from, whatever happens to the path afterwards. NONBLOCK keeps a FIFO at the path from hanging
    // the open; the regular-file check below then refuses it.
    var dbuf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = std.fs.path.dirname(path) orelse ".";
    const dz = cPath(&dbuf, dir_path) orelse return refuse(.unreadable);
    // a path with a trailing slash names a directory, so it cannot name the table file
    if (path.len == 0 or path[path.len - 1] == '/') return refuse(.unreadable);
    // NONBLOCK: a parent that is a FIFO must not hang the open; it is refused as not a directory just below
    const dfd = std.c.open(dz, .{ .ACCMODE = .RDONLY, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (dfd < 0) return if (std.c.errno(dfd) == .NOENT) .absent else refuse(.unreadable);
    defer _ = std.c.close(dfd);
    var bbuf: [std.fs.max_path_bytes]u8 = undefined;
    const bz = cPath(&bbuf, std.fs.path.basename(path)) orelse return refuse(.unreadable);
    // NOFOLLOW: a symlinked table is refused, because the folder whose mode is checked must be the one that holds the bytes
    const fd = std.c.openat(dfd, bz, .{ .ACCMODE = .RDONLY, .NONBLOCK = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) {
        const e = std.c.errno(fd);
        if (e == .NOENT) return .absent;
        return refuse(.unreadable);
    }
    defer _ = std.c.close(fd);

    var fst: std.c.Stat = undefined;
    if (std.c.fstat(fd, &fst) != 0) return refuse(.unreadable);
    var dst: std.c.Stat = undefined;
    if (std.c.fstat(dfd, &dst) != 0) return refuse(.unreadable);
    if (@as(u32, @intCast(dst.mode)) & S_IFMT != S_IFDIR) return refuse(.unreadable);
    if (tablePermsSafe(@intCast(std.c.geteuid()), statLite(fst), statLite(dst))) |r| return refuse(r);
    if (@as(u32, @intCast(fst.mode)) & S_IFMT != S_IFREG) return refuse(.unreadable);
    const size: usize = @intCast(fst.size);
    if (size > MAX_TABLE_BYTES) return refuse(.unreadable);

    const bytes = alloc.alloc(u8, size) catch return refuse(.unreadable);
    defer alloc.free(bytes);
    var got: usize = 0;
    while (got < size) {
        const n = std.c.read(fd, bytes[got..].ptr, size - got);
        if (n < 0 and std.c.errno(n) == .INTR) continue; // interrupted by a signal: retry at the same offset
        if (n <= 0) return refuse(.unreadable);
        got += @intCast(n);
    }

    // The schema alone first: a 0.26.0 table (schema 1, no `body` in the down object) would fail the full parse, and the
    // refusal must name the real fix (`unknown_schema`), not `unparseable`.
    {
        var peek = std.heap.ArenaAllocator.init(alloc);
        defer peek.deinit();
        const head = std.json.parseFromSlice(struct { schema: u32 }, peek.allocator(), bytes, .{ .ignore_unknown_fields = true }) catch return refuse(.unparseable);
        if (head.value.schema != TABLE_SCHEMA) return refuse(.unknown_schema);
    }
    const parsed = std.json.parseFromSlice(Table, alloc, bytes, .{ .ignore_unknown_fields = false, .allocate = .alloc_always }) catch return refuse(.unparseable);
    var keep = false;
    defer if (!keep) parsed.deinit();
    const t = parsed.value;
    if (t.schema != TABLE_SCHEMA) return refuse(.unknown_schema);

    var scratch = std.heap.ArenaAllocator.init(alloc);
    defer scratch.deinit();
    const sum = checksumHex(scratch.allocator(), t) catch return refuse(.unparseable);
    if (!std.mem.eql(u8, &sum, t.checksum)) return refuse(.checksum_mismatch);

    if (t.shape.hidden != TUNED_SHAPE.hidden or t.shape.inter != TUNED_SHAPE.inter or t.shape.bits != TUNED_SHAPE.bits or
        t.shape.group != TUNED_SHAPE.group or !std.mem.eql(u8, t.shape.dtype, TUNED_SHAPE.dtype)) return refuse(.shape_unknown);

    if (t.plan.gateup == .out_of_range or t.plan.down == .out_of_range) return refuse(.plan_value_out_of_range);
    switch (t.plan.gateup) {
        .layout => |g| if (!gateUpInSpace(g)) return refuse(.plan_outside_space),
        else => {},
    }
    switch (t.plan.down) {
        .layout => |d| if (!downInSpace(d)) return refuse(.plan_outside_space),
        else => {},
    }
    if (!t.identity.all_identical) return refuse(.identity_not_true);

    keep = true;
    if (checkKey(t, mac)) |d| return .{ .key_mismatch = .{ .detail = d, .table = parsed } };
    return .{ .valid = parsed };
}

fn sysctlString(alloc: std.mem.Allocator, name: [*:0]const u8) ![]u8 {
    var len: usize = 0;
    if (std.c.sysctlbyname(name, null, &len, null, 0) != 0) return error.SysctlFailed;
    const buf = try alloc.alloc(u8, len);
    errdefer alloc.free(buf);
    if (std.c.sysctlbyname(name, buf.ptr, &len, null, 0) != 0) return error.SysctlFailed;
    // an exact-size copy, so the caller frees what it was given with the length it was allocated at
    const text = try alloc.dupe(u8, buf[0 .. if (len > 0 and buf[len - 1] == 0) len - 1 else len]);
    alloc.free(buf);
    return text;
}

/// This Mac's key. Strings are allocated with `alloc` (best an arena). `gpu_arch` is empty when the GPU is not reported.
pub fn readMacKey(alloc: std.mem.Allocator) !Key {
    var abuf: [128]u8 = undefined;
    const hash = engineKernelHashHex();
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    if (mlx.mlx_version(&ver) != 0) return error.MlxVersionUnavailable;
    // each field is freed if a later one fails, so a caller with a non-arena allocator leaks nothing
    const arch = try alloc.dupe(u8, transformer.gpuArchitecture(&abuf) orelse "");
    errdefer alloc.free(arch);
    const hw_model = try sysctlString(alloc, "hw.model");
    errdefer alloc.free(hw_model);
    const os_build = try sysctlString(alloc, "kern.osversion");
    errdefer alloc.free(os_build);
    const kernel_hash = try alloc.dupe(u8, &hash);
    errdefer alloc.free(kernel_hash);
    const mlx_version = try alloc.dupe(u8, std.mem.span(mlx.mlx_string_data(ver)));
    return .{
        .gpu_arch = arch,
        .gpu_cores = status.gpuCoreCount(),
        .hw_model = hw_model,
        .os_build = os_build,
        .kernel_hash = kernel_hash,
        .mlx_version = mlx_version,
    };
}

// ---------------------------------------------------------------------------------------------------------------
// Atomic write (spec 7.7)

/// Test seam: `writeAtomic` returns `error.TestStopped` after the temporary file's fsync, before the rename.
pub var write_atomic_stop_before_rename: bool = false;
var write_mu: os_mutex.Mutex = .{};

extern "c" fn kill(pid: c_int, sig: c_int) c_int;
extern "c" fn unlinkat(fd: c_int, path: [*:0]const u8, flag: c_int) c_int;
extern "c" fn renameat(fromfd: c_int, from: [*:0]const u8, tofd: c_int, to: [*:0]const u8) c_int;

/// True when `pid` names a live process other than this one (signal 0 only checks; EPERM means it exists).
fn otherProcessAlive(pid: c_int) bool {
    if (pid == std.c.getpid()) return false;
    const rc = kill(pid, 0);
    if (rc == 0) return true;
    return std.c.errno(rc) == .PERM;
}

/// Removes the `<basename>.tmp-<pid>` files beside `target` that a killed writer left: the suffix is all digits and no
/// other live process owns that pid. Any other sibling (a name with a different suffix) is left alone. The names are
/// listed by path, but each one is removed inside `dfd`, the folder whose mode was checked, so a path that now resolves
/// elsewhere can only supply names: nothing outside the checked folder is deleted.
fn removeLeftovers(io: std.Io, dir_path: []const u8, base: []const u8, dfd: c_int) void {
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(io);
    var prefix_buf: [std.fs.max_path_bytes]u8 = undefined;
    const prefix = std.fmt.bufPrint(&prefix_buf, "{s}.tmp-", .{base}) catch return;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.startsWith(u8, entry.name, prefix)) continue;
        const suffix = entry.name[prefix.len..];
        if (suffix.len == 0) continue;
        const all_digits = for (suffix) |ch| {
            if (ch < '0' or ch > '9') break false;
        } else true;
        if (!all_digits) continue;
        const pid = std.fmt.parseInt(c_int, suffix, 10) catch continue;
        // only the spelling this writer produces: no leading zeros
        var canon: [16]u8 = undefined;
        if (!std.mem.eql(u8, std.fmt.bufPrint(&canon, "{d}", .{pid}) catch continue, suffix)) continue;
        if (pid <= 0 or otherProcessAlive(pid)) continue;
        var nbuf: [std.fs.max_path_bytes]u8 = undefined;
        const nz = cPath(&nbuf, entry.name) orelse continue;
        _ = unlinkat(dfd, nz, 0);
    }
}

/// Writes `bytes` to `target` through `<target>.tmp-<pid>`: fsync, rename, fsync the folder; mode 0600. The folder is
/// created with mode 0700 when missing (one level). A leftover temporary file of an earlier writer is removed first.
/// A table larger than the reader accepts (`MAX_TABLE_BYTES`) is refused before the folder is touched.
pub fn writeAtomic(alloc: std.mem.Allocator, io: std.Io, target: []const u8, bytes: []const u8) !void {
    // the reader refuses a file over this size, so writing it would replace a good table with one no read accepts
    if (bytes.len > MAX_TABLE_BYTES) return error.TableTooLarge;
    // Writers in one process share a temporary name (`<target>.tmp-<pid>`), so they take turns: one never sees the
    // other's half-written file as a stale leftover.
    write_mu.lock();
    defer write_mu.unlock();
    const dir_path = std.fs.path.dirname(target) orelse ".";
    const base = std.fs.path.basename(target);
    var dbuf: [std.fs.max_path_bytes]u8 = undefined;
    const dz = cPath(&dbuf, dir_path) orelse return error.NameTooLong;
    var gbuf: [std.fs.max_path_bytes]u8 = undefined;
    const gz = cPath(&gbuf, base) orelse return error.NameTooLong;
    // basename() drops a trailing slash, which would move the write one level up: such a path names a folder, not a file
    if (target.len == 0 or target[target.len - 1] == '/') return error.CannotCreateTemporary;
    const tmp_base = try std.fmt.allocPrint(alloc, "{s}.tmp-{d}", .{ base, std.c.getpid() });
    defer alloc.free(tmp_base);
    var tbuf: [std.fs.max_path_bytes]u8 = undefined;
    const tz = cPath(&tbuf, tmp_base) orelse return error.NameTooLong;

    _ = std.c.mkdir(dz, @as(std.c.mode_t, 0o700)); // an existing folder keeps its mode
    // The folder is opened once, and every step after the mode check (leftover removal, temporary creation, rename,
    // folder sync) goes through this descriptor, so a renamed ancestor or a retargeted symlink cannot move the write into
    // a folder that was not checked. NONBLOCK: a FIFO at the folder path must not hang the open; it is refused below.
    const dfd = std.c.open(dz, .{ .ACCMODE = .RDONLY, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (dfd < 0) return error.FolderUnavailable;
    defer _ = std.c.close(dfd);
    // A folder others can write to lets them swap the temporary entry between the sync and the rename, and the reader
    // refuses such a folder anyway: so nothing is written there.
    var dst: std.c.Stat = undefined;
    if (std.c.fstat(dfd, &dst) != 0) return error.FolderUnavailable;
    if (@as(u32, @intCast(dst.mode)) & S_IFMT != S_IFDIR) return error.FolderUnavailable;
    if (@as(u32, @intCast(dst.mode)) & 0o022 != 0) return error.UnsafeFolder;
    removeLeftovers(io, dir_path, base, dfd);

    // Exclusive creation: never open (or follow a symlink at) a name that exists. A stale file at this exact name is ours
    // from an earlier run of this pid, so it is unlinked once and the creation retried.
    var fd = std.c.openat(dfd, tz, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0 and std.c.errno(fd) == .EXIST) {
        _ = unlinkat(dfd, tz, 0);
        fd = std.c.openat(dfd, tz, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
    }
    if (fd < 0) return error.CannotCreateTemporary;
    var closed = false;
    var keep_tmp = false; // the test seam stands in for a killed writer, which leaves its temporary file
    // declared after `defer close(dfd)`, so it runs first, while `dfd` is still open
    errdefer {
        if (!closed) _ = std.c.close(fd);
        if (!keep_tmp) _ = unlinkat(dfd, tz, 0);
    }
    if (std.c.fchmod(fd, @as(std.c.mode_t, 0o600)) != 0) return error.ChmodFailed;
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n < 0 and std.c.errno(n) == .INTR) continue; // interrupted by a signal: retry at the same offset
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
    if (std.c.fsync(fd) != 0) return error.FsyncFailed;
    closed = true; // closed below whatever its result; the descriptor is not usable after a failed close either
    if (std.c.close(fd) != 0) return error.CloseFailed;
    if (write_atomic_stop_before_rename) {
        keep_tmp = true;
        return error.TestStopped;
    }

    if (renameat(dfd, tz, dfd, gz) != 0) return error.RenameFailed;
    // The rename is done; a folder that cannot be synced means the new table is not confirmed durable, which the caller
    // must hear about.
    if (std.c.fsync(dfd) != 0) return error.FolderSyncFailed;
}

// ---------------------------------------------------------------------------------------------------------------
// Where the table lives (spec 3.2)

/// `MLX_SERVE_KERNEL_TABLE`: unset, `off` (or empty: the same), or a path.
pub const TableSwitch = union(enum) { unset, off, empty, path: []const u8 };

pub fn tableSwitchFromEnv(raw: ?[]const u8) TableSwitch {
    const v = raw orelse return .unset;
    if (v.len == 0) return .empty;
    if (std.mem.eql(u8, v, "off")) return .off;
    return .{ .path = v };
}

/// `<home>/.mlx-serve/tune`, null when `home` is null: the one place the default folder is derived.
pub fn tuneRoot(alloc: std.mem.Allocator, home: ?[]const u8) !?[]u8 {
    const h = home orelse return null;
    if (h.len == 0) return null; // an empty HOME names no folder: it would read as /.mlx-serve/tune
    return try std.fmt.allocPrint(alloc, "{s}/.mlx-serve/tune", .{h});
}

/// The table path every `lesto tune` command and the engine use: the custom path for `.path`, else
/// `<tune_root>/kernel-plans.json`; null when there is neither a custom path nor a folder.
pub fn tablePath(alloc: std.mem.Allocator, tune_root: ?[]const u8, sw: TableSwitch) !?[]u8 {
    switch (sw) {
        .path => |p| return try alloc.dupe(u8, p),
        else => {},
    }
    const root = tune_root orelse return null;
    return try std.fmt.allocPrint(alloc, "{s}/kernel-plans.json", .{root});
}

/// `~` for a path under `home`, the path as given otherwise.
pub fn printablePath(buf: []u8, home: ?[]const u8, path: []const u8) []const u8 {
    const h = home orelse return path;
    if (h.len > 0 and std.mem.startsWith(u8, path, h) and (path.len == h.len or path[h.len] == '/')) {
        return std.fmt.bufPrint(buf, "~{s}", .{path[h.len..]}) catch path;
    }
    return path;
}

// ---------------------------------------------------------------------------------------------------------------
// Tests

const testing = std.testing;

const SampleKey = Key{ .gpu_arch = "applegpu_g16s", .gpu_cores = 20, .hw_model = "Mac16,7", .os_build = "25B78", .kernel_hash = "0000000000000000000000000000000000000000000000000000000000000000", .mlx_version = "0.31.1" };

fn sampleTable() Table {
    const ce = ConfirmStat{ .median_ns = -120, .lo_ns = -300, .hi_ns = -10, .window = 2 };
    const ev = KernelEvidence{ .ref_ns = 90_000, .pick_ns = 85_000, .confirm = .{ ce, ce, ce, ce } };
    var disc: [7]Discarded = undefined;
    for (DISCARD_REASONS, 0..) |r, i| disc[i] = .{ .reason = r, .n = @intCast(i) };
    return .{
        .schema = 2,
        .key = SampleKey,
        .shape = TUNED_SHAPE,
        .plan = .{ .gateup = .{ .layout = .{ .sg = 32, .ax = 1 } }, .down = .{ .layout = .{ .sps = 6, .tpt = 1, .ax = 1, .body = .scalar } } },
        .evidence = .{ .gateup = ev, .down = ev },
        .identity = .{ .plans_checked = 58, .cases = 10, .calls_compared = 680, .os_build = "25B78", .all_identical = true, .excluded = &.{
            .{ .kernel = "gateup", .cand = "sg=1 ax=0", .reason = "bits differ in case 3" },
            .{ .kernel = "down", .cand = "sps=2 tpt=4 ax=0", .reason = "launch refused: too many threads" },
        } },
        .info = .{
            .lesto_version = "0.26.0",
            .tuned_at = "2026-11-21T10:00:00Z",
            .signals = "real",
            .windows_used = 6,
            .batches_kept = 88,
            .batches_discarded = disc,
            .not_measurable = &.{.{ .kernel = "down", .cand = "sps=3 tpt=2 ax=1" }},
            .served_reasons = .{ .gateup = null, .down = "below_floor" },
        },
        .checksum = "",
    };
}

/// A finalized sample table's text in a fresh arena.
fn sampleText(arena: std.mem.Allocator) ![]u8 {
    var t = sampleTable();
    try finalize(arena, &t);
    return serialize(arena, t);
}

/// Replaces the first `from` in `text` with `to`, then rewrites the trailing checksum to match the edited table.
fn editFixed(arena: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]u8 {
    const edited = try std.mem.replaceOwned(u8, arena, text, from, to);
    try testing.expect(!std.mem.eql(u8, edited, text));
    const parsed = try std.json.parseFromSlice(Table, arena, edited, .{});
    const hex = try checksumHex(arena, parsed.value);
    const mark = "\"checksum\":\"";
    const at = std.mem.lastIndexOf(u8, edited, mark).? + mark.len;
    @memcpy(edited[at .. at + 64], &hex);
    return edited;
}

/// Writes `bytes` as `<tmp>/<name>` (mode 0600 in a 0700 folder) and returns the absolute path.
fn putFile(arena: std.mem.Allocator, tmp: *testing.TmpDir, name: []const u8, bytes: []const u8) ![]u8 {
    const io = testing.io;
    try tmp.dir.setPermissions(io, .fromMode(0o700));
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });
    const f = try tmp.dir.openFile(io, name, .{});
    defer f.close(io);
    try f.setPermissions(io, .fromMode(0o600));
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(io, &dir_buf)];
    return std.fs.path.join(arena, &.{ dir, name });
}

fn expectRefusal(alloc: std.mem.Allocator, path: []const u8, want: KernelTableRefusal) !void {
    const r = readTable(alloc, testing.io, path, SampleKey);
    defer r.deinit();
    const got = r.refusal() orelse return error.TestExpectedRefusal;
    try testing.expectEqual(want, got.reason);
}

test "kernel_table: round trip keeps every field, the three lists included" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try sampleText(a);
    const parsed = try std.json.parseFromSlice(Table, a, text, .{ .ignore_unknown_fields = false });
    const t = parsed.value;
    try testing.expectEqual(@as(usize, 2), t.identity.excluded.len);
    try testing.expectEqualStrings("launch refused: too many threads", t.identity.excluded[1].reason);
    try testing.expectEqual(@as(usize, 1), t.info.not_measurable.len);
    try testing.expectEqual(@as(usize, 7), t.info.batches_discarded.len);
    try testing.expectEqual(@as(u32, 6), t.info.batches_discarded[6].n);
    try testing.expectEqual(@as(?[]const u8, null), t.info.served_reasons.gateup);
    try testing.expectEqualStrings("below_floor", t.info.served_reasons.down.?);
    try testing.expectEqual(@as(i64, -300), t.evidence.gateup.?.confirm[3].lo_ns);
    const again = try serialize(a, t);
    try testing.expectEqualStrings(text, again);
    const third = try serialize(a, t);
    try testing.expectEqualStrings(again, third);
    var served = sampleTable();
    served.plan = .{ .gateup = .served, .down = .served };
    served.evidence = .{ .gateup = null, .down = null };
    const st = try serialize(a, served);
    try testing.expect(std.mem.indexOf(u8, st, "\"gateup\":\"served\"") != null);
    const sp = try std.json.parseFromSlice(Table, a, st, .{});
    try testing.expectEqual(GateUpEntry.served, sp.value.plan.gateup);
    try testing.expectEqual(DownEntry.served, sp.value.plan.down);
    try testing.expect(sp.value.evidence.gateup == null and sp.value.evidence.down == null);
}

test "kernel_table: field order is the spec's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const text = try sampleText(arena.allocator());
    const names = [_][]const u8{ "\"schema\":", "\"key\":", "\"shape\":", "\"plan\":", "\"evidence\":", "\"identity\":", "\"info\":", "\"checksum\":" };
    var last: usize = 0;
    for (names) |n| {
        const at = std.mem.indexOf(u8, text, n) orelse return error.TestKeyMissing;
        try testing.expect(at >= last);
        last = at;
    }
    var rlast: usize = 0;
    // the order is the spec's (section 4.4), written out here, not read back from the constant that builds the fixture
    const spec_order = [_][]const u8{ "input", "power", "low_power", "thermal", "sleep", "stopped", "gate" };
    for (spec_order) |r| {
        var nb: [32]u8 = undefined;
        const needle = try std.fmt.bufPrint(&nb, "{{\"reason\":\"{s}\"", .{r});
        const at = std.mem.indexOf(u8, text, needle) orelse return error.TestReasonMissing;
        try testing.expect(at >= rlast);
        rlast = at;
    }
}

test "kernel_table: refused unreadable (mode 0000)" {
    if (std.c.geteuid() == 0) return error.SkipZigTest; // root can read a mode 0000 file: the check would mean nothing
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "t.json", try sampleText(a));
    const f = try tmp.dir.openFile(testing.io, "t.json", .{});
    defer f.close(testing.io);
    try f.setPermissions(testing.io, .fromMode(0o000));
    try expectRefusal(testing.allocator, path, .unreadable);
}

test "kernel_table: refused unparseable (not JSON)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", "this is not json"), .unparseable);
}

test "kernel_table: refused unparseable (unknown field)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const text = try sampleText(a);
    const edited = try std.mem.concat(a, u8, &.{ "{\"extra\":1,", text[1..] });
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .unparseable);
}

test "kernel_table: refused unparseable (missing field)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try std.mem.replaceOwned(u8, a, try sampleText(a), "\"cases\":10,", "");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .unparseable);
}

test "kernel_table: refused unknown_schema (schema 3)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "{\"schema\":2,", "{\"schema\":3,");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .unknown_schema);
}

test "kernel_table: a 0.26.0 table is refused unknown_schema" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a schema 1 file: no `body` in the down object, which a full parse would call unparseable
    const v1 = try std.mem.replaceOwned(u8, a, try sampleText(a), "{\"schema\":2,", "{\"schema\":1,");
    const old = try std.mem.replaceOwned(u8, a, v1, ",\"body\":0", "");
    try testing.expect(std.mem.indexOf(u8, old, "\"body\"") == null);
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", old), .unknown_schema);
}

test "kernel_table: refused unparseable (down without body)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try std.mem.replaceOwned(u8, a, try sampleText(a), ",\"body\":0", "");
    try testing.expect(std.mem.indexOf(u8, edited, "{\"schema\":2,") != null);
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .unparseable);
}

test "kernel_table: refused plan_value_out_of_range (body 2)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"body\":0", "\"body\":2");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .plan_value_out_of_range);
}

test "kernel_table: the served triple with the array body is a layout object" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var t = sampleTable();
    t.plan.down = .{ .layout = .{ .sps = 1, .tpt = 1, .ax = 0, .body = .array } };
    const text = try serialize(a, t);
    try testing.expect(std.mem.indexOf(u8, text, "{\"sps\":1,\"tpt\":1,\"ax\":0,\"body\":1}") != null);
    try testing.expect(std.mem.indexOf(u8, text, "\"down\":\"served\"") == null);
    const parsed = try std.json.parseFromSlice(Table, a, text, .{});
    try testing.expectEqual(MoeRowsLayoutDown{ .sps = 1, .tpt = 1, .ax = 0, .body = .array }, parsed.value.plan.down.layout);
    // both bodies round trip at a pick with its own text
    var scalar = sampleTable();
    scalar.plan.down = .{ .layout = .{ .sps = 6, .tpt = 4, .ax = 1, .body = .scalar } };
    const stext = try serialize(a, scalar);
    try testing.expect(std.mem.indexOf(u8, stext, "{\"sps\":6,\"tpt\":4,\"ax\":1,\"body\":0}") != null);
    const sparsed = try std.json.parseFromSlice(Table, a, stext, .{});
    try testing.expectEqual(MoeRowsLayoutDown{ .sps = 6, .tpt = 4, .ax = 1, .body = .scalar }, sparsed.value.plan.down.layout);
}

test "kernel_table: downInSpace over body" {
    try testing.expect(downInSpace(.{ .sps = 6, .tpt = 4, .ax = 1, .body = .array }));
    try testing.expect(downInSpace(.{ .sps = 1, .tpt = 1, .ax = 0, .body = .array }));
    try testing.expect(downInSpace(.{ .sps = 1, .tpt = 1, .ax = 0, .body = .scalar }));
    try testing.expect(!downInSpace(.{ .sps = 4, .tpt = 1, .ax = 0, .body = .array }));
    try testing.expect(!downInSpace(.{ .sps = 1, .tpt = 3, .ax = 0, .body = .array }));
}

test "kernel_table: refused checksum_mismatch (one edited digit)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try std.mem.replaceOwned(u8, a, try sampleText(a), "\"windows_used\":6", "\"windows_used\":7");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .checksum_mismatch);
}

test "kernel_table: refused shape_unknown (inter 705)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"inter\":704", "\"inter\":705");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .shape_unknown);
}

test "kernel_table: refused shape_unknown (dtype float16)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"dtype\":\"bfloat16\"", "\"dtype\":\"float16\"");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .shape_unknown);
}

test "kernel_table: refused plan_value_out_of_range (sg 33)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"sg\":32", "\"sg\":33");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .plan_value_out_of_range);
}

test "kernel_table: refused plan_outside_space (sps 4)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"sps\":6", "\"sps\":4");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .plan_outside_space);
}

test "kernel_table: refused plan_outside_space (tpt 3)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"tpt\":1", "\"tpt\":3");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .plan_outside_space);
}

test "kernel_table: refused identity_not_true" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"all_identical\":true", "\"all_identical\":false");
    try expectRefusal(testing.allocator, try putFile(a, &tmp, "t.json", edited), .identity_not_true);
}

/// One `key_mismatch` case: the Mac differs from the table in `field` only.
fn keyMismatchCase(field: KeyField) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "t.json", try sampleText(a));
    var mac = SampleKey;
    switch (field) {
        .gpu_arch => mac.gpu_arch = "applegpu_g17p",
        .gpu_cores => mac.gpu_cores = 40,
        .hw_model => mac.hw_model = "Mac17,1",
        .kernel_hash => mac.kernel_hash = "1111111111111111111111111111111111111111111111111111111111111111",
        .mlx_version => mac.mlx_version = "0.32.0",
        .os_build => mac.os_build = "25C1",
    }
    const r = readTable(testing.allocator, testing.io, path, mac);
    defer r.deinit();
    switch (r) {
        .key_mismatch => |k| {
            try testing.expectEqual(KernelTableRefusal.key_mismatch, k.detail.reason);
            try testing.expectEqual(field, k.detail.field.?);
            try testing.expectEqualStrings("25B78", k.table.value.key.os_build);
            const table_value: []const u8 = switch (field) {
                .gpu_arch => SampleKey.gpu_arch,
                .gpu_cores => "",
                .hw_model => SampleKey.hw_model,
                .kernel_hash => SampleKey.kernel_hash,
                .mlx_version => SampleKey.mlx_version,
                .os_build => SampleKey.os_build,
            };
            const mac_value: []const u8 = switch (field) {
                .gpu_arch => mac.gpu_arch,
                .gpu_cores => "",
                .hw_model => mac.hw_model,
                .kernel_hash => mac.kernel_hash,
                .mlx_version => mac.mlx_version,
                .os_build => mac.os_build,
            };
            try testing.expectEqualStrings(table_value, k.detail.table_value);
            try testing.expectEqualStrings(mac_value, k.detail.mac_value);
            if (field == .gpu_cores) {
                try testing.expectEqual(@as(u32, 20), k.detail.table_cores);
                try testing.expectEqual(@as(u32, 40), k.detail.mac_cores);
            }
            var buf: [512]u8 = undefined;
            const line = k.detail.text(&buf);
            try testing.expect(std.mem.indexOf(u8, line, @tagName(field)) != null);
            if (field == .gpu_cores) try testing.expect(std.mem.indexOf(u8, line, "is 20 in the table and 40 on this Mac") != null);
            const verify = std.mem.indexOf(u8, line, "`lesto tune --verify`") != null;
            try testing.expectEqual(field == .os_build, verify);
            if (field != .os_build) try testing.expect(std.mem.indexOf(u8, line, "`lesto tune`") != null);
        },
        else => return error.TestExpectedKeyMismatch,
    }
}

test "kernel_table: key_mismatch gpu_arch" {
    try keyMismatchCase(.gpu_arch);
}
test "kernel_table: key_mismatch gpu_cores" {
    try keyMismatchCase(.gpu_cores);
}
test "kernel_table: key_mismatch hw_model" {
    try keyMismatchCase(.hw_model);
}
test "kernel_table: key_mismatch kernel_hash" {
    try keyMismatchCase(.kernel_hash);
}
test "kernel_table: key_mismatch mlx_version" {
    try keyMismatchCase(.mlx_version);
}
test "kernel_table: key_mismatch os_build" {
    try keyMismatchCase(.os_build);
}

test "kernel_table: the first failing check wins" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const edited = try editFixed(a, try sampleText(a), "\"inter\":704", "\"inter\":705");
    const path = try putFile(a, &tmp, "t.json", edited);
    var mac = SampleKey;
    mac.os_build = "25C1";
    const r = readTable(testing.allocator, testing.io, path, mac);
    defer r.deinit();
    try testing.expectEqual(KernelTableRefusal.shape_unknown, r.refusal().?.reason);
    // only os_build changed: key_mismatch with that field
    var tmp2 = testing.tmpDir(.{ .iterate = true });
    defer tmp2.cleanup();
    const path2 = try putFile(a, &tmp2, "t.json", try sampleText(a));
    const r2 = readTable(testing.allocator, testing.io, path2, mac);
    defer r2.deinit();
    try testing.expectEqual(KeyField.os_build, r2.refusal().?.field.?);
}

test "kernel_table: permissions" {
    const uid: u32 = 501;
    try testing.expectEqual(@as(?KernelTableRefusal, .unsafe_permissions), tablePermsSafe(uid, .{ .uid = uid, .mode = 0o620 }, .{ .uid = uid, .mode = 0o700 }));
    try testing.expectEqual(@as(?KernelTableRefusal, .unsafe_permissions), tablePermsSafe(uid, .{ .uid = uid, .mode = 0o600 }, .{ .uid = uid, .mode = 0o707 }));
    try testing.expectEqual(@as(?KernelTableRefusal, .unsafe_permissions), tablePermsSafe(uid, .{ .uid = uid + 1, .mode = 0o600 }, .{ .uid = uid, .mode = 0o700 }));
    try testing.expectEqual(@as(?KernelTableRefusal, null), tablePermsSafe(uid, .{ .uid = uid, .mode = 0o600 }, .{ .uid = uid, .mode = 0o700 }));
    // through readTable: the file holds invalid JSON, so a parse would say `unparseable`
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "t.json", "not json");
    const f = try tmp.dir.openFile(testing.io, "t.json", .{});
    defer f.close(testing.io);
    try f.setPermissions(testing.io, .fromMode(0o620));
    try expectRefusal(testing.allocator, path, .unsafe_permissions);
    try f.setPermissions(testing.io, .fromMode(0o600));
    try tmp.dir.setPermissions(testing.io, .fromMode(0o707));
    try expectRefusal(testing.allocator, path, .unsafe_permissions);
    // a custom path in a world-writable folder is refused the same way
    var tmp2 = testing.tmpDir(.{ .iterate = true });
    defer tmp2.cleanup();
    const path2 = try putFile(a, &tmp2, "custom.json", "not json");
    try tmp2.dir.setPermissions(testing.io, .fromMode(0o777));
    try expectRefusal(testing.allocator, path2, .unsafe_permissions);
}

test "kernel_table: a missing file is absent" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "other.json", "x");
    const missing = try std.fs.path.join(a, &.{ std.fs.path.dirname(path).?, "kernel-plans.json" });
    const r = readTable(testing.allocator, testing.io, missing, SampleKey);
    defer r.deinit();
    try testing.expect(r == .absent);
    try testing.expect(r.refusal() == null);
}

test "kernel_table: a directory at the table path is unreadable" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    _ = try putFile(a, &tmp, "other.json", "x");
    try tmp.dir.createDir(testing.io, "table.json", .default_dir);
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const path = try std.fs.path.join(a, &.{ dir, "table.json" });
    try expectRefusal(testing.allocator, path, .unreadable);
}

fn countTmpFiles(dir: std.Io.Dir) !usize {
    var n: usize = 0;
    var it = dir.iterate();
    while (try it.next(testing.io)) |e| {
        if (std.mem.indexOf(u8, e.name, ".tmp-") != null) n += 1;
    }
    return n;
}

test "kernel_table: an interrupted write leaves the old table" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const text_a = try sampleText(a);
    const path = try putFile(a, &tmp, "kernel-plans.json", text_a);
    var tb = sampleTable();
    tb.info.windows_used = 9;
    try finalize(a, &tb);
    const text_b = try serialize(a, tb);
    write_atomic_stop_before_rename = true;
    defer write_atomic_stop_before_rename = false;
    try testing.expectError(error.TestStopped, writeAtomic(testing.allocator, testing.io, path, text_b));
    const r = readTable(testing.allocator, testing.io, path, SampleKey);
    defer r.deinit();
    try testing.expectEqual(@as(u32, 6), r.valid.value.info.windows_used);
    try testing.expectEqual(@as(usize, 1), try countTmpFiles(tmp.dir));
}

test "kernel_table: a leftover temporary file is ignored and then removed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    var tb = sampleTable();
    tb.info.windows_used = 9;
    try finalize(a, &tb);
    const text_b = try serialize(a, tb);
    {
        write_atomic_stop_before_rename = true;
        defer write_atomic_stop_before_rename = false;
        try testing.expectError(error.TestStopped, writeAtomic(testing.allocator, testing.io, path, text_b));
    }
    try testing.expectEqual(@as(usize, 1), try countTmpFiles(tmp.dir));
    {
        const r = readTable(testing.allocator, testing.io, path, SampleKey);
        defer r.deinit();
        try testing.expectEqual(@as(u32, 6), r.valid.value.info.windows_used);
    }
    try writeAtomic(testing.allocator, testing.io, path, text_b);
    try testing.expectEqual(@as(usize, 0), try countTmpFiles(tmp.dir));
    const r = readTable(testing.allocator, testing.io, path, SampleKey);
    defer r.deinit();
    try testing.expectEqual(@as(u32, 9), r.valid.value.info.windows_used);
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const pz = cPath(&pb, path).?;
    var st: std.c.Stat = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.stat(pz, &st));
    try testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.mode)) & 0o777);
}

test "kernel_table: a valid table of exactly the read limit is written and read back" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const base = try sampleText(a);
    const path = try putFile(a, &tmp, "kernel-plans.json", base);
    // The checksum is always 64 characters, so padding `lesto_version` by the missing bytes gives exactly the limit.
    var t = sampleTable();
    try testing.expect(base.len < MAX_TABLE_BYTES);
    const extra = MAX_TABLE_BYTES - base.len;
    const version = try a.alloc(u8, t.info.lesto_version.len + extra);
    @memcpy(version[0..t.info.lesto_version.len], t.info.lesto_version);
    @memset(version[t.info.lesto_version.len..], 'x');
    t.info.lesto_version = version;
    try finalize(a, &t);
    const bytes = try serialize(a, t);
    try testing.expectEqual(MAX_TABLE_BYTES, bytes.len);
    try writeAtomic(testing.allocator, testing.io, path, bytes);
    const r = readTable(testing.allocator, testing.io, path, SampleKey);
    defer r.deinit();
    try testing.expect(r == .valid);
    try testing.expectEqualStrings(version, r.valid.value.info.lesto_version);
}

test "kernel_table: a table over the read limit is refused and the old table stays readable" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    const pad = try a.alloc(u8, MAX_TABLE_BYTES + 1);
    @memset(pad, ' ');
    try testing.expectError(error.TableTooLarge, writeAtomic(testing.allocator, testing.io, path, pad));
    const r = readTable(testing.allocator, testing.io, path, SampleKey);
    defer r.deinit();
    try testing.expect(r == .valid);
    try testing.expectEqual(@as(usize, 0), try countTmpFiles(tmp.dir));
}

test "kernel_table: kernelHash" {
    const base = [7][]const u8{ "part-one", "part-two", "part-three", "part-four", "part-five", "part-six", "part-seven" };
    const h0 = kernelHash(&base);
    for (0..7) |i| {
        var buf: [32]u8 = undefined;
        @memcpy(buf[0..base[i].len], base[i]);
        buf[0] ^= 1;
        var parts = base;
        parts[i] = buf[0..base[i].len];
        try testing.expect(!std.mem.eql(u8, &h0, &kernelHash(&parts)));
    }
    var swapped = base;
    swapped[0] = base[1];
    swapped[1] = base[0];
    try testing.expect(!std.mem.eql(u8, &h0, &kernelHash(&swapped)));
    try testing.expect(!std.mem.eql(u8, &kernelHash(&.{ "ab", "c" }), &kernelHash(&.{ "a", "bc" })));
    // the engine's own eight parts: none empty, the array body's source is part 5 (before the header), and the hash is
    // stable within the process
    const parts = engineParts();
    try testing.expectEqual(@as(usize, 8), parts.len);
    for (parts) |p| try testing.expect(p.len > 0);
    try testing.expectEqualStrings(transformer.GQMV_DOWNRED_ROWS_ARRAY_SOURCE, parts[4]);
    try testing.expectEqualStrings(transformer.GQMV_AFFINE_HEADER, parts[5]);
    try testing.expectEqual(@as(usize, 4), parts[7].len);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, parts[7][0..4], .little));
    try testing.expectEqualStrings(&engineKernelHashHex(), &engineKernelHashHex());
    // and it is the digest of those eight parts, not whatever the cache happened to hold
    const want = std.fmt.bytesToHex(kernelHash(&parts), .lower);
    try testing.expectEqualStrings(&want, &engineKernelHashHex());
}

test "kernel_table: a path with an embedded NUL is refused, not cut short" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const real = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    const cut = try std.mem.concat(a, u8, &.{ real, "\x00junk" });
    // read: the part before the NUL is a valid table, and must not be read through the longer name
    try expectRefusal(testing.allocator, cut, .unreadable);
    // write: refused, and the existing table is untouched
    try testing.expectError(error.NameTooLong, writeAtomic(testing.allocator, testing.io, cut, "x"));
    const r = readTable(testing.allocator, testing.io, real, SampleKey);
    defer r.deinit();
    try testing.expect(r == .valid);
}

extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;

test "kernel_table: a FIFO at the table path is unreadable and never blocks" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const other = try putFile(a, &tmp, "other.json", "x");
    const fifo = try std.fs.path.join(a, &.{ std.fs.path.dirname(other).?, "table.json" });
    const fz = try a.dupeSentinel(u8, fifo, 0);
    try testing.expectEqual(@as(c_int, 0), mkfifo(fz.ptr, 0o600));
    try expectRefusal(testing.allocator, fifo, .unreadable);
}

test "kernel_table: the writer removes only dead writers' temporary files" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    // a sibling that is not a writer's temporary file, one owned by a live process (pid 1), one by a dead pid
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "kernel-plans.json.tmp-backup", .data = "keep" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "kernel-plans.json.tmp-1", .data = "live" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "kernel-plans.json.tmp-2147483000", .data = "dead" });
    var tb = sampleTable();
    try finalize(a, &tb);
    try writeAtomic(testing.allocator, testing.io, path, try serialize(a, tb));
    try tmp.dir.access(testing.io, "kernel-plans.json.tmp-backup", .{});
    try tmp.dir.access(testing.io, "kernel-plans.json.tmp-1", .{});
    try testing.expectError(error.FileNotFound, tmp.dir.access(testing.io, "kernel-plans.json.tmp-2147483000", .{}));
}

test "kernel_table: a trailing slash and a symlinked table are refused, the plain path is read" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    {
        const ok = readTable(testing.allocator, testing.io, path, SampleKey);
        defer ok.deinit();
        try testing.expect(ok == .valid);
    }
    try expectRefusal(testing.allocator, try std.mem.concat(a, u8, &.{ path, "/" }), .unreadable);
    const link = try std.fs.path.joinZ(a, &.{ std.fs.path.dirname(path).?, "link.json" });
    const target_z = try a.dupeSentinel(u8, path, 0);
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(target_z.ptr, link.ptr));
    try expectRefusal(testing.allocator, link, .unreadable);
}

test "kernel_table: temporary names that are not a pid are kept, two core mismatches keep their own digits" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "kernel-plans.json.tmp-+123", .data = "x" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "kernel-plans.json.tmp-00042", .data = "x" });
    var tb = sampleTable();
    try finalize(a, &tb);
    try writeAtomic(testing.allocator, testing.io, path, try serialize(a, tb));
    try tmp.dir.access(testing.io, "kernel-plans.json.tmp-+123", .{});
    try tmp.dir.access(testing.io, "kernel-plans.json.tmp-00042", .{});
    // more core mismatches than any buffer a detail could borrow: each detail still prints its own counts
    var details: [20]RefusalDetail = undefined;
    for (&details, 0..) |*d, i| {
        var mac = SampleKey;
        mac.gpu_cores = @intCast(100 + i);
        d.* = checkKey(sampleTable(), mac).?;
    }
    for (details, 0..) |d, i| {
        var buf: [256]u8 = undefined;
        var want: [48]u8 = undefined;
        const w = try std.fmt.bufPrint(&want, "and {d} on this Mac", .{100 + i});
        try testing.expect(std.mem.indexOf(u8, d.text(&buf), w) != null);
    }
}

test "kernel_table: nothing is written into a folder others can write to" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    var tb = sampleTable();
    try finalize(a, &tb);
    const bytes = try serialize(a, tb);
    try tmp.dir.setPermissions(testing.io, .fromMode(0o770));
    try testing.expectError(error.UnsafeFolder, writeAtomic(testing.allocator, testing.io, path, bytes));
    try tmp.dir.setPermissions(testing.io, .fromMode(0o700));
    try writeAtomic(testing.allocator, testing.io, path, bytes);
}

test "kernel_table: a symlink at the temporary path is never followed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try putFile(a, &tmp, "kernel-plans.json", try sampleText(a));
    const victim = try putFile(a, &tmp, "victim.txt", "precious");
    const link = try std.fmt.allocPrintSentinel(a, "{s}.tmp-{d}", .{ path, std.c.getpid() }, 0);
    const victim_z = try a.dupeSentinel(u8, victim, 0);
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(victim_z.ptr, link.ptr));
    var tb = sampleTable();
    tb.info.windows_used = 9;
    try finalize(a, &tb);
    try writeAtomic(testing.allocator, testing.io, path, try serialize(a, tb));
    var buf: [32]u8 = undefined;
    const got = try tmp.dir.readFile(testing.io, "victim.txt", &buf);
    try testing.expectEqualStrings("precious", got);
    const r = readTable(testing.allocator, testing.io, path, SampleKey);
    defer r.deinit();
    try testing.expectEqual(@as(u32, 9), r.valid.value.info.windows_used);
}

test "kernel_table: the rule texts are the three functions" {
    const r = ruleTexts();
    try testing.expect(std.mem.startsWith(u8, r[0], "fn downredLanesFor("));
    try testing.expect(std.mem.startsWith(u8, r[1], "fn moeRowsLayoutGateUp("));
    try testing.expect(std.mem.startsWith(u8, r[2], "fn moeRowsLayoutDown("));
    for (r) |t| try testing.expect(std.mem.endsWith(u8, t, "\n}"));
}

test "kernel_table: the plan space" {
    const gu = gateUpCandidates();
    try testing.expectEqual(@as(usize, 11), gu.len);
    for (gu, 0..) |g, i| {
        try testing.expect(!std.meta.eql(g, transformer.MOE_ROWS_LAYOUT_GATEUP_SERVED));
        try testing.expect(gateUpInSpace(g));
        try testing.expect(@rem(704, g.sg) == 0 and g.sg * 32 <= 1024);
        for (gu[i + 1 ..]) |h| try testing.expect(!std.meta.eql(g, h));
    }
    const dn = downCandidates();
    // spec 6: SPS {1,2,3,6} x TPT {1,2,4} x AX {0,1} x body {scalar,array} is 48 points, minus the served one
    try testing.expectEqual(@as(usize, 47), dn.len);
    try testing.expectEqual(MoeRowsLayoutDown{ .sps = 1, .tpt = 1, .ax = 0, .body = .array }, dn[0]);
    try testing.expectEqual(MoeRowsLayoutDown{ .sps = 1, .tpt = 1, .ax = 1, .body = .scalar }, dn[1]);
    try testing.expectEqual(MoeRowsLayoutDown{ .sps = 6, .tpt = 4, .ax = 1, .body = .array }, dn[46]);
    var dbuf: [64]u8 = undefined;
    for (dn) |d| {
        try testing.expectEqual(d, try transformer.moeRowsLayoutParseDown(downText(&dbuf, d)));
    }
    for (dn, 0..) |d, i| {
        try testing.expect(!std.meta.eql(d, transformer.MOE_ROWS_LAYOUT_DOWN_SERVED));
        try testing.expect(downInSpace(d));
        try testing.expect(@rem(6, d.sps) == 0);
        try testing.expect(@rem(704, d.tpt) == 0 and @rem(1408, d.tpt) == 0);
        try testing.expect(@divExact(6, d.sps) * d.tpt * 32 <= 1024);
        for (dn[i + 1 ..]) |e| try testing.expect(!std.meta.eql(d, e));
    }
    try testing.expect(!gateUpInSpace(.{ .sg = 3, .ax = 0 }));
    try testing.expect(!downInSpace(.{ .sps = 4, .tpt = 1, .ax = 0 }));
    var buf: [64]u8 = undefined;
    const text = gateUpText(&buf, .{ .sg = 32, .ax = 1 });
    try testing.expectEqualStrings("sg=32 ax=1", text);
    try testing.expectEqual(MoeRowsLayoutGateUp{ .sg = 32, .ax = 1 }, try transformer.moeRowsLayoutParseGateUp(text));
    var buf2: [64]u8 = undefined;
    const dtext = downText(&buf2, .{ .sps = 6, .tpt = 1, .ax = 1, .body = .scalar });
    try testing.expectEqualStrings("sps=6 tpt=1 ax=1 body=0", dtext);
    try testing.expectEqual(MoeRowsLayoutDown{ .sps = 6, .tpt = 1, .ax = 1, .body = .scalar }, try transformer.moeRowsLayoutParseDown(dtext));
}

test "kernel_table: the table switch" {
    const a = testing.allocator;
    try testing.expectEqual(TableSwitch.unset, tableSwitchFromEnv(null));
    try testing.expectEqual(TableSwitch.off, tableSwitchFromEnv("off"));
    try testing.expectEqual(TableSwitch.empty, tableSwitchFromEnv(""));
    try testing.expectEqualStrings("/t/p.json", tableSwitchFromEnv("/t/p.json").path);
    try testing.expectEqual(@as(?[]u8, null), try tuneRoot(a, null));
    try testing.expectEqual(@as(?[]u8, null), try tuneRoot(a, "")); // an empty HOME is no HOME
    const root = (try tuneRoot(a, "/Users/x")).?;
    defer a.free(root);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/tune", root);
    try testing.expectEqual(@as(?[]u8, null), try tablePath(a, null, .unset));
    const custom = (try tablePath(a, null, .{ .path = "/t/p.json" })).?;
    defer a.free(custom);
    try testing.expectEqualStrings("/t/p.json", custom);
    const def = (try tablePath(a, "/r", .unset)).?;
    defer a.free(def);
    try testing.expectEqualStrings("/r/kernel-plans.json", def);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("~/.mlx-serve/tune/kernel-plans.json", printablePath(&buf, "/Users/x", "/Users/x/.mlx-serve/tune/kernel-plans.json"));
    try testing.expectEqualStrings("/other/p.json", printablePath(&buf, "/Users/x", "/other/p.json"));
}
