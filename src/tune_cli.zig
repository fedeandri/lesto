//! `lesto tune` (spec 7): the flags, the lock, the progress line, the default action (wait for idle windows, tune, write
//! the table), `--background` and `--agent` (the LaunchAgent), `--stop`, `--show`, `--verify` and `--reset`. The search
//! itself is the scheduler's (src/tune_sched.zig) over the GPU worker (src/tune_gpu.zig); the table and its checks are
//! src/kernel_table.zig; which layout the engine serves is `transformer.moeRowsLayoutDecideSources`, which `--show` calls too.
//!
//! `envFromProcess` is the only place this file reads the environment; every command takes an `Env`, so the tests build
//! one on a temporary folder and never touch the real `~/.mlx-serve/` or `~/Library/LaunchAgents/`, nor call `launchctl`.
const std = @import("std");
const build_options = @import("build_options");
const transformer = @import("transformer.zig");
const kernel_table = @import("kernel_table.zig");
const tune_signals = @import("tune_signals.zig");
const tune_stats = @import("tune_stats.zig");
const tune_sched = @import("tune_sched.zig");
const tune_gpu = @import("tune_gpu.zig");

const Allocator = std.mem.Allocator;
const ServedReason = tune_stats.ServedReason;
const MAX_PATH = std.fs.max_path_bytes;

pub const AGENT_LABEL = "com.lestoengine.lesto.tune";
const PLIST_NAME = AGENT_LABEL ++ ".plist";
const LOCK_NAME = "lock";
/// Serializes `--background` installs (the tune's `lock` is held by a running tune for hours).
const INSTALL_LOCK_NAME = "install.lock";
const LOG_NAME = "tune.log";
const PROGRESS_NAME = "progress.jsonl";
const AGENT_LOG_NAME = "agent.log";
const WORKER_SEED: u64 = 0x7475_6e65;
const DEFAULT_IDLE_MINUTES: u32 = 5;
/// The largest plist `--show` reads: far above the 1 to 2 KB this file writes, so only a foreign file reaches it.
const PLIST_READ_MAX: usize = 1 << 20;
const CANDIDATES = kernel_table.GATEUP_CANDIDATES + kernel_table.DOWN_CANDIDATES;
/// The screen times every candidate in both regimes (2 x CANDIDATES batches); the final stage adds 16 and the second
/// confirmation 4, so at most 2 * CANDIDATES + 20 timing batches with no discards.
const MAX_TIMING_BATCHES = 2 * CANDIDATES + 20;

const NO_HOME = "lesto tune: no tune folder (HOME is not set)";
const CHECK_MODEL = "lesto tune: --check-model is not in this release";
const OFF_NOTE = "[tune] MLX_SERVE_KERNEL_TABLE is off: the engine will not read the table while the switch is off";

const USAGE =
    \\usage: lesto tune [--background | --stop | --show | --verify | --reset] [--idle-minutes <n>]
    \\  (no flag)           tune in this terminal while the Mac is idle, then write the table
    \\  --background        install a LaunchAgent that tunes while the Mac is idle
    \\  --stop              remove the LaunchAgent and stop a running tune (progress is kept)
    \\  --show              print the table, the layouts the engine would use and the progress
    \\  --verify            re-check the table's layouts on this macOS build
    \\  --reset             delete the table and the progress
    \\  --idle-minutes <n>  minutes without input before a window opens (default 5); with no
    \\                      flag, --background or --verify
    \\
;

extern "c" fn flock(fd: c_int, op: c_int) c_int;
extern "c" fn kill(pid: c_int, sig: c_int) c_int;
extern "c" fn realpath(noalias file_name: [*:0]const u8, noalias resolved_name: [*]u8) ?[*:0]u8;
extern "c" fn proc_pidinfo(pid: c_int, flavor: c_int, arg: u64, buffer: ?*anyopaque, buffersize: c_int) c_int;
extern "c" fn time(t: ?*i64) i64;
extern "c" fn fork() c_int;
extern "c" fn pause() c_int;
extern "c" fn usleep(us: c_uint) c_int;

const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;
const LOCK_UN: c_int = 8;
const SIGTERM: c_int = 15;
const SIGKILL: c_int = 9;
const PROC_PIDTBSDINFO: c_int = 3;
const S_IFMT: u32 = 0o170000;
const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;
const S_IFLNK: u32 = 0o120000;

// ---------------------------------------------------------------------------------------------------------------
// The environment

pub const Env = struct {
    /// The engine's own view (spec 3.7): `home`, `tune_root`, `table_switch` and the three moe-rows-layout variables.
    engine: transformer.MoeRowsLayoutEnv,
    /// `<HOME>/Library/LaunchAgents`, null without `HOME`.
    launch_agents_dir: ?[]const u8,
    /// `MLX_SERVE_TUNE_FAKE_SIGNALS`: a scripted signals file, null for the real signals.
    fake_signals: ?[]const u8,
    stdout_is_tty: bool,
    /// False in the tests: `launchctl` is never called.
    launchctl: bool,
    out: *std.Io.Writer,
};

var stdout_buf: [4096]u8 = undefined;
var stdout_fw: std.Io.File.Writer = undefined;

fn getenvSlice(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.sliceTo(v, 0);
}

/// Reads the process environment once. Allocations go to `alloc` (the caller's arena).
pub fn envFromProcess(alloc: Allocator, io: std.Io) !Env {
    const engine = try transformer.moeRowsLayoutEnvFromProcess(alloc);
    const agents: ?[]const u8 = if (engine.home) |h| (if (h.len > 0) try std.fmt.allocPrint(alloc, "{s}/Library/LaunchAgents", .{h}) else null) else null;
    const fake = getenvSlice("MLX_SERVE_TUNE_FAKE_SIGNALS");
    stdout_fw = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    return .{
        .engine = engine,
        .launch_agents_dir = agents,
        .fake_signals = if (fake) |f| (if (f.len == 0) null else f) else null,
        .stdout_is_tty = std.c.isatty(1) == 1,
        .launchctl = true,
        .out = &stdout_fw.interface,
    };
}

/// Set when a write to `Env.out` failed during the command: `runWith` then fails the command (exit 2), because the
/// caller did not get the output it asked for (a plist cut short, a report missing lines).
var out_failed: bool = false;

fn outFail() void {
    out_failed = true;
}

fn say(env: Env, comptime fmt: []const u8, args: anytype) void {
    env.out.print(fmt ++ "\n", args) catch outFail();
    env.out.flush() catch outFail();
}

// ---------------------------------------------------------------------------------------------------------------
// Flags (spec 7.2)

pub const Action = enum { default, background, agent, stop, show, verify, reset };
pub const Flags = struct { action: Action, idle_minutes: u32 = DEFAULT_IDLE_MINUTES };
pub const Parsed = union(enum) { ok: Flags, usage: []const u8, check_model, print_plist: []const u8 };

const ACTION_FLAGS = [_]struct { flag: []const u8, action: Action }{
    .{ .flag = "--background", .action = .background },
    .{ .flag = "--agent", .action = .agent },
    .{ .flag = "--stop", .action = .stop },
    .{ .flag = "--show", .action = .show },
    .{ .flag = "--verify", .action = .verify },
    .{ .flag = "--reset", .action = .reset },
};

/// The longest idle threshold: one week. Its nanoseconds (about 6e14) fit a u64 with room to spare.
pub const MAX_IDLE_MINUTES: u32 = 7 * 24 * 60;
const IDLE_USAGE = std.fmt.comptimePrint("--idle-minutes needs a whole number from 1 to {d} (one week)", .{MAX_IDLE_MINUTES});

/// A whole number from 1 to `MAX_IDLE_MINUTES`, digits only (no sign, no unit, no fraction).
fn parseMinutes(s: []const u8) ?u32 {
    if (s.len == 0 or s.len > 9) return null;
    for (s) |c| if (c < '0' or c > '9') return null;
    const n = std.fmt.parseInt(u32, s, 10) catch return null;
    return if (n >= 1 and n <= MAX_IDLE_MINUTES) n else null;
}

pub fn parseFlags(args: []const []const u8) Parsed {
    var action: ?Action = null;
    var idle: ?u32 = null;
    var print_bin: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--check-model")) return .check_model;
        if (std.mem.eql(u8, a, "--print-plist-for-test")) {
            // the whole command line is still checked: it is the only action and takes exactly its one value
            if (i + 1 >= args.len) return .{ .usage = "--print-plist-for-test needs a binary path" };
            if (print_bin != null) return .{ .usage = "--print-plist-for-test is given once" };
            print_bin = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, a, "--idle-minutes")) {
            if (i + 1 >= args.len) return .{ .usage = IDLE_USAGE };
            idle = parseMinutes(args[i + 1]) orelse return .{ .usage = IDLE_USAGE };
            i += 1;
            continue;
        }
        const found: ?Action = for (ACTION_FLAGS) |f| {
            if (std.mem.eql(u8, a, f.flag)) break f.action;
        } else null;
        const this_action = found orelse return .{ .usage = "unknown flag or argument" };
        if (action != null) return .{ .usage = "at most one of --background, --stop, --show, --verify, --reset" };
        action = this_action;
    }
    if (print_bin) |bin| {
        if (action != null or idle != null) return .{ .usage = "--print-plist-for-test <bin> goes alone" };
        return .{ .print_plist = bin };
    }
    const act = action orelse .default;
    if (idle != null) switch (act) {
        .default, .background, .agent, .verify => {},
        else => return .{ .usage = "--idle-minutes goes with no action flag, --background or --verify" },
    };
    return .{ .ok = .{ .action = act, .idle_minutes = idle orelse DEFAULT_IDLE_MINUTES } };
}

// ---------------------------------------------------------------------------------------------------------------
// The LaunchAgent's plist (spec 7.5)

/// Escapes `s` for XML element text so a plist parser reads back exactly `s`: the five markup characters as entities,
/// a carriage return as `&#13;` (a literal one is normalized to a line feed), and `error.NotXmlText` for the control
/// characters XML 1.0 cannot carry in any form (below 0x20 except tab, line feed and carriage return).
/// Beyond ASCII, only valid UTF-8 (no overlong form, no surrogate half, nothing past U+10FFFF) whose code point XML 1.0
/// allows (not U+FFFE or U+FFFF) is copied; anything else is `error.NotXmlText`.
fn appendXml(l: *std.ArrayList(u8), alloc: Allocator, s: []const u8) !void {
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            const n = std.unicode.utf8ByteSequenceLength(c) catch return error.NotXmlText;
            if (i + n > s.len) return error.NotXmlText;
            const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return error.NotXmlText;
            if (cp == 0xFFFE or cp == 0xFFFF) return error.NotXmlText;
            try l.appendSlice(alloc, s[i .. i + n]);
            i += n;
            continue;
        }
        switch (c) {
            '&' => try l.appendSlice(alloc, "&amp;"),
            '<' => try l.appendSlice(alloc, "&lt;"),
            '>' => try l.appendSlice(alloc, "&gt;"),
            '"' => try l.appendSlice(alloc, "&quot;"),
            '\'' => try l.appendSlice(alloc, "&apos;"),
            '\r' => try l.appendSlice(alloc, "&#13;"),
            '\t', '\n' => try l.append(alloc, c),
            0...8, 11, 12, 14...31 => return error.NotXmlText,
            else => try l.append(alloc, c),
        }
        i += 1;
    }
}

fn appendString(l: *std.ArrayList(u8), alloc: Allocator, indent: []const u8, s: []const u8) !void {
    try l.appendSlice(alloc, indent);
    try l.appendSlice(alloc, "<string>");
    try appendXml(l, alloc, s);
    try l.appendSlice(alloc, "</string>\n");
}

pub fn plistText(alloc: Allocator, bin_path: []const u8, idle_minutes: u32, agent_log: []const u8) ![]u8 {
    return plistTextWith(alloc, bin_path, idle_minutes, agent_log, null);
}

/// `plistText` plus, when `table_path` is set, an `EnvironmentVariables` dict carrying `MLX_SERVE_KERNEL_TABLE` into the
/// job: launchd does not pass the installing shell's environment, so a custom table path would otherwise be lost.
/// `error.NotXmlText` when a path holds a byte or character a plist cannot carry.
pub fn plistTextWith(alloc: Allocator, bin_path: []const u8, idle_minutes: u32, agent_log: []const u8, table_path: ?[]const u8) ![]u8 {
    var l: std.ArrayList(u8) = .empty;
    errdefer l.deinit(alloc);
    var nb: [16]u8 = undefined;
    const n = std.fmt.bufPrint(&nb, "{d}", .{idle_minutes}) catch unreachable;
    try l.appendSlice(alloc, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" ++
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n" ++
        "<plist version=\"1.0\">\n<dict>\n\t<key>Label</key>\n");
    try appendString(&l, alloc, "\t", AGENT_LABEL);
    try l.appendSlice(alloc, "\t<key>ProgramArguments</key>\n\t<array>\n");
    for ([_][]const u8{ bin_path, "tune", "--agent", "--idle-minutes", n }) |s| try appendString(&l, alloc, "\t\t", s);
    // ProcessType Standard, not Background: launchd throttles a Background job, and on an M3 that made every timing
    // batch slower and unsteady (the noise gate discarded 185 to 195 batches a run and nothing was picked), while the
    // same binary in the foreground, or as a Standard agent, kept every batch. The agent still times only in idle windows.
    try l.appendSlice(alloc, "\t</array>\n\t<key>RunAtLoad</key>\n\t<true/>\n\t<key>KeepAlive</key>\n\t<false/>\n" ++
        "\t<key>ProcessType</key>\n\t<string>Standard</string>\n\t<key>StandardOutPath</key>\n");
    try appendString(&l, alloc, "\t", agent_log);
    try l.appendSlice(alloc, "\t<key>StandardErrorPath</key>\n");
    try appendString(&l, alloc, "\t", agent_log);
    if (table_path) |p| {
        try l.appendSlice(alloc, "\t<key>EnvironmentVariables</key>\n\t<dict>\n\t\t<key>MLX_SERVE_KERNEL_TABLE</key>\n");
        try appendString(&l, alloc, "\t\t", p);
        try l.appendSlice(alloc, "\t</dict>\n");
    }
    try l.appendSlice(alloc, "</dict>\n</plist>\n");
    return l.toOwnedSlice(alloc);
}

/// The running binary's absolute real path: `_NSGetExecutablePath`, then `realpath`.
pub fn exePath(alloc: Allocator) ![]u8 {
    var raw: [MAX_PATH]u8 = undefined;
    var n: u32 = raw.len;
    if (std.c._NSGetExecutablePath(&raw, &n) != 0) return error.ExePathUnavailable;
    var real: [MAX_PATH]u8 = undefined;
    const r = realpath(@ptrCast(&raw), &real) orelse return error.ExePathUnavailable;
    return alloc.dupe(u8, std.mem.span(r));
}

/// `text` with every XML comment (`<!-- ... -->`, an unterminated one to the end) removed and every CDATA section
/// replaced by its content escaped as element text, so a later search for markup sees only real markup and a CDATA
/// value still decodes to its text. A comment opener inside CDATA is text, and a CDATA opener inside a comment is
/// comment.
fn xmlMarkupOnly(alloc: Allocator, text: []const u8) ![]u8 {
    var l: std.ArrayList(u8) = .empty;
    errdefer l.deinit(alloc);
    var i: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], "<!--")) {
            const end = std.mem.indexOfPos(u8, text, i + 4, "-->") orelse text.len;
            i = @min(end + 3, text.len);
            continue;
        }
        if (std.mem.startsWith(u8, text[i..], "<![CDATA[")) {
            const from = i + "<![CDATA[".len;
            const end = std.mem.indexOfPos(u8, text, from, "]]>") orelse text.len;
            for (text[from..end]) |c| switch (c) {
                '&' => try l.appendSlice(alloc, "&amp;"),
                '<' => try l.appendSlice(alloc, "&lt;"),
                '>' => try l.appendSlice(alloc, "&gt;"),
                else => try l.append(alloc, c),
            };
            i = @min(end + 3, text.len);
            continue;
        }
        try l.append(alloc, text[i]);
        i += 1;
    }
    return l.toOwnedSlice(alloc);
}

/// The first string of the `<array>` that is the value of the `ProgramArguments` key, XML entities decoded; null when
/// the key's value is not an array, the array is empty, or its first string does not end inside it.
fn plistProgram(alloc: Allocator, raw: []const u8) !?[]u8 {
    // markup is matched only where it is markup: comments are dropped and CDATA becomes escaped text first
    const text = try xmlMarkupOnly(alloc, raw);
    defer alloc.free(text);
    const key = "<key>ProgramArguments</key>";
    const at = std.mem.indexOf(u8, text, key) orelse return null;
    // the value is the next element after the key: only whitespace may come between them
    const rest = std.mem.trimStart(u8, text[at + key.len ..], " \t\r\n");
    if (!std.mem.startsWith(u8, rest, "<array>")) return null;
    const arr_from = text.len - rest.len + "<array>".len;
    const arr_end = std.mem.indexOfPos(u8, text, arr_from, "</array>") orelse return null;
    const open = std.mem.indexOfPos(u8, text[0..arr_end], arr_from, "<string>") orelse return null;
    const from = open + "<string>".len;
    const to = std.mem.indexOfPos(u8, text[0..arr_end], from, "</string>") orelse return null;
    var l: std.ArrayList(u8) = .empty;
    var i = from;
    while (i < to) {
        const ents = [_]struct { e: []const u8, c: u8 }{ .{ .e = "&amp;", .c = '&' }, .{ .e = "&lt;", .c = '<' }, .{ .e = "&gt;", .c = '>' }, .{ .e = "&quot;", .c = '"' }, .{ .e = "&apos;", .c = '\'' }, .{ .e = "&#13;", .c = '\r' } };
        const hit = for (ents) |e| {
            if (std.mem.startsWith(u8, text[i..to], e.e)) break e;
        } else null;
        if (hit) |e| {
            try l.append(alloc, e.c);
            i += e.e.len;
        } else {
            try l.append(alloc, text[i]);
            i += 1;
        }
    }
    return try l.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------------------------------------------
// Files and folders (spec 3.2: the folder 0700, every file 0600)

fn cz(buf: *[MAX_PATH]u8, path: []const u8) ?[*:0]const u8 {
    if (path.len >= buf.len or std.mem.indexOfScalar(u8, path, 0) != null) return null;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return @ptrCast(buf);
}

fn joinZ(buf: *[MAX_PATH]u8, dir: []const u8, name: []const u8) ?[*:0]const u8 {
    const p = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/{s}", .{ dir, name }) catch return null;
    buf[p.len] = 0;
    return @ptrCast(buf);
}

fn exists(path: [*:0]const u8) bool {
    return std.c.access(path, 0) == 0;
}

fn isDir(path: [*:0]const u8) bool {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return false;
    return @as(u32, @intCast(st.mode)) & S_IFMT == S_IFDIR;
}

/// Creates every missing folder of `path` (the parents 0755, the last one 0700); an existing folder keeps its mode.
fn ensureFolder(path: []const u8) !void {
    var buf: [MAX_PATH]u8 = undefined;
    if (path.len == 0) return error.CannotCreateFolder;
    _ = cz(&buf, path) orelse return error.NameTooLong;
    var i: usize = 1;
    while (i < path.len) : (i += 1) {
        if (path[i] != '/') continue;
        buf[i] = 0;
        _ = std.c.mkdir(@ptrCast(&buf), @as(std.c.mode_t, 0o755));
        buf[i] = '/';
    }
    const z: [*:0]const u8 = @ptrCast(&buf);
    const rc = std.c.mkdir(z, @as(std.c.mode_t, 0o700));
    if (rc != 0 and std.c.errno(rc) != .EXIST) return error.CannotCreateFolder;
    if (!isDir(z)) return error.NotAFolder;
}

/// The errno of the last libc call this file reported as an error, for the message that names it.
pub var last_errno: std.c.E = .SUCCESS;

fn failWith(rc: anytype, err: anyerror) anyerror {
    last_errno = std.c.errno(rc);
    return err;
}

/// What the tuner does with a folder or file it finds in the tune folder: use it, tighten its mode (ours, but with bits
/// beyond the private mode, such as group or world write), or refuse it (another user owns it, or a regular file has a
/// second hard link, so writing it would write another file too).
pub const PrivateVerdict = enum { ok, tighten, refuse };

/// Whether a file carries a macOS extended ACL: `none` only when the lookup answered so (ENOENT, or an ACL with no
/// entry); `unknown` when the lookup failed for any other reason, which is never read as "none".
pub const AclState = enum { none, present, unknown };

/// The `fstat` fields the verdict reads; `mode` includes the file type bits. `has_acl`: whether the file carries a macOS
/// extended ACL with at least one entry, which can grant access the mode bits do not show.
pub const PrivStat = struct { uid: u32, mode: u32, nlink: u32, has_acl: AclState = .none };

pub fn privateVerdict(euid: u32, st: PrivStat, private_mode: u32) PrivateVerdict {
    if (st.uid != euid) return .refuse;
    // an ACL that is there, or one the lookup could not rule out, is a refusal
    if (st.has_acl != .none) return .refuse;
    if (st.mode & S_IFMT == S_IFREG and st.nlink != 1) return .refuse;
    if (st.mode & 0o777 & ~private_mode != 0) return .tighten;
    return .ok;
}

extern "c" fn acl_get_fd_np(fd: c_int, acl_type: c_int) ?*anyopaque;
extern "c" fn acl_get_entry(acl: *anyopaque, entry_id: c_int, entry: *?*anyopaque) c_int;
extern "c" fn acl_free(obj: *anyopaque) c_int;
const ACL_TYPE_EXTENDED: c_int = 0x00000100;
const ACL_FIRST_ENTRY: c_int = 0;

/// Test seam: replaces `acl_get_fd_np` (it sets errno and returns null to inject a failed or an empty lookup).
pub var acl_get_override: ?*const fn (c_int, c_int) ?*anyopaque = null;

/// The errno of the last ACL lookup that answered `unknown`.
var acl_errno: std.c.E = .SUCCESS;

/// Whether the opened file carries a macOS extended ACL with at least one entry. `acl_get_fd_np` returning null with
/// errno ENOENT is the only "no ACL"; null with any other errno is `unknown` (errno kept in `acl_errno`).
fn fdAcl(fd: c_int) AclState {
    std.c._errno().* = 0;
    const got = if (acl_get_override) |f| f(fd, ACL_TYPE_EXTENDED) else acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
    const acl = got orelse {
        const e = std.c.errno(@as(c_int, -1));
        if (e == .NOENT) return .none;
        acl_errno = if (e == .SUCCESS) .IO else e;
        return .unknown;
    };
    defer _ = acl_free(acl);
    var entry: ?*anyopaque = null;
    std.c._errno().* = 0;
    if (acl_get_entry(acl, ACL_FIRST_ENTRY, &entry) == 0) return .present;
    // an ACL object with no entry answers EINVAL for the first one; any other failure leaves the question open
    const e = std.c.errno(@as(c_int, -1));
    if (e == .INVAL) return .none;
    acl_errno = if (e == .SUCCESS) .IO else e;
    return .unknown;
}

fn privStat(st: std.c.Stat, fd: c_int) PrivStat {
    return .{ .uid = @intCast(st.uid), .mode = @intCast(st.mode), .nlink = @intCast(st.nlink), .has_acl = fdAcl(fd) };
}

/// Checks an opened folder or file of the tune folder: the right type, owned by the effective uid, one link for a file,
/// no ACL (and an ACL lookup that answered), and whether a mode bit goes beyond `private_mode`. `.ok` or `.tighten`;
/// every refusal is an error (an unanswered ACL lookup is `error.AclUnknown`, its errno in `last_errno`).
fn checkPrivateFd(fd: c_int, private_mode: u32, want_dir: bool) !PrivateVerdict {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return failWith(@as(c_int, -1), error.StatFailed);
    const kind = @as(u32, @intCast(st.mode)) & S_IFMT;
    if (want_dir and kind != S_IFDIR) return error.NotAFolder;
    if (!want_dir and kind != S_IFREG) return error.NotARegularFile;
    const ps = privStat(st, fd);
    const v = privateVerdict(@intCast(std.c.geteuid()), ps, private_mode);
    if (v == .refuse) {
        if (st.uid != std.c.geteuid()) return if (want_dir) error.FolderNotOurs else error.FileNotOurs;
        if (ps.has_acl == .unknown) {
            last_errno = acl_errno;
            return error.AclUnknown;
        }
        if (ps.has_acl == .present) return if (want_dir) error.FolderHasAcl else error.FileHasAcl;
        return error.FileHardLinked;
    }
    return v;
}

/// `checkPrivateFd`, then a mode of ours that is wider than `private_mode` is tightened. Returns true when it tightened
/// the mode: the file was writable by others until now. Not for the two lock files: see `replaceWideLock`.
fn securePrivateFd(fd: c_int, private_mode: u32, want_dir: bool) !bool {
    if (try checkPrivateFd(fd, private_mode, want_dir) == .ok) return false;
    if (std.c.fchmod(fd, @as(std.c.mode_t, @intCast(private_mode))) != 0) return failWith(@as(c_int, -1), error.ChmodFailed);
    return true;
}

/// A lock file found wider than private is never reused: a descriptor opened while others could write it keeps that
/// access after any chmod, so the record it reaches would not be trusted. When no one holds the lock, its name is
/// unlinked and a fresh empty 0600 file (a new inode) is created in its place: `true`. When another open file
/// description holds it, nothing is changed (its mode stays wide, so `--stop` never trusts it): `false`.
/// `fd` is the wide lock, opened read-write; the real `flock` always (the seams stand in for the tune's own take).
fn replaceWideLock(fd: c_int, z: [*:0]const u8) !bool {
    const rc = flock(fd, LOCK_EX | LOCK_NB);
    if (rc != 0) {
        const e = std.c.errno(rc);
        if (e == .AGAIN) return false;
        last_errno = e;
        return error.LockFailed;
    }
    defer _ = flock(fd, LOCK_UN);
    const urc = std.c.unlink(z);
    if (urc != 0 and std.c.errno(urc) != .NOENT) return failWith(urc, error.LockUnavailable);
    const nfd = std.c.open(z, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .NONBLOCK = true }, @as(std.c.mode_t, 0o600));
    if (nfd < 0) {
        // another process made a fresh one first: it is checked like any other when it is opened
        if (std.c.errno(nfd) == .EXIST) return true;
        return failWith(nfd, error.LockUnavailable);
    }
    _ = std.c.close(nfd);
    return true;
}

/// Read-only form for `--stop`: null when the opened lock is a private regular file of this user with one link, else
/// why its contents are not trusted. Nothing is tightened: a lock others could write may hold any PID.
fn lockUntrusted(fd: c_int) ?[]const u8 {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return "it cannot be examined";
    if (@as(u32, @intCast(st.mode)) & S_IFMT != S_IFREG) return "it is not a regular file";
    if (st.uid != std.c.geteuid()) return "another user owns it";
    if (st.nlink != 1) return "it has another hard link";
    if (@as(u32, @intCast(st.mode)) & 0o022 != 0) return "group or others can write it";
    return switch (fdAcl(fd)) {
        .none => null,
        .present => "it has an access control list",
        .unknown => "its access control list cannot be read",
    };
}

/// The table is not here: its own refusal rule (`unsafe_permissions`) covers it, and the tuner never loosens or
/// tightens a table the engine refused.
const TUNE_FILES = [_][]const u8{ LOCK_NAME, LOG_NAME, PROGRESS_NAME, AGENT_LOG_NAME, INSTALL_LOCK_NAME };

/// The two lock files: one found wider than private is replaced by a new file, never tightened in place.
fn isLockName(name: []const u8) bool {
    return std.mem.eql(u8, name, LOCK_NAME) or std.mem.eql(u8, name, INSTALL_LOCK_NAME);
}

/// Creates the tune folder when missing, then makes sure it and every tuner file already in it are this user's and
/// private (folder 0700, files 0600), tightening a wider mode of our own and refusing anything another user owns.
fn ensureTuneFolder(root: []const u8) !void {
    last_errno = .SUCCESS;
    ensureFolder(root) catch |e| return failWith(@as(c_int, -1), e);
    var buf: [MAX_PATH]u8 = undefined;
    const z = cz(&buf, root) orelse return error.NameTooLong;
    const dfd = std.c.open(z, .{ .ACCMODE = .RDONLY, .NONBLOCK = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (dfd < 0) return failWith(dfd, error.FolderUnreadable);
    defer _ = std.c.close(dfd);
    _ = try securePrivateFd(dfd, 0o700, true);
    for (TUNE_FILES) |name| {
        var nb: [MAX_PATH]u8 = undefined;
        const nz = cz(&nb, name) orelse return error.NameTooLong;
        const lock_file = isLockName(name);
        const fd = std.c.openat(dfd, nz, .{ .ACCMODE = if (lock_file) .RDWR else .RDONLY, .NONBLOCK = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
        if (fd < 0) {
            if (std.c.errno(fd) == .NOENT) continue;
            return failWith(fd, error.FileUnreadable);
        }
        defer _ = std.c.close(fd);
        if (!lock_file) {
            _ = try securePrivateFd(fd, 0o600, false);
            continue;
        }
        // a lock that others could write is not reused: a descriptor they opened then still writes it after a chmod,
        // so a fresh file takes its name (or, while someone holds it, it is left wide and never trusted)
        if (try checkPrivateFd(fd, 0o600, false) == .tighten) {
            var pb: [MAX_PATH]u8 = undefined;
            const pz = joinZ(&pb, root, name) orelse return error.NameTooLong;
            _ = try replaceWideLock(fd, pz);
        }
    }
}

/// Creates `<dir>/<name>` with mode 0600 when missing; an existing one must be this user's and is made private.
fn touchPrivate(dir: []const u8, name: []const u8) !void {
    var buf: [MAX_PATH]u8 = undefined;
    const z = joinZ(&buf, dir, name) orelse return error.NameTooLong;
    const fd = std.c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .NOFOLLOW = true, .NONBLOCK = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return failWith(fd, error.CannotCreateFile);
    defer _ = std.c.close(fd);
    _ = try securePrivateFd(fd, 0o600, false);
}

const c_stat = @extern(*const fn ([*:0]const u8, *std.c.Stat) callconv(.c) c_int, .{ .name = "stat" });
/// The directory entry itself: a symlink is not followed.
const c_lstat = @extern(*const fn ([*:0]const u8, *std.c.Stat) callconv(.c) c_int, .{ .name = "lstat" });

fn statOf(path: [*:0]const u8) ?std.c.Stat {
    var st: std.c.Stat = undefined;
    return if (c_stat(path, &st) == 0) st else null;
}

/// `path` made absolute (against the real working folder) and normalized by its text alone: empty and `.` parts are
/// dropped and `..` takes back the part before it. null when it does not fit.
fn lexicalPath(out: *[MAX_PATH]u8, path: []const u8) ?[]const u8 {
    var joined: [MAX_PATH]u8 = undefined;
    var src = path;
    if (path.len == 0 or path[0] != '/') {
        var cb: [MAX_PATH]u8 = undefined;
        const c = realpath(".", &cb) orelse return null;
        src = std.fmt.bufPrint(&joined, "{s}/{s}", .{ std.mem.span(c), path }) catch return null;
    }
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, src, '/');
    while (it.next()) |part| {
        if (std.mem.eql(u8, part, ".")) continue;
        if (std.mem.eql(u8, part, "..")) {
            while (n > 0 and out[n - 1] != '/') n -= 1;
            if (n > 0) n -= 1;
            continue;
        }
        if (n + 1 + part.len > out.len) return null;
        out[n] = '/';
        @memcpy(out[n + 1 ..][0..part.len], part);
        n += 1 + part.len;
    }
    if (n == 0) {
        out[0] = '/';
        n = 1;
    }
    return out[0..n];
}

/// `<real path of the parent folder>/<last part>`: the name the kernel opens for `path`, with every link and `.` or `..`
/// in the folders resolved. null when the last part is not a name or the parent folder does not resolve.
fn parentResolved(out: *[MAX_PATH]u8, path: []const u8) ?[]const u8 {
    const base = std.fs.path.basename(path);
    if (base.len == 0 or std.mem.eql(u8, base, ".") or std.mem.eql(u8, base, "..")) return null;
    const parent = std.fs.path.dirname(path) orelse (if (path.len > 0 and path[0] == '/') "/" else ".");
    var pb: [MAX_PATH]u8 = undefined;
    const pz = cz(&pb, parent) orelse return null;
    var rb: [MAX_PATH]u8 = undefined;
    const r = std.mem.span(realpath(pz, &rb) orelse return null);
    const sep: []const u8 = if (r.len > 0 and r[r.len - 1] == '/') "" else "/";
    return std.fmt.bufPrint(out, "{s}{s}{s}", .{ r, sep, base }) catch null;
}

/// What the configured table path really names when that is the tune folder or one of the tuner's own files; null for
/// any other path. Identity is decided on canonical paths, so a tuner file that does not exist yet is still found
/// under any spelling (`./`, `//`, `sub/..`, a linked folder): the table path's parent resolved as the kernel resolves
/// it, and its text-normalized form with that parent resolved, are each compared with the tune folder's real path and
/// its text-normalized one (plus each reserved name); a file that exists is also matched by device and inode.
fn tableCollision(path: []const u8, root: []const u8) ?[]const u8 {
    return reservedCollision(path, .{ .dir = root, .dir_word = "tune folder", .names = &TUNE_FILES });
}

/// A folder whose own path and some of whose names a table path must never name.
const Reserved = struct {
    dir: []const u8,
    /// What the folder itself is called in the refusal.
    dir_word: []const u8,
    names: []const []const u8,
    /// Names that start with one of these (and go on) are reserved too: the install's staged and set-aside plists.
    prefixes: []const []const u8 = &.{},
};

/// The reserved name `base` is, compared ignoring ASCII case: the default macOS volume is case-insensitive, so
/// `TUNE.LOG` opens `tune.log` (or creates the file that later is `tune.log`).
fn reservedName(base: []const u8, r: Reserved) ?[]const u8 {
    if (base.len == 0 or std.mem.indexOfScalar(u8, base, '/') != null) return null;
    for (r.names) |n| if (std.ascii.eqlIgnoreCase(base, n)) return n;
    for (r.prefixes) |p| if (base.len > p.len and std.ascii.eqlIgnoreCase(base[0..p.len], p)) return p;
    return null;
}

/// `tableCollision` for any reserved folder: canonical spellings compared with the folder's, the last part compared
/// ignoring ASCII case, the parent matched by device and inode when it exists, and an existing file by its own.
fn reservedCollision(path: []const u8, r: Reserved) ?[]const u8 {
    const root = r.dir;
    var pb: [MAX_PATH]u8 = undefined;
    const pz = cz(&pb, path) orelse return null;
    const target = statOf(pz);
    var rb: [MAX_PATH]u8 = undefined;
    const root_st: ?std.c.Stat = if (cz(&rb, root)) |rz| statOf(rz) else null;
    if (root_st) |rs| if (target) |t| if (t.dev == rs.dev and t.ino == rs.ino) return r.dir_word;
    for (r.names) |name| {
        var nb: [MAX_PATH]u8 = undefined;
        const nz = joinZ(&nb, root, name) orelse continue;
        if (target) |t| if (statOf(nz)) |s| if (s.dev == t.dev and s.ino == t.ino) return name;
    }
    // the parent folder is the reserved folder (any spelling, any case): the last part decides
    if (root_st) |rs| {
        const base = std.fs.path.basename(path);
        const parent = std.fs.path.dirname(path) orelse (if (path.len > 0 and path[0] == '/') "/" else ".");
        var qb: [MAX_PATH]u8 = undefined;
        if (cz(&qb, parent)) |qz| {
            if (statOf(qz)) |ps| {
                if (ps.dev == rs.dev and ps.ino == rs.ino) {
                    if (reservedName(base, r)) |n| return n;
                }
            }
        }
    }
    // the table path's canonical spellings
    var kb: [MAX_PATH]u8 = undefined;
    var lb: [MAX_PATH]u8 = undefined;
    var lrb: [MAX_PATH]u8 = undefined;
    const lex = lexicalPath(&lb, path);
    const forms = [3]?[]const u8{ parentResolved(&kb, path), if (lex) |l| parentResolved(&lrb, l) else null, lex };
    // the tune folder's: its real path, and its text-normalized one when that differs
    var rrb: [MAX_PATH]u8 = undefined;
    var rlb: [MAX_PATH]u8 = undefined;
    var rzb: [MAX_PATH]u8 = undefined;
    const real_root: ?[]const u8 = if (cz(&rzb, root)) |rz| (if (realpath(rz, &rrb)) |rp| std.mem.span(rp) else null) else null;
    const roots = [2]?[]const u8{ real_root, lexicalPath(&rlb, root) };
    for (forms) |fo| {
        const f = fo orelse continue;
        for (roots) |ro| {
            const rt = ro orelse continue;
            if (std.ascii.eqlIgnoreCase(f, rt)) return r.dir_word;
            if (f.len <= rt.len + 1 or !std.ascii.eqlIgnoreCase(f[0..rt.len], rt)) continue;
            const r_sep = rt.len > 0 and rt[rt.len - 1] == '/';
            const at = if (r_sep) rt.len else rt.len + 1;
            if (!r_sep and f[rt.len] != '/') continue;
            if (reservedName(f[at..], r)) |n| return n;
        }
    }
    return null;
}

/// The LaunchAgent plist and the names an install gives it while it works: a table written there would replace the
/// agent's plist, and the agent's own removal would then delete the table.
const PLIST_NAMES = [_][]const u8{PLIST_NAME};
const PLIST_PREFIXES = [_][]const u8{ PLIST_NAME ++ ".staged-", PLIST_NAME ++ ".previous-" };

fn plistCollision(path: []const u8, agents_dir: []const u8) ?[]const u8 {
    return reservedCollision(path, .{ .dir = agents_dir, .dir_word = "LaunchAgents folder", .names = &PLIST_NAMES, .prefixes = &PLIST_PREFIXES });
}

/// Exit 2 when the table path names a tuner file, the tune folder, or the LaunchAgent plist (or its install names):
/// no command deletes or replaces those as a table.
fn refuseCollision(env: Env, path: []const u8, root: []const u8) ?u8 {
    const what = tableCollision(path, root) orelse (if (env.launch_agents_dir) |d| plistCollision(path, d) else null) orelse return null;
    say(env, "lesto tune: the table path {s} names the tuner's own {s}; it is not deleted or written as a table (set MLX_SERVE_KERNEL_TABLE to a table file)", .{ path, what });
    return 2;
}

/// Unlinks `path`: true when a file was removed, false when there was none (ENOENT only); any other failure is an error
/// with its errno in `last_errno`.
fn removeFile(path: []const u8) !bool {
    var buf: [MAX_PATH]u8 = undefined;
    const z = cz(&buf, path) orelse return error.NameTooLong;
    const rc = std.c.unlink(z);
    if (rc == 0) return true;
    if (std.c.errno(rc) == .NOENT) return false;
    return failWith(rc, error.CannotRemove);
}

// ---------------------------------------------------------------------------------------------------------------
// The lock (spec 7.4)

const ProcBsdInfo = extern struct {
    pbi_flags: u32,
    pbi_status: u32,
    pbi_xstatus: u32,
    pbi_pid: u32,
    pbi_ppid: u32,
    pbi_uid: u32,
    pbi_gid: u32,
    pbi_ruid: u32,
    pbi_rgid: u32,
    pbi_svuid: u32,
    pbi_svgid: u32,
    rfu_1: u32,
    pbi_comm: [16]u8,
    pbi_name: [32]u8,
    pbi_nfiles: u32,
    pbi_pgid: u32,
    pbi_pjobc: u32,
    e_tdev: u32,
    e_tpgid: u32,
    pbi_nice: i32,
    pbi_start_tvsec: u64,
    pbi_start_tvusec: u64,
};

comptime {
    std.debug.assert(@sizeOf(ProcBsdInfo) == 136);
}

pub const StartTime = struct { sec: u64, usec: u64 };

/// A live process's start time from `proc_pidinfo(PROC_PIDTBSDINFO)`; null when there is no such process.
pub fn procStart(pid: c_int) ?StartTime {
    var info: ProcBsdInfo = undefined;
    const n = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, @sizeOf(ProcBsdInfo));
    if (n != @sizeOf(ProcBsdInfo)) return null;
    return .{ .sec = info.pbi_start_tvsec, .usec = info.pbi_start_tvusec };
}

pub const Lock = struct {
    fd: c_int,

    pub fn release(self: Lock) void {
        _ = flock(self.fd, LOCK_UN);
        _ = std.c.close(self.fd);
    }
};

pub const LockResult = union(enum) { held: Lock, busy: struct { pid: i32 } };

/// Test seam: replaces `flock` in `takeLock` and `stopRunning` (it sets errno and returns -1 to inject a failure).
pub var flock_override: ?*const fn (c_int, c_int) c_int = null;

fn doFlock(fd: c_int, op: c_int) c_int {
    if (flock_override) |f| return f(fd, op);
    return flock(fd, op);
}

/// Test seam: replaces the start-time lookup `takeLock` records (it sets errno and returns null to inject a failure).
pub var pidinfo_override: ?*const fn (c_int) ?StartTime = null;

/// This process's start time for the lock record; null (errno set by the lookup, or left 0) when it cannot be read.
fn startForLock(pid: c_int) ?StartTime {
    if (pidinfo_override) |f| return f(pid);
    std.c._errno().* = 0;
    return procStart(pid);
}

const LockText = struct { pid: i32, start: ?StartTime };

fn readLockText(fd: c_int) LockText {
    var b: [96]u8 = undefined;
    const n = std.c.pread(fd, &b, b.len, 0);
    if (n <= 0) return .{ .pid = 0, .start = null };
    var it = std.mem.tokenizeAny(u8, b[0..@intCast(n)], " \n");
    const pid = std.fmt.parseInt(i32, it.next() orelse "", 10) catch 0;
    const sec = std.fmt.parseInt(u64, it.next() orelse "", 10) catch return .{ .pid = pid, .start = null };
    const usec = std.fmt.parseInt(u64, it.next() orelse "", 10) catch return .{ .pid = pid, .start = null };
    return .{ .pid = pid, .start = .{ .sec = sec, .usec = usec } };
}

/// Opens a lock file read-write (created 0600, close-on-exec) after the private checks. One found wider than private is
/// replaced by a fresh file (`replaceWideLock`) and the new one is opened; null when a wide one is held by another open
/// file description, so it is left as it is. Every other refusal is an error.
fn openPrivateLock(z: [*:0]const u8) !?c_int {
    var tries: u32 = 0;
    while (tries < 3) : (tries += 1) {
        const fd = std.c.open(z, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true, .NONBLOCK = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return failWith(fd, error.LockUnavailable);
        _ = std.c.fcntl(fd, std.c.F.SETFD, @as(c_int, 1)); // FD_CLOEXEC: a launchctl child never holds the lock
        const v = checkPrivateFd(fd, 0o600, false) catch |e| {
            _ = std.c.close(fd);
            return e;
        };
        if (v == .ok) return fd;
        const replaced = replaceWideLock(fd, z) catch |e| {
            _ = std.c.close(fd);
            return e;
        };
        _ = std.c.close(fd);
        if (!replaced) return null;
    }
    last_errno = .AGAIN;
    return error.LockUnavailable;
}

/// Opens `<tune root>/lock` (created 0600), takes `flock(LOCK_EX | LOCK_NB)` and writes `<pid> <start_sec> <start_usec>`.
/// Busy when another open file description holds it, with the PID it wrote. Only libc calls: a forked child may call it.
pub fn takeLock(io: std.Io, env: Env) !LockResult {
    _ = io;
    last_errno = .SUCCESS;
    const dir = env.engine.tune_root orelse return error.NoTuneFolder;
    var buf: [MAX_PATH]u8 = undefined;
    const z = joinZ(&buf, dir, LOCK_NAME) orelse return error.NameTooLong;
    // a wide lock someone holds: busy, and its record is not read (pid 0)
    const fd = try openPrivateLock(z) orelse return .{ .busy = .{ .pid = 0 } };
    const rc = doFlock(fd, LOCK_EX | LOCK_NB);
    if (rc != 0) {
        const e = std.c.errno(rc);
        if (e != .AGAIN) {
            // not contention (EWOULDBLOCK is EAGAIN here): the lock cannot work on this folder, which is an error
            last_errno = e;
            _ = std.c.close(fd);
            return error.LockFailed;
        }
        const t = readLockText(fd);
        _ = std.c.close(fd);
        return .{ .busy = .{ .pid = t.pid } };
    }
    errdefer {
        _ = doFlock(fd, LOCK_UN);
        _ = std.c.close(fd);
    }
    // the file was checked (one link, ours, private) before this first write through it
    const trc = std.c.ftruncate(fd, 0);
    if (trc != 0) return failWith(trc, error.LockUnavailable);
    const me = std.c.getpid();
    // the record names this process by its real start time or not at all: a lookup that fails twice fails the take
    // (the errdefer above frees the lock), because `--stop` could never match a made-up time to the running tune
    const st = startForLock(me) orelse startForLock(me) orelse {
        const e = std.c.errno(@as(c_int, -1));
        last_errno = if (e == .SUCCESS) .SRCH else e;
        return error.LockFailed;
    };
    var tb: [96]u8 = undefined;
    const text = std.fmt.bufPrint(&tb, "{d} {d} {d}\n", .{ me, st.sec, st.usec }) catch unreachable;
    const n = doLockWrite(fd, text);
    if (n < 0) return failWith(n, error.LockUnavailable);
    if (n != @as(isize, @intCast(text.len))) {
        // a short write sets no errno: it is the disk running out of room
        last_errno = .NOSPC;
        return error.LockUnavailable;
    }
    return .{ .held = .{ .fd = fd } };
}

/// Test seam: replaces the `kill` that `stopRunning` sends to an identified holder (it sets errno and returns -1).
pub var kill_override: ?*const fn (c_int, c_int) c_int = null;

/// Test seam: replaces the `pwrite` of the lock record (it sets errno and returns -1, or returns a short count).
pub var lock_write_override: ?*const fn (c_int, []const u8) isize = null;

fn doLockWrite(fd: c_int, text: []const u8) isize {
    if (lock_write_override) |f| return f(fd, text);
    return std.c.pwrite(fd, text.ptr, text.len, 0);
}

pub const StopResult = union(enum) {
    /// The lock was free (or absent): no tune runs; nothing was signalled.
    not_running,
    /// SIGTERM sent to this PID, whose start time matched the lock's.
    signalled: i32,
    /// The lock is held, but its record is missing or unparseable (pid 0) or its PID's start time does not match:
    /// nothing was signalled.
    unidentified: i32,
    /// The lock is held, but the file is not this user's private file, so its record is not acted on.
    untrusted: []const u8,
    /// The holder was identified, but SIGTERM could not be sent.
    signal_failed: struct { pid: i32, err: std.c.E },
    /// The lock file exists but could not be opened or locked for a reason other than its absence or contention.
    failed: std.c.E,

    /// True only when the postcondition of `--stop` holds: no tune runs, or the identified tune was signalled.
    pub fn ok(self: StopResult) bool {
        return self == .not_running or self == .signalled;
    }
};

/// Spec 7.4: sends SIGTERM only to a holder whose live start time equals the one it wrote. Only a missing lock file
/// (ENOENT) or a lock this call can take means no tune runs.
pub fn stopRunning(io: std.Io, env: Env) StopResult {
    _ = io;
    // no tune folder known (HOME unset) proves nothing about a tune started under the normal environment
    const dir = env.engine.tune_root orelse return .{ .failed = .NOENT };
    var buf: [MAX_PATH]u8 = undefined;
    const z = joinZ(&buf, dir, LOCK_NAME) orelse return .{ .failed = .NAMETOOLONG };
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (fd < 0) {
        const e = std.c.errno(fd);
        return if (e == .NOENT) .not_running else .{ .failed = e };
    }
    defer _ = std.c.close(fd);
    const rc = doFlock(fd, LOCK_EX | LOCK_NB);
    if (rc == 0) {
        _ = doFlock(fd, LOCK_UN);
        return .not_running;
    }
    const fe = std.c.errno(rc);
    if (fe != .AGAIN) return .{ .failed = fe };
    // the record is acted on only when nobody but this user could have written it
    if (lockUntrusted(fd)) |why| return .{ .untrusted = why };
    const t = readLockText(fd);
    const want = t.start orelse return .{ .unidentified = t.pid };
    if (t.pid <= 0) return .{ .unidentified = t.pid };
    const live = procStart(t.pid) orelse return .{ .unidentified = t.pid };
    if (live.sec != want.sec or live.usec != want.usec) return .{ .unidentified = t.pid };
    const krc = if (kill_override) |kf| kf(t.pid, SIGTERM) else kill(t.pid, SIGTERM);
    if (krc != 0) return .{ .signal_failed = .{ .pid = t.pid, .err = std.c.errno(krc) } };
    return .{ .signalled = t.pid };
}

// ---------------------------------------------------------------------------------------------------------------
// The stop flag (SIGINT, SIGTERM)

var stop_flag: std.atomic.Value(bool) = .init(false);

fn onStopSignal(_: std.posix.SIG) callconv(.c) void {
    stop_flag.store(true, .release);
}

const SavedHandlers = struct { int: std.posix.Sigaction, term: std.posix.Sigaction };

/// Installed only by the actions that run work; the previous handlers come back when the action ends.
fn installStopHandlers() SavedHandlers {
    var saved: SavedHandlers = undefined;
    const act = std.posix.Sigaction{ .handler = .{ .handler = onStopSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(std.posix.SIG.INT, &act, &saved.int);
    std.posix.sigaction(std.posix.SIG.TERM, &act, &saved.term);
    return saved;
}

fn restoreStopHandlers(s: SavedHandlers) void {
    std.posix.sigaction(std.posix.SIG.INT, &s.int, null);
    std.posix.sigaction(std.posix.SIG.TERM, &s.term, null);
}

// ---------------------------------------------------------------------------------------------------------------
// The progress line (spec 7.2)

/// With a terminal, one line rewritten in place with `\r`; without, one line per change, each ending in `\n`.
pub const ProgressLine = struct {
    out: *std.Io.Writer,
    tty: bool,
    inline_open: bool = false,
    last: [512]u8 = undefined,
    last_len: usize = 0,

    pub fn update(self: *ProgressLine, text: []const u8) void {
        if (self.tty) {
            self.out.writeAll("\r") catch outFail();
            self.out.writeAll(text) catch outFail();
            self.out.writeAll("\x1b[K") catch outFail();
            self.inline_open = true;
        } else {
            if (text.len <= self.last.len and std.mem.eql(u8, self.last[0..self.last_len], text)) return;
            self.out.writeAll(text) catch outFail();
            self.out.writeAll("\n") catch outFail();
            const n = @min(text.len, self.last.len);
            @memcpy(self.last[0..n], text[0..n]);
            self.last_len = n;
        }
        self.out.flush() catch outFail();
    }

    /// Ends an in-place line, so what follows starts on its own line.
    pub fn end(self: *ProgressLine) void {
        if (!self.inline_open) return;
        self.out.writeAll("\n") catch outFail();
        self.out.flush() catch outFail();
        self.inline_open = false;
    }

    fn onLine(ctx: ?*anyopaque, text: []const u8) void {
        const self: *ProgressLine = @ptrCast(@alignCast(ctx.?));
        self.update(text);
    }

    pub fn reporter(self: *ProgressLine) tune_sched.Reporter {
        return .{ .ctx = self, .onLine = onLine };
    }
};

// ---------------------------------------------------------------------------------------------------------------
// The table a run writes

pub const InfoExtras = struct {
    /// The run's signals were scripted (`MLX_SERVE_TUNE_FAKE_SIGNALS`).
    scripted: bool = false,
    /// UTC ISO 8601; null takes the current time.
    tuned_at: ?[]const u8 = null,
};

fn isoNow(alloc: Allocator) ![]u8 {
    const now = time(null);
    const es = std.time.epoch.EpochSeconds{ .secs = @intCast(@max(now, 0)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.allocPrint(alloc, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year, md.month.numeric(), @as(u32, md.day_index) + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    });
}

fn dupeKey(alloc: Allocator, k: kernel_table.Key) !kernel_table.Key {
    return .{
        .gpu_arch = try alloc.dupe(u8, k.gpu_arch),
        .gpu_cores = k.gpu_cores,
        .hw_model = try alloc.dupe(u8, k.hw_model),
        .os_build = try alloc.dupe(u8, k.os_build),
        .kernel_hash = try alloc.dupe(u8, k.kernel_hash),
        .mlx_version = try alloc.dupe(u8, k.mlx_version),
    };
}

/// Maps a finished run to the table (spec 3.3) and sets its checksum. Everything the table references is either static
/// or allocated here with `alloc` (an arena): the key, the excluded and not-measurable lists and their strings are
/// copied, so the table outlives the caller's buffers. (`batches_discarded` is a 7-entry array held by value, and its
/// reason strings are the static `DISCARD_REASONS`.)
pub fn buildTable(alloc: Allocator, key: kernel_table.Key, outcome: tune_sched.Outcome, info_extras: InfoExtras) !kernel_table.Table {
    var disc: [7]kernel_table.Discarded = undefined;
    for (kernel_table.DISCARD_REASONS, 0..) |r, i| disc[i] = .{ .reason = r, .n = outcome.discarded[i] };
    const gu = outcome.plan.gateup;
    const dn = outcome.plan.down;
    const own_key = try dupeKey(alloc, key);
    const excluded = try alloc.alloc(kernel_table.Excluded, outcome.identity.excluded.len);
    for (outcome.identity.excluded, excluded) |x, *y| y.* = .{ .kernel = try alloc.dupe(u8, x.kernel), .cand = try alloc.dupe(u8, x.cand), .reason = try alloc.dupe(u8, x.reason) };
    const nm = try alloc.alloc(kernel_table.NotMeasurable, outcome.not_measurable.len);
    for (outcome.not_measurable, nm) |x, *y| y.* = .{ .kernel = try alloc.dupe(u8, x.kernel), .cand = try alloc.dupe(u8, x.cand) };
    var t: kernel_table.Table = .{
        .schema = kernel_table.TABLE_SCHEMA,
        .key = own_key,
        .shape = kernel_table.TUNED_SHAPE,
        .plan = .{
            .gateup = if (gu) |g| .{ .layout = g } else .served,
            .down = if (dn) |d| .{ .layout = d } else .served,
        },
        .evidence = .{
            .gateup = if (gu != null) outcome.evidence[0] else null,
            .down = if (dn != null) outcome.evidence[1] else null,
        },
        .identity = .{
            .plans_checked = outcome.identity.plans_checked,
            .cases = outcome.identity.cases,
            .calls_compared = outcome.identity.calls_compared,
            .os_build = own_key.os_build,
            .all_identical = true,
            .excluded = excluded,
        },
        .info = .{
            .lesto_version = build_options.version,
            .tuned_at = info_extras.tuned_at orelse try isoNow(alloc),
            .signals = if (info_extras.scripted) "scripted" else "real",
            .windows_used = outcome.windows_used,
            .batches_kept = outcome.batches_kept,
            .batches_discarded = disc,
            .not_measurable = nm,
            .served_reasons = .{
                .gateup = if (gu == null) (if (outcome.served_reasons[0]) |r| @tagName(r) else null) else null,
                .down = if (dn == null) (if (outcome.served_reasons[1]) |r| @tagName(r) else null) else null,
            },
        },
        .checksum = "",
    };
    try kernel_table.finalize(alloc, &t);
    return t;
}

// ---------------------------------------------------------------------------------------------------------------
// Entry points

/// Test seam: replaces the scheduler call of the default action.
pub var sched_override: ?*const fn () tune_sched.SchedResult = null;

/// How often `--agent` retries a busy lock (a test sets a few milliseconds).
pub var agent_lock_poll_ns: u64 = 30 * std.time.ns_per_s;

/// Sleeps `ns` in short slices; false as soon as SIGINT or SIGTERM set the stop flag.
fn waitUnlessStopped(ns: u64) bool {
    const slice_ns: u64 = 50 * std.time.ns_per_ms;
    var left = ns;
    while (true) {
        if (stop_flag.load(.acquire)) return false;
        if (left == 0) return true;
        const step = @min(left, slice_ns);
        _ = usleep(@intCast(step / std.time.ns_per_us));
        left -= step;
    }
}

pub fn run(alloc: Allocator, io: std.Io, args: []const []const u8) u8 {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const env = envFromProcess(arena.allocator(), io) catch {
        var b: [256]u8 = undefined;
        var fw = std.Io.File.stderr().writerStreaming(io, &b);
        fw.interface.writeAll("lesto tune: out of memory\n") catch {};
        fw.interface.flush() catch {};
        return 2;
    };
    return runWith(alloc, io, env, args);
}

pub fn runWith(alloc: Allocator, io: std.Io, env: Env, args: []const []const u8) u8 {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    out_failed = false;
    const code: u8 = switch (parseFlags(args)) {
        .usage => |why| blk: {
            say(env, "lesto tune: {s}", .{why});
            env.out.writeAll(USAGE) catch outFail();
            break :blk 2;
        },
        .check_model => blk: {
            say(env, CHECK_MODEL, .{});
            break :blk 2;
        },
        .print_plist => |bin| blk: {
            const root = env.engine.tune_root orelse {
                say(env, NO_HOME, .{});
                break :blk 2;
            };
            const text = agentPlist(a, env, root, bin, DEFAULT_IDLE_MINUTES) catch |e| break :blk plistFailed(env, e);
            env.out.writeAll(text) catch outFail();
            break :blk 0;
        },
        .ok => |f| switch (f.action) {
            .default, .agent => tuneAction(a, io, env, f),
            .background => backgroundAction(a, io, env, f),
            .stop => stopAction(a, io, env),
            .show => showAction(a, io, env),
            .verify => verifyAction(a, io, env, f),
            .reset => resetAction(a, io, env),
        },
    };
    env.out.flush() catch outFail();
    if (out_failed) {
        // the output did not reach its destination: say so where it still can go, and fail a command that had succeeded
        // (a command that had already failed keeps its own code)
        var b: [256]u8 = undefined;
        var fw = std.Io.File.stderr().writerStreaming(io, &b);
        fw.interface.writeAll("lesto tune: writing the output failed; it is incomplete\n") catch {};
        fw.interface.flush() catch {};
        if (code == 0) return 2;
    }
    return code;
}

fn oom(env: Env) u8 {
    say(env, "lesto tune: out of memory", .{});
    return 2;
}

fn noHome(env: Env) u8 {
    say(env, NO_HOME, .{});
    return 2;
}

fn folderFailed(env: Env, root: []const u8, e: anyerror) u8 {
    say(env, "lesto tune: the tune folder {s} is not usable ({s}, errno {s}): it must be a folder of this user, created if missing", .{ root, @errorName(e), @tagName(last_errno) });
    return 2;
}

fn lockFailed(env: Env, e: anyerror) u8 {
    say(env, "lesto tune: the lock could not be taken ({s}, errno {s})", .{ @errorName(e), @tagName(last_errno) });
    return 2;
}

fn plistFailed(env: Env, e: anyerror) u8 {
    if (e == error.NotXmlText) {
        say(env, "lesto tune: a path for the LaunchAgent holds a byte or character a plist cannot carry (a control character, invalid UTF-8, U+FFFE or U+FFFF); use a plain path", .{});
    } else if (e == error.TablePathUnresolved or e == error.TablePathNotAFile) {
        say(env, "lesto tune: the MLX_SERVE_KERNEL_TABLE path cannot be made absolute for the LaunchAgent ({s}, errno {s}); give an absolute path to a file in an existing folder", .{ @errorName(e), @tagName(last_errno) });
    } else say(env, "lesto tune: the LaunchAgent text could not be built ({s})", .{@errorName(e)});
    return 2;
}

/// The plist `--background` writes (and `--print-plist-for-test` prints): the agent log in the tune folder, and a custom
/// `MLX_SERVE_KERNEL_TABLE` path carried into the job's environment.
fn agentPlist(a: Allocator, env: Env, root: []const u8, bin: []const u8, idle: u32) ![]u8 {
    const log_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, AGENT_LOG_NAME });
    return plistTextWith(a, bin, idle, log_path, try carriedTablePath(a, env));
}

/// The custom table path the LaunchAgent carries, absolute: launchd starts the job in another working directory, so a
/// relative path is resolved here against this process's one (the real path of its folder, then its name).
fn carriedTablePath(a: Allocator, env: Env) !?[]const u8 {
    const p = switch (env.engine.table_switch) {
        .path => |p| p,
        else => return null,
    };
    if (p.len > 0 and p[0] == '/') return p;
    const base = std.fs.path.basename(p);
    if (base.len == 0 or std.mem.eql(u8, base, ".") or std.mem.eql(u8, base, "..") or p[p.len - 1] == '/') return error.TablePathNotAFile;
    const parent = std.fs.path.dirname(p) orelse ".";
    var pb: [MAX_PATH]u8 = undefined;
    const pz = cz(&pb, parent) orelse return error.NameTooLong;
    var rb: [MAX_PATH]u8 = undefined;
    const r = realpath(pz, &rb) orelse return failWith(@as(c_int, -1), error.TablePathUnresolved);
    return try std.fmt.allocPrint(a, "{s}/{s}", .{ std.mem.span(r), base });
}

fn printKey(env: Env, k: kernel_table.Key) void {
    say(env, "[tune] this Mac: gpu_arch={s} gpu_cores={d} hw_model={s} os_build={s} kernel_hash={s} mlx_version={s}", .{ k.gpu_arch, k.gpu_cores, k.hw_model, k.os_build, k.kernel_hash, k.mlx_version });
}

fn statusText(buf: []u8, res: kernel_table.ReadResult) []const u8 {
    return switch (res) {
        .absent => "absent",
        .valid => "valid",
        .refused, .key_mismatch => blk: {
            var tb: [320]u8 = undefined;
            const why = res.refusal().?.text(&tb);
            break :blk std.fmt.bufPrint(buf, "refused ({s})", .{why}) catch "refused";
        },
    };
}

fn switchIsOff(env: Env) bool {
    return switch (env.engine.table_switch) {
        .off, .empty => true,
        else => false,
    };
}

fn shownPath(buf: []u8, env: Env, path: []const u8) []const u8 {
    return kernel_table.printablePath(buf, env.engine.home, path);
}

fn readKey(env: Env, a: Allocator) ?kernel_table.Key {
    const k = kernel_table.readMacKey(a) catch |e| {
        say(env, "lesto tune: this Mac's key could not be read ({s})", .{@errorName(e)});
        return null;
    };
    if (k.gpu_arch.len == 0) {
        say(env, "lesto tune: no Metal device", .{});
        return null;
    }
    return k;
}

fn signalsOf(c: *tune_signals.Choice) tune_signals.Signals {
    return switch (c.*) {
        .real => |*r| r.signals(),
        .scripted => |*s| s.s.signals(),
    };
}

fn deinitChoice(c: *tune_signals.Choice) void {
    switch (c.*) {
        .real => |*r| r.deinit(),
        .scripted => |*s| s.s.deinit(),
    }
}

fn kernelResultLine(env: Env, name: []const u8, layout: ?[]const u8, reason: ?ServedReason, ev: ?kernel_table.KernelEvidence) void {
    if (layout) |l| {
        if (ev) |e| {
            say(env, "[tune] {s}: {s}, {d:.1}% faster than the served layout", .{ name, l, gainPct(e) });
        } else say(env, "[tune] {s}: {s}", .{ name, l });
    } else {
        say(env, "[tune] {s}: served ({s})", .{ name, if (reason) |r| @tagName(r) else "no reason recorded" });
    }
}

fn gainPct(e: kernel_table.KernelEvidence) f64 {
    if (e.ref_ns <= 0) return 0;
    // both table-supplied values go to f64 before the subtraction: an i64 difference can overflow
    const ref: f64 = @floatFromInt(e.ref_ns);
    const pick: f64 = @floatFromInt(e.pick_ns);
    return (ref - pick) * 100.0 / ref;
}

pub const AgentRemoval = struct {
    /// A plist was there and is now deleted.
    removed: bool,
    /// Every step either did its work or found nothing to do; false after a reported failure.
    ok: bool,
};

/// How many times `--agent`'s own removal tries a busy install lock, and the pause between tries (a test shortens it).
const INSTALL_LOCK_TRIES: u32 = 5;
pub var install_lock_retry_ns: u64 = std.time.ns_per_s;

/// Deletes the plist, then (with `launchctl`) boots the agent out, holding `install.lock` for the whole removal: every
/// change to the LaunchAgent (install or removal) is serialized by it, so a `--background` cannot bootstrap a new plist
/// between this unlink and this bootout. A busy install lock: `--stop` (`retry` false) reports it at once; the agent's
/// own removal tries a few times a second apart, then reports it. A plist that is absent (ENOENT) and a service that
/// is not loaded are nothing to do; every other failure is reported and makes `ok` false, after the remaining step ran.
fn removeLaunchAgent(a: Allocator, io: std.Io, env: Env, retry: bool) AgentRemoval {
    var r: AgentRemoval = .{ .removed = false, .ok = true };
    const not_done: AgentRemoval = .{ .removed = false, .ok = false };
    const dir = env.launch_agents_dir orelse return r;
    const root = env.engine.tune_root orelse {
        say(env, NO_HOME, .{});
        return not_done;
    };
    const plist = std.fmt.allocPrint(a, "{s}/{s}", .{ dir, PLIST_NAME }) catch {
        say(env, "lesto tune: out of memory; the LaunchAgent was not touched", .{});
        return not_done;
    };
    // only the folder itself is made (0700 when missing): the other tuner files are left for stopRunning to judge
    ensureFolder(root) catch |e| {
        say(env, "lesto tune: the tune folder {s} cannot hold the install lock ({s}); the LaunchAgent was not touched", .{ root, @errorName(e) });
        return not_done;
    };
    var tries: u32 = 0;
    var ifd: c_int = -1;
    while (ifd < 0) {
        const got = takeInstallLock(root) catch |e| {
            say(env, "lesto tune: the install lock could not be taken ({s}, errno {s}); the LaunchAgent was not touched", .{ @errorName(e), @tagName(last_errno) });
            return not_done;
        };
        if (got) |fd| {
            ifd = fd;
            break;
        }
        tries += 1;
        if (!retry or tries >= INSTALL_LOCK_TRIES) {
            if (retry) {
                say(env, "[tune] an install is running; the LaunchAgent was left in place (lesto tune --stop removes it)", .{});
            } else say(env, "[tune] an install is running", .{});
            return not_done;
        }
        _ = usleep(@intCast(@min(install_lock_retry_ns / std.time.ns_per_us, 1_000_000)));
    }
    defer {
        _ = flock(ifd, LOCK_UN);
        _ = std.c.close(ifd);
    }
    if (removeFile(plist)) |gone| {
        r.removed = gone;
        if (gone) say(env, "[tune] LaunchAgent removed: {s}", .{plist});
    } else |e| {
        say(env, "lesto tune: the LaunchAgent {s} could not be removed ({s}, errno {s})", .{ plist, @errorName(e), @tagName(last_errno) });
        r.ok = false;
    }
    // the last act of `--agent`: this bootout ends the agent itself, which is its expected end
    env.out.flush() catch outFail();
    const target = guiTarget(a, true) catch return .{ .removed = r.removed, .ok = false };
    if (runLaunchctl(a, io, env, &.{ "/bin/launchctl", "bootout", target })) |res| {
        if (bootoutVerdict(res) == .failed) {
            say(env, "lesto tune: launchctl bootout {s} failed: {s}", .{ target, std.mem.trim(u8, res.text, " \n") });
            r.ok = false;
        }
    }
    return r;
}

fn guiTarget(a: Allocator, with_label: bool) ![]u8 {
    const uid = std.c.getuid();
    return if (with_label) std.fmt.allocPrint(a, "gui/{d}/{s}", .{ uid, AGENT_LABEL }) else std.fmt.allocPrint(a, "gui/{d}", .{uid});
}

pub const LaunchctlResult = struct {
    /// False when launchctl could not be started at all.
    ran: bool,
    /// Its exit status; null when it ended by a signal.
    code: ?u8,
    /// Its stdout and stderr, or the start error's name.
    text: []const u8,

    fn ok(self: LaunchctlResult) bool {
        return self.ran and self.code != null and self.code.? == 0;
    }
};

pub const BootoutVerdict = enum { removed, not_loaded, failed };

/// What a `launchctl bootout` result means. A service that is not loaded is nothing to remove, not a failure: launchctl
/// reports it with exit status 3 (ESRCH, "No such process") or 113 ("Could not find specified service").
pub fn bootoutVerdict(r: LaunchctlResult) BootoutVerdict {
    if (!r.ran) return .failed;
    const c = r.code orelse return .failed;
    if (c == 0) return .removed;
    if (c == 3 or c == 113) return .not_loaded;
    if (std.mem.indexOf(u8, r.text, "No such process") != null or std.mem.indexOf(u8, r.text, "Could not find specified service") != null) return .not_loaded;
    return .failed;
}

/// Test seam: answers every launchctl call in place of the real one (the tests keep `Env.launchctl` false, so with no
/// override set no call is made at all).
pub var launchctl_override: ?*const fn ([]const []const u8) LaunchctlResult = null;

/// One launchctl call: the seam when set, else the real `/bin/launchctl` when `env.launchctl`, else null (skipped).
fn runLaunchctl(a: Allocator, io: std.Io, env: Env, argv: []const []const u8) ?LaunchctlResult {
    if (launchctl_override) |f| return f(argv);
    if (!env.launchctl) return null;
    return launchctl(a, io, argv);
}

fn launchctl(a: Allocator, io: std.Io, argv: []const []const u8) LaunchctlResult {
    const r = std.process.run(a, io, .{ .argv = argv }) catch |e| return .{ .ran = false, .code = null, .text = @errorName(e) };
    const code: ?u8 = switch (r.term) {
        .exited => |c| c,
        else => null,
    };
    const text = std.fmt.allocPrint(a, "{s}{s}", .{ r.stdout, r.stderr }) catch "";
    return .{ .ran = true, .code = code, .text = text };
}

/// `--agent`'s last act: removes the LaunchAgent; a failed removal makes the agent exit 2 so agent.log shows it.
fn agentRemoveSelf(a: Allocator, io: std.Io, env: Env) u8 {
    return if (removeLaunchAgent(a, io, env, true).ok) 0 else 2;
}

/// The run's layout for `k` written into `b`; null means only "no pick" (served). Nothing is allocated, so no failure
/// can share null with "served".
fn layoutText(b: []u8, p: transformer.MoeRowsLayoutPlan, k: tune_stats.Kernel) ?[]const u8 {
    return switch (k) {
        .gateup => if (p.gateup) |g| kernel_table.gateUpText(b, g) else null,
        .down => if (p.down) |d| kernel_table.downText(b, d) else null,
    };
}

// ── the default action and `--agent` ──

fn tuneAction(a: Allocator, io: std.Io, env: Env, f: Flags) u8 {
    const agent = f.action == .agent;
    const root = env.engine.tune_root orelse return noHome(env);
    var choice = tune_signals.fromValue(a, io, env.fake_signals) catch |e| {
        say(env, "lesto tune: the signals could not be read ({s})", .{@errorName(e)});
        return 2;
    };
    defer deinitChoice(&choice);
    const sig = signalsOf(&choice);
    if (agent) {
        var b: [256]u8 = undefined;
        say(env, "{s}", .{tune_signals.agentSignalsLine(&b, sig)});
    }
    const scripted = switch (choice) {
        .scripted => |s| blk: {
            say(env, "[tune] signals: scripted from {s}", .{s.path});
            break :blk true;
        },
        .real => false,
    };
    const mac = readKey(env, a) orelse return 2;
    printKey(env, mac);
    const path = (kernel_table.tablePath(a, root, env.engine.table_switch) catch return oom(env)).?;
    // checked before the first read too: a "current" table found at a reserved path must not lead to the agent's
    // self-removal deleting it (checked again under the lock, once the folder surely exists)
    if (refuseCollision(env, path, root)) |code| return code;
    const res = kernel_table.readTable(a, io, path, mac);
    defer res.deinit();
    var pb: [MAX_PATH + 2]u8 = undefined;
    var sb: [512]u8 = undefined;
    say(env, "[tune] table {s}: {s}", .{ shownPath(&pb, env, path), statusText(&sb, res) });
    say(env, "[tune] {d} candidates; with no discards at most {d} timing batches plus 4 to 12 calibration batches a window; discards add up to 4 re-timed batches a candidate and up to 80 for abandoned final-stage attempts, and an interruption adds the batch or attempt it cut; about 2 minutes of GPU on an M4 Pro (estimate)", .{ CANDIDATES, MAX_TIMING_BATCHES });
    if (switchIsOff(env)) say(env, OFF_NOTE, .{});
    if (res == .valid) {
        say(env, "[tune] the table is current; lesto tune --reset and then lesto tune tunes again", .{});
        if (agent) return agentRemoveSelf(a, io, env);
        return 0;
    }

    ensureTuneFolder(root) catch |e| return folderFailed(env, root, e);
    stop_flag.store(false, .release);
    const saved = installStopHandlers();
    defer restoreStopHandlers(saved);
    // The foreground command reports a busy lock and exits 2. The agent runs once per login (KeepAlive false), so it
    // must not end as if it had tuned: it waits for the lock, polling, until it gets it or is stopped.
    var lock: Lock = undefined;
    var told = false;
    while (true) {
        switch (takeLock(io, env) catch |e| return lockFailed(env, e)) {
            .held => |l| {
                lock = l;
                break;
            },
            .busy => |b| {
                if (!agent) {
                    say(env, "[tune] a tune is running (pid {d})", .{b.pid});
                    return 2;
                }
                if (!told) {
                    say(env, "[tune] a tune is running (pid {d}); the agent waits for it, checking the lock every {d} s", .{ b.pid, agent_lock_poll_ns / std.time.ns_per_s });
                    told = true;
                }
                if (!waitUnlessStopped(agent_lock_poll_ns)) {
                    say(env, "[tune] stopped while waiting for the lock; nothing was run", .{});
                    return 130;
                }
            },
        }
    }
    var lock_held = true;
    defer if (lock_held) lock.release();
    if (refuseCollision(env, path, root)) |code| return code;
    // the decision to tune is made again now that the lock is held: a tune that finished while this one waited for the
    // lock (or between the first read and the lock) wrote a current table, which is kept
    {
        const again = kernel_table.readTable(a, io, path, mac);
        defer again.deinit();
        if (again == .valid) {
            say(env, "[tune] the table became current while this run waited for the lock; nothing was run", .{});
            lock.release();
            lock_held = false;
            if (agent) return agentRemoveSelf(a, io, env);
            return 0;
        }
    }
    touchPrivate(root, LOG_NAME) catch |e| return folderFailed(env, root, e);
    var line: ProgressLine = .{ .out = env.out, .tty = env.stdout_is_tty and !agent };
    const result: tune_sched.SchedResult = if (sched_override) |so| so() else runScheduler(a, io, root, mac, f.idle_minutes, sig, &line) catch |e| {
        line.end();
        if (e == error.TuneWorkerFailed) {
            say(env, "lesto tune: the GPU worker failed; the run ended (progress kept)", .{});
        } else say(env, "lesto tune: the tune could not run ({s}; progress kept)", .{@errorName(e)});
        return 2;
    };
    line.end();
    const o = switch (result) {
        .stopped => {
            say(env, "[tune] stopped; progress kept in {s}/{s}", .{ root, PROGRESS_NAME });
            return 130;
        },
        .finished => |fin| fin,
    };
    const t = buildTable(a, mac, o, .{ .scripted = scripted }) catch return oom(env);
    const bytes = kernel_table.serialize(a, t) catch return oom(env);
    kernel_table.writeAtomic(a, io, path, bytes) catch |e| {
        say(env, "lesto tune: the table could not be written to {s} ({s})", .{ path, @errorName(e) });
        return 2;
    };
    var gb: [48]u8 = undefined;
    var db: [48]u8 = undefined;
    kernelResultLine(env, "gate/up", layoutText(&gb, o.plan, .gateup), o.served_reasons[0], o.evidence[0]);
    kernelResultLine(env, "down", layoutText(&db, o.plan, .down), o.served_reasons[1], o.evidence[1]);
    say(env, "[tune] table written to {s}; applies at the next lesto serve start", .{shownPath(&pb, env, path)});
    if (switchIsOff(env)) say(env, OFF_NOTE, .{});
    if (agent) {
        lock.release();
        lock_held = false;
        return agentRemoveSelf(a, io, env);
    }
    return 0;
}

fn runScheduler(a: Allocator, io: std.Io, root: []const u8, mac: kernel_table.Key, idle: u32, sig: tune_signals.Signals, line: *ProgressLine) !tune_sched.SchedResult {
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{});
    defer dir.close(io);
    var gw = tune_gpu.GpuWorker.init(a, WORKER_SEED);
    defer gw.deinit();
    const cfg: tune_sched.RunConfig = .{ .io = io, .root = dir, .monitor = .thread, .key = mac, .idle_minutes = idle, .reporter = line.reporter() };
    return tune_sched.run(a, cfg, sig, gw.worker(), &stop_flag);
}

// ── `--background` ──

fn backgroundAction(a: Allocator, io: std.Io, env: Env, f: Flags) u8 {
    const root = env.engine.tune_root orelse return noHome(env);
    const agents = env.launch_agents_dir orelse return noHome(env);
    if (switchIsOff(env)) {
        // the job would tune a table the engine does not read while the switch is off
        say(env, "lesto tune: --background refused: MLX_SERVE_KERNEL_TABLE is off, so the engine would not read the table it tunes; unset it (or set it to a table path) and run lesto tune --background again", .{});
        return 2;
    }
    // a custom table path that names the plist (or a tuner file) is refused before anything is written: the install
    // would put XML there, the agent its table, and its self-removal would delete the table
    if (carriedTablePath(a, env) catch |e| return plistFailed(env, e)) |p| {
        if (refuseCollision(env, p, root)) |code| return code;
    }
    say(env, "macOS will show a \"Background Items Added\" notice for lesto", .{});
    const bin = exePath(a) catch |e| {
        say(env, "lesto tune: the lesto binary's path could not be read ({s})", .{@errorName(e)});
        return 2;
    };
    ensureTuneFolder(root) catch |e| return folderFailed(env, root, e);
    touchPrivate(root, AGENT_LOG_NAME) catch |e| return folderFailed(env, root, e);
    const text = agentPlist(a, env, root, bin, f.idle_minutes) catch |e| return plistFailed(env, e);
    if (carriedTablePath(a, env) catch null) |p| say(env, "[tune] the LaunchAgent carries MLX_SERVE_KERNEL_TABLE={s}", .{p});
    ensureFolder(agents) catch |e| {
        say(env, "lesto tune: cannot create {s} ({s})", .{ agents, @errorName(e) });
        return 2;
    };
    const plist = std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME }) catch return oom(env);
    // one install at a time: two would share the target, and the rollback of one could undo the other
    const ilock = takeInstallLock(root) catch |e| return lockFailed(env, e);
    const ifd = ilock orelse {
        say(env, "[tune] another install is running", .{});
        return 2;
    };
    defer {
        _ = flock(ifd, LOCK_UN);
        _ = std.c.close(ifd);
    }
    return installPlist(a, io, env, plist, text);
}

/// `<tune root>/install.lock` (0600, the private checks), `flock(LOCK_EX | LOCK_NB)`: the descriptor, or null when
/// another install or removal holds it. The real `flock` always: `flock_override` stands in for the tune's lock only.
fn takeInstallLock(root: []const u8) !?c_int {
    last_errno = .SUCCESS;
    var buf: [MAX_PATH]u8 = undefined;
    const z = joinZ(&buf, root, INSTALL_LOCK_NAME) orelse return error.NameTooLong;
    // a wide install lock someone holds is reported as a running install
    const fd = try openPrivateLock(z) orelse return null;
    const rc = flock(fd, LOCK_EX | LOCK_NB);
    if (rc != 0) {
        const e = std.c.errno(rc);
        _ = std.c.close(fd);
        if (e == .AGAIN) return null;
        last_errno = e;
        return error.LockFailed;
    }
    return fd;
}

/// Every name and argument the install needs, made before the old agent is touched, so that between its bootout and
/// the new bootstrap nothing can fail but launchctl and the renames, which all go through the rollback.
const InstallPaths = struct {
    plist: [*:0]const u8,
    staged: [*:0]const u8,
    backup: [*:0]const u8,
    plist_text: []const u8,
    staged_text: []const u8,
    target: []const u8,
    gui: []const u8,
};

fn installPaths(a: Allocator, plist: []const u8) !InstallPaths {
    const pid = std.c.getpid();
    const staged = try std.fmt.allocPrintSentinel(a, "{s}.staged-{d}", .{ plist, pid }, 0);
    const backup = try std.fmt.allocPrintSentinel(a, "{s}.previous-{d}", .{ plist, pid }, 0);
    const plist_z = try std.fmt.allocPrintSentinel(a, "{s}", .{plist}, 0);
    if (std.mem.indexOfScalar(u8, plist, 0) != null) return error.NameTooLong;
    return .{
        .plist = plist_z.ptr,
        .staged = staged.ptr,
        .backup = backup.ptr,
        .plist_text = plist,
        .staged_text = staged,
        .target = try guiTarget(a, true),
        .gui = try guiTarget(a, false),
    };
}

/// The install as one step that either completes or leaves the previous state: the new plist is staged beside the
/// target, the old agent is booted out only once the new file exists, the staged file is renamed into place, and a
/// failed bootstrap (or rename) puts the previous plist back (and loads it again when it was loaded), or removes the
/// new one when there was none.
fn installPlist(a: Allocator, io: std.Io, env: Env, plist: []const u8, text: []const u8) u8 {
    const p = installPaths(a, plist) catch |e| {
        say(env, "lesto tune: the LaunchAgent install could not be prepared ({s}); nothing changed", .{@errorName(e)});
        return 2;
    };
    kernel_table.writeAtomic(a, io, p.staged_text, text) catch |e| {
        say(env, "lesto tune: the LaunchAgent could not be staged at {s} ({s}); nothing changed", .{ p.staged_text, @errorName(e) });
        return 2;
    };
    // whether a previous directory entry is there is decided without following a link: a dangling symlink is an entry,
    // which the rename below moves aside (the link itself) and a rollback renames back
    var lst: std.c.Stat = undefined;
    const had = if (c_lstat(p.plist, &lst) == 0) true else blk: {
        const le = std.c.errno(@as(c_int, -1));
        if (le == .NOENT) break :blk false;
        _ = std.c.unlink(p.staged);
        say(env, "lesto tune: the installed LaunchAgent entry {s} cannot be examined (errno {s}); nothing changed", .{ plist, @tagName(le) });
        return 2;
    };
    // only a file or a link is a previous plist: a folder (or any other kind of entry) there is refused untouched
    if (had) {
        const kind = @as(u32, @intCast(lst.mode)) & S_IFMT;
        if (kind != S_IFREG and kind != S_IFLNK) {
            _ = std.c.unlink(p.staged);
            say(env, "lesto tune: {s} is {s}, not a plist file; nothing changed (move it away and run lesto tune --background again)", .{ plist, entryKindWord(kind) });
            return 2;
        }
    }
    var booted_out = false;
    if (had) {
        if (runLaunchctl(a, io, env, &.{ "/bin/launchctl", "bootout", p.target })) |r| switch (bootoutVerdict(r)) {
            .removed => booted_out = true,
            .not_loaded => {},
            .failed => {
                _ = std.c.unlink(p.staged);
                say(env, "lesto tune: launchctl bootout of the installed LaunchAgent failed: {s}; nothing changed", .{std.mem.trim(u8, r.text, " \n")});
                return 2;
            },
        };
        // the previous plist is kept aside (a rename, no allocation) so a failure below can put it back as it was
        if (std.c.rename(p.plist, p.backup) != 0) {
            last_errno = std.c.errno(@as(c_int, -1));
            _ = std.c.unlink(p.staged);
            say(env, "lesto tune: the installed LaunchAgent could not be set aside (errno {s})", .{@tagName(last_errno)});
            reload(a, io, env, p, booted_out);
            return 2;
        }
    }
    if (std.c.rename(p.staged, p.plist) != 0) {
        last_errno = std.c.errno(@as(c_int, -1));
        _ = std.c.unlink(p.staged);
        say(env, "lesto tune: the LaunchAgent could not be put in place at {s} (errno {s})", .{ plist, @tagName(last_errno) });
        restorePrevious(a, io, env, p, had, booted_out);
        return 2;
    }
    if (runLaunchctl(a, io, env, &.{ "/bin/launchctl", "bootstrap", p.gui, p.plist_text })) |r| {
        if (!r.ok()) {
            say(env, "lesto tune: launchctl bootstrap failed: {s}", .{std.mem.trim(u8, r.text, " \n")});
            restorePrevious(a, io, env, p, had, booted_out);
            return 2;
        }
    }
    say(env, "[tune] LaunchAgent installed: {s}; it tunes while this Mac is idle and removes itself when done; lesto tune --stop removes it", .{plist});
    if (had) {
        const urc = std.c.unlink(p.backup);
        if (urc != 0 and std.c.errno(urc) != .NOENT) {
            say(env, "lesto tune: the previous LaunchAgent set aside at {s} could not be removed (errno {s}); delete it by hand", .{ std.mem.span(p.backup), @tagName(std.c.errno(urc)) });
            return 2;
        }
    }
    return 0;
}

fn entryKindWord(kind: u32) []const u8 {
    return switch (kind) {
        S_IFDIR => "a folder",
        0o010000 => "a FIFO",
        0o020000 => "a character device",
        0o060000 => "a block device",
        0o140000 => "a socket",
        else => "an entry of another kind",
    };
}

/// Puts the previous plist back after a failed install (a rename over the new one) and loads it again when the install
/// had booted it out, or removes the new plist when there was none; says what it did. Allocates nothing.
fn restorePrevious(a: Allocator, io: std.Io, env: Env, p: InstallPaths, had: bool, booted_out: bool) void {
    if (!had) {
        if (std.c.unlink(p.plist) == 0 or std.c.errno(@as(c_int, -1)) == .NOENT) {
            say(env, "[tune] the new LaunchAgent was removed; none is installed, as before", .{});
        } else say(env, "lesto tune: the new LaunchAgent {s} could not be removed (errno {s}); delete it by hand", .{ p.plist_text, @tagName(std.c.errno(@as(c_int, -1))) });
        return;
    }
    if (std.c.rename(p.backup, p.plist) != 0) {
        say(env, "lesto tune: the previous LaunchAgent could not be put back (errno {s}); it is at {s}", .{ @tagName(std.c.errno(@as(c_int, -1))), std.mem.span(p.backup) });
        return;
    }
    say(env, "[tune] the previous LaunchAgent was put back at {s}", .{p.plist_text});
    reload(a, io, env, p, booted_out);
}

/// Loads the previous agent again when the install had booted it out.
fn reload(a: Allocator, io: std.Io, env: Env, p: InstallPaths, booted_out: bool) void {
    if (!booted_out) return;
    if (runLaunchctl(a, io, env, &.{ "/bin/launchctl", "bootstrap", p.gui, p.plist_text })) |r| {
        if (r.ok()) {
            say(env, "[tune] the previous LaunchAgent was loaded again", .{});
        } else say(env, "[tune] the previous LaunchAgent could not be loaded again ({s}); it loads at the next login", .{std.mem.trim(u8, r.text, " \n")});
    }
}

// ── `--stop` ──

fn stopAction(a: Allocator, io: std.Io, env: Env) u8 {
    // an unknown tune folder is an error, never "nothing runs": exit 2 before anything is touched
    const root = env.engine.tune_root orelse return noHome(env);
    var code: u8 = 0;
    const removal = removeLaunchAgent(a, io, env, false);
    if (!removal.ok) {
        code = 2;
    } else if (!removal.removed) {
        say(env, "[tune] no LaunchAgent installed", .{});
    }
    // the running tune is stopped even when the LaunchAgent could not be removed
    const sr = stopRunning(io, env);
    switch (sr) {
        .not_running => say(env, "[tune] no tune is running", .{}),
        .signalled => |pid| say(env, "[tune] sent SIGTERM to the tune (pid {d}); its progress is kept", .{pid}),
        .unidentified => |pid| if (pid > 0) {
            say(env, "lesto tune: a tune holds the lock, but its holder (pid {d}) could not be identified as the tune (its start time differs); nothing was signalled", .{pid});
        } else say(env, "lesto tune: a tune holds the lock in {s}, but its record names no process yet; nothing was signalled, run lesto tune --stop again", .{root}),
        .untrusted => |why| say(env, "lesto tune: a process holds the lock in {s}, but the lock file is not trusted ({s}); nothing was signalled", .{ root, why }),
        .signal_failed => |sf| say(env, "lesto tune: SIGTERM to the tune (pid {d}) failed (errno {s})", .{ sf.pid, @tagName(sf.err) }),
        .failed => |e| say(env, "lesto tune: the lock in {s} could not be checked (errno {s}); nothing was signalled", .{ root, @tagName(e) }),
    }
    // exit 0 only when no tune runs or the identified tune was signalled (and the LaunchAgent step held)
    if (!sr.ok()) code = 2;
    return code;
}

// ── `--reset` ──

fn resetAction(a: Allocator, io: std.Io, env: Env) u8 {
    const root = env.engine.tune_root orelse return noHome(env);
    const path = (kernel_table.tablePath(a, root, env.engine.table_switch) catch return oom(env)).?;
    ensureTuneFolder(root) catch |e| return folderFailed(env, root, e);
    const lock = switch (takeLock(io, env) catch |e| return lockFailed(env, e)) {
        .held => |l| l,
        .busy => |b| {
            say(env, "[tune] a tune is running (pid {d})", .{b.pid});
            return 2;
        },
    };
    defer lock.release();
    if (refuseCollision(env, path, root)) |code| return code;
    const progress = std.fmt.allocPrint(a, "{s}/{s}", .{ root, PROGRESS_NAME }) catch return oom(env);
    var removed: u32 = 0;
    for ([_][]const u8{ path, progress }) |p| {
        const gone = removeFile(p) catch |e| {
            say(env, "lesto tune: {s} could not be removed ({s}, errno {s})", .{ p, @errorName(e), @tagName(last_errno) });
            return 2;
        };
        if (gone) {
            removed += 1;
            say(env, "[tune] removed {s}", .{p});
        }
    }
    if (removed == 0) say(env, "[tune] nothing to remove: no table and no progress", .{});
    return 0;
}

// ── `--show` (spec 7.3) ──

fn keyValue(buf: []u8, k: kernel_table.Key, f: kernel_table.KeyField) []const u8 {
    return switch (f) {
        .gpu_arch => k.gpu_arch,
        .gpu_cores => std.fmt.bufPrint(buf, "{d}", .{k.gpu_cores}) catch "",
        .hw_model => k.hw_model,
        .kernel_hash => k.kernel_hash,
        .mlx_version => k.mlx_version,
        .os_build => k.os_build,
    };
}

fn planLine(env: Env, name: []const u8, layout: ?[]const u8, reason: ?[]const u8, ev: ?kernel_table.KernelEvidence) void {
    const w = env.out;
    const l = layout orelse {
        say(env, "[tune] plan {s}: served ({s})", .{ name, reason orelse "no reason recorded" });
        return;
    };
    w.print("[tune] plan {s}: {s}", .{ name, l }) catch outFail();
    if (ev) |e| {
        w.print(", gain {d:.1}% (reference {d} ns, pick {d} ns a call)", .{ gainPct(e), e.ref_ns, e.pick_ns }) catch outFail();
        const names = [4][]const u8{ "confirm1 indep", "confirm1 shared", "confirm2 indep", "confirm2 shared" };
        for (e.confirm, names) |c, n| w.print("; {s} {d} ns [{d}, {d}] window {d}", .{ n, c.median_ns, c.lo_ns, c.hi_ns, c.window }) catch outFail();
    }
    say(env, "", .{});
}

fn sourceWord(s: transformer.MoeRowsLayoutSource) []const u8 {
    return switch (s) {
        .env => "environment override",
        .env_refused => "served (environment override refused)",
        .local => "local table",
        .builtin => "built-in entry",
        .none => "served",
    };
}

fn excludedLine(env: Env, label: []const u8, list: []const kernel_table.Excluded) void {
    env.out.print("[tune] {s}:", .{label}) catch outFail();
    if (list.len == 0) env.out.writeAll(" none") catch outFail();
    for (list, 0..) |x, i| env.out.print("{s} {s} {s} ({s})", .{ if (i == 0) "" else ";", x.kernel, x.cand, x.reason }) catch outFail();
    say(env, "", .{});
}

fn notMeasurableLine(env: Env, label: []const u8, list: []const kernel_table.NotMeasurable) void {
    env.out.print("[tune] {s}:", .{label}) catch outFail();
    if (list.len == 0) env.out.writeAll(" none") catch outFail();
    for (list, 0..) |x, i| env.out.print("{s} {s} {s}", .{ if (i == 0) "" else ";", x.kernel, x.cand }) catch outFail();
    say(env, "", .{});
}

/// The table's layout for `k` written into `b` (no allocation, so it cannot fail and read as "served"); null when served.
fn tablePlanText(b: []u8, t: kernel_table.Table, k: tune_stats.Kernel) ?[]const u8 {
    return switch (k) {
        .gateup => switch (t.plan.gateup) {
            .layout => |g| kernel_table.gateUpText(b, g),
            else => null,
        },
        .down => switch (t.plan.down) {
            .layout => |d| kernel_table.downText(b, d),
            else => null,
        },
    };
}

fn showAction(a: Allocator, io: std.Io, env: Env) u8 {
    const path = (kernel_table.tablePath(a, env.engine.tune_root, env.engine.table_switch) catch return oom(env)) orelse return noHome(env);
    const mac = readKey(env, a) orelse return 2;
    const res = kernel_table.readTable(a, io, path, mac);
    defer res.deinit();
    const table: ?kernel_table.Table = switch (res) {
        .valid => |p| p.value,
        .key_mismatch => |k| k.table.value,
        else => null,
    };
    var pb: [MAX_PATH + 2]u8 = undefined;
    var sb: [512]u8 = undefined;
    say(env, "[tune] table: {s} ({s})", .{ shownPath(&pb, env, path), statusText(&sb, res) });

    // the key, field by field
    for (std.enums.values(kernel_table.KeyField)) |field| {
        var mb: [32]u8 = undefined;
        const mine = keyValue(&mb, mac, field);
        if (table) |t| {
            var tb: [32]u8 = undefined;
            const theirs = keyValue(&tb, t.key, field);
            if (std.mem.eql(u8, mine, theirs)) {
                say(env, "[tune] key {s}: matches ({s})", .{ @tagName(field), mine });
            } else say(env, "[tune] key {s}: table {s}, this Mac {s}", .{ @tagName(field), theirs, mine });
        } else say(env, "[tune] key {s}: this Mac {s} (no table to compare)", .{ @tagName(field), mine });
    }

    // the plans
    if (table) |t| {
        var gb: [48]u8 = undefined;
        var db: [48]u8 = undefined;
        planLine(env, "gateup", tablePlanText(&gb, t, .gateup), t.info.served_reasons.gateup, t.evidence.gateup);
        planLine(env, "down", tablePlanText(&db, t, .down), t.info.served_reasons.down, t.evidence.down);
    } else {
        say(env, "[tune] plan gateup: no table", .{});
        say(env, "[tune] plan down: no table", .{});
    }

    // the source the engine would use now, and why
    var source_ok = true;
    if (transformer.moeRowsLayoutGatherInputs(a, io, env.engine)) |in| {
        const d = transformer.moeRowsLayoutDecideSources(in);
        say(env, "[tune] source: gate/up {s}, down {s} (switch {s})", .{ sourceWord(d.gateup_src), sourceWord(d.down_src), @tagName(d.sw) });
        for (0..d.n_lines) |i| say(env, "[tune]   {s}", .{d.line(i)});
        if (d.gateup_src == .builtin or d.down_src == .builtin) say(env, "[tune] the built-in entry is not identity-checked on this Mac's macOS build; lesto tune --verify checks it", .{});
    } else |e| {
        say(env, "[tune] source: unknown ({s})", .{@errorName(e)});
        source_ok = false;
    }

    // the table's lists
    if (table) |t| {
        excludedLine(env, "excluded", t.identity.excluded);
        notMeasurableLine(env, "not measurable", t.info.not_measurable);
        say(env, "[tune] served: gateup {s}, down {s}", .{ t.info.served_reasons.gateup orelse "no (a pick)", t.info.served_reasons.down orelse "no (a pick)" });
    } else {
        say(env, "[tune] excluded: no table", .{});
        say(env, "[tune] not measurable: no table", .{});
        say(env, "[tune] served: no table", .{});
    }

    const progress_ok = showProgress(a, io, env, mac);
    const agent_ok = showAgent(a, io, env);
    // the report is printed in full either way; a section that failed (the source, progress that exists but could not
    // be read, the agent's status) fails the command
    return if (progress_ok and agent_ok and source_ok) 0 else 2;
}

/// Prints the progress section; false (after printing `progress: unreadable (<error>)`) when progress exists but could
/// not be read or summarized: only a missing folder or file means "no unfinished run".
fn showProgress(a: Allocator, io: std.Io, env: Env, mac: kernel_table.Key) bool {
    const root = env.engine.tune_root orelse {
        say(env, "[tune] progress: no tune folder", .{});
        return true;
    };
    var dir = std.Io.Dir.cwd().openDir(io, root, .{}) catch |e| {
        if (e == error.FileNotFound) {
            say(env, "[tune] progress: no unfinished run", .{});
            return true;
        }
        say(env, "[tune] progress: unreadable ({s})", .{@errorName(e)});
        return false;
    };
    defer dir.close(io);
    const p = tune_sched.loadProgress(a, io, dir, mac) catch |e| {
        say(env, "[tune] progress: unreadable ({s})", .{@errorName(e)});
        return false;
    };
    if (!p.had_file) {
        say(env, "[tune] progress: no unfinished run", .{});
        return true;
    }
    if (p.restarted) {
        say(env, "[tune] progress: {s}; the next run starts over", .{p.restart_reason});
        return true;
    }
    const s = tune_sched.summarize(a, p) catch |e| {
        say(env, "[tune] progress: unreadable ({s})", .{@errorName(e)});
        return false;
    };
    var total: u32 = 0;
    for (s.discarded) |n| total +|= n;
    env.out.print("[tune] progress: {d} batches kept, {d} discarded (", .{ s.kept, total }) catch outFail();
    for (kernel_table.DISCARD_REASONS, s.discarded, 0..) |r, n, i| env.out.print("{s}{s} {d}", .{ if (i == 0) "" else ", ", r, n }) catch outFail();
    say(env, "), {d} windows used, {d} void final-stage lines", .{ s.windows_used, s.void_lines });
    excludedLine(env, "progress excluded so far", s.excluded);
    notMeasurableLine(env, "progress not measurable so far", s.not_measurable);
    return true;
}

/// Prints the LaunchAgent section, always exactly one line; false (after `LaunchAgent: unreadable (<error>)`) when this
/// command failed while building or reading the agent's status, so `--show` exits 2 instead of skipping the section.
/// A plist the file system refuses to read, or an oversized one, is a state of an installed agent and is reported as such.
fn showAgent(a: Allocator, io: std.Io, env: Env) bool {
    const dir = env.launch_agents_dir orelse {
        say(env, "[tune] LaunchAgent: not installed", .{});
        return true;
    };
    const plist = std.fmt.allocPrint(a, "{s}/{s}", .{ dir, PLIST_NAME }) catch |e| {
        say(env, "[tune] LaunchAgent: unreadable ({s})", .{@errorName(e)});
        return false;
    };
    // only a missing plist is "not installed"; one that cannot be read, or is larger than any plist this file writes
    // by far, is installed and says why it cannot be shown
    const text = std.Io.Dir.cwd().readFileAlloc(io, plist, a, .limited(PLIST_READ_MAX)) catch |e| {
        if (e == error.OutOfMemory) {
            say(env, "[tune] LaunchAgent: unreadable ({s})", .{@errorName(e)});
            return false;
        }
        if (e == error.FileNotFound) {
            say(env, "[tune] LaunchAgent: not installed", .{});
        } else if (e == error.StreamTooLong) {
            say(env, "[tune] LaunchAgent: installed ({s}) but oversized (over {d} bytes); not read", .{ plist, PLIST_READ_MAX });
        } else say(env, "[tune] LaunchAgent: installed ({s}) but unreadable ({s})", .{ plist, @errorName(e) });
        return true;
    };
    const found = plistProgram(a, text) catch |e| {
        say(env, "[tune] LaunchAgent: unreadable ({s})", .{@errorName(e)});
        return false;
    };
    const prog = found orelse {
        say(env, "[tune] LaunchAgent: installed ({s}), no program path in it", .{plist});
        return true;
    };
    var zb: [MAX_PATH]u8 = undefined;
    const there = if (cz(&zb, prog)) |z| exists(z) else false;
    say(env, "[tune] LaunchAgent: installed ({s}), program {s} {s}", .{ plist, prog, if (there) "exists" else "is missing" });
    return true;
}

// ── `--verify` (spec 7.2) ──

const VerifyCase = enum { a, b, c };

fn verifyAction(a: Allocator, io: std.Io, env: Env, f: Flags) u8 {
    const root = env.engine.tune_root orelse return noHome(env);
    const path = (kernel_table.tablePath(a, root, env.engine.table_switch) catch return oom(env)).?;
    if (env.engine.moe_rows_layout_raw != null or env.engine.gateup_env != null or env.engine.down_env != null) {
        say(env, "[tune] --verify ignores MLX_SERVE_MOE_ROWS_LAYOUT, MLX_SERVE_MOE_ROWS_LAYOUT_GATEUP and MLX_SERVE_MOE_ROWS_LAYOUT_DOWN (set here): it checks the tables, not the overrides", .{});
    }
    const off = switchIsOff(env);
    if (off) say(env, OFF_NOTE, .{});
    // The read, the decision and every write happen under the lock, so a --reset or a tune that finishes meanwhile
    // cannot have its result overwritten by a table read before it.
    ensureTuneFolder(root) catch |e| return folderFailed(env, root, e);
    const lock = switch (takeLock(io, env) catch |e| return lockFailed(env, e)) {
        .held => |l| l,
        .busy => |b| {
            say(env, "[tune] a tune is running (pid {d})", .{b.pid});
            return 2;
        },
    };
    defer lock.release();
    if (refuseCollision(env, path, root)) |code| return code;
    const mac = readKey(env, a) orelse return 2;
    var pb: [MAX_PATH + 2]u8 = undefined;
    const shown = shownPath(&pb, env, path);

    // step 1, the source
    var res: kernel_table.ReadResult = .absent;
    if (!off) res = kernel_table.readTable(a, io, path, mac);
    defer res.deinit();
    var case: VerifyCase = .c;
    var table: ?kernel_table.Table = null;
    switch (res) {
        .valid => |p| {
            case = .a;
            table = p.value;
        },
        .key_mismatch => |k| if (k.detail.field == .os_build) {
            case = .b;
            table = k.table.value;
        },
        else => {},
    }
    var plan: transformer.MoeRowsLayoutPlan = .{ .gateup = null, .down = null };
    if (table) |t| {
        plan.gateup = switch (t.plan.gateup) {
            .layout => |g| g,
            else => null,
        };
        plan.down = switch (t.plan.down) {
            .layout => |d| d,
            else => null,
        };
        say(env, "[tune] verify: the local table {s}{s}", .{ shown, if (case == .b) " (its os_build differs from this Mac's)" else "" });
    } else {
        var sb: [512]u8 = undefined;
        if (off) {
            say(env, "[tune] verify: the local table is skipped; the built-in entry for this GPU", .{});
        } else say(env, "[tune] verify: the local table {s} is {s}; the built-in entry for this GPU", .{ shown, statusText(&sb, res) });
        plan = transformer.moeRowsLayoutPlanFor(transformer.deviceGeneration());
    }

    // no layout to check
    if (!transformer.moeRowsLayoutHasPlan(plan)) {
        if (case == .b) {
            var t = table.?;
            t.key.os_build = mac.os_build;
            t.identity.os_build = mac.os_build;
            if (writeTable(a, io, env, path, &t) != 0) return 2;
            say(env, "[tune] the table says \"served\" for both kernels; os_build refreshed, no GPU check needed", .{});
        } else say(env, "[tune] nothing to verify: the served layouts serve", .{});
        return 0;
    }

    // step 2, the identity check (still under the lock taken above)
    touchPrivate(root, LOG_NAME) catch |e| return folderFailed(env, root, e);
    var choice = tune_signals.fromValue(a, io, env.fake_signals) catch |e| {
        say(env, "lesto tune: the signals could not be read ({s})", .{@errorName(e)});
        return 2;
    };
    defer deinitChoice(&choice);
    if (choice == .scripted) say(env, "[tune] signals: scripted from {s}", .{choice.scripted.path});
    stop_flag.store(false, .release);
    const saved = installStopHandlers();
    defer restoreStopHandlers(saved);
    var line: ProgressLine = .{ .out = env.out, .tty = env.stdout_is_tty };
    const vr = runVerify(a, io, root, mac, f.idle_minutes, plan, signalsOf(&choice), &line) catch |e| {
        line.end();
        say(env, "lesto tune: the check could not run ({s})", .{@errorName(e)});
        return 2;
    };
    line.end();
    switch (vr) {
        .stopped => {
            say(env, "[tune] stopped; nothing written", .{});
            return 130;
        },
        .failed => |fl| {
            say(env, "[tune] verify failed: {s} {s}; nothing written", .{ @tagName(fl.kernel), fl.reason });
            if (case == .c) {
                say(env, "[tune] undo now: MLX_SERVE_MOE_ROWS_LAYOUT=0; fix: lesto tune", .{});
            } else say(env, "[tune] undo now: MLX_SERVE_KERNEL_TABLE=off; fix: lesto tune --reset, then lesto tune", .{});
            return 1;
        },
        .identical => |ok| {
            say(env, "[tune] verify: identical ({d} layouts, {d} calls compared)", .{ ok.kernels, ok.calls_compared });
            if (case == .c) return 0;
            var t = table.?;
            if (case == .b) t.key.os_build = mac.os_build;
            t.identity = .{
                .plans_checked = ok.kernels,
                .cases = 10,
                .calls_compared = ok.calls_compared,
                .os_build = mac.os_build,
                .all_identical = true,
                .excluded = t.identity.excluded,
            };
            if (writeTable(a, io, env, path, &t) != 0) return 2;
            say(env, "[tune] table rewritten: {s}", .{shown});
            return 0;
        },
    }
}

fn writeTable(a: Allocator, io: std.Io, env: Env, path: []const u8, t: *kernel_table.Table) u8 {
    kernel_table.finalize(a, t) catch return oom(env);
    const bytes = kernel_table.serialize(a, t.*) catch return oom(env);
    kernel_table.writeAtomic(a, io, path, bytes) catch |e| {
        say(env, "lesto tune: the table could not be written to {s} ({s})", .{ path, @errorName(e) });
        return 2;
    };
    return 0;
}

fn runVerify(a: Allocator, io: std.Io, root: []const u8, mac: kernel_table.Key, idle: u32, plan: transformer.MoeRowsLayoutPlan, sig: tune_signals.Signals, line: *ProgressLine) !tune_sched.VerifyResult {
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{});
    defer dir.close(io);
    var gw = tune_gpu.GpuWorker.init(a, WORKER_SEED);
    defer gw.deinit();
    const cfg: tune_sched.RunConfig = .{ .io = io, .root = dir, .monitor = .thread, .key = mac, .idle_minutes = idle, .mode = .verify, .verify_plan = plan, .reporter = line.reporter() };
    return tune_sched.verify(a, cfg, sig, gw.worker(), &stop_flag);
}

// ---------------------------------------------------------------------------------------------------------------
// Tests

const testing = std.testing;

/// A temporary home: `<tmp>/.mlx-serve/tune` as the tune root (not created), `<tmp>/LaunchAgents`, an allocating writer.
const Harness = struct {
    tmp: testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    aw: std.Io.Writer.Allocating,
    home: []const u8,
    root: []const u8,

    fn init(h: *Harness) !void {
        h.tmp = testing.tmpDir(.{ .iterate = true });
        h.arena = std.heap.ArenaAllocator.init(testing.allocator);
        h.aw = .init(testing.allocator);
        try h.tmp.dir.setPermissions(testing.io, .fromMode(0o700));
        var buf: [MAX_PATH]u8 = undefined;
        const a = h.arena.allocator();
        h.home = try a.dupe(u8, buf[0..try h.tmp.dir.realPath(testing.io, &buf)]);
        h.root = try std.fmt.allocPrint(a, "{s}/.mlx-serve/tune", .{h.home});
    }

    fn deinit(h: *Harness) void {
        h.aw.deinit();
        h.arena.deinit();
        h.tmp.cleanup();
    }

    fn env(h: *Harness) Env {
        return .{
            .engine = .{ .home = h.home, .tune_root = h.root, .table_switch = .unset, .moe_rows_layout_raw = null, .gateup_env = null, .down_env = null },
            .launch_agents_dir = std.fmt.allocPrint(h.arena.allocator(), "{s}/LaunchAgents", .{h.home}) catch unreachable,
            .fake_signals = null,
            .stdout_is_tty = false,
            .launchctl = false,
            .out = &h.aw.writer,
        };
    }

    fn run(h: *Harness, e: Env, args: []const []const u8) u8 {
        h.aw.clearRetainingCapacity();
        return runWith(testing.allocator, testing.io, e, args);
    }

    fn output(h: *Harness) []const u8 {
        return h.aw.written();
    }

    fn path(h: *Harness, name: []const u8) []const u8 {
        return std.fmt.allocPrint(h.arena.allocator(), "{s}/{s}", .{ h.root, name }) catch unreachable;
    }

    fn put(h: *Harness, name: []const u8, data: []const u8) !void {
        try ensureFolder(h.root);
        try kernel_table.writeAtomic(h.arena.allocator(), testing.io, h.path(name), data);
    }
};

fn fileExists(p: []const u8) bool {
    var b: [MAX_PATH]u8 = undefined;
    return exists(cz(&b, p).?);
}

fn modeOf(p: []const u8) !u32 {
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, p).?, .{ .ACCMODE = .RDONLY, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestNoFile;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.TestNoFile;
    return @as(u32, @intCast(st.mode)) & 0o777;
}

fn readAll(a: Allocator, p: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(testing.io, p, a, .limited(1 << 20));
}

fn returnsStopped() tune_sched.SchedResult {
    return .stopped;
}

fn sampleOutcome(gateup: ?transformer.MoeRowsLayoutGateUp, down: ?transformer.MoeRowsLayoutDown, reasons: [2]?ServedReason) tune_sched.Outcome {
    const c = kernel_table.ConfirmStat{ .median_ns = -900, .lo_ns = -1500, .hi_ns = -300, .window = 1 };
    const ev = kernel_table.KernelEvidence{ .ref_ns = 90_000, .pick_ns = 82_000, .confirm = .{ c, c, c, c } };
    return .{
        .plan = .{ .gateup = gateup, .down = down },
        .served_reasons = reasons,
        .evidence = .{ if (gateup != null) ev else null, if (down != null) ev else null },
        .identity = .{ .plans_checked = CANDIDATES, .cases = 10, .calls_compared = CANDIDATES * 10, .excluded = &.{.{ .kernel = "gateup", .cand = "sg=1 ax=0", .reason = "bits differ in case 2" }} },
        .not_measurable = &.{.{ .kernel = "down", .cand = "sps=3 tpt=2 ax=1" }},
        .windows_used = 2,
        .batches_kept = MAX_TIMING_BATCHES,
        .discarded = .{ 1, 0, 0, 0, 0, 0, 2 },
    };
}

const FAKE_KEY = kernel_table.Key{ .gpu_arch = "applegpu_g16s", .gpu_cores = 20, .hw_model = "Mac16,7", .os_build = "25B78", .kernel_hash = "ab12", .mlx_version = "0.31.1" };

test "tune cli: flags" {
    try testing.expectEqual(Action.default, parseFlags(&.{}).ok.action);
    for (ACTION_FLAGS) |f| try testing.expectEqual(f.action, parseFlags(&.{f.flag}).ok.action);
    try testing.expect(parseFlags(&.{ "--show", "--reset" }) == .usage);
    try testing.expect(parseFlags(&.{"--bogus"}) == .usage);
    try testing.expect(parseFlags(&.{ "--check-model", "gemma" }) == .check_model);
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{ "--check-model", "gemma" }));
    try testing.expectEqualStrings(CHECK_MODEL ++ "\n", h.output());
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{ "--show", "--reset" }));
    try testing.expect(std.mem.indexOf(u8, h.output(), "usage: lesto tune") != null);
}

test "tune cli: --idle-minutes takes a whole number of at least 1" {
    const ok = parseFlags(&.{ "--idle-minutes", "7" });
    try testing.expectEqual(Action.default, ok.ok.action);
    try testing.expectEqual(@as(u32, 7), ok.ok.idle_minutes);
    try testing.expectEqual(@as(u32, 9), parseFlags(&.{ "--background", "--idle-minutes", "9" }).ok.idle_minutes);
    for ([_][]const u8{ "0", "-1", "5m", "1.5", "+3", "" }) |v| try testing.expect(parseFlags(&.{ "--idle-minutes", v }) == .usage);
    try testing.expect(parseFlags(&.{"--idle-minutes"}) == .usage);
    try testing.expect(parseFlags(&.{ "--show", "--idle-minutes", "3" }) == .usage);
    try testing.expect(parseFlags(&.{ "--reset", "--idle-minutes", "3" }) == .usage);
    // a usage error touches nothing: no folder is created
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{ "--idle-minutes", "0" }));
    try testing.expect(!fileExists(h.root));
}

test "tune cli: the plist" {
    const t = try plistText(testing.allocator, "/opt/lesto/bin/lesto", 7, "/h/.mlx-serve/tune/agent.log");
    defer testing.allocator.free(t);
    try testing.expect(std.mem.indexOf(u8, t, "<key>Label</key>\n\t<string>com.lestoengine.lesto.tune</string>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "<key>ProgramArguments</key>\n\t<array>\n\t\t<string>/opt/lesto/bin/lesto</string>\n\t\t<string>tune</string>\n\t\t<string>--agent</string>\n\t\t<string>--idle-minutes</string>\n\t\t<string>7</string>\n\t</array>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "<key>RunAtLoad</key>\n\t<true/>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "<key>KeepAlive</key>\n\t<false/>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "<key>ProcessType</key>\n\t<string>Standard</string>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "Background") == null);
    try testing.expect(std.mem.indexOf(u8, t, "<key>StandardOutPath</key>\n\t<string>/h/.mlx-serve/tune/agent.log</string>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "<key>StandardErrorPath</key>\n\t<string>/h/.mlx-serve/tune/agent.log</string>") != null);
    try testing.expectEqual(std.mem.count(u8, t, "<dict>"), std.mem.count(u8, t, "</dict>"));
    try testing.expectEqual(std.mem.count(u8, t, "<array>"), std.mem.count(u8, t, "</array>"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, "<dict>"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, "<array>"));
    const prog = (try plistProgram(testing.allocator, t)).?;
    defer testing.allocator.free(prog);
    try testing.expectEqualStrings("/opt/lesto/bin/lesto", prog);
}

test "tune cli: the plist escapes XML" {
    const t = try plistText(testing.allocator, "/opt/a&b/<x>/lesto", 5, "/h/'q\"/agent.log");
    defer testing.allocator.free(t);
    try testing.expect(std.mem.indexOf(u8, t, "<string>/opt/a&amp;b/&lt;x&gt;/lesto</string>") != null);
    try testing.expect(std.mem.indexOf(u8, t, "/h/&apos;q&quot;/agent.log") != null);
    // every `&` starts an entity
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, t, i, '&')) |at| : (i = at + 1) {
        const rest = t[at..];
        const ok = std.mem.startsWith(u8, rest, "&amp;") or std.mem.startsWith(u8, rest, "&lt;") or std.mem.startsWith(u8, rest, "&gt;") or
            std.mem.startsWith(u8, rest, "&quot;") or std.mem.startsWith(u8, rest, "&apos;");
        try testing.expect(ok);
    }
    const prog = (try plistProgram(testing.allocator, t)).?;
    defer testing.allocator.free(prog);
    try testing.expectEqualStrings("/opt/a&b/<x>/lesto", prog);
}

test "tune cli: --reset removes exactly the two files" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    for ([_][]const u8{ "kernel-plans.json", "progress.jsonl", "tune.log", "agent.log", "lock" }) |n| try h.put(n, "x\n");
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--reset"}));
    try testing.expect(!fileExists(h.path("kernel-plans.json")));
    try testing.expect(!fileExists(h.path("progress.jsonl")));
    for ([_][]const u8{ "tune.log", "agent.log", "lock" }) |n| try testing.expect(fileExists(h.path(n)));
    const out = h.output();
    try testing.expect(std.mem.indexOf(u8, out, try std.fmt.allocPrint(h.arena.allocator(), "[tune] removed {s}\n", .{h.path("kernel-plans.json")})) != null);
    try testing.expect(std.mem.indexOf(u8, out, try std.fmt.allocPrint(h.arena.allocator(), "[tune] removed {s}\n", .{h.path("progress.jsonl")})) != null);
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, out, "[tune] removed "));

    // a custom table path: that file goes, the default one stays, progress.jsonl still goes from the tune root
    try h.put("kernel-plans.json", "x\n");
    try h.put("progress.jsonl", "x\n");
    const custom = try std.fmt.allocPrint(h.arena.allocator(), "{s}/custom.json", .{h.home});
    try kernel_table.writeAtomic(h.arena.allocator(), testing.io, custom, "x\n");
    var e = h.env();
    e.engine.table_switch = .{ .path = custom };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--reset"}));
    try testing.expect(!fileExists(custom));
    try testing.expect(fileExists(h.path("kernel-plans.json")));
    try testing.expect(!fileExists(h.path("progress.jsonl")));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, h.output(), "[tune] removed "));
}

test "tune cli: --reset refuses while a tune holds the lock" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try h.put("kernel-plans.json", "x\n");
    try h.put("progress.jsonl", "x\n");
    const me = std.c.getpid();
    try h.put("lock", try std.fmt.allocPrint(h.arena.allocator(), "{d} 1 2\n", .{me}));
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    const want = try std.fmt.allocPrint(h.arena.allocator(), "[tune] a tune is running (pid {d})\n", .{me});
    try testing.expectEqualStrings(want, h.output());
    try testing.expect(fileExists(h.path("kernel-plans.json")));
    try testing.expect(fileExists(h.path("progress.jsonl")));
}

test "tune cli: --stop signals nothing without a matching start time" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const me = std.c.getpid();
    const real = procStart(me).?;
    try h.put("lock", try std.fmt.allocPrint(h.arena.allocator(), "{d} {d} {d}\n", .{ me, real.sec + 1, real.usec }));
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    // a SIGTERM sent here would run this handler instead of ending the test process
    stop_flag.store(false, .release);
    const saved = installStopHandlers();
    defer restoreStopHandlers(saved);
    const r = stopRunning(testing.io, h.env());
    try testing.expectEqual(StopResult{ .unidentified = me }, r);
    _ = usleep(50_000);
    try testing.expect(!stop_flag.load(.acquire));
    // and through the command: exit 2 (a tune still holds the lock), the holder named, nothing signalled
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--stop"}));
    try testing.expect(std.mem.indexOf(u8, h.output(), "could not be identified") != null);
    _ = usleep(50_000);
    try testing.expect(!stop_flag.load(.acquire));
}

test "tune cli: --stop signals a matching holder" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try ensureFolder(h.root);
    const e = h.env();
    var fds: [2]c_int = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    const pid = fork();
    try testing.expect(pid >= 0);
    if (pid == 0) {
        _ = std.c.close(fds[0]);
        const r = takeLock(testing.io, e) catch std.c._exit(3);
        if (r != .held) std.c._exit(4);
        _ = std.c.write(fds[1], "k", 1);
        while (true) _ = pause();
    }
    _ = std.c.close(fds[1]);
    var reaped = false;
    defer if (!reaped) {
        _ = kill(pid, SIGKILL);
        _ = std.c.waitpid(pid, null, 0);
    };
    var byte: [1]u8 = undefined;
    const n = std.c.read(fds[0], &byte, 1);
    _ = std.c.close(fds[0]);
    try testing.expectEqual(@as(isize, 1), n);
    try testing.expectEqual(StopResult{ .signalled = pid }, stopRunning(testing.io, e));
    var status: c_int = 0;
    var waited: u32 = 0;
    while (waited < 200) : (waited += 1) {
        const w = std.c.waitpid(pid, &status, 1); // WNOHANG
        if (w == pid) {
            reaped = true;
            break;
        }
        _ = usleep(10_000);
    }
    try testing.expect(reaped);
    const st: u32 = @bitCast(status);
    try testing.expect(st & 0x7f != 0 and st & 0x7f != 0x7f); // ended by a signal
    try testing.expectEqual(@as(u32, @intCast(SIGTERM)), st & 0x7f);
    const again = try takeLock(testing.io, e);
    try testing.expect(again == .held);
    again.held.release();
}

test "tune cli: the folder and files are private" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    sched_override = returnsStopped;
    defer sched_override = null;
    _ = h.run(h.env(), &.{});
    try testing.expectEqual(@as(u32, 0o700), try modeOf(h.root));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("lock")));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("tune.log")));
}

test "tune cli: the progress line with and without a terminal" {
    var aw: std.Io.Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    var tty: ProgressLine = .{ .out = &aw.writer, .tty = true };
    tty.update("[tune] waiting: no input for 0m10s of 5m, on AC");
    tty.update("[tune] waiting: no input for 0m15s of 5m, on AC");
    const t = aw.written();
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, "\n"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, t, "\r"));
    try testing.expect(std.mem.startsWith(u8, t, "\r[tune] waiting: no input for 0m10s"));
    tty.end();
    try testing.expect(std.mem.endsWith(u8, aw.written(), "\n"));

    aw.clearRetainingCapacity();
    var plain: ProgressLine = .{ .out = &aw.writer, .tty = false };
    plain.update("[tune] waiting: no input for 0m10s of 5m, on AC");
    plain.update("[tune] waiting: no input for 0m10s of 5m, on AC"); // no change: no new line
    plain.update("[tune] waiting: no input for 0m15s of 5m, on AC");
    plain.end();
    try testing.expectEqualStrings("[tune] waiting: no input for 0m10s of 5m, on AC\n[tune] waiting: no input for 0m15s of 5m, on AC\n", aw.written());
}

test "tune cli: --show reads the table and the progress file" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const mac = try kernel_table.readMacKey(a);
    const t = try buildTable(a, mac, sampleOutcome(.{ .sg = 32, .ax = 1 }, .{ .sps = 1, .tpt = 1, .ax = 0, .body = .array }, .{ null, null }), .{});
    try h.put("kernel-plans.json", try kernel_table.serialize(a, t));
    const header = try std.fmt.allocPrint(a, "{{\"progress\":1,\"key\":{{\"gpu_arch\":\"{s}\",\"gpu_cores\":{d},\"hw_model\":\"{s}\",\"os_build\":\"{s}\",\"kernel_hash\":\"{s}\",\"mlx_version\":\"{s}\"}},\"space\":{d}}}\n", .{ mac.gpu_arch, mac.gpu_cores, mac.hw_model, mac.os_build, mac.kernel_hash, mac.mlx_version, kernel_table.TUNE_PLAN_SPACE_VERSION });
    const rec = "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=1\",\"regime\":\"indep\",\"window\":1,\"kept\":false,\"reason\":\"input\"}\n";
    try h.put("progress.jsonl", try std.mem.concat(a, u8, &.{ header, rec }));

    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--show"}));
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, h.output(), '\n');
    while (it.next()) |l| if (l.len > 0) try lines.append(a, l);
    const ls = lines.items;
    try testing.expect(ls.len > 14);
    try testing.expect(std.mem.startsWith(u8, ls[0], "[tune] table: "));
    try testing.expect(std.mem.indexOf(u8, ls[0], "/.mlx-serve/tune/kernel-plans.json (valid)") != null);
    for (std.enums.values(kernel_table.KeyField), 1..) |f, i| {
        try testing.expect(std.mem.startsWith(u8, ls[i], try std.fmt.allocPrint(a, "[tune] key {s}: matches", .{@tagName(f)})));
    }
    try testing.expect(std.mem.startsWith(u8, ls[7], "[tune] plan gateup: sg=32 ax=1, gain "));
    // a pick on the served triple with the array body is a layout, not "served", and prints its body
    try testing.expect(std.mem.startsWith(u8, ls[8], "[tune] plan down: sps=1 tpt=1 ax=0 body=1, gain "));
    try testing.expect(std.mem.startsWith(u8, ls[9], "[tune] source: gate/up local table, down local table"));
    var at: usize = 10;
    for ([_][]const u8{ "[tune] excluded: gateup sg=1 ax=0 (bits differ in case 2)", "[tune] not measurable: down sps=3 tpt=2 ax=1", "[tune] served: gateup no (a pick), down no (a pick)", "[tune] progress: 0 batches kept, 1 discarded (input 1,", "[tune] LaunchAgent: not installed" }) |want| {
        while (at < ls.len and !std.mem.startsWith(u8, ls[at], want)) at += 1;
        try testing.expect(at < ls.len);
    }
}

test "tune cli: --verify refreshes os_build on the table it read" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const mac = try kernel_table.readMacKey(a);
    var old_key = mac;
    old_key.os_build = "0A000";
    const t = try buildTable(a, old_key, sampleOutcome(null, null, .{ .no_gain_in_screen, .confirm1_not_confirmed }), .{});
    const old_bytes = try kernel_table.serialize(a, t);
    try h.put("kernel-plans.json", old_bytes);

    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--verify"}));
    try testing.expect(std.mem.indexOf(u8, h.output(), "[tune] the table says \"served\" for both kernels; os_build refreshed, no GPU check needed\n") != null);
    const new_bytes = try readAll(a, h.path("kernel-plans.json"));
    const r = kernel_table.readTable(a, testing.io, h.path("kernel-plans.json"), mac);
    defer r.deinit();
    try testing.expect(r == .valid);
    const nt = r.valid.value;
    try testing.expectEqualStrings(mac.os_build, nt.key.os_build);
    try testing.expectEqualStrings(mac.os_build, nt.identity.os_build);
    var back = nt;
    back.key.os_build = "0A000";
    back.identity.os_build = "0A000";
    back.checksum = t.checksum;
    try testing.expectEqualStrings(old_bytes, try kernel_table.serialize(a, back));
    try testing.expect(!std.mem.eql(u8, old_bytes, new_bytes));
    // no other file in the folder but the lock (created 0600): no temporary, nothing else
    var dir = try std.Io.Dir.cwd().openDir(testing.io, h.root, .{ .iterate = true });
    defer dir.close(testing.io);
    var names: u32 = 0;
    var dit = dir.iterate();
    while (try dit.next(testing.io)) |entry| {
        if (std.mem.eql(u8, entry.name, "lock")) continue;
        names += 1;
        try testing.expectEqualStrings("kernel-plans.json", entry.name);
    }
    try testing.expectEqual(@as(u32, 1), names);
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("lock")));

    // case (a): a valid table with "served" for both: nothing to verify, the file untouched
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--verify"}));
    try testing.expect(std.mem.indexOf(u8, h.output(), "[tune] nothing to verify: the served layouts serve\n") != null);
    try testing.expectEqualStrings(new_bytes, try readAll(a, h.path("kernel-plans.json")));
}

test "tune cli: no HOME exits 2" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    var e = h.env();
    e.engine.home = null;
    e.engine.tune_root = null;
    e.engine.table_switch = .unset;
    e.launch_agents_dir = null;
    sched_override = returnsStopped;
    defer sched_override = null;
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{}));
    try testing.expectEqualStrings(NO_HOME ++ "\n", h.output());
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
    try testing.expectEqualStrings(NO_HOME ++ "\n", h.output());
    // a custom table path without a tune root: --reset still needs the folder (progress.jsonl, the lock)
    const custom = try std.fmt.allocPrint(h.arena.allocator(), "{s}/custom.json", .{h.home});
    try kernel_table.writeAtomic(h.arena.allocator(), testing.io, custom, "x\n");
    e.engine.table_switch = .{ .path = custom };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
    try testing.expect(fileExists(custom));
    try testing.expect(!fileExists(h.root));
}

test "tune cli: a stop maps to exit 130" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    sched_override = returnsStopped;
    defer sched_override = null;
    try testing.expectEqual(@as(u8, 130), h.run(h.env(), &.{}));
    try testing.expect(std.mem.indexOf(u8, h.output(), "[tune] stopped; progress kept") != null);
    try testing.expect(!fileExists(h.path("kernel-plans.json")));
}

test "tune cli: the built table is valid" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const t = try buildTable(a, FAKE_KEY, sampleOutcome(.{ .sg = 32, .ax = 1 }, null, .{ null, .below_floor }), .{ .scripted = true, .tuned_at = "2026-11-21T10:00:00Z" });
    try testing.expectEqualStrings("scripted", t.info.signals);
    try testing.expectEqualStrings(build_options.version, t.info.lesto_version);
    try testing.expect(t.identity.all_identical);
    try testing.expect(t.evidence.gateup != null and t.evidence.down == null);
    try h.put("kernel-plans.json", try kernel_table.serialize(a, t));
    const r = kernel_table.readTable(a, testing.io, h.path("kernel-plans.json"), FAKE_KEY);
    defer r.deinit();
    try testing.expect(r == .valid);
    // the current time, in UTC ISO 8601
    const now = try buildTable(a, FAKE_KEY, sampleOutcome(null, null, .{ .no_gain_in_screen, .no_gain_in_screen }), .{});
    try testing.expectEqual(@as(usize, 20), now.info.tuned_at.len);
    try testing.expectEqual(@as(u8, 'T'), now.info.tuned_at[10]);
    try testing.expectEqual(@as(u8, 'Z'), now.info.tuned_at[19]);
}

test "tune cli: every served reason reaches info.served_reasons" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const all = std.enums.values(ServedReason);
    try testing.expectEqual(@as(usize, 10), all.len);
    for (all) |reason| {
        const t = try buildTable(a, FAKE_KEY, sampleOutcome(null, .{ .sps = 6, .tpt = 1, .ax = 1 }, .{ reason, null }), .{});
        try testing.expectEqualStrings(@tagName(reason), t.info.served_reasons.gateup.?);
        try testing.expectEqual(@as(?[]const u8, null), t.info.served_reasons.down);
        try h.put("kernel-plans.json", try kernel_table.serialize(a, t));
        const r = kernel_table.readTable(a, testing.io, h.path("kernel-plans.json"), FAKE_KEY);
        defer r.deinit();
        try testing.expect(r == .valid);
        try testing.expectEqualStrings(@tagName(reason), r.valid.value.info.served_reasons.gateup.?);
        try testing.expectEqual(@as(?[]const u8, null), r.valid.value.info.served_reasons.down);
    }
}

fn chmodPath(p: []const u8, mode: u32) !void {
    var b: [MAX_PATH]u8 = undefined;
    if (std.c.chmod(cz(&b, p).?, @as(std.c.mode_t, @intCast(mode))) != 0) return error.TestChmodFailed;
}

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

test "tune cli: --idle-minutes stops at one week" {
    try testing.expectEqual(MAX_IDLE_MINUTES, parseFlags(&.{ "--idle-minutes", "10080" }).ok.idle_minutes);
    try testing.expectEqual(@as(u32, 1), parseFlags(&.{ "--idle-minutes", "1" }).ok.idle_minutes);
    for ([_][]const u8{ "10081", "307445735", "999999999", "4294967295" }) |v| try testing.expect(parseFlags(&.{ "--idle-minutes", v }) == .usage);
    // the threshold saturates: a minute count past any real one never reads as idle, a small one still does
    var sc = try tune_signals.ScriptedSignals.parse(testing.allocator, "{\"at_ms\":0,\"hid_idle_s\":600}\n", .manual);
    defer sc.deinit();
    try testing.expectEqual(tune_signals.Cond.fails, tune_signals.checkOpen(sc.signals(), std.math.maxInt(u32)).input);
    try testing.expectEqual(tune_signals.Cond.holds, tune_signals.checkOpen(sc.signals(), 5).input);
}

test "tune cli: the plist carries a carriage return and refuses other control characters" {
    const t = try plistText(testing.allocator, "/opt/a\rb/lesto", 5, "/h/a\tb/agent.log");
    defer testing.allocator.free(t);
    try testing.expect(contains(t, "<string>/opt/a&#13;b/lesto</string>"));
    try testing.expect(std.mem.indexOfScalar(u8, t, '\r') == null);
    try testing.expect(contains(t, "<string>/h/a\tb/agent.log</string>")); // a tab is XML text as it is
    const prog = (try plistProgram(testing.allocator, t)).?;
    defer testing.allocator.free(prog);
    try testing.expectEqualStrings("/opt/a\rb/lesto", prog);
    for ([_][]const u8{ "/opt/a\x01b/lesto", "/opt/\x1f/lesto", "/opt/\x00/lesto" }) |bad| {
        try testing.expectError(error.NotXmlText, plistText(testing.allocator, bad, 5, "/h/agent.log"));
    }
    try testing.expectError(error.NotXmlText, plistText(testing.allocator, "/opt/lesto", 5, "/h/\x0b/agent.log"));
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{ "--print-plist-for-test", "/opt/\x07/lesto" }));
    try testing.expect(contains(h.output(), "holds a byte or character a plist cannot carry"));
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), "<string>/opt/lesto</string>"));
}

test "tune cli: an existing tune folder and its files are made private, another user's are refused" {
    try testing.expectEqual(PrivateVerdict.ok, privateVerdict(501, .{ .uid = 501, .mode = 0o040700, .nlink = 3 }, 0o700));
    try testing.expectEqual(PrivateVerdict.tighten, privateVerdict(501, .{ .uid = 501, .mode = 0o040777, .nlink = 3 }, 0o700));
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 502, .mode = 0o040700, .nlink = 3 }, 0o700));
    try testing.expectEqual(PrivateVerdict.ok, privateVerdict(501, .{ .uid = 501, .mode = 0o100600, .nlink = 1 }, 0o600));
    try testing.expectEqual(PrivateVerdict.tighten, privateVerdict(501, .{ .uid = 501, .mode = 0o100644, .nlink = 1 }, 0o600));
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 0, .mode = 0o100600, .nlink = 1 }, 0o600));

    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try h.put("lock", "x\n");
    try h.put("tune.log", "x\n");
    try h.put("progress.jsonl", "x\n");
    try chmodPath(h.root, 0o777);
    try chmodPath(h.path("lock"), 0o666);
    try chmodPath(h.path("tune.log"), 0o644);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--reset"}));
    try testing.expectEqual(@as(u32, 0o700), try modeOf(h.root));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("lock")));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("tune.log")));
    // a link in place of a tuner file is not used: the command stops before the lock, naming the folder
    var lb: [MAX_PATH]u8 = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.symlink("/etc/hosts", cz(&lb, h.path("progress.jsonl")).?));
    try h.put("kernel-plans.json", "x\n");
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expect(contains(h.output(), "is not usable"));
    try testing.expect(fileExists(h.path("kernel-plans.json")));
}

fn flockFailsIo(_: c_int, _: c_int) c_int {
    std.c._errno().* = @intFromEnum(std.c.E.IO);
    return -1;
}

fn flockBusy(_: c_int, _: c_int) c_int {
    std.c._errno().* = @intFromEnum(std.c.E.AGAIN);
    return -1;
}

test "tune cli: a lock that fails for a reason other than contention is an error" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try ensureFolder(h.root);
    defer flock_override = null;
    flock_override = flockFailsIo;
    try testing.expectError(error.LockFailed, takeLock(testing.io, h.env()));
    try testing.expectEqual(std.c.E.IO, last_errno);
    try testing.expectEqual(StopResult{ .failed = .IO }, stopRunning(testing.io, h.env()));
    // --agent must not take an unusable lock for a running tune: exit 2, naming the errno
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--agent"}));
    try testing.expect(contains(h.output(), "the lock could not be taken (LockFailed, errno IO)"));
    // contention is still contention: the foreground command reports it and exits 2 (the agent waits instead, tested
    // in "the agent waits for a busy lock")
    flock_override = flockBusy;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{}));
    try testing.expect(contains(h.output(), "[tune] a tune is running (pid "));
}

test "tune cli: --stop reports a lock it cannot open" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try ensureFolder(h.root);
    try testing.expectEqual(StopResult.not_running, stopRunning(testing.io, h.env()));
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] no tune is running"));
    try h.put("lock", "1 2 3\n");
    try chmodPath(h.path("lock"), 0o000);
    defer chmodPath(h.path("lock"), 0o600) catch {};
    try testing.expectEqual(StopResult{ .failed = .ACCES }, stopRunning(testing.io, h.env()));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "could not be checked (errno ACCES)"));
}

test "tune cli: --stop reports a LaunchAgent it cannot delete" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const e = h.env();
    const agents = e.launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(h.arena.allocator(), "{s}/{s}", .{ agents, PLIST_NAME });
    try kernel_table.writeAtomic(h.arena.allocator(), testing.io, plist, "x\n");
    try chmodPath(agents, 0o500);
    defer chmodPath(agents, 0o700) catch {};
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--stop"}));
    try testing.expect(contains(h.output(), "could not be removed (CannotRemove, errno ACCES)"));
    try testing.expect(contains(h.output(), "[tune] no tune is running")); // the rest of the stop still ran
    try testing.expect(fileExists(plist));
    try chmodPath(agents, 0o700);
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] LaunchAgent removed: "));
    try testing.expect(!fileExists(plist));
}

test "tune cli: the bootout verdict" {
    try testing.expectEqual(BootoutVerdict.removed, bootoutVerdict(.{ .ran = true, .code = 0, .text = "" }));
    try testing.expectEqual(BootoutVerdict.not_loaded, bootoutVerdict(.{ .ran = true, .code = 3, .text = "Boot-out failed: 3: No such process\n" }));
    try testing.expectEqual(BootoutVerdict.not_loaded, bootoutVerdict(.{ .ran = true, .code = 113, .text = "Could not find specified service\n" }));
    try testing.expectEqual(BootoutVerdict.failed, bootoutVerdict(.{ .ran = true, .code = 5, .text = "Boot-out failed: 5: Input/output error\n" }));
    try testing.expectEqual(BootoutVerdict.failed, bootoutVerdict(.{ .ran = true, .code = 1, .text = "" }));
    try testing.expectEqual(BootoutVerdict.failed, bootoutVerdict(.{ .ran = true, .code = null, .text = "" }));
    try testing.expectEqual(BootoutVerdict.failed, bootoutVerdict(.{ .ran = false, .code = null, .text = "FileNotFound" }));
}

test "tune cli: --background carries a custom table path and refuses the off switch" {
    const with = try plistTextWith(testing.allocator, "/opt/lesto", 5, "/h/agent.log", "/x/a&b.json");
    defer testing.allocator.free(with);
    try testing.expect(contains(with, "\t<key>EnvironmentVariables</key>\n\t<dict>\n\t\t<key>MLX_SERVE_KERNEL_TABLE</key>\n\t\t<string>/x/a&amp;b.json</string>\n\t</dict>\n</dict>\n</plist>\n"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, with, "<dict>"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, with, "</dict>"));
    const without = try plistText(testing.allocator, "/opt/lesto", 5, "/h/agent.log");
    defer testing.allocator.free(without);
    try testing.expect(!contains(without, "EnvironmentVariables"));

    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const plist = try std.fmt.allocPrint(h.arena.allocator(), "{s}/LaunchAgents/{s}", .{ h.home, PLIST_NAME });
    for ([_]kernel_table.TableSwitch{ .off, .empty }) |sw| {
        var e = h.env();
        e.engine.table_switch = sw;
        try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--background"}));
        try testing.expect(contains(h.output(), "--background refused: MLX_SERVE_KERNEL_TABLE is off"));
        try testing.expect(!fileExists(plist));
        try testing.expect(!fileExists(h.root));
    }
    const custom = try std.fmt.allocPrint(h.arena.allocator(), "{s}/custom.json", .{h.home});
    var e = h.env();
    e.engine.table_switch = .{ .path = custom };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--background"}));
    const text = try readAll(h.arena.allocator(), plist);
    try testing.expect(contains(text, try std.fmt.allocPrint(h.arena.allocator(), "<key>MLX_SERVE_KERNEL_TABLE</key>\n\t\t<string>{s}</string>", .{custom})));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(plist));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("agent.log")));
}

test "tune cli: --show reports progress it cannot read" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "[tune] progress: no unfinished run\n"));
    try h.put("progress.jsonl", "x\n");
    try chmodPath(h.path("progress.jsonl"), 0o000);
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "[tune] progress: unreadable ("));
    try testing.expect(contains(h.output(), "[tune] LaunchAgent: ")); // the rest of the report still prints
    try chmodPath(h.path("progress.jsonl"), 0o600);
    try chmodPath(h.root, 0o000);
    defer chmodPath(h.root, 0o700) catch {};
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "[tune] progress: unreadable ("));
    try testing.expect(!contains(h.output(), "no unfinished run"));
}

test "tune cli: --show tells an unreadable or oversized LaunchAgent from a missing one" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    // over the old 64 KiB read limit, still a plist: read and shown
    const base = try plistText(a, "/opt/no-such-dir/lesto", 5, "/h/agent.log");
    const padded = try a.alloc(u8, base.len + 100 * 1024);
    @memcpy(padded[0..base.len], base);
    @memset(padded[base.len..], ' ');
    try kernel_table.writeAtomic(a, testing.io, plist, padded);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "program /opt/no-such-dir/lesto is missing"));
    // unreadable: installed, never "not installed"
    try chmodPath(plist, 0o000);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "but unreadable ("));
    try testing.expect(!contains(h.output(), "LaunchAgent: not installed"));
    try chmodPath(plist, 0o600);
    // oversized
    const huge = try a.alloc(u8, PLIST_READ_MAX);
    @memset(huge, ' ');
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, plist).?, .{ .ACCMODE = .WRONLY, .TRUNC = true }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    try testing.expectEqual(@as(isize, @intCast(huge.len)), std.c.write(fd, huge.ptr, huge.len));
    _ = std.c.close(fd);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "but oversized"));
    try testing.expect(!contains(h.output(), "LaunchAgent: not installed"));
}

test "tune cli: --verify takes the lock before it reads the table" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const mac = try kernel_table.readMacKey(a);
    var old_key = mac;
    old_key.os_build = "0A000";
    const t = try buildTable(a, old_key, sampleOutcome(null, null, .{ .no_gain_in_screen, .no_gain_in_screen }), .{});
    const old_bytes = try kernel_table.serialize(a, t);
    try h.put("kernel-plans.json", old_bytes);
    try h.put("lock", try std.fmt.allocPrint(a, "{d} 1 2\n", .{std.c.getpid()}));
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--verify"}));
    try testing.expect(contains(h.output(), "[tune] a tune is running (pid "));
    try testing.expect(!contains(h.output(), "os_build refreshed"));
    try testing.expectEqualStrings(old_bytes, try readAll(a, h.path("kernel-plans.json")));
    _ = flock(fd, LOCK_UN);
    _ = std.c.close(fd);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--verify"}));
    try testing.expect(contains(h.output(), "os_build refreshed"));
    try testing.expect(!std.mem.eql(u8, old_bytes, try readAll(a, h.path("kernel-plans.json"))));
}

/// A child that takes the lock with its real PID and start time, then waits; the parent reads one byte when it holds it.
const LockChild = struct {
    pid: c_int,
    reaped: bool = false,

    fn start(e: Env) !LockChild {
        var fds: [2]c_int = undefined;
        if (std.c.pipe(&fds) != 0) return error.TestPipe;
        const pid = fork();
        if (pid < 0) return error.TestFork;
        if (pid == 0) {
            _ = std.c.close(fds[0]);
            const r = takeLock(testing.io, e) catch std.c._exit(3);
            if (r != .held) std.c._exit(4);
            _ = std.c.write(fds[1], "k", 1);
            while (true) _ = pause();
        }
        _ = std.c.close(fds[1]);
        var byte: [1]u8 = undefined;
        const n = std.c.read(fds[0], &byte, 1);
        _ = std.c.close(fds[0]);
        var c: LockChild = .{ .pid = pid };
        if (n != 1) {
            c.stop();
            return error.TestChildNoLock;
        }
        return c;
    }

    fn alive(self: *LockChild) bool {
        if (self.reaped) return false;
        var status: c_int = 0;
        if (std.c.waitpid(self.pid, &status, 1) == self.pid) self.reaped = true;
        return !self.reaped;
    }

    /// Waits up to 2 s for the child to end; true when SIGTERM ended it.
    fn endedByTerm(self: *LockChild) bool {
        var status: c_int = 0;
        var waited: u32 = 0;
        while (waited < 200) : (waited += 1) {
            if (std.c.waitpid(self.pid, &status, 1) == self.pid) {
                self.reaped = true;
                const st: u32 = @bitCast(status);
                return st & 0x7f == @as(u32, @intCast(SIGTERM));
            }
            _ = usleep(10_000);
        }
        return false;
    }

    fn stop(self: *LockChild) void {
        if (self.reaped) return;
        _ = kill(self.pid, SIGKILL);
        _ = std.c.waitpid(self.pid, null, 0);
        self.reaped = true;
    }
};

test "tune cli: --stop acts only on a private lock file" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try ensureFolder(h.root);
    const e = h.env();
    var child = try LockChild.start(e);
    defer child.stop();
    // writable by others: its record could name any process, so nothing is signalled
    try chmodPath(h.path("lock"), 0o666);
    const r1 = stopRunning(testing.io, e);
    try testing.expect(r1 == .untrusted);
    try testing.expectEqualStrings("group or others can write it", r1.untrusted);
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--stop"}));
    try testing.expect(contains(h.output(), "the lock file is not trusted (group or others can write it)"));
    try chmodPath(h.path("lock"), 0o600);
    // a second hard link: not trusted either
    const extra = try std.fmt.allocPrint(h.arena.allocator(), "{s}/lock-link", .{h.home});
    var b1: [MAX_PATH]u8 = undefined;
    var b2: [MAX_PATH]u8 = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.link(cz(&b1, h.path("lock")).?, cz(&b2, extra).?));
    try testing.expectEqualStrings("it has another hard link", stopRunning(testing.io, e).untrusted);
    _ = std.c.unlink(cz(&b2, extra).?);
    _ = usleep(50_000);
    try testing.expect(child.alive());
    // the private lock of the identified holder: SIGTERM, exit 0
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] sent SIGTERM to the tune (pid "));
    try testing.expect(child.endedByTerm());
}

test "tune cli: a hard-linked tuner file is never written through" {
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 501, .mode = 0o100600, .nlink = 2 }, 0o600));
    try testing.expectEqual(PrivateVerdict.ok, privateVerdict(501, .{ .uid = 501, .mode = 0o040700, .nlink = 5 }, 0o700));
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    try ensureFolder(h.root);
    const victim = try std.fmt.allocPrint(a, "{s}/victim.txt", .{h.home});
    try kernel_table.writeAtomic(a, testing.io, victim, "keep me\n");
    var b1: [MAX_PATH]u8 = undefined;
    var b2: [MAX_PATH]u8 = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.link(cz(&b1, victim).?, cz(&b2, h.path("lock")).?));
    try testing.expectError(error.FileHardLinked, takeLock(testing.io, h.env()));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expect(contains(h.output(), "is not usable (FileHardLinked"));
    try testing.expectEqualStrings("keep me\n", try readAll(a, victim));
    // the same through tune.log
    _ = std.c.unlink(cz(&b2, h.path("lock")).?);
    try testing.expectEqual(@as(c_int, 0), std.c.link(cz(&b1, victim).?, cz(&b2, h.path("tune.log")).?));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expectEqualStrings("keep me\n", try readAll(a, victim));
    _ = std.c.unlink(cz(&b2, h.path("tune.log")).?);
    // one link: the lock is taken
    const r = try takeLock(testing.io, h.env());
    try testing.expect(r == .held);
    r.held.release();
    try testing.expectEqualStrings("keep me\n", try readAll(a, victim));
}

fn killFailsPerm(_: c_int, _: c_int) c_int {
    std.c._errno().* = @intFromEnum(std.c.E.PERM);
    return -1;
}

test "tune cli: --stop exits 2 unless no tune runs or the tune was signalled" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const me = std.c.getpid();
    const real = procStart(me).?;
    // a held lock with no record yet
    try h.put("lock", "");
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    try testing.expectEqual(StopResult{ .unidentified = 0 }, stopRunning(testing.io, h.env()));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "its record names no process yet"));
    // an identified holder whose SIGTERM fails (a seam stands in for kill, so nothing is sent)
    const rec = try std.fmt.allocPrint(h.arena.allocator(), "{d} {d} {d}\n", .{ me, real.sec, real.usec });
    try testing.expectEqual(@as(isize, @intCast(rec.len)), std.c.pwrite(fd, rec.ptr, rec.len, 0));
    kill_override = killFailsPerm;
    defer kill_override = null;
    try testing.expectEqual(StopResult{ .signal_failed = .{ .pid = me, .err = .PERM } }, stopRunning(testing.io, h.env()));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "failed (errno PERM)"));
    // the complement: the lock free, no tune runs, exit 0
    _ = flock(fd, LOCK_UN);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] no tune is running"));
}

test "tune cli: a relative custom table path reaches the plist absolute" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    var cb: [MAX_PATH]u8 = undefined;
    const cwd = std.mem.span(realpath(".", &cb).?);
    var e = h.env();
    e.engine.table_switch = .{ .path = "plans-rel.json" };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), try std.fmt.allocPrint(a, "<key>MLX_SERVE_KERNEL_TABLE</key>\n\t\t<string>{s}/plans-rel.json</string>", .{cwd})));
    // an absolute path passes as it is
    const abs = try std.fmt.allocPrint(a, "{s}/plans.json", .{h.home});
    e.engine.table_switch = .{ .path = abs };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), try std.fmt.allocPrint(a, "<string>{s}</string>", .{abs})));
    // a relative path whose folder does not exist cannot be made absolute: exit 2, nothing printed as a plist
    e.engine.table_switch = .{ .path = "no-such-dir-lesto-tune-test/x.json" };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), "cannot be made absolute for the LaunchAgent"));
    try testing.expect(!contains(h.output(), "<plist"));
}

var fake_bootstrap_code: u8 = 0;
var fake_calls: [8][]const u8 = undefined;
var fake_n: usize = 0;
var fake_staged_path: []const u8 = "";
var fake_staged_seen: bool = false;

fn fakeLaunchctl(argv: []const []const u8) LaunchctlResult {
    if (fake_n < fake_calls.len) {
        fake_calls[fake_n] = argv[1];
        fake_n += 1;
    }
    if (std.mem.eql(u8, argv[1], "bootout")) {
        fake_staged_seen = fileExists(fake_staged_path);
        return .{ .ran = true, .code = 0, .text = "" };
    }
    return .{ .ran = true, .code = fake_bootstrap_code, .text = if (fake_bootstrap_code == 0) "" else "Bootstrap failed: 5: Input/output error" };
}

fn fakeCalls() []const []const u8 {
    return fake_calls[0..fake_n];
}

test "tune cli: a failed bootstrap leaves the previous LaunchAgent" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    fake_staged_path = try std.fmt.allocPrint(a, "{s}.staged-{d}", .{ plist, std.c.getpid() });
    launchctl_override = fakeLaunchctl;
    defer launchctl_override = null;

    // a previous plist and a failing bootstrap: the previous one is back, loaded again, nothing staged is left
    try kernel_table.writeAtomic(a, testing.io, plist, "OLD\n");
    fake_n = 0;
    fake_bootstrap_code = 5;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    try testing.expectEqualStrings("OLD\n", try readAll(a, plist));
    try testing.expect(!fileExists(fake_staged_path));
    try testing.expect(fake_staged_seen); // the old agent was booted out only once the new plist was staged
    try testing.expectEqual(@as(usize, 3), fake_n);
    try testing.expectEqualStrings("bootout", fakeCalls()[0]);
    try testing.expectEqualStrings("bootstrap", fakeCalls()[1]);
    try testing.expectEqualStrings("bootstrap", fakeCalls()[2]);
    try testing.expect(contains(h.output(), "launchctl bootstrap failed: Bootstrap failed: 5"));
    try testing.expect(contains(h.output(), "the previous LaunchAgent was put back at "));

    // no previous plist and a failing bootstrap: none is left installed
    _ = try removeFile(plist);
    fake_n = 0;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    try testing.expect(!fileExists(plist));
    try testing.expect(!fileExists(fake_staged_path));
    try testing.expectEqual(@as(usize, 1), fake_n);
    try testing.expect(contains(h.output(), "the new LaunchAgent was removed; none is installed, as before"));

    // the complement: bootstrap succeeds, the new plist is in place
    try kernel_table.writeAtomic(a, testing.io, plist, "OLD\n");
    fake_n = 0;
    fake_bootstrap_code = 0;
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--background"}));
    try testing.expect(contains(try readAll(a, plist), "<key>ProgramArguments</key>"));
    try testing.expect(!fileExists(fake_staged_path));
    try testing.expectEqual(@as(usize, 2), fake_n);
}

test "tune cli: the gain survives extreme table values" {
    const c = kernel_table.ConfirmStat{ .median_ns = 0, .lo_ns = 0, .hi_ns = 0, .window = 1 };
    const extreme = kernel_table.KernelEvidence{ .ref_ns = 1, .pick_ns = std.math.minInt(i64), .confirm = .{ c, c, c, c } };
    const g = gainPct(extreme);
    try testing.expect(std.math.isFinite(g) and g > 0);
    const wide = kernel_table.KernelEvidence{ .ref_ns = std.math.maxInt(i64), .pick_ns = std.math.minInt(i64), .confirm = .{ c, c, c, c } };
    try testing.expect(std.math.isFinite(gainPct(wide)));
    const normal = kernel_table.KernelEvidence{ .ref_ns = 90_000, .pick_ns = 82_000, .confirm = .{ c, c, c, c } };
    try testing.expectApproxEqAbs(@as(f64, 8.888_888), gainPct(normal), 0.001);
}

test "tune cli: the plist encoder takes only legal XML characters" {
    const bad = [_][]const u8{ "/a/\xff/l", "/a/\x80/l", "/a/\xc3", "/a/\xc0\xaf/l", "/a/\xed\xa0\x80/l", "/a/\xef\xbf\xbf/l", "/a/\xef\xbf\xbe/l", "/a/\xf4\x90\x80\x80/l" };
    for (bad) |p| try testing.expectError(error.NotXmlText, plistText(testing.allocator, p, 5, "/h/agent.log"));
    const good = [_][]const u8{ "/a/caf\xc3\xa9/l", "/a/\xf0\x9f\x98\x80/l", "/a/\xef\xbf\xbd/l", "/a/\xc2\x85/l" };
    for (good) |p| {
        const t = try plistText(testing.allocator, p, 5, "/h/agent.log");
        defer testing.allocator.free(t);
        try testing.expect(contains(t, p));
    }
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    var e = h.env();
    e.engine.table_switch = .{ .path = "/x/\xef\xbf\xbf.json" };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), "invalid UTF-8, U+FFFE or U+FFFF"));
}

test "tune cli: a failed write of the output fails the command" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    var small: [32]u8 = undefined;
    var fw = std.Io.Writer.fixed(&small);
    var e = h.env();
    e.out = &fw;
    try testing.expectEqual(@as(u8, 2), runWith(testing.allocator, testing.io, e, &.{ "--print-plist-for-test", "/opt/lesto" }));
    fw = std.Io.Writer.fixed(&small);
    try testing.expectEqual(@as(u8, 2), runWith(testing.allocator, testing.io, e, &.{"--reset"}));
    // the complement: an output with room
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--reset"}));
}

/// Builds a table from an outcome and a key whose lists and strings live in this frame only.
fn buildFromFrame(a: Allocator) !kernel_table.Table {
    var reason_buf: [32]u8 = undefined;
    const reason = "bits differ in case 2";
    @memcpy(reason_buf[0..reason.len], reason);
    var cand_buf: [16]u8 = undefined;
    @memcpy(cand_buf[0..9], "sg=1 ax=0");
    var exc: [1]kernel_table.Excluded = undefined;
    exc[0] = .{ .kernel = "gateup", .cand = cand_buf[0..9], .reason = reason_buf[0..reason.len] };
    var os_buf: [8]u8 = undefined;
    @memcpy(os_buf[0..5], "25B78");
    var key = FAKE_KEY;
    key.os_build = os_buf[0..5];
    var o = sampleOutcome(.{ .sg = 32, .ax = 1 }, null, .{ null, .below_floor });
    o.identity.excluded = &exc;
    return buildTable(a, key, o, .{ .tuned_at = "2026-11-21T10:00:00Z" });
}

fn clobberStack() u8 {
    var junk: [8192]u8 = undefined;
    @memset(&junk, 0xAA);
    std.mem.doNotOptimizeAway(&junk);
    return junk[777];
}

test "tune cli: the built table owns everything it references" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const t = try buildFromFrame(a);
    try testing.expectEqual(@as(u8, 0xAA), clobberStack()); // the frame buildFromFrame used is overwritten
    const bytes = try kernel_table.serialize(a, t);
    try h.put("kernel-plans.json", bytes);
    var key = FAKE_KEY;
    key.os_build = "25B78";
    const r = kernel_table.readTable(a, testing.io, h.path("kernel-plans.json"), key);
    defer r.deinit();
    try testing.expect(r == .valid);
    try testing.expectEqualStrings("bits differ in case 2", r.valid.value.identity.excluded[0].reason);
    try testing.expectEqualStrings("sg=1 ax=0", r.valid.value.identity.excluded[0].cand);
    try testing.expectEqualStrings("25B78", r.valid.value.key.os_build);
    try testing.expectEqual(@as(u32, 1), r.valid.value.info.batches_discarded[0].n);
    try testing.expectEqualStrings("gate", r.valid.value.info.batches_discarded[6].reason);
}

test "tune cli: a lock that was writable by others loses its record" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    try h.put("lock", "4242 1 2\n");
    try chmodPath(h.path("lock"), 0o666);
    try ensureTuneFolder(h.root);
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("lock")));
    try testing.expectEqualStrings("", try readAll(a, h.path("lock")));
    // a lock that was private keeps its record
    try h.put("lock", "4242 1 2\n");
    try chmodPath(h.path("lock"), 0o600);
    try ensureTuneFolder(h.root);
    try testing.expectEqualStrings("4242 1 2\n", try readAll(a, h.path("lock")));
    // and through takeLock alone: a wider lock someone holds is busy with no PID read from it, and is left wide (so
    // `--stop` never trusts it) rather than tightened while that holder's descriptor can still write it
    try chmodPath(h.path("lock"), 0o644);
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    const r = try takeLock(testing.io, h.env());
    try testing.expectEqual(@as(i32, 0), r.busy.pid);
    try testing.expectEqualStrings("4242 1 2\n", try readAll(a, h.path("lock")));
    try testing.expectEqual(@as(u32, 0o644), try modeOf(h.path("lock")));
}

fn runChmod(args: []const []const u8) !void {
    const r = try std.process.run(testing.allocator, testing.io, .{ .argv = args });
    defer testing.allocator.free(r.stdout);
    defer testing.allocator.free(r.stderr);
    switch (r.term) {
        .exited => |c| if (c != 0) return error.TestChmodFailed,
        else => return error.TestChmodFailed,
    }
}

test "tune cli: a tuner file with an access control list is not private" {
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 501, .mode = 0o100600, .nlink = 1, .has_acl = .present }, 0o600));
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 501, .mode = 0o040700, .nlink = 3, .has_acl = .present }, 0o700));
    try testing.expectEqual(PrivateVerdict.ok, privateVerdict(501, .{ .uid = 501, .mode = 0o100600, .nlink = 1, .has_acl = .none }, 0o600));
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try h.put("tune.log", "x\n");
    try h.put("lock", "x\n");
    // a real ACL entry on the M4's file system
    try runChmod(&.{ "/bin/chmod", "+a", "everyone allow write", h.path("tune.log") });
    try testing.expectError(error.FileHasAcl, ensureTuneFolder(h.root));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expect(contains(h.output(), "is not usable (FileHasAcl"));
    try runChmod(&.{ "/bin/chmod", "-N", h.path("tune.log") });
    try ensureTuneFolder(h.root);
    // the lock: --stop does not act on its record
    try runChmod(&.{ "/bin/chmod", "+a", "everyone allow write", h.path("lock") });
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    const r = stopRunning(testing.io, h.env());
    try testing.expect(r == .untrusted);
    try testing.expectEqualStrings("it has an access control list", r.untrusted);
    try runChmod(&.{ "/bin/chmod", "-N", h.path("lock") });
    try testing.expectEqual(StopResult{ .unidentified = 0 }, stopRunning(testing.io, h.env()));
}

var busy_left: u32 = 0;
var ex_calls: u32 = 0;
var stop_on_call: u32 = 0;
var sched_calls: u32 = 0;

fn flockScripted(fd: c_int, op: c_int) c_int {
    if (op & LOCK_EX == 0) return flock(fd, op);
    ex_calls += 1;
    if (stop_on_call != 0 and ex_calls >= stop_on_call) stop_flag.store(true, .release);
    if (busy_left > 0) {
        busy_left -= 1;
        std.c._errno().* = @intFromEnum(std.c.E.AGAIN);
        return -1;
    }
    return flock(fd, op);
}

fn schedFinishes() tune_sched.SchedResult {
    sched_calls += 1;
    return .{ .finished = sampleOutcome(.{ .sg = 32, .ax = 1 }, null, .{ null, .below_floor }) };
}

test "tune cli: the agent waits for a busy lock" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const saved_poll = agent_lock_poll_ns;
    agent_lock_poll_ns = std.time.ns_per_ms;
    defer agent_lock_poll_ns = saved_poll;
    flock_override = flockScripted;
    defer flock_override = null;
    sched_override = schedFinishes;
    defer sched_override = null;
    // busy twice, then free: the agent waits, then tunes once and writes the table
    busy_left = 2;
    ex_calls = 0;
    stop_on_call = 0;
    sched_calls = 0;
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--agent"}));
    try testing.expectEqual(@as(u32, 3), ex_calls);
    try testing.expectEqual(@as(u32, 1), sched_calls);
    try testing.expect(contains(h.output(), "the agent waits for it, checking the lock every 0 s"));
    try testing.expect(fileExists(h.path("kernel-plans.json")));
    // busy until stopped: only the stop ends it, with nothing run
    _ = try removeFile(h.path("kernel-plans.json"));
    busy_left = 1_000_000;
    ex_calls = 0;
    stop_on_call = 3;
    sched_calls = 0;
    try testing.expectEqual(@as(u8, 130), h.run(h.env(), &.{"--agent"}));
    try testing.expectEqual(@as(u32, 0), sched_calls);
    try testing.expect(contains(h.output(), "stopped while waiting for the lock"));
    // the foreground command still reports a busy lock at once
    busy_left = 1;
    ex_calls = 0;
    stop_on_call = 0;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{}));
    try testing.expectEqual(@as(u32, 1), ex_calls);
    busy_left = 0;
}

test "tune cli: installs are serialized" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const plist = try std.fmt.allocPrint(a, "{s}/LaunchAgents/{s}", .{ h.home, PLIST_NAME });
    try h.put("install.lock", "");
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("install.lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    try testing.expect(contains(h.output(), "[tune] another install is running"));
    try testing.expect(!fileExists(plist));
    _ = flock(fd, LOCK_UN);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--background"}));
    try testing.expect(fileExists(plist));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("install.lock")));
}

test "tune cli: an allocation failure never leaves the old agent unloaded" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    fake_staged_path = try std.fmt.allocPrint(a, "{s}.staged-{d}", .{ plist, std.c.getpid() });
    launchctl_override = fakeLaunchctl;
    defer launchctl_override = null;
    fake_bootstrap_code = 0;
    var failed_runs: u32 = 0;
    var i: usize = 0;
    var done = false;
    while (i < 400) : (i += 1) {
        try kernel_table.writeAtomic(a, testing.io, plist, "OLD\n");
        fake_n = 0;
        var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = i });
        h.aw.clearRetainingCapacity();
        const rc = runWith(fa.allocator(), testing.io, h.env(), &.{"--background"});
        if (rc == 0) {
            done = true;
            break;
        }
        failed_runs += 1;
        // the old plist is as it was and the old agent was never booted out
        try testing.expectEqualStrings("OLD\n", try readAll(a, plist));
        for (fakeCalls()) |c| try testing.expect(!std.mem.eql(u8, c, "bootout"));
        var dir = try std.Io.Dir.cwd().openDir(testing.io, agents, .{ .iterate = true });
        defer dir.close(testing.io);
        var it = dir.iterate();
        while (try it.next(testing.io)) |entry| try testing.expectEqualStrings(PLIST_NAME, entry.name);
    }
    try testing.expect(done);
    try testing.expect(failed_runs > 0);
    try testing.expect(contains(try readAll(a, plist), "<key>ProgramArguments</key>"));
}

test "tune cli: --reset never deletes a tuner file named as the table" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    try h.put("progress.jsonl", "x\n");
    try h.put("tune.log", "x\n");
    var e = h.env();
    for ([_][]const u8{ "lock", "tune.log", "install.lock" }) |name| {
        e.engine.table_switch = .{ .path = h.path(name) };
        try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
        try testing.expect(contains(h.output(), try std.fmt.allocPrint(a, "names the tuner's own {s};", .{name})));
        try testing.expect(fileExists(h.path("progress.jsonl")));
        try testing.expect(fileExists(h.path("tune.log")));
    }
    e.engine.table_switch = .{ .path = h.root };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
    try testing.expect(contains(h.output(), "names the tuner's own tune folder"));
    // a link to the lock is the lock
    const alias = try std.fmt.allocPrint(a, "{s}/alias.json", .{h.home});
    var b1: [MAX_PATH]u8 = undefined;
    var b2: [MAX_PATH]u8 = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(cz(&b1, h.path("lock")).?, cz(&b2, alias).?));
    e.engine.table_switch = .{ .path = alias };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
    try testing.expect(fileExists(h.path("lock")));
    // the complement: a table file of its own is removed, with the progress
    const custom = try std.fmt.allocPrint(a, "{s}/custom.json", .{h.home});
    try kernel_table.writeAtomic(a, testing.io, custom, "x\n");
    e.engine.table_switch = .{ .path = custom };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--reset"}));
    try testing.expect(!fileExists(custom));
    try testing.expect(!fileExists(h.path("progress.jsonl")));
    try testing.expect(fileExists(h.path("lock")));
}

test "tune cli: a progress summary that fails is reported, never omitted" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const mac = try kernel_table.readMacKey(a);
    const header = try std.fmt.allocPrint(a, "{{\"progress\":1,\"key\":{{\"gpu_arch\":\"{s}\",\"gpu_cores\":{d},\"hw_model\":\"{s}\",\"os_build\":\"{s}\",\"kernel_hash\":\"{s}\",\"mlx_version\":\"{s}\"}},\"space\":{d}}}\n", .{ mac.gpu_arch, mac.gpu_cores, mac.hw_model, mac.os_build, mac.kernel_hash, mac.mlx_version, kernel_table.TUNE_PLAN_SPACE_VERSION });
    const rec = "{\"phase\":\"screen\",\"kernel\":\"gateup\",\"cand\":\"sg=1 ax=1\",\"regime\":\"indep\",\"window\":1,\"kept\":false,\"reason\":\"input\"}\n";
    try h.put("progress.jsonl", try std.mem.concat(a, u8, &.{ header, rec }));
    var reported: u32 = 0;
    var i: usize = 0;
    var done = false;
    while (i < 400) : (i += 1) {
        var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = i });
        var ar = std.heap.ArenaAllocator.init(fa.allocator());
        defer ar.deinit();
        h.aw.clearRetainingCapacity();
        const ok = showProgress(ar.allocator(), testing.io, h.env(), mac);
        const out = h.output();
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "[tune] progress: "));
        if (ok) {
            try testing.expect(contains(out, "1 discarded (input 1,"));
            done = true;
            break;
        }
        try testing.expect(contains(out, "[tune] progress: unreadable (OutOfMemory)"));
        reported += 1;
    }
    try testing.expect(done);
    try testing.expect(reported > 0);
}

fn lockWriteNoSpace(_: c_int, _: []const u8) isize {
    std.c._errno().* = @intFromEnum(std.c.E.NOSPC);
    return -1;
}

fn lockWriteShort(_: c_int, _: []const u8) isize {
    return 1;
}

test "tune cli: a failed lock record write names its errno" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try ensureFolder(h.root);
    defer lock_write_override = null;
    lock_write_override = lockWriteNoSpace;
    try testing.expectError(error.LockUnavailable, takeLock(testing.io, h.env()));
    try testing.expectEqual(std.c.E.NOSPC, last_errno);
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expect(contains(h.output(), "the lock could not be taken (LockUnavailable, errno NOSPC)"));
    lock_write_override = lockWriteShort;
    try testing.expectError(error.LockUnavailable, takeLock(testing.io, h.env()));
    try testing.expect(last_errno != .SUCCESS);
    // the complement: the real write
    lock_write_override = null;
    const r = try takeLock(testing.io, h.env());
    try testing.expect(r == .held);
    r.held.release();
}

test "tune cli: a tuner file is found under any spelling of the table path" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    // a fresh folder: no tuner file exists yet, so only the canonical path can tell
    try ensureFolder(h.root);
    const cases = [_]struct { p: []const u8, want: []const u8 }{
        .{ .p = "/./tune.log", .want = "tune.log" },
        .{ .p = "//lock", .want = "lock" },
        .{ .p = "/sub/../install.lock", .want = "install.lock" },
    };
    for (cases) |c| {
        const p = try std.mem.concat(a, u8, &.{ h.root, c.p });
        try testing.expect(!fileExists(h.path(c.want)));
        try testing.expectEqualStrings(c.want, tableCollision(p, h.root) orelse "none");
    }
    // through the command on a fresh folder: refused, and the log is not replaced by a table
    var e = h.env();
    e.engine.table_switch = .{ .path = try std.mem.concat(a, u8, &.{ h.root, "/./tune.log" }) };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
    try testing.expect(contains(h.output(), "names the tuner's own tune.log;"));
    // the complement: a table file of its own, under a plain or a dotted spelling, is not a tuner file
    const custom = try std.fmt.allocPrint(a, "{s}/custom.json", .{h.home});
    try testing.expect(tableCollision(custom, h.root) == null);
    try testing.expect(tableCollision(try std.fmt.allocPrint(a, "{s}/x/../custom.json", .{h.home}), h.root) == null);
    try testing.expect(tableCollision(try std.fmt.allocPrint(a, "{s}/lock.json", .{h.root}), h.root) == null);
    try kernel_table.writeAtomic(a, testing.io, custom, "x\n");
    e.engine.table_switch = .{ .path = custom };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--reset"}));
    try testing.expect(!fileExists(custom));
}

var write_on_busy_path: []const u8 = "";
var write_on_busy_bytes: []const u8 = "";

/// Busy for `busy_left` exclusive tries; on the last busy one, writes `write_on_busy_bytes` to `write_on_busy_path` (a
/// tune finishing while the agent waits).
fn flockBusyThenWrite(fd: c_int, op: c_int) c_int {
    if (op & LOCK_EX == 0) return flock(fd, op);
    ex_calls += 1;
    if (busy_left > 0) {
        busy_left -= 1;
        if (busy_left == 0 and write_on_busy_bytes.len > 0) {
            kernel_table.writeAtomic(std.heap.page_allocator, testing.io, write_on_busy_path, write_on_busy_bytes) catch {};
        }
        std.c._errno().* = @intFromEnum(std.c.E.AGAIN);
        return -1;
    }
    return flock(fd, op);
}

test "tune cli: the agent decides to tune only once it holds the lock" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const mac = try kernel_table.readMacKey(a);
    const bytes = try kernel_table.serialize(a, try buildTable(a, mac, sampleOutcome(null, .{ .sps = 6, .tpt = 1, .ax = 1 }, .{ .below_floor, null }), .{}));
    const saved_poll = agent_lock_poll_ns;
    agent_lock_poll_ns = std.time.ns_per_ms;
    defer agent_lock_poll_ns = saved_poll;
    flock_override = flockBusyThenWrite;
    defer flock_override = null;
    sched_override = schedFinishes;
    defer sched_override = null;
    defer write_on_busy_bytes = "";
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    // a current table appears while the agent waits: the scheduler is never called, the table is kept, the agent leaves
    try kernel_table.writeAtomic(a, testing.io, plist, "x\n");
    write_on_busy_path = h.path("kernel-plans.json");
    write_on_busy_bytes = bytes;
    busy_left = 2;
    ex_calls = 0;
    sched_calls = 0;
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--agent"}));
    try testing.expectEqual(@as(u32, 0), sched_calls);
    try testing.expect(contains(h.output(), "[tune] the table became current while this run waited for the lock; nothing was run"));
    try testing.expectEqualStrings(bytes, try readAll(a, h.path("kernel-plans.json")));
    try testing.expect(!fileExists(plist));
    // the complement: the table is still absent once the lock is held, so the scheduler runs once
    _ = try removeFile(h.path("kernel-plans.json"));
    try kernel_table.writeAtomic(a, testing.io, plist, "x\n");
    write_on_busy_bytes = "";
    busy_left = 2;
    ex_calls = 0;
    sched_calls = 0;
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--agent"}));
    try testing.expectEqual(@as(u32, 1), sched_calls);
    try testing.expect(!contains(h.output(), "became current"));
    try testing.expect(fileExists(h.path("kernel-plans.json")));
    try testing.expect(!fileExists(plist));
}

test "tune cli: a LaunchAgent removal holds the install lock" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    try kernel_table.writeAtomic(a, testing.io, plist, "x\n");
    try h.put("install.lock", "");
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("install.lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    // an install holds it: --stop exits 2 and leaves the plist as it is
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] an install is running\n"));
    try testing.expectEqualStrings("x\n", try readAll(a, plist));
    // the agent's own removal tries a few times, then reports it
    const saved = install_lock_retry_ns;
    install_lock_retry_ns = std.time.ns_per_ms;
    defer install_lock_retry_ns = saved;
    h.aw.clearRetainingCapacity();
    const r = removeLaunchAgent(a, testing.io, h.env(), true);
    try testing.expect(!r.ok and !r.removed);
    try testing.expect(contains(h.output(), "an install is running; the LaunchAgent was left in place"));
    try testing.expectEqualStrings("x\n", try readAll(a, plist));
    // the complement: the install lock free, the plist is removed
    _ = flock(fd, LOCK_UN);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] LaunchAgent removed: "));
    try testing.expect(!fileExists(plist));
}

fn entryExists(p: []const u8) bool {
    var b: [MAX_PATH]u8 = undefined;
    var st: std.c.Stat = undefined;
    return c_lstat(cz(&b, p).?, &st) == 0;
}

test "tune cli: a dangling LaunchAgent symlink survives a failed install" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    fake_staged_path = try std.fmt.allocPrint(a, "{s}.staged-{d}", .{ plist, std.c.getpid() });
    launchctl_override = fakeLaunchctl;
    defer launchctl_override = null;
    defer fake_bootstrap_code = 0;
    const target = "/nonexistent-lesto-tune-test/old.plist";
    var b1: [MAX_PATH]u8 = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(target, cz(&b1, plist).?));
    try testing.expect(!fileExists(plist)); // dangling: a link-following check sees nothing
    fake_n = 0;
    fake_bootstrap_code = 5;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    var lb: [MAX_PATH]u8 = undefined;
    const n = std.c.readlink(cz(&b1, plist).?, &lb, lb.len);
    try testing.expectEqual(@as(isize, @intCast(target.len)), n);
    try testing.expectEqualStrings(target, lb[0..target.len]);
    try testing.expect(!entryExists(fake_staged_path));
    try testing.expect(contains(h.output(), "the previous LaunchAgent was put back at "));
    // the complement: no previous entry, a failed install leaves none
    try testing.expectEqual(@as(c_int, 0), std.c.unlink(cz(&b1, plist).?));
    fake_n = 0;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    try testing.expect(!entryExists(plist));
    try testing.expect(!entryExists(fake_staged_path));
    try testing.expect(contains(h.output(), "none is installed, as before"));
}

test "tune cli: --stop without a tune folder is an error" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    var e = h.env();
    e.engine.home = null;
    e.engine.tune_root = null;
    const agents = e.launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    try kernel_table.writeAtomic(a, testing.io, plist, "x\n");
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--stop"}));
    try testing.expectEqualStrings(NO_HOME ++ "\n", h.output());
    try testing.expectEqualStrings("x\n", try readAll(a, plist));
    try testing.expectEqual(StopResult{ .failed = .NOENT }, stopRunning(testing.io, e));
    // the complement: with the folder known, nothing runs and the LaunchAgent is removed
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--stop"}));
    try testing.expect(contains(h.output(), "[tune] no tune is running"));
    try testing.expect(!fileExists(plist));
}

var pidinfo_calls: u32 = 0;
var pidinfo_fail_left: u32 = 0;

fn pidinfoScripted(pid: c_int) ?StartTime {
    pidinfo_calls += 1;
    if (pidinfo_fail_left > 0) {
        pidinfo_fail_left -= 1;
        std.c._errno().* = @intFromEnum(std.c.E.PERM);
        return null;
    }
    return procStart(pid);
}

test "tune cli: the lock never records a made-up start time" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    try ensureFolder(h.root);
    pidinfo_override = pidinfoScripted;
    defer pidinfo_override = null;
    // the lookup fails twice: the take fails naming the errno, and the lock is free with no record
    pidinfo_calls = 0;
    pidinfo_fail_left = 2;
    try testing.expectError(error.LockFailed, takeLock(testing.io, h.env()));
    try testing.expectEqual(std.c.E.PERM, last_errno);
    try testing.expectEqual(@as(u32, 2), pidinfo_calls);
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    _ = flock(fd, LOCK_UN);
    _ = std.c.close(fd);
    try testing.expectEqualStrings("", try readAll(a, h.path("lock")));
    pidinfo_fail_left = 2;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expect(contains(h.output(), "the lock could not be taken (LockFailed, errno PERM)"));
    // the complement: one failed lookup is retried, and the record is the real start time
    pidinfo_calls = 0;
    pidinfo_fail_left = 1;
    const r = try takeLock(testing.io, h.env());
    try testing.expect(r == .held);
    defer r.held.release();
    try testing.expectEqual(@as(u32, 2), pidinfo_calls);
    const me = std.c.getpid();
    const real = procStart(me).?;
    try testing.expectEqualStrings(try std.fmt.allocPrint(a, "{d} {d} {d}\n", .{ me, real.sec, real.usec }), try readAll(a, h.path("lock")));
}

test "tune cli: the program path is read inside the ProgramArguments array" {
    const head = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\">\n<dict>\n\t<key>ProgramArguments</key>\n";
    const tail = "\t<key>StandardOutPath</key>\n\t<string>/h/agent.log</string>\n</dict>\n</plist>\n";
    const empties = [_][]const u8{ head ++ "\t<array>\n\t</array>\n" ++ tail, head ++ "\t<array/>\n" ++ tail, head ++ "\t<string>/not/in/an/array</string>\n" ++ tail };
    for (empties) |t| try testing.expect((try plistProgram(testing.allocator, t)) == null);
    const normal = try plistText(testing.allocator, "/opt/lesto/bin/lesto", 5, "/h/agent.log");
    defer testing.allocator.free(normal);
    const prog = (try plistProgram(testing.allocator, normal)).?;
    defer testing.allocator.free(prog);
    try testing.expectEqualStrings("/opt/lesto/bin/lesto", prog);
    // through --show: an empty array is no program path, never the log file
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    try kernel_table.writeAtomic(a, testing.io, try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME }), empties[0]);
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--show"}));
    try testing.expect(contains(h.output(), "no program path in it"));
    try testing.expect(!contains(h.output(), "program /h/agent.log"));
}

test "tune cli: --show never skips the LaunchAgent section" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    try kernel_table.writeAtomic(a, testing.io, try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME }), try plistText(a, "/opt/no-such-dir/lesto", 5, "/h/agent.log"));
    var reported: u32 = 0;
    var i: usize = 0;
    var done = false;
    while (i < 400) : (i += 1) {
        var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = i });
        var ar = std.heap.ArenaAllocator.init(fa.allocator());
        defer ar.deinit();
        h.aw.clearRetainingCapacity();
        const ok = showAgent(ar.allocator(), testing.io, h.env());
        const out = h.output();
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "[tune] LaunchAgent: "));
        if (ok) {
            try testing.expect(contains(out, "program /opt/no-such-dir/lesto is missing"));
            done = true;
            break;
        }
        try testing.expect(contains(out, "[tune] LaunchAgent: unreadable (OutOfMemory)"));
        reported += 1;
    }
    try testing.expect(done);
    try testing.expect(reported > 0);
}

fn aclLookupFails(_: c_int, _: c_int) ?*anyopaque {
    std.c._errno().* = @intFromEnum(std.c.E.NOMEM);
    return null;
}

fn aclLookupNone(_: c_int, _: c_int) ?*anyopaque {
    std.c._errno().* = @intFromEnum(std.c.E.NOENT);
    return null;
}

test "tune cli: an ACL lookup that fails is never read as no ACL" {
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 501, .mode = 0o100600, .nlink = 1, .has_acl = .unknown }, 0o600));
    try testing.expectEqual(PrivateVerdict.refuse, privateVerdict(501, .{ .uid = 501, .mode = 0o040700, .nlink = 3, .has_acl = .unknown }, 0o700));
    try testing.expectEqual(PrivateVerdict.ok, privateVerdict(501, .{ .uid = 501, .mode = 0o100600, .nlink = 1, .has_acl = .none }, 0o600));
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try h.put("tune.log", "x\n");
    try h.put("lock", "x\n");
    defer acl_get_override = null;
    acl_get_override = aclLookupFails;
    try testing.expectError(error.AclUnknown, ensureTuneFolder(h.root));
    try testing.expectEqual(std.c.E.NOMEM, last_errno);
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--reset"}));
    try testing.expect(contains(h.output(), "is not usable (AclUnknown, errno NOMEM)"));
    // a held lock whose ACL cannot be read is not trusted by --stop
    var b: [MAX_PATH]u8 = undefined;
    const fd = std.c.open(cz(&b, h.path("lock")).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectEqual(@as(c_int, 0), flock(fd, LOCK_EX | LOCK_NB));
    const r = stopRunning(testing.io, h.env());
    try testing.expect(r == .untrusted);
    try testing.expectEqualStrings("its access control list cannot be read", r.untrusted);
    // the complement: ENOENT is the one answer that means no ACL
    acl_get_override = aclLookupNone;
    try ensureTuneFolder(h.root);
    try testing.expectEqual(StopResult{ .unidentified = 0 }, stopRunning(testing.io, h.env()));
    acl_get_override = null;
    try ensureTuneFolder(h.root);
}

fn inodeOf(p: []const u8) !u64 {
    var b: [MAX_PATH]u8 = undefined;
    const st = statOf(cz(&b, p).?) orelse return error.TestNoFile;
    return @intCast(st.ino);
}

test "tune cli: a lock that was ever wider than private is replaced, never reused" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    for ([_][]const u8{ "lock", "install.lock" }) |name| {
        try h.put(name, "4242 1 2\n");
        try chmodPath(h.path(name), 0o666);
        const before = try inodeOf(h.path(name));
        // a descriptor opened while others could write it
        var b: [MAX_PATH]u8 = undefined;
        const old = std.c.open(cz(&b, h.path(name)).?, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
        try testing.expect(old >= 0);
        defer _ = std.c.close(old);
        try ensureTuneFolder(h.root);
        try testing.expect(try inodeOf(h.path(name)) != before);
        try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path(name)));
        try testing.expectEqualStrings("", try readAll(a, h.path(name)));
        // that descriptor now writes only the old inode, never the lock
        try testing.expectEqual(@as(isize, 6), std.c.pwrite(old, "1 2 3\n", 6, 0));
        try testing.expectEqualStrings("", try readAll(a, h.path(name)));
    }
    // the complement: a private lock keeps its inode and its record
    try h.put("lock", "4242 1 2\n");
    try chmodPath(h.path("lock"), 0o600);
    const kept = try inodeOf(h.path("lock"));
    try ensureTuneFolder(h.root);
    try testing.expectEqual(kept, try inodeOf(h.path("lock")));
    try testing.expectEqualStrings("4242 1 2\n", try readAll(a, h.path("lock")));
    // takeLock alone: a free wide lock is replaced and the new file is the one held
    try chmodPath(h.path("lock"), 0o644);
    const wide = try inodeOf(h.path("lock"));
    const r = try takeLock(testing.io, h.env());
    try testing.expect(r == .held);
    defer r.held.release();
    try testing.expect(try inodeOf(h.path("lock")) != wide);
    try testing.expectEqual(@as(u32, 0o600), try modeOf(h.path("lock")));
    var st: std.c.Stat = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.fstat(r.held.fd, &st));
    try testing.expectEqual(try inodeOf(h.path("lock")), @as(u64, @intCast(st.ino)));
}

test "tune cli: reserved names are compared ignoring case" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    // a fresh folder: none of the tuner files exists yet
    try ensureFolder(h.root);
    const cases = [_]struct { p: []const u8, want: []const u8 }{
        .{ .p = "TUNE.LOG", .want = "tune.log" },
        .{ .p = "Lock", .want = "lock" },
        .{ .p = "INSTALL.LOCK", .want = "install.lock" },
        .{ .p = "Progress.JSONL", .want = "progress.jsonl" },
        .{ .p = "./Agent.Log", .want = "agent.log" },
    };
    for (cases) |c| {
        try testing.expect(!fileExists(h.path(c.want)));
        try testing.expectEqualStrings(c.want, tableCollision(h.path(c.p), h.root) orelse "none");
    }
    var e = h.env();
    e.engine.table_switch = .{ .path = h.path("TUNE.LOG") };
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--reset"}));
    try testing.expect(contains(h.output(), "names the tuner's own tune.log;"));
    // a parent spelled in another case names the same folder on a case-insensitive volume
    const upper_root = try std.fmt.allocPrint(a, "{s}/.MLX-SERVE/TUNE", .{h.home});
    if (fileExists(upper_root)) {
        try testing.expectEqualStrings("tune.log", tableCollision(try std.fmt.allocPrint(a, "{s}/tune.log", .{upper_root}), h.root) orelse "none");
    }
    // the complement: the table's own names are not reserved
    try testing.expect(tableCollision(h.path("kernel-plans.json"), h.root) == null);
    try testing.expect(tableCollision(h.path("plans.json"), h.root) == null);
    try testing.expect(tableCollision(h.path("lock.json"), h.root) == null);
    try h.put("plans.json", "x\n");
    e.engine.table_switch = .{ .path = h.path("plans.json") };
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--reset"}));
    try testing.expect(!fileExists(h.path("plans.json")));
}

test "tune cli: a table path naming the LaunchAgent plist is refused" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    fake_staged_path = try std.fmt.allocPrint(a, "{s}.staged-{d}", .{ plist, std.c.getpid() });
    launchctl_override = fakeLaunchctl;
    defer launchctl_override = null;
    fake_bootstrap_code = 0;
    var e = h.env();
    e.engine.table_switch = .{ .path = plist };
    // --background: refused before anything is written or any launchctl call
    fake_n = 0;
    try testing.expectEqual(@as(u8, 2), h.run(e, &.{"--background"}));
    try testing.expect(contains(h.output(), "names the tuner's own " ++ PLIST_NAME ++ ";"));
    try testing.expect(!entryExists(plist));
    try testing.expectEqual(@as(usize, 0), fake_n);
    // a plist there: the default action, --verify and --reset leave it as it is
    try kernel_table.writeAtomic(a, testing.io, plist, "PLIST\n");
    sched_override = schedFinishes;
    defer sched_override = null;
    for ([_][]const []const u8{ &.{}, &.{"--verify"}, &.{"--reset"} }) |args| {
        try testing.expectEqual(@as(u8, 2), h.run(e, args));
        try testing.expect(contains(h.output(), "names the tuner's own " ++ PLIST_NAME ++ ";"));
        try testing.expectEqualStrings("PLIST\n", try readAll(a, plist));
    }
    // the install's own names and any case of the plist's are reserved too
    try testing.expect(plistCollision(try std.fmt.allocPrint(a, "{s}.staged-77", .{plist}), agents) != null);
    try testing.expect(plistCollision(try std.fmt.allocPrint(a, "{s}.previous-77", .{plist}), agents) != null);
    try testing.expect(plistCollision(try std.fmt.allocPrint(a, "{s}/COM.LESTOENGINE.LESTO.TUNE.PLIST", .{agents}), agents) != null);
    // the complement: a table file of its own beside the plist is not the plist, and --background installs
    try testing.expect(plistCollision(try std.fmt.allocPrint(a, "{s}/tune-table.json", .{agents}), agents) == null);
    e.engine.table_switch = .{ .path = try std.fmt.allocPrint(a, "{s}/plans.json", .{h.home}) };
    fake_n = 0;
    try testing.expectEqual(@as(u8, 0), h.run(e, &.{"--background"}));
    try testing.expect(contains(try readAll(a, plist), "<key>ProgramArguments</key>"));
}

var block_backup_path: []const u8 = "";

/// The fake launchctl, plus: at bootstrap, the set-aside plist becomes a non-empty folder, which `unlink` refuses.
fn fakeLaunchctlBlocksBackup(argv: []const []const u8) LaunchctlResult {
    if (std.mem.eql(u8, argv[1], "bootstrap") and block_backup_path.len > 0) {
        var b: [MAX_PATH]u8 = undefined;
        const z = cz(&b, block_backup_path).?;
        _ = std.c.unlink(z);
        _ = std.c.mkdir(z, @as(std.c.mode_t, 0o700));
        var fb: [MAX_PATH]u8 = undefined;
        const fd = std.c.open(joinZ(&fb, block_backup_path, "x").?, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
        if (fd >= 0) _ = std.c.close(fd);
    }
    return fakeLaunchctl(argv);
}

test "tune cli: only a file or a link is a previous LaunchAgent" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    const agents = h.env().launch_agents_dir.?;
    try ensureFolder(agents);
    const plist = try std.fmt.allocPrint(a, "{s}/{s}", .{ agents, PLIST_NAME });
    const backup = try std.fmt.allocPrint(a, "{s}.previous-{d}", .{ plist, std.c.getpid() });
    fake_staged_path = try std.fmt.allocPrint(a, "{s}.staged-{d}", .{ plist, std.c.getpid() });
    launchctl_override = fakeLaunchctl;
    defer launchctl_override = null;
    fake_bootstrap_code = 0;
    // a folder at the plist path: refused before anything changes, the folder and its contents left as they are
    try ensureFolder(plist);
    const keep = try std.fmt.allocPrint(a, "{s}/keep.txt", .{plist});
    try kernel_table.writeAtomic(a, testing.io, keep, "k\n");
    fake_n = 0;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    try testing.expect(contains(h.output(), "is a folder, not a plist file; nothing changed"));
    var zb: [MAX_PATH]u8 = undefined;
    try testing.expect(isDir(cz(&zb, plist).?));
    try testing.expectEqualStrings("k\n", try readAll(a, keep));
    try testing.expect(!entryExists(fake_staged_path));
    try testing.expect(!entryExists(backup));
    try testing.expectEqual(@as(usize, 0), fake_n);
    try testing.expectEqual(@as(c_int, 0), std.c.unlink(cz(&zb, keep).?));
    try testing.expectEqual(@as(c_int, 0), std.c.rmdir(cz(&zb, plist).?));
    // the complement: a regular previous plist is replaced and its set-aside copy is gone
    try kernel_table.writeAtomic(a, testing.io, plist, "OLD\n");
    fake_n = 0;
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{"--background"}));
    try testing.expect(contains(try readAll(a, plist), "<key>ProgramArguments</key>"));
    try testing.expect(!entryExists(backup));
    // a set-aside copy that cannot be removed: the agent is installed, the command exits 2 naming the copy
    try kernel_table.writeAtomic(a, testing.io, plist, "OLD\n");
    launchctl_override = fakeLaunchctlBlocksBackup;
    block_backup_path = backup;
    defer block_backup_path = "";
    fake_n = 0;
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{"--background"}));
    try testing.expect(contains(h.output(), "[tune] LaunchAgent installed: "));
    try testing.expect(contains(h.output(), try std.fmt.allocPrint(a, "the previous LaunchAgent set aside at {s} could not be removed (errno ", .{backup})));
    try testing.expect(contains(try readAll(a, plist), "<key>ProgramArguments</key>"));
}

test "tune cli: a layout line never reads served when the run picked one" {
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    const a = h.arena.allocator();
    sched_override = schedFinishes;
    defer sched_override = null;
    var lb: [48]u8 = undefined;
    const want = try std.fmt.allocPrint(a, "[tune] gate/up: {s}, ", .{kernel_table.gateUpText(&lb, .{ .sg = 32, .ax = 1 })});
    var failed: u32 = 0;
    var i: usize = 0;
    var done = false;
    while (i < 1000) : (i += 1) {
        _ = removeFile(h.path("kernel-plans.json")) catch {};
        var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = i });
        h.aw.clearRetainingCapacity();
        const rc = runWith(fa.allocator(), testing.io, h.env(), &.{});
        const out = h.output();
        try testing.expect(!contains(out, "[tune] gate/up: served"));
        if (!fa.has_induced_failure) {
            try testing.expectEqual(@as(u8, 0), rc);
            try testing.expect(contains(out, want));
            done = true;
            break;
        }
        if (rc == 0) {
            try testing.expect(contains(out, want));
        } else {
            try testing.expectEqual(@as(u8, 2), rc);
            try testing.expect(contains(out, "lesto tune: "));
            failed += 1;
        }
    }
    try testing.expect(done);
    try testing.expect(failed > 0);
}

test "tune cli: a commented-out ProgramArguments never stands for the real one" {
    const al = testing.allocator;
    const real = try plistText(al, "/opt/real/lesto", 5, "/h/agent.log");
    defer al.free(real);
    const at = std.mem.indexOf(u8, real, "<dict>\n").? + "<dict>\n".len;
    const variants = [_][]const u8{
        "\t<!-- <key>ProgramArguments</key><array><string>/opt/fake/lesto</string></array> -->\n",
        "\t<key>Comment</key>\n\t<string><![CDATA[<key>ProgramArguments</key><array><string>/opt/fake2</string></array>]]></string>\n",
    };
    for (variants) |v| {
        const t = try std.mem.concat(al, u8, &.{ real[0..at], v, real[at..] });
        defer al.free(t);
        const prog = (try plistProgram(al, t)).?;
        defer al.free(prog);
        try testing.expectEqualStrings("/opt/real/lesto", prog);
    }
    // the complement: no comment, the same answer
    const plain = (try plistProgram(al, real)).?;
    defer al.free(plain);
    try testing.expectEqualStrings("/opt/real/lesto", plain);
    // a real value given as CDATA reads back as its text
    const head = "<plist version=\"1.0\">\n<dict>\n\t<key>ProgramArguments</key>\n\t<array>\n";
    const cdata = try std.mem.concat(al, u8, &.{ head, "\t\t<string><![CDATA[/opt/a&b/lesto]]></string>\n\t</array>\n</dict>\n</plist>\n" });
    defer al.free(cdata);
    const cp = (try plistProgram(al, cdata)).?;
    defer al.free(cp);
    try testing.expectEqualStrings("/opt/a&b/lesto", cp);
}

test "tune cli: --print-plist-for-test goes alone with exactly its value" {
    try testing.expectEqualStrings("/x", parseFlags(&.{ "--print-plist-for-test", "/x" }).print_plist);
    const bad = [_][]const []const u8{
        &.{ "--verify", "--print-plist-for-test", "/x" },
        &.{ "--print-plist-for-test", "/x", "--verify" },
        &.{ "--print-plist-for-test", "/x", "extra" },
        &.{"--print-plist-for-test"},
        &.{ "--print-plist-for-test", "/x", "--idle-minutes", "3" },
        &.{ "--print-plist-for-test", "/x", "--print-plist-for-test", "/y" },
    };
    for (bad) |args| try testing.expect(parseFlags(args) == .usage);
    var h: Harness = undefined;
    try h.init();
    defer h.deinit();
    try testing.expectEqual(@as(u8, 2), h.run(h.env(), &.{ "--verify", "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), "usage: lesto tune"));
    try testing.expect(!contains(h.output(), "<plist"));
    // the complement: alone with its value, it prints the plist
    try testing.expectEqual(@as(u8, 0), h.run(h.env(), &.{ "--print-plist-for-test", "/opt/lesto" }));
    try testing.expect(contains(h.output(), "<plist"));
}
