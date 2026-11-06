const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;

pub const RestoreDumpMeta = struct {
    kind: []const u8,
    pos: usize,
    source: []const u8 = "own",
    entry_idx: usize = 0,
    entry_count: usize = 0,
};

pub var dump_restore_override: ?[]const u8 = null;
var dump_seq: std.atomic.Value(u64) = .init(0);
var dump_dir_cached: ?[]const u8 = null;
var dump_dir_read: bool = false;

pub fn dumpRestoreDir() ?[]const u8 {
    if (dump_restore_override) |d| {
        if (d.len == 0 or d[0] == '0') return null;
        return d;
    }
    if (!dump_dir_read) {
        dump_dir_cached = dumpRestoreDirFromEnv(std.c.getenv("MLX_SERVE_DUMP_RESTORE"));
        dump_dir_read = true;
    }
    return dump_dir_cached;
}

pub fn dumpRestoreDirFromEnv(raw: ?[*:0]const u8) ?[]const u8 {
    const v = raw orelse return null;
    if (v[0] == 0 or v[0] == '0') return null;
    return std.mem.span(v);
}

pub fn dumpRestoreIfEnabled(
    cache: *KVCache,
    s: mlx.mlx_stream,
    meta: RestoreDumpMeta,
) ?u64 {
    const dir = dumpRestoreDir() orelse return null;
    const n = dump_seq.fetchAdd(1, .monotonic) + 1;
    // A failed save raises inside mlx and our handler latches it; a diagnostic
    // must never turn into the request's MlxFailure, so drop what it set.
    const had_error = mlx.errorPending();
    dumpRestore(dir, n, cache, s, meta) catch |err| {
        log.warn("  [hot-cache] restore dump failed: {s}\n", .{@errorName(err)});
    };
    mlx.dropLatchedErrorUnless(had_error);
    return n;
}

fn dumpRestore(
    dir: []const u8,
    n: u64,
    cache: *KVCache,
    s: mlx.mlx_stream,
    meta: RestoreDumpMeta,
) !void {
    var dir_z_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_z = zfmt(&dir_z_buf, "{s}", .{dir}) catch return error.NameTooLong;
    _ = std.c.mkdir(dir_z.ptr, @as(std.c.mode_t, 0o755));
    const gpa = std.heap.page_allocator;
    const stem = try std.fmt.allocPrint(gpa, "{s}/{s}-{d}-pos{d}", .{ dir, meta.kind, n, meta.pos });
    defer gpa.free(stem);
    const st_path = try std.fmt.allocPrintSentinel(gpa, "{s}.safetensors", .{stem}, 0);
    defer gpa.free(st_path);
    const json_path = try std.fmt.allocPrint(gpa, "{s}.json", .{stem});
    defer gpa.free(json_path);

    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const mmap = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(mmap);
    var key_buf: [96]u8 = undefined;

    try putInt(map, try zfmt(&key_buf, "pos", .{}), @intCast(meta.pos));
    for (cache.entries, 0..) |*entry, i| {
        if (!entry.initialized) continue;
        var view = cache.denseView(@intCast(i), s) catch continue;
        defer view.deinit();
        try putKvTail(map, try zfmt(&key_buf, "kv.{d}.k_tail", .{i}), view.k, s);
        try putKvTail(map, try zfmt(&key_buf, "kv.{d}.v_tail", .{i}), view.v, s);
    }

    try mlx.check(mlx.mlx_save_safetensors(st_path, map, mmap));

    const json = try std.fmt.allocPrint(
        gpa,
        "{{\"kind\":\"{s}\",\"pos\":{d},\"source\":\"{s}\",\"entry\":{d},\"entry_count\":{d}}}\n",
        .{ meta.kind, meta.pos, meta.source, meta.entry_idx, meta.entry_count },
    );
    defer gpa.free(json);
    var json_z_buf: [std.fs.max_path_bytes]u8 = undefined;
    const json_z = zfmt(&json_z_buf, "{s}", .{json_path}) catch return error.NameTooLong;
    const jf = std.c.fopen(json_z.ptr, "w") orelse return error.FileNotFound;
    defer _ = std.c.fclose(jf);
    if (std.c.fwrite(json.ptr, 1, json.len, jf) != json.len) return error.WriteFailed;
}

fn zfmt(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]u8 {
    const n = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[n.len] = 0;
    return buf[0..n.len :0];
}

fn putInt(map: mlx.mlx_map_string_to_array, key: [:0]const u8, v: c_int) !void {
    const arr = mlx.mlx_array_new_int(v);
    defer _ = mlx.mlx_array_free(arr);
    _ = mlx.mlx_array_eval(arr);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, arr));
}

fn putArr(map: mlx.mlx_map_string_to_array, key: [:0]const u8, arr: mlx.mlx_array, s: mlx.mlx_stream) !void {
    _ = s;
    if (arr.ctx == null or mlx.mlx_array_size(arr) == 0) return;
    _ = mlx.mlx_array_eval(arr);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, arr));
}

fn putKvTail(map: mlx.mlx_map_string_to_array, key: [:0]const u8, arr: mlx.mlx_array, s: mlx.mlx_stream) !void {
    if (arr.ctx == null or mlx.mlx_array_size(arr) == 0) return;
    const sh = mlx.getShape(arr);
    if (sh.len != 4) {
        try putArr(map, key, arr, s);
        return;
    }
    const t = sh[2];
    if (t <= 0) return;
    const n: c_int = @min(64, t);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    const start = [_]c_int{ 0, 0, t - n, 0 };
    const stop = [_]c_int{ sh[0], sh[1], t, sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&tail, arr, &start, 4, &stop, 4, &strides, 4, s));
    _ = mlx.mlx_array_eval(tail);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, tail));
}

test "dumpRestoreDirFromEnv: absent or 0 is off" {
    try std.testing.expect(dumpRestoreDirFromEnv(null) == null);
    try std.testing.expect(dumpRestoreDirFromEnv("0") == null);
    try std.testing.expect(dumpRestoreDirFromEnv("") == null);
    const on = dumpRestoreDirFromEnv("/tmp/restore-dump");
    try std.testing.expect(on != null);
    try std.testing.expectEqualStrings("/tmp/restore-dump", on.?);
}
