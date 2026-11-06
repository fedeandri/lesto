//! OS calls for threads that carry no `Io` handle (log sink, pollers): C-runtime integer
//! file descriptors, sleep and a monotonic clock.
const std = @import("std");

/// Create-or-open for writing: appends, or truncates when `truncate`. -1 on failure.
pub fn openWrite(path: [*:0]const u8, truncate: bool) c_int {
    return std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = !truncate, .TRUNC = truncate }, @as(std.c.mode_t, 0o644));
}

pub fn openRead(path: [*:0]const u8) c_int {
    return std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
}

pub fn close(fd: c_int) void {
    _ = std.c.close(fd);
}

/// Bytes written, or <= 0 on error.
pub fn write(fd: c_int, bytes: []const u8) isize {
    return std.c.write(fd, bytes.ptr, bytes.len);
}

/// Bytes read, or <= 0 on EOF/error.
pub fn read(fd: c_int, buf: []u8) isize {
    return std.c.read(fd, buf.ptr, buf.len);
}

/// Size of the open file (seeks to its end); <= 0 on error.
pub fn seekEnd(fd: c_int) i64 {
    return std.c.lseek(fd, 0, std.c.SEEK.END);
}

/// Existing directories are fine; failures surface at the next open.
pub fn mkdir(path: [*:0]const u8) void {
    _ = std.c.mkdir(path, @as(std.c.mode_t, 0o755));
}

/// Atomically replaces an existing `new`.
pub fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int {
    return std.c.rename(old, new);
}

pub fn unlink(path: [*:0]const u8) void {
    _ = std.c.unlink(path);
}

/// Positioned read (std.c.pread's argument order).
pub fn pread(fd: c_int, buf: [*]u8, len: usize, off: i64) isize {
    return std.c.pread(fd, buf, len, off);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Sets (overwrites) an environment variable for this process.
pub fn setEnv(name: [*:0]const u8, value: [*:0]const u8) void {
    _ = setenv(name, value, 1);
}

pub fn sleepMs(ms: u32) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast(@as(u64, ms % 1000) * 1_000_000) };
    _ = std.c.nanosleep(&ts, null);
}

pub fn monotonicMs() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

test "crt: write, reopen to append, read back, rename over an existing file" {
    const a = "crt_test_a.tmp";
    const b = "crt_test_b.tmp";
    defer unlink(a);
    defer unlink(b);

    var fd = openWrite(a, true);
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, 3), write(fd, "abc"));
    close(fd);
    fd = openWrite(a, false);
    try std.testing.expectEqual(@as(i64, 3), seekEnd(fd));
    _ = write(fd, "de");
    close(fd);

    fd = openWrite(b, true);
    close(fd);
    try std.testing.expectEqual(@as(c_int, 0), rename(a, b));

    fd = openRead(b);
    defer close(fd);
    var buf: [16]u8 = undefined;
    const n = read(fd, &buf);
    try std.testing.expectEqualStrings("abcde", buf[0..@intCast(n)]);
    try std.testing.expectEqual(@as(isize, 2), pread(fd, &buf, 2, 3));
    try std.testing.expectEqualStrings("de", buf[0..2]);
}
