//! A mutex for threads that carry no `Io` handle (log sinks, mlx-c's error handler):
//! a pthread mutex, zero-initialized.
const std = @import("std");

pub const Mutex = struct {
    m: std.c.pthread_mutex_t = .{},

    pub fn lock(m: *@This()) void {
        _ = std.c.pthread_mutex_lock(&m.m);
    }
    pub fn unlock(m: *@This()) void {
        _ = std.c.pthread_mutex_unlock(&m.m);
    }
};

test "os_mutex: lock and unlock" {
    var m: Mutex = .{};
    m.lock();
    m.unlock();
    m.lock();
    m.unlock();
}
