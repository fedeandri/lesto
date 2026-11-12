//! Keeps one CPU core busy while chat turns are served. With decode steps committing on MLX's ops limit alone, the decode
//! thread spends most of each step blocked on the GPU, and macOS clocks the P cores down: the busiest P core sat at about
//! 1.8 to 2.4 GHz instead of 4.4, the same graph build of a step took 1.58 ms instead of 0.72, and a turn that follows at once
//! lost 15 ms to its first token. A spinning process removed that loss, and a higher QoS class did not, so the clock
//! follows demand. A thread here spins while a turn is active and for a short window
//! after the last activity (MLX_SERVE_KEEP_WARM_MS, default 500), then sleeps in 1 ms naps until the next activity.
//! MLX_SERVE_KEEP_WARM_OFF=1 turns it off. It changes no computation, only how busy one core is.
const std = @import("std");
const builtin = @import("builtin");
const crt = @import("crt.zig");
const log = @import("log.zig");

const DEFAULT_WINDOW_MS: i64 = 500;
const PROOF_TOUCHES: u64 = 2048;

var last_ms = std.atomic.Value(i64).init(0);
/// 0 not started, 1 running, 2 off
var state = std.atomic.Value(u8).init(0);
var spun_ms = std.atomic.Value(u64).init(0);
var touches = std.atomic.Value(u64).init(0);

fn windowMs() i64 {
    const raw = std.c.getenv("MLX_SERVE_KEEP_WARM_MS") orelse return DEFAULT_WINDOW_MS;
    const v = std.fmt.parseInt(i64, std.mem.span(raw), 10) catch return DEFAULT_WINDOW_MS;
    return if (v > 0) v else DEFAULT_WINDOW_MS;
}

/// Marks activity now (a request read, a decode step) and starts the spinning thread on the first call.
pub fn touch() void {
    const s = state.load(.acquire);
    if (s == 2) return;
    // fetchMax: a touch that read the clock earlier and stores later never moves the mark back
    _ = last_ms.fetchMax(crt.monotonicMs(), .acq_rel);
    if (s == 0) start(windowMs());
    const n = touches.fetchAdd(1, .monotonic) + 1;
    if (n == PROOF_TOUCHES and state.load(.acquire) == 1) {
        log.info("[keep-warm] spun {d} ms over the first {d} activity marks\n", .{ spun_ms.load(.monotonic), PROOF_TOUCHES });
    }
}

fn start(window: i64) void {
    if (state.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return;
    if (builtin.os.tag != .macos or std.c.getenv("MLX_SERVE_KEEP_WARM_OFF") != null) {
        state.store(2, .release);
        return;
    }
    const t = std.Thread.spawn(.{}, loop, .{window}) catch |err| {
        log.warn("[keep-warm] thread spawn failed ({s}); off\n", .{@errorName(err)});
        state.store(2, .release);
        return;
    };
    t.detach();
    log.info("[keep-warm] a thread keeps one CPU core busy during turns and {d} ms after (MLX_SERVE_KEEP_WARM_OFF=1 turns it off)\n", .{window});
}

fn loop(window: i64) void {
    while (true) {
        const now = crt.monotonicMs();
        if (now - last_ms.load(.acquire) < window) {
            const until = now + 1;
            while (crt.monotonicMs() < until) std.atomic.spinLoopHint();
            _ = spun_ms.fetchAdd(1, .monotonic);
        } else {
            crt.sleepMs(1);
        }
    }
}

/// Waits up to 1 s for the spin count to pass `above`, renewing the activity mark at every poll so the window cannot lapse
/// while a loaded machine runs the thread late; no fixed sleep is assumed.
fn waitSpunAbove(above: u64) bool {
    const deadline = crt.monotonicMs() + 1000;
    while (crt.monotonicMs() < deadline) {
        _ = last_ms.fetchMax(crt.monotonicMs(), .acq_rel);
        if (spun_ms.load(.monotonic) > above) return true;
        crt.sleepMs(5);
    }
    return false;
}

test "keep-warm spins after activity and stops once the window has passed" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    if (std.c.getenv("MLX_SERVE_KEEP_WARM_OFF") != null) return error.SkipZigTest;
    const before = spun_ms.load(.monotonic);
    _ = last_ms.fetchMax(crt.monotonicMs(), .acq_rel);
    start(100);
    try std.testing.expectEqual(@as(u8, 1), state.load(.acquire));
    try std.testing.expect(waitSpunAbove(before));
    crt.sleepMs(700); // past the window: 100 ms here, or the default 500 when an earlier test's touch started the thread
    const after_window = spun_ms.load(.monotonic);
    crt.sleepMs(200);
    try std.testing.expect(spun_ms.load(.monotonic) - after_window <= 2);
    touch();
    try std.testing.expect(waitSpunAbove(after_window + 2));
}

test "decode steps and request reads mark activity" {
    const gen = @embedFile("generate.zig");
    const at = std.mem.indexOf(u8, gen, "fn asyncEvalDecodeStep(") orelse return error.MissingDecodeStep;
    const call = std.mem.indexOfPos(u8, gen, at, "keep_warm.touch();") orelse return error.MissingDecodeTouch;
    try std.testing.expect(call - at < 120);
    const server = @embedFile("server.zig");
    const at_conn = std.mem.indexOf(u8, server, "fn handleConnectionThread(") orelse return error.MissingConnThread;
    const call_conn = std.mem.indexOfPos(u8, server, at_conn, "keep_warm.touch();") orelse return error.MissingConnTouch;
    try std.testing.expect(call_conn - at_conn < 120);
}
