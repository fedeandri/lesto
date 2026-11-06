//! Machine facts the engine reads from the OS: total RAM and the chip's brand string.
const std = @import("std");

/// Total physical RAM (`hw.memsize`); 0 when the read fails (gates that consume this must
/// treat 0 as "unknown", never as "tiny machine").
pub fn totalMemBytes() u64 {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    _ = std.c.sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    return mem;
}

/// Chip name via sysctl ("Apple M3 Ultra"); empty on failure, so callers fall
/// to their default row. The GPU arch string cannot tell Ultra from Max, hence the CPU brand.
var chip_brand_buf: [128]u8 = undefined;
var chip_brand_len: usize = 0;
/// 0 = unread, 1 = one thread is reading it, 2 = published.
var chip_brand_state = std.atomic.Value(u8).init(0);

/// Cached `machdep.cpu.brand_string`: THE accessor for every per-silicon
/// table (speculative depth cap, DFlash block cap). The chip cannot change
/// under a running process and these are read on request-shaped paths
/// (Generator init), so the sysctl runs exactly once. Callers that need to
/// inject a chip string take it as a parameter; nobody re-wraps this.
pub fn chipBrand() []const u8 {
    if (chip_brand_state.load(.acquire) != 2) {
        if (chip_brand_state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
            chip_brand_len = chipBrandString(&chip_brand_buf).len;
            chip_brand_state.store(2, .release);
        } else {
            while (chip_brand_state.load(.acquire) != 2) std.atomic.spinLoopHint();
        }
    }
    return chip_brand_buf[0..chip_brand_len];
}

pub fn chipBrandString(buf: []u8) []const u8 {
    var len: usize = buf.len;
    if (std.c.sysctlbyname("machdep.cpu.brand_string", buf.ptr, &len, null, 0) != 0) return "";
    if (len > 0 and buf[len - 1] == 0) len -= 1;
    return buf[0..len];
}
