//! N-gram index over a request's committed tokens: the continuation of an earlier occurrence of the last
//! committed tokens makes a draft with no model forward. Only committed tokens are indexed.

const std = @import("std");

pub const NGRAM: usize = 4;
/// Shorter context matches land too few drafts to pay for their verify.
pub const MIN_SUFFIX: u32 = 8;
const SUFFIX_CAP: u32 = 64;
/// Earlier occurrences `matchBest` compares.
const CANDIDATES: u32 = 8;

pub const Match = struct {
    /// Borrowed from the index.
    draft: []const u32,
    /// Tokens the two sites agree on going back, n-gram included, capped.
    suffix: u32,
};

pub const Index = struct {
    allocator: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    /// n-gram -> position of the token after its latest occurrence.
    next: std.AutoHashMapUnmanaged(u128, u32) = .empty,
    /// Position of the token after an earlier occurrence of the same n-gram, by position of the later one.
    prev: std.ArrayList(u32) = .empty,

    pub fn init(allocator: std.mem.Allocator) Index {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Index) void {
        self.toks.deinit(self.allocator);
        self.next.deinit(self.allocator);
        self.prev.deinit(self.allocator);
    }

    pub fn extend(self: *Index, ids: []const u32) !void {
        try self.toks.ensureUnusedCapacity(self.allocator, ids.len);
        for (ids) |id| {
            self.toks.appendAssumeCapacity(id);
            const n = self.toks.items.len;
            if (n <= NGRAM) continue;
            const g = self.toks.items[n - 1 - NGRAM .. n - 1];
            const gop = try self.next.getOrPut(self.allocator, key(g[0], g[1], g[2], g[3]));
            try self.prev.resize(self.allocator, n);
            self.prev.items[n - 1] = if (gop.found_existing) gop.value_ptr.* else none;
            gop.value_ptr.* = @intCast(n - 1);
        }
    }

    const none: u32 = std.math.maxInt(u32);

    /// Continuation after the latest earlier occurrence of the last NGRAM-1
    /// committed tokens plus `t1` (not yet committed).
    pub fn match(self: *const Index, t1: u32, max_draft: u32) ?Match {
        const toks = self.toks.items;
        if (toks.len < NGRAM - 1 or max_draft == 0) return null;
        const t = toks[toks.len - (NGRAM - 1) ..];
        const p: usize = self.next.get(key(t[0], t[1], t[2], t1)) orelse return null;
        var suffix: u32 = 0;
        var a: usize = p;
        var b: usize = toks.len + 1;
        while (suffix < SUFFIX_CAP and a > 0) : (suffix += 1) {
            a -= 1;
            b -= 1;
            if (toks[a] != (if (b == toks.len) t1 else toks[b])) break;
        }
        return .{ .draft = toks[p..@min(p + max_draft, toks.len)], .suffix = suffix };
    }

    /// `match` over the last `CANDIDATES` occurrences of the n-gram: the one that agrees furthest back wins,
    /// the latest among equals. An edit that left the latest occurrence on the wrong copy still finds the right one.
    pub fn matchBest(self: *const Index, t1: u32, max_draft: u32) ?Match {
        const toks = self.toks.items;
        if (toks.len < NGRAM - 1 or max_draft == 0) return null;
        const t = toks[toks.len - (NGRAM - 1) ..];
        var p: usize = self.next.get(key(t[0], t[1], t[2], t1)) orelse return null;
        var best: ?Match = null;
        var seen: u32 = 0;
        while (seen < CANDIDATES) : (seen += 1) {
            const suffix = agreement(toks, p, t1);
            if (best == null or suffix > best.?.suffix) best = .{ .draft = toks[p..@min(p + max_draft, toks.len)], .suffix = suffix };
            if (suffix >= SUFFIX_CAP or p >= self.prev.items.len or self.prev.items[p] == none) break;
            p = self.prev.items[p];
        }
        return best;
    }

    /// Tokens before `p` that agree with the committed tail (plus `t1`), capped.
    fn agreement(toks: []const u32, p: usize, t1: u32) u32 {
        var suffix: u32 = 0;
        var a: usize = p;
        var b: usize = toks.len + 1;
        while (suffix < SUFFIX_CAP and a > 0) : (suffix += 1) {
            a -= 1;
            b -= 1;
            if (toks[a] != (if (b == toks.len) t1 else toks[b])) break;
        }
        return suffix;
    }

    fn key(a: u32, b: u32, c: u32, d: u32) u128 {
        return (@as(u128, a) << 96) | (@as(u128, b) << 64) | (@as(u128, c) << 32) | d;
    }
};

const testing = std.testing;

test "lookup: latest earlier continuation, keyed on the tail plus t1, with how far back it agrees" {
    var idx = Index.init(testing.allocator);
    defer idx.deinit();
    // Two earlier "1 2 3 4" sites; the later one agrees back through "0 11 12 13".
    try idx.extend(&.{ 1, 2, 3, 4, 10, 50, 0, 11, 12, 13, 1, 2, 3, 4, 77, 5 });
    try idx.extend(&.{ 60, 0, 11, 12, 13, 1, 2, 3 });
    const m = idx.match(4, 3) orelse return error.NoMatch;
    try testing.expectEqualSlices(u32, &.{ 77, 5, 60 }, m.draft);
    try testing.expectEqual(@as(u32, 8), m.suffix);
    try testing.expect(idx.match(9, 3) == null);
}

test "lookup: matchBest takes the earlier occurrence that agrees further back over the latest one" {
    var ix = Index.init(testing.allocator);
    defer ix.deinit();
    try ix.extend(&.{ 9, 9, 9, 1, 2, 3, 4, 50, 51, 52, 53, 54, 7, 1, 2, 3, 4, 60, 61, 70, 9, 9, 9, 1, 2, 3 });
    const latest = ix.match(4, 4).?;
    try testing.expectEqual(@as(u32, 60), latest.draft[0]);
    try testing.expectEqual(@as(u32, 4), latest.suffix);
    const best = ix.matchBest(4, 4).?;
    try testing.expectEqual(@as(u32, 50), best.draft[0]);
    try testing.expectEqual(@as(u32, 7), best.suffix);
    // Alone, or with the latest already best, it is the latest.
    var one = Index.init(testing.allocator);
    defer one.deinit();
    try one.extend(&.{ 1, 2, 3, 4, 5, 6, 1, 2, 3 });
    try testing.expectEqual(one.match(4, 3).?.draft[0], one.matchBest(4, 3).?.draft[0]);
}
