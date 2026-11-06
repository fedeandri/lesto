//! Plan 03 — hot prefix cache (Phase 1).
//!
//! Replaces the legacy single-slot `cached_prompt_ids` with a small bounded
//! LRU keyed by `(prompt_ids ++ generated_ids, has_tools)`. Each entry owns a
//! `KVCacheSnapshot` of the live cache at the moment the request finished —
//! refcount-shared handles point at the GPU buffers that filled positions
//! 0..len. On a new request we longest-prefix match across entries, restore
//! the best one back into `xfm.cache`, and let the existing
//! truncate-then-prefill path handle the diverged tail.
//!

const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const restore_dump = @import("restore_dump.zig");

const KVCache = transformer_mod.KVCache;
const KVCacheSnapshot = transformer_mod.KVCacheSnapshot;

/// Minimum forwarded-prefix length for committing a CANCELLED prefill
/// (client disconnect mid-prefill). Below this an entry is LRU pollution —
/// chat-template prologues (Gemma=12 tokens, other families 4 to 8) are identical
/// across every request and "reusable" only in a worthless sense. Same
/// rationale as the llama session pool's `min_prefix_to_claim`, applied at
/// commit time instead of claim time.
pub const MIN_CANCELLED_COMMIT_TOKENS: usize = 256;

/// Media span identity lives at the disk tier (its v9 records carry spans);
/// re-exported here — the RAM tier and every caller speak this name.
pub const MediaSpan = kv_disk_cache.MediaSpan;
pub const mediaSharedBound = kv_disk_cache.mediaSharedBound;
pub const spansBelow = kv_disk_cache.spansBelow;

fn dupeEntryKeys(allocator: std.mem.Allocator, tokens: []const u32, media: []const MediaSpan) !struct { []u32, []MediaSpan } {
    const t = try allocator.dupe(u32, tokens);
    errdefer allocator.free(t);
    return .{ t, try allocator.dupe(MediaSpan, media) };
}

/// Why an oversized commit retained nothing; three outcomes used to print one identical line.
pub const TrimDecline = enum {
    no_restorable_prefix,
    snapshot_copy_failed,

    pub fn reason(self: TrimDecline) []const u8 {
        return switch (self) {
            .no_restorable_prefix => "no restorable prefix fits the budget",
            .snapshot_copy_failed => "every trimmed KV copy failed",
        };
    }
};

/// Outcome of a cache commit, so callers can log the truth. A budget
/// decline used to exit `commitWithMediaState` via plain return and the
/// scheduler logged "committed N/M" for it (live 2026-09-07: a 122k agent
/// session cold-prefilled ~95k tokens per retry while the log claimed
/// cancelled-prefill commits every time).
pub const CommitStatus = union(enum) {
    /// Entry committed (inserted or replaced) at this many tokens — the
    /// post-trim EFFECTIVE length, never the candidate's forwarded length.
    ok: usize,
    /// Full state captured for the SSD tier; no idle RAM entry retained.
    disk_only: usize,
    /// The budget decline kept a resident entry that already covers this
    /// many tokens; the longer candidate was discarded (details logged).
    kept_resident: usize,
    /// Declined: oversized with no restorable trim under the budget, or the
    /// trim copy failed. The decline site logs the reason.
    declined,
};

/// Result of a cache lookup. Tells the caller how many tokens of `prompt_ids`
/// are already in the live cache after a successful restore — the caller
/// then prefills only the trailing diverged tokens (`prompt_ids[matched..]`).
pub const LookupResult = struct {
    /// Tokens already in the live cache. Caller prefills `prompt_ids[matched..]`.
    matched: usize,
    /// Did the restore land on an entry whose tokens span the FULL new prompt?
    /// Then identical-re-issue logic kicks in (truncate to len-1 and re-forward
    /// the last token), matching the existing reuseKVCache behavior.
    full_match: bool,
    /// Non-null iff a DFlash assistant context was restored into the caller's
    /// target: the absolute trunk position its index 0 represents, with
    /// `base + cache.step == matched` on return. Both tiers serve it (the
    /// SSD tier persists the snapshot in the v4 spec sidecar). Null on EVERY
    /// other path (no target, no payload, miss) and the target is untouched —
    /// the caller then starts the assistant blind at `matched`.
    dflash_base: ?usize = null,
    /// Did this restore check out its entry (`checkoutEligible`)? Only then does the first
    /// append donate in place; every other restore is a refcount share copied by that append.
    checked_out: bool = false,
    /// Did a disk restore hand the slot rows it owns outright? The restore copies its chunks
    /// into slot-held arrays, so those rows are the slot's own without a RAM entry moving.
    slot_owned: bool = false,
    /// `Entry.id` of the RAM entry restored from; 0 = none.
    entry_id: u64 = 0,
};

const Entry = struct {
    /// `prompt_ids ++ generated_ids` from the request that produced this snapshot.
    /// Owned by the entry; freed on eviction.
    tokens: []u32,
    /// Whether the request had tools enabled (different chat template, can't
    /// share cache across).
    has_tools: bool,
    /// Media items inside `tokens`, ascending, owned. Entries with media stay in
    /// RAM (never spilled to the SSD tier).
    media: []MediaSpan = &.{},
    /// Workload the request belonged to (`server.requestCacheKey`, 0 = anonymous).
    /// Eviction is fair across keys: the key holding the most entries pays first.
    cache_key: u64 = 0,
    /// Stable identity for observers; `last_used` moves on every touch.
    id: u64 = 0,
    /// Snapshot of the live KVCache at end of generation. Owns refcount-shared
    /// handles to the GPU buffers backing positions 0..tokens.len.
    snapshot: KVCacheSnapshot,
    /// Monotonic counter for LRU. Higher = more recent.
    last_used: u64,
    /// Wave 1.A: full KV-quant config active when this entry was committed.
    /// A new request whose `KVQuantConfig` differs in any field cannot
    /// restore from this entry — the underlying buffer layout (dense bf16 vs
    /// packed uint32 triples; 4-bit vs 8-bit packing) differs, and
    /// dequantization would have happened at commit time anyway. Filter at
    /// lookup so per-request `kv_quant` overrides never produce a hit
    /// against an entry that was committed under another config.
    ///
    /// Storing the full `KVQuantConfig` (not just `Scheme`) is what
    /// distinguishes `affine 4` from `affine 8` and a `group_size` change —
    /// without that, a 4-bit entry would alias to an
    /// 8-bit slot's findBestMatch lookup and crash SDPA with a packed-shape
    /// mismatch on restore. Repro: `tests/test_kv_quant_per_request.sh`.
    quant_config: kv_quant.KVQuantConfig,
    /// Transient, one `spillIdleEntries` pass only: did this pass leave a durable copy on the SSD tier?
    spill_durable: bool = false,
    /// The SSD tier does not hold this entry whole yet; the idle flush owes it slices.
    disk_dirty: bool = false,
    /// KV-resident bytes for this entry, computed at commit time (Wave 1.B).
    /// Used for `--prefix-cache-mem` memory-budget enforcement; sum across
    /// all entries == `current_kv_bytes`.
    kv_bytes: u64,
    /// DFlash assistant context for this prefix (dflash.zig). The assistant's
    /// K/V is built from trunk hiddens at `target_layer_ids`, and a restore
    /// forwards NOTHING — so without this the assistant starts every reused
    /// turn blind and drafts against an empty context. Optional in both
    /// directions: an entry committed by a non-dflash request has none, and a
    /// request that finds none simply starts blind (the state is DRAFT-side —
    /// a missing or stale context costs acceptance, never a token).
    dflash: ?DflashSnap = null,
    /// Bytes resident in `dflash`, folded into `kv_bytes`.
    dflash_bytes: u64 = 0,
    /// Restore by move: the slot that took ownership of this entry's KV buffers (`KVCache.adopt`).
    /// While set the snapshot holds empty handles and the entry is invisible to every other
    /// reader. Cleared by the commit that replaces it; dropped at slot end otherwise.
    checked_out_by: ?usize = null,
    /// Has the slot's append donated these buffers in place (`donateCheckout`)? Until then the
    /// entry's handles are live and a slot that ends hands the entry back; only a donated checkout is dropped.
    checkout_donated: bool = false,
};

/// What a spec-snap adoption may do, decided before any mlx call.
pub const SpecAdopt = union(enum) {
    /// Nothing to adopt: no payload, or the snap does not cover the reused range.
    skip,
    /// A KV-only spec cache (the dflash context).
    kv_only: usize,
};

pub fn specAdoptPlan(base_pos: usize, snap_step: usize, matched: usize) SpecAdopt {
    if (base_pos > matched) return .skip;
    const want = matched - base_pos;
    if (want > snap_step) return .skip;
    return .{ .kv_only = want };
}

var ssd_first_env_cached: ?bool = null;
/// Test/bench override for `ssdFirstEnabled()`. Null = read the environment.
pub var ssd_first_override: ?bool = null;

/// Resolve this module's lazily-cached env reads once, from the main thread.
pub fn warmEnvCaches() void {
    _ = ssdFirstEnabled();
    _ = restoreMoveEnabled();
}

/// SSD-first prefix cache mode (`MLX_SERVE_PREFIX_SSD_FIRST=0` restores RAM-first); armed only
/// where `ModelConfig.ssdFirstCapable()`.
pub fn ssdFirstEnabled() bool {
    if (ssd_first_override) |v| return v;
    if (ssd_first_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_PREFIX_SSD_FIRST") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    ssd_first_env_cached = v;
    return v;
}

var restore_move_env_cached: ?bool = null;
/// Test/bench override for `restoreMoveEnabled()`. Null = read the environment.
pub var restore_move_override: ?bool = null;

/// Restore by move. `MLX_SERVE_RESTORE_MOVE=0` restores the refcount share whose first append
/// copies the whole prefix. Armed where `HotPrefixCache.ssd_first` is, or on demand when a share
/// does not fit (`checkoutRestored`).
pub fn restoreMoveEnabled() bool {
    if (restore_move_override) |v| return v;
    if (restore_move_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_RESTORE_MOVE") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    restore_move_env_cached = v;
    return v;
}

/// What the shutdown flush did, for its one summary line.
pub const ShutdownTally = struct {
    /// Records a shutdown slice brought whole (a record already covered on disk is not counted).
    written: u32 = 0,
    /// Bytes the shutdown slices wrote (staged, on the writer arm).
    bytes: u64 = 0,
    /// Records still owed when the deadline stopped the flush.
    left_deadline: u32 = 0,
    /// Records the tier refused because the volume is under its reserve.
    left_declined: u32 = 0,
    /// Owed entries an in-flight request still held.
    checked_out: u32 = 0,
    /// Writes that failed during the flush (the entry is poisoned; the flush moved on).
    failures: u64 = 0,
    /// Staged bytes still queued at the deadline, dropped instead of written.
    dropped_bytes: u64 = 0,
};

/// One resident model's hot cache and its stream: what a disk flush runs on.
pub const FlushTarget = struct { hc: *HotPrefixCache, s: mlx.mlx_stream };

/// SIGTERM: finish what the idle flush has not landed, oldest owed first, one slice per model in
/// turn, then let the staged writes land, all before `deadline_ms` on `clock` (`nowMs() i64`).
/// A slice that the longest one so far says would overrun the deadline is not started.
/// Inference thread only.
pub fn shutdownPersist(targets: []const FlushTarget, clock: anytype, deadline_ms: i64) ShutdownTally {
    var t: ShutdownTally = .{};
    var errors_before: u64 = 0;
    for (targets) |x| {
        if (x.hc.disk) |*d| errors_before += d.writeErrors();
    }
    var longest: i64 = 0;
    var cut = false;
    while (!cut) {
        var sliced = false;
        for (targets) |x| {
            if (!x.hc.diskOwed()) continue;
            const start = clock.nowMs();
            if (start >= deadline_ms or start + longest > deadline_ms) {
                cut = true;
                break;
            }
            x.hc.shutdownSlice(x.s, &t);
            longest = @max(longest, clock.nowMs() - start);
            sliced = true;
        }
        if (!sliced) break;
    }
    var errors_after: u64 = 0;
    for (targets) |x| {
        const d = if (x.hc.disk) |*dd| dd else continue;
        if (cut) t.left_deadline += x.hc.owedCount();
        t.checked_out += x.hc.checkedOutOwed();
        t.dropped_bytes += d.drainWriterUntil(clock, deadline_ms);
        errors_after += d.writeErrors();
    }
    t.failures += errors_after -| errors_before;
    return t;
}

/// The SSD-first predicate: arch, env switch, AND a disk tier. Without the tier the mode used
/// to arm with nowhere to spill and a budget floor sized for a tier that did not exist. The
/// budget resolver asks `--prefix-cache-disk > 0`, the arming asks `disk != null`.
pub fn ssdFirstActive(config: *const model_mod.ModelConfig, has_disk: bool, ram_enabled: bool) bool {
    return has_disk and ssdFirstEnabled() and (config.ssdFirstCapable() or !ram_enabled);
}

/// What the live cache held at commit time, captured before the RAM byte-budget trim: the
/// disk tier gets the full prefix even when RAM keeps a trimmed one. Refcount-shared.
const PendingDiskFlush = struct {
    snapshot: KVCacheSnapshot,
    tokens: []u32,
    has_tools: bool,
    /// v9: the media spans the token record was committed under (owned).
    media: []MediaSpan = &.{},
    dflash: ?DflashSnap = null,

    fn deinit(self: *PendingDiskFlush, allocator: std.mem.Allocator) void {
        self.snapshot.deinit();
        allocator.free(self.tokens);
        allocator.free(self.media);
        if (self.dflash) |*d| d.deinit();
    }
};

/// A committed speculative-side cache: the snapshot plus the absolute trunk
/// position its index 0 represents (nonzero when the committing request was
/// itself a cache hit). The DFlash assistant context.
pub const DflashSnap = struct {
    snapshot: KVCacheSnapshot,
    base_pos: usize,

    pub fn deinit(self: *DflashSnap) void {
        self.snapshot.deinit();
    }
};

/// What `commitWithState` reads to build a `DflashSnap`.
pub const DflashCommit = struct {
    cache: *const KVCache,
    base_pos: usize,
};

/// Where `lookupAndRestore` puts a restored assistant context. `base_pos` is
/// written on every path so the caller can build `DflashCtx` from it.
pub const DflashTarget = struct {
    cache: *KVCache,
    base_pos: *usize,
};

/// What `evictLruToAdmit` gave up, and whether it was enough.
pub const EvictionReport = struct {
    entries: usize = 0,
    /// Bytes the allocator actually got back (live delta).
    bytes: u64 = 0,
    /// Bytes the cache had billed for those entries; larger whenever a snapshot is refcount-shared.
    accounted_bytes: u64 = 0,
    /// The pass stopped because an eviction returned nothing (shared with a live request).
    shared_stop: bool = false,
    /// False = the cache is empty (or down to this request's entry) and the request still does not fit.
    admitted: bool = false,
};

pub const HotPrefixCache = struct {
    entries: std.ArrayList(Entry),
    max_entries: u32,
    /// Wave 1.B: total KV bytes the cache is allowed to keep resident across
    /// all entries. 0 disables the byte budget (count cap still applies).
    /// Enforced on `commit`: evict LRU entries (in addition to the count
    /// cap) until `current_kv_bytes + new_entry_bytes <= max_kv_bytes`.
    max_kv_bytes: u64,
    /// Running total of `kv_bytes` across all live entries. Updated on
    /// commit/evict/invalidate.
    current_kv_bytes: u64,
    allocator: std.mem.Allocator,
    counter: u64 = 0,
    /// Set to true once we've called `xfm.resetCache()` at least once after
    /// init. The first commit on a fresh cache must seed an empty entry so
    /// future restores have something to land on.
    initialized: bool = false,
    /// SSD tier (kv_disk_cache.zig). Attached by the scheduler at model load
    /// when `--prefix-cache-disk` is non-zero and the arch is pure-attention.
    /// Lookup falls back to it when it beats the RAM match; commits mark
    /// `disk_dirty` and the scheduler flushes AFTER the response finishes so
    /// the client never waits on the SSD write.
    disk: ?kv_disk_cache.DiskTier = null,
    /// A commit landed since the last `flushPendingDisk`.
    disk_dirty: bool = false,
    /// `last_used` of the entry the current request restored from; `evictLruToAdmit` refuses to evict it.
    last_restored_used: ?u64 = null,
    /// The raw prefix that restore matched (`checkoutRestored`).
    last_restored_shared: usize = 0,
    last_restored_disk_id: ?u64 = null,
    /// Whether completed requests retain reusable KV in RAM. The live slot still owns its
    /// working KV; false keeps reusable prefixes only on the SSD tier.
    ram_enabled: bool = true,
    /// SSD-first mode; set by the scheduler at load.
    ssd_first: bool = false,
    /// SSD-first: the RAM allowance for idle entries (the resolved `--prefix-cache-mem`).
    /// `spillIdleEntries` evicts only past it; 0 = nothing idle stays resident.
    ssd_idle_mem: u64 = 0,
    /// The live-cache state of the most recent commit, flushed instead of the (possibly trimmed) RAM entry.
    pending_disk: ?PendingDiskFlush = null,
    /// An earlier session's pending snapshot a newer commit displaced before it landed whole;
    /// flushes finish it first, one slice each, so no commit pays for its write.
    finishing_disk: ?PendingDiskFlush = null,
    /// Byte floor for `restoreWouldPinEntry`; a field so a test can reproduce the live shape.
    restore_pin_min_bytes: u64 = RESTORE_PIN_MIN_BYTES,

    pub fn init(allocator: std.mem.Allocator, max_entries: u32) HotPrefixCache {
        return initWithMem(allocator, max_entries, 0);
    }

    pub fn initWithMem(allocator: std.mem.Allocator, max_entries: u32, max_kv_bytes: u64) HotPrefixCache {
        return .{
            .entries = std.ArrayList(Entry).empty,
            .max_entries = max_entries,
            .ram_enabled = max_entries > 0,
            .max_kv_bytes = max_kv_bytes,
            .current_kv_bytes = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HotPrefixCache) void {
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.deinit(self.allocator);
        self.dropPendingSnapshots();
        if (self.disk) |*d| d.deinit();
        self.disk = null;
    }

    fn dropPendingSnapshots(self: *HotPrefixCache) void {
        if (self.pending_disk) |*p| p.deinit(self.allocator);
        self.pending_disk = null;
        if (self.finishing_disk) |*p| p.deinit(self.allocator);
        self.finishing_disk = null;
    }

    /// Free everything an Entry owns: token buffer, KV snapshot, DFlash
    /// context. Used by `deinit`, eviction, and replace paths so they don't
    /// drift apart.
    fn freeEntryOwnedState(allocator: std.mem.Allocator, e: *Entry) void {
        allocator.free(e.tokens);
        allocator.free(e.media);
        e.snapshot.deinit();
        if (e.dflash) |*d| {
            d.deinit();
            e.dflash = null;
        }
    }

    /// Restore a committed speculative-side cache (the DFlash context) into the caller's cache, clamped to the trunk's restored
    /// length. Returns the base position on success, null when there is
    /// nothing to restore, the snap starts PAST what the trunk actually
    /// reused, or the snap ends BEFORE it (a history with a gap right below
    /// the generation point is worse than a blind start). Best-effort by
    /// contract: a failure leaves the caller blind, never wrong.
    fn restoreSpecSnap(snap_opt: ?*const DflashSnap, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream, what: []const u8) ?usize {
        const t = target orelse return null;
        const snap = snap_opt orelse return null;
        const want = switch (specAdoptPlan(snap.base_pos, snap.snapshot.step, matched)) {
            .skip => {
                log.info("  [hot-cache] {s} not adopted: want {d} tokens from base {d}, snap holds {d} (matched {d})\n", .{ what, matched -| snap.base_pos, snap.base_pos, snap.snapshot.step, matched });
                return null;
            },
            .kv_only => |w| w,
        };
        t.cache.restore(&snap.snapshot) catch |err| {
            log.warn("  [hot-cache] {s} restore failed: {s} — starts blind\n", .{ what, @errorName(err) });
            return null;
        };
        t.cache.truncate(want, s) catch |err| {
            log.warn("  [hot-cache] {s} clamp failed: {s} — starts blind\n", .{ what, @errorName(err) });
            return null;
        };
        t.base_pos.* = snap.base_pos;
        log.debug("  [hot-cache] {s} restored: {d} tokens from base {d}\n", .{ what, want, snap.base_pos });
        return snap.base_pos;
    }

    fn restoreDflash(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
        return restoreSpecSnap(if (e.dflash) |*d| d else null, target, matched, s, "dflash context");
    }

    /// Disk-tier variant: load the persisted spec snapshot (v4 sidecar) as a
    /// transient and adopt it under the EXACT same clamp rule as the RAM
    /// tier (`restoreSpecSnap`). The trunk restore already forwarded nothing,
    /// so without this a disk hit drafted blind — the 92.6% → 66.5%
    /// acceptance class the RAM tier already fixed.
    fn diskRestoreSpec(
        d: *kv_disk_cache.DiskTier,
        idx: usize,
        target: ?DflashTarget,
        matched: usize,
        s: mlx.mlx_stream,
        which: kv_disk_cache.SpecKind,
    ) ?usize {
        const t = target orelse return null;
        const loaded = d.loadSpecSnap(idx, which, t.cache.entries.len, t.cache.config) orelse return null;
        // restore() refcount-shares the arrays into the target, so the
        // transient snapshot is freed right after.
        var snap = DflashSnap{
            .snapshot = loaded.snap,
            .base_pos = loaded.base,
        };
        defer snap.deinit();
        return restoreSpecSnap(&snap, target, matched, s, switch (which) {
            .dflash => "dflash context",
        });
    }

    fn bumpCounter(self: *HotPrefixCache) u64 {
        self.counter += 1;
        return self.counter;
    }

    /// Wave 1.B: total KV bytes held by a snapshot — sum of `size * itemsize`
    /// across every initialized entry's storage arrays. mlx-c arrays carry
    /// their shape + dtype so this is exact, not a heuristic. Quant schemes
    /// account for q, scales, biases together.
    fn snapshotBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized) continue;
            total += @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
            total += @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
            if (snap.config.scheme != .off) {
                total += @as(u64, mlx.mlx_array_size(e.keys_scales)) * @as(u64, mlx.mlx_array_itemsize(e.keys_scales));
                total += @as(u64, mlx.mlx_array_size(e.keys_biases)) * @as(u64, mlx.mlx_array_itemsize(e.keys_biases));
                total += @as(u64, mlx.mlx_array_size(e.values_scales)) * @as(u64, mlx.mlx_array_itemsize(e.values_scales));
                total += @as(u64, mlx.mlx_array_size(e.values_biases)) * @as(u64, mlx.mlx_array_itemsize(e.values_biases));
            }
        }
        return total;
    }

    /// Issue #330: per-token bytes of a snapshot — what one retained token
    /// costs after a `trimmedCopy` materializes exactly `len` rows. Derived
    /// from each array's own shape (bytes / capacity rows), so it prices
    /// quantized triples correctly too.
    fn snapshotRowBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized) continue;
            inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                // A dense snapshot leaves the quant handles as empty 0-dim arrays; the ndim check makes axis 2 a fact.
                if (arr.ctx != null and mlx.mlx_array_ndim(arr) > 2) {
                    const rows: u64 = @intCast(mlx.mlx_array_shape(arr)[2]);
                    if (rows > 0) {
                        total += (@as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr))) / rows;
                    }
                }
            }
        }
        return total;
    }

    /// Issue #330: the longest retainable prefix length under `budget`, or
    /// null when nothing at or above the commit floor fits. Plain attention
    /// restores at any length. `limit` caps the answer.
    fn trimLenForBudget(
        budget: u64,
        limit: usize,
        row_bytes: u64,
    ) ?usize {
        if (row_bytes == 0) return null;
        const fit: usize = @intCast(budget / row_bytes);
        const len = @min(fit, limit);
        if (len < MIN_CANCELLED_COMMIT_TOKENS) return null;
        return len;
    }

    /// Find the entry with the longest restorable prefix shared
    /// with `prompt_ids` and matching `(has_tools, quant_config)`. Returns the entry index and raw
    /// shared-prefix length; null if no entry matches the key. Wave 1.A:
    /// the config filter exists because cross-config buffer layouts differ
    /// — a slot running `kv_quant=4` cannot restore from an entry committed
    /// in dense (or 8-bit) mode and vice versa. The full `KVQuantConfig`
    /// (scheme + bits + group_size) is compared because `Scheme.affine`
    /// covers BOTH 4-bit and 8-bit packings: filtering on `Scheme` alone
    /// would let a 4-bit entry alias to an 8-bit slot and crash SDPA on
    /// restore. The match stops at the first media item whose position or
    /// pixels differ (`mediaSharedBound`). See
    /// `tests/test_kv_quant_per_request.sh`.
    fn findBestRestorableMatch(
        self: *const HotPrefixCache,
        prompt_ids: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        quant_config: kv_quant.KVQuantConfig,
    ) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            // A checked-out entry's buffers belong to another slot; its snapshot is empty.
            if (e.checked_out_by != null) continue;
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant_config, quant_config)) continue;

            const max_shared = @min(e.tokens.len, prompt_ids.len, mediaSharedBound(e.media, media));
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;

            if (!ringRestores(&e.snapshot, shared, prompt_ids.len)) continue;
            if (shared > best_shared) {
                best_shared = shared;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .shared = best_shared };
        return null;
    }

    /// Tokens a RAM restore would cover for a text prompt, read without touching any state. A
    /// pre-filter only: the SSD tier, a lien, a QSA gap or a checkout taken since can still change
    /// what the real lookup restores.
    pub fn peekMatchedTokens(self: *const HotPrefixCache, prompt_ids: []const u32, has_tools: bool, quant_config: kv_quant.KVQuantConfig) usize {
        const m = self.findBestRestorableMatch(prompt_ids, has_tools, &.{}, quant_config) orelse return 0;
        return m.shared;
    }

    /// A sliding ring restores only where it still holds the window behind the match; a full
    /// match restores one token short.
    fn ringRestores(snap: *const KVCacheSnapshot, matched: usize, prompt_len: usize) bool {
        return matched - @intFromBool(matched == prompt_len and matched > 0) >= snap.ringFloor();
    }

    fn findBestMatch(self: *const HotPrefixCache, prompt_ids: []const u32, has_tools: bool, media: []const MediaSpan, quant_config: kv_quant.KVQuantConfig) ?struct { idx: usize, shared: usize } {
        const match = self.findBestRestorableMatch(prompt_ids, has_tools, media, quant_config) orelse return null;
        return .{ .idx = match.idx, .shared = match.shared };
    }

    /// Try to restore a matching entry into `target_cache`. On success, returns
    /// the matched prefix length. On miss, fully resets `target_cache` and
    /// returns 0. Caller should prefill the trailing tokens after this.
    ///
    /// The `target_*` parameters generalize the legacy single-slot path
    /// (`xfm.cache`, `xfm.moe_seq_offset`, `xfm.ssm_entries`) so Phase 2
    /// per-slot caches can reuse the same restore machinery.
    pub fn lookupAndRestore(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash_target: ?DflashTarget,
    ) !LookupResult {
        return self.lookupAndRestoreWithMedia(target_cache, target_moe_seq_offset, s, prompt_ids, has_tools, media, dflash_target, null, false);
    }

    /// The checkout-capable entry point: `slot_id` names the slot that will own the restored buffers.
    pub fn lookupAndRestoreForSlot(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash_target: ?DflashTarget,
        slot_id: usize,
    ) !LookupResult {
        return self.lookupAndRestoreWithMedia(target_cache, target_moe_seq_offset, s, prompt_ids, has_tools, media, dflash_target, slot_id, false);
    }

    pub fn lookupAndRestoreWithMedia(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash_target: ?DflashTarget,
        /// Restore by move: non-null opts this request into the checkout (the caller promises
        /// `releaseCheckout` on every path that ends the slot).
        slot_id: ?usize,
        skip: bool,
    ) !LookupResult {
        self.last_restored_used = null;
        self.last_restored_disk_id = null;
        if (skip) {
            try target_cache.truncate(0, s);
            target_moe_seq_offset.* = 0;
            return .{ .matched = 0, .full_match = false };
        }
        const match = self.findBestRestorableMatch(
            prompt_ids,
            has_tools,
            media,
            target_cache.config,
        );

        // ── SSD tier: consult when it can beat the RAM match meaningfully
        // (fresh boot, post-eviction).
        if (self.disk) |*d| disk: {
            // v9: the tier's records are (tokens, spans). The clamp moved
            // per-entry — `bestMatch` bounds each candidate at
            // `mediaSharedBound(entry.media, media)`: a request whose spans
            // agree reads its media rows back, one that diverges stops below
            // the first item whose start or pixel key differs, and a text turn
            // clamps at every media entry's first item exactly like the old
            // tier-wide `disk_limit` did. A row AT a placeholder is restored
            // only under an agreeing span list — the same pixels, so the
            // restored row IS what the vision splice would have produced.
            const dm = d.bestMatch(prompt_ids, has_tools, target_cache.config, media) orelse {
                // Silent no-entry misses are why a 40 GB disk tier looked
                // dead in a live post-mortem (2026-09-07): nothing in the
                // log ever said the tier was consulted and found nothing.
                if (d.entryCount() > 0)
                    log.debug("  [disk-cache] lookup miss: {d} entries on disk, none share a token prefix under this (tools, quant) key\n", .{d.entryCount()});
                break :disk;
            };
            // `dm.usable` is already clamped to the entry's kv_len (see
            // DiskTier.bestMatch) and to the per-entry media bound.
            const usable: u32 = dm.usable;

            const ram_len: usize = if (match) |m| m.shared else 0;
            if (usable <= ram_len) break :disk;
            if (usable - ram_len < kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
            // `usable` is `dm.usable` clamped to the entry's kv_len (see
            // DiskTier.bestMatch) and to the request's media boundary, so it
            // IS the restorable length — restore
            // only the chunks covering it. Loading the whole entry
            // (restoreInto) would read a long stored prefix in full to serve a
            // short shared prefix, making a diverged-prefix "hit" slower than a
            // cold miss.
            const effective: usize = usable;
            const full_match = effective == prompt_ids.len;
            const final_len: usize = if (full_match and effective > 1) effective - 1 else effective;
            const sw = io_util.Stopwatch.init(d.io);
            d.restorePrefixInto(target_cache, dm.idx, @intCast(final_len), s) catch |err| {
                log.warn("  [disk-cache] restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                if (err == error.MlxError) _ = d.poisonId(d.entries.items[dm.idx].id, @errorName(err));
                // A failed restore can leave a half-rebuilt cache; reset it.
                target_cache.truncate(0, s) catch {};
                break :disk;
            };
            target_moe_seq_offset.* = final_len;
            const ms = sw.read() / std.time.ns_per_ms;
            log.info("  [disk-cache] restored {d}/{d} tokens from SSD ({d} chunks) in {d}ms (entry e{d}, match {d})\n", .{ final_len, prompt_ids.len, d.chunks_loaded_last, ms, d.entries.items[dm.idx].id, dm.usable });
            return .{
                .matched = final_len,
                .full_match = full_match,
                .dflash_base = diskRestoreSpec(d, dm.idx, dflash_target, final_len, s, .dflash),
                .slot_owned = true,
            };
        }

        if (match == null) {
            try target_cache.truncate(0, s);
            target_moe_seq_offset.* = 0;
            return .{ .matched = 0, .full_match = false };
        }
        const m = match.?;
        const e = &self.entries.items[m.idx];
        // Decline a restore that is a lien (`restoreWouldPinEntry`), before the bump and before
        // `last_restored_used`; weighed on the share. SSD-first only.
        if (self.ssd_first and restoreWouldPinEntry(e.kv_bytes, self.restore_pin_min_bytes, e.tokens.len, m.shared)) {
            try target_cache.truncate(0, s);
            target_moe_seq_offset.* = 0;
            log.info("  [hot-cache] declined a {d}-token restore from a {d}-token entry ({d} MB): the share is a lien on the whole entry; cold prefill\n", .{
                m.shared,
                e.tokens.len,
                e.kv_bytes / (1024 * 1024),
            });
            return .{ .matched = 0, .full_match = false };
        }
        // A full reuse re-forwards the last token; a one-token prompt has nothing before it.
        if (prompt_ids.len == 1) {
            try target_cache.truncate(0, s);
            target_moe_seq_offset.* = 0;
            return .{ .matched = 0, .full_match = false };
        }
        const restore_cap: usize = m.shared;
        const used_before_restore = e.last_used;
        e.last_used = self.bumpCounter();
        // Identity of the entry this request runs on: evicting it frees nothing (shared buffers).
        self.last_restored_used = e.last_used;
        self.last_restored_shared = m.shared;

        // The sole caller reads an error as "no match" and cold-prefills the whole prompt, so a
        // failed restore must hand back an EMPTY cache, never a half-bound one.
        errdefer {
            target_cache.truncate(0, s) catch {};
            target_moe_seq_offset.* = 0;
            self.last_restored_used = null;
        }
        try target_cache.restore(&e.snapshot);

        var dump = restore_dump.RestoreDumpMeta{ .kind = "restore", .pos = 0 };
        var effective_matched: usize = restore_cap;
        if (effective_matched > 0 and !ringRestores(&e.snapshot, effective_matched, prompt_ids.len)) {
            log.info("  [hot-cache] sliding ring holds rows from {d}; match {d} falls behind them\n", .{ e.snapshot.ringFloor(), effective_matched });
            effective_matched = 0;
        }
        target_moe_seq_offset.* = effective_matched;

        // Miss path (a sliding ring past the match): also reset KV.
        if (effective_matched == 0) {
            try target_cache.truncate(0, s);
            // A 0-token outcome is not a restore: the marker was set above the restore and would
            // otherwise shield a fully reclaimable entry from the admission pass. The LRU hand-back
            // takes the same `ssd_first` gate as the lien decline.
            self.last_restored_used = null;
            if (self.ssd_first) e.last_used = used_before_restore;
            return .{ .matched = 0, .full_match = false };
        }

        const full_match = effective_matched == prompt_ids.len;
        const final_len: usize = if (full_match and effective_matched > 1) effective_matched - 1 else effective_matched;

        // ALWAYS clamp the restored cache to the matched length. The old guard
        // (`final_len < e.tokens.len`) skipped this on a WHOLE-entry match — but a
        // snapshot can be committed with a KV buffer LONGER than its logical token
        // count: PLD/speculative decode leaves stale draft positions in the buffer
        // past the committed step, and `commit` snapshots them. Restoring that
        // (then skipping the truncate) left `cache.offset` AHEAD of the matched
        // length that generation tracks (`moe_seq_offset`) — a silent drift that
        // corrupts RoPE positions and CRASHES the Gemma sliding-window prefill mask
        // (`broadcast_shapes` mask-vs-KV mismatch; live 2026-07-09 on
        // gemma-4-26B-A4B at ~16K ctx). truncate is a no-op when the buffer is
        // already `final_len`, so unconditional clamping is safe and restores the
        // invariant cache.offset == matched. The stale KV tail has no matching
        // token id, so the match can never reach into it — discarding it is correct.
        try target_cache.truncate(final_len, s);

        const full_reuse = full_match and effective_matched > 1;
        const matched = if (full_reuse) effective_matched - 1 else effective_matched;
        if (full_reuse) {
            target_moe_seq_offset.* = matched;
            log.info("  [hot-cache] full reuse {d}/{d}, re-forwarding last token\n", .{ matched, prompt_ids.len });
        } else {
            log.info("  [hot-cache] reused {d}/{d} tokens (matched {d}; entry {d}/{d})\n", .{ effective_matched, prompt_ids.len, m.shared, m.idx + 1, self.entries.items.len });
        }
        var res: LookupResult = .{
            .matched = matched,
            .full_match = full_match,
            .dflash_base = restoreDflash(e, dflash_target, matched, s),
            .entry_id = e.id,
        };
        if (!full_reuse) {
            res.checked_out = self.checkoutIfEligible(m.idx, m.shared, prompt_ids.len, slot_id);
        }
        dump.pos = res.matched;
        dump.entry_idx = m.idx + 1;
        dump.entry_count = self.entries.items.len;
        _ = restore_dump.dumpRestoreIfEnabled(target_cache, s, dump);
        return res;
    }

    /// Restore by move, the decision: only on a full-prefix hit (the commit's replace path lands
    /// on this same entry) with something to append. A partial hit keeps the refcount share.
    pub fn checkoutEligible(
        ssd_first: bool,
        move_enabled: bool,
        pending_disk: bool,
        entry_tokens: usize,
        shared: usize,
        prompt_len: usize,
        has_slot: bool,
    ) bool {
        if (!ssd_first or !move_enabled or !has_slot) return false;
        // The pending disk record shares the same buffers, so a checkout could not donate anyway.
        if (pending_disk) return false;
        if (entry_tokens == 0 or shared != entry_tokens) return false;
        return prompt_len > shared;
    }

    /// Take the checkout: mark the entry as this slot's; `donateCheckout` releases the handles
    /// at the last moment before the slot's first write. Returns whether it was taken (the
    /// admission bill's input).
    fn checkoutIfEligible(self: *HotPrefixCache, idx: usize, shared: usize, prompt_len: usize, slot_id: ?usize) bool {
        const e = &self.entries.items[idx];
        if (!checkoutEligible(
            self.ssd_first,
            restoreMoveEnabled(),
            self.pending_disk != null or self.finishing_disk != null,
            e.tokens.len,
            shared,
            prompt_len,
            slot_id != null,
        )) return false;
        e.checked_out_by = slot_id;
        log.info("  [hot-cache] checked out {d}-token entry to the slot (restore by move; the append donates in place)\n", .{e.tokens.len});
        return true;
    }

    /// Restore by move on demand: the admission pass found this slot's share does not fit (its first
    /// append would copy the whole prefix). Takes the checkout `checkoutIfEligible` takes up front in
    /// SSD-first mode, on the entry the slot just restored.
    pub fn checkoutRestored(self: *HotPrefixCache, slot_id: usize, prompt_len: usize) bool {
        const used = self.last_restored_used orelse return false;
        for (self.entries.items) |*e| {
            if (e.last_used != used) continue;
            if (e.checked_out_by != null) return false;
            if (!checkoutEligible(true, restoreMoveEnabled(), self.pending_disk != null or self.finishing_disk != null, e.tokens.len, self.last_restored_shared, prompt_len, true)) return false;
            e.checked_out_by = slot_id;
            log.info("  [hot-cache] checked out {d}-token entry to the slot (the share does not fit; the append donates in place)\n", .{e.tokens.len});
            return true;
        }
        return false;
    }

    /// Restore by move, the transfer: give up the entry's own handles so the slot is the sole
    /// owner and its first `writeAtOffset` donates in place. Called right before
    /// `Generator.initWithOptions`, after the admission pass that can still refuse: releasing at
    /// restore time threw away a 364k session for a request that never ran. Idempotent.
    pub fn donateCheckout(self: *HotPrefixCache, slot_id: usize) void {
        for (self.entries.items) |*e| {
            if (e.checked_out_by != slot_id) continue;
            if (e.checkout_donated) continue;
            e.snapshot.releaseHandles();
            e.checkout_donated = true;
        }
    }

    /// End of a slot's life. Not donated: the entry never gave its handles up, hand it back
    /// unchanged. Donated: the bytes die with the slot, drop the record. Reaching here with a
    /// mark set means the slot ended without committing. Idempotent.
    pub fn releaseCheckout(self: *HotPrefixCache, slot_id: usize, reason: []const u8) void {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            const e = &self.entries.items[i];
            if (e.checked_out_by != slot_id) continue;
            const tokens_len = e.tokens.len;
            if (!e.checkout_donated) {
                // Nothing was handed over; only the mark is undone (no LRU bump for a request that never ran).
                e.checked_out_by = null;
                if (e.disk_dirty) self.disk_dirty = true;
                log.info("  [hot-cache] checked-out entry returned intact: {s} ({d} tokens; the append never ran)\n", .{ reason, tokens_len });
                continue;
            }
            // Clear the mark before `evictAt`; the snapshot is already empty.
            e.checked_out_by = null;
            e.checkout_donated = false;
            self.evictAt(i, "checked-out entry dropped");
            log.info("  [hot-cache] checked-out entry dropped: {s} ({d} tokens; its KV died with the slot)\n", .{ reason, tokens_len });
        }
    }

    /// Commit the current `source_cache` state under the given key. Updates
    /// the matching entry if one exists for this exact prefix, otherwise
    /// inserts a new entry, evicting the oldest if at capacity. Snapshot is
    /// taken here (cheap — refcount-share, no data copy).
    pub fn commit(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
    ) !CommitStatus {
        return self.commitWithState(source_cache, tokens, has_tools, &.{}, null);
    }

    /// Commit with the media spans and the optional DFlash context payload.
    pub fn commitWithState(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash: ?DflashCommit,
    ) !CommitStatus {
        return self.commitWithMediaState(source_cache, tokens, has_tools, media, 0, dflash);
    }

    pub fn commitWithMediaState(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        cache_key: u64,
        dflash: ?DflashCommit,
    ) !CommitStatus {
        const quant_config = source_cache.config;
        // An item's key applies only to rows the entry covers: a cancelled
        // prefill that stopped before an item commits without it.
        var eff_media = spansBelow(media, tokens.len);
        // Taken before the budget trim moves the record: the disk side persists
        // the full media-keyed prefix (v9); the per-request depth is clamped at
        // lookup by `mediaSharedBound`. The spec snapshots describe rows past
        // the first item under NO media key gate, so only a text-only commit
        // rides them to the tier.
        const disk_tokens = tokens;
        const disk_media = spansBelow(eff_media, disk_tokens.len);
        const disk_spec = eff_media.len == 0;

        // Record what the live cache holds now, before any byte-budget trim.
        if (self.ssd_first and self.disk != null and disk_tokens.len > 0) {
            self.capturePendingDisk(source_cache, disk_tokens, has_tools, disk_media, if (disk_spec) dflash else null);
        }
        // The record shares the live KV; on an error return nothing consumes it and the slot's
        // KVCache deinit then frees nothing. Function scope on purpose.
        errdefer if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        };

        if (!self.ram_enabled) {
            if (self.pending_disk == null) return .declined;
            self.disk_dirty = true;
            return .{ .disk_only = tokens.len };
        }

        var replace_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (mediaSharedBound(e.media, eff_media) < e.tokens.len) continue;
            if (!std.meta.eql(e.quant_config, quant_config)) continue;
            if (e.tokens.len <= tokens.len) {
                var shared: usize = 0;
                while (shared < e.tokens.len and e.tokens[shared] == tokens[shared]) shared += 1;
                if (shared == e.tokens.len) {
                    replace_idx = i;
                    break;
                }
            }
        }

        var new_snap = try source_cache.snapshot();
        // The speculative-side payloads are best-effort: a snapshot failure
        // must not cost the trunk KV entry they ride on.
        var new_dflash: ?DflashSnap = null;
        var new_dflash_bytes: u64 = 0;
        if (dflash) |d| {
            if (d.cache.snapshot()) |snap| {
                new_dflash = .{ .snapshot = snap, .base_pos = d.base_pos };
                new_dflash_bytes = snapshotBytes(&new_dflash.?.snapshot);
            } else |err| {
                log.warn("  [hot-cache] dflash context snapshot failed: {s}\n", .{@errorName(err)});
            }
        }
        var new_kv_bytes = snapshotBytes(&new_snap);
        var new_bytes = new_kv_bytes + new_dflash_bytes;
        // Effective candidate: a byte-budget trim below shortens these.
        var eff_tokens = tokens;
        // The byte budget is a hard retention cap, including for the first (or
        // only) entry. The old eviction loop could empty the cache and then
        // append an entry larger than the cap, defeating the load-time clamp
        // precisely for long single-conversation prefixes (#326). But a flat
        // decline is a CLIFF (#330): a long agent session crosses the budget
        // once mid-conversation and then cold-prefills every turn while the
        // cap "holds" zero bytes. Retain the longest restorable prefix that
        // fits instead; decline only when nothing above the floor does.
        if (self.max_kv_bytes > 0 and new_bytes > self.max_kv_bytes) {
            var trimmed_ok = false;
            var decline: TrimDecline = .no_restorable_prefix;
            var decline_err: ?anyerror = null;
            trim_blk: while (true) {
                const row_bytes = snapshotRowBytes(&new_snap);
                const tl = trimLenForBudget(self.max_kv_bytes, tokens.len, row_bytes) orelse break :trim_blk;
                // One-shot: when the resident covered entry already retains
                // the trim target, keep it and drop the candidate — the
                // target is budget-derived and stable, so replacing would
                // re-copy an identical multi-GB prefix every turn.
                if (replace_idx) |idx| {
                    // ...unless the entry is checked out: its snapshot is empty, so fall through to the trim/replace.
                    if (self.entries.items[idx].checked_out_by == null and
                        self.entries.items[idx].tokens.len >= tl)
                    {
                        // The resident entry already covers the trim target;
                        // the candidate's EXTRA tokens still belong on disk.
                        self.spillDeclinedToDisk(&new_snap, disk_tokens, has_tools, disk_media);
                        var discarded = new_snap;
                        discarded.deinit();
                        if (new_dflash) |*d| d.deinit();
                        log.info("  [hot-cache] kept resident {d}-token prefix; oversized candidate ({d} tokens, {d:.2} MB > {d:.2} MB budget) trims no further\n", .{
                            self.entries.items[idx].tokens.len,
                            tokens.len,
                            @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                            @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                        });
                        return .{ .kept_resident = self.entries.items[idx].tokens.len };
                    }
                }
                const trimmed = new_snap.trimmedCopy(tl, mlx.gpuStream()) catch |err| {
                    decline = .snapshot_copy_failed;
                    decline_err = err;
                    break :trim_blk;
                };
                new_snap.deinit();
                new_snap = trimmed;
                // Spec payloads describe the FULL-length state; a trimmed
                // prefix rebuilds them on its first reused turn.
                if (new_dflash) |*d| {
                    d.deinit();
                    new_dflash = null;
                    new_dflash_bytes = 0;
                }
                eff_tokens = tokens[0..tl];
                // The entry keeps only the items whose rows it still holds.
                eff_media = spansBelow(eff_media, tl);
                new_kv_bytes = snapshotBytes(&new_snap);
                new_bytes = new_kv_bytes;
                log.info("  [hot-cache] trimmed oversized entry to {d}/{d} tokens ({d:.2} MB; {d:.2} MB budget)\n", .{
                    tl,
                    tokens.len,
                    @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                });
                trimmed_ok = true;
                break;
            }
            if (!trimmed_ok) {
                // RAM decline is not a value verdict: offer the candidate to
                // the SSD tier before discarding it.
                self.spillDeclinedToDisk(&new_snap, disk_tokens, has_tools, disk_media);
                var discarded_snap = new_snap;
                discarded_snap.deinit();
                if (new_dflash) |*d| d.deinit();
                const err_sep: []const u8 = if (decline_err != null) ": " else "";
                const err_name: []const u8 = if (decline_err) |e| @errorName(e) else "";
                log.info("  [hot-cache] skipped oversized entry ({d} tokens, {d:.2} MB > {d:.2} MB budget): {s}{s}{s}\n", .{
                    tokens.len,
                    @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                    decline.reason(),
                    err_sep,
                    err_name,
                });
                return .declined;
            }
        }
        const tokens_owned, const media_owned = dupeEntryKeys(self.allocator, eff_tokens, eff_media) catch |err| {
            var snap = new_snap;
            snap.deinit();
            if (new_dflash) |*d| d.deinit();
            return err;
        };

        if (replace_idx) |idx| {
            const e = &self.entries.items[idx];

            // Free everything the old entry owned.
            self.allocator.free(e.tokens);
            self.allocator.free(e.media);
            e.snapshot.deinit();
            // The old speculative payloads describe a strict PREFIX of the
            // new tokens, but they are keyed to their own base_pos and
            // length; the new ones supersede them outright. A commit with no
            // payload drops the old rather than keeping a shorter stale one.
            if (e.dflash) |*d| d.deinit();
            e.dflash = null;
            self.current_kv_bytes -|= e.kv_bytes;

            e.tokens = tokens_owned;
            e.snapshot = new_snap;
            e.has_tools = has_tools;
            e.media = media_owned;
            e.cache_key = cache_key;
            e.quant_config = quant_config;
            e.kv_bytes = new_kv_bytes + new_dflash_bytes;
            e.dflash = new_dflash;
            e.dflash_bytes = new_dflash_bytes;
            // Restore by move: the replacement the checkout promised; the entry is whole again.
            e.checked_out_by = null;
            e.checkout_donated = false;
            e.last_used = self.bumpCounter();
            e.disk_dirty = self.disk != null;
            self.current_kv_bytes += e.kv_bytes;
            // Enforce the cap again on the final entry, but never by evicting
            // the entry we just paid to update (#330 adjacent: near the budget
            // that thrashed commit → evict all → cold prefill, every turn).
            // Evict OTHER entries first; eviction of the sole entry is the
            // last resort that keeps the load-time headroom clamp real.
            if (self.max_kv_bytes > 0) {
                while (self.current_kv_bytes > self.max_kv_bytes and
                    self.entries.items.len > 1)
                {
                    if (!self.evictOneLruProgress("byte budget", null)) break;
                }
                while (self.current_kv_bytes > self.max_kv_bytes and
                    self.entries.items.len > 0)
                {
                    if (!self.evictOneLruProgress("byte budget", null)) break;
                }
            }
            if (self.disk != null) self.disk_dirty = true;
            self.logResident();
            return .{ .ok = eff_tokens.len };
        }

        while (self.entries.items.len >= self.max_entries) {
            if (!self.evictOneLruProgress("count cap", cache_key)) break;
        }
        if (self.max_kv_bytes > 0) {
            while (self.current_kv_bytes + new_bytes > self.max_kv_bytes and self.entries.items.len > 0) {
                if (!self.evictOneLruProgress("byte budget", cache_key)) break;
            }
        }

        self.entries.append(self.allocator, .{
            .tokens = tokens_owned,
            .has_tools = has_tools,
            .media = media_owned,
            .cache_key = cache_key,
            .snapshot = new_snap,
            .id = self.bumpCounter(),
            .last_used = self.bumpCounter(),
            .quant_config = quant_config,
            .kv_bytes = new_bytes,
            .dflash = new_dflash,
            .dflash_bytes = new_dflash_bytes,
            .disk_dirty = self.disk != null,
        }) catch |err| {
            self.allocator.free(tokens_owned);
            self.allocator.free(media_owned);
            var snap = new_snap;
            snap.deinit();
            if (new_dflash) |*d| d.deinit();
            return err;
        };
        self.current_kv_bytes += new_bytes;
        if (self.disk != null) self.disk_dirty = true;
        self.logResident();
        return .{ .ok = eff_tokens.len };
    }

    /// Offer a budget-declined candidate to the SSD tier before discarding
    /// it. The RAM budget declines RETENTION, not value: under a small
    /// --prefix-cache-mem a long agent session's cancelled-prefill prefixes
    /// were thrown away on every retry while a much larger disk tier sat
    /// idle (live 2026-09-07). Text-prefix candidates only — the disk never
    /// holds pixel-keyed rows. Best-effort: a failed write costs nothing the
    /// RAM decline hadn't already lost, and a byte-capped write continues
    /// from the next commit of the same conversation (chunks are
    /// content-deduped by token range). Since v9 a media candidate rides too:
    /// `media` carries the spans the record was keyed under (clamped to it).
    fn spillDeclinedToDisk(
        self: *HotPrefixCache,
        snap: *const KVCacheSnapshot,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
    ) void {
        const d = if (self.disk) |*dd| dd else return;
        // SSD-first captured the live state as `pending_disk` before the trim; the
        // normal flush lands it under the writer's own readback bound.
        if (self.ssd_first) {
            if (self.pending_disk != null) self.disk_dirty = true;
            return;
        }
        if (tokens.len < kv_disk_cache.MIN_PERSIST_TOKENS) return;
        // Spans at/after the record cover no persisted row — carrying one past
        // the record would make the tier reject the manifest at scan.
        const rec_media = spansBelow(media, tokens.len);
        // A decline-spill runs after the client is gone: the per-flush cap
        // exists to bound the stall a LIVE next request pays, and capping
        // here stranded most of each retry's work (live 2026-09-07: 512 MB
        // ≈ 13k tokens banked per retry while ~35k were recomputed every
        // time). Raise the budget for the spill's duration — the tier's
        // byte budget + LRU eviction is the real bound.
        const saved_cap = d.max_flush_bytes;
        d.max_flush_bytes = @max(saved_cap, kv_disk_cache.DECLINE_SPILL_FLUSH_FLOOR);
        defer d.max_flush_bytes = saved_cap;
        const outcome = d.appendCommitWithSpec(snap.entries, snap.step, snap.config, tokens, has_tools, rec_media, null, mlx.gpuStream()) catch |err| {
            log.warn("  [disk-cache] declined-candidate spill failed: {s}\n", .{@errorName(err)});
            return;
        };
        const note: []const u8 = switch (outcome) {
            .persisted => "complete",
            .partial => "byte-capped, continues on later commits",
            .skipped => "already persisted — nothing new to write",
        };
        log.info("  [disk-cache] declined RAM candidate spilled to SSD ({d} tokens, {s})\n", .{ tokens.len, note });
    }

    /// Snapshot the live cache (refcount-shared) plus the full token record.
    /// Best effort; the caller still owns `dflash`.
    fn capturePendingDisk(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash: ?DflashCommit,
    ) void {
        if (self.pending_disk) |*old| {
            // A commit that extends the pending prefix covers it; another session's is parked for
            // the flushes to finish. At most one waits: an older one still parked is dropped.
            if (old.has_tools != has_tools or !std.mem.startsWith(u32, tokens, old.tokens)) {
                if (self.finishing_disk) |*f| {
                    log.info("  [disk-cache] dropped an unfinished {d}-token snapshot for a newer session's\n", .{f.tokens.len});
                    f.deinit(self.allocator);
                }
                self.finishing_disk = old.*;
            } else {
                old.deinit(self.allocator);
            }
            self.pending_disk = null;
        }
        var snap = source_cache.snapshot() catch |err| {
            log.warn("  [disk-cache] live snapshot failed: {s} — flushing the RAM entry instead\n", .{@errorName(err)});
            return;
        };
        // Dupe the spans FIRST: a media record must never reach the tier with
        // an empty list (token-only matching would then read placeholders as
        // text). OOM here drops the pending capture whole — the RAM entry is
        // still the source of truth, same as any failed capture.
        const media_owned = self.allocator.dupe(MediaSpan, media) catch {
            snap.deinit();
            return;
        };
        var rec: PendingDiskFlush = .{
            .snapshot = snap,
            .tokens = self.allocator.dupe(u32, tokens) catch {
                self.allocator.free(media_owned);
                snap.deinit();
                return;
            },
            .has_tools = has_tools,
            .media = media_owned,
        };
        if (dflash) |d| {
            if (d.cache.snapshot()) |ds| {
                rec.dflash = .{ .snapshot = ds, .base_pos = d.base_pos };
            } else |_| {}
        }
        self.pending_disk = rec;
    }

    const EntrySpecs = struct {
        dflash: ?kv_disk_cache.SpecCommit = null,
    };

    /// The disk-tier spec payloads for one RAM entry; the flush and the idle spill must not drift.
    fn entrySpecCommits(e: *Entry) EntrySpecs {
        return .{
            .dflash = if (e.dflash) |*df| .{
                .entries = df.snapshot.entries,
                .step = df.snapshot.step,
                .config = df.snapshot.config,
                .base_pos = df.base_pos,
            } else null,
        };
    }

    /// At the end of a request every idle entry is written to the SSD tier, and RAM is trimmed
    /// back to the active session plus the idle allowance. Writing is unconditional; the
    /// allowance is a hard cap shed in two tiers: entries with a proven durable copy first,
    /// then (naming the reason) the rest, oldest first.
    pub fn spillIdleEntries(self: *HotPrefixCache, s: mlx.mlx_stream) void {
        if (!self.ssd_first) return;
        if (self.entries.items.len <= 1) return;
        const d = if (self.disk) |*dd| dd else return;

        // The active session is the single MRU entry; `last_used` is a strictly increasing counter.
        var newest_used: u64 = 0;
        for (self.entries.items) |*e| newest_used = @max(newest_used, e.last_used);

        // Pass 1: write, unconditionally. Attribute anything the writer lost since the last pass
        // before sampling the error counter: a failure between two passes is invisible to the delta.
        _ = d.harvestWriteFailures();
        const errs_before = d.writeErrors();
        var spilled: usize = 0;
        var idle_bytes: u64 = 0;
        for (self.entries.items) |*e| {
            e.spill_durable = false;
            if (e.last_used == newest_used) continue; // the active session
            // Checked out: the bytes are the slot's, the snapshot is empty. Skipped before the allowance count.
            if (e.checked_out_by != null) continue;
            // Counted against the allowance before the vision skip: it occupies RAM either way.
            idle_bytes +|= e.kv_bytes;
            // v9: media entries spill too — the tier's record carries the spans,
            // so the media-keyed rows are no longer ambiguous on disk. The spec
            // payloads describe rows past the first item under NO media key
            // gate, so they stay text-only rides (same rule as the commit path).
            const specs: EntrySpecs = if (e.media.len == 0) entrySpecCommits(e) else .{};
            const outcome = d.appendCommitWithSpec(
                e.snapshot.entries,
                e.snapshot.step,
                e.snapshot.config,
                e.tokens,
                e.has_tools,
                e.media,
                specs.dflash,
                s,
            ) catch |err| {
                log.warn("  [hot-cache] idle spill failed: {s} — entry stays resident\n", .{@errorName(err)});
                continue;
            };
            // Only `.persisted`: a silent skip or a `.partial` copy is not a copy.
            if (outcome != .persisted) continue;
            // `.persisted` is the write path's claim; the index has to agree.
            const disk_id = d.fullPrefixEntryId(e.snapshot.entries, e.snapshot.step, e.tokens, e.has_tools, e.snapshot.config, e.media) orelse {
                log.warn("  [hot-cache] idle spill: the tier does not hold the full prefix — entry stays resident\n", .{});
                continue;
            };
            // A staged copy is not durable yet, and the check must not be a drain (this runs on
            // the inference thread at every finish); the next pass asks again.
            if (d.entryWritesPending(disk_id)) continue;
            if (d.writeErrors() != errs_before) {
                log.warn("  [hot-cache] idle spill: background write failed — entry stays resident\n", .{});
                continue;
            }
            // Ask the filesystem too: one stat per chunk, against the eviction of a whole session.
            if (!d.entryWholeOnDisk(disk_id)) {
                log.warn("  [hot-cache] idle spill: the persisted chunks do not match the index — entry stays resident\n", .{});
                continue;
            }
            e.spill_durable = true;
            spilled += 1;
        }

        // Pass 2: evict the durable, oldest first, down to the allowance.
        var evicted: usize = 0;
        while (idle_bytes > self.ssd_idle_mem) {
            const idx = self.oldestIdleIndex(newest_used, true) orelse break;
            idle_bytes -|= self.entries.items[idx].kv_bytes;
            self.evictAt(idx, "SSD-first idle spill");
            evicted += 1;
        }

        // Pass 3: still over. Entries with no durable copy go too, oldest first, naming why.
        while (idle_bytes > self.ssd_idle_mem) {
            const idx = self.oldestIdleIndex(newest_used, false) orelse break;
            const e = &self.entries.items[idx];
            log.info("  [hot-cache] idle allowance exceeded: dropped unpersistable entry ({s}) {d} tokens, {d:.1} MB\n", .{
                unpersistableReason(d, e),
                e.tokens.len,
                @as(f64, @floatFromInt(e.kv_bytes)) / (1024.0 * 1024.0),
            });
            idle_bytes -|= e.kv_bytes;
            self.evictAt(idx, "SSD-first idle allowance");
            evicted += 1;
        }

        if (spilled > 0 or evicted > 0) {
            log.info("  [hot-cache] SSD-first: wrote {d} idle entries to disk, evicted {d}; RAM holds the active session + {d} MB idle allowance\n", .{
                spilled, evicted, self.ssd_idle_mem >> 20,
            });
        }
    }

    /// Least-recently-used idle entry (never the active session). Since v9 a
    /// media entry is ordinary durable-LRU prey once its media-keyed copy is
    /// durable (pass 2's `durable_only` already proves it); an UNPERSISTED media
    /// entry stays exempt from the non-durable shed — that single RAM copy is
    /// the only home of its media-suffix state until the tier holds one.
    fn oldestIdleIndex(self: *HotPrefixCache, newest_used: u64, durable_only: bool) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used == newest_used) continue;
            if (e.checked_out_by != null) continue;
            if (durable_only and !e.spill_durable) continue;
            if (!durable_only and e.media.len != 0 and !e.spill_durable) continue;
            if (e.last_used < best_used) {
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
    }

    /// Why pass 1 could not leave a durable copy of `e` (for the log).
    fn unpersistableReason(d: *kv_disk_cache.DiskTier, e: *const Entry) []const u8 {
        if (d.store_declined) return "store declined: volume is short";
        if (e.tokens.len < @as(usize, kv_disk_cache.MIN_PERSIST_TOKENS)) return "under the persist floor";
        const target = kv_disk_cache.persistTargetLen(e.snapshot.entries, e.snapshot.step, e.tokens.len);
        for (e.snapshot.entries) |*le| {
            if (le.initialized and le.offset < target) return "layer offset short of the range";
        }
        return "partial copy";
    }

    /// What one slice did to its record. Every result but `owed` gives the record up until its
    /// next commit: retrying a failed or empty slice would spin the idle loop.
    const SliceResult = enum {
        /// Bytes landed and the tier owes the record more.
        owed,
        /// The tier holds the record whole.
        whole,
        /// Nothing to write: under the floor, unpersistable, the store declined, or nothing landed.
        skipped,
        /// The write raised.
        failed,
        /// An earlier write for the record failed and the tier poisoned it.
        poisoned,
    };

    /// One bounded append of a record.
    fn appendSlice(
        d: *kv_disk_cache.DiskTier,
        snap: *const KVCacheSnapshot,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        specs: EntrySpecs,
        s: mlx.mlx_stream,
    ) SliceResult {
        if (d.resumesPoisoned(tokens, has_tools, media, snap.config)) {
            log.warn("  [disk-cache] a write for this {d}-token session failed; it waits for its next commit\n", .{tokens.len});
            return .poisoned;
        }
        const landed_before = d.counter;
        const out = d.appendCommitWithSpec(snap.entries, snap.step, snap.config, tokens, has_tools, media, specs.dflash, s) catch |err| {
            log.warn("  [disk-cache] persist failed: {s}; the {d}-token session waits for its next commit\n", .{ @errorName(err), tokens.len });
            return .failed;
        };
        return switch (out) {
            .persisted => .whole,
            .skipped => .skipped,
            .partial => if (d.counter != landed_before) .owed else .skipped,
        };
    }

    /// One slice of a parked snapshot; dropped once whole or given up.
    fn slicePending(self: *HotPrefixCache, d: *kv_disk_cache.DiskTier, slot: *?PendingDiskFlush, s: mlx.mlx_stream) ?SliceResult {
        const p = if (slot.*) |*pp| pp else return null;
        const specs: EntrySpecs = .{
            .dflash = if (p.dflash) |*df| .{
                .entries = df.snapshot.entries,
                .step = df.snapshot.step,
                .config = df.snapshot.config,
                .base_pos = df.base_pos,
            } else null,
        };
        const r = appendSlice(d, &p.snapshot, p.tokens, p.has_tools, p.media, specs, s);
        if (r == .owed) return r;
        p.deinit(self.allocator);
        slot.* = null;
        return r;
    }

    /// One slice of a RAM entry; its flag stays set only while the tier owes it more.
    fn sliceEntry(d: *kv_disk_cache.DiskTier, e: *Entry, s: mlx.mlx_stream) SliceResult {
        // A media entry's spec payloads stay behind (they describe rows past the first item).
        const r = appendSlice(d, &e.snapshot, e.tokens, e.has_tools, e.media, if (e.media.len == 0) entrySpecCommits(e) else .{}, s);
        e.disk_dirty = r == .owed;
        return r;
    }

    /// The owed entry the next slice goes to, newest or oldest; never one checked out (its
    /// bytes are the slot's until the commit that replaces it).
    fn owedEntryIndex(self: *const HotPrefixCache, newest: bool) ?usize {
        var best: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (!e.disk_dirty or e.checked_out_by != null) continue;
            const b = best orelse {
                best = i;
                continue;
            };
            if ((e.last_used > self.entries.items[b].last_used) == newest) best = i;
        }
        return best;
    }

    /// Does the tier still owe a slice anything it can write?
    fn diskOwed(self: *const HotPrefixCache) bool {
        if (self.disk == null) return false;
        if (self.finishing_disk != null or self.pending_disk != null) return true;
        return self.owedEntryIndex(false) != null;
    }

    const SliceOrder = enum { newest_first, oldest_first };

    /// One bounded slice, then `disk_dirty` says whether anything is still owed. A parked
    /// snapshot goes first: it is an earlier session's, and RAM holds at most a trimmed copy.
    fn flushSlice(self: *HotPrefixCache, s: mlx.mlx_stream, order: SliceOrder) ?SliceResult {
        if (!self.disk_dirty) return null;
        defer self.disk_dirty = self.diskOwed();
        const d = if (self.disk) |*dd| dd else return null;
        if (self.finishing_disk != null) return self.slicePending(d, &self.finishing_disk, s);
        // The live state captured at commit, not what the RAM entry kept after its trim.
        if (order == .newest_first and self.pending_disk != null) return self.slicePending(d, &self.pending_disk, s);
        if (self.owedEntryIndex(order == .newest_first)) |i| return sliceEntry(d, &self.entries.items[i], s);
        return self.slicePending(d, &self.pending_disk, s);
    }

    /// The finish path: one slice of the turn that just committed. Called AFTER `markFinished`,
    /// on the inference thread; snapshot arrays are refcount-shared with the RAM entry.
    pub fn flushPendingDisk(self: *HotPrefixCache, s: mlx.mlx_stream) void {
        _ = self.flushSlice(s, .newest_first);
    }

    /// The idle loop's flush: one bounded slice to the OLDEST owed session (the next the LRU
    /// drops), so every finished session lands whole, not only the newest. True while owed.
    pub fn flushOwedSlice(self: *HotPrefixCache, s: mlx.mlx_stream) bool {
        _ = self.flushSlice(s, .oldest_first);
        return self.disk_dirty;
    }

    /// One shutdown slice in the idle loop's order, tallied. The caller saw `diskOwed()`.
    fn shutdownSlice(self: *HotPrefixCache, s: mlx.mlx_stream, t: *ShutdownTally) void {
        const d = if (self.disk) |*dd| dd else return;
        self.disk_dirty = true;
        const before = d.bytes_persisted;
        const r = self.flushSlice(s, .oldest_first) orelse return;
        const landed = d.bytes_persisted - before;
        t.bytes += landed;
        switch (r) {
            // A record already covered on disk (a parked snapshot's RAM twin) writes nothing.
            .whole => t.written += @intFromBool(landed > 0),
            .skipped => t.left_declined += @intFromBool(d.store_declined),
            .failed => t.failures += 1,
            // The writer's failure that poisoned it is counted from the writer.
            .owed, .poisoned => {},
        }
    }

    /// How many records the tier still owes and a slice could write.
    fn owedCount(self: *const HotPrefixCache) u32 {
        var n: u32 = @intFromBool(self.finishing_disk != null) + @intFromBool(self.pending_disk != null);
        for (self.entries.items) |*e| n += @intFromBool(e.disk_dirty and e.checked_out_by == null);
        return n;
    }

    /// Owed entries an in-flight request holds.
    fn checkedOutOwed(self: *const HotPrefixCache) u32 {
        var n: u32 = 0;
        for (self.entries.items) |*e| n += @intFromBool(e.disk_dirty and e.checked_out_by != null);
        return n;
    }

    /// Arm the attached disk tier for this model: SSD-first per `ssdFirstActive`, and on both
    /// arms the background writer, which chunk share rides (it lands files by rename).
    pub fn armDiskTier(self: *HotPrefixCache, config: *const model_mod.ModelConfig) void {
        self.ssd_first = ssdFirstActive(config, self.disk != null, self.ram_enabled);
        const d = if (self.disk) |*dd| dd else return;
        d.ssd_first = self.ssd_first;
        d.enableBackgroundWriter();
        // Startup sweep of strays + root-wide LRU across sibling fingerprints.
        if (self.ssd_first) d.sweepSiblings();
    }

    /// Would `next` cap the cache below `current`? 0 is uncapped.
    pub fn budgetShrinks(current: u64, next: u64) bool {
        return next != 0 and (current == 0 or next < current);
    }

    /// Re-clamp the byte budget after the machine's residency changed (a model loaded
    /// or unloaded beside this one). Shrinking evicts LRU down to the new cap; 0 = uncapped.
    pub fn setBudget(self: *HotPrefixCache, max_kv_bytes: u64) void {
        if (max_kv_bytes == self.max_kv_bytes) return;
        if (max_kv_bytes >> 20 != self.max_kv_bytes >> 20)
            log.info("  [hot-cache] budget revised {d} -> {d} MB\n", .{ self.max_kv_bytes >> 20, max_kv_bytes >> 20 });
        const shrank = budgetShrinks(self.max_kv_bytes, max_kv_bytes);
        self.max_kv_bytes = max_kv_bytes;
        if (!shrank) return;
        while (self.current_kv_bytes > max_kv_bytes and self.entries.items.len > 0) {
            if (!self.evictOneLruProgress("budget revised", null)) break;
        }
        self.logResident();
    }

    fn evictOneLru(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) void {
        const idx = self.lruIndexExcluding(null, incoming_key) orelse return;
        self.evictAt(idx, reason);
    }

    /// False when nothing was evictable (every survivor is checked out) — the termination
    /// condition every budget loop needs, since `evictOneLru` is then a no-op.
    fn evictOneLruProgress(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) bool {
        const before = self.entries.items.len;
        self.evictOneLru(reason, incoming_key);
        return self.entries.items.len != before;
    }

    pub fn dropLastRestored(self: *HotPrefixCache) bool {
        var dropped = false;
        if (self.last_restored_disk_id) |id| {
            self.last_restored_disk_id = null;
            if (self.disk) |*d| {
                if (d.poisonId(id, "history check")) dropped = true;
            }
        }
        const used = self.last_restored_used orelse return dropped;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used != used) continue;
            self.last_restored_used = null;
            self.evictAt(i, "history check");
            return true;
        }
        self.last_restored_used = null;
        return dropped;
    }

    fn evictAt(self: *HotPrefixCache, lru_idx: usize, reason: []const u8) void {
        var evicted = self.entries.swapRemove(lru_idx);
        const tokens_len = evicted.tokens.len;
        const kv_mb = @as(f64, @floatFromInt(evicted.kv_bytes)) / (1024.0 * 1024.0);
        const key = evicted.cache_key;
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        log.info("  [hot-cache] evicted LRU entry ({s}; key={x}; was {d} tokens, {d:.2} MB)\n", .{
            reason, key, tokens_len, kv_mb,
        });
    }

    /// Bytes the cache currently holds resident. A hint for the connection thread's admission
    /// guard; the decision that matters is made on the inference thread by `evictLruToAdmit`.
    /// Host bytes this cache's SSD writer holds for files not yet written (up to the permit,
    /// ~1 GiB), peaking at a long prefill's chunk boundary. Inference thread only.
    pub fn stagedHostBytes(self: *HotPrefixCache) u64 {
        const d = if (self.disk) |*dd| dd else return 0;
        return d.stagedHostBytes();
    }

    pub fn residentBytes(self: *const HotPrefixCache) u64 {
        return self.current_kv_bytes;
    }

    /// Bytes an eviction pass can prove it will get back: the residency minus the largest
    /// single entry (a restore pins at most one, and the guard cannot know which).
    pub fn reclaimableBytes(self: *const HotPrefixCache) u64 {
        var largest: u64 = 0;
        // A checked-out entry is neither restorable nor evictable; it comes off the base.
        var checked_out: u64 = 0;
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) {
                checked_out += e.kv_bytes;
                continue;
            }
            largest = @max(largest, e.kv_bytes);
        }
        return self.current_kv_bytes -| checked_out -| largest;
    }

    /// One resident entry, reduced to what a connection thread may know about it (it may never
    /// dereference `hot_prefix_cache`). `fingerprint` hashes the first `DIGEST_PREFIX_TOKENS`
    /// ids, the restore floor; a shorter entry can never be pinned and gets no digest.
    pub const EntryDigest = struct {
        fingerprint: u64,
        len: u32,
        kv_bytes: u64,
    };

    pub const DIGEST_PREFIX_TOKENS: usize = MIN_CANCELLED_COMMIT_TOKENS;

    /// FNV-1a over the first `DIGEST_PREFIX_TOKENS` ids; null under the restore floor.
    pub fn prefixFingerprint(tokens: []const u32) ?u64 {
        if (tokens.len < DIGEST_PREFIX_TOKENS) return null;
        var h: u64 = 0xcbf29ce484222325;
        for (tokens[0..DIGEST_PREFIX_TOKENS]) |t| {
            h ^= t;
            h *%= 0x100000001b3;
        }
        return h;
    }

    /// Snapshot the resident entries for publication. Caller owns the slice.
    pub fn digestsAlloc(self: *const HotPrefixCache, allocator: std.mem.Allocator) ![]EntryDigest {
        var out = std.ArrayList(EntryDigest).empty;
        errdefer out.deinit(allocator);
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) continue;
            const fp = prefixFingerprint(e.tokens) orelse continue;
            try out.append(allocator, .{
                .fingerprint = fp,
                .len = @intCast(@min(e.tokens.len, std.math.maxInt(u32))),
                .kv_bytes = e.kv_bytes,
            });
        }
        return out.toOwnedSlice(allocator);
    }

    /// The connection thread's half of the rule: residency minus the largest entry this prompt
    /// could restore from. The pin condition is the fingerprint match alone (a longer record
    /// still restores, clamped); over-crediting is the unsafe direction.
    pub fn reclaimableFromDigests(
        digests: []const EntryDigest,
        residency: u64,
        prompt_fingerprint: ?u64,
    ) u64 {
        const fp = prompt_fingerprint orelse return residency;
        var pinned: u64 = 0;
        for (digests) |d| {
            if (d.fingerprint == fp) pinned = @max(pinned, d.kv_bytes);
        }
        return residency -| pinned;
    }

    /// `reclaimableBytes` with the prompt in hand: only an entry this prompt could restore from
    /// is unevictable. Conservative on both counts (key filters not applied, the LARGEST match
    /// withheld), so never larger than the truth and never smaller than `reclaimableBytes()`.
    pub fn reclaimableBytesFor(self: *const HotPrefixCache, prompt_tokens: []const u32) u64 {
        var pinned: u64 = 0;
        var checked_out: u64 = 0;
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) {
                checked_out += e.kv_bytes;
                continue;
            }
            const max_shared = @min(e.tokens.len, prompt_tokens.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_tokens[shared]) shared += 1;
            if (shared < MIN_CANCELLED_COMMIT_TOKENS) continue;
            pinned = @max(pinned, e.kv_bytes);
        }
        return self.current_kv_bytes -| checked_out -| pinned;
    }

    /// Evict least-recently-used entries until `fits()` says the request fits (#353): a cached
    /// prefix is an optimization, the request is the work. Never evicts the entry this request
    /// restored from; `fits` is re-asked after every eviction.
    /// Smallest eviction whose live/billed ratio is meaningful.
    pub const SHARED_RATIO_MIN_BYTES: u64 = 1 << 20;

    /// An eviction returning less than 1/Nth of what the entry was billed gave the allocator
    /// nothing (refcount-shared with a live cache). A ratio, not a floor: a small exclusive
    /// entry also returns little.
    pub const SHARED_RETURN_DIVISOR: u64 = 4;

    /// A restore is a lien on the whole entry: `KVCache.restore` shares the entire allocation
    /// and `truncate` frees nothing, so a slot that matched 11 tokens of a 524k entry held all
    /// 11.5 GB and the admission pass could not touch it. A restore may not pin an entry that
    /// hands back less than 1/`RESTORE_PIN_RATIO` of what it pins; small entries never reach the test.
    pub const RESTORE_PIN_MIN_BYTES: u64 = 1 << 30;
    pub const RESTORE_PIN_RATIO: usize = 64;

    /// Would restoring `shared` rows from this entry be a lien rather than a hit?
    pub fn restoreWouldPinEntry(kv_bytes: u64, min_bytes: u64, entry_tokens: usize, shared: usize) bool {
        if (kv_bytes < min_bytes) return false;
        if (shared == 0) return true;
        return entry_tokens > shared *| RESTORE_PIN_RATIO;
    }

    pub fn evictLruToAdmit(
        self: *HotPrefixCache,
        seq_tokens: u64,
        ctx: ?*anyopaque,
        fits: *const fn (?*anyopaque) bool,
        protect_restored: bool,
    ) EvictionReport {
        var report = EvictionReport{};
        while (!fits(ctx)) {
            const idx = self.lruIndexExcluding(if (protect_restored) self.last_restored_used else null, null) orelse break;
            // Accounting bytes are what the entry was billed; live bytes are what the allocator got back.
            var live_before: usize = 0;
            _ = mlx.mlx_get_active_memory(&live_before);
            const acct_before = self.current_kv_bytes;
            self.evictAt(idx, "admitting a long prefill");
            var live_after: usize = 0;
            _ = mlx.mlx_get_active_memory(&live_after);
            const freed_live: u64 = @as(u64, live_before) -| @as(u64, live_after);
            report.entries += 1;
            report.bytes += freed_live;
            const acct_delta = acct_before -| self.current_kv_bytes;
            report.accounted_bytes += acct_delta;
            // Judge only entries big enough for the ratio to mean something.
            if (acct_delta >= SHARED_RATIO_MIN_BYTES and
                freed_live * SHARED_RETURN_DIVISOR < acct_delta)
            {
                report.shared_stop = true;
                break;
            }
        }
        report.admitted = fits(ctx);
        if (report.entries > 0) {
            log.info("  [hot-cache] evicted {d} entries ({d} MB live, {d} MB billed) to admit a {d}-token prefill{s}\n", .{
                report.entries,
                report.bytes / (1024 * 1024),
                report.accounted_bytes / (1024 * 1024),
                seq_tokens,
                if (report.shared_stop) " — stopped: the next entry is shared with a live request" else "",
            });
        }
        return report;
    }

    /// Least-recently-used entry index, skipping the one whose `last_used` equals `protect`.
    /// Workload-fair: only entries of the `cache_key` holding the most eligible entries
    /// (`incoming_key` counts as one more) are candidates, so a sweep evicts its own
    /// documents before another workload's conversation. One key = plain LRU.
    fn lruIndexExcluding(self: *const HotPrefixCache, protect: ?u64, incoming_key: ?u64) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        var max_count: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            // Held by a live slot that owns its buffers: evicting it frees nothing.
            if (e.checked_out_by != null) continue;
            if (protect) |p| {
                if (e.last_used == p) continue;
            }
            var count: usize = if (incoming_key != null and incoming_key.? == e.cache_key) 1 else 0;
            for (self.entries.items) |*o| {
                if (o.checked_out_by != null or o.cache_key != e.cache_key) continue;
                if (protect) |p| {
                    if (o.last_used == p) continue;
                }
                count += 1;
            }
            if (count > max_count or (count == max_count and e.last_used < best_used)) {
                max_count = count;
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
    }

    fn logResident(self: *const HotPrefixCache) void {
        const mb = @as(f64, @floatFromInt(self.current_kv_bytes)) / (1024.0 * 1024.0);
        if (self.max_kv_bytes == 0) {
            log.info("  [hot-cache] resident={d:.2} MB ({d}/{d} entries)\n", .{ mb, self.entries.items.len, self.max_entries });
        } else {
            const cap_mb = @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0);
            log.info("  [hot-cache] resident={d:.2} / {d:.2} MB ({d}/{d} entries)\n", .{ mb, cap_mb, self.entries.items.len, self.max_entries });
        }
    }

    /// Drop all entries. Called when the cache is suspect (pad-only generation, image-bearing
    /// prompt, tools toggle change).
    pub fn invalidateAll(self: *HotPrefixCache, reason: []const u8) void {
        // Suspect state must die on both tiers.
        if (self.disk) |*d| d.invalidateAll();
        self.disk_dirty = false;
        self.dropPendingSnapshots();
        if (self.entries.items.len == 0) return;
        log.info("  [hot-cache] invalidating all {d} entries: {s}\n", .{ self.entries.items.len, reason });
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.clearRetainingCapacity();
        self.current_kv_bytes = 0;
    }

    /// Drop the most recently committed entry — used after a pad-only
    /// generation: the entry we just wrote may have stale K/V from the bad
    /// generation in tail positions. Other entries from prior healthy
    /// requests remain untouched (improvement over the legacy nuke-everything).
    pub fn invalidateLatest(self: *HotPrefixCache, reason: []const u8) void {
        if (self.disk) |*d| d.invalidateNewest();
        // The parked snapshot and older entries are earlier, healthy sessions': they still land.
        defer self.disk_dirty = self.diskOwed();
        if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        }
        if (self.entries.items.len == 0) return;
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        var evicted = self.entries.swapRemove(newest_idx);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        log.info("  [hot-cache] invalidated latest entry: {s}\n", .{reason});
    }

    pub fn entryCount(self: *const HotPrefixCache) usize {
        return self.entries.items.len;
    }
};

// ── Tests ──

const testing = std.testing;

test "HotPrefixCache: zero capacity disables RAM retention" {
    var cache = HotPrefixCache.init(testing.allocator, 0);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 0), cache.max_entries);
    try testing.expect(!cache.ram_enabled);
    try testing.expectEqual(@as(usize, 0), cache.entryCount());
}

test "HotPrefixCache: findBestMatch returns longest shared prefix" {
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    // Two synthetic entries (snapshots are no-ops on freshly-zero KVCache; we
    // never restore in this unit test, so no GPU work).
    const ids_a = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    const ids_b = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 9, 9, 9 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_a,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
    });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_b,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 2,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
    });

    // Looking up [1,2,3,4,5,6] should match entry A (5 shared tokens).
    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const m = cache.findBestMatch(&lookup_ids, false, &.{}, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 5), m.shared);

    // Looking up [1,2,3,9,9,9,7] should match entry B (6 shared).
    const lookup_ids2 = [_]u32{ 1, 2, 3, 9, 9, 9, 7 };
    const m2 = cache.findBestMatch(&lookup_ids2, false, &.{}, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 1), m2.idx);
    try testing.expectEqual(@as(usize, 6), m2.shared);

    // has_tools mismatch returns null.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, true, &.{}, kv_quant.KVQuantConfig.dense));
    // A media item at row 0 matches only the same pixels, both ways.
    var k7 = [_]MediaSpan{.{ .start = 0, .key = 7 }};
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, &k7, kv_quant.KVQuantConfig.dense));
    cache.entries.items[0].media = &k7;
    // Text lookup falls through to entry B (3 shared); the keyed lookup gets A.
    try testing.expectEqual(@as(usize, 1), cache.findBestMatch(&lookup_ids, false, &.{}, kv_quant.KVQuantConfig.dense).?.idx);
    try testing.expectEqual(@as(usize, 0), cache.findBestMatch(&lookup_ids, false, &k7, kv_quant.KVQuantConfig.dense).?.idx);
    cache.entries.items[0].media = &.{};
    // Scheme mismatch returns null — entries are dense, a query for affine
    // 4-bit cannot match (Wave 1.A: cross-scheme cache hits never happen).
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, &.{}, kv_quant.KVQuantConfig.affine(4)));
}

test "HotPrefixCache: restore clamps an inflated snapshot to the matched length (gemma mask crash)" {
    // Root cause of the live gemma-4-26B-A4B crash (2026-07-09, broadcast_shapes
    // mask 16890 vs KV 16892 at ~16K ctx): a snapshot committed with a KV buffer
    // LONGER than its logical token count — PLD/speculative decode leaves stale
    // draft positions in the buffer past the committed step. When the NEXT prompt
    // matches the entry's ENTIRE token sequence but is longer (a partial hit,
    // effective_matched == e.tokens.len < prompt_ids.len), the old truncate guard
    // `final_len < e.tokens.len` was FALSE, so the restored cache offset kept the
    // inflated snapshot length — drifting ahead of the matched length generation
    // tracks. That drift corrupts RoPE and crashes the sliding-window prefill mask.
    const s = mlx.gpuStream();

    var toks: [67]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 11);
    const logical_len: usize = 64;

    // Source cache: 64 logical tokens, then 2 STALE tokens (offset 66) — the
    // shape a PLD round leaves behind before commit.
    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, @intCast(logical_len));
    try testFillCache(&src, s, 2, 2); // stale draft tail → offset 66
    try testing.expectEqual(@as(usize, 66), src.step);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // Commit with the LOGICAL token count (64) — but the snapshot carries 66.
    _ = try hc.commit(&src, toks[0..logical_len], false);

    // Reuse with a prompt that matches all 64 entry tokens but is LONGER (67):
    // effective_matched == e.tokens.len == 64 < prompt_ids.len — the crash path.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, s, &toks, false, &.{}, null);

    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 64), res.matched);
    // The invariant: restored cache offset == matched length, NOT the inflated 66.
    try testing.expectEqual(@as(usize, 64), moe_off);
    try testing.expectEqual(@as(usize, 64), dst.step);
    for (dst.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 64), e.offset); // clamped, not 66
    }
}

test "prefix cache: DFlash assistant context round-trips, clamped to the trunk's matched length" {
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    // Trunk KV for 64 tokens; the assistant context covers the same span but
    // starts at 10 (the committing request was itself a partial cache hit).
    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    var assist = try KVCache.init(testing.allocator, 2);
    defer assist.deinit();
    try testFillCache(&assist, s, 2, 54);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithState(&trunk, &toks, false, &.{}, .{ .cache = &assist, .base_pos = 10 });
    // The assistant context is billed like recurrent state, so the memory cap sees it.
    try testing.expect(hc.entries.items[0].dflash_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].dflash_bytes);

    // A shorter prompt that the entry fully covers: the full-match path
    // re-forwards the last token, so the trunk lands at 31 and the assistant
    // context must be clamped to 31-10 = 21 — absLen == matched, or the first
    // round's `dctx.absLen() == anchor_pos` assert fires.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    var moe_off: usize = 0;
    var base: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, s, toks[0..32], false, &.{}, .{ .cache = &dfl, .base_pos = &base });
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 10), res.dflash_base);
    try testing.expectEqual(@as(usize, 10), base);
    try testing.expectEqual(@as(usize, 21), dfl.step);
    try testing.expectEqual(base + dfl.step, res.matched);

    // An entry with no assistant payload leaves the target untouched, and the
    // caller is told so — a blind start is a valid outcome, never a wrong one.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    _ = try hc2.commit(&trunk, &toks, false);
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    var moe_off2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc2.lookupAndRestore(&dst2, &moe_off2, s, toks[0..32], false, &.{}, .{ .cache = &dfl2, .base_pos = &base2 });
    try testing.expectEqual(@as(usize, 31), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.dflash_base);
    try testing.expectEqual(@as(usize, 0), dfl2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

fn testWriteCacheLayer(cache: *KVCache, s: mlx.mlx_stream, layer: u32, written: u32, step: u32) !void {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const count: f64 = @floatFromInt(step * 8);
    const base: f64 = @floatFromInt(written * 8 + layer * 1_000_000);
    try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    const shape = [_]c_int{ 1, 1, @intCast(step), 8 };
    try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s));
    var view = try cache.update(layer, k, k, s, 0);
    view.deinit();
}

fn testFillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) try testWriteCacheLayer(cache, s, li, written, step);
        written += step;
    }
}

test "HotPrefixCache: disk-only commit persists the full prefix and retains no RAM entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        try testing.expectEqual(@as(u32, 0), hc.max_entries);
        try testing.expect(!hc.ram_enabled);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-disk-only", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        hc.disk.?.enableBackgroundWriter();
        try testing.expect(hc.disk.?.writer != null);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, tokens.len);
        const status = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(@as(usize, tokens.len), status.disk_only);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
        try testing.expect(hc.pending_disk != null);

        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        try testing.expectEqual(@as(u32, 0), hc.max_entries);
        try testing.expect(!hc.ram_enabled);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-disk-only", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        hc.disk.?.enableBackgroundWriter();
        try testing.expect(hc.disk.?.writer != null);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        var moe_offset: usize = 0;
        const restored = try hc.lookupAndRestore(&cache, &moe_offset, s, &tokens, false, &.{}, null);
        try testing.expect(restored.slot_owned);
        try testing.expectEqual(@as(usize, tokens.len - 1), restored.matched);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    }
}

test "HotPrefixCache: disk tier restores across a fresh cache instance (restart shape)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit through the NORMAL RAM path, then flush to disk
    // (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("server restart"): fresh RAM cache, fresh tier over the same
    // root. The lookup must land on the SSD tier and restore the prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount()); // RAM empty
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, s, &tokens, false, &.{}, null);
        // Full match: identical re-issue semantics — truncate to len-1 and
        // re-forward the last token, exactly like a RAM full-match hit.
        try testing.expect(res.full_match);
        try testing.expect(res.slot_owned); // a disk restore's rows are the slot's own
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(usize, 599), cache2.step);
        try testing.expectEqual(@as(usize, 599), moe_off);
        for (cache2.entries) |*e| {
            try testing.expect(e.initialized);
            try testing.expectEqual(@as(usize, 599), e.offset);
        }

        // Diverged-tail shape: shares the first 400 tokens, then differs.
        // Restore must land at 400 and leave the tail to prefill.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 2);
        defer cache3.deinit();
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, s, &tokens_div, false, &.{}, null);
        try testing.expect(!res3.full_match);
        try testing.expect(res3.slot_owned); // a disk restore's rows are the slot's own
        try testing.expectEqual(@as(usize, 400), res3.matched);
        try testing.expectEqual(@as(usize, 400), cache3.step);
        // A diverged short prefix must read ONLY the chunks covering the
        // usable 400 positions (ceil(400/128) = 4), NOT the whole 600-token
        // stored entry (5 chunks). Loading the full entry to serve a short
        // shared prefix makes a diverged "hit" slower than a cold prefill.
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last);
    }
}

test "HotPrefixCache: a disk restore leaves room for the token it re-forwards" {
    // A grow is not in place: an exact-size restore made the first append copy the whole
    // cache, and the prefill guard refused a restored 38k prompt on a 16 GB Mac for it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const Grows = &transformer_mod.KVCache.kv_cap_buf_grows;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-room", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
    }
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-room", 0, 128);
    defer hc2.deinit();
    var slot = try KVCache.init(testing.allocator, 2);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc2.lookupAndRestore(&slot, &moe_off, s, &tokens, false, &.{}, null);
    try testing.expect(res.full_match and res.slot_owned);
    try testing.expect(slot.residentCapacityTokens() > res.matched);
    const g = Grows.*;
    try testFillCache(&slot, s, 2, 1); // the re-forwarded last token
    try testing.expectEqual(@as(usize, 0), Grows.* - g);
}

test "HotPrefixCache: budget decline reports a status, not a silent success" {
    // Live 2026-09-07: every oversized decline from a cancelled prefill
    // logged "[hot-cache] committed N/M" at the scheduler because the
    // decline exits commitWithMediaState via plain return — a 122k agent
    // session looked cached while it cold-prefilled ~95k tokens per retry.
    // The commit contract must be observable: ok / kept_resident / declined.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // 16 KB budget against ~128 B/token rows: even the 256-token commit
    // floor cannot fit, so the candidate must be DECLINED — and the caller
    // must be able to SEE that instead of mistaking it for a commit.
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());

    // A candidate that fits commits with its effective token length.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commit(&cache2, &tokens, false);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 600), st2.ok);
}

test "HotPrefixCache: media spans bound the reusable prefix at the first differing item" {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const a = MediaSpan{ .start = 100, .key = 0xA };
    const b = MediaSpan{ .start = 300, .key = 0xB };
    // An item past the committed rows (a cancelled prefill) is not the entry's.
    _ = try hc.commitWithMediaState(&cache, tokens[0..400], false, &.{ a, b, .{ .start = 450, .key = 0xC } }, 0, null);
    try testing.expectEqualSlices(MediaSpan, &.{ a, b }, hc.entries.items[0].media);

    const Case = struct { media: []const MediaSpan, want: usize };
    const cases = [_]Case{
        .{ .media = &.{ a, b }, .want = 400 },
        .{ .media = &.{ a, b, .{ .start = 500, .key = 0xD } }, .want = 400 }, // new item past the entry
        .{ .media = &.{ a, .{ .start = 300, .key = 0xE } }, .want = 300 }, // later item changed
        .{ .media = &.{ .{ .start = 100, .key = 0xE }, b }, .want = 100 }, // earlier item changed
        .{ .media = &.{a}, .want = 300 }, // entry's second item missing
        .{ .media = &.{}, .want = 100 },
    };
    for (cases) |c| {
        var dst = try KVCache.init(testing.allocator, 2);
        defer dst.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&dst, &moe, s, &tokens, false, c.media, null, null, false);
        try testing.expectEqual(c.want, res.matched);
    }

    // A budget trim keeps the items below the cut.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 32 * 1024);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commitWithMediaState(&cache2, &tokens, false, &.{ a, b }, 0, null);
    try testing.expectEqual(@as(usize, 256), st2.ok); // 32 KB / 128 B rows
    try testing.expectEqualSlices(MediaSpan, &.{a}, hc2.entries.items[0].media);
}

test "HotPrefixCache: media request restores the pre-media text prefix from SSD" {
    // Live 2026-09-07: the SSD-tier lookup skipped ANY request with a
    // non-zero vision_key (`vision_key != 0 → break :disk`) — a 122k agent
    // session whose LAST turn carries one screenshot never touched the
    // 40 GB of persisted text prefixes, though 99.8% of the prompt is
    // pixel-independent text. Disk entries are text-only by construction
    // (the flush refuses entries with media), so the restore caps at the
    // request's first media row.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: a text-only conversation commits + flushes to disk.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 (RAM empty): an image-bearing request over the same text —
    // the disk tier must serve the shared text prefix, capped at the
    // request's media boundary (400), reading only the covering chunks.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc2.deinit();

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestoreWithMedia(&cache2, &moe, s, &tokens, false, &.{.{ .start = 400, .key = 0xDEAD }}, null, null, false);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 400), res.matched);
        try testing.expectEqual(@as(usize, 400), cache2.step);
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last); // ceil(400/128)
    }
}

test "HotPrefixCache: the dflash snapshot survives the SSD tier across a restart" {
    // A disk-tier restore forwards NO trunk layers, so state derived from
    // trunk hiddens (the dflash context) started EMPTY on every
    // disk hit — multi-turn across a restart drafted blind (the same
    // 92.6% → 66.5% acceptance class the RAM tier fixed). v4 persists it
    // in the entry's spec sidecar and restores it under the RAM tier's
    // exact clamp rule.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit with the spec payload through the normal RAM path,
    // then flush (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc.deinit();

        var trunk = try KVCache.init(testing.allocator, 2);
        defer trunk.deinit();
        try testFillCache(&trunk, s, 2, 600);
        var assist = try KVCache.init(testing.allocator, 2);
        defer assist.deinit();
        try testFillCache(&assist, s, 2, 600);

        _ = try hc.commitWithState(&trunk, &tokens, false, &.{}, .{ .cache = &assist, .base_pos = 0 });
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expect(hc.disk.?.entries.items[0].spec_dflash != null);
    }

    // Session 2 ("server restart"): RAM empty, disk serves the prefix AND
    // the spec snapshot, clamped to the trunk's matched length.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var trunk2 = try KVCache.init(testing.allocator, 2);
        defer trunk2.deinit();
        var dfl = try KVCache.init(testing.allocator, 2);
        defer dfl.deinit();
        var moe_off: usize = 0;
        var dbase: usize = 99;
        const res = try hc2.lookupAndRestore(&trunk2, &moe_off, s, &tokens, false, &.{}, .{ .cache = &dfl, .base_pos = &dbase });
        // Full match: identical re-issue → trunk lands at 599; the spec
        // cache clamps to base + step == matched.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(?usize, 0), res.dflash_base);
        try testing.expectEqual(@as(usize, 0), dbase);
        try testing.expectEqual(@as(usize, 599), dfl.step);

        // A geometry the persisted snap doesn't fit starts BLIND, never
        // wrong: a 3-layer dflash target declines.
        var trunk3 = try KVCache.init(testing.allocator, 2);
        defer trunk3.deinit();
        var dfl3 = try KVCache.init(testing.allocator, 3);
        defer dfl3.deinit();
        var moe3: usize = 0;
        var dbase3: usize = 42;
        const res3 = try hc2.lookupAndRestore(&trunk3, &moe3, s, &tokens, false, &.{}, .{ .cache = &dfl3, .base_pos = &dbase3 });
        try testing.expectEqual(@as(usize, 599), res3.matched);
        try testing.expectEqual(@as(?usize, null), res3.dflash_base);
        try testing.expectEqual(@as(usize, 0), dfl3.step);
        try testing.expectEqual(@as(usize, 42), dbase3); // untouched
    }
}

test "HotPrefixCache: an SSD entry that fails to restore is poisoned, not retried" {
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-bad", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
    }
    // A chunk whose bytes rotted at the same size: the scan's size check passes, the load fails.
    var path_buf: [64]u8 = undefined;
    const chunk = try std.fmt.bufPrint(&path_buf, "fp-bad/e1/c000000.safetensors", .{});
    const st = try tmp.dir.statFile(io, chunk, .{});
    const junk = try testing.allocator.alloc(u8, @intCast(st.size));
    defer testing.allocator.free(junk);
    @memset(junk, 0xff);
    try tmp.dir.writeFile(io, .{ .sub_path = chunk, .data = junk });

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-bad", 0, 128);
    defer hc2.deinit();
    const id = hc2.disk.?.entries.items[0].id;
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, s, &tokens, false, &.{}, null);
    try testing.expectEqual(@as(usize, 0), res.matched);
    try testing.expect(hc2.disk.?.entryPoisoned(id));
    mlx.dropLatchedErrorUnless(false);
}

test "HotPrefixCache: RAM match at least as long as disk skips the SSD read" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry covers it fully, so the disk tier's
    // LRU counter must not move (no restore happened).
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, s, &tokens, false, &.{}, null);
    try testing.expect(res.full_match);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
}

test "HotPrefixCache: invalidation propagates to the disk tier" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-inv", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());

    hc.invalidateAll("test poison");
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expect(!hc.disk_dirty);
}

test "HotPrefixCache: findBestMatch isolates affine 4-bit from affine 8-bit" {
    // Regression for the cross-bit-width hit that crashed SDPA in
    // tests/test_kv_quant_per_request.sh. With Entry.scheme tracking only
    // the `Scheme` enum, `affine(4)` and `affine(8)` both matched as
    // `.affine` and a 4-bit snapshot would be restored into an 8-bit slot
    // → broadcast_shapes (1,H,1,64) vs (1,H,1,32) MLX abort. After moving
    // to a full-`KVQuantConfig` filter, the two are distinct keys and
    // can't alias.
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    const ids = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.affine(4) },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.affine(4),
        .kv_bytes = 0,
    });

    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    // Matching config (affine 4) hits the entry.
    const hit = cache.findBestMatch(&lookup_ids, false, &.{}, kv_quant.KVQuantConfig.affine(4)).?;
    try testing.expectEqual(@as(usize, 0), hit.idx);
    try testing.expectEqual(@as(usize, 5), hit.shared);
    // Same Scheme (.affine) but different bits MUST NOT hit — that's the
    // cross-scheme buffer-layout crash this filter guards against.
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, &.{}, kv_quant.KVQuantConfig.affine(8)));
    // Dense query against an affine entry: also null (existing guarantee).
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, &.{}, kv_quant.KVQuantConfig.dense));
}

// ── test helpers ──

// A vision turn can move the current image span when the same image becomes
// conversation history. The newest entry then has the longest raw token match
// (ending exactly at the old image boundary), but its first state checkpoint can
// sit just AFTER that boundary. An older entry for the same pixels may have a
// slightly shorter token match with a usable checkpoint. Picking by raw token
// match alone turns that recoverable case into a full cold prefill.
test "HotPrefixCache: peekMatchedTokens reads the match without touching the cache" {
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    const ids_a = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    const ids_b = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 9, 9, 9 });
    for ([_][]u32{ ids_a, ids_b }, 1..) |ids, used| {
        try cache.entries.append(testing.allocator, .{
            .tokens = ids,
            .has_tools = false,
            .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
            .last_used = @intCast(used),
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 0,
        });
    }
    const dense = kv_quant.KVQuantConfig.dense;

    try testing.expectEqual(@as(usize, 5), cache.peekMatchedTokens(&[_]u32{ 1, 2, 3, 4, 5, 6 }, false, dense));
    try testing.expectEqual(@as(usize, 6), cache.peekMatchedTokens(&[_]u32{ 1, 2, 3, 9, 9, 9, 7 }, false, dense));
    try testing.expectEqual(@as(usize, 0), cache.peekMatchedTokens(&[_]u32{ 9, 9 }, false, dense));
    // A different tool context never restores.
    try testing.expectEqual(@as(usize, 0), cache.peekMatchedTokens(&[_]u32{ 1, 2, 3, 4, 5, 6 }, true, dense));
    // A checked-out entry belongs to another slot: it is invisible, and the peek leaves it checked out.
    cache.entries.items[0].checked_out_by = 7;
    try testing.expectEqual(@as(usize, 3), cache.peekMatchedTokens(&[_]u32{ 1, 2, 3, 4, 5, 6 }, false, dense));
    try testing.expectEqual(@as(?usize, 7), cache.entries.items[0].checked_out_by);
    try testing.expect(cache.last_restored_used == null);
}

test "HotPrefixCache: under SSD-first a decline rides pending_disk, never a second synchronous spill" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill-ssdfirst", 0, 128);
    defer hc.deinit();
    const cap_before = hc.disk.?.max_flush_bytes;

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    // The capture at commit time is the one record; nothing lands until the flush.
    try testing.expect(hc.pending_disk != null);
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expectEqual(cap_before, hc.disk.?.max_flush_bytes);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
}

test "HotPrefixCache: a budget-declined candidate spills to the SSD tier" {
    // Live 2026-09-07: an oversized decline discarded the cancelled
    // prefill's work entirely - under a 1 GB hot budget a 122k session's
    // 76-82k-token forwarded prefixes were thrown away on every retry
    // while a 40 GB SSD tier sat idle. The decline must OFFER the
    // candidate to the disk tier instead of discarding it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // A RAM budget too small for even the commit floor forces the decline.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 0), hc.entryCount()); // RAM keeps nothing
        // The candidate's work must live on the SSD tier.
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The next request (fresh RAM, generous budget) restores from SSD
    // instead of cold-prefilling the whole prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, s, &tokens, false, &.{}, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: a decline-spill is not bounded by the per-flush byte cap" {
    // The per-flush cap bounds the stall a LIVE next request pays after a
    // response. A decline-spill runs on a request whose client is already
    // gone — capping it (live 2026-09-07: 512 MB ≈ 13k tokens banked per
    // retry) stranded most of each retry's work while ~35k tokens were
    // recomputed every time. The spill owes no latency; the tier's byte
    // budget + LRU is the real bound.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc.deinit();
        // A cap of two KV chunks — the exact starvation shape, scaled down.
        hc.disk.?.max_flush_bytes = 2 * 128 * 2 * 2 * 8 * 4;

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The WHOLE candidate must be restorable, not just the cap's worth.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc2.deinit();
        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, s, &tokens, false, &.{}, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: a revised budget evicts down to fit and a raised one keeps everything" {
    const s = mlx.gpuStream();
    var a = try KVCache.init(testing.allocator, 2);
    defer a.deinit();
    try testFillCache(&a, s, 2, 600);
    var b = try KVCache.init(testing.allocator, 2);
    defer b.deinit();
    try testFillCache(&b, s, 2, 600);
    var toks_a: [600]u32 = undefined;
    for (&toks_a, 0..) |*t2, i| t2.* = @intCast(i + 1);
    var toks_b: [600]u32 = undefined;
    for (&toks_b, 0..) |*t2, i| t2.* = @intCast(i + 1000);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commit(&a, &toks_a, false);
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const one = hc.entries.items[0].kv_bytes;

    // Shrink to one entry's worth: the LRU (A) goes, B stays whole.
    hc.setBudget(one);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    try testing.expectEqual(@as(u32, 1000), hc.entries.items[0].tokens[0]);
    // Raise (0 = uncapped): nothing moves.
    hc.setBudget(0);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.max_kv_bytes);
}

test "HotPrefixCache: byte budget rejects an oversized sole entry and preserves a smaller prefix" {
    const s = mlx.gpuStream();

    var a = try KVCache.init(testing.allocator, 2);
    defer a.deinit();
    try testFillCache(&a, s, 2, 8);
    // KV buffers grow in 256-token chunks, so the sizes must straddle a chunk
    // boundary for the two entries' snapshot bytes to differ.
    var b = try KVCache.init(testing.allocator, 2);
    defer b.deinit();
    try testFillCache(&b, s, 2, 600);

    var toks_a: [8]u32 = undefined;
    for (&toks_a, 0..) |*t2, i| t2.* = @intCast(i + 1);
    var toks_b: [600]u32 = undefined;
    for (&toks_b, 0..) |*t2, i| t2.* = @intCast(i + 100);

    var a_snap = try a.snapshot();
    defer a_snap.deinit();
    var b_snap = try b.snapshot();
    defer b_snap.deinit();
    const small = HotPrefixCache.snapshotBytes(&a_snap);
    const big = HotPrefixCache.snapshotBytes(&b_snap);
    try testing.expect(big > small);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, small);
    defer hc.deinit();

    _ = try hc.commit(&a, &toks_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // B extends A, so this exercises the replacement path. It is too large
    // for the cap: the byte budget still holds — but as a TRIM (#330), not a
    // decline. The retained prefix is longer than A's, shorter than B's.
    @memcpy(toks_b[0..toks_a.len], &toks_a);
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const kept_len = hc.entries.items[0].tokens.len;
    try testing.expect(kept_len >= MIN_CANCELLED_COMMIT_TOKENS);
    try testing.expect(kept_len < toks_b.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // An empty cache retains the same trimmed prefix from the oversized
    // candidate rather than staying empty (#330: the pre-fix decline held the
    // cap by holding zero bytes).
    hc.invalidateAll("test");
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(kept_len, hc.entries.items[0].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
}

// ── Issue #330: the oversized-entry decline is a cliff, not a cap ──

/// Test-local: dense per-token KV bytes of a snapshot (k + v across layers).
fn pcRowBytes(snap: *const transformer_mod.KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized) continue;
        const rows: u64 = @intCast(mlx.mlx_array_shape(e.keys)[2]);
        if (rows == 0) continue;
        const kb = @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
        const vb = @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
        total += (kb + vb) / rows;
    }
    return total;
}

/// Test-local: assert rows [0:rows] of `a` and `b` (axis 2) are identical.
fn pcExpectRowsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, rows: usize) !void {
    var sliced_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced_a);
    var sliced_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced_b);
    const sh_a = mlx.mlx_array_shape(a);
    const sh_b = mlx.mlx_array_shape(b);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    const stop_a = [_]c_int{ sh_a[0], sh_a[1], @intCast(rows), sh_a[3] };
    const stop_b = [_]c_int{ sh_b[0], sh_b[1], @intCast(rows), sh_b[3] };
    try mlx.check(mlx.mlx_slice(&sliced_a, a, &start, 4, &stop_a, 4, &strides, 4, s));
    try mlx.check(mlx.mlx_slice(&sliced_b, b, &start, 4, &stop_b, 4, &strides, 4, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, sliced_a, sliced_b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, all));
    try testing.expect(ok);
}

test "HotPrefixCache: oversized entry trims to the longest prefix that fits (#330)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, 600);
    var toks: [600]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var probe = try src.snapshot();
    defer probe.deinit();
    const row = pcRowBytes(&probe);
    // Room for exactly 400 tokens — over the 256-token commit floor, under
    // the 600-token candidate.
    const budget = row * 400;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    _ = try hc.commit(&src, &toks, false);

    // The cliff: pre-fix this declines outright and the cache stays empty.
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // The trimmed prefix restores: same prompt matches 400 tokens.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, s, &toks, false, &.{}, null);
    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 400), res.matched);
    try testing.expectEqual(@as(usize, 400), dst.step);
    for (dst.entries, src.entries) |*d, *e| {
        try testing.expectEqual(@as(usize, 400), d.offset);
        // Trimmed rows must be the SOURCE's rows — a slice-math bug here
        // serves a wrong prefix as a cache hit.
        try pcExpectRowsEqual(s, d.keys, e.keys, 400);
        try pcExpectRowsEqual(s, d.values, e.values, 400);
    }
}

test "HotPrefixCache: trimmed entry is one-shot — a covered re-commit keeps the resident copy (#330)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, 600);
    var toks: [700]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var probe = try src.snapshot();
    defer probe.deinit();
    const budget = pcRowBytes(&probe) * 400;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    _ = try hc.commit(&src, toks[0..600], false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    const resident_keys_ctx = hc.entries.items[0].snapshot.entries[0].keys.ctx;

    // Next turn: the conversation grew, the trim target did not. Re-copying
    // an identical prefix every turn would be a per-turn multi-GB memcpy.
    var src2 = try KVCache.init(testing.allocator, 2);
    defer src2.deinit();
    try testFillCache(&src2, s, 2, 700);
    _ = try hc.commit(&src2, &toks, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expectEqual(resident_keys_ctx, hc.entries.items[0].snapshot.entries[0].keys.ctx);
}

test "evictLruToAdmit: oldest first, never the entry THIS request restored, and shared bytes are not counted as freed" {
    // "Most recently used" is not "the entry this request restored", and a restored entry's
    // shared buffers return nothing to the allocator when evicted.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var toks_a: [4096]u32 = undefined;
    for (&toks_a, 0..) |*t, i| t.* = @intCast(i + 1);
    var toks_b: [4096]u32 = undefined;
    for (&toks_b, 0..) |*t, i| t.* = @intCast(i + 1_000_001);
    var toks_c: [4096]u32 = undefined;
    for (&toks_c, 0..) |*t, i| t.* = @intCast(i + 2_000_001);

    inline for (.{ &toks_a, &toks_b, &toks_c }) |toks| {
        var cache = try KVCache.init(testing.allocator, 8);
        defer cache.deinit();
        try testFillCache(&cache, s, 8, 4096);
        // Materialize before committing: an unevaluated cache owns no Metal buffer at all.
        for (cache.entries) |*e| {
            if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
            if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
        }
        _ = try hc.commit(&cache, toks, false);
    }
    try testing.expectEqual(@as(usize, 3), hc.entryCount());
    var live_resident: usize = 0;
    _ = mlx.mlx_get_active_memory(&live_resident);
    try testing.expect(live_resident > 4 * 1024 * 1024);

    // This request restores B, the middle entry. The target cache stays alive: that is what makes B shared.
    var live_b = try KVCache.init(testing.allocator, 8);
    defer live_b.deinit();
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&live_b, &moe_off, s, &toks_b, false, &.{}, null);
    try testing.expect(hit.full_match);
    try testing.expect(hc.last_restored_used != null);

    const Never = struct {
        fn call(ctx: ?*anyopaque) bool {
            _ = ctx;
            return false;
        }
    };
    const rep = hc.evictLruToAdmit(458_832, null, Never.call, true);
    try testing.expect(!rep.admitted);
    try testing.expectEqual(@as(usize, 2), rep.entries); // A and C, oldest first
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    var probe_cache = try KVCache.init(testing.allocator, 8);
    defer probe_cache.deinit();
    var off2: usize = 0;
    const still_b = try hc.lookupAndRestore(&probe_cache, &off2, s, &toks_b, false, &.{}, null);
    try testing.expect(still_b.full_match);
    // An exclusive eviction may return MORE than it was billed (capacity rounding).
    try testing.expect(rep.bytes > 0);
    try testing.expect(rep.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR >= rep.accounted_bytes);
    try testing.expect(!rep.shared_stop);

    // Unprotected, B goes too, and the allocator gets ~nothing back, which the pass notices.
    const rest = hc.evictLruToAdmit(458_832, null, Never.call, false);
    try testing.expect(!rest.admitted);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expect(rest.accounted_bytes > 0);
    try testing.expect(rest.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR < rest.accounted_bytes);
    try testing.expect(rest.shared_stop);
}

test "a trivial share of a whole session is a LIEN, and the admission pass gets the entry back" {
    // Live shape: a cold 786k prompt shared eleven tokens with the one resident 524k entry, the
    // restore shared the whole 11.5 GB into the slot, and the admission pass could not touch it.
    const t = testing;
    try t.expect(23_692 > 14_713); // refused
    try t.expect(14_713 + 11_476 >= 23_692); // admitted

    const MB: u64 = 1 << 20;
    const floor = HotPrefixCache.RESTORE_PIN_MIN_BYTES;
    try t.expect(HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 11));
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 100_000));
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(60 * MB, floor, 218, 11));

    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true; // the arm this was measured on
    hc.restore_pin_min_bytes = 1 << 20; // a 2 MB entry stands in for 11.5 GB

    var session: [4096]u32 = undefined;
    for (&session, 0..) |*x, i| x.* = @intCast(i + 1);
    var cold: [4096]u32 = undefined;
    for (&cold, 0..) |*x, i| x.* = if (i < 11) @as(u32, @intCast(i + 1)) else @intCast(i + 5_000_001);

    // One cache: the slot that served the session is handed the next request.
    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    try testFillCache(&slot_cache, s, 8, 4096);
    for (slot_cache.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    _ = try hc.commit(&slot_cache, &session, false);
    try t.expectEqual(@as(usize, 1), hc.entryCount());
    try t.expect(hc.entries.items[0].kv_bytes > hc.restore_pin_min_bytes);

    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, s, &cold, false, &.{}, null);

    // Declined: eleven rows are not worth a lien on the session.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expect(!hit.full_match);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(@as(usize, 0), slot_cache.step);

    // ...so the admission pass reclaims it, and reclaims real bytes.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(786_369, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 1), rep.entries);
    try t.expect(rep.bytes > 0);
    try t.expect(!rep.shared_stop);
}

test "a PROPORTIONATE share still restores, and is still protected" {
    // The inverse: an entry a request genuinely continues hands back most of what it pins.
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.restore_pin_min_bytes = 1 << 20;

    var session: [4096]u32 = undefined;
    for (&session, 0..) |*x, i| x.* = @intCast(i + 1);
    var warm: [4096]u32 = undefined;
    for (&warm, 0..) |*x, i| x.* = if (i < 2048) @as(u32, @intCast(i + 1)) else @intCast(i + 5_000_001);

    var src_cache = try KVCache.init(testing.allocator, 8);
    defer src_cache.deinit();
    try testFillCache(&src_cache, s, 8, 4096);
    for (src_cache.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    _ = try hc.commit(&src_cache, &session, false);

    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, s, &warm, false, &.{}, null);
    try t.expectEqual(@as(usize, 2048), hit.matched);
    try t.expect(hc.last_restored_used != null);

    const Never = struct {
        fn call(ctx: ?*anyopaque) bool {
            _ = ctx;
            return false;
        }
    };
    const rep = hc.evictLruToAdmit(786_369, null, Never.call, true);
    try t.expectEqual(@as(usize, 0), rep.entries);
    try t.expectEqual(@as(usize, 1), hc.entryCount());

    // The rule is SSD-first's. Off it, even the lien restores.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    hc2.restore_pin_min_bytes = 1 << 20;
    var src2 = try KVCache.init(testing.allocator, 8);
    defer src2.deinit();
    try testFillCache(&src2, s, 8, 4096);
    _ = try hc2.commit(&src2, &session, false);
    var cold: [4096]u32 = undefined;
    for (&cold, 0..) |*x, i| x.* = if (i < 11) @as(u32, @intCast(i + 1)) else @intCast(i + 6_000_001);
    var slot2 = try KVCache.init(testing.allocator, 8);
    defer slot2.deinit();
    var off2: usize = 0;
    const hit2 = try hc2.lookupAndRestore(&slot2, &off2, s, &cold, false, &.{}, null);
    try t.expectEqual(@as(usize, 11), hit2.matched);
}

test "SSD-first: the disk flush carries the full prefix while RAM keeps a trim" {
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 1200);
    var probe = try cache.snapshot();
    const row_bytes = HotPrefixCache.snapshotRowBytes(&probe);
    probe.deinit();
    try testing.expect(row_bytes > 0);
    const budget: u64 = row_bytes * 768;

    // Arm A (ssd_first on): RAM trims, the disk entry covers the full prompt.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-on", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        try testing.expect(hc.entries.items[0].tokens.len < tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
        try testing.expectEqual(@as(usize, tokens.len), hc.disk.?.entries.items[0].tokens.len);
        try testing.expect(hc.pending_disk == null);
    }

    // Arm B (ssd_first off): the disk copy is exactly what RAM retained.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-off", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expect(hc.disk.?.entries.items[0].kv_len < tokens.len);
    }
}

/// SSD-first tier over a tmp dir with one KV chunk per flush; the caller frees `tmp` and `hc`.
fn testBoundedSsdFirst(tmp: *std.testing.TmpDir, buf: []u8, row_bytes: u64) !HotPrefixCache {
    const root_len = try tmp.dir.realPath(std.testing.io, buf);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, row_bytes * 768);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, std.testing.io, buf[0..root_len], "fp-ssd-bounded", 0, 128);
    hc.disk.?.max_flush_bytes = 1; // one chunk per append
    return hc;
}

test "SSD-first: a bounded flush keeps its snapshot and lands it whole over later flushes, one chunk each (#797)" {
    const s = mlx.gpuStream();
    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 1200);
    var probe = try cache.snapshot();
    const row_bytes = HotPrefixCache.snapshotRowBytes(&probe);
    probe.deinit();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    var hc = try testBoundedSsdFirst(&tmp, &buf, row_bytes);
    defer hc.deinit();

    _ = try hc.commit(&cache, &tokens, false);
    try testing.expect(hc.entries.items[0].tokens.len < tokens.len); // RAM holds only a trim
    var landed: u32 = 0;
    var flushes: usize = 0;
    while (hc.pending_disk != null and flushes < 32) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const now = hc.disk.?.entries.items[0].kv_len;
        try testing.expect(now - landed <= 128); // the readback bound holds per flush
        landed = now;
    }
    try testing.expect(flushes > 1);
    try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
}

fn testDiskLen(d: *const kv_disk_cache.DiskTier, first_token: u32) u32 {
    for (d.entries.items) |e| {
        if (e.tokens.len > 0 and e.tokens[0] == first_token) return e.kv_len;
    }
    return 0;
}

test "SSD-first: a different session's commit parks the pending snapshot; later flushes land both whole, one chunk each (#797)" {
    const s = mlx.gpuStream();
    var a_tokens: [1200]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var b_tokens: [1200]u32 = undefined;
    for (&b_tokens, 0..) |*t, i| t.* = @intCast(i + 50_000);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 1200);
    var probe = try cache.snapshot();
    const row_bytes = HotPrefixCache.snapshotRowBytes(&probe);
    probe.deinit();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    var hc = try testBoundedSsdFirst(&tmp, &buf, row_bytes);
    defer hc.deinit();

    _ = try hc.commit(&cache, &a_tokens, false);
    hc.flushPendingDisk(s); // one slice of A
    const d = &hc.disk.?;
    const a_sliced = testDiskLen(d, a_tokens[0]);
    try testing.expect(a_sliced < a_tokens.len);
    _ = try hc.commit(&cache, &b_tokens, false); // B's commit writes nothing of A
    try testing.expectEqual(a_sliced, testDiskLen(d, a_tokens[0]));
    var written: u32 = a_sliced;
    var flushes: usize = 0;
    while ((hc.pending_disk != null or hc.finishing_disk != null) and flushes < 64) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const now = testDiskLen(d, a_tokens[0]) + testDiskLen(d, b_tokens[0]);
        try testing.expect(now - written <= 128); // the readback bound holds per flush
        written = now;
    }
    try testing.expectEqual(@as(u32, a_tokens.len), testDiskLen(d, a_tokens[0]));
    try testing.expectEqual(@as(u32, b_tokens.len), testDiskLen(d, b_tokens[0]));
}

test "SSD-first companion: a restore adopts the entry's buffer when its capacity suffices" {
    // A grow is not in place, so a restore must land in the donor's buffer (which already
    // carries the previous turn's reservation) rather than allocate the entry's whole size.
    const s = mlx.gpuStream();
    const Grows = &transformer_mod.KVCache.kv_cap_buf_grows;

    // Turn 1: a reserved cache grows once, to the reservation.
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(4096);
    const g0 = Grows.*;
    try testFillCache(&donor, s, 1, 600);
    try testing.expectEqual(@as(usize, 1), Grows.* - g0);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try hc.commit(&donor, &tokens, false);

    // Turn 2: restore into a fresh slot cache that reserves the same length: nothing allocates.
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&slot, &moe_off, s, &tokens, false, &.{}, null);
    try testing.expect(res.full_match);
    slot.reserve(4096);
    const g1 = Grows.*;
    try testFillCache(&slot, s, 1, 8); // the diverged tail
    try testing.expectEqual(@as(usize, 0), Grows.* - g1);

    // A reservation is not retroactive: reserving more than the donor holds still allocates nothing.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    _ = try hc.lookupAndRestore(&slot2, &moe_off2, s, &tokens, false, &.{}, null);
    slot2.reserve(65536);
    const g2 = Grows.*;
    try testFillCache(&slot2, s, 1, 8);
    try testing.expectEqual(@as(usize, 0), Grows.* - g2);

    // Negative arm: writing past the donor's capacity does grow, exactly once.
    const g3 = Grows.*;
    try testFillCache(&slot2, s, 1, 4096);
    try testing.expect(Grows.* - g3 >= 1);
}

test "SSD-first: an idle entry spills to disk and leaves RAM; the active session stays" {
    // RAM floors at one entry under SSD-first: everything but the active session goes to the
    // SSD, and only once its copy is complete.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Arm A: SSD-first spills the idle session.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());

        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
        // ...and A is still served, from disk.
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
        hc.disk.?.drainWriter();
        var back = try KVCache.init(testing.allocator, 1);
        defer back.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestore(&back, &moe_off, s, &tokens_a, false, &.{}, null);
        // A restore always leaves the last token to forward.
        try testing.expectEqual(@as(usize, 599), res.matched);
    }

    // Arm B: every other arch keeps both entries resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-nospill", 0, 128);
        defer hc.deinit();
        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

/// The idle-spill path's premise: an entry only in RAM, its pending snapshot already gone.
fn testDropPending(hc: *HotPrefixCache) void {
    if (hc.pending_disk) |*p| p.deinit(testing.allocator);
    hc.pending_disk = null;
}

test "SSD-first: an in-flight write does not stall the tick — the entry is re-checked next pass" {
    // The durability check must not drain the writer on the inference thread: an entry whose
    // files are still staged is not evictable on this pass, and the next pass evicts.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-inflight", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    // Generous allowance: this test is about the write state, not the cap.
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);

    testDropPending(&hc); // the spill path: A only in RAM
    _ = try hc.commit(&cache, &tok_b, false);

    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    // It returned, the index knows the entry, and RAM still holds it.
    try testing.expect(hc.disk.?.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Let the writer run and drop the allowance: the next pass evicts.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);
}

test "SSD-first: the idle ALLOWANCE bounds eviction, not the fact of being idle" {
    // The spill used to evict every non-newest entry on every finish, ignoring
    // `--prefix-cache-mem`; writing stays unconditional, evicting is what the allowance bounds.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Two sessions, an allowance that covers the idle one: both stay.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for one idle entry
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        // ...and the write still happened.
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // A third session past the allowance: the oldest idle entry goes, and only that one.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow3", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for ONE of the two idle
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
        var saw_b = false;
        var saw_c = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_b)) saw_b = true;
            if (std.mem.eql(u32, e.tokens, &tok_c)) saw_c = true;
        }
        try testing.expect(saw_b and saw_c);
    }

    // Allowance 0 means what it says: nothing idle stays resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow0", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }
}

test "SSD-first: the allowance is a HARD cap, shed in two tiers (durable first)" {
    // Shed the entries that have a durable copy first, then the rest: an unpersistable entry
    // survives while the cache is under the cap and is dropped only past it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // A is the oldest and unpersistable (a layer offset short of the persist target); B is newer and persists; C is active.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) e.snapshot.entries[0].offset = 300;
        }
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);

        // Tier 1 shed B, the durable one, even though A is older.
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        var saw_a = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) saw_a = true;
            try testing.expect(!std.mem.eql(u32, e.tokens, &tok_b));
        }
        try testing.expect(saw_a);

        // Allowance zero: A has nowhere to go and the cap is hard, so tier 2 drops it.
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }

    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2b", 0, 128);
        defer hc.deinit();
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
    }
}

test "SSD-first: a silent SKIP is not a durable copy — the idle entry stays resident" {
    // Every silent skip used to read as "the SSD holds this session" and evicted the RAM copy.
    // Four skip reasons, each asserting both halves: the tier holds nothing, RAM still holds the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    // Arm 1: the volume declined the store.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-declined", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try hc.commit(&cache, &tokens_a, false);

        testDropPending(&hc); // the spill path: A only in RAM
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 2: under `MIN_PERSIST_TOKENS`.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 400);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-short", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try hc.commit(&cache, tokens_a[0..400], false);
        testDropPending(&hc); // the spill path: A only in RAM
        _ = try hc.commit(&cache, tokens_b[0..400], false);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 3: a layer offset short of the persist target.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-offset", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try hc.commit(&cache, &tokens_a, false);
        testDropPending(&hc); // the spill path: A only in RAM
        _ = try hc.commit(&cache, &tokens_b, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.entries[1].offset = 300;
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: a PARTIAL copy is not a copy — the idle entry stays resident" {
    // A byte-capped flush lands real bytes and stops on a chunk boundary: the entry on disk is short and RAM keeps it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-partial", 0, 128);
    defer hc.deinit();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
    // One byte: the loop writes chunk 0 and stops.
    hc.disk.?.max_flush_bytes = 1;

    _ = try hc.commit(&cache, &tokens_a, false);

    testDropPending(&hc); // the spill path: A only in RAM
    _ = try hc.commit(&cache, &tokens_b, false);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].kv_len < 600);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Lift the cap and the allowance: the next pass completes the copy and the durable entry goes.
    hc.disk.?.max_flush_bytes = 512 * 1024 * 1024;
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

test "DiskTier.holdsFullPrefix: the INDEX must agree before a RAM copy is discarded" {
    // `.persisted` is the write path's claim; this is the manifest's.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-holds", 0, 128);
    defer tier.deinit();

    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    tier.drainWriter();
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, true, cache.config, &.{}));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, .{ .scheme = .affine, .bits = 4, .group_size = 64 }, &.{}));

    // A truncated tail chunk, the shape `scan` records after a kill -9.
    const cb = tier.entries.items[0].chunk_bytes;
    const keep = cb[cb.len - 1];
    cb[cb.len - 1] = 0;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));
    cb[cb.len - 1] = keep;
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));

    tier.entries.items[0].kv_len = 400;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));
}

test "SSD-first: one resident session makes reclaimableBytes truthfully ZERO" {
    // Under SSD-first RAM holds exactly the active session at rest, so the largest entry is the
    // only entry and the provable discount is 0 (its buffers are the live KV).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-reclaim", 0, 128);
    defer hc.deinit();

    _ = try hc.commit(&cache, &tokens_a, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.residentBytes() > 0);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // Mid-switch, two sessions are briefly resident and the non-active one is reclaimable.
    _ = try hc.commit(&cache, &tokens_b, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(hc.reclaimableBytes() > 0);

    // ...and the idle spill returns it to 0 without evicting the session being served.
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
}

test "reclaimableBytesFor: only an entry the PROMPT could restore from is unevictable" {
    // The guard's credit with the prompt in hand: the prompt-blind rule always subtracts the
    // whole cache under one-session-resident.
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const resident = hc.residentBytes();
    try testing.expect(resident > 0);

    // (1) The prompt extends the resident session: nothing to reclaim.
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&tokens_a));
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // (2) A different session's prompt: the bytes are reclaimable.
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&tokens_b));
    try testing.expect(hc.reclaimableBytesFor(&tokens_b) > hc.reclaimableBytes());

    // A prefix too short to restore from does not pin the entry either.
    var barely: [600]u32 = undefined;
    for (&barely, 0..) |*t, i| t.* = @intCast(i + 7);
    for (barely[MIN_CANCELLED_COMMIT_TOKENS - 8 ..]) |*t| t.* = 424_242;
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&barely));

    // (3) Two entries, prompt matches one: only that one is withheld.
    _ = try hc.commit(&cache, &tokens_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const both = hc.residentBytes();
    const credit_a = hc.reclaimableBytesFor(&tokens_a);
    try testing.expect(credit_a > 0 and credit_a < both);
    try testing.expect(credit_a >= hc.reclaimableBytes());
}

test "EntryDigest: the published snapshot answers the reclaimable question without the cache" {
    // The guard reads a published snapshot of these digests instead of the cache.
    const s = mlx.gpuStream();
    const A = testing.allocator;

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var short: [64]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(A, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(A, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    const resident = hc.residentBytes();

    // Publish, then replace: the superseded slice is the caller's to free.
    var d1 = try hc.digestsAlloc(A);
    try testing.expectEqual(@as(usize, 1), d1.len);
    try testing.expectEqual(resident, d1[0].kv_bytes);
    _ = try hc.commit(&cache, &tokens_b, false);
    const d2 = try hc.digestsAlloc(A);
    A.free(d1);
    d1 = d2;
    defer A.free(d1);
    try testing.expectEqual(@as(usize, 2), d1.len);

    const fp_a = HotPrefixCache.prefixFingerprint(&tokens_a).?;
    const both = hc.residentBytes();
    const credit_a = HotPrefixCache.reclaimableFromDigests(d1, both, fp_a);
    try testing.expect(credit_a > 0 and credit_a < both);

    // A prompt matching neither session credits the whole residency.
    var tokens_c: [600]u32 = undefined;
    for (&tokens_c, 0..) |*t, i| t.* = @intCast(i + 500_000);
    const fp_c = HotPrefixCache.prefixFingerprint(&tokens_c);
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, fp_c));

    // A prompt under the restore floor pins nothing, and hashes to null.
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.prefixFingerprint(&short));
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, null));

    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_a), credit_a);
    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_c), both);
}

test "prefix cache: an oversized commit names WHICH outcome declined it" {
    const a = TrimDecline.no_restorable_prefix.reason();
    const b = TrimDecline.snapshot_copy_failed.reason();
    try testing.expect(a.len > 0 and b.len > 0);
    try testing.expect(!std.mem.eql(u8, a, b));
}

test "prefix cache: the trim's row price is the entry's own bytes divided by its rows" {
    // The trim price divides each of the six quantized arrays by its own `shape[2]`, a layout
    // assumption; pin the invariant (price x capacity == snapshot bytes) rather than the constant.
    const s = mlx.gpuStream();

    for ([_]kv_quant.KVQuantConfig{
        kv_quant.KVQuantConfig.dense,
        kv_quant.KVQuantConfig.affine(8),
        kv_quant.KVQuantConfig.affine(4),
    }) |cfg| {
        var cache = try KVCache.initWithConfig(testing.allocator, 2, cfg);
        defer cache.deinit();

        // An attention shape: 2 kv heads, head_dim 256.
        const mk = struct {
            fn f(str: mlx.mlx_stream, len: c_int) !mlx.mlx_array {
                const shape = [_]c_int{ 1, 2, len, 256 };
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
                return a;
            }
        }.f;
        // Two writes so the second crosses a growth event.
        for ([_]c_int{ 64, 40 }) |n| {
            const k = try mk(s, n);
            defer _ = mlx.mlx_array_free(k);
            var dv = try cache.update(0, k, k, s, 0);
            dv.deinit();
            const k2 = try mk(s, n);
            defer _ = mlx.mlx_array_free(k2);
            var dv2 = try cache.update(1, k2, k2, s, 0);
            dv2.deinit();
        }

        var snap = try cache.snapshot();
        defer snap.deinit();

        const row_bytes = HotPrefixCache.snapshotRowBytes(&snap);
        const total = HotPrefixCache.snapshotBytes(&snap);
        try testing.expect(row_bytes > 0);

        const cap: u64 = @intCast(mlx.getShape(snap.entries[0].keys)[2]);
        try testing.expect(cap >= 104);
        try testing.expectEqual(total, row_bytes * cap);

        // Per token: 2 layers of (K+V) at 2 heads x 256 dims; affine packs to `bits` plus scale and bias per group of 64.
        const per_layer: u64 = switch (cfg.scheme) {
            .off => 2 * (2 * 256 * 2),
            else => blk: {
                const packed_b: u64 = 2 * 256 * @as(u64, cfg.bits) / 8;
                const groups: u64 = 2 * 256 / cfg.group_size;
                break :blk 2 * (packed_b + groups * 2 * 2);
            },
        };
        try testing.expectEqual(2 * per_layer, row_bytes);
    }
}

test "HotPrefixCache: a bounded disk flush leaves disk_dirty set and later flushes complete the entry — never a whole-entry claim in between" {
    // A long entry reaches the tier in pieces; each piece is a valid shorter entry and the next
    // `flushPendingDisk` extends it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-partial", 0, 128);
    defer hc.deinit();
    hc.disk.?.max_flush_bytes = 1; // one chunk per flush: the bounded shape

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    _ = try hc.commit(&cache, &tokens, false);
    try testing.expect(hc.disk_dirty);

    // 600 tokens at 128/chunk = 5 chunks; each flush lands one.
    var flushes: usize = 0;
    var last_kv: u32 = 0;
    while (hc.disk_dirty and flushes < 10) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const d = &hc.disk.?;
        try testing.expectEqual(@as(usize, 1), d.entryCount());
        const kv = d.entries.items[0].kv_len;
        try testing.expect(kv >= last_kv);
        if (hc.disk_dirty) try testing.expectEqual(@as(u32, 0), kv % 128);
        const m = d.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, &.{}).?;
        try testing.expectEqual(kv, m.usable);
        last_kv = kv;
    }
    try testing.expect(!hc.disk_dirty);
    try testing.expectEqual(@as(usize, 5), flushes);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

/// Persisted rows of the disk entry whose token record is exactly `toks`; 0 = none.
fn testDiskKvLen(d: *const kv_disk_cache.DiskTier, toks: []const u32) u32 {
    for (d.entries.items) |*e| {
        if (std.mem.eql(u32, e.tokens, toks)) return e.kv_len;
    }
    return 0;
}

/// Three distinct 600-token sessions over one filled cache (5 chunks of 128 each).
fn testThreeSessions() [3][600]u32 {
    var toks: [3][600]u32 = undefined;
    for (&toks, 0..) |*row, k| {
        for (row, 0..) |*t, i| t.* = @intCast(i + 7 + k * 100_000);
    }
    return toks;
}

/// Is `toks` whole on the tier, by the index and by the files?
fn testWholeOnDisk(d: *kv_disk_cache.DiskTier, cache: *const KVCache, toks: []const u32) bool {
    const id = d.fullPrefixEntryId(cache.entries, cache.step, toks, false, cache.config, &.{}) orelse return false;
    return d.entryWholeOnDisk(id);
}

test "HotPrefixCache: idle flushes land every finished session whole, oldest owed first" {
    // Sessions that finish back to back each land one slice. The idle loop must then finish all of
    // them, starting with the oldest; finishing only the newest restored the others short on restart.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-idle-all", 0, 128);
    defer hc.deinit();
    hc.disk.?.max_flush_bytes = 1; // one chunk per slice
    const d = &hc.disk.?;

    for (&toks) |*t| {
        _ = try hc.commit(&cache, t, false);
        hc.flushPendingDisk(s); // the finish path
    }
    for (&toks) |*t| try testing.expectEqual(@as(u32, 128), testDiskKvLen(d, t));

    try testing.expect(hc.flushOwedSlice(s));
    try testing.expectEqual(@as(u32, 256), testDiskKvLen(d, &toks[0]));
    try testing.expectEqual(@as(u32, 128), testDiskKvLen(d, &toks[1]));
    try testing.expectEqual(@as(u32, 128), testDiskKvLen(d, &toks[2]));

    var calls: usize = 1;
    while (calls < 50 and hc.flushOwedSlice(s)) calls += 1;
    try testing.expect(calls < 50);
    try testing.expect(!hc.disk_dirty);
    for (&toks) |*t| try testing.expect(testWholeOnDisk(d, &cache, t));
}

test "HotPrefixCache: the idle flush skips a checked-out entry and never reports an unwritable one owed" {
    // A checked-out entry's bytes are the slot's; an entry under the persist floor never lands.
    // Neither may keep the idle loop spinning, and a returned checkout is owed again.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-idle-skip", 0, 128);
    defer hc.deinit();
    const d = &hc.disk.?;

    _ = try hc.commit(&cache, toks[1][0..100], false); // under MIN_PERSIST_TOKENS
    _ = try hc.commit(&cache, &toks[0], false);
    (try testEntryFor(&hc, &toks[0])).checked_out_by = 1;

    var calls: usize = 0;
    while (calls < 5 and hc.flushOwedSlice(s)) calls += 1;
    try testing.expect(calls < 5);
    try testing.expectEqual(@as(usize, 0), d.entryCount());

    hc.releaseCheckout(1, "test");
    calls = 0;
    while (calls < 5 and hc.flushOwedSlice(s)) calls += 1;
    try testing.expect(calls < 5);
    try testing.expect(testWholeOnDisk(d, &cache, &toks[0]));
}

test "HotPrefixCache: a failed background write costs only its own session; the idle flush moves on" {
    // Failure matrix row 1: the writer loses one of the oldest session's chunks. The others still
    // land whole and the idle loop ends.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-idle-fail", 0, 128);
    defer hc.deinit();
    const d = &hc.disk.?;
    d.enableBackgroundWriter();
    try testing.expect(d.writer != null);
    d.max_flush_bytes = 1;

    for (&toks) |*t| {
        _ = try hc.commit(&cache, t, false);
        hc.flushPendingDisk(s);
    }
    d.writer.?.injectFailure("/e1/c000002", .write);

    var calls: usize = 0;
    while (calls < 50 and hc.flushOwedSlice(s)) {
        d.drainWriter(); // let each failure land before the next slice reads the index
        calls += 1;
    }
    try testing.expect(calls < 50);
    d.drainWriter();
    _ = d.harvestWriteFailures();
    try testing.expect(d.writeErrors() >= 1);
    try testing.expect(testWholeOnDisk(d, &cache, &toks[1]));
    try testing.expect(testWholeOnDisk(d, &cache, &toks[2]));
}

test "HotPrefixCache: a RAM-first cache arms the background writer and chunk share" {
    // RAM-first used to write synchronously and rewrite every chunk of a turn that diverged
    // inside the generated span. SSD-first stays as it was.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const orig_ssd = ssd_first_override;
    defer ssd_first_override = orig_ssd;
    ssd_first_override = true; // the env switch is not on trial
    kv_disk_cache.chunk_share_override = true;
    defer kv_disk_cache.chunk_share_override = null;
    const config = model_mod.ModelConfig{ .model_type = "gemma4" };

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tok_a: [600]u32 = undefined;
    var tok_b: [600]u32 = undefined;
    for (&tok_a, &tok_b, 0..) |*a, *b, i| {
        a.* = @intCast(i + 7);
        b.* = if (i < 520) @intCast(i + 7) else @intCast(9000 + i); // 4 whole chunks shared
    }

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-arm-ram", 0, 128);
        defer hc.deinit();
        hc.armDiskTier(&config);
        const d = &hc.disk.?;
        try testing.expect(!hc.ssd_first);
        try testing.expect(!d.ssd_first);
        try testing.expect(d.writer != null);
        try testing.expectEqual(@as(u64, 512 * 1024 * 1024), d.max_flush_bytes);

        _ = try hc.commit(&cache, &tok_a, false);
        while (hc.flushOwedSlice(s)) {}
        d.drainWriter();
        _ = try hc.commit(&cache, &tok_b, false);
        while (hc.flushOwedSlice(s)) {}
        d.drainWriter();
        try testing.expectEqual(@as(usize, 2), d.entryCount());
        var inherited: u32 = 0;
        for (d.entries.items) |*e| inherited = @max(inherited, e.inherited_chunks);
        try testing.expectEqual(@as(u32, 4), inherited);
        try testing.expect(testWholeOnDisk(d, &cache, &tok_b));
    }
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0); // --no-prefix-cache-ram
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-arm-ssd", 0, 128);
        defer hc.deinit();
        hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        hc.armDiskTier(&config);
        try testing.expect(hc.ssd_first);
        try testing.expect(hc.disk.?.ssd_first);
        try testing.expect(hc.disk.?.writer != null);
    }
}

/// Advances `step` ms on every read, so a deadline cuts after a known number of slices.
const TestStepClock = struct {
    now: *i64,
    step: i64,
    pub fn nowMs(self: TestStepClock) i64 {
        self.now.* += self.step;
        return self.now.*;
    }
};

/// Three sessions committed back to back, each with one finish-path slice landed (128 of 600).
fn testOwedSessions(hc: *HotPrefixCache, cache: *const KVCache, toks: *const [3][600]u32, s: mlx.mlx_stream) !void {
    hc.disk.?.max_flush_bytes = 1; // one chunk per slice
    for (toks) |*t| {
        _ = try hc.commit(cache, t, false);
        hc.flushPendingDisk(s);
    }
}

test "HotPrefixCache: the shutdown flush lands every owed session whole and tallies it" {
    // SIGTERM with sessions still owed slices: before the deadline, each lands whole.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sd-all", 0, 128);
    defer hc.deinit();
    try testOwedSessions(&hc, &cache, &toks, s);

    var now: i64 = 0;
    const t = shutdownPersist(&.{.{ .hc = &hc, .s = s }}, TestStepClock{ .now = &now, .step = 1 }, 1_000_000);
    for (&toks) |*k| try testing.expect(testWholeOnDisk(&hc.disk.?, &cache, k));
    try testing.expectEqual(@as(u32, 3), t.written);
    try testing.expectEqual(@as(u32, 0), t.left_deadline);
    try testing.expectEqual(@as(u64, 0), t.failures);
    try testing.expect(t.bytes > 0);
    try testing.expect(!hc.disk_dirty);
}

test "HotPrefixCache: a shutdown deadline stops between slices, oldest first, and leaves valid shorter entries" {
    // The clock reads 10 before slice 1 and 20 after it; the next start (30) plus the longest
    // slice (10) passes 35, so exactly one slice runs. A restart must read what landed.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sd-cut", 0, 128);
    defer hc.deinit();
    try testOwedSessions(&hc, &cache, &toks, s);

    var now: i64 = 0;
    const t = shutdownPersist(&.{.{ .hc = &hc, .s = s }}, TestStepClock{ .now = &now, .step = 10 }, 35);
    try testing.expectEqual(@as(u32, 3), t.left_deadline);
    try testing.expectEqual(@as(u32, 0), t.written);
    try testing.expectEqual(@as(u32, 256), testDiskKvLen(&hc.disk.?, &toks[0]));
    try testing.expectEqual(@as(u32, 128), testDiskKvLen(&hc.disk.?, &toks[1]));
    try testing.expectEqual(@as(u32, 128), testDiskKvLen(&hc.disk.?, &toks[2]));

    var again = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sd-cut", 0, 128);
    defer again.deinit();
    try testing.expectEqual(@as(usize, 3), again.entryCount());
    try testing.expectEqual(@as(u32, 256), testDiskKvLen(&again, &toks[0]));
    try testing.expectEqual(@as(u32, 128), testDiskKvLen(&again, &toks[2]));
}

test "HotPrefixCache: the shutdown flush skips a checked-out entry and counts it" {
    // Its bytes are the in-flight slot's; the flush writes the others and reports the skip.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sd-out", 0, 128);
    defer hc.deinit();
    try testOwedSessions(&hc, &cache, &toks, s);
    (try testEntryFor(&hc, &toks[0])).checked_out_by = 1;
    defer hc.releaseCheckout(1, "test");

    var now: i64 = 0;
    const t = shutdownPersist(&.{.{ .hc = &hc, .s = s }}, TestStepClock{ .now = &now, .step = 1 }, 1_000_000);
    try testing.expectEqual(@as(u32, 1), t.checked_out);
    try testing.expectEqual(@as(u32, 2), t.written);
    try testing.expectEqual(@as(u32, 128), testDiskKvLen(&hc.disk.?, &toks[0]));
    try testing.expect(testWholeOnDisk(&hc.disk.?, &cache, &toks[1]));
}

test "HotPrefixCache: a failed write during the shutdown flush is counted and the flush moves on" {
    // Failure matrix row 1 at SIGTERM: the writer loses one of the oldest session's chunks.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sd-fail", 0, 128);
    defer hc.deinit();
    hc.disk.?.enableBackgroundWriter();
    try testing.expect(hc.disk.?.writer != null);
    try testOwedSessions(&hc, &cache, &toks, s);
    hc.disk.?.drainWriter();
    hc.disk.?.writer.?.injectFailure("/e1/c000002", .write);

    var now: i64 = 0;
    const t = shutdownPersist(&.{.{ .hc = &hc, .s = s }}, TestStepClock{ .now = &now, .step = 1 }, 1_000_000);
    try testing.expect(t.failures >= 1);
    try testing.expectEqual(@as(u32, 0), t.left_deadline);
    try testing.expect(testWholeOnDisk(&hc.disk.?, &cache, &toks[1]));
    try testing.expect(testWholeOnDisk(&hc.disk.?, &cache, &toks[2]));
}

test "HotPrefixCache: the shutdown flush reports a volume under its reserve as declined, not written" {
    // A declined store writes nothing; the summary must say why the sessions were left.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    const toks = testThreeSessions();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sd-declined", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);
    _ = try hc.commit(&cache, &toks[0], false);
    _ = try hc.commit(&cache, &toks[1], false);

    var now: i64 = 0;
    const t = shutdownPersist(&.{.{ .hc = &hc, .s = s }}, TestStepClock{ .now = &now, .step = 1 }, 1_000_000);
    try testing.expect(t.left_declined >= 2);
    try testing.expectEqual(@as(u32, 0), t.written);
    try testing.expectEqual(@as(u32, 0), t.left_deadline);
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
}

// ── Restore by move (checkout) ──

/// Live Metal bytes, with the allocator pool pinned first.
fn testLiveBytes(s: mlx.mlx_stream) u64 {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var live: usize = 0;
    _ = mlx.mlx_get_active_memory(&live);
    return @intCast(live);
}

/// Bytes of one layer's key buffer (capacity, not logical rows).
fn testKeyBufferBytes(cache: *KVCache, layer: usize) u64 {
    const sh = mlx.getShape(cache.entries[layer].keys);
    return @as(u64, @intCast(sh[0])) * @as(u64, @intCast(sh[1])) *
        @as(u64, @intCast(sh[2])) * @as(u64, @intCast(sh[3])) * 4; // f32
}

fn testCheckoutCache(hc: *HotPrefixCache, s: mlx.mlx_stream, tokens: []const u32, reserve: usize) !void {
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(reserve);
    try testFillCache(&donor, s, 1, @intCast(tokens.len));
    _ = try hc.commit(&donor, tokens, false);
}

test "restore by move: a full-prefix hit checks the entry out and the append donates in place" {
    // `KVCache.restore` binds through `mlx_array_set`, so the entry keeps a second reference and
    // the first `writeAtOffset` cannot donate: `copy_gpu` privatised the whole prefix (5.13 GB /
    // ~110 ms at 393k, 45% of the warm TTFT). The observable is allocation, not address: under
    // suite-wide pressure the buffer pool recycles addresses.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // The prompt extends the entry: a full-prefix hit whose commit will replace this same entry.
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    // A reservation big enough that one buffer dwarfs any pool noise.
    const reserve: usize = 1 << 20;

    var moved_bytes: [64]f32 = undefined;
    var copied_bytes: [64]f32 = undefined;
    var moved_delta: u64 = 0;
    var copied_delta: u64 = 0;
    var buf_bytes: u64 = 0;

    // Arm A: the move.
    {
        restore_move_override = true;
        defer restore_move_override = null;
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        defer hc.deinit();
        try testCheckoutCache(&hc, s, &tokens, reserve);

        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 0xA11CE);
        try testing.expectEqual(@as(usize, 600), res.matched);
        // The transfer is the scheduler's second step, taken at the last point
        // before the first write (see `donateCheckout`).
        hc.donateCheckout(0xA11CE);

        // The entry gave the buffers up: its handles are empty and it names the slot.
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(?usize, 0xA11CE), e.checked_out_by);
        try testing.expect(e.snapshot.entries[0].keys.ctx == null);
        try testing.expect(e.snapshot.entries[0].values.ctx == null);

        slot.evalState();
        buf_bytes = testKeyBufferBytes(&slot, 0);
        const before = testLiveBytes(s);
        try testWriteCacheLayer(&slot, s, 0, 600, 8);
        slot.evalState();
        moved_delta = testLiveBytes(s) -| before;
        try testReadKeyRows(&slot, 0, 596, &moved_bytes);
    }

    // Arm B: the kill switch, the refcount share, same bytes by a different buffer.
    {
        restore_move_override = false;
        defer restore_move_override = null;
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        defer hc.deinit();
        try testCheckoutCache(&hc, s, &tokens, reserve);

        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 0xA11CE);
        try testing.expectEqual(@as(usize, 600), res.matched);
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(?usize, null), e.checked_out_by);
        try testing.expect(e.snapshot.entries[0].keys.ctx != null);

        slot.evalState();
        try testing.expectEqual(buf_bytes, testKeyBufferBytes(&slot, 0));
        const before = testLiveBytes(s);
        try testWriteCacheLayer(&slot, s, 0, 600, 8);
        slot.evalState();
        copied_delta = testLiveBytes(s) -| before;
        try testReadKeyRows(&slot, 0, 596, &copied_bytes);
    }

    // The share arm had to copy a second capacity-shaped buffer.
    try testing.expect(copied_delta > buf_bytes);
    // The move arm allocated nothing beyond the tail; a quarter of one buffer is a loose ceiling.
    try testing.expect(moved_delta * 4 < buf_bytes);
    try testing.expectEqualSlices(f32, &copied_bytes, &moved_bytes);
}

fn testReadKeyRows(cache: *KVCache, layer: usize, row: usize, out: []f32) !void {
    cache.evalState();
    const p = mlx.mlx_array_data_float32(cache.entries[layer].keys) orelse return error.NotEvaluated;
    for (out, 0..) |*v, i| v.* = p[row * 8 + i];
}

test "a one-token prompt that hits its own entry prefills cold: a restore leaves a token to forward" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{42};
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 64);
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &tokens, false, &.{}, null, 7);
    try testing.expectEqual(@as(usize, 0), res.matched);
    try testing.expectEqual(@as(usize, 0), moe_off);
    try testing.expect(hc.last_restored_used == null);
}

test "restore by move ON DEMAND: a share that does not fit is taken over, off SSD-first" {
    // The admission pass converts a full-entry hit into a checkout; a partial hit and a second call decline.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    var diverged = prompt;
    diverged[500] = 999_999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const partial = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &diverged, false, &.{}, null, 7);
    try testing.expectEqual(@as(usize, 500), partial.matched);
    try testing.expect(!hc.checkoutRestored(7, diverged.len));

    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 7);
    try testing.expectEqual(@as(usize, 600), res.matched);
    try testing.expect(!res.checked_out);
    try testing.expect(hc.checkoutRestored(7, prompt.len));
    try testing.expectEqual(@as(?usize, 7), hc.entries.items[0].checked_out_by);
    try testing.expect(!hc.checkoutRestored(7, prompt.len));

    hc.releaseCheckout(7, "prefill refused");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(?usize, null), hc.entries.items[0].checked_out_by);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx != null);
}

test "restore by move: a partial-prefix hit keeps the refcount-share" {
    // A prompt that diverges from the entry makes no replace promise: its commit lands as a new entry.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    prompt[500] = 999_999; // diverge

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 7);
    try testing.expectEqual(@as(usize, 500), res.matched);
    try testing.expectEqual(@as(?usize, null), hc.entries.items[0].checked_out_by);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx != null);
}

test "restore by move: a refusal BEFORE the append hands the entry back INTACT" {
    // Turn B restored a 364k entry by move, the admission pass refused before any forward, and
    // `finishSlot` dropped the checked-out entry. The checkout is a promise, not a transfer:
    // a slot that ends before the donate hands the entry back as it was.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);
    const billed_before = hc.current_kv_bytes;
    try testing.expect(billed_before > 0);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 42);
    try testing.expect(res.checked_out);
    const lru_before = hc.entries.items[0].last_used;

    // The refusal: `runPrefill` returns before `Generator.initWithOptions`, so
    // `donateCheckout` never ran and the entry never gave its handles up.
    try testing.expect(!hc.entries.items[0].checkout_donated);
    hc.releaseCheckout(42, "prefill refused");

    // The entry is whole: same tokens, live handles, same bill, same LRU position, restorable.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 600), e.tokens.len);
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    try testing.expect(e.snapshot.entries[0].values.ctx != null);
    try testing.expectEqual(billed_before, hc.current_kv_bytes);
    try testing.expectEqual(lru_before, e.last_used);

    // ...and the proof: a second slot restores the same prefix from it.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const again = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, s, &prompt, false, &.{}, null, 43);
    try testing.expectEqual(@as(usize, 600), again.matched);
    hc.releaseCheckout(43, "second slot ended");
}

test "restore by move: a slot that ends without committing DROPS its checked-out entry" {
    // The bytes die with the slot; `finishSlot` releases unconditionally. `testing.allocator` is the free-once bar.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);
    const billed_before = hc.current_kv_bytes;
    try testing.expect(billed_before > 0);

    var slot = try KVCache.init(testing.allocator, 1);
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 42);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    // Admitted and about to write: the handles go over.
    hc.donateCheckout(42);
    try testing.expect(hc.entries.items[0].checkout_donated);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx == null);

    hc.releaseCheckout(42, "cancelled");
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    // Idempotent.
    hc.releaseCheckout(42, "cancelled");
    hc.releaseCheckout(43, "cancelled");
    slot.deinit();
}

test "restore by move: a commit RECLAIMS the checked-out entry with the grown buffers" {
    // The happy path: the replace arm installs the grown snapshot and clears the mark.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 42);
    try testing.expect(hc.entries.items[0].checked_out_by != null);
    try testWriteCacheLayer(&slot, s, 0, 600, 8);

    _ = try hc.commit(&slot, &prompt, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expectEqual(@as(usize, 608), e.tokens.len);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    hc.releaseCheckout(42, "finished");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestore(&slot2, &moe_off2, s, &prompt, false, &.{}, null);
    try testing.expectEqual(@as(usize, 607), res2.matched);
}

test "restore by move: a checked-out entry is invisible to a second slot, to eviction and to the published residency" {
    // Its snapshot is empty: invisible to a second slot, to eviction and to the guard.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &prompt, false, &.{}, null, 42);
    const billed = hc.entries.items[0].kv_bytes;
    try testing.expect(billed > 0);

    // (c) a second slot misses.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, s, &prompt, false, &.{}, null, 43);
    try testing.expectEqual(@as(usize, 0), res2.matched);
    try testing.expect(!res2.full_match);
    try testing.expectEqual(@as(usize, 0), slot2.step);
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    // (d) nothing to reclaim, no digest to publish.
    const digests = try hc.digestsAlloc(testing.allocator);
    defer testing.allocator.free(digests);
    try testing.expectEqual(@as(usize, 0), digests.len);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&prompt));
    // The bill still counts it: the bytes really are resident, in the slot.
    try testing.expectEqual(billed, hc.residentBytes());

    // Eviction cannot take it.
    const Never = struct {
        fn fits(_: ?*anyopaque) bool {
            return false;
        }
    };
    const report = hc.evictLruToAdmit(608, null, Never.fits, false);
    try testing.expectEqual(@as(usize, 0), report.entries);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    hc.releaseCheckout(42, "test teardown");
}

test "restore by move: the policy is off outside the SSD-first arm and under the kill switch" {
    // Eligible: SSD-first, enabled, no pending flush, whole entry matched, something to append.
    try testing.expect(HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(false, true, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, false, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, false));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, true, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 500, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 600, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 0, 0, 608, true));
}

/// The resident entry whose token record is exactly `toks`.
fn testEntryFor(hc: *HotPrefixCache, toks: []const u32) !*Entry {
    for (hc.entries.items) |*e| {
        if (std.mem.eql(u32, e.tokens, toks)) return e;
    }
    return error.EntryGone;
}

test "SSD-first: a chunk write that fails AFTER the pass invalidates the entry — RAM is never dropped against it" {
    // Pass N stages A and skips it (pending). The writer then loses `c000003` but, FIFO, still
    // lands meta.json, so the index keeps a non-zero `chunk_bytes[3]`. Pass N+1 must not call
    // A durable. The bar is the verdict (`spill_durable`), not the eviction it licenses.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-poison", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);

    testDropPending(&hc); // the spill path: A only in RAM
    _ = try hc.commit(&cache, &tok_b, false);

    // Pass N: the writer is held, so A's files stay staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.writer.?.injectFailure("c000003", .write);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const dead_id = hc.disk.?.entries.items[0].id;

    // Between the passes: chunk 3 dies, meta.json lands.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expectEqual(@as(usize, 5), hc.disk.?.entries.items[0].chunk_bytes.len);

    // The failure names its entry, so a restore from it misses.
    try testing.expectEqual(@as(usize, 1), hc.disk.?.harvestWriteFailures());
    try testing.expect(hc.disk.?.entries.items[0].poisoned);
    try testing.expect(hc.disk.?.bestMatch(&tok_a, false, cache.config, &.{}) == null);
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config, &.{}));

    // Pass N+1: A is not durable; the dead directory is reclaimed and the rebuild is staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].id != dead_id);

    // Control: with the writer healthy the same entry is durable and restores whole.
    hc.disk.?.writer.?.injectFailure(null, .write);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);

    var back = try KVCache.init(testing.allocator, 1);
    defer back.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&back, &moe_off, s, &tok_a, false, &.{}, null);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "SSD-first: a write failure inside the SAME pass still keeps the entry resident" {
    // The same-pass interleaving (`writeErrors() != errs_before`), via a submit-time injection.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-samepass", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);

    testDropPending(&hc); // the spill path: A only in RAM
    _ = try hc.commit(&cache, &tok_b, false);
    hc.disk.?.writer.?.injectFailure("c000002", .submit);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config, &.{}));
}

test "SSD-first: a spill pass probes the volume only for a store, never for a copy already on disk" {
    // A spill pass probes the volume only for an entry it will write.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 180_000);
    var tok_d: [600]u32 = undefined;
    for (&tok_d, 0..) |*t, i| t.* = @intCast(i + 270_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-probe", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    const roomy: u64 = 1024 * 1024 * 1024 * 1024;
    hc.disk.?.armTestSpace(roomy, 2 * roomy);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);

    testDropPending(&hc); // the spill path: A only in RAM
    _ = try hc.commit(&cache, &tok_b, false);
    _ = try hc.commit(&cache, &tok_c, false);

    // The first pass stores A and B: each store is gated on a fresh probe.
    const p0 = kv_disk_cache.test_space_probes;
    hc.spillIdleEntries(s);
    try testing.expect(kv_disk_cache.test_space_probes > p0);
    hc.disk.?.drainWriter();

    // Both copies are whole now. Passes that find them on disk probe nothing and call them durable.
    const p1 = kv_disk_cache.test_space_probes;
    hc.spillIdleEntries(s);
    hc.spillIdleEntries(s);
    try testing.expectEqual(p1, kv_disk_cache.test_space_probes);
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expect((try testEntryFor(&hc, &tok_b)).spill_durable);

    // Below the store floor no NEW entry persists, but a copy already on disk is still a copy:
    // it stays restorable, so RAM may be shed against it.
    hc.disk.?.armTestSpace(0, 2 * roomy);
    _ = try hc.commit(&cache, &tok_d, false);
    hc.spillIdleEntries(s);
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expect(!(try testEntryFor(&hc, &tok_c)).spill_durable);

    // Room again: the new idle entry is a store, so its pass probes.
    hc.disk.?.armTestSpace(roomy, 2 * roomy);
    const p2 = kv_disk_cache.test_space_probes;
    hc.spillIdleEntries(s);
    try testing.expect(kv_disk_cache.test_space_probes > p2);
}

test "SSD-first: the durability check STATS the chunks — a truncated file is never durable" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-stat", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);

    testDropPending(&hc); // the spill path: A only in RAM
    _ = try hc.commit(&cache, &tok_b, false);

    // Pass 1 stages A with the writer held; pass 2, after the files land, is the healthy control.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    // One chunk loses its bytes behind the tier's back; every in-memory bar still passes.
    const id = hc.disk.?.entries.items[0].id;
    var sub: [64]u8 = undefined;
    const rel = try std.fmt.bufPrint(&sub, "fp-stat/e{d}/c000002.safetensors", .{id});
    try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = "short" });

    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
}

/// A resident entry a live slot holds: eviction can never take it.
fn pcAppendCheckedOut(hc: *HotPrefixCache, tokens: []const u32, kv_bytes: u64, used: u64) !void {
    try hc.entries.append(testing.allocator, .{
        .tokens = try testing.allocator.dupe(u32, tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = used,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = kv_bytes,
        .checked_out_by = 1,
    });
    hc.current_kv_bytes += kv_bytes;
}

test "HotPrefixCache: a commit whose only eviction candidates are checked out still returns" {
    // Every budget loop in the commit path must stop when eviction can make no progress —
    // SSD-first with two slots reaches exactly that state on the inference thread.
    const s = mlx.gpuStream();
    const fresh = [_]u32{ 90, 91, 92, 93 };
    const held = [_]u32{ 1, 2 };
    const extend = [_]u32{ 1, 2, 3, 4 };
    const big: u64 = 1 << 30;

    // Count cap: the incoming entry has nowhere to go.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, fresh.len);
        try pcAppendCheckedOut(&hc, &held, 0, 1);
        _ = try hc.commit(&cache, &fresh, false);
        try testing.expect(hc.entries.items[0].checked_out_by != null);
    }

    // Append byte budget: already over the cap, nothing reclaimable.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, big);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, fresh.len);
        try pcAppendCheckedOut(&hc, &held, big, 1);
        _ = try hc.commit(&cache, &fresh, false);
        try testing.expect(hc.entries.items[0].checked_out_by != null);
    }

    // Replace path: the target is evictable, the survivors are not.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, big);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, extend.len);
        try pcAppendCheckedOut(&hc, &fresh, big, 1);
        try pcAppendCheckedOut(&hc, &fresh, big, 2);
        var target = try KVCache.init(testing.allocator, 3);
        defer target.deinit();
        try testFillCache(&target, s, 3, held.len);
        _ = try hc.commit(&target, &held, false);
        _ = try hc.commit(&cache, &extend, false);
        try testing.expect(hc.entries.items.len >= 2);
    }
}

test "HotPrefixCache: eviction picks the LRU of the key holding the most entries" {
    var cache = HotPrefixCache.init(testing.allocator, 8);
    defer cache.deinit();
    const key_a: u64 = 0xa;
    const key_b: u64 = 0xb;
    // C (conversation, key A, the global LRU) then D1..D3 (sweep docs, key B).
    const keys = [_]u64{ key_a, key_b, key_b, key_b };
    for (keys, 1..) |k, used| {
        try cache.entries.append(testing.allocator, .{
            .tokens = try testing.allocator.dupe(u32, &[_]u32{ 1, @intCast(used) }),
            .has_tools = false,
            .cache_key = k,
            .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
            .last_used = used,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 0,
        });
    }
    // A fourth sweep doc arriving: the sweep evicts its own oldest, never C.
    try testing.expectEqual(@as(?usize, 1), cache.lruIndexExcluding(null, key_b));
    // No incoming key: the largest existing group (B) still pays.
    try testing.expectEqual(@as(?usize, 1), cache.lruIndexExcluding(null, null));
    // D1 protected / checked out: next LRU within B.
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(2, key_b));
    cache.entries.items[1].checked_out_by = 7;
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(null, key_b));
    cache.entries.items[1].checked_out_by = null;
    // One workload = plain LRU: C goes first.
    for (cache.entries.items) |*e| e.cache_key = 0;
    try testing.expectEqual(@as(?usize, 0), cache.lruIndexExcluding(null, 0));
}

test "a restore names its entry, and a commit that extends it in place keeps the id" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var longer: [80]u32 = undefined;
    for (&longer, 0..) |*t, i| t.* = @intCast(i + 7);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 64);
    const id = hc.entries.items[0].id;
    try testing.expect(id != 0);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, s, &longer, false, &.{}, null, 0);
    try testing.expectEqual(id, res.entry_id);

    try testFillCache(&slot, s, 1, longer.len);
    _ = try hc.commit(&slot, &longer, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(id, hc.entries.items[0].id);
}

test "an image turn persists its full media-keyed prefix to the SSD tier" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tokens: [3072]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const media = [_]MediaSpan{.{ .start = 2600, .key = 0xABCD }};
    // v9: the tier record is the WHOLE media-keyed prefix — KV rows past the first
    // item included — and the depth is decided when a request matches it. So each
    // arm states three numbers: `want_kv` rows on disk, `want_media` for a re-send
    // whose spans agree (the media rows come back), `want_text` for the same tokens
    // re-sent as a text turn (clamped at this entry's first item, the old floor).
    const Arm = struct {
        ssd_first: bool = false,
        mem: u64 = 0,
        want_kv: u32,
        want_media: u32,
        want_text: u32,
    };
    const arms = [_]Arm{
        // An agreeing re-send reads every row; a restore that covers
        // the whole prompt still holds back its last row (one token must forward).
        .{ .ssd_first = true, .want_kv = 3072, .want_media = 3071, .want_text = 2600 },
        .{ .want_kv = 3072, .want_media = 3071, .want_text = 2600 },
        .{ .mem = 16 * 1024, .want_kv = 3072, .want_media = 3071, .want_text = 2600 }, // a RAM decline that spills
    };
    for (arms) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        {
            var hc = HotPrefixCache.initWithMem(testing.allocator, 4, arm.mem);
            hc.ssd_first = arm.ssd_first;
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media", 0, 1024);
            defer hc.deinit();
            var cache = try KVCache.init(testing.allocator, 3);
            defer cache.deinit();
            try testFillCache(&cache, s, 3, tokens.len);
            _ = try hc.commitWithMediaState(&cache, &tokens, false, &media, 0, null);
            hc.flushPendingDisk(s);
        }
        // A cold RAM tier (idle unload, restart) holds nothing of its own: the record
        // and reads it back whole when the spans agree; a text re-send stops below the item.
        const Lookup = struct { media: []const MediaSpan, want: u32 };
        const lookups = [_]Lookup{
            .{ .media = &media, .want = arm.want_media },
            .{ .media = &.{}, .want = arm.want_text },
        };
        for (lookups) |lk| {
            var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc2.ssd_first = arm.ssd_first;
            hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media", 0, 1024);
            defer hc2.deinit();
            // Every arm leaves exactly one record: the full media-keyed prefix,
            // stamped with the spans it was keyed under.
            try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
            try testing.expectEqual(arm.want_kv, hc2.disk.?.entries.items[0].kv_len);
            try testing.expectEqualSlices(MediaSpan, &media, hc2.disk.?.entries.items[0].media);
            var dst = try KVCache.init(testing.allocator, 3);
            defer dst.deinit();
            var moe: usize = 0;
            const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, s, &tokens, false, lk.media, null, null, false);
            try testing.expectEqual(@as(usize, lk.want), res.matched);
        }
    }
}
