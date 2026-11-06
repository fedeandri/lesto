//! SSD tier for the hot prefix cache — chunked KV persistence.
//!
//! Committed KV prefixes are persisted to disk as position-chunked
//! safetensors, so previously-seen prefixes survive server restarts and RAM
//! evictions and are RESTORED instead of recomputed. Two-tier flow:
//!
//!   commit  → RAM entry (refcount snapshot, unchanged) + chunk-APPEND to
//!             disk. Only chunks not yet on disk are written — a multi-turn
//!             agent session pays one bounded partial-chunk rewrite + the new
//!             tail per turn, never a full re-serialize.
//!   lookup  → longest-prefix match across RAM entries as before; the disk
//!             index is consulted when it can beat the RAM match by at least
//!             one chunk (fresh boot, post-eviction). The entry is rebuilt
//!             into the live cache and the normal truncate-then-prefill path
//!             continues.
//!
//! Layout (one root per model fingerprint — path + config.json identity):
//!   <base>/<fingerprint>/e<id>/meta.json    commit point (written tmp+rename
//!                                           LAST; an entry without it is a
//!                                           crash leftover and is GC'd)
//!   <base>/<fingerprint>/e<id>/tokens.bin   LE u32 token ids (prompt ++ gen)
//!   <base>/<fingerprint>/e<id>/c000000.safetensors   KV positions [0, chunk_tokens)
//!   <base>/<fingerprint>/e<id>/c000001.safetensors   ...
//!
//! Chunk files hold per-layer K/V slices keyed "l{i}.k"/"l{i}.v" (plus
//! ".ks/.kb/.vs/.vb" scale/bias triples in affine mode). The final chunk may
//! be partial; a commit that extends the entry rewrites ONLY that chunk and
//! appends new ones. A chunk file holding MORE positions than meta.json
//! claims (crash between chunk write and meta rename) is sliced down at
//! restore, never trusted.
//!
//! meta.json still writes `"ssm":[]` (the retired SSM checkpoint list) so a manifest's bytes
//! stay the same; the loader drops an entry that lists a non-empty one.
//!
//! Scope: B==1 slot caches. All mlx work runs on the
//! inference thread; safetensors loads use a private CPU stream
//! (`Load::eval_gpu` is Not Implemented — the lora.zig/model.zig precedent).

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const transformer_mod = @import("transformer.zig");
const model = @import("model.zig");
const io_util = @import("io_util.zig");
const disk_writer = @import("kv_disk_writer.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;

/// One media item's rows in a prompt: its first placeholder row and its
/// pixel/PCM hash. Placeholder ids are identical across items, so state at and
/// after `start` is keyed on `key`. Lives at the disk layer so the tier's
/// per-entry identity can carry spans; `prefix_cache` re-exports it.
pub const MediaSpan = struct { start: u32, key: u64 };

/// First position where state under two span lists can differ; maxInt when
/// they agree. Rows before the first span that differs are text.
pub fn mediaSharedBound(a: []const MediaSpan, b: []const MediaSpan) usize {
    const n = @min(a.len, b.len);
    for (a[0..n], b[0..n]) |x, y| {
        if (x.start != y.start or x.key != y.key) return @min(x.start, y.start);
    }
    if (a.len > n) return a[n].start;
    if (b.len > n) return b[n].start;
    return std.math.maxInt(usize);
}

/// The spans whose rows lie inside `[0, len)`.
pub fn spansBelow(spans: []const MediaSpan, len: usize) []const MediaSpan {
    var n: usize = 0;
    while (n < spans.len and spans[n].start < len) n += 1;
    return spans[0..n];
}

/// Restoring from disk only happens when it beats the best RAM match by at
/// least this many tokens — a disk read + rebuild is only worth it when it
/// replaces a meaningful amount of prefill.
pub const MIN_DISK_ADVANTAGE_TOKENS: u32 = 256;

/// Entries shorter than this are never persisted (a short prefix re-prefills
/// in well under the restore cost).
pub const MIN_PERSIST_TOKENS: u32 = 512;

pub const DEFAULT_CHUNK_TOKENS: u32 = 1024;

/// Budget floor for a decline-spill's flush: the client is already gone, so
/// the only cost is the synchronous write itself (~2-4 s at SSD speeds); the
/// tier's byte budget and LRU eviction are the real bounds. Sized to bank a
/// 122k-token hybrid candidate's KV in one spill.
pub const DECLINE_SPILL_FLUSH_FLOOR: u64 = 4 * 1024 * 1024 * 1024;

/// SSD-first per-flush readback bound: only the device->host copy on the inference thread; the file write is off thread.
pub const SSD_FIRST_READBACK_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The disk budget is derived from the volume, not only the operator's cap.
pub const DISK_RESERVE_CAP: u64 = 64 * 1024 * 1024 * 1024;
/// Below this there is no point storing anything.
pub const DISK_STORE_FLOOR: u64 = 1024 * 1024 * 1024;

/// Bytes this tier may occupy given the operator cap (0 = none) and the volume. Reserve =
/// min(64 GiB, 10% of the volume); under `DISK_STORE_FLOOR` = null ("do not store"), never 0.
pub fn diskBudgetFromFreeSpace(operator_cap: u64, free_bytes: u64, volume_bytes: u64) ?u64 {
    const reserve = @min(DISK_RESERVE_CAP, volume_bytes / 10);
    const avail = free_bytes -| reserve;
    const budget = if (operator_cap == 0) avail else @min(operator_cap, avail);
    if (budget < DISK_STORE_FLOOR) return null;
    return budget;
}

/// macOS `struct statfs`, leading fields only; the rest is slack. std has no binding for it.
const DarwinStatfs = extern struct {
    f_bsize: u32,
    f_iosize: i32,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    tail: [4096]u8,
};

extern "c" fn statfs(path: [*:0]const u8, buf: *DarwinStatfs) c_int;
extern fn msv_volume_free_for_use(path: [*:0]const u8) u64;

pub const VolumeSpace = struct { free: u64, total: u64 };

/// Free and total bytes of the volume holding `path`, or null when the query fails or returns
/// implausible numbers (the plausibility check is the ABI guard).
pub fn volumeSpace(path: []const u8) ?VolumeSpace {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return null;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var st: DarwinStatfs = undefined;
    if (statfs(buf[0..path.len :0].ptr, &st) != 0) return null;
    const bsize: u64 = @intCast(@max(st.f_bsize, 0));
    if (bsize < 512 or bsize > (1 << 20) or !std.math.isPowerOfTwo(bsize)) return null;
    if (st.f_blocks == 0 or st.f_bavail > st.f_blocks) return null;
    const total = bsize *| st.f_blocks;
    // statfs excludes purgeable space the OS releases on demand; ask what a write really gets.
    const granted = msv_volume_free_for_use(buf[0..path.len :0].ptr);
    const free = if (granted > 0 and granted <= total) granted else bsize *| st.f_bavail;
    return .{ .free = free, .total = total };
}

/// How a `DiskTier` asks what the volume has left. Injectable: with the live probe hard-wired
/// every SSD-first test asserted a property of the tester's free disk space.
pub const SpaceProbeFn = *const fn (path: []const u8) ?VolumeSpace;

/// Test hook, armed through `DiskTier.armTestSpace`.
var test_space: ?VolumeSpace = null;
/// How many free-space probes the test hook answered.
pub var test_space_probes: usize = 0;

fn testSpaceProbe(path: []const u8) ?VolumeSpace {
    _ = path;
    test_space_probes += 1;
    return test_space;
}

pub const IndexEntry = struct {
    /// Directory id — the `e<id>` component.
    id: u64,
    /// Full committed token sequence (prompt ++ generated). Owned.
    tokens: []u32,
    /// KV positions actually persisted (== snapshot `step` at commit; may be
    /// tokens.len - 1 when the final sampled token was never forwarded).
    kv_len: u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    /// Total on-disk bytes (chunks + tokens.bin + meta).
    bytes: u64,
    /// Per-chunk file sizes recorded at commit (meta.json "chunk_bytes").
    /// The scan validates actual file sizes against these and clamps kv_len
    /// to the last contiguous valid chunk — a kill -9 mid-flush truncates a
    /// chunk, and restoring it would poison the cache. Owned.
    chunk_bytes: []u64,
    /// The first `inherited_chunks` chunk files are hard links into a donor entry's chunks;
    /// this entry never billed them. meta.json v6 `inherited_chunks`; 0 on older manifests.
    inherited_chunks: u32 = 0,
    /// v4: spec-snapshot sidecar (`spec.safetensors`) byte size; 0 = none.
    /// The same kill -9 salvage as chunks — a size mismatch at scan drops the
    /// SPEC only (a restore then starts blind), never the entry.
    spec_bytes: u64 = 0,
    /// v4: the dflash assistant context persisted in the spec sidecar.
    /// DRAFT-side state - a missing or dropped snap costs acceptance on the
    /// first reused turn, never a token.
    spec_dflash: ?SpecMeta = null,
    /// A background write for this entry failed: never matched, never a donor, never reported
    /// complete; the next commit reclaims its directory. Not persisted (`scan` re-validates).
    poisoned: bool = false,
    /// v9: media items inside `tokens` (ascending by start; empty = a text-only
    /// record). A media entry's KV rows past a placeholder are keyed on the
    /// spans, not the tokens alone: matching, supersede and chunk share clamp at
    /// `mediaSharedBound`. Owned.
    media: []MediaSpan = &.{},
    /// In-process LRU stamp; seeded from meta.json mtime order at scan.
    last_used: u64,
};

/// v4 spec-snapshot metadata for one speculative-side cache (the dflash assistant
/// context). The tensors live in the entry's ONE `spec.safetensors` file,
/// keyed `d{layer}.*`.
pub const SpecMeta = struct {
    /// Absolute trunk position the snapshot's index 0 represents.
    base: u64,
    /// Positions persisted (the snapshot's logical length).
    step: u32,
    /// Layer count of the source cache — a restore target with a different
    /// count declines (KVCache.restore asserts equal lengths).
    layers: u32,
    quant: kv_quant.KVQuantConfig,
};

/// What `appendCommitWithSpec` reads to persist one spec snapshot — the same
/// snapshot-shaped parts the trunk flush takes, plus the base position.
pub const SpecCommit = struct {
    entries: []const transformer_mod.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    base_pos: usize,
};

/// What a commit actually achieved on disk. The old bool meant "nothing more to write", and
/// every silent skip returned it too, so `spillIdleEntries` dropped RAM copies that had no
/// disk copy at all. Only one of the three outcomes is a promise.
pub const PersistOutcome = enum {
    /// The tier holds the full prefix. The only value that may license discarding the RAM copy.
    persisted,
    /// Real bytes landed but the entry is not whole yet; the next commit resumes.
    partial,
    /// Nothing was written and nothing is promised.
    skipped,

    /// "Nothing more for the caller to write", the old bool's meaning.
    pub fn nothingPending(self: PersistOutcome) bool {
        return self != .partial;
    }
};

/// The KV extent one commit would persist, in tokens: clamped to the token record (the cache
/// runs 1-2 positions ahead on EOS turns), from the initialized layers' offset, not `step`
/// (0 on gated-delta hybrids). Shared by the commit and `holdsFullPrefix`.
pub fn persistTargetLen(
    kv_entries: []const transformer_mod.KVCacheEntry,
    step: usize,
    tokens_len: usize,
) usize {
    var max_off: usize = 0;
    for (kv_entries) |*entry| {
        if (entry.initialized and entry.offset > max_off) max_off = entry.offset;
    }
    return @min(@max(step, max_off), tokens_len);
}

pub const SpecKind = enum { dflash };

pub const Match = struct {
    idx: usize,
    /// Shared-prefix length clamped to kv_len — the positions a restore can
    /// actually rebuild.
    usable: u32,
};

fn nbytesOf(a: mlx.mlx_array) u64 {
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// Batched-eval meter for the staged serializer: one increment per chunk file, never per tensor.
pub var serialize_eval_count = std.atomic.Value(u64).init(0);

/// safetensors dtype spelling (`mlx::core::dtype_to_safetensor_str`); unknown dtypes refuse the staged write.
fn safetensorsDtypeName(d: mlx.mlx_dtype) ?[]const u8 {
    return switch (d) {
        .bool_ => "BOOL",
        .uint8 => "U8",
        .uint16 => "U16",
        .uint32 => "U32",
        .uint64 => "U64",
        .int8 => "I8",
        .int16 => "I16",
        .int32 => "I32",
        .int64 => "I64",
        .float16 => "F16",
        .float32 => "F32",
        .bfloat16 => "BF16",
        else => null,
    };
}

/// Raw contiguous bytes of an evaluated, contiguous array; null for a dtype with no accessor.
fn rawBytes(a: mlx.mlx_array) ?[*]const u8 {
    return switch (mlx.mlx_array_dtype(a)) {
        .bool_ => @ptrCast(mlx.mlx_array_data_bool(a) orelse return null),
        .uint8, .int8 => @ptrCast(mlx.mlx_array_data_uint8(a) orelse return null),
        .uint32, .int32 => @ptrCast(mlx.mlx_array_data_uint32(a) orelse return null),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(a) orelse return null),
        .float16 => @ptrCast(mlx.mlx_array_data_float16(a) orelse return null),
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(a) orelse return null),
        else => null,
    };
}

pub const DiskTier = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Absolute root for this model's entries (`<base>/<fingerprint>`). Owned.
    root: []u8,
    /// Byte budget across all entries. 0 = unbounded.
    max_bytes: u64,
    chunk_tokens: u32,
    /// Max bytes written per appendCommit call (default 512 MB). The flush
    /// runs synchronously on the inference thread after the response; a
    /// 4 GB first-commit write measurably stalls the NEXT request, so large
    /// entries persist incrementally across turns (appendCommit reports
    /// incomplete and the hot cache keeps its dirty flag set).
    max_flush_bytes: u64 = 512 * 1024 * 1024,
    /// SSD-first mode (mirrored from `HotPrefixCache.ssd_first`).
    ssd_first: bool = false,
    /// SSD-first background writer (heap-allocated so the mutex survives `init`'s by-value
    /// return). Null = the synchronous `mlx_save_safetensors` path.
    writer: ?*disk_writer.Writer = null,
    /// The operator's `--prefix-cache-disk` value; `max_bytes` is re-derived from it before every store.
    operator_cap: u64 = 0,
    /// The free-space probe; tests arm a fixed answer with `armTestSpace`.
    space_probe: SpaceProbeFn = volumeSpace,
    /// The volume is under `DISK_STORE_FLOOR`: no new entry persists, existing ones stay restorable.
    store_declined: bool = false,
    /// `<base>` (the parent of `root`), for the root-wide sweep. Null when the dupe failed.
    base_dir: ?[]u8 = null,
    entries: std.ArrayList(IndexEntry),
    next_id: u64,
    total_bytes: u64,
    counter: u64,
    /// KV chunk bytes the commits so far wrote (staged, on the writer arm).
    /// Diagnostics; not persisted.
    bytes_persisted: u64 = 0,
    /// Chunk count read by the most recent restore. Diagnostics + a
    /// red-on-revert guard that a short-prefix restore reads only the chunks
    /// covering the usable prefix, not the whole stored entry. Not persisted.
    chunks_loaded_last: u32 = 0,

    /// Create the tier rooted at `<base>/<fingerprint>` and scan whatever
    /// already exists there. Crash leftovers (no meta.json) are deleted.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        base_dir: []const u8,
        fingerprint: []const u8,
        max_bytes: u64,
        chunk_tokens: u32,
    ) !DiskTier {
        if (base_dir.len == 0 or !std.fs.path.isAbsolute(base_dir)) return error.BadDiskCacheDir;
        const root = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fingerprint });
        errdefer allocator.free(root);
        try std.Io.Dir.cwd().createDirPath(io, root);
        var self: DiskTier = .{
            .allocator = allocator,
            .io = io,
            .root = root,
            .max_bytes = max_bytes,
            .operator_cap = max_bytes,
            .chunk_tokens = if (chunk_tokens == 0) DEFAULT_CHUNK_TOKENS else chunk_tokens,
            .entries = std.ArrayList(IndexEntry).empty,
            .next_id = 1,
            .total_bytes = 0,
            .counter = 0,
        };
        self.scan() catch |err| {
            log.warn("[disk-cache] scan failed: {s} — starting empty\n", .{@errorName(err)});
        };
        self.gcToBudget();
        self.base_dir = allocator.dupe(u8, base_dir) catch null;
        return self;
    }

    /// Test hook: answer every free-space probe with these numbers. Every SSD-first test must call this.
    pub fn armTestSpace(self: *DiskTier, free: u64, total: u64) void {
        test_space = .{ .free = free, .total = total };
        self.space_probe = testSpaceProbe;
    }

    /// Arm the background writer (both arms; set `ssd_first` first). A spawn failure keeps the
    /// synchronous path.
    pub fn enableBackgroundWriter(self: *DiskTier) void {
        if (self.writer != null) return;
        const w = self.allocator.create(disk_writer.Writer) catch return;
        w.* = disk_writer.Writer.init(self.allocator, self.io);
        w.start() catch {
            self.allocator.destroy(w);
            log.warn("[disk-cache] background writer unavailable — writing synchronously\n", .{});
            return;
        };
        self.writer = w;
        // With the write off-thread the only inference-thread cost is the readback, and the
        // bound is no longer a correctness cliff (checkpoints ride outside it). RAM-first keeps
        // its bound: an idle slice is what a request arriving mid-flush waits for.
        if (self.ssd_first) self.max_flush_bytes = SSD_FIRST_READBACK_BYTES;
        log.info("[disk-cache] background writer armed (permit {d} MB, readback bound {d} MB/flush)\n", .{
            w.permit_bytes / (1024 * 1024),
            self.max_flush_bytes / (1024 * 1024),
        });
    }

    /// Wait for every staged file to land.
    pub fn drainWriter(self: *DiskTier) void {
        if (self.writer) |w| w.drain();
    }

    /// `drainWriter` bounded by `deadline_ms` on `clock` (any type with `nowMs() i64`). At the
    /// deadline whatever is still queued is fenced, so neither this call nor `deinit` writes it;
    /// each entry keeps its last landed manifest. Returns the bytes dropped.
    pub fn drainWriterUntil(self: *DiskTier, clock: anytype, deadline_ms: i64) u64 {
        const w = self.writer orelse return 0;
        while (true) {
            const left_ms = deadline_ms - clock.nowMs();
            if (left_ms <= 0) break;
            // Short waits, so the deadline is read on the caller's clock.
            const wait: std.Io.Timeout = .{ .duration = .{ .raw = .fromMilliseconds(@min(left_ms, 50)), .clock = .boot } };
            if (w.drainUntil(wait)) return 0;
        }
        const dropped = w.queuedBytes();
        w.fence(null);
        return dropped;
    }

    /// Host bytes staged for the writer and not yet written. Zero when the writer is not armed.
    pub fn stagedHostBytes(self: *DiskTier) u64 {
        const w = self.writer orelse return 0;
        return w.pendingBytes();
    }

    /// Background write failures so far.
    pub fn writeErrors(self: *DiskTier) u64 {
        const w = self.writer orelse return 0;
        return w.writeErrorCount();
    }

    /// Non-blocking: does entry `id` still have files staged or in flight? An unarmed tier
    /// answers false (its writes were synchronous).
    pub fn entryWritesPending(self: *DiskTier, id: u64) bool {
        const w = self.writer orelse return false;
        const pre = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch return true;
        defer self.allocator.free(pre);
        return w.pendingPrefix(pre);
    }

    /// Attribute the writer's failed blobs to the entries that staged them and poison those
    /// entries. A counter alone misses the failure that lands between two spill passes.
    pub fn harvestWriteFailures(self: *DiskTier) usize {
        const w = self.writer orelse return 0;
        const fails = w.takeFailures();
        defer {
            for (fails) |f| self.allocator.free(f.path);
            self.allocator.free(fails);
        }
        var poisoned: usize = 0;
        for (fails) |f| {
            const id = self.entryIdFromPath(f.path) orelse {
                poisoned += self.poisonAll(f.err_name);
                continue;
            };
            for (self.entries.items) |*e| {
                if (e.id != id or e.poisoned) continue;
                self.poisonEntry(e, f.err_name);
                poisoned += 1;
            }
        }
        if (w.takeUnattributed()) poisoned += self.poisonAll("unrecorded");
        return poisoned;
    }

    /// The `e<id>` this absolute path belongs to, or null.
    fn entryIdFromPath(self: *const DiskTier, path: []const u8) ?u64 {
        if (!std.mem.startsWith(u8, path, self.root)) return null;
        var rest = path[self.root.len..];
        if (rest.len == 0 or rest[0] != '/') return null;
        rest = rest[1..];
        if (rest.len < 2 or rest[0] != 'e') return null;
        rest = rest[1..];
        const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
        return std.fmt.parseInt(u64, rest[0..slash], 10) catch null;
    }

    pub fn poisonId(self: *DiskTier, id: u64, err_name: []const u8) bool {
        for (self.entries.items) |*e| {
            if (e.id != id) continue;
            if (!e.poisoned) self.poisonEntry(e, err_name);
            return true;
        }
        return false;
    }

    /// Would a commit of this record resume an entry a failed write invalidated? Attributes the
    /// writer's failures first. A flush gives such a record up instead of rewriting it in a loop.
    pub fn resumesPoisoned(self: *DiskTier, tokens: []const u32, has_tools: bool, media: []const MediaSpan, config: kv_quant.KVQuantConfig) bool {
        _ = self.harvestWriteFailures();
        const media_in = spansBelow(media, tokens.len);
        for (self.entries.items) |*e| {
            if (!e.poisoned or e.has_tools != has_tools or !std.meta.eql(e.quant, config)) continue;
            const n = @min(e.tokens.len, tokens.len);
            if (mediaSharedBound(e.media, media_in) >= n and std.mem.eql(u32, e.tokens[0..n], tokens[0..n])) return true;
        }
        return false;
    }

    /// Mark one entry dead; `chunk_bytes` is zeroed so every "is this whole?" reader answers no.
    fn poisonEntry(self: *DiskTier, e: *IndexEntry, err_name: []const u8) void {
        _ = self;
        e.poisoned = true;
        @memset(e.chunk_bytes, 0);
        log.warn("  [disk-cache] e{d} unusable ({s}) — entry invalidated\n", .{ e.id, err_name });
    }

    fn poisonAll(self: *DiskTier, err_name: []const u8) usize {
        var n: usize = 0;
        for (self.entries.items) |*e| {
            if (e.poisoned) continue;
            self.poisonEntry(e, err_name);
            n += 1;
        }
        return n;
    }

    /// Reclaim the directories of poisoned entries. Runs on the commit path, never on the spill's check.
    fn dropPoisonedEntries(self: *DiskTier) void {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            if (!self.entries.items[i].poisoned) continue;
            log.info("  [disk-cache] dropping invalidated entry e{d}\n", .{self.entries.items[i].id});
            // `removeAt` swap-removes; the element moved into `i` was already checked.
            self.removeAt(i);
        }
    }

    /// Does the filesystem agree with the index about entry `id`? One stat per chunk; catches
    /// a byte that went missing with no failed write at all.
    pub fn entryWholeOnDisk(self: *DiskTier, id: u64) bool {
        for (self.entries.items) |*e| {
            if (e.id != id) continue;
            if (e.poisoned) return false;
            for (e.chunk_bytes, 0..) |want, i| {
                if (want == 0) return false;
                const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, i }) catch return false;
                defer self.allocator.free(cp);
                const st = statFile(self.io, cp) orelse return false;
                if (st.size != want) return false;
            }
            return true;
        }
        return false;
    }

    /// Is entry `id` poisoned, or gone?
    pub fn entryPoisoned(self: *const DiskTier, id: u64) bool {
        for (self.entries.items) |*e| {
            if (e.id == id) return e.poisoned;
        }
        return true;
    }

    /// Wait only for the files of entry `id`.
    fn drainEntry(self: *DiskTier, id: u64) void {
        const w = self.writer orelse return;
        const pre = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch {
            w.drain();
            return;
        };
        defer self.allocator.free(pre);
        w.drainPrefix(pre);
    }

    pub fn deinit(self: *DiskTier) void {
        if (self.base_dir) |b| self.allocator.free(b);
        self.base_dir = null;
        if (self.writer) |w| {
            // Teardown must never block on a paused writer: lift the pause so the queue drains for real.
            w.setPaused(false);
            w.drain();
            w.deinit();
            self.allocator.destroy(w);
            self.writer = null;
        }
        for (self.entries.items) |*e| {
            self.freeIndexEntryOwned(e);
        }
        self.entries.deinit(self.allocator);
        self.allocator.free(self.root);
    }

    /// Free everything an IndexEntry owns. Every removal path (deinit,
    /// eviction, invalidation, scan-append failure, extend-replace) must go
    /// through this so a new owned field can't leak on one of them.
    fn freeIndexEntryOwned(self: *DiskTier, e: *IndexEntry) void {
        self.allocator.free(e.tokens);
        self.allocator.free(e.chunk_bytes);
        self.allocator.free(e.media);
    }

    /// Re-derive `max_bytes` from the volume. A failed probe keeps the operator cap; a budget
    /// under the store floor declines new stores without touching what is already persisted.
    fn refreshDiskBudget(self: *DiskTier) void {
        const vs = self.space_probe(self.root) orelse return;
        // Our own entries are already counted in `used`; add what the tier holds back.
        const budget = diskBudgetFromFreeSpace(self.operator_cap, vs.free +| self.total_bytes, vs.total);
        if (budget) |b| {
            self.store_declined = false;
            if (b != self.max_bytes) {
                self.max_bytes = b;
                self.gcToBudget();
            }
        } else if (!self.store_declined) {
            self.store_declined = true;
            // The number compared is free less the reserve, not free.
            log.warn("[disk-cache] {s}: {d} MB free less the {d} MB reserve (min 64 GiB, 10% of the volume) is below the {d} MB store floor — no NEW entries persist (already-persisted entries stay restorable)\n", .{
                self.root,
                vs.free >> 20,
                @min(DISK_RESERVE_CAP, vs.total / 10) >> 20,
                DISK_STORE_FLOOR >> 20,
            });
        }
    }

    /// Sweep other models' fingerprints under the same base: strays always, LRU once they
    /// collectively exceed one budget's worth. SSD-first only.
    pub fn sweepSiblings(self: *DiskTier) void {
        if (!self.ssd_first) return;
        const base = self.base_dir orelse return;
        self.refreshDiskBudget();
        sweepBase(self.allocator, self.io, base, self.root, self.max_bytes);
    }

    pub fn entryCount(self: *const DiskTier) usize {
        return self.entries.items.len;
    }

    // ── Lookup ──

    /// Longest usable shared prefix across persisted entries with a matching
    /// (has_tools, quant) key. Same filter semantics as the RAM cache: a
    /// cross-config restore would hand SDPA a wrong buffer layout. An entry's
    /// shared prefix also clamps at `mediaSharedBound(entry.media, media)`: a
    /// re-sent turn with agreeing spans reads its media rows back, a divergent
    /// list stops below the first item whose start or pixel key differs, and a
    /// text turn (`media` empty) clamps at every media entry's first item.
    pub fn bestMatch(
        self: *const DiskTier,
        prompt_ids: []const u32,
        has_tools: bool,
        quant: kv_quant.KVQuantConfig,
        media: []const MediaSpan,
    ) ?Match {
        var best_idx: ?usize = null;
        var best_usable: u32 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.poisoned) continue; // a failed write killed it: this is a MISS
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, quant)) continue;
            const max_shared = @min(@min(e.tokens.len, prompt_ids.len), mediaSharedBound(e.media, media));
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
            const usable: u32 = @intCast(@min(shared, e.kv_len));
            if (usable > best_usable) {
                best_usable = usable;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .usable = best_usable };
        return null;
    }

    /// Rebuild the persisted KV state of `entries[idx]` into `cache`:
    /// per-layer chunk tensors are loaded (CPU stream), concatenated along
    /// the sequence axis, and installed as the cache's storage buffers.
    /// Views are left empty — identical contract to `KVCache.restore` (the
    /// next `update`/`truncate` rebuilds them). Returns kv_len.
    pub fn restoreInto(self: *DiskTier, cache: *KVCache, idx: usize, s: mlx.mlx_stream) !u32 {
        const kv_len = self.entries.items[idx].kv_len;
        try self.restorePrefixInto(cache, idx, kv_len, s);
        return kv_len;
    }

    /// Rebuild ONLY positions [0, limit) of `entries[idx]` — `limit` must be
    /// ≤ its kv_len. A short shared prefix against a long stored entry then
    /// reads just the chunks covering `limit` instead of the whole entry (a
    /// diverged-prefix "hit" that would otherwise read every stored chunk to
    /// serve a few hundred tokens — slower than a cold prefill).
    pub fn restorePrefixInto(self: *DiskTier, cache: *KVCache, idx: usize, limit: u32, s: mlx.mlx_stream) !void {
        // Callers swallow a restore failure and prefill cold; the latch it raised must not fail that prefill.
        const had_error = mlx.errorPending();
        errdefer mlx.dropLatchedErrorUnless(had_error);
        // This entry's staged chunks must be on disk before the readback.
        self.drainEntry(self.entries.items[idx].id);
        const e = &self.entries.items[idx];
        try self.restoreKvInto(cache, e, limit, s);
        e.last_used = self.bump();
        // Bump meta.json mtime so cross-restart LRU sees the use.
        self.writeMeta(e.*) catch {};
    }

    /// Shared chunk-loading body: rebuild positions [0, limit) of entry `e`
    /// into `cache` (limit == e.kv_len for the plain-attention path; a
    /// checkpoint position for the hybrid path — the final chunk is sliced
    /// down so KV lands exactly at the checkpoint).
    fn restoreKvInto(self: *DiskTier, cache: *KVCache, e: *const IndexEntry, limit: u32, s: mlx.mlx_stream) !void {
        const quant = e.quant;
        if (!std.meta.eql(cache.config, quant)) return error.DiskCacheConfigMismatch;
        if (limit == 0 or limit > e.kv_len) return error.DiskCacheEmptyEntry;
        const n_chunks: u32 = @intCast((@as(u64, limit) + self.chunk_tokens - 1) / self.chunk_tokens);
        if (n_chunks == 0) return error.DiskCacheEmptyEntry;
        self.chunks_loaded_last = n_chunks;

        const load_s = mlx.weightLoadStream();
        defer _ = mlx.mlx_stream_free(load_s);

        const kinds: []const []const u8 = if (quant.scheme == .off)
            &.{ "k", "v" }
        else
            &.{ "k", "v", "ks", "kb", "vs", "vb" };

        // Per-layer per-kind accumulation vectors. Layers absent from chunk 0
        // stay uninitialized — the GatedDeltaNet layers of a hybrid arch have
        // no KV (their state rides the state checkpoints), so only the
        // full-attention layers appear in the chunks.
        const n_layers = cache.entries.len;
        const vecs = try self.allocator.alloc(mlx.mlx_vector_array, n_layers * kinds.len);
        for (vecs) |*v| v.* = mlx.mlx_vector_array_new();
        defer {
            for (vecs) |v| _ = mlx.mlx_vector_array_free(v);
            self.allocator.free(vecs);
        }
        const present = try self.allocator.alloc(bool, n_layers);
        defer self.allocator.free(present);
        @memset(present, false);

        var chunk_i: u32 = 0;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            const c0: u64 = @as(u64, chunk_i) * self.chunk_tokens;
            const need: u64 = @min(@as(u64, self.chunk_tokens), limit - c0);

            const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors\x00", .{ self.root, e.id, chunk_i });
            defer self.allocator.free(path);
            var tensor_map = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
            var meta_map = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta_map);
            try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), load_s));
            // A lazy load holds its file open until eval, so each chunk is evaluated before the
            // next opens; one eval at the end needed a descriptor per chunk.
            const chunk_arrays = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(chunk_arrays);

            for (0..n_layers) |li| {
                for (kinds, 0..) |kind, ki| {
                    const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}\x00", .{ li, kind });
                    defer self.allocator.free(key);
                    var arr = mlx.mlx_array_new();
                    if (mlx.mlx_map_string_to_array_get(&arr, tensor_map, @ptrCast(key.ptr)) != 0) {
                        _ = mlx.mlx_array_free(arr);
                        if (ki == 0) break; // layer absent from this chunk
                        return error.DiskCacheCorruptChunk;
                    }
                    if (ki == 0) present[li] = true;
                    // Crash-tolerance: a chunk file may hold MORE positions
                    // than meta.json committed to (rewrite raced a crash).
                    // Slice down to the committed range; never trust the file.
                    const shape = mlx.getShape(arr);
                    if (shape.len != 4) {
                        _ = mlx.mlx_array_free(arr);
                        return error.DiskCacheCorruptChunk;
                    }
                    const have: u64 = @intCast(shape[2]);
                    if (have < need) {
                        _ = mlx.mlx_array_free(arr);
                        return error.DiskCacheCorruptChunk;
                    }
                    if (have > need) {
                        var sliced = mlx.mlx_array_new();
                        const st = [_]c_int{ 0, 0, 0, 0 };
                        const sp = [_]c_int{ shape[0], shape[1], @intCast(need), shape[3] };
                        const sd = [_]c_int{ 1, 1, 1, 1 };
                        const rc = mlx.mlx_slice(&sliced, arr, &st, 4, &sp, 4, &sd, 4, s);
                        _ = mlx.mlx_array_free(arr);
                        try mlx.check(rc);
                        arr = sliced;
                    }
                    _ = mlx.mlx_vector_array_append_value(vecs[li * kinds.len + ki], arr);
                    _ = mlx.mlx_vector_array_append_value(chunk_arrays, arr);
                    _ = mlx.mlx_array_free(arr);
                }
            }
            try mlx.check(mlx.mlx_eval(chunk_arrays));
        }

        // Install per-layer concatenations as the cache's storage buffers.
        // Mirrors `KVCache.restore`: views stay empty, offset/initialized set,
        // step = kv_len.
        // A grow is not in place: leave the room the first grow would add, so the slot's next
        // append writes into this buffer instead of copying the whole restored cache.
        const room: c_int = @intCast(KVCache.nextCapacity(limit, @as(usize, limit) + 1) - limit);
        for (cache.entries, 0..) |*dst, li| {
            transformer_mod.resetKVEntry(dst);
            if (!present[li]) continue;
            const base = li * kinds.len;
            for (0..kinds.len) |ki| try appendRoom(vecs[base + ki], room, s);
            try concatInto(&dst.keys, vecs[base + 0], s);
            try concatInto(&dst.values, vecs[base + 1], s);
            if (quant.scheme != .off) {
                try concatInto(&dst.keys_scales, vecs[base + 2], s);
                try concatInto(&dst.keys_biases, vecs[base + 3], s);
                try concatInto(&dst.values_scales, vecs[base + 4], s);
                try concatInto(&dst.values_biases, vecs[base + 5], s);
            }
            dst.offset = limit;
            dst.initialized = true;
        }
        cache.step = limit;
        // Materialize with a CHECKED eval: a corrupt chunk surfaces its MLX
        // error HERE (lazy Load reads data at eval), so the caller's catch
        // resets the cache and falls back to cold prefill instead of running
        // a forward over poisoned buffers.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values);
                if (quant.scheme != .off) {
                    _ = mlx.mlx_vector_array_append_value(vec, entry.keys_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.keys_biases);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.values_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.values_biases);
                }
            }
            try mlx.check(mlx.mlx_eval(vec));
        }
    }

    /// Zero rows after the restored ones, in the dtype and head shape of the stored chunks.
    fn appendRoom(vec: mlx.mlx_vector_array, room: c_int, s: mlx.mlx_stream) !void {
        var first = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(first);
        try mlx.check(mlx.mlx_vector_array_get(&first, vec, 0));
        const sh = mlx.getShape(first);
        var z = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ sh[0], sh[1], room, sh[3] }, 4, mlx.mlx_array_dtype(first), s));
        try mlx.check(mlx.mlx_vector_array_append_value(vec, z));
    }

    fn concatInto(dst: *mlx.mlx_array, vec: mlx.mlx_vector_array, s: mlx.mlx_stream) !void {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
        _ = mlx.mlx_array_free(dst.*);
        dst.* = out;
    }

    // ── Commit ──

    /// Persist a cache state under `tokens`. Called on the inference thread
    /// AFTER the response is finished (the write is bounded but synchronous).
    /// Takes snapshot-shaped parts (entries + step + config) so callers can
    /// flush either a live `KVCache` or a committed `KVCacheSnapshot` — the
    /// hot cache flushes the RAM entry it just committed, post-markFinished,
    /// so the client never waits on the SSD write. Skips ineligible states
    /// silently; never fails the request.
    pub fn appendCommit(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        s: mlx.mlx_stream,
    ) !PersistOutcome {
        return self.appendCommitWithSpec(kv_entries, step, config, tokens, has_tools, &.{}, null, s);
    }

    /// `appendCommit` with an explicit per-call flush bound (bytes): the loop stops after the
    /// first chunk that crosses it. The write-through hook passes one byte (one chunk per boundary).
    pub fn appendCommitBounded(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        s: mlx.mlx_stream,
        flush_bound: u64,
    ) !PersistOutcome {
        return self.appendCommitWithSpecBounded(kv_entries, step, config, tokens, has_tools, media, null, s, flush_bound);
    }

    /// `appendCommit` plus the v4 spec snapshot (the dflash assistant context)
    /// and the turn's media spans (v9 disk identity).
    /// Eligibility is enforced UPSTREAM, same as the RAM tier: the caller
    /// passes only what `commitWithState` was handed. The entry's record then
    /// carries (tokens, spans); supersede, extend, chunk share and the
    /// full-prefix promise all clamp at `mediaSharedBound`, so a later turn
    /// with a different pixel key can never inherit its rows off a list that
    /// disagrees below them.
    pub fn appendCommitWithSpec(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash_snap: ?SpecCommit,
        s: mlx.mlx_stream,
    ) !PersistOutcome {
        return self.appendCommitWithSpecBounded(kv_entries, step, config, tokens, has_tools, media, dflash_snap, s, self.max_flush_bytes);
    }

    fn appendCommitWithSpecBounded(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        media: []const MediaSpan,
        dflash_snap: ?SpecCommit,
        s: mlx.mlx_stream,
        flush_bound: u64,
    ) !PersistOutcome {
        // Every production caller swallows our error (the disk tier is best-effort end to
        // end), and two writers inside can raise: writeChunkFile,
        // appendSpecOnly. Drop the latch THIS call raised at the one funnel they all pass
        // through — otherwise the next decode tick's checkErrorDecode charges it to an
        // unrelated request. `writeSpecSidecar` swallows internally, so it keeps its own pair.
        const had_error = mlx.errorPending();
        errdefer mlx.dropLatchedErrorUnless(had_error);
        // On EOS-terminated turns the cache runs 1-2 positions AHEAD of the
        // committed token record (forwarded terminator tokens that never
        // land in `tokens`). Persist the prefix covered by the record —
        // positions beyond it are unusable for matching anyway.
        // The KV extent is `max(step, max initialized offset)`.
        // Anything the writer lost since the last commit is attributed before the index is read.
        _ = self.harvestWriteFailures();
        self.dropPoisonedEntries();

        const kv_target_u: usize = persistTargetLen(kv_entries, step, tokens.len);
        if (kv_target_u < MIN_PERSIST_TOKENS) return .skipped;
        const kv_target: u32 = @intCast(kv_target_u);
        // Every initialized layer must cover the persisted range with B == 1
        // — anything else (mid-spec-decode state, batched cache) is not a
        // persistable snapshot.
        for (kv_entries) |*entry| {
            if (!entry.initialized) continue;
            // The chunk files hold rows from position 0; a sliding ring dropped them.
            if (entry.keep > 0) {
                log.debug("  [disk-cache] skip: sliding ring layer\n", .{});
                return .skipped;
            }
            if (entry.offset < kv_target_u) {
                log.debug("  [disk-cache] skip: layer offset {d} < kv_len {d}\n", .{ entry.offset, kv_target_u });
                return .skipped;
            }
            const shape = mlx.getShape(entry.keys);
            if (shape.len != 4 or shape[0] != 1) {
                log.debug("  [disk-cache] skip: non-B1 cache shape\n", .{});
                return .skipped;
            }
        }

        // Superseded check: an existing entry that already covers `tokens`
        // (same key, tokens is a prefix of its tokens, kv already >= ours)
        // makes this commit a no-op - UNLESS this commit carries a spec payload the
        // entry lacks, which takes a dedicated spec-only append path (the KV chunks are
        // all present, so the extend machinery would pointlessly rewrite the tail chunk).
        var extend_idx: ?usize = null;
        var spec_only_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.poisoned) continue; // dead: never superseded, never extended
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            if (e.tokens.len >= tokens.len) {
                if (std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) {
                    // v9: supersede / incremental-resume requires the media
                    // lists to agree across the WHOLE committed record — the
                    // same u32 tokens can sit under different pixels, and rows
                    // already on disk under a divergent list are not this
                    // commit's state.
                    if (mediaSharedBound(e.media, media) < tokens.len) continue;
                    if (e.kv_len >= kv_target) {
                        if (!specWorkPending(e, dflash_snap)) {
                            // Superseded: the tier already holds this prefix in full.
                            e.last_used = self.bump();
                            return .persisted;
                        }
                        spec_only_idx = i;
                        break;
                    }
                    // Same token record, SHORTER persisted KV — a byte-capped
                    // incremental flush in progress. Resume into its dir.
                    extend_idx = i;
                }
            } else if (std.mem.eql(u32, e.tokens, tokens[0..e.tokens.len])) {
                // v9: extending an entry keeps its on-disk chunks, so its span
                // list must agree with this commit's across everything the
                // entry has already persisted a record for.
                if (mediaSharedBound(e.media, media) < e.tokens.len) continue;
                // This commit extends `e` — reuse its directory and chunks.
                extend_idx = i;
            }
        }
        // Re-derive the budget from free space before every store, and only a store: the
        // purgeable-space query is slow and the idle spill reaches this point once per idle entry.
        if (self.ssd_first) self.refreshDiskBudget();
        // The refresh gates THIS store, not merely the next one.
        if (self.store_declined) return .skipped;
        if (spec_only_idx) |i| return self.appendSpecOnly(i, dflash_snap, s);

        const sw = io_util.Stopwatch.init(self.io);

        const id: u64 = if (extend_idx) |i| self.entries.items[i].id else blk: {
            const nid = self.next_id;
            self.next_id += 1;
            break :blk nid;
        };
        const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, id });
        defer self.allocator.free(dir_rel);
        try std.Io.Dir.cwd().createDirPath(self.io, dir_rel);

        // Chunks [0, keep) are full chunks already on disk from the entry we
        // extend; everything from `keep` on (the old partial tail + the new
        // positions) is (re)written — up to the per-flush byte cap. Stopping
        // early lands on a full-chunk boundary; the entry then records the
        // shorter kv_len and the NEXT flush resumes from there.
        //
        // A fresh entry may inherit its leading whole chunks from a resident entry that shares a
        // prefix, by hard link: a persisted entry's tokens are `prompt ++ generated`, so the next
        // turn diverges inside the generated span and used to rewrite every chunk. Off on a synchronous RAM-first tier.
        const old_kv: u32 = if (extend_idx) |i| self.entries.items[i].kv_len else 0;
        const donor = if (extend_idx == null) self.chunkShareDonor(tokens, kv_target, has_tools, config, media) else null;
        var inherited: u32 = if (extend_idx) |i| self.entries.items[i].inherited_chunks else if (donor) |d| d.chunks else 0;
        var keep: u32 = if (extend_idx != null) old_kv / self.chunk_tokens else inherited;
        const n_chunks: u32 = @intCast((@as(u64, kv_target) + self.chunk_tokens - 1) / self.chunk_tokens);

        var chunk_sizes = std.ArrayList(u64).empty;
        errdefer chunk_sizes.deinit(self.allocator);
        if (extend_idx) |i| {
            const old_cb = self.entries.items[i].chunk_bytes;
            try chunk_sizes.appendSlice(self.allocator, old_cb[0..@min(keep, old_cb.len)]);
        } else if (donor) |d| {
            // Only the donor's landed chunks are linked (a contiguous prefix); may be zero.
            const linked = self.linkInheritedChunks(d, id, &chunk_sizes) catch 0;
            if (linked == 0) chunk_sizes.clearRetainingCapacity();
            inherited = linked;
            keep = linked;
        }
        // A real check: a rewrite must never land on a link (the sync arm truncates in place).
        if (keep < inherited) {
            var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{}) catch null;
            defer if (root_dir) |*rd| rd.close(self.io);
            if (root_dir) |rd| self.unlinkChunkRange(rd, id, keep, inherited);
            log.warn("  [disk-cache] chunk share: e{d} would rewrite an inherited chunk (keep {d} < inherited {d}) — writing from {d} instead\n", .{ id, keep, inherited, keep });
            inherited = keep;
            if (chunk_sizes.items.len > keep) chunk_sizes.shrinkRetainingCapacity(keep);
        }

        var written_bytes: u64 = 0;

        // Chunks [0, keep) are full chunks already on disk (see above); the
        // rewrite runs from `keep` up to the flush bound.
        var chunk_i: u32 = keep;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            if (written_bytes >= flush_bound and chunk_i > keep) break;
            const c0: u32 = chunk_i * self.chunk_tokens;
            const c1: u32 = @intCast(@min(@as(u64, c0) + self.chunk_tokens, kv_target));
            const csize = try self.writeChunkFile(kv_entries, config, dir_rel, chunk_i, c0, c1, s);
            written_bytes += csize;
            try chunk_sizes.append(self.allocator, csize);
        }
        const chunks_done: u32 = chunk_i;
        const chunk_complete = chunks_done == n_chunks;
        const kv_len: u32 = if (chunk_complete) kv_target else chunks_done * self.chunk_tokens;
        if (kv_len <= old_kv) {
            // Cap so tight nothing new landed - nothing to commit.
            chunk_sizes.deinit(self.allocator);
            return if (chunk_complete) .persisted else .partial;
        }

        const complete = chunk_complete;

        // v4 spec snapshots — one sidecar file, REPLACED wholesale by every
        // commit (a commit with no payload deletes a stale one, the RAM
        // tier's supersede rule). Best-effort DRAFT-side state: a failed
        // write costs the entry its spec, never the entry.
        const spec_res: SpecSidecarResult = self.writeSpecSidecar(dir_rel, dflash_snap, s) catch |err| blk: {
            log.warn("  [disk-cache] spec persist failed: {s} — entry keeps no spec\n", .{@errorName(err)});
            break :blk .{};
        };

        // Token record — the LONGER of the existing record and this commit's
        // tokens (a resumed incremental flush must not shrink the record its
        // earlier chunks were committed against). Rewritten only on growth;
        // tens of KB at most.
        const record: []const u32 = if (extend_idx) |i| blk: {
            const et = self.entries.items[i].tokens;
            break :blk if (et.len >= tokens.len) et else tokens;
        } else tokens;
        if (extend_idx == null or record.ptr == tokens.ptr) {
            const tpath = try std.fmt.allocPrint(self.allocator, "{s}/tokens.bin", .{dir_rel});
            defer self.allocator.free(tpath);
            const f = try std.Io.Dir.createFileAbsolute(self.io, tpath, .{});
            defer f.close(self.io);
            var wb: [8192]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.writeSliceEndian(u32, record, .little);
            try fw.interface.flush();
        }

        // `bytes` is what this entry created on disk: inherited (linked) chunks bill 0.
        const non_chunk: u64 = @as(u64, record.len) * 4 + spec_res.bytes;
        var bytes: u64 = non_chunk;
        for (chunk_sizes.items[@min(inherited, chunk_sizes.items.len)..]) |b| bytes += b;

        var new_entry: IndexEntry = .{
            .id = id,
            .tokens = try self.allocator.dupe(u32, record),
            .kv_len = kv_len,
            .has_tools = has_tools,
            .quant = config,
            .bytes = bytes,
            .chunk_bytes = try chunk_sizes.toOwnedSlice(self.allocator),
            .inherited_chunks = inherited,
            .spec_bytes = spec_res.bytes,
            .spec_dflash = spec_res.dflash,
            .media = try self.allocator.dupe(MediaSpan, media),
            .last_used = self.bump(),
        };
        errdefer {
            self.allocator.free(new_entry.tokens);
            self.allocator.free(new_entry.chunk_bytes);
            self.allocator.free(new_entry.media);
        }

        if (extend_idx) |i| {
            const e = &self.entries.items[i];
            // An extension's delta is file-based; `e.bytes` after a `scan` may include chunks
            // the manifest lists as inherited (their donor died first), so it is carried forward.
            var delta: i64 = @as(i64, @intCast(non_chunk)) - @as(i64, @intCast(nonChunkBytes(e)));
            for (new_entry.chunk_bytes[@min(keep, new_entry.chunk_bytes.len)..]) |b| delta += @as(i64, @intCast(b));
            if (e.chunk_bytes.len > keep) delta -= @as(i64, @intCast(e.chunk_bytes[keep]));
            new_entry.bytes = clampAdd(e.bytes, delta);
            // meta.json is the commit point: written last, atomically.
            try self.writeMeta(new_entry);
            self.total_bytes = clampAdd(self.total_bytes, delta);
            self.allocator.free(e.tokens);
            self.allocator.free(e.chunk_bytes);
            self.allocator.free(e.media);
            e.* = new_entry;
        } else {
            // meta.json is the commit point: written last, atomically.
            try self.writeMeta(new_entry);
            try self.entries.append(self.allocator, new_entry);
            self.total_bytes += new_entry.bytes;
        }
        self.bytes_persisted += written_bytes;
        self.gcToBudget();

        const wrote_mb = @as(f64, @floatFromInt(written_bytes)) / (1024.0 * 1024.0);
        const ms: u64 = sw.read() / std.time.ns_per_ms;
        log.info("  [disk-cache] persisted {d}/{d} tokens (+{d} chunks, {d:.1} MB, {d}ms); resident={d:.1} MB ({d} entries)\n", .{
            kv_len,                                                        kv_target,              chunks_done - keep, wrote_mb, ms,
            @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0), self.entries.items.len,
        });
        // The one completion marker: every chunk is staged. A
        // bounded flush leaves it out and `disk_dirty` set. A write that already failed for this
        // entry makes `.persisted` a promise nothing can keep.
        var whole = complete;
        if (whole) {
            _ = self.harvestWriteFailures();
            if (self.entryPoisoned(id)) whole = false;
        }
        if (whole) {
            log.info("  [disk-cache] e{d} complete on disk: {d} tokens, {d} chunks\n", .{ id, kv_len, new_entry.chunk_bytes.len });
        }
        return if (whole) .persisted else .partial;
    }

    /// Does the index agree that this tier holds a complete, restorable copy of `tokens`? Two
    /// bars before a RAM entry may be discarded: the token record covers `tokens` at the same
    /// key with `kv_len` reaching the persist target, and `chunk_bytes` has one non-zero entry
    /// per implied chunk.
    pub fn holdsFullPrefix(
        self: *const DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        tokens: []const u32,
        has_tools: bool,
        config: kv_quant.KVQuantConfig,
        media: []const MediaSpan,
    ) bool {
        return self.fullPrefixEntryId(kv_entries, step, tokens, has_tools, config, media) != null;
    }

    /// `holdsFullPrefix`, returning the entry's id so the caller can ask
    /// `entryWritesPending`. The record must carry a span list agreeing
    /// across `tokens` (`media` empty = a text turn).
    pub fn fullPrefixEntryId(
        self: *const DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        tokens: []const u32,
        has_tools: bool,
        config: kv_quant.KVQuantConfig,
        media: []const MediaSpan,
    ) ?u64 {
        const target = persistTargetLen(kv_entries, step, tokens.len);
        if (target == 0) return null;
        for (self.entries.items) |*e| {
            if (e.poisoned) continue;
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            if (e.tokens.len < tokens.len) continue;
            if (!std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) continue;
            if (mediaSharedBound(e.media, media) < tokens.len) continue;
            if (e.kv_len < target) continue;
            const want: usize = (@as(usize, e.kv_len) + self.chunk_tokens - 1) / self.chunk_tokens;
            if (e.chunk_bytes.len < want) continue;
            var whole = true;
            for (e.chunk_bytes[0..want]) |b| {
                if (b == 0) whole = false;
            }
            if (whole) return e.id;
        }
        return null;
    }

    /// Sidecar-only append: KV chunks are already fully on disk (superseded on KV) but this
    /// commit carries a spec snapshot the entry lacks. Writes it into the existing dir and
    /// rewrites meta. Never touches the KV chunks or the token record.
    fn appendSpecOnly(
        self: *DiskTier,
        idx: usize,
        dflash_snap: ?SpecCommit,
        s: mlx.mlx_stream,
    ) !PersistOutcome {
        const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, self.entries.items[idx].id });
        defer self.allocator.free(dir_rel);
        const e = &self.entries.items[idx];
        // Chunks and the token record are untouched here, so the bill moves by the non-chunk files.
        const old_non_chunk = nonChunkBytes(e);
        if (specWorkPending(e, dflash_snap)) {
            const spec_res: SpecSidecarResult = self.writeSpecSidecar(dir_rel, dflash_snap, s) catch |err| blk: {
                log.warn("  [disk-cache] spec persist failed: {s} — entry keeps its old spec\n", .{@errorName(err)});
                break :blk .{ .bytes = e.spec_bytes, .dflash = e.spec_dflash };
            };
            e.spec_bytes = spec_res.bytes;
            e.spec_dflash = spec_res.dflash;
        }

        const delta: i64 = @as(i64, @intCast(nonChunkBytes(e))) - @as(i64, @intCast(old_non_chunk));
        e.bytes = clampAdd(e.bytes, delta);
        self.total_bytes = clampAdd(self.total_bytes, delta);
        e.last_used = self.bump();
        try self.writeMeta(e.*);
        self.gcToBudget();
        return .persisted;
    }

    // ── Spec-snapshot persistence (v4: the dflash context) ──

    const SpecSidecarResult = struct {
        bytes: u64 = 0,
        dflash: ?SpecMeta = null,
    };

    /// Does this commit carry a spec payload the entry lacks? Mirrors
    /// The superseded no-op decision's spec check. A present
    /// spec is never "updated" at the same tokens — same tokens, same
    /// committed state.
    fn specWorkPending(e: *const IndexEntry, dflash: ?SpecCommit) bool {
        return dflash != null and e.spec_dflash == null;
    }

    /// Write (or delete) the entry's ONE spec sidecar from this commit's
    /// snapshots. Tensors are sliced to `step` positions (the snapshot buffer
    /// can hold a stale draft tail past it) and keyed `d{layer}.*`
    /// with the trunk chunks' kind suffixes.
    fn writeSpecSidecar(self: *DiskTier, dir_abs: []const u8, dflash: ?SpecCommit, s: mlx.mlx_stream) !SpecSidecarResult {
        // Best-effort by contract: the callers log the failure and keep going. The
        // latch a raise plants is process-wide, so a write we already reported must
        // drop what it set — or the next decode tick fails an unrelated request.
        const had_error = mlx.errorPending();
        errdefer mlx.dropLatchedErrorUnless(had_error);
        const path = try std.fmt.allocPrint(self.allocator, "{s}/spec.safetensors\x00", .{dir_abs});
        defer self.allocator.free(path);
        if (dflash == null) {
            std.Io.Dir.deleteFileAbsolute(self.io, path[0 .. path.len - 1]) catch {};
            return .{};
        }
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        var res: SpecSidecarResult = .{};
        if (dflash) |dc| res.dflash = try self.insertSpecTensors(tensor_map, "d", dc, s);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
        res.bytes = fileSize(self.io, path[0 .. path.len - 1]) orelse 0;
        return res;
    }

    fn insertSpecTensors(self: *DiskTier, map: mlx.mlx_map_string_to_array, prefix: []const u8, sc: SpecCommit, s: mlx.mlx_stream) !SpecMeta {
        if (sc.step == 0) return error.DiskCacheEmptyEntry;
        const limit: u32 = @intCast(sc.step);
        const affine = sc.config.scheme != .off;
        for (sc.entries, 0..) |*entry, li| {
            if (!entry.initialized) continue;
            try self.insertSpecSlice(map, prefix, li, "k", entry.keys, limit, s);
            try self.insertSpecSlice(map, prefix, li, "v", entry.values, limit, s);
            if (affine) {
                try self.insertSpecSlice(map, prefix, li, "ks", entry.keys_scales, limit, s);
                try self.insertSpecSlice(map, prefix, li, "kb", entry.keys_biases, limit, s);
                try self.insertSpecSlice(map, prefix, li, "vs", entry.values_scales, limit, s);
                try self.insertSpecSlice(map, prefix, li, "vb", entry.values_biases, limit, s);
            }
        }
        return .{
            .base = sc.base_pos,
            .step = limit,
            .layers = @intCast(sc.entries.len),
            .quant = sc.config,
        };
    }

    fn insertSpecArray(self: *DiskTier, map: mlx.mlx_map_string_to_array, prefix: []const u8, kind: []const u8, arr: mlx.mlx_array) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}{s}\x00", .{ prefix, kind });
        defer self.allocator.free(key);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, @ptrCast(key.ptr), arr));
    }

    fn insertSpecSlice(self: *DiskTier, map: mlx.mlx_map_string_to_array, prefix: []const u8, layer: usize, kind: []const u8, buf: mlx.mlx_array, limit: u32, s: mlx.mlx_stream) !void {
        const shape = mlx.getShape(buf);
        if (shape.len != 4) return error.DiskCacheBadShape;
        if (shape[2] < limit) return error.DiskCacheBadShape;
        var sliced = mlx.mlx_array_new();
        const st = [_]c_int{ 0, 0, 0, 0 };
        const sp = [_]c_int{ shape[0], shape[1], @intCast(limit), shape[3] };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
        defer _ = mlx.mlx_array_free(sliced);
        const key = try std.fmt.allocPrint(self.allocator, "{s}{d}.{s}\x00", .{ prefix, layer, kind });
        defer self.allocator.free(key);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, @ptrCast(key.ptr), sliced));
    }

    /// Load one persisted spec snapshot as a transient `KVCacheSnapshot` the
    /// caller restores from and then deinits. Best-effort in every direction:
    /// null when the entry has none, the recorded geometry doesn't fit the
    /// target (layer count / quant config — `KVCache.restore` asserts equal
    /// lengths, so the check lives here), or the file is unreadable. The
    /// caller then starts blind, never wrong.
    pub fn loadSpecSnap(
        self: *DiskTier,
        idx: usize,
        which: SpecKind,
        expected_layers: usize,
        target_config: kv_quant.KVQuantConfig,
    ) ?struct {
        snap: transformer_mod.KVCacheSnapshot,
        base: usize,
    } {
        const e = &self.entries.items[idx];
        const meta = (switch (which) {
            .dflash => e.spec_dflash,
        }) orelse return null;
        if (meta.layers != expected_layers) return null;
        if (!std.meta.eql(meta.quant, target_config)) {
            log.info("  [disk-cache] spec sidecar declined (quant mismatch)\n", .{});
            return null;
        }

        const load_s = mlx.weightLoadStream();
        defer _ = mlx.mlx_stream_free(load_s);
        const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/spec.safetensors\x00", .{ self.root, e.id }) catch return null;
        defer self.allocator.free(path);
        var tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        var meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), load_s)) catch return null;

        const prefix: []const u8 = switch (which) {
            .dflash => "d",
        };
        const kinds: []const []const u8 = if (meta.quant.scheme == .off)
            &.{ "k", "v" }
        else
            &.{ "k", "v", "ks", "kb", "vs", "vb" };

        const entries = self.allocator.alloc(transformer_mod.KVCacheEntry, expected_layers) catch return null;
        for (entries) |*en| en.* = transformer_mod.newEmptyKVEntry();
        var snap: transformer_mod.KVCacheSnapshot = .{
            .entries = entries,
            .step = meta.step,
            .allocator = self.allocator,
            .config = meta.quant,
        };
        var ok = true;
        outer: for (entries, 0..) |*en, li| {
            for (kinds, 0..) |kind, ki| {
                const key = std.fmt.allocPrint(self.allocator, "{s}{d}.{s}\x00", .{ prefix, li, kind }) catch {
                    ok = false;
                    break :outer;
                };
                defer self.allocator.free(key);
                var arr = mlx.mlx_array_new();
                if (mlx.mlx_map_string_to_array_get(&arr, tensor_map, @ptrCast(key.ptr)) != 0) {
                    _ = mlx.mlx_array_free(arr);
                    if (ki == 0) continue :outer; // layer absent — stays uninitialized
                    ok = false; // partial layer = corrupt
                    break :outer;
                }
                // transfer the +1 handed by _get, replacing the empty handle
                switch (ki) {
                    0 => {
                        _ = mlx.mlx_array_free(en.keys);
                        en.keys = arr;
                    },
                    1 => {
                        _ = mlx.mlx_array_free(en.values);
                        en.values = arr;
                    },
                    2 => {
                        _ = mlx.mlx_array_free(en.keys_scales);
                        en.keys_scales = arr;
                    },
                    3 => {
                        _ = mlx.mlx_array_free(en.keys_biases);
                        en.keys_biases = arr;
                    },
                    4 => {
                        _ = mlx.mlx_array_free(en.values_scales);
                        en.values_scales = arr;
                    },
                    5 => {
                        _ = mlx.mlx_array_free(en.values_biases);
                        en.values_biases = arr;
                    },
                    else => unreachable,
                }
            }
            en.offset = meta.step;
            en.initialized = true;
        }
        if (!ok) {
            snap.deinit();
            return null;
        }
        // Checked eval: a corrupt file surfaces its MLX error HERE (lazy Load
        // reads data at eval), not mid-forward after the restore.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var count: usize = 0;
            for (entries) |*en| {
                if (!en.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, en.keys);
                _ = mlx.mlx_vector_array_append_value(vec, en.values);
                if (meta.quant.scheme != .off) {
                    _ = mlx.mlx_vector_array_append_value(vec, en.keys_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, en.keys_biases);
                    _ = mlx.mlx_vector_array_append_value(vec, en.values_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, en.values_biases);
                }
                count += 1;
            }
            if (count > 0) {
                mlx.check(mlx.mlx_eval(vec)) catch {
                    snap.deinit();
                    return null;
                };
            }
        }
        return .{ .snap = snap, .base = meta.base };
    }

    /// One optional non-layer tensor out of the loaded sidecar map (+1 handle), or null.
    fn getSpecArray(map: mlx.mlx_map_string_to_array, allocator: std.mem.Allocator, prefix: []const u8, kind: []const u8) ?mlx.mlx_array {
        const key = std.fmt.allocPrint(allocator, "{s}{s}\x00", .{ prefix, kind }) catch return null;
        defer allocator.free(key);
        var arr = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_get(&arr, map, @ptrCast(key.ptr)) != 0) {
            _ = mlx.mlx_array_free(arr);
            return null;
        }
        return arr;
    }

    /// One staged tensor on the way to a safetensors file.
    const NamedTensor = struct { key: []u8, arr: mlx.mlx_array };

    fn freeNamed(self: *DiskTier, list: *std.ArrayList(NamedTensor)) void {
        for (list.items) |*t| {
            self.allocator.free(t.key);
            _ = mlx.mlx_array_free(t.arr);
        }
        list.deinit(self.allocator);
    }

    /// Write (or, under SSD-first, stage) one KV chunk. Returns the file's byte size.
    fn writeChunkFile(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        config: kv_quant.KVQuantConfig,
        dir_abs: []const u8,
        chunk_idx: u32,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !u64 {
        var list = std.ArrayList(NamedTensor).empty;
        defer self.freeNamed(&list);

        const affine = config.scheme != .off;
        for (kv_entries, 0..) |*entry, li| {
            if (!entry.initialized) continue;
            try self.appendSlice(&list, li, "k", entry.keys, c0, c1, s);
            try self.appendSlice(&list, li, "v", entry.values, c0, c1, s);
            if (affine) {
                try self.appendSlice(&list, li, "ks", entry.keys_scales, c0, c1, s);
                try self.appendSlice(&list, li, "kb", entry.keys_biases, c0, c1, s);
                try self.appendSlice(&list, li, "vs", entry.values_scales, c0, c1, s);
                try self.appendSlice(&list, li, "vb", entry.values_biases, c0, c1, s);
            }
        }

        const path = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors", .{ dir_abs, chunk_idx });
        if (self.writer) |w| {
            // The readback stays here (mlx arrays are inference-thread-owned); only bytes cross.
            const bytes = self.serializeSafetensors(list.items, no_meta, s) catch |err| {
                self.allocator.free(path);
                return err;
            };
            const n = bytes.len;
            w.submit(path, bytes); // takes both buffers
            return n;
        }
        defer self.allocator.free(path);
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        for (list.items) |*t| {
            const key_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{t.key});
            defer self.allocator.free(key_z);
            try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, @ptrCast(key_z.ptr), t.arr));
        }
        const path_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{path});
        defer self.allocator.free(path_z);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path_z.ptr), tensor_map, meta_map));
        return fileSize(self.io, path) orelse 0;
    }

    fn appendSlice(
        self: *DiskTier,
        list: *std.ArrayList(NamedTensor),
        layer: usize,
        kind: []const u8,
        buf: mlx.mlx_array,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !void {
        const shape = mlx.getShape(buf);
        if (shape.len != 4) return error.DiskCacheBadShape;
        var sliced = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(sliced);
        const st = [_]c_int{ 0, 0, @intCast(c0), 0 };
        const sp = [_]c_int{ shape[0], shape[1], @intCast(c1), shape[3] };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
        const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}", .{ layer, kind });
        errdefer self.allocator.free(key);
        try list.append(self.allocator, .{ .key = key, .arr = sliced });
    }

    /// Make every tensor contiguous in place and materialize the list with exactly one batched
    /// `mlx_eval`, as `mlx::core::save_safetensors` does. A per-tensor eval is a full GPU sync each.
    fn materializeContiguous(tensors: []NamedTensor, s: mlx.mlx_stream) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (tensors) |*t| {
            var cont = mlx.mlx_array_new();
            {
                // `cont` is owned locally only until the transfer below; nothing fallible may sit
                // between this scope's close and `t.arr = cont`, or the list's `freeNamed` double-frees.
                errdefer _ = mlx.mlx_array_free(cont);
                try mlx.check(mlx.mlx_contiguous(&cont, t.arr, false, s));
            }
            _ = mlx.mlx_array_free(t.arr);
            t.arr = cont;
            try mlx.check(mlx.mlx_vector_array_append_value(vec, t.arr));
        }
        _ = serialize_eval_count.fetchAdd(1, .monotonic);
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// Serialize a tensor list into one safetensors byte image, exactly as `mlx::core::save_safetensors` does.
    fn serializeSafetensors(self: *DiskTier, tensors: []NamedTensor, meta: []const MetaPair, s: mlx.mlx_stream) ![]u8 {
        try materializeContiguous(tensors, s);
        return self.encodeSafetensors(tensors, meta);
    }

    /// One `__metadata__` entry. The encoder refuses anything needing JSON escaping.
    pub const MetaPair = struct { key: []const u8, value: []const u8 };

    const no_meta: []const MetaPair = &[_]MetaPair{};

    /// Is `v` safe inside a JSON string with no escaping?
    fn plainJsonAscii(v: []const u8) bool {
        for (v) |c| {
            if (c < 0x20 or c > 0x7e or c == '"' or c == '\\') return false;
        }
        return true;
    }

    /// Header + payload encode over an already-materialized tensor list. Evaluates nothing.
    fn encodeSafetensors(self: *DiskTier, tensors: []const NamedTensor, meta: []const MetaPair) ![]u8 {
        var data_len: u64 = 0;
        for (tensors) |*t| data_len += nbytesOf(t.arr);

        var header = std.ArrayList(u8).empty;
        defer header.deinit(self.allocator);
        const hw = &header;
        try hw.appendSlice(self.allocator, "{\"__metadata__\":{");
        for (meta, 0..) |m, mi| {
            if (!plainJsonAscii(m.key) or !plainJsonAscii(m.value)) return error.DiskCacheBadMetadata;
            if (mi > 0) try hw.appendSlice(self.allocator, ",");
            try hw.print(self.allocator, "\"{s}\":\"{s}\"", .{ m.key, m.value });
        }
        try hw.appendSlice(self.allocator, "}");
        var off: u64 = 0;
        for (tensors) |*t| {
            const nb = nbytesOf(t.arr);
            const dname = safetensorsDtypeName(mlx.mlx_array_dtype(t.arr)) orelse return error.DiskCacheBadDtype;
            try hw.print(self.allocator, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.key, dname });
            for (mlx.getShape(t.arr), 0..) |d, i| {
                if (i > 0) try hw.appendSlice(self.allocator, ",");
                try hw.print(self.allocator, "{d}", .{d});
            }
            try hw.print(self.allocator, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + nb });
            off += nb;
        }
        try hw.appendSlice(self.allocator, "}");

        const total = 8 + header.items.len + data_len;
        const out = try self.allocator.alloc(u8, total);
        errdefer self.allocator.free(out);
        std.mem.writeInt(u64, out[0..8], @intCast(header.items.len), .little);
        @memcpy(out[8 .. 8 + header.items.len], header.items);
        var cursor: usize = 8 + header.items.len;
        for (tensors) |*t| {
            const nb: usize = @intCast(nbytesOf(t.arr));
            if (nb == 0) continue;
            const src = rawBytes(t.arr) orelse return error.DiskCacheUnreadable;
            @memcpy(out[cursor .. cursor + nb], src[0..nb]);
            cursor += nb;
        }
        return out;
    }

    // ── Invalidation (mirrors the RAM cache API) ──

    pub fn invalidateAll(self: *DiskTier) void {
        for (self.entries.items) |*e| {
            self.deleteEntryDir(e.id);
            self.freeIndexEntryOwned(e);
        }
        self.entries.clearRetainingCapacity();
        self.total_bytes = 0;
    }

    pub fn invalidateNewest(self: *DiskTier) void {
        if (self.entries.items.len == 0) return;
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        self.removeAt(newest_idx);
    }

    // ── Internals ──

    fn bump(self: *DiskTier) u64 {
        self.counter += 1;
        return self.counter;
    }

    fn removeAt(self: *DiskTier, idx: usize) void {
        var e = self.entries.swapRemove(idx);
        self.total_bytes -|= self.bytesFreedByRemoving(&e);
        self.deleteEntryDir(e.id);
        self.freeIndexEntryOwned(&e);
    }

    /// Bytes deleting `e`'s directory returns to the volume: its non-chunk files plus every
    /// chunk file nobody else links (`nlink == 1`). The filesystem is the refcount.
    fn bytesFreedByRemoving(self: *DiskTier, e: *const IndexEntry) u64 {
        const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, e.id }) catch return e.bytes;
        defer self.allocator.free(dir_abs);
        if (self.writer) |w| w.fence(dir_abs);
        var freed: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
        for (e.chunk_bytes, 0..) |cb, i| {
            const cp = std.fmt.allocPrint(self.allocator, "{s}c{d:0>6}.safetensors", .{ dir_abs, i }) catch {
                freed += cb;
                continue;
            };
            defer self.allocator.free(cp);
            if (statFile(self.io, cp)) |st| {
                if (st.nlink <= 1) freed += st.size;
            } else if (i >= e.inherited_chunks) {
                freed += cb;
            }
        }
        return freed;
    }

    /// The resident entry whose leading chunk files a new entry for `tokens` may hard-link:
    /// same tool flag and kv-quant config, most whole chunks below the common prefix. Null on
    /// a synchronous RAM-first tier or under `MLX_SERVE_SSD_CHUNK_SHARE=0`.
    fn chunkShareDonor(self: *DiskTier, tokens: []const u32, kv_target: u32, has_tools: bool, config: kv_quant.KVQuantConfig, media: []const MediaSpan) ?ChunkDonor {
        if (!(self.ssd_first or self.writer != null) or !chunkShareEnabled()) return null;
        // `media` (v9): chunks may only be linked below the point where the two
        // span lists can diverge — above it the donor's rows describe different
        // pixels, not this commit's state.
        var best: ?ChunkDonor = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.poisoned) continue; // never inherit from a dead entry
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            const shared: u64 = @min(@min(@min(@as(u64, commonPrefixLen(e.tokens, tokens)), @as(u64, kv_target)),
                @as(u64, e.kv_len)), @as(u64, mediaSharedBound(e.media, media)));
            const whole: u32 = @intCast(shared / self.chunk_tokens);
            const usable: u32 = @min(whole, @as(u32, @intCast(e.chunk_bytes.len)));
            if (usable == 0) continue;
            if (best == null or usable > best.?.chunks) best = .{ .idx = i, .id = e.id, .chunks = usable };
        }
        return best;
    }

    const ChunkDonor = struct { idx: usize, id: u64, chunks: u32 };

    /// Hard-link the donor's leading landed chunk files into `e<id>/`; returns how many. Stops
    /// at the first chunk not landed and never touches the donor's queue (`Writer.fence` would
    /// discard it). A link failure unwinds and the caller writes every chunk itself.
    fn linkInheritedChunks(self: *DiskTier, d: ChunkDonor, id: u64, chunk_sizes: *std.ArrayList(u64)) !u32 {
        var root_dir = try std.Io.Dir.openDirAbsolute(self.io, self.root, .{});
        defer root_dir.close(self.io);
        const donor_cb = self.entries.items[d.idx].chunk_bytes;
        var linked: u32 = 0;
        errdefer self.unlinkChunkRange(root_dir, id, 0, linked);
        var i: u32 = 0;
        while (i < d.chunks) : (i += 1) {
            const old_abs = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, d.id, i });
            defer self.allocator.free(old_abs);
            if (!self.chunkLanded(old_abs, donor_cb[i])) break;
            const old_sub = try std.fmt.allocPrint(self.allocator, "e{d}/c{d:0>6}.safetensors", .{ d.id, i });
            defer self.allocator.free(old_sub);
            const new_sub = try std.fmt.allocPrint(self.allocator, "e{d}/c{d:0>6}.safetensors", .{ id, i });
            defer self.allocator.free(new_sub);
            std.Io.Dir.hardLink(root_dir, old_sub, root_dir, new_sub, self.io, .{}) catch |err| {
                log.warn("  [disk-cache] chunk share: link e{d}/c{d} -> e{d} failed: {s} — writing the chunks instead\n", .{ d.id, i, id, @errorName(err) });
                return error.ChunkShareLinkFailed;
            };
            linked += 1;
            try chunk_sizes.append(self.allocator, donor_cb[i]);
        }
        if (linked == 0) return 0;
        var mb: f64 = 0;
        for (donor_cb[0..linked]) |b| mb += @as(f64, @floatFromInt(b));
        mb /= 1024.0 * 1024.0;
        log.info("  [disk-cache] chunk share: e{d} inherits {d} chunks ({d:.1} MB) from e{d} by hard link\n", .{ id, linked, mb, d.id });
        return linked;
    }

    /// Has a chunk file landed: final name, recorded size, and no write to it queued or in flight?
    fn chunkLanded(self: *DiskTier, abs_path: []const u8, want_size: u64) bool {
        const st = statFile(self.io, abs_path) orelse return false;
        if (st.size != want_size) return false;
        if (self.writer) |w| {
            if (w.isPending(abs_path)) return false;
        }
        return true;
    }

    fn unlinkChunkRange(self: *DiskTier, root_dir: std.Io.Dir, id: u64, from: u32, to: u32) void {
        var i: u32 = from;
        while (i < to) : (i += 1) {
            const sub = std.fmt.allocPrint(self.allocator, "e{d}/c{d:0>6}.safetensors", .{ id, i }) catch continue;
            defer self.allocator.free(sub);
            root_dir.deleteFile(self.io, sub) catch {};
        }
    }

    fn deleteEntryDir(self: *DiskTier, id: u64) void {
        // Epoch fence, the one removal site: staged bytes for this directory are discarded.
        const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch null;
        defer if (dir_abs) |da| self.allocator.free(da);
        if (self.writer) |w| w.fence(dir_abs);
        const rel = std.fmt.allocPrint(self.allocator, "e{d}", .{id}) catch return;
        defer self.allocator.free(rel);
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
        defer root_dir.close(self.io);
        root_dir.deleteTree(self.io, rel) catch {};
    }

    fn gcToBudget(self: *DiskTier) void {
        if (self.max_bytes == 0) return;
        while (self.total_bytes > self.max_bytes and self.entries.items.len > 1) {
            var lru_idx: usize = 0;
            var lru_used: u64 = std.math.maxInt(u64);
            for (self.entries.items, 0..) |*e, i| {
                if (e.last_used < lru_used) {
                    lru_used = e.last_used;
                    lru_idx = i;
                }
            }
            const mb = @as(f64, @floatFromInt(self.entries.items[lru_idx].bytes)) / (1024.0 * 1024.0);
            log.info("  [disk-cache] evicted LRU entry (byte budget; {d:.1} MB)\n", .{mb});
            self.removeAt(lru_idx);
        }
    }

    fn writeMeta(self: *DiskTier, e: IndexEntry) !void {
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(self.allocator);
        try self.renderMeta(&buf, e);

        const final_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, e.id });
        // One owner per branch: the staged branch's cleanup is an `errdefer` scoped to it (a
        // `defer` does not cancel an enclosing `errdefer`, and that was a double free on ENOSPC).
        if (self.writer) |w| {
            // The index rides the same FIFO queue as the chunks, so it is the last file to land.
            errdefer self.allocator.free(final_path);
            const bytes = try self.allocator.dupe(u8, buf.items);
            w.submit(final_path, bytes);
            return;
        }
        defer self.allocator.free(final_path);
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json.tmp", .{ self.root, e.id });
        defer self.allocator.free(tmp_path);
        {
            const f = try std.Io.Dir.createFileAbsolute(self.io, tmp_path, .{});
            defer f.close(self.io);
            var wb: [1024]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.writeAll(buf.items);
            try fw.interface.flush();
        }
        try std.Io.Dir.renameAbsolute(tmp_path, final_path, self.io);
    }

    /// The lowest manifest version that describes this entry. The version is a compatibility
    /// claim: an older reader accepts only 2..4, so stamping v6 unconditionally made a binary
    /// downgrade discard the whole tier. v6 = inherited chunks.
    fn metaVersionFor(e: IndexEntry) u8 {
        // v9 = media-keyed identity: an older reader must NOT load these into
        // token-only matching (it would restore past a placeholder under a
        // possibly different image), so a media entry claims a version the old
        // reader gate rejects.
        if (e.media.len > 0) return 9;
        if (e.inherited_chunks > 0) return 6;
        return 4;
    }

    /// The meta.json body; one renderer for the synchronous and the staged path.
    fn renderMeta(self: *DiskTier, out: *std.ArrayList(u8), e: IndexEntry) !void {
        const a = self.allocator;
        try out.print(
            a,
            "{{\"v\":{d},\"kv_len\":{d},\"tokens\":{d},\"has_tools\":{},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d},\"chunk_tokens\":{d},\"inherited_chunks\":{d},\"bytes\":{d},\"chunk_bytes\":[",
            .{
                metaVersionFor(e),
                e.kv_len,
                e.tokens.len,
                e.has_tools,
                @tagName(e.quant.scheme),
                e.quant.bits,
                e.quant.group_size,
                self.chunk_tokens,
                e.inherited_chunks,
                e.bytes,
            },
        );
        for (e.chunk_bytes, 0..) |cb, i| {
            if (i > 0) try out.appendSlice(a, ",");
            try out.print(a, "{d}", .{cb});
        }
        // v3: the state checkpoint list, always empty now (a manifest keeps the field).
        try out.appendSlice(a, "],\"ssm\":[]");
        // v4: the dflash context snapshot; a size mismatch drops only the spec.
        if (e.spec_bytes > 0 and e.spec_dflash != null) {
            try out.print(a, ",\"spec\":{{\"bytes\":{d}", .{e.spec_bytes});
            if (e.spec_dflash) |sm| try writeSpecMetaJson(a, out, "dflash", sm);
            try out.appendSlice(a, "}");
        }
        // v9: media spans as [[start,key],...] — the entry's media identity.
        if (e.media.len > 0) {
            try out.appendSlice(a, ",\"media\":[");
            for (e.media, 0..) |sp, i| {
                if (i > 0) try out.appendSlice(a, ",");
                // Signed decimal: the JSON reader yields a SIGNED integer, and a
                // pixel hash is a full u64 (high bit set about half the time). An
                // unsigned print makes those metas unparseable — the entry would
                // be dropped at scan. `loadEntry` bit-casts it back, so the u64
                // bits round-trip exactly.
                try out.print(a, "[{d},{d}]", .{ sp.start, @as(i64, @bitCast(sp.key)) });
            }
            try out.appendSlice(a, "]");
        }
        try out.appendSlice(a, "}");
    }

    fn scan(self: *DiskTier) !void {
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
        defer root_dir.close(self.io);

        // Collected (entry, mtime) pairs; sorted by mtime → LRU order.
        const Pending = struct { e: IndexEntry, mtime: i128 };
        var pending = std.ArrayList(Pending).empty;
        defer pending.deinit(self.allocator);

        var it = root_dir.iterate();
        while (it.next(self.io) catch null) |dent| {
            if (dent.kind != .directory) continue;
            if (dent.name.len < 2 or dent.name[0] != 'e') continue;
            const id = std.fmt.parseInt(u64, dent.name[1..], 10) catch continue;
            // Never reuse an id that has ever existed on disk — even a
            // dropped leftover's delete could fail and leave a dirty dir.
            if (id >= self.next_id) self.next_id = id + 1;
            if (self.loadEntry(id)) |loaded| {
                pending.append(self.allocator, .{ .e = loaded.e, .mtime = loaded.mtime }) catch {
                    var le = loaded.e;
                    self.freeIndexEntryOwned(&le);
                    continue;
                };
            } else {
                // Crash leftover / corrupt — remove it.
                log.info("  [disk-cache] dropping incomplete entry e{d}\n", .{id});
                self.deleteEntryDir(id);
            }
        }

        std.mem.sort(Pending, pending.items, {}, struct {
            fn lessThan(_: void, a: Pending, b: Pending) bool {
                return a.mtime < b.mtime;
            }
        }.lessThan);

        // Shared chunk files are counted once.
        var seen = std.AutoHashMap(u64, void).init(self.allocator);
        defer seen.deinit();
        for (pending.items) |*p| {
            p.e.last_used = self.bump();
            self.billChunksOnce(&p.e, &seen);
            self.entries.append(self.allocator, p.e) catch {
                self.freeIndexEntryOwned(&p.e);
                continue;
            };
            self.total_bytes += p.e.bytes;
        }
        if (self.entries.items.len > 0) {
            log.info("  [disk-cache] scanned {d} persisted entries ({d:.1} MB) at {s}\n", .{
                self.entries.items.len,
                @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0),
                self.root,
            });
        }
    }

    /// Re-bill a scanned entry's chunk files against the inodes already counted this scan: the
    /// first entry to see an inode pays for it.
    fn billChunksOnce(self: *DiskTier, e: *IndexEntry, seen: *std.AutoHashMap(u64, void)) void {
        var billed: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
        for (e.chunk_bytes, 0..) |cb, i| {
            const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, e.id, i }) catch {
                billed += cb;
                continue;
            };
            defer self.allocator.free(cp);
            const st = statFile(self.io, cp) orelse {
                billed += cb;
                continue;
            };
            if (st.nlink <= 1) {
                billed += st.size;
                continue;
            }
            const ino: u64 = @intCast(st.inode);
            const gop = seen.getOrPut(ino) catch {
                billed += st.size;
                continue;
            };
            if (!gop.found_existing) billed += st.size;
        }
        e.bytes = billed;
    }

    fn loadEntry(self: *DiskTier, id: u64) ?struct { e: IndexEntry, mtime: i128 } {
        const meta_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, id }) catch return null;
        defer self.allocator.free(meta_path);

        const stat = statFile(self.io, meta_path) orelse return null;
        const content = readFileAlloc(self.allocator, self.io, meta_path, 64 * 1024) orelse return null;
        defer self.allocator.free(content);

        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const obj = parsed.value.object;

        const version = jsonU64(obj, "v") orelse return null;
        // v2 = pure-attention (no ssm field); v3 adds state checkpoints; v4
        // adds optional spec snapshots; v5 a head's QSA half; v6 inherited
        // chunks. All restore; a lower-version entry just carries none of the newer state.
        // v9 adds media spans (meta "media"). A v9 entry read by an older binary
        // fails that binary's gate and is dropped — the honest outcome for
        // media-keyed rows, never a token-only restore past a placeholder.
        if (version < 2 or version > 9) return null;
        // v6: the leading `inherited_chunks` chunk files are hard links into a donor's.
        const inherited_rec: u64 = jsonU64(obj, "inherited_chunks") orelse 0;
        var kv_len = jsonU64(obj, "kv_len") orelse return null;
        const n_tokens = jsonU64(obj, "tokens") orelse return null;
        const chunk_tokens = jsonU64(obj, "chunk_tokens") orelse return null;
        const has_tools_v = obj.get("has_tools") orelse return null;
        if (has_tools_v != .bool) return null;
        const chunk_bytes_v = obj.get("chunk_bytes") orelse return null;
        if (chunk_bytes_v != .array) return null;

        // Chunk geometry must match this tier's configuration — a stale root
        // written under a different chunk size can't be extended coherently.
        if (chunk_tokens != self.chunk_tokens) return null;
        if (kv_len == 0 or kv_len > n_tokens) return null;

        const quant = manifestQuant(obj) orelse return null;

        const n_chunks: u64 = (kv_len + chunk_tokens - 1) / chunk_tokens;
        if (chunk_bytes_v.array.items.len != n_chunks) return null;

        // Validate each chunk file's size against the recorded one. A kill -9
        // mid-flush truncates the chunk being (re)written while meta still
        // describes the previous valid state — restoring it would poison the
        // cache (live: MLX "invalid data offsets exceeding the size of the
        // file"). Clamp to the last contiguous valid chunk and salvage the
        // prefix.
        var valid_chunks: u64 = 0;
        while (valid_chunks < n_chunks) : (valid_chunks += 1) {
            const want_v = chunk_bytes_v.array.items[@intCast(valid_chunks)];
            if (want_v != .integer or want_v.integer < 0) break;
            const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, valid_chunks }) catch return null;
            defer self.allocator.free(cp);
            const have = fileSize(self.io, cp) orelse break;
            if (have != @as(u64, @intCast(want_v.integer))) break;
        }
        if (valid_chunks < n_chunks) {
            const salvaged = valid_chunks * chunk_tokens;
            log.info("  [disk-cache] e{d}: chunk {d} invalid — salvaging {d}/{d} tokens\n", .{ id, valid_chunks, salvaged, kv_len });
            kv_len = salvaged;
            if (kv_len < MIN_PERSIST_TOKENS) return null;
        }

        const chunk_bytes = self.allocator.alloc(u64, @intCast(valid_chunks)) catch return null;
        for (chunk_bytes, 0..) |*cb, i| cb.* = @intCast(chunk_bytes_v.array.items[i].integer);

        // Token record.
        const tokens_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/tokens.bin", .{ self.root, id }) catch {
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(tokens_path);
        const raw = readFileAlloc(self.allocator, self.io, tokens_path, 64 * 1024 * 1024) orelse {
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(raw);
        if (raw.len != n_tokens * 4) {
            self.allocator.free(chunk_bytes);
            return null;
        }
        const tokens = self.allocator.alloc(u32, n_tokens) catch {
            self.allocator.free(chunk_bytes);
            return null;
        };
        for (tokens, 0..) |*t, i| {
            t.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        }

        const inherited: u32 = @intCast(@min(inherited_rec, chunk_bytes.len));
        var total: u64 = @as(u64, tokens.len) * 4;
        for (chunk_bytes[inherited..]) |cb| total += cb;

        // v3 listed state checkpoints for a recurrent architecture. No such architecture is served
        // any more, so an entry that lists one is unusable (KV without its state) and is dropped;
        // v2 entries and an empty list (what every current entry writes) load as before.
        if (obj.get("ssm")) |ssm_v| {
            if (ssm_v == .array and ssm_v.array.items.len > 0) {
                log.info("  [disk-cache] e{d}: lists state checkpoints of a removed architecture - dropping the entry\n", .{id});
                self.allocator.free(tokens);
                self.allocator.free(chunk_bytes);
                return null;
            }
        }

        // v4 spec snapshots: validated by recorded file size (kill -9
        // salvage), but a bad spec drops only the SPEC — a restore then
        // starts blind, which is today's v2/v3 behavior anyway.
        var spec_bytes: u64 = 0;
        var spec_dflash: ?SpecMeta = null;
        if (obj.get("spec")) |spec_v| parse_spec: {
            if (spec_v != .object) break :parse_spec;
            const so = spec_v.object;
            const rec_bytes = jsonU64(so, "bytes") orelse break :parse_spec;
            const sp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/spec.safetensors", .{ self.root, id }) catch break :parse_spec;
            defer self.allocator.free(sp);
            const have = fileSize(self.io, sp) orelse break :parse_spec;
            if (have != rec_bytes) {
                log.info("  [disk-cache] e{d}: spec sidecar size mismatch — dropping the spec\n", .{id});
                break :parse_spec;
            }
            spec_dflash = parseSpecMeta(so, "dflash");
            if (spec_dflash == null) break :parse_spec;
            spec_bytes = rec_bytes;
            total += rec_bytes;
        }
        // v9: media spans [[start,key],...] — the key is the u64 pixel hash
        // printed as its SIGNED decimal (bit-exact round-trip past 2^63). An
        // unreadable, empty, out-of-order, or version-inconsistent list drops
        // the WHOLE entry: loading a media record as text-only would let
        // token-prefix matching restore rows past a placeholder under a
        // possibly different image — a wrong restore, the one failure mode
        // worse than slow. Spans at/above kv_len stay loaded: the token record
        // reaches them in the supersede gate even while an incremental flush
        // has not persisted their rows yet.
        var media: []MediaSpan = &[_]MediaSpan{};
        var media_ok = false;
        media_parse: {
            const mv = obj.get("media") orelse break :media_parse;
            if (mv != .array or mv.array.items.len == 0) break :media_parse;
            if (version < 9) break :media_parse; // a media list on a pre-v9 stamp is a lie
            var list = std.ArrayList(MediaSpan).empty;
            defer list.deinit(self.allocator);
            for (mv.array.items) |sp_v| {
                if (sp_v != .array or sp_v.array.items.len != 2 or
                    sp_v.array.items[0] != .integer or sp_v.array.items[1] != .integer) break :media_parse;
                const st = std.math.cast(u32, sp_v.array.items[0].integer) orelse break :media_parse;
                if (@as(u64, st) >= n_tokens) break :media_parse; // span past the token record
                if (list.items.len > 0 and st <= list.items[list.items.len - 1].start) break :media_parse;
                list.append(self.allocator, .{ .start = st, .key = @bitCast(sp_v.array.items[1].integer) }) catch break :media_parse;
            }
            media = self.allocator.dupe(MediaSpan, list.items) catch break :media_parse;
            media_ok = true;
        }
        if ((obj.contains("media") or version == 9) and !media_ok) {
            log.info("  [disk-cache] e{d}: media record unreadable — dropping the entry (never restore text-only past an item)\n", .{id});
            self.allocator.free(tokens);
            self.allocator.free(chunk_bytes);
            return null;
        }

        return .{
            .e = .{
                .id = id,
                .tokens = tokens,
                .kv_len = @intCast(kv_len),
                .has_tools = has_tools_v.bool,
                .quant = quant,
                .bytes = total,
                .inherited_chunks = inherited,
                .chunk_bytes = chunk_bytes,
                .spec_bytes = spec_bytes,
                .spec_dflash = spec_dflash,
                .media = media,
                .last_used = 0,
            },
            .mtime = stat.mtime.nanoseconds,
        };
    }
};

// ── Model fingerprint ──

/// Identity of the weights the persisted KV was computed against: absolute
/// model dir + config.json size/mtime. A re-downloaded or re-quantized
/// checkpoint rewrites config.json, which rolls the fingerprint and orphans
/// the stale KV (GC'd by the disk budget eventually; different fingerprint
/// dirs never mix). 16 hex chars of XxHash64.
/// Root-wide LRU sweep across sibling model fingerprints (never `keep_root`, which the tier's
/// own `scan` owns), oldest first until they fit `sibling_budget`, plus index-less strays that
/// are old enough to be crash leftovers. Best effort.
pub fn sweepBase(
    allocator: std.mem.Allocator,
    io: std.Io,
    base_dir: []const u8,
    keep_root: []const u8,
    sibling_budget: u64,
) void {
    var base = std.Io.Dir.openDirAbsolute(io, base_dir, .{ .iterate = true }) catch return;
    defer base.close(io);

    const Victim = struct { path: []u8, bytes: u64, mtime: i128 };
    var victims = std.ArrayList(Victim).empty;
    defer {
        for (victims.items) |v| allocator.free(v.path);
        victims.deinit(allocator);
    }
    var total: u64 = 0;
    var strays: usize = 0;

    var fps = base.iterate();
    while (fps.next(io) catch null) |fp| {
        if (fp.kind != .directory) continue;
        const fp_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fp.name }) catch continue;
        defer allocator.free(fp_abs);
        if (std.mem.eql(u8, fp_abs, keep_root)) continue; // the live tier owns its own

        var fpd = std.Io.Dir.openDirAbsolute(io, fp_abs, .{ .iterate = true }) catch continue;
        defer fpd.close(io);
        var es = fpd.iterate();
        while (es.next(io) catch null) |dent| {
            if (dent.kind != .directory) continue;
            if (dent.name.len < 2 or dent.name[0] != 'e') continue;
            const e_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ fp_abs, dent.name }) catch continue;
            const meta = std.fmt.allocPrint(allocator, "{s}/meta.json", .{e_abs}) catch {
                allocator.free(e_abs);
                continue;
            };
            defer allocator.free(meta);
            const st = std.Io.Dir.cwd().statFile(io, meta, .{}) catch {
                // No index: a crash leftover, OR another server's flush in progress (meta lands
                // last). Age is the only signal.
                if (dirYoungerThan(io, e_abs, STRAY_MIN_AGE_NS)) {
                    allocator.free(e_abs);
                    continue;
                }
                deleteTreeAbsolute(io, e_abs);
                allocator.free(e_abs);
                strays += 1;
                continue;
            };
            // A `.tmp` older than the same bar is a crash leftover of the writer's tmp+rename.
            reapStaleTmp(io, e_abs);
            const bytes = dirBytes(io, e_abs);
            total += bytes;
            victims.append(allocator, .{ .path = e_abs, .bytes = bytes, .mtime = st.mtime.nanoseconds }) catch {
                allocator.free(e_abs);
            };
        }
    }

    if (strays > 0) log.info("  [disk-cache] swept {d} stray entry directories under {s}\n", .{ strays, base_dir });
    if (total <= sibling_budget) return;

    std.mem.sort(Victim, victims.items, {}, struct {
        fn lt(_: void, a: Victim, b: Victim) bool {
            return a.mtime < b.mtime;
        }
    }.lt);
    var freed: u64 = 0;
    for (victims.items) |v| {
        if (total -| freed <= sibling_budget) break;
        deleteTreeAbsolute(io, v.path);
        freed += v.bytes;
    }
    if (freed > 0) {
        log.info("  [disk-cache] root-wide LRU freed {d} MB across other models ({d} MB held, {d} MB budget)\n", .{
            freed >> 20,
            total >> 20,
            sibling_budget >> 20,
        });
    }
}

/// How old an index-less entry directory must be before a sweep may treat it as a crash leftover.
const STRAY_MIN_AGE_NS: i128 = 10 * 60 * @as(i128, std.time.ns_per_s);

/// True when any regular file directly inside `dir_abs` was modified within `age_ns`. Unreadable = young.
fn dirYoungerThan(io: std.Io, dir_abs: []const u8, age_ns: i128) bool {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return true;
    defer d.close(io);
    const now = std.Io.Timestamp.now(io, .real).nanoseconds;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        const st = d.statFile(io, dent.name, .{}) catch return true;
        if (now -| st.mtime.nanoseconds < age_ns) return true;
    }
    return false;
}

/// Delete `.tmp` files in `dir_abs` older than `STRAY_MIN_AGE_NS`.
fn reapStaleTmp(io: std.Io, dir_abs: []const u8) void {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return;
    defer d.close(io);
    const now = std.Io.Timestamp.now(io, .real).nanoseconds;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        if (!std.mem.endsWith(u8, dent.name, ".tmp")) continue;
        const st = d.statFile(io, dent.name, .{}) catch continue;
        if (now -| st.mtime.nanoseconds < STRAY_MIN_AGE_NS) continue;
        d.deleteFile(io, dent.name) catch {};
    }
}

/// Total bytes of the regular files directly inside `dir_abs`.
fn dirBytes(io: std.Io, dir_abs: []const u8) u64 {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        const st = d.statFile(io, dent.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

fn deleteTreeAbsolute(io: std.Io, dir_abs: []const u8) void {
    const parent = std.fs.path.dirname(dir_abs) orelse return;
    const name = std.fs.path.basename(dir_abs);
    var pd = std.Io.Dir.openDirAbsolute(io, parent, .{ .iterate = true }) catch return;
    defer pd.close(io);
    pd.deleteTree(io, name) catch {};
}

/// Historical fingerprint for the default cache layout. Kept as a wrapper so
/// existing callers and existing SSD roots remain stable.
pub fn modelFingerprint(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]u8 {
    return modelFingerprintWithLayout(allocator, io, model_dir, null);
}

/// Fingerprint a model plus an optional cache-layout namespace
/// (`ModelConfig.cacheLayoutNamespace`). A layout that changes what the stored
/// K/V mean gets a versioned marker, so an old SSD root is never restored into
/// it; null keeps the historical fingerprint.
pub fn modelFingerprintWithLayout(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_dir: []const u8,
    layout_namespace: ?[]const u8,
) ![]u8 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.BadModelDir;
    var h = std.hash.XxHash64.init(0x6b76_6361_6368_6531);
    h.update(model_dir);
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    if (statFile(io, cfg_path)) |st| {
        h.update(std.mem.asBytes(&st.size));
        const mt: i128 = st.mtime.nanoseconds;
        h.update(std.mem.asBytes(&mt));
    }
    if (model.getConfigOverrides()) |raw| h.update(raw);
    if (layout_namespace) |layout| {
        h.update("\x00mlx-serve-kv-layout\x00");
        h.update(layout);
    }
    return std.fmt.allocPrint(allocator, "{x:0>16}", .{h.final()});
}

/// Default persistence root: `~/.mlx-serve/kv-cache`.
pub fn defaultBaseDir(allocator: std.mem.Allocator) ![]u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoHome);
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/kv-cache", .{home});
}

// ── Small fs helpers ──

/// Bytes of an entry's files that are never shared: tokens.bin and the spec sidecar.
fn nonChunkBytes(e: *const IndexEntry) u64 {
    const n: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
    return n;
}

fn clampAdd(base: u64, delta: i64) u64 {
    const v: i128 = @as(i128, base) + @as(i128, delta);
    return if (v < 0) 0 else @intCast(v);
}

/// Length of the longest common prefix of two token slices.
pub fn commonPrefixLen(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

var chunk_share_env_cached: ?bool = null;
pub var chunk_share_override: ?bool = null;

/// `MLX_SERVE_SSD_CHUNK_SHARE=0` restores the write-everything commit (SSD-first arm only).
pub fn chunkShareEnabled() bool {
    if (chunk_share_override) |v| return v;
    if (chunk_share_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_SSD_CHUNK_SHARE") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    chunk_share_env_cached = v;
    return v;
}

fn statFile(io: std.Io, abs_path: []const u8) ?std.Io.File.Stat {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    return f.stat(io) catch null;
}

fn fileSize(io: std.Io, abs_path: []const u8) ?u64 {
    const st = statFile(io, abs_path) orelse return null;
    return st.size;
}

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, abs_path: []const u8, limit: usize) ?[]u8 {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

fn jsonU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    if (v != .integer) return null;
    if (v.integer < 0) return null;
    return @intCast(v.integer);
}

/// A manifest scalar narrowed to the field that will hold it. meta.json is on disk and
/// hand-editable, so a value that does not fit drops its record rather than being cast.
fn jsonInt(comptime T: type, obj: std.json.ObjectMap, key: []const u8) ?T {
    const v = obj.get(key) orelse return null;
    if (v != .integer) return null;
    return std.math.cast(T, v.integer);
}

/// The KV quant config a manifest describes, or null when it is not one this build can hold.
fn manifestQuant(obj: std.json.ObjectMap) ?kv_quant.KVQuantConfig {
    const scheme_v = obj.get("scheme") orelse return null;
    if (scheme_v != .string) return null;
    const scheme = std.meta.stringToEnum(kv_quant.Scheme, scheme_v.string) orelse return null;
    if (scheme == .off) return kv_quant.KVQuantConfig.dense;
    if (scheme != .affine) return null;
    const bits = jsonInt(u8, obj, "bits") orelse return null;
    if (bits != 4 and bits != 8) return null;
    const gs = jsonInt(u32, obj, "group_size") orelse return null;
    if (gs == 0) return null;
    var q = kv_quant.KVQuantConfig.affine(bits);
    q.group_size = gs;
    return q;
}

fn writeSpecMetaJson(a: std.mem.Allocator, w: *std.ArrayList(u8), name: []const u8, sm: SpecMeta) !void {
    try w.print(a, ",\"{s}\":{{\"base\":{d},\"step\":{d},\"layers\":{d},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d}", .{
        name, sm.base, sm.step, sm.layers, @tagName(sm.quant.scheme), sm.quant.bits, sm.quant.group_size,
    });
    try w.appendSlice(a, "}");
}

fn parseSpecMeta(obj: std.json.ObjectMap, key: []const u8) ?SpecMeta {
    const v = obj.get(key) orelse return null;
    if (v != .object) return null;
    const o = v.object;
    const base = jsonU64(o, "base") orelse return null;
    const step = jsonInt(u32, o, "step") orelse return null;
    const layers = jsonInt(u32, o, "layers") orelse return null;
    if (step == 0 or layers == 0) return null;
    const quant = manifestQuant(o) orelse return null;
    return .{ .base = base, .step = step, .layers = layers, .quant = quant };
}

// ── Tests ──

const testing = std.testing;

fn fillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32, head_dim: u32, seed: f64, dtype: mlx.mlx_dtype) !void {
    // Drive the cache through its real update path with deterministic
    // arange-derived K/V so restored values are checkable. Dense tests use
    // float32 (every position stays exactly distinguishable); the affine test
    // uses bf16, the production dtype the quant write path expects.
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const count: f64 = @floatFromInt(step * head_dim);
            const base: f64 = seed + @as(f64, @floatFromInt(written * head_dim + li * 1_000_000));
            try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            const shape = [_]c_int{ 1, 1, @intCast(step), @intCast(head_dim) };
            try mlx.check(mlx.mlx_reshape(&shaped, flat, &shape, 4, s));
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_astype(&k, shaped, dtype, s));
            // V = -K so a restore-side K/V swap can't false-pass.
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_negative(&v, k, s));
            var view = try cache.update(li, k, v, s, 0);
            view.deinit();
        }
        written += step;
    }
}

fn cacheValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream) !f32 {
    return cacheBufValueAt(cache, layer, pos, d, s, false);
}

fn cacheBufValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream, values: bool) !f32 {
    const entry = &cache.entries[layer];
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const st = [_]c_int{ 0, 0, @intCast(pos), @intCast(d) };
    const sp = [_]c_int{ 1, 1, @intCast(pos + 1), @intCast(d + 1) };
    const sd = [_]c_int{ 1, 1, 1, 1 };
    const buf = if (values) entry.values else entry.keys;
    try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, sliced, .float32, s));
    _ = mlx.mlx_array_eval(f);
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return ptr[0];
}

fn tmpRoot(tmp: *std.testing.TmpDir, io: std.Io, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(io, buf);
    return buf[0..n];
}

test "DiskTier: chunked commit + restore round-trips exact KV, step, offsets" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier.deinit();

    // 600 tokens => 5 chunks at 128 (last partial: 88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    try testing.expectEqual(@as(usize, 600), cache.step);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // Restore into a fresh cache (fresh tier too — proves the restart path).
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());

    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, &.{}).?;
    try testing.expectEqual(@as(u32, 600), m.usable);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
    try testing.expectEqual(@as(usize, 600), cache2.step);
    for (cache2.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 600), e.offset);
    }

    // Spot-check exact values across chunk boundaries and layers.
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 7 }, .{ 128, 0 }, .{ 300, 3 }, .{ 511, 7 }, .{ 512, 0 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 3) : (li += 1) {
            const want = try cacheValueAt(&cache, li, p[0], p[1], s);
            const got = try cacheValueAt(&cache2, li, p[0], p[1], s);
            try testing.expectEqual(want, got);
            // V was written as -K: restored values must mirror that, so a
            // restore-side K/V swap or shared-buffer mixup fails here.
            const got_v = try cacheBufValueAt(&cache2, li, p[0], p[1], s, true);
            try testing.expectEqual(-want, got_v);
        }
    }

    // Mismatched key never matches.
    try testing.expect(tier2.bestMatch(&tokens, true, kv_quant.KVQuantConfig.dense, &.{}) == null);
    try testing.expect(tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.affine(4), &.{}) == null);
}

test "DiskTier: extend commit appends only new chunks (full chunks untouched)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ext", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, tokens[0..600], false, s);

    // Tamper-mark chunk 0 (a FULL chunk): record its mtime, then extend the
    // entry and assert chunk 0 was not rewritten while chunk 4 (the old
    // partial) was, and new chunks appeared.
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const c4_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000004.safetensors", .{base});
    defer testing.allocator.free(c4_path);
    const c6_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000006.safetensors", .{base});
    defer testing.allocator.free(c6_path);
    const c0_before = statFile(io, c0_path).?.mtime.nanoseconds;
    const c4_before = statFile(io, c4_path).?.mtime.nanoseconds;
    try testing.expect(fileSize(io, c6_path) == null);

    // Ensure the extend write lands at a measurably later mtime.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Same prefix, 300 more tokens.
    try fillCache(&cache, s, 1, 300, 8, 4800.0, .float32);
    try testing.expectEqual(@as(usize, 900), cache.step);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 900), tier.entries.items[0].kv_len);

    try testing.expectEqual(c0_before, statFile(io, c0_path).?.mtime.nanoseconds); // untouched
    try testing.expect(statFile(io, c4_path).?.mtime.nanoseconds != c4_before); // partial rewritten
    try testing.expect(fileSize(io, c6_path) != null); // new tail chunk

    // Restore the extended entry and check a value in the extension range.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 900), restored);
    const want = try cacheValueAt(&cache, 0, 750, 5, s);
    const got = try cacheValueAt(&cache2, 0, 750, 5, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: identical re-commit is a no-op; shorter prefix is superseded" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-noop", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-noop/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const before = statFile(io, c0_path).?.mtime.nanoseconds;

    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Identical commit — nothing rewritten, no second entry.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(before, statFile(io, c0_path).?.mtime.nanoseconds);

    // A shorter-prefix commit of the same conversation is covered by the
    // existing entry — also a no-op.
    var short_cache = try KVCache.init(testing.allocator, 1);
    defer short_cache.deinit();
    try fillCache(&short_cache, s, 1, 512, 8, 0.0, .float32);
    _ = try tier.appendCommit(short_cache.entries, short_cache.step, short_cache.config, tokens[0..512], false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
}

test "DiskTier: byte budget evicts LRU entries, keeps newest" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    // Budget deliberately tiny: every new entry evicts the previous one.
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-gc", 4096, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 520, 8, 0.0, .float32);

    var tokens_a: [520]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [520]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_a, false, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, s);
    // Both entries exceed 4 KB each — only the newest survives.
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expect(std.mem.eql(u32, tier.entries.items[0].tokens, &tokens_b));

    // The evicted directory is gone from disk.
    const e1_meta = try std.fmt.allocPrint(testing.allocator, "{s}/fp-gc/e1/meta.json", .{base});
    defer testing.allocator.free(e1_meta);
    try testing.expect(statFile(io, e1_meta) == null);
}

test "DiskTier: scan drops crash leftovers (no meta.json)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
        // Simulate a crash mid-write of a SECOND entry: chunks, no meta.
        try tmp.dir.createDirPath(io, "fp-crash/e9");
        try tmp.dir.writeFile(io, .{ .sub_path = "fp-crash/e9/c000000.safetensors", .data = "junk" });
    }

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // The leftover dir was removed.
    const leftover = try std.fmt.allocPrint(testing.allocator, "{s}/fp-crash/e9/c000000.safetensors", .{base});
    defer testing.allocator.free(leftover);
    try testing.expect(statFile(io, leftover) == null);
    // next_id moved past the dropped id (no reuse of a dirty dir name).
    try testing.expect(tier2.next_id >= 10);
}

test "DiskTier: affine-quant cache round-trips all six buffers" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-q", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    // head_dim must be a multiple of group_size (64) for affine quant.
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    try testing.expectEqual(@as(usize, 520), cache.step);

    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);

    var cache2 = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache2.deinit();
    const m = tier.bestMatch(&tokens, false, qcfg, &.{}).?;
    try testing.expectEqual(@as(u32, 520), m.usable);
    const restored = try tier.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 520), restored);

    // Dense read-back through the cache's own dequant path must agree.
    // Truncate BOTH caches to the same length first — restore leaves views
    // empty (the KVCache.restore contract) and truncate to len < offset
    // rebuilds them on both sides identically.
    try cache.truncate(519, s);
    try cache2.truncate(519, s);
    var v1 = try cache.denseView(0, s);
    defer v1.deinit();
    var v2 = try cache2.denseView(0, s);
    defer v2.deinit();
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 63 }, .{ 128, 0 }, .{ 300, 5 }, .{ 511, 1 }, .{ 518, 63 } };
    for (probes) |p| {
        var d1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d1);
        var d2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d2);
        const st = [_]c_int{ 0, 0, @intCast(p[0]), @intCast(p[1]) };
        const sp = [_]c_int{ 1, 1, @intCast(p[0] + 1), @intCast(p[1] + 1) };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&d1, v1.k, &st, 4, &sp, 4, &sd, 4, s));
        try mlx.check(mlx.mlx_slice(&d2, v2.k, &st, 4, &sp, 4, &sd, 4, s));
        var f1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f1);
        var f2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f2);
        try mlx.check(mlx.mlx_astype(&f1, d1, .float32, s));
        try mlx.check(mlx.mlx_astype(&f2, d2, .float32, s));
        _ = mlx.mlx_array_eval(f1);
        _ = mlx.mlx_array_eval(f2);
        try testing.expectEqual(mlx.mlx_array_data_float32(f1).?[0], mlx.mlx_array_data_float32(f2).?[0]);
    }
}

test "DiskTier: truncated chunk file salvages the valid prefix at scan (kill -9 shape)" {
    // A kill -9 mid-flush leaves a chunk file truncated while meta.json (the
    // commit point, written last) still describes the PREVIOUS valid state —
    // whose recorded size for that chunk no longer matches the file. Live
    // capture: MLX "invalid data offsets exceeding the size of the file" on
    // restore. The scan must clamp the entry to the last contiguous chunk
    // whose size matches meta, salvaging the prefix instead of poisoning a
    // restore (or dropping everything).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 700, 8, 0.0, .float32);
        var tokens: [700]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    }

    // Truncate chunk 4 (positions [512, 640)) — chunks 0-3 stay valid.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-trunc/e1/c000004.safetensors", .data = "trunc" });

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // kv_len clamped to the last valid chunk boundary: 4 * 128 = 512.
    try testing.expectEqual(@as(u32, 512), tier2.entries.items[0].kv_len);

    // The salvaged prefix restores cleanly.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 512), restored);
    try testing.expectEqual(@as(usize, 512), cache2.step);
}

test "DiskTier: flush byte cap persists incrementally across commits" {
    // A 4 GB first-commit write used to stall the NEXT request ~2.5 s (the
    // flush runs on the inference thread). appendCommit caps the bytes
    // written per call at max_flush_bytes, persists a chunk-aligned prefix,
    // and reports incomplete so the caller re-flushes on later turns.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-cap", 0, 128);
    defer tier.deinit();
    tier.max_flush_bytes = 1; // every chunk write exceeds the cap -> 1 chunk/flush

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // First flush: 1 chunk (128 tokens), incomplete.
    const c1 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(PersistOutcome.partial, c1);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].kv_len);
    // Second flush continues from where it left off.
    const c2 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(PersistOutcome.partial, c2);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    // Keep flushing until complete; entry must land at the full 600.
    var guard: u32 = 0;
    while (guard < 10) : (guard += 1) {
        if (try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s) == .persisted) break;
    }
    try testing.expectEqual(@as(u32, 600), tier.entries.items[0].kv_len);

    // Restored content from an incrementally-persisted entry is exact.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 600), restored);
    const want = try cacheValueAt(&cache, 0, 599, 7, s);
    const got = try cacheValueAt(&cache2, 0, 599, 7, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: cache ahead of the token record persists the clamped prefix (EOS-stop shape)" {
    // On an EOS stop the generator has forwarded the terminator tokens into
    // the cache but they're not part of the committed token record — live
    // capture: step=2054 vs tokens=2052. The RAM tier tolerates this
    // (truncate hides the tail); the disk tier must persist min(step,
    // tokens.len) positions instead of silently skipping the whole commit.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eos", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 604, 8, 0.0, .float32); // 2 positions past the record
    var tokens: [602]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 602), tier.entries.items[0].kv_len);

    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 602), restored);
    const want = try cacheValueAt(&cache, 0, 601, 3, s);
    const got = try cacheValueAt(&cache2, 0, 601, 3, s);
    try testing.expectEqual(want, got);
}

// ── test helpers ──

test "DiskTier: short caches are never persisted" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer tier.deinit();

    // Below MIN_PERSIST_TOKENS → skipped.
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 128, 8, 0.0, .float32);
    var tokens: [128]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
}

test "DiskTier: v4 spec snapshots round-trip; geometry mismatches decline; v3 restores clean" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    // dflash assistant context: 2 layers over the full 600 positions, base 0.
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    try fillCache(&dfl, s, 2, 600, 8, 3.5, .float32);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{}, .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 }, s);
    try testing.expect(tier.entries.items[0].spec_bytes > 0);

    // Restart shape: a fresh tier over the same root re-reads the spec meta.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, &.{}).?;

    var loaded = tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    try testing.expectEqual(@as(usize, 0), loaded.base);
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    try dfl2.restore(&loaded.snap);
    loaded.snap.deinit();
    // Exact values, K and V (V = -K in fillCache — a swap can't false-pass).
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 300, 3 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 2) : (li += 1) {
            try testing.expectEqual(
                try cacheValueAt(&dfl, li, p[0], p[1], s),
                try cacheValueAt(&dfl2, li, p[0], p[1], s),
            );
            try testing.expectEqual(
                try cacheBufValueAt(&dfl, li, p[0], p[1], s, true),
                try cacheBufValueAt(&dfl2, li, p[0], p[1], s, true),
            );
        }
    }

    // A target the geometry doesn't fit DECLINES (KVCache.restore asserts
    // equal layer counts — the check must fire before it).
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 3, kv_quant.KVQuantConfig.dense) == null);
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.affine(8)) == null);

    // A commit WITHOUT spec payloads carries none (and, per the supersede
    // rule, would delete a stale sidecar on its own entry).
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);
    _ = try tier2.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, s);
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense, &.{}).?;
    try testing.expect(tier2.loadSpecSnap(mb.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);

    // v3 entry (written by an older binary): rewrite the manifest to v3 with
    // no spec object — the entry must restore CLEAN, spec-less.
    {
        const e_id = tier2.entries.items[m.idx].id;
        const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-spec/e{d}/meta.json", .{ base, e_id });
        defer testing.allocator.free(meta_path);
        const content = readFileAlloc(testing.allocator, io, meta_path, 64 * 1024) orelse return error.TestMetaUnreadable;
        defer testing.allocator.free(content);
        const spec_at = std.mem.indexOf(u8, content, ",\"spec\":") orelse return error.TestSpecFieldMissing;
        var rewritten = std.ArrayList(u8).empty;
        defer rewritten.deinit(testing.allocator);
        try rewritten.appendSlice(testing.allocator, content[0..spec_at]);
        try rewritten.append(testing.allocator, '}');
        // The manifest stamps the lowest version that describes the entry.
        _ = std.mem.replace(u8, rewritten.items, "\"v\":6", "\"v\":3", rewritten.items);
        _ = std.mem.replace(u8, rewritten.items, "\"v\":5", "\"v\":3", rewritten.items);
        _ = std.mem.replace(u8, rewritten.items, "\"v\":4", "\"v\":3", rewritten.items);
        const f = try std.Io.Dir.createFileAbsolute(io, meta_path, .{});
        defer f.close(io);
        var wb: [4096]u8 = undefined;
        var fw = f.writer(io, &wb);
        try fw.interface.writeAll(rewritten.items);
        try fw.interface.flush();
    }
    var tier3 = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier3.deinit();
    try testing.expectEqual(@as(usize, 2), tier3.entryCount());
    const m3 = tier3.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, &.{}).?;
    try testing.expectEqual(@as(u32, 600), m3.usable);
    try testing.expect(tier3.loadSpecSnap(m3.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);
    var cache3 = try KVCache.init(testing.allocator, 2);
    defer cache3.deinit();
    const restored = try tier3.restoreInto(&cache3, m3.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
}

test "modelFingerprint: stable per path, rolls with config.json changes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    const fp1 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp1);
    const fp2 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp2);
    try testing.expectEqualStrings(fp1, fp2);
    try testing.expectEqual(@as(usize, 16), fp1.len);

    // Rewriting config.json (re-download / re-quant) rolls the fingerprint.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"y\",\"pad\":1}" });
    const fp3 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp3);
    try testing.expect(!std.mem.eql(u8, fp1, fp3));

    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, ""));
    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, "rel/path"));
}

// Bar: the model root rolls with --config-overrides, so a cache written under other overrides never restores.
test "modelFingerprint: rolls with --config-overrides" {
    defer model.setConfigOverrides(null);
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    model.setConfigOverrides(null);
    const fp_none = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_none);

    const yarn = "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0}}}";
    model.setConfigOverrides(yarn);
    const fp_over = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_over);
    try testing.expectEqual(@as(usize, 16), fp_over.len);
    try testing.expect(!std.mem.eql(u8, fp_none, fp_over));

    model.setConfigOverrides(yarn);
    const fp_again = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_again);
    try testing.expectEqualStrings(fp_over, fp_again);

    // Raw-bytes pin: a whitespace-different spelling of the same JSON is a
    // different fingerprint (canonicalizing would hide a YaRN boot restoring
    // an unscaled SSD prefix if the override was re-spelled).
    const spaced_override = "{\"text_config\": {\"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": 4.0}}}";
    model.setConfigOverrides(spaced_override);
    const fp_ws = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_ws);
    try testing.expect(!std.mem.eql(u8, fp_over, fp_ws));
}

test "spec meta json: a record round-trips" {
    var w = std.ArrayList(u8).empty;
    defer w.deinit(testing.allocator);
    const sm: SpecMeta = .{
        .base = 62_000,
        .step = 700,
        .layers = 1,
        .quant = kv_quant.KVQuantConfig.dense,
    };
    try writeSpecMetaJson(testing.allocator, &w, "dflash", sm);
    const rec = try std.fmt.allocPrint(testing.allocator, "{{\"bytes\":1{s}}}", .{w.items});
    defer testing.allocator.free(rec);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, rec, .{});
    defer parsed.deinit();
    const back = parseSpecMeta(parsed.value.object, "dflash") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(sm.base, back.base);
    try testing.expectEqual(sm.step, back.step);
}

test "DiskTier: SSD-first stages the flush off-thread and indexes LAST" {
    // The commit returns with the whole entry still staged, and meta.json is submitted after
    // every chunk so the FIFO lands it last.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssd-writer", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    try testing.expectEqual(SSD_FIRST_READBACK_BYTES, tier.max_flush_bytes);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(PersistOutcome.persisted, complete);
    try testing.expect(tier.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(u64, 0), tier.writer.?.filesWritten());
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-ssd-writer/e1/meta.json", .{}));

    var paths = std.ArrayList([]const u8).empty;
    defer {
        for (paths.items) |p| testing.allocator.free(p);
        paths.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&paths, testing.allocator);
    try testing.expectEqual(@as(usize, 6), paths.items.len); // 5 chunks + meta
    for (paths.items[0 .. paths.items.len - 1]) |p| {
        try testing.expect(std.mem.indexOf(u8, p, "/c0000") != null);
    }
    try testing.expect(std.mem.endsWith(u8, paths.items[paths.items.len - 1], "/meta.json"));

    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 6), tier.writer.?.filesWritten());

    // The staged bytes are a real safetensors image: a fresh tier restores it.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-ssd-writer", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    try testing.expectEqual(@as(u32, 640), try tier2.restoreInto(&cache2, 0, s));
    for ([_]u32{ 0, 1, 2 }) |li| {
        for ([_]u32{ 0, 127, 128, 511, 639 }) |pos| {
            try testing.expectEqual(
                try cacheValueAt(&cache, li, pos, 3, s),
                try cacheValueAt(&cache2, li, pos, 3, s),
            );
        }
    }
}

test "diskBudgetFromFreeSpace: reserve is min(64 GiB, 10% of volume); below the floor stores nothing" {
    const GB: u64 = 1 << 30;
    // 4 TB volume, 1 TB free: reserve is the 64 GiB cap.
    try testing.expectEqual(@as(?u64, 1024 * GB - 64 * GB), diskBudgetFromFreeSpace(0, 1024 * GB, 4096 * GB));
    try testing.expectEqual(@as(?u64, 100 * GB), diskBudgetFromFreeSpace(100 * GB, 1024 * GB, 4096 * GB));
    // Small volume: 10% is the binding reserve.
    try testing.expectEqual(@as(?u64, 60 * GB), diskBudgetFromFreeSpace(0, 80 * GB, 200 * GB));
    // Under the store floor: refuse, never a silent 0.
    try testing.expectEqual(@as(?u64, null), diskBudgetFromFreeSpace(0, 20 * GB, 200 * GB));
    try testing.expectEqual(@as(?u64, null), diskBudgetFromFreeSpace(500 * GB, 20 * GB, 200 * GB));
}

test "volumeSpace: the live probe is plausible or null (statfs ABI guard)" {
    // On this machine the probe must succeed: a null means the struct layout broke.
    const vs = volumeSpace("/") orelse return error.VolumeSpaceProbeFailed;
    try testing.expect(vs.total > 0);
    try testing.expect(vs.free <= vs.total);
    try testing.expect(vs.total > 1024 * 1024 * 1024); // a macOS root volume
}

test "volumeSpace: free is what the OS grants, never less than statfs' f_bavail" {
    // Purgeable space is not in f_bavail; the tier used to refuse a volume with 117 GB usable.
    // Three separate live probes: other writers move free space between them, hence the slack.
    const slack: u64 = 1 << 30;
    var st: DarwinStatfs = undefined;
    try testing.expect(statfs("/", &st) == 0);
    const vs = volumeSpace("/") orelse return error.VolumeSpaceProbeFailed;
    const granted = msv_volume_free_for_use("/");
    const bavail_bytes = @as(u64, @intCast(@max(st.f_bsize, 0))) * st.f_bavail;
    try testing.expect(vs.free + slack >= bavail_bytes);
    // `granted` is 0 when the OS probe fails (a session without the volume resource keys): `volumeSpace` then answers f_bavail.
    if (granted > 0 and granted <= vs.total) {
        try testing.expect(@max(granted, vs.free) - @min(granted, vs.free) < slack);
    } else {
        try testing.expect(@max(bavail_bytes, vs.free) - @min(bavail_bytes, vs.free) < slack);
    }
}

test "DiskTier: SSD-first declines to store when the VOLUME is short, and says so" {
    // 10 GiB free against a 512 GiB volume leaves nothing after the reserve: the tier stores nothing.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-short", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, s);
    tier.drainWriter();
    try testing.expect(tier.store_declined);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
    try testing.expectEqual(@as(u64, 0), tier.total_bytes);

    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, s);
    tier.drainWriter();
    try testing.expect(!tier.store_declined);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 640), tier.entries.items[0].kv_len);
}

test "DiskTier: the shutdown drain fences what is still queued at its deadline, and deinit writes none of it" {
    // Exit never waits on staged bytes past the deadline; the entry keeps its last landed manifest
    // (here none), so a restart scans nothing half-written.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const PastClock = struct {
        pub fn nowMs(_: @This()) i64 {
            return 1_000;
        }
    };
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-deadline", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        tier.enableBackgroundWriter();
        try testing.expect(tier.writer != null);
        tier.writer.?.setPaused(true);
        _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, s);
        try testing.expect(tier.drainWriterUntil(PastClock{}, 1_000) > 0);
    }
    var rescanned = try DiskTier.init(testing.allocator, io, base, "fp-deadline", 0, 128);
    defer rescanned.deinit();
    try testing.expectEqual(@as(usize, 0), rescanned.entryCount());
}

test "DiskTier: entries cross the SSD-first boundary in BOTH directions (SSD-first itself bumps no manifest)" {
    // SSD-first changes when chunks are written, never the
    // on-disk format: entries restore across both arms.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Written by the legacy path, read by SSD-first.
    {
        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-x-legacy", 0, 128);
        _ = try legacy.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
        legacy.deinit();

        var ssd = try DiskTier.init(testing.allocator, io, base, "fp-x-legacy", 0, 128);
        defer ssd.deinit();
        ssd.ssd_first = true;
        ssd.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        ssd.enableBackgroundWriter();
        try testing.expectEqual(@as(usize, 1), ssd.entryCount());
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        try testing.expectEqual(@as(u32, 600), try ssd.restoreInto(&out, 0, s));
        for ([_]u32{ 0, 1, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
    }

    // Written by SSD-first (background writer, hand-serialized), read by the legacy path.
    {
        var ssd = try DiskTier.init(testing.allocator, io, base, "fp-x-ssd", 0, 128);
        ssd.ssd_first = true;
        ssd.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        ssd.enableBackgroundWriter();
        _ = try ssd.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
        ssd.drainWriter();
        ssd.deinit();

        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-x-ssd", 0, 128);
        defer legacy.deinit();
        try testing.expect(legacy.writer == null);
        try testing.expectEqual(@as(usize, 1), legacy.entryCount());
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        try testing.expectEqual(@as(u32, 600), try legacy.restoreInto(&out, 0, s));
        for ([_]u32{ 0, 1, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
        // No inherited chunks and no spec sidecar: stamps v4, the version an older reader accepts.
        const meta = try tmp.dir.readFileAlloc(io, "fp-x-ssd/e1/meta.json", testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(meta);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":4") != null);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":6") == null);
    }
}

test "DiskTier: the root-wide sweep drops strays and never touches the live tier's own root" {
    // A sibling's index-less entry is a crash leftover only once it is older than
    // `STRAY_MIN_AGE_NS` (another server's flush writes meta.json last); the live tier's own
    // root is skipped entirely.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var sib = try DiskTier.init(testing.allocator, io, base, "fp-sibling", 0, 128);
        defer sib.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try sib.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    }
    try tmp.dir.createDirPath(io, "fp-stray/e9");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-stray/e9/c000000.safetensors", .data = "orphan" });
    const aged: std.Io.Timestamp = .{
        .nanoseconds = std.Io.Timestamp.now(io, .real).nanoseconds - 2 * @as(i96, @intCast(STRAY_MIN_AGE_NS)),
    };
    try tmp.dir.setTimestamps(io, "fp-stray/e9/c000000.safetensors", .{ .modify_timestamp = .{ .new = aged } });
    // Its young twin: index-less but written just now, so it must survive.
    try tmp.dir.createDirPath(io, "fp-inflight/e3");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-inflight/e3/c000000.safetensors", .data = "another server" });
    var live = try DiskTier.init(testing.allocator, io, base, "fp-live", 0, 128);
    defer live.deinit();
    live.ssd_first = true;
    live.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    // The live tier's own root, mid-write-through: chunks, no index yet. Staged after init
    // (init's own `scan` rightly drops an index-less entry).
    try tmp.dir.createDirPath(io, "fp-live/e1");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-live/e1/c000000.safetensors", .data = "inflight" });
    live.max_bytes = 1 << 40;
    sweepBase(testing.allocator, io, base, live.root, live.max_bytes);

    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-stray/e9/c000000.safetensors", .{}));
    try testing.expect(tmp.dir.statFile(io, "fp-inflight/e3/c000000.safetensors", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-sibling/e1/meta.json", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-live/e1/c000000.safetensors", .{}) catch null != null);

    // With a budget of zero the siblings' real entries go too, and the live root is still untouched.
    sweepBase(testing.allocator, io, base, live.root, 0);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-sibling/e1/meta.json", .{}));
    try testing.expect(tmp.dir.statFile(io, "fp-live/e1/c000000.safetensors", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-inflight/e3/c000000.safetensors", .{}) catch null != null);
}

/// The pre-fix materializer (one `mlx_array_eval` per tensor), kept only as the byte-identity golden below.
fn materializeLegacyPerTensorForTest(tensors: []DiskTier.NamedTensor, s: mlx.mlx_stream) !void {
    for (tensors) |*t| {
        var cont = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cont);
        try mlx.check(mlx.mlx_contiguous(&cont, t.arr, false, s));
        _ = mlx.mlx_array_free(t.arr);
        t.arr = cont;
        try mlx.check(mlx.mlx_array_eval(t.arr));
    }
}

test "DiskTier: the staged serializer evals ONCE per chunk, byte-identically to the per-tensor path" {
    // One eval per chunk file (the per-tensor eval cost ~2,300 GPU syncs on a 32k warm turn), and the bytes are unchanged.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eval", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 3, qcfg);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 256, 64, 0.0, .bfloat16);

    var a = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&a);
    var b = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&b);
    for ([_]*std.ArrayList(DiskTier.NamedTensor){ &a, &b }) |list| {
        for (cache.entries, 0..) |*e, li| {
            try testing.expect(e.initialized);
            try tier.appendSlice(list, li, "k", e.keys, 0, 128, s);
            try tier.appendSlice(list, li, "v", e.values, 0, 128, s);
            try tier.appendSlice(list, li, "ks", e.keys_scales, 0, 128, s);
            try tier.appendSlice(list, li, "kb", e.keys_biases, 0, 128, s);
            try tier.appendSlice(list, li, "vs", e.values_scales, 0, 128, s);
            try tier.appendSlice(list, li, "vb", e.values_biases, 0, 128, s);
        }
    }
    try testing.expectEqual(@as(usize, 18), a.items.len);

    const before = serialize_eval_count.load(.monotonic);
    const got = try tier.serializeSafetensors(a.items, DiskTier.no_meta, s);
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(u64, 1), serialize_eval_count.load(.monotonic) - before);

    try materializeLegacyPerTensorForTest(b.items, s);
    const mid = serialize_eval_count.load(.monotonic);
    const want = try tier.encodeSafetensors(b.items, DiskTier.no_meta);
    defer testing.allocator.free(want);
    try testing.expectEqual(mid, serialize_eval_count.load(.monotonic));
    try testing.expect(got.len > 4096);
    try testing.expectEqualSlices(u8, want, got);
}

test "DiskTier: a failing synchronous writeMeta frees the final path exactly once" {
    // `writeMeta` allocates `final_path` and forks; a function-scope `errdefer` plus the
    // synchronous branch's `defer` freed it twice on any disk write failure. The missing
    // `e<id>/` directory makes `createFileAbsolute` fail, the first fallible step past the defer.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-writemeta-err", 0, 128);
    defer tier.deinit();
    try testing.expect(tier.writer == null);

    var toks = [_]u32{ 1, 2, 3 };
    var cb = [_]u64{64};
    const e = IndexEntry{
        .id = 777, // no e777/ directory was ever created
        .tokens = &toks,
        .kv_len = 3,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 64,
        .chunk_bytes = &cb,
        .last_used = 0,
    };
    // The write must fail and leave exactly one free behind.
    if (tier.writeMeta(e)) |_| {
        return error.WriteMetaUnexpectedlySucceeded;
    } else |_| {}
}

test "DiskTier: a new entry whose meta write fails frees the media spans it duped" {
    // The new_entry errdefer must free every array the commit owns, the media dupe included.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media-metaerr", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // meta.json is a DIRECTORY before the first commit: the entry's chunks land
    // and only the final meta rename fails.
    const meta = try std.fmt.allocPrint(testing.allocator, "{s}/fp-media-metaerr/e1/meta.json", .{base});
    defer testing.allocator.free(meta);
    try std.Io.Dir.cwd().createDirPath(io, meta);

    const spans = [_]MediaSpan{ .{ .start = 100, .key = 0xA0 }, .{ .start = 300, .key = 0xB0 } };
    if (tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &spans, null, s)) |_| {
        return error.AppendCommitUnexpectedlySucceeded;
    } else |_| {}
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
}

test "DiskTier: an extend commit whose meta write fails frees the media spans it duped" {
    // The extend path shares the new_entry errdefer: the failed write frees the
    // dupe and leaves the entry it was extending untouched.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ext-media-metaerr", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const spans = [_]MediaSpan{ .{ .start = 100, .key = 0xA0 }, .{ .start = 300, .key = 0xB0 } };
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, &spans, null, s);
    try testing.expectEqual(@as(u32, 600), tier.entries.items[0].kv_len);

    // meta.json becomes a DIRECTORY: the extension's chunks land in the entry's
    // dir and only the final meta rename fails.
    const meta = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext-media-metaerr/e1/meta.json", .{base});
    defer testing.allocator.free(meta);
    std.Io.Dir.deleteFileAbsolute(io, meta) catch {};
    try std.Io.Dir.cwd().createDirPath(io, meta);

    try fillCache(&cache, s, 3, 300, 8, 4800.0, .float32);
    if (tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &spans, null, s)) |_| {
        return error.AppendCommitUnexpectedlySucceeded;
    } else |_| {}

    // The old entry stands as it was: same record, same spans, nothing committed.
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 600), tier.entries.items[0].kv_len);
    try testing.expectEqual(@as(usize, 2), tier.entries.items[0].media.len);
}

test "materializeContiguous: the fresh-handle errdefer never outlives the transfer" {
    // The fresh handle's errdefer must not stay armed across `t.arr = cont`, or a later failure
    // frees the same array twice. Transliterated onto a heap allocation so the testing
    // allocator catches both a leak and a double free.
    const t = testing;

    const Item = struct { arr: *u32 };
    const S = struct {
        fn run(a: std.mem.Allocator, items: []Item, fail_at: usize) !void {
            for (items, 0..) |*it, i| {
                const cont = try a.create(u32);
                {
                    errdefer a.destroy(cont);
                    cont.* = it.arr.* + 1;
                }
                a.destroy(it.arr);
                it.arr = cont;
                if (i == fail_at) return error.Injected;
            }
        }
    };

    var items: [4]Item = undefined;
    for (&items, 0..) |*it, i| {
        it.* = .{ .arr = try testing.allocator.create(u32) };
        it.arr.* = @intCast(i);
    }
    // The caller's `defer freeNamed(&list)`: it owns every `arr`, on every outcome.
    defer for (&items) |*it| testing.allocator.destroy(it.arr);
    try t.expectError(error.Injected, S.run(testing.allocator, &items, 2));
    try t.expectEqual(@as(u32, 1), items[0].arr.*);
    try t.expectEqual(@as(u32, 3), items[2].arr.*);
    try t.expectEqual(@as(u32, 3), items[3].arr.*);
}

// ── SSD-first chunk sharing ──

/// Two 600-token sequences that agree for the first `shared` tokens and diverge after.
const ShareToks = struct { a: [600]u32, b: [600]u32 };
fn chunkShareTokens(shared: usize) ShareToks {
    var out: ShareToks = undefined;
    for (&out.a, 0..) |*t, i| t.* = @intCast(i + 7);
    for (&out.b, 0..) |*t, i| t.* = if (i < shared) @intCast(i + 7) else @intCast(9000 + i);
    return out;
}

fn chunkStat(io: std.Io, base: []const u8, fp: []const u8, id: u64, chunk: u32) ?std.Io.File.Stat {
    var buf: [1024]u8 = undefined;
    const p = std.fmt.bufPrint(&buf, "{s}/{s}/e{d}/c{d:0>6}.safetensors", .{ base, fp, id, chunk }) catch return null;
    return statFile(io, p);
}

test "commonPrefixLen: the longest shared prefix, never past the shorter slice" {
    const a = [_]u32{ 1, 2, 3, 4 };
    const b = [_]u32{ 1, 2, 9, 4, 5 };
    try testing.expectEqual(@as(usize, 2), commonPrefixLen(&a, &b));
    try testing.expectEqual(@as(usize, 4), commonPrefixLen(&a, &a));
    try testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]u32{}));
    try testing.expectEqual(@as(usize, 3), commonPrefixLen(&a, a[0..3]));
}

test "DiskTier chunk share: a prefix-diverging entry hard-links the donor's whole chunks, writes only its tail, bills once, restores whole" {
    // Turn N+1's prompt diverges inside turn N's generated span: the heir links the whole
    // chunks below the common prefix and writes the rest.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    chunk_share_override = true;
    defer chunk_share_override = null;
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-share", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    // 600 tokens => chunks 0..3 whole (512 tokens), chunk 4 partial (88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520); // diverges at 520: 4 whole chunks shared
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const donor_bytes = tier.total_bytes;
    const donor_id = tier.entries.items[0].id;

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 4), heir.inherited_chunks);
    try testing.expectEqual(@as(usize, 5), heir.chunk_bytes.len);
    try testing.expectEqual(@as(u32, 600), heir.kv_len);
    // The leading files are one inode with two links; the tail is its own.
    const d0 = chunkStat(io, base, "fp-share", donor_id, 0).?;
    const h0 = chunkStat(io, base, "fp-share", heir.id, 0).?;
    try testing.expectEqual(d0.inode, h0.inode);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(h0.nlink)));
    const h4 = chunkStat(io, base, "fp-share", heir.id, 4).?;
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(h4.nlink)));
    try testing.expectEqual(donor_bytes + heir.chunk_bytes[4] + 600 * 4, tier.total_bytes);
    try testing.expectEqual(heir.chunk_bytes[4] + 600 * 4, heir.bytes);

    // The heir restores through a fresh tier, and its bill survives the scan.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-share", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 2), tier2.entryCount());
    try testing.expectEqual(tier.total_bytes, tier2.total_bytes);
    const m = tier2.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense, &.{}).?;
    try testing.expectEqual(@as(u32, 600), m.usable);
    var out = try KVCache.init(testing.allocator, 3);
    defer out.deinit();
    try tier2.restorePrefixInto(&out, m.idx, 600, s);
    try testing.expectEqual(@as(usize, 600), out.step);
    inline for (.{ 0, 300, 599 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, 1, pos, 3, s), try cacheValueAt(&out, 1, pos, 3, s));
    }
}

test "DiskTier chunk share: total_bytes is bytes on disk whichever holder dies first" {
    // The filesystem is the refcount: donor-then-heir and heir-then-donor both land back on the pre-commit number.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    chunk_share_override = true;
    defer chunk_share_override = null;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    for ([_]bool{ true, false }) |donor_first| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-order", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        try testing.expectEqual(@as(u64, 0), tier.total_bytes);

        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, s);
        try testing.expectEqual(@as(usize, 2), tier.entryCount());
        const donor_idx: usize = 0;
        const heir_idx: usize = 1;
        const donor_id = tier.entries.items[donor_idx].id;
        const heir_id = tier.entries.items[heir_idx].id;
        var shared_bytes: u64 = 0;
        for (tier.entries.items[heir_idx].chunk_bytes[0..4]) |b| shared_bytes += b;
        const donor_own = tier.entries.items[donor_idx].bytes - shared_bytes; // its tail + tokens
        const heir_own = tier.entries.items[heir_idx].bytes; // its tail + tokens (links billed 0)
        try testing.expectEqual(shared_bytes + donor_own + heir_own, tier.total_bytes);

        if (donor_first) {
            tier.removeAt(donor_idx);
            try testing.expectEqual(shared_bytes + heir_own, tier.total_bytes);
            try testing.expect(chunkStat(io, base, "fp-order", heir_id, 0) != null);
            try testing.expect(chunkStat(io, base, "fp-order", donor_id, 0) == null);
            try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-order", heir_id, 0).?.nlink)));
            tier.removeAt(0);
        } else {
            tier.removeAt(heir_idx);
            try testing.expectEqual(shared_bytes + donor_own, tier.total_bytes);
            try testing.expect(chunkStat(io, base, "fp-order", donor_id, 0) != null);
            try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-order", donor_id, 0).?.nlink)));
            tier.removeAt(0);
        }
        try testing.expectEqual(@as(usize, 0), tier.entryCount());
        try testing.expectEqual(@as(u64, 0), tier.total_bytes);
    }
}

test "DiskTier chunk share: the legacy arm and the kill switch never link" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    // Arm 1: legacy tier, switch on. Arm 2: SSD-first, switch off.
    for ([_]struct { ssd: bool, share: bool }{ .{ .ssd = false, .share = true }, .{ .ssd = true, .share = false } }) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        chunk_share_override = arm.share;
        defer chunk_share_override = null;
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-nolink", 0, 128);
        defer tier.deinit();
        tier.ssd_first = arm.ssd;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 4096 * 1024 * 1024 * 1024);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, s);
        const before = tier.total_bytes;
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, s);
        const heir = &tier.entries.items[1];
        try testing.expectEqual(@as(u32, 0), heir.inherited_chunks);
        try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-nolink", heir.id, 0).?.nlink)));
        var all: u64 = 600 * 4;
        for (heir.chunk_bytes) |b| all += b;
        try testing.expectEqual(before + all, tier.total_bytes);
    }
}

test "DiskTier chunk share: meta v6 carries inherited_chunks; a v5 manifest loads with none" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    chunk_share_override = true;
    defer chunk_share_override = null;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    var heir_id: u64 = 0;
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, s);
        heir_id = tier.entries.items[1].id;
    }
    var mp: [128]u8 = undefined;
    const meta_rel = try std.fmt.bufPrint(&mp, "fp-v6/e{d}/meta.json", .{heir_id});
    const meta = try tmp.dir.readFileAlloc(io, meta_rel, testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(meta);
    try testing.expect(std.mem.indexOf(u8, meta, "\"v\":6") != null);
    try testing.expect(std.mem.indexOf(u8, meta, "\"inherited_chunks\":4") != null);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        var found = false;
        for (tier.entries.items) |*e| {
            if (e.id == heir_id) {
                found = true;
                try testing.expectEqual(@as(u32, 4), e.inherited_chunks);
            }
        }
        try testing.expect(found);
    }
    // An older binary's manifest (v5, no field) still loads: nothing inherited.
    var rewritten = std.ArrayList(u8).empty;
    defer rewritten.deinit(testing.allocator);
    try rewritten.appendSlice(testing.allocator, meta);
    _ = std.mem.replace(u8, rewritten.items, "\"v\":6", "\"v\":5", rewritten.items);
    const stripped = try std.mem.replaceOwned(u8, testing.allocator, rewritten.items, "\"inherited_chunks\":4,", "");
    defer testing.allocator.free(stripped);
    try tmp.dir.writeFile(io, .{ .sub_path = meta_rel, .data = stripped });
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        for (tier.entries.items) |*e| {
            if (e.id == heir_id) try testing.expectEqual(@as(u32, 0), e.inherited_chunks);
        }
    }
}

test "DiskTier chunk share: an heir links ONLY the donor's landed chunks; the donor's queued chunks and meta drain intact" {
    // A donor mid-persist is the common case: the heir links what is on disk, writes the rest,
    // and never touches the donor's queue.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    chunk_share_override = true;
    defer chunk_share_override = null;

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-landed", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    // The bound goes on the donor's commits (the write-through hook's form), not on the tier;
    // the heir commits unbounded, the way `flushPendingDisk` completes an entry.
    const donor_bound: u64 = 1;

    // Donor: chunks 0 and 1 landed, chunk 2 queued (writer paused).
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, &.{}, s, donor_bound);
    tier.drainWriter();
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, &.{}, s, donor_bound);
    tier.drainWriter();
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, &.{}, s, donor_bound);
    try testing.expectEqual(@as(u32, 384), tier.entries.items[0].kv_len);
    const donor_id = tier.entries.items[0].id;
    try testing.expect(chunkStat(io, base, "fp-landed", donor_id, 2) == null); // still queued
    const dropped_before = tier.writer.?.files_dropped;

    // Heir: the overlap allows 4 whole chunks; only 2 have landed, so it links 2 and writes 2..4.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 2), heir.inherited_chunks);
    try testing.expectEqual(@as(usize, 5), heir.chunk_bytes.len);
    try testing.expectEqual(@as(u32, 600), heir.kv_len);

    // The donor's queue was never touched.
    try testing.expectEqual(dropped_before, tier.writer.?.files_dropped);
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 0), tier.writeErrors());
    const d2 = chunkStat(io, base, "fp-landed", donor_id, 2).?;
    try testing.expectEqual(tier.entries.items[0].chunk_bytes[2], d2.size);
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(d2.nlink))); // the heir wrote its own chunk 2
    inline for (.{ 0, 1 }) |i| try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(chunkStat(io, base, "fp-landed", heir.id, i).?.nlink)));
    inline for (.{ 2, 3, 4 }) |i| try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-landed", heir.id, i).?.nlink)));

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-landed", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 2), tier2.entryCount());
    for (tier2.entries.items) |*e| {
        if (e.id == donor_id) try testing.expectEqual(@as(u32, 384), e.kv_len);
        if (e.id == heir.id) {
            try testing.expectEqual(@as(u32, 600), e.kv_len);
            try testing.expectEqual(@as(u32, 2), e.inherited_chunks);
        }
    }
}

test "DiskTier: deinit RETURNS with the writer paused, and lands what was queued" {
    // `DiskTier.deinit` drains before it deinits the writer; teardown must lift a pause itself.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 640, 8, 0.0, .float32); // > MIN_PERSIST_TOKENS
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-paused-deinit", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    tier.writer.?.setPaused(true);
    // No deferred unpause on purpose: teardown is the release under test.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expect(tier.writer.?.pendingBytes() > 0);

    // Not deferred: the bar is that the call returns.
    tier.deinit();

    // It lifted the pause rather than skipping the drain: the entry is really on disk.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-paused-deinit", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(u32, 640), tier2.entries.items[0].kv_len);
}

test "DiskTier: a spec-only append bills the SPEC sidecar's byte delta" {
    // `appendSpecOnly` overwrote `e.spec_bytes` before taking its delta, so every commit landing
    // a spec sidecar onto a complete entry left `total_bytes` short by the sidecar (every arch).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-specbill", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Turn 1: the entry lands complete, with no spec sidecar.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entries.items.len);
    try testing.expectEqual(@as(u64, 0), tier.entries.items[0].spec_bytes);
    const bytes_before = tier.entries.items[0].bytes;
    const total_before = tier.total_bytes;

    // Turn 2: same tokens, same KV, now carrying a dflash context snap (`appendSpecOnly`).
    var dfl = try KVCache.init(testing.allocator, 1);
    defer dfl.deinit();
    try fillCache(&dfl, s, 1, 590, 8, 9.5, .float32);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{}, .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 }, s);

    const e = &tier.entries.items[0];
    try testing.expect(e.spec_bytes > 0); // the sidecar really was written
    try testing.expectEqual(bytes_before + e.spec_bytes, e.bytes);
    try testing.expectEqual(total_before + e.spec_bytes, tier.total_bytes);
    var own_chunks: u64 = 0;
    for (e.chunk_bytes[@min(e.inherited_chunks, e.chunk_bytes.len)..]) |b| own_chunks += b;
    try testing.expectEqual(nonChunkBytes(e) + own_chunks, e.bytes);

    // A rescan of the same root reaches the same total.
    var rescanned = try DiskTier.init(testing.allocator, io, base, "fp-specbill", 0, 128);
    defer rescanned.deinit();
    try testing.expectEqual(@as(usize, 1), rescanned.entryCount());
    try testing.expectEqual(tier.total_bytes, rescanned.total_bytes);
}

test "DiskTier: the manifest stamps the LOWEST version that describes the entry" {
    // The version is a compatibility claim: an older reader accepts 2..4 only, so a v6 stamp on
    // every entry made a downgrade discard the whole tier.
    const t = std.testing;

    // Plain entry: v4.
    var toks = [_]u32{ 1, 2, 3 };
    var cbytes = [_]u64{4096};
    const kv_only = SpecMeta{ .base = 0, .step = 600, .layers = 2, .quant = kv_quant.KVQuantConfig.dense };

    var e = IndexEntry{
        .id = 1,
        .tokens = &toks,
        .kv_len = 600,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 4096,
        .chunk_bytes = &cbytes,
        .last_used = 1,
    };
    try t.expectEqual(@as(u8, 4), DiskTier.metaVersionFor(e));

    // A dflash sidecar is v4 shape too.
    e.spec_bytes = 4096;
    e.spec_dflash = kv_only;
    try t.expectEqual(@as(u8, 4), DiskTier.metaVersionFor(e));

    // Inherited (hard-linked) chunks are v6: an older reader would bill and delete a donor's files.
    e.inherited_chunks = 4;
    try t.expectEqual(@as(u8, 6), DiskTier.metaVersionFor(e));
    e.spec_dflash = null;
    try t.expectEqual(@as(u8, 6), DiskTier.metaVersionFor(e));

    // v9: media always lifts the claim to 9 — an older reader must NOT run
    // token-only matching over media rows.
    var media9 = [_]MediaSpan{.{ .start = 100, .key = 0xA0 }};
    e.media = &media9;
    try t.expectEqual(@as(u8, 9), DiskTier.metaVersionFor(e));
}

// ── v9: media-keyed disk identity ──

test "DiskTier v9: media spans round-trip through meta and clamp matching per entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // The second key carries the high bit, like half of all pixel hashes: the
    // meta round-trip has to prove it survives the JSON trip.
    const spans = [_]MediaSpan{ .{ .start = 100, .key = 0xA0 }, .{ .start = 300, .key = 0x8000_0000_0000_00B0 } };
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &spans, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const dense = kv_quant.KVQuantConfig.dense;

    // Restart: the v9 meta carries the spans back, and claims v9 so a pre-v9
    // binary drops the entry instead of token-only matching past a placeholder.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-media", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqualSlices(MediaSpan, &spans, tier2.entries.items[0].media);
    {
        const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-media/e{d}/meta.json", .{ base, tier2.entries.items[0].id });
        defer testing.allocator.free(meta_path);
        const meta = readFileAlloc(testing.allocator, io, meta_path, 64 * 1024).?;
        defer testing.allocator.free(meta);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":9") != null);
        // The u64 key is written as its signed decimal (the reader's integer is
        // signed), so the needle has to be built the same way.
        const key_needle = try std.fmt.allocPrint(testing.allocator, "\"media\":[[100,160],[300,{d}]]", .{@as(i64, @bitCast(spans[1].key))});
        defer testing.allocator.free(key_needle);
        try testing.expect(std.mem.indexOf(u8, meta, key_needle) != null);
    }

    // Identical span list: full depth.
    const deep = tier2.bestMatch(&tokens, false, dense, &spans).?;
    try testing.expectEqual(@as(u32, 600), deep.usable);
    // A re-sent later item with different pixels: clamps at its start.
    try testing.expectEqual(@as(u32, 300), tier2.bestMatch(&tokens, false, dense, &.{ spans[0], .{ .start = 300, .key = 0xB1 } }).?.usable);
    // The same image at a different placeholder row: clamps at the earlier start.
    try testing.expectEqual(@as(u32, 96), tier2.bestMatch(&tokens, false, dense, &.{ .{ .start = 96, .key = 0xA0 }, spans[1] }).?.usable);
    // A re-sent list missing the second item: clamps at that item's start.
    try testing.expectEqual(@as(u32, 300), tier2.bestMatch(&tokens, false, dense, &.{spans[0]}).?.usable);
    // A new item appended below the stored ones: clamps at the new start.
    try testing.expectEqual(@as(u32, 450), tier2.bestMatch(&tokens, false, dense, &.{ spans[0], spans[1], .{ .start = 450, .key = 0xC0 } }).?.usable);
    // A text-only request: clamps at the FIRST item — the old tier-wide cap.
    try testing.expectEqual(@as(u32, 100), tier2.bestMatch(&tokens, false, dense, &.{}).?.usable);
    // The legacy wrapper keeps its meaning for older call sites.
    try testing.expectEqual(@as(u32, 100), tier2.bestMatch(&tokens, false, dense, &.{}).?.usable);

    // The deep match serves the media rows themselves: spot-check KV values
    // across both boundaries against the original cache.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    try tier2.restorePrefixInto(&cache2, deep.idx, deep.usable, s);
    for ([_]u32{ 0, 99, 100, 299, 300, 450, 599 }) |pos| {
        const want = try cacheValueAt(&cache, 1, pos, 5, s);
        const got = try cacheValueAt(&cache2, 1, pos, 5, s);
        try testing.expectEqual(want, got);
    }
}

test "DiskTier v9: a bounded write-through commit keeps its spans, so a restore past the image needs its pixels" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-wt-media", 0, 128);
    defer tier.deinit();
    // The write-through hook runs only on an SSD-first tier with its background writer.
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 700, 8, 0.0, .float32);
    var tokens: [700]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const spans = [_]MediaSpan{.{ .start = 200, .key = 0xA0 }};

    // The write-through hook's form: one chunk per call, the spans below the chunk boundary.
    for ([_]usize{ 512, 640 }) |boundary| {
        _ = try tier.appendCommitBounded(cache.entries, boundary, cache.config, tokens[0..boundary], false, &spans, s, 1);
        tier.drainWriter();
    }
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-wt-media", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqualSlices(MediaSpan, &spans, tier2.entries.items[0].media);
    const kv = tier2.entries.items[0].kv_len;
    try testing.expect(kv > 200);
    const dense = kv_quant.KVQuantConfig.dense;
    try testing.expectEqual(kv, tier2.bestMatch(&tokens, false, dense, &spans).?.usable);
    try testing.expectEqual(@as(u32, 200), tier2.bestMatch(&tokens, false, dense, &.{.{ .start = 200, .key = 0xB1 }}).?.usable);
}

test "DiskTier v9: supersede and extend are media-gated; chunk share clamps at divergence" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media-sg", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    // 768 rows: the 768-token records below need their KV actually present,
    // or every entry truncates to the 640 rows the live cache holds.
    try fillCache(&cache, s, 3, 768, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const span_a = MediaSpan{ .start = 512, .key = 0xAA };
    const span_b = MediaSpan{ .start = 512, .key = 0xBB };
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{span_a}, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // The identical token record under different pixels is a DIFFERENT entry:
    // supersede must not no-op the commit (its rows describe another image).
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{span_b}, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    // Under the same span list it is superseded again: still 2 entries.
    const sup = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{span_a}, null, s);
    try testing.expectEqual(PersistOutcome.persisted, sup);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());

    // A continuation under A grows the A record in place (extend, media gate
    // open) — never a second entry.
    var cont_a: [768]u32 = undefined;
    for (&cont_a, 0..) |*t, i| t.* = if (i < 640) tokens[i] else @intCast(50_000 + i);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &cont_a, false, &.{span_a}, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    // Chunk share, same span list: a twin commit diverging INSIDE the tail at
    // 740 hard-links all 5 whole chunks below the token divergence.
    var twin_a: [768]u32 = cont_a;
    twin_a[740] = 555_001;
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &twin_a, false, &.{span_a}, null, s);
    // The same tokens under B's pixels: rows 512.. up describe a different
    // image, so the share clamps to the 4 whole chunks below the span start.
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &twin_a, false, &.{span_b}, null, s);
    var inherited_a: ?u32 = null;
    var inherited_b: ?u32 = null;
    for (tier.entries.items) |*e| {
        if (e.tokens.len != 768) continue;
        if (e.media.len != 1) continue;
        if (e.tokens[740] == 555_001 and e.media[0].key == 0xAA) inherited_a = e.inherited_chunks;
        if (e.tokens[740] == 555_001 and e.media[0].key == 0xBB) inherited_b = e.inherited_chunks;
    }
    try testing.expectEqual(@as(?u32, 5), inherited_a);
    try testing.expectEqual(@as(?u32, 4), inherited_b);

    // Extend gate (fresh root): a commit that grows a B record must extend
    // the B entry, never the token-prefix-matching A entry…
    var tier3 = try DiskTier.init(testing.allocator, io, base, "fp-media-ex", 0, 128);
    defer tier3.deinit();
    _ = try tier3.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{span_b}, null, s);
    var ext: [700]u32 = undefined;
    for (&ext, 0..) |*t, i| t.* = if (i < 640) tokens[i] else @intCast(70_000 + i);
    // …the A entry shares the 640-token prefix but its span list diverges at
    // 512, so this media-B commit under span A must NOT extend it either:
    // it lands as a second entry.
    _ = try tier3.appendCommitWithSpec(cache.entries, cache.step, cache.config, &ext, false, &.{span_a}, null, s);
    try testing.expectEqual(@as(usize, 2), tier3.entryCount());
    // The same growth under span B now extends the B entry (media gate open).
    var ext_b: [760]u32 = undefined;
    for (&ext_b, 0..) |*t, i| t.* = if (i < 640) tokens[i] else @intCast(71_000 + i);
    _ = try tier3.appendCommitWithSpec(cache.entries, cache.step, cache.config, &ext_b, false, &.{span_b}, null, s);
    try testing.expectEqual(@as(usize, 2), tier3.entryCount());
    var grew = false;
    for (tier3.entries.items) |*e| {
        if (e.tokens.len == 760 and e.media.len == 1 and e.media[0].key == 0xBB) grew = true;
    }
    try testing.expect(grew);
}

test "DiskTier v9 scan: a media list on a pre-v9 stamp, or a v9 stamp without one, drops the entry" {
    // The wrong-restore class, guarded at load: a media record whose spans
    // cannot be read must never be matched by tokens alone.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media-meta", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const spans = [_]MediaSpan{.{ .start = 100, .key = 0xA0 }};
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &spans, null, s);
    const id = tier.entries.items[0].id;
    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-media-meta/e{d}/meta.json", .{ base, id });
    defer testing.allocator.free(meta_path);
    const orig = readFileAlloc(testing.allocator, io, meta_path, 64 * 1024).?;
    defer testing.allocator.free(orig);
    // Sanity: the entry as written loads.
    if (tier.loadEntry(id)) |l| {
        var e = l.e;
        tier.freeIndexEntryOwned(&e);
        try testing.expectEqual(@as(usize, 1), e.media.len);
    } else return error.TestUnexpectedResult;
    const cases = [_]struct { find: []const u8, replace: []const u8 }{
        // A media list under a pre-v9 stamp is a lie: a v8 reader would
        // token-only match the media rows. Reject the whole entry.
        .{ .find = "\"v\":9", .replace = "\"v\":8" },
        // v9 without the spans it claims: same failure mode, same verdict.
        .{ .find = ",\"media\":[[100,160]]", .replace = "" },
        // Malformed span shapes never load as text-only either.
        .{ .find = "[[100,160]", .replace = "[[100]]" },
        .{ .find = "[[100,160]", .replace = "[[600,160]]" }, // span at/after the token record
    };
    for (cases) |c| {
        const patched = try std.mem.replaceOwned(u8, testing.allocator, orig, c.find, c.replace);
        defer testing.allocator.free(patched);
        try testing.expect(!std.mem.eql(u8, patched, orig));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = meta_path, .data = patched });
        try testing.expect(tier.loadEntry(id) == null);
    }
}

test "DiskTier: a failed background write INVALIDATES the entry it belonged to (no completion claim, restore misses)" {
    // The writer drops a failed blob, and `writeMeta` appends the index before a byte reaches
    // disk, so the RAM copy was evicted against a hole. The failed path names the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-poison", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Hold the writer so the failure lands strictly after the commit.
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    tier.writer.?.injectFailure("c000003", .write);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    const dead_id = tier.entries.items[0].id;
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expect(tier.writeErrors() > 0);
    // meta.json rode the same FIFO and did land: the index still describes five whole chunks.
    try testing.expectEqual(@as(usize, 5), tier.entries.items[0].chunk_bytes.len);

    try testing.expectEqual(@as(usize, 1), tier.harvestWriteFailures());
    try testing.expect(tier.entries.items[0].poisoned);
    for (tier.entries.items[0].chunk_bytes) |b| try testing.expectEqual(@as(u64, 0), b);
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));
    try testing.expect(tier.fullPrefixEntryId(cache.entries, cache.step, &tokens, false, cache.config, &.{}) == null);
    try testing.expect(tier.bestMatch(&tokens, false, cache.config, &.{}) == null);
    try testing.expect(!tier.entryWholeOnDisk(dead_id));
    // Attribution happens once.
    try testing.expectEqual(@as(usize, 0), tier.harvestWriteFailures());

    // The next commit reclaims the dead directory and rebuilds from scratch.
    tier.writer.?.injectFailure(null, .write);
    const out = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(PersistOutcome.persisted, out);
    tier.drainWriter();
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const live = &tier.entries.items[0];
    try testing.expect(live.id != dead_id);
    try testing.expect(!live.poisoned);
    try testing.expect(tier.entryWholeOnDisk(live.id));
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config, &.{}));

    var back = try KVCache.init(testing.allocator, 2);
    defer back.deinit();
    try testing.expectEqual(@as(u32, 600), try tier.restoreInto(&back, 0, s));
}

test "DiskTier: a manifest scalar that does not fit its field drops the record, never casts" {
    // meta.json is on disk and hand-editable; an unchecked @intCast of a corrupt scalar is
    // ReleaseFast UB.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-badmeta", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    var dfl = try KVCache.init(testing.allocator, 1);
    defer dfl.deinit();
    try fillCache(&dfl, s, 1, 520, 8, 9.5, .float32);
    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, &.{}, .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 }, s);
    const id = tier.entries.items[0].id;
    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-badmeta/e{d}/meta.json", .{ base, id });
    defer testing.allocator.free(meta_path);
    const orig = readFileAlloc(testing.allocator, io, meta_path, 64 * 1024).?;
    defer testing.allocator.free(orig);
    if (tier.loadEntry(id)) |l| {
        var e = l.e;
        defer tier.freeIndexEntryOwned(&e);
        try testing.expect(e.spec_dflash != null);
    } else return error.TestUnexpectedResult;

    const cases = [_]struct { find: []const u8, replace: []const u8, entry_survives: bool }{
        .{ .find = "\"bits\":4", .replace = "\"bits\":4000", .entry_survives = false },
        .{ .find = "\"group_size\":64", .replace = "\"group_size\":68719476736", .entry_survives = false },
        .{ .find = "\"layers\":1", .replace = "\"layers\":68719476736", .entry_survives = true },
    };
    for (cases) |c| {
        const patched = try std.mem.replaceOwned(u8, testing.allocator, orig, c.find, c.replace);
        defer testing.allocator.free(patched);
        try testing.expect(!std.mem.eql(u8, patched, orig));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = meta_path, .data = patched });
        if (tier.loadEntry(id)) |l| {
            var e = l.e;
            defer tier.freeIndexEntryOwned(&e);
            try testing.expect(c.entry_survives);
            // The spec half is the only part an out-of-range spec scalar costs.
            try testing.expect(e.spec_dflash == null);
        } else {
            try testing.expect(!c.entry_survives);
        }
    }
}

test "DiskTier: entryWholeOnDisk stats what the index NAMES (a truncated chunk fails it with a clean writer)" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-stat", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    const id = tier.entries.items[0].id;
    try testing.expect(tier.entryWholeOnDisk(id));

    // Truncate one chunk behind the tier's back.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-stat/e1/c000002.safetensors", .data = "short" });
    try testing.expect(!tier.entryWholeOnDisk(id));
    try tmp.dir.deleteFile(io, "fp-stat/e1/c000002.safetensors");
    try testing.expect(!tier.entryWholeOnDisk(id));
    try testing.expect(!tier.entryWholeOnDisk(id + 999));
}

test "DiskTier: a restore wider than the fd limit closes each chunk as it goes" {
    // Bar: restoring more chunk files than the soft RLIMIT_NOFILE allows open at once succeeds.
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    // 600 tokens at 8 per chunk = 75 chunk files.
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-fdlimit", 0, 8);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, &.{}).?;
    tier.drainEntry(tier.entries.items[m.idx].id);

    const saved = try std.posix.getrlimit(.NOFILE);
    defer std.posix.setrlimit(.NOFILE, saved) catch {};
    var top: usize = 0;
    for (0..@min(saved.cur, 4096)) |fd| {
        if (std.c.fcntl(@intCast(fd), std.c.F.GETFD) != -1) top = fd;
    }
    try std.posix.setrlimit(.NOFILE, .{ .cur = top + 1 + 16, .max = saved.max });

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const restored = tier.restoreInto(&cache2, m.idx, s);
    std.posix.setrlimit(.NOFILE, saved) catch {};
    try testing.expectEqual(@as(u32, 600), try restored);
    for ([_]u32{ 0, 7, 8, 333, 599 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, 2, pos, 5, s), try cacheValueAt(&cache2, 2, pos, 5, s));
    }
}

test "DiskTier: a failed restore drops the latch it raised and keeps a foreign one" {
    // Bar: the caller's cold-prefill fallback must not inherit this restore's MLX error.
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-restore-latch", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 13);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, &.{}).?;
    const id = tier.entries.items[m.idx].id;
    tier.drainEntry(id);
    var path_buf: [1024]u8 = undefined;
    const chunk = try std.fmt.bufPrint(&path_buf, "{s}/e{d}/c000002.safetensors", .{ tier.root, id });
    try std.Io.Dir.deleteFileAbsolute(io, chunk);

    var target = try KVCache.init(testing.allocator, 2);
    defer target.deinit();
    try testing.expectError(error.MlxError, tier.restoreInto(&target, m.idx, s));
    try testing.expect(!mlx.errorPending());

    mlx.latchErrorForTest("foreign pre-existing error");
    _ = tier.restoreInto(&target, m.idx, s) catch {};
    var msg: [512]u8 = undefined;
    try testing.expectEqualStrings("foreign pre-existing error", mlx.takeError(&msg).?);
}

test "mediaSharedBound: a repeated pixel key in a superset does not collapse the bound" {
    // A pixel key may legally repeat across spans: the comparison is
    // positional, so it must walk past the reuse and clamp only where a
    // start/key pair actually differs.
    const K1: u64 = 8_376_670_522_757_638_769;
    const K2: u64 = 0xB2;
    const a = [_]MediaSpan{ .{ .start = 100, .key = K1 }, .{ .start = 300, .key = K2 } };
    const b = [_]MediaSpan{ .{ .start = 100, .key = K1 }, .{ .start = 300, .key = K2 }, .{ .start = 500, .key = K1 } };

    // The superset's extra item REUSES an earlier key: the bound is that item's
    // start, not a collapse back to the first K1 occurrence.
    try testing.expectEqual(@as(usize, 500), mediaSharedBound(&a, &b));
    try testing.expectEqual(@as(usize, 500), mediaSharedBound(&b, &a));
    // Identical lists agree everywhere.
    try testing.expectEqual(std.math.maxInt(usize), mediaSharedBound(&a, &a));
    try testing.expectEqual(std.math.maxInt(usize), mediaSharedBound(&b, &b));
    // A repeat that sits behind a real divergence still clamps at the divergence.
    const c = [_]MediaSpan{ .{ .start = 100, .key = K1 }, .{ .start = 300, .key = 0xB9 }, .{ .start = 500, .key = K1 } };
    try testing.expectEqual(@as(usize, 300), mediaSharedBound(&a, &c));
    // A text turn still clamps at the first item.
    try testing.expectEqual(@as(usize, 100), mediaSharedBound(&a, &.{}));
}
