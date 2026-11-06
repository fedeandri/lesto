//! Plan 05 — model discovery (Phase 1 minimal).
//!
//! Walks a directory looking for subdirectories that contain `config.json`,
//! treats each as a discoverable model. Used by `--model-dir` to enumerate
//! candidate models for `/v1/models` listing without loading them.
//!
//! v1 scope: discovery + listing only. Request routing still goes to the
//! single `--model` loaded at startup; if the user passes `--model-dir`
//! WITHOUT `--model`, we pick the first discovered model as the loaded one
//! and surface the rest as `loaded:false` siblings.
//!
//! On-demand load + LRU eviction live in plan 05 phases 2-5 and depend on
//! plan 01 Phase 0 (detangling Transformer state).

const std = @import("std");
const log = @import("log.zig");
// Only the pure JSON contract predicate is referenced — lazy analysis keeps
// dflash.zig's mlx FFI out of this filesystem-only module.
const dflash = @import("dflash.zig");

/// Architecture allow-list for discovery. Must stay in sync with the
/// `model_type` branches in `model.zig:parseConfigFromJson`. Discovery
/// silently skips any subdirectory whose `config.json` declares a
/// `model_type` outside this list — that prevents `--model-dir` from
/// picking up partially-downloaded or unsupported checkpoints (e.g. a
/// directory of an unsupported arch next to gemma ones) which would otherwise
/// crash the server when the tokenizer for the unknown arch is loaded.
///
/// `gemma4_assistant` is deliberately excluded: those are speculative-
/// decoding drafters, not standalone primary models. Bare `gemma4`
/// drafters can't decode on their own, and users shouldn't see them in
/// `/v1/models`.
const supported_model_types = [_][]const u8{
    "gemma4",           "gemma4_text",
    "gemma4_unified",   "gemma4_unified_text",
};

fn isSupportedModelType(model_type: []const u8) bool {
    for (supported_model_types) |t| {
        if (std.mem.eql(u8, model_type, t)) return true;
    }
    return false;
}

/// Quantization modes the MLX loader supports. Must stay in sync with
/// `model.zig:QuantMode` (discovery deliberately avoids importing model.zig,
/// which would drag the mlx FFI into this filesystem-only module).
const supported_quant_modes = [_][]const u8{ "affine", "nvfp4", "mxfp4", "mxfp8" };

fn isSupportedQuantMode(mode: []const u8) bool {
    for (supported_quant_modes) |m| {
        if (std.mem.eql(u8, mode, m)) return true;
    }
    return false;
}

/// Outcome of reading a candidate's config.json. Discovery treats any
/// non-`.supported` result as "skip this directory."
const ConfigPeek = union(enum) {
    supported: []const u8, // owned dupe of model_type
    unsupported_arch: []const u8, // owned dupe of model_type
    unsupported_quant: []const u8, // owned dupe of quantization.mode
    /// Declares the DFlash config contract — a spec-decode sidecar whatever
    /// its `model_type` says (DFlash2 ships a bare chat model_type with no embed
    /// weights; registering it as chat dies at cold load).
    drafter,
    missing_or_unparseable,
};

/// Peek at a candidate's `config.json`: classify by `model_type` and
/// `quantization.mode`. Discovery uses this to filter out:
///   - unsupported archs (any model_type outside the allow-list, which would crash the tokenizer)
///   - unsupported quantization modes (anything outside
///     `supported_quant_modes` — affine, nvfp4, mxfp4, mxfp8).
///
/// Returned strings are owned by `allocator`; the caller frees them via
/// the helpers in `freeConfigPeek`.
fn peekConfig(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, entry_name: []const u8) ConfigPeek {
    var sub = dir.openDir(io, entry_name, .{}) catch return .missing_or_unparseable;
    defer sub.close(io);
    var file = sub.openFile(io, "config.json", .{}) catch return .missing_or_unparseable;
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return .missing_or_unparseable;
    defer allocator.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return .missing_or_unparseable;
    defer parsed.deinit();
    if (parsed.value != .object) return .missing_or_unparseable;
    const root = parsed.value.object;
    // The DFlash contract outranks model_type: v1 assistants at least carry a
    // `*_assistant` suffix, but a DFlash2 sidecar is config-indistinguishable
    // from its trunk family without this probe (one predicate, shared with
    // the loader's own detection).
    if (dflash.isDflashConfigJson(root)) return .drafter;
    const mt_val = root.get("model_type") orelse return .missing_or_unparseable;
    if (mt_val != .string) return .missing_or_unparseable;
    if (!isSupportedModelType(mt_val.string)) {
        const dup = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable;
        return .{ .unsupported_arch = dup };
    }
    // Quantization gate: if a model declares a `quantization.mode`, accept
    // only the schemes the loader supports. Models without a quantization
    // block (bf16 / unquantized) pass through.
    if (root.get("quantization")) |q_val| {
        if (q_val == .object) {
            if (q_val.object.get("mode")) |mode_val| {
                if (mode_val == .string and !isSupportedQuantMode(mode_val.string)) {
                    const dup = allocator.dupe(u8, mode_val.string) catch return .missing_or_unparseable;
                    return .{ .unsupported_quant = dup };
                }
            }
        }
    }
    return .{ .supported = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable };
}

/// The set of shard basenames in `model.safetensors.index.json`'s `weight_map`,
/// or null when there is no usable index (single-file packs, media packs).
pub fn indexShardSet(io: std.Io, dir: std.Io.Dir) ?std.StringHashMapUnmanaged(void) {
    const a = std.heap.page_allocator;
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 * 1024 * 1024)) catch return null;
    defer a.free(raw);
    var parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const wm = parsed.value.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| {
        if (v != .string) continue;
        if (set.contains(v.string)) continue;
        const key = a.dupe(u8, v.string) catch continue;
        set.put(a, key, {}) catch {
            a.free(key);
            continue;
        };
    }
    // An index none of whose shards exist is stale (the repo was re-sharded
    // after this index was written); the directory is then the set.
    var any_present = false;
    var keys = set.keyIterator();
    while (keys.next()) |k| {
        _ = dir.statFile(io, k.*, .{}) catch continue;
        any_present = true;
        break;
    }
    if (!any_present) {
        log.warn("model.safetensors.index.json names no shard in this directory; loading every *.safetensors instead\n", .{});
        freeShardSet(&set);
        return null;
    }
    return set;
}

/// Free a set from `indexShardSet`.
pub fn freeShardSet(set: *std.StringHashMapUnmanaged(void)) void {
    var keys = set.keyIterator();
    while (keys.next()) |k| std.heap.page_allocator.free(k.*);
    set.deinit(std.heap.page_allocator);
}

/// Coarse classification of a model for UX surfaces, such as a model listing's
/// type column. Mirrors how serving
/// actually routes the directory (drafter sidecars).
pub const ModelKind = enum {
    chat,
    drafter,
    unsupported,

    /// Short label for the `list` TYPE column.
    pub fn label(self: ModelKind) []const u8 {
        return switch (self) {
            .chat => "chat",
            .drafter => "drafter",
            .unsupported => "unsupported",
        };
    }

    /// Human phrase for refusal messages ("'X' is <describe>").
    pub fn describe(self: ModelKind) []const u8 {
        return switch (self) {
            .chat => "a chat model",
            .drafter => "a speculative-decoding drafter sidecar, not a standalone model (load it via --drafter beside a Gemma 4 target)",
            .unsupported => "an architecture lesto does not support",
        };
    }

    /// The generation endpoint that DOES serve this kind, when one exists.
    pub fn genEndpoint(self: ModelKind) ?[]const u8 {
        _ = self;
        return null;
    }
};

/// Map a config.json `model_type` to its ModelKind.
pub fn modelKindFromType(model_type: []const u8) ModelKind {
    if (std.mem.endsWith(u8, model_type, "_assistant")) return .drafter;
    if (isSupportedModelType(model_type)) return .chat;
    return .unsupported;
}

/// Classify an ABSOLUTE model dir by its config.json model_type;
/// null when the dir is not model-shaped at all.
pub fn classifyModelPath(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) ?ModelKind {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return null;
    if (base.len == 0 or parent.len == 0) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return null;
    defer dir.close(io);
    return switch (peekConfig(io, allocator, dir, base)) {
        .missing_or_unparseable => null,
        // Raw model_type still classifies the KIND even when serving would
        // skip it (drafters, vit, ...) — that's exactly what the label is for.
        .unsupported_arch => |mt| blk: {
            defer allocator.free(mt);
            break :blk modelKindFromType(mt);
        },
        .unsupported_quant => |mode| blk: {
            allocator.free(mode);
            break :blk .unsupported;
        },
        .drafter => .drafter,
        .supported => |mt| blk: {
            defer allocator.free(mt);
            break :blk modelKindFromType(mt);
        },
    };
}

pub const DiscoveredModel = struct {
    /// Model id (subdirectory basename, e.g. "gemma-4-e4b-it-4bit").
    id: []const u8,
    /// Absolute path to the model directory.
    path: []const u8,
    /// Approximate weight size on disk in bytes (sum of *.safetensors). Used
    /// later by eviction; null if scan failed.
    bytes_on_disk: ?u64,
    /// `model_type` peeked from config.json, so registry stubs
    /// can advertise arch-derived capabilities before a cold load. Empty
    /// when unknown.
    model_type: []const u8 = "",
};

pub const DiscoveryResult = struct {
    models: []DiscoveredModel,
    allocator: std.mem.Allocator,
    /// The roots this result was scanned from (owned dupes; set by
    /// `discoverModelsMany`). Kept so `ModelRegistry.rescan` can re-walk
    /// them at runtime — models downloaded after boot are invisible to a
    /// boot-only scan.
    roots: []const []const u8 = &.{},

    pub fn deinit(self: *DiscoveryResult) void {
        for (self.models) |*m| {
            self.allocator.free(m.id);
            self.allocator.free(m.path);
            if (m.model_type.len > 0) self.allocator.free(m.model_type);
        }
        self.allocator.free(self.models);
        if (self.roots.len > 0) {
            for (self.roots) |r| self.allocator.free(r);
            self.allocator.free(self.roots);
        }
    }
};

/// Scan `model_dir` for subdirectories containing `config.json`.
/// Returns DiscoveryResult; caller owns memory via deinit().
/// Symlinks followed; permission errors on individual subdirs skipped silently.
pub fn discoverModels(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !DiscoveryResult {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch |err| {
        return err;
    };
    defer dir.close(io);
    return discoverModelsInDir(io, allocator, dir, model_dir);
}

/// Scan several roots and merge them into ONE result, first-root-wins on a
/// repeated id.
///
/// De-dup is load-bearing, not tidiness: `registerStubWithArch` answers
/// `error.DuplicateId` and `ModelRegistry.registerDiscovered` does `try`, so
/// two roots holding the same `org/name` would fail the whole registry init —
/// the server would not start. First wins because the caller orders the roots
/// and the first is where downloads land, so a stale copy on a second disk must
/// never shadow the live one.
///
/// A root that cannot be opened is SKIPPED with a warning rather than failing:
/// the second folder can live on an external drive, and unplugging it must not
/// stop the server serving everything else.
pub fn discoverModelsMany(io: std.Io, allocator: std.mem.Allocator, roots: []const []const u8) !DiscoveryResult {
    // Dupe the roots first so the result can carry them (see
    // `DiscoveryResult.roots`).
    var roots_owned: []const []const u8 = &.{};
    if (roots.len > 0) {
        const dupes = try allocator.alloc([]const u8, roots.len);
        var done: usize = 0;
        errdefer {
            for (dupes[0..done]) |r| allocator.free(r);
            allocator.free(dupes);
        }
        for (roots) |r| {
            dupes[done] = try allocator.dupe(u8, r);
            done += 1;
        }
        roots_owned = dupes;
    }
    errdefer if (roots_owned.len > 0) {
        for (roots_owned) |r| allocator.free(r);
        allocator.free(roots_owned);
    };

    var merged = std.ArrayList(DiscoveredModel).empty;
    errdefer {
        for (merged.items) |*m| {
            allocator.free(m.id);
            allocator.free(m.path);
            if (m.model_type.len > 0) allocator.free(m.model_type);
        }
        merged.deinit(allocator);
    }

    for (roots) |root| {
        const one = discoverModels(io, allocator, root) catch |err| {
            log.warn("--model-dir scan failed ({s}): {s}\n", .{ root, @errorName(err) });
            continue;
        };
        // Transfer ownership per model, freeing only the ones we drop — the
        // whole result's `deinit` would free the strings we just handed over.
        defer allocator.free(one.models);
        for (one.models) |m| {
            var dup = false;
            for (merged.items) |seen| {
                if (std.mem.eql(u8, seen.id, m.id)) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                log.warn("[discovery] {s}: already found in an earlier --model-dir, skipping {s}\n", .{ m.id, m.path });
                allocator.free(m.id);
                allocator.free(m.path);
                if (m.model_type.len > 0) allocator.free(m.model_type);
                continue;
            }
            try merged.append(allocator, m);
        }
    }

    return .{ .models = try merged.toOwnedSlice(allocator), .allocator = allocator, .roots = roots_owned };
}

/// Core scan over an already-open root. Two layouts are recognized:
///   <root>/<model>/config.json               → id "<model>"
///   <root>/<org>/<model>/config.json         → id "<org>/<model>"
/// The second is the HF-style layout `~/.mlx-serve/models` uses (the app's
/// DownloadManager writes there).
pub fn discoverModelsInDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8) !DiscoveryResult {
    var found = std.ArrayList(DiscoveredModel).empty;
    errdefer {
        for (found.items) |*m| {
            allocator.free(m.id);
            allocator.free(m.path);
            if (m.model_type.len > 0) allocator.free(m.model_type);
        }
        found.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .directory and entry.kind != .sym_link) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;

        const is_model_dir = try tryAddModel(io, allocator, dir, entry.name, "", model_dir, &found);
        if (is_model_dir) continue;

        // No config.json at this level — maybe an org dir (org/repo layout).
        var org = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer org.close(io);
        var org_iter = org.iterate();
        while (org_iter.next(io) catch null) |org_entry| {
            if (org_entry.kind != .directory and org_entry.kind != .sym_link) continue;
            if (org_entry.name.len == 0 or org_entry.name[0] == '.') continue;
            _ = try tryAddModel(io, allocator, org, org_entry.name, entry.name, model_dir, &found);
        }
    }

    // Stable order: by id ascending, so listing is deterministic.
    std.sort.pdq(DiscoveredModel, found.items, {}, lessThanById);

    return .{
        .models = try found.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// Inspect `parent/<name>` as a model-dir candidate; append to `found` if
/// it holds a supported config.json. Returns true when
/// the dir was model-shaped (config.json present, even
/// if unsupported), so callers know not to descend into it looking for an
/// org layout.
fn tryAddModel(
    io: std.Io,
    allocator: std.mem.Allocator,
    parent: std.Io.Dir,
    name: []const u8,
    id_prefix: []const u8,
    model_dir: []const u8,
    found: *std.ArrayList(DiscoveredModel),
) !bool {
    var sub = parent.openDir(io, name, .{ .iterate = true }) catch return false;
    defer sub.close(io);

    var bytes: u64 = 0;
    var bytes_ok = false;

    const model_type: []const u8 = blk: {
        // A directory without a root config.json is not a model.
        const has_config = if (sub.statFile(io, "config.json", .{})) |st| st.kind == .file else |_| false;
        if (!has_config) return false;

        // Filter by supported model_type AND quantization scheme. Catches:
        //   - partially-downloaded checkpoints (missing/garbage config)
        //   - unsupported arches (any model_type outside the allow-list)
        //   - unsupported quants (modes outside supported_quant_modes)
        // before they reach the tokenizer/weight loaders.
        break :blk switch (peekConfig(io, allocator, parent, name)) {
            .missing_or_unparseable => {
                log.info("[discovery] skip {s}: config.json missing or unparseable", .{name});
                return true;
            },
            .unsupported_arch => |mt| {
                defer allocator.free(mt);
                log.info("[discovery] skip {s}: unsupported model_type '{s}'", .{ name, mt });
                return true;
            },
            .unsupported_quant => |mode| {
                defer allocator.free(mode);
                log.info("[discovery] skip {s}: unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)", .{ name, mode });
                return true;
            },
            .drafter => {
                log.info("[discovery] skip {s}: DFlash drafter sidecar, not a standalone model", .{name});
                return true;
            },
            .supported => |mt| mt, // ownership moves to the DiscoveredModel
        };
    };
    errdefer if (model_type.len > 0) allocator.free(model_type);

    // Compute weight bytes (sum of *.safetensors sizes) — best-effort.
    if (!bytes_ok) {
        var sub_iter_dir = parent.openDir(io, name, .{ .iterate = true }) catch null;
        if (sub_iter_dir) |*sd| {
            defer sd.close(io);
            var referenced = indexShardSet(io, sd.*);
            defer if (referenced) |*r| freeShardSet(r);
            var sd_iter = sd.iterate();
            while (sd_iter.next(io) catch null) |sub_entry| {
                if (sub_entry.kind != .file and sub_entry.kind != .sym_link) continue;
                if (!std.mem.endsWith(u8, sub_entry.name, ".safetensors")) continue;
                if (referenced) |r| if (!r.contains(sub_entry.name)) continue;
                const st = sd.statFile(io, sub_entry.name, .{}) catch continue;
                if (st.kind != .file) continue;
                bytes += @intCast(st.size);
                bytes_ok = true;
            }
        }
        // A diffusers-shaped repo (MageFlow) holds NO weights at its root —
        // they live one level down in transformer/, text_encoder/, vae/. Only
        // descend when the flat scan came up empty, so the common MLX layout
        // pays nothing.
        if (!bytes_ok) bytes_ok = sumComponentWeights(io, parent, name, &bytes);
    }

    const id = if (id_prefix.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ id_prefix, name })
    else
        try allocator.dupe(u8, name);
    errdefer allocator.free(id);
    const path = if (id_prefix.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ trimTrailingSlash(model_dir), id_prefix, name })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ trimTrailingSlash(model_dir), name });
    try found.append(allocator, .{
        .id = id,
        .path = path,
        .bytes_on_disk = if (bytes_ok) bytes else null,
        .model_type = model_type,
    });
    return true;
}

/// Sum `*.safetensors` one level below `parent/name` (the component subdirs of
/// a diffusers repo). Returns true when anything was found. Best-effort: any
/// unreadable dir is skipped, exactly like the flat scan above.
fn sumComponentWeights(io: std.Io, parent: std.Io.Dir, name: []const u8, bytes: *u64) bool {
    var top = parent.openDir(io, name, .{ .iterate = true }) catch return false;
    defer top.close(io);
    var found_any = false;
    var it = top.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .directory) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        var comp = top.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer comp.close(io);
        var cit = comp.iterate();
        while (cit.next(io) catch null) |f| {
            if (f.kind != .file and f.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, f.name, ".safetensors")) continue;
            const st = comp.statFile(io, f.name, .{}) catch continue;
            if (st.kind != .file) continue;
            bytes.* += @intCast(st.size);
            found_any = true;
        }
    }
    return found_any;
}

fn trimTrailingSlash(s: []const u8) []const u8 {
    var p = s;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    return p;
}

pub const ProbeResult = struct {
    /// Owned dupe of the supported model_type.
    model_type: []const u8,
    /// Sum of *.safetensors bytes; null if the scan failed.
    bytes_on_disk: ?u64,
};

/// Validate an arbitrary absolute model directory the way discovery would
/// (config.json present, supported model_type and quant mode) and report its
/// weight bytes. Used by the register-by-path branch for models
/// OUTSIDE the --model-dir scan — e.g. the app's auto-downloaded embedding
/// encoder, which lands wherever the download root is regardless of which
/// org dir the chat model came from.
pub fn probeModelDir(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) !ProbeResult {
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return error.InvalidModelPath;
    if (base.len == 0 or parent.len == 0) return error.InvalidModelPath;

    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return error.ModelDirNotFound;
    defer dir.close(io);

    const model_type: []const u8 = switch (peekConfig(io, allocator, dir, base)) {
        .missing_or_unparseable => return error.ModelDirNotFound,
        .unsupported_arch => |mt| {
            allocator.free(mt);
            return error.UnsupportedArch;
        },
        .unsupported_quant => |mode| {
            allocator.free(mode);
            return error.UnsupportedQuantMode;
        },
        .drafter => return error.UnsupportedArch,
        .supported => |mt| mt,
    };
    errdefer allocator.free(model_type);

    var bytes: u64 = 0;
    var bytes_ok = false;
    var sub = dir.openDir(io, base, .{ .iterate = true }) catch null;
    if (sub) |*sd| {
        defer sd.close(io);
        var it = sd.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file and entry.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            const st = sd.statFile(io, entry.name, .{}) catch continue;
            if (st.kind != .file) continue;
            bytes += @intCast(st.size);
            bytes_ok = true;
        }
    }
    return .{ .model_type = model_type, .bytes_on_disk = if (bytes_ok) bytes else null };
}

/// Metadata an `unloaded` stub can advertise via `/v1/models` without faulting
/// in weights — all sourced from `config.json` (+ chat-template presence). Lets
/// clients see context window, dims, MoE-ness, and capabilities (tools/vision)
/// before a cold load. `found=false` means config.json couldn't be read/parsed.
pub const StubMeta = struct {
    found: bool = false,
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    max_position_embeddings: u32 = 0,
    quant_bits: u32 = 0,
    is_moe: bool = false,
    has_vision: bool = false,
    /// Video input: a `video_token_id` alongside `has_vision`
    /// (video piggybacks the vision tower).
    has_video: bool = false,
    has_chat: bool = false,
    /// The chat template can open a reasoning block (`templateSupportsThinking`).
    has_thinking: bool = false,
    /// Embedding capability: a decoder with a declared pooling contract
    /// (config.json `pooling_mode`, or, added by `readStubMeta`, a
    /// `1_Pooling/config.json` sidecar).
    has_embedding: bool = false,
};

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) u32 {
    if (obj.get(key)) |v| {
        if (v == .integer and v.integer > 0) return std.math.cast(u32, v.integer) orelse 0;
    }
    return 0;
}

/// Pure: extract `StubMeta` from raw config.json bytes. `has_chat_template` is
/// supplied by the caller (the template lives outside config.json), and
/// determines chat/tool capabilities. Mirrors the loaded-path capability rules in `server.renderModelEntry`
/// (chat-template presence ⇒ chat/tool_use/streaming/json_schema). Returns
/// `.found=false` on any parse failure.
pub fn parseStubMeta(allocator: std.mem.Allocator, config_json: []const u8, has_chat_template: bool) StubMeta {
    var meta: StubMeta = .{};
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return meta;
    defer parsed.deinit();
    if (parsed.value != .object) return meta;
    const root = parsed.value.object;
    meta.found = true;

    // A multimodal checkpoint puts EVERY text dim under `text_config` and
    // leaves only model_type / vision_config / quantization at the root. Read
    // the nested block first and fall back to the root per field (a minimal
    // `text_config` may omit fields the root still carries) — the same merge
    // `model.zig parseConfigFromJson` does. Reading the root alone made
    // /v1/models report hidden=0 / layers=0 / ctx=0 / is_moe=false for every
    // unloaded Gemma 4 and removed-model VL model.
    const text_cfg: ?std.json.ObjectMap = if (root.get("text_config")) |tc|
        (if (tc == .object) tc.object else null)
    else
        null;
    const cfgU32 = struct {
        fn get(r: std.json.ObjectMap, tc: ?std.json.ObjectMap, key: []const u8) u32 {
            if (tc) |t| {
                const v = jsonU32(t, key);
                if (v > 0) return v;
            }
            return jsonU32(r, key);
        }
    }.get;

    meta.vocab_size = cfgU32(root, text_cfg, "vocab_size");
    meta.hidden_size = cfgU32(root, text_cfg, "hidden_size");
    meta.num_hidden_layers = cfgU32(root, text_cfg, "num_hidden_layers");
    if (meta.num_hidden_layers == 0) {
        const tc = text_cfg orelse root;
        if (tc.get("layers_block_type") orelse tc.get("block_configs")) |layers| {
            const count = switch (layers) {
                .array => |arr| arr.items.len,
                .string => |str| str.len,
                else => 0,
            };
            meta.num_hidden_layers = std.math.cast(u32, count) orelse 0;
        }
    }
    meta.max_position_embeddings = cfgU32(root, text_cfg, "max_position_embeddings");
    if (root.get("quantization")) |q| {
        if (q == .object) meta.quant_bits = jsonU32(q.object, "bits");
    }
    meta.is_moe = cfgU32(root, text_cfg, "num_experts") > 0 or
        cfgU32(root, text_cfg, "num_local_experts") > 0 or
        cfgU32(root, text_cfg, "n_routed_experts") > 0;
    const mt: []const u8 = if (root.get("model_type")) |v|
        (if (v == .string) v.string else "")
    else
        "";
    // Vision: a `vision_config` block on a non-`_text` arch (the `_text` guard
    // skips text-only quantized checkpoints with a vestigial block).
    meta.has_vision = root.get("vision_config") != null and !std.mem.endsWith(u8, mt, "_text");
    meta.has_video = meta.has_vision and cfgU32(root, text_cfg, "video_token_id") > 0;
    meta.has_chat = has_chat_template;
    if (root.get("pooling_mode")) |v| {
        if (v == .string) meta.has_embedding = true;
    }
    return meta;
}

/// Read `StubMeta` for the model directory at `abs_path` (config.json + a
/// chat-template-presence check). Best-effort: any I/O / parse failure yields
/// `.found=false`. Called per unloaded entry from `/v1/models` — cheap (small
/// JSON files) and that endpoint isn't hot.
pub fn readStubMeta(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) StubMeta {
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return .{};
    if (base.len == 0 or parent.len == 0) return .{};
    var parent_dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return .{};
    defer parent_dir.close(io);
    var dir = parent_dir.openDir(io, base, .{}) catch return .{};
    defer dir.close(io);

    var file = dir.openFile(io, "config.json", .{}) catch return .{};
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return .{};
    defer allocator.free(bytes);

    const sniff = sniffChatTemplate(io, allocator, dir);
    var meta = parseStubMeta(allocator, bytes, sniff.present);
    meta.has_thinking = meta.has_chat and sniff.thinking;
    // A sentence-transformers pooling sidecar marks embedding capability even
    // when config.json says nothing (the load path parses its mode; the stub
    // only needs existence). Issue #116.
    if (meta.found and !meta.has_embedding) {
        if (dir.statFile(io, "1_Pooling/config.json", .{})) |st| {
            if (st.kind == .file) meta.has_embedding = true;
        } else |_| {}
    }
    return meta;
}

/// Whether a chat template can open a reasoning block; one rule for the stub
/// and loaded capability paths.
pub fn templateSupportsThinking(tmpl: []const u8) bool {
    return std.mem.indexOf(u8, tmpl, "enable_thinking") != null or
        std.mem.indexOf(u8, tmpl, "<think>") != null or
        std.mem.indexOf(u8, tmpl, "<ifm|think") != null or
        std.mem.indexOf(u8, tmpl, "thought") != null or
        std.mem.indexOf(u8, tmpl, "<|channel>") != null;
}

const TemplateSniff = struct { present: bool = false, thinking: bool = false };

/// Whether the model dir ships a chat template — a `chat_template.jinja` file,
/// or a `tokenizer_config.json` that carries a `chat_template` key — and whether
/// it can think. Cheap proxy used to gate capabilities on unloaded stubs.
fn sniffChatTemplate(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) TemplateSniff {
    if (dir.readFileAlloc(io, "chat_template.jinja", allocator, .limited(8 * 1024 * 1024))) |bytes| {
        defer allocator.free(bytes);
        return .{ .present = true, .thinking = templateSupportsThinking(bytes) };
    } else |_| {}
    const bytes = dir.readFileAlloc(io, "tokenizer_config.json", allocator, .limited(8 * 1024 * 1024)) catch return .{};
    defer allocator.free(bytes);
    if (std.mem.indexOf(u8, bytes, "\"chat_template\"") == null) return .{};
    // Only the template value: `added_tokens` name `<think>` on non-thinking packs too.
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return .{ .present = true };
    defer parsed.deinit();
    const tmpl = if (parsed.value == .object) parsed.value.object.get("chat_template") else null;
    const v = tmpl orelse return .{ .present = true };
    var sniff: TemplateSniff = .{ .present = true };
    switch (v) {
        .string => |t| sniff.thinking = templateSupportsThinking(t),
        .array => |arr| for (arr.items) |item| {
            if (item != .object) continue;
            const t = item.object.get("template") orelse continue;
            if (t == .string and templateSupportsThinking(t.string)) sniff.thinking = true;
        },
        else => {},
    }
    return sniff;
}

fn lessThanById(_: void, a: DiscoveredModel, b: DiscoveredModel) bool {
    return std.mem.lessThan(u8, a.id, b.id);
}

// ── Tests ──

const testing = std.testing;

test "an unsupported architecture classifies as unsupported" {
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("nemotron_diarization"));
}

test "discoverModels finds flat and org/repo model dirs" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "flat-model");
    try tmp.dir.writeFile(io, .{ .sub_path = "flat-model/config.json", .data = "{\"model_type\":\"gemma4\"}" });
    // HF-style org/repo layout, the DownloadManager
    // convention. Must be discovered with id "org/name".
    try tmp.dir.createDirPath(io, "mlx-community/nested-model");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx-community/nested-model/config.json", .data = "{\"model_type\":\"gemma4_text\"}" });
    // Junk that must not surface: an org dir with a non-model child, and a
    // dot-dir.
    try tmp.dir.createDirPath(io, "empty-org/not-a-model");
    try tmp.dir.createDirPath(io, ".hidden/whatever");

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/models-root");
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.models.len);
    try std.testing.expectEqualStrings("flat-model", result.models[0].id);
    try std.testing.expectEqualStrings("/models-root/flat-model", result.models[0].path);
    try std.testing.expectEqualStrings("gemma4", result.models[0].model_type);
    try std.testing.expectEqualStrings("mlx-community/nested-model", result.models[1].id);
    try std.testing.expectEqualStrings("/models-root/mlx-community/nested-model", result.models[1].path);
    try std.testing.expectEqualStrings("gemma4_text", result.models[1].model_type);
}

test "discoverModelsMany merges roots in order and de-dups by id" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var a_dir = std.testing.tmpDir(.{ .iterate = true });
    defer a_dir.cleanup();
    var b_dir = std.testing.tmpDir(.{ .iterate = true });
    defer b_dir.cleanup();

    // Same id in BOTH roots. The registry answers `error.DuplicateId` and
    // `registerDiscovered` does `try`, so an un-deduped merge does not produce
    // a duplicate entry — it fails the whole registry init, i.e. the server
    // does not start. De-dup is not tidiness here.
    try a_dir.dir.createDirPath(io, "org/shared");
    try a_dir.dir.writeFile(io, .{ .sub_path = "org/shared/config.json", .data = "{\"model_type\":\"gemma4_text\"}" });
    try a_dir.dir.createDirPath(io, "org/only-in-a");
    try a_dir.dir.writeFile(io, .{ .sub_path = "org/only-in-a/config.json", .data = "{\"model_type\":\"gemma4\"}" });
    try b_dir.dir.createDirPath(io, "org/shared");
    try b_dir.dir.writeFile(io, .{ .sub_path = "org/shared/config.json", .data = "{\"model_type\":\"gemma4_text\"}" });
    try b_dir.dir.createDirPath(io, "org/only-in-b");
    try b_dir.dir.writeFile(io, .{ .sub_path = "org/only-in-b/config.json", .data = "{\"model_type\":\"gemma4_unified\"}" });

    // `discoverModels` opens by ABSOLUTE path (the openDirAbsolute UB class),
    // and a testing tmpDir is `<cwd>/.zig-cache/tmp/<sub_path>`.
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const a_path = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, a_dir.sub_path });
    defer allocator.free(a_path);
    const b_path = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, b_dir.sub_path });
    defer allocator.free(b_path);

    var result = try discoverModelsMany(io, allocator, &.{ a_path, b_path });
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.models.len);
    // FIRST root wins the shared id: roots are ordered by the caller and the
    // first is the one downloads land in, so a stale copy elsewhere must not
    // shadow the live one.
    var shared_type: []const u8 = "";
    var saw_a = false;
    var saw_b = false;
    for (result.models) |m| {
        if (std.mem.eql(u8, m.id, "org/shared")) shared_type = m.model_type;
        if (std.mem.eql(u8, m.id, "org/only-in-a")) saw_a = true;
        if (std.mem.eql(u8, m.id, "org/only-in-b")) saw_b = true;
        // Every path must be absolute and under the root it came from, or the
        // registry stores a path nothing can open.
        try std.testing.expect(m.path.len > 0 and m.path[0] == '/');
    }
    try std.testing.expectEqualStrings("gemma4_text", shared_type);
    try std.testing.expect(saw_a and saw_b);

    // One root behaves exactly like `discoverModels` — the multi path is the
    // only path, so the single-root case cannot be left behind.
    var one = try discoverModelsMany(io, allocator, &.{a_path});
    defer one.deinit();
    try std.testing.expectEqual(@as(usize, 2), one.models.len);

    // A root that does not exist is SKIPPED, not fatal: a user can unplug the
    // external drive their second folder lives on, and that must not stop the
    // server from serving everything else.
    var with_missing = try discoverModelsMany(io, allocator, &.{ a_path, "/nope/not/here", b_path });
    defer with_missing.deinit();
    try std.testing.expectEqual(@as(usize, 3), with_missing.models.len);

    // No roots at all is an empty result, never an error.
    var none = try discoverModelsMany(io, allocator, &.{});
    defer none.deinit();
    try std.testing.expectEqual(@as(usize, 0), none.models.len);
}

test "discovery measures a SYMLINKED (HF hub cache) model dir's real bytes" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // HF hub-cache snapshot shape: every file is a symlink into ../../blobs.
    // bytes_on_disk feeds /v1/models, the app's RAM column AND
    // scheduler.gateEstimateBytes — all of which saw ~0 for a 121 GB model.
    try tmp.dir.createDirPath(io, "blobs");
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/cfg", .data = "{\"model_type\":\"gemma4_text\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/w1", .data = "0123456789" });
    try tmp.dir.createDirPath(io, "org/snap-model");
    try tmp.dir.symLink(io, "../../blobs/cfg", "org/snap-model/config.json", .{});
    try tmp.dir.symLink(io, "../../blobs/w1", "org/snap-model/model.safetensors", .{});

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/root");
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.models.len);
    try testing.expectEqualStrings("org/snap-model", result.models[0].id);
    try testing.expectEqual(@as(?u64, 10), result.models[0].bytes_on_disk);
}

test "modelKindFromType labels every family (list TYPE column + run preflight)" {
    // Chat: the MLX archs.
    try testing.expectEqual(ModelKind.chat, modelKindFromType("gemma4"));
    // Drafter sidecars, and genuinely unsupported archs.
    try testing.expectEqual(ModelKind.drafter, modelKindFromType("gemma4_assistant"));
    try testing.expectEqual(ModelKind.drafter, modelKindFromType("gemma4_unified_assistant"));
    // DFlash sidecars ride the same `*_assistant` suffix rule — never listed
    // as primary models.
    try testing.expectEqual(ModelKind.drafter, modelKindFromType("llama_assistant"));
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("vit"));
    // Labels stay column-friendly.
    try testing.expectEqualStrings("chat", ModelKind.chat.label());
}

test "classifyModelPath: drafter dirs classify; junk is null" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "drafter");
    try tmp.dir.writeFile(io, .{ .sub_path = "drafter/config.json", .data = "{\"model_type\":\"gemma4_assistant\"}" });
    try tmp.dir.createDirPath(io, "junk");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];

    const cases = .{
        .{ "drafter", ModelKind.drafter },
    };
    inline for (cases) |c| {
        const p = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, c[0] });
        defer allocator.free(p);
        try testing.expectEqual(c[1], classifyModelPath(io, allocator, p).?);
    }
    const junk = try std.fmt.allocPrint(allocator, "{s}/junk", .{root});
    defer allocator.free(junk);
    try testing.expect(classifyModelPath(io, allocator, junk) == null);
    try testing.expect(classifyModelPath(io, allocator, "") == null);
    try testing.expect(classifyModelPath(io, allocator, "rel/path") == null);
}

test "a DFlash2 sidecar (bare chat model_type + dflash_config) classifies as drafter, never listed" {
    // incoai ships a bare chat `model_type` - the
    // `*_assistant` suffix rule alone would register it as a standalone chat
    // model and die at cold load (no embed weights). Classification consults
    // the DFlash config contract too (dflash.isDflashConfigJson — the ONE
    // predicate the loader itself keys on).
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const d2_config =
        \\{"model_type":"qwen3",
        \\ "dflash_config":{"block_size":8,"mask_token_id":248070,"target_layer_ids":[5,19],
        \\   "conv_kernel_size":2,"conv_group_size":16,"selector_rank":256,"selector_top_k":16},
        \\ "hidden_size":5120,"num_hidden_layers":5,"num_attention_heads":32,"head_dim":128,
        \\ "intermediate_size":17408,"rms_norm_eps":1e-6,"sliding_window":2048,
        \\ "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"]}
    ;
    try tmp.dir.createDirPath(io, "d2");
    try tmp.dir.writeFile(io, .{ .sub_path = "d2/config.json", .data = d2_config });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];
    const p = try std.fmt.allocPrint(allocator, "{s}/d2", .{root});
    defer allocator.free(p);
    try testing.expectEqual(ModelKind.drafter, classifyModelPath(io, allocator, p).?);

    // The discovery scan skips it entirely.
    var result = try discoverModels(io, allocator, root);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 0), result.models.len);
}

test "trimTrailingSlash" {
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo/"));
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo//"));
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo"));
    try testing.expectEqualStrings("", trimTrailingSlash("//"));
}

test "lessThanById sorts ascending" {
    const a: DiscoveredModel = .{ .id = "a", .path = "x", .bytes_on_disk = null };
    const b: DiscoveredModel = .{ .id = "b", .path = "x", .bytes_on_disk = null };
    try testing.expect(lessThanById({}, a, b));
    try testing.expect(!lessThanById({}, b, a));
    try testing.expect(!lessThanById({}, a, a));
}

test "isSupportedModelType: the removed families, including the hybrid trunk, are rejected" {
    try testing.expect(!isSupportedModelType("qwen2"));
    try testing.expect(!isSupportedModelType("qwen3"));
    try testing.expect(!isSupportedModelType("qwen3_moe"));
    try testing.expect(!isSupportedModelType("qwen3_moe_text"));
    try testing.expect(!isSupportedModelType("qwen3_next"));
    try testing.expect(!isSupportedModelType("qwen3_asr"));
    try testing.expect(!isSupportedModelType("qwen4_exp"));
    try testing.expect(!isSupportedModelType("qwen3_5"));
    try testing.expect(!isSupportedModelType("qwen3_5_text"));
    try testing.expect(!isSupportedModelType("qwen3_5_moe"));
    try testing.expect(!isSupportedModelType("qwen3_5_moe_text"));
    try testing.expect(!isSupportedModelType("prism_hadamard_qwen35"));
    try testing.expect(!isSupportedModelType("bailing_hybrid"));
    try testing.expect(!isSupportedModelType("minicpmv4_7"));
    try testing.expect(!isSupportedModelType("muse_glimmer"));
    try testing.expect(!isSupportedModelType("qwen3_forced_aligner"));
    try testing.expect(!isSupportedModelType("kolibri1"));
    try testing.expect(!isSupportedModelType("gpt_oss"));
    try testing.expect(!isSupportedModelType("hy_v3"));
    try testing.expect(!isSupportedModelType("laguna"));
    try testing.expect(!isSupportedModelType("inkling_mm_model"));
    try testing.expect(!isSupportedModelType("spark2_5"));
    try testing.expect(!isSupportedModelType("k2_horizon"));
    try testing.expect(!isSupportedModelType("llama"));
    try testing.expect(!isSupportedModelType("mistral"));
    try testing.expect(!isSupportedModelType("lfm2_moe"));
    try testing.expect(!isSupportedModelType("nemotron_h"));
    // A genuinely unknown arch is still rejected.
    try testing.expect(!isSupportedModelType("totally_made_up_arch"));
}

test "isSupportedModelType refuses the removed model-type spellings" {
    for ([_][]const u8{ "glm5_next", "glm5_next_text", "glm_moe_dsa", "mimo_v2", "mimo_v2_flash", "flint" }) |t| {
        try testing.expect(!isSupportedModelType(t));
    }
}

test "isSupportedQuantMode accepts nvfp4 (issue #24), rejects unknown" {
    // Regression for "[discovery] skip ...: unsupported quantization mode
    // 'nvfp4'": nvfp4 / mxfp4 / mxfp8 checkpoints are loadable and must be
    // discoverable.
    try testing.expect(isSupportedQuantMode("affine"));
    try testing.expect(isSupportedQuantMode("nvfp4"));
    try testing.expect(isSupportedQuantMode("mxfp4"));
    try testing.expect(isSupportedQuantMode("mxfp8"));
    try testing.expect(!isSupportedQuantMode("fp99"));
}

test "parseStubMeta extracts dims/ctx/quant/MoE + chat/vision capabilities" {
    const a = testing.allocator;
    // MoE chat model (a MoE shape), chat template present.
    {
        const json =
            \\{"model_type":"gemma4","vocab_size":151936,"hidden_size":2048,
            \\"num_hidden_layers":48,"max_position_embeddings":262144,
            \\"num_experts":128,"num_experts_per_tok":8,"quantization":{"bits":8,"group_size":64}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expect(m.found);
        try testing.expectEqual(@as(u32, 151936), m.vocab_size);
        try testing.expectEqual(@as(u32, 2048), m.hidden_size);
        try testing.expectEqual(@as(u32, 48), m.num_hidden_layers);
        try testing.expectEqual(@as(u32, 262144), m.max_position_embeddings);
        try testing.expectEqual(@as(u32, 8), m.quant_bits);
        try testing.expect(m.is_moe);
        try testing.expect(m.has_chat); // template present, not encoder
        try testing.expect(!m.has_vision);
    }
    // Dense model, no template → no chat caps; no quant block → 0 bits.
    {
        const json =
            \\{"model_type":"gemma4","hidden_size":5120,"num_attention_heads":40,
            \\"max_position_embeddings":32768}
        ;
        const m = parseStubMeta(a, json, false);
        try testing.expect(m.found);
        try testing.expect(!m.is_moe);
        try testing.expect(!m.has_chat);
        try testing.expectEqual(@as(u32, 0), m.quant_bits);
        try testing.expectEqual(@as(u32, 32768), m.max_position_embeddings);
    }
    // Vision: vision_config on a non-_text arch → has_vision.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"gemma4\",\"vision_config\":{\"hidden_size\":1152}}", true);
        try testing.expect(m.has_vision);
    }
    // …but a `_text` arch with a vestigial vision_config must NOT report vision.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"gemma4_text\",\"vision_config\":{}}", true);
        try testing.expect(!m.has_vision);
    }
    // Decoder embedding checkpoint: an explicit `pooling_mode` marks embeddings
    // capability and keeps the causal arch (and its chat template).
    {
        const m = parseStubMeta(a, "{\"model_type\":\"gemma4\",\"hidden_size\":2560,\"pooling_mode\":\"mean\"}", true);
        try testing.expect(m.has_embedding);
        try testing.expect(m.has_chat);
    }
    // A plain chat model advertises no embeddings capability.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"gemma4\",\"hidden_size\":2560}", true);
        try testing.expect(!m.has_embedding);
    }
    // A MULTIMODAL checkpoint keeps every text dim under `text_config` — the
    // root carries only model_type / vision_config / quantization. Reading the
    // root alone reported hidden=0, layers=0, ctx=0, is_moe=false on /v1/models
    // for EVERY unloaded Gemma 4 and removed-model VL model (which is most of them),
    // so a client couldn't tell a 128-expert MoE from a dense model without
    // cold-loading 16 GB of weights. Same class as the "Config fields omitted
    // by nested text_config" gotcha, different parser.
    {
        // gemma-4-26B-A4B-it-qat-4bit's real shape.
        const json =
            \\{"model_type":"gemma4","vision_config":{"hidden_size":1152},
            \\"quantization":{"bits":4,"group_size":32},
            \\"text_config":{"vocab_size":262144,"hidden_size":2560,"num_hidden_layers":62,
            \\"max_position_embeddings":131072,"num_experts":128,"top_k_experts":8}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expect(m.found);
        try testing.expectEqual(@as(u32, 262144), m.vocab_size);
        try testing.expectEqual(@as(u32, 2560), m.hidden_size);
        try testing.expectEqual(@as(u32, 62), m.num_hidden_layers);
        try testing.expectEqual(@as(u32, 131072), m.max_position_embeddings);
        try testing.expectEqual(@as(u32, 4), m.quant_bits); // still root-level
        try testing.expect(m.is_moe);
        try testing.expect(m.has_vision);
    }
    // A flat checkpoint whose text_config is absent keeps reading the root, and
    // a nested block that OMITS a field falls back to the root rather than
    // reporting 0 (mirrors the model.zig text_config merge).
    {
        const json =
            \\{"model_type":"gemma4","max_position_embeddings":8192,
            \\"text_config":{"hidden_size":3840}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expectEqual(@as(u32, 3840), m.hidden_size);
        try testing.expectEqual(@as(u32, 8192), m.max_position_embeddings);
    }
    // A MoE checkpoint nests its expert count too.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"gemma4\",\"text_config\":{\"num_experts\":256}}", true);
        try testing.expect(m.is_moe);
    }
    // Malformed → found=false.
    {
        const m = parseStubMeta(a, "not json", true);
        try testing.expect(!m.found);
    }
}

test "readStubMeta: has_thinking reads the template on disk" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/config.json", .data = "{\"model_type\":\"gemma4\",\"hidden_size\":8}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = try std.fmt.allocPrint(allocator, "{s}/m", .{path_buf[0..root_len]});
    defer allocator.free(model_dir);

    try tmp.dir.writeFile(io, .{ .sub_path = "m/tokenizer_config.json", .data = "{\"chat_template\":\"{% if enable_thinking %}x{% endif %}\"}" });
    try std.testing.expect(readStubMeta(io, allocator, model_dir).has_thinking);
    try tmp.dir.writeFile(io, .{ .sub_path = "m/chat_template.jinja", .data = "{{ messages[0].content }}" });
    try std.testing.expect(!readStubMeta(io, allocator, model_dir).has_thinking);
    try tmp.dir.writeFile(io, .{ .sub_path = "m/chat_template.jinja", .data = "<|im_start|>assistant\n<think>\n" });
    try std.testing.expect(readStubMeta(io, allocator, model_dir).has_thinking);
}

test "config discovery tolerates invalid roots and oversized metadata" {
    const io = testing.io;
    const allocator = testing.allocator;
    const meta = parseStubMeta(allocator, "{\"hidden_size\":4294967297}", false);
    try testing.expectEqual(@as(u32, 0), meta.hidden_size);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "[]", "null", "false", "17", "\"bad\"", "[{}]" }) |content| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = content });
        try testing.expect(peekConfig(io, allocator, tmp.dir, ".") == .missing_or_unparseable);
    }
}
