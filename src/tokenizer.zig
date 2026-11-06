const std = @import("std");
const log = @import("log.zig");
const io_util = @import("io_util.zig");

pub const TokenizerType = enum { sentencepiece_bpe };

/// An added token flagged `special: true` in tokenizer.json.
pub const FlaggedSpecial = struct { id: u32, content: []const u8 };

pub fn reservedOutputIds(
    allocator: std.mem.Allocator,
    flagged: []const FlaggedSpecial,
    template_source: []const u8,
    exempt_ids: []const u32,
) ![]u32 {
    if (template_source.len == 0) return allocator.alloc(u32, 0);
    var out = std.ArrayList(u32).empty;
    errdefer out.deinit(allocator);
    outer: for (flagged) |sp| {
        for (exempt_ids) |e| {
            if (sp.id == e) continue :outer;
        }
        if (std.mem.indexOf(u8, template_source, sp.content) != null) continue;
        try out.append(allocator, sp.id);
    }
    return out.toOwnedSlice(allocator);
}

/// BPE tokenizer in SentencePiece mode (Gemma). A byte-level BPE `tokenizer.json` is refused at load.
pub const Tokenizer = struct {
    /// Token string -> id
    vocab: std.StringHashMap(u32),
    /// Id -> token string
    id_to_token: std.AutoHashMap(u32, []const u8),
    /// (left_str, right_str) -> merge rank (lower = higher priority)
    merge_ranks: std.HashMap(MergePair, u32, MergePairContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,
    /// Special tokens (string -> id)
    special_tokens: std.StringHashMap(u32),
    /// Tokenizer type determines encode/decode behavior
    tok_type: TokenizerType,
    /// HF `Metaspace` pre-tokenizer (mmBERT / Gemma-2 class tokenizer.json):
    /// a leading ▁ is prepended when the text does not already start with
    /// one (`prepend_scheme`; `first` = only text at offset 0, never after a
    /// special token), and the text is split at every ▁ (kept with the
    /// following piece) before BPE (`split: true`).
    metaspace_prepend: MetaspacePrepend = .never,
    metaspace_split: bool = false,
    /// HF BPE `byte_fallback`: a character with no vocab entry becomes its
    /// UTF-8 bytes as `<0xNN>` tokens (all must exist, else `unk_token`);
    /// consecutive unks fuse when `fuse_unk`. Gemma / mmBERT class.
    byte_fallback: bool = false,
    fuse_unk: bool = false,
    /// HF BPE `ignore_merges`: a word the vocab holds whole is that one token, before any merge.
    ignore_merges: bool = false,
    unk_id: ?u32 = null,
    /// tokenizer.json decoder carries `ByteFallback`: runs of `<0xNN>`
    /// tokens decode as the bytes they spell (U+FFFD per byte when the run
    /// is not valid UTF-8).
    byte_fallback_decode: bool = false,
    /// The special a `TemplateProcessing` post-processor appends after a single sequence
    /// (for example `<|endoftext|>`); `encode` never adds it, a last-token pooler must.
    appended_special: ?u32 = null,
    /// GPT-2 byte-to-unicode table (256 entries, index = byte value), read where `bpeMerge` meets a symbol the vocab lacks
    byte_to_unicode: [256]u21,

    // Dynamic token IDs (populated from tokenizer.json added_tokens)
    bos_id: ?u32,
    eos_id: ?u32,

    /// Parsed `tokenizer.json`. We keep it alive so the map keys/values
    /// (vocab strings, merge pair halves, special-token names) can be
    /// borrowed directly from its arena instead of duped per entry — a 30×
    /// speedup on Gemma-class tokenizers (262k vocab + 514k merges).
    parsed_json: ?std.json.Parsed(std.json.Value) = null,

    /// Added tokens flagged `special: true` in tokenizer.json — the
    /// candidate set for reserved-output suppression. NOT the same as
    /// `special_tokens`, which deliberately holds ALL added tokens
    /// (`special: false` entries like `<think>` are atomic-encode units but
    /// perfectly legitimate output). Content slices borrow `parsed_json`'s
    /// arena; the slice itself is owned and freed in deinit.
    flagged_specials: []const FlaggedSpecial = &.{},

    /// `special_tokens` bucketed for `encode`, built once by the loader. Null
    /// (a tokenizer assembled by hand): `encode` builds it per call.
    special_index: ?SpecialIndex = null,

    /// Special tokens sorted by first byte, longest first within a byte;
    /// `start[b]..start[b + 1]` are the candidates starting with byte `b`.
    pub const SpecialIndex = struct {
        cands: []Cand,
        start: [257]u32,

        pub const Cand = struct { bytes: []const u8, id: u32 };

        pub fn init(allocator: std.mem.Allocator, specials: *const std.StringHashMap(u32)) !SpecialIndex {
            const cands = try allocator.alloc(Cand, specials.count());
            var i: usize = 0;
            var sit = specials.iterator();
            while (sit.next()) |entry| : (i += 1) {
                cands[i] = .{ .bytes = entry.key_ptr.*, .id = entry.value_ptr.* };
            }
            std.mem.sort(Cand, cands, {}, struct {
                fn lessThan(_: void, a: Cand, b: Cand) bool {
                    const ab: u8 = if (a.bytes.len > 0) a.bytes[0] else 0;
                    const bb: u8 = if (b.bytes.len > 0) b.bytes[0] else 0;
                    if (ab != bb) return ab < bb;
                    return a.bytes.len > b.bytes.len;
                }
            }.lessThan);
            var start: [257]u32 = @splat(0);
            var ci: usize = 0;
            for (0..256) |b| {
                start[b] = @intCast(ci);
                while (ci < cands.len and cands[ci].bytes.len > 0 and cands[ci].bytes[0] == b) ci += 1;
            }
            start[256] = @intCast(cands.len);
            return .{ .cands = cands, .start = start };
        }

        pub fn deinit(self: *SpecialIndex, allocator: std.mem.Allocator) void {
            allocator.free(self.cands);
        }
    };

    const MergePair = struct {
        left: []const u8,
        right: []const u8,
    };

    const MergePairContext = struct {
        pub fn hash(_: MergePairContext, key: MergePair) u64 {
            var h = std.hash.Wyhash.init(0);
            h.update(key.left);
            h.update("\x00");
            h.update(key.right);
            return h.final();
        }
        pub fn eql(_: MergePairContext, a: MergePair, b: MergePair) bool {
            return std.mem.eql(u8, a.left, b.left) and std.mem.eql(u8, a.right, b.right);
        }
    };

    /// A decode-capable tokenizer with EMPTY maps, for tests in other modules
    /// that need to turn ids into text without a checkpoint on disk. The caller
    /// owns every map and must `deinit` them; nothing here reads tokenizer.json,
    /// so encode is not meaningfully usable — populate `id_to_token` and decode.
    pub fn initEmptyForTests(allocator: std.mem.Allocator, tok_type: TokenizerType) Tokenizer {
        return .{
            .vocab = std.StringHashMap(u32).init(allocator),
            .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
            .merge_ranks = std.HashMap(MergePair, u32, MergePairContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
            .special_tokens = std.StringHashMap(u32).init(allocator),
            .tok_type = tok_type,
            .byte_to_unicode = buildBytesToUnicode(),
            .bos_id = null,
            .eos_id = null,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        // Map keys/values either point into `parsed_json`'s arena (no
        // per-entry free needed) or were duped explicitly when no parsed
        // JSON is held (e.g., the test-only constructors). Freeing the
        // parsed JSON deinits its arena in one shot.
        if (self.flagged_specials.len > 0) self.allocator.free(self.flagged_specials);
        if (self.special_index) |*x| x.deinit(self.allocator);
        if (self.parsed_json) |*p| {
            self.vocab.deinit();
            self.id_to_token.deinit();
            self.merge_ranks.deinit();
            self.special_tokens.deinit();
            p.deinit();
            return;
        }
        var vit = self.vocab.iterator();
        while (vit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();

        var mit = self.merge_ranks.iterator();
        while (mit.next()) |entry| {
            self.allocator.free(entry.key_ptr.left);
            self.allocator.free(entry.key_ptr.right);
        }
        self.merge_ranks.deinit();

        var sit = self.special_tokens.iterator();
        while (sit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.special_tokens.deinit();
    }

    /// Encode text to token IDs (no BOS/EOS added).
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        // Split text around special tokens, encode segments with BPE, insert special token IDs
        var result = std.ArrayList(u32).empty;
        errdefer result.deinit(allocator);

        // Special-token splitter. The old scan re-searched the ENTIRE
        // remaining text for EVERY special token per segment
        // (O(specials × text), ~12 s per 66 KB prompt on a removed family's
        // 6415-special vocabulary; gemma-4's 24 specials never noticed).
        // Instead: specials bucketed by first byte (`SpecialIndex`, built
        // once at load), then a single left-to-right pass tries only the
        // candidates whose first byte matches. Semantics unchanged — earliest
        // occurrence wins, longest special wins at the same position
        // (buckets are sorted by descending length, so the first hit is the longest).
        var local_index: ?SpecialIndex = null;
        defer if (local_index) |*x| x.deinit(allocator);
        const index = if (self.special_index) |*x| x else blk: {
            local_index = try SpecialIndex.init(allocator, &self.special_tokens);
            break :blk &local_index.?;
        };
        const cands = index.cands;
        const bucket_start = &index.start;
        const Cand = SpecialIndex.Cand;

        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            const b = text[pos];
            var ci = bucket_start[b];
            const cend = bucket_start[@as(usize, b) + 1];
            var matched: ?Cand = null;
            while (ci < cend) : (ci += 1) {
                const c = cands[ci];
                if (c.bytes.len <= text.len - pos and std.mem.startsWith(u8, text[pos..], c.bytes)) {
                    matched = c;
                    break;
                }
            }
            if (matched) |m| {
                if (pos > seg_start) {
                    const ids = try self.encodeSegment(allocator, text[seg_start..pos], seg_start == 0);
                    defer allocator.free(ids);
                    try result.appendSlice(allocator, ids);
                }
                try result.append(allocator, m.id);
                pos += m.bytes.len;
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        if (seg_start < text.len) {
            const ids = try self.encodeSegment(allocator, text[seg_start..], seg_start == 0);
            defer allocator.free(ids);
            try result.appendSlice(allocator, ids);
        }

        return result.toOwnedSlice(allocator);
    }

    /// Encode a text segment (no special tokens) using the appropriate method.
    fn encodeSegment(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, at_start: bool) ![]u32 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.encodeSentencePiece(allocator, text, at_start),
        };
    }

    /// Decode token IDs to text.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, strip_leading_space: bool) ![]u8 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.decodeSentencePiece(allocator, ids, strip_leading_space),
        };
    }

    /// Look up a special token ID by its string representation.
    pub fn specialTokenId(self: *const Tokenizer, name: []const u8) ?u32 {
        return self.special_tokens.get(name);
    }

    /// Reserved-output ids for this tokenizer: `reservedOutputIds` over the
    /// `special: true` added tokens recorded at load. See that function for
    /// the derivation rules.
    pub fn reservedIds(self: *const Tokenizer, allocator: std.mem.Allocator, template_source: []const u8, exempt_ids: []const u32) ![]u32 {
        return reservedOutputIds(allocator, self.flagged_specials, template_source, exempt_ids);
    }

    /// Highest DEFINED id + 1 — the real vocabulary, which is not the config's
    /// `vocab_size`: checkpoints pad the embedding/lm_head rows out to a
    /// friendly multiple (one model: 248044 base + 33 added = 248077 defined
    /// against a declared 248320, so 243 rows decode to nothing at all). Those
    /// rows carry whatever the initializer left and a collapsed distribution
    /// can rank one top-1; `installSuppressMask` masks them out of sampling.
    /// 0 when nothing is defined (an `initEmptyForTests` tokenizer) — callers
    /// treat that as "no padding known", never as "suppress everything".
    pub fn definedVocabSize(self: *const Tokenizer) usize {
        var highest: ?u32 = null;
        var it = self.id_to_token.keyIterator();
        while (it.next()) |id| {
            if (highest == null or id.* > highest.?) highest = id.*;
        }
        return if (highest) |h| @as(usize, h) + 1 else 0;
    }

    // ── SentencePiece BPE (Gemma-style) ──

    fn encodeSentencePiece(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, at_start: bool) ![]u32 {
        // Normalize: replace spaces with ▁ (U+2581)
        var normalized: std.ArrayList(u8) = .empty;
        defer normalized.deinit(allocator);

        const sep = "\xe2\x96\x81";
        const prepend = switch (self.metaspace_prepend) {
            .never => false,
            .first => at_start,
            .always => true,
        };
        if (prepend and !std.mem.startsWith(u8, text, " ") and !std.mem.startsWith(u8, text, sep)) {
            try normalized.appendSlice(allocator, sep);
        }
        for (text) |c| {
            if (c == ' ') {
                try normalized.appendSlice(allocator, sep);
            } else {
                try normalized.append(allocator, c);
            }
        }

        if (!self.metaspace_split) return self.bpeMerge(allocator, normalized.items);

        // Metaspace `split: true` = SplitDelimiterBehavior::MergedWithNext on
        // ▁: every piece starts at a ▁ (a run of ▁ yields lone "▁" pieces) and
        // BPE never merges across pieces.
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);
        const n = normalized.items;
        var start: usize = 0;
        var i: usize = if (std.mem.startsWith(u8, n, sep)) sep.len else 0;
        while (i <= n.len) {
            const at_sep = i + sep.len <= n.len and std.mem.eql(u8, n[i .. i + sep.len], sep);
            if (i == n.len or at_sep) {
                if (i > start) {
                    const piece = try self.bpeMerge(allocator, n[start..i]);
                    defer allocator.free(piece);
                    try ids.appendSlice(allocator, piece);
                }
                start = i;
                if (i == n.len) break;
                i += sep.len;
            } else {
                i += 1;
            }
        }
        return ids.toOwnedSlice(allocator);
    }

    fn decodeSentencePiece(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, strip_leading_space: bool) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        defer result.deinit(allocator);

        // `ByteFallback` decoder: a run of `<0xNN>` tokens is one byte string,
        // appended verbatim when it is valid UTF-8, else U+FFFD per byte.
        var pending: std.ArrayList(u8) = .empty;
        defer pending.deinit(allocator);
        for (ids) |id| {
            const token = self.id_to_token.get(id) orelse continue;
            if (self.byte_fallback_decode) {
                if (byteTokenValue(token)) |b| {
                    try pending.append(allocator, b);
                    continue;
                }
                try flushByteRun(allocator, &result, &pending);
            }
            try result.appendSlice(allocator, token);
        }
        try flushByteRun(allocator, &result, &pending);

        // Replace ▁ (0xE2 0x96 0x81) with space
        var output = try allocator.alloc(u8, result.items.len);
        var out_len: usize = 0;
        var i: usize = 0;
        while (i < result.items.len) {
            if (i + 2 < result.items.len and
                result.items[i] == 0xE2 and
                result.items[i + 1] == 0x96 and
                result.items[i + 2] == 0x81)
            {
                output[out_len] = ' ';
                out_len += 1;
                i += 3;
            } else {
                output[out_len] = result.items[i];
                out_len += 1;
                i += 1;
            }
        }

        const start: usize = if (strip_leading_space and out_len > 0 and output[0] == ' ') 1 else 0;
        const final_out = try allocator.dupe(u8, output[start..out_len]);
        allocator.free(output);
        return final_out;
    }

    // ── Shared BPE merge logic ──

    /// One symbol in the BPE merge worklist. Symbols are always contiguous
    /// slices `input[start..end]` — merging adjacent symbols just extends the
    /// left node's range and unlinks the right, so no bytes are ever copied.
    /// `ver` invalidates stale heap candidates: any merge touching a node
    /// bumps its version, and a candidate is only applied when both stored
    /// versions still match.
    const BpeNode = struct {
        start: u32,
        end: u32,
        prev: i32,
        next: i32,
        ver: u32,
    };

    /// Candidate pair in the merge heap, ordered by (rank, left node index).
    /// Node indices are creation order = text order and merges never create
    /// nodes, so the secondary key reproduces the naive scan's leftmost-wins
    /// tie-break exactly.
    const BpeCand = struct {
        rank: u32,
        left: u32,
        right: u32,
        lver: u32,
        rver: u32,

        fn order(_: void, a: BpeCand, b: BpeCand) std.math.Order {
            if (a.rank != b.rank) return std.math.order(a.rank, b.rank);
            return std.math.order(a.left, b.left);
        }
    };

    const BpeHeap = std.PriorityQueue(BpeCand, void, BpeCand.order);

    fn bpePushCand(self: *const Tokenizer, allocator: std.mem.Allocator, heap: *BpeHeap, nodes: []const BpeNode, input: []const u8, l: u32, r: u32) !void {
        const pair = MergePair{
            .left = input[nodes[l].start..nodes[l].end],
            .right = input[nodes[r].start..nodes[r].end],
        };
        if (self.merge_ranks.get(pair)) |rank| {
            try heap.push(allocator, .{ .rank = rank, .left = l, .right = r, .lver = nodes[l].ver, .rver = nodes[r].ver });
        }
    }

    /// Greedy BPE: repeatedly merge the lowest-ranked adjacent pair
    /// (leftmost on ties) until no pair has a rank. Heap + linked-list,
    /// O(n log n); `encodeSentencePiece` feeds entire prompts through here
    /// with no pre-tokenization, so the previous rescan-all-pairs loop was
    /// O(n²) and cost seconds on agent-sized (tens-of-KB) system prompts.
    fn bpeMerge(self: *const Tokenizer, allocator: std.mem.Allocator, input: []const u8) ![]u32 {
        if (self.ignore_merges) if (self.vocab.get(input)) |id| {
            const out = try allocator.alloc(u32, 1);
            out[0] = id;
            return out;
        };
        // Split into individual UTF-8 characters.
        var nodes: std.ArrayList(BpeNode) = .empty;
        defer nodes.deinit(allocator);

        var idx: usize = 0;
        while (idx < input.len) {
            const char_len = std.unicode.utf8ByteSequenceLength(input[idx]) catch 1;
            const end = @min(idx + char_len, input.len);
            const i: i32 = @intCast(nodes.items.len);
            try nodes.append(allocator, .{
                .start = @intCast(idx),
                .end = @intCast(end),
                .prev = i - 1,
                .next = if (end < input.len) i + 1 else -1,
                .ver = 0,
            });
            idx = end;
        }

        var heap: BpeHeap = .initContext({});
        defer heap.deinit(allocator);

        if (nodes.items.len > 1) {
            for (0..nodes.items.len - 1) |j| {
                try self.bpePushCand(allocator, &heap, nodes.items, input, @intCast(j), @intCast(j + 1));
            }
        }

        while (heap.pop()) |c| {
            const ns = nodes.items;
            if (ns[c.left].ver != c.lver or ns[c.right].ver != c.rver) continue;

            // Merge right into left: extend range, unlink right, invalidate
            // every candidate that referenced either node's old content.
            ns[c.left].end = ns[c.right].end;
            ns[c.left].ver += 1;
            ns[c.right].ver += 1;
            ns[c.left].next = ns[c.right].next;
            if (ns[c.right].next >= 0) ns[@intCast(ns[c.right].next)].prev = @intCast(c.left);

            if (ns[c.left].prev >= 0) try self.bpePushCand(allocator, &heap, ns, input, @intCast(ns[c.left].prev), c.left);
            if (ns[c.left].next >= 0) try self.bpePushCand(allocator, &heap, ns, input, c.left, @intCast(ns[c.left].next));
        }

        // Map surviving symbols to vocab IDs.
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        var cur: i32 = if (nodes.items.len > 0) 0 else -1;
        while (cur >= 0) : (cur = nodes.items[@intCast(cur)].next) {
            const n = nodes.items[@intCast(cur)];
            const sym = input[n.start..n.end];
            if (self.vocab.get(sym)) |id| {
                try ids.append(allocator, id);
            } else if (self.byte_fallback) {
                try self.appendByteFallback(allocator, &ids, sym);
            } else {
                // Unknown token — try to encode individual bytes
                for (sym) |byte| {
                    var utf8_buf: [4]u8 = undefined;
                    const cp = self.byte_to_unicode[byte];
                    const len = std.unicode.utf8Encode(cp, &utf8_buf) catch 1;
                    if (self.vocab.get(utf8_buf[0..len])) |id| {
                        try ids.append(allocator, id);
                    }
                }
            }
        }

        return ids.toOwnedSlice(allocator);
    }

    /// `BPE::merge_word` with `byte_fallback`: `sym` (one character HF found
    /// no vocab entry for) becomes one `<0xNN>` id per byte when every byte
    /// token exists, else the unk id — fused with a directly preceding unk
    /// when `fuse_unk`.
    fn appendByteFallback(self: *const Tokenizer, allocator: std.mem.Allocator, ids: *std.ArrayList(u32), sym: []const u8) !void {
        var byte_ids: [4]u32 = undefined;
        var all = sym.len <= byte_ids.len;
        if (all) for (sym, 0..) |byte, i| {
            var name: [6]u8 = undefined;
            const key = std.fmt.bufPrint(&name, "<0x{X:0>2}>", .{byte}) catch unreachable;
            byte_ids[i] = self.vocab.get(key) orelse {
                all = false;
                break;
            };
        };
        if (all) {
            try ids.appendSlice(allocator, byte_ids[0..sym.len]);
            return;
        }
        const unk = self.unk_id orelse return;
        if (self.fuse_unk and ids.items.len > 0 and ids.items[ids.items.len - 1] == unk) return;
        try ids.append(allocator, unk);
    }
};

/// `\p{N}` (Nd, Nl, No) as inclusive ranges, Unicode 15.1.0:
/// `[(a, b) for runs of unicodedata.category(c)[0] == 'N']`.
const NUMBER_RANGES = [_][2]u21{
    .{ 0x30, 0x39 }, .{ 0xB2, 0xB3 }, .{ 0xB9, 0xB9 }, .{ 0xBC, 0xBE }, .{ 0x660, 0x669 },
    .{ 0x6F0, 0x6F9 }, .{ 0x7C0, 0x7C9 }, .{ 0x966, 0x96F }, .{ 0x9E6, 0x9EF }, .{ 0x9F4, 0x9F9 },
    .{ 0xA66, 0xA6F }, .{ 0xAE6, 0xAEF }, .{ 0xB66, 0xB6F }, .{ 0xB72, 0xB77 }, .{ 0xBE6, 0xBF2 },
    .{ 0xC66, 0xC6F }, .{ 0xC78, 0xC7E }, .{ 0xCE6, 0xCEF }, .{ 0xD58, 0xD5E }, .{ 0xD66, 0xD78 },
    .{ 0xDE6, 0xDEF }, .{ 0xE50, 0xE59 }, .{ 0xED0, 0xED9 }, .{ 0xF20, 0xF33 }, .{ 0x1040, 0x1049 },
    .{ 0x1090, 0x1099 }, .{ 0x1369, 0x137C }, .{ 0x16EE, 0x16F0 }, .{ 0x17E0, 0x17E9 },
    .{ 0x17F0, 0x17F9 }, .{ 0x1810, 0x1819 }, .{ 0x1946, 0x194F }, .{ 0x19D0, 0x19DA },
    .{ 0x1A80, 0x1A89 }, .{ 0x1A90, 0x1A99 }, .{ 0x1B50, 0x1B59 }, .{ 0x1BB0, 0x1BB9 },
    .{ 0x1C40, 0x1C49 }, .{ 0x1C50, 0x1C59 }, .{ 0x2070, 0x2070 }, .{ 0x2074, 0x2079 },
    .{ 0x2080, 0x2089 }, .{ 0x2150, 0x2182 }, .{ 0x2185, 0x2189 }, .{ 0x2460, 0x249B },
    .{ 0x24EA, 0x24FF }, .{ 0x2776, 0x2793 }, .{ 0x2CFD, 0x2CFD }, .{ 0x3007, 0x3007 },
    .{ 0x3021, 0x3029 }, .{ 0x3038, 0x303A }, .{ 0x3192, 0x3195 }, .{ 0x3220, 0x3229 },
    .{ 0x3248, 0x324F }, .{ 0x3251, 0x325F }, .{ 0x3280, 0x3289 }, .{ 0x32B1, 0x32BF },
    .{ 0xA620, 0xA629 }, .{ 0xA6E6, 0xA6EF }, .{ 0xA830, 0xA835 }, .{ 0xA8D0, 0xA8D9 },
    .{ 0xA900, 0xA909 }, .{ 0xA9D0, 0xA9D9 }, .{ 0xA9F0, 0xA9F9 }, .{ 0xAA50, 0xAA59 },
    .{ 0xABF0, 0xABF9 }, .{ 0xFF10, 0xFF19 }, .{ 0x10107, 0x10133 }, .{ 0x10140, 0x10178 },
    .{ 0x1018A, 0x1018B }, .{ 0x102E1, 0x102FB }, .{ 0x10320, 0x10323 }, .{ 0x10341, 0x10341 },
    .{ 0x1034A, 0x1034A }, .{ 0x103D1, 0x103D5 }, .{ 0x104A0, 0x104A9 }, .{ 0x10858, 0x1085F },
    .{ 0x10879, 0x1087F }, .{ 0x108A7, 0x108AF }, .{ 0x108FB, 0x108FF }, .{ 0x10916, 0x1091B },
    .{ 0x109BC, 0x109BD }, .{ 0x109C0, 0x109CF }, .{ 0x109D2, 0x109FF }, .{ 0x10A40, 0x10A48 },
    .{ 0x10A7D, 0x10A7E }, .{ 0x10A9D, 0x10A9F }, .{ 0x10AEB, 0x10AEF }, .{ 0x10B58, 0x10B5F },
    .{ 0x10B78, 0x10B7F }, .{ 0x10BA9, 0x10BAF }, .{ 0x10CFA, 0x10CFF }, .{ 0x10D30, 0x10D39 },
    .{ 0x10E60, 0x10E7E }, .{ 0x10F1D, 0x10F26 }, .{ 0x10F51, 0x10F54 }, .{ 0x10FC5, 0x10FCB },
    .{ 0x11052, 0x1106F }, .{ 0x110F0, 0x110F9 }, .{ 0x11136, 0x1113F }, .{ 0x111D0, 0x111D9 },
    .{ 0x111E1, 0x111F4 }, .{ 0x112F0, 0x112F9 }, .{ 0x11450, 0x11459 }, .{ 0x114D0, 0x114D9 },
    .{ 0x11650, 0x11659 }, .{ 0x116C0, 0x116C9 }, .{ 0x11730, 0x1173B }, .{ 0x118E0, 0x118F2 },
    .{ 0x11950, 0x11959 }, .{ 0x11C50, 0x11C6C }, .{ 0x11D50, 0x11D59 }, .{ 0x11DA0, 0x11DA9 },
    .{ 0x11F50, 0x11F59 }, .{ 0x11FC0, 0x11FD4 }, .{ 0x12400, 0x1246E }, .{ 0x16A60, 0x16A69 },
    .{ 0x16AC0, 0x16AC9 }, .{ 0x16B50, 0x16B59 }, .{ 0x16B5B, 0x16B61 }, .{ 0x16E80, 0x16E96 },
    .{ 0x1D2C0, 0x1D2D3 }, .{ 0x1D2E0, 0x1D2F3 }, .{ 0x1D360, 0x1D378 }, .{ 0x1D7CE, 0x1D7FF },
    .{ 0x1E140, 0x1E149 }, .{ 0x1E2F0, 0x1E2F9 }, .{ 0x1E4F0, 0x1E4F9 }, .{ 0x1E8C7, 0x1E8CF },
    .{ 0x1E950, 0x1E959 }, .{ 0x1EC71, 0x1ECAB }, .{ 0x1ECAD, 0x1ECAF }, .{ 0x1ECB1, 0x1ECB4 },
    .{ 0x1ED01, 0x1ED2D }, .{ 0x1ED2F, 0x1ED3D }, .{ 0x1F100, 0x1F10C }, .{ 0x1FBF0, 0x1FBF9 },
};

/// `\p{N}`: every script's digits, superscripts, fractions and numerals, not just ASCII.
pub fn isDigit(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    var lo: usize = 0;
    var hi: usize = NUMBER_RANGES.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (cp < NUMBER_RANGES[mid][0]) hi = mid else if (cp > NUMBER_RANGES[mid][1]) lo = mid + 1 else return true;
    }
    return false;
}

/// Build the GPT-2 bytes_to_unicode mapping (256 entries).
fn buildBytesToUnicode() [256]u21 {
    var table: [256]u21 = undefined;
    var n: u21 = 256; // Next available unicode codepoint for unmapped bytes

    for (0..256) |b| {
        const byte: u8 = @intCast(b);
        // Printable ASCII range + some Latin-1 chars get identity mapping
        if ((byte >= '!' and byte <= '~') or
            (byte >= 0xA1 and byte <= 0xAC) or
            (byte >= 0xAE and byte <= 0xFF))
        {
            table[b] = @intCast(b);
        } else {
            // Non-printable bytes get mapped to codepoints starting at U+0100
            table[b] = n;
            n += 1;
        }
    }
    return table;
}

/// Parse tokenizer.json and return a Tokenizer.
pub fn loadTokenizer(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
    const content = blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(256 * 1024 * 1024));
    };
    defer allocator.free(content);
    return parseTokenizerContent(io, allocator, content);
}

/// Parse a `tokenizer.json`-shaped JSON document into a `Tokenizer`. Split out
/// from `loadTokenizer` so the slow-format loader (`loadTokenizerSlow`) can feed
/// synthesized content. `parsed` (and the string keys borrowed from it) is owned
/// by the returned `Tokenizer.parsed_json`, so `content` may be freed after.
fn parseTokenizerContent(io: std.Io, allocator: std.mem.Allocator, content: []const u8) !Tokenizer {
    var sw = io_util.Stopwatch.init(io);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    errdefer parsed.deinit();
    log.info("  parsed tokenizer ({d} MB) in {d}ms\n", .{ content.len / (1024 * 1024), sw.read() / std.time.ns_per_ms });

    const root = parsed.value.object;
    const model_obj = root.get("model").?.object;

    // Detect tokenizer type
    const tok_type: TokenizerType = .sentencepiece_bpe;
    if (root.get("pre_tokenizer")) |pt| {
        if (pt == .object) {
            if (hasByteLevel(pt)) {
                log.err("unsupported tokenizer: byte-level BPE (only SentencePiece BPE is served)\n", .{});
                return error.UnsupportedTokenizer;
            }
        }
    }

    // Parse vocab — keys borrow directly from `parsed`'s arena (no per-
    // entry dupe). Pre-size to avoid rehashing during the inserts.
    var vocab = std.StringHashMap(u32).init(allocator);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);

    const vocab_obj = model_obj.get("vocab").?.object;
    try vocab.ensureTotalCapacity(@intCast(vocab_obj.count()));
    try id_to_token.ensureTotalCapacity(@intCast(vocab_obj.count()));
    sw = io_util.Stopwatch.init(io);
    var vit = vocab_obj.iterator();
    while (vit.next()) |entry| {
        const key = entry.key_ptr.*;
        const id: u32 = @intCast(entry.value_ptr.integer);
        try vocab.put(key, id);
        try id_to_token.put(id, key);
    }
    log.info("  loaded {d} vocab entries in {d}ms\n", .{ vocab.count(), sw.read() / std.time.ns_per_ms });

    // Parse merges (array format: [["a", "b"], ...]). String halves are
    // borrowed from `parsed`'s arena.
    var merge_ranks = std.HashMap(
        Tokenizer.MergePair,
        u32,
        Tokenizer.MergePairContext,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);

    if (model_obj.get("merges")) |merges_val| {
        const merges_arr = merges_val.array;
        try merge_ranks.ensureTotalCapacity(@intCast(merges_arr.items.len));
        sw = io_util.Stopwatch.init(io);
        for (merges_arr.items, 0..) |merge_val, rank| {
            // Two on-disk formats: newer HF tokenizers store each merge as a
            // 2-element array `["a","b"]` (Gemma 4); older / GPT-2-style
            // exports store it as a single space-joined string `"a b"` (
            // many exports). `parseMergePair` handles both. Reading `.array`
            // off a string Value in ReleaseFast spins the loop on garbage, which
            // hung model loads before this fix.
            const pair = parseMergePair(merge_val) orelse continue;
            try merge_ranks.put(pair, @intCast(rank));
        }
        log.info("  loaded {d} merges in {d}ms\n", .{ merge_ranks.count(), sw.read() / std.time.ns_per_ms });
    }

    // Parse added_tokens
    var special_tokens = std.StringHashMap(u32).init(allocator);
    var flagged = std.ArrayList(FlaggedSpecial).empty;
    errdefer flagged.deinit(allocator);
    var bos_id: ?u32 = null;
    var eos_id: ?u32 = null;

    if (root.get("added_tokens")) |at_val| {
        for (at_val.array.items) |token_val| {
            const obj = token_val.object;
            const content_str = obj.get("content").?.string;
            const id: u32 = @intCast(obj.get("id").?.integer);
            // Include ALL added tokens (both special=true and special=false like <think>, <tool_call>)
            // so they are tokenized as single atomic units. The string is
            // borrowed from `parsed`'s arena — no per-entry dupe.
            try special_tokens.put(content_str, id);
            // `special: true` entries additionally feed reserved-output
            // suppression (see `reservedOutputIds`).
            if (obj.get("special")) |sv| {
                if (sv == .bool and sv.bool) {
                    try flagged.append(allocator, .{ .id = id, .content = content_str });
                }
            }
            if (!vocab.contains(content_str)) {
                try vocab.put(content_str, id);
                try id_to_token.put(id, content_str);
            }
        }
    }

    // Try to find BOS/EOS from common special token patterns
    bos_id = special_tokens.get("<bos>") orelse special_tokens.get("<|startoftext|>");
    eos_id = special_tokens.get("<eos>") orelse special_tokens.get("<|im_end|>") orelse special_tokens.get("<|endoftext|>") orelse special_tokens.get("[SEP]");

    const byte_to_unicode = buildBytesToUnicode();

    log.info("Tokenizer loaded: {d} vocab, {d} merges, {d} special tokens ({s})\n", .{
        vocab.count(),
        merge_ranks.count(),
        special_tokens.count(),
        switch (tok_type) {
            .sentencepiece_bpe => "SentencePiece BPE",
        },
    });

    var special_index = try Tokenizer.SpecialIndex.init(allocator, &special_tokens);
    errdefer special_index.deinit(allocator);
    const built: Tokenizer = .{
        .special_index = special_index,
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .flagged_specials = try flagged.toOwnedSlice(allocator),
        .tok_type = tok_type,
        .metaspace_prepend = if (root.get("pre_tokenizer")) |pt| metaspaceFromPreTokenizer(pt).prepend else .never,
        .metaspace_split = if (root.get("pre_tokenizer")) |pt| metaspaceFromPreTokenizer(pt).split else false,
        .byte_fallback = if (model_obj.get("byte_fallback")) |v| v == .bool and v.bool else false,
        .ignore_merges = if (model_obj.get("ignore_merges")) |v| v == .bool and v.bool else false,
        .fuse_unk = if (model_obj.get("fuse_unk")) |v| v == .bool and v.bool else false,
        .unk_id = if (model_obj.get("unk_token")) |v| (if (v == .string) vocab.get(v.string) else null) else null,
        .byte_fallback_decode = if (root.get("decoder")) |d| decoderHasByteFallback(d) else false,
        .appended_special = if (root.get("post_processor")) |pp| appendedSpecialId(pp, &special_tokens) else null,
        .byte_to_unicode = byte_to_unicode,
        .bos_id = bos_id,
        .eos_id = eos_id,
        .parsed_json = parsed,
    };
    return built;
}

/// Parse one BPE merge entry, accepting both on-disk formats:
///   - 2-element array `["left","right"]` (newer HF: Gemma 4)
///   - space-joined string `"left right"` (GPT-2-style, many exports)
/// Returns null for malformed entries (skip). The returned slices borrow from
/// `merge_val`'s backing arena, same lifetime as the array-format halves.
fn parseMergePair(merge_val: std.json.Value) ?Tokenizer.MergePair {
    switch (merge_val) {
        .array => |pair| {
            if (pair.items.len < 2) return null;
            if (pair.items[0] != .string or pair.items[1] != .string) return null;
            return .{ .left = pair.items[0].string, .right = pair.items[1].string };
        },
        .string => |s| {
            // Split on the FIRST space: byte-level BPE encodes spaces as 'Ġ',
            // so neither half contains a literal space separator.
            const sp = std.mem.indexOfScalar(u8, s, ' ') orelse return null;
            return .{ .left = s[0..sp], .right = s[sp + 1 ..] };
        },
        else => return null,
    }
}

/// Check if a pre_tokenizer JSON value contains a ByteLevel type.
const MetaspacePrepend = enum { never, first, always };
const MetaspaceOpts = struct { prepend: MetaspacePrepend = .never, split: bool = false };

/// `Metaspace` pre-tokenizer options from tokenizer.json (top-level or inside
/// a `Sequence`). Absent → both false, which is the pre-existing behaviour
/// (Gemma 4's `Split " " MergedWithPrevious` never reaches here).
/// `<0xNN>` -> NN, null for any other token.
fn byteTokenValue(token: []const u8) ?u8 {
    if (token.len != 6 or !std.mem.startsWith(u8, token, "<0x") or token[5] != '>') return null;
    return std.fmt.parseInt(u8, token[3..5], 16) catch null;
}

fn flushByteRun(allocator: std.mem.Allocator, out: *std.ArrayList(u8), pending: *std.ArrayList(u8)) !void {
    if (pending.items.len == 0) return;
    if (std.unicode.utf8ValidateSlice(pending.items)) {
        try out.appendSlice(allocator, pending.items);
    } else {
        for (pending.items) |_| try out.appendSlice(allocator, "\xef\xbf\xbd");
    }
    pending.clearRetainingCapacity();
}

/// True when the tokenizer.json `decoder` is `ByteFallback` or a `Sequence`
/// containing one.
fn decoderHasByteFallback(dec: std.json.Value) bool {
    if (dec != .object) return false;
    const t = dec.object.get("type") orelse return false;
    if (t == .string and std.mem.eql(u8, t.string, "ByteFallback")) return true;
    if (t == .string and std.mem.eql(u8, t.string, "Sequence")) {
        const subs = dec.object.get("decoders") orelse return false;
        if (subs != .array) return false;
        for (subs.array.items) |sub| if (decoderHasByteFallback(sub)) return true;
    }
    return false;
}

/// The id of the `SpecialToken` a `TemplateProcessing` (top-level or inside a `Sequence`) puts
/// LAST in its `single` template, or null when nothing is appended.
fn appendedSpecialId(pp: std.json.Value, specials: *const std.StringHashMap(u32)) ?u32 {
    if (pp != .object) return null;
    const t = pp.object.get("type") orelse return null;
    if (t != .string) return null;
    if (std.mem.eql(u8, t.string, "Sequence")) {
        const subs = pp.object.get("processors") orelse return null;
        if (subs != .array) return null;
        for (subs.array.items) |sub| if (appendedSpecialId(sub, specials)) |id| return id;
        return null;
    }
    if (!std.mem.eql(u8, t.string, "TemplateProcessing")) return null;
    const single = pp.object.get("single") orelse return null;
    if (single != .array or single.array.items.len == 0) return null;
    const last = single.array.items[single.array.items.len - 1];
    if (last != .object) return null;
    const st = last.object.get("SpecialToken") orelse return null;
    if (st != .object) return null;
    const name = st.object.get("id") orelse return null;
    return if (name == .string) specials.get(name.string) else null;
}

fn metaspaceFromPreTokenizer(pt: std.json.Value) MetaspaceOpts {
    if (pt != .object) return .{};
    const t = pt.object.get("type") orelse return .{};
    if (t != .string) return .{};
    if (std.mem.eql(u8, t.string, "Sequence")) {
        if (pt.object.get("pretokenizers")) |pts| {
            if (pts == .array) for (pts.array.items) |sub| {
                const o = metaspaceFromPreTokenizer(sub);
                if (o.prepend != .never or o.split) return o;
            };
        }
        return .{};
    }
    if (!std.mem.eql(u8, t.string, "Metaspace")) return .{};
    var opts = MetaspaceOpts{ .prepend = .always, .split = true };
    if (pt.object.get("prepend_scheme")) |ps| {
        if (ps == .string) opts.prepend = std.meta.stringToEnum(MetaspacePrepend, ps.string) orelse .always;
    }
    if (pt.object.get("split")) |sp| {
        if (sp == .bool) opts.split = sp.bool;
    }
    return opts;
}

test "metaspaceFromPreTokenizer: Metaspace sets prepend+split, Split/ByteLevel do not" {
    var p1 = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always","split":true}
    , .{});
    defer p1.deinit();
    try testing.expectEqual(MetaspaceOpts{ .prepend = .always, .split = true }, metaspaceFromPreTokenizer(p1.value));
    var p2 = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}
    , .{});
    defer p2.deinit();
    try testing.expectEqual(MetaspaceOpts{}, metaspaceFromPreTokenizer(p2.value));
    var p3 = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"never","split":false}
    , .{});
    defer p3.deinit();
    try testing.expectEqual(MetaspaceOpts{}, metaspaceFromPreTokenizer(p3.value));
}

test "encodeSentencePiece: Metaspace prepends ▁ once and splits pieces at ▁ before BPE" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("\xe2\x96\x81a", 1);
    try vocab.put("\xe2\x96\x81", 2);
    try vocab.put("\xe2\x96\x81b", 3);
    try vocab.put("a", 4);
    try vocab.put("b", 5);
    try vocab.put("\xe2\x96\x81a\xe2\x96\x81b", 6);
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "\xe2\x96\x81", .right = "a" }, 0);
    try merge_ranks.put(.{ .left = "\xe2\x96\x81", .right = "b" }, 1);
    try merge_ranks.put(.{ .left = "\xe2\x96\x81a", .right = "\xe2\x96\x81b" }, 2);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    try special_tokens.put("[I]", 9);
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    // Legacy path (no Metaspace): no prefix.
    const legacy = try tok.encodeSentencePiece(allocator, "a b", true);
    defer allocator.free(legacy);
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 3 }, legacy);

    // Prefix without split: BPE merges straight across the ▁ boundary.
    tok.metaspace_prepend = .always;
    const unsplit = try tok.encodeSentencePiece(allocator, "a b", true);
    defer allocator.free(unsplit);
    try testing.expectEqualSlices(u32, &[_]u32{6}, unsplit);

    tok.metaspace_split = true;
    const split = try tok.encodeSentencePiece(allocator, "a b", true);
    defer allocator.free(split);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 3 }, split);
    // A leading space already IS the prefix; a double space yields a lone ▁ (HF: 'a  b' -> ▁a ▁ ▁b).
    const lead = try tok.encodeSentencePiece(allocator, " a  b", true);
    defer allocator.free(lead);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, lead);

    // prepend_scheme "first" (a v0.3 export): only text at offset 0 gets the ▁; HF `[INST]Use` -> `Use`.
    var first = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"first","split":false}
    , .{});
    defer first.deinit();
    const opts = metaspaceFromPreTokenizer(first.value);
    tok.metaspace_prepend = opts.prepend;
    tok.metaspace_split = opts.split;
    const after_special = try tok.encode(allocator, "a[I]a");
    defer allocator.free(after_special);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 9, 4 }, after_special);
}

fn hasByteLevel(pt: std.json.Value) bool {
    if (pt != .object) return false;
    if (pt.object.get("type")) |t| {
        if (t == .string) {
            if (std.mem.eql(u8, t.string, "ByteLevel")) return true;
            if (std.mem.eql(u8, t.string, "Sequence")) {
                if (pt.object.get("pretokenizers")) |pts| {
                    for (pts.array.items) |sub| {
                        if (hasByteLevel(sub)) return true;
                    }
                }
            }
        }
    }
    return false;
}

// ── Tests ──

const testing = std.testing;

test "encode special-token scan: earliest match, longest tie-break, adjacency" {
    // Characterization for the special-token splitter in `encode` — pins the
    // exact semantics (earliest occurrence wins; at the same position the
    // LONGEST special wins; adjacent specials produce no empty segments) so
    // the O(text) bucketed scan that replaced the per-special re-search
    // (12 s per 66 KB prompt on a 6415-special vocab) can't drift.
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("b", 2);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(1, "a");
    try id_to_token.put(2, "b");

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    try special_tokens.put("<tok>", 90);
    try special_tokens.put("<tok>x", 91);
    try special_tokens.put("<s>", 92);
    try special_tokens.put("<u>", 93);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .sentencepiece_bpe,
        .byte_to_unicode = buildBytesToUnicode(),
        .bos_id = null,
        .eos_id = null,
    };

    // Longest tie-break: "<tok>x" and "<tok>" both match at position 1 —
    // the longer one must win.
    {
        const ids = try tok.encode(allocator, "a<tok>xb");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 91, 2 }, ids);
    }
    // Adjacent specials, special at start, trailing text.
    {
        const ids = try tok.encode(allocator, "<s><s>a");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 92, 92, 1 }, ids);
    }
    // Earliest occurrence wins regardless of hashmap iteration order.
    {
        const ids = try tok.encode(allocator, "b<u>a<s>");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 2, 93, 1, 92 }, ids);
    }
    // No specials present at all: whole text is one segment.
    {
        const ids = try tok.encode(allocator, "ab");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, ids);
    }
    // Special token at the very end, nothing after it.
    {
        const ids = try tok.encode(allocator, "a<s>");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 92 }, ids);
    }
}

test "tokenizer parity: encode() matches HF `tokenizers` ids (TOKENIZER_PARITY_MODEL)" {
    // Bar: encode() returns HF `tokenizers`' ids for every case tests/dump_tokenizer_parity_fixtures.py wrote.
    const model = std.c.getenv("TOKENIZER_PARITY_MODEL") orelse return error.SkipZigTest;
    const cases_path = std.c.getenv("TOKENIZER_PARITY_CASES") orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = try loadTokenizer(io, allocator, std.mem.span(model));
    defer tok.deinit();

    const f = try std.Io.Dir.openFileAbsolute(io, std.mem.span(cases_path), .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = try rs.interface.allocRemaining(allocator, .limited(16 << 20));
    defer allocator.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();

    var mismatches: usize = 0;
    for (parsed.value.array.items) |c| {
        const input = c.object.get("text").?.string;
        const want = c.object.get("ids").?.array.items;
        const ids = try tok.encode(allocator, input);
        defer allocator.free(ids);
        var same = ids.len == want.len;
        if (same) for (ids, want) |g, w| {
            if (g != w.integer) same = false;
        };
        if (same) continue;
        mismatches += 1;
        if (mismatches > 10) continue;
        std.debug.print("\n[parity] {s} {f}\n  got  {any}\n  want", .{ c.object.get("name").?.string, std.json.fmt(input, .{ .escape_unicode = true }), ids });
        for (want) |w| std.debug.print(" {d}", .{w.integer});
        std.debug.print("\n", .{});
    }
    const n = parsed.value.array.items.len;
    std.debug.print("\n[parity] {d}/{d} cases match HF tokenizers\n", .{ n - mismatches, n });
    try testing.expectEqual(@as(usize, 0), mismatches);
}

// ── BPE merge characterization tests ──
// Pin the exact greedy semantics of `bpeMerge` (lowest rank first across the
// whole sequence; leftmost wins rank ties; unknown symbols fall back to
// byte-level pieces) so the merge algorithm can be swapped for a faster one
// without changing a single output id.

fn makeBpeTestTokenizer(
    allocator: std.mem.Allocator,
    vocab: *std.StringHashMap(u32),
    merge_ranks: *std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage),
    id_to_token: *std.AutoHashMap(u32, []const u8),
    special_tokens: *std.StringHashMap(u32),
) Tokenizer {
    return Tokenizer{
        .vocab = vocab.*,
        .id_to_token = id_to_token.*,
        .merge_ranks = merge_ranks.*,
        .allocator = allocator,
        .special_tokens = special_tokens.*,
        .tok_type = .sentencepiece_bpe,
        .byte_to_unicode = buildBytesToUnicode(),
        .bos_id = null,
        .eos_id = null,
    };
}

test "bpeMerge: lowest rank merges first regardless of position" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("b", 2);
    try vocab.put("c", 3);
    try vocab.put("d", 4);
    try vocab.put("ab", 5);
    try vocab.put("cd", 6);
    try vocab.put("abcd", 7);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "c", .right = "d" }, 0);
    try merge_ranks.put(.{ .left = "a", .right = "b" }, 1);
    try merge_ranks.put(.{ .left = "ab", .right = "cd" }, 2);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    const ids = try tok.bpeMerge(allocator, "abcd");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{7}, ids);
}

test "bpeMerge: leftmost pair wins rank ties" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("aa", 2);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "a", .right = "a" }, 0);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    // "aaa": leftmost (a,a) merges first -> [aa, a]; (aa,a) has no rank.
    const ids = try tok.bpeMerge(allocator, "aaa");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 1 }, ids);
}

test "bpeMerge: cascading merges across merge sites" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("h", 1);
    try vocab.put("e", 2);
    try vocab.put("l", 3);
    try vocab.put("o", 4);
    try vocab.put("he", 5);
    try vocab.put("ll", 6);
    try vocab.put("hell", 7);
    try vocab.put("hello", 8);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "h", .right = "e" }, 0);
    try merge_ranks.put(.{ .left = "l", .right = "l" }, 1);
    try merge_ranks.put(.{ .left = "he", .right = "ll" }, 2);
    try merge_ranks.put(.{ .left = "hell", .right = "o" }, 3);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    const ids = try tok.bpeMerge(allocator, "hellohello");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 8, 8 }, ids);
}

test "bpeMerge: symbols missing from vocab fall back to byte pieces" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    // 'z' itself is NOT in vocab; its byte-to-unicode form is.
    const z_cp = buildBytesToUnicode()['z'];
    var z_buf: [4]u8 = undefined;
    const z_len = try std.unicode.utf8Encode(z_cp, &z_buf);
    try vocab.put(z_buf[0..z_len], 99);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    const ids = try tok.bpeMerge(allocator, "az");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 99 }, ids);
}

test "ignore_merges: a word the vocab holds whole is one token, whatever the merges reach" {
    // HF BPE `ignore_merges` : such merges never build
    // `_tokens`, its vocab does, and the reference emits the vocab entry.
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    for ([_][]const u8{ "_", "t", "o", "k", "e", "n", "s", "_t", "ok", "ens", "_tokens" }, 0..) |w, i| try vocab.put(w, @intCast(i));
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "_", .right = "t" }, 0);
    try merge_ranks.put(.{ .left = "o", .right = "k" }, 1);
    try merge_ranks.put(.{ .left = "e", .right = "n" }, 2);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    tok.ignore_merges = true;
    const whole = try tok.bpeMerge(allocator, "_tokens");
    defer allocator.free(whole);
    try testing.expectEqualSlices(u32, &[_]u32{10}, whole);

    tok.ignore_merges = false;
    const merged = try tok.bpeMerge(allocator, "_tokens");
    defer allocator.free(merged);
    try testing.expect(merged.len > 1);
}

test "byte_fallback: an unknown character becomes its UTF-8 bytes as <0xNN> tokens; missing byte token -> fused unk" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("<unk>", 3);
    try vocab.put("<0xF0>", 10);
    try vocab.put("<0xA0>", 11);
    try vocab.put("<0x80>", 12);
    try vocab.put("<0x8B>", 13);
    try vocab.put("<0xE1>", 14);
    try vocab.put("<0xA8>", 15);
    // 'z' is NOT in vocab and neither is <0x7A>: unk.
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    tok.byte_fallback = true;
    tok.fuse_unk = true;
    tok.unk_id = 3;

    // U+2000B (CJK ext-B) = F0 A0 80 8B.
    const cjk = try tok.bpeMerge(allocator, "a\xf0\xa0\x80\x8ba");
    defer allocator.free(cjk);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 10, 11, 12, 13, 1 }, cjk);

    // U+1A00 = E1 A8 80 -> bytes; 'z' twice -> ONE fused unk, then a lone unk.
    const mixed = try tok.bpeMerge(allocator, "zz\xe1\xa8\x80z");
    defer allocator.free(mixed);
    try testing.expectEqualSlices(u32, &[_]u32{ 3, 14, 15, 12, 3 }, mixed);

    // Without fuse_unk every unknown char is its own unk.
    tok.fuse_unk = false;
    const unfused = try tok.bpeMerge(allocator, "zz");
    defer allocator.free(unfused);
    try testing.expectEqualSlices(u32, &[_]u32{ 3, 3 }, unfused);
}

test "ByteFallback decoder: <0xNN> runs decode as bytes, invalid runs as U+FFFD per byte, other tokens verbatim" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(1, "\xe2\x96\x81a");
    try id_to_token.put(10, "<0xF0>");
    try id_to_token.put(11, "<0xA0>");
    try id_to_token.put(12, "<0x80>");
    try id_to_token.put(13, "<0x8B>");
    try id_to_token.put(20, "<0x00>");
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);

    // Decoder flag off: the token strings leak through (legacy behaviour).
    const raw = try tok.decode(allocator, &[_]u32{ 1, 10, 11, 12, 13 }, false);
    defer allocator.free(raw);
    try testing.expectEqualStrings(" a<0xF0><0xA0><0x80><0x8B>", raw);

    tok.byte_fallback_decode = true;
    const ok = try tok.decode(allocator, &[_]u32{ 1, 10, 11, 12, 13, 1 }, false);
    defer allocator.free(ok);
    try testing.expectEqualStrings(" a\xf0\xa0\x80\x8b a", ok);

    // Truncated sequence F0 A0 + NUL byte: invalid as a whole -> three U+FFFD.
    const bad = try tok.decode(allocator, &[_]u32{ 10, 11, 20, 1 }, false);
    defer allocator.free(bad);
    try testing.expectEqualStrings("\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd a", bad);

    try testing.expectEqual(@as(?u8, 0xF0), byteTokenValue("<0xF0>"));
    try testing.expectEqual(@as(?u8, null), byteTokenValue("<0xF>"));
    try testing.expectEqual(@as(?u8, null), byteTokenValue("<0xGG>"));
    try testing.expectEqual(@as(?u8, null), byteTokenValue("\xe2\x96\x81a"));
}

test "decoderHasByteFallback: Sequence containing ByteFallback, bare ByteFallback, others" {
    const allocator = testing.allocator;
    const seq = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]}
    , .{});
    defer seq.deinit();
    try testing.expect(decoderHasByteFallback(seq.value));
    const bare = try std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"ByteFallback\"}", .{});
    defer bare.deinit();
    try testing.expect(decoderHasByteFallback(bare.value));
    const bl = try std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"ByteLevel\"}", .{});
    defer bl.deinit();
    try testing.expect(!decoderHasByteFallback(bl.value));
}

test "appendedSpecialId: the special a TemplateProcessing appends after a single sequence" {
    const allocator = testing.allocator;
    var specials = std.StringHashMap(u32).init(allocator);
    defer specials.deinit();
    try specials.put("<|endoftext|>", 151643);
    try specials.put("[CLS]", 101);
    // ByteLevel + TemplateProcessing inside a Sequence.
    const sequence_pp = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"Sequence","processors":[{"type":"ByteLevel"},{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<|endoftext|>","type_id":0}}]}]}
    , .{});
    defer sequence_pp.deinit();
    try testing.expectEqual(@as(?u32, 151643), appendedSpecialId(sequence_pp.value, &specials));
    // A template that only PREPENDS appends nothing.
    const pre = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}]}
    , .{});
    defer pre.deinit();
    try testing.expectEqual(@as(?u32, null), appendedSpecialId(pre.value, &specials));
    const bl = try std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"ByteLevel\"}", .{});
    defer bl.deinit();
    try testing.expectEqual(@as(?u32, null), appendedSpecialId(bl.value, &specials));
}

test "parseMergePair handles both array and space-joined-string formats" {
    const alloc = testing.allocator;
    // Array format (Gemma 4): ["left","right"].
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "[\"AB\",\"cd\"]", .{});
        defer p.deinit();
        const mp = parseMergePair(p.value).?;
        try testing.expectEqualStrings("AB", mp.left);
        try testing.expectEqualStrings("cd", mp.right);
    }
    // String format (GPT-2-style): "left right" split on first space.
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "\"AB cd\"", .{});
        defer p.deinit();
        const mp = parseMergePair(p.value).?;
        try testing.expectEqualStrings("AB", mp.left);
        try testing.expectEqualStrings("cd", mp.right);
    }
    // Malformed → null (no crash): string without a space, short array.
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "\"nospace\"", .{});
        defer p.deinit();
        try testing.expect(parseMergePair(p.value) == null);
    }
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "[\"only\"]", .{});
        defer p.deinit();
        try testing.expect(parseMergePair(p.value) == null);
    }
}

test "reservedOutputIds: specials minus template markers minus eos" {
    // A reserved marker (`<|fim_hole|>` at a collapsed position) in
    // chat output is always a bug; the suppression set is every `special:
    // true` added token MINUS legitimate output. Legitimacy is DERIVED, never
    // hardcoded: EOS/stop ids stay, and so does any special whose literal
    // text appears in the chat template source (thinking tags, tool markers,
    // role markers) — a hardcoded list breaks thinking/tool-calling on the
    // next arch.
    const alloc = testing.allocator;
    const flagged = [_]FlaggedSpecial{
        .{ .id = 10, .content = "<|fim_hole|>" },
        .{ .id = 11, .content = "<|fim_begin|>" },
        .{ .id = 20, .content = "<think>" },
        .{ .id = 21, .content = "</think>" },
        .{ .id = 22, .content = "<tool_call>" },
        .{ .id = 30, .content = "<|endoftext|>" },
        .{ .id = 31, .content = "<|role_end|>" },
    };
    const template = "<role>HUMAN</role>{{ content }}<|role_end|>" ++
        "<role>ASSISTANT</role>\n<think></think>{% if tools %}<tool_call>{% endif %}";
    const eos = [_]u32{30};

    const ids = try reservedOutputIds(alloc, &flagged, template, &eos);
    defer alloc.free(ids);
    // Only the FIM markers survive: think/tool/role markers appear in the
    // template, <|endoftext|> is EOS.
    try testing.expectEqualSlices(u32, &[_]u32{ 10, 11 }, ids);

    // No template at all => suppression disabled entirely (fallback-formatted
    // models keep pre-change behavior; ChatML markers are not in any template
    // source we could consult).
    const none = try reservedOutputIds(alloc, &flagged, "", &eos);
    defer alloc.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);
}

test "reservedOutputIds: every flagged special exempt yields empty set" {
    const alloc = testing.allocator;
    const flagged = [_]FlaggedSpecial{
        .{ .id = 5, .content = "<eos>" },
        .{ .id = 6, .content = "<think>" },
    };
    const ids = try reservedOutputIds(alloc, &flagged, "x<think>y", &[_]u32{5});
    defer alloc.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);
}
