const std = @import("std");
const mlx = @import("mlx.zig");
const keyed_sample = @import("keyed_sample.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const keep_warm = @import("keep_warm.zig");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const rp_mod = @import("reasoning_protocol.zig");
const io_util = @import("io_util.zig");
const pld_index = @import("pld_index.zig");
const ngram_lookup = @import("ngram_lookup.zig");
const drafter_mod = @import("drafter.zig");
const round_cost = @import("round_cost.zig");
const dflash_policy = @import("dflash_policy.zig");
const sys_info = @import("sys_info.zig");
const scheduler_mod = @import("scheduler.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("dflash.zig");
const DflashModel = dflash_mod.DflashModel;

const KVCache = transformer_mod.KVCache;

/// Module-level overrides for prefill behavior. Defaults match the original
/// hardcoded values; main.zig may overwrite these from CLI flags before
/// `serve()` runs. Per-request reads happen on the same thread that did the
/// CLI parse, so no atomicity needed.
pub var prefill_chunk_override: usize = 8192;
/// Default chunk of an arch that builds no prefill score tensor (score width 0):
/// only its MoE gather transient grows with the chunk, and the guard bills that.
pub const WIDE_PREFILL_CHUNK: usize = 16384;

/// Set by `--prefill-chunk` in main.zig. An operator-chosen width outranks the
/// per-model chunk the memory sizer pins (`ModelConfig.pinned_prefill_chunk`);
/// without this flag the default 8192 is indistinguishable from a request.
/// Same set-once-at-CLI-parse contract as `prefill_chunk_override`.
pub var prefill_chunk_explicit: bool = false;

/// The widest chunk this machine runs without `--prefill-chunk` (`server.autoPrefillChunkCap`),
/// set once by main.zig. Every auto width obeys it, the per-request and adaptive rungs included.
pub var machine_prefill_chunk_cap: usize = std.math.maxInt(usize);

pub var prefill_trace_force: bool = false;

/// The width `MLX_SERVE_PREFILL_CHUNK` asked for, or 0. A pinned width also turns the per-chunk adaptive width off.
pub fn envPrefillChunk() usize {
    return readEnvUsize("MLX_SERVE_PREFILL_CHUNK", 0);
}

/// Per-prefill state for the adaptive width; lives on `runPrefill`'s stack. The policy is
/// `server.adaptivePrefillWidth`.
/// The one place a chosen width becomes the running width.
pub fn commitAdaptiveWidth(cur: *usize, st: *AdaptiveWidthState, w: u32) void {
    cur.* = w;
    st.transitions +|= 1;
    st.width_min = @min(st.width_min, w);
    st.width_max = @max(st.width_max, w);
}

pub const AdaptiveWidthState = struct {
    supporting: u8 = 0,
    /// One-way ratchet: a prefill that has stepped down never widens again.
    ratcheted: bool = false,
    transitions: u32 = 0,
    width_min: u32 = 0,
    width_max: u32 = 0,
};

/// Why a request left speculative decoding for the rest of its generation.
pub const SpecDisableReason = enum {
    none,
    acceptance,
    max_ctx,
    /// A thinking budget closed the thought through a plain forward; the spec
    /// state is not resynced, the answer decodes regular.
    think_bound,
    /// A forced tool-call opener went through a plain forward (`ToolForce`).
    tool_choice,
    /// The measured round cost more per token than a measured serial token.
    adaptive,
    /// A speculating slot gained company: it decodes plain so it can join the batched group.
    company,
};

/// One layer's materialized-score budget for unfused-SDPA prefill (see
/// boundedPrefillChunk). 4 GiB keeps the full 8K chunk for every context up
/// to ~16K on 16-head models (further on fewer heads) and degrades gradually
/// to the 512 floor as heads × context grows toward 262K.
pub const PREFILL_SCORES_BUDGET_BYTES: u64 = 4 << 30;
/// Lower bound for the auto-capped prefill chunk; also its rounding grain
/// (repeating chunk sizes let the MLX allocator cache reuse score buffers).
pub const PREFILL_CHUNK_FLOOR: usize = 512;

/// MLX's fused SDPA kernels cover head_dim <= 128. Every Gemma-4 and
/// A checkpoint with head_dim=256, which falls back to the
/// composed path that MATERIALIZES a [heads, chunk, total_kv] bf16 score
/// tensor per layer — at an 8K chunk × 255K ctx × 16 heads that is ~67 GB
/// and an uncatchable Metal command-buffer OOM. Cap the chunk so ONE layer's
/// score tensor stays within PREFILL_SCORES_BUDGET_BYTES at this prompt's
/// FINAL KV length (the last chunk attends to everything). Fused head dims
/// and short contexts return `base_chunk` untouched, so typical traffic
/// keeps full prefill throughput; the cap only bites when heads × total_ctx
/// actually outgrows the budget. Never raises a caller-lowered base.
///
/// DELIBERATELY ignores the msv_attn_p256 fused kernel (unlike
/// prefillEvalCadence / prefillMemoryNeeded, which drop their score term via
/// transformer.prefillHeadDimFused): the fused kernel removes the SCORE
/// transient, but a big chunk still scales the OTHER per-chunk transients
/// (MoE gather buffers, per-chunk KV concat) — measured LIVE on
/// gemma-4-26B-A4B at a 99K prompt: fused @ chunk 8192 = 736 tok/s / 61.2 GB
/// peak vs fused @ chunk 1024 = 712 tok/s / 39.5 GB. +3% speed is not worth
/// +22 GB peak (a 64 GB Mac dies), so the cap stays keyed on raw head_dim.
///
/// `sliding_band_arch` (config.has_sliding_window) picks the policy family:
/// archs WITHOUT sliding-band layers (a gated-delta + full attention hybrid)
/// additionally cap the auto chunk at 2048 — composed-causal prefill
/// measured strictly faster and ~9 GB lighter there (see the inline
/// comment). Gemma keeps the formula-only policy for its fused band layers.
/// `score_head_dim` is the width the SCORE tensor is contracted at
/// (`ModelConfig.prefillScoreHeadDim`) — not necessarily `head_dim`. An MLA arch
/// scores at 192 while storing 128-wide values, and reading `head_dim` there
/// silently exempted the one arch in the tree whose composed path materializes
/// the biggest score tensors. The two hd-256 policy branches below stay keyed
/// on 256 exactly: both were measured on hd-256 checkpoints and neither
/// generalizes (the fused kernel is hd-256-only; the 2048 composed cap was
/// tuned against a 27B's own prefill ladder).
pub fn boundedPrefillChunk(base_chunk: usize, score_head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool) usize {
    const head_dim = score_head_dim;
    if (head_dim <= 128 or n_heads == 0 or total_ctx == 0) return base_chunk;
    // Non-sliding hd-256 archs under FUSED causal (the default since the
    // budgeted-dispatch flip): no score tensor exists, so the scores-budget
    // formula below is moot — and its old shrink (1024 at 64K on 24 heads)
    // starved the dequant+GEMM qmm route, which needs M >= 2048 to engage
    // (the 64K rung was the ladder's weakest for exactly this reason).
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (the gemma-26B@99K lesson: +3% speed for +22 GB peak is a bad trade).
    // DENSE hybrids have no gather transients and a full-size chunk halves
    // the per-chunk dequant sweeps: chunk 8192 measured +1.4% over 4096 at
    // the 8K rung on a dense model (M4 Max, 2026-07-30), flat at 32K.
    // Never raises a caller-lowered base.
    if (head_dim == 256 and !sliding_band_arch and transformer_mod.fused256CausalMode() == .all) {
        // A long-context-gated MoE prices the 8192 rung per request in `chooseRequestPrefillChunk`
        // (+8% prefill for +3 GB peak here); every other MoE keeps the measured 4096 cap.
        return @min(base_chunk, if (is_moe and !long_ctx_gated) @as(usize, 4096) else @as(usize, 8192));
    }
    // Composed-causal fallback (MLX_SERVE_FUSED_256_CAUSAL=0): SMALL chunks
    // measured strictly faster AND lighter on the 27B (2026-07-12 ladder,
    // M4 Max): 8K 225 -> 235.8 tok/s and peak 28.9 -> 19.8 GB at chunk
    // 2048; 32K 205.4 -> 209.3. Chunk boundaries ARE block-level causal
    // skipping for composed attention. Sliding-band archs (gemma) keep big
    // chunks — their local layers run the fused band kernel, which
    // block-skips in-kernel and wants the fewest KV re-walks (26B@99K:
    // 712 tok/s at the formula chunk).
    const causal_cap: usize = if (sliding_band_arch or head_dim != 256) base_chunk else @min(base_chunk, 2048);
    const per_row: u64 = @as(u64, n_heads) * @as(u64, total_ctx) * 2;
    const max_chunk: u64 = PREFILL_SCORES_BUDGET_BYTES / per_row;
    if (max_chunk >= causal_cap) return causal_cap;
    const floored = @max(
        @as(u64, PREFILL_CHUNK_FLOOR),
        max_chunk - (max_chunk % PREFILL_CHUNK_FLOOR),
    );
    return @intCast(@min(floored, @as(u64, causal_cap)));
}

/// The prefill chunk `initWithOptions` will actually use for a request:
/// MLX_SERVE_PREFILL_CHUNK env (explicit tuning knob — honored verbatim,
/// never safety-capped) > --prefill-chunk / default, capped by
/// boundedPrefillChunk. Exported so server.zig's admission guard
/// (checkAttentionMemory) models the SAME chunk the prefill will run with —
/// the guard and the real prefill must not drift.
pub fn effectivePrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool, pinned_chunk: usize) usize {
    const env_chunk = readEnvUsize("MLX_SERVE_PREFILL_CHUNK", 0);
    if (env_chunk > 0) return env_chunk;
    // `pinned_chunk` is the machine-sized cap frozen at load
    // (`ModelConfig.pinned_prefill_chunk`, from `server.resolvePrefillChunk`):
    // the widest chunk whose one-off transient reserve still leaves this box a
    // real KV budget. It NARROWS, never raises, and an explicit
    // `--prefill-chunk` outranks it — the operator asked for a width.
    const launch: usize = if (head_dim == 0 and !prefill_chunk_explicit) WIDE_PREFILL_CHUNK else prefill_chunk_override;
    const base: usize = if (prefill_chunk_explicit)
        launch
    else if (pinned_chunk == 0)
        @min(launch, machine_prefill_chunk_cap)
    else
        @min(launch, pinned_chunk, machine_prefill_chunk_cap);
    return boundedPrefillChunk(base, head_dim, n_heads, total_ctx, sliding_band_arch, is_moe, long_ctx_gated);
}

/// A prefill width under the decode share's cap (0 = no cap).
pub fn decodeShareCapped(width: usize, cap: usize) usize {
    return if (cap == 0) width else @min(width, cap);
}

test "decodeShareCapped: the share cap narrows an explicit and an env width" {
    const saved_override = prefill_chunk_override;
    const saved_explicit = prefill_chunk_explicit;
    defer {
        prefill_chunk_override = saved_override;
        prefill_chunk_explicit = saved_explicit;
    }
    prefill_chunk_override = 8192;
    prefill_chunk_explicit = true;
    const explicit = effectivePrefillChunk(128, 8, 1024, false, false, false, 2048);
    try std.testing.expectEqual(@as(usize, 8192), explicit);
    try std.testing.expectEqual(@as(usize, 1024), decodeShareCapped(explicit, 1024));
    try std.testing.expectEqual(@as(usize, 8192), decodeShareCapped(explicit, 0));

    _ = setenv("MLX_SERVE_PREFILL_CHUNK", "8192", 1);
    defer _ = unsetenv("MLX_SERVE_PREFILL_CHUNK");
    prefill_chunk_explicit = false;
    const env = effectivePrefillChunk(128, 8, 1024, false, false, false, 2048);
    try std.testing.expectEqual(@as(usize, 8192), env);
    try std.testing.expectEqual(@as(usize, 1024), decodeShareCapped(env, 1024));
}

test "effectivePrefillChunk: the machine cap bounds every auto width, an explicit width passes" {
    const saved_cap = machine_prefill_chunk_cap;
    const saved_explicit = prefill_chunk_explicit;
    const saved_override = prefill_chunk_override;
    defer {
        machine_prefill_chunk_cap = saved_cap;
        prefill_chunk_explicit = saved_explicit;
        prefill_chunk_override = saved_override;
    }
    machine_prefill_chunk_cap = 2048;
    prefill_chunk_explicit = false;
    prefill_chunk_override = WIDE_PREFILL_CHUNK;
    // A per-request rung (8192) and an unpinned adaptive cap (0) both stop at the machine's cap.
    try std.testing.expectEqual(@as(usize, 2048), effectivePrefillChunk(256, 24, 16384, false, true, true, 8192));
    try std.testing.expectEqual(@as(usize, 2048), effectivePrefillChunk(256, 24, 16384, false, true, true, 0));
    prefill_chunk_explicit = true;
    prefill_chunk_override = 8192;
    try std.testing.expectEqual(@as(usize, 8192), effectivePrefillChunk(256, 24, 16384, false, true, true, 8192));
}

/// Read an unsigned integer from an environment variable, falling back to
/// `default` when unset, empty, or unparseable. Uses libc getenv to stay
/// allocator-free at call sites.
fn readEnvUsize(name: [:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    return std.fmt.parseInt(usize, slice, 10) catch default;
}

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}

pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
    /// Final-answer grammars normally apply at token 0 (thinking-off and all
    /// prompt-committed content channels). A reasoning protocol instead makes
    /// generation start in the model's channel machinery: either inside an
    /// already-open reasoning block (prompt-opened) or at the unresolved
    /// opener choice. The protocol state drives the phase switches below.
    proto: ?*const rp_mod.Protocol = null,
    pstate: rp_mod.State = .{},
    /// Where the constrained payload begins in the generated stream — the
    /// authoritative reasoning→answer boundary the response emitters route
    /// on. Set exactly once, at the transition; `null` until then.
    pending_span: ?rp_mod.ConstraintSpan = null,
};

/// A forced recovery must leave room for the whole remaining transition AND
/// at least one constrained answer token. At the ordinary completion cap the
/// caller keeps the existing length-stop behavior — no reserve is created.
pub fn forcedBoundaryCanContinue(completion_tokens: u32, max_tokens: u32, transition_tokens: usize) bool {
    return completion_tokens +| @as(u32, @intCast(transition_tokens)) + 1 <= max_tokens;
}

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }

    /// Route this request through a resolved reasoning protocol. Must be
    /// called before generation starts; the protocol is a per-request value
    /// the caller keeps alive (the Constraint borrows it).
    pub fn deferWithProtocol(self: *SchemaConstraint, proto: *const rp_mod.Protocol) void {
        self.constraint.proto = proto;
        self.constraint.pstate = proto.startState();
        if (self.constraint.pstate.phase == .json_body) self.constraint.pending_span = .{ .token_index = 0, .byte_offset = 0 };
    }
};

test "forced recovery leaves room for the whole transition and one answer token" {
    try std.testing.expect(forcedBoundaryCanContinue(0, 2, 1));
    try std.testing.expect(forcedBoundaryCanContinue(8, 10, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 1, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(9, 10, 1));
    // A multi-token transition plans against the WHOLE sequence.
    try std.testing.expect(forcedBoundaryCanContinue(0, 4, 3));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 3, 3));
    // An in-place activation needs no transition token.
    try std.testing.expect(forcedBoundaryCanContinue(0, 1, 0));
}

test "forced recovery is committed and counted exactly once" {
    const src = @embedFile("generate.zig");
    const start = std.mem.indexOf(u8, src, "pub fn forceConstraint" ++ "Transition(") orelse return error.CallSiteMoved;
    const tail = src[start..];
    const end = std.mem.indexOf(u8, tail, "\n    pub fn deinit(") orelse return error.CallSiteMoved;
    const body = tail[0..end];
    // One committed token per call, through the live forward path.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "generated_ids.append(allocator, tok)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "advanceStep(1)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "xfm.forwardWith(&self.ctx, tok_input)"));
}

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

/// A thinking budget enforced while decoding: once `budget` tokens have been
/// generated inside the think block, `forced` (an early-stop sentence, the
/// pack's atomic closer, a separator) is committed through the model and the
/// answer follows. Markers are single token ids, so only packs whose think
/// tags are atomic tokens can be bound; `observe` walks the generated ids
/// incrementally and `due` asks whether the tail is an open block at budget.
pub const ThinkBound = struct {
    budget: u32,
    opener_id: ?u32,
    closer_id: u32,
    forced: []const u32,
    in_think: bool,
    count: u32 = 0,
    cursor: usize = 0,
    fired: bool = false,

    pub fn observe(self: *ThinkBound, ids: []const u32) void {
        while (self.cursor < ids.len) : (self.cursor += 1) {
            const id = ids[self.cursor];
            if (id == self.closer_id) {
                self.in_think = false;
            } else if (self.opener_id != null and id == self.opener_id.?) {
                self.in_think = true;
                self.count = 0;
            } else if (self.in_think) {
                self.count += 1;
            }
        }
    }

    pub fn due(self: *const ThinkBound) bool {
        return !self.fired and self.in_think and self.count >= self.budget;
    }
};

/// `tool_choice` that obliges a call: the family's tool-call opener is
/// committed once the think block closes, or at once when none is open. A
/// thought still open with a quarter of `max_tokens` left is closed for it, so
/// the call fits. `forced` = closer, separator, opener; `opener_at` indexes the opener.
pub const ToolForce = struct {
    closer_id: ?u32,
    forced: []const u32,
    opener_at: usize,
    in_think: bool,
    cursor: usize = 0,
    fired: bool = false,

    pub const Due = struct { tokens: []const u32, closes_thought: bool };

    /// The tokens to commit now, or null.
    pub fn due(self: *ToolForce, ids: []const u32, completion_tokens: u32, max_tokens: u32) ?Due {
        if (self.fired) return null;
        while (self.in_think and self.cursor < ids.len) : (self.cursor += 1) {
            if (ids[self.cursor] == self.closer_id) self.in_think = false;
        }
        if (!self.in_think) return .{ .tokens = self.forced[self.opener_at..], .closes_thought = false };
        if (completion_tokens +| max_tokens / 4 >= max_tokens) return .{ .tokens = self.forced, .closes_thought = true };
        return null;
    }
};

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    min_p: ?f32 = null,
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    /// OpenAI's: subtracted once per earlier occurrence of a token (0.0 = disabled).
    frequency_penalty: f32 = 0.0,
    seed: ?u64 = null,
    /// Draw index under `seed`: every sample takes a fresh key.
    draw: u64 = 0,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
    /// In-stream thinking budget (`ThinkBound`), owned by the request handler
    /// like `constraint`; null = no bound.
    think_bound: ?*ThinkBound = null,
    /// Forced tool-call opener (`ToolForce`), owned like `think_bound`.
    tool_force: ?*ToolForce = null,
    /// Reserved-token suppression mask: `[vocab]` bool, true = the sampler
    /// must never draw this id (reserved specials like `<|fim_hole|>`, which
    /// a degenerate distribution can rank top-5 at a collapsed position — a
    /// reserved marker in chat output is always a bug). Model-lifetime,
    /// OWNED by the Transformer (`suppress_mask`), non-owning here; wired by
    /// `Generator.initWithOptions` so every sampling path inherits it.
    /// Applied by both samplers and both stochastic-verify filters, fully
    /// lazy (`mlx_where` + -inf, no host sync); logprobs deliberately keep
    /// reading the RAW logits — the field reports the model, the mask is
    /// sampling policy. Null = no suppression (kill switch, no-template
    /// models, every non-suppressing arch).
    suppress_mask: ?mlx.mlx_array = null,
    /// Seeded draws go through `keyed_sample`: a token's noise is a hash of
    /// (seed, position, id), so a draft shares the draw its verify row makes
    /// (row-exact archs, `ModelConfig.rowExactDecode`).
    keyed: bool = false,
    /// Absolute position of generated token 0 (the prompt length): a keyed
    /// draw's position is `position_base + draw`.
    position_base: u64 = 0,

    /// Penalties read the realized history, so they keep a request off the
    /// pipelined fast path, spec verify and batched decode.
    pub fn penalized(self: SamplingParams) bool {
        return self.repeat_penalty != 1.0 or self.presence_penalty != 0.0 or self.frequency_penalty != 0.0;
    }

    /// A grammar or a penalty reshapes the logits spec verify compares against.
    pub fn shapesLogits(self: SamplingParams) bool {
        return self.constraint != null or self.penalized();
    }
};

/// Build the `[vocab]` bool suppression mask (true = never sample) on the
/// host, once per model load. Caller owns the returned array.
///
/// Two disjoint sources of "never sample this row":
///   - `ids`: the reserved specials (`tokenizer.reservedOutputIds`).
///   - `[defined_vocab, vocab)`: the checkpoint's PADDING rows. A config's
///     `vocab_size` is a padded matrix dimension, not a vocabulary — the
///     tokenizer's highest defined id + 1 is. The rows in between decode to
///     nothing, so a sample from one emits nothing while consuming a step and
///     poisoning the KV; they were the one class of unsampleable row this
///     mask never covered.
/// `defined_vocab == 0` or `>= vocab` means "no padding known" and adds
/// nothing — a mask that suppressed the whole vocab is `-inf` everywhere,
/// which argmaxes to id 0 (the all-false-mask class, from the other side).
pub fn buildSuppressMask(ids: []const u32, defined_vocab: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const alloc = std.heap.page_allocator;
    const buf = try alloc.alloc(bool, vocab);
    defer alloc.free(buf);
    fillSuppressMask(buf, ids, defined_vocab);
    const shape = [_]c_int{@intCast(vocab)};
    return mlx.mlx_array_new_data(buf.ptr, &shape, 1, .bool_);
}

/// The host half of `buildSuppressMask`, pure so the row set is testable
/// without a GPU. `buf.len` is the mask's vocab.
pub fn fillSuppressMask(buf: []bool, ids: []const u32, defined_vocab: usize) void {
    @memset(buf, false);
    for (ids) |id| {
        if (id < buf.len) buf[id] = true;
    }
    if (defined_vocab > 0 and defined_vocab < buf.len) {
        @memset(buf[defined_vocab..], true);
    }
}

/// Derive + install the reserved-token suppression mask on a freshly-built
/// Transformer (both the serve load path and the CLI run path call this).
/// The id set is `tokenizer.reservedOutputIds`: `special: true` added tokens
/// minus EOS/stop ids minus template-emitted markers. Kill switch
/// `MLX_SERVE_SUPPRESS_RESERVED=0`. Never fails a load — any error logs and
/// leaves the mask null (suppression off), and engagement is one-shot-logged
/// so a silent no-op is visible (the silent-fallback class).
/// `MLX_SERVE_SUPPRESS_RESERVED=0`: reserved specials and padding rows sample like any other id.
pub fn suppressionDisabled() bool {
    const v = std.c.getenv("MLX_SERVE_SUPPRESS_RESERVED") orelse return false;
    return v[0] == '0';
}

pub fn installSuppressMask(xfm: *Transformer, tok: *const Tokenizer, chat_template: []const u8, eos_ids: []const u32) void {
    if (suppressionDisabled()) {
        log.info("[suppress] reserved-token suppression disabled by env\n", .{});
        return;
    }
    const ids = tokenizer_mod.reservedOutputIds(xfm.allocator, tok.flagged_specials, chat_template, eos_ids) catch |err| {
        log.warn("[suppress] derivation failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    const logits_dim: usize = xfm.config.vocab_size;
    // Padding rows past the tokenizer's last defined id.
    const defined_vocab: usize = tok.definedVocabSize();
    const pad_rows: usize = if (defined_vocab > 0 and defined_vocab < logits_dim)
        logits_dim - defined_vocab
    else
        0;
    if (ids.len == 0 and pad_rows == 0) return;
    xfm.suppress_mask = buildSuppressMask(ids, defined_vocab, logits_dim, xfm.s) catch |err| {
        log.warn("[suppress] mask build failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    log.info(
        "[suppress] {d} of {d} flagged specials masked from sampling (template + eos exempt); {d} padding rows past id {d} of {d}\n",
        .{ ids.len, tok.flagged_specials.len, pad_rows, defined_vocab, logits_dim },
    );
}

/// `out = where(mask, -inf, logits)` — the masked lanes get EXACTLY -inf
/// (never an additive -inf, whose 0×-inf/NaN edge the parity rules exist
/// for). `[V]` broadcasts over both `[1, V]` and `[1, 1, V]` logits.
fn applySuppressMask(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(out, mask, neg_inf, logits, s));
}

/// A scalar in the array's own dtype, so a masked bf16 block is not widened to
/// f32 by its own `-inf`.
fn scalarLike(v: f32, like: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(like), s));
    return out;
}

/// A greedy spec-verify argmax whose entries are COMMITTED as next tokens.
/// `verifyArgmax` is its only constructor and every commit site takes this type,
/// so a raw `mlx_argmax_axis` cannot reach a commit without the reserved-id mask.
pub const CommittedArgmax = struct {
    arr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *CommittedArgmax) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }

    pub fn present(self: CommittedArgmax) bool {
        return self.arr.ctx != null;
    }

    /// The lazy handle, for an eval vector or a forced eval - never for a commit.
    pub fn lazy(self: CommittedArgmax) mlx.mlx_array {
        return self.arr;
    }

    /// The realized ids; the caller must have evaluated `lazy()` first.
    pub fn ids(self: CommittedArgmax, len: usize) ![]const i32 {
        const data = mlx.mlx_array_data_int32(self.arr) orelse return error.MlxArrayDataNull;
        return data[0..len];
    }
};

/// The greedy spec-verify argmax, over the reserved-id mask when the model has
/// one. The sampled verify filters through `applySuppressMask` and the rerank
/// draft select takes the mask, so without this the greedy verify was the one
/// path that could commit a flagged special or a padding row past the defined
/// vocabulary — ids the serial sampler can never draw.
fn verifyArgmax(logits: mlx.mlx_array, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !CommittedArgmax {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (mask) |m| {
        const neg_inf = try scalarLike(-std.math.inf(f32), logits, s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        try mlx.check(mlx.mlx_argmax_axis(&out, masked, 2, false, s));
    } else {
        try mlx.check(mlx.mlx_argmax_axis(&out, logits, 2, false, s));
    }
    return .{ .arr = out };
}

/// A speculative decode step (PLD / drafter / DFlash) cannot honor a grammar
/// constraint — the drafts bypass the per-token grammar mask — nor per-token
/// logprobs, which the draft path never captures. The scheduler and server
/// already select a spec mode only when both are absent, but `nextPld` /
/// `nextDrafter` / `nextDflash` only *documented* that with `std.debug.assert`,
/// which compiles out in the ReleaseFast build users actually run (issue #97).
/// A future dispatch bug that let a constrained request reach a spec step would
/// then read the constrained branch's placeholder token as a real commit and
/// stream silently off-schema output as a normal 200. So the spec steps gate on
/// this real, release-enforced check and return `error.SpecDecodeUnsupported`
/// instead of asserting.
fn specDecodeUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return sampling.constraint != null or logprobs_n != 0;
}

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    /// Wall-clock nanoseconds spent on prefill (prompt processing).
    prefill_ns: u64 = 0,
    /// Wall-clock nanoseconds spent on decode (token generation).
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix (hot prefix cache for MLX,
    /// persistent-session prefix reuse for llama). `prompt_tokens - cached_tokens`
    /// is what was actually run through the model this turn, so `prefill_tps`
    /// reflects real compute rather than an inflated full-prompt rate.
    cached_tokens: u32 = 0,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
    /// `/v1/completions` echo: prompt tokens 1..N-1, each scored by the row before it (caller must free).
    prompt_logprobs: ?[]LogprobResult = null,
    /// Non-null only when the degenerate-tail guard cut this generation:
    /// the `finish_details.type` value emitted beside `finish_reason`
    /// ("stop" for loop cuts — see `scheduler.loopStopReason`).
    /// Static string; nothing to free.
    finish_details: ?[]const u8 = null,
    /// Absolute byte offset into `text` where the constrained JSON payload
    /// begins, when a reasoning protocol was active and the payload began.
    /// 0 = direct answer. Null when no protocol ran, the payload never began
    /// (completion cap mid-reasoning), or the boundary fell outside the
    /// emitted (loop-trimmed) tokens.
    constraint_payload_byte: ?usize = null,
};

/// Throughput in tokens/sec. Returns 0 when no time elapsed so unmeasured paths
/// report 0 rather than inf / NaN.
pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    const tok_f: f64 = @floatFromInt(tokens);
    const ns_f: f64 = @floatFromInt(elapsed_ns);
    return tok_f * @as(f64, @floatFromInt(std.time.ns_per_s)) / ns_f;
}

/// True prefill compute throughput: divides by the tokens actually pushed through
/// the model (prompt minus the prefix served from KV cache). A near-full cache
/// hit therefore reports the small suffix's real rate, not an inflated
/// full-prompt number. With `cached_tokens == 0` this is just the full-prompt
/// rate, matching the pre-instrumentation behavior.
pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    const uncached: u32 = if (prompt_tokens > cached_tokens) prompt_tokens - cached_tokens else 0;
    return tokensPerSec(uncached, prefill_ns);
}

/// Pick the end position of the next prefill chunk starting at `pos`.
///
/// Base behavior: advance by `default_chunk` (the memory-bound `PREFILL_CHUNK`),
/// clamped to `prefix_len`.
///
/// Pulled out of the prefill loop so the chunk-count behavior is unit-testable:
/// on large MoE models each extra chunk re-streams the (huge) expert weights
/// from HBM, the dominant cold-prefill cost.
/// A trailing remainder smaller than this merges into the preceding chunk
/// instead of becoming its own chunk. Chat-templated prompts routinely land a
/// token or two past a chunk multiple (an "8192-token" prompt tokenizes to
/// 8193); a 1-token final chunk pays a full graph build + eval barrier +
/// cache clear for one token. The merged chunk's attention-score transient
/// grows by at most TAIL_MERGE_MAX/default_chunk (~6% at 8192) — within the
/// score-budget slack `boundedPrefillChunk` already carries.
pub const TAIL_MERGE_MAX: usize = 512;

/// The tail a chunk of `default_chunk` tokens may absorb where the width is chosen per chunk:
/// an eighth of the width keeps the flat 512's ~6% bound at every rung (at the ladder floor
/// 512 is +100% of the transient a step-down just bought). Reach it through `tailMergeMaxFor`.
pub fn tailMergeMax(default_chunk: usize) usize {
    return @min(TAIL_MERGE_MAX, @max(default_chunk / 8, 1));
}

/// The tail bound the chunk loop uses: scaled only under the per-chunk adaptive width, the
/// flat constant everywhere else (chunk boundaries are not byte-stable).
pub fn tailMergeMaxFor(default_chunk: usize, adaptive_width: bool) usize {
    return if (adaptive_width) tailMergeMax(default_chunk) else TAIL_MERGE_MAX;
}

pub fn nextChunkEnd(
    pos: usize,
    prefix_len: usize,
    default_chunk: usize,
    // Is the per-chunk adaptive width live for this prefill? Only then does the merge bound scale.
    adaptive_width: bool,
) usize {
    var end = @min(pos + default_chunk, prefix_len);
    if (end < prefix_len and prefix_len - end < tailMergeMaxFor(default_chunk, adaptive_width)) {
        // Absorb a tiny tail instead of paying a full graph build + eval
        // barrier for a few tokens.
        end = prefix_len;
    }
    return end;
}

/// Tokens of KV capacity this prefill reserves up front (#353). A multi-chunk prompt reserves
/// itself plus one chunk of slack on every arch, since each chunk-by-chunk grow copies the whole
/// cache; generation headroom is pre-bought only under `longCtxGated`. `MLX_SERVE_KV_RESERVE=0`
/// turns it off.
pub fn reservedPrefillTokens(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u64,
    chunk: u64,
) u64 {
    const prompt = if (seq > chunk) seq +| chunk else 0;
    if (!config.longCtxGated()) return prompt;
    return @max(prompt, transformer_mod.KVCache.reservedTokens(seq, max_tokens, chunk, config.max_position_embeddings));
}

/// Chunked vision prefill (issue #197). Default ON: the splice resumes its
/// row index across chunk boundaries via `ForwardCtx.vision_splice_offset`,
/// so an image-bearing prompt prefills under the same chunk-bounded memory
/// envelope as text. MLX_SERVE_VISION_CHUNKED=0 restores the whole-prompt
/// single forward (and the memory guard's full-width bill with it).
var vision_chunked_cached: ?bool = null;
pub fn visionChunkedPrefillEnabled() bool {
    if (vision_chunked_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_VISION_CHUNKED");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    vision_chunked_cached = on;
    return on;
}

/// What the memory guard should bill a prefill at: full width only for a
/// vision prompt with chunking killed. The guard and this loop must agree or
/// admission either over-refuses (bills seq for a chunked prefill) or
/// under-bills (uncatchable Metal OOM).
pub fn visionPrefillUnchunked(has_vision: bool) bool {
    return has_vision and !visionChunkedPrefillEnabled();
}

/// Placeholder tokens (vision + audio soft tokens) in `ids` — the number of
/// source rows a chunk's splice consumes. Host-side count, no GPU sync.
pub fn countSpliceRows(ids: []const i32, image_token_id: u32, audio_token_id: u32, video_token_id: u32) usize {
    var n: usize = 0;
    for (ids) |id| {
        if ((image_token_id > 0 and id == @as(i32, @intCast(image_token_id))) or
            (audio_token_id > 0 and id == @as(i32, @intCast(audio_token_id))) or
            (video_token_id > 0 and id == @as(i32, @intCast(video_token_id)))) n += 1;
    }
    return n;
}

/// Number of chunks a cold prefill of `prefix_len` tokens splits into for the
/// given chunk size. Mirrors the loop in `init` exactly
/// (drives the same `nextChunkEnd`), so a test on this is a faithful proxy for
/// the real prefill chunk count. Each chunk on a memory-bound MoE prefill
/// re-streams the expert weights, so this is effectively the cold-prefill
/// weight-traffic multiplier.
pub fn prefillChunkCount(
    prefix_len: usize,
    default_chunk: usize,
    adaptive_width: bool,
) usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len) {
        const end = nextChunkEnd(pos, prefix_len, default_chunk, adaptive_width);
        pos = end;
        n += 1;
    }
    return n;
}

/// Generated tokens between two returns of the decode loop's transients to
/// MLX's allocator. 256 is what the non-speculative paths have always used.
pub const CACHE_CLEAR_INTERVAL: u32 = 256;

/// PURE: has `interval` tokens passed since the last clear?
///
/// Interval arithmetic, not `step % interval == 0`: a spec-decode round emits
/// `1 + accepted` tokens, so a modulo test can step clean over every multiple
/// and never fire (issue #110 — at stride 5 it fires zero times in the first
/// 1024 steps). For the stride-1 paths this is byte-identical to the modulo
/// form it replaces.
pub fn shouldClearAllocatorCache(step: u32, last_clear: u32, interval: u32) bool {
    if (interval == 0) return false;
    return step -| last_clear >= interval;
}

/// PURE: does this request return MLX's allocator pool once, between its last
/// prefill chunk and its first decode tick?
///
/// The per-chunk clear runs before the last chunks' transient is freed, so that
/// transient parks in the pool up to the cap and the first decode tick allocates
/// on top of it (8.1 GB parked at the first tick of a 393k prefill). Gated on
/// `longCtxGated`: every other arch keeps its previous call pattern exactly.
pub fn clearsPoolAtPrefillEnd(config: *const model_mod.ModelConfig) bool {
    return config.longCtxGated();
}

/// Number of accepted draft tokens that may accompany the always-committed
/// anchor token without crossing the request's output-token ceiling.
pub fn capAcceptedForTokenBudget(accepted: u32, completion: u32, max_tokens: u32) u32 {
    const remaining = max_tokens -| completion;
    if (remaining <= 1) return 0;
    return @min(accepted, remaining - 1);
}

test "capAcceptedForTokenBudget keeps speculative commits inside max_tokens" {
    try std.testing.expectEqual(@as(u32, 15), capAcceptedForTokenBudget(15, 0, 400));
    try std.testing.expectEqual(@as(u32, 2), capAcceptedForTokenBudget(15, 397, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 399, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 400, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 401, 400));
}

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
/// A request's round-policy state (`dflash_policy.zig`).
const DflashPol = struct {
    calib: dflash_policy.Calibration = .{},
    drafting: dflash_policy.Drafting = .{},
    /// Verify exactly this many rows (t1 included) whenever the draft has them; 0 = the policy chooses.
    /// `MLX_SERVE_DFLASH_ROWS`: how width costs and accepts are measured.
    pin_rows: u32 = 0,
    /// What a plain tick costs on this machine at this context, from this request's own pipelined ticks.
    /// The shared round-cost table is no source: its plain cell folds transition ticks and, once stale,
    /// takes one outlier at half weight.
    plain: dflash_policy.PlainCost = .{},
    /// How copied continuations have fared.
    copy: dflash_policy.CopyStats = .{},
    /// Since the last pipelined plain tick ended; null once any other kind of round has run.
    plain_watch: ?io_util.Stopwatch = null,
};

/// A draft tree's host table for rows with `parents` (-1 = the root): the parents,
/// each row's four conv-input sources oldest first (< 3 a conv-state row, else
/// 3 + a window row), resolved on the host once instead of per channel, then
/// which rows a later node restarts from.
fn treeTable(parents: []const i32, out: []i32) void {
    const n = parents.len;
    std.debug.assert(out.len == 6 * n);
    @memcpy(out[0..n], parents);
    @memset(out[5 * n ..], 0);
    for (parents, 0..) |p, t| {
        if (p >= 0 and p != @as(i32, @intCast(t)) - 1) out[5 * n + @as(usize, @intCast(p))] = 1;
    }
    for (0..n) |t| {
        var depth: i32 = 0;
        var r: i32 = @intCast(t);
        while (parents[@intCast(r)] >= 0) : (r = parents[@intCast(r)]) depth += 1;
        for (0..4) |tap| {
            const up: i32 = 3 - @as(i32, @intCast(tap));
            const dd = depth - up;
            if (dd < 0) {
                out[n + t * 4 + tap] = dd + 3;
                continue;
            }
            var a: i32 = @intCast(t);
            var k = up;
            while (k > 0) : (k -= 1) a = parents[@intCast(a)];
            out[n + t * 4 + tap] = 3 + a;
        }
    }
}

test "treeTable: a chain reads t + tap, a branch reads its own ancestors" {
    var out: [6 * 4]i32 = undefined;
    treeTable(&.{ -1, 0, 1, 2 }, &out);
    for (0..4) |t| for (0..4) |tap| try std.testing.expectEqual(@as(i32, @intCast(t + tap)), out[4 + t * 4 + tap]);
    var tree: [6 * 3]i32 = undefined;
    treeTable(&.{ -1, 0, 0 }, &tree);
    // row 2 (depth 1, parent 0): conv rows 1, 2, then window rows 0 and 2.
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 5 }, tree[3 + 8 .. 3 + 12]);
}

pub const Generator = struct {
    /// The verify forward's inputs for a draft tree of `parents` (row 0 the root,
    /// -1) at `depth`: the tree table and each row's ancestor path.
    pub fn specTreeFor(parents: []const i32, depth: []const i32, max_depth: i32) transformer_mod.SpecTree {
        const MAX_W = 16;
        const w = parents.len;
        const maxd: usize = @intCast(max_depth + 1);
        var path: [MAX_W * MAX_W]i32 = @splat(0);
        for (0..w) |r| {
            var cur: i32 = @intCast(r);
            var d = depth[r];
            while (cur >= 0) : (d -= 1) {
                path[r * maxd + @as(usize, @intCast(d))] = cur;
                cur = parents[@intCast(cur)];
            }
        }
        const wc: c_int = @intCast(w);
        var table: [6 * MAX_W]i32 = undefined;
        treeTable(parents, table[0 .. 6 * w]);
        return .{
            .parents = mlx.mlx_array_new_data(&table, &[_]c_int{6 * wc}, 1, .int32),
            .attn = .{
                .depth = mlx.mlx_array_new_data(depth.ptr, &[_]c_int{wc}, 1, .int32),
                .path = mlx.mlx_array_new_data(&path, &[_]c_int{ wc, @intCast(maxd) }, 2, .int32),
                .max_depth = max_depth,
            },
        };
    }

    xfm: *Transformer,
    /// Forward-pass context. Stores per-request KVCache pointer, moe_seq_offset
    /// pointer, ssm_entries slice, vision_embeddings handle, and capture_hidden
    /// override. The legacy single-slot path uses `xfm.defaultCtx()` (pointing at
    /// the Transformer's own fields). Phase 2 concurrent batching constructs a
    /// per-slot ForwardCtx pointing at the slot's own KVCache, etc., so multiple
    /// generators can share one Transformer's weights without colliding on
    /// per-request state. Stored by value; `&self.ctx` is what we pass to
    /// `xfm.forwardWith` / `lazyForward` / drafter step.
    ctx: ForwardCtx,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    /// `step` at the last `mlx_clear_cache()`. Advanced only by `advanceStep`,
    /// which is the ONE place `step` may move.
    last_cache_clear_step: u32 = 0,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    /// Degenerate-tail detection must not reconsider a reasoning loop after
    /// its boundary has activated the final-answer grammar. The scheduler
    /// scans only this suffix; a later loop in the answer is still handled by
    /// the ordinary repetition-stop policy.
    loop_guard_start: usize = 0,
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout; measures SILENCE, not total time (see StallClock)
    stall: StallClock = .{},
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    /// Logprobs for the token the NEXT `next()` call will return.
    ///
    /// The decode loop returns `next_token_id` and, in the same call, forwards
    /// it to sample its successor — so the result `sampleToken` hands back
    /// describes the token AFTER the one being returned. Publishing it as
    /// `last_logprob` there shifted the whole array by one: the caller zips
    /// token_ids with logprobs by index, so at temp 0 every entry reported the
    /// distribution of the token that FOLLOWED it (a one-token "OK" reply came
    /// back with `<|role_end|>` at rank 1). Seeded at init with t1's own
    /// distribution — t1 is sampled from the prefill's final forward, which
    /// this loop never sees.
    pending_logprob: ?LogprobResult = null,
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    // ── Spec-decode shared state (PLD + drafter) ──
    // Post-final-norm hidden state at the last produced token's position.
    // Owned by the Generator (freed in `deinit`). Captured by
    // `forwardCaptureHidden` during prefill final-token forward and every
    // verify forward — used by drafter as h_prev seed and by PLD verify
    // partial-accept rollback.
    last_hidden: mlx.mlx_array = .{},
    has_last_hidden: bool = false,
    /// PRNG for PLD / drafter stochastic-verify accept test (probability-
    /// ratio requires a uniform draw per draft step). Seeded from
    /// `sampling.seed` when set, otherwise from system time at init.
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    // ── PLD (Prompt Lookup Decoding) state ──
    // Owned copy of the input prompt ids — needed because PLD's n-gram lookup
    // table is `prompt + generated_ids`, and the caller-supplied `prompt_ids`
    // slice is freed after `init` returns. `generated_ids` (above) tracks
    // post-prefill tokens; `prompt_ids_owned` is the immutable prefix.
    prompt_ids_owned: []u32 = &.{},
    /// Allocator that owns `prompt_ids_owned`. Stored so `deinit` can free it
    /// without requiring callers to thread the allocator a second time. (Other
    /// owned slices are freed via the `allocator` argument to `deinit` for
    /// historical reasons; this one is set during `initWithOptions`.)
    prompt_ids_alloc: ?std.mem.Allocator = null,
    /// Did init actually arm PLD for this generator (`InitOptions.pld_enabled`)?
    /// `nextPld` declines to the
    /// plain serial step when false, and the scheduler's tick dispatch
    /// (`specTickMode`) requires it alongside `slot.enable_pld` — the caller's
    /// flag alone must never put a verify forward through the trunk. This is
    /// PLD's counterpart of the `gen.drafter != null` conjunct;
    /// PLD has no model handle, so the bit has to be explicit.
    pld_enabled: bool = false,
    /// Stats for PLD benchmark logging. `pld_attempted` counts every step
    /// where lookup found a candidate (so a verify forward ran);
    /// `pld_accepted_tokens` is the cumulative number of *drafted* tokens
    /// (not including the always-accepted t1) that were successfully verified.
    pld_attempted: u64 = 0,
    pld_accepted_tokens: u64 = 0,

    // ── Gemma 4 assistant drafter state ──
    // External drafter model (cross-attends into target's KV). When
    // `drafter != null`, callers use `nextDrafter` instead of `next`. The
    // drafter is owned by the server (loaded once at startup); the Generator
    // only holds a non-owning pointer.
    drafter: ?*DrafterModel = null,
    /// Number of tokens proposed per round (= drafter forwards + 1 verify token).
    /// Defaults to 4 (3 drafter steps + 1 t1 prepend → length-4 verify).
    drafter_block_size: u32 = 4,
    /// Stats: count of nextDrafter calls that ran a verify forward.
    drafter_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    drafter_accepted_tokens: u64 = 0,

    // ── DFlash block-drafter state ──
    // External block-parallel assistant (src/dflash.zig). When
    // `dflash != null`, callers use `nextDflash` instead of `next`. Weights
    // are owned by the server (per-model); the per-request context cache
    // (`dflash_ctx`) is OWNED by the Generator — built during prefill from
    // the trunk's capture_layers hiddens, freed in `deinit`.
    dflash: ?*DflashModel = null,
    dflash_ctx: ?dflash_mod.DflashCtx = null,
    /// Effective block size (assistant config, clamped by --draft-block-size).
    /// Drafts per round = dflash_block_size - 1.
    dflash_block_size: u32 = 0,
    /// Request-class break-even yield, normalized by the scheduler to the
    /// effective draft width. Thinking requests use the lower calibration
    /// because their less-predictable reasoning preamble pays back later.
    dflash_min_accepted_per_round: f32 = DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    /// Per-round width chooser over the model's round-cost table (null =
    /// MLX_SERVE_DFLASH_CHOOSER=0, the fixed block + sticky yield gate).
    dflash_chooser: ?round_cost.WidthChooser = null,
    /// Drafts this round ran (0 = serial), read by the table feed.
    dflash_round_width: u32 = 0,
    /// Stats: count of nextDflash calls that ran a verify forward.
    dflash_attempted: u64 = 0,
    /// Round policy (`dflash_policy.zig`); null = the fixed block and the sticky yield gate.
    dflash_pol: ?DflashPol = null,
    /// Stats: rounds drafted from the context (no assistant forward), the tokens they proposed and the ones kept.
    dflash_copies: u64 = 0,
    dflash_copy_drafted: u64 = 0,
    dflash_copy_accepted: u64 = 0,
    /// Stats: rounds that verified one row (no drafts), whether or not the assistant drafted for them.
    dflash_plain: u64 = 0,
    /// The round's wall time bills a draft nobody verified, so it says nothing about the width's cost.
    dflash_unobserved: bool = false,
    /// Per-position acceptance and verify widths (`[spec-stats]`).
    dflash_stats: dflash_policy.PositionStats = .{},
    /// Plain-round captures not yet appended to the assistant context: one array per target layer
    /// per round, oldest first. One append later costs what one token's does, so a plain stretch pays none.
    dflash_pending: std.ArrayList(mlx.mlx_array) = .empty,
    dflash_pending_rounds: u32 = 0,
    /// Tree rounds that verified a context copy: consecutive ones that kept
    /// none of it, and rounds left without copies after four of those.
    dflash_copy_misses: u8 = 0,
    dflash_copy_silent: u8 = 0,
    /// Emitted tokens past the KV cache: the last one, when the budget ended on
    /// a token the previous verify already decided. Commits key on the rest.
    unforwarded_tail: u32 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    dflash_accepted_tokens: u64 = 0,
    /// Per-phase wall-time trace (MLX_SERVE_DFLASH_TRACE=1; else untouched).
    /// This trace INSERTS eval barriers to attribute a
    /// fully-lazy round — a traced round is slower than a real one by
    /// whatever overlap the barriers destroy. Diagnostic only.
    dflash_trace: DflashTrace = .{},
    /// Trace-only stopwatch across the scheduler gap (round return → next
    /// round entry); bills the `.gap` phase. Null when not tracing.
    dflash_gap_watch: ?io_util.Stopwatch = null,

    /// Prompt + committed output, indexed up to `ngram_gen` generated tokens.
    ngram_idx: ?ngram_lookup.Index = null,
    ngram_gen: usize = 0,
    /// Speculative rounds that rolled recurrent state back on a partial accept.
    partial_rounds: u64 = 0,
    /// Round-cost surface selected once for this generator.
    spec_ev_costs: SpecEvCosts = SPEC_EV_DEFAULT_COSTS,
    /// Drafts of the previous TWO rounds (0 = serial), every round: a round
    /// is a transition unless both predecessors ran its width (the first
    /// round after a width change is the transition, the second is still
    /// elevated — live, every w4 sample taken beside a trial read 17% high
    /// on the M4 base and fed the churn that produced it).
    spec_round_prev_width: ?u32 = null,
    spec_round_prev_width2: ?u32 = null,
    /// Wall clock between round ends: the regime compares the quantity
    /// tok/s reports, so per-round work OUTSIDE the round stopwatch (token
    /// publish, stop checks, scheduler) must be in the denominator — fewer
    /// rounds is part of a wider shape's win.
    spec_round_clock: ?io_util.Stopwatch = null,
    /// True while this generator is the ONLY decoding stream. Contention is
    /// strictly one-sided — it only ADDS time — so a busy server simply
    /// stops feeding the kv-term learner rather than teaching it a lie.
    /// Set per tick by the scheduler.
    spec_cost_solo: bool = true,

    // ── Runtime acceptance gate ──
    // Set to true mid-request when the per-request acceptance rate
    // (`*_accepted_tokens / *_attempted`) falls below
    // `RUNTIME_GATE_MIN_RATE` after `RUNTIME_GATE_WARMUP` attempts. When set,
    // `nextPld`, `nextDrafter`, and `nextDflash` short-circuit to `next()` for the
    // remainder of the request — the prompt-time gate could not foresee that
    // the workload's *runtime* draft acceptance rate wasn't paying for the
    // per-step verify overhead. The flag is sticky for the rest of the
    // generation; we never re-enable speculation within a single request.
    spec_disabled_runtime: bool = false,
    /// Why `spec_disabled_runtime` was set, for `[spec-stats]`.
    spec_disable_reason: SpecDisableReason = .none,
    /// Yield-gate counters: enabled-mode `nextPld` steps and drafted tokens
    /// accepted since the last (re-)enable. Reset on mid-request re-enable so
    /// a fresh workload region (e.g. file echo after a novel preamble) gets a
    /// fresh economic evaluation instead of inheriting the bad early yield.
    yield_steps: u64 = 0,
    yield_accepted: u64 = 0,
    /// Steps spent in disabled mode since the gate tripped (drives the
    /// periodic `specShouldReenable` re-check).
    disabled_steps: u64 = 0,
    /// Number of attempts before the runtime gate considers disabling.
    /// Below this we trust the prompt-time gate.
    ///
    /// Override at runtime via `SPEC_GATE_WARMUP` env var (parsed in `runtimeGateWarmup()`
    /// once per request). Lower values make the gate trip sooner,
    /// reducing regression-tail damage at the cost of fewer chances for slow-warmup
    /// workloads to amortize spec overhead.
    pub const RUNTIME_GATE_WARMUP: u64 = 5;

    /// Read the warmup threshold for this call. Env-overridable so we can A/B
    /// without rebuilding. Anything outside `[1, 64]` falls back to the default.
    pub fn runtimeGateWarmup() u64 {
        const n = readEnvUsize("SPEC_GATE_WARMUP", @intCast(RUNTIME_GATE_WARMUP));
        if (n < 1 or n > 64) return RUNTIME_GATE_WARMUP;
        return @intCast(n);
    }
    /// Minimum per-draft acceptance probability. Below this after warmup,
    /// speculation is disabled for the rest of the request.
    ///
    /// History: pre-v5 this gate compared `accepted/attempted` (per-round
    /// average) against 0.30 — but with `block_size=4` the max value of that
    /// ratio is 3.0, so the 0.30 threshold corresponded to ~10% per-draft
    /// probability, well below where verify+draft overhead actually breaks
    /// even. Empirically creative-content workloads regress at 22-47% per-draft
    /// acceptance
    /// while the gate stayed off (per-round avg 0.66-1.58, all above 0.30).
    /// Switching to a per-draft probability with threshold 0.50 cleanly cuts
    /// off the regressing tail while leaving heavy-echo workloads (84-97%
    /// per-draft) running unmolested.
    pub const RUNTIME_GATE_MIN_PER_DRAFT_RATE: f32 = 0.50;

    /// Pure helper: should the runtime gate disable speculation given the
    /// observed per-request stats? `drafts_per_round` is the number of
    /// drafted tokens proposed in each verify (= `block_size - 1` for the
    /// drafter, or `pld_draft_len` for PLD); we divide accepts by attempts ×
    /// drafts_per_round to get the per-draft acceptance probability.
    /// Returns true iff `attempted >= warmup` AND per-draft probability is
    /// below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`.
    ///
    /// `drafts_per_round == 0` is treated as "no speculative work happens
    /// per round" → never trip (defensive — current callers always pass
    /// >= 1).
    pub fn runtimeGateShouldDisable(attempted: u64, accepted: u64, drafts_per_round: u32) bool {
        if (attempted < runtimeGateWarmup()) return false;
        if (drafts_per_round == 0) return false;
        const drafts_proposed = attempted * @as(u64, drafts_per_round);
        const rate = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(drafts_proposed));
        return rate < RUNTIME_GATE_MIN_PER_DRAFT_RATE;
    }

    // DFlash has different economics from sequential-token drafters: one
    // assistant call proposes the whole block and the M5 NAX verify lane makes
    // wide blocks cheap. Per-draft percentage therefore penalizes block 16
    // even when it is decisively profitable. The stable cross-width signal is
    // accepted drafts per verify round. M5 sweeps put code/tool traffic at
    // 4.4-15.0 accepted/round and the regressing prose/vision class at
    // 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
    // workloads accepting 62-86% remain above it as well.
    // Warmup 8, not 3 (2026-09-03): three rounds tripped on ONE dry opening
    // (llmprobe's 16k rung: 0/3, drafter off for the whole request at 25
    // tok/s; the same prompt restored from cache decoded at 73).
    pub const DFLASH_GATE_WARMUP: u64 = 8;
    pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 2.0;
    pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.0;
    /// Absolute floor for a SPARSE target, applied after the width scaling.
    /// See `scheduler.dflashGateMinimum` for the measurement it comes from —
    /// a MoE verify reads every expert its block's positions route to, so its
    /// break-even acceptance is several times a dense trunk's.
    pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.8;

    /// Width chooser — OPT-IN (MLX_SERVE_DFLASH_CHOOSER=1) until measured on
    /// the five peer cells; default is the fixed block + sticky yield gate.
    var dflash_chooser_cache: ?bool = null;
    fn dflashChooserEnabled() bool {
        if (dflash_chooser_cache) |v| return v;
        const raw = std.c.getenv("MLX_SERVE_DFLASH_CHOOSER");
        const on = raw != null and std.mem.eql(u8, std.mem.span(raw.?), "1");
        dflash_chooser_cache = on;
        return on;
    }

    pub fn dflashGateWarmup() u64 {
        const n = readEnvUsize("DFLASH_GATE_WARMUP", @intCast(DFLASH_GATE_WARMUP));
        if (n < 1 or n > 64) return DFLASH_GATE_WARMUP;
        return @intCast(n);
    }

    pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, min_avg: f32) bool {
        if (attempted < dflashGateWarmup()) return false;
        const avg = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(attempted));
        return avg < min_avg;
    }

    // ── PLD yield gate (cold-path economics) ──
    // The per-draft gate above only counts verify ROUNDS, so a workload where
    // the n-gram lookup rarely matches never accumulates enough "attempts" to
    // trip it — yet every no-match step pays PLD's unpipelined cold forward
    // (measured −14% vs the async-pipelined `next()` on creative content).
    // The yield gate instead counts EVERY enabled-mode nextPld step: if the
    // speculation is yielding fewer than YIELD_GATE_MIN_YIELD extra (drafted,
    // accepted) tokens per step after YIELD_GATE_WARMUP steps, the cold-path
    // tax outweighs the wins → disable. Paired with `specShouldReenable`,
    // which flips PLD back on when the generated tail turns repetitive.
    // Warmup 32 (not higher): the re-enable check bounds the cost of a
    // premature trip to ≤SPEC_REENABLE_INTERVAL pipelined-fallback steps,
    // so we can gate early and recover the pipeline sooner on novel content.
    /// Steps of enabled-mode PLD before the yield gate may trip. See the
    /// `yield-gate warmup is 8` test for the sweep this was picked from and
    /// why it moved from 32.
    pub const YIELD_GATE_WARMUP: u64 = 8;
    pub const YIELD_GATE_MIN_YIELD: f32 = 0.25;

    /// Read the yield-gate warmup for this call. Env-overridable
    /// (`SPEC_YIELD_WARMUP`) so the threshold can be swept without a rebuild,
    /// exactly like `runtimeGateWarmup`. Outside `[1, 256]` falls back.
    ///
    /// This number is pure economics, and the economics moved: every warmup
    /// step pays PLD's UNPIPELINED cold forward (plus a synchronous host read
    /// of the sampled token) against `next()`'s async-pipelined step, so the
    /// tax is a fraction of the AR step cost — and the AR step got ~3x cheaper
    /// when the mscale promotion was fixed, which makes the same
    /// absolute tax a ~3x larger share. 32 was calibrated before that.
    pub fn yieldGateWarmup() u64 {
        const n = readEnvUsize("SPEC_YIELD_WARMUP", @intCast(YIELD_GATE_WARMUP));
        if (n < 1 or n > 256) return YIELD_GATE_WARMUP;
        return @intCast(n);
    }

    pub fn yieldGateShouldDisable(steps_total: u64, accepted: u64) bool {
        if (steps_total < yieldGateWarmup()) return false;
        const yield_rate = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(steps_total));
        return yield_rate < YIELD_GATE_MIN_YIELD;
    }

    // ── Mid-request spec re-enable ──
    // While the yield gate has PLD disabled, the COMMITTED sequence (prompt +
    // generated) is re-scored every SPEC_REENABLE_INTERVAL steps: what
    // fraction of the recent generated positions would have had a PLD lookup
    // hit (their key-gram appears earlier in committed)? This catches the
    // echo workload where the model repeats PROMPT content (file edits, tool
    // results) — self-repetition scoring misses it because the echoed tail
    // never repeats itself. Above the threshold, PLD is worth re-engaging at
    // the cost of one pipeline drain.
    pub const SPEC_REENABLE_INTERVAL: u64 = 32;
    pub const SPEC_REENABLE_WINDOW: usize = 32;
    pub const SPEC_REENABLE_MIN_FRACTION: f32 = 0.25;
    pub const SPEC_REENABLE_MIN_TOKENS: usize = 16;

    pub fn specShouldReenable(committed: []const u32, generated_len: usize) bool {
        if (generated_len < SPEC_REENABLE_MIN_TOKENS) return false;
        const window = @min(SPEC_REENABLE_WINDOW, generated_len);
        const frac = pld_index.tailMatchFraction(committed, window, 3);
        return frac >= SPEC_REENABLE_MIN_FRACTION;
    }

    pub fn logSpecStats(self: *const Generator) void {
        StepTrace.report();
        var table_buf: [256]u8 = undefined;
        var hist_buf: [256]u8 = undefined;
        var acc_buf: [128]u8 = undefined;
        var rows_buf: [160]u8 = undefined;
        const table_bucket = self.xfm.round_cost.bucketOf(self.specKvLen());
        if (self.dflash != null and self.dflash_attempted + self.dflash_plain > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                @as(f64, @floatFromInt(@max(self.dflash_attempted, 1)));
            const drafts_per_round: u32 = if (self.dflash_block_size >= 1) self.dflash_block_size - 1 else 0;
            // Under the chooser the width varies per round: drafts proposed
            // is the histogram's sum, not attempts x a fixed block.
            const drafts_proposed: u64 = if (self.dflash_chooser) |ch| ch.draftsProposed() else self.dflash_attempted * @as(u64, drafts_per_round);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=dflash attempts={d} accepts={d} avg_per_round={d:.2} gate_min={d:.2} per_draft_pct={d:.1}% block_size={d} partial_rounds={d} runtime_disabled={s} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d} block_avg={d:.2} block_hist={s} chooser_trials={d} plain={d} skipped={d} probes={d} copies={d}/{d}/{d} acc_pos={s} rows_hist={s}\n",
                .{
                    self.dflash_attempted,
                    self.dflash_accepted_tokens,
                    avg_per_round,
                    self.dflash_min_accepted_per_round,
                    per_draft_pct,
                    if (self.dflash_chooser) |ch| ch.current + 1 else self.dflash_block_size,
                    self.partial_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    self.xfm.round_cost.dropped_implausible,
                    if (self.dflash_chooser) |ch| ch.avgWidth() else @as(f32, @floatFromInt(drafts_per_round)),
                    if (self.dflash_chooser) |*ch| ch.formatHist(&hist_buf) else "",
                    if (self.dflash_chooser) |ch| ch.trial.trials else 0,
                    self.dflash_plain,
                    if (self.dflash_pol) |pol| pol.drafting.skipped else 0,
                    if (self.dflash_pol) |pol| pol.drafting.probes else 0,
                    self.dflash_copies,
                    self.dflash_copy_drafted,
                    self.dflash_copy_accepted,
                    self.dflash_stats.formatAcc(&acc_buf),
                    self.dflash_stats.formatRows(&rows_buf),
                },
            );
            return;
        }
        if (self.drafter != null and self.drafter_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f64, @floatFromInt(self.drafter_attempted));
            const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
            const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=drafter attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% block_size={d} runtime_disabled={s}\n",
                .{
                    self.drafter_attempted,
                    self.drafter_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.drafter_block_size,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
        } else if (self.pld_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.pld_accepted_tokens)) /
                @as(f64, @floatFromInt(self.pld_attempted));
            log.info(
                "  [spec-stats] mode=pld attempts={d} accepts={d} avg_per_round={d:.2} runtime_disabled={s}\n",
                .{
                    self.pld_attempted,
                    self.pld_accepted_tokens,
                    avg_per_round,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
        }
    }

    /// Prefill the prompt and prepare for token-by-token generation.
    /// Backwards-compatible — prefer `initWithOptions` for new callers.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        return initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    }

    pub const InitOptions = struct {
        /// Skip the lazy pre-forward of the first sampled token. When set,
        /// init samples t1 synchronously and leaves `pending_logits/pending_token`
        /// empty — the cache lands at exactly `prompt_len` with t1 NOT in cache.
        /// `nextPld` v2 (mirroring `nextDrafter`) drives every step from that
        /// invariant: verify input is `[t1, draft[0..m-1]]` length `1+m`; full
        /// accept commits `1+m` tokens with cache landing at `prompt_len + TE_new`
        /// and NO post-step forward. Saves one decode-step forward per accepted
        /// PLD step at the cost of losing the lazy-pipeline overlap on cold
        /// (no-match) steps. The prompt-time gate disables PLD on novel content
        /// where cold-path dominates.
        pld_enabled: bool = false,
        /// The machine-sized prefill chunk frozen at load
        /// (`ModelConfig.pinned_prefill_chunk`, resolved by
        /// `server.resolvePrefillChunk`). 0 = unpinned, keep the launch width.
        ///
        /// It arrives per REQUEST rather than off `xfm.config` because the
        /// Transformer holds a COPY of the config taken when it was built, and
        /// the pin is written to the registry's config after load — the copy
        /// never sees it. The scheduler reads `slot.model.config`, which IS the
        /// object the admission guard bills against, so bill and forward cannot
        /// disagree. (Live 2026-08-14: pinned 4096, prefilled at 8192.)
        pinned_prefill_chunk: usize = 0,
        /// `--prefill-decode-share` width while someone decodes, 0 = none. Applied after
        /// `effectivePrefillChunk`, so it narrows an explicit or env width too: the
        /// operator asked for the share as well.
        decode_share_width_cap: usize = 0,
        /// Enable Gemma 4 assistant drafter. When set, `drafter` must be
        /// non-null and already `bind()`-ed to `xfm`. Init's prefill final-token
        /// forward captures the post-final-norm hidden state into
        /// `Generator.last_hidden` (reused for the drafter's first-step
        /// h_prev — see comment in `nextDrafter`). Same lazy-pre-forward
        /// skip semantics as PLD.
        drafter_enabled: bool = false,
        /// Non-owning pointer to the loaded drafter (must be non-null when
        /// `drafter_enabled` is true).
        drafter: ?*DrafterModel = null,
        /// Number of tokens per draft round. Default 4 (3 drafter steps +
        /// 1 t1 prepend → length-4 verify forward).
        drafter_block_size: u32 = 4,
        /// Enable the DFlash block-drafter. When set, `dflash` must be
        /// non-null and `bind()`-ed to `xfm`. Prefill captures the trunk's
        /// target_layer_ids outputs chunk-by-chunk into the per-request
        /// assistant context cache (`Generator.dflash_ctx`). Same
        /// lazy-pre-forward skip semantics as PLD/drafter.
        dflash_enabled: bool = false,
        /// Non-owning pointer to the loaded DFlash assistant.
        dflash: ?*DflashModel = null,
        /// Effective DFlash block size (loader-resolved: assistant config
        /// clamped by --draft-block-size). 0 → the assistant config's value.
        dflash_block_size: u32 = 0,
        /// Accepted drafts/round required after warmup. Scheduler scales the
        /// M5/block-16 calibration to the effective draft width and lowers it
        /// for requests whose resolved mode has thinking enabled.
        dflash_min_accepted_per_round: f32 = DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
        /// Drive the DFlash rounds with `dflash_policy` (a sparse target, where a verify row costs):
        /// plain rounds replace the sticky serial fallback and the rows verified follow the draft's confidence.
        dflash_policy: bool = false,
        /// When set, this slice (rather than `prompt_ids`) becomes the
        /// `prompt_ids_owned` source for PLD's n-gram lookup. Used by the
        /// server's KV-cache-reuse path to forward only the trailing tokens
        /// while still giving PLD the full prompt for matching.
        lookup_prompt: ?[]const u32 = null,
        /// Per-slot forward context (Phase 2 concurrent batching). When null,
        /// `initWithOptions` builds one from `xfm.defaultCtx()` so the legacy
        /// single-slot path is unchanged. Phase 2 callers pass a ForwardCtx
        /// whose `cache` / `moe_seq_offset` / `ssm_entries` / `vision_embeddings`
        /// point at the slot's own state. Stored by value on the Generator.
        ctx: ?ForwardCtx = null,
        /// Skip the lazy first-token pre-forward (regular path only). When set,
        /// init samples t1 synchronously and leaves `pending_logits` /
        /// `pending_token` empty — cache.step lands at exactly prompt_len with
        /// t1 NOT in cache. The first `next()` call's transition shim will
        /// sync-forward `[t1]` to seed pending_logits before the lazy chain.
        /// Used by the Phase 2 scheduler so a slot's cache state matches
        /// `forwardBatchedDecode`'s expectation (cache.step == prompt_len at
        /// the start of every decode tick). PLD / drafter paths already skip
        /// the lazy pre-forward unconditionally; this flag generalizes that
        /// behavior to the regular sampling path. Has no effect when
        /// `pld_enabled` or `drafter_enabled` is true.
        skip_lazy_preforward: bool = false,
        /// Absolute position of the FIRST token in `prompt_ids`.
        /// On a cold prefill this is 0. On the warm path (where the
        /// scheduler restored some prefix and now forwards only the tail),
        /// callers pass `hot_matched`.
        prefix_pos_offset: usize = 0,
        /// Placeholder rows inside a restored prefix (prefix-cache hit on an
        /// image prompt): the vision splice starts here, not at row 0.
        vision_rows_before: usize = 0,
        /// A DFlash assistant context restored from the prefix cache, whose
        /// `absLen()` already equals `prefix_pos_offset`. Ownership
        /// transfers to the Generator. Null = start the assistant blind at
        /// the offset (a restore forwards no trunk layers, so a cold context
        /// is what a reused prefix would otherwise get).
        dflash_ctx_restored: ?dflash_mod.DflashCtx = null,
        /// Cooperative abort for abandoned requests: checked between prefill
        /// chunks. The scheduler passes `&slot.cancelled`, set by the conn
        /// thread when the client disconnects mid-prefill. When it flips,
        /// `initWithOptions` returns `error.Cancelled` instead of grinding
        /// out the rest of a multi-minute ghost prefill.
        cancel_flag: ?*const std.atomic.Value(bool) = null,
        /// When the chunk loop aborts on `cancel_flag`, the absolute prompt-token count already
        /// forwarded into the KV lands here: the authoritative length for a cancelled-prefill
        /// commit (`cache.step` only advances when init completes).
        cancelled_forwarded: ?*usize = null,
        /// Per-token logprobs count for this request (0 = disabled). Callers
        /// that set `Generator.logprobs_n` after init must ALSO pass it here:
        /// init's split-prefill final-token forward reads it for the first token.
        logprobs_n: u32 = 0,
        /// `/v1/completions` echo: every prompt token after the first gets its
        /// `logprobs_n`-wide entry here. The prompt must be forwarded whole (no restore).
        prompt_logprobs: ?*std.ArrayList(LogprobResult) = null,
        /// LIVE prefill progress, in tokens actually forwarded so far by THIS
        /// prefill. Bumped once per chunk (not per token), read off-thread by
        /// the metrics gauge sampler.
        ///
        /// Without it the panel is blind during prefill: `prompt_tokens_total`
        /// and `prefill_time_seconds` only advance at request COMPLETION, and
        /// generated tokens only accrue during decode — so a multi-minute
        /// prefill saturates the GPU while both tiles read 0 / "—". The
        /// scheduler resets it to 0 when the prefill ends.
        prefill_progress: ?*std.atomic.Value(u64) = null,
        /// Total tokens this prefill will forward (post-cache tail, same scale
        /// as `prefill_progress`), published once; null on the bypass engines.
        prefill_expected: ?*std.atomic.Value(u64) = null,

        /// Called once per completed prefill chunk boundary (chunk state
        /// evaluated, allocator cache cleared), except the final one. The
        /// scheduler runs decode ticks for the streams already decoding at
        /// this seam, so a long cold prefill stalls them for at most one
        /// chunk-forward instead of its whole duration. Null = the prefill
        /// runs atomically (pre-interleave behavior, and the
        /// MLX_SERVE_PREFILL_INTERLEAVE=0 kill switch).
        interleave_hook: ?InterleaveHook = null,
        /// SSD-first write-through (see `WriteThroughHook`).
        write_through_hook: ?WriteThroughHook = null,
        /// The per-chunk prefill width, asked at every chunk boundary after that chunk's
        /// `mlx_clear_cache` and before the interleave tick. Null keeps the admitted width.
        /// Presence is NOT the arch gate (`serve` installs it process-wide); read `adaptive_chunk_width`.
        chunk_width_hook: ?ChunkWidthHook = null,
        /// Is the per-chunk adaptive width live for this request (`server.adaptivePrefillChunkEnabled`)?
        /// The only thing that may widen `tailMergeMaxFor`'s bound.
        adaptive_chunk_width: bool = false,
    };

    pub const InterleaveHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque) void,
    };

    /// SSD-first write-through: called at every completed prefill chunk with the absolute KV
    /// position, so a cancelled prefill leaves a restorable prefix.
    pub const WriteThroughHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, abs_kv_pos: usize) void,
    };

    /// Returns the width the next chunk should run at (`cur` unchanged = no change).
    pub const ChunkWidthHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *AdaptiveWidthState) u32,
        /// Re-price a widen after the interleave tick has allocated for co-tenant slots. A
        /// step-down needs no confirmation: it is the safe direction.
        confirm: *const fn (ctx: *anyopaque, pos: usize, want: u32) bool,
    };

    /// Selects the source slice that `initWithOptions` will dupe into
    /// `prompt_ids_owned`. When `lookup_prompt` is non-null it wins (server
    /// cache-reuse path: full original prompt for PLD lookup); otherwise the
    /// caller's `prompt_ids` is used (back-compat path).
    pub fn pickLookupPromptSource(prompt_ids: []const u32, lookup_prompt: ?[]const u32) []const u32 {
        return lookup_prompt orelse prompt_ids;
    }

    pub fn initWithOptions(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling_in: SamplingParams,
        eos_token_ids: []const u32,
        options_in: InitOptions,
    ) !Generator {
        // Reserved-token suppression rides the sampling params from HERE —
        // the one chokepoint every init site funnels through — so every
        // sampling path (serial, PLD/drafter corrections, draft heads,
        // stochastic-verify filters, batched decode via `gen.sampling`)
        // inherits the model's mask without per-site wiring.
        var sampling = sampling_in;
        sampling.suppress_mask = xfm.suppress_mask;
        sampling.keyed = xfm.config.rowExactDecode() and mlx.metalKernelsAvailable(); // the keyed sampler is a Metal kernel
        sampling.position_base = pickLookupPromptSource(prompt_ids, options_in.lookup_prompt).len;
        // Spec drafts and verify rows share the serial sampler's keys, so an
        // unseeded sampled request gets its own random seed.
        if (sampling.keyed and sampling.seed == null and sampling.temperature > 0.01) {
            var prng = std.Random.DefaultPrng.init(@truncate(@as(u96, @bitCast(std.Io.Timestamp.now(io, .real).toNanoseconds())) ^ @intFromPtr(xfm)));
            sampling.seed = prng.random().int(u64);
        }
        // Lesto top-k schedule (MLX_SERVE_TOPK_SCHEDULE): the prompt's forwards, here, route to k1 and restart the count;
        // the defer covers the error paths before the prompt's last forward ends the prefill.
        xfm.beginTopKPrefill();
        defer xfm.endTopKPrefill();
        const options = options_in;
        const s = xfm.s;
        // Per-slot ForwardCtx (Phase 2). Stored by value on the Generator;
        // callers either supply one (scheduler) or fall through to
        // `xfm.defaultCtx()` for the legacy single-slot path. We pass
        // `&ctx` to every forward call below; the cache/moe/state fields
        // mutate in-place through their pointers.
        var ctx: ForwardCtx = options.ctx orelse xfm.defaultCtx();

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Clone the lookup prompt for the lifetime of the Generator. PLD's
        // n-gram lookup needs `prompt + generated`, and the caller-owned
        // slice can be freed before `nextPld` runs. When `options.lookup_prompt`
        // is set (server cache-reuse path), it carries the full original prompt
        // so PLD's match coverage isn't gutted when only a trailing tail was
        // forwarded into the KV cache. Defaults to `prompt_ids` otherwise.
        // Allocated up front so init's errdefer paths don't have to track
        // partial state.
        const owned_src = pickLookupPromptSource(prompt_ids, options.lookup_prompt);
        const prompt_owned = try allocator.dupe(u32, owned_src);
        errdefer allocator.free(prompt_owned);

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        //
        // Chunked prefill: large prompts are processed in PREFILL_CHUNK-sized pieces
        // to bound peak activation memory. Each chunk fills KV cache entries for its
        // positions, gets eval'd, and intermediates are freed before the next chunk.
        // Without chunking, Gemma-4 MoE's 2 MLPs × 4 stacked layers can spike to
        // ~20 GB of activations alone on a 50k-token prompt, causing Metal OOM.
        // Vision requests skip chunking since image token positions must be visible
        // in a single forward pass for spliceVisionEmbeddings to work correctly.
        // PREFILL_CHUNK overridable via env MLX_SERVE_PREFILL_CHUNK for tuning,
        // or via the module-level `prefill_chunk_override` (set by --prefill-chunk
        // CLI flag in main.zig). Env var wins if both are set (and skips the
        // safety cap below — it's the explicit escape hatch).
        //
        // Safety cap: on unfused head dims (>128 - every Gemma-4)
        // the composed SDPA materializes [heads, chunk, total_kv] scores per
        // layer, so the chunk shrinks with the prompt's FINAL KV length to keep
        // that one tensor bounded (boundedPrefillChunk). Warm-path restores
        // start at prefix_pos_offset, so the final KV length is that
        // offset plus everything we're about to forward.
        const total_ctx_for_chunk = options.prefix_pos_offset + prompt_ids.len;
        const PREFILL_CHUNK: usize = decodeShareCapped(effectivePrefillChunk(
            xfm.config.prefillScoreHeadDim(),
            xfm.config.num_attention_heads,
            total_ctx_for_chunk,
            xfm.config.has_sliding_window,
            xfm.config.isMoe(),
            xfm.config.longCtxGated(),
            options.pinned_prefill_chunk,
        ), options.decode_share_width_cap);
        // Phase-level prefill instrumentation. Enabled at debug level OR via
        // MLX_SERVE_PREFILL_TRACE=1 (which forces the trace line at info).
        // Phase 0 of plan 04 — gives us a decomposed view of where cold prefill
        // time goes (chunked-forward vs eval vs last-token-forward).
        const trace_force: bool = prefill_trace_force or readEnvBool("MLX_SERVE_PREFILL_TRACE");
        const trace_enabled = log.isDebug() or trace_force;
        var prefill_sw = io_util.Stopwatch.init(io);
        var chunked_ns: u64 = 0;
        var eval_ns: u64 = 0;
        var n_chunks: usize = 0;

        const has_vision = ctx.vision_embeddings != null;
        // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
        // scheduler after restoring a prefix) pass the matched prefix length.
        const pos_offset: usize = options.prefix_pos_offset;

        // DFlash: build the assistant's context cache during prefill — every
        // chunk's target_layer_ids captures project through the encoder and
        // append immediately (the chunk's activation graph then frees with
        // the chunk eval). On KV-prefix reuse the context covers only the
        // freshly forwarded tail (`base_pos = pos_offset`) - sliding-
        // window semantics.
        const dflash_active = options.dflash_enabled and options.dflash != null;
        var dflash_ctx: ?dflash_mod.DflashCtx = if (!dflash_active)
            null
        else if (options.dflash_ctx_restored) |restored| blk: {
            std.debug.assert(restored.absLen() == pos_offset);
            break :blk restored;
        } else try dflash_mod.DflashCtx.init(allocator, options.dflash.?, pos_offset);
        errdefer if (dflash_ctx) |*dc| dc.deinit();
        var dfl_out_buf: []mlx.mlx_array = &.{};
        defer if (dfl_out_buf.len > 0) allocator.free(dfl_out_buf);
        var dfl_cl: transformer_mod.CaptureLayers = undefined;
        if (dflash_active) {
            dfl_out_buf = try allocator.alloc(mlx.mlx_array, options.dflash.?.config.target_layer_ids.len);
            dfl_cl = .{ .ids = options.dflash.?.config.captureIds(), .out = dfl_out_buf };
        }

        // Start of the final (logits) forward's token span: the last prompt
        // token (the classic 1-token logits forward).
        var final_start: usize = prompt_ids.len - 1;
        // Chunked vision prefill (issue #197): placeholder rows consumed by
        // completed chunks, fed to ctx.vision_splice_offset per forward
        // (chunk loop AND final-span forward). Stays 0 on the kill-switch arm
        // so that path is byte-identical to the old whole-prompt behavior.
        const vision_chunked = has_vision and visionChunkedPrefillEnabled();
        var vision_rows_consumed: usize = options.vision_rows_before;
        var adapt_state: AdaptiveWidthState = .{
            .width_min = @intCast(PREFILL_CHUNK),
            .width_max = @intCast(PREFILL_CHUNK),
        };

        if (final_start > 0) {
            const prefix_len = prompt_ids.len - 1;
            const loop_end = prefix_len;
            final_start = loop_end;
            // Vision prompts chunk like text (issue #197) — the splice offset
            // below keeps the row scatter chunk-exact. Kill switch restores
            // the whole-prompt forward.
            const default_chunk = if (has_vision and !vision_chunked) loop_end else PREFILL_CHUNK;
            // Per-chunk adaptive width: the first chunk runs the admitted width; every boundary
            // after it re-asks the same estimator. `cap_adapt` is the widest this arch forwards
            // for this prompt.
            const adapt_chunked = !(has_vision and !vision_chunked);
            // The scaled tail-merge bound reads the arch predicate, never `chunk_width_hook != null`
            // (installed process-wide).
            const width_is_adaptive = adapt_chunked and options.adaptive_chunk_width;
            const cap_adapt: u32 = if (!adapt_chunked) 0 else @intCast(effectivePrefillChunk(
                xfm.config.prefillScoreHeadDim(),
                xfm.config.num_attention_heads,
                total_ctx_for_chunk,
                xfm.config.has_sliding_window,
                xfm.config.isMoe(),
                xfm.config.longCtxGated(),
                0,
            ));
            adapt_state.width_min = @intCast(default_chunk);
            adapt_state.width_max = @intCast(default_chunk);
            var cur_chunk: usize = default_chunk;

            // Reserve the whole request's cache capacity before the first chunk writes: a grow
            // is not in place, so a long prefill's peak carried a second copy of everything so
            // far (~7.75 GB at 458k, #353). Same number the admission guard billed
            // (`KVCache.reservedTokens`), in ABSOLUTE positions: `prompt_ids` is the tail after
            // a prefix-cache hit.
            const reserved_tokens = reservedPrefillTokens(
                &xfm.config,
                total_ctx_for_chunk,
                max_tokens,
                default_chunk,
            );
            ctx.cache.reserve(@intCast(reserved_tokens));

            var pos: usize = 0;
            if (options.prefill_expected) |e| e.store(@intCast(loop_end), .monotonic);
            while (pos < loop_end) {
                // Abandoned-request abort: the client disconnected and the
                // conn thread flagged the slot. Bail before the next chunk —
                // the KV built so far is freed with the slot.
                if (options.cancel_flag) |cf| {
                    if (cf.load(.acquire)) {
                        if (options.cancelled_forwarded) |fwd| fwd.* = pos_offset + pos;
                        return error.Cancelled;
                    }
                }
                // An MLX failure in a previous chunk latched instead of killing the process (#353);
                // abandon this request, keep the server.
                try mlx.checkError();
                // Pick this chunk's end: the configured chunk size, with a tiny tail merged in.
                const end = nextChunkEnd(pos, loop_end, cur_chunk, width_is_adaptive);
                if (has_vision) ctx.vision_splice_offset = vision_rows_consumed;
                const chunk_len: c_int = @intCast(end - pos);
                const chunk_shape = [_]c_int{ 1, chunk_len };
                const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(chunk_input);

                const chunk_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                // Phase 2 experiment: when MLX_SERVE_COMPILE_FORWARD=1 wired a
                // compiled closure at load time, route this chunk through it.
                // The compiled closure uses xfm.defaultCtx (xfm.cache),
                // which matches the prefill `ctx` when the scheduler has swapped
                // the slot's cache onto the Transformer (the single-slot legacy
                // and Phase-2-swapped path both satisfy this). Hidden-capture
                // and vision splice paths don't pass through this chunk loop
                // (they take the last_input branch), so they're already safe.
                if (dflash_active) {
                    for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
                    ctx.capture_layers = &dfl_cl;
                }
                errdefer if (dflash_active) {
                    for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
                };
                // The compiled forward bypasses `forwardWith`, so the top-k schedule (MLX_SERVE_TOPK_SCHEDULE) never reaches it.
                const chunk_logits = if (xfm.compiled_forward != null and
                    Transformer.topKSchedule() == null and
                    !dflash_active and
                    ctx.cache == &xfm.cache and
                    ctx.capture_hidden == null and
                    ctx.vision_embeddings == null)
                    try xfm.forwardCompiled(chunk_input)
                else
                    try xfm.forwardWith(&ctx, chunk_input);
                if (options.prompt_logprobs) |out| promptLogprobs(allocator, chunk_logits, prompt_ids[pos + 1 .. end + 1], options.logprobs_n, xfm.s, out) catch |err| {
                    _ = mlx.mlx_array_free(chunk_logits);
                    return err;
                };
                _ = mlx.mlx_array_free(chunk_logits);
                if (dflash_active) {
                    ctx.capture_layers = null;
                    try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, pos_offset + pos);
                    for (dfl_out_buf) |*a| {
                        _ = mlx.mlx_array_free(a.*);
                        a.* = .{ .ctx = null };
                    }
                }
                if (trace_enabled) chunked_ns += prefill_sw.read() - chunk_start_ns;

                // Eval KV cache — materializes this chunk's K/V, frees activation graph
                const eval_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                {
                    const eval_vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(eval_vec);
                    for (ctx.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                    }
                    // Same discipline for the DFlash context appended above.
                    if (dflash_ctx) |*dc| dc.appendEvalArrays(eval_vec);
                    _ = mlx.mlx_eval(eval_vec);
                }
                _ = mlx.mlx_clear_cache();
                if (trace_enabled) eval_ns += prefill_sw.read() - eval_start_ns;

                // This chunk's latch, read before anything persists it, snapshots it or yields
                // the thread: Metal returns zeros before it aborts, so a later read let the
                // write-through index garbage and a co-tenant decode eat the latch.
                try mlx.checkError();

                if (vision_chunked) {
                    vision_rows_consumed += countSpliceRows(
                        ids_i32[pos..end],
                        xfm.config.image_token_id,
                        xfm.config.audio_token_id,
                        xfm.config.video_token_id,
                    );
                }
                pos = end;
                n_chunks += 1;
                // Publish progress once per chunk — same cadence discipline as
                // `inflight_generated_tokens` (once per decode tick), never per token.
                if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);
                // Persist this chunk now: the KV bytes for [0, pos) are final.
                if (options.write_through_hook) |wt| {
                    wt.call(wt.ctx, pos + pos_offset);
                }
                // The next chunk's width, re-priced at this boundary: after the `mlx_clear_cache`
                // (steady state) and before the interleave tick (a co-tenant's decode is not this
                // prefill's pressure). A step-down commits here; a widen is re-priced after the tick.
                var pending_widen: u32 = 0;
                var widen_confirm: ?*const fn (*anyopaque, usize, u32) bool = null;
                var widen_ctx: ?*anyopaque = null;
                if (adapt_chunked and pos < loop_end) {
                    if (options.chunk_width_hook) |hk| {
                        const next_w = hk.call(hk.ctx, pos_offset + pos, @intCast(cur_chunk), cap_adapt, &adapt_state);
                        if (next_w != 0 and next_w != cur_chunk) {
                            if (next_w < cur_chunk) {
                                commitAdaptiveWidth(&cur_chunk, &adapt_state, next_w);
                            } else {
                                pending_widen = next_w;
                                widen_confirm = hk.confirm;
                                widen_ctx = hk.ctx;
                            }
                        }
                    }
                }
                // Yield to the scheduler between chunks — never after the
                // last (the post-prefill decode tick covers that boundary).
                if (pos < loop_end) {
                    if (options.interleave_hook) |hk| hk.call(hk.ctx);
                }
                // The tick allocated for co-tenant slots; re-price the widen against what is left.
                if (pending_widen != 0) {
                    if (widen_confirm) |cf| {
                        if (cf(widen_ctx.?, pos_offset + pos, pending_widen)) {
                            commitAdaptiveWidth(&cur_chunk, &adapt_state, pending_widen);
                        } else {
                            adapt_state.supporting = 0;
                        }
                    }
                }
            }
            try mlx.checkError();
            // One line per request when the width moved.
            if (adapt_state.transitions > 0) {
                log.info("[prefill] adaptive: {d} chunks, width {d}..{d}, {d} change(s)\n", .{
                    n_chunks,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    adapt_state.transitions,
                });
            }

        }

        // Chunked vision: the final-span forward continues the row scatter
        // where the chunk loop left it (an image ending in the final span
        // otherwise re-splices from row 0). One-shot engagement line — the
        // silent-no-op class; tests grep for it per arm.
        if (has_vision) {
            ctx.vision_splice_offset = vision_rows_consumed;
            if (vision_chunked and n_chunks > 1) log.debug("[vision] chunked prefill: {d} chunks, {d} placeholder rows consumed\n", .{ n_chunks, vision_rows_consumed });
        }

        // Process the final span: the last token. One forward, logits sliced to the last
        // position so every consumer sees the classic [1, 1, V] shape.
        const tail_len: usize = prompt_ids.len - final_start;
        const last_shape = [_]c_int{ 1, @as(c_int, @intCast(tail_len)) };
        const last_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[final_start]), &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        // Drafter (Gemma 4 assistant) needs the post-final-norm hidden as
        // its first-step h_prev — captured here so we don't need a second
        // forward at the start of `nextDrafter`. `forwardWithCapture`
        // captures the LAST position regardless of span length. When the
        const drafter_active = options.drafter_enabled and options.drafter != null;
        const pld_active = options.pld_enabled;
        const need_capture = drafter_active;
        var captured_hidden: mlx.mlx_array = mlx.mlx_array_new();
        var has_captured_hidden = false;
        const last_start_ns = if (trace_enabled) prefill_sw.read() else 0;
        // DFlash context for the tail span (rides ctx into all three forward
        // arms below; the forwardWithCapture* wrappers only override
        // capture_hidden/_all).
        if (dflash_active) {
            for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
            ctx.capture_layers = &dfl_cl;
        }
        errdefer if (dflash_active) {
            for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
        };
        const raw_logits = if (need_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCapture(&ctx, last_input, &captured_hidden);
        } else try xfm.forwardWith(&ctx, last_input);
        // The prompt's last forward has run: the first reply token's forward (the lazy pre-forward below) is counted.
        xfm.endTopKPrefill();
        if (dflash_active) {
            ctx.capture_layers = null;
            try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, pos_offset + final_start);
            for (dfl_out_buf) |*a| {
                _ = mlx.mlx_array_free(a.*);
                a.* = .{ .ctx = null };
            }
        }
        if (options.prompt_logprobs) |out| {
            errdefer _ = mlx.mlx_array_free(raw_logits);
            try promptLogprobs(allocator, raw_logits, prompt_ids[final_start + 1 ..], options.logprobs_n, xfm.s, out);
            if (out.items.len != prompt_ids.len - 1) return error.PromptLogprobsUnavailable;
        }
        // Slice to the last position when the span is longer than one token,
        // so downstream sampling/grammar paths see the classic shape.
        const logits = if (tail_len == 1) raw_logits else blk: {
            defer _ = mlx.mlx_array_free(raw_logits);
            const lshape = mlx.getShape(raw_logits);
            var sliced = mlx.mlx_array_new();
            const start = [_]c_int{ 0, lshape[1] - 1, 0 };
            const stop = [_]c_int{ lshape[0], lshape[1], lshape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&sliced, raw_logits, &start, 3, &stop, 3, &strides, 3, xfm.s));
            break :blk sliced;
        };
        if (trace_enabled) {
            const last_ns = prefill_sw.read() - last_start_ns;
            const total_ns = prefill_sw.read();
            const ms = std.time.ns_per_ms;
            std.debug.print(
                "  [prefill-trace] tokens={d} chunks={d} chunk_size={d} chunk_widths={d}..{d} chunked={d}ms eval={d}ms last_token={d}ms total={d}ms{s}{s}\n",
                .{
                    prompt_ids.len,
                    n_chunks,
                    PREFILL_CHUNK,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    chunked_ns / ms,
                    eval_ns / ms,
                    last_ns / ms,
                    total_ns / ms,
                    if (need_capture) " [capture-hidden]" else "",
                    if (pld_active) " [pld]" else "",
                },
            );
        }
        errdefer if (has_captured_hidden) {
            _ = mlx.mlx_array_free(captured_hidden);
        };

        // Constrained generation skips the lazy first-sample fast path: we cannot
        // sample the first token until we have applied the grammar mask, and we
        // cannot pipeline because grammar advancement depends on the realized id.
        if (sampling.constraint != null) {
            // Grammar-constrained requests never speculate; release the
            // DFlash context if dispatch enabled it anyway.
            if (dflash_ctx) |*dc| {
                dc.deinit();
                dflash_ctx = null;
            }
            var gen = Generator{
                .xfm = xfm,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = 0,
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            gen.pending_logits = logits;
            gen.has_pending_logits = true;
            return gen;
        }

        // Drafter / PLD-v2 / DFlash path: sample synchronously and DO NOT
        // pre-forward the sampled token. The first nextDrafter / nextPld /
        // nextDflash call needs the cache at exactly prompt_len (last prompt
        // token forwarded; first sampled token deferred). The lazy
        // pre-forward path below would over-advance the cache and corrupt
        // every verify forward.
        if (drafter_active or pld_active or dflash_active) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            const dflash_bs: u32 = if (dflash_active)
                (if (options.dflash_block_size > 0) options.dflash_block_size else options.dflash.?.config.block_size)
            else if (options.dflash_block_size > 0)
                options.dflash_block_size
            else
                0;
            const gen = Generator{
                .xfm = xfm,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .pld_enabled = pld_active,
                .drafter = if (drafter_active) options.drafter else null,
                .drafter_block_size = options.drafter_block_size,
                .dflash = if (dflash_active) options.dflash else null,
                .dflash_ctx = dflash_ctx,
                .dflash_block_size = dflash_bs,
                .dflash_pol = if (dflash_active and options.dflash_policy) .{ .pin_rows = @intCast(@min(readEnvUsize("MLX_SERVE_DFLASH_ROWS", 0), dflash_policy.MAX_ROWS)) } else null,
                .dflash_chooser = if (dflash_active and dflashChooserEnabled() and !options.dflash_policy)
                    round_cost.WidthChooser.init(@max(dflash_bs, 2) - 1, options.dflash.?.config.block_size -| 1)
                else
                    null,
                .dflash_min_accepted_per_round = options.dflash_min_accepted_per_round,
                .spec_ev_costs = specEvCosts(),
            };
            dflash_ctx = null; // ownership transferred to the Generator
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under PLD / drafter / DFlash. The speculative `next*` paths
            // drive every subsequent step with predictable cache offset.
            return gen;
        }

        // Phase 2: scheduler-managed slots ask init to sample t1 synchronously
        // and skip the lazy pre-forward. Cache lands at prompt_len with t1 NOT
        // in cache — matches `forwardBatchedDecode`'s expectation and the
        // PLD / drafter init path's invariant. Generator.next's transition
        // shim handles the bootstrap on the first decode tick.
        if (options.skip_lazy_preforward) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);
            // t1's own distribution comes from the prefill's final forward and
            // is invisible to the decode loop — read it here or the first
            // returned token has no logprobs and the array shifts by one.
            const first_lp = try firstTokenLogprobs(allocator, logits, @intCast(first_val), options.logprobs_n, s);
            _ = mlx.mlx_array_free(logits);

            const gen = Generator{
                .pending_logprob = first_lp,
                .xfm = xfm,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            return gen;
        }

        // Regular path: sample first token lazily, then build the next forward pass
        const lazy_token = sampleTokenLazy(logits, sampling, s);

        const next_logits = try lazyForward(xfm, &ctx, lazy_token);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);
        // See the skip_lazy_preforward branch: t1's distribution lives only in
        // the prefill's own output, so it is read here, against the id that was
        // actually drawn.
        const first_lp = try firstTokenLogprobs(allocator, logits, @intCast(val), options.logprobs_n, s);
        _ = mlx.mlx_array_free(logits);

        var gen = Generator{
            .pending_logprob = first_lp,
            .xfm = xfm,
            .ctx = ctx,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = blk: {
                var sp = sampling;
                sp.draw = 1; // t1 above was draw 0
                break :blk sp;
            },
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = io_util.Stopwatch.init(io),
            .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
            .has_last_hidden = has_captured_hidden,
            .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        return gen;
    }

    /// Return the prefill's parked transients once, at the handover to decode.
    /// Shares `advanceStep`'s clock: a clear here is a clear, so the decode
    /// cadence measures its next interval from this point.
    pub fn clearPoolBeforeDecode(self: *Generator) void {
        if (!clearsPoolAtPrefillEnd(&self.xfm.config)) return;
        _ = mlx.mlx_clear_cache();
        self.last_cache_clear_step = self.step;
    }

    /// The ONE place `step` and `completion_tokens` advance — every decode path
    /// (plain, constrained, PLD, drafter, DFlash, batched) routes through here.
    ///
    /// Advancing them by hand is how three paths - `nextDrafter`, `nextDflash` and
    /// `scheduler.runBatchedDecodeTick` — ended up never calling
    /// `mlx_clear_cache()` at all (issue #110). MLX parks freed buffers in a
    /// size-keyed pool instead of returning them to the OS, so a decode path
    /// with no clear ratchets the process footprint while `active_bytes` stays
    /// flat: the reporter's process reached 81 GB with the panel reading 19.6.
    /// A source-scan test pins that no new path can reintroduce the hole.
    pub fn advanceStep(self: *Generator, n: u32) void {
        self.completion_tokens += n;
        self.step += n;
        if (shouldClearAllocatorCache(self.step, self.last_cache_clear_step, CACHE_CLEAR_INTERVAL)) {
            _ = mlx.mlx_clear_cache();
            self.last_cache_clear_step = self.step;
        }
    }

    pub fn hasPrePayloadConstraint(self: *const Generator) bool {
        const constraint = self.sampling.constraint orelse return false;
        return constraint.proto != null and constraint.pstate.phase != .json_body;
    }

    /// The pending reasoning→payload boundary, consumed by the scheduler when
    /// it publishes the token that carries it (same delivery path as the
    /// token itself, so a streaming reader cannot see one without the other).
    pub fn takeConstraintSpan(self: *Generator) ?rp_mod.ConstraintSpan {
        const span = self.sampling.constraint orelse return null;
        const s = span.pending_span orelse return null;
        span.pending_span = null;
        return s;
    }

    pub const ConstraintRecovery = union(enum) {
        /// One forced token was committed through the live forward path; the
        /// caller publishes it like a sampled token.
        committed: u32,
        /// The choice state recovers by answering directly: nothing to
        /// commit, the next sample is JSON-masked.
        activated,
        /// The remaining transition does not fit the completion allowance.
        refused,
    };

    /// Force the reasoning→constrained transition when the protocol has not
    /// reached the JSON body: a reasoning loop, premature EOS, or terminal
    /// padding recovers into a valid final answer when the remaining
    /// completion allowance fits the WHOLE remaining transition plus one
    /// answer token. Commits at most one forced token per call (the next
    /// tick drains the rest), so cancellation, stop sequences, inference
    /// errors, and stalled forwards keep their usual precedence.
    pub fn forceConstraintTransition(self: *Generator, allocator: std.mem.Allocator) !ConstraintRecovery {
        const constraint = self.sampling.constraint orelse return .refused;
        const proto = constraint.proto orelse return .refused;
        std.debug.assert(!self.has_pending_token);

        switch (constraint.pstate.phase) {
            .json_body => return .refused,
            .choice => {
                if (constraint.pstate.open_cursor == 0) {
                    // The model never committed a channel and emitted no
                    // opener bytes: a direct constrained answer needs no
                    // transition token at all. The pending logits stay —
                    // they are the distribution the next JSON-masked sample
                    // reads.
                    constraint.pstate.phase = .json_body;
                    constraint.pstate.recovering = false;
                    constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
                    self.loop_guard_start = self.generated_ids.items.len;
                    log.info("[grammar] reasoning loop at the channel choice; enforcing JSON schema immediately\n", .{});
                    return .activated;
                }
                // Opener bytes are already in the output: starting the JSON
                // here would strand them (the response split reads them as an
                // unclosed opener and files the answer as reasoning).
                // Complete the opener, close it, then constrain. Fall through
                // to the shared planner below.
                log.info("[grammar] completing the generated reasoning opener before the boundary\n", .{});
                if (self.has_pending_logits) {
                    _ = mlx.mlx_array_free(self.pending_logits);
                    self.has_pending_logits = false;
                }
                if (!constraint.pstate.recovering) {
                    var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                    const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                    if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                    constraint.pstate.recovering = true;
                    constraint.pstate.forced_cursor = 0;
                    constraint.pstate.pending_len = @intCast(planned_len);
                    @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
                }
                return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
            },
            .reasoning, .header => {
                if (self.has_pending_logits) {
                    _ = mlx.mlx_array_free(self.pending_logits);
                    self.has_pending_logits = false;
                }
                if (!constraint.pstate.recovering) {
                    var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                    const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                    if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                    constraint.pstate.recovering = true;
                    constraint.pstate.forced_cursor = 0;
                    constraint.pstate.pending_len = @intCast(planned_len);
                    @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
                }
                return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
            },
        }
    }

    /// Commit ONE canonical transition token through the live model state,
    /// then leave its logits pending for the next sample. Called once per
    /// scheduler tick while `pstate.recovering` drains.
    fn commitForcedConstraintToken(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint) !u32 {
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        const seq = constraint.pstate.pending[0..constraint.pstate.pending_len];
        const tok = seq[constraint.pstate.forced_cursor];
        constraint.pstate.forced_cursor += 1;

        try self.generated_ids.append(allocator, tok);
        self.loop_guard_start = self.generated_ids.items.len;
        self.next_token_id = tok;
        self.advanceStep(1);
        self.consecutive_pad = 0;

        const tok_i32: i32 = @intCast(tok);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
        const arr = [_]mlx.mlx_array{next_logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = next_logits;
        self.has_pending_logits = true;

        if (constraint.pstate.forced_cursor >= seq.len) {
            // The transition drained: the next sample is the first constrained
            // payload token. Reset the repetition window at the final-body
            // transition so the drained sequence cannot re-trigger recovery.
            constraint.pstate.recovering = false;
            constraint.pstate.phase = .json_body;
            constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
            self.loop_guard_start = self.generated_ids.items.len;
        }
        return tok;
    }

    pub fn loopGuardStart(self: *const Generator) usize {
        return @min(self.loop_guard_start, self.generated_ids.items.len);
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.pending_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        if (self.has_last_hidden) {
            _ = mlx.mlx_array_free(self.last_hidden);
            self.has_last_hidden = false;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        if (self.dflash_ctx) |*dc| {
            dc.deinit();
            self.dflash_ctx = null;
        }
        for (self.dflash_pending.items) |a| _ = mlx.mlx_array_free(a);
        self.dflash_pending.deinit(allocator);
        self.dflash_pending = .empty;
        self.dflash_pending_rounds = 0;
        if (self.ngram_idx) |*idx| {
            idx.deinit();
            self.ngram_idx = null;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Legacy→batched transition (scheduler.runBatchedDecodeTick): consume
    /// the lazy pipeline state so the slot can join a batched tick. The
    /// legacy pipelined decode keeps a lookahead token ALREADY FORWARDED
    /// into the KV cache (`pending_token` / `next_token_id`) plus
    /// `pending_logits` for the position after it. Dropping that state and
    /// re-forwarding `next_token_id` would append a duplicate position to
    /// the cache and re-emit an already-emitted token — corrupting every
    /// stream whose slot enters a batch mid-generation
    /// (tests/test_batched_transition.sh).
    ///
    /// Returns the token to emit this step (the pipelined lookahead), or
    /// null when generation stopped (`checkStop`: EOS / pad-run /
    /// max_tokens / timeout — `finish_reason` is set). On return:
    /// `next_token_id` is sampled but NOT in the cache and pending state is
    /// empty — exactly the batched-tick entry invariant.
    pub fn drainPipelineForBatch(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        try self.resolvePendingToken();
        if (try self.checkStop()) {
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            return null;
        }
        // Both pipeline shapes (fresh-from-prefill and post-`next()` fast
        // path) carry pending_logits alongside the in-cache lookahead; a
        // lookahead without logits would force a re-forward of an in-cache
        // token, which is the corruption this method exists to prevent.
        if (!self.has_pending_logits) return error.MissingPendingLogits;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        const lazy = self.sampleLazy(step_logits);
        _ = mlx.mlx_array_free(step_logits);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        self.next_token_id = @intCast(val);
        return token;
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    /// The ONE lazy sampler for a slot's own draws: advances the seed draw index.
    pub fn sampleLazy(self: *Generator, logits: mlx.mlx_array) mlx.mlx_array {
        defer self.sampling.draw +%= 1;
        defer self.ctx.head_tail_want = self.headTailGreedy();
        // A greedy draw from a head forward that already reduced its kept rows to the token id.
        if (self.headTailGreedy()) {
            if (Transformer.headTailTakeToken(logits, self.sampling.suppress_mask)) |tok| return tok;
        }
        // Not taken: drop it, so the slot never outlives the draw that follows its forward.
        Transformer.headTailDrop();
        if (self.sampling.penalized() and self.generated_ids.items.len > 0) {
            var pen = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pen);
            if (applyPenalties(&pen, logits, self.generated_ids.items, self.sampling, self.xfm.s)) |_| {
                return sampleTokenLazy(pen, self.sampling, self.xfm.s);
            } else |err| log.warn("[sampling] repeat penalty failed ({s}); this draw is unpenalized\n", .{@errorName(err)});
        }
        return sampleTokenLazy(logits, self.sampling, self.xfm.s);
    }

    /// sampleLazy would take sampleTokenLazy's greedy fast path (after suppression) for this draw.
    fn headTailGreedy(self: *const Generator) bool {
        return self.sampling.temperature < 0.01 and self.sampling.constraint == null and
            !(self.sampling.penalized() and self.generated_ids.items.len > 0);
    }

    /// What a pipelined step evaluates for the next draw: the head's token when that draw will take it,
    /// so the full logits row stays lazy; else the row itself.
    fn headTailEvalTarget(self: *const Generator, next_logits: mlx.mlx_array) mlx.mlx_array {
        if (self.logprobs_n != 0 or self.sampling.penalized() or !self.headTailGreedy()) return next_logits;
        return Transformer.headTailPeekToken(next_logits, self.sampling.suppress_mask) orelse next_logits;
    }

    /// The token the next publish emits; a pipelined one is resolved now (one sync).
    pub fn upcomingToken(self: *Generator) !u32 {
        try self.resolvePendingToken();
        return self.next_token_id;
    }

    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    const DrainResult = union(enum) {
        /// No pending pipeline state — the spec entry invariant already holds.
        already_clean,
        /// One token was emitted while draining; caller returns it this step.
        drained: u32,
        /// The drained token hit a stop condition — generation is over.
        stopped,
        /// Unexpected half-state; do not re-enable speculation.
        stay_disabled,
    };

    /// Transition from the pipelined `next()` state back to the spec-decode
    /// entry invariant (next_token_id known but NOT in cache, no pending
    /// state). The pipeline holds `pending_token` (lazy, its forward already
    /// in the cache) and `pending_logits` (logits for the position after it):
    /// resolving the token, emitting it, and sampling its successor from
    /// `pending_logits` WITHOUT forwarding lands exactly on the invariant.
    /// One sync. Also handles the shim-seeded state (`pending_logits` only).
    fn drainPipelineForSpec(self: *Generator, allocator: std.mem.Allocator) !DrainResult {
        if (!self.has_pending_logits) {
            if (!self.has_pending_token) return .already_clean;
            // pending_token without pending_logits never occurs in the
            // pipelined state machine; bail rather than risk the invariant.
            return .stay_disabled;
        }
        try self.resolvePendingToken();
        if (try self.checkStop()) return .stopped;
        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        // With the latch these checks return instead of ending the process, so the handle needs an owner.
        const val: i32 = blk: {
            const lazy = self.sampleLazy(step_logits);
            _ = mlx.mlx_array_free(step_logits);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
            break :blk v;
        };
        self.next_token_id = @intCast(val);
        return .{ .drained = token };
    }

    pub const ForcedCommit = struct {
        /// Tokens emitted by this call: any pending token plus every forced
        /// token but the last, which stays as `next_token_id` for the next
        /// regular tick to publish.
        emitted: []const u32,
        stopped: bool,
    };

    /// Commit `forced` through the model as ONE multi-token forward, from any
    /// of the three inter-tick states (pipelined, spec exit, shim-seeded), and
    /// leave the generator in the shim-seeded state a regular `next()` reads.
    /// The caller routes every later tick through the regular path.
    pub fn commitForcedTokens(self: *Generator, allocator: std.mem.Allocator, forced: []const u32) !ForcedCommit {
        std.debug.assert(forced.len > 0);
        var emitted = std.ArrayList(u32).empty;
        errdefer emitted.deinit(allocator);
        var input = std.ArrayList(u32).empty;
        defer input.deinit(allocator);

        if (self.has_pending_token or self.has_pending_logits) {
            // Pipelined or shim-seeded: `next_token_id` is in the cache, unpublished.
            try self.resolvePendingToken();
            if (try self.checkStop()) return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = true };
            try self.generated_ids.append(allocator, self.next_token_id);
            try emitted.append(allocator, self.next_token_id);
            self.advanceStep(1);
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
        } else {
            // Spec exit: `next_token_id` is decided but not in the cache.
            try input.append(allocator, self.next_token_id);
            try self.generated_ids.append(allocator, self.next_token_id);
            try emitted.append(allocator, self.next_token_id);
            self.advanceStep(1);
        }
        try input.appendSlice(allocator, forced);
        for (forced[0 .. forced.len - 1]) |f| {
            try self.generated_ids.append(allocator, f);
            try emitted.append(allocator, f);
        }
        self.advanceStep(@intCast(forced.len - 1));
        self.consecutive_pad = 0;
        self.loop_guard_start = self.generated_ids.items.len;

        const ids_i32 = try allocator.alloc(i32, input.items.len);
        defer allocator.free(ids_i32);
        for (input.items, 0..) |t, i| ids_i32[i] = @intCast(t);
        const shape = [_]c_int{ 1, @intCast(input.items.len) };
        const tok_input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const all_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
        defer _ = mlx.mlx_array_free(all_logits);
        const shp = mlx.getShape(all_logits);
        var logits = mlx.mlx_array_new();
        const start = [_]c_int{ 0, shp[1] - 1, 0 };
        const stop = [_]c_int{ shp[0], shp[1], shp[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&logits, all_logits, &start, 3, &stop, 3, &strides, 3, self.xfm.s));
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = logits;
        self.has_pending_logits = true;
        self.next_token_id = forced[forced.len - 1];
        return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = false };
    }

    /// Result of one `nextPld` step. Yields 1..=(1+max_draft_len) tokens.
    /// Caller owns `tokens` (must `allocator.free` it).
    pub const PldStepResult = struct {
        /// Tokens to emit this step (always at least the already-decided t1).
        /// On a full-accept, contains [t1, ...all_drafts]. On partial accept j,
        /// contains [t1, draft[0..j]] (the corrected fallback is stored as the
        /// generator's pending `next_token_id`, NOT included here — same
        /// "pending becomes next-step's first" convention as `nextDrafter`).
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (not counting t1). 0..=draft_len.
        accepted_tokens: u32,
        /// Whether n-gram lookup found a candidate this step. False means PLD
        /// degraded to a single regular forward (no speculative work done).
        used_lookup: bool,
    };

    /// PLD draft+verify decode step. The draft comes from an n-gram lookup
    /// over `prompt_ids_owned ++ generated_ids`, NOT a model call — that's
    /// what makes PLD model-agnostic and cheap.
    ///
    /// `key_len` is the n-gram size used for matching (default 3). `draft_len`
    /// is the maximum number of speculative tokens to verify per step (default
    /// 5). Both are clamped to safe upper bounds internally.
    ///
    /// Returns `null` only when generation is already done. When no n-gram
    /// match exists (cold start, novel output), falls back to the regular
    /// `next()` path and returns a single-token result with `used_lookup=false`.
    pub fn nextPld(
        self: *Generator,
        allocator: std.mem.Allocator,
        draft_len: u32,
        key_len: u32,
    ) !?PldStepResult {
        if (self.done) return null;
        // Init never armed PLD for this generator (a caller that simply
        // didn't ask). The dispatching caller's flag alone must never put a
        // verify forward through the trunk. Unlike `spec_disabled_runtime`
        // below this is permanent: no re-enable check can ever resurrect it.
        if (!self.pld_enabled) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return PldStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
                .used_lookup = false,
            };
        }
        // Release-enforced guard (issue #97): PLD cannot honor a grammar
        // constraint or logprobs. These were std.debug.asserts, compiled out in
        // ReleaseFast; fail loud instead of streaming off-schema output if a
        // dispatch bug ever routes such a request here.
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        // Runtime acceptance gate: if a prior step set the flag, fall back
        // to the regular `next()` path. Under v2, PLD's exit invariant has
        // `t1 NOT in cache` (matches `nextDrafter`) — `next()`'s transition
        // shim seeds `pending_logits` synchronously via `forward([t1])` when
        // it sees `!has_pending_logits and !has_pending_token`. So the
        // hand-off works even though pending state is empty.
        if (self.spec_disabled_runtime) {
            self.disabled_steps += 1;
            // Periodic re-enable check: when the generated tail turns
            // repetitive (file/tool echo after a novel preamble), PLD pays
            // again. Drain the `next()` pipeline back to the spec entry
            // invariant; the drained token (if any) is this step's emit and
            // speculation resumes on the following call.
            if (self.disabled_steps % SPEC_REENABLE_INTERVAL == 0) reenable: {
                const gen = self.generated_ids.items;
                const prompt_toks = self.prompt_ids_owned;
                const committed_check = try allocator.alloc(u32, prompt_toks.len + gen.len);
                defer allocator.free(committed_check);
                @memcpy(committed_check[0..prompt_toks.len], prompt_toks);
                @memcpy(committed_check[prompt_toks.len..], gen);
                //if (log.isDebug()) {
                //    const dbg_frac = pld_index.tailMatchFraction(committed_check, @min(SPEC_REENABLE_WINDOW, gen.len), 3);
                //    log.debug("  pld re-enable check: disabled_steps={d} gen={d} tail_match={d:.2}\n", .{ self.disabled_steps, gen.len, dbg_frac });
                //}
                if (!specShouldReenable(committed_check, gen.len)) break :reenable;
                switch (try self.drainPipelineForSpec(allocator)) {
                    .stay_disabled => break :reenable,
                    .stopped => return null,
                    .already_clean => {
                        //log.info("  pld=re-enabled (generated tail turned repetitive after {d} disabled steps)\n", .{self.disabled_steps});
                        self.spec_disabled_runtime = false;
                        self.disabled_steps = 0;
                        self.yield_steps = 0;
                        self.yield_accepted = 0;
                        // Invariant already holds — fall through to the
                        // enabled flow below in this same call.
                    },
                    .drained => |drained_tok| {
                        //log.info("  pld=re-enabled (generated tail turned repetitive after {d} disabled steps)\n", .{self.disabled_steps});
                        self.spec_disabled_runtime = false;
                        self.disabled_steps = 0;
                        self.yield_steps = 0;
                        self.yield_accepted = 0;
                        const tokens = try allocator.alloc(u32, 1);
                        tokens[0] = drained_tok;
                        return PldStepResult{
                            .tokens = tokens,
                            .accepted_tokens = 0,
                            .used_lookup = false,
                        };
                    },
                }
            }
            if (self.spec_disabled_runtime) {
                const tok_opt = try self.next(allocator);
                if (tok_opt == null) return null;
                const tokens = try allocator.alloc(u32, 1);
                tokens[0] = tok_opt.?;
                return PldStepResult{
                    .tokens = tokens,
                    .accepted_tokens = 0,
                    .used_lookup = false,
                };
            }
        }

        if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

        const xfm = self.xfm;
        const s = xfm.s;

        // ── INVARIANT going INTO this call (mirrors `nextDrafter`) ──
        //   cache.step = prompt_len + tokens_emitted   (NOT + 1)
        //   t1 = next_token_id (= "this step's first emit"); NOT in cache yet.
        //   pending_logits / pending_token are empty (init's PLD branch and
        //   every nextPld exit leave them empty under v2).
        //
        // Cold path (no n-gram match): forward([t1]) length 1 advances cache
        // by 1, produces logits at position +1 → sample lookahead → emit t1,
        // set next_token_id = lookahead. Loses A's lazy pipeline overlap on
        // cold steps; the prompt-time n-gram gate disables PLD on novel
        // content where cold-path dominates.
        //
        // Verify path: input = `[t1, draft[0..m-1]]` length 1+m. Walk
        // verify_logits[i] vs draft[i] for i=0..m-1; full accept commits 1+m
        // tokens and exits with cache at prompt_len + TE_new (no post-step
        // forward — that is the per-step saving over v1).
        const t1: u32 = self.next_token_id;

        // Cap draft_len so the verify forward stays a small fixed cost.
        const max_draft: u32 = @min(draft_len, 15);
        const klen: u32 = @max(@as(u32, 1), key_len);

        // ── Phase 1: Lookup ──
        // committed = prompt + generated_ids + [t1]. Key = trailing klen tokens
        // (ends at t1). The lookup returns candidates for "what comes after t1".
        const prompt = self.prompt_ids_owned;
        const generated = self.generated_ids.items;
        const total_len = prompt.len + generated.len + 1;

        var committed = try allocator.alloc(u32, total_len);
        defer allocator.free(committed);
        @memcpy(committed[0..prompt.len], prompt);
        @memcpy(committed[prompt.len .. prompt.len + generated.len], generated);
        committed[total_len - 1] = t1;

        var draft_slice: ?[]const u32 = null;
        if (klen <= total_len - 1) {
            const key_start = total_len - klen;
            const key = committed[key_start..total_len];
            const lookup = pld_index.PldLookup{ .committed = committed, .key_len = klen };
            draft_slice = lookup.findMatch(key, max_draft);
        }
        if (draft_slice) |d| {
            if (d.len == 0) draft_slice = null;
        }

        const stochastic = self.sampling.temperature > 0.01;

        // ── Phase 2: Cold path (no n-gram match) ──
        // Forward([t1]) length 1: cache.step += 1, produces logits at that
        // position. Sample the lookahead, emit t1, set next_token_id =
        // lookahead. Cache exits at prompt_len + TE_new where TE_new = TE + 1.
        if (draft_slice == null) {
            const t1_i32: i32 = @intCast(t1);
            const t1_shape = [_]c_int{ 1, 1 };
            const t1_input = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(t1_input);

            const cold_logits = try xfm.forwardWith(&self.ctx, t1_input); // cache.step += 1
            defer _ = mlx.mlx_array_free(cold_logits);

            const lazy = self.sampleLazy(cold_logits);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var lv: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&lv, lazy));
            _ = mlx.mlx_array_free(lazy);
            const new_t1: u32 = @intCast(lv);

            try self.generated_ids.append(allocator, t1);
            self.advanceStep(1);
            self.next_token_id = new_t1;

            // Yield gate: cold steps pay the unpipelined forward; if the
            // workload isn't yielding accepted drafts to pay for it, fall
            // back to the pipelined `next()` (re-enable check above can
            // bring PLD back when the tail turns repetitive).
            self.yield_steps += 1;
            if (yieldGateShouldDisable(self.yield_steps, self.yield_accepted)) {
                log.info(
                    "  pld=disabled (yield gate: {d} drafted tokens over {d} steps < {d:.2}/step)\n",
                    .{ self.yield_accepted, self.yield_steps, YIELD_GATE_MIN_YIELD },
                );
                self.spec_disabled_runtime = true;
                self.disabled_steps = 0;
            }

            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = t1;
            return PldStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
                .used_lookup = false,
            };
        }

        const draft = draft_slice.?;
        const m: u32 = @intCast(draft.len);

        // ── Phase 3: Snapshot KV + moe_seq_offset ──
        // Cache enters at cache.step = prompt_len + TE.
        var kv_snap = try self.ctx.cache.snapshot();
        defer kv_snap.deinit();
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 4: Verify forward `[t1, draft[0..m-1]]` length 1+m ──
        // cache.step at start = prompt_len + TE; after = prompt_len + TE + 1 + m.
        //   verify_logits[0]   predicts the slot AFTER t1     → candidate for draft[0]
        //   verify_logits[i]   predicts the slot AFTER draft[i-1] (i = 1..m-1)
        //                                                     → candidate for draft[i]
        //   verify_logits[m]   predicts the slot AFTER draft[m-1]
        //                                                     → "bonus" position (full-accept new_t1)
        const seq_len: c_int = @intCast(1 + m);
        const verify_input_buf = try allocator.alloc(i32, 1 + m);
        defer allocator.free(verify_input_buf);
        verify_input_buf[0] = @intCast(t1);
        for (draft, 0..) |d, i| verify_input_buf[1 + i] = @intCast(d);
        const verify_shape = [_]c_int{ 1, seq_len };
        const verify_input = mlx.mlx_array_new_data(verify_input_buf.ptr, &verify_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(verify_input);

        var verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
        errdefer _ = mlx.mlx_array_free(verify_logits);
        // verify_logits shape [1, 1+m, V]. Sliced and freed below.
        self.pld_attempted += 1;

        const vl_shape = mlx.getShape(verify_logits);
        const slice_strides = [_]c_int{ 1, 1, 1 };

        // Slice all 1+m per-position logits up front so we can sample the
        // correction from the original verify forward (cache state aligned)
        // without re-running forward, and re-use them for both stochastic
        // accept tests and the correction sample.
        const per_pos_logits = try allocator.alloc(mlx.mlx_array, 1 + m);
        var per_pos_logits_n: usize = 0;
        defer {
            for (per_pos_logits[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(per_pos_logits);
        }
        for (per_pos_logits, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            per_pos_logits_n = idx + 1;
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
        _ = mlx.mlx_array_free(verify_logits);
        verify_logits = .{ .ctx = null };

        // ── Phase 5: Walk drafts. accepted ∈ [0, m]. Full accept = m. ──
        // verify_logits[i] is the prediction for draft[i] (i = 0..m-1).
        // No separate "first-position" test under v2 — the verify forward
        // covers it.
        var accepted: u32 = 0;
        if (stochastic) {
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const target_p = try probsAtLastPos(per_pos_logits[i], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, draft[i], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                var argmax_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(argmax_arr);
                try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, per_pos_logits[i], 2, false, s));
                try mlx.check(mlx.mlx_array_eval(argmax_arr));
                var argmax_val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&argmax_val, argmax_arr));
                if (@as(u32, @intCast(argmax_val)) != draft[i]) break;
                accepted += 1;
            }
        }
        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );
        const full_accept = accepted == m;

        // ── Phase 6: Sample new_t1 from per_pos_logits[accepted] ──
        //   - full accept (accepted == m): per_pos_logits[m] predicts the slot
        //     after the last accepted draft (= "bonus" token).
        //   - partial (accepted < m):  per_pos_logits[accepted] is the model's
        //     prediction at the rejected slot. Stochastic samples from the
        //     residual `max(target_p − one_hot(draft[accepted]), 0)` to preserve
        //     the marginal distribution conditional on "not draft[accepted]"
        //     (Leviathan et al). Greedy: argmax of the rejected slot's logits.
        //
        // This indexing differs from v1: v1 sampled from `verify_logits[accepted-1]`
        // because t1 occupied no input slot; v2 has t1 at index 0 of the verify
        // input, so the "prediction one past the last accepted" lives at
        // index `accepted`. Off-by-one here would silently corrupt output.
        const correction_logits = per_pos_logits[accepted];
        const new_t1: u32 = blk: {
            if (stochastic) {
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (!full_accept) {
                    const onehot = try pldOneHotRow(draft[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try sampleResidual(probs, onehot, self.correctionDraw(accepted), s);
                } else {
                    break :blk try sampleFromProbs(probs, self.correctionDraw(accepted), s);
                }
            } else {
                const lazy = self.sampleLazy(correction_logits);
                try mlx.check(mlx.mlx_array_eval(lazy));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                _ = mlx.mlx_array_free(lazy);
                break :blk @intCast(v);
            }
        };

        // ── Phase 7: Cache rollback on partial accept ──
        // After verify (length 1+m), cache.step = prompt_len + TE + 1 + m.
        // Full accept: TE_new = TE + 1 + m → no rollback.
        // Partial: must land at prompt_len + TE + 1 + accepted = prompt_len + TE_new
        // (TE_new = TE + 1 + accepted). Rollback then re-forward
        // `[t1, draft[0..accepted-1]]` length 1+accepted (with hidden capture
        // not needed here — just the cache advance).
        //
        // The accepted=0 case (= first draft rejected) MUST still re-forward
        // [t1] length 1: in v1 the t1 forward had been done eagerly before
        // verify; v2 includes t1 IN the verify forward, so rollback rolls
        // both t1 AND the drafts. Skipping the re-forward here would leave
        // the cache at prompt_len + TE — one short of the post-emit invariant.
        if (!full_accept) {
            try self.ctx.cache.restore(&kv_snap);
            self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

            const re_seq_len: c_int = @intCast(1 + accepted);
            const re_input_buf = try allocator.alloc(i32, 1 + accepted);
            defer allocator.free(re_input_buf);
            re_input_buf[0] = @intCast(t1);
            for (draft[0..accepted], 0..) |d, i| re_input_buf[1 + i] = @intCast(d);
            const re_shape = [_]c_int{ 1, re_seq_len };
            const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(re_input);
            const re_logits = try xfm.forwardWith(&self.ctx, re_input);
            _ = mlx.mlx_array_free(re_logits);
        }

        // ── Phase 8: Commit emitted tokens ──
        // Tokens emitted: [t1, draft[0..accepted]] = 1 + accepted.
        const num_emit: u32 = 1 + accepted;
        const tokens = try allocator.alloc(u32, num_emit);
        tokens[0] = t1;
        for (draft[0..accepted], 0..) |d, i| tokens[1 + i] = d;

        try self.generated_ids.append(allocator, t1);
        for (draft[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        self.pld_accepted_tokens += accepted;
        self.advanceStep(num_emit);

        // No post-step forward — `next_token_id = new_t1` and exit. The next
        // nextPld call sees t1 NOT in cache (new invariant).
        self.next_token_id = new_t1;

        // Yield-gate accounting for verify steps (cold steps update in their
        // own branch above).
        self.yield_steps += 1;
        self.yield_accepted += accepted;

        // Runtime acceptance gate: after warmup, if the per-draft acceptance
        // probability is below the threshold, disable speculation for the rest
        // of this request (the re-enable check can bring it back when the
        // generated tail turns repetitive). PLD's `drafts_per_round` is the
        // upper-bound draft length (`max_draft`); matches with shorter accepts
        // still divide by this max so a workload with consistently-short
        // n-gram matches DOES get throttled.
        if (runtimeGateShouldDisable(self.pld_attempted, self.pld_accepted_tokens, max_draft)) {
            const drafts_proposed: u64 = self.pld_attempted * @as(u64, max_draft);
            const rate: f32 = if (drafts_proposed > 0)
                @as(f32, @floatFromInt(self.pld_accepted_tokens)) /
                    @as(f32, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  pld=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
                .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.pld_attempted },
            );
            self.spec_disabled_runtime = true;
            self.disabled_steps = 0;
        }

        return PldStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
            .used_lookup = true,
        };
    }

    /// Result of one `nextDrafter` step. Same shape as PLD's result so the
    /// outer wrapper can share token-emit / EOS-check logic.
    pub const DrafterStepResult = struct {
        /// Tokens to emit this step. On a full accept this is
        /// `[t1, ...all_drafts]` (length `block_size`); on partial accept j
        /// it is `[t1, draft[0..j]]` (length `1+j`). The corrected fallback
        /// becomes `next_token_id` for the next call.
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (excludes always-accepted t1).
        accepted_tokens: u32,
    };

    /// Drafter-assisted decode step. Mirrors `nextPld` but the draft comes
    /// from `block_size - 1` autoregressive forwards through the Gemma 4
    /// assistant drafter (cross-attending into target's KV) instead of an
    /// n-gram lookup. Verify is identical: target forward over
    /// `[t1, draft0..draft_{m-1}]` with greedy / stochastic accept.
    ///
    /// Algorithm:
    ///   1. Run `block_size - 1` drafter steps. Each step's input is
    ///      `concat(target.embed(prev_tok)*scale, h_prev)`. `prev_tok` starts
    ///      at `next_token_id` (= t1); after step i it's the just-sampled
    ///      `draft[i]`. `h_prev` starts at `last_hidden` (captured at
    ///      prefill or the previous accept's verify-forward); after step i
    ///      it's the drafter's own `post_proj` output.
    ///      All drafter forwards in one round share `rope_offset =
    ///      target.cache.step` (per upstream `set_shared_kv`).
    ///   2. Snapshot KV + recurrent state, run target verify forward over
    ///      `[t1, draft0..draft_{m-1}]` length `block_size` with
    ///      `forwardCaptureHidden` so we get the new `h_prev` at position m.
    ///   3. Walk argmax(verify_logits[i]) vs draft[i] for i in 0..m-1.
    ///      Greedy: equal → accept. Stochastic: standard speculative-decoding
    ///      ratio test using `probAt(target_p, draft[i])` (the drafter's
    ///      masked-LM-head produces probabilistic logits, so we treat its
    ///      sampled draft as a one-hot proposal — same simplification PLD
    ///      uses).
    ///   4. Full accept (j == m): emit drafts, sample new pending from
    ///      verify_logits[m-1] (the target's prediction one position past the
    ///      last accepted draft — already computed during verify), update
    ///      `last_hidden` to the captured post-final-norm hidden.
    ///   5. Partial accept (j < m): roll back KV+state, re-forward
    ///      `[t1, draft[0..j-1]]` length `j+1` (with hidden capture) so
    ///      cache lands at exactly `+j+1`. Sample correction from the
    ///      *original* verify_logits[j] (the model's prediction at the
    ///      rejected position).
    /// The sampling for a verify round's correction token (the bonus or the residual
    /// draw after `accepted` drafts): keyed at that token's output position, so a
    /// seeded request replays under every drafter and no two positions share a key.
    fn correctionDraw(self: *const Generator, accepted: u32) SamplingParams {
        var sp = self.sampling;
        sp.draw = self.generated_ids.items.len + accepted + 2;
        return sp;
    }

    pub fn nextDrafter(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        std.debug.assert(self.drafter != null);
        std.debug.assert(self.has_last_hidden); // captured at init or last accept
        // Release-enforced guard (issue #97): the drafter path cannot honor a
        // grammar constraint or logprobs (compiled-out asserts before).
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        // Runtime acceptance gate: if a prior step set the flag, fall back
        // to the regular `next()` path. Drafter's exit invariant is "t1 NOT
        // in cache" (different from `next()`'s expected entry), so `next()`
        // contains a transition shim that synchronously seeds pending_logits
        // when has_pending_logits is false. The shim makes this hand-off safe.
        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
            };
        }

        if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

        const xfm = self.xfm;
        const s = xfm.s;
        const drafter = self.drafter.?;
        const m: u32 = @max(@as(u32, 1), self.drafter_block_size - 1);
        const t1: u32 = self.next_token_id; // already-decided token at position cache.step

        // RoPE offset: position the drafter's queries rotate by. Per upstream
        // `set_shared_kv`, this is `target.cache.step` and stays constant
        // across all `m` drafter steps in this round.
        const rope_offset: c_int = @intCast(self.ctx.cache.step);

        // ── Phase 1: draft `m` tokens lazily, no per-step CPU sync ──
        //
        // The drafter loop builds a chained lazy graph: each step's sampled
        // token is a [1]-shaped mlx_array fed directly to the next step's
        // `embedTargetTokenArr` as the indexer, and forward as the next step's
        // `prev_token`. No `mlx_array_eval` / `mlx_array_item_int32` calls
        // here — the entire m-step chain plus the verify forward (built
        // below) materialize as a single async graph and evaluate together.
        // For block_size=8 (31B), this collapses 7 GPU→CPU syncs into 0,
        // saving ~70-100ms of Metal command-buffer sync latency per round.
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);

        // `draft_arrs[i]` is the lazy [1] argmax output of drafter step i.
        // Owned here; freed at end of nextDrafter (after verify uses them).
        const draft_arrs = try allocator.alloc(mlx.mlx_array, m);
        var draft_arrs_n: usize = 0;
        defer {
            for (draft_arrs[0..draft_arrs_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(draft_arrs);
        }

        // Wrap t1 as a [1] mlx_array so the FIRST drafter step can use the
        // same lazy-chain helper as subsequent steps. This array is also
        // reshaped + reused as the leading element of the verify input below.
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{1};
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);

        // `h_prev_owner` rolls forward through the drafter. Starts at the
        // captured target hidden; subsequent steps use the drafter's
        // post_proj output. The output is itself a lazy mlx_array, so the
        // chain stays lazy across all m steps.
        var h_prev_owner: ?mlx.mlx_array = null;
        defer if (h_prev_owner) |h| {
            _ = mlx.mlx_array_free(h);
        };

        {
            var prev_tok_arr: mlx.mlx_array = t1_arr;
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const h_prev_arg: mlx.mlx_array = if (h_prev_owner) |h| h else self.last_hidden;
                const step_out = try drafter_mod.stepArr(drafter, xfm, self.ctx.cache, prev_tok_arr, h_prev_arg, rope_offset);
                // Sample lazily — `sampleTokenLazy` for greedy returns the
                // argmax as a [1]-shaped lazy array. NO eval here.
                draft_arrs[i] = self.sampleLazy(step_out.logits);
                draft_arrs_n = i + 1;
                _ = mlx.mlx_array_free(step_out.logits);

                // Roll h_prev forward.
                if (h_prev_owner) |h_old| {
                    _ = mlx.mlx_array_free(h_old);
                }
                h_prev_owner = step_out.h_prev_next;
                // The next step's prev_token is THIS step's lazy sample.
                prev_tok_arr = draft_arrs[i];
            }
        }

        // ── Phase 2: snapshot KV ──
        var kv_snap = try self.ctx.cache.snapshot();
        defer kv_snap.deinit();
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 3: build verify input by concatenating [t1, drafts...] ──
        //
        // Build verify_input as a [1, 1+m] tensor without any CPU sync. The
        // m draft tokens are still lazy mlx_arrays at this point; we reshape
        // each [1] → [1,1] and stack along axis=1 with t1 reshaped the same
        // way. The forward pass that consumes verify_input is then chained
        // onto the drafter's lazy graph.
        const reshape_2d = [_]c_int{ 1, 1 };
        var t1_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1_2d);
        try mlx.check(mlx.mlx_reshape(&t1_2d, t1_arr, &reshape_2d, 2, s));

        // Stack: each draft_arr[i] is shape [1]; reshape each to [1,1] and
        // collect into a vector_array along with t1_2d, then concat axis=1.
        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
            var drafts_2d_n: usize = 0;
            defer {
                for (drafts_2d[0..drafts_2d_n]) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(drafts_2d);
            }
            for (draft_arrs, drafts_2d) |dlazy, *out| {
                out.* = mlx.mlx_array_new();
                drafts_2d_n += 1;
                try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
            for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        var new_hidden = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(new_hidden);
        // Captures the post-final-norm hidden at the LAST input position
        // (= position m, predicting the bonus token if all drafts accept).
        var verify_logits = try xfm.forwardWithCapture(&self.ctx, verify_input, &new_hidden);
        errdefer _ = mlx.mlx_array_free(verify_logits);
        // verify_logits shape: [1, 1+m, V]
        self.drafter_attempted += 1;

        // ── Phase 4: decide longest accepted prefix ──
        //
        // Greedy mode: argmax over the entire [1, 1+m, V] verify_logits in
        // one op (yields [1, 1+m] indices). Stochastic mode: sample-residual
        // / accept-prob path needs per-position logits, so it slices below.
        // Either way, we collapse all per-step syncs into ONE eval at the
        // end of this round.
        const stochastic = self.sampling.temperature > 0.01;
        const vl_shape = mlx.getShape(verify_logits);

        // Stochastic path needs per-position logits to compute target probs
        // and (on partial accept) build the residual. Greedy path skips
        // slicing entirely. `per_pos_logits` is null in greedy mode.
        var per_pos_logits: ?[]mlx.mlx_array = null;
        var per_pos_logits_n: usize = 0;
        defer if (per_pos_logits) |slots| {
            for (slots[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        if (stochastic) {
            const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
            per_pos_logits = slots;
            const slice_strides = [_]c_int{ 1, 1, 1 };
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                per_pos_logits_n = idx + 1;
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
            }
        }

        // Build the greedy argmax tensor lazily; it'll be eval'd alongside
        // the rest of the round below. Its entries are COMMITTED, so they mask
        // reserved ids like every other block decoder's.
        var verify_argmax: CommittedArgmax = .{};
        defer verify_argmax.deinit();
        if (!stochastic) {
            verify_argmax = try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
        }
        _ = mlx.mlx_array_free(verify_logits);
        verify_logits = .{ .ctx = null };

        // ── Phase 4b: batched eval — drafts + verify_argmax + new_hidden ──
        //
        // Submit the entire round (drafter chain + verify forward + argmax)
        // to the GPU in a single async dispatch. Then sync ONCE per array we
        // need on the CPU. For block_size=8, this collapses ~14 individual
        // sync points (7 drafter samples + 7 per-position argmaxes in the
        // old code) into approximately 2: one effective sync to wait for
        // GPU completion (the first `mlx_array_eval`), and zero-cost evals
        // afterward since the work is already done.
        //
        // CORRECTNESS: `mlx_array_data_int32` only returns valid data once
        // the array is eval'd. We explicitly eval each array we will read.
        // `verify_input` is NOT eval'd separately because MLX may fuse it
        // into the forward pass without materializing a CPU-readable buffer
        // — instead we read drafts via per-array `mlx_array_item_int32` on
        // each `draft_arrs[i]` (cheap after the first sync).
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            if (!stochastic) {
                _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
            }
            _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        // Extract drafts. First eval sync waits for the GPU; subsequent
        // evals are no-ops since they were queued together.
        for (draft_arrs, 0..) |arr, idx| {
            try mlx.check(mlx.mlx_array_eval(arr));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, arr));
            drafts[idx] = @intCast(v);
        }
        if (!stochastic) {
            // Force verify_argmax to materialize before bulk-reading. It's a
            // separate branch from the drafter chain (drafts → concat →
            // verify → argmax), so eval'ing the drafts above doesn't pull
            // verify_argmax along with them. This was the v26.5.6 bug that
            // produced 0% acceptance on 26B/31B (verify ran longer than the
            // drafter chain, so the data buffer was read while the GPU was
            // still writing it).
            try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));
        }

        var accepted: u32 = 0;
        if (stochastic) {
            // Stochastic verify (Leviathan et al. probability-ratio test).
            // The drafted token came from argmax of the drafter's masked LM
            // head, so we treat it as a one-hot proposal: accept with
            // probability `min(1, target_p[draft[i]])`, otherwise stop and
            // sample from the residual at the rejected position.
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, drafts[k], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            // Bulk-read the [1, 1+m] argmax indices and scan for first
            // mismatch in CPU. No more GPU syncs in this branch.
            const argmax_data = try verify_argmax.ids(1 + m);
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_argmax: u32 = @intCast(argmax_data[k]);
                if (target_argmax != drafts[k]) break;
                accepted += 1;
            }
        }

        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );

        // Sample the next pending token from the verify output at position
        // `accepted`:
        //   - full accept (accepted == m): position m predicts the bonus
        //     token one past the last draft.
        //   - partial accept: position `accepted` predicts the model's
        //     replacement for the rejected draft.
        // For greedy, position `accepted`'s argmax is already in
        // `argmax_data[accepted]` — no extra GPU work. For stochastic, we
        // need the actual probability distribution at that position, so we
        // sample from `per_pos_logits[accepted]` (with residual correction
        // on partial accept per Leviathan et al).
        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    const onehot = try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try sampleResidual(probs, onehot, self.correctionDraw(accepted), s);
                } else {
                    break :blk try sampleFromProbs(probs, self.correctionDraw(accepted), s);
                }
            } else {
                // Greedy: reuse the bulk-read argmax row. Already eval'd in
                // the single async eval above; no GPU sync here.
                const argmax_data = try verify_argmax.ids(1 + m);
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        // ── Phase 5: commit / rollback ──
        if (accepted == m) {
            // Full accept: cache at +1+m. Emit [t1, ...drafts]. Pending = next_pending.
            // The captured `new_hidden` is the post-final-norm hidden at
            // position m — the last accepted draft's position. That's the
            // h_prev for the NEXT round (drafting from t = next_pending; the
            // hidden corresponds to draft[m-1], which is what next_pending
            // follows). This matches the convention `nextDrafter` uses.
            const tokens = try allocator.alloc(u32, 1 + m);
            tokens[0] = t1;
            for (drafts, 0..) |d, idx| tokens[1 + idx] = d;

            try self.generated_ids.append(allocator, t1);
            for (drafts) |d| try self.generated_ids.append(allocator, d);

            if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
            self.last_hidden = new_hidden;
            self.has_last_hidden = true;
            new_hidden = .{ .ctx = null };

            self.drafter_accepted_tokens += m;
            self.next_token_id = next_pending;
            self.advanceStep(1 + m);

            // drafts buffer transferred into tokens copy; free original.
            allocator.free(drafts);
            self.checkDrafterRuntimeGate();
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = m,
            };
        }

        // Partial accept (accepted < m). Cache over-advanced by (m - accepted).
        // The captured new_hidden is for position m (which we're rolling back
        // past) - discard it. Roll back the KV cache, then re-forward
        // [t1, drafts[0..accepted]] length 1+accepted with hidden capture so
        // last_hidden lands at the position immediately past the last
        // accepted draft (where next_pending will live).
        _ = mlx.mlx_array_free(new_hidden);
        new_hidden = .{ .ctx = null };

        try self.ctx.cache.restore(&kv_snap);
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

        const re_seq_len: c_int = @intCast(1 + accepted);
        const re_input_buf = try allocator.alloc(i32, 1 + accepted);
        defer allocator.free(re_input_buf);
        re_input_buf[0] = @intCast(t1);
        for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
        const re_shape = [_]c_int{ 1, re_seq_len };
        const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(re_input);

        var re_new_hidden = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(re_new_hidden);
        const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
        _ = mlx.mlx_array_free(re_logits);

        const tokens = try allocator.alloc(u32, 1 + accepted);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = re_new_hidden;
        self.has_last_hidden = true;
        re_new_hidden = .{ .ctx = null };

        self.drafter_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(1 + accepted);

        allocator.free(drafts);
        self.checkDrafterRuntimeGate();
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// DFlash block-drafter step. ONE assistant forward proposes
    /// `block_size - 1` drafts, conditioned on the trunk's cached
    /// target_layer_ids hiddens (the per-request `dflash_ctx`); the trunk
    /// verify over `[t1, drafts...]` follows the standard spec invariant
    /// (`cache.step = prompt_len + emitted`, t1 NOT in cache on entry,
    /// correction from ORIGINAL `verify_logits[accepted]`).
    ///
    /// Rollback is an offset-only `cache.truncate` — legal because
    /// `DflashModel.bind` restricts targets to the pure-KVCache standard
    /// path (no recurrent state, no module-owned state), where a position's K/V depend
    /// only on earlier positions (causal) and are identical whether computed
    /// in a width-16 verify or any re-forward. No re-forward runs at all:
    /// the next round's context comes from slicing THIS round's verify
    /// captures to the committed prefix — exactly the reference's
    /// `hidden_states[i+1][:, :n_accepted]`.
    ///
    /// Drafts are greedy (argmax over the trunk lm_head on assistant
    /// hiddens, anchor row DROPPED — reference `[:, 1:]`); sampled requests
    /// use the same one-hot Leviathan acceptance the drafter/PLD paths use.
    pub fn nextDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        // The budget's last token is the previous verify's: emit it without a round. (A pipelined plain
        // stretch has no such token: its last one comes out of `next`.)
        if (!self.done and !self.spec_disabled_runtime and self.max_tokens -| self.completion_tokens == 1 and !self.has_pending_logits and !self.has_pending_token) {
            if (try self.checkStop()) return null;
            const tokens = try allocator.alloc(u32, 1);
            errdefer allocator.free(tokens);
            tokens[0] = self.next_token_id;
            try self.generated_ids.append(allocator, tokens[0]);
            self.advanceStep(1);
            self.unforwarded_tail = 1;
            self.done = true;
            self.finish_reason = "length";
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }
        // The kv term is the same physics for either block decoder (one
        // forward, one KV read, shared across the block's rows), so a DFlash
        // round is an observation for it too — and on a DFlash-only server
        // it is the ONLY source.
        const dflash_kv_watch = io_util.Stopwatch.init(self.timer.io);
        // The round-cost table sees the same round: drafts = block - 1, or
        // 0 when the runtime gate fell back to serial (a serial sample is
        // the "no spec" candidate a width chooser needs).
        const dflash_gen_before = self.generated_ids.items.len;
        self.dflash_round_width = 0;
        self.dflash_unobserved = false;
        const dflash_rounds_before: u64 = if (self.dflash_chooser) |ch| ch.rounds else self.dflash_attempted + self.dflash_plain;
        defer {
            const ms = @as(f32, @floatFromInt(dflash_kv_watch.read())) / @as(f32, std.time.ns_per_ms);
            const emitted = self.generated_ids.items.len - dflash_gen_before;
            const post_warmup = dflash_rounds_before >= dflashGateWarmup();
            const wall = if (post_warmup) self.specRoundWallMs(ms) else ms;
            self.specObserveRound(self.dflash_round_width, wall, @floatFromInt(emitted), post_warmup and emitted > 0 and !self.dflash_unobserved, false);
            if (self.dflash_chooser) |*ch| ch.note(self.dflash_round_width);
        }
        if (self.done) return null;
        std.debug.assert(self.dflash != null);
        std.debug.assert(self.dflash_ctx != null);
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }

        // The round policy: a plain tick runs on the serial pipeline; every other round starts from the drained state.
        var policy_draft = true;
        if (self.dflash_pol) |*pol| {
            policy_draft = pol.drafting.shouldDraft();
            if (!policy_draft and self.dflashCanPipeline()) return try self.dflashPlainPipelined(allocator);
            pol.plain_watch = null;
            if (self.has_pending_logits or self.has_pending_token) return try self.dflashLeavePipeline(allocator);
        }

        // Per-round width: the chooser's argmax over measured tokens/ms
        // (serial = width 0 is a candidate, so "serial wins" is the gate),
        // the fixed block until it has data.
        var round_width: u32 = @max(self.dflash_block_size, 2) - 1;
        if (self.dflash_chooser) |*ch| {
            if (ch.rounds >= dflashGateWarmup()) {
                const d = if (specDepthPolicy() == .accept) self.dflashAcceptChoose(ch) else ch.choose(&self.xfm.round_cost, self.specKvLen(), ch.rounds);
                round_width = d.width;
                // Serial is sticky: a plain decode round does not extend the
                // assistant context, so there is no way back this request.
                if (round_width == 0) self.spec_disabled_runtime = true;
                if (ch.logged != ch.current) {
                    if (self.xfm.round_cost.bucketToRead(self.specKvLen())) |b| {
                        ch.logged = ch.current;
                        var buf: [256]u8 = undefined;
                        log.info("[dflash] width chooser: standing w{d} ({s}) from {s} {s} (ms/tok)\n", .{
                            ch.current,
                            if (ch.current == 0) "serial" else "block",
                            round_cost.bucketName(self.xfm.round_cost.layout, b),
                            self.xfm.round_cost.formatBucket(b, &buf),
                        });
                    }
                }
            }
        }
        // The round policy replaces the chooser and the serial fallback: a round with no drafts is the verify
        // forward at one row, so DFlash resumes whenever drafting pays again.
        if (self.dflash_pol != null and !policy_draft) round_width = 0;
        self.dflash_round_width = round_width;
        if (round_width == 0 and self.dflash_pol == null) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }

        if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

        const xfm = self.xfm;
        const s = xfm.s;
        const model = self.dflash.?;
        const dctx = &self.dflash_ctx.?;
        // Drafts the assistant proposes (0 = a plain round) and drafts the verify carries: the policy may keep fewer.
        const m_draft: u32 = round_width;
        var m: u32 = m_draft;
        const t1: u32 = self.next_token_id;
        // On the moe/gated-delta path positions come from moe_seq_offset (cache.step
        // is a bookkeeping counter the model never reads there);
        // the standard path keeps cache.step as the anchor.
        const moe_path = self.xfm.moe_layers != null;
        const anchor_pos: usize = if (moe_path) self.ctx.moe_seq_offset.* else self.ctx.cache.step;
        const kv_step_snap = self.ctx.cache.step;
        // A drafted round reads the whole context: whatever the plain rounds parked joins it first.
        if (m_draft > 0) try self.dflashFlushPending(allocator);
        std.debug.assert(dctx.absLen() + self.dflash_pending_rounds == anchor_pos);

        const tracing = dflashTraceEnabled();
        var ph: io_util.Stopwatch = undefined;
        if (tracing) {
            ph = io_util.Stopwatch.init(self.timer.io);
            if (self.dflash_gap_watch) |*gw| self.dflash_trace.add(.gap, gw.read());
            self.dflash_gap_watch = null;
        }

        // Sampled requests accept through the full Leviathan ratio
        // min(1, p/q), so a GREEDY draft is a one-hot q — and at temperature
        // the target row is flat, which is exactly where min(1, p(argmax))
        // collapses. Drafting FROM the request's own distribution makes q
        // track p and keeps acceptance flat across temperature. Greedy
        // requests keep the argmax path untouched, so the byte-equality
        // guard is unaffected.
        // A row-exact trunk samples with the serial sampler's keys: drafts
        // drawn with the verify row's key, rows kept only while they match.
        const serial = self.sampling.keyed and self.sampling.temperature > 0.01 and self.sampling.seed != null;
        const stochastic = self.sampling.temperature > 0.01 and !serial;
        if (m_draft == 0)
            return try self.dflashPlainRound(allocator, t1, serial, stochastic);
        if (self.dflash_pol != null) if (try self.dflashCopyRound(allocator, t1, anchor_pos, kv_step_snap, serial, stochastic, tracing, &ph, dflash_kv_watch)) |res| return res;
        // DFlash2 path selector: when the sidecar ships one, drafts come from
        // the pairwise-scored path trace instead of per-position argmax /
        // block sampling. Greedy requests keep the byte-equality bar (a
        // selector draft only survives verify if it IS the trunk argmax);
        // stochastic requests sample the selector's own candidate softmax and
        // accept through min(1, p/q) with q read off the traced path —
        // exact by construction. `MLX_SERVE_DFLASH_SELECTOR=0` forces the v1
        // arms for A/Bs.
        const use_selector = model.selector != null and dflashSelectorEnabled();
        // DSpark: the block's base logits are position-parallel, but each
        // position's draft is picked from logits CORRECTED by the token
        // drafted at the previous one (the Markov bigram bias). Chaining it
        // is what the head is for — dropping it drafts every position from
        // an uncorrected distribution the sidecar was never trained to emit.
        const use_markov = model.markov != null and dflashMarkovEnabled();
        const tree_round = use_selector and !use_markov and (serial or !stochastic) and xfm.specTreeSupported() and self.dflash_pol == null;
        // A continuation the context backs long enough to fill the tree is
        // verified as the round's chain, with no assistant forward.
        if (tree_round and self.dflash_copy_silent > 0) {
            self.dflash_copy_silent -= 1;
        } else if (tree_round) {
            const prompt = self.prompt_ids_owned;
            const gen = self.generated_ids.items;
            const committed = try allocator.alloc(u32, @min(prompt.len + gen.len + 1, DFLASH_COPY_WINDOW));
            defer allocator.free(committed);
            pld_index.tailWindow(committed, prompt, gen, t1);
            if (pld_index.backedCopy(committed, dflashTreeNodes(m), DFLASH_COPY_MATCH)) |c| if (c.draft.len == dflashTreeNodes(m))
                return try self.dflashTreeRound(allocator, model, dctx, null, null, t1, m, anchor_pos, kv_step_snap, serial, c.draft);
        }

        // ── Phase 1: one assistant forward drafts all m tokens ──
        // Row mapping is the export's convention (`anchor_row_drafts`):
        // DFlash reads mask rows 1..bs-1 (anchor row dropped), DSpark reads
        // ALL rows starting at the anchor — so DSpark needs only m noise rows
        // for the same m drafts and the same verify width.
        const noise_rows: u32 = if (model.config.anchor_row_drafts)
            m_draft
        else
            m_draft + 1;
        const noise_ids = try allocator.alloc(i32, noise_rows);
        defer allocator.free(noise_ids);
        noise_ids[0] = @intCast(t1);
        for (noise_ids[1..]) |*v| v.* = @intCast(model.config.mask_token_id);
        const noise_shape = [_]c_int{ 1, @intCast(noise_rows) };
        const noise_input = mlx.mlx_array_new_data(noise_ids.ptr, &noise_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(noise_input);
        // RAW table rows — no embed norm, no scale (the DFlash contract).
        const noise_embeds = try xfm.rawEmbedding(noise_input);
        defer _ = mlx.mlx_array_free(noise_embeds);

        const blk_hidden = try dflash_mod.forwardBlock(model, dctx, noise_embeds, anchor_pos);
        defer _ = mlx.mlx_array_free(blk_hidden);
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(blk_hidden));
            self.dflash_trace.add(.assist, ph.read());
            ph.reset();
        }
        const draft_logits_all = try model.draftLogits(xfm, blk_hidden);
        defer _ = mlx.mlx_array_free(draft_logits_all);

        // Slice the m draft rows: DFlash drops the anchor row (1..bs-1),
        // DSpark reads every row (0..m) — the anchor row IS draft 0.
        const dl_shape = mlx.getShape(draft_logits_all);
        var draft_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_logits);
        {
            const row0: c_int = if (model.config.anchor_row_drafts) 0 else 1;
            const start = [_]c_int{ 0, row0, 0 };
            const stop = [_]c_int{ 1, row0 + @as(c_int, @intCast(m_draft)), dl_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&draft_logits, draft_logits_all, &start, 3, &stop, 3, &strides, 3, s));
        }
        const sample_drafts = stochastic and !use_selector and !use_markov and dflashSampledDraftsEnabled();
        if (tree_round) {
            return try self.dflashTreeRound(allocator, model, dctx, blk_hidden, draft_logits, t1, m, anchor_pos, kv_step_snap, serial, &.{});
        }
        var sel_path: ?dflash_mod.SelectedPath = null;
        defer if (sel_path) |*sp| sp.deinit(allocator);
        var draft_q: mlx.mlx_array = .{ .ctx = null }; // [m, V] proposal density
        defer if (draft_q.ctx != null) {
            _ = mlx.mlx_array_free(draft_q);
        };
        var draft_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_ids);
        if (use_markov) {
            const mh = &model.markov.?;
            const ids_i32 = try allocator.alloc(i32, m);
            defer allocator.free(ids_i32);
            var q_rows: ?[]mlx.mlx_array = null;
            defer if (q_rows) |rows| {
                for (rows) |r| _ = mlx.mlx_array_free(r);
                allocator.free(rows);
            };
            if (stochastic) {
                const rows = try allocator.alloc(mlx.mlx_array, m);
                for (rows) |*r| r.* = .{ .ctx = null };
                q_rows = rows;
            }
            var prev: u32 = t1;
            var step: u32 = 0;
            while (step < m) : (step += 1) {
                var base_row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(base_row);
                {
                    const start = [_]c_int{ 0, @intCast(step), 0 };
                    const stop = [_]c_int{ 1, @as(c_int, @intCast(step)) + 1, dl_shape[2] };
                    const strides = [_]c_int{ 1, 1, 1 };
                    try mlx.check(mlx.mlx_slice(&base_row, draft_logits, &start, 3, &stop, 3, &strides, 3, s));
                }
                const corrected = try mh.stepLogits(base_row, prev, s);
                defer _ = mlx.mlx_array_free(corrected);
                if (stochastic) {
                    // Sample the step from the request's own filtered
                    // distribution and keep the row as q — the accept ratio
                    // and the reject residual both need the density the
                    // draft was actually drawn from.
                    const qrow = try filteredProbsBlock(corrected, self.sampling, s);
                    var logq = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(logq);
                    try mlx.check(mlx.mlx_log(&logq, qrow, s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    var sampled = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(sampled);
                    try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
                    try mlx.check(mlx.mlx_array_eval(sampled));
                    const sd = mlx.mlx_array_data_int32(sampled) orelse return error.MlxArrayDataNull;
                    prev = @intCast(sd[0]);
                    q_rows.?[step] = qrow;
                } else {
                    var amax = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(amax);
                    try mlx.check(mlx.mlx_argmax_axis(&amax, corrected, 2, false, s));
                    var as_i32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(as_i32);
                    try mlx.check(mlx.mlx_astype(&as_i32, amax, .int32, s));
                    try mlx.check(mlx.mlx_array_eval(as_i32));
                    const ad = mlx.mlx_array_data_int32(as_i32) orelse return error.MlxArrayDataNull;
                    prev = @intCast(ad[0]);
                }
                ids_i32[step] = @intCast(prev);
            }
            if (q_rows) |rows| {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (rows) |r| _ = mlx.mlx_vector_array_append_value(vec, r);
                draft_q = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_concatenate_axis(&draft_q, vec, 0, s));
            }
            const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
            const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(host_arr);
            try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
        } else if (use_selector) {
            // A serial-sampled request drafts the greedy path: over 20 seeds it
            // kept more drafts than paths scored with the verify rows' noise.
            const sel_temp: f32 = if (stochastic) self.sampling.temperature else 0.0;
            sel_path = try dflash_mod.selectPath(
                allocator,
                &model.selector.?,
                model.config.selector_top_k,
                blk_hidden,
                draft_logits,
                t1,
                sel_temp,
                self.prng.random(),
                s,
            );
            // The policy verifies the rows the selector's confidence pays for; none at all makes this a plain round.
            if (self.dflash_pol != null) {
                m = self.dflashPolicyRows(&sel_path.?, m_draft) - 1;
                self.dflash_round_width = m;
                if (m == 0) {
                    self.dflash_unobserved = true;
                    const plain = try self.dflashPlainRound(allocator, t1, serial, stochastic);
                    self.dflashRecordDrafted(1, dflash_kv_watch.read());
                    return plain;
                }
            }
            const ids_i32 = try allocator.alloc(i32, m);
            defer allocator.free(ids_i32);
            for (sel_path.?.ids[0..m], ids_i32) |v, *d| d.* = @intCast(v);
            const row_shape = [_]c_int{ 1, @intCast(m) };
            const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(host_arr);
            try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
        } else if (sample_drafts) {
            draft_q = try filteredProbsBlock(draft_logits, self.sampling, s);
            var logq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(logq);
            try mlx.check(mlx.mlx_log(&logq, draft_q, s));
            const null_key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(null_key);
            var sampled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sampled);
            try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
            var as_i32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(as_i32);
            try mlx.check(mlx.mlx_astype(&as_i32, sampled, .int32, s));
            const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
            try mlx.check(mlx.mlx_reshape(&draft_ids, as_i32, &row_shape, 2, s));
        } else {
            var draft_amax = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(draft_amax);
            try mlx.check(mlx.mlx_argmax_axis(&draft_amax, draft_logits, 2, false, s));
            try mlx.check(mlx.mlx_astype(&draft_ids, draft_amax, .int32, s));
        }
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(draft_ids));
            self.dflash_trace.add(.head, ph.read());
            ph.reset();
        }

        return try self.dflashVerify(allocator, .{
            .t1 = t1,
            .m = m,
            .m_draft = m_draft,
            .anchor_pos = anchor_pos,
            .kv_step_snap = kv_step_snap,
            .serial = serial,
            .stochastic = stochastic,
            .tracing = tracing,
            .ph = &ph,
            .watch = dflash_kv_watch,
            .draft_ids = draft_ids,
            .sel_path = if (sel_path) |*sp| sp else null,
            .draft_q = draft_q,
        });
    }

    /// What a DFlash round's verify needs from its draft: the drafts as an int32 `[1, m]` array, the selector's
    /// path (its confidences and q) when there is one, and the density they were drawn from when sampled.
    const VerifyIn = struct {
        t1: u32,
        m: u32,
        m_draft: u32,
        anchor_pos: usize,
        kv_step_snap: usize,
        serial: bool,
        stochastic: bool,
        tracing: bool,
        ph: *io_util.Stopwatch,
        watch: io_util.Stopwatch,
        draft_ids: mlx.mlx_array,
        sel_path: ?*dflash_mod.SelectedPath,
        draft_q: mlx.mlx_array,
    };

    /// Phases 2-4 of a DFlash round: verify `[t1, drafts]` with the layer taps, accept, roll back and commit.
    fn dflashVerify(self: *Generator, allocator: std.mem.Allocator, r: VerifyIn) !DrafterStepResult {
        const xfm = self.xfm;
        const s = xfm.s;
        const model = self.dflash.?;
        const dctx = &self.dflash_ctx.?;
        const moe_path = xfm.moe_layers != null;
        const t1 = r.t1;
        const m = r.m;
        const m_draft = r.m_draft;
        const anchor_pos = r.anchor_pos;
        const kv_step_snap = r.kv_step_snap;
        const serial = r.serial;
        const stochastic = r.stochastic;
        const tracing = r.tracing;
        const ph = r.ph;
        const dflash_kv_watch = r.watch;
        const draft_ids = r.draft_ids;
        const sel_path = r.sel_path;
        const draft_q = r.draft_q;
        const bs: u32 = m + 1;

        // ── Phase 2: verify input [t1, drafts...] — [1, bs] int32, lazy ──
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{ 1, 1 };
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_arr);
            _ = mlx.mlx_vector_array_append_value(vec, draft_ids);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        // Verify forward with layer captures — this round's verify IS the
        // next round's context producer.
        const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
        defer {
            for (cap_out) |a| _ = mlx.mlx_array_free(a);
            allocator.free(cap_out);
        }
        for (cap_out) |*a| a.* = mlx.mlx_array_new();
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = cap_out };
        self.ctx.capture_layers = &cl;
        defer self.ctx.capture_layers = null;
        var verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
        errdefer _ = mlx.mlx_array_free(verify_logits);
        self.dflash_attempted += 1;
        if (tracing) {
            // Captures ride the same forward — eval them here or their cost
            // is billed to `append`.
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, verify_logits);
            for (cap_out) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
            try mlx.check(mlx.mlx_eval(vec));
            self.dflash_trace.add(.verify, ph.read());
            ph.reset();
        }

        // ── Phase 3: decide the longest accepted prefix ──
        const vl_shape = mlx.getShape(verify_logits);
        var per_pos_logits: ?[]mlx.mlx_array = null;
        var per_pos_logits_n: usize = 0;
        defer if (per_pos_logits) |slots| {
            for (slots[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        if (stochastic) {
            const slots = try allocator.alloc(mlx.mlx_array, bs);
            per_pos_logits = slots;
            const slice_strides = [_]c_int{ 1, 1, 1 };
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                per_pos_logits_n = idx + 1;
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
            }
        }
        // Committed entries: masked like every other block decoder's.
        var verify_argmax: CommittedArgmax = .{};
        defer verify_argmax.deinit();
        if (!stochastic) {
            verify_argmax = if (serial)
                try self.verifySerialSamples(verify_logits, 1 + m)
            else
                try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
        }
        _ = mlx.mlx_array_free(verify_logits);
        verify_logits = .{ .ctx = null };

        // One batched dispatch, then read (drafts branch + verify branch are
        // separate graphs — eval BOTH before reading either; the v26.5.6
        // 0%-acceptance class).
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, draft_ids);
            if (!stochastic) _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);
        if (sel_path) |sp| {
            @memcpy(drafts, sp.ids[0..m]);
        } else {
            try mlx.check(mlx.mlx_array_eval(draft_ids));
            const draft_data = mlx.mlx_array_data_int32(draft_ids) orelse return error.MlxArrayDataNull;
            for (drafts, 0..) |*d, idx| d.* = @intCast(draft_data[idx]);
        }
        if (!stochastic) try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));

        var accepted: u32 = 0;
        if (stochastic) {
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, drafts[k], s);
                const accept_prob: f32 = if (draft_q.ctx != null) blk: {
                    const q_row = try sliceProbRow(draft_q, k, s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk specAcceptProb(p_draft, try probAt(q_row, drafts[k], s));
                } else if (sel_path) |sp| blk: {
                    // Selector-sampled draft: q is the traced step's own
                    // candidate softmax — exact, no GPU read.
                    const kk = sp.cand_ids.len / @as(usize, m_draft);
                    break :blk specAcceptProb(p_draft, sp.q.?[@as(usize, k) * kk + sp.chosen_idx[k]]);
                } else @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            const argmax_data = try verify_argmax.ids(1 + m);
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                if (@as(u32, @intCast(argmax_data[k])) != drafts[k]) break;
                accepted += 1;
            }
        }

        // A speculative verify may accept the whole block even when only a
        // few output positions remain. Cap the accepted prefix BEFORE cache
        // and assistant-context commit so every representation of the turn —
        // returned tokens, generated_ids, usage, KV and DFlash context — ends
        // at exactly max_tokens.
        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );
        if (serial) self.sampling.draw = self.generated_ids.items.len + accepted + 2;

        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    // norm(max(0, p − q)) against the q the draft was drawn
                    // from — a one-hot here is the WRONG residual for a
                    // sampled draft, and wrong silently.
                    const q_row = if (draft_q.ctx != null)
                        try sliceProbRow(draft_q, accepted, s)
                    else if (sel_path) |sp|
                        try selectorQRow(sp, accepted, m_draft, vl_shape[2], s)
                    else
                        try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk try sampleResidual(probs, q_row, self.correctionDraw(accepted), s);
                } else {
                    break :blk try sampleFromProbs(probs, self.correctionDraw(accepted), s);
                }
            } else {
                const argmax_data = try verify_argmax.ids(1 + m);
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        if (tracing) {
            self.dflash_trace.add(.accept, ph.read());
            ph.reset();
        }

        // ── Phase 4: commit — truncate on partial accept, then grow the
        // assistant context by exactly the committed positions ──
        const n_commit: usize = 1 + @as(usize, accepted);
        if (accepted < m) {
            try self.ctx.cache.truncate(anchor_pos + n_commit, s);
            if (moe_path) {
                // Bookkeeping for the MoE arm: preserve the pre-verify cache.step
                // (prefix-cache kv_step contract) and re-point moe_seq_offset at
                // the committed length.
                self.ctx.cache.step = kv_step_snap;
                self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
            }
        }
        if (accepted == m) {
            try dflash_mod.appendContext(model, dctx, cap_out, anchor_pos);
        } else {
            const sliced = try allocator.alloc(mlx.mlx_array, cap_out.len);
            var sliced_n: usize = 0;
            defer {
                for (sliced[0..sliced_n]) |a| _ = mlx.mlx_array_free(a);
                allocator.free(sliced);
            }
            for (cap_out, sliced) |full, *out| {
                out.* = mlx.mlx_array_new();
                sliced_n += 1;
                const fsh = mlx.getShape(full);
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ 1, @intCast(n_commit), fsh[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(out, full, &start, 3, &stop, 3, &strides, 3, s));
            }
            try dflash_mod.appendContext(model, dctx, sliced, anchor_pos);
        }
        // Materialize the appended context off the critical path so next
        // round's assistant forward doesn't drag this round's verify graph.
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            dctx.appendEvalArrays(eval_vec);
            if (tracing) {
                try mlx.check(mlx.mlx_eval(eval_vec));
                self.dflash_trace.add(.append, ph.read());
                ph.reset();
            } else {
                try mlx.check(mlx.mlx_async_eval(eval_vec));
            }
        }

        const tokens = try allocator.alloc(u32, n_commit);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;
        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);
        allocator.free(drafts);

        self.dflash_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(@intCast(n_commit));
        if (self.dflash_chooser) |*ch| ch.observe(m, accepted, SPEC_EV_EMA_BETA);
        self.dflash_stats.record(m, accepted);
        if (self.dflash_pol) |*pol| {
            if (sel_path) |sp| {
                for (sp.conf[0..m], 0..) |c, k| {
                    if (k > accepted) break;
                    pol.calib.observe(c, k < accepted);
                }
                if (log.enabled(.debug)) self.logDflashRound(sp.conf[0..m_draft], m, accepted);
            }
        }
        self.dflashRecordDrafted(n_commit, dflash_kv_watch.read());
        // The calibrated sticky gate stays as the bootstrap that gets serial
        // MEASURED (the chooser picks serial only from a measured w0 cell).
        self.checkDflashRuntimeGate();
        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
        }

        if (tracing) {
            self.dflashTraceRoundEnd(accepted);
            self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }

        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    var dflash_copy_off_cache: ?bool = null;
    /// `MLX_SERVE_DFLASH_COPY=0` drafts only from the assistant (the A/B arm for copied continuations).
    fn dflashCopyOff() bool {
        if (dflash_copy_off_cache) |v| return v;
        const raw = std.c.getenv("MLX_SERVE_DFLASH_COPY");
        const off = raw != null and raw.?[0] == '0';
        dflash_copy_off_cache = off;
        return off;
    }

    /// Longest copied continuation a round verifies: the fused row kernels reach 16 rows.
    const DFLASH_COPY_MAX: u32 = dflash_policy.MAX_DRAFTS;

    /// A round whose drafts are the context's own continuation after `t1` (an echo, an edit, a repeated block):
    /// no assistant forward, and as many rows as the cost ladder and the copies' record pay for.
    fn dflashCopyRound(self: *Generator, allocator: std.mem.Allocator, t1: u32, anchor_pos: usize, kv_step_snap: usize, serial: bool, stochastic: bool, tracing: bool, ph: *io_util.Stopwatch, watch: io_util.Stopwatch) !?DrafterStepResult {
        if (dflashCopyOff()) return null;
        const pol = &self.dflash_pol.?;
        const idx = try self.ngramIndex(allocator);
        const found = idx.matchBest(t1, DFLASH_COPY_MAX) orelse return null;
        if (found.suffix < ngram_lookup.MIN_SUFFIX or found.draft.len == 0) return null;
        var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
        const avail: usize = @min(found.draft.len, p.len, @as(usize, self.max_tokens -| self.completion_tokens -| 1));
        if (avail == 0) return null;
        @memset(p[0..avail], pol.copy.probability(found.suffix));
        const costs = dflash_policy.priorCosts(pol.plain.ms);
        const rows = self.dflashRowsFor(p[0..avail], &costs);
        if (rows < 2) return null;
        const m: u32 = rows - 1;
        var ids: [dflash_policy.MAX_DRAFTS]i32 = undefined;
        for (found.draft[0..m], ids[0..m]) |t, *d| d.* = @intCast(t);
        const draft_ids = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, @intCast(m) }, 2, .int32);
        defer _ = mlx.mlx_array_free(draft_ids);
        self.dflash_round_width = m;
        const suffix = found.suffix;
        const res = try self.dflashVerify(allocator, .{
            .t1 = t1,
            .m = m,
            .m_draft = m,
            .anchor_pos = anchor_pos,
            .kv_step_snap = kv_step_snap,
            .serial = serial,
            .stochastic = stochastic,
            .tracing = tracing,
            .ph = ph,
            .watch = watch,
            .draft_ids = draft_ids,
            .sel_path = null,
            .draft_q = .{ .ctx = null },
        });
        pol.copy.observe(suffix, m, res.accepted_tokens);
        log.debug("  [dflash-copy] suffix={d} avail={d} drafts={d} accepted={d}\n", .{ suffix, avail, m, res.accepted_tokens });
        self.dflash_copies += 1;
        self.dflash_copy_drafted += m;
        self.dflash_copy_accepted += res.accepted_tokens;
        return res;
    }

    /// Parked captures per target layer before a plain stretch appends them anyway (a stretch the
    /// policy's backoff allows is at most 32 rounds; this bounds the parked arrays whatever the caller does).
    const DFLASH_PENDING_MAX: u32 = 32;

    /// A plain DFlash round: t1 through the trunk alone, the verify forward at one row. Its captures
    /// are parked in `dflash_pending` and join the assistant context with the next drafted round's append.
    fn dflashPlainRound(self: *Generator, allocator: std.mem.Allocator, t1: u32, serial: bool, stochastic: bool) !DrafterStepResult {
        const xfm = self.xfm;
        const s = xfm.s;
        const model = self.dflash.?;
        self.dflash_round_width = 0;

        const t1_i32: i32 = @intCast(t1);
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &[_]c_int{ 1, 1 }, 2, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        const caps = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
        defer {
            for (caps) |a| _ = mlx.mlx_array_free(a);
            allocator.free(caps);
        }
        for (caps) |*a| a.* = mlx.mlx_array_new();
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = caps };
        self.ctx.capture_layers = &cl;
        const forward = xfm.forwardWith(&self.ctx, t1_arr);
        self.ctx.capture_layers = null;
        const logits = try forward;
        defer _ = mlx.mlx_array_free(logits);

        // The captures are no dependency of the logits: evaluate them with the next token, or the
        // forward's streams stay alive behind them.
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (caps) |c| _ = mlx.mlx_vector_array_append_value(eval_vec, c);
        var picked: CommittedArgmax = .{};
        defer picked.deinit();
        const next_pending: u32 = blk: {
            if (stochastic) {
                const probs = try probsAtLastPos(logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                try mlx.check(mlx.mlx_async_eval(eval_vec));
                break :blk try sampleFromProbs(probs, self.correctionDraw(0), s);
            }
            picked = if (serial) try self.verifySerialSamples(logits, 1) else try verifyArgmax(logits, self.sampling.suppress_mask, s);
            _ = mlx.mlx_vector_array_append_value(eval_vec, picked.lazy());
            try mlx.check(mlx.mlx_async_eval(eval_vec));
            try mlx.check(mlx.mlx_array_eval(picked.lazy()));
            break :blk @intCast((try picked.ids(1))[0]);
        };
        if (serial) self.sampling.draw = self.generated_ids.items.len + 2;

        try self.dflashPark(allocator, caps);

        const tokens = try allocator.alloc(u32, 1);
        errdefer allocator.free(tokens);
        tokens[0] = t1;
        try self.generated_ids.append(allocator, t1);
        self.dflash_plain += 1;
        self.dflash_stats.record(0, 0);
        self.next_token_id = next_pending;
        self.advanceStep(1);
        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
        }
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    /// Park one forward's captures (and take them over: the caller's handles are left empty).
    fn dflashPark(self: *Generator, allocator: std.mem.Allocator, caps: []mlx.mlx_array) !void {
        try self.dflash_pending.appendSlice(allocator, caps);
        for (caps) |*a| a.* = .{ .ctx = null };
        self.dflash_pending_rounds += 1;
        if (self.dflash_pending_rounds >= DFLASH_PENDING_MAX) try self.dflashFlushPending(allocator);
    }

    /// Whether a plain tick can ride the serial pipeline (`next`'s fast path).
    fn dflashCanPipeline(self: *const Generator) bool {
        return self.logprobs_n == 0 and !self.sampling.penalized() and self.sampling.constraint == null and self.step + 1 < self.max_tokens;
    }

    /// A plain tick on the serial pipeline: `next` overlaps the following forward with the host's work,
    /// which a round of its own cannot. Its forwards carry the DFlash taps, parked like a plain round's.
    fn dflashPlainPipelined(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        const model = self.dflash.?;
        const caps = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
        defer {
            for (caps) |a| _ = mlx.mlx_array_free(a);
            allocator.free(caps);
        }
        for (caps) |*a| a.* = mlx.mlx_array_new();
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = caps };
        self.ctx.capture_layers = &cl;
        defer self.ctx.capture_layers = null;
        // The stretch starts from the DFlash invariant (t1 not in the cache): forward it as `next`'s own bootstrap
        // would, here with the taps, so `next` finds the pipeline seeded and runs only its lookahead.
        if (!self.has_pending_logits and !self.has_pending_token) {
            const t1_i32: i32 = @intCast(self.next_token_id);
            const t1_arr = mlx.mlx_array_new_data(&t1_i32, &[_]c_int{ 1, 1 }, 2, .int32);
            defer _ = mlx.mlx_array_free(t1_arr);
            self.pending_logits = try self.xfm.forwardWith(&self.ctx, t1_arr);
            self.has_pending_logits = true;
            try self.dflashEvalPark(allocator, caps);
        }
        const tok = (try self.next(allocator)) orelse return null;
        // The lookahead's forward (none on the budget's last token) is already queued; its taps are small work behind it.
        if (caps[0].ctx != null) try self.dflashEvalPark(allocator, caps);
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok;
        self.dflash_plain += 1;
        self.dflash_stats.record(0, 0);
        // Consecutive pipelined ticks are the machine's plain period; the first of a stretch has no predecessor.
        const pol = &self.dflash_pol.?;
        if (pol.plain_watch) |*w| pol.plain.sample(@as(f32, @floatFromInt(w.read())) / @as(f32, std.time.ns_per_ms));
        pol.plain_watch = io_util.Stopwatch.init(self.timer.io);
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    /// Evaluate a forward's taps (they are no dependency of its logits) and park them.
    fn dflashEvalPark(self: *Generator, allocator: std.mem.Allocator, caps: []mlx.mlx_array) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (caps) |c| _ = mlx.mlx_vector_array_append_value(vec, c);
        try mlx.check(mlx.mlx_async_eval(vec));
        try self.dflashPark(allocator, caps);
        for (caps) |*a| a.* = mlx.mlx_array_new();
    }

    /// A drafted round after a pipelined stretch: land on the spec entry invariant, one token of the stretch emitted.
    fn dflashLeavePipeline(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        switch (try self.drainPipelineForSpec(allocator)) {
            .drained => |tok| {
                const tokens = try allocator.alloc(u32, 1);
                tokens[0] = tok;
                self.dflash_unobserved = true;
                return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
            },
            .stopped => return null,
            .already_clean, .stay_disabled => return error.PipelineHalfState,
        }
    }

    /// Append the plain rounds' parked captures to the assistant context as one block.
    fn dflashFlushPending(self: *Generator, allocator: std.mem.Allocator) !void {
        const rounds = self.dflash_pending_rounds;
        if (rounds == 0) return;
        const model = self.dflash.?;
        const dctx = &self.dflash_ctx.?;
        const n_t = model.config.target_layer_ids.len;
        const merged = try allocator.alloc(mlx.mlx_array, n_t);
        defer {
            for (merged) |a| _ = mlx.mlx_array_free(a);
            allocator.free(merged);
        }
        for (merged) |*a| a.* = mlx.mlx_array_new();
        for (merged, 0..) |*out, t| {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (0..rounds) |r| _ = mlx.mlx_vector_array_append_value(vec, self.dflash_pending.items[r * n_t + t]);
            try mlx.check(mlx.mlx_concatenate_axis(out, vec, 1, self.xfm.s));
        }
        try dflash_mod.appendContext(model, dctx, merged, dctx.absLen());
        for (self.dflash_pending.items) |a| _ = mlx.mlx_array_free(a);
        self.dflash_pending.clearRetainingCapacity();
        self.dflash_pending_rounds = 0;
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        dctx.appendEvalArrays(eval_vec);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }

    /// The assistant context's length once the parked captures are counted: the trunk's cursor, always.
    pub fn dflashContextLen(self: *const Generator) usize {
        return self.dflash_ctx.?.absLen() + self.dflash_pending_rounds;
    }

    /// Bring the assistant context to the position `covers` (the committed prefix): parked captures appended, and a
    /// pipelined lookahead's position dropped. A prefix-cache commit pairs the two only when they end together.
    pub fn dflashSettle(self: *Generator, allocator: std.mem.Allocator, covers: usize) !void {
        const dctx = if (self.dflash_ctx) |*d| d else return;
        try self.dflashFlushPending(allocator);
        if (dctx.absLen() > covers and covers >= dctx.base_pos) try dctx.cache.truncate(covers - dctx.base_pos, self.xfm.s);
    }

    /// Tell the drafting backoff what a drafted round bought: the plain rounds its tokens would have cost, and what it took.
    fn dflashRecordDrafted(self: *Generator, tokens: usize, wall_ns: u64) void {
        const pol = if (self.dflash_pol) |*p| p else return;
        const wall_ms = @as(f32, @floatFromInt(wall_ns)) / @as(f32, std.time.ns_per_ms);
        pol.drafting.record(@as(f32, @floatFromInt(tokens)) * self.dflashPlainMs(), wall_ms);
    }

    /// ms of a plain round for this request (policy rounds only).
    fn dflashPlainMs(self: *const Generator) f32 {
        return self.dflash_pol.?.plain.ms;
    }

    /// Rows to verify (t1 included) for drafts with acceptance probabilities `p`: the pin, else the best tokens per ms.
    fn dflashRowsFor(self: *const Generator, p: []const f32, costs: *const dflash_policy.Costs) u32 {
        const pol = &self.dflash_pol.?;
        if (pol.pin_rows > 0) return @min(pol.pin_rows, @as(u32, @intCast(p.len)) + 1);
        return dflash_policy.chooseRows(p, costs).rows;
    }

    /// Rows to verify (t1 included) for a draft whose per-position confidence the selector reports.
    fn dflashPolicyRows(self: *Generator, sel: *const dflash_mod.SelectedPath, m_draft: u32) u32 {
        const pol = &self.dflash_pol.?;
        var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
        for (sel.conf[0..m_draft], p[0..m_draft]) |c, *out| out.* = pol.calib.probability(c);
        const costs = dflash_policy.priorCosts(self.dflashPlainMs());
        return self.dflashRowsFor(p[0..m_draft], &costs);
    }

    /// `[dflash-round]`: what the selector was sure of against what the target kept (the calibration's raw material).
    fn logDflashRound(self: *const Generator, conf: []const f32, rows_drafts: u32, accepted: u32) void {
        _ = self;
        var buf: [160]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        for (conf, 0..) |c, k| w.print("{s}{d:.2}", .{ if (k == 0) "" else "/", c }) catch break;
        log.debug("  [dflash-round] verified={d}/{d} accepted={d} conf={s}\n", .{ rows_drafts, conf.len, accepted, w.buffered() });
    }

    var dflash_tree_logged = false;
    var dflash_copy_logged = false;
    /// A verify forward goes to the GPU every this many layers while the rest
    /// of its graph is built (27B tree rounds: 52.8 -> 50.3 ms).
    const VERIFY_PIPELINE_LAYERS: u32 = 4;
    /// Rows of a draft-tree verify: the pending token and up to 15 nodes.
    const DFLASH_TREE_W = 16;
    /// Context tokens that must back a copied continuation (TensorFold's
    /// "confident" bar: shorter backing lost more rounds than it won at 4k-16k).
    const DFLASH_COPY_MATCH = 24;
    /// Tokens the copy lookup scans back over each round (the copy and the scan are per round).
    const DFLASH_COPY_WINDOW = 65536;

    /// `[m, k]` Gumbel noise each lattice position's candidates get from the
    /// keyed draw of its verify row (generated index G + 1 + position).
    fn treeNoise(self: *const Generator, allocator: std.mem.Allocator, lat: *const dflash_mod.Lattice) ![]f32 {
        const out = try allocator.alloc(f32, lat.m * lat.k);
        const base = self.sampling.position_base + self.generated_ids.items.len + 1;
        for (0..lat.m) |d| for (0..lat.k) |j| {
            out[d * lat.k + j] = keyed_sample.gumbel(self.sampling.seed.?, base + d, @intCast(lat.cands[d * lat.k + j]));
        };
        return out;
    }

    /// One DFlash2 round over a best-first draft TREE instead of the selector's
    /// single path: every node is verified in one forward (each row the serial
    /// step along its ancestors), the target's token walked down the tree from
    /// the root, and the accepted path committed into the KV cache, the gated-delta
    /// state and the assistant context.
    /// Nodes a draft tree takes: on the tensor units a 16-row window costs
    /// about what an 8-row one does, so 15 within the lattice's depth;
    /// elsewhere a node per position.
    fn dflashTreeNodes(m: u32) usize {
        return if (transformer_mod.naxAvailable()) DFLASH_TREE_W - 1 else @min(m, DFLASH_TREE_W - 1);
    }

    fn dflashTreeRound(
        self: *Generator,
        allocator: std.mem.Allocator,
        model: *DflashModel,
        dctx: *dflash_mod.DflashCtx,
        blk_hidden: ?mlx.mlx_array,
        draft_logits: ?mlx.mlx_array,
        t1: u32,
        m: u32,
        anchor_pos: usize,
        kv_step_snap: usize,
        serial: bool,
        copy: []const u32,
    ) !DrafterStepResult {
        const xfm = self.xfm;
        const s = xfm.s;
        const MAX_W = DFLASH_TREE_W;
        const tracing = dflashTraceEnabled();
        var ph = io_util.Stopwatch.init(self.timer.io);
        const max_nodes = dflashTreeNodes(m);
        if (!dflash_tree_logged) {
            dflash_tree_logged = true;
            log.info("[dflash] draft trees engaged: up to {d} nodes a round\n", .{max_nodes});
        }

        // Rows: 0 = t1 (the root), then the lattice's best-first tree, or the
        // copied continuation as a chain.
        var w: usize = 1;
        var parents: [MAX_W]i32 = undefined;
        var depth: [MAX_W]i32 = undefined;
        var toks: [MAX_W]i32 = undefined;
        parents[0] = -1;
        depth[0] = 0;
        toks[0] = @intCast(t1);
        if (copy.len == 0) {
            var lat = try dflash_mod.lattice(allocator, &model.selector.?, model.config.selector_top_k, blk_hidden.?, draft_logits.?, t1, s);
            defer lat.deinit(allocator);
            // A sampled target's scores carry its own noise at each position's candidates.
            const noise: ?[]f32 = if (serial) try self.treeNoise(allocator, &lat) else null;
            defer if (noise) |nz| allocator.free(nz);
            var tree = try dflash_mod.bestFirstTree(allocator, &lat, .{
                .max_nodes = max_nodes,
                .temperature = if (serial) self.sampling.temperature else 1.0,
                .noise = noise,
            });
            defer tree.deinit(allocator);
            for (tree.tokens, tree.parents, tree.depth) |tok, par, d| {
                parents[w] = if (par < 0) 0 else par + 1;
                depth[w] = @intCast(d + 1);
                toks[w] = @intCast(tok);
                w += 1;
            }
        }
        if (copy.len > 0 and !dflash_copy_logged) {
            dflash_copy_logged = true;
            log.info("[dflash] context copy verified as the round's chain ({d} rows)\n", .{copy.len + 1});
        }
        for (copy, 1..) |tok, d| {
            parents[w] = @intCast(w - 1);
            depth[w] = @intCast(d);
            toks[w] = @intCast(tok);
            w += 1;
        }
        var max_depth: i32 = 0;
        for (depth[0..w]) |d| max_depth = @max(max_depth, d);
        if (tracing) {
            self.dflash_trace.add(.head, ph.read());
            ph.reset();
        }
        const spec_tree = Generator.specTreeFor(parents[0..w], depth[0..w], max_depth);
        defer spec_tree.deinit();
        const wc: c_int = @intCast(w);
        const verify_input = mlx.mlx_array_new_data(&toks, &[_]c_int{ 1, wc }, 2, .int32);
        defer _ = mlx.mlx_array_free(verify_input);

        const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
        defer {
            for (cap_out) |a| _ = mlx.mlx_array_free(a);
            allocator.free(cap_out);
        }
        for (cap_out) |*a| a.* = mlx.mlx_array_new();
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = cap_out };
        self.ctx.capture_layers = &cl;
        self.ctx.tree = &spec_tree;
        self.ctx.pipeline_build = VERIFY_PIPELINE_LAYERS;
        const verify_logits = xfm.forwardWith(&self.ctx, verify_input);
        self.ctx.capture_layers = null;
        self.ctx.tree = null;
        self.ctx.pipeline_build = 0;
        const logits = try verify_logits;
        self.dflash_attempted += 1;

        var targets = blk: {
            defer _ = mlx.mlx_array_free(logits);
            break :blk if (serial)
                try self.verifySerialSamplesAt(logits, w, depth[0..w])
            else
                try verifyArgmax(logits, self.sampling.suppress_mask, s);
        };
        defer targets.deinit();
        try mlx.check(mlx.mlx_array_eval(targets.lazy()));
        const ids = try targets.ids(w);
        if (tracing) {
            self.dflash_trace.add(.verify, ph.read());
            ph.reset();
        }

        // Walk the target's tokens down the tree.
        var path_rows: [MAX_W]u32 = undefined;
        path_rows[0] = 0;
        var accepted: u32 = 0;
        walk: while (true) {
            const cur = path_rows[accepted];
            const want: i32 = ids[cur];
            for (1..w) |c| {
                if (parents[c] == @as(i32, @intCast(cur)) and toks[c] == want) {
                    accepted += 1;
                    path_rows[accepted] = @intCast(c);
                    continue :walk;
                }
            }
            break;
        }
        accepted = capAcceptedForTokenBudget(accepted, self.completion_tokens, self.max_tokens);
        const next_pending: u32 = @intCast(ids[path_rows[accepted]]);
        if (serial) self.sampling.draw = self.generated_ids.items.len + accepted + 2;
        // A copy that keeps nothing four rounds running pauses the copies.
        if (copy.len > 0) {
            self.dflash_copy_misses = if (accepted == 0) self.dflash_copy_misses + 1 else 0;
            if (self.dflash_copy_misses >= 4) {
                self.dflash_copy_misses = 0;
                self.dflash_copy_silent = 16;
            }
        }

        // Commit: the accepted path's KV rows move to consecutive positions,
        // the gated-delta state is the last kept node's, the conv window its path's.
        const n_commit: usize = 1 + @as(usize, accepted);
        var moved = false;
        for (1..n_commit) |j| {
            if (path_rows[j] != j) moved = true;
        }
        if (moved) try self.ctx.cache.compactRows(anchor_pos, path_rows[1..n_commit], s);
        try self.ctx.cache.truncate(anchor_pos + n_commit, s);
        self.ctx.cache.step = kv_step_snap;
        self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
        if (accepted + 1 < w) self.partial_rounds += 1;
        if (tracing) {
            self.dflash_trace.add(.accept, ph.read());
            ph.reset();
        }

        // The assistant context grows by the kept rows' captures.
        {
            var idx_buf: [MAX_W]i32 = undefined;
            for (0..n_commit) |j| idx_buf[j] = @intCast(path_rows[j]);
            const idx = mlx.mlx_array_new_data(&idx_buf, &[_]c_int{@intCast(n_commit)}, 1, .int32);
            defer _ = mlx.mlx_array_free(idx);
            const kept = try allocator.alloc(mlx.mlx_array, cap_out.len);
            var kept_n: usize = 0;
            defer {
                for (kept[0..kept_n]) |a| _ = mlx.mlx_array_free(a);
                allocator.free(kept);
            }
            for (cap_out, kept) |full, *out| {
                out.* = mlx.mlx_array_new();
                kept_n += 1;
                try mlx.check(mlx.mlx_take_axis(out, full, idx, 1, s));
            }
            try dflash_mod.appendContext(model, dctx, kept, anchor_pos);
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            dctx.appendEvalArrays(eval_vec);
            if (tracing) {
                try mlx.check(mlx.mlx_eval(eval_vec));
                self.dflash_trace.add(.append, ph.read());
            } else {
                try mlx.check(mlx.mlx_async_eval(eval_vec));
            }
        }

        const tokens = try allocator.alloc(u32, n_commit);
        for (0..n_commit) |j| tokens[j] = @intCast(toks[path_rows[j]]);
        for (tokens) |t| try self.generated_ids.append(allocator, t);
        self.dflash_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(@intCast(n_commit));
        self.checkDflashRuntimeGate();
        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
        }
        if (tracing) {
            self.dflashTraceRoundEnd(accepted);
            self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = accepted };
    }

    /// Decode plain in the batched group. Sticky for DFlash (plain rounds do not extend the
    /// assistant context); PLD's periodic re-enable check resumes once the slot is solo.
    pub fn yieldSpecToCompany(self: *Generator, mode: []const u8) void {
        if (self.spec_disabled_runtime) return;
        log.info("  {s}=disabled (company: decoding plain in the batched group)\n", .{mode});
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .company;
    }

    /// DFlash runtime economics gate. Sticky within the request: once a
    /// `DFLASH_GATE_WARMUP`-round sample proves the block-parallel path yields
    /// less than its width-normalized request-class threshold, subsequent
    /// ticks use the regular pipelined decoder through `nextDflash`'s entry
    /// fallback.
    fn checkDflashRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime or self.dflash_pol != null) return;
        if (!dflashGateShouldDisable(
            self.dflash_attempted,
            self.dflash_accepted_tokens,
            self.dflash_min_accepted_per_round,
        )) return;
        // The table's own measure outranks the calibrated bar: a round emitting
        // tokens cheaper than the plain step it would yield to stays on.
        if (self.xfm.round_cost.roundBeatsSerial(self.dflash_block_size -| 1, self.specKvLen()) == true) return;
        const avg = @as(f32, @floatFromInt(self.dflash_accepted_tokens)) /
            @as(f32, @floatFromInt(self.dflash_attempted));
        log.info(
            "  dflash=disabled (runtime yield {d:.2} accepted/round < {d:.2} after {d} attempts)\n",
            .{ avg, self.dflash_min_accepted_per_round, self.dflash_attempted },
        );
        self.spec_disabled_runtime = true;
    }

    /// Runtime acceptance gate for the drafter: after warmup, if the per-draft
    /// acceptance probability is below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`,
    /// disable speculation for the rest of this request. Sticky for the rest
    /// of the generation.
    fn checkDrafterRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime) return;
        const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
        if (!runtimeGateShouldDisable(self.drafter_attempted, self.drafter_accepted_tokens, drafts_per_round)) return;
        const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
        const rate: f32 = if (drafts_proposed > 0)
            @as(f32, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f32, @floatFromInt(drafts_proposed))
        else
            0.0;
        log.info(
            "  drafter=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
            .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.drafter_attempted },
        );
        self.spec_disabled_runtime = true;
    }

    /// The serial sampler's draw for every verify row: row i is the token at
    /// generated index G + 1 + i (t1 sits at G), keyed by that index.
    fn verifySerialSamples(self: *const Generator, logits: mlx.mlx_array, rows: usize) !CommittedArgmax {
        return self.verifySerialSamplesAt(logits, rows, null);
    }

    /// `verifySerialSamples` for a tree's rows: row i is the token at generated
    /// index G + 1 + depth[i].
    fn verifySerialSamplesAt(self: *const Generator, logits: mlx.mlx_array, rows: usize, depth: ?[]const i32) !CommittedArgmax {
        const s = self.xfm.s;
        const vocab = mlx.getShape(logits)[2];
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        const base = self.generated_ids.items.len + 1;
        for (0..rows) |i| {
            const ri: c_int = @intCast(i);
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, logits, &[_]c_int{ 0, ri, 0 }, 3, &[_]c_int{ 1, ri + 1, vocab }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var sp = self.sampling;
            sp.draw = base + (if (depth) |d| @as(usize, @intCast(d[i])) else i);
            const tok = sampleTokenLazy(row, sp, s);
            defer _ = mlx.mlx_array_free(tok);
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, tok, &[_]c_int{1}, 1, s));
            _ = mlx.mlx_vector_array_append_value(vec, flat);
        }
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, cat, &[_]c_int{ 1, @intCast(rows) }, 2, s));
        return .{ .arr = out };
    }

    /// The committed-token index, synced to `generated_ids`.
    fn ngramIndex(self: *Generator, allocator: std.mem.Allocator) !*ngram_lookup.Index {
        if (self.ngram_idx == null) {
            var idx = ngram_lookup.Index.init(allocator);
            errdefer idx.deinit();
            try idx.extend(self.prompt_ids_owned);
            self.ngram_idx = idx;
            self.ngram_gen = 0;
        }
        const gen = self.generated_ids.items;
        try self.ngram_idx.?.extend(gen[self.ngram_gen..]);
        self.ngram_gen = gen.len;
        return &self.ngram_idx.?;
    }

    /// The spec round's clock: the interval since the previous round end held a
    /// prefill chunk. The next round times itself and is dropped as a transition.
    pub fn invalidateRoundClock(self: *Generator) void {
        self.spec_round_clock = null;
        self.spec_round_prev_width = null;
    }

    /// Full Leviathan acceptance ratio; q clamped so a sampled draft (q > 0
    /// by construction) can never divide by an underflowed zero.
    pub fn specAcceptProb(p: f32, q_draft: f32) f32 {
        return @min(1.0, p / @max(q_draft, 1e-12));
    }

    // Speculative round-cost model: the width chooser prices a round from per-index acceptance
    // EMAs and the measured round-cost table.

    /// EMA step for the per-index acceptance estimates. 0.15 demotes fast on
    /// cold streaks (5 consecutive rejects: 0.72 -> 0.32) without letting a
    /// single unlucky round move the plan.
    pub const SPEC_EV_EMA_BETA: f32 = 0.15;
    /// Clamp band for the extension confidence threshold.
    pub const SPEC_EV_TAU_MIN: f32 = 0.05;
    pub const SPEC_EV_TAU_MAX: f32 = 0.95;
    /// Minimum base-round EV (tokens per round-cost) for the exploration
    /// horizon: below this the base itself barely beats AR, so keeping an
    /// extension position reachable would only add sync tax to a dying
    /// speculation (the sticky-disable floor is likely next anyway). 1.10
    /// keeps fully-cold EMAs collapsed on every cost profile (cold best_r:
    /// generic 1.00, G17-NAX 1.07/1.08).
    pub const SPEC_EV_EXPLORE_MIN_R: f32 = 1.10;

    /// Round-cost model in units of the fixed round cost (verify-forward floor + round
    /// eval/read + commit = 1.0). Ratios are machine-stable where absolute ms are not.
    /// Override for live tuning: MLX_SERVE_MTP_EV_COSTS="draft,per_pos_lo,per_pos_hi,sync".
    pub const SpecEvCosts = struct {
        draft: f32, // one sequential draft step
        per_pos_lo: f32, // marginal verify+capture per position, flat region
        per_pos_hi: f32, // ... beyond flat_max (qmm row-tile ramp)
        flat_max: u32, // last draft index in the flat verify region
        sync: f32, // the chunk-A confidence read-back
        /// First draft depth whose verify forward lands on the M5 NAX tile.
        /// Zero disables the third region. Depth k verifies k+1 rows, so the
        /// default NAX M=8 takeover starts at draft position 7.
        nax_from: u32 = 0,
        per_pos_nax: f32 = 0.0,
        /// Serial-floor cost in ms at kv ~= 0 (the probe's width-1 rung).
        /// Zero - every hand-typed table - makes the kv term below a no-op,
        /// so the tables keep behaving exactly as they did.
        floor_ms: f32 = 0,
        /// Context length the marginals above were FITTED AT. The kv term
        /// scales relative to THIS, not to zero: refit #4 was measured at 8K,
        /// so its 0.20 already contains 8K of KV read and re-scaling it from
        /// a kv~=0 floor would discount it twice. Zero = unknown, which
        /// disables the kv term for that surface rather than guessing.
        kv_ref_tokens: u32 = 0,
        /// Learned per-KV-token round cost B (ms/token), the term that makes
        /// wide speculation approach FREE at long context.
        ///
        /// A verify forward of width k reads the weights once and the KV
        /// once, both SHARED across all k query rows; only arithmetic scales
        /// with k. So T(k, L) ~= W + B*L + C(k) and T(k,L)/T(1,L) -> 1 as L
        /// grows: the optimal width RISES with context, and every constant
        /// this controller ever shipped was fitted at ONE context length.
        /// Learned online rather than probed - a boot probe measures at
        /// kv ~= 0, where B is invisible, and probing it would mean
        /// allocating a 32k KV cache purely to time it.
        kv_ms_per_token: f32 = 0,
    };
    /// The default surface: flat to 4 positions, a ramp to 6, the NAX tile from 7.
    pub const SPEC_EV_DEFAULT_COSTS: SpecEvCosts = .{ .draft = 0.10, .per_pos_lo = 0.10, .per_pos_hi = 0.26, .flat_max = 4, .sync = 0.01, .nax_from = 7, .per_pos_nax = 0.52 };

    fn dflashAcceptChoose(self: *Generator, ch: *round_cost.WidthChooser) round_cost.WidthChooser.Decision {
        const t = &self.xfm.round_cost;
        const src = SpecCostSource.init(self.spec_ev_costs, self.specKvLen(), t);
        const a = ch.accept[0..ch.max_width];
        ch.current = dflashAcceptWidth(a, ch.max_width, src, ch.current);
        const up = ch.current + 1;
        if (ch.current == 0 or up > ch.max_width or !src.fromTable()) return .{ .width = ch.current, .trial = false };
        // The probe is what refreshes a[current]; a width nobody has timed is learned at once.
        if (t.rawMs(up, src.bucket) == null) ch.trial.startAt(ch.rounds);
        const probe = ch.trial.force(ch.rounds, specProbePeriod(a, src, ch.current), round_cost.schedulePeriodReread(t.layout));
        return .{ .width = if (probe) up else ch.current, .trial = probe };
    }

    /// Feed the model's round-cost table (`Transformer.round_cost`, shared
    /// by every request and both block decoders). Width = drafts this round
    /// (0 = serial). The previous width is tracked on EVERY round so the
    /// first observed one is not a transition by accident; a rejected
    /// sample that is not an expected drop (contended, transition) logs its
    /// numbers - a silent reject is the probe's anti-pattern.
    fn specObserveRound(self: *Generator, width: u32, wall_ms: f32, tokens: f32, observe: bool, shape_changed: bool) void {
        const transition = shape_changed or
            (if (self.spec_round_prev_width) |p| p != width else true) or
            (if (self.spec_round_prev_width2) |p| p != width else true);
        self.spec_round_prev_width2 = self.spec_round_prev_width;
        self.spec_round_prev_width = width;
        if (!observe) {
            log.debug("  [spec-cost] w{d} ms={d:.2} tok={d:.0} -> unobserved\n", .{ width, wall_ms, tokens });
            return;
        }
        const kv = self.specKvLen();
        const v = self.xfm.round_cost.observe(width, kv, wall_ms, tokens, self.spec_cost_solo, transition);
        log.debug("  [spec-cost] w{d} kv={d} ms={d:.2} tok={d:.0} -> {s}\n", .{ width, kv, wall_ms, tokens, @tagName(v) });
        if (v == .bad_sample or v == .out_of_range) {
            log.warn("[spec-cost] table rejected sample ({s}): width={d} kv={d} ms={d:.2} tokens={d:.1}\n", .{ @tagName(v), width, kv, wall_ms, tokens });
        }
    }

    /// Write the model's round-cost table to disk when this request folded
    /// new samples (request end; inference thread, so no lock).
    pub fn persistRoundCost(self: *Generator) void {
        const t = &self.xfm.round_cost;
        const folded = round_cost.totalFolded(t);
        if (folded == t.stored_at) return;
        round_cost.storeCached(self.timer.io, self.xfm.round_cost_key_buf[0..self.xfm.round_cost_key_len], t);
        t.stored_at = folded;
    }

    /// KV length the next verify forward reads.
    fn specKvLen(self: *const Generator) u32 {
        return @intCast(self.prompt_ids_owned.len + self.generated_ids.items.len);
    }

    /// Marginal round cost of draft position k (1-based), kv-blind. This is
    /// the fitted SHAPE; `specEvMarginalCostAt` applies the kv term.
    pub fn specEvMarginalCost(costs: SpecEvCosts, k: u32) f32 {
        const verify_cost = if (costs.nax_from != 0 and k >= costs.nax_from)
            costs.per_pos_nax
        else if (k <= costs.flat_max)
            costs.per_pos_lo
        else
            costs.per_pos_hi;
        return costs.draft + verify_cost;
    }

    /// One round's draft plan. `m_hi > m_lo` means "pay the chunk-A sync and
    /// extend to m_hi when the chain log-confidence clears tau_ln".
    pub const SpecWidthPlan = struct {
        m_lo: u32,
        m_hi: u32,
        tau_ln: f32,
        /// A width-trial round (single-chunk at m_lo+1, measuring the round
        /// cost table): skipped by the regime gate, which compares shapes
        /// at ONE base depth.
        width_trial: bool = false,
    };

    /// Expected committed tokens for an m-deep round: the always-committed t1
    /// plus the acceptance chain sum (draft k lands iff drafts 0..k all land).
    pub fn specEvExpectedTokens(a: []const f32, m: u32) f32 {
        var chain: f32 = 1.0;
        var tok: f32 = 1.0;
        var k: u32 = 0;
        while (k < m and k < a.len) : (k += 1) {
            chain *= a[k];
            tok += chain;
        }
        return tok;
    }

    pub fn specEvMarginalCostAt(costs: SpecEvCosts, k: u32, kv_len: u32) f32 {
        _ = kv_len;
        return specEvMarginalCost(costs, k);
    }

    /// Round cost in verify-base units (piecewise per-position marginals).
    pub fn specEvRoundCost(costs: SpecEvCosts, m: u32, with_sync: bool) f32 {
        return specEvRoundCostAt(costs, m, with_sync, 0);
    }

    pub fn specEvRoundCostAt(costs: SpecEvCosts, m: u32, with_sync: bool, kv_len: u32) f32 {
        var c: f32 = 1.0 + (if (with_sync) costs.sync else 0.0);
        var k: u32 = 1;
        while (k <= m) : (k += 1) c += specEvMarginalCostAt(costs, k, kv_len);
        return c;
    }

    /// Smallest marginal the table may report, in floor units: a measured
    /// pair that reads a wider round CHEAPER is noise, and a non-positive
    /// marginal would make every extension free.
    pub const SPEC_EV_TABLE_MIN_MARGINAL: f32 = 0.02;

    /// Where the EV plan's round costs come from: the measured round-cost
    /// table when the kv bucket (or its nearest active neighbour) has
    /// MIN_WIDTHS measured widths, else the fitted surface as the cold-start
    /// prior. The table is in ms; the plan compares ratios but has one
    /// absolute threshold (`SPEC_EV_EXPLORE_MIN_R`), so the table is scaled
    /// into FLOOR UNITS at its narrowest measured width - the two sources
    /// agree there by construction and the table's slopes take over.
    pub const SpecCostSource = struct {
        costs: SpecEvCosts,
        kv_len: u32 = 0,
        table: ?*const round_cost.Table = null,
        bucket: usize = 0,
        scale: f32 = 0,
        /// False = the table prices round TIME only and tokens come from the acceptance EMAs.
        table_tokens: bool = true,

        /// The widest measured width the table may anchor the cost scale on.
        const MAX_REF_WIDTH: u32 = 8;

        pub fn init(costs: SpecEvCosts, kv_len: u32, table: ?*const round_cost.Table) SpecCostSource {
            var src = SpecCostSource{ .costs = costs, .kv_len = kv_len, .table = table };
            const t = table orelse return src;
            const b = t.bucketToRead(kv_len) orelse return src;
            const ref = t.narrowestMeasured(b) orelse return src;
            const ref_ms = t.measuredMs(ref, b) orelse return src;
            if (!(ref_ms > 0) or ref > MAX_REF_WIDTH) return src;
            src.bucket = b;
            src.scale = specEvRoundCostAt(costs, ref, false, kv_len) / ref_ms;
            return src;
        }

        pub fn fromTable(self: SpecCostSource) bool {
            return self.scale > 0;
        }

        /// Realized tokens per single-chunk round at a MEASURED width, else
        /// null (the acceptance-EMA model fills in). The model's E(6) on the
        /// M4 Max @16k said 6.85 while the measured single-6 round emitted
        /// 6.0 - the 6th draft's rejections cost a rollback the model cannot
        /// see, and the table can.
        pub fn measuredTokens(self: SpecCostSource, m: u32) ?f32 {
            if (self.scale <= 0 or !self.table_tokens) return null;
            return self.table.?.measuredTok(m, self.bucket);
        }

        pub fn roundCost(self: SpecCostSource, m: u32, with_sync: bool) f32 {
            if (self.scale > 0) {
                const sync: f32 = if (with_sync) self.costs.sync else 0.0;
                const t = self.table.?;
                if (t.roundMs(m, self.bucket)) |ms| return ms * self.scale + sync;
                // Past the widest measured width every extra position costs
                // max(last measured slope, prior marginal): continuous with
                // the table, never more optimistic than the prior (a shallow
                // measured slope run upward let the plan race to 8 unmeasured).
                if (t.widestMeasured(self.bucket)) |w| {
                    if (m > w) {
                        const slope = (t.lastSlope(self.bucket) orelse 0.0) * self.scale;
                        var c = t.measuredMs(w, self.bucket).? * self.scale;
                        var k: u32 = w + 1;
                        while (k <= m) : (k += 1) {
                            c += @max(slope, specEvMarginalCostAt(self.costs, k, self.kv_len));
                            // One sample is evidence for WORSE, never for
                            // cheaper: an untrusted cell floors the cost. Round TIME
                            // alone is stable to a few percent, so where no token
                            // column is read the sample prices its width.
                            if (t.rawMs(k, self.bucket)) |raw| c = if (self.table_tokens) @max(c, raw * self.scale) else raw * self.scale;
                        }
                        return c + sync;
                    }
                }
            }
            return specEvRoundCostAt(self.costs, m, with_sync, self.kv_len);
        }

        pub fn marginal(self: SpecCostSource, k: u32) f32 {
            if (self.scale > 0 and k >= 1) {
                return @max(self.roundCost(k, false) - self.roundCost(k - 1, false), SPEC_EV_TABLE_MIN_MARGINAL);
            }
            return specEvMarginalCostAt(self.costs, k, self.kv_len);
        }
    };

    /// How the base depth is chosen. `.accept`: expected tokens come from the acceptance EMAs
    /// (every round refreshes every index it drafted) and the table prices round TIME only, in
    /// ONE chunk, so no round reads confidences mid-round. `.legacy`: the table's realized
    /// tokens where measured, plus the confidence-gated chunk B (MLX_SERVE_MTP_DEPTH_POLICY=legacy).
    pub const SpecDepthPolicy = enum { accept, legacy };

    pub fn specDepthPolicyFromEnv(raw: ?[]const u8) SpecDepthPolicy {
        return if (std.mem.eql(u8, raw orelse "", "legacy")) .legacy else .accept;
    }

    var spec_depth_policy_cache: ?SpecDepthPolicy = null;
    pub fn specDepthPolicy() SpecDepthPolicy {
        if (spec_depth_policy_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_DEPTH_POLICY")) |p| std.mem.span(p) else null;
        const v = specDepthPolicyFromEnv(raw);
        spec_depth_policy_cache = v;
        return v;
    }

    pub fn specBasePlan(policy: SpecDepthPolicy, a: []const f32, cap: u32, src_in: SpecCostSource, m_lo_max: u32) SpecWidthPlan {
        var src = src_in;
        src.table_tokens = policy == .legacy;
        var plan = specEvPlanSrc(a, cap, src, m_lo_max);
        if (policy == .accept) {
            plan.m_hi = plan.m_lo;
            plan.tau_ln = 0.0;
        }
        return plan;
    }

    /// A block drafter's width under `.accept`: the same argmax over ITS acceptance chain and the
    /// table's round times. It can fall any number of widths in one decision and climbs one per
    /// round; serial (0) wins only where a serial round was measured.
    pub fn dflashAcceptWidth(a: []const f32, max_width: u32, src: SpecCostSource, current: u32) u32 {
        const width = specBasePlan(.accept, a, max_width, src, @max(current, 1) + 1).m_lo;
        if (!src.fromTable()) return width;
        const serial_ms = src.table.?.measuredMs(0, src.bucket) orelse return width;
        const rate = specEvExpectedTokens(a, width) * src.scale / src.roundCost(width, false);
        return if (1.0 / serial_ms > rate * (1.0 + round_cost.SWITCH_MARGIN)) 0 else width;
    }

    pub fn specEvPlanSrc(a: []const f32, cap_in: u32, src: SpecCostSource, m_lo_max: u32) SpecWidthPlan {
        const cap: u32 = @intCast(@min(@as(usize, @max(1, cap_in)), a.len));
        const lo_cap: u32 = @min(cap, @max(1, m_lo_max));
        var m_lo: u32 = 1;
        var best_r: f32 = 0.0;
        // With measured costs the standing base (m_lo_max - 1, what the
        // caller planned last round) keeps its place unless a challenger
        // beats it by the switch margin: cells are EMAs of a few samples
        // and a 5% reversal between neighbours is noise, while every
        // change of base is a transition round the table will not count
        // and the model pays for.
        const standing: u32 = if (src.fromTable()) @min(lo_cap, m_lo_max -| 1) else 0;
        var standing_r: f32 = 0.0;
        var m: u32 = 1;
        while (m <= lo_cap) : (m += 1) {
            const tok = src.measuredTokens(m) orelse specEvExpectedTokens(a, m);
            const r = tok / src.roundCost(m, false);
            if (m == standing) standing_r = r;
            if (r > best_r) {
                best_r = r;
                m_lo = m;
            }
        }
        if (standing >= 1 and m_lo != standing and best_r <= standing_r * (1.0 + round_cost.SWITCH_MARGIN)) {
            m_lo = standing;
            best_r = standing_r;
        }
        if (m_lo >= cap) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        var m_hi: u32 = m_lo;
        var cond: f32 = 1.0;
        var s_sum: f32 = 0.0; // expected extension tokens, conditional on chunk A
        var t_sum: f32 = 0.0; // extension marginal cost (piecewise)
        while (m_hi < cap) {
            cond *= a[m_hi];
            const mc = src.marginal(m_hi + 1);
            if (cond <= best_r * mc) break;
            s_sum += cond;
            t_sum += mc;
            m_hi += 1;
        }
        if (m_hi == m_lo) {
            // Exploration valve for the horizon itself (2026-07-22): the
            // index one past m_lo is observable ONLY when extension fires -
            // at m_lo-deep rounds a[m_lo] never updates - so an EMA dragged
            // cold under an earlier workload (or an easier tau regime)
            // would otherwise close the horizon FOREVER: m stays at m_lo,
            // a[m_lo] never re-proves itself, and pure echo runs at
            // ext_rounds=0 (the exact class the TAU_MAX valve exists for,
            // resurfaced by refit #3's honest marginals). Whenever the base
            // itself pays, keep ONE extension position reachable at the
            // clamped tau; the dry-spell gate bounds the consideration tax
            // on workloads where confidence never clears it.
            if (best_r <= SPEC_EV_EXPLORE_MIN_R) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
            const mc = src.marginal(m_lo + 1);
            // A MEASURED marginal the position cannot repay even at full
            // confidence (1 <= best_r * mc) closes the valve: the valve
            // exists to observe a[m_lo], and with the table that is the
            // width trial's job. The fitted prior keeps the valve exactly
            // as Phase 1 measured it.
            if (src.fromTable() and best_r * mc >= 1.0) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
            const s = @max(a[m_lo], 1e-6);
            const tau_x = std.math.clamp(best_r * mc / s, SPEC_EV_TAU_MIN, SPEC_EV_TAU_MAX);
            return .{ .m_lo = m_lo, .m_hi = m_lo + 1, .tau_ln = @log(tau_x) };
        }
        // The TAU_MAX clamp doubles as the exploration valve: on razor-thin
        // horizons the honest tau approaches 1 ("never extend"), and 0.95
        // lets near-perfect-confidence rounds through so the deep EMAs can
        // observe reality at all.
        const tau = std.math.clamp(best_r * t_sum / s_sum, SPEC_EV_TAU_MIN, SPEC_EV_TAU_MAX);
        return .{ .m_lo = m_lo, .m_hi = m_hi, .tau_ln = @log(tau) };
    }

    /// Per-phase wall-time accumulator behind MLX_SERVE_DFLASH_TRACE=1.
    /// `nextDflash` stamps phases with a Stopwatch and emits one summary line
    /// every LOG_EVERY rounds. Zero cost when the env is absent.
    pub const DflashTrace = struct {
        pub const LOG_EVERY: u32 = 16;
        pub const Phase = enum(u4) { assist, head, verify, accept, append, gap };
        pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = @splat(0),
        accepted: u64 = 0,

        pub fn add(self: *DflashTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@backingInt(phase)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *DflashTrace, accepted_n: u32) bool {
            self.rounds += 1;
            self.accepted += accepted_n;
            return self.rounds >= LOG_EVERY;
        }

        pub fn avgMs(self: *const DflashTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const DflashTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn reset(self: *DflashTrace) void {
            self.* = .{};
        }
    };

    /// Sampled DFlash drafts - DEFAULT OFF, measured negative.
    /// `MLX_SERVE_DFLASH_SAMPLED_DRAFTS=1` draws each draft from the
    /// request's own filtered distribution and accepts through the full
    /// Leviathan ratio; off keeps the argmax draft and its one-hot q.
    ///
    /// The theory said this should help: a greedy draft's acceptance is
    /// p(argmax q), which a flattened target row deflates, while a matched
    /// proposal accepts at 1 − TV(p, q). Measured on a 4-bit target at temp 0.7
    /// (3 reps x 4 prompts, same-boot serial reference), it is a LOSS -
    /// 52.4 tok/s / 54.5% per-draft against 55.5 / 57.7% greedy. This
    /// assistant's ARGMAX tracks the trunk well while its distribution SHAPE
    /// does not, so a one-hot proposal is the better proposal and sampling
    /// spends the round's budget landing on tokens the trunk would not pick.
    /// Greedy requests never reach this either way.
    var dflash_sampled_drafts_cache: ?bool = null;
    fn dflashSampledDraftsEnabled() bool {
        if (dflash_sampled_drafts_cache) |v| return v;
        var on = false;
        if (std.c.getenv("MLX_SERVE_DFLASH_SAMPLED_DRAFTS")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '1') on = true;
        }
        dflash_sampled_drafts_cache = on;
        return on;
    }

    /// `MLX_SERVE_DFLASH_SELECTOR`: default ON when the sidecar ships a
    /// selector (DFlash2). "0" forces the v1 draft arms (argmax / block
    /// sampling) for A/Bs - the conv layers stay, they are the checkpoint.
    var dflash_selector_cache: ?bool = null;
    /// `MLX_SERVE_DFLASH_MARKOV=0` drafts a DSpark sidecar's block from its
    /// UNCORRECTED base logits (the v1 arm) - an A/B lever for measuring what
    /// the Markov chain is worth, never a default.
    fn dflashMarkovEnabled() bool {
        const p = std.c.getenv("MLX_SERVE_DFLASH_MARKOV") orelse return true;
        return !std.mem.eql(u8, std.mem.span(p), "0");
    }

    fn dflashSelectorEnabled() bool {
        if (dflash_selector_cache) |v| return v;
        var on = true;
        if (std.c.getenv("MLX_SERVE_DFLASH_SELECTOR")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '0') on = false;
        }
        dflash_selector_cache = on;
        return on;
    }

    var dflash_trace_cache: ?bool = null;
    fn dflashTraceEnabled() bool {
        if (dflash_trace_cache) |v| return v;
        const on = readEnvBool("MLX_SERVE_DFLASH_TRACE");
        dflash_trace_cache = on;
        return on;
    }

    /// Close one traced dflash round; emits + resets at the cadence.
    fn dflashTraceRoundEnd(self: *Generator, accepted: u32) void {
        if (!self.dflash_trace.endRound(accepted)) return;
        const t = &self.dflash_trace;
        log.info(
            "  [dflash-trace] rounds={d} avg_ms assist={d:.2} head={d:.2} verify={d:.2} accept={d:.2} append={d:.2} gap={d:.2} total={d:.2} | acc_avg={d:.2}\n",
            .{
                t.rounds,
                t.avgMs(.assist),
                t.avgMs(.head),
                t.avgMs(.verify),
                t.avgMs(.accept),
                t.avgMs(.append),
                t.avgMs(.gap),
                t.totalAvgMs(),
                @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
            },
        );
        t.reset();
    }

    fn parseSpecEvCostsOverride(raw: []const u8) ?SpecEvCosts {
        var values: [4]f32 = undefined;
        var it = std.mem.splitScalar(u8, raw, ',');
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            const part = it.next() orelse return null;
            const value = std.fmt.parseFloat(f32, std.mem.trim(u8, part, " ")) catch return null;
            if (!std.math.isFinite(value)) return null;
            values[i] = value;
        }
        if (it.next() != null) return null;
        if (values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or values[3] < 0.0) return null;

        var c = SPEC_EV_DEFAULT_COSTS;
        c.draft = values[0];
        c.per_pos_lo = values[1];
        c.per_pos_hi = values[2];
        c.sync = values[3];
        // The override contract is the two-region surface: never inherit a
        // third region (M5 NAX or the generic k>=7 cliff) the four values
        // can't express.
        c.nax_from = 0;
        c.per_pos_nax = 0.0;
        return c;
    }

    /// The cost surface for an optional four-value override. A valid override starts from the
    /// default surface; an empty, partial or malformed one is ignored atomically.
    pub fn specEvCostsFor(override: ?[]const u8) SpecEvCosts {
        if (override) |raw| return parseSpecEvCostsOverride(raw) orelse SPEC_EV_DEFAULT_COSTS;
        return SPEC_EV_DEFAULT_COSTS;
    }

    fn specEvCosts() SpecEvCosts {
        return specEvCostsFor(if (std.c.getenv("MLX_SERVE_MTP_EV_COSTS")) |p| std.mem.span(p) else null);
    }

    /// Wall time since the previous round ended (the first round reads its
    /// own stopwatch), so inter-round work is charged to the shape.
    fn specRoundWallMs(self: *Generator, round_ms: f32) f32 {
        if (self.spec_round_clock) |*c| {
            const ns = c.read();
            c.reset();
            return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
        }
        self.spec_round_clock = io_util.Stopwatch.init(self.timer.io);
        return round_ms;
    }

    /// Probe period when the plan's tokens come from the acceptance EMAs: the same drag rule,
    /// over the EV rates of m_lo and m_lo+1. The probe is what refreshes `a[m_lo]`, so a stale
    /// estimate shortens its own period as soon as the shallower indices move.
    pub fn specProbePeriod(a: []const f32, src: SpecCostSource, m_lo: u32) u32 {
        const t = src.table orelse return round_cost.EXPLORE_PERIOD_COLD;
        if (!src.fromTable()) return round_cost.EXPLORE_PERIOD_COLD;
        if (t.measuredMs(m_lo + 1, src.bucket) == null) {
            // One sample at a verify-width cliff is enough to stop asking at the cold period.
            if (!t.clearlyWorse(m_lo + 1, m_lo, src.bucket)) return round_cost.EXPLORE_PERIOD_COLD;
            return round_cost.trialPeriod(t.msPerTok(m_lo, src.bucket), t.rawMsPerTok(m_lo + 1, src.bucket));
        }
        const here = src.roundCost(m_lo, false) / specEvExpectedTokens(a, m_lo);
        const up = src.roundCost(m_lo + 1, false) / specEvExpectedTokens(a, m_lo + 1);
        return round_cost.trialPeriod(here, up);
    }

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.done) return null;
        if (self.sampling.constraint != null) return self.nextConstrained(allocator);

        // Transition shim: speculative-decode paths may exit with
        // `next_token_id` set but `pending_logits` unset (drafter's exit
        // invariant is "t1 NOT in cache" — its hand-off to `next()` would
        // otherwise crash on the slow path which assumes pending_logits is
        // always lazily seeded). When we observe that state, synchronously
        // forward `[next_token_id]` to seed `pending_logits` so the fast
        // path picks up cleanly. PLD's exit state already matches `next()`'s
        // invariant, so this only fires for drafter→next runtime-gate
        // fallbacks (and any future spec methods that share drafter's shape).
        if (!self.has_pending_logits and !self.has_pending_token and
            self.step < self.max_tokens and self.logprobs_n == 0)
        {
            const tok_i32: i32 = @intCast(self.next_token_id);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            self.pending_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            self.has_pending_logits = true;
        }

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        // A penalty stays on the slow path: here the pending token is not in `generated_ids` yet.
        if (self.has_pending_logits and self.logprobs_n == 0 and !self.sampling.penalized() and self.step + 1 < self.max_tokens) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const lazy_token = self.sampleLazy(step_logits);
            _ = mlx.mlx_array_free(step_logits);

            var adopted = false;
            defer if (!adopted) {
                _ = mlx.mlx_array_free(lazy_token);
            };
            if (lazyForward(self.xfm, &self.ctx, lazy_token)) |next_logits| {
                defer if (!adopted) {
                    _ = mlx.mlx_array_free(next_logits);
                };
                var sw = StepTrace.watch();
                const arr = [_]mlx.mlx_array{ lazy_token, self.headTailEvalTarget(next_logits) };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                asyncEvalDecodeStep(vec);
                _ = mlx.mlx_vector_array_free(vec);
                StepTrace.lap(&sw, .submit);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                try self.resolvePendingToken();
                StepTrace.lap(&sw, .resolve);
                if (sw != null) StepTrace.steps += 1;

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.advanceStep(1);
                try self.generated_ids.append(allocator, token);

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;
                adopted = true;

                return token;
            } else |_| {
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                adopted = true;
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        try self.resolvePendingToken();

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            break :blk try self.xfm.forwardWith(&self.ctx, tok_input);
        };

        // Logprobs: fully synchronous
        if (self.logprobs_n > 0) {
            defer _ = mlx.mlx_array_free(step_logits);
            const result = try sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.xfm.s);
            self.sampling.draw +%= 1;
            self.next_token_id = result.token_id;
            // `result` belongs to the token we just SAMPLED, which the next
            // call returns; `token` is carrying the previous round's result.
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = self.pending_logprob;
            self.pending_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const lazy_token = self.sampleLazy(step_logits);
        _ = mlx.mlx_array_free(step_logits);

        if (self.step < self.max_tokens) {
            const next_logits = lazyForward(self.xfm, &self.ctx, lazy_token) catch {
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };

            const arr = [_]mlx.mlx_array{ lazy_token, self.headTailEvalTarget(next_logits) };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Sample one token while the reasoning protocol has not reached the JSON
    /// body: drain a forced recovery sequence, work the unresolved opener
    /// choice under the union mask, or sample unconstrained reasoning while
    /// the close-delimiter matcher watches. The active JSON phase reuses the
    /// existing grammar mask builder.
    fn nextReasoningStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
        // Recovery drain: at most one canonical transition token per tick,
        // through the same forward/accounting path as a sampled token.
        if (constraint.pstate.recovering) {
            return try self.commitForcedConstraintToken(allocator, constraint);
        }
        switch (constraint.pstate.phase) {
            .choice, .header => return self.nextChoiceStep(allocator, constraint, proto),
            .reasoning => {},
            .json_body => unreachable,
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        var owns_step_logits = true;
        defer {
            if (owns_step_logits) _ = mlx.mlx_array_free(step_logits);
        }

        // Boundary-crossing candidates (tokens carrying the close delimiter
        // plus a payload suffix) are validated BEFORE they can be sampled: a
        // suffix the schema rejects is masked out here, never sampled and
        // degraded afterwards.
        const invalid_crossers = try rp_mod.applyReasoningMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        const lazy = blk: {
            if (invalid_crossers == 0) break :blk self.sampleLazy(step_logits);
            var masked_logits = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked_logits);
            try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, self.xfm.s);
            break :blk self.sampleLazy(masked_logits);
        };
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id and (proto.closer_atomic == null or token != proto.closer_atomic.?)) {
                _ = mlx.mlx_array_free(step_logits);
                owns_step_logits = false;
                switch (try self.forceConstraintTransition(allocator)) {
                    .committed => |forced| {
                        log.info("[grammar] reasoning boundary forced after premature EOS\n", .{});
                        return forced;
                    },
                    .activated => unreachable, // the reasoning phase always needs its close delimiter
                    .refused => {
                        self.done = true;
                        self.finish_reason = "stop";
                        return null;
                    },
                }
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                _ = mlx.mlx_array_free(step_logits);
                owns_step_logits = false;
                switch (try self.forceConstraintTransition(allocator)) {
                    .committed => |forced| {
                        log.info("[grammar] reasoning boundary forced after terminal padding\n", .{});
                        return forced;
                    },
                    .activated => unreachable,
                    .refused => {
                        self.done = true;
                        self.finish_reason = "stop";
                        return null;
                    },
                }
            }
        } else {
            self.consecutive_pad = 0;
        }

        const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;
        const payload_suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
        if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
        if (payload_suffix) |suffix| {
            // The close delimiter completed inside this token; any bytes after
            // it are the first payload bytes. The pre-sample mask validated
            // every crossing candidate's suffix. A mismatch is an internal
            // error and must never disable the constraint.
            for (suffix) |b| {
                const ok = try constraint.grammar.acceptByte(b);
                if (!ok) {
                    return error.InvalidProtocolPayload;
                }
            }
            constraint.pstate.phase = .json_body;
            // A byte-spelled close shares this token: the payload begins at
            // byte_offset inside it. An atomic close has no bytes at all —
            // the payload begins with the NEXT token (never unwrap the
            // null token bytes).
            constraint.pending_span = if (token_bytes == null)
                .{ .token_index = @intCast(self.generated_ids.items.len + 1), .byte_offset = 0 }
            else
                .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = rp_mod.payloadByteOffset(token_bytes, suffix) };
            self.loop_guard_start = self.generated_ids.items.len;
            log.info("[grammar] reasoning boundary reached; enforcing final-answer schema\n", .{});
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }
        return token;
    }

    /// One constrained sample at the unresolved opener choice. The mask is
    /// the union of schema-legal tokens (the direct answer is constrained
    /// from its first byte) and opener-prefix candidates; once opener bytes
    /// are in flight only opener continuations remain. The JSON grammar never
    /// consumes protocol bytes.
    fn nextChoiceStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
        const s = self.xfm.s;

        const allowed = if (constraint.pstate.phase == .header)
            try rp_mod.applyHeaderMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf)
        else
            try rp_mod.applyChoiceMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        if (allowed == 0) {
            return error.NoValidProtocolToken;
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        const lazy = self.sampleLazy(masked_logits);
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;

        // The choice mask never admits EOS (the grammar is incomplete) and
        // pad tokens carry no admissible bytes; stop the way the constrained
        // path does if one is sampled anyway.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id and constraint.pstate.phase != .header) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        const outcome: rp_mod.ChoiceOutcome = if (constraint.pstate.phase == .header) blk: {
            const suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
            if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
            if (suffix) |bytes| break :blk .{ .payload = bytes };
            break :blk if (constraint.pstate.phase == .reasoning) .opened else .progress;
        } else rp_mod.observeChoice(proto, &constraint.pstate, token, token_bytes);
        switch (outcome) {
            .progress => {},
            .opened => {
                self.loop_guard_start = self.generated_ids.items.len;
                log.info("[grammar] reasoning opener generated; reasoning unconstrained\n", .{});
            },
            .json, .payload => {
                const payload = if (outcome == .payload) outcome.payload else token_bytes;
                if (payload) |bytes| {
                    for (bytes) |b| {
                        const ok = try constraint.grammar.acceptByte(b);
                        if (!ok) {
                            return error.InvalidProtocolPayload;
                        }
                    }
                }
                constraint.pstate.phase = .json_body;
                constraint.pending_span = .{
                    .token_index = @intCast(self.generated_ids.items.len + @as(usize, if (token_bytes == null) 1 else 0)),
                    .byte_offset = if (outcome == .payload) rp_mod.payloadByteOffset(token_bytes, outcome.payload) else 0,
                };
                self.loop_guard_start = self.generated_ids.items.len;
            },
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }
        return token;
    }

    /// Synchronous, grammar-constrained sampling step. Used whenever
    /// `sampling.constraint` is non-null. Builds a token mask from the grammar's
    /// current state, applies it to the pending logits, samples, advances the
    /// grammar by the sampled token's bytes, and pre-launches the next forward
    /// pass to overlap with the next mask build.
    fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.logprobs_n == 0 or !self.has_pending_logits) return self.nextConstrainedToken(allocator);
        // Every constrained arm returns a token drawn (or forced) at THIS position, with no
        // one-token delay, so its entry comes from these logits: raw, before the grammar mask.
        var logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_set(&logits, self.pending_logits));
        const token = (try self.nextConstrainedToken(allocator)) orelse return null;
        if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
        self.pending_logprob = try computeLogprobs(allocator, logits, token, self.logprobs_n, self.xfm.s);
        self.last_logprob = self.pending_logprob;
        self.pending_logprob = null;
        return token;
    }

    fn nextConstrainedToken(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (!self.has_pending_logits) {
            self.done = true;
            return null;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        const constraint = self.sampling.constraint.?;
        if (constraint.proto) |proto| {
            if (constraint.pstate.phase != .json_body) return self.nextReasoningStep(allocator, constraint, proto);
        }
        const s = self.xfm.s;

        var allowed = (try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf)).allowed;
        // The tokenizer's EOS is not necessarily the model's only stop id.
        if (constraint.grammar.isComplete()) {
            for (self.eos_token_ids) |eos_id| {
                if (eos_id < constraint.mask_buf.len and !constraint.mask_buf[eos_id]) {
                    constraint.mask_buf[eos_id] = true;
                    allowed += 1;
                }
            }
        }
        if (allowed == 0) {
            log.warn("[grammar] no token satisfies the schema at this position — disabling further mask enforcement\n", .{});
            constraint.grammar.dead = true;
            @memset(constraint.mask_buf, true);
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        // The next forward is built on the still-lazy sample so the CPU graph
        // build overlaps this step's GPU work; the token is realized after
        // dispatch and only then advances the grammar (the mask for the next
        // logits is built on the next call, off the realized state).
        const lazy = self.sampleLazy(masked_logits);
        defer _ = mlx.mlx_array_free(lazy);
        var next_logits: ?mlx.mlx_array = null;
        if (self.step + 1 < self.max_tokens) {
            if (lazyForward(self.xfm, &self.ctx, lazy)) |nl| {
                const arr = [_]mlx.mlx_array{ lazy, nl };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);
                next_logits = nl;
            } else |_| {}
        }
        errdefer if (next_logits) |nl| {
            _ = mlx.mlx_array_free(nl);
        };
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        // Stop on EOS — do not advance grammar or include in output. A
        // dispatched forward on the stop token is dropped, as on the lazy path.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id) {
                if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        // Advance the grammar by the sampled token's byte sequence. The mask
        // guarantees every byte is accepted (or the token has no byte form, e.g. a
        // special tag) — so a rejection here means a bug we want to surface.
        if (token < constraint.token_bytes.bytes.len) {
            if (constraint.token_bytes.bytes[token]) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        log.warn("[grammar] sampled token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                        constraint.grammar.dead = true;
                        break;
                    }
                }
            }
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (next_logits) |nl| {
            self.pending_logits = nl;
            self.has_pending_logits = true;
        } else if (self.step < self.max_tokens) {
            // Pipeline miss: forward the realized token.
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const nl = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{nl};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = nl;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }

        return token;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    pub fn checkStop(self: *Generator) !bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (tokenStops(self.next_token_id, self.eos_token_ids, &self.consecutive_pad)) {
            self.done = true;
            self.finish_reason = "stop";
            return true;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forwardWith(&self.ctx, tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Diagnostic (`MLX_SERVE_STEP_TRACE=1`): host time of a pipelined serial step, per phase,
/// summed across the request and reported with the spec stats.
const StepTrace = struct {
    const Phase = enum { build, flush, submit, resolve };
    var on: ?bool = null;
    var steps: u64 = 0;
    var ns: [4]u64 = .{ 0, 0, 0, 0 };

    fn enabled() bool {
        if (on) |v| return v;
        const raw = std.c.getenv("MLX_SERVE_STEP_TRACE");
        const v = raw != null and raw.?[0] != '0';
        on = v;
        return v;
    }
    fn watch() ?io_util.Stopwatch {
        return if (enabled()) io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io()) else null;
    }
    fn lap(sw: *?io_util.Stopwatch, phase: Phase) void {
        if (sw.*) |*w| {
            ns[@backingInt(phase)] += w.read();
            w.reset();
        }
    }
    fn report() void {
        if (!enabled() or steps == 0) return;
        const n: f64 = @floatFromInt(steps);
        log.info("  [step-trace] steps={d} build={d:.3} flush={d:.3} submit={d:.3} resolve={d:.3} ms/step\n", .{
            steps,
            @as(f64, @floatFromInt(ns[0])) / n / 1e6,
            @as(f64, @floatFromInt(ns[1])) / n / 1e6,
            @as(f64, @floatFromInt(ns[2])) / n / 1e6,
            @as(f64, @floatFromInt(ns[3])) / n / 1e6,
        });
        steps = 0;
        ns = .{ 0, 0, 0, 0 };
    }
};

/// Evaluates one pipelined decode step with MLX's byte limit on a command buffer off, so the ops limit alone commits:
/// at one token the bound weights trip the byte limit about twice per layer, and those extra command buffers cost decode
/// time. MLX encodes on the thread that calls eval (`needs_commit()` is read at the end of each encode), so the scope covers
/// exactly what this call encodes, and it must stay around this call: a flag set by a forward would also reach whatever
/// that forward left lazy and a cached compiled forward. No prefill work can be in this graph: a prompt's last forward is
/// one token, evaluated when `initWithOptions` draws the first token. MLX_SERVE_DECODE_OPS_ONLY_COMMIT_OFF=1 keeps the
/// byte limit.
fn asyncEvalDecodeStep(vec: mlx.mlx_vector_array) void {
    keep_warm.touch();
    const S = struct {
        var off: ?bool = null;
    };
    const off = S.off orelse blk: {
        const v = std.c.getenv("MLX_SERVE_DECODE_OPS_ONLY_COMMIT_OFF") != null;
        S.off = v;
        if (!v) log.info("[decode] one-token steps commit on the ops limit alone (MLX_SERVE_DECODE_OPS_ONLY_COMMIT_OFF=1 keeps the byte limit)\n", .{});
        break :blk v;
    };
    if (off) {
        _ = mlx.mlx_async_eval(vec);
        return;
    }
    _ = mlx.mlx_lesto_commit_on_ops_limit_only(true);
    _ = mlx.mlx_async_eval(vec);
    _ = mlx.mlx_lesto_commit_on_ops_limit_only(false);
}

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, ctx: *ForwardCtx, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    var sw = StepTrace.watch();
    const logits = try xfm.forwardWith(ctx, reshaped);
    StepTrace.lap(&sw, .build);
    return logits;
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
/// Batched sibling of `probsAtLastPos`: temperature → top-k → top-p → min-p →
/// softmax over EVERY position of `[1, L, V]` logits in one set of
/// row-parallel kernels. A per-position loop pays L separate ~vocab-sized
/// sort/topk kernel launches per spec-decode round; batched it's one each.
/// All filter helpers operate on the last axis, so leading dims pass through.
fn probsAllPositions(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    try mlx.check(mlx.mlx_array_set(&current, logits_3d));

    // Reserved-token suppression (batched stochastic verify) - same
    // rationale as `probsAtLastPos`.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if ((sampling.min_p orelse 0) > 0) {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applyMinP(&masked, current, sampling.min_p.?, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

fn probsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Reserved-token suppression: the filtered probs feed spec-verify
    // acceptance AND the residual corrections, so a suppressed draft's
    // acceptance probability is exactly 0 and the residual can't re-draw it.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    const owned = current;
    current = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Temperature → top-k → top-p → min-p → softmax over the LAST axis of an owned
/// `[.., V]` array (consumed). The ONE place a proposal or target density is
/// built: a draft sampled from a distribution that is not byte-for-byte the
/// `q` handed to `specAcceptProb` breaks exactness silently, so both sides
/// come through here. Row-independent, so it serves a single `[1, V]` row and
/// a whole `[m, V]` block identically.
fn filteredProbsRows(owned_rows: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = owned_rows;
    errdefer _ = mlx.mlx_array_free(current);
    // Apply temperature → top-k → top-p → min-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if ((sampling.min_p orelse 0) > 0) {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applyMinP(&masked, current, sampling.min_p.?, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// Proposal densities for EVERY row of a `[1, m, V]` draft-logit block, as
/// `[m, V]`. One filter pass and (at the call site) one categorical over m
/// rows, rather than m of each — the draft block is on the critical path
/// ahead of the verify forward.
fn filteredProbsBlock(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    const rows_shape = [_]c_int{ shape[0] * shape[1], shape[2] };
    try mlx.check(mlx.mlx_reshape(&rows, logits_3d, &rows_shape, 2, s));
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, rows, m, s);
        _ = mlx.mlx_array_free(rows);
        rows = masked;
    }
    const owned = rows;
    rows = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
fn probAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`) with
/// the request's key (`seedKey`: a seeded request replays, `draw` = the position).
/// Returns a u32 token id (caller can append directly).
fn sampleFromProbs(probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by PLD's stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// One row of a `[m, V]` density block as its own `[1, V]` array, so it can
/// feed `probAt` and `sampleResidual` (both row-shaped). Caller owns it.
fn sliceProbRow(rows_2d: mlx.mlx_array, row: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(rows_2d);
    const start = [_]c_int{ @intCast(row), 0 };
    const stop = [_]c_int{ @as(c_int, @intCast(row)) + 1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, rows_2d, &start, 2, &stop, 2, &strides, 2, s));
    return out;
}

/// The DFlash2 selector's proposal density for one step as a `[1, V]` row:
/// its per-candidate softmax scattered into zeros - q is zero off the
/// candidate set by construction, so this is the exact density the draft was
/// drawn from (the residual-correction contract).
fn selectorQRow(sp: *const dflash_mod.SelectedPath, step: usize, m: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const kk = sp.cand_ids.len / @as(usize, m);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    const z_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_zeros(&zeros, &z_shape, 2, .float32, s));
    const row_shape = [_]c_int{ 1, @intCast(kk) };
    const ids_arr = mlx.mlx_array_new_data(sp.cand_ids.ptr + step * kk, &row_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const vals_arr = mlx.mlx_array_new_data(sp.q.?.ptr + step * kk, &row_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(vals_arr);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, ids_arr, vals_arr, 1, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
fn sampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return sampleFromProbs(residual, sampling, s);
}

pub fn sampleTokenLazy(logits_in: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    // Reserved-token suppression first, so every path below (greedy fast
    // path included — the collapse drew `<|fim_hole|>` at temp 0) sees the
    // masked row. Lazy like everything else here. The handle starts null-ctx
    // and is only materialized when a mask exists — the fast path below
    // exists to save ONE FFI call per decode step, so the no-mask case (every
    // non-suppressing model) must not spend two on an empty array.
    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    const logits = if (sampling.suppress_mask) |m| blk: {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, logits_in, m, s) catch break :blk logits_in;
        break :blk suppressed;
    } else logits_in;

    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Greedy + seq_len==1 (the decode hot path): one mlx op total. argmax_axis
    // over the vocab dim of a `[1, 1, V]` tensor yields a `[1, 1]` int array,
    // which downstream (resolvePendingToken / lazyForward / async_eval vector)
    // treats identically to `[1]`. Skipping the otherwise-needed reshape +
    // argmax-on-2D combo cuts ~one FFI call per decode step.
    if (seq_len == 1 and sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, logits, -1, false, s);
        return result;
    }

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    if (sampling.keyed and sampling.seed != null) {
        defer _ = mlx.mlx_array_free(current);
        const p: keyed_sample.Params = .{ .seed = sampling.seed.?, .temperature = sampling.temperature, .top_p = sampling.top_p, .top_k = sampling.top_k, .min_p = sampling.min_p orelse 0 };
        const pos = [_]u32{@intCast(sampling.position_base + sampling.draw)};
        return keyed_sample.sample(current, p, &pos, s) catch mlx.mlx_array_new();
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k and top-p in one pass (lazy)
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }
    if ((sampling.min_p orelse 0) > 0) {
        var next = mlx.mlx_array_new();
        if (applyMinP(&next, current, sampling.min_p.?, s)) |_| {
            _ = mlx.mlx_array_free(current);
            current = next;
        } else |err| {
            _ = mlx.mlx_array_free(next);
            log.warn("[sampling] min_p failed ({s}); this draw skips min_p\n", .{@errorName(err)});
        }
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

pub var sample_rows_evals: u64 = 0;

/// One sampling eval for every row of a batched tick. Homogeneous unseeded rows
/// draw over one stacked `[N, V]` graph (one argmax or one categorical draw);
/// a seed, mixed filters or a lone row take the per-row arm, whose parts are
/// reshaped to `[1]` before the concat because a greedy draw is `[1, 1]` and a
/// categorical draw is `[1]`. Every row advances its `draw` before the eval, so
/// a failed eval never replays a key.
pub fn sampleRows(out: []i32, row_logits: []const mlx.mlx_array, params: []SamplingParams, s: mlx.mlx_stream) !void {
    std.debug.assert(out.len == row_logits.len and out.len == params.len);
    const n = row_logits.len;
    if (n == 0) return;
    const lazy = try sampleRowsLazy(row_logits, params, s);
    defer _ = mlx.mlx_array_free(lazy);
    for (params) |*p| p.draw +%= 1;
    sample_rows_evals += 1;
    try mlx.check(mlx.mlx_array_eval(lazy));
    if (n == 1) {
        try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy));
        return;
    }
    const data = mlx.mlx_array_data_uint32(lazy) orelse return error.MlxError;
    for (out, data[0..n]) |*dst, v| dst.* = @intCast(v);
}

/// Rows may share one stacked graph only when every filter matches; a seed
/// always takes the per-row arm so its key stays per row.
fn sampleRowsHomogeneous(params: []const SamplingParams) bool {
    const p0 = params[0];
    for (params[1..]) |p| {
        if (p.temperature != p0.temperature or p.top_k != p0.top_k or p.top_p != p0.top_p or (p.min_p orelse 0) != (p0.min_p orelse 0)) return false;
        const m0 = p0.suppress_mask;
        const m1 = p.suppress_mask;
        if (m0 == null and m1 == null) continue;
        if (m0 == null or m1 == null) return false;
        if (m0.?.ctx != m1.?.ctx) return false;
    }
    return true;
}

fn sampleRowsAllGreedy(params: []const SamplingParams) bool {
    for (params) |p| {
        if (p.temperature >= 0.01) return false;
    }
    return true;
}

fn concatLogitRows(rows: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (rows.len == 1) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, rows[0]));
        return out;
    }
    const vec = mlx.mlx_vector_array_new_data(rows.ptr, rows.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

fn logitsAsNV(logits: mlx.mlx_array, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (sh.len == 2) {
        try mlx.check(mlx.mlx_array_set(&out, logits));
        return out;
    }
    if (sh.len != 3 or sh[0] != n) return error.BadShape;
    if (sh[1] == 1) {
        const sq = [_]c_int{ n, sh[2] };
        try mlx.check(mlx.mlx_reshape(&out, logits, &sq, 2, s));
        return out;
    }
    const start = [_]c_int{ 0, sh[1] - 1, 0 };
    const stop = [_]c_int{ n, sh[1], sh[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
    const sq = [_]c_int{ n, sh[2] };
    try mlx.check(mlx.mlx_reshape(&out, sliced, &sq, 2, s));
    return out;
}

fn sampleRowsLazyConcat(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var parts: [scheduler_mod.MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    const n = row_logits.len;
    if (n > parts.len) return error.BatchTooWide;
    var n_parts: usize = 0;
    // One release on every exit: the concat output holds its own references.
    defer {
        for (parts[0..n_parts]) |a| _ = mlx.mlx_array_free(a);
    }
    const one = [_]c_int{1};
    for (row_logits, params) |logits, p| {
        const raw = sampleTokenLazy(logits, p, s);
        defer _ = mlx.mlx_array_free(raw);
        parts[n_parts] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&parts[n_parts], raw, &one, 1, s));
        n_parts += 1;
    }
    return concatLogitRows(parts[0..n_parts], s);
}

fn sampleRowsAnySeed(params: []const SamplingParams) bool {
    for (params) |p| {
        if (p.seed != null) return true;
    }
    return false;
}

fn sampleRowsLazy(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const n: c_int = @intCast(row_logits.len);
    if (row_logits.len == 1 or !sampleRowsHomogeneous(params) or sampleRowsAnySeed(params)) {
        return sampleRowsLazyConcat(row_logits, params, s);
    }
    const stacked = try concatLogitRows(row_logits, s);
    defer _ = mlx.mlx_array_free(stacked);
    var current = try logitsAsNV(stacked, n, s);
    var owned = true;
    errdefer if (owned) {
        _ = mlx.mlx_array_free(current);
    };
    const p0 = params[0];

    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (p0.suppress_mask) |m| {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, current, m, s) catch {
            _ = mlx.mlx_array_free(suppressed);
            suppressed = .{ .ctx = null };
        };
        if (suppressed.ctx != null) {
            _ = mlx.mlx_array_free(current);
            current = suppressed;
            suppressed = .{ .ctx = null };
        }
    }

    if (sampleRowsAllGreedy(params)) {
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_argmax_axis(&result, current, -1, false, s));
        _ = mlx.mlx_array_free(current);
        owned = false;
        return result;
    }

    if (p0.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(p0.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        mlx.check(mlx.mlx_divide(&next, current, temp_arr, s)) catch |err| {
            _ = mlx.mlx_array_free(next);
            return err;
        };
        _ = mlx.mlx_array_free(current);
        current = next;
    }
    if (p0.top_k > 0 or p0.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, p0.top_p, p0.top_k, s) catch {
            _ = mlx.mlx_array_free(next);
            next = current;
            current = mlx.mlx_array_new();
        };
        if (current.ctx != null) _ = mlx.mlx_array_free(current);
        current = next;
    }
    if ((p0.min_p orelse 0) > 0) {
        var next = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(next);
        try applyMinP(&next, current, p0.min_p.?, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    const key = seedKey(p0);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    _ = mlx.mlx_array_free(current);
    owned = false;
    return sampled;
}

/// PRNG key for one draw: `seed` mixed with the draw index, so a seeded
/// request replays byte-for-byte and consecutive draws never share a key.
/// Null-ctx (MLX global RNG) when the request set no seed.
/// `logits + Gumbel(key)` over the vocab row `[1, V]`, in f32.
fn seedKey(sampling: SamplingParams) mlx.mlx_array {
    const seed = sampling.seed orelse return mlx.mlx_array_new();
    var key = mlx.mlx_array_new();
    _ = mlx.mlx_random_key(&key, seed +% sampling.draw *% 0x9E3779B97F4A7C15);
    return key;
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    // logprobs_n rides InitOptions so init's argmax-only gate sees it (a
    // post-init field write would let the split-prefill final-token forward
    // engage the pruned lm_head on a logprobs request).
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .logprobs_n = logprobs_n });
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// PLD-enabled non-streaming variant of `generate`. Model-agnostic — works on
/// every supported architecture, no extra weights required. Logprobs and
/// constrained sampling are unsupported (asserted out by `nextPld`).
///
/// `draft_len` and `key_len` come from server config (`--pld-draft-len` /
/// `--pld-key-len`); typical values are 5 and 3 respectively.
pub fn generatePld(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    draft_len: u32,
    key_len: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .pld_enabled = true, .lookup_prompt = lookup_prompt });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (PLD): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    // Decode loop. Each `nextPld` returns 1..=(1+draft_len) tokens. Stop on
    // EOS / max_tokens / timeout. We check stop conditions on every emitted
    // token (drafts can include EOS just like regular sampling) so the early
    // exit is correct.
    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextPld(allocator, draft_len, key_len)) orelse break;
        defer allocator.free(result.tokens);
        // Match `generate`'s convention: stop tokens are NOT included in
        // output_ids. Check before appending — the speculative path has to do
        // this explicitly because `nextPld` emits multiple tokens at once and
        // can't return-null mid-batch like the single-token `next` does.
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishPldResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

/// Drafter-enabled non-streaming variant of `generate`. Mirrors
/// `generatePld` (multi-token-per-step emit pattern) but the draft comes from
/// a Gemma 4 assistant drafter cross-attending into the target's KV cache
/// instead of an n-gram lookup.
///
/// `drafter` must already be `bind()`-ed to `xfm`. `block_size` is the
/// per-round token budget (drafter forwards = block_size - 1; verify forward
/// length = block_size).
pub fn generateDrafter(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    drafter: *DrafterModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    block_size: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .drafter_enabled = true,
        .drafter = drafter,
        .drafter_block_size = block_size,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (drafter): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextDrafter(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishDrafterResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

fn finishDrafterResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.drafter_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.drafter_accepted_tokens)) / @as(f64, @floatFromInt(gen.drafter_attempted));
        log.info("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; drafter accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.drafter_accepted_tokens,
            gen.drafter_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

fn finishPldResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.pld_attempted > 0) {
        // "Tokens saved" = accepted_tokens (drafts that landed) + 0 from
        // verify forwards that ran. Acceptance ratio is per-position, so we
        // compute average tokens accepted per attempt for visibility.
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.pld_accepted_tokens)) / @as(f64, @floatFromInt(gen.pld_attempted));
        log.info("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; pld accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.pld_accepted_tokens,
            gen.pld_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; no n-gram matches found)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

/// Token-only stop conditions of `checkStop`; `consecutive_pad` is carried state reset by any real token.
pub fn tokenStops(next_token_id: u32, eos_token_ids: []const u32, consecutive_pad: *u32) bool {
    if (isEosId(next_token_id, eos_token_ids)) return true;
    if (next_token_id != 0) {
        consecutive_pad.* = 0;
        return false;
    }
    consecutive_pad.* += 1;
    return consecutive_pad.* >= 3;
}

/// Max cycle length (in tokens) scanned by `isDegenerateTailLoop`, and how many
/// identical repetitions of that cycle count as "stuck". A real answer — prose,
/// code, a markdown table — essentially never repeats an identical ≤8-token
/// cycle 16 times in a row, so these won't fire on legitimate output, while a
/// model that has collapsed into spamming one short phrase is caught within a
/// few dozen tokens instead of running all the way to `max_tokens`.
pub const degenerate_loop_max_period: usize = 8;
pub const degenerate_loop_reps: usize = 16;
/// A short cycle must also cover this many tokens before it convicts: at
/// period 1 or 2, 16 reps is a digit row or a zeroed array, not a loop.
pub const degenerate_loop_min_span: usize = 128;
// Tier 2 (2026-08-02 shooter wrap-up class): a two-sentence cycle of ~58
// tokens repeated 26 times evaded the 8-token tier. Long periods demand
// fewer reps — 10 verbatim repetitions of a 9..64-token cycle is
// degeneration with overwhelming probability (identical long lines in real
// code repeat a handful of times, not ten).
pub const degenerate_loop_long_max_period: usize = 64;
pub const degenerate_loop_long_reps: usize = 10;
/// ...except when the "line" is a tile-map row (2026-09-15: a 27B wrote ten
/// identical `"100000000000000000000001",` rows, a 28-token cycle, in an
/// honest map file). The long tier also demands this much cycling: 37 such
/// rows, or a 58-token sentence 18 times. A real loop still dies in ~13 s.
pub const degenerate_loop_long_min_span: usize = 1024;

// Tier 3 (2026-08-04 agent-traffic class): a restatement loop that VARIES its
// phrasing has no exact cycle at any period, so both tiers above are blind by
// construction. What it does have is a long stretch of output that recycles a
// tiny vocabulary and introduces almost no new n-grams.
//
// Everything about this tier is deliberately reluctant, because unlike the
// exact tiers it is a fuzzy judgement and a false cut truncates a real answer:
// the window is LONG (a legitimate repetitive passage — a table, a block of
// near-identical code, a list scaffold — is finite and ends well inside it),
// the two ratios must BOTH be low (either one alone convicts honest output —
// a numeric table has few distinct tokens, a repeated code scaffold has few
// distinct n-grams), and the window is longer than the exact tiers' reach so
// this tier only ever speaks about spans they have already declined.
pub const near_repeat_window: usize = 1024;
pub const near_repeat_ngram: usize = 4;
pub const near_repeat_max_ngram_ratio: f32 = 0.35;
pub const near_repeat_max_token_ratio: f32 = 0.12;
/// Third ratio (2026-08-05): how much of the window's SECOND half is n-grams
/// its first half never had. Measured across both shapes at the shipped
/// window: restatement loops 0.019-0.022, healthy repetitive output
/// 0.298-0.827 (dense procedural scene code, a markdown table, the voxel
/// artifact that was wrongly cut). 0.10 sits ~4.5x above the loops and ~3x
/// below the closest healthy case.
pub const near_repeat_max_novelty: f32 = 0.10;

/// Open-addressed distinct-counter sized for one window. Stack-resident, so
/// the whole tier is allocation-free and runs in O(window) per decode tick.
fn DistinctSet(comptime cap: usize) type {
    return struct {
        const Self = @This();
        const empty_key: u64 = std.math.maxInt(u64);
        keys: [cap]u64 = @splat(empty_key),
        n: usize = 0,

        /// True when `key` is already present. Read-only; used to ask whether
        /// the window's second half is introducing anything its first half
        /// did not have.
        fn contains(self: *const Self, key: u64) bool {
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) return false;
                if (self.keys[i] == k) return true;
                i = (i + 1) % cap;
            }
        }

        /// True when `key` was not already present.
        fn insert(self: *Self, key: u64) bool {
            // maxInt is the empty sentinel; fold the one colliding key onto 0.
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) {
                    self.keys[i] = k;
                    self.n += 1;
                    return true;
                }
                if (self.keys[i] == k) return false;
                i = (i + 1) % cap;
            }
        }
    };
}

/// The three-ratio judgement over ONE window-sized span. Split out so the trim
/// search can slide the same window backwards without re-deriving the rule.
fn nearRepeatWindowIsDegenerate(window: []const u32) bool {
    // Load factor 0.5 keeps the linear probe short even when every entry is
    // distinct (the healthy case, which is also the hot one).
    var toks = DistinctSet(near_repeat_window * 2){};
    for (window) |t| _ = toks.insert(t);
    const token_ratio = @as(f32, @floatFromInt(toks.n)) / @as(f32, @floatFromInt(window.len));
    if (token_ratio > near_repeat_max_token_ratio) return false;

    var grams = DistinctSet(near_repeat_window * 2){};
    var i: usize = 0;
    while (i + near_repeat_ngram <= window.len) : (i += 1) {
        _ = grams.insert(gramHash(window[i .. i + near_repeat_ngram]));
    }
    const n_gram_positions = window.len - near_repeat_ngram + 1;
    const gram_ratio = @as(f32, @floatFromInt(grams.n)) / @as(f32, @floatFromInt(n_gram_positions));
    if (gram_ratio > near_repeat_max_ngram_ratio) return false;

    // Third ratio: is the window still PROGRESSING? Both ratios above are
    // properties of a vocabulary, and procedurally generated code has the same
    // vocabulary profile as a loop — a fixed call template plus a small colour
    // palette (live 2026-08-05: a voxel scene cut at 16241 tokens, the user got
    // no file at all). What a loop does NOT do is keep introducing material:
    // measured, a restatement loop's second half brings 1.9-2.2% n-grams its
    // first half never had, while healthy repetitive output brings 29.8-82.7%.
    // Requiring all THREE keeps the tier's reluctance in the direction that
    // matters — a missed loop still ends at max_tokens, a false cut destroys
    // work that was going fine.
    return halfNovelty(window, near_repeat_ngram) <= near_repeat_max_novelty;
}

/// Share of the window's second-half distinct n-grams absent from its first
/// half. 1.0 when the second half has no n-grams at all (nothing new because
/// there is nothing) is the loop reading, so that case returns 0.
fn halfNovelty(window: []const u32, n: usize) f32 {
    var first_half = DistinctSet(near_repeat_window * 2){};
    const mid = window.len / 2;
    var fi: usize = 0;
    while (fi + n <= mid) : (fi += 1) {
        _ = first_half.insert(gramHash(window[fi .. fi + n]));
    }
    var second_half = DistinctSet(near_repeat_window * 2){};
    var novel: usize = 0;
    var distinct_second: usize = 0;
    var si: usize = mid;
    while (si + n <= window.len) : (si += 1) {
        const h = gramHash(window[si .. si + n]);
        if (!second_half.insert(h)) continue; // count each distinct gram once
        distinct_second += 1;
        if (!first_half.contains(h)) novel += 1;
    }
    if (distinct_second == 0) return 0;
    return @as(f32, @floatFromInt(novel)) / @as(f32, @floatFromInt(distinct_second));
}

/// One hash for both the ratio pass and the novelty pass — two spellings of
/// this would silently compare different things.
fn gramHash(gram: []const u32) u64 {
    var h: u64 = 0;
    for (gram) |t| h = h *% 0x100000001b3 ^ t;
    return h;
}

/// Detect a NEAR-repeat tail loop: the last `near_repeat_window` tokens keep
/// restating the same thing in slightly different words. Pure; reads only the
/// tail, so cost is independent of total generated length.
pub fn isNearRepeatTailLoop(tokens: []const u32) bool {
    if (tokens.len < near_repeat_window) return false;
    return nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..]);
}

/// How far back the near-repeat trim search steps, and how far it may reach.
/// The window is judged whole at each stop, so the set stays window-sized
/// however long the loop ran — sizing a set to the FULL span instead would
/// put tens of KB on the inference thread's stack.
pub const near_repeat_step: usize = 128;
/// Live 2026-09-15: a 27B wrote five 24x19 maps of identical rows in 2.8k tokens.
pub const near_repeat_min_span: usize = 4096;
pub const near_repeat_max_lookback: usize = 8192;

/// A convicted degenerate tail: which tier saw it, and where the degenerate
/// span BEGINS in the generated ids.
///
/// The start is what makes a loop cut recoverable. An agent client re-sends
/// the cut turn's content as history, the model reads its own loop back and
/// resumes it — five loop-stops in a row, each firing sooner than the last
/// (live 2026-08-05, under pi). Emitting only the prefix means
/// the loop cannot round-trip into the next prompt.
pub const DegenerateTail = struct {
    tier: Tier,
    /// First index of the degenerate span; `tokens[0..start]` is what a
    /// client should be shown. For the exact tiers ONE copy of the cycle is
    /// deliberately kept — the truncated answer should still show what the
    /// model was doing when it got stuck, and one copy cannot sustain a loop.
    start: usize,

    pub const Tier = enum { exact_cycle, long_cycle, near_repeat };
};

/// The smallest period in `[min_period, max_period]` whose cycle repeats
/// `reps` times at the tail, or null. `isDegenerateTailLoopRange` is this
/// predicate — one implementation, so the detector and the trim can never
/// disagree about what was convicted.
fn exactCyclePeriod(tokens: []const u32, min_period: usize, max_period: usize, reps: usize, min_span: usize) ?usize {
    if (max_period == 0 or reps < 2) return null;
    var p: usize = @max(min_period, 1);
    while (p <= max_period) : (p += 1) {
        const span = @max(p * reps, min_span);
        if (tokens.len < span) continue;
        const tail = tokens[tokens.len - span ..];
        var periodic = true;
        var i: usize = p;
        while (i < tail.len) : (i += 1) {
            if (tail[i] != tail[i - p]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return p;
    }
    return null;
}

/// Walk the period-`p` cycle backwards past the `reps` that convicted it: a
/// loop that ran 200 times must be trimmed at 200, not at the threshold.
fn trailingCycleStart(tokens: []const u32, p: usize) usize {
    var start = tokens.len - p;
    while (start >= p) {
        if (!std.mem.eql(u32, tokens[start - p .. start], tokens[start .. start + p])) break;
        start -= p;
    }
    return start; // first index of the FIRST copy of the cycle
}

/// Convict a degenerate tail and say where it starts. Tier order matches
/// `scheduler.loopStopReason`: the exact tiers speak first, and the fuzzy
/// near-repeat tier only ever judges spans they have already declined.
pub fn degenerateTail(tokens: []const u32) ?DegenerateTail {
    if (exactCyclePeriod(tokens, 1, degenerate_loop_max_period, degenerate_loop_reps, degenerate_loop_min_span)) |p| {
        return .{ .tier = .exact_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (exactCyclePeriod(
        tokens,
        degenerate_loop_max_period + 1,
        degenerate_loop_long_max_period,
        degenerate_loop_long_reps,
        degenerate_loop_long_min_span,
    )) |p| {
        return .{ .tier = .long_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (tokens.len < near_repeat_window) return null;
    if (!nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..])) return null;

    // Slide the window back while it keeps convicting. A restatement loop
    // that has run for 3000 tokens is degenerate for all 3000 — trimming only
    // the last window would hand the client the rest of the loop back.
    var start = tokens.len - near_repeat_window;
    const floor = if (tokens.len > near_repeat_window + near_repeat_max_lookback)
        tokens.len - near_repeat_window - near_repeat_max_lookback
    else
        0;
    while (start >= floor + near_repeat_step) {
        const cand = start - near_repeat_step;
        if (!nearRepeatWindowIsDegenerate(tokens[cand .. cand + near_repeat_window])) break;
        start = cand;
    }
    // A file of near-identical rows (a lazy tile map, a zero-heavy table)
    // reads as a loop under every content measure; what it does that a loop
    // never does is END. Convict only once the degenerate span has outrun any
    // such file — a real restatement loop is still cut at ~4k tokens instead
    // of max_tokens, and a false cut destroys work.
    if (tokens.len - start < near_repeat_min_span) return null;
    return .{ .tier = .near_repeat, .start = start };
}

/// Stall clock for the request timeout: the deadline measures time since the
/// last PRODUCED token, not since the request started. A wall-clock request
/// timeout kills legitimate long generations — live capture 2026-07-03:
/// A model writing a 33KB file in one tool call decodes for >300s at
/// ~30 tok/s and was guillotined mid-call by the 300s default, which then
/// surfaced as a "butchered" path-only tool call. Progress is detected from
/// the generated-token COUNT at each check, so every decode path (regular,
/// PLD, drafter, DFlash - which don't all share an emit site) resets the clock
/// without instrumentation; a request that stops producing (hung forward,
/// deadlock) still times out after `timeout_ns` of silence.
pub const StallClock = struct {
    last_progress_ns: u64 = 0,
    last_progress_count: usize = 0,

    pub fn expired(self: *StallClock, now_ns: u64, generated_count: usize, timeout_ns: u64) bool {
        if (generated_count != self.last_progress_count) {
            self.last_progress_count = generated_count;
            self.last_progress_ns = now_ns;
        }
        if (timeout_ns == 0) return false;
        return now_ns -| self.last_progress_ns >= timeout_ns;
    }
};

/// Detect a degenerate tail loop: the model is stuck emitting the same short
/// token cycle over and over. Returns true when the last `reps` repetitions of
/// some period-`p` cycle (1 ≤ p ≤ `max_period`) are byte-identical.
///
/// Motivation: Gemma 4 12B sometimes collapses after a large/confusing tool
/// result and spams the thinking opener `<|channel>thought` forever; with no
/// repeat penalty (the default) and a now-generous `max_tokens`, nothing else
/// stops it. The decode loop calls this each tick and cuts the slot short.
///
/// Pure and cheap: only the trailing `max_period * reps` ids are inspected, so
/// cost is independent of total generated length.
pub fn isDegenerateTailLoop(tokens: []const u32, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, 1, max_period, reps, degenerate_loop_min_span) != null;
}

/// Range variant so a long-period tier can scan 9..64 without also lowering
/// the rep threshold for short cycles (a few "ha ha ha" reps stay legal).
pub fn isDegenerateTailLoopRange(tokens: []const u32, min_period: usize, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, min_period, max_period, reps, 0) != null;
}

/// Compute a pooled (per the model's `pooling_mode` — mean by default),
/// L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const seqs = [_][]const u32{token_ids};
    const rows = try computeEmbeddingsBatch(allocator, xfm, &seqs);
    defer allocator.free(rows);
    return rows[0];
}

/// GPU batch-size cap for encoder embedding forwards: bounds padded-batch
/// memory while keeping the GPU saturated.
pub const EMBED_MAX_BATCH: usize = 64;

/// Padded-token budget per embedding sub-batch (issue #117): the item cap
/// alone lets ONE long input inflate the whole padded allocation (64 rows
/// padded to a 32K outlier = 2M positions of hidden state). The budget bounds
/// rows × padded length at the historical worst case (64 rows × 512 — every
/// legacy batch is unchanged); long-context inputs simply ride smaller
/// sub-batches. A single over-budget input still runs, alone.
pub const EMBED_TOKEN_BUDGET: usize = EMBED_MAX_BATCH * 512;

/// End index (exclusive) of the embedding sub-batch starting at `start`:
/// grows while under BOTH the item cap and the padded-footprint budget
/// (rows × running max length). Always takes at least one item, so an input
/// longer than the whole budget is processed rather than looping forever.
/// Input order is preserved — sub-batches are contiguous slices.
pub fn embedSubBatchEnd(seqs: []const []const u32, start: usize, max_items: usize, budget: usize) usize {
    var end = start;
    var max_len: usize = 0;
    while (end < seqs.len and end - start < max_items) {
        const grown_max = @max(max_len, seqs[end].len);
        if (end > start and (end - start + 1) * grown_max > budget) break;
        max_len = grown_max;
        end += 1;
    }
    return end;
}

/// One padded batch of token sequences ready for an encoder forward.
pub const PaddedBatch = struct {
    ids: []i32, // [B * max_len] row-major, pad id 0
    lengths: []usize, // [B]
    max_len: usize,

    pub fn deinit(self: *PaddedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.lengths);
    }
};

/// Pad `seqs` into one [B, max_len] i32 buffer (pad id 0). Padded positions
/// are excluded from pooling (`maskedMeanPoolNormalize`), and the causal
/// forward keeps a real row from attending to the later pad positions, so the
/// pad id value never leaks into results.
pub fn buildPaddedBatch(allocator: std.mem.Allocator, seqs: []const []const u32) !PaddedBatch {
    var max_len: usize = 0;
    for (seqs) |seq| max_len = @max(max_len, seq.len);
    if (max_len == 0) return error.EmptyInput;

    const ids = try allocator.alloc(i32, seqs.len * max_len);
    errdefer allocator.free(ids);
    const lengths = try allocator.alloc(usize, seqs.len);
    errdefer allocator.free(lengths);
    @memset(ids, 0);
    for (seqs, 0..) |seq, b| {
        lengths[b] = seq.len;
        for (seq, 0..) |id, t| ids[b * max_len + t] = @intCast(id);
    }
    return .{ .ids = ids, .lengths = lengths, .max_len = max_len };
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions.
/// Returns the pooled [B, H] mlx array (f32-promoted); caller frees.
pub fn maskedMeanPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    // Pool weights [B, T, 1]: 1/len over real positions, 0 over padding — a
    // weighted sum along T is then exactly the masked mean. f32 weights also
    // promote a bf16 hidden so the final data extraction is float32-safe.
    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    for (lengths, 0..) |len, b| {
        const denom: f32 = @floatFromInt(@max(len, 1));
        for (0..seq_len) |t| {
            wbuf[b * seq_len + t] = if (t < len) 1.0 / denom else 0.0;
        }
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions
/// and L2-normalize. Returns B owned rows of H f32 each (plus the outer
/// slice); caller frees all.
pub fn maskedMeanPoolNormalize(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) ![][]f32 {
    const pooled = try maskedMeanPool(allocator, hidden, lengths, s);
    defer _ = mlx.mlx_array_free(pooled);
    return l2NormalizeRows(allocator, pooled, s);
}

/// L2-normalize each row of `pooled` [B, H] and read out as owned f32 rows.
pub fn l2NormalizeRows(allocator: std.mem.Allocator, pooled: mlx.mlx_array, s: mlx.mlx_stream) ![][]f32 {
    const pshape = mlx.getShape(pooled);
    const batch: usize = @intCast(pshape[0]);
    const dim: usize = @intCast(pshape[1]);

    // L2 normalize rows: pooled / max(sqrt(sum(pooled^2)), eps).
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    try mlx.check(mlx.mlx_array_eval(normalized));
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const rows = try allocator.alloc([]f32, batch);
    var done: usize = 0;
    errdefer {
        for (rows[0..done]) |r| allocator.free(r);
        allocator.free(rows);
    }
    for (rows, 0..) |*row, b| {
        row.* = try allocator.alloc(f32, dim);
        @memcpy(row.*, data_ptr[b * dim .. (b + 1) * dim]);
        done += 1;
    }
    return rows;
}

/// Compute embeddings for many token sequences in GPU batches: each chunk of
/// up to `EMBED_MAX_BATCH` sequences is padded to its own max length,
/// forwarded ONCE through the encoder with a key-padding mask,
/// masked-mean-pooled, and L2-normalized. Input order preserved. Caller
/// frees every returned row and the outer slice.
pub fn computeEmbeddingsBatch(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
) ![][]f32 {
    return computeEmbeddingsBatchWith(allocator, xfm, seqs);
}

/// Mean-pooled, L2-normalized embeddings of `seqs`, forwarded in padded, key-masked sub-batches.
pub fn computeEmbeddingsBatchWith(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
) ![][]f32 {
    const results = try allocator.alloc([]f32, seqs.len);
    var filled: usize = 0;
    errdefer {
        for (results[0..filled]) |r| allocator.free(r);
        allocator.free(results);
    }
    var start: usize = 0;
    while (start < seqs.len) {
        // The decoder trunk forwards through the KV cache: without a reset, a
        // sub-batch attends to earlier sub-batches' rows.
        try xfm.resetCache();
        const sub = seqs[start..embedSubBatchEnd(seqs, start, EMBED_MAX_BATCH, EMBED_TOKEN_BUDGET)];
        var pb = try buildPaddedBatch(allocator, sub);
        defer pb.deinit(allocator);

        const shape = [_]c_int{ @intCast(sub.len), @intCast(pb.max_len) };
        const input = mlx.mlx_array_new_data(pb.ids.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);

        // Right padding is harmless to a causal pass, so no key-padding mask is built.
        const hidden = try xfm.forwardEmbedding(input);
        defer _ = mlx.mlx_array_free(hidden);

        // Pipeline order: masked mean pool, then L2 normalize.
        const pooled = switch (xfm.config.effectivePooling()) {
            .mean => try maskedMeanPool(allocator, hidden, pb.lengths, xfm.s),
        };
        defer _ = mlx.mlx_array_free(pooled);
        const rows = try l2NormalizeRows(allocator, pooled, xfm.s);
        defer allocator.free(rows);
        for (rows, 0..) |r, i| {
            results[start + i] = r;
            filled += 1;
        }
        start += sub.len;
    }
    return results;
}

const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Reserved-token suppression — before any other transform, so the greedy
    // arm below can't draw a suppressed id. `logprobs_logits` stays
    // `last_logits`: the reported distribution is the MODEL's, so under
    // suppression rank 1 may legitimately differ from the chosen token.
    var suppressed = mlx.mlx_array_new();
    var suppressed_owned = false;
    defer if (suppressed_owned) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (sampling.suppress_mask) |m| {
        try applySuppressMask(&suppressed, current, m, s);
        current = suppressed;
        suppressed_owned = true;
    }

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (sampling.penalized()) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyPenalties(&penalized, current, ids, sampling, s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (sampling.temperature < 0.01) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    if (sampling.keyed and sampling.seed != null) {
        const pos = [_]u32{@intCast(sampling.position_base + sampling.draw)};
        const tok = try keyed_sample.sample(current, .{ .seed = sampling.seed.?, .temperature = sampling.temperature, .top_p = sampling.top_p, .top_k = sampling.top_k, .min_p = sampling.min_p orelse 0 }, &pos, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        const token_id = (mlx.mlx_array_data_uint32(tok) orelse return error.MlxArrayDataNull)[0];
        const logprob_result: ?LogprobResult = if (logprobs_n > 0) try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s) else null;
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // Logprobs report the MODEL's distribution, so they read the position's raw
    // logits — not `current`, which by here carries the client's temperature
    // (and any penalty) and would make the same token's logprob move with a
    // sampling knob.
    const logprobs_logits = last_logits;

    // Apply top-k and top-p in one pass
    var filtered = mlx.mlx_array_new();
    var filtered_owned = false;
    defer if (filtered_owned) {
        _ = mlx.mlx_array_free(filtered);
    };

    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        try filterTopKTopP(&filtered, current, sampling.top_p, sampling.top_k, s);
        current = filtered;
        filtered_owned = true;
    }
    var min_filtered: mlx.mlx_array = .{ .ctx = null };
    defer if (min_filtered.ctx != null) {
        _ = mlx.mlx_array_free(min_filtered);
    };
    if ((sampling.min_p orelse 0) > 0) {
        min_filtered = mlx.mlx_array_new();
        try applyMinP(&min_filtered, current, sampling.min_p.?, s);
        current = min_filtered;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Logprobs for the FIRST generated token, taken from the prefill's
/// final-position logits — the one distribution the decode loop never sees.
/// `chosen` is the id the lazy sampler actually drew, so the reported
/// `token_logprob` belongs to the token that was really emitted (re-sampling
/// here would disagree with it at any temperature above 0).
fn firstTokenLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: u32, logprobs_n: u32, s: mlx.mlx_stream) !?LogprobResult {
    if (logprobs_n == 0) return null;
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return null;
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    const sq_shape = [_]c_int{ 1, shape[2] };
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&last, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, shape[1] - 1, 0 };
        const stop = [_]c_int{ 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_reshape(&last, sliced, &sq_shape, 2, s));
    }
    return try computeLogprobs(allocator, last, chosen, logprobs_n, s);
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
///
/// `logits` are the MODEL's logits for the position — before temperature,
/// penalties and top-k/top-p — so the reported distribution is the model's, as
/// OpenAI's is. Reading them after temperature made every value move with a
/// knob the client set (and at temp 0 the distribution SATURATES, so most
/// entries report exactly 0.0).
///
/// Ids travel WITH their values through `mlx_argpartition_axis` + a gather.
/// Recovering them afterwards by scanning the vocab for float equality — what
/// this did — is ambiguous the moment two logits tie, which under the
/// saturation above is everywhere: rank 1 was measured to be the chosen token
/// in 0 of 5 positions on a trivial greedy prompt.
/// Widest `top_logprobs` / `logprobs` a request may ask for. OpenAI stops at 20; a
/// fidelity probe (KL against a reference distribution) needs the pack's tail too.
pub const MAX_TOP_LOGPROBS: u32 = 1024;

fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // Every caller passes one position, as [1, V] or [1, 1, V].
    const shape = mlx.getShape(logits);
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_reshape(&row, logits, &[_]c_int{ 1, shape[shape.len - 1] }, 2, s));
    var one: std.ArrayList(LogprobResult) = .empty;
    defer one.deinit(allocator);
    try rowLogprobs(allocator, row, &.{chosen_token}, top_n, s, &one);
    return one.items[0];
}

/// argpartition leaves the winners UNORDERED; ties break on the lower id so
/// the ranking is deterministic run to run.
fn rankedBefore(_: void, a: TokenLogprob, b: TokenLogprob) bool {
    if (a.logprob != b.logprob) return a.logprob > b.logprob;
    return a.token_id < b.token_id;
}

/// Row r of `logits` ([rows, V]) scores `chosen[r]` and lists its `top_n` best; an id past
/// the vocab scores -100. log_softmax is a per-row shift, so ranks and gathers read the f32
/// logits and only the gathered values subtract the row's logsumexp. Every intermediate
/// handle is gone before the one eval, so the peak is the top-k argpartition: its f32
/// input, its indices and the sort's scratch, five [rows, V] 4-byte arrays (none at top_n 0).
fn rowLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: []const u32, top_n: u32, s: mlx.mlx_stream, out: *std.ArrayList(LogprobResult)) !void {
    const shape = mlx.getShape(logits);
    const rows: c_int = @intCast(chosen.len);
    const vocab: usize = @intCast(shape[shape.len - 1]);
    const k: usize = @min(@as(usize, @min(top_n, MAX_TOP_LOGPROBS)), vocab);
    const clamped = try allocator.alloc(u32, chosen.len);
    defer allocator.free(clamped);
    for (chosen, clamped) |c, *d| d.* = @min(c, @as(u32, @intCast(vocab - 1)));

    var chosen_lp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chosen_lp);
    var top_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_vals);
    var top_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_ids);
    {
        // f32 first: the logits dtype rounds bf16/f16 before the log, and f16 underflows to -inf.
        var logits32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits32);
        try mlx.check(mlx.mlx_astype(&logits32, logits, .float32, s));
        var lse = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lse);
        try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits32, -1, true, s));
        const ids = mlx.mlx_array_new_data(@ptrCast(clamped.ptr), &[_]c_int{ rows, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_along_axis(&picked, logits32, ids, -1, s));
        try mlx.check(mlx.mlx_subtract(&chosen_lp, picked, lse, s));
        if (k > 0) {
            // Ascending partition at V - k leaves the k largest in the tail.
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_argpartition_axis(&part, logits32, @intCast(vocab - k), -1, s));
            var idx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(idx);
            try mlx.check(mlx.mlx_slice(&idx, part, &[_]c_int{ 0, @intCast(vocab - k) }, 2, &[_]c_int{ rows, @intCast(vocab) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vals);
            try mlx.check(mlx.mlx_take_along_axis(&vals, logits32, idx, -1, s));
            try mlx.check(mlx.mlx_subtract(&top_vals, vals, lse, s));
            try mlx.check(mlx.mlx_astype(&top_ids, idx, .int32, s));
        }
    }
    const outs = [_]mlx.mlx_array{ chosen_lp, top_vals, top_ids };
    const vec = mlx.mlx_vector_array_new_data(&outs, if (k > 0) 3 else 1);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));

    const chosen_ptr = mlx.mlx_array_data_float32(chosen_lp) orelse return error.LogprobsUnreadable;
    const val_ptr: [*]const f32 = if (k > 0) mlx.mlx_array_data_float32(top_vals) orelse return error.LogprobsUnreadable else undefined;
    const id_ptr: [*]const i32 = if (k > 0) mlx.mlx_array_data_int32(top_ids) orelse return error.LogprobsUnreadable else undefined;
    try out.ensureUnusedCapacity(allocator, chosen.len);
    for (chosen, 0..) |id, r| {
        const top = try allocator.alloc(TokenLogprob, k);
        for (top, 0..) |*t, j| t.* = .{ .token_id = @intCast(id_ptr[r * k + j]), .logprob = val_ptr[r * k + j] };
        std.mem.sort(TokenLogprob, top, {}, rankedBefore);
        out.appendAssumeCapacity(.{ .token_logprob = if (id < vocab) chosen_ptr[r] else -100.0, .top_logprobs = top });
    }
}

/// Logprobs of a prompt's own tokens (`/v1/completions` echo): row r of `logits`
/// ([1, S, V]) scores `next_ids[r]`, the token after it.
pub fn promptLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, next_ids: []const u32, top_n: u32, s: mlx.mlx_stream, out: *std.ArrayList(LogprobResult)) !void {
    if (next_ids.len == 0) return;
    const shape = mlx.getShape(logits);
    if (shape.len != 3 or shape[1] < next_ids.len) return error.PromptLogprobsUnavailable;
    const rows: c_int = @intCast(next_ids.len);
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, logits, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, rows, shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, head, &[_]c_int{ rows, shape[2] }, 2, s));
    try rowLogprobs(allocator, flat, next_ids, top_n, s, out);
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

/// Rank cap for the filter helpers. The widest shape a caller passes is
/// `[B, S, V]`; anything past this is a named error, never a wrong axis.
const SAMPLER_MAX_NDIM = 4;

/// `a[..., start..stop]`, every leading axis passed through whole.
fn sliceLastAxis(res: *mlx.mlx_array, a: mlx.mlx_array, start: usize, stop: usize, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(a);
    if (shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    var lo: [SAMPLER_MAX_NDIM]c_int = undefined;
    var hi: [SAMPLER_MAX_NDIM]c_int = undefined;
    var st: [SAMPLER_MAX_NDIM]c_int = undefined;
    for (shape, 0..) |dim, i| {
        const last = i == shape.len - 1;
        lo[i] = if (last) @intCast(start) else 0;
        hi[i] = if (last) @intCast(stop) else dim;
        st[i] = 1;
    }
    const n = shape.len;
    try mlx.check(mlx.mlx_slice(res, a, &lo, n, &hi, n, &st, n, s));
}

/// Column ids of `a` ranked by value DESCENDING, ties by LOWEST ORIGINAL INDEX.
///
/// mlx's merge sort is stable — `ThreadSort` swaps only on a strict less-than and
/// the merge takes from B only when `b < a` — so an ascending argsort of the
/// negated row is exactly that order, and it is the `msv_qsa_select` tie rule.
/// Both filters and both of their routes rank through here, which is what makes a
/// shortlist cutoff agree with the whole row's on a tie-heavy row. Pinned by
/// `sampler ranks break ties by the lowest original index`.
fn ranksDescending(res: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, a, s));
    try mlx.check(mlx.mlx_argsort_axis(res, neg, -1, s));
}

/// Candidate-chunk width for `topRanksDescending`: cost is one sort of
/// `V / chunk` chunk maxima plus one of `m * chunk` candidates, balanced at
/// sqrt(V / m).
fn shortlistChunkWidth(v: usize, m: usize) usize {
    const bal = @sqrt(@as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(@max(m, 1))));
    const c: usize = @intFromFloat(@max(16.0, @min(2048.0, bal)));
    return @min(c, v);
}

/// The top `m` of every row — values and their column ids, both in
/// `ranksDescending` order — without ranking the whole row.
///
/// Every element of a chunk is at most that chunk's max, so a chunk whose max is
/// not among the `m` largest chunk maxima holds no top-`m` element: at most `m`
/// chunks can, and ranking their `m * chunk` candidates is exact. On Metal a
/// full-row `mlx_argsort_axis` is a multi-block merge sort over the whole
/// vocabulary — and so are `mlx_topk` and `mlx_argpartition`, which mlx routes to
/// that same sort, so neither of those buys a shortlist.
fn topRanksDescending(
    vals: *mlx.mlx_array,
    ids: *mlx.mlx_array,
    logits: mlx.mlx_array,
    m: u32,
    s: mlx.mlx_stream,
) !void {
    const shape = mlx.getShape(logits);
    if (shape.len == 0 or shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    const v: usize = @intCast(shape[shape.len - 1]);
    const want: usize = @min(@as(usize, m), v);
    if (want == 0) return error.EmptyShortlist;
    const chunk = shortlistChunkWidth(v, want);
    const groups = (v + chunk - 1) / chunk;

    // Nothing left to win once the candidate set is the row itself.
    if (want >= v or want * chunk >= v) {
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, logits, s);
        try sliceLastAxis(ids, order, 0, want, s);
        return mlx.check(mlx.mlx_take_along_axis(vals, logits, ids.*, -1, s));
    }

    // Pad the last chunk with -inf so the reshape is exact.
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    if (groups * chunk != v) {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var pad_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pad_val);
        try mlx.check(mlx.mlx_astype(&pad_val, neg_inf, mlx.mlx_array_dtype(logits), s));
        const axes = [_]c_int{@intCast(shape.len - 1)};
        const low = [_]c_int{0};
        const high = [_]c_int{@intCast(groups * chunk - v)};
        try mlx.check(mlx.mlx_pad(&padded, logits, &axes, 1, &low, 1, &high, 1, pad_val, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&padded, logits));
    }

    var dims: [SAMPLER_MAX_NDIM + 1]c_int = undefined;
    @memcpy(dims[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    const lead = shape.len - 1;
    dims[lead] = @intCast(groups);
    dims[lead + 1] = @intCast(chunk);
    var blk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk);
    try mlx.check(mlx.mlx_reshape(&blk, padded, &dims, lead + 2, s));

    var chunk_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_max);
    try mlx.check(mlx.mlx_max_axis(&chunk_max, blk, -1, false, s));

    // Chunks ranked by max, ties by lowest chunk id. A tie group's contributing
    // chunks are a PREFIX of its ascending-id list — a lower id holds a lower
    // column id, which outranks — so taking the lowest ids is what keeps the
    // shortlist equal to the whole row's ranking.
    var chunk_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_ranks);
    try ranksDescending(&chunk_ranks, chunk_max, s);
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try sliceLastAxis(&picked, chunk_ranks, 0, want, s);
    // Ascending chunk ids lay the candidates out in ascending COLUMN id, so the
    // stable rank below breaks candidate ties the way the whole row would.
    var win = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win);
    try mlx.check(mlx.mlx_sort_axis(&win, picked, -1, s));
    var win_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_i32);
    try mlx.check(mlx.mlx_astype(&win_i32, win, .int32, s));

    var win_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_col);
    try mlx.check(mlx.mlx_expand_dims(&win_col, win_i32, -1, s));
    dims[lead] = @intCast(want);
    var gather_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather_idx);
    try mlx.check(mlx.mlx_broadcast_to(&gather_idx, win_col, &dims, lead + 2, s));
    var cand_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_2d);
    try mlx.check(mlx.mlx_take_along_axis(&cand_2d, blk, gather_idx, @intCast(lead), s));

    // Column id of every candidate: chunk id * chunk + offset within the chunk.
    const chunk_arr = mlx.mlx_array_new_int(@intCast(chunk));
    defer _ = mlx.mlx_array_free(chunk_arr);
    var base = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(base);
    try mlx.check(mlx.mlx_multiply(&base, win_col, chunk_arr, s));
    var offsets = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(offsets);
    try mlx.check(mlx.mlx_arange(&offsets, 0, @floatFromInt(chunk), 1, .int32, s));
    var cand_ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_2d);
    try mlx.check(mlx.mlx_add(&cand_ids_2d, base, offsets, s));

    dims[lead] = @intCast(want * chunk);
    var cand_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_vals);
    try mlx.check(mlx.mlx_reshape(&cand_vals, cand_2d, &dims, lead + 1, s));
    var cand_ids_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_flat);
    try mlx.check(mlx.mlx_reshape(&cand_ids_flat, cand_ids_2d, &dims, lead + 1, s));
    // A padded tail column is -inf, so it only ranks when the row holds fewer
    // than `want` finite values; clamp so the scatter stays in range regardless.
    const last_col = mlx.mlx_array_new_int(@intCast(v - 1));
    defer _ = mlx.mlx_array_free(last_col);
    var cand_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids);
    try mlx.check(mlx.mlx_minimum(&cand_ids, cand_ids_flat, last_col, s));

    var cand_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ranks);
    try ranksDescending(&cand_ranks, cand_vals, s);
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try sliceLastAxis(&sel, cand_ranks, 0, want, s);
    try mlx.check(mlx.mlx_take_along_axis(vals, cand_vals, sel, -1, s));
    try mlx.check(mlx.mlx_take_along_axis(ids, cand_ids, sel, -1, s));
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Per-ROW top-k by RANK: the k highest-ranked column ids, scattered back as a
    // keep mask. A value cutoff (`logits >= kth largest`) keeps every token tied
    // with the k-th, and bf16 logits tie at the top constantly on a 250k vocab —
    // top_k 1 stopped being greedy.
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    if (k == 0 or k >= v) return mlx.check(mlx.mlx_array_set(res, logits));
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try topRanksDescending(&vals, &ids, logits, k, s);
    try keepByIndex(res, logits, ids, s);
}

/// Mask `logits` to -inf everywhere except the per-row `keep_idx` columns.
fn keepByIndex(res: *mlx.mlx_array, logits: mlx.mlx_array, keep_idx: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    const t = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(t);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, keep_idx, t, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Top-k and top-p in ONE pass. A bounded `applyTopP` ranks the row's top
/// `top_k` columns and scatters its keep flags at exactly those column ids, so a
/// preceding `applyTopK` selects the same shortlist and masks a SUPERSET of what
/// the nucleus keeps — one full-vocab selection and one full-row `where` of pure
/// tax on every filtered row. Byte-identical to the pair either way round.
fn filterTopKTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    if (top_p >= 1.0) return applyTopK(res, logits, top_k, s);
    return applyTopP(res, logits, top_p, top_k, s);
}

fn applyMinP(res: *mlx.mlx_array, logits: mlx.mlx_array, min_p: f32, s: mlx.mlx_stream) !void {
    var max_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(max_logits);
    try mlx.check(mlx.mlx_max_axis(&max_logits, logits, -1, true, s));
    const log_min_p = mlx.mlx_array_new_float(@log(min_p));
    defer _ = mlx.mlx_array_free(log_min_p);
    var threshold = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(threshold);
    try mlx.check(mlx.mlx_add(&threshold, max_logits, log_min_p, s));
    var keep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep);
    try mlx.check(mlx.mlx_greater_equal(&keep, logits, threshold, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, keep, logits, neg_inf, s));
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability
/// mass. The nucleus is decided over ranked values and scattered back by column
/// id, so a token tied with the cutoff value but outside the mass is masked
/// (top_p -> 0 is greedy even on tied bf16 logits).
///
/// `nucleus_bound` is the caller's top-k. With top-k applied first the row holds
/// exactly `k` finite columns, so the nucleus cannot reach past the k-shortlist
/// and the cutoff is exact without ranking the row; 0 means the row is unfiltered
/// and the nucleus is unbounded.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, nucleus_bound: u32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    const bounded = nucleus_bound > 0 and nucleus_bound < v;

    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    if (bounded) {
        try topRanksDescending(&vals, &ids, logits, nucleus_bound, s);
    } else {
        try ranksDescending(&ids, logits, s);
        try mlx.check(mlx.mlx_take_along_axis(&vals, logits, ids, -1, s));
    }

    // f32 throughout: mlx instantiates cumsum at the INPUT dtype and a bf16
    // accumulator loses nucleus tail mass. The ranked values ARE the row's
    // finite entries (top-k left exactly k), so their softmax is the row's.
    var vals_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_f32);
    try mlx.check(mlx.mlx_astype(&vals_f32, vals, .float32, s));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, vals_f32, -1, true, s));

    // Mass STRICTLY above each rank — an exclusive scan down the ranking, so the
    // term is a function of the shortlist alone (everything outranking a top-k
    // column is itself top-k). Rank 0 sees 0, so the argmax is always kept —
    // the threshold floors above 0 so a literal top_p 0 is greedy, not empty.
    var above = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(above);
    try mlx.check(mlx.mlx_cumsum(&above, probs, -1, false, false, s));
    const threshold = mlx.mlx_array_new_float(@max(top_p, std.math.floatMin(f32)));
    defer _ = mlx.mlx_array_free(threshold);
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_less(&in_nucleus, above, threshold, s));

    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, ids, in_nucleus, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Penalties over the tokens generated so far: `repeat_penalty` divides a positive logit and
/// multiplies a negative one; `presence_penalty` is subtracted once per seen id and
/// `frequency_penalty` once per occurrence (OpenAI's definition).
fn applyPenalties(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, sampling: SamplingParams, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };

    // Occurrences per id; reused below for what each id loses.
    const table = try std.heap.page_allocator.alloc(f32, vocab_size);
    defer std.heap.page_allocator.free(table);
    @memset(table, 0);
    for (generated_ids) |id| if (id < vocab_size) {
        table[id] += 1;
    };

    var current = logits;
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (sampling.repeat_penalty != 1.0) {
        const counts = mlx.mlx_array_new_data(table.ptr, &arr_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(counts);
        const rp = mlx.mlx_array_new_float(sampling.repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / sampling.repeat_penalty);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        var seen = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(seen);
        try mlx.check(mlx.mlx_greater(&seen, counts, zero, s));
        var positive = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive);
        try mlx.check(mlx.mlx_greater(&positive, logits, zero, s));
        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, logits, inv_rp, s));
        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, logits, rp, s));
        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive, pen_pos, pen_neg, s));
        try mlx.check(mlx.mlx_where(&scaled, seen, sign_selected, logits, s));
        current = scaled;
    }

    if (sampling.presence_penalty == 0.0 and sampling.frequency_penalty == 0.0) {
        try mlx.check(mlx.mlx_copy(res, current, s));
        return;
    }
    for (table) |*v| if (v.* > 0) {
        v.* = sampling.presence_penalty + sampling.frequency_penalty * v.*;
    };
    const subtract = mlx.mlx_array_new_data(table.ptr, &arr_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(subtract);
    try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── Tests ──

const testing = std.testing;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "frequency_penalty subtracts per occurrence, presence_penalty once, and neither scales a logit" {
    const s = mlx.gpuStream();
    var data = [_]f32{ 2, 2, 2, 2, 2, -2, 2, 2 };
    const logits = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, 8 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try applyPenalties(&out, logits, &.{ 3, 3, 5 }, .{ .frequency_penalty = 0.5, .presence_penalty = 0.25 }, s);
    try mlx.check(mlx.mlx_array_eval(out));
    const got = mlx.mlx_array_data_float32(out).?[0..8];
    try testing.expectEqualSlices(f32, &.{ 2, 2, 2, 0.75, 2, -2.75, 2, 2 }, got);
    try testing.expect((SamplingParams{ .frequency_penalty = 0.5 }).penalized());
}

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
}

test "tokensPerSec basic and zero-time" {
    // 100 tokens in 1 second = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(100, std.time.ns_per_s), 1e-6);
    // 50 tokens in 0.5s = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(50, std.time.ns_per_s / 2), 1e-6);
    // Zero elapsed → 0, never inf/NaN.
    try testing.expectEqual(@as(f64, 0.0), tokensPerSec(100, 0));
}

test "prefillTokensPerSec divides by uncached tokens" {
    // Cold prefill: 754 tokens, none cached, 2s → 377 tok/s.
    try testing.expectApproxEqAbs(
        @as(f64, 377.0),
        prefillTokensPerSec(754, 0, 2 * std.time.ns_per_s),
        1e-6,
    );
    // Warm prefill: 754-token prompt, 700 cached, only 54 ran. A fast 54-token
    // suffix in 0.1s is 540 tok/s — NOT 7540 (the inflated full-prompt rate).
    try testing.expectApproxEqAbs(
        @as(f64, 540.0),
        prefillTokensPerSec(754, 700, std.time.ns_per_s / 10),
        1e-6,
    );
    // Full cache hit: 0 uncached → 0 tok/s (no compute happened).
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(754, 754, std.time.ns_per_s));
    // Defensive: cached > prompt (shouldn't happen) clamps to 0, no underflow.
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(10, 20, std.time.ns_per_s));
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyPenalties reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat_penalty = 2.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyPenalties with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat_penalty = 2.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyPenalties presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .presence_penalty = 0.5 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "applyPenalties combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat_penalty = 2.0, .presence_penalty = 1.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "min_p keeps only eligible tokens in scalar, lazy and batched sampling" {
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2, 0, -5, -5, 0, 2, 3 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 2, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const strict = SamplingParams{ .temperature = 1, .top_k = 2, .min_p = 0.5 };
    const probs = try probsAllPositions(logits, strict, s);
    defer _ = mlx.mlx_array_free(probs);
    const values = try samplerTestReadFlat(testing.allocator, probs, raw.len, s);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 0, 0, 0, 1 }, values);

    const first = mlx.mlx_array_new_data(&raw[0], &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(first);
    const second = mlx.mlx_array_new_data(&raw[4], &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(second);
    try testing.expectEqual(@as(u32, 0), (try sampleToken(testing.allocator, first, strict, null, 0, s)).token_id);
    try testing.expectEqual(@as(u32, 0), try evalLazyToken(sampleTokenLazy(first, strict, s)));
    var params = [_]SamplingParams{ strict, strict };
    var ids: [2]i32 = undefined;
    try sampleRows(&ids, &.{ first, second }, &params, s);
    try testing.expectEqualSlices(i32, &.{ 0, 3 }, &ids);
    try testing.expect(!sampleRowsHomogeneous(&.{ strict, .{ .temperature = 1, .top_k = 2, .min_p = 0 } }));
    try testing.expect(sampleRowsHomogeneous(&.{ SamplingParams{ .temperature = 1, .top_k = 2 }, .{ .temperature = 1, .top_k = 2, .min_p = 0 } }));
}

test "min_p keeps tied maxima and zero matches the omitted filter" {
    const s = mlx.gpuStream();
    const tied = mlx.mlx_array_new_data(&[_]f32{ 0, 0, -1, -2 }, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(tied);
    const strict = try probsAllPositions(tied, .{ .min_p = 1 }, s);
    defer _ = mlx.mlx_array_free(strict);
    const values = try samplerTestReadFlat(testing.allocator, strict, 4, s);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0, 0 }, values);

    const omitted = try probsAllPositions(tied, .{}, s);
    defer _ = mlx.mlx_array_free(omitted);
    const zero = try probsAllPositions(tied, .{ .min_p = 0 }, s);
    defer _ = mlx.mlx_array_free(zero);
    try samplerTestExpectSameBits(omitted, zero, 4, s);
}

test "min_p uses temperature and agrees across speculative target and proposal" {
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2.8, 0, -5, -5, 0, 2.8, 3 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 2, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const warm = try probsAllPositions(logits, .{ .temperature = 2, .min_p = 0.9 }, s);
    defer _ = mlx.mlx_array_free(warm);
    const warm_values = try samplerTestReadFlat(testing.allocator, warm, raw.len, s);
    defer testing.allocator.free(warm_values);
    try testing.expect(warm_values[0] > 0 and warm_values[1] > 0);
    try testing.expectEqual(@as(f32, 0), warm_values[2]);
    const cold = try probsAllPositions(logits, .{ .temperature = 1, .min_p = 0.9 }, s);
    defer _ = mlx.mlx_array_free(cold);
    const cold_values = try samplerTestReadFlat(testing.allocator, cold, raw.len, s);
    defer testing.allocator.free(cold_values);
    try testing.expectEqual(@as(f32, 0), cold_values[1]);

    const policy = SamplingParams{ .temperature = 2, .top_k = 3, .top_p = 0.9, .min_p = 0.5 };
    const target = try probsAllPositions(logits, policy, s);
    defer _ = mlx.mlx_array_free(target);
    const proposal = try filteredProbsBlock(logits, policy, s);
    defer _ = mlx.mlx_array_free(proposal);
    try samplerTestExpectSameBits(target, proposal, raw.len, s);
}

test "seeded lazy sampling replays the same draws and advances per draw" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 0.0, 0.0 };
    const shape = [_]c_int{ 1, 1, 2 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var runs: [2][24]i32 = undefined;
    for (&runs) |*run| {
        var params = SamplingParams{ .temperature = 1.0, .seed = 42 };
        for (run) |*out| {
            const lazy = sampleTokenLazy(logits, params, s);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            try mlx.check(mlx.mlx_array_item_int32(out, lazy));
            params.draw +%= 1;
        }
    }
    try testing.expectEqualSlices(i32, &runs[0], &runs[1]);
    // The same key every draw replays ONE coin flip 24 times (p = 2^-23).
    var all_same = true;
    for (runs[0][1..]) |v| all_same = all_same and v == runs[0][0];
    try testing.expect(!all_same);
}

test "batched greedy ids equal two solo ids and one sampling eval" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, greedy, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, greedy };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "batched mixed greedy and sampled rows produce two ids" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const sampled = SamplingParams{ .temperature = 0.7 };
    const solo_g = try evalLazyToken(sampleTokenLazy(r0, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, sampled };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo_g)), ids[0]);
    try testing.expect(ids[1] >= 0 and ids[1] < 4);
}

test "batched stacked seedless top_k top_p rows equal solo draws" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "top-k and top-p break exact ties by rank, not by value" {
    // bf16 logits tie at the top constantly on a 250k vocab; a value cutoff
    // keeps every tied token and top_k 1 / top_p -> 0 stop being greedy.
    const s = mlx.gpuStream();
    const data = [_]f32{
        2.0, 5.0, 5.0, 5.0, 1.0,
        0.1, 0.2, 9.0, 0.3, 0.4,
    };
    const shape = [_]c_int{ 2, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for ([_]bool{ true, false }) |use_k| {
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        if (use_k) try applyTopK(&masked, logits, 1, s) else try applyTopP(&masked, logits, 1e-6, 0, s);
        try mlx.check(mlx.mlx_array_eval(masked));
        const out = mlx.mlx_array_data_float32(masked).?[0..10];
        var kept0: usize = 0;
        for (out[0..5]) |v| kept0 += @intFromBool(v != -std.math.inf(f32));
        try testing.expectEqual(@as(usize, 1), kept0);
        for (out[5..10], 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i == 2));
    }
    // A 3-of-5 top-k on the tied row keeps exactly the three tied tokens.
    var masked3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked3);
    try applyTopK(&masked3, logits, 3, s);
    try mlx.check(mlx.mlx_array_eval(masked3));
    const out3 = mlx.mlx_array_data_float32(masked3).?[0..5];
    for (out3, 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i >= 1 and i <= 3));
}

test "batched stacked seedless rows honour a suppress mask" {
    const s = mlx.gpuStream();
    const mask = try buildSuppressMask(&[_]u32{1}, 0, 4, s);
    defer _ = mlx.mlx_array_free(mask);
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 9.0, 0.5, 0.6,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9, .suppress_mask = mask };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expect(ids[0] != 1);
    try testing.expect(ids[1] != 1);
}

test "batched seeded draws keep per-row keys" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p0 = SamplingParams{ .temperature = 1.0, .seed = 7, .draw = 2 };
    const p1 = SamplingParams{ .temperature = 1.0, .seed = 99, .draw = 5 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p0, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p1, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p0, p1 };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expectEqual(p0.draw +% 1, params[0].draw);
    try testing.expectEqual(p1.draw +% 1, params[1].draw);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}

test "Generator.runtimeGateShouldDisable below warmup never trips" {
    // Even with zero accepts, before the warmup count we trust the prompt-time
    // gate and never disable speculation mid-decode. drafts_per_round is the
    // typical drafter setting (block_size=4 → 3 drafts per round).
    try testing.expect(!Generator.runtimeGateShouldDisable(0, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(1, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP - 1, 0, 3));
}

test "Generator.runtimeGateShouldDisable trips at warmup with low per-draft rate" {
    // Synthetic low-accept scenario: 5 verify attempts, drafts_per_round=3
    // (drafter at block_size=4). 5 attempts × 3 = 15 drafts proposed.
    // 0 accepted → 0.00 < 0.50 → trip. Same with 1 accepted (0.067).
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 0, 3));
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 1, 3));
    // 7 accepted out of 15 = 0.467 — still below 0.50 → trip.
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 7, 3));
    // 8 accepted out of 15 = 0.533 → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 8, 3));
}

test "Generator.runtimeGateShouldDisable does not trip with high per-draft rate" {
    // Echo workloads on Gemma drafter: ~93% per-draft acceptance (E4B from
    // bench: 67/(24*3) = 93.1%). Well above threshold → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(24, 67, 3));
    // PLD heavy-echo: ~4 of 5 drafts accepted per attempt = 0.80 per-draft.
    try testing.expect(!Generator.runtimeGateShouldDisable(20, 80, 5));
    // Edge case at exactly the threshold (rate == 0.50) — strict less-than,
    // so does NOT trip.
    try testing.expect(!Generator.runtimeGateShouldDisable(10, 15, 3)); // 15/30 = 0.50
}

test "Generator.runtimeGateShouldDisable creative-content regression scenario" {
    // The Phase 1 bench's exact regression cases on creative prompts:
    //   E4B drafter (bs=4 → drafts_per_round=3): 39/59 attempts → 22.0% per-draft → trip
    //   E2B drafter (bs=2 → drafts_per_round=1): 31/66 attempts → 47.0% per-draft → trip
    //   31B drafter (bs=8 → drafts_per_round=7): 60/(38*7) → 22.6% per-draft → trip
    try testing.expect(Generator.runtimeGateShouldDisable(59, 39, 3)); // E4B creative
    try testing.expect(Generator.runtimeGateShouldDisable(66, 31, 1)); // E2B creative
    try testing.expect(Generator.runtimeGateShouldDisable(38, 60, 7)); // 31B creative
    // The 26B-A4B@bs=2 creative case: 37/(60*1) = 61.7% → above threshold,
    // so the runtime gate alone does NOT save it. MoE regressions need the
    // separate `default_enable_drafter` opt-out at startup.
    try testing.expect(!Generator.runtimeGateShouldDisable(60, 37, 1));
}

test "Generator.runtimeGateShouldDisable handles drafts_per_round=0" {
    // Defensive: if a caller somehow passes a degenerate config (block_size=1
    // → drafts_per_round=0), don't divide by zero. We return false (no trip).
    try testing.expect(!Generator.runtimeGateShouldDisable(100, 0, 0));
}

test "Generator.dflashGateShouldDisable uses accepted yield across block widths" {
    // Never decide from fewer than the DFlash warmup number of paid verifies.
    try testing.expect(!Generator.dflashGateShouldDisable(
        Generator.DFLASH_GATE_WARMUP - 1,
        0,
        Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    ));

    // Prose/vision measured 1.0-1.5 accepted drafts per round at both
    // block 8 and block 16: that class loses to serial and must fall back.
    const w = Generator.DFLASH_GATE_WARMUP;
    try testing.expect(Generator.dflashGateShouldDisable(w, w * 1, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(w, w * 2 - 1, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Exactly two is the strict break-even boundary; code/tool traffic at
    // 4.4+ and echo traffic near a full block remain on DFlash.
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 2, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 5, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 15, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Thinking preambles recovered from ~1.4 early to 4.4 whole-request; the
    // resolved-mode threshold keeps that path alive while still cutting off
    // a truly non-yielding reasoning request.
    try testing.expect(!Generator.dflashGateShouldDisable(w, w + w / 3, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(w, w - w / 3, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
}

test "Generator.yieldGateShouldDisable trips on cold-path-dominated workloads" {
    // The 2026-06-10 baseline regression: PLD forced on for a creative essay
    // prompt where the n-gram lookup almost never matches. The per-draft gate
    // never trips (it only counts verify ROUNDS, and there are few), but every
    // step pays the unpipelined cold forward → −14% measured on E2B. The
    // yield gate counts ALL enabled-mode steps: accepted-drafted-tokens per
    // step below the threshold after warmup → disable.
    // Creative: 128 steps, ~6 drafted tokens accepted → yield 0.047 → trip.
    try testing.expect(Generator.yieldGateShouldDisable(128, 6));
    // Heavy echo: 40 steps, 80 accepted (2.0/step) → stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(40, 80));
    // Inside warmup: never trip, even at zero yield.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP - 1, 0));
    // Exactly at warmup with healthy yield: stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP, Generator.YIELD_GATE_WARMUP));
}

test "yield-gate warmup is 8 — the cold-path tax tripled when the AR step got 3x cheaper" {
    // This constant is pure economics and the economics MOVED. Every warmup
    // step pays PLD's unpipelined cold forward plus a synchronous host read of
    // the sampled token, against `next()`'s async-pipelined step — an
    // ~absolute tax measured as a share of the AR step. Fixing the
    // mscale promotion made that AR step ~3x cheaper, so the same tax became a
    // ~3x larger share and 32 steps of it stopped being affordable.
    //
    // Swept on one model (one boot, serial vs unconstrained alternating per
    // request, server-reported timings, 5 runs median), against a matrix that
    // deliberately contains PLD's WIN case as well as its loss cases —
    // tuning on loss cases alone drives the warmup to zero and throws the win
    // away:
    //
    //   warmup   echo-edit   code-edit  free-form   explain      qa
    //       32     +76.9%       -4.3%      -1.2%     -1.4%     -7.2%
    //       16     +70.1%       -3.4%      -0.4%     -0.3%     -0.8%
    //        8     +77.4%       +0.3%      -0.2%     -0.3%     -1.5%
    //        4     +76.6%       +3.9%      -0.2%     -0.1%     -1.7%
    //
    // 8 recovers essentially the whole loss while leaving the +77% untouched.
    // 4 measured no worse, but a gate that decides on four observations is
    // fitting noise, and a short preamble before a file echo is the NORMAL
    // agent shape — exactly what a too-eager trip would punish. A premature
    // trip is bounded anyway: `specShouldReenable` re-checks every
    // SPEC_REENABLE_INTERVAL steps.
    try testing.expectEqual(@as(u64, 8), Generator.YIELD_GATE_WARMUP);

    // Sweepable without a rebuild, same contract as `runtimeGateWarmup`:
    // out-of-range values fall back rather than silently disabling the gate.
    try testing.expectEqual(Generator.YIELD_GATE_WARMUP, Generator.yieldGateWarmup());
}

test "Generator.specShouldReenable gates mid-request PLD re-activation" {
    // Disabled-mode periodic check on the COMMITTED sequence (prompt +
    // generated). The decisive case: the model echoes PROMPT content (file
    // edit / tool result) after a novel preamble tripped the yield gate. The
    // echoed tail never repeats ITSELF, so self-repetition scoring misses it;
    // tailMatchFraction sees the prompt occurrence.
    var committed: [96]u32 = undefined;
    // prompt = 48-token "file", generated = 16 novel preamble + 32 echo of the file
    for (committed[0..48], 0..) |*t, i| t.* = @intCast(i + 100);
    for (committed[48..64], 0..) |*t, i| t.* = @intCast(i + 9000);
    for (committed[64..96], 0..) |*t, i| t.* = @intCast(i + 100);
    try testing.expect(Generator.specShouldReenable(&committed, 48));

    // Fully novel committed sequence → stay disabled.
    var novel: [96]u32 = undefined;
    for (&novel, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    try testing.expect(!Generator.specShouldReenable(&novel, 48));

    // Too little generated yet → not enough signal, stay disabled.
    try testing.expect(!Generator.specShouldReenable(&committed, 8));
}

test "InitOptions.lookup_prompt overrides prompt_ids_owned source" {
    // When the server's cache-reuse path forwards only a trailing-tail
    // prompt slice but supplies the full original prompt via
    // `InitOptions.lookup_prompt`, PLD's n-gram buffer must be cloned from
    // the full slice — not the truncated tail.
    const tail = [_]u32{99};
    const full = [_]u32{ 10, 20, 30, 99 };
    const src = Generator.pickLookupPromptSource(&tail, &full);
    try testing.expectEqual(@as(usize, 4), src.len);
    try testing.expectEqualSlices(u32, &full, src);
}

test "InitOptions.lookup_prompt = null preserves existing behavior" {
    // Back-compat path: when callers don't set `lookup_prompt`, the source
    // is the unmodified `prompt_ids` slice — same buffer the Generator
    // received pre-fix.
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const src = Generator.pickLookupPromptSource(&prompt, null);
    try testing.expectEqual(prompt.len, src.len);
    try testing.expectEqualSlices(u32, &prompt, src);
    try testing.expectEqual(@as([*]const u32, prompt[0..].ptr), src.ptr);
}

test "StallClock: progress resets the deadline, silence expires it, 0 disables" {
    var clock = StallClock{};
    const s = std.time.ns_per_s;
    // Producing tokens keeps resetting the deadline — a healthy generation
    // can run arbitrarily long (the live bug: a 33KB tool call at 30 tok/s
    // takes >300s and was guillotined mid-call by the wall-clock timeout).
    try std.testing.expect(!clock.expired(0 * s, 0, 300 * s));
    try std.testing.expect(!clock.expired(299 * s, 1000, 300 * s)); // progress at 299s
    try std.testing.expect(!clock.expired(598 * s, 2000, 300 * s)); // progress again
    // No new tokens for the full window -> stalled.
    try std.testing.expect(!clock.expired(700 * s, 2000, 300 * s));
    try std.testing.expect(clock.expired(898 * s, 2000, 300 * s));
    // 0 = disabled, even after silence.
    var off = StallClock{};
    try std.testing.expect(!off.expired(0, 0, 0));
    try std.testing.expect(!off.expired(10_000 * s, 0, 0));
}

test "isDegenerateTailLoop catches a repeated channel-opener cycle" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Gemma 4 12B failure mode: the model spams the thinking opener
    // `<|channel>thought\n` — model that as a 3-token cycle. After enough
    // identical repetitions the tail is a pure period-3 loop → fire.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 }); // some real prefix
        var k: usize = 0;
        while (k < degenerate_loop_min_span / 3 + 1) : (k += 1) {
            try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 }); // <|channel>,thought,\n
        }
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }

    // A single token stuck on repeat (period 1) counts once it fills the span.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        while (k < degenerate_loop_min_span + 1) : (k += 1) try ids.append(testing.allocator, 42);
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }
}

test "ToolForce: the opener after the thought closes, the closer too at the deadline, never twice" {
    const CLOSE: u32 = 11;
    const forced = [_]u32{ CLOSE, 20, 40, 41 }; // closer, separator, opener
    var tf = ToolForce{ .closer_id = CLOSE, .forced = &forced, .opener_at = 2, .in_think = true };
    try testing.expect(tf.due(&[_]u32{ 6, 7 }, 2, 100) == null);
    const after = tf.due(&[_]u32{ 6, 7, CLOSE }, 3, 100).?;
    try testing.expectEqualSlices(u32, &[_]u32{ 40, 41 }, after.tokens);
    try testing.expect(!after.closes_thought);
    tf.fired = true;
    try testing.expect(tf.due(&[_]u32{ 6, 7, CLOSE, 40 }, 4, 100) == null);

    // Still thinking with a quarter of max_tokens left: close it and call.
    var late = ToolForce{ .closer_id = CLOSE, .forced = &forced, .opener_at = 2, .in_think = true };
    try testing.expect(late.due(&[_]u32{6}, 74, 100) == null);
    const closing = late.due(&[_]u32{6}, 75, 100).?;
    try testing.expectEqualSlices(u32, &forced, closing.tokens);
    try testing.expect(closing.closes_thought);

    var open = ToolForce{ .closer_id = null, .forced = forced[2..], .opener_at = 0, .in_think = false };
    try testing.expectEqualSlices(u32, &[_]u32{ 40, 41 }, open.due(&[_]u32{}, 0, 100).?.tokens);
}

test "ThinkBound: counts only tokens inside the think block and fires at the budget" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = false };

    // Content before any opener never counts.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5 });
    try testing.expect(!tb.due());

    // Opened by the model: the budget counts from the opener.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7 });
    try testing.expect(!tb.due());
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7, 8 });
    try testing.expect(tb.due());

    // A model that closed on its own is never forced.
    var closed = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    closed.observe(&[_]u32{ 6, 7, CLOSE, 8, 9, 10, 11 });
    try testing.expect(!closed.due());

    // Prompt-opened: the count starts at token 0 and the forced closer ends it.
    var po = ThinkBound{ .budget = 2, .opener_id = null, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    po.observe(&[_]u32{ 6, 7 });
    try testing.expect(po.due());
    po.fired = true;
    po.observe(&[_]u32{ 6, 7, 30, 31, CLOSE, 32, 40 });
    try testing.expect(!po.in_think);
    try testing.expect(!po.due());
}

test "degenerateTail: a short exact cycle convicts only past the minimum span" {
    const al = testing.allocator;
    // Live 2026-09-15 (pi, a per-digit tokenizer): a 24-wide map wall
    // row "111111111111111111111111" is 24 identical tokens and was cut as a
    // period-1 loop mid-thought. Short cycles are common in honest code (digit
    // rows, zeroed arrays), so the bar is a SPAN of identical cycling, not a
    // rep count that period 1 reaches in a few dozen bytes.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 1 });
    for (0..24) |_| try ids.append(al, 16);
    try testing.expect(degenerateTail(ids.items) == null);

    // A 32-element zeroed row: `0, 0, 0, ...` is a period-2 cycle of 64 tokens.
    var zeros = std.ArrayList(u32).empty;
    defer zeros.deinit(al);
    try zeros.appendSlice(al, &[_]u32{ 7, 8, 9, 1 });
    for (0..32) |_| try zeros.appendSlice(al, &[_]u32{ 15, 11 });
    try testing.expect(degenerateTail(zeros.items) == null);

    // Stuck for real: one token past the span is a loop and trims to one copy.
    for (24..degenerate_loop_min_span) |_| try ids.append(al, 16);
    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    try testing.expectEqual(@as(usize, 5), d.start);
}

/// Build `n` tokens by cycling through `phrasings`, which share a vocabulary.
fn tVaried(al: std.mem.Allocator, phrasings: []const []const u32, n: usize) !std.ArrayList(u32) {
    var ids = std.ArrayList(u32).empty;
    var i: usize = 0;
    while (ids.items.len < n) : (i += 1) {
        try ids.appendSlice(al, phrasings[i % phrasings.len]);
    }
    return ids;
}

test "isNearRepeatTailLoop catches a VARIED-phrasing restatement loop" {
    // Live 2026-08-04, under pi: the model restated the same
    // intent forever while varying the wording — "I need to break this down."
    // / "I need to break this." / "I need to break this down into pieces." —
    // so no exact cycle exists at ANY period and both exact tiers are blind by
    // construction. What IS invariant is that a long stretch of output recycles
    // a tiny vocabulary and introduces almost no new n-grams.
    const al = testing.allocator;
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 }, // I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 46 }, // I need to break this .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 }, // ... down into pieces .
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 }, // So I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 }, // ... down first .
        &[_]u32{ 51, 42, 43, 44, 45, 46 }, // Let me break this down .
    };
    var ids = try tVaried(al, &phrasings, near_repeat_window + 64);
    defer ids.deinit(al);
    try testing.expect(isNearRepeatTailLoop(ids.items));

    // Below the window the tier says nothing at all: it is a last-resort net
    // for output that has already run a long way, and a false cut truncates a
    // real answer. Short exact cycles remain tier 1/2's job.
    try testing.expect(!isNearRepeatTailLoop(ids.items[0 .. near_repeat_window - 1]));
}

test "isNearRepeatTailLoop leaves legitimately repetitive output alone" {
    const al = testing.allocator;

    // Healthy prose/code: a repeated scaffold, but every line introduces a new
    // identifier. Recycled STRUCTURE is normal; recycled VOCABULARY is not.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var line: u32 = 0;
        while (ids.items.len < near_repeat_window + 64) : (line += 1) {
            try ids.appendSlice(al, &[_]u32{ 10, 11, 12 }); // `const x =`
            try ids.append(al, 1000 + line); // a fresh identifier
            try ids.appendSlice(al, &[_]u32{ 13, 14 }); // `;\n`
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // A numeric table: FEW distinct tokens (digits + separators, and this
    // family pre-tokenizes digits singly) but the 4-grams keep changing. The
    // two ratios have to be read together — either one alone convicts this.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 12345;
        while (ids.items.len < near_repeat_window + 64) {
            try ids.append(al, 200); // '|'
            for (0..4) |_| {
                seed = seed *% 1664525 +% 1013904223;
                try ids.append(al, 100 + (seed >> 16) % 10); // a digit
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201 }); // '|', '\n'
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // Fully novel output.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        for (0..near_repeat_window + 64) |i| try ids.append(al, @intCast(i));
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }
}

test "degenerateTail acquits a low-entropy STRUCTURED file that ends inside the span bar" {
    // Live 2026-09-15 under pi: a tool call rewriting five 24x19 tile maps of
    // '0'/'1' on a per-digit tokenizer was cut as a near-repeat loop at window
    // fill and the agent got an empty turn. Six distinct tokens and sixteen
    // possible 4-grams satisfy every content ratio by construction, and the
    // real 27B output (rows nearly all `100000000000000000000001`) is
    // indistinguishable from a loop by content. What such a file does, and a
    // loop never does, is END: the bar is the degenerate SPAN.
    const al = testing.allocator;
    const Shape = enum { bit_grid, lazy_map, hex_dump };
    for ([_]Shape{ .bit_grid, .lazy_map, .hex_dump }) |shape| {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 777;
        var row: usize = 0;
        while (ids.items.len < 2800) : (row += 1) {
            try ids.append(al, 200); // '"'
            for (0..24) |col| {
                seed = seed *% 1664525 +% 1013904223;
                const r = seed >> 16;
                const tok: u32 = switch (shape) {
                    .bit_grid => if (col == 0 or col == 23 or row % 19 == 0) 101 else @as(u32, if (r % 10 < 3) 101 else 100),
                    .lazy_map => if (col == 0 or col == 23 or row % 19 == 0 or (row % 5 == 0 and col != 10)) 101 else 100,
                    .hex_dump => 100 + r % 16,
                };
                try ids.append(al, tok);
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201, 202 }); // '"', ',', '\n'
        }
        try testing.expect(degenerateTail(ids.items) == null);
    }
}

test "isDegenerateTailLoop does not fire on healthy or briefly-repeating output" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Strictly increasing ids — no cycle at all.
    {
        var ids: [200]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        try testing.expect(!isDegenerateTailLoop(&ids, P, R));
    }
    // A short burst of repetition (well under R reps) must be left alone — a
    // model legitimately writing "ha ha ha" or a few identical list bullets.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 1, 2, 3, 4, 5 });
        var k: usize = 0;
        while (k < R - 1) : (k += 1) try ids.appendSlice(testing.allocator, &[_]u32{ 50, 51 });
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Periodic tail but with a longer period than we scan for → ignored.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        var base: u32 = 0;
        while (k < R) : (k += 1) {
            // period = P + 3 (> max_period); never a pure short cycle.
            var j: u32 = 0;
            while (j < P + 3) : (j += 1) try ids.append(testing.allocator, base + j);
            base = 0; // same long block repeats, but its period exceeds the scan window
        }
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Too few tokens to judge.
    try testing.expect(!isDegenerateTailLoop(&[_]u32{ 1, 1 }, P, R));
}

test "isNearRepeatTailLoop leaves PROCEDURAL code alone — it recycles a vocabulary while still progressing" {
    // Live 2026-08-05: a pi session was asked for an elaborate voxel scene and
    // the tier cut it at 16241 generated tokens, so the user got NO file at
    // all. The output was healthy — dense `fillBox(x,y,z, x,y,z, C.name);`
    // lines — but it is exactly the shape the first two ratios were built to
    // tolerate and cannot: a fixed template plus a small colour palette gives
    // a tiny distinct-token ratio, and the templated call shape keeps the
    // 4-gram ratio low too. Measured on the real artifact: 0.068 / 0.351
    // against bars of 0.12 / 0.35 — it cleared conviction by 0.001, and the
    // generation's own tail did not.
    //
    // What separates it from a loop is PROGRESS: every line carries new
    // coordinates, so the window's second half keeps introducing n-grams the
    // first half never had (0.298-0.632 measured, against 0.019-0.022 for the
    // restatement loops this tier exists for).
    // Shape taken from the measured artifact, not invented: a CONTIGUOUS run
    // of template tokens (`\n  fillBox(`, `, C.`, `);`) followed by the
    // varying coordinates. The contiguity is what makes it convict — most
    // 4-gram windows sit entirely inside the fixed run and repeat every line.
    // This fixture scores 0.033 / 0.316 against bars of 0.12 / 0.35.
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    var rng: u32 = 99;
    while (ids.items.len < near_repeat_window * 2) {
        var f: u32 = 0;
        while (f < 14) : (f += 1) try ids.append(al, 500 + f);
        var v: usize = 0;
        while (v < 4) : (v += 1) {
            rng = rng *% 1664525 +% 1013904223;
            try ids.append(al, 600 + (rng >> 16) % 19); // a fresh coordinate
            try ids.append(al, 499); // separator
        }
    }
    try testing.expect(!isNearRepeatTailLoop(ids.items));
    try testing.expect(degenerateTail(ids.items) == null);
}

test "degenerateTail: the exact tier reports its tier and keeps ONE cycle" {
    const al = testing.allocator;
    // Identical 3-token cycles past the span bar after a real prefix. The cut
    // is a truncation, so what is emitted should still SHOW what the model got
    // stuck on — one copy of the cycle survives, the rest do not.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 10 });
    var k: usize = 0;
    while (k < degenerate_loop_min_span / 3 + 1) : (k += 1) try ids.appendSlice(al, &[_]u32{ 101, 102, 103 });

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    // 4 prefix + 1 kept cycle = 7 tokens survive.
    try testing.expectEqual(@as(usize, 7), d.start);
    // What survives is the honest prefix plus exactly one cycle.
    try testing.expectEqualSlices(u32, &[_]u32{ 7, 8, 9, 10, 101, 102, 103 }, ids.items[0..d.start]);
}

test "degenerateTail: the trim start walks back PAST the near-repeat window" {
    const al = testing.allocator;
    // The near-repeat tier judges the last 1024 tokens, but a restatement
    // loop that has been running for 3000 tokens is degenerate for all 3000.
    // Trimming only the window would hand the client the other ~2000 back,
    // which is the whole failure this exists to stop.
    const phrasings = [_][]const u32{
        &[_]u32{ 1, 2, 3, 4, 5 },
        &[_]u32{ 1, 2, 3, 5, 4 },
        &[_]u32{ 1, 2, 4, 3, 5 },
    };
    var honest = std.ArrayList(u32).empty;
    defer honest.deinit(al);
    var i: u32 = 0;
    while (i < 900) : (i += 1) try honest.append(al, 1000 + i); // all distinct = healthy

    // Pick phrasings pseudo-randomly: a deterministic rotation would be an
    // exact cycle and the long-period tier would convict it first, which is
    // not the tier under test.
    var loop = std.ArrayList(u32).empty;
    defer loop.deinit(al);
    var rng: u32 = 12345;
    while (loop.items.len < near_repeat_min_span + 1000) {
        rng = rng *% 1664525 +% 1013904223;
        try loop.appendSlice(al, phrasings[(rng >> 16) % phrasings.len]);
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, honest.items);
    try ids.appendSlice(al, loop.items);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.near_repeat, d.tier);
    // Well past the single window, and never into the healthy prefix.
    try testing.expect(d.start < ids.items.len - near_repeat_window);
    try testing.expect(d.start >= honest.items.len - near_repeat_step);
}

test "degenerateTail: healthy output is never convicted, so nothing is trimmed" {
    var ids: [4000]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(degenerateTail(&ids) == null);
}

test "degenerateTail: the long-period tier keeps one copy of its sentence cycle" {
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 1, 2 });
    var cycle: [40]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @intCast(500 + i);
    var k: usize = 0;
    while (k < degenerate_loop_long_min_span / cycle.len + 1) : (k += 1) try ids.appendSlice(al, &cycle);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.long_cycle, d.tier);
    try testing.expectEqual(@as(usize, 2 + cycle.len), d.start);
}

test "commitAdaptiveWidth: the summary follows the running width from both commit sites" {
    const t = testing;
    var cur: usize = 2048;
    var st: AdaptiveWidthState = .{ .width_min = 2048, .width_max = 2048 };
    commitAdaptiveWidth(&cur, &st, 1024);
    try t.expectEqual(@as(usize, 1024), cur);
    try t.expectEqual(@as(u32, 1), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 2048), st.width_max);
    commitAdaptiveWidth(&cur, &st, 4096);
    try t.expectEqual(@as(usize, 4096), cur);
    try t.expectEqual(@as(u32, 2), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 4096), st.width_max);
}

test "tailMergeMax: the tail a chunk may absorb scales with the width" {
    const t = testing;
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(8192)); // no-op where it was justified
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(4096));
    try t.expectEqual(@as(usize, 256), tailMergeMax(2048));
    try t.expectEqual(@as(usize, 128), tailMergeMax(1024));
    try t.expectEqual(@as(usize, 64), tailMergeMax(PREFILL_CHUNK_FLOOR));
    try t.expectEqual(@as(usize, 1), tailMergeMax(1));
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |w| {
        try t.expect(tailMergeMax(w) * 8 <= w);
    }

    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, true));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, true));
}

test "the scaled tail-merge bound is gated on the per-chunk adaptive width" {
    // The scaled bound is gated: every arch whose resolved chunk is under 4096 would otherwise
    // merge a smaller tail and run an extra chunk.
    const t = testing;

    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(PREFILL_CHUNK_FLOOR, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(1024, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(2048, false));
    try t.expectEqual(@as(usize, 64), tailMergeMaxFor(PREFILL_CHUNK_FLOOR, true));
    try t.expectEqual(@as(usize, 256), tailMergeMaxFor(2048, true));
    try t.expectEqual(tailMergeMaxFor(4096, false), tailMergeMaxFor(4096, true));
    try t.expectEqual(tailMergeMaxFor(8192, false), tailMergeMaxFor(8192, true));

    try t.expectEqual(@as(usize, 812), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, true));
    try t.expectEqual(@as(usize, 1324), nextChunkEnd(0, 1324, 1024, false));
    try t.expectEqual(@as(usize, 1024), nextChunkEnd(0, 1324, 1024, true));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, true));

    try t.expectEqual(@as(usize, 1), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false));
    try t.expectEqual(@as(usize, 2), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, true));
}

test "nextChunkEnd: a tiny trailing remainder merges into the last chunk" {
    // A chat-templated prompt often lands a token or two past the chunk size
    // (8192-target prompts tokenize to 8193). A 1-token trailing chunk pays a
    // FULL graph + eval-barrier + cache-clear for one token — pure overhead.
    // Remainders under the merge floor extend the current chunk instead.
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, false));
    // A substantial remainder stays its own chunk.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 8192 + 600, 8192, false));
    // Mid-prompt chunks are untouched.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 16385, 8192, false));
    try testing.expectEqual(@as(usize, 16385), nextChunkEnd(8192, 16385, 8192, false));
}

test "boundedPrefillChunk: fused head dims and short contexts keep the base chunk" {
    // head_dim <= 128 rides MLX's fused SDPA — no materialized scores, no cap,
    // at ANY context length.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 16, 1_000_000, true, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 64, 32, 1_000_000, true, false, false));
    // hd 256 but short context: 16 heads x 8192 ctx x 8192 chunk x 2B
    // = 2 GiB scores, inside the 4 GiB budget -> full chunk kept. This is the
    // fleet-protection property: every Gemma-4 checkpoint ships
    // head_dim 256, so typical prompts must keep full prefill throughput.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false, false));
    // Degenerate inputs never cap.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 0, 100_000, true, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 0, true, false, false));
}

test "boundedPrefillChunk: caps hd-256 long context even with the fused kernel active" {
    // The msv_attn_p256 kernel removes the SCORE transient, but a big chunk
    // still scales the MoE-gather / KV-concat transients — measured +22 GB
    // peak for +3% speed at a 99K prompt. The cap deliberately ignores
    // prefillHeadDimFused (see the fn doc); pin that with the override ON.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
}

test "boundedPrefillChunk: long context shrinks to the scores budget, floored and rounded" {
    // gemma-4-26B geometry (16 heads): budget/(16*ctx*2) …
    // ctx 32768 -> exactly 4096.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 16, 32768, true, false, false));
    // ctx 100000 -> raw 1342, rounded down to the 512 grain -> 1024.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
    // ctx 262144 (the PR-#69 255K case) -> 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 16, 262_144, true, false, false));
    // A 24-head geometry at 262144: raw 341 -> floor 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 24, 262_144, true, false, false));
    // e4b geometry (8 heads) at 131072: exactly 2048.
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 8, 131_072, true, false, false));
}

test "boundedPrefillChunk: a 192-wide MLA score is budgeted, and the hd-256 policies stay hd-256" {
    // A hybrid MLA arch can declare head_dim 128 (its value width) so it fell
    // through the "fused SDPA covers it" early-out — while its MLA scores at
    // qk width 192 on the composed path. 32 heads.
    //
    // Short prompts keep the full chunk: 32 * 8192 * 8192 * 2B = 4 GiB exactly.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
    // Then it halves with the prompt, holding one score tensor at the budget.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 192, 32, 16384, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 32768, false, true, false));
    // The 38201-token prompt that measured a +28.8 GB peak at chunk 8192:
    // 4 GiB / (32 * 38201 * 2) = 1757 -> floored to the 512 grain.
    try testing.expectEqual(@as(usize, 1536), boundedPrefillChunk(8192, 192, 32, 38_201, false, true, false));
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));

    // The two hd-256-measured policies must NOT adopt this arch:
    // - the fused-kernel branch (no score tensor) is hd-256-only,
    // - the 2048 composed cap was tuned on a 27B's own prefill ladder.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    // With the fused override on, a real hd-256 MoE takes the 4096 branch...
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 32, 8192, false, true, false));
    // ...and the 192-wide arch still gets its honest full chunk at the same shape.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
}

test "boundedPrefillChunk: never raises a caller-lowered base chunk" {
    // --prefill-chunk 1024 with headroom for 4096: the explicit lower value wins.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 16, 32768, true, false, false));
    // Even the floor never raises a tiny explicit base.
    try testing.expectEqual(@as(usize, 256), boundedPrefillChunk(256, 256, 16, 262_144, true, false, false));
}

test "boundedPrefillChunk: fused-causal (default) non-sliding hd-256 — MoE caps at 4096, dense keeps the full chunk" {
    // With the causal arm FUSED (default since the budgeted-dispatch flip) no
    // score tensor exists, so the scores-budget formula is moot for the
    // A hybrid class - and its old shrink to 1024 at 64K starved the
    // dequant+GEMM qmm route (engages at M >= 2048): the 64K rung was the
    // ladder's weakest for exactly this reason.
    std.debug.assert(transformer_mod.fused256_override == null);
    // DENSE hybrids (a hybrid class): no expert-gather transients, and a
    // full-size chunk halves per-chunk dequant sweeps — chunk 8192 measured
    // +1.4% over 4096 at the 8K rung (M4 Max, 2026-07-30), flat at 32K.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, false));
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (gemma-26B@99K: +3% speed for +22 GB peak is a bad trade).
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 24, 8192, false, false, false));
    // Sliding-band archs (gemma: fused band kernel wants big chunks) keep
    // the formula-only policy.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false, false));
    // Fused head dims never cap regardless of arch.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 24, 1_000_000, false, false, false));
}

test "effectivePrefillChunk: an arch with no prefill score tensor defaults to the wide chunk, still under the pin and an explicit width" {
    const saved = .{ prefill_chunk_override, prefill_chunk_explicit };
    defer {
        prefill_chunk_override = saved[0];
        prefill_chunk_explicit = saved[1];
    }
    prefill_chunk_override = 8192;
    prefill_chunk_explicit = false;
    try testing.expectEqual(WIDE_PREFILL_CHUNK, effectivePrefillChunk(0, 64, 32768, true, true, false, 0));
    try testing.expectEqual(WIDE_PREFILL_CHUNK, effectivePrefillChunk(0, 64, 32768, true, true, false, WIDE_PREFILL_CHUNK));
    try testing.expectEqual(@as(usize, 8192), effectivePrefillChunk(0, 64, 32768, true, true, false, 8192));
    // Every other arch keeps the launch width.
    try testing.expectEqual(@as(usize, 8192), effectivePrefillChunk(128, 32, 32768, false, false, false, WIDE_PREFILL_CHUNK));
    prefill_chunk_override = 4096;
    prefill_chunk_explicit = true;
    try testing.expectEqual(@as(usize, 4096), effectivePrefillChunk(0, 64, 32768, true, true, false, WIDE_PREFILL_CHUNK));
}

test "boundedPrefillChunk: a long-context-gated MoE hd-256 arch offers the 8192 rung, an ungated one caps at 4096" {
    std.debug.assert(transformer_mod.fused256_override == null);
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 400_000, false, true, true));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(2048, 256, 24, 8192, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, true));
}

test "boundedPrefillChunk: composed-causal (kill switch) keeps the 2048 cap + score formula" {
    // MLX_SERVE_FUSED_256_CAUSAL=0 restores composed causal, where SMALLER
    // chunks measured faster on the 27B ladder (2026-07-12, M4 Max): 8K
    // prompt 225 -> 235.8 tok/s at chunk 2048 (peak 28.9 -> 19.8 GB).
    // Chunking IS block-level causal skipping for composed attention, and
    // the score transient shrinks with it.
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 32768, false, false, false));
    // The scores-budget formula still wins BELOW the cap: 64K on 24 heads
    // yields 1024 (measured better than 2048 there: 186 vs 182.3 tok/s).
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
}

test "filteredProbsBlock: every draft row is the SAME density the sample is drawn from" {
    // Exactness rests on q being the true proposal: the draft must be drawn
    // from byte-for-byte the row handed to `specAcceptProb`. The way that
    // breaks silently is filtering the two differently — so pin that a row
    // is a normalized distribution supported ONLY on the kept set, and that
    // `mlx_random_categorical` over log(q) can never land outside it.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 8;
    const M: c_int = 3;
    // Three rows with deliberately different argmaxes, so a row-blind
    // implementation (broadcasting one row's mask) fails.
    const raw = [_]f32{
        5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 0.0,
        0.0, 9.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0,
    };
    const shape = [_]c_int{ 1, M, V };
    const logits = mlx.mlx_array_new_data(&raw, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var sampling = SamplingParams{};
    sampling.temperature = 0.8;
    sampling.top_k = 2;
    const q = try filteredProbsBlock(logits, sampling, s);
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_array_eval(q));
    const qsh = mlx.getShape(q);
    try testing.expectEqual(M, qsh[0]);
    try testing.expectEqual(V, qsh[1]);

    const data = mlx.mlx_array_data_float32(q) orelse return error.MlxArrayDataNull;
    // Per row: sums to 1, and exactly top_k entries carry mass.
    const want_support = [_][2]usize{ .{ 0, 1 }, .{ 5, 6 }, .{ 1, 3 } };
    for (0..@intCast(M)) |r| {
        var sum: f32 = 0;
        var nonzero: usize = 0;
        for (0..@intCast(V)) |c| {
            const v = data[r * @as(usize, @intCast(V)) + c];
            sum += v;
            if (v > 0) nonzero += 1;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-5);
        try testing.expectEqual(@as(usize, 2), nonzero);
        for (want_support[r]) |c| {
            try testing.expect(data[r * @as(usize, @intCast(V)) + c] > 0);
        }
    }

    // Draw the way nextDflash does and assert every sample is in q's support.
    var logq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logq);
    try mlx.check(mlx.mlx_log(&logq, q, s));
    var draws: usize = 0;
    while (draws < 24) : (draws += 1) {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_astype(&ids, sampled, .int32, s));
        try mlx.check(mlx.mlx_array_eval(ids));
        const idd = mlx.mlx_array_data_int32(ids) orelse return error.MlxArrayDataNull;
        for (0..@intCast(M)) |r| {
            const tok: usize = @intCast(idd[r]);
            // q(draft) > 0 is what keeps the accept ratio finite.
            try testing.expect(data[r * @as(usize, @intCast(V)) + tok] > 0);
            // And `sliceProbRow` must hand back THAT row, not another.
            const row = try sliceProbRow(q, @intCast(r), s);
            defer _ = mlx.mlx_array_free(row);
            const got = try probAt(row, @intCast(tok), s);
            try testing.expectApproxEqAbs(data[r * @as(usize, @intCast(V)) + tok], got, 1e-6);
        }
    }
    _ = allocator;
}

test "specAcceptProb: full Leviathan ratio, q-clamped" {
    // p <= q: accept with p/q.
    try testing.expect(@abs(Generator.specAcceptProb(0.2, 0.4) - 0.5) < 1e-6);
    // p > q: always accept.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.2));
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.4));
    // Degenerate q underflow never divides by zero.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.5, 0.0));
}

test "spec sampling exactness: draft-from-q + ratio-accept + residual reproduces target p (toy vocab)" {
    // Host-level simulation of the exact per-position algorithm the
    // stochastic speculative round runs: draft ~ q, accept with min(1, p/q), on reject
    // sample from normalize(max(p - q, 0)). The output distribution must
    // equal p (Leviathan/Chen) — this is the invariant the one-hot rule
    // broke for sampled proposals.
    const p = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    const q = [_]f64{ 0.4, 0.3, 0.2, 0.1 };
    var residual: [4]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..4) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x5A3E);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        // draft ~ q
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        const a = Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]));
        if (rnd.float(f32) < a) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 3;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..4) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
}

test "spec sampling exactness: a SHORTLIST proposal (q zero off the top-k) still reproduces target p" {
    // Bar: with q supported on a subset, the marginal is still p, and the
    // off-shortlist tokens arrive only through the residual.
    const p = [_]f64{ 0.30, 0.25, 0.20, 0.15, 0.10 };
    const q = [_]f64{ 0.50, 0.30, 0.20, 0.00, 0.00 }; // shortlist = {0, 1, 2}
    const off_shortlist = [_]usize{ 3, 4 };

    var residual: [5]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..5) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x51157);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        // A draft is never drawn off the shortlist, so q[draft] > 0 always.
        try testing.expect(q[draft] > 0);
        if (rnd.float(f32) < Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]))) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 4;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..5) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
    // The residual is the ONLY route to a token the shortlist missed.
    for (off_shortlist) |i| try testing.expect(counts[i] > 0);
}

test "specEvExpectedTokens: 1 + sum of acceptance chain products" {
    const a = [_]f32{ 0.5, 0.5, 0.5 };
    try testing.expectApproxEqAbs(@as(f32, 1.5), Generator.specEvExpectedTokens(&a, 1), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.75), Generator.specEvExpectedTokens(&a, 2), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.875), Generator.specEvExpectedTokens(&a, 3), 1e-5);
    // Zero acceptance: every round still commits exactly the t1 bonus token.
    const z = [_]f32{ 0.0, 0.0 };
    try testing.expectApproxEqAbs(@as(f32, 1.0), Generator.specEvExpectedTokens(&z, 2), 1e-5);
}

test "specEvRoundCost: piecewise marginals (flat verify region, then the gated-delta width ramp)" {
    const costs = Generator.SpecEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    try testing.expectApproxEqAbs(@as(f32, 1.19), Generator.specEvRoundCost(costs, 1, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.57), Generator.specEvRoundCost(costs, 3, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.59), Generator.specEvRoundCost(costs, 3, true), 1e-5);
    // Positions 4+ pay the ramp: +0.32 each instead of +0.19.
    try testing.expectApproxEqAbs(@as(f32, 2.21), Generator.specEvRoundCost(costs, 5, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.specEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.32), Generator.specEvMarginalCost(costs, 4), 1e-6);
}

test "specEvCostsFor: the default surface, an explicit override and atomic rejection of a bad one" {
    try testing.expectEqual(Generator.SPEC_EV_DEFAULT_COSTS, Generator.specEvCostsFor(null));

    // An explicit four-value override replaces the four values and drops the third region.
    const tuned = Generator.specEvCostsFor("0.10, 0.11, 0.22, 0.03");
    try testing.expectApproxEqAbs(@as(f32, 0.10), tuned.draft, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.11), tuned.per_pos_lo, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.22), tuned.per_pos_hi, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.03), tuned.sync, 1e-6);
    try testing.expectEqual(@as(u32, 0), tuned.nax_from);
    try testing.expectApproxEqAbs(@as(f32, 0.0), tuned.per_pos_nax, 1e-6);

    // Invalid overrides are atomic no-ops.
    const invalid = [_][]const u8{
        "",
        "0.10,0.11",
        "garbage,0.11,0.22,0.03",
        "nan,0.11,0.22,0.03",
        "0.10,0.11,0.22,0.03,0.04",
        "0.10,-0.11,0.22,0.03",
    };
    for (invalid) |raw| try testing.expectEqual(Generator.SPEC_EV_DEFAULT_COSTS, Generator.specEvCostsFor(raw));
}

test "SpecCostSource: a measured cliff stops the plan where the fitted surface would extend" {
    // M1 Pro 27B: depth 4 -> 5 is a +43% round. The fitted surface prices
    // position 5 at 0.26 floor units and extends; the table has measured
    // it. Same acceptance, same cap, two answers.
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    const prior = Generator.specEvPlanSrc(&a, 8, Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 10000, null), 8);
    try testing.expect(prior.m_hi >= 5);
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t.observe(4, 10000, 70.0, 5.0, true, false);
        _ = t.observe(5, 10000, 102.0, 6.0, true, false);
    }
    const src = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 10000, &t);
    try testing.expect(src.fromTable());
    // Anchored at the narrowest measured width: the two sources agree
    // there, and below it the prior's shape fills in.
    try testing.expectApproxEqAbs(Generator.specEvRoundCostAt(Generator.SPEC_EV_DEFAULT_COSTS, 3, false, 10000), src.roundCost(3, false), 1e-4);
    try testing.expectApproxEqAbs(Generator.specEvRoundCostAt(Generator.SPEC_EV_DEFAULT_COSTS, 2, false, 10000), src.roundCost(2, false), 1e-4);
    try testing.expect(src.marginal(5) > 2.5 * src.marginal(4));
    // Past the widest measured width the cliff's slope continues (it is
    // steeper than the prior's marginal here).
    try testing.expectApproxEqAbs(src.roundCost(5, false) + (src.roundCost(5, false) - src.roundCost(4, false)), src.roundCost(6, false), 1e-3);
    const measured = Generator.specEvPlanSrc(&a, 8, src, 8);
    try testing.expectEqual(@as(u32, 4), measured.m_hi);
    try testing.expect(measured.m_lo <= 4);
    // ONE sample at the cliff (untrusted) already floors the horizon.
    var t1 = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t1.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t1.observe(4, 10000, 70.0, 5.0, true, false);
    }
    _ = t1.observe(5, 10000, 102.0, 6.0, true, false);
    const src1 = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 10000, &t1);
    try testing.expectEqual(@as(u32, 4), Generator.specEvPlanSrc(&a, 8, src1, 8).m_hi);
    // An inactive bucket with no active neighbour is the prior, verbatim.
    const empty = round_cost.Table{};
    const cold = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 10000, &empty);
    try testing.expect(!cold.fromTable());
    const cold_plan = Generator.specEvPlanSrc(&a, 8, cold, 8);
    try testing.expectEqual(prior.m_hi, cold_plan.m_hi);
    try testing.expectEqual(prior.m_lo, cold_plan.m_lo);
}

test "specBasePlan: the accept policy plans one chunk from the acceptance EMAs, whatever the table's token column says" {
    // Round TIME favours depth 3; the w3 cell's token count froze pessimistic (one bad trial).
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(2, 1000, 42.0, 3.0, true, false);
        _ = t.observe(3, 1000, 48.0, 2.2, true, false);
        _ = t.observe(4, 1000, 56.0, 3.9, true, false);
    }
    const a = [_]f32{ 0.9, 0.85, 0.8, 0.6, 0.5, 0.5 };
    const src = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 1000, &t);
    const legacy = Generator.specBasePlan(.legacy, &a, 6, src, 6);
    try testing.expect(legacy.m_lo != 3);
    const plan = Generator.specBasePlan(.accept, &a, 6, src, 6);
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expectEqual(plan.m_lo, plan.m_hi); // no chunk B: no confidence sync, drafts stay on the rerank path
}

test "SpecCostSource: under the accept policy one clean time sample prices an unmeasured width" {
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(3, 1000, 45.0, 4.0, true, false);
        _ = t.observe(4, 1000, 51.0, 5.0, true, false);
    }
    _ = t.observe(5, 1000, 57.0, 6.0, true, false);
    var src = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 1000, &t);
    const guarded = src.roundCost(5, false) / src.scale;
    try testing.expect(guarded > 58.0); // the prior's marginal, floored by the sample
    src.table_tokens = false;
    try testing.expectApproxEqAbs(@as(f32, 57.0), src.roundCost(5, false) / src.scale, 1e-3);
}

test "dflashAcceptWidth: falls any number of widths on cold acceptance, climbs one per round on echo" {
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(4, 1000, 48.0, 3.0, true, false);
        _ = t.observe(5, 1000, 54.0, 3.0, true, false);
        _ = t.observe(6, 1000, 60.0, 3.0, true, false);
    }
    const src = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 1000, &t);
    const cold = [_]f32{ 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 };
    try testing.expect(Generator.dflashAcceptWidth(&cold, 7, src, 4) <= 2);
    const echo = [_]f32{ 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 };
    try testing.expectEqual(@as(u32, 5), Generator.dflashAcceptWidth(&echo, 7, src, 4));
    // Serial wins only against a measured serial round.
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(0, 1000, 12.0, 1.0, true, false);
    const src0 = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 1000, &t);
    const dead = [_]f32{ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 };
    try testing.expectEqual(@as(u32, 0), Generator.dflashAcceptWidth(&dead, 7, src0, 4));
}

test "specProbePeriod: the probe date follows the acceptance EMAs, and one clearly worse sample still backs it off" {
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(2, 1000, 42.0, 3.0, true, false);
        _ = t.observe(3, 1000, 48.0, 3.5, true, false);
    }
    const src = Generator.SpecCostSource.init(Generator.SPEC_EV_DEFAULT_COSTS, 1000, &t);
    // Depth 3 reads clearly worse than 2: a long wait.
    const cold = [_]f32{ 0.7, 0.6, 0.2, 0.5, 0.5, 0.5 };
    const far = Generator.specProbePeriod(&cold, src, 2);
    try testing.expect(far >= 48);
    // The shallow indices saturate (prose turned into echo): the stale a[2] no longer holds the probe off.
    const echo = [_]f32{ 1.0, 1.0, 0.2, 0.5, 0.5, 0.5 };
    try testing.expect(Generator.specProbePeriod(&echo, src, 2) < far);
    // An unmeasured next width is learned at the cold period, unless its one sample was a cliff.
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.specProbePeriod(&echo, src, 3));
    _ = t.observe(4, 1000, 70.0, 2.0, true, false);
    try testing.expect(Generator.specProbePeriod(&echo, src, 3) >= 100);
}

test "buildPaddedBatch pads to max length with zeros and records lengths" {
    const seqs = [_][]const u32{
        &[_]u32{ 101, 7592, 102 },
        &[_]u32{ 101, 102 },
    };
    var pb = try buildPaddedBatch(testing.allocator, &seqs);
    defer pb.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), pb.max_len);
    try testing.expectEqual(@as(usize, 2), pb.lengths.len);
    try testing.expectEqual(@as(usize, 3), pb.lengths[0]);
    try testing.expectEqual(@as(usize, 2), pb.lengths[1]);
    const expected = [_]i32{ 101, 7592, 102, 101, 102, 0 };
    try testing.expectEqualSlices(i32, &expected, pb.ids);
}

test "maskedMeanPoolNormalize excludes padded positions and unit-normalizes" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (pad slot holds garbage
    // that must not leak into the pool), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };
    const rows = try maskedMeanPoolNormalize(testing.allocator, hidden, &lengths, s);
    defer {
        for (rows) |r| testing.allocator.free(r);
        testing.allocator.free(rows);
    }
    // Row 0: mean of (1,0),(3,4) = (2,2) → L2-normalized (1/√2, 1/√2).
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][1], 1e-4);
    // Row 1: mean of (0,2),(0,4),(0,6) = (0,4) → normalized (0, 1).
    try testing.expectApproxEqAbs(@as(f32, 0.0), rows[1][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rows[1][1], 1e-4);
}

test "embedSubBatchEnd: item cap, padded-footprint budget, oversize singleton, order" {
    const short = [_]u32{ 1, 2 };
    const long: [600]u32 = @splat(0);
    // Item cap alone (all short): full batch.
    const all_short = [_][]const u32{ &short, &short, &short };
    try testing.expectEqual(@as(usize, 3), embedSubBatchEnd(&all_short, 0, 64, 64 * 512));
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&all_short, 0, 2, 64 * 512));

    // One long input caps how many rows pad to its length: 600-token rows fit
    // budget/600 = 54 per sub-batch at the default budget, not 64.
    const many_long: [60][]const u32 = @splat(&long);
    const end = embedSubBatchEnd(&many_long, 0, 64, 64 * 512);
    try testing.expectEqual(@as(usize, (64 * 512) / 600), end);

    // A mixed batch stops BEFORE a long input would inflate the padded
    // footprint of everything before it.
    const mixed = [_][]const u32{ &short, &short, &long };
    const mixed_end = embedSubBatchEnd(&mixed, 0, 64, 4);
    try testing.expectEqual(@as(usize, 2), mixed_end);

    // An input larger than the whole budget still runs — alone.
    const huge: [40000]u32 = @splat(0);
    const with_huge = [_][]const u32{ &huge, &short };
    try testing.expectEqual(@as(usize, 1), embedSubBatchEnd(&with_huge, 0, 64, 64 * 512));
    // ...and the next sub-batch starts right after it (order preserved).
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&with_huge, 1, 64, 64 * 512));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// ── Allocator-cache clear cadence (issue #110) ───────────────────────────────

test "clear cadence survives variable spec strides" {
    // A spec-decode round emits `1 + accepted` tokens, so `step` advances by a
    // VARIABLE amount and `step % 256 == 0` can walk clean over every multiple
    // — silently, because a decode path that never clears is output-identical
    // to one that does. Interval arithmetic against the last clear cannot be
    // stepped over.
    for ([_]u32{ 1, 3, 5, 9 }) |stride| {
        var step: u32 = 0;
        var last_clear: u32 = 0;
        var clears: usize = 0;
        var max_gap: u32 = 0;
        while (step < 4096) {
            step += stride;
            if (shouldClearAllocatorCache(step, last_clear, CACHE_CLEAR_INTERVAL)) {
                const gap = step - last_clear;
                if (gap > max_gap) max_gap = gap;
                last_clear = step;
                clears += 1;
            }
        }
        try testing.expect(clears >= 15);
        // A stride can overshoot the interval by at most stride-1 tokens.
        try testing.expect(max_gap <= CACHE_CLEAR_INTERVAL + stride);
    }
}

fn evalLazyToken(lazy: mlx.mlx_array) !u32 {
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    return @intCast(val);
}

test "fillSuppressMask: padding rows past the tokenizer's last defined id are suppressed" {
    // A config's `vocab_size` is a padded matrix dimension, not a vocabulary.
    // One model declares 248320 while the tokenizer defines 248044 + 33 =
    // 248077 ids: 243 rows that decode to NOTHING, carry whatever the
    // initializer left, and were sampleable — a drawn one emits no text while
    // consuming a step and entering the KV. Same mask, second row class.
    var buf: [16]bool = undefined;
    fillSuppressMask(&buf, &[_]u32{3}, 12);
    for (buf, 0..) |v, i| {
        const want = (i == 3) or (i >= 12);
        try testing.expectEqual(want, v);
    }

    // No padding known (0) and a fully-defined vocab (== len) both add
    // nothing: a mask over the WHOLE vocab is -inf everywhere, and argmax
    // over that row returns id 0 — the all-false-mask class from the other
    // side. The reserved id must still be the only true.
    for ([_]usize{ 0, 16, 99 }) |defined| {
        fillSuppressMask(&buf, &[_]u32{3}, defined);
        for (buf, 0..) |v, i| try testing.expectEqual(i == 3, v);
    }
}

test "Tokenizer.definedVocabSize: highest defined id + 1, 0 when nothing is defined" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .sentencepiece_bpe);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 0), tok.definedVocabSize());
    // Added specials sit ABOVE the base vocab and are not contiguous with it,
    // so the answer is the maximum key, never the entry count.
    try tok.id_to_token.put(0, "a");
    try tok.id_to_token.put(1, "b");
    try tok.id_to_token.put(11, "<|special|>");
    try testing.expectEqual(@as(usize, 12), tok.definedVocabSize());
}

test "suppress_mask: a suppressed id is unreachable from both samplers, everything else stays" {
    // A collapsed distribution can rank `<|fim_hole|>` (a reserved FIM marker) in the
    // top-5 at every degenerate position, and greedy DREW it live. A reserved
    // marker in chat output is always a bug, so the sampler carries a
    // model-lifetime suppression mask. Contract pinned here: with the mask,
    // neither sampler can return a suppressed id on any of its paths (greedy
    // fast path, greedy general, categorical, sync sampleToken); with a null
    // mask the same logits DO pick it (red-on-revert); and logprobs keep
    // reporting the MODEL's raw distribution — the suppressed id still ranks
    // where the model put it, because the field reports the model while the
    // mask is sampling policy.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const V: usize = 8;
    const mask = try buildSuppressMask(&[_]u32{3}, 0, V, s);
    defer _ = mlx.mlx_array_free(mask);

    // id 3 (suppressed) dominates, id 5 is the runner-up, the rest are so far
    // down that after masking the categorical underflows to exactly id 5.
    var data = [_]f32{ -1000, -1000, -1000, 100, -1000, 50, -1000, -1000 };

    // Greedy fast path: [1, 1, V].
    const shape3 = [_]c_int{ 1, 1, V };
    const logits3 = mlx.mlx_array_new_data(&data, &shape3, 3, .float32);
    defer _ = mlx.mlx_array_free(logits3);
    const masked_sp = SamplingParams{ .temperature = 0.0, .suppress_mask = mask };
    const open_sp = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_sp, s)));
    try testing.expectEqual(@as(u32, 3), try evalLazyToken(sampleTokenLazy(logits3, open_sp, s)));

    // Categorical path at temp 1.0 (exercises the seq_len==1 reshape arm and
    // the sampler proper; the masked lane's probability is exactly 0 and the
    // -1000 lanes underflow, so the draw is deterministic).
    const masked_t1 = SamplingParams{ .temperature = 1.0, .suppress_mask = mask };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_t1, s)));

    // Sync sampler (logprobs/penalty path): same policy, and the reported
    // distribution stays the model's own — rank 1 is the SUPPRESSED id.
    const r = try sampleToken(allocator, logits3, masked_sp, null, 2, s);
    const lp = r.logprob_result orelse return error.NoLogprobs;
    defer allocator.free(lp.top_logprobs);
    try testing.expectEqual(@as(u32, 5), r.token_id);
    try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);

    // Stochastic-verify filters: a suppressed draft's acceptance probability
    // must be exactly 0 (always rejected), and the residual it corrects from
    // can never re-draw it.
    const probs = try probsAtLastPos(logits3, masked_t1, s);
    defer _ = mlx.mlx_array_free(probs);
    try testing.expectEqual(@as(f32, 0.0), try probAt(probs, 3, s));
    try testing.expect(try probAt(probs, 5, s) > 0.99);
}

test "computeLogprobs: rank 1 is the argmax and ranks descend, under a tie-saturated distribution" {
    // The producer used to recover token ids by scanning the whole vocab for
    // FLOAT EQUALITY with each `mlx_topk` value. Ties make that ambiguous, and
    // at temp 0 the post-temperature distribution saturates so ties are
    // everywhere — measured 0/5 positions where rank 1 was the chosen token on
    // a trivial prompt. The observable contract: with greedy sampling the
    // chosen token IS the argmax, so rank 1 must equal it.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Six exactly-tied entries, then a strict runner-up and a strict max.
    const data = [_]f32{ 1, 1, 1, 1, 1, 1, 2, 3 };
    const shape = [_]c_int{ 1, 8 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const chosen: u32 = 7; // argmax
    const r = try computeLogprobs(allocator, logits, chosen, 3, s);
    defer allocator.free(r.top_logprobs);

    try testing.expectEqual(@as(usize, 3), r.top_logprobs.len);
    try testing.expectEqual(chosen, r.top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[2].token_id < 6);
    // Strictly descending, and ids distinct.
    try testing.expect(r.top_logprobs[0].logprob > r.top_logprobs[1].logprob);
    try testing.expect(r.top_logprobs[1].logprob > r.top_logprobs[2].logprob);
    try testing.expect(r.top_logprobs[0].token_id != r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[1].token_id != r.top_logprobs[2].token_id);
    // The chosen token's own logprob must agree with its rank-1 entry.
    try testing.expectApproxEqAbs(r.top_logprobs[0].logprob, r.token_logprob, 1e-6);
}

test "computeLogprobs: f16 logits keep finite, exact log-probabilities" {
    // p(token 1) = e^-25 underflows f16, so log(softmax) in the logits dtype read -inf.
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const raw = [_]f32{ 0.0, -20.0, -30.0, 5.0 };
    const a32 = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var a16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a16);
    try mlx.check(mlx.mlx_astype(&a16, a32, .float16, s));
    const r = try computeLogprobs(testing.allocator, a16, 1, 4, s);
    defer testing.allocator.free(r.top_logprobs);
    const lse: f32 = 5.0 + @log(1.0 + @exp(@as(f32, -5.0)) + @exp(@as(f32, -25.0)) + @exp(@as(f32, -35.0)));
    try testing.expectApproxEqAbs(-20.0 - lse, r.token_logprob, 1e-4);
    for (r.top_logprobs) |t| try testing.expect(std.math.isFinite(t.logprob));
}

test "computeLogprobs: an id past the vocab reports -100, top_n 0 lists none, [1,1,V] reads as [1,V]" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const allocator = testing.allocator;
    const raw = [_]f32{ 0, 1, 2, 3 };
    const flat = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(flat);
    const deep = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(deep);
    const lse: f32 = 3.0 + @log(1.0 + @exp(@as(f32, -1)) + @exp(@as(f32, -2)) + @exp(@as(f32, -3)));

    const past = try computeLogprobs(allocator, flat, 9, 2, s);
    defer allocator.free(past.top_logprobs);
    try testing.expectEqual(@as(f32, -100.0), past.token_logprob);
    try testing.expectEqual(@as(usize, 2), past.top_logprobs.len);
    try testing.expectEqual(@as(u32, 3), past.top_logprobs[0].token_id);

    const none = try computeLogprobs(allocator, flat, 1, 0, s);
    defer allocator.free(none.top_logprobs);
    try testing.expectEqual(@as(usize, 0), none.top_logprobs.len);
    try testing.expectApproxEqAbs(1.0 - lse, none.token_logprob, 1e-5);

    const d = try computeLogprobs(allocator, deep, 2, 3, s);
    defer allocator.free(d.top_logprobs);
    try testing.expectApproxEqAbs(2.0 - lse, d.token_logprob, 1e-5);
    try testing.expectEqual(@as(usize, 3), d.top_logprobs.len);
    try testing.expectEqual(@as(u32, 1), d.top_logprobs[2].token_id);
}

test "promptLogprobs: no vocab-wide array outlives its use, so the top-k sort is the peak" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const rows = 64;
    const vocab = 65536;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_random_normal(&logits, &[_]c_int{ 1, rows, vocab }, 3, .bfloat16, 0, 1, key, s));
    try mlx.check(mlx.mlx_array_eval(logits));
    var ids: [rows]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i * 997 % vocab);
    var out: std.ArrayList(LogprobResult) = .empty;
    defer {
        for (out.items) |r| allocator.free(r.top_logprobs);
        out.deinit(allocator);
    }

    var base: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&base));
    _ = mlx.mlx_reset_peak_memory();
    try promptLogprobs(allocator, logits, &ids, 5, s, &out);
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    // Bar: under six [rows, vocab] 4-byte arrays above what was live before (the sort alone holds five).
    try testing.expect(peak - base < 6 * rows * vocab * 4);
    try testing.expectEqual(@as(usize, rows), out.items.len);
}

test "promptLogprobs: every row scores its NEXT token, ranks tie on the lower id" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const allocator = testing.allocator;
    const rows = [3][8]f32{
        .{ 1, 1, 1, 1, 1, 1, 2, 3 },
        .{ 0, 0, 0, 0, 0, 3, 0, 3 },
        .{ -1.5, 0.25, 4, -3, 2, 0, 1, 0.5 },
    };
    const logits = mlx.mlx_array_new_data(&rows, &[_]c_int{ 1, 3, 8 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const next = [_]u32{ 7, 5, 3 };

    var out: std.ArrayList(LogprobResult) = .empty;
    defer {
        for (out.items) |r| allocator.free(r.top_logprobs);
        out.deinit(allocator);
    }
    try promptLogprobs(allocator, logits, &next, 2, s, &out);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    for (rows, next, out.items) |row, id, r| {
        var lse: f64 = 0;
        for (row) |v| lse += @exp(@as(f64, v));
        lse = @log(lse);
        try testing.expectApproxEqAbs(@as(f64, row[id]) - lse, r.token_logprob, 1e-5);
        try testing.expectEqual(@as(usize, 2), r.top_logprobs.len);
        try testing.expectApproxEqAbs(@as(f64, row[r.top_logprobs[0].token_id]) - lse, r.top_logprobs[0].logprob, 1e-5);
    }
    try testing.expectEqual(@as(u32, 7), out.items[0].top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), out.items[0].top_logprobs[1].token_id);
    try testing.expectEqual(@as(u32, 5), out.items[1].top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 7), out.items[1].top_logprobs[1].token_id);
    try testing.expectEqual(@as(u32, 2), out.items[2].top_logprobs[0].token_id);

    const too_many = [_]u32{ 1, 2, 3, 4 };
    try testing.expectError(error.PromptLogprobsUnavailable, promptLogprobs(allocator, logits, &too_many, 2, s, &out));
}

test "sampleToken: reported logprobs are the model's, not the client's temperature" {
    // Same prompt, same chosen token, three temperatures used to report three
    // different logprobs (-0.2129 / -0.0607 / -2.1566 at 0 / 0.6 / 2.0). The
    // value belongs to the model, so it must not move with a sampling knob.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const data = [_]f32{ 0, 1, 2, 3 };
    const shape = [_]c_int{ 1, 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // log_softmax([0,1,2,3])[3], computed independently.
    var denom: f64 = 0;
    for (data) |v| denom += @exp(@as(f64, v));
    const expect_top: f32 = @floatCast(3.0 - @log(denom));

    const temps = [_]f32{ 0.0, 0.6, 2.0 };
    for (temps) |t| {
        const sp = SamplingParams{ .temperature = t, .seed = 7 };
        const r = try sampleToken(allocator, logits, sp, null, 2, s);
        const lp = r.logprob_result orelse return error.NoLogprobs;
        defer allocator.free(lp.top_logprobs);
        try testing.expectEqual(@as(usize, 2), lp.top_logprobs.len);
        try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);
        try testing.expectEqual(@as(u32, 2), lp.top_logprobs[1].token_id);
        try testing.expectApproxEqAbs(expect_top, lp.top_logprobs[0].logprob, 1e-5);
    }
}

test "logprobs publish through the one-token delay, never straight from sampleToken" {
    // The decode loop returns `next_token_id` and, in the SAME call, forwards
    // it to sample its successor — so a path that publishes `sampleToken`'s
    // result as `last_logprob` shifts the whole array by one. Live symptom: a
    // one-token "OK" reply came back with `<|role_end|>` at rank 1, which reads
    // as a broken ranking rather than a broken pairing, and sent a real
    // quantization hunt down the wrong path for a day.
    //
    // The needles are concatenated so this test's OWN source doesn't match.
    const needle = "last_logprob" ++ " = ";
    const allowed = "last_logprob" ++ " = self.pending_logprob";
    const src = @embedFile("generate.zig");
    var it = std.mem.splitScalar(u8, src, '\n');
    var lineno: usize = 0;
    var seen: usize = 0;
    while (it.next()) |raw| {
        lineno += 1;
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        if (std.mem.indexOf(u8, line, needle) == null) continue;
        // `= null` is the ownership hand-off to the scheduler, not a publish.
        if (std.mem.indexOf(u8, line, "= null") != null) continue;
        seen += 1;
        if (std.mem.indexOf(u8, line, allowed) == null) {
            std.debug.print("generate.zig:{d}: logprobs published without the one-token delay: {s}\n", .{
                lineno, std.mem.trim(u8, line, " "),
            });
            return error.LogprobPairingBypass;
        }
    }
    try testing.expect(seen >= 1);
}

test "dflash: nextDflash greedy equals serial decode, invariants exact each round" {
    // Hermetic end-to-end round loop on the tiny dense Gemma 4 trunk + tiny DFlash
    // assistant (dflash.TinyFix). The bar is the spec-decode contract:
    //   - greedy dflash-on emits EXACTLY the serial greedy tokens (any bug in
    //     verify alignment, rollback truncate, correction indexing, or the
    //     anchor-row drop shifts the stream);
    //   - after EVERY round: trunk cache.step == prompt_len + emitted and the
    //     assistant context tracks it exactly (base + step == cache.step) at
    //     whatever accepted counts the round produced.
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_len = try tmp_trunk.dir.realPath(io, &trunk_buf);
    const trunk_path = trunk_buf[0..trunk_len];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);

    var tmp_asst = std.testing.tmpDir(.{});
    defer tmp_asst.cleanup();
    var asst_buf: [512]u8 = undefined;
    const asst_len = try tmp_asst.dir.realPath(io, &asst_buf);
    const asst_path = asst_buf[0..asst_len];
    try dflash_mod.TinyFix.writeAssistant(io, tmp_asst.dir, asst_path, s);

    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    var weights = try model_mod.loadWeights(io, allocator, trunk_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 16;

    // Serial baseline.
    var serial: [want]u32 = undefined;
    {
        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .skip_lazy_preforward = true,
        });
        defer gen.deinit(allocator);
        var n: usize = 0;
        while (n < want) {
            const t = (try gen.next(allocator)) orelse break;
            serial[n] = t;
            n += 1;
        }
        try testing.expectEqual(want, n);
    }

    // DFlash arm.
    {
        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var dm = try dflash_mod.loadDflash(io, allocator, s, asst_path);
        defer dm.deinit();
        try dm.bind(&xfm);

        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .dflash_enabled = true,
            .dflash = &dm,
            // This is the numeric DFlash guard, not the economics-gate test:
            // every emitted token must pass through nextDflash itself.
            .dflash_min_accepted_per_round = 0,
        });
        defer gen.deinit(allocator);
        try testing.expectEqual(@as(u32, 4), gen.dflash_block_size);
        // Prefill built context for the whole prompt.
        try testing.expectEqual(prompt.len, gen.dflash_ctx.?.absLen());
        try testing.expectEqual(prompt.len, gen.ctx.cache.step);

        var got = std.ArrayList(u32).empty;
        defer got.deinit(allocator);
        while (true) {
            const attempts_before = gen.dflash_attempted;
            // The budget's last token is the previous verify's: it leaves without a round.
            const last = gen.max_tokens -| gen.completion_tokens == 1;
            const res = (try gen.nextDflash(allocator)) orelse break;
            defer allocator.free(res.tokens);
            try got.appendSlice(allocator, res.tokens);
            // A real DFlash round commits trunk and assistant context to the
            // same exact boundary at every accepted count. Falling back here
            // would make the remaining equality checks compare serial decode
            // with itself and gut the default-on draft-quantization guard.
            try testing.expect(last or gen.dflash_attempted != attempts_before);
            try testing.expectEqual(prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail, gen.ctx.cache.step);
            try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        }
        try testing.expect(gen.dflash_attempted > 0);
        try testing.expect(!gen.spec_disabled_runtime);
        try testing.expectEqual(want, got.items.len);
        try testing.expectEqual(@as(u32, @intCast(want)), gen.completion_tokens);
        try testing.expectEqual(want, gen.generated_ids.items.len);
        for (serial, got.items) |a, b| try testing.expectEqual(a, b);
    }

    // DFlash2 arm: selector-traced greedy drafts + dyn-conv forward. The bar
    // is unchanged — a selector draft only survives verify when it IS the
    // trunk argmax, so the emitted stream must still be byte-identical to
    // serial at every accepted count.
    {
        var tmp_a2 = std.testing.tmpDir(.{});
        defer tmp_a2.cleanup();
        var a2_buf: [512]u8 = undefined;
        const a2_path = a2_buf[0..try tmp_a2.dir.realPath(io, &a2_buf)];
        try dflash_mod.TinyFix.writeAssistant2(io, tmp_a2.dir, a2_path, s, false);

        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var dm = try dflash_mod.loadDflash(io, allocator, s, a2_path);
        defer dm.deinit();
        try testing.expect(dm.selector != null);
        try dm.bind(&xfm);

        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .dflash_enabled = true,
            .dflash = &dm,
            .dflash_min_accepted_per_round = 0,
        });
        defer gen.deinit(allocator);

        var got = std.ArrayList(u32).empty;
        defer got.deinit(allocator);
        while (true) {
            const attempts_before = gen.dflash_attempted;
            // The budget's last token is the previous verify's: it leaves without a round.
            const last = gen.max_tokens -| gen.completion_tokens == 1;
            const res = (try gen.nextDflash(allocator)) orelse break;
            defer allocator.free(res.tokens);
            try got.appendSlice(allocator, res.tokens);
            try testing.expect(last or gen.dflash_attempted != attempts_before);
            try testing.expectEqual(prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail, gen.ctx.cache.step);
            try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        }
        try testing.expect(gen.dflash_attempted > 0);
        try testing.expectEqual(want, got.items.len);
        for (serial, got.items) |a, b| try testing.expectEqual(a, b);
    }
}

const PolicyArm = struct {
    /// A DFlash2 assistant (selector confidence) instead of the v1 block drafter.
    v2: bool,
    /// Verify exactly this many rows whenever the draft has them (0 = the policy chooses).
    pin_rows: u32 = 0,
    /// Pre-train the calibration to trust every draft, so the policy verifies the whole block.
    trust: bool = false,
    sampled: bool = false,
    /// A prompt that ends mid-repeat, so the first round copies the context's continuation.
    copy: bool = false,
};

/// Ten distinct ids repeated, then the first nine again: `COPY_PATTERN[9]` as t1 continues the repeat.
const COPY_PATTERN = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17 };

/// One request through `nextDflash` under the round policy on the tiny fixture pair, three plain
/// rounds first: the trunk and assistant cursors agree after EVERY round whatever mix of plain and
/// drafted rounds ran, and a greedy stream is the serial stream. Returns the generator's stats.
fn policyArm(arm: PolicyArm, serial: []const u32) !dflash_policy.PositionStats {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_path = trunk_buf[0..try tmp_trunk.dir.realPath(io, &trunk_buf)];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);
    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    var weights = try model_mod.loadWeights(io, allocator, trunk_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);

    var tmp_asst = std.testing.tmpDir(.{});
    defer tmp_asst.cleanup();
    var asst_buf: [512]u8 = undefined;
    const asst_path = asst_buf[0..try tmp_asst.dir.realPath(io, &asst_buf)];
    if (arm.v2) try dflash_mod.TinyFix.writeAssistant2(io, tmp_asst.dir, asst_path, s, false) else try dflash_mod.TinyFix.writeAssistant(io, tmp_asst.dir, asst_path, s);

    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    var dm = try dflash_mod.loadDflash(io, allocator, s, asst_path);
    defer dm.deinit();
    try dm.bind(&xfm);

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const copy_prompt = COPY_PATTERN ++ COPY_PATTERN ++ COPY_PATTERN[0..9].*;
    const plain_prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    const prompt: []const u32 = if (arm.copy) &copy_prompt else &plain_prompt;
    var copy_serial: [16]u32 = undefined;
    if (arm.copy) {
        var sx = try Transformer.init(io, allocator, config, &weights);
        defer sx.deinit();
        var sg = try Generator.initWithOptions(io, allocator, &sx, &tok_dummy, prompt, 16, SamplingParams{ .temperature = 0.0 }, &.{}, .{ .skip_lazy_preforward = true });
        defer sg.deinit(allocator);
        sg.next_token_id = COPY_PATTERN[9];
        for (&copy_serial) |*t| t.* = (try sg.next(allocator)).?;
    }
    const reference: []const u32 = if (arm.copy) &copy_serial else serial;
    const sampling = if (arm.sampled) SamplingParams{ .temperature = 1.0, .seed = 7 } else SamplingParams{ .temperature = 0.0 };
    const want: u32 = 16;
    var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, prompt, want, sampling, &.{}, .{
        .dflash_enabled = true,
        .dflash = &dm,
        .dflash_policy = true,
    });
    defer gen.deinit(allocator);
    const pol = &gen.dflash_pol.?;
    pol.pin_rows = arm.pin_rows;
    pol.drafting.owed = if (arm.copy) 0 else 3;
    if (arm.copy) gen.next_token_id = COPY_PATTERN[9];
    const plain_rounds: usize = if (arm.copy) 0 else 3;
    if (arm.trust) for (0..10) |b| for (0..40) |_| pol.calib.observe((@as(f32, @floatFromInt(b)) + 0.5) / 10.0, true);

    var got = std.ArrayList(u32).empty;
    defer got.deinit(allocator);
    var round: usize = 0;
    while (true) : (round += 1) {
        const plain_before = gen.dflash_plain;
        const attempts_before = gen.dflash_attempted;
        const last = gen.max_tokens -| gen.completion_tokens == 1;
        const res = (try gen.nextDflash(allocator)) orelse break;
        defer allocator.free(res.tokens);
        try got.appendSlice(allocator, res.tokens);
        if (round < plain_rounds) {
            try testing.expectEqual(@as(usize, 1), res.tokens.len);
            try testing.expectEqual(plain_before + 1, gen.dflash_plain);
            try testing.expectEqual(attempts_before, gen.dflash_attempted);
        }
        // A plain stretch rides the serial pipeline: one lookahead token is in the trunk's cache, unemitted.
        const lookahead: usize = if (gen.has_pending_logits or gen.has_pending_token) 1 else 0;
        try testing.expectEqual(prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail + lookahead, gen.ctx.cache.step);
        // The assistant context reaches the trunk's cursor, appended or still parked.
        try testing.expectEqual(gen.ctx.cache.step, gen.dflashContextLen());
        // A drafted round appends whatever the plain ones left parked.
        if (gen.dflash_attempted != attempts_before) try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        _ = last;
    }
    try testing.expect(gen.dflash_plain >= plain_rounds);
    if (arm.copy) try testing.expect(gen.dflash_copies > 0);
    try testing.expect(!gen.spec_disabled_runtime);
    try testing.expectEqual(@as(usize, want), got.items.len);
    if (!arm.sampled) for (reference, got.items) |a, b| try testing.expectEqual(a, b);
    // The scheduler settles the context to the committed prefix, whatever lookahead the run ended on.
    const committed = prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail;
    try gen.dflashSettle(allocator, committed);
    try testing.expectEqual(committed, gen.dflash_ctx.?.absLen());
    return gen.dflash_stats;
}

test "dflash: the round policy runs plain rounds between drafted ones, the stream stays serial's" {
    // A plain round is the verify forward at one row (no assistant forward): it must advance the trunk
    // and the assistant context exactly like a drafted round, so DFlash can resume after any stretch of
    // them; a draft the policy verifies only in part must not shift the accept, rollback or q indexing.
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_path = trunk_buf[0..try tmp_trunk.dir.realPath(io, &trunk_buf)];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);
    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    var weights = try model_mod.loadWeights(io, allocator, trunk_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var tok_dummy: Tokenizer = undefined;
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    var serial: [16]u32 = undefined;
    {
        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, SamplingParams{ .temperature = 0.0 }, &.{}, .{ .skip_lazy_preforward = true });
        defer gen.deinit(allocator);
        for (&serial) |*t| t.* = (try gen.next(allocator)).?;
    }

    // The v1 block drafter verifies its whole block.
    const v1 = try policyArm(.{ .v2 = false }, &serial);
    try testing.expect(v1.rows[4] > 0);
    // The DFlash2 selector: two rows of a three-draft block, then the whole block once it is trusted.
    const two = try policyArm(.{ .v2 = true, .pin_rows = 2 }, &serial);
    try testing.expect(two.rows[2] > 0 and two.rows[3] == 0 and two.rows[4] == 0);
    const trusted = try policyArm(.{ .v2 = true, .trust = true }, &serial);
    try testing.expect(trusted.rows[4] > 0);
    // A copied continuation (no assistant forward) is verified like any draft.
    _ = try policyArm(.{ .v2 = false, .copy = true }, &serial);
    _ = try policyArm(.{ .v2 = false, .copy = true, .sampled = true }, &serial);
    // Sampled requests: the Leviathan accept reads q at the draft's own stride, whatever was kept.
    _ = try policyArm(.{ .v2 = true, .pin_rows = 2, .sampled = true }, &serial);
    _ = try policyArm(.{ .v2 = true, .trust = true, .sampled = true }, &serial);
}

test "prefill chunk loop yields to the interleave hook between chunks, never after the last" {
    // The scheduler runs decode ticks for active streams at this seam so a
    // long prefill cannot stall them for its whole duration. The final
    // boundary is excluded: the post-prefill decode tick covers it.
    const src = @embedFile("generate.zig");
    const progress = "if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);";
    const at = std.mem.indexOf(u8, src, progress) orelse return error.ProgressPublishMissing;
    const guard = "if (pos < loop_end) {";
    const guard_at = std.mem.indexOfPos(u8, src, at, guard) orelse return error.InterleaveGuardMissing;
    const call = "if (options.interleave" ++ "_hook) |hk| hk.call(hk.ctx);";
    const call_at = std.mem.indexOfPos(u8, src, guard_at, call) orelse return error.InterleaveCallMissing;
    // The call sits INSIDE the guard (same statement block, before the loop
    // re-tests `pos`), not somewhere later in the file.
    try std.testing.expect(call_at - guard_at < 200);
}

test "countSpliceRows counts image, audio and video placeholders, extras only when declared" {
    const ids = [_]i32{ 7, 99, 99, 8, 88, 9, 77 };
    try testing.expectEqual(@as(usize, 2), countSpliceRows(&ids, 99, 0, 0));
    try testing.expectEqual(@as(usize, 3), countSpliceRows(&ids, 99, 88, 0));
    try testing.expectEqual(@as(usize, 4), countSpliceRows(&ids, 99, 88, 77));
    try testing.expectEqual(@as(usize, 0), countSpliceRows(ids[0..1], 99, 88, 77));
    // An audio-only model declares no image token: token 0 is text.
    try testing.expectEqual(@as(usize, 1), countSpliceRows(&[_]i32{ 0, 88, 0 }, 0, 88, 0));
}

test "vision prefill chunks by default and the splice offset feeds every prefill forward" {
    // Source scan: the chunk-size pick keys on the kill switch (not bare
    // has_vision), the chunk loop and the final-span forward both set
    // ctx.vision_splice_offset, and the loop advances the consumed-row count.
    const src = @embedFile("generate.zig");
    const marker = "const default_chunk = if (has_vision and !vision" ++ "_chunked) loop_end else PREFILL_CHUNK;";
    try testing.expect(std.mem.indexOf(u8, src, marker) != null);
    const set_off = "ctx.vision_splice" ++ "_offset = vision_rows_consumed;";
    const first = std.mem.indexOf(u8, src, set_off) orelse return error.ChunkOffsetMissing;
    try testing.expect(std.mem.indexOfPos(u8, src, first + 1, set_off) != null); // final-span site too
    try testing.expect(std.mem.indexOf(u8, src, "vision_rows_consumed += countSplice" ++ "Rows(") != null);
}

test "dspark drafting backoff: prose rounds that lose to the plain step are skipped, copy-like rounds are not" {
    // A round that commits few tokens for its cost loses to the plain step (costs in ms).
    var prose = dflash_policy.Drafting{};
    try testing.expect(prose.shouldDraft());
    prose.record(3.2 * 17.6, 66);
    try testing.expect(!prose.shouldDraft());
    var echo = dflash_policy.Drafting{};
    try testing.expect(echo.shouldDraft());
    echo.record(5.5 * 17.6, 70);
    try testing.expect(echo.shouldDraft());
}

test "a block decoder's entry token stops the round before it drafts" {
    // Bar: an EOS or third pad as the round's entry token ends the request before any draft.
    const eos = [_]u32{ 151645, 151643 };
    var pad: u32 = 0;
    try testing.expect(tokenStops(151645, &eos, &pad));
    try testing.expect(tokenStops(151643, &eos, &pad));
    try testing.expect(!tokenStops(17, &eos, &pad));

    // A pad run stops on the third; any real token resets it.
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(tokenStops(0, &eos, &pad));
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(9, &eos, &pad));
    try testing.expectEqual(@as(u32, 0), pad);
}

/// Deterministic Gaussian logits (Box-Muller over an xorshift stream): diffuse
/// enough that the nucleus for top_p 0.95 spans thousands of tokens, which is
/// where a narrow cumulative accumulator loses the tail.
fn samplerTestRow(host: []f32, seed: u64, sigma: f32) void {
    var rng: u64 = seed | 1;
    var i: usize = 0;
    while (i < host.len) : (i += 2) {
        var u: [2]f64 = undefined;
        for (&u) |*d| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            d.* = (@as(f64, @floatFromInt(rng >> 11)) + 1.0) / 9007199254740994.0;
        }
        const r = @sqrt(-2.0 * @log(u[0]));
        const a = 6.283185307179586 * u[1];
        host[i] = @floatCast(r * @cos(a) * @as(f64, sigma));
        if (i + 1 < host.len) host[i + 1] = @floatCast(r * @sin(a) * @as(f64, sigma));
    }
}

/// Nucleus the sampler is meant to keep, in f64: ranked descending, a column is
/// in when less than top_p of the mass outranks it.
fn samplerTestExactKeep(allocator: std.mem.Allocator, row: []const f32, top_p: f32) !usize {
    const sorted = try allocator.alloc(f64, row.len);
    defer allocator.free(sorted);
    for (row, sorted) |x, *d| d.* = @floatCast(x);
    std.mem.sort(f64, sorted, {}, comptime std.sort.desc(f64));
    const mx = sorted[0];
    var z: f64 = 0;
    for (sorted) |d| z += @exp(d - mx);
    var above: f64 = 0;
    var keep: usize = 0;
    for (sorted) |d| {
        if (above >= @as(f64, @floatCast(top_p))) break;
        above += @exp(d - mx) / z;
        keep += 1;
    }
    return keep;
}

/// Read an mlx array back as f32 host values. A slice is a strided VIEW, so the
/// read goes through a contiguous copy and a flat reshape, never the base
/// pointer.
fn samplerTestReadFlat(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, arr, .float32, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_contiguous(&dense, as_f32, false, s));
    const flat_shape = [_]c_int{@intCast(n)};
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, dense, &flat_shape, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const out = try allocator.alloc(f32, n);
    @memcpy(out, mlx.mlx_array_data_float32(flat).?[0..n]);
    return out;
}

/// Bit equality of two mlx arrays read back as f32: both routes end in one
/// `where` over the same logits, so agreeing on the kept columns is agreeing on
/// every byte.
fn samplerTestExpectSameBits(a: mlx.mlx_array, b: mlx.mlx_array, n: usize, s: mlx.mlx_stream) !void {
    const ad = try samplerTestReadFlat(testing.allocator, a, n, s);
    defer testing.allocator.free(ad);
    const bd = try samplerTestReadFlat(testing.allocator, b, n, s);
    defer testing.allocator.free(bd);
    for (ad, bd, 0..) |x, y, i| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, x, y });
            return error.FilteredLogitsDiffer;
        }
    }
}

fn samplerTestKeptCount(row: []const f32) usize {
    var n: usize = 0;
    for (row) |x| {
        if (x != -std.math.inf(f32)) n += 1;
    }
    return n;
}

test "sampler ranks break ties by the lowest original index" {
    // The contract both filters and both of their routes share. mlx's stable
    // merge sort is what delivers it; if a bump breaks stability this fails.
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, 5.0, 5.0, 5.0, 1.0, 5.0 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ranks);
    try ranksDescending(&ranks, logits, s);
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, ranks, .float32, s));
    const got = try samplerTestReadFlat(testing.allocator, as_f32, 6, s);
    defer testing.allocator.free(got);
    // The four tied 5.0s come first in ascending column order, then 2.0, then 1.0.
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 5, 0, 4 }, got);
}

test "applyTopP keeps the exact nucleus on a bf16 vocab row" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    const rounded = try allocator.alloc(f32, v);
    defer allocator.free(rounded);

    for ([_]u64{ 1, 2, 3, 4, 5, 6 }) |seed| {
        samplerTestRow(host, seed *% 0x9E3779B97F4A7C15, 4.0);
        const shape = [_]c_int{ 1, @intCast(v) };
        const f32_row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_row);
        var bf16_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_row);
        try mlx.check(mlx.mlx_astype(&bf16_row, f32_row, .bfloat16, s));

        // The reference reads the row the kernel sees: bf16-rounded values.
        var rt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rt);
        try mlx.check(mlx.mlx_astype(&rt, bf16_row, .float32, s));
        const rt_host = try samplerTestReadFlat(allocator, rt, v, s);
        defer allocator.free(rt_host);
        @memcpy(rounded, rt_host);
        const want = try samplerTestExactKeep(allocator, rounded, 0.95);

        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, bf16_row, 0.95, 0, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        // The boundary column's own probability is ~1e-5 here while the f32 scan
        // carries ~1e-7 of error, so the exclusive mass can cross top_p one
        // column early or late against the f64 reference. The defect this pins
        // was 148 columns wide.
        const got = samplerTestKeptCount(res_host);
        const diff = @as(i64, @intCast(want)) - @as(i64, @intCast(got));
        try testing.expect(diff >= -1 and diff <= 1);
    }
}

test "applyTopP at top_p 0 keeps exactly the argmax" {
    // The nucleus is "mass strictly above the rank < top_p"; rank 0 sees 0, so a
    // literal 0 kept nothing and the sampler drew uniformly from the vocabulary.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    samplerTestRow(host, 0x9E3779B97F4A7C15, 4.0);
    const shape = [_]c_int{ 1, @intCast(v) };
    const row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(row);
    for ([_]u32{ 0, 40 }) |bound| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, row, 0.0, bound, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        try testing.expectEqual(@as(usize, 1), samplerTestKeptCount(res_host));
        var best: usize = 0;
        for (host, 0..) |x, i| if (x > host[best]) {
            best = i;
        };
        try testing.expect(std.math.isFinite(res_host[best]));
    }
}

test "topRanksDescending returns the exact top m values and column ids" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 1) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2 quantized to 0.25 steps: thousands of exact ties.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]u32{ 1, 20, 257 }) |m| {
            var vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vals);
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try topRanksDescending(&vals, &ids, block, m, s);
            // The whole row's ranking, restricted to its first m: the bar the
            // shortlist has to reproduce, ties included.
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            try ranksDescending(&order, block, s);
            var ref_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids);
            try sliceLastAxis(&ref_ids, order, 0, m, s);
            var ref_vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_vals);
            try mlx.check(mlx.mlx_take_along_axis(&ref_vals, block, ref_ids, -1, s));
            var ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids_f);
            try mlx.check(mlx.mlx_astype(&ids_f, ids, .float32, s));
            var ref_ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids_f);
            try mlx.check(mlx.mlx_astype(&ref_ids_f, ref_ids, .float32, s));
            try samplerTestExpectSameBits(vals, ref_vals, rows * m, s);
            try samplerTestExpectSameBits(ids_f, ref_ids_f, rows * m, s);
        }
    }
}

test "applyTopK keeps exactly k columns, the highest ranked ones" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 7) *% 0xD1B54A32D192ED03, 4.0);
    // Row 1 quantized to 1.0 steps: a value cutoff would keep thousands here.
    for (host[v..][0..v]) |*x| x.* = @round(x.*);
    const shape = [_]c_int{ @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(block);

    for ([_]u32{ 1, 20, 64 }) |k| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopK(&res, block, k, s);
        const rd = try samplerTestReadFlat(allocator, res, rows * v, s);
        defer allocator.free(rd);
        for (0..rows) |r| {
            try testing.expectEqual(@as(usize, k), samplerTestKeptCount(rd[r * v ..][0..v]));
        }
        // Same columns as ranking the whole row.
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, block, s);
        var ref_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_ids);
        try sliceLastAxis(&ref_ids, order, 0, k, s);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try keepByIndex(&ref, block, ref_ids, s);
        try samplerTestExpectSameBits(res, ref, rows * v, s);
    }
}

test "shortlist top-p equals the whole row's ranking, ties and 248320-wide rows" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 4;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 11) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2: 0.25-step ties. Row 3: every column identical — the extreme the
        // rank contract exists for.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        for (host[3 * v ..][0..v]) |*x| x.* = 1.5;

        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]struct { p: f32, k: u32 }{
            .{ .p = 0.95, .k = 20 },
            .{ .p = 0.8, .k = 50 },
            .{ .p = 0.99, .k = 5 },
            .{ .p = 0.5, .k = 1 },
            .{ .p = 0.95, .k = 257 },
            .{ .p = 1e-6, .k = 20 },
        }) |cfg| {
            var masked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked);
            try applyTopK(&masked, block, cfg.k, s);

            var short = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(short);
            try applyTopP(&short, masked, cfg.p, cfg.k, s);
            var full = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(full);
            try applyTopP(&full, masked, cfg.p, 0, s);
            try samplerTestExpectSameBits(short, full, rows * v, s);

            // Same filtered logits and the same key must draw the same token.
            for (0..8) |d| {
                const key = seedKey(.{ .seed = 12345, .draw = @intCast(d) });
                defer _ = mlx.mlx_array_free(key);
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                var b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(b);
                try mlx.check(mlx.mlx_random_categorical(&a, short, -1, key, s));
                try mlx.check(mlx.mlx_random_categorical(&b, full, -1, key, s));
                try samplerTestExpectSameBits(a, b, rows, s);
            }
        }
    }
}

test "a seeded correction draw replays: same seed and position, same token; positions differ" {
    // Bar: the bonus/residual draw of a verify round is keyed like every other draw.
    const s = mlx.gpuStream();
    const v: c_int = 64;
    var vals: [64]f32 = undefined;
    var sum: f32 = 0;
    for (&vals, 0..) |*x, i| {
        x.* = 1.0 / @as(f32, @floatFromInt(i + 1));
        sum += x.*;
    }
    for (&vals) |*x| x.* /= sum;
    const shape = [_]c_int{ 1, v };
    const probs = mlx.mlx_array_new_data(&vals, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const onehot = try pldOneHotRow(0, v, s);
    defer _ = mlx.mlx_array_free(onehot);

    var distinct = false;
    var prev: ?u32 = null;
    for (0..16) |d| {
        const sp: SamplingParams = .{ .seed = 11, .draw = d };
        const a = try sampleFromProbs(probs, sp, s);
        try testing.expectEqual(a, try sampleFromProbs(probs, sp, s));
        try testing.expectEqual(try sampleResidual(probs, onehot, sp, s), try sampleResidual(probs, onehot, sp, s));
        try testing.expect(try sampleResidual(probs, onehot, sp, s) != 0); // the rejected draft is never redrawn
        if (prev) |q| distinct = distinct or q != a;
        prev = a;
    }
    try testing.expect(distinct);
}

test "shortlist top-p holds on rank-3 [B, L, V] blocks" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 23) *% 0xD1B54A32D192ED03, 4.0);
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(block);

    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try applyTopK(&masked, block, 20, s);
    var short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(short);
    try applyTopP(&short, masked, 0.95, 20, s);
    var full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full);
    try applyTopP(&full, masked, 0.95, 0, s);
    try samplerTestExpectSameBits(short, full, rows * v, s);

    // Per-ROW cutoffs: a block-wide filter would mask whole rows to -inf.
    const sd = try samplerTestReadFlat(allocator, short, rows * v, s);
    defer allocator.free(sd);
    for (0..rows) |r| {
        const kept = samplerTestKeptCount(sd[r * v ..][0..v]);
        try testing.expect(kept > 0 and kept <= 20);
    }
}

/// The composition every sampling site ran before `filterTopKTopP`: top-k, then
/// top-p bounded by that same k. The reference the one-pass filter is pinned to.
fn samplerTestComposed(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    var after_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(after_k);
    if (top_k > 0) {
        try applyTopK(&after_k, logits, top_k, s);
    } else {
        try mlx.check(mlx.mlx_array_set(&after_k, logits));
    }
    if (top_p < 1.0) return applyTopP(res, after_k, top_p, top_k, s);
    return mlx.check(mlx.mlx_array_set(res, after_k));
}

test "filterTopKTopP is byte-identical to the two-pass composition" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const Setting = struct { p: f32, k: u32 };
    const settings = [_]Setting{
        .{ .p = 0.95, .k = 20 }, .{ .p = 0.9, .k = 1 },  .{ .p = 1.0, .k = 20 },
        .{ .p = 0.95, .k = 0 },  .{ .p = 0.5, .k = 50 }, .{ .p = 0.95, .k = 999999 },
    };
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        samplerTestRow(host[0..v], 0x9E3779B97F4A7C15, 4.0);
        // Row 1 quantized to 0.25 steps: thousands of exact ties at the cutoff.
        samplerTestRow(host[v..][0..v], 0xD1B54A32D192ED03, 4.0);
        for (host[v..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        // Row 2 all equal: every column ties, so the ranking alone decides.
        @memset(host[2 * v ..][0..v], 1.25);

        // `[m, V]` and `[1, L, V]`: the mask must stay per ROW on both ranks.
        for ([_]usize{ 2, 3 }) |rank| {
            const shape2 = [_]c_int{ @intCast(rows), @intCast(v) };
            const shape3 = [_]c_int{ 1, @intCast(rows), @intCast(v) };
            const shape: []const c_int = if (rank == 2) &shape2 else &shape3;
            const f32_block = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_block);
            var block = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(block);
            try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
            for (settings) |cfg| {
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try samplerTestComposed(&want, block, cfg.p, cfg.k, s);
                var got = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(got);
                try filterTopKTopP(&got, block, cfg.p, cfg.k, s);
                samplerTestExpectSameBits(got, want, rows * v, s) catch |e| {
                    std.debug.print("v={d} rank={d} top_p={d} top_k={d}\n", .{ v, rank, cfg.p, cfg.k });
                    return e;
                };
            }
        }
    }
}

test "constrained generation returns one logprob entry per token, paired with that token" {
    // Bar: a json_schema request with logprobs gets an entry for every returned token, describing it.
    const path = std.c.getenv("LOGPROBS_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(path);
    var config = try model_mod.parseConfig(io, a, dir);
    var weights = try model_mod.loadWeights(io, a, dir);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = try tokenizer_mod.loadTokenizer(io, a, dir);
    defer tok.deinit();
    var token_bytes = try token_mask.build(a, &tok);
    defer token_bytes.deinit();

    const schema_src =
        \\{"type":"object","properties":{"a":{"type":"string","enum":["yes","no"]}},"required":["a"],"additionalProperties":false}
    ;
    var schema = try std.json.parseFromSlice(std.json.Value, a, schema_src, .{});
    defer schema.deinit();
    var sc: SchemaConstraint = undefined;
    try sc.initFromValue(a, schema.value, &token_bytes);
    defer sc.deinit();

    const prompt = try tok.encode(a, "Answer yes or no in JSON: is water wet?\n");
    defer a.free(prompt);
    const result = try generate(io, a, &xfm, &tok, prompt, 16, .{ .temperature = 0, .constraint = &sc.constraint }, config.eosTokenSlice(), 0, 5);
    defer {
        a.free(result.text);
        a.free(result.token_ids);
        if (result.logprobs) |lps| {
            for (lps) |lp| a.free(lp.top_logprobs);
            a.free(lps);
        }
    }

    const lps = result.logprobs orelse return error.NoLogprobs;
    try testing.expect(result.token_ids.len > 0);
    try testing.expectEqual(result.token_ids.len, lps.len);
    var paired: usize = 0;
    for (result.token_ids, lps) |id, lp| {
        try testing.expect(lp.token_logprob <= 0);
        for (lp.top_logprobs) |alt| {
            if (alt.token_id != id) continue;
            try testing.expectApproxEqAbs(alt.logprob, lp.token_logprob, 1e-5);
            paired += 1;
        }
    }
    try testing.expect(paired > 0);
}

test "repeat penalty shapes every sampling path, not only the logprobs one" {
    // Bar: a penalty changes greedy output with or without logprobs, the same way, and under a grammar.
    const path = std.c.getenv("LOGPROBS_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(path);
    var config = try model_mod.parseConfig(io, a, dir);
    var weights = try model_mod.loadWeights(io, a, dir);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = try tokenizer_mod.loadTokenizer(io, a, dir);
    defer tok.deinit();
    config.applyTokenizer(&tok, "<|im_end|>");
    var token_bytes = try token_mask.build(a, &tok);
    defer token_bytes.deinit();

    const prompt = try tok.encode(a, "<|im_start|>user\nWrite the word apple 40 times separated by spaces. Answer in JSON with a words array.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    defer a.free(prompt);
    const schema_src =
        \\{"type":"object","properties":{"words":{"type":"array","items":{"type":"string"}}},"required":["words"],"additionalProperties":false}
    ;
    var schema = try std.json.parseFromSlice(std.json.Value, a, schema_src, .{});
    defer schema.deinit();

    const Run = struct {
        fn ids(al: std.mem.Allocator, x: *Transformer, t: *const Tokenizer, p: []const u32, eos: []const u32, rp: f32, lp: u32, sv: ?std.json.Value, tb: *token_mask.TokenBytes) ![]u32 {
            var sc: SchemaConstraint = undefined;
            if (sv) |v| try sc.initFromValue(al, v, tb);
            defer if (sv != null) sc.deinit();
            const sampling: SamplingParams = .{ .temperature = 0, .repeat_penalty = rp, .constraint = if (sv != null) &sc.constraint else null };
            const r = try generate(std.Io.Threaded.global_single_threaded.io(), al, x, t, p, 48, sampling, eos, 0, lp);
            al.free(r.text);
            if (r.logprobs) |lps| {
                for (lps) |e| al.free(e.top_logprobs);
                al.free(lps);
            }
            return r.token_ids;
        }
    };
    const eos = config.eosTokenSlice();
    const plain = try Run.ids(a, &xfm, &tok, prompt, eos, 1.0, 0, null, &token_bytes);
    defer a.free(plain);
    const pen = try Run.ids(a, &xfm, &tok, prompt, eos, 2.0, 0, null, &token_bytes);
    defer a.free(pen);
    const pen_lp = try Run.ids(a, &xfm, &tok, prompt, eos, 2.0, 1, null, &token_bytes);
    defer a.free(pen_lp);
    try testing.expect(!std.mem.eql(u32, plain, pen_lp));
    try testing.expectEqualSlices(u32, pen_lp, pen);

    const json_plain = try Run.ids(a, &xfm, &tok, prompt, eos, 1.0, 0, schema.value, &token_bytes);
    defer a.free(json_plain);
    const json_pen = try Run.ids(a, &xfm, &tok, prompt, eos, 4.0, 0, schema.value, &token_bytes);
    defer a.free(json_pen);
    try testing.expect(!std.mem.eql(u32, json_plain, json_pen));
}

test "keyed sampling draws the softmax distribution" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const probs = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var logv: [4]f32 = undefined;
    for (probs, &logv) |p, *l| l.* = @log(p);
    const logits = mlx.mlx_array_new_data(&logv, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var counts: [4]u32 = @splat(0);
    const n: u32 = 4000;
    for (0..n) |d| {
        const tok = sampleTokenLazy(logits, .{ .temperature = 1.0, .seed = 11, .draw = d, .keyed = true }, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, tok));
        counts[@intCast(v)] += 1;
    }
    for (probs, counts) |p, c| try testing.expect(@abs(@as(f32, @floatFromInt(c)) / @as(f32, @floatFromInt(n)) - p) < 0.03);
}

test "every speculative decoder caps accepted drafts before commit" {
    // Class guard: every speculative path commits an always-emitted anchor
    // plus zero or more accepted drafts. A new/forgotten arm must cap that
    // accepted prefix before it mutates generated ids, KV/module state, or
    // usage. Output-only clipping in the scheduler is already too late.
    const source = @embedFile("generate.zig");
    const names = [_][]const u8{
        "nextPld",
        "nextDrafter",
        "nextDflash",
    };
    for (names) |name| {
        const signature = try std.fmt.allocPrint(testing.allocator, "    pub fn {s}", .{name});
        defer testing.allocator.free(signature);
        const start = std.mem.indexOf(u8, source, signature) orelse return error.MissingSpecDecoder;
        const end = std.mem.indexOfPos(u8, source, start + signature.len, "\n    pub ") orelse source.len;
        if (std.mem.indexOf(u8, source[start..end], "capAcceptedForTokenBudget(") == null) {
            std.debug.print("{s} does not cap accepted drafts before commit\n", .{name});
            return error.UncappedSpecDecoder;
        }
    }
}

test "specDecodeUnsupported: release-enforced guard for spec + constraint/logprobs (issue #97)" {
    // Speculative decode (PLD/drafter/DFlash) cannot honor a grammar constraint or
    // per-token logprobs. nextPld/nextDrafter/nextDflash only asserted this, which
    // compiles out in ReleaseFast (issue #97) - this is the real check they now
    // use to fail loud instead of streaming silently off-schema output.
    try testing.expect(!specDecodeUnsupported(.{}, 0)); // plain sampling → spec OK
    try testing.expect(specDecodeUnsupported(.{}, 1)); // any logprobs → not OK
    try testing.expect(specDecodeUnsupported(.{}, 5));
    // A grammar constraint present → not OK. The guard only null-checks the
    // pointer (never dereferences), so a dummy address is sufficient here.
    var dummy: Constraint = undefined;
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 0));
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 3));
}

test "only the pipelined decode step runs on the ops limit alone" {
    const src = @embedFile("generate.zig");
    const call = "asyncEvalDecodeStep" ++ "(vec)";
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, call));
    const next_fn = std.mem.indexOf(u8, src, "pub fn next(self: *Generator") orelse return error.MissingNext;
    const slow = std.mem.indexOfPos(u8, src, next_fn, "Phase 2: Slow path") orelse return error.MissingSlowPath;
    const at = std.mem.indexOf(u8, src, call).?;
    try std.testing.expect(at > next_fn and at < slow);
    const set_call = "mlx_lesto_commit_on_ops_limit_only" ++ "(";
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, src, set_call));
}

test "no decode path advances `step` outside advanceStep" {
    // Class guard for #110. `Generator.step` is the clear cadence's clock, so a
    // path that bumps it by hand is a path that strands its round's transients
    // in MLX's buffer pool forever - which is exactly what `nextDrafter`,
    // `nextDflash` and `scheduler.runBatchedDecodeTick` did. A new decode path added later must
    // route through `advanceStep` or fail here.
    //
    // The needles are concatenated so this test's OWN source doesn't match them.
    const needle = ".step" ++ " +=";
    const allowed = "self.step" ++ " += n";
    const sources = [_]struct { name: []const u8, src: []const u8 }{
        .{ .name = "generate.zig", .src = @embedFile("generate.zig") },
        .{ .name = "scheduler.zig", .src = @embedFile("scheduler.zig") },
    };
    var total: usize = 0;
    for (sources) |file| {
        var it = std.mem.splitScalar(u8, file.src, '\n');
        var lineno: usize = 0;
        while (it.next()) |raw| {
            lineno += 1;
            // Strip trailing line comments: several of them spell out
            // `cache.step += 1` while describing the KV cache's own counter.
            const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
            if (std.mem.indexOf(u8, line, needle) == null) continue;
            total += 1;
            if (std.mem.indexOf(u8, line, allowed) == null) {
                std.debug.print("{s}:{d}: raw step advance outside advanceStep: {s}\n", .{
                    file.name, lineno, std.mem.trim(u8, line, " "),
                });
                return error.RawStepAdvance;
            }
        }
    }
    // Exactly one: the assignment inside `advanceStep`. Zero means the field was
    // renamed and this guard went vacuous - update it with the rename.
    try testing.expectEqual(@as(usize, 1), total);

    // Same discipline for the clock itself: every clear the Generator makes records
    // it, so the decode cadence measures from the last ACTUAL clear. Two writers -
    // `advanceStep` and the prefill/decode handover - and a third that forgot the
    // write would leave the cadence firing against a stale clock.
    const clock = "last_cache_clear_step" ++ " =";
    var writers: usize = 0;
    var clock_it = std.mem.splitScalar(u8, sources[0].src, '\n');
    while (clock_it.next()) |raw| {
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        if (std.mem.indexOf(u8, line, clock) != null) writers += 1;
    }
    try testing.expectEqual(@as(usize, 2), writers);
}

test "no block decoder commits an argmax over a suppressed id" {
    // A row whose raw argmax is a padding id past the defined vocabulary, plus a
    // flagged special: both must lose to the best legal column. Run at the block
    // widths the committing decoders verify at (the gemma drafter `1 + m`,
    // DFlash a whole block), so a per-position axis bug cannot hide. WHICH decoders
    // are covered is a compile-time property: every commit site takes a
    // `CommittedArgmax`, and only `verifyArgmax` returns one.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 512;
    const defined: usize = 500;
    const flagged = [_]u32{7};
    for ([_]usize{ 2, 3, 5 }) |rows| try suppressedArgmaxCase(allocator, s, v, defined, &flagged, rows);
}

fn suppressedArgmaxCase(allocator: std.mem.Allocator, s: mlx.mlx_stream, v: usize, defined: usize, flagged: []const u32, rows: usize) !void {
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    @memset(host, 0.0);
    for (0..rows) |r| {
        const row = host[r * v ..][0..v];
        row[505] = 9.0; // padding row: highest of all
        row[7] = 8.0; // flagged special: second
        row[100 + r] = 7.0; // the best LEGAL id, different per position
    }
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const f32_block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f32_block);
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
    const mask = try buildSuppressMask(flagged, defined, v, s);
    defer _ = mlx.mlx_array_free(mask);

    var raw = try verifyArgmax(block, null, s);
    defer raw.deinit();
    const raw_ids = try samplerTestReadFlat(allocator, raw.lazy(), rows, s);
    defer allocator.free(raw_ids);
    for (raw_ids) |id| try testing.expectEqual(@as(f32, 505), id);

    var masked = try verifyArgmax(block, mask, s);
    defer masked.deinit();
    const ids = try samplerTestReadFlat(allocator, masked.lazy(), rows, s);
    defer allocator.free(ids);
    for (ids, 0..) |id, r| try testing.expectEqual(@as(f32, @floatFromInt(100 + r)), id);
}
