//! Plan 01 Phase 2 — continuous-batching scheduler.
//!
//! Owns the single inference thread (the only thread that calls into mlx
//! ops; mlx 0.31.2 made GPU streams thread-local so the model weights are
//! bound to whichever thread first calls `useCurrentThreadStream` after
//! load). Connection threads parse HTTP, build prompt token ids, call
//! `submit()` and then loop on `Slot.waitNext()` to read generated tokens
//! one at a time. The state machines for tool-call detection, thinking
//! blocks, SSE streaming, etc. live on the connection thread, unchanged.
//!
//! Per-slot state (KVCache, moe_seq_offset, ssm_entries, vision_embeddings)
//! lives on `Slot` itself; each slot's `ForwardCtx` points at those fields,
//! and the `Generator` constructed for the slot stores the ctx so
//! `xfm.forwardWith(&self.ctx, ...)` routes through slot-local state. The
//! shared Transformer holds the weights only — single-writer-per-tick on
//! the inference thread guards correctness while many slots can be in
//! flight at once.
//!
//! Decode tick logic (the Phase 3 gate):
//!   * `active.len == 1` (the common single-stream case) → legacy path:
//!     `Generator.next` / `nextPld` / `nextDrafter` with the same lazy
//!     pipeline that today's serial path uses. Bit-identical to pre-Phase-2.
//!   * `active.len >= 2` → batched: `forwardBatchedDecode` produces N logits
//!     in one kernel pass, sampled per-slot. PLD / drafter are forced off in
//!     this path (the speculative paths assume a single in-flight slot's
//!     KV cache; expensive to interleave).
//!   * `cancelled` slots are skipped and culled from `decoding`.
//!
//! Output channel: each slot owns a bounded ring (`std.ArrayList(u32)` +
//! cursor + cv) the inference thread pushes into and the connection thread
//! drains. Generation end is signaled by `state == .finished` (or `.errored`)
//! plus a cv broadcast.

const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const rp_mod = @import("reasoning_protocol.zig");
const drafter_mod = @import("drafter.zig");
const sys_info = @import("sys_info.zig");
const model_mod = @import("model.zig");
const modelopt_pack = @import("modelopt_pack.zig");
const chat_mod = @import("chat.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const restore_dump = @import("restore_dump.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");
const sleep_inhibit = @import("sleep_inhibit.zig");

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const ForwardCtx = transformer_mod.ForwardCtx;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const SamplingParams = generate_mod.SamplingParams;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("dflash.zig");
const round_cost_mod = @import("round_cost.zig");
const DflashModel = dflash_mod.DflashModel;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;

/// Phase A1: model-load plan executed on the scheduler's inference thread.
///
/// mlx 0.31.2 uses thread-local GPU streams: every `mlx_*` op binds to the
/// stream of the calling thread, and JIT-compiled closures are tied to that
/// stream too. If main loads the model and the scheduler later calls forward,
/// mlx aborts with "no Stream(gpu, N) in current thread". Solution: the
/// scheduler's inference thread does the load itself so the stream is bound
/// on the right thread from t0.
///
/// CPU-only state (parsed config, tokenizer, chat config) is loaded by main
/// and passed in by reference. The mlx-allocating pieces (weights tensors,
/// Transformer, vision encoder, drafter, JIT compile, warmup) all run on the
/// inference thread before `init()` returns.
pub const LoadParams = struct {
    /// Registry that owns the entry to populate + provides snapshot/eviction
    /// bookkeeping. Outlives the scheduler.
    registry: *ModelRegistry,
    /// The (pre-registered) LoadedModel stub that will be promoted to
    /// `.ready` by the inference-thread load. Holds id/path on entry;
    /// `loadModelOnInferenceThread` installs weights/transformer/vision/
    /// drafter/tokenizer/chat_config/config into this slot. Lifetime
    /// matches the registry.
    entry: *LoadedModel,
    /// Heap-allocated parsed config. Ownership transfers to `entry` on
    /// successful load; caller must NOT deinit/free externally after
    /// `Scheduler.init` returns.
    config: *ModelConfig,
    /// Heap-allocated tokenizer. Ownership transfers to `entry`.
    tok: *Tokenizer,
    /// Heap-allocated chat config. Ownership transfers to `entry`.
    chat_config: *ChatConfig,
    /// Path to the model directory. Borrowed; outlive scheduler.
    model_dir: []const u8,
    resident_model_bytes: ?u64 = null,
    /// Path to the assistant drafter checkpoint. Empty disables the drafter.
    /// Borrowed; outlive scheduler.
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,
    /// Eager warmup: fault weight pages + run a tiny forward to JIT-compile
    /// the decode path on the inference thread. Adds ~600-900 ms at boot but
    /// keeps the first user request fast.
    warmup_eager: bool = true,
    /// Drafter block size (caller computed via `drafter.recommendedBlockSize`).
    /// Ignored when `drafter_dir` is empty.
    draft_block_size: u32 = 4,
    /// Whether the user passed --draft-block-size explicitly (used for
    /// human-readable startup logging). Ignored when `drafter_dir` is empty.
    draft_block_size_explicit: bool = false,
    /// KV-cache storage backend. Defaults to dense bf16; user opts into
    /// 4/8-bit affine quantization via `--kv-quant {4,8}`. Stored on every
    /// per-slot KVCache and consulted at every read/write boundary.
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense,
    /// Zero disables all prefix reuse; RAM retention is selected separately.
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    /// Per-model hot prefix cache KV-bytes budget. 0 disables the byte cap.
    prefix_cache_mem_bytes: u64 = 0,
    /// Clamp the hot-cache byte budget against live post-load headroom
    /// (`server.prefixCacheMemForLoad`) — a pointer because the scheduler
    /// deliberately has no server.zig import. Null = no clamp (tests).
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64 = null,
    /// SSD tier byte budget for the hot prefix cache (`--prefix-cache-disk`).
    /// 0 disables persistence. Attached per model at load for pure-attention
    /// archs; entries live under `~/.mlx-serve/kv-cache/<fingerprint>`.
    prefix_cache_disk_bytes: u64 = 0,
    /// Iteration 2 (perf-plan Phase 4 #3): per-LoadedModel LRU cache
    /// of chat-template render+tokenize results. 0 disables the cache
    /// (useful for ablation benches / debugging). Default 4 matches
    /// `prefix_cache_capacity` — most warm-reuse benches exercise a
    /// handful of repeated prompts, and full chat conversations bump
    /// this counter anyway via LRU as new turns arrive.
    tokenize_cache_entries: u32 = 4,
    /// Headless boot: skip the startup load entirely and run the inference
    /// loop idle. The registry holds discovery stubs (or nothing); the first
    /// model — chat or media — loads on demand when a request names it. `entry`/
    /// `config`/`tok`/`chat_config` are still required (they seed the
    /// scheduler's borrowed-view fields) but are never installed on an entry.
    no_initial_load: bool = false,
    /// Optional metrics sink. Null when --metrics is off (the default).
    /// Stored on the Scheduler and read by the `finishSlot` per-request funnel.
    metrics: ?*metrics_mod.Metrics = null,
    /// The --ctx-size launch flag (0 = unset). MLX cold loads read their
    /// own config.json and ignore it.
    ctx_size: u32 = 0,
};

/// Submit-time parameters. `prompt_ids` and `eos_token_ids` are duped into the
/// slot so callers can free their copies immediately. `vision_embeddings`
/// ownership transfers into the slot when non-null (the slot will free on
/// deinit).
pub const SubmitParams = struct {
    prompt_ids: []const u32,
    /// Full original prompt for PLD lookup. When null, defaults to
    /// `prompt_ids` (PLD's lookup table = full_prompt + generated).
    full_prompt: ?[]const u32 = null,
    cached_tokens: u32 = 0,
    has_tools: bool = false,
    /// The server's final resolved thinking mode after request overrides and
    /// model defaults. DFlash economics key off this, not tool presence.
    enable_thinking: bool = false,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    max_tokens: u32,
    timeout_ns: u64 = 0,
    enable_pld: bool = false,
    enable_drafter: bool = false,
    drafter: ?*DrafterModel = null,
    /// DFlash assistant for this model. Rides the SAME `enable_drafter`
    /// request switch (a model loads at most ONE of drafter/dflash);
    /// `drafter_block_size` carries the dflash-resolved block size too.
    dflash: ?*DflashModel = null,
    drafter_block_size: u32 = 4,
    pld_draft_len: u32 = 5,
    pld_key_len: u32 = 3,
    /// Phase 2 (Plan ricky): route SDPA through `kv_quant.quantAttention`
    /// instead of dequant + dense SDPA. No effect when the cache scheme
    /// isn't `.affine`. Default false → unchanged behavior.
    kv_attn_fused: bool = false,
    /// Vision embeddings spliced at image-token positions during prefill.
    /// Ownership transferred to the slot; freed on slot.deinit.
    vision_embeddings: ?mlx.mlx_array = null,
    /// Media items in the prompt, for the prefix-cache key. Borrowed; the slot copies it.
    media: []const prefix_cache_mod.MediaSpan = &.{},
    /// Workload key for hot-cache eviction (`server.requestCacheKey`, 0 = anonymous).
    cache_key: u64 = 0,
    client: metrics_mod.Client = .other,
    logprobs_n: u32 = 0,
    /// `/v1/completions` echo: score every prompt token (cold prefill, no restore).
    prompt_logprobs: bool = false,
    /// Wave 1.A: per-request override of the process-default KV-cache quant
    /// scheme. When non-null, this slot's KVCache is constructed with this
    /// config instead of `Scheduler.kv_quant_config`. Lets one server host a
    /// single model and let clients trade accuracy for context length on a
    /// per-call basis (`{"kv_quant": "off"|4|8}` body field).
    kv_quant_config: ?transformer_mod.KVQuantConfig = null,
    /// Plan 05 Phase D: the target model for this request. The conn thread
    /// resolves this via `scheduler.ensureLoaded(id)` BEFORE submitting and
    /// keeps a refcount on it for the slot's lifetime, so the model can't
    /// be evicted mid-flight. The scheduler routes prefill/decode through
    /// `slot.model.transformer.?` (and friends) instead of `sch.xfm`, so
    /// per-tick model switching is just a pointer hop. Required field
    /// post-Phase-D; tests using the legacy path pass the default model.
    model: *model_registry_mod.LoadedModel,
};

/// Set by `server.installPrefillAdmission`: does a request of this shape fit in GPU memory
/// right now? Null (unit tests, no HTTP server) disables evict-to-admit. The trailing warm
/// arguments are the restored rows, their capacity, and whether the restore checked the
/// entry out (the only restore whose rows the request will not allocate).
/// {needed, available} of the same cold bill, for the sibling ledger.
pub var prefill_admission_numbers: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool) [2]u64 = null;

/// Bytes admitted requests were promised and have not allocated yet. A restored prefix shares
/// the cached buffers until its first append, so live memory does not show a sibling's claim
/// when the next one is billed; four such requests each fit alone and overran together.
pub const PromiseLedger = struct {
    base_active: u64 = 0,
    promised: u64 = 0,

    pub fn outstanding(self: PromiseLedger, active_now: u64) u64 {
        return self.promised -| (active_now -| self.base_active);
    }

    pub fn admit(self: *PromiseLedger, needed: u64, active_now: u64) void {
        if (self.promised == 0) self.base_active = active_now;
        self.promised +|= needed;
    }

    pub fn release(self: *PromiseLedger, needed: u64) void {
        self.promised -|= needed;
    }
};

/// A slot's promise stands only until its claim shows in live memory: the restored KV is
/// copied on the first append and the verify transients are steady a few rounds in.
pub const PROMISE_MATERIALIZED_TOKENS: usize = 64;
/// Slack a sibling's bill keeps under the ceiling: a group's verify transients grow with its
/// lanes, which no per-request bill prices.
pub const SIBLING_MARGIN_BYTES: u64 = 2 << 30;

pub fn promiseMaterialized(completion_tokens: usize) bool {
    return completion_tokens >= PROMISE_MATERIALIZED_TOKENS;
}

pub var prefill_admission_fits: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool) bool = null;

/// The prefill width this request should run at, chosen against live post-eviction memory
/// (`server.requestPrefillChunkNow`). Null keeps the model's load-time pin.
pub var prefill_request_chunk: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool) u32 = null;

pub const PostEvictionWidth = struct {
    /// The width the prefill runs at. Always the re-ask.
    width: u32,
    widened: bool,
    /// The re-ask came back narrower than the admitted width: memory moved between the reads.
    moved: bool,
};

/// The prefill width to run, given the width admission was billed at before the eviction
/// pass and the width re-asked against live memory after it. The re-ask wins in both
/// directions: `@max` would widen on memory that is gone. `admitted == 0` = no pass ran.
pub fn postEvictionPrefillChunk(admitted: u32, reasked: u32) PostEvictionWidth {
    return .{
        .width = reasked,
        .widened = admitted != 0 and reasked > admitted,
        .moved = admitted != 0 and reasked < admitted,
    };
}

/// Asked by the prefill loop: the width of the next chunk, re-priced at every chunk boundary
/// (`server.adaptivePrefillWidthNow`). Null keeps the admitted width.
pub var prefill_chunk_adapt: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u32,
    *generate_mod.AdaptiveWidthState,
    u64,
) u32 = null;

/// Whether the per-chunk adaptive width is enabled for this model. The hook above is
/// installed process-wide, so its presence is not the arch gate.
pub var prefill_chunk_adaptive_enabled: ?*const fn (*const model_mod.ModelConfig) bool = null;

/// What `Generator.InitOptions.adaptive_chunk_width` gets for a slot.
pub fn adaptiveChunkWidthFor(cfg: ?*const model_mod.ModelConfig) bool {
    const c = cfg orelse return false;
    const enabled = prefill_chunk_adaptive_enabled orelse return false;
    return enabled(c);
}

/// Re-price a widen after the interleave tick (`server.adaptivePrefillWidenStillFits`). Null declines every widen.
pub var prefill_chunk_widen_ok: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u64,
) bool = null;

/// Logs the numbers the estimator compared on a refusal and returns them, {needed, available} bytes.
pub var prefill_admission_refused_log: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool) [2]u64 = null;

/// Invalidate the published hot-cache budget on unload/switch (`server.clearResolvedPrefixCacheMem`).
pub var hot_cache_budget_invalidate: ?*const fn () void = null;

pub const SlotState = enum { pending_prefill, decoding, finished, errored };

/// Result of `Slot.waitNext`. Driven by the inference thread; consumed by
/// the connection thread.
pub const NextResult = union(enum) {
    /// Next decoded token id.
    token: u32,
    /// Generation completed (EOS, max_tokens, timeout, etc.). `finish_reason`
    /// is set on the slot at this point.
    done: void,
    /// Generation errored. `error_code` is set on the slot; the caller
    /// should surface it to the client and call `complete(slot)`.
    err: void,
};

/// Per-request state. Owned by the Scheduler from `submit` until `complete`.
pub const Slot = struct {
    allocator: std.mem.Allocator,
    /// io reference for Stopwatch / async-eval (captured from Scheduler).
    io: std.Io,

    /// Plan 05 Phase D: target model for this request. Captured from
    /// `SubmitParams.model` and borrowed for the slot's lifetime; the conn
    /// thread holds a refcount (via `scheduler.ensureLoaded`) so the
    /// pointer stays valid until `complete()`. Prefill/decode route forward
    /// passes through `model.transformer.?` (and `model.drafter`,
    /// `model.prefix_cache`). Batched decode groups slots
    /// by this field so kernels never cross model boundaries.
    model: *model_registry_mod.LoadedModel,

    // ── Per-slot model state. Owned by the slot. ──
    cache: KVCache,
    moe_seq_offset: usize,
    /// Absolute prompt-token count a prefill the client cancelled had forwarded into the KV
    /// (`Generator.InitOptions.cancelled_forwarded`); consumed by `commitCancelledPrefillSlot`.
    cancelled_forwarded: usize = 0,
    vision_embeddings: ?mlx.mlx_array,
    /// Media items in `full_prompt` (owned): the prefix-cache key of their rows.
    media: []prefix_cache_mod.MediaSpan,
    cache_key: u64 = 0,
    client: metrics_mod.Client = .other,
    request_id: u64 = 0,
    /// Hot-cache entry this request restored from (`LookupResult.entry_id`).
    restored_entry: u64 = 0,
    skip_prefix_cache: bool = false,
    /// Set while the slot is prefilled at another prefill's chunk boundary: the most uncached tokens it
    /// may forward there. A restore that covers less than the pre-filter peeked hands the request back.
    boundary_max_tail: ?usize = null,

    /// Forward context backed by the fields above. Initialized in `init`
    /// and aliased by the Generator's own `ctx` field at prefill time.
    ctx: ForwardCtx,

    /// Generator (constructed on inference thread post-prefill).
    legacy_gen: ?Generator,

    // ── Submission data. Owned by the slot, freed in deinit. ──
    prompt_ids: []u32,
    full_prompt: []u32,
    sampling: SamplingParams,
    eos_token_ids: []u32,
    max_tokens: u32,
    timeout_ns: u64,
    has_tools: bool,
    enable_thinking: bool,
    enable_pld: bool,
    enable_drafter: bool,
    drafter: ?*DrafterModel,
    dflash: ?*DflashModel,
    drafter_block_size: u32,
    pld_draft_len: u32,
    pld_key_len: u32,
    /// Phase 2 (Plan ricky): see SubmitParams.kv_attn_fused.
    kv_attn_fused: bool,
    cached_tokens: u32,
    logprobs_n: u32,
    serial_reason_logged: bool = false,
    memory_hold_logged: bool = false,
    memory_promised: u64 = 0,
    company_ticks: u8 = 0,

    // ── State + output channel. ──
    state: SlotState,

    out_mu: std.Io.Mutex,
    out_cond: std.Io.Condition,
    /// Wake signal for `waitNextTimeout` — set alongside every `out_cond`
    /// broadcast. Events support timed waits (Io.Condition does not), which
    /// is what lets the conn thread poll the peer socket during long
    /// prefills instead of blocking until the first token.
    out_event: std.Io.Event,
    out_buf: std.ArrayList(u32),
    out_idx: usize,
    finished: bool,
    error_code: ?[]const u8,
    /// {needed, available} bytes the inference thread compared when it refused this prefill, so the
    /// client's `PrefillDoesNotFit` can quote them as the connection thread's refusal does.
    refused_bill: ?[2]u64 = null,
    finish_reason: []const u8,
    /// Set ONLY by the degenerate-tail guard. The wire reason is "stop" so a
    /// client does not mistake the guard for output/context exhaustion; this
    /// sibling signal preserves the specific cause. Static string, never freed.
    finish_details: ?[]const u8,
    /// Index into the emitted tokens where the degenerate span begins;
    /// everything from here on is the loop. Non-streaming responses are cut
    /// here so the client cannot round-trip the loop into the next prompt.
    loop_trim_start: ?usize,
    cancelled: std.atomic.Value(bool),
    /// Inference thread only: the request's outcome has been counted in `Metrics`.
    metrics_recorded: bool = false,
    /// Inference-thread passes (a prefill, a decode tick) holding this slot, taken
    /// under `queue_mu`. `complete` waits it out: the handler owns sampling state
    /// the pass reads (`think_bound`, `constraint`) and frees it once `complete` returns.
    in_pass: std.atomic.Value(u32) = .init(0),
    /// Reasoning-protocol payload boundary, published by the inference thread
    /// BEFORE the token that carries it is pushed (single-writer atomics, so
    /// the conn thread reading token i already sees its span). `token_index`
    /// is the index into the generated stream; `maxInt(u32)` = not started.
    constraint_payload_at: std.atomic.Value(u32),
    constraint_payload_offset: std.atomic.Value(u32),

    // ── Stats (filled by inference thread, safe to read after finish). ──
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_tps: f64,
    decode_tps: f64,
    /// Monotonic timestamp captured in `Slot.init`, BEFORE the queue wait.
    /// Anchors the exact time-to-first-token measurement.
    request_start_ts: std.Io.Timestamp,
    /// Wall-clock nanoseconds from `request_start_ts` (request arrival) to
    /// prefill completion = queue_wait + prefill = real time-to-first-token.
    /// Captured directly when prefill finishes (never derived by subtracting
    /// decode time), so it stays exact even if a slot finishes mid-tick. Used
    /// as `real_ttft_ns` for the metrics histogram; e2e = first_token_ns + decode_ns.
    first_token_ns: u64,
    /// Wall-clock nanoseconds spent in `runPrefill` for this slot. Includes
    /// hot-prefix-cache lookup/restore and the model forward over the
    /// uncached tail. Populated by the scheduler main loop.
    prefill_ns: u64,
    /// The prefill's sampled token, already published; its next push is swallowed.
    early_first: ?u32 = null,
    /// Wall-clock nanoseconds of interleaved decode ticks hosted INSIDE this
    /// slot's prefill (chunk-boundary yields). Charged to the decoding slots
    /// that received the tokens; subtracted from this slot's `prefill_ns` so
    /// prefill_tps stays a statement about the prefill forward.
    prefill_interleaved_ns: u64,
    /// Wall-clock nanoseconds the slot spent in decode ticks. For batched
    /// decode the full tick wall-clock is added to every participating slot,
    /// so this matches the per-slot throughput a user actually observes
    /// (`completion_tokens / decode_ns`).
    decode_ns: u64,
    /// Actual generated ids (from legacy_gen.generated_ids). Shallow copy at
    /// completion so the connection thread can read them without locking.
    generated_ids: ?[]u32,
    /// pad-only flag: set by inference thread when the entire generation was
    /// token id 0. Server-level cache invalidation reads this after complete.
    was_pad_only: bool,
    /// Phase A5: per-token logprobs accumulated by the inference thread when
    /// `logprobs_n > 0`. The conn thread takes ownership via
    /// `nonStreamingViaScheduler` (or equivalent) at completion; if the
    /// caller doesn't consume, `Slot.deinit` frees the contents.
    logprobs_buf: std.ArrayList(generate_mod.LogprobResult),
    /// Prompt-token logprobs for an echo request (`SubmitParams.prompt_logprobs`),
    /// filled by prefill; taken by the conn thread like `logprobs_buf`.
    prompt_logprobs: bool,
    prompt_logprobs_buf: std.ArrayList(generate_mod.LogprobResult),

    /// Initialize but do NOT take ownership of caches — those are allocated
    /// inside `init` from the slot's allocator.
    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        config: *const ModelConfig,
        params: SubmitParams,
        kv_quant_config: transformer_mod.KVQuantConfig,
    ) !*Slot {
        const slot = try allocator.create(Slot);
        errdefer allocator.destroy(slot);

        // Per-slot KVCache, honoring the process-level kv-quant setting.
        var cache = try KVCache.initWithConfig(allocator, config.num_hidden_layers, kv_quant_config);
        errdefer cache.deinit();

        // Dup owned slices.
        const prompt_owned = try allocator.dupe(u32, params.prompt_ids);
        errdefer allocator.free(prompt_owned);
        const full_prompt_src = params.full_prompt orelse params.prompt_ids;
        const full_prompt_owned = try allocator.dupe(u32, full_prompt_src);
        errdefer allocator.free(full_prompt_owned);
        const media_owned = try allocator.dupe(prefix_cache_mod.MediaSpan, params.media);
        errdefer allocator.free(media_owned);
        const eos_owned = try allocator.dupe(u32, params.eos_token_ids);
        errdefer allocator.free(eos_owned);

        slot.* = .{
            .allocator = allocator,
            .io = io,
            .model = params.model,
            .cache = cache,
            .moe_seq_offset = 0,
            .vision_embeddings = params.vision_embeddings,
            .media = media_owned,
            .cache_key = params.cache_key,
            .client = params.client,
            .request_id = metrics_mod.nextRequestId(),
            .ctx = undefined, // set after slot is in stable storage so pointers are valid
            .legacy_gen = null,
            .prompt_ids = prompt_owned,
            .full_prompt = full_prompt_owned,
            .sampling = params.sampling,
            .eos_token_ids = eos_owned,
            .max_tokens = params.max_tokens,
            .timeout_ns = params.timeout_ns,
            .has_tools = params.has_tools,
            .enable_thinking = params.enable_thinking,
            .enable_pld = params.enable_pld,
            // An external drafter that lacks M-RoPE positions: a vision request
            // never takes a sidecar.
            .enable_drafter = assistantSidecarEnabledForRequest(params.enable_drafter, params.vision_embeddings != null),
            .drafter = params.drafter,
            .dflash = params.dflash,
            .drafter_block_size = params.drafter_block_size,
            .pld_draft_len = params.pld_draft_len,
            .pld_key_len = params.pld_key_len,
            .kv_attn_fused = params.kv_attn_fused,
            .cached_tokens = params.cached_tokens,
            .logprobs_n = params.logprobs_n,
            .state = .pending_prefill,
            .out_mu = .init,
            .out_cond = .init,
            .out_event = .unset,
            .out_buf = std.ArrayList(u32).empty,
            .out_idx = 0,
            .finished = false,
            .error_code = null,
            .finish_reason = "length",
            .finish_details = null,
            .loop_trim_start = null,
            .cancelled = std.atomic.Value(bool).init(false),
            .constraint_payload_at = std.atomic.Value(u32).init(std.math.maxInt(u32)),
            .constraint_payload_offset = std.atomic.Value(u32).init(0),
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
            .request_start_ts = std.Io.Timestamp.now(io, .boot),
            .first_token_ns = 0,
            .prefill_ns = 0,
            .prefill_interleaved_ns = 0,
            .decode_ns = 0,
            .generated_ids = null,
            .was_pad_only = true,
            .logprobs_buf = .empty,
            .prompt_logprobs = params.prompt_logprobs,
            .prompt_logprobs_buf = .empty,
            .skip_prefix_cache = params.prompt_logprobs,
        };

        // ForwardCtx points at fields owned by `slot` — must outlive the
        // Generator. The slot is heap-allocated so addresses are stable
        // until `complete` frees it.
        slot.ctx = .{
            .cache = &slot.cache,
            .moe_seq_offset = &slot.moe_seq_offset,
            .vision_embeddings = slot.vision_embeddings,
            .capture_hidden = null,
            .kv_attn_fused = params.kv_attn_fused,
        };

        return slot;
    }

    /// Free everything the slot owns. Only safe to call when no thread can
    /// observe the slot anymore (i.e. after the inference thread has
    /// finished/errored it AND the connection thread has consumed the final
    /// `done`/`err` from `waitNext`).
    pub fn deinit(self: *Slot) void {
        if (self.legacy_gen) |*gen| {
            gen.deinit(self.allocator);
        }
        self.cache.deinit();
        if (self.vision_embeddings) |ve| _ = mlx.mlx_array_free(ve);
        self.allocator.free(self.prompt_ids);
        self.allocator.free(self.full_prompt);
        self.allocator.free(self.media);
        self.allocator.free(self.eos_token_ids);
        if (self.error_code) |code| self.allocator.free(code);
        if (self.generated_ids) |g| self.allocator.free(g);
        // Free any logprobs the conn thread didn't claim. After
        // `nonStreamingViaScheduler` calls `toOwnedSlice`, items.len becomes
        // 0 so this is a no-op on the success path.
        for (self.logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.logprobs_buf.deinit(self.allocator);
        for (self.prompt_logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.prompt_logprobs_buf.deinit(self.allocator);
        self.out_buf.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Inference thread: the echo sink, emptied first, since a retried prefill starts over.
    fn freshPromptLogprobs(self: *Slot) *std.ArrayList(generate_mod.LogprobResult) {
        for (self.prompt_logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.prompt_logprobs_buf.clearRetainingCapacity();
        return &self.prompt_logprobs_buf;
    }

    /// Inference thread: record the reasoning→payload boundary so it is
    /// visible to the conn thread no later than the token that carries it.
    fn publishConstraintSpan(self: *Slot, token_index: u32, byte_offset: u32) void {
        self.constraint_payload_offset.store(byte_offset, .release);
        self.constraint_payload_at.store(token_index, .release);
    }

    /// Connection thread: where the constrained payload begins relative to
    /// the generated stream, or null while it has not begun. `token_index`
    /// matches the reader's own token count (token i carries the payload
    /// start when the returned index equals i).
    pub fn constraintPayloadStart(self: *const Slot) ?rp_mod.ConstraintSpan {
        const at = self.constraint_payload_at.load(.acquire);
        if (at == std.math.maxInt(u32)) return null;
        return .{ .token_index = at, .byte_offset = self.constraint_payload_offset.load(.acquire) };
    }

    /// Inference thread: enqueue a generated token for the consumer.
    fn pushToken(self: *Slot, t: u32) void {
        self.pushTokenWithLogprob(t, null);
    }

    /// Publish one token and, when the request asked for logprobs, the entry
    /// describing it — in ONE critical section.
    ///
    /// Both must move under `out_mu` together. The streaming path reads entry
    /// i as soon as it is handed token i, so an append outside the lock races
    /// the reader two ways: the entry may not be visible yet (a silently
    /// missing trailing logprob), and a concurrent grow reallocates the
    /// backing array under a reader mid-copy. Non-streaming consumes the whole
    /// buffer at completion and is blind to both, which is why the streaming
    /// gap survived: it is invisible to output-equality tests AND to llmprobe,
    /// which probes logprobs non-streaming only.
    fn pushTokenWithLogprob(self: *Slot, t: u32, lp: ?generate_mod.LogprobResult) void {
        // The prefill's token went out early (`publishFirstToken`): the decoder's own push of it is swallowed.
        if (self.early_first) |e| {
            self.early_first = null;
            if (e == t and lp == null) return;
        }
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (lp) |entry| {
            // Degrade to error like the token append below — and do NOT return:
            // the broadcast at the bottom is what wakes a reader blocked on the
            // condvar, so an early exit here trades an OOM for a hang.
            self.logprobs_buf.append(self.allocator, entry) catch |err| {
                self.error_code = self.allocator.dupe(u8, @errorName(err)) catch null;
                self.state = .errored;
            };
        }
        self.out_buf.append(self.allocator, t) catch |err| {
            // Allocation failure: degrade to error.
            self.error_code = self.allocator.dupe(u8, @errorName(err)) catch null;
            self.state = .errored;
        };
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Connection thread: copy logprob entries produced since `cursor` into
    /// `out`, returning the new cursor. Under `out_mu`, so it cannot observe a
    /// half-written entry or a reallocating buffer.
    ///
    /// The COPY is shallow and that is deliberate: each entry's `top_logprobs`
    /// is its own allocation, stable for the life of the slot, and owned by
    /// the slot (`Slot.deinit` frees it). The caller borrows and must not free
    /// — only the ArrayList's backing array is at risk from a concurrent grow,
    /// and that is exactly what the lock covers.
    pub fn copyLogprobsFrom(
        self: *Slot,
        allocator: std.mem.Allocator,
        cursor: usize,
        out: *std.ArrayList(generate_mod.LogprobResult),
    ) !usize {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const items = self.logprobs_buf.items;
        if (cursor >= items.len) return cursor;
        try out.appendSlice(allocator, items[cursor..]);
        return items.len;
    }

    /// Inference thread: signal normal completion. Safe to call multiple
    /// times (idempotent on `finished`).
    fn markFinished(self: *Slot, reason: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.finished) return;
        self.finished = true;
        self.state = .finished;
        self.finish_reason = reason;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Inference thread: signal error. `name` is borrowed; we dupe so the
    /// connection thread can read it after the inference loop drops the slot.
    /// Whether a slot error name names a memory failure (`OutOfMemory` from the MLX latch or Zig's allocator).
    pub fn errorNameIsMemory(name: []const u8) bool {
        return std.mem.eql(u8, name, "OutOfMemory") or
            std.mem.eql(u8, name, "InsufficientMemory");
    }

    /// Whether the slot's latched error name is exactly `name`.
    pub fn errorNameIs(self: *Slot, name: []const u8) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const have = self.error_code orelse return false;
        return std.mem.eql(u8, have, name);
    }

    pub fn errorIsMemory(self: *Slot) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const name = self.error_code orelse return false;
        return errorNameIsMemory(name);
    }

    fn markError(self: *Slot, name: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.error_code != null or self.finished) return;
        self.error_code = self.allocator.dupe(u8, name) catch null;
        self.state = .errored;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Connection thread: block until the next token, completion, or error.
    /// Returns `.token` for each generated id, then exactly one terminator
    /// (`.done` or `.err`).
    pub fn waitNext(self: *Slot) NextResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (true) {
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                return .{ .token = t };
            }
            if (self.error_code != null) return .{ .err = {} };
            if (self.finished) return .{ .done = {} };
            // Cancellation (client disconnect or server shutdown) must unblock a
            // blocked reader — `cancel()` only broadcasts; without this the
            // reader would sleep until the inference thread happened to finish
            // the slot, so shutdown could never drain in-flight requests and
            // raced `Scheduler.deinit` into a use-after-free (SIGSEGV in
            // `complete`). Buffered tokens above still drain first.
            if (self.cancelled.load(.acquire)) return .{ .done = {} };
            self.out_cond.waitUncancelable(self.io, &self.out_mu);
        }
    }

    /// Connection thread: like `waitNext`, but wakes with `null` (idle)
    /// after `timeout_ms` with no token or terminator. Lets the caller poll
    /// the peer socket and emit SSE keepalives during long prefills —
    /// Claude Code disconnects after ~60s of stream silence, and pre-2026-06
    /// the handler sat blocked in `waitNext` for the whole multi-minute
    /// prefill, never noticed the disconnect, and abandoned giant prefills
    /// piled up serially behind every client retry (the server looked dead
    /// while the GPU ground ghosts; observed live with Claude Code + a
    /// 40K-token MCP prompt on gemma-4-12b).
    pub fn waitNextTimeout(self: *Slot, timeout_ms: i64) ?NextResult {
        while (true) {
            self.out_mu.lockUncancelable(self.io);
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                self.out_mu.unlock(self.io);
                return .{ .token = t };
            }
            if (self.error_code != null) {
                self.out_mu.unlock(self.io);
                return .{ .err = {} };
            }
            if (self.finished) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Cancellation unblocks the reader promptly (see waitNext) so a
            // disconnected/shutdown request stops instead of waiting out the
            // generation — the precondition for draining conn threads before
            // teardown.
            if (self.cancelled.load(.acquire)) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Arm the event under the lock: producers mutate under this lock
            // and set() before releasing it, so a set racing our reset leaves
            // the event set and the wait below returns immediately — no lost
            // wakeups. A spurious wake just reads as an early idle (benign).
            self.out_event.reset();
            self.out_mu.unlock(self.io);
            self.out_event.waitTimeout(self.io, .{ .duration = .{
                .raw = .fromMilliseconds(timeout_ms),
                .clock = .awake,
            } }) catch return null;
        }
    }

    /// Connection thread: signal cancellation. The inference thread will
    /// drop this slot at the next tick boundary.
    pub fn cancel(self: *Slot) void {
        self.cancelled.store(true, .release);
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }
};

/// Phase: embedding work item. Conn thread fills
/// `token_seqs` and calls `Scheduler.computeEmbeddings`; the inference
/// thread services the request via `generate.computeEmbeddingsBatch` — one
/// padded, key-masked GPU forward per EMBED_MAX_BATCH chunk — and writes
/// the float vectors into `results` (caller frees).
pub const EmbedRequest = struct {
    /// Plan 05 Phase D: target model whose `transformer` services this
    /// request. The conn thread holds a refcount for the duration.
    model: *model_registry_mod.LoadedModel,
    /// Tokenized inputs, one slice per text. Borrowed; must outlive the call.
    token_seqs: []const []const u32,
    /// Output: one pooled L2-normalized embedding per input on success.
    /// Rows + outer slice owned by `allocator`; caller frees.
    results: ?[][]f32 = null,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Plan 05 Phase D: cold-load work item. Posted by `Scheduler.ensureLoaded`
/// when a request targets an `.unloaded` (or freshly-evicted) entry; the
/// inference thread drains the queue between ticks. The conn thread parses
/// `config.json` / tokenizer / chat_config on its own thread (CPU only,
/// no mlx ops) and hands the pre-parsed CPU state to the inference thread,
/// which does the mlx-allocating work (weights + Transformer + vision +
/// drafter + JIT + warmup) and installs everything on `entry`.
///
/// Eviction: when set, `evict_entry` is unloaded on the inference thread
/// BEFORE the new load starts. The conn thread has already marked the
/// victim `.evicting` and waited for refcount == 0, so freeing GPU memory
/// is safe.
pub const LoadRequest = struct {
    /// Target entry to populate. Already transitioned to `.loading` by
    /// the conn thread before posting; the inference thread completes the
    /// load and calls `registry.markReadyLocked(entry, bytes)` or
    /// `markErrorLocked` on failure.
    entry: *LoadedModel,
    /// Pre-parsed CPU state. Ownership transfers to `entry` on success;
    /// on failure the conn thread takes them back via the `done` cond-var
    /// and frees them.
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,

    /// Borrowed paths. Conn thread keeps the buffers alive until `done`.
    model_dir: []const u8,
    resident_model_bytes: ?u64 = null,
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,

    warmup_eager: bool = true,
    draft_block_size: u32 = 4,
    draft_block_size_explicit: bool = false,
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense,
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    prefix_cache_mem_bytes: u64 = 0,
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64 = null,
    /// SSD tier byte budget (mirrors `LoadParams.prefix_cache_disk_bytes`).
    prefix_cache_disk_bytes: u64 = 0,
    /// Iteration 2: tokenize cache LRU capacity. Mirrored on
    /// `LoadParams.tokenize_cache_entries`; both paths feed
    /// `doLoadOnInferenceThread`.
    tokenize_cache_entries: u32 = 4,

    /// Victims to evict before the load (LRU-selected by the planner). Each is
    /// already marked `.evicting` with refcount == 0 by the conn thread under
    /// registry.mutex. The inference thread calls `unloadResident()` on each to
    /// free GPU memory, drops its resident-bytes accounting via
    /// `registry.accountEvictedLocked`, then `registry.finalizeEvictionLocked`.
    /// Borrows the conn thread's stack buffer; valid until `done`.
    evict_entries: []*LoadedModel = &.{},

    /// Output: error name on failure (owned by `allocator`; conn thread
    /// frees). Null on success.
    error_name: ?[]const u8 = null,

    /// Conn-thread synchronization. Inference thread broadcasts when done.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Model-unload work item. Posted by `unloadModelIfIdle` after the conn thread
/// marked the entry `.evicting` and drained its refcount. The inference thread
/// frees the entry's resident mlx state (stream-bound) and finalizes the
/// eviction accounting, then the entry returns to `.unloaded` (the stub stays
/// in the registry so it can reload later).
pub const UnloadRequest = struct {
    entry: *model_registry_mod.LoadedModel,
    done: bool = false,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Continuous-batching scheduler. One per server. Owns the inference
/// thread, the queue of in-flight slots, AND (post-A1) the loaded model
/// state - Transformer + weights + drafter all live here,
/// allocated on the inference thread so mlx's thread-local GPU stream is
/// bound on the right thread from the start.
pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    // ── Plan 05 — multi-model state. Source of truth for model fields is
    //    `current_model` (a borrowed *LoadedModel owned by the registry).
    //    The fields below (`xfm`, `weights`, …) are *borrowed views* into
    //    `current_model` set at load time so existing scheduler-internal
    //    code can keep reading them as fields. Phase D will refresh these
    //    views on every model swap inside `pickNextTickWork`.
    registry: *ModelRegistry,
    current_model: ?*LoadedModel,

    // ── Borrowed views (non-owning). Cleared on shutdown via
    //    `clearCurrentModelViews`. Null only before load completes or after
    //    an eviction in Phase D.
    xfm: ?*Transformer,
    weights: ?*Weights,
    drafter: ?*DrafterModel,
    dflash: ?*DflashModel = null,
    drafter_block_size: u32,
    kv_quant_config: transformer_mod.KVQuantConfig,
    /// Launch-flag prefix-cache settings, retained so COLD-LOADED models
    /// (`ensureLoaded`, model switches) get the same
    /// prefix-cache behavior as the `--model` primary. Pre-plumbing these
    /// were hardcoded to (1, 0, stride 0) on the cold path, which silently
    /// crippled warm reuse — and disabled it entirely on hybrids — after
    /// every model switch.
    prefix_cache_capacity: u32,
    prefix_cache_ram_enabled: bool,
    prefix_cache_mem_bytes: u64,
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64,
    prefix_cache_disk_bytes: u64,
    /// Launch-flag drafter settings, retained for cold loads. `--no-drafter`
    /// became load-bearing on this path the moment `dflash.resolveInDirDrafter`
    /// started probing `<model_dir>/drafter` at load: without it here, a server
    /// launched with speculation off re-enabled it on every model switched to.
    /// `drafter_dir` is the launch `--drafter` path and `primary_model_dir` the
    /// `--model` it belongs to — see `coldLoadDrafterDir` for why it is not
    /// simply copied across.
    no_drafter: bool,
    drafter_dir: []const u8,
    primary_model_dir: []const u8,
    draft_block_size: u32,
    draft_block_size_explicit: bool,

    // ── Borrowed refs (CPU-only state owned by the LoadedModel). ──
    drafter_path: []const u8,

    /// Phase A6 → Plan 05: per-model hot prefix cache. Pre-Plan-05 this was
    /// a server-owned global; Plan 05 moves it onto `LoadedModel` so each
    /// model gets isolation by construction. Borrowed view here is the
    /// current model's cache (or null when the cache isn't applicable for
    /// the model, e.g. hybrid recurrent-state archs).
    hot_prefix_cache: ?*prefix_cache_mod.HotPrefixCache,
    /// Resident hot-cache bytes, published for the connection thread: `hot_prefix_cache` is
    /// inference-thread state, freed on every model switch, so the guard reads this number
    /// and never the pointer.
    resident_hot_cache_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// KV + recurrent state the decoding slots own, published once per tick (`/props`).
    resident_live_kv_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Per-slot context snapshot for `/metrics.json`, published with the above under `queue_mu`.
    live_sessions: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined,
    live_session_count: usize = 0,
    /// Every ready model's hot-cache entries for `/metrics.json`, under `digest_mu`.
    cached_sessions: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined,
    cached_session_count: usize = 0,

    /// The part of the above an eviction can prove it will return (residency minus the largest entry).
    reclaimable_hot_cache_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Set on unload: the OS hands freed pages back lazily, so the budget revise repeats
    /// before each prefill batch while this is armed and settles as the ceiling recovers.
    budget_revise_sw: ?io_util.Stopwatch = null,
    promise_ledger: PromiseLedger = .{},
    /// Inference thread only. Greater than 0 while a prefill started at a chunk boundary runs: it may
    /// tick decoders at its own boundaries but never admits.
    boundary_depth: u8 = 0,
    /// The prefill whose chunk boundary is hosting a nested admission; counted as company for memory.
    boundary_outer: ?*Slot = null,
    /// The request being prefilled at that boundary; in no queue, so `cancelAllInFlight` reads it here.
    /// Guarded by `queue_mu`.
    boundary_slot: ?*Slot = null,

    /// Per-entry digest snapshot the connection-thread guard reads instead of the cache.
    /// Replaced under `digest_mu` by the inference thread; readers copy under the lock.
    hot_cache_digests: []prefix_cache_mod.HotPrefixCache.EntryDigest = &.{},
    /// The residency the digests describe, published in the same critical section.
    digest_residency: u64 = 0,
    digest_mu: std.Io.Mutex = .init,

    max_concurrent: u32,
    /// Phase A7 test hook: when true, `runDecodeTick` forces the batched
    /// kernel even at `active.len == 1`. Set via the `MLX_SERVE_FORCE_BATCHED`
    /// environment variable (`=1` to enable). Test-only — production uses
    /// the auto-gate that drops to `runSingleDecodeTick` for single-slot
    /// requests because that path is bit-identical to legacy and supports
    /// speculative decoding (which the batched kernel doesn't).
    force_batched: bool,

    queue_mu: std.Io.Mutex,
    queue_cond: std.Io.Condition,
    pending: std.ArrayList(*Slot),
    decoding: std.ArrayList(*Slot),
    /// Pending embedding requests (encoder-only models). The inference thread
    /// drains these in the gap before/after each prefill+decode tick. Posting
    /// broadcasts on `queue_cond` to wake an idle inference thread.
    embed_queue: std.ArrayList(*EmbedRequest),
    /// Phase D: pending cold-load requests. Conn threads post here via
    /// `scheduler.ensureLoaded`; the inference thread drains between ticks
    /// (load runs after cleanup + embed, before prefill). Multiple
    /// concurrent requesters for the same id share one load via the
    /// `.loading` state on the entry — `ensureLoaded` only posts when it
    /// successfully flips the entry from `.unloaded` → `.loading`.
    load_queue: std.ArrayList(*LoadRequest),
    /// Pending model-unload jobs. Conn threads post here via `unloadModelIfIdle`
    /// after marking the entry `.evicting` + draining its refcount; the
    /// inference thread frees the mlx state (stream-bound, like cleanup).
    unload_queue: std.ArrayList(*UnloadRequest),
    /// Slots awaiting cleanup. The conn thread queues a slot here in
    /// `complete()` instead of calling `slot.deinit()` directly — `deinit`
    /// frees mlx_arrays via refcount-decrement, and the underlying GPU
    /// memory release races against the inference thread's stream. The
    /// inference thread drains this queue between ticks where it owns the
    /// stream binding, so all mlx ops stay on one thread.
    cleanup_queue: std.ArrayList(*Slot),
    /// Metrics sink. Null when --metrics is off. Populated from LoadParams.
    /// Read once per REQUEST in `finishSlot` — never on the per-token path.
    metrics: ?*metrics_mod.Metrics,
    /// In-flight generated-token aggregate: the sum of `completion_tokens`
    /// over the slots still decoding, republished by the inference thread once
    /// per decode tick (O(1) at the tick boundary, NOT per token). The gauge
    /// sampler reads this race-free to derive a live tok/s — it never touches
    /// per-slot fields off-thread. Zero when nothing is decoding.
    inflight_generated_tokens: std.atomic.Value(u64),
    /// Tokens forwarded so far by the prefill currently running on the
    /// inference thread; 0 when no prefill is in flight. Mirror of
    /// `inflight_generated_tokens` for the OTHER phase — without it the metrics
    /// panel shows nothing while a multi-minute prefill pins the GPU, because
    /// prompt-token counters and prefill-time histograms only advance when the
    /// request finishes. Written per prefill CHUNK, read by the gauge sampler.
    inflight_prefill_tokens: std.atomic.Value(u64),
    /// Post-cache tail the in-flight prefill will forward, same scale as
    /// `inflight_prefill_tokens`; 0 when none is running.
    inflight_prefill_expected: std.atomic.Value(u64),
    /// Number of slots currently inside `runPrefill`. Set on entry, cleared on
    /// every exit — so the panel can say "prefilling" IMMEDIATELY, rather than
    /// waiting for the first 8192-token chunk to land (~40 s on a 27B).
    requests_prefilling: std.atomic.Value(u64),
    /// Counts `pending.len + decoding.len` for back-pressure.
    in_flight: u32,
    /// Capacity for back-pressure. `submit` waits when in_flight >= cap.
    /// `cap = max_concurrent + queue_depth`. queue_depth = 32 hardcoded
    /// (matches the legacy `max_queue_size`).
    queue_cap: u32,
    submit_cond: std.Io.Condition,

    inference_thread: ?std.Thread,
    shutdown: std.atomic.Value(bool),
    /// The shutdown flush's deadline on the `nowMsMonotonic` clock; 0 = off. Set before `shutdown`.
    shutdown_deadline_ms: std.atomic.Value(i64) = .init(0),
    /// The batch the inference thread took out of `pending` and is prefilling (`queue_mu`): in
    /// neither queue, so `cancelAllInFlight` reads it here or SIGTERM waits out the whole prefill.
    /// Live entries are `[prefilling_from, n_prefilling)`; each holds `in_pass` while listed.
    prefilling: [16]*Slot = undefined,
    prefilling_from: usize = 0,
    n_prefilling: usize = 0,
    started: std.atomic.Value(bool),
    started_mu: std.Io.Mutex,
    started_cond: std.Io.Condition,

    /// Set true by the inference thread if model load fails. Read by `init`
    /// after `started` is signaled to decide whether to surface a load error.
    load_failed: std.atomic.Value(bool),
    /// Owned, dupe'd error name (e.g. "MissingVisionWeights"). null on
    /// success. Freed in `deinit`.
    load_error_name: ?[]const u8,

    /// Construct a Scheduler whose inference thread loads the model. Returns
    /// only after load + (optional) warmup completes. On load failure, returns
    /// `error.LoadFailed`; the inference thread has already cleaned up any
    /// partially-allocated mlx state by then.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        params: LoadParams,
        max_concurrent: u32,
    ) !*Scheduler {
        const self = try allocator.create(Scheduler);
        errdefer allocator.destroy(self);

        const cap = if (max_concurrent == 0) 1 else max_concurrent;
        // Phase A7: force-batched test hook. The byte-equivalence test sets
        // `MLX_SERVE_FORCE_BATCHED=1` to verify that the batched-kernel
        // output matches the single-slot path token-for-token at temp=0,
        // single client. Uses libc getenv to stay allocator-free.
        const force_batched = blk: {
            const raw = std.c.getenv("MLX_SERVE_FORCE_BATCHED");
            if (raw == null) break :blk false;
            const slice = std.mem.sliceTo(raw.?, 0);
            break :blk std.mem.eql(u8, slice, "1");
        };
        if (force_batched) {
            log.info("[scheduler] force_batched=on (MLX_SERVE_FORCE_BATCHED=1) — single-slot ticks will route through batched kernel\n", .{});
        }
        self.* = .{
            .allocator = allocator,
            .io = io,
            .registry = params.registry,
            .current_model = null,
            .xfm = null,
            .weights = null,
            .drafter = null,
            .drafter_block_size = params.draft_block_size,
            .kv_quant_config = params.kv_quant_config,
            .prefix_cache_capacity = params.prefix_cache_capacity,
            .prefix_cache_ram_enabled = params.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = params.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = params.prefix_cache_mem_resolver,
            .prefix_cache_disk_bytes = params.prefix_cache_disk_bytes,
            .no_drafter = params.no_drafter,
            .drafter_dir = params.drafter_dir,
            .primary_model_dir = params.model_dir,
            .draft_block_size = params.draft_block_size,
            .draft_block_size_explicit = params.draft_block_size_explicit,
            .drafter_path = params.drafter_dir,
            .hot_prefix_cache = null,
            .max_concurrent = cap,
            .force_batched = force_batched,
            .queue_mu = .init,
            .queue_cond = .init,
            .pending = std.ArrayList(*Slot).empty,
            .decoding = std.ArrayList(*Slot).empty,
            .embed_queue = std.ArrayList(*EmbedRequest).empty,
            .load_queue = std.ArrayList(*LoadRequest).empty,
            .unload_queue = std.ArrayList(*UnloadRequest).empty,
            .cleanup_queue = std.ArrayList(*Slot).empty,
            .metrics = params.metrics,
            .inflight_generated_tokens = std.atomic.Value(u64).init(0),
            .inflight_prefill_tokens = std.atomic.Value(u64).init(0),
            .inflight_prefill_expected = std.atomic.Value(u64).init(0),
            .requests_prefilling = std.atomic.Value(u64).init(0),
            .in_flight = 0,
            .queue_cap = cap + 32,
            .submit_cond = .init,
            .inference_thread = null,
            .shutdown = std.atomic.Value(bool).init(false),
            .started = std.atomic.Value(bool).init(false),
            .started_mu = .init,
            .started_cond = .init,
            .load_failed = std.atomic.Value(bool).init(false),
            .load_error_name = null,
        };

        const ctx = ThreadCtx{ .scheduler = self, .params = params };
        self.inference_thread = try std.Thread.spawn(.{}, inferenceLoop, .{ctx});

        // Wait until inference thread has loaded the model and (optionally)
        // warmed up. submit() relies on xfm/weights being live.
        self.started_mu.lockUncancelable(io);
        defer self.started_mu.unlock(io);
        while (!self.started.load(.acquire)) {
            self.started_cond.waitUncancelable(io, &self.started_mu);
        }

        if (self.load_failed.load(.acquire)) {
            // Inference thread already exited cleanly. Join + free + bubble up.
            if (self.inference_thread) |t| t.join();
            self.inference_thread = null;
            const name = self.load_error_name orelse "unknown";
            log.err("[scheduler] model load failed: {s}\n", .{name});
            return error.LoadFailed;
        }

        return self;
    }

    pub fn deinit(self: *Scheduler) void {
        self.shutdown.store(true, .release);
        // Wake inference thread if it's waiting on queue_cond.
        self.queue_mu.lockUncancelable(self.io);
        self.queue_cond.broadcast(self.io);
        self.submit_cond.broadcast(self.io);
        self.queue_mu.unlock(self.io);

        if (self.inference_thread) |t| t.join();

        // Freed below the join: the publisher is gone, so this is the last writer.
        if (self.hot_cache_digests.len > 0) {
            self.allocator.free(self.hot_cache_digests);
            self.hot_cache_digests = &.{};
        }

        // Drain any leftover slots — should be empty if all conn threads
        // called `complete` properly, but defensive. Inference thread has
        // already exited by now (joined above), so freeing here is safe.
        for (self.pending.items) |slot| slot.deinit();
        self.pending.deinit(self.allocator);
        for (self.decoding.items) |slot| slot.deinit();
        self.decoding.deinit(self.allocator);
        for (self.cleanup_queue.items) |slot| slot.deinit();
        self.cleanup_queue.deinit(self.allocator);
        // The embed queue should be empty (computeEmbedding blocks until done)
        // but guard against shutdown-mid-encode by signaling done with an error.
        for (self.embed_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.embed_queue.deinit(self.allocator);
        // Phase D: signal any pending cold-load requesters that the
        // server is shutting down. They roll back their entry state and
        // free pre-loaded CPU resources.
        for (self.load_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.load_queue.deinit(self.allocator);
        // Wake any conn threads blocked in unloadModelIfIdle. The jobs never
        // ran, but we must release them so they don't hang.
        for (self.unload_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.unload_queue.deinit(self.allocator);

        // Plan 05: mlx-allocating state lives on the `LoadedModel` owned by
        // the registry. We can't free the entries here (registry teardown
        // happens later, after serve() returns), but we DO need to release
        // their mlx pieces while we still have a thread bound to the mlx
        // GPU stream — that's actually the calling thread, since the
        // existing pattern frees mlx_array refcount-zeros from
        // `Scheduler.deinit` directly. Walk EVERY .ready entry in the
        // registry (multi-model: more than just `current_model`).
        {
            self.registry.mutex.lockUncancelable(self.io);
            defer self.registry.mutex.unlock(self.io);
            var it = self.registry.entries.valueIterator();
            while (it.next()) |entry_ptr| {
                const entry = entry_ptr.*;
                if (entry.state == .ready or entry.state == .evicting) {
                    entry.unloadResident();
                }
            }
        }
        self.current_model = null;
        // Clear the borrowed views so post-shutdown reads (defensive) see
        // null rather than dangling pointers.
        self.xfm = null;
        self.weights = null;
        self.drafter = null;
        self.dflash = null;
        self.hot_prefix_cache = null;
        self.resident_hot_cache_bytes.store(0, .monotonic);
        if (hot_cache_budget_invalidate) |f| f();
        self.reclaimable_hot_cache_bytes.store(0, .monotonic);
        if (self.load_error_name) |n| self.allocator.free(n);

        self.allocator.destroy(self);
    }

    /// Decoders a tick would advance now; finished slots linger in `decoding` until the cull.
    fn liveDecodingCount(self: *Scheduler) usize {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        var n: usize = 0;
        for (self.decoding.items) |s| {
            if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
            n += 1;
        }
        return n;
    }

    /// Another request is queued, prefilling or decoding right now.
    fn decodingCount(self: *Scheduler) usize {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        return self.decoding.items.len;
    }

    /// Another request this one could batch with is live.
    pub fn hasCompany(self: *Scheduler) bool {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        return self.in_flight > 0;
    }

    /// Submit a new request. Builds a Slot, queues it, returns the handle.
    /// Blocks if the queue is full (i.e. `in_flight >= queue_cap`). When
    /// the scheduler is shutting down this returns `error.Shutdown`.
    pub fn submit(self: *Scheduler, params: SubmitParams) !*Slot {
        // Construct the slot up front so we don't hold the queue mutex
        // through any allocation. Per-request `kv_quant_config` override (Wave
        // 1.A) wins over the process-level default carried on the scheduler.
        // Phase D fix: use the slot's target-model config (not the
        // scheduler's startup-model config) so per-slot state allocation
        // (KVCache shape, state entries) matches the model that will
        // actually run the request. Critical when two models with
        // different architectures (e.g. pure-attention + hybrid recurrent-state)
        // share one scheduler.
        const slot_config: *const ModelConfig = params.model.config orelse return error.ModelNotReady;
        // Every decode path reads the prompt's last token; `""` on a BOS-less vocab encodes to none.
        if (params.prompt_ids.len == 0) return error.EmptyPrompt;
        const eff_kv_quant = params.kv_quant_config orelse slot_config.kv_quant_override orelse self.kv_quant_config;
        const slot = try Slot.init(self.allocator, self.io, slot_config, params, eff_kv_quant);
        errdefer slot.deinit();

        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);

        while (self.in_flight >= self.queue_cap and !self.shutdown.load(.acquire)) {
            self.submit_cond.waitUncancelable(self.io, &self.queue_mu);
        }
        if (self.shutdown.load(.acquire)) return error.Shutdown;

        try self.pending.append(self.allocator, slot);
        self.in_flight += 1;
        self.queue_cond.broadcast(self.io);
        return slot;
    }

    /// Hand the slot off to the inference thread for cleanup, and notify any
    /// submitter waiting for queue space. Must be called once per slot
    /// returned by `submit`. Safe whether or not the slot finished normally.
    ///
    /// Why not free here: `slot.deinit()` walks the per-slot KVCache and
    /// calls `mlx_array_free` on each entry. Refcount-shared GPU memory
    /// release queues work against the array's owning stream, which is the
    /// inference thread's. Freeing from a conn thread without that stream
    /// binding crashes mlx 0.31.2 ("no Stream(gpu, N) in current thread").
    /// So we remove the slot from any active list (so the inference thread
    /// stops touching it) and queue it for cleanup on the inference thread.
    pub fn complete(self: *Scheduler, slot: *Slot) void {
        // Mark cancelled so any in-flight tick filters this slot out of its
        // active list before we remove it from `decoding`. Idempotent.
        slot.cancelled.store(true, .release);

        self.queue_mu.lockUncancelable(self.io);

        // Remove from pending (rare — only if conn thread cancels before
        // prefill) and from decoding (the common case). After this, only the
        // cleanup queue references the slot, so the next inference-thread
        // cleanup drain can safely deinit it.
        var i: usize = 0;
        while (i < self.pending.items.len) : (i += 1) {
            if (self.pending.items[i] == slot) {
                _ = self.pending.orderedRemove(i);
                break;
            }
        }
        i = 0;
        while (i < self.decoding.items.len) : (i += 1) {
            if (self.decoding.items[i] == slot) {
                _ = self.decoding.orderedRemove(i);
                break;
            }
        }

        // Out of every list, so no new pass can take it; wait out the one that has it,
        // before the cleanup queue owns (and may free) the slot.
        if (slot.in_pass.load(.acquire) != 0) {
            self.queue_mu.unlock(self.io);
            while (slot.in_pass.load(.acquire) != 0) std.Io.sleep(self.io, .fromMilliseconds(1), .real) catch {};
            self.queue_mu.lockUncancelable(self.io);
        }

        self.cleanup_queue.append(self.allocator, slot) catch {
            // OOM on the cleanup list — fall back to inline deinit. This
            // races on mlx but is strictly better than the leak; the slot
            // is no longer referenced from pending/decoding above.
            self.queue_mu.unlock(self.io);
            slot.deinit();
            self.queue_mu.lockUncancelable(self.io);
        };
        if (self.in_flight > 0) self.in_flight -= 1;
        self.queue_cond.broadcast(self.io); // wake inference thread to drain
        self.submit_cond.broadcast(self.io); // wake any blocked submitter
        self.queue_mu.unlock(self.io);
    }

    /// Shutdown helper: signal every in-flight slot (pending + decoding) to
    /// cancel so their owning connection threads unblock from `waitNext` and
    /// run their `defer complete(...)` promptly. `server.serve` calls this when
    /// the accept loop exits, THEN waits for the connection threads to drain
    /// before returning (which triggers `deinit`) — otherwise a thread still in
    /// `complete()` races `deinit`'s teardown of `pending`/`decoding`/
    /// `cleanup_queue` into a use-after-free. Does NOT remove or free slots;
    /// the conn threads own that via `complete()`.
    pub fn cancelAllInFlight(self: *Scheduler) void {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        for (self.pending.items) |slot| slot.cancel();
        if (self.boundary_slot) |slot| slot.cancel();
        if (self.prefilling_from < self.n_prefilling) {
            for (self.prefilling[self.prefilling_from..self.n_prefilling]) |slot| slot.cancel();
        }
        for (self.decoding.items) |slot| slot.cancel();
    }

    /// Plan 05 Phase D: resolve `id_or_empty` ("" / "lesto" / "mlx-serve" → default)
    /// to a refcounted, ready `*LoadedModel`. Cold-loads on demand: if the
    /// entry is `.unloaded`, parses CPU state, picks an LRU victim if
    /// over caps, and posts a `LoadRequest` to the inference thread,
    /// blocking until the load completes.
    ///
    /// Caller MUST call `release(lm)` once done.
    ///
    /// Errors:
    ///   error.UnknownModelId    — id isn't in the registry.
    ///   error.NoDefaultModel    — id empty AND no default set.
    ///   error.NotEnoughMemory   — would exceed caps and no LRU victim.
    ///   error.InsufficientMemory — memory preflight refused the load (free
    ///                             RAM can't hold weights + headroom).
    ///   error.LoadFailed        — inference thread reported a load failure.
    ///   error.Shutdown          — scheduler is shutting down.
    pub fn ensureLoaded(self: *Scheduler, id_or_empty: []const u8) !*LoadedModel {
        // A lost GPU context fails every MLX call: nothing is loaded or served until a restart.
        if (mlx.gpuContextLost()) return error.GpuContextLost;
        // Fast path: ready entries. registry.ensureLoaded handles waiting
        // out .loading / .evicting transitions by other callers.
        const fast_result = self.registry.ensureLoaded(id_or_empty);
        if (fast_result) |lm| return lm else |err| switch (err) {
            error.NotLoaded => {}, // fall through to slow path
            else => return err,
        }

        // Slow path: cold load. Resolve the entry. Re-acquire mutex and
        // re-check state — between the fast-path call and now another
        // caller could have completed the load.
        const entry = try self.registry.resolveEntry(id_or_empty);

        // CPU-only pre-load: parse config / load tokenizer / load chat
        // config. Cheap (~tens of ms); kept outside the mutex so other
        // requests on other models stay unblocked. On failure, mark the
        // entry `.error_state` so /v1/models surfaces the failure (and
        // future ensureLoaded calls fail fast instead of re-tripping the
        // same parse error). FileNotFound / parse errors land here.
        var settings = model_settings.overrideFor(self.allocator, self.io, entry.path);
        defer settings.deinit(self.allocator);
        const cpu_state = preloadCpuState(self.allocator, self.io, entry.path) catch |err| {
            self.registry.mutex.lockUncancelable(self.io);
            self.registry.markErrorLocked(entry, @errorName(err));
            self.registry.mutex.unlock(self.io);
            return error.LoadFailed;
        };
        // Ownership: on success transfers to the entry inside the
        // inference thread's `doLoadOnInferenceThread`; on any error from
        // here on, free them ourselves before returning.
        var owned = cpu_state;
        var owned_active: bool = true;
        defer if (owned_active) freeCpuState(self.allocator, &owned);
        applyModelSettings(owned.config, owned.chat_config, &settings);

        // Victims selected by the eviction planner (multi-victim: one load may
        // need to free several models to fit). Lives on this stack frame; the
        // slice handed to the LoadRequest stays valid while we block on `done`.
        var victims_buf: [16]*LoadedModel = undefined;
        var n_victims: usize = 0;
        // The requester's reference, taken when the load is claimed: a model that turns `.ready`
        // with no reference is another load's free eviction victim before this thread wakes.
        var pinned = false;
        errdefer if (pinned) self.registry.releaseStatus(entry);

        // Peeked OUTSIDE the registry mutex — it stats the model dir, and no
        // other load should block on our filesystem.
        const validated_model_bytes: ?u64 = residentModelDiskBytes(self.io, entry.path, owned.config) catch |err| {
            self.registry.mutex.lockUncancelable(self.io);
            if (entry.state == .unloaded) self.registry.markErrorLocked(entry, @errorName(err));
            self.registry.mutex.unlock(self.io);
            return error.LoadFailed;
        };
        const resident_model_bytes = residentGateBytes(validated_model_bytes, entry.bytes_on_disk);

        // ── Stage 1 (registry mutex): claim .loading, plan eviction.
        {
            self.registry.mutex.lockUncancelable(self.io);
            errdefer self.registry.mutex.unlock(self.io); // bail on early returns

            // Wait out any concurrent .loading / .evicting state.
            wait_loop: while (true) {
                switch (entry.state) {
                    .ready => {
                        _ = entry.refcount.fetchAdd(1, .acq_rel);
                        self.registry.mutex.unlock(self.io);
                        return entry;
                    },
                    .loading, .evicting => {
                        self.registry.state_cond.waitUncancelable(self.io, &self.registry.mutex);
                        continue :wait_loop;
                    },
                    .error_state => {
                        const load_err = ModelRegistry.loadErrorFromName(entry.error_name);
                        self.registry.mutex.unlock(self.io);
                        return load_err;
                    },
                    .unloaded => break :wait_loop,
                }
            }

            // Claim the slot.
            std.debug.assert(self.registry.tryBeginLoadLocked(entry));
            // From the claim: while victims unload or the queue waits, not the last load's numbers.
            entry.load_progress.start(io_util.nowMsMonotonic(self.io));
            _ = entry.refcount.fetchAdd(1, .acq_rel);
            pinned = true;

            // Estimate post-load bytes (see `gateEstimateBytes` for why a media
            // entry cannot be billed by its directory's size).
            const estimated: u64 = gateEstimateBytes(resident_model_bytes, owned.config.num_hidden_layers, owned.config.hidden_size);

            // Reserve this load's estimate BEFORE planning eviction, so a
            // concurrent loader sees the pending allocation in its own gate.
            // Without this, two loads can both read a stale resident total
            // (one's bytes not yet committed at markReady), both skip eviction,
            // and oversubscribe GPU memory → Metal OOM → process crash.
            self.registry.reserveLoadLocked(entry, estimated);

            // Evict LRU victims until both caps hold for this reservation
            // (multi-victim). On failure — every other resident model is pinned
            // by an in-flight request — roll back and surface a 503 instead of
            // loading anyway and crashing.
            const n = self.registry.planEvictionsLocked(entry.id, null, &victims_buf, mlxActiveBytes()) orelse {
                // Name the numbers. A refusal that logs NOTHING sends the user
                // hunting for a concurrent request that does not exist: on an
                // idle server the cause is always the static cap (#126), and
                // the flag that moves it is not otherwise discoverable.
                const gb = 1024.0 * 1024.0 * 1024.0;
                if (loadExceedsMemCap(estimated, self.registry.max_resident_mem)) {
                    log.err("Refusing to load {s}: needs ~{d:.2} GB, more than --max-resident-mem {d:.2} GB on its own, so unloading other models cannot help. Raise the cap.\n", .{
                        entry.id,
                        @as(f64, @floatFromInt(estimated)) / gb,
                        @as(f64, @floatFromInt(self.registry.max_resident_mem)) / gb,
                    });
                } else log.err("Refusing to load {s}: needs ~{d:.2} GB with {d:.2} GB resident ({d:.2} GB live, {d:.2} GB reserved by loads in flight, this one included) across {d} model(s), and unloading the idle ones does not fit it under --max-resident-models {d} and --max-resident-mem {d:.2} GB (0 = off). Unload a model or raise a cap.\n", .{
                    entry.id,
                    @as(f64, @floatFromInt(estimated)) / gb,
                    @as(f64, @floatFromInt(self.registry.current_resident_bytes)) / gb,
                    @as(f64, @floatFromInt(mlxActiveBytes())) / gb,
                    @as(f64, @floatFromInt(self.registry.reserved_bytes)) / gb,
                    self.registry.countLoadedLocked(),
                    self.registry.max_resident_models,
                    @as(f64, @floatFromInt(self.registry.max_resident_mem)) / gb,
                });
                self.registry.markUnloadedLocked(entry); // releases the reservation
                self.registry.mutex.unlock(self.io);
                return error.NotEnoughMemory;
            };
            n_victims = n;
            // Drain readers on each victim before the inference thread frees it.
            for (victims_buf[0..n_victims]) |v| self.registry.waitForRefcountZeroLocked(v);
            self.registry.mutex.unlock(self.io);
        }

        // ── Stage 2: build + post LoadRequest, wait for completion.
        var req = LoadRequest{
            .entry = entry,
            .config = owned.config,
            .tok = owned.tok,
            .chat_config = owned.chat_config,
            .model_dir = entry.path,
            .resident_model_bytes = resident_model_bytes,
            // `--no-drafter` / `--drafter` / `--draft-block-size` reach cold
            // loads too; the path itself is scoped by `coldLoadDrafterDir`.
            .drafter_dir = coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path),
            .no_drafter = self.no_drafter,
            .warmup_eager = true,
            .draft_block_size = self.draft_block_size,
            .draft_block_size_explicit = self.draft_block_size_explicit,
            .kv_quant_config = self.kv_quant_config,
            // Cold loads get the SAME prefix-cache configuration as the
            // startup model — pre-plumbing these were (1, 0, stride 0),
            // which silently degraded warm reuse after every model switch.
            .prefix_cache_capacity = self.prefix_cache_capacity,
            .prefix_cache_ram_enabled = self.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = self.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = self.prefix_cache_mem_resolver,
            .prefix_cache_disk_bytes = self.prefix_cache_disk_bytes,
            .evict_entries = victims_buf[0..n_victims],
            .allocator = self.allocator,
        };

        {
            self.queue_mu.lockUncancelable(self.io);
            defer self.queue_mu.unlock(self.io);
            if (self.shutdown.load(.acquire)) return error.Shutdown;
            try self.load_queue.append(self.allocator, &req);
            self.queue_cond.broadcast(self.io);
        }

        // Block until the inference thread signals done.
        req.done_mu.lockUncancelable(self.io);
        while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
        req.done_mu.unlock(self.io);

        if (req.error_name) |name| {
            // The failure crosses the thread boundary by NAME — map it back
            // to a typed error so a memory-preflight refusal surfaces as a
            // named 503, not the generic "Model load failed" 500 (#144).
            defer self.allocator.free(name);
            // On success the inference thread took ownership of cpu_state;
            // on failure it didn't, so we still hold it.
            return ModelRegistry.loadErrorFromName(name);
        }

        // Success — inference thread installed cpu_state onto the entry.
        owned_active = false;

        // Already refcounted (`pinned`), so no other load evicted it; only an explicit
        // unload can have claimed it since `markReady`, and that is a 503, not a failed load.
        self.registry.mutex.lockUncancelable(self.io);
        const ready = entry.state == .ready;
        self.registry.mutex.unlock(self.io);
        if (!ready) return error.NotLoaded;
        return entry;
    }

    /// Release a borrowed pointer obtained from `ensureLoaded`. Forwards
    /// to the registry so the refcount decrement + LRU clock bump happen
    /// under registry.mutex.
    pub fn release(self: *Scheduler, lm: *LoadedModel) void {
        self.registry.release(lm);
    }

    /// `release` for a status read — same refcount protocol, no recency
    /// stamp. See `ModelRegistry.releaseStatus`.
    pub fn releaseStatus(self: *Scheduler, lm: *LoadedModel) void {
        self.registry.releaseStatus(lm);
    }

    /// Duped stored failure name for the id `ensureLoaded` just refused with
    /// `error.LoadFailed` — feeds the "Model load failed: <name>" HTTP
    /// message (#144). Caller frees.
    pub fn loadErrorName(self: *Scheduler, alloc: std.mem.Allocator, id_or_empty: []const u8) ?[]u8 {
        return self.registry.loadErrorNameDupe(alloc, id_or_empty);
    }

    /// Free a model's resident GPU state, returning its registry stub to
    /// `.unloaded` so it can reload later. Idempotent: a non-resident model
    /// returns immediately. Marks the entry `.evicting`, drains in-flight
    /// requests (refcount to 0), then hands the mlx free to the inference
    /// thread (stream-bound). Blocks until the free completes. An optional
    /// idle precondition is re-checked under the registry mutex at the
    /// moment we commit (null = unload unconditionally).
    ///
    /// The sweep picks a victim, drops the mutex, then calls in, so a request
    /// can arrive in that gap: refcount and age are re-checked under the mutex
    /// before committing. (`planEvictionsLocked` picks and marks under one
    /// hold instead; the sweep cannot, because the unload itself is slow.)
    pub fn unloadModelIfIdle(self: *Scheduler, id_or_empty: []const u8, idle_window_ms: ?i64) !void {
        const entry = try self.registry.resolveEntry(id_or_empty);
        {
            self.registry.mutex.lockUncancelable(self.io);
            wait: while (true) {
                switch (entry.state) {
                    // Already free (or failed-load stub) → nothing to do.
                    .unloaded, .error_state => {
                        self.registry.mutex.unlock(self.io);
                        return;
                    },
                    // Another caller is mid load/evict — wait it out, then
                    // re-check (it may end up resident or unloaded).
                    .loading, .evicting => {
                        self.registry.state_cond.waitUncancelable(self.io, &self.registry.mutex);
                        continue :wait;
                    },
                    .ready => break :wait,
                }
            }
            if (idle_window_ms) |window_ms| {
                const now_ms = io_util.nowMsMonotonic(self.io);
                if (!model_registry_mod.ModelRegistry.idleEvictable(entry, now_ms, window_ms)) {
                    self.registry.mutex.unlock(self.io);
                    return;
                }
            }
            self.registry.markEvictingLocked(entry);
            self.registry.waitForRefcountZeroLocked(entry);
            self.registry.mutex.unlock(self.io);
        }

        var req = UnloadRequest{ .entry = entry };
        {
            self.queue_mu.lockUncancelable(self.io);
            defer self.queue_mu.unlock(self.io);
            // On shutdown the entry stays `.evicting`; `Scheduler.deinit`
            // unloads every `.ready`/`.evicting` entry, so it's still freed.
            if (self.shutdown.load(.acquire)) return error.Shutdown;
            try self.unload_queue.append(self.allocator, &req);
            self.queue_cond.broadcast(self.io);
        }
        req.done_mu.lockUncancelable(self.io);
        while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
        req.done_mu.unlock(self.io);
    }

    /// The least recently used resident model nobody is holding, other than `keep`; its id
    /// outlives the unload (the entry stays registered as a stub).
    pub fn idleModelToUnload(self: *Scheduler, keep: []const u8) ?[]const u8 {
        self.registry.mutex.lockUncancelable(self.io);
        defer self.registry.mutex.unlock(self.io);
        const entry = self.registry.pickLruEvictable(keep) orelse return null;
        return entry.id;
    }

    /// Synchronously compute embeddings for `req.token_seqs` using the
    /// batched encoder forward pass on the inference thread. Same lifecycle
    /// as a load request: post + block + return results. Caller frees the
    /// returned rows + outer slice (allocated with `req.allocator`).
    pub fn computeEmbeddings(self: *Scheduler, req: *EmbedRequest) ![][]f32 {
        self.queue_mu.lockUncancelable(self.io);
        self.embed_queue.append(self.allocator, req) catch |err| {
            self.queue_mu.unlock(self.io);
            return err;
        };
        self.queue_cond.broadcast(self.io);
        self.queue_mu.unlock(self.io);

        req.done_mu.lockUncancelable(self.io);
        defer req.done_mu.unlock(self.io);
        while (!req.done) {
            req.done_cond.waitUncancelable(self.io, &req.done_mu);
        }
        if (req.error_name) |_| return error.EmbedFailed;
        return req.results orelse error.EmbedFailed;
    }

    /// Does this slot's next decode tick actually run the regular (non-speculative)
    /// path? The slot's `enable_*` flags carry the REQUEST's wish; `specTickMode` is
    /// the authoritative dispatch answer, and a generator can also have turned spec
    /// off at runtime. Reading the armed flags here vetoed batched decode for any
    /// slot whose prompt merely n-gram-scored high enough to ARM PLD, even after
    /// PLD's own yield gate had disabled itself — neither speculation nor batching
    /// (measured 2.4x on concurrent gated-delta decode). Same class as "a guard that shapes
    /// INIT options does not bind DISPATCH".
    fn slotTicksRegular(slot: *const Slot) bool {
        const gen = if (slot.legacy_gen) |*g| g else return !(slot.enable_pld or slot.enable_drafter);
        // A runtime-disabled generator is already ticking regular, so batching it
        // dispatches what it was going to dispatch anyway.
        //
        // The POLICY this encodes, which is deliberate and not free: `nextPld`'s
        // periodic re-enable check (`SPEC_REENABLE_INTERVAL`) lives on the serial
        // path, and the batched tick never calls it. So a slot whose PLD yield
        // gate disabled itself stays disabled for as long as it keeps company —
        // even if its tail later turns into the file/tool echo PLD is best at.
        // That is the right trade at N>1 (the batched kernel reads the weight set
        // once for the whole group, which beats one slot's lookup wins) and it
        // self-corrects: a SOLO slot never reaches the batched path, so the
        // re-enable check resumes the moment concurrency drops back to one.
        // Pinned by `a spec_disabled_runtime slot is batchable, and that is the
        // documented trade` below — flip either half deliberately, not by accident.
        return gen.spec_disabled_runtime or slotSpecMode(slot, gen) == .regular;
    }

    /// Active-tick gate. Decides whether a slot is eligible for the batched
    /// decode kernel. Hybrid recurrent-state / MoE / encoder models can't ride
    /// the batched kernel (it doesn't model their state), so any slot
    /// targeting such a model falls through to the single-slot path. Phase
    /// D: the gate reads off the slot's own model config — multi-model
    /// means the scheduler's startup config is no longer authoritative.
    fn batchable(self: *const Scheduler, slot: *const Slot) bool {
        return self.batchVerdict(slot) == .ok;
    }

    /// Why a slot does or does not ride the batched kernel. `.ok` batches; every
    /// other arm decodes serial this tick and is what the `[batched] serial` line
    /// and `decode_serial_total{reason}` name.
    fn batchVerdict(self: *const Scheduler, slot: *const Slot) BatchVerdict {
        _ = self;
        if (!slotTicksRegular(slot)) return .spec_active;
        // A slot whose module-head release is armed but not landed still holds the head.
        if (slot.sampling.constraint != null) return .grammar;
        if (slot.logprobs_n > 0) return .logprobs;
        if (slot.sampling.penalized()) return .penalty;
        const cfg = slot.model.config orelse return .arch;
        return if (modelBatchable(cfg)) .ok else .arch;
    }
};

/// Slots one batched forward can carry; the tail decodes serial this tick.
pub const MAX_BATCH_GROUP = 32;

pub const BatchVerdict = enum {
    ok,
    spec_active,
    grammar,
    logprobs,
    penalty,
    embedded_engine,
    arch,
    pad_waste,
};

/// Does the loaded model's config batch at all? The arch half of `batchVerdict`,
/// shared with `/props`, `/v1/models` and the serve-mode startup line.
pub fn configBatchesDecode(cfg: *const model_mod.ModelConfig) bool {
    return modelBatchable(cfg);
}

/// One line per slot the first time it decodes serial beside live company;
/// the counter moves every tick so the rate is visible under `--metrics`.
fn noteSerial(sch: *Scheduler, slot: *Slot, why: BatchVerdict) void {
    if (sch.metrics) |m| m.decode_serial_total[@backingInt(why)].inc();
    if (slot.serial_reason_logged) return;
    slot.serial_reason_logged = true;
    log.info("[batched] slot serial: {s} (model={s})\n", .{ @tagName(why), slot.model.id });
}

/// Batched decode pads every slot's KV to the group's LONGEST (`padAndStackBatchedKV`),
/// so the tensor it builds is `N x kv_max`, not `sum(kv_len)`. A group mixing one
/// 100k-token stream with three 1k ones therefore materializes ~100x the bytes the
/// short slots need - on a hd-256 trunk ( `--ctx-size` up to 262144) that is
/// ~410 MB per full-attention layer, ~6.5 GB across the 16 of them, and NOTHING bills
/// it: it is a per-tick transient, invisible to `prefillTransientReserve` and to the
/// load-time gate. An uncatchable Metal OOM is the failure mode.
///
/// So the group is capped by PADDING WASTE, which is the quantity that actually hurts,
/// rather than by a length ratio: keep the largest prefix of the ascending-sorted
/// lengths whose padded tensor stays within `MAX_PAD_WASTE` of its useful bytes. The
/// slots that fall out are the LONGEST ones — the ones dominating kv_max — and they
/// decode serially this tick, so every slot still advances.
///
/// Returns how many of `kv_lens_asc` may batch together (0 or 1 = nobody batches).
/// The lengths are the arch's true attention KV length (`batchKvLenOf`), pre-tick counts,
/// so this is a heuristic bar on the padding the forward will build, not an accounting identity.
/// The padded tensor may be at most this multiple of the bytes the group
/// actually needs. It must stay BELOW 2.0 or a two-slot group can never be
/// vetoed: one 1-token slot beside one 200k slot pads to exactly 2x, which is
/// the worst case for N=2 and the pathological pair this cap exists for.
pub const MAX_PAD_WASTE: f64 = 1.5;

var kv_skew_split_logged: bool = false; // one-shot log guard

pub fn batchedKvKeepCount(kv_lens_asc: []const u32) usize {
    if (kv_lens_asc.len < 2) return 0;
    var k: usize = kv_lens_asc.len;
    while (k >= 2) : (k -= 1) {
        var sum: u64 = 0;
        for (kv_lens_asc[0..k]) |l| sum += l;
        if (sum == 0) return k; // nothing prefilled yet: no padding to waste
        const padded: f64 = @floatFromInt(@as(u64, k) * kv_lens_asc[k - 1]);
        if (padded <= MAX_PAD_WASTE * @as(f64, @floatFromInt(sum))) return k;
    }
    return 0;
}

/// A group whose longest billed slot is at most this long pads a tensor that is small in absolute
/// terms, and splitting it sends a slot through a whole serial forward instead. Under the
/// gather arm every long slot bills at the indexer budget (2052 rows), so a short sub-agent beside
/// long streams reads as a 2x pad and was split off.
pub const PAD_FREE_KV: u32 = 4096;

/// The floor above only engages once some slot in the group holds at least this much context:
/// below it the serial forward it saves is cheap and keeping the group whole does not pay.
pub const PAD_FREE_MIN_CTX: u32 = 131072;

/// Does the small-pad floor keep this group whole? `billed_max` is the longest billed length,
/// `ctx_max` the longest true context.
pub fn padFreeFloorApplies(billed_max: u32, ctx_max: u32) bool {
    return ctx_max >= PAD_FREE_MIN_CTX and billed_max <= PAD_FREE_KV;
}

/// `batchedKvKeepCount` for a group whose forward may attend per slot: past the per-slot
/// floor nothing is padded, so every slot batches. `ctx_max` is the group's longest true context.
pub fn groupKeepCount(kv_lens_asc: []const u32, per_slot_capable: bool, ctx_max: u32) usize {
    if (kv_lens_asc.len >= 2) {
        const billed_max = kv_lens_asc[kv_lens_asc.len - 1];
        if (per_slot_capable and billed_max >= transformer_mod.Transformer.BATCHED_PER_SLOT_ATTN_MIN_KV) return kv_lens_asc.len;
        if (padFreeFloorApplies(billed_max, ctx_max)) return kv_lens_asc.len;
    }
    return batchedKvKeepCount(kv_lens_asc);
}

/// The longest true context (not the billed length) among the group's caches.
fn groupCtxMax(caches: []const *const KVCache) u32 {
    var m: usize = 0;
    for (caches) |c| m = @max(m, c.kvLenForBatching());
    return @intCast(@min(m, std.math.maxInt(u32)));
}

/// The per-slot attention arm serves every batched trunk.
fn groupAttendsPerSlot(slot: *const Slot) bool {
    const cfg = slot.model.config orelse return true;
    return !cfg.longCtxGated();
}

/// The padding waste the whole group would pay; reported by the cap's log.
pub fn batchedPadWaste(kv_lens_asc: []const u32) f64 {
    var sum: u64 = 0;
    for (kv_lens_asc) |l| sum += l;
    if (sum == 0 or kv_lens_asc.len == 0) return 1.0;
    const padded: f64 = @floatFromInt(@as(u64, kv_lens_asc.len) * kv_lens_asc[kv_lens_asc.len - 1]);
    return padded / @as(f64, @floatFromInt(sum));
}

pub fn batchKvLenOfWith(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig, seq_len: c_int) u32 {
    _ = cfg;
    _ = seq_len;
    return @intCast(cache.kvLenForBatching());
}

pub fn fillGroupPadWasteKvLens(
    caches: []const *const KVCache,
    cfg: ?*const model_mod.ModelConfig,
    seq_len: c_int,
    out: []u32,
) void {
    for (caches, 0..) |c, i| out[i] = batchKvLenOfWith(c, cfg, seq_len);
}

/// Pure-config predicate: is this model's architecture compatible with the
/// batched-decode kernel? Used by `Scheduler.batchable` after slot-level
/// flags are checked. MoE / hybrid / encoder have shape mismatches with
/// `forwardBatchedDecode` and fall through to per-slot dispatch.
pub fn modelBatchable(cfg: *const model_mod.ModelConfig) bool {
    if (cfg.isMoe()) return false;
    return true;
}

/// A model whose per-request decode state is MODULE-OWNED (one per model,
/// not per slot) - at most ONE in-flight request may touch it. Such state is
/// rebuilt at cache.step==0 and advanced by every decode tick, so a second
/// interleaved slot deinit+rebuilds the active request's state and both then
/// append tokens to the ONE state.
/// Serial-tick interleave stays safe for per-slot-state archs (module-state trunks)
/// — ask the transformer which archs own their decode state, NEVER key on
/// !modelBatchable, and never name ONE arch here: when a second module-owned
/// arch arrived, a hardcoded check silently never reached the gate for it.
fn modelExclusiveDecode(model: *const model_registry_mod.LoadedModel) bool {
    const t = model.transformer orelse return false;
    return t.ownsModuleDecodeState();
}

/// A pending prefill that does not fit beside live requests waits for them to finish. Alone it
/// proceeds: the connection thread already admitted it, and nobody would ever free memory for it.
pub fn holdsForMemory(fits: bool, live_company: bool) bool {
    return !fits and live_company;
}

/// Does a prefill needing `need` fit `avail` beside siblings' unallocated promises and the
/// sibling margin? Metal OOM is uncatchable, so the transients are billed before chat may take them.
pub fn admissionFits(need: u64, avail: u64, promised: u64, live: usize) bool {
    const margin: u64 = if (live > 0) SIBLING_MARGIN_BYTES else 0;
    return need +| promised +| margin <= avail;
}

fn liveDecodingCount(sch: *Scheduler) usize {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    var n: usize = 0;
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        n += 1;
    }
    return n;
}

/// Cold bill against live memory; the gated arch runs its own warm pass inside `runPrefill`.
fn slotHoldsForMemory(sch: *Scheduler, slot: *Slot) bool {
    const cfg = slot.model.config orelse return false;
    if (cfg.longCtxGated()) return false;
    if (slot.model.transformer == null) return false;
    const numbers_fn = prefill_admission_numbers orelse return false;
    // A prefill hosting a nested admission holds memory but is in no list: it is company too.
    const live = liveDecodingCount(sch) + @intFromBool(sch.boundary_outer != null);
    if (live == 0) sch.promise_ledger = .{};
    const bill = numbers_fn(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, generate_mod.visionPrefillUnchunked(slot.vision_embeddings != null));
    var active_now: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_now);
    const fits = admissionFits(bill[0], bill[1], sch.promise_ledger.outstanding(active_now), live);
    if (!holdsForMemory(fits, live > 0)) {
        if (slot.memory_promised == 0) {
            slot.memory_promised = bill[0];
            sch.promise_ledger.admit(bill[0], active_now);
        }
        return false;
    }
    if (!slot.memory_hold_logged) {
        slot.memory_hold_logged = true;
        log.info("[admission] held: {d} tokens do not fit beside {d} live request(s); waiting for one to finish\n", .{ slot.full_prompt.len, live });
    }
    return true;
}

test "boundaryEligible: a small plain request is admitted only under a share, never nested, never beside an exclusive model" {
    const base_cand: BoundaryCand = .{ .exclusive = false, .has_media = false, .prompt_logprobs = false, .generator_path = true, .module_state = false, .tail = 31 };
    const base_ctx: BoundaryCtx = .{ .share = 0.5, .nested = false, .outer_exclusive = false, .decoding = 2, .max_concurrent = 8, .max_tail = BOUNDARY_MAX_TAIL };
    try testing.expect(boundaryEligible(base_cand, base_ctx));

    var x = base_ctx;
    x.share = 0;
    try testing.expect(!boundaryEligible(base_cand, x));
    x = base_ctx;
    x.nested = true;
    try testing.expect(!boundaryEligible(base_cand, x));
    x = base_ctx;
    x.outer_exclusive = true;
    try testing.expect(!boundaryEligible(base_cand, x));
    x = base_ctx;
    x.decoding = x.max_concurrent;
    try testing.expect(!boundaryEligible(base_cand, x));

    var c = base_cand;
    c.exclusive = true;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.has_media = true;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.prompt_logprobs = true;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.generator_path = false;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.module_state = true;
    try testing.expect(!boundaryEligible(c, base_ctx));

    // The limit is inclusive.
    c = base_cand;
    c.tail = base_ctx.max_tail;
    try testing.expect(boundaryEligible(c, base_ctx));
    c.tail = base_ctx.max_tail + 1;
    try testing.expect(!boundaryEligible(c, base_ctx));
}

test "BoundaryBudget: stops at the count or the time cap, whichever comes first" {
    var b: BoundaryBudget = .{ .max_count = 2, .ns_cap = 1000 };
    try testing.expect(b.mayStart());
    b.charge(10);
    try testing.expect(b.mayStart());
    b.charge(10);
    try testing.expect(!b.mayStart());

    var t: BoundaryBudget = .{ .max_count = 5, .ns_cap = 100 };
    t.charge(100);
    try testing.expect(!t.mayStart());

    var s: BoundaryBudget = .{ .max_count = 5, .ns_cap = std.math.maxInt(u64), .spent_ns = std.math.maxInt(u64) - 1 };
    s.charge(10);
    try testing.expectEqual(@as(u64, std.math.maxInt(u64)), s.spent_ns);
}

test "orderByTail: ascending by uncached tail, stable for equal tails" {
    const tails = [_]usize{ 500, 31, 31, 2000 };
    var order: [4]usize = undefined;
    orderByTail(&tails, &order);
    try testing.expectEqualSlices(usize, &.{ 1, 2, 0, 3 }, &order);

    var one: [1]usize = undefined;
    orderByTail(&[_]usize{7}, &one);
    try testing.expectEqualSlices(usize, &.{0}, &one);
    orderByTail(&[_]usize{}, order[0..0]);
}

test "promiseMaterialized: a promise is dropped once the slot is decoding steadily" {
    try testing.expect(!promiseMaterialized(0));
    try testing.expect(!promiseMaterialized(PROMISE_MATERIALIZED_TOKENS - 1));
    try testing.expect(promiseMaterialized(PROMISE_MATERIALIZED_TOKENS));
}

test "PromiseLedger: a sibling is billed against what earlier admits have not allocated yet" {
    var l: PromiseLedger = .{};
    const gb: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 0), l.outstanding(20 * gb));
    l.admit(3 * gb, 20 * gb);
    l.admit(3 * gb, 20 * gb); // restored prefixes share buffers: active has not moved
    try testing.expectEqual(6 * gb, l.outstanding(20 * gb));
    try testing.expectEqual(2 * gb, l.outstanding(24 * gb)); // 4 GB of the promise materialized
    try testing.expectEqual(@as(u64, 0), l.outstanding(30 * gb));
    l.release(3 * gb);
    l.release(3 * gb);
    l.admit(1 * gb, 30 * gb); // an empty ledger re-bases
    try testing.expectEqual(1 * gb, l.outstanding(30 * gb));
}

test "a prefill that does not fit waits only while another request is live" {
    try testing.expect(holdsForMemory(false, true));
    try testing.expect(!holdsForMemory(false, false)); // alone: never waits, or it waits forever
    try testing.expect(!holdsForMemory(true, true));
}

test "admissionFits: promises add, and a live sibling keeps its margin" {
    const gb: u64 = 1 << 30;
    try testing.expect(admissionFits(10 * gb, 20 * gb, 0, 0));
    try testing.expect(!admissionFits(10 * gb, 20 * gb, 11 * gb, 0)); // promises add
    try testing.expect(admissionFits(17 * gb, 20 * gb, 0, 1));
    try testing.expect(!admissionFits(19 * gb, 20 * gb, 0, 1)); // a sibling keeps its margin
}

/// One pending-drain candidate (or live decoding slot), reduced to what
/// admission needs: an opaque model identity + the exclusive-decode bit.
pub const AdmitCand = struct { model: usize, exclusive: bool };

/// FIFO admission for one drain tick. An EXCLUSIVE candidate admits only if
/// no EXCLUSIVE `active` slot (live decoding) holds its model and no earlier
/// admitted exclusive candidate claimed it this tick
/// (a same-tick sibling is not in `decoding` yet — the claim covers the
/// window). Non-exclusive candidates always admit — no head-of-line
/// blocking behind a held exclusive request. Writes admitted candidate
/// indices to `out` (queue order preserved), returns the count. Held
/// candidates stay where they are and retry next tick; no release
/// bookkeeping exists — the busy signal IS presence in `decoding`.
pub fn admitPendingTick(cands: []const AdmitCand, active: []const AdmitCand, out: []usize) usize {
    var n: usize = 0;
    outer: for (cands, 0..) |c, i| {
        if (n >= out.len) break;
        if (c.exclusive) {
            for (active) |a| if (a.exclusive and a.model == c.model) continue :outer;
            for (out[0..n]) |j| {
                if (cands[j].exclusive and cands[j].model == c.model) continue :outer;
            }
        }
        out[n] = i;
        n += 1;
    }
    return n;
}

const ThreadCtx = struct {
    scheduler: *Scheduler,
    params: LoadParams,
};

/// Signal to the parent waiting in `init()` that the inference thread is done
/// with its load (or has failed). After `started` flips, the parent reads
/// `load_failed` to decide whether to surface a startup error.
fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}
/// Set the load_error_name + flip load_failed. Best-effort dupe; on OOM the
/// parent still sees `load_failed=true` and surfaces "unknown".
fn recordLoadError(sch: *Scheduler, err_name: []const u8) void {
    if (sch.load_error_name) |old| sch.allocator.free(old);
    sch.load_error_name = sch.allocator.dupe(u8, err_name) catch null;
    sch.load_failed.store(true, .release);
}

/// Heap-allocate `T`, run `init_fn`, return owning pointer. On `init_fn`
/// failure, the heap slot is freed before the error propagates so the
/// scheduler never holds a half-initialized struct.
fn boxInit(
    allocator: std.mem.Allocator,
    comptime T: type,
    init_fn: anytype,
    args: anytype,
) !*T {
    const ptr = try allocator.create(T);
    errdefer allocator.destroy(ptr);
    ptr.* = try @call(.auto, init_fn, args);
    return ptr;
}

/// Both load construction sites (here and main.zig's startup load) stamp the
/// per-model settings onto the config the bills and defaults read.
/// The kwargs strings MOVE to the freshly loaded `chat_config` (same allocator).
pub fn applyModelSettings(config: *ModelConfig, chat_config: *ChatConfig, o: *model_settings.Override) void {
    config.ctx_override = o.ctx_size orelse 0;
    config.kv_quant_override = o.kv_quant;
    config.int8_prefill_override = o.int8_prefill;
    config.lm_head_mxfp8_override = o.lm_head_mxfp8;
    config.drafter_override = o.drafter;
    o.drafter = null;
    chat_config.chat_template_kwargs = o.chat_template_kwargs;
    o.chat_template_kwargs = null;
}

/// Plan 05 Phase D: pre-loaded CPU state bundle. Built by the conn thread
/// (CPU only — file I/O + parse, no mlx) ahead of posting a LoadRequest.
/// Ownership transfers to the entry on successful load; on failure the
/// conn thread frees via `freeCpuState`.
const CpuState = struct {
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,
};

/// Phase D: parse config.json, tokenizer, and chat config from `model_dir`
/// into heap pointers ready to hand off to `LoadRequest`. Mirrors the
/// pre-load that main.zig does for the startup model in serve mode.
///
/// Errdefer pattern: for each `try ... else error`, the `deinit` errdefer
/// is registered AFTER the successful init so a downstream failure doesn't
/// call deinit on uninitialized memory. `ModelConfig.deinit` frees the config's
/// one owned field; `allocator.destroy` alone would leak it.
fn preloadCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !CpuState {
    return preloadTextCpuState(allocator, io, model_dir);
}

fn preloadTextCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !CpuState {
    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = try model_mod.parseConfig(io, allocator, model_dir);

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    tok.* = try tokenizer_mod.loadTokenizer(io, allocator, model_dir);
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = try chat_mod.loadChatConfig(io, allocator, model_dir);
    errdefer cc.deinit();

    config.applyTokenizer(tok, cc.eos_token);

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// Frees the three CPU-state pointers (on success they transfer to the entry,
/// so this function is skipped).
fn freeCpuState(allocator: std.mem.Allocator, s: *CpuState) void {
    s.config.deinit(allocator);
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// A load whose own bill exceeds the memory cap is refused whatever else is resident.
fn loadExceedsMemCap(estimated: u64, max_resident_mem: u64) bool {
    return max_resident_mem != 0 and estimated > max_resident_mem;
}

const TEXT_GATE_MARGIN_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;

/// The post-load residency bill the eviction gate reserves, in bytes.
///
/// A text bill is weights only, so it takes 10% headroom for KV / vision /
/// drafter overhead, up to `TEXT_GATE_MARGIN_CAP_BYTES`: that overhead does not
/// grow with the weights.
pub fn gateEstimateBytes(bytes_on_disk: ?u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    const base: u64 = if (bytes_on_disk) |b|
        b
    else
        @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
    return base + @min(base / 10, TEXT_GATE_MARGIN_CAP_BYTES);
}

/// Bytes MLX holds right now across every resident model (an allocator counter, safe off the inference thread).
fn mlxActiveBytes() u64 {
    var n: usize = 0;
    _ = mlx.mlx_get_active_memory(&n);
    return n;
}

/// Sum of `*.safetensors` bytes in `model_dir` — the MLX weight footprint used
/// by the load pre-flight. Returns 0 if the dir can't be read (treated as
/// "unknown" by the caller, which then skips the check). Symlinked weights
/// count (statFile follows links) — an HF hub-cache snapshot is ALL symlinks.
fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    // A pack's index names the shards the loader reads; a stray shard beside
    // them (issue #274) is dead weight and must not be billed.
    var referenced: ?std.StringHashMapUnmanaged(void) = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);
    var it = dir.iterate();
    var total: u64 = 0;
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        total += @intCast(st.size);
    }
    if (referenced) |r| {
        var kit = r.keyIterator();
        while (kit.next()) |name| {
            if (std.mem.indexOfScalar(u8, name.*, '/') == null or std.mem.indexOf(u8, name.*, "..") != null) continue;
            const st = dir.statFile(io, name.*, .{}) catch continue;
            if (st.kind == .file) total += @intCast(st.size);
        }
    }
    return total;
}

/// A ModelOpt pack also bills the banks' decoded scales, which are smaller on disk.
fn residentModelDiskBytes(io: std.Io, model_dir: []const u8, config: *const model_mod.ModelConfig) !u64 {
    const shards = try shardResidentBytes(io, model_dir, config);
    const whole = if (config.modelopt) shards +| modelopt_pack.bankExpansionBytes(model_dir) else shards;
    return whole;
}

fn shardResidentBytes(io: std.Io, model_dir: []const u8, config: *const model_mod.ModelConfig) !u64 {
    _ = config;
    return modelDiskBytes(io, model_dir);
}

test "residentModelDiskBytes: a ModelOpt pack is billed its decoded expert banks" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try modelopt_pack.writeMiniPack(a, io, tmp.dir, .{ .layers = 2, .experts = 3, .cols = 64, .scale_suffix = "weight_scale.nvfp4_csf_fixed", .scale_len = 10 });
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr))), tmp.sub_path });
    defer a.free(path);
    const on_disk = modelDiskBytes(io, path);
    var config: model_mod.ModelConfig = .{ .model_type = "gemma4" };
    try std.testing.expectEqual(on_disk, try residentModelDiskBytes(io, path, &config));
    config.modelopt = true;
    try std.testing.expectEqual(on_disk + 18 * 18, try residentModelDiskBytes(io, path, &config));
}

fn residentGateBytes(validated: ?u64, discovered: ?u64) ?u64 {
    if (validated) |bytes| if (bytes > 0) return bytes;
    return discovered orelse 0;
}

test "the eviction gate keeps discovery bytes when the validated sum is unavailable" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(@as(?u64, 70 * GB), residentGateBytes(70 * GB, 100 * GB));
    try std.testing.expectEqual(@as(?u64, 100 * GB), residentGateBytes(0, 100 * GB));
    try std.testing.expectEqual(@as(?u64, 0), residentGateBytes(0, null));
}

test "a text pack's gate margin stops growing with its weights" {
    const GiB: u64 = 1024 * 1024 * 1024;
    // A 203.83 GiB text pack must fit a 222.72 GiB cap with nothing else loaded.
    const ds: u64 = 218_858_232_429;
    try std.testing.expect(gateEstimateBytes(ds, 43, 4096) < 222 * GiB);
    // Up to 100 GiB the margin is still 10%.
    try std.testing.expectEqual(@as(u64, 66 * GiB), gateEstimateBytes(60 * GiB, 48, 2560));
}

test "a load over the memory cap on its own is named as such, not as a busy registry" {
    const GiB: u64 = 1024 * 1024 * 1024;
    try std.testing.expect(loadExceedsMemCap(225 * GiB, 222 * GiB));
    try std.testing.expect(!loadExceedsMemCap(200 * GiB, 222 * GiB));
    try std.testing.expect(!loadExceedsMemCap(225 * GiB, 0));
}

fn committedTextBytes(model_bytes: u64, config: *const model_mod.ModelConfig) u64 {
    if (model_bytes > 0) return model_bytes;
    return @as(u64, config.num_hidden_layers) * @as(u64, config.hidden_size) * 4 * 4;
}

test "modelDiskBytes follows HF-cache symlinks (a snapshot dir measured ZERO)" {
    // A model served straight out of the HuggingFace hub cache is a snapshot
    // dir of SYMLINKS into ../../blobs. Skipping .sym_link entries measured a
    // 121 GB checkpoint at 0 bytes, and memInsufficientForLoad treats 0 as
    // "unknown" → the preflight waved the load through into 34 GB of swap.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "blobs");
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/abc123", .data = "0123456789abcdef" });
    try tmp.dir.createDirPath(io, "snapshots/rev");
    try tmp.dir.symLink(io, "../../blobs/abc123", "snapshots/rev/model.safetensors", .{});
    // A dangling link (blob pruned) is skipped, never an error…
    try tmp.dir.symLink(io, "../../blobs/gone", "snapshots/rev/model-00002.safetensors", .{});
    // …and a symlink to a DIRECTORY must not be summed (statFile follows it).
    try tmp.dir.symLink(io, "../../blobs", "snapshots/rev/dir.safetensors", .{});

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const snap = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/snapshots/rev", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(snap);

    try std.testing.expectEqual(@as(u64, 16), modelDiskBytes(io, snap));
}

test "modelDiskBytes bills only the shards the index names (issue #274)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00001-of-00002.safetensors", .data = "0123456789" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00002-of-00002.safetensors", .data = "01234" });
    // Dead weight: present on disk, referenced by nothing.
    try tmp.dir.writeFile(io, .{ .sub_path = "m/stray.safetensors", .data = "0123456789abcdef0123456789abcdef" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model.safetensors.index.json", .data =
        \\{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}
    });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/m", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(dir);
    try std.testing.expectEqual(@as(u64, 15), modelDiskBytes(io, dir));
}

/// Pure: would loading `weights_bytes` of model with `avail_bytes` free RAM risk
/// a Metal OOM? Requires the weights plus ~1/12 (≈8%) + 0.25 GB headroom for the
/// warmup KV cache + compute buffers. Deliberately lean: `avail_bytes` (active +
/// wired + compressed subtracted) under-counts what macOS reclaims from file
/// cache the moment MLX allocates, so a fat headroom wrongly refuses loads that
/// fit. The guard's real job is the gross case (restart a 42 GB model into 44 GB
/// free → hard process-killing OOM), which this still catches. Returns false
/// (allow the load) when either figure is 0 — a failed memory query must never
/// block a load.
/// Set by `--skip-mem-preflight` (main.zig) to bypass the model-load memory
/// pre-flight below. A module global, not a `LoadParams` field, so it applies
/// uniformly to startup loads AND later hot-loads — matching the env var
/// (`MLX_SERVE_SKIP_MEM_PREFLIGHT`) it replaced.
pub var skip_mem_preflight: bool = false;

/// Which drafter directory a COLD load should use.
///
/// `--no-drafter` is a policy — it silences every model, including one whose
/// own dir ships a sidecar `dflash.resolveInDirDrafter` would otherwise find.
/// `--drafter <path>`, by contrast, names a sidecar for the checkpoint it was
/// passed beside: handing it to whatever model is swapped in next would load a
/// mismatched assistant, so it applies only when the entry being loaded IS the
/// launch model (which happens on a reload after eviction). Every other model
/// is served by the in-dir probe.
pub fn coldLoadDrafterDir(
    no_drafter: bool,
    primary_model_dir: []const u8,
    drafter_dir: []const u8,
    entry_path: []const u8,
) []const u8 {
    if (no_drafter) return "";
    if (drafter_dir.len == 0) return "";
    if (!std.mem.eql(u8, primary_model_dir, entry_path)) return "";
    return drafter_dir;
}

/// The drafter a load uses: "" = none, null = the in-dir probe decides.
/// `--no-drafter`/`--drafter` win over the per-model `drafter` setting.
pub fn drafterFor(no_drafter: bool, explicit: []const u8, setting: ?[]const u8) ?[]const u8 {
    if (no_drafter) return "";
    if (explicit.len > 0) return explicit;
    const s = setting orelse return null;
    if (std.mem.eql(u8, s, "off")) return "";
    if (std.mem.eql(u8, s, "auto")) return null;
    return s;
}

test "coldLoadDrafterDir: --no-drafter wins, an explicit --drafter belongs to its OWN model" {
    // `--drafter <path>` names a sidecar for the checkpoint it was passed
    // with; handing it to whatever model gets swapped in next would load a
    // mismatched assistant. Other models are served by the in-dir probe.
    // Reloading the launch model AFTER an eviction must still get it back.
    try testing.expectEqualStrings("/d", coldLoadDrafterDir(false, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "/d", "/other"));
    // --no-drafter is a policy, not a path: it silences every model, including
    // one whose own dir ships a sidecar the in-dir probe would find.
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/other"));
    // No --drafter at launch: nothing to carry, the in-dir probe decides.
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "", "/m"));
}

test "drafterFor: launch flags win, then the per-model setting, then the in-dir probe" {
    try testing.expectEqualStrings("", drafterFor(true, "/d", "/s").?);
    try testing.expectEqualStrings("/d", drafterFor(false, "/d", "off").?);
    try testing.expectEqualStrings("", drafterFor(false, "", "off").?);
    try testing.expectEqualStrings("/s", drafterFor(false, "", "/s").?);
    try testing.expectEqual(@as(?[]const u8, null), drafterFor(false, "", "auto"));
    try testing.expectEqual(@as(?[]const u8, null), drafterFor(false, "", null));
}

test "the cold-load LoadRequest re-applies EVERY retained launch setting" {
    // Separate rounds of this bug shipped: prefix-cache, then
    // the drafter group - each time a launch flag reached
    // only `--model` and the cold path (hot switch, first
    // request naming an unloaded model) quietly used a struct default. The
    // scan is the class guard: a field retained on the Scheduler for this
    // purpose that no cold-load assignment mentions is the next round.
    // Needles are ++-split so this test's own source can't satisfy the scan.
    const src = @embedFile("scheduler.zig");
    inline for (.{
        "kv_quant_config",           "prefix_cache_capacity",     "prefix_cache_mem_bytes",
        "prefix_cache_disk_bytes",
        "no_drafter",                "draft_block_size",          "draft_block_size_explicit",
        "prefix_cache_mem_resolver",
    }) |field| {
        const needle = "." ++ field ++ " = self" ++ "." ++ field ++ ",";
        try testing.expect(std.mem.indexOf(u8, src, needle) != null);
    }
    // The drafter path is the one retained setting that must NOT be copied
    // straight across — it goes through the ownership rule above.
    const via_rule = "coldLoadDrafterDir(" ++ "self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path)";
    try testing.expect(std.mem.indexOf(u8, src, via_rule) != null);
    // The stale TODO that stood in for the wiring must be gone.
    const old = "Phase E will wire the load-model API" ++ " to set this.";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
}

test "every HotPrefixCache.initWithMem load site reads the CLAMPED budget, never the raw launch flag" {
    // 2026-08-30 Metal OOM: a 40 GB `--prefix-cache-mem` beside a ~70 GB pack
    // was never validated against what the weights left under the GPU ceiling;
    // the cache filled to its cap and a 143k prefill died uncatchably. The
    // budget must pass through the resolver (server.prefixCacheMemForLoad)
    // before it reaches initWithMem — and a SECOND construction site must not
    // be able to skip the clamp (the cold-load launch flags class).
    const src = @embedFile("scheduler.zig");
    const needle = "HotPrefixCache.initWith" ++ "Mem(";
    var found: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, src, pos, needle)) |at| : (pos = at + needle.len) {
        found += 1;
        const window = src[at..@min(at + 240, src.len)];
        try testing.expect(std.mem.indexOf(u8, window, "clamped_prefix_mem") != null);
        try testing.expect(std.mem.indexOf(u8, window, "params.prefix_cache_mem_bytes") == null);
    }
    try testing.expect(found >= 1);
}

/// Which "available memory" figure the preflight should trust. On iOS the
/// host-wide number is meaningless — the OS keeps RAM full (file cache,
/// jetsam-evictable background apps) and will evict on our behalf, so the
/// per-PROCESS jetsam headroom (`os_proc_available_memory`, nonzero only on
/// iOS) is the figure that decides whether the load survives. Live bug: an
/// 8 GB iPhone reported ~4 GB host-free and the preflight refused a 3.6 GB
/// model that fit comfortably inside the ~6.4 GB process limit.
/// `gpu_limit` = Metal's working-set limit (0 = unknown): a lowered `iogpu.wired_limit_mb` makes
/// it bind below free RAM, and weights past it OOM in warmup instead of refusing by name.
/// Memory a load of `need` bytes can use.
fn availForLoad(io: std.Io, need: u64) u64 {
    const avail = loadAvail(io);
    if (!waitForUnload(need, avail, model_registry_mod.last_unload, model_registry_mod.monotonicMs())) return avail;
    return settledAvail(need, avail, io, loadAvail, pauseLoadAvail);
}

const UNLOAD_SETTLE_MS = model_registry_mod.UNLOAD_SETTLE_MS;

/// Whether a load refused at `avail` waits for memory still coming back: only right after an
/// unload whose bytes could close the gap. Any other refusal is instant, since the wait stalls every stream.
fn waitForUnload(need: u64, avail: u64, mark: model_registry_mod.UnloadMark, now_ms: i64) bool {
    if (mark.bytes == 0 or now_ms - mark.at_ms > UNLOAD_SETTLE_MS) return false;
    return !memInsufficientForLoad(need, avail +| mark.bytes);
}

fn loadAvail(_: std.Io) u64 {
    return effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), loadGpuLimit());
}

/// RAM a load can still take beside everything resident: what the OS grants less the reserve it
/// keeps (iOS's per-process headroom already is what the process may take). Null when unreadable.
fn loadRamRoom() ?u64 {
    const proc = status.getProcAvailableMemBytes();
    if (proc > 0) return proc;
    const host = status.getAvailableMemBytes();
    if (host == 0) return null;
    return host -| status.osReserveBytes(status.getTotalMemBytes());
}

fn pauseLoadAvail(io: std.Io) void {
    std.Io.sleep(io, .fromMilliseconds(100), .real) catch {};
}

/// Memory an unload just freed reaches the OS over a second or two (150 GB of wired weights on an
/// M5 Ultra), or all at once ~700 ms late (16 GB M4): a reading too low for the load is re-read
/// every pause, for at most 5 s; two seconds without growth leave the refusal standing.
fn settledAvail(need: u64, first: u64, ctx: anytype, comptime measure: fn (@TypeOf(ctx)) u64, comptime pause: fn (@TypeOf(ctx)) void) u64 {
    const step: u64 = 64 << 20; // below this the reading is noise, not memory coming back
    var avail = first;
    var flat: u32 = 0;
    var reads: u32 = 0;
    while (memInsufficientForLoad(need, avail) and flat < 20 and reads < 50) : (reads += 1) {
        pause(ctx);
        const next = measure(ctx);
        if (next > avail +| step) {
            avail = next;
            flat = 0;
        } else flat += 1;
    }
    return avail;
}

/// The GPU ceiling on loaded weights (Metal's working-set limit).
fn loadGpuLimit() u64 {
    return mlx.maxRecommendedWorkingSet();
}

fn effectiveAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64) u64 {
    const avail = if (proc_avail > 0) proc_avail else host_avail;
    return if (gpu_limit > 0) @min(avail, gpu_limit) else avail;
}

test "settledAvail: a refused reading waits while freed memory still reaches the OS" {
    const GB: u64 = 1024 * 1024 * 1024;
    const Seq = struct {
        vals: []const u64,
        i: usize = 0,
        fn measure(self: *@This()) u64 {
            const v = self.vals[@min(self.i, self.vals.len - 1)];
            self.i += 1;
            return v;
        }
        fn pause(_: *@This()) void {}
    };
    // An unload's 150 GB unwires over a few readings: the load fits once it lands.
    var climbing = Seq{ .vals = &.{ 80 * GB, 140 * GB, 200 * GB, 230 * GB } };
    try std.testing.expectEqual(200 * GB, settledAvail(180 * GB, 66 * GB, &climbing, Seq.measure, Seq.pause));
    // A 16 GB M4 unloading 7.3 GB: flat for ~700 ms, then the whole release lands at once.
    var stepped = Seq{ .vals = &.{ 5 * GB, 5 * GB, 5 * GB, 5 * GB, 5 * GB, 5 * GB, 5 * GB, 11 * GB } };
    try std.testing.expectEqual(11 * GB, settledAvail(6 * GB, 5 * GB, &stepped, Seq.measure, Seq.pause));
    // Nothing being freed: the refusal stands after two seconds of flat readings, not five.
    var flat = Seq{ .vals = &.{66 * GB} };
    try std.testing.expectEqual(66 * GB, settledAvail(180 * GB, 66 * GB, &flat, Seq.measure, Seq.pause));
    try std.testing.expectEqual(@as(usize, 20), flat.i);
    // A load that already fits never waits.
    var fits = Seq{ .vals = &.{0} };
    try std.testing.expectEqual(200 * GB, settledAvail(10 * GB, 200 * GB, &fits, Seq.measure, Seq.pause));
    try std.testing.expectEqual(@as(usize, 0), fits.i);
}

test "waitForUnload: only a load an unload just freed room for waits" {
    const GB: u64 = 1024 * 1024 * 1024;
    const now: i64 = 100_000;
    const freed: model_registry_mod.UnloadMark = .{ .at_ms = now - 500, .bytes = 150 * GB };
    try std.testing.expect(waitForUnload(180 * GB, 66 * GB, freed, now));
    try std.testing.expect(!waitForUnload(180 * GB, 66 * GB, .{}, now)); // no unload yet
    try std.testing.expect(!waitForUnload(180 * GB, 66 * GB, .{ .at_ms = now - 60_000, .bytes = 150 * GB }, now)); // long settled
    try std.testing.expect(!waitForUnload(500 * GB, 66 * GB, freed, now)); // even the freed bytes cannot fit it
    var taken = freed;
    taken.consume(150 * GB); // a load committed into the freed room
    try std.testing.expect(!waitForUnload(180 * GB, 66 * GB, taken, now));
}

test "effectiveAvailableBytes is capped by the GPU working-set limit" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(36 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB));
    try std.testing.expect(memInsufficientForLoad(70 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB)));
}

test "effectiveAvailableBytes prefers the per-process jetsam headroom when present" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(6 * GB, effectiveAvailableBytes(4 * GB, 6 * GB, 0)); // iOS: proc wins
    try std.testing.expectEqual(4 * GB, effectiveAvailableBytes(4 * GB, 0, 0)); // macOS: proc query = 0 → host
    try std.testing.expectEqual(@as(u64, 0), effectiveAvailableBytes(0, 0, 0)); // both unknown → 0 (never blocks)
}

fn memInsufficientForLoad(weights_bytes: u64, avail_bytes: u64) bool {
    if (weights_bytes == 0 or avail_bytes == 0) return false;
    // Headroom over the weights for warmup buffers + a baseline KV cache (see
    // `loadRequirementBytes`). `avail_bytes` (status.getAvailableMemBytes) is
    // what the OS grants a new wiring, never total − used. --skip-mem-preflight bypasses.
    return avail_bytes < loadRequirementBytes(weights_bytes);
}

/// Resident bytes a sidecar takes once loaded: `bits` per weight plus a bf16
/// (scale, bias) pair per group of 64 for a dense DFlash assistant quantized
/// at load; its file size otherwise (`bits == 0`).
pub fn drafterResidentBytes(disk_bytes: u64, bits: u32) u64 {
    if (bits == 0) return disk_bytes;
    return disk_bytes * (2 * bits + 1) / 32;
}

const PreflightVerdict = enum { fits, drop_drafter, refuse };

/// The load preflight over the model AND its sidecar: a sidecar the checkpoint
/// brought along (`drafter_optional`) is dropped before the model is refused.
fn preflightVerdict(weights_bytes: u64, drafter_bytes: u64, avail_bytes: u64, drafter_optional: bool) PreflightVerdict {
    if (!memInsufficientForLoad(weights_bytes + drafter_bytes, avail_bytes)) return .fits;
    if (drafter_optional and drafter_bytes > 0 and !memInsufficientForLoad(weights_bytes, avail_bytes)) return .drop_drafter;
    return .refuse;
}

/// Total free memory a load demands: the model's own peak plus the headroom the
/// guard wants for warmup buffers and a baseline KV cache.
///
/// It exists so a REFUSAL can quote the number it actually compared. The media
/// preflight used to say "generation peaks at ~4.3 GB but only 5.4 GB is free"
/// — two figures that, read together, say the load should have worked. It was
/// refused for the headroom, which the sentence never mentioned, so the user
/// went hunting for a problem that wasn't there (live 2026-08-08). Same class as
/// the context-overflow 400: a rejection has to state the bar it set.
pub fn loadRequirementBytes(weights_bytes: u64) u64 {
    const HEADROOM_CAP: u64 = 6 * 1024 * 1024 * 1024;
    const headroom: u64 = @min(weights_bytes / 8, HEADROOM_CAP) + 1024 * 1024 * 1024;
    return weights_bytes + headroom;
}

test "a refusal quotes the number it actually compared" {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    // Live report 2026-08-08: FLUX.2-klein 4B refused with "generation peaks at
    // ~4.3 GB but only 5.4 GB is free" — two numbers that say the load should
    // have worked. The guard was right (it also wants ~1.5 GB of headroom for
    // warmup buffers) but the message quoted the PEAK, so the user went looking
    // for a problem that wasn't there and then tried the same prompt in the
    // other pane. What a refusal must state is the TOTAL it demanded.
    const peak: u64 = 4300 * MB;
    const avail: u64 = 5400 * MB;
    try std.testing.expect(memInsufficientForLoad(peak, avail));
    try std.testing.expect(loadRequirementBytes(peak) > avail);

    // The requirement IS the comparison — not a second formula that can drift
    // from it. At exactly the requirement a load is allowed; a byte under is not.
    try std.testing.expect(!memInsufficientForLoad(peak, loadRequirementBytes(peak)));
    try std.testing.expect(memInsufficientForLoad(peak, loadRequirementBytes(peak) - 1));
    try std.testing.expect(!memInsufficientForLoad(42 * GB, loadRequirementBytes(42 * GB)));
}

test "the preflight refusal quotes the number it compared, not the weights" {
    // The refusal once printed "weights ~8.4 GB but only 10.1 GB free", which reads as
    // "this should have worked" — the load was refused for the 2.05 GB of
    // headroom the sentence never named (live 2026-08-17, a gemma-4-12B QAT pack
    // on a 16 GB M4, where the bar is 10.45 GB). `insufficient_free_memory_message`
    // sends the client to this very line for the figures, so a line that omits
    // the bar makes the client message a dead end too. Needles are ++-split so
    // this test's own source cannot satisfy the scan.
    const src = @embedFile("scheduler.zig");
    // The refusal formats the REQUIREMENT as its first figure, from the one
    // helper the comparison itself uses — never a second formula that can drift.
    const text_arg = "loadRequirement" ++ "Bytes(weights_bytes))) / gb";
    try testing.expect(std.mem.indexOf(u8, src, text_arg) != null);
    // The weights-first shape that could not state its own bar must be GONE.
    const old = "Insufficient memory to load model: weights ~" ++ "{d:.1} GB but only";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
}

test "the eviction gate bills a text entry its weights plus the capped margin" {
    // #126. The dir's safetensors sum is the bill, with 10% headroom.
    const disk_sum: u64 = 40_316_668_515;
    const cap: u64 = 30 * 1024 * 1024 * 1024;
    try testing.expect(gateEstimateBytes(disk_sum, 0, 0) > cap);
    try testing.expectEqual(@as(u64, 1100), gateEstimateBytes(1000, 0, 0));
    try testing.expectEqual(@as(u64, 0), gateEstimateBytes(0, 0, 0));
    // No bytes_on_disk → the layers × hidden × 16 fallback, unchanged.
    const fallback: u64 = 32 * 4096 * 16;
    try testing.expectEqual(fallback + fallback / 10, gateEstimateBytes(null, 32, 4096));
}

test "a sidecar bills its loaded width, and only an in-dir one is dropped to fit" {
    const GB: u64 = 1024 * 1024 * 1024;
    // A dense bf16 assistant quantized at load: bits per weight plus a bf16
    // (scale, bias) pair per group of 64; a packed or dense-loaded one its own bytes.
    try std.testing.expectEqual(@as(u64, 32 * 9), drafterResidentBytes(32 * 32, 4));
    try std.testing.expectEqual(@as(u64, 32 * 17), drafterResidentBytes(32 * 32, 8));
    try std.testing.expectEqual(@as(u64, 1000), drafterResidentBytes(1000, 0));
    // 20 GB of weights want ~23.5 GB; a 3 GB drafter pushes that past 26 GB.
    try std.testing.expectEqual(PreflightVerdict.fits, preflightVerdict(20 * GB, 3 * GB, 30 * GB, true));
    try std.testing.expectEqual(PreflightVerdict.drop_drafter, preflightVerdict(20 * GB, 3 * GB, 25 * GB, true));
    try std.testing.expectEqual(PreflightVerdict.refuse, preflightVerdict(20 * GB, 3 * GB, 25 * GB, false));
    try std.testing.expectEqual(PreflightVerdict.refuse, preflightVerdict(20 * GB, 3 * GB, 20 * GB, true));
    try std.testing.expectEqual(PreflightVerdict.fits, preflightVerdict(20 * GB, 0, 25 * GB, true));
}

test "memInsufficientForLoad: headroom + unknown-query guards" {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    // A 6.9 GB 4-bit model with ~10 GB available (free + speculative + the file
    // cache the OS will grant) — what a 16 GB Mac reports pre-load. Needs ~8.8 GB
    // (weights + weights/8 + 1 GB for warmup + baseline KV) → loads.
    try std.testing.expect(!memInsufficientForLoad(6900 * MB, 10 * GB));
    // Restart-into-pressure: 42 GB weights, only 44 GB free → needs ~46, refuse.
    try std.testing.expect(memInsufficientForLoad(42 * GB, 44 * GB));
    // #45 end-to-end: a prior 7 GB model resident in the anon set leaves 3 GB of
    // reclaimable classes — the guard must refuse the second 7 GB load, or warmup
    // dies in an uncatchable Metal OOM instead of this 400.
    try std.testing.expect(memInsufficientForLoad(7 * GB, 3 * GB));
    // Plenty of headroom → allow.
    try std.testing.expect(!memInsufficientForLoad(42 * GB, 86 * GB));
    // Exactly weights, no headroom → refuse.
    try std.testing.expect(memInsufficientForLoad(42 * GB, 42 * GB));
    // Unknown figures (query failed / size unknown) → never block.
    try std.testing.expect(!memInsufficientForLoad(0, 44 * GB));
    try std.testing.expect(!memInsufficientForLoad(42 * GB, 0));

    // A PROPORTIONAL margin becomes impossible at the top of the range. A
    // 109.7 GB checkpoint on a 128 GB Mac, which reports
    // ~118 GB available with nothing else loaded — but weights/8 demanded 14.7
    // GB of headroom, i.e. 124.4 GB, which that machine cannot have. The guard
    // refused to load the flagship checkpoint on exactly the hardware its model
    // card names, while `--skip-mem-preflight` booted it repeatedly and served
    // 6.7K-token prefills with ~8.6 GB to spare. Headroom covers warmup
    // buffers + a baseline KV cache, and neither scales with a MoE's total
    // weights (13B active here) — so the proportional term is CAPPED.
    try std.testing.expect(!memInsufficientForLoad(109_730 * MB, 118_330 * MB));
    // Still refuses when the box genuinely cannot fit it.
    try std.testing.expect(memInsufficientForLoad(109_730 * MB, 112 * GB));
}

/// Phase A1 → Plan 05: do the full model load on the inference thread.
/// mlx ops here bind to this thread's GPU stream from t0; subsequent
/// forwards stay on the same thread.
///
/// `params` is duck-typed (`anytype`): both `LoadParams` (startup) and
/// `*LoadRequest` (on-demand) supply the same field set — `entry`,
/// `config`/`tok`/`chat_config` (heap pointers), `model_dir`,
/// `drafter_dir`, `warmup_eager`, `draft_block_size`,
/// `draft_block_size_explicit`, `kv_quant_config`, `prefix_cache_capacity`,
/// `prefix_cache_mem_bytes`. The function reads them by name.
///
/// On any error the partial state has already been freed via errdefer; the
/// caller decides how to surface (startup → recordLoadError + signal
/// started; on-demand → req.error_name + done broadcast).
fn doLoadOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    params.entry.load_progress.start(io_util.nowMsMonotonic(sch.io));
    // Optional load steps (kernel warm-ups, mlx_compile, the Latin-head check, the decode micro-bench, the warm-up)
    // fall back when MLX fails and carry on, but MLX's error stays latched; left there, the first request's prefill
    // checkError consumed it and failed with MlxFailure (macOS 14, 2026-11-12). A load that succeeds drops any latch
    // it raised just before the model is marked ready; one pending before the load belongs to someone else and stays.
    const load_had_error = mlx.errorPending();
    const model_bytes = params.resident_model_bytes orelse blk: {
        const scanned = try residentModelDiskBytes(sch.io, params.model_dir, params.config);
        break :blk residentGateBytes(scanned, params.entry.bytes_on_disk).?;
    };

    // GPU-memory pre-flight (MLX path). A Metal OOM during weight load / warmup
    // is thrown by MLX as a C++ exception that can't be caught across the C ABI,
    // so it terminates the whole process. Refuse the load up front instead, with
    // an actionable error, when free RAM clearly can't hold the weights + warmup
    // headroom — catches the common "restarted before the prior server released
    // its memory" case. Bypass with --skip-mem-preflight.
    // The sidecar is resolved here so the preflight bills it with the weights:
    // launch flags, then the per-model setting; otherwise the checkpoint's own
    // `drafter/` subdir (dflash.resolveInDirDrafter). That is what makes the
    // drafter a LOAD-time dependency rather than a launch flag: a hot model
    // switch brings its own, and no pairing table has to decide which sidecar
    // goes with which checkpoint.
    const chosen_drafter = drafterFor(params.no_drafter, params.drafter_dir, params.config.drafter_override);
    var in_dir_drafter: ?[]u8 = if (chosen_drafter == null)
        dflash_mod.resolveInDirDrafter(sch.io, sch.allocator, params.model_dir)
    else
        null;
    defer if (in_dir_drafter) |p| sch.allocator.free(p);
    makeRoomInRam(sch, params.entry.id, model_bytes);
    if (!skip_mem_preflight) {
        const gb = 1024.0 * 1024.0 * 1024.0;
        const sidecar: []const u8 = chosen_drafter orelse in_dir_drafter orelse "";
        const drafter_bytes: u64 = if (sidecar.len == 0) 0 else drafterResidentBytes(modelDiskBytes(sch.io, sidecar), dflash_mod.sidecarQuantBits(sch.io, sch.allocator, sidecar));
        const weights_bytes = model_bytes + drafter_bytes;
        const avail_bytes = availForLoad(sch.io, weights_bytes);
        log.info("[preflight] weights ~{d:.2} GB, available {d:.2} GB\n", .{
            @as(f64, @floatFromInt(model_bytes)) / gb,
            @as(f64, @floatFromInt(avail_bytes)) / gb,
        });
        if (drafter_bytes > 0) log.info("[preflight] drafter ~{d:.2} GB at {s}\n", .{ @as(f64, @floatFromInt(drafter_bytes)) / gb, sidecar });
        switch (preflightVerdict(model_bytes, drafter_bytes, avail_bytes, in_dir_drafter != null)) {
            .fits => {},
            .drop_drafter => {
                log.warn("[dflash] sidecar at {s} skipped: model + drafter need ~{d:.1} GB free, {d:.1} GB available; the model loads without it\n", .{
                    sidecar,
                    @as(f64, @floatFromInt(loadRequirementBytes(weights_bytes))) / gb,
                    @as(f64, @floatFromInt(avail_bytes)) / gb,
                });
                sch.allocator.free(in_dir_drafter.?);
                in_dir_drafter = null;
            },
            .refuse => {
                log.err("Insufficient memory to load model: needs ~{d:.1} GB free ({d:.1} GB of weights{s} plus headroom for warmup buffers and a baseline KV cache) but only {d:.1} GB is available. Close other models/apps (or wait for a prior mlx-serve to fully exit) and retry; pass --skip-mem-preflight to override.\n", .{
                    @as(f64, @floatFromInt(loadRequirementBytes(weights_bytes))) / gb,
                    @as(f64, @floatFromInt(weights_bytes)) / gb,
                    if (drafter_bytes > 0) " and drafter" else "",
                    @as(f64, @floatFromInt(avail_bytes)) / gb,
                });
                return error.InsufficientMemory;
            },
        }
    }

    {
        const sidecar = chosen_drafter orelse in_dir_drafter orelse "";
        const total = model_bytes + (if (sidecar.len > 0) modelDiskBytes(sch.io, sidecar) else 0);
        params.entry.load_progress.beginReading(total, status.getAppMemFootprintMb(), io_util.nowMsMonotonic(sch.io));
    }

    // Allocate the drafter_path dupe up front so the post-publish step
    // (lower down) has no fallible operations — once we start assigning
    // pointers onto `params.entry`, an OOM during a dupe would leave the
    // entry holding pointers that the per-ptr errdefers would double-free.
    var drafter_path_owned: []u8 = &[_]u8{};
    errdefer if (drafter_path_owned.len > 0) sch.allocator.free(drafter_path_owned);
    if (params.drafter_dir.len > 0) {
        drafter_path_owned = try sch.allocator.dupe(u8, params.drafter_dir);
    }

    // Weights — first mlx call. Binds the stream on this thread.
    const weights_ptr = try sch.allocator.create(Weights);
    errdefer sch.allocator.destroy(weights_ptr);
    weights_ptr.* = try model_mod.loadModelWeights(sch.io, sch.allocator, params.model_dir, params.config, false);
    errdefer weights_ptr.deinit();
    model_mod.resolveWeightPrefix(params.config, weights_ptr);
    try model_mod.narrowHadamardPackTables(params.config, weights_ptr, mlx.gpuStream());

    // Transformer — owns the bulk of the GPU memory.
    const xfm_ptr = try sch.allocator.create(Transformer);
    errdefer sch.allocator.destroy(xfm_ptr);
    xfm_ptr.* = try Transformer.init(sch.io, sch.allocator, params.config.*, weights_ptr);
    errdefer xfm_ptr.deinit();

    // Reserved-token suppression mask (never sample `<|fim_hole|>`-class
    // specials): derived per model from tokenizer + template + eos.
    generate_mod.installSuppressMask(xfm_ptr, params.tok, params.chat_config.chat_template, params.config.eosTokenSlice());

    // Propagate the kv-quant config to the Transformer's own cache. Slot
    // caches in serve mode honor this independently in `Slot.init`; this
    // call covers any path that still touches `xfm.cache` directly (legacy
    // single-slot fallbacks, prompt-cache reuse).
    // Per-model settings (stamped on the config at BOTH construction sites) outrank the flags.
    const kv_quant_config = params.config.kv_quant_override orelse params.kv_quant_config;
    if (kv_quant_config.scheme != .off) {
        try xfm_ptr.cache.reinit(params.config.num_hidden_layers, kv_quant_config);
    }

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm).
    // Policy in mlx.applyWiredPolicy; re-applied in runLoadRequest /
    // runUnloadRequest so `fit` capacity tracks the live set across
    // load/unload churn (this early call covers warmup's forwards).
    logWiredPolicy(mlx.applyWiredPolicy());

    // JIT-compile activation kernels. These are bound to THIS thread's mlx
    // stream — that's exactly the point of doing them here. Skipped entirely
    // when there's no GPU backend (iOS Simulator's CPU-only MLX): mlx_compile
    // is Metal kernel fusion and requests a GPU stream that doesn't exist; the
    // runtime falls back to the uncompiled paths when compiled_* stays null.
    if (!mlx.noGpuBackend()) {
        if (params.config.hidden_act == .gelu_approx) {
            xfm_ptr.compileGelu();
            xfm_ptr.compileGeglu();
        }
        if (params.config.final_logit_softcapping > 0.0) {
            xfm_ptr.compileSoftcap();
        }
        if (xfm_ptr.moe_layers != null) {
            xfm_ptr.compileMoeRouting();
        }
    }

    // Phase 2 experiment: opt-in full-forward Metal fusion via
    // MLX_SERVE_COMPILE_FORWARD=1. This wraps the entire forward pass in
    // mlx_compile so the chunked-prefill loop dispatches a fused graph
    // instead of ~hundreds of separate ops per chunk. Gated because
    // (a) the compiled closure captures `xfm.cache` / `xfm.ssm_entries`
    // as state, and any path that swaps those (multi-slot scheduler,
    // future re-entrant callers) must verify they're not racing the
    // compiled call; (b) mlx_compile with shapeless=false recompiles
    // per unique input shape, which thrashes if the prefill loop sees
    // many different chunk sizes. Tied to byte-equivalence pin in
    // tests/test_phase2_forward_equivalence.sh.
    if (std.c.getenv("MLX_SERVE_COMPILE_FORWARD") != null) {
        const raw = std.c.getenv("MLX_SERVE_COMPILE_FORWARD").?;
        const slice = std.mem.sliceTo(raw, 0);
        if (std.mem.eql(u8, slice, "1")) {
            xfm_ptr.compileForward();
        }
    }

    // The measured round-cost table (`round_cost.zig`) is keyed per (chip,
    // model, quant, OS build); restored here, written at request end.
    {
        var quant_buf: [32]u8 = undefined;
        const quant = std.fmt.bufPrint(&quant_buf, "q{d}g{d}", .{
            params.config.quant_bits,
            params.config.quant_group_size,
        }) catch "q?";
        var os_buf: [64]u8 = undefined;
        const os_build = transformer_mod.macosProductVersion(&os_buf) orelse "";
        // The measured round-cost table rides the same identity: restored
        // here, written at the end of any request that folded new samples.
        // The bucket grid and store version are the arch's (only one arch gets the long
        // grid); every other arch keeps the `rc1` table 26.9.1 wrote and boots warm.
        const rc_layout: round_cost_mod.Layout = round_cost_mod.layoutFor(params.config);
        xfm_ptr.round_cost.layout = rc_layout;
        const rc_key = round_cost_mod.cacheKey(&xfm_ptr.round_cost_key_buf, sys_info.chipBrand(), params.model_dir, quant, os_build, rc_layout, round_cost_mod.engineBuildId());
        xfm_ptr.round_cost_key_len = @intCast(rc_key.len);
        if (round_cost_mod.loadCached(sch.allocator, sch.io, rc_key, rc_layout)) |t| {
            xfm_ptr.round_cost = t;
            log.info("[spec-cost] round-cost table restored ({d} width cells, {d} serial cells)\n", .{ t.restored, t.restored_serial });
            if (t.restored_dropped > 0) log.info("[spec-cost] dropped {d} implausible persisted cell(s)\n", .{t.restored_dropped});
        }
    }

    // Assistant sidecar (optional). Loaded only when `drafter_dir` is
    // non-empty. The sidecar KIND is decided by its config CONTRACT: a
    // config declaring block_size + mask_token_id + target_layer_ids is a
    // DFlash block-drafter (any `*_assistant` family); anything else goes
    // to the Gemma cross-attention drafter loader.
    var drafter_ptr: ?*DrafterModel = null;
    var dflash_ptr: ?*DflashModel = null;
    errdefer if (drafter_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };
    errdefer if (dflash_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };
    const drafter_dir: []const u8 = chosen_drafter orelse in_dir_drafter orelse "";
    if (drafter_dir.len > 0 and dflash_mod.probeIsDflash(sch.io, sch.allocator, drafter_dir)) {
        const env_off = if (std.c.getenv("MLX_SERVE_DFLASH")) |v| v[0] == '0' else false;
        if (env_off) {
            log.info("[dflash] sidecar at {s} skipped (MLX_SERVE_DFLASH=0)\n", .{drafter_dir});
        } else {
            const d = try sch.allocator.create(DflashModel);
            d.* = dflash_mod.loadDflash(sch.io, sch.allocator, mlx.gpuStream(), drafter_dir) catch |err| {
                sch.allocator.destroy(d);
                log.err("Failed to load DFlash assistant at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
                return err;
            };
            d.bind(xfm_ptr) catch |err| {
                d.deinit();
                sch.allocator.destroy(d);
                log.err(
                    "DFlash assistant at {s} is incompatible with target: {s}\n" ++
                        "  (assistant+target must share hidden_size, the mask token and\n" ++
                        "  target_layer_ids must exist in the target, and the target must\n" ++
                        "  run the standard dense-attention forward path)\n",
                    .{ drafter_dir, @errorName(err) },
                );
                return err;
            };
            dflash_ptr = d;
            const wide_lane = dflash_mod.wideVerifyLaneAvailable() or xfm_ptr.ternaryWideVerify();
            const block_cap = dflash_mod.blockCapForMachine(sys_info.chipBrand(), d.selector != null and xfm_ptr.specTreeSupported());
            sch.drafter_block_size = dflash_mod.resolveBlockSize(
                d.config.block_size,
                params.draft_block_size,
                params.draft_block_size_explicit,
                wide_lane,
                block_cap.cap,
            );
            if (params.draft_block_size_explicit and params.draft_block_size > sch.drafter_block_size)
                log.warn("--draft-block-size {d} is past the drafter's trained block; using {d}\n", .{ params.draft_block_size, sch.drafter_block_size });
            // Trees on the tensor units draft past the drafter's trained block (TensorFold's
            // tree_block): the lattice reaches depth 15 and a 16-row window costs about an 8-row one.
            if (!params.draft_block_size_explicit and d.selector != null and xfm_ptr.specTreeSupported() and transformer_mod.naxAvailable())
                sch.drafter_block_size = dflash_mod.TREE_NAX_BLOCK;
            var cap_note_buf: [96]u8 = undefined;
            const cap_note: []const u8 = if (params.draft_block_size_explicit)
                ", user-clamped"
            else if (!wide_lane and d.config.block_size > sch.drafter_block_size)
                std.fmt.bufPrint(&cap_note_buf, ", capped ({s} cap {d})", .{
                    block_cap.label,
                    block_cap.cap,
                }) catch ", capped"
            else
                "";
            log.info("DFlash drafter ready (block_size={d}{s}, wide_verify_lane={}, targets={any}).\n", .{
                sch.drafter_block_size,
                cap_note,
                wide_lane,
                d.config.target_layer_ids,
            });
            const tiled_bytes = try xfm_ptr.tileLaneWeights();
            if (tiled_bytes > 0) log.info("[lane] trunk projections tiled in place ({d:.1} GB, no copy)\n", .{@as(f64, @floatFromInt(tiled_bytes)) / (1 << 30)});
            const drafter_tiled = try d.tileLaneWeights(xfm_ptr.actDtype(), mlx.gpuStream());
            if (drafter_tiled > 0) log.info("[lane] drafter projections tiled in place ({d:.2} GB, no copy)\n", .{@as(f64, @floatFromInt(drafter_tiled)) / (1 << 30)});
        }
    } else if (drafter_dir.len > 0) {
        const d = try sch.allocator.create(DrafterModel);
        d.* = drafter_mod.loadDrafter(sch.io, sch.allocator, mlx.gpuStream(), drafter_dir) catch |err| {
            sch.allocator.destroy(d);
            log.err("Failed to load drafter at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
            return err;
        };
        d.bind(xfm_ptr) catch |err| {
            d.deinit();
            sch.allocator.destroy(d);
            log.err(
                "Drafter checkpoint at {s} is incompatible with target: {s}\n" ++
                    "  (drafter+target must share backbone_hidden_size, vocab_size, and have\n" ++
                    "  matching layer types in the target's non-shared K/V layers)\n",
                .{ drafter_dir, @errorName(err) },
            );
            return err;
        };
        drafter_ptr = d;

        // Auto-detect block_size unless the user pinned it explicitly.
        if (!params.draft_block_size_explicit) {
            const auto_bs = drafter_mod.recommendedBlockSize(params.config);
            sch.drafter_block_size = auto_bs;
            log.info(
                "Drafter ready (block_size={d}, auto-detected for {s}/{d}-layer{s}).\n",
                .{
                    auto_bs,
                    params.config.model_type,
                    params.config.num_hidden_layers,
                    if (params.config.isMoe()) ",moe" else "",
                },
            );
        } else {
            log.info("Drafter ready (block_size={d}, user override).\n", .{params.draft_block_size});
        }

        if (params.config.isMoe()) {
            log.warn(
                "Drafter loaded but target is MoE ({s}); per-request " ++
                    "enable_drafter defaults to OFF — drafter+MoE regresses " ++
                    "at single-stream batch=1 (verify forward expert-routing " ++
                    "penalty). Pass enable_drafter:true per request to opt-in.\n",
                .{params.config.model_type},
            );
        }
    }
    params.config.row_exact_covered = xfm_ptr.config.row_exact_covered;
    params.config.dflash_bound = xfm_ptr.config.dflash_bound;

    // After the drafter binds: an exact-mode arch runs its row kernels only then.
    // DIAGNOSTIC (MLX_SERVE_DECODE_FWD_UBENCH=N): time N decode-width forward
    // passes back to back, with NO sampling, detokenization, stop-checking or
    // cache bookkeeping around them. The server reports `predicted_ms` around
    // the whole decode LOOP, so this is the only way to say how much of a
    // token is the model and how much is everything else. Resets the KV cache
    // afterwards so the probe cannot pollute real requests.
    if (std.c.getenv("MLX_SERVE_DECODE_FWD_UBENCH")) |raw| {
        const n = std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        if (n > 0) {
            const io_u = @import("io_util.zig");
            const tio = std.Io.Threaded.global_single_threaded.io();
            var ctx = xfm_ptr.defaultCtx();
            // MLX_SERVE_DECODE_FWD_UBENCH_S=<rows>: verify-width forwards
            // (per-position state capture on, as spec verify runs them).
            // MLX_SERVE_DECODE_FWD_UBENCH_KV=<tokens>: prefill that many
            // tokens first so the meter runs at a real context length.
            const rows: usize = blk: {
                const r = std.c.getenv("MLX_SERVE_DECODE_FWD_UBENCH_S") orelse break :blk 1;
                break :blk @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 1);
            };
            const kv_pre: usize = blk: {
                const r = std.c.getenv("MLX_SERVE_DECODE_FWD_UBENCH_KV") orelse break :blk 0;
                break :blk std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0;
            };
            const tok_slice = try sch.allocator.alloc(i32, @min(rows, 4096));
            defer sch.allocator.free(tok_slice);
            for (tok_slice, 0..) |*v, i| v.* = @intCast(1 + (i % 997));
            const tok = tok_slice.ptr;
            const tsh = [_]c_int{ 1, @intCast(tok_slice.len) };
            if (kv_pre > 0) {
                var done_pre: usize = 0;
                const pre_buf = try sch.allocator.alloc(i32, 2048);
                defer sch.allocator.free(pre_buf);
                for (pre_buf, 0..) |*v, i| v.* = @intCast(1 + (i % 1000));
                while (done_pre < kv_pre) {
                    const n_chunk = @min(2048, kv_pre - done_pre);
                    const psh = [_]c_int{ 1, @intCast(n_chunk) };
                    const ti = mlx.mlx_array_new_data(pre_buf.ptr, &psh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                    done_pre += n_chunk;
                }
                log.info("[fwd-ubench] prefilled {d} tokens\n", .{done_pre});
            }
            // Rows past 16 are a prefill chunk unless `_KV` says a context stands behind them (verify widths up to 32).
            const prefill_rows = rows > 16 and kv_pre == 0;
            // MLX_SERVE_DECODE_FWD_UBENCH_TREE=1 at 16 rows: verify a fixed draft
            // tree (a 9-row trunk, siblings at depths 1-4, their children), as a round does.
            const tree_parents = [16]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 9, 10, 11 };
            var tree_depth: [16]i32 = undefined;
            for (tree_parents, 0..) |p, i| tree_depth[i] = if (p < 0) 0 else tree_depth[@intCast(p)] + 1;
            const spec_tree = generate_mod.Generator.specTreeFor(&tree_parents, &tree_depth, 8);
            defer spec_tree.deinit();
            if (rows == 16 and std.c.getenv("MLX_SERVE_DECODE_FWD_UBENCH_TREE") != null) ctx.tree = &spec_tree;
            // Prefill widths: every forward starts from an empty cache (else each
            // one attends over the previous ones' rows) and skips the lm_head,
            // which a real intermediate chunk never evaluates.
            ctx.skip_lm_head = prefill_rows;
            log.info("[fwd-ubench] rows={d} prefill={}\n", .{ tok_slice.len, prefill_rows });
            // Warm: first forward pays kernel JIT + lazy weight materialization.
            for (0..3) |_| {
                if (prefill_rows) try xfm_ptr.resetCache();
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            // DIAGNOSTIC (MLX_SERVE_DECODE_GRAPH_DUMP=<path>): print the lazy
            // graph of ONE forward (every primitive with its shape) before it
            // is evaluated, to map which ops a decode step dispatches.
            if (std.c.getenv("MLX_SERVE_DECODE_GRAPH_DUMP")) |path| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                if (xfm_ptr.forwardWith(&ctx, ti)) |lg| {
                    defer _ = mlx.mlx_array_free(lg);
                    if (std.c.fopen(path, "w")) |f| {
                        const outs = mlx.mlx_vector_array_new_value(lg);
                        defer _ = mlx.mlx_vector_array_free(outs);
                        const namer = mlx.mlx_node_namer_new();
                        defer _ = mlx.mlx_node_namer_free(namer);
                        _ = mlx.mlx_print_graph(f, namer, outs);
                        _ = std.c.fclose(f);
                        log.info("[fwd-ubench] graph dumped to {s}\n", .{std.mem.sliceTo(path, 0)});
                    }
                    _ = mlx.mlx_array_eval(lg);
                } else |_| {}
            }
            // Split CPU graph CONSTRUCTION from GPU execution. MLX is lazy, so
            // `forwardWith` only issues ops — if that half dominates, the token
            // is bounded by op count / FFI overhead, not by memory bandwidth,
            // and no kernel-level optimization can reach it.
            var sw = io_u.Stopwatch.init(tio);
            var build_ns: u64 = 0;
            var eval_ns: u64 = 0;
            var encode_ns: u64 = 0;
            var ops_total: u64 = 0;
            var done: usize = 0;
            for (0..n) |_| {
                if (prefill_rows) try xfm_ptr.resetCache();
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const ops_before = mlx.op_count.load(.monotonic);
                var swb = io_u.Stopwatch.init(tio);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                build_ns += swb.read();
                ops_total += mlx.op_count.load(.monotonic) - ops_before;
                // The eval is two host phases: encoding every kernel (MLX runs `eval_gpu` on the
                // calling thread, throttled to MAX_ACTIVE_TASKS open command buffers), then the
                // wait for the GPU. The split says which side bounds the forward.
                var swe = io_u.Stopwatch.init(tio);
                const lgv = mlx.mlx_vector_array_new_value(lg);
                _ = mlx.mlx_async_eval(lgv);
                _ = mlx.mlx_vector_array_free(lgv);
                encode_ns += swe.read();
                _ = mlx.mlx_array_eval(lg);
                eval_ns += swe.read();
                _ = mlx.mlx_array_free(lg);
                done += 1;
            }
            const dn: f64 = @floatFromInt(@max(done, 1));
            const ms = if (prefill_rows) @as(f64, @floatFromInt(build_ns + eval_ns)) / 1.0e6 / dn else @as(f64, @floatFromInt(sw.read())) / 1.0e6 / dn;
            transformer_mod.decodeProfReport();
            log.info("[fwd-ubench] {d} decode forwards, eval-per-step: {d:.3} ms/forward (build {d:.3} ms CPU + eval {d:.3} ms GPU [encode {d:.3} + wait {d:.3}], {d:.0} ops/forward)\n", .{
                done,
                ms,
                @as(f64, @floatFromInt(build_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(encode_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns - encode_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(ops_total)) / dn,
            });

            // Same forward with the vocab projection suppressed. lm_head is
            // terminal — nothing downstream depends on it — so dropping it
            // cannot change the work the rest of the graph does, which makes
            // this the one sound ablation in the probe.
            ctx.skip_lm_head = true;
            for (0..3) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            var sw_nolm = io_u.Stopwatch.init(tio);
            var done_nolm: usize = 0;
            for (0..n) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
                done_nolm += 1;
            }
            const ms_nolm = @as(f64, @floatFromInt(sw_nolm.read())) / 1.0e6 / @as(f64, @floatFromInt(@max(done_nolm, 1)));
            ctx.skip_lm_head = false;
            log.info("[fwd-ubench] without lm_head: {d:.3} ms/forward  => lm_head = {d:.3} ms\n", .{ ms_nolm, ms - ms_nolm });
            xfm_ptr.diagProjBench(20);
            log.info("[fwd-ubench] done\n", .{});
            xfm_ptr.resetCache() catch {};
        }
    }

    // Eager warmup: faults weight pages + compiles the decode-path kernels
    // on this thread's stream. ~600-900 ms at boot but the first user request
    // skips a cold path — observed savings on Gemma 4 E4B 4-bit.
    if (params.warmup_eager) {
        params.entry.load_progress.begin(.warming_up, 0, io_util.nowMsMonotonic(sch.io));
        const warmup_start = std.Io.Timestamp.now(sch.io, .awake);
        xfm_ptr.warmup() catch |err| {
            log.warn("Warmup failed ({s}); continuing without it, the first request may be slow.\n", .{@errorName(err)});
        };
        const warmup_ns: u64 = @intCast(warmup_start.untilNow(sch.io, .awake).nanoseconds);
        log.info("Warmup complete ({d} ms).\n", .{warmup_ns / std.time.ns_per_ms});
    }

    // A MERGED drafter has no `--drafter` to echo: the reported path comes
    // from what was actually resolved, so `drafter_loaded` and `drafter_path`
    // must not disagree about the same sidecar. Allocated here, before the latch drop below and Phase 05, which
    // must not fail.
    if (drafter_path_owned.len == 0 and drafter_dir.len > 0 and
        (dflash_ptr != null or drafter_ptr != null))
    {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
    }

    if (!load_had_error and mlx.errorPending()) {
        var buf: [512]u8 = undefined;
        const msg = mlx.takeError(&buf) orelse "";
        log.warn("[load] dropped an MLX error an optional load step latched and fell back from; the first request starts clean: {s}\n", .{msg});
    }

    // ── Phase 05: install everything onto the LoadedModel entry, mark
    //    ready, and update the scheduler's borrowed views. The registry
    //    mutex guards the state transition + `current_resident_bytes`
    //    accounting; `state_cond.broadcast` (inside markReadyLocked) wakes
    //    any waiter blocked in ensureLoaded.
    //
    //    Everything below this comment must be infallible — once we begin
    //    assigning to `params.entry`, the per-ptr errdefers above would
    //    double-free if we error-return. `drafter_path_owned` was alloc'd
    //    up front for exactly this reason; the per-model prefix cache
    //    init is a struct literal (no fallible alloc).
    const entry = params.entry;
    entry.weights = weights_ptr;
    entry.transformer = xfm_ptr;
    entry.drafter = drafter_ptr;
    entry.dflash = dflash_ptr;
    if (entry.config) |c| c.drafter_ctx_bytes_per_token = if (dflash_ptr) |d| dflash_mod.ctxBytesPerToken(&d.config) else 0;
    entry.drafter_block_size = sch.drafter_block_size;
    entry.drafter_path = drafter_path_owned;
    drafter_path_owned = &[_]u8{}; // disarm the errdefer
    // Transfer ownership of the heap-allocated CPU state from `params` to
    // the entry. The caller (main.zig) MUST NOT free these — `LoadedModel.deinit`
    // walks them in the same `*X` pointer form they came in.
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    // Per-model hot prefix cache (Plan 03 → Plan 05 move).
    const ram_prefix_cache = params.prefix_cache_ram_enabled;
    const disk_prefix_cache = params.prefix_cache_disk_bytes > 0;
    if (params.prefix_cache_capacity > 0 and (ram_prefix_cache or disk_prefix_cache)) {
        // The weights are resident here, so the resolver's active-memory read
        // is honest; the raw launch budget never reaches initWithMem (a 40 GB
        // cap beside a ~70 GB pack was the 2026-08-30 uncatchable Metal OOM).
        // Disk-only mode retains no reusable KV in RAM, so it asks for no RAM budget.
        var ssd_idle_mem: u64 = 0;
        const clamped_prefix_mem: u64 = if (!ram_prefix_cache)
            0
        else if (params.prefix_cache_mem_resolver) |resolve|
            resolve(params.config, params.prefix_cache_mem_bytes, .{}, &ssd_idle_mem)
        else
            params.prefix_cache_mem_bytes;
        entry.prefix_cache = prefix_cache_mod.HotPrefixCache.initWithMem(
            sch.allocator,
            if (ram_prefix_cache) params.prefix_cache_capacity else 0,
            clamped_prefix_mem,
        );
        entry.prefix_cache.?.ssd_idle_mem = ssd_idle_mem;
        // SSD tier (`--prefix-cache-disk`). Every failure mode is caught: persistence silently
        // stays off, the RAM cache is unaffected.
        if (params.prefix_cache_disk_bytes > 0) attach: {
            const fp = kv_disk_cache.modelFingerprintWithLayout(
                sch.allocator,
                sch.io,
                entry.path,
                params.config.cacheLayoutNamespace(),
            ) catch |err| {
                log.warn("[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
                break :attach;
            };
            defer sch.allocator.free(fp);
            const base = kv_disk_cache.defaultBaseDir(sch.allocator) catch break :attach;
            defer sch.allocator.free(base);
            entry.prefix_cache.?.disk = kv_disk_cache.DiskTier.init(
                sch.allocator,
                sch.io,
                base,
                fp,
                params.prefix_cache_disk_bytes,
                kv_disk_cache.DEFAULT_CHUNK_TOKENS,
            ) catch |err| {
                log.warn("[disk-cache] init failed: {s} — persistence off for this model\n", .{@errorName(err)});
                break :attach;
            };
        }
        // SSD-first: arch + env switch + a live disk tier. Below the attach because the tier
        // is part of the answer.
        entry.prefix_cache.?.armDiskTier(params.config);
    }
    // Iteration 2 (perf-plan Phase 4 #3): tokenize cache for warm-path
    // chat-template renders. Funnels through `chat_mod.formatChat`
    // at the handler boundary.
    // Default capacity is small (4 entries) because chat conversations
    // mutate the messages list every turn; the goal is to catch warm
    // reuse benches and repeated agent-loop probes, not to memoize a
    // full session.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }
    // Commit the same validated weight estimate used by the eviction gate
    // and preflight (zero means no disk estimate), plus a GPU-resident n-gram table.
    const bytes_resident = committedTextBytes(model_bytes, params.config);

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Set borrowed views for scheduler-internal code (and `current_model`
    // for ensureLoaded → request handlers in Phase C).
    sch.current_model = entry;
    sch.weights = weights_ptr;
    sch.xfm = xfm_ptr;
    sch.drafter = drafter_ptr;
    sch.dflash = dflash_ptr;
    // The connection thread bills chat against the view, so a model that cannot chat never takes it.
    if (entry.prefix_cache) |*hc| {
        if (ModelRegistry.chatCapable(entry)) sch.hot_prefix_cache = hc;
    }
    publishHotCacheResidency(sch);
}

/// Republish the hot cache's residency for the connection thread's admission guard.
/// Called from the inference thread after a commit, eviction, invalidation or model switch.
const BUDGET_REVISE_WINDOW_NS: u64 = 10 * std.time.ns_per_s;

/// How a budget resolve differs from the load-time one: the cache's own resident bytes
/// are excluded from the machine read, and the resolver keeps quiet (`setBudget` logs).
pub const BudgetRevise = struct { exclude_bytes: u64 = 0, quiet: bool = false };

/// Re-clamp every resident model's hot-cache budget after residency changed (#364): the
/// load-time clamp read the machine with the other models on it and was never revisited, so
/// a model loaded beside a large one kept a ~0 budget for life. Each cache's own resident
/// entries are excluded from the read so a full cache cannot ratchet itself down.
/// `grow_only`: the free figure lags an unload's freed pages, so a revise that follows one may
/// raise a budget but never lower it; lowering evicts entries the settled figure would keep.
fn reviseHotCacheBudgets(sch: *Scheduler, grow_only: bool) void {
    const resolve = sch.prefix_cache_mem_resolver orelse return;
    sch.registry.mutex.lockUncancelable(sch.io);
    // The resolver publishes the process-global budget the admission guard reads, so the
    // current model goes last.
    for ([_]bool{ false, true }) |current_pass| {
        var it = sch.registry.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if ((entry == sch.current_model) != current_pass) continue;
            if (entry.state != .ready) continue;
            const hc = if (entry.prefix_cache) |*h| h else continue;
            if (!hc.ram_enabled) continue;
            const config = entry.config orelse continue;
            var idle: u64 = 0;
            const budget = resolve(config, sch.prefix_cache_mem_bytes, .{ .exclude_bytes = hc.residentBytes(), .quiet = true }, &idle);
            if (grow_only and prefix_cache_mod.HotPrefixCache.budgetShrinks(hc.max_kv_bytes, budget)) continue;
            hc.setBudget(budget);
            hc.ssd_idle_mem = idle;
        }
    }
    sch.registry.mutex.unlock(sch.io);
    publishHotCacheResidency(sch);
}

/// The view must not outlive its cache: before `leaving`'s cache is freed, hand the view to
/// another resident chat model's, the default's when it has one, or to none.
fn moveHotCacheViewOff(sch: *Scheduler, leaving: *LoadedModel) void {
    const view = sch.hot_prefix_cache orelse return;
    if (leaving.prefix_cache == null or view != &leaving.prefix_cache.?) return;
    sch.hot_prefix_cache = null;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        if (entry == leaving or entry.state != .ready or !ModelRegistry.chatCapable(entry)) continue;
        const hc = if (entry.prefix_cache) |*h| h else continue;
        sch.hot_prefix_cache = hc;
        if (std.mem.eql(u8, sch.registry.default_id, entry.id)) break;
    }
}

pub fn publishHotCacheResidency(sch: *Scheduler) void {
    const bytes: u64 = if (sch.hot_prefix_cache) |hc| hc.residentBytes() else 0;
    sch.resident_hot_cache_bytes.store(bytes, .monotonic);
    const reclaimable: u64 = if (sch.hot_prefix_cache) |hc| hc.reclaimableBytes() else 0;
    sch.reclaimable_hot_cache_bytes.store(reclaimable, .monotonic);
    publishHotCacheDigests(sch);
    if (sch.metrics != null) publishCachedSessions(sch);
}

/// Caller must not hold `registry.mutex`.
fn publishCachedSessions(sch: *Scheduler) void {
    var rows: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined;
    var n: usize = 0;
    {
        sch.registry.mutex.lockUncancelable(sch.io);
        defer sch.registry.mutex.unlock(sch.io);
        var it = sch.registry.entries.valueIterator();
        outer: while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.state != .ready) continue;
            const hc = if (entry.prefix_cache) |*h| h else continue;
            for (hc.entries.items) |*e| {
                if (n == rows.len) break :outer;
                const len: u32 = @intCast(@min(e.tokens.len, std.math.maxInt(u32)));
                rows[n] = .init(entry.id, .cached, len, len, 0, e.kv_bytes);
                rows[n].entry_id = e.id;
                n += 1;
            }
        }
    }
    sch.digest_mu.lockUncancelable(sch.io);
    defer sch.digest_mu.unlock(sch.io);
    @memcpy(sch.cached_sessions[0..n], rows[0..n]);
    sch.cached_session_count = n;
}

/// Swap in a fresh digest snapshot and free the one it supersedes (inference thread only).
/// An allocation failure keeps the previous snapshot: a stale digest is a hint, an empty one
/// credits bytes that exist.
fn publishHotCacheDigests(sch: *Scheduler) void {
    const fresh: []prefix_cache_mod.HotPrefixCache.EntryDigest = if (sch.hot_prefix_cache) |hc|
        (hc.digestsAlloc(sch.allocator) catch return)
    else
        &.{};
    const residency: u64 = if (sch.hot_prefix_cache) |hc| hc.residentBytes() else 0;
    sch.digest_mu.lockUncancelable(sch.io);
    const old = sch.hot_cache_digests;
    sch.hot_cache_digests = fresh;
    // Published under the same lock as the digests it describes.
    sch.digest_residency = residency;
    sch.digest_mu.unlock(sch.io);
    if (old.len > 0) sch.allocator.free(old);
}

/// Connection-thread entry point: what an eviction pass can prove it will get back for this prompt.
pub fn reclaimableHotCacheBytesFor(sch: *Scheduler, prompt_tokens: []const u32) u64 {
    const fp = prefix_cache_mod.HotPrefixCache.prefixFingerprint(prompt_tokens);
    sch.digest_mu.lockUncancelable(sch.io);
    defer sch.digest_mu.unlock(sch.io);
    return prefix_cache_mod.HotPrefixCache.reclaimableFromDigests(
        sch.hot_cache_digests,
        sch.digest_residency,
        fp,
    );
}

/// Caller holds `queue_mu`. Shared with the wait condition below.
/// One bounded SSD slice per resident model that still owes one (its oldest owed session first);
/// returns whether any still does. The caller re-checks for work between calls.
/// Inference thread only: unloads run here too, so the collected pointers stay valid.
fn flushOwedDiskSlices(sch: *Scheduler) bool {
    var owed: [16]prefix_cache_mod.FlushTarget = undefined;
    var still = false;
    for (owed[0..diskCaches(sch, &owed, true)]) |o| {
        if (o.hc.flushOwedSlice(o.s)) still = true;
    }
    return still;
}

/// The hot caches of ready models with a disk tier (only those owing a slice when `owed_only`).
/// Inference thread only: unloads run here too, so the collected pointers stay valid.
fn diskCaches(sch: *Scheduler, out: []prefix_cache_mod.FlushTarget, owed_only: bool) usize {
    var n: usize = 0;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        if (n == out.len) break;
        const entry = entry_ptr.*;
        if (entry.state != .ready) continue;
        const hc = if (entry.prefix_cache) |*h| h else continue;
        if (hc.disk == null or (owed_only and !hc.disk_dirty)) continue;
        const xf = entry.transformer orelse continue;
        out[n] = .{ .hc = hc, .s = xf.s };
        n += 1;
    }
    return n;
}

/// SIGTERM's bounded SSD flush (`--shutdown-persist-secs`), on this thread once the loop has
/// stopped: what the idle flush has not landed, until the deadline `serve` armed. Off when unarmed.
fn persistAtShutdown(sch: *Scheduler) void {
    const deadline_ms = sch.shutdown_deadline_ms.load(.acquire);
    if (deadline_ms == 0) return;
    // A slot cancelled as `shutdown` flipped can still sit in the cleanup queue: commit it and
    // return its checkout, or its session reads as checked out.
    _ = drainSideQueues(sch);
    var targets: [16]prefix_cache_mod.FlushTarget = undefined;
    const n = diskCaches(sch, &targets, false);
    if (n == 0) return;
    const sw = io_util.Stopwatch.init(sch.io);
    const t = prefix_cache_mod.shutdownPersist(targets[0..n], io_util.MonotonicClock{ .io = sch.io }, deadline_ms);
    const mb = 1024.0 * 1024.0;
    log.info("[disk-cache] shutdown persist: {d} written ({d:.1} MB), {d} left (deadline), {d} left (store declined), {d} skipped checked-out, {d} failures, {d:.1} MB dropped at the deadline, {d:.1} s\n", .{
        t.written,
        @as(f64, @floatFromInt(t.bytes)) / mb,
        t.left_deadline,
        t.left_declined,
        t.checked_out,
        t.failures,
        @as(f64, @floatFromInt(t.dropped_bytes)) / mb,
        @as(f64, @floatFromInt(sw.read())) / std.time.ns_per_s,
    });
}

/// Arm the shutdown flush's deadline `secs` after the first call; a later call (a second
/// SIGTERM) never moves it. 0 seconds leaves it unarmed, which turns the flush off.
pub fn armShutdownDeadline(slot: *std.atomic.Value(i64), now_ms: i64, secs: u32) void {
    if (secs == 0) return;
    _ = slot.cmpxchgStrong(0, now_ms + @as(i64, secs) * 1000, .acq_rel, .acquire);
}

fn hasWorkPendingLocked(sch: *const Scheduler) bool {
    return sch.pending.items.len > 0 or
        sch.decoding.items.len > 0 or
        sch.embed_queue.items.len > 0 or
        sch.cleanup_queue.items.len > 0 or
        sch.load_queue.items.len > 0 or
        sch.unload_queue.items.len > 0;
}

/// Poll interval for the idle-eviction sweep, given the configured window.
///
/// A quarter of the window, so a model is evicted within ~1.25x the configured
/// idle time rather than up to 2x it. Floored at a second so `--idle-evict-secs 1`
/// does not spin, capped at 30s so a long window still costs ~nothing.
pub fn idleEvictTickMs(window_ms: i64) i64 {
    return @max(1000, @min(@divTrunc(window_ms, 4), 30_000));
}

fn inferenceLoop(ctx: ThreadCtx) void {
    const sch = ctx.scheduler;
    const params = ctx.params;
    // Covers shutdown and startup-load failure.
    defer sleep_inhibit.release();

    // ── Phase A1 → Plan 05: load runs on this thread (mlx GPU stream
    //    binding). On failure, mark the entry `.error_state` in the
    //    registry AND set load_failed so `Scheduler.init`'s parent sees
    //    the same shape it always has.
    //
    // Headless boot (`no_initial_load`): start with NO primary model. The
    // server runs idle until a chat or media model is loaded on demand by
    // a request that names a discovered id.
    // Headless start: models load on demand.
    if (params.no_initial_load) {
        log.info("Headless: no primary model loaded; models load on demand.\n", .{});
        signalStarted(sch);
    } else {
        // Startup load runs before the wait loop can acquire.
        sleep_inhibit.setActive(true);
        // `.loading` while it runs, so the startup responder's `/v1/load-progress` lists it.
        sch.registry.mutex.lockUncancelable(sch.io);
        _ = sch.registry.tryBeginLoadLocked(params.entry);
        sch.registry.mutex.unlock(sch.io);
        if (doLoadOnInferenceThread(sch, params)) |_| {
            log.info("Model ready (loaded on inference thread).\n", .{});
            signalStarted(sch);
        } else |err| {
            recordLoadError(sch, @errorName(err));
            sch.registry.mutex.lockUncancelable(sch.io);
            sch.registry.markErrorLocked(params.entry, @errorName(err));
            sch.registry.mutex.unlock(sch.io);
            signalStarted(sch);
            return;
        }
    }

    while (!sch.shutdown.load(.acquire)) {
        // Phase D: cold-load drain. Process ONE load per tick — loading a
        // model is heavy (~seconds; weight read + JIT compile + warmup)
        // and we want the rest of the inference loop to stay responsive.
        // Other pending loads wait in queue and get picked up next tick.
        var load_req: ?*LoadRequest = null;
        // Unload work items (one per tick, like load, both are heavy and we
        // re-check the loop between them).
        var unload_req: ?*UnloadRequest = null;
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            if (sch.load_queue.items.len > 0) {
                load_req = sch.load_queue.orderedRemove(0);
            }
            if (sch.unload_queue.items.len > 0) {
                unload_req = sch.unload_queue.orderedRemove(0);
            }
        }
        _ = drainSideQueues(sch);
        if (load_req) |req| runLoadRequest(sch, req);
        if (unload_req) |req| runUnloadRequest(sch, req);
        if (load_req != null or unload_req != null) reviseHotCacheBudgets(sch, load_req == null);
        if (unload_req != null) sch.budget_revise_sw = io_util.Stopwatch.init(sch.io);
        if (chatPass(sch, .main) == .shutdown) break;
    }
    persistAtShutdown(sch);
}

/// Cleanup and embed work, drained on the inference thread. Returns
/// whether there was any.
fn drainSideQueues(sch: *Scheduler) bool {
    // 0a. Drain slots queued for cleanup. Conn threads hand finished
    //     slots here in `complete()` — we own the mlx stream binding,
    //     so freeing per-slot KVCache + vision_embeddings + ssm_entries
    //     is safe here even though those slots' arrays might trigger
    //     real GPU memory release on refcount-zero.
    var cleanup_batch: [16]*Slot = undefined;
    var cleanup_n: usize = 0;
    // 0b. Drain any pending embed work. These run synchronously on
    //     behalf of conn threads waiting in `computeEmbedding`. Processed
    //     here (not concurrently with decode ticks) so they share the
    //     inference thread's mlx stream cleanly.
    var embed_batch: [4]*EmbedRequest = undefined;
    var embed_n: usize = 0;
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        while (cleanup_n < cleanup_batch.len and sch.cleanup_queue.items.len > 0) {
            cleanup_batch[cleanup_n] = sch.cleanup_queue.orderedRemove(0);
            cleanup_n += 1;
        }
        while (embed_n < embed_batch.len and sch.embed_queue.items.len > 0) {
            embed_batch[embed_n] = sch.embed_queue.orderedRemove(0);
            embed_n += 1;
        }
    }
    for (cleanup_batch[0..cleanup_n]) |s| {
        // Decode-phase cancel: `complete()` pulled this slot straight
        // into the cleanup queue, so it never went through finishSlot
        // and its committed KV (prompt + every emitted token) would die
        // right here with the slot. Commit it first — the same guards
        // as a normal finish apply inside (pad-only / error / vision /
        // empty all decline) — then flush what was committed to the
        // SSD tier, since no finishSlot will. Normally-finished slots
        // arrive here with `finished` already set (finishSlot committed
        // them) and skip; errored slots decline via the error guard.
        // Runs on the inference thread — the sole mlx caller — which is
        // what makes the refcount-sharing snapshot legal here.
        if (s.cancelled.load(.acquire) and !s.finished and s.error_code == null) {
            commitSlotIfApplicable(sch, s);
            if (s.model.prefix_cache) |*hc| {
                if (s.model.transformer) |xf| hc.flushPendingDisk(xf.s);
            }
        }
        recordSlotCleanup(sch.metrics, s);
        // Second slot-end path (a decode-phase cancel never reaches finishSlot); the
        // record must not outlive the bytes `s.deinit()` frees.
        if (s.model.prefix_cache) |*hc| hc.releaseCheckout(@intFromPtr(s), "slot cleanup");
        s.deinit();
    }
    for (embed_batch[0..embed_n]) |req| runEmbedRequest(sch, req);
    return cleanup_n + embed_n > 0;
}

const ChatPassMode = enum { main, yield };
const ChatPassResult = enum { shutdown, idle, ran };

/// What became of a slot `prefillAdmitted` took through admission and prefill.
const AdmitOutcome = enum {
    /// Prefilled and appended to `decoding`.
    decoding,
    /// Finished before or during its prefill (cancelled).
    finished,
    /// Failed; the slot carries the error.
    errored,
    /// Does not fit beside live requests yet: the caller puts it back in `pending`.
    held,
    /// After the hot-cache restore the uncached tail is over the slot's boundary limit.
    tail_too_large,
};

/// One admitted slot's cancelled check, memory hold, prefill, first-token
/// publish and move into `decoding`. The caller holds the slot's `in_pass`.
fn prefillAdmitted(sch: *Scheduler, slot: *Slot) AdmitOutcome {
    if (slot.cancelled.load(.acquire)) {
        // finishSlot (not raw markFinished) so the metrics sink
        // counts the cancellation; safe pre-prefill — commit
        // no-ops with legacy_gen==null.
        finishSlot(sch, slot, "cancelled");
        return .finished;
    }
    if (slotHoldsForMemory(sch, slot)) return .held;
    var prefill_sw = io_util.Stopwatch.init(sch.io);
    prefill: while (true) {
        runPrefill(sch, slot) catch |err| {
            if (err == error.Cancelled) {
                log.info("[scheduler] prefill aborted: client disconnected\n", .{});
                finishSlot(sch, slot, "cancelled");
                return .finished;
            }
            if (err == error.BoundaryTailTooLarge) return .tail_too_large;
            log.err("[scheduler] prefill failed for slot: {s}\n", .{@errorName(err)});
            slot.markError(@errorName(err));
            return .errored;
        };
        break :prefill;
    }
    if (slot.state == .errored or slot.cancelled.load(.acquire)) return .errored;
    slot.prefill_ns = prefill_sw.read() -| slot.prefill_interleaved_ns;
    publishFirstToken(slot);
    if (slot.prefill_interleaved_ns > 0) log.debug("[interleave] prefill {d} ms, hosted decode {d} ms\n", .{
        slot.prefill_ns / std.time.ns_per_ms, slot.prefill_interleaved_ns / std.time.ns_per_ms,
    });
    // Exact time-to-first-token: elapsed from request arrival
    // (Slot.init, pre-queue-wait) to prefill completion. Captured
    // here rather than derived by subtraction in finishSlot, so a
    // slot that finishes mid-tick can't skew it (metrics TTFT fix).
    slot.first_token_ns = @intCast(slot.request_start_ts.untilNow(sch.io, .boot).nanoseconds);
    sch.queue_mu.lockUncancelable(sch.io);
    sch.decoding.append(sch.allocator, slot) catch |err| {
        sch.queue_mu.unlock(sch.io);
        slot.markError(@errorName(err));
        return .errored;
    };
    sch.queue_mu.unlock(sch.io);
    return .decoding;
}

/// One admission -> prefill -> decode tick -> cull pass. `.main` parks while there is no
/// work at all; `.yield` runs inside a media job, so it never parks and never touches the
/// load, unload or gen queues. `.ran` = something advanced.
fn chatPass(sch: *Scheduler, mode: ChatPassMode) ChatPassResult {
    // 1. Wait for work. Drain pending slots into a local list under lock,
    //    run prefills outside the lock.
    var ran = mode == .yield and drainSideQueues(sch);
    var to_prefill: [16]*Slot = undefined;
    var n_prefill: usize = 0;
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        if (mode == .main) {
            var disk_owed = true;
            while (!hasWorkPendingLocked(sch) and !sch.shutdown.load(.acquire)) {
                // Idle: land owed SSD slices before parking, so a later commit rarely finds one.
                if (disk_owed) {
                    sch.queue_mu.unlock(sch.io);
                    disk_owed = flushOwedDiskSlices(sch);
                    sch.queue_mu.lockUncancelable(sch.io);
                    continue;
                }
                // No later tick runs while parked, so release here.
                sleep_inhibit.setActive(false);
                sch.queue_cond.waitUncancelable(sch.io, &sch.queue_mu);
            }
            // Hold until the loop parks again.
            if (!sch.shutdown.load(.acquire)) sleep_inhibit.setActive(true);
        }
        if (sch.shutdown.load(.acquire)) return .shutdown;

        // If only vision/embed/cleanup/load work is pending, loop back to drain it.
        if (sch.pending.items.len == 0 and sch.decoding.items.len == 0) return if (ran) .ran else .idle;

        // Single-flight admission (the module-owned class): a model with
        // MODULE-OWNED decode state admits at most one live slot.
        // Snapshot live exclusive-model slots (same liveness predicate
        // as the step-3 active list), let `admitPendingTick` decide,
        // and leave held slots in `pending` — their conn threads keep
        // flowing SSE keepalives while they wait, and the wait
        // condition above never blocks while `pending` is non-empty,
        // so a held slot admits on the first tick after the active one
        // is culled (step 5, same mutex).
        var live_buf: [32]AdmitCand = undefined;
        var n_live: usize = 0;
        for (sch.decoding.items) |s| {
            if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
            if (!modelExclusiveDecode(s.model)) continue;
            if (n_live >= live_buf.len) break;
            live_buf[n_live] = .{ .model = @intFromPtr(s.model), .exclusive = true };
            n_live += 1;
        }
        var cand_buf: [32]AdmitCand = undefined;
        const n_cands = @min(sch.pending.items.len, cand_buf.len);
        for (sch.pending.items[0..n_cands], 0..) |s, i| {
            cand_buf[i] = .{ .model = @intFromPtr(s.model), .exclusive = modelExclusiveDecode(s.model) };
        }
        var admit_idx: [to_prefill.len]usize = undefined;
        const n_admit = admitPendingTick(cand_buf[0..n_cands], live_buf[0..n_live], &admit_idx);
        for (admit_idx[0..n_admit]) |idx| {
            to_prefill[n_prefill] = sch.pending.items[idx];
            _ = to_prefill[n_prefill].in_pass.fetchAdd(1, .acq_rel);
            n_prefill += 1;
        }
        // Remove admitted entries in DESCENDING index order so the
        // earlier (ascending) indices stay valid during removal.
        var r = n_admit;
        while (r > 0) {
            r -= 1;
            _ = sch.pending.orderedRemove(admit_idx[r]);
        }
        @memcpy(sch.prefilling[0..n_prefill], to_prefill[0..n_prefill]);
        sch.prefilling_from = 0;
        sch.n_prefilling = n_prefill;
    }

    if (sch.budget_revise_sw) |sw| {
        if (sw.read() > BUDGET_REVISE_WINDOW_NS) sch.budget_revise_sw = null;
        if (n_prefill > 0) reviseHotCacheBudgets(sch, true);
    }

    // A batch admitted together prefills in order. A long cold prompt first would hold every short
    // request in the batch until it ends, and the chunk-boundary hook never sees them because
    // they already left `pending`. With a decode share, the smallest uncached tail goes first.
    if (n_prefill > 1 and prefillDecodeShare() > 0) {
        var tails: [to_prefill.len]usize = undefined;
        var order: [to_prefill.len]usize = undefined;
        for (to_prefill[0..n_prefill], 0..) |s, i| tails[i] = slotUncachedTail(s);
        orderByTail(tails[0..n_prefill], order[0..n_prefill]);
        var sorted: [to_prefill.len]*Slot = undefined;
        for (order[0..n_prefill], 0..) |src, i| sorted[i] = to_prefill[src];
        @memcpy(to_prefill[0..n_prefill], sorted[0..n_prefill]);
        // `prefilling_from` counts positions in this order.
        sch.queue_mu.lockUncancelable(sch.io);
        @memcpy(sch.prefilling[0..n_prefill], to_prefill[0..n_prefill]);
        sch.queue_mu.unlock(sch.io);
    }

    // 2. Prefill each pending slot (heavy; mlx ops on this thread).
    //    The inference thread is the sole mlx caller post-cleanup, so
    //    no per-tick stream rebind / mutex coexistence is needed.
    if (n_prefill > 0) {
        defer {
            sch.queue_mu.lockUncancelable(sch.io);
            sch.prefilling_from = 0;
            sch.n_prefilling = 0;
            sch.queue_mu.unlock(sch.io);
        }
        for (to_prefill[0..n_prefill], 0..) |slot, pi| {
            defer _ = slot.in_pass.fetchSub(1, .acq_rel);
            // Unlisted BEFORE its pass ends: `complete()` (which may free the slot) waits on
            // `in_pass`, so a listed slot is always alive for `cancelAllInFlight`.
            defer {
                sch.queue_mu.lockUncancelable(sch.io);
                sch.prefilling_from = @max(sch.prefilling_from, pi + 1);
                sch.queue_mu.unlock(sch.io);
            }
            // Between the slots of one admitted batch, tick the streams
            // that just started decoding — a single-chunk prefill exposes
            // no chunk-boundary yield, so without this every slot's first
            // token waits for the LAST slot's prefill (the TTFT
            // staircase collapse).
            if (pi > 0 and prefillInterleaveEnabled()) _ = interleaveDecodeTick(sch);
            switch (prefillAdmitted(sch, slot)) {
                .held => {
                    // Back to the head of `pending`, in order; the decode tick below runs first.
                    sch.queue_mu.lockUncancelable(sch.io);
                    defer sch.queue_mu.unlock(sch.io);
                    sch.n_prefilling = pi; // back in `pending`, which `cancelAllInFlight` walks
                    var r = n_prefill;
                    while (r > pi) {
                        r -= 1;
                        sch.pending.insert(sch.allocator, 0, to_prefill[r]) catch {
                            to_prefill[r].markError("OutOfMemory");
                        };
                        if (r > pi) _ = to_prefill[r].in_pass.fetchSub(1, .acq_rel);
                    }
                    break;
                },
                .decoding, .finished, .errored, .tail_too_large => ran = true,
            }
        }
    }

    // 3. Build active-list snapshot (skip cancelled / finished / errored).
    var active: std.ArrayList(*Slot) = .empty;
    defer active.deinit(sch.allocator);
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        for (sch.decoding.items) |s| {
            if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
            active.append(sch.allocator, s) catch break;
            _ = s.in_pass.fetchAdd(1, .acq_rel);
        }
    }
    defer for (active.items) |s| {
        _ = s.in_pass.fetchSub(1, .acq_rel);
    };

    specYieldTick(active.items);

    // 4. Decode tick. Charge the full wall-clock tick time to each
    //    participating slot — for batched ticks this matches the per-slot
    //    throughput a user actually observes (their stream advances at
    //    the tick cadence regardless of how many peers share it).
    if (active.items.len > 0) {
        ran = true;
        var decode_sw = io_util.Stopwatch.init(sch.io);
        runDecodeTick(sch, active.items) catch |err| {
            log.err("[scheduler] decode tick failed: {s}\n", .{@errorName(err)});
            for (active.items) |s| s.markError(@errorName(err));
        };
        const tick_ns = decode_sw.read();
        for (active.items) |s| s.decode_ns +|= tick_ns;
    }

    // 5. Cull finished / errored / cancelled from `decoding`. The slot
    //    still belongs to its connection thread until that thread calls
    //    `complete`; we just stop touching it.
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        var i: usize = 0;
        while (i < sch.decoding.items.len) {
            const s = sch.decoding.items[i];
            const drop = s.cancelled.load(.acquire) or s.finished or s.error_code != null;
            const grown = if (s.legacy_gen) |*g| promiseMaterialized(g.completion_tokens) else false;
            if (drop or grown) {
                sch.promise_ledger.release(s.memory_promised);
                s.memory_promised = 0;
            }
            if (drop) {
                _ = sch.decoding.orderedRemove(i);
            } else i += 1;
        }
        publishLiveKvResidency(sch, null);
    }
    return if (ran) .ran else .idle;
}

/// Caller holds `queue_mu`; inference thread only (it owns the slots' arrays).
/// `prefilling` is the slot mid-prefill, which is not in `decoding` yet.
fn publishLiveKvResidency(sch: *Scheduler, prefilling: ?*Slot) void {
    const observe = sch.metrics != null;
    sch.live_session_count = 0;
    var bytes: u64 = 0;
    if (prefilling) |p| {
        const b = slotStateBytes(p);
        bytes += b;
        if (observe) recordLiveSession(sch, p, .prefill, b);
    }
    for (sch.decoding.items) |s| {
        const b = slotStateBytes(s);
        bytes += b;
        if (observe) recordLiveSession(sch, s, .decode, b);
    }
    sch.resident_live_kv_bytes.store(bytes, .monotonic);
}

fn recordLiveSession(sch: *Scheduler, s: *const Slot, phase: metrics_mod.Session.Phase, state_bytes: u64) void {
    if (sch.live_session_count == sch.live_sessions.len) return;
    const prompt: u32 = if (phase == .prefill) @intCast(s.full_prompt.len) else s.prompt_tokens;
    sch.live_sessions[sch.live_session_count] = .init(s.model.id, phase, prompt + s.completion_tokens, s.cached_tokens, s.completion_tokens, state_bytes);
    sch.live_sessions[sch.live_session_count].entry_id = s.restored_entry;
    sch.live_sessions[sch.live_session_count].request_id = s.request_id;
    sch.live_sessions[sch.live_session_count].client = s.client;
    sch.live_session_count += 1;
}

fn slotStateBytes(s: *const Slot) u64 {
    return s.cache.residentBytes();
}

/// Service one embedding request on the inference thread. Runs the batched
/// forward pass via `generate.computeEmbeddingsBatch(xfm, ...)`, which resets
/// the global xfm.cache before every sub-batch, and wakes the conn thread.
fn runEmbedRequest(sch: *Scheduler, req: *EmbedRequest) void {
    const xfm_ptr = req.model.transformer.?;
    const results = generate_mod.computeEmbeddingsBatchWith(req.allocator, xfm_ptr, req.token_seqs) catch |err| {
        finishEmbedRequest(sch, req, @errorName(err));
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.results = results;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishEmbedRequest(sch: *Scheduler, req: *EmbedRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Plan 05 Phase D: service a cold-load work item on the inference thread.
/// Conn thread has already:
///   * Marked `req.entry.state = .loading` (transitioned from .unloaded).
///   * (Optionally) marked `req.evict_entry.state = .evicting` and drained
///     its refcount to 0.
///   * Parsed CPU-only state (config/tok/chat_config) and handed pointers
///     in via the request.
/// We unload the victim (if any), run the load body, install everything on
/// `req.entry`, mark ready, and broadcast `req.done_cond` so the conn
/// thread wakes. On failure, mark `.error_state` so future ensureLoaded
/// calls fail fast; the conn thread surfaces a 500.
fn logWiredPolicy(r: mlx.WiredPolicyResult) void {
    if (r.target) |t| {
        log.info("[wired] mode={s} limit={d} MB\n", .{ @tagName(r.mode), t / (1024 * 1024) });
    } else {
        log.debug("[wired] mode={s} declined (no gpu / empty live set)\n", .{@tagName(r.mode)});
    }
}

/// Free `victims` (each marked `.evicting`, nobody holding it) on this thread.
fn evictResident(sch: *Scheduler, victims: []const *LoadedModel) void {
    for (victims) |victim| {
        const victim_bytes = victim.bytes_resident; // unloadResident zeroes it
        log.info("[registry] evicting model id={s} ({d:.2} GB resident)\n", .{
            victim.id,
            @as(f64, @floatFromInt(victim_bytes)) / 1_073_741_824.0,
        });
        moveHotCacheViewOff(sch, victim);
        victim.unloadResident();
        sch.registry.mutex.lockUncancelable(sch.io);
        sch.registry.accountEvictedLocked(victim_bytes);
        sch.registry.finalizeEvictionLocked(victim);
        sch.registry.mutex.unlock(sch.io);
    }
    // unloadResident freed the victims into MLX's allocator cache — clear it
    // BEFORE the load: its preflight reads OS-level availability, and the
    // parked pool would make it refuse a load that fits.
    _ = mlx.mlx_clear_cache();
}

/// Unloads idle models, oldest first, when that lets a load of `weights` fit in RAM: as few as fit
/// it, none when it already fits or when even all of them cannot (`planRamEvictionsLocked`). Runs
/// past `--skip-mem-preflight` too: an idle model reloads, an overcommitted Mac kills the server.
fn makeRoomInRam(sch: *Scheduler, keep_id: []const u8, weights: u64) void {
    const room = loadRamRoom() orelse return;
    const need = loadRequirementBytes(weights);
    var victims: [16]*LoadedModel = undefined;
    sch.registry.mutex.lockUncancelable(sch.io);
    const n = sch.registry.planRamEvictionsLocked(keep_id, &victims, room, need);
    sch.registry.mutex.unlock(sch.io);
    if (n == 0) return;
    log.info("[registry] {s} needs ~{d:.1} GB beside {d:.1} GB of free RAM: unloading {d} idle model(s)\n", .{
        keep_id,
        @as(f64, @floatFromInt(need)) / 1_073_741_824.0,
        @as(f64, @floatFromInt(room)) / 1_073_741_824.0,
        n,
    });
    evictResident(sch, victims[0..n]);
}

fn runLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    // Step 1: evict victims (if any) BEFORE the load, so peak GPU residency
    // never holds the old + new model at once. unloadResident() drops
    // mlx_arrays — same thread-stream invariant as cleanup_queue drain.
    evictResident(sch, req.evict_entries);
    req.evict_entries = &.{};

    // Step 2: the actual load. On error, mark .error_state and signal done
    // (conn thread frees pre-parsed CPU state — ownership stays on req on
    // the failure path).
    const had_error = mlx.errorPending();
    doLoadOnInferenceThread(sch, req) catch |err| {
        // The load's own MLX error is reported here; latched, it would fail the next request.
        mlx.dropLatchedErrorUnless(had_error);
        return failLoadRequest(sch, req, err);
    };
    // Re-apply so `fit` capacity covers everything this load brought in
    // (drafter / vision land after the mid-load apply).
    logWiredPolicy(mlx.applyWiredPolicy());
    readyLoadRequest(sch, req);
}

fn readyLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    log.info("[registry] model id={s} ready ({d:.2} GB resident)\n", .{
        req.entry.id,
        @as(f64, @floatFromInt(req.entry.bytes_resident)) / 1_073_741_824.0,
    });
    finishLoadRequest(sch, req, null);
}

/// Mark the entry and signal the waiter (conn thread frees pre-parsed CPU state — ownership stays
/// on req on the failure path).
fn failLoadRequest(sch: *Scheduler, req: *LoadRequest, err: anyerror) void {
    log.err("[registry] load failed for model id={s}: {s}\n", .{ req.entry.id, @errorName(err) });
    sch.registry.mutex.lockUncancelable(sch.io);
    if (err == error.InsufficientMemory) {
        // A memory-preflight refusal is transient, not a property of the
        // checkpoint: the 503 tells the user to free memory and retry, so
        // the entry must go back to .unloaded — stuck in .error_state the
        // retry would fail fast until a server restart (#144).
        sch.registry.markUnloadedLocked(req.entry);
    } else {
        // markErrorLocked dupes the error name onto the entry; the
        // conn thread reads it back from the entry, not from req.
        sch.registry.markErrorLocked(req.entry, @errorName(err));
    }
    sch.registry.mutex.unlock(sch.io);
    finishLoadRequest(sch, req, @errorName(err));
}

fn finishLoadRequest(sch: *Scheduler, req: *LoadRequest, err_name: ?[]const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (err_name) |name| {
        if (req.error_name) |old| req.allocator.free(old);
        req.error_name = req.allocator.dupe(u8, name) catch null;
    }
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Free a model's resident mlx state on the inference thread (stream-bound,
/// same invariant as the cleanup-queue drain) and finalize the eviction
/// accounting. The conn thread already marked the entry `.evicting` and
/// drained its refcount in `unloadModelIfIdle`.
fn runUnloadRequest(sch: *Scheduler, req: *UnloadRequest) void {
    const entry = req.entry;
    const bytes = entry.bytes_resident; // unloadResident zeroes it
    log.info("[registry] unloading model id={s} ({d:.2} GB resident)\n", .{
        entry.id,
        @as(f64, @floatFromInt(bytes)) / 1_073_741_824.0,
    });
    moveHotCacheViewOff(sch, entry);
    entry.unloadResident();
    // unloadResident freed the arrays into MLX's allocator cache — clear it
    // so the unload actually returns the memory to the OS (the whole point
    // of the load→generate→unload flow).
    _ = mlx.mlx_clear_cache();
    // Drop any borrowed views that pointed at this entry so post-unload reads
    // don't dangle (gen entries leave xfm null already, but an LLM unload
    // must clear them).
    if (sch.current_model == entry) {
        sch.current_model = null;
        sch.xfm = null;
        sch.weights = null;
        sch.drafter = null;
        sch.dflash = null;
        if (hot_cache_budget_invalidate) |f| f();
    }
    publishHotCacheResidency(sch);
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.accountEvictedLocked(bytes);
    sch.registry.finalizeEvictionLocked(entry);
    sch.registry.mutex.unlock(sch.io);

    // Shrink `fit` capacity back to the surviving live set — leaving the
    // freed model's headroom in place is exactly the per-transient-commit
    // configuration the policy exists to avoid.
    logWiredPolicy(mlx.applyWiredPolicy());

    req.done_mu.lockUncancelable(sch.io);
    req.done = true;
    req.done_cond.broadcast(sch.io);
    req.done_mu.unlock(sch.io);
}

fn dflashContextCoversPrefix(context_len: usize, prefix_len: usize) bool {
    return context_len == prefix_len;
}

/// Only an all-pad generation poisons the prefix; a zero-token one still holds a real prefill.
fn commitDeclinesPadOnly(n_gen: usize, all_pad: bool) bool {
    return n_gen > 0 and all_pad;
}

/// Phase A6: commit a successfully completed slot's KV cache to the hot
/// prefix cache. Called from the inference thread BEFORE `markFinished`
/// broadcasts, so the slot is still alive (the conn thread is blocked in
/// `waitNext`). Skipped for pad-only generations, vision-bearing slots
/// (stale embeddings would be reused), and slots with no generated tokens.
fn commitSlotIfApplicable(sch: *Scheduler, slot: *Slot) void {
    // Phase D: per-model prefix cache — read off the slot's LoadedModel.
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    if (slot.error_code != null) return;
    // `finishSlot` can be reached on an EOS the failing forward itself produced, before the
    // tick wrapper reads the latch: an unread latch means the KV under this commit may be garbage.
    if (mlx.errorPending()) return;
    const gen_ptr = if (slot.legacy_gen) |*g| g else {
        // A Generator-less slot whose cache is non-empty is a prefill the
        // client disconnected from mid-chunk-loop: initWithOptions threw
        // error.Cancelled before `slot.legacy_gen = gen` ever ran, but every
        // chunk that DID forward still lives in slot.cache. Commit that
        // forwarded prefix instead of dropping it. This arm sits ABOVE the
        // pad-only guard: `was_pad_only` starts true and only flips on the
        // first pushed token, so a slot that never pushed one is not
        // pad-POISONED, it is merely empty.
        commitCancelledPrefillSlot(slot, hc);
        publishHotCacheResidency(sch);
        return;
    };
    const n_gen = gen_ptr.generated_ids.items.len;
    if (commitDeclinesPadOnly(n_gen, slot.was_pad_only)) return;

    // Construct the full token sequence: the original prompt + everything
    // generated this turn that the cache holds — Generator forwarded each
    // emitted token into slot.cache as it was sampled, except an
    // `unforwarded_tail` the key must not claim.
    const total_len = slot.full_prompt.len + gen_ptr.generated_ids.items.len - gen_ptr.unforwarded_tail;
    const total_tokens = sch.allocator.alloc(u32, total_len) catch return;
    defer sch.allocator.free(total_tokens);
    @memcpy(total_tokens[0..slot.full_prompt.len], slot.full_prompt);
    @memcpy(total_tokens[slot.full_prompt.len..], gen_ptr.generated_ids.items[0 .. total_len - slot.full_prompt.len]);

    // Plain DFlash rounds park their captures; they join the context before it is paired with the prefix.
    gen_ptr.dflashSettle(slot.allocator, total_len) catch |err| log.warn("[hot-cache] dflash context settle failed: {s} — not committed\n", .{@errorName(err)});
    // A runtime fallback leaves the dormant assistant context at its last
    // speculative boundary while serial decode continues growing the trunk.
    // Only pair the assistant payload with this prefix when both end at the
    // exact same absolute position; otherwise the next turn must rebuild it.
    const dflash_commit: ?prefix_cache_mod.DflashCommit = if (gen_ptr.dflash_ctx) |*dc|
        if (dflashContextCoversPrefix(dc.absLen(), total_len))
            .{ .cache = &dc.cache, .base_pos = dc.base_pos }
        else
            null
    else
        null;
    const finish_st = hc.commitWithMediaState(&slot.cache, total_tokens, slot.has_tools, slot.media, slot.cache_key, dflash_commit) catch |err| {
        log.warn("[hot-cache] commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    publishHotCacheResidency(sch);
    // The decline paths already log their reason inside the cache; a silent
    // `_ =` here is fine — this caller has no commit-shaped log to lie about.
    _ = finish_st;
}

/// Logical committed length for a cancelled-prefill commit: the tokens
/// actually forwarded into the KV when the chunk loop aborted, clamped to
/// the prompt and gated on the floor below which an entry is LRU pollution
/// rather than saved work. Pure so the policy is unit-testable without a slot.
fn cancelledPrefillCommitLen(step: usize, prompt_len: usize) ?usize {
    const len = @min(step, prompt_len);
    if (len < prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS) return null;
    return len;
}

/// Commit the forwarded prefix of a prefill the client disconnected from
/// (error.Cancelled out of `Generator.initWithOptions`; `legacy_gen` was
/// never assigned). The partial KV in `slot.cache` is valid — every chunk
/// that ran was a real forward — so the next request sharing this prefix
/// skips exactly those chunks. The entry key is the forwarded prefix ONLY:
/// `full_prompt` beyond `cache.step` was never forwarded and must not ride
/// the key (a key longer than its KV is the alignment-error class).
///
/// The slot's `cancelled_forwarded` (set by `initWithOptions` on the abort) is the length.
fn commitCancelledPrefillSlot(slot: *Slot, hc: *prefix_cache_mod.HotPrefixCache) void {
    // `cancelled_forwarded` is the authoritative length: `cache.step` only advances when
    // Generator init completes, so it reads 0 on every aborted prefill.
    const len = cancelledPrefillCommitLen(slot.cancelled_forwarded, slot.full_prompt.len) orelse {
        log.debug("[hot-cache] cancelled prefill forwarded < {d} tokens — below the commit floor; not committed\n", .{prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS});
        return;
    };
    const st = hc.commitWithMediaState(&slot.cache, slot.full_prompt[0..len], slot.has_tools, slot.media, slot.cache_key, null) catch |err| {
        log.warn("[hot-cache] cancelled-prefill commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    // Truthful outcome only: a budget decline must never print as a commit
    // (live 2026-09-07: "skipped oversized" + "committed N/M" were the SAME
    // event and the cache looked healthy while a 122k session re-prefilled
    // ~95k tokens per retry).
    switch (st) {
        .ok => |n| log.info("[hot-cache] committed {d}/{d} prompt tokens from a cancelled prefill\n", .{ n, slot.full_prompt.len }),
        .disk_only => |n| log.info("[disk-cache] captured {d}/{d} prompt tokens from a cancelled prefill\n", .{ n, slot.full_prompt.len }),
        .kept_resident => |n| log.info("[hot-cache] kept resident {d}-token entry; oversized candidate declined\n", .{n}),
        .declined => {},
    }
}

/// Phase A6: finalize a slot. Commits to hot prefix cache (if applicable)
/// before signaling completion. The order matters: commit first while the
/// slot is alive, then markFinished — the conn thread's waitNext might
/// return immediately after the broadcast and call complete()→deinit, so
/// we cannot reach into the slot afterwards.
/// Pure decision for the degenerate-tail-loop guard: when the generated tail
/// has collapsed into a short repeating cycle, returns the finish reason to
/// cut the request with; null while generation is healthy.
///
/// The reason is "stop", not "length": this guard fired far below the requested
/// output cap, and clients such as pi treat "length" as context-overflow recovery
/// and compact unnecessarily. The server separately suppresses tool-call parsing
/// for this cause, so a cut fragment cannot be promoted to a completed call.
pub fn loopStopReason(generated_ids: []const u32) ?[]const u8 {
    const d = loopStopDecision(generated_ids) orelse return null;
    return d.finish_reason;
}

/// What a loop cut tells the rest of the server. `finish_reason` is the wire
/// value (always "stop" — see above); `finish_details` is the sibling
/// signal that names the CAUSE, and `trim_start` is where the client's copy
/// of the answer should end.
pub const LoopStop = struct {
    finish_reason: []const u8 = "stop",
    /// The `finish_details.type` value. One string for all three tiers: a
    /// client's decision ("this turn is unusable, don't feed it back") is the
    /// same whichever tier convicted, and the tier is in the log.
    finish_details: []const u8 = "repetition_loop",
    tier: generate_mod.DegenerateTail.Tier,
    trim_start: usize,
};

/// Pure decision + trim point. The three tiers live in generate.zig
/// (`degenerateTail`); this is where their verdict becomes server behaviour.
pub fn loopStopDecision(generated_ids: []const u32) ?LoopStop {
    const d = generate_mod.degenerateTail(generated_ids) orelse return null;
    return .{ .tier = d.tier, .trim_start = d.start };
}

const LoopAction = enum { cut, close_thought };

/// A loop convicted INSIDE an open think block is recoverable once per request:
/// commit the bound's early-stop line + closer and let the model answer. Anything
/// else (answer-side loop, second loop, no room, no bound) keeps the ordinary cut.
fn loopRecoveryAction(tb: ?*generate_mod.ThinkBound, ids: []const u32, completion_tokens: u32, max_tokens: u32) LoopAction {
    const b = tb orelse return .cut;
    b.observe(ids);
    if (b.fired or !b.in_think) return .cut;
    if (!generate_mod.forcedBoundaryCanContinue(completion_tokens, max_tokens, b.forced.len + 1)) return .cut;
    return .close_thought;
}

var loop_trim_env: ?bool = null;
/// `MLX_SERVE_LOOP_TRIM=0` keeps the whole degenerate tail in the response —
/// the A/B arm, and the escape hatch for anyone who needs to see exactly what
/// the model emitted. The cut itself is unaffected either way.
pub fn loopTrimEnabled() bool {
    if (loop_trim_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_LOOP_TRIM");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    loop_trim_env = enabled;
    return enabled;
}

/// The terminator a finishing slot publishes, and the only caller of `Slot.markFinished`.
/// `latched` is a peek (`mlx.peekErrorName`): a decode forward that failed can still hand
/// back a plausible EOS, and the request must not finish 200 on it. The tick wrapper still
/// owns the latch and fails the rest of a batched group with it.
fn publishSlotTerminator(slot: anytype, reason: []const u8, latched: ?[]const u8) void {
    if (latched) |name| {
        slot.markError(name);
        return;
    }
    slot.markFinished(reason);
}

/// Decoded bytes of the answer that ride the `[short-gen]` line.
const SHORT_GEN_TEXT_CAP = 200;

/// `emitted` (what the client got) and `realized` (what the generator appended) diverge only on a block path.
fn formatShortGen(
    out: []u8,
    reason: []const u8,
    emitted: u32,
    ids: []const u32,
    text: []const u8,
    path: []const u8,
) []const u8 {
    var n: usize = 0;
    n += (std.fmt.bufPrint(out[n..], "[short-gen] reason={s} emitted={d} realized={d} ids=[", .{ reason, emitted, ids.len }) catch return out[0..n]).len;
    for (ids, 0..) |id, i| {
        const sep: []const u8 = if (i == 0) "" else ",";
        n += (std.fmt.bufPrint(out[n..], "{s}{d}", .{ sep, id }) catch break).len;
    }
    n += (std.fmt.bufPrint(out[n..], "] bytes=\"", .{}) catch return out[0..n]).len;
    var cap = @min(text.len, SHORT_GEN_TEXT_CAP);
    // A token is a BPE fragment: never cut inside a multi-byte sequence.
    while (cap > 0 and cap < text.len and (text[cap] & 0xC0) == 0x80) cap -= 1;
    for (text[0..cap]) |c| {
        var one = [_]u8{if (c < 0x20) ' ' else c};
        const esc: []const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => one[0..1],
        };
        if (n + esc.len > out.len) break;
        @memcpy(out[n..][0..esc.len], esc);
        n += esc.len;
    }
    n += (std.fmt.bufPrint(out[n..], "\" path={s}", .{path}) catch return out[0..n]).len;
    return out[0..n];
}

/// A one-word answer and a block-path handling bug look identical until the ids are named.
fn logShortGen(slot: *Slot, reason: []const u8) void {
    const gen = if (slot.legacy_gen) |*g| g else return;
    const mode = specTickMode(
        slot.enable_drafter,
        gen.drafter != null,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
    );
    const ids = gen.generated_ids.items;
    const decoded: ?[]u8 = if (slot.model.tokenizer) |tok|
        (tok.decode(slot.allocator, ids, false) catch null)
    else
        null;
    defer if (decoded) |d| slot.allocator.free(d);
    var buf: [1024]u8 = undefined;
    log.info("{s}\n", .{formatShortGen(
        &buf,
        reason,
        slot.completion_tokens,
        ids,
        decoded orelse "",
        if (mode == .regular) "serial" else @tagName(mode),
    )});
}

/// Outcome of a slot that reached `finishSlot`. A latched MLX failure turns the finish into an error.
fn finishOutcome(reason: []const u8, latched: ?[]const u8) metrics_mod.Outcome {
    if (latched != null) return .failed;
    if (std.mem.eql(u8, reason, "cancelled")) return .cancelled;
    return .success;
}

/// Outcome of a slot seen by the cleanup drain without `finishSlot` having recorded it, read only
/// from state the inference thread set. `cancelled` is never consulted: `Scheduler.complete` sets it
/// on every completion, normal ones included.
fn cleanupOutcome(slot: anytype) metrics_mod.Outcome {
    if (slot.error_code != null) return .failed;
    if (slot.finished) return finishOutcome(slot.finish_reason, null);
    return .cancelled;
}

/// Count a slot's outcome once. The first path to reach a slot wins.
fn recordSlotEnd(metrics: ?*metrics_mod.Metrics, slot: anytype, outcome: metrics_mod.Outcome) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    slot.metrics_recorded = true;
    m.recordRequest(
        outcome,
        slot.first_token_ns,
        slot.prefill_ns,
        slot.decode_ns,
        slot.prompt_tokens,
        slot.completion_tokens,
        slot.cached_tokens,
    );
}

/// The cleanup drain's count: a slot refused before its first forward is a rejection, anything
/// `finishSlot` already counted is skipped.
fn recordSlotCleanup(metrics: ?*metrics_mod.Metrics, slot: anytype) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    if (slot.error_code) |name| if (std.mem.eql(u8, name, "PrefillDoesNotFit")) {
        slot.metrics_recorded = true;
        m.recordRejected();
        return;
    };
    recordSlotEnd(m, slot, cleanupOutcome(slot));
}

fn finishSlot(sch: *Scheduler, slot: *Slot, reason: []const u8) void {
    // Emit the `[spec-stats]` summary (no-op for non-speculative slots).
    // The legacy generate() path logs this itself; scheduler-driven slots
    // finalize here instead.
    if (slot.legacy_gen) |*g| {
        g.logSpecStats();
        g.persistRoundCost();
    }
    const latched: ?[]const u8 = mlx.peekErrorName();
    if (latched) |name| {
        log.err("[scheduler] finish suppressed: the last forward failed ({s}) but produced a \"{s}\" — failing this request rather than answering 200 with what Metal never wrote\n", .{ name, reason });
    }
    if (slot.completion_tokens <= 2) logShortGen(slot, reason);
    commitSlotIfApplicable(sch, slot);
    // Restore by move: a slot that ended without committing still holds its checkout, and
    // the record now describes bytes that die with `slot.cache`. Above every early return.
    if (slot.model.prefix_cache) |*hc| hc.releaseCheckout(@intFromPtr(slot), reason);
    // SSD flush runs AFTER markFinished so the client never waits on the
    // chunk-append — but everything it needs must be captured BEFORE the
    // broadcast: the conn thread may complete()+free the slot immediately.
    // The prefix cache and stream live on the registry-owned LoadedModel,
    // which outlives the slot (unload also runs on this thread).
    const hc_opt: ?*prefix_cache_mod.HotPrefixCache =
        if (slot.model.prefix_cache) |*p| p else null;
    const stream_opt: ?mlx.mlx_stream =
        if (slot.model.transformer) |x| x.s else null;
    // Record per-request metrics while slot fields are still live (before
    // markFinished broadcasts — the conn thread may complete()+free the slot
    // immediately after). Off the per-token path; a null sink (metrics off) is
    // a single per-request branch. real_ttft = first_token_ns (queue+prefill,
    // captured exactly at prefill completion); recordRequest derives
    // e2e = first_token_ns + decode_ns.
    recordSlotEnd(sch.metrics, slot, finishOutcome(reason, latched));
    publishSlotTerminator(slot, reason, latched);
    if (hc_opt) |hc| {
        if (stream_opt) |s| {
            hc.flushPendingDisk(s);
            // After the flush, so this turn's entry is the MRU one and any entry spilled
            // already has a complete copy to spill into.
            hc.spillIdleEntries(s);
            publishHotCacheResidency(sch);
        }
    }
    // Return this turn's transients to the OS. The per-`CACHE_CLEAR_INTERVAL`
    // clear inside `Generator.advanceStep` can't cover the tail of a turn, and
    // MLX parks freed buffers in a size-keyed pool rather than releasing them —
    // so without this a short turn hands everything it stranded to the next one
    // and the process footprint ratchets across a session (issue #110).
    _ = mlx.mlx_clear_cache();
}

/// Scale the measured M5/block-16 break-even to the effective number of
/// draft positions. `block_size` includes the always-emitted anchor. Thinking
/// is the actual resolved request mode; tools are neither necessary nor
/// sufficient for a reasoning preamble.
fn dflashGateMinimum(block_size: u32, enable_thinking: bool, moe_target: bool) f32 {
    const drafts = block_size -| 1;
    if (drafts == 0) return 0;
    const calibrated_min = if (enable_thinking)
        generate_mod.Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND
    else
        generate_mod.Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND;
    const scaled = calibrated_min * @as(f32, @floatFromInt(drafts)) / 15.0;
    // The bar is really "round cost / serial step cost", and the whole
    // calibration above was measured on DENSE trunks where one verify forward
    // costs about one serial step. A sparse trunk breaks that: an A1B MoE
    // decodes ~1B of weights per token but its verify reads every expert the
    // block's positions route to, so a round costs far more than a step while
    // the bar stayed at 0.53 and nothing ever disabled.
    //
    // Measured with a DSpark sidecar, M4 Max, block 5, greedy:
    // novel prose accepts 1.40/round and runs 171 tok/s against 199 serial
    // (round cost = 1.40 x 199/171 = 1.63 steps), while an echo prompt accepts
    // 4.00 and runs 273. A floor of 1.8 disables the losing class after its
    // three warmup rounds and leaves the winning one untouched.
    if (!moe_target) return scaled;
    return @max(scaled, generate_mod.Generator.DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND);
}

/// Allocate the slot's KVCache state (already done in Slot.init), construct
/// the per-slot Generator via `Generator.initWithOptions(.{ .ctx = slot.ctx,
/// .skip_lazy_preforward = true_for_regular, ... })`, and store it on the
/// slot. After return, the slot is ready for decode ticks.
/// Kill switch for prefill-side interleaving (MLX_SERVE_PREFILL_INTERLEAVE=0
/// restores whole-prefill-then-decode scheduling). Default ON: the hook
/// no-ops when nothing is decoding, so an idle or single-stream server never
/// pays for it.
var prefill_interleave_cached: ?bool = null;
pub fn prefillInterleaveEnabled() bool {
    if (prefill_interleave_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_PREFILL_INTERLEAVE");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    prefill_interleave_cached = on;
    return on;
}

/// Target fraction of wall time the decoding streams get while another slot prefills,
/// set once by `main` from `--prefill-decode-share` or MLX_SERVE_PREFILL_DECODE_SHARE
/// (`resolveDecodeShare`). 0 = one boundary's `interleaveTicksFor` ticks and the company
/// width, as before. Reads 0 under the MLX_SERVE_PREFILL_INTERLEAVE=0 kill switch: no
/// hook, nothing to share.
pub var prefill_decode_share: f32 = 0;
pub fn prefillDecodeShare() f32 {
    if (!prefillInterleaveEnabled()) return 0;
    return prefill_decode_share;
}

const InterleaveCtx = struct {
    sch: *Scheduler,
    /// The slot being prefilled: not in `decoding` yet, its KV still counts.
    slot: *Slot,
    decode_ns: u64 = 0,
    ticks: u32 = 0,
    /// Wall clock since the previous boundary's ticks ended: the chunk just forwarded.
    chunk_sw: io_util.Stopwatch,
};

/// Ticks a speculating slot must see company before it gives up speculation for the batch.
const COMPANY_YIELD_TICKS: u8 = 2;

pub fn companyStreak(streak: u8, has_company: bool) u8 {
    return if (has_company) streak +| 1 else 0;
}

/// Spec modes that give way to the batched group once a slot has company: DFlash on a
/// sparse target (its wide verify reads every expert the block routes to). A dense
/// DFlash slot yields only to a crowd (`crowd` others): below it the yielded slot ticks plain
/// beside the verify group instead of inside it, which costs everyone a forward.
pub fn yieldsToCompany(mode: SpecTickMode, sparse_target: bool, others: usize, crowd: usize) bool {
    return mode == .dflash and (sparse_target or others >= crowd);
}

/// The first request of a burst was admitted alone, so it armed speculation and ticks
/// serial beside the group; once company is steady it yields.
fn specYieldTick(active: []const *Slot) void {
    for (active) |s| {
        const gen = if (s.legacy_gen) |*g| g else continue;
        if (gen.spec_disabled_runtime) continue;
        const mode = slotSpecMode(s, gen);
        var others: usize = 0;
        for (active) |o| {
            if (o != s and o.model == s.model) others += 1;
        }
        if (!yieldsToCompany(mode, s.model.config.?.isMoe(), others, 2)) {
            s.company_ticks = 0;
            continue;
        }
        s.company_ticks = companyStreak(s.company_ticks, others > 0);
        if (s.company_ticks >= COMPANY_YIELD_TICKS) gen.yieldSpecToCompany(@tagName(mode));
    }
}

/// Prefill width while other streams decode: a chunk boundary is their only yield point.
const COMPANY_PREFILL_CHUNK: u32 = 2048;

pub fn companyPrefillChunk(chunk: u32, decoding: usize) u32 {
    if (decoding == 0) return chunk;
    return if (chunk == 0) COMPANY_PREFILL_CHUNK else @min(chunk, COMPANY_PREFILL_CHUNK);
}

/// Largest accepted decode share: at 0.9 the decoders own nine times each chunk's time.
pub const PREFILL_DECODE_SHARE_MAX: f32 = 0.9;
/// Prefill width while a decode share is live and someone decodes.
pub const DECODE_SHARE_PREFILL_CHUNK: u32 = 1024;

/// `--prefill-decode-share` / MLX_SERVE_PREFILL_DECODE_SHARE text to a share in [0, 0.9].
/// Above 0.9 clamps; anything that is not a number >= 0 is an error.
pub fn parseDecodeShare(text: []const u8) error{InvalidDecodeShare}!f32 {
    const v = std.fmt.parseFloat(f32, text) catch return error.InvalidDecodeShare;
    if (std.math.isNan(v) or v < 0) return error.InvalidDecodeShare;
    return @min(v, PREFILL_DECODE_SHARE_MAX);
}

/// The flag's text if given, else the env's, else 0.
/// The decode share a server runs with when nothing names one: the boundary admission and the
/// smallest-tail-first batch only engage under a share, and without it a quick question waited
/// 90-120 s behind a 90k prefill (26.10.3 validation). `--prefill-decode-share 0` restores the old schedule.
pub const DEFAULT_DECODE_SHARE: f32 = 0.25;

pub fn resolveDecodeShare(flag: ?[]const u8, env: ?[]const u8) error{InvalidDecodeShare}!f32 {
    const text = flag orelse env orelse return DEFAULT_DECODE_SHARE;
    return parseDecodeShare(text);
}

/// An echo prefill ranks every row against the whole vocab: the top-k sort peaks at five
/// [chunk, vocab] 4-byte arrays (1.25 GiB at 256 rows of Gemma's 262k vocab), beside the
/// chunk's own bf16 logits (128 MiB).
const PROMPT_LOGPROBS_CHUNK: u32 = 256;

fn promptLogprobsWidthCap(echo: bool, cap: u32) u32 {
    if (!echo) return cap;
    return if (cap == 0) PROMPT_LOGPROBS_CHUNK else @min(cap, PROMPT_LOGPROBS_CHUNK);
}

/// The width a prefill started beside live decoders may run at, 0 = no cap. Decided when
/// the prefill starts, after admission, so admission bills the uncapped width. A decoder
/// that finishes mid-prefill lifts the cap only where the adaptive width hook runs.
pub fn decodeShareAdmissionCap(decoding: usize, share: f32) u32 {
    if (decoding == 0 or share <= 0) return 0;
    return DECODE_SHARE_PREFILL_CHUNK;
}

/// Decode wall time owed after a prefill chunk of `chunk_ns`: chunk * S / (1 - S).
pub fn decodeShareBudgetNs(chunk_ns: u64, share: f32) u64 {
    if (share <= 0) return 0;
    const s: f64 = @min(share, PREFILL_DECODE_SHARE_MAX);
    return @intFromFloat(@as(f64, @floatFromInt(chunk_ns)) * s / (1 - s));
}

/// Is another decode tick owed at this boundary? Never fewer than `interleaveTicksFor`.
pub fn interleaveTickOwed(share: f32, chunk_ns: u64, first_ns: u64, ticks: u32, decode_ns: u64) bool {
    if (first_ns == 0) return false;
    if (ticks < interleaveTicksFor(chunk_ns, first_ns)) return true;
    return decode_ns < decodeShareBudgetNs(chunk_ns, share);
}

/// The adaptive width hook's pick, capped while a decode share is live and someone decodes.
pub fn decodeShareWidthCap(width: u32, decoding: usize, share: f32) u32 {
    const cap = decodeShareAdmissionCap(decoding, share);
    return if (cap == 0) width else @min(width, cap);
}

/// The largest uncached tail a request may have to be prefilled between another prefill's chunks:
/// one decode-share chunk, so the nested prefill costs the outer about one chunk of time.
pub const BOUNDARY_MAX_TAIL: usize = DECODE_SHARE_PREFILL_CHUNK;
/// Requests admitted at one chunk boundary, across its admission scan and the gaps between its owed
/// decode ticks. The wall-time cap in `BoundaryBudget` is the real bound.
pub const BOUNDARY_MAX_COUNT: u32 = 4;

/// What the boundary gate reads about a pending request.
pub const BoundaryCand = struct {
    exclusive: bool,
    has_media: bool,
    prompt_logprobs: bool,
    generator_path: bool,
    /// Module-side per-request state that a declined restore would reset under the outer.
    module_state: bool,
    /// Uncached tail, from `peekMatchedTokens`.
    tail: usize,
};

pub const BoundaryCtx = struct {
    share: f32,
    /// A prefill started at a boundary may tick decoders at its own boundaries but never admits.
    nested: bool,
    outer_exclusive: bool,
    decoding: usize,
    max_concurrent: usize,
    max_tail: usize,
};

/// May this pending request be prefilled at the outer prefill's chunk boundary?
pub fn boundaryEligible(c: BoundaryCand, x: BoundaryCtx) bool {
    if (x.share <= 0 or x.nested or x.outer_exclusive) return false;
    if (c.exclusive or c.has_media or c.prompt_logprobs or !c.generator_path or c.module_state) return false;
    if (x.decoding >= x.max_concurrent) return false;
    return c.tail <= x.max_tail;
}

/// What one boundary may spend on nested prefills, so the outer always advances.
pub const BoundaryBudget = struct {
    max_count: u32,
    ns_cap: u64,
    count: u32 = 0,
    spent_ns: u64 = 0,

    pub fn mayStart(self: BoundaryBudget) bool {
        return self.count < self.max_count and self.spent_ns < self.ns_cap;
    }

    pub fn charge(self: *BoundaryBudget, ns: u64) void {
        self.count += 1;
        self.spent_ns +|= ns;
    }
};

/// Stable ascending order of `tails` written to `order` as indices: smallest uncached tail first.
pub fn orderByTail(tails: []const usize, order: []usize) void {
    for (order[0..tails.len], 0..) |*o, i| o.* = i;
    const by = struct {
        fn lessThan(t: []const usize, a: usize, b: usize) bool {
            return t[a] < t[b];
        }
    };
    std.sort.insertion(usize, order[0..tails.len], tails, by.lessThan);
}

/// Undo a hot-cache restore the slot has not started on: hand the checkout back and empty the
/// slot's cache, so the request can wait in `pending` and restore again when its turn comes.
fn resetRestoredSlot(slot: *Slot) void {
    if (slot.model.prefix_cache) |*hc| hc.releaseCheckout(@intFromPtr(slot), "tail over the boundary limit");
    emptySlotCache(slot);
    slot.restored_entry = 0;
}

/// Back to a cold slot: no restored KV.
fn emptySlotCache(slot: *Slot) void {
    if (slot.model.transformer) |xf| {
        slot.cache.truncate(0, xf.s) catch {};
    }
    slot.moe_seq_offset = 0;
    slot.cached_tokens = 0;
}

/// The first pending request that passes the boundary gate, removed from `pending` with its
/// `in_pass` held. `outer` is the prefill whose chunk boundary this is.
fn takeBoundaryCandidate(sch: *Scheduler, outer: *Slot) ?*Slot {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    var decoding: usize = 0;
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        decoding += 1;
    }
    const x: BoundaryCtx = .{
        .share = prefillDecodeShare(),
        .nested = sch.boundary_depth > 0,
        .outer_exclusive = modelExclusiveDecode(outer.model),
        .decoding = decoding,
        .max_concurrent = sch.max_concurrent,
        .max_tail = BOUNDARY_MAX_TAIL,
    };
    const n = @min(sch.pending.items.len, 32);
    for (sch.pending.items[0..n], 0..) |s, i| {
        if (s.cancelled.load(.acquire)) continue;
        const c: BoundaryCand = .{
            .exclusive = modelExclusiveDecode(s.model),
            .has_media = s.vision_embeddings != null or s.media.len > 0,
            .prompt_logprobs = s.prompt_logprobs,
            .generator_path = s.model.transformer != null,
            .module_state = false,
            .tail = slotUncachedTail(s),
        };
        if (!boundaryEligible(c, x)) continue;
        _ = s.in_pass.fetchAdd(1, .acq_rel);
        _ = sch.pending.orderedRemove(i);
        sch.boundary_slot = s;
        return s;
    }
    return null;
}

/// At a chunk boundary of `ic.slot`'s prefill: admit and prefill small pending requests, so their
/// first token does not wait for the whole prefill. The decode ticks that follow include them.
/// `budget` is the boundary's, shared by its admission scan and the gaps between its owed ticks.
/// Returns the wall time spent.
fn admitSmallAtBoundary(ic: *InterleaveCtx, share: f32, budget: *BoundaryBudget) u64 {
    const sch = ic.sch;
    if (share <= 0 or sch.boundary_depth > 0) return 0;
    var total = io_util.Stopwatch.init(sch.io);
    while (budget.mayStart()) {
        const slot = takeBoundaryCandidate(sch, ic.slot) orelse break;
        var sw = io_util.Stopwatch.init(sch.io);
        slot.boundary_max_tail = BOUNDARY_MAX_TAIL;
        sch.boundary_depth += 1;
        sch.boundary_outer = ic.slot;
        const saved_tokens = sch.inflight_prefill_tokens.load(.monotonic);
        const saved_expected = sch.inflight_prefill_expected.load(.monotonic);
        const outcome = prefillAdmitted(sch, slot);
        // runPrefill zeroes the progress gauges on exit; the outer prefill's go back.
        sch.inflight_prefill_tokens.store(saved_tokens, .monotonic);
        sch.inflight_prefill_expected.store(saved_expected, .monotonic);
        sch.boundary_outer = null;
        sch.boundary_depth -= 1;
        slot.boundary_max_tail = null;
        const wait = outcome == .held or outcome == .tail_too_large;
        // Back to the head of `pending` before `in_pass` drops, so `complete` never sees a slot that
        // is in no list.
        sch.queue_mu.lockUncancelable(sch.io);
        sch.boundary_slot = null;
        if (wait) sch.pending.insert(sch.allocator, 0, slot) catch slot.markError("OutOfMemory");
        sch.queue_mu.unlock(sch.io);
        _ = slot.in_pass.fetchSub(1, .acq_rel);
        if (wait) break;
        if (outcome == .decoding) log.info("[boundary] admitted a {d}-token request beside a {d}-token prefill\n", .{ slot.prompt_tokens, ic.slot.full_prompt.len });
        budget.charge(sw.read());
    }
    return total.read();
}

/// The tokens `slot` would still have to prefill, from a read-only peek at the hot cache. A slot
/// that skips the cache, or carries media (media rows are never matched by token), counts whole.
fn slotUncachedTail(slot: *Slot) usize {
    const total = slot.full_prompt.len;
    if (slot.skip_prefix_cache or slot.media.len > 0) return total;
    const hc = if (slot.model.prefix_cache) |*p| p else return total;
    return total - @min(total, hc.peekMatchedTokens(slot.full_prompt, slot.has_tools, slot.cache.config));
}

pub const OwedTicks = struct { ticks: u32, spent_ns: u64 };

/// The ticks after a boundary's first one: run until none is owed or the decoders run out
/// (`tick` returns 0).
pub fn runOwedDecodeTicks(share: f32, chunk_ns: u64, first_ns: u64, ctx: *anyopaque, tick: *const fn (*anyopaque) u64) OwedTicks {
    var r: OwedTicks = .{ .ticks = 1, .spent_ns = first_ns };
    while (interleaveTickOwed(share, chunk_ns, first_ns, r.ticks, r.spent_ns)) {
        const ns = tick(ctx);
        if (ns == 0) break;
        r.ticks += 1;
        r.spent_ns +|= ns;
    }
    return r;
}

fn interleaveDecodeTickOpaque(ctx: *anyopaque) u64 {
    return interleaveDecodeTick(@ptrCast(@alignCast(ctx)));
}

const INTERLEAVE_MAX_TICKS: u32 = 8;

/// Decode ticks owed at a chunk boundary: enough to keep the decoding streams at a quarter
/// of wall time (chunk / 3), at least one, capped so a slow chunk cannot stall its own TTFT.
pub fn interleaveTicksFor(chunk_ns: u64, tick_ns: u64) u32 {
    if (tick_ns == 0) return 1;
    return @intCast(std.math.clamp(chunk_ns / 3 / tick_ns, 1, INTERLEAVE_MAX_TICKS));
}

/// SSD-first write-through: persist each completed prefill chunk as the prefill produces it,
/// so a cancelled or killed prefill leaves a restorable chunk-aligned prefix. Armed only with
/// a live background writer, so the inference thread never pays the file write.
fn prefillWriteThroughCb(opaque_ctx: *anyopaque, abs_kv_pos: usize) void {
    const wc: *WriteThroughCtx = @ptrCast(@alignCast(opaque_ctx));
    const slot = wc.slot;
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    if (!hc.ssd_first) return;
    const d = if (hc.disk) |*dd| dd else return;
    if (d.writer == null) return;
    if (abs_kv_pos == 0 or abs_kv_pos > slot.full_prompt.len) return;
    const s = if (slot.model.transformer) |x| x.s else return;
    wc.chunks += 1;
    // RAM-backed mode banks one chunk as crash salvage. SSD-only mode must keep pace with
    // prefill or the first completed turn can leave most of its prefix unavailable after restart.
    _ = d.appendCommitBounded(
        slot.cache.entries,
        abs_kv_pos,
        slot.cache.config,
        slot.full_prompt[0..abs_kv_pos],
        slot.has_tools,
        // The items whose rows this chunk boundary covers; their keys travel with the rows.
        prefix_cache_mod.spansBelow(slot.media, abs_kv_pos),
        s,
        if (hc.ram_enabled) WRITE_THROUGH_FLUSH_BOUND_BYTES else std.math.maxInt(u64),
    ) catch |err| {
        log.warn("  [disk-cache] prefill write-through failed: {s}\n", .{@errorName(err)});
    };
}

/// The write-through hook's per-call flush bound: one byte, so the chunk loop stops after the first chunk it writes.
pub const WRITE_THROUGH_FLUSH_BOUND_BYTES: u64 = 1;

const WriteThroughCtx = struct {
    slot: *Slot,
    chunks: u32 = 0,
};

/// The write-through only pays when this turn's prefill produces at least one whole disk
/// chunk: below that it persisted a warm turn's whole restored prefix inside TTFT
/// (+183/+369/+737 ms at 16k/32k/64k) for a prefix the end-of-request commit persists anyway.
fn writeThroughSpanReached(new_span: usize, chunk_tokens: u32) bool {
    return new_span >= chunk_tokens;
}

var write_through_span_declined_logged = std.atomic.Value(bool).init(false);
var write_through_off_logged = std.atomic.Value(bool).init(false);
var write_through_env_cached: ?bool = null;

/// `MLX_SERVE_SSD_WRITE_THROUGH=0` takes the write-through out of the prefill loop; the
/// end-of-request commit then persists the whole turn. A real two-arm tradeoff (crash-safe
/// prefix vs TTFT) and the only way to A/B its cost.
pub fn writeThroughEnabledFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}

/// Read on the inference thread only.
fn writeThroughEnabled() bool {
    if (write_through_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_SSD_WRITE_THROUGH") orelse break :blk writeThroughEnabledFromEnv(null);
        break :blk writeThroughEnabledFromEnv(std.mem.sliceTo(raw, 0));
    };
    write_through_env_cached = v;
    return v;
}

/// The one gate for the write-through: SSD-first + a live writer + a disk tier + a new span
/// worth at least one chunk.
fn writeThroughArmed(slot: *Slot, new_span: usize) bool {
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return false;
    if (!writeThroughEnabled()) {
        if (hc.ssd_first and !write_through_off_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through disabled by MLX_SERVE_SSD_WRITE_THROUGH=0 — the end-of-request commit persists every turn\n", .{});
        }
        return false;
    }
    if (!hc.ssd_first) return false;
    const d = if (hc.disk) |*dd| dd else return false;
    const writer_up = d.writer != null;
    if (!writer_up) return false;
    if (!writeThroughSpanReached(new_span, d.chunk_tokens)) {
        if (!write_through_span_declined_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through declined: {d} new tokens is under one chunk ({d}) — the end-of-request commit persists this turn\n", .{ new_span, d.chunk_tokens });
        }
        return false;
    }
    return true;
}

/// Context for `Generator.InitOptions.chunk_width_hook`. `cfg` is optional (embedded engines).
const ChunkWidthCtx = struct {
    sch: *Scheduler,
    cfg: ?*const model_mod.ModelConfig,
    kv_bits: u64,
    /// This slot's own per-model cache, never `sch.hot_prefix_cache`; only `stagedHostBytes`
    /// is read, on the inference thread.
    hc: ?*prefix_cache_mod.HotPrefixCache,
};

/// Host bytes the SSD writer is holding for this slot right now.
fn chunkWidthStagedBytes(wc: *ChunkWidthCtx) u64 {
    const hc = wc.hc orelse return 0;
    return hc.stagedHostBytes();
}

fn chunkWidenConfirmCb(opaque_ctx: *anyopaque, pos: usize, want: u32) bool {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return false;
    const ok = prefill_chunk_widen_ok orelse return false;
    return ok(cfg, wc.kv_bits, pos, want, chunkWidthStagedBytes(wc));
}

fn chunkWidthCb(
    opaque_ctx: *anyopaque,
    pos: usize,
    cur: u32,
    cap: u32,
    st: *generate_mod.AdaptiveWidthState,
) u32 {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return cur;
    const pick = prefill_chunk_adapt orelse return cur;
    const next = pick(cfg, wc.kv_bits, pos, cur, cap, st, chunkWidthStagedBytes(wc));
    // Only a live adaptive width may move; elsewhere the admitted width stands.
    if (!adaptiveChunkWidthFor(cfg)) return next;
    return decodeShareWidthCap(next, wc.sch.liveDecodingCount(), prefillDecodeShare());
}

/// What the owed-tick loop runs between boundaries: admit what arrived, then tick. Returns the
/// admission time plus the tick's (0 only when nothing ran, which ends the loop).
const OwedTickCtx = struct { ic: *InterleaveCtx, share: f32, budget: *BoundaryBudget };

fn admitThenTick(opaque_ctx: *anyopaque) u64 {
    const t: *OwedTickCtx = @ptrCast(@alignCast(opaque_ctx));
    const nested = admitSmallAtBoundary(t.ic, t.share, t.budget);
    return interleaveDecodeTick(t.ic.sch) +| nested;
}

fn interleaveDecodeTickCb(opaque_ctx: *anyopaque) void {
    const ic: *InterleaveCtx = @ptrCast(@alignCast(opaque_ctx));
    if (ic.ticks == 0) {
        log.debug("[interleave] engaged: decode ticks between prefill chunks\n", .{});
    }
    const chunk_ns = ic.chunk_sw.read();
    const share = prefillDecodeShare();
    // Small pending requests first, so the ticks below include them; their time is hosted time. A
    // request that arrives during the owed ticks is admitted between them, so no newcomer waits
    // longer than the chunk in flight.
    var budget: BoundaryBudget = .{ .max_count = BOUNDARY_MAX_COUNT, .ns_cap = decodeShareBudgetNs(chunk_ns, share) };
    const nested_ns = admitSmallAtBoundary(ic, share, &budget);
    const first_ns = interleaveDecodeTick(ic.sch);
    var owed: OwedTickCtx = .{ .ic = ic, .share = share, .budget = &budget };
    const r = runOwedDecodeTicks(share, chunk_ns +| nested_ns, first_ns, &owed, admitThenTick);
    ic.ticks += r.ticks;
    ic.decode_ns +|= r.spent_ns +| nested_ns;
    {
        ic.sch.queue_mu.lockUncancelable(ic.sch.io);
        defer ic.sch.queue_mu.unlock(ic.sch.io);
        publishLiveKvResidency(ic.sch, ic.slot);
    }
    ic.chunk_sw.reset();
}

/// The interval since these slots' previous tick held other work (a prefill chunk, a media
/// step); the round-cost table must not fold it as a token's wall time.
/// Drop it; the next tick seeds afresh.
fn invalidateDecodeClocks(slots: []const *Slot) void {
    for (slots) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        if (s.legacy_gen) |*g| {
            g.invalidateRoundClock();
        }
    }
}

/// One decode tick for the streams currently decoding, run from INSIDE a
/// prefill (between chunks, and between the slots of one admitted batch).
/// Returns the tick's wall-clock ns (0 when no stream is active). The
/// prefilling slot is not in `decoding` yet, so the tick only advances OTHER
/// requests' Generators — same-thread MLX, no reentrancy into this prefill.
fn interleaveDecodeTick(sch: *Scheduler) u64 {
    var buf: [32]*Slot = undefined;
    var n: usize = 0;
    sch.queue_mu.lockUncancelable(sch.io);
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        if (n >= buf.len) break;
        buf[n] = s;
        _ = s.in_pass.fetchAdd(1, .acq_rel);
        n += 1;
    }
    sch.queue_mu.unlock(sch.io);
    defer for (buf[0..n]) |s| {
        _ = s.in_pass.fetchSub(1, .acq_rel);
    };
    if (n == 0) return 0;
    invalidateDecodeClocks(buf[0..n]);
    var sw = io_util.Stopwatch.init(sch.io);
    runDecodeTick(sch, buf[0..n]) catch |err| {
        log.err("[interleave] decode tick failed: {s}\n", .{@errorName(err)});
        for (buf[0..n]) |s| s.markError(@errorName(err));
    };
    const tick_ns = sw.read();
    for (buf[0..n]) |s| s.decode_ns +|= tick_ns;
    return tick_ns;
}

fn runPrefill(sch: *Scheduler, slot: *Slot) !void {
    // Mark the phase for the whole of prefill, and clear both signals on EVERY
    // exit path (success, cancel, error). `requests_prefilling` flips at entry
    // so the panel isn't blind until the first chunk lands; the chunk loop in
    // generate.zig stores absolute progress into `inflight_prefill_tokens`.
    //
    // Gated on `--metrics`, per the observability contract: when it's off the
    // prefill path executes NO extra instruction at all (the chunk loop's hook
    // is null too — see the `prefill_progress` option below).
    const observe = sch.metrics != null;
    if (observe) {
        _ = sch.requests_prefilling.fetchAdd(1, .monotonic);
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        publishLiveKvResidency(sch, slot);
    }
    defer if (observe) {
        _ = sch.requests_prefilling.fetchSub(1, .monotonic);
        sch.inflight_prefill_tokens.store(0, .monotonic);
        sch.inflight_prefill_expected.store(0, .monotonic);
    };

    const sampling = slot.sampling;
    // Refresh ctx in case slot was relocated (paranoia — slot is heap so no,
    // but cheap).
    slot.ctx.cache = &slot.cache;
    slot.ctx.moe_seq_offset = &slot.moe_seq_offset;
    slot.ctx.vision_embeddings = slot.vision_embeddings;
    slot.ctx.capture_hidden = null;
    slot.ctx.kv_attn_fused = slot.kv_attn_fused;

    // Verify passes through forwardWith on a module-owned arch would APPEND
    // draft tokens to module-owned state and corrupt every later step, so
    // the shell spec handles stay off there (see `specInitWiring`).
    // Module CLASS (owned or shared-readonly), not ownership alone: a module-owned trunk
    // batches its plain slots but its spec wiring stays the module one.
    const owns_module_state = slot.model.transformer != null and
        slot.model.transformer.?.moduleSpecWiring();
    const wiring = specInitWiring(
        owns_module_state,
        slot.enable_drafter,
        slot.drafter != null,
        slot.dflash != null,
        slot.enable_pld,
    );
    const use_drafter = wiring.use_drafter;
    const use_dflash = wiring.use_dflash;
    const use_pld = wiring.use_pld;
    log.debug("[spec-wiring] dflash={} drafter={} pld={} (slot: drafter_flag={} dflash_handle={} drafter_handle={})\n", .{
        use_dflash,          use_drafter,          use_pld,
        slot.enable_drafter, slot.dflash != null, slot.drafter != null,
    });

    // Phase A6: prefill source-of-truth is `slot.full_prompt` — the conn
    // thread's `reuseKVCache` may have trimmed `slot.prompt_ids` based on
    // `xfm.cache` (the legacy global cache), but the slot has its own cache
    // which started empty. Using `slot.prompt_ids` would cause the model to
    // attend to only the trailing portion with empty cache, producing
    // garbage. Always start from the full prompt and let the hot prefix
    // cache (if configured) trim it back via the slot's own cache state.
    //
    // Vision-bearing slots: skip the hot cache altogether. Image tokens
    // have identical IDs but the underlying vision embeddings differ
    // per-request, so prefix matching would reuse stale features.
    var prefill_tokens: []const u32 = slot.full_prompt;
    var hot_matched: u32 = 0;
    // Are the restored rows the slot's own? A RAM checkout or a disk restore both credit them.
    var hot_checked_out: bool = false;
    // The DFlash assistant's context rides the prefix cache: a restore
    // forwards no trunk layers, so without it the assistant starts every
    // reused turn blind and drafts against nothing. Measured on a 4-bit target,
    // 160-token generations: a full-prefix hit cost 92.6% -> 66.5% per-draft
    // acceptance and 80.2 -> 60.9 tok/s. Adopted by the Generator below.
    var dflash_restored: ?dflash_mod.DflashCtx = null;
    errdefer if (dflash_restored) |*dc| dc.deinit();
    // Phase D: per-slot model — pull transformer + prefix cache off the
    // slot's LoadedModel. Both stay resident for the slot's lifetime
    // because the conn thread holds a refcount on slot.model.
    const xfm_ptr: *Transformer = slot.model.transformer.?;
    if (slot.model.prefix_cache) |*hc| {
        {
            // Only build a restore target when this request will actually
            // draft — a non-dflash turn leaves the payload in the entry for
            // the next one that does.
            var dfl_target: ?dflash_mod.DflashCtx = if (use_dflash and slot.dflash != null)
                dflash_mod.DflashCtx.init(slot.allocator, slot.dflash.?, 0) catch null
            else
                null;
            errdefer if (dfl_target) |*dc| dc.deinit();
            var dfl_base: usize = 0;
            // Restore by move: the slot names itself, opting into the checkout; `finishSlot`
            // releases it on every path that ends the slot.
            const lookup = hc.lookupAndRestoreWithMedia(
                &slot.cache,
                &slot.moe_seq_offset,
                xfm_ptr.s,
                slot.full_prompt,
                slot.has_tools,
                slot.media,
                if (dfl_target) |*dc| .{ .cache = &dc.cache, .base_pos = &dfl_base } else null,
                @intFromPtr(slot),
                slot.skip_prefix_cache,
            ) catch |err| blk: {
                log.warn("[hot-cache] lookup failed: {s} — proceeding with cold prefill\n", .{@errorName(err)});
                break :blk prefix_cache_mod.LookupResult{ .matched = 0, .full_match = false };
            };
            if (lookup.matched > 0 and lookup.matched <= slot.full_prompt.len) {
                hot_matched = @intCast(lookup.matched);
                prefill_tokens = slot.full_prompt[hot_matched..];
                hot_checked_out = lookup.checked_out or lookup.slot_owned;
                slot.restored_entry = lookup.entry_id;
            }
            if (dfl_target) |*dc| {
                // Adopt only a context that lines up EXACTLY with the trunk
                // cursor — `nextDflash` asserts `absLen() == cache.step`, and
                // a blind start is always a valid fallback.
                if (lookup.dflash_base != null and dfl_base + dc.cache.step == hot_matched) {
                    dc.base_pos = dfl_base;
                    dflash_restored = dc.*;
                } else {
                    dc.deinit();
                }
                dfl_target = null;
            }
        }
    }

    // A request admitted at another prefill's chunk boundary must not turn into a long prefill of its
    // own: the restore covered less than the pre-filter peeked (SSD tier, lien, QSA gap, a checkout
    // taken meanwhile). Nothing has been forwarded yet, so hand the entry back and let it wait.
    if (slot.boundary_max_tail) |max_tail| if (prefill_tokens.len > max_tail) {
        resetRestoredSlot(slot);
        return error.BoundaryTailTooLarge;
    };

    // Evict the hot cache to admit (#353). Here, not on the connection thread: the inference
    // thread is the sole mlx caller and the restore has already happened, so the entry this
    // request uses is the MRU one. The estimator is re-asked after every eviction.
    // `admitted_prefill_chunk` is the pre-eviction width (0 = no pass ran); the width is
    // re-asked after the pass against the live delta.
    var admitted_prefill_chunk: u32 = 0;
    var evicted_live_bytes: u64 = 0;
    if (prefill_admission_fits) |fits_fn| {
        if (slot.model.config) |cfg| {
            // Bills the request's own kv-quant scheme and vision chunking, not the process defaults.
            const Probe = struct {
                cfg: *const model_mod.ModelConfig,
                seq: usize,
                max_tokens: u32,
                kv_cfg: transformer_mod.KVQuantConfig,
                unchunked: bool,
                /// The prefix the hot cache restored and the capacity its buffers hold: already
                /// inside active memory, so billing them again invents a copy.
                warm_matched: u64,
                warm_capacity: u64,
                /// Only a checked-out restore is credited: a refcount share is copied by the first append.
                warm_will_donate: bool,
                fits: *const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool) bool,
                fn call(ctx: ?*anyopaque) bool {
                    const self: *@This() = @ptrCast(@alignCast(ctx.?));
                    return self.fits(self.cfg, self.seq, self.max_tokens, self.kv_cfg, self.unchunked, self.warm_matched, self.warm_capacity, self.warm_will_donate);
                }
            };
            var probe = Probe{
                .cfg = cfg,
                .seq = slot.full_prompt.len,
                .max_tokens = slot.max_tokens,
                .kv_cfg = slot.cache.config,
                .unchunked = generate_mod.visionPrefillUnchunked(slot.vision_embeddings != null),
                .warm_matched = hot_matched,
                .warm_capacity = slot.cache.residentCapacityTokens(),
                .warm_will_donate = hot_checked_out,
                .fits = fits_fn,
            };
            var fits = Probe.call(&probe);
            // A shared restore is billed a whole second copy; taking the entry over moves it instead.
            if (!fits and !hot_checked_out and hot_matched > 0) if (slot.model.prefix_cache) |*hc| {
                if (hc.checkoutRestored(@intFromPtr(slot), slot.full_prompt.len)) {
                    hot_checked_out = true;
                    probe.warm_will_donate = true;
                    fits = Probe.call(&probe);
                }
            };
            if (!fits) {
                // The width admission was billed at, read before anything is evicted.
                if (prefill_request_chunk) |pick_pre| {
                    admitted_prefill_chunk = pick_pre(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, probe.unchunked, probe.warm_matched, probe.warm_capacity, probe.warm_will_donate);
                }
                // Per-model, off the slot: `sch.hot_prefix_cache` is whichever model loaded last.
                // Captured by pointer: a `|hc|` capture would evict from a stack copy.
                const report = if (slot.model.prefix_cache) |*hc|
                    // Never evict the entry this request restored from: its buffers are shared.
                    hc.evictLruToAdmit(slot.full_prompt.len, &probe, Probe.call, true)
                else
                    prefix_cache_mod.EvictionReport{ .admitted = false };
                publishHotCacheResidency(sch);
                // What the allocator returned, not what the cache was billed for.
                evicted_live_bytes = report.bytes;
                if (!report.admitted) {
                    log.warn("[scheduler] prefill refused: {d} tokens do not fit even with an empty hot cache\n", .{slot.full_prompt.len});
                    if (prefill_admission_refused_log) |report_fn| {
                        slot.refused_bill = report_fn(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, probe.unchunked, probe.warm_matched, probe.warm_capacity, probe.warm_will_donate);
                    }
                    // Not `error.OutOfMemory` (the MLX latch's name, a 503): this is a request the
                    // machine cannot hold, a named 400.
                    return error.PrefillDoesNotFit;
                }
            }
        }
    }

    // The prefill width for this request, chosen after the admission pass evicted. Falls
    // back to the load-time pin without the hook or the arch opt-in.
    const req_prefill_chunk: u32 = if (slot.model.config) |cfg| blk: {
        const pin = cfg.pinned_prefill_chunk;
        const pick = prefill_request_chunk orelse break :blk pin;
        const reasked = pick(
            cfg,
            slot.full_prompt.len,
            slot.max_tokens,
            slot.cache.config,
            generate_mod.visionPrefillUnchunked(slot.vision_embeddings != null),
            hot_matched,
            slot.cache.residentCapacityTokens(),
            hot_checked_out,
        );
        const decision = postEvictionPrefillChunk(admitted_prefill_chunk, reasked);
        if (decision.widened) {
            log.info("[prefill] re-ask: width {d} -> {d} after the eviction pass returned {d} MB\n", .{ admitted_prefill_chunk, decision.width, evicted_live_bytes >> 20 });
        } else if (decision.moved) {
            log.warn("[prefill] re-ask: width {d} is NARROWER than the admitted {d} after the eviction pass returned {d} MB — memory moved between the two reads; the re-ask is the live reading, so it runs\n", .{ decision.width, admitted_prefill_chunk, evicted_live_bytes >> 20 });
        }
        break :blk decision.width;
    } else 0;

    // Chunk-boundary decode yields: the hook advances already-decoding
    // streams between this prefill's chunks. Ticks hosted here are billed
    // out of prefill_ns below (the decoding slots got the time).
    var interleave_ctx = InterleaveCtx{ .sch = sch, .slot = slot, .chunk_sw = io_util.Stopwatch.init(sch.io) };
    var write_through_ctx = WriteThroughCtx{ .slot = slot };
    // Per-chunk prefill width context. Stack-scoped like `interleave_ctx`.
    var width_ctx = ChunkWidthCtx{
        .sch = sch,
        .cfg = slot.model.config,
        .kv_bits = if (slot.cache.config.scheme == .off) 16 else slot.cache.config.bits,
        .hc = if (slot.model.prefix_cache) |*p| p else null,
    };

    // Ownership of the restored spec caches transfers AT THE CALL:
    // initWithOptions adopts them and frees them via its own errdefers on
    // any failure past adoption (a mid-prefill disconnect throws
    // error.Cancelled from its chunk loop). DflashCtx holds the
    // KVCache BY VALUE, so a second deinit from our errdefers walked freed
    // mlx handles — SIGSEGV in freeKVEntry (issue #266). Clear FIRST.
    const dflash_pass = dflash_restored;
    dflash_restored = null;
    // Restore by move, the transfer: the last point at which nothing has written to `slot.cache`.
    // Everything above can still refuse and `releaseCheckout` then hands the entry back whole;
    // nothing between here and `initWithOptions` may fail.
    if (slot.model.prefix_cache) |*hc| hc.donateCheckout(@intFromPtr(slot));
    var gen = try Generator.initWithOptions(
        sch.io,
        slot.allocator,
        xfm_ptr,
        slot.model.tokenizer.?,
        prefill_tokens,
        slot.max_tokens,
        sampling,
        slot.eos_token_ids,
        .{
            .pld_enabled = use_pld,
            .drafter_enabled = use_drafter,
            .drafter = if (use_drafter) slot.drafter else null,
            .drafter_block_size = slot.drafter_block_size,
            .dflash_enabled = use_dflash,
            .dflash = if (use_dflash) slot.dflash else null,
            // The dflash-resolved block rides the shared drafter_block_size.
            .dflash_block_size = slot.drafter_block_size,
            // Use the resolved thinking mode and normalize the M5/block-16
            // calibration to this machine's effective assistant width.
            .dflash_min_accepted_per_round = dflashGateMinimum(
                slot.drafter_block_size,
                slot.enable_thinking,
                slot.model.config.?.isMoe(),
            ),
            .lookup_prompt = slot.full_prompt,
            .ctx = slot.ctx,
            // Regular path: skip the lazy preforward so cache.step lands at
            // exactly prompt_len with t1 NOT in cache. Generator.next's
            // transition shim sync-forwards [t1] on the first decode call.
            // PLD/drafter/DFlash init paths already skip preforward unconditionally.
            .skip_lazy_preforward = !use_pld and !use_drafter and !use_dflash,
            .prefix_pos_offset = hot_matched,
            // A restored prefix already holds its image rows: the splice
            // resumes at the placeholder count inside the matched prefix.
            .vision_rows_before = if (slot.vision_embeddings != null and hot_matched > 0)
                generate_mod.countSpliceRows(@ptrCast(slot.full_prompt[0..hot_matched]), xfm_ptr.config.image_token_id, xfm_ptr.config.audio_token_id, xfm_ptr.config.video_token_id)
            else
                0,
            // The width the admission guard billed for this request; the forward can never run wider.
            .pinned_prefill_chunk = companyPrefillChunk(req_prefill_chunk, sch.decodingCount()),
            .decode_share_width_cap = promptLogprobsWidthCap(slot.prompt_logprobs, decodeShareAdmissionCap(sch.liveDecodingCount(), prefillDecodeShare())),
            .dflash_ctx_restored = dflash_pass,
            // Abandoned-prefill abort: the conn thread sets slot.cancelled
            // when the client disconnects; the chunk loop checks it between
            // chunks so a ghost 40K prefill stops within one chunk.
            .cancel_flag = &slot.cancelled,
            // On abort the Generator stores the forwarded prompt length here so the
            // cancelled-prefill commit can bank the partial prefix.
            .cancelled_forwarded = &slot.cancelled_forwarded,
            .prefill_progress = if (observe) &sch.inflight_prefill_tokens else null,
            .prefill_expected = if (observe) &sch.inflight_prefill_expected else null,
            .interleave_hook = if (prefillInterleaveEnabled())
                .{ .ctx = &interleave_ctx, .call = interleaveDecodeTickCb }
            else
                null,
            .write_through_hook = if (writeThroughArmed(slot, prefill_tokens.len))
                .{ .ctx = &write_through_ctx, .call = prefillWriteThroughCb }
            else
                null,
            .chunk_width_hook = if (prefill_chunk_adapt != null and !slot.prompt_logprobs)
                .{ .ctx = &width_ctx, .call = chunkWidthCb, .confirm = chunkWidenConfirmCb }
            else
                null,
            // The install above is process-wide, so it is not the arch gate.
            .adaptive_chunk_width = adaptiveChunkWidthFor(width_ctx.cfg),
            // Init reads logprobs in the split-prefill final-token forward,
            // before any post-init field write.
            .logprobs_n = slot.logprobs_n,
            .prompt_logprobs = if (slot.prompt_logprobs) slot.freshPromptLogprobs() else null,
        },
    );
    slot.prefill_interleaved_ns = interleave_ctx.decode_ns;
    gen.timeout_ns = slot.timeout_ns;
    gen.logprobs_n = slot.logprobs_n;

    slot.legacy_gen = gen;
    // The last chunks' transient is freed AFTER the loop's own per-chunk clear, so it
    // parks in MLX's pool up to the cap and the first decode tick allocates on top of
    // it. Returned once here, at the handover — long-context gate only.
    if (slot.legacy_gen) |*g| g.clearPoolBeforeDecode();
    // The conn thread's `cached_tokens` counted against `xfm.cache` (legacy
    // global cache) which the slot doesn't use. The slot's `cached_tokens`
    // is the hot-cache match (or 0 if the hot cache missed / isn't
    // configured) — those are the only tokens actually present in slot.cache
    // before this turn's prefill. With this override, slot.prompt_tokens
    // reports the full prompt length: gen.prompt_tokens (= prefill_tokens.len)
    // covers the un-cached tail, and `hot_matched` covers the restored prefix.
    slot.cached_tokens = hot_matched;
    slot.prompt_tokens = gen.prompt_tokens + slot.cached_tokens;
    slot.state = .decoding;
    if (hot_matched == 0) {
        const cold_pos = if (slot.moe_seq_offset > 0) slot.moe_seq_offset else slot.cache.step;
        _ = restore_dump.dumpRestoreIfEnabled(&slot.cache, xfm_ptr.s, .{
            .kind = "cold",
            .pos = cold_pos,
            .source = "cold",
        });
    }
}

/// Sum the in-flight generated tokens over the active slots for the live-tok/s
/// gauge, EXCLUDING any slot that already finished/cancelled/errored this tick
/// (its tokens were counted into `generation_tokens_total` by `finishSlot`, so
/// including them here would double-count — and would leave a stale non-zero
/// aggregate once the last slot finishes, breaking the "at rest ⇒ live == total"
/// invariant). Generic over the slot type so the exact filter is unit-testable
/// with a lightweight stub; the real caller passes `[]*Slot`.
fn sumInflightGeneratedTokens(active: anytype) u64 {
    var inflight: u64 = 0;
    for (active) |s| {
        if (s.finished or s.error_code != null or s.cancelled.load(.acquire)) continue;
        inflight += @as(u64, s.completion_tokens);
    }
    return inflight;
}

fn runDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    if (active.len == 0) return;

    // Publish the in-flight generated-token aggregate for the gauge sampler.
    // Runs after the whole tick (all decode work + any finishSlot calls) on
    // every return path — race-free (inference thread owns these slots' fields)
    // and O(active), never per token. Finished slots are excluded, so the last
    // slot's completion drives this to 0 (live == total at rest).
    defer sch.inflight_generated_tokens.store(sumInflightGeneratedTokens(active), .monotonic);

    // Contention discipline for the spec cost model's kv term: it learns
    // from realized round times, and contention only ever ADDS time. Rather
    // than try to correct for it, a busy server simply stops sampling.
    for (active) |s| {
        if (s.legacy_gen) |*g| g.spec_cost_solo = active.len == 1;
    }

    // Phase 3 gate: at len==1, route to legacy single-slot path. Bit-identical
    // to pre-Phase-2 behavior including PLD/drafter speculative decoding.
    // Phase A7 test hook: `MLX_SERVE_FORCE_BATCHED=1` bypasses the gate so the
    // byte-equivalence test can run the batched kernel at active.len==1 and
    // assert it matches the single-slot path token-for-token.
    if (active.len == 1 and !sch.force_batched) {
        try runSingleDecodeTick(sch, active[0]);
        return;
    }

    // active.len >= 2 (or force_batched at len==1): split into batchable +
    // non-batchable. Batchable slots share one `forwardBatchedDecode` call.
    // Non-batchable (slots running PLD/drafter or grammar-constrained) fall
    // back to legacy single-slot decode this tick.
    //
    // Plan 05 Phase D: batched decode requires all participating slots to
    // share a transformer. We partition `batchable` by `slot.model` and
    // emit one batched call per model. The non-batchable bucket doesn't
    // care — each slot runs against its own `slot.model.transformer.?`.
    var batchable_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var batchable_n: usize = 0;
    for (active) |s| {
        const why = sch.batchVerdict(s);
        if (why == .ok and batchable_n < batchable_buf.len) {
            batchable_buf[batchable_n] = s;
            batchable_n += 1;
        } else {
            // legacy single-slot for spec / grammar / overflow
            noteSerial(sch, s, why);
            try runSingleDecodeTick(sch, s);
        }
    }
    if (batchable_n == 0) {
        if (sch.metrics) |m| m.batched_group_size.set(0);
        return;
    }

    // Group batchable slots by model pointer (in-place partition by sort).
    // The order of slots within a model doesn't matter for batched decode;
    // we only need contiguous runs per model.
    std.sort.pdq(*Slot, batchable_buf[0..batchable_n], {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);

    var start: usize = 0;
    while (start < batchable_n) {
        var end = start + 1;
        while (end < batchable_n and batchable_buf[end].model == batchable_buf[start].model) end += 1;
        var group = batchable_buf[start..end];
        // Cap the group by padding waste: the batched kernel pads every slot's
        // KV to the longest in the group, so one long-context stream would make
        // its short neighbours build a tensor orders of magnitude bigger than
        // they need. Sort ascending by kv_len and let `batchedKvKeepCount` say
        // how many still fit; the tail decodes serially this tick.
        if (group.len >= 2) {
            var kv_lens: [32]u32 = undefined;
            var ctx_max: u32 = 0;
            {
                var caches_buf: [MAX_BATCH_GROUP]*const KVCache = undefined;
                for (group, 0..) |g, i| {
                    caches_buf[i] = &g.cache;
                }
                fillGroupPadWasteKvLens(caches_buf[0..group.len], group[0].model.config, 1, kv_lens[0..group.len]);
                ctx_max = groupCtxMax(caches_buf[0..group.len]);
                // Stable insertion sort, ascending, slots and lengths moving together.
                var i: usize = 1;
                while (i < group.len) : (i += 1) {
                    const slot_i = group[i];
                    const len_i = kv_lens[i];
                    var j = i;
                    while (j > 0 and kv_lens[j - 1] > len_i) : (j -= 1) {
                        group[j] = group[j - 1];
                        kv_lens[j] = kv_lens[j - 1];
                    }
                    group[j] = slot_i;
                    kv_lens[j] = len_i;
                }
            }
            const keep = groupKeepCount(kv_lens[0..group.len], groupAttendsPerSlot(group[0]), ctx_max);
            if (keep < group.len) {
                if (!kv_skew_split_logged) {
                    kv_skew_split_logged = true;
                    log.info("[batched] pad-waste cap: kept {d} of {d} slots (waste {d:.2}x, kv_len {d}..{d}), rest serial\n", .{
                        keep,
                        group.len,
                        batchedPadWaste(kv_lens[0..group.len]),
                        kv_lens[0],
                        kv_lens[group.len - 1],
                    });
                }
                for (group[keep..]) |s| {
                    noteSerial(sch, s, .pad_waste);
                    try runSingleDecodeTick(sch, s);
                }
                group = group[0..keep];
            }
        }
        // Honor force_batched even when only one slot is batchable so the
        // test hook actually exercises forwardBatchedDecode at N=1.
        if (group.len >= 2 or (sch.force_batched and group.len == 1)) {
            if (sch.metrics) |m| m.batched_group_size.set(group.len);
            try runBatchedDecodeTick(sch, group);
        } else if (group.len == 1) {
            try runSingleDecodeTick(sch, group[0]);
        }
        start = end;
    }
}

/// What `runPrefill` arms in a slot's Generator init options.
pub const SpecInitWiring = struct {
    use_drafter: bool,
    use_dflash: bool,
    use_pld: bool,
};

/// Per-site spec gating for one slot, as a pure function.
///
/// Speculative decode must be able to ROLL BACK a rejected tail, and the shell
/// rolls back what it owns: the slot's KVCache and ssm_entries. A MODULE-OWNED
/// arch uses neither — by the time a verify forward returns, its own state has
/// already absorbed every draft token, and the shell's snapshot/truncate run
/// over an empty entries array. So every shell-driven spec mode is off there.
///
/// This used to be hand-written per-arch conjuncts. A second module-owned
/// arch arrived with the same `Model.state` shape and a 0-layer shell cache,
/// got no conjunct, and `--pld` drove verify forwards straight through it.
/// One predicate, one place to extend.
pub fn specInitWiring(
    owns_module_state: bool,
    enable_drafter: bool,
    has_drafter: bool,
    has_dflash: bool,
    enable_pld: bool,
) SpecInitWiring {
    if (owns_module_state) return .{
        // PLD and the drafters stay off on a module-owned arch: the drafters
        // need a bound sidecar (DflashModel.bind refuses these archs anyway),
        // and an unmeasured spec mode is worse than none.
        .use_drafter = false,
        .use_dflash = false,
        .use_pld = false,
    };
    // enable_drafter is the request-level "assistant sidecar" switch for BOTH
    // sidecar kinds; the loader guarantees at most one of drafter/dflash is
    // loaded per model. Priority: dflash > gemma drafter > PLD.
    const use_dflash = enable_drafter and has_dflash;
    const use_drafter = !use_dflash and enable_drafter and has_drafter;
    return .{
        .use_drafter = use_drafter,
        .use_dflash = use_dflash,
        .use_pld = !use_dflash and !use_drafter and enable_pld,
    };
}

/// Resolve the request-level switch shared by the classic external drafter
/// and DFlash. Vision stays guarded: placeholder ids alone do not tell an
/// external sidecar how to position image tokens.
pub fn assistantSidecarEnabledForRequest(requested: bool, has_vision: bool) bool {
    return requested and !has_vision;
}

test "drafter admission: a vision request never takes a sidecar, a text one follows the request" {
    try testing.expect(assistantSidecarEnabledForRequest(true, false));
    try testing.expect(!assistantSidecarEnabledForRequest(false, false));
    try testing.expect(!assistantSidecarEnabledForRequest(true, true));
    try testing.expect(!assistantSidecarEnabledForRequest(false, true));
}

/// Which spec mode a decode tick drives for a slot.
pub const SpecTickMode = enum { dflash, drafter, pld, regular };

/// Pure decode-tick dispatch decision. The slot flags carry the REQUEST's
/// wish; the generator-side values carry what `Generator.initWithOptions`
/// actually armed. Every arm must
/// require BOTH: dispatching on the slot flag alone is the exact wiring that
/// put PLD verify forwards through a module-owned trunk (2026-07-31) - the drafter
/// had its generator-state conjunct (`gen.drafter != null`), model-less PLD
/// did not, so runPrefill's `use_pld=false` shaped init options while every
/// tick still called `gen.nextPld`.
pub fn specTickMode(
    slot_enable_drafter: bool,
    gen_has_drafter: bool,
    gen_has_dflash: bool,
    slot_enable_pld: bool,
    gen_pld_enabled: bool,
) SpecTickMode {
    if (slot_enable_drafter and gen_has_dflash) return .dflash;
    if (slot_enable_drafter and gen_has_drafter) return .drafter;
    if (slot_enable_pld and gen_pld_enabled) return .pld;
    return .regular;
}

/// `specTickMode` for a slot and its generator.
fn slotSpecMode(slot: *const Slot, gen: *const Generator) SpecTickMode {
    return specTickMode(
        slot.enable_drafter,
        gen.drafter != null,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
    );
}

/// The prefill already sampled the first token: send it now instead of with the
/// first decode step or speculative round, which would hold it a forward longer.
/// The count stays with the decoder that emits it (its push is swallowed).
fn publishFirstToken(slot: *Slot) void {
    const gen = if (slot.legacy_gen) |*g| g else return;
    if (gen.done or gen.completion_tokens != 0 or slot.logprobs_n > 0 or gen.sampling.constraint != null) return;
    const t1 = gen.next_token_id;
    if (t1 == 0 or generate_mod.isEosId(t1, slot.eos_token_ids)) return;
    slot.pushTokenWithLogprob(t1, null);
    slot.early_first = t1;
}

fn publishSpeculativeBlock(sch: *Scheduler, slot: *Slot, gen: *Generator, tokens: []const u32) void {
    // The Generator has already committed the whole block internally. Publish
    // it incrementally so streaming, cancellation, EOS and usage all observe
    // the same per-token boundary. The generator-side cap guarantees max_tokens
    // cannot land before the returned block ends.
    for (tokens) |t| {
        if (slot.cancelled.load(.acquire)) return;
        if (generate_mod.isEosId(t, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            return;
        }
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            return;
        }
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
}

/// Every single-slot decode tick funnels through here, so a decode-time MLX failure is
/// attributed to the slot whose forward raised it instead of the next request's prefill.
/// The repetition-loop guard every decode tick runs first. True = the slot was finished here.
fn loopGuardTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    // Stop a runaway repetition loop before generating more. Some models (seen
    // on Gemma 4 12B after a large/confusing tool result) collapse into spamming
    // one short cycle — e.g. the thinking opener `<|channel>thought` — forever;
    // with no repeat penalty by default and a generous max_tokens, nothing else
    // halts it until the cap. Checked here, before this tick's step, so it
    // covers the regular, PLD, and drafter paths uniformly.
    const loop_guard_start = gen.loopGuardStart();
    if (loopStopDecision(gen.generated_ids.items[loop_guard_start..])) |relative_stop| {
        if (gen.hasPrePayloadConstraint()) {
            switch (try gen.forceConstraintTransition(slot.allocator)) {
                .committed => |boundary| {
                    slot.pushToken(boundary);
                    slot.completion_tokens += 1;
                    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
                    if (boundary != 0) slot.was_pad_only = false;
                    log.warn("[grammar] reasoning boundary forced after repetition loop at {d} generated tokens\n", .{gen.generated_ids.items.len - 1});
                    return true;
                },
                .activated => return true,
                // The completion cap won the race with recovery: fall through
                // to the ordinary loop-stop cut.
                .refused => {},
            }
        }
        if (loopRecoveryAction(gen.sampling.think_bound, gen.generated_ids.items, gen.completion_tokens, gen.max_tokens) == .close_thought) {
            log.warn("[loop-recover] repetition loop inside the thought at {d} generated tokens; closing it\n", .{gen.generated_ids.items.len});
            const tb = gen.sampling.think_bound.?;
            tb.fired = true;
            _ = try commitForcedTick(sch, slot, gen, tb.forced, .think_bound);
            return true;
        }
        var stop = relative_stop;
        stop.trim_start += loop_guard_start;
        // Never cut silently: the 2026-07-14 php.html post-mortem took log
        // archaeology because this guard left no trace of having fired. The
        // tier and the trim point are logged too — five cuts in a row is a
        // different diagnosis from one, and the trim is what breaks the chain.
        log.warn("[loop-stop] degenerate tail loop cut after {d} generated tokens (finish_reason={s} details={s} tier={s} trim_start={d})\n", .{
            gen.generated_ids.items.len, stop.finish_reason, stop.finish_details,
            @tagName(stop.tier),         stop.trim_start,
        });
        slot.finish_details = stop.finish_details;
        if (loopTrimEnabled()) slot.loop_trim_start = stop.trim_start;
        finishSlot(sch, slot, stop.finish_reason);
        return true;
    }
    if (try thinkBoundTick(sch, slot, gen)) return true;
    return toolForceTick(sch, slot, gen);
}

/// A thinking budget at its limit: commit the early-stop line and the closer
/// through the model this tick, and decode the answer regular from here on
/// (`spec_disable_reason = .think_bound`). True = the slot's tick is spent.
fn thinkBoundTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    const tb = gen.sampling.think_bound orelse return false;
    tb.observe(gen.generated_ids.items);
    if (!tb.due()) return false;
    tb.fired = true;
    if (!generate_mod.forcedBoundaryCanContinue(gen.completion_tokens, gen.max_tokens, tb.forced.len + 1)) {
        log.warn("[think-bound] budget {d} reached with no room to close the thought (max_tokens {d})\n", .{ tb.budget, gen.max_tokens });
        return false;
    }
    if (try commitForcedTick(sch, slot, gen, tb.forced, .think_bound)) {
        log.info("[think-bound] reasoning budget {d} reached at {d} generated tokens; thought closed\n", .{ tb.budget, gen.generated_ids.items.len });
    }
    return true;
}

/// `tool_choice` that obliges a call: once the thought closes, commit the
/// tool-call opener through the model, which writes the call from there.
fn toolForceTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    const tf = gen.sampling.tool_force orelse return false;
    const d = tf.due(gen.generated_ids.items, gen.completion_tokens, gen.max_tokens) orelse return false;
    tf.fired = true;
    // The model's own next token is published first: a call it already opened is not opened twice.
    const tokens = if (try gen.upcomingToken() == d.tokens[0]) d.tokens[1..] else d.tokens;
    if (tokens.len == 0) return false;
    if (!generate_mod.forcedBoundaryCanContinue(gen.completion_tokens, gen.max_tokens, tokens.len + 1)) {
        log.warn("[tool-choice] no room to force the call (max_tokens {d})\n", .{gen.max_tokens});
        return false;
    }
    if (try commitForcedTick(sch, slot, gen, tokens, .tool_choice)) {
        log.info("[tool-choice] call opener forced at {d} generated tokens{s}\n", .{ gen.generated_ids.items.len, if (d.closes_thought) " (thought closed for it)" else "" });
    }
    return true;
}

/// Commit `forced` through the model and publish it; the slot decodes regular
/// from here. False = the pending token stopped the slot first.
fn commitForcedTick(sch: *Scheduler, slot: *Slot, gen: *Generator, forced: []const u32, reason: generate_mod.SpecDisableReason) !bool {
    const r = try gen.commitForcedTokens(slot.allocator, forced);
    defer slot.allocator.free(r.emitted);
    for (r.emitted) |t| {
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
    if (r.stopped) {
        finishSlot(sch, slot, gen.finish_reason);
        return false;
    }
    gen.spec_disabled_runtime = true;
    gen.spec_disable_reason = reason;
    return true;
}

fn thinkBoundFired(gen: *const Generator) bool {
    const tb = gen.sampling.think_bound orelse return false;
    return tb.fired;
}

fn runSingleDecodeTick(sch: *Scheduler, slot: *Slot) !void {
    var inner_err: ?anyerror = null;
    runSingleDecodeTickInner(sch, slot) catch |e| switch (e) {
        // A reasoning-protocol dead end is this request's failure; the tick
        // error path would fail every active slot.
        error.NoValidProtocolToken, error.InvalidProtocolTransition, error.InvalidProtocolPayload => {
            log.err("[grammar] reasoning protocol failed ({s}); failing this request only\n", .{@errorName(e)});
            slot.markError(@errorName(e));
        },
        else => inner_err = e,
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] decode aborted: MLX failure mid-generation ({s}) — failing this request, the server keeps serving\n", .{@errorName(mlx_err)});
        slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runSingleDecodeTickInner(sch: *Scheduler, slot: *Slot) !void {
    const gen = if (slot.legacy_gen) |*g| g else {
        slot.markError("no_generator");
        return;
    };

    if (try loopGuardTick(sch, slot, gen)) return;

    // NOTE: no `!gen.spec_disabled_runtime` short-circuit here — the
    // generators handle the disabled fallback internally, and `nextPld`'s
    // disabled branch is also where the mid-request RE-ENABLE check lives
    // (bypassing it pinned PLD off for the rest of the request even when the
    // generated tail turned echo-heavy).
    const tick_mode: SpecTickMode = if (thinkBoundFired(gen)) .regular else specTickMode(
        slot.enable_drafter,
        gen.drafter != null,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
    );
    if (tick_mode == .drafter) {
        const result = try gen.nextDrafter(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    if (tick_mode == .dflash) {
        const result = try gen.nextDflash(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    if (tick_mode == .pld) {
        const result = try gen.nextPld(slot.allocator, slot.pld_draft_len, slot.pld_key_len);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    // Regular path.
    const tok_opt = try gen.next(slot.allocator);
    if (tok_opt == null) {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    }
    const t = tok_opt.?;
    // Phase A5: capture per-token logprob. `gen.last_logprob` ownership
    // transfers into slot.logprobs_buf (gen sets the field, we null it here).
    // Published in the SAME critical section as the token so a streaming
    // reader that has been handed token i can read entry i — see
    // `pushTokenWithLogprob`.
    var lp_take: ?generate_mod.LogprobResult = null;
    if (slot.logprobs_n > 0) {
        if (gen.last_logprob) |lp| {
            lp_take = lp;
            gen.last_logprob = null;
        }
    }
    // Publish the reasoning→payload boundary (if this is the token that
    // carries it) BEFORE the token itself, so a streaming reader handed token
    // i already sees its span.
    if (gen.takeConstraintSpan()) |span| {
        slot.publishConstraintSpan(span.token_index, span.byte_offset);
    }
    slot.pushTokenWithLogprob(t, lp_take);
    if (t != 0) slot.was_pad_only = false;
    slot.completion_tokens = gen.completion_tokens;
}

test "all speculative blocks publish through one per-token accounting loop" {
    // Class guard for block-returning decoders. The Generator has already
    // advanced by the whole accepted block when the scheduler receives it;
    // copying that final count while publishing token 1 truncates the stream
    // and over-reports usage. Every mode must use the shared incremental loop.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn runSingleDecodeTick(") orelse return error.MissingDecodeTick;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\n}\n\ntest \"all speculative blocks") orelse return error.MissingDecodeTickEnd;
    const body = source[start..end];
    const shared_call = "publishSpeculativeBlock(sch, slot, gen, result.?.tokens);";
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, body, shared_call));
    // The one remaining final-count assignment belongs to the scalar regular
    // path, after every speculative arm. It must never reappear in a block.
    const final_assign = "slot.completion_tokens = gen.completion_tokens;";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, final_assign));
    try testing.expect(std.mem.lastIndexOf(u8, body, shared_call).? < std.mem.indexOf(u8, body, final_assign).?);
}

test "DFlash cache payload is committed only when it spans the trunk prefix" {
    try testing.expect(dflashContextCoversPrefix(128, 128));
    try testing.expect(!dflashContextCoversPrefix(96, 128));
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn commitSlotIfApplicable(") orelse return error.MissingCommitSlot;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn finishSlot(") orelse return error.MissingFinishSlot;
    const body = source[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "dflashContextCoversPrefix(dc.absLen(), total_len)") != null);
}

test "a finish over a latched MLX failure ends the request as an ERROR, never a 200" {
    const Stub = struct {
        finished: ?[]const u8 = null,
        errored: ?[]const u8 = null,
        fn markFinished(self: *@This(), reason: []const u8) void {
            self.finished = reason;
        }
        fn markError(self: *@This(), name: []const u8) void {
            self.errored = name;
        }
    };

    var clean = Stub{};
    publishSlotTerminator(&clean, "stop", null);
    try testing.expectEqualStrings("stop", clean.finished.?);
    try testing.expect(clean.errored == null);

    // Latched: no "stop" is published at all.
    var poisoned = Stub{};
    publishSlotTerminator(&poisoned, "stop", "OutOfMemory");
    try testing.expect(poisoned.finished == null);
    try testing.expectEqualStrings("OutOfMemory", poisoned.errored.?);
    try testing.expect(Slot.errorNameIsMemory(poisoned.errored.?));

    var shape = Stub{};
    publishSlotTerminator(&shape, "length", "MlxFailure");
    try testing.expect(shape.finished == null);
    try testing.expectEqualStrings("MlxFailure", shape.errored.?);
    try testing.expect(!Slot.errorNameIsMemory(shape.errored.?));
}

test "cancelled-prefill commit length: floor, clamp, and zero" {
    // A cancelled prefill only pays its way into the LRU once the forwarded
    // prefix is past the chat-template-prologue class (~dozens of tokens);
    // below the floor the entry is pollution, not saved work.
    try testing.expect(cancelledPrefillCommitLen(0, 1000) == null);
    try testing.expect(cancelledPrefillCommitLen(1, 1000) == null);
    try testing.expect(cancelledPrefillCommitLen(255, 1000) == null);
    try testing.expectEqual(@as(?usize, 256), cancelledPrefillCommitLen(256, 1000));
    try testing.expectEqual(@as(?usize, 300), cancelledPrefillCommitLen(300, 1000));
    // cache.step can never exceed the prompt (restore clamps to matched and
    // chunks stop at the tail); clamp defensively anyway so a bad step can
    // never key tokens the KV does not hold.
    try testing.expectEqual(@as(?usize, 1000), cancelledPrefillCommitLen(5000, 1000));
    // A short prompt is not worth an entry at any step.
    try testing.expect(cancelledPrefillCommitLen(50, 50) == null);
}

test "the cleanup drain commits a cancelled slot before deinit" {
    // Class guard: a decode-phase cancel is pulled straight into the cleanup
    // queue by `complete()` and NEVER passes through finishSlot — without a
    // commit in the drain, its prompt+generated KV dies with the slot. The
    // commit must run BEFORE deinit (the snapshot refcount-shares live
    // buffers; after deinit they are freed).
    const source = @embedFile("scheduler.zig");
    const drain_start = std.mem.indexOf(u8, source, "for (cleanup_batch[0..cleanup_n])") orelse return error.MissingCleanupDrain;
    const region = source[drain_start..@min(drain_start + 1600, source.len)];
    const commit_pos = std.mem.indexOf(u8, region, "commitSlotIfApplicable") orelse return error.DrainDoesNotCommit;
    const deinit_pos = std.mem.indexOf(u8, region, ".deinit()") orelse return error.MissingDeinit;
    try testing.expect(commit_pos < deinit_pos);
    // The SSD tier has no finishSlot flush on this path — the drain must
    // flush what it just committed itself.
    try testing.expect(std.mem.indexOf(u8, region, "flushPendingDisk") != null);
}

test "armShutdownDeadline: the first SIGTERM sets the deadline and a second never moves it" {
    var slot = std.atomic.Value(i64).init(0);
    armShutdownDeadline(&slot, 5_000, 45);
    try testing.expectEqual(@as(i64, 50_000), slot.load(.acquire));
    armShutdownDeadline(&slot, 9_000, 45);
    try testing.expectEqual(@as(i64, 50_000), slot.load(.acquire));
    var off = std.atomic.Value(i64).init(0);
    armShutdownDeadline(&off, 5_000, 0);
    try testing.expectEqual(@as(i64, 0), off.load(.acquire));
}

test "idleEvictTickMs: sweeps well inside the window without spinning" {
    // Bar: a quarter of the window, clamped to [1s, 30s] — eviction lands near
    // the configured time without the sweep becoming a busy loop.
    try testing.expectEqual(@as(i64, 5_000), idleEvictTickMs(20_000));
    try testing.expectEqual(@as(i64, 15_000), idleEvictTickMs(60_000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(900_000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(1000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(2000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(28_800_000));
    try testing.expect(idleEvictTickMs(0) >= 1000);
}

test "the inference loop parks without holding the sleep-inhibition assertion" {
    // Pin release < park < acquire and startup acquire < load.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn inferenceLoop(") orelse return error.MissingInferenceLoop;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn ") orelse return error.MissingInferenceLoopEnd;
    const body = source[start..end];
    const pass_start = std.mem.indexOf(u8, source, "fn chatPass(") orelse return error.MissingChatPass;
    const pass_end = std.mem.indexOfPos(u8, source, pass_start + 1, "\nfn ") orelse return error.MissingChatPassEnd;
    const pass = source[pass_start..pass_end];
    const drop = std.mem.indexOf(u8, pass, "sleep_inhibit.setActive(false);") orelse return error.MissingSleepRelease;
    const park = std.mem.indexOf(u8, pass, "sch.queue_cond.waitUncancelable(sch.io, &sch.queue_mu);") orelse return error.MissingPark;
    const hold = std.mem.indexOfPos(u8, pass, park, "sleep_inhibit.setActive(true);") orelse return error.MissingSleepAcquire;
    try testing.expect(drop < park);
    try testing.expect(park < hold);
    try testing.expect(std.mem.indexOf(u8, pass, "while (!hasWorkPendingLocked(sch)") != null);
    const boot_load = std.mem.indexOf(u8, body, "doLoadOnInferenceThread(sch, params)") orelse return error.MissingStartupLoad;
    const boot_arm = std.mem.indexOf(u8, body, "sleep_inhibit.setActive(true);") orelse return error.MissingSleepAcquire;
    try testing.expect(boot_arm < boot_load);
    try testing.expect(std.mem.indexOf(u8, body, "defer sleep_inhibit.release();") != null);
}

test "commitSlotIfApplicable routes a Generator-less slot to the cancelled-prefill commit" {
    // Prefill abort: `Generator.initWithOptions` throws error.Cancelled from
    // its chunk loop, so `slot.legacy_gen` is never assigned — the legacy
    // `else return` silently dropped every chunk that DID forward. The
    // cancelled-prefill arm must be reachable from commitSlotIfApplicable,
    // and hybrids are excluded inside it (their stride checkpoints die with
    // the failed Generator init; a checkpoint-less hybrid entry restores as
    // a cold miss while still occupying an LRU slot).
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn commitSlotIfApplicable(") orelse return error.MissingCommitSlot;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn finishSlot(") orelse return error.MissingFinishSlot;
    const body = source[start..end];
    const route_pos = std.mem.indexOf(u8, body, "commitCancelledPrefillSlot") orelse return error.MissingRoute;
    // `was_pad_only` initializes TRUE and only flips on the first pushed
    // token, so a guard on it ABOVE the Generator-less arm makes that arm
    // unreachable (live 2026-08-22: "prefill aborted" logged, nothing
    // committed, full re-prefill on retry).
    const pad_pos = std.mem.indexOf(u8, body, "slot.was_pad_only") orelse return error.MissingPadGuard;
    try testing.expect(route_pos < pad_pos);

    const cp_start = std.mem.indexOf(u8, source, "fn commitCancelledPrefillSlot(") orelse return error.MissingCancelledPrefillFn;
    const cp_end = std.mem.indexOfPos(u8, source, cp_start + 1, "\nfn ") orelse return error.MissingCancelledPrefillEnd;
    const cp_body = source[cp_start..cp_end];
    try testing.expect(std.mem.indexOf(u8, cp_body, "cancelledPrefillCommitLen") != null);
    // The authoritative commit length is the slot's `cancelled_forwarded` counter - 
    // `cache.step` only advances when Generator init COMPLETES, so it reads
    // 0 on every aborted prefill (found live: step=0 while pos=1536).
    try testing.expect(std.mem.indexOf(u8, cp_body, "slot.cancelled_forwarded") != null);
    try testing.expect(std.mem.indexOf(u8, cp_body, "commitWithMediaState") != null);
}

test "Generator.initWithOptions reports the forwarded length on cancel" {
    // The chunk-loop cancel path must store the forwarded prompt length before returning
    // error.Cancelled: the cancelled-prefill commit has no other authoritative length.
    // Anchored on the prefill loop's abandoned-request abort comment so a decode-loop
    // cancel check can't satisfy it.
    const source = @embedFile("generate.zig");
    const anchor = std.mem.indexOf(u8, source, "Abandoned-request abort") orelse return error.MissingAbortComment;
    const region = source[anchor..@min(anchor + 2400, source.len)];
    try testing.expect(std.mem.indexOf(u8, region, "cancelled_forwarded") != null);
    try testing.expect(std.mem.indexOf(u8, region, "error.Cancelled") != null);
}

test "runPrefill clears restored spec-cache ownership BEFORE Generator.initWithOptions (issue #266)" {
    // Generator.initWithOptions ADOPTS the hot-cache-restored DFlash
    // context and frees it via its own errdefers on any failure past the
    // adoption point — a mid-prefill client disconnect throws
    // error.Cancelled from its chunk loop. DflashCtx holds its
    // KVCache BY VALUE, so runPrefill's own errdefers then walked the same
    // entries slice + mlx handles a second time: SIGSEGV in
    // KVCache.deinit -> freeKVEntry (issue #266, disconnect storms on long
    // agent prompts). Ownership transfers AT THE CALL, so the locals must
    // be cleared before the try, never after it.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn runPrefill(") orelse return error.MissingRunPrefill;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn ") orelse return error.MissingRunPrefillEnd;
    const body = source[start..end];
    const call = std.mem.indexOf(u8, body, "try Generator.initWithOptions(") orelse return error.MissingInitCall;
    const dfl = std.mem.indexOf(u8, body, "dflash_restored = null") orelse return error.MissingDflashClear;
    try testing.expect(dfl < call);
}

test "DFlash gate policy follows effective block width and resolved thinking" {
    // block_size includes the always-emitted anchor, so block 16 has 15 draft
    // positions and block 7 has 6. The M5 calibration is normalized to that
    // actual draft width instead of being imposed as an absolute threshold on
    // machines without the wide verification lane.
    try testing.expectApproxEqAbs(@as(f32, 2.0), dflashGateMinimum(16, false, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dflashGateMinimum(16, true, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), dflashGateMinimum(7, false, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.4), dflashGateMinimum(7, true, false), 0.0001);
    try testing.expectEqual(@as(f32, 0), dflashGateMinimum(1, false, false));

    // A SPARSE target's verify reads every expert its block routes to, so the
    // dense width scaling under-bars it: at block 5 the scaled value is 0.53
    // and one target measured break-even at 1.63 accepted/round. The floor
    // binds there and in the thinking arm, and never lowers a bar the width
    // scaling already set higher.
    try testing.expectApproxEqAbs(@as(f32, 1.8), dflashGateMinimum(5, false, true), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.8), dflashGateMinimum(5, true, true), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), dflashGateMinimum(16, false, true), 0.0001);
    try testing.expectEqual(@as(f32, 0), dflashGateMinimum(1, false, true));
}

test "every server scheduler path forwards resolved thinking to the DFlash gate" {
    const source = @embedFile("server.zig");

    // The direct streaming submission and the shared non-streaming submit
    // must both populate the field.
    var submit_pos: usize = 0;
    var submits: usize = 0;
    while (std.mem.indexOfPos(u8, source, submit_pos, "sch.submit(.{")) |start| {
        const end = std.mem.indexOfPos(u8, source, start, "});") orelse return error.UnclosedSchedulerSubmit;
        try testing.expect(std.mem.indexOf(u8, source[start..end], ".enable_thinking =") != null);
        submits += 1;
        submit_pos = end + 3;
    }
    try testing.expectEqual(@as(usize, 2), submits);

    // Positional calls into the non-streaming wrapper must pass the resolved
    // value.
    var lines = std.mem.splitScalar(u8, source, '\n');
    var calls: usize = 0;
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "nonStreamingViaScheduler(") == null) continue;
        if (std.mem.indexOf(u8, line, "fn nonStreamingViaScheduler(") != null) continue;
        calls += 1;
        try testing.expect(std.mem.indexOf(u8, line, "enable_thinking") != null or
            std.mem.indexOf(u8, line, ", false, false, use_pld") != null);
    }
    try testing.expectEqual(@as(usize, 1), calls);
}

/// Per-slot follow-up to a batched forward: `record` moves the id ledger, `publish` samples and streams.
const BatchedTickAction = struct { record: bool, publish: bool };

fn batchedTickAction(cancelled: bool) BatchedTickAction {
    // The KV row is already written, so the ledger records it even when cancelled.
    return .{ .record = true, .publish = !cancelled };
}

fn runBatchedDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runBatchedDecodeTickInner(sch, active) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] batched decode aborted: MLX failure ({s}) — failing all {d} slots in the group\n", .{ @errorName(mlx_err), active.len });
        for (active) |s| s.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runBatchedDecodeTickInner(sch: *Scheduler, active: []*Slot) !void {
    const N = active.len;
    if (N == 0) return;
    const allocator = sch.allocator;

    // Phase D: all batched slots must share the same model — the caller
    // (`runDecodeTick`) partitions by `slot.model` before dispatching here.
    // The first slot's transformer is authoritative; debug-assert the rest
    // match to surface partitioning bugs early.
    const xfm_ptr: *Transformer = active[0].model.transformer.?;
    if (std.debug.runtime_safety) {
        for (active) |s| std.debug.assert(s.model.transformer.? == xfm_ptr);
    }

    // Legacy→batched transition: a slot arriving from a legacy single-slot
    // tick (or fresh from prefill) carries lazy pipeline state — a lookahead
    // token ALREADY FORWARDED into its KV cache plus `pending_logits` for
    // the position after it. Consume that state via `drainPipelineForBatch`
    // (emit the lookahead, sample the new next_token_id from the pending
    // logits). Dropping it and re-forwarding `next_token_id` — the pre-fix
    // behavior — appended a duplicate cache position and re-emitted an
    // already-emitted token, corrupting any stream whose slot joined a
    // batch mid-generation (tests/test_batched_transition.sh). Slots that
    // finish during the drain are excluded from the batch.
    const live = try allocator.alloc(*Slot, N);
    defer allocator.free(live);
    var live_n: usize = 0;
    for (active) |slot| {
        const gen = if (slot.legacy_gen) |*g| g else {
            slot.markError("no_generator");
            continue;
        };
        if (gen.has_pending_logits or gen.has_pending_token) {
            const emitted = gen.drainPipelineForBatch(slot.allocator) catch |err| {
                slot.markError(@errorName(err));
                continue;
            };
            if (emitted) |tok| {
                slot.pushToken(tok);
                if (tok != 0) slot.was_pad_only = false;
                slot.completion_tokens = gen.completion_tokens;
                if (generate_mod.isEosId(gen.next_token_id, slot.eos_token_ids)) {
                    finishSlot(sch, slot, "stop");
                    continue;
                }
                if (slot.completion_tokens >= slot.max_tokens) {
                    finishSlot(sch, slot, "length");
                    continue;
                }
            } else {
                // checkStop fired on the pipelined lookahead (EOS / pad-run
                // / max_tokens / timeout); nothing to emit.
                slot.completion_tokens = gen.completion_tokens;
                finishSlot(sch, slot, gen.finish_reason);
                continue;
            }
        }
        live[live_n] = slot;
        live_n += 1;
    }
    if (live_n == 0) return;
    const batch = live[0..live_n];

    // Build inputs.
    const next_tokens = try allocator.alloc(u32, live_n);
    defer allocator.free(next_tokens);
    const ctxs = try allocator.alloc(*ForwardCtx, live_n);
    defer allocator.free(ctxs);
    const rope_offsets = try allocator.alloc(u32, live_n);
    defer allocator.free(rope_offsets);

    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        next_tokens[i] = gen.next_token_id;
        ctxs[i] = &gen.ctx;
    }

    for (batch, 0..) |slot, i| rope_offsets[i] = @intCast(slot.cache.step);
    const logits_arr = try xfm_ptr.forwardBatchedDecode(next_tokens, ctxs, rope_offsets);
    defer {
        for (logits_arr) |a| _ = mlx.mlx_array_free(a);
        allocator.free(logits_arr);
    }
    // The batched forward advances only its scratch offset; each slot's own
    // position moves here so a slot leaving the batch resumes serial from
    // the right place (QSA reads it for kv length + tail rule).
    for (batch) |slot| slot.moe_seq_offset += 1;

    // `gen.sampling`, not `slot.sampling`: the Generator's copy passed
    // the initWithOptions chokepoint and carries the model's
    // reserved-token suppression mask; the slot's copy is the raw
    // request params.
    var sample_params: [MAX_BATCH_GROUP]generate_mod.SamplingParams = undefined;
    var sample_rows: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var sample_ids: [MAX_BATCH_GROUP]i32 = undefined;
    std.debug.assert(live_n == logits_arr.len);
    std.debug.assert(live_n <= sample_params.len);
    for (batch, 0..) |slot, i| {
        sample_params[i] = slot.legacy_gen.?.sampling;
        sample_rows[i] = logits_arr[i];
    }
    if (live_n > 0) {
        try generate_mod.sampleRows(sample_ids[0..live_n], sample_rows[0..live_n], sample_params[0..live_n], xfm_ptr.s);
        for (batch, 0..) |slot, i| slot.legacy_gen.?.sampling.draw = sample_params[i].draw;
    }

    // Sample per slot, emit prev id, set new next_token_id.
    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        const act = batchedTickAction(slot.cancelled.load(.acquire));
        const sampled: ?i32 = if (act.publish) sample_ids[i] else null;

        const emit = gen.next_token_id;
        gen.generated_ids.append(slot.allocator, emit) catch |err| {
            slot.markError(@errorName(err));
            continue;
        };
        gen.advanceStep(1);
        if (emit != 0) slot.was_pad_only = false;
        const val = sampled orelse continue;
        gen.next_token_id = @intCast(val);

        // Stop checks (mirrors Generator.checkStop).
        if (generate_mod.isEosId(emit, slot.eos_token_ids)) {
            // Per existing contract, emit IS NOT yielded when it's EOS — the
            // STOP token comes BEFORE the yield. But here the cache has
            // already moved past it. The legacy path's checkStop runs on the
            // NEXT token (it's checked before emit). To preserve that
            // behavior we emit and then mark finished if `next_token_id` is
            // EOS (i.e. STOP is the next sampled token, ignored).
            slot.pushToken(emit);
            slot.completion_tokens = gen.completion_tokens;
            // not finished yet; next tick's checkStop on next_token_id ends it
        } else {
            slot.pushToken(emit);
            slot.completion_tokens = gen.completion_tokens;
        }

        if (generate_mod.isEosId(gen.next_token_id, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            continue;
        }
        if (gen.next_token_id == 0) {
            gen.consecutive_pad += 1;
            if (gen.consecutive_pad >= 3) {
                finishSlot(sch, slot, "stop");
                continue;
            }
        } else {
            gen.consecutive_pad = 0;
        }
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            continue;
        }
    }
}

const testing = std.testing;

test "runPrefill wires the interleave hook and bills its decode ticks out of prefill_ns" {
    const src = @embedFile("scheduler.zig");
    // The hook is wired at the ONE Generator construction site, env-gated.
    const wire = ".interleave" ++ "_hook = if (prefillInterleaveEnabled())";
    try testing.expect(std.mem.indexOf(u8, src, wire) != null);
    // Interleaved decode time is charged to the DECODING slots (they got the
    // tokens), so the prefilling slot's prefill_ns must exclude it or
    // prefill_tps under-reports on every interleaved prefill.
    const bill = "slot.prefill_ns = prefill_sw.read() -| slot.prefill_" ++ "interleaved_ns;";
    try testing.expect(std.mem.indexOf(u8, src, bill) != null);
}

test "modelBatchable rejects MoE" {
    {
        // MoE: isMoe() returns true when num_experts > 0.
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.num_experts = 8;
        try testing.expect(!modelBatchable(&cfg));
    }
}

test "batchedKvKeepCount: padding waste caps the group, and the long slots are the ones dropped" {
    // Even lengths: no padding waste, everybody batches.
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 1000 }));
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 900, 1000, 1100, 1200 }));

    // The case this exists for: three short streams and one long one. Batching
    // all four pads to 4 x 100000 = 400k against 103k useful (3.9x); dropping
    // the long one leaves 3 x 1000 vs 3000 (1.0x).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 100_000 }));

    // Two long ones: the pair still batches together, since 2 x 100000 against
    // 200000 useful wastes nothing — the veto is about the padding, not length.
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&[_]u32{ 100_000, 100_000 }));
    // One short slot among two long ones still batches: 3 x 100000 padded
    // against 200010 useful is 1.5x, inside the bar. The veto is about the
    // WASTE the padding creates, not about any slot being an outlier.
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 10, 100_000, 100_000 }));

    // One slot never "batches", and neither does an empty group.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{1000}));
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{}));

    // Nothing prefilled yet: no padding to waste, so nothing is vetoed (the
    // ready gate, not this one, is what keeps unprefilled slots out).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));

    // A pathological pair degrades to no batch at all rather than making the
    // 1-token slot build a 200k-wide tensor nobody billed. This is why the bar
    // has to sit below 2.0 — at 2.0 a pair is unvetoable by construction.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{ 1, 200_000 }));
}

test "the batched group is capped by padding waste before it is dispatched" {
    // Source-scan class guard: the grouping loop must consult the cap. Without
    // it a single long-context stream makes every short neighbour materialize
    // a padded KV tensor sized by ITS length — a per-tick transient no gate
    // bills, whose failure mode is an uncatchable Metal OOM.
    const src = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, src, "// Group batchable slots by model pointer") orelse return error.MissingGrouping;
    const end = std.mem.indexOfPos(u8, src, start, "\n}\n") orelse return error.MissingGroupingEnd;
    const body = src[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "groupKeepCount(") != null);
    // ...and the dropped slots must still be ticked, or they never advance.
    try testing.expect(std.mem.indexOf(u8, body, "noteSerial(sch, s, .pad_waste)") != null);
}

test "groupKeepCount: a group that attends per slot pads nothing, so the cap is the stacked arm's" {
    const skew = [_]u32{ 1000, 1000, 1000, 100_000 };
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&skew, false, 0));
    try testing.expectEqual(@as(usize, 4), groupKeepCount(&skew, true, 0));
    // Below the per-slot floor the stacked arm runs and its cap holds.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{ 10, 900 }, true, 0));
}

test "groupKeepCount: the small-pad floor keeps a group whole only at long context" {
    // A long stream billed at the gather arm's 2052-row cap beside two short sub-agents: 3x pad.
    const billed = [_]u32{ 10, 10, 2052 };
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&billed, false, PAD_FREE_MIN_CTX - 1));
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&billed, false, PAD_FREE_MIN_CTX));
    // Past the floor's billed bound the cap holds at any context.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV + 1 }, false, 300_000));
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV }, false, 300_000));
}

test "modelBatchable permits pure-attention" {
    // Defaults are all zero / null → vanilla pure-attention path.
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    try testing.expect(modelBatchable(&cfg));
}

test "a spec_disabled_runtime slot is batchable, and that is the documented trade" {
    // `slotTicksRegular` admits a generator whose speculation turned itself off
    // at runtime. That is what recovered the throughput (9.6 -> 12.4 tok/s per
    // stream on 4 concurrent 5.5k prompts), and it costs something real: the
    // batched tick does not call `nextPld`, so its periodic re-enable check
    // cannot run while the slot is batched. Both halves are load-bearing, so
    // both are stated here — if the re-enable check ever moves onto the batched
    // path, or the clause is dropped, this is the note to revisit.
    const src = @embedFile("scheduler.zig");
    const hs = std.mem.indexOf(u8, src, "fn slotTicksRegular(") orelse return error.MissingHelper;
    const he = std.mem.indexOfPos(u8, src, hs + 1, "\n    }\n") orelse return error.MissingHelperEnd;
    const hbody = src[hs..he];
    // The clause that admits it...
    try testing.expect(std.mem.indexOf(u8, hbody, "gen.spec_disabled_runtime or") != null);
    // ...and the trade it makes, written down where it is made.
    try testing.expect(std.mem.indexOf(u8, hbody, "re-enable check") != null);
    try testing.expect(std.mem.indexOf(u8, hbody, "SOLO slot never reaches the batched path") != null);
}

test "the batched gate reads DISPATCH, not the armed spec flags" {
    // A prompt that merely n-gram-scores high enough to ARM PLD used to veto
    // batched decode forever, even after PLD's own yield gate disabled itself
    // at runtime: neither speculation nor batching (9.7 vs 14.3 tok/s per
    // stream, 4 concurrent 4232-token prompts). The armed flags
    // are the REQUEST's wish; specTickMode is what the tick actually
    // dispatches, and spec_disabled_runtime is what recovered the throughput.
    const src = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, src, "fn batchVerdict(self: *const Scheduler") orelse return error.MissingBatchable;
    const end = std.mem.indexOfPos(u8, src, start + 1, "\n    }\n") orelse return error.MissingBatchableEnd;
    const body = src[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "slotTicksRegular(slot)") != null);
    // No armed-flag read may come back into the gate.
    try testing.expect(std.mem.indexOf(u8, body, "slot.enable_pld") == null);
    try testing.expect(std.mem.indexOf(u8, body, "slot.enable_drafter") == null);

    // ...and the helper answers from BOTH: the generator's runtime kill and
    // the authoritative dispatch decision.
    const hs = std.mem.indexOf(u8, src, "fn slotTicksRegular(") orelse return error.MissingHelper;
    const he = std.mem.indexOfPos(u8, src, hs + 1, "\n    }\n") orelse return error.MissingHelperEnd;
    const hbody = src[hs..he];
    try testing.expect(std.mem.indexOf(u8, hbody, "gen.spec_disabled_runtime") != null);
    try testing.expect(std.mem.indexOf(u8, hbody, "slotSpecMode(") != null);

    // The dispatch answer itself: a live drafter slot never ticks regular, an
    // unarmed one always does.
    try testing.expectEqual(SpecTickMode.drafter, specTickMode(true, true, false, true, true));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, true, false));
}

test "admitPendingTick: exclusive single-flight FIFO contract" {
    const A: usize = 0xA0;
    const B: usize = 0xB0;
    var out: [16]usize = undefined;

    // Held while a live slot on the same exclusive model is decoding — the
    // module-owned class: a second admitted slot deinit+rebuilds the module-owned
    // decode state at cache.step==0 and both requests then interleave on it.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // Admits once no live slot holds the model.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Two exclusive candidates on the SAME model in one tick: only the
    // first admits — the tick-claim covers slots not yet in `decoding`.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = A, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Distinct exclusive models admit independently.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = B, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
        try testing.expectEqual(@as(usize, 1), out[1]);
    }
}

test "admitPendingTick: non-exclusive concurrency and queue order preserved" {
    const A: usize = 0xA0;
    const L: usize = 0x10;
    var out: [16]usize = undefined;

    // Non-exclusive candidates (per-slot state) admit freely
    // even beside live slots — their serial-tick interleave is safe.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = L, .exclusive = false }};
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &active, &out));
    }
    // A held exclusive candidate must not head-of-line-block a later
    // candidate on another model (requests for OTHER models keep flowing).
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
        try testing.expectEqual(@as(usize, 1), out[0]);
    }
    // `out` caps the admitted count (mirrors to_prefill's 16), in order.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        var small: [2]usize = undefined;
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &small));
        try testing.expectEqual(@as(usize, 0), small[0]);
        try testing.expectEqual(@as(usize, 1), small[1]);
    }
}

test "inferenceLoop pending drain routes through admitPendingTick" {
    // A pure admission fn nobody calls is a silent no-op (the specTickMode /
    // hardcoded use_drafter=false class). Needles are ++-split so this
    // test's own source can't satisfy the scan.
    const src = @embedFile("scheduler.zig");
    const call = "admitPendingTick(" ++ "cand_buf[0..n_cands], live_buf[0..n_live], &admit_idx)";
    try testing.expect(std.mem.indexOf(u8, src, call) != null);
    // The candidates' exclusive bit must come from the model predicate.
    const pred = ".exclusive = modelExclusiveDecode(" ++ "s.model)";
    try testing.expect(std.mem.indexOf(u8, src, pred) != null);
    // The pre-gate unconditional drain shape must be GONE — its survival
    // would mean a path still admits without the gate.
    const old = "to_prefill[n_prefill] = sch.pending." ++ "orderedRemove(0)";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
}

test "modelExclusiveDecode asks the transformer, never one hardcoded arch" {
    // A hardcoded per-arch check here once missed a second module-owned arch
    // (same `Model.state` shape, same `reset = cache.step == 0` rebuild), and
    // two concurrent requests shared one state. The predicate now lives beside the fields it reads
    // (`Transformer.module_owned_state_fields`), and this pins the delegation.
    // Needles are ++-split so this test's source can't satisfy the scan.
    const src = @embedFile("scheduler.zig");
    const delegated = "t.ownsModuleDecode" ++ "State()";
    try testing.expect(std.mem.indexOf(u8, src, delegated) != null);
}

test "sumInflightGeneratedTokens sums active slots, excludes finished/cancelled/errored" {
    // Lightweight stub carrying exactly the fields the aggregate reads — proves
    // the live-gauge filter without constructing a real (mlx-backed) Slot.
    const StubSlot = struct {
        completion_tokens: u32,
        finished: bool,
        error_code: ?[]const u8,
        cancelled: std.atomic.Value(bool),
        fn make(tok: u32) @This() {
            return .{ .completion_tokens = tok, .finished = false, .error_code = null, .cancelled = std.atomic.Value(bool).init(false) };
        }
    };
    var a = StubSlot.make(10);
    var b = StubSlot.make(20);
    var done = StubSlot.make(99);
    done.finished = true; // already counted in generation_tokens_total
    var errd = StubSlot.make(50);
    errd.error_code = "OutOfMemory";
    var cxl = StubSlot.make(7);
    cxl.cancelled.store(true, .release);

    // Two still-decoding slots (10 + 20); the finished/errored/cancelled ones
    // are excluded, so the aggregate is 30 — never double-counting the tail.
    const active = [_]*StubSlot{ &a, &b, &done, &errd, &cxl };
    try testing.expectEqual(@as(u64, 30), sumInflightGeneratedTokens(active[0..]));

    // At rest: the last decoding slot finishes ⇒ aggregate collapses to 0,
    // pinning the "live == total at rest" invariant test_metrics.sh checks.
    a.finished = true;
    b.finished = true;
    try testing.expectEqual(@as(u64, 0), sumInflightGeneratedTokens(active[0..]));
}

const OutcomeStub = struct {
    finished: bool = false,
    error_code: ?[]const u8 = null,
    finish_reason: []const u8 = "",
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    metrics_recorded: bool = false,
    first_token_ns: u64 = 5_000_000,
    prefill_ns: u64 = 4_000_000,
    decode_ns: u64 = 20_000_000,
    prompt_tokens: u32 = 100,
    completion_tokens: u32 = 10,
    cached_tokens: u32 = 0,

    /// `finishSlot` on the inference thread: records, then publishes the terminator.
    fn finish(self: *OutcomeStub, m: ?*metrics_mod.Metrics, reason: []const u8) void {
        recordSlotEnd(m, self, finishOutcome(reason, null));
        self.finished = true;
        self.finish_reason = reason;
    }
    /// `Slot.markError` mid-decode: no finishSlot follows.
    fn fail(self: *OutcomeStub, name: []const u8) void {
        if (self.error_code != null or self.finished) return;
        self.error_code = name;
    }
    /// The connection thread's `Scheduler.complete`: sets `cancelled` on EVERY completion.
    fn complete(self: *OutcomeStub) void {
        self.cancelled.store(true, .release);
    }
};

fn expectOutcomes(m: *const metrics_mod.Metrics, success: u64, cancelled: u64, failed: u64, rejected: u64) !void {
    try testing.expectEqual(success, m.requests_success_total.load());
    try testing.expectEqual(cancelled, m.requests_cancelled_total.load());
    try testing.expectEqual(failed, m.requests_failed_total.load());
    try testing.expectEqual(rejected, m.requests_rejected_total.load());
}

test "outcome row 1: a disconnect while the request waits in pending counts cancelled once" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "outcome row 2: a disconnect during prefill is counted once, not again by the drain" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.complete();
    s.finish(&m, "cancelled");
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "outcome row 3: a disconnect during decode never reaches finishSlot and the drain counts it" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.completion_tokens = 7;
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
    // A cancelled request never feeds the success histograms.
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
}

test "outcome row 4: a normal completion is success once even though complete() sets cancelled" {
    var m = metrics_mod.Metrics.init();
    for ([_][]const u8{ "stop", "length" }) |reason| {
        var s = OutcomeStub{};
        s.finish(&m, reason);
        s.complete();
        recordSlotCleanup(&m, &s);
    }
    try expectOutcomes(&m, 2, 0, 0, 0);
    try testing.expectEqual(@as(u64, 20), m.generation_tokens_total.load());
}

test "outcome row 5: a mid-decode generation error counts failed and leaves the histograms alone" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("MlxFailure");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.e2e_latency_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
}

test "outcome row 5b: a finish over a latched MLX failure counts failed, not success" {
    try testing.expectEqual(metrics_mod.Outcome.failed, finishOutcome("stop", "OutOfMemory"));
    try testing.expectEqual(metrics_mod.Outcome.success, finishOutcome("stop", null));
    try testing.expectEqual(metrics_mod.Outcome.cancelled, finishOutcome("cancelled", null));
}

test "outcome row 6: an error followed by a client disconnect is one failure, never two outcomes" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("MlxFailure");
    recordSlotCleanup(&m, &s);
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
}

test "outcome row 7: a slot refused before its first forward is a rejection, not a failure" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("PrefillDoesNotFit");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 0, 1);
}

test "outcome row 9: metrics off does no work and leaves the slot untouched" {
    var s = OutcomeStub{};
    s.complete();
    recordSlotCleanup(null, &s);
    s.finish(null, "stop");
    try testing.expect(!s.metrics_recorded);
}

test "loopStopReason: a degenerate tail cut reports stop, a healthy tail is not cut" {
    // A loop guard is an intentional server stop, not exhaustion of the
    // requested output budget. Reporting "length" makes clients such as pi
    // run context-overflow recovery and compact a mostly-empty context. Tool
    // calls from this truncated buffer are suppressed separately at emission.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);

    // Healthy varied tail: never cut.
    for (0..40) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 7 + 3)));
    try testing.expect(loopStopReason(ids.items) == null);

    // Collapse into a short cycle (the php.html shape: "server-side scripting
    // language, " ≈ a 6-token cycle) past the guard's span threshold.
    for (0..generate_mod.degenerate_loop_min_span / 6 + 1) |_| {
        for ([_]u32{ 101, 202, 303, 404, 505, 606 }) |t| {
            try ids.append(testing.allocator, t);
        }
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);
}

test "loopStopReason: a LONG-period sentence loop is cut at the second tier" {
    // The 2026-08-02 shooter wrap-up failure: a two-sentence cycle ("The game
    // is complete. Let me do a final review... Let me verify main.js...") of
    // ~58 tokens repeated 26 times sailed through the 8-token-period tier and
    // was never cut. Tier 2 scans periods 9..64 and requires 10 exact
    // repetitions — verbatim-identical long cycles at that count are
    // degeneration, not content.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    for (0..30) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 3 + 11)));

    // 58-token cycle, one rep short of the tier's span bar: NOT cut.
    var cycle: [58]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @as(u32, @intCast(1000 + i));
    for (0..generate_mod.degenerate_loop_long_min_span / cycle.len) |_| try ids.appendSlice(testing.allocator, &cycle);
    try testing.expect(loopStopReason(ids.items) == null);

    // The next repetition crosses it — cut as an intentional stop.
    try ids.appendSlice(testing.allocator, &cycle);
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);
}

test "loopStopDecision: the wire reason is stop and the CAUSE rides beside it" {
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &[_]u32{ 5, 6, 7 });
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 });
    }

    const stop = loopStopDecision(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", stop.finish_reason);
    try testing.expectEqualStrings("repetition_loop", stop.finish_details);
    try testing.expectEqual(generate_mod.DegenerateTail.Tier.exact_cycle, stop.tier);
    // Trimmed to the honest prefix plus one copy of the cycle.
    try testing.expectEqual(@as(usize, 6), stop.trim_start);

    // Once a deferred boundary has been committed, the old reasoning loop is
    // outside the guard window and cannot stop the very next answer tick.
    try ids.append(testing.allocator, 99);
    const answer_start = ids.items.len;
    try testing.expect(loopStopDecision(ids.items[answer_start..]) == null);

    // A new loop wholly inside the constrained answer retains the existing
    // stop/repetition result, with an absolute trim point for response code.
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 });
    }
    const answer_loop = loopStopDecision(ids.items[answer_start..]) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", answer_loop.finish_reason);
    try testing.expect(answer_loop.trim_start + answer_start >= answer_start);

    // Healthy output decides nothing at all — no reason, and nothing to trim.
    var healthy: [512]u32 = undefined;
    for (&healthy, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(loopStopDecision(&healthy) == null);
    try testing.expect(loopStopReason(&healthy) == null);
}

test "loopStopReason: periods past the long tier stay uncut" {
    // A 70-token exact cycle (> long-tier max 64) repeated many times is
    // outside both tiers — the guard stays scoped rather than judging whole
    // repeated paragraphs.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    var cycle: [70]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @as(u32, @intCast(2000 + i));
    for (0..12) |_| try ids.appendSlice(testing.allocator, &cycle);
    try testing.expect(loopStopReason(ids.items) == null);
}

test "loopStopReason: a VARIED-phrasing restatement loop is cut at the near-repeat tier" {
    // The agent-traffic shape (2026-08-04): the same intent restated forever in
    // slightly different words. No exact cycle exists, so tiers 1 and 2 are
    // blind and this ran to max_tokens.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 },
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 },
    };
    var i: usize = 0;
    while (ids.items.len < generate_mod.near_repeat_min_span + 32) : (i += 1) {
        try ids.appendSlice(testing.allocator, phrasings[i % phrasings.len]);
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);

    // A long answer that keeps introducing new material is untouched, however
    // repetitive its scaffolding.
    var healthy = std.ArrayList(u32).empty;
    defer healthy.deinit(testing.allocator);
    var line: u32 = 0;
    while (healthy.items.len < generate_mod.near_repeat_window + 32) : (line += 1) {
        try healthy.appendSlice(testing.allocator, &[_]u32{ 10, 11, 12 });
        try healthy.append(testing.allocator, 1000 + line);
        try healthy.appendSlice(testing.allocator, &[_]u32{ 13, 14 });
    }
    try testing.expect(loopStopReason(healthy.items) == null);
}

test "specInitWiring: a module-owned arch gets no shell spec mode" {
    // Spec decode must be able to ROLL BACK a rejected tail. The shell rolls
    // back its KVCache/ssm_entries - which a module-owned arch does not use, so
    // by the time the verify returns, the module has already absorbed every
    // draft and the shell's snapshot/truncate run over an EMPTY entries array.
    // A module-owned arch (0-layer shell cache, state on the module) once got
    // no conjunct, so `--pld` drove verify forwards straight through it.
    // Args: (owns_module_state, enable_drafter, has_drafter, has_dflash, enable_pld)

    // Plain arch: a loaded drafter outranks PLD.
    {
        const w = specInitWiring(false, true, true, false, true);
        try testing.expect(w.use_drafter and !w.use_dflash and !w.use_pld);
    }
    {
        const w = specInitWiring(false, false, false, false, true);
        try testing.expect(!w.use_drafter and w.use_pld);
    }
    // A flag with no loaded handle never arms.
    {
        const w = specInitWiring(false, true, false, false, false);
        try testing.expect(!w.use_drafter and !w.use_dflash and !w.use_pld);
    }

    // DFlash rides the enable_drafter switch: dflash > drafter > PLD.
    {
        const w = specInitWiring(false, true, false, true, true);
        try testing.expect(w.use_dflash and !w.use_drafter and !w.use_pld);
    }
    {
        const w = specInitWiring(false, true, true, true, true);
        try testing.expect(w.use_dflash and !w.use_drafter and !w.use_pld);
    }
    // enable_drafter:false opts BOTH sidecar kinds out.
    {
        const w = specInitWiring(false, false, false, true, true);
        try testing.expect(!w.use_dflash and !w.use_drafter and w.use_pld);
    }

    // Module-owned: everything off.
    {
        const w = specInitWiring(true, true, true, true, true);
        try testing.expect(!w.use_drafter and !w.use_dflash and !w.use_pld);
    }
}

test "runPrefill gates spec through specInitWiring, not per-arch conjuncts" {
    // A pure predicate nobody calls is a silent no-op. Needles are ++-split so
    // this test's own source cannot satisfy the scan.
    const src = @embedFile("scheduler.zig");
    // Keyed on the call site's own bindings, not on a `specInitWiring(` prefix
    // this test's own arms would satisfy.
    inline for (.{ "const use_drafter = wiring" ++ ".use_drafter;", "const use_dflash = wiring" ++ ".use_dflash;", "const use_pld = wiring" ++ ".use_pld;" }) |needle| {
        try testing.expect(std.mem.indexOf(u8, src, needle) != null);
    }
    // The exclusion must come from the shared predicate, not a new arch list.
    const from_predicate = "transformer.?.moduleSpec" ++ "Wiring()";
    try testing.expect(std.mem.indexOf(u8, src, from_predicate) != null);
}

test "specTickMode: every spec arm requires the GENERATOR's armed state, not the slot flag alone" {
    // The PLD-corruption wiring class (2026-07-31): runPrefill's
    // per-site guard computed use_pld=false for a module-owned slot and wired
    // it into Generator init options, but the decode tick dispatched on
    // `slot.enable_pld` alone, so every tick still called `gen.nextPld` and
    // its verify forward appended draft tokens into the module-owned
    // state with no rollback. The drafter was saved only by its accidental
    // generator-state conjunct (`gen.drafter != null`); PLD has no model
    // handle, so its conjunct must be the generator's post-chokepoint `pld_enabled`.
    // Args: (slot_enable_drafter, gen_has_drafter, gen_has_dflash, slot_enable_pld, gen_pld_enabled)

    // Slot wants PLD, generator was NOT armed (chokepoint or per-site guard
    // flipped it off) -> the tick must run the regular path.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, true, false));
    // Slot wants PLD and init armed it -> PLD runs.
    try testing.expectEqual(SpecTickMode.pld, specTickMode(false, false, false, true, true));
    // Generator armed but the slot never asked (stale generator state must
    // not resurrect spec either) -> regular.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, true));

    // The drafter keeps its both-sides contract.
    try testing.expectEqual(SpecTickMode.drafter, specTickMode(true, true, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false, false));

    // Priority: drafter > PLD (the spec-dispatch rule).
    try testing.expectEqual(SpecTickMode.drafter, specTickMode(true, true, false, true, true));

    // DFlash: slot's enable_drafter + generator's dflash handle; outranks the
    // gemma drafter (a loaded sidecar is the explicit choice). Generator handle
    // alone never resurrects it, and a dflash generator with the slot flag off
    // stays regular (the specTickMode both-sides contract).
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(true, false, true, false, false));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(true, true, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false, false));
}

test "MLX_SERVE_SSD_WRITE_THROUGH=0 takes mechanism 3 out of the prefill, and only that" {
    try testing.expect(writeThroughEnabledFromEnv(null));
    try testing.expect(writeThroughEnabledFromEnv(""));
    try testing.expect(writeThroughEnabledFromEnv("1"));
    try testing.expect(writeThroughEnabledFromEnv("true"));
    try testing.expect(!writeThroughEnabledFromEnv("0"));
}

test "writeThroughSpanReached: a sub-chunk warm turn never persists inside the prefill" {
    const chunk: u32 = 1024;
    // Warm turn: a restored 32k prefix plus a 31-token instruction tail.
    try testing.expect(!writeThroughSpanReached(31, chunk));
    try testing.expect(!writeThroughSpanReached(0, chunk));
    try testing.expect(!writeThroughSpanReached(1023, chunk));
    try testing.expect(writeThroughSpanReached(1024, chunk));
    try testing.expect(writeThroughSpanReached(4096, chunk));
    try testing.expect(writeThroughSpanReached(65_665, chunk));
    try testing.expect(writeThroughSpanReached(4096, 4096));
    try testing.expect(!writeThroughSpanReached(4095, 4096));
}

test "postEvictionPrefillChunk: the re-asked width is the one that runs, in BOTH directions" {
    // 1. Admitted at 1024, affords 4096 after the pass: runs at 4096, logs the widen.
    {
        const d = postEvictionPrefillChunk(1024, 4096);
        try testing.expectEqual(@as(u32, 4096), d.width);
        try testing.expect(d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 8192), postEvictionPrefillChunk(512, 8192).width);

    // 2. No eviction pass ran: the re-ask is the only ask, nothing reportable.
    {
        const d = postEvictionPrefillChunk(0, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(!d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(0, 512).width);
    try testing.expectEqual(@as(u32, 0), postEvictionPrefillChunk(0, 0).width);

    // 3. A narrower re-ask still wins (`@max` would widen on memory that is gone).
    {
        const d = postEvictionPrefillChunk(4096, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(d.moved);
        try testing.expect(!d.widened);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(2048, 512).width);

    // 4. Unchanged memory is a no-op in both directions.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |w| {
        const d = postEvictionPrefillChunk(w, w);
        try testing.expectEqual(@as(u32, w), d.width);
        try testing.expect(!d.widened and !d.moved);
    }

    // 5. Never below the ladder floor: this function computes no width of its own.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |floor| {
        try testing.expectEqual(@as(u32, floor), postEvictionPrefillChunk(8192, floor).width);
    }
}

test "prefix-cache commit declines a pad-only generation, never a zero-token one" {
    // Bar: a zero-token answer still commits; only an all-pad generation is declined.
    try testing.expect(!commitDeclinesPadOnly(0, true));
    try testing.expect(!commitDeclinesPadOnly(0, false));
    try testing.expect(commitDeclinesPadOnly(3, true));
    try testing.expect(!commitDeclinesPadOnly(3, false));
}

test "[short-gen] carries both token counts, the ids and what they decode to" {
    // Bar: the line carries emitted vs realized and survives quotes, newlines and a 300-byte answer.
    var buf: [1024]u8 = undefined;
    const ids = [_]u32{ 151645, 17 };
    try testing.expectEqualStrings(
        "[short-gen] reason=stop emitted=0 realized=2 ids=[151645,17] bytes=\"hi\" path=dflash",
        formatShortGen(&buf, "stop", 0, &ids, "hi", "dflash"),
    );

    const esc = formatShortGen(&buf, "length", 2, &.{}, "a\"b\nc", "serial");
    try testing.expectEqualStrings(
        "[short-gen] reason=length emitted=2 realized=0 ids=[] bytes=\"a\\\"b\\nc\" path=serial",
        esc,
    );

    var long: [300]u8 = undefined;
    @memset(&long, 'x');
    const capped = formatShortGen(&buf, "stop", 1, &.{}, &long, "pld");
    try testing.expectEqual(@as(usize, SHORT_GEN_TEXT_CAP), std.mem.count(u8, capped, "x"));
}

test "a cancel mid batched tick keeps generated_ids level with the KV rows" {
    // Bar: kv rows == full_prompt + generated_ids even when the slot cancels mid tick.
    const prompt_len: usize = 7;
    var kv_rows = prompt_len;
    var ids: usize = 0;
    for ([_]bool{ false, false, false, true }) |cancelled| {
        kv_rows += 1;
        const act = batchedTickAction(cancelled);
        if (act.record) ids += 1;
        if (cancelled) try testing.expect(!act.publish);
    }
    try testing.expectEqual(kv_rows, prompt_len + ids);
}

test "companyStreak: only consecutive ticks with company count" {
    try testing.expectEqual(@as(u8, 2), companyStreak(companyStreak(0, true), true));
    try testing.expectEqual(@as(u8, 0), companyStreak(1, false));
}

test "yieldsToCompany: sparse-target DFlash yields, PLD and the others do not" {
    try testing.expect(yieldsToCompany(.dflash, true, 1, 5));
    try testing.expect(!yieldsToCompany(.pld, false, 1, 5));
    for ([_]SpecTickMode{ .drafter, .regular, .pld }) |m| try testing.expect(!yieldsToCompany(m, true, 9, 5));
}

test "yieldsToCompany: a dense DFlash slot keeps speculating until the others are a crowd" {
    try testing.expect(!yieldsToCompany(.dflash, false, 1, 5));
    try testing.expect(!yieldsToCompany(.dflash, false, 4, 5));
    try testing.expect(yieldsToCompany(.dflash, false, 5, 5));
    try testing.expect(yieldsToCompany(.dflash, false, 7, 5));
}

test "companyPrefillChunk: a prefill narrows only while someone decodes" {
    try testing.expectEqual(@as(u32, 8192), companyPrefillChunk(8192, 0));
    try testing.expectEqual(@as(u32, 2048), companyPrefillChunk(8192, 1));
    try testing.expectEqual(@as(u32, 1024), companyPrefillChunk(1024, 3));
    try testing.expectEqual(@as(u32, 2048), companyPrefillChunk(0, 1));
}

test "promptLogprobsWidthCap: an echo prefill narrows to its cap, 0 (uncapped) included" {
    try testing.expectEqual(@as(u32, 0), promptLogprobsWidthCap(false, 0));
    try testing.expectEqual(@as(u32, 1024), promptLogprobsWidthCap(false, 1024));
    try testing.expectEqual(PROMPT_LOGPROBS_CHUNK, promptLogprobsWidthCap(true, 0));
    try testing.expectEqual(PROMPT_LOGPROBS_CHUNK, promptLogprobsWidthCap(true, 1024));
    try testing.expectEqual(@as(u32, 128), promptLogprobsWidthCap(true, 128));
}

test "decode share: a live share caps the width only while someone decodes" {
    const w = DECODE_SHARE_PREFILL_CHUNK;
    try testing.expectEqual(@as(u32, 0), decodeShareAdmissionCap(0, 0.5));
    try testing.expectEqual(@as(u32, 0), decodeShareAdmissionCap(2, 0));
    try testing.expectEqual(w, decodeShareAdmissionCap(1, 0.5));
    try testing.expectEqual(@as(u32, 8192), decodeShareWidthCap(8192, 1, 0));
    try testing.expectEqual(@as(u32, 8192), decodeShareWidthCap(8192, 0, 0.5));
    try testing.expectEqual(w, decodeShareWidthCap(8192, 1, 0.5));
    try testing.expectEqual(@as(u32, 512), decodeShareWidthCap(512, 1, 0.5));
}

test "decode share: parse clamps above 0.9 and rejects anything that is not a share" {
    try testing.expectEqual(@as(f32, 0), try parseDecodeShare("0"));
    try testing.expectEqual(@as(f32, 0.5), try parseDecodeShare("0.5"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("0.95"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("inf"));
    for ([_][]const u8{ "", "abc", "nan", "-0.3", "0.5x" }) |bad| {
        try testing.expectError(error.InvalidDecodeShare, parseDecodeShare(bad));
    }
    // The flag outranks the env; neither set is 0; a bad env is an error, not 0.
    try testing.expectEqual(@as(f32, 0.3), try resolveDecodeShare("0.3", "0.5"));
    try testing.expectEqual(@as(f32, 0.5), try resolveDecodeShare(null, "0.5"));
    try testing.expectEqual(DEFAULT_DECODE_SHARE, try resolveDecodeShare(null, null));
    try testing.expectEqual(@as(f32, 0), try resolveDecodeShare("0", null));
    try testing.expectError(error.InvalidDecodeShare, resolveDecodeShare(null, "abc"));
}

test "decode share: the budget is chunk * S / (1 - S), zero when off" {
    const ms = std.time.ns_per_ms;
    try testing.expectEqual(@as(u64, 0), decodeShareBudgetNs(400 * ms, 0));
    try testing.expectApproxEqAbs(@as(f64, 400 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.5))), 1e3);
    try testing.expectApproxEqAbs(@as(f64, 3600 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.9))), 1e4);
    try testing.expectApproxEqAbs(@as(f64, 400 * ms * 3 / 7), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.3))), 1e4);
}

const FakeDecodeTicks = struct {
    left: u32,
    ns: u64,
    calls: u32 = 0,
    fn tick(ctx: *anyopaque) u64 {
        const f: *FakeDecodeTicks = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        if (f.left == 0) return 0;
        f.left -= 1;
        return f.ns;
    }
};

test "decode share: a boundary ticks to the budget, keeps today's count at S=0, stops when decoders run out" {
    const ms = std.time.ns_per_ms;
    // S=0.5 on a 400 ms chunk with 8 ms ticks: 50 ticks, 400 ms of decode.
    var full = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const r = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &full, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 50), r.ticks);
    try testing.expectEqual(@as(u64, 400 * ms), r.spent_ns);
    // S=0 is exactly today's schedule, whatever the chunk cost.
    for ([_]u64{ 100, 300, 2300, 8000 }) |c| {
        var legacy = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
        const l = runOwedDecodeTicks(0, c * ms, 8 * ms, &legacy, FakeDecodeTicks.tick);
        try testing.expectEqual(interleaveTicksFor(c * ms, 8 * ms), l.ticks);
    }
    // The decoders finish after 3 more ticks: the 4th call returns 0 and the loop stops.
    var dry = FakeDecodeTicks{ .left = 3, .ns = 8 * ms };
    const d = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &dry, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 4), d.ticks);
    try testing.expectEqual(@as(u32, 4), dry.calls);
    try testing.expectEqual(@as(u64, 32 * ms), d.spent_ns);
    // Nobody decoding at the boundary: the first tick measured 0, no more are tried.
    var none = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const n = runOwedDecodeTicks(0.5, 2300 * ms, 0, &none, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 1), n.ticks);
    try testing.expectEqual(@as(u32, 0), none.calls);
}

test "interleaveTicksFor: decode keeps a quarter of wall time across a slow chunk, capped" {
    const ms = std.time.ns_per_ms;
    try testing.expectEqual(@as(u32, 8), interleaveTicksFor(8000 * ms, 80 * ms));
    try testing.expectEqual(@as(u32, 2), interleaveTicksFor(300 * ms, 50 * ms));
    try testing.expectEqual(@as(u32, 1), interleaveTicksFor(100 * ms, 80 * ms));
    try testing.expectEqual(@as(u32, 1), interleaveTicksFor(8000 * ms, 0));
}

test "loop recovery: a loop inside an open think block closes the thought once" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = generate_mod.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &[_]u32{ 5, 6 });
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 });

    try testing.expectEqual(LoopAction.close_thought, loopRecoveryAction(&tb, ids.items, 400, 4096));
    // No room to close the thought: the ordinary cut.
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, ids.items, 4095, 4096));
    // Once fired, a second loop is cut.
    tb.fired = true;
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, ids.items, 400, 4096));
}

test "loop recovery: an answer-side loop and an unarmed request are still cut" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, CLOSE, 32 };
    var tb = generate_mod.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    const closed = [_]u32{ 5, 6, CLOSE, 7, 8 };
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, &closed, 5, 4096));
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(null, &closed, 5, 4096));
}

test "cancelAllInFlight: a request being prefilled at a chunk boundary is cancelled too" {
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.queue_mu = .init;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.prefilling_from = 0;
    sch.n_prefilling = 0;
    var slot: Slot = undefined;
    slot.io = testing.io;
    slot.cancelled = .init(false);
    slot.out_mu = .init;
    slot.out_cond = .init;
    slot.out_event = .unset;
    sch.boundary_slot = &slot;
    sch.cancelAllInFlight();
    try testing.expect(slot.cancelled.load(.acquire));
}

test "hot-cache view: a model's cache leaving takes the view with it, to the default chat model first" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 4, 0, null);
    defer reg.deinit();
    var chat_cfg = model_mod.ModelConfig{};
    chat_cfg.model_type = "gemma4";
    const a = try reg.registerStub("a", "/m/a", 64);
    const b = try reg.registerStub("b", "/m/b", 64);
    const c = try reg.registerStub("c", "/m/c", 64);
    const media = try reg.registerStub("media", "/m/media", 64);
    defer for ([_]*LoadedModel{ a, b, c, media }) |e| {
        e.config = null;
    };
    for ([_]*LoadedModel{ a, b, c, media }) |e| {
        e.config = &chat_cfg;
        e.state = .ready;
        e.prefix_cache = prefix_cache_mod.HotPrefixCache.init(testing.allocator, 4);
    }
    media.config = null;
    reg.default_id = "c";
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = reg.io;

    // Another model leaving does not touch the view.
    sch.hot_prefix_cache = &a.prefix_cache.?;
    moveHotCacheViewOff(&sch, b);
    try testing.expectEqual(@as(?*prefix_cache_mod.HotPrefixCache, &a.prefix_cache.?), sch.hot_prefix_cache);
    // The owner leaving hands it to the default chat model, never to an entry with no config.
    moveHotCacheViewOff(&sch, a);
    try testing.expectEqual(@as(?*prefix_cache_mod.HotPrefixCache, &c.prefix_cache.?), sch.hot_prefix_cache);
    // With no other chat model resident, the view is dropped, never left on a freed cache.
    b.state = .unloaded;
    a.state = .unloaded;
    moveHotCacheViewOff(&sch, c);
    try testing.expectEqual(@as(?*prefix_cache_mod.HotPrefixCache, null), sch.hot_prefix_cache);
}

fn budget200Mb(_: *model_mod.ModelConfig, _: u64, _: BudgetRevise, idle: *u64) u64 {
    idle.* = 0;
    return 200 << 20;
}

test "hot-cache view: a revise that follows an unload raises a budget but never lowers it" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 4, 0, null);
    defer reg.deinit();
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "gemma4";
    const m = try reg.registerStub("m", "/m/m", 64);
    defer m.config = null;
    m.config = &cfg;
    m.state = .ready;
    m.prefix_cache = prefix_cache_mod.HotPrefixCache.init(testing.allocator, 4);
    const hc = &m.prefix_cache.?;
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = reg.io;
    sch.allocator = testing.allocator;
    sch.current_model = null;
    sch.hot_prefix_cache = null;
    sch.prefix_cache_mem_resolver = budget200Mb;
    sch.prefix_cache_mem_bytes = 0;
    sch.metrics = null;
    sch.digest_mu = .init;
    sch.hot_cache_digests = &.{};
    sch.digest_residency = 0;
    sch.resident_hot_cache_bytes = std.atomic.Value(u64).init(0);
    sch.reclaimable_hot_cache_bytes = std.atomic.Value(u64).init(0);

    hc.max_kv_bytes = 600 << 20;
    reviseHotCacheBudgets(&sch, true);
    try testing.expectEqual(@as(u64, 600 << 20), hc.max_kv_bytes);
    hc.max_kv_bytes = 100 << 20;
    reviseHotCacheBudgets(&sch, true);
    try testing.expectEqual(@as(u64, 200 << 20), hc.max_kv_bytes);
    // A load's revise is honest both ways.
    hc.max_kv_bytes = 600 << 20;
    reviseHotCacheBudgets(&sch, false);
    try testing.expectEqual(@as(u64, 200 << 20), hc.max_kv_bytes);
}

test "a refusal right after a two-model eviction waits for the memory to come back" {
    const GiB: u64 = 1024 * 1024 * 1024;
    var mark: model_registry_mod.UnloadMark = .{};
    mark.record(1_000, 100 * GiB);
    mark.record(1_100, 17 * GiB);
    try std.testing.expect(waitForUnload(211 * GiB, 108 * GiB, mark, 1_200));
}
