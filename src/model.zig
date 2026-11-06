const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const modelopt_pack = @import("modelopt_pack.zig");
const model_discovery = @import("model_discovery.zig");
const tokenizer_mod = @import("tokenizer.zig");
const kv_quant_mod = @import("kv_quant.zig");

pub const HiddenAct = enum { gelu_approx, silu };

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
/// Upper bound on a vision tower's per-layer type table.
pub const MAX_VISION_LAYERS = 64;

pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,

    pub fn fromString(name: []const u8) ?QuantMode {
        return std.meta.stringToEnum(QuantMode, name);
    }

    /// Mode string for mlx_quantized_matmul / mlx_gather_qmm / mlx_dequantize.
    pub fn cstr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
            .mxfp8 => "mxfp8",
        };
    }

    /// Affine is the only mode whose checkpoints carry per-group biases.
    pub fn hasBiases(self: QuantMode) bool {
        return self == .affine;
    }

    /// The e2m1 nibble modes (byte scales, no biases): MXFP4 g32, NVFP4 g16.
    pub fn isFp4(self: QuantMode) bool {
        return self == .mxfp4 or self == .nvfp4;
    }
};

/// Pooling operation for embedding requests: masked mean over real positions,
/// followed by L2 normalization. It is the only mode served.
pub const PoolingMode = enum {
    mean,

    pub fn fromString(s: []const u8) ?PoolingMode {
        if (std.mem.eql(u8, s, "mean")) return .mean;
        return null;
    }
};

/// Parse a sentence-transformers `1_Pooling/config.json`. Returns the pooling
/// mode when the file declares one we implement, null when the content isn't a
/// pooling config at all (malformed JSON, unrelated object — best-effort, like
/// generation_config.json), and `error.UnsupportedPoolingMode` when the file
/// DOES declare pooling but only modes we don't implement (cls, last token,
/// weighted-mean, max): serving those checkpoints mean-pooled would be silent
/// corruption.
pub fn parsePoolingSidecar(content: []const u8) !?PoolingMode {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), content, .{}) catch return null;
    if (parsed != .object) return null;
    const obj = parsed.object;

    const getBool = struct {
        fn get(o: std.json.ObjectMap, key: []const u8) bool {
            if (o.get(key)) |v| {
                if (v == .bool) return v.bool;
            }
            return false;
        }
    }.get;
    // ST configs set exactly one mode true; a mode other than mean is refused below.
    if (getBool(obj, "pooling_mode_lasttoken")) return error.UnsupportedPoolingMode;
    if (getBool(obj, "pooling_mode_cls_token")) return error.UnsupportedPoolingMode;
    if (getBool(obj, "pooling_mode_mean_tokens")) return .mean;
    // ST 6 names the mode in one string instead of a boolean per mode.
    if (obj.get("pooling_mode")) |v| if (v == .string) {
        if (std.mem.eql(u8, v.string, "mean")) return .mean;
        return error.UnsupportedPoolingMode;
    };
    // Declares pooling, but none we support → refuse rather than mean-pool.
    var it = obj.iterator();
    while (it.next()) |e| {
        if (std.mem.startsWith(u8, e.key_ptr.*, "pooling_mode_")) return error.UnsupportedPoolingMode;
    }
    return null;
}

pub const ModelConfig = struct {
    // Architecture identity
    model_type: []const u8 = "",
    weight_prefix: []const u8 = "language_model.model",

    // Core dimensions
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    /// Whether `intermediate_size` came from the JSON or is the struct default
    /// above. MoE checkpoints routinely omit the key,
    /// and a consumer that cannot tell the two apart bills the 15360 default as
    /// though the model declared it — 3.96 GB of phantom MLP transient in the
    /// prefill guard. Only readers that need the DISTINCTION should look here;
    /// everyone else keeps using `intermediate_size` and its fallback value.
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    rms_norm_eps: f32 = 1e-6,
    /// Prism Hadamard packs: every `<linear>.signs` weight reads `H_block(signs * x)` (rht.zig); 0 = none.
    hadamard_block: u32 = 0,

    // RoPE
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_proportional: bool = false, // Gemma 4: full attention uses proportional RoPE
    rope_proportional_factor: f32 = 1.0,

    // Sliding window attention
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,

    // Quantization. 0 = dense bf16 (config.json has no "quantization" key);
    // quantized checkpoints always set this from that key (see parseConfig).
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    // Attention scale: 1/sqrt(query_pre_attn_scalar) for Gemma, 1/sqrt(head_dim) for others
    query_pre_attn_scalar: u32 = 256,

    // Architectural differences between model families
    tie_word_embeddings: bool = false,
    hidden_act: HiddenAct = .gelu_approx,
    norm_has_offset: bool = true,
    scale_embeddings: bool = true,
    has_pre_ff_norm: bool = true,
    has_qk_norm: bool = true,
    /// SwiGLU clamp: silu(min(gate, limit)) * clip(up, ±limit) when non-zero. The fused decode gates read it
    /// (`swigluClampLimit`); 0 = none, which every served config leaves.
    swiglu_limit: f32 = 0.0,

    // MoE
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    partial_rotary_factor: f32 = 1.0,

    /// An NVIDIA ModelOpt checkpoint served as published (`modelopt_pack`).
    modelopt: bool = false,
    /// Pooling for /v1/embeddings. null = no explicit signal → masked mean.
    /// Set from config.json `pooling_mode` or the ST `1_Pooling/config.json`
    /// sidecar; only `mean` is accepted. A non-null mode advertises the
    /// `embeddings` capability; the forward stays the arch's own causal pass.
    pooling_mode: ?PoolingMode = null,

    // BOS id from config.json (embedding models wrap inputs <bos>…<eos>).
    bos_token_id: ?u32 = null,

    // Context length from config.json (0 = unknown)
    max_position_embeddings: u32 = 0,

    /// Auto-context, FROZEN at model-load time (`server.pinAutoContext`).
    /// 0 = not pinned yet.
    ///
    /// Without `--ctx-size` the effective context used to be recomputed from
    /// LIVE memory on every request, so the number the server advertised drifted
    /// as other processes took RAM (measured: 92,387–94,883 across one session).
    /// Agent CLIs budget their own `max_tokens` against that advertised value,
    /// so it has to hold still for the model's whole residency. Explicit
    /// `--ctx-size` still wins over this.
    pinned_context: u32 = 0,

    /// Per-model settings from `model-settings.json`, set at the load
    /// construction site. `ctx_override` 0 = the process `--ctx-size`/auto;
    /// `kv_quant_override` null = the process `--kv-quant`.
    ctx_override: u32 = 0,
    kv_quant_override: ?kv_quant_mod.KVQuantConfig = null,
    /// null = the process default (`MLX_SERVE_BONSAI_INT8_PREFILL`).
    int8_prefill_override: ?bool = null,
    /// LOSSY MXFP8 copy of a dense head; null = `--lm-head-mxfp8`, else off.
    lm_head_mxfp8_override: ?bool = null,
    /// Dense context K/V a loaded DFlash drafter keeps per trunk token, per request. Stamped at load.
    drafter_ctx_bytes_per_token: u64 = 0,
    /// `drafter` setting: null or "auto" = the in-dir probe, "off", or a path. Owned.
    drafter_override: ?[]const u8 = null,
    /// Every quantized projection fits the row-exact kernels (`Transformer.init`).
    row_exact_covered: bool = false,
    /// A DFlash drafter is bound (`DflashModel.bind`): its block tree needs exact verify rows.
    dflash_bound: bool = false,

    /// The prefill chunk this model was sized for, FROZEN at load
    /// (`server.pinPrefillChunk`). 0 = not pinned yet, which keeps the
    /// launch/base chunk.
    ///
    /// The chunk is the multiplier on the biggest transient in the memory bill
    /// (`8 x chunk x max(hidden, ffn) x 2`, three of them). Nothing used to size
    /// it to the MACHINE, so a 16 GB Mac reserved the same 5-7 GB envelope a
    /// 128 GB one does, which is most of its budget: the sizer then reported a
    /// 1024-token context and the admission guard refused prompts whose real
    /// peak was a third of the bill. The sizer, `checkAttentionMemory` and
    /// `generate.effectivePrefillChunk` all read THIS field, so the bill and the
    /// forward can never disagree. Explicit `--prefill-chunk` still wins.
    pinned_prefill_chunk: u32 = 0,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    // Model-author sampling recommendations from generation_config.json
    // (e.g. temp 1.0 / top_p 0.95 / top_k 20; Gemma 4: top_k 64).
    // null = the file or key is absent. Used as defaults for request fields
    // the client OMITTED — Claude Code sends no sampling params at all, and
    // pre-2026-06 it sampled the full untruncated distribution at temp 1.0,
    // well outside the model card's intended envelope.
    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,
    gen_min_p: ?f32 = null,

    // The checkpoint's OWN thinking default, from generation_config.json's
    // `default_chat_template_kwargs.enable_thinking`. null = the file or key
    // is absent. Read by `defaultEnableThinking` for requests that name no
    // thinking preference; an explicit request value still outranks it.
    gen_enable_thinking: ?bool = null,

    // Gemma 4: explicit layer type map (bit = 1 means full/global attention)
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),

    // Vision encoder (Gemma 4 SigLIP)
    has_vision: bool = false,
    vision_hidden_size: u32 = 768,
    vision_num_layers: u32 = 16,
    vision_num_heads: u32 = 12,
    vision_head_dim: u32 = 64,
    vision_intermediate_size: u32 = 3072,
    vision_patch_size: u32 = 16,
    vision_pooling_kernel: u32 = 3,
    vision_soft_tokens: u32 = 280,
    vision_position_embedding_size: u32 = 10240,
    vision_rope_theta: f32 = 100.0,
    vision_use_clipped_linears: bool = true,
    image_token_id: u32 = 0, // 0 = no image token
    boi_token_id: u32 = 0, // beginning of image
    eoi_token_id: u32 = 0, // end of image

    // Gemma 4 12B "unified" (encoder-free) multimodal. Instead of the SigLIP
    // transformer tower, vision is a single patch embedder
    // (LN → Dense → LN → +factorized 2D posemb → LN → RMSNorm → Linear) and
    // audio is raw 640-sample frames projected straight to text space. Set
    // when model_type is gemma4_unified*. See src/vision.zig (UnifiedEmbedder).
    is_gemma4_unified: bool = false,
    vision_mm_embed_dim: u32 = 0, // unified: mm_embed_dim (3840 for 12B = text hidden)
    vision_model_patch_size: u32 = 0, // unified: 48px merged "model patch" (16px teacher × 3 pool)
    vision_mm_posemb_size: u32 = 0, // unified: factorized position table size per axis (1120)
    // Audio (gemma4_unified). embed_audio projects audio_embed_dim → text hidden.
    audio_token_id: u32 = 0, // 0 = no audio token
    boa_token_id: u32 = 0, // beginning of audio
    eoa_token_id: u32 = 0, // end of audio
    audio_embed_dim: u32 = 0, // unified: raw samples per token (640)
    audio_samples_per_token: u32 = 640, // 40ms @ 16kHz

    // Video placeholder token id (top-level config.json). image_token_id reuses the
    // shared field above (parsed generically at the image_token_id block).
    video_token_id: u32 = 0,

    // Gemma 4: dual head dimensions and KV sharing
    global_head_dim: u32 = 0, // 0 = same as head_dim
    num_global_key_value_heads: u32 = 0, // 0 = same as num_key_value_heads
    num_kv_shared_layers: u32 = 0,
    final_logit_softcapping: f32 = 0.0, // 0 = disabled
    hidden_size_per_layer_input: u32 = 0, // >0 enables PLE
    partial_rotary_factor_global: f32 = 1.0, // for global/full attention layers
    has_v_norm: bool = false, // parameter-free RMS norm on values
    // Gemma 4 (31B): full_attention layers share V with K (no v_proj stored)
    attention_k_eq_v: bool = false,

    /// How many keys ONE query actually reads during prefill at this prompt
    /// length: `seq` (dense causal) for every served architecture. The prefill
    /// admission guard's score term multiplies by this.
    pub fn prefillAttnKeys(self: *const ModelConfig, seq: u64) u64 {
        _ = self;
        return seq;
    }

    pub fn firstSlidingLayer(self: ModelConfig) ?u32 {
        for (0..self.num_hidden_layers) |i| if (!self.isGlobalLayer(@intCast(i))) return @intCast(i);
        return null;
    }

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        if (self.has_explicit_layer_types and layer_idx < 128) {
            return self.layer_is_global[layer_idx];
        }
        // HF/mlx-lm sliding-window convention: the GLOBAL layer closes each group:
        // global when `(idx + 1) % pattern == 0` (layers 5, 11, … for pattern 6).
        return (layer_idx % self.sliding_window_pattern) == self.sliding_window_pattern - 1;
    }

    /// For Gemma 4 KV sharing: get the source layer index for a shared layer.
    /// Returns null if the layer computes its own KV (not shared).
    pub fn getKVSourceLayer(self: ModelConfig, layer_idx: u32) ?u32 {
        if (self.num_kv_shared_layers == 0) return null;
        const first_shared = self.num_hidden_layers - self.num_kv_shared_layers;
        if (layer_idx < first_shared) return null;
        const is_global = self.isGlobalLayer(layer_idx);
        // Find last concrete layer of the same type (scanning downward)
        var j: u32 = first_shared;
        while (j > 0) {
            j -= 1;
            if (self.isGlobalLayer(j) == is_global) return j;
        }
        return null;
    }

    /// Get effective head_dim for a layer (global layers may use global_head_dim).
    pub fn layerHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        if (self.global_head_dim > 0 and self.isGlobalLayer(layer_idx)) {
            return self.global_head_dim;
        }
        return self.head_dim;
    }

    /// Leading channels of a head that RoPE rotates: `int(head_dim * partial)`.
    pub fn layerRopeDims(self: ModelConfig, layer_idx: u32) u32 {
        const partial = if (self.isGlobalLayer(layer_idx)) self.partial_rotary_factor_global else self.partial_rotary_factor;
        return @intFromFloat(@as(f32, @floatFromInt(self.layerHeadDim(layer_idx))) * partial);
    }

    /// Get effective num_kv_heads for a layer.
    pub fn layerKVHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.num_global_key_value_heads > 0 and self.isGlobalLayer(layer_idx)) {
            return self.num_global_key_value_heads;
        }
        return self.num_key_value_heads;
    }

    /// The longest sequence the rope can resolve: the declared `max_position_embeddings`.
    /// 0 = no rope-derived cap.
    pub fn contextCap(self: *const ModelConfig) u32 {
        return self.max_position_embeddings;
    }

    /// Whether the prefill chunk is resolved per request (by the admission bill) instead of
    /// once at load. Long-context models only: a 1M session's load-time reserve pins every ordinary
    /// prompt to a narrow rung.
    pub fn perRequestPrefillChunk(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// Dense (bf16) KV-cache bytes ONE token occupies across the whole model: every layer
    /// caches, K and V are both head_dim wide. Every memory estimate that sizes a KV cache
    /// reads this one helper so the auto-context sizer and the prefill admission guard
    /// cannot disagree.
    pub fn kvBytesPerToken(self: *const ModelConfig) u64 {
        const widths: u64 = 2 * @as(u64, self.head_dim);
        const heads: u64 = @as(u64, self.num_key_value_heads);
        return @as(u64, self.num_hidden_layers) * heads * widths * 2;
    }

    /// Activation dtype: f16 for a Prism Hadamard pack (its own contract: f16
    /// activations over f16 scales), bf16 everywhere else.
    pub fn actDtype(self: *const ModelConfig) mlx.mlx_dtype {
        return if (self.hadamard_block > 0) .float16 else .bfloat16;
    }

    /// SSD prefix-cache layout marker (`kv_disk_cache.modelFingerprintWithLayout`).
    /// A layout that changed the stored keys gets a fresh root. Null keeps the arch's root.
    pub fn cacheLayoutNamespace(self: *const ModelConfig) ?[]const u8 {
        _ = self;
        return null;
    }

    /// Decode-width projections go through the row-exact kernels, whose rows
    /// get the same bits at any window width: a verify row reproduces serial
    /// decoding. Only where every projection fits them AND a DFlash drafter is
    /// bound: serial keeps the faster stock paths otherwise.
    pub fn rowExactDecode(self: *const ModelConfig) bool {
        return self.row_exact_covered and self.dflash_bound and self.rowExactArch();
    }

    /// No architecture runs the row-exact decode kernels today (only a since-removed trunk did).
    /// To serve a Gemma 4 trunk through them, return true here for the operands its kernels cover (for
    /// example `!self.isMoe() and self.hadamard_block == 0`); never relax this to those operands alone
    /// before the Gemma 4 path is wired, since they hold for dense Gemma 4.
    pub fn rowExactArch(self: *const ModelConfig) bool {
        _ = self;
        return false;
    }

    /// Vocab rows a spec drafter proposes from (0 = all): a tokenizer that
    /// puts 99.64% of committed tokens below id 98304, so a draft reads 40% of
    /// the head. A token past it is never drafted: speed only, never output.
    pub fn draftVocab(self: *const ModelConfig) c_int {
        _ = self;
        return 0;
    }

    pub fn isMoe(self: *const ModelConfig) bool {
        return self.num_experts > 0;
    }

    /// The pooling op /v1/embeddings runs: masked mean.
    pub fn effectivePooling(self: *const ModelConfig) PoolingMode {
        return self.pooling_mode orelse .mean;
    }

    /// Whether this model serves /v1/embeddings meaningfully: a decoder with a
    /// declared pooling contract. Drives capability advertising, never dispatch.
    pub fn hasEmbeddingCapability(self: *const ModelConfig) bool {
        return self.pooling_mode != null;
    }

    /// The long-context blast-radius predicate: every long-context mechanism (KV
    /// reservation, pad-waste cap, checkpoint thinning, admission terms, chunk bar) was
    /// measured on one removed family only, so it is opt-in by arch and no remaining model
    /// type opts in. Never hand-roll it at a site.
    pub fn longCtxGated(self: *const ModelConfig) bool {
        _ = self;
        return false;
    }

    /// SSD-first prefix cache arch predicate; delegates to `longCtxGated`.
    pub fn ssdFirstCapable(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers). transformer.zig's gemma4 forward/binding
    /// paths key on this rather than on the model_type string.
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4");
    }

    /// Takes the sampling recommendations + extra eos ids of a generation_config.json.
    fn applyGenerationDefaults(self: *ModelConfig, gen_content: []const u8) void {
        const gd = parseGenerationDefaultsFromJson(gen_content);
        self.gen_temperature = gd.temperature;
        self.gen_top_p = gd.top_p;
        self.gen_top_k = gd.top_k;
        self.gen_min_p = gd.min_p;
        self.gen_enable_thinking = gd.enable_thinking;
        self.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
    }

    /// Additive + dedup-guarded, like every terminator merge here.
    pub fn mergeEosTokens(self: *ModelConfig, ids: []const u32) void {
        for (ids) |id| if (!self.isEosToken(id)) self.addEosToken(id);
    }

    pub fn addEosToken(self: *ModelConfig, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    /// Gemma's chat template always ends turns with `<end_of_turn>` (id 106)
    /// and emits `<eos>` (id 1) at sequence end, so both must be stop tokens
    /// for EVERY Gemma family served (gemma4). Some
    /// checkpoints declare only a SCALAR `eos_token_id: 1` (e.g. the
    /// abliterated text-only `-lm-` builds) — gating the 106 add on
    /// `num_eos_tokens == 0` then leaves it out and it leaks into output as
    /// repeated `<end_of_turn>`. Merge both ADDITIVELY + dedup-guarded (never
    /// removes a config-declared stop). Same leak class as the
    /// `<|im_end|>` merge performed at load time (main.zig / scheduler doLoad).
    pub fn ensureGemmaTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(1)) self.addEosToken(1);
        if (!self.isEosToken(106)) self.addEosToken(106);
    }

    /// Post-attention / post-feedforward norm epsilon (one value with rms_norm_eps).
    pub fn postNormEps(self: *const ModelConfig) f32 {
        return self.rms_norm_eps;
    }

    /// SDPA softmax scale for the standard dense forward. Gemma 4's QK-norm
    /// handles normalization (scale 1.0); everything else keys on
    /// query_pre_attn_scalar.
    pub fn attnScale(self: *const ModelConfig) f32 {
        if (std.mem.eql(u8, self.model_type, "gemma4")) return 1.0;
        return 1.0 / @sqrt(@as(f32, @floatFromInt(self.query_pre_attn_scalar)));
    }

    /// The head width the PREFILL SCORE tensor is actually built at. Normally
    /// `head_dim`, but an arch can score at a different width than it stores
    /// values at (an MLA q.k can contract over nope+rope widths while
    /// `head_dim` stays the value width) — reading `head_dim` there puts such
    /// an arch under the `<= 128` "fused SDPA covers it" early-out, so the
    /// score budget that exists for exactly this materializing path never
    /// applies. A new arch scoring wider than it stores adds its arm here.
    pub fn prefillScoreHeadDim(self: *const ModelConfig) u32 {
        return self.head_dim;
    }

    /// Whether a chat request that names NO thinking preference should render
    /// with thinking on. Our server always passes `enable_thinking` explicitly,
    /// so a template whose own default is 'on' is silently overridden to off
    /// for every client that omits the field — the vendor's default mode
    /// becomes unreachable without a vendor-specific flag. An EXPLICIT request
    /// value always outranks this (see `server.resolveEnableThinking`); it
    /// only fills a silent request.
    ///
    /// First the checkpoint's OWN declaration
    /// (`generation_config.json` -> `default_chat_template_kwargs.enable_thinking`),
    /// then the per-arch allowlist below, which stays opt-in and only where the
    /// vendor documents thinking-on AND the shipped template agrees — never
    /// inferred from "the template mentions enable_thinking".
    pub fn defaultEnableThinking(self: *const ModelConfig) bool {
        // The checkpoint's own declared default outranks the arch allowlist:
        // it is the model author speaking, not our guess about the family.
        if (self.gen_enable_thinking) |v| return v;
        return false;
    }

    /// Fill still-null sampling recommendations with the FAMILY's documented
    /// upstream defaults. Community re-quants/distills routinely ship no
    /// generation_config.json (live 2026-07-13: a distilled model served
    /// to pi resolved omitted fields to the hardcoded 1.0/1.0/off — full
    /// untruncated tail sampling on a 4-bit MoE — and a 16K-token agent turn
    /// degenerated into word salad). Same lesson as a head-count
    /// default a minimal checkpoint omits: when resolution relies on per-arch
    /// defaults, fill them explicitly.
    ///
    /// Deliberately fills ONLY the truncation knobs (top_k/top_p — what keeps
    /// the tail out of the sample space), never temperature: a null temp stays
    /// the neutral 1.0, and explicit request/flag/file values always win
    /// (this runs AFTER generation_config.json parse, nulls only).
    pub fn applyFamilySamplingDefaults(self: *ModelConfig) void {
        const t = self.model_type;
        const is_gemma = std.mem.eql(u8, t, "gemma4");
        if (is_gemma) {
            if (self.gen_top_k == null) self.gen_top_k = 64;
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        }
    }

    pub fn isEosToken(self: *const ModelConfig, id: u32) bool {
        for (self.eos_token_ids[0..self.num_eos_tokens]) |eos| {
            if (id == eos) return true;
        }
        return false;
    }

    pub fn eosTokenSlice(self: *const ModelConfig) []const u32 {
        return self.eos_token_ids[0..self.num_eos_tokens];
    }

    /// Everything a load derives from the tokenizer; both load paths call it.
    pub fn applyTokenizer(
        self: *ModelConfig,
        tok: *const tokenizer_mod.Tokenizer,
        eos_token: ?[]const u8,
    ) void {
        // Always merge the chat terminator, even when config.json names an EOS:
        // Some configs say <|endoftext|> but turns end on <|im_end|>.
        if (eos_token) |eos_str| {
            if (tok.special_tokens.get(eos_str)) |eos_id| {
                if (!self.isEosToken(eos_id)) {
                    self.addEosToken(eos_id);
                    log.info("EOS token from tokenizer: {s} (id={d})\n", .{ eos_str, eos_id });
                }
            }
        }
        if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
            if (!self.isEosToken(eot_id)) self.addEosToken(eot_id);
        }
        // Id 0 can be produced spuriously under long/confusing prompts.
        if (tok.special_tokens.get("<pad>")) |pad_id| {
            if (pad_id > 0 and !self.isEosToken(pad_id)) {
                self.addEosToken(pad_id);
                log.info("Added <pad> as stop token (id={d})\n", .{pad_id});
            }
        }
    }

    /// Free the allocator-owned fields (`drafter_override`); everything else is plain data or a borrowed slice. Every
    /// `destroy` of a parsed config pairs with this. Idempotent.
    pub fn deinit(self: *ModelConfig, allocator: std.mem.Allocator) void {
        if (self.drafter_override) |p| allocator.free(p);
        self.drafter_override = null;
    }
};

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const content = blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    };
    defer allocator.free(content);

    var config = try parseConfigFromJson(allocator, content);
    errdefer config.deinit(allocator);
    // Model-author sampling recommendations ride in a sibling file. Optional —
    // any failure (missing file, bad JSON) leaves the fields null.
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/generation_config.json", .{model_dir});
    defer allocator.free(gen_path);
    if (std.Io.Dir.openFileAbsolute(io, gen_path, .{})) |gen_file| {
        defer gen_file.close(io);
        var gen_buf: [4096]u8 = undefined;
        var gen_reader = gen_file.reader(io, &gen_buf);
        if (gen_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_content| {
            defer allocator.free(gen_content);
            config.applyGenerationDefaults(gen_content);
        } else |_| {}
    } else |_| {}
    // Pooling, priority: explicit config.json `pooling_mode` (already parsed)
    // > the ST `1_Pooling/config.json` sidecar. A sidecar declaring a mode
    // other than mean fails the load here: explicitly, never a silent mean-pool.
    if (config.pooling_mode == null) {
        const pool_path = try std.fmt.allocPrint(allocator, "{s}/1_Pooling/config.json", .{model_dir});
        defer allocator.free(pool_path);
        if (std.Io.Dir.openFileAbsolute(io, pool_path, .{})) |pool_file| {
            defer pool_file.close(io);
            var pool_buf: [4096]u8 = undefined;
            var pool_reader = pool_file.reader(io, &pool_buf);
            if (pool_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |pool_content| {
                defer allocator.free(pool_content);
                config.pooling_mode = try parsePoolingSidecar(pool_content);
                if (config.pooling_mode) |m|
                    log.info("[embed] pooling from 1_Pooling/config.json: {s}\n", .{@tagName(m)});
            } else |_| {}
        } else |_| {}
    }

    // Community re-quants often ship NO generation_config.json; fill the
    // still-null truncation knobs with the family's documented defaults so
    // omitted-field resolution never bottoms out at untruncated sampling.
    config.applyFamilySamplingDefaults();

    return config;
}

/// Sampling recommendations parsed out of a model's generation_config.json.
pub const GenerationDefaults = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    min_p: ?f32 = null,
    /// `default_chat_template_kwargs.enable_thinking` — the checkpoint's own
    /// thinking default. null when absent or not a bool.
    enable_thinking: ?bool = null,
    /// `eos_token_id` (scalar or list): HF stops generation on these, and a
    /// checkpoint may name the chat terminator ONLY here (for example
    /// `<|ifm|im_end|>` rides beside config.json's `<|ifm|endoftext|>`).
    eos_token_ids: [8]u32 = @splat(0),
    num_eos: usize = 0,
};

fn positiveJsonU32(value: ?std.json.Value) ?u32 {
    const actual = value orelse return null;
    return switch (actual) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

/// Pure parser for generation_config.json content. Total: malformed JSON or
/// out-of-range values yield nulls — a corrupt config must never pin
/// sampling to an extreme. (`do_sample` is deliberately ignored: HF uses it
/// for greedy-vs-sample mode selection, which the request's own temperature
/// already expresses.)
pub fn parseGenerationDefaultsFromJson(content: []const u8) GenerationDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;

    var gd = GenerationDefaults{};
    if (root.get("temperature")) |v| {
        const t: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (t) |tv| {
            if (tv >= 0.0 and tv <= 2.0) gd.temperature = tv;
        }
    }
    if (root.get("top_p")) |v| {
        const p: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (p) |pv| {
            if (pv > 0.0 and pv <= 1.0) gd.top_p = pv;
        }
    }
    if (root.get("top_k")) |v| {
        switch (v) {
            .integer => |i| if (i > 0 and i <= 1000) {
                gd.top_k = @intCast(i);
            },
            else => {},
        }
    }
    if (root.get("min_p")) |v| {
        const m: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (m) |mv| {
            if (mv >= 0.0 and mv <= 1.0) gd.min_p = mv;
        }
    }
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |i| if (i >= 0) {
                gd.eos_token_ids[0] = @intCast(i);
                gd.num_eos = 1;
            },
            .array => |arr| for (arr.items) |item| {
                if (item == .integer and item.integer >= 0 and gd.num_eos < gd.eos_token_ids.len) {
                    gd.eos_token_ids[gd.num_eos] = @intCast(item.integer);
                    gd.num_eos += 1;
                }
            },
            else => {},
        }
    }
    // The checkpoint's own chat-template kwargs. Only a real bool counts —
    // anything else leaves the field null and the arch default in charge.
    if (root.get("default_chat_template_kwargs")) |v| {
        if (v == .object) {
            if (v.object.get("enable_thinking")) |et| {
                if (et == .bool) gd.enable_thinking = et.bool;
            }
        }
    }
    return gd;
}

/// The Hadamard rotation block of a rotated pack: the `block` its top-level `modules` array declares.
/// One block for every packed module; mixed blocks are not a layout we serve.
fn hadamardBlockFromModules(root: std.json.ObjectMap) !u32 {
    const modules = (try jsonValue(.array, root.get("modules") orelse return error.UnsupportedHadamardLayout)).items;
    var block: u32 = 0;
    for (modules) |m| {
        const module = try jsonValue(.object, m);
        const b = try jsonU32(module.get("block") orelse return error.UnsupportedHadamardLayout);
        if (b <= 0 or (block != 0 and b != block)) return error.UnsupportedHadamardLayout;
        block = b;
    }
    if (block == 0) return error.UnsupportedHadamardLayout;
    return block;
}

/// Launch-time JSON deep-merged into every `config.json` before it is parsed,
/// set once from `--config-overrides`. vLLM's `--hf-overrides` analogue: the
/// only way to re-shape a checkpoint's declared geometry — most often to scale
/// its rope and widen the context — without editing the model directory, and
/// therefore the way to A/B a scaling experiment on identical weights.
var config_overrides: ?[]const u8 = null;

pub fn setConfigOverrides(raw: ?[]const u8) void {
    config_overrides = raw;
}

pub fn getConfigOverrides() ?[]const u8 {
    return config_overrides;
}

/// Deep-merge `overrides` into a config.json document: objects merge key by key
/// — so `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4}}}`
/// keeps every sibling it passes through, exactly like vLLM's
/// `_apply_dict_overrides` — and anything else replaces. The whole merge lives
/// in an arena that dies before this returns; only the re-serialized bytes (in
/// `allocator`) escape.
fn mergeConfigJson(allocator: std.mem.Allocator, base: []const u8, overrides: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dst = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const src = try std.json.parseFromSliceLeaky(std.json.Value, a, overrides, .{});
    if (dst != .object or src != .object) return error.ConfigOverridesMustBeObject;
    try mergeObjects(a, &dst.object, src.object);
    var out: std.Io.Writer.Allocating = .init(a);
    var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try dst.jsonStringify(&jws);
    return allocator.dupe(u8, out.written());
}

fn mergeObjects(a: std.mem.Allocator, dst: *std.json.ObjectMap, src: std.json.ObjectMap) !void {
    var it = src.iterator();
    while (it.next()) |e| {
        if (dst.getPtr(e.key_ptr.*)) |p| {
            if (p.* == .object and e.value_ptr.* == .object) {
                // The handle is copied, so a rehash inside the recursion would
                // be lost — merge through the copy and store it back. `p` stays
                // valid: `dst` itself is not written during the recursion.
                var child = p.object;
                try mergeObjects(a, &child, e.value_ptr.object);
                p.* = .{ .object = child };
                continue;
            }
        }
        // Keys and values are arena-owned by the override document, which
        // outlives this merge.
        try dst.put(a, e.key_ptr.*, e.value_ptr.*);
    }
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !ModelConfig {
    // The launch-time overrides apply to EVERY parse (primary load, on-demand
    // load, discovery stubs), so the advertised context and the loaded model
    // can never disagree about what window the checkpoint has.
    const merged: ?[]const u8 = if (config_overrides) |ov| blk: {
        break :blk try mergeConfigJson(allocator, content, ov);
    } else null;
    defer if (merged) |m| allocator.free(m);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, merged orelse content, .{});
    defer parsed.deinit();

    const outer = try jsonValue(.object, parsed.value);
    const root = outer;
    var config = ModelConfig{};

    // Detect model_type from top-level (always present)
    // A config without a model_type is refused: no family arm is the default.
    const model_type: []const u8 = if (jsonField(root, "model_type")) |v| try jsonValue(.string, v) else return error.UnsupportedModelType;
    // Removed families: refused by name, never served as a Llama-shaped default.
    for ([_][]const u8{ "qwen2", "qwen3", "qwen3_moe", "qwen3_moe_text", "qwen3_next", "qwen3_asr", "qwen4", "qwen4_exp", "qwen4_exp_text", "glm5_next", "glm5_next_text", "glm_moe_dsa", "mimo_v2", "mimo_v2_flash", "flint", "nemotron_h", "nemotron_h_puzzle", "nemotron_h_omni", "lfm2", "lfm2_vl", "lfm2_moe", "minicpmv4_7", "muse_glimmer", "muse_glimmer_text", "diffusion_gemma", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "prism_hadamard_qwen35", "bailing_hybrid", "gpt_oss", "hy_v3", "laguna", "kolibri1", "inkling_mm_model", "spark2_5", "k2_horizon", "llama", "mistral", "bert", "embedding_gemma2", "gemma3", "gemma3_text" }) |removed| {
        if (std.mem.eql(u8, model_type, removed)) return error.UnsupportedModelType;
    }
    if (std.mem.startsWith(u8, model_type, "lfm2")) return error.UnsupportedModelType;

    // Determine which object to read config from: text_config (nested) or root (flat).
    const cfg_obj: std.json.ObjectMap = if (jsonField(root, "text_config")) |tc_val| try jsonValue(.object, tc_val) else root;

    // Parse common fields
    if (jsonField(cfg_obj, "vocab_size")) |v| config.vocab_size = try jsonU32(v);
    if (jsonField(cfg_obj, "hidden_size")) |v| config.hidden_size = try jsonU32(v);
    if (jsonField(cfg_obj, "intermediate_size")) |v| {
        config.intermediate_size = try jsonU32(v);
        config.intermediate_size_declared = true;
    }
    if (jsonField(cfg_obj, "num_hidden_layers")) |v| config.num_hidden_layers = try jsonU32(v);
    if (jsonField(cfg_obj, "num_attention_heads")) |v| config.num_attention_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "num_key_value_heads")) |v| config.num_key_value_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "head_dim")) |v| config.head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "max_position_embeddings")) |v| config.max_position_embeddings = try jsonU32(v);
    if (jsonField(cfg_obj, "rms_norm_eps")) |v| config.rms_norm_eps = try jsonFloat(v);
    if (jsonField(cfg_obj, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
    if (jsonField(cfg_obj, "query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = try jsonU32(v);

    // MoE fields (guard against JSON null values)
    if (jsonField(cfg_obj, "num_experts")) |v| {
        if (v == .integer) config.num_experts = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_experts_per_tok")) |v| {
        if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "top_k_experts")) |v| {
        if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "moe_intermediate_size")) |v| {
        if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "shared_expert_intermediate_size")) |v| {
        if (v == .integer) config.shared_expert_intermediate_size = try jsonU32(v);
    }

    // A Hadamard-rotated pack declares its rotation block per packed module at the top
    // level of config.json; a config without `modules` reads 0 (no rotation).
    if (jsonField(root, "modules") != null) config.hadamard_block = try hadamardBlockFromModules(root);

    // Explicit pooling contract: "mean" in config.json marks a checkpoint as an
    // embedding model. Any other value is a parse error, never a silent
    // mean-pool: wrong-semantics vectors are harder to detect than a refused load.
    if (jsonField(root, "pooling_mode")) |v| {
        if (v == .string) {
            config.pooling_mode = PoolingMode.fromString(v.string) orelse
                return error.UnsupportedPoolingMode;
        }
    }
    if (jsonField(cfg_obj, "bos_token_id")) |v| {
        if (v == .integer and v.integer >= 0) config.bos_token_id = try jsonU32(v);
    }

    // Rope parameters (nested)
    if (jsonField(cfg_obj, "rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (jsonField(rp_val.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
            if (jsonField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        }
    }

    // Sliding window
    if (cfg_obj.get("sliding_window")) |v| {
        if (v == .null) {
            config.has_sliding_window = false;
        } else {
            config.sliding_window = try jsonU32(v);
            config.has_sliding_window = true;
        }
    }
    if (jsonField(cfg_obj, "sliding_window_pattern")) |v| config.sliding_window_pattern = try jsonU32(v);

    // Gemma-specific: dual RoPE bases
    if (jsonField(cfg_obj, "rope_local_base_freq")) |v| config.rope_local_base_freq = try jsonFloat(v);
    if (jsonField(cfg_obj, "rope_scaling")) |rs_val| {
        if (rs_val == .object) {
            if (jsonField(rs_val.object, "factor")) |v| config.rope_scaling_factor = try jsonFloat(v);
        }
    }

    // Gemma 4: explicit layer_types array
    if (jsonField(cfg_obj, "layer_types")) |lt_val| {
        if (lt_val == .array) {
            config.has_explicit_layer_types = true;
            for (lt_val.array.items, 0..) |item, i| {
                if (i >= 128) break;
                if (item == .string) config.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
            }
        }
    }

    // Gemma 4: dual head dimensions and KV sharing
    if (jsonField(cfg_obj, "global_head_dim")) |v| {
        if (v == .integer) config.global_head_dim = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_global_key_value_heads")) |v| {
        if (v == .integer) config.num_global_key_value_heads = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_kv_shared_layers")) |v| {
        if (v == .integer) config.num_kv_shared_layers = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "attention_k_eq_v")) |v| {
        if (v == .bool) config.attention_k_eq_v = v.bool;
    }
    if (jsonField(cfg_obj, "final_logit_softcapping")) |v| {
        config.final_logit_softcapping = try jsonFloat(v);
    }
    if (jsonField(cfg_obj, "hidden_size_per_layer_input")) |v| {
        if (v == .integer) config.hidden_size_per_layer_input = try jsonU32(v);
    }

    // Gemma 4: nested rope_parameters with per-attention-type config
    if (jsonField(cfg_obj, "rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            // Gemma 4 style: { "full_attention": {...}, "sliding_attention": {...} }
            if (jsonField(rp_val.object, "full_attention")) |fa| {
                if (fa == .object) {
                    if (jsonField(fa.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
                    if (jsonField(fa.object, "partial_rotary_factor")) |v| config.partial_rotary_factor_global = try jsonFloat(v);
                    if (jsonField(fa.object, "rope_type")) |v| {
                        if (v == .string and std.mem.eql(u8, v.string, "proportional")) {
                            config.rope_proportional = true;
                            if (jsonField(fa.object, "factor")) |fv| config.rope_proportional_factor = try jsonFloat(fv);
                        }
                    }
                }
            }
            if (jsonField(rp_val.object, "sliding_attention")) |sa| {
                if (sa == .object) {
                    if (jsonField(sa.object, "rope_theta")) |v| config.rope_local_base_freq = try jsonFloat(v);
                }
            }
            // Nested style: { "rope_theta": ..., "partial_rotary_factor": ... }
            if (jsonField(rp_val.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
            if (jsonField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        }
    }

    // Tie word embeddings
    if (jsonField(root, "tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }
    if (jsonField(cfg_obj, "tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }

    // Check root level for max_position_embeddings (may not be in text_config)
    if (config.max_position_embeddings == 0) {
        if (jsonField(root, "max_position_embeddings")) |v| {
            if (v == .integer) config.max_position_embeddings = try jsonU32(v);
        }
    }

    // Parse quantization from top level
    if (jsonField(outer, "quantization")) |q_val| {
        const q = try jsonValue(.object, q_val);
        if (jsonField(q, "bits")) |v| config.quant_bits = try jsonU32(v);
        if (jsonField(q, "group_size")) |v| config.quant_group_size = try jsonU32(v);
        if (jsonField(q, "mode")) |v| {
            if (v == .string) {
                config.quant_mode = QuantMode.fromString(v.string) orelse {
                    log.err("unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{v.string});
                    return error.UnsupportedQuantMode;
                };
            }
        }
        // Per-weight overrides carry their own mode. An unknown one (EigenLabs' `iq2_xxs` experts) was read
        // as the top-level affine layout and failed at the FIRST request's matmul; refuse it here, by name.
        var overrides = q.iterator();
        while (overrides.next()) |kv| {
            if (kv.value_ptr.* != .object) continue;
            const mode = kv.value_ptr.object.get("mode") orelse continue;
            if (mode != .string) continue;
            if (QuantMode.fromString(mode.string) == null) {
                log.err("unsupported quantization mode '{s}' on '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{ mode.string, kv.key_ptr.* });
                return error.UnsupportedQuantMode;
            }
        }
        // MLX ships affine kernels only for bits {2,3,4,5,6,8} (ops.cpp
        // rejects the rest at quantize() time, but an ALREADY-quantized
        // checkpoint skips that check and dies at Metal kernel load during
        // warmup — an uncatchable process kill). Reject at parse instead.
        if (config.quant_mode == .affine and config.quant_bits != 0) {
            switch (config.quant_bits) {
                2, 3, 4, 5, 6, 8 => {},
                else => {
                    log.err("unsupported affine quantization: {d}-bit (this MLX runtime supports 2, 3, 4, 5, 6, 8)\n", .{config.quant_bits});
                    return error.UnsupportedQuantBits;
                },
            }
        }
    }

    if (modelopt_pack.declared(outer)) {
        config.modelopt = true;
        config.quant_bits = 4;
        config.quant_group_size = 16;
        config.quant_mode = .nvfp4;
    }

    // EOS tokens
    if (jsonField(root, "eos_token_id")) |v| {
        switch (v) {
            .integer => |i| config.addEosToken(std.math.cast(u32, i) orelse return error.InvalidConfigField),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .integer) config.addEosToken(try jsonU32(item));
                }
            },
            else => {},
        }
    }

    // Vision config (Gemma 4 SigLIP)
    if (jsonField(root, "vision_config")) |vc_val| {
        if (vc_val == .object) {
            config.has_vision = true;
            const vc = vc_val.object;
            if (jsonField(vc, "hidden_size")) |v| {
                if (v == .integer) config.vision_hidden_size = try jsonU32(v);
            }
            if (jsonField(vc, "num_hidden_layers")) |v| {
                if (v == .integer) config.vision_num_layers = try jsonU32(v);
            }
            if (jsonField(vc, "num_attention_heads")) |v| {
                if (v == .integer) config.vision_num_heads = try jsonU32(v);
            }
            if (jsonField(vc, "head_dim")) |v| {
                if (v == .integer) config.vision_head_dim = try jsonU32(v);
            }
            if (jsonField(vc, "global_head_dim")) |v| {
                if (v == .integer) config.vision_head_dim = try jsonU32(v);
            }
            if (jsonField(vc, "intermediate_size")) |v| {
                if (v == .integer) config.vision_intermediate_size = try jsonU32(v);
            }
            if (jsonField(vc, "patch_size")) |v| {
                if (v == .integer) config.vision_patch_size = try jsonU32(v);
            }
            if (jsonField(vc, "pooling_kernel_size")) |v| {
                if (v == .integer) config.vision_pooling_kernel = try jsonU32(v);
            }
            if (jsonField(vc, "default_output_length")) |v| {
                if (v == .integer) config.vision_soft_tokens = try jsonU32(v);
            }
            if (jsonField(vc, "position_embedding_size")) |v| {
                if (v == .integer) config.vision_position_embedding_size = try jsonU32(v);
            }
            if (jsonField(vc, "rope_parameters")) |rp| {
                if (rp == .object) {
                    if (jsonField(rp.object, "rope_theta")) |v| config.vision_rope_theta = try jsonFloat(v);
                }
            }
            if (jsonField(vc, "use_clipped_linears")) |v| {
                if (v == .bool) config.vision_use_clipped_linears = v.bool;
            }
            // vision_config.standardize is presence-only — the actual `std_scale`/`std_bias`
            // safetensors presence drives behavior in the vision tower loader, so the config
            // flag needs no field.

            // Gemma 4 12B unified (encoder-free) vision fields. Distinct names
            // from the SigLIP tower: mm_embed_dim (vs hidden_size),
            // model_patch_size (48px merged patch vs 16px teacher patch_size),
            // num_soft_tokens (vs default_output_length), mm_posemb_size.
            if (jsonField(vc, "mm_embed_dim")) |v| {
                if (v == .integer) config.vision_mm_embed_dim = try jsonU32(v);
            }
            if (jsonField(vc, "model_patch_size")) |v| {
                if (v == .integer) config.vision_model_patch_size = try jsonU32(v);
            }
            if (jsonField(vc, "num_soft_tokens")) |v| {
                if (v == .integer) config.vision_soft_tokens = try jsonU32(v);
            }
            if (jsonField(vc, "mm_posemb_size")) |v| {
                if (v == .integer) config.vision_mm_posemb_size = try jsonU32(v);
            }
        }
    }
    // Audio config (Gemma 4 12B unified — raw-waveform projection, no conformer)
    if (jsonField(root, "audio_config")) |ac_val| {
        if (ac_val == .object) {
            const ac = ac_val.object;
            if (jsonField(ac, "audio_embed_dim")) |v| {
                if (v == .integer) config.audio_embed_dim = try jsonU32(v);
            }
            // audio_samples_per_token lives in processor_config, not config.json;
            // default 640 (40ms @ 16kHz) matches the only shipped unified checkpoint.
            if (jsonField(ac, "audio_samples_per_token")) |v| {
                if (v == .integer) config.audio_samples_per_token = try jsonU32(v);
            }
        }
    }
    if (jsonField(root, "audio_token_id")) |v| {
        if (v == .integer) config.audio_token_id = try jsonU32(v);
    }
    if (jsonField(root, "boa_token_id")) |v| {
        if (v == .integer) config.boa_token_id = try jsonU32(v);
    }
    // eoa lives under `eoa_token_index` in the unified config.json.
    if (jsonField(root, "eoa_token_index")) |v| {
        if (v == .integer) config.eoa_token_id = try jsonU32(v);
    }
    if (jsonField(root, "eoa_token_id")) |v| {
        if (v == .integer and config.eoa_token_id == 0) config.eoa_token_id = try jsonU32(v);
    }
    // Image token ID (top-level or in mm_tokens_per_image config)
    if (jsonField(root, "image_token_id")) |v| {
        if (v == .integer) config.image_token_id = try jsonU32(v);
    }
    if (jsonField(root, "image_token_index")) |v| {
        if (v == .integer and config.image_token_id == 0) config.image_token_id = try jsonU32(v);
    }
    if (jsonField(root, "boi_token_id")) |v| {
        if (v == .integer) config.boi_token_id = try jsonU32(v);
    }
    if (jsonField(root, "eoi_token_id")) |v| {
        if (v == .integer) config.eoi_token_id = try jsonU32(v);
    }

    // Set model-family defaults based on model_type
    if (std.mem.eql(u8, model_type, "gemma4") or
        std.mem.eql(u8, model_type, "gemma4_text") or
        std.mem.eql(u8, model_type, "gemma4_unified") or
        std.mem.eql(u8, model_type, "gemma4_unified_text"))
    {
        // Per the Gemma 4 12B developer guide, `gemma4_unified` "contains the
        // same advanced decoder structure as the Gemma 4 31B Dense model" —
        // the unified-ness lives in a tiny vision/audio embedder we don't
        // wire here. So we collapse the internal tag onto plain "gemma4" so
        // every downstream model_type comparison in transformer.zig/drafter.zig
        // (attn_scale gate, recommendedBlockSize, etc.) treats it identically
        // to 31B Dense without per-arch fan-out.
        // Detect the unified (12B encoder-free multimodal) variant before we
        // collapse the tag — drives vision_embedder/embed_audio weight loading
        // and the encoder-free forward in src/vision.zig.
        config.is_gemma4_unified = std.mem.indexOf(u8, model_type, "unified") != null;
        config.model_type = "gemma4";
        config.weight_prefix = "language_model.model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false; // Gemma 4 norms have NO offset (plain weight, not 1+weight)
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true; // Parameter-free RMS norm on values
        config.rope_scaling_factor = 1.0; // No scaling, uses proportional RoPE via theta
        // [1, 106, 50] from the config array (when present) plus an additive
        // guarantee of <eos>(1) + <end_of_turn>(106) for checkpoints that
        // declare only a scalar eos_token_id.
        config.ensureGemmaTerminators();
        // Note on `attention_k_eq_v` for the 12B unified checkpoint: the
        // weights ship separate v_proj for SLIDING layers and omit them for
        // FULL_ATTENTION layers — i.e. K==V alias is a per-layer choice
        // keyed on global-layer-ness. The existing bindModelWeights logic
        // (transformer.zig:5677) already encodes that:
        //   `k_eq_v = config.attention_k_eq_v and isGlobalLayer(li)`.
        // So we leave the parsed flag intact.
    } else {
        // No other family is served: an unlisted model type is refused, never given a default.
        return error.UnsupportedModelType;
    }

    return config;
}

fn jsonValue(comptime tag: std.meta.Tag(std.json.Value), v: std.json.Value) !@TypeOf(@field(@as(std.json.Value, undefined), @tagName(tag))) {
    if (v != tag) return error.InvalidConfigField;
    return @field(v, @tagName(tag));
}

fn jsonU32(v: std.json.Value) !u32 {
    return std.math.cast(u32, try jsonValue(.integer, v)) orelse error.InvalidConfigField;
}

// Optional config nulls have the same fallback and precedence as omitted keys.
fn jsonField(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn jsonFloat64(v: std.json.Value) !f64 {
    const value: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.InvalidConfigField,
    };
    if (!std.math.isFinite(value)) return error.InvalidConfigField;
    return value;
}

fn jsonFloat(v: std.json.Value) !f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,
    /// Int8 copies of block linears that replace their packed arrays (`--w8a8`); only the media
    /// DiTs that read through a name-keyed map fill it.
    w8: @import("w8a8.zig").Overlay = .{},

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
        self.w8.deinit(self.allocator);
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Hand the map a new array under `name`, freeing the one it held (load-time
    /// weight fusion parks its row views here so the originals go away).
    pub fn replace(self: *Weights, name: []const u8, arr: mlx.mlx_array) void {
        if (self.map.getPtr(name)) |p| {
            _ = mlx.mlx_array_free(p.*);
            p.* = arr;
        }
    }

    /// Drop every entry named `prefix...` and free its array: a loader that built a block from them (its
    /// structs hold their own refs) lets the originals go before the next block loads, which is when an
    /// int8 copy and the packed weight it replaced would otherwise both be resident.
    pub fn dropPrefix(self: *Weights, prefix: []const u8) void {
        var found = true;
        while (found) {
            found = false;
            var it = self.map.iterator();
            while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, prefix)) {
                self.remove(e.key_ptr.*);
                found = true;
                break;
            };
        }
    }

    /// `dropPrefix` for the `idx`-th block of a stack, named by `fmt` (`"blocks.{d}."`).
    pub fn dropBlock(self: *Weights, comptime fmt: []const u8, idx: usize) void {
        var buf: [128]u8 = undefined;
        self.dropPrefix(std.fmt.bufPrint(&buf, fmt, .{idx}) catch return);
    }

    /// Hand the map `arr` under `name` (the key is copied), freeing an array it held there.
    pub fn put(self: *Weights, name: []const u8, arr: mlx.mlx_array) !void {
        if (self.map.contains(name)) return self.replace(name, arr);
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try self.map.put(key, arr);
    }

    /// Drop `name` and free its array (a loader that copied it elsewhere lets the original go).
    pub fn remove(self: *Weights, name: []const u8) void {
        const kv = self.map.fetchRemove(name) orelse return;
        _ = mlx.mlx_array_free(kv.value);
        self.allocator.free(kv.key);
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The generic nestings a text trunk ships under: flat, mlx-community's
/// re-nest, and meta's VL original. `parseConfigFromJson`
/// picks from config KEYS; this probe corrects it from the checkpoint.
const FLAT_PREFIX = "model";
const NESTED_PREFIX = "language_model.model";
const VL_NESTED_PREFIX = "model.language_model";

fn hasWeightsUnder(weights: *const Weights, prefix: []const u8) bool {
    var it = weights.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        if (key.len > prefix.len and key[prefix.len] == '.' and std.mem.startsWith(u8, key, prefix)) return true;
    }
    return false;
}

/// Re-point `config.weight_prefix` at the nesting the CHECKPOINT actually uses.
///
/// Which nesting a converter emits is not reliably declared in config.json,
/// so `parseConfigFromJson` guesses from `text_config` presence — wrong for any
/// checkpoint that nests without declaring one (every weight under
/// `language_model.model.*`; the guess picked `model` and the load died on
/// `MISSING WEIGHT: model.embed_tokens.weight`). The class has now shipped in
/// both directions, so the weights get the last word.
///
/// Conservative by construction: one spelling set for every trunk. A swap
/// happens only when the configured one holds NOTHING, so every checkpoint
/// that already loaded binds byte-identically. Scan order puts the most
/// specific spelling first: a `model.language_model.*` checkpoint also
/// satisfies the bare "model" probe.
/// Prism Hadamard packs run f16 activations over f16 scales (the pack's own
/// contract) and ship the unpacked tensors f32 (norms, conv, the dense gated-delta a/b
/// rows, A_log, dt_bias): an f32 table widens the f16 residual, so those
/// narrow to f16. Signs stay f32: the rotation runs in f32.
pub fn narrowHadamardPackTables(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    if (config.hadamard_block == 0) return;
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        const v = kv.value_ptr.*;
        if (mlx.mlx_array_dtype(v) != .float32) continue;
        if (std.mem.endsWith(u8, kv.key_ptr.*, ".signs")) continue;
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, v, config.actDtype(), s));
        _ = mlx.mlx_array_free(v);
        kv.value_ptr.* = cast;
    }
}

pub fn resolveWeightPrefix(config: *ModelConfig, weights: *const Weights) void {
    const candidates: []const []const u8 = &.{ NESTED_PREFIX, VL_NESTED_PREFIX, FLAT_PREFIX };
    var known = false;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) known = true;
    }
    if (!known) return;

    if (hasWeightsUnder(weights, config.weight_prefix)) return;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) continue;
        if (!hasWeightsUnder(weights, p)) continue;
        log.info("weight prefix: config implies \"{s}\", checkpoint uses \"{s}\" — using the checkpoint's\n", .{ config.weight_prefix, p });
        config.weight_prefix = p;
        return;
    }
}

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, .{});
}

/// How a load treats stored dtypes. `keep_f16`: a pack whose activation dtype
/// is f16 (Prism Hadamard packs) keeps its f16 side tensors and tables as
/// stored; narrowing them to bf16 drops 3 mantissa bits of every group scale.
pub const LoadOpts = struct { vision: bool = false, modelopt: bool = false, keep_f16: bool = false };

fn retainPart(weights: *Weights, keep: anytype) !void {
    const a = weights.allocator;
    var kept = std.StringHashMap(mlx.mlx_array).init(a);
    errdefer kept.deinit();
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        var buf: [512]u8 = undefined;
        const name = keep.remap(kv.key_ptr.*, &buf) orelse {
            _ = mlx.mlx_array_free(kv.value_ptr.*);
            a.free(kv.key_ptr.*);
            continue;
        };
        if (name.ptr == kv.key_ptr.*.ptr) {
            try kept.put(kv.key_ptr.*, kv.value_ptr.*);
        } else {
            try kept.put(try a.dupe(u8, name), kv.value_ptr.*);
            a.free(kv.key_ptr.*);
        }
    }
    weights.map.deinit();
    weights.map = kept;
}

/// The text model's weights for `config`.
pub fn loadModelWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *ModelConfig, load_vision: bool) !Weights {
    var weights = try loadWeightsOpt(io, allocator, model_dir, .{ .vision = load_vision, .keep_f16 = config.actDtype() == .float16, .modelopt = config.modelopt });
    errdefer weights.deinit();
    if (config.modelopt) try modelopt_pack.adapt(&weights, model_dir);
    return weights;
}

/// Load ONE safetensors file (absolute path) into a Weights map — for
/// sidecar files that live beside the trunk shards (e.g. a root-level
/// a draft-head sidecar file), where a directory scan would sweep in the trunk.
pub fn loadWeightsSingleFile(allocator: std.mem.Allocator, abs_path: []const u8) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);

    const pathz = try allocator.dupeSentinel(u8, abs_path, 0);
    defer allocator.free(pathz);
    try loadSafetensorsFile(allocator, &weights, pathz, s, .{});

    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} — corrupt or empty safetensors file?\n", .{abs_path});
        return error.NoWeightFiles;
    }
    return weights;
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, .{ .vision = true });
}

fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDir(io, allocator, dir, model_dir, opts);
}

/// Load every `*.safetensors` in an already-open `dir` into a Weights map.
/// `model_dir` is the on-disk path string, used both to build the per-file
/// absolute path for `mlx_load_safetensors` and to phrase the error message.
/// Split out of `loadWeightsOpt` so the incomplete-checkpoint guard below is
/// unit-testable against a `tmpDir` (mirrors `model_discovery.discoverModelsInDir`).
fn loadWeightsFromOpenDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, opts: LoadOpts) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);

    // The index names the shards; anything else is dead weight (issue #274:
    // a pack shipped two shards no weight_map entry names — RAM for nothing)
    // or a foreign file whose parse failure would be an uncatchable MLX abort.
    var referenced = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Accept regular files AND symlinks: HuggingFace cache snapshots store
        // every weight file as a symlink into ../../blobs/<hash>. mlx_load_safetensors
        // resolves the link at the OS level, so a symlinked *.safetensors loads fine.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) {
            log.warn("skipping {s}: not named by model.safetensors.index.json\n", .{entry.name});
            continue;
        };

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeSentinel(u8, path_slice, 0);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        try loadSafetensorsFile(allocator, &weights, path, s, opts);
        file_count += 1;
    }
    // A ModelOpt index may name shards in a subdirectory (NVIDIA's `tensors/`); the scan above sees the root only.
    if (opts.modelopt) if (referenced) |r| {
        var kit = r.keyIterator();
        while (kit.next()) |name| {
            if (std.mem.indexOfScalar(u8, name.*, '/') == null or std.mem.indexOf(u8, name.*, "..") != null) continue;
            const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, name.* }, 0);
            defer allocator.free(path);
            log.info("Loading {s}...\n", .{name.*});
            try loadSafetensorsFile(allocator, &weights, path, s, opts);
            file_count += 1;
        }
    };

    // Incomplete-checkpoint guard. A dir with config/tokenizer but no (or no
    // usable) *.safetensors is the classic interrupted-download shape: the
    // small files land first, the multi-GB weight shards never finalize. Before
    // this guard the loader returned an empty map and the caller crashed with a
    // misleading `MISSING WEIGHT: <prefix>.embed_tokens.weight` (the first
    // weight looked up) + `unreachable`, pointing at the model arch instead of
    // the download. Fail here with an actionable message, mirroring the
    // tokenizer path's "incomplete download?" hint (see main.zig).
    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} ({d} *.safetensors file(s) found): the checkpoint looks like an incomplete download (config/tokenizer present, weight shards missing). Re-download the model or delete the dir and re-fetch.\n", .{ model_dir, file_count });
        return error.NoWeightFiles;
    }

    try viewByteAffineWeights(&weights, s);
    log.info("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
    reportF16Narrowing();
    return weights;
}

/// An affine weight (`.scales` AND `.biases` beside it) stored as uint8 is MLX's packed
/// uint32 words written byte-wise (Recursive's export); MLX's quantized kernels read
/// uint32 only. NVFP4/MXFP8/FP8 bytes carry no `.biases` and keep their dtype.
fn viewByteAffineWeights(weights: *Weights, s: mlx.mlx_stream) !void {
    var buf: [512]u8 = undefined;
    var it = weights.map.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (!std.mem.endsWith(u8, key, ".weight") or mlx.mlx_array_dtype(e.value_ptr.*) != .uint8) continue;
        const shape = mlx.getShape(e.value_ptr.*);
        if (shape.len == 0 or @rem(shape[shape.len - 1], 4) != 0) continue;
        const base = key[0 .. key.len - ".weight".len];
        const scales = std.fmt.bufPrint(&buf, "{s}.scales", .{base}) catch continue;
        if (weights.map.get(scales) == null) continue;
        const biases = std.fmt.bufPrint(&buf, "{s}.biases", .{base}) catch continue;
        if (weights.map.get(biases) == null) continue;
        var words = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_view(&words, e.value_ptr.*, .uint32, s));
        _ = mlx.mlx_array_free(e.value_ptr.*);
        e.value_ptr.* = words;
    }
}

/// Whether a just-loaded f16 tensor must be narrowed to the engine's bf16
/// activation dtype.
///
/// Two shapes qualify, for the same underlying reason — an f16 value that
/// meets a bf16 activation promotes the RESULT to f32:
///
///   - Quant SIDE tensors (scales/biases), which can be 2-D so they are keyed
///     on the suffix. f16 side tensors force gather_qmm/qmatmul onto a ~4x
///     slower mixed-dtype path (2-bit live, 2026-07-14: 0.70 vs 0.18 ms
///     per 8-expert gather — 1.2 tok/s on the 295B instead of ~15+).
///   - ANY 1-D f16 tensor: a per-channel table (norm weight, bias, A_log,
///     dt_bias) that is multiplied or added straight into the residual. Leave
///     one f16 and the residual turns f32 at the first layer and STAYS f32,
///     so every later weight read is upcast - the YaRN-mscale class,
///     one level up. Measured on prism-ml/Ternary-Bonsai-27B-mlx-2bit (the
///     only f16 checkpoint on hand, a gated-delta hybrid): 27.99 -> 23.88
///     ms/forward, 14.7%, three paired boots with cooldown.
///
/// Plain multi-dimensional WEIGHTS keep their dtype. They are matmul
/// OPERANDS, and MLX selects its kernel off that dtype, so narrowing one is a
/// kernel-selection change rather than a promotion fix — measured as a wash
/// here (23.15 vs 23.47 ms, inside boot-to-boot drift), so the minimal rule
/// is the one that ships.
///
/// The cast node stays lazy, so the load-time batch eval materializes bf16
/// directly. Delta from the 3 dropped mantissa bits: cos 0.99999994 — far
/// below any quant noise floor.
pub fn narrowsLoadedF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16) return false;
    if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) return true;
    return ndim == 1;
}

/// An f32 layer-norm weight beside bf16 activations (NVIDIA's ModelOpt pack stores
/// `post_attention_layernorm` f32; the loader narrows it for ModelOpt packs only, the one
/// reference measured): `rms_norm` promotes to the weight's dtype, so the MoE reads f32
/// and the fused hyper-connection read declines. Narrowed like the f16 norms; the f32 per-channel
/// state tables (`A_log`, `dt_bias`, `hc_*`, `e_score_correction_bias`) are not norms and stay f32, and a
/// vision tower keeps the f32 its reference runs in.
pub fn narrowsLoadedF32Norm(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    return dtype == .float32 and ndim == 1 and std.mem.indexOf(u8, key, "language_model.") != null and
        (std.mem.endsWith(u8, key, "layernorm.weight") or std.mem.endsWith(u8, key, "language_model.norm.weight"));
}

/// Kill switch for the 1-D arm (`MLX_SERVE_F16_NARROW_1D=0`). A load-time
/// dtype normalization is invisible once the model is up, so a one-boot A/B
/// switch is the only way to attribute a future f16-checkpoint regression to
/// it. The side-tensor arm predates this and is not switchable.
var narrow_1d_env: ?bool = null;

/// ON for Metal: an f16 norm left beside bf16 activations widens the whole residual to f32.
fn narrow1dDefault(darwin: bool) bool {
    return darwin;
}

test "f16 1-D tables narrow to bf16 on Metal only" {
    try std.testing.expect(narrow1dDefault(true));
    try std.testing.expect(!narrow1dDefault(false));
}
fn narrow1dEnabled() bool {
    if (narrow_1d_env) |v| return v;
    const on = blk: {
        const raw = std.c.getenv("MLX_SERVE_F16_NARROW_1D") orelse
            break :blk narrow1dDefault(builtin.os.tag.isDarwin());
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    narrow_1d_env = on;
    return on;
}

/// Count of 1-D f16 tables narrowed this load — reported once per model so a
/// declined normalization is nameable from the log instead of silently
/// reading as "this checkpoint just isn't f16".
var narrowed_1d: usize = 0;

pub fn reportF16Narrowing() void {
    if (narrowed_1d == 0) return;
    log.info("[dtype] narrowed {d} 1-D f16 tables to bf16 (MLX_SERVE_F16_NARROW_1D=0 disables)\n", .{narrowed_1d});
    narrowed_1d = 0;
}

/// Load ONE safetensors file into a Weights map, tensors as stored (lazy), on
/// `mlx.weightLoadStream()` (Metal has no GPU `Load`). The iterator hands a +1
/// reference in `value`; it transfers straight into the map.
pub fn loadWeightsFile(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = allocator.dupe(u8, std.mem.span(key.?)) catch |e| {
            _ = mlx.mlx_array_free(value);
            return e;
        };
        w.map.put(owned_key, value) catch |e| {
            allocator.free(owned_key);
            _ = mlx.mlx_array_free(value);
            return e;
        };
    }
    log.info("[weights] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

pub fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    opts: LoadOpts,
) !void {
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);

    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();

        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }

        const key_str = std.mem.span(key.?);

        if (!keepsTensor(key_str, opts)) {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        // Read the shape BEFORE the cast frees `value` — a freed handle's
        // ndim is a use-after-free, not a zero.
        const ndim = mlx.mlx_array_ndim(value);
        var final_value = value;
        if (!opts.keep_f16 and (narrowsLoadedF16(key_str, ndim, mlx.mlx_array_dtype(value)) or (opts.modelopt and narrowsLoadedF32Norm(key_str, ndim, mlx.mlx_array_dtype(value)))) and
            (ndim != 1 or narrow1dEnabled()))
        {
            var cast = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            _ = mlx.mlx_array_free(value);
            final_value = cast;
            if (ndim == 1) narrowed_1d += 1;
        }

        const owned_key = try allocator.dupe(u8, key_str);
        try weights.map.put(owned_key, final_value);
    }
}

/// `shouldKeepWeightKey`, plus the ModelOpt skip list.
fn keepsTensor(key: []const u8, opts: LoadOpts) bool {
    if (opts.modelopt and modelopt_pack.skipsAtLoad(key)) return false;
    return shouldKeepWeightKey(key, opts.vision);
}

/// True if the safetensors weight `key` should be retained for the text
/// forward pass. Audio is always dropped; vision is dropped unless
/// `load_vision` is set. Draft-head tensors (`*.mtp.*`) on some
/// checkpoints are kept (the binder ignores them, but the loader doesn't
/// need to know that).
pub fn shouldKeepWeightKey(key: []const u8, load_vision: bool) bool {
    // Gemma 4 12B `gemma4_unified` is encoder-free: it ships a tiny vision
    // patch embedder (`vision_embedder.*` + `embed_vision.*`) and a raw-waveform
    // audio projection (`embed_audio.*`) instead of the SigLIP vision tower and
    // conformer audio tower of earlier Gemma 4 variants. Those embedders are
    // wired in src/vision.zig (UnifiedEmbedder), so keep them under the same
    // `load_vision` gate as the SigLIP weights (`load_vision` is false → text only).
    const is_vision = std.mem.startsWith(u8, key, "vision_tower.") or
        std.mem.startsWith(u8, key, "vision_model.") or
        std.mem.startsWith(u8, key, "mlp1.") or
        std.mem.startsWith(u8, key, "embed_vision.") or
        std.mem.startsWith(u8, key, "merger.") or
        std.mem.startsWith(u8, key, "vit_merger.") or
        std.mem.startsWith(u8, key, "vision_embedder.") or
        std.mem.startsWith(u8, key, "embed_audio.") or
        std.mem.startsWith(u8, key, "multi_modal_projector.") or
        std.mem.startsWith(u8, key, "language_model.multi_modal_projector.");
    // The heavy SigLIP-era conformer audio tower is still not wired — drop it.
    const is_audio_tower = std.mem.startsWith(u8, key, "audio_tower.") or
        std.mem.startsWith(u8, key, "language_model.audio_multi_modal_projector.");
    if (is_audio_tower) return false;
    // A tower nested under "model." follows the same gate as the bare
    // "vision_tower." that rides the is_vision gate above.
    if (!load_vision and (std.mem.startsWith(u8, key, "model.vision_tower.") or
        // avlp12's "Alis" packs spell the vision tower
        // `model.visual.` (pure rename of `vision_tower.`).
        std.mem.startsWith(u8, key, "model.visual."))) return false;
    if (is_vision and !load_vision) return false;
    return true;
}

// ── Tests ──

const testing = std.testing;

test "ModelConfig defaults" {
    const config = ModelConfig{};
    try testing.expectEqual(@as(u32, 0), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 0), config.max_position_embeddings);
    try testing.expectEqual(@as(u32, 0), config.quant_bits); // 0 = dense bf16 (no "quantization" key)
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expect(!config.tie_word_embeddings);
}

test "loadWeights casts f16 quant scales/biases to bf16 (mixed-dtype qmm slow-path class)" {
    // A 2-bit (ox-ox) pack ships F16 scales/biases beside bf16 activations -
    // MLX's gather_qmm/qmatmul take a ~4x slower mixed-dtype path (measured
    // 2026-07-14: 0.70 vs 0.18 ms per 8-expert gather; 1.2 tok/s on the 295B
    // instead of ~15+). The loader must cast quant SIDE tensors to bf16 once;
    // weights and non-quant tensors keep their dtype. Dequant delta from the
    // 3 dropped mantissa bits: cos 0.99999994 — under the 2-bit noise floor.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);

    // Build a tiny map: an f16 "scales", an f16 "biases", an f16 plain weight
    // (must NOT be cast), and a bf16 scales (no-op).
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const shape = [_]c_int{ 4, 4 };
        const data: [16]f32 = @splat(0.5);
        const f32_arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var f16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f16_arr);
        try mlx.check(mlx.mlx_astype(&f16_arr, f32_arr, .float16, s));
        var bf16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_arr);
        try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(f16_arr));
        try mlx.check(mlx.mlx_array_eval(bf16_arr));

        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.scales", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.biases", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.up_proj.weight", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.down_proj.scales", bf16_arr);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var weights = try loadWeights(io, allocator, dir_path);
    defer weights.deinit();

    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.scales").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.biases").?));
    // A plain WEIGHT stays f16 (dense-f16 tables are legitimate — only the
    // quant side tensors force the mixed-dtype qmm path).
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.up_proj.weight").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.down_proj.scales").?));
}

test "loadWeights reads an affine weight packed as uint8 (a vendor export) as MLX's uint32 words" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        var bytes: [16]u8 = undefined;
        for (&bytes, 0..) |*b, i| b.* = @intCast(i);
        const packed_u8 = mlx.mlx_array_new_data(&bytes, &[_]c_int{ 2, 8 }, 2, .uint8);
        defer _ = mlx.mlx_array_free(packed_u8);
        const side_data: [2]f32 = .{ 1, 2 };
        const side_f32 = mlx.mlx_array_new_data(&side_data, &[_]c_int{ 2, 1 }, 2, .float32);
        defer _ = mlx.mlx_array_free(side_f32);
        var side = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(side);
        try mlx.check(mlx.mlx_astype(&side, side_f32, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(side));
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.self_attn.q_proj.weight", packed_u8);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.self_attn.q_proj.scales", side);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.self_attn.q_proj.biases", side);
        // No biases: a byte format of its own (NVFP4, FP8), never reinterpreted.
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.up_proj.weight", packed_u8);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.up_proj.scales", side);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }
    var weights = try loadWeights(io, allocator, dir_path);
    defer weights.deinit();

    const w = weights.get("model.layers.0.self_attn.q_proj.weight").?;
    try mlx.check(mlx.mlx_array_eval(w));
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(w));
    try testing.expectEqualSlices(c_int, &.{ 2, 2 }, mlx.getShape(w));
    const words = mlx.mlx_array_data_uint32(w).?[0..4];
    try testing.expectEqual(@as(u32, 0x03020100), words[0]);
    try testing.expectEqual(@as(u32, 0x0f0e0d0c), words[3]);
    try testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.up_proj.weight").?));
}

test "loadWeights on a weightless dir (incomplete download) errors clearly, not empty map" {
    // Reproduces the live misdiagnosis: an interrupted `hf download`/`mlx-serve
    // pull` lands config + tokenizer but never finalizes the *.safetensors
    // weight shards. Before the guard, loadWeights returned an empty map and
    // the caller crashed with a misleading "MISSING WEIGHT:
    // model.embed_tokens.weight" (the first weight looked up) + `unreachable`,
    // pointing at the model arch instead of the incomplete checkpoint.
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"gemma4\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    // The index file names the shards but is NOT itself a weight file — it must
    // not be mistaken for one (it ends in .json, not .safetensors).
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{}" });

    try std.testing.expectError(
        error.NoWeightFiles,
        loadWeightsFromOpenDir(io, allocator, tmp.dir, "/incomplete-model", .{}),
    );
}

test "loadWeights reads only the shards the index names (issue #274)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // One real (hand-built) shard + one garbage file that mlx would abort on.
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var st: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "stray.safetensors", .data = "not a safetensors file" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-00001.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, .{});
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 1), w.count());
}

test "loadWeights ignores an index that names no shard on disk (re-sharded upload, stale index)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var st: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-00001-of-00005.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, .{});
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 1), w.count());
}

test "resolveWeightPrefix: the CHECKPOINT decides the nesting, not the config keys" {
    // A checkpoint that declares NO text_config yet ships every weight under
    // `language_model.model.*`: the config-key guess picked "model" and the
    // load died on `MISSING WEIGHT: model.embed_tokens.weight`. The same class
    // shipped in the opposite direction before, so the probe corrects either way.
    const allocator = testing.allocator;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // Nested checkpoint, flat guess → re-pointed.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.layers.0.self_attn.q_proj.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Flat checkpoint, nested guess → re-pointed the other way.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model", config.weight_prefix);
    }
    // Both present (a real VL checkpoint) → the configured prefix stands, so
    // nothing that loads today can be re-pointed by this probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // A prefix that is a strict PREFIX of the key's first segment must not
    // count as a hit ("model" vs "model_extra.*").
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model_extra.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // A mlx-community pack (live 2026-08-11): meta's config
    // keeps text_config, so the guess is the VL-original "model.language_model"
    // — but mlx_lm convert re-nests every text weight under
    // "language_model.model.*". The third spelling joins the probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.lm_head.weight");
        try put(&w, allocator, "vision_tower.layers.0.norm1.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Our own mirror layout (meta-original nesting) stays put.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
    // Ordering: a "model.language_model.*" checkpoint ALSO matches the bare
    // "model" probe (the '.' check passes at "model.language_model"), so the
    // most specific spelling must win the scan.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "gemma4", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
}

test "applyFamilySamplingDefaults: gemma4 gets top_k 64 / top_p 0.95 when the checkpoint ships no generation_config" {
    // A community re-quant ships no generation_config.json; omitted-field sampling would
    // bottom out at untruncated tail sampling. Fill exactly the truncation knobs, never temperature.
    var gemma = ModelConfig{ .model_type = "gemma4" };
    gemma.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 64), gemma.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.95), gemma.gen_top_p);
    try testing.expectEqual(@as(?f32, null), gemma.gen_temperature);

    // A type with no documented upstream recommendation stays null.
    var other = ModelConfig{ .model_type = "unlisted" };
    other.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, null), other.gen_top_k);
    try testing.expectEqual(@as(?f32, null), other.gen_top_p);
}

test "applyFamilySamplingDefaults never overrides explicit generation_config values" {
    // The checkpoint's own generation_config.json (parsed before this runs)
    // always wins — the family fallback fills NULLS only.
    var config = ModelConfig{ .model_type = "gemma4" };
    config.gen_top_k = 40;
    config.gen_top_p = 0.8;
    config.gen_temperature = 0.6;
    config.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 40), config.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.8), config.gen_top_p);
    try testing.expectEqual(@as(?f32, 0.6), config.gen_temperature);
    // Partial file: only the missing knob is filled.
    var partial = ModelConfig{ .model_type = "gemma4" };
    partial.gen_top_p = 0.8;
    partial.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 64), partial.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.8), partial.gen_top_p);
}

test "defaultEnableThinking: no served arch opts in" {
    // Gemma 4's template mentions enable_thinking, which is not evidence of a thinking-on default.
    const c = ModelConfig{ .model_type = "gemma4" };
    try testing.expect(!c.defaultEnableThinking());
}

test "defaultEnableThinking: the checkpoint's own generation_config default outranks the arch allowlist" {
    // A thinking model whose arch is not on the allowlist still thinks when
    // its own generation_config declares it — this is the case a silent
    // request used to lose (3 tokens and no reasoning where the same weights
    // reason for ~1000 tokens under a runner that obeys the template).
    var on = ModelConfig{ .model_type = "gemma4" };
    on.gen_enable_thinking = true;
    try testing.expect(on.defaultEnableThinking());
    try testing.expect(on.defaultEnableThinking());
    // And a checkpoint that declares thinking OFF stays off.
    var off = ModelConfig{ .model_type = "gemma4" };
    off.gen_enable_thinking = false;
    try testing.expect(!off.defaultEnableThinking());
}

test "parseGenerationDefaultsFromJson: reads default_chat_template_kwargs.enable_thinking" {
    const on = parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": true}}",
    );
    try testing.expectEqual(@as(?bool, true), on.enable_thinking);
    const off = parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": false}}",
    );
    try testing.expectEqual(@as(?bool, false), off.enable_thinking);
    // Absent, wrong shape, or a non-bool value: null, arch default stays.
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson("{\"top_k\": 20}").enable_thinking);
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": \"on\"}",
    ).enable_thinking);
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": \"yes\"}}",
    ).enable_thinking);
}

test "parseGenerationDefaultsFromJson: eos_token_id list merges additively into the stop set" {
    const gd = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 250019]}");
    try testing.expectEqual(@as(usize, 2), gd.num_eos);
    var config = ModelConfig{};
    config.addEosToken(1);
    config.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(250019));
    const scalar = parseGenerationDefaultsFromJson("{\"eos_token_id\": 7}");
    try testing.expectEqual(@as(u32, 7), scalar.eos_token_ids[0]);
    try testing.expectEqual(@as(usize, 0), parseGenerationDefaultsFromJson("{\"eos_token_id\": \"x\"}").num_eos);
}

test "ModelConfig addEosToken" {
    var config = ModelConfig{};
    config.addEosToken(1);
    config.addEosToken(106);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(1));
    try testing.expect(config.isEosToken(106));
    try testing.expect(!config.isEosToken(42));
}

test "ModelConfig addEosToken max capacity" {
    var config = ModelConfig{};
    // Fill all 8 slots
    for (0..8) |i| {
        config.addEosToken(@intCast(i + 100));
    }
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    // 9th should be silently dropped
    config.addEosToken(999);
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    try testing.expect(!config.isEosToken(999));
}

test "EOS merge: chat-terminator added even when config already provided an eos" {
    // Regression for the eos leak: its config.json sets
    // eos_token_id=<|endoftext|> (151643), but its chat template ends turns
    // with <|im_end|> (151645). The load path (main.zig / scheduler doLoad)
    // must ALWAYS merge the tokenizer's chat-terminator EOS — additively and
    // dedup-guarded — not only when config provided none; otherwise <|im_end|>
    // is never a stop token and leaks into the output (broke structured JSON /
    // tool calling). This pins the merge invariant those call sites implement.
    var config = ModelConfig{};
    config.addEosToken(151643); // from config.json eos_token_id
    try testing.expectEqual(@as(u32, 1), config.num_eos_tokens);

    // Merge step the fix performs: add the chat terminator if absent.
    const chat_eos: u32 = 151645; // <|im_end|>, from tokenizer_config eos_token
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);

    try testing.expect(config.isEosToken(151645)); // now stops on <|im_end|>
    try testing.expect(config.isEosToken(151643)); // original preserved
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);

    // Idempotent: re-running the merge must not duplicate.
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
}

test "ModelConfig eosTokenSlice" {
    var config = ModelConfig{};
    config.addEosToken(10);
    config.addEosToken(20);
    const slice = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 2), slice.len);
    try testing.expectEqual(@as(u32, 10), slice[0]);
    try testing.expectEqual(@as(u32, 20), slice[1]);
}

test "ModelConfig isGlobalLayer with sliding window" {
    // Sliding-window convention (HF + mlx-lm): every Nth layer is global, with the
    // pattern anchored at the END of each group — global when
    // `(idx + 1) % pattern == 0`, i.e. layers 5, 11, 17… for pattern 6.
    // The old `% pattern == 0` phase made layer 0 global and layer 5 local —
    // every layer got the wrong RoPE base/scale and attention scope, which
    // surfaced as fluent-but-wrong output (spaced digits, broken arithmetic)
    // on a 12B dense model. Gemma 4 ships explicit layer_types and never hits this
    // fallback.
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.sliding_window_pattern = 6;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(!config.isGlobalLayer(6));
    try testing.expect(config.isGlobalLayer(11));
    try testing.expect(!config.isGlobalLayer(12));
}

test "ModelConfig isGlobalLayer without sliding window" {
    var config = ModelConfig{};
    config.has_sliding_window = false;
    // All layers should be global
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
}

test "ModelConfig isMoe" {
    var config = ModelConfig{};
    try testing.expect(!config.isMoe());
    config.num_experts = 8;
    try testing.expect(config.isMoe());
}

test "jsonFloat converts integer" {
    const val = std.json.Value{ .integer = 42 };
    try testing.expectApproxEqAbs(@as(f32, 42.0), try jsonFloat(val), 0.001);
}

test "jsonFloat converts float" {
    const val = std.json.Value{ .float = 3.14 };
    try testing.expectApproxEqAbs(@as(f32, 3.14), try jsonFloat(val), 0.01);
}

test "ModelConfig isGlobalLayer with explicit layer_types" {
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // Set layer 4 and 9 as global (like Gemma 4 E2B pattern)
    config.layer_is_global[4] = true;
    config.layer_is_global[9] = true;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(3));
    try testing.expect(config.isGlobalLayer(4));
    try testing.expect(!config.isGlobalLayer(5));
    try testing.expect(config.isGlobalLayer(9));
}

test "ModelConfig getKVSourceLayer" {
    var config = ModelConfig{};
    config.num_hidden_layers = 35;
    config.num_kv_shared_layers = 20;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // E2B pattern: every 5th layer starting from 4 is global
    for (0..35) |i| {
        config.layer_is_global[i] = (i % 5 == 4);
    }
    // Layers 0-14 are concrete (no source)
    try testing.expect(config.getKVSourceLayer(0) == null);
    try testing.expect(config.getKVSourceLayer(14) == null);
    // Layer 15 (sliding) -> should map to layer 13 (last concrete sliding)
    try testing.expectEqual(@as(?u32, 13), config.getKVSourceLayer(15));
    // Layer 19 (full) -> should map to layer 14 (last concrete full)
    try testing.expectEqual(@as(?u32, 14), config.getKVSourceLayer(19));
    // Layer 20 (sliding) -> should also map to layer 13
    try testing.expectEqual(@as(?u32, 13), config.getKVSourceLayer(20));
}

test "ModelConfig layerHeadDim" {
    var config = ModelConfig{};
    config.head_dim = 256;
    config.global_head_dim = 512;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    config.layer_is_global[4] = true;
    try testing.expectEqual(@as(u32, 256), config.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 512), config.layerHeadDim(4));
}

test "ModelConfig without a sliding window: every layer is global" {
    var config = ModelConfig{};
    config.has_sliding_window = false;
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(5));
}

test "shouldKeepWeightKey accepts orphan draft-head weights on hybrid-trunk checkpoints" {
    // Some checkpoints embed `*.mtp.*` tensors in the MAIN
    // shards (the engine no longer loads them).
    // The safetensors iterator must let them through (they're neither vision
    // nor audio) so the model loads cleanly; the trunk binder ignores them.
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.mtp.0.shared_head.head.weight", false));
}

test "shouldKeepWeightKey filters audio and gated vision weights" {
    // Regression: the existing filter should still reject audio and reject
    // vision when load_vision is false.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
    try testing.expect(!shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", false));
    // Alis packs spell the vision tower `model.visual.`; `load_vision` false drops it too.
    try testing.expect(!shouldKeepWeightKey("model.visual.blocks.0.attn.qkv.weight", false));
    try testing.expect(shouldKeepWeightKey("model.visual.blocks.0.attn.qkv.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.layers.0.self_attn.q_proj.weight", false));
}

test "shouldKeepWeightKey keeps Gemma 4 12B unified embedder weights when vision enabled" {
    // gemma4_unified is encoder-free: vision_embedder.* (patch embedder),
    // embed_vision.* and embed_audio.* (raw projections) ARE wired in
    // src/vision.zig (UnifiedEmbedder), so they must be kept under load_vision.
    // The heavy SigLIP-era conformer audio_tower.* stays dropped.
    try testing.expect(shouldKeepWeightKey("vision_embedder.patch_dense.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_vision.embedding_projection.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_audio.embedding_projection.weight", true));
    // Gated off while `load_vision` is false.
    try testing.expect(!shouldKeepWeightKey("vision_embedder.patch_dense.weight", false));
    try testing.expect(!shouldKeepWeightKey("embed_audio.embedding_projection.weight", false));
    // The conformer audio tower is never wired — always dropped.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
}

test "ModelConfig parses gemma4_unified text_config" {
    // Gemma 4 12B base ships `model_type: gemma4_unified` with text_config
    // carrying the language tower. The dispatch arm must:
    //   - tag as gemma4_unified
    //   - inherit the gemma4 weight prefix + norm/scale flags
    //   - pass attention_k_eq_v through unchanged so the per-layer binder
    //     (transformer.zig:5677) aliases V to K on global layers but uses
    //     the shipped v_proj on sliding layers
    //   - pass through gemma4 fields like global_head_dim, final_logit_softcapping.
    const json =
        \\{
        \\  "model_type": "gemma4_unified",
        \\  "text_config": {
        \\    "model_type": "gemma4_unified_text",
        \\    "hidden_size": 3840,
        \\    "intermediate_size": 15360,
        \\    "num_hidden_layers": 48,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 256,
        \\    "global_head_dim": 512,
        \\    "num_global_key_value_heads": 8,
        \\    "num_kv_shared_layers": 0,
        \\    "hidden_size_per_layer_input": 0,
        \\    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        \\    "rope_parameters": {
        \\      "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "factor": 1.0},
        \\      "sliding_attention": {"rope_theta": 10000.0}
        \\    },
        \\    "attention_k_eq_v": true,
        \\    "final_logit_softcapping": 30.0,
        \\    "hidden_activation": "gelu_pytorch_tanh",
        \\    "rms_norm_eps": 1e-06,
        \\    "max_position_embeddings": 8192,
        \\    "sliding_window": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    // Collapsed onto "gemma4" so downstream code paths (attn_scale gate,
    // recommendedBlockSize) match the 31B Dense decoder it inherits.
    try testing.expectEqualStrings("gemma4", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.has_v_norm);
    try testing.expect(config.has_pre_ff_norm);
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(config.scale_embeddings);
    // 12B's k_proj/v_proj layout is mixed: full_attention layers omit v_proj
    // (K=V alias), sliding layers ship it. The existing per-layer binder
    // (transformer.zig:5677) keys the alias on isGlobalLayer; we must keep
    // the config flag intact so that AND-clause fires on global layers only.
    try testing.expect(config.attention_k_eq_v);
    try testing.expectEqual(@as(u32, 512), config.global_head_dim);
    try testing.expectEqual(@as(u32, 0), config.hidden_size_per_layer_input);
    try testing.expectApproxEqAbs(@as(f32, 30.0), config.final_logit_softcapping, 0.001);
    try testing.expectEqual(@as(u32, 3840), config.hidden_size);
    try testing.expectEqual(@as(u32, 48), config.num_hidden_layers);
    // Unified flag drives the encoder-free vision/audio embedder path.
    try testing.expect(config.is_gemma4_unified);
}

test "loadWeightsOpt: an f32 trunk norm narrows to bf16 for a ModelOpt pack, any other keeps its dtype" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const name = "model.language_model.layers.0.post_attention_layernorm.weight";
    const hdr = "{\"" ++ name ++ "\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    var shard: [8 + hdr.len + 8]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &shard });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    for ([_]bool{ true, false }) |modelopt| {
        var w = try loadWeightsOpt(io, testing.allocator, dir, .{ .modelopt = modelopt });
        defer w.deinit();
        const want: mlx.mlx_dtype = if (modelopt and narrow1dEnabled()) .bfloat16 else .float32;
        try testing.expectEqual(want, mlx.mlx_array_dtype(w.get(name).?));
    }
}

test "prefillAttnKeys: every architecture bills the whole prompt" {
    // The admission guard's score term asks how many keys one query reads during
    // prefill: dense causal attention reads the whole prompt.
    var dense = ModelConfig{};
    dense.model_type = "gemma4";
    try testing.expectEqual(@as(u64, 100_000), dense.prefillAttnKeys(100_000));
    var gemma = ModelConfig{};
    gemma.model_type = "gemma4";
    gemma.sliding_window = 128;
    try testing.expectEqual(@as(u64, 100_000), gemma.prefillAttnKeys(100_000));
}

test "ensureGemmaTerminators is additive and dedup-guarded" {
    var c = ModelConfig{};
    c.addEosToken(1); // config-provided scalar eos
    c.ensureGemmaTerminators();
    try testing.expect(c.isEosToken(1));
    try testing.expect(c.isEosToken(106));
    try testing.expectEqual(@as(u32, 2), c.num_eos_tokens);
    // Idempotent: re-running adds nothing.
    c.ensureGemmaTerminators();
    try testing.expectEqual(@as(u32, 2), c.num_eos_tokens);
}

test "ModelConfig parses gemma4_unified vision + audio multimodal fields" {
    // The 12B unified config.json carries top-level vision_config/audio_config
    // with encoder-free dims plus image/audio/boi/boa/eoi/eoa token ids. These
    // drive the UnifiedEmbedder forward and placeholder insertion.
    const json =
        \\{
        \\  "model_type": "gemma4_unified",
        \\  "image_token_id": 258880,
        \\  "audio_token_id": 258881,
        \\  "boi_token_id": 255999,
        \\  "eoi_token_id": 258882,
        \\  "boa_token_id": 256000,
        \\  "eoa_token_index": 258883,
        \\  "vision_config": {
        \\    "model_type": "gemma4_unified_vision",
        \\    "mm_embed_dim": 3840,
        \\    "mm_posemb_size": 1120,
        \\    "model_patch_size": 48,
        \\    "patch_size": 16,
        \\    "pooling_kernel_size": 3,
        \\    "num_soft_tokens": 280,
        \\    "output_proj_dims": 3840,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "audio_config": {
        \\    "model_type": "gemma4_unified_audio",
        \\    "audio_embed_dim": 640,
        \\    "output_proj_dims": 640,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "text_config": {
        \\    "hidden_size": 3840,
        \\    "num_hidden_layers": 48,
        \\    "head_dim": 256
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.is_gemma4_unified);
    try testing.expect(config.has_vision);
    // Vision (encoder-free) dims.
    try testing.expectEqual(@as(u32, 3840), config.vision_mm_embed_dim);
    try testing.expectEqual(@as(u32, 48), config.vision_model_patch_size);
    try testing.expectEqual(@as(u32, 1120), config.vision_mm_posemb_size);
    try testing.expectEqual(@as(u32, 280), config.vision_soft_tokens);
    try testing.expectEqual(@as(u32, 3), config.vision_pooling_kernel);
    // Audio.
    try testing.expectEqual(@as(u32, 640), config.audio_embed_dim);
    // Multimodal token ids.
    try testing.expectEqual(@as(u32, 258880), config.image_token_id);
    try testing.expectEqual(@as(u32, 258881), config.audio_token_id);
    try testing.expectEqual(@as(u32, 255999), config.boi_token_id);
    try testing.expectEqual(@as(u32, 258882), config.eoi_token_id);
    try testing.expectEqual(@as(u32, 256000), config.boa_token_id);
    try testing.expectEqual(@as(u32, 258883), config.eoa_token_id);
}

test "parseConfigFromJson dense bf16 gemma4 MoE → quant_bits 0" {
    // A fully-dense bf16 checkpoint has NO
    // "quantization" key. quant_bits must stay 0 so the loader skips every
    // .scales/.biases fetch and the forward pass dispatches to plain matmul.
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "text_config": {
        \\    "hidden_size": 2048,
        \\    "head_dim": 256,
        \\    "num_hidden_layers": 40,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 2,
        \\    "num_experts": 256,
        \\    "num_experts_per_tok": 8,
        \\    "moe_intermediate_size": 512,
        \\    "tie_word_embeddings": false
        \\  }
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 0), config.quant_bits);
    try testing.expectEqualStrings("gemma4", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 256), config.num_experts);
}

test "parseConfigFromJson quantized gemma4 MoE → quant_bits from key" {
    // Same arch but with a "quantization" block: quant_bits must reflect it so
    // the mandatory scale/bias fetches still fire (a missing scale is a clear
    // MISSING WEIGHT error, not a silent dense fallback). Guards the default flip.
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "text_config": {"hidden_size": 2048, "num_experts": 256},
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
}

test "parseConfigFromJson rejects affine bits MLX has no kernels for" {
    // A checkpoint declaring an affine bit-width outside MLX's kernel set
    // ({2,3,4,5,6,8}) must fail at PARSE, not at warmup: mlx only validates
    // bits inside quantize(), so an already-quantized 1-bit checkpoint sails
    // through load and dies with an uncatchable Metal kernel-load error
    // ("Unable to load kernel affine_dequantize_..._b_1") that kills the
    // whole server. Live bite: prism-ml/Bonsai-27B-mlx-1bit.
    const json_1bit =
        \\{
        \\  "model_type": "gemma4",
        \\  "text_config": {"hidden_size": 5120},
        \\  "quantization": {"bits": 1, "group_size": 128}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantBits, parseConfigFromJson(testing.allocator, json_1bit));

    const json_7bit =
        \\{
        \\  "model_type": "gemma4",
        \\  "hidden_size": 1024,
        \\  "quantization": {"bits": 7, "group_size": 64}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantBits, parseConfigFromJson(testing.allocator, json_7bit));
}

test "parseConfigFromJson nvfp4 quantization mode" {
    // NVFP4 checkpoints (issue #24): {"group_size": 16, "bits": 4, "mode": "nvfp4"}.
    // The mode must land on config.quant_mode so the loader skips the .biases
    // fetches (nvfp4 stores no biases tensors) and the matmul call sites pass
    // "nvfp4" to mlx instead of "affine".
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.nvfp4, config.quant_mode);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 16), config.quant_group_size);
    try testing.expect(!config.quant_mode.hasBiases());
}

test "parseConfigFromJson explicit affine mode keeps biases" {
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
    try testing.expect(config.quant_mode.hasBiases());
}

test "parseConfigFromJson refuses an unknown PER-WEIGHT quantization mode by name" {
    // A pack with affine at the top and `iq2_xxs` experts underneath. Read as affine the
    // first request's matmul failed; the load refuses instead.
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 32, "bits": 8, "mode": "affine",
        \\    "layers.0.mlp.switch_mlp.fc1": {"group_size": 256, "bits": 2, "mode": "iq2_xxs"}}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantMode, parseConfigFromJson(testing.allocator, json));
    // A known per-weight mode beside the top-level one still parses.
    const ok =
        \\{
        \\  "model_type": "gemma4",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 32, "bits": 8, "mode": "affine",
        \\    "layers.0.mlp.switch_mlp.fc1": {"group_size": 16, "bits": 4, "mode": "nvfp4"}}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, ok);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
}

test "parseConfigFromJson unknown quantization mode → error" {
    // An unrecognized mode must fail loudly at config parse — not crash later
    // in the weight loader with a misleading MISSING WEIGHT error.
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 32, "bits": 4, "mode": "fp99"}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantMode, parseConfigFromJson(testing.allocator, json));
}

test "narrowsLoadedF16 catches per-channel tables, not matmul operands" {
    // Quant side tensors: the pre-existing rule, keyed on the suffix because
    // they can be 2-D.
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .float16));
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.biases", 2, .float16));

    // Any 1-D f16 tensor is a PER-CHANNEL table — a norm weight, a bias, a
    // gate table. It gets multiplied or added straight into the activation
    // stream, so leaving it f16 beside a bf16 residual promotes the residual
    // (and therefore every later weight read) to f32.
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.input_layernorm.weight", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.A_log", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.dt_bias", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.norm.weight", 1, .float16));

    // A 2-D dense f16 weight is a MATMUL OPERAND, not a table. MLX picks its
    // kernel off that dtype, so narrowing it is a kernel-selection change and
    // not this rule's business — it stays per-site.
    try testing.expect(!narrowsLoadedF16("vision_tower.blocks.0.attn.qkv.weight", 2, .float16));
    try testing.expect(!narrowsLoadedF16("language_model.model.layers.0.linear_attn.conv1d.weight", 3, .float16));

    // Everything already in the engine's dtype, and packed weights, are left
    // alone.
    try testing.expect(!narrowsLoadedF16("model.layers.0.input_layernorm.weight", 1, .bfloat16));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.weight", 2, .uint32));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .bfloat16));
}

test "narrowsLoadedF32Norm takes f32 layer norms, not f32 state tables or other-dtype norms" {
    try testing.expect(narrowsLoadedF32Norm("model.language_model.layers.3.post_attention_layernorm.weight", 1, .float32));
    try testing.expect(narrowsLoadedF32Norm("model.language_model.layers.3.input_layernorm.weight", 1, .float32));
    try testing.expect(narrowsLoadedF32Norm("model.language_model.norm.weight", 1, .float32));
    // Stay f32 by design: hyper-connection norm and tables, recurrent-state constants, routing bias.
    try testing.expect(!narrowsLoadedF32Norm("model.language_model.layers.3.attn_hyper_connection.hc_norm.weight", 1, .float32));
    try testing.expect(!narrowsLoadedF32Norm("model.language_model.layers.3.hc_ffn_base", 1, .float32));
    try testing.expect(!narrowsLoadedF32Norm("model.language_model.layers.3.self_attn.A_log", 1, .float32));
    try testing.expect(!narrowsLoadedF32Norm("model.language_model.layers.3.mlp.gate.e_score_correction_bias", 1, .float32));
    // A vision tower runs in its reference's f32: only the language trunk's norms narrow.
    try testing.expect(!narrowsLoadedF32Norm("vision_tower.layer_norm.weight", 1, .float32));
    // Already the engine's dtype, or not a vector.
    try testing.expect(!narrowsLoadedF32Norm("model.language_model.layers.3.post_attention_layernorm.weight", 1, .bfloat16));
    try testing.expect(!narrowsLoadedF32Norm("model.language_model.layers.3.post_attention_layernorm.weight", 2, .float32));
}

test "parseGenerationDefaultsFromJson: reads model sampling recommendations" {
    // Verbatim shape of Gemma 4 checkpoints' generation_config.json.
    const json =
        \\{"bos_token_id": 248044, "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    ;
    const gd = parseGenerationDefaultsFromJson(json);
    try testing.expectEqual(@as(?f32, 1.0), gd.temperature);
    try testing.expectEqual(@as(?f32, 0.95), gd.top_p);
    try testing.expectEqual(@as(?u32, 20), gd.top_k);
}

test "parseGenerationDefaultsFromJson: min_p reads in range, rejects out of range" {
    try testing.expectEqual(@as(?f32, 0.05), parseGenerationDefaultsFromJson("{\"min_p\":0.05}").min_p);
    try testing.expectEqual(@as(?f32, 0), parseGenerationDefaultsFromJson("{\"min_p\":0}").min_p);
    try testing.expectEqual(@as(?f32, null), parseGenerationDefaultsFromJson("{\"min_p\":-0.1}").min_p);
    try testing.expectEqual(@as(?f32, null), parseGenerationDefaultsFromJson("{\"min_p\":1.5}").min_p);
}

test "pooling: config.json pooling_mode key parses mean; cls, last_token and unknown values are rejected at parse" {
    // Explicit converter/operator contract for checkpoints whose config alone
    // can't reveal pooling (a plain decoder model type).
    const base = "{{\"model_type\":\"gemma4\",\"hidden_size\":64,\"num_attention_heads\":8,\"num_hidden_layers\":2,\"pooling_mode\":\"{s}\"}}";
    const json = try std.fmt.allocPrint(testing.allocator, base, .{"mean"});
    defer testing.allocator.free(json);
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(?PoolingMode, .mean), config.pooling_mode);
    try testing.expect(config.hasEmbeddingCapability());
    // A mode other than mean is a parse error, not a silent mean-pool: wrong-semantics
    // vectors are harder to detect than a refused load.
    inline for (.{ "cls", "last_token", "weighted_mean" }) |mode| {
        const bad = try std.fmt.allocPrint(testing.allocator, base, .{mode});
        defer testing.allocator.free(bad);
        try testing.expectError(error.UnsupportedPoolingMode, parseConfigFromJson(testing.allocator, bad));
    }
}

test "pooling: sentence-transformers 1_Pooling sidecar parses mean and refuses cls and last token" {
    // Verbatim shape of ST `1_Pooling/config.json` (one model sets
    // lasttoken, bge/mxbai set cls_token, MiniLM sets mean_tokens).
    const last =
        \\{"word_embedding_dimension": 2560, "pooling_mode_cls_token": false,
        \\ "pooling_mode_mean_tokens": false, "pooling_mode_max_tokens": false,
        \\ "pooling_mode_mean_sqrt_len_tokens": false, "pooling_mode_lasttoken": true}
    ;
    try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(last));
    const cls =
        \\{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false, "pooling_mode_lasttoken": false}
    ;
    try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(cls));
    const mean =
        \\{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .mean), try parsePoolingSidecar(mean));
}

test "pooling: a sentence-transformers 6 sidecar spells the mode as a string" {
    // The string spelling of ST 6; only mean is served.
    const mean =
        \\{"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .mean), try parsePoolingSidecar(mean));
    // Modes we do not implement refuse the load, as the boolean spelling does.
    for ([_][]const u8{ "cls", "lasttoken", "max", "mean_sqrt_len", "weightedmean" }) |mode| {
        var buf: [64]u8 = undefined;
        const json = try std.fmt.bufPrint(&buf, "{{\"pooling_mode\": \"{s}\"}}", .{mode});
        try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(json));
    }
}

test "pooling: sidecar demanding an unsupported mode errors; non-pooling JSON is ignored" {
    // A sidecar that DOES declare pooling but none we implement (weighted-mean,
    // max) must refuse the load - mean-pooling it anyway is silent corruption.
    const unsupported =
        \\{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": false,
        \\ "pooling_mode_max_tokens": true, "pooling_mode_lasttoken": false}
    ;
    try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(unsupported));
    // Malformed / unrelated JSON: best-effort null, like generation_config.json.
    try testing.expectEqual(@as(?PoolingMode, null), try parsePoolingSidecar("not json"));
    try testing.expectEqual(@as(?PoolingMode, null), try parsePoolingSidecar("{\"dimension\": 384}"));
}

test "pooling: effectivePooling is mean; the embeddings capability follows pooling_mode" {
    var config = ModelConfig{};
    try testing.expectEqual(PoolingMode.mean, config.effectivePooling());
    try testing.expect(!config.hasEmbeddingCapability());
    config.pooling_mode = .mean;
    try testing.expectEqual(PoolingMode.mean, config.effectivePooling());
    try testing.expect(config.hasEmbeddingCapability());
}

test "parseGenerationDefaultsFromJson: missing keys and malformed input give nulls" {
    const partial = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 2]}");
    try testing.expectEqual(@as(?f32, null), partial.temperature);
    try testing.expectEqual(@as(?f32, null), partial.top_p);
    try testing.expectEqual(@as(?u32, null), partial.top_k);

    const broken = parseGenerationDefaultsFromJson("not json at all");
    try testing.expectEqual(@as(?f32, null), broken.temperature);

    // Out-of-range values are dropped, not clamped — a corrupt config must
    // not silently pin sampling to an extreme.
    const insane = parseGenerationDefaultsFromJson("{\"temperature\": 99.0, \"top_p\": 7.0, \"top_k\": -5}");
    try testing.expectEqual(@as(?f32, null), insane.temperature);
    try testing.expectEqual(@as(?f32, null), insane.top_p);
    try testing.expectEqual(@as(?u32, null), insane.top_k);
}

test "parseConfigFromJson refuses the removed model type qwen2 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen2\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_moe by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_moe\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_next by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_next\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_asr by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_asr\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen4 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen4\"}"));
}

test "parseConfigFromJson refuses the removed model type glm5_next by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"glm5_next\"}"));
}

test "parseConfigFromJson refuses the removed model type glm5_next_text by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"glm5_next_text\"}"));
}

test "parseConfigFromJson refuses the removed model type glm_moe_dsa by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"glm_moe_dsa\"}"));
}

test "parseConfigFromJson refuses the removed model type mimo_v2 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"mimo_v2\"}"));
}

test "parseConfigFromJson refuses the removed model type mimo_v2_flash by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"mimo_v2_flash\"}"));
}

test "parseConfigFromJson refuses the removed model type flint by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"flint\"}"));
}

test "parseConfigFromJson refuses the removed model type nemotron_h by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"nemotron_h\"}"));
}

test "parseConfigFromJson refuses the removed model type nemotron_h_puzzle by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"nemotron_h_puzzle\"}"));
}

test "parseConfigFromJson refuses the removed model type nemotron_h_omni by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"nemotron_h_omni\"}"));
}

test "parseConfigFromJson refuses the removed model type lfm2 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"lfm2\"}"));
}

test "parseConfigFromJson refuses the removed model type lfm2_vl by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"lfm2_vl\"}"));
}

test "parseConfigFromJson refuses the removed model type lfm2_moe by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"lfm2_moe\"}"));
}

test "parseConfigFromJson refuses the removed model type lfm2_extra by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"lfm2_extra\"}"));
}

test "parseConfigFromJson refuses the removed model type minicpmv4_7 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"minicpmv4_7\"}"));
}

test "parseConfigFromJson refuses the removed model type muse_glimmer by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"muse_glimmer\"}"));
}

test "parseConfigFromJson refuses the removed model type muse_glimmer_text by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"muse_glimmer_text\"}"));
}

test "parseConfigFromJson refuses the removed model type diffusion_gemma by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"diffusion_gemma\"}"));
}

test "parseConfigFromJson refuses the removed model type bert by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"bert\"}"));
}

test "parseConfigFromJson refuses the removed model type embedding_gemma2 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"embedding_gemma2\"}"));
}

test "parseConfigFromJson refuses the removed model type gemma3 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"gemma3\"}"));
}

test "parseConfigFromJson refuses the removed model type gemma3_text by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"gemma3_text\"}"));
}

test "parseConfigFromJson refuses a config without a model_type, absent or null" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{}"));
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":null,\"hidden_size\":64}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_5 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_5\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_5_text by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_5_text\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_5_moe by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_5_moe\"}"));
}

test "parseConfigFromJson refuses the removed model type qwen3_5_moe_text by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"qwen3_5_moe_text\"}"));
}

test "parseConfigFromJson refuses the removed model type prism_hadamard_qwen35 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"prism_hadamard_qwen35\"}"));
}

test "parseConfigFromJson refuses the removed model type gpt_oss by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"gpt_oss\"}"));
}

test "parseConfigFromJson refuses the removed model type hy_v3 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"hy_v3\"}"));
}

test "parseConfigFromJson refuses the removed model type laguna by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"laguna\"}"));
}

test "parseConfigFromJson refuses the removed model type kolibri1 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"kolibri1\"}"));
}

test "parseConfigFromJson refuses the removed model type inkling_mm_model by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"inkling_mm_model\"}"));
}

test "parseConfigFromJson refuses the removed model type spark2_5 by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"spark2_5\"}"));
}

test "parseConfigFromJson refuses the removed model type k2_horizon by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"k2_horizon\"}"));
}

test "parseConfigFromJson refuses the removed model type llama by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"llama\"}"));
}

test "parseConfigFromJson refuses the removed model type mistral by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"mistral\"}"));
}

test "parseConfigFromJson refuses an unlisted model type" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"not_a_listed_type\"}"));
}

test "parseConfigFromJson refuses the removed model type bailing_hybrid by name" {
    try testing.expectError(error.UnsupportedModelType, parseConfigFromJson(testing.allocator, "{\"model_type\":\"bailing_hybrid\"}"));
}

test "ModelConfig gemma4 reads the Hadamard block of a rotated pack from its modules, and 0 without them" {
    const json =
        \\{
        \\  "model_type": "gemma4",
        \\  "text_config": {"hidden_size": 2816, "num_hidden_layers": 30},
        \\  "modules": [{"path": "lm_head", "block": 1024}, {"path": "model.layers.0.mlp.up_proj", "block": 1024}],
        \\  "quantization": {"bits": 2, "group_size": 128, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("gemma4", config.model_type);
    try testing.expectEqual(@as(u32, 1024), config.hadamard_block);
    const mixed =
        \\{"model_type": "gemma4", "text_config": {"hidden_size": 2816},
        \\ "modules": [{"path": "a", "block": 1024}, {"path": "b", "block": 512}]}
    ;
    try testing.expectError(error.UnsupportedHadamardLayout, parseConfigFromJson(testing.allocator, mixed));
    const plain = try parseConfigFromJson(testing.allocator, "{\"model_type\":\"gemma4\",\"text_config\":{\"hidden_size\":2816}}");
    try testing.expectEqual(@as(u32, 0), plain.hadamard_block);
}

test "rowExactDecode: no architecture decodes row-exact today, whatever the drafter and coverage flags" {
    var cfg = ModelConfig{};
    cfg.model_type = "gemma4";
    cfg.row_exact_covered = true;
    cfg.dflash_bound = true;
    try std.testing.expect(!cfg.rowExactArch());
    try std.testing.expect(!cfg.rowExactDecode());
    cfg.hadamard_block = 0;
    cfg.num_experts = 0;
    try std.testing.expect(!cfg.rowExactDecode());
}

test "parseConfigFromJson rejects invalid field types and ranges" {
    const bad = [_][]const u8{
        "[]",
        "{\"model_type\":3}",
        "{\"model_type\":\"gemma4\",\"text_config\":[]}",
        "{\"model_type\":\"gemma4\",\"hidden_size\":\"4096\"}",
        "{\"model_type\":\"gemma4\",\"num_hidden_layers\":1.5}",
        "{\"model_type\":\"gemma4\",\"vocab_size\":4294967296}",
        "{\"model_type\":\"gemma4\",\"sliding_window\":\"bad\"}",
        "{\"model_type\":\"gemma4\",\"num_experts\":-1}",
        "{\"model_type\":\"gemma4\",\"eos_token_id\":-1}",
        "{\"model_type\":\"gemma4\",\"eos_token_id\":[2,4294967296]}",
        "{\"model_type\":\"gemma4\",\"rope_theta\":1e100}",
        "{\"model_type\":\"gemma4\",\"quantization\":[]}",
        "{\"model_type\":\"gemma4\",\"quantization\":{\"bits\":\"8\"}}",
        "{\"model_type\":\"gemma4\",\"vision_config\":{\"hidden_size\":-1}}",
        "{\"model_type\":\"gemma4\",\"modules\":{}}",
        "{\"model_type\":\"gemma4\",\"modules\":[{\"block\":4294967296}]}",
    };
    for (bad) |content| {
        if (parseConfigFromJson(testing.allocator, content)) |parsed_config| {
            var config = parsed_config;
            defer config.deinit(testing.allocator);
            std.debug.print("accepted invalid config: {s}\n", .{content});
            return error.TestExpectedError;
        } else |err| try testing.expectEqual(error.InvalidConfigField, err);
    }

    const valid = try parseConfigFromJson(testing.allocator,
        \\{"model_type":"gemma4","hidden_size":4294967295,"num_experts":null,
        \\ "sliding_window":null,"eos_token_id":[0,4294967295],"rope_theta":10000,
        \\ "final_logit_softcapping":null,"unknown_vendor_field":{"future":true}}
    );
    try testing.expectEqual(std.math.maxInt(u32), valid.hidden_size);
    try testing.expectEqual(@as(u32, 0), valid.num_experts);
    try testing.expect(!valid.has_sliding_window);
    try testing.expectEqual(std.math.maxInt(u32), valid.eos_token_ids[1]);
    try testing.expectEqual(@as(f32, 10000), valid.rope_theta);
    try testing.expectEqual(@as(f32, 0), valid.final_logit_softcapping);
}

test "parseConfigFromJson preserves optional nulls and skipped fields" {
    const cases = [_][2][]const u8{
        .{ "{\"model_type\":\"gemma4\"}", "{\"model_type\":\"gemma4\",\"head_dim\":null,\"rope_theta\":null,\"hidden_size\":null,\"intermediate_size\":null}" },
        .{ "{\"model_type\":\"gemma4\"}", "{\"model_type\":\"gemma4\",\"bos_token_id\":-1}" },
        .{ "{\"model_type\":\"gemma4\"}", "{\"model_type\":\"gemma4\",\"text_config\":null,\"quantization\":null,\"linear_num_key_heads\":null}" },
        .{ "{\"model_type\":\"gemma4\"}", "{\"model_type\":\"gemma4\",\"num_experts\":\"unused\",\"attn_output_gate\":1,\"pooling_mode\":false}" },
        .{ "{\"model_type\":\"gemma4\",\"eos_token_id\":[2]}", "{\"model_type\":\"gemma4\",\"eos_token_id\":[2,\"unused\",null]}" },
        .{ "{\"model_type\":\"gemma4\"}", "{\"model_type\":\"gemma4\",\"partial_rotary_factor\":null,\"rope_parameters\":{\"rope_theta\":null,\"partial_rotary_factor\":null}}" },
        .{ "{\"model_type\":\"gemma4\",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":1024,\"attn_factor\":2}}", "{\"model_type\":\"gemma4\",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":1024,\"factor\":null,\"attention_factor\":null,\"attn_factor\":2,\"beta_fast\":null,\"beta_slow\":null}}" },
    };
    for (cases) |pair| {
        var expected = try parseConfigFromJson(testing.allocator, pair[0]);
        defer expected.deinit(testing.allocator);
        var actual = try parseConfigFromJson(testing.allocator, pair[1]);
        defer actual.deinit(testing.allocator);
        try testing.expectEqualDeep(expected, actual);
    }
}

test "parseConfigFromJson accepts real checkpoint configs" {
    const names = .{ "gemma4-text", "gemma4-unified-text" };
    var failed = false;
    inline for (names) |name| {
        if (parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/" ++ name ++ ".json"))) |parsed_config| {
            var config = parsed_config;
            defer config.deinit(testing.allocator);
            try testing.expectEqual(@as(f32, 1e6), config.rope_theta);
        } else |err| {
            std.debug.print("{s}: {s}\n", .{ name, @errorName(err) });
            failed = true;
        }
    }
    try testing.expect(!failed);
}

test "Weights.dropPrefix drops one block's tensors and nothing that merely starts like it" {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    for ([_][]const u8{ "blocks.1.attn.q.weight", "blocks.1.attn.q.scales", "blocks.10.attn.q.weight", "blocks.2.attn.q.weight", "norm.weight" }) |k|
        try w.map.put(try std.testing.allocator.dupe(u8, k), mlx.mlx_array_new_float(1.0));
    w.dropPrefix("blocks.1.");
    try std.testing.expectEqual(@as(u32, 3), w.count());
    try std.testing.expect(w.get("blocks.1.attn.q.weight") == null and w.get("blocks.1.attn.q.scales") == null);
    try std.testing.expect(w.get("blocks.10.attn.q.weight") != null and w.get("blocks.2.attn.q.weight") != null and w.get("norm.weight") != null);
    w.dropPrefix("absent.");
    try std.testing.expectEqual(@as(u32, 3), w.count());
}
