const std = @import("std");
const build_options = @import("build_options");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const round_cost_mod = @import("round_cost.zig");
const generate_mod = @import("generate.zig");
const model_discovery = @import("model_discovery.zig");
const model_registry_mod = @import("model_registry.zig");
const drafter_mod = @import("drafter.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");
const scheduler_mod = @import("scheduler.zig");
const status = @import("status.zig");
const model_settings_mod = @import("model_settings.zig");
const cli_mod = @import("cli.zig");

// codesign names an executable after its embedded Info.plist; without one it makes up a new name for
// every build.
const info_plist = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" ++
    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n" ++
    "<plist version=\"1.0\"><dict><key>CFBundleIdentifier</key><string>com.lestoengine.lesto</string>" ++
    "<key>CFBundleName</key><string>lesto</string></dict></plist>\n").*;
comptime {
    if (@import("builtin").os.tag == .macos) @export(&info_plist, .{ .name = "mlx_serve_info_plist", .section = "__TEXT,__info_plist" });
}
pub const log = @import("log.zig");
const metrics_mod = @import("metrics.zig");
const sleep_inhibit_mod = @import("sleep_inhibit.zig");
const version_mod = @import("version.zig");
const sys_info = @import("sys_info.zig");

pub const VERSION: []const u8 = build_options.version;
const DEFAULT_MODEL_DIR = ""; // pass --model <path> to specify

// Serve-mode default for requests that omit max_tokens (0 = flag not given).
var serve_default_max_tokens: u32 = 0;

fn printUsage(io: std.Io) void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    stdout_w.interface.writeAll(
        \\lesto - inference server for Gemma 4 on Apple Silicon
        \\
        \\Usage: lesto <command> [options]
        \\       lesto [options]
        \\
        \\Commands:
        \\  serve               Start the server over ~/.mlx-serve/models
        \\                      (every model there loads on demand by name)
        \\
        \\Options:
        \\  --model <dir>       Path to MLX model directory
        \\  --serve             Start HTTP server mode (the default unless --prompt)
        \\  --host <ip>         Bind address (default: 0.0.0.0 - open to the local
        \\                      network; a future version will default to 127.0.0.1)
        \\  --port <n>          Bind port (default: 11234)
        \\  --ctx-size <n>      Maximum context length (default: model max)
        \\  --config-overrides <json>   JSON object deep-merged into EVERY
        \\                      model's config.json this process loads or
        \\                      discovers (alias: --hf-overrides). Generic keys
        \\                      like max_position_embeddings hit everything.
        \\                      HF attention_factor replaces the computed YaRN
        \\                      mscale; vLLM attn_factor multiplies it.
        \\                      e.g. '{"text_config":{"rope_parameters":{"rope_type":
        \\                      "yarn","factor":4.0,"original_max_position_embeddings":
        \\                      262144},"max_position_embeddings":1048576}}'
        \\  --embedding-max-length <n>  Per-input token ceiling for /v1/embeddings
        \\                      (default auto = the model's declared window; over-limit
        \\                      inputs get a 400 naming index/count/limit, never truncation)
        \\  --prompt <text>     Run one prompt offline and exit (no server)
        \\  --stream            Stream tokens as they are generated (with --prompt)
        \\  --max-tokens <n>    Max tokens to generate (default: 100); in --serve
        \\                      mode, the default for requests that omit the field
        \\  --temp <f>          Temperature. Offline: sampling temp (default 0.0).
        \\                      Serve: default for requests that omit `temperature`
        \\                      (otherwise the model's generation_config.json, then 1.0)
        \\  --top-p <f>         Serve-mode default top_p for requests that omit it
        \\                      (otherwise generation_config.json, then 1.0 = off)
        \\  --top-k <n>         Serve-mode default top_k for requests that omit it
        \\                      (otherwise generation_config.json, then 0 = off)
        \\  --timeout <n>       Stall timeout in seconds: abort a request after n seconds
        \\                      WITHOUT producing a token (default: 300, 0=none). A request
        \\                      that keeps generating never times out, however long it runs.
        \\  --reasoning-budget <n>  Max thinking tokens per request (default: unlimited)
        \\  --no-prevent-sleep  Allow Mac idle sleep during inference and model
        \\                      loads. Display sleep is always allowed.
        \\  --os-reserve-gib <n>  Free RAM left out of every memory plan so macOS keeps
        \\                        room (default: an eighth of RAM, 2 to 8 GB). 0 turns
        \\                        it off: more context and concurrency, but a small Mac
        \\                        under heavy load can freeze or restart.
        \\  --skip-mem-preflight  Bypass the model-load free-RAM pre-flight that
        \\                        refuses a load whose weights + warmup headroom
        \\                        look too big for current free memory. The check
        \\                        is conservative (macOS reclaims file cache as
        \\                        MLX allocates); use this if a load you know fits
        \\                        is being refused. A genuine over-commit can
        \\                        hard-crash the server.
        \\  --pld               Enable Prompt Lookup Decoding (default: ON).
        \\                        Model-agnostic speculative decoding via n-gram
        \\                        matches in the prompt + generated tokens. Big
        \\                        wins on echo-heavy workloads (code editing, RAG,
        \\                        agentic loops). Adaptive prompt-time gate
        \\                        auto-disables it on novel content. Pass
        \\                        --no-pld to force-disable.
        \\  --no-pld            Force-disable Prompt Lookup Decoding.
        \\  --pld-draft-len <n> Max draft tokens per PLD step (default: 5).
        \\  --pld-key-len <n>   N-gram match key length for PLD (default: 3).
        \\  --drafter <dir>     Path to an assistant drafter checkpoint -
        \\                        either a Gemma 4 cross-attention drafter or
        \\                        a DFlash block-drafter (auto-detected from
        \\                        its config: block_size + mask_token_id +
        \\                        target_layer_ids). Loaded at startup, bound
        \\                        to the target model, default draft source
        \\                        for new requests (priority: dflash >
        \\                        drafter > PLD > regular).
        \\  --draft-block-size <n>  Tokens per drafter round. Gemma default is
        \\                        auto-detected per target (E2B=2, E4B=4,
        \\                        26B-A4B=4, 31B=8); DFlash uses its config's
        \\                        block_size (an explicit value only clamps
        \\                        it DOWN). Pass to override.
        \\  --no-drafter        Never load a speculative-decoding drafter, including
        \\                      one shipped inside the checkpoint (drafter/ subdir)
        \\  --int8-qk           Run the head-dim-256 prefill attention with int8
        \\                        Q/K (16-bit activations on NAX GPUs only, lossy,
        \\                        wins from ~32k tokens). Also MLX_SERVE_INT8_QK=1.
        \\  --lm-head-mxfp8 / --no-lm-head-mxfp8
        \\                      Quantize a dense bf16 lm_head to MXFP8 at load, vLLM's
        \\                        VLLM_MXFP8_LM_HEAD bytes (lossy; half the head's bytes
        \\                        per token). On by default for quantized Flash Next
        \\                        packs on an integrated NVIDIA GPU.
        \\                        A model's "lm_head_mxfp8" setting wins.
        \\  --w8a8 / --no-w8a8  Run image and video DiT linears as rotated
        \\                        per-row int8 x int8 on the Neural Accelerators
        \\                        (lossy, faster; 4-bit packs hold more memory, bf16
        \\                        packs half). On by default only on an M5-class
        \\                        GPU with 64 GiB of RAM; --w8a8 opts in below that.
        \\  --decode-attn-quant / --no-decode-attn-quant
        \\                      Serve decode from quantized side copies of
        \\                      DENSE (bf16/f16) attention projection weights:
        \\                      INT8 group-32 for most layers, NVFP4 for the
        \\                      last 20% (late layers amplify quantization
        \\                      error far less). Cuts their per-token weight
        \\                      read by half or more on models that ship dense
        \\                      attention (~-25% decode overall).
        \\                      LOSSY: a real requantization, applied to
        \\                      decode/verify steps only; prefill keeps the
        \\                      dense weights. Default ON; --no-… restores
        \\                      exact dense decode. Env tuning:
        \\                      MLX_SERVE_DECODE_ATTN_QUANT_NVFP4_FROM=<layer>
        \\                      moves the 4-bit boundary, =off keeps the whole
        \\                      stack INT8.
        \\  --kv-quant <mode>   KV-cache quantization scheme:
        \\                        off (default), 4, 8     - affine group quant.
        \\                          Per-request override via the `kv_quant`
        \\                          body field.
        \\  --kv-attn-mode {{auto|dense|fused}}
        \\                      Decode read path for quantized KV. `dense`
        \\                        dequantizes K/V before SDPA; `fused` reads
        \\                        the packed cache in place at decode width
        \\                        (spec verify + prefill always read dense);
        \\                        `auto` (default) picks fused from 8K prompt
        \\                        tokens. Only effective at --kv-quant 4 or 8;
        \\                        per-request `kv_attn_mode` field overrides.
        \\  --prefill-chunk <n> Max tokens forwarded per prefill chunk
        \\                        (default: 8192). Auto-capped further per model
        \\                        so one layer's attention scores stay within
        \\                        budget; this flag is the ceiling, not a floor.
        \\                        Lower it if a long prompt spikes memory.
        \\  --prefill-decode-share <s>
        \\                      Target fraction of wall time (0..0.9) the
        \\                        decoding streams keep while another request
        \\                        prefills; also narrows that prefill's chunks.
        \\                        Default 0.25; 0 = the old schedule (env MLX_SERVE_PREFILL_DECODE_SHARE).
        \\  --prefix-cache-entries <n>
        \\                      Hot prefix cache LRU capacity in entries
        \\                        (default: 32). 0 disables all prefix reuse.
        \\  --no-prefix-cache-ram
        \\                      Disable idle RAM retention; an enabled SSD tier
        \\                        still persists and restores reusable prefixes.
        \\  --prefix-cache-mem <n>{{KB,MB,GB}}
        \\                      Hot prefix cache KV-bytes budget (default: 2GB,
        \\                        or one session at the working context on
        \\                        sliding-ring models when that is
        \\                        larger).
        \\                      Evicts LRU entries until the budget fits.
        \\                      Pass 0/off to disable the byte budget.
        \\  --prefix-cache-disk <n>{{KB,MB,GB}}
        \\                      SSD tier for the prefix cache (default: off).
        \\                      Seen prefixes persist under ~/.mlx-serve/kv-cache
        \\                        and are restored across restarts and RAM
        \\                        evictions instead of recomputed. Can use many
        \\                        GB of disk, so it's opt-in; e.g. 10GB. 0/off
        \\                        disables.
        \\  --shutdown-persist-secs <n>
        \\                      On SIGTERM, finish writing the SSD-tier slices not
        \\                        yet on disk, for at most <n> seconds, then exit
        \\                        (default: 45; 0 disables). A launchd job needs
        \\                        ExitTimeOut above it, or the flush is killed.
        \\  --wired-margin-gib <n>
        \\                      How far under iogpu.wired_limit_mb a plan may
        \\                        reach (default: 8, integers 2..32).
        \\  --tokenize-cache-entries <n>
        \\                      Per-model LRU cache of chat-template render +
        \\                        tokenize results (default: 4). Skips re-
        \\                        rendering identical messages on warm reuse.
        \\                        0 disables.
        \\  --model-dir <dir>   Directory of MLX models to discover at startup.
        \\                        Discovered siblings appear in /v1/models and
        \\                        load on the first request that names them
        \\                        (model=<id>).
        \\                        REPEATABLE (up to 8) - pass it once per folder
        \\                        your models live in. Scanned in order; the
        \\                        first folder wins a repeated model id, and a
        \\                        folder that can't be opened is skipped.
        \\  --max-resident-models <n>
        \\                      Maximum loaded models in memory (default: 3).
        \\                        ensureLoaded evicts LRU before exceeding.
        \\  --max-resident-mem <n>{{KB,MB,GB}}|auto
        \\                      Summed resident-bytes cap across all loaded
        \\                        models. Default 'auto' = the GPU working-set
        \\                        limit. Pass 0 to disable.
        \\  --idle-evict-secs <n>
        \\                      Evict .ready entries with refcount==0 if
        \\                        idle for this many seconds. Default: off.
        \\  --metrics           Enable Prometheus metrics at GET /metrics
        \\                        (opt-in; zero cost when off). Also
        \\                        GET /metrics.json.
        \\  --no-tool-autocorrect
        \\                      Disable tool-call ARGUMENT auto-correct - the
        \\                        coercion of parsed args to the tool schema's
        \\                        declared types (e.g. Python `False` -> JSON
        \\                        `false`). Args then pass through as the model
        \\                        emitted them (still valid JSON). Default: on.
        \\  --api-key <token>   Require this key on every request (chat
        \\                        completions, embeddings, models + metrics).
        \\                        Accepts Authorization: Bearer, x-api-key, HTTP
        \\                        Basic (key = password), or ?api_key=. /health
        \\                        stays open. Unset = no auth (default).
        \\  --api-key-strict    Require the key from loopback too (localhost is
        \\                        exempt by default). For embedders that want
        \\                        "only the key holder drives inference" on a
        \\                        shared machine. No effect without --api-key.
        \\  --api-key-env <VAR> Read the key from environment variable VAR
        \\                        instead of argv (the process table is
        \\                        world-readable). Unset/empty VAR = no auth.
        \\  --log-level <lvl>   Log level: error, warn, info, debug (default: info)
        \\  --log-file <path>   Persist the server log ("off" disables).
        \\                      Default: ~/.mlx-serve/logs/mlx-serve-<port>.log
        \\  --version           Print version and exit
        \\  --help              Show this help
        \\
    ) catch {};
    stdout_w.interface.flush() catch {};
}

pub fn main(init: std.process.Init) !void {
    // Referenced, or the linker strips the unreferenced section.
    if (@import("builtin").os.tag == .macos) std.mem.doNotOptimizeAway(&info_plist);
    const allocator = init.gpa;
    const io = init.io;

    // Bound MLX's reclaimable buffer pool before anything can allocate. ONCE,
    // above every subcommand branch - a per-serve-path call is how
    // `runHeadlessServe` (the mode the app always launches) silently ate the
    // --pld* flags. See server.mlxCacheLimitBytes for why MLX's own default
    // (~121 GB on a 128 GB Mac) is no defense.
    server_mod.applyMlxCacheLimit();
    server_mod.applyGpuCeilingEnv();
    // Resolve lazily-cached env reads on the main thread before other threads exist.
    @import("prefix_cache.zig").warmEnvCaches();

    // mlx-c's default handler exits the process; latch MLX failures instead (#353).
    mlx.installErrorHandler();

    // Materialize CLI args from the iterator API into a flat slice
    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer {
        for (args_list.items) |a| allocator.free(a);
        args_list.deinit(allocator);
    }
    while (args_iter.next()) |arg| {
        try args_list.append(allocator, try allocator.dupe(u8, arg));
    }
    const args = args_list.items;

    if (args.len == 1) {
        printUsage(io);
        return;
    }

    // Subcommands: `lesto serve`. `serve` falls
    // through into the normal flag parse (skipping the consumed positional)
    // and the serve path.
    var arg_start: usize = 1;
    var use_default_models_root = false;
    if (args.len >= 2 and args[1].len > 0 and args[1][0] != '-') {
        const cmd = args[1];
        if (std.mem.eql(u8, cmd, "serve")) {
            arg_start = 2;
            use_default_models_root = true;
        } else {
            log.err("unknown command '{s}' (expected serve)\n", .{cmd});
            std.process.exit(1);
        }
    }

    var model_dir: []const u8 = DEFAULT_MODEL_DIR;
    var models_root: ?[]const u8 = null; // --model-dir for plan 05 discovery
    // Additional `--model-dir` folders, scanned after the first. Fixed-size:
    // a handful of library folders is the shape this serves, and a bound the
    // parser enforces beats an allocation the arg loop has to unwind.
    var extra_roots: [7][]const u8 = undefined;
    var extra_roots_n: usize = 0;
    var port: u16 = 11234;
    var host: []const u8 = "0.0.0.0";
    var host_explicit = false;
    // `--log-file <path|off>`. null = default (`~/.mlx-serve/logs/mlx-serve-<port>.log`).
    var log_file_arg: ?[]const u8 = null;
    var serve_mode = false;
    var stream_mode = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var temperature: f32 = 0.0;
    // Serve-mode sampling defaults for requests that omit the field
    // (request > flag > model generation_config.json > hardcoded). `--temp`
    // doubles as the offline --prompt sampling temp, so track whether it was
    // explicitly given - only then does it become the serve default.
    var temp_explicit = false;
    var top_p_flag: ?f32 = null;
    var top_k_flag: ?u32 = null;
    var ctx_size: u32 = 0; // 0 = use model default
    var timeout: u32 = 300; // seconds, 0 = no timeout
    var reasoning_budget: i32 = -1; // -1 = unlimited
    var enable_pld = true; // Prompt Lookup Decoding (on by default; --no-pld to disable)
    var pld_draft_len: u32 = 5;
    var pld_key_len: u32 = 3;
    var drafter_dir: ?[]const u8 = null; // Path to Gemma 4 assistant drafter checkpoint
    var no_drafter = false; // --no-drafter: never load one, merged-in ones included
    var draft_block_size: u32 = drafter_mod.DEFAULT_BLOCK_SIZE;
    var draft_block_size_explicit: bool = false; // user passed --draft-block-size?
    // Plan 04 Phase 1: pre-fault weights and pre-compile kernels at boot.
    // Default ON in serve mode - small boot-time cost, big cold-prefill win.
    // --no-warmup-eager opts out for benchmarking / minimal-footprint deployments.
    var warmup_eager: bool = true;
    var kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense;
    // Phase 2 (Plan ricky): fused attention reads K/V triples directly via
    // mlx_quantized_matmul instead of dequantizing through DenseKVView.
    // Off by default - only the `.affine` cache scheme has a fused path.
    var kv_attn_mode: server_mod.KvAttnMode = .auto;
    // Plan 05 Phase D: multi-model caps. Defaults aim for "comfortable on
    // 32–64 GB systems running Gemma 4 E4B-class models". Override via the
    // CLI flags below; the Swift app exposes them under Advanced settings.
    var max_resident_models: u32 = 3;
    var max_resident_mem: u64 = 0; // 0 = auto (the GPU working-set limit)
    var max_resident_mem_explicit: bool = false;
    var idle_evict_secs: ?u32 = null;
    var metrics_enabled = false;
    @import("w8a8.zig").enabled = @import("w8a8.zig").defaultOn(transformer_mod.naxAvailable(), sys_info.totalMemBytes());
    var decode_share_flag: ?[]const u8 = null;
    var i: usize = arg_start;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--version")) {
            // Report app + engine versions WITHOUT booting the
            // server (the macOS app spawns this and parses it, src/version.zig,
            // Swift EngineVersions). MLX self-reports at runtime; mlx-c
            // has no runtime API and rides a build option.
            var mlx_ver = mlx.mlx_string_new();
            defer _ = mlx.mlx_string_free(mlx_ver);
            _ = mlx.mlx_version(&mlx_ver);
            const info = version_mod.Info{
                .app = VERSION,
                .mlx = std.mem.span(mlx.mlx_string_data(mlx_ver)),
                .mlx_c = build_options.mlx_c_version,
                .nax = transformer_mod.naxStatus(),
            };
            var ver_buf: [512]u8 = undefined;
            var ver_w = std.Io.File.stdout().writerStreaming(io, &ver_buf);
            version_mod.writeReport(&ver_w.interface, info) catch {};
            ver_w.interface.flush() catch {};
            return;
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage(io);
            return;
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            i += 1;
            port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
            i += 1;
            host = args[i];
            host_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--serve")) {
            serve_mode = true;
        } else if (std.mem.eql(u8, args[i], "--stream")) {
            stream_mode = true;
        } else if (std.mem.eql(u8, args[i], "--prompt") and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
            serve_default_max_tokens = max_tokens;
        } else if (std.mem.eql(u8, args[i], "--temp") and i + 1 < args.len) {
            i += 1;
            temperature = try std.fmt.parseFloat(f32, args[i]);
            temp_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--top-p") and i + 1 < args.len) {
            i += 1;
            top_p_flag = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--top-k") and i + 1 < args.len) {
            i += 1;
            top_k_flag = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--ctx-size") and i + 1 < args.len) {
            i += 1;
            ctx_size = try std.fmt.parseInt(u32, args[i], 10);
        } else if ((std.mem.eql(u8, args[i], "--config-overrides") or
            std.mem.eql(u8, args[i], "--hf-overrides")) and i + 1 < args.len)
        {
            // vLLM's `--hf-overrides` analogue. Applied the moment the flag is
            // seen, so no config.json is ever parsed without it - the loaded
            // model, the registry stubs and `/v1/models` all agree on the
            // resulting document. Validated here so a typo names the flag
            // instead of surfacing as a model-load parse error.
            i += 1;
            if (!configOverridesJsonValid(args[i])) {
                log.err("--config-overrides: expected a JSON object; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
            model_mod.setConfigOverrides(args[i]);
            log.info("[args] config-overrides: {s}\n", .{args[i]});
        } else if (std.mem.eql(u8, args[i], "--embedding-max-length") and i + 1 < args.len) {
            i += 1;
            // Module global (like --max-concurrent): every serve path reads it,
            // so a hand-rolled ServerConfig can't eat it (the runHeadlessServe
            // class). "auto" = 0 = bound only by the model's declared window.
            server_mod.embedding_max_length = if (std.mem.eql(u8, args[i], "auto"))
                0
            else
                try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-prevent-sleep")) {
            sleep_inhibit_mod.setEnabled(false);
        } else if (std.mem.eql(u8, args[i], "--skip-mem-preflight")) {
            scheduler_mod.skip_mem_preflight = true;
        } else if (std.mem.eql(u8, args[i], "--no-safety")) {
            // Retired image content filter; accepted as a no-op.
        } else if (std.mem.eql(u8, args[i], "--pld")) {
            enable_pld = true;
        } else if (std.mem.eql(u8, args[i], "--no-tool-autocorrect")) {
            server_mod.g_tool_autocorrect = false;
        } else if (std.mem.eql(u8, args[i], "--no-pld")) {
            enable_pld = false;
        } else if (std.mem.eql(u8, args[i], "--pld-draft-len") and i + 1 < args.len) {
            i += 1;
            pld_draft_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--pld-key-len") and i + 1 < args.len) {
            i += 1;
            pld_key_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--drafter") and i + 1 < args.len) {
            i += 1;
            drafter_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--draft-block-size") and i + 1 < args.len) {
            i += 1;
            draft_block_size = try std.fmt.parseInt(u32, args[i], 10);
            draft_block_size_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--metrics")) {
            metrics_enabled = true;
        } else if (std.mem.eql(u8, args[i], "--api-key") and i + 1 < args.len) {
            i += 1;
            // Borrowed from argv (lives for the process). Empty ⇒ leave open.
            if (args[i].len > 0) server_mod.g_api_key = args[i];
        } else if (std.mem.eql(u8, args[i], "--api-key-strict")) {
            server_mod.g_api_key_strict = true;
        } else if (std.mem.eql(u8, args[i], "--api-key-env") and i + 1 < args.len) {
            i += 1;
            // The key read from a named environment variable instead of argv:
            // the process table is world-readable, argv with it. Same
            // borrow-for-the-process lifetime as --api-key; empty/unset
            // leaves the server open, exactly like an empty --api-key.
            // getenv needs a null-terminated name; args[i] is a plain
            // slice, so print a `:0` copy (process-lifetime, like the argv
            // borrow --api-key uses). std.c.getenv is how every other env
            // read in this codebase works. An unset var leaves the server
            // open, exactly like an empty --api-key.
            const name = std.fmt.allocPrintSentinel(allocator, "{s}", .{args[i]}, 0) catch null;
            if (name) |name_z| {
                if (std.c.getenv(name_z.ptr)) |value| {
                    const key = std.mem.span(value);
                    if (key.len > 0) server_mod.g_api_key = key;
                }
            }
        } else if (std.mem.eql(u8, args[i], "--no-drafter")) {
            no_drafter = true;
        } else if (std.mem.eql(u8, args[i], "--w8a8")) {
            @import("w8a8.zig").enabled = true;
        } else if (std.mem.eql(u8, args[i], "--no-w8a8")) {
            @import("w8a8.zig").enabled = false;
        } else if (std.mem.eql(u8, args[i], "--int8-qk")) {
            @import("nax_attention.zig").int8_qk_cli = true;
        } else if (std.mem.eql(u8, args[i], "--lm-head-mxfp8")) {
            @import("transformer.zig").lm_head_mxfp8_cli = true;
        } else if (std.mem.eql(u8, args[i], "--no-lm-head-mxfp8")) {
            @import("transformer.zig").lm_head_mxfp8_cli = false;
        } else if (std.mem.eql(u8, args[i], "--decode-attn-quant")) {
            transformer_mod.decode_attn_quant_flag = true;
        } else if (std.mem.eql(u8, args[i], "--no-decode-attn-quant")) {
            transformer_mod.decode_attn_quant_flag = false;
        } else if (std.mem.eql(u8, args[i], "--reasoning-budget") and i + 1 < args.len) {
            i += 1;
            reasoning_budget = try std.fmt.parseInt(i32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--log-level") and i + 1 < args.len) {
            i += 1;
            if (log.Level.fromString(args[i])) |level| log.setLevel(level);
        } else if (std.mem.eql(u8, args[i], "--log-file") and i + 1 < args.len) {
            i += 1;
            log_file_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--warmup-eager")) {
            warmup_eager = true;
        } else if (std.mem.eql(u8, args[i], "--no-warmup-eager")) {
            warmup_eager = false;
        } else if (std.mem.eql(u8, args[i], "--prefill-chunk") and i + 1 < args.len) {
            i += 1;
            // `explicit` disables the machine-sized pin, so only a real width
            // earns it - a typo'd value keeps the defaults (flag-absent
            // behavior), never a silent 8192 that also switches sizing off.
            if (std.fmt.parseInt(usize, args[i], 10)) |v| {
                if (v > 0) {
                    generate_mod.prefill_chunk_override = v;
                    generate_mod.prefill_chunk_explicit = true;
                }
            } else |_| {}
        } else if (std.mem.eql(u8, args[i], "--prefill-trace")) {
            generate_mod.prefill_trace_force = true;
        } else if (std.mem.eql(u8, args[i], "--no-prefix-cache-ram")) {
            server_mod.prefix_cache_ram_enabled = false;
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-entries") and i + 1 < args.len) {
            i += 1;
            server_mod.prefix_cache_capacity = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-mem") and i + 1 < args.len) {
            // Wave 1.B - KV-bytes budget for the hot prefix cache. Accepts
            // bare numbers (bytes), a suffix of `MB`/`GB`/`KB` (case-
            // insensitive), or `0`/`off` to disable the byte budget entirely
            // (count cap from --prefix-cache-entries still applies).
            i += 1;
            server_mod.prefix_cache_mem_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-mem: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_mod.prefix_cache_mem_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-disk") and i + 1 < args.len) {
            // SSD tier for the hot prefix cache: previously-seen prefixes are
            // persisted as chunked safetensors and restored across restarts
            // and RAM evictions instead of recomputed. Byte budget with the
            // same size grammar as --prefix-cache-mem; `0`/`off` disables.
            i += 1;
            server_mod.prefix_cache_disk_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-disk: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--shutdown-persist-secs") and i + 1 < args.len) {
            i += 1;
            server_mod.shutdown_persist_secs = std.fmt.parseInt(u32, args[i], 10) catch {
                log.err("--shutdown-persist-secs: expected whole seconds (0 = off); got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--tokenize-cache-entries") and i + 1 < args.len) {
            // Iteration 2 (perf-plan Phase 4 #3): caps the per-LoadedModel
            // chat-template tokenize cache. 0 = off (every request re-
            // renders+re-tokenizes, mirrors pre-Iteration-2 behavior).
            i += 1;
            server_mod.tokenize_cache_entries = std.fmt.parseInt(u32, args[i], 10) catch 4;
        } else if (std.mem.eql(u8, args[i], "--os-reserve-gib") and i + 1 < args.len) {
            i += 1;
            status.os_reserve_override = status.parseOsReserveGib(args[i]) catch {
                log.err("--os-reserve-gib: expected an integer 0..64, got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--wired-margin-gib") and i + 1 < args.len) {
            i += 1;
            server_mod.wired_limit_margin_bytes = server_mod.parseWiredMarginGib(args[i]) catch {
                log.err("--wired-margin-gib: expected an integer 2..32, got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--prefill-decode-share") and i + 1 < args.len) {
            i += 1;
            decode_share_flag = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-concurrent") and i + 1 < args.len) {
            i += 1;
            server_mod.max_concurrent = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, args[i], "--model-dir") and i + 1 < args.len) {
            // REPEATABLE. A user's library can live in more than one place (the
            // app's download folder, an external drive, an LM Studio tree), and
            // with one root the others are invisible to /v1/models even though
            // the picker lists them. Extras past the cap are refused loudly -
            // silently dropping a folder the user asked us to scan is the
            // silent-flag-eater class.
            i += 1;
            if (models_root == null) {
                models_root = args[i];
            } else if (extra_roots_n < extra_roots.len) {
                extra_roots[extra_roots_n] = args[i];
                extra_roots_n += 1;
            } else {
                log.err("--model-dir: at most {d} folders (got one more: {s})\n", .{ extra_roots.len + 1, args[i] });
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[i], "--max-resident-models") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on .ready entries in the registry.
            // ensureLoaded evicts LRU before loading when this would be exceeded.
            i += 1;
            max_resident_models = std.fmt.parseInt(u32, args[i], 10) catch 3;
            if (max_resident_models == 0) max_resident_models = 1;
        } else if (std.mem.eql(u8, args[i], "--max-resident-mem") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on summed resident bytes. Accepts the
            // same suffixes as --prefix-cache-mem. Special string "auto"
            // (or default 0) → the GPU working-set limit.
            i += 1;
            if (std.mem.eql(u8, args[i], "auto")) {
                max_resident_mem = 0;
            } else {
                max_resident_mem = parseSizeArg(args[i]) catch {
                    log.err("--max-resident-mem: expected '<n>{{MB,GB,KB}}' or 'auto'; got '{s}'\n", .{args[i]});
                    std.process.exit(1);
                };
                max_resident_mem_explicit = true;
            }
        } else if (std.mem.eql(u8, args[i], "--idle-evict-secs") and i + 1 < args.len) {
            // Idle eviction window. When set, `server.idleEvictLoop` unloads
            // .ready entries (refcount==0) whose last_used_ms is older than
            // this. Default off - eviction is on-demand only.
            i += 1;
            const n = std.fmt.parseInt(u32, args[i], 10) catch 0;
            idle_evict_secs = if (n > 0) n else null;
        } else if (std.mem.eql(u8, args[i], "--kv-quant") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "off") or std.mem.eql(u8, args[i], "0")) {
                kv_quant_config = transformer_mod.KVQuantConfig.dense;
            } else if (std.mem.eql(u8, args[i], "4")) {
                kv_quant_config = transformer_mod.KVQuantConfig.affine(4);
            } else if (std.mem.eql(u8, args[i], "8")) {
                kv_quant_config = transformer_mod.KVQuantConfig.affine(8);
            } else {
                log.err("--kv-quant: expected one of {{off, 4, 8}}; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[i], "--kv-attn-mode") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "dense")) {
                kv_attn_mode = .dense;
            } else if (std.mem.eql(u8, args[i], "fused")) {
                kv_attn_mode = .fused;
            } else if (std.mem.eql(u8, args[i], "auto")) {
                kv_attn_mode = .auto;
            } else {
                log.err("--kv-attn-mode: expected 'dense', 'fused' or 'auto'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        } else {
            // Nothing above consumed it. This loop used to end here with no
            // else at all, so an unrecognized argument was dropped in SILENCE
            // - `--model=<path>` (the '='-joined form none of the arms match)
            // booted a clean-looking headless server that then auto-picked
            // some other model. A launcher that ignores what it was asked for
            // is worse than one that refuses to start.
            const reason = cli_mod.classifyUnparsedArg(args[i], i + 1 == args.len);
            log.err("unrecognized argument '{s}' - {s}\n", .{ args[i], reason.hint() });
            std.process.exit(1);
        }
    }
    generate_mod.machine_prefill_chunk_cap = server_mod.autoPrefillChunkCap();

    const decode_share_env: ?[]const u8 = if (std.c.getenv("MLX_SERVE_PREFILL_DECODE_SHARE")) |r| std.mem.sliceTo(r, 0) else null;
    scheduler_mod.prefill_decode_share = scheduler_mod.resolveDecodeShare(decode_share_flag, decode_share_env) catch {
        log.err("--prefill-decode-share / MLX_SERVE_PREFILL_DECODE_SHARE: expected a number >= 0 (above 0.9 clamps to 0.9), got '{s}'\n", .{decode_share_flag orelse decode_share_env.?});
        std.process.exit(1);
    };

    // Subcommand plumbing: `serve` defaults the discovery root to
    // ~/.mlx-serve/models so every model there is loadable by name.
    var default_models_root_storage: ?[]u8 = null;
    defer if (default_models_root_storage) |r| allocator.free(r);
    if (use_default_models_root) serve_mode = true;
    // Offline mode is one --prompt; without one, serving is the default (--serve stays accepted).
    if (prompt == null) serve_mode = true;
    // An unspecified `--model-dir` falls back to the shared models root that
    // the app already agrees on. Gated so `--model <path> --serve`
    // still serves exactly the one model it named (cli.shouldDefaultModelsRoot).
    if (models_root == null and cli_mod.shouldDefaultModelsRoot(.{
        .subcommand = use_default_models_root,
        .serve_mode = serve_mode,
        .has_explicit_model = model_dir.len > 0,
    })) {
        const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
        default_models_root_storage = try cli_mod.modelsRootPath(allocator, home);
        models_root = default_models_root_storage;
    }

    // Persist the server log. The macOS app only keeps stderr in a 64 KB
    // in-memory ring, so a server that crashed or was restarted takes its
    // history with it - exactly when you need it (see the 2026-07-08 pi
    // session post-mortem). Serving paths only.
    //
    // NOTE: opened AFTER `--log-level` is parsed (so the level gates what
    // reaches disk) and BEFORE model discovery/loading (so weight-load and
    // auto-context lines land in the file).
    if (serve_mode) {
        var log_path_buf: [1024]u8 = undefined;
        const chosen: ?[]const u8 = if (log_file_arg) |a|
            (if (std.mem.eql(u8, a, "off") or std.mem.eql(u8, a, "none")) null else a)
        else if (std.c.getenv("HOME")) |h|
            log.defaultLogPath(&log_path_buf, std.mem.span(h), port) catch null
        else
            null;
        if (chosen) |p| {
            if (log.openFile(p, log.default_max_bytes)) |_| {
                log.info("Logging to {s} (rotates at {d} MB)\n", .{ p, log.default_max_bytes / (1024 * 1024) });
            } else |e| {
                log.warn("could not open log file {s}: {s} (stderr only)\n", .{ p, @errorName(e) });
            }
        }
    }
    defer log.closeFile();

    // Plan 05 Phase 1: model discovery. When --model-dir is passed, scan
    // the directory for subdirectories containing config.json. The
    // discovered list is published via /v1/models. v1: routing still goes
    // to a single loaded model - if --model isn't set, pick the first
    // discovered. v2 (plan 05 phases 2-5) adds on-demand load and LRU.
    var discovery_storage: ?model_discovery.DiscoveryResult = null;
    defer if (discovery_storage) |*d| d.deinit();
    if (models_root) |root| {
        // Every `--model-dir`, first-wins on a repeated id (see
        // `discoverModelsMany` for why de-dup is not optional here).
        var roots_buf: [8][]const u8 = undefined;
        roots_buf[0] = root;
        for (extra_roots[0..extra_roots_n], 0..) |r, n| roots_buf[n + 1] = r;
        const roots = roots_buf[0 .. 1 + extra_roots_n];
        discovery_storage = model_discovery.discoverModelsMany(io, allocator, roots) catch |err| blk: {
            log.warn("--model-dir scan failed: {s}\n", .{@errorName(err)});
            break :blk null;
        };
        if (discovery_storage) |*d| {
            if (roots.len == 1) {
                log.info("Discovered {d} model(s) under {s}:\n", .{ d.models.len, root });
            } else {
                log.info("Discovered {d} model(s) under {d} folders:\n", .{ d.models.len, roots.len });
                for (roots) |r| log.info("  (scanning {s})\n", .{r});
            }
            for (d.models) |m| {
                if (m.bytes_on_disk) |b| {
                    log.info("  - {s} ({d:.1} GB)\n", .{ m.id, @as(f64, @floatFromInt(b)) / 1_073_741_824.0 });
                } else {
                    log.info("  - {s}\n", .{m.id});
                }
            }
            // No auto-select: when `--model` is omitted but `--model-dir` is
            // present, the server starts HEADLESS (no primary model). All
            // discovered models are registered as stubs and load on demand by
            // the first request that names them - chat OR media. The headless
            // branch in the serve block below handles this.
        }
    }
    // In serve mode, check if the port is already in use before loading the model
    // (model loading takes seconds - fail fast instead of wasting time)
    if (serve_mode) {
        if (portInUse(io, port)) {
            log.err("Port {d} is already in use - another lesto instance may be running.\n", .{port});
            log.err("Port {d} is in use: stop the process listening on it (lsof -ti:{d}) or use a different port (--port {d}).\n", .{ port, port, port + 1 });
            std.process.exit(1);
        }
        // Above every serve dispatch (headless/media return early below).
        if (server_mod.shouldWarnOpenBind(host_explicit, host)) {
            log.warn("Listening on {s}:{d} - reachable by every device on the network this machine is on.\n", .{ host, port });
            log.warn("Restrict to this machine with --host 127.0.0.1 (a future version will make that the default).\n", .{});
        }
    }

    // Observability: allocate the metrics core once (when --metrics is on) and
    // publish it via the server-global `g_metrics`. Declared here - above every
    // serve-dispatch path (headless, media, and the primary MLX
    // path) - so `server_mod.serve()` spawns the gauge sampler + routes /metrics
    // regardless of engine, and each LoadParams builder reads it back into the
    // scheduler's per-request sink via `.metrics = server_mod.g_metrics`. The
    // instance lives on this stack frame for the whole process lifetime; the
    // defer clears the global so an early serve() failure can't leave it
    // dangling. Off (the default) → null: a single per-request branch, no cost.
    var metrics_instance: ?metrics_mod.Metrics = if (metrics_enabled) metrics_mod.Metrics.init() else null;
    if (metrics_instance) |*m| {
        m.process_start_time_seconds.set(@intCast(@max(0, io_util.nowSecs(io))));
        server_mod.g_metrics = m;
    }
    defer server_mod.g_metrics = null;

    // Print MLX version
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    log.info("lesto {s} (MLX {s})\n", .{ VERSION, mlx.mlx_string_data(ver) });

    // Every text-gen serve path takes the PLD defaults from this ONE value -
    // see `server.PldDefaults`. Built after arg parsing so it can't capture a
    // pre-flag default, and passed whole so a path can't honor `--pld` while
    // dropping the two lengths next to it (which is precisely what headless
    // mode did).
    const cli_pld = server_mod.PldDefaults.fromCli(enable_pld, pld_draft_len, pld_key_len);

    // Echo the resolved arguments - makes drafter/target mismatches obvious
    // from the log without having to scroll through the whole launch line in
    // the parent's process listing.
    log.info("[args] model: {s}\n", .{model_dir});
    if (drafter_dir) |dir| {
        log.info("[args] drafter: {s} (block_size={d}{s})\n", .{
            dir,
            draft_block_size,
            if (draft_block_size_explicit) "" else ", auto",
        });
    } else {
        log.info("[args] drafter: <none>\n", .{});
    }
    if (serve_mode) {
        log.info("[args] serve: {s}:{d}, ctx-size={d}, pld={s}, prevent-sleep={}\n", .{
            host,
            port,
            ctx_size,
            if (enable_pld) "on" else "off",
            sleep_inhibit_mod.isEnabled(),
        });
    }
    switch (kv_quant_config.scheme) {
        .off => log.info("[args] kv-quant: off\n", .{}),
        .affine => log.info("[args] kv-quant: affine {d}-bit (group={d})\n", .{ kv_quant_config.bits, kv_quant_config.group_size }),
    }
    log.info("[args] kv-attn-mode: {s}\n", .{@tagName(kv_attn_mode)});

    // Set GPU as default
    var metal_avail: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_avail));
    log.info("Metal GPU: {}\n", .{metal_avail});

    if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        try mlx.check(mlx.mlx_set_default_device(gpu_dev));
    }

    // Seed MLX RNG with current wall-clock time for non-deterministic sampling
    _ = mlx.mlx_random_seed(@intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()));

    if (serve_mode) {
        // Headless boot: `--model-dir` given, no `--model`. Start with no
        // primary model; everything (chat + hosted models) loads on demand through the
        // registry. The app uses this so a single server hosts chat and the
        // hosted models, coexisting under one memory budget.
        if (model_dir.len == 0) {
            const discovery_for_registry = discovery_storage;
            discovery_storage = null; // ownership moves to the registry
            try runHeadlessServe(io, allocator, discovery_for_registry, host, port, ctx_size, timeout, reasoning_budget, max_resident_models, max_resident_mem, max_resident_mem_explicit, idle_evict_secs, kv_quant_config, cli_pld, .{
                .no_drafter = no_drafter,
                .draft_block_size = draft_block_size,
                .draft_block_size_explicit = draft_block_size_explicit,
            });
            return;
        }
    }

    // Parse config - heap allocate so the LoadedModel can take ownership
    // (Plan 05). Free path in serve_mode = registry.deinit; offline mode =
    // explicit defer on `config_storage`.
    const config_storage = try allocator.create(model_mod.ModelConfig);
    var config_owned_by_registry = false;
    // defer-only, NOT errdefer + defer: a plain `defer` already runs on the
    // error-return path, so pairing it with an errdefer that has the same body
    // frees the resource twice on error (double-free / SIGSEGV). The runtime
    // `owned_by_registry` guard makes the single defer correct on every exit.
    // `create` hands back UNINITIALIZED memory and the defer below READS a
    // field, so the struct gets a valid value before that defer can ever run:
    // a `parseConfig` failure would otherwise free a garbage pointer.
    config_storage.* = std.mem.zeroes(model_mod.ModelConfig);
    defer if (!config_owned_by_registry) {
        config_storage.deinit(allocator);
        allocator.destroy(config_storage);
    };
    config_storage.* = try model_mod.parseConfig(io, allocator, model_dir);
    const config = config_storage;
    log.info("Model: {s} ({d} layers, {d}-dim, head_dim={d}, {d}h/{d}kv, {d}-bit {s} quant)\n", .{
        config.model_type,
        config.num_hidden_layers,
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.quant_bits,
        @tagName(config.quant_mode),
    });

    // Load tokenizer - heap-allocated, ownership transfers to registry on serve_mode.
    log.info("Loading tokenizer...\n", .{});
    const tok = try allocator.create(tokenizer_mod.Tokenizer);
    var tok_owned_by_registry = false;
    tok.* = tokenizer_mod.loadTokenizer(io, allocator, model_dir) catch |err| {
        // Raw memory only - nothing initialized to deinit. The cleanup defer
        // below must NOT be registered yet: deinit on the undefined pointee
        // was a live SIGSEGV on a partially-downloaded model dir (the
        // preloadCpuState errdefer-after-init pattern applies here too).
        allocator.destroy(tok);
        log.err("failed to load tokenizer from {s}: {s} (incomplete download? fetch the model again, or delete the dir)\n", .{ model_dir, @errorName(err) });
        return err;
    };
    // defer-only (see config note above): errdefer + defer with the same body
    // double-frees on the error-return path.
    defer if (!tok_owned_by_registry) {
        tok.deinit();
        allocator.destroy(tok);
    };

    // Load chat config - heap-allocated, ownership transfers to registry on serve_mode.
    const chat_config = try allocator.create(chat_mod.ChatConfig);
    var chat_config_owned_by_registry = false;
    chat_config.* = chat_mod.loadChatConfig(io, allocator, model_dir) catch |err| {
        // Raw memory only - see the tokenizer catch above.
        allocator.destroy(chat_config);
        return err;
    };
    // defer-only (see config note above): errdefer + defer with the same body
    // double-frees on the error-return path - this is the one that crashed in
    // the #45 GPU-OOM pre-flight refusal (ChatConfig.deinit ran twice).
    defer if (!chat_config_owned_by_registry) {
        chat_config.deinit();
        allocator.destroy(chat_config);
    };

    {
        var settings = model_settings_mod.overrideFor(allocator, io, model_dir);
        defer settings.deinit(allocator);
        scheduler_mod.applyModelSettings(config, chat_config, &settings);
    }
    config.applyTokenizer(tok, chat_config.eos_token);

    if (serve_mode) {
        // ── Plan 05: build the ModelRegistry, register a stub for the
        //    loaded model, and pass everything to serve(). The registry
        //    takes ownership of `discovery_storage` (if any) and, once
        //    the inference thread completes loading, ownership of
        //    config/tok/chat_config too.
        const model_id = blk: {
            var p = model_dir;
            while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
            if (p.len == 0) break :blk config.model_type;
            if (std.mem.lastIndexOfScalar(u8, p, '/')) |slash_idx| break :blk p[slash_idx + 1 ..];
            break :blk p;
        };

        const discovery_for_registry = discovery_storage;
        discovery_storage = null; // ownership moves to the registry

        // Without an explicit cap the registry gate is Metal's working-set limit. 0 disables it.
        const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_explicit, max_resident_mem);
        if (effective_max_resident_mem > 0) {
            log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{
                max_resident_models,
                @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0,
            });
        } else {
            log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
        }

        const registry = try model_registry_mod.ModelRegistry.init(
            allocator,
            io,
            discovery_for_registry,
            max_resident_models,
            effective_max_resident_mem,
            idle_evict_secs,
        );
        defer registry.deinit();

        // Register the loaded model. Use the pre-registered discovery entry
        // when available (so id/path/bytes_on_disk are consistent across
        // /v1/models listings); otherwise create a fresh stub.
        const entry = if (registry.peek(model_id)) |e|
            e
        else if (registry.peekByPath(model_dir)) |e|
            // Discovered under an org/name id whose basename differs from
            // model_id - reuse it, never register the same path twice.
            e
        else
            try registry.registerStub(model_id, model_dir, null);
        try registry.setDefault(entry.id);

        // Ownership-transfer defer: registry takes ownership of
        // config/tok/chat_config IF the inference-thread load installed
        // them on `entry` (entry.config != null). Declared AFTER
        // registry.deinit so it fires BEFORE it on scope exit - by the
        // time registry.deinit walks the entry we've already decided who
        // owns the heap pointers, so the early defers can no-op.
        defer if (entry.config != null) {
            config_owned_by_registry = true;
            tok_owned_by_registry = true;
            chat_config_owned_by_registry = true;
        };

        const params = scheduler_mod.LoadParams{
            .registry = registry,
            .entry = entry,
            .config = config,
            .tok = tok,
            .chat_config = chat_config,
            .model_dir = model_dir,
            .ctx_size = ctx_size,
            .drafter_dir = drafter_dir orelse "",
            .no_drafter = no_drafter,
            .warmup_eager = warmup_eager,
            .draft_block_size = draft_block_size,
            .draft_block_size_explicit = draft_block_size_explicit,
            .kv_quant_config = kv_quant_config,
            .prefix_cache_capacity = server_mod.prefix_cache_capacity,
            .prefix_cache_ram_enabled = server_mod.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = server_mod.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = server_mod.prefixCacheMemForLoad,
            .prefix_cache_disk_bytes = server_mod.prefix_cache_disk_bytes,
            .tokenize_cache_entries = server_mod.tokenize_cache_entries,
            .metrics = server_mod.g_metrics,
        };
        try server_mod.serve(io, allocator, params, config, host, port, .{
            .max_context_size = ctx_size,
            .request_timeout_sec = timeout,
            .default_reasoning_budget = reasoning_budget,
            .default_max_tokens = serve_default_max_tokens,
            .default_temperature = if (temp_explicit) temperature else null,
            .default_top_p = top_p_flag,
            .default_top_k = top_k_flag,
            .default_enable_pld = cli_pld.enable,
            .default_pld_draft_len = cli_pld.draft_len,
            .default_pld_key_len = cli_pld.key_len,
            .kv_attn_mode = kv_attn_mode,
        });
    } else {
        // ── Offline single-prompt mode. mlx ops run on this thread, no
        //    scheduler. The same load path as pre-A1.
        log.info("Loading weights...\n", .{});
        var weights = try model_mod.loadModelWeights(io, allocator, model_dir, config, false);
        defer weights.deinit();
        model_mod.resolveWeightPrefix(config, &weights);
        try model_mod.narrowHadamardPackTables(config, &weights, mlx.gpuStream());

        var xfm = try transformer_mod.Transformer.init(io, allocator, config.*, &weights);
        defer xfm.deinit();

        xfm.round_cost.layout = round_cost_mod.layoutFor(config);

        // Reserved-token suppression, same derivation as the serve path.
        generate_mod.installSuppressMask(&xfm, tok, chat_config.chat_template, config.eosTokenSlice());

        // Honor --kv-quant in offline mode too. The serve path threads this
        // through Slot caches via the scheduler; here we swap the
        // Transformer's own legacy cache to match.
        if (kv_quant_config.scheme != .off) {
            try xfm.cache.reinit(config.num_hidden_layers, kv_quant_config);
        }

        // JIT-compile + wire memory limits (policy: mlx.applyWiredPolicy).
        {
            const wired = mlx.applyWiredPolicy();
            if (wired.target) |t| log.debug("[wired] mode={s} limit={d} MB\n", .{ @tagName(wired.mode), t / (1024 * 1024) });
        }
        if (config.hidden_act == .gelu_approx) {
            xfm.compileGelu();
            xfm.compileGeglu();
        }
        if (config.final_logit_softcapping > 0.0) {
            xfm.compileSoftcap();
        }
        if (xfm.moe_layers != null) {
            xfm.compileMoeRouting();
        }
        log.info("Model ready.\n", .{});

        const user_prompt = prompt.?; // offline mode only runs with --prompt
        const messages = [_]chat_mod.Message{
            .{ .role = "user", .content = user_prompt },
        };

        const prompt_ids = try chat_mod.formatChat(allocator, tok, &messages, chat_config, null, null, false, false);
        defer allocator.free(prompt_ids);

        // Reset peak memory before generation
        _ = mlx.mlx_reset_peak_memory();

        const eos_slice = config.eosTokenSlice();
        const sampling = generate_mod.SamplingParams{ .temperature = temperature };

        var stdout_buf: [16 * 1024]u8 = undefined;
        var stdout_w_state = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
        const stdout_w = &stdout_w_state.interface;
        defer stdout_w.flush() catch {};

        if (stream_mode) {
            // Streaming: print tokens as they're generated
            const prefill_start = std.Io.Timestamp.now(io, .awake);
            var gen = try generate_mod.Generator.init(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice);
            defer gen.deinit(allocator);

            const prefill_ns: u64 = @intCast(prefill_start.untilNow(io, .awake).nanoseconds);
            const prefill_tps: f64 = if (prefill_ns > 0)
                @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
            else
                0.0;

            try stdout_w.writeAll("==========\n");
            const decode_start = std.Io.Timestamp.now(io, .awake);
            var completion_tokens: u32 = 0;
            while (try gen.next(allocator)) |token_id| {
                const ids = [_]u32{token_id};
                const piece = try tok.decode(allocator, &ids, completion_tokens == 0);
                defer allocator.free(piece);
                if (piece.len > 0) {
                    try stdout_w.writeAll(piece);
                    try stdout_w.flush();
                }
                completion_tokens += 1;
            }
            const decode_ns: u64 = @intCast(decode_start.untilNow(io, .awake).nanoseconds);
            const decode_tps: f64 = if (decode_ns > 0)
                @as(f64, @floatFromInt(completion_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
            else
                0.0;

            try stdout_w.writeAll("\n==========\n");
            try stdout_w.print("Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ prompt_ids.len, prefill_tps });
            try stdout_w.print("Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ completion_tokens, decode_tps });
        } else {
            // Non-streaming: generate all tokens then print
            const result = try generate_mod.generate(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, 0);
            defer allocator.free(result.text);
            defer allocator.free(result.token_ids);

            try stdout_w.writeAll("==========\n");
            try stdout_w.writeAll(result.text);
            try stdout_w.writeAll("\n==========\n");
            try stdout_w.print("Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.prompt_tokens, result.prefill_tps });
            try stdout_w.print("Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.completion_tokens, result.decode_tps });
        }

        var peak_mem: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak_mem);
        const peak_gb = @as(f64, @floatFromInt(peak_mem)) / (1024.0 * 1024.0 * 1024.0);
        try stdout_w.print("Peak memory: {d:.3} GB\n", .{peak_gb});
    }
}

/// Check if a port is already in use by trying to connect to it.
fn portInUse(io: std.Io, port: u16) bool {
    const addr: std.Io.net.IpAddress = .{ .ip4 = std.Io.net.Ip4Address.loopback(port) };
    const stream = addr.connect(io, .{ .mode = .stream }) catch return false;
    stream.close(io);
    return true;
}
/// Registry resident-memory cap: the user's explicit value, or Metal's working-set limit.
/// 0 = query failed → unlimited (the count cap still applies).
fn autoResidentMemBytes(explicit: bool, val: u64) u64 {
    return model_registry_mod.ModelRegistry.residentMemCap(explicit, val, mlx.maxRecommendedWorkingSet());
}

fn dirBasename(path: []const u8) []const u8 {
    var p = path;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    if (p.len == 0) return p;
    if (std.mem.lastIndexOfScalar(u8, p, '/')) |i| return p[i + 1 ..];
    return p;
}

/// Launch flags that shape every on-demand load. Headless takes them as one
/// value so its LoadParams cannot leave any of them at a struct default.
const SpecLoadFlags = struct {
    no_drafter: bool,
    draft_block_size: u32,
    draft_block_size_explicit: bool,
};

const tok_mod = tokenizer_mod;

const StubCpuState = struct {
    config: *model_mod.ModelConfig,
    tok: *tok_mod.Tokenizer,
    chat_config: *chat_mod.ChatConfig,
};

/// Build heap-allocated placeholder config/tokenizer/chat_config that seed the
/// scheduler's borrowed-view fields; never installed on an entry.
/// `freeStubCpuState` frees them.
fn buildStubCpuState(allocator: std.mem.Allocator) !StubCpuState {
    const config = try allocator.create(model_mod.ModelConfig);
    errdefer allocator.destroy(config);
    config.* = model_mod.ModelConfig{
        .model_type = "",
        .weight_prefix = "model",
        .num_hidden_layers = 1,
        .hidden_size = 1,
        .head_dim = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .max_position_embeddings = 4096,
    };

    const tok = try allocator.create(tok_mod.Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    var b: usize = 0;
    while (b < 256) : (b += 1) byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .sentencepiece_bpe,
        .byte_to_unicode = byte_map,
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(chat_mod.ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

fn freeStubCpuState(allocator: std.mem.Allocator, s: *StubCpuState) void {
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// Headless serve mode: start with NO primary model. The registry holds all
/// discovery stubs; chat AND media models load on demand when a request
/// names a discovered id, coexisting under one memory
/// budget. The scheduler's borrowed-view fields are seeded from a throwaway
/// stub that's never installed on an entry (`no_initial_load`).
fn runHeadlessServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    discovery: ?model_discovery.DiscoveryResult,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    max_resident_models: u32,
    max_resident_mem: u64,
    max_resident_mem_explicit: bool,
    idle_evict_secs: ?u32,
    kv_quant_config: transformer_mod.KVQuantConfig,
    pld: server_mod.PldDefaults,
    spec: SpecLoadFlags,
) !void {
    log.info("lesto {s} (headless - models load on demand)\n", .{VERSION});
    log.info("[args] serve: {s}:{d}\n", .{ host, port });

    var stub = try buildStubCpuState(allocator);
    defer freeStubCpuState(allocator, &stub);

    const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_explicit, max_resident_mem);
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{ max_resident_models, @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0 });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(allocator, io, discovery, max_resident_models, effective_max_resident_mem, idle_evict_secs);
    defer registry.deinit();

    // Carrier entry for LoadParams (required field), never loaded here
    // (`no_initial_load`). Prefer a discovered stub (so it's listed in
    // /v1/models); else a throwaway placeholder - headless is valid with empty
    // discovery.
    // No default is set, so a request that omits `model` gets a clean 503
    // until a model is loaded.
    var placeholder = model_registry_mod.LoadedModel{
        .allocator = allocator,
        .id = "",
        .path = "",
        .bytes_on_disk = null,
        .arch_hint = "",
        .config = null,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .drafter = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .unloaded,
        .error_name = null,
    };
    const carrier: *model_registry_mod.LoadedModel = blk: {
        var it = registry.entries.valueIterator();
        if (it.next()) |e| break :blk e.*;
        log.info("Headless: no models under --model-dir.\n", .{});
        break :blk &placeholder;
    };

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = carrier,
        .config = stub.config,
        .tok = stub.tok,
        .chat_config = stub.chat_config,
        .model_dir = "",
        .ctx_size = ctx_size,
        .no_initial_load = true,
        .warmup_eager = false,
        .no_drafter = spec.no_drafter,
        .draft_block_size = spec.draft_block_size,
        .draft_block_size_explicit = spec.draft_block_size_explicit,
        .kv_quant_config = kv_quant_config,
        // Seed the scheduler's prefix-cache config from the server globals so
        // on-demand (headless/discover-mode) loads get the SAME hot prefix
        // cache as a `--model` startup load. Previously hardcoded to 0, which
        // left `Scheduler.prefix_cache_capacity == 0` → every model loaded via
        // `ensureLoaded` skipped `HotPrefixCache` init → cross-turn KV reuse
        // was silently dead for the entire headless serving mode (the default
        // `serve` path). Mirrors the LoadParams built in `main()`.
        .prefix_cache_capacity = server_mod.prefix_cache_capacity,
        .prefix_cache_ram_enabled = server_mod.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = server_mod.prefix_cache_mem_bytes,
        .prefix_cache_mem_resolver = server_mod.prefixCacheMemForLoad,
        .prefix_cache_disk_bytes = server_mod.prefix_cache_disk_bytes,
        .tokenize_cache_entries = server_mod.tokenize_cache_entries,
        .metrics = server_mod.g_metrics,
    };

    try server_mod.serve(io, allocator, params, stub.config, host, port, .{
        .max_context_size = ctx_size,
        .request_timeout_sec = timeout,
        .default_reasoning_budget = reasoning_budget,
        .default_max_tokens = serve_default_max_tokens,
        .default_temperature = null,
        .default_top_p = null,
        .default_top_k = null,
        // Honor the whole --pld/--pld-draft-len/--pld-key-len trio in headless
        // mode. All three were hardcoded here, so none of them reached a
        // headless request - only an explicit per-request "enable_pld": true
        // did - while MLX Core's own UI describes Auto as "follow the server's
        // --pld setting". Headless is the mode the app ALWAYS launches, and it
        // always passes all three flags. Taking them as one `PldDefaults`
        // is what keeps the next edit from honoring one and dropping two.
        .default_enable_pld = pld.enable,
        .default_pld_draft_len = pld.draft_len,
        .default_pld_key_len = pld.key_len,
        .kv_attn_mode = .auto,
    });
}

/// Parse a size-style CLI argument: bare integer = bytes, suffix `KB`/`MB`/
/// `GB` (case-insensitive) multiplies by 1024^N, "0"/"off" = 0. Used by
/// `--prefix-cache-mem`; returns `error.InvalidSize` on malformed input.
/// `--config-overrides` takes a JSON OBJECT (every field it names is merged into
/// config.json). Parsed here purely to fail the launch on a typo; the merge
/// itself happens per-config in `model.parseConfigFromJson`.
fn configOverridesJsonValid(raw: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), raw, .{}) catch return false;
    return v == .object;
}

fn parseSizeArg(s: []const u8) !u64 {
    if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) return 0;
    var end: usize = s.len;
    var mult: u64 = 1;
    if (std.mem.endsWith(u8, s, "GB") or std.mem.endsWith(u8, s, "gb")) {
        end -= 2;
        mult = 1024 * 1024 * 1024;
    } else if (std.mem.endsWith(u8, s, "MB") or std.mem.endsWith(u8, s, "mb")) {
        end -= 2;
        mult = 1024 * 1024;
    } else if (std.mem.endsWith(u8, s, "KB") or std.mem.endsWith(u8, s, "kb")) {
        end -= 2;
        mult = 1024;
    } else if (std.mem.endsWith(u8, s, "B") or std.mem.endsWith(u8, s, "b")) {
        end -= 1;
    }
    if (end == 0) return error.InvalidSize;
    const n = std.fmt.parseInt(u64, s[0..end], 10) catch return error.InvalidSize;
    return n * mult;
}
