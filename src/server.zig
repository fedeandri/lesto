const std = @import("std");
const keep_warm = @import("keep_warm.zig");
const crt = @import("crt.zig");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const kv_quant_mod = @import("kv_quant.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const generation_settings = @import("generation_settings.zig");
const drafter_mod = @import("drafter.zig");
const chat_mod = @import("chat.zig");
const rp_mod = @import("reasoning_protocol.zig");
const token_mask = @import("token_mask.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const pld_index = @import("pld_index.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const model_registry_mod = @import("model_registry.zig");
const model_discovery = @import("model_discovery.zig");
const metrics = @import("status.zig");
const instr = @import("metrics.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
/// Global flag set by signal handler for graceful shutdown.
var shutdown_requested = std.atomic.Value(bool).init(false);
/// Number of live per-connection threads. Incremented before each spawn,
/// decremented when the thread's handler returns. On shutdown, `serve` waits
/// for this to reach 0 before returning (which runs `scheduler.deinit`) — a
/// detached conn thread still in `Scheduler.complete` would otherwise race
/// deinit's teardown of the slot queues into a use-after-free (SIGSEGV).
var active_conn_threads = std.atomic.Value(u32).init(0);
/// Set from main.zig before serve() is called when --metrics is on; null
/// otherwise. Gates the gauge-sampler thread and the /metrics + /metrics.json
/// routes. When null, `/metrics*` return 503, which the console reads as "metrics off".
pub var g_metrics: ?*instr.Metrics = null;
/// Optional global API key (`--api-key`). When set, every NON-LOOPBACK request
/// (i.e. from another machine over the network) except the `/health` probe and
/// CORS preflight requires the key: the OpenAI/Anthropic APIs AND the
/// index page + metrics panel/feed. Loopback (127.0.0.0/8, ::1) is TRUSTED and
/// exempt, so the local app + a local browser keep working with no credentials;
/// the key protects the surface that actually matters — network exposure
/// (`--host 0.0.0.0`). Accepted as `Authorization: Bearer <key>`, `x-api-key:
/// <key>` (Anthropic), HTTP Basic (browser pages — the key is the password), or
/// an `?api_key=` / `?key=` query param. null ⇒ fully open (default, unchanged).
pub var g_api_key: ?[]const u8 = null;

/// `--api-key-strict`: drop the loopback exemption, so the key is required
/// from 127.0.0.1 too (`/health` and CORS preflight stay open). For hosts
/// where localhost is NOT one trust domain — an embedding application that
/// generates a fresh key per start wants "only the process holding the key
/// can drive inference", and loopback exemption made any local process (and
/// any browser page via a simple no-preflight POST) a key holder. No effect
/// unless `--api-key` is set.
pub var g_api_key_strict: bool = false;

/// Tool-call ARGUMENT auto-correct. When true (default), parsed tool-call
/// arguments are coerced to the types the request's tool schema declares
/// (`chat.coerceToolArgsToSchema` — e.g. Python's `False` → JSON `false`, an
/// `edits` array shipped as a string → a real array). `--no-tool-autocorrect`
/// turns it OFF: arguments pass through exactly as the model emitted them (still
/// valid JSON — the parse-repair + safety net always run; this gates ONLY the
/// schema-type coercion, the opinionated layer). Off means a weak model's
/// mistyped value reaches the client verbatim, which strict clients reject.
pub var g_tool_autocorrect: bool = true;

/// Should boot print the open-bind warning? True only when serve mode is about
/// to listen on a non-loopback address the user never chose: no explicit
/// `--host` (the default is still 0.0.0.0). A future release flips the default to 127.0.0.1 — at which
/// point the host check silences this without a code change.
pub fn shouldWarnOpenBind(host_explicit: bool, host: []const u8) bool {
    if (host_explicit) return false;
    return !(std.mem.startsWith(u8, host, "127.") or
        std.mem.eql(u8, host, "::1") or
        std.mem.eql(u8, host, "localhost"));
}

test "shouldWarnOpenBind: warn only on an UNCHOSEN non-loopback bind" {
    // Default bind (0.0.0.0, nobody asked) → warn: a first-launch user is
    // serving whatever network the laptop joins.
    try std.testing.expect(shouldWarnOpenBind(false, "0.0.0.0"));
    // Someone who CHOSE the bind is not nagged: explicit --host (any value).
    try std.testing.expect(!shouldWarnOpenBind(true, "0.0.0.0"));
    // Loopback defaults never warn (the future 127.0.0.1 default).
    try std.testing.expect(!shouldWarnOpenBind(false, "127.0.0.1"));
    try std.testing.expect(!shouldWarnOpenBind(false, "localhost"));
    try std.testing.expect(!shouldWarnOpenBind(false, "::1"));
}

const io_util = @import("io_util.zig");
const model_settings_mod = @import("model_settings.zig");
const build_options = @import("build_options");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const nowMsMonotonic = io_util.nowMsMonotonic;
const Stopwatch = io_util.Stopwatch;

/// Tracks how long a streaming connection has been SILENT so the token loops
/// can emit a protocol-level keepalive before a client's idle-body timeout
/// fires. The mirror image of `generate.StallClock`: that one protects the
/// server from a wedged model, this one protects the client from a wedged-
/// looking socket.
///
/// It exists because every streaming surface buffers tokens while it might be
/// looking at a tool call or an unclosed thinking block — during a large
/// tool call (an agent one-shotting a whole file into `write_file`) the
/// handler produces no SSE output at all, sometimes for minutes. Node's
/// `fetch`/undici drops such a body after 300 s (`TypeError: terminated`).
pub const StreamHeartbeat = struct {
    /// MONOTONIC milliseconds (`io_util.nowMsMonotonic`) — an NTP step must not
    /// be able to stall the keepalive, nor spam one per token.
    last_write_ms: i64 = 0,
    interval_ms: i64 = Conn.STREAM_KEEPALIVE_MS,

    pub fn noteWrite(self: *StreamHeartbeat, now_ms: i64) void {
        self.last_write_ms = now_ms;
    }

    /// True once `interval_ms` has elapsed with no bytes handed to the socket.
    pub fn due(self: *const StreamHeartbeat, now_ms: i64) bool {
        return now_ms - self.last_write_ms >= self.interval_ms;
    }
};

/// Connection wrapper bundling a TCP stream with its `Io` and per-connection
/// reader/writer buffers. Replaces `std.net.Stream` in 0.16 — methods that took
/// a bare stream now take `*Conn` so the IO interface and buffers travel together.
pub const Conn = struct {
    stream: std.Io.net.Stream,
    io: std.Io,
    write_buf: [16 * 1024]u8,
    read_buf: [16 * 1024]u8,
    write_state: std.Io.net.Stream.Writer,
    read_state: std.Io.net.Stream.Reader,
    /// Silence tracker for the streaming keepalive. Stamped by every write
    /// path below, so it measures client-visible silence rather than "time
    /// since the last token" — a buffered tool call keeps producing tokens
    /// while emitting nothing. (`writeAllNoFlush` stamps too: every SSE emit
    /// path flushes before returning, so a stamp can't outrun the socket by
    /// more than one event.)
    heartbeat: StreamHeartbeat = .{},
    /// True once this connection has written a `text/event-stream` head: past it a generation
    /// failure can only be an SSE `error` event. Set only in `sendSseHeaders`.
    sse_headers_sent: bool = false,
    /// True once this connection has written a response head with a `Content-Length`: the body
    /// ends by its length, not at the close, so `close` need not wait for the peer.
    length_framed: bool = false,
    /// Agent named by the current request's User-Agent.
    client: instr.Client = .other,

    pub fn init(c: *Conn, stream: std.Io.net.Stream, io: std.Io) void {
        c.stream = stream;
        c.io = io;
        c.client = .other;
        c.write_state = stream.writer(io, &c.write_buf);
        c.read_state = stream.reader(io, &c.read_buf);
        c.sse_headers_sent = false;
        c.length_framed = false;
        c.heartbeat = .{ .last_write_ms = nowMsMonotonic(io) };
    }

    /// Replaces the previous request's client, so a request without a User-Agent never inherits it.
    pub fn noteRequestHeaders(c: *Conn, headers: []const u8) void {
        c.client = instr.Client.fromUserAgent(findHeaderValue(headers, "user-agent"));
    }

    /// True when the connection has been silent long enough that a streaming
    /// keepalive is due.
    pub fn keepaliveDue(c: *Conn) bool {
        return c.heartbeat.due(nowMsMonotonic(c.io));
    }

    pub fn writer(c: *Conn) *std.Io.Writer {
        return &c.write_state.interface;
    }

    pub fn reader(c: *Conn) *std.Io.Reader {
        return &c.read_state.interface;
    }

    pub fn writeAll(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        try c.writer().writeAll(data);
        try c.writer().flush();
    }

    pub fn writeAllNoFlush(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        try c.writer().writeAll(data);
    }

    pub fn flush(c: *Conn) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        try c.writer().flush();
    }

    /// Read up to buf.len bytes; may return fewer (HTTP-style short read). Returns 0 on EOF.
    /// Uses `readVec` because `readSliceShort` blocks until the buffer is full or EOF —
    /// wrong semantics for HTTP request parsing where we read until headers terminate.
    pub fn read(c: *Conn, buf: []u8) !usize {
        var bufs: [1][]u8 = .{buf};
        return c.reader().readVec(&bufs) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => |e| e,
        };
    }

    /// Waits up to `ms` for bytes to read; false when none came. Bytes the reader already
    /// buffered count as readable.
    fn awaitReadable(c: *Conn, ms: i64) bool {
        if (c.reader().bufferedLen() > 0) return true;
        if (ms <= 0) return false;
        return socketReadable(c.stream.socket.handle, @intCast(@min(ms, std.math.maxInt(i32)))) catch true;
    }

    /// Longest `close` waits for the peer to hang up before releasing the socket anyway.
    const CLOSE_WAIT_MS: i64 = 5 * 60 * 1000;

    /// A close-delimited body (SSE, NDJSON: no `Content-Length`) ends at the close, so this sends
    /// our FIN and holds the socket until the peer closes. A socket closed under a reader that
    /// still lags is dropped by the kernel with an RST once its FIN_WAIT_2 timer runs out (60 s
    /// on macOS), so the reader gets the whole body and then ECONNRESET, not EOF. A length-framed
    /// response closes at once: its reader knows where the body ends.
    pub fn close(c: *Conn) void {
        c.flush() catch {};
        if (c.length_framed) return c.stream.close(c.io);
        const fd = c.stream.socket.handle;
        _ = std.posix.system.shutdown(fd, std.posix.SHUT.WR);
        const deadline = nowMsMonotonic(c.io) + CLOSE_WAIT_MS;
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        var sink: [256]u8 = undefined;
        while (nowMsMonotonic(c.io) < deadline and !shutdown_requested.load(.acquire)) {
            const n = std.posix.poll(&fds, 500) catch break;
            if (n == 0) continue;
            // EOF or reset ends the wait; a byte from a pipelining client is dropped.
            if (std.c.recv(fd, &sink, sink.len, std.posix.MSG.DONTWAIT) <= 0) break;
        }
        c.stream.close(c.io);
    }

    /// Non-blocking probe: has the peer closed the connection? TCP send
    /// buffers absorb hundreds of SSE writes after the client FIN/RST so
    /// `writeAll` only fails seconds late; this surfaces the disconnect
    /// promptly so a long-running decode can be cancelled before the GPU
    /// burns more cycles.
    ///
    /// Uses `poll(timeout=0)` for HUP/ERR, and a zero-byte `recv(MSG_PEEK)`
    /// to disambiguate `POLLIN`-with-data (still alive, client sending) from
    /// `POLLIN`-with-FIN (peer closed, `recv` returns 0). Stray bytes from a
    /// pipelined client are not expected on an SSE response stream — if we
    /// see them, we conservatively treat the connection as live.
    /// Idle interval (ms) between keepalive probes while a streaming
    /// request waits on its first tokens (prefill). Short enough that an
    /// abandoned prefill is noticed and cancelled quickly and that clients
    /// never hit stream-idle timeouts; tiny traffic cost.
    pub const STREAM_KEEPALIVE_MS: i64 = 5000;

    pub fn peerClosed(c: *Conn) bool {
        const fd = c.stream.socket.handle;
        var fds = [_]std.posix.pollfd{.{
            .fd = fd,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const n = std.posix.poll(&fds, 0) catch return false;
        if (n == 0) return false;
        const revents = fds[0].revents;
        if ((revents & (std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL)) != 0) return true;
        if ((revents & std.posix.POLL.IN) == 0) return false;
        var peek_buf: [1]u8 = undefined;
        const flags: c_int = std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT;
        const r = std.c.recv(fd, &peek_buf, peek_buf.len, flags);
        if (r == 0) return true; // FIN: peer closed cleanly
        return false; // negative (EAGAIN) or data available → assume alive
    }
};

fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    shutdown_requested.store(true, .release);
}

/// Adaptive spec-decode gate threshold. Per-request, we score the prompt's
/// 3-gram repetition density; if `score < spec_gate_threshold` AND the user
/// did not explicitly set the flag, PLD/drafter are disabled for this request.
///
/// Empirically tuned (24-300 token prompts):
///   - "creative story" / "write an essay" prompts:                    0.000
///   - 82-token echo-heavy code rename:                                0.013
///   - 155-token JS-translation w/ many `add/sub/mul/div` repetitions: 0.128
/// The plan started from 0.15 but BPE tokenization fragments echo-heavy
/// content enough that even strong echo cases land in the 0.01–0.10 range.
/// 0.01 cleanly separates "any repetition at all" (PLD likely helps) from
/// "pure novel" (PLD overhead-only). Re-tune by sweeping `enable_pld` on/off
/// across an echo-heavy vs novel-content prompt set and picking the threshold
/// that maximizes (echo_speedup / novel_overhead).
const spec_gate_threshold: f32 = 0.01;

// Plan 05: drafter state moved to `LoadedModel` (per-model `drafter`,
// `drafter_block_size`, `drafter_path`). The previous module-level
// `default_drafter` / `default_enable_drafter` / `default_draft_block_size`
// / `global_drafter_path` singletons were removed; handlers now read
// these fields off the request's resolved `*LoadedModel` (`lm.drafter`,
// etc.). `default_enable_drafter` semantics (`drafter != null and !isMoe()`)
// is re-computed per-request as `lm.drafter != null and !config.isMoe()`.

/// Server-level configuration. Single source of truth for all process-wide
/// defaults that handlers might consult. Populated once by `serve()` from
/// its CLI args; read-only afterwards (no synchronization required, the
/// values don't change for the server's lifetime).
pub const KvAttnMode = enum { dense, fused, auto };

/// docs/kv-quant-perf.md Phase 2 — auto-mode crossover. µbench (Phase 0
/// table in the doc): the decode kernel is ~break-even with dense-mode reads
/// at 8K context and wins from ~16K on every measured geometry (48/8 and
/// 32/8 at hd 128, 16/8 at hd 256), so `auto` engages fused reads from 8K
/// prompt tokens. One constant to start (per the plan); the per-request
/// `kv_attn_mode` field and an explicit --kv-attn-mode outrank it.
pub const KV_ATTN_AUTO_CROSSOVER_TOKENS: usize = 8192;

/// Pure resolution core (unit-tested): explicit per-request choice >
/// server mode; `auto` = fused iff the request's EFFECTIVE cache scheme is
/// .affine and the prompt clears the crossover.
pub fn resolveKvAttnFusedPure(mode: KvAttnMode, explicit: ?bool, prompt_len: usize, scheme: kv_quant_mod.Scheme) bool {
    if (explicit) |e| return e;
    return switch (mode) {
        .dense => false,
        .fused => true,
        .auto => scheme == .affine and prompt_len >= KV_ATTN_AUTO_CROSSOVER_TOKENS,
    };
}

/// Wrapper reading the live server config + scheduler default scheme.
/// Exact decode (a DFlash drafter bound) reads every row through `row_attn`
/// over the dense view: the packed kernels neither take a draft tree's mask
/// nor give a serial step and a verify row the same bits.
fn resolveKvAttnFused(config: *const model_mod.ModelConfig, explicit: ?bool, prompt_len: usize, kv_override: ?transformer_mod.KVQuantConfig) bool {
    if (config.rowExactDecode()) return false;
    const scheme: kv_quant_mod.Scheme = (kv_override orelse configuredKvQuantFor(config)).scheme;
    return resolveKvAttnFusedPure(server_config.kv_attn_mode, explicit, prompt_len, scheme);
}

/// Per-request `kv_attn_mode` body field: "fused" | "dense" force the read
/// path; "auto" or an absent/unrecognized value falls back to the server
/// mode (the kv_quant field's tolerant precedent).
fn parseKvAttnExplicit(root: std.json.ObjectMap) ?bool {
    const v = root.get("kv_attn_mode") orelse return null;
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "fused")) return true;
            if (std.mem.eql(u8, s, "dense")) return false;
            return null;
        },
        else => return null,
    }
}

pub const ServerConfig = struct {
    /// Maximum context size (0 = unlimited). `--ctx-size N`.
    max_context_size: u32 = 0,
    /// Request timeout in seconds (0 = no timeout). `--timeout N`.
    request_timeout_sec: u32 = 300,
    /// Default reasoning budget in tokens (-1 = unlimited).
    /// `--reasoning-budget N`. Per-request body fields override.
    default_reasoning_budget: i32 = -1,
    /// Default PLD enabled state. Per-request `enable_pld` JSON overrides.
    default_enable_pld: bool = false,
    /// Maximum draft tokens proposed per PLD step.
    default_pld_draft_len: u32 = 5,
    /// N-gram match key length for PLD.
    default_pld_key_len: u32 = 3,
    /// docs/kv-quant-perf.md Phase 2: how quantized-KV requests READ the
    /// cache at decode. `--kv-attn-mode dense|fused|auto`; per-request
    /// `kv_attn_mode` body field ("dense"/"fused") outranks it. Only takes
    /// effect at `--kv-quant 4|8` (.affine cache scheme); other schemes
    /// always read dense. `auto` (the default) engages fused reads when the
    /// prompt is at/above the measured crossover — see `resolveKvAttnFused`.
    kv_attn_mode: KvAttnMode = .auto,
    /// Defaults for sampling fields the request OMITS, set by `--temp` /
    /// `--top-p` / `--top-k` in serve mode (the macOS app passes its Settings
    /// values so external clients like Claude Code — which send no sampling
    /// params at all — inherit them). null = flag not given; resolution falls
    /// through to the model's generation_config.json recommendation, then the
    /// hardcoded fallback. Request fields and generation-defaults rules outrank
    /// them (`resolveSampling`).
    default_temperature: ?f32 = null,
    default_top_p: ?f32 = null,
    default_top_k: ?u32 = null,
    /// Default `max_tokens` for a request that omits the field (`--max-tokens N` in serve mode;
    /// 0 = flag not given). Agent clients send nothing, and an omitted budget is the rest of
    /// the window, which the admission bill reserves for. Read through `launchMaxTokensDefault`.
    default_max_tokens: u32 = 0,
};

const Sampling = struct { temperature: f32, top_p: f32, top_k: u32, min_p: ?f32, repeat_penalty: f32, presence_penalty: f32, frequency_penalty: f32 };

/// Sampling for one request: a forced generation-defaults rule > request body > rule >
/// CLI launch flag > model generation_config.json > hardcoded fallback. An explicit request
/// value of 0 (greedy / disabled) is a value, not an omission.
fn resolveSampling(gen: generation_settings.Profile, root: std.json.ObjectMap, config: *const model_mod.ModelConfig) Sampling {
    return .{
        .temperature = gen.resolve(f32, .temperature, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_config.default_temperature orelse config.gen_temperature orelse 1.0),
        .top_p = gen.resolve(f32, .top_p, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_config.default_top_p orelse config.gen_top_p orelse 1.0),
        .top_k = gen.resolve(u32, .top_k, parseJsonTopKOpt(root, "top_k"), server_config.default_top_k orelse config.gen_top_k orelse 0),
        .min_p = gen.resolveOpt(f32, .min_p, parseJsonFloatOpt(root, "min_p", 0.0, 1.0), config.gen_min_p),
        .repeat_penalty = requestRepeatPenalty(root, gen),
        .presence_penalty = gen.resolve(f32, .presence_penalty, parseJsonFloatOpt(root, "presence_penalty", 0.0, 2.0), 0.0),
        .frequency_penalty = gen.resolve(f32, .frequency_penalty, parseJsonFloatOpt(root, "frequency_penalty", 0.0, 2.0), 0.0),
    };
}

/// A zero rule means remaining context; an omitted rule uses the surface's fallback.
fn generationMaxTokens(gen: generation_settings.Profile, v: ?std.json.Value, fallback: u32) u32 {
    const req = resolveRequestMaxTokens(v, 0);
    const cap = gen.resolve(u32, .max_tokens, if (req > 0) req else null, fallback);
    return if (cap == 0 and gen.rules.get(.max_tokens) != null) AUTO_MAX_TOKENS_SENTINEL else cap;
}

var g_generation_settings: model_settings_mod.Cache = .{ .path = "" };
var g_generation_settings_path: [std.fs.max_path_bytes]u8 = undefined;

/// A request's generation defaults: the model's rules over the global file's, re-read when
/// either file changes.
fn requestProfile(io: std.Io, lm: *const LoadedModel) generation_settings.Profile {
    return .overlay(g_generation_settings.generationDefaults(io, null), g_model_aliases.generationDefaults(io, lm.path));
}

/// parseJsonFloat variant that distinguishes "omitted / wrong type" (null)
/// from an explicit value, for fields whose default comes from the
/// resolution chain above.
fn parseJsonFloatOpt(root: std.json.ObjectMap, key: []const u8, min: f32, max: f32) ?f32 {
    const v = root.get(key) orelse return null;
    const raw: f32 = switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    return std.math.clamp(raw, min, max);
}

/// Optional top_k body-field parse (positive integer, capped at 1000).
/// null = omitted or unusable; explicit 0 means "disable top-k".
fn parseJsonTopKOpt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = root.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
        else => null,
    };
}

/// Live process-wide config. Mutated by `serve()` at startup from its
/// parameters; never written from a handler after that. Public so main.zig
/// can supply defaults (notably the per-request override CLI flags don't
/// flow through `serve()` arguments).
pub var server_config: ServerConfig = .{};

fn getTimeoutNs() u64 {
    if (server_config.request_timeout_sec == 0) return 0;
    return @as(u64, server_config.request_timeout_sec) * std.time.ns_per_s;
}

/// Plan 01 Phase 2 — continuous-batching scheduler. Always non-null in
/// serve mode (set by `serve()`); null only before `serve()` runs. Every
/// inference request handler routes through this; the scheduler's
/// inference thread is the single mlx-call site.
var global_scheduler: ?*scheduler_mod.Scheduler = null;

/// The boot's parsed `--kv-quant`, published by `serve()` before `Scheduler.init` so a
/// load-time bill sees it (the scheduler does not exist yet during the load; asking it billed
/// the SSD-first session at bf16, ~8.6 GB of idle allowance lost at 786k).
var configured_kv_quant: ?transformer_mod.KVQuantConfig = null;

/// The process-wide kv-quant config at any point: the scheduler's once it exists, the boot's
/// published value during the load, dense when no flag was given.
fn configuredKvQuant() transformer_mod.KVQuantConfig {
    if (global_scheduler) |sch| return sch.kv_quant_config;
    return configured_kv_quant orelse transformer_mod.KVQuantConfig.dense;
}

/// The width THIS model stores at: its `model-settings.json` override, else the process default.
pub fn configuredKvQuantFor(config: *const model_mod.ModelConfig) transformer_mod.KVQuantConfig {
    return config.kv_quant_override orelse configuredKvQuant();
}

/// The explicit context for THIS model: its `model-settings.json` `ctx_size`, else `--ctx-size`.
/// 0 = auto. Every reader of the manual context goes through here, never `server_config.max_context_size`.
pub fn manualContext(config: *const model_mod.ModelConfig) u32 {
    return if (config.ctx_override > 0) config.ctx_override else server_config.max_context_size;
}

/// Plan 05 — model registry. Always non-null in serve mode (set by
/// `serve()`); handleConnection resolves `model` body fields against this
/// per request via `ensureLoaded`/`release`.
var global_registry: ?*ModelRegistry = null;

/// Plan 05 — extract the `"model":"..."` value from a JSON request body
/// without doing a full parse. Returns a borrowed slice into `body` (valid
/// until the body buffer is freed) or null if the field is missing or
/// malformed. Used by `handleConnection` to route each POST to the right
/// LoadedModel before invoking the handler.
///
/// Cheap (single linear scan, ~tens of nanoseconds for typical bodies);
/// the per-handler full JSON parse runs immediately after and rejects any
/// truly malformed body, so robustness here is fine.
pub fn parseModelFromBody(body: []const u8) ?[]const u8 {
    var idx: usize = 0;
    while (idx < body.len) {
        const key_start = std.mem.indexOfPos(u8, body, idx, "\"model\"") orelse return null;
        // Must be JSON object key: preceded by `{` or `,` (skipping whitespace).
        var prev = key_start;
        while (prev > 0) {
            prev -= 1;
            const c = body[prev];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') continue;
            if (c == '{' or c == ',') break;
            // Not a top-level key — keep searching after this match.
            idx = key_start + 1;
            return parseModelFromBody(body[idx..]);
        }
        // Skip past the key + the colon.
        var pos = key_start + "\"model\"".len;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t')) pos += 1;
        if (pos >= body.len or body[pos] != ':') return null;
        pos += 1;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t' or body[pos] == '\n' or body[pos] == '\r')) pos += 1;
        if (pos >= body.len) return null;
        // Accept "string" only — null/numbers/etc fall back to default.
        if (body[pos] != '"') return null;
        pos += 1;
        const val_start = pos;
        while (pos < body.len and body[pos] != '"') {
            if (body[pos] == '\\' and pos + 1 < body.len) pos += 1; // skip escape
            pos += 1;
        }
        if (pos >= body.len) return null;
        return body[val_start..pos];
    }
    return null;
}

/// Every path `handleConnection` dispatches. Kept beside the chain rather than
/// derived from it because one question has to be answerable BEFORE a model is
/// resolved: does this endpoint exist at all?
///
/// Resolution runs ahead of dispatch, so without this an unknown path on a
/// server with no default model returned 503 "No default model configured"
/// instead of 404 — and a path's existence has nothing to do with model state.
/// The drift guard is a test that reads the dispatch chain out of this file.
const ROUTE_PATHS = [_][]const u8{
    "/detokenize",
    "/health",
    "/metrics",
    "/metrics.json",
    "/props",
    "/tokenize",
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/load-progress",
    "/v1/models",
    "/v1/models/rescan",
};

/// Is `path` an endpoint this server serves at all (any method)?
fn routeExists(path: []const u8) bool {
    for (ROUTE_PATHS) |p| if (std.mem.eql(u8, path, p)) return true;
    return false;
}

/// A coding agent keeps a long history: the chat cap is generous.
pub const max_request_bytes: usize = 512 * 1024 * 1024;

/// Per-route request-body cap. An agent's chat history is large, so every route
/// shares the one generous cap.
pub fn maxRequestBytesFor(target: []const u8) usize {
    _ = target;
    return max_request_bytes;
}

/// The 413 names BOTH numbers it compared (context-overflow-400 class): a
/// refusal that only names the cap reads as "this should have worked".
fn payloadTooLargeMessage(buf: []u8, got: usize, cap: usize) []const u8 {
    const mb = 1024 * 1024;
    return std.fmt.bufPrint(buf, "Request body too large: {d} MB exceeds this endpoint's {d} MB limit", .{
        got / mb + @intFromBool(got % mb != 0), cap / mb,
    }) catch "Request body too large";
}

pub const RequestModel = union(enum) {
    /// A registered id, or "" / "lesto" / "mlx-serve" (the default; "mlx-serve" is the old name), passed through.
    id: []const u8,
    /// Not registered: dispatch serves the default so SDK names like "gpt-4" work.
    unknown_name,
    /// An absolute path no entry lives at.
    unknown_path,
};

/// Model Settings aliases, parsed once and re-read when the file changes.
var g_model_aliases_path: [std.fs.max_path_bytes]u8 = undefined;
var g_model_aliases: model_settings_mod.Cache = .{ .path = "" };

/// Which model a request's `model` field names: an id, a path, a Model Settings
/// alias. Dispatch reads this. The registry is
/// NOT locked here.
fn resolveRequestModelId(registry: *ModelRegistry, aliases: *model_settings_mod.Cache, id: []const u8) RequestModel {
    if (id.len == 0 or std.mem.eql(u8, id, "lesto") or std.mem.eql(u8, id, "mlx-serve")) return .{ .id = id };
    if (registry.peek(id)) |e| return .{ .id = e.id };
    // A path is never an SDK marketing name, and a default-model answer to it
    // would serve a pack that failed to load with another model's weights.
    if (std.fs.path.isAbsolute(id)) {
        if (registry.peekByPath(id)) |e| return .{ .id = e.id };
        return .unknown_path;
    }
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (aliases.pathForAlias(registry.io, id, &buf)) |p| if (registry.peekByPath(p)) |e| return .{ .id = e.id };
    return .unknown_name;
}

/// The requested model id from a JSON request body.
///
/// Every consumer of the requested id must go through `parseModelFromBody`, so every
/// reader agrees about which model a request names.
/// `?model=<id>` for bodiless GETs (`/props`), percent-decoded into `buf`.
fn queryModel(buf: []u8, raw_path: []const u8) ?[]const u8 {
    return queryDecoded(buf, raw_path, "model");
}

/// A non-empty query parameter, percent-decoded into `buf`.
fn queryDecoded(buf: []u8, raw_path: []const u8, name: []const u8) ?[]const u8 {
    const q = raw_path[(std.mem.indexOfScalar(u8, raw_path, '?') orelse return null) + 1 ..];
    return percentDecoded(buf, pairValue(q, '&', name) orelse return null);
}

/// The raw value of `name=` in a `sep`-separated list: a query string, a form body, a Cookie header.
fn pairValue(list: []const u8, sep: u8, name: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, list, sep);
    while (it.next()) |raw| {
        const kv = std.mem.trim(u8, raw, " ");
        if (kv.len <= name.len + 1 or !std.mem.startsWith(u8, kv, name) or kv[name.len] != '=') continue;
        return kv[name.len + 1 ..];
    }
    return null;
}

fn percentDecoded(buf: []u8, v: []const u8) ?[]const u8 {
    if (v.len > buf.len) return null;
    @memcpy(buf[0..v.len], v);
    return std.Uri.percentDecodeInPlace(buf[0..v.len]);
}

/// Module-level capacity for plan 03 hot prefix cache. main.zig writes this
/// from `--prefix-cache-entries N` before calling `serve()`.
// Default 32: agent clients (Claude Code, the app's agent loop) interleave
// requests from several conversation roots — main thread, subagents, title
// generation — and a single-entry cache gets its long system-prompt prefix
// evicted by every interleaved request, forcing a full re-prefill each
// turn. The count cap is pure retention metadata; the byte budget
// (`--prefix-cache-mem`, default 2 GB) is what actually bounds memory,
// evicting LRU entries by size. 0 disables.
pub var prefix_cache_capacity: u32 = 32;
/// `--no-prefix-cache-ram` keeps the disk tier eligible without idle RAM retention.
pub var prefix_cache_ram_enabled: bool = true;

/// Wave 1.B — hot prefix cache memory budget. The cache evicts LRU entries on
/// commit until `current_kv_bytes + new_bytes <= prefix_cache_mem_bytes`. The
/// default (2 GB) is generous for one or two long conversations on a Gemma 4
/// E4B-sized model and tiny relative to total wired-limit budget; tune via
/// `--prefix-cache-mem <N>{GB,MB}`. 0 disables the byte budget (count cap
/// from `--prefix-cache-entries` still applies). On the RAM-first arm an
/// unset budget grows to one session at the working context (`defaultPrefixCacheAsk`).
pub var prefix_cache_mem_bytes: u64 = PREFIX_CACHE_MEM_DEFAULT;
pub const PREFIX_CACHE_MEM_DEFAULT: u64 = 2 * 1024 * 1024 * 1024;
/// Set by `--prefix-cache-mem`: an operator's number is used as given, even when it equals the default.
pub var prefix_cache_mem_explicit = false;

/// The hot-cache ask when nobody named one: one session at the working context, never under
/// 2 GB. Below one session the cache keeps only a prefix of the longest conversations, the
/// ones whose reuse saves the most prefill. Any other ask (an operator's, an embedder's) stands.
/// Bytes one cached session at `ctx_tokens` holds: its KV and state.
pub fn oneSessionEntryBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ctx_tokens: u64) u64 {
    return sessionBytesPerToken(config, kv_bits) *| ctx_tokens;
}

pub fn defaultPrefixCacheAsk(requested: u64, explicit: bool, session_kv: u64) u64 {
    if (explicit or requested != PREFIX_CACHE_MEM_DEFAULT) return requested;
    return @max(requested, session_kv);
}

/// What the hot cache was actually given for the loaded model, after `clampedPrefixCacheMem`.
/// Every post-load reserve reads it through `resolvedPrefixCacheMem()`. Atomic: written on the
/// inference thread, read from connection threads. `maxInt(u64)` is the unresolved sentinel,
/// not 0 (`--prefix-cache-mem 0` under SSD-first publishes a real zero).
const HOT_CACHE_MEM_UNRESOLVED: u64 = std.math.maxInt(u64);
var hot_cache_mem_resolved = std.atomic.Value(u64).init(HOT_CACHE_MEM_UNRESOLVED);

/// The hot-cache byte budget every post-load reserve must bill: the clamp's answer once loaded, the raw ask before.
pub fn resolvedPrefixCacheMem() u64 {
    if (prefix_cache_capacity == 0 or !prefix_cache_ram_enabled) return 0;
    const v = hot_cache_mem_resolved.load(.monotonic);
    return if (v != HOT_CACHE_MEM_UNRESOLVED) v else prefix_cache_mem_bytes;
}

/// Publish the clamp's answer (inference thread, end of a load); `clearResolvedPrefixCacheMem`
/// resets it on unload so the next model does not inherit this one's budget.
pub fn publishResolvedPrefixCacheMem(v: u64) void {
    hot_cache_mem_resolved.store(v, .monotonic);
}

pub fn clearResolvedPrefixCacheMem() void {
    hot_cache_mem_resolved.store(HOT_CACHE_MEM_UNRESOLVED, .monotonic);
}

/// SSD tier for the hot prefix cache (`--prefix-cache-disk`). Committed KV
/// prefixes persist as chunked safetensors under
/// `~/.mlx-serve/kv-cache/<model-fingerprint>` and are restored across RAM
/// evictions AND server restarts instead of recomputed — a cold 30-50 s
/// long-context TTFT becomes a bounded SSD read. LRU-evicted to this byte
/// budget.
///
/// DEFAULT OFF (`0`): the tier can hold gigabytes of KV on disk, so it's opt-in
/// — pass `--prefix-cache-disk <n>{KB,MB,GB}` to enable (10 GB is a sensible
/// value). The Swift app exposes this as a Settings toggle, default off.
pub var prefix_cache_disk_bytes: u64 = 0;

/// `--shutdown-persist-secs`: on SIGTERM, finish the SSD slices the idle flush has not landed,
/// for at most this long after the signal. 0 turns it off.
pub var shutdown_persist_secs: u32 = 45;

/// PLD request defaults carried as ONE value, so a `ServerConfig` builder
/// cannot thread the enable bit and forget the two lengths beside it.
///
/// That is exactly how this broke: `runHeadlessServe` hand-rolled all three
/// as `false`/`5`/`3` literals, so no `--pld*` flag reached a headless
/// request. Threading `enable` alone fixed a third of it and left
/// `--pld-draft-len` / `--pld-key-len` silently dropped — and headless is
/// the mode the Swift app ALWAYS launches (`--serve --model-dir`, no
/// `--model`), passing all three flags on every boot. Single chokepoint for
/// every ServerConfig builder.
///
/// Serve paths whose decode never routes through the PLD-capable generator
/// (media gen) take `.off`: the field is dead weight there,
/// and saying so once beats five hand-written `false`s that read like a
/// decision but drift like a typo.
pub const PldDefaults = struct {
    enable: bool,
    draft_len: u32,
    key_len: u32,

    /// Embedded-engine and media-gen serve paths — PLD is unreachable.
    pub const off: PldDefaults = .{ .enable = false, .draft_len = 5, .key_len = 3 };

    /// Text-gen serve paths: the CLI's values verbatim, no reinterpretation.
    /// `--no-pld` arrives as `enable == false` and the lengths ride along
    /// unused, so flipping PLD back on later cannot resurrect stale numbers.
    pub fn fromCli(enable: bool, draft_len: u32, key_len: u32) PldDefaults {
        return .{ .enable = enable, .draft_len = draft_len, .key_len = key_len };
    }
};

test "PldDefaults: CLI lengths survive alongside the enable bit" {
    // The regression: `enable` reached headless while the lengths stayed at
    // the 5/3 literals. All three travel together or the class is back.
    const cli = PldDefaults.fromCli(true, 8, 4);
    try std.testing.expect(cli.enable);
    try std.testing.expectEqual(@as(u32, 8), cli.draft_len);
    try std.testing.expectEqual(@as(u32, 4), cli.key_len);

    // --no-pld carries its lengths unchanged rather than snapping to defaults.
    const disabled = PldDefaults.fromCli(false, 8, 4);
    try std.testing.expect(!disabled.enable);
    try std.testing.expectEqual(@as(u32, 8), disabled.draft_len);

    // The non-text serve paths stay pinned to the historical literals.
    try std.testing.expect(!PldDefaults.off.enable);
    try std.testing.expectEqual(@as(u32, 5), PldDefaults.off.draft_len);
    try std.testing.expectEqual(@as(u32, 3), PldDefaults.off.key_len);
}

test "PldDefaults: ServerConfig built from it reports the CLI values" {
    // Pins the wiring, not just the struct: a ServerConfig fed from
    // `fromCli` must expose the CLI numbers on the exact fields every
    // request path and the boot banner read.
    const cfg = ServerConfig{
        .default_enable_pld = PldDefaults.fromCli(true, 8, 4).enable,
        .default_pld_draft_len = PldDefaults.fromCli(true, 8, 4).draft_len,
        .default_pld_key_len = PldDefaults.fromCli(true, 8, 4).key_len,
    };
    try std.testing.expect(cfg.default_enable_pld);
    try std.testing.expectEqual(@as(u32, 8), cfg.default_pld_draft_len);
    try std.testing.expectEqual(@as(u32, 4), cfg.default_pld_key_len);
}

/// Iteration 2 (perf-plan Phase 4 #3): LRU capacity of the per-LoadedModel
/// chat-template tokenize cache. 0 disables (every request re-renders +
/// re-tokenizes, restoring pre-Iteration-2 behavior). Default 4 is small
/// because real chat conversations mutate the messages list every turn;
/// the cache catches warm-reuse benches + repeated agent probes without
/// hoarding token buffers across a long session.
pub var tokenize_cache_entries: u32 = 4;

/// Plan 01 — continuous batching: maximum concurrent in-flight requests sharing
/// the inference thread's batched-decode pass. Set via `--max-concurrent N`.
/// Default 1 = legacy single-slot behavior (no scheduler engagement, every
/// existing test bit-identical). Values >1 require Phase 2's scheduler wiring
/// (handler refactor + per-connection threads), which the `serve()` startup
/// guard checks before enabling.
pub var max_concurrent: u32 = 1;

/// Issue #117: operator ceiling on per-input embedding length, in TOKENS.
/// 0 = auto (bound only by the loaded model's declared context window). Set
/// via `--embedding-max-length N`. Over-limit inputs earn a structured 400
/// naming the input index, its token count and the effective limit — never a
/// silent truncation (the Python-server `max_length=512` class). Applies to
/// the embedding route (/v1/embeddings, which funnels into `handleEmbeddings`).
pub var embedding_max_length: u32 = 0;

/// Effective per-input embedding token ceiling: the tighter of the operator
/// flag and the model's declared window (0 on either side = no bound from
/// that side; 0 result = unbounded). A flag above the model's window clamps
/// to the window: a 512-position model never accepts a 4096 override.
pub fn embedEffectiveLimit(flag: u32, model_max: u32) u32 {
    if (flag == 0) return model_max;
    if (model_max == 0) return flag;
    return @min(flag, model_max);
}

/// Overflow 400 body text: names the offending input INDEX and both counts,
/// because the client can only fix what it can see (the contextOverflowMessage
/// principle). bufPrint failure falls back to the bare sentence rather than
/// sending no body (the media-gen fixed-buffer class).
pub fn embedOverflowMessage(buf: []u8, index: usize, tokens: usize, limit: u32) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "Input at index {d} exceeds the maximum embedding input length: {d} tokens given, {d} allowed",
        .{ index, tokens, limit },
    ) catch "Input exceeds the maximum embedding input length";
}

// Plan 05: the model id moved to `LoadedModel.id`. Handlers read it off `lm`.
// The `global_model_id` singleton was removed. The `discovered_models`
// slice was also removed — `/v1/models` iterates `registry.entries` directly.

/// Decode a slice of token IDs to bytes through the loaded tokenizer. Used by
/// the request handlers' streaming + final-decode paths.
fn decodeTokens(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    ids: []const u32,
    strip_leading_space: bool,
) ![]u8 {
    _ = lm;
    return tok.decode(allocator, ids, strip_leading_space);
}

/// Assistant-history reasoning field on an incoming chat message:
/// `reasoning_content` (our own SSE/vLLM field, what pi rounds-trips) with
/// `reasoning` (the vLLM request spelling a template may read first) as
/// the fallback. Non-empty strings only — an empty string would render an
/// empty <think></think> block, which is exactly the nothink signature this
/// field exists to avoid.
fn messageReasoningFromObj(obj: std.json.ObjectMap) ?[]const u8 {
    inline for (.{ "reasoning_content", "reasoning" }) |key| {
        if (obj.get(key)) |v| {
            if (v == .string and v.string.len > 0) return v.string;
        }
    }
    return null;
}

/// True when the rendered generation prompt ends inside a template-opened
/// think block (some templates render `…assistant\n<think>\n` when thinking is
/// on). Decodes the last few prompt tokens — cheap, engine-agnostic, and
/// independent of the tokenize cache. Drives the unclosed-think split policy
/// in `chat.splitThinkBlock` so a length-truncated thought never leaks into
/// visible content.
fn promptOpensThink(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
) bool {
    if (prompt_ids.len == 0) return false;
    const n = @min(prompt_ids.len, 8);
    const tail = decodeTokens(allocator, lm, tok, prompt_ids[prompt_ids.len - n ..], false) catch return false;
    defer allocator.free(tail);
    return chat_mod.promptTailOpensThink(tail);
}

/// One token id for `text`, or null when the tokenizer spells it in pieces.
fn atomicTokenId(allocator: std.mem.Allocator, tok: *const Tokenizer, text: []const u8) ?u32 {
    const ids = tok.encode(allocator, text) catch return null;
    defer allocator.free(ids);
    return if (ids.len == 1) ids[0] else null;
}

/// The in-stream thinking bound needs atomic think markers (bare `<think>`
/// family).
fn thinkMarkersAtomic(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    if (lm.transformer == null) return false;
    return atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER) != null;
}

/// The thinking-budget recipe: the model is told time is up, then the
/// closer is committed and the answer follows.
const THINK_BOUND_EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

/// No budget (< 0) still arms the bound at an unreachable limit: it exists to
/// close a thought the loop guard convicts (`scheduler.loopRecoveryAction`).
fn thinkBoundBudget(budget: i32) u32 {
    return if (budget < 0) std.math.maxInt(u32) else @intCast(budget);
}

/// Arm a decode-time thinking bound for this request, or null when it cannot
/// close a thought (thinking off, markers not atomic). `forced` is
/// allocated; the caller frees it after generation.
fn armThinkBound(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, enable_thinking: bool, budget: i32) ?generate_mod.ThinkBound {
    if (!enable_thinking or lm.transformer == null) return null;
    const closer = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER) orelse {
        if (budget >= 0) log.info("  reasoning budget {d}: think markers are not atomic tokens, delivery cap only\n", .{budget});
        return null;
    };
    const opener = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_OPENER);
    const opened = promptOpensThink(allocator, lm, tok, prompt_ids);
    if (opener == null and !opened) return null;

    const stop_ids = tok.encode(allocator, THINK_BOUND_EARLY_STOP) catch return null;
    defer allocator.free(stop_ids);
    const sep_ids = tok.encode(allocator, "\n\n") catch return null;
    defer allocator.free(sep_ids);
    const forced = allocator.alloc(u32, stop_ids.len + 1 + sep_ids.len) catch return null;
    @memcpy(forced[0..stop_ids.len], stop_ids);
    forced[stop_ids.len] = closer;
    @memcpy(forced[stop_ids.len + 1 ..], sep_ids);
    if (budget >= 0) log.info("  reasoning budget {d}: enforced in-stream\n", .{budget});
    return .{ .budget = thinkBoundBudget(budget), .opener_id = opener, .closer_id = closer, .forced = forced, .in_think = opened };
}

/// Arm the forced tool-call opener for a `tool_choice` that obliges a call, or
/// null when the dialect has no opener we can spell or the thought cannot be
/// tracked. `forced` is allocated; the caller frees it after generation.
fn armToolForce(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, forced_tool: ?chat_mod.ForcedTool) ?generate_mod.ToolForce {
    const ft = forced_tool orelse return null;
    if (lm.transformer == null) return null;
    const cc = lm.chat_config orelse return null;
    const text = (chat_mod.forcedToolOpener(allocator, cc.chat_template, ft) catch return null) orelse return null;
    defer allocator.free(text);
    const opened = promptOpensThink(allocator, lm, tok, prompt_ids);
    const closer = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER);
    if (opened and closer == null) return null;
    const opener = tok.encode(allocator, text) catch return null;
    defer allocator.free(opener);
    const sep = tok.encode(allocator, "\n\n") catch return null;
    defer allocator.free(sep);
    const head: usize = if (opened) 1 + sep.len else 0;
    const forced = allocator.alloc(u32, head + opener.len) catch return null;
    if (opened) {
        forced[0] = closer.?;
        @memcpy(forced[1..head], sep);
    }
    @memcpy(forced[head..], opener);
    log.info("  tool_choice: call opener enforced in-stream\n", .{});
    return .{ .closer_id = closer, .forced = forced, .opener_at = head, .in_think = opened };
}

/// A generated tag needs one unambiguous literal spelling in the template.
/// Dynamic or conflicting spellings retain the unsupported-format fallback.
fn templateThinkOpener(template: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    var offset: usize = 0;
    while (std.mem.indexOfPos(u8, template, offset, "<think")) |start| {
        const len = chat_mod.thinkOpenTagLenAt(template[start..]) orelse return null;
        const tag = template[start..][0..len];
        if (found) |previous| {
            if (!std.mem.eql(u8, previous, tag)) return null;
        }
        found = tag;
        offset = start + len;
    }
    return found;
}

/// Resolve a reasoning protocol from the rendered prompt and template. Tokenizer
/// indexes and exact recovery suffixes are cached on the loaded model.
fn resolveReasoningProtocol(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    prompt_ids: []const u32,
    proto_out: *rp_mod.Protocol,
    allow_reasoning: bool,
) bool {
    if (prompt_ids.len == 0 or lm.transformer == null) return false;
    const tail_n = @min(prompt_ids.len, 64);
    const tail = decodeTokens(allocator, lm, tok, prompt_ids[prompt_ids.len - tail_n ..], false) catch return false;
    defer allocator.free(tail);
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");

    const template = chat_config.chat_template;
    const kind: rp_mod.Kind = if (std.mem.indexOf(u8, template, "<|channel>") != null)
        .gemma
    else
        .bare_think;
    proto_out.* = .{ .kind = kind };
    const proto = proto_out;
    if (kind != .bare_think) {
        proto.configureChannels();
        if (!proto.startFromPrompt(tail)) return false;
    } else {
        switch (chat_mod.promptThinkTailClass(trimmed)) {
            .bare => {
                proto.kind = .bare_think;
                if (!proto.setCloser(chat_mod.BARE_THINK_CLOSER)) return false;
            },
            .none => {
                const opener = templateThinkOpener(template) orelse return false;
                if (!std.mem.eql(u8, opener, rp_mod.BARE_THINK_OPENER)) return false;
                proto.kind = .bare_think;
                var closer_buf: [rp_mod.MAX_MARKER_BYTES]u8 = undefined;
                const close = std.fmt.bufPrint(&closer_buf, "</{s}", .{opener[1..]}) catch return false;
                if (!proto.setOpener(opener) or !proto.setCloser(close)) return false;
            },
        }
    }

    var final_header: [rp_mod.MAX_MARKER_BYTES]u8 = undefined;
    if (!allow_reasoning and !proto.finalOnly(&final_header)) return false;

    const closer = lm.reasoningMarker(allocator, io, proto.closerText()) catch return false;
    proto.closer_atomic = closer.atomic;
    proto.closer_suffix = closer.suffix;
    proto.closer_suffix_buf = closer.tokens;
    proto.closer_span_candidates = closer.close_candidates;
    if (closer.atomic == null) {
        const run = closer.suffix[0];
        if (run.len == 0 or !proto.setForced(closer.tokens[run.offset..][0..run.len])) return false;
    }
    if (proto.openerText()) |text| {
        const opener = lm.reasoningMarker(allocator, io, text) catch return false;
        proto.opener_atomic = opener.atomic;
        proto.opener_suffix = opener.suffix;
        proto.opener_suffix_buf = opener.tokens;
        proto.opener_candidates = opener.open_candidates;
        if (opener.atomic == null and opener.open_candidates.len == 0) return false;
    }
    for (proto.headers[0..proto.header_len]) |*rule| {
        const marker = lm.reasoningMarker(allocator, io, rule.text) catch return false;
        if (marker.suffix[0].len == 0) return false;
        rule.text = marker.text;
        rule.candidates = marker.open_candidates;
        rule.suffix = marker.suffix;
        rule.tokens = marker.tokens;
    }
    // Structural special ids remain separate from JSON token bytes.
    if (proto.closer_atomic) |id| {
        proto.specials[proto.special_len] = .{ .id = id, .text = closer.text };
        proto.special_len += 1;
    }
    for ([_][]const u8{
        "<|start|>",         "<|channel|>",          "<|message|>",      "<|channel>",      "<channel|>",
        "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>", "<|end|>",
    }) |text| {
        if (tok.specialTokenId(text)) |id| {
            proto.specials[proto.special_len] = .{ .id = id, .text = text };
            proto.special_len += 1;
        }
    }
    return true;
}

/// Default when an OpenAI-style client omits `max_tokens` /
/// `max_completion_tokens` / `max_output_tokens`. OpenAI semantics: omitted
/// means "generate until EOS, bounded by the context window" — NOT a small
/// fixed cap. (The old 256 default silently broke agent clients like pi:
/// every thinking-enabled turn hit `length` mid-reasoning, so no tool call or
/// answer ever came back.) The sentinel is huge so the downstream
/// `clampMaxTokens` resolves it to the remaining context; only when the context
/// is genuinely unknown (0) do we fall back to a finite 4096.
///
/// Takes the EFFECTIVE context, not `server_config.max_context_size`: that
/// global is set only by `--ctx-size`, so under auto-context this used to
/// return 4096 — an omitted-max_tokens client silently capped at 4096 tokens,
/// truncating any large tool call. Same class as the 256 default it replaced.
fn omittedMaxTokensDefault(effective_ctx: u32) u32 {
    return omittedMaxTokensDefaultWith(effective_ctx, launchMaxTokensDefault());
}

/// Pure core of the above with the launch default (`--max-tokens N`, 0 = unset) passed in.
/// The flag outranks both fallbacks.
fn omittedMaxTokensDefaultWith(effective_ctx: u32, launch_default: u32) u32 {
    if (launch_default > 0) return launch_default;
    return if (effective_ctx > 0) AUTO_MAX_TOKENS_SENTINEL else 4096;
}

/// The "no budget named, peg it to the window" sentinel, in one place; `clampMaxTokens` reduces it to `context - prompt`.
const AUTO_MAX_TOKENS_SENTINEL: u32 = std.math.maxInt(u32) / 4;

/// The one read of `ServerConfig.default_max_tokens`. 0 = flag not given.
pub fn launchMaxTokensDefault() u32 {
    return server_config.default_max_tokens;
}

/// Resolve a request's `max_tokens` (or its aliases) to an effective cap.
/// Absent OR `<= 0` means **auto**: peg generation to the remaining context
/// window via `auto_default` (the `omittedMaxTokensDefault` sentinel, which
/// `clampMaxTokens` then reduces to `context - prompt`). The app's "Auto"
/// setting sends 0 (or omits the field); both land on the context-pegged budget
/// instead of a fixed ceiling. A positive integer is an explicit cap.
fn resolveRequestMaxTokens(v: ?std.json.Value, auto_default: u32) u32 {
    const val = v orelse return auto_default;
    return switch (val) {
        .integer => |i| if (i > 0) @intCast(@min(i, std.math.maxInt(u32))) else auto_default,
        else => auto_default,
    };
}

/// A request's `seed`; negative is the "random" convention and means unseeded.
fn parseRequestSeed(v: ?std.json.Value) ?u64 {
    const val = v orelse return null;
    return switch (val) {
        .integer => |i| if (i < 0) null else @as(u64, @intCast(i)),
        else => null,
    };
}

/// A client-supplied JSON integer narrowed to i32 by clamping, never by `@intCast`.
fn clampJsonI32(i: i64) i32 {
    return @intCast(std.math.clamp(i, std.math.minInt(i32), std.math.maxInt(i32)));
}

/// Who decided the effective `max_tokens`: the client, the `--max-tokens` launch default, or nobody (auto).
const MaxTokensOrigin = enum { client, launch_default, auto };

/// The origin of the number `resolveRequestMaxTokens` returns, from the same two inputs.
fn maxTokensOrigin(requested: ?std.json.Value, launch_default: u32) MaxTokensOrigin {
    if (requested) |val| switch (val) {
        .integer => |i| if (i > 0) return .client,
        else => {},
    };
    return if (launch_default > 0) .launch_default else .auto;
}

/// Widest string `describeMaxTokens` writes: a 10-digit budget plus
/// " (launch default)".
const MAX_TOKENS_DESC_LEN: usize = 27;

/// Render the request log's `max_tokens=` field: auto prints as `auto` (or `auto (N)` for the
/// finite unknown-context fallback), a launch-filled budget names its source, a client's number
/// prints bare. The sentinel used to print as a client asking for a billion tokens.
fn describeMaxTokens(buf: []u8, value: u32, origin: MaxTokensOrigin) []const u8 {
    return switch (origin) {
        .client => std.fmt.bufPrint(buf, "{d}", .{value}) catch "?",
        .launch_default => std.fmt.bufPrint(buf, "{d} (launch default)", .{value}) catch "?",
        .auto => if (value == AUTO_MAX_TOKENS_SENTINEL)
            "auto"
        else
            std.fmt.bufPrint(buf, "auto ({d})", .{value}) catch "auto",
    };
}

/// Does the generation budget warrant the "budget squeezed" warning? The auto sentinel is
/// excluded: it is always more than 4x anything that remains.
fn maxTokensBudgetSqueezed(max_tokens: u32, remaining: u32) bool {
    if (max_tokens == AUTO_MAX_TOKENS_SENTINEL) return false;
    return remaining < max_tokens / 4;
}

/// vLLM `ignore_eos`: when true, EOS and stop tokens do not end the reply —
/// only `max_tokens` (or the request timeout) does. Stop sequences are a
/// separate field and stay active.
fn requestEosSlice(config: *const model_mod.ModelConfig, root: std.json.ObjectMap) []const u32 {
    if (root.get("ignore_eos")) |v| {
        if (v == .bool and v.bool) return &.{};
    }
    return config.eosTokenSlice();
}

test "requestEosSlice: ignore_eos swaps the EOS list for empty" {
    var config = model_mod.ModelConfig{};
    config.addEosToken(151645);
    config.addEosToken(151643);
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const parsed_off = try std.json.parseFromSlice(std.json.Value, arena_state.allocator(), "{\"ignore_eos\": true}", .{});
    defer parsed_off.deinit();
    try std.testing.expectEqual(@as(usize, 0), requestEosSlice(&config, parsed_off.value.object).len);
    const parsed_on = try std.json.parseFromSlice(std.json.Value, arena_state.allocator(), "{\"ignore_eos\": false}", .{});
    defer parsed_on.deinit();
    try std.testing.expectEqual(@as(usize, 2), requestEosSlice(&config, parsed_on.value.object).len);
    const parsed_absent = try std.json.parseFromSlice(std.json.Value, arena_state.allocator(), "{\"max_tokens\": 64}", .{});
    defer parsed_absent.deinit();
    try std.testing.expectEqual(@as(usize, 2), requestEosSlice(&config, parsed_absent.value.object).len);
}

/// The auto budget's own tightness question: under a quarter of the window left.
fn autoBudgetWindowTight(remaining: u32, effective_ctx: u32) bool {
    return remaining < effective_ctx / 4;
}

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when the cut-off text salvaged a partial tool
/// call (the truncated-opener recovery in chat.parseToolCalls) — clients key
/// their truncation recovery on it (the app's chunk-and-retry nudge fires on
/// finish_reason "length" + tool calls present). Overriding to "tool_calls"
/// hides the cut: the client executes a half-argument call, blames the model
/// ("you sent no content"), and the model re-emits the same doomed mega-call.
fn toolCallFinishReason(pre_parse: []const u8) []const u8 {
    return if (std.mem.eql(u8, pre_parse, "length")) "length" else "tool_calls";
}

/// The `tool_calls` array of one streamed chat-completions delta: name, id and
/// the full arguments in ONE delta, every string escaped.
fn streamToolCallDelta(allocator: std.mem.Allocator, index: usize, id: []const u8, name: []const u8, arguments: []const u8) ![]u8 {
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments);
    defer allocator.free(esc_args);
    return std.fmt.allocPrint(allocator,
        \\[{{"index":{d},"id":"{s}","type":"function","function":{{"name":{s},"arguments":{s}}}}}]
    , .{ index, id, esc_name, esc_args });
}

test "streamed tool-call delta escapes the name and arguments (#748)" {
    const allocator = std.testing.allocator;
    const name = "read_file\n</parameter \"x\" \\";
    const args = "{\"path\":\"a\\nb\"}";
    const delta = try streamToolCallDelta(allocator, 0, "call_1_0", name, args);
    defer allocator.free(delta);
    try std.testing.expect(std.mem.indexOfScalar(u8, delta, '\n') == null);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, delta, .{});
    defer parsed.deinit();
    const function = parsed.value.array.items[0].object.get("function").?.object;
    try std.testing.expectEqualStrings(name, function.get("name").?.string);
    try std.testing.expectEqualStrings(args, function.get("arguments").?.string);
}

/// A repetition-loop cut may land inside an otherwise recognizable tool call.
/// Never emit that buffer as executable work: unlike a genuine max-token cut,
/// this intentional stop must not ask clients to recover by compacting/retrying.
fn shouldParseToolCalls(finish_details: ?[]const u8) bool {
    const d = finish_details orelse return true;
    return !std.mem.eql(u8, d, "repetition_loop");
}

/// The `finish_details` object emitted BESIDE `finish_reason` on a choice,
/// or "" when there is nothing to say. Comes with its leading comma so call
/// sites splice it straight into the choice literal.
///
/// The loop guard uses `finish_reason: "stop"`; the sibling preserves why the
/// server stopped without overloading "length" (which makes agents run output-
/// or context-exhaustion recovery). OpenAI's deprecated `finish_details` is the
/// closest precedent, and conforming clients ignore keys they don't know.
fn finishDetailsField(reason: []const u8, details: ?[]const u8) []const u8 {
    const d = details orelse return "";
    // Every emitter can override the slot's reason after the fact. Gate on the
    // reason actually being emitted so a cause never accompanies a genuine
    // length exhaustion, tool completion, or client disconnect.
    if (!std.mem.eql(u8, reason, "stop")) return "";
    // One known value today; a switch here keeps an unknown string from
    // reaching the wire as an unescaped literal.
    if (std.mem.eql(u8, d, "repetition_loop")) return ",\"finish_details\":{\"type\":\"repetition_loop\"}";
    return "";
}

/// Tokens a client should be shown after a loop cut: everything before the
/// degenerate span. `start` is an index into the SAME emitted-token sequence,
/// but it is computed from the generator's own list, so it is clamped rather
/// than trusted — a mismatch must degrade to "emit everything", never to a
/// slice out of bounds.
fn loopTrimmedIds(ids: []const u32, start: ?usize) []const u32 {
    const s = start orelse return ids;
    if (s >= ids.len) return ids;
    return ids[0..s];
}

/// Debug-only corpus-harvest aid (`MLX_SERVE_RAW_DUMP_FILE`). Appends ONE framed
/// record per tools request: the DECLARED TOOLS SCHEMA and the raw pre-parse text
/// together, written at the one site where both are in scope. Correlating them
/// any other way is unsound — the debug log truncates lines at 16 KB (so a large
/// request body never re-parses, and a harvester silently pairs a model output
/// with some EARLIER request's schema), and concurrent requests interleave.
///
/// Framing is by byte COUNT, never a delimiter: both payloads are arbitrary bytes
/// and any sentinel can occur inside them. libc `write(2)` like log.zig's sink
/// (callers here carry no `Io`); O_APPEND keeps conn threads from interleaving.
fn appendRawToolDump(path: []const u8, tools_json: ?[]const u8, text: []const u8) void {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len == 0 or path.len >= path_buf.len) return;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const path_z: [*:0]const u8 = @ptrCast(&path_buf);

    const fd = crt.openWrite(path_z, false);
    if (fd < 0) return;
    defer crt.close(fd);

    const tools = tools_json orelse "";
    var hdr: [96]u8 = undefined;
    const h = std.fmt.bufPrint(&hdr, "\n===MLX_RAW_DUMP tools={d} raw={d}===\n", .{ tools.len, text.len }) catch return;
    _ = crt.write(fd, h);
    if (tools.len > 0) _ = crt.write(fd, tools);
    if (text.len > 0) _ = crt.write(fd, text);
}

/// The ONE tool-call extraction path for every HTTP surface: parse, fall back to
/// bare-JSON inference, then coerce the arguments to the schema the request
/// declared. Every surface must go through this — a site that calls
/// `chat_mod.parseToolCalls` directly silently reintroduces the value-spelling
/// type-inference bug (a boolean param arriving as the string "False") on that
/// surface alone, which no output-equality test can see.
fn parseToolCallsForRequest(
    allocator: std.mem.Allocator,
    text: []const u8,
    tools_json: ?[]const u8,
    /// OpenAI `parallel_tool_calls` (Anthropic spelling:
    /// !tool_choice.disable_parallel_tool_use). false = the client accepts AT
    /// MOST ONE call per response, so the chokepoint keeps the first parsed
    /// call and drops the rest (the model re-issues them next round after the
    /// first result). Deliberately NOT gated by --no-tool-autocorrect:
    /// client-requested behavior, not a repair heuristic.
    allow_parallel: bool,
) !?[]chat_mod.ParsedToolCall {
    var parsed_calls = try chat_mod.parseToolCalls(allocator, text);
    // Heuristically-inferred raw-JSON calls must name a DECLARED tool — a
    // truncated data object ({"name": "George Washington", …}) is not a call.
    // Deliberately NOT gated on g_tool_autocorrect: this corrects our own
    // heuristic's false positive, not the model's output.
    if (parsed_calls) |c| {
        if (tools_json) |tj| parsed_calls = try chat_mod.filterInferredBySchema(allocator, c, tj);
    }
    var calls = parsed_calls orelse
        (if (tools_json) |tj| try chat_mod.inferBareJsonToolCalls(allocator, text, tj) else null) orelse
        return null;
    if (!allow_parallel and calls.len > 1) {
        log.info("  parallel_tool_calls=false: clamping {d} parsed calls to the first\n", .{calls.len});
        for (calls[1..]) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        const kept = calls[0];
        allocator.free(calls);
        const one = try allocator.alloc(chat_mod.ParsedToolCall, 1);
        one[0] = kept;
        calls = one;
    }
    if (g_tool_autocorrect) {
        if (tools_json) |tj| {
            // Order matters: put a buried required param back where the schema
            // says it lives BEFORE coercing types, so the hoisted value is
            // type-checked like any other top-level arg.
            try chat_mod.hoistMisplacedRequiredParams(allocator, calls, tj);
            try chat_mod.coerceToolArgsToSchema(allocator, calls, tj);
        }
    }
    return calls;
}

/// Listen with SO_REUSEADDR (a restart can rebind over TIME_WAIT) but WITHOUT
/// SO_REUSEPORT, which std sets beside it: with it a second instance binds the
/// same port and silently takes the connections. The kernel checks the flag on
/// the socket already bound, so clearing it after bind is enough.
fn listenExclusive(io: std.Io, addr: std.Io.net.IpAddress) !std.Io.net.Server {
    var server = try addr.listen(io, .{ .reuse_address = true });
    errdefer server.deinit(io);
    try std.posix.setsockopt(server.socket.handle, std.posix.SOL.SOCKET, std.posix.SO.REUSEPORT, &std.mem.toBytes(@as(c_int, 0)));
    return server;
}

/// Whether `handle` turns readable (a connection to accept, bytes, or EOF) within `timeout_ms`.
fn socketReadable(handle: std.Io.net.Socket.Handle, timeout_ms: i32) !bool {
    var fds = [_]std.posix.pollfd{.{ .fd = handle, .events = std.posix.POLL.IN, .revents = 0 }};
    return try std.posix.poll(&fds, timeout_ms) > 0;
}

/// Start the HTTP server on the given host and port.
///
/// `cfg` carries all process-wide defaults (context size, timeouts, PLD
/// defaults, reasoning budget). It's copied into the module-level
/// `server_config` before the listen loop starts.
/// Dotted-decimal IPv4 (`0.0.0.0` and anything unparsable read as zeros).
fn parseIp4(host: []const u8) [4]u8 {
    var out: [4]u8 = .{ 0, 0, 0, 0 };
    if (std.mem.eql(u8, host, "0.0.0.0")) return out;
    var parts = std.mem.splitScalar(u8, host, '.');
    var idx: usize = 0;
    while (parts.next()) |part| {
        if (idx >= 4) break;
        out[idx] = std.fmt.parseInt(u8, part, 10) catch 0;
        idx += 1;
    }
    return out;
}

/// Answers on the bound socket while the startup load runs, before the accept loop takes it:
/// `/health` is 503 `loading` (clients keep waiting), a loopback client reads `/v1/load-progress`,
/// anything else is 503. One connection at a time; nothing here touches the scheduler.
const StartupResponder = struct {
    server: *std.Io.net.Server,
    registry: *ModelRegistry,
    allocator: std.mem.Allocator,
    io: std.Io,
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn start(self: *StartupResponder) void {
        self.thread = std.Thread.spawn(.{}, run, .{self}) catch null;
    }

    fn stop(self: *StartupResponder) void {
        self.done.store(true, .release);
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    fn run(self: *StartupResponder) void {
        while (!self.done.load(.acquire)) {
            const ready = socketReadable(self.server.socket.handle, 100) catch continue;
            // Once done, a waiting connection is the accept loop's.
            if (!ready or self.done.load(.acquire)) continue;
            const accepted = self.server.accept(self.io) catch continue;
            var conn: Conn = undefined;
            Conn.init(&conn, accepted, self.io);
            defer conn.close();
            self.answer(&conn) catch {};
        }
    }

    fn answer(self: *StartupResponder, conn: *Conn) !void {
        const raw = (try readRequest(self.allocator, conn)) orelse return;
        defer self.allocator.free(raw.buf);
        const line_end = std.mem.indexOf(u8, raw.buf[0..raw.len], "\r\n") orelse return;
        var words = std.mem.splitScalar(u8, raw.buf[0..line_end], ' ');
        _ = words.next();
        const raw_path = words.next() orelse return;
        const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;
        if (std.mem.eql(u8, path, "/health")) return sendResponse(conn, "503 Service Unavailable", "application/json", "{\"status\":\"loading\"}");
        if (std.mem.eql(u8, path, "/v1/load-progress") and peerIsLoopback(conn)) {
            const body = try loadProgressBody(self.allocator, self.io, self.registry);
            defer self.allocator.free(body);
            return sendResponse(conn, "200 OK", "application/json", body);
        }
        return sendErrorResponse(self.allocator, conn, "503 Service Unavailable", "loading", "The model is still loading", 503);
    }
};

pub fn serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    /// Phase A1: model load happens on the scheduler's inference thread. The
    /// caller (main.zig) is responsible only for CPU-side setup (config parse,
    /// tokenizer load, EOS resolution, chat-template load); everything that
    /// touches mlx is done by `Scheduler.init` before this fn proceeds.
    load_params: scheduler_mod.LoadParams,
    config: *const model_mod.ModelConfig,
    host: []const u8,
    port: u16,
    cfg: ServerConfig,
) !void {
    server_config = cfg;
    // Before `Scheduler.init`: that call is the model load and every load-time bill inside it
    // asks for the KV width.
    configured_kv_quant = load_params.kv_quant_config;
    defer configured_kv_quant = null;

    // ── Phase A1: spin up the scheduler. Its inference thread does the
    //    Transformer/vision/drafter load, JIT compile, and warmup before
    //    `Scheduler.init` returns. From this point on, mlx ops are bound
    //    to the inference thread's GPU stream.
    // Bound before the startup load, which can take minutes:
    // until it ends, `StartupResponder` answers in place of the accept loop below.
    const ip_addr: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = parseIp4(host), .port = port } };
    var server = listenExclusive(io, ip_addr) catch |err| {
        if (err == error.AddressInUse) log.err("Port {d} is already in use - another lesto instance may be running.\n", .{port});
        return err;
    };
    defer server.deinit(io);
    var startup: StartupResponder = .{ .server = &server, .registry = load_params.registry, .allocator = allocator, .io = io };
    if (!load_params.no_initial_load) startup.start();
    defer startup.stop();

    var scheduler = try scheduler_mod.Scheduler.init(
        allocator,
        io,
        load_params,
        max_concurrent,
    );
    defer scheduler.deinit();
    startup.stop();
    global_scheduler = scheduler;
    defer global_scheduler = null;
    // The inference thread's evict-or-refuse hook (#353); the scheduler has no server import.
    scheduler_mod.prefill_admission_fits = &prefillFitsNow;
    defer scheduler_mod.prefill_admission_fits = null;
    scheduler_mod.prefill_admission_numbers = &prefillBillNumbersNow;
    defer scheduler_mod.prefill_admission_numbers = null;
    scheduler_mod.prefill_request_chunk = &requestPrefillChunkNow;
    defer scheduler_mod.prefill_request_chunk = null;
    // The published budget is per model; retire it whenever the scheduler drops the cache.
    scheduler_mod.hot_cache_budget_invalidate = &clearResolvedPrefixCacheMem;
    defer scheduler_mod.hot_cache_budget_invalidate = null;
    scheduler_mod.prefill_admission_refused_log = &logPrefillRefusal;
    defer scheduler_mod.prefill_admission_refused_log = null;
    // Asked by the prefill loop: the width of the next chunk, re-priced at every boundary.
    scheduler_mod.prefill_chunk_adapt = &adaptivePrefillWidthNow;
    defer scheduler_mod.prefill_chunk_adapt = null;
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    // Whether the hook above can move anything for a given model (the installs are process-wide, so presence is not a gate).
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;

    // Gauge sampler: samples instantaneous system + queue state every 2 s and
    // writes the metrics gauges. Only runs when --metrics is on. Trivial cost
    // (a few non-blocking syscalls + one brief queue_mu lock per 2 s); never on
    // the per-token decode path.
    // LIFO defer ordering: `stop` is registered AFTER `join` so it runs FIRST
    // (setting the flag), letting the thread exit before join() blocks. Do NOT
    // gate on scheduler.shutdown here — deinit() (which sets that flag) is the
    // earliest defer and runs LAST, after the join, so the loop would never see
    // it; the dedicated `sampler_stop` flag avoids that LIFO deadlock.
    var sampler_thread: ?std.Thread = null;
    var sampler_stop = std.atomic.Value(bool).init(false);
    if (g_metrics) |m| {
        const ctx = GaugeSamplerCtx{
            .metrics = m,
            .scheduler = scheduler,
            .stop = &sampler_stop,
        };
        sampler_thread = try std.Thread.spawn(.{}, gaugeSamplerLoop, .{ctx});
        log.info("Prometheus metrics: ENABLED — GET http://{s}:{d}/metrics (live panel at GET /)\n", .{ host, port });
    }
    // LIFO: join deferred first → runs second. stop deferred second → runs first.
    defer if (sampler_thread) |t| t.join();
    defer sampler_stop.store(true, .monotonic);

    // Idle eviction (--idle-evict-secs). Same lifetime shape as the sampler,
    // and never in `Scheduler.deinit` — see `IdleEvictCtx.stop`.
    var idle_evict_thread: ?std.Thread = null;
    var idle_evict_stop = std.atomic.Value(bool).init(false);
    if (scheduler.registry.idle_evict_secs) |secs| {
        idle_evict_thread = try std.Thread.spawn(.{}, idleEvictLoop, .{IdleEvictCtx{
            .scheduler = scheduler,
            .window_ms = @as(i64, secs) * 1000,
            .stop = &idle_evict_stop,
        }});
        log.info("[registry] idle eviction ON — unloading models idle for {d}s\n", .{secs});
    }
    defer if (idle_evict_thread) |t| t.join();
    defer idle_evict_stop.store(true, .monotonic);

    global_registry = scheduler.registry;
    defer global_registry = null;
    g_model_aliases.alloc = scheduler.registry.allocator;
    g_model_aliases.path = model_settings_mod.defaultPath(&g_model_aliases_path);
    g_generation_settings.alloc = scheduler.registry.allocator;
    g_generation_settings.path = model_settings_mod.homeFile(&g_generation_settings_path, "generation-settings.json");

    // Prefix cache lives on LoadedModel and may retain idle prefixes in RAM, SSD, or both.
    if (scheduler.hot_prefix_cache != null) {
        if (!prefix_cache_ram_enabled and prefix_cache_disk_bytes > 0) {
            log.info("Prefix cache: SSD ONLY (RAM retention disabled, disk cap={d:.1} MB)\n", .{
                @as(f64, @floatFromInt(prefix_cache_disk_bytes)) / (1024.0 * 1024.0),
            });
        } else if (resolvedPrefixCacheMem() > 0) {
            const cap_mb = @as(f64, @floatFromInt(resolvedPrefixCacheMem())) / (1024.0 * 1024.0);
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap={d:.1} MB)\n", .{ prefix_cache_capacity, cap_mb });
        } else {
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap=unlimited)\n", .{prefix_cache_capacity});
        }
    } else if (prefix_cache_capacity > 0) {
        log.info("Hot prefix cache: requested capacity={d} but disabled by scheduler\n", .{prefix_cache_capacity});
    }

    // `--max-concurrent N` is honored when the model can ride the batched
    // decode kernel (pure-attention, no recurrent-state/MoE/encoder). Hybrid / MoE /
    // encoder architectures clamp to 1 — they need single-slot serial
    // because the batched kernel doesn't model their state.
    // `--max-concurrent` sizes the submit queue; requests decode together at
    // any value. Whether they share ONE forward is the model's answer, printed
    // at every value so a default-1 boot does not read as "one at a time".
    if (scheduler_mod.configBatchesDecode(config)) {
        log.info("Concurrency: --max-concurrent={d}, batched decode on\n", .{max_concurrent});
        if (prefix_cache_capacity > 0 and prefix_cache_capacity < max_concurrent) prefix_cache_capacity = max_concurrent;
    } else {
        log.info("Concurrency: --max-concurrent={d}, batched decode off (arch: {s}); concurrent requests interleave serially\n", .{ max_concurrent, config.model_type });
    }
    // Install signal handlers for graceful shutdown.
    const sigact = std.posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
    std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);

    // Freeze the auto-context NOW, at startup, while the model is freshly
    // loaded and nothing else has taken RAM. Clients read this number once
    // (pi/opencode bake it into a config file) and budget against it for the
    // whole session, so it must not drift with system load. `--ctx-size` wins.
    const pinned = pinAutoContext(@constCast(config));
    if (manualContext(config) > 0) {
        log.info("Context size: {d} tokens (manual)\n", .{manualContext(config)});
    } else {
        const memory_ctx = computeMemoryContext(config);
        const memory_allows = safeAutoContext(memory_ctx);
        const rope_ctx = config.contextCap();
        if (rope_ctx > 0 and pinned >= rope_ctx) {
            // The checkpoint's own maximum binds; memory had room to spare.
            log.info("Context size: {d} tokens (auto: the model's maximum; memory would allow {d}) [pinned]\n", .{ pinned, memory_allows });
        } else {
            log.info("Context size: {d} tokens (auto: {d}% of the {d}-token memory ceiling, reserving headroom) [pinned]\n", .{ pinned, auto_ctx_safety_pct, memory_ctx });
        }
    }

    if (server_config.request_timeout_sec > 0) {
        log.info("Request timeout: {d}s\n", .{server_config.request_timeout_sec});
    }
    const model_ctx = config.contextCap();
    if (model_ctx > 0) {
        log.info("Model context length: {d} tokens\n", .{model_ctx});
    }
    if (server_config.default_reasoning_budget >= 0) {
        log.info("Reasoning budget: {d} tokens\n", .{server_config.default_reasoning_budget});
    } else {
        log.info("Reasoning budget: unlimited\n", .{});
    }
    if (launchMaxTokensDefault() > 0) {
        log.info("default max_tokens for omitted requests: {d}\n", .{launchMaxTokensDefault()});
    }
    if (server_config.default_enable_pld) {
        log.info("PLD speculative decoding: ENABLED (draft_len={d}, key_len={d}; default for new requests)\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
    }
    if (scheduler.dflash != null) {
        log.info("DFlash speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    } else if (scheduler.drafter != null and scheduler.dflash == null) {
        log.info("Drafter speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    }
    log.info("\nServer listening on http://{s}:{d}\n", .{ host, port });
    if (g_api_key != null) {
        log.info("API key auth: ENABLED for non-loopback requests (localhost is trusted; /health stays open)\n", .{});
    }
    log.info("  GET  /\n", .{});
    log.info("  GET  /health\n", .{});
    log.info("  GET  /props\n", .{});
    log.info("  GET  /v1/models\n", .{});
    log.info("  POST /v1/chat/completions\n", .{});
    log.info("  POST /v1/embeddings\n", .{});
    log.info("  POST /tokenize\n", .{});
    log.info("  POST /detokenize\n\n", .{});

    // Print system metrics once at startup
    const rss = metrics.getAppRssMb();
    if (rss >= 1024) {
        log.info("RSS: {d}.{d}G  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss / 1024,             (rss % 1024) * 10 / 1024,
            metrics.getSysMemPct(), metrics.getCpuPct(),
            metrics.getGpuPct(),
        });
    } else {
        log.info("RSS: {d}M  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss, metrics.getSysMemPct(), metrics.getCpuPct(), metrics.getGpuPct(),
        });
    }

    while (!shutdown_requested.load(.acquire)) {
        // Poll with 1-second timeout so we can check shutdown flag
        const ready = socketReadable(server.socket.handle, 1000) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("poll error: {}\n", .{err});
            continue;
        };
        if (!ready) continue; // timeout, re-check shutdown flag
        if (shutdown_requested.load(.acquire)) break;

        const accepted_stream = server.accept(io) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("accept error: {}\n", .{err});
            continue;
        };

        // When the scheduler is engaged AND the model is pure-attention,
        // Spawn a per-connection thread so HTTP I/O for one request can
        // overlap with another request's generation. mlx ops (forward
        // passes, eval, etc.) live exclusively on the scheduler's inference
        // thread, so the connection thread NEVER touches `xfm.s` directly —
        // it parses HTTP, encodes the prompt, calls `scheduler.submit`,
        // reads tokens via `slot.waitNext`, and writes the response.
        const args = allocator.create(ConnThreadArgs) catch |err| {
            log.err("conn thread args alloc failed: {}\n", .{err});
            accepted_stream.close(io);
            continue;
        };
        args.* = .{
            .allocator = allocator,
            .accepted_stream = accepted_stream,
            .io = io,
        };
        // Count the thread before spawning so the shutdown drain can't miss a
        // thread that's about to start; the thread decrements on return.
        _ = active_conn_threads.fetchAdd(1, .acq_rel);
        const conn_thread = std.Thread.spawn(.{}, handleConnectionThread, .{args}) catch |err| {
            log.err("spawn conn thread failed: {}\n", .{err});
            _ = active_conn_threads.fetchSub(1, .acq_rel);
            accepted_stream.close(io);
            allocator.destroy(args);
            continue;
        };
        conn_thread.detach();
    }

    // Shutdown ordering (prevents a SIGSEGV in Scheduler.complete): the accept
    // loop has exited. Cancel every in-flight slot so its connection thread
    // unblocks from waitNext and runs its `defer complete(...)`, then WAIT for
    // all connection threads to finish before returning — `serve`'s caller runs
    // `scheduler.deinit` on return, which frees the slot queues that an
    // in-flight `complete()` is still touching. The inference thread is still
    // alive here (deinit joins it only after we return), so cancelled slots
    // settle promptly. Bounded so a wedged thread can't hang shutdown forever.
    // The SSD flush's deadline starts here, once. The drain keeps its own bound: cutting it short
    // saves nothing (the join below waits out the inference thread's current chunk anyway) and
    // lets a late `complete()` race the queues `deinit` frees.
    if (global_scheduler) |sch| {
        scheduler_mod.armShutdownDeadline(&sch.shutdown_deadline_ms, nowMsMonotonic(io), shutdown_persist_secs);
        sch.cancelAllInFlight();
    }
    {
        var waited_ms: u64 = 0;
        while (active_conn_threads.load(.acquire) > 0 and waited_ms < 30_000) {
            crt.sleepMs(20);
            waited_ms += 20;
        }
        const remaining = active_conn_threads.load(.acquire);
        if (remaining > 0)
            log.warn("shutdown: {d} connection thread(s) still active after {d}ms — proceeding\n", .{ remaining, waited_ms });
    }

    log.info("\nShutting down gracefully...\n", .{});
}

/// Per-connection thread arguments. Heap-allocated so the spawned thread
/// owns its lifetime; freed in `handleConnectionThread` after the handler
/// returns. The accepted stream is moved into a stack-allocated `Conn`
/// inside the thread.
const ConnThreadArgs = struct {
    allocator: std.mem.Allocator,
    accepted_stream: std.Io.net.Stream,
    io: std.Io,
};

fn handleConnectionThread(args: *ConnThreadArgs) void {
    keep_warm.touch();
    // Decrement LAST (declared first → runs after every other defer), so the
    // shutdown drain in `serve` only sees this thread leave once it has fully
    // returned from `complete()` and closed its socket.
    defer _ = active_conn_threads.fetchSub(1, .acq_rel);
    var conn: Conn = undefined;
    Conn.init(&conn, args.accepted_stream, args.io);
    defer conn.close();
    handleConnection(args.allocator, &conn) catch |err| {
        switch (err) {
            error.WriteFailed, error.ReadFailed => {
                // error.WriteFailed/ReadFailed collapses BrokenPipe + ConnectionResetByPeer
                // + other low-level errors. Surface the actual cause from write_state.err /
                // read_state.err so debug logs distinguish "client hung up" from real bugs.
                if (err == error.WriteFailed) {
                    if (conn.write_state.err) |we| {
                        log.debug("  -> client disconnected (write: {s})\n", .{@errorName(we)});
                    } else {
                        log.debug("  -> client disconnected (write)\n", .{});
                    }
                } else {
                    if (conn.read_state.err) |re| {
                        log.debug("  -> client disconnected (read: {s})\n", .{@errorName(re)});
                    } else {
                        log.debug("  -> client disconnected (read)\n", .{});
                    }
                }
            },
            else => log.err("  -> error: {}\n", .{err}),
        }
    };
    args.allocator.destroy(args);
}

/// Deepest JSON nesting a request may carry. Tool schemas and template kwargs are walked and
/// re-serialized recursively on a connection thread's stack: 50000 levels overflowed it.
const MAX_JSON_DEPTH = 4096;

/// Deepest `[`/`{` nesting in `body`, brackets inside strings skipped. A plain byte scan, so it
/// also bounds bodies the JSON parser would reject.
fn jsonNestingDepth(body: []const u8) usize {
    var depth: usize = 0;
    var max: usize = 0;
    var in_str = false;
    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        const c = body[i];
        if (in_str) {
            if (c == '\\') i += 1 else if (c == '"') in_str = false;
            continue;
        }
        switch (c) {
            '"' => in_str = true,
            '[', '{' => {
                depth += 1;
                max = @max(max, depth);
            },
            ']', '}' => depth -|= 1,
            else => {},
        }
    }
    return max;
}

const RawRequest = struct {
    /// Owned; `buf[0..len]` is the head plus as much of the body as arrived.
    buf: []u8,
    len: usize,
    header_end: usize,
};

/// Reads one request: the head (up to 16 KB), then its `Content-Length` body. Null when no
/// complete head arrived, the client stalled, or the body is over the route's cap (the 413 is
/// already sent).
/// Bytes past the declared body are dropped: one request per connection.
fn readRequest(allocator: std.mem.Allocator, stream: *Conn) !?RawRequest {
    return readRequestWithin(allocator, stream, REQUEST_READ_LIMIT_MS);
}

/// A client gets this long between bytes of its request, and in all for its head: an idle or
/// dripping connection otherwise holds its thread forever.
const REQUEST_READ_LIMIT_MS: i64 = 30_000;

fn readRequestWithin(allocator: std.mem.Allocator, stream: *Conn, limit_ms: i64) !?RawRequest {
    const head_deadline = nowMsMonotonic(stream.io) + limit_ms;
    var hdr_buf: [16 * 1024]u8 = undefined;
    var total_read: usize = 0;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;

    while (total_read < hdr_buf.len) {
        if (!stream.awaitReadable(head_deadline - nowMsMonotonic(stream.io))) return null;
        const n = try stream.read(hdr_buf[total_read..]);
        if (n == 0) break;
        total_read += n;

        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| {
            header_end_pos = he + 4;
            content_length = findContentLength(hdr_buf[0..he]);
            break;
        }
    }
    if (header_end_pos == 0) return null;

    const total_size = header_end_pos +| (content_length orelse 0);
    // The cap is per ROUTE, so peek the path off the request line we already
    // have - every body takes the chat cap.
    const req_path = blk: {
        const line_end = std.mem.indexOf(u8, hdr_buf[0..header_end_pos], "\r\n") orelse break :blk "";
        var it = std.mem.splitScalar(u8, hdr_buf[0..line_end], ' ');
        _ = it.next();
        break :blk it.next() orelse "";
    };
    const max_request_size = maxRequestBytesFor(req_path);
    if (total_size > max_request_size) {
        var msg_buf: [128]u8 = undefined;
        const msg = payloadTooLargeMessage(&msg_buf, total_size, max_request_size);
        log.warn("[http] 413 {s}: {s}\n", .{ logSafePath(req_path), msg });
        try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
        // Closing with body bytes unread resets the connection, and the client
        // can lose the 413 before reading it: read what is left, up to the chat cap.
        var left = @min(total_size, max_request_bytes) -| total_read;
        while (left > 0 and stream.awaitReadable(limit_ms)) {
            const n = stream.read(hdr_buf[0..@min(left, hdr_buf.len)]) catch break;
            if (n == 0) break;
            left -= n;
        }
        return null;
    }

    // curl asks before a body over 1 MB and otherwise waits a second for the answer.
    if (total_read < total_size) if (findHeaderValue(hdr_buf[0 .. header_end_pos - 4], "expect")) |v| {
        if (std.ascii.eqlIgnoreCase(v, "100-continue")) {
            try stream.writeAll("HTTP/1.1 100 Continue\r\n\r\n");
            try stream.flush();
        }
    };

    const buf = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buf);
    total_read = @min(total_read, total_size);
    @memcpy(buf[0..total_read], hdr_buf[0..total_read]);
    while (total_read < total_size) {
        if (!stream.awaitReadable(limit_ms)) {
            allocator.free(buf);
            return null;
        }
        const n = try stream.read(buf[total_read..total_size]);
        // The client hung up mid-body (an aborted upload): there is no request to answer.
        if (n == 0) {
            allocator.free(buf);
            return null;
        }
        total_read += n;
    }
    return .{ .buf = buf, .len = total_read, .header_end = header_end_pos };
}

/// Collapse JSON's slash escapes: the optional `\/` to `/` and `\\` to `\`. Swift's
/// JSONSerialization (and PHP's json_encode) escape every slash, so an id's org prefix
/// arrives as `org\/gemma…` from the app while the registry stores the canonical form,
/// and a Windows path id arrives as
/// `C:\\Users…`. Returns the input verbatim (zero-copy) when there is nothing to collapse or
/// the scratch buffer is too small.
pub fn unescapeJsonSlashes(buf: []u8, s: []const u8) []const u8 {
    if ((std.mem.indexOf(u8, s, "\\/") == null and std.mem.indexOf(u8, s, "\\\\") == null) or s.len > buf.len) return s;
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len and (s[i + 1] == '/' or s[i + 1] == '\\')) i += 1;
        buf[n] = s[i];
        n += 1;
    }
    return buf[0..n];
}

fn handleConnection(
    allocator: std.mem.Allocator,
    stream: *Conn,
) !void {
    // Plan 05: resolve which model this request targets. The registry was
    // set up in `serve()`; per-POST routing happens after we read the body
    // and parse out the optional `"model"` field. For Phase C we use the
    // default model for everything (only one is loaded), but the plumbing
    // is in place for Phase D's hot-load path.
    const registry = global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Server not ready", 503);
        return;
    };
    const raw = (try readRequest(allocator, stream)) orelse return;
    defer allocator.free(raw.buf);
    const total_read = raw.len;
    const header_end_pos = raw.header_end;
    const request = raw.buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];

    var line_iter = std.mem.splitScalar(u8, first_line, ' ');
    const method = line_iter.next() orelse return;
    const raw_path = line_iter.next() orelse return;
    // Strip query string for route matching (e.g. /v1/chat/completions?beta=true -> /v1/chat/completions)
    const path = if (std.mem.indexOf(u8, raw_path, "?")) |qpos| raw_path[0..qpos] else raw_path;
    const request_body = if (total_read > header_end_pos) request[header_end_pos..total_read] else "";
    const request_content_type = findHeaderValue(request[0..header_end_pos], "content-type") orelse "";
    logHttpRequest(method, logSafePath(raw_path), logSafeBody(method, path, request_content_type, request_body));
    if (jsonNestingDepth(request_body) > MAX_JSON_DEPTH) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request JSON nests deeper than 4096 levels", 400);
        return;
    }
    stream.noteRequestHeaders(request[0..header_end_pos]);

    // ── API-key auth gate. When --api-key is set, every NON-LOOPBACK request
    //    requires the key (the OpenAI/Anthropic APIs AND the index page
    //    + metrics panel/feed) — EXCEPT `/health` and CORS preflight, which
    //    load balancers and browsers must reach unauthenticated. Loopback is
    //    trusted (the local app connects via 127.0.0.1), so it's exempt: the app
    //    + a local browser never need credentials, and the key protects network
    //    exposure. Browser-facing pages get a `WWW-Authenticate: Basic`
    //    challenge so the index + metrics panel prompt for the key; API clients
    //    pass Bearer / x-api-key. null key ⇒ fully open. See `apiKeyAuthorized`.
    if (apiKeyGateApplies(g_api_key != null, g_api_key_strict, peerIsLoopback(stream)) and
        !std.mem.eql(u8, method, "OPTIONS") and
        !(std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")))
    {
        const raw_headers = request[0..header_end_pos];
        if (loginFormKey(method, path, request_content_type, request_body)) |k| {
            if (encodedKeyMatches(k, g_api_key.?)) return sendKeyCookie(stream, k);
            return sendResponse(stream, "401 Unauthorized", "text/html; charset=utf-8", loginPageHtml("<p class=err>That key was not accepted. Try again.</p>"));
        }
        if (!apiKeyAuthorized(raw_headers, raw_path)) {
            log.debug("{s} {s} -> 401 (missing/invalid API key)\n", .{ method, path });
            if (wantsLoginPage(method, path, raw_headers))
                try sendResponse(stream, "401 Unauthorized", "text/html; charset=utf-8", loginPage(raw_headers, raw_path))
            else
                try sendUnauthorized(stream);
            return;
        }
        // A page opened with `?api_key=` trades it for the cookie, so the key leaves the address bar.
        if (wantsLoginPage(method, path, raw_headers)) {
            if (queryParamValue(raw_path, "api_key") orelse queryParamValue(raw_path, "key")) |k| {
                if (cookieSafe(k) and encodedKeyMatches(k, g_api_key.?)) return sendKeyCookie(stream, k);
            }
        }
    }

    // ── Plan 05: routes that don't depend on a loaded model (connectivity
    //    probes + CORS preflight + listing endpoints). Handle these BEFORE
    //    `scheduler.ensureLoaded` so they don't trigger a cold load of the
    //    default model just to read metadata. `/v1/models` is a pure
    //    registry read, no model needed.
    if (std.mem.eql(u8, method, "HEAD") and std.mem.eql(u8, path, "/")) {
        log.debug("HEAD / -> 200\n", .{});
        try sendResponse(stream, "200 OK", "text/plain", "");
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        if (mlx.gpuContextLost()) return sendResponse(stream, "503 Service Unavailable", "application/json", "{\"status\":\"cuda context lost\"}");
        log.debug("GET  /health -> 200\n", .{});
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
        return;
    }
    // Prometheus scrape endpoint. 503 when --metrics is off. Behind the global
    // API-key gate above when --api-key is set (auth already enforced here).
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/metrics")) {
        if (g_metrics) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            try instr.renderPrometheus(m, &out.writer);
            // Quiet: don't dump the ~3 KB scrape body every 2 s at debug level.
            try sendResponseQuiet(stream, "200 OK", "text/plain; version=0.0.4; charset=utf-8", out.written());
        } else {
            try sendResponse(stream, "503 Service Unavailable", "text/plain", "metrics not enabled (start with --metrics)");
        }
        return;
    }
    // JSON feed — drives the console's Monitoring pane. Behind the
    // global API-key gate above when --api-key is set (same-origin browser
    // fetch inherits the page's Basic credentials).
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/metrics.json")) {
        if (g_metrics) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            var sessions: [2 * instr.MAX_SESSIONS]instr.Session = undefined;
            try instr.renderJson(m, liveSessions(registry, &sessions), &out.writer);
            // Quiet: the index panel polls this ~1 Hz — don't log the body.
            try sendResponseQuiet(stream, "200 OK", "application/json", out.written());
        } else {
            try sendResponse(stream, "503 Service Unavailable", "application/json", "{\"error\":\"metrics not enabled — start with --metrics\"}");
        }
        return;
    }
    if (std.mem.eql(u8, method, "OPTIONS")) {
        log.debug("OPTIONS {s} -> 204\n", .{path});
        try sendResponse(stream, "204 No Content", "text/plain", "");
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        log.debug("GET  /v1/models -> 200\n", .{});
        try handleModels(allocator, stream);
        return;
    }
    // Any method: a status route never reaches `ensureLoaded`, whatever its body names.
    if (std.mem.eql(u8, path, "/v1/load-progress")) {
        const body = renderLoadProgressBody(allocator, stream.io) catch |err| switch (err) {
            error.RegistryNotReady => return sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503),
            else => return err,
        };
        defer allocator.free(body);
        try sendResponse(stream, "200 OK", "application/json", body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/models/rescan")) {
        // Absorb models downloaded AFTER boot (the Model Browser pulls while
        // the server runs; discovery only walks the roots at startup).
        try handleModelsRescan(allocator, stream);
        return;
    }

    // ── Plan 05 Phase D: resolve the request's target model via the
    //    scheduler (which delegates the fast path to registry.ensureLoaded
    //    and handles cold-load + eviction internally). Absent `model`
    //    field → default. Unknown id → 404. Unloaded id with no room and
    //    no LRU victim → 503 (NotEnoughMemory). Block-until-loaded is
    //    intentional; clients targeting an unloaded model should set a
    //    longer request timeout.
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Strip whatever the client passed in `"model":"..."` — except when the
    // id literally matches one we've discovered, in which case we honor it
    // and route. The OpenAI / Anthropic ecosystem commonly sends marketing
    // names like "gpt-4" or "claude-opus-4-x" expecting the local server to
    // just respond with whatever it has loaded; the multi-model registry's
    // strict-id semantics are opt-in by sending an id we registered.
    // Swift's JSON writers escape '/' as '\\/' (Foundation does this
    // unconditionally, both JSONSerialization and JSONEncoder). That is legal
    // JSON, but the id is read out of the RAW body, so an HF-style `org/repo`
    // id arrived as `Runpod\\/FLUX.2-…`, matched nothing in the registry, and
    // silently fell through to the default model — a 400 "does not support
    // this media modality" on the gen endpoints, and a wrong-model answer on
    // chat (live from the iPhone app, 2026-07-25). Canonicalise once, here,
    // so every consumer below (peek, ensureLoaded) sees the real id.
    var model_id_buf: [512]u8 = undefined;
    var query_model_buf: [512]u8 = undefined;
    var requested_model_id = unescapeJsonSlashes(
        &model_id_buf,
        parseModelFromBody(request_body) orelse queryModel(&query_model_buf, raw_path) orelse "",
    );
    switch (resolveRequestModelId(registry, &g_model_aliases, requested_model_id)) {
        .id => |id| requested_model_id = id,
        .unknown_path => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "No model is registered at that path", 404);
            return;
        },
        // Unknown name — fall back to the default model rather than 404, so
        // off-the-shelf SDK clients keep working.
        .unknown_name => requested_model_id = "",
    }
    // Text-gen route aimed at a KNOWN non-text model: reject before
    // ensureLoaded, or the request cold-loads a multi-GB media model just
    // to earn its 400. The post-load `text_gen_reject` below stays the
    // authoritative crash barrier (this peek can't see `--model` primaries
    // with no arch hint until they're resident).
    if (requested_model_id.len > 0 and isTextGenRoute(method, path)) {
        // Entry and snapshot under ONE hold — plain `peek` unlocks before it
        // returns. `arch_hint` outlives the unlock: entry-owned, freed in
        // `deinit`.
        const target: ?TextGenTarget = blk: {
            registry.mutex.lockUncancelable(stream.io);
            defer registry.mutex.unlock(stream.io);
            const peeked = registry.peekLocked(requested_model_id) orelse break :blk null;
            break :blk textGenTargetOf(peeked);
        };
        if (target) |t| {
            if (textGenRejectReason(t)) |reason| {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
                return;
            }
        }
    }
    // An endpoint that doesn't exist is a 404, and it must cost NOTHING to say
    // so. Resolution runs ahead of dispatch, so this has to be answered here
    // rather than inside one of `ensureLoaded`'s error arms: with a resolvable
    // `model` in the body, resolution SUCCEEDS — cold-loading the checkpoint —
    // and the unknown path only 404s afterwards. `POST /v1/load` (a route that
    // never existed) cost 2m42s and 121 GB resident for a typo, and the same
    // one-liner pins the box for anyone who sends it. Placed above every load.
    if (!routeExists(path)) {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Unknown endpoint", 404);
        return;
    }
    // `GET /props` is a STATUS read, and falling into `ensureLoaded` makes a
    // poll cold-load the model. With idle eviction on that is a loop: the sweep
    // hands the memory back and the next poll takes it straight again (the app's
    // tray polls every 3s). Nothing resident means nothing to report — the same
    // body the `NoDefaultModel` arm below already sends.
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        const resident = if (registry.resolveEntry(requested_model_id)) |e| blk: {
            registry.mutex.lockUncancelable(stream.io);
            defer registry.mutex.unlock(stream.io);
            break :blk e.state == .ready;
        } else |_| false;
        if (!resident) {
            try handlePropsNoModel(allocator, stream);
            return;
        }
    }
    const lm = scheduler.ensureLoaded(requested_model_id) catch |err| {
        // Headless gen-only boot (media models resident, no default chat
        // model): /props keeps answering with the live memory counters —
        // the app's tray polls it, and a 503 here read as "0 MB".
        if (err == error.NoDefaultModel and std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
            try handlePropsNoModel(allocator, stream);
            return;
        }
        try sendEnsureLoadedError(allocator, stream, scheduler, requested_model_id, err);
        return;
    };
    // A status read must not look like use — see `isStatusRoute`.
    const status_read = isStatusRoute(method, path);
    defer if (status_read) scheduler.releaseStatus(lm) else scheduler.release(lm);

    // A model loaded on demand freezes its own auto-context here, for the same
    // reason the `--model` primary does at startup: the number we advertise is
    // what clients budget against, and each model has its own dims. Idempotent,
    // and a no-op under `--ctx-size`.
    if (lm.config) |c| _ = pinAutoContext(c);

    // ── Phase C: handlers take `lm` directly and extract their own
    //    locals. handleConnection only needs one decision for every
    //    text-generation route: entries with no
    //    language model resident get a clean 400 instead of reaching a prefill that has
    //    no transformer to run (the SIGSEGV class documented on
    //    textGenRejectReason).
    const text_gen_reject: ?[]const u8 = textGenRejectReason(textGenTargetOf(lm));

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try handleProps(allocator, stream, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/embeddings")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleEmbeddings(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/tokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleTokenize(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/detokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleDetokenize(allocator, stream, body, lm);
    } else {
        log.warn("{s} {s} -> 404\n", .{ method, path });
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "The requested endpoint does not exist", null);
    }
}

/// `meta.quantization` as a JSON value: `"8-bit"`, or null when the root config declares no
/// width (media packs, bf16 and per-module mixed packs).
fn quantizationJson(buf: *[16]u8, bits: u32) []const u8 {
    if (bits == 0) return "null";
    return std.fmt.bufPrint(buf, "\"{d}-bit\"", .{bits}) catch "null";
}

test "quantizationJson: a declared width is N-bit, an undeclared one is null" {
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("\"8-bit\"", quantizationJson(&buf, 8));
    try testing.expectEqualStrings("null", quantizationJson(&buf, 0));
}

/// Percentage of the memory-derived ceiling we actually admit as context when
/// `--ctx-size` is absent.
///
/// `computeMaxSafeContext` returns the largest context that FITS right now.
/// Serving at exactly that leaves nothing for the prefix cache to grow into, a
/// second model to load beside this one, or another app to take RAM — and the
/// Metal OOM it would eventually hit is uncatchable (see the auto-context
/// gotcha). Reserve headroom instead.
const auto_ctx_safety_pct: u32 = 85;

/// PURE: apply the safety margin and round down to a 1024 boundary so the
/// number reads sanely in logs and client configs. Never returns 0.
///
/// Takes the MEMORY-derived ceiling only. The model's own `max_position_embeddings`
/// is applied afterwards, un-margined: when the checkpoint's max is the binding
/// constraint there is nothing to reserve memory headroom against, and shaving
/// 15% off a 131,072-token model that comfortably fits in RAM just throws
/// context away.
fn safeAutoContext(raw: u32) u32 {
    const scaled: u64 = (@as(u64, raw) * auto_ctx_safety_pct) / 100;
    const rounded: u64 = (scaled / 1024) * 1024;
    if (rounded == 0) return @intCast(@max(scaled, 1));
    return @intCast(rounded);
}

/// This model's auto-context: memory ceiling with headroom, then capped by what
/// the checkpoint actually supports. `contextCap` is the rope-derived window:
/// `max_position_embeddings`, or `original_max_position_embeddings × factor` for
/// a YaRN-scaled checkpoint — the same derivation vLLM applies to
/// `max_model_len`, because a position past the scaled window aliases back
/// inside the ramp rather than reading as a longer distance.
/// The context the server advertises, from the largest context memory alone allows (85%
/// margin on the memory number, checkpoint cap afterwards). Shared by the load-time session
/// bill and the boot-time sizer so the two cannot spell the relation two ways.
fn autoContextFrom(memory_ctx: u32, ctx_cap: u32) u32 {
    const with_headroom = safeAutoContext(memory_ctx);
    return if (ctx_cap > 0) @min(with_headroom, ctx_cap) else with_headroom;
}

fn autoContextFor(config: *const model_mod.ModelConfig) u32 {
    return autoContextFrom(computeMemoryContext(config), config.contextCap());
}

/// Freeze this model's auto-context at load time. Idempotent; a no-op (and
/// never a write) when `--ctx-size` pinned it explicitly. Returns the value
/// `getEffectiveContextLength` will report from now on.
///
/// Called once per model, at startup for the `--model` primary and right after
/// an on-demand load. The advertised context must not drift: agent CLIs read it
/// once (pi/opencode never re-read `/v1/models`) and budget their own
/// `max_tokens` against it for the rest of the session.
fn pinAutoContext(config: *model_mod.ModelConfig) u32 {
    // Order is load-bearing: the sizer bills THIS chunk's transient reserve and
    // the admission guard bills the same frozen value, so the width has to be
    // resolved before any context is computed from it. Pinned even under an
    // explicit `--ctx-size`, because the guard reads it on every request.
    _ = pinPrefillChunk(config);
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context == 0) {
        config.pinned_context = autoContextFor(config);
    }
    return config.pinned_context;
}

/// Live requests, then every hot-cache entry no live request restored from, each row stamped
/// with its model's context limit.
fn liveSessions(registry: *ModelRegistry, buf: *[2 * instr.MAX_SESSIONS]instr.Session) []instr.Session {
    const sch = global_scheduler orelse return buf[0..0];
    sch.queue_mu.lockUncancelable(sch.io);
    const live = sch.live_session_count;
    @memcpy(buf[0..live], sch.live_sessions[0..live]);
    sch.queue_mu.unlock(sch.io);

    var n = live;
    sch.digest_mu.lockUncancelable(sch.io);
    for (sch.cached_sessions[0..sch.cached_session_count]) |c| {
        const in_use = for (buf[0..live]) |l| {
            if (l.entry_id == c.entry_id and std.mem.eql(u8, l.model(), c.model())) break true;
        } else false;
        if (in_use) continue;
        buf[n] = c;
        n += 1;
    }
    sch.digest_mu.unlock(sch.io);

    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    for (buf[0..n]) |*s| {
        const entry = registry.peekLocked(s.model()) orelse continue;
        if (entry.state != .ready) continue;
        if (entry.config) |cfg| s.context_length = getEffectiveContextLength(cfg);
    }
    return buf[0..n];
}

fn getEffectiveContextLength(config: *const model_mod.ModelConfig) u32 {
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context > 0) return config.pinned_context;
    // Not pinned yet (a discovery stub that was never loaded): compute from
    // current GPU memory rather than a fixed 16K cap.
    return autoContextFor(config);
}

/// Metal's recommended max working-set size for the default device — the real
/// ceiling whose breach throws `[METAL] … Insufficient Memory` from the
/// command-buffer completion handler (which terminates the process).
/// `getMetalBufferLimit()` (75% of physical RAM) over-estimates this on
/// small-RAM Macs — a 16 GB Mac reports ~11.9 GB recommended vs the 12 GB that
/// hw.memsize×0.75 yields — so budgeting against hw.memsize lets auto-context
/// oversubscribe. Falls back to `getMetalBufferLimit()` when the device query
/// is unavailable (CI / non-Metal hosts).
fn getGpuWorkingSetLimit() u64 {
    if (static_ceiling_override) |v| return v;
    const max_rec = mlx.maxRecommendedWorkingSet();
    return if (max_rec > 0) max_rec else getMetalBufferLimit();
}

/// PURE (unit-testable): the real ceiling a NEW MLX allocation must fit under.
///
/// `working_set_limit` (Metal's `max_recommended_working_set_size`) is a STATIC
/// device maximum — it assumes the whole GPU working set is MLX's to claim and
/// is BLIND to memory held by anything else on the machine. `mlx_footprint`
/// (MLX active + reclaimable cache) + `free_system` (physically free RAM) is
/// what MLX can actually reach RIGHT NOW; when another process holds a big
/// chunk of unified memory the second term binds and the ceiling collapses.
///
/// #64 (2026-07): a Claude Code session on a 128 GB Mac ran a docker-compose
/// stack (firecrawl/rabbitmq/postgres/playwright) holding tens of GB. The guard
/// budgeted against the static 115 GB max, admitted a 90 K-token MoE prefill,
/// and Metal threw `Insufficient Memory` from the command-buffer completion
/// handler — an UNCATCHABLE async C++ throw that terminates the process
/// (the throw is MLX's). Capping by real free RAM makes the two prefill guards
/// reject/clamp before that allocation is ever submitted.
fn physicalMemoryCeiling(working_set_limit: u64, mlx_footprint: u64, free_system: u64) u64 {
    return @min(working_set_limit, mlx_footprint +| free_system);
}

/// How far under the enforced wired limit a plan may reach: past the limit Metal returns
/// zeros before an uncatchable abort, so a real transient's worth of margin stays unplanned.
pub const WIRED_LIMIT_MARGIN_BYTES: u64 = 8 << 30;

/// PURE: the floor an explicitly raised `iogpu.wired_limit_mb` puts under the ceiling; 0 when
/// the sysctl is absent or at the macOS default (75% of RAM), which leaves the ceiling as is.
/// A floor, not a cap: the working-set term already tracks the sysctl. The term it lifts is
/// the free-RAM one, which counts other processes' pages as gone; the operator who raised the
/// enforced limit has said the GPU may claim that much. `margin` (`--wired-margin-gib`)
/// is what stays unplanned under the limit: other processes' wired GPU memory, the
/// allocator cache and the estimator's slack; past the limit Metal returns zeros.
pub fn wiredLimitFloor(wired_limit: u64, total_ram: u64, margin: u64) u64 {
    if (wired_limit == 0 or total_ram == 0) return 0;
    if (wired_limit <= total_ram * 75 / 100) return 0; // the macOS default: no declaration
    return @min(wired_limit -| margin, total_ram -| margin);
}

pub fn parseWiredMarginGib(raw: []const u8) error{InvalidWiredMargin}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidWiredMargin;
    if (n < 2 or n > 32) return error.InvalidWiredMargin;
    return @as(u64, n) << 30;
}

/// `--wired-margin-gib` in bytes: what stays unplanned under a raised wired limit.
pub var wired_limit_margin_bytes: u64 = WIRED_LIMIT_MARGIN_BYTES;

/// PURE: the ceiling with the wired-limit floor applied. `wired_floor == 0` returns
/// `physicalMemoryCeiling` byte for byte.
pub fn gpuCeilingWithWiredFloor(
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
) u64 {
    const physical = physicalMemoryCeiling(working_set_limit, mlx_footprint, free_system);
    return @max(physical, wired_floor);
}

/// Test seam: `iogpu.wired_limit_mb` in MB; `null` reads the machine.
pub var wired_limit_mb_override: ?u64 = null;

var wired_limit_bytes_cached: u64 = 0;
var wired_limit_read: bool = false;

/// `iogpu.wired_limit_mb` in bytes, read ONCE per process: re-reading it per admission would
/// let the ceiling move under a live request. 0 when the OID is absent.
pub fn wiredLimitBytes() u64 {
    if (wired_limit_mb_override) |mb| return mb *| (1024 * 1024);
    // iogpu.wired_limit_mb is an Apple GPU sysctl; Linux has no wired-limit
    // concept, so the query reads as absent (0), same as a non-Apple-Silicon Mac.
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    if (wired_limit_read) return wired_limit_bytes_cached;
    var v: u32 = 0;
    var len: usize = @sizeOf(u32);
    wired_limit_bytes_cached = if (sysctlbyname("iogpu.wired_limit_mb", @ptrCast(&v), &len, null, 0) == 0)
        @as(u64, v) * 1024 * 1024
    else
        0;
    wired_limit_read = true;
    return wired_limit_bytes_cached;
}

var wired_floor_logged = std.atomic.Value(bool).init(false);

/// The floor for THIS model, gated on `ModelConfig.longCtxGated()`. `null` config (no model
/// resolved yet) takes no floor.
fn wiredCeilingFloorFor(config: ?*const model_mod.ModelConfig) u64 {
    return wiredCeilingFloorForRam(config, metrics.getTotalMemBytes());
}

/// PURE: the floor for a machine with `total_ram` bytes; the wrapper above reads the machine.
fn wiredCeilingFloorForRam(config: ?*const model_mod.ModelConfig, total_ram: u64) u64 {
    const c = config orelse return 0;
    if (!c.longCtxGated()) return 0;
    const floor = wiredLimitFloor(wiredLimitBytes(), total_ram, wired_limit_margin_bytes);
    if (floor > 0 and wired_floor_logged.cmpxchgStrong(false, true, .monotonic, .monotonic) == null) {
        log.info("[mem] ceiling {d} MB from iogpu.wired_limit_mb={d} (working set {d} MB, margin {d} MB)\n", .{
            floor >> 20,
            wiredLimitBytes() >> 20,
            getGpuWorkingSetLimit() >> 20,
            wired_limit_margin_bytes >> 20,
        });
    }
    return floor;
}

/// PURE (unit-testable): the cap to put on MLX's reclaimable buffer pool for a
/// machine with `total_ram` bytes of physical memory. 0 when the RAM query
/// failed — never clamp on bad data.
///
/// MLX's own default is `min(1.5 x working_set, 0.95 x RAM)` with a GC limit at
/// `0.95 x working_set` (`backend/metal/allocator.cpp`) — ~121 GB / ~91 GB on a
/// 128 GB Mac, i.e. it does not trim until the machine is already dead. Freed
/// buffers are parked in that pool instead of going back to the OS, so anything
/// that frees never-repeating sizes in a loop (KV growth, decode transients)
/// grows the process footprint without moving `active_bytes` at all. Issue #110:
/// 81.4 GB in Activity Monitor against 19.6 GB in the panel.
///
/// RAM/16 is big enough for the step-to-step buffer reuse the pool exists for
/// (8 GB on a 128 GB Mac) and small enough that it can never be the footprint.
/// The 2 GB floor keeps 8/16 GB machines from thrashing the allocator.
pub fn mlxCacheLimitBytes(total_ram: u64) u64 {
    if (total_ram == 0) return 0;
    const GB: u64 = 1 << 30;
    return @max(2 * GB, @min(8 * GB, total_ram / 16));
}

/// PURE: resolve the cap from `MLX_SERVE_CACHE_LIMIT` (bytes) over the
/// RAM-proportional default. `0` means "leave MLX's default alone" — the
/// same-boot A/B off-switch. Anything unparseable falls through to the default
/// rather than silently disabling the cap.
pub fn mlxCacheLimitFromEnv(raw: ?[]const u8, total_ram: u64) u64 {
    if (raw) |v| {
        const trimmed = std.mem.trim(u8, v, " \t\r\n");
        if (std.fmt.parseInt(u64, trimmed, 10) catch null) |n| return n;
    }
    return mlxCacheLimitBytes(total_ram);
}

/// Impure wrapper: apply the cap to the live MLX allocator. Called ONCE from
/// `main()`, above every subcommand branch — not per serve path, because a
/// hand-rolled per-path config is exactly how `runHeadlessServe` (the mode the
/// app always launches) came to silently eat the `--pld*` flags.
///
/// Never RAISES a tighter existing cap: iOS boots at 384 MB.
pub fn applyMlxCacheLimit() void {
    const env: ?[]const u8 = if (std.c.getenv("MLX_SERVE_CACHE_LIMIT")) |p|
        std.mem.span(p)
    else
        null;
    const cap = mlxCacheLimitFromEnv(env, metrics.getTotalMemBytes());
    if (cap == 0) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(cap));
    if (prev < cap) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
        return;
    }
    log.info("[mem] MLX buffer-pool cap {d} MB (was {d} MB)\n", .{ cap >> 20, prev >> 20 });
}

/// Stands in for the machine's working-set limit (Metal's recommended working set
/// or the wired limit): tests (CI runners have 7 GB) and `applyGpuCeilingEnv`.
/// Load-time bills price this static term (stable across boots); request-time admission reads live RAM.
pub var static_ceiling_override: ?u64 = null;

pub fn staticGpuMemoryCeiling() u64 {
    return getGpuWorkingSetLimit();
}

/// Diagnostic: `MLX_SERVE_GPU_CEILING_MB=N` stands in for the working-set limit in every bill
/// (absent or 0 = the machine's). It bounds the guards' arithmetic, not what Metal will grant.
pub fn applyGpuCeilingEnv() void {
    const raw = std.c.getenv("MLX_SERVE_GPU_CEILING_MB") orelse return;
    const mb = std.fmt.parseInt(u64, std.mem.sliceTo(raw, 0), 10) catch return;
    if (mb == 0) return;
    static_ceiling_override = mb << 20;
    log.warn("[diag] GPU ceiling forced to {d} MB (MLX_SERVE_GPU_CEILING_MB)\n", .{mb});
}

/// THE ceiling helper: `available`, `/props`, the hot-cache clamp, the auto-context pin and the
/// admission bill all read it, so the wired-limit floor cannot reach some of them and not others.
fn currentGpuMemoryCeiling(config: ?*const model_mod.ModelConfig, active_mem: u64) u64 {
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    return gpuCeilingWithWiredFloor(
        getGpuWorkingSetLimit(),
        active_mem +| @as(u64, cache_mem),
        metrics.getAvailableMemBytes() -| metrics.osReserveBytes(metrics.getTotalMemBytes()),
        wiredCeilingFloorFor(config),
    );
}

/// PURE budget math (no MLX/Metal calls — unit-testable): the largest context
/// length whose KV cache + per-token working set fits under the GPU
/// working-set ceiling, after subtracting what's already resident
/// (`active_mem` — model weights + any resident hot-cache KV) AND the hot
/// prefix cache's FULL byte budget (`hot_cache_reserve`).
///
/// Reserving the whole cache budget — not just its current residency — budgets
/// for STEADY STATE. Over an agentic session the hot cache fills to its cap, so
/// an auto-context computed against an empty cache (24k tokens on a 16 GB Mac,
/// 2026-06-19 live) later collides with the filled 2 GB cache plus a
/// large cold MoE prefill and the command buffer OOMs. Subtracting it up front
/// keeps the reported context stable across the session and within the ceiling
/// once the cache is full. The 0.64 factor (two ×4/5 margins) absorbs the
/// prefill compute spike that lands on top of `active_mem`.
/// PURE: the raw largest context the budget fits — no 1024 floor, no
/// `max_pos` clamp. Split out of `safeContextForBudget` because
/// `resolvePrefillChunk` compares rungs against each other and the floor makes
/// a rung that fits NOTHING look identical to one that fits 1024 tokens.
/// The 0.64 factor is the same two x4/5 margins documented below.
fn contextForBudget(working_set_limit: u64, active_mem: u64, hot_cache_reserve: u64, per_tok: u64) u64 {
    if (per_tok == 0) return 0;
    const spoken_for: u64 = active_mem +| hot_cache_reserve;
    const usable: u64 = if (working_set_limit > spoken_for) working_set_limit - spoken_for else 0;
    return ((usable * 4 / 5) * 4 / 5) / per_tok;
}

fn safeContextForBudget(
    working_set_limit: u64,
    active_mem: u64,
    hot_cache_reserve: u64,
    per_tok: u64,
    max_pos: u32,
) u32 {
    if (per_tok == 0) return 1024;
    const max_seq: u64 = contextForBudget(working_set_limit, active_mem, hot_cache_reserve, per_tok);
    if (max_seq < 1024) return 1024;

    var result: u32 = if (max_seq > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(max_seq);
    // Cap at model's max position embeddings.
    if (max_pos > 0) result = @min(result, max_pos);
    return result;
}

/// Compute the maximum safe context length based on the GPU working-set
/// ceiling, the model's per-token footprint, and the hot prefix cache budget.
/// Linear in seq (per_tok × seq ≤ budget) — no seq² term, MLX's fused SDPA
/// tiles over seq and never materializes [heads, seq, seq]. See
/// `safeContextForBudget` for the steady-state reservation rationale.
/// `computeMemoryContext` capped by the checkpoint's `max_position_embeddings`.
/// This is the raw "largest context that fits" figure — reported in logs and
/// `/props` for diagnostics. The context we actually SERVE is `autoContextFor`,
/// which reserves headroom below it.
fn computeMaxSafeContext(config: *const model_mod.ModelConfig) u32 {
    const memory_ctx = computeMemoryContext(config);
    const max_pos = config.contextCap();
    return if (max_pos > 0) @min(memory_ctx, max_pos) else memory_ctx;
}

/// The process-wide KV width a request gets when it names none — the same
/// expression `checkAttentionMemory` falls back to, so the sizer and the
/// admission guard read one answer.
///
/// Through `configuredKvQuant`, so a load-time bill gets the boot's `--kv-quant`, not the dense fallback.
fn defaultKvBits(config: *const model_mod.ModelConfig) u64 {
    const cfg: transformer_mod.KVQuantConfig = configuredKvQuantFor(config);
    return if (cfg.scheme == .off) 16 else cfg.bits;
}

/// The prefill working set that is NOT the KV cache: score scratch, the
/// quantized-KV dequant, and the QKV/MLP transient envelope — all bounded by
/// the prefill CHUNK, at the widest chunk any prompt on this model can run.
///
/// It is a one-off reserve, not a per-token cost. `computeMemoryContext` used
/// to bill the MLP envelope (`8 × max(hidden, ffn) × 2`) against every token of
/// CONTEXT: 272 KB/token beside a 64 KB/token KV cache on a 27B hybrid, i.e.
/// 81% of the budget spent on a transient that never scales with the context
/// length, which caps a 16 GB Mac under 4k tokens whatever the weights cost.
/// Measured on the shipped 4-bit 27B (2026-08-14): the peak above steady state
/// is flat in prompt length and tracks the chunk — 3.34 GB at chunk 2048 for
/// prompts from 3k to 51k tokens.
///
/// Same estimator as the admission guard with the KV term zeroed, because the
/// KV cache is exactly what the sizer is solving for.
pub fn prefillTransientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64) u64 {
    return prefillTransientReserveAtKv(config, kv_bits, chunk, chunk);
}

/// `prefillTransientReserve` with the KV length made explicit. The load-time sizer asks at
/// `kv = chunk`; a per-chunk decision has a KV, and the attention score sheet and the QSA
/// sheet are linear in it (~0.5 GB under-billed at the default sheet, ~9 GB at 4096 MB).
pub fn prefillTransientReserveAtKv(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    chunk: u64,
    kv_len: u64,
) u64 {
    const seq: u64 = @max(kv_len, chunk);
    return prefillMemoryNeeded(
        seq,
        config.num_attention_heads,
        config.num_key_value_heads,
        0,
        config.head_dim,
        config.prefillScoreHeadDim(),
        config.hidden_size,
        prefillFfnWidth(config),
        kv_bits,
        chunk,
        config.prefillAttnKeys(seq),
        prefillStreamBytesPerToken(config),
        prefillDequantWeightBytes(config),
        .{ .dq_min_rows = transformer_mod.prefillDqGemmMinRows(config.quant_bits) },
    );
}

var ctx_bar_cached: ?bool = null;

/// `MLX_SERVE_PREFILL_CHUNK_CTX_BAR=0` restores the share-only rung cap.
pub fn ctxBarEnabled() bool {
    if (ctx_bar_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_PREFILL_CHUNK_CTX_BAR") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    ctx_bar_cached = v;
    return v;
}

/// Widths `resolvePrefillChunk` will step down through. Descending, floored at
/// `generate.PREFILL_CHUNK_FLOOR` — below that the score-budget path refuses to
/// go either, and a 256-token forward stops amortizing the per-chunk sweeps.
/// The top rung is `WIDE_PREFILL_CHUNK`, pinned only by an arch with no prefill score tensor.
pub const PREFILL_CHUNK_LADDER = [_]u32{ @intCast(generate_mod.WIDE_PREFILL_CHUNK), 8192, 4096, 2048, 1024, 512 };

/// How much of the post-weights serving budget a ONE-OFF prefill transient may
/// claim before the chunk steps down. A quarter: past that the machine is
/// trading the whole session's context for one forward's speed, which is the
/// trade that reported a 1024-token context on a 16 GB Mac.
const PREFILL_RESERVE_BUDGET_SHARE: u64 = 4;

/// The hot prefix cache's smallest useful budget: a prefill chunk that would price the cache
/// under this steps down instead. Stops `requested == 0` from resolving to the 1-byte floor.
const HOT_CACHE_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

/// PURE: the widest prefill chunk THIS machine can afford for THIS model.
///
/// The chunk multiplies the biggest term in the memory bill, and until now
/// nothing tied it to the machine: `--prefill-chunk` defaults to 8192, the app
/// never passes the flag, so a 16 GB Mac reserved the same 5-7 GB MLP envelope
/// a 128 GB one does. Measured on a 16 GB profile (a 7B 4-bit model, weights
/// 3.80 GB against an 11.9 GB Metal ceiling): the reserve at chunk 8192 alone
/// exceeded what was left after the weights, so the sizer reported 1024 tokens
/// and the admission guard refused a 10k-token prompt for "needing" 9.2 GB —
/// a prompt whose measured peak is 2.39 GB. Stepping to chunk 512 costs prefill
/// throughput and buys 22k tokens of context back.
///
/// Never raises anything: `effectivePrefillChunk` takes the MIN of this and the
/// launch chunk, and an explicit `--prefill-chunk` outranks it entirely.
///
/// It does not read the hot-cache ask: the clamp downstream subtracts this chunk's reserve
/// from the cache's headroom, so feeding the ask in here was non-monotone (a bigger
/// `--prefix-cache-mem` stepped the chunk down and handed the cache MORE bytes). The chunk is
/// sized against the serving budget alone; the cache is the residual claimant. `ctx_kv_bytes`
/// is the KV a pinned context has already spoken for (0 under auto-context).
/// The byte cap one prefill forward's transient may claim, and the one place the sizer's two
/// changes (dropping the ask, adding the ctx bar) are gated by arch; every other arch keeps
/// the previous arithmetic. `MLX_SERVE_PREFILL_CHUNK_CTX_BAR=0` drops the ctx bar inside the gate.
pub fn prefillChunkCap(
    config: *const model_mod.ModelConfig,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u64 {
    if (!config.longCtxGated()) {
        // Ungated: the ask is spoken for, and a quarter of what is left is the whole cap.
        const spoken_for: u64 = active_mem +| hot_cache_ask;
        return (ceiling -| spoken_for) / PREFILL_RESERVE_BUDGET_SHARE;
    }
    const serving_budget: u64 = ceiling -| active_mem;
    // Two ask-independent bars: a share of the serving budget, and whatever the pinned context leaves less a cache worth having.
    const share_bar: u64 = serving_budget / PREFILL_RESERVE_BUDGET_SHARE;
    // The ctx bar can only ever narrow; `--ctx-size` is a cap, not an allocation, so it can pin
    // chunk 512 for the life of a process. The narrowing is logged.
    const ctx_bar: u64 = if (ctxBarEnabled()) (serving_budget -| ctx_kv_bytes) -| HOT_CACHE_FLOOR_BYTES else share_bar;
    return @min(share_bar, ctx_bar);
}

pub fn resolvePrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u32 {
    const cap: u64 = prefillChunkCap(config, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask);
    for (PREFILL_CHUNK_LADDER) |chunk| {
        if (chunk == generate_mod.WIDE_PREFILL_CHUNK and config.prefillScoreHeadDim() != 0) continue;
        if (prefillTransientReserve(config, kv_bits, chunk) <= cap) return chunk;
    }
    // Nothing fits the share — the model barely fits at all. Take the narrowest
    // rung: it is the smallest bill this box can be asked for, and returning 0
    // would read as "not pinned" and hand the forward the launch width.
    return PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
}

/// The width `--prefill-chunk` asked for, or 0. The bill must let it outrank the pin as the
/// forward does: `--prefill-chunk 4096` used to run against a 512-token reserve.
pub fn explicitPrefillChunk() u32 {
    // `MLX_SERVE_PREFILL_CHUNK` outranks everything in `generate.effectivePrefillChunk`, so the
    // bill must see it too. Clamped: the env value is unbounded at its source.
    const env_w: usize = @min(generate_mod.envPrefillChunk(), @as(usize, std.math.maxInt(u32)));
    if (env_w > 0) return @intCast(env_w);
    if (!generate_mod.prefill_chunk_explicit) return 0;
    const raw: usize = @min(generate_mod.prefill_chunk_override, @as(usize, std.math.maxInt(u32)));
    return @intCast(raw);
}

/// The chunk this model is billed at, and therefore runs at. `chunk_override` (0 = none)
/// mirrors `generate.effectivePrefillChunk`'s precedence.
pub fn billedPrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
    chunk_override: u32,
) u32 {
    if (chunk_override > 0) return chunk_override;
    return @min(resolvePrefillChunk(config, kv_bits, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask), autoPrefillChunkCap());
}

/// The widest chunk the auto sizer allows: unbounded, the billed size alone limits it. An
/// explicit `--prefill-chunk` still wins.
pub fn autoPrefillChunkCap() u32 {
    return std.math.maxInt(u32);
}

test "the auto prefill chunk has no cap of its own" {
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), autoPrefillChunkCap());
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 28;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 3072;
    cfg.intermediate_size = 8192;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    const roomy: u64 = 1 << 40;
    const chunk = billedPrefillChunk(&cfg, 16, roomy, 2 << 30, 0, 0, 0);
    try std.testing.expect(chunk <= autoPrefillChunkCap());
    try std.testing.expectEqual(@as(u32, 4096), billedPrefillChunk(&cfg, 16, roomy, 2 << 30, 0, 0, 4096));
}

/// The context KV the chunk sizer must leave standing: the pinned context's bill under an
/// explicit `--ctx-size`, 0 while the context is auto (the auto sizer adapts to the rung).
pub fn sizerCtxKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (manualContext(config) == 0) return 0;
    return sessionBytesPerToken(config, kv_bits) *| manualContext(config);
}

/// Freeze this model's prefill chunk at load, from live memory. Idempotent.
/// Must run BEFORE the auto-context sizer: the sizer bills this chunk's
/// transient reserve, and `checkAttentionMemory` and
/// `generate.effectivePrefillChunk` then read the same frozen value, so the
/// bill and the forward cannot drift.
pub fn pinPrefillChunk(config: *model_mod.ModelConfig) u32 {
    if (config.pinned_prefill_chunk == 0) {
        var active_mem: usize = 0;
        _ = mlx.mlx_get_active_memory(&active_mem);
        const kv_bits: u64 = defaultKvBits(config);
        // Which ask: the gated arch reads the resolved budget; every other arch reads the raw
        // ask as before (the accessor would pin a second model against the first's budget).
        const hot_cache_ask = if (config.longCtxGated()) resolvedPrefixCacheMem() else legacyPrefixCacheAsk();
        // The override's precedence is the second gated input: ungated, `--prefill-chunk`
        // still outranks the pin only at forward time, as before (a known inconsistency).
        const pin_override: u32 = if (config.longCtxGated()) explicitPrefillChunk() else 0;
        config.pinned_prefill_chunk = billedPrefillChunk(
            config,
            kv_bits,
            currentGpuMemoryCeiling(config, active_mem),
            active_mem,
            sizerCtxKvBytes(config, kv_bits),
            hot_cache_ask,
            pin_override,
        );
        // Say when the ctx bar is the binding one.
        if (ctxBarEnabled() and config.longCtxGated() and explicitPrefillChunk() == 0) {
            const share_only = resolvePrefillChunk(config, kv_bits, currentGpuMemoryCeiling(config, active_mem), active_mem, 0, hot_cache_ask);
            if (share_only > config.pinned_prefill_chunk) {
                log.info("[prefill] chunk {d} (ctx bar: the --ctx-size KV bill; share-only would allow {d}; MLX_SERVE_PREFILL_CHUNK_CTX_BAR=0 restores)\n", .{ config.pinned_prefill_chunk, share_only });
            }
        }
        // Say it once per model, wherever the model was pinned from (startup
        // primary or on-demand load) — a narrowed prefill otherwise reads as an
        // unexplained slowdown.
        if (config.pinned_prefill_chunk < generate_mod.prefill_chunk_override and
            !generate_mod.prefill_chunk_explicit)
        {
            log.info("Prefill chunk: {d} tokens (memory-sized down from {d}; --prefill-chunk overrides)\n", .{ config.pinned_prefill_chunk, generate_mod.prefill_chunk_override });
        }
    }
    return config.pinned_prefill_chunk;
}

/// PURE: clamp the hot prefix cache's byte budget to what the loaded weights
/// leave under the GPU ceiling after the serving context's KV and the prefill
/// transient reserve. A 40 GB `--prefix-cache-mem` beside a ~70 GB pack was
/// never validated against this headroom: the cache filled toward its cap and
/// a 143k prefill died in an uncatchable Metal OOM (2026-08-30, Flash-Next on
/// 128 GB). `requested == 0` (byte cap disabled) is bounded too — an uncapped
/// cache beside a large model is exactly that crash. Never returns 0:
/// `initWithMem` reads 0 as "no byte cap", the opposite of no headroom.
pub fn clampedPrefixCacheMem(
    requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = @max(gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve), 1);
    if (requested == 0) return headroom;
    return @min(requested, headroom);
}

/// What the hot cache gets and what it was charged for getting it.
pub const HotCachePlan = struct {
    /// The width the sizer pinned.
    chunk: u32,
    /// The width the clamp reserved for; not `chunk` on an arch that re-bills per request (`clampReserveWidth`).
    reserve_chunk: u32,
    reserve: u64,
    ctx_kv: u64,
    /// The clamped byte budget handed to `initWithMem`.
    budget: u64,
};

/// The whole chain in the one order that is monotone in the ask: chunk first
/// (ask-independent), then the cache takes the residual.
/// The prefill width the load-time clamp reserves for: the ladder floor on an arch that
/// re-bills its width per request (the load-time reserve is a promise to the first request
/// only). Billing the sizer's rung made the clamp non-monotone in the ceiling (a bigger
/// ceiling bought a wider rung and a smaller cache: 3873 vs 1076 MB). An explicit
/// `--prefill-chunk` is billed as-is; ungated archs keep the sizer's rung.
pub fn clampReserveWidth(config: *const model_mod.ModelConfig, pinned_width: u32) u32 {
    if (explicitPrefillChunk() > 0) return pinned_width;
    if (!perRequestPrefillChunkEnabled(config)) return pinned_width;
    return PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
}

pub fn planHotCache(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_weights: u64,
    ctx_tokens: u64,
    sizer_ctx_kv: u64,
    requested: u64,
    chunk_override: u32,
) HotCachePlan {
    const chunk = billedPrefillChunk(config, kv_bits, ceiling, active_weights, sizer_ctx_kv, requested, chunk_override);
    const reserve_chunk = clampReserveWidth(config, chunk);
    const reserve = prefillTransientReserve(config, kv_bits, reserve_chunk);
    const ctx_kv: u64 = sessionBytesPerToken(config, kv_bits) *| ctx_tokens;
    return .{
        .chunk = chunk,
        .reserve_chunk = reserve_chunk,
        .reserve = reserve,
        .ctx_kv = ctx_kv,
        .budget = clampedPrefixCacheMem(requested, ceiling, active_weights, ctx_kv, reserve),
    };
}

/// SSD-first budget semantics. The RAM-first formula treats the hot cache as what is left
/// after the live session's reserve, which double-counts (the resident entry's arrays are
/// shared with the live KV). Here the budget floors at one entry at the working context and
/// `--prefix-cache-mem` is the RAM allowance for idle entries on top (0 = none idle).
pub fn ssdFirstPrefixCacheMem(
    idle_requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve);
    const idle = @min(idle_requested, headroom);
    // Never 0: `initWithMem` reads 0 as "no byte cap".
    return @max(ctx_kv_bytes +| idle, 1);
}

/// The cache reserve the context sizer bills: at most half the room beside the weights and the
/// prefill, so a small Mac's context is not starved for a cache the load-time clamp shrinks anyway.
fn ctxCacheShare(cache_reserve: u64, ceiling: u64, active_mem: u64, transient_reserve: u64) u64 {
    return @min(cache_reserve, (ceiling -| (active_mem +| transient_reserve)) / 2);
}

/// The context a load-time bill must use, before `pinAutoContext` has run: on an auto boot
/// `getEffectiveContextLength` re-derives through `computeMemoryContext`, which reserves the
/// very ask being resolved and answered a 1024-token placeholder. `cache_reserve` is
/// `CTX_SIZING_CACHE_RESERVE` on both arms so the billed session is the advertised one; it
/// must never be the ask or anything derived from it. Runs `autoContextFrom`, the same helper
/// `autoContextFor` runs.
pub fn resolvedContextForLoad(
    explicit_ctx: u32,
    pinned_ctx: u32,
    ceiling: u64,
    active_mem: u64,
    cache_reserve: u64,
    transient_reserve: u64,
    per_tok: u64,
    ctx_cap: u32,
) u32 {
    if (explicit_ctx > 0) return explicit_ctx;
    if (pinned_ctx > 0) return pinned_ctx;
    return autoContextFrom(safeContextForBudget(
        ceiling,
        active_mem,
        ctxCacheShare(cache_reserve, ceiling, active_mem, transient_reserve) +| transient_reserve,
        per_tok,
        0,
    ), ctx_cap);
}

/// The SSD-first call site. Takes the static ceiling (the budget is a property of the machine),
/// the same reserve constant the RAM arm and `computeMemoryContext` pass, and the boot's
/// configured KV width (billing at bf16 on a `--kv-quant 8` boot was the 22,464-vs-13,824 MB defect).
fn ssdFirstSessionTokensNow(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u32 {
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        ceiling,
        active_mem,
        ctxSizingCacheReserve(config),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
    );
}

/// The bytes the SSD-first budget floors at: one session at the working context, at the width the cache stores.
fn ssdFirstSessionKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u64 {
    return sessionBytesPerToken(config, kv_bits) *|
        ssdFirstSessionTokensNow(config, kv_bits, ceiling, active_mem, chunk);
}

/// The SSD-first arm of `prefixCacheMemForLoad`, gate and log included. Null when the arch
/// gate is off; the caller falls through to `clampedPrefixCacheMem`.
fn ssdFirstBudgetForLoad(
    config: *model_mod.ModelConfig,
    requested: u64,
    ceiling: u64,
    active_mem: usize,
    ctx_kv: u64,
    transient_reserve: u64,
    idle_out: *u64,
    quiet: bool,
) ?u64 {
    // The predicate, shared with the spill site: without a disk tier the mode's floor would be RAM the server cannot use.
    if (prefix_cache_capacity == 0 or !prefix_cache_mod.ssdFirstActive(config, prefix_cache_disk_bytes > 0, prefix_cache_ram_enabled)) return null;
    const budget = ssdFirstPrefixCacheMem(
        requested,
        ceiling,
        active_mem,
        ctx_kv,
        transient_reserve,
    );
    // Publish the cache's cost BEYOND the live session, not the whole budget: `budget` includes
    // `ctx_kv`, and `computeMemoryContext` subtracts the published figure, so publishing the
    // whole would bill the session twice and halve the auto-context.
    publishResolvedPrefixCacheMem(budget -| ctx_kv);
    // The idle allowance, handed to the cache for `spillIdleEntries` to enforce.
    idle_out.* = budget -| ctx_kv;
    // On this arch `--prefix-cache-mem` is the idle allowance, not the whole cache; say so once.
    if (quiet) return budget;
    log.info("[hot-cache] SSD-first budget {d} MB = one session at the working context ({d} MB) + {d} MB idle (--prefix-cache-mem {d} MB = the IDLE allowance on this arch; 0 = no idle entries)\n", .{
        budget >> 20,
        ctx_kv >> 20,
        (budget -| ctx_kv) >> 20,
        requested >> 20,
    });
    return budget;
}

/// Impure wrapper for the model-load site (`Scheduler.doLoadOnInferenceThread`,
/// reached through the LoadParams/LoadRequest resolver pointer — the scheduler
/// deliberately has no server.zig import): the weights are resident there, so
/// `mlx_get_active_memory` is honest. Logs one line when the clamp bites, and publishes the
/// answer so the auto-context sizer stops reserving a budget the cache never got.
/// The hot-cache reserve the context is sized against at load time on the RAM-first arm: a
/// constant (the server's own `--prefix-cache-mem` default), never the ask. Reading the ask
/// let a 60GB ask collapse the advertised context to 870 tokens; context is the primary
/// claimant and the cache the residual.
const CTX_SIZING_CACHE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// The previous context-sizing cache reserve: the raw `--prefix-cache-mem` ask. Kept for
/// ungated archs so their advertised `context_length` does not move.
fn legacyPrefixCacheAsk() u64 {
    if (prefix_cache_capacity == 0) return 0;
    return prefix_cache_mem_bytes; // legacy_ask_read
}

/// The context the RAM-first clamp bills, through the one shared pure resolver.
fn ramFirstContextForLoad(config: *const model_mod.ModelConfig, kv_bits: u64, active_mem: u64, chunk: u32) u32 {
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        currentGpuMemoryCeiling(config, active_mem),
        active_mem,
        ctxSizingCacheReserve(config),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
    );
}

/// `revise` marks a post-load re-clamp (#364): the cache's own resident bytes come off the
/// machine read so it never bills itself, and the budget lines stay quiet.
pub fn prefixCacheMemForLoad(config: *model_mod.ModelConfig, requested: u64, revise: scheduler_mod.BudgetRevise, idle_out: *u64) u64 {
    // RAM-first default: no idle allowance. Written first so every early return carries it.
    idle_out.* = 0;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    active_mem -|= @as(usize, @intCast(@min(revise.exclude_bytes, std.math.maxInt(usize))));
    const kv_bits: u64 = defaultKvBits(config);
    // Arch gate: the budget resolver changed three inputs at once (static ceiling, the
    // load-time context resolver, the floor-width reserve), all measured on one hybrid trunk alone.
    // Ungated is the previous `prefixCacheMemForLoad`.
    if (!config.longCtxGated()) {
        const chunk: u64 = pinPrefillChunk(config);
        // `statePerTokenBilled` is 0 off that trunk.
        const ctx_kv: u64 = (kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config)) *|
            getEffectiveContextLength(config);
        const ask = requested;
        const clamped = clampedPrefixCacheMem(
            ask,
            currentGpuMemoryCeiling(config, active_mem),
            active_mem,
            ctx_kv,
            prefillTransientReserve(config, kv_bits, chunk),
        );
        publishResolvedPrefixCacheMem(clamped);
        if (revise.quiet) return clamped;
        if (ask > 0 and clamped < ask) {
            log.info("[hot-cache] budget clamped {d} -> {d} MB (weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{ ask >> 20, clamped >> 20 });
        } else if (requested == 0) {
            log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{clamped >> 20});
        }
        return clamped;
    }
    // SSD-first takes the budget before the RAM-first plan: its resident entry IS the live KV.
    const ssd_chunk: u64 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: on an auto boot it is still unpinned here.
    const ssd_ctx_kv: u64 = ssdFirstSessionKvBytes(config, kv_bits, staticGpuMemoryCeiling(), active_mem, @intCast(ssd_chunk));
    // Static ceiling on both arms (the budget is a property of the machine), and the clamp
    // reserve at the same width `planHotCache` reserves for: the ladder floor on an arch that
    // re-bills per request. The session is billed at the pinned width, the budget at the floor.
    const ssd_clamp_reserve: u64 = prefillTransientReserve(config, kv_bits, clampReserveWidth(config, @intCast(ssd_chunk)));
    if (ssdFirstBudgetForLoad(config, requested, staticGpuMemoryCeiling(), active_mem, ssd_ctx_kv, ssd_clamp_reserve, idle_out, revise.quiet)) |b| return b;
    // Pin first, then hand the pinned width in as the override.
    const pinned: u32 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: still a placeholder on an auto boot.
    const ctx_tokens = ramFirstContextForLoad(config, kv_bits, active_mem, pinned);
    const ask = defaultPrefixCacheAsk(requested, prefix_cache_mem_explicit, oneSessionEntryBytes(config, kv_bits, ctx_tokens));
    // Static ceiling, not the live one: the budget must be reproducible boot to boot.
    const plan = planHotCache(
        config,
        kv_bits,
        staticGpuMemoryCeiling(),
        active_mem,
        ctx_tokens,
        sizerCtxKvBytes(config, kv_bits),
        ask,
        pinned,
    );
    publishResolvedPrefixCacheMem(plan.budget);
    if (revise.quiet) return plan.budget;
    // Say so when the box is under external pressure right now, but do not shrink the budget for it.
    const live_ceiling: u64 = currentGpuMemoryCeiling(config, active_mem);
    if (live_ceiling < staticGpuMemoryCeiling()) {
        const free_gb = @as(f64, @floatFromInt(live_ceiling -| active_mem)) / (1024.0 * 1024.0 * 1024.0);
        log.info("[hot-cache] budget {d} MB (static ceiling); free at load {d:.1} GB — live admission will evict as needed\n", .{ plan.budget >> 20, free_gb });
    }
    if (ask != requested) {
        log.info("[hot-cache] budget {d} MB = one session at the working context (no --prefix-cache-mem)\n", .{ask >> 20});
    }
    if (ask > 0 and plan.budget < ask) {
        log.info("[hot-cache] budget clamped {d} -> {d} MB (chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ ask >> 20, plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    } else if (requested == 0) {
        log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    }
    return plan.budget;
}

test "clampedPrefixCacheMem: the budget never exceeds what the weights leave under the ceiling" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The crash's numbers: 96 GB ceiling, ~70 GB weights, ~6.4 GB of 262k-ctx
    // KV, ~4 GB prefill reserve, --prefix-cache-mem 40 GB → ~15.6 GB budget.
    const crash = clampedPrefixCacheMem(40 * GB, 96 * GB, 70 * GB, 6 * GB + (400 << 20), 4 * GB);
    try t.expect(crash >= 10 * GB and crash < 16 * GB);
    // Small model with plenty of headroom: the request is untouched.
    try t.expectEqual(8 * GB, clampedPrefixCacheMem(8 * GB, 96 * GB, 20 * GB, 2 * GB, 2 * GB));
    // requested == 0 (byte cap disabled today): headroom still bounds it.
    try t.expectEqual(96 * GB - 80 * GB, clampedPrefixCacheMem(0, 96 * GB, 70 * GB, 6 * GB, 4 * GB));
    // No headroom at all: never 0 — initWithMem reads 0 as "no byte cap".
    try t.expectEqual(@as(u64, 1), clampedPrefixCacheMem(40 * GB, 64 * GB, 70 * GB, 6 * GB, 4 * GB));
}

test "the clamp's live inversion, as arithmetic" {
    // The reported boot reduced to `clampedPrefixCacheMem`: ceiling - weights = 28,909 MiB, ctx KV 20,736 MiB.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const serving: u64 = 28_909 * MiB;
    const ctx_kv: u64 = 20_736 * MiB;
    const reserve_1024: u64 = 4_300 * MiB;
    const reserve_512: u64 = 2_470 * MiB;

    const ask10 = clampedPrefixCacheMem(10 * GiB, serving, 0, ctx_kv, reserve_1024);
    const ask24 = clampedPrefixCacheMem(24 * GiB, serving, 0, ctx_kv, reserve_512);
    try t.expectEqual(3_873 * MiB, ask10);
    try t.expectEqual(5_703 * MiB, ask24);
    try t.expect(ask10 < ask24); // the inversion, in the numbers that shipped

    // One ask-independent reserve: the clamp is `@min(ask, headroom)` over a headroom no ask can move.
    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const got = clampedPrefixCacheMem(ask, serving, 0, ctx_kv, reserve_1024);
        try t.expect(got >= prev);
        try t.expect(got <= @max((serving -| ctx_kv) -| reserve_1024, 1));
        prev = got;
    }
    try t.expectEqual(3_873 * MiB, clampedPrefixCacheMem(0, serving, 0, ctx_kv, reserve_1024));
}

/// The width a ladder rung actually forwards at for this prompt.
fn widthForRung(cfg: *const model_mod.ModelConfig, seq: u64, rung: u32) u32 {
    return @intCast(rungWidth(cfg, seq, rung, cfg.longCtxGated()));
}

test "an explicit --ctx-size keeps the load-time context byte-identical" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = 20_736;
    try t.expectEqual(@as(u32, 1_048_576), resolvedContextForLoad(1_048_576, 0, 0, 0, 99_000 * MiB, 99_000 * MiB, per_tok, 262_144));
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(0, 262_144, 0, 0, 99_000 * MiB, 0, per_tok, 1_048_576));
}

// Bar: the ring is billed once per slot, independent of prompt length.
test "admissionLogLevel: free on the first request after a load and at the edge, quiet in the middle" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;

    // A fresh load is always free.
    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 100 * GiB, true));

    // Inside 10% of what is available.
    try t.expectEqual(log.Level.info, admissionLogLevel(95 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(120 * GiB, 100 * GiB, false)); // already over
    // The boundary itself: exactly 0.9 is quiet.
    try t.expectEqual(log.Level.debug, admissionLogLevel(90 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(90 * GiB + 1, 100 * GiB, false));

    try t.expectEqual(log.Level.debug, admissionLogLevel(1 * GiB, 100 * GiB, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 0, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(std.math.maxInt(u64), 1, false));
    try t.expectEqual(log.Level.debug, admissionLogLevel(1, std.math.maxInt(u64), false));
}

test "admissionVerdict names the same three arms the guard acts on" {
    const t = std.testing;
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 10, .available = 20 }));
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 20, .available = 20 }));
    // Fits once the cache gives back what it can prove: `reclaimable`, never `evictable`.
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(.{ .needed = 30, .available = 20, .reclaimable = 10 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 30, .available = 20, .evictable = 1000, .reclaimable = 0 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 100, .available = 20, .reclaimable = 10 }));

    try t.expectEqualStrings("admit", AdmissionVerdict.admit.name());
    try t.expectEqualStrings("evict", AdmissionVerdict.evict.name());
    try t.expectEqualStrings("refuse", AdmissionVerdict.refuse.name());
}

/// The largest context this model's per-token footprint fits into RAM right now, ignoring the checkpoint's own maximum.
fn computeMemoryContext(config: *const model_mod.ModelConfig) u32 {
    const heads: u64 = config.num_attention_heads;
    if (heads == 0) return 16384;

    //   KV cache: the arch's own caching-layer count and K/V widths, billed at the active kv-quant width.
    const kv_bits: u64 = defaultKvBits(config);
    const per_tok: u64 = sessionBytesPerToken(config, kv_bits);

    // `total_ctx = 0` asks for the unshrunk cap: the widest forward any prompt can run.
    const chunk: u64 = @intCast(generate_mod.effectivePrefillChunk(
        config.prefillScoreHeadDim(),
        config.num_attention_heads,
        0,
        config.has_sliding_window,
        config.isMoe(),
        config.longCtxGated(),
        config.pinned_prefill_chunk,
    ));

    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);

    // Real reachable ceiling, so auto-context shrinks under external memory pressure (#64).
    const ceiling = currentGpuMemoryCeiling(config, active_mem);
    const transient = prefillTransientReserve(config, kv_bits, chunk);
    return safeContextForBudget(
        ceiling,
        active_mem,
        // The hot prefix cache fills to this cap over a session, and a prefill has to land on
        // top of whatever context we report. `CTX_SIZING_CACHE_RESERVE`, not the resolved
        // budget and not the ask: the ask starved the context, the resolved budget was a
        // one-step loop. Gated: off the gate this is the raw ask, so `/v1/models` is unchanged.
        ctxCacheShare(ctxSizingCacheReserve(config), ceiling, active_mem, transient) +| transient,
        per_tok,
        // 0 = do not clamp to the checkpoint's max here; the caller applies it after the margin.
        0,
    );
}

/// The cache reserve the context sizer bills. Gated: the ask-independent constant. Ungated:
/// the raw `--prefix-cache-mem`. Both load-time wrappers must pass the same value.
fn ctxSizingCacheReserve(config: *const model_mod.ModelConfig) u64 {
    if (prefix_cache_capacity == 0) return 0;
    return if (config.longCtxGated()) CTX_SIZING_CACHE_RESERVE else legacyPrefixCacheAsk();
}

/// Pure memory model behind checkAttentionMemory. All quantities in bytes,
/// 25% safety margin included. `chunk` must be the SAME prefill chunk
/// generate.zig will pick (generate_mod.effectivePrefillChunk) so the
/// admission guard and the real prefill cannot drift. Terms:
///   - kv: persistent for the request; billed at the ACTIVE kv-quant width
///     (+0.5 bit/elem group scale/bias overhead when quantized), not a
///     hardcoded fp16.
///   - scores: the composed-SDPA scratch [heads, chunk, seq] — materialized
///     only for head_dims no fused kernel covers (transformer.
///     prefillHeadDimFused: <= 128 via MLX, 256 via msv_attn_p256 — unfused
///     only under the MLX_SERVE_FUSED_256=0 kill switch or an exotic dim).
///     Bounded to ~one layer by the adaptive eval cadence
///     (transformer.prefillEvalCadence).
///   - dequant: dense-fp16 rebuild of the FULL quantized cache each layer
///     (KVCache.denseView under --kv-quant).
///   - mlp: QKV/MLP transient envelope; ~3 layers coexist between eval
///     points, and it is CHUNK-bounded, never seq-scaled — the old
///     8×seq×ffn envelope over-billed a 255K prompt by ~60 GB and rejected
///     requests that fit comfortably (PR #69).
///
/// `attn_keys` is how many keys one query actually reads (config.
/// prefillAttnKeys): `seq` for dense causal attention, a much smaller bound for
/// a sparse arch. It scales the SCORE term only — KV/dequant are storage, and
/// a sparse reader still stores every latent it might later attend to.
/// `kv_per_tok` is the whole-model DENSE (f16) KV bytes for one token —
/// `ModelConfig.kvBytesPerToken()`. It replaced a `layers` parameter because
/// neither the layer count nor the per-head width is uniform once an arch
/// interleaves linear-attention layers or stores keys wider than values.
///
/// `hdim` and `score_hdim` are two DIFFERENT widths and only coincide on a
/// symmetric arch. `hdim` is the STORED width the quantized-KV dequant
/// transient is read at; `score_hdim` is the width the score is contracted at
/// (`ModelConfig.prefillScoreHeadDim`), which is what decides whether a fused
/// kernel exists — MLA scores over 192 while storing values at 128, and billing
/// the score term against `head_dim` 128 declared it fused and billed ZERO for
/// a composed path that really does materialize [heads, chunk, seq].
/// KV bytes for ONE token at the width the cache actually STORES. `dense` is
/// `ModelConfig.kvBytesPerToken()` — the arch's own caching-layer count and its
/// own K and V widths, at fp16. A quantized cache stores `kv_bits` per element
/// plus the group scale/bias pair, which is the `+1` in `(2·bits + 1)/32`
/// (~0.5 bit/elem at group 64 with bf16 params).
///
/// Both the auto-context sizer and the prefill admission guard bill through
/// here. The sizer used to bill fp16 unconditionally, so a `--kv-quant 4`
/// server reported — and served — under a third of the context it can hold.
pub fn kvBytesPerTokenAtBits(dense: u64, kv_bits: u64) u64 {
    if (kv_bits >= 16) return dense;
    return dense * (2 * kv_bits + 1) / 32;
}

/// The chunk-independent floor every prefill pays: MLX runtime scratch, the
/// KV cache's proportional capacity growth (old + new buffer coexist across a
/// grow), and the graph live-set no other term models. MEASURED as the
/// intercept of peak-above-steady-state against the chunk, on five checkpoints
/// spanning 2.6B to 30B (M4 Max, 2026-08-14): 0.39 GB (27B), 0.67
/// (4B), 0.81 (2.6B), ~0.1 (gemma4 26B-A4B), 1.27 (30B). It
/// does NOT scale with the weights — a dense bf16 checkpoint pays it too — so
/// it is billed as what it measures as: a constant.
const PREFILL_RUNTIME_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// How many MoE layers' gather-sort expansions coexist between eval points.
/// Mirrors `Transformer.MOE_EVAL_EVERY_N_LAYERS` — the MoE prefill loop evals
/// every 4th layer, so 4 layers' worth is what the peak holds.
const MOE_PREFILL_COEXIST: u64 = transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT;

/// What a warm turn already holds. A RAM hot-cache restore hands the slot the entry's KV
/// buffers without allocating another copy, so `available` is already net of them and
/// billing them again refused a 786k warm turn whose own entry was resident. KV only: the
/// QSA history is privatised on restore. Credited only while the prefix fits the destination
/// capacity (a grow allocates a fresh buffer beside the protected entry).
pub const WarmPrefix = struct {
    /// Prompt rows the hot cache restored. 0 = cold.
    matched_tokens: u64 = 0,
    /// Tokens of capacity the restored buffers hold (`KVCache.residentCapacityTokens`). The
    /// connection thread passes 0 and defers the warm decision to the inference thread.
    capacity_tokens: u64 = 0,
    /// Did the restore check out its entry (`prefix_cache.checkoutEligible`)? A shared restore
    /// is copied whole by the first append, so crediting it is an under-bill.
    will_donate: bool = false,

    /// Rows of KV this request will not allocate: the rows the restored buffer already holds.
    /// Capacity is a ceiling on the credit, never a switch: a growing chain outruns its capacity
    /// on nearly every turn. The grow's own cost is `reserved - credited` plus
    /// `PrefillRequestTerms.grow_coexist_bytes`. `will_donate` gates the whole credit.
    pub fn creditedRows(self: WarmPrefix, reserved: u64) u64 {
        if (!self.will_donate) return 0;
        if (self.matched_tokens == 0) return 0;
        return @min(@min(self.matched_tokens, self.capacity_tokens), reserved);
    }

    /// Does this append grow the buffers at all? The per-layer grow fires on `offset + new_len
    /// > capacity`; a reservation above the prompt allocates nothing by itself.
    pub fn grows(self: WarmPrefix, seq: u64) bool {
        if (!self.will_donate or self.matched_tokens == 0) return false;
        return seq > self.capacity_tokens;
    }
};

/// PURE: the most KV-caching layers that fall inside one window of `window` consecutive layers,
/// i.e. how many old buffers can be alive at once at that eval cadence.
fn attnLayersPerEvalWindow(config: *const model_mod.ModelConfig, window: u32) u32 {
    const n = config.num_hidden_layers;
    if (window == 0 or n == 0) return 0;
    return @min(window, n);
}

/// PURE: bytes of OLD KV buffer alive beside the new ones while a warm append grows the cache.
/// `growQuantBuf` drops our reference to the old buffer at once, but the graph node holds it
/// until it EVALUATES, so the coexistence window is the prefill loop's eval cadence rather than
/// the size of the cache. A forward narrower than `prefillEvalCadenceApplies` runs one eval for
/// the whole layer loop and therefore does pay the whole old cache.
fn growCoexistBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix, seq: u64, kv_per_tok: u64) u64 {
    if (!warm.grows(seq)) return 0;
    const attn: u32 = config.num_hidden_layers;
    if (attn == 0) return 0;
    // A MoE runs a forward under 1024 tokens with one eval for the whole layer loop, and any append
    // can have one (a short last chunk), so bill it the whole old cache: an upper bound, and the
    // plain one (a per-chunk width rule misses checkpoint-aligned first chunks).
    if (transformer_mod.Transformer.moeShortPrefillSkipsCadence(config)) return warm.capacity_tokens *| kv_per_tok;
    const span: u64 = seq -| warm.matched_tokens;
    const window: u32 = if (transformer_mod.Transformer.prefillEvalCadenceApplies(@intCast(@min(span, 1 << 20))))
        attnLayersPerEvalWindow(config, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS)
    else
        attn;
    if (window == 0) return 0;
    // The old buffer's own size, spread over the layers that cache.
    return warm.capacity_tokens *| kv_per_tok / attn *| window;
}

/// Per-request bytes that scale with the prompt rather than the chunk.
pub const PrefillRequestTerms = struct {
    /// Dense operands of one caching layer; null retains the generic GQA width.
    dequant_width_per_token: ?u64 = null,
    /// Cache capacity reserved at admission beyond `seq * kv_per_tok`: the generation headroom
    /// plus a chunk of slack, so no buffer grows during the prefill (#353).
    reserved_kv_bytes: u64 = 0,
    /// Per-request state the arch keeps outside the KV cache (the QSA indexer history), at its reserved size.
    state_bytes: u64 = 0,
    /// Bytes of the KV terms above already resident in the buffer the restore handed this slot
    /// (`WarmPrefix`). KV only; subtracted once, in `prefillMemoryNeeded`.
    shared_resident_bytes: u64 = 0,
    /// Old KV buffers that coexist with the new ones while a warm append GROWS the cache.
    /// Zero when nothing grows and when the restore was shared (the whole copy is billed
    /// uncredited instead).
    grow_coexist_bytes: u64 = 0,
    /// Forward width from which the dequant+GEMM route fires (`prefillDqGemmMinRows`).
    dq_min_rows: u64 = transformer_mod.PREFILL_DQ_GEMM_MIN_M,
};

pub fn prefillMemoryNeeded(seq: u64, heads: u64, kv_heads: u64, kv_per_tok: u64, hdim: u64, score_hdim: u64, hidden: u64, ffn: u64, kv_bits: u64, chunk: u64, attn_keys: u64, stream_per_tok: u64, dequant_weights: u64, req: PrefillRequestTerms) u64 {
    // A forward is never wider than the prompt: a prompt shorter than the
    // chunk runs ONE seq-wide forward, so the per-chunk transient envelope
    // scales with min(chunk, seq) — billing the raw chunk cap rejected a
    // 31-token prompt for "needing" 6.2 GB on a memory-squeezed host
    // (live 2026-07-14). No-op for prompts >= one chunk.
    const fwd: u64 = @min(chunk, @max(seq, 1));
    const kv_bytes: u64 = seq * kvBytesPerTokenAtBits(kv_per_tok, kv_bits);
    const scores: u64 = if (!transformer_mod.prefillHeadDimFused(@intCast(score_hdim))) heads * fwd * @min(attn_keys, seq) * 2 else 0;
    const dequant: u64 = if (kv_bits < 16) (if (req.dequant_width_per_token) |width| seq * width else 2 * seq * kv_heads * hdim * 2) else 0;
    const mlp: u64 = 8 * fwd * @max(hidden, ffn) * 2;
    // The MLP envelope alone is not the whole per-token working set: an arch
    // with its own prefill streams (linear-attention layers, MoE gather-sort
    // expansion — `prefillStreamBytesPerToken`) adds to it. Floored at the
    // historical 3-envelope bill so no arch's admission ever LOOSENS: that
    // figure is measured 1.7x-5x conservative on plain attention archs
    // and would be a 33% UNDER-bill on a GatedDeltaNet hybrid.
    const envelope: u64 = @max(3 * mlp, mlp + fwd * stream_per_tok);
    // The dequant+GEMM prefill route (transformer.prefillDqGemm) materializes
    // a bf16 copy of the weight it is about to multiply plus its transpose,
    // and only at forwards wide enough to take that route. Kill-switch A/B
    // (MLX_SERVE_PREFILL_DQ_GEMM=0): +0.51 GB on a 27B at chunk
    // 2048, +0.17-0.21 on a 2.6B, ~0 at chunk 8192 where the envelope
    // already dominates.
    const dq_weights: u64 = if (fwd >= req.dq_min_rows) dequant_weights else 0;
    const gross: u64 = kv_bytes + req.reserved_kv_bytes + req.state_bytes +
        req.grow_coexist_bytes + scores + dequant + envelope + dq_weights + PREFILL_RUNTIME_FLOOR_BYTES;
    // The one place a warm turn's resident rows leave the bill, inside the 5/4.
    return (gross -| req.shared_resident_bytes) * 5 / 4;
}

/// Per-TOKEN bytes of prefill working set that live OUTSIDE the MLP envelope,
/// because the arch runs streams the envelope does not model. MEASURED as the
/// slope of peak-above-steady-state against the prefill chunk (M4 Max,
/// 2026-08-14, one request per boot, byte-identical across repeats):
///
/// | checkpoint | measured B/chunk-token | one MLP envelope | this term |
/// |---|---|---|---|
/// | 30B (dense) | 190,300 | 319,488 | 0 |
/// | 2.6B (conv hybrid) | 171,150 | 172,032 | 0 |
/// | 4B (GatedDeltaNet) | 513,500 | 147,456 | 393,216 |
/// | 27B (GatedDeltaNet) | 1,263,700 | 278,528 | 983,040 |
/// | gemma4 26B-A4B (MoE) | 486,800 | 90,112 | 450,560 |
///
/// Two families need it. A linear-attention hybrid holds one chunk-wide q/k/v
/// stream per LINEAR layer — all of them, not the ~3 the eval cadence bounds
/// (48 x 10240 elems on the 27B, 24 x 8192 on the 4B, both landing within 7%
/// of the measured excess over the envelope). A MoE prefill sorts and gathers,
/// which replicates the hidden stream `top_k` times per layer alongside the
/// expert rows, for the 4 layers the MoE eval cadence lets coexist.
///
/// The linear term bills only `cadence + 1` layers: since the conv_state tail is
/// an owned copy evaluated at each cadence point, older layers' chunk inputs are
/// released inside the loop (the old all-layers figure WAS that pin).
///
/// Attention-only archs return 0 and keep exactly the bill they had.
fn prefillStreamBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    return prefillStreamBytesPerTokenOn(config, transformer_mod.verifyQmmNaxAvailable());
}

fn prefillStreamBytesPerTokenOn(config: *const model_mod.ModelConfig, nax: bool) u64 {
    _ = nax;
    var per_tok: u64 = 0;
    if (config.isMoe()) {
        const top_k: u64 = @max(@as(u64, config.num_experts_per_tok), @as(u64, transformer_mod.Transformer.topKScheduleMaxK(config)), 1);
        per_tok += MOE_PREFILL_COEXIST * top_k * 2 *
            (@as(u64, config.hidden_size) + config.moe_intermediate_size) * 2;
    }
    return per_tok;
}

/// The dequantized-weight working set `transformer.prefillDqGemm` materializes
/// (a bf16 copy of the weight plus its transpose, ~3 alive across the layer
/// loop). Chunk-INDEPENDENT — it is weights, not activations — and billed only
/// where the route can fire: affine-quantized weights, and (inside
/// `prefillMemoryNeeded`) forwards at least `PREFILL_DQ_GEMM_MIN_M` wide.
/// Reads the same kill switch the route does, so an A/B with the route off
/// does not get billed for it.
fn prefillDequantWeightBytes(config: *const model_mod.ModelConfig) u64 {
    if (config.quant_bits == 0 or config.quant_mode != .affine) return 0;
    if (!transformer_mod.prefillDqGemmEnabled()) return 0;
    return 3 * prefillFfnWidth(config) * @as(u64, config.hidden_size) * 2;
}

fn prefillFfnWidth(config: *const model_mod.ModelConfig) u64 {
    const per_tok: u64 = @max(@as(u64, config.num_experts_per_tok), @as(u64, transformer_mod.Transformer.topKScheduleMaxK(config)), 1);
    const moe_w: u64 = @as(u64, config.moe_intermediate_size) * per_tok +
        config.shared_expert_intermediate_size;
    const dense_w: u64 = if (config.intermediate_size_declared) config.intermediate_size else 0;
    const w = @max(dense_w, moe_w);
    return if (w > 0) w else config.intermediate_size;
}

/// Whether the MLX-prefill attention-memory preflight applies to a request.
/// The guard (`checkAttentionMemory`/`prefillMemoryNeeded`) models the MLX
/// transformer's per-token working set. An engine-backed model (the Neural
/// Engine path) never takes that path, so the estimate is pure fiction for it.
/// Skip the memory guard whenever an engine-backed model will serve; the
/// context-length guard still bounds the prompt.
fn mlxMemoryGuardApplies(engine_backed: bool) bool {
    return !engine_backed;
}

/// Estimate peak GPU memory for prefill and reject if it would exceed the Metal
/// working-set ceiling. Exceeding it throws an uncatchable C++ exception on a
/// Metal completion-handler thread and kills the process, so PREVENTION is the
/// only lever — this guard must track what the prefill actually allocates
/// (prefillMemoryNeeded above; chunk choice from generate.effectivePrefillChunk).
/// `kv_override` is the per-request `kv_quant` body field where the surface
/// parses one (chat/messages/responses); null falls back to the process default.
/// `lm` is the resolved model: an engine-backed model skips this guard
/// entirely (see `mlxMemoryGuardApplies`) — this is the single chokepoint, so
/// every current and future call site is covered without per-site gating.
/// Per-token bytes of the arch's out-of-cache request state as billed: the QSA indexer history
/// at the copies a live slot holds (one by default; two with `MLX_SERVE_QSA_HISTORY_SHARE=0`
/// or `MLX_SERVE_KV_RESERVE=0`), plus the f32 block-score bank the two-copy slack used to
/// hide. One helper because the auto-context sizer and the admission guard must agree.
pub fn statePerTokenBilled(config: *const model_mod.ModelConfig) u64 {
    _ = config;
    return 0;
}

/// Tokens of cache capacity a request reserves at admission (prompt + generation + one chunk
/// of slack); `KVCache.reserve_tokens` allocates it on the first grow so the prefill never
/// runs a second grow. The guard and the allocator are the same number by construction.
pub fn reservedCacheTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
    return transformer_mod.KVCache.reservedTokens(seq, max_tokens, chunk, ctx);
}

fn sessionBytesPerToken(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    return kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config);
}

/// The per-request terms of the admission bill, in one place.
pub fn prefillRequestTerms(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64, warm: WarmPrefix) PrefillRequestTerms {
    // Arch gate for the reservation and state terms (measured on one hybrid trunk alone; the
    // reservation's allocator twin is gated too, so an ungated guard billed memory never reserved).
    const dq_min_rows: u64 = transformer_mod.prefillDqGemmMinRows(config.quant_bits);
    if (!config.longCtxGated()) {
        // A checked-out entry is the slot's own buffer on every arch; both terms are zero for a share.
        const kv_per_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
        return .{
            .shared_resident_bytes = warm.creditedRows(seq) *| kv_per_tok,
            .grow_coexist_bytes = growCoexistBytes(config, warm, seq, kv_per_tok),
            .dq_min_rows = dq_min_rows,
        };
    }
    // `reservedTokens` returns 0 below its length threshold; floor the reserved length at `seq`.
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(config)), seq);
    // Only the headroom is new here: the prompt's own rows are already billed.
    const kv_per_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    // Rows the restore hands over without allocating another copy. KV only: the history is
    // privatised on restore (`seedCapBuf`), so crediting it cancelled the one-copy bill outright.
    const credited = warm.creditedRows(reserved);
    return .{
        .reserved_kv_bytes = (reserved -| seq) * kv_per_tok,
        .state_bytes = reserved * statePerTokenBilled(config),
        .shared_resident_bytes = credited *| kv_per_tok,
        .grow_coexist_bytes = growCoexistBytes(config, warm, seq, kv_per_tok),
        .dq_min_rows = dq_min_rows,
    };
}

/// A slot that errored reports why: the MLX error latch's `error.OutOfMemory` is a memory
/// failure the operator can act on. One helper for the three slot-drain sites.
fn slotFailure(slot: *scheduler_mod.Slot) anyerror {
    // A pre-prefill admission refusal is not an abandoned generation: the client gets a 400.
    if (slot.errorNameIs("PrefillDoesNotFit")) {
        prefill_refusal = if (slot.refused_bill) |bill| .{ .tokens = slot.full_prompt.len, .needed = bill[0], .available = bill[1] } else null;
        return error.PrefillDoesNotFit;
    }
    if (slot.errorNameIs("PromptLogprobsUnavailable")) return error.PromptLogprobsUnavailable;
    if (slot.errorIsMemory()) return error.GenerationOutOfMemory;
    return error.GenerationFailed;
}

/// The message `error.GenerationOutOfMemory` sends on every surface and both paths.
const GEN_OOM_MSG = "The engine ran out of GPU memory during this request and it was abandoned. The server is still running. Reduce the prompt length, lower --ctx-size, or free memory on the machine.";

/// The body of the pre-flight memory 400, quoting the hot-cache credits the guard compared.
/// Caller owns the returned bytes.
fn memoryRefusalMessage(
    allocator: std.mem.Allocator,
    prompt_len: usize,
    needed_mb: u64,
    avail_mb: u64,
    bill: AdmissionBill,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB is available and the hot prefix cache holds ~{d}MB more, all of which can be reclaimed (~{d}MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.", .{ prompt_len, needed_mb, avail_mb, bill.evictable / (1024 * 1024), bill.evictionCredit() / (1024 * 1024) });
}

/// The message `error.PrefillDoesNotFit` sends: refused before its first forward by the same
/// estimator the guard uses, re-asked after the hot cache gave back everything it could.
const PREFILL_NOFIT_MSG = "This prompt does not fit in GPU memory even after freeing the prefix cache; it was refused before any work started. Reduce the prompt length, lower --ctx-size, or free memory on the machine (the server log quotes the byte counts it compared).";

/// The figures of the `PrefillDoesNotFit` this connection thread last drained, read once by
/// `mapGenerationError`: the error name alone cannot carry them past `slotFailure`.
threadlocal var prefill_refusal: ?struct { tokens: usize, needed: u64, available: u64 } = null;

/// `PREFILL_NOFIT_MSG` with the figures the refusal compared, worded like the guard's 400 so a client
/// reads both refusals the same way. The bare message when no figures were drained.
fn prefillNoFitMessage(buf: []u8) []const u8 {
    const refusal = prefill_refusal orelse return PREFILL_NOFIT_MSG;
    prefill_refusal = null;
    const mb = 1024 * 1024;
    return std.fmt.bufPrint(buf, "This prompt ({d} tokens) does not fit in GPU memory even after freeing the prefix cache: it requires ~{d}MB GPU memory but only ~{d}MB is available. It was refused before any work started. Reduce the prompt length, lower --ctx-size, or free memory on the machine.", .{ refusal.tokens, refusal.needed / mb, refusal.available / mb }) catch PREFILL_NOFIT_MSG;
}

/// A generation failure as the wire sees it: one status, one message, the type.
pub const GenErrorWire = struct {
    status_line: []const u8,
    code: u32,
    openai_type: []const u8,
    message: []const u8,
};

/// The mapping from a failed generation to what the client is told. It existed only on the
/// non-streaming arms; streaming arms wrote a bare `server_error` into an SSE frame, so agents
/// never saw the actionable 400/503. The unknown arm keeps the error name at a mapped 500.
pub fn mapGenerationError(err: anyerror, buf: []u8) GenErrorWire {
    // Whatever the request saw, a lost context is the cause and a restart the remedy.
    if (mlx.gpuContextLost()) return .{
        .status_line = "503 Service Unavailable",
        .code = 503,
        .openai_type = "server_error",
        .message = gpu_context_lost_message,
    };
    return switch (err) {
        // The MLX working-set latch (#353): a memory 503.
        error.GenerationOutOfMemory, error.OutOfMemory => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .message = GEN_OOM_MSG,
        },
        // Refused before the first forward: the request is too large, a named 400.
        error.PrefillDoesNotFit => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .message = prefillNoFitMessage(buf),
        },
        error.EmptyPrompt => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .message = "the prompt encodes to no tokens",
        },
        // Backstop for a forward that returns fewer rows than the prompt: the request's shape, not a fault.
        error.PromptLogprobsUnavailable => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .message = PROMPT_LOGPROBS_UNAVAILABLE_MSG,
        },
        error.GenerationFailed => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .message = "generation failed",
        },
        else => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .message = std.fmt.bufPrint(buf, "Internal server error: {s}", .{@errorName(err)}) catch
                "Internal server error",
        },
    };
}

/// The terminal SSE frame for the OpenAI dialects, with a `finish_reason: "error"` choice so a
/// chunk-only client terminates.
fn buildOpenAiStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\data: {{"error":{{"message":{s},"type":"{s}","param":null,"code":{d}}},"choices":[{{"index":0,"delta":{{}},"finish_reason":"error"}}]}}
    ++ "\n\n", .{ esc, w.openai_type, w.code });
}

/// Which wire dialect a generative surface speaks its errors in.
const ErrorSurface = union(enum) {
    openai,
};

/// The error arm for every generative surface, streaming or not: before the SSE head the
/// stream gets the same response its non-streaming twin would; past it the same `type` and
/// `message` ride the surface's terminal `error` event.
/// A client that hung up mid-stream (`WriteFailed`) is not a server error.
fn logStreamError(err: anyerror) void {
    if (err == error.WriteFailed) log.debug("  -> client disconnected mid-stream\n", .{}) else log.err("  -> streaming error: {}\n", .{err});
}

fn sendGenerationError(allocator: std.mem.Allocator, stream: *Conn, err: anyerror, surface: ErrorSurface) !void {
    var msg_buf: [320]u8 = undefined;
    const w = mapGenerationError(err, &msg_buf);

    if (!stream.sse_headers_sent) {
        return sendErrorResponse(allocator, stream, w.status_line, w.openai_type, w.message, w.code);
    }

    switch (surface) {
        .openai => {
            const body = try buildOpenAiStreamError(allocator, w);
            defer allocator.free(body);
            try stream.writeAllNoFlush(body);
            try stream.writeAll("data: [DONE]\n\n");
        },
    }
}

/// The admission bill and the headroom it is compared against, so the connection thread's
/// guard and the inference thread's evict-or-refuse decision cannot compute them differently.
pub const AdmissionBill = struct {
    needed: u64 = 0,
    available: u64 = 0,
    /// Bytes the hot prefix cache is holding. Reported in a refusal, never credited (see `reclaimable`).
    evictable: u64 = 0,
    /// The part of `evictable` an eviction can prove it will return: residency minus the
    /// largest entry (a restore pins at most one).
    reclaimable: u64 = 0,
    /// The prefill width this bill was taken at, so a refusal can name it.
    chunk: u64 = 0,

    pub fn fits(self: AdmissionBill) bool {
        return self.needed <= self.available;
    }

    /// The one credit the decision and both sentences read: `reclaimable`, never `evictable`
    /// (a refusal once quoted the slot's own checked-out prefix as "evicted").
    pub fn evictionCredit(self: AdmissionBill) u64 {
        return self.reclaimable;
    }

    /// Fits once the cache has given back what it provably can. Credits `reclaimable`, not
    /// `evictable`: a 383k prompt was admitted on one entry it then restored from and refused
    /// after the eviction pass returned nothing.
    pub fn fitsAfterEviction(self: AdmissionBill) bool {
        return self.needed <= self.available +| self.evictionCredit();
    }
};

/// Unloads idle resident models, least recently used first, until `fits` holds; false when
/// nothing idle is left. An idle model is memory its next request reloads.
pub fn fitAfterUnloadingIdle(ctx: anytype, fits: fn (@TypeOf(ctx)) bool, unloadOne: fn (@TypeOf(ctx)) bool) bool {
    while (!fits(ctx)) if (!unloadOne(ctx)) return false;
    return true;
}

test "a prompt that does not fit unloads idle models, oldest first, until it fits" {
    const t = std.testing;
    const Fake = struct {
        needed: u64,
        available: u64,
        idle: []const u64, // resident bytes, least recently used first
        unloaded: usize = 0,
        fn fits(f: *@This()) bool {
            return f.needed <= f.available;
        }
        fn unloadOne(f: *@This()) bool {
            if (f.unloaded == f.idle.len) return false;
            f.available += f.idle[f.unloaded];
            f.unloaded += 1;
            return true;
        }
    };
    // Already fits: nothing is unloaded.
    var a = Fake{ .needed = 5, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(fitAfterUnloadingIdle(&a, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 0), a.unloaded);
    // The live shape: a prompt needs 10.5 GB beside 9.4 GB free and an idle 14.7 GB image model.
    var b = Fake{ .needed = 10_558, .available = 9_466, .idle = &.{ 14_730, 8_000 } };
    try t.expect(fitAfterUnloadingIdle(&b, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 1), b.unloaded);
    // Nothing idle left to give back: still refused.
    var c = Fake{ .needed = 50, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(!fitAfterUnloadingIdle(&c, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 2), c.unloaded);
}

test "a refusal names the credit it COMPARED, never the residency it cannot take" {
    const t = std.testing;
    const mb: u64 = 1024 * 1024;

    // The live shape: the whole residency is the checked-out entry.
    const pinned_only = AdmissionBill{
        .needed = 13_664 * mb,
        .available = 13_374 * mb,
        .evictable = 7_802 * mb,
        .reclaimable = 0,
        .chunk = 512,
    };
    try t.expectEqual(@as(u64, 0), pinned_only.evictionCredit());
    try t.expectEqual(7_802 * mb, pinnedResidentBytes(pinned_only));
    try t.expect(!pinned_only.fits());
    try t.expect(!pinned_only.fitsAfterEviction());
    try t.expectEqual(
        pinned_only.needed <= pinned_only.available +| pinned_only.evictionCredit(),
        pinned_only.fitsAfterEviction(),
    );

    // The same residency, genuinely reclaimable: the credit is real and the request is admitted to evict.
    var reclaimable = pinned_only;
    reclaimable.reclaimable = 7_802 * mb;
    try t.expectEqual(7_802 * mb, reclaimable.evictionCredit());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(reclaimable));
    try t.expect(reclaimable.fitsAfterEviction());
}

/// Everything the admission guard bills for one request at one prefill width, so the
/// per-request chunk chooser prices a rung with the same function that admits it. `chunk` is
/// the width the forward will run, already through `generate.effectivePrefillChunk`.
pub fn prefillNeededAtChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    chunk: u64,
    warm: WarmPrefix,
) u64 {
    const heads: u64 = config.num_attention_heads;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = prefillFfnWidth(config);
    const drafter: u64 = (seq +| @min(@as(u64, max_tokens), transformer_mod.KVCache.RESERVE_GEN_HEADROOM)) *| config.drafter_ctx_bytes_per_token;
    return prefillMemoryNeeded(seq, heads, kv_heads, config.kvBytesPerToken(), hdim, config.prefillScoreHeadDim(), hidden, ffn, kv_bits, chunk, config.prefillAttnKeys(seq), prefillStreamBytesPerToken(config), prefillDequantWeightBytes(config), prefillRequestTerms(config, seq, max_tokens, kv_bits, chunk, warm)) +
        drafter;
}

/// Headroom a rung must clear beyond its own bill when the long-context gate is what widened
/// it: past that width the machine cap frozen at load no longer bounds the forward, and
/// `prefillMemoryNeeded` is documented to UNDER-bill quantized prefill by about this much.
/// Narrower rungs keep the plain bill, so no other admission decision moves.
pub const PREFILL_WIDE_RUNG_MARGIN_PCT: u64 = 22;

/// Test hook for `perRequestPrefillChunkEnabled`'s kill switch.
pub var per_request_chunk_override: ?bool = null;

/// The arch capability AND the kill switch (`MLX_SERVE_PREFILL_CHUNK_PER_REQUEST=0` restores the load-time pin).
pub fn perRequestPrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    if (!config.perRequestPrefillChunk()) return false;
    if (per_request_chunk_override) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_PREFILL_CHUNK_PER_REQUEST") orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

/// The widest ladder rung this request can afford, priced by the same estimator that admits
/// it against post-eviction memory. Candidates are priced at the width the forward will run
/// (`effectivePrefillChunk` caps by arch), and the width is returned, not the rung. An explicit
/// `--prefill-chunk` wins outright; nothing fitting means the ladder floor.
pub fn chooseRequestPrefillChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    available: u64,
    load_time_pin: u32,
    chunk_override: u32,
    warm: WarmPrefix,
) u32 {
    // The arch gate first: an ungated arch's answer is the load-time pin, as before.
    if (!perRequestPrefillChunkEnabled(config)) return load_time_pin;
    if (chunk_override > 0) return chunk_override;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const width: u64 = rungWidth(config, seq, rung, config.longCtxGated());
        const bill = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, width, warm);
        const needed = if (width > rungWidth(config, seq, rung, false))
            bill *| (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        if (needed <= available) return @intCast(width);
    }
    return @intCast(rungWidth(config, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], config.longCtxGated()));
}

/// The width one ladder rung forwards at. `long_ctx_gated = false` asks what the rung would be
/// without the long-context gate, i.e. the width the load-time machine cap still bounds.
fn rungWidth(config: *const model_mod.ModelConfig, seq: u64, rung: u32, long_ctx_gated: bool) u64 {
    // `effectivePrefillChunk` can return `MLX_SERVE_PREFILL_CHUNK` verbatim; clamp before narrowing.
    return @min(
        @as(u64, generate_mod.effectivePrefillChunk(
            config.prefillScoreHeadDim(),
            config.num_attention_heads,
            @intCast(seq),
            config.has_sliding_window,
            config.isMoe(),
            long_ctx_gated,
            rung,
        )),
        @as(u64, std.math.maxInt(u32)),
    );
}

/// Impure wrapper the scheduler reaches through `prefill_request_chunk`, mirroring
/// `prefillFitsNow`'s inputs. Reads live memory, so it must run after the eviction pass.
pub fn requestPrefillChunkNow(
    config: *const model_mod.ModelConfig,
    prompt_len: usize,
    max_tokens: u32,
    kv_cfg: transformer_mod.KVQuantConfig,
    unchunked_prefill: bool,
    warm_matched: u64,
    warm_capacity: u64,
    warm_will_donate: bool,
) u32 {
    const pin: u32 = config.pinned_prefill_chunk;
    const explicit: u32 = explicitPrefillChunk();
    if (explicit > 0) return explicit;
    // The vision kill switch forwards the whole prompt; there is no chunk to choose.
    if (unchunked_prefill) return pin;
    if (!perRequestPrefillChunkEnabled(config)) return pin;

    const seq: u64 = @intCast(prompt_len);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const available: u64 = currentGpuMemoryCeiling(config, active_mem) -| active_mem;
    const warm = WarmPrefix{ .matched_tokens = warm_matched, .capacity_tokens = warm_capacity, .will_donate = warm_will_donate };
    const chosen = chooseRequestPrefillChunk(config, seq, max_tokens, kv_bits, available, pin, 0, warm);
    if (chosen != pin) {
        const width: u64 = chosen;
        const terms = prefillRequestTerms(config, seq, max_tokens, kv_bits, width, warm);
        const kv_bytes: u64 = seq *| kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| terms.reserved_kv_bytes;
        log.debug("[prefill] chunk {d} for this request (reserve {d} MB beside KV {d} MB)\n", .{ width, prefillTransientReserve(config, kv_bits, width) >> 20, kv_bytes >> 20 });
    }
    return chosen;
}

/// Test hook for `adaptivePrefillChunkEnabled`'s kill switch.
pub var adaptive_chunk_override: ?bool = null;

/// The per-chunk adaptive width: subordinate to the per-request gate, plus
/// `MLX_SERVE_PREFILL_CHUNK_ADAPTIVE=0`, plus "no operator pinned a width".
pub fn adaptivePrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    // Arch first: the cheapest and most selective, and this runs once per chunk boundary.
    if (!config.perRequestPrefillChunk()) return false;
    if (explicitPrefillChunk() > 0) return false;
    if (generate_mod.envPrefillChunk() > 0) return false;
    if (!perRequestPrefillChunkEnabled(config)) return false;
    if (adaptive_chunk_override) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_PREFILL_CHUNK_ADAPTIVE") orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

/// Consecutive supporting probes a widening needs: the probe is taken after `mlx_clear_cache`
/// and never contains a chunk's own peak, so the second reading pays for that blind spot.
pub const PREFILL_WIDEN_PROBES: u8 = 2;

/// What one chunk of `width` costs beyond what is already resident: its transient envelope
/// plus the per-token state the chunk is about to write. The growth term is billed even under
/// the default reservation (`MLX_SERVE_KV_RESERVE=0` and short prompts do grow per chunk).
pub fn prefillChunkCost(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    width: u64,
    kv_len: u64,
) u64 {
    const per_tok: u64 = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +|
        statePerTokenBilled(config);
    return prefillTransientReserveAtKv(config, kv_bits, width, kv_len) +| (width *| per_tok);
}

/// The width the next chunk runs at. Asymmetric because a Metal abort is uncatchable: HOLD
/// while `headroom >= cost(current)` (the estimator carries its own 5/4); STEP DOWN at once,
/// by as many rungs as it takes, floored at `PREFILL_CHUNK_FLOOR`; WIDEN one rung only at
/// 1.25x its cost, after `PREFILL_WIDEN_PROBES` supporting probes, up to `cap`, never after a
/// step-down. `cap` is the widest width this arch forwards for this prompt, not a raw rung.
pub fn adaptivePrefillWidth(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    kv_len: u64,
    headroom: u64,
    current: u32,
    cap: u32,
    st: *generate_mod.AdaptiveWidthState,
) u32 {
    if (cap == 0 or current == 0) return current;
    const floor: u32 = @min(cap, @as(u32, @intCast(generate_mod.PREFILL_CHUNK_FLOOR)));

    if (prefillChunkCost(config, kv_bits, current, kv_len) > headroom) {
        st.supporting = 0;
        st.ratcheted = true;
        var best: u32 = floor;
        for (PREFILL_CHUNK_LADDER) |rung| {
            const w: u32 = @min(rung, cap);
            if (w >= current) continue;
            if (prefillChunkCost(config, kv_bits, w, kv_len) <= headroom) {
                best = w;
                break;
            }
        }
        return @min(best, current);
    }

    var next: u32 = 0;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const w: u32 = @min(rung, cap);
        if (w > current and (next == 0 or w < next)) next = w;
    }
    if (next == 0 or st.ratcheted) {
        st.supporting = 0;
        return current;
    }
    const asked: u64 = prefillChunkCost(config, kv_bits, next, kv_len) *| 5 / 4;
    if (asked > headroom) {
        st.supporting = 0;
        return current;
    }
    st.supporting +|= 1;
    if (st.supporting < PREFILL_WIDEN_PROBES) return current;
    st.supporting = 0;
    return next;
}

/// The live headroom one prefill chunk has to fit in: the GPU ceiling less what MLX's
/// allocator holds, read once per chunk boundary after its `mlx_clear_cache`. The same
/// expression `prefillAdmissionBill` compares against.
pub fn prefillHeadroomNow(config: *const model_mod.ModelConfig, staged_host: u64) u64 {
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    // The SSD write-through stages chunk blobs as host bytes behind a ~1 GiB permit that
    // `mlx_get_active_memory` does not count; on unified memory they are the same pool. Read at
    // the hook site (inference thread) and passed in.
    return prefillHeadroomFrom(currentGpuMemoryCeiling(config, active_mem), active_mem, staged_host);
}

/// Pure half of `prefillHeadroomNow`.
pub fn prefillHeadroomFrom(ceiling: u64, active: u64, staged_host: u64) u64 {
    return ceiling -| active -| staged_host;
}

/// Does a widen the pre-tick probe chose still fit after the interleave tick allocated for
/// co-tenant slots? Same 1.25 bar, same estimator, same live KV.
pub fn adaptivePrefillWidenStillFits(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    want: u32,
    staged_host: u64,
) bool {
    if (!adaptivePrefillChunkEnabled(config)) return false;
    const headroom = prefillHeadroomNow(config, staged_host);
    const asked: u64 = prefillChunkCost(config, kv_bits, want, pos) *| 5 / 4;
    if (asked <= headroom) return true;
    log.info("[prefill] widen to {d} withdrawn at pos {d} (headroom {d} MB after the decode tick, cost {d} MB)\n", .{
        want,
        pos,
        headroom >> 20,
        asked >> 20,
    });
    return false;
}

/// Impure wrapper the prefill loop reaches through `generate.prefill_chunk_adapt`. Logs one
/// line per change, never per boundary (the bench harness greps it).
pub fn adaptivePrefillWidthNow(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    current: u32,
    cap: u32,
    st: *generate_mod.AdaptiveWidthState,
    staged_host: u64,
) u32 {
    if (!adaptivePrefillChunkEnabled(config)) return current;
    const headroom = prefillHeadroomNow(config, staged_host);
    // `pos` is the live KV this chunk will attend to (see `prefillTransientReserveAtKv`).
    const next = adaptivePrefillWidth(config, kv_bits, pos, headroom, current, cap, st);
    if (next != current) {
        // Quote the number it compared: `prefillChunkCost` at this KV.
        log.info("[prefill] width {d} -> {d} at pos {d} (headroom {d} MB, cost {d} MB)\n", .{
            current,
            next,
            pos,
            headroom >> 20,
            prefillChunkCost(config, kv_bits, next, pos) >> 20,
        });
    }
    return next;
}

pub fn prefillAdmissionBill(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_override: ?transformer_mod.KVQuantConfig, unchunked_prefill: bool, prompt_tokens: ?[]const u32, warm: WarmPrefix) AdmissionBill {
    const heads = config.num_attention_heads;
    if (heads == 0) return .{ .needed = 0, .available = std.math.maxInt(u64) };
    const seq: u64 = @intCast(prompt_len);

    const kv_cfg: transformer_mod.KVQuantConfig = kv_override orelse configuredKvQuantFor(config);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    // Available = the lesser of Metal's static working-set max and what is physically reachable
    // now (#64). Read before the width is chosen: on a per-request arch the width is a function of it.
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const total_limit: u64 = currentGpuMemoryCeiling(config, active_mem);
    const available = if (total_limit > active_mem) total_limit - active_mem else 0;

    // A vision prefill chunks like text since issue #197 (the splice resumes
    // its row index across chunks), so it bills the chunk-bounded envelope —
    // UNLESS the MLX_SERVE_VISION_CHUNKED=0 kill switch restored the
    // whole-prompt forward, in which case `unchunked_prefill` bills the real
    // width. Call sites pass generate_mod.visionPrefillUnchunked(has_vision)
    // so the guard and the prefill loop cannot disagree.
    //
    // Otherwise the width comes from `chooseRequestPrefillChunk`, the same rule the scheduler
    // pins with. An ungated arch (or an explicit `--prefill-chunk`) gets the load-time pin.
    const chosen: u32 = if (unchunked_prefill) 0 else chooseRequestPrefillChunk(
        config,
        seq,
        max_tokens,
        kv_bits,
        available,
        config.pinned_prefill_chunk,
        explicitPrefillChunk(),
        warm,
    );
    const chunk: u64 = if (unchunked_prefill)
        @max(seq, 1)
    else
        @intCast(generate_mod.effectivePrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, prompt_len, config.has_sliding_window, config.isMoe(), config.longCtxGated(), chosen));
    // RAM hot-cache restores rebind MLX array handles by refcount; they do not
    // allocate another copy of the cached buffers. `active_mem` above already
    // includes the resident entry, while `prefillMemoryNeeded` bills the full
    // destination KV capacity that may be allocated when the restored cache
    // grows. Adding the resident entry here again would invent a third copy
    // and reject long warm prompts (and even cache misses) spuriously.
    const needed: u64 = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, chunk, warm);
    // Published number, never the pointer: `hot_prefix_cache` is inference-thread state, freed
    // on every model switch. Stale by at most one entry, which is all a hint has to be.
    const evictable: u64 = if (global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic)
    else
        0;
    // Prompt-aware when the caller has the ids: an entry this prompt could not restore from is
    // genuinely evictable. Still a published snapshot, never the cache pointer.
    const reclaimable: u64 = if (global_scheduler) |sch|
        if (prompt_tokens) |toks|
            scheduler_mod.reclaimableHotCacheBytesFor(sch, toks)
        else
            sch.reclaimable_hot_cache_bytes.load(.monotonic)
    else
        0;
    return .{ .needed = needed, .available = available, .evictable = evictable, .reclaimable = reclaimable, .chunk = chunk };
}

/// Bytes of the resident hot cache this prompt pins: residency less what an eviction can prove
/// it will return. Non-zero means an entry this prompt would restore from is holding memory
/// that neither goes back nor has to be allocated again; the connection thread cannot price
/// how much, so it defers to the inference thread's `WarmPrefix` bill.
pub fn pinnedResidentBytes(bill: AdmissionBill) u64 {
    return bill.evictable -| bill.evictionCredit();
}

/// What the admission guard decided, from one bill so the log line and the arm agree: fits
/// outright, fits once the hot cache gives back what it provably can, or does not fit.
pub const AdmissionVerdict = enum {
    admit,
    evict,
    refuse,

    pub fn name(self: AdmissionVerdict) []const u8 {
        return switch (self) {
            .admit => "admit",
            .evict => "evict",
            .refuse => "refuse",
        };
    }
};

pub fn admissionVerdict(bill: AdmissionBill) AdmissionVerdict {
    if (bill.fits()) return .admit;
    if (bill.fitsAfterEviction()) return .evict;
    return .refuse;
}

/// The generation of `markReadyLocked` the last info-level admission line was emitted for.
var admission_logged_generation = std.atomic.Value(u64).init(std.math.maxInt(u64));

/// Has a model become resident since the last time this returned true? Consuming, so N racing connection threads produce one line.
fn firstRequestAfterLoad() bool {
    const gen = model_registry_mod.load_generation.load(.monotonic);
    const last = admission_logged_generation.load(.monotonic);
    if (gen == last) return false;
    return admission_logged_generation.cmpxchgStrong(last, gen, .monotonic, .monotonic) == null;
}

/// The level one admission line is emitted at: info on the first request after a load and
/// whenever the bill is within 10% of what is available; debug otherwise.
pub fn admissionLogLevel(needed: u64, available: u64, first_after_load: bool) log.Level {
    if (first_after_load) return .info;
    // `needed > 0.9 * available` without floating point or overflow.
    if (needed *| 10 > available *| 9) return .info;
    return .debug;
}

/// One line per request at the admission decision, on admitted requests too. Every field comes
/// off the same `bill` the arms act on. The level check runs first.
fn logAdmissionDecision(bill: AdmissionBill) void {
    // `info` is the loudest level this line is emitted at, and `firstRequestAfterLoad` consumes its token.
    if (!log.enabled(.info)) return;
    const level = admissionLogLevel(bill.needed, bill.available, firstRequestAfterLoad());
    if (!log.enabled(level)) return;
    const mb = 1024 * 1024;
    log.atLevel(level, "[admission] needed={d} MB available={d} MB reclaimable={d} MB width={d} verdict={s}\n", .{
        bill.needed / mb,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.chunk,
        admissionVerdict(bill).name(),
    });
}

/// Inference-thread predicate behind the scheduler's admission hook: does this request fit
/// right now, with live memory re-read? Every input is forwarded, none defaulted (a hardcoded
/// `kv_override = null` priced a `kv_quant: 4` request at fp16 and evicted a cache for nothing).
pub fn prefillFitsNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool) bool {
    return prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
    }).fits();
}

/// {needed, available} of the cold bill, live memory re-read, for the scheduler's sibling ledger.
pub fn prefillBillNumbersNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool) [2]u64 {
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{});
    return .{ bill.needed, bill.available };
}

/// The inference thread's refusal, quoting the numbers it compared; returns {needed, available} for the client.
pub fn logPrefillRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool) [2]u64 {
    // The same warm inputs the probe was refused on, the checkout decision included.
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
    });
    const mb = 1024 * 1024;
    // "Narrowest width tried" is only true where the ladder was walked.
    const width_note: []const u8 = if (perRequestPrefillChunkEnabled(config)) "the narrowest width tried" else "the pinned width";
    // The comparison first (`fitsAfterEviction` spelled out), then the residency as context, with
    // the part eviction cannot take named.
    log.warn("  prompt {d} tokens needs ~{d}MB at prefill chunk {d} ({s}), ~{d}MB available + ~{d}MB the hot cache can give back (~{d}MB resident, ~{d}MB of it pinned by this request's own prefix) — refused before prefill\n", .{
        prompt_len,
        bill.needed / mb,
        bill.chunk,
        width_note,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.evictable / mb,
        pinnedResidentBytes(bill) / mb,
    });
    return .{ bill.needed, bill.available };
}

fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *Conn, prompt_ids: []const u32, max_tokens: u32, config: *const model_mod.ModelConfig, kv_override: ?transformer_mod.KVQuantConfig, lm: *const LoadedModel, unchunked_prefill: bool, media_bytes: u64) !bool {
    const prompt_len: usize = prompt_ids.len;
    if (!mlxMemoryGuardApplies(lm.engineBacked())) return true;
    if (config.num_attention_heads == 0) return true; // unknown architecture, skip check
    // The connection thread has no slot and no cache: it bills cold and defers a warm prompt.
    var bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_override, unchunked_prefill, prompt_ids, .{});
    // The request's media rows are one more tensor the prefill allocates.
    bill.needed += media_bytes;
    logAdmissionDecision(bill);
    const needed = bill.needed;
    const available = bill.available;
    // Evict to admit (#353): when the bill only fails because the cache is holding memory, admit;
    // the inference thread evicts LRU entries until it fits and refuses by name only if it still does not.
    if (needed > available and bill.fitsAfterEviction()) {
        log.info("  prompt {d} tokens needs ~{d}MB, ~{d}MB available + ~{d}MB reclaimable hot cache (of ~{d}MB resident) — admitting, the prefill will evict\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024),
        });
        return true;
    }
    // Warm deferral. The bill above is cold; a non-zero `pinnedResidentBytes` means an entry
    // this prompt would restore from is holding bytes that will neither be allocated nor
    // reclaimed. How many is knowable only on the inference thread, which bills with `WarmPrefix`.
    if (needed > available and pinnedResidentBytes(bill) > 0) {
        log.info("  prompt {d} tokens needs ~{d}MB cold, ~{d}MB available, and ~{d}MB of hot cache is a prefix this prompt would SHARE — deferring to the prefill, which bills the warm request\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), pinnedResidentBytes(bill) / (1024 * 1024),
        });
        return true;
    }
    if (needed > available) if (global_scheduler) |sch| {
        const Rebill = struct {
            sch: *scheduler_mod.Scheduler,
            keep: []const u8,
            bill: *AdmissionBill,
            config: *const model_mod.ModelConfig,
            prompt_ids: []const u32,
            max_tokens: u32,
            kv_override: ?transformer_mod.KVQuantConfig,
            unchunked: bool,
            media_bytes: u64,
            fn fits(r: *@This()) bool {
                return r.bill.fits();
            }
            fn unloadOne(r: *@This()) bool {
                const id = r.sch.idleModelToUnload(r.keep) orelse return false;
                log.info("  prompt {d} tokens does not fit — unloading idle model {s} first\n", .{ r.prompt_ids.len, id });
                r.sch.unloadModelIfIdle(id, 0) catch return false;
                r.bill.* = prefillAdmissionBill(r.config, r.prompt_ids.len, r.max_tokens, r.kv_override, r.unchunked, r.prompt_ids, .{});
                r.bill.needed += r.media_bytes;
                return true;
            }
        };
        var rebill = Rebill{ .sch = sch, .keep = lm.id, .bill = &bill, .config = config, .prompt_ids = prompt_ids, .max_tokens = max_tokens, .kv_override = kv_override, .unchunked = unchunked_prefill, .media_bytes = media_bytes };
        if (fitAfterUnloadingIdle(&rebill, Rebill.fits, Rebill.unloadOne)) return true;
    };
    if (bill.needed > bill.available) {
        const needed_mb = bill.needed / (1024 * 1024);
        const avail_mb = bill.available / (1024 * 1024);
        log.warn("  prompt {d} tokens needs ~{d}MB (KV+working+margin) at prefill chunk {d}, ~{d}MB available + ~{d}MB reclaimable of ~{d}MB resident hot cache — rejecting\n", .{ prompt_len, needed_mb, bill.chunk, avail_mb, bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024) });
        // A refusal quotes the numbers it compared, hot cache included; everything the cache holds
        // is evictable here by construction (the withheld case took the deferral arm).
        const msg = try memoryRefusalMessage(allocator, prompt_len, needed_mb, avail_mb, bill);
        defer allocator.free(msg);
        countRejected();
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return false;
    }
    return true;
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Get the max buffer allocation limit (~75% of system unified memory).
/// `hw.memsize` on Darwin; on Linux there is no unified-memory sysctl, so the
/// same 75%-of-physical-RAM heuristic runs against /proc/meminfo.
fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    if (comptime builtin.os.tag.isDarwin()) {
        var len: usize = @sizeOf(u64);
        _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    } else {
        mem = metrics.getTotalMemBytes();
    }
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

/// Clamp max_tokens so prompt + completion doesn't exceed context length.
/// Clamp a request's `max_tokens` so `prompt + generation` stays inside
/// `effective_ctx`. `effective_ctx == 0` means "unknown" and imposes no limit.
///
/// Takes the context explicitly rather than reading `server_config.max_context_size`:
/// that global is only set by `--ctx-size`, so under AUTO context this function
/// used to return `max_tokens` untouched — the server never trimmed a client's
/// budget, never emitted the "budget squeezed" warning, and generation could run
/// past the memory-safe window. Callers pass `getEffectiveContextLength(config)`,
/// which is the same number the prompt-length guard and `/v1/models` report.
fn clampMaxTokens(max_tokens: u32, prompt_len: usize, effective_ctx: u32) u32 {
    if (effective_ctx == 0) return max_tokens;
    const prompt: u32 = @intCast(@min(prompt_len, effective_ctx));
    if (prompt >= effective_ctx) return 1; // at least 1 token
    const remaining = effective_ctx - prompt;
    if (maxTokensBudgetSqueezed(max_tokens, remaining)) {
        log.warn("  generation budget squeezed: {d}/{d} tokens remaining (prompt={d}, ctx={d}) — tool call arguments may be truncated\n", .{ remaining, max_tokens, prompt, effective_ctx });
    } else if (max_tokens == AUTO_MAX_TOKENS_SENTINEL and autoBudgetWindowTight(remaining, effective_ctx)) {
        // The client named no budget, so this is context, not a deviation: debug.
        log.debug("  generation budget squeezed: {d} of the window remain (auto budget; prompt={d}, ctx={d})\n", .{ remaining, prompt, effective_ctx });
    }
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx={d}, prompt={d})\n", .{ max_tokens, remaining, effective_ctx, prompt });
        return remaining;
    }
    return max_tokens;
}

/// Render an optional model-author sampling recommendation (from the model's
/// generation_config.json) as a JSON scalar: the number when present, the
/// literal `null` when the model ships no value. Caller owns the slice.
/// Used for the `gen_temperature`/`gen_top_p`/`gen_top_k` meta fields the
/// Swift Settings UI reads to show "model recommends" guidance pills.
fn optSamplingRecJson(allocator: std.mem.Allocator, comptime T: type, v: ?T) ![]u8 {
    if (v) |val| return std.fmt.allocPrint(allocator, "{d}", .{val});
    return allocator.dupe(u8, "null");
}

/// Render the JSON metadata fragment for one entry. For `.ready` entries
/// Capability flags for a READY registry entry, kept as plain booleans so the
/// JSON assembly below is hermetically testable without a LoadedModel. Every
/// engine slot on LoadedModel must have a flag here — a missing arm renders a
/// ready model with an empty capabilities list.
const ReadyCaps = struct {
    has_chat: bool = false,
    has_reasoning: bool = false,
    /// A decoder with a pooling contract: `config.hasEmbeddingCapability()`.
    has_embedding: bool = false,
};

/// Chat capability for a READY entry. Template presence is NOT the gate for
/// an engine-backed model, which serves chat without one.
fn readyHasChat(chat_template_len: usize, has_embedded_lm: bool) bool {
    return chat_template_len > 0 or has_embedded_lm;
}

/// `capabilities` JSON array for a ready model. Caller deinits.
fn readyCapsJson(allocator: std.mem.Allocator, c: ReadyCaps) !std.ArrayList(u8) {
    var caps = std.ArrayList(u8).empty;
    errdefer caps.deinit(allocator);
    try caps.append(allocator, '[');
    var n_caps: usize = 0;
    const append_cap = struct {
        fn call(a: std.mem.Allocator, b: *std.ArrayList(u8), n: *usize, name: []const u8) !void {
            if (n.* > 0) try b.append(a, ',');
            try b.append(a, '"');
            try b.appendSlice(a, name);
            try b.append(a, '"');
            n.* += 1;
        }
    }.call;
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "chat");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "tool_use");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "streaming");
    if (c.has_reasoning) try append_cap(allocator, &caps, &n_caps, "reasoning");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "json_schema");
    if (c.has_embedding) try append_cap(allocator, &caps, &n_caps, "embeddings");
    try caps.append(allocator, ']');
    return caps;
}

/// Facts about a request's target model that decide whether a TEXT-
/// GENERATION route (/v1/chat/completions) may serve it. Extracted
/// from a LoadedModel by `textGenTargetOf`; kept as plain facts so the
/// decision is hermetically testable.
const TextGenTarget = struct {
    arch_hint: []const u8 = "",
    /// A text-capable LM is resident (transformer / engine-backed) -
    /// or the entry isn't loaded yet, in which case stubs default to
    /// "assume text until the arch hint or a load says otherwise".
    has_text_lm: bool = true,
};

/// This engine serves text only: an image, video or audio part in ANY message
/// of a request (every turn, tool results included) is refused with this one
/// message. The parse sites record the part as a non-null empty slice instead
/// of decoding it, so this check tests `!= null` and needs no decoder.
pub const TEXT_ONLY_MEDIA_REJECT = "This engine serves text only: image, video and audio content is not supported";

/// Whether a `messages` entry (any API surface) carries media in its `content`, whatever its
/// role, type or content shape. Parse sites call it before any skip for a missing or invalid role.
fn messageEntryHasMedia(entry: std.json.Value) bool {
    if (entry != .object) return false;
    const cv = entry.object.get("content") orelse return false;
    return chat_mod.contentValueHasMedia(cv);
}

/// Which media kinds a chat `content` value carries (an array of parts, or one part without
/// the array). Pure, so the chat parse site and its tests share it.
const MediaMarks = struct { image: bool = false, video: bool = false, audio: bool = false };

fn contentMediaMarks(content: std.json.Value) MediaMarks {
    var marks: MediaMarks = .{};
    switch (content) {
        .array => |arr| for (arr.items) |part| markMediaPart(&marks, part),
        .object => markMediaPart(&marks, content),
        else => {},
    }
    return marks;
}

fn markMediaPart(marks: *MediaMarks, part: std.json.Value) void {
    const kind = chat_mod.mediaKindOfPartValue(part) orelse return;
    switch (kind) {
        .image => marks.image = true,
        .video => marks.video = true,
        .audio => marks.audio = true,
    }
}

fn mediaRejectReason(messages: []const chat_mod.Message) ?[]const u8 {
    for (messages) |m| {
        if (m.images != null or m.videos != null or m.audio != null) return TEXT_ONLY_MEDIA_REJECT;
    }
    return null;
}

test "mediaRejectReason: image, video and audio parts get the text-only message, text passes" {
    const t = std.testing;
    const text = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    try t.expect(mediaRejectReason(&text) == null);
    const img = [_]chat_mod.Message{ .{ .role = "user", .content = "hi" }, .{ .role = "user", .content = "look", .images = &[_]chat_mod.ImageData{} } };
    try t.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, mediaRejectReason(&img).?);
    const vid = [_]chat_mod.Message{.{ .role = "user", .content = "watch", .videos = &[_]chat_mod.VideoData{} }};
    try t.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, mediaRejectReason(&vid).?);
    const aud = [_]chat_mod.Message{.{ .role = "user", .content = "listen", .audio = &[_]chat_mod.AudioData{} }};
    try t.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, mediaRejectReason(&aud).?);
    try t.expect(std.mem.indexOf(u8, TEXT_ONLY_MEDIA_REJECT, "text") != null);
    // A tool result's image is refused too.
    const tool = [_]chat_mod.Message{.{ .role = "tool", .content = "Read image file", .images = &[_]chat_mod.ImageData{} }};
    try t.expect(mediaRejectReason(&tool) != null);
}

test "mediaRejectReason: an image in an EARLIER turn is refused" {
    const t = std.testing;
    const history = [_]chat_mod.Message{
        .{ .role = "user", .content = "look", .images = &[_]chat_mod.ImageData{} },
        .{ .role = "assistant", .content = "a house" },
        .{ .role = "user", .content = "thanks" },
    };
    try t.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, mediaRejectReason(&history).?);
    // A continuation's trailing assistant prefix does not hide it either.
    const prefill = [_]chat_mod.Message{ history[0], .{ .role = "assistant", .content = "It shows" } };
    try t.expect(mediaRejectReason(&prefill) != null);
}

/// Request-level media check over a `messages` value: the text-only 400 when any entry's content
/// (whatever its role, or with none, in any shape, nested tool_result content included) carries
/// media; null for a missing or non-array value, which the handler's own checks then name.
fn messagesValueMediaReject(messages_val: ?std.json.Value) ?[]const u8 {
    const mv = messages_val orelse return null;
    if (mv != .array) return null;
    for (mv.array.items) |entry| {
        if (messageEntryHasMedia(entry)) return TEXT_ONLY_MEDIA_REJECT;
    }
    return null;
}

/// The media check `/v1/chat/completions` runs right after the body parses, before `n`, the
/// `messages` checks or any other validation can answer.
fn chatRequestMediaReject(root: std.json.ObjectMap) ?[]const u8 {
    return messagesValueMediaReject(root.get("messages"));
}

/// The media check `/v1/embeddings` runs right after the body parses, before the engine,
/// `dimensions` and `input`/`messages` checks: media in any `messages` entry, or a media part
/// in `input` (which otherwise skips non-string items).
fn embeddingsRequestMediaReject(body_val: std.json.Value) ?[]const u8 {
    if (body_val != .object) return null;
    const root = body_val.object;
    if (root.get("messages")) |mv| {
        if (mv == .array) {
            if (embeddingMessagesMediaReject(mv.array.items)) |why| return why;
        }
    }
    if (root.get("input")) |iv| {
        if (chat_mod.contentValueHasMedia(iv)) return TEXT_ONLY_MEDIA_REJECT;
    }
    return null;
}

/// Reason a text-generation route must reject this model with a 400, or
/// null when it can serve text. A model with no language model resident (it
/// would dereference a null transformer in prefill) is rejected here before it
/// can reach the scheduler.
fn textGenRejectReason(t: TextGenTarget) ?[]const u8 {
    if (!t.has_text_lm) return "This model cannot serve text generation (no language model resident).";
    return null;
}

fn textGenTargetOf(lm: *LoadedModel) TextGenTarget {
    return .{
        .arch_hint = lm.arch_hint,
        .has_text_lm = lm.state != .ready or lm.transformer != null or lm.engineBacked(),
    };
}

/// True for a read-only route that reports on a model without using it.
/// `GET /props` is the only one: every other status endpoint either answers
/// before `ensureLoaded` (`/health`, `/metrics`, `/v1/models` return above it) or is a real
/// request. `/props` alone falls through to the shared
/// `ensureLoaded`/release pair, so without this a poll stamps `last_used_ms`
/// and a client polling faster than `--idle-evict-secs` pins every model.
fn isStatusRoute(method: []const u8, path: []const u8) bool {
    return std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props");
}

/// True for the routes textGenRejectReason protects — used for the
/// pre-load peek so naming a media model in a chat request doesn't
/// cold-load gigabytes just to earn its 400.
fn isTextGenRoute(method: []const u8, path: []const u8) bool {
    return std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions");
}

/// pulls full capabilities/dimensions off the resident config/chat_config;
/// for non-ready entries renders a lightweight stub with state +
/// bytes_on_disk only. Returns an allocator-owned string; caller frees.
/// Called from `handleModels` per registry entry. Registry mutex must be
/// held by the caller (entry fields are read directly).
fn renderModelEntry(
    allocator: std.mem.Allocator,
    io: std.Io,
    entry: *LoadedModel,
) ![]u8 {
    var quant_buf: [16]u8 = undefined;
    if (entry.state == .ready and entry.config != null and entry.chat_config != null) {
        const config = entry.config.?;
        const chat_config = entry.chat_config.?;
        const ctx_len = getEffectiveContextLength(config);
        const ctx_str = if (ctx_len > 0)
            try std.fmt.allocPrint(allocator, "{d}", .{ctx_len})
        else
            try std.fmt.allocPrint(allocator, "null", .{});
        defer allocator.free(ctx_str);

        const has_chat = readyHasChat(
            chat_config.chat_template.len,
            entry.engineBacked(),
        );
        var caps = try readyCapsJson(allocator, .{
            .has_chat = has_chat,
            .has_reasoning = has_chat and model_discovery.templateSupportsThinking(chat_config.chat_template),
            .has_embedding = config.hasEmbeddingCapability(),
        });
        defer caps.deinit(allocator);

        // This engine serves text only: the one input modality.
        const mods = "[\"text\"]";

        // Loaded and unloaded rows report the same `architecture` — the
        // registry's arch hint (the pack's real model_type), never the stub
        // config's placeholder.
        const arch_label: []const u8 = if (entry.arch_hint.len > 0) entry.arch_hint else config.model_type;
        const model_id_json = try jsonEscape(allocator, if (entry.id.len > 0) entry.id else config.model_type);
        defer allocator.free(model_id_json);
        const drafter_loaded = entry.hasDrafter();
        const drafter_path_json = if (drafter_loaded)
            try jsonEscape(allocator, entry.drafter_path)
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(drafter_path_json);
        const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
            try std.fmt.allocPrint(allocator, "{d}", .{b})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(bytes_on_disk_str);

        // Model-author sampling recommendations from generation_config.json,
        // surfaced so the Swift Settings UI can show "model recommends" pills
        // next to the per-request sampling sliders. `null` when the model
        // ships no value for that field.
        const gen_temp_str = try optSamplingRecJson(allocator, f32, config.gen_temperature);
        defer allocator.free(gen_temp_str);
        const gen_top_p_str = try optSamplingRecJson(allocator, f32, config.gen_top_p);
        defer allocator.free(gen_top_p_str);
        const gen_top_k_str = try optSamplingRecJson(allocator, u32, config.gen_top_k);
        defer allocator.free(gen_top_k_str);

        // Effective embedding input ceiling (issue #117): surfaced so clients
        // can discover the limit instead of learning it from a 400. null for
        // models with no embedding capability; 0 = embedding-capable, unbounded.
        const embed_limit_str = if (config.hasEmbeddingCapability())
            try std.fmt.allocPrint(allocator, "{d}", .{embedEffectiveLimit(embedding_max_length, config.max_position_embeddings)})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(embed_limit_str);

        return std.fmt.allocPrint(allocator,
            \\{{"id":{s},"object":"model","created":{d},"owned_by":"lesto","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"context_length":{s},"max_model_len":{s},"batched_decode":{s},"capabilities":{s},"input_modalities":{s},"meta":{{"architecture":"{s}","engine":"{s}","vocab_size":{d},"hidden_size":{d},"num_layers":{d},"quantization":{s},"context_length":{s},"model_max_tokens":{d},"embedding_max_length":{s},"is_moe":{s},"drafter_loaded":{s},"drafter_path":{s},"spec_exact":{s},"kv_quant":"{s}","gen_temperature":{s},"gen_top_p":{s},"gen_top_k":{s}}}}}
        , .{
            model_id_json,
            nowSecs(io),
            entry.bytes_resident,
            bytes_on_disk_str,
            // Top-level twins of meta.context_length: openai-models-list
            // discovery clients (oh-my-pi, vLLM-shaped readers) look for
            // row-level max_model_len/context_length and substitute their own
            // defaults when absent (issue #188).
            ctx_str,
            ctx_str,
            if (batchVerdictFor(entry) == .ok) "true" else "false",
            caps.items,
            mods,
            arch_label,
            "mlx",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            quantizationJson(&quant_buf, config.quant_bits),
            ctx_str,
            config.max_position_embeddings,
            embed_limit_str,
            if (config.isMoe()) "true" else "false",
            if (drafter_loaded) "true" else "false",
            drafter_path_json,
            if (config.rowExactDecode()) "true" else "false",
            configuredKvQuantFor(config).wireName(),
            gen_temp_str,
            gen_top_p_str,
            gen_top_k_str,
        });
    }

    // Non-ready entry: stub form with state + bytes_on_disk.
    const state_str = switch (entry.state) {
        .unloaded => "unloaded",
        .loading => "loading",
        .ready => "ready",
        .error_state => "error",
        .evicting => "evicting",
    };
    const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const err_part: []const u8 = if (entry.error_name) |name| blk: {
        // Inline escape to avoid double allocation; the names we emit
        // never contain quotes/backslashes (they're @errorName output).
        break :blk try std.fmt.allocPrint(
            allocator,
            ",\"error\":\"{s}\"",
            .{name},
        );
    } else &[_]u8{};
    defer if (err_part.len > 0) allocator.free(err_part);
    // Stub metadata sourced from config.json (+ chat-template presence) WITHOUT
    // faulting in weights: dims, context window, MoE-ness, and capabilities.
    // `/v1/models` isn't hot, so reading a small JSON per unloaded entry is fine.
    const sm = model_discovery.readStubMeta(io, allocator, entry.path);

    // Capabilities. Chat models get chat/tool_use/streaming/json_schema (gated
    // on chat-template presence, the same rule the loaded path uses). Reasoning
    // stays load-gated (it needs the live template), so an unloaded stub may
    // under-report it.
    // Decoder embedding stubs: StubMeta's pooling signal (config key / sidecar).
    const stub_has_embedding = sm.found and sm.has_embedding;
    const caps_part: []const u8 = blk: {
        if (!sm.found or !(sm.has_chat or stub_has_embedding)) break :blk try allocator.dupe(u8, "");
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(allocator);
        try b.appendSlice(allocator, ",\"capabilities\":[");
        var n: usize = 0;
        const add = struct {
            fn call(a: std.mem.Allocator, list: *std.ArrayList(u8), cnt: *usize, name: []const u8) !void {
                if (cnt.* > 0) try list.append(a, ',');
                try list.append(a, '"');
                try list.appendSlice(a, name);
                try list.append(a, '"');
                cnt.* += 1;
            }
        }.call;
        if (sm.has_chat) {
            try add(allocator, &b, &n, "chat");
            try add(allocator, &b, &n, "tool_use");
            try add(allocator, &b, &n, "streaming");
            try add(allocator, &b, &n, "json_schema");
        }
        if (stub_has_embedding) try add(allocator, &b, &n, "embeddings");
        try b.append(allocator, ']');
        break :blk try b.toOwnedSlice(allocator);
    };
    defer allocator.free(caps_part);

    const mods_part: []const u8 = if (sm.found and sm.has_chat)
        ",\"input_modalities\":[\"text\"]"
    else
        "";

    const arch_part: []const u8 = if (entry.arch_hint.len > 0) blk: {
        // arch_hint comes from config.json's model_type via discovery — the
        // supported-type allowlist guarantees no JSON metacharacters.
        break :blk try std.fmt.allocPrint(allocator, "\"architecture\":\"{s}\",", .{entry.arch_hint});
    } else &[_]u8{};
    defer if (arch_part.len > 0) allocator.free(arch_part);

    // Unloaded entries have no engine attached yet, so they report "mlx".
    const engine_part = try std.fmt.allocPrint(allocator, "\"engine\":\"{s}\",", .{
        "mlx",
    });
    defer allocator.free(engine_part);

    // Top-level twins of meta.context_length (issue #188): openai-models-list
    // discovery clients read row-level max_model_len/context_length. Unloaded
    // stubs advertise the architectural max — the same number meta carries here.
    const top_ctx_part: []const u8 = if (sm.found) blk: {
        break :blk try std.fmt.allocPrint(allocator, ",\"context_length\":{d},\"max_model_len\":{d}", .{
            sm.max_position_embeddings,
            sm.max_position_embeddings,
        });
    } else &[_]u8{};
    defer if (top_ctx_part.len > 0) allocator.free(top_ctx_part);

    // Dimensions/context/quant/MoE — emitted only when config.json was readable.
    const dims_part: []const u8 = if (sm.found) blk: {
        break :blk try std.fmt.allocPrint(allocator, "\"vocab_size\":{d},\"hidden_size\":{d},\"num_layers\":{d},\"quantization\":{s},\"context_length\":{d},\"model_max_tokens\":{d},\"is_moe\":{s},", .{
            sm.vocab_size,
            sm.hidden_size,
            sm.num_hidden_layers,
            quantizationJson(&quant_buf, sm.quant_bits),
            sm.max_position_embeddings,
            sm.max_position_embeddings,
            if (sm.is_moe) "true" else "false",
        });
    } else &[_]u8{};
    defer if (dims_part.len > 0) allocator.free(dims_part);

    const id_json = try jsonEscape(allocator, entry.id);
    defer allocator.free(id_json);
    return std.fmt.allocPrint(allocator,
        \\{{"id":{s},"object":"model","created":0,"owned_by":"lesto","loaded":false,"state":"{s}","bytes_resident":0,"bytes_on_disk":{s}{s}{s}{s}{s},"meta":{{{s}{s}{s}"bytes_on_disk":{s}}}}}
    , .{ id_json, state_str, bytes_on_disk_str, err_part, top_ctx_part, caps_part, mods_part, arch_part, engine_part, dims_part, bytes_on_disk_str });
}

test "renderModelEntry: a Windows path id stays valid JSON" {
    const a = std.testing.allocator;
    var e = model_registry_mod.LoadedModel{
        .allocator = a,
        .id = "C:\\Users\\d\\.mlx-serve\\models\\org\\m",
        .path = "C:\\Users\\d\\.mlx-serve\\models\\org\\m",
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
    const row = try renderModelEntry(a, std.testing.io, &e);
    defer a.free(row);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, row, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(e.id, parsed.value.object.get("id").?.string);
}

fn handleModels(
    allocator: std.mem.Allocator,
    stream: *Conn,
) !void {
    const body = renderModelsBody(allocator, stream.io) catch |err| switch (err) {
        error.RegistryNotReady => return sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503),
        else => return err,
    };
    defer allocator.free(body);
    try sendModelsResponse(stream, body);
}

/// The `/v1/models` body.
fn renderModelsBody(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    // Plan 05 Phase E: emit every registry entry (loaded + unloaded), not
    // just the default model + flat discovery list. Default model is sorted
    // first so single-model clients reading `data[0]` continue to work.
    // The mutex is held for the whole render so eviction can't fire under
    // us; each rendered entry is at most a few hundred bytes. This handler
    // does NOT route through `scheduler.ensureLoaded` — listing metadata
    // shouldn't trigger a cold load of the default model.
    const registry = global_registry orelse return error.RegistryNotReady;

    var entries_buf = std.ArrayList(u8).empty;
    defer entries_buf.deinit(allocator);

    // Collect entries while holding the mutex so iteration is safe.
    // Sort: default first, then by last_used_ns desc.
    var ordered = std.ArrayList(*LoadedModel).empty;
    defer ordered.deinit(allocator);
    var alias_buf: [std.fs.max_path_bytes]u8 = undefined;
    // Any re-read happens here, not under the registry lock below.
    g_model_aliases.refreshNow(io);
    {
        registry.mutex.lockUncancelable(io);
        defer registry.mutex.unlock(io);
        try ordered.ensureTotalCapacity(allocator, registry.entries.count());
        var it = registry.entries.valueIterator();
        while (it.next()) |entry_ptr| ordered.appendAssumeCapacity(entry_ptr.*);
        const default_id = registry.default_id;
        const Cmp = struct {
            fn lt(ctx: []const u8, a: *LoadedModel, b: *LoadedModel) bool {
                const a_def = std.mem.eql(u8, a.id, ctx);
                const b_def = std.mem.eql(u8, b.id, ctx);
                if (a_def != b_def) return a_def;
                return a.last_used_ns > b.last_used_ns;
            }
        };
        std.sort.pdq(*LoadedModel, ordered.items, default_id, Cmp.lt);

        for (ordered.items) |entry| {
            if (entries_buf.items.len > 0) try entries_buf.append(allocator, ',');
            const json = try renderModelEntry(allocator, io, entry);
            defer allocator.free(json);
            try appendAliasedRow(allocator, &entries_buf, json, g_model_aliases.aliasForPath(io, entry.path, &alias_buf));
        }
    }

    // What this machine is.
    var extras = std.ArrayList(u8).empty;
    defer extras.deinit(allocator);
    const machine = try machineJson(allocator);
    defer allocator.free(machine);
    try extras.appendSlice(allocator, machine);
    return std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{s}]{s}}}
    , .{ entries_buf.items, extras.items });
}

/// `/v1/models`' "machine" (behind the API key, unlike `/health`): the OS, the device model
/// id the app draws its icon from, and its memory.
fn machineJson(allocator: std.mem.Allocator) ![]u8 {
    var model_buf: [64]u8 = undefined;
    const status_mod = @import("status.zig");
    return std.fmt.allocPrint(allocator, ",\"machine\":{{\"os\":\"{s}\",\"model\":{f},\"free_mib\":{d},\"total_mib\":{d}}}", .{
        @tagName(builtin.os.tag), std.json.fmt(machineModel(&model_buf), .{}),
        status_mod.getAvailableMemBytes() >> 20, status_mod.getTotalMemBytes() >> 20,
    });
}

/// The device model id ("Mac16,5"); empty off macOS.
fn machineModel(buf: *[64]u8) []const u8 {
    const key = switch (comptime builtin.os.tag) {
        .macos => "hw.model",
        else => return "",
    };
    var len: usize = buf.len;
    if (std.c.sysctlbyname(key, buf, &len, null, 0) != 0) return "";
    return std.mem.sliceTo(buf[0..len], 0);
}

test "machineJson names the OS and the device model, as valid JSON" {
    const a = std.testing.allocator;
    const field = try machineJson(a);
    defer a.free(field);
    const j = try std.fmt.allocPrint(a, "{{\"data\":[]{s}}}", .{field});
    defer a.free(j);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, j, .{});
    defer parsed.deinit();
    const m = parsed.value.object.get("machine").?.object;
    try std.testing.expectEqualStrings(@tagName(builtin.os.tag), m.get("os").?.string);
    if (builtin.os.tag == .macos) try std.testing.expect(m.get("model").?.string.len > 0);
    try std.testing.expect(m.get("total_mib").?.integer >= m.get("free_mib").?.integer);
}

/// A `/v1/models` row with its Model Settings `alias` as the first field; `id`
/// stays the canonical registry id.
fn appendAliasedRow(allocator: std.mem.Allocator, out: *std.ArrayList(u8), row: []const u8, alias: ?[]const u8) !void {
    const a = alias orelse return out.appendSlice(allocator, row);
    const esc = try jsonEscape(allocator, a);
    defer allocator.free(esc);
    try out.print(allocator, "{{\"alias\":{s},{s}", .{ esc, row[1..] });
}

fn sendModelsResponse(stream: *Conn, body: []const u8) !void {
    logHttpResponse("200 OK", "application/json", body);
    return sendResponseFramed(stream, "200 OK", "application/json", body);
}

/// `GET /v1/load-progress`: every model mid-load and what its load is doing (`load_progress.zig`).
/// Cheap enough to poll while a load runs: it reads no configs.
fn renderLoadProgressBody(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    return loadProgressBody(allocator, io, global_registry orelse return error.RegistryNotReady);
}

fn loadProgressBody(allocator: std.mem.Allocator, io: std.Io, registry: *ModelRegistry) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const now = io_util.nowMsMonotonic(io);
    const footprint = metrics.getAppMemFootprintMb();
    try buf.appendSlice(allocator, "{\"loading\":[");
    registry.mutex.lockUncancelable(io);
    defer registry.mutex.unlock(io);
    var it = registry.entries.valueIterator();
    var first = true;
    while (it.next()) |e| {
        const entry = e.*;
        if (entry.state != .loading) continue;
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "{\"id\":");
        try chat_mod.appendJsonString(allocator, &buf, entry.id);
        try buf.appendSlice(allocator, ",\"path\":");
        try chat_mod.appendJsonString(allocator, &buf, entry.path);
        try buf.appendSlice(allocator, ",\"progress\":");
        var jb: [256]u8 = undefined;
        var w: std.Io.Writer = .fixed(&jb);
        try entry.load_progress.snapshot(now, footprint).writeJson(&w);
        try buf.appendSlice(allocator, w.buffered());
        try buf.append(allocator, '}');
    }
    try buf.appendSlice(allocator, "]}");
    return buf.toOwnedSlice(allocator);
}

/// `POST /v1/models/rescan`: re-walk the boot roots and register stubs for
/// new dirs, clearing failed loads (`ModelRegistry.rescan`). Answers `{"added":N}`.
fn handleModelsRescan(allocator: std.mem.Allocator, stream: *Conn) !void {
    const registry = global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };
    const added = registry.rescan() catch |err| {
        log.warn("/v1/models/rescan failed: {s}\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "Model rescan failed", 500);
        return;
    };
    if (added > 0) log.info("[registry] rescan added {d} model(s)\n", .{added});
    const body = try std.fmt.allocPrint(allocator, "{{\"added\":{d}}}", .{added});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Pure body-builder for `GET /props`. Split out from `handleProps` so it
/// can be unit-tested without standing up a real `LoadedModel`. Caller owns
/// the returned slice.
///
/// Note: the `chat_template` field that upstream llama-server exposes here
/// is intentionally omitted. Our Swift app never read it, and stock chat
/// templates are 10–100 KB of Jinja — that turned every /props poll into
/// wasted bandwidth. Capabilities still come from `/v1/models` per entry;
/// clients that genuinely need the template can still get it from the
/// model's `config.json` / `chat_template.jinja` on disk, or by rendering
/// through `/v1/chat/completions` server-side.
fn renderPropsBody(
    allocator: std.mem.Allocator,
    config: *const model_mod.ModelConfig,
    ctx_str: []const u8,
    active_mem: usize,
    peak_mem: usize,
    available_mem: u64,
    safe_ctx: u32,
    cache_mem: usize,
    /// Registry residency of every ready model, and the KV of the hot prefix cache + live slots.
    weights_mem: u64,
    kv_cache_mem: u64,
    /// Leading-comma JSON fragments spliced before the root close (the
    /// n-gram warm object). Concatenated by the handler.
    extra_json: []const u8,
) ![]u8 {
    // `available_bytes` is free SYSTEM RAM, computed with the SAME formula the
    // model-load pre-flight uses (`metrics.getAvailableMemBytes`), so the tray's
    // "Free RAM" line can never drift from the number that gates a load. Distinct
    // axis from `active_bytes` (the MLX GPU-allocator footprint).
    //
    // `cache_bytes` is MLX's reclaimable buffer pool — memory the process HOLDS
    // but is not using. It is a third axis again, and its absence is why #110
    // was invisible: the panel read 19.6 GB of `active_bytes` while the process
    // sat at 81.4 GB, and nothing we served named the other 61.
    return std.fmt.allocPrint(allocator,
        \\{{"default_generation_settings":{{"model":"{s}","n_ctx":{s}}},"total_slots":1,"model_info":{{"vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"quantization_bits":{d},"quantization_group_size":{d},"max_position_embeddings":{d}}},"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":{d},"cache_bytes":{d},"weights_bytes":{d},"kv_cache_bytes":{d}}}{s}}}
    , .{
        config.model_type,              ctx_str,
        config.vocab_size,              config.hidden_size,
        config.num_hidden_layers,       config.num_attention_heads,
        config.num_key_value_heads,     config.head_dim,
        config.quant_bits,              config.quant_group_size,
        config.max_position_embeddings, active_mem,
        peak_mem,                       available_mem,
        safe_ctx,                       cache_mem,
        weights_mem,                    kv_cache_mem,
        extra_json,
    });
}

/// The model-level half of `Scheduler.batchVerdict`: does this loaded model
/// batch decode at all? Per-slot arms (spec, grammar, logprobs) come later.
fn batchVerdictFor(entry: *const LoadedModel) scheduler_mod.BatchVerdict {
    if (entry.engineBacked()) return .embedded_engine;
    const cfg = entry.config orelse return .arch;
    return if (scheduler_mod.configBatchesDecode(cfg)) .ok else .arch;
}

/// The /props "batching" object: whether the loaded model rides the batched
/// decode kernel, and why not when it does not.
fn batchingPropsJson(allocator: std.mem.Allocator, why: scheduler_mod.BatchVerdict) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"batching\":{{\"supported\":{s},\"reason\":\"{s}\",\"max_group\":{d}}}", .{
        if (why == .ok) "true" else "false",
        @tagName(why),
        scheduler_mod.MAX_BATCH_GROUP,
    });
}

/// Effective per-model serving settings, resolved the way a request that sends no overrides sees them.
const PropsSettings = struct {
    engine: []const u8,
    kv_quant: []const u8,
    kv_attn_mode: KvAttnMode,
    decode_attn_quant: bool,
    prefill_chunk: usize,
    drafter: []const u8,
    pld: PldDefaults,
    max_concurrent: u32,
    prefix_cache_mem_bytes: u64,
    prefix_cache_disk_bytes: u64,
    prefix_cache_ram_enabled: bool = true,
    /// `--prefill-decode-share`: decode's target wall-time fraction during another slot's prefill.
    prefill_decode_share: f32 = 0,
};

fn mlxPropsSettings(lm: *LoadedModel) PropsSettings {
    const config = lm.config.?;
    const kv = configuredKvQuantFor(config);
    return .{
        .engine = "mlx",
        .kv_quant = if (kv.isQuant()) (if (kv.bits == 4) "4" else "8") else "off",
        .kv_attn_mode = server_config.kv_attn_mode,
        .decode_attn_quant = transformer_mod.decodeAttnQuantEnabled() and (if (lm.transformer) |x| x.dense_attn_proj else false),
        .prefill_chunk = generate_mod.prefill_chunk_override,
        .drafter = if (lm.dflash != null) "dflash" else if (lm.drafter != null) "assistant" else "none",
        .pld = .{ .enable = server_config.default_enable_pld, .draft_len = server_config.default_pld_draft_len, .key_len = server_config.default_pld_key_len },
        .max_concurrent = max_concurrent,
        .prefix_cache_mem_bytes = resolvedPrefixCacheMem(),
        .prefix_cache_disk_bytes = prefix_cache_disk_bytes,
        .prefix_cache_ram_enabled = prefix_cache_capacity > 0 and prefix_cache_ram_enabled,
        .prefill_decode_share = scheduler_mod.prefillDecodeShare(),
    };
}

/// The /props "settings" object: leading-comma fragment spliced before the props root close.
fn settingsPropsJson(allocator: std.mem.Allocator, st: PropsSettings) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"settings\":{{\"version\":\"{s}\",\"engine\":\"{s}\",\"kv_quant\":\"{s}\",\"kv_attn_mode\":\"{s}\",\"decode_attn_quant\":{},\"prefill_chunk\":{d},\"drafter\":\"{s}\",\"pld\":{{\"default_on\":{},\"draft_len\":{d},\"key_len\":{d}}},\"max_concurrent\":{d},\"prefill_decode_share\":{d:.2},\"prefix_cache\":{{\"ram_enabled\":{},\"mem_bytes\":{d},\"disk_bytes\":{d}}}}}", .{
        build_options.version,                      st.engine,
        st.kv_quant,                                @tagName(st.kv_attn_mode),
        st.decode_attn_quant,                       st.prefill_chunk,
        st.drafter,                                 st.pld.enable,
        st.pld.draft_len,                           st.pld.key_len,
        st.max_concurrent,                          st.prefill_decode_share,
        st.prefix_cache_ram_enabled,                st.prefix_cache_mem_bytes,
        st.prefix_cache_disk_bytes,
    });
}

fn handleProps(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel) !void {
    const config = lm.config.?;
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "0", .{});
    defer allocator.free(ctx_str);

    const safe_ctx = computeMaxSafeContext(config);

    // Query memory usage from mlx's allocator counters.
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    // MLX's reclaimable buffer pool.
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);

    // Free system RAM — same calc as the model-load pre-flight, so the app's
    // "Free RAM" line stays in lockstep with what gates a load.
    const available_mem = metrics.getAvailableMemBytes();

    const batching_json = try batchingPropsJson(allocator, batchVerdictFor(lm));
    defer allocator.free(batching_json);
    const settings_json = try settingsPropsJson(allocator, mlxPropsSettings(lm));
    defer allocator.free(settings_json);
    const extra_json = try std.fmt.allocPrint(allocator, "{s}{s},\"registry_generation\":{d}", .{ batching_json, settings_json, registryGeneration() });
    defer allocator.free(extra_json);

    const kv_cache_mem: u64 = if (global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic) + sch.resident_live_kv_bytes.load(.monotonic)
    else
        0;
    const body = try renderPropsBody(allocator, config, ctx_str, active_mem, peak_mem, available_mem, safe_ctx, cache_mem, residentWeightsBytes(stream.io), kv_cache_mem, extra_json);
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// `ModelRegistry.generation`: moves with every model added and every load state change.
/// Moves whenever `/v1/models` would list something else.
fn registryGeneration() u64 {
    return if (global_registry) |r| r.generation.load(.monotonic) else 0;
}

/// The registry's residency bill: the weights of every ready model, media ones included.
fn residentWeightsBytes(io: std.Io) u64 {
    const reg = global_registry orelse return 0;
    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    return reg.current_resident_bytes;
}

/// Memory-only `/props` for a boot with no default chat model (headless
/// media-gen serving). Same `memory` object shape as `renderPropsBody` —
/// `MemoryInfo.parse` client-side reads only that key — with the model
/// fields omitted (there is no model config to describe).
fn handlePropsNoModel(allocator: std.mem.Allocator, stream: *Conn) !void {
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);
    const available_mem = metrics.getAvailableMemBytes();
    const body = try std.fmt.allocPrint(allocator,
        \\{{"total_slots":1,"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":0,"weights_bytes":{d}}},"batching":{{"supported":false,"reason":"no_model","max_group":{d}}},"registry_generation":{d}}}
    , .{ active_mem, peak_mem, available_mem, residentWeightsBytes(stream.io), scheduler_mod.MAX_BATCH_GROUP, registryGeneration() });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

fn handleEmbeddings(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    // The body is parsed first so media is refused before anything else answers; a parse
    // failure is still reported after the engine check below, as before.
    const parsed_or_null: ?std.json.Parsed(std.json.Value) = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch null;
    defer if (parsed_or_null) |p| p.deinit();
    // This engine serves text only: media in `messages` or `input` gets the one 400 first.
    if (parsed_or_null) |p| {
        if (embeddingsRequestMediaReject(p.value)) |why| {
            log.warn("POST /v1/embeddings -> 400 ({s})\n", .{why});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, null);
            return;
        }
    }
    // Engine-backed models have no MLX transformer and both embed
    // paths forward through it: refuse by name before anything is queued.
    const xfm = lm.transformer orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "This model runs on an embedded engine; embeddings are not supported. Load an MLX embedding model instead.", null);
        return;
    };
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    const parsed = parsed_or_null orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", null);
        return;
    };
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", null);
        return;
    }
    const root = parsed.value.object;

    const model_name = echoModelName(root, config.model_type);

    // OpenAI `dimensions` (text-embedding-3 semantics): keep the first N
    // components, L2-renormalize. Anything not a positive integer is a 400 —
    // accepting the field and ignoring it would hand callers wrong-width
    // vectors they only discover at retrieval time. Over-native is checked
    // after the forward, when the model's width is known.
    const req_dims: ?usize = if (root.get("dimensions")) |v| blk: {
        if (v != .integer or v.integer < 1) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'dimensions' must be a positive integer", null);
            return;
        }
        break :blk @intCast(v.integer);
    } else null;

    // `messages` (vLLM's chat embeddings): one prompt, one embedding.
    if (root.get("messages")) |messages_val| {
        if (root.get("input") != null) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Send either 'input' or 'messages', not both", null);
            return;
        }
        return handleMessagesEmbedding(allocator, stream, messages_val);
    }

    // Parse input — can be a string or array of strings
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing 'input' field", null);
        return;
    };

    // Collect input texts
    var texts = std.ArrayList([]const u8).empty;
    defer texts.deinit(allocator);

    switch (input_val) {
        .string => |s| try texts.append(allocator, s),
        .array => |arr| {
            for (arr.items) |item| {
                if (item == .string) {
                    try texts.append(allocator, item.string);
                }
            }
        },
        else => {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' must be a string or array of strings", null);
            return;
        },
    }

    if (texts.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' must not be empty", null);
        return;
    }

    log.info("POST /v1/embeddings ({d} inputs)\n", .{texts.items.len});

    // Tokenize every input up front so the whole request rides ONE scheduler
    // round-trip; the inference thread embeds the sequences in padded,
    // key-masked GPU batches (generate.EMBED_MAX_BATCH per forward) instead
    // of one forward per text.
    var total_tokens: usize = 0;
    var seqs = std.ArrayList([]const u32).empty;
    defer {
        for (seqs.items) |ids| allocator.free(ids);
        seqs.deinit(allocator);
    }
    for (texts.items) |text| {
        if (std.mem.trim(u8, text, " \t\r\n").len == 0) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' entries must be non-empty strings", null);
            return;
        }
        const raw_ids = try tok.encode(allocator, text);
        const ids = wrapEmbeddingIds(raw_ids);
        total_tokens += ids.len;
        try seqs.append(allocator, ids);
    }

    var msg_buf: [160]u8 = undefined;
    for (seqs.items, 0..) |ids, idx| {
        if (embeddingLimitReject(&msg_buf, config, idx, ids.len)) |msg| {
            log.warn("  embedding input {d} over limit: {d} tokens\n", .{ idx, ids.len });
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
            return;
        }
    }

    const embeddings = (try embeddingRows(allocator, stream, lm, xfm, seqs.items)) orelse return;
    defer {
        for (embeddings) |e| allocator.free(e);
        allocator.free(embeddings);
    }
    try sendEmbeddings(allocator, stream, embeddings, req_dims, model_name, total_tokens);
}

/// A tokenized input as the model's pipeline feeds it: a Gemma 4 LM embeds the raw ids, with no special
/// token added.
fn wrapEmbeddingIds(raw_ids: []u32) []u32 {
    return raw_ids;
}

/// Issue #117: the effective per-input token ceiling is enforced BEFORE the forward pass — the server owns the
/// exact tokenizer, so the counts are authoritative. The over-limit 400's text (an explicit refusal, never
/// truncation), or null when `tokens` fits.
fn embeddingLimitReject(buf: []u8, config: *const model_mod.ModelConfig, index: usize, tokens: usize) ?[]const u8 {
    const limit = embedEffectiveLimit(embedding_max_length, config.max_position_embeddings);
    if (limit == 0 or tokens <= limit) return null;
    return embedOverflowMessage(buf, index, tokens, limit);
}

/// Phase A: route through scheduler when available so the encoder forward pass runs on the inference thread
/// (mlx 0.31.2 thread-local streams). Falls back to a direct call only in the offline path where no scheduler
/// exists. `computeEmbeddingsBatch` resets the KV cache before every sub-batch on both paths. Null after the 500.
fn embeddingRows(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    xfm: *transformer_mod.Transformer,
    seqs: []const []const u32,
) !?[][]f32 {
    if (global_scheduler) |sch| {
        var req = scheduler_mod.EmbedRequest{
            .model = lm,
            .token_seqs = seqs,
            .allocator = allocator,
        };
        return sch.computeEmbeddings(&req) catch |err| {
            if (req.error_name) |e| {
                log.err("  embedding error: {s}\n", .{e});
                allocator.free(e);
            } else {
                log.err("  embedding error: {}\n", .{err});
            }
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "Failed to compute embedding", null);
            return null;
        };
    }
    return generate_mod.computeEmbeddingsBatchWith(allocator, xfm, seqs) catch |err| {
        log.err("  embedding error: {}\n", .{err});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "Failed to compute embedding", null);
        return null;
    };
}

fn sendEmbeddings(
    allocator: std.mem.Allocator,
    stream: *Conn,
    embeddings: []const []f32,
    req_dims: ?usize,
    model_name: []const u8,
    total_tokens: usize,
) !void {
    if (req_dims) |d| {
        const native = if (embeddings.len > 0) embeddings[0].len else 0;
        if (d > native) {
            const msg = try std.fmt.allocPrint(allocator, "'dimensions' must be between 1 and {d} for this model", .{native});
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
            return;
        }
    }

    // Build response JSON
    var resp_buf = std.ArrayList(u8).empty;
    defer resp_buf.deinit(allocator);

    try resp_buf.appendSlice(allocator, "{\"object\":\"list\",\"data\":[");

    for (embeddings, 0..) |full_embedding, idx| {
        const embedding = if (req_dims) |d| truncateEmbeddingDims(full_embedding, d) else full_embedding;
        if (idx > 0) try resp_buf.appendSlice(allocator, ",");

        // Format: {"object":"embedding","embedding":[...floats...],"index":N}
        const idx_str = try std.fmt.allocPrint(allocator, "{d}", .{idx});
        defer allocator.free(idx_str);
        try resp_buf.appendSlice(allocator, "{\"object\":\"embedding\",\"embedding\":[");

        for (embedding, 0..) |val, i| {
            if (i > 0) try resp_buf.appendSlice(allocator, ",");
            var buf: [32]u8 = undefined;
            const float_str = std.fmt.bufPrint(&buf, "{d:.8}", .{val}) catch "0";
            try resp_buf.appendSlice(allocator, float_str);
        }

        try resp_buf.appendSlice(allocator, "],\"index\":");
        try resp_buf.appendSlice(allocator, idx_str);
        try resp_buf.appendSlice(allocator, "}");
    }

    const total_str = try std.fmt.allocPrint(allocator, "{d}", .{total_tokens});
    defer allocator.free(total_str);
    const model_escaped = try jsonEscape(allocator, model_name);
    defer allocator.free(model_escaped);

    try resp_buf.appendSlice(allocator, "],\"model\":");
    try resp_buf.appendSlice(allocator, model_escaped);
    try resp_buf.appendSlice(allocator, ",\"usage\":{\"prompt_tokens\":");
    try resp_buf.appendSlice(allocator, total_str);
    try resp_buf.appendSlice(allocator, ",\"total_tokens\":");
    try resp_buf.appendSlice(allocator, total_str);
    try resp_buf.appendSlice(allocator, "}}");

    try sendResponse(stream, "200 OK", "application/json", resp_buf.items);
    log.info("  <- {d} embeddings ({d} tokens)\n", .{ embeddings.len, total_tokens });
}

/// `POST /v1/embeddings` with `messages`: only a multimodal embedding model serves it, so a Gemma 4 LM
/// answers one of three 400s: an empty `messages`, an image, video or audio part, or the pointer to `input`.
fn handleMessagesEmbedding(
    allocator: std.mem.Allocator,
    stream: *Conn,
    messages_val: std.json.Value,
) !void {
    if (messages_val != .array or messages_val.array.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' must be a non-empty array", null);
        return;
    }
    // This engine serves text only: an image, video or audio part is refused first,
    // whatever the model, with the one text-only message.
    if (embeddingMessagesMediaReject(messages_val.array.items)) |why| {
        log.warn("POST /v1/embeddings -> 400 ({s})\n", .{why});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, null);
        return;
    }
    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is not supported by this model; send the text in 'input'", null);
}

/// OpenAI `dimensions` semantics (text-embedding-3 class): keep the first
/// `dims` components and L2-renormalize, in place. `dims >= len` returns the
/// vector untouched so the default (no `dimensions`) path stays byte-identical.
/// A zero prefix is returned unnormalized rather than divided by zero.
fn truncateEmbeddingDims(embedding: []f32, dims: usize) []f32 {
    if (dims == 0 or dims >= embedding.len) return embedding;
    const out = embedding[0..dims];
    var sumsq: f64 = 0;
    for (out) |v| sumsq += @as(f64, v) * @as(f64, v);
    if (sumsq <= 0) return out;
    const inv = 1.0 / @sqrt(sumsq);
    for (out) |*v| v.* = @floatCast(@as(f64, v.*) * inv);
    return out;
}

fn handleTokenize(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    const content = if (root.get("content")) |v| (if (v == .string) v.string else null) else null;
    if (content == null) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'content' is required", 400);
        return;
    }

    const ids = try tok.encode(allocator, content.?);
    defer allocator.free(ids);

    var result = std.ArrayList(u8).empty;
    defer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"tokens\":[");
    for (ids, 0..) |id, i| {
        if (i > 0) try result.append(allocator, ',');
        var num_buf: [12]u8 = undefined;
        const num = std.fmt.bufPrint(&num_buf, "{d}", .{id}) catch continue;
        try result.appendSlice(allocator, num);
    }
    try result.appendSlice(allocator, "]}");

    log.debug("POST /tokenize -> {d} tokens\n", .{ids.len});
    try sendResponse(stream, "200 OK", "application/json", result.items);
}

fn handleDetokenize(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    const tokens_val = root.get("tokens") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' is required", 400);
        return;
    };
    if (tokens_val != .array) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' must be an array", 400);
        return;
    }

    const ids = try allocator.alloc(u32, tokens_val.array.items.len);
    defer allocator.free(ids);
    if (!parseTokenIds(tokens_val.array.items, tokenVocabSize(lm), ids)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' must be integers inside the model's vocabulary", 400);
        return;
    }

    const text = try decodeTokens(allocator, lm, tok, ids, false);
    defer allocator.free(text);

    const result = try detokenizeResponseJson(allocator, text);
    defer allocator.free(result);

    log.debug("POST /detokenize -> {d} tokens -> {d} chars\n", .{ ids.len, text.len });
    try sendResponse(stream, "200 OK", "application/json", result);
}

/// `{"content": <text>}` with the text escaped by the SHARED escaper.
///
/// This handler used to hand-roll a five-character escape table and pass every
/// other byte through raw, so any token whose bytes are below 0x20 produced a
/// body no JSON parser accepts — found live 2026-08-04 detokenizing single ids
/// while cross-checking a vocabulary. Same class as the control-byte rule on
/// the chat render path: a decoded token is arbitrary bytes, so it goes through
/// `appendJsonString`, never a local switch.
fn detokenizeResponseJson(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"content\":");
    try chat_mod.appendJsonString(allocator, &result, text);
    try result.append(allocator, '}');
    return result.toOwnedSlice(allocator);
}

/// OpenAI `n` (choice count): this is a single-choice engine and n>1 is
/// deliberately unimplemented (YAGNI — no agent client sends it). Silently
/// returning one choice for n=2 is the silent-no-op class llmprobe flags;
/// reject with an honest 400 instead. Absent, null, 1, or 1.0 all mean
/// "one choice" and pass.
/// jsonEscape with an OOM fallback to the literal `""`. Ownership is reported
/// by PROVENANCE (`owned`), never inferred from content — escaping an empty
/// string also yields `""` but that one is allocated (2-byte leak per
/// all-reasoning request under the old content-equality defer).
const EscapedText = struct { slice: []const u8, owned: bool };
fn jsonEscapeOrEmpty(allocator: std.mem.Allocator, text: []const u8) EscapedText {
    const s = jsonEscape(allocator, text) catch return .{ .slice = "\"\"", .owned = false };
    return .{ .slice = s, .owned = true };
}

/// `effort` is the client's raw string, borrowed from the parsed request JSON
/// (which outlives the handler) - templates that read the effort word map it
/// into the render; every other consumer only reads enable/budget.
const ReasoningEffort = struct { enable: bool, budget: i32, effort: ?[]const u8 = null };

/// OpenAI-standard `reasoning_effort` on chat/completions (none | minimal |
/// low | medium | high | xhigh — values are model-dependent, so unknown
/// strings still enable). "none" is an explicit off (the gpt-5.1 default
/// spelling). Absent or non-string → null: the vendor `enable_thinking` bool
/// stays in charge and existing clients see zero behavior change.
/// The effort word is the ONLY lever when the template reads it and no
/// decode-time bound can enforce a budget derived from it (a budget that is
/// only a delivery cap fights the word: `chat.templateConsumesEffort`).
fn effortWordOnly(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    const cc = lm.chat_config orelse return false;
    return chat_mod.templateConsumesEffort(cc.chat_template) and !thinkMarkersAtomic(allocator, lm, tok);
}

/// Effort → thinking-budget mapping shared by the Responses `reasoning.effort`
/// object and the chat-completions `reasoning_effort` string. Unknown efforts
/// (model-dependent spec values like "xhigh") fall back to the default budget.
fn effortBudget(effort: []const u8, default_budget: i32) i32 {
    if (std.mem.eql(u8, effort, "minimal")) return 1024;
    if (std.mem.eql(u8, effort, "low")) return 2048;
    if (std.mem.eql(u8, effort, "medium")) return 8192;
    return default_budget;
}

/// The request's `--reasoning-budget`: the generation-defaults budget rule replaces the flag,
/// and effort words map against it exactly as they map against the flag.
fn defaultReasoningBudget(gen: generation_settings.Profile) i32 {
    return gen.value(i32, .reasoning_budget) orelse server_config.default_reasoning_budget;
}

/// A forced budget rule replaces whatever the request and its effort word decided.
fn forcedReasoningBudget(gen: generation_settings.Profile, budget: i32) i32 {
    return if (gen.forced(.reasoning_budget)) gen.value(i32, .reasoning_budget).? else budget;
}

/// A request field at the top level, else inside vLLM's `chat_template_kwargs` object.
fn requestField(root: std.json.ObjectMap, name: []const u8) ?std.json.Value {
    if (root.get(name)) |v| return v;
    const kw = root.get("chat_template_kwargs") orelse return null;
    return if (kw == .object) kw.object.get(name) else null;
}

fn parseReasoningEffort(root: std.json.ObjectMap, default_budget: i32, template_consumes_effort: bool) ?ReasoningEffort {
    const v = requestField(root, "reasoning_effort") orelse return null;
    if (v != .string) return null;
    return reasoningEffortFromWord(v.string, default_budget, template_consumes_effort);
}

/// One effort word → one thinking config, whatever field carried the word.
fn reasoningEffortFromWord(word: []const u8, default_budget: i32, template_consumes_effort: bool) ReasoningEffort {
    if (std.mem.eql(u8, word, "none")) return .{ .enable = false, .budget = default_budget, .effort = word };
    // Where the TEMPLATE reads the effort word, the word is the behavioral
    // lever and a budget derived from the same string is pure display
    // truncation on top of it (`chat.templateConsumesEffort`). An explicit
    // `--reasoning-budget` still rides `default_budget`.
    const budget = if (template_consumes_effort)
        default_budget
    else
        effortBudget(word, default_budget);
    return .{ .enable = true, .budget = budget, .effort = word };
}

fn thinkingFrom(enable: ?bool, effort: ?ReasoningEffort, fallback: bool) bool {
    if (enable == null and effort == null) return fallback;
    return (enable orelse false) or (if (effort) |e| e.enable else false);
}

/// The model's effort word fills in only when thinking ended up on and the request named none.
fn modelEffortIfThinking(model: ?ReasoningEffort, thinking: bool) ?ReasoningEffort {
    const m = model orelse return null;
    return if (thinking and m.enable) m else null;
}

const ChatThinking = struct { enable: bool, effort: ?ReasoningEffort };

/// An EXPLICIT client signal wins (`enable_thinking`, or an effort word, "none" being an
/// explicit OFF; the two stay OR'd). The generation-defaults rules (the model's
/// `chat_template_kwargs` thinking keys among them) act as a default request, consulted only
/// when the request names neither; then the arch. A forced thinking or effort rule replaces the
/// request's switches.
fn resolveThinking(gen: generation_settings.Profile, enable: ?bool, request: ?ReasoningEffort, arch_default: bool, default_budget: i32, word_only: bool) ChatThinking {
    const ignore = gen.forced(.enable_thinking) or gen.forced(.reasoning_effort);
    const req_effort = if (ignore) null else request;
    const model: ?ReasoningEffort = if (gen.value(generation_settings.Effort, .reasoning_effort)) |e|
        reasoningEffortFromWord(@tagName(e), default_budget, word_only)
    else
        null;
    const on = if (gen.forced(.enable_thinking))
        gen.value(bool, .enable_thinking).?
    else if (gen.forced(.reasoning_effort))
        model.?.enable
    else
        thinkingFrom(enable, req_effort, thinkingFrom(gen.value(bool, .enable_thinking), model, arch_default));
    return .{ .enable = on, .effort = req_effort orelse modelEffortIfThinking(model, on) };
}

fn resolveChatThinking(root: std.json.ObjectMap, gen: generation_settings.Profile, arch_default: bool, default_budget: i32, word_only: bool) ChatThinking {
    const enable: ?bool = if (requestField(root, "enable_thinking")) |v| (if (v == .bool) v.bool else null) else null;
    return resolveThinking(gen, enable, parseReasoningEffort(root, default_budget, word_only), arch_default, default_budget, word_only);
}

const SchemaThinkingPolicy = enum {
    no_mask,
    token_zero,
    deferred,
    fallback_thinking_off,
};

/// A schema grammar is safe with thinking only when generation can hand the
/// channel machinery to a resolved reasoning protocol (a prompt-opened think
/// block whose close the tokenizer can represent, or — for think-tag
/// templates — the unresolved opener choice). Every other masked surface
/// keeps the upstream thinking-off safety net. A finite response-side
/// reasoning budget also keeps that fallback because it can end the visible
/// think block before generation reaches the grammar boundary. Tools still
/// skip the mask.
fn schemaMasksThinking(
    has_schema: bool,
    has_tools: bool,
    enable_thinking: bool,
    has_finite_reasoning_budget: bool,
    protocol_ready: bool,
) SchemaThinkingPolicy {
    if (!has_schema or has_tools) return .no_mask;
    if (!enable_thinking) return .token_zero;
    if (has_finite_reasoning_budget) return .fallback_thinking_off;
    if (protocol_ready) return .deferred;
    return .fallback_thinking_off;
}

/// `continue_final_message`: extend the trailing assistant message instead of
/// answering after it. vLLM's spelling, because a client that already knows how
/// to ask another local server for this should not have to learn a second name.
///
/// The flag alone is not enough — the conversation has to BE continuable
/// (`chat_mod.continuationRequested`). A flag set on a user-final request is
/// ignored rather than 400'd: the request is perfectly answerable as an
/// ordinary turn, and refusing it would break the obvious client that sets the
/// field once for the whole session.
fn continueFinalMessageRequested(root: std.json.ObjectMap, messages: []const chat_mod.Message) bool {
    const v = root.get("continue_final_message") orelse return false;
    if (v != .bool or !v.bool) return false;
    return chat_mod.continuationRequested(messages);
}

const JoinedText = struct { text: []const u8, owned: bool };

/// Collect every `{"type":"text"}` part of a content array into one string,
/// in order, '\n'-joined. A single (or no) text part borrows the JSON's
/// bytes (`owned=false`); 2+ parts allocate (`owned=true`, caller frees).
///
/// Multiple text parts are real traffic: opencode plan mode appends its
/// "Plan Mode - System Reminder" as a SECOND text part on the user message,
/// and last-wins here dropped the user's actual prompt — the model answered
/// "What would you like to accomplish?" to every first plan-mode message
/// (issue #195).
fn joinedTextParts(allocator: std.mem.Allocator, parts: []const std.json.Value) !JoinedText {
    var single: []const u8 = "";
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var n: usize = 0;
    for (parts) |part| {
        if (part != .object) continue;
        const ptype = part.object.get("type") orelse continue;
        if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
        const tv = part.object.get("text") orelse continue;
        if (tv != .string or tv.string.len == 0) continue;
        n += 1;
        if (n == 1) {
            single = tv.string;
            continue;
        }
        if (n == 2) try buf.appendSlice(allocator, single);
        try buf.append(allocator, '\n');
        try buf.appendSlice(allocator, tv.string);
    }
    if (n <= 1) return .{ .text = single, .owned = false };
    return .{ .text = try buf.toOwnedSlice(allocator), .owned = true };
}

/// Workload key for hot-cache eviction (#378): `prompt_cache_key` (OpenAI's own
/// routing field) > `metadata.user_id` (Anthropic; Claude Code sends its session
/// id) > the system prompt (OpenAI `messages[0]`, Anthropic `system`, Responses
/// `instructions`) > 0 = anonymous. Text parts hash in order like `joinedTextParts`.
fn requestCacheKey(root: std.json.ObjectMap) u64 {
    if (root.get("prompt_cache_key")) |v| if (v == .string and v.string.len > 0)
        return std.hash.Wyhash.hash(1, v.string);
    if (root.get("metadata")) |mv| if (mv == .object) {
        if (mv.object.get("user_id")) |v| if (v == .string and v.string.len > 0)
            return std.hash.Wyhash.hash(2, v.string);
    };
    if (root.get("messages")) |mv| if (mv == .array and mv.array.items.len > 0) {
        const first = mv.array.items[0];
        if (first == .object) if (first.object.get("role")) |r| if (r == .string and std.mem.eql(u8, chat_mod.canonicalRole(r.string), "system")) {
            if (first.object.get("content")) |c| return hashTextValue(c) orelse 0;
        };
    };
    if (root.get("system")) |v| return hashTextValue(v) orelse 0;
    if (root.get("instructions")) |v| return hashTextValue(v) orelse 0;
    return 0;
}

fn hashTextValue(v: std.json.Value) ?u64 {
    var h = std.hash.Wyhash.init(3);
    var n: usize = 0;
    switch (v) {
        .string => |t| {
            if (t.len == 0) return null;
            h.update(t);
            n = 1;
        },
        .array => |parts| for (parts.items) |part| {
            if (part != .object) continue;
            const ptype = part.object.get("type") orelse continue;
            if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
            const tv = part.object.get("text") orelse continue;
            if (tv != .string or tv.string.len == 0) continue;
            if (n > 0) h.update("\n");
            h.update(tv.string);
            n += 1;
        },
        else => {},
    }
    return if (n == 0) null else h.final();
}

/// The `model` a response echoes: the client's own string, as SDKs expect, unless it cannot
/// sit verbatim inside a JSON string (the response templates splice it unescaped).
fn echoModelName(root: std.json.ObjectMap, fallback: []const u8) []const u8 {
    const v = root.get("model") orelse return fallback;
    if (v != .string or !std.unicode.utf8ValidateSlice(v.string)) return fallback;
    for (v.string) |c| if (c < 0x20 or c == '"' or c == '\\') return fallback;
    return v.string;
}

/// The vocabulary a request's token ids index.
fn tokenVocabSize(lm: *const LoadedModel) u32 {
    return if (lm.config) |c| c.vocab_size else 0;
}

/// Client token ids into `out` (same length); false unless every item is an integer in [0, vocab_size).
fn parseTokenIds(items: []const std.json.Value, vocab_size: u32, out: []u32) bool {
    for (items, out) |item, *id| {
        if (item != .integer or item.integer < 0 or item.integer >= vocab_size) return false;
        id.* = @intCast(item.integer);
    }
    return true;
}

const PROMPT_LOGPROBS_UNAVAILABLE_MSG = "'echo' with 'logprobs' is not supported on this model: its prefill does not return every prompt position's logits";

fn nChoicesRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    const v = root.get("n") orelse return null;
    switch (v) {
        .null => return null,
        .integer => |i| if (i == 1) return null,
        .float => |f| if (f == 1.0) return null,
        else => {},
    }
    return "'n' > 1 is not supported: this server returns a single choice per request; omit 'n' or set it to 1";
}

/// The request's `enable_pld`, else the launch default.
fn requestPld(root: std.json.ObjectMap) bool {
    return if (root.get("enable_pld")) |v| v == .bool and v.bool else server_config.default_enable_pld;
}

fn handleChatCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    // NOTE: no `lm.transformer.?` here. The only MLX-specific gate below reads
    // `config.has_hybrid_layers`, so the transformer is never needed at this level.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    // Parse JSON body
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/chat/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    // A valid-JSON body that isn't an object (e.g. `42` or `[1,2]`) would panic
    // on the `.object` field access below and take the whole server down, so
    // reject it as a 400 like any other malformed request.
    if (parsed.value != .object) {
        log.warn("POST /v1/chat/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // This engine serves text only: media anywhere in `messages` is refused before any other
    // field is validated. The per-entry flag in the parse loop below stays as the second net.
    if (chatRequestMediaReject(root)) |reason| {
        log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{reason});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    if (nChoicesRejectReason(root)) |reason| {
        log.warn("POST /v1/chat/completions -> 400 (unsupported n)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    // Extract messages
    const messages_val = root.get("messages") orelse {
        log.warn("POST /v1/chat/completions -> 400 (missing messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is a required field", 400);
        return;
    };

    // `messages` present but not an array (e.g. a string) would panic on the
    // `.array` access below. Reject rather than crash.
    if (messages_val != .array) {
        log.warn("POST /v1/chat/completions -> 400 (messages is not an array)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' must be an array", 400);
        return;
    }

    var messages = std.ArrayList(chat_mod.Message).empty;
    defer messages.deinit(allocator);

    // Parse tool call structs for assistant messages (stored temporarily)
    var tool_call_lists = std.ArrayList([]const chat_mod.ToolCall).empty;
    defer {
        for (tool_call_lists.items) |tcs| allocator.free(tcs);
        tool_call_lists.deinit(allocator);
    }

    // Joined multi-part text content (allocated only when a message carries
    // 2+ text parts).
    var content_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (content_allocs.items) |s| allocator.free(s);
        content_allocs.deinit(allocator);
    }

    var saw_entry_media = false;
    for (messages_val.array.items) |msg_val| {
        // A non-object array element (e.g. `messages:[1,2,3]`) would panic on
        // `.object`. Skip it rather than crash — consistent with how malformed
        // inner fields are already tolerated below.
        if (msg_val != .object) continue;
        const obj = msg_val.object;
        // Media is refused whatever the entry's role, even one the parse below skips.
        if (messageEntryHasMedia(msg_val)) saw_entry_media = true;
        const role_val = obj.get("role") orelse continue;
        if (role_val != .string) continue;

        // Content can be null for assistant messages with tool_calls
        const content_val = obj.get("content");
        // This engine serves text only: an image, video or audio part is
        // recorded as a non-null empty slice (never decoded), and
        // `mediaRejectReason` answers the request with the one 400.
        const marks: MediaMarks = if (content_val) |cv| contentMediaMarks(cv) else .{};
        const msg_images: ?[]const chat_mod.ImageData = if (marks.image) &[_]chat_mod.ImageData{} else null;
        const msg_videos: ?[]const chat_mod.VideoData = if (marks.video) &[_]chat_mod.VideoData{} else null;
        const msg_audio: ?[]const chat_mod.AudioData = if (marks.audio) &[_]chat_mod.AudioData{} else null;
        const content: []const u8 = if (content_val) |cv| switch (cv) {
            .string => |s| s,
            .array => |arr| blk: {
                const joined = try joinedTextParts(allocator, arr.items);
                if (joined.owned) try content_allocs.append(allocator, joined.text);
                break :blk joined.text;
            },
            else => "",
        } else "";

        // Parse tool_calls from assistant messages
        var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
        if (std.mem.eql(u8, role_val.string, "assistant")) {
            if (obj.get("tool_calls")) |tc_val| {
                if (tc_val == .array) {
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;
                    for (tc_val.array.items) |tc_item| {
                        if (tc_item != .object) continue;
                        const tc_id = if (tc_item.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                        const func = tc_item.object.get("function") orelse continue;
                        if (func != .object) continue;
                        const fn_name = if (func.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                        const fn_args = if (func.object.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";
                        try tcs.append(allocator, .{ .id = tc_id, .name = fn_name, .arguments = fn_args });
                    }
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }
                }
            }
        }

        // Parse tool_call_id from tool messages
        const tool_call_id: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "tool"))
            (if (obj.get("tool_call_id")) |v| (if (v == .string) v.string else null) else null)
        else
            null;

        // Reasoning the client round-trips on assistant history. Dropping it
        // starves templates that persist reasoning across turns: every
        // prior turn renders the empty <think></think> nothink signature and
        // the model stops thinking from turn 2 of a session.
        const msg_reasoning: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "assistant"))
            messageReasoningFromObj(obj)
        else
            null;

        // Skip messages with no content, no tool_calls, and no images/videos/audio
        if (content.len == 0 and msg_tool_calls == null and msg_images == null and msg_videos == null and msg_audio == null and msg_reasoning == null and !std.mem.eql(u8, role_val.string, "tool")) continue;

        try messages.append(allocator, .{
            .role = chat_mod.canonicalRole(role_val.string),
            .content = content,
            .tool_calls = msg_tool_calls,
            .tool_call_id = tool_call_id,
            .images = msg_images,
            .videos = msg_videos,
            .audio = msg_audio,
            .reasoning_content = msg_reasoning,
        });
    }

    if (saw_entry_media) {
        log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{TEXT_ONLY_MEDIA_REJECT});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", TEXT_ONLY_MEDIA_REJECT, 400);
        return;
    }

    if (messages.items.len == 0) {
        log.warn("POST /v1/chat/completions -> 400 (no valid messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    // Support both max_tokens and max_completion_tokens (OpenAI alias). Absent
    // or <= 0 → auto (peg to remaining context).
    const requested_max_tokens = root.get("max_tokens") orelse root.get("max_completion_tokens");
    const gen = requestProfile(stream.io, lm);
    const max_tokens: u32 = generationMaxTokens(
        gen,
        requested_max_tokens,
        omittedMaxTokensDefault(getEffectiveContextLength(config)),
    );
    // For the request log line only: which of the three sources this came from.
    const max_tokens_origin = maxTokensOrigin(requested_max_tokens, launchMaxTokensDefault());

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const sampled = resolveSampling(gen, root, config);
    const temperature = sampled.temperature;
    const top_p = sampled.top_p;
    const top_k = sampled.top_k;
    const min_p = sampled.min_p;

    const repeat_penalty = sampled.repeat_penalty;

    const presence_penalty = sampled.presence_penalty;

    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // Parse logprobs: "logprobs": true, "top_logprobs": N (0..MAX_TOP_LOGPROBS)
    const logprobs_n: u32 = blk: {
        const lp = root.get("logprobs") orelse break :blk 0;
        if (lp != .bool or !lp.bool) break :blk 0;
        // logprobs=true without top_logprobs defaults to 0 (just the chosen token's logprob)
        const tlp = root.get("top_logprobs") orelse break :blk 1;
        break :blk switch (tlp) {
            .integer => |i| @intCast(@min(@max(i, 0), generate_mod.MAX_TOP_LOGPROBS)),
            else => 1,
        };
    };

    // Extract tools JSON from request body for chat template injection
    var tools_json: ?[]const u8 = null;
    var has_tools = root.get("tools") != null;
    var forced_tool: ?chat_mod.ForcedTool = null;

    // OpenAI parallel_tool_calls: only an explicit false clamps to one call
    // per response (the SDK sets false in strict structured-output mode).
    const allow_parallel_tools = if (root.get("parallel_tool_calls")) |v|
        !(v == .bool and !v.bool)
    else
        true;

    if (has_tools) {
        // Parse tool_choice: "none" | "auto" | "required" | {"type":"function","function":{"name":"..."}}
        if (root.get("tool_choice")) |tc| {
            if (tc == .string) {
                if (std.mem.eql(u8, tc.string, "none")) {
                    has_tools = false; // Don't inject tools at all
                } else if (std.mem.eql(u8, tc.string, "required")) {
                    forced_tool = .any;
                }
                // "auto" is the default behavior
            } else if (tc == .object) {
                // Specific function: {"type":"function","function":{"name":"fn_name"}}
                if (tc.object.get("function")) |func| {
                    if (func == .object) {
                        if (func.object.get("name")) |name_val| {
                            if (name_val == .string) forced_tool = .{ .name = name_val.string };
                        }
                    }
                }
            }
        }

        if (has_tools) {
            // Find the tools array in the raw JSON body and extract it
            if (extractJsonField(body, "tools")) |tools_str| {
                tools_json = tools_str;
            }
        }
    }
    if (forced_tool) |ft| if (ft == .name and !chat_mod.toolsDeclare(root.get("tools"), ft.name)) {
        const msg = try std.fmt.allocPrint(allocator, "tool_choice names function \"{s}\", which is not in tools", .{ft.name});
        defer allocator.free(msg);
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return;
    };
    const tool_choice_instruction: ?[]const u8 = if (forced_tool) |ft| try chat_mod.toolChoiceInstruction(allocator, ft) else null;
    defer if (tool_choice_instruction) |tci| allocator.free(tci);

    // Parse stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |stop_val| {
        switch (stop_val) {
            .string => |s| if (s.len > 0) try stop_sequences.append(allocator, s),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string and item.string.len > 0) try stop_sequences.append(allocator, item.string);
                }
            },
            else => {},
        }
    }

    // Parse model name from request (use for response, fallback to config)
    const model_name = echoModelName(root, config.model_type);

    // Track allocations from response_format injection so we can free them
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }

    // Parse response_format — inject JSON schema constraint into system message,
    // and capture the schema value for grammar-constrained sampling below.
    var grammar_schema_val: ?std.json.Value = null;
    // Backing storage for the synthetic `{"type":"object"}` schema used for
    // `response_format: {type: "json_object"}`. Lives for the request and is
    // freed at the bottom; the schema parser arena copies what it needs.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (root.get("response_format")) |rf| {
        if (rf == .object) {
            const rf_type = if (rf.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, rf_type, "json_schema")) {
                const declared: ?std.json.Value = if (rf.object.get("json_schema")) |js| (if (js == .object) js.object.get("schema") else null) else null;
                if (declared == null or declared.? != .object) {
                    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "response_format.json_schema.schema must be a JSON object (use {\"type\":\"json_object\"} for unconstrained JSON)", 400);
                    return;
                }
                // Extract the schema JSON string from the raw body
                var schema_instruction = std.ArrayList(u8).empty;
                defer schema_instruction.deinit(allocator);
                try schema_instruction.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");

                if (rf.object.get("json_schema")) |js| {
                    if (js == .object) {
                        if (js.object.get("schema")) |schema_val| {
                            grammar_schema_val = schema_val;
                            var out: std.Io.Writer.Allocating = .init(allocator);
                            defer out.deinit();
                            var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
                            schema_val.jsonStringify(&jws) catch {};
                            try schema_instruction.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
                            try schema_instruction.appendSlice(allocator, out.written());
                        }
                    }
                }

                const instruction = try allocator.dupe(u8, schema_instruction.items);
                try rf_allocs.append(allocator, instruction);
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
            } else if (std.mem.eql(u8, rf_type, "json_object")) {
                const instruction = "Respond with valid JSON only. No other text, no markdown fences (no ``` or ```json), no explanation. Begin your response with `{` or `[`.";
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
                // Belt + braces: also constrain decoding with a permissive
                // object schema so the very first token cannot be a leading
                // backtick (Gemma 4 ignores the prompt-side instruction
                // otherwise). `additionalProperties: true` allows any keys
                // and values — `json_object` only constrains JSON-ness, not
                // a specific shape.
                if (json_object_schema_holder == null) {
                    json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
                }
                if (json_object_schema_holder) |*p| {
                    grammar_schema_val = p.value;
                }
            }
        }
    }

    // Parse stream_options
    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Thinking opt-ins: the OpenAI-standard `reasoning_effort` string and the
    // vendor `enable_thinking` bool (the vLLM chat_template_kwargs family).
    // Either switch turns thinking on; effort "none" alone never does.
    // A request naming NEITHER takes the arch default (off for every arch but
    // the ones whose vendor documents thinking-on).
    const default_budget = defaultReasoningBudget(gen);
    const thinking = resolveChatThinking(root, gen, config.defaultEnableThinking(), default_budget, effortWordOnly(allocator, lm, tok));
    const effort_cfg = thinking.effort;
    var enable_thinking = thinking.enable;

    // Reasoning budget (max tokens in <think> block, -1 = unlimited):
    // explicit reasoning_budget_tokens > effort-mapped budget > --reasoning-budget flag
    const effort_budget: i32 = if (effort_cfg) |e| e.budget else default_budget;
    const reasoning_budget: i32 = forcedReasoningBudget(gen, if (root.get("reasoning_budget_tokens")) |v| switch (v) {
        .integer => |i| clampJsonI32(i),
        else => effort_budget,
    } else effort_budget);

    // Wave 1.A: per-request KV-quant override. When unset, falls back to the
    // process-level --kv-quant default carried on the scheduler. Cross-scheme
    // hot-prefix-cache hits never happen — entries record their scheme and
    // findBestMatch filters on it.
    const kv_quant_override = parseKvQuantOverride(root);
    if (kv_quant_override) |kq| {
        switch (kq.scheme) {
            .off => log.info("  kv-quant override: off (per-request)\n", .{}),
            .affine => log.info("  kv-quant override: affine {d}-bit (per-request)\n", .{kq.bits}),
        }
    }
    const kv_attn_explicit = parseKvAttnExplicit(root);

    // Parse enable_pld: per-request override of the --pld default.
    //
    // `pld_explicit_in_json` records whether the request body had `enable_pld`
    // as an explicit field (vs falling back to the server default). The
    // adaptive gate (later, after prompt is tokenized) only disables flags
    // that came from the default — explicit user overrides are honored even
    // on novel content.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = requestPld(root);
    // Tools do NOT disable PLD: tool-pattern detection operates on emitted
    // text and is agnostic to how many tokens a decode step yields, and
    // agent traffic (tool results echoed into edits) is PLD's best workload.
    // The original blanket gate predated streaming PLD + scheduler slots;
    // equivalence with tools is pinned by tests/test_pld_tools.sh.
    //
    // PLD on hybrid recurrent-state models (GatedDeltaNet) works once the state
    // snapshot/restore paths handle null ssm_state correctly — see
    // `ssmSnapshot`/`ssmRestore` in transformer.zig.
    if (enable_pld) log.info("  pld=enabled (draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });

    // Tools do not disable the drafter; logprobs require the target's plain path.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter = requestEnableDrafter(root.get("enable_drafter"), lm.drafter != null, lm.dflash != null, config.isMoe());
    if (enable_drafter and logprobs_n > 0) {
        log.info("  drafter=disabled (logprobs requested)\n", .{});
        enable_drafter = false;
    }
    if (enable_drafter) {
        if (enable_pld) {
            log.info("  pld=disabled (drafter takes priority for this request)\n", .{});
            enable_pld = false;
        }
        log.info("  drafter=enabled (block_size={d})\n", .{lm.drafter_block_size});
    }

    // Log the request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);

    // Compute sizes for debug info
    var system_chars: usize = 0;
    var user_chars: usize = 0;
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            system_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "user")) {
            user_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            tool_msg_count += 1;
        }
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;

    // Format chat template: the MLX path renders via Jinja and tokenizes
    // through the loaded BPE tokenizer, threading `tools_json` and
    // `tool_choice_instruction` through.
    //
    // Phase 4 instrumentation + Iteration 2 cache: time the render+tokenize step.
    // Continuation is EXPLICIT on this surface (`continue_final_message`, the
    // spelling vLLM uses). It cannot be implied by a trailing assistant
    // message the way other surfaces imply it: agent frameworks legitimately
    // POST assistant-last history here and expect a fresh turn, and turning
    // that into a prefill would change what every one of them gets back.
    const continue_final = continueFinalMessageRequested(root, messages.items);
    if (mediaRejectReason(messages.items)) |reason| {
        log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{reason});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }
    // The request's `chat_template_kwargs` merge over the model's for this render only.
    var render_config = chat_config.*;
    const request_kwargs: ?[]const u8 = if (root.get("chat_template_kwargs")) |kw|
        (if (kw == .object) try chat_mod.mergeTemplateKwargs(allocator, chat_config.chat_template_kwargs, kw.object) else null)
    else
        null;
    defer if (request_kwargs) |k| allocator.free(k);
    if (request_kwargs) |k| render_config.chat_template_kwargs = k;
    var tokenize_sw = Stopwatch.init(stream.io);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, enable_thinking, if (effort_cfg) |e| e.effort else null, continue_final);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (grammar_schema_val != null and !has_tools and enable_thinking and reasoning_budget < 0) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (schemaMasksThinking(
        grammar_schema_val != null,
        has_tools,
        enable_thinking,
        reasoning_budget >= 0,
        schema_proto_active,
    )) {
        .deferred => {},
        .fallback_thinking_off => {
            schema_proto_active = false;
            allocator.free(prompt_ids_raw);
            enable_thinking = false;
            prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, false, if (effort_cfg) |e| e.effort else null, continue_final);
            log.info("[grammar] {s}; rerendered with thinking off\n", .{if (reasoning_budget >= 0) "finite reasoning budget" else "reasoning protocol unsupported for this model/prompt"});
        },
        .no_mask, .token_zero => schema_proto_active = false,
    }
    if (grammar_schema_val != null and !has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, false)) {
            schema_proto = rp_mod.Protocol.jsonOnly();
        }
        schema_proto_active = true;
    }
    const tokenize_ns = tokenize_sw.read();

    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    log.info("POST /v1/chat/completions ({d} msgs, max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, user={d}b, tools={d}b, tool_msgs={d}) \n", .{ messages.items.len, describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream, enable_thinking, system_chars, user_chars, tools_len, tool_msg_count });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    const cache_key = requestCacheKey(root);
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/chat/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, kv_quant_override, lm, generate_mod.visionPrefillUnchunked(false), 0)) return;

    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    // ── Adaptive spec-decode gating ──
    //
    // PLD and drafter both pay a per-step overhead (lookup + verify forward
    // for PLD; drafter forwards + verify forward for drafter) that only pays
    // off on echo-heavy content. On novel content, the verify forward is wasted
    // work — the bench shows drafter at 0.55× on creative content and PLD at
    // 0.91× on a heavy-echo prompt. To make the default-on flip safe we
    // gate per-request on the prompt's n-gram repetition score.
    //
    // Rule: if the score (= ratio of distinct 3-grams that recur in the prompt)
    // is below `spec_gate_threshold` AND the user did not explicitly set the
    // flag in the JSON, disable PLD/drafter for this request. Explicit user
    // overrides are honored — if you `enable_pld:true` on novel content, you
    // get PLD even though it's likely a perf loss; user knows best.
    if ((enable_pld and !pld_explicit_in_json) or (enable_drafter and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0; // on error, don't gate
        log.info("  spec-gate: ngram-score={d:.3} (threshold={d:.3})\n", .{ score, spec_gate_threshold });
        if (score < spec_gate_threshold) {
            if (enable_pld and !pld_explicit_in_json) {
                log.info("  pld=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_pld = false;
            }
            // A DFlash drafter is exempt: its runtime yield gate disables on
            // REALIZED acceptance after `DFLASH_GATE_WARMUP` rounds (8 wasted
            // verifies at worst), strictly better evidence than a prompt-time heuristic
            // that cannot see output echo — and llmprobe/bench bodies cannot
            // carry enable_drafter:true. The gemma cross-attention drafter
            // (0.55x measured on novel) keeps the gate.
            if (enable_drafter and !drafter_explicit_in_json and lm.dflash == null) {
                log.info("  drafter=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_drafter = false;
            }
        }
    }

    // Prompt caching: reuse KV cache for shared prefix.
    // Force invalidation when images are present — image tokens have identical IDs
    // but different vision embeddings, so prefix matching would reuse stale features.
    const eos_slice = requestEosSlice(config, root);

    var sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .frequency_penalty = sampled.frequency_penalty,
        .seed = seed,
    };

    // Build grammar-constrained sampling state if a JSON schema was supplied.
    // Lifetime is scoped to this request — the SchemaConstraint must NOT be
    // moved (the embedded Constraint holds pointers into it).
    var sc: generate_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();

    if (grammar_schema_val) |sv| {
        if (has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try lm.grammarTokenBytes(allocator, stream.io);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                var deferred = false;
                if (schema_proto_active) {
                    sc.deferWithProtocol(&schema_proto);
                    deferred = schema_proto.startState().phase != .json_body;
                }
                sampling.constraint = &sc.constraint;
                if (deferred) {
                    log.info("[grammar] deferring JSON schema until the reasoning boundary (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                } else {
                    log.info("[grammar] enforcing JSON schema from token 0 (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                }
            } else |err| {
                log.warn("[grammar] schema parse failed ({s}); falling back to prompt-only enforcement\n", .{@errorName(err)});
            }
        }
    }

    // A decode-time bound owns the budget; the surfaces then deliver the
    // whole (closed) thought instead of trimming it.
    var think_bound = armThinkBound(allocator, lm, tok, prompt_ids, enable_thinking, reasoning_budget);
    defer if (think_bound) |tb| allocator.free(tb.forced);
    const surface_budget: i32 = if (think_bound != null) -1 else reasoning_budget;
    if (think_bound) |*tb| sampling.think_bound = tb;

    var tool_force = armToolForce(allocator, lm, tok, prompt_ids, forced_tool);
    defer if (tool_force) |tf| allocator.free(tf.forced);
    if (tool_force) |*tf| sampling.tool_force = tf;
    // One token per tick: no draft round can carry the turn past the forced opener.
    if (tool_force != null) {
        enable_pld = false;
        enable_drafter = false;
    }

    if (is_stream) {
        handleStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_pld, enable_drafter, null, &.{}, cache_key, kv_quant_override, kv_attn_explicit, tokenize_ns) catch |err| {
            logStreamError(err);
            // One mapping with the non-streaming arm: an HTTP status before the SSE head, an SSE `error` event after.
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_pld, enable_drafter, null, &.{}, cache_key, kv_quant_override, kv_attn_explicit, tokenize_ns) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    }
}

/// Where a stop sequence cuts the accumulated text and how many bytes of the arriving token
/// survive: streaming delivers the same `text[0..index]` non-streaming keeps.
const StopCut = struct { index: usize, token_keep: usize, matched: []const u8 };

fn stopSequenceCut(text: []const u8, token_len: usize, stops: []const []const u8) ?StopCut {
    const emitted = text.len - @min(token_len, text.len);
    var best: ?StopCut = null;
    for (stops) |stop_seq| {
        if (stop_seq.len == 0) continue;
        const idx = std.mem.indexOf(u8, text, stop_seq) orelse continue;
        if (best) |b| if (b.index <= idx) continue;
        best = .{ .index = idx, .token_keep = if (idx > emitted) idx - emitted else 0, .matched = stop_seq };
    }
    return best;
}

/// Bytes at the end of `text`, at most `fresh` of them, that could still begin a stop string.
fn stopPrefixTail(text: []const u8, fresh: usize, stops: []const []const u8) usize {
    var hold: usize = 0;
    for (stops) |stop| {
        var n = @min(stop.len -| 1, fresh);
        while (n > hold) : (n -= 1) {
            if (std.mem.endsWith(u8, text, stop[0..n])) {
                hold = n;
                break;
            }
        }
    }
    return hold;
}

/// Client stop strings on a stream: a tail that could still begin one is held until the next
/// token decides it, so the stream sends exactly the bytes the finished reply keeps.
const StopStream = struct {
    held: std.ArrayList(u8) = .empty,
    /// The held tail went out as the stream's last token.
    flushed: bool = false,

    const Fed = struct { send: []u8, stop: ?[]const u8 };

    fn deinit(self: *StopStream, a: std.mem.Allocator) void {
        self.held.deinit(a);
    }

    fn holding(self: *const StopStream) bool {
        return self.held.items.len > 0 and !self.flushed;
    }

    /// The held tail as one last token, once generation ended without a match.
    fn takeHeld(self: *StopStream, a: std.mem.Allocator) ![]u8 {
        self.flushed = true;
        const out = try a.dupe(u8, self.held.items);
        self.held.clearRetainingCapacity();
        return out;
    }

    /// Appends `token` to `buf` (the text sent so far) and returns what may go out now: up to
    /// a stop match (judged answer-only like `answerStopCut` when `answer_opened` is set), else
    /// everything but a tail that could still begin a stop string. The caller owns `send`.
    fn feed(self: *StopStream, a: std.mem.Allocator, buf: *std.ArrayList(u8), token: []const u8, stops: []const []const u8, answer_opened: ?bool) !Fed {
        const sent = buf.items.len;
        try buf.appendSlice(a, self.held.items);
        self.held.clearRetainingCapacity();
        try buf.appendSlice(a, token);
        const fresh = buf.items.len - sent;
        const cut = if (answer_opened) |o| answerStopCut(buf.items, fresh, stops, o) else stopSequenceCut(buf.items, fresh, stops);
        var end = buf.items.len;
        if (cut) |c| {
            end = @max(c.index, sent);
        } else if (!self.flushed) {
            end -= stopPrefixTail(buf.items, fresh, stops);
            try self.held.appendSlice(a, buf.items[end..]);
        }
        const send = try a.dupe(u8, buf.items[sent..end]);
        buf.shrinkRetainingCapacity(end);
        return .{ .send = send, .stop = if (cut) |c| c.matched else null };
    }
};

/// The text a stream of `text` split at `i` and `j` sends through a `StopStream`, and whether it stopped.
fn stopStreamReplay(a: std.mem.Allocator, text: []const u8, i: usize, j: usize, stops: []const []const u8) !struct { sent: []u8, stopped: bool } {
    var ss: StopStream = .{};
    defer ss.deinit(a);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for ([_][]const u8{ text[0..i], text[i..j], text[j..] }) |token| {
        const fed = try ss.feed(a, &buf, token, stops, null);
        defer a.free(fed.send);
        try out.appendSlice(a, fed.send);
        if (fed.stop != null) return .{ .sent = try out.toOwnedSlice(a), .stopped = true };
    }
    if (ss.holding()) {
        const tail = try ss.takeHeld(a);
        defer a.free(tail);
        const fed = try ss.feed(a, &buf, tail, stops, null);
        defer a.free(fed.send);
        try out.appendSlice(a, fed.send);
    }
    return .{ .sent = try out.toOwnedSlice(a), .stopped = false };
}

test "StopStream: at every token split, a stream sends what the finished reply keeps" {
    const a = std.testing.allocator;
    const Case = struct { text: []const u8, stop: []const u8, want: []const u8, stopped: bool };
    const cases = [_]Case{
        .{ .text = "9, 10, 11, 12, 13", .stop = ", 12", .want = "9, 10, 11", .stopped = true },
        .{ .text = "28, 29, 30", .stop = ", 30x", .want = "28, 29, 30", .stopped = false },
        .{ .text = "a, b, a", .stop = "a, c", .want = "a, b, a", .stopped = false },
    };
    for (cases) |c| {
        const stops = [_][]const u8{c.stop};
        for (0..c.text.len + 1) |i| for (i..c.text.len + 1) |j| {
            const r = try stopStreamReplay(a, c.text, i, j, &stops);
            defer a.free(r.sent);
            try std.testing.expectEqualStrings(c.want, r.sent);
            try std.testing.expectEqual(c.stopped, r.stopped);
        };
    }
}

/// `stopSequenceCut` for a chat surface: a stop string ends the answer, never the reasoning.
/// Only matches ending in the arriving token are new; earlier ones were judged already.
fn answerStopCut(text: []const u8, token_len: usize, stops: []const []const u8, opened_by_template: bool) ?StopCut {
    const emitted = text.len - @min(token_len, text.len);
    var best: ?StopCut = null;
    for (stops) |stop_seq| {
        if (stop_seq.len == 0) continue;
        const idx = chat_mod.answerStopIndex(text, emitted -| (stop_seq.len - 1), stop_seq, opened_by_template) orelse continue;
        if (best) |b| if (b.index <= idx) continue;
        best = .{ .index = idx, .token_keep = if (idx > emitted) idx - emitted else 0, .matched = stop_seq };
    }
    return best;
}

/// A client that vanished mid-generation did not hit a token cap.
fn nonStreamFinishReason(client_gone: bool, slot_reason: []const u8) []const u8 {
    return if (client_gone) "client_disconnect" else slot_reason;
}

/// The text the model said before the tool markup, byte for byte as the stream flushes it.
fn visibleToolPreamble(content: []const u8) []const u8 {
    return if (std.mem.trim(u8, content, " \t\r\n").len > 0) content else "";
}

/// Run a non-streaming generation through the scheduler. Returns the same
/// shape as `generate.generate` so the calling handler's response builder
/// is unchanged.
///
/// `vision_embeddings` (Phase A4/A8): when non-null, ownership transfers
/// into the slot — the slot's deinit frees the array. Each handler holds a
/// per-request `?mlx_array` local; it nulls its copy before passing here so
/// its own `defer` doesn't double-free.
fn nonStreamingViaScheduler(
    allocator: std.mem.Allocator,
    sch: *scheduler_mod.Scheduler,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    full_prompt: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    cached_tokens: u32,
    has_tools: bool,
    enable_thinking: bool,
    enable_pld: bool,
    enable_drafter: bool,
    timeout_ns: u64,
    vision_embeddings: ?mlx.mlx_array,
    media: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    logprobs_n: u32,
    /// Echo: score the prompt's own tokens too (`GenerationResult.prompt_logprobs`).
    prompt_logprobs: bool,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// When non-null, the peer socket is probed on idle wakeups during the
    /// wait — a vanished client cancels the slot (aborting its prefill)
    /// instead of grinding out a ghost generation nobody will read.
    conn: ?*Conn,
) !generate_mod.GenerationResult {
    var slot = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = full_prompt,
        .cached_tokens = cached_tokens,
        .has_tools = has_tools,
        .enable_thinking = enable_thinking,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = timeout_ns,
        .enable_pld = enable_pld,
        .enable_drafter = enable_drafter and (lm.drafter != null or lm.dflash != null),
        .drafter = if (enable_drafter) lm.drafter else null,
        .dflash = if (enable_drafter) lm.dflash else null,
        .drafter_block_size = lm.drafter_block_size,
        .pld_draft_len = server_config.default_pld_draft_len,
        .pld_key_len = server_config.default_pld_key_len,
        .kv_attn_fused = resolveKvAttnFused(lm.config.?, kv_attn_explicit, prompt_ids.len, kv_quant_override),
        .vision_embeddings = vision_embeddings,
        .media = media,
        .cache_key = cache_key,
        .client = if (conn) |c| c.client else .other,
        .logprobs_n = logprobs_n,
        .prompt_logprobs = prompt_logprobs,
        .kv_quant_config = kv_quant_override,
    });
    defer sch.complete(slot);

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var client_gone = false;
    wait: while (true) {
        const nr = slot.waitNextTimeout(Conn.STREAM_KEEPALIVE_MS) orelse {
            // Idle (long prefill). If the client is gone, cancel and serve
            // whatever accumulated — the response write will fail upstream,
            // which is fine; the win is freeing the GPU.
            if (conn) |c| {
                if (c.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting (non-stream) — cancelling slot\n", .{});
                    slot.cancel();
                    client_gone = true;
                    break :wait;
                }
            }
            continue :wait;
        };
        switch (nr) {
            .token => |t| try output_ids.append(allocator, t),
            .done => break :wait,
            .err => return slotFailure(slot),
        }
        if (conn) |c| {
            if (c.peerClosed()) {
                log.info("  [cancel] client disconnected while decoding (non-stream) — cancelling slot\n", .{});
                slot.cancel();
                client_gone = true;
                break :wait;
            }
        }
    }

    // The scheduler measures prefill_ns / decode_ns per-slot directly. Pull
    // those instead of the old single-wall-clock approximation, which
    // double-counted total time into both phases.
    const prefill_tps = generate_mod.prefillTokensPerSec(slot.prompt_tokens, slot.cached_tokens, slot.prefill_ns);
    const decode_tps = generate_mod.tokensPerSec(slot.completion_tokens, slot.decode_ns);

    const strip_leading = tok.tok_type == .sentencepiece_bpe and (sampling.constraint == null or sampling.constraint.?.proto == null);
    // A loop cut is a truncation, and the degenerate span is the part the
    // client must not get: an agent re-sends the cut turn as history, the
    // model reads its own loop back and resumes it — five loop-stops in a
    // row, each firing sooner than the last (live 2026-08-05, under pi).
    // Only reachable non-streaming: a delta cannot be retracted, so a
    // streaming client has already received the tail.
    const emit_ids = loopTrimmedIds(output_ids.items, slot.loop_trim_start);
    const text = try decodeTokens(allocator, lm, tok, emit_ids, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    // Convert the generator's (token index, byte offset) payload boundary
    // into a byte offset inside `text`. The prefix decode uses the same
    // leading-space convention as the full text, so offsets are exact.
    var constraint_payload_byte: ?usize = null;
    if (slot.constraintPayloadStart()) |ps| {
        var pb: ?usize = ps.byte_offset;
        if (ps.token_index > 0) {
            if (ps.token_index <= emit_ids.len) {
                const prefix = decodeTokens(allocator, lm, tok, emit_ids[0..ps.token_index], strip_leading) catch null;
                if (prefix) |p| {
                    pb = p.len + ps.byte_offset;
                    allocator.free(p);
                } else pb = null;
            } else pb = null;
        }
        if (pb) |b| {
            if (b <= text.len) constraint_payload_byte = b;
        }
    }

    // Phase A5: take ownership of the slot's accumulated logprobs. After
    // `toOwnedSlice` the slot's list is empty, so `Slot.deinit` doesn't try
    // to free what we just transferred to the caller.
    const logprobs_slice: ?[]generate_mod.LogprobResult = if (slot.logprobs_buf.items.len > 0)
        slot.logprobs_buf.toOwnedSlice(slot.allocator) catch null
    else
        null;
    const prompt_logprobs_slice: ?[]generate_mod.LogprobResult = if (slot.prompt_logprobs_buf.items.len > 0)
        slot.prompt_logprobs_buf.toOwnedSlice(slot.allocator) catch null
    else
        null;

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = slot.prompt_tokens,
        .completion_tokens = slot.completion_tokens,
        .finish_reason = nonStreamFinishReason(client_gone, slot.finish_reason),
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .prefill_ns = slot.prefill_ns,
        .decode_ns = slot.decode_ns,
        .cached_tokens = slot.cached_tokens,
        .logprobs = logprobs_slice,
        .prompt_logprobs = prompt_logprobs_slice,
        .finish_details = slot.finish_details,
        .constraint_payload_byte = constraint_payload_byte,
    };
}

/// Generation is authoritative for constrained responses: when the generator
/// published a payload boundary, the split is a byte cut at that offset —
/// reasoning is everything before it (delimiter markup stripped, whitespace
/// policy matched to the streaming suppression), content is everything after,
/// UNREParsed. A standalone re-scan of the payload would read marker text
/// inside JSON string data ("the close delimiter") as protocol structure and
/// truncate valid JSON. Without a boundary the existing parser applies.
fn emitConstrainedChat(a: std.mem.Allocator, stream: *Conn, id: i64, model: []const u8, lps: *StreamLogprobs, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = null, .reasoning_content = d.reasoning.items }, null, null, null, .{});
    }
    if (d.content.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = d.content.items }, null, null, null, .{ .logprobs_json = try lps.take() });
    } else lps.dropPending();
}

fn splitConstrainedResponse(allocator: std.mem.Allocator, routed: *?rp_mod.Delivery, sampling: generate_mod.SamplingParams, text: []const u8, payload_byte: ?usize, keep_markup: bool, opened: bool) !chat_mod.ThinkSplit {
    if (sampling.constraint) |c| {
        if (c.proto) |p| {
            routed.* = rp_mod.Delivery.init(p);
            const d = &routed.*.?;
            d.payload_byte = payload_byte;
            try d.feed(allocator, text);
            try d.finish(allocator);
            const reasoning = std.mem.trim(u8, d.reasoning.items, "\n ");
            return .{ .reasoning_content = if (reasoning.len == 0) null else reasoning, .content = d.content.items };
        }
    }
    return if (keep_markup) chat_mod.splitThinkBlockKeepingMarkup(text, true, opened) else chat_mod.splitThinkBlock(text, true, opened);
}

fn handleNonStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_pld: bool,
    enable_drafter: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// Iteration 1: tokenize_ns from the parent handleChatCompletions, so
    /// the non-streaming chat response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
) !void {
    // Phase A8: this handler owns the vision array on entry; ownership
    // transfers to the slot on submit (the scheduler's `Slot.deinit` frees
    // it). Nulled before transfer so the early-return defer is a no-op.
    var ve_local = vision_embeddings;
    defer {
        if (ve_local) |arr| _ = mlx.mlx_array_free(arr);
    }

    var timer = Stopwatch.init(stream.io);

    // Speculative-decoding dispatch (priority: drafter > PLD > regular).
    //   1. Drafter wins if loaded AND requested AND no logprobs (drafter
    //      cannot expose realized log-probs from the draft side).
    //   2. PLD next if requested AND no logprobs AND no grammar constraint
    //      (constrained decode requires per-token state advancement).
    //   3. Otherwise the regular pipeline.
    const spec = requestSpecModes(enable_pld, enable_drafter, lm.drafter != null, lm.dflash != null, sampling.shapesLogits(), logprobs_n);
    const use_drafter = spec.use_drafter;
    const use_pld = spec.use_pld;

    // Transfer vision ownership into the slot via the scheduler.
    const slot_ve: ?mlx.mlx_array = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // Propagates to `handleChatCompletions`' one error arm, shared with the streaming twin.
    const result = try nonStreamingViaScheduler(allocator, global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, enable_thinking, use_pld, use_drafter, getTimeoutNs(), slot_ve, media, cache_key, logprobs_n, false, kv_quant_override, kv_attn_explicit, stream);
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer if (result.logprobs) |lps| {
        for (lps) |*lp| allocator.free(lp.top_logprobs);
        allocator.free(lps);
    };

    // Template-opened think block: unclosed output is reasoning.
    // Not ANDed with enable_thinking: generated reasoning is always DELIVERED,
    // never paid-for-and-dropped — thinking-off is enforced on the PROMPT side
    // (chat.noThinkTailSuffix commits the channel), so any reasoning that
    // still shows up here is real work the client gets to see.
    const opens_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;

    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    if (answerStopCut(result.text, result.text.len, stop_sequences, opens_think and !constrained_proto)) |cut| {
        final_text = result.text[0..cut.index];
        finish_reason = "stop";
    }

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels
    // mid-turn). A span-routed constrained response skips this: its payload
    // is authoritative and marker text inside it is DATA.
    var normalized_text: ?[]u8 = null;
    defer if (normalized_text) |n| allocator.free(n);
    if (sampling.constraint == null or sampling.constraint.?.proto == null) {
        normalized_text = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, final_text);
        if (normalized_text) |n| final_text = n;
    }

    // Apply reasoning budget: truncate reasoning by token count
    // For non-streaming, we truncate after generation since we can't interrupt mid-generation
    var budget_truncated_reasoning: ?[]const u8 = null;
    var budget_reasoning_allocated = false;
    defer if (budget_reasoning_allocated) allocator.free(budget_truncated_reasoning.?);

    if (enable_thinking and reasoning_budget >= 0) {
        const think_split = if (has_tools) chat_mod.splitThinkBlock(final_text, true, opens_think) else chat_mod.splitThinkBlockKeepingMarkup(final_text, true, opens_think);
        if (think_split.reasoning_content) |reasoning| {
            // Count tokens in reasoning by encoding it
            const reasoning_ids = try tok.encode(allocator, reasoning);
            defer allocator.free(reasoning_ids);
            if (reasoning_ids.len > @as(usize, @intCast(reasoning_budget))) {
                // Truncate: decode only budget-many tokens
                const budget_usize: usize = @intCast(reasoning_budget);
                const truncated_ids = reasoning_ids[0..budget_usize];
                const truncated_text = try tok.decode(allocator, truncated_ids, false);
                budget_truncated_reasoning = truncated_text;
                budget_reasoning_allocated = true;
                log.info("  reasoning budget truncated ({d}/{d} tokens)\n", .{ budget_usize, reasoning_ids.len });
            }
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Check for tool calls in the output
    if (has_tools and shouldParseToolCalls(result.finish_details)) {
        log.debug("  checking {d} bytes of generated text for tool calls\n", .{final_text.len});
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json, allow_parallel_tools);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tc_perf_buf: [160]u8 = undefined;
            const tc_perf = formatPerfBracket(&tc_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_calls: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tc_perf, tool_calls.len,
            });

            // Build tool_calls JSON array
            var tc_buf = std.ArrayList(u8).empty;
            defer tc_buf.deinit(allocator);
            try tc_buf.appendSlice(allocator, "[");
            for (tool_calls, 0..) |tc, i| {
                if (i > 0) try tc_buf.appendSlice(allocator, ",");
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const escaped_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(escaped_name);
                const escaped_args = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(escaped_args);
                const tc_json = try std.fmt.allocPrint(allocator,
                    \\{{"id":"{s}","type":"function","function":{{"name":{s},"arguments":{s}}}}}
                , .{ tc_id, escaped_name, escaped_args });
                defer allocator.free(tc_json);
                try tc_buf.appendSlice(allocator, tc_json);
            }
            try tc_buf.appendSlice(allocator, "]");

            // Reasoning is delivered whenever the model produced it — the
            // request's thinking flag shaped the prompt, not the delivery.
            const tc_think_split = chat_mod.splitThinkBlock(final_text, true, opens_think);
            var tc_reasoning_json: []const u8 = "";
            var tc_reasoning_allocated = false;
            {
                if (tc_think_split.reasoning_content) |reasoning| {
                    const escaped_reasoning = try jsonEscape(allocator, reasoning);
                    tc_reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
                    allocator.free(escaped_reasoning);
                    tc_reasoning_allocated = true;
                }
            }
            defer if (tc_reasoning_allocated) allocator.free(tc_reasoning_json);

            // The same pre-markup text the stream flushes as a text delta.
            const tc_preamble = visibleToolPreamble(tc_think_split.content);
            const tc_content_esc = jsonEscapeOrEmpty(allocator, tc_preamble);
            defer if (tc_content_esc.owned) allocator.free(tc_content_esc.slice);
            const tc_content_json: []const u8 = if (tc_preamble.len > 0) tc_content_esc.slice else "null";

            const tc_timings = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
            defer allocator.free(tc_timings);
            const tc_timings_field = if (tc_timings.len > 0)
                try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{tc_timings})
            else
                try allocator.alloc(u8, 0);
            defer allocator.free(tc_timings_field);

            const tc_usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, "");
            defer allocator.free(tc_usage_obj);

            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"lesto","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s},"tool_calls":{s}}},"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
            , .{
                nowMs(stream.io),
                nowSecs(stream.io),
                model_name,
                tc_content_json,
                tc_reasoning_json,
                tc_buf.items,
                toolCallFinishReason(finish_reason),
                finishDetailsField(toolCallFinishReason(finish_reason), result.finish_details),
                tc_usage_obj,
                tc_timings_field,
            });
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
            return;
        }
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
    });

    // Split thinking content from response. Always the think-capable split:
    // whatever reasoning was generated ships as reasoning_content, never
    // stripped (tokens we discarded still counted against tok/s).
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, !has_tools, opens_think);
    const content_text = think_split.content;

    const escaped = jsonEscapeOrEmpty(allocator, content_text);
    const escaped_text = escaped.slice;
    defer if (escaped.owned) allocator.free(escaped.slice);

    // Build logprobs JSON if requested. The array describes `message.content`,
    // so it stops at the tokens that survived the think split — see
    // `contentTokenRange`.
    var logprobs_json: []const u8 = "null";
    var logprobs_allocated = false;
    if (result.logprobs) |lps| {
        const n = @min(result.token_ids.len, lps.len);
        const r = contentTokenRange(allocator, tok, result.token_ids[0..n], final_text, content_text);
        const a = @min(r.start, n);
        const b = @min(@max(r.end, a), n);
        logprobs_json = try formatLogprobsObject(allocator, tok, result.token_ids[a..b], lps[a..b]);
        logprobs_allocated = true;
    }
    defer if (logprobs_allocated) allocator.free(logprobs_json);

    // Build reasoning_content whenever reasoning exists — delivery does not
    // key on the request's thinking flag.
    var reasoning_json: []const u8 = "";
    var reasoning_allocated = false;
    var usage_details_json: []const u8 = "";
    var usage_details_allocated = false;
    {
        // Use budget-truncated reasoning if available, otherwise use full reasoning
        const reasoning_text = if (budget_truncated_reasoning) |tr| tr else think_split.reasoning_content;
        if (reasoning_text) |reasoning| {
            const escaped_reasoning = try jsonEscape(allocator, reasoning);
            reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
            allocator.free(escaped_reasoning);
            reasoning_allocated = true;
            // usage.completion_tokens_details.reasoning_tokens (OpenAI/LM Studio
            // parity) so clients can budget visible content separately.
            if (tok.encode(allocator, reasoning)) |rids| {
                defer allocator.free(rids);
                usage_details_json = try std.fmt.allocPrint(allocator, ",\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}", .{rids.len});
                usage_details_allocated = true;
            } else |_| {}
        }
    }
    defer if (reasoning_allocated) allocator.free(reasoning_json);
    defer if (usage_details_allocated) allocator.free(usage_details_json);

    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);

    const usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, usage_details_json);
    defer allocator.free(usage_obj);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"lesto","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s}}},"logprobs":{s},"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
    , .{
        nowMs(stream.io),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        reasoning_json,
        logprobs_json,
        finish_reason,
        finishDetailsField(finish_reason, result.finish_details),
        usage_obj,
        timings_field,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

/// Token-stream adapter that drives the streaming SSE state machine the same
/// way regardless of whether speculative decoding (PLD / drafter) is on. Each
/// `next()` call yields exactly one token (or null on EOS / max_tokens), so
/// the per-token state machine in `handleStreamingGeneration` does not need to
/// know anything about multi-token batches that PLD and drafter emit per step.
///
/// EOS-in-batch behavior matches the non-streaming `generatePld` /
/// `generateDrafter`: the stop token is NOT yielded — the loop just
/// terminates. This keeps tokens leaking past EOS impossible regardless of
/// where in an N-token batch the EOS lands.
///
/// `pld_drop_buf` is a heap-owned pending buffer (PLD can yield up to
/// `1+max_draft_len`=16 tokens per step; drafter yields up to `block_size`).
/// Caller must `deinit`.
const StreamMode = enum { regular, pld, drafter };
/// Source of streamed tokens. Either a legacy Generator (single-slot mlx on
/// the calling thread) or a Scheduler Slot (mlx on the scheduler's inference
/// thread, this thread reads via waitNext). `next()` yields one token id at
/// a time regardless of source. Post-generation stats are surfaced via the
/// `prompt_tokens` / `completion_tokens` / `finish_reason` fields populated
/// by `finalize()`.
const StreamingTokenStream = struct {
    /// Active source. Exactly one of `gen`/`slot` is set per stream.
    gen: ?*Generator = null,
    slot: ?*scheduler_mod.Slot = null,
    mode: StreamMode,
    pld_draft_len: u32 = 0,
    pld_key_len: u32 = 0,
    eos_token_ids: []const u32,
    /// Pending tokens from a multi-token speculative step (legacy path only).
    /// Drained one at a time before the next call to nextPld / nextDrafter.
    pending_buf: std.ArrayList(u32) = .empty,
    pending_idx: usize = 0,
    finished: bool = false,

    /// Stats populated by `finalize()` after the stream ends. Consumers read
    /// these instead of touching the underlying gen/slot directly so the same
    /// post-generation code works for both legacy and scheduler paths.
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    finish_reason: []const u8 = "stop",
    /// Set only by the degenerate-tail guard (scheduler path). Rides the
    /// FINAL chunk beside `finish_reason`; the tokens themselves are already
    /// gone — a delta cannot be retracted, so streaming gets the signal, not
    /// the trim.
    finish_details: ?[]const u8 = null,
    /// Wall-clock ns for prefill / decode (scheduler path only; the legacy
    /// Generator path leaves these at 0, in which case the server omits the
    /// `timings` block from the usage chunk).
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,

    fn init(gen: *Generator, mode: StreamMode, pld_draft_len: u32, pld_key_len: u32, eos: []const u32) StreamingTokenStream {
        return .{
            .gen = gen,
            .mode = mode,
            .pld_draft_len = pld_draft_len,
            .pld_key_len = pld_key_len,
            .eos_token_ids = eos,
        };
    }

    /// Phase A2-streaming: build a token stream backed by a scheduler Slot.
    /// `mode` is recorded for telemetry but doesn't drive next()'s dispatch
    /// (the scheduler's inference thread already routed PLD/drafter inside
    /// runSingleDecodeTick — this side just drains the resulting token ring).
    fn initFromSlot(slot: *scheduler_mod.Slot, mode: StreamMode, eos: []const u32) StreamingTokenStream {
        return .{
            .slot = slot,
            .mode = mode,
            .eos_token_ids = eos,
        };
    }

    fn deinit(self: *StreamingTokenStream, allocator: std.mem.Allocator) void {
        self.pending_buf.deinit(allocator);
    }

    /// Snapshot prompt/completion tokens + finish reason from the underlying
    /// source so callers can read them after the loop exits without having
    /// to know which source was active. Safe to call multiple times; idempotent.
    fn finalize(self: *StreamingTokenStream) void {
        if (self.slot) |s| {
            self.prompt_tokens = s.prompt_tokens;
            self.cached_tokens = s.cached_tokens;
            self.completion_tokens = s.completion_tokens;
            self.finish_reason = s.finish_reason;
            self.finish_details = s.finish_details;
            self.prefill_ns = s.prefill_ns;
            self.decode_ns = s.decode_ns;
        } else if (self.gen) |g| {
            self.prompt_tokens = g.prompt_tokens;
            self.completion_tokens = g.completion_tokens;
            self.finish_reason = g.finish_reason;
        }
    }

    /// The finalized stream as a `GenerationResult`, for surfaces that share
    /// their post-generation tail with the non-streaming path.
    fn generationResult(self: *const StreamingTokenStream, text: []u8, token_ids: []u32, finish_reason: []const u8, constraint_payload_byte: ?usize) generate_mod.GenerationResult {
        return .{
            .text = text,
            .token_ids = token_ids,
            .prompt_tokens = self.prompt_tokens,
            .completion_tokens = self.completion_tokens,
            .finish_reason = finish_reason,
            .prefill_tps = generate_mod.prefillTokensPerSec(self.prompt_tokens, self.cached_tokens, self.prefill_ns),
            .decode_tps = generate_mod.tokensPerSec(self.completion_tokens, self.decode_ns),
            .prefill_ns = self.prefill_ns,
            .decode_ns = self.decode_ns,
            .cached_tokens = self.cached_tokens,
            .finish_details = self.finish_details,
            .constraint_payload_byte = constraint_payload_byte,
        };
    }

    const NextOrIdle = union(enum) { token: u32, done, idle };

    /// `next` with an idle timeout (scheduler path only): returns `.idle`
    /// after `timeout_ms` with no progress so the caller can poll the peer
    /// socket and emit SSE keepalives during long prefills. The local-
    /// generator path computes synchronously and never idles.
    fn nextOrIdle(self: *StreamingTokenStream, allocator: std.mem.Allocator, timeout_ms: i64) !NextOrIdle {
        if (self.slot) |s| {
            if (self.finished) return .done;
            const nr = s.waitNextTimeout(timeout_ms) orelse return .idle;
            switch (nr) {
                .token => |t| return .{ .token = t },
                .done => {
                    self.finished = true;
                    return .done;
                },
                .err => return slotFailure(s),
            }
        }
        if (try self.next(allocator)) |t| return .{ .token = t };
        return .done;
    }

    /// Yield the next decoded token id, or null if generation is complete.
    /// Mirrors the contract of `Generator.next` for the regular path.
    fn next(self: *StreamingTokenStream, allocator: std.mem.Allocator) !?u32 {
        // Scheduler path: drain the slot's output ring one token at a time.
        // The inference thread already handled regular/PLD/drafter dispatch
        // via runSingleDecodeTick; we just consume what it pushed.
        if (self.slot) |s| {
            if (self.finished) return null;
            switch (s.waitNext()) {
                .token => |t| return t,
                .done => {
                    self.finished = true;
                    return null;
                },
                .err => return slotFailure(s),
            }
        }
        // Drain any pending tokens from a previous speculative step FIRST,
        // before honoring `self.finished`. The PLD/drafter branches set
        // `self.finished = true` mid-batch when an EOS lands at index > 0;
        // tokens *before* that EOS were pushed to pending_buf and must still
        // flush before we terminate. Skipping the drain here would silently
        // truncate the last few output tokens (observed on Gemma 4 drafter
        // with `print(sum_two(10, 20))` losing its trailing `))`).
        if (self.pending_idx < self.pending_buf.items.len) {
            const tok = self.pending_buf.items[self.pending_idx];
            self.pending_idx += 1;
            return tok;
        }
        // Reset the pending buffer once drained so the speculative path can
        // refill it.
        if (self.pending_idx > 0) {
            self.pending_buf.clearRetainingCapacity();
            self.pending_idx = 0;
        }

        if (self.finished) return null;

        const gen = self.gen.?;
        switch (self.mode) {
            .regular => return gen.next(allocator),
            .pld => {
                const r = (try gen.nextPld(allocator, self.pld_draft_len, self.pld_key_len)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                // Walk tokens in order, stopping at the first EOS (and not
                // emitting it) — matches `generatePld`.
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                // Append the remaining (non-EOS-prefixed) tokens to pending,
                // stopping at the first EOS in the tail.
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
            .drafter => {
                // Mirror the PLD branch — `nextDrafter` returns the same
                // `{tokens, accepted_tokens}` shape (full accept yields
                // `block_size` tokens, partial accept yields `1+j`). Walk the
                // batch, stop at the first EOS, push the rest into pending.
                const r = (try gen.nextDrafter(allocator)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
        }
    }
};

/// Choose the speculative-decoding mode for a streaming request based on
/// the request flags and the model capabilities. Mirrors the dispatch in
/// `handleNonStreamingGeneration` so streaming and non-streaming pick the
/// same path for the same inputs.
///
/// Priority: drafter > PLD > regular (matches the non-streaming dispatch in
/// `handleNonStreamingGeneration`). The same per-mode disable rules (no
/// logprobs, no grammar constraint, no hybrid recurrent state for drafter) apply at the
/// request-entry parse site; `pickStreamMode` re-enforces them defensively
/// here so a missed gate at the parse site doesn't crash the dispatch.
/// Resolve sidecar admission before the request's logprobs, grammar and yield gates.
fn requestEnableDrafter(explicit: ?std.json.Value, gemma_loaded: bool, dflash_loaded: bool, is_moe: bool) bool {
    const enabled = if (explicit) |v| v == .bool and v.bool else (gemma_loaded and !is_moe) or dflash_loaded;
    if (!enabled or (!gemma_loaded and !dflash_loaded)) return false;
    return true;
}

test "drafter admission: defaults and explicit opt-in cover every sidecar and architecture" {
    const Case = struct {
        gemma: bool,
        dflash: bool,
        moe: bool,
        default: bool,
        explicit: bool,
    };
    const cases = [_]Case{
        .{ .gemma = false, .dflash = false, .moe = false, .default = false, .explicit = false },
        .{ .gemma = false, .dflash = false, .moe = true, .default = false, .explicit = false },
        .{ .gemma = true, .dflash = false, .moe = false, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = false, .moe = true, .default = false, .explicit = true },
        .{ .gemma = false, .dflash = true, .moe = false, .default = true, .explicit = true },
        .{ .gemma = false, .dflash = true, .moe = true, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = true, .moe = false, .default = true, .explicit = true },
        .{ .gemma = true, .dflash = true, .moe = true, .default = true, .explicit = true },
    };
    for (cases) |c| {
        const default = requestEnableDrafter(null, c.gemma, c.dflash, c.moe);
        const explicit = requestEnableDrafter(.{ .bool = true }, c.gemma, c.dflash, c.moe);
        try testing.expectEqual(c.default, default);
        try testing.expectEqual(c.explicit, explicit);
        try testing.expect(!requestEnableDrafter(.{ .bool = false }, c.gemma, c.dflash, c.moe));
        for ([_]bool{ default, explicit }) |enabled| {
            const spec = requestSpecModes(false, enabled, c.gemma, c.dflash, false, 0);
            try testing.expectEqual(enabled, spec.use_drafter);
            try testing.expect(!spec.use_pld);
            try testing.expectEqual(if (enabled) StreamMode.drafter else .regular, pickStreamMode(false, enabled, c.gemma, c.dflash, false, 0));
        }
    }
}

test "drafter admission: present non-boolean overrides disable either sidecar" {
    for ([_][]const u8{ "null", "\"true\"", "1", "1.0", "[]", "{}" }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
        defer parsed.deinit();
        try testing.expect(!requestEnableDrafter(parsed.value, true, false, false));
        try testing.expect(!requestEnableDrafter(parsed.value, false, true, false));
        try testing.expect(!requestEnableDrafter(parsed.value, false, true, true));
    }
}

pub const RequestSpec = struct { use_drafter: bool, use_pld: bool };

/// The ONE request-level spec dispatch, shared by every surface x stream /
/// non-stream. Priority: DFlash and the gemma drafter (one sidecar switch) > PLD,
/// mirroring `scheduler.specInitWiring`. logprobs, a grammar constraint or a penalty
/// (`shaped_logits`) disable every mode: verify compares raw logits.
/// `enable_drafter` arrives already resolved by `requestEnableDrafter` at every surface's parse site.
pub fn requestSpecModes(
    enable_pld: bool,
    enable_drafter: bool,
    gemma_drafter_loaded: bool,
    dflash_loaded: bool,
    shaped_logits: bool,
    logprobs_n: u32,
) RequestSpec {
    const spec_ok = logprobs_n == 0 and !shaped_logits;
    const sidecar = spec_ok and enable_drafter and (gemma_drafter_loaded or dflash_loaded);
    return .{
        .use_drafter = sidecar,
        .use_pld = !sidecar and spec_ok and enable_pld,
    };
}

fn pickStreamMode(
    enable_pld: bool,
    enable_drafter: bool,
    gemma_drafter_loaded: bool,
    dflash_loaded: bool,
    shaped_logits: bool,
    logprobs_n: u32,
) StreamMode {
    const r = requestSpecModes(enable_pld, enable_drafter, gemma_drafter_loaded, dflash_loaded, shaped_logits, logprobs_n);
    if (r.use_drafter) return .drafter;
    if (r.use_pld) return .pld;
    return .regular;
}

test "requestSpecModes: a loaded sidecar outranks PLD, and logprobs or a shaped distribution disable every mode" {
    // A sidecar (dflash or gemma drafter) rides the drafter arm.
    var r = requestSpecModes(true, true, false, true, false, 0);
    try std.testing.expect(r.use_drafter and !r.use_pld);
    try std.testing.expectEqual(StreamMode.drafter, pickStreamMode(true, true, true, true, false, 0));
    r = requestSpecModes(true, true, true, false, false, 0);
    try std.testing.expect(r.use_drafter and !r.use_pld);
    // enable_drafter:false (or the parse-site hybrid veto) hands the round to PLD.
    r = requestSpecModes(true, false, false, true, false, 0);
    try std.testing.expect(!r.use_drafter and r.use_pld);
    // logprobs / grammar disable every spec mode, dflash included.
    r = requestSpecModes(true, true, false, true, false, 3);
    try std.testing.expect(!r.use_drafter and !r.use_pld);
    r = requestSpecModes(true, true, false, true, true, 0);
    try std.testing.expect(!r.use_drafter and !r.use_pld);
    // No sidecar loaded -> PLD; nothing requested -> regular.
    r = requestSpecModes(true, true, false, false, false, 0);
    try std.testing.expect(!r.use_drafter and r.use_pld);
    try std.testing.expectEqual(StreamMode.regular, pickStreamMode(false, true, false, false, false, 0));
}

fn handleStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_pld: bool,
    enable_drafter: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// Iteration 1: tokenize_ns measured by the request handler before
    /// dispatching here. Surfaced via `timings.tokenize_ms` on the final
    /// usage SSE chunk so streaming clients see the same metric as
    /// non-streaming.
    tokenize_ns: u64,
) !void {
    // Vision array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer so
    // the early-return defer is a no-op.
    var ve_local = vision_embeddings;
    defer {
        if (ve_local) |arr| _ = mlx.mlx_array_free(arr);
    }

    const config = lm.config.?;
    const chat_id = nowMs(stream.io);

    // Template-opened think block: unclosed buffered output is
    // reasoning, never content (mirrors the non-streaming split policy).
    // `prompt_opened_think` drops the `enable_thinking` conjunct on purpose:
    // whether the prompt ends inside a think block is a property of the
    // RENDERED BYTES. A template can pre-open `<think>` unconditionally, so with
    // thinking off its reasoning is tag-free prose the gate flushed as the
    // visible answer (live 2026-08-04). Families that gate the opener on
    // enable_thinking render the closed signature and stay false here.
    const prompt_opened_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const opens_think = prompt_opened_think;

    // Pick the speculative-decoding mode (regular / PLD / drafter). The
    // per-token state machine below is driven by `StreamingTokenStream`,
    // which feeds `next` (regular), `nextPld` (1..1+draft_len tokens/step),
    // or `nextDrafter` (1..block_size tokens/step) through the same
    // one-token-at-a-time interface.
    const stream_mode = pickStreamMode(enable_pld, enable_drafter, lm.drafter != null, lm.dflash != null, sampling.shapesLogits(), logprobs_n);
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});

    // Scheduler's inference thread runs prefill + per-tick decode (regular
    // / PLD / drafter) and pushes generated tokens into the slot's output
    // ring; this thread reads via `slot.waitNext` through the
    // `StreamingTokenStream.initFromSlot` adapter.
    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| global_scheduler.?.complete(s);

    // Transfer vision ownership into the slot.
    const slot_ve_s = ve_local;
    ve_local = null;
    const sch = global_scheduler.?;
    slot_handle = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = has_tools,
        .enable_thinking = enable_thinking,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = getTimeoutNs(),
        .enable_pld = stream_mode == .pld,
        .enable_drafter = stream_mode == .drafter,
        .drafter = if (stream_mode == .drafter) lm.drafter else null,
        .dflash = if (stream_mode == .drafter) lm.dflash else null,
        .drafter_block_size = lm.drafter_block_size,
        .pld_draft_len = server_config.default_pld_draft_len,
        .pld_key_len = server_config.default_pld_key_len,
        .kv_attn_fused = resolveKvAttnFused(config, kv_attn_explicit, prompt_ids.len, kv_quant_override),
        .logprobs_n = logprobs_n,
        .vision_embeddings = slot_ve_s,
        .media = media,
        .cache_key = cache_key,
        .client = stream.client,
        .kv_quant_config = kv_quant_override,
    });
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);

    // Send SSE headers (no Content-Length — we stream until done)
    try sendSseHeaders(stream, "chat.completions", SSE_ALLOW_HEADERS_DEFAULT);

    // Logprobs ride the chunks beside their deltas. Requesting them already
    // forced this stream off every speculative path (see `pickStreamMode`), so
    // the cost is paid whether or not they are delivered — dropping them was
    // the worst of both.
    var lps = StreamLogprobs{
        .allocator = allocator,
        .tok = tok,
        .slot = slot_handle,
        .enabled = logprobs_n > 0,
    };
    defer lps.deinit();

    // First chunk: role announcement
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = "assistant", .content = "" }, null, null, null, .{ .logprobs_json = try lps.take() });

    // Buffer for stop sequence and tool call detection
    var text_buf = std.ArrayList(u8).empty;
    var stop_stream: StopStream = .{};
    defer stop_stream.deinit(allocator);
    defer text_buf.deinit(allocator);
    // Memoized marker scan for the think gate. Owned BESIDE text_buf and reset
    // with it — the gate is otherwise O(buffer) per token for as long as a
    // think block is open and unclosed (47.99 us/token at 113 KB).
    var think_scan: chat_mod.ThinkScan = .{};
    // When tools are present, buffer individual token texts for deferred streaming
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;
    var client_gone = false;

    // Buffer for incomplete UTF-8 sequences split across BPE tokens
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;

    // Thinking state for real-time streaming of reasoning_content vs content
    // Supports both <think>...</think> and Gemma 4's <|channel>thought\n...<channel|>
    // Starts true when thinking is enabled (model outputs <think> first) OR
    // the prompt itself opened a think block: generated reasoning is always
    // DELIVERED as reasoning_content, never paid-for-and-dropped. Thinking-off
    // is enforced prompt-side (chat.noThinkTailSuffix commits the channel).
    // A stream starts inside a think block only when the RENDERED PROMPT ends
    // inside one - never because the request asked for thinking. Gemma
    // templates render the opener when thinking is on, so `promptOpensThink`
    // sees it; a template with no opener has a generation prompt that is a bare `<|im_start|>assistant`
    // and its model answers directly, so seeding from the flag routed the whole
    // answer into reasoning_content and left `content` empty (live 2026-08-13).
    // A model that opens the block itself is picked up by `saw_think_open`.
    var channel_armed = false; // the `<|channel>` marker that may open a Gemma 4 thought
    const gated_stream = has_tools;
    // Constrained requests route through the protocol, not the response-side
    // close scan. The stream seeds the think arm while the payload has not
    // begun (a generated opener must strip, not leak), and flips straight to
    // the content arm when the generator reports a DIRECT answer — the whole
    // point is that no reasoning block exists to scan for.
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
    var in_think_block = prompt_opened_think and !constrained_proto;
    var delivery: ?rp_mod.Delivery = if (constrained_proto) rp_mod.Delivery.init(sampling.constraint.?.proto.?) else null;
    defer if (delivery) |*d| d.deinit(allocator);
    var think_closed = false; // a complete think block was already split+emitted this stream
    // Leading whitespace is suppressed until the first visible byte, so the
    // stream reaches the same content bytes as splitThinkBlock's own
    // trimStart (chat.streamContentLead).
    var content_started = false;
    var think_buf = std.ArrayList(u8).empty; // buffer to detect close tag across token boundaries
    defer think_buf.deinit(allocator);
    var skipped_think_open = false; // track if we've skipped the initial think tag
    // Positive evidence the model itself emitted a think OPENER. Distinct
    // from `skipped_think_open`, whose else-branch also fires for "no known
    // opener — the template must have injected one", which is the very case
    // that has to be told apart at the end-of-stream flush below.
    var saw_think_open = false;
    // Bytes of THIS turn's reasoning already streamed. The tools path emits the
    // thought incrementally now, so every later emit site sends the remainder.
    var reasoning_streamed: usize = 0;
    var reasoning_tokens_sent: usize = 0; // reasoning deltas actually emitted (the budget is counted in these)
    var think_tokens: i32 = 0; // count of tokens generated in think block
    var budget_exhausted = false; // true when reasoning budget hit

    // Generate tokens via the adapter — yields one decoded token id per call
    // regardless of whether the underlying decode is regular, PLD, or drafter.
    while (true) {
        // A stop cut resolved on the previous token ends the turn here.
        if (stopped or stop_stream.flushed) break;
        const next_id: ?u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
            .token => |t| t,
            .done => if (stop_stream.holding()) null else break,
            .idle => {
                // No tokens yet (long prefill). Probe the peer: an abandoned
                // request must cancel instead of grinding a ghost prefill
                // (Claude Code retries pile up serially otherwise), and the
                // keepalive stops clients timing out on stream silence.
                if (stream.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                }
                sendStreamKeepalive(stream) catch {
                    log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                };
                continue;
            },
        };
        if (stream.peerClosed()) {
            slot_handle.?.cancel();
            client_gone = true;
            break;
        }
        // Prepend any carried-over bytes from a previous incomplete UTF-8 sequence,
        // then strip any new trailing incomplete bytes into the carry buffer.
        var token_text = if (next_id) |token_id| blk: {
            try lps.note(token_id);
            // Byte tokens come back raw so the carry below can join a multi-byte character.
            const raw_decoded = try tok.decodeStreamPiece(allocator, token_id);
            if (delivery) |*d| d.noteToken(raw_decoded.len, slot_handle.?.constraintPayloadStart());
            // Step 1: prepend carry-over from previous token
            const with_carry = if (utf8_carry_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                @memcpy(combined[utf8_carry_len..], raw_decoded);
                allocator.free(raw_decoded);
                utf8_carry_len = 0;
                break :cc combined;
            } else raw_decoded;

            // Step 2: check for trailing incomplete UTF-8 sequence
            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_len = @intCast(tail);
            }

            // Step 3: if everything was incomplete, skip this iteration
            if (with_carry.len == tail) {
                allocator.free(with_carry);
                continue;
            }

            // Step 4: send the complete prefix. Raw byte tokens can form an invalid sequence
            // (a stray continuation byte): each such byte becomes '?', never invalid JSON text.
            // One byte for one byte, so the logprob and constrained-delivery byte counts, which
            // note each token's raw length, still match what is sent.
            const complete = with_carry[0 .. with_carry.len - tail];
            if (tail > 0 or !std.unicode.utf8ValidateSlice(complete)) {
                const fixed = utf8Replaced(allocator, complete) catch |e| {
                    allocator.free(with_carry);
                    return e;
                };
                allocator.free(with_carry);
                break :blk fixed;
            }

            break :blk with_carry;
        } else try stop_stream.takeHeld(allocator);

        // Accumulate for stop sequence and tool call detection

        // The stop cut is an INDEX, not a token boundary: bytes before the match still go out.
        if (stop_sequences.len > 0) {
            const fed = try stop_stream.feed(allocator, &text_buf, token_text, stop_sequences, prompt_opened_think and !constrained_proto and !think_closed);
            allocator.free(token_text);
            token_text = fed.send;
            if (fed.stop != null) stopped = true;
            if (token_text.len == 0) {
                allocator.free(token_text);
                continue;
            }
        } else if (gated_stream) try text_buf.appendSlice(allocator, token_text);
        lps.sent(token_text.len, utf8_carry_len);

        if (delivery) |*d| {
            defer allocator.free(token_text);
            try d.feed(allocator, token_text);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, &lps, d);
            try beatStreamKeepalive(stream);
            continue;
        }

        if (gated_stream) {
            // Stream tokens until we detect a tool call pattern starting, then buffer.
            // Detection rules live in `chat.streamShouldBufferForTools` — it
            // covers the full `<tool…>` family (including a bare `<tool>`),
            // Gemma 4 `<|tool_call`, raw JSON, plus partial-prefix growth from
            // a single `<` all the way to `<|tool_cal`. Without the partial
            // coverage, a `<tool` single-BPE-token leaks as visible content
            // (a model that emits `<tool` then gets stuck looping the partial).
            try token_texts.append(allocator, token_text);
            // text_buf already updated above

            const buf = text_buf.items;
            const maybe_tool = chat_mod.streamShouldBufferForTools(buf);

            if (!maybe_tool) {
                // No tool call pattern: ask the shared gate (chat.streamThinkGate)
                // whether the buffer is thinking that
                // must be held, a completed think block to split, or visible
                // prose to flush. Hermetically pinned per recorded model family
                // by the format corpus streaming-gate test.
                switch (chat_mod.streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think_scan)) {
                    .hold_thinking => {
                        // Stream the thought AS IT ARRIVES instead of holding the
                        // whole block. Safe here by construction:
                        // `streamShouldBufferForTools` returned false just above,
                        // so the buffer carries no tool markup and no partial
                        // marker at its tail — these bytes are reasoning and
                        // nothing else. Without it a tools request shows NOTHING
                        // until `</think>` (measured 4.4 s on a 2.6B, and it grows
                        // with the thought) while the no-tools path is already
                        // streaming at 0.05 s.
                        //
                        // A reasoning BUDGET is a cap, not a reason to hide the
                        // thought: you never exceed a cap you STOP emitting at.
                        // This was gated on `reasoning_budget < 0`, so a capped
                        // tools request showed nothing for the whole generation
                        // and got one dump at the end — a capped agent session
                        // looked frozen (live 2026-08-14, pi).
                        // Each loop iteration is one model token, so an
                        // iteration that produced fresh reasoning IS one
                        // reasoning token; at the cap the latch stops every
                        // later emitter from shipping the remainder.
                        if (!budget_exhausted) {
                            const so_far = chat_mod.splitThinkBlock(buf, true, opens_think);
                            if (so_far.reasoning_content) |split_rc| {
                                const rc = chat_mod.streamableReasoning(split_rc);
                                if (chat_mod.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = fresh }, null, null, null, .{});
                                    reasoning_streamed = rc.len;
                                    reasoning_tokens_sent += 1;
                                    if (reasoning_budget >= 0 and reasoning_tokens_sent >= @as(usize, @intCast(reasoning_budget))) {
                                        budget_exhausted = true;
                                        log.info("  reasoning budget exhausted ({d}/{d} tokens, streamed)\n", .{ reasoning_tokens_sent, reasoning_budget });
                                    }
                                }
                            }
                        }
                    },
                    .split_think => {
                        // Complete thinking block — split into reasoning + content
                        const split = chat_mod.splitThinkBlock(buf, true, opens_think);
                        // The thought's tokens never reach the client, so their
                        // entries must not ride the content chunk below — the
                        // reasoning emitters above deliberately do not drain.
                        // Empty content means the block closed with nothing
                        // after it: everything buffered so far was reasoning,
                        // and leaving it pending hands it to the NEXT chunk
                        // (measured: 42 entries on a 1-char delta).
                        if (split.content.len > 0) lps.skipToContent(buf, split.content) else lps.dropPending();
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        think_scan.reset();
                        think_closed = true;
                        // Past the cap the tail is exactly what the budget
                        // exists to withhold. This arm applied no budget at all,
                        // so a capped request received the WHOLE thought the
                        // moment the block closed.
                        if (!budget_exhausted) {
                            if (split.reasoning_content) |rc| {
                                if (chat_mod.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = fresh }, null, null, null, .{});
                                }
                            }
                        }
                        // Block done and `text_buf` was just cleared — a thought
                        // re-opened later in the turn starts counting from zero.
                        reasoning_streamed = 0;
                        if (split.content.len > 0) {
                            content_started = true;
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = split.content }, null, null, null, .{ .logprobs_json = try lps.take() });
                        }
                    },
                    .flush_text => {
                        // Freed here, not per item: a failed send returns mid-loop, and the
                        // handler's own defer would free the already-sent items a second time.
                        defer {
                            for (token_texts.items) |tt| allocator.free(tt);
                            token_texts.clearRetainingCapacity();
                        }
                        for (token_texts.items) |tt| {
                            // Skip bare channel/think tags that leak without a full block
                            if (chat_mod.isChannelMarkerToken(tt)) {
                                continue;
                            }
                            const vis = chat_mod.streamContentLead(tt, content_started);
                            // Suppressed lead: nothing pending describes text the
                            // client received, so retire it rather than letting it
                            // ride the next chunk (logprobs.content ↔ content).
                            if (vis.len == 0) {
                                lps.dropPending();
                                continue;
                            }
                            content_started = true;
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis }, null, null, null, .{ .logprobs_json = try lps.take() });
                        }
                    },
                }
            }
            // Otherwise keep buffering — tool call may be in progress
        } else if (in_think_block) {
            // Inside <think> block — stream as reasoning_content with </think>
            // detection. Also covers thinking-off + template-opened (a template that
            // renders `…assistant\n<think>` unconditionally, live 2026-08-04):
            // the block used to be buffered and DROPPED there; now it streams
            // as reasoning_content like any other thought — generated tokens
            // are always delivered.
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            // Skip the initial think tag prefix (<think> or <|channel>thought\n).
            // Many templates (e.g. some Gemma 4 variants) pre-inject
            // the opener into the prompt so the model's first tokens are already
            // INSIDE the thinking block — no opener appears in the streamed text.
            if (!skipped_think_open and (think_buf.items.len >= 7 or chat_mod.cannotOpenThink(think_buf.items))) {
                if (chat_mod.thinkOpenTagLenAt(think_buf.items)) |olen| {
                    // Remove the opener (<think>) and any leading newline.
                    saw_think_open = true;
                    var skip: usize = olen;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    // Gemma 4 think format — switch close tag
                    saw_think_open = true;
                    var skip: usize = 17; // len of "<|channel>thought"
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Buffer is still a partial prefix of `<|channel>thought` —
                    // wait for more tokens before deciding.
                } else if (think_buf.items.len < 32 and chat_mod.endsWithPartialThinkOpen(think_buf.items)) {
                    // Still a growing `<think>` opener -
                    // wait before deciding, or the raw tag leaks as reasoning.
                } else {
                    // Not a known opener — template already injected one.
                    // Stay inside the think block; close tag is detected dynamically below.
                    skipped_think_open = true;
                }
            }

            // A spent budget caps DELIVERY only: the tail stays withheld reasoning until the close tag.
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                log.info("  reasoning budget exhausted ({d}/{d} tokens)\n", .{ think_tokens, reasoning_budget });
            }

            // Check for the close tag — accept whichever appears first.
            // Models with templates that pre-inject the opener (
            // some Gemma 4 variants) don't reveal which
            // format they use until the close tag arrives, so look for both
            // families (indexOfThinkCloseTag covers bare and suffixed </think…>).
            const think_match = chat_mod.indexOfThinkCloseTag(think_buf.items, 0);
            const channel_pos = std.mem.indexOf(u8, think_buf.items, "<channel|>");
            const close_match: ?struct { pos: usize, len: usize, is_channel: bool } = blk: {
                if (think_match == null and channel_pos == null) break :blk null;
                if (think_match == null) break :blk .{ .pos = channel_pos.?, .len = "<channel|>".len, .is_channel = true };
                if (channel_pos == null) break :blk .{ .pos = think_match.?.pos, .len = think_match.?.len, .is_channel = false };
                if (think_match.?.pos <= channel_pos.?) break :blk .{ .pos = think_match.?.pos, .len = think_match.?.len, .is_channel = false };
                break :blk .{ .pos = channel_pos.?, .len = "<channel|>".len, .is_channel = true };
            };

            if (close_match) |m| {
                if (m.pos > 0 and !budget_exhausted) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..m.pos] }, null, null, null, .{});
                }
                const after = m.pos + m.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                // Strip Gemma 4 content channel tag: <|channel>\n or <|channel>
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                    content_after = content_after[11..];
                } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                    content_after = content_after[10..];
                }
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    // The reasoning above this point never reaches the client,
                    // so its entries stop here rather than riding the answer.
                    lps.skipToContent(think_buf.items, content_after);
                    const vis_content_after = chat_mod.streamContentLead(content_after, content_started);
                    if (vis_content_after.len > 0) {
                        content_started = true;
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_content_after }, null, null, null, .{ .logprobs_json = try lps.take() });
                    }
                } else {
                    lps.dropPending();
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
            } else if (skipped_think_open) {
                // Flush reasoning tokens that can't be part of a close tag.
                // Hold back a still-growing partial close tag (a suffixed form
                // included), then back the cut off
                // to a UTF-8 boundary — a flush cut mid-codepoint ships a lone
                // continuation byte in the delta JSON (live 2026-07-14:
                // `"reasoning_content":"2\xc2"` — '²' split across deltas).
                var safe_len = think_buf.items.len - chat_mod.partialThinkCloseSuffixLen(think_buf.items);
                while (safe_len > 0 and safe_len < think_buf.items.len and (think_buf.items[safe_len] & 0xC0) == 0x80) safe_len -= 1;
                if (safe_len > 0) {
                    if (!budget_exhausted) {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..safe_len] }, null, null, null, .{});
                    }
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            defer allocator.free(token_text);
            // Skip Gemma 4 channel tags that leak after thinking blocks
            // A think block the MODEL opens: seeded like a template-injected one.
            // Checked before the marker skip, which would swallow a bare `<think>`.
            if (!content_started) if (chat_mod.modelThinkOpener(channel_armed, token_text)) |opener| {
                channel_armed = false;
                in_think_block = true;
                saw_think_open = true;
                try think_buf.appendSlice(allocator, opener);
                continue;
            };
            if (chat_mod.isChannelMarkerToken(token_text)) {
                channel_armed = !content_started and std.mem.eql(u8, token_text, "<|channel>");
                continue;
            }
            channel_armed = false;
            const vis_token_text = chat_mod.streamContentLead(token_text, content_started);
            if (vis_token_text.len == 0) {
                lps.dropPending();
            } else {
                content_started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_token_text }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
        }

        // Every branch above may have written NOTHING (tool-call detection and
        // unclosed thinking blocks buffer silently, sometimes for minutes on a
        // one-shot whole-file `write_file`). Tokens flowing is not bytes
        // flowing — without this the client's idle-body timeout kills the
        // stream mid-generation.
        beatStreamKeepalive(stream) catch {
            log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
            slot_handle.?.cancel();
            client_gone = true;
            break;
        };
    }

    if (!client_gone) {
        if (delivery) |*d| {
            d.clearOutput();
            try d.finish(allocator);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, &lps, d);
        }
    }

    // Flush any remaining think buffer
    if (!client_gone and think_buf.items.len > 0) {
        // No close tag arrived. Reasoning ONLY with positive evidence a block
        // was open — the prompt injected the opener, or the model emitted one.
        // `in_think_block` alone is not that evidence: it STARTS true whenever
        // thinking was requested, so a Gemma turn answered directly (its
        // template renders a bare `<|turn>model` and lets the model decide)
        // shipped its whole answer as reasoning with EMPTY content, while
        // non-streaming returned it correctly (live 2026-08-04).
        if (chat_mod.streamTailIsReasoning(in_think_block, prompt_opened_think, saw_think_open)) {
            if (!budget_exhausted) {
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items }, null, null, null, .{});
            }
        } else {
            const vis_tail = chat_mod.streamContentLead(think_buf.items, content_started);
            if (vis_tail.len > 0) {
                content_started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_tail }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
        }
    }

    // After generation: capture stats from whichever source was active
    // (Generator on legacy, Slot on scheduler).
    ts.finalize();

    // Check for tool calls in accumulated text
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    if (gated_stream and !client_gone) {
        log.debug("  checking {d} bytes of streamed text for tool calls\n", .{text_buf.items.len});
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            // Corpus-harvest aid: the inline dump caps at 4KB, useless for
            // >30KB mega-tool-calls. Set MLX_SERVE_RAW_DUMP_FILE=<abs path> to
            // APPEND the FULL pre-parse buffer of every tools request, framed so
            // a harvester can slice each record exactly (the text is arbitrary
            // bytes, so the byte count — not a delimiter — defines the record).
            if (std.c.getenv("MLX_SERVE_RAW_DUMP_FILE")) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B tail behavior).
        const norm_owned = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, text_buf.items);
        defer if (norm_owned) |n| allocator.free(n);
        const gen_text: []const u8 = norm_owned orelse text_buf.items;
        const found_calls = if (has_tools and shouldParseToolCalls(ts.finish_details)) try parseToolCallsForRequest(allocator, gen_text, tools_json, allow_parallel_tools) else null;
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            // Emit reasoning_content before tool calls — whatever the model
            // generated is delivered, thinking flag or not.
            {
                const think_split = chat_mod.splitThinkBlock(gen_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null, null, .{});
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null, null, .{});
                    }
                }
            }

            // Emit tool call deltas in OpenAI streaming format
            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ chat_id, i });
                defer allocator.free(tc_id);
                const first_delta = try streamToolCallDelta(allocator, i, tc_id, tc.name, tc.arguments);
                defer allocator.free(first_delta);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls found — flush buffered tokens, splitting thinking
            // from content. Runs regardless of the request's thinking flag:
            // reasoning the model produced ships as reasoning_content, and
            // split.content already passes trimLeakedToolMarkup, so unparsed
            // wreckage that held the buffer (markers the tokenizer split
            // across tokens) is still cut once, at the end.
            {
                // Concatenate all buffered tokens and split thinking from content
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| {
                    try full_text.appendSlice(allocator, t);
                }
                const flush_norm = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = chat_mod.splitThinkBlock(flush_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (chat_mod.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null, null, .{});
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null, null, .{});
                    }
                }
                if (think_split.content.len > 0) {
                    // Locate against the RAW concatenation: it is always a true
                    // suffix of the generation, where `flush_text` may be a
                    // normalized rewrite. If the content cannot be found there
                    // (normalization moved it) nothing is skipped, which is the
                    // old behaviour rather than a wrong boundary.
                    lps.skipToContent(full_text.items, think_split.content);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_split.content }, null, null, null, .{ .logprobs_json = try lps.take() });
                } else {
                    // All reasoning, no content: those entries describe text the
                    // client never received.
                    lps.dropPending();
                }
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    if (!client_gone) {
        // Final chunk with finish_reason
        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, .{ .reason = finish_reason, .details = ts.finish_details }, null, null, .{ .logprobs_json = try lps.take() });

        // Usage chunk (if requested via stream_options.include_usage). Scheduler
        // accounts for any prompt-cache hits in `ts.prompt_tokens` directly.
        // OpenAI ships this chunk with an EMPTY choices array — restating
        // finish_reason/finish_details here made every per-event client render
        // the ending twice (PR #147's doubled truncation banner). Pending
        // logprobs all drained on the final chunk above.
        if (include_usage) {
            const usage_json = try formatChatUsage(allocator, total_prompt, ts.completion_tokens, ts.cached_tokens, "");
            defer allocator.free(usage_json);
            const timings_obj = try formatTimingsObject(allocator, total_prompt, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns, tokenize_ns);
            defer allocator.free(timings_obj);
            const timings_opt: ?[]const u8 = if (timings_obj.len > 0) timings_obj else null;
            try sendSSEUsageChunk(allocator, stream, chat_id, model_name, usage_json, timings_opt);
        }

        // Done sentinel
        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    // Per-slot timings come from the scheduler (ts.prefill_ns / ts.decode_ns,
    // populated in finalize); the bracket reports compute + any prefix reuse.
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, finish_reason,
    });
}

const DeltaFields = struct {
    role: ?[]const u8,
    content: ?[]const u8,
    reasoning_content: ?[]const u8 = null,
    tool_calls_json: ?[]const u8 = null,
};

/// Per-choice fields that sit BESIDE `delta`, not inside it. Defaulted so a
/// call site that has nothing to add passes `.{}` — every existing emitter
/// keeps its exact bytes, which is what makes this additive.
const ChunkExtras = struct {
    /// Already-rendered `{"content":[…]}` from `formatLogprobsObject`, or null
    /// to omit the key entirely.
    logprobs_json: ?[]const u8 = null,
};

/// Streaming-side logprobs accumulator.
///
/// The generator produces one entry per token; the SSE side emits chunks on
/// its own cadence — the think gate and tool detection buffer many tokens into
/// one delta, and some tokens produce no chunk at all. So entries are drained
/// against a high-water mark rather than paired 1:1 with chunks, and each is
/// shipped EXACTLY once: a delta cannot be retracted, so a re-send is as wrong
/// as a drop.
///
/// Entries are BORROWED from the slot, which owns each `top_logprobs`
/// allocation and frees it in `Slot.deinit`. Nothing here frees them.
const StreamLogprobs = struct {
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    slot: ?*scheduler_mod.Slot,
    enabled: bool,
    /// High-water mark in the slot's buffer (what we have copied out).
    cursor: usize = 0,
    /// How many entries have already ridden a chunk.
    emitted: usize = 0,
    ids: std.ArrayList(u32) = .empty,
    /// Decoded byte length of each noted token, so a byte offset in the
    /// generated text converts to a token index WITHOUT re-decoding and without
    /// depending on how far entry publication has lagged behind `ids`.
    lens: std.ArrayList(usize) = .empty,
    /// Total decoded bytes noted so far — the generation's length.
    bytes_noted: usize = 0,
    /// Decoded bytes the stream has SENT: a stop-string hold keeps a token's
    /// bytes back and a stop cut drops them, and the entry goes with the bytes
    /// (`sentTokens`). A trailing incomplete UTF-8 sequence (`carry`) is the
    /// stream's own buffering, not a hold, so its token still counts.
    bytes_sent: usize = 0,
    carry: usize = 0,
    entries: std.ArrayList(generate_mod.LogprobResult) = .empty,
    /// The last rendered JSON, owned here and freed on the next drain — the
    /// caller passes it straight into a chunk and never sees the lifetime.
    rendered: ?[]const u8 = null,
    /// `/v1/completions` takes the LEGACY shape (four parallel arrays) rather
    /// than chat's list, and its `text_offset` indexes the whole completion —
    /// so the running base has to survive across chunks.
    legacy: bool = false,
    text_offset: usize = 0,

    fn deinit(self: *StreamLogprobs) void {
        if (self.rendered) |r| self.allocator.free(r);
        self.ids.deinit(self.allocator);
        self.lens.deinit(self.allocator);
        self.entries.deinit(self.allocator);
    }

    /// Record the id of a token the loop just consumed. The id is what renders
    /// the `token`/`bytes` fields; the logprob values arrive separately.
    fn note(self: *StreamLogprobs, id: u32) !void {
        if (!self.enabled) return;
        try self.ids.append(self.allocator, id);
        // The stream's own decode, so a byte token counts the one byte the stream sends.
        const text = self.tok.decodeStreamPiece(self.allocator, id) catch "";
        defer if (text.len > 0) self.allocator.free(text);
        try self.lens.append(self.allocator, text.len);
        self.bytes_noted += text.len;
    }

    fn sent(self: *StreamLogprobs, n: usize, carry: usize) void {
        self.bytes_sent += n;
        self.carry = carry;
    }

    /// Drop the entries for tokens whose bytes never reach the client.
    ///
    /// `logprobs.content` describes `message.content`, and the think gate cuts
    /// the reasoning block out of the stream — so without this the entries
    /// shipped alongside the answer were the model's *thinking*. `tail` is the
    /// gate's buffer (always a suffix of the generation, since it is only ever
    /// cleared wholesale) and `content` the part of it that survives.
    ///
    /// Indexes `ids`/`lens`, which are complete, rather than the pending
    /// window — a token whose logprob has not been published yet still has a
    /// length, so a lagging publisher cannot skew the boundary.
    fn skipToContent(self: *StreamLogprobs, tail: []const u8, content: []const u8) void {
        if (!self.enabled or content.len == 0) return;
        const base = @intFromPtr(tail.ptr);
        const cptr = @intFromPtr(content.ptr);
        const off_in_tail = if (cptr >= base and cptr + content.len <= base + tail.len)
            cptr - base
        else
            std.mem.indexOf(u8, tail, content) orelse return;
        if (self.bytes_noted < tail.len) return;
        const abs = self.bytes_noted - tail.len + off_in_tail;

        var off: usize = 0;
        for (self.lens.items, 0..) |len, i| {
            // A token straddling the boundary put bytes into content, so it is
            // the first content token rather than the last reasoning one.
            if (off + len > abs) {
                if (i > self.emitted) self.emitted = i;
                return;
            }
            off += len;
        }
        self.emitted = self.ids.items.len;
    }

    /// Retire every pending entry unshipped — the turn produced no content, so
    /// nothing pending describes text the client received.
    fn dropPending(self: *StreamLogprobs) void {
        if (!self.enabled) return;
        self.emitted = self.ids.items.len;
    }

    /// Render everything new, or null when logprobs weren't requested or
    /// nothing has landed since the last chunk.
    fn take(self: *StreamLogprobs) !?[]const u8 {
        if (!self.enabled) return null;
        if (self.rendered) |r| {
            self.allocator.free(r);
            self.rendered = null;
        }
        const slot = self.slot orelse return null;
        self.cursor = try slot.copyLogprobsFrom(self.allocator, self.cursor, &self.entries);
        // An id with no entry yet (or the reverse) is a partially published
        // token — hold it for the next chunk rather than shipping a half pair;
        // so is one whose bytes a stop-string hold has not released.
        const avail = @min(@min(self.ids.items.len, self.entries.items.len), sentTokens(self.lens.items, self.bytes_sent + self.carry));
        if (avail <= self.emitted) return null;
        const ids = self.ids.items[self.emitted..avail];
        const lps = self.entries.items[self.emitted..avail];
        const json = if (self.legacy)
            try formatCompletionsLogprobs(self.allocator, self.tok, ids, lps, false, &self.text_offset)
        else
            try formatLogprobsObject(self.allocator, self.tok, ids, lps);
        self.emitted = avail;
        self.rendered = json;
        return json;
    }
};

/// How many leading tokens of `lens` are wholly inside the first `bytes_sent` bytes.
fn sentTokens(lens: []const usize, bytes_sent: usize) usize {
    var acc: usize = 0;
    for (lens, 0..) |len, i| {
        if (acc + len > bytes_sent) return i;
        acc += len;
    }
    return lens.len;
}

test "stream logprobs: a token whose bytes a stop hold keeps back waits for its chunk" {
    const lens = [_]usize{ 2, 3, 0, 4 };
    try std.testing.expectEqual(@as(usize, 0), sentTokens(&lens, 1));
    try std.testing.expectEqual(@as(usize, 1), sentTokens(&lens, 2));
    try std.testing.expectEqual(@as(usize, 1), sentTokens(&lens, 4));
    try std.testing.expectEqual(@as(usize, 3), sentTokens(&lens, 5));
    try std.testing.expectEqual(@as(usize, 4), sentTokens(&lens, 9));
    try std.testing.expectEqual(@as(usize, 4), sentTokens(&lens, 50));
}

/// How a stream ended: the wire `finish_reason` plus, when the
/// degenerate-tail guard cut it, the sibling cause (`finishDetailsField`).
const Finish = struct {
    reason: []const u8,
    details: ?[]const u8 = null,
};

fn sendSSEChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    delta: DeltaFields,
    /// `null` on every mid-stream delta; the two FINAL chunks pass the reason
    /// and, when the degenerate-tail guard cut the turn, its cause. Modelled
    /// as one value so a site cannot report the reason and forget the cause.
    finish: ?Finish,
    usage_json: ?[]const u8,
    timings_json: ?[]const u8,
    extras: ChunkExtras,
) !void {
    // Build the delta JSON object
    var delta_buf = std.ArrayList(u8).empty;
    defer delta_buf.deinit(allocator);

    try delta_buf.appendSlice(allocator, "{");
    var need_comma = false;

    if (delta.role) |role| {
        try delta_buf.appendSlice(allocator, "\"role\":\"");
        try delta_buf.appendSlice(allocator, role);
        try delta_buf.appendSlice(allocator, "\"");
        need_comma = true;
    }

    if (delta.content) |content| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"content\":");
        const escaped = try jsonEscape(allocator, content);
        defer allocator.free(escaped);
        try delta_buf.appendSlice(allocator, escaped);
        need_comma = true;
    }

    if (delta.reasoning_content) |reasoning| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"reasoning_content\":");
        const escaped_r = try jsonEscape(allocator, reasoning);
        defer allocator.free(escaped_r);
        try delta_buf.appendSlice(allocator, escaped_r);
        need_comma = true;
    }

    if (delta.tool_calls_json) |tc_json| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"tool_calls\":");
        try delta_buf.appendSlice(allocator, tc_json);
    }

    try delta_buf.appendSlice(allocator, "}");

    // Build the finish_reason field (+ its sibling cause, when there is one)
    var fr_buf: [64]u8 = undefined;
    const fr_str = if (finish) |f|
        std.fmt.bufPrint(&fr_buf, "\"{s}\"", .{f.reason}) catch "null"
    else
        "null";
    const fd_str = if (finish) |f| finishDetailsField(f.reason, f.details) else "";

    // `logprobs` is a sibling of `delta` on the choice, not a field inside it.
    // Emitted only when the request asked; OpenAI's own chunk shape carries
    // `"logprobs":null` otherwise, which is what the empty case renders.
    var lp_buf = std.ArrayList(u8).empty;
    defer lp_buf.deinit(allocator);
    if (extras.logprobs_json) |lp| {
        try lp_buf.appendSlice(allocator, ",\"logprobs\":");
        try lp_buf.appendSlice(allocator, lp);
    }

    // Build usage field
    const usage_str = if (usage_json) |u| u else "null";

    // Optional `timings` tail. Inserted as a top-level field next to `usage`
    // — matches the llama.cpp shape so clients that key off it work as-is.
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }

    // Build the full SSE chunk
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"lesto","choices":[{{"index":0,"delta":{s},"finish_reason":{s}{s}{s}}}],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, delta_buf.items, fr_str, fd_str, lp_buf.items, usage_str, timings_tail_buf.items });
    defer allocator.free(chunk);

    // Write as SSE event
    logHttpSseData(chunk);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(chunk);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

/// The `stream_options.include_usage` chunk: OpenAI's shape is an EMPTY
/// `choices` array beside the usage object — no delta, no finish_reason, no
/// logprobs (all per-choice fields; the final chunk already carried them).
fn sendSSEUsageChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    usage_json: []const u8,
    timings_json: ?[]const u8,
) !void {
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"lesto","choices":[],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, usage_json, timings_tail_buf.items });
    defer allocator.free(chunk);

    logHttpSseData(chunk);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(chunk);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

// ── Shared utilities ──

fn sendResponse(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    logHttpResponse(status, content_type, body);
    try sendResponseFramed(stream, status, content_type, body);
}

/// Like `sendResponse` but does NOT dump the response body to the debug log —
/// only a one-line status summary. For high-frequency, self-describing bodies
/// where the full dump is pure noise: the Prometheus scrape (`/metrics`, hit
/// every couple seconds) and the panel's JSON feed (`/metrics.json`, ~1 Hz).
fn sendResponseQuiet(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (log.isDebug()) log.debug("[http] <- {s} {s} body={d}b (scrape; body not logged)\n", .{ status, content_type, body.len });
    try sendResponseFramed(stream, status, content_type, body);
}

fn sendResponseFramed(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n", .{
        status,
        content_type,
        body.len,
    }) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
    if (body.len > 0) try stream.writeAll(body);
}

fn logHttpRequest(method: []const u8, path: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] -> {s} {s} body={d}b\n", .{ method, path, body.len });
    logHttpBody("[http] request body", body);
}

fn logHttpResponse(status: []const u8, content_type: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- {s} {s} body={d}b\n", .{ status, content_type, body.len });
    if (responseBodyLogged(content_type)) logHttpBody("[http] response body", body);
}

/// The path as logged: a query carrying the API key is dropped.
fn logSafePath(raw_path: []const u8) []const u8 {
    if (queryParamValue(raw_path, "api_key") == null and queryParamValue(raw_path, "key") == null) return raw_path;
    return raw_path[0 .. std.mem.indexOfScalar(u8, raw_path, '?') orelse raw_path.len];
}

/// The body as logged: the login form's POST carries the API key.
fn logSafeBody(method: []const u8, path: []const u8, content_type: []const u8, body: []const u8) []const u8 {
    return if (loginFormKey(method, path, content_type, body) != null) "" else body;
}

test "a logged request never carries the API key" {
    try testing.expectEqualStrings("/metrics", logSafePath("/metrics?api_key=s3cret"));
    try testing.expectEqualStrings("/v1/models", logSafePath("/v1/models?x=1&key=s3cret"));
    try testing.expectEqualStrings("/v1/chat/completions?beta=true", logSafePath("/v1/chat/completions?beta=true"));
    const ct = "application/x-www-form-urlencoded";
    try testing.expectEqualStrings("", logSafeBody("POST", "/", ct, "api_key=s3cret"));
    try testing.expectEqualStrings("{\"a\":1}", logSafeBody("POST", "/v1/chat/completions", "application/json", "{\"a\":1}"));
}

test "the default-level log lines never carry the API key either" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.Io.Threaded.global_single_threaded.io(), &root_buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/s.log", .{root});
    try log.openFile(path, 0);
    defer log.closeFile();

    // The 413 for an oversized body (warn).
    var body_buf: [64]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    _ = try readRequestOver("POST /v1/chat/completions?api_key=s3cret HTTP/1.1\r\nContent-Length: 18446744073709551615\r\n\r\n{}", &body_buf, &reply_buf);

    const a = testing.allocator;
    const logged = try tmp.dir.readFileAlloc(std.Io.Threaded.global_single_threaded.io(), "s.log", a, .limited(1 << 20));
    defer a.free(logged);
    try testing.expect(std.mem.indexOf(u8, logged, "413") != null);
    try testing.expect(std.mem.indexOf(u8, logged, "s3cret") == null);
}

/// The console page is half a megabyte of our own HTML and says nothing about a request;
/// API responses (JSON, text) are what a post-mortem reads.
fn responseBodyLogged(content_type: []const u8) bool {
    return !std.mem.startsWith(u8, content_type, "text/html");
}

test "responseBodyLogged: API bodies are logged, the console page is not" {
    try testing.expect(responseBodyLogged("application/json"));
    try testing.expect(responseBodyLogged("text/plain; charset=utf-8"));
    try testing.expect(!responseBodyLogged("text/html; charset=utf-8"));
}

/// The `Access-Control-Allow-Headers` value each SSE surface advertises.
const SSE_ALLOW_HEADERS_DEFAULT = "Content-Type, Authorization";

/// The one place a `text/event-stream` response head is written. Past this call a generation
/// failure is an SSE `error` event; before it the stream answers like its non-streaming twin.
fn sendSseHeaders(stream: *Conn, kind: []const u8, allow_headers: []const u8) !void {
    try stream.writeAllNoFlush("HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: close\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: ");
    try stream.writeAllNoFlush(allow_headers);
    try stream.writeAll("\r\n\r\n");
    stream.sse_headers_sent = true;
    logHttpStreamStart(kind);
}

fn logHttpStreamStart(kind: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- 200 OK text/event-stream ({s})\n", .{kind});
}

fn logHttpSseData(data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse data={d}b\n", .{data.len});
    logHttpBody("[http] sse data", data);
}

/// Cap on how much of a BINARY body reaches the debug log.
const BODY_LOG_LIMIT = 4096;

/// True when every byte is printable or ordinary whitespace. Multibyte UTF-8
/// (≥0x80) counts as text, so an emoji in a chat body doesn't demote it.
fn bodyIsText(body: []const u8) bool {
    for (body) |c| {
        if (c == '\n' or c == '\r' or c == '\t') continue;
        if (c < 0x20 or c == 0x7f) return false;
    }
    return true;
}

/// Bounded, strictly-printable view of a body, copied into `out`.
fn bodyPreview(out: []u8, body: []const u8) []const u8 {
    const n = @min(@min(body.len, out.len), BODY_LOG_LIMIT);
    for (body[0..n], 0..) |c, i| {
        out[i] = if (c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f)) c else '.';
    }
    return out[0..n];
}

/// A run of base64 this long is an embedded image or clip, not text anyone reads out of a log.
const LOG_BASE64_RUN: usize = 4096;

fn isBase64Byte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '+' or c == '/' or c == '=' or c == '-' or c == '_';
}

/// `body` with every base64 run of `LOG_BASE64_RUN` bytes or more replaced by `<N base64 chars>`.
fn elideLongBase64(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    var out = try std.ArrayList(u8).initCapacity(allocator, @min(body.len, 1 << 20));
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < body.len) {
        var j = i;
        while (j < body.len and isBase64Byte(body[j])) j += 1;
        if (j - i >= LOG_BASE64_RUN) {
            try out.print(allocator, "<{d} base64 chars>", .{j - i});
        } else {
            try out.appendSlice(allocator, body[i..j]);
        }
        if (j < body.len) {
            try out.append(allocator, body[j]);
            j += 1;
        }
        i = j;
    }
    return out.toOwnedSlice(allocator);
}

fn logHttpBody(label: []const u8, body: []const u8) void {
    if (body.len == 0) return;
    // Text bodies go out WHOLE — reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug — minus embedded
    // images: a vision agent's history is megabytes of base64 per request.
    if (bodyIsText(body)) {
        if (body.len > LOG_BASE64_RUN) if (elideLongBase64(std.heap.page_allocator, body)) |shown| {
            defer std.heap.page_allocator.free(shown);
            log.debug("{s} ({d}b):\n{s}\n", .{ label, body.len, shown });
            return;
        } else |_| {};
        log.debug("{s} ({d}b):\n{s}\n", .{ label, body.len, body });
        return;
    }
    // Binary is a different story. A multipart upload
    // wrote raw file bytes into the log, and a body near the request
    // cap blew through the 32 MB rotation — taking the post-mortem file with it.
    var buf: [BODY_LOG_LIMIT]u8 = undefined;
    const preview = bodyPreview(&buf, body);
    log.debug("{s} ({d}b binary, first {d} sanitized):\n{s}…\n", .{ label, body.len, preview.len, preview });
}

test "debug body log: an embedded image is elided and the text around it is kept whole" {
    const gpa = std.testing.allocator;
    var b64: [5000]u8 = undefined;
    @memset(&b64, 'A');
    const body = try std.fmt.allocPrint(gpa, "{{\"role\":\"user\",\"image\":\"data:image/png;base64,{s}\",\"text\":\"keep this\"}}", .{&b64});
    defer gpa.free(body);
    const out = try elideLongBase64(gpa, body);
    defer gpa.free(out);
    try std.testing.expect(out.len < 200);
    try std.testing.expect(std.mem.indexOf(u8, out, "<5000 base64 chars>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "keep this") != null);
    // A short run (a token, a hash, a small id) is ordinary text.
    const short = "{\"id\":\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\"}";
    const kept = try elideLongBase64(gpa, short);
    defer gpa.free(kept);
    try std.testing.expectEqualStrings(short, kept);
}

const GaugeSamplerCtx = struct {
    metrics: *instr.Metrics,
    scheduler: *scheduler_mod.Scheduler,
    /// Dedicated stop flag — set before joining the thread (LIFO defers ensure
    /// this is written before join() blocks). A separate flag (not
    /// scheduler.shutdown) avoids a LIFO deadlock: scheduler.deinit() (which
    /// sets shutdown) is the FIRST-registered defer and runs LAST, after join.
    stop: *std.atomic.Value(bool),
};

/// Sample instantaneous system/queue state into the gauges. Non-blocking
/// syscalls plus a brief scheduler-counter lock. Called by the sampler thread.
/// The live-token gauge reads the scheduler's `inflight_generated_tokens`
/// aggregate (published once per decode tick) — it NEVER touches per-slot
/// fields off-thread, so the inference path stays lock-free per token.
fn sampleGauges(ctx: GaugeSamplerCtx) void {
    // System gauges (non-blocking syscalls).
    ctx.metrics.gpu_utilization_pct.set(@as(u64, metrics.getGpuPct()));
    ctx.metrics.memory_mb.set(@as(u64, metrics.getAppMemFootprintMb()));
    // The two halves of the MLX allocator, so `memory_mb`'s gap has a name.
    var mlx_active: usize = 0;
    var mlx_cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&mlx_active);
    _ = mlx.mlx_get_cache_memory(&mlx_cache);
    ctx.metrics.mlx_active_bytes.set(@as(u64, mlx_active));
    ctx.metrics.mlx_cache_bytes.set(@as(u64, mlx_cache));

    // Request queue depth — brief lock to read two scheduler counters only.
    ctx.scheduler.queue_mu.lockUncancelable(ctx.scheduler.io);
    const running = @as(u64, ctx.scheduler.in_flight);
    const waiting = @as(u64, ctx.scheduler.pending.items.len);
    ctx.scheduler.queue_mu.unlock(ctx.scheduler.io);
    ctx.metrics.requests_running.set(running);
    ctx.metrics.requests_waiting.set(waiting);

    // Real-time throughput source: completed generation tokens plus the tokens
    // generated so far by still-decoding slots (the scheduler publishes that
    // aggregate atomically once per decode tick — race-free, no per-token
    // write). Non-zero even mid-request, unlike the completion-only counter.
    const inflight = ctx.scheduler.inflight_generated_tokens.load(.monotonic);
    ctx.metrics.generation_tokens_live.set(instr.liveGenerationTokens(ctx.metrics, inflight));

    // The other phase. Prompt-token counters and the prefill-time histogram
    // only advance at request COMPLETION, so a long prefill was invisible: GPU
    // pinned, decode reading 0, prefill reading "—". This gauge moves per
    // prefill chunk and returns to 0 the moment the prefill ends.
    ctx.metrics.prefill_tokens_live.set(ctx.scheduler.inflight_prefill_tokens.load(.monotonic));
    ctx.metrics.prefill_tokens_expected.set(ctx.scheduler.inflight_prefill_expected.load(.monotonic));
    ctx.metrics.requests_prefilling.set(ctx.scheduler.requests_prefilling.load(.monotonic));
}

const IdleEvictCtx = struct {
    scheduler: *scheduler_mod.Scheduler,
    /// `--idle-evict-secs`, in ms.
    window_ms: i64,
    /// Dedicated stop flag, for the same reason `GaugeSamplerCtx` has one:
    /// gating on `scheduler.shutdown` would never be observed, because
    /// `scheduler.deinit()` (which sets it) is the FIRST-registered defer and
    /// runs LAST, after this thread's join. Worse here than for the sampler —
    /// a sweep blocked inside `unloadModelIfIdle` is waiting on the inference thread,
    /// and `deinit` only rescues orphaned unload requests AFTER joining it, so
    /// a join placed inside `deinit` would hang forever.
    stop: *std.atomic.Value(bool),
};

/// `--idle-evict-secs`: unload models that have served nothing for the window.
///
/// Goes through `Scheduler.unloadModelIfIdle`, so nothing here touches mlx
/// directly.
fn idleEvictLoop(ctx: IdleEvictCtx) void {
    const sch = ctx.scheduler;
    const tick_ms = scheduler_mod.idleEvictTickMs(ctx.window_ms);
    // Wake every 500 ms to check the stop flag; sweep once per tick interval.
    // `crt.sleepMs` for the same reason as the gauge sampler: a blocking Io timer
    // here could outlive the stop.
    const ticks_per_sweep: u64 = @max(1, @as(u64, @intCast(@divTrunc(tick_ms, 500))));
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        crt.sleepMs(500);
        tick += 1;
        if (tick < ticks_per_sweep) continue;
        tick = 0;
        if (ctx.stop.load(.monotonic)) return;

        // One victim per sweep. `unloadModelIfIdle` blocks on the inference
        // thread, so holding the registry mutex across it would stall every
        // request; the next sweep takes anything else that has gone idle.
        const now_ms = io_util.nowMsMonotonic(sch.io);
        sch.registry.mutex.lockUncancelable(sch.io);
        const victim = sch.registry.pickIdleEvictable(now_ms, ctx.window_ms);
        const idle_ms = if (victim) |v| now_ms - v.last_used_ms.load(.acquire) else 0;
        const bytes = if (victim) |v| v.bytes_resident else 0;
        sch.registry.mutex.unlock(sch.io);

        if (victim) |entry| {
            log.info("[registry] idle-evicting model id={s} ({d:.2} GB resident, idle {d}s)\n", .{
                entry.id,
                @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0),
                @divTrunc(idle_ms, 1000),
            });
            // Age and refcount are re-checked under the mutex inside, so a
            // request arriving since the pick cancels the eviction instead of
            // pinning the model in `.evicting` behind a live stream.
            sch.unloadModelIfIdle(entry.id, ctx.window_ms) catch |err| {
                log.warn("[registry] idle eviction failed for {s}: {t}\n", .{ entry.id, err });
            };
        }
    }
}

fn gaugeSamplerLoop(ctx: GaugeSamplerCtx) void {
    // Wake every 500 ms to check the stop flag; sample system state once per
    // 2-second interval. The 2 s cadence is what makes generation_tokens_live a
    // real-time tok/s source — the panel windows over a few of these samples.
    // `crt.sleepMs`: this thread has no Io handle.
    const SAMPLE_INTERVAL_TICKS: u64 = 4; // 4 × 500 ms = 2 s
    // Sample once immediately so the panel / Grafana show real values from the
    // first scrape instead of zeros for the first interval.
    sampleGauges(ctx);
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        crt.sleepMs(500);
        tick += 1;
        if (tick < SAMPLE_INTERVAL_TICKS) continue;
        tick = 0;
        sampleGauges(ctx);
    }
}

fn findContentLength(headers: []const u8) ?usize {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        const lower = "content-length: ";
        if (line.len >= lower.len) {
            var match = true;
            for (0..lower.len) |j| {
                if (std.ascii.toLower(line[j]) != lower[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return std.fmt.parseInt(usize, std.mem.trim(u8, line[lower.len..], " "), 10) catch null;
            }
        }
    }
    return null;
}

/// Case-insensitive header lookup returning the trimmed VALUE.
fn findHeaderValue(headers: []const u8, comptime name_lower: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        if (line.len <= name_lower.len or line[name_lower.len] != ':') continue;
        var match = true;
        for (0..name_lower.len) |j| {
            if (std.ascii.toLower(line[j]) != name_lower[j]) {
                match = false;
                break;
            }
        }
        if (match) return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
}

// ── API-key auth helpers (used only when --api-key / g_api_key is set) ──

/// True if the connection's peer is a loopback address (127.0.0.0/8, ::1, or an
/// IPv4-mapped-IPv6 loopback). Loopback is the local machine → trusted and
/// exempt from the API key, so the same-host app never needs credentials.
fn ipIsLoopback(addr: std.Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |a4| a4.bytes[0] == 127, // 127.0.0.0/8
        .ip6 => |a6| blk: {
            const b = a6.bytes; // big-endian 16 bytes
            const zeros10: [10]u8 = @splat(0);
            // ::1
            const zeros15: [15]u8 = @splat(0);
            if (std.mem.eql(u8, b[0..15], &zeros15) and b[15] == 1) break :blk true;
            // ::ffff:127.x.x.x  (IPv4-mapped loopback)
            if (std.mem.eql(u8, b[0..10], &zeros10) and b[10] == 0xff and b[11] == 0xff and b[12] == 127) break :blk true;
            break :blk false;
        },
    };
}

/// Whether the API-key auth gate covers this peer at all: a key must be set,
/// and the peer must be non-loopback — unless `--api-key-strict` removed the
/// loopback exemption. Pure so the truth table is hermetically testable.
fn apiKeyGateApplies(key_set: bool, strict: bool, peer_loopback: bool) bool {
    return key_set and (!peer_loopback or strict);
}

test "apiKeyGateApplies: strict removes exactly the loopback exemption" {
    // No key: fully open regardless of strictness or peer.
    try std.testing.expect(!apiKeyGateApplies(false, false, true));
    try std.testing.expect(!apiKeyGateApplies(false, true, false));
    // Key, default: loopback exempt, network gated.
    try std.testing.expect(!apiKeyGateApplies(true, false, true));
    try std.testing.expect(apiKeyGateApplies(true, false, false));
    // Key, strict: everyone gated.
    try std.testing.expect(apiKeyGateApplies(true, true, true));
    try std.testing.expect(apiKeyGateApplies(true, true, false));
}

pub fn peerIsLoopback(conn: *const Conn) bool {
    return ipIsLoopback(conn.stream.socket.address);
}

/// Case-insensitive HTTP header lookup in the raw header block. `name_lower`
/// must be lowercase (e.g. "authorization"). Returns the trimmed value, or null.
fn findHeaderValueCI(headers: []const u8, name_lower: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        if (line.len <= name_lower.len + 1) continue;
        var match = true;
        for (0..name_lower.len) |j| {
            if (std.ascii.toLower(line[j]) != name_lower[j]) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        if (line[name_lower.len] != ':') continue;
        return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
}

/// Constant-time equality — avoids a timing oracle on the API key.
fn constTimeEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| diff |= x ^ y;
    return diff == 0;
}

/// Extract a query-string parameter from the RAW (un-stripped) request path,
/// e.g. `queryParamValue("/metrics?api_key=abc", "api_key") == "abc"`.
fn queryParamValue(raw_path: []const u8, name: []const u8) ?[]const u8 {
    const qpos = std.mem.indexOfScalar(u8, raw_path, '?') orelse return null;
    var it = std.mem.splitScalar(u8, raw_path[qpos + 1 ..], '&');
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        if (std.mem.eql(u8, pair[0..eq], name)) return pair[eq + 1 ..];
    }
    return null;
}

/// True if an `Authorization: Basic base64(user:pass)` value carries the key as
/// its password (browser index/metrics pages send this after the Basic prompt).
/// A user-only entry (no colon) matching the key is also accepted.
fn basicAuthMatches(enc: []const u8, key: []const u8) bool {
    var dbuf: [1024]u8 = undefined;
    const dec_len = std.base64.standard.Decoder.calcSizeForSlice(enc) catch return false;
    if (dec_len > dbuf.len) return false;
    std.base64.standard.Decoder.decode(dbuf[0..dec_len], enc) catch return false;
    const creds = dbuf[0..dec_len];
    const pass = if (std.mem.indexOfScalar(u8, creds, ':')) |c| creds[c + 1 ..] else creds;
    return constTimeEql(pass, key);
}

/// True if the request is authorized for the configured `g_api_key` (or no key
/// is set). Accepts, in order: `Authorization: Bearer <key>`, HTTP Basic (key as
/// the password), `x-api-key: <key>` (Anthropic), and `?api_key=` / `?key=`.
fn apiKeyAuthorized(raw_headers: []const u8, raw_path: []const u8) bool {
    const key = g_api_key orelse return true;
    if (findHeaderValueCI(raw_headers, "authorization")) |auth| {
        if (std.mem.startsWith(u8, auth, "Bearer ")) {
            if (constTimeEql(std.mem.trim(u8, auth["Bearer ".len..], " \t"), key)) return true;
        } else if (std.mem.startsWith(u8, auth, "Basic ")) {
            if (basicAuthMatches(std.mem.trim(u8, auth["Basic ".len..], " \t"), key)) return true;
        }
    }
    if (findHeaderValueCI(raw_headers, "x-api-key")) |xk| {
        if (constTimeEql(xk, key)) return true;
    }
    if (findHeaderValueCI(raw_headers, "cookie")) |c| {
        if (pairValue(c, ';', KEY_COOKIE)) |v| if (encodedKeyMatches(v, key)) return true;
    }
    var qbuf: [512]u8 = undefined;
    inline for (.{ "api_key", "key" }) |name| {
        if (queryDecoded(&qbuf, raw_path, name)) |q| if (constTimeEql(q, key)) return true;
    }
    return false;
}

/// The console's login cookie: the key as the form percent-encoded it. HttpOnly, so page
/// scripts never read it; the browser sends it on the page load and on the console's own calls.
const KEY_COOKIE = "mlx_serve_key";

fn encodedKeyMatches(encoded: []const u8, key: []const u8) bool {
    var buf: [512]u8 = undefined;
    return constTimeEql(percentDecoded(&buf, encoded) orelse return false, key);
}

/// Form encoding's own alphabet: nothing that could end a header or a cookie.
fn cookieSafe(v: []const u8) bool {
    if (v.len == 0 or v.len > 512) return false;
    for (v) |c| if (!std.ascii.isAlphanumeric(c) and std.mem.indexOfScalar(u8, "%+-._*~", c) == null) return false;
    return true;
}

/// The still-encoded `api_key` of the login form's POST, or null when this is not that POST.
fn loginFormKey(method: []const u8, path: []const u8, content_type: []const u8, body: []const u8) ?[]const u8 {
    if (!std.mem.eql(u8, method, "POST") or !std.mem.eql(u8, path, "/")) return null;
    if (!std.mem.startsWith(u8, content_type, "application/x-www-form-urlencoded")) return null;
    const v = pairValue(body, '&', "api_key") orelse return null;
    return if (cookieSafe(v)) v else null;
}

/// Trade an accepted key for the cookie and send the browser to the bare page, so the key
/// never sits in the URL (`./` keeps a reverse proxy's path prefix).
fn sendKeyCookie(stream: *Conn, encoded: []const u8) !void {
    logHttpResponse("303 See Other", "", "");
    var hdr_buf: [1024]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 303 See Other\r\nLocation: ./\r\nSet-Cookie: " ++ KEY_COOKIE ++ "={s}; Path=/; HttpOnly; SameSite=Strict\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{encoded}) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
}

/// A browser opening the page (`GET /` that accepts HTML) without the key gets
/// the login form: a Basic prompt is easy to miss and never returns once cancelled.
fn wantsLoginPage(method: []const u8, path: []const u8, raw_headers: []const u8) bool {
    if (!std.mem.eql(u8, method, "GET") or !std.mem.eql(u8, path, "/")) return false;
    const accept = findHeaderValueCI(raw_headers, "accept") orelse return false;
    return std.mem.indexOf(u8, accept, "text/html") != null;
}

/// The login form, saying so when a key was offered and refused. Script-free:
/// it POSTs the key back to `/` (`loginFormKey`), which answers with the cookie.
fn loginPage(raw_headers: []const u8, raw_path: []const u8) []const u8 {
    const cookie_key = if (findHeaderValueCI(raw_headers, "cookie")) |c| pairValue(c, ';', KEY_COOKIE) else null;
    const tried = findHeaderValueCI(raw_headers, "authorization") != null or
        findHeaderValueCI(raw_headers, "x-api-key") != null or
        cookie_key != null or
        queryParamValue(raw_path, "api_key") != null or
        queryParamValue(raw_path, "key") != null;
    return if (tried) loginPageHtml("<p class=err>That key was not accepted. Try again.</p>") else loginPageHtml("");
}

fn loginPageHtml(comptime note: []const u8) []const u8 {
    return
    \\<!doctype html>
    \\<html lang="en"><head><meta charset="utf-8">
    \\<meta name="viewport" content="width=device-width,initial-scale=1">
    \\<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'">
    \\<title>MLX Serve · API key required</title>
    \\<style>
    \\:root{color-scheme:light dark;font-family:system-ui,sans-serif}
    \\body{margin:0;min-height:100vh;display:grid;place-items:center;background:Canvas;color:CanvasText}
    \\main{width:min(22rem,calc(100vw - 2rem));padding:1.75rem;border:1px solid color-mix(in srgb,CanvasText 15%,transparent);border-radius:.75rem}
    \\h1{font-size:1.15rem;margin:0 0 .5rem}p{margin:0 0 1rem;font-size:.9rem;opacity:.8}
    \\.err{color:#d33;opacity:1}
    \\input,button{box-sizing:border-box;width:100%;font:inherit;padding:.6rem .7rem;border-radius:.5rem}
    \\input{border:1px solid color-mix(in srgb,CanvasText 30%,transparent);background:Canvas;color:CanvasText;margin-bottom:.75rem}
    \\button{border:0;background:#0a6cff;color:#fff;cursor:pointer}
    \\</style></head><body><main>
    \\<h1>API key required</h1>
    \\<p>This MLX Serve server requires an API key. Enter the key the server was started with.</p>
    \\
++ note ++
    \\<form method="post">
    \\<input type="password" name="api_key" placeholder="API key" autocomplete="current-password" required autofocus>
    \\<button type="submit">Continue</button>
    \\</form>
    \\</main></body></html>
    \\
    ;
}

/// Send a 401 with a Basic-auth challenge so browsers prompt for the key on the
/// index + metrics pages; API clients read the JSON error body.
fn sendUnauthorized(stream: *Conn) !void {
    const body = "{\"error\":{\"message\":\"missing or invalid API key\",\"type\":\"authentication_error\"}}";
    logHttpResponse("401 Unauthorized", "application/json", body);
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Basic realm=\"mlx-serve\"\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization, x-api-key\r\n\r\n", .{body.len}) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
    try stream.writeAll(body);
}

test "apiKeyAuthorized accepts Bearer, x-api-key, Basic, and query param" {
    const prev = g_api_key;
    defer g_api_key = prev;
    g_api_key = "s3cret";

    // Bearer
    try std.testing.expect(apiKeyAuthorized("Authorization: Bearer s3cret\r\n", "/v1/chat/completions"));
    // x-api-key
    try std.testing.expect(apiKeyAuthorized("x-api-key: s3cret\r\n", "/v1/chat/completions"));
    // Basic — base64("user:s3cret") = "dXNlcjpzM2NyZXQ="
    try std.testing.expect(apiKeyAuthorized("Authorization: Basic dXNlcjpzM2NyZXQ=\r\n", "/"));
    // Query param (browser fallback)
    try std.testing.expect(apiKeyAuthorized("", "/metrics.json?api_key=s3cret"));
    try std.testing.expect(apiKeyAuthorized("", "/metrics?key=s3cret"));

    // Wrong / missing key is rejected
    try std.testing.expect(!apiKeyAuthorized("Authorization: Bearer wrong\r\n", "/"));
    try std.testing.expect(!apiKeyAuthorized("x-api-key: nope\r\n", "/"));
    try std.testing.expect(!apiKeyAuthorized("", "/v1/models"));

    // The login form percent-encodes the key it sends back (a base64 key has + / =).
    g_api_key = "a+b/c=";
    try std.testing.expect(apiKeyAuthorized("", "/?api_key=a%2Bb%2Fc%3D"));
    try std.testing.expect(!apiKeyAuthorized("", "/?api_key=a%2Bb"));

    // No key configured ⇒ always authorized (open mode)
    g_api_key = null;
    try std.testing.expect(apiKeyAuthorized("", "/v1/chat/completions"));
}

test "a browser opening the page without the key gets a login form; API clients do not" {
    const html = "Accept: text/html,application/xhtml+xml,*/*;q=0.8\r\n";
    try std.testing.expect(wantsLoginPage("GET", "/", html));
    try std.testing.expect(!wantsLoginPage("GET", "/", "Accept: */*\r\n"));
    try std.testing.expect(!wantsLoginPage("GET", "/v1/models", html));
    try std.testing.expect(!wantsLoginPage("POST", "/", html));

    const first = loginPage("", "/");
    try std.testing.expect(std.mem.indexOf(u8, first, "name=\"api_key\"") != null);
    // POST, so the key never lands in the URL, history or a Referer (#887).
    try std.testing.expect(std.mem.indexOf(u8, first, "method=\"post\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, first, "<script") == null);
    try std.testing.expect(std.mem.indexOf(u8, first, "not accepted") == null);
    // A key that was tried and failed says so, whichever way it came.
    try std.testing.expect(std.mem.indexOf(u8, loginPage("", "/?api_key=bad"), "not accepted") != null);
    try std.testing.expect(std.mem.indexOf(u8, loginPage("Authorization: Basic dTpi\r\n", "/"), "not accepted") != null);
}

test "the login form's key comes back as a cookie that authorizes the page and its API calls (#887)" {
    const prev = g_api_key;
    defer g_api_key = prev;
    g_api_key = "a+b/c=";
    // The form body is percent-encoded; its value is what the cookie stores.
    try std.testing.expectEqualStrings("a%2Bb%2Fc%3D", loginFormKey("POST", "/", "application/x-www-form-urlencoded", "api_key=a%2Bb%2Fc%3D").?);
    try std.testing.expect(loginFormKey("GET", "/", "application/x-www-form-urlencoded", "api_key=x") == null);
    try std.testing.expect(loginFormKey("POST", "/v1/models", "application/x-www-form-urlencoded", "api_key=x") == null);
    try std.testing.expect(loginFormKey("POST", "/", "application/json", "{}") == null);
    try std.testing.expect(apiKeyAuthorized("Cookie: theme=dark; " ++ KEY_COOKIE ++ "=a%2Bb%2Fc%3D\r\n", "/v1/models"));
    try std.testing.expect(!apiKeyAuthorized("Cookie: " ++ KEY_COOKIE ++ "=a%2Bb\r\n", "/v1/models"));
    try std.testing.expect(!apiKeyAuthorized("Cookie: x" ++ KEY_COOKIE ++ "=a%2Bb%2Fc%3D\r\n", "/"));
}

test "resolveRequestModelId: an alias names its model; an id beats it" {
    const a = std.testing.allocator;
    const reg = try ModelRegistry.init(a, std.Io.Threaded.global_single_threaded.io(), null, 8, 0, null);
    defer reg.deinit();
    _ = try reg.registerStub("org/Example-27B-4bit", "/m/q", 1);
    _ = try reg.registerStub("example", "/m/other", 1);
    _ = try reg.registerStub("gemma-4-e4b-it-4bit", "/m/g", 1);
    var aliases: model_settings_mod.Cache = .{ .path = "", .settings = try model_settings_mod.parse(a,
        \\{"/m/q/": {"alias": "q"}, "/m/g": {"alias": "example"}, "/m/gone": {"alias": "ghost"}}
    ) };
    defer aliases.deinit();
    try std.testing.expectEqualStrings("org/Example-27B-4bit", resolveRequestModelId(reg, &aliases, "q").id);
    try std.testing.expectEqualStrings("example", resolveRequestModelId(reg, &aliases, "example").id);
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "q:latest") == .unknown_name);
    // An alias for a path nothing registered falls back like any unknown name.
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "ghost") == .unknown_name);
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "gpt-4") == .unknown_name);
}

test "appendAliasedRow: the alias rides the row, the id stays canonical" {
    const a = std.testing.allocator;
    var out = std.ArrayList(u8).empty;
    defer out.deinit(a);
    try appendAliasedRow(a, &out, "{\"id\":\"org/Q\"}", "q");
    try appendAliasedRow(a, &out, "{\"id\":\"g\"}", null);
    try std.testing.expectEqualStrings("{\"alias\":\"q\",\"id\":\"org/Q\"}{\"id\":\"g\"}", out.items);
}

test "resolveRequestModelId: a path names its own entry, never the default model" {
    const a = std.testing.allocator;
    const reg = try ModelRegistry.init(a, std.Io.Threaded.global_single_threaded.io(), null, 8, 0, null);
    defer reg.deinit();
    const good = try reg.registerStub("org/good", "/m/org/good", 1);
    reg.default_id = good.id;
    _ = try reg.registerStub("org/broken", "/m/org/broken", 1);
    var aliases: model_settings_mod.Cache = .{ .path = "" };
    defer aliases.deinit();

    try std.testing.expectEqualStrings("org/broken", resolveRequestModelId(reg, &aliases, "/m/org/broken").id);
    try std.testing.expectEqualStrings("org/broken", resolveRequestModelId(reg, &aliases, "/m/org/broken/").id);
    try std.testing.expectEqualStrings("org/broken", resolveRequestModelId(reg, &aliases, "org/broken").id);
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "/m/org/missing") == .unknown_path);
    // Plain unknown names and the alias keep the default-model fallback.
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "gpt-4") == .unknown_name);
    try std.testing.expectEqualStrings("mlx-serve", resolveRequestModelId(reg, &aliases, "mlx-serve").id);
    try std.testing.expectEqualStrings("lesto", resolveRequestModelId(reg, &aliases, "lesto").id);
    try std.testing.expectEqualStrings("", resolveRequestModelId(reg, &aliases, "").id);
}

test "the route-existence 404 is answered BEFORE the model is resolved" {
    // A path we do not serve must cost nothing. Resolution ran first, so
    // `POST /v1/load` (a route that never existed) with a `model` field
    // cold-loaded that model and THEN 404'd — 2m42s and 121 GB resident for a
    // typo, and a one-line way for any client to pin the box. The existence
    // answer has to precede `ensureLoaded`, not sit in one of its error arms.
    const src = @embedFile("server.zig");
    const gate = "if (!routeExists(" ++ "path)) {";
    const resolve = "scheduler.ensureLoaded(" ++ "requested_model_id)";
    const gate_at = std.mem.indexOf(u8, src, gate) orelse return error.GateMissing;
    const resolve_at = std.mem.indexOf(u8, src, resolve) orelse return error.ResolveMissing;
    try std.testing.expect(gate_at < resolve_at);
    // Exactly one gate — a second copy in an error arm is a second answer to
    // a question that has one.
    try std.testing.expect(std.mem.indexOf(u8, src[gate_at + gate.len ..], gate) == null);
}

test "each connection thread handle is detached after spawn" {
    // A finished pthread keeps its stack mapping and kernel bookkeeping until
    // its handle is joined or detached. The server cannot join per-request
    // threads individually, so dropping this handle leaks one stack region per
    // request even though `handleConnectionThread` has returned.
    const src = @embedFile("server.zig");
    const spawn = "std.Thread.spawn(.{}, handleConnection" ++ "Thread, .{args})";
    const spawn_at = std.mem.indexOf(u8, src, spawn) orelse return error.ConnectionSpawnMissing;
    const accept_end = std.mem.indexOfPos(
        u8,
        src,
        spawn_at,
        "\n    }\n\n    // Shutdown ordering",
    ) orelse return error.AcceptLoopEndMissing;
    const detach = "conn_thread." ++ "detach();";
    try std.testing.expect(std.mem.indexOf(u8, src[spawn_at..accept_end], detach) != null);

    // Class guard: no spawn site anywhere may discard its joinable handle —
    // that is the exact pre-fix shape, and a new worker elsewhere would
    // otherwise re-ship the leak unguarded.
    const sources = [_][]const u8{
        src,
        @embedFile("scheduler.zig"),
        @embedFile("main.zig"),
        @embedFile("metrics.zig"),
    };
    const discard = "_ = " ++ "std.Thread.spawn";
    const discard_try = "_ = try " ++ "std.Thread.spawn";
    for (sources) |s| {
        try std.testing.expect(std.mem.indexOf(u8, s, discard) == null);
        try std.testing.expect(std.mem.indexOf(u8, s, discard_try) == null);
    }
}

test "input_modalities is text only, resident or not, and no vision or audio capability is reported" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const native = "{\"model_type\":\"gemma4\",\"vision_config\":{\"model_type\":\"gemma4_vision\"},\"audio_config\":{\"model_type\":\"gemma4_audio\"}}";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = native });
    var path_buf: [512]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const reg = try model_registry_mod.ModelRegistry.init(a, io, null, 1, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("stub-model", path, 0);
    defer {
        entry.config = null;
        entry.chat_config = null;
        entry.state = .unloaded;
    }
    var video_config = model_mod.ModelConfig{ .video_token_id = 7 };
    var chat_config = chat_mod.ChatConfig{
        .chat_template = "template", .bos_token = null, .eos_token = null,
        .add_bos_token = false, .allocator = a,
    };
    entry.config = &video_config;
    entry.chat_config = &chat_config;
    entry.state = .ready;
    const expect_text_only = struct {
        fn check(alloc: std.mem.Allocator, runtime_io: std.Io, model: *LoadedModel, resident: bool) !void {
            const json = try renderModelEntry(alloc, runtime_io, model);
            defer alloc.free(json);
            // Neither row names a media capability or modality, whatever the checkpoint's config says.
            for ([_][]const u8{ "\"vision\"", "\"image\"", "\"video\"", "\"audio\"" }) |word| {
                if (std.mem.indexOf(u8, json, word) != null) std.debug.print("model entry names {s}: {s}\n", .{ word, json });
                try testing.expect(std.mem.indexOf(u8, json, word) == null);
            }
            if (!resident) return;
            const parsed = try std.json.parseFromSlice(std.json.Value, alloc, json, .{});
            defer parsed.deinit();
            const modalities = parsed.value.object.get("input_modalities").?.array.items;
            if (modalities.len != 1) std.debug.print("input_modalities: {s}\n", .{json});
            try testing.expectEqual(@as(usize, 1), modalities.len);
            try testing.expectEqualStrings("text", modalities[0].string);
        }
    }.check;
    try expect_text_only(a, io, entry, true);
    entry.state = .unloaded;
    try expect_text_only(a, io, entry, false);
}

test "routeExists answers endpoint existence without consulting the model" {
    // Whether a path EXISTS has nothing to do with which models are loaded, but
    // model resolution runs ahead of dispatch — so on a headless boot (`--serve
    // --model-dir`, no `--model`, how the app always launches) an unknown path
    // never reached the 404 branch and came back 503 "No default model
    // configured". Any client that maps endpoints by probing them then concludes
    // the server implements EVERYTHING (llmprobe: "server answers unknown paths
    // with HTTP 503" → every surface scored absent, live 2026-07-25).
    try std.testing.expect(routeExists("/v1/chat/completions"));
    try std.testing.expect(routeExists("/v1/embeddings"));
    try std.testing.expect(routeExists("/health"));

    try std.testing.expect(!routeExists("/v1/__llmprobe_no_such_endpoint__"));
    try std.testing.expect(!routeExists("/v1/audio/translations")); // real OpenAI route we don't serve
    try std.testing.expect(!routeExists("/nope"));
    try std.testing.expect(!routeExists(""));
}

test "ROUTE_PATHS covers every path the dispatch chain compares (drift guard)" {
    // Two lists that must agree is the drift class this file already warns about
    // elsewhere. Rather than trust a comment, read the dispatch chain itself: a
    // new path-equality arm that nobody adds to ROUTE_PATHS makes that endpoint
    // a 404 on a headless server — the exact failure this table exists to
    // prevent, inverted. (This test scans for the literal call form, so don't
    // write one inside a comment: it would be scanned like real code.)
    const src = @embedFile("server.zig");
    const needle = "std.mem.eql(u8, path, \"";
    var i: usize = 0;
    var checked: usize = 0;
    while (std.mem.indexOfPos(u8, src, i, needle)) |at| {
        const start = at + needle.len;
        const end = std.mem.indexOfScalarPos(u8, src, start, '"') orelse break;
        const literal = src[start..end];
        i = end;
        checked += 1;
        // "/" is not a route: it is the HEAD liveness probe and the login form's path, both answered before routing
        if (std.mem.eql(u8, literal, "/")) continue;
        if (!routeExists(literal)) {
            std.debug.print("dispatched path missing from ROUTE_PATHS: {s}\n", .{literal});
            return error.RoutePathNotRegistered;
        }
    }
    // The scan itself must not silently find nothing.
    try std.testing.expect(checked >= 21);
}

test "every streaming chat emitter carries logprobs (silently-ignored-field guard)" {
    // The class: `logprobs` was parsed, honored to the point of disabling every
    // speculative path (so the throughput was paid), computed per token by the
    // generator — and then dropped, because the SSE chunk template had no
    // `logprobs` field at all. Non-streaming was perfect the whole time.
    //
    // Nothing existing could see it. Output-equality tests can't: the deltas
    // are byte-identical either way. llmprobe can't: it probes logprobs
    // non-streaming only, and scored this server 100% on `Logprob consistency`
    // while streaming returned nothing at all.
    //
    // So the guard is structural — a NEW emitter added to the streaming loop
    // must not be able to forget. `.{}` is the "nothing to add" form; inside
    // this one function it is the bug.
    const src = @embedFile("server.zig");
    const fn_start = std.mem.indexOf(u8, src, "fn handleStreamingGeneration(") orelse
        return error.StreamingHandlerNotFound;
    // The function ends at the next top-level `fn ` after it.
    const rest = src[fn_start + 8 ..];
    const fn_end = fn_start + 8 + (std.mem.indexOf(u8, rest, "\nfn ") orelse rest.len);
    const body = src[fn_start..fn_end];

    // The rule has two halves, because `logprobs.content` describes
    // `message.content`. A CONTENT emitter must drain, or the field is built
    // and dropped. A REASONING emitter must NOT: the thought is cut before the
    // client sees it, so entries riding a reasoning chunk describe text that
    // was never delivered — which is how `logprobs.content[0]` came to be the
    // first token of the model's thinking on every model that thinks.
    const needle = "sendSSEChunk" ++ "(allocator";
    const bare = ", ." ++ "{});";
    const reasoning = ".reasoning" ++ "_content = ";
    var i: usize = 0;
    var content_emitters: usize = 0;
    var reasoning_emitters: usize = 0;
    while (std.mem.indexOfPos(u8, body, i, needle)) |at| {
        const line_end = std.mem.indexOfScalarPos(u8, body, at, '\n') orelse body.len;
        const line = body[at..line_end];
        const is_reasoning = std.mem.indexOf(u8, line, reasoning) != null;
        const is_bare = std.mem.indexOf(u8, line, bare) != null;
        if (is_reasoning) {
            reasoning_emitters += 1;
            if (!is_bare) {
                std.debug.print("reasoning emitter ships CONTENT logprobs: {s}\n", .{line});
                return error.ReasoningEmitterShipsLogprobs;
            }
        } else {
            content_emitters += 1;
            if (is_bare) {
                std.debug.print("streaming emitter drops logprobs: {s}\n", .{line});
                return error.StreamingEmitterDropsLogprobs;
            }
        }
        i = line_end;
    }
    // Zeroes would pass the loops vacuously — both shapes must exist. (The
    // include_usage chunk consolidation — the ending ships on exactly ONE
    // chunk — took content emitters from 10 to 9.)
    try std.testing.expect(content_emitters >= 9);
    try std.testing.expect(reasoning_emitters >= 5);

    // The boundary itself: a content chunk emitted after a think block has to
    // move the cursor past the reasoning tokens first, or the answer's chunk
    // ships the whole thought's entries (measured: 42 on a 1-char delta).
    const skips = "lps.skipToContent" ++ "(";
    const drops = "lps.dropPending" ++ "(";
    try std.testing.expect(std.mem.indexOf(u8, body, skips) != null);
    try std.testing.expect(std.mem.indexOf(u8, body, drops) != null);

    // Both halves of the path to the wire, or the field is built and dropped:
    // sendSSEChunk must READ the extra, and the rendered bytes must be
    // INTERPOLATED into the chunk (a `logprobs` sibling of `delta` on the
    // choice — not a field inside it, which is where it does not belong).
    const reads_extra = "extras." ++ "logprobs_json";
    const interpolates = "lp_buf." ++ "items";
    try std.testing.expect(std.mem.indexOf(u8, src, reads_extra) != null);
    const chunk_at = std.mem.indexOf(u8, src, "chat.completion.chunk") orelse
        return error.ChunkTemplateNotFound;
    try std.testing.expect(std.mem.indexOfPos(u8, src, chunk_at, interpolates) != null);
}

test "ipIsLoopback exempts local addresses only" {
    // IPv4 loopback (whole 127.0.0.0/8) is exempt; a LAN address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 3, 2, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 192, 168, 1, 42 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 10, 0, 0, 5 }, .port = 0 } }));
    // IPv6 ::1 is exempt; a routable v6 address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
    // IPv4-mapped loopback ::ffff:127.0.0.1
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
}

/// Extract a JSON field's raw value from a JSON body string.
/// Returns the raw substring for the field value (e.g., the array or object).
fn extractJsonField(body: []const u8, field: []const u8) ?[]const u8 {
    // Search for "field": or "field" :
    var pos: usize = 0;
    while (pos < body.len) {
        const quote_pos = std.mem.indexOf(u8, body[pos..], "\"") orelse return null;
        const key_start = pos + quote_pos + 1;
        if (key_start + field.len >= body.len) return null;

        if (std.mem.eql(u8, body[key_start .. key_start + field.len], field) and
            body[key_start + field.len] == '"')
        {
            // Found the key, skip to colon
            var i = key_start + field.len + 1;
            while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\n' or body[i] == '\r' or body[i] == '\t')) {
                i += 1;
            }
            if (i >= body.len) return null;

            // Now extract the value - find matching bracket/brace
            const start = i;
            const open = body[start];
            const close: u8 = if (open == '[') ']' else if (open == '{') '}' else return null;
            var depth: usize = 1;
            var j = start + 1;
            var in_string = false;
            while (j < body.len and depth > 0) {
                if (body[j] == '\\' and in_string) {
                    j += 1; // skip escaped char
                } else if (body[j] == '"') {
                    in_string = !in_string;
                } else if (!in_string) {
                    if (body[j] == open) depth += 1;
                    if (body[j] == close) depth -= 1;
                }
                j += 1;
            }
            if (depth == 0) return body[start..j];
            return null;
        }
        pos = key_start;
    }
    return null;
}

pub const gpu_context_lost_message =
    "The GPU context was lost (a CUDA kernel faulted) and every GPU call now fails. Restart the server; the server log names the first error.";

/// Body text for a context-overflow 400, in a caller-owned buffer.
///
/// The COUNTS are the point. "Prompt exceeds maximum context length" tells a
/// client only that it failed; with both numbers the client can say how far
/// over it went and offer the one action that fixes it (raise the context to
/// N). Our own app renders exactly that, and the numbers are only knowable
/// here — the request is rejected before any usage is reported.
///
/// The legacy sentence stays the PREFIX: `APIError.looksLikeContextOverflow`
/// and third-party clients key on it, so appending must not move it. Output is
/// digits and spaces, so it carries no escaping hazard into the JSON sink, and
/// a bufPrint failure falls back to the bare sentence rather than sending no
/// body at all (the media-gen fixed-buffer class).
/// The `error.NotEnoughMemory` refusal, in ONE place because two identical
/// copies of it is how they drift.
///
/// The old wording ("retry after current requests complete") named concurrency,
/// which is the one thing it is never waiting for: the gate refuses against a
/// STATIC cap, so on an idle server with nothing loaded there is nothing to
/// retry after (#126). The counts are knowable only server-side and go to the
/// log; what the client needs is the knob.
///
/// ONE gate reaches this arm: the resident-model budget with no evictable
/// LRU victim (scheduler's `planEvictionsLocked` refusal). The free-memory
/// PREFLIGHT refuses as `error.InsufficientMemory` and gets its own message
/// below (#144) — blaming free memory here sends a user quitting apps when
/// the fix is unloading a model or moving the cap.
pub const not_enough_memory_message =
    "Not enough memory to load model: it would exceed the resident-model budget and no loaded model can be evicted. " ++
    "Unload the model you are chatting with (tray > Models > eject), " ++
    "or raise a cap: --max-resident-models <n>, --max-resident-mem <size>|0. " ++
    "The server log names the exact figures and the current cap.";

/// #144: the memory PREFLIGHT refusal (free RAM can't hold weights + warmup
/// headroom) is distinct from the resident-budget gate above and has a
/// different knob. Counts go to the log (#126); the client gets the remedy.
pub const insufficient_free_memory_message =
    "Not enough free memory to load model: weights + warmup headroom exceed what is currently available. " ++
    "Close other apps or models and retry (the server log names the peak estimate and available memory), or pass --skip-mem-preflight to override.";

/// The response for a failed `Scheduler.ensureLoaded(id)`.
fn sendEnsureLoadedError(allocator: std.mem.Allocator, stream: *Conn, sched: *scheduler_mod.Scheduler, id: []const u8, err: anyerror) !void {
    switch (err) {
        error.UnknownModelId => try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404),
        error.NotLoaded => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "model_not_loaded", "Requested model is not currently loaded", 503),
        error.NoDefaultModel => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503),
        error.NotEnoughMemory => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "out_of_memory", not_enough_memory_message, 503),
        error.InsufficientMemory => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "out_of_memory", insufficient_free_memory_message, 503),
        error.LoadFailed => try sendLoadFailedResponse(allocator, stream, sched, id),
        error.Shutdown => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503),
        error.GpuContextLost => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "gpu_context_lost", gpu_context_lost_message, 503),
        // Other errors (CPU-side preload failures like FileNotFound on a
        // missing tokenizer.json, JSON parse errors on a malformed
        // config.json, etc.) bubble out of `preloadCpuState`. Surface a
        // 500 with the error name so the client gets a clean failure
        // instead of a hung connection.
        else => {
            log.warn("  -> 500 ({s}) while loading {s}\n", .{ @errorName(err), id });
            const msg = std.fmt.allocPrint(allocator, "Failed to load model: {s}", .{@errorName(err)}) catch {
                try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Failed to load model", 500);
                return;
            };
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
        },
    }
}

/// #144: `error.LoadFailed` used to surface as a bare "Model load failed" for
/// every on-demand load failure — the inference thread's error name was
/// dropped at the thread boundary. The registry keeps it; echo it.
fn sendLoadFailedResponse(allocator: std.mem.Allocator, stream: *Conn, sched: *scheduler_mod.Scheduler, id: []const u8) !void {
    if (sched.loadErrorName(allocator, id)) |name| {
        defer allocator.free(name);
        if (std.fmt.allocPrint(allocator, "Model load failed: {s}", .{name}) catch null) |msg| {
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        }
    }
    try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
}

/// A request refused before it owned a slot. A null sink (metrics off) is one branch.
fn countRejected() void {
    if (g_metrics) |m| m.recordRejected();
}

fn contextOverflowMessage(buf: []u8, prompt_tokens: usize, ctx: usize) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "Prompt exceeds maximum context length: {d} tokens requested, {d} available",
        .{ prompt_tokens, ctx },
    ) catch "Prompt exceeds maximum context length";
}

fn sendErrorResponse(allocator: std.mem.Allocator, stream: *Conn, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);

    var code_buf: [16]u8 = undefined;
    const code_str = if (code) |c|
        std.fmt.bufPrint(&code_buf, "{d}", .{c}) catch "null"
    else
        "null";

    const body = try std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":{s},"type":"{s}","param":null,"code":{s}}}}}
    , .{ escaped_msg, err_type, code_str });
    defer allocator.free(body);
    try sendResponse(stream, status, "application/json", body);
}

/// A copy of `s`, the same length, with every byte that does not start a valid UTF-8
/// sequence replaced by '?'. Length-preserving on purpose: the stream's byte accounting
/// counts raw token bytes (U+FFFD would add two bytes per replaced byte). Caller frees.
fn utf8Replaced(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 0;
        if (n > 0 and i + n <= s.len and std.unicode.utf8ValidateSlice(s[i .. i + n])) {
            try out.appendSlice(allocator, s[i .. i + n]);
            i += n;
        } else {
            try out.append(allocator, '?');
            i += 1;
        }
    }
    return out.toOwnedSlice(allocator);
}

test "utf8Replaced keeps valid text and replaces stray bytes one for one" {
    const a = testing.allocator;
    const ok = try utf8Replaced(a, "ciao \xF0\x9F\x87\xAE\xF0\x9F\x87\xB9!");
    defer a.free(ok);
    try testing.expectEqualStrings("ciao \xF0\x9F\x87\xAE\xF0\x9F\x87\xB9!", ok);
    const bad = try utf8Replaced(a, "a\x80\xFFb");
    defer a.free(bad);
    try testing.expectEqualStrings("a??b", bad);
}

/// Returns the number of trailing bytes that form an incomplete UTF-8 sequence.
/// If the string ends with a complete codepoint (or is empty), returns 0.
fn utf8TrailingIncomplete(s: []const u8) usize {
    if (s.len == 0) return 0;
    // Walk backwards to find the last leading byte (one with bit pattern 11xxxxxx or 0xxxxxxx)
    var i: usize = s.len;
    // Check up to 3 trailing continuation bytes (10xxxxxx)
    var cont: usize = 0;
    while (cont < 3 and i > 0) {
        i -= 1;
        if (s[i] & 0xC0 != 0x80) break; // found a non-continuation byte
        cont += 1;
    }
    // i now points to the last leading byte (or the byte that broke the loop)
    if (i >= s.len) return 0;
    const lead = s[i];
    // Determine expected sequence length from leading byte
    const expected: usize = if (lead & 0x80 == 0) 1 // 0xxxxxxx — ASCII
        else if (lead & 0xE0 == 0xC0) 2 // 110xxxxx
        else if (lead & 0xF0 == 0xE0) 3 // 1110xxxx
        else if (lead & 0xF8 == 0xF0) 4 // 11110xxx
        else return 0; // invalid leading byte, don't buffer
    const actual = s.len - i;
    return if (actual < expected) actual else 0;
}

/// Build a llama.cpp-style `timings` JSON object (no surrounding key) from
/// raw nanosecond counts and token totals. Caller frees. Returns an empty
/// string when `prefill_ns`, `decode_ns`, AND `tokenize_ns` are all zero
/// (legacy paths that don't measure timing) so the field can be omitted.
///
/// `tokenize_ns` is the wall-clock cost of the synchronous
/// `renderChatTemplate + tokenizer.encode` step on the request thread.
/// Phase 4 #1 of `performance-plan.md` calls for instrumentation-first:
/// before we move tokenize onto a worker thread, we need numbers per
/// engine / prompt length. Pass 0 for legacy paths that don't measure it
/// (the field is then omitted from the JSON; existing callers stay shape-
/// compatible while the new ones surface the metric).
///
/// Iteration 2: chat-template render+tokenize wrapper that consults a
/// per-LoadedModel LRU cache before calling the underlying engine's
/// encoder. On hit the returned slice is a fresh allocation owned by the
/// caller (drop-in replacement for the engine encoders). On miss it
/// stores a copy of the result in the cache.
fn cachedFormatChat(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    messages: []const chat_mod.Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    reasoning_effort: ?[]const u8,
    /// Extend the trailing assistant message rather than answering after it.
    continue_final: bool,
) ![]u32 {
    const cache_ptr: ?*tokenize_cache_mod.TokenizeCache = if (lm.tokenize_cache) |*tc| tc else null;
    const key_opt: ?u64 = if (cache_ptr != null)
        tokenize_cache_mod.TokenizeCache.keyFor(messages, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final, chat_config.chat_template_kwargs)
    else
        null;
    if (cache_ptr) |cache| if (key_opt) |key| {
        if (try cache.get(io, key, allocator)) |cached| return cached;
    };
    const ids = try chat_mod.formatChat(allocator, tok, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, continue_final);
    if (cache_ptr) |cache| if (key_opt) |key| {
        // Insert is best-effort; an OOM in the cache shouldn't fail the
        // request — the user already has their tokenized prompt.
        cache.put(io, key, ids) catch |err| {
            log.warn("[tokenize-cache] put failed: {s}\n", .{@errorName(err)});
        };
    };
    return ids;
}

/// The chat-completions `usage` object, shared by the non-stream, tool-call
/// and streaming-final-chunk emitters so the shape never drifts per path.
/// `prompt_tokens_details.cached_tokens` is ALWAYS present (0 when cold) —
/// OpenAI emits it unconditionally, and black-box conformance/cost tooling
/// (llmprobe) treats a missing field as "no prompt caching" even though the
/// engine caches. `extra_details` is a pre-formatted `,"key":{...}` fragment
/// (completion_tokens_details) or "".
fn formatChatUsage(
    allocator: std.mem.Allocator,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    extra_details: []const u8,
) ![]u8 {
    return try std.fmt.allocPrint(allocator,
        \\{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d},"prompt_tokens_details":{{"cached_tokens":{d}}}{s}}}
    , .{ prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, cached_tokens, extra_details });
}

fn formatTimingsObject(
    allocator: std.mem.Allocator,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
    tokenize_ns: u64,
) ![]u8 {
    if (prefill_ns == 0 and decode_ns == 0 and tokenize_ns == 0) return try allocator.alloc(u8, 0);
    const ns_per_ms_f: f64 = @as(f64, @floatFromInt(std.time.ns_per_ms));
    const p_ms = @as(f64, @floatFromInt(prefill_ns)) / ns_per_ms_f;
    const d_ms = @as(f64, @floatFromInt(decode_ns)) / ns_per_ms_f;
    const t_ms = @as(f64, @floatFromInt(tokenize_ns)) / ns_per_ms_f;
    // prompt_per_second reflects compute: divide by the tokens actually run
    // (prompt minus the KV-cache prefix). `cached_n` exposes the reuse so a
    // bench / client can tell a warm hit from a cold prefill.
    const p_tps = generate_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = generate_mod.tokensPerSec(completion_tokens, decode_ns);
    // Always emit `tokenize_ms` (even at 0.0) so clients can rely on the key
    // being present — they can branch on the value, not on presence/absence.
    return try std.fmt.allocPrint(
        allocator,
        \\{{"prompt_n":{d},"cached_n":{d},"prompt_ms":{d:.3},"prompt_per_second":{d:.3},"predicted_n":{d},"predicted_ms":{d:.3},"predicted_per_second":{d:.3},"tokenize_ms":{d:.3}}}
    ,
        .{ prompt_tokens, cached_tokens, p_ms, p_tps, completion_tokens, d_ms, d_tps, t_ms },
    );
}

/// Format the "prefill: X tok/s [(C cached / P total)], decode: Y tok/s" perf
/// bracket into `buf` so every API path logs prefill compute (and any prefix
/// reuse) identically. `cached > 0` adds the cached/total hint; otherwise the
/// short form keeps the existing log shape for cold prefills. Single source of
/// truth so the format never drifts across the chat/anthropic/streaming logs.
fn formatPerfBracket(
    buf: []u8,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
) []const u8 {
    const p_tps = generate_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = generate_mod.tokensPerSec(completion_tokens, decode_ns);
    const s = if (cached_tokens > 0)
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s ({d} cached / {d} total), decode: {d:.1} tok/s", .{ p_tps, cached_tokens, prompt_tokens, d_tps })
    else
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s, decode: {d:.1} tok/s", .{ p_tps, d_tps });
    return s catch buf[0..0];
}

/// THE JSON string escaper for every response body. Every string here is model bytes
/// somewhere and a token is a BPE fragment, so invalid UTF-8 is sanitized INSIDE the
/// escaper (one U+FFFD per bad sequence); valid input passes through byte for byte.
fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (std.unicode.utf8ValidateSlice(input)) return jsonEscapeValid(allocator, input);
    const clean = try chat_mod.utf8Sanitize(allocator, input);
    defer allocator.free(clean);
    return jsonEscapeValid(allocator, clean);
}

fn jsonEscapeValid(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    try result.append(allocator, '"');
    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var esc_buf: [6]u8 = undefined;
                    const s = std.fmt.bufPrint(&esc_buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try result.appendSlice(allocator, s);
                } else {
                    try result.append(allocator, c);
                }
            },
        }
    }
    try result.append(allocator, '"');
    return result.toOwnedSlice(allocator);
}

test "EVERY JSON string escaper survives bytes that are not valid UTF-8" {
    // Class guard over the two escapers (the reply writer, the prompt
    // render): one passing bytes through loses a whole reply.
    const a = std.testing.allocator;
    const cases = [_][]const u8{
        "near\xe8\x91\xe4\xb8\x89 in", // 3-byte lead, one continuation, then another lead
        "\xff\xfe", // never-valid lead bytes
        "tail\xf0\x9f", // first half of a 4-byte emoji, cut by max_tokens
    };
    for (cases) |bad| {
        const a1 = try jsonEscape(a, bad);
        defer a.free(a1);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a1));

        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(a);
        try chat_mod.appendJsonString(a, &buf, bad);
        try std.testing.expect(std.unicode.utf8ValidateSlice(buf.items));
    }
    // Valid input is passed through byte for byte.
    const good = try jsonEscape(a, "héllo \u{1F600}");
    defer a.free(good);
    try std.testing.expectEqualStrings("\"héllo \u{1F600}\"", good);
}

/// Build logprobs JSON for a single token (for both streaming and non-streaming).
/// Returns a string like: {"token":"hello","logprob":-1.23,"bytes":[104,101],"top_logprobs":[...]}
fn formatTokenLogprob(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_id: u32,
    logprob: f32,
    top_logprobs: []const generate_mod.TokenLogprob,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    const strip = tok.tok_type == .sentencepiece_bpe;
    const token_text = try tok.decode(allocator, &[_]u32{token_id}, strip and false);
    defer allocator.free(token_text);

    const escaped_token = try jsonEscape(allocator, token_text);
    defer allocator.free(escaped_token);

    // Build bytes array
    var bytes_buf = std.ArrayList(u8).empty;
    defer bytes_buf.deinit(allocator);
    try bytes_buf.appendSlice(allocator, "[");
    for (token_text, 0..) |b, i| {
        if (i > 0) try bytes_buf.appendSlice(allocator, ",");
        const num = try std.fmt.allocPrint(allocator, "{d}", .{b});
        defer allocator.free(num);
        try bytes_buf.appendSlice(allocator, num);
    }
    try bytes_buf.appendSlice(allocator, "]");

    // Build top_logprobs array
    var top_buf = std.ArrayList(u8).empty;
    defer top_buf.deinit(allocator);
    try top_buf.appendSlice(allocator, "[");
    for (top_logprobs, 0..) |tlp, i| {
        if (i > 0) try top_buf.appendSlice(allocator, ",");

        const tlp_text = try tok.decode(allocator, &[_]u32{tlp.token_id}, strip and false);
        defer allocator.free(tlp_text);
        const escaped_tlp = try jsonEscape(allocator, tlp_text);
        defer allocator.free(escaped_tlp);

        // Bytes for this token
        var tlp_bytes = std.ArrayList(u8).empty;
        defer tlp_bytes.deinit(allocator);
        try tlp_bytes.appendSlice(allocator, "[");
        for (tlp_text, 0..) |b, j| {
            if (j > 0) try tlp_bytes.appendSlice(allocator, ",");
            const num = try std.fmt.allocPrint(allocator, "{d}", .{b});
            defer allocator.free(num);
            try tlp_bytes.appendSlice(allocator, num);
        }
        try tlp_bytes.appendSlice(allocator, "]");

        const entry = try std.fmt.allocPrint(allocator,
            \\{{"token":{s},"logprob":{d:.6},"bytes":{s}}}
        , .{ escaped_tlp, tlp.logprob, tlp_bytes.items });
        defer allocator.free(entry);
        try top_buf.appendSlice(allocator, entry);
    }
    try top_buf.appendSlice(allocator, "]");

    const result = try std.fmt.allocPrint(allocator,
        \\{{"token":{s},"logprob":{d:.6},"bytes":{s},"top_logprobs":{s}}}
    , .{ escaped_token, logprob, bytes_buf.items, top_buf.items });

    return result;
}

/// Build the LEGACY `/v1/completions` logprobs object — a different shape from
/// chat's: four parallel arrays (`tokens`, `token_logprobs`, `top_logprobs`,
/// `text_offset`), with `top_logprobs` a map from token text to logprob.
/// `text_offset` is the byte offset of each token within the completion text,
/// which is what a FIM client uses to align alternatives with the buffer.
/// `offset_base` is READ and ADVANCED: `text_offset` indexes into the whole
/// completion, so a streaming caller emitting one chunk per token has to carry
/// the running total across calls. Non-streaming passes a local zero.
fn formatCompletionsLogprobs(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    logprobs: []const generate_mod.LogprobResult,
    /// Echo: the first token has no logprob (`null`), so `logprobs[i]` belongs to token i + 1.
    null_first: bool,
    offset_base: *usize,
) ![]const u8 {
    var toks = std.ArrayList(u8).empty;
    defer toks.deinit(allocator);
    var lps = std.ArrayList(u8).empty;
    defer lps.deinit(allocator);
    var tops = std.ArrayList(u8).empty;
    defer tops.deinit(allocator);
    var offs = std.ArrayList(u8).empty;
    defer offs.deinit(allocator);
    try toks.appendSlice(allocator, "[");
    try lps.appendSlice(allocator, "[");
    try tops.appendSlice(allocator, "[");
    try offs.appendSlice(allocator, "[");

    var offset: usize = offset_base.*;
    const lead: usize = @intFromBool(null_first);
    const count = @min(token_ids.len, logprobs.len + lead);
    for (0..count) |i| {
        if (i > 0) {
            try toks.appendSlice(allocator, ",");
            try lps.appendSlice(allocator, ",");
            try tops.appendSlice(allocator, ",");
            try offs.appendSlice(allocator, ",");
        }
        const text = try tok.decode(allocator, &[_]u32{token_ids[i]}, false);
        defer allocator.free(text);
        const esc = try jsonEscape(allocator, text);
        defer allocator.free(esc);
        try toks.appendSlice(allocator, esc);

        var num: [48]u8 = undefined;
        try offs.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d}", .{offset}));
        offset += text.len;
        if (i < lead) {
            try lps.appendSlice(allocator, "null");
            try tops.appendSlice(allocator, "null");
            continue;
        }
        const lp = logprobs[i - lead];
        try lps.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d:.6}", .{lp.token_logprob}));

        try tops.appendSlice(allocator, "{");
        for (lp.top_logprobs, 0..) |t, j| {
            if (j > 0) try tops.appendSlice(allocator, ",");
            const ttext = try tok.decode(allocator, &[_]u32{t.token_id}, false);
            defer allocator.free(ttext);
            const tesc = try jsonEscape(allocator, ttext);
            defer allocator.free(tesc);
            try tops.appendSlice(allocator, tesc);
            try tops.appendSlice(allocator, ":");
            try tops.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d:.6}", .{t.logprob}));
        }
        try tops.appendSlice(allocator, "}");
    }
    try toks.appendSlice(allocator, "]");
    try lps.appendSlice(allocator, "]");
    try tops.appendSlice(allocator, "]");
    try offs.appendSlice(allocator, "]");
    offset_base.* = offset;

    return try std.fmt.allocPrint(allocator,
        \\{{"tokens":{s},"token_logprobs":{s},"top_logprobs":{s},"text_offset":{s}}}
    , .{ toks.items, lps.items, tops.items, offs.items });
}

/// The half-open token range whose decoded text lands inside `content`.
const ContentTokenRange = struct { start: usize, end: usize };

/// Which of the generated tokens survive into `message.content`.
///
/// OpenAI defines `logprobs.content` as the tokens of the message CONTENT, but
/// the generation also carries the reasoning block, any leaked tool markup and
/// a loop-trimmed tail — all cut before the text reaches the client. Emitting
/// entries for cut text makes the array correspond to nothing the client can
/// see: on every model that thinks, `logprobs.content[0]` was the first token
/// of the reasoning.
///
/// The split helpers return raw slices into `full_text`, so the offset is exact
/// pointer arithmetic. A content that is not a slice of it (a future transform
/// that rewrites rather than cuts) falls back to a search, and failing that to
/// the full range — an array we cannot align is still better than none.
fn contentTokenRange(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    full_text: []const u8,
    content: []const u8,
) ContentTokenRange {
    const all = ContentTokenRange{ .start = 0, .end = token_ids.len };
    if (content.len == 0) return .{ .start = 0, .end = 0 };

    const base = @intFromPtr(full_text.ptr);
    const cptr = @intFromPtr(content.ptr);
    const start_off = if (cptr >= base and cptr + content.len <= base + full_text.len)
        cptr - base
    else
        std.mem.indexOf(u8, full_text, content) orelse return all;
    const end_off = start_off + content.len;

    var start: ?usize = null;
    var off: usize = 0;
    for (token_ids, 0..) |id, i| {
        if (off >= end_off) return .{ .start = start orelse i, .end = i };
        const text = tok.decode(allocator, &[_]u32{id}, false) catch return all;
        defer allocator.free(text);
        // A token straddling the boundary belongs to content: it put bytes there.
        if (start == null and off + text.len > start_off) start = i;
        off += text.len;
    }
    return .{ .start = start orelse token_ids.len, .end = token_ids.len };
}

/// Build the full logprobs object for a non-streaming response.
fn formatLogprobsObject(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    logprobs: []const generate_mod.LogprobResult,
) ![]const u8 {
    var content_buf = std.ArrayList(u8).empty;
    defer content_buf.deinit(allocator);

    try content_buf.appendSlice(allocator, "[");
    const count = @min(token_ids.len, logprobs.len);
    for (0..count) |i| {
        if (i > 0) try content_buf.appendSlice(allocator, ",");
        const entry = try formatTokenLogprob(allocator, tok, token_ids[i], logprobs[i].token_logprob, logprobs[i].top_logprobs);
        defer allocator.free(entry);
        try content_buf.appendSlice(allocator, entry);
    }
    try content_buf.appendSlice(allocator, "]");

    return try std.fmt.allocPrint(allocator, "{{\"content\":{s}}}", .{content_buf.items});
}

/// The multiplicative `repeat_penalty` (llama.cpp's); a value <= 0 is an omission.
fn requestRepeatPenalty(root: std.json.ObjectMap, gen: generation_settings.Profile) f32 {
    const repeat = parseJsonFloat(root, "repeat_penalty", 0.0, 0.0, 10.0);
    return gen.resolve(f32, .repeat_penalty, if (repeat > 0) repeat else null, 1.0);
}

/// Parse a float from a JSON value, clamping to [min, max]. Returns default if missing/invalid.
fn parseJsonFloat(root: std.json.ObjectMap, key: []const u8, default: f32, min: f32, max: f32) f32 {
    const raw = if (root.get(key)) |v| switch (v) {
        .float => |f| @as(f32, @floatCast(f)),
        .integer => |i| @as(f32, @floatFromInt(i)),
        else => default,
    } else default;
    return std.math.clamp(raw, min, max);
}

/// Wave 1.A — parse the optional per-request `kv_quant` body field. Accepts
/// the string forms `"off"`, `"0"`, `"4"`, `"8"` and the integer forms `0`,
/// `4`, `8`. Returns null when the field is absent or unrecognized (the
/// caller falls back to the process-level `--kv-quant` default carried on
/// the scheduler). Returns `KVQuantConfig.dense` for "off"/0.
fn parseKvQuantOverride(root: std.json.ObjectMap) ?transformer_mod.KVQuantConfig {
    const v = root.get("kv_quant") orelse return null;
    return transformer_mod.KVQuantConfig.fromJsonValue(v);
}

/// The text-only 400 for a `/v1/embeddings` `messages` array that carries an image, video or audio
/// part in any message; null when every part is text.
fn embeddingMessagesMediaReject(messages: []const std.json.Value) ?[]const u8 {
    for (messages) |message| {
        if (message != .object) continue;
        const content = message.object.get("content") orelse continue;
        if (chat_mod.contentValueHasMedia(content)) return TEXT_ONLY_MEDIA_REJECT;
    }
    return null;
}

/// SSE comment keepalive for the OpenAI-style streaming surfaces. Comments
/// are SSE-spec-legal and skipped by every SSE parser.
fn sendStreamKeepalive(stream: *Conn) !void {
    try stream.writeAll(": keepalive\n\n");
}

/// Emit a keepalive iff the socket has been SILENT for `STREAM_KEEPALIVE_MS`.
///
/// Every streaming token loop calls this once per iteration — not just on the
/// `.idle` (waiting-for-first-token) path. Those two are different facts: a
/// surface buffering tokens for tool-call or thinking detection is receiving
/// tokens steadily while writing nothing, and only bytes on the wire hold off
/// a client's idle-body timeout (Node's `fetch`/undici: 300 s, then
/// `TypeError: terminated`). A one-shot `write_file` of a whole source file
/// buffers for exactly that long.
fn beatStreamKeepalive(stream: *Conn) !void {
    if (!stream.keepaliveDue()) return;
    try sendStreamKeepalive(stream);
    stream.heartbeat.noteWrite(nowMsMonotonic(stream.io));
}

// ── Tests ──

const testing = std.testing;

test "Conn.peerClosed: alive socket returns false, closed peer returns true" {
    // Create a connected socket pair (AF_UNIX SOCK_STREAM via socketpair).
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    const rc = std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv);
    try testing.expect(rc == 0);

    const server_fd = sv[0];
    const client_fd = sv[1];

    // Build a Conn that wraps the server-side fd. We only need
    // `stream.socket.handle` for peerClosed, so the Reader/Writer state
    // can stay zeroed.
    var conn: Conn = undefined;
    conn.stream = .{ .socket = .{ .handle = server_fd, .address = undefined } };

    // Healthy connection: no data pending, no FIN → peerClosed returns false.
    try testing.expect(!conn.peerClosed());

    // Client closes its side → server should observe FIN/HUP.
    _ = std.c.close(client_fd);

    // socketpair() returns connected sockets in the kernel; close-of-peer
    // is observable immediately on the other side without delay.
    const closed = conn.peerClosed();
    _ = std.c.close(server_fd);
    try testing.expect(closed);
}

/// Runs `Conn.close` after one response, with a peer that stays connected, and checks whether it
/// held the socket (`close_delimited`) or returned at once.
fn expectCloseWaitsForPeer(close_delimited: bool) !void {
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    try testing.expect(std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv) == 0);

    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    const tail: []const u8 = if (close_delimited) "data: [DONE]\n\n" else "{}";
    if (close_delimited) {
        try sendSseHeaders(&conn, "test", SSE_ALLOW_HEADERS_DEFAULT);
        try conn.writeAll(tail);
    } else {
        try sendResponseFramed(&conn, "200 OK", "application/json", tail);
    }

    var closed = std.atomic.Value(bool).init(false);
    const closer = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, done: *std.atomic.Value(bool)) void {
            c.close();
            done.store(true, .release);
        }
    }.run, .{ &conn, &closed });
    var client_open = true;
    defer closer.join();
    defer if (client_open) {
        _ = std.c.close(sv[1]);
    };

    // The reader gets the whole response and then EOF ...
    var buf: [512]u8 = undefined;
    var got: usize = 0;
    while (true) {
        const n = std.c.recv(sv[1], &buf[got], buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    try testing.expect(std.mem.endsWith(u8, buf[0..got], tail));

    // ... and only a close-delimited body keeps the server's end open, so the kernel cannot
    // reset a lagging reader.
    var waited: u32 = 0;
    while (!closed.load(.acquire) and waited < 50) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(10), .real) catch {};
    }
    try testing.expectEqual(!close_delimited, closed.load(.acquire));

    _ = std.c.close(sv[1]);
    client_open = false;
    waited = 0;
    while (!closed.load(.acquire) and waited < 300) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(10), .real) catch {};
    }
    try testing.expect(closed.load(.acquire));
}

test "Conn.close keeps the socket until the peer hangs up, after a close-delimited body" {
    try expectCloseWaitsForPeer(true);
}

test "Conn.close does not wait for the peer after a Content-Length response" {
    try expectCloseWaitsForPeer(false);
}

/// `request` through `StartupResponder.answer` over a socketpair from `peer`; the reply's bytes.
fn startupReply(r: *StartupResponder, request: []const u8, peer: std.Io.net.IpAddress, out: []u8) ![]const u8 {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = peer } }, testing.io);
    defer conn.close();
    try testing.expectEqual(@as(isize, @intCast(request.len)), std.c.send(sv[1], request.ptr, request.len, 0));
    _ = std.c.shutdown(sv[1], 1);
    try r.answer(&conn);
    try conn.flush();
    const got = std.c.recv(sv[1], out.ptr, out.len, std.c.MSG.DONTWAIT);
    return out[0..@intCast(@max(got, 0))];
}

test "PLD: the request's flag, else the default" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"enable_pld\":true}", .{});
    defer parsed.deinit();
    try testing.expect(requestPld(parsed.value.object));
}

test "startup: /health is 503 loading, a loopback client reads the load, the rest is 503" {
    var reg = try ModelRegistry.init(testing.allocator, testing.io, null, 3, 0, null);
    defer reg.deinit();
    const stub = try reg.registerStub("m", "/models/m", 10);
    reg.mutex.lockUncancelable(testing.io);
    try testing.expect(reg.tryBeginLoadLocked(stub));
    reg.mutex.unlock(testing.io);
    var r: StartupResponder = .{ .server = undefined, .registry = reg, .allocator = testing.allocator, .io = testing.io };
    const local: std.Io.net.IpAddress = .{ .ip4 = std.Io.net.Ip4Address.loopback(0) };
    const lan: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 1, 5 }, .port = 0 } };
    var buf: [4096]u8 = undefined;

    var reply = try startupReply(&r, "GET /health HTTP/1.1\r\n\r\n", local, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
    try testing.expect(std.mem.endsWith(u8, reply, "{\"status\":\"loading\"}"));
    reply = try startupReply(&r, "GET /v1/load-progress HTTP/1.1\r\n\r\n", local, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    try testing.expect(std.mem.indexOf(u8, reply, "\"path\":\"/models/m\",\"progress\":{\"phase\":\"starting\"") != null);
    reply = try startupReply(&r, "GET /v1/load-progress HTTP/1.1\r\n\r\n", lan, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
    reply = try startupReply(&r, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", local, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
}

test "listenExclusive: a second server cannot bind a port that is already listening" {
    var first = try listenExclusive(testing.io, .{ .ip4 = std.Io.net.Ip4Address.loopback(0) });
    defer first.deinit(testing.io);
    try testing.expectError(error.AddressInUse, listenExclusive(testing.io, first.socket.address));
}

/// Sends `request` (then EOF) over a socketpair and runs `readRequest` on the server end;
/// returns the body it read, or null, plus whatever the server wrote back.
fn readRequestOver(request: []const u8, body_out: []u8, reply_out: []u8) !struct { body: ?[]const u8, reply: []const u8 } {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    try testing.expectEqual(@as(isize, @intCast(request.len)), std.c.send(sv[1], request.ptr, request.len, 0));
    _ = std.c.shutdown(sv[1], 1);

    const raw = try readRequest(testing.allocator, &conn);
    try conn.flush();
    const got = std.c.recv(sv[1], reply_out.ptr, reply_out.len, std.c.MSG.DONTWAIT);
    const reply = reply_out[0..@intCast(@max(got, 0))];
    const r = raw orelse return .{ .body = null, .reply = reply };
    defer testing.allocator.free(r.buf);
    const body = r.buf[r.header_end..r.len];
    @memcpy(body_out[0..body.len], body);
    return .{ .body = body_out[0..body.len], .reply = reply };
}

/// Runs `readRequestWithin(limit_ms)` against a client that sends `sent`, then one byte per
/// `drip_ms` (0 = nothing more), and never finishes; true when the reader gave up on its own.
fn readerGivesUp(sent: []const u8, limit_ms: i64, drip_ms: u64) !bool {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    _ = std.c.send(sv[1], sent.ptr, sent.len, 0);

    var done = std.atomic.Value(bool).init(false);
    const reader = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, limit: i64, d: *std.atomic.Value(bool)) void {
            if (readRequestWithin(testing.allocator, c, limit) catch null) |r| testing.allocator.free(r.buf);
            d.store(true, .release);
        }
    }.run, .{ &conn, limit_ms, &done });
    var waited: u64 = 0;
    while (!done.load(.acquire) and waited < 2000) : (waited += 10) {
        std.Io.sleep(testing.io, .fromMilliseconds(10), .real) catch {};
        if (drip_ms > 0 and waited % drip_ms == 0) _ = std.c.send(sv[1], "x", 1, 0);
    }
    const gave_up = done.load(.acquire);
    _ = std.c.close(sv[1]);
    reader.join();
    return gave_up;
}

test "readRequest: a client that stalls its head or body, or drips its head, is dropped, not held forever" {
    const head = "POST /v1/chat/completions HTTP/1.1\r\n";
    try testing.expect(try readerGivesUp(head, 200, 0));
    try testing.expect(try readerGivesUp(head, 300, 50));
    try testing.expect(try readerGivesUp(head ++ "Content-Length: 100\r\n\r\n{}", 200, 0));
}

test "readRequest: bytes past Content-Length are dropped, a short body is no request, a wrapping Content-Length is a 413" {
    var body_buf: [64]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    const extra: [4000]u8 = @splat('x');
    const short = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}" ++ extra, &body_buf, &reply_buf);
    try testing.expectEqualStrings("{}", short.body.?);

    const none = try readRequestOver("GET /health HTTP/1.1\r\n\r\n" ++ extra, &body_buf, &reply_buf);
    try testing.expectEqualStrings("", none.body.?);

    // A client that hangs up mid-body sent no request: nothing is parsed, nothing answered.
    const cut = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hel", &body_buf, &reply_buf);
    try testing.expect(cut.body == null);
    try testing.expectEqualStrings("", cut.reply);

    const huge = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 18446744073709551615\r\n\r\n{}", &body_buf, &reply_buf);
    try testing.expect(huge.body == null);
    try testing.expect(std.mem.startsWith(u8, huge.reply, "HTTP/1.1 413"));
}

test "jsonNestingDepth: brackets inside strings never count" {
    try testing.expectEqual(@as(usize, 0), jsonNestingDepth("42"));
    try testing.expectEqual(@as(usize, 3), jsonNestingDepth("{\"a\":[{\"b\":1}]}"));
    try testing.expectEqual(@as(usize, 1), jsonNestingDepth("{\"a\":\"[[[{{{\"}"));
    try testing.expectEqual(@as(usize, 1), jsonNestingDepth("[\"\\\"[[\"]")); // escaped quote stays in the string
    const open: [5000]u8 = @splat('[');
    const close: [5000]u8 = @splat(']');
    try testing.expectEqual(@as(usize, 5000), jsonNestingDepth(&(open ++ close)));
}

test "readRequest: a head that expects 100-continue gets it before the body is read" {
    var body_buf: [64]u8 = undefined;
    var reply_buf: [256]u8 = undefined;
    // curl sends this for bodies over 1 MB and waits a second for the 100 before sending them.
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    const client = try std.Thread.spawn(.{}, struct {
        fn run(fd: std.posix.fd_t, reply: []u8) void {
            const head = "POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 10\r\n\r\n";
            _ = std.c.send(fd, head, head.len, 0);
            const got = std.c.recv(fd, reply.ptr, reply.len, 0); // the body waits for the 100
            if (got > 0) _ = std.c.send(fd, "0123456789", 10, 0);
            _ = std.c.shutdown(fd, 1);
        }
    }.run, .{ sv[1], &reply_buf });
    const raw = try readRequest(testing.allocator, &conn);
    client.join();
    _ = std.c.close(sv[1]);
    const r = raw orelse return error.TestUnexpectedResult;
    defer testing.allocator.free(r.buf);
    try testing.expectEqualStrings("0123456789", r.buf[r.header_end..r.len]);
    try testing.expect(std.mem.startsWith(u8, &reply_buf, "HTTP/1.1 100 Continue\r\n\r\n"));
    const whole = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n{}", &body_buf, &reply_buf);
    try testing.expectEqualStrings("", whole.reply); // the body already arrived
}

test "findContentLength parses header" {
    try testing.expectEqual(@as(?usize, 42), findContentLength("Host: localhost\r\nContent-Length: 42\r\nAccept: */*"));
}

test "findContentLength case insensitive" {
    try testing.expectEqual(@as(?usize, 100), findContentLength("content-length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("Content-Length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("CONTENT-LENGTH: 100"));
}

test "findContentLength returns null when missing" {
    try testing.expect(findContentLength("Host: localhost\r\nAccept: */*") == null);
    try testing.expect(findContentLength("") == null);
}

test "debug body log bounds BINARY but never truncates text (multipart PNG class)" {
    // A multipart upload is the first binary body we serve. `logHttpBody` dumped
    // whatever it got verbatim, so at --log-level debug one image upload wrote
    // raw PNG bytes into the log — and a body near the request cap blew
    // straight through the 32 MB rotation, taking the post-mortem file with it.
    //
    // Text bodies must stay WHOLE: reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug.
    try testing.expect(bodyIsText("{\"model\":\"x\",\"prompt\":\"hi\"}"));
    try testing.expect(bodyIsText("multi\nline\twith\r\nwhitespace"));
    try testing.expect(bodyIsText("utf8 stays text: é ✓ 🎉")); // multibyte, not control
    try testing.expect(!bodyIsText("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"));
    // The real shape: text multipart framing wrapped around a binary payload.
    try testing.expect(!bodyIsText("--B\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n\x89PNG\x00\x01\r\n--B--"));

    // A binary preview is bounded AND strictly printable ASCII.
    var buf: [64]u8 = undefined;
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xffbinary tail";
    const p = bodyPreview(&buf, png);
    try testing.expect(p.len <= buf.len);
    for (p) |c| try testing.expect(c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f));

    // Larger than the sink → truncated, never streamed.
    const big = try testing.allocator.alloc(u8, 1024 * 1024);
    defer testing.allocator.free(big);
    @memset(big, 0);
    try testing.expectEqual(@as(usize, buf.len), bodyPreview(&buf, big).len);
    // The cap is the SMALLER of the caller's buffer and the byte limit, so a
    // generous buffer can't reintroduce the megabyte dump.
    var huge: [BODY_LOG_LIMIT * 2]u8 = undefined;
    try testing.expectEqual(@as(usize, BODY_LOG_LIMIT), bodyPreview(&huge, big).len);
}

test "extractJsonField extracts array" {
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}
    ;
    const result = extractJsonField(body, "messages").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "extractJsonField extracts nested object" {
    const body =
        \\{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}
    ;
    const result = extractJsonField(body, "response_format").?;
    try testing.expect(std.mem.startsWith(u8, result, "{"));
    try testing.expect(std.mem.endsWith(u8, result, "}"));
}

test "extractJsonField returns null for missing field" {
    const body = "{\"messages\":[]}";
    try testing.expect(extractJsonField(body, "tools") == null);
}

test "extractJsonField handles escaped quotes in strings" {
    const body =
        \\{"tools":[{"type":"function","function":{"name":"say_\"hello\""}}]}
    ;
    const result = extractJsonField(body, "tools").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "jsonEscape basic string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "hello");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"hello\"", result);
}

test "jsonEscape special characters" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "line1\nline2\t\"quoted\"\\back");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"line1\\nline2\\t\\\"quoted\\\"\\\\back\"", result);
}

test "jsonEscape control characters" {
    const allocator = testing.allocator;
    const input = &[_]u8{ 0x01, 0x02 };
    const result = try jsonEscape(allocator, input);
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\\u0001\\u0002\"", result);
}

test "jsonEscape empty string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\"", result);
}

test "utf8TrailingIncomplete complete ASCII" {
    try testing.expectEqual(@as(usize, 0), utf8TrailingIncomplete("hello"));
}

test "utf8TrailingIncomplete complete multibyte" {
    // 🎉 = F0 9F 8E 89 (4-byte sequence, complete)
    try testing.expectEqual(@as(usize, 0), utf8TrailingIncomplete("\xF0\x9F\x8E\x89"));
}

test "utf8TrailingIncomplete partial 4-byte" {
    // First 3 bytes of a 4-byte sequence
    try testing.expectEqual(@as(usize, 3), utf8TrailingIncomplete("\xF0\x9F\x8E"));
    // First 2 bytes
    try testing.expectEqual(@as(usize, 2), utf8TrailingIncomplete("\xF0\x9F"));
    // First 1 byte
    try testing.expectEqual(@as(usize, 1), utf8TrailingIncomplete("\xF0"));
}

test "utf8TrailingIncomplete partial after complete" {
    // "hi" + first 2 bytes of emoji
    try testing.expectEqual(@as(usize, 2), utf8TrailingIncomplete("hi\xF0\x9F"));
}

test "utf8TrailingIncomplete empty" {
    try testing.expectEqual(@as(usize, 0), utf8TrailingIncomplete(""));
}

test "parseJsonFloat returns value when present" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":0.7}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 0.7), result, 0.001);
}

test "parseJsonFloat returns default when missing" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "parseJsonFloat clamps to range" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":5.0}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result, 0.001);
}

test "parseJsonFloat handles integer value" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":1}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 0.5, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "safeAutoContext leaves headroom below the memory ceiling and rounds to 1024" {
    // The raw ceiling is the largest context that FITS. Running at exactly that
    // leaves nothing for the prefix cache to grow into, a second model, or
    // another app — so admit only `auto_ctx_safety_pct` of it.
    try testing.expectEqual(@as(u32, 79872), safeAutoContext(94729)); // 85% = 80519 -> 1024-floor
    try testing.expectEqual(@as(u32, 27648), safeAutoContext(32768)); // 85% = 27852 -> 1024-floor
    // Never rounds down to zero on a tiny ceiling.
    try testing.expect(safeAutoContext(1000) > 0);
    try testing.expect(safeAutoContext(1) > 0);
    // Monotonic, and always strictly inside the ceiling for real-sized contexts.
    var prev: u32 = 0;
    for ([_]u32{ 4096, 8192, 16384, 32768, 65536, 94729, 262144 }) |raw| {
        const got = safeAutoContext(raw);
        try testing.expect(got <= raw);
        try testing.expect(got >= prev);
        prev = got;
    }
}

test "autoContextFor: the safety margin applies to MEMORY, never to the model's own max" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0;
    // The OS reserve alone eats a 7 GB CI runner's free RAM; it is not what is on trial.
    const orig_reserve = metrics.os_reserve_override;
    defer metrics.os_reserve_override = orig_reserve;
    metrics.os_reserve_override = 0;

    // A tiny model whose `max_position_embeddings` is far below anything memory
    // could constrain: it must get its FULL declared context, un-margined.
    // (Regression: applying 85% AFTER the model-max clamp shaved 15% off a
    // 131,072-token checkpoint that fits in RAM with room to spare.)
    var small = model_mod.ModelConfig{};
    small.max_position_embeddings = 4096;
    small.num_attention_heads = 8;
    small.num_hidden_layers = 4;
    small.num_key_value_heads = 2;
    small.head_dim = 64;
    small.hidden_size = 512;
    small.intermediate_size = 1024;
    try testing.expectEqual(@as(u32, 4096), autoContextFor(&small));

    // No declared max: fall back to the margined memory ceiling.
    var unbounded = small;
    unbounded.max_position_embeddings = 0;
    const got = autoContextFor(&unbounded);
    try testing.expect(got > 0);
    try testing.expectEqual(safeAutoContext(computeMemoryContext(&unbounded)), got);
}

test "getEffectiveContextLength returns the PINNED value, not a fresh memory reading" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0; // auto

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    // Once pinned, the advertised context must NOT move — clients budget
    // against it, and a fresh memory reading drifts with system load.
    config.pinned_context = 12345;
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));

    // ...but an explicit --ctx-size still wins.
    server_config.max_context_size = 4096;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "pinAutoContext is idempotent and a no-op under --ctx-size" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    server_config.max_context_size = 0;
    const first = pinAutoContext(&config);
    try testing.expect(first > 0);
    try testing.expectEqual(first, config.pinned_context);
    // Second call must not re-read memory and shift the value.
    try testing.expectEqual(first, pinAutoContext(&config));

    // Explicit --ctx-size: never overwrite, always report the override.
    var manual = model_mod.ModelConfig{};
    manual.max_position_embeddings = 131072;
    server_config.max_context_size = 8192;
    try testing.expectEqual(@as(u32, 8192), pinAutoContext(&manual));
    try testing.expectEqual(@as(u32, 0), manual.pinned_context);
}

test "clampMaxTokens honors the effective context (auto-pinned, not just --ctx-size)" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0; // auto: the old code never clamped here

    // 4096-token window, 4000-token prompt -> only 96 tokens of generation left.
    try testing.expectEqual(@as(u32, 96), clampMaxTokens(8192, 4000, 4096));
    // Comfortable budget is passed through untouched.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 100, 65536));
    // Prompt at/over the window still yields at least one token.
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 4096, 4096));
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 9000, 4096));
    // ctx 0 (unknown) = no limit.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 4000, 0));
}

test "getEffectiveContextLength uses ctx_size override" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 4096;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "getEffectiveContextLength computes safe default from GPU memory" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 0;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    // Should compute a value based on GPU memory, not hardcoded 16K
    const computed = getEffectiveContextLength(&config);
    try testing.expect(computed > 0);
    try testing.expect(computed <= config.max_position_embeddings);

    // Explicit --ctx-size overrides the computed default
    server_config.max_context_size = 32768;
    try testing.expectEqual(@as(u32, 32768), getEffectiveContextLength(&config));
}

test "safeContextForBudget reserves the hot-cache budget (2026-06-19 OOM regression)" {
    const GB: u64 = 1 << 30;
    // A footprint that OOM'd a 16 GB Mac: 32 layers, 4 kv heads, head_dim 256.
    const kv_per_tok: u64 = 32 * 2 * 4 * 256 * 2; // 131072
    const work_per_tok: u64 = 8 * 9216 * 2; //        147456
    const per_tok: u64 = kv_per_tok + work_per_tok;

    // Mid-session: ~6 GB resident (model + a partly-filled hot cache), 2 GB cache cap.
    const ws: u64 = 12 * GB; // ~Metal recommended working set on a 16 GB Mac
    const active: u64 = 6 * GB;
    const cache: u64 = 2 * GB;

    const with_reserve = safeContextForBudget(ws, active, cache, per_tok, 0);
    const without_reserve = safeContextForBudget(ws, active, 0, per_tok, 0);

    // The whole point of the fix: subtracting the hot-cache budget shrinks the
    // reported context so it still fits once the cache fills (it didn't before
    // — auto-ctx was computed against an empty cache and later collided with it).
    try testing.expect(with_reserve < without_reserve);
    // Still usable, never floored to nothing.
    try testing.expect(with_reserve > 1024);
}

test "parseWiredMarginGib accepts 2..32 and names a bad value" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 8) << 30, try parseWiredMarginGib("8"));
    try t.expectEqual(@as(u64, 6) << 30, try parseWiredMarginGib("6"));
    try t.expectEqual(@as(u64, 2) << 30, try parseWiredMarginGib("2"));
    try t.expectEqual(@as(u64, 32) << 30, try parseWiredMarginGib("32"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("1"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("33"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("nope"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib(""));
}

test "physicalMemoryCeiling caps the static GPU max by real free RAM (#64 docker OOM)" {
    const GB: u64 = 1 << 30;
    const static_max: u64 = 115 * GB; // Metal max_recommended_working_set_size, 128 GB Mac

    // Idle: MLX holds 26 GB (active) + 5 GB (reclaimable cache), ~84 GB free.
    // 31 + 84 = 115 ⇒ the static device max still binds — NO regression when the
    // machine is uncontended (auto-context stays generous).
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, 31 * GB, 84 * GB));

    // #64: a docker-compose stack (firecrawl/rabbitmq/postgres/playwright) holds
    // ~45 GB. MLX holds 26 GB, only ~35 GB is physically free. The static max is
    // BLIND to docker, but the ceiling must collapse to 26 + 35 = 61 GB — otherwise
    // a large MoE prefill is admitted and Metal throws an uncatchable OOM.
    try testing.expectEqual(@as(u64, 61 * GB), physicalMemoryCeiling(static_max, 26 * GB, 35 * GB));

    // Saturating add: a bogus/huge footprint reading can never wrap to a tiny ceiling.
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, std.math.maxInt(u64), GB));
}

test "auto-context tightens under external memory pressure vs the pre-fix static ceiling (#64)" {
    const GB: u64 = 1 << 30;
    // A 35B-A3B footprint (per-token KV + working envelope).
    const per_tok: u64 = 328 * 1024;
    const static_max: u64 = 115 * GB;
    const active: u64 = 26 * GB;
    const mlx_cache: u64 = 4 * GB;
    const hot_cache_reserve: u64 = 2 * GB;
    const max_pos: u32 = 262144;

    // Pre-fix behaviour: budget straight against the static device max, blind to
    // free RAM — the exact code path that OOM'd #64.
    const prefix_ctx = safeContextForBudget(static_max, active, hot_cache_reserve, per_tok, max_pos);

    // Post-fix, idle (~84 GB free): ceiling ≈ static max ⇒ context essentially unchanged.
    const idle_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 84 * GB);
    const idle_ctx = safeContextForBudget(idle_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(idle_ctx >= prefix_ctx * 9 / 10); // within 10% of the old generous value

    // Post-fix, docker holds ~45 GB (~35 GB free): ceiling collapses, so the safe
    // context is far smaller than the pre-fix static-max computation would allow.
    const busy_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 35 * GB);
    const busy_ctx = safeContextForBudget(busy_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(busy_ctx < prefix_ctx / 2);
    try testing.expect(busy_ctx > 1024); // still usable, never floored to nothing
}

test "safeContextForBudget floors when memory is exhausted and caps at max_pos" {
    const GB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    // working set already below what's resident → no room → minimum floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(4 * GB, 5 * GB, 0, per_tok, 0));
    // active + reserve exceeds the ceiling → floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 6 * GB, 4 * GB, per_tok, 0));
    // Plenty of memory, but capped at the model's max position embeddings.
    try testing.expectEqual(@as(u32, 4096), safeContextForBudget(256 * GB, 0, 0, per_tok, 4096));
    // Degenerate per_tok never divides by zero.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 0, 0, 0, 0));
    // A budget that fits a few tokens gets the same floor as one that fits none.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(6 * GB + 100 * per_tok, 6 * GB, 0, per_tok, 0));
}

test "context sizer: the prompt-cache reserve never takes more than half the room beside the weights" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = 28 * 1024;
    const transient: u64 = 300 * MiB;
    const reserve: u64 = 2048 * MiB;

    // 16 GB Mac, 8 GB pack: 1.8 GB beside the weights. A full 2 GB reserve left the context 870 tokens.
    const small = resolvedContextForLoad(0, 0, 10_200 * MiB, 8_100 * MiB, reserve, transient, per_tok, 262_144);
    try t.expect(small >= 16_384);

    // 128 GB Mac: the room dwarfs the reserve, which is billed in full as before.
    const big = resolvedContextForLoad(0, 0, 100_000 * MiB, 60_000 * MiB, reserve, transient, per_tok, 0);
    try t.expectEqual(autoContextFrom(safeContextForBudget(100_000 * MiB, 60_000 * MiB, reserve +| transient, per_tok, 0), 0), big);
}

test "kvBytesPerTokenAtBits: the KV bill follows the CONFIGURED quant width" {
    const t = std.testing;
    // A 27B dense: 16 caching layers x 4 kv heads x 256 head_dim x (K+V) x 2 B.
    const dense: u64 = 16 * 4 * 256 * 2 * 2; // 65536

    // Dense/off bills fp16 verbatim — no change for anyone not using --kv-quant.
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 16));
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 32));

    // Quantized widths carry the group scale/bias overhead (+0.5 bit/elem), the
    // SAME expression prefillMemoryNeeded bills — the sizer and the admission
    // guard must not disagree about what one token costs.
    try t.expectEqual(dense * 9 / 32, kvBytesPerTokenAtBits(dense, 4)); // 18432
    try t.expectEqual(dense * 17 / 32, kvBytesPerTokenAtBits(dense, 8));
    // 4-bit is what makes the 16 GB tier reachable: under a third of dense.
    try t.expect(kvBytesPerTokenAtBits(dense, 4) * 3 < dense);
}

test "auto-context never sizes past the ceiling as weights grow" {
    const t = std.testing;
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);
    var cfg = model_mod.ModelConfig{};
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    const reserve = prefillTransientReserve(&cfg, 4, 512);

    // Under-billing here ends in an uncatchable Metal OOM, not a 400: at every
    // weight size the reported context plus what is already spoken for has to
    // stay inside the ceiling, and the reported figure must fall as the pack
    // grows rather than staying pinned to the checkpoint's max.
    var prev: u32 = std.math.maxInt(u32);
    var weights: u64 = 2_000 * 1000 * 1000;
    while (weights < ceiling) : (weights += 1_000 * 1000 * 1000) {
        const ctx = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);
        try t.expect(weights + reserve + @as(u64, ctx) * per_tok <= ceiling or ctx == 1024);
        try t.expect(ctx <= prev);
        prev = ctx;
    }
    // Squeezed to nothing it floors rather than reporting a context that cannot exist.
    try t.expectEqual(@as(u32, 1024), safeContextForBudget(ceiling, ceiling, reserve, per_tok, 262144));
}

test "omittedMaxTokensDefault: context-bound when ctx is known, finite fallback otherwise" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    // OpenAI semantics: omitted max_tokens = generate until EOS bounded by the
    // context window. The old fixed 256 default broke agent clients (pi) whose
    // thinking-enabled turns hit `length` mid-reasoning on EVERY request.
    server_config.max_context_size = 32768;
    const sentinel = omittedMaxTokensDefault(32768);
    // Big enough that clampMaxTokens (the downstream bound) always wins…
    try testing.expect(sentinel > 32768);
    // …and the composition resolves to exactly the remaining context.
    try testing.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(sentinel, 1500, 32768));

    // ctx unknown (max_context_size=0): clampMaxTokens won't bound anything,
    // so the default itself must stay finite.
    server_config.max_context_size = 0;
    try testing.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(0));
}

test "resolveRequestMaxTokens: absent / 0 / negative / non-int → auto; positive → value" {
    const auto: u32 = 12345;
    // Absent → auto (the omitted-default path the app's "Auto" setting rides).
    try testing.expectEqual(auto, resolveRequestMaxTokens(null, auto));
    // 0 and negatives are "auto" too — a client must not be able to request a
    // 0-token (generate-nothing) response by selecting Auto.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = 0 }, auto));
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = -5 }, auto));
    // Wrong type (e.g. a string) → auto rather than crashing.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .string = "999" }, auto));
    // A real positive cap is honored verbatim.
    try testing.expectEqual(@as(u32, 4096), resolveRequestMaxTokens(.{ .integer = 4096 }, auto));
}

test "--max-tokens is the omitted-field default: unset is byte-identical, set wins" {
    const t = std.testing;
    // With no flag the omitted-field default is byte-for-byte what it was.
    try t.expectEqual(@as(u32, std.math.maxInt(u32) / 4), omittedMaxTokensDefaultWith(32768, 0));
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefaultWith(0, 0));

    // Flag set: the launch value outranks both fallbacks.
    try t.expectEqual(@as(u32, 65536), omittedMaxTokensDefaultWith(786_432, 65536));
    try t.expectEqual(@as(u32, 65536), omittedMaxTokensDefaultWith(0, 65536));

    // The context clamp still sits on top.
    try t.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(omittedMaxTokensDefaultWith(32768, 65536), 1500, 32768));

    // A request that sends max_tokens is untouched by the flag.
    const auto = omittedMaxTokensDefaultWith(786_432, 65536);
    try t.expectEqual(@as(u32, 128), resolveRequestMaxTokens(.{ .integer = 128 }, auto));
    try t.expectEqual(@as(u32, 200_000), resolveRequestMaxTokens(.{ .integer = 200_000 }, auto));
    try t.expectEqual(@as(u32, 65536), resolveRequestMaxTokens(.{ .integer = 0 }, auto));

    // The live wrapper reads the ServerConfig field through the one accessor.
    const slot = &@field(server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    try t.expectEqual(@as(u32, 0), launchMaxTokensDefault());
    try t.expectEqual(@as(u32, std.math.maxInt(u32) / 4), omittedMaxTokensDefault(32768));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), launchMaxTokensDefault());
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(32768));
}

test "describeMaxTokens: an omitted max_tokens logs as auto, never as the sentinel" {
    const t = std.testing;
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;

    try t.expectEqualStrings("auto", describeMaxTokens(&buf, AUTO_MAX_TOKENS_SENTINEL, .auto));

    // With an unknown context the auto budget is a finite 4096 that really caps generation.
    try t.expectEqualStrings("auto (4096)", describeMaxTokens(&buf, 4096, .auto));

    // `--max-tokens N` filled an omitted field: the number is real, but it is
    // the operator's, not the client's, and the line says whose it is.
    try t.expectEqualStrings("65536 (launch default)", describeMaxTokens(&buf, 65536, .launch_default));

    // A client that sent the field gets its own number back, unadorned.
    try t.expectEqualStrings("4096", describeMaxTokens(&buf, 4096, .client));

    // Origin comes from the SAME inputs `resolveRequestMaxTokens` reads, so
    // the label can't disagree with the number beside it.
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(null, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .integer = 0 }, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .integer = -5 }, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .string = "999" }, 0));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(null, 65536));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(.{ .integer = 0 }, 65536));
    // A sent field outranks the flag, so it is the CLIENT's number either way.
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 65536));
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 0));

    try t.expectEqualStrings(
        "1073741823 (launch default)",
        describeMaxTokens(&buf, AUTO_MAX_TOKENS_SENTINEL, .launch_default),
    );
}

test "the max_tokens squeeze warning is silent for the auto budget, loud for a real over-ask" {
    const t = std.testing;

    // A real over-ask: the client named a number and the window can't pay it.
    // That is the warning's whole point (the tool-call truncation nudge).
    try t.expect(maxTokensBudgetSqueezed(32768, 4000));
    try t.expect(!maxTokensBudgetSqueezed(32768, 30000));

    // The sentinel is always > 4x whatever remains; there is nothing to be short of.
    try t.expect(!maxTokensBudgetSqueezed(AUTO_MAX_TOKENS_SENTINEL, 4000));
    try t.expect(!maxTokensBudgetSqueezed(AUTO_MAX_TOKENS_SENTINEL, 1));

    try t.expect(autoBudgetWindowTight(4000, 32768));
    try t.expect(!autoBudgetWindowTight(30000, 32768));
    // A launch default is a real number and takes the real predicate.
    try t.expect(maxTokensBudgetSqueezed(65536, 4000));
}

test "an omitted max_tokens reserves the LAUNCH default, not the whole window" {
    const t = std.testing;
    // The admission reservation reads the resolved budget, so the flag reaches it.
    const KVCache = transformer_mod.KVCache;
    const ctx: u64 = 786_432;
    const seq: u64 = 400_000; // past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;

    // No flag: the sentinel becomes the whole remaining window; only the headroom cap keeps that off the bill.
    const no_flag = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 0), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, @intCast(ctx - seq)), no_flag);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // Flag set below the headroom: the reservation is prompt + the flag.
    const flagged = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 2048), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 2048), flagged);
    try t.expectEqual(seq + 2048 + chunk, KVCache.reservedTokens(seq, flagged, chunk, ctx));
    try t.expect(KVCache.reservedTokens(seq, flagged, chunk, ctx) < KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // A flag above the headroom bills only the headroom.
    const big = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 65536), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 65536), big);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, KVCache.reservedTokens(seq, big, chunk, ctx));
}

test "StreamHeartbeat: a write resets the deadline, silence expires it" {
    var hb = StreamHeartbeat{ .last_write_ms = 1_000, .interval_ms = 5_000 };

    // Silent, but not yet for a full interval.
    try testing.expect(!hb.due(1_000));
    try testing.expect(!hb.due(5_999));
    // Exactly the interval is due (the client's clock is not ours to trust).
    try testing.expect(hb.due(6_000));
    try testing.expect(hb.due(60_000));

    // Any byte written to the socket pushes the deadline out.
    hb.noteWrite(60_000);
    try testing.expect(!hb.due(60_001));
    try testing.expect(hb.due(65_000));
}

test "StreamHeartbeat: buffered tokens do NOT reset the deadline" {
    // The regression this guards: the old code beat only when NO token was
    // available (`.idle`). A tool call buffers tokens continuously while
    // writing nothing, so token arrival must not be mistaken for liveness —
    // only `noteWrite` (bytes on the socket) counts.
    var hb = StreamHeartbeat{ .last_write_ms = 0, .interval_ms = 5_000 };
    var now: i64 = 0;
    // 300 tokens arrive over 30s. Not one is written out (buffered tool call).
    var i: usize = 0;
    while (i < 300) : (i += 1) now += 100;
    try testing.expectEqual(@as(i64, 30_000), now);
    try testing.expect(hb.due(now));
}

test "clampMaxTokens no limit when ctx_size=0" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0;

    try testing.expectEqual(@as(u32, 1000), clampMaxTokens(1000, 500, 0));
}

test "clampMaxTokens clamps when would exceed" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 4096;

    // prompt=3000, max_tokens=2000 → clamp to 1096
    try testing.expectEqual(@as(u32, 1096), clampMaxTokens(2000, 3000, 4096));
}

test "clampMaxTokens no clamp when fits" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 4096;

    // prompt=100, max_tokens=200 → fits, no clamp
    try testing.expectEqual(@as(u32, 200), clampMaxTokens(200, 100, 4096));
}

test "clampMaxTokens at boundary" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 4096;

    // prompt=4096 → only 1 token allowed
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4096, 4096));
    // prompt=4095 → only 1 token remaining
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4095, 4096));
}

test "getTimeoutNs computes correctly" {
    const original = server_config.request_timeout_sec;
    defer server_config.request_timeout_sec = original;

    server_config.request_timeout_sec = 300;
    try testing.expectEqual(@as(u64, 300_000_000_000), getTimeoutNs());

    server_config.request_timeout_sec = 0;
    try testing.expectEqual(@as(u64, 0), getTimeoutNs());
}

test "a stream never starts inside a think block because the REQUEST asked for thinking" {
    // Every streaming surface seeded the think flag by OR-ing the request's
    // enable-thinking flag into the prompt-derived one (the needle below is
    // ++-split so this comment cannot satisfy the scan), on the assumption
    // that a thinking-enabled model
    // opens `<think>` as its first token. That is true for Gemma -
    // whose templates render the opener, which `promptOpensThink` then SEES in
    // the rendered bytes anyway - and false for a template without an opener, whose generation
    // prompt is a bare `<|im_start|>assistant\n` and whose model answers
    // directly. With `enable_thinking: true` the whole answer streamed as
    // `reasoning_content` and `content` came back EMPTY (live 2026-08-13: the
    // app renders it as a Thinking block with no reply under it). The same
    // request non-streaming was correct, because the non-streaming split has
    // always keyed on the prompt — so the two surfaces disagreed about the
    // bytes of one answer, which is the class this scan pins.
    //
    // Whether a stream starts inside a think block is a property of the
    // RENDERED PROMPT. A request flag is a request.
    const src = @embedFile("server.zig");
    const needle = "in_think_block = enable" ++ "_thinking";
    try testing.expect(std.mem.indexOf(u8, src, needle) == null);
    // …and the initialization that replaces it must still exist, or the scan
    // passes on a file that stopped tracking think state at all.
    var count: usize = 0;
    var i: usize = 0;
    const init = "in_think_block = prompt_opened_think";
    while (std.mem.indexOfPos(u8, src, i, init)) |at| : (i = at + init.len) count += 1;
    try testing.expect(count >= 2);
}

test "embeddings messages: an image, video or audio part gets the text-only 400" {
    const a = testing.allocator;
    const cases = [_][]const u8{
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"video_url\",\"video_url\":{\"frames\":[]}}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\"}}]}]",
        "[{\"role\":\"system\",\"content\":[{\"type\":\"image_url\",\"image_url\":\"data:x\"}]},{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    for (cases) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        // The handler's own pre-check, which runs before the model is consulted.
        try testing.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, embeddingMessagesMediaReject(parsed.value.array.items).?);
    }
    const text_only = try std.json.parseFromSlice(std.json.Value, a, "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}]", .{});
    defer text_only.deinit();
    try testing.expect(embeddingMessagesMediaReject(text_only.value.array.items) == null);
}

test "embeddings messages: every media part shape gets the text-only 400, a text file does not" {
    const a = testing.allocator;
    const media = [_][]const u8{
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AAAA\"}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"input_video\",\"video_url\":\"x\"}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"x\"}}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_file\",\"file_data\":\"data:image/png;base64,AAAA\"}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"file\",\"file\":{\"filename\":\"a.mp3\"}}]}]",
        "[{\"role\":\"user\",\"content\":\"earlier\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"image\"}]}]",
    };
    for (media) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, embeddingMessagesMediaReject(parsed.value.array.items).?);
    }
    // A text file part is not media.
    const file_body = "[{\"role\":\"user\",\"content\":[{\"type\":\"input_file\",\"filename\":\"a.txt\",\"file_data\":\"data:text/plain;base64,aGk=\"}]}]";
    const file_parsed = try std.json.parseFromSlice(std.json.Value, a, file_body, .{});
    defer file_parsed.deinit();
    try testing.expect(embeddingMessagesMediaReject(file_parsed.value.array.items) == null);
}

test "contentMediaMarks: chat content parts of every media kind are marked, text and text files are not" {
    const a = testing.allocator;
    const cases = [_]struct { body: []const u8, image: bool, video: bool, audio: bool }{
        .{ .body = "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]", .image = true, .video = false, .audio = false },
        .{ .body = "[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"input_image\"}]", .image = true, .video = false, .audio = false },
        .{ .body = "[{\"type\":\"video_url\"}]", .image = false, .video = true, .audio = false },
        .{ .body = "[{\"type\":\"input_video\"}]", .image = false, .video = true, .audio = false },
        .{ .body = "[{\"type\":\"input_audio\"}]", .image = false, .video = false, .audio = true },
        .{ .body = "[{\"type\":\"audio_url\"}]", .image = false, .video = false, .audio = true },
        .{ .body = "[{\"type\":\"file\",\"file\":{\"file_data\":\"data:image/png;base64,AAAA\"}}]", .image = true, .video = false, .audio = false },
        .{ .body = "[{\"type\":\"input_file\",\"filename\":\"a.wav\"}]", .image = false, .video = false, .audio = true },
        .{ .body = "{\"type\":\"image_url\"}", .image = true, .video = false, .audio = false },
        .{ .body = "[{\"type\":\"text\",\"text\":\"a\"}]", .image = false, .video = false, .audio = false },
        .{ .body = "[{\"type\":\"file\",\"file\":{\"filename\":\"a.pdf\",\"file_data\":\"data:application/pdf;base64,AAAA\"}}]", .image = false, .video = false, .audio = false },
        .{ .body = "\"plain string\"", .image = false, .video = false, .audio = false },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        const marks = contentMediaMarks(parsed.value);
        try testing.expectEqual(c.image, marks.image);
        try testing.expectEqual(c.video, marks.video);
        try testing.expectEqual(c.audio, marks.audio);
    }
}

test "embeddings messages: content given as one bare media object gets the text-only 400, a bare text object does not" {
    const a = testing.allocator;
    const media = [_][]const u8{
        "[{\"role\":\"user\",\"content\":{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AAAA\"}}]",
        "[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":{\"type\":\"audio_url\"}}]",
        "[{\"role\":\"system\",\"content\":{\"type\":\"input_file\",\"filename\":\"a.mp4\"}},{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    for (media) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, embeddingMessagesMediaReject(parsed.value.array.items).?);
    }
    // A bare text object is not media.
    const text_body = "[{\"role\":\"user\",\"content\":{\"type\":\"text\",\"text\":\"hi\"}}]";
    const text_parsed = try std.json.parseFromSlice(std.json.Value, a, text_body, .{});
    defer text_parsed.deinit();
    try testing.expect(embeddingMessagesMediaReject(text_parsed.value.array.items) == null);
}

test "messageEntryHasMedia: media counts whatever the entry's role, a text entry without a role does not" {
    const a = testing.allocator;
    const media = [_][]const u8{
        "{\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}",
        "{\"role\":7,\"content\":{\"type\":\"input_audio\"}}",
        "{\"role\":\"narrator\",\"content\":[{\"type\":\"video_url\"}]}",
        "{\"role\":\"user\",\"content\":[{\"type\":\"input_file\",\"file_data\":\"data:image/png;base64,AAAA\"}]}",
    };
    for (media) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expect(messageEntryHasMedia(parsed.value));
    }
    const plain = [_][]const u8{
        "{\"content\":[{\"type\":\"text\",\"text\":\"a\"}]}",
        "{\"role\":7,\"content\":\"hi\"}",
        "{\"role\":\"user\"}",
        "{\"role\":\"user\",\"content\":null}",
        "[1,2]",
        "3",
    };
    for (plain) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expect(!messageEntryHasMedia(parsed.value));
    }
}

test "chatRequestMediaReject: nested media is refused whatever n or the roles say, text is not" {
    const a = testing.allocator;
    const media = [_][]const u8{
        "{\"n\":5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_result\",\"content\":{\"type\":\"input_video\"}}]}]}",
        "{\"messages\":[{\"content\":{\"type\":\"tool_result\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}}]}",
        "{\"messages\":[{\"role\":\"tool\",\"content\":[{\"type\":\"input_audio\"}]}]}",
    };
    for (media) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, chatRequestMediaReject(parsed.value.object).?);
    }
    const plain = [_][]const u8{
        "{\"n\":5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "{\"messages\":[{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_result\",\"content\":\"ok\"}]}]}",
        "{\"model\":\"m\"}",
        "{\"messages\":7}",
    };
    for (plain) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expect(chatRequestMediaReject(parsed.value.object) == null);
    }
}

test "embeddingsRequestMediaReject: media in messages or input is refused whatever dimensions say, text is not" {
    const a = testing.allocator;
    const media = [_][]const u8{
        "{\"dimensions\":-1,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]}",
        "{\"input\":\"x\",\"messages\":[{\"content\":{\"type\":\"tool_result\",\"content\":{\"type\":\"input_video\"}}}]}",
        "{\"input\":[\"a\",{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}",
        "{\"input\":{\"type\":\"input_audio\"}}",
    };
    for (media) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings(TEXT_ONLY_MEDIA_REJECT, embeddingsRequestMediaReject(parsed.value).?);
    }
    const plain = [_][]const u8{
        "{\"dimensions\":-1,\"input\":[\"a\",\"b\"]}",
        "{\"input\":\"a\"}",
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"}]}]}",
        "{}",
        "[1,2]",
    };
    for (plain) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        try testing.expect(embeddingsRequestMediaReject(parsed.value) == null);
    }
}

/// One top-level `fn <name>(` of this file, up to its closing brace, for ordering checks.
fn mediaOrderHandlerBody(comptime name: []const u8) ![]const u8 {
    const src = @embedFile("server.zig");
    const at = std.mem.indexOf(u8, src, "\nfn " ++ name ++ "(") orelse return error.HandlerNotFound;
    const end = std.mem.indexOfPos(u8, src, at + 1, "\n}\n") orelse return error.HandlerEndNotFound;
    return src[at..end];
}

fn expectMediaCheckFirst(body: []const u8, media_call: []const u8, laters: []const []const u8) !void {
    const media_at = std.mem.indexOf(u8, body, media_call) orelse return error.MediaCheckMissing;
    for (laters) |later| {
        const later_at = std.mem.indexOf(u8, body, later) orelse return error.LaterCheckMissing;
        try testing.expect(media_at < later_at);
    }
}

test "media ordering: every handler refuses media before its other early returns" {
    try expectMediaCheckFirst(try mediaOrderHandlerBody("handleChatCompletions"), "chatRequestMediaReject(root)", &.{
        "nChoicesRejectReason(root)",
        "'messages' is a required field",
        "No valid messages found in request",
    });
    try expectMediaCheckFirst(try mediaOrderHandlerBody("handleEmbeddings"), "embeddingsRequestMediaReject(", &.{
        "runs on an embedded engine",
        "'dimensions' must be a positive integer",
        "Send either 'input' or 'messages'",
    });
}

test "handleMessagesEmbedding: a Gemma 4 LM answers the three messages 400s, in order" {
    const body = try mediaOrderHandlerBody("handleMessagesEmbedding");
    try expectMediaCheckFirst(body, "'messages' must be a non-empty array", &.{
        "embeddingMessagesMediaReject(",
        "'messages' is not supported by this model; send the text in 'input'",
    });
    try expectMediaCheckFirst(body, "embeddingMessagesMediaReject(", &.{
        "'messages' is not supported by this model; send the text in 'input'",
    });
}

test "content-array text parts JOIN in order — plan-mode's [prompt, reminder] shape (issue #195)" {
    // opencode plan mode appends its system-reminder as a SECOND text part on
    // the user message. Last-wins parsing kept only the reminder, so the model
    // saw no user prompt at all.
    const body =
        \\[{"type":"text","text":"analyze project"},
        \\ {"type":"image_url","image_url":{"url":"data:x"}},
        \\ {"type":"text","text":"<system-reminder>plan mode</system-reminder>"}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    defer if (j.owned) testing.allocator.free(j.text);
    try testing.expect(j.owned);
    try testing.expectEqualStrings("analyze project\n<system-reminder>plan mode</system-reminder>", j.text);
}

test "joinedTextParts: single text part borrows; empty and non-text parts ignored" {
    const body =
        \\[{"type":"text","text":""}, {"type":"input_audio"}, {"type":"text","text":"hello"}, {"nope":1}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    try testing.expect(!j.owned);
    try testing.expectEqualStrings("hello", j.text);

    const none = try joinedTextParts(testing.allocator, &.{});
    try testing.expect(!none.owned);
    try testing.expectEqualStrings("", none.text);
}

fn cacheKeyOf(body: []const u8) !u64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    return requestCacheKey(parsed.value.object);
}

test "canonicalRole: developer is the system turn; every other role is itself" {
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("developer"));
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("system"));
    try testing.expectEqualStrings("tool", chat_mod.canonicalRole("tool"));
    try testing.expectEqualStrings("assistant", chat_mod.canonicalRole("assistant"));
}

test "requestCacheKey: prompt_cache_key > metadata.user_id > system prompt > anonymous" {
    const sys_str = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are S."},{"role":"user","content":"hi"}]}
    );
    const sys_parts = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":[{"type":"text","text":"You are S."}]}]}
    );
    const anth = try cacheKeyOf(
        \\{"system":[{"type":"text","text":"You are S."}],"messages":[{"role":"user","content":"hi"}]}
    );
    const resp = try cacheKeyOf(
        \\{"instructions":"You are S.","input":"hi"}
    );
    try testing.expect(sys_str != 0);
    try testing.expectEqual(sys_str, sys_parts);
    try testing.expectEqual(sys_str, anth);
    try testing.expectEqual(sys_str, resp);
    // OpenAI's reasoning-model spelling of the system turn (pi sends it).
    try testing.expectEqual(sys_str, try cacheKeyOf(
        \\{"messages":[{"role":"developer","content":"You are S."},{"role":"user","content":"hi"}]}
    ));
    try testing.expect(sys_str != try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are T."}]}
    ));

    const uid = try cacheKeyOf(
        \\{"metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(uid != 0 and uid != sys_str);
    const pck = try cacheKeyOf(
        \\{"prompt_cache_key":"batch","metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(pck != 0 and pck != uid);
    try testing.expectEqual(pck, try cacheKeyOf(
        \\{"prompt_cache_key":"batch","prompt":"doc"}
    ));

    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"messages":[{"role":"user","content":"hi"}]}
    ));
    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"prompt_cache_key":"","prompt":"doc"}
    ));
}

// --- /props payload regression ------------------------------------------------
//
// The `chat_template` field used to be emitted by /props for llama.cpp
// server-compat clients. Our own Swift app never read it, and stock chat
// templates are 10–100 KB of Jinja — that's wasted bandwidth on every poll.
// `renderPropsBody` is the pure body-builder behind `handleProps`; these
// tests pin the schema so a future "let's add it back" can't slip through
// without flipping these assertions intentionally.

test "renderPropsBody omits chat_template" {
    var config = model_mod.ModelConfig{};
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;
    config.model_type = "gemma4";

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384, 4321, 7777, 8888, "");
    defer testing.allocator.free(body);

    try testing.expect(std.mem.indexOf(u8, body, "\"chat_template\"") == null);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    parsed.deinit();
}

test "queryModel: GET /props?model=<id> routes to that model, percent-decoded" {
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings("mlx-community/Example-0.8B", queryModel(&buf, "/props?x=1&model=mlx-community%2FExample-0.8B").?);
    try testing.expect(queryModel(&buf, "/props") == null);
    try testing.expect(queryModel(&buf, "/props?models=a") == null);
}

test "settingsPropsJson: /props names the effective serving settings a benchmark ran under" {
    const frag = try settingsPropsJson(testing.allocator, .{
        .engine = "mlx",
        .kv_quant = "8",
        .kv_attn_mode = .auto,
        .decode_attn_quant = true,
        .prefill_chunk = 8192,
        .drafter = "none",
        .pld = .{ .enable = false, .draft_len = 5, .key_len = 3 },
        .max_concurrent = 4,
        .prefix_cache_mem_bytes = 2048,
        .prefix_cache_disk_bytes = 0,
    });
    defer testing.allocator.free(frag);
    var config = model_mod.ModelConfig{};
    config.model_type = "gemma4";
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const st = (parsed.value.object.get("settings") orelse return error.MissingSettings).object;
    try testing.expectEqualStrings(build_options.version, st.get("version").?.string);
    try testing.expectEqualStrings("8", st.get("kv_quant").?.string);
    try testing.expectEqualStrings("auto", st.get("kv_attn_mode").?.string);
    try testing.expect(st.get("decode_attn_quant").?.bool);
    try testing.expect(st.get("mtp") == null);
    try testing.expectEqual(@as(i64, 4), st.get("max_concurrent").?.integer);

    const second = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "8", .kv_attn_mode = .dense, .decode_attn_quant = false, .prefill_chunk = 4096, .drafter = "dflash", .pld = PldDefaults.off, .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0 });
    defer testing.allocator.free(second);
    var sp = try std.json.parseFromSlice(std.json.Value, testing.allocator, second[",\"settings\":".len..], .{});
    defer sp.deinit();
    try testing.expectEqualStrings("dflash", sp.value.object.get("drafter").?.string);
    try testing.expect(sp.value.object.get("mtp") == null);
}

test "settingsPropsJson: /props reports the prefill decode share" {
    const frag = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "off", .kv_attn_mode = .auto, .decode_attn_quant = false, .prefill_chunk = 8192, .drafter = "none", .pld = PldDefaults.off, .max_concurrent = 8, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0, .prefill_decode_share = 0.5 });
    defer testing.allocator.free(frag);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
    defer parsed.deinit();
    const v = parsed.value.object.get("prefill_decode_share") orelse return error.MissingShare;
    try testing.expectApproxEqAbs(@as(f64, 0.5), v.float, 1e-6);
}

test "renderPropsBody keeps fields the Swift app + integration tests rely on" {
    var config = model_mod.ModelConfig{};
    config.model_type = "gemma4";
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384, 4321, 7777, 8888, "");
    defer testing.allocator.free(body);

    // Hit every field a known consumer reads.
    try testing.expect(std.mem.indexOf(u8, body, "\"n_ctx\":4096") != null); // integration_test.sh
    try testing.expect(std.mem.indexOf(u8, body, "\"total_slots\":1") != null); // integration_test.sh
    try testing.expect(std.mem.indexOf(u8, body, "\"model_info\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"memory\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"active_bytes\":1234") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"peak_bytes\":5678") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"available_bytes\":9000000000") != null); // Swift fetchProps (Free RAM line)
    try testing.expect(std.mem.indexOf(u8, body, "\"max_safe_context\":16384") != null); // Swift fetchProps
    // #110: the panel showed 19.6 GB (active) while the process held 81 GB.
    // The missing 61 GB was MLX's reclaimable buffer pool, which nothing we
    // expose reported — so the bug was invisible from every surface.
    try testing.expect(std.mem.indexOf(u8, body, "\"cache_bytes\":4321") != null); // Swift fetchProps
    // The tray splits active_bytes into loaded weights, the prefix KV cache and the rest.
    try testing.expect(std.mem.indexOf(u8, body, "\"weights_bytes\":7777") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"kv_cache_bytes\":8888") != null);
}

test "mlxCacheLimitBytes: RAM-proportional cap, 2 GB floor, 8 GB ceiling" {
    const GB: u64 = 1 << 30;
    // MLX's own default is min(1.5 x working_set, 0.95 x RAM) — ~121 GB on a
    // 128 GB Mac (`backend/metal/allocator.cpp`), i.e. it will not trim until
    // the machine is already dead. RAM/16 keeps the pool big enough for
    // step-to-step buffer reuse without letting it become the footprint.
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(128 * GB));
    try testing.expectEqual(4 * GB, mlxCacheLimitBytes(64 * GB));
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(32 * GB)); // 2 GB exactly
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(16 * GB)); // floor
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(8 * GB)); // floor
    // Ceiling holds above 128 GB (Mac Studio / M3 Ultra 512 GB).
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(512 * GB));
    // A failed `hw.memsize` read is 0 — never clamp on bad data.
    try testing.expectEqual(@as(u64, 0), mlxCacheLimitBytes(0));
}

test "mlxCacheLimitFromEnv: explicit bytes win, 0 disables, garbage falls through" {
    const GB: u64 = 1 << 30;
    // The A/B off-switch: MLX_SERVE_CACHE_LIMIT=0 leaves MLX's default in
    // place so the pre-#110 behavior stays reachable in a same-boot A/B.
    try testing.expectEqual(@as(u64, 0), mlxCacheLimitFromEnv("0", 128 * GB));
    try testing.expectEqual(@as(u64, 3 * GB), mlxCacheLimitFromEnv("3221225472", 128 * GB));
    // Unset or unparseable → the RAM-proportional default.
    try testing.expectEqual(8 * GB, mlxCacheLimitFromEnv(null, 128 * GB));
    try testing.expectEqual(8 * GB, mlxCacheLimitFromEnv("lots", 128 * GB));
    try testing.expectEqual(8 * GB, mlxCacheLimitFromEnv("", 128 * GB));
}

test "prefix cache default capacity covers interleaved agent flows" {
    // Claude Code-style clients interleave several conversation roots (main
    // thread, subagents, title generation). With capacity 1, every
    // interleaved request evicted the long system-prompt prefix and forced a
    // full re-prefill per turn. The byte budget (prefix_cache_mem_bytes)
    // still bounds memory.
    try testing.expect(prefix_cache_capacity >= 4);
    try testing.expect(resolvedPrefixCacheMem() > 0);
}

test "parseToolCallsForRequest coerces args to the schema (server-side chokepoint wiring)" {
    // The chokepoint every HTTP surface routes through must actually invoke the
    // schema coercion — not just parse. This pins the WIRING (a mis-wire that
    // parsed but skipped coerce would reintroduce the string-"False" bug on every
    // surface at once). Uses the exact Hermes-XML `replace_all=False` shape from
    // the live Claude Code failure.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .bool); // coerced from the STRING "False"
    try std.testing.expectEqual(false, ra.bool);
    // Null tools_json path must still parse (no coercion, no crash).
    const calls2 = try parseToolCallsForRequest(allocator, text, null, true);
    defer if (calls2) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls2 != null);
}

test "parseToolCallsForRequest hoists a misplaced required param (server-side chokepoint wiring)" {
    // Live 2026-07-13 pi session, gemma-4-26B-A4B: the model buried `path` inside
    // each edits[] item and emitted none at the top level, so pi rejected three
    // consecutive multi-thousand-token generations with
    //   Validation failed for tool "edit": - path: must have required properties path
    // This pins the WIRING — every HTTP surface reaches the hoist through this one
    // function, and the hoist must run BEFORE coercion so the lifted value is
    // type-checked like any other top-level arg.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]}}},"required":["path","edits"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"edit\",\"arguments\":{\"edits\":[{\"newText\":\"new\",\"oldText\":\"old\"," ++
        "\"path\":\"us_presidents/generate_site.sh\"}]}}</tool_call>";

    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try std.testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    try std.testing.expect(parsed.value.object.get("edits").?.array.items[0].object.get("path") == null);
    try std.testing.expect(chat_mod.toolCallConformsToSchema(allocator, calls[0], tools));

    // The escape hatch covers this correction too: it repairs the MODEL's output
    // (unlike the inferred-name filter, which corrects our own heuristic).
    g_tool_autocorrect = false;
    defer g_tool_autocorrect = true;
    const raw_calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (raw_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(raw_calls);
    }
    const raw_parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_calls[0].arguments, .{});
    defer raw_parsed.deinit();
    try std.testing.expect(raw_parsed.value.object.get("path") == null); // verbatim: still buried
}

test "parseToolCallsForRequest: --no-tool-autocorrect leaves args verbatim" {
    // The escape hatch: with g_tool_autocorrect off, the schema coercion is
    // skipped and the model's value passes through EXACTLY (here the STRING
    // "False", which is what a strict client would then reject — the user's
    // choice). Parse-repair + valid-JSON safety net still run, so args stay
    // well-formed JSON regardless.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";

    g_tool_autocorrect = false;
    defer g_tool_autocorrect = true; // restore the default for other tests
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .string); // NOT coerced — verbatim
    try std.testing.expectEqualStrings("False", ra.string);
}

test "parseToolCallsForRequest: truncated DATA object is not promoted to a tool call (George Washington class)" {
    // Live pi capture 2026-07-13 (a distilled model): generation hit
    // max_tokens midway through a presidents data script. The raw-JSON fallback
    // found the first balanced object — {"name": "George Washington", "num": 1,
    // …} — and the flat-shape synthesis promoted it to a TOOL CALL named
    // "George Washington" (args = every key but "name"). pi answered "Tool
    // George Washington not found", the model retried the identical mega-write,
    // and the session burned two full 16K-token turns making zero progress.
    // A HEURISTICALLY inferred call (no tag syntax — the model never said
    // "tool call") must carry a name the request actually declared; otherwise
    // the text stays visible and finish_reason="length" reaches the client
    // untouched, so its truncation recovery fires instead of a bogus tool loop.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "Now let me build a generator script that creates all 46 president pages with real historical data:\n\n" ++
        "presidents = [\n" ++
        "  {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None (Federalist-leaning)\", \"term\": \"1789\u{2013}1797\", \"vice\": \"John Adams\"},\n" ++
        "  {\"name\": \"John Adams\", \"num\": 2, \"party\": \"Federalist\",";
    const calls = try parseToolCallsForRequest(allocator, text, tools, true);
    defer if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls == null);
}

test "parseToolCallsForRequest: raw-JSON call with a DECLARED name still parses" {
    // The counterweight to the George Washington guard: models without a
    // trained tool format emit fenced raw-JSON calls, and those must
    // keep working when the name matches a declared tool.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    ;
    const text = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n```";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("write", calls[0].name);
}

test "parseToolCallsForRequest: tag-format call with an UNDECLARED name is kept" {
    // An EXPLICIT tag-format call (<tool_call>…) to a name the request never
    // declared still goes to the client — "tool not found" is model-visible
    // feedback the model can correct from. Only HEURISTIC raw-JSON inference
    // gets schema-name validation; this pins the filter against over-reach.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"searchWeb\",\"arguments\":{\"q\":\"zig\"}}</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("searchWeb", calls[0].name);
}

test "parseToolCallsForRequest: parallel_tool_calls=false clamps to the FIRST call" {
    // OpenAI contract: parallel_tool_calls=false means AT MOST ONE call per
    // response (the SDK sets it in strict structured-output mode; those
    // clients execute tool_calls[0] and assume length 1). We can't constrain
    // generation, so the chokepoint clamps post-parse: keep the first, drop
    // the rest — the model re-issues them next round after seeing the first
    // result. Whole valid calls only; nothing partial ships.
    const allocator = std.testing.allocator;
    const text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>\n" ++
        "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"b\"}}</tool_call>";

    // allow_parallel=true (default wiring): both calls survive.
    const both = (try parseToolCallsForRequest(allocator, text, null, true)).?;
    defer {
        for (both) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(both);
    }
    try std.testing.expectEqual(@as(usize, 2), both.len);

    // allow_parallel=false: only the FIRST survives, intact.
    const one = (try parseToolCallsForRequest(allocator, text, null, false)).?;
    defer {
        for (one) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(one);
    }
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqualStrings("read", one[0].name);

    // A single call with the flag off passes through untouched.
    const single_text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>";
    const single = (try parseToolCallsForRequest(allocator, single_text, null, false)).?;
    defer {
        for (single) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(single);
    }
    try std.testing.expectEqual(@as(usize, 1), single.len);
}

test "parseToolCallsForRequest: coercion fires across think on/off × hermes/gemma" {
    // The tool-call parse strips the leading think block BEFORE parsing args, and
    // the two families strip DIFFERENTLY (`<think>…</think>`, Gemma
    // `<|channel>thought…<channel|>`). Pin that a preceding reasoning block never
    // blocks the schema coercion, for BOTH families, WITH and WITHOUT thinking.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const gtools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;

    const Case = struct { name: []const u8, text: []const u8, tools: []const u8 };
    const cases = [_]Case{
        // Thinking OFF - Hermes XML, no reasoning block.
        .{ .name = "hermes think-off", .tools = tools, .text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Thinking ON - a `<think>…</think>` block precedes the call.
        .{ .name = "hermes think-on", .tools = tools, .text = "<think>\nThe user wants replace_all disabled, so I'll pass it false.\n</think>\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Gemma thinking OFF — custom `call:` form, no channel block.
        .{ .name = "gemma think-off", .tools = gtools, .text = "<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
        // Gemma thinking ON — a `<|channel>thought…<channel|>` block precedes.
        .{ .name = "gemma think-on", .tools = gtools, .text = "<|channel>thought\nI should disable replace_all for a single edit.<channel|>\n<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
    };

    for (cases) |c| {
        const calls = (try parseToolCallsForRequest(allocator, c.text, c.tools, true)) orelse {
            std.debug.print("\n[{s}] no tool call parsed\n", .{c.name});
            return error.NoToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try std.testing.expectEqualStrings("Edit", calls[0].name);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        const ra = parsed.value.object.get("replace_all") orelse {
            std.debug.print("\n[{s}] replace_all missing in {s}\n", .{ c.name, calls[0].arguments });
            return error.ArgMissing;
        };
        if (ra != .bool or ra.bool != false) {
            std.debug.print("\n[{s}] replace_all NOT coerced to bool false: {s}\n", .{ c.name, calls[0].arguments });
            return error.NotCoerced;
        }
    }
}

test "toolCallFinishReason preserves truncation over parsed tool calls" {
    // A truncated generation ("length": max_tokens OR request timeout) whose
    // cut-off text still salvaged a partial tool call must KEEP reporting
    // "length" — clients key their truncation recovery (chunk-and-retry
    // nudges) on it. Reporting "tool_calls" hides the cut: the client
    // executes a half-argument call and tells the model IT made a mistake.
    // Live capture 2026-07-03: a 33KB writeFile guillotined by the
    // 300s default timeout arrived as {"path":...} with finish "tool_calls".
    try std.testing.expectEqualStrings("length", toolCallFinishReason("length"));
    // Everything else stays the normal tool-call finish.
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("stop"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("tool_calls"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("client_disconnect"));
}

test "repetition-loop cuts decline tool-call parsing" {
    // The loop detector can fire inside a tool argument. The response must
    // remain a safe terminal stop without emitting an executable fragment.
    try std.testing.expect(shouldParseToolCalls(null));
    try std.testing.expect(!shouldParseToolCalls("repetition_loop"));
    try std.testing.expect(shouldParseToolCalls("unknown_future_detail"));
}

test "every OpenAI-shaped finish_reason emitter also carries finish_details" {
    // Dispatch-hole class: a surface that reports the reason and drops the
    // cause is silent — the response still validates, and no output-equality
    // test can see the missing field (the two
    // hardcoded `use_drafter=false` call sites lived for a month this way).
    // Needles are split with `++` so this test's own source can't match them.
    const t = std.testing;
    const src = @embedFile("server.zig");

    // The two literals that print a concrete reason: chat non-stream (plain
    // + tool-calls). Every one must be followed immediately by the details slot.
    const quoted = "\"finish_rea" ++ "son\":\"{s}\"";
    var seen: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, src, i, quoted)) |at| : (i = at + quoted.len) {
        seen += 1;
        try t.expectEqualStrings("{s}", src[at + quoted.len .. at + quoted.len + 3]);
    }
    try t.expectEqual(@as(usize, 2), seen);

    // The chat SSE chunk builds its reason separately (it may be JSON null
    // mid-stream), so it takes the details slot right after that field.
    const chunk = "\"finish_rea" ++ "son\":{s}{s}";
    try t.expect(std.mem.indexOf(u8, src, chunk) != null);

}

test "finishDetailsField: the loop cause rides beside finish_reason, and only a known cause reaches the wire" {
    // Absent = the field is not emitted at all, so every ordinary response
    // is byte-identical to what it was before this existed.
    try std.testing.expectEqualStrings("", finishDetailsField("stop", null));
    try std.testing.expectEqualStrings(
        ",\"finish_details\":{\"type\":\"repetition_loop\"}",
        finishDetailsField("stop", "repetition_loop"),
    );
    // An unknown value is DROPPED rather than interpolated: this string is
    // spliced into a JSON literal, and a literal is arbitrary bytes too (the
    // media-gen `sendError` class). A future cause adds an arm here.
    try std.testing.expectEqualStrings("", finishDetailsField("stop", "something new"));
    try std.testing.expectEqualStrings("", finishDetailsField("stop", "\",\"x\":\""));
    // The cause belongs only beside the intentional loop stop.
    try std.testing.expectEqualStrings("", finishDetailsField("length", "repetition_loop"));
    try std.testing.expectEqualStrings("", finishDetailsField("tool_calls", "repetition_loop"));
    try std.testing.expectEqualStrings("", finishDetailsField("client_disconnect", "repetition_loop"));
}

test "loopTrimmedIds: the degenerate span is cut, and a bad index degrades to emitting everything" {
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    // No cut → untouched (identity for every non-loop response).
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, null));
    try std.testing.expectEqualSlices(u32, ids[0..3], loopTrimmedIds(&ids, 3));
    // A whole-generation loop leaves nothing: honest, and finish_reason
    // "length" already says the server cut it.
    try std.testing.expectEqualSlices(u32, ids[0..0], loopTrimmedIds(&ids, 0));
    // The index comes from the generator's own token list, so a mismatch
    // must emit everything rather than slice out of bounds.
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, 5));
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, 99));
}

test "echoModelName: a client model name the templates cannot splice falls back to the model type" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, want: []const u8 }{
        .{ .body = "{\"model\":\"gpt-4\"}", .want = "gpt-4" },
        .{ .body = "{\"model\":\"org\\/repo\"}", .want = "org/repo" },
        .{ .body = "{\"model\":\"漢字\"}", .want = "漢字" },
        .{ .body = "{}", .want = "default-model" },
        .{ .body = "{\"model\":5}", .want = "default-model" },
        .{ .body = "{\"model\":\"a\\u0000b\"}", .want = "default-model" },
        .{ .body = "{\"model\":\"a\\\"b\"}", .want = "default-model" },
        .{ .body = "{\"model\":\"a\\\\b\"}", .want = "default-model" },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings(case.want, echoModelName(parsed.value.object, "default-model"));
    }
}

test "parseTokenIds: only integers inside the vocabulary reach the decoder" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, ok: bool }{
        .{ .body = "[]", .ok = true },
        .{ .body = "[0,99]", .ok = true },
        .{ .body = "[-1]", .ok = false },
        .{ .body = "[100]", .ok = false },
        .{ .body = "[1099511627776]", .ok = false },
        .{ .body = "[1.5]", .ok = false },
        .{ .body = "[\"a\",null]", .ok = false },
    };
    var out: [4]u32 = undefined;
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const items = parsed.value.array.items;
        try std.testing.expectEqual(case.ok, parseTokenIds(items, 100, out[0..items.len]));
    }
}

test "nChoicesRejectReason: n>1 earns an honest 400, single-choice spellings pass" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, rejected: bool }{
        // Absent, explicit null, and the two spellings of "one choice" pass.
        .{ .body = "{}", .rejected = false },
        .{ .body = "{\"n\":null}", .rejected = false },
        .{ .body = "{\"n\":1}", .rejected = false },
        .{ .body = "{\"n\":1.0}", .rejected = false },
        // Anything else is a request for multi-choice (or garbage) — reject
        // instead of the silent one-choice no-op.
        .{ .body = "{\"n\":2}", .rejected = true },
        .{ .body = "{\"n\":0}", .rejected = true },
        .{ .body = "{\"n\":\"2\"}", .rejected = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const reason = nChoicesRejectReason(parsed.value.object);
        try std.testing.expectEqual(case.rejected, reason != null);
    }
}

test "jsonEscapeOrEmpty: escaping an empty string is OWNED (leak class 2026-07-19)" {
    // The old inline pattern decided ownership by CONTENT
    // (`if (!mem.eql(escaped, "\"\"")) free`), so a legitimately-allocated
    // escape of "" — every all-reasoning generation with empty visible
    // content — matched the OOM-fallback literal and leaked 2 bytes per
    // request. Ownership keys on provenance; std.testing.allocator fails
    // this test on any leak.
    const a = std.testing.allocator;
    const empty = jsonEscapeOrEmpty(a, "");
    try std.testing.expect(empty.owned);
    try std.testing.expectEqualStrings("\"\"", empty.slice);
    if (empty.owned) a.free(empty.slice);

    const text = jsonEscapeOrEmpty(a, "hi \"there\"");
    try std.testing.expect(text.owned);
    try std.testing.expectEqualStrings("\"hi \\\"there\\\"\"", text.slice);
    if (text.owned) a.free(text.slice);
}

test "parseReasoningEffort: standard chat reasoning_effort opt-in maps to thinking + budget" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, expect: ?ReasoningEffort }{
        // Absent or wrong type → null: the vendor enable_thinking flag stays
        // in charge and nothing changes for existing clients.
        .{ .body = "{}", .expect = null },
        .{ .body = "{\"reasoning_effort\":42}", .expect = null },
        // "none" is an explicit OFF (gpt-5.1 default), never an enable.
        .{ .body = "{\"reasoning_effort\":\"none\"}", .expect = .{ .enable = false, .budget = -1 } },
        // Known efforts enable thinking with the shared Responses budget map.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .expect = .{ .enable = true, .budget = 1024 } },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .expect = .{ .enable = true, .budget = 2048 } },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .expect = .{ .enable = true, .budget = 8192 } },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .expect = .{ .enable = true, .budget = -1 } },
        // Unknown efforts (xhigh, future values) enable with the default
        // budget — spec values are model-dependent, never reject them.
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .expect = .{ .enable = true, .budget = -1 } },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = parseReasoningEffort(parsed.value.object, -1, false);
        if (case.expect) |want| {
            try std.testing.expectEqual(want.enable, got.?.enable);
            try std.testing.expectEqual(want.budget, got.?.budget);
            // The raw client string always rides along (templates that read the
            // effort word map it into the render); it is the request body's own bytes.
            const v = parsed.value.object.get("reasoning_effort").?;
            try std.testing.expectEqualStrings(v.string, got.?.effort.?);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "parseReasoningEffort: a template that READS the effort word gets no budget from it unless a decode-time bound can enforce it" {
    // Two levers, one word. Where the template acts on `reasoning_effort` the
    // word already shortens the THOUGHT; a budget that is only a delivery cap
    // fights it: pi asking for `medium` got the model's unguided (long)
    // thinking AND a 2048-token cut, followed by 25.9k tokens of invisible
    // generation (live 2026-08-14). The third argument is `effortWordOnly`:
    // true when the template reads the word AND `armThinkBound` cannot close
    // the thought at the budget (non-atomic markers). With atomic markers the
    // budget is enforced in-stream and the word maps like everywhere else.
    // An EXPLICIT cap is someone asking on purpose and still applies — that is
    // `default_budget` here, which carries `--reasoning-budget`.
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, consumes: bool, default_budget: i32, want: i32 }{
        // Consuming template: no effort-derived cap.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = true, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = true, .default_budget = -1, .want = -1 },
        // ...but `--reasoning-budget` still binds there.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .default_budget = 4096, .want = 4096 },
        // Non-consuming template (gemma 4,
        // Ling): the effort word reaches no template, so the budget is the ONLY
        // thing it does and every mapping stays exactly as it shipped.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .consumes = false, .default_budget = -1, .want = 1024 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = false, .default_budget = -1, .want = 2048 },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = false, .default_budget = -1, .want = 8192 },
        // high and xhigh are uncapped: pi's own ladder is 2048 / 8192 / 16384.
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = false, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .consumes = false, .default_budget = -1, .want = -1 },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = parseReasoningEffort(parsed.value.object, case.default_budget, case.consumes).?;
        try std.testing.expect(got.enable);
        try std.testing.expectEqual(case.want, got.budget);
        // The raw string rides along either way — the template still renders it.
        try std.testing.expectEqualStrings(parsed.value.object.get("reasoning_effort").?.string, got.effort.?);
    }
}

test "reasoningEffortFromWord: none disables, words budget exactly like the OpenAI surface" {
    // "none" is an explicit off with the word kept for the template.
    const off = reasoningEffortFromWord("none", -1, false);
    try std.testing.expect(!off.enable);
    try std.testing.expectEqualStrings("none", off.effort.?);
    // A word maps through the ONE effortBudget table…
    const low = reasoningEffortFromWord("low", -1, false);
    try std.testing.expect(low.enable);
    try std.testing.expectEqual(@as(i32, 2048), low.budget);
    // …unless the template consumes the word (an effort-reading template): then the word is
    // the lever and the budget stays the launch default.
    const consumed = reasoningEffortFromWord("low", -1, true);
    try std.testing.expect(consumed.enable);
    try std.testing.expectEqual(@as(i32, -1), consumed.budget);
}

test "resolveChatThinking: an explicit request value outranks the arch default, silence takes it" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, arch: bool, want: bool }{
        // Silent request → the arch default, either way.
        .{ .body = "{}", .arch = false, .want = false },
        .{ .body = "{}", .arch = true, .want = true },
        // An explicit false must survive a thinking-on arch — silently
        // ignoring a field the client set is the class this exists to avoid.
        .{ .body = "{\"enable_thinking\":false}", .arch = true, .want = false },
        .{ .body = "{\"reasoning_effort\":\"none\"}", .arch = true, .want = false },
        // An explicit enable works on an arch that defaults off.
        .{ .body = "{\"enable_thinking\":true}", .arch = false, .want = true },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .arch = false, .want = true },
        // Both present stay OR'd, as they always were.
        .{ .body = "{\"enable_thinking\":false,\"reasoning_effort\":\"high\"}", .arch = false, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"reasoning_effort\":\"none\"}", .arch = false, .want = true },
        // A non-bool `enable_thinking` is not a signal; with nothing else in
        // the body the arch default still applies.
        .{ .body = "{\"enable_thinking\":\"yes\"}", .arch = true, .want = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(case.want, resolveChatThinking(parsed.value.object, .{}, case.arch, -1, false).enable);
    }
}

test "resolveChatThinking: the request decides, the generation-defaults rules fill its silence or force it, then the arch" {
    const allocator = std.testing.allocator;
    const Case = struct { body: []const u8, model_on: ?bool = null, model_effort: ?generation_settings.Effort = null, force: bool = false, arch: bool = false, want: bool, effort: ?[]const u8 = null };
    const cases = [_]Case{
        .{ .body = "{}", .arch = true, .want = true },
        .{ .body = "{}", .model_on = true, .want = true },
        .{ .body = "{}", .model_on = false, .arch = true, .want = false },
        .{ .body = "{}", .model_effort = .high, .want = true, .effort = "high" },
        .{ .body = "{}", .model_effort = .none, .arch = true, .want = false },
        .{ .body = "{\"enable_thinking\":false}", .model_effort = .high, .want = false },
        .{ .body = "{\"enable_thinking\":true}", .model_effort = .high, .want = true, .effort = "high" },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .model_effort = .high, .want = true, .effort = "low" },
        // A forced rule replaces the request's switches, effort word included.
        .{ .body = "{\"enable_thinking\":false,\"reasoning_effort\":\"none\"}", .model_on = true, .force = true, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"reasoning_effort\":\"high\"}", .model_on = false, .force = true, .arch = true, .want = false },
        .{ .body = "{\"enable_thinking\":false}", .model_effort = .low, .force = true, .want = true, .effort = "low" },
        // vLLM's spelling; the top-level field wins over it.
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .model_on = true, .want = false },
        .{ .body = "{\"chat_template_kwargs\":{\"reasoning_effort\":\"medium\"}}", .want = true, .effort = "medium" },
        .{ .body = "{\"enable_thinking\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}", .want = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        var gen = generation_settings.Profile{};
        if (case.model_on) |on| gen.set(.enable_thinking, .{ .boolean = on }, case.force);
        if (case.model_effort) |e| gen.set(.reasoning_effort, .{ .effort = e }, case.force);
        const got = resolveChatThinking(parsed.value.object, gen, case.arch, -1, false);
        try std.testing.expectEqual(case.want, got.enable);
        if (case.effort) |w| try std.testing.expectEqualStrings(w, got.effort.?.effort.?) else try std.testing.expect(got.effort == null);
    }
}

test "resolveThinking: forced effort outranks unforced thinking defaults" {
    inline for (.{ false, true }) |arch_default| {
        inline for (.{ false, true }) |word_only| {
            var gen = generation_settings.Profile{};
            gen.set(.enable_thinking, .{ .boolean = true }, false);
            gen.set(.reasoning_effort, .{ .effort = .none }, true);
            const off = resolveThinking(gen, false, reasoningEffortFromWord("high", -1, word_only), arch_default, -1, word_only);
            try std.testing.expect(!off.enable);
            try std.testing.expect(off.effort == null);

            gen.set(.enable_thinking, .{ .boolean = false }, false);
            gen.set(.reasoning_effort, .{ .effort = .low }, true);
            const on = resolveThinking(gen, false, reasoningEffortFromWord("none", -1, word_only), arch_default, -1, word_only);
            try std.testing.expect(on.enable);
            try std.testing.expectEqualStrings("low", on.effort.?.effort.?);

            gen.set(.enable_thinking, .{ .boolean = false }, true);
            try std.testing.expect(!resolveThinking(gen, true, null, arch_default, -1, word_only).enable);
            gen.set(.enable_thinking, .{ .boolean = true }, true);
            gen.set(.reasoning_effort, .{ .effort = .none }, true);
            try std.testing.expect(resolveThinking(gen, false, null, arch_default, -1, word_only).enable);
        }
    }
}

test "schema thinking policy defers only a resolved reasoning protocol" {
    const decide = schemaMasksThinking;
    try std.testing.expectEqual(SchemaThinkingPolicy.no_mask, decide(false, false, true, false, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.no_mask, decide(true, true, true, true, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.token_zero, decide(true, false, false, true, false));
    try std.testing.expectEqual(SchemaThinkingPolicy.deferred, decide(true, false, true, false, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, true, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, false, false));

    // A bare think tail classifies as resolvable (the resolver owns the
    // capability checks); channel-family and plain ChatML tails do not, and
    // keep the fallback — capability is never the tail's alone to decide.
    try std.testing.expect(chat_mod.promptThinkTailClass("<|im_start|>assistant\n<think>") == .bare);
    const unsupported_tails = [_][]const u8{
        "<|channel|>analysis<|message|>",
        "<|im_start|>assistant\n",
    };
    for (unsupported_tails) |tail| {
        try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, false, chat_mod.promptThinkTailClass(tail) != .none));
    }
}

test "every JSON grammar mask site consults the fallback or deferral policy" {
    const src = @embedFile("server.zig");
    const mask_line = "[grammar] enforcing JSON schema from " ++ "token 0";
    const call = "schemaMasks" ++ "Thinking(";
    const def = "fn schemaMasks" ++ "Thinking(";
    const masks = std.mem.count(u8, src, mask_line);
    const calls = std.mem.count(u8, src, call) - std.mem.count(u8, src, def);
    try std.testing.expect(masks >= 1);
    try std.testing.expectEqual(masks, calls);
}

test "resolveSampling: forced rule > request > rule > generation_config > fallback" {
    const a = std.testing.allocator;
    const zeros = try std.json.parseFromSlice(std.json.Value, a, "{\"temperature\":0,\"top_k\":0,\"min_p\":0,\"frequency_penalty\":0}", .{});
    defer zeros.deinit();
    const config = model_mod.ModelConfig{ .gen_temperature = 0.8, .gen_top_p = 0.95, .gen_top_k = 40, .gen_min_p = 0.1 };
    var got = resolveSampling(.{}, .empty, &config);
    try std.testing.expectEqual(@as(f32, 0.8), got.temperature);
    try std.testing.expectEqual(@as(u32, 40), got.top_k);
    try std.testing.expectEqual(@as(?f32, 0.1), got.min_p);
    try std.testing.expectEqual(@as(f32, 1.0), got.repeat_penalty);
    got = resolveSampling(.{}, .empty, &.{});
    try std.testing.expectEqual(@as(f32, 1.0), got.temperature);
    try std.testing.expectEqual(@as(?f32, null), got.min_p);
    var gen = generation_settings.Profile{};
    gen.set(.temperature, .{ .number = 0.25 }, false);
    gen.set(.min_p, .{ .number = 0.05 }, true);
    gen.set(.repeat_penalty, .{ .number = 1.5 }, false);
    got = resolveSampling(gen, .empty, &config);
    try std.testing.expectEqual(@as(f32, 0.25), got.temperature);
    try std.testing.expectEqual(@as(f32, 1.5), got.repeat_penalty);
    // An explicit zero is a value: it beats an unforced rule, never a forced one.
    got = resolveSampling(gen, zeros.value.object, &config);
    try std.testing.expectEqual(@as(f32, 0.0), got.temperature);
    try std.testing.expectEqual(@as(u32, 0), got.top_k);
    try std.testing.expectEqual(@as(?f32, 0.05), got.min_p);
    try std.testing.expectEqual(@as(f32, 1.5), got.repeat_penalty);
    try std.testing.expectEqual(@as(f32, 0.0), got.frequency_penalty);
    gen.set(.repeat_penalty, .{ .number = 1.0 }, false);
    gen.set(.frequency_penalty, .{ .number = 0.5 }, false);
    got = resolveSampling(gen, .empty, &config);
    try std.testing.expectEqual(@as(f32, 1.0), got.repeat_penalty);
    try std.testing.expectEqual(@as(f32, 0.5), got.frequency_penalty);
    gen.set(.repeat_penalty, .{ .number = 1.2 }, true);
    try std.testing.expectEqual(@as(f32, 1.2), resolveSampling(gen, zeros.value.object, &config).repeat_penalty);
}

test "OpenAI frequency_penalty is its own additive penalty, never a repeat_penalty multiplier (#912)" {
    const a = std.testing.allocator;
    const body = try std.json.parseFromSlice(std.json.Value, a, "{\"frequency_penalty\":1,\"presence_penalty\":0}", .{});
    defer body.deinit();
    const got = resolveSampling(.{}, body.value.object, &.{});
    try std.testing.expectEqual(@as(f32, 1.0), got.frequency_penalty);
    try std.testing.expectEqual(@as(f32, 1.0), got.repeat_penalty);
}

test "generationMaxTokens: an auto request (0, negative, omitted) takes the rule, a forced rule takes everything" {
    var gen = generation_settings.Profile{};
    try std.testing.expectEqual(@as(u32, 7), generationMaxTokens(gen, .{ .integer = -1 }, 7));
    try std.testing.expectEqual(@as(u32, 64), generationMaxTokens(gen, .{ .integer = 64 }, 7));
    gen.set(.max_tokens, .{ .number = 12 }, false);
    try std.testing.expectEqual(@as(u32, 12), generationMaxTokens(gen, .{ .integer = -1 }, 7));
    try std.testing.expectEqual(@as(u32, 12), generationMaxTokens(gen, null, 7));
    try std.testing.expectEqual(@as(u32, 64), generationMaxTokens(gen, .{ .integer = 64 }, 7));
    gen.set(.max_tokens, .{ .number = 0 }, false); // 0 = auto
    try std.testing.expectEqual(AUTO_MAX_TOKENS_SENTINEL, generationMaxTokens(gen, null, 7));
    try std.testing.expectEqual(@as(u32, 64), generationMaxTokens(gen, .{ .integer = 64 }, 0));
    gen.set(.max_tokens, .{ .number = 0 }, true);
    for ([_]?std.json.Value{ null, .{ .integer = 0 }, .{ .integer = -1 }, .{ .integer = 64 } }) |value| {
        const cap = generationMaxTokens(gen, value, 0);
        try std.testing.expectEqual(AUTO_MAX_TOKENS_SENTINEL, cap);
        try std.testing.expectEqual(@as(u32, 8092), clampMaxTokens(cap, 100, 8192));
    }
    try std.testing.expectEqual(@as(u32, 0), generationMaxTokens(.{}, null, 0));
    gen.set(.max_tokens, .{ .number = 12 }, true);
    try std.testing.expectEqual(@as(u32, 12), generationMaxTokens(gen, .{ .integer = 64 }, 7));
}

test "reasoning budget: the rule stands in for --reasoning-budget, forced it replaces the request's" {
    var gen = generation_settings.Profile{};
    try std.testing.expectEqual(server_config.default_reasoning_budget, defaultReasoningBudget(gen));
    gen.set(.reasoning_budget, .{ .number = 1024 }, false);
    try std.testing.expectEqual(@as(i32, 1024), defaultReasoningBudget(gen));
    // xhigh maps to the default budget, so an unforced rule caps it; medium keeps its own.
    try std.testing.expectEqual(@as(i32, 1024), reasoningEffortFromWord("xhigh", defaultReasoningBudget(gen), false).budget);
    try std.testing.expectEqual(@as(i32, 8192), forcedReasoningBudget(gen, reasoningEffortFromWord("medium", 1024, false).budget));
    gen.set(.reasoning_budget, .{ .number = 16 }, true);
    try std.testing.expectEqual(@as(i32, 16), forcedReasoningBudget(gen, 32000));
}

test "detokenizeResponseJson escapes arbitrary token bytes (control-byte class)" {
    const allocator = testing.allocator;
    // A decoded token is arbitrary bytes. The old hand-rolled escaper covered
    // five characters and passed everything else through, so a single id whose
    // bytes are below 0x20 shipped a body no JSON parser accepts — live
    // 2026-08-04 while cross-checking a vocabulary id by id.
    const nasty = "a\x01b\x1fc\"d\\e\nf\tg";
    const out = try detokenizeResponseJson(allocator, nasty);
    defer allocator.free(out);
    for (out) |c| try testing.expect(c >= 0x20);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, out, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(nasty, parsed.value.object.get("content").?.string);

    // Ordinary text is untouched apart from the quoting.
    const plain = try detokenizeResponseJson(allocator, "hello");
    defer allocator.free(plain);
    try testing.expectEqualStrings("{\"content\":\"hello\"}", plain);
}

test "optSamplingRecJson emits number when present, null when absent" {
    const a = std.testing.allocator;

    // Present -> bare JSON number (no quotes), so the Swift side decodes it
    // straight to Double/Int.
    const top_k = try optSamplingRecJson(a, u32, 20);
    defer a.free(top_k);
    try std.testing.expectEqualStrings("20", top_k);

    const top_p = try optSamplingRecJson(a, f32, 0.95);
    defer a.free(top_p);
    try std.testing.expectEqualStrings("0.95", top_p);

    // Absent -> JSON null literal so the model-author recommendation reads as
    // "no opinion" rather than a spurious 0.
    const none = try optSamplingRecJson(a, u32, null);
    defer a.free(none);
    try std.testing.expectEqualStrings("null", none);
}

test "textGenRejectReason: no-LM models rejected, chat models pass (SIGSEGV class 2026-07-06)" {
    // Chat-capable targets pass: a resident MLX LM, and
    // unloaded stubs with a chat arch hint (or none: benefit of the doubt
    // until load).
    try std.testing.expect(textGenRejectReason(.{ .arch_hint = "gemma4" }) == null);
    try std.testing.expect(textGenRejectReason(.{}) == null);

    // The live crash shape: a READY entry with no LM resident must reject,
    // never reach prefill.
    try std.testing.expect(textGenRejectReason(.{ .has_text_lm = false }) != null);
}

test "isTextGenRoute covers exactly the guarded surfaces" {
    try std.testing.expect(isTextGenRoute("POST", "/v1/chat/completions"));
    // The embeddings route must NOT be gated.
    try std.testing.expect(!isTextGenRoute("POST", "/v1/embeddings"));
    try std.testing.expect(!isTextGenRoute("GET", "/v1/models"));
}

test "isStatusRoute: only /props reaches the shared release" {
    try std.testing.expect(isStatusRoute("GET", "/props"));
    // Generation is use.
    try std.testing.expect(!isStatusRoute("POST", "/v1/chat/completions"));
    // Method matters: the path alone is not the predicate.
    try std.testing.expect(!isStatusRoute("POST", "/props"));
}

test "embedEffectiveLimit: tighter of flag and model window; zeros mean unbounded" {
    // Auto (no flag): the model's window rules; a windowless model is unbounded.
    try std.testing.expectEqual(@as(u32, 512), embedEffectiveLimit(0, 512));
    try std.testing.expectEqual(@as(u32, 0), embedEffectiveLimit(0, 0));
    // Operator override: enforced when tighter…
    try std.testing.expectEqual(@as(u32, 1024), embedEffectiveLimit(1024, 32768));
    // …and clamped to the model's declared window when looser — a 512-position
    // model never accepts a 4096 override (issue #117 acceptance case).
    try std.testing.expectEqual(@as(u32, 512), embedEffectiveLimit(4096, 512));
    // Flag against a windowless model still binds.
    try std.testing.expectEqual(@as(u32, 2048), embedEffectiveLimit(2048, 0));
}

test "embedOverflowMessage: names the input index and both counts" {
    var buf: [160]u8 = undefined;
    const msg = embedOverflowMessage(&buf, 3, 1301, 1024);
    try std.testing.expectEqualStrings(
        "Input at index 3 exceeds the maximum embedding input length: 1301 tokens given, 1024 allowed",
        msg,
    );
}

test "readyCapsJson: embeddings capability, alone and beside chat" {
    const a = std.testing.allocator;
    // Embeddings and nothing else.
    var enc = try readyCapsJson(a, .{ .has_embedding = true });
    defer enc.deinit(a);
    try std.testing.expectEqualStrings("[\"embeddings\"]", enc.items);
    // A pooling-contracted decoder keeps its
    // chat set AND advertises embeddings — a READY model never advertises
    // less capability than its stub.
    var both = try readyCapsJson(a, .{ .has_chat = true, .has_embedding = true });
    defer both.deinit(a);
    try std.testing.expect(std.mem.indexOf(u8, both.items, "\"chat\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, both.items, "\"embeddings\"") != null);
}

test "readyCapsJson: chat-class flags advertise their capabilities" {
    const a = std.testing.allocator;

    var chat = try readyCapsJson(a, .{ .has_chat = true, .has_reasoning = true });
    defer chat.deinit(a);
    try std.testing.expectEqualStrings(
        "[\"chat\",\"tool_use\",\"streaming\",\"reasoning\",\"json_schema\"]",
        chat.items,
    );
}

test "readyHasChat: an engine-backed model advertises chat without a template" {
    const t = std.testing;
    // MLX model with a template: chat.
    try t.expect(readyHasChat(1234, false));
    try t.expect(readyHasChat(0, true));
    // Templateless pure-MLX entry keeps the conservative behavior.
    try t.expect(!readyHasChat(0, false));
}

test "mlxMemoryGuardApplies: engine-backed models skip the MLX-prefill memory guard" {
    const t = std.testing;
    // MLX model — the guard applies (real per-token working set).
    try t.expect(mlxMemoryGuardApplies(false));
    // No engine-backed model prefills in MLX.
    try t.expect(!mlxMemoryGuardApplies(true));
}

test "prefillMemoryNeeded: a prompt shorter than the chunk bills the prompt-width forward, not the chunk cap" {
    // Live 2026-07-14: with 111 GB of weights resident (~4 GB headroom),
    // a 31-TOKEN prompt was rejected "needs ~6246MB" — the MLP envelope was
    // billed at the hd-128 8192 chunk cap while the actual prefill runs ONE
    // 31-token forward (~25 MB). A forward is never wider than the prompt:
    // the envelope must use min(chunk, seq). Guard-tracks-reality rule.
    const small = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 31, 31, 0, 0, .{});
    const capped = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 8192, 31, 0, 0, .{});
    try std.testing.expectEqual(small, capped);
    // Sanity: the clamped estimate for the live 31-token case sits far below
    // the ~4 GB that was actually available. It is not zero — every prefill
    // pays PREFILL_RUNTIME_FLOOR_BYTES — but a tiny prompt pays little else,
    // and the dequant-weight term never fires this far under the route's
    // minimum forward width.
    try std.testing.expect(capped < 1024 * 1024 * 1024);
}

test "prefillMemoryNeeded: quantized KV is billed at its real width, not fp16" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // 26B-A4B shape at 100K ctx, chunk 512 (what the bounded chunk picks there).
    // fp16: kv 30x2x100000x8x256x2 = 24.576 GB; scores 16x512x100000x2 = 1.6384 GB;
    //       mlp 3x8x512x2816x2 = 69.2 MB; x1.25 margin.
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // 4-bit: kv billed at (2*4+1)/16 bytes/elem (payload + group scale/bias)
    // = 6.912 GB, plus the 0.8192 GB per-layer dense dequant transient.
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
    // Direction: quantized admission must be under half the fp16 bill.
    try t.expect(prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}) * 2 <
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}));
}

test "prefillMemoryNeeded: working set is chunk-bounded — the 255K MoE prompt is admittable" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // The PR-#69 failure case: a shape (whose checkpoint OMITS
    // intermediate_size, so the struct default 15360 leaks into ffn) at 262144
    // tokens with 4-bit KV. kv 6.04 GB + scores 4.29 GB + dequant 0.54 GB +
    // mlp 0.377 GB, x1.25 = ~13.1 GiB — admittable on a big Mac.
    const needed = prefillMemoryNeeded(262_144, 16, 2, 81920, 256, 256, 2048, 15360, 4, 512, 262_144, 0, 0, .{});
    try t.expectEqual(@as(u64, 14_061_404_160 + 671_088_640), needed);
    try t.expect(needed < 16 << 30);
    // The retired seq-scaled envelope billed 8 x seq x ffn x 2 working bytes
    // (~64 GB) on top of fp16 KV (~21.5 GB) -> ~107 GB and a spurious 400.
    const old_estimate: u64 = (40 * 2 * 262_144 * 2 * 256 * 2 + 8 * 262_144 * 15360 * 2) * 5 / 4;
    try t.expect(needed < old_estimate / 7);
}

test "prefillMemoryNeeded: unfused head dims bill the materialized score scratch" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // Same shape, head_dim 128 vs 256 (kv scaled by hdim too, so compare the
    // full bills computed by hand): hd=128 fp16 -> kv 12.288 GB + 0 scores +
    // mlp 69.2 MB, x1.25.
    try t.expectEqual(
        @as(u64, 15_446_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 122880, 128, 128, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // hd=256 adds the [heads, chunk, seq] score tensor (the guard must see
    // what the composed SDPA path actually allocates). Decomposition: the
    // hd-128 bill + the KV doubling from 128->256 (12.288 GB x1.25) + the
    // score tensor (16x512x100000x2 = 1.6384 GB x1.25).
    const with_scores = prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{});
    try t.expectEqual(@as(u64, 15_446_507_520 + 671_088_640 + 15_360_000_000 + 2_048_000_000), with_scores);
}

test "prefillMemoryNeeded: the SCORE width decides the score term, not the stored width" {
    // A latent-attention model contracts scores over qk_head_dim 192 while storing
    // values at v_head_dim 128 and declaring head_dim 128. mlx has a fused
    // vector kernel for that pair at DECODE and falls back to the composed
    // path at prefill widths, so the [heads, chunk, seq] score tensor is real —
    // but `prefillHeadDimFused(128)` is true, so passing the stored width for
    // both billed it at ZERO. Under-billing ends in an uncatchable Metal OOM
    // instead of a 400, which is the whole reason this guard exists.
    const t = std.testing;
    const kv_per_tok: u64 = 6 * 16 * (192 + 128) * 2; // 6 caching layers of 24
    const stored: u64 = 128;
    const scored: u64 = 192;
    const honest = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    const blind = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, stored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    // The difference is exactly the score tensor: heads x chunk x seq x 2, x1.25.
    const score_bytes: u64 = 16 * 4096 * 32_768 * 2;
    try t.expectEqual(honest - blind, score_bytes * 5 / 4);
    try t.expect(honest > 2 * blind);
    // The two widths are independent: at fp16 (no dequant transient) the
    // stored width touches nothing the score term reads.
    try t.expectEqual(
        honest,
        prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, scored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{}),
    );
}

test "prefillMemoryNeeded: fused hd-256 kernel drops the score bill, keeps KV + dequant" {
    const t = std.testing;
    // Default (msv_attn_p256 active): the composed score scratch never exists,
    // so hd 256 bills exactly the hd-256 KV + mlp — the unfused bill minus the
    // 1.6384 GB x1.25 score term from the test above.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // The quantized-KV dequant transient is a denseView property, NOT a score
    // property — it must survive the fused kernel (fires on every kv-quant
    // request regardless of head_dim).
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: a squeezed box steps the chunk down; a roomy one keeps it" {
    // The reserve at the widest chunk must never claim more than a quarter of
    // what is left after the weights. Numbers are the measured 16 GB profile: Metal's working
    // set ~11.9 GiB, a 7B 4-bit model 3.80 GiB resident. The hot-cache budget is no longer an
    // input; the former `2 * GiB` argument is the pinned-context KV bar and is 0 under auto-context.
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    var dense7b = model_mod.ModelConfig{};
    dense7b.num_hidden_layers = 32;
    dense7b.num_attention_heads = 32;
    dense7b.num_key_value_heads = 8;
    dense7b.head_dim = 128;
    dense7b.hidden_size = 4096;
    dense7b.intermediate_size = 14336;
    dense7b.intermediate_size_declared = true;
    dense7b.quant_bits = 4;
    dense7b.max_position_embeddings = 32768;

    const ceiling_16: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    const narrowed = resolvePrefillChunk(&dense7b, 16, ceiling_16, weights, 0, 0);
    try t.expect(narrowed < 8192);
    try t.expect(narrowed >= 512);
    // It is the WIDEST rung that fits, not simply the floor.
    try t.expect(prefillTransientReserve(&dense7b, 16, narrowed) <=
        (ceiling_16 - weights) / 4);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= narrowed) break;
        try t.expect(prefillTransientReserve(&dense7b, 16, c) > (ceiling_16 - weights) / 4);
    }

    // Same model, a machine with room: nothing narrows.
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&dense7b, 16, 95 * GiB, weights, 0, 0));

    // A model that barely fits gets the narrowest rung rather than a wide one
    // it cannot pay for — and never 0, which would read as "not pinned".
    try t.expectEqual(@as(u32, 512), resolvePrefillChunk(&dense7b, 16, weights + 64 * 1024 * 1024, weights, 0, 0));

    // Every rung is a real ladder entry, descending, floored at generate's own
    // minimum — a chunk generate would refuse is a chunk the bill cannot model.
    var prev: u32 = std.math.maxInt(u32);
    for (PREFILL_CHUNK_LADDER) |c| {
        try t.expect(c < prev);
        try t.expect(c >= generate_mod.PREFILL_CHUNK_FLOOR);
        prev = c;
    }
}

test "prefillMemoryNeeded: every prefill peak the box measured, re-derived for the conv_state compaction" {
    // Peak GPU bytes above steady state for ONE request on a clean boot
    // (`--prefix-cache-entries 0 --no-drafter --no-pld`, /props
    // peak_bytes minus active_bytes, M4 Max, 2026-08-14) — byte-identical
    // across repeat boots, so these are exact figures, not samples. An
    // estimate BELOW a peak we have actually seen is the case that ends in an
    // uncatchable Metal OOM instead of a 400, so every row must be covered.
    // Pre-fix this function billed 5 of these 8 shapes short, worst 0.58x.
    // the 27B and 4B rows are the measured peaks minus the exact conv_state pin the compaction removed ((linear_layers - 5) x chunk x qkv x 2); the packs are not on the box to re-measure.
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;

    // A 27B: hidden 5120, ffn 17408, 64 layers / 16 caching, hd 256,
    // 48 GatedDeltaNet layers x (2x16x128 + 48x128) elems x 2 B of stream.
    const q27_stream: u64 = 5 * (2 * 16 * 128 + 48 * 128) * 2;
    const q27_dq: u64 = 3 * 17408 * 5120 * 2;
    try t.expect(prefillMemoryNeeded(9320, 24, 4, 65536, 256, 256, 5120, 17408, 16, 2048, 9320, q27_stream, q27_dq, .{}) >= 2_174_449_280);
    try t.expect(prefillMemoryNeeded(9270, 24, 4, 65536, 256, 256, 5120, 17408, 16, 8192, 9270, q27_stream, q27_dq, .{}) >= 3_623_797_120);
    // A 4B: hidden 2560, ffn 9216, 32 layers / 8 caching, 24 gated-delta layers.
    const q4_stream: u64 = 5 * (2 * 16 * 128 + 32 * 128) * 2;
    const q4_dq: u64 = 3 * 9216 * 2560 * 2;
    try t.expect(prefillMemoryNeeded(9340, 16, 4, 32768, 256, 256, 2560, 9216, 16, 1024, 9340, q4_stream, q4_dq, .{}) >= 1_329_232_896);
    try t.expect(prefillMemoryNeeded(9270, 16, 4, 32768, 256, 256, 2560, 9216, 16, 8192, 9270, q4_stream, q4_dq, .{}) >= 2_559_863_168);
    // gemma4 26B-A4B MoE: hidden 2816, top_k 8 x moe 704, 30 sliding layers.
    const g4_stream: u64 = 4 * 8 * 2 * (2816 + 704) * 2;
    const g4_dq: u64 = 3 * 5632 * 2816 * 2;
    try t.expect(prefillMemoryNeeded(8204, 16, 8, 245760, 256, 256, 2816, 5632, 16, 4096, 8204, g4_stream, g4_dq, .{}) >= 3_960_000_000);
    // A 2.6B conv hybrid at the narrowest chunk measured: no stream term,
    // and the forward is under the dequant route's minimum width.
    try t.expect(prefillMemoryNeeded(10089, 32, 8, 16384, 64, 64, 2048, 10752, 16, 256, 10089, 0, 3 * 10752 * 2048 * 2, .{}) >= 782_000_000);
    try t.expect(prefillMemoryNeeded(10206, 32, 8, 16384, 64, 64, 2048, 10752, 16, 8192, 10206, 0, 3 * 10752 * 2048 * 2, .{}) >= 2_527_000_000);
    // A 30B dense: hidden 6656, ffn 19968, 52 layers, hd 128.
    try t.expect(prefillMemoryNeeded(9827, 32, 2, 53248, 128, 128, 6656, 19968, 16, 2048, 9827, 0, 3 * 19968 * 6656 * 2, .{}) >= 2_178_000_000);
}

test "prefillMemoryNeeded: the new terms fire only where the measurement put them" {
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const q27 = .{ .seq = 9320, .h = 24, .kvh = 4, .kvpt = 65536, .hd = 256, .hidden = 5120, .ffn = 17408 };
    const stream: u64 = 48 * (2 * 16 * 128 + 48 * 128) * 2;
    const dq: u64 = 3 * 17408 * 5120 * 2;

    // The stream term is per CHUNK-TOKEN: it scales with the forward width and
    // not with the prompt, which is what the whole chunk-bounded model says.
    const at2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, stream, 0, .{});
    const at8k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 8192, q27.seq, stream, 0, .{});
    const flat2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{});
    const mlp2k: u64 = 8 * 2048 * q27.ffn * 2;
    const floor: u64 = 512 * 1024 * 1024;
    // The stream arm wins on this arch — that IS the fix — so the bill is one
    // MLP envelope plus the streams, not the three-envelope floor.
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + mlp2k + 2048 * stream + floor) * 5 / 4), at2k);
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + 3 * mlp2k + floor) * 5 / 4), flat2k);
    try t.expect(at2k > flat2k);
    try t.expect(at8k > at2k);

    // A longer prompt at the same chunk adds KV only — the stream term does
    // not move, or it would be the per-token multiplier this class is about.
    const long = prefillMemoryNeeded(4 * q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, 4 * q27.seq, stream, 0, .{});
    try t.expectEqual(long - at2k, 3 * q27.seq * q27.kvpt * 5 / 4);

    // The dequant-weight term is chunk-INDEPENDENT (it is weights) but only
    // bills where the route can fire: at or above PREFILL_DQ_GEMM_MIN_M.
    const dq_wide = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, dq, .{});
    try t.expectEqual(dq_wide - flat2k, dq * 5 / 4);
    const narrow_with = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, dq, .{});
    const narrow_without = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, 0, .{});
    try t.expectEqual(narrow_with, narrow_without);
    // A dense checkpoint passes 0 and is billed nothing for it (measured: a
    // bf16 a 27B sits within 0.19 GB of the 8-bit one, which IS this term).
    try t.expectEqual(flat2k, prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{}));

    // An arch with neither keeps the 3-envelope bill it had, plus the floor —
    // no plain-attention arch's admission tightens by more than the floor.
    const dense_only = prefillMemoryNeeded(9827, 32, 2, 53248, 128, 128, 6656, 19968, 16, 2048, 9827, 0, 0, .{});
    const pre_fix: u64 = (9827 * 53248 + 3 * 8 * 2048 * 19968 * 2) * 5 / 4;
    try t.expectEqual(dense_only - pre_fix, 512 * 1024 * 1024 * 5 / 4);
}

test "prefillStreamBytesPerToken: keyed on the arch's own geometry, zero for plain attention" {
    const t = std.testing;
    // gemma4 26B-A4B: no linear layers, but a MoE prefill gathers a top_k-fold
    // copy of the hidden stream beside the expert rows, for the 4 layers the
    // MoE eval cadence lets coexist.
    var g4 = model_mod.ModelConfig{ .model_type = "gemma4" };
    g4.num_hidden_layers = 30;
    g4.hidden_size = 2816;
    g4.num_experts = 128;
    g4.num_experts_per_tok = 8;
    g4.moe_intermediate_size = 704;
    try t.expectEqual(@as(u64, 4 * 8 * 2 * (2816 + 704) * 2), prefillStreamBytesPerToken(&g4));

    // A dense config declares neither and is billed exactly what it was
    // billed before this term existed.
    var dense = model_mod.ModelConfig{ .model_type = "gemma4" };
    dense.num_hidden_layers = 52;
    dense.hidden_size = 6656;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&dense));

    // A sparse MoE trunk bills the MoE stream term, and its FFN width counts
    // the shared expert.
    var sparse = model_mod.ModelConfig{ .model_type = "gemma4" };
    sparse.num_hidden_layers = 52;
    sparse.hidden_size = 2688;
    sparse.num_experts = 128;
    sparse.num_experts_per_tok = 6;
    sparse.moe_intermediate_size = 1856;
    sparse.shared_expert_intermediate_size = 3712;
    try t.expectEqual(@as(u64, 4 * 6 * 2 * (2688 + 1856) * 2), prefillStreamBytesPerToken(&sparse));
    try t.expectEqual(@as(u64, 1856 * 6 + 3712), prefillFfnWidth(&sparse));
}

test "prefillDequantWeightBytes: affine-quantized weights only, and it reads the route's kill switch" {
    const t = std.testing;
    var cfg = model_mod.ModelConfig{ .model_type = "gemma4" };
    cfg.hidden_size = 6656;
    cfg.intermediate_size = 19968;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    try t.expectEqual(@as(u64, 3 * 19968 * 6656 * 2), prefillDequantWeightBytes(&cfg));

    // Dense bf16: the route never runs, and the measurement agrees — a bf16
    // checkpoint's prefill peak sits below the quantized one by this term.
    var dense = cfg;
    dense.quant_bits = 0;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&dense));

    // Non-affine (nvfp4/mxfp8) stays in mlx_quantized_matmul — no dequantized
    // copy exists to bill (measured: nvfp4 lands with the dq-off arm).
    var nvfp4 = cfg;
    nvfp4.quant_mode = .nvfp4;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&nvfp4));

    // The estimator must not bill an allocation the kill switch removed.
    transformer_mod.prefill_dq_gemm_override = false;
    defer transformer_mod.prefill_dq_gemm_override = null;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&cfg));
}

test "prefillMemoryNeeded: a sparse-attention arch bills its KEY BOUND, not the whole prompt" {
    const t = std.testing;
    // A sparse-attention shape: 64 heads, ONE latent kv head, 43 layers,
    // head_dim 512 - outside every fused kernel, so the score term fires. But
    // such an arch attends to a bounded key set per query (window + top-k +
    // sink keys), NEVER the whole prompt. Billing a dense 5806-wide key axis
    // put a 5806-token prompt at ~10.3 GB and 400-rejected it on a
    // 1M-context model with 8.6 GB free.
    // The live request verbatim: 5806 tokens at kv-quant 8, chunk 5632, with
    // 8629 MB free. Pre-fix it billed 10277 MB — a dense 5806-wide key axis
    // PLUS the 15360 intermediate_size struct default — and 400'd.
    const available: u64 = 8629 * 1024 * 1024;
    const before = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 15360, 8, 5632, 5806, 0, 0, .{});
    try t.expectEqual(@as(u64, 10277 + 640), before / (1024 * 1024));
    try t.expect(before > available);
    // Fixed: 641 keys (window 128 + top-512 + sink) and the width the config
    // actually states (top_k 6 x moe_intermediate 2048 = 12288, what
    // prefillFfnWidth resolves for this checkpoint) — 4848 MB, admitted.
    const after = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 12288, 8, 5632, 641, 0, 0, .{});
    try t.expectEqual(@as(u64, 4848 + 640), after / (1024 * 1024));
    try t.expect(after < available);
    // The score term is the ONLY one the key bound may touch: same call with a
    // fused head_dim has no score scratch, so the bound changes nothing.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try t.expectEqual(
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 5806, 0, 0, .{}),
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 641, 0, 0, .{}),
    );
}

test "checkAttentionMemory does not bill resident hot-cache buffers twice" {
    // RAM restore uses refcount-sharing. The resident entry is already inside
    // `active_mem`, and prefillMemoryNeeded bills the destination KV growth.
    // Pin the chokepoint so a future cache-accounting change cannot add the
    // resident entry to `needed` again (the 138k-token false-400 regression).
    const t = std.testing;
    const src = @embedFile("server.zig");
    // The bill and the headroom moved into `prefillAdmissionBill`.
    const start = std.mem.indexOf(u8, src, "pub fn prefillAdmission" ++ "Bill(") orelse return error.CallSiteMoved;
    const tail = src[start..];
    const end = std.mem.indexOf(u8, tail, "\nextern \"c\" fn sysctlbyname") orelse return error.CallSiteMoved;
    const body = tail[0..end];
    try t.expect(std.mem.indexOf(u8, body, "largestEntry" ++ "Bytes") == null);
    try t.expect(std.mem.indexOf(u8, body, "hot_" ++ "restore") == null);
    try t.expect(std.mem.indexOf(u8, body, "mlx_get_active_memory(&active_mem)") != null);
    // The hot cache appears once, as `evictable`, read as the published atomic.
    try t.expect(std.mem.indexOf(u8, body, "resident_hot_cache_bytes.load(.monotonic)") != null);
    try t.expect(std.mem.indexOf(u8, body, "sch." ++ "hot_prefix_cache") == null);
    try t.expect(std.mem.indexOf(u8, body, ".hot_prefix_cache.?") == null);
    try t.expect(std.mem.indexOf(u8, body, "needed = needed +") == null);
}

test "the chunk the guard BILLS is the chunk the forward will RUN" {
    // Fatal-class drift: the bill is linear in the chunk, so a guard that
    // models 512 while `generate` forwards 8192 admits a prefill that dies in
    // an uncatchable Metal completion-handler throw. `effectivePrefillChunk`
    // is the single resolver and every call site must hand it the model's
    // pinned width — the sizer, the admission guard, and the prefill loop.
    const t = std.testing;
    const srcs = [_][]const u8{ @embedFile("server.zig"), @embedFile("generate.zig") };
    var sites: usize = 0;
    for (srcs) |src| {
        var i: usize = 0;
        // Needle split so this scan's own source cannot satisfy it.
        const needle = "effectivePrefill" ++ "Chunk(";
        while (std.mem.indexOfPos(u8, src, i, needle)) |at| {
            i = at + 1;
            // Skip the declaration itself.
            if (at >= 3 and std.mem.eql(u8, src[at - 3 .. at], "fn ")) continue;
            // Skip test call sites: a hit belongs to a test when the nearest `test "` above it is
            // nearer than the nearest function header.
            const before = src[0..at];
            const last_test = std.mem.lastIndexOf(u8, before, "\ntest \"");
            const last_fn = std.mem.lastIndexOf(u8, before, "\nfn ") orelse 0;
            const last_pub_fn = std.mem.lastIndexOf(u8, before, "\npub fn ") orelse 0;
            if (last_test) |lt| {
                if (lt > @max(last_fn, last_pub_fn)) continue;
            }
            const tail = src[at..@min(src.len, at + 512)];
            const end = std.mem.indexOf(u8, tail, "));") orelse
                std.mem.indexOf(u8, tail, ");") orelse return error.CallSiteMoved;
            // `rung` is a candidate being priced and `chosen`/the ladder floor becomes the pin; all
            // are billed by `prefillNeededAtChunk`. A bare launch-width argument still fails.
            const args = tail[0..end];
            try t.expect(std.mem.indexOf(u8, args, "pinned_prefill_chunk") != null or
                std.mem.indexOf(u8, args, "rung") != null or
                std.mem.indexOf(u8, args, "chosen") != null or
                std.mem.indexOf(u8, args, "PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]") != null);
            sites += 1;
        }
    }
    try t.expect(sites >= 3);

    // The prefill loop's copy arrives through `InitOptions`, NOT off
    // `xfm.config`: the Transformer holds a COPY of the config taken when it
    // was built and the pin is written afterwards, so reading it there is a
    // silent no-op (live 2026-08-14: pinned 4096, prefilled at 8192). The
    // scheduler sources it from `slot.model.config` — the same object the
    // guard bills against.
    const sched = @embedFile("scheduler.zig");
    // The width is chosen per request (`req_prefill_chunk`) and falls back to `cfg.pinned_prefill_chunk`.
    // `companyPrefillChunk` only narrows it, so the bill stays an upper bound.
    try t.expect(std.mem.indexOf(u8, sched, ".pinned_prefill_chunk = companyPrefillChunk(req_prefill_chunk,") != null);
    try t.expect(std.mem.indexOf(u8, sched, "const req_prefill_chunk: u32 = if (slot.model.config) |cfg| blk: {") != null);
    try t.expect(std.mem.indexOf(u8, sched, "const pin = cfg.pinned_prefill_chunk;") != null);
    try t.expect(std.mem.indexOf(u8, srcs[1], "xfm.config.pinned_prefill_chunk") == null);

    // And the width has to be frozen BEFORE anything is computed from it:
    // `pinPrefillChunk` runs at the top of `pinAutoContext`, above the
    // `--ctx-size` early-out, because the guard reads it on every request.
    const server_src = srcs[0];
    const pin = std.mem.indexOf(u8, server_src, "fn pinAutoContext(") orelse return error.CallSiteMoved;
    const body = server_src[pin..@min(server_src.len, pin + 800)];
    const chunk_at = std.mem.indexOf(u8, body, "pinPrefillChunk(config)") orelse return error.PinMissing;
    const ctx_at = std.mem.indexOf(u8, body, "autoContextFor(config)") orelse return error.CallSiteMoved;
    const ctxsize_at = std.mem.indexOf(u8, body, "manualContext(config) > 0") orelse return error.CallSiteMoved;
    try t.expect(chunk_at < ctx_at);
    try t.expect(chunk_at < ctxsize_at);
}

test "resolvePrefillChunk: the 16 GB tier gets a working context instead of the 1024 floor" {
    // The reported bug, end to end. a 7B 4-bit model on a 16 GB Mac: ~11.9 GiB
    // Metal working set, 3.80 GiB of weights (measured), the hot prefix cache
    // at its 2 GB default. At the launch chunk the transient reserve alone is
    // larger than what is left, so the sizer floored the advertised context at
    // 1024 and the guard 400'd a 10,348-token prompt for "needing" 9.18 GiB —
    // a prompt whose peak measures 2.391 GiB (M4 Max, one request per boot).
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    const ceiling: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 32768;
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16);

    const launch = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, 8192), per_tok, 32768);
    try t.expectEqual(@as(u32, 1024), launch); // the floor: nothing fits

    // The hot-cache budget is no longer an input; auto-context means the pinned-context bar is 0.
    const chunk = resolvePrefillChunk(&cfg, 16, ceiling, weights, 0, 0);
    const sized = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, chunk), per_tok, 32768);
    try t.expect(sized > 12_000);

    // And the guard now admits a real agent prompt at that context, billed
    // against the same chunk. 8.10 GiB is what a 16 GB Mac has free with only
    // the weights resident.
    const needed = prefillMemoryNeeded(
        10_348,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.kvBytesPerToken(),
        cfg.head_dim,
        cfg.prefillScoreHeadDim(),
        cfg.hidden_size,
        prefillFfnWidth(&cfg),
        16,
        chunk,
        cfg.prefillAttnKeys(10_348),
        prefillStreamBytesPerToken(&cfg),
        prefillDequantWeightBytes(&cfg),
        .{},
    );
    try t.expect(needed < ceiling - weights);
    // Still comfortably above the measured 2.391 GiB peak — narrowing the chunk
    // must not turn a conservative bill into an optimistic one.
    try t.expect(needed > 2_391_000_000);
}

test "prefillFfnWidth: a declared MoE width beats the struct default, which is only a fallback" {
    const t = std.testing;
    // An MoE config can ship NO intermediate_size at all, so the struct default
    // (15360) was billed as though declared. The width the config DOES state is
    // top_k(6) x moe_intermediate(2048) — a chunk gathers that many expert rows
    // per token, so billing the bare 2048 would UNDER-bill 3x, and under-
    // billing ends in an uncatchable Metal OOM rather than a 400.
    var moe = model_mod.ModelConfig{};
    moe.intermediate_size = 15360; // struct default, never seen in the JSON
    moe.intermediate_size_declared = false;
    moe.moe_intermediate_size = 2048;
    moe.num_experts_per_tok = 6;
    moe.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 12288), prefillFfnWidth(&moe));

    // The shared expert runs on every token too, so it adds rather than maxes.
    var shared = moe;
    shared.shared_expert_intermediate_size = 1024;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&shared));

    // A dense checkpoint is unaffected: no MoE width, declared ffn wins.
    var dense = model_mod.ModelConfig{};
    dense.intermediate_size = 13312;
    dense.intermediate_size_declared = true;
    dense.moe_intermediate_size = 0;
    dense.num_experts_per_tok = 0;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&dense));

    // A checkpoint that DECLARES a dense ffn keeps the conservative @max — a
    // hybrid with dense layers really does run the wider one.
    var declared = moe;
    declared.intermediate_size_declared = true;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&declared));

    // Declares nothing: the struct default is all we have, so it still wins.
    var bare = model_mod.ModelConfig{};
    bare.intermediate_size = 15360;
    bare.intermediate_size_declared = false;
    bare.moe_intermediate_size = 0;
    bare.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&bare));
}

test "truncateEmbeddingDims: OpenAI dimensions semantics (truncate + L2-renormalize)" {
    const t = std.testing;

    // Truncate to 2 of 4: keeps the first components, renormalizes to unit L2.
    var v = [_]f32{ 3.0, 4.0, 100.0, 100.0 };
    const out = truncateEmbeddingDims(&v, 2);
    try t.expectEqual(@as(usize, 2), out.len);
    try t.expectApproxEqAbs(@as(f32, 0.6), out[0], 1e-6); // 3/5
    try t.expectApproxEqAbs(@as(f32, 0.8), out[1], 1e-6); // 4/5
    var norm: f64 = 0;
    for (out) |x| norm += @as(f64, x) * @as(f64, x);
    try t.expectApproxEqAbs(@as(f64, 1.0), norm, 1e-6);

    // dims == native width: byte-identical pass-through, no renorm — the
    // default (no `dimensions`) path must never change.
    var w = [_]f32{ 0.5, -0.25, 0.125 };
    const same = truncateEmbeddingDims(&w, 3);
    try t.expectEqual(@as(usize, 3), same.len);
    try t.expectEqual(@as(f32, 0.5), same[0]);
    try t.expectEqual(@as(f32, -0.25), same[1]);
    try t.expectEqual(@as(f32, 0.125), same[2]);

    // Zero prefix: truncation must not divide by zero or emit NaN.
    var z = [_]f32{ 0.0, 0.0, 1.0 };
    const zt = truncateEmbeddingDims(&z, 2);
    try t.expectEqual(@as(usize, 2), zt.len);
    for (zt) |x| try t.expect(!std.math.isNan(x));
}

test "resolveKvAttnFused: row-exact decode is off for every arch today, so a request's choice stands" {
    // The gate (`rowExactDecode()` returns false) stays in `resolveKvAttnFused` for the day a Gemma 4
    // row-exact path exists; then this test returns to asserting that exact decode never reads packed KV.
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "gemma4";
    cfg.row_exact_covered = true;
    cfg.dflash_bound = true;
    const kv8 = transformer_mod.KVQuantConfig.affine(8);
    try std.testing.expect(!cfg.rowExactDecode());
    try std.testing.expect(resolveKvAttnFused(&cfg, true, 8192, kv8));
    cfg.dflash_bound = false;
    try std.testing.expect(resolveKvAttnFused(&cfg, true, 8192, kv8));
}

test "resolveKvAttnFusedPure: explicit > mode; auto keys on scheme + crossover" {
    const t = std.testing;
    // Explicit per-request choice outranks every mode.
    try t.expect(resolveKvAttnFusedPure(.dense, true, 0, .off));
    try t.expect(!resolveKvAttnFusedPure(.fused, false, 1 << 20, .affine));
    // Fixed modes.
    try t.expect(!resolveKvAttnFusedPure(.dense, null, 1 << 20, .affine));
    try t.expect(resolveKvAttnFusedPure(.fused, null, 0, .affine));
    // Auto: affine + prompt at/above the crossover.
    try t.expect(resolveKvAttnFusedPure(.auto, null, KV_ATTN_AUTO_CROSSOVER_TOKENS, .affine));
    try t.expect(resolveKvAttnFusedPure(.auto, null, KV_ATTN_AUTO_CROSSOVER_TOKENS + 1, .affine));
    try t.expect(!resolveKvAttnFusedPure(.auto, null, KV_ATTN_AUTO_CROSSOVER_TOKENS - 1, .affine));
    // Auto never engages off (no triples to consume).
    try t.expect(!resolveKvAttnFusedPure(.auto, null, 1 << 20, .off));
}

test "formatChatUsage: prompt_tokens_details.cached_tokens always present (llmprobe chat caching)" {
    const t = std.testing;
    const a = t.allocator;

    // Cold request: field present with 0, matching OpenAI's unconditional emission.
    const cold = try formatChatUsage(a, 100, 5, 0, "");
    defer a.free(cold);
    try t.expectEqualStrings(
        \\{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":0}}
    , cold);

    // Warm request: cached count surfaces; cached ≤ prompt by construction
    // (scheduler folds cached into prompt_tokens).
    const warm = try formatChatUsage(a, 300, 12, 250, "");
    defer a.free(warm);
    try t.expectEqualStrings(
        \\{"prompt_tokens":300,"completion_tokens":12,"total_tokens":312,"prompt_tokens_details":{"cached_tokens":250}}
    , warm);

    // Reasoning responses append completion_tokens_details after the prompt details.
    const with_details = try formatChatUsage(a, 10, 20, 4, ",\"completion_tokens_details\":{\"reasoning_tokens\":7}");
    defer a.free(with_details);
    try t.expectEqualStrings(
        \\{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":7}}
    , with_details);
}

test "the out-of-memory 503 names the cap's flag and never blames concurrency" {
    // #126: an idle server with zero models loaded and 21 MB RSS answered
    // "retry after current requests complete", which is unactionable advice
    // pointing at the wrong subsystem — the cause is the static
    // --max-resident-mem cap, a flag the message never mentioned and the app
    // does not pass. Both 503 sites read the one constant.
    const m = not_enough_memory_message;
    try testing.expect(std.mem.indexOf(u8, m, "--max-resident-mem") != null);
    try testing.expect(std.mem.indexOf(u8, m, "retry after current requests") == null);
    // Only the resident-budget gate raises NotEnoughMemory; the free-memory
    // preflight has its own arm (insufficient_free_memory_message). Blaming
    // free memory here sends the user quitting apps when the fix is the cap.
    try testing.expect(std.mem.indexOf(u8, m, "free memory") == null);
    try testing.expect(std.mem.indexOf(u8, m, "resident-model budget") != null);

    const src = @embedFile("server.zig");
    const stale = "\"Not enough memory to load model; retry " ++ "after current requests complete\"";
    try testing.expect(std.mem.indexOf(u8, src, stale) == null);
    // Every ensureLoaded failure answers through `sendEnsureLoadedError`, the one site.
    var n: usize = 0;
    var i: usize = 0;
    const needle = "\"out_of_memory\", not_enough_memory" ++ "_message,";
    while (std.mem.indexOfPos(u8, src, i, needle)) |p| : (i = p + needle.len) n += 1;
    try testing.expectEqual(@as(usize, 1), n);
}

test "PrefillDoesNotFit quotes the figures the inference thread refused on, once" {
    const t = std.testing;
    const mb = 1024 * 1024;
    var buf: [320]u8 = undefined;
    prefill_refusal = .{ .tokens = 12447, .needed = 1733 * mb, .available = 77 * mb };
    const w = mapGenerationError(error.PrefillDoesNotFit, &buf);
    try t.expectEqual(@as(u32, 400), w.code);
    try t.expect(std.mem.indexOf(u8, w.message, "This prompt (12447 tokens) does not fit in GPU memory") != null);
    // The guard's wording, so a client parses both refusals with one pattern.
    try t.expect(std.mem.indexOf(u8, w.message, "requires ~1733MB GPU memory but only ~77MB is available") != null);
    // Read once: the next refusal without figures on this thread gets the bare message.
    try t.expect(prefill_refusal == null);
    try t.expectEqualStrings(PREFILL_NOFIT_MSG, mapGenerationError(error.PrefillDoesNotFit, &buf).message);
}

test "mapGenerationError: after a lost GPU context any failure is the named 503" {
    const t = std.testing;
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale);
    defer mlx.resetGpuContextLostForTest();
    var buf: [192]u8 = undefined;
    try t.expectEqual(@as(u32, 500), mapGenerationError(error.GenerationFailed, &buf).code);
    mlx.latchErrorForTest("cudaGraphLaunch(graph_exec, stream_) failed: an illegal memory access was encountered");
    _ = mlx.takeError(&stale);
    const w = mapGenerationError(error.GenerationFailed, &buf);
    try t.expectEqual(@as(u32, 503), w.code);
    try t.expectEqualStrings(gpu_context_lost_message, w.message);
}

test "a streaming fault answers with the SAME mapped error a non-streaming one does" {
    // The 400/503 mapping existed only on the non-streaming arms; streaming arms wrote a bare `server_error` frame.
    const t = std.testing;
    const a = t.allocator;

    var buf: [192]u8 = undefined;
    {
        const w = mapGenerationError(error.GenerationOutOfMemory, &buf);
        try t.expectEqual(@as(u32, 503), w.code);
        try t.expectEqualStrings("503 Service Unavailable", w.status_line);
        try t.expectEqualStrings(GEN_OOM_MSG, w.message);
        try t.expectEqualStrings("server_error", w.openai_type);
    }
    {
        const w = mapGenerationError(error.PrefillDoesNotFit, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings(PREFILL_NOFIT_MSG, w.message);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
    }
    {
        const w = mapGenerationError(error.GenerationFailed, &buf);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("generation failed", w.message);
    }
    {
        // A prefill that cannot return every prompt row is the request's shape, not a fault.
        const w = mapGenerationError(error.PromptLogprobsUnavailable, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
    }
    {
        // A prompt that encodes to no tokens is refused before a slot exists.
        const w = mapGenerationError(error.EmptyPrompt, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
    }
    {
        // Anything else keeps its name in the message, at a mapped status.
        const w = mapGenerationError(error.WriteFailed, &buf);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("Internal server error: WriteFailed", w.message);
    }

    // The SSE bodies carry the same type and message, plus the surface's terminal marker.
    {
        const w = mapGenerationError(error.GenerationOutOfMemory, &buf);
        const body = try buildOpenAiStreamError(a, w);
        defer a.free(body);
        try t.expect(std.mem.indexOf(u8, body, "\"type\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, body, "\"code\":503") != null);
        try t.expect(std.mem.indexOf(u8, body, GEN_OOM_MSG) != null);
        try t.expect(std.mem.indexOf(u8, body, "\"finish_reason\":\"error\"") != null);
    }
}

test "contextOverflowMessage: the 400 names both counts so a client can act on it" {
    const t = std.testing;
    var buf: [160]u8 = undefined;

    // The counts ARE the feature: without them a client can only say "too
    // long", and our own app can't offer "raise the context to N" — which is
    // the one action that fixes it.
    const msg = contextOverflowMessage(&buf, 4108, 4096);
    try t.expectEqualStrings("Prompt exceeds maximum context length: 4108 tokens requested, 4096 available", msg);

    // Legacy prefix preserved verbatim — APIError.looksLikeContextOverflow and
    // every third-party client key on this exact phrase.
    try t.expect(std.mem.startsWith(u8, msg, "Prompt exceeds maximum context length"));

    // No quotes or control bytes, so the message is safe to drop into a JSON
    // error body without escaping (the appendJsonString class).
    for (msg) |c| try t.expect(c >= 0x20 and c != '"' and c != '\\');
}

test "contextOverflowMessage: a buffer too small falls back rather than sending nothing" {
    const t = std.testing;
    // bufPrint failing must not propagate — the media-gen 400 that sent NO body
    // at all when its fixed buffer overflowed is the class this guards.
    var tiny: [8]u8 = undefined;
    try t.expectEqualStrings("Prompt exceeds maximum context length", contextOverflowMessage(&tiny, 999999, 1));
}

test "messageReasoningFromObj: reasoning_content round-trip, reasoning fallback, empty dropped" {
    const t = std.testing;
    const allocator = t.allocator;
    const cases = [_]struct { body: []const u8, want: ?[]const u8 }{
        // pi/vLLM SSE spelling - the live case (2026-07-29).
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"two plus two\"}", .want = "two plus two" },
        // vLLM request spelling.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning\":\"twice two\"}", .want = "twice two" },
        // reasoning_content wins when both are present.
        .{ .body = "{\"role\":\"assistant\",\"reasoning_content\":\"a\",\"reasoning\":\"b\"}", .want = "a" },
        // Empty string is NOT reasoning — it would render the empty
        // <think></think> nothink signature the field exists to avoid.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"\"}", .want = null },
        // Non-string shapes are ignored, never crash.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":{\"x\":1}}", .want = null },
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\"}", .want = null },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = messageReasoningFromObj(parsed.value.object);
        if (case.want) |w| {
            try t.expectEqualStrings(w, got.?);
        } else {
            try t.expect(got == null);
        }
    }
}

test "formatCompletionsLogprobs: an echoed prompt's first token carries null, the rest shift by one" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .sentencepiece_bpe);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    try tok.id_to_token.put(1, "The");
    try tok.id_to_token.put(2, " sky");
    try tok.id_to_token.put(3, " is");

    const token_ids = [_]u32{ 1, 2, 3 };
    var t1 = [_]generate_mod.TokenLogprob{.{ .token_id = 2, .logprob = -0.5 }};
    var t2 = [_]generate_mod.TokenLogprob{.{ .token_id = 3, .logprob = -1.0 }};
    const lps = [_]generate_mod.LogprobResult{
        .{ .token_logprob = -0.5, .top_logprobs = &t1 },
        .{ .token_logprob = -1.0, .top_logprobs = &t2 },
    };
    var tbase: usize = 0;
    const json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &lps, true, &tbase);
    defer allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    for ([_][]const u8{ "tokens", "token_logprobs", "top_logprobs", "text_offset" }) |k|
        try std.testing.expectEqual(@as(usize, 3), o.get(k).?.array.items.len);
    try std.testing.expect(o.get("token_logprobs").?.array.items[0] == .null);
    try std.testing.expect(o.get("top_logprobs").?.array.items[0] == .null);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), o.get("token_logprobs").?.array.items[1].float, 1e-6);
    try std.testing.expect(o.get("top_logprobs").?.array.items[2].object.get(" is") != null);
    try std.testing.expectEqual(@as(i64, 7), o.get("text_offset").?.array.items[2].integer);
}

test "formatCompletionsLogprobs: legacy shape, byte-aligned offsets, escaped tokens" {
    // /v1/completions ignored its `logprobs` field entirely (hardcoded 0) while
    // still emitting the key, so a client read "no alternatives exist" rather
    // than "never asked". Its shape is NOT chat's: four parallel arrays, and
    // `top_logprobs` is a map keyed by token TEXT.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmptyForTests(allocator, .sentencepiece_bpe);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    // Ids 1..4. Id 4 carries a control byte AND a quote — the escape has to be
    // the shared sink, not a hand-rolled table (the /detokenize class).
    try tok.id_to_token.put(1, "Paris");
    try tok.id_to_token.put(2, ",");
    try tok.id_to_token.put(3, " the");
    try tok.id_to_token.put(4, "a\x01\"b");

    const token_ids = [_]u32{ 1, 2, 3 };
    var t0 = [_]generate_mod.TokenLogprob{
        .{ .token_id = 1, .logprob = -0.25 },
        .{ .token_id = 4, .logprob = -3.5 },
    };
    var t1 = [_]generate_mod.TokenLogprob{.{ .token_id = 2, .logprob = -0.5 }};
    var t2 = [_]generate_mod.TokenLogprob{.{ .token_id = 3, .logprob = -1.0 }};
    const lps = [_]generate_mod.LogprobResult{
        .{ .token_logprob = -0.25, .top_logprobs = &t0 },
        .{ .token_logprob = -0.5, .top_logprobs = &t1 },
        .{ .token_logprob = -1.0, .top_logprobs = &t2 },
    };

    var tbase: usize = 0;
    const json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &lps, false, &tbase);
    defer allocator.free(json);

    // Must be parseable at all — the control byte is the reason that matters.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    // Four arrays, all the same length as the token list.
    for ([_][]const u8{ "tokens", "token_logprobs", "top_logprobs", "text_offset" }) |k| {
        const arr = o.get(k) orelse return error.MissingKey;
        try std.testing.expectEqual(@as(usize, 3), arr.array.items.len);
    }
    try std.testing.expectEqualStrings("Paris", o.get("tokens").?.array.items[0].string);
    // text_offset is the BYTE offset of each token within the completion, which
    // is what a FIM client aligns against — not a token index.
    const offs = o.get("text_offset").?.array.items;
    try std.testing.expectEqual(@as(i64, 0), offs[0].integer); // ""
    try std.testing.expectEqual(@as(i64, 5), offs[1].integer); // "Paris"
    try std.testing.expectEqual(@as(i64, 6), offs[2].integer); // "Paris,"
    // The chosen token's own logprob rides beside the alternatives.
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), o.get("token_logprobs").?.array.items[0].float, 1e-6);
    // top_logprobs is a MAP keyed by token text (OpenAI's shape), and the
    // control-byte token survived the round trip through the escaper.
    const top0 = o.get("top_logprobs").?.array.items[0].object;
    try std.testing.expectEqual(@as(usize, 2), top0.count());
    try std.testing.expect(top0.get("Paris") != null);
    try std.testing.expect(top0.get("a\x01\"b") != null);
}

test "logprobs token strings are valid UTF-8 — a split multi-byte token can't break the body" {
    // A single token is a BPE fragment, so it can carry HALF a multi-byte
    // character; `jsonEscape` passes every byte >= 0x20 through verbatim, so
    // those raw bytes landed in the JSON string and the WHOLE response body
    // stopped being valid UTF-8 — not a degraded field, an unparseable
    // response. Live on an oQ4e checkpoint: a `b"\xf0\x9f"` candidate
    // (the first 2 bytes of a 4-byte emoji) in `top_logprobs`.
    //
    // `bytes` already carries the truth, so the string is the lossy view:
    // U+FFFD per invalid sequence, which is what OpenAI's own shape does.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmptyForTests(allocator, .sentencepiece_bpe);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    // A SentencePiece vocab holds the raw bytes of a piece, so the split
    // sequences sit in the vocab as they are.
    try tok.id_to_token.put(1, "ok");
    // Leading half of U+1F600, exactly as the live checkpoint emitted it.
    try tok.id_to_token.put(2, "\xf0\x9f");
    // A lone continuation byte, and a 3-byte lead truncated to 2.
    try tok.id_to_token.put(3, "\x9f");
    try tok.id_to_token.put(4, "\xe2\x82");

    var t0 = [_]generate_mod.TokenLogprob{
        .{ .token_id = 1, .logprob = -0.1 },
        .{ .token_id = 2, .logprob = -9.0 },
        .{ .token_id = 3, .logprob = -9.5 },
        .{ .token_id = 4, .logprob = -9.9 },
    };
    const token_ids = [_]u32{1};

    // Chat shape.
    const chat_json = try formatTokenLogprob(allocator, &tok, 2, -9.0, &t0);
    defer allocator.free(chat_json);
    try std.testing.expect(std.unicode.utf8ValidateSlice(chat_json));
    const chat_parsed = try std.json.parseFromSlice(std.json.Value, allocator, chat_json, .{});
    defer chat_parsed.deinit();
    // The lossy string stands in for the split character...
    try std.testing.expectEqualStrings("\u{FFFD}", chat_parsed.value.object.get("token").?.string);
    // ...while `bytes` still reports exactly what the model emitted, so a
    // client that cares can reassemble it across tokens.
    const bytes = chat_parsed.value.object.get("bytes").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), bytes.len);
    try std.testing.expectEqual(@as(i64, 0xf0), bytes[0].integer);
    try std.testing.expectEqual(@as(i64, 0x9f), bytes[1].integer);

    // Legacy /v1/completions shape — same escaper, same guarantee. Its
    // `top_logprobs` is a map keyed by token TEXT, so it carries ONE invalid
    // candidate here: two would both render as U+FFFD and collide into a single
    // key. That collision is OpenAI's own shape reproduced faithfully, not a
    // defect this test should pin away.
    var comp_top = [_]generate_mod.TokenLogprob{
        .{ .token_id = 1, .logprob = -0.1 },
        .{ .token_id = 2, .logprob = -9.0 },
    };
    const comp_lps = [_]generate_mod.LogprobResult{.{ .token_logprob = -0.1, .top_logprobs = &comp_top }};
    var tbase: usize = 0;
    const comp_json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &comp_lps, false, &tbase);
    defer allocator.free(comp_json);
    try std.testing.expect(std.unicode.utf8ValidateSlice(comp_json));
    const comp_parsed = try std.json.parseFromSlice(std.json.Value, allocator, comp_json, .{});
    defer comp_parsed.deinit();
    // text_offset stays on the RAW byte length — it indexes the completion the
    // client received, which is unaffected by how the string renders.
    try std.testing.expectEqual(@as(usize, 2), tbase);

    // Valid UTF-8 is untouched, including multi-byte that arrives whole.
    try tok.id_to_token.put(5, "é€");
    const clean = try formatTokenLogprob(allocator, &tok, 5, -1.0, &[_]generate_mod.TokenLogprob{});
    defer allocator.free(clean);
    const clean_parsed = try std.json.parseFromSlice(std.json.Value, allocator, clean, .{});
    defer clean_parsed.deinit();
    try std.testing.expectEqualStrings("é€", clean_parsed.value.object.get("token").?.string);
}

test "contentTokenRange: logprobs cover the CONTENT, not the reasoning we stripped" {
    // `logprobs.content` is defined as the tokens of the message content, but we
    // built it from the whole generation — so on every model that thinks, entry
    // 0 was the first token of the *reasoning* and the array lined up with
    // nothing the client could see. Measured on three builds,
    // on gemma-4-31b and gemma-4-e4b and others.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmptyForTests(allocator, .sentencepiece_bpe);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    try tok.id_to_token.put(1, "<think>");
    try tok.id_to_token.put(2, "\nBecause.\n");
    try tok.id_to_token.put(3, "</think>");
    try tok.id_to_token.put(4, "Can");
    try tok.id_to_token.put(5, "berra");

    const token_ids = [_]u32{ 1, 2, 3, 4, 5 };
    const full = "<think>\nBecause.\n</think>Canberra";
    const content = full[("<think>\nBecause.\n</think>").len..];
    try std.testing.expectEqualStrings("Canberra", content);

    const r = contentTokenRange(allocator, &tok, &token_ids, full, content);
    try std.testing.expectEqual(@as(usize, 3), r.start);
    try std.testing.expectEqual(@as(usize, 5), r.end);

    // Whole text as content (a model that did not think) keeps every entry.
    const all = contentTokenRange(allocator, &tok, &token_ids, full, full);
    try std.testing.expectEqual(@as(usize, 0), all.start);
    try std.testing.expectEqual(@as(usize, 5), all.end);

    // Empty content (a turn that was all reasoning) emits no entries rather
    // than the reasoning's.
    const none = contentTokenRange(allocator, &tok, &token_ids, full, full[0..0]);
    try std.testing.expectEqual(@as(usize, 0), none.start);
    try std.testing.expectEqual(@as(usize, 0), none.end);

    // A content slice we cannot locate keeps the full range: an array we can't
    // align is still better than dropping it silently.
    const foreign = contentTokenRange(allocator, &tok, &token_ids, full, "not in there");
    try std.testing.expectEqual(@as(usize, 0), foreign.start);
    try std.testing.expectEqual(@as(usize, 5), foreign.end);

    // A token straddling the boundary belongs to content — it put bytes there.
    const mid = contentTokenRange(allocator, &tok, &token_ids, full, full[("<think>\nBecause.\n</thi").len..]);
    try std.testing.expectEqual(@as(usize, 2), mid.start);
}

test "request body cap is one generous cap for every route" {
    for ([_][]const u8{ "/v1/chat/completions", "/v1/embeddings", "/", "" }) |p|
        try std.testing.expectEqual(max_request_bytes, maxRequestBytesFor(p));
}

test "the chat body cap admits an agent session that keeps its screenshots" {
    // 100 screenshots at 4 MiB of base64 each.
    const screenshots = 100;
    const base64_png = 4 * 1024 * 1024;
    for ([_][]const u8{"/v1/chat/completions"}) |p|
        try std.testing.expect(maxRequestBytesFor(p) >= screenshots * base64_png);
}

test "the 413 names both counts it compared" {
    var buf: [128]u8 = undefined;
    const msg = payloadTooLargeMessage(&buf, 97 * 1024 * 1024 + 1, 64 * 1024 * 1024);
    try std.testing.expect(std.mem.indexOf(u8, msg, "98 MB") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "64 MB") != null);
}

/// The generic admission bill for the OOM test config, minus the live-memory read a unit test cannot make.
fn oomBillFor(cfg: *const model_mod.ModelConfig, seq: u64, chunk: u64, terms: PrefillRequestTerms) u64 {
    return prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(cfg), prefillDequantWeightBytes(cfg), terms) +
        0;
}

test "the guard credits only PROVABLY reclaimable cache bytes, never the entry a restore will pin" {
    const t = std.testing;
    // The guard admitted on `available + evictable >= needed`, but the evictable 1,564 MB was
    // the one entry this prompt then restored from; the provable credit is residency minus the largest entry.
    const MB: u64 = 1024 * 1024;
    var bill = AdmissionBill{ .needed = 37_394 * MB, .available = 37_130 * MB, .evictable = 1_564 * MB, .reclaimable = 0 };
    try t.expect(!bill.fits());
    try t.expect(!bill.fitsAfterEviction()); // it DID, before this fix
    bill.reclaimable = 1_564 * MB;
    try t.expect(bill.fitsAfterEviction());
}

test "ssdFirstPrefixCacheMem: the active session's KV is billed ONCE and is the floor" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The 1M case: 107.5 GB ceiling, ~70 GB weights, ~24 GB of 1M-ctx KV, ~3 GB reserve.
    const ceiling: u64 = 107 * GB;
    const weights: u64 = 70 * GB;
    const ctx_kv: u64 = 24 * GB;
    const transient: u64 = 3 * GB;

    // RAM-first clamps the cache to the leftover: less than half an entry.
    const ram_first = clampedPrefixCacheMem(0, ceiling, weights, ctx_kv, transient);
    try t.expect(ram_first < ctx_kv);

    // SSD-first floors at one entry and hands `--prefix-cache-mem` to idle entries on top.
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(0, ceiling, weights, ctx_kv, transient));
    try t.expectEqual(ctx_kv + 4 * GB, ssdFirstPrefixCacheMem(4 * GB, ceiling, weights, ctx_kv, transient));
    const huge = ssdFirstPrefixCacheMem(500 * GB, ceiling, weights, ctx_kv, transient);
    try t.expectEqual(ctx_kv + (ceiling - weights - ctx_kv - transient), huge);
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(8 * GB, 80 * GB, weights, ctx_kv, transient));
    try t.expect(ssdFirstPrefixCacheMem(0, 0, 0, 0, 0) >= 1);
}

/// The widest width the adaptive chooser may take for a prompt on this config.
fn adaptCapFor(cfg: *const model_mod.ModelConfig, seq: u64) u32 {
    return widthForRung(cfg, seq, PREFILL_CHUNK_LADDER[0]);
}

test "prefillHeadroomNow subtracts the SSD writer's staged HOST bytes" {
    // Staged host bytes come out of the same unified pool the widen bar reads.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    try t.expectEqual(90 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, 0));
    try t.expectEqual(89 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, GiB));
    // Saturating: a permit larger than the headroom refuses every widen.
    try t.expectEqual(@as(u64, 0), prefillHeadroomFrom(120 * GiB, 119 * GiB, 8 * GiB));
}

test "resolvedContextForLoad: an auto boot bills the session it will serve, not the placeholder" {
    // On an auto boot the effective-context accessor answered a 1024-token placeholder during the load.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    // This box: ~28,909 MiB after the weights, 20,736 B/token at 1M/kv8, ~3 GiB transient.
    const per_tok: u64 = 20_736;
    const active: u64 = 70_000 * MiB;
    const ceiling: u64 = active + 28_909 * MiB;
    const transient: u64 = 3 * 1024 * MiB;
    const cap: u32 = 1_048_576;

    // Auto boot, nothing pinned: the session is what the machine can serve. `cache_reserve = 0`
    // is a test isolation.
    const auto = resolvedContextForLoad(0, 0, ceiling, active, 0, transient, per_tok, cap);
    try t.expect(auto > 100_000);
    try t.expect(auto <= cap);

    // The placeholder bills ~26 MB of session and hands the idle allowance almost the whole machine.
    const placeholder_kv: u64 = per_tok * 1024;
    const real_kv: u64 = per_tok * auto;
    try t.expect(placeholder_kv < 30 * MiB);
    try t.expect(real_kv > 100 * placeholder_kv);
    const ask: u64 = 24 * 1024 * MiB;
    const bogus = ssdFirstPrefixCacheMem(ask, ceiling, active, placeholder_kv, transient);
    const fixed = ssdFirstPrefixCacheMem(ask, ceiling, active, real_kv, transient);
    try t.expect(bogus -| placeholder_kv == ask);
    // Under the real session bill the same ask is clamped to what is left after the weights, one
    // session and the transient (the helper's own contract, not a hand-picked byte count).
    const idle_headroom: u64 = ceiling - (active + real_kv + transient);
    try t.expectEqual(real_kv + idle_headroom, fixed);
    try t.expect(fixed -| real_kv < ask);
    try t.expect(fixed -| real_kv < (bogus -| placeholder_kv) / 2);

    // An explicit --ctx-size wins outright, and a pinned context is used as-is.
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(262_144, 0, ceiling, active, 0, transient, per_tok, cap));
    try t.expectEqual(@as(u32, 131_072), resolvedContextForLoad(0, 131_072, ceiling, active, 0, transient, per_tok, cap));
    try t.expectEqual(cap, resolvedContextForLoad(0, 0, active + 900_000 * MiB, active, 0, transient, per_tok, cap));
}

test "the published hot-cache budget is retired when the cache is dropped" {
    // The budget is per model, the global is process-wide.
    const t = std.testing;
    publishResolvedPrefixCacheMem(4096);
    try t.expectEqual(@as(u64, 4096), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    // Back to the ask, the honest pre-load answer.
    try t.expectEqual(prefix_cache_mem_bytes, resolvedPrefixCacheMem());

    // A resolved zero is a real budget (`--prefix-cache-mem 0` under SSD-first); the sentinel is maxInt.
    publishResolvedPrefixCacheMem(0);
    try t.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    try t.expectEqual(prefix_cache_mem_bytes, resolvedPrefixCacheMem());
}

test "resolvedPrefixCacheMem: RAM-off bills zero despite a configured disk tier" {
    const capacity = prefix_cache_capacity;
    const ram = prefix_cache_ram_enabled;
    defer prefix_cache_capacity = capacity;
    defer prefix_cache_ram_enabled = ram;
    prefix_cache_capacity = 8;
    prefix_cache_ram_enabled = false;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    prefix_cache_ram_enabled = true;
    prefix_cache_capacity = 0;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
}

/// The live 768k scenario's checkout inputs, as `prefix_cache.checkoutEligible` takes them.
const CheckoutCase = struct {
    ssd_first: bool = true,
    move_enabled: bool = true,
    pending_disk: bool = false,
    entry_tokens: usize = 786_695,
    shared: usize = 786_695,
    prompt_len: usize = 786_707,
    has_slot: bool = true,

    /// The WarmPrefix the scheduler builds, with `will_donate` from restore-move's own predicate.
    fn warm(self: CheckoutCase, matched: u64, capacity: u64) WarmPrefix {
        return .{
            .matched_tokens = matched,
            .capacity_tokens = capacity,
            .will_donate = prefix_cache_mod.HotPrefixCache.checkoutEligible(
                self.ssd_first,
                self.move_enabled,
                self.pending_disk,
                self.entry_tokens,
                self.shared,
                self.prompt_len,
                self.has_slot,
            ),
        };
    }
};

test "growCoexistBytes: a MoE is billed the whole old cache on a growing append" {
    // The switch turns the predicate off for the whole process (cached env read): nothing to test then.
    if (std.c.getenv("MLX_SERVE_MOE_SHORT_PREFILL_ONE_EVAL_OFF") != null) return error.SkipZigTest;
    // Its short forwards run without the mid-loop cadence, so the old buffer can live the whole loop.
    const t = std.testing;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 30;
    cfg.num_experts = 128;
    const kv: u64 = 1000;
    const warm = WarmPrefix{ .matched_tokens = 6000, .capacity_tokens = 6000, .will_donate = true };
    try t.expectEqual(6000 * kv, growCoexistBytes(&cfg, warm, 6300, kv));
    try t.expectEqual(6000 * kv, growCoexistBytes(&cfg, warm, 10096, kv));
    // Not growing: nothing.
    try t.expectEqual(@as(u64, 0), growCoexistBytes(&cfg, warm, 5000, kv));
}

test "pinnedResidentBytes names the entry a restore would share, and only that" {
    const t = std.testing;
    // Residency the eviction pass cannot return: the connection thread's one signal that this prompt is warm.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 0, .reclaimable = 0 }));
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 4096, .reclaimable = 4096 }));
    // The live 768k reading: 24,826 MB resident, 10,294 MB reclaimable.
    const mb: u64 = 1024 * 1024;
    try t.expectEqual(@as(u64, 14_532 * mb), pinnedResidentBytes(.{ .evictable = 24_826 * mb, .reclaimable = 10_294 * mb }));
    // Saturating: the two numbers are published from different sites.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 1, .reclaimable = 2 }));
}

test "prefillAdmissionBill: both admit arms are driven by the two credits" {
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const evict = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 15 * MB };
    try t.expect(!evict.fits());
    try t.expect(evict.fitsAfterEviction());
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(evict));

    const deferral = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 0 };
    try t.expect(!deferral.fitsAfterEviction());
    try t.expect(pinnedResidentBytes(deferral) > 0); // warm deferral

    // An empty cache carries neither credit, and both arms go dead.
    const empty = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB };
    try t.expectEqual(empty.fits(), empty.fitsAfterEviction());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(empty));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(empty));
    // A bill that fits still admits on both arms.
    const roomy = AdmissionBill{ .needed = 10 * MB, .available = 20 * MB };
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(roomy));
}

test "memoryRefusalMessage: the 400 quotes the hot-cache credits the guard compared" {
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    });
    defer t.allocator.free(rich);
    try t.expect(std.mem.indexOf(u8, rich, "holds ~1500MB more") != null);
    try t.expect(std.mem.indexOf(u8, rich, "(~900MB") != null);

    const cold = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB });
    defer t.allocator.free(cold);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB is available and the hot prefix cache holds ~0MB more, all of which can be reclaimed (~0MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.",
        cold,
    );
}

test "the MLX buffer pool is inside `available` by construction, so a refusal cannot buy it back" {
    // `currentGpuMemoryCeiling` adds the pool to the footprint before the physical minimum, so on
    // the physical arm `available` is `cache_bytes + free_system` and a clear moves it by zero.
    const t = std.testing;
    const wired: u64 = 120 << 30;
    const active: u64 = 71_310 << 20;
    const pool: u64 = 8_268 << 20;
    const free_system: u64 = 12_546 << 20;
    const ceiling = physicalMemoryCeiling(wired, active + pool, free_system);
    try t.expect(ceiling < wired); // the physical arm, as on the box
    try t.expectEqual(pool + free_system, ceiling - active);
    const cleared = physicalMemoryCeiling(wired, active, free_system + pool);
    try t.expectEqual(ceiling, cleared);
}

test "a request integer from a client is clamped, never blindly @intCast" {
    // Bar: `seed: -1`, a 2^32 max_tokens and an out-of-range budget resolve without a cast.
    const t = std.testing;
    try t.expectEqual(@as(?u64, null), parseRequestSeed(.{ .integer = -1 }));
    try t.expectEqual(@as(?u64, 0), parseRequestSeed(.{ .integer = 0 }));
    try t.expectEqual(@as(?u64, 42), parseRequestSeed(.{ .integer = 42 }));
    try t.expectEqual(@as(?u64, null), parseRequestSeed(.{ .string = "42" }));
    try t.expectEqual(@as(?u64, null), parseRequestSeed(null));

    try t.expectEqual(@as(u32, std.math.maxInt(u32)), resolveRequestMaxTokens(.{ .integer = 1 << 32 }, 7));
    try t.expectEqual(@as(u32, 512), resolveRequestMaxTokens(.{ .integer = 512 }, 7));
    try t.expectEqual(@as(u32, 7), resolveRequestMaxTokens(.{ .integer = 0 }, 7));

    try t.expectEqual(@as(i32, std.math.maxInt(i32)), clampJsonI32(1 << 40));
    try t.expectEqual(@as(i32, std.math.minInt(i32)), clampJsonI32(-(1 << 40)));
    try t.expectEqual(@as(i32, -1), clampJsonI32(-1));
    try t.expectEqual(@as(i32, 1024), clampJsonI32(1024));
}

test "a vanished client is not a token cap: the non-stream finish reason says so" {
    // Bar: the non-streaming path reports what the three streaming paths do.
    const t = std.testing;
    try t.expectEqualStrings("client_disconnect", nonStreamFinishReason(true, "length"));
    try t.expectEqualStrings("stop", nonStreamFinishReason(false, "stop"));
    try t.expectEqualStrings("length", nonStreamFinishReason(false, "length"));
}

test "a tool-call answer keeps the model's visible preamble" {
    // Bar: non-streamed content == what the stream flushes before the tool markup.
    const t = std.testing;
    const raw = "<think>pick a tool</think>Checking the weather.\n<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>";
    const split = chat_mod.splitThinkBlock(raw, true, false);
    try t.expectEqualStrings("Checking the weather.", visibleToolPreamble(split.content));
    const bare = chat_mod.splitThinkBlock("<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>", true, false);
    try t.expectEqualStrings("", visibleToolPreamble(bare.content));
    try t.expectEqualStrings("", visibleToolPreamble("  \n "));
}

test "a streaming stop sequence cuts at the match, not at the token boundary" {
    // Bar: the streamed bytes are the bytes non-streaming keeps (`text[0..idx]`).
    const t = std.testing;
    const stops = [_][]const u8{"\n\nHuman:"};
    // The stop opens inside the token that completes it: the leading bytes of
    // that token are answer text and must still be delivered.
    const buf = "Paris.\n\nHuman:";
    const cut = stopSequenceCut(buf, "s.\n\nHuman:".len, &stops).?;
    try t.expectEqual(@as(usize, 6), cut.index);
    try t.expectEqual(@as(usize, 2), cut.token_keep);
    try t.expectEqualStrings("\n\nHuman:", cut.matched);
    // Match entirely inside already-delivered bytes: nothing of this token survives.
    const late = stopSequenceCut(buf, 1, &stops).?;
    try t.expectEqual(@as(usize, 6), late.index);
    try t.expectEqual(@as(usize, 0), late.token_keep);
    // Earliest match wins, and a clean buffer cuts nothing.
    const many = [_][]const u8{ "END", "N" };
    try t.expectEqual(@as(usize, 1), stopSequenceCut("aNbEND", 6, &many).?.index);
    try t.expectEqual(@as(?StopCut, null), stopSequenceCut("all clear", 3, &stops));
}

test "a stop string matches the answer, never the reasoning" {
    const t = std.testing;
    const stops = [_][]const u8{"the"};
    // Template-opened thought: nothing matches until the block closes.
    try t.expectEqual(@as(?StopCut, null), answerStopCut("so the answer", 13, &stops, true));
    try t.expectEqual(@as(?StopCut, null), answerStopCut("<think>so the answer", 20, &stops, false));
    // The token straddling the close keeps its bytes up to the match in the answer.
    const buf = "so the</think>\nIn the";
    const cut = answerStopCut(buf, "</think>\nIn the".len, &stops, true).?;
    try t.expectEqual(std.mem.lastIndexOf(u8, buf, "the").?, cut.index);
    try t.expectEqual("</think>\nIn ".len, cut.token_keep);
    // No thought at all: the whole output is the answer.
    try t.expectEqual(@as(usize, 3), answerStopCut("in the", 6, &stops, false).?.index);
}

test "generated think tags require an unambiguous literal template opener" {
    try std.testing.expectEqualStrings("<think>", templateThinkOpener("{{ '<think>' }} <think>").?);
    try std.testing.expect(templateThinkOpener("<think> <think:opensource>") == null);
    try std.testing.expect(templateThinkOpener("<think:{{ suffix }}>") == null);
    try std.testing.expect(templateThinkOpener("assistant") == null);
}

test "defaultPrefixCacheAsk: an unnamed budget holds one session at the working context, never under 2 GB" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576; // a 262k-token session, ~6 GB
    try t.expectEqual(session, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, session));
    // A short working context keeps the old floor.
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, gb));
    // An operator's number stands, even one equal to the default.
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, true, session));
    try t.expectEqual(32 * gb, defaultPrefixCacheAsk(32 * gb, true, session));
    try t.expectEqual(@as(u64, 0), defaultPrefixCacheAsk(0, true, session));
    // An embedder's own ask (ios_lib passes 256 MB) is not the default and stands.
    try t.expectEqual(256 * 1024 * 1024, defaultPrefixCacheAsk(256 * 1024 * 1024, false, session));
}

test "defaultPrefixCacheAsk: the machine's headroom still caps the defaulted ask" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576;
    const ask = defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, session);
    // 16 GB ceiling, 11 GB weights, 1 GB live context, 1.5 GB prefill reserve: 2.5 GB left.
    try t.expectEqual(gb * 5 / 2, clampedPrefixCacheMem(ask, 16 * gb, 11 * gb, gb, gb * 3 / 2));
    // A big machine gets the whole session.
    try t.expectEqual(session, clampedPrefixCacheMem(ask, 200 * gb, 80 * gb, gb, 2 * gb));
}

test "a streamed turn's result keeps the prompt-cache hit and timings the slot measured" {
    const t = std.testing;
    const ts: StreamingTokenStream = .{ .mode = .regular, .eos_token_ids = &.{}, .prompt_tokens = 3016, .cached_tokens = 2985, .completion_tokens = 40, .prefill_ns = 90_000_000, .decode_ns = 1_200_000_000 };
    var text = "ok".*;
    var ids = [_]u32{ 7, 8 };
    const r = ts.generationResult(&text, &ids, "stop", null);
    try t.expectEqual(@as(u32, 2985), r.cached_tokens);
    try t.expectEqual(ts.prefill_ns, r.prefill_ns);
    try t.expectEqual(ts.decode_ns, r.decode_ns);
}

test "Conn names the client from each request's User-Agent and keeps no header text" {
    const peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 10, 0, 0, 9 }, .port = 4242 } };
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = -1, .address = peer } }, std.Io.Threaded.global_single_threaded.io());
    try std.testing.expectEqual(instr.Client.other, conn.client);

    conn.noteRequestHeaders("POST /v1/chat/completions HTTP/1.1\r\nUser-Agent: opencode/1.4\r\n\r\n");
    try std.testing.expectEqual(instr.Client.opencode, conn.client);
    conn.noteRequestHeaders("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n");
    try std.testing.expectEqual(instr.Client.other, conn.client);
}

test "outcome row 7/9: countRejected moves the rejected counter, and does nothing with metrics off" {
    const t = std.testing;
    const saved = g_metrics;
    defer g_metrics = saved;

    g_metrics = null;
    countRejected();

    var m = instr.Metrics.init();
    g_metrics = &m;
    countRejected();
    try t.expectEqual(@as(u64, 1), m.requests_rejected_total.load());
    try t.expectEqual(@as(u64, 0), m.requests_failed_total.load());
    try t.expectEqual(@as(u64, 0), m.requests_cancelled_total.load());
}

test "an absent reasoning budget arms an unreachable bound, a set one keeps its value" {
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), thinkBoundBudget(-1));
    try std.testing.expectEqual(@as(u32, 0), thinkBoundBudget(0));
    try std.testing.expectEqual(@as(u32, 2048), thinkBoundBudget(2048));
}

test "kvBytesPerToken bills only the CACHING layers, at the arch's own K and V widths" {
    const t = std.testing;
    // Uniform arch: every layer caches, K and V are both head_dim wide.
    var dense = model_mod.ModelConfig{ .model_type = "gemma4" };
    dense.num_hidden_layers = 24;
    dense.num_key_value_heads = 8;
    dense.head_dim = 128;
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), dense.kvBytesPerToken());

    // An arch bills its KV heads, grouping and all.
    var gqa = dense;
    gqa.num_attention_heads = 32;
    gqa.num_key_value_heads = 8;
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), gqa.kvBytesPerToken());
}

test "prefillTransientReserveAtKv: a Q8 KV cache adds the dequantized-window reserve over the dense one" {
    const t = std.testing;
    var trunk = model_mod.ModelConfig{ .model_type = "gemma4" };
    trunk.num_key_value_heads = 8;
    trunk.head_dim = 128;
    const dense = prefillTransientReserveAtKv(&trunk, 16, 512, 131072);
    const q8 = prefillTransientReserveAtKv(&trunk, 8, 512, 131072);
    try t.expectEqual(@as(u64, 131072 * 2 * 8 * 128 * 2 * 5 / 4), q8 - dense);
}

test "resolvePrefillChunk: a dense arch keeps the 8192 rung" {
    const dense: model_mod.ModelConfig = .{ .model_type = "gemma4", .head_dim = 128, .num_attention_heads = 32, .num_key_value_heads = 8, .hidden_size = 4096, .intermediate_size = 14336, .num_hidden_layers = 32 };
    const huge: u64 = 1 << 50;
    try std.testing.expectEqual(@as(u32, 8192), resolvePrefillChunk(&dense, 16, huge, 0, 0, 0));
}
