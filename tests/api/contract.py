#!/usr/bin/env python3
"""API contract test for the engine's OpenAI-compatible server. Stdlib only.

    python3 contract.py --binary <path> --model <dir> --port <n> --log-dir <dir> [--rows <file>]
                        [--env KEY=VALUE ...] [--text-only]

It starts and stops its own servers with the flags of the best engine build
(`serve --model <dir> --host 127.0.0.1 --port <n> --prefill-chunk 2048 --no-pld`, plus `--prefix-cache-disk 2GB` for
the SSD test and the `--env` entries added to the environment), so one command tests one binary.

Per test it prints `PASS <name>` or `FAIL <name>: <why>` and appends `{"test", "ok", "seconds"}` to `--rows`. Last line
`contract: <passed> of <total> passed`. Exit 0 when all pass, 1 when any fails, 2 when it could not test (the port is
held, a server does not answer `GET /v1/models` within 600 s, or `~/.mlx-serve/kv-cache` exists at start).

Every request is greedy (`temperature: 0`) with `enable_thinking` false unless a test says otherwise. Prompts are built in
code from neutral filler: no data file and no text from any bench or session.
"""
import argparse
import http.client
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HOST = "127.0.0.1"
KV_CACHE = Path.home() / ".mlx-serve" / "kv-cache"
MLX_SERVE_DIR = Path.home() / ".mlx-serve"
READY_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 900
# Environment switches that change engine behaviour; cleared so only --env decides (a copy of the job scripts' unset list).
UNSET = (
    "MLX_SERVE_KV_REUSE_OFF MLX_SERVE_SEAM_ROUTER_NORM_OFF MLX_SERVE_ROUTER_KERNEL_OFF MLX_SERVE_ROUTER_KERNEL_CHECK "
    "MLX_SERVE_LATIN_HEAD_OFF MLX_SERVE_QK_JOINED_OFF MLX_SERVE_HEAD_TAIL_OFF MLX_SERVE_KV_WRITE_IN_PLACE_OFF "
    "MLX_SERVE_MOE_DOWN_ROWS4_OFF MLX_SERVE_LAYER_PREFETCH_OFF "
    "MLX_SERVE_MOE_SHORT_PREFILL_ONE_EVAL_OFF MLX_SERVE_MOE_PREFILL_REDUCE_2816_OFF "
    "MLX_SERVE_MOE_PREFILL_FUSED_OFF MLX_SERVE_MOE_PREFILL_FUSED_CHECK "
    "MLX_SERVE_DECODE_OPS_ONLY_COMMIT_OFF MLX_SERVE_KEEP_WARM_OFF MLX_SERVE_KEEP_WARM_MS MLX_SERVE_TOPK_SCHEDULE "
    "MLX_MAX_OPS_PER_BUFFER MLX_MAX_MB_PER_BUFFER MLX_SERVE_SIMD_QMM_DECODE MLX_SDPA_BLOCKS "
    "MLX_SDPA_D512_MIN_KL MLX_SERVE_MOE_AFFINE4 MLX_SERVE_MOE_BANK_VIEWS"
).split()

JSON_SCHEMA_TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "count": {"type": "integer", "minimum": 1, "maximum": 9},
        "kind": {"type": "string", "enum": ["small", "medium", "large"]},
    },
    "required": ["title", "count", "kind"],
    "additionalProperties": False,
}


class CouldNotTest(Exception):
    """The harness could not run the test (exit 2), as opposed to the engine failing it (exit 1)."""


class Fail(Exception):
    """A test assertion failed."""


# ── pure helpers (tested offline) ───────────────────────────────────────────────────────────────────────────────────

def validate(obj, schema, path="$"):
    """Validate `obj` against the JSON Schema subset the json_schema test sends. Returns a list of error strings."""
    errs = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            return [f"{path}: expected object"]
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in obj:
                errs.append(f"{path}.{name}: required property missing")
        if schema.get("additionalProperties") is False:
            for name in obj:
                if name not in props:
                    errs.append(f"{path}.{name}: additional property not allowed")
        for name, sub in props.items():
            if name in obj:
                errs.extend(validate(obj[name], sub, f"{path}.{name}"))
        return errs
    if t == "string":
        if not isinstance(obj, str):
            errs.append(f"{path}: expected string")
    elif t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            errs.append(f"{path}: expected integer")
    elif t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            errs.append(f"{path}: expected number")
    elif t == "boolean":
        if not isinstance(obj, bool):
            errs.append(f"{path}: expected boolean")
    elif t == "array":
        if not isinstance(obj, list):
            return [f"{path}: expected array"]
        for i, item in enumerate(obj):
            errs.extend(validate(item, schema.get("items", {}), f"{path}[{i}]"))
        return errs
    if errs:
        return errs
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} is not one of {schema['enum']}")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} below minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} above maximum {schema['maximum']}")
    return errs


def parse_sse(lines):
    """Parse the `data:` lines of a server-sent-events stream into dicts; stops at `data: [DONE]`."""
    out = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        out.append(json.loads(payload))
    return out


def is_count(x, minimum=0):
    """An integer token count (never a bool) of at least `minimum`."""
    return isinstance(x, int) and not isinstance(x, bool) and x >= minimum


def valid_usage(u):
    """A usage object with real token counts: prompt and completion above 0 (the final streamed usage chunk and the
    non-stream usage both carry them for a request that generated text)."""
    return isinstance(u, dict) and is_count(u.get("prompt_tokens"), 1) and is_count(u.get("completion_tokens"), 1)


def finite_numbers(xs):
    """A non-empty list of finite int/float values (a bool is not a number here)."""
    return (isinstance(xs, list) and bool(xs)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs))


def cache_usage_problem(usage, min_cached=1, min_prompt=1):
    """Why a usage object does not report a consistent prefix-cache hit, or None when it does: prompt_tokens a count of
    at least `min_prompt`, and prompt_tokens_details.cached_tokens a count of at least `min_cached` and never above
    prompt_tokens (a cache cannot cover more tokens than the prompt has)."""
    if not isinstance(usage, dict):
        return f"usage is not an object: {usage!r}"
    prompt = usage.get("prompt_tokens")
    if not is_count(prompt, min_prompt):
        return f"prompt_tokens {prompt!r} is not a count of at least {min_prompt}"
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        return f"no prompt_tokens_details.cached_tokens in usage: {usage}"
    cached = details["cached_tokens"]
    if not is_count(cached, min_cached):
        return f"cached_tokens {cached!r} is not a count of at least {min_cached}"
    if cached > prompt:
        return f"cached_tokens {cached} is above prompt_tokens {prompt}"
    return None


def restore_problem(usage):
    """Why a usage object does not prove a restore from the SSD tier, or None when it does: prompt_tokens a count of at
    least 500, and prompt_tokens_details.cached_tokens a count of at least 1, at most prompt_tokens, covering at least
    75% of the prompt."""
    problem = cache_usage_problem(usage, min_cached=1, min_prompt=500)
    if problem is not None:
        if isinstance(usage, dict) and not is_count(usage.get("prompt_tokens"), 500):
            problem += " (the test needs about 1,000)"
        return problem
    prompt = usage["prompt_tokens"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    if 4 * cached < 3 * prompt:
        return f"cached_tokens {cached} of {prompt} prompt tokens (under 75%)"
    return None


FINISH_REASONS = ("stop", "length", "tool_calls")


def filler(n_sentences, salt=0):
    """Deterministic neutral filler text of about 8 tokens a sentence: a numbered inventory with no real-world content."""
    return " ".join(
        f"Bin {i + salt} holds {(i * 7 + salt) % 53 + 1} units of stock code {(i * 13 + salt) % 97:02d}."
        for i in range(n_sentences)
    )


# ── servers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

LIVE = []
# Servers whose process had already exited, without a stop from us, when stop() was entered: (server, returncode).
# run_one resets it per test, so a crash that a test's own cleanup stopped (removing it from LIVE) is still seen.
DEAD = []


def port_held(port):
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex((HOST, port)) == 0


class Server:
    def __init__(self, args, log_name, extra_args=(), extra_env=None):
        self.args = args
        self.log_path = Path(args.log_dir) / log_name
        self.extra_args = list(extra_args)
        self.extra_env = dict(extra_env or {})
        self.proc = None
        self.killed = False
        self.stopped = False

    def start(self):
        if port_held(self.args.port):
            raise CouldNotTest(f"port {self.args.port} is held")
        env = {k: v for k, v in os.environ.items() if k not in UNSET}
        env.update(self.args.env_map)
        env.update(self.extra_env)
        cmd = [self.args.binary, "serve", "--model", self.args.model, "--host", HOST, "--port", str(self.args.port),
               "--prefill-chunk", "2048", "--no-pld", *self.extra_args]
        self.log_file = None
        try:
            self.log_file = open(self.log_path, "wb")
            self.proc = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
        except OSError as e:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            raise CouldNotTest(f"server did not start ({e})")
        LIVE.append(self)
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise CouldNotTest(f"server exited with {self.proc.returncode} before answering /v1/models (see {self.log_path})")
            try:
                status, _, _ = http_json(self.args.port, "GET", "/v1/models", timeout=5)
                if status == 200:
                    return self
            except (OSError, http.client.HTTPException, ValueError):
                pass
            time.sleep(2)
        raise CouldNotTest(f"no answer from /v1/models within {READY_TIMEOUT_S} s (see {self.log_path})")

    def stop(self, grace=60):
        """SIGTERM to the server's own PID; SIGKILL after `grace` s (then `killed` is True). Returns the exit code."""
        if self.proc is None:
            return None
        if self.proc.poll() is not None:
            if not self.stopped:
                DEAD.append((self, self.proc.returncode))
        else:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.killed = True
                self.proc.kill()
                self.proc.wait()
        self.stopped = True
        if self in LIVE:
            LIVE.remove(self)
        self.log_file.close()
        return self.proc.returncode

    def log_text(self, start=0):
        with open(self.log_path, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace")

    def log_size(self):
        return self.log_path.stat().st_size


def stop_all(*_):
    for s in list(LIVE):
        try:
            s.stop(grace=15)
        except Exception:
            pass


# ── HTTP ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def http_json(port, method, path, body=None, timeout=REQUEST_TIMEOUT_S):
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data) if data else None
        except ValueError:
            parsed = data.decode("utf-8", "replace")
        return resp.status, dict(resp.getheaders()), parsed
    finally:
        conn.close()


def http_stream(port, path, body, timeout=REQUEST_TIMEOUT_S):
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return resp.status, resp.read().decode("utf-8", "replace"), False
        lines = []
        done = False
        while True:
            line = resp.readline()
            if not line:
                break
            lines.append(line.decode("utf-8", "replace"))
            if lines[-1].strip() == "data: [DONE]":
                done = True
                break
        return resp.status, parse_sse(lines), done
    finally:
        conn.close()


def chat_body(model, messages, **extra):
    body = {"model": model, "messages": messages, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra)
    return body


def expect(cond, why):
    if not cond:
        raise Fail(why)


# ── tests ───────────────────────────────────────────────────────────────────────────────────────────────────────────

class Ctx:
    pass


QUESTION = [{"role": "user", "content": "Name three kinds of citrus fruit in one short sentence."}]


def t_models(ctx):
    status, _, body = http_json(ctx.port, "GET", "/v1/models")
    expect(status == 200, f"status {status}")
    mid = body["data"][0]["id"]
    expect(isinstance(mid, str) and mid, "data[0].id is not a non-empty string")
    ctx.model = mid


def t_chat(ctx):
    ctx.chat_log_start = ctx.main.log_size()
    status, _, body = http_json(ctx.port, "POST", "/v1/chat/completions", chat_body(ctx.model, QUESTION, max_tokens=64))
    expect(status == 200, f"status {status}: {body}")
    choice = body["choices"][0]
    msg = choice["message"]
    expect(isinstance(msg.get("content"), str) and msg["content"].strip(), "empty message.content")
    expect(choice["finish_reason"] in ("stop", "length"), f"finish_reason {choice['finish_reason']!r}")
    usage = body["usage"]
    expect(valid_usage(usage), f"usage {usage}")
    expect(not (msg.get("reasoning_content") or "").strip(), "reasoning_content is non-empty with thinking off")
    ctx.chat_content = msg["content"]


def t_stream(ctx):
    body = chat_body(ctx.model, QUESTION, max_tokens=64, stream=True, stream_options={"include_usage": True})
    status, chunks, done = http_stream(ctx.port, "/v1/chat/completions", body)
    expect(status == 200, f"status {status}: {chunks}")
    expect(done, "the stream ended without 'data: [DONE]'")
    reasons = [c.get("finish_reason") for ch in chunks for c in ch.get("choices", []) if c.get("finish_reason")]
    expect(reasons and all(r in FINISH_REASONS for r in reasons), f"finish_reason values {reasons}")
    content = ""
    for ch in chunks:
        for c in ch.get("choices", []):
            delta = c.get("delta", {})
            content += delta.get("content") or ""
            expect(not (delta.get("reasoning_content") or "").strip(), "delta.reasoning_content non-empty with thinking off")
    expect(content.strip(), "no delta.content")
    expect(chunks and valid_usage(chunks[-1].get("usage")), f"the final chunk's usage is missing or invalid: {chunks[-1].get('usage') if chunks else None}")
    expect(content == ctx.chat_content, f"streamed text differs from the non-stream reply: {content!r} vs {ctx.chat_content!r}")


def t_thinking(ctx):
    body = chat_body(ctx.model, QUESTION, max_tokens=512, stream=True)
    body["chat_template_kwargs"] = {"enable_thinking": True}
    status, chunks, done = http_stream(ctx.port, "/v1/chat/completions", body)
    expect(status == 200, f"status {status}: {chunks}")
    expect(done, "the stream ended without 'data: [DONE]'")
    reasoning = "".join((c.get("delta", {}).get("reasoning_content") or "") for ch in chunks for c in ch.get("choices", []))
    expect(reasoning.strip(), "no delta.reasoning_content with thinking on")


def t_json_schema(ctx):
    rf = {"type": "json_schema", "json_schema": {"name": "widget", "strict": True, "schema": JSON_SCHEMA_TEST_SCHEMA}}
    msgs = [{"role": "user", "content": "Describe a small storage widget as an object of the given shape."}]
    status, _, body = http_json(ctx.port, "POST", "/v1/chat/completions", chat_body(ctx.model, msgs, max_tokens=200, response_format=rf))
    expect(status == 200, f"status {status}: {body}")
    content = body["choices"][0]["message"]["content"]
    try:
        obj = json.loads(content)
    except ValueError as e:
        raise Fail(f"content is not JSON ({e}): {content[:200]!r}")
    errs = validate(obj, JSON_SCHEMA_TEST_SCHEMA)
    expect(not errs, "; ".join(errs))


def t_tool_call(ctx):
    tool = {"type": "function", "function": {
        "name": "look_up_price",
        "description": "Returns the price of one catalogue item.",
        "parameters": {"type": "object", "properties": {"item_name": {"type": "string", "description": "the catalogue item to price"}},
                       "required": ["item_name"]}}}
    msgs = [{"role": "user", "content": "How much does the brass lantern cost? Use the available tool to find out."}]
    status, _, body = http_json(ctx.port, "POST", "/v1/chat/completions", chat_body(ctx.model, msgs, max_tokens=200, tools=[tool]))
    expect(status == 200, f"status {status}: {body}")
    calls = body["choices"][0]["message"].get("tool_calls") or []
    expect(calls, f"no tool_calls: {body['choices'][0]['message']}")
    # a client links the tool result it sends back to the call by id, and ends its turn on finish_reason tool_calls
    expect(isinstance(calls[0].get("id"), str) and calls[0]["id"], f"tool call without a string id: {calls[0]}")
    expect(calls[0].get("type") == "function", f"tool call type {calls[0].get('type')!r}, expected 'function'")
    expect(body["choices"][0]["finish_reason"] == "tool_calls", f"finish_reason {body['choices'][0]['finish_reason']!r}, expected 'tool_calls'")
    fn = calls[0]["function"]
    expect(fn["name"] == "look_up_price", f"tool name {fn['name']!r}")
    try:
        args = json.loads(fn["arguments"])
    except ValueError as e:
        raise Fail(f"arguments are not JSON ({e}): {fn['arguments']!r}")
    expect(isinstance(args, dict) and isinstance(args.get("item_name"), str) and args["item_name"], f"arguments {args!r}")


def t_embeddings(ctx):
    status, _, body = http_json(ctx.port, "POST", "/v1/embeddings", {"model": ctx.model, "input": "a short sentence to embed"})
    expect(status == 200, f"status {status}: {body}")
    emb = body["data"][0]["embedding"]
    expect(finite_numbers(emb), "embedding holds a non-number, a bool or a non-finite value")


def t_reuse(ctx):
    first = [{"role": "user", "content": filler(60) + " Reply with the single word ready."}]
    status, _, b1 = http_json(ctx.port, "POST", "/v1/chat/completions", chat_body(ctx.model, first, max_tokens=16))
    expect(status == 200, f"turn 1 status {status}: {b1}")
    reply = b1["choices"][0]["message"]["content"]
    start = ctx.main.log_size()
    second = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "Reply with the single word again."}]
    status, _, b2 = http_json(ctx.port, "POST", "/v1/chat/completions", chat_body(ctx.model, second, max_tokens=16))
    expect(status == 200, f"turn 2 status {status}: {b2}")
    problem = cache_usage_problem(b2.get("usage"), min_cached=1)
    expect(problem is None, f"turn 2 did not report a prefix-cache hit: {problem}")
    expect("[hot-cache] reused" in ctx.main.log_text(start), "no new '[hot-cache] reused' line in main.log after turn 2")


ERE_SPECIAL = set(".[]()*+?{}|^$\\")


def ere_escape(s):
    """`s` as a literal POSIX extended regular expression (pgrep's syntax, not Python's)."""
    return "".join("\\" + c if c in ERE_SPECIAL else c for c in s)


def self_safe_pattern(s):
    """A pgrep -f pattern matching `s` literally, with an alphanumeric first character in brackets so a shell wrapping
    the pgrep call (whose command line holds the pattern) does not match it."""
    if s and s[0].isalnum():
        return f"[{s[0]}]" + ere_escape(s[1:])
    return ere_escape(s)


def server_patterns(binary=None):
    """The pgrep -f patterns that find an engine server whatever its binary is called: the upstream name, the
    `serve --model` arguments every server of this test and of the reference jobs carries, and `<basename of binary> serve`."""
    patterns = ["[m]lx-serve serve", self_safe_pattern("serve --model")]
    if binary:
        name = os.path.basename(str(binary))
        if name:
            patterns.append(self_safe_pattern(name + " serve"))
    return patterns


def other_servers_running(binary=None):
    """Whether any engine server runs on this machine (ours are stopped when this is asked), as (state, detail), from
    one pgrep per server_patterns(binary): state True when any pattern matched (status 0), False only when every
    pattern matched nothing (status 1), None otherwise (another status, or pgrep could not be run); detail names the
    pgrep outcomes."""
    outcomes = []
    for pattern in server_patterns(binary):
        try:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            outcomes.append((None, f"pgrep {pattern!r} could not run ({type(e).__name__}: {e})"))
            continue
        if r.returncode == 0:
            outcomes.append((True, f"pgrep {pattern!r} matched " + ", ".join(r.stdout.split())))
        elif r.returncode == 1:
            outcomes.append((False, f"pgrep {pattern!r} matched nothing"))
        else:
            outcomes.append((None, f"pgrep {pattern!r} exited {r.returncode}: {(r.stderr or '').strip()}"))
    for want in (True, None):
        hits = [d for s, d in outcomes if s is want]
        if hits:
            return want, "; ".join(hits)
    return False, "pgrep matched nothing for " + ", ".join(repr(p) for p in server_patterns(binary))


def list_tree(root):
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            out.add(os.path.join(dirpath, n))
    return out


def clean_kv_cache(first_listing, root=None, binary=None):
    """Remove what the SSD test wrote under `root` (default ~/.mlx-serve/kv-cache). `first_listing` is the listing
    taken right after server 1 exited, or None when it never exited cleanly; `binary` is the tested binary, whose name
    other_servers_running also looks for. Returns the problems, each naming what was left in place and why; an empty
    list means the directory is gone or held nothing of ours that could not go."""
    root = Path(root) if root is not None else KV_CACHE
    if not root.exists():
        return []
    running, detail = other_servers_running(binary)
    if running is not False:
        files = sorted(p for p in list_tree(root) if os.path.isfile(p))
        why = ("another engine server is running" if running else "it could not be established that no other engine server is running")
        return [f"{why} ({detail}), so nothing under {root} can be told apart as ours; left in place: "
                + (", ".join(files) if files else "no files, the directory only")]
    leftovers = []
    now = list_tree(root)
    if first_listing is None:
        # Server 1 never exited cleanly (an earlier assertion or an error ended the test): the directory did not exist
        # at start, and pgrep just found no other server, so what is there now is ours.
        first_listing = now
    for p in sorted(first_listing, key=len, reverse=True):
        try:
            os.remove(p) if os.path.isfile(p) or os.path.islink(p) else os.rmdir(p)
        except FileNotFoundError:
            pass
        except OSError as e:
            leftovers.append(f"{p} could not be removed ({e})")
    leftovers.extend(f"{p} is not in the listing taken after server 1 exited; left in place"
                     for p in sorted(p for p in now - first_listing if os.path.isfile(p)))
    try:
        root.rmdir()
    except OSError:
        pass
    return leftovers


def t_ssd_tier(ctx):
    import re
    args = ctx.args
    # About 1,000 tokens: the filler runs at about 16 tokens a sentence.
    body = chat_body(ctx.model, [{"role": "user", "content": filler(62) + " Reply with the single word ready."}], max_tokens=64)
    first_listing = None  # taken right after server 1 exits, before any assertion can end the test
    created_root = not MLX_SERVE_DIR.exists()
    failure = None
    try:
        s1 = Server(args, "ssd-1.log", ["--prefix-cache-disk", "2GB"]).start()
        status, _, b1 = http_json(args.port, "POST", "/v1/chat/completions", body)
        expect(status == 200, f"server 1 status {status}: {b1}")
        prompt_tokens = b1["usage"]["prompt_tokens"]
        expect(is_count(prompt_tokens, 500), f"prompt_tokens {prompt_tokens!r} is not a count of at least 500 (the test needs about 1,000)")
        reply1 = b1["choices"][0]["message"]["content"]
        rc1 = s1.stop(grace=90)
        first_listing = list_tree(KV_CACHE) if KV_CACHE.exists() else set()
        expect(not s1.killed, "server 1 did not exit within 90 s of SIGTERM (SIGKILL needed)")
        expect(rc1 == 0, f"server 1 exited with status {rc1} after SIGTERM, expected 0")
        text = s1.log_text()
        # The engine writes a slice to disk in the background right after the request ("[disk-cache] e<id> complete on
        # disk", kv_disk_cache.zig:1689); the SIGTERM flush then reports what was still unwritten, which is 0 when the
        # background writer was done (scheduler.zig:5674). Pinned against the reference build: the flush line must be there, and at
        # least one slice must have reached disk by the end of shutdown, in either place.
        m = re.search(r"\[disk-cache\] shutdown persist: (\d+) written", text)
        expect(m, "no '[disk-cache] shutdown persist: <n> written' line in ssd-1.log (the SIGTERM flush did not run)")
        completed = len(re.findall(r"\[disk-cache\] e\d+ complete on disk", text))
        expect(int(m.group(1)) + completed >= 1, "no slice reached disk: shutdown persist wrote 0 and no 'complete on disk' line in ssd-1.log")
        expect(any(p.endswith(".safetensors") for p in first_listing), "no .safetensors file under ~/.mlx-serve/kv-cache after server 1 exited")
        s2 = Server(args, "ssd-2.log", ["--prefix-cache-disk", "2GB"]).start()
        status, _, b2 = http_json(args.port, "POST", "/v1/chat/completions", body)
        expect(status == 200, f"server 2 status {status}: {b2}")
        problem = restore_problem(b2.get("usage"))
        expect(problem is None, f"server 2 did not restore from the SSD tier: {problem}")
        expect(b2["choices"][0]["message"]["content"] == reply1, "the reply after restore differs from the first reply")
        rc2 = s2.stop(grace=90)
        expect(not s2.killed and rc2 == 0, f"server 2 exited with status {rc2} (killed: {s2.killed}) after SIGTERM, expected a clean 0")
    except BaseException as e:
        failure = e
        raise
    finally:
        stop_all()
        leftovers = clean_kv_cache(first_listing, binary=args.binary)
        if created_root:
            try:
                MLX_SERVE_DIR.rmdir()
            except OSError:
                pass
        if leftovers:
            text = "SSD cache cleanup incomplete: " + "; ".join(leftovers)
            if failure is None:
                raise Fail(text)
            print(f"NOTE ssd_tier: {text}", flush=True)


def t_image_rejected(ctx):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHnOcQAAAAABJRU5ErkJggg=="}}]}]
    status, _, body = http_json(ctx.port, "POST", "/v1/chat/completions", chat_body(ctx.model, msgs, max_tokens=16))
    rc = ctx.main.stop()
    expect(not ctx.main.killed and rc == 0, f"the image-test server exited with status {rc} (killed: {ctx.main.killed}) after SIGTERM, expected a clean 0")
    expect(status == 400, f"status {status}, expected 400: {body}")
    text = json.dumps(body).lower()
    expect("text" in text, f"error message does not mention text: {body}")


def t_decode_line(ctx):
    import re
    # The last test that uses the main server: stop it here (so its exit is judged) and read its log whole.
    rc = ctx.main.stop()
    expect(not ctx.main.killed and rc == 0, f"the main server exited with status {rc} (killed: {ctx.main.killed}) after SIGTERM, expected a clean 0")
    text = ctx.main.log_text(ctx.chat_log_start)
    expect(re.search(r"decode: [0-9.]+ tok/s", text), "no 'decode: <n> tok/s' line in main.log after the chat test")


def t_serve_flags(ctx):
    args = ctx.args
    s = Server(args, "flags.log", ["--ctx-size", "8192", "--skip-mem-preflight", "--os-reserve-gib", "4"]).start()
    rc = s.stop()
    expect(not s.killed and rc == 0, f"the flags server exited with status {rc} (killed: {s.killed}) after SIGTERM, expected a clean 0")
    try:
        r = subprocess.run([args.binary, "serve", "--help"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise Fail("'serve --help' did not finish within 60 s")
    out = r.stdout + r.stderr
    expect(r.returncode == 0, f"'serve --help' exited {r.returncode}")
    missing = [f for f in ("--os-reserve-gib", "--wired-margin-gib", "--prefix-cache-disk", "--shutdown-persist-secs") if f not in out]
    expect(not missing, f"serve --help does not list {missing}")


# ── runner ──────────────────────────────────────────────────────────────────────────────────────────────────────────

MAIN_TESTS = [("models", t_models), ("chat", t_chat), ("stream", t_stream), ("thinking", t_thinking),
              ("json_schema", t_json_schema), ("tool_call", t_tool_call), ("embeddings", t_embeddings), ("reuse", t_reuse)]


def record(rows, name, ok, seconds):
    if rows:
        with open(rows, "a") as f:
            f.write(json.dumps({"test": name, "ok": ok, "seconds": round(seconds, 3)}) + "\n")


def resolve_binary(path):
    """The absolute path of the binary to test. A name with a slash is taken relative to the working directory (as the
    executable check does); a bare name is looked up on PATH, so the file that is checked is the file that is started."""
    if os.sep in path:
        return os.path.abspath(path)
    return shutil.which(path) and os.path.abspath(shutil.which(path))


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--rows")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--text-only", action="store_true")
    a = p.parse_args(argv)
    a.env_map = {}
    for kv in a.env:
        if "=" not in kv:
            p.error(f"--env {kv!r} is not KEY=VALUE")
        k, v = kv.split("=", 1)
        a.env_map[k] = v
    return a


def run_one(name, fn, ctx, results, rows):
    t0 = time.time()
    DEAD.clear()
    try:
        fn(ctx)
        ok, why = True, ""
    except Fail as e:
        ok, why = False, str(e)
    except ValueError as e:  # includes json.JSONDecodeError: a malformed response body or stream chunk
        ok, why = False, f"malformed JSON in a response ({e})"
    except CouldNotTest:
        raise
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        ok, why = False, f"unexpected response shape ({type(e).__name__}: {e})"
    except (OSError, http.client.HTTPException) as e:
        # A server still in LIVE that has exited, or one that had exited on its own before this test's cleanup stopped
        # it (DEAD), is an engine failure, not a harness one.
        died = [x.proc.returncode for x in LIVE if x.proc is not None and x.proc.poll() is not None]
        died += [rc for _, rc in DEAD]
        if not died:
            raise CouldNotTest(f"{name}: transport error {type(e).__name__}: {e}")
        ok, why = False, f"the server exited with status {died[0]} during the test ({type(e).__name__}: {e})"
    dt = time.time() - t0
    print(f"PASS {name}" if ok else f"FAIL {name}: {why}", flush=True)
    record(rows, name, ok, dt)
    results.append(ok)


def main(argv):
    args = parse_args(argv)
    signal.signal(signal.SIGINT, lambda *_: (stop_all(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (stop_all(), sys.exit(143)))
    args.binary = resolve_binary(args.binary)
    if not args.binary or not os.access(args.binary, os.X_OK):
        print("could not test: the binary is not executable")
        return 2
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if KV_CACHE.exists():
        print(f"could not test: {KV_CACHE} exists (someone else's SSD cache; this test never deletes what it did not write)")
        return 2
    results = []
    ctx = Ctx()
    ctx.args = args
    ctx.port = args.port
    try:
        ctx.main = Server(args, "main.log").start()
        for name, fn in MAIN_TESTS:
            run_one(name, fn, ctx, results, args.rows)
        if "chat_log_start" in ctx.__dict__:
            run_one("decode_line", t_decode_line, ctx, results, args.rows)
        else:
            print("FAIL decode_line: the chat test did not run far enough to set the log offset")
            record(args.rows, "decode_line", False, 0)
            results.append(False)
            ctx.main.stop()
        run_one("ssd_tier", t_ssd_tier, ctx, results, args.rows)
        if args.text_only:
            ctx.main = Server(args, "image.log").start()
            run_one("image_rejected", t_image_rejected, ctx, results, args.rows)
        else:
            print("SKIP image_rejected (not counted; pass --text-only from Stage 4 on)")
        run_one("serve_flags", t_serve_flags, ctx, results, args.rows)
    except CouldNotTest as e:
        stop_all()
        print(f"could not test: {e}")
        return 2
    finally:
        stop_all()
    print(f"contract: {sum(results)} of {len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
