#!/usr/bin/env python3
"""Long-running agent memory test (driven from test_long_agent_memory.sh).

Plants three orthogonal facts in turn 1, then drives an 11-turn
Claude-Code-style conversation that mixes tool calls, thinking turns, and
plain recall questions. Asserts that planted facts survive every mode
transition and that the model never "acts like the first time it's seen the
task" deep in the conversation.

The mocked tools (`shell`, `writeFile`) return short canned strings so the
test is offline and deterministic — what's under test is the server's
KV-cache + chat-template + thinking-channel handling across many turns,
not the live web.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
BASE = f"http://127.0.0.1:{PORT}"

# ANSI colors (kept identical to other tests for visual consistency).
RED = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"; DIM = "\033[2m"; NC = "\033[0m"

# Three orthogonal facts we plant in turn 1. Distinctive enough that the
# model can't "guess" them — a real recall failure shows up clearly. Codename
# uses an unusual hyphen pattern; deadline is an unambiguous date; language
# is a short common word that we anchor with the unique project context.
CODENAME = "FALCON-7X"
DEADLINE = "March 15, 2026"
LANGUAGE = "Rust"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "writeFile",
            "description": "Write text to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def mock_tool(name, args):
    """Return a short canned tool result. Length kept small so tool output
    doesn't dominate the conversation — we want the model's *recall* of the
    user's planted facts to be the dominant signal, not its summarization
    of long tool blobs."""
    if name == "shell":
        cmd = args.get("command", "")
        if "ls" in cmd or "list" in cmd or "find" in cmd:
            return "Cargo.toml\nsrc/\nREADME.md\ntests/"
        if "date" in cmd:
            return "Mon Mar  3 12:34:56 PST 2026"
        return "(no output)"
    if name == "writeFile":
        path = args.get("path", "?")
        content = args.get("content", "")
        return f"Wrote {len(content)} bytes to {path}."
    return f"(mocked) {name}({json.dumps(args)})"


def chat(messages, *, enable_thinking=False, with_tools=True, max_tokens=512, temperature=0.0):
    """Send a non-streaming chat completion and return parsed fields.

    Non-streaming on purpose: SSE parsing is exercised plenty in
    test_thinking_streaming, test_streaming_drafter, etc. This test cares
    about *content correctness across many turns*, not stream framing.
    """
    body = {
        "model": "mlx-serve",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if with_tools:
        body["tools"] = TOOLS
    if enable_thinking:
        body["enable_thinking"] = True

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"}
    except Exception as e:
        return {"error": str(e)}

    try:
        d = json.loads(raw)
    except Exception as e:
        return {"error": f"json parse: {e}; raw[:200]={raw[:200]!r}"}

    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": ch.get("finish_reason") or "?",
        "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens", 0),
        "completion_tokens": (d.get("usage") or {}).get("completion_tokens", 0),
    }


passes = 0
fails = 0
failed_assertions: list[str] = []


def check(cond, desc, detail=""):
    global passes, fails
    if cond:
        passes += 1
        print(f"  {GREEN}PASS{NC} {desc}")
    else:
        fails += 1
        failed_assertions.append(desc)
        print(f"  {RED}FAIL{NC} {desc}")
        if detail:
            print(f"    {DIM}{detail}{NC}")
    return cond


def run_tool_calls(resp, messages):
    """Execute every tool call returned in `resp` against `mock_tool` and
    append the assistant + tool messages back into `messages`. Returns the
    list of tool names that were called."""
    names: list[str] = []
    for tc in resp["tool_calls"]:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        names.append(name)
        result = mock_tool(name, args)
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tc.get("id", f"call_{len(names)}"),
                "type": "function",
                "function": {"name": name, "arguments": fn.get("arguments") or "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"call_{len(names)}"),
            "content": result,
        })
    return names


SYSTEM = (
    "You are a helpful coding assistant. Answer questions concisely. "
    "Use tools when they help — `shell` to run commands and `writeFile` to "
    "save files. When the user asks you to recall something they told you "
    "earlier in this conversation, answer from memory; do not call tools."
)


def turn_header(n: int, label: str):
    print(f"\n{CYAN}━━━ Turn {n}: {label} ━━━{NC}")


def main() -> int:
    t0 = time.time()
    print(f"=== Long agent memory test ===")
    print(f"Server: {BASE}")
    print(f"Planted facts: codename={CODENAME!r} deadline={DEADLINE!r} language={LANGUAGE!r}")

    # Sanity-check the server.
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            r.read()
    except Exception as e:
        print(f"{RED}FAIL{NC} server not reachable at {BASE}: {e}")
        return 1

    messages = [{"role": "system", "content": SYSTEM}]

    # ── Turn 1: plant the three facts. Phrase as a single user message so
    # the model has to retain all three across subsequent turns. ──
    turn_header(1, "plant facts")
    user_msg = (
        f"I'm building a project codenamed {CODENAME}. "
        f"It's written in {LANGUAGE}, and the deadline is {DEADLINE}. "
        "Acknowledge that you've got it."
    )
    messages.append({"role": "user", "content": user_msg})
    r = chat(messages, with_tools=True)
    if "error" in r:
        check(False, "turn 1 returned cleanly", r["error"]); return 1
    print(f"  {DIM}reply ({len(r['content'])} chars): {r['content'][:120]!r}{NC}")
    check(r["finish_reason"] in ("stop", "length"),
          "turn 1 finished cleanly", f"finish_reason={r['finish_reason']}")
    check(len(r["content"]) > 0, "turn 1 produced text content")
    messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 2: tool call (no thinking). Forces a cache-invalidation
    # transition — the agent loop now has tool_calls in history. ──
    turn_header(2, "tool call (shell)")
    messages.append({
        "role": "user",
        "content": "Use the shell tool to list the files in the current directory.",
    })
    r = chat(messages, with_tools=True)
    if "error" in r:
        check(False, "turn 2 returned cleanly", r["error"]); return 1
    called = run_tool_calls(r, messages)
    print(f"  {DIM}tools={called} content={r['content'][:80]!r}{NC}")
    check("shell" in called or len(r["content"]) > 0,
          "turn 2 emitted a tool call or text response",
          f"tools={called}, content_len={len(r['content'])}")
    if called:
        # Need a follow-up call so the model produces a final answer.
        r = chat(messages, with_tools=True)
        if "error" not in r:
            messages.append({"role": "assistant", "content": r["content"]})
    else:
        messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 3: recall check #1 (no tools, no thinking). The model has
    # seen its own assistant text + a tool call + a tool result since
    # turn 1. If the codename has been forgotten, this fails. ──
    turn_header(3, "recall codename (no tools)")
    messages.append({"role": "user", "content": "What is the codename of my project?"})
    r = chat(messages, with_tools=False)
    if "error" in r:
        check(False, "turn 3 returned cleanly", r["error"]); return 1
    print(f"  {DIM}reply: {r['content'][:200]!r}{NC}")
    # Codename is `FALCON-7X` — match case-insensitively, allow optional
    # quoting/markdown. The model often answers in different surface forms
    # (e.g. **FALCON-7X**, "Falcon-7X", FALCON 7X) so we anchor on the
    # distinctive token.
    check(re.search(r"falcon[-\s]?7x", r["content"], re.I) is not None,
          f"turn 3 recalled codename ({CODENAME})",
          f"reply did not contain codename: {r['content'][:200]!r}")
    messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 4: thinking + tool turn. Exercise both channels on the same
    # turn deep in the conversation. The reasoning_content from this turn
    # MUST NOT leak into turn 5's prompt as user input. ──
    turn_header(4, "thinking + tool (writeFile)")
    messages.append({
        "role": "user",
        "content": (
            "Think briefly about a good README structure for this project, "
            "then use writeFile to save a 3-line README at /tmp/test_long_agent_memory_README.md."
        ),
    })
    r = chat(messages, with_tools=True, enable_thinking=True, max_tokens=1024)
    if "error" in r:
        check(False, "turn 4 returned cleanly", r["error"]); return 1
    print(f"  {DIM}reasoning={len(r['reasoning'])} chars, "
          f"tool_calls={[(tc.get('function') or {}).get('name','?') for tc in r['tool_calls']]}, "
          f"finish={r['finish_reason']}{NC}")
    check(r["finish_reason"] in ("stop", "tool_calls", "length"),
          "turn 4 finished in a known state", f"got {r['finish_reason']}")
    # Critical for memory: thinking text must not be empty AND must not show
    # up as user-facing content (already covered by other tests, but a
    # sanity check here too — leaked thinking is a common source of "model
    # acts like it never saw the task").
    if r["reasoning"]:
        check("falcon" in r["reasoning"].lower() or "rust" in r["reasoning"].lower()
              or len(r["reasoning"]) > 30,
              "turn 4 reasoning is substantive (not empty residue)",
              f"reasoning[:200]={r['reasoning'][:200]!r}")
    called = run_tool_calls(r, messages)
    if not called:
        # Model answered directly without a tool call — that's still fine.
        messages.append({"role": "assistant", "content": r["content"]})
    else:
        # Need a follow-up to settle the tool round-trip into a final answer.
        r2 = chat(messages, with_tools=True)
        if "error" not in r2:
            messages.append({"role": "assistant", "content": r2["content"]})

    # ── Turn 5: recall check #2 with thinking enabled. Specifically
    # designed to surface "acts like first time it's seen the task" — if
    # turn 4's thinking_content polluted the prompt, the model often
    # responds as if no prior context exists. ──
    turn_header(5, "recall language (thinking on)")
    messages.append({
        "role": "user",
        "content": "Think for one sentence, then tell me: what programming language am I using?",
    })
    r = chat(messages, with_tools=False, enable_thinking=True, max_tokens=512)
    if "error" in r:
        check(False, "turn 5 returned cleanly", r["error"]); return 1
    combined = (r["content"] + " " + r["reasoning"]).lower()
    print(f"  {DIM}reasoning={len(r['reasoning'])} chars, content={r['content'][:150]!r}{NC}")
    check("rust" in combined,
          f"turn 5 recalled language ({LANGUAGE})",
          f"reply (content+reasoning) did not contain language: content={r['content'][:200]!r}")
    # If the model responded as if it has no context, it usually says
    # something like "I don't know" / "you haven't told me" / "what
    # language" — flag that explicitly.
    confused_phrases = [
        "i don't have", "you haven't", "i don't know what language",
        "what language are you", "could you tell me what language",
        "you didn't tell me",
    ]
    confused = any(p in combined for p in confused_phrases)
    check(not confused,
          "turn 5 did not act 'first-time-seen'",
          f"model appears to have lost context: {r['content'][:200]!r}")
    messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 6: another tool call to push the conversation past the
    # sliding-window threshold and force cache truncation/reuse. ──
    turn_header(6, "tool call (shell again)")
    messages.append({
        "role": "user",
        "content": "Run `date` via shell.",
    })
    r = chat(messages, with_tools=True)
    if "error" in r:
        check(False, "turn 6 returned cleanly", r["error"]); return 1
    called = run_tool_calls(r, messages)
    print(f"  {DIM}tools={called}{NC}")
    if not called:
        messages.append({"role": "assistant", "content": r["content"]})
    else:
        r2 = chat(messages, with_tools=True)
        if "error" not in r2:
            messages.append({"role": "assistant", "content": r2["content"]})

    # ── Turn 7: recall check #3 — deadline. Most distant from turn 1 so
    # this is the strongest test of long-context retention. ──
    turn_header(7, "recall deadline (no tools, no thinking)")
    messages.append({
        "role": "user",
        "content": "What was the deadline I mentioned?",
    })
    r = chat(messages, with_tools=False, max_tokens=128)
    if "error" in r:
        check(False, "turn 7 returned cleanly", r["error"]); return 1
    print(f"  {DIM}reply: {r['content'][:200]!r}{NC}")
    # Match "March 15" — the year is sometimes omitted in concise answers.
    check(re.search(r"march\s*1?5", r["content"], re.I) is not None,
          f"turn 7 recalled deadline ({DEADLINE})",
          f"reply did not contain deadline: {r['content'][:200]!r}")
    messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 8: rapid mode-transition stress. Toggle tools on/off in
    # consecutive turns to exercise the `cache reset — tools config changed`
    # path that the kv_cache_poison test single-shots. ──
    turn_header(8, "tools off → on transition")
    messages.append({"role": "user", "content": "Quickly: what's 2+2?"})
    r = chat(messages, with_tools=False, max_tokens=64)
    if "error" in r:
        check(False, "turn 8 returned cleanly", r["error"]); return 1
    print(f"  {DIM}reply: {r['content'][:80]!r}{NC}")
    check("4" in r["content"] or "four" in r["content"].lower(),
          "turn 8 produced a sensible math answer",
          f"reply: {r['content'][:200]!r}")
    messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 9: tools back on — must still recall facts from turn 1. ──
    turn_header(9, "tools back on, recall codename again")
    messages.append({
        "role": "user",
        "content": "I want to double-check the project name. What did I tell you it was?",
    })
    r = chat(messages, with_tools=True, max_tokens=128)
    if "error" in r:
        check(False, "turn 9 returned cleanly", r["error"]); return 1
    print(f"  {DIM}reply: {r['content'][:200]!r}{NC}")
    # Either text or a tool call would be valid output, but we want the text.
    text = r["content"] or " ".join((tc.get("function") or {}).get("arguments", "")
                                     for tc in r["tool_calls"])
    check(re.search(r"falcon[-\s]?7x", text, re.I) is not None,
          f"turn 9 recalled codename ({CODENAME}) after tools-on transition",
          f"reply did not contain codename: text={text[:200]!r}")
    messages.append({"role": "assistant", "content": r["content"]})

    # ── Turn 10: final summary turn (thinking enabled). All three planted
    # facts must surface. Strongest end-to-end memory check. ──
    turn_header(10, "summary — must surface all three facts")
    messages.append({
        "role": "user",
        "content": (
            "Please summarize what I've told you about the project so far in 2-3 sentences. "
            "Include the codename, the language, and the deadline."
        ),
    })
    r = chat(messages, with_tools=False, enable_thinking=True, max_tokens=768)
    if "error" in r:
        check(False, "turn 10 returned cleanly", r["error"]); return 1
    text = r["content"]
    print(f"  {DIM}reasoning={len(r['reasoning'])}c, content={text[:300]!r}{NC}")
    check(re.search(r"falcon[-\s]?7x", text, re.I) is not None,
          "summary mentions codename",
          f"text: {text[:300]!r}")
    check("rust" in text.lower(),
          "summary mentions language",
          f"text: {text[:300]!r}")
    check(re.search(r"march\s*1?5", text, re.I) is not None,
          "summary mentions deadline",
          f"text: {text[:300]!r}")
    # And a sanity check that the prompt actually grew across turns —
    # if usage.prompt_tokens didn't grow we're not really testing long
    # context, just one short prompt over and over.
    check(r["prompt_tokens"] > 400,
          f"final prompt grew past 400 tokens (got {r['prompt_tokens']})",
          "if this fails, the conversation isn't actually accumulating")

    elapsed = time.time() - t0
    print(f"\n{CYAN}═══════════════════════════════════════════════════════{NC}")
    print(f"  Turns:        10")
    print(f"  Assertions:   {passes + fails}  ({GREEN}{passes} pass{NC}, {RED}{fails} fail{NC})")
    print(f"  Final prompt: {r['prompt_tokens']} tokens")
    print(f"  Elapsed:      {elapsed:.0f}s")
    if fails:
        print(f"\n  {RED}FAILURES:{NC}")
        for d in failed_assertions:
            print(f"    - {d}")
    print(f"{CYAN}═══════════════════════════════════════════════════════{NC}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
