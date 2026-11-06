#!/usr/bin/env python3
"""Lesto's benchmark: the same tests, the same way, every time. Standard library only. See docs/benchmark.md.

    scripts/bench.py run --mode release --lesto zig-out/bin/lesto --previous ../old/zig-out/bin/lesto
    scripts/bench.py run --mode field --lesto zig-out/bin/lesto
    scripts/bench.py report bench-results/<run>
    scripts/bench.py compare bench-results/<old> bench-results/<new>

The prompts, sizes, rounds and reply length are constants here, not options, so one run cannot quietly differ from the last;
changing them means raising METHOD_VERSION, and compare refuses runs of different versions. Engines and checkpoints come from
a JSON config (scripts/bench-config.json by default).

Exit 0: everything measured (compare: the runs are comparable). Exit 2: could not measure or compare (a server did not start, a request failed, a reply was short, reported no token count or
spent tokens on reasoning, or the runs differ in machine or method, or one did not finish); what could not be measured is never
scored, and an unfinished run makes no claim.
"""
import argparse
import hashlib
import json
import os
import platform
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

METHOD_VERSION = 1
PARAGRAPH = ("The history of mechanical clocks spans many centuries, from water clocks and sundials to escapements, "
             "pendulums and balance springs, each step improving accuracy and changing how people organised their days. ")
QUESTION = "Write a long, detailed essay about this history, with one section for each invention."
FOLLOW_UPS = ("Which of these inventions mattered most, and why? Answer in three sentences.",
              "Now name one thing the text leaves out. Answer in two sentences.")
SIZES = {"1k": 30, "8k": 207}  # paragraph repeats: about 1.2k and 8k prompt tokens
ROUNDS, REQUESTS, MAX_TOKENS = 3, 5, 200
WARM_CONVERSATIONS, WARM_MAX_TOKENS = 3, 64
# Decode speed is timed from the first text event to the last, so tokens held back before the first event inflate it.
# The stream gives no token count per event, so the first event's share of the characters stands in: at most 2%, about 4
# tokens of ordinary text in 200, which keeps the error well under the 5% a claim needs. It is a guard against buffering,
# not a proof: a first event of many one-character tokens could slip through, and the 200-token reply time is immune to
# it. Later batching only coarsens the timing (one real engine sends about 8 tokens per event, first token alone), so it
# is allowed down to one event per 20 tokens.
MIN_EVENTS_PER_TOKEN, MAX_FIRST_EVENT_SHARE = 0.05, 0.02
GOOD_FINISH = ("stop", "length")
CLAIM_PCT = 5.0  # a speed claim needs at least this gap, above the same-versus-same gap, in every round
START_TIMEOUT_S = 600


class CouldNotMeasure(Exception):
    pass


def post_stream(port, model, messages, max_tokens):
    """One greedy streamed request. Returns its tokens, time to first text, decode time and text."""
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = usage = finish = None
    text, reasoning = [], 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise CouldNotMeasure(f"error in the stream: {event['error']}")
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                reasoning += sum(len(delta.get(k) or "") for k in ("reasoning", "reasoning_content", "thinking"))
                if delta.get("content"):
                    text.append(delta["content"])
                    last = time.monotonic()
                    first = first or last
    if first is None or not usage:
        raise CouldNotMeasure(f"no text or no usage in the stream (usage={usage})")
    if finish not in GOOD_FINISH:
        raise CouldNotMeasure(f"the stream ended with finish reason {finish!r}")
    hidden = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    if reasoning or hidden:
        raise CouldNotMeasure(f"{reasoning} characters and {hidden} tokens of reasoning with thinking off")
    return {"prompt_tokens": usage["prompt_tokens"], "tokens": usage["completion_tokens"], "ttft": first - t0,
            "decode_s": last - first, "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "events": len(text), "first_event_chars": len(text[0]), "text": "".join(text)}


def model_id(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as resp:
        return json.load(resp)["data"][0]["id"]


class Server:
    """One engine resident: started on entry once it answers, its whole process group stopped on exit."""

    def __init__(self, name, command, port, log_path):
        self.name, self.command, self.port, self.log_path = name, command, port, log_path

    def __enter__(self):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", self.port)) == 0:
                raise CouldNotMeasure(f"port {self.port} is already in use")
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen(shlex.split(self.command), stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + START_TIMEOUT_S
        try:
            while True:
                if self.proc.poll() is not None:
                    raise CouldNotMeasure(f"{self.name} exited with {self.proc.returncode} before answering; see {self.log_path}")
                try:
                    self.model = model_id(self.port)
                    post_stream(self.port, self.model, [{"role": "user", "content": "Say OK."}], 8)  # not measured
                    return self
                except Exception as e:
                    if time.monotonic() > deadline:
                        raise CouldNotMeasure(f"{self.name} did not answer in {START_TIMEOUT_S} s ({e}); see {self.log_path}")
                    time.sleep(3)
        except BaseException:
            self.__exit__()
            raise

    def group_alive(self):
        try:
            os.killpg(self.proc.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def __exit__(self, *_):
        # the whole group, even when its leader has exited: a launcher can leave the real server behind
        if self.group_alive():
            os.killpg(self.proc.pid, signal.SIGTERM)
            deadline = time.monotonic() + 30
            while self.group_alive() and time.monotonic() < deadline:
                self.proc.poll()  # reaps the leader, or it would count as alive
                time.sleep(0.2)
            if self.group_alive():
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        self.log.close()


def cold_cell(server, cell, round_no, sizes, requests, run_id, counter, out):
    for size in sizes:
        for _ in range(requests):
            counter[0] += 1
            # the first line is unique to this run and request, so no prefix is reused, even one saved to disk by an earlier run
            content = f"Run {run_id}, request {counter[0]}.\n\n" + PARAGRAPH * SIZES[size] + "\n\n" + QUESTION
            try:
                r = post_stream(server.port, server.model, [{"role": "user", "content": content}], MAX_TOKENS)
            except CouldNotMeasure:
                raise
            except Exception as e:
                raise CouldNotMeasure(f"{cell} {size}: {e}")
            if r["tokens"] != MAX_TOKENS:
                raise CouldNotMeasure(f"{cell} {size}: reply of {r['tokens']} tokens, not the fixed {MAX_TOKENS}")
            if (r["events"] < MIN_EVENTS_PER_TOKEN * r["tokens"] or r["first_event_chars"] > MAX_FIRST_EVENT_SHARE * len(r["text"])
                    or r["decode_s"] <= 0):
                raise CouldNotMeasure(f"{cell} {size}: {r['events']} text events for {r['tokens']} tokens, the first holding "
                                      f"{r['first_event_chars']} of {len(r['text'])} characters, so no decode speed")
            r.update(cell=cell, size=size, round=round_no, decode_tps=(r["tokens"] - 1) / r["decode_s"])
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"{cell} {size}: prompt {r['prompt_tokens']}, reply {r['tokens']}, first text {r['ttft']:.2f} s, "
                  f"decode {r['decode_tps']:.1f} tok/s", flush=True)


def warm_cell(server, cell, sizes, conversations, run_id, counter, out):
    """Conversations of one document and two follow-ups: turn 0 is cold, the later turns can reuse what the engine processed."""
    for size in sizes:
        for conv in range(conversations):
            counter[0] += 1
            messages = [{"role": "user", "content": f"Run {run_id}, conversation {counter[0]}.\n\n" + PARAGRAPH * SIZES[size] +
                         "\n\nSummarise this in three sentences."}]
            for turn in range(len(FOLLOW_UPS) + 1):
                try:
                    r = post_stream(server.port, server.model, messages, WARM_MAX_TOKENS)
                except CouldNotMeasure:
                    raise
                except Exception as e:
                    raise CouldNotMeasure(f"{cell} {size} warm: {e}")
                row = {k: r[k] for k in ("prompt_tokens", "tokens", "ttft", "cached_tokens")}
                row.update(cell=cell, size=size, conversation=conv, turn=turn)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{cell} {size} warm: turn {turn}, prompt {r['prompt_tokens']}, cached {r['cached_tokens']}, "
                      f"first text {r['ttft']:.2f} s", flush=True)
                messages.append({"role": "assistant", "content": r["text"]})
                if turn < len(FOLLOW_UPS):
                    messages.append({"role": "user", "content": FOLLOW_UPS[turn]})


def fingerprint(path):
    """The name and content of every file, weights included, so any changed checkpoint is noticed."""
    h = hashlib.sha256()
    root = Path(path)
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(root)}\n".encode())
        with open(f, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def shell_output(command):
    try:
        p = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=60)
        return "\n".join(p.stdout.strip().splitlines()[:3]) or f"(exit {p.returncode}, no output)"
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return f"(could not run: {e})"


def gpu_cores():
    try:
        p = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, timeout=60)
        return str(json.loads(p.stdout)["SPDisplaysDataType"][0]["sppci_cores"])
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return "unknown"


def machine_state():
    sysctl = lambda k: shell_output(f"sysctl -n {k}")
    return {"chip": sysctl("machdep.cpu.brand_string"), "gpu_cores": gpu_cores(), "memory_bytes": sysctl("hw.memsize"), "macos": platform.mac_ver()[0]}


def load_state():
    sysctl = lambda k: shell_output(f"sysctl -n {k}")
    return {"load": sysctl("vm.loadavg"), "swap": sysctl("vm.swapusage")}


def fill(template, values):
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def stop_on_signal(signum, _frame):
    sys.exit(128 + signum)  # unwinds through Server.__exit__, so no server outlives the run


def run(args):
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    config = json.loads(Path(args.config).read_text())
    mode = config["modes"][args.mode]
    out_dir = Path(args.out or f"bench-results/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{args.mode}")
    if out_dir.exists():
        print(f"{out_dir} already exists: a run never overwrites another", flush=True)
        return 2
    values = {"lesto": os.path.abspath(args.lesto) if args.lesto else "", "previous": os.path.abspath(args.previous) if args.previous else ""}
    cells = {}
    for engine, checkpoint in mode["cold"] + mode["warm"]:
        spec = config["engines"].get(engine)
        path = os.path.abspath(os.path.expanduser(config["checkpoints"].get(checkpoint, "")))
        if not spec or not os.path.isdir(path):
            print(f"{engine}/{checkpoint}: engine not in the config, or no checkpoint at '{path}'", flush=True)
            return 2
        if ("{lesto}" in spec["command"] and not args.lesto) or ("{previous}" in spec["command"] and not args.previous):
            print(f"{engine} needs {'--lesto' if '{lesto}' in spec['command'] else '--previous'}", flush=True)
            return 2
        cells[f"{engine}/{checkpoint}"] = (spec, path, checkpoint)
    out_dir.mkdir(parents=True)
    # some engines read a folder of model folders: {model_dir} is one holding only this checkpoint
    rounds, requests, sizes = (1, 1, ["1k"]) if args.smoke else (ROUNDS, REQUESTS, list(SIZES))
    meta = {"method_version": METHOD_VERSION, "mode": args.mode, "smoke": args.smoke, "rounds": rounds, "requests": requests, "sizes": sizes, "conversations": 1 if args.smoke else WARM_CONVERSATIONS, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "machine": machine_state(), "state_start": load_state(),
            "checkpoints": {c: {"path": os.path.expanduser(p), "fingerprint": fingerprint(os.path.expanduser(p))} for c, p in config["checkpoints"].items()
                            if any(k.split("/")[1] == c for k in cells)},
            "engines": {}, "cold_cells": ["/".join(c) for c in mode["cold"]], "warm_cells": ["/".join(c) for c in mode["warm"]],
            "compare_to": mode.get("compare_to"), "noise_pair": mode.get("noise_pair"), "dropped": {}, "dropped_warm": {}}
    commands = {}
    for cell, (spec, path, checkpoint) in cells.items():
        model_dir = out_dir / "model-dirs" / checkpoint
        model_dir.mkdir(parents=True, exist_ok=True)
        link = model_dir / Path(path).name
        if not link.exists():
            link.symlink_to(path)
        commands[cell] = fill(spec["command"], dict(values, model=path, model_dir=model_dir, port=args.port))
        engine = cell.split("/")[0]
        meta["engines"].setdefault(engine, {"version": shell_output(fill(spec["version"], values)) if spec.get("version") else "(not stated)",
                                            "command": spec["command"]})
    write_meta = lambda: (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    write_meta()
    run_id, counter = int(time.time()), [0]
    conversations = meta["conversations"]
    order = meta["cold_cells"]
    rows = open(out_dir / "cold.jsonl", "w")
    warm_rows = open(out_dir / "warm.jsonl", "w")

    def attempt(cell, kind, body, dropped):
        if cell in dropped:
            return
        try:
            with Server(cell, commands[cell], args.port, out_dir / f"server-{cell.replace('/', '-')}.log") as server:
                print(f"{cell}: {kind} ready", flush=True)
                body(server)
        except CouldNotMeasure as e:
            dropped[cell] = str(e)
            write_meta()
            print(f"dropped: {cell}: {e}", flush=True)
        time.sleep(args.rest)

    for round_no in range(rounds):
        turn = round_no % len(order)  # the order rotates, so no engine always follows the same one
        for cell in order[turn:] + order[:turn]:
            attempt(cell, f"round {round_no + 1}", lambda s, c=cell, r=round_no: cold_cell(s, c, r, sizes, requests, run_id, counter, rows), meta["dropped"])
    for cell in meta["warm_cells"]:
        attempt(cell, "warm", lambda s, c=cell: warm_cell(s, c, sizes, conversations, run_id, counter, warm_rows), meta["dropped_warm"])
    rows.close()
    warm_rows.close()
    meta["state_end"] = load_state()
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_meta()
    report_text = report_run(out_dir)
    (out_dir / "report.md").write_text(report_text + "\n")
    print(report_text, flush=True)
    return 2 if meta["dropped"] or meta["dropped_warm"] else 0


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if os.path.exists(path) else []


def load_run(path):
    """A dropped cell's rows stay on disk but are never scored, even those written before it failed."""
    run_dir = Path(path)
    meta = json.loads((run_dir / "meta.json").read_text())
    return {"meta": meta, "cold": [r for r in read_jsonl(run_dir / "cold.jsonl") if r["cell"] not in meta["dropped"]],
            "warm": [r for r in read_jsonl(run_dir / "warm.jsonl") if r["cell"] not in meta["dropped_warm"]]}


def is_result(meta):
    return bool(meta.get("finished_utc")) and not meta.get("smoke")


def median(rows, key):
    return statistics.median(r[key] for r in rows)


def cold_summary(data, cell, size):
    rows = [r for r in data["cold"] if r["cell"] == cell and r["size"] == size]
    if not rows:
        return None
    return {"n": len(rows), "decode": median(rows, "decode_tps"), "ttft": median(rows, "ttft"),
            "reply": statistics.median(r["ttft"] + r["decode_s"] for r in rows),
            "rounds": {k: statistics.median(r["decode_tps"] for r in rows if r["round"] == k) for k in sorted({r["round"] for r in rows})}}


def cold_complete(data, summary):
    return summary["n"] == data["meta"]["rounds"] * data["meta"]["requests"]


def warm_summary(data, cell, size):
    rows = [r for r in data["warm"] if r["cell"] == cell and r["size"] == size]
    first, later = [r for r in rows if r["turn"] == 0], [r for r in rows if r["turn"] > 0]
    if not first or not later:
        return None
    cached = [r["cached_tokens"] / r["prompt_tokens"] for r in later if r["cached_tokens"] is not None]
    if len(rows) != data["meta"]["conversations"] * (len(FOLLOW_UPS) + 1):
        return None
    return {"n": len(later), "first": median(first, "ttft"), "later": median(later, "ttft"), "cached": statistics.median(cached) if cached else None}


def pct(new, old):
    return 100.0 * (new - old) / old


def noise_gap(data, size="1k"):
    """The same-versus-same gap in percent: the largest over the checkpoints that ran the noise pair, None when none did."""
    pair = data["meta"].get("noise_pair")
    gaps = []
    if pair:
        for checkpoint in {c.split("/")[1] for c in data["meta"]["cold_cells"]}:
            a, b = (cold_summary(data, f"{e}/{checkpoint}", size) for e in pair)
            if a and b and cold_complete(data, a) and cold_complete(data, b):
                gaps.append(abs(pct(a["decode"], b["decode"])))
    return max(gaps) if gaps else None


def claims(data, size="1k"):
    """The written rule: faster only at CLAIM_PCT or more, above the same-versus-same gap, in every round, on a shared checkpoint."""
    if not is_result(data["meta"]):
        return []
    base = data["meta"].get("compare_to")
    pair = data["meta"].get("noise_pair") or []
    floor = noise_gap(data, size)
    lines = []
    for cell in data["meta"]["cold_cells"]:
        engine, checkpoint = cell.split("/")
        mine = cold_summary(data, f"{base}/{checkpoint}", size)
        other = cold_summary(data, cell, size)
        if engine == base or engine in pair or not mine or not other:
            continue
        if floor is None:
            lines.append(f"- {cell}: no same-versus-same run, no claim")
            continue
        gap = pct(mine["decode"], other["decode"])
        if not cold_complete(data, mine) or not cold_complete(data, other):
            lines.append(f"- {cell}: {other['n']} and {mine['n']} requests, not the full count, no claim")
            continue
        every_round = set(mine["rounds"]) == set(other["rounds"]) and all(mine["rounds"][k] > other["rounds"][k] for k in mine["rounds"])
        ok = gap >= CLAIM_PCT and gap > floor and every_round
        lines.append(f"- {base} against {cell}: {gap:+.1f}%, same-versus-same {floor:.1f}%, every round: {'yes' if every_round else 'no'}: "
                     + ("faster" if ok else "no claim"))
    return lines


def report_run(path):
    data = load_run(path)
    meta = data["meta"]
    out = [f"# Bench report: {meta['mode']}, {meta['started_utc']}" + (" (SMOKE: not a result)" if meta.get("smoke") else "") + ("" if meta.get("finished_utc") else " (UNFINISHED: not a result)"), "",
           f"Method {meta['method_version']}; {meta['machine']['chip']}, {meta['machine']['gpu_cores']} GPU cores, {int(meta['machine']['memory_bytes']) >> 30} GB, macOS {meta['machine']['macos']}.", "",
           "| Cell | Prompt | Requests | Decode tok/s | First text s | 200-token reply s |", "|---|---|---:|---:|---:|---:|"]
    for cell in meta["cold_cells"]:
        summaries = [(size, cold_summary(data, cell, size)) for size in SIZES]
        if not any(s for _, s in summaries):
            out.append(f"| {cell} | not measured | | | | |")
        for size, s in summaries:
            if s:
                out.append(f"| {cell} | {size} | {s['n']}{'' if cold_complete(data, s) else ' (INCOMPLETE)'} | {s['decode']:.1f} | {s['ttft']:.2f} | {s['reply']:.2f} |")
    warm_lines = []
    for cell in meta["warm_cells"]:
        for size in SIZES:
            s = warm_summary(data, cell, size)
            if s:
                share = "not reported" if s["cached"] is None else f"{100 * s['cached']:.0f}%"
                warm_lines.append(f"| {cell} | {size} | {s['first']:.2f} | {s['later']:.2f} | {share} |")
    if warm_lines:
        out += ["", "| Cell | Prompt | First turn, first text s | Later turns, first text s | Later turns, cached |", "|---|---|---:|---:|---:|"] + warm_lines
    verdicts = claims(data)
    if verdicts:
        out += ["", f"Speed claims at 1k (rule: {CLAIM_PCT:g}% or more, above the same-versus-same gap, in every round):"] + verdicts
    if meta["dropped"] or meta["dropped_warm"]:
        out += ["", "Not measured:"] + [f"- {c}: {why}" for c, why in meta["dropped"].items()] + [f"- {c}, conversations: {why}" for c, why in meta["dropped_warm"].items()]
    out += ["", "Versions:"] + [f"- {e}: {v['version'].splitlines()[0]}" for e, v in meta["engines"].items()]
    return "\n".join(out)


def compare_runs(old_path, new_path):
    """Returns (comparable, text). Runs of different machines or methods are not compared; a changed checkpoint skips its cells."""
    old, new = load_run(old_path), load_run(new_path)
    problems = []
    if old["meta"]["method_version"] != new["meta"]["method_version"]:
        problems.append(f"method {old['meta']['method_version']} against {new['meta']['method_version']}")
    for k in ("chip", "gpu_cores", "memory_bytes"):
        if old["meta"]["machine"][k] != new["meta"]["machine"][k]:
            problems.append(f"{k}: {old['meta']['machine'][k]} against {new['meta']['machine'][k]}")
    if not is_result(old["meta"]) or not is_result(new["meta"]):
        problems.append("a smoke or unfinished run is not a result")
    if problems:
        return False, "NOT COMPARABLE: " + "; ".join(problems)
    out = []
    if old["meta"]["machine"]["macos"] != new["meta"]["machine"]["macos"]:
        out.append(f"Note: macOS changed, {old['meta']['machine']['macos']} to {new['meta']['machine']['macos']}.")
    floor = noise_gap(new)
    out.append("Same-versus-same gap in the new run: " + (f"{floor:.1f}%" if floor is not None else "not measured") + ".")
    out += ["", "| Cell | Prompt | Measure | Old | New | Change |", "|---|---|---|---:|---:|---:|"]
    old_prints, new_prints = old["meta"]["checkpoints"], new["meta"]["checkpoints"]
    skipped = set()
    for cell in new["meta"]["cold_cells"] + new["meta"]["warm_cells"]:
        checkpoint = cell.split("/")[1]
        if checkpoint in old_prints and checkpoint in new_prints and old_prints[checkpoint]["fingerprint"] != new_prints[checkpoint]["fingerprint"]:
            skipped.add(checkpoint)
    seen, incomplete = set(), set()
    for cell in new["meta"]["cold_cells"]:
        if cell.split("/")[1] in skipped or cell in seen:
            continue
        seen.add(cell)
        for size in SIZES:
            a, b = cold_summary(old, cell, size), cold_summary(new, cell, size)
            if a and b and not (cold_complete(old, a) and cold_complete(new, b)):
                incomplete.add(cell)
            elif a and b:
                for key, label, unit in (("decode", "decode tok/s", ".1f"), ("ttft", "first text s", ".2f"), ("reply", "200-token reply s", ".2f")):
                    out.append(f"| {cell} | {size} | {label} | {a[key]:{unit}} | {b[key]:{unit}} | {pct(b[key], a[key]):+.1f}% |")
    for cell in new["meta"]["warm_cells"]:
        if cell.split("/")[1] in skipped:
            continue
        for size in SIZES:
            a, b = warm_summary(old, cell, size), warm_summary(new, cell, size)
            if a and b:
                out.append(f"| {cell} | {size} | later-turn first text s | {a['later']:.2f} | {b['later']:.2f} | {pct(b['later'], a['later']):+.1f}% |")
    if skipped:
        out += ["", "Skipped, the checkpoint changed between the runs: " + ", ".join(sorted(skipped)) + "."]
    if incomplete:
        out += ["", "Skipped, missing measurements: " + ", ".join(sorted(incomplete)) + "."]
    changed = [e for e, v in new["meta"]["engines"].items() if e in old["meta"]["engines"] and v.get("command") != old["meta"]["engines"][e].get("command")]
    if changed:
        out += ["", "Note: the start command changed for " + ", ".join(sorted(changed)) + "; check that its settings did not."]
    out += ["", "Versions, old to new:"]
    for engine, v in new["meta"]["engines"].items():
        before = old["meta"]["engines"].get(engine, {}).get("version", "(not in the old run)")
        out.append(f"- {engine}: {before.splitlines()[0]} to {v['version'].splitlines()[0]}")
    return True, "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a mode's tests into a new folder")
    r.add_argument("--mode", required=True)
    r.add_argument("--config", default=str(Path(__file__).with_name("bench-config.json")))
    r.add_argument("--lesto", help="the Lesto binary under test")
    r.add_argument("--previous", help="the previous release's binary, for --mode release")
    r.add_argument("--out", help="a new folder; default bench-results/<time>-<mode>")
    r.add_argument("--port", type=int, default=8321)
    r.add_argument("--rest", type=int, default=15, help="seconds between servers, so memory is returned")
    r.add_argument("--smoke", action="store_true", help="one short pass to prove every engine answers; never a result")
    p = sub.add_parser("report", help="print a finished run's report")
    p.add_argument("run")
    c = sub.add_parser("compare", help="what changed between two runs")
    c.add_argument("old")
    c.add_argument("new")
    args = ap.parse_args()
    if args.command == "run":
        return run(args)
    if args.command == "report":
        print(report_run(args.run))
        return 0
    comparable, text = compare_runs(args.old, args.new)
    print(text)
    return 0 if comparable else 2


if __name__ == "__main__":
    sys.exit(main())
