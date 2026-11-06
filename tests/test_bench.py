#!/usr/bin/env python3
"""scripts/bench.py against a fake server; no model, no GPU. Run: python3 tests/test_bench.py"""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
FAKE = ROOT / "tests" / "bench_fake_server.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.ckpt = self.dir / "ckpt"
        self.ckpt.mkdir()
        (self.ckpt / "config.json").write_text("{}")
        self.record = self.dir / "requests.jsonl"

    def engine(self, flags=""):
        return {"command": f"{sys.executable} {FAKE} --port {{port}} --record {self.record} {flags}", "version": f"{sys.executable} -c \"print('fake 1.0')\""}

    def config(self, engines, cold, warm):
        path = self.dir / "config.json"
        path.write_text(json.dumps({"checkpoints": {"c": str(self.ckpt)}, "engines": engines,
                                    "modes": {"m": {"compare_to": "a", "noise_pair": ["a", "a2"], "cold": cold, "warm": warm}}}))
        return path

    def run_bench(self, config, *extra):
        out = self.dir / "run"
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench.py"), "run", "--mode", "m", "--config", str(config), "--out", str(out),
                            "--port", str(free_port()), "--rest", "0", *extra], capture_output=True, text=True, timeout=120)
        return p, out


class RunTests(Tmp):
    def test_run_writes_every_measure_and_never_repeats_a_prompt(self):
        cfg = self.config({"a": self.engine(), "a2": self.engine(), "b": self.engine()}, [["a", "c"], ["b", "c"], ["a2", "c"]], [["a", "c"]])
        p, out = self.run_bench(cfg, "--smoke")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        data = bench.load_run(out)
        self.assertEqual({r["cell"] for r in data["cold"]}, {"a/c", "b/c", "a2/c"})
        self.assertTrue(all(r["tokens"] == bench.MAX_TOKENS and r["ttft"] > 0 and r["decode_tps"] > 0 for r in data["cold"]))
        self.assertTrue(data["meta"]["smoke"])
        self.assertEqual(data["meta"]["method_version"], bench.METHOD_VERSION)
        self.assertIn("fake 1.0", data["meta"]["engines"]["a"]["version"])
        self.assertEqual(len(data["meta"]["checkpoints"]["c"]["fingerprint"]), 64)
        requests = [json.loads(line) for line in open(self.record)]
        measured = [r for r in requests if r["messages"][0]["content"] != "Say OK."]
        self.assertTrue(all(r["temperature"] == 0 for r in measured))
        firsts = [r["messages"][0]["content"].splitlines()[0] for r in measured if len(r["messages"]) == 1]
        self.assertEqual(len(firsts), len(set(firsts)))
        self.assertIn("SMOKE", (out / "report.md").read_text())

    def test_warm_later_turns_carry_the_conversation_and_report_the_cache(self):
        cfg = self.config({"a": self.engine()}, [], [["a", "c"]])
        cfg_data = json.loads(cfg.read_text())
        cfg_data["modes"]["m"]["cold"] = [["a", "c"]]
        cfg.write_text(json.dumps(cfg_data))
        p, out = self.run_bench(cfg, "--smoke")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        rows = bench.load_run(out)["warm"]
        self.assertEqual([r["turn"] for r in rows], [0, 1, 2])
        self.assertLess(rows[0]["cached_tokens"], rows[1]["cached_tokens"])
        last = [json.loads(line) for line in open(self.record)][-1]
        self.assertEqual([m["role"] for m in last["messages"]], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(last["messages"][-1]["content"], bench.FOLLOW_UPS[1])
        self.assertEqual(last["messages"][1]["content"], "".join(f"word{i} " for i in range(bench.WARM_MAX_TOKENS)))
        self.assertIn("Later turns", (out / "report.md").read_text())

    def test_an_engine_that_cannot_be_measured_is_dropped_and_exit_is_2(self):
        for flags, needle in (("--reasoning", "reasoning"), ("--short", "not the fixed"), ("--fail-load", "exited with 7"), ("--error-event", "error in the stream"),
                              ("--no-finish", "finish reason None"), ("--finish-error", "finish reason 'error'"), ("--buffered", "text events"), ("--buffered-first", "text events"),
                              ("--hidden-reasoning", "40 tokens of reasoning")):
            self.setUp()
            cfg = self.config({"a": self.engine(), "bad": self.engine(flags)}, [["bad", "c"], ["a", "c"]], [])
            p, out = self.run_bench(cfg, "--smoke")
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            data = bench.load_run(out)
            self.assertIn(needle, data["meta"]["dropped"]["bad/c"])
            self.assertTrue(any(r["cell"] == "a/c" for r in data["cold"]))
            self.assertFalse(any(r["cell"] == "bad/c" for r in data["cold"]))
            self.assertIn("not measured", (out / "report.md").read_text())

    def test_an_engine_that_batches_after_a_lone_first_token_is_measured(self):
        cfg = self.config({"a": self.engine("--batched")}, [["a", "c"]], [])
        p, out = self.run_bench(cfg, "--smoke")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        rows = bench.load_run(out)["cold"]
        self.assertTrue(rows and all(r["cell"] == "a/c" and r["events"] < r["tokens"] / 2 for r in rows))

    def test_refusals_before_any_server_starts(self):
        cfg = self.config({"a": self.engine()}, [["a", "c"]], [])
        (self.dir / "run").mkdir()
        (self.dir / "run" / "meta.json").write_text("earlier run")
        self.assertEqual(self.run_bench(cfg, "--smoke")[0].returncode, 2)  # never overwrites a run
        self.assertEqual((self.dir / "run" / "meta.json").read_text(), "earlier run")
        self.assertEqual(sorted(p.name for p in (self.dir / "run").iterdir()), ["meta.json"])
        (self.dir / "run" / "meta.json").unlink()
        (self.dir / "run").rmdir()
        data = json.loads(cfg.read_text())
        data["checkpoints"]["c"] = str(self.dir / "missing")
        cfg.write_text(json.dumps(data))
        p, out = self.run_bench(cfg, "--smoke")
        self.assertEqual(p.returncode, 2)
        self.assertFalse(out.exists())
        data["checkpoints"]["c"] = str(self.ckpt)
        data["engines"]["a"]["command"] = "{lesto} serve"
        cfg.write_text(json.dumps(data))
        p, out = self.run_bench(cfg, "--smoke")
        self.assertEqual(p.returncode, 2)
        self.assertIn("--lesto", p.stdout)
        self.assertFalse(self.record.exists())


def fake_run(path, decode, machine="M4 Pro", method=bench.METHOD_VERSION, fingerprint="f" * 64, smoke=False, warm_later=0.5, finished=True, dropped=None, gpu="20", command="x {port}"):
    """decode: {cell: [round medians]}."""
    path.mkdir()
    cells = list(decode)
    (path / "meta.json").write_text(json.dumps({
        "method_version": method, "mode": "m", "smoke": smoke, "rounds": 3, "requests": 1, "sizes": ["1k"], "conversations": 1, "started_utc": "t", **({"finished_utc": "t"} if finished else {}),
        "machine": {"chip": machine, "gpu_cores": gpu, "memory_bytes": "68719476736", "macos": "26.2"},
        "checkpoints": {"c": {"path": "p", "fingerprint": fingerprint}}, "engines": {c.split("/")[0]: {"version": "v1", "command": command} for c in cells},
        "cold_cells": cells, "warm_cells": cells[:1], "compare_to": "a", "noise_pair": ["a", "a2"], "dropped": dropped or {}, "dropped_warm": {}}))
    with open(path / "cold.jsonl", "w") as f:
        for cell, rounds in decode.items():
            for k, tps in enumerate(rounds):
                f.write(json.dumps({"cell": cell, "size": "1k", "round": k, "decode_tps": tps, "ttft": 1.0, "decode_s": 2.0}) + "\n")
    with open(path / "warm.jsonl", "w") as f:
        f.write(json.dumps({"cell": cells[0], "size": "1k", "conversation": 0, "turn": 0, "ttft": 1.0, "prompt_tokens": 100, "cached_tokens": 0}) + "\n")
        for turn in (1, 2):
            f.write(json.dumps({"cell": cells[0], "size": "1k", "conversation": 0, "turn": turn, "ttft": warm_later, "prompt_tokens": 100, "cached_tokens": None}) + "\n")
    return path


class VerdictTests(Tmp):
    def verdict(self, a, a2, other):
        data = bench.load_run(fake_run(self.dir / f"r{len(list(self.dir.iterdir()))}", {"a/c": a, "a2/c": a2, "b/c": other}))
        lines = bench.claims(data)
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def test_clear_gap_is_faster(self):
        self.assertTrue(self.verdict([85, 85, 85], [85, 85, 85], [70, 70, 70]).endswith(": faster"))

    def test_under_the_threshold_is_no_claim(self):
        self.assertTrue(self.verdict([85, 85, 85], [85, 85, 85], [83, 83, 83]).endswith("no claim"))

    def test_gap_inside_the_noise_is_no_claim(self):
        self.assertTrue(self.verdict([85, 85, 85], [75, 75, 75], [78, 78, 78]).endswith("no claim"))

    def test_losing_one_round_is_no_claim(self):
        self.assertTrue(self.verdict([85, 85, 60], [85, 85, 60], [70, 70, 70]).endswith("no claim"))

    def test_a_dropped_cell_is_not_scored_even_with_rows_on_disk(self):
        run = fake_run(self.dir / "d", {"a/c": [85, 85, 85], "a2/c": [85, 85, 85], "b/c": [70, 70, 70]}, dropped={"b/c": "failed in round 3"})
        data = bench.load_run(run)
        self.assertEqual(bench.claims(data), [])
        self.assertIn("| b/c | not measured |", bench.report_run(run))

    def test_an_unfinished_run_makes_no_claim(self):
        run = fake_run(self.dir / "u", {"a/c": [85, 85, 85], "a2/c": [85, 85, 85], "b/c": [70, 70, 70]}, finished=False)
        self.assertEqual(bench.claims(bench.load_run(run)), [])
        self.assertIn("UNFINISHED", bench.report_run(run))

    def test_an_incomplete_same_versus_same_run_gives_no_noise_floor(self):
        self.assertIn("no same-versus-same run", self.verdict([100, 100, 100], [100], [80, 80, 80]))

    def test_a_cell_short_of_its_requests_makes_no_claim(self):
        self.assertTrue(self.verdict([85, 85], [85, 85, 85], [70, 70, 70]).endswith("no claim"))

    def test_report_names_cached_as_not_reported_when_the_engine_does_not_say(self):
        out = bench.report_run(fake_run(self.dir / "r", {"a/c": [85, 85, 85], "a2/c": [85, 85, 85]}))
        self.assertIn("not reported", out)


class CleanupTests(Tmp):
    def test_a_server_child_left_by_its_launcher_is_stopped(self):
        pidfile = self.dir / "child.pid"
        port = free_port()
        launcher = self.dir / "launcher.py"
        launcher.write_text(f"""import subprocess, sys
p = subprocess.Popen([{sys.executable!r}, {str(FAKE)!r}, "--port", "{port}", "--record", {str(self.record)!r}])
open({str(pidfile)!r}, "w").write(str(p.pid))
""")
        server = bench.Server("x", f"{sys.executable} {launcher}", port, self.dir / "s.log")
        with self.assertRaises(bench.CouldNotMeasure):  # the launcher exits, which reads as a failed start
            with server:
                pass
        pid = int(pidfile.read_text())
        for _ in range(100):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                return
        self.fail("the server child is still running")

    def test_sigterm_stops_the_run_and_its_server(self):
        cfg = self.config({"a": self.engine("--delay 0.5")}, [["a", "c"]], [])
        port = free_port()
        p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "bench.py"), "run", "--mode", "m", "--config", str(cfg), "--out", str(self.dir / "run"),
                              "--port", str(port), "--rest", "0", "--smoke"], stdout=subprocess.DEVNULL)
        for _ in range(100):
            if self.record.exists():
                break
            time.sleep(0.1)
        p.terminate()
        self.assertEqual(p.wait(40), 128 + 15)
        with socket.socket() as s:
            self.assertNotEqual(s.connect_ex(("127.0.0.1", port)), 0)


class CompareTests(Tmp):
    def test_compares_the_same_setup_and_shows_the_change(self):
        ok, text = bench.compare_runs(fake_run(self.dir / "o", {"a/c": [80, 80, 80], "a2/c": [80, 80, 80]}, warm_later=1.0),
                                      fake_run(self.dir / "n", {"a/c": [88, 88, 88], "a2/c": [88, 88, 88]}, warm_later=0.5))
        self.assertTrue(ok)
        self.assertIn("+10.0%", text)
        self.assertIn("-50.0%", text)
        self.assertIn("Same-versus-same gap in the new run: 0.0%", text)

    def test_another_machine_or_method_or_a_smoke_is_not_comparable(self):
        base = fake_run(self.dir / "o", {"a/c": [80, 80, 80]})
        for i, kw in enumerate(({"machine": "M3 Pro"}, {"gpu": "16"}, {"method": bench.METHOD_VERSION + 1}, {"smoke": True}, {"finished": False})):
            ok, text = bench.compare_runs(base, fake_run(self.dir / f"n{i}", {"a/c": [80, 80, 80]}, **kw))
            self.assertFalse(ok, kw)
            self.assertIn("NOT COMPARABLE", text)

    def test_a_run_missing_measurements_is_not_compared_on_them(self):
        ok, text = bench.compare_runs(fake_run(self.dir / "o", {"a/c": [80, 80, 80]}), fake_run(self.dir / "n", {"a/c": [88, 88]}))
        self.assertIn("Skipped, missing measurements: a/c", text)
        self.assertNotIn("decode tok/s |", text)

    def test_a_changed_start_command_is_noted(self):
        ok, text = bench.compare_runs(fake_run(self.dir / "o", {"a/c": [80, 80, 80]}), fake_run(self.dir / "n", {"a/c": [80, 80, 80]}, command="x --cache {port}"))
        self.assertIn("the start command changed for a", text)

    def test_a_changed_checkpoint_skips_its_cells(self):
        ok, text = bench.compare_runs(fake_run(self.dir / "o", {"a/c": [80, 80, 80]}), fake_run(self.dir / "n", {"a/c": [88, 88, 88]}, fingerprint="e" * 64))
        self.assertTrue(ok)
        self.assertNotIn("decode tok/s |", text)
        self.assertIn("the checkpoint changed", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
