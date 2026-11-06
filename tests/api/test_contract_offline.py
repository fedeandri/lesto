"""Offline tests for contract.py's schema validator and SSE parser (no server). Last line OK when all pass."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract  # noqa: E402

SCHEMA = contract.JSON_SCHEMA_TEST_SCHEMA


def valid():
    return {"title": "a title", "count": 3, "kind": SCHEMA["properties"]["kind"]["enum"][0]}


class ValidateTests(unittest.TestCase):
    def test_schema_accepts_valid(self):
        self.assertEqual(contract.validate(valid(), SCHEMA), [])

    def test_schema_rejects_missing_required(self):
        obj = valid()
        del obj["count"]
        errs = contract.validate(obj, SCHEMA)
        self.assertTrue(errs and any("count" in e for e in errs), errs)

    def test_schema_rejects_extra_property(self):
        obj = valid()
        obj["other"] = 1
        errs = contract.validate(obj, SCHEMA)
        self.assertTrue(errs and any("other" in e for e in errs), errs)

    def test_schema_rejects_out_of_enum(self):
        obj = valid()
        obj["kind"] = "not-a-listed-kind"
        errs = contract.validate(obj, SCHEMA)
        self.assertTrue(errs and any("kind" in e for e in errs), errs)

    def test_schema_rejects_out_of_range_integer(self):
        for bad in (SCHEMA["properties"]["count"]["maximum"] + 1, SCHEMA["properties"]["count"]["minimum"] - 1, 2.5, True):
            obj = valid()
            obj["count"] = bad
            self.assertTrue(contract.validate(obj, SCHEMA), bad)

    def test_schema_rejects_wrong_type_and_non_object(self):
        obj = valid()
        obj["title"] = 5
        self.assertTrue(contract.validate(obj, SCHEMA))
        self.assertTrue(contract.validate([valid()], SCHEMA))


class SseTests(unittest.TestCase):
    STREAM = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "",
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "",
        "data: [DONE]",
        "",
        'data: {"choices":[{"delta":{"content":"after"}}]}',
    ]

    def test_sse_parser(self):
        chunks = contract.parse_sse(self.STREAM)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Hel")
        self.assertEqual(chunks[2]["choices"], [])
        self.assertIn("usage", chunks[2])

    def test_sse_parser_ignores_comments_and_blank_lines(self):
        chunks = contract.parse_sse([": keep-alive", "", 'data: {"a":1}', "", "data: [DONE]"])
        self.assertEqual(chunks, [{"a": 1}])

    def test_sse_parser_stream_without_done(self):
        self.assertEqual(contract.parse_sse(['data: {"a":1}']), [{"a": 1}])

    def test_sse_parser_bad_json_raises(self):
        with self.assertRaises(ValueError):
            contract.parse_sse(["data: {not json"])


class ShapeTests(unittest.TestCase):
    def test_valid_usage(self):
        self.assertTrue(contract.valid_usage({"prompt_tokens": 5, "completion_tokens": 2}))
        for bad in (None, {}, {"prompt_tokens": -1, "completion_tokens": -2}, {"prompt_tokens": 0, "completion_tokens": 1},
                    {"prompt_tokens": True, "completion_tokens": 2}, {"prompt_tokens": 5.0, "completion_tokens": 2}):
            self.assertFalse(contract.valid_usage(bad), bad)

    def test_finite_numbers_rejects_bool_nan_inf_empty(self):
        self.assertTrue(contract.finite_numbers([0.1, -2, 3]))
        for bad in ([], [True, False], [float("nan")], [float("inf")], ["1"], None, [1, None]):
            self.assertFalse(contract.finite_numbers(bad), bad)

    def test_finish_reasons_are_closed(self):
        self.assertIn("stop", contract.FINISH_REASONS)
        self.assertNotIn("invalid", contract.FINISH_REASONS)


class FakeProc:
    """A process that is running (returncode None) until it gets a signal, or has already exited with `returncode`."""

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeLog:
    def close(self):
        pass


def fake_server(returncode):
    s = contract.Server.__new__(contract.Server)
    s.proc = FakeProc(returncode)
    s.killed = False
    s.stopped = False
    s.log_file = FakeLog()
    return s


class RestoreTests(unittest.TestCase):
    def usage(self, prompt, cached=None, details=True):
        u = {"prompt_tokens": prompt, "completion_tokens": 2}
        if details:
            u["prompt_tokens_details"] = {} if cached is None else {"cached_tokens": cached}
        return u

    def test_restore_passes_at_75_percent(self):
        self.assertIsNone(contract.restore_problem(self.usage(1000, 750)))
        self.assertIsNone(contract.restore_problem(self.usage(1000, 1000)))

    def test_restore_fails(self):
        for bad in (self.usage(0, 0), self.usage(1000, 749), self.usage(1000, True), self.usage(1000, 0),
                    self.usage(1000, 750.0), self.usage(499, 499), self.usage(True, 1),
                    self.usage(1000, details=False), self.usage(1000, None), None, {}):
            self.assertIsNotNone(contract.restore_problem(bad), bad)

    def test_restore_rejects_cached_above_prompt(self):
        problem = contract.restore_problem(self.usage(1000, 1_000_000))
        self.assertIsNotNone(problem)
        self.assertIn("above prompt_tokens", problem)
        self.assertIsNotNone(contract.restore_problem(self.usage(1000, 1001)))


class CacheUsageTests(unittest.TestCase):
    def usage(self, prompt, cached=None, details=True):
        u = {"prompt_tokens": prompt, "completion_tokens": 2}
        if details:
            u["prompt_tokens_details"] = {} if cached is None else {"cached_tokens": cached}
        return u

    def test_reuse_passes(self):
        for good in (self.usage(100, 1), self.usage(100, 50), self.usage(100, 100)):
            self.assertIsNone(contract.cache_usage_problem(good, min_cached=1), good)

    def test_reuse_fails(self):
        for bad in (self.usage(100, True), self.usage(100, 0.5), self.usage(100, 0), self.usage(100, None),
                    self.usage(100, details=False), self.usage(100, 101), self.usage(100, 1.0), self.usage(100, -1),
                    self.usage(True, 1), self.usage(1.5, 1), self.usage(0, 0), None, {}, []):
            self.assertIsNotNone(contract.cache_usage_problem(bad, min_cached=1), bad)

    def test_details_null_is_missing(self):
        u = {"prompt_tokens": 100, "completion_tokens": 2, "prompt_tokens_details": None}
        self.assertIsNotNone(contract.cache_usage_problem(u, min_cached=1))


class PgrepPatternTests(unittest.TestCase):
    def run_by_pattern(self, codes, binary="/opt/x/lesto"):
        """codes maps a pattern's index in server_patterns(binary) to the status pgrep returns for it."""
        patterns = contract.server_patterns(binary)
        seen = []

        def fake_run(cmd, **k):
            self.assertEqual(cmd[:2], ["pgrep", "-f"])
            seen.append(cmd[2])
            rc = codes[patterns.index(cmd[2])]
            if isinstance(rc, BaseException):
                raise rc
            return subprocess.CompletedProcess(cmd, rc, stdout="77\n" if rc == 0 else "", stderr="err")

        with mock.patch.object(contract.subprocess, "run", fake_run):
            state, detail = contract.other_servers_running(binary)
        self.assertEqual(seen, patterns)
        return state, detail

    def test_three_patterns_for_a_binary(self):
        self.assertEqual(contract.server_patterns("/opt/x/lesto"), ["[m]lx-serve serve", "[s]erve --model", "[l]esto serve"])
        self.assertEqual(contract.server_patterns(None), ["[m]lx-serve serve", "[s]erve --model"])

    def test_any_match_is_true(self):
        for i in range(3):
            codes = [1, 1, 1]
            codes[i] = 0
            state, detail = self.run_by_pattern(codes)
            self.assertIs(state, True, codes)
            self.assertIn("77", detail)

    def test_match_wins_over_unknown(self):
        self.assertIs(self.run_by_pattern([2, 0, 1])[0], True)

    def test_all_nothing_is_false(self):
        self.assertIs(self.run_by_pattern([1, 1, 1])[0], False)

    def test_one_unknown_is_none(self):
        for i in range(3):
            codes = [1, 1, 1]
            codes[i] = 2
            self.assertIsNone(self.run_by_pattern(codes)[0], codes)
        self.assertIsNone(self.run_by_pattern([1, OSError("no pgrep"), 1])[0])
        self.assertIsNone(self.run_by_pattern([1, 1, subprocess.TimeoutExpired("pgrep", 30)])[0])

    def test_renamed_binary_matched_by_serve_model(self):
        """A server started as `lesto serve --model ...` is found without its name: the real pgrep would return 0 for
        pattern 2. Checked on the pattern itself with Python's re, which agrees with POSIX ERE on these patterns."""
        import re
        cmdline = "/opt/other/lesto serve --model /models/g4 --host 127.0.0.1 --port 8090"
        pattern = contract.server_patterns(None)[1]
        self.assertTrue(re.search(pattern, cmdline))
        self.assertFalse(re.search(contract.server_patterns(None)[0], cmdline))
        # The pattern does not match its own text (what a shell wrapping pgrep shows), nor the contract runner's line.
        self.assertFalse(re.search(pattern, "sh -c pgrep -f '[s]erve --model'"))
        self.assertFalse(re.search(pattern, "python3 contract.py --binary /opt/other/lesto --model /models/g4 --port 8090"))
        self.assertFalse(re.search(contract.server_patterns("/opt/other/lesto")[2],
                                   "python3 contract.py --binary /opt/other/lesto --model /models/g4 --port 8090"))
        self.assertTrue(re.search(contract.server_patterns("/opt/other/lesto")[2], cmdline))

    def test_binary_name_is_escaped(self):
        self.assertEqual(contract.server_patterns("/x/a.b+c")[2], "[a]\\.b\\+c serve")
        self.assertEqual(contract.server_patterns("/x/.hidden")[2], "\\.hidden serve")


class PgrepTests(unittest.TestCase):
    def run_with(self, returncode=None, raises=None):
        def fake_run(*a, **k):
            if raises:
                raise raises
            return subprocess.CompletedProcess(a[0], returncode, stdout="123\n" if returncode == 0 else "", stderr="err")

        with mock.patch.object(contract.subprocess, "run", fake_run):
            return contract.other_servers_running()

    def test_pgrep_states(self):
        self.assertIs(self.run_with(0)[0], True)
        self.assertIs(self.run_with(1)[0], False)
        self.assertIsNone(self.run_with(2)[0])
        self.assertIsNone(self.run_with(3)[0])
        self.assertIsNone(self.run_with(raises=FileNotFoundError("pgrep"))[0])

    def make_cache(self, root):
        (root / "a").mkdir(parents=True)
        (root / "a" / "slice.safetensors").write_bytes(b"x")
        return contract.list_tree(root)

    def test_cleanup_keeps_files_unless_pgrep_says_none(self):
        for state, detail in ((True, "pgrep matched 123"), (None, "pgrep exited 2: err")):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d) / "kv-cache"
                listing = self.make_cache(root)
                with mock.patch.object(contract, "other_servers_running", return_value=(state, detail)):
                    problems = contract.clean_kv_cache(listing, root)
                self.assertTrue((root / "a" / "slice.safetensors").exists(), state)
                self.assertEqual(len(problems), 1)
                self.assertIn(detail, problems[0])
                self.assertIn("slice.safetensors", problems[0])

    def test_cleanup_removes_ours_when_pgrep_says_none(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "kv-cache"
            listing = self.make_cache(root)
            with mock.patch.object(contract, "other_servers_running", return_value=(False, "pgrep matched nothing")):
                problems = contract.clean_kv_cache(listing, root)
            self.assertEqual(problems, [])
            self.assertFalse(root.exists())


class BinaryPathTests(unittest.TestCase):
    def test_relative_path_with_slash_becomes_absolute(self):
        out = contract.resolve_binary("./some/bin")
        self.assertTrue(Path(out).is_absolute() and out.endswith("some/bin"))

    def test_bare_name_is_looked_up_on_path(self):
        import os
        import tempfile
        d = tempfile.mkdtemp(prefix="lesto-bin-")
        exe = Path(d) / "enginebin"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        old = os.environ["PATH"]
        os.environ["PATH"] = d + os.pathsep + old
        try:
            self.assertEqual(Path(contract.resolve_binary("enginebin")), Path(d) / "enginebin")
        finally:
            os.environ["PATH"] = old

    def test_unknown_bare_name_is_none(self):
        self.assertFalse(contract.resolve_binary("no-such-binary-name-xyz"))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        contract.DEAD.clear()

    def tearDown(self):
        contract.DEAD.clear()
        contract.LIVE.clear()

    def test_malformed_json_is_a_failed_test_not_a_crash(self):
        results = []

        def bad(ctx):
            contract.parse_sse(["data: {not json"])

        contract.run_one("bad", bad, None, results, None)
        self.assertEqual(results, [False])

    def test_transport_error_without_a_dead_server_is_could_not_test(self):
        def broken(ctx):
            raise ConnectionResetError("reset")

        with self.assertRaises(contract.CouldNotTest):
            contract.run_one("net", broken, None, [], None)

    def test_transport_error_with_a_dead_server_is_a_failed_test(self):
        class P:
            returncode = -11

            def poll(self):
                return self.returncode

        dead = contract.Server.__new__(contract.Server)
        dead.proc = P()
        contract.LIVE.append(dead)
        try:
            results = []

            def broken(ctx):
                raise ConnectionResetError("reset")

            contract.run_one("net", broken, None, results, None)
            self.assertEqual(results, [False])
        finally:
            contract.LIVE.remove(dead)

    def test_transport_error_after_cleanup_stopped_a_crashed_server_is_a_failed_test(self):
        """t_ssd_tier's finally runs stop_all(), which takes the crashed server out of LIVE before run_one sees the
        error; the DEAD record must still make it a failed test."""
        dead = fake_server(-11)

        def crashing_test(ctx):
            contract.LIVE.append(dead)
            try:
                raise ConnectionResetError("reset")
            finally:
                contract.stop_all()

        results = []
        contract.run_one("net", crashing_test, None, results, None)
        self.assertEqual(results, [False])
        self.assertNotIn(dead, contract.LIVE)

    def test_transport_error_with_all_servers_alive_is_could_not_test_even_after_cleanup(self):
        alive = fake_server(None)

        def broken(ctx):
            contract.LIVE.append(alive)
            try:
                raise ConnectionResetError("reset")
            finally:
                contract.stop_all()

        with self.assertRaises(contract.CouldNotTest):
            contract.run_one("net", broken, None, [], None)

    def test_dead_record_is_reset_per_test(self):
        dead = fake_server(-11)

        def crashing_test(ctx):
            contract.LIVE.append(dead)
            try:
                raise ConnectionResetError("reset")
            finally:
                contract.stop_all()

        contract.run_one("first", crashing_test, None, [], None)

        def broken(ctx):
            raise ConnectionResetError("reset")

        with self.assertRaises(contract.CouldNotTest):
            contract.run_one("second", broken, None, [], None)

    def test_a_server_we_stopped_is_not_recorded_dead_on_a_second_stop(self):
        s = fake_server(None)
        s.stop()
        s.stop()
        self.assertEqual(contract.DEAD, [])

    def test_passing_and_failing_tests_are_recorded(self):
        results = []
        contract.run_one("ok", lambda ctx: None, None, results, None)

        def failing(ctx):
            contract.expect(False, "why")

        contract.run_one("no", failing, None, results, None)
        self.assertEqual(results, [True, False])


class ServerStartTests(unittest.TestCase):
    def test_server_start_oserror_is_could_not_test(self):
        import os
        import socket
        tmp = Path(tempfile.mkdtemp(prefix="lesto-start-"))
        fake = tmp / "notaprogram"
        fake.write_bytes(b"not a program\n")
        fake.chmod(0o755)
        home = tmp / "home"
        home.mkdir()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, HOME=str(home))
        r = subprocess.run([sys.executable, str(Path(contract.__file__).resolve()), "--binary", str(fake), "--model", "/nonexistent",
                            "--port", str(port), "--log-dir", str(tmp / "logs"), "--rows", str(tmp / "rows.jsonl")],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("could not test: server did not start (", r.stdout)


if __name__ == "__main__":
    ok = unittest.main(exit=False, verbosity=0).result.wasSuccessful()
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
