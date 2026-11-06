"""Tests for check.py: run with `python3 tests/replies/test_check.py`; the last line is OK when all pass."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECK = str(Path(__file__).resolve().parent / "check.py")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def run(*args):
    return subprocess.run([sys.executable, CHECK, *map(str, args)], capture_output=True, text=True)


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lesto-check-"))
        self.hashes = self.tmp / "hashes.jsonl"
        self.rows = self.tmp / "rows.jsonl"
        write_jsonl(self.hashes, [{"id": "a", "sha256": sha("alpha")}, {"id": "b", "sha256": sha("beta")}])

    def test_identical(self):
        write_jsonl(self.rows, [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}])
        r = run("first-turns", self.rows, "--hashes", self.hashes)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "first-turns: identical 2 of 2")

    def test_one_differs(self):
        write_jsonl(self.rows, [{"id": "a", "text": "alpha"}, {"id": "b", "text": "betA"}])
        r = run("first-turns", self.rows, "--hashes", self.hashes)
        self.assertEqual(r.returncode, 1)
        self.assertIn("differs: b", r.stdout.splitlines())
        self.assertEqual(r.stdout.strip().splitlines()[-1], "first-turns: identical 1 of 2")

    def test_missing_row(self):
        write_jsonl(self.rows, [{"id": "a", "text": "alpha"}])
        r = run("multiturn", self.rows, "--hashes", self.hashes)
        self.assertEqual(r.returncode, 1)
        self.assertIn("missing: b", r.stdout.splitlines())

    def test_make_matches_hashlib(self):
        write_jsonl(self.rows, [{"id": "a", "text": "alpha"}, {"id": "b", "text": "héllo"}])
        r = run("--make", self.rows)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = [json.loads(line) for line in r.stdout.splitlines()]
        self.assertEqual(out, [{"id": "a", "sha256": sha("alpha")}, {"id": "b", "sha256": sha("héllo")}])
        self.assertTrue(all("text" not in o for o in out))

    def test_unreadable_is_exit_2(self):
        r = run("first-turns", self.tmp / "nope.jsonl", "--hashes", self.hashes)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("identical", r.stdout)

    def test_unicode_line_separators_inside_text_are_hashed(self):
        text = "a\u2028b\u2029c\u0085d"
        write_jsonl(self.hashes, [{"id": "a", "sha256": sha(text)}])
        self.rows.write_text(json.dumps({"id": "a", "text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
        r = run("first-turns", self.rows, "--hashes", self.hashes)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "first-turns: identical 1 of 1")

    def test_unpaired_surrogate_is_exit_2(self):
        self.rows.write_text('{"id": "a", "text": "\\ud800"}\n', encoding="utf-8")
        for args in (("--make", self.rows), ("first-turns", self.rows, "--hashes", self.hashes)):
            r = run(*args)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_e4b_set(self):
        sys.path.insert(0, str(Path(CHECK).parent))
        import check
        self.assertEqual(check.SETS["e4b-first10"], "e4b_first10.jsonl")
        write_jsonl(self.rows, [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}])
        r = run("e4b-first10", self.rows, "--hashes", self.hashes)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "e4b-first10: identical 2 of 2")

    def test_extra_row_is_not_identical(self):
        write_jsonl(self.rows, [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}, {"id": "c", "text": "x"}])
        r = run("first-turns", self.rows, "--hashes", self.hashes)
        self.assertEqual(r.returncode, 1)
        self.assertIn("extra: c", r.stdout.splitlines())


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=0).result
    ok = result.wasSuccessful()
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
