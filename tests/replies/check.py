#!/usr/bin/env python3
"""Compares greedy replies with the pinned hashes of the reference build.

The hash files next to this script pin the greedy replies of the pinned reference build: `first_turns.jsonl` holds the 40 first
turns (20 prompts of about 1k tokens and 20 of about 8k), `multiturn.jsonl` the 30 replies of 10 three-turn
conversations. They hold only `{"id", "sha256"}`; the prompts and the replies themselves are kept outside the
repository.

    python3 check.py <first-turns|multiturn|e4b-first10> <rows.jsonl> [--hashes <file>]
    python3 check.py --make <rows.jsonl>

`e4b_first10.jsonl` pins the greedy replies of a Gemma 4 E4B checkpoint to ten 1k first turns.

Rows are JSON lines with `id` and `text`. The `text` field is hashed as the sha256 of its UTF-8 bytes. Prints one
`differs: <id>`, `missing: <id>` or `extra: <id>` line per mismatch, then `<set>: identical <n> of <m>` (m is the number
of pinned ids). Exit 0 only when every pinned id is identical and no row is extra, 1 otherwise, 2 on unreadable input.
`--make` prints one `{"id": ..., "sha256": ...}` line per row and nothing else.
"""
import hashlib
import json
import sys
from pathlib import Path

SETS = {"first-turns": "first_turns.jsonl", "multiturn": "multiturn.jsonl", "e4b-first10": "e4b_first10.jsonl"}


def die(msg):
    print(f"check.py: {msg}", file=sys.stderr)
    sys.exit(2)


def read_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            # JSONL lines end at "\n" only: str.splitlines() would also split inside a string at U+2028, U+2029 or U+0085
            lines = [line for line in f.read().split("\n") if line.strip()]
    except (OSError, UnicodeDecodeError) as e:
        die(f"cannot read {path}: {e}")
    rows = []
    for n, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError as e:
            die(f"{path}:{n}: not JSON: {e}")
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            die(f"{path}:{n}: no string id")
        rows.append(row)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        die(f"{path}: duplicate ids")
    return rows


def digest(row, path):
    text = row.get("text")
    if not isinstance(text, str):
        die(f"{path}: row {row['id']} has no string text")
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError:
        die(f"{path}: row {row['id']} text is not valid Unicode (unpaired surrogate)")
    return hashlib.sha256(data).hexdigest()


def main(argv):
    if len(argv) == 2 and argv[0] == "--make":
        rows = read_jsonl(argv[1])
        for r in rows:
            print(json.dumps({"id": r["id"], "sha256": digest(r, argv[1])}))
        return 0
    hashes_path = None
    args = list(argv)
    if "--hashes" in args:
        i = args.index("--hashes")
        if i + 1 >= len(args):
            die("--hashes needs a file")
        hashes_path = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2 or args[0] not in SETS:
        die("usage: check.py <first-turns|multiturn|e4b-first10> <rows.jsonl> [--hashes <file>] | --make <rows.jsonl>")
    name, rows_path = args
    if hashes_path is None:
        hashes_path = str(Path(__file__).resolve().parent / SETS[name])
    pinned = {}
    for h in read_jsonl(hashes_path):
        sha = h.get("sha256")
        if not isinstance(sha, str):
            die(f"{hashes_path}: row {h['id']} has no sha256")
        pinned[h["id"]] = sha
    rows = {r["id"]: r for r in read_jsonl(rows_path)}
    identical = 0
    mismatches = 0
    for pid, sha in pinned.items():
        if pid not in rows:
            print(f"missing: {pid}")
            mismatches += 1
        elif digest(rows[pid], rows_path) != sha:
            print(f"differs: {pid}")
            mismatches += 1
        else:
            identical += 1
    for rid in rows:
        if rid not in pinned:
            print(f"extra: {rid}")
            mismatches += 1
    print(f"{name}: identical {identical} of {len(pinned)}")
    return 0 if mismatches == 0 and pinned else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
