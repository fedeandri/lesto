#!/usr/bin/env python3
"""Sends one seeded, sampled chat request and writes the reply as one row. Runs against a running
OpenAI-style server. Stdlib only, Python 3.9.

    python3 sampled.py <port> <conversations.jsonl> <out.jsonl>

The request is the first turn of the first non-empty line of the conversations file (`system`, `turns[0]`), with
temperature 0.8, seed 1234 and at most 64 tokens, thinking off. The row is `{"id", "text", "tokens", "finish"}`.
Exit 0, or exit 2 with `could not test: <reason>` on a transport error or unreadable input.
"""
import http.client
import json
import sys
import urllib.request


def could_not_test(reason):
    print(f"could not test: {reason}")
    sys.exit(2)


def main(argv):
    if len(argv) != 3:
        could_not_test("usage: sampled.py <port> <conversations.jsonl> <out.jsonl>")
    port, conv_path, out_path = argv
    try:
        conv = None
        with open(conv_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    break
        if conv is None:
            could_not_test(f"{conv_path} has no conversation")
        mid = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60))["data"][0]["id"]
        body = {"model": mid, "temperature": 0.8, "seed": 1234, "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": conv["system"]},
                             {"role": "user", "content": conv["turns"][0]}]}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        row = {"id": conv["id"], "text": r["choices"][0]["message"]["content"],
               "tokens": r["usage"]["completion_tokens"], "finish": r["choices"][0]["finish_reason"]}
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
    except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        could_not_test(f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
