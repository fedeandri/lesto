#!/usr/bin/env python3
"""A fake OpenAI-style streaming server for tests/test_bench.py. Records each request, reuses a matching prefix like an engine.

    bench_fake_server.py --port N --record FILE [--reasoning] [--short] [--fail-load] [--error-event] [--no-finish] [--finish-error] [--buffered] [--buffered-first] [--batched] [--hidden-reasoning] [--delay S]"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--record", required=True)
ap.add_argument("--reasoning", action="store_true")
ap.add_argument("--short", action="store_true")
ap.add_argument("--fail-load", action="store_true")
ap.add_argument("--error-event", action="store_true")
ap.add_argument("--no-finish", action="store_true")
ap.add_argument("--finish-error", action="store_true")
ap.add_argument("--buffered", action="store_true")
ap.add_argument("--buffered-first", action="store_true")
ap.add_argument("--batched", action="store_true")
ap.add_argument("--hidden-reasoning", action="store_true")
ap.add_argument("--delay", type=float, default=0.001)
args = ap.parse_args()
if args.fail_load:
    sys.exit(7)
last_prompt = [""]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_event(self, event):
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = json.dumps(req["messages"])
        with open(args.record, "a") as f:
            f.write(json.dumps({"messages": req["messages"], "max_tokens": req["max_tokens"], "temperature": req["temperature"]}) + "\n")
        shared = len(os.path.commonprefix([prompt, last_prompt[0]]))
        last_prompt[0] = prompt
        self.send_response(200)
        self.end_headers()
        measured = req["max_tokens"] > 8
        tokens = 100 if args.short and measured else req["max_tokens"]
        if args.reasoning and measured:
            self.send_event({"choices": [{"delta": {"reasoning": "hmm"}}]})
        if args.buffered and measured:
            self.send_event({"choices": [{"delta": {"content": "many words "}}]})
            self.send_event({"choices": [{"delta": {"content": "at once "}}]})
        elif args.batched and measured:
            self.send_event({"choices": [{"delta": {"content": "word0 "}}]})
            for i in range(1, tokens, 8):
                time.sleep(args.delay * 8)
                self.send_event({"choices": [{"delta": {"content": "".join(f"word{j} " for j in range(i, min(i + 8, tokens)))}}]})
        else:
            if args.buffered_first and measured:
                self.send_event({"choices": [{"delta": {"content": "word " * (tokens // 2)}}]})
            for i in range(tokens):
                self.send_event({"choices": [{"delta": {"content": f"word{i} "}}]})
                time.sleep(args.delay)
        if args.error_event and measured:
            self.send_event({"error": {"message": "out of memory"}})
        if not (args.no_finish and measured):
            self.send_event({"choices": [{"delta": {}, "finish_reason": "error" if args.finish_error and measured else "length"}]})
        self.send_event({"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": tokens,
                                                  "prompt_tokens_details": {"cached_tokens": shared},
                                                  "completion_tokens_details": {"reasoning_tokens": 40 if args.hidden_reasoning and measured else 0}}})
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
