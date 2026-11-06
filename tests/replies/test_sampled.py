"""Tests for sampled.py: run with `python3 tests/replies/test_sampled.py`; the last line is OK when all pass."""
import http.server
import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

SAMPLED = str(Path(__file__).resolve().parent / "sampled.py")


class Handler(http.server.BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        Handler.bodies.append(json.loads(self.rfile.read(n)))
        self._send({"choices": [{"message": {"content": "hello"}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 7}})


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SampledTests(unittest.TestCase):
    def setUp(self):
        Handler.bodies = []
        self.tmp = Path(tempfile.mkdtemp(prefix="lesto-sampled-"))
        self.conv = self.tmp / "conv.jsonl"
        rows = [{"id": "1k-00", "system": "SYS", "turns": ["first turn", "second"]},
                {"id": "1k-01", "system": "S2", "turns": ["other"]}]
        self.conv.write_text("\n" + "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.out = self.tmp / "out.jsonl"

    def run_client(self, port):
        return subprocess.run([sys.executable, SAMPLED, str(port), str(self.conv), str(self.out)],
                              capture_output=True, text=True)

    def serve(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def test_request_body(self):
        r = self.run_client(self.serve())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(Handler.bodies), 1)
        b = Handler.bodies[0]
        self.assertEqual(b["temperature"], 0.8)
        self.assertEqual(b["seed"], 1234)
        self.assertEqual(b["max_tokens"], 64)
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(b["messages"], [{"role": "system", "content": "SYS"},
                                         {"role": "user", "content": "first turn"}])

    def test_row(self):
        r = self.run_client(self.serve())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rows = [json.loads(x) for x in self.out.read_text().splitlines() if x.strip()]
        self.assertEqual(rows, [{"id": "1k-00", "text": "hello", "tokens": 7, "finish": "length"}])

    def test_transport_error(self):
        r = self.run_client(free_port())
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("could not test:", r.stdout + r.stderr)

    def test_non_object_conversation_is_could_not_test(self):
        self.conv.write_text("[1, 2]\n", encoding="utf-8")
        r = self.run_client(self.serve())
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("could not test:", r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_truncated_response_is_could_not_test(self):
        class Truncated(Handler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "500")
                self.end_headers()
                self.wfile.write(b"{")
                self.close_connection = True

        srv = http.server.HTTPServer(("127.0.0.1", 0), Truncated)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        r = self.run_client(srv.server_address[1])
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=0).result
    ok = result.wasSuccessful()
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
