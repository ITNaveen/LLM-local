"""A tiny stand-in for the Ollama HTTP API (used by tests and UI checks).

Implements /api/version, /api/tags, /api/chat (streaming NDJSON), /api/generate
and /api/pull. Translations come from a dictionary, otherwise "[EN] <german>".
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from livetranslator.simulate import SENTENCES

DICT = {de: en for de, en in SENTENCES}


class FakeOllama:
    def __init__(self, models=("gemma3:12b",), token_delay=0.01, fail=False):
        self.models = list(models)
        self.token_delay = token_delay
        self.fail = fail
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def translate(self, german: str) -> str:
        return DICT.get(german.strip(), f"[EN] {german.strip()}")

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/version":
                    return self._json(200, {"version": "0.0-fake"})
                if self.path == "/api/tags":
                    return self._json(200, {"models": [{"name": m} for m in fake.models]})
                self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append({"path": self.path, "body": body})
                if fake.fail:
                    return self._json(500, {"error": "simulated failure"})
                if self.path in ("/api/chat", "/api/generate"):
                    model = body.get("model", "")
                    if model not in fake.models:
                        return self._json(404, {"error": f"model '{model}' not found"})
                    if self.path == "/api/generate":
                        return self._json(200, {"model": model, "response": "", "done": True})
                    last = body["messages"][-1]["content"]
                    out = fake.translate(last) if "transcript" not in last.lower() else "**Summary**\n- fake notes"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    words = out.split(" ")
                    for i, w in enumerate(words):
                        piece = w if i == 0 else " " + w
                        self.wfile.write((json.dumps({"message": {"role": "assistant", "content": piece},
                                                      "done": False}) + "\n").encode())
                        self.wfile.flush()
                        time.sleep(fake.token_delay)
                    self.wfile.write((json.dumps({"message": {"role": "assistant", "content": ""},
                                                  "done": True}) + "\n").encode())
                    return
                if self.path == "/api/pull":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for i in range(0, 101, 25):
                        self.wfile.write((json.dumps({"status": "pulling", "completed": i, "total": 100}) + "\n").encode())
                        self.wfile.flush()
                        time.sleep(0.01)
                    self.wfile.write(b'{"status":"success"}\n')
                    fake.models.append(body.get("model"))
                    return
                self._json(404, {"error": "not found"})

        return H
