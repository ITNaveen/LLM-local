"""A stand-in for the Ollama HTTP API (used by tests and UI checks).

Implements /api/version, /api/tags, /api/ps, /api/chat (streaming NDJSON with Ollama's
timing fields), /api/generate and /api/pull. Translations come from a dictionary,
otherwise "[EN] <german>".

Per-model behaviour can simulate a struggling machine:
  load_s       seconds to "load" the model on first use (or after an unload)
  first_s      extra delay before the first word, every request
  token_s      delay between words
  hang         never answer (the client must time out)
  gpu_share    fraction of the model reported on the GPU by /api/ps
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from livetranslator.simulate import SENTENCES

DICT = {de: en for de, en in SENTENCES}


class FakeOllama:
    def __init__(self, models=("gemma3:12b",), token_delay=0.01, fail=False, behaviour=None):
        self.models = list(models)
        self.token_delay = token_delay
        self.fail = fail
        self.behaviour = dict(behaviour or {})   # model -> {load_s, first_s, token_s, hang, gpu_share}
        self.loaded: dict[str, float] = {}       # model -> time it finished loading
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def translate(self, german: str) -> str:
        return DICT.get(german.strip(), f"[EN] {german.strip()}")

    def chats(self, model=None):
        return [r for r in self.requests if r["path"] == "/api/chat" and (model is None or r["body"].get("model") == model)]

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
                if self.path == "/api/ps":
                    out = []
                    for m in list(fake.loaded):
                        size = 8_000_000_000
                        share = fake.behaviour.get(m, {}).get("gpu_share", 1.0)
                        out.append({"name": m, "model": m, "size": size, "size_vram": int(size * share),
                                    "context_length": 4096})
                    return self._json(200, {"models": out})
                self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append({"path": self.path, "body": body, "t": time.monotonic()})
                if fake.fail:
                    return self._json(500, {"error": "simulated failure"})
                if self.path in ("/api/chat", "/api/generate"):
                    model = body.get("model", "")
                    if model not in fake.models:
                        return self._json(404, {"error": f"model '{model}' not found"})
                    b = fake.behaviour.get(model, {})
                    if self.path == "/api/generate" and body.get("keep_alive") == 0:
                        fake.loaded.pop(model, None)
                        return self._json(200, {"model": model, "response": "", "done": True})
                    t0 = time.monotonic()
                    load_s = 0.0
                    with fake._lock:
                        if model not in fake.loaded:
                            load_s = b.get("load_s", 0.0)
                            time.sleep(load_s)
                            fake.loaded[model] = time.monotonic()
                    if b.get("hang"):
                        time.sleep(3600)
                        return
                    if self.path == "/api/generate":
                        return self._json(200, {"model": model, "response": "", "done": True})
                    time.sleep(b.get("first_s", 0.0))
                    last = body["messages"][-1]["content"]
                    out = fake.translate(last) if "transcript" not in last.lower() else "**Summary**\n- fake notes"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    words = out.split(" ")
                    tok_delay = b.get("token_s", fake.token_delay)
                    t_gen = time.monotonic()
                    try:
                        for i, w in enumerate(words):
                            piece = w if i == 0 else " " + w
                            self.wfile.write((json.dumps({"message": {"role": "assistant", "content": piece},
                                                          "done": False}) + "\n").encode())
                            self.wfile.flush()
                            time.sleep(tok_delay)
                        gen_ns = int((time.monotonic() - t_gen) * 1e9)
                        self.wfile.write((json.dumps({
                            "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                            "total_duration": int((time.monotonic() - t0) * 1e9), "load_duration": int(load_s * 1e9),
                            "prompt_eval_count": 50, "prompt_eval_duration": 20_000_000,
                            "eval_count": len(words), "eval_duration": max(gen_ns, 1)}) + "\n").encode())
                    except (BrokenPipeError, ConnectionResetError):
                        pass
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
