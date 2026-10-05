"""End-to-end: the whole editor on synthetic footage, plus the web API and Ollama client."""

import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from storymaker import pipeline
from storymaker.llm import OllamaLLM
from storymaker.source import FixtureSource
from storymaker.util import media_duration, probe


def _loudness(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                          "ebur128=framelog=quiet", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def test_full_pipeline_with_ai(settings, fake_llm):
    job = pipeline.Job.create({"topic": "Virat Kohli", "description": "pressure then a century",
                               "minutes": 2.5, "theme": "epic", "narration": "light", "demo": True})
    pipeline.run_job(job, settings, source=FixtureSource(settings), llm=fake_llm)
    assert job.state["status"] == "done", job.state.get("error") or job.state["log"][-5:]
    final = job.path("final.mp4")
    tl = json.loads(job.path("timeline.json").read_text())
    streams = {s["codec_type"]: s for s in probe(final)["streams"]}
    assert set(streams) == {"video", "audio"}
    assert (streams["video"]["width"], streams["video"]["height"]) == (320, 180)
    assert media_duration(final) == pytest.approx(tl["duration"], abs=0.1)
    assert _loudness(final) == pytest.approx(-14, abs=1.5)
    story = json.loads(job.path("story.json").read_text())
    assert story["source"] == "ai" and story["title_hi"] == "विराट का जवाब"
    kit = job.path("youtube.txt").read_text()
    assert "Virat Kohli Century" in kit and "Footage credits" in kit and "youtube.com/watch" in kit
    assert "0:00" in kit   # chapters
    srt = job.path("narration_hi.srt").read_text()
    assert "-->" in srt and "विराट" in srt
    assert job.path("thumbnail.jpg").exists()


def test_review_flow_edit_and_resume(settings, fake_llm):
    job = pipeline.Job.create({"topic": "Virat Kohli", "minutes": 2.5, "review": True,
                               "narration": "medium", "demo": True})
    src = FixtureSource(settings)
    pipeline.run_job(job, settings, source=src, llm=fake_llm)
    assert job.state["status"] == "awaiting_review"
    assert not job.path("final.mp4").exists()
    outline = json.loads(job.path("outline.json").read_text())
    narr = next(sc for sc in outline["scenes"] if sc["type"] == "narration")
    drop = [sc for sc in outline["scenes"] if sc["type"] == "dialogue"][-1]
    assert pipeline.apply_review_edits(job, {narr["id"]: "नई लाइन जो मैंने खुद लिखी है।"},
                                       remove=[drop["id"]])
    job.set(approved=True)
    pipeline.run_job(job, settings, source=src, llm=fake_llm)
    assert job.state["status"] == "done", job.state.get("error")
    srt = job.path("narration_hi.srt").read_text()
    assert "नई लाइन जो मैंने खुद लिखी" in srt and "है।" in srt
    story = json.loads(job.path("story.json").read_text())
    ids = {b.get("scene_id") for a in story["acts"] for b in a["beats"]}
    assert drop["id"] not in ids and narr["id"] in ids


def test_resume_after_failure_reuses_finished_stages(settings, fake_llm):
    class Flaky(FixtureSource):
        fail = True
        def download_section(self, *a, **k):
            if Flaky.fail:
                raise RuntimeError("network down")
            return super().download_section(*a, **k)
    src = Flaky(settings)
    job = pipeline.Job.create({"topic": "x", "minutes": 1.0, "demo": True})
    pipeline.run_job(job, settings, source=src, llm=fake_llm)
    assert job.state["status"] == "failed" and "No footage" in job.state["error"]
    research_mtime = job.path("research.json").stat().st_mtime
    Flaky.fail = False
    pipeline.run_job(job, settings, source=src, llm=fake_llm)
    assert job.state["status"] == "done", job.state.get("error")
    assert job.path("research.json").stat().st_mtime == research_mtime


# ---------------------------------------------------------------- Ollama client
class FakeOllama(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"models": [{"name": "qwen3.5:9b"}, {"name": "bge-m3:latest"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append(payload)
        if self.path == "/api/embed":
            self._send(200, {"embeddings": [[1.0, float(len(t))] for t in payload["input"]]})
        elif "think" in payload:
            self._send(400, {"error": '"qwen3.5:9b" does not support thinking'})
        else:
            self._send(200, {"message": {"content": '```json\n{"ok": true}\n```'}})


def test_ollama_client_retries_without_think_and_embeds():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        llm = OllamaLLM(url, "qwen3.5:9b", ["bge-m3", "nomic-embed-text"])
        assert llm.available()
        assert llm.embed_model() == "bge-m3"
        assert llm.chat_json("sys", "user") == {"ok": True}
        assert "think" not in FakeOllama.seen[-1]
        assert llm.embed(["a", "bbb"]).shape == (2, 2)
        assert not OllamaLLM(url, "llama3:70b").available()
    finally:
        server.shutdown()
    assert not OllamaLLM("http://127.0.0.1:9", "x").available()


# ---------------------------------------------------------------- web API
@pytest.fixture
def client():
    from storymaker.web import app
    app.testing = True
    return app.test_client()


def test_web_api(client):
    assert client.get("/").status_code == 200
    s = client.get("/api/status").get_json()
    assert s["ffmpeg"] is True and "music" in s
    assert client.post("/api/jobs", json={"topic": "  "}).status_code == 400
    r = client.post("/api/jobs", json={"topic": "Modi Operation Sindoor", "minutes": 40,
                                       "theme": "hacker", "narration": "lots"})
    jid = r.get_json()["id"]
    d = client.get(f"/api/jobs/{jid}").get_json()["state"]["request"]
    assert d["minutes"] == 15 and d["theme"] == "auto" and d["narration"] == "light"
    assert client.get(f"/jobs/{jid}/job.json").status_code == 404        # not downloadable
    assert client.get("/api/jobs/..%2Fsettings").status_code == 404
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    st = client.get("/api/styles").get_json()
    assert st["styles"][0]["name"] == "cinematic"
    assert client.post("/api/settings", json={"burn_subtitles": False, "unknown": 1}).get_json()["burn_subtitles"] is False
