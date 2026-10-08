"""Server + pipeline integration over real HTTP / WebSockets (stand-in models)."""
import json
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import uvicorn
from fake_ollama import FakeOllama
from websockets.sync.client import connect

from livetranslator import simulate as sim
from livetranslator.asr import ScriptedASR
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.server import create_app
from livetranslator.storage import MeetingStore

pytestmark = pytest.mark.skipif(sim.tts_available() is None, reason="needs TTS")


def http(base, path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("content-type", "").startswith("application/json") else raw.decode()


@pytest.fixture()
def app_server(tmp_path):
    fake = FakeOllama(token_delay=0.002).start()
    settings = SettingsStore(tmp_path / "settings.json")
    settings.update({"ollama_url": fake.url, "input_source": "browser", "live_preview": False})
    texts = [de for de, _ in sim.SENTENCES] * 10

    def factory(st, store, publish):
        return Pipeline(st, store, publish, asr_factory=lambda s: ScriptedASR(texts))

    app = create_app(pipeline_factory=factory, store=MeetingStore(tmp_path / "Meetings"), settings=settings,
                     start_ollama=False)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if http(base, "/api/state")["status"]["asr"]["status"] == "ready":
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    yield base, f"ws://127.0.0.1:{port}", fake, tmp_path
    server.should_exit = True
    th.join(10)
    fake.stop()


def send_audio(ws_base, clips, scenario="desk"):
    audio, _ = sim.make_meeting(clips, sim.SCENARIOS[scenario])
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    with connect(ws_base + "/ws/audio") as a:
        a.send(json.dumps({"rate": 16000, "format": "i16"}))
        step = 16000 * 2 // 10  # 100 ms
        for i in range(0, len(pcm), step):
            a.send(pcm[i:i + step])
            time.sleep(0.004)  # ~25x real time


def collect(ev_ws, until, timeout=30):
    out = []
    end = time.time() + timeout
    while time.time() < end:
        try:
            ev = json.loads(ev_ws.recv(timeout=max(0.1, end - time.time())))
        except TimeoutError:
            break
        out.append(ev)
        if until(ev, out):
            return out
    raise AssertionError(f"condition not met; got {[e['type'] for e in out][-30:]}")


def finals(evs):
    return [e for e in evs if e["type"] == "tr" and e.get("final")]


def test_full_meeting_lifecycle(app_server):
    base, wsb, fake, tmp = app_server
    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:3])]
    with connect(wsb + "/ws") as ev:
        snap = json.loads(ev.recv(timeout=5))
        assert snap["type"] == "snapshot" and snap["status"]["state"] == "idle"
        meta = http(base, "/api/start", {"name": "Budget call"})
        assert meta["name"] == "Budget call"
        send_audio(wsb, clips)
        evs = collect(ev, lambda e, out: len(finals(out)) >= 3)
        lines = [e["line"] for e in evs if e["type"] == "line"]
        assert [ln["de"] for ln in lines] == [de for de, _ in sim.SENTENCES[:3]]
        assert [e["en"] for e in finals(evs)] == [en for _, en in sim.SENTENCES[:3]]
        assert any(e["type"] == "level" for e in evs)
        http(base, "/api/stop", {})
        collect(ev, lambda e, out: e["type"] == "session_end")

    ms = http(base, "/api/meetings")
    assert len(ms) == 1 and ms[0]["line_count"] == 3 and ms[0]["ended"]
    mid = ms[0]["id"]

    # continue the same meeting later
    time.sleep(1.1)
    with connect(wsb + "/ws") as ev:
        json.loads(ev.recv(timeout=5))
        http(base, "/api/start", {"continue_id": mid})
        send_audio(wsb, clips[:1])
        collect(ev, lambda e, out: len(finals(out)) >= 1)
        http(base, "/api/stop", {})
        collect(ev, lambda e, out: e["type"] == "session_end")
    m = http(base, f"/api/meetings/{mid}")
    assert len(m["lines"]) == 4
    offs = [ln["offset"] for ln in m["lines"]]
    assert offs == sorted(offs)   # (wall-clock times are only meaningful at real-time speed)
    assert [ln["id"] for ln in m["lines"]] == [1, 2, 3, 4]

    # rename, download, bad ids
    http(base, f"/api/meetings/{mid}/rename", {"name": "Budget call (part 2)"})
    md = http(base, f"/api/meetings/{mid}/download?fmt=md")
    assert md.startswith("# Budget call (part 2)") and md.count("DE: ") == 4
    txt = http(base, f"/api/meetings/{mid}/download?fmt=txt")
    assert "Budget call (part 2)" in txt
    for bad in ("..%2F..%2Fx", "nope", "2026-01-01_000000"):
        with pytest.raises(HTTPError) as e:
            http(base, f"/api/meetings/{bad}")
        assert e.value.code == 404

    # meeting notes
    with connect(wsb + "/ws") as ev:
        json.loads(ev.recv(timeout=5))
        http(base, f"/api/meetings/{mid}/summary", {})
        collect(ev, lambda e, out: e["type"] == "summary" and e.get("done"))
    assert "Summary" in http(base, f"/api/meetings/{mid}")["summary"]

    # delete
    http(base, f"/api/meetings/{mid}", method="DELETE")
    assert http(base, "/api/meetings") == []


def test_translation_failure_then_retry(app_server):
    base, wsb, fake, tmp = app_server
    clips = [sim.synthesize(sim.SENTENCES[0][0], 0)]
    fake.fail = True
    with connect(wsb + "/ws") as ev:
        json.loads(ev.recv(timeout=5))
        http(base, "/api/start", {})
        send_audio(wsb, clips)
        evs = collect(ev, lambda e, out: len(finals(out)) >= 1)
        assert finals(evs)[0]["ok"] is False
        assert any(e["type"] == "notice" and e["level"] == "error" for e in evs)
        mid = http(base, "/api/state")["status"]["meeting"]["id"]
        fake.fail = False
        http(base, f"/api/meetings/{mid}/lines/1/retranslate", {})
        evs = collect(ev, lambda e, out: len(finals(out)) >= 1)
        assert finals(evs)[0]["ok"] is True and finals(evs)[0]["en"] == sim.SENTENCES[0][1]
        http(base, "/api/stop", {})
        collect(ev, lambda e, out: e["type"] == "session_end")
    m = http(base, f"/api/meetings/{mid}")
    assert m["lines"][0]["en"] == sim.SENTENCES[0][1] and m["lines"][0]["ok"]


def test_settings_roundtrip_and_validation(app_server):
    base, *_ = app_server
    s = http(base, "/api/settings", {"pause_ms": 50, "sensitivity": "bogus", "glossary": "Müller, SAP", "unknown": 1})
    assert s["pause_ms"] == 200 and s["sensitivity"] == "normal" and s["glossary"] == "Müller, SAP"
    assert "unknown" not in s
    models = http(base, "/api/models")
    assert "turbo" in models["asr"] and models["ollama_running"]


def test_token_protects_everything(tmp_path):
    from fastapi.testclient import TestClient

    settings = SettingsStore(tmp_path / "s.json")
    app = create_app(pipeline_factory=lambda st, sto, pub: Pipeline(st, sto, pub, asr_factory=lambda s: ScriptedASR()),
                     store=MeetingStore(tmp_path / "M"), settings=settings, token="s3cret", start_ollama=False)
    with TestClient(app) as c:
        assert c.get("/api/meetings").status_code == 401
        assert c.get("/").status_code == 401
        assert c.get("/?token=s3cret").status_code == 200   # sets the cookie
        assert c.get("/api/meetings").status_code == 200
