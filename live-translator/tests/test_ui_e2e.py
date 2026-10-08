"""Browser end-to-end test.

Headless Chromium plays a simulated meeting (laptop speaker + room + noise)
through its fake microphone. The page streams it to the server over the
browser-audio WebSocket; the real front end detects the lines; stand-in
models transcribe/translate; the page must show German + English, save the
meeting under the date/time folder, and support rename / view / download /
delete.
"""
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pytest

from livetranslator import simulate as sim
from livetranslator.audio_io import write_wav

pw = pytest.importorskip("playwright.sync_api")
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
ROOT = Path(__file__).resolve().parents[1]
SHOTS = Path(os.environ.get("LT_SHOTS", "/tmp/lt-shots"))

pytestmark = pytest.mark.skipif(sim.tts_available() is None, reason="needs TTS")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "demo_server.py"), "--port", str(port),
                             "--home", str(home)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            urlopen(f"http://127.0.0.1:{port}/api/info", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(proc.stdout.read())
    yield f"http://127.0.0.1:{port}", home
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def meeting_wav(tmp_path_factory):
    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:5])]
    audio, _ = sim.make_meeting(clips, sim.SCENARIOS["desk"], pauses_s=[1.2, 1.0, 1.4, 1.1, 1.3])
    p = tmp_path_factory.mktemp("wav") / "meeting.wav"
    # Chrome's fake device wants 48 kHz-ish audio; resample for realism
    from scipy.signal import resample_poly
    write_wav(str(p), resample_poly(audio, 3, 1).astype("float32"), 48000)
    return str(p)


def test_live_meeting_in_browser(server, meeting_wav):
    url, home = server
    SHOTS.mkdir(parents=True, exist_ok=True)
    with pw.sync_playwright() as p:
        kw = {"executable_path": CHROME} if Path(CHROME).exists() else {}
        browser = p.chromium.launch(args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={meeting_wav}", "--autoplay-policy=no-user-gesture-required"], **kw)
        ctx = browser.new_context(viewport={"width": 1280, "height": 820}, accept_downloads=True)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_selector("#startBtn:not([disabled])")
        page.wait_for_selector("#asrPill.ok")
        page.wait_for_selector("#llmPill.ok", timeout=15000)
        page.screenshot(path=str(SHOTS / "01-ready.png"))

        page.fill("#meetingName", "Weekly sync")
        page.click("#startBtn")
        page.wait_for_selector("#startBtn.stop")
        page.wait_for_selector("#browserAudio:not([hidden])")
        page.click("#baMic")
        page.wait_for_function("document.querySelectorAll('#liveFeed .line').length >= 3", timeout=90000)
        page.wait_for_function(
            "[...document.querySelectorAll('#liveFeed .line .en')].filter(e => !e.classList.contains('pending')).length >= 3",
            timeout=30000)
        # rename while live
        page.fill("#meetingName", "Weekly sync with Thomas")
        page.keyboard.press("Enter")
        time.sleep(1.2)
        page.screenshot(path=str(SHOTS / "02-live.png"))
        lines = page.eval_on_selector_all("#liveFeed .line", "els => els.map(e => [e.querySelector('.de').textContent, e.querySelector('.en').textContent])")
        de = [x[0] for x in lines]
        assert de[0] == sim.SENTENCES[0][0]
        assert lines[0][1] == sim.SENTENCES[0][1]
        assert page.eval_on_selector("#meter", "e => e.title").startswith("Level")

        page.click("#startBtn")  # stop
        page.wait_for_selector("#startBtn.primary", timeout=30000)
        page.wait_for_selector(".toast.ok")

        # saved on disk under Meetings/<date>/<HH-MM> - <name>/
        day = Path(home) / "Meetings" / datetime.now().strftime("%Y-%m-%d")
        folders = [f for f in day.iterdir() if f.is_dir()]
        assert len(folders) == 1 and folders[0].name.endswith(" - Weekly sync with Thomas"), folders
        md = (folders[0] / "transcript.md").read_text()
        assert "# Weekly sync with Thomas" in md and f"DE: {sim.SENTENCES[0][0]}" in md and f"EN: {sim.SENTENCES[0][1]}" in md
        meta = json.loads((folders[0] / "meta.json").read_text())
        assert meta["ended"] and meta["line_count"] >= 3

        # meetings drawer -> open -> download -> back
        page.click("#meetingsBtn")
        page.wait_for_selector(".mitem")
        assert page.inner_text(".day") == "TODAY" or page.inner_text(".day").lower() == "today"
        page.screenshot(path=str(SHOTS / "03-meetings.png"))
        page.click(".mitem")
        page.wait_for_selector("#viewBanner:not([hidden])")
        assert page.inner_text("#vbTitle") == "Weekly sync with Thomas"
        with page.expect_download() as dl:
            page.click("#vbDownloadMd")
        assert "Weekly sync with Thomas" in Path(dl.value.path()).read_text()
        page.screenshot(path=str(SHOTS / "04-view.png"))
        page.click("#vbBack")
        page.wait_for_selector("#viewBanner", state="hidden")

        # settings: change the pause, persisted on the server
        page.click("#settingsBtn")
        page.wait_for_selector("#settingsDrawer:not([hidden])")
        page.eval_on_selector("input[name=pause_ms]", "e => { e.value = 650; e.dispatchEvent(new Event('input', {bubbles: true})); }")
        page.wait_for_selector("#savedMark:not([hidden])", timeout=5000)
        page.screenshot(path=str(SHOTS / "05-settings.png"))
        assert json.loads(urlopen(url + "/api/settings").read())["pause_ms"] == 650
        page.keyboard.press("Escape")

        # delete
        page.click("#meetingsBtn")
        page.click(".mitem")
        page.once("dialog", lambda d: d.accept())
        page.click("#vbDelete")
        page.wait_for_selector("#viewBanner", state="hidden")
        assert not any(day.iterdir()) if day.exists() else True
        assert not errors, errors
        browser.close()
