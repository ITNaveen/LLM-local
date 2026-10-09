"""Audio sources. Each delivers (float32 mono samples, sample_rate) chunks to a callback.

* MicSource    - a microphone on the machine running the server (default)
* PushSource   - audio pushed over the network by a browser (for running the
                 server on another machine / under a host name later)
* FileSource   - a WAV file played in (real-)time; used by tests and the self-test
"""
from __future__ import annotations

import logging
import threading
import time
import wave
from typing import Callable

import numpy as np

log = logging.getLogger("lt.audio")

AudioCallback = Callable[[np.ndarray, int], None]


# Names of virtual / app-provided inputs that are silent unless that app feeds them.
_VIRTUAL_HINTS = ("teams", "zoom", "blackhole", "loopback", "soundflower", "krisp", "aggregate", "multi-output",
                  "background music", "vb-cable", "cable output", "voicemeeter", "obs", "camo", "rogue amoeba",
                  "audio hijack", "nvidia broadcast", "ishowu", "screenflow", "descript", "riverside", "virtual",
                  "webex", "discord", "boom", "elgato", "mmhmm")
_BUILTIN_HINTS = ("macbook", "built-in", "internal", "imac", "mac mini", "mac studio", "studio display")
# a PortAudio call must never run while another one re-initialises the library
_PA_LOCK = threading.RLock()


def classify_device(name: str) -> str:
    n = name.lower()
    if any(h in n for h in _BUILTIN_HINTS):
        return "builtin"
    if any(h in n for h in _VIRTUAL_HINTS):
        return "virtual"
    if "iphone" in n or "ipad" in n:
        return "phone"   # Continuity microphone: only works while the phone is actively connected
    return "other"


def _default_input_index(sd) -> int:
    """The macOS/Windows system default input (System Settings → Sound → Input)."""
    try:
        return int(sd.query_devices(kind="input")["index"])
    except Exception:  # noqa: BLE001
        pass
    try:
        idx = sd.default.device[0]   # an _InputOutputPair - index it, it is not a list/tuple
        return int(idx) if idx is not None else -1
    except Exception:  # noqa: BLE001
        return -1


def list_input_devices(refresh: bool = False) -> list[dict]:
    """Input devices; refresh=True re-scans (new USB / Bluetooth devices, changed system default)."""
    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001  (PortAudio missing)
        log.warning("sounddevice unavailable: %s", e)
        return []
    with _PA_LOCK:
        try:
            if refresh:
                sd._terminate()
                sd._initialize()
            default_in = _default_input_index(sd)
            out = []
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"],
                                "rate": int(d.get("default_samplerate") or 48000), "default": i == default_in,
                                "kind": classify_device(d["name"])})
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("could not list audio devices: %s", e)
            return []


def probe_device(dev: dict, seconds: float = 0.9) -> dict:
    """Record briefly from one input. Real microphones always show some noise;
    exact digital silence means: no permission, or a virtual device nobody feeds."""
    import sounddevice as sd

    rate = dev.get("rate") or 48000
    channels = min(2, dev.get("channels") or 1)
    chunks: list[np.ndarray] = []

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    res = {"name": dev["name"], "index": dev["index"], "kind": dev.get("kind", "other"),
           "default": dev.get("default", False), "ok": False, "peak_db": None, "rms_db": None, "error": ""}
    try:
        with _PA_LOCK:  # held for the whole probe: no re-scan may close this stream under us
            stream = sd.InputStream(device=dev["index"], channels=channels, samplerate=rate, dtype="float32",
                                    blocksize=int(rate * 0.02), callback=cb)
            stream.start()
            time.sleep(seconds)
            stream.stop()
            stream.close()
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)
        return res
    if not chunks:
        res["error"] = "no audio callbacks"
        return res
    x = np.concatenate(chunks).astype(np.float64)
    x = x[int(rate * 0.2):] if x.shape[0] > rate * 0.3 else x   # devices may start with a few silent buffers
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms_v = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    res["peak_db"] = round(20 * np.log10(peak), 1) if peak > 0 else None
    res["rms_db"] = round(20 * np.log10(rms_v), 1) if rms_v > 0 else None
    res["ok"] = peak > 0.0
    return res


def mic_permission_status() -> str:
    """macOS microphone permission of the app that launched us (Terminal):
    authorized | denied | restricted | not_determined | unknown (not macOS / pyobjc missing)."""
    import platform

    if platform.system() != "Darwin":
        return "unknown"
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio  # pyobjc-framework-AVFoundation
    except Exception:  # noqa: BLE001
        return "unknown"
    try:
        st = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio))
    except Exception:  # noqa: BLE001
        return "unknown"
    return {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(st, "unknown")


def request_mic_permission(timeout: float = 60.0) -> str:
    """Show the macOS 'allow microphone' prompt (only possible while not yet decided)."""
    status = mic_permission_status()
    if status != "not_determined":
        return status
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        done = threading.Event()
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, lambda granted: done.set())
        done.wait(timeout)
    except Exception:  # noqa: BLE001
        pass
    return mic_permission_status()


def preferred_order(devices: list[dict], exclude: str = "") -> list[dict]:
    """Which inputs to try when the chosen one is silent: built-in mic first, virtual devices last."""
    rank = {"builtin": 0, "other": 1, "phone": 2, "virtual": 3}
    return sorted((d for d in devices if d["name"] != exclude),
                  key=lambda d: (rank.get(d.get("kind", "other"), 1), not d.get("default", False)))


class MicSource:
    def __init__(self, device_name: str, on_audio: AudioCallback):
        self.device_name = device_name
        self.on_audio = on_audio
        self.stream = None
        self.rate = 0
        self.status_flags = 0
        self.device: dict = {}

    def _resolve(self):
        devices = list_input_devices(refresh=True)
        if not devices:
            raise RuntimeError("No microphone found. Check that a microphone is connected and allowed.")
        if self.device_name:
            for d in devices:
                if d["name"] == self.device_name:
                    return d
            for d in devices:
                if self.device_name.lower() in d["name"].lower():
                    return d
            log.warning("microphone %r not found, using the default one", self.device_name)
        return next((d for d in devices if d["default"]), devices[0])

    def start(self) -> str:
        import sounddevice as sd

        dev = self._resolve()
        self.rate = dev["rate"]
        channels = min(2, dev["channels"])

        def cb(indata, frames, t, status):  # PortAudio thread: keep it tiny
            if status:
                self.status_flags += 1
            mono = indata[:, 0] if channels == 1 else indata.mean(axis=1)
            self.on_audio(mono.astype(np.float32, copy=True), self.rate)

        with _PA_LOCK:
            self.stream = sd.InputStream(device=dev["index"], channels=channels, samplerate=self.rate,
                                         dtype="float32", blocksize=int(self.rate * 0.02), latency="low", callback=cb)
            self.stream.start()
        self.device = dev
        log.info("microphone: %s @ %d Hz (%s%s)", dev["name"], self.rate, dev.get("kind"),
                 ", system default" if dev.get("default") else "")
        return dev["name"]

    def stop(self) -> None:
        s, self.stream = self.stream, None
        if s is not None:
            try:
                with _PA_LOCK:
                    s.stop()
                    s.close()
            except Exception:  # noqa: BLE001
                pass


class PushSource:
    """Receives PCM from the browser (see static/app.js -> /ws/audio)."""

    def __init__(self, on_audio: AudioCallback):
        self.on_audio = on_audio
        self.active = False
        self.last_data = 0.0

    def start(self) -> str:
        self.active = True
        return "Browser audio"

    def push_int16(self, data: bytes, rate: int) -> None:
        if not self.active or not data:
            return
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.last_data = time.monotonic()
        self.on_audio(x, int(rate))

    def push_float32(self, data: bytes, rate: int) -> None:
        if not self.active or not data:
            return
        self.last_data = time.monotonic()
        self.on_audio(np.frombuffer(data, dtype="<f4").copy(), int(rate))

    def stop(self) -> None:
        self.active = False


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        rate, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def write_wav(path: str, x: np.ndarray, rate: int = 16000) -> None:
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(y.tobytes())


class FileSource:
    def __init__(self, path: str, on_audio: AudioCallback, speed: float = 1.0, on_end: Callable[[], None] | None = None):
        self.path = path
        self.on_audio = on_audio
        self.speed = speed
        self.on_end = on_end
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> str:
        x, rate = read_wav(self.path)
        chunk = int(rate * 0.02)

        def run():
            t0 = time.monotonic()
            for i in range(0, len(x), chunk):
                if self._stop.is_set():
                    return
                self.on_audio(x[i:i + chunk].copy(), rate)
                if self.speed > 0:
                    due = t0 + (i + chunk) / rate / self.speed
                    d = due - time.monotonic()
                    if d > 0:
                        time.sleep(d)
            if self.on_end:
                self.on_end()

        self._thread = threading.Thread(target=run, name="file-source", daemon=True)
        self._thread.start()
        return f"File: {self.path}"

    def stop(self) -> None:
        self._stop.set()
