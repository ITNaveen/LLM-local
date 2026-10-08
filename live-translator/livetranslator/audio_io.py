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


def list_input_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001  (PortAudio missing)
        log.warning("sounddevice unavailable: %s", e)
        return []
    try:
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"],
                            "rate": int(d.get("default_samplerate") or 48000), "default": i == default_in})
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("could not list audio devices: %s", e)
        return []


class MicSource:
    def __init__(self, device_name: str, on_audio: AudioCallback):
        self.device_name = device_name
        self.on_audio = on_audio
        self.stream = None
        self.rate = 0
        self.status_flags = 0

    def _resolve(self):
        devices = list_input_devices()
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

        self.stream = sd.InputStream(device=dev["index"], channels=channels, samplerate=self.rate,
                                     dtype="float32", blocksize=int(self.rate * 0.02), latency="low", callback=cb)
        self.stream.start()
        log.info("microphone: %s @ %d Hz", dev["name"], self.rate)
        return dev["name"]

    def stop(self) -> None:
        s, self.stream = self.stream, None
        if s is not None:
            try:
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
