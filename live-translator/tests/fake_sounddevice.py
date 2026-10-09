"""A stand-in for the `sounddevice` module that behaves like a Mac's audio inputs.

Each device has a behaviour:
  "noise"            a working microphone (always some noise, never exact zeros)
  "zeros"            digital silence: macOS permission denied, or a virtual device nobody feeds
  "zeros_first_open" silent the first time it is opened, working afterwards
                     (permission granted while the first recording was already running)
"""
from __future__ import annotations

import threading
import time

import numpy as np


class _Pair:
    """Like sounddevice._InputOutputPair: indexable, but NOT a list/tuple."""

    def __init__(self, i, o):
        self._v = [i, o]

    def __getitem__(self, k):
        return self._v[{"input": 0, "output": 1}.get(k, k)]

    def __repr__(self):
        return repr(self._v)


class FakeSoundDevice:
    def __init__(self, devices, default_index, behaviour):
        self.devices = devices              # list of (name, channels, rate)
        self.default_index = default_index
        self.behaviour = behaviour          # name -> "noise" | "zeros" | "zeros_first_open"
        self.opened = []                    # names in the order they were opened
        self.reinits = 0
        self.live = []                      # streams currently running
        self.killed = []                    # streams killed by a re-init (real Pa_Terminate does that)
        fake = self

        class default:
            device = _Pair(default_index, -1)

        self.default = default

        class InputStream:
            def __init__(self, device=None, channels=1, samplerate=48000, dtype="float32", blocksize=960,
                         callback=None, latency=None, **kw):
                self.name = fake.devices[device][0]
                self.channels, self.rate, self.block, self.cb = channels, samplerate, blocksize or 960, callback
                fake.opened.append(self.name)
                mode = fake.behaviour.get(self.name, "noise")
                if mode == "zeros_first_open":
                    mode = "zeros" if fake.opened.count(self.name) == 1 else "noise"
                self.mode = mode
                self._stop = threading.Event()
                self._t = None

            def start(self):
                fake.live.append(self)

                def run():
                    rng = np.random.default_rng(len(fake.opened))
                    while not self._stop.is_set():
                        if self.mode == "zeros":
                            x = np.zeros((self.block, self.channels), np.float32)
                        else:
                            x = (rng.standard_normal((self.block, self.channels)) * 0.003).astype(np.float32)
                        self.cb(x, self.block, None, None)
                        time.sleep(self.block / self.rate)

                self._t = threading.Thread(target=run, daemon=True)
                self._t.start()

            def stop(self):
                self._stop.set()
                if self in fake.live:
                    fake.live.remove(self)
                if self._t and self._t is not threading.current_thread():
                    self._t.join(1)

            def close(self):
                self.stop()

        self.InputStream = InputStream

    def query_devices(self, device=None, kind=None):
        infos = [{"name": n, "max_input_channels": ch, "max_output_channels": 0, "default_samplerate": float(r),
                  "index": i, "hostapi": 0} for i, (n, ch, r) in enumerate(self.devices)]
        if kind == "input":
            return infos[self.default_index]
        if device is not None:
            return infos[device]
        return infos

    def _terminate(self):
        # like Pa_Terminate: every open stream dies
        self.reinits += 1
        for st in list(self.live):
            self.killed.append(st.name)
            st.stop()

    def _initialize(self):
        pass


MAC_DEVICES = [
    ("iPhone Microphone", 1, 48000),          # Continuity mic listed FIRST, delivers nothing
    ("MacBook Pro Microphone", 1, 48000),     # the system default
    ("Microsoft Teams Audio", 2, 48000),      # virtual
    ("External USB Mic", 1, 44100),
]
