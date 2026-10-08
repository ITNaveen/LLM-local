"""Acoustic evaluation of the listening front end (no speech model needed).

For several room/volume scenarios and voices: does every sentence end up in
exactly one line, is nothing lost, are there false lines, and how quickly
is a line closed after the speaker stops?

    python tools/eval_segmentation.py [--voices 4] [--no-agc] [--sensitivity normal]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livetranslator import simulate as sim  # noqa: E402
from livetranslator.config import Settings  # noqa: E402
from livetranslator.frontend import FrontEnd  # noqa: E402
from livetranslator.vad import SileroVAD  # noqa: E402

EXTRA = {
    "very_quiet": sim.Scenario("very_quiet", speaker=True, rt60=0.4, noise_db=-62, levels_db=(-44.0, -48.0, -40.0)),
    "noisy": sim.Scenario("noisy", speaker=True, rt60=0.5, noise_db=-38, levels_db=(-22.0, -26.0)),
    "swings": sim.Scenario("swings", speaker=True, rt60=0.4, noise_db=-55, levels_db=(-18.0, -40.0, -24.0, -44.0),
                           swing_db=10.0),
}


def evaluate(audio, spans, settings, agc=True, chunk=320):
    fe = FrontEnd(settings, SileroVAD(), agc=agc)
    segs = []
    for i in range(0, audio.size, chunk):
        segs += fe.process(audio[i:i + chunk], 16000)
    segs += fe.flush()
    res = {"sent": len(spans), "lines": len(segs), "split": 0, "merged": 0, "false": 0, "lost_s": 0.0, "lat": []}
    for a, b in spans:
        over = [s for s in segs if s.start < b and s.end > a]
        covered = 0.0
        for s in over:
            covered += max(0.0, min(b, s.end) - max(a, s.start))
        res["lost_s"] += max(0.0, (b - a) - covered)
        if len(over) > 1:
            res["split"] += 1
        if over:
            last = max(over, key=lambda s: s.end)
            res["lat"].append(last.detected_at - b)
    for s in segs:
        n = sum(1 for a, b in spans if s.start < b and s.end > a)
        if n == 0:
            res["false"] += 1
        if n > 1:
            res["merged"] += 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=int, default=3)
    ap.add_argument("--no-agc", action="store_true")
    ap.add_argument("--sensitivity", default="normal")
    ap.add_argument("--pause", type=int, default=500)
    a = ap.parse_args()
    settings = Settings(sensitivity=a.sensitivity, pause_ms=a.pause)
    scen = {**sim.SCENARIOS, **EXTRA}
    tot = {"sent": 0, "lines": 0, "split": 0, "merged": 0, "false": 0, "lost_s": 0.0, "lat": []}
    print(f"{'scenario':<11}{'voice':>6}{'sent':>6}{'lines':>6}{'split':>6}{'merge':>6}{'false':>6}{'lost s':>8}{'close lag':>10}")
    for v in range(a.voices):
        clips = [sim.synthesize(de, v) for de, _ in sim.SENTENCES]
        for name, sc in scen.items():
            audio, spans = sim.make_meeting(clips, sc, seed=v + 1)
            r = evaluate(audio, spans, settings, agc=not a.no_agc)
            lat = np.median(r["lat"]) if r["lat"] else float("nan")
            print(f"{name:<11}{v:>6}{r['sent']:>6}{r['lines']:>6}{r['split']:>6}{r['merged']:>6}{r['false']:>6}"
                  f"{r['lost_s']:>8.2f}{lat:>9.2f}s")
            for k in ("sent", "lines", "split", "merged", "false", "lost_s"):
                tot[k] += r[k]
            tot["lat"] += r["lat"]
    print(f"\nTOTAL sentences={tot['sent']} lines={tot['lines']} split={tot['split']} merged={tot['merged']} "
          f"false={tot['false']} lost={tot['lost_s']:.2f}s  close-lag median={np.median(tot['lat']):.2f}s "
          f"p90={np.percentile(tot['lat'], 90):.2f}s")


if __name__ == "__main__":
    main()
