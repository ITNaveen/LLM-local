"""Stage 2 - moments: read each shortlisted video's transcript and YouTube's
"Most replayed" graph, and cut it into candidate moments (the raw material of the edit)."""

import re

JUNK = re.compile(
    r"subscribe|bell icon|like (this|the) video|share (this|the) video|comment (below|box)|"
    r"link in (the )?description|my channel|our channel|sponsor|सब्सक्राइब|लाइक|शेयर|चैनल|"
    r"घंटी|कमेंट|डिस्क्रिप्शन", re.I)
SENTENCE_END = re.compile(r"[।?!.]\s*$")


def heat_at(heatmap, t, default=0.3):
    if not heatmap:
        return default
    for h in heatmap:
        if h["start"] <= t < h["end"]:
            return float(h["value"])
    return float(heatmap[-1]["value"]) if t >= heatmap[-1]["start"] else default


def mean_heat(heatmap, start, end, default=0.3):
    if not heatmap:
        return default
    n = max(1, int((end - start) / 0.5))
    return sum(heat_at(heatmap, start + (end - start) * (i + 0.5) / n, default)
               for i in range(n)) / n


def heat_peaks(heatmap, min_value=0.45, min_gap=12.0):
    """Local maxima of the replay graph, strongest first, spaced apart."""
    if len(heatmap) < 3:
        return []
    peaks = []
    for i in range(1, len(heatmap) - 1):
        v = heatmap[i]["value"]
        if v >= min_value and v >= heatmap[i - 1]["value"] and v >= heatmap[i + 1]["value"]:
            peaks.append(((heatmap[i]["start"] + heatmap[i]["end"]) / 2, v))
    peaks.sort(key=lambda p: p[1], reverse=True)
    chosen = []
    for t, v in peaks:
        if all(abs(t - c[0]) >= min_gap for c in chosen):
            chosen.append((t, v))
    return chosen


def speech_windows(captions, lo, hi, min_len=3.5, max_len=12.0, max_gap=1.2):
    """Group caption cues into sentence-like windows that make sense on their own."""
    windows, cur = [], []

    def flush():
        if cur:
            start, end = cur[0]["start"], cur[-1]["end"]
            if end - start >= min_len:
                windows.append({"start": start, "end": end,
                                "text": " ".join(c["text"] for c in cur)})
        cur.clear()

    for cue in captions:
        if cue["start"] < lo or cue["end"] > hi:
            flush()
            continue
        if cur and (cue["start"] - cur[-1]["end"] > max_gap
                    or cue["end"] - cur[0]["start"] > max_len):
            flush()
        cur.append(cue)
        if SENTENCE_END.search(cue["text"]) and cue["end"] - cur[0]["start"] >= min_len:
            flush()
    flush()
    return windows


def build_moments(video, per_video=45):
    vid, dur = video["id"], float(video.get("duration") or 0)
    if dur <= 0:
        return []
    lo = min(8.0, dur * 0.04)
    hi = dur - min(15.0, dur * 0.05)
    heatmap = video.get("heatmap") or []
    captions = video.get("captions") or []
    vscore = float(video.get("score", 0.5))
    moments = []

    for w in speech_windows(captions, lo, hi):
        if JUNK.search(w["text"]):
            continue
        moments.append({"start": w["start"], "end": w["end"], "text": w["text"], "kind": "speech"})

    for t, _v in heat_peaks(heatmap):
        start, end = max(lo, t - 3.0), min(hi, t + 5.0)
        if end - start < 3:
            continue
        overlapping = [m for m in moments if m["start"] < end and m["end"] > start]
        text = " ".join(m["text"] for m in overlapping)[:300]
        if JUNK.search(text):
            continue
        moments.append({"start": start, "end": end, "text": text, "kind": "peak"})

    if not moments:  # no captions and no replay graph: sample evenly
        t = lo
        while t + 6 <= hi:
            moments.append({"start": t, "end": t + 6, "text": "", "kind": "visual"})
            t += max(15.0, (hi - lo) / 20)

    out = []
    for m in moments:
        heat = mean_heat(heatmap, m["start"], m["end"])
        peak_val = max([heat_at(heatmap, m["start"] + k) for k in range(int(m["end"] - m["start"]) + 1)]
                       or [heat])
        text_q = min(1.0, len(m["text"]) / 80) if m["text"] else 0.0
        m.update({
            "id": f"{vid}@{m['start']:.1f}",
            "video_id": vid,
            "start": round(m["start"], 2),
            "end": round(m["end"], 2),
            "heat": round(heat, 3),
            "peak": round(peak_val, 3),
            "score": round(0.5 * max(heat, peak_val * 0.9) + 0.3 * vscore + 0.2 * text_q, 4),
            "video_title": video.get("title", ""),
            "channel": video.get("channel", ""),
            "upload_date": video.get("upload_date", ""),
            "src_w": video.get("width") or 0,
            "src_h": video.get("height") or 0,
            "video_duration": dur,
        })
        out.append(m)
    out.sort(key=lambda m: m["score"], reverse=True)
    return out[:per_video]


def gather(source, shortlist, log, progress=None, workers=4):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(cand):
        t0 = time.time()
        return cand, source.details(cand["id"]), time.time() - t0

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, c) for c in shortlist]
        for n, fut in enumerate(as_completed(futures), 1):
            if progress:
                progress(n / len(shortlist))
            try:
                cand, d, took = fut.result()
            except Exception as e:  # noqa: BLE001 - skip unavailable / age-restricted videos
                log(f"  [{n}/{len(shortlist)}] skipped a video: {str(e)[:120]}")
                continue
            d["score"] = cand.get("score", 0.5)
            ms = build_moments(d)
            results[cand["id"]] = (d, ms)
            log(f"  [{n}/{len(shortlist)}] {d['title'][:60]} - {len(d.get('captions') or [])} "
                f"caption lines, replay graph {'yes' if d.get('heatmap') else 'no'}, "
                f"{len(ms)} moments ({took:.0f}s)")
    videos, moments = [], []
    for cand in shortlist:  # keep ranking order, deterministic output
        if cand["id"] in results:
            d, ms = results[cand["id"]]
            videos.append({k: v for k, v in d.items() if k not in ("captions", "heatmap")})
            moments.extend(ms)
    if not moments:
        raise RuntimeError("Could not read any of the shortlisted videos.")
    return {"videos": videos, "moments": moments}
