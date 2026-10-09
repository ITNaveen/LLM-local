"""Stage 5 - timeline: turns the filled story into exact, frame-accurate segments with
audio levels (who is heard), narration placement, beat-snapped cuts, subtitles and
YouTube chapters."""

import re

# How loud the footage's own audio and the music are, per beat type.
CLIP_GAIN = {"original": 1.0, "narration": 0.12, "music": 0.16, "text": 0.22}
MUSIC_GAIN = {"original": 0.10, "narration": 0.26, "music": 0.9, "card": 0.85, "text": 0.8}
NARRATION_LEAD = 0.35     # voice starts this long after its beat begins
NARRATION_TAIL = 0.55     # breathing room after the voice ends
ACT_FADE = 0.4
TITLE_CARD_SECONDS = 4.0


def snap_to_beats(durations, modes, beats, max_shift=0.3, min_len=1.0):
    """Move cut points (end of non-dialogue shots) onto the nearest music beat."""
    if not beats:
        return durations
    out, t = list(durations), 0.0
    for i, d in enumerate(out):
        end = t + d
        if modes[i] != "original" and i < len(out) - 1:
            nearest = min(beats, key=lambda b: abs(b - end))
            if abs(nearest - end) <= max_shift and nearest - t >= min_len:
                out[i] = nearest - t
        t += out[i]
    return out


def frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


# Without a music track the footage's own sound carries montages and bridges.
CLIP_GAIN_NO_MUSIC = {"original": 1.0, "narration": 0.18, "music": 0.75, "text": 0.35}
TEXT_CARD_FADE = 0.35


def build(story, voice, tracks, beat_grids, settings):
    fps = settings["fps"]
    segments, acts_out, subtitles, chapters = [], [], [], []
    t = 0.0
    n_acts = len(story["acts"])
    for ai, act in enumerate(story["acts"]):
        act_start = t
        has_music = bool(tracks.get(act["key"])) or settings.get("generated_music", False)
        clip_gain = CLIP_GAIN if has_music else CLIP_GAIN_NO_MUSIC
        # 1. flatten the act into items (clips and on-screen text cards), in order
        items, mins = [], {}
        for bi, beat in enumerate(act["beats"]):
            audio = beat.get("audio", "music")
            if audio == "text" and not beat.get("clips"):
                items.append({"kind": "card", "beat": bi, "dur": beat.get("seconds", 3.0),
                              "text": beat.get("narration", ""), "mode": "card"})
                continue
            first = len(items)
            for c in beat.get("clips", []):
                items.append({"kind": "clip", "beat": bi, "dur": c["end"] - c["start"],
                              "clip": c, "mode": audio,
                              "overlay": beat.get("narration", "") if audio == "text" else ""})
            if audio == "narration" and beat.get("narration_id") in voice and len(items) > first:
                mins[bi] = voice[beat["narration_id"]]["duration"] + NARRATION_LEAD + NARRATION_TAIL
        if not items:
            continue
        durs = snap_to_beats([it["dur"] for it in items], [it["mode"] for it in items],
                             beat_grids.get(act["key"]) or [])
        for bi, need in mins.items():           # narration fits inside its pictures
            idx = [i for i, it in enumerate(items) if it["beat"] == bi]
            have = sum(durs[i] for i in idx)
            if have < need:
                durs[idx[-1]] += need - have
        durs = [frames(d, fps) / fps for d in durs]

        # 2. segments
        act_segments, narration, gain_points, beat_start = [], [], [], {}
        for i, it in enumerate(items):
            beat_start.setdefault(it["beat"], t)
            if it["kind"] == "card":
                seg = {"type": "card", "text": it["text"], "dur": durs[i], "frames": frames(durs[i], fps),
                       "t": round(t, 4), "act": act["key"], "beat": it["beat"], "mode": "card",
                       "clip_gain": 0.0, "fade_in": TEXT_CARD_FADE, "fade_out": TEXT_CARD_FADE,
                       "style": "text"}
                gain_points.append((t - act_start, t - act_start + durs[i], MUSIC_GAIN["card"]))
            else:
                c, mode = it["clip"], it["mode"]
                heat_boost = act["key"] == "climax" and mode == "music" and c.get("heat", 0) > 0.6
                seg = {
                    "type": "clip", "video_id": c["video_id"], "src_start": c["start"],
                    "dur": durs[i], "frames": frames(durs[i], fps), "t": round(t, 4),
                    "act": act["key"], "beat": it["beat"], "mode": mode,
                    "clip_gain": max(clip_gain[mode], 0.35 if heat_boost else 0.0),
                    "fade_in": 0.0, "fade_out": 0.0, "heat": c.get("heat", 0),
                    "video_title": c.get("video_title", ""), "channel": c.get("channel", ""),
                }
                if it.get("overlay"):
                    seg["overlay"] = it["overlay"]   # the text fades, the footage keeps moving
                gain_points.append((t - act_start, t - act_start + durs[i], MUSIC_GAIN[mode]))
            act_segments.append(seg)
            t += durs[i]
        if act_segments[0]["type"] == "clip":
            act_segments[0]["fade_in"] = 1.0 if ai == 0 else ACT_FADE
        if act_segments[-1]["type"] == "clip":
            act_segments[-1]["fade_out"] = 1.6 if ai == n_acts - 1 else ACT_FADE

        for bi, beat in enumerate(act["beats"]):
            nid = beat.get("narration_id")
            if beat.get("audio") == "narration" and nid in voice and bi in beat_start:
                vt = beat_start[bi] + NARRATION_LEAD
                narration.append({"file": voice[nid]["file"], "t": round(vt - act_start, 4),
                                  "abs": round(vt, 4), "duration": voice[nid]["duration"]})
                subtitles.extend(split_subtitles(beat["narration"], vt, voice[nid]["duration"]))

        # Title card right after the opening act (hook -> title -> story).
        if ai == 0 and settings.get("title_card", True):
            card_dur = frames(TITLE_CARD_SECONDS, fps) / fps
            act_segments.append({
                "type": "card", "text": story.get("title_hi", ""), "dur": card_dur,
                "frames": frames(card_dur, fps), "t": round(t, 4), "act": act["key"],
                "beat": -1, "mode": "card", "clip_gain": 0.0, "fade_in": 0.6, "fade_out": 0.6,
                "style": "title",
            })
            gain_points.append((t - act_start, t - act_start + card_dur, MUSIC_GAIN["card"]))
            t += card_dur

        chapters.append({"t": round(act_start, 2), "title": act.get("title_hi") or act["key"]})
        acts_out.append({
            "key": act["key"], "title_hi": act.get("title_hi", ""), "mood": act.get("mood", "epic"),
            "start": round(act_start, 4), "end": round(t, 4),
            "frames": sum(s["frames"] for s in act_segments),
            "track": tracks.get(act["key"]), "gain_points": gain_points, "narration": narration,
        })
        segments.extend(act_segments)

    for i, s in enumerate(segments):
        s["i"] = i
    total_frames = sum(s["frames"] for s in segments)
    return {
        "fps": fps, "width": settings["width"], "height": settings["height"],
        "segments": segments, "acts": acts_out, "subtitles": subtitles,
        "chapters": chapters, "duration": round(total_frames / fps, 3),
    }


def split_subtitles(text, start, duration, max_words=6):
    parts = [p.strip() for p in re.split(r"(?<=[,।!?.])\s+", text) if p.strip()]
    chunks = []
    for p in parts:
        words = p.split()
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
    if not chunks:
        return []
    total_chars = sum(len(c) for c in chunks)
    out, t = [], start
    for c in chunks:
        d = duration * len(c) / total_chars
        out.append({"start": round(t, 3), "end": round(t + d, 3), "text": c})
        t += d
    return out
