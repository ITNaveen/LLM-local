import copy

import pytest

from storymaker import moments as M
from storymaker import story, timeline
from storymaker.llm import NoLLM
from storymaker.source import FixtureSource
from storymaker.style import ACT_KEYS, DEFAULT_STYLE


@pytest.fixture(scope="module")
def all_moments():
    src = FixtureSource(n_videos=12)
    shortlist = src.search("kohli", 12)
    for c in shortlist:
        c["score"] = 0.6
    return M.gather(src, shortlist, lambda m: None)["moments"]


def test_heat_peaks_spaced_and_sorted():
    hm = [{"start": i, "end": i + 1, "value": v} for i, v in
          enumerate([0.1, 0.9, 0.2, 0.3, 0.95, 0.2, 0.1, 0.5, 0.2])]
    peaks = M.heat_peaks(hm, min_value=0.45, min_gap=2)
    assert [round(t, 1) for t, _ in peaks] == [4.5, 1.5, 7.5]


def test_speech_windows_group_sentences():
    caps = [{"start": 10, "end": 12, "text": "हम जीतेंगे"}, {"start": 12.2, "end": 14.5, "text": "ज़रूर जीतेंगे।"},
            {"start": 20, "end": 22, "text": "new thought"}, {"start": 22.1, "end": 25, "text": "continues here"}]
    w = M.speech_windows(caps, 0, 100)
    assert len(w) == 2
    assert w[0]["text"] == "हम जीतेंगे ज़रूर जीतेंगे।"
    assert w[0]["start"] == 10 and w[0]["end"] == 14.5


def test_build_moments_skips_intro_outro_and_junk():
    video = {"id": "v1", "duration": 300, "title": "t", "score": 0.5,
             "captions": [{"start": 1, "end": 6, "text": "welcome to my channel friends"},
                          {"start": 50, "end": 55, "text": "please subscribe and press the bell icon"},
                          {"start": 100, "end": 106, "text": "this is the real historic speech."},
                          {"start": 292, "end": 299, "text": "thanks for watching everyone."}],
             "heatmap": []}
    ms = M.build_moments(video)
    texts = [m["text"] for m in ms]
    assert texts == ["this is the real historic speech."]


def test_normalize_outline_repairs_bad_llm_output(all_moments):
    menu = story.moment_menu(all_moments)
    raw = {"title_hi": "**शीर्षक** #viral", "acts": [
        {"key": "OPENING", "beats": [{"audio": "Narration", "narration": "", "moments": [1]}]},
        {"key": "buildup", "beats": [{"audio": "whatever", "moments": ["2", 9999, "abc"]}]},
        {"key": "rising", "beats": []}, {"key": "climax", "beats": "oops"},
        {"key": "ending", "beats": [{"audio": "narration", "narration": "अंत। subscribe करें"}]}]}
    out = story.normalize_outline(raw, menu, "topic", "epic", "light")
    assert [a["key"] for a in out["acts"]] == ACT_KEYS
    assert out["title_hi"] == "शीर्षक viral"
    for a in out["acts"]:
        assert story.ACT_BEATS[a["key"]][0] <= len(a["beats"])
        for b in a["beats"]:
            assert b["audio"] in story.AUDIO_MODES
            assert (b["audio"] == "narration") == bool(b["narration"])
            assert "subscribe" not in b["narration"].lower()
    assert out["acts"][1]["beats"][0]["moment_ids"] == [menu[1]["id"]]
    assert story.normalize_outline({"acts": "nope"}, menu, "t", "epic", "light") is None


def test_narration_none_and_limit(all_moments):
    menu = story.moment_menu(all_moments)
    raw = {"acts": [{"key": k, "beats": [{"audio": "narration", "narration": f"पंक्ति {i}"}
                                          for i in range(5)]} for k in ACT_KEYS]}
    none = story.normalize_outline(copy.deepcopy(raw), menu, "t", "epic", "none")
    assert not any(b["narration"] for a in none["acts"] for b in a["beats"])
    light = story.normalize_outline(copy.deepcopy(raw), menu, "t", "epic", "light")
    assert sum(bool(b["narration"]) for a in light["acts"] for b in a["beats"]) <= 7


def _filled(all_moments, minutes, llm, narration="light"):
    outline = story.plan_outline(llm, "Virat Kohli", "century", "epic", minutes, narration,
                                 all_moments, lambda m: None)
    story.assign_narration_ids(outline)
    dur = {b["narration_id"]: 6.0 for a in outline["acts"] for b in a["beats"] if b.get("narration_id")}
    filler = story.Filler(all_moments, DEFAULT_STYLE, minutes * 60, "Virat Kohli", llm=llm)
    return filler.fill(outline, dur), dur


@pytest.mark.parametrize("minutes", [8, 10, 15])
def test_fill_hits_requested_length_without_reusing_footage(all_moments, fake_llm, minutes):
    st, _ = _filled(all_moments, minutes, fake_llm)
    total = sum(b["seconds"] for a in st["acts"] for b in a["beats"]) + timeline.TITLE_CARD_SECONDS
    assert minutes * 60 * 0.93 <= total <= minutes * 60 * 1.07, total
    # no footage used twice (the cold-open teaser is allowed to flash-forward)
    used = {}
    for a in st["acts"]:
        for b in a["beats"]:
            if b.get("teaser"):
                continue
            for c in b["clips"]:
                for s, e in used.get(c["video_id"], []):
                    assert c["end"] <= s or c["start"] >= e, (c, s, e)
                used.setdefault(c["video_id"], []).append((c["start"], c["end"]))
    assert st["acts"][0]["beats"][0].get("teaser")


def test_original_beats_use_speech(all_moments, fake_llm):
    st, _ = _filled(all_moments, 10, fake_llm)
    for a in st["acts"]:
        for b in a["beats"]:
            if b["audio"] == "original":
                assert b["clips"] and all(c["text"] for c in b["clips"])


def test_llm_preferred_moments_are_used(all_moments, fake_llm):
    st, _ = _filled(all_moments, 10, fake_llm)
    chosen = {c["moment_id"] for a in st["acts"] for b in a["beats"] for c in b["clips"]}
    preferred = {i for a in st["acts"] for b in a["beats"] for i in b.get("moment_ids", [])}
    assert len(chosen & preferred) >= len(preferred) * 0.5


def test_fallback_story_without_llm(all_moments):
    st, _ = _filled(all_moments, 10, NoLLM())
    assert st["source"] == "template"
    assert [a["key"] for a in st["acts"]] == ACT_KEYS


def test_broken_llm_outline_falls_back(all_moments):
    from conftest import FakeLLM
    outline = story.plan_outline(FakeLLM(broken_outline=True), "t", "", "epic", 10, "light",
                                 all_moments, lambda m: None)
    assert outline["source"] == "template"


def test_snap_to_beats():
    durs = timeline.snap_to_beats([2.1, 1.9, 3.0, 2.0], ["music", "music", "original", "music"],
                                  [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    assert durs[0] == pytest.approx(2.0)
    assert durs[0] + durs[1] == pytest.approx(4.0)
    assert durs[2] == 3.0   # dialogue is never cut short


def test_timeline_build(all_moments, fake_llm, settings):
    st, dur = _filled(all_moments, 8, fake_llm)
    voice = {k: {"file": "x.wav", "duration": v} for k, v in dur.items()}
    grids = {k: [i * 0.5 for i in range(2000)] for k in ACT_KEYS}
    tl = timeline.build(st, voice, {k: None for k in ACT_KEYS}, grids, settings)
    segs = tl["segments"]
    assert tl["duration"] == pytest.approx(sum(s["frames"] for s in segs) / 30)
    for a, b in zip(segs, segs[1:]):
        assert b["t"] == pytest.approx(a["t"] + a["frames"] / 30, abs=1e-3)
    assert sum(s["type"] == "card" for s in segs) == 1
    assert tl["chapters"][0]["t"] == 0 and len(tl["chapters"]) == 5
    # every narration line fits inside its beat
    for act in tl["acts"]:
        for n in act["narration"]:
            assert act["start"] <= n["abs"] and n["abs"] + n["duration"] <= act["end"] + 0.01
    # dialogue loud, music low during dialogue
    orig = [s for s in segs if s["mode"] == "original"]
    assert orig and all(s["clip_gain"] == 1.0 for s in orig)
    assert tl["subtitles"] and all(s["end"] > s["start"] for s in tl["subtitles"])


def test_split_subtitles_covers_duration():
    subs = timeline.split_subtitles("एक दो तीन चार पाँच छह सात आठ नौ दस, ग्यारह बारह।", 10.0, 6.0)
    assert subs[0]["start"] == 10.0 and subs[-1]["end"] == pytest.approx(16.0)
    assert all(len(s["text"].split()) <= 6 for s in subs)
