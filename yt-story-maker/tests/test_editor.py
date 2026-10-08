"""The editor brain: screening, passages, scene planning, editor review, assembly, timeline."""

import pytest

from storymaker import editor, timeline
from storymaker.llm import NoLLM
from storymaker.source import FixtureSource
from storymaker.style import ACT_KEYS, DEFAULT_STYLE
from storymaker.util import keywords

TOPIC, DESC = "Virat Kohli", "pressure then a century"


@pytest.fixture(scope="module")
def material():
    """Fixture videos run through screening + passage understanding with the fake AI."""
    from conftest import FakeLLM
    llm = FakeLLM()
    src = FixtureSource(n_videos=12)
    shortlist = src.search("kohli", 12)
    details = editor.read_videos(src, shortlist, lambda m: None)
    kept, rejected = editor.screen(llm, TOPIC, DESC, {"must_keywords": ["kohli"]}, details, lambda m: None)
    passages, visuals = [], []
    words = keywords(f"{TOPIC} {DESC}")
    for v in kept:
        visuals += editor.visual_moments(v)
        if v["role"] != "speech":
            continue
        allp = editor.build_passages(v)
        chosen = {p["id"] for p in editor.preselect(allp, words)}
        ann = {p["id"]: p for p in editor.annotate(llm, TOPIC, DESC, v,
                                                   [p for p in allp if p["id"] in chosen], print)}
        passages += [ann.get(p["id"], p) for p in allp]
    videos = [{k: x for k, x in v.items() if k not in ("captions", "heatmap")} for v in kept]
    return {"llm": llm, "kept": kept, "rejected": rejected, "passages": passages,
            "visuals": visuals, "videos": videos}


# ------------------------------------------------------------------ screening
@pytest.mark.parametrize("video,role", [
    ({"spoken_lang": "te", "title": "x", "captions": [1], "caption_lang": "te"}, "reject"),
    ({"spoken_lang": "ta", "title": "Germany jobs", "captions": [1], "caption_lang": "en"}, "reject"),
    ({"spoken_lang": "", "title": "జర్మనీ vlog", "captions": [], "caption_lang": ""}, "reject"),
    ({"spoken_lang": "de", "title": "Tagesschau", "captions": [1], "caption_lang": "en"}, "visual"),
    ({"spoken_lang": "hi", "title": "भाषण", "captions": [1], "caption_lang": "hi"}, "speech"),
    ({"spoken_lang": "en", "title": "Speech", "captions": [], "caption_lang": ""}, "visual"),
    ({"spoken_lang": "", "title": "Drone", "captions": [], "caption_lang": ""}, "visual"),
])
def test_language_roles(video, role):
    assert editor.language_role(video)[0] == role


def test_screening_rejects_offtopic_and_regional(material):
    rejected = {r["title"]: r["reason"] for r in material["rejected"]}
    assert "Stand-up comedy special 2025" in rejected            # title filter / AI verdict
    assert any("regional" in why for t, why in rejected.items() if "vlog" in t)
    roles = {v["title"]: v["role"] for v in material["kept"]}
    assert roles["Tagesschau Wahlabend Sondersendung"] == "visual"   # foreign: visuals only
    assert roles["Drone footage of the rally"] == "visual"
    assert roles["Election night: full report"] == "speech"


def test_screening_without_ai_needs_the_topic(settings):
    videos = [{"id": "a", "title": "Kohli century full speech", "captions": [{"text": "x"}],
               "spoken_lang": "en", "caption_lang": "en", "description": ""},
              {"id": "b", "title": "Cooking with spices", "spoken_lang": "en", "caption_lang": "en",
               "captions": [{"text": "add salt"}], "description": ""}]
    kept, rejected = editor.screen(NoLLM(), "Virat Kohli", "", {"must_keywords": ["kohli"]},
                                   videos, lambda m: None)
    assert [v["id"] for v in kept] == ["a"] and rejected[0]["id"] == "b"


# ------------------------------------------------------------------ passages
def test_passages_are_complete_contiguous_thoughts(material):
    for p in material["passages"]:
        assert 6.0 <= p["end"] - p["start"] <= 42
        assert p["cues"][0]["s"] == p["start"] and p["cues"][-1]["e"] == p["end"]
        assert all(a["e"] <= b["s"] + 1e-6 for a, b in zip(p["cues"], p["cues"][1:]))
    # no speech from rejected or visual-only videos is ever used as dialogue
    speech_ids = {v["id"] for v in material["kept"] if v["role"] == "speech"}
    assert {p["video_id"] for p in material["passages"]} <= speech_ids


def test_junk_passages_are_dropped():
    video = {"id": "v", "duration": 300, "title": "t", "heatmap": [], "captions": [
        {"start": 20, "end": 28, "text": "please subscribe and press the bell icon."},
        {"start": 40, "end": 52, "text": "the result changed everything for the party."}]}
    texts = [p["text"] for p in editor.build_passages(video)]
    assert texts == ["the result changed everything for the party."]


# ------------------------------------------------------------------ story planning
def test_plan_story_with_ai_and_editor_review(material):
    logs = []
    out = editor.plan_story(material["llm"], TOPIC, DESC, "epic", 8, "light",
                            material["passages"], material["videos"], logs.append)
    scenes = out["scenes"]
    assert out["source"] == "ai" and out["title_hi"] == "विराट का जवाब"
    assert out["report"]["score"] == 7 and out["report"]["issues"]
    acts = [ACT_KEYS.index(s["act"]) for s in scenes]
    assert acts == sorted(acts)                                   # acts only move forward
    used = [s["pid"] for s in scenes if s["type"] == "dialogue"]
    assert len(used) == len(set(used))                            # each passage plays once
    assert all(pid in out["catalog"] for pid in used)             # P999 was thrown away
    assert sum(s["type"] == "hook" for s in scenes) <= 2
    assert any(s["text"] == "इसी बीच मैदान पर।" for s in scenes if s["type"] == "text")  # bridge
    assert all(s["type"] != "montage" or s["seconds"] <= 25 for s in scenes)
    assert any("Editor review" in line for line in logs)


def test_broken_ai_output_falls_back_to_rules(material):
    from conftest import FakeLLM
    out = editor.plan_story(FakeLLM(broken_outline=True), TOPIC, DESC, "epic", 8, "light",
                            material["passages"], material["videos"], lambda m: None)
    assert out["source"] == "template"
    dia = [s for s in out["scenes"] if s["type"] == "dialogue"]
    assert dia
    # fallback keeps each source together and in time order
    cat = {k: v for k, v in out["catalog"].items()}
    pmap = {p["id"]: p for p in material["passages"]}
    for a, b in zip(dia, dia[1:]):
        pa, pb = pmap[cat[a["pid"]]], pmap[cat[b["pid"]]]
        if pa["video_id"] == pb["video_id"]:
            assert pa["start"] < pb["start"]


def test_same_speaker_plays_in_original_order():
    cat = {"P1": {"video_id": "a", "start": 50}, "P2": {"video_id": "a", "start": 10},
           "P3": {"video_id": "b", "start": 5}}
    scenes = [{"act": "rising", "type": "dialogue", "pid": "P1"},
              {"act": "rising", "type": "text", "text": "x"},
              {"act": "rising", "type": "dialogue", "pid": "P3"},
              {"act": "rising", "type": "dialogue", "pid": "P2"}]
    out = editor.enforce_rules(scenes, cat, "light")
    assert [s.get("pid") for s in out] == ["P2", None, "P3", "P1"]


def test_enforce_rules():
    cat = {"P1": {"video_id": "a", "start": 50}, "P2": {"video_id": "a", "start": 10}}
    scenes = [{"act": "opening", "type": "hook", "pid": "P1"},
              {"act": "opening", "type": "hook", "pid": "P2"},
              {"act": "opening", "type": "hook", "pid": "P1"},
              {"act": "buildup", "type": "dialogue", "pid": "P1"},
              {"act": "buildup", "type": "dialogue", "pid": "P1"}] + \
             [{"act": "rising", "type": "narration", "text": f"line {i}"} for i in range(8)]
    out = editor.enforce_rules(scenes, cat, "light")
    assert sum(s["type"] == "hook" for s in out) == 2
    assert sum(1 for s in out if s["type"] == "dialogue" and s["pid"] == "P1") == 1
    assert sum(s["type"] == "narration" for s in out) == 6       # rest become text cards
    assert sum(s["type"] == "text" for s in out) == 2


# ------------------------------------------------------------------ assembly
def _assemble(material, minutes, narration="light"):
    outline = editor.plan_story(material["llm"], TOPIC, DESC, "epic", minutes, narration,
                                material["passages"], material["videos"], lambda m: None)
    voice = {nid: 5.0 for nid in editor.narration_lines(outline)}
    asm = editor.Assembler(material["passages"], material["videos"], material["visuals"],
                           DEFAULT_STYLE, minutes * 60)
    return outline, asm.build(outline, voice, "epic"), voice


def test_dialogue_scenes_play_one_continuous_passage(material):
    outline, story, _ = _assemble(material, 6)
    pmap = {p["id"]: p for p in material["passages"]}
    for act in story["acts"]:
        for b in act["beats"]:
            if b["kind"] == "dialogue":
                assert len({c["video_id"] for c in b["clips"]}) == 1   # one source per scene
                starts = [c["start"] for c in b["clips"]]
                assert starts == sorted(starts)                        # speaker moves forward
                c, p = b["clips"][0], pmap[b["passage_id"]]
                assert c["video_id"] == p["video_id"]
                assert c["start"] <= p["start"] and c["end"] >= p["end"] - 0.01
                assert b["said"].startswith(p["text"][:40])
            if b["kind"] == "hook":
                c = b["clips"][0]
                assert c["end"] - c["start"] <= 9.5


def test_no_footage_used_twice_except_hook(material):
    _, story, _ = _assemble(material, 8)
    seen = {}
    for act in story["acts"]:
        for b in act["beats"]:
            if b["kind"] == "hook":
                continue
            for c in b["clips"]:
                for s, e in seen.get(c["video_id"], []):
                    assert c["end"] <= s + 0.6 or c["start"] >= e - 0.6, (c, s, e)
                seen.setdefault(c["video_id"], []).append((c["start"], c["end"]))


@pytest.mark.parametrize("minutes", [4, 8])
def test_length_is_met_by_letting_speakers_continue(material, minutes):
    _, story, _ = _assemble(material, minutes)
    total = sum(editor.beat_seconds(b) for a in story["acts"] for b in a["beats"]) + 4
    assert minutes * 60 * 0.9 <= total <= minutes * 60 * 1.1, total
    assert not story["warnings"]


def test_too_little_material_is_reported_not_padded(material):
    _, story, _ = _assemble(material, 40)
    assert story["warnings"] and "shorter" in story["warnings"][0]


def test_timeline_with_text_cards_and_no_music(material, settings):
    outline, story, voice_d = _assemble(material, 6)
    voice = {k: {"file": "x.wav", "duration": v} for k, v in voice_d.items()}
    tl = timeline.build(story, voice, {k: None for k in ACT_KEYS}, {}, settings)
    segs = tl["segments"]
    cards = [s for s in segs if s["type"] == "card"]
    assert any(s.get("style") == "text" and s["text"] for s in cards)
    assert sum(s.get("style") == "title" for s in cards) == 1
    for a, b in zip(segs, segs[1:]):
        assert b["t"] == pytest.approx(a["t"] + a["frames"] / 30, abs=1e-3)
    # without music the montage keeps its own sound, dialogue stays at full level
    assert all(s["clip_gain"] >= 0.7 for s in segs if s.get("mode") == "music")
    assert all(s["clip_gain"] == 1.0 for s in segs if s.get("mode") == "original")
    for act in tl["acts"]:
        for n in act["narration"]:
            assert act["start"] <= n["abs"] and n["abs"] + n["duration"] <= act["end"] + 0.01


def test_numbers_from_ai_text():
    assert [editor._num(x) for x in (7, "7/10", "score 8 of 10", "P12", None, "x")] == \
        [7, 7, 8, 12, None, None]
