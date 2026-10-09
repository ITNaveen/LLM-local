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
    assert any("Editor review" in line for line in logs)
    texts = " ".join(s.get("text", "") for s in scenes)
    # the critic may set the base as an act opens, never talk between clips of a thread
    assert "और फिर आया वो दिन" in texts and "बीच में बोलने वाली लाइन" not in texts
    assert "एक और बात" not in texts                              # extend adds clips, not talk
    # invented lines are gone: a picture description, a big name nobody in the footage mentions
    assert "देख रहे हो" not in texts and "प्रधानमंत्री" not in texts
    assert any("doesn't support" in line and "प्रधानमंत्री" in line for line in logs)


def test_narrator_only_sets_the_base(material):
    out = editor.plan_story(material["llm"], TOPIC, DESC, "sensational", 8, "medium",
                            material["passages"], material["videos"], lambda m: None)
    scenes = out["scenes"]
    kinds = [s["type"] for s in scenes]
    assert all(not (a == b == "narration") for a, b in zip(kinds, kinds[1:]))
    narr = [s for s in scenes if s["type"] == "narration"]
    assert len(narr) <= editor.NARRATION_LIMITS["medium"] + 1        # + the closing question
    for act in ACT_KEYS:
        assert sum(s["act"] == act for s in narr) <= 2
    assert sum(s["type"] == "voiceover" for s in scenes) <= editor.VOICEOVER_LIMITS["medium"]
    assert all(len(s["text"].split()) <= editor.NARRATION_WORDS + 8 for s in narr)
    assert editor.CTA_LINE in narr[-1]["text"]                       # asks the viewer at the end
    pmap = {p["id"]: p for p in material["passages"]}
    cat = {k: pmap[v] for k, v in out["catalog"].items()}
    talk = editor.estimate_seconds([s for s in scenes if s["type"] in ("narration", "voiceover")], cat, 20)
    assert talk <= 0.25 * editor.estimate_seconds(scenes, cat, 20)   # the clips carry the story


def test_cold_open_teaser(material):
    logs = []
    out = editor.plan_story(material["llm"], TOPIC, DESC, "sensational", 8, "medium",
                            material["passages"], material["videos"], logs.append)
    teaser = out["teaser"]
    assert 2 <= len(teaser) <= 4
    assert len({b["video_id"] for b in teaser}) == len(teaser)      # different sources
    pmap = {p["id"]: p for p in material["passages"]}
    for b in teaser:
        p = pmap[b["passage_id"]]
        assert p["start"] <= b["start"] < b["end"] <= p["end"] and 1.5 <= b["end"] - b["start"] <= 8
        assert editor._speaks_hindi(p)
    body = {out["catalog"][s["pid"]] for s in out["scenes"] if s.get("pid")}
    assert not body & {b["passage_id"] for b in teaser}               # never replayed later
    assert not any(s["type"] == "hook" for s in out["scenes"])
    first = out["scenes"][0]
    assert first["type"] == "narration" and first["text"].startswith("क्या विराट")
    assert any("Cold open" in line for line in logs)


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


def _p(vid, start, text, lang="hi", channel=None, strength=3, hindi=""):
    return {"video_id": vid, "start": start, "text": text, "lang": lang, "strength": strength,
            "heat": 0.5, "channel": channel or f"ch-{vid}", "hindi": hindi}


def test_same_speaker_plays_in_original_order():
    cat = {"P1": _p("a", 50, "alpha one"), "P2": _p("a", 10, "alpha two"), "P3": _p("b", 5, "beta")}
    scenes = [{"act": "opening", "type": "hook", "pid": "P3"},
              {"act": "rising", "type": "dialogue", "pid": "P1"},
              {"act": "rising", "type": "text", "text": "x"},
              {"act": "rising", "type": "dialogue", "pid": "P3"},
              {"act": "rising", "type": "dialogue", "pid": "P2"}]
    out = editor.enforce_rules(scenes, cat, "light")
    assert [s.get("pid") for s in out] == ["P3", "P2", None, "P1"]   # hook line not replayed


def test_rules_hindi_audio_only():
    cat = {"P1": _p("a", 5, "hindi line one", strength=5),
           "P2": _p("b", 5, "english minister speech", lang="en", hindi="मंत्री ने साफ़ कहा कि..."),
           "P3": _p("c", 5, "english with no translation", lang="en")}
    scenes = [{"act": "opening", "type": "hook", "pid": "P2"},          # English hook: dropped
              {"act": "buildup", "type": "dialogue", "pid": "P2"},
              {"act": "buildup", "type": "dialogue", "pid": "P3"},
              {"act": "buildup", "type": "dialogue", "pid": "P1"}]
    out = editor.enforce_rules(scenes, cat, "light")
    assert out[0] == {"act": "opening", "type": "hook", "pid": "P1",
                      "link": "the most explosive line first"}
    vo = [s for s in out if s["type"] == "voiceover"]
    assert len(vo) == 1 and vo[0]["pid"] == "P2" and vo[0]["text"].startswith("मंत्री")
    assert all(s.get("pid") != "P3" for s in out)                      # can't be heard in Hindi
    assert all(cat[s["pid"]]["lang"] == "hi" for s in out if s["type"] in ("hook", "dialogue"))


def test_rules_variety_and_no_repeats():
    cat = {f"P{i}": _p("same", i * 30, f"unique point number {i} about topic{i} details{i}")
           for i in range(1, 5)}
    cat["P5"] = _p("x", 0, "unique point number 1 about topic1 details1")   # same story, other channel
    cat["P6"] = _p("y", 0, "fresh angle completely different words", channel="ch-same")
    scenes = [{"act": "buildup", "type": "dialogue", "pid": f"P{i}"} for i in range(1, 7)]
    out = editor.enforce_rules(scenes, cat, "light")
    used = [s["pid"] for s in out if s["type"] == "dialogue"]
    assert used == ["P1", "P2", "P3", "P6"]    # max 3 per video, P5 repeats P1


def test_rules_text_cards():
    cat = {"P1": _p("a", 5, "hindi line")}
    scenes = [{"act": "opening", "type": "text", "text": "पहली लाइन यहाँ है"},
              {"act": "opening", "type": "text", "text": "दूसरी अलग लाइन"},
              {"act": "buildup", "type": "narration", "text": "पहली लाइन यहाँ है"},   # repeat
              {"act": "buildup", "type": "dialogue", "pid": "P1"}]
    out = editor.enforce_rules(scenes, cat, "light")
    assert out[0]["type"] == "hook"                                    # never open on a card
    types = [s["type"] for s in out]
    assert types.count("text") == 1 and "narration" not in types       # no 2 cards, no repeat
    spoken = editor.enforce_rules(scenes, cat, "light", can_text=False)
    assert "text" not in [s["type"] for s in spoken]                   # narrator says them
    # never two narration lines in a row: the narrator says one line, then the clips talk
    assert [s["text"] for s in spoken if s["type"] == "narration"] == ["पहली लाइन यहाँ है।"]


# ------------------------------------------------------------------ assembly
def _assemble(material, minutes, narration="light"):
    outline = editor.plan_story(material["llm"], TOPIC, DESC, "epic", minutes, narration,
                                material["passages"], material["videos"], lambda m: None)
    voice = {nid: 5.0 for nid in editor.narration_lines(outline)}
    asm = editor.Assembler(material["passages"], material["videos"], material["visuals"],
                           DEFAULT_STYLE, minutes * 60)
    return outline, asm.build(outline, voice, "epic"), voice


def _sound(c):
    """(video, start, end) of what a clip plays: its own sound, or the speaker's for a cutaway."""
    a = c.get("audio")
    return (a["video_id"], a["start"], a["start"] + c["end"] - c["start"]) if a else \
        (c["video_id"], c["start"], c["end"])


def test_dialogue_scenes_play_one_continuous_passage(material):
    outline, story, _ = _assemble(material, 6)
    pmap = {p["id"]: p for p in material["passages"]}
    cutaways = 0
    for act in story["acts"]:
        for b in act["beats"]:
            if b["kind"] == "dialogue":
                sounds = [_sound(c) for c in b["clips"]]
                assert len({v for v, _s, _e in sounds}) == 1           # one voice per scene
                assert [s for _v, s, _e in sounds] == sorted(s for _v, s, _e in sounds)
                p = pmap[b["passage_id"]]
                assert sounds[0][0] == p["video_id"] and b["clips"][0]["video_id"] == p["video_id"]
                assert sounds[0][1] <= p["start"]                      # the speaker is seen first
                # cutaways keep the words continuous: each piece starts where the last ended
                for (_v1, _s1, e1), (_v2, s2, _e2) in zip(sounds, sounds[1:]):
                    assert s2 >= e1 - 0.02
                cutaways += sum(bool(c.get("audio")) for c in b["clips"])
                assert all(c["video_id"] != p["video_id"] for c in b["clips"] if c.get("audio"))
                assert b["said"].startswith(p["text"][:40])
            if b["kind"] == "teaser":
                c = b["clips"][0]
                assert c["end"] - c["start"] <= 8.5 and c["fx"] == "punch"
            if b["kind"] == "voiceover":                 # English speaker, Hindi narrator
                p = pmap[b["passage_id"]]
                assert p["lang"] == "en" and b["narration"] and b["narration_id"]
                assert b["clips"][0]["video_id"] == p["video_id"]
            if b["kind"] in ("dialogue", "teaser"):      # everything heard is Hindi
                assert pmap[b["passage_id"]]["lang"] in editor.DIALOGUE_LANGS
    assert cutaways                                       # long soundbites show the action
    beats = [b for a in story["acts"] for b in a["beats"]]
    assert [b["kind"] for b in beats[:2]] == ["teaser", "teaser"]
    assert any(b["kind"] == "title" and b["sfx"] == "boom" for b in beats)
    teaser = {b["passage_id"] for b in beats if b["kind"] == "teaser"}
    dialogue_passages = [b["passage_id"] for b in beats if b["kind"] in ("dialogue", "voiceover")]
    assert not teaser & set(dialogue_passages)            # cold-open lines are never replayed
    assert len(dialogue_passages) == len(set(dialogue_passages))


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


@pytest.mark.parametrize("minutes", [3, 5])
def test_length_is_met_by_letting_speakers_continue(material, minutes):
    _, story, _ = _assemble(material, minutes)
    total = sum(editor.beat_seconds(b) for a in story["acts"] for b in a["beats"]) + 4
    assert minutes * 60 * 0.9 <= total <= minutes * 60 * 1.1, total
    assert not story["warnings"]


def test_too_little_material_is_reported_not_padded(material):
    _, story, _ = _assemble(material, 15)
    assert story["warnings"] and "shorter" in story["warnings"][0]


def test_timeline_with_text_cards_and_no_music(material, settings):
    settings["builtin_music"] = False
    outline, story, voice_d = _assemble(material, 6)
    voice = {k: {"file": "x.wav", "duration": v} for k, v in voice_d.items()}
    tl = timeline.build(story, voice, {k: None for k in ACT_KEYS}, {}, settings)
    segs = tl["segments"]
    cards = [s for s in segs if s["type"] == "card"]
    # the title sits on moving action footage (not a frozen card), with a boom
    assert not cards and sum(s.get("overlay_style") == "title" for s in segs) == 1
    assert any(fx["kind"] == "boom" for fx in tl["acts"][0]["effects"])
    # on-screen text sits on moving footage, never on a frozen frame
    assert any(s["type"] == "clip" and s.get("overlay") for s in segs)
    assert not any(s.get("style") == "text" for s in cards)
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


def test_undownloadable_videos_are_rejected():
    v = {"spoken_lang": "en", "title": "LIVE", "captions": [1], "caption_lang": "en", "downloadable": False}
    assert editor.language_role(v)[0] == "reject"


def test_length_is_guaranteed_from_unused_material():
    cat = {f"P{i}": _p(f"v{i % 15}", 10 + i * 40, f"distinct fact {i} word{i} other{i * 7}",
                       lang="hi" if i % 3 else "en", strength=3 + i % 3,
                       hindi=f"इस बात {i} पर नज़र डालिए, ये बहुत बड़ा मुद्दा है और सबको जानना चाहिए।")
           for i in range(1, 40)}
    for i, p in enumerate(cat.values()):
        p["end"] = p["start"] + 30
        p["subtopic"] = "t%d" % (i % 4)
    scenes = [{"act": "opening", "type": "hook", "pid": "P1"},
              {"act": "buildup", "type": "dialogue", "pid": "P2"},
              {"act": "climax", "type": "dialogue", "pid": "P4"}]
    out = editor.fill_to_length(scenes, cat, 8, clip_cap=20)
    assert editor.estimate_seconds(out, cat, 20) >= 8 * 60 * 0.92 - 4
    acts = [editor.ACT_KEYS.index(s["act"]) for s in out]
    assert acts == sorted(acts) and out[0]["type"] == "hook"
    per_video = {}
    for s in out:
        if s["type"] in ("dialogue", "voiceover"):
            per_video[cat[s["pid"]]["video_id"]] = per_video.get(cat[s["pid"]]["video_id"], 0) + 1
    assert max(per_video.values()) <= editor.MAX_PER_VIDEO + 1
    assert all(s["type"] == "voiceover" for s in out if s.get("pid") and cat[s["pid"]]["lang"] == "en")


def test_hindi_news_style_cuts_clips_to_their_sharpest_part(material):
    outline = editor.plan_story(material["llm"], TOPIC, DESC, "sensational", 5, "medium",
                                material["passages"], material["videos"], lambda m: None)
    assert outline["clip_cap"] == editor.CLIP_CAP
    asm = editor.Assembler(material["passages"], material["videos"], material["visuals"],
                           DEFAULT_STYLE, 5 * 60)
    story = asm.build(outline, {n: 5.0 for n in editor.narration_lines(outline)}, "sensational")
    teaser = [b for a in story["acts"] for b in a["beats"] if b["kind"] == "teaser"]
    assert 2 <= len(teaser) <= 4 and story["acts"][0]["beats"][0]["kind"] == "teaser"
    for a in story["acts"]:
        for b in a["beats"]:
            if b["kind"] == "dialogue":
                # what is heard: max 2 cuts of the speaker, each within the cap (pictures may
                # change underneath - cutaways - while the words run on)
                groups = []
                for v, s0, e0 in (_sound(c) for c in b["clips"]):
                    if groups and abs(groups[-1][1] - s0) < 0.005:      # cutaway pieces join exactly
                        groups[-1][1] = e0
                    else:
                        groups.append([s0, e0])
                assert len(groups) <= 2
                assert all(e0 - s0 <= editor.CLIP_CAP + 0.6 + 1e-6 for s0, e0 in groups)
    kinds = [b["kind"] for a in story["acts"] for b in a["beats"]]
    assert all(not (x == y == "narration") for x, y in zip(kinds, kinds[1:]))
