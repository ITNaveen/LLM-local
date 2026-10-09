"""Relevance (editor v6): one production brief - the creator's angle, story beats, wanted
visuals, claims to avoid and closing line - drives the search, the screening, the passages, the
plan and the pictures. Built from the real failure: an 'Elon Musk vs Bharat' brief typed into
the outline box never reached the search, and the film was full of unrelated clips."""

import json

from storymaker import editor, research
from storymaker.llm import NoLLM
from storymaker.style import DEFAULT_STYLE

BRIEF_TEXT = ("Create a high-energy, sarcastic Hinglish video exposing Elon Musk's double standards "
              "over Starlink's delayed entry into India. Musk questioned whether Mukesh Ambani is the "
              "real boss of India. America restricted Huawei over national security. End with: Elon "
              "bhai, Bharat kisi billionaire ke pressure mein nahi chalega! Do not claim Starlink spies.")

BRIEF = {"angle": "India has every right to security checks; Musk is a hypocrite.", "tone": "sarcastic",
         "entities": ["Elon Musk", "एलन मस्क", "Starlink", "Mukesh Ambani", "Huawei"],
         "beats": [{"name": "Musk's Ambani jibe", "about": "the X post", "queries": ["musk ambani post"]},
                   {"name": "India's reply", "about": "government says same rules", "queries": ["starlink india govt"]},
                   {"name": "Huawei ban", "about": "US banned Huawei", "queries": ["huawei ban america"]}],
         "broll": [{"what": "Starlink satellites", "query": "starlink satellites footage"}],
         "avoid": ["that Starlink spies on India"],
         "closing_line": "एलन भाई, भारत किसी billionaire के pressure में नहीं चलेगा!"}


class BriefLLM:
    """Answers the brief request and records what it was asked."""
    def __init__(self):
        self.prompts = []

    def available(self):
        return True

    def chat_json(self, system, user, **kw):
        self.prompts.append(user)
        if system == research.QUERY_SYSTEM:
            return dict(BRIEF, must_keywords=["starlink"], years=["2026"])
        return {}


# ------------------------------------------------------------------ the brief
def test_the_whole_brief_reaches_the_search_even_from_the_outline_box():
    llm = BriefLLM()
    plan = research.make_queries(llm, "ELON MUSK vs BHARAT", "", print, outline=BRIEF_TEXT)
    assert "Mukesh Ambani is the real boss" in llm.prompts[0]           # the outline was read
    assert plan["brief"]["beats"][2]["name"] == "Huawei ban"
    assert plan["queries"][:3] == ["musk ambani post", "starlink india govt", "huawei ban america"]
    assert {"q": "starlink satellites footage", "beat": 0, "broll": True} in plan["tagged"]
    assert plan["brief"]["closing_line"].startswith("एलन भाई")


def test_brief_without_ai_uses_the_outline_lines():
    plan = research.make_queries(NoLLM(), "Kohli comeback", "", print,
                                 outline="1. Two low scores\n2. Media doubts him\n- The century")
    assert [b["name"] for b in plan["brief"]["beats"]] == ["Two low scores", "Media doubts him", "The century"]


def test_shortlist_takes_from_every_beat_and_never_pads_with_rejected_videos():
    pool = [{"id": f"a{i}", "channel": f"c{i}", "beats": [1], "broll": False} for i in range(10)] + \
           [{"id": "b0", "channel": "x", "beats": [2], "broll": False},
            {"id": "s0", "channel": "y", "beats": [], "broll": True}]
    top = research.balanced_top(pool, 4, 2)
    assert {c["id"] for c in top} >= {"a0", "b0", "s0"}

    class Picky:
        def chat_json(self, *a, **k):
            return {"keep": [1, 2, 3, 4]}
    short = [dict(c, title=c["id"], duration=300, views=1) for c in pool]
    kept = research._llm_rerank(Picky(), "t", BRIEF, short, 6, print)
    assert [c["id"] for c in kept] == ["a0", "a1", "a2", "a3"]           # nothing else sneaks in


# ------------------------------------------------------------------ screening
def _video(vid, title, channel="News", beats=(1,), broll=False):
    return {"id": vid, "title": title, "channel": channel, "duration": 600, "description": "",
            "spoken_lang": "hi", "caption_lang": "hi", "beats": list(beats), "broll": broll,
            "captions": [{"start": 10, "end": 14, "text": "एलन मस्क और स्टारलिंक पर बड़ी खबर"}]}


class ScreenLLM:
    def available(self):
        return True

    def chat_json(self, system, user, **kw):
        title = user.split("Video title: ")[1].split("\n")[0]
        if "Rathee" in title or "Reality" in title:
            return {"relevant": True, "beat": 1, "kind": "opinion", "stance": "opposes", "reason": "x"}
        if "Musk speaks" in title:     # the other side's own words: kept, the film answers them
            return {"relevant": True, "beat": 1, "kind": "speech", "stance": "opposes", "reason": "x"}
        if "Tesla" in title:
            return {"relevant": False, "beat": 0, "kind": "news", "stance": "neutral", "reason": "other story"}
        assert "Huawei ban" in user and "Bharat kisi" not in user          # the beats are shown
        return {"relevant": True, "beat": 3, "kind": "footage", "stance": "neutral", "reason": "x"}


def test_screening_follows_the_brief_and_the_creators_side():
    videos = [_video("v1", "Starlink India security news"), _video("v2", "Dhruv Rathee on Ambani", "Dhruv Rathee"),
              _video("v3", "The Reality of Ambani", "Some Commentator"), _video("v4", "Musk speaks on India"),
              _video("v5", "Tesla Model Y launch"), _video("v6", "Starlink satellites 4K", beats=(), broll=True)]
    kept, rejected = editor.screen(ScreenLLM(), "ELON MUSK vs BHARAT", "", {"brief": BRIEF}, videos,
                                   lambda m: None, avoid=["dhruv rathee"])
    why = {r["id"]: r["reason"] for r in rejected}
    assert "never want" in why["v2"]                                      # the creator's list
    assert "against the film" in why["v3"]                                # opposing commentary
    assert "not relevant" in why["v5"]
    by_id = {v["id"]: v for v in kept}
    assert set(by_id) == {"v1", "v4", "v6"}
    assert by_id["v1"]["beats"] == [3]
    assert by_id["v6"]["role"] == "visual"                                # pictures only


# ------------------------------------------------------------------ passages and plan
def _p(pid, beat, rel=4, vid=None, strength=3, lang="hi", start=10.0):
    return {"id": pid, "video_id": vid or f"v{pid}", "start": start, "end": start + 15, "text": f"text {pid}",
            "lang": lang, "use": True, "beat": beat, "relevance": rel, "strength": strength, "heat": 0.5,
            "subtopic": f"t{beat}", "summary": pid, "channel": f"ch{pid}", "cues": [], "hindi": ""}


def test_catalog_keeps_only_on_story_passages_from_every_beat():
    passages = [_p(f"a{i}", 1, strength=5) for i in range(8)] + [_p("b1", 2), _p("c1", 3), _p("off", 2, rel=2)]
    cat = editor.make_catalog(passages, limit=4)
    ids = [p["id"] for p in cat.values()]
    assert "off" not in ids and {"b1", "c1"} <= set(ids)
    assert [p["beat"] for p in cat.values()] == sorted(p["beat"] for p in cat.values())


def test_a_weaker_beat_is_never_starved():
    # every video's strongest passages are about beat 2 - beat 1 must still get its place
    passages = []
    for v in range(4):
        passages += [_p(f"{v}s{i}", 2, vid=f"v{v}", strength=5, start=10 + i * 20) for i in range(5)]
        passages += [_p(f"{v}w", 1, vid=f"v{v}", strength=2, start=200)]
    cat = editor.make_catalog(passages, limit=12, per_video=4)
    beats = [p["beat"] for p in cat.values()]
    assert beats.count(1) >= 3
    assert max(sum(p["video_id"] == f"v{v}" for p in cat.values()) for v in range(4)) <= 4


def test_film_follows_the_beats_in_order():
    cat = {"P1": _p("x", 3), "P2": _p("y", 1), "P3": _p("z", 2), "P4": _p("w", 1)}
    scenes = [{"act": "opening", "type": "hook", "pid": "P2"},
              {"act": "buildup", "type": "narration", "text": "Huawei!"},
              {"act": "buildup", "type": "dialogue", "pid": "P1"},
              {"act": "rising", "type": "dialogue", "pid": "P3"},
              {"act": "rising", "type": "narration", "text": "Musk ne kaha"},
              {"act": "climax", "type": "dialogue", "pid": "P4"},
              {"act": "ending", "type": "narration", "text": "Bharat ke rules Bharat tay karega!"}]
    out = editor.order_by_beats(scenes, cat, 3)
    assert [s.get("pid") or s["text"] for s in out] == \
        ["P2", "Musk ne kaha", "P4", "P3", "Huawei!", "P1", "Bharat ke rules Bharat tay karega!"]
    assert [s["act"] for s in out] == ["opening", "buildup", "buildup", "rising", "ending", "ending", "ending"]


def test_creators_punchline_closes_the_film_untouched():
    scenes = [{"act": "climax", "type": "dialogue", "pid": "P1"},
              {"act": "ending", "type": "narration", "text": "पुरानी लाइन।"}]
    out = editor.add_closing(scenes, "medium", BRIEF["closing_line"])
    assert out[-1]["text"] == BRIEF["closing_line"] and out[-1]["creator"]
    outline = {"scenes": out + []}
    editor.check_facts(NoLLM(), "t", "", outline, {}, [], [], lambda m: None, BRIEF)
    assert outline["scenes"][-1]["text"] == BRIEF["closing_line"]          # never "fact-checked" away
    assert editor.space_out_narrator([{"type": "narration", "text": "a"}, out[-1]])[-1]["creator"]


def test_pictures_come_only_from_videos_that_tell_this_story():
    videos = [{"id": "news", "title": "Starlink India", "kind": "news", "beats": [2], "stance": "neutral"},
              {"id": "rathee", "title": "Reality", "kind": "opinion", "beats": [2], "stance": "opposes"},
              {"id": "sat", "title": "Starlink satellites", "kind": "footage", "beats": [], "broll": True},
              {"id": "other", "title": "Starlink India 2", "kind": "news", "beats": [1], "stance": "neutral"}]
    visuals = [{"video_id": v["id"], "start": 20, "end": 30, "peak": 0.9 if v["id"] == "rathee" else 0.5,
                "talk": 0.0, "video_duration": 300, "video_title": v["title"], "channel": "c"} for v in videos]
    asm = editor.Assembler([], videos, visuals, DEFAULT_STYLE, 300)
    assert "rathee" not in {m["video_id"] for m in asm.visuals}
    shots = asm.broll(9, shot=3.0, action=True, beat=2)
    assert [c["video_id"] for c in shots] == ["news", "sat", "other"]      # same beat, visuals, rest


def test_brief_is_saved_with_the_plan(tmp_path):
    from conftest import FakeLLM
    from storymaker.source import FixtureSource
    src = FixtureSource(n_videos=12)
    llm = FakeLLM()
    plan = research.make_queries(llm, "Virat Kohli", "century", print)
    details = editor.read_videos(src, src.search("kohli", 12), lambda m: None)
    kept, _ = editor.screen(llm, "Virat Kohli", "", plan, details, lambda m: None)
    assert all(v.get("beats") for v in kept if v["role"] == "speech")
    assert not any(v["title"].endswith("Expert analysis: what changes now") for v in kept)  # opposing opinion
    json.dumps(plan)                                                         # research.json-safe


def test_a_beat_only_english_footage_covers_keeps_one_retold_clip():
    cat = {"P1": _p("h1", 1), "P2": _p("h2", 1, vid="v2"),
           "E1": dict(_p("e1", 2, lang="en"), hindi="अमेरिका ने Huawei पर रोक लगाई थी।"),
           "E2": dict(_p("e2", 2, lang="en", vid="v9"), hindi="एक और बात।"),
           "E3": dict(_p("e3", 1, lang="en", vid="v8"), hindi="अंग्रेज़ी में वही बात।")}
    scenes = [{"act": "buildup", "type": "dialogue", "pid": k} for k in ("P1", "P2", "E3", "E1", "E2")]
    out = editor.enforce_rules(scenes, cat, "none")
    assert [s["pid"] for s in out if s["type"] == "voiceover"] == []        # no narrator at all
    out = editor.enforce_rules(scenes, cat, "light")
    vo = [s["pid"] for s in out if s["type"] == "voiceover"]
    assert "E1" in vo                                                         # beat 2's only voice


def test_every_beat_of_the_brief_gets_a_clip():
    cat = {"P1": _p("a", 1), "P2": _p("b", 2), "P3": _p("c", 3, strength=2),
           "P4": dict(_p("d", 4, lang="en"), hindi="अमेरिका ने Huawei पर रोक लगाई थी।")}
    scenes = [{"act": "buildup", "type": "dialogue", "pid": "P1"},
              {"act": "rising", "type": "dialogue", "pid": "P2"}]
    out = editor.order_by_beats(editor.cover_beats(scenes, cat, 4), cat, 4)
    assert [s["pid"] for s in out] == ["P1", "P2", "P3", "P4"]
    assert out[-1]["type"] == "voiceover"                       # English-only beat, retold
