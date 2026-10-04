import json

from storymaker import research
from storymaker.llm import NoLLM, parse_json_loose, tfidf_similarity
from storymaker.source import FixtureSource, clean_cues, parse_json3, parse_vtt, pick_caption_track
from storymaker.util import keywords, tokens


def test_hindi_tokens_keep_matras():
    # \w-based tokenizers break "विराट" into pieces; ours must keep whole words
    assert tokens("विराट कोहली का शतक!") == ["विराट", "कोहली", "का", "शतक"]
    assert "का" not in keywords("विराट कोहली का शतक")


def test_parse_json3_skips_appends_and_newlines():
    data = {"events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "नमस्ते "}, {"utf8": "भारत"}]},
        {"tStartMs": 1500, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "[Music]"}]},
        {"tStartMs": 4000, "dDurationMs": 1000, "segs": [{"utf8": "jai hind"}]},
    ]}
    cues = parse_json3(json.dumps(data))
    assert [c["text"] for c in cues] == ["नमस्ते भारत", "jai hind"]
    assert cues[0]["start"] == 0 and cues[1]["start"] == 4.0


def test_parse_vtt_rolling_duplicates():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
<c>this is</c> a test

00:00:02.500 --> 00:00:04.000
this is a test

00:00:04.000 --> 00:00:06.500
this is a test of captions

01:00:00.000 --> 01:00:01.000
late cue
"""
    cues = parse_vtt(vtt)
    assert cues[0]["text"] == "this is a test"
    assert cues[1]["text"] == "of captions"
    assert cues[-1]["start"] == 3600.0
    assert all(cues[i]["end"] <= cues[i + 1]["start"] for i in range(len(cues) - 1))


def test_clean_cues_drops_overlap():
    cues = clean_cues([{"start": 0, "end": 5, "text": "a"}, {"start": 3, "end": 6, "text": "b"}])
    assert cues[0]["end"] == 3


def test_pick_caption_track_prefers_manual_hindi_json3():
    info = {"subtitles": {"en": [{"ext": "vtt", "url": "u-en"}]},
            "automatic_captions": {"hi-orig": [{"ext": "json3", "url": "u-hi"}]}}
    assert pick_caption_track(info)["url"] == "u-en"   # manual beats auto
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": "en"}],
                                   "hi-orig": [{"ext": "vtt", "url": "hi-vtt"}, {"ext": "json3", "url": "hi"}]}}
    t = pick_caption_track(info)
    assert t["url"] == "hi" and t["lang"] == "hi-orig"
    assert pick_caption_track({}) is None


def test_parse_json_loose():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('<think>hmm</think>{"b": 2}') == {"b": 2}
    assert parse_json_loose('Sure! {"c": [1,2]} hope it helps') == {"c": [1, 2]}


def test_tfidf_similarity_prefers_matching_doc():
    sims = tfidf_similarity(["kohli century celebration"],
                            ["modi speech parliament", "kohli hits century, celebration"])
    assert sims[0, 1] > sims[0, 0]


PLAN = {"must_keywords": ["kohli"], "nice_keywords": ["century", "west", "indies"], "years": ["2026"]}


def test_score_candidate_ranks_relevant_original_footage_first():
    good = {"title": "Virat Kohli century highlights West Indies 2026", "views": 2_000_000, "duration": 600}
    reaction = {"title": "Virat Kohli century REACTION", "views": 5_000_000, "duration": 600}
    short = {"title": "Virat Kohli century #shorts", "views": 9_000_000, "duration": 30}
    offtopic = {"title": "Best cooking recipes", "views": 9_000_000, "duration": 600}
    s = {k: research.score_candidate(v, PLAN) for k, v in
         dict(good=good, reaction=reaction, short=short, offtopic=offtopic).items()}
    assert s["good"] > s["reaction"] and s["good"] > s["short"] and s["good"] > s["offtopic"]


def test_make_queries_without_llm_builds_hindi_friendly_queries():
    plan = research.make_queries(NoLLM(), "Virat Kohli", "West Indies tour 2026 century", print)
    assert plan["queries"][0] == "Virat Kohli"
    assert any("highlights" in q for q in plan["queries"])
    assert "2026" in plan["years"]


def test_research_with_llm_and_diversity(settings, fake_llm):
    res = research.research(FixtureSource(settings), fake_llm, "Virat Kohli", "century", settings, print)
    assert res["total_candidates"] == 12
    assert len(res["shortlist"]) == settings["shortlist_size"]
    per_channel = {}
    for c in res["shortlist"]:
        per_channel[c["channel"]] = per_channel.get(c["channel"], 0) + 1
    assert max(per_channel.values()) <= 3
    assert "virat kohli century" in res["plan"]["queries"]


def test_research_runs_searches_in_parallel_with_time_limit(settings, fake_llm):
    import time

    class SlowYouTube(FixtureSource):
        def search(self, query, n):
            time.sleep(30 if "press conference" in query else 1.0)   # one query hangs
            return super().search(query, n)

    settings.update(search_workers=4, research_minutes=0.08)   # ~5 s budget
    logs = []
    t0 = time.time()
    res = research.research(SlowYouTube(settings), fake_llm, "Virat Kohli", "century",
                            settings, logs.append)
    took = time.time() - t0
    assert took < 8, took                      # did not wait for the hanging query
    assert res["total_candidates"] == 12      # but used everything that came back
    assert any("time limit" in line for line in logs)
