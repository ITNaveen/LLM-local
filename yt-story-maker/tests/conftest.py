import os
import re
import sys
import tempfile
from pathlib import Path

# Isolated data dir for every test run (must be set before storymaker.config is imported).
os.environ["STORYMAKER_DATA"] = tempfile.mkdtemp(prefix="storymaker-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from storymaker import config  # noqa: E402

config.ensure_dirs()


@pytest.fixture
def settings():
    s = config.load_settings()
    s.update(width=320, height=180, preset="ultrafast", shortlist_size=8, fps=30,
             music_dir=str(Path(os.environ["STORYMAKER_DATA"]) / "no-music"))
    return s


class FakeLLM:
    """Plays the part of a local Ollama model with sensible, valid answers."""

    model = "fake"

    def __init__(self, broken_outline=False):
        self.calls = []
        self.broken_outline = broken_outline

    def available(self):
        return True

    def models(self):
        return ["fake"]

    def embed_model(self):
        return ""

    def embed(self, texts):
        raise RuntimeError("no embeddings")

    def chat_json(self, system, user, **kw):
        from storymaker import editor, publish, research
        self.calls.append(system[:40])
        if system == research.QUERY_SYSTEM:
            return {"angle": "Kohli silenced his critics with a century.", "tone": "proud, punchy",
                    "entities": ["Virat Kohli", "विराट कोहली", "West Indies"],
                    "beats": [{"name": "Pressure", "about": "two low scores, media doubts",
                               "queries": ["virat kohli century", "kohli press conference"]},
                              {"name": "The century", "about": "the match-winning hundred",
                               "queries": ["विराट कोहली शतक", "kohli century highlights"]},
                              {"name": "Celebration", "about": "crowd and dressing room",
                               "queries": ["kohli celebration crowd"]}],
                    "broll": [{"what": "stadium crowd", "query": "stadium crowd cheering"}],
                    "avoid": ["claims that Kohli will retire"],
                    "closing_line": "विराट ने फिर साबित कर दिया - बल्ला बोलता है!",
                    "must_keywords": ["kohli"], "years": ["2026"]}
        if system == research.RERANK_SYSTEM:
            n = len(re.findall(r"^\d+\. ", user, flags=re.M))
            return {"keep": list(range(n, 0, -1))}
        if system == editor.SCREEN_SYSTEM:
            title = re.search(r"Video title: (.*)", user).group(1)
            ok = "comedy" not in title.lower()
            opinion = "expert analysis" in title.lower()          # a commentator against the angle
            beat = 1 + sum(map(ord, title)) % 3
            return {"relevant": ok, "beat": beat, "kind": "opinion" if opinion else ("news" if ok else "comedy"),
                    "stance": "opposes" if opinion else "neutral", "reason": "test verdict"}
        if system == editor.ANNOTATE_SYSTEM:
            items = re.findall(r"^(\d+)\. \[\d+s\] (.*)$", user, flags=re.M)
            english = "only hears Hindi" in user
            beats = len(re.findall(r"^\d+\. \w", user.split("Source video:")[0], flags=re.M))
            return {"passages": [{"n": int(k), "use": True, "summary": f"speaker makes point {k}",
                                  "beat": 1 + int(k) % max(1, beats),
                                  "relevance": 2 if int(k) % 7 == 0 else 4,
                                  "topic": "result" if int(k) <= len(items) // 2 else "reaction",
                                  "strength": 3 + int(k) % 3, "standalone": True,
                                  "emotion": ["anger", "pride", "shock"][int(k) % 3],
                                  "punch": " ".join(text.split()[:7]),
                                  **({"hindi": f"वक्ता ने साफ़ कहा कि यह मुद्दा नंबर {k} बेहद गंभीर है।"}
                                     if english else {})} for k, text in items]}
        if system == editor.EXTEND_SYSTEM:
            unused = re.findall(r"^(P\d+) \| ([^|]+)\|", user.split("UNUSED PASSAGES")[1], flags=re.M)
            n = len(re.findall(r"^\d+ \| ", user.split("UNUSED PASSAGES")[0], flags=re.M))
            ins = []
            for k, (pid, _src) in enumerate(unused[:12]):
                ins.append({"after": max(1, n - k % max(1, n - 1)), "type": "dialogue", "use": pid,
                            "link": "continues this thread"})
            ins.append({"after": 2, "type": "dialogue", "use": "P999"})            # invalid: ignored
            ins.append({"after": 3, "type": "narration", "text": "एक और बात।"})    # narration: ignored
            return {"insert": ins}
        if system == editor.ARCHITECT_SYSTEM:
            if self.broken_outline:
                return {"scenes": [{"type": "dance", "use": "P999"}, "junk"]}
            pids = re.findall(r"^(P\d+) \|", user, flags=re.M)
            vids = re.findall(r"^(V\d+) \|", user, flags=re.M)
            d = iter(pids[::3] + pids[1::3])           # spread over the sources, like a real plan
            def dia(act):
                return {"act": act, "type": "dialogue", "use": next(d, pids[0]), "link": "continues the thread"}
            return {
                "title_hi": "विराट का जवाब", "theme": "comeback",
                "chapters": {"opening": "शुरुआत", "buildup": "दबाव", "rising": "तूफ़ान",
                             "climax": "शतक", "ending": "जीत"},
                "scenes": [
                    {"act": "opening", "type": "hook", "use": pids[0], "link": "strongest line"},
                    {"act": "opening", "type": "text", "text": "दो पारियाँ, दो नाकामियाँ।", "link": "context"},
                    {"act": "buildup", "type": "narration", "text": "सवाल उठने लगे थे, लेकिन विराट चुप रहे।", "link": "bridge"},
                    dia("buildup"), dia("buildup"), {"act": "buildup", "type": "dialogue", "use": "P999"},
                    {"act": "buildup", "type": "narration", "text": "देख रहे हो ये लाइटें? ये तूफ़ान है।", "link": "invented"},
                    {"act": "rising", "type": "text", "text": "और फिर आया आख़िरी टेस्ट।", "link": "new chapter"},
                    dia("rising"), dia("rising"),
                    {"act": "rising", "type": "narration", "text": "प्रधानमंत्री ने भी विराट को बधाई दी।", "link": "invented"},
                    {"act": "buildup", "type": "dialogue", "use": pids[1], "link": "acts never go back"},
                    dia("climax"), {"act": "climax", "type": "montage", "use": vids[:2], "seconds": 40},
                    {"act": "ending", "type": "narration", "text": "और इस तरह, एक बार फिर, विराट ने इतिहास लिख दिया।", "link": "close"},
                    dia("ending"),
                ]}
        if system == editor.CRITIC_SYSTEM:
            rows = re.findall(r"^(\d+) \| (\w+) \| (\w+) \|", user, flags=re.M)
            order = [int(n) for n, _a, _t in rows if int(n) != 5]      # drop scene 5
            opener = next((int(n) for n, a, t in rows if a == "climax" and t == "dialogue"), None)
            between = next((int(n) for n, a, t in rows[6:] if t == "dialogue"), None)
            return {"order": order,
                    "bridges": [{"before": opener, "type": "narration", "text": "और फिर आया वो दिन, जिसका सबको इंतज़ार था।"},
                                {"before": between, "type": "narration", "text": "बीच में बोलने वाली लाइन।"}],
                    "score": 7, "issues": ["scene 5 repeated the previous point"]}
        if system == editor.TEASER_SYSTEM:
            n = len(re.findall(r"^B\d+ \|", user, flags=re.M))
            return {"bites": [f"B{min(n, 3)}", "B1", "B2"][:max(2, min(3, n))],
                    "hook_line": "क्या विराट का करियर खत्म हो चुका था? या ये तूफ़ान से पहले की शांति थी?"}
        if system.startswith(editor.FACT_SYSTEM[:60]):
            lines = re.findall(r"^(\d+)\. (.*)$", user.split("NARRATOR LINES:")[1], flags=re.M)
            return {"lines": [{"n": int(k), "text": t} for k, t in lines]}
        if system == publish.META_SYSTEM:
            return {"youtube_title": "विराट कोहली का सबसे बड़ा जवाब | Virat Kohli Century",
                    "description_hi": "विराट की वापसी की पूरी कहानी।",
                    "tags": ["virat kohli", "विराट कोहली"]}
        return {}


@pytest.fixture
def fake_llm():
    return FakeLLM()
