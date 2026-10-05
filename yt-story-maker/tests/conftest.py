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
        self.calls.append(system[:40])
        if "research assistant" in system:
            return {"queries": ["virat kohli century", "विराट कोहली शतक", "kohli press conference"],
                    "must_keywords": ["kohli"], "nice_keywords": ["century", "west indies"],
                    "years": ["2026"]}
        if "select raw footage" in system:
            n = len(re.findall(r"^\d+\. ", user, flags=re.M))
            return {"keep": list(range(n, 0, -1))}
        if "strict researcher" in system:
            title = re.search(r"Video title: (.*)", user).group(1)
            ok = "comedy" not in title.lower()
            return {"relevant": ok, "kind": "news" if ok else "comedy", "reason": "test verdict"}
        if "help a top Hindi documentary editor" in system:
            n = len(re.findall(r"^\d+\. \[", user, flags=re.M))
            return {"passages": [{"n": i, "use": True, "summary": f"speaker makes point {i}",
                                  "topic": "result" if i <= n // 2 else "reaction",
                                  "strength": 3 + i % 3, "standalone": True} for i in range(1, n + 1)]}
        if "The film\nis too short" in system:
            unused = re.findall(r"^(P\d+) \| ([^|]+)\|", user.split("UNUSED PASSAGES")[1], flags=re.M)
            n = len(re.findall(r"^\d+ \| ", user.split("UNUSED PASSAGES")[0], flags=re.M))
            ins = []
            for k, (pid, _src) in enumerate(unused[:12]):
                ins.append({"after": max(1, n - k % max(1, n - 1)), "type": "dialogue", "use": pid,
                            "link": "continues this thread"})
            ins.append({"after": 2, "type": "dialogue", "use": "P999"})            # invalid: ignored
            return {"insert": ins}
        if "lead editor" in system:
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
                    {"act": "rising", "type": "text", "text": "और फिर आया आख़िरी टेस्ट।", "link": "new chapter"},
                    dia("rising"), dia("rising"),
                    {"act": "buildup", "type": "dialogue", "use": pids[1], "link": "acts never go back"},
                    dia("climax"), {"act": "climax", "type": "montage", "use": vids[:2], "seconds": 40},
                    {"act": "ending", "type": "narration", "text": "और इस तरह, एक बार फिर, विराट ने इतिहास लिख दिया।", "link": "close"},
                    dia("ending"),
                ]}
        if "ruthless senior editor" in system:
            n = len(re.findall(r"^\d+ \| ", user, flags=re.M))
            order = [i for i in range(1, n + 1) if i != 5]      # drop scene 5
            return {"order": order, "bridges": [{"before": 8, "type": "text", "text": "इसी बीच मैदान पर।"}],
                    "score": 7, "issues": ["scene 5 repeated the previous point"]}
        if "YouTube metadata" in system:
            return {"youtube_title": "विराट कोहली का सबसे बड़ा जवाब | Virat Kohli Century",
                    "description_hi": "विराट की वापसी की पूरी कहानी।",
                    "tags": ["virat kohli", "विराट कोहली"]}
        return {}


@pytest.fixture
def fake_llm():
    return FakeLLM()
