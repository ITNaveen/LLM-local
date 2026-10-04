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
        if "story-video editor" in system:
            if self.broken_outline:
                return {"acts": [{"key": "climax", "beats": [{"audio": "dance", "moments": ["x", 999]}]}]}
            n = len(re.findall(r"^\d+ \| ", user, flags=re.M))
            num = iter(range(1, n + 1))
            def beat(audio, narration=""):
                return {"idea": f"{audio} beat", "audio": audio, "narration": narration,
                        "moments": [next(num, 1), next(num, 1)]}
            return {
                "title_hi": "विराट का जवाब",
                "theme": "epic comeback",
                "acts": [
                    {"key": "opening", "title_hi": "शुरुआत", "beats": [
                        beat("music"), beat("narration", "एक ऐसी कहानी जो हर भारतीय को गर्व से भर देगी।")]},
                    {"key": "buildup", "title_hi": "दबाव", "beats": [
                        beat("narration", "दो पारियाँ, दो नाकामियाँ। सवाल उठने लगे थे।"),
                        beat("original"), beat("original"), beat("narration", "लेकिन विराट चुप रहे।")]},
                    {"key": "rising", "title_hi": "तूफ़ान", "beats": [beat("music"), beat("original"),
                                                                    beat("original")]},
                    {"key": "climax", "title_hi": "शतक", "beats": [beat("music"), beat("original"),
                                                                 beat("music")]},
                    {"key": "ending", "title_hi": "जीत", "beats": [
                        beat("narration", "और इस तरह, एक बार फिर, विराट ने इतिहास लिख दिया।"),
                        beat("music")]},
                ],
            }
        if "YouTube metadata" in system:
            return {"youtube_title": "विराट कोहली का सबसे बड़ा जवाब | Virat Kohli Century",
                    "description_hi": "विराट की वापसी की पूरी कहानी।",
                    "tags": ["virat kohli", "विराट कोहली"]}
        return {}


@pytest.fixture
def fake_llm():
    return FakeLLM()
