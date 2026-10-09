"""Emotional Hindi narration with AI4Bharat's Indic Parler-TTS (free, offline, Apache-2.0).

Runs inside its own environment (.venv-voice, created by install-emotional-voice.command), so
the big AI libraries never touch the main app. StoryMaker calls it once per video with all
narration lines; the model loads once and writes <id>.raw.wav for every line.

    python voice_parler.py job.json                 # used by StoryMaker
    python voice_parler.py --sample out.wav "text"  # try the voice yourself
"""

import json
import os
import re
import sys
from pathlib import Path

MODEL_ID = "ai4bharat/indic-parler-tts"
DEFAULT_DESCRIPTION = (
    "Rohit speaks in an excited, energetic and highly expressive tone, like a passionate "
    "YouTube presenter, at a moderately fast pace, with a very clear, close-sounding recording "
    "and no background noise.")


def split_sentences(text, max_words=22):
    """Parler sounds best on short chunks: split on sentence ends, then on commas."""
    parts = [p.strip() for p in re.split(r"(?<=[।!?.])\s+", text) if p.strip()]
    out = []
    for p in parts:
        words = p.split()
        while len(words) > max_words:
            cut = max((i for i, w in enumerate(words[:max_words]) if w.endswith(",")), default=max_words - 1)
            out.append(" ".join(words[:cut + 1]))
            words = words[cut + 1:]
        if words:
            out.append(" ".join(words))
    return out or [text]


class Voice:
    def __init__(self, description=DEFAULT_DESCRIPTION, model_id=MODEL_ID):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Loading the Hindi voice model on {self.device}...", flush=True)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sampling_rate = self.model.config.sampling_rate
        self.set_description(description)

    def set_description(self, description):
        self.desc = self.desc_tokenizer(description, return_tensors="pt").to(self.device)

    def _generate(self, text):
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen = self.model.generate(input_ids=self.desc.input_ids, attention_mask=self.desc.attention_mask,
                                  prompt_input_ids=prompt.input_ids,
                                  prompt_attention_mask=prompt.attention_mask)
        return gen.cpu().float().numpy().squeeze()

    def say(self, text):
        import numpy as np
        pause = np.zeros(int(0.12 * self.sampling_rate), dtype="float32")
        chunks = []
        for part in split_sentences(text):
            try:
                audio = self._generate(part)
            except RuntimeError:
                if self.device == "cpu":
                    raise
                print("  GPU path failed, continuing on CPU", flush=True)
                self.device = "cpu"
                self.model = self.model.to("cpu")
                self.desc = self.desc.to("cpu")
                audio = self._generate(part)
            chunks += [audio.astype("float32"), pause]
        return np.concatenate(chunks)


def run_job(job_path):
    import soundfile as sf
    job = json.loads(Path(job_path).read_text())
    out_dir = Path(job["out_dir"])
    voice = Voice(job.get("description") or DEFAULT_DESCRIPTION, job.get("model") or MODEL_ID)
    lines = job["lines"]
    for i, (nid, text) in enumerate(lines.items(), 1):
        target = out_dir / f"{nid}.raw.wav"
        if target.exists():
            continue
        audio = voice.say(text)
        sf.write(str(target), audio, voice.sampling_rate)
        print(f"PROGRESS {i}/{len(lines)} {nid}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--sample":
        import soundfile as sf
        v = Voice(sys.argv[4] if len(sys.argv) > 4 else DEFAULT_DESCRIPTION)
        sf.write(sys.argv[2], v.say(sys.argv[3]), v.sampling_rate)
        print(f"Wrote {sys.argv[2]}")
    elif len(sys.argv) == 2:
        run_job(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(2)
