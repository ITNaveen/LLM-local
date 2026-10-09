#!/bin/bash
# One-time install of the emotional Hindi voice: AI4Bharat Indic Parler-TTS (free, offline).
# Downloads about 5 GB (AI libraries + the voice model). Safe to run again.
cd "$(dirname "$0")"

PY=""
for p in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))'; then
    PY="$p"; break
  fi
done
if [ -z "$PY" ]; then echo "Python 3.10-3.13 is needed: brew install python@3.12"; exit 1; fi

echo "1/3  Creating a separate environment for the voice (keeps the main app untouched)..."
[ -d .venv-voice ] || "$PY" -m venv .venv-voice
.venv-voice/bin/pip install -q --upgrade pip

echo "2/3  Installing PyTorch + Parler-TTS (a few minutes)..."
if ! .venv-voice/bin/pip install torch "parler-tts==0.2.3" soundfile; then
  echo "Install failed. Paste the error above to Claude."; exit 1
fi

echo "3/3  Downloading the Hindi voice model (about 3.5 GB) and recording a test line..."
if .venv-voice/bin/python storymaker/voice_parler.py --sample voice-test.wav \
   "सोचिए ज़रा! लाखों भारतीय इंजीनियर... और एक झटके में सब कुछ बदल गया। ये है असली कहानी!"; then
  echo ""
  echo "Done! Playing the test line..."
  afplay voice-test.wav 2>/dev/null || true
  echo "StoryMaker will now use this voice automatically (Settings → Voice engine)."
else
  echo ""
  echo "The voice model could not be loaded. If the error says 'gated' or '401', run:"
  echo "   .venv-voice/bin/huggingface-cli login"
  echo "accept the model terms at https://huggingface.co/ai4bharat/indic-parler-tts and run me again."
fi
