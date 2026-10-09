#!/bin/bash
# One-time install of the emotional Hindi voice: AI4Bharat Indic Parler-TTS (free, offline).
# Downloads about 5 GB (AI libraries + the voice model). Safe to run again.
cd "$(dirname "$0")"
MODEL_PAGE="https://huggingface.co/ai4bharat/indic-parler-tts"
TOKEN_PAGE="https://huggingface.co/settings/tokens/new?tokenType=read&description=StoryMaker"

PY=""
for p in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))'; then
    PY="$p"; break
  fi
done
if [ -z "$PY" ]; then echo "Python 3.10-3.13 is needed: brew install python@3.12"; exit 1; fi

echo "1/4  Creating a separate environment for the voice (keeps the main app untouched)..."
[ -d .venv-voice ] || "$PY" -m venv .venv-voice
.venv-voice/bin/pip install -q --upgrade pip

echo "2/4  Installing PyTorch + Parler-TTS (a few minutes)..."
if ! .venv-voice/bin/pip install -q torch "parler-tts==0.2.3" soundfile; then
  echo "Install failed. Paste the error above to Claude."; exit 1
fi

can_access() {
  .venv-voice/bin/python - <<'PY' >/dev/null 2>&1
from huggingface_hub import hf_hub_download
hf_hub_download("ai4bharat/indic-parler-tts", "config.json")
PY
}

echo "3/4  Checking access to the voice model..."
tries=0
until can_access; do
  tries=$((tries + 1))
  if [ $tries -gt 3 ]; then
    echo "Still no access. Check that you clicked 'Agree' on $MODEL_PAGE with the same account"
    echo "as the token, then run me again."
    exit 1
  fi
  echo ""
  echo "  The voice model is free, but its makers (AI4Bharat) ask everyone to accept their licence once."
  echo "  A) Your browser opens the model page. Log in or sign up (free), then click"
  echo "     'Agree and access repository'."
  open "$MODEL_PAGE" 2>/dev/null || echo "     Open: $MODEL_PAGE"
  read -r -p "     Press Enter here when you have clicked Agree... " _
  echo "  B) Now a page opens to create a free access token. Click 'Create token', then copy it."
  open "$TOKEN_PAGE" 2>/dev/null || echo "     Open: $TOKEN_PAGE"
  read -r -s -p "     Paste the token here (it stays hidden) and press Enter: " TOKEN
  echo ""
  HF_TOKEN_INPUT="$TOKEN" .venv-voice/bin/python - <<'PY' || echo "     That token didn't work - let's try again."
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN_INPUT"].strip(), add_to_git_credential=False)
PY
done
echo "     Access OK."

echo "4/4  Downloading the Hindi voice model (about 3.5 GB) and recording a test line..."
if .venv-voice/bin/python storymaker/voice_parler.py --sample voice-test.wav \
   "दिल्ली पुलिस ने साफ़ कह दिया, कोई इजाज़त नहीं! अब सवाल ये है... कल क्या होगा?" \
   "Rohit speaks in an angry, intense and powerful tone, like a passionate Hindi YouTube news presenter, at a fast pace, with a very clear, close-sounding recording and no background noise."; then
  touch .venv-voice/READY
  echo ""
  echo "Done! Playing the test line..."
  afplay voice-test.wav 2>/dev/null || true
  echo "StoryMaker now uses this voice automatically. Restart StoryMaker if it is running."
else
  echo ""
  echo "The voice model could not be loaded. Paste the error above to Claude."
  exit 1
fi
