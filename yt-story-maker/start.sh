#!/bin/bash
# Double-click on your Mac to start StoryMaker. First run installs everything it needs.
cd "$(dirname "$0")"
set -e

need_brew() {
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is needed first. Install it from https://brew.sh and run me again."
    read -r -p "Press Enter to close." _
    exit 1
  fi
}

# ffmpeg with libass (Hindi titles/subtitles). Prefer Homebrew's 'ffmpeg-full' if present
# (it is keg-only, so put it first on PATH).
if command -v brew >/dev/null 2>&1; then
  FULL="$(brew --prefix ffmpeg-full 2>/dev/null)/bin"
  [ -x "$FULL/ffmpeg" ] && export PATH="$FULL:$PATH"
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  need_brew; echo "Installing ffmpeg..."; brew install ffmpeg
fi
if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q " ass "; then
  echo "NOTE: your ffmpeg cannot draw Hindi text (titles, burned subtitles, thumbnail text)."
  echo "      Fix once with:  brew install ffmpeg-full   then start StoryMaker again."
fi
# yt-dlp needs a JavaScript runtime for YouTube nowadays.
if ! command -v deno >/dev/null 2>&1 && ! command -v node >/dev/null 2>&1; then
  need_brew; echo "Installing deno (needed by yt-dlp for YouTube)..."; brew install deno
fi

# Python 3.10+
PY=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$p"; break
  fi
done
if [ -z "$PY" ]; then need_brew; brew install python@3.12; PY=python3.12; fi

if [ ! -d .venv ]; then
  echo "First run: setting up Python packages (one time)..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements.txt
pip install -q --upgrade "yt-dlp[default]"   # YouTube changes often; stay current

# Start Ollama if it's installed but not running.
if command -v ollama >/dev/null 2>&1 && ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
  (ollama serve >/dev/null 2>&1 &)
  sleep 3
fi

PORT=$(python -c "from storymaker.config import load_settings; print(load_settings()['port'])")
(sleep 2; (open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null || true)) &
python -m storymaker
