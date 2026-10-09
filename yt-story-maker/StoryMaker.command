#!/bin/bash
# Double-click on your Mac to start StoryMaker. First run installs everything it needs.
cd "$(dirname "$0")"

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
if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q " ass " && command -v brew >/dev/null 2>&1 \
   && [ ! -f "$HOME/.storymaker-ffmpeg-full-tried" ]; then
  echo "Installing the full ffmpeg so videos get Hindi text cards and subtitles (one time, a few minutes)..."
  touch "$HOME/.storymaker-ffmpeg-full-tried"
  if brew install ffmpeg-full; then
    FULL="$(brew --prefix ffmpeg-full 2>/dev/null)/bin"
    [ -x "$FULL/ffmpeg" ] && export PATH="$FULL:$PATH"
  fi
fi
if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q " ass "; then
  echo "NOTE: ffmpeg cannot draw Hindi text here; on-screen lines will be spoken by the narrator."
  echo "      For text cards and burned subtitles run:  brew install ffmpeg-full"
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
  echo "First run: creating the Python environment (one time)..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Install packages only when requirements.txt changed (fast restarts, no silent waiting).
REQ_HASH=$(shasum requirements.txt | cut -d' ' -f1)
if [ "$(cat .venv/.req-hash 2>/dev/null)" != "$REQ_HASH" ]; then
  echo "Installing Python packages (a minute or two)..."
  pip install --timeout 30 -r requirements.txt && echo "$REQ_HASH" > .venv/.req-hash
fi
# YouTube changes often: refresh yt-dlp at most once a day, never block the start.
if [ -z "$(find .venv/.ytdlp-checked -mtime -1 2>/dev/null)" ]; then
  echo "Checking for a newer yt-dlp (max 60 s)..."
  ( pip install -q --timeout 20 --upgrade "yt-dlp[default]" >/dev/null 2>&1 & PID=$!
    ( sleep 60; kill $PID 2>/dev/null ) & wait $PID ) || true
  touch .venv/.ytdlp-checked
fi

# Start Ollama if it's installed but not running.
if command -v ollama >/dev/null 2>&1 && ! curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Starting Ollama..."
  (ollama serve >/dev/null 2>&1 &)
  sleep 3
fi

PORT=$(python -c "from storymaker.config import load_settings; print(load_settings()['port'])")
echo "Starting StoryMaker at http://localhost:$PORT  (keep this window open; Ctrl+C stops it)"
(sleep 2; open "http://localhost:$PORT") &
# Keep the Mac awake while StoryMaker runs (screen may turn off; jobs keep going).
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i python -m storymaker
else
  python -m storymaker
fi
