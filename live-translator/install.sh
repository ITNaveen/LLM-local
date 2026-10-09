#!/usr/bin/env bash
# Live Translator - one-time setup (macOS / Linux).
#   ./install.sh            full setup (Python env, speech model, Ollama + translation model)
#   ./install.sh --no-test  skip the self-test at the end
set -euo pipefail
cd "$(dirname "$0")"
APP_DIR="$(pwd)"
RUN_TEST=1
for a in "$@"; do [ "$a" = "--no-test" ] && RUN_TEST=0; done

bold() { printf "\n\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }

bold "Live Translator - setup"
OS="$(uname -s)"; ARCH="$(uname -m)"
if [ "$OS" = "Darwin" ]; then
  # downloaded files are quarantined by macOS; allow double-clicking the launcher
  xattr -dr com.apple.quarantine "$APP_DIR" 2>/dev/null || true
fi
echo "  System: $OS $ARCH"
if [ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ]; then
  warn "Intel Mac detected - speech recognition will run on the CPU (slower)."
fi

# ---------------------------------------------------------------- Python
bold "1/4  Python"
PY=""
for p in python3.12 python3.13 python3.11 python3.10 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
         /opt/homebrew/bin/python3.11 /usr/local/bin/python3.12 python3; do
  if command -v "$p" >/dev/null 2>&1; then
    v=$("$p" -c 'import sys; print("%d%02d" % sys.version_info[:2])' 2>/dev/null || echo 0)
    if [ "$v" -ge 310 ] && [ "$v" -le 313 ]; then PY="$p"; break; fi
  fi
done
if [ -z "$PY" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "  Installing Python 3.12 with Homebrew…"
    brew install python@3.12
    PY="$(brew --prefix)/bin/python3.12"
  else
    echo "  No suitable Python (3.10-3.13) found - installing it with 'uv' (a small Python installer)…"
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    "$UV" python install 3.12
    PY="$("$UV" python find 3.12)"
  fi
fi
ok "using $PY ($("$PY" --version))"

if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv
fi
VPY="$APP_DIR/.venv/bin/python"
"$VPY" -m pip install --upgrade pip wheel >/dev/null
echo "  Installing packages (a few minutes the first time)…"
"$VPY" -m pip install -r requirements.txt
ok "Python packages installed"

# ---------------------------------------------------------------- Ollama
bold "2/4  Ollama (runs the translation model locally)"
have_ollama() { command -v ollama >/dev/null 2>&1 || [ -d "/Applications/Ollama.app" ]; }
if ! have_ollama; then
  if command -v brew >/dev/null 2>&1; then
    echo "  Installing Ollama with Homebrew…"
    brew install ollama
  elif [ "$OS" = "Linux" ]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    warn "Ollama is not installed. Download it from https://ollama.com/download , open it once, then run ./install.sh again."
  fi
fi
if have_ollama; then
  if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null; then
    if [ -d "/Applications/Ollama.app" ]; then open -g -a Ollama; else (nohup ollama serve >/dev/null 2>&1 &); fi
    for _ in $(seq 1 30); do curl -s -m 1 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
  fi
  curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null && ok "Ollama is running" || warn "Ollama did not start"
fi

# ---------------------------------------------------------------- models
bold "3/4  Models (speech ≈1.6 GB, translation ≈3 GB - downloaded once)"
"$VPY" -m livetranslator download || warn "Model download incomplete - run ./install.sh again or download from Settings in the app."

chmod +x start.sh "Live Translator.command" 2>/dev/null || true

# ---------------------------------------------------------------- self-test
if [ "$RUN_TEST" = "1" ]; then
  bold "4/4  Self-test (about 5 minutes: measures accuracy and speed on this Mac)"
  "$VPY" -m livetranslator selftest || warn "Self-test reported a problem - see above."
fi

# remember which requirements are installed (setup-mac.sh re-installs when they change)
shasum requirements.txt 2>/dev/null | cut -d' ' -f1 > "$APP_DIR/.venv/.installed-ok" || touch "$APP_DIR/.venv/.installed-ok"

bold "Done."
echo "  Start / stop:   double-click 'Live Translator' on the Desktop (created by setup-mac.sh)"
echo "                  or 'Live Translator.command' in this folder"
echo "  The first time, macOS asks whether Terminal may use the microphone - click Allow."
echo "  Meetings are saved in: ~/LiveTranslator/Meetings"
