#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Live Translator - one-time setup on the Mac
#
#    bash ~/Downloads/live-translator/setup-mac.sh
#
#  1. moves the app to ~/Documents/LiveTranslator  (keeps an existing install there)
#  2. installs it (Python packages + models) if that hasn't been done yet
#  3. puts "Live Translator" on the Desktop: double-click = START, again = STOP
#
#  Safe to run again (e.g. for an update): your meetings in ~/LiveTranslator are never touched.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${LT_APP_DIR:-$HOME/Documents/LiveTranslator}"
DESKTOP="${LT_DESKTOP:-$HOME/Desktop}"
LAUNCHER="$DESKTOP/Live Translator.command"

echo ""
echo "══════════════════════════════════════════"
echo "  Live Translator - setup"
echo "══════════════════════════════════════════"
echo ""

# ---------------------------------------------------------------- 1. move to Documents
if [ "$SRC" != "$DEST" ]; then
  # a copy that is already running must be stopped before it is moved
  if [ -x "$SRC/toggle.sh" ] && "$SRC/toggle.sh" status >/dev/null 2>&1; then
    echo "• Stopping the running copy first…"
    "$SRC/toggle.sh" stop >/dev/null 2>&1 || true
  fi
  mkdir -p "$DEST"
  # copy everything (an existing .venv / models in DEST are kept unless SRC brings its own)
  cp -R "$SRC/." "$DEST/"
  echo "✓ App copied to $DEST"
  # a virtual environment that was created in Downloads still works after the move;
  # if not, it is rebuilt below
  if [ -x "$DEST/.venv/bin/python" ] && ! (cd "$DEST" && ./.venv/bin/python -c "import livetranslator.server") >/dev/null 2>&1; then
    echo "• Rebuilding the Python environment for the new location…"
    rm -rf "$DEST/.venv"
  fi
  # remove the Downloads copy (only if it really is in Downloads)
  case "$SRC" in
    "$HOME/Downloads/"*) rm -rf "$SRC"; echo "✓ Removed the copy in Downloads" ;;
    *) echo "  (left the original folder in place: $SRC)" ;;
  esac
else
  echo "✓ Already in $DEST"
  if [ -x "$DEST/toggle.sh" ] && "$DEST/toggle.sh" status >/dev/null 2>&1; then
    echo "• Stopping the running app for the update…"
    "$DEST/toggle.sh" stop >/dev/null 2>&1 || true
  fi
fi

cd "$DEST"
if [ "$(uname -s)" = "Darwin" ]; then
  xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
fi
chmod +x install.sh start.sh toggle.sh "Live Translator.command" setup-mac.sh tools/*.sh 2>/dev/null || true

# ---------------------------------------------------------------- 2. install (once)
if [ -x ".venv/bin/python" ] && [ -f ".venv/.installed-ok" ]; then
  echo "✓ Already installed (models downloaded)"
  # an update may bring new Python packages
  want="$(shasum requirements.txt 2>/dev/null | cut -d' ' -f1)"
  if [ -n "$want" ] && [ "$want" != "$(cat .venv/.installed-ok 2>/dev/null)" ]; then
    echo "• Updating Python packages…"
    if ./.venv/bin/python -m pip install -q -r requirements.txt; then
      echo "$want" > .venv/.installed-ok
      echo "✓ Packages up to date"
    else
      echo "⚠ Package update failed - run ./install.sh in $DEST"
    fi
  fi
  # models added by an update (e.g. the fast fallback translation model) - already-present ones are skipped
  echo "• Checking models…"
  ./.venv/bin/python -m livetranslator download || echo "⚠ Model download incomplete - it is retried next time"
else
  echo "• Installing (one time, ~15 min - mostly downloading the models)…"
  ./install.sh
fi

# ---------------------------------------------------------------- 3. Desktop start/stop icon
mkdir -p "$DESKTOP"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Live Translator - double-click to START, double-click again to STOP.
# (The app lives in: $DEST)
exec "$DEST/toggle.sh"
EOF
chmod +x "$LAUNCHER"
echo "✓ Desktop icon created: $LAUNCHER"

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ All set"
echo ""
echo "  Desktop → 'Live Translator'"
echo "    double-click        = START"
echo "    double-click again  = STOP"
echo "══════════════════════════════════════════"
echo ""
echo "  App:      $DEST"
echo "  Meetings: $HOME/LiveTranslator/Meetings"
echo "  First start: macOS asks if Terminal may use the microphone → Allow."
echo ""
