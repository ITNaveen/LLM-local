#!/bin/bash
ROOT="${LOCALLLM_ROOT:-$HOME/Documents}"
[ -z "$LOCALLLM_ROOT" ] && [ -f "$HOME/Documents/LLM/local-llm-app/app.py" ] && ROOT="$HOME/Documents/LLM"
export LOCALLLM_ROOT="$ROOT"
APP_DIR="$ROOT/local-llm-app"
LOG="$ROOT/local-llm-db/server.log"
PID_F="$ROOT/local-llm-db/server.pid"
CAFF_F="$ROOT/local-llm-db/caff.pid"

echo ""
echo "╔══════════════════════════════════╗"
echo "║    LocalLLM — Starting...        ║"
echo "╚══════════════════════════════════╝"
echo ""

# Already running?
if [ -f "$PID_F" ] && kill -0 $(cat "$PID_F") 2>/dev/null; then
  echo "Already running. Open http://localhost:8080"
  exit 0
fi

# Warn if no charger
if ! pmset -g ps | grep -q "AC Power"; then
  echo "WARNING: No charger connected."
  echo "Mac may sleep when lid is closed. Connect charger for best results."
  echo ""
  read -p "Continue anyway? (y/N): " c
  [[ "$c" != "y" && "$c" != "Y" ]] && exit 1
fi

# Prevent Mac sleeping with lid closed
caffeinate -s &
echo $! > "$CAFF_F"
echo "✓ Sleep prevention ON (Mac stays awake with lid closed)"

# Start LocalLLM
cd "$APP_DIR"
nohup python3.11 app.py >> "$LOG" 2>&1 &
echo $! > "$PID_F"
sleep 2

if kill -0 $(cat "$PID_F") 2>/dev/null; then
  IP=$(ipconfig getifaddr en0 2>/dev/null || echo "192.168.x.x")
  echo "✓ LocalLLM is running"
  echo ""
  echo "  Local  → http://localhost:8080"
  echo "  Phone  → http://$IP:8080"
  echo ""
  echo "  You can now close the lid (keep charger connected)."
  echo "  Run stop-llm.sh when done."
else
  echo "✗ Failed to start. Check log: $LOG"
  kill $(cat "$CAFF_F") 2>/dev/null
fi
