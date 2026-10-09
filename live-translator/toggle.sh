#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Live Translator - START / STOP toggle
#
#  Double-click (Desktop icon)  → starts it if it is off, stops it if it is on.
#  toggle.sh start | stop | status   for scripts.
#  toggle.sh mictest                 which microphones really hear sound
#
#  START: runs the app in the background, opens it in the browser.
#  STOP : finishes and saves the meeting in progress, stops the app and
#         unloads the translation model from Ollama (frees ~8 GB of memory).
#         Ollama itself keeps running (LocalLLM uses it too).
# ═══════════════════════════════════════════════════════════════════════════
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${LT_HOME:-$HOME/LiveTranslator}"
PORT="${LT_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
PID_F="$DATA/server.pid"
LOG="$DATA/logs/app.log"
VPY="$APP_DIR/.venv/bin/python"
mkdir -p "$DATA/logs"

info()   { curl -s -m 2 "$URL/api/info" 2>/dev/null; }
is_ours(){ info | grep -q '"app":"Live Translator"'; }

server_pid() {
  local pid=""
  [ -f "$PID_F" ] && pid="$(cat "$PID_F" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= 2>/dev/null | grep -q livetranslator; then
    echo "$pid"; return
  fi
  # started some other way (e.g. start.sh): ask the app itself
  info | grep -o '"pid":[0-9]*' | head -1 | cut -d: -f2
}

open_browser() {
  if command -v open >/dev/null 2>&1; then open "$URL/"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL/" >/dev/null 2>&1
  fi
}

banner() {
  echo ""
  echo "══════════════════════════════════════════"
  echo "  $1"
  echo "══════════════════════════════════════════"
  echo ""
}

do_start() {
  if [ ! -x "$VPY" ] || [ ! -f "$APP_DIR/.venv/.installed-ok" ]; then
    echo "First start - installing Live Translator (one time)…"
    "$APP_DIR/install.sh" --no-test || { echo "✗ Installation failed - see above."; exit 1; }
  fi
  banner "Live Translator - Starting…"

  # Ollama (translation) - the app also retries on its own
  if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    local launched=0
    if [ -d "/Applications/Ollama.app" ]; then open -g -a Ollama; launched=1
    elif command -v ollama >/dev/null 2>&1; then nohup ollama serve >> "$DATA/logs/ollama.log" 2>&1 & launched=1
    fi
    if [ "$launched" = 1 ]; then
      for _ in $(seq 1 20); do curl -s -m 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 1; done
    fi
  fi
  if curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then echo "✓ Ollama (translation) running"
  else echo "⚠ Ollama is not running - German will be shown, English needs Ollama"; fi

  # keep the log small
  if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 5000000 ]; then mv -f "$LOG" "$LOG.old"; fi
  echo "" >> "$LOG"; echo "=== $(date) start ===" >> "$LOG"
  cd "$APP_DIR" || exit 1
  nohup "$VPY" -m livetranslator serve --port "$PORT" >> "$LOG" 2>&1 &
  local pid=$!
  disown 2>/dev/null || true
  echo "$pid" > "$PID_F"

  for _ in $(seq 1 60); do
    is_ours && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "✗ Live Translator could not start. Last lines of $LOG:"
      tail -n 25 "$LOG"
      rm -f "$PID_F"
      exit 1
    fi
    sleep 0.5
  done
  if ! is_ours; then echo "✗ No answer from the app yet - check $LOG"; exit 1; fi
  echo "✓ App running (speech model loads in the background - watch the 'Speech' dot turn green)"
  open_browser
  banner "✅ Live Translator is ON"
  echo "  Open:     $URL"
  echo "  Meetings: $DATA/Meetings"
  echo ""
  echo "  ▶ Double-click the Desktop icon again to STOP."
  echo "  (You can close this window - the app keeps running.)"
  echo ""
}

do_stop() {
  local pid
  pid="$(server_pid)"
  if [ -z "$pid" ] && ! is_ours; then
    echo "Live Translator is not running."
    rm -f "$PID_F"
    return 0
  fi
  banner "Live Translator - Stopping…"
  if is_ours; then
    if info | grep -q '"state":"listening"'; then
      echo "• Saving the meeting in progress…"
    fi
    curl -s -m 5 -X POST "$URL/api/stop" >/dev/null 2>&1
    for _ in $(seq 1 120); do
      info | grep -q '"state":"idle"' && break
      is_ours || break
      sleep 0.5
    done
    echo "✓ Meeting saved"
  fi
  if [ -n "$pid" ]; then
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$PID_F"
  echo "✓ App stopped, translation model unloaded (memory freed)"
  banner "⏹  Live Translator is OFF"
  echo "  Your meetings: $DATA/Meetings"
  echo "  ▶ Double-click the Desktop icon to START again."
  echo ""
}

running() { [ -n "$(server_pid)" ] || is_ours; }

case "${1:-toggle}" in
  start)  if running; then echo "Already running - opening it."; open_browser; else do_start; fi ;;
  stop)   do_stop ;;
  status) if running; then echo "running (pid $(server_pid)) - $URL"; else echo "stopped"; exit 1; fi ;;
  mictest) cd "$APP_DIR" && exec "$VPY" -m livetranslator mictest ;;
  toggle) if running; then do_stop; else do_start; fi ;;
  *) echo "usage: $0 [start|stop|status|mictest]"; exit 2 ;;
esac
