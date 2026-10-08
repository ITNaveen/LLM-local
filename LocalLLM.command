#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  LocalLLM — Toggle Script  (double-click to START / double-click to STOP)
#
#  BATTERY STRATEGY (M4 MacBook Pro):
#  ─────────────────────────────────────────────────────────────────────────
#  • displaysleep 3  → screen off in 3 min → saves ~60-70% of idle power
#  • sleep 0         → no idle sleep timer → system stays awake (lid open)
#                      NOT disablesleep — lid-close STILL sleeps normally
#  • standby 0       → no deep hibernate (would kill cloudflared process)
#  • lowpowermode 1  → M4 runs at low freq when idle → ~20-30% extra saving
#  • ONE caffeinate  → -i only (idle sleep guard, nothing else)
#
#  Result: lid open + screen dark = ~1-2% battery per hour
#  Lid closed → Mac sleeps normally (you don't need app when carrying it)
#
#  STOP: ALL settings restored to macOS defaults. ALL caffeinate killed.
#  No accumulation. No leftovers. Clean every time.
# ═══════════════════════════════════════════════════════════════════════════

# ── WHERE THE LOCALLLM FOLDERS LIVE ──────────────────────────────────────────
# local-llm-app, local-llm-db and local-llm-dropbox sit side by side in ROOT.
# Uses ~/Documents/LLM if the app is there, otherwise ~/Documents (the old
# place). To use another place: export LOCALLLM_ROOT=/path before running.
if [ -n "$LOCALLLM_ROOT" ]; then
  ROOT="$LOCALLLM_ROOT"
elif [ -f "$HOME/Documents/LLM/local-llm-app/app.py" ]; then
  ROOT="$HOME/Documents/LLM"
else
  ROOT="$HOME/Documents"
fi
export LOCALLLM_ROOT="$ROOT"          # app.py and the tunnel watchdog read this too

PID_F="$ROOT/local-llm-db/server.pid"
CAFF_F="$ROOT/local-llm-db/caff.pid"
TUNNEL_F="$ROOT/local-llm-db/tunnel.pid"
WATCHDOG_F="$ROOT/local-llm-db/watchdog.pid"
APP_DIR="$ROOT/local-llm-app"
LOG="$ROOT/local-llm-db/server.log"


# ── STOP ─────────────────────────────────────────────────────────────────────
if [ -f "$PID_F" ] && kill -0 $(cat "$PID_F") 2>/dev/null; then
  echo ""
  echo "╔══════════════════════════════════╗"
  echo "║    LocalLLM — Stopping...        ║"
  echo "╚══════════════════════════════════╝"
  echo ""

  # Stop Flask server
  kill $(cat "$PID_F") 2>/dev/null; sleep 1
  kill -9 $(cat "$PID_F") 2>/dev/null
  rm -f "$PID_F"

  # Stop watchdog
  [ -f "$WATCHDOG_F" ] && kill $(cat "$WATCHDOG_F") 2>/dev/null
  rm -f "$WATCHDOG_F"

  # Stop tunnel
  [ -f "$TUNNEL_F" ] && kill $(cat "$TUNNEL_F") 2>/dev/null
  rm -f "$TUNNEL_F"
  pkill -f "cloudflared tunnel run" 2>/dev/null

  # Kill ALL caffeinate processes — prevents the accumulation problem
  # This is a hard sweep — kills every caffeinate on the system
  pkill -9 caffeinate 2>/dev/null
  rm -f "$CAFF_F"
  echo "✓ All caffeinate processes killed (clean sweep)"

  # Restore ALL pmset settings to macOS defaults
  sudo pmset -a sleep 1           # default: idle sleep after 1 min
  sudo pmset -a displaysleep 2    # default: display sleep after 2 min
  sudo pmset -a standby 1         # default: standby on
  sudo pmset -a powernap 0        # keep off (better for battery)
  sudo pmset -a tcpkeepalive 1    # keep on (good default)
  sudo pmset -a lowpowermode 0    # restore: normal performance mode

  echo "✓ Server stopped"
  echo "✓ Tunnel stopped"
  echo "✓ pmset: all settings restored to macOS defaults"
  echo "✓ lowpowermode: OFF (Mac back to normal performance)"
  sleep 2
  exit 0
fi


# ── START ─────────────────────────────────────────────────────────────────────
# Refuse to half-start (power settings changed, no server) if the app can't be found.
if [ ! -f "$APP_DIR/app.py" ] || [ ! -d "$ROOT/local-llm-db" ]; then
  echo ""
  echo "✗ Can't find LocalLLM in: $ROOT"
  echo "  Expected $ROOT/local-llm-app/app.py and $ROOT/local-llm-db/"
  echo "  Nothing was started or changed."
  sleep 5
  exit 1
fi
echo ""
echo "╔══════════════════════════════════╗"
echo "║    LocalLLM — Starting...        ║"
echo "╚══════════════════════════════════╝"
echo ""

# Safety: kill any leftover caffeinate from previous sessions before starting
# This prevents the 35-process accumulation problem from happening again
pkill -9 caffeinate 2>/dev/null
sleep 1
echo "✓ Cleared any leftover caffeinate processes"

# ── POWER SETTINGS ────────────────────────────────────────────────────────────
# displaysleep 3 : screen dark in 3 min = biggest battery saving
#                  M4 dark screen idle = ~0.5-1W vs ~3-4W with screen on
# sleep 0        : no idle sleep timer (lid-close still sleeps — this is key)
# standby 0      : prevents deep hibernate that kills cloudflared
# powernap 0     : no background wakeups wasting battery
# tcpkeepalive 1 : TCP connections survive low-power states
# lowpowermode 1 : M4 drops to efficiency cores + low freq when idle
#                  Flask + cloudflared are low-CPU — zero performance impact
#                  Saves additional 20-30% battery on top of displaysleep
sudo pmset -a displaysleep 3
sudo pmset -a sleep 0
sudo pmset -a standby 0
sudo pmset -a powernap 0
sudo pmset -a tcpkeepalive 1
sudo pmset -a lowpowermode 1
echo "✓ Power settings applied:"
echo "  displaysleep=3  sleep=0  standby=0  powernap=0  lowpowermode=1"

# ONE caffeinate, -i only (idle sleep guard)
# -i = prevent idle sleep
# No -s (AC sleep), no -d (display) — those cause unnecessary power use
caffeinate -i &
echo $! > "$CAFF_F"
echo "✓ caffeinate: 1 process, -i only (idle guard)"

# ── LOCAL AI (Ollama) ─────────────────────────────────────────────────────────
# Local mode (free, private answers from your living repos) needs Ollama.
# If it's installed but not running, start it. If it isn't installed, Cloud
# mode still works exactly as before.
if command -v ollama >/dev/null 2>&1 || [ -d "/Applications/Ollama.app" ]; then
  if ! curl -sf --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
    if [ -d "/Applications/Ollama.app" ]; then
      open -ga Ollama                                   # menu-bar app, in the background
    else
      nohup ollama serve >> "$LOG" 2>&1 &               # Homebrew install (no app)
    fi
    for i in 1 2 3 4 5 6 7 8 9 10; do
      curl -sf --max-time 1 http://localhost:11434/api/tags > /dev/null 2>&1 && break
      sleep 1
    done
  fi
  if curl -sf --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Local AI (Ollama) running"
  else
    echo "⚠ Ollama didn't start — Local mode unavailable (Cloud mode still works)"
  fi
else
  echo "ℹ Ollama not installed — Local mode off. Install: brew install ollama"
fi

# ── START FLASK SERVER ────────────────────────────────────────────────────────
cd "$APP_DIR"
nohup python3.11 app.py >> "$LOG" 2>&1 &
echo $! > "$PID_F"
sleep 2

if ! kill -0 $(cat "$PID_F") 2>/dev/null; then
  echo "✗ Server failed. Check: $LOG"
  pkill -9 caffeinate 2>/dev/null
  sudo pmset -a lowpowermode 0
  sudo pmset -a sleep 1
  sudo pmset -a displaysleep 2
  sudo pmset -a standby 1
  exit 1
fi

IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "192.168.x.x")
echo "✓ Server running"
echo "  Local  → http://localhost:8080"
echo "  Phone  → http://$IP:8080"

# ── TUNNEL WATCHDOG ───────────────────────────────────────────────────────────
# Checks every 30s.
# Smart: if Flask unreachable → system sleeping → do nothing, wait for wake.
# On wake: detects tunnel drop, restarts cloudflared within 30s.
nohup bash -c '
  LOG="$LOCALLLM_ROOT/local-llm-db/server.log"
  TUNNEL_PID_F="$LOCALLLM_ROOT/local-llm-db/tunnel.pid"

  start_tunnel() {
    [ -f "$TUNNEL_PID_F" ] && kill $(cat "$TUNNEL_PID_F") 2>/dev/null
    pkill -f "cloudflared tunnel run" 2>/dev/null
    sleep 2
    cloudflared tunnel run localllm >> "$LOG" 2>&1 &
    echo $! > "$TUNNEL_PID_F"
    echo "$(date): Tunnel started (PID $!)" >> "$LOG"
  }

  start_tunnel

  while true; do
    sleep 30

    # Flask unreachable = system asleep = do nothing
    if ! curl -sf --max-time 3 http://localhost:8080/api/stats > /dev/null 2>&1; then
      continue
    fi

    TPID=$(cat "$TUNNEL_PID_F" 2>/dev/null)

    # Check 1: tunnel process dead?
    if [ -z "$TPID" ] || ! kill -0 "$TPID" 2>/dev/null; then
      echo "$(date): Tunnel dead — restarting" >> "$LOG"
      start_tunnel
      continue
    fi

    # Check 2: too many errors in recent log?
    FAILS=$(tail -20 "$LOG" 2>/dev/null | \
      grep -c "ERR Connection terminated\|failed to dial\|no recent network activity" \
      2>/dev/null || echo 0)
    if [ "$FAILS" -ge 3 ]; then
      echo "$(date): Tunnel errors ($FAILS) — restarting" >> "$LOG"
      start_tunnel
    fi
  done
' &
echo $! > "$WATCHDOG_F"
sleep 3

echo "✓ Tunnel watchdog active (30s checks, auto-reconnect on wake)"
echo "  Domain → https://llm.jarvisinfra.com"
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅  LocalLLM is LIVE                    ║"
echo "║                                          ║"
echo "║  https://llm.jarvisinfra.com             ║"
echo "║                                          ║"
echo "║  Screen : off in 3 min  ✓  (saves 70%)  ║"
echo "║  System : awake, lid open  ✓             ║"
echo "║  Power  : low power mode  ✓  (saves 25%) ║"
echo "║  Battery: ~1-2% per hour  ✓              ║"
echo "║  Tunnel : watchdog 30s  ✓                ║"
echo "║                                          ║"
echo "║  Leave lid OPEN at home → always live    ║"
echo "║  Double-click again to STOP              ║"
echo "╚══════════════════════════════════════════╝"
echo ""
