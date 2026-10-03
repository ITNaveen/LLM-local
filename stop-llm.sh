#!/bin/bash
PID_F="$HOME/Documents/local-llm-db/server.pid"
CAFF_F="$HOME/Documents/local-llm-db/caff.pid"

echo ""
echo "╔══════════════════════════════════╗"
echo "║    LocalLLM — Stopping...        ║"
echo "╚══════════════════════════════════╝"
echo ""

# Stop server
if [ -f "$PID_F" ]; then
  PID=$(cat "$PID_F")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" && sleep 1
    kill -9 "$PID" 2>/dev/null
    echo "✓ Server stopped"
  else
    echo "  Server was not running"
  fi
  rm -f "$PID_F"
else
  # Fallback: kill by port
  lsof -ti:8080 | xargs kill -9 2>/dev/null && echo "✓ Server stopped (by port)"
fi

# Release caffeinate (Mac can sleep normally again)
if [ -f "$CAFF_F" ]; then
  kill $(cat "$CAFF_F") 2>/dev/null
  rm -f "$CAFF_F"
  echo "✓ Sleep prevention OFF (Mac can sleep normally)"
fi

echo ""
echo "LocalLLM is stopped."
