#!/usr/bin/env bash
# Start Live Translator and open it in the browser.
#   ./start.sh                         local use (http://127.0.0.1:8765)
#   ./start.sh --host 0.0.0.0 --token SECRET [--ssl-certfile cert.pem --ssl-keyfile key.pem]
#                                      reachable from other computers (see README)
cd "$(dirname "$0")"
VPY="./.venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "Not installed yet - running ./install.sh first."
  ./install.sh || exit 1
fi
PORT="${LT_PORT:-8765}"
prev=""
for a in "$@"; do [ "$prev" = "--port" ] && PORT="$a"; prev="$a"; done
if curl -s -m 1 "http://127.0.0.1:$PORT/api/info" >/dev/null 2>&1; then
  echo "Live Translator is already running - opening it."
  (command -v open >/dev/null && open "http://127.0.0.1:$PORT/") || (command -v xdg-open >/dev/null && xdg-open "http://127.0.0.1:$PORT/")
  exit 0
fi
# start Ollama if needed (the app also tries)
if ! curl -s -m 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  if [ -d "/Applications/Ollama.app" ]; then open -g -a Ollama
  elif command -v ollama >/dev/null 2>&1; then (nohup ollama serve >/dev/null 2>&1 &)
  fi
fi
echo "Starting Live Translator… (keep this window open; close it or press Ctrl+C to quit)"
exec "$VPY" -m livetranslator serve --open "$@"
