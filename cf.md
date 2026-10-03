brew install cloudflared

cloudflared tunnel login

The in cloud flare click on domain name and it will give success message 

cloudflared tunnel create localllm (in terminal)
Created tunnel localllm with id e685b45f-1241-4ba6-ab67-5f4fca76e087 (message in terminal)

# Create config
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: e685b45f-1241-4ba6-ab67-5f4fca76e087
ingress:
  - hostname: llm.jarvisinfra.com
    service: http://localhost:8080
  - service: http_status:404
EOF

# Add DNS record
cloudflared tunnel route dns localllm llm.jarvisinfra.com

# Start tunnel
cloudflared tunnel run localllm

https://llm.jarvisinfra.com 

# then ############################# - 
cat > ~/Desktop/LocalLLM.command << 'LOCALEOF'
#!/bin/bash
PID_F="$HOME/Documents/local-llm-db/server.pid"
CAFF_F="$HOME/Documents/local-llm-db/caff.pid"
TUNNEL_F="$HOME/Documents/local-llm-db/tunnel.pid"
APP_DIR="$HOME/Documents/local-llm-app"
LOG="$HOME/Documents/local-llm-db/server.log"

# ── If running → STOP ──────────────────────────────────────
if [ -f "$PID_F" ] && kill -0 $(cat "$PID_F") 2>/dev/null; then
  echo ""
  echo "╔══════════════════════════════════╗"
  echo "║    LocalLLM — Stopping...        ║"
  echo "╚══════════════════════════════════╝"
  echo ""
  kill $(cat "$PID_F") 2>/dev/null; sleep 1; kill -9 $(cat "$PID_F") 2>/dev/null; rm -f "$PID_F"
  [ -f "$CAFF_F" ] && kill $(cat "$CAFF_F") 2>/dev/null && rm -f "$CAFF_F"
  [ -f "$TUNNEL_F" ] && kill $(cat "$TUNNEL_F") 2>/dev/null && rm -f "$TUNNEL_F"
  echo "✓ Server stopped"
  echo "✓ Tunnel stopped"
  echo "✓ Mac can sleep normally"
  sleep 2
  exit 0
fi

# ── If NOT running → START ─────────────────────────────────
echo ""
echo "╔══════════════════════════════════╗"
echo "║    LocalLLM — Starting...        ║"
echo "╚══════════════════════════════════╝"
echo ""

if ! pmset -g ps | grep -q "AC Power"; then
  echo "WARNING: No charger connected."
  read -p "Continue anyway? (y/N): " c
  [[ "$c" != "y" && "$c" != "Y" ]] && exit 1
fi

caffeinate -s &
echo $! > "$CAFF_F"
echo "✓ Sleep prevention ON"

cd "$APP_DIR"
nohup python3.11 app.py >> "$LOG" 2>&1 &
echo $! > "$PID_F"
sleep 2

if kill -0 $(cat "$PID_F") 2>/dev/null; then
  IP=$(ipconfig getifaddr en0 2>/dev/null || echo "192.168.x.x")
  echo "✓ Server running"
  echo "  Local  → http://localhost:8080"
  echo "  Phone  → http://$IP:8080"
else
  echo "✗ Server failed. Check: $LOG"
  exit 1
fi

# Start Cloudflare tunnel
nohup cloudflared tunnel run localllm >> "$LOG" 2>&1 &
echo $! > "$TUNNEL_F"
sleep 3
echo "✓ Tunnel running"
echo "  Domain → https://llm.jarvisinfra.com"

echo ""
echo "╔══════════════════════════════════╗"
echo "║  ✅ LocalLLM is LIVE              ║"
echo "║                                  ║"
echo "║  https://llm.jarvisinfra.com     ║"
echo "║                                  ║"
echo "║  Double-click again to STOP      ║"
echo "╚══════════════════════════════════╝"
LOCALEOF

chmod +x ~/Desktop/LocalLLM.command
xattr -d com.apple.quarantine ~/Desktop/LocalLLM.command 2>/dev/null

# Copy backup to app folder
cp ~/Desktop/LocalLLM.command ~/Documents/local-llm-app/LocalLLM.command

echo "Done — restart LocalLLM now"