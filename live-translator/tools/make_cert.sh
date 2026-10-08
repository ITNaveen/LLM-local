#!/usr/bin/env bash
# Self-signed HTTPS certificate for using Live Translator from another computer.
# Browsers only allow microphone / tab-audio capture on https:// (or localhost).
#   tools/make_cert.sh my-mac.local      -> certs/cert.pem + certs/key.pem
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-$(hostname)}"
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=$HOST" -addext "subjectAltName=DNS:$HOST,DNS:localhost,IP:127.0.0.1"
echo "Created certs/cert.pem and certs/key.pem for $HOST"
echo "Start with: ./start.sh --host 0.0.0.0 --token <secret> --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem"
