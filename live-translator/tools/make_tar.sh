#!/usr/bin/env bash
# Package the app as dist/live-translator-<version>.tar.gz (no virtualenv, caches or tests output).
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' livetranslator/__init__.py)
mkdir -p dist
OUT="dist/live-translator-$VER.tar.gz"
tar --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='dist' \
    --exclude='certs' --exclude='.DS_Store' -C .. -czf "$OUT" "$(basename "$(pwd)")"
echo "$OUT ($(du -h "$OUT" | cut -f1))"
