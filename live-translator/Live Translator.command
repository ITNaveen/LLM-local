#!/bin/bash
# Double-click: starts Live Translator if it is off, stops it if it is on.
exec "$(dirname "$0")/toggle.sh"
