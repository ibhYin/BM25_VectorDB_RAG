#!/bin/zsh

PID_FILE="/tmp/documind-web.pid"

if [[ -f "$PID_FILE" ]]; then
  pid=$(<"$PID_FILE")
  kill "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
fi

exit 0
