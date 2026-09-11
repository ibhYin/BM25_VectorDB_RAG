#!/bin/zsh

PID_FILE="/tmp/documind-web.pid"
PROXY_PID_FILE="/tmp/documind-proxy.pid"

for pid_file in "$PID_FILE" "$PROXY_PID_FILE"; do
  if [[ -f "$pid_file" ]]; then
    pid=$(<"$pid_file")
    kill "$pid" 2>/dev/null || true
    rm -f "$pid_file"
  fi
done

exit 0
