#!/bin/zsh

set -u

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
HOST="127.0.0.1"
PORT="8000"
BACKEND_URL="http://${HOST}:${PORT}"
LOG_FILE="/tmp/documind-web.log"
PID_FILE="/tmp/documind-web.pid"

cd "$PROJECT_DIR" || exit 1

python_candidates=(
  "${RAG_PYTHON:-}"
  "$PROJECT_DIR/.venv/bin/python"
  "/opt/anaconda3/envs/langchain/bin/python"
  "$(command -v python3 2>/dev/null || true)"
)
PYTHON=""
for candidate in "${python_candidates[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && \
    "$candidate" -c 'import typer, langchain' >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  printf '%s\n' "No Python environment with the project dependencies was found." >"$LOG_FILE"
  open -a "TextEdit" "$LOG_FILE"
  exit 1
fi

# Ensure the API backend is up.
if ! curl --max-time 2 --silent --fail "$BACKEND_URL/" >/dev/null 2>&1; then
  rm -f "$PID_FILE"
  nohup "$PYTHON" -m app.main web --host "$HOST" --port "$PORT" \
    >"$LOG_FILE" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" >"$PID_FILE"
fi

for _ in {1..30}; do
  if curl --max-time 2 --silent --fail "$BACKEND_URL/" >/dev/null 2>&1; then
    open -a "Google Chrome" "$BACKEND_URL"
    exit 0
  fi
  sleep 1
done

open -a "TextEdit" "$LOG_FILE"
exit 1
