#!/bin/zsh

set -u

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
HOST="127.0.0.1"
PORT="8000"
BACKEND_URL="http://${HOST}:${PORT}"
LOG_FILE="/tmp/documind-web.log"
PID_FILE="/tmp/documind-web.pid"
COMMIT_FILE="/tmp/documind-web-commit"

cd "$PROJECT_DIR" || exit 1

# 代码当前版本；跑不了 git（比如不是仓库）就标 unknown。
CURRENT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

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

# 后端在跑、但启动时记下的代码版本和现在不一致（代码更新了没重启）就杀掉，
# 让它按新代码重启——「设置页缺字段/配置存不进去」正是旧进程造成的。
if curl --max-time 2 --silent --fail "$BACKEND_URL/" >/dev/null 2>&1 \
  && [[ -f "$PID_FILE" ]] && [[ -f "$COMMIT_FILE" ]] \
  && [[ "$(<"$COMMIT_FILE")" != "$CURRENT_COMMIT" ]]; then
  kill "$(<"$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE" "$COMMIT_FILE"
fi

# Ensure the API backend is up.
if ! curl --max-time 2 --silent --fail "$BACKEND_URL/" >/dev/null 2>&1; then
  rm -f "$PID_FILE"
  nohup "$PYTHON" -m app.main web --host "$HOST" --port "$PORT" \
    >"$LOG_FILE" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" >"$PID_FILE"
  printf '%s\n' "$CURRENT_COMMIT" >"$COMMIT_FILE"
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
