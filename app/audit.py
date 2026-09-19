"""Append-only audit trail for every change to the vector store.

Every write path (directory ingest, browser upload, document delete) records one
JSON line so it is possible to answer "who changed the index, when, and what
changed". The log is append-only and never rewritten, so it survives crashes.

Timestamps are local time with a UTC offset (`2026-09-12T15:30:00.123456+08:00`):
unambiguous to sort, and readable without converting.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from threading import Lock


DEFAULT_AUDIT_LOG = Path("data/vector_audit.jsonl")
AUDIT_LOG_ENV = "VECTOR_AUDIT_LOG"

# Serializes appends so concurrent writers cannot interleave a line.
_append_lock = Lock()
logger = logging.getLogger(__name__)

# 审计日志可能被管理员直接查看，异常原文里又可能带预签名 URL、Bearer token 或
# `api_key=...`。保留可读原因，但不能把凭据顺手写进永久只追加的日志。
_SENSITIVE_PAIRS = re.compile(
    r"(?i)(authorization|api[\s_-]?key|token|secret|signature|credential)"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)
_URL_QUERY = re.compile(r"(https?://[^\s?]+)\?[^\s]+", re.IGNORECASE)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
MAX_ERROR_CHARS = 600


def safe_error(exc: BaseException | str) -> str:
    """适合永久写入审计日志的错误原因（不含 traceback 与常见凭据）。"""

    message = " ".join(str(exc).split()) or exc.__class__.__name__
    message = _BEARER.sub("Bearer [REDACTED]", message)
    message = _SENSITIVE_PAIRS.sub(r"\1\2[REDACTED]", message)
    message = _URL_QUERY.sub(r"\1?[REDACTED]", message)
    return message[:MAX_ERROR_CHARS]


def failure_fields(exc: BaseException | str, stage: str = "") -> dict[str, str]:
    """失败记录共用字段。字符串错误没有可靠类型，记为 Error。"""

    fields = {
        "status": "failed",
        "error_type": exc.__class__.__name__ if isinstance(exc, BaseException) else "Error",
        "error": safe_error(exc),
    }
    if stage:
        fields["failure_stage"] = stage
    return fields


def audit_log_path() -> Path:
    return Path(os.environ.get(AUDIT_LOG_ENV, str(DEFAULT_AUDIT_LOG)))


def record(event: str, actor: str = "", **fields: object) -> None:
    """Append one audit entry. Never raises: logging must not break a write."""

    entry: dict[str, object] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": event,
        "status": "success",
        "actor": actor or "unknown",
    }
    entry.update(fields)

    try:
        path = audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _append_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        # A missing audit trail is bad, but losing an ingest is worse.
        logger.warning("Could not write audit entry %s: %s", event, exc)


def read_entries(limit: int | None = 100) -> list[dict[str, object]]:
    """Return the most recent entries, newest first.

    ``limit=None`` scans the whole trail. Callers that attribute a document to
    its ingest need that: a document indexed long ago sits far past the tail,
    and stopping early would silently render it as "no record".
    """

    path = audit_log_path()
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    entries: list[dict[str, object]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # A torn final line (crash mid-append) must not hide the rest.
            continue
        if isinstance(payload, dict):
            entries.append(payload)
        if limit is not None and len(entries) >= limit:
            break
    return entries
