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
from datetime import datetime
from pathlib import Path
from threading import Lock


DEFAULT_AUDIT_LOG = Path("data/vector_audit.jsonl")
AUDIT_LOG_ENV = "VECTOR_AUDIT_LOG"

# Serializes appends so concurrent writers cannot interleave a line.
_append_lock = Lock()
logger = logging.getLogger(__name__)


def audit_log_path() -> Path:
    return Path(os.environ.get(AUDIT_LOG_ENV, str(DEFAULT_AUDIT_LOG)))


def record(event: str, actor: str = "", **fields: object) -> None:
    """Append one audit entry. Never raises: logging must not break a write."""

    entry: dict[str, object] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": event,
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


def read_entries(limit: int = 100) -> list[dict[str, object]]:
    """Return the most recent entries, newest first."""

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
        if len(entries) >= limit:
            break
    return entries
