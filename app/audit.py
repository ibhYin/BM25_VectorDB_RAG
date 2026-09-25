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

# 审计日志保留上限：超过 MAX 行时裁掉最旧、只留最近 TRIM_TO 行，避免无限增长。
# 进程内用近似行数避免每次追加都全量数行。
MAX_AUDIT_LINES = 2000
TRIM_AUDIT_TO_LINES = 1000
_approx_lines: int | None = None

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
        with _append_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            _trim_audit(path)
    except OSError as exc:
        # A missing audit trail is bad, but losing an ingest is worse.
        logger.warning("Could not write audit entry %s: %s", event, exc)


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _trim_audit(path: Path) -> None:
    """行数超上限时裁掉最旧、只留最近 TRIM_AUDIT_TO_LINES 行。

    调用方必须已持有 _append_lock。进程内用近似行数：首次追加时数一次，
    之后每次 +1，只在接近上限时才真的读文件，避免每次追加都全量扫描。
    """

    global _approx_lines
    if _approx_lines is None:
        _approx_lines = _count_lines(path)
        return
    _approx_lines += 1
    if _approx_lines < MAX_AUDIT_LINES:
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= MAX_AUDIT_LINES:
        _approx_lines = len(lines)
        return

    keep = lines[-TRIM_AUDIT_TO_LINES:]
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text("\n".join(keep) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        _approx_lines = len(keep)
    except OSError as exc:
        logger.warning("Could not trim audit log %s: %s", path, exc)


def _reverse_lines(path: Path):
    """从文件末尾分块反向逐行产出原始行（新→旧），提前停止时不用整份读入。"""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            buffer = b""
            position = size
            chunk_size = 8192
            while position > 0:
                read = min(chunk_size, position)
                position -= read
                handle.seek(position)
                chunk = handle.read(read)
                buffer = chunk + buffer
                lines = buffer.split(b"\n")
                buffer = lines[0]
                for raw in reversed(lines[1:]):
                    if raw.strip():
                        yield raw.decode("utf-8", errors="replace")
            if buffer.strip():
                yield buffer.decode("utf-8", errors="replace")
    except OSError:
        return


def iter_entries_newest_first(stop_when=None):
    """从尾部反向逐个产出审计条目（新→旧），不整份拉回内存。

    ``stop_when(entry)`` 为真时提前停止。是**惰性生成器**：调用方在处理每条时
    更新自己的状态（比如文档归属的 remaining），stop_when 才能看到进度、真正
    提前停——如果先攒成完整列表再返回，提前停就无从谈起。
    """

    for line in _reverse_lines(audit_log_path()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # A torn final line (crash mid-append) must not hide the rest.
            continue
        if not isinstance(payload, dict):
            continue
        yield payload
        if stop_when is not None and stop_when(payload):
            return


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
