"""Tests for the vector-store audit trail."""

import json
import re
from pathlib import Path

import pytest

from app import audit


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "audit" / "vector_audit.jsonl"
    monkeypatch.setenv("VECTOR_AUDIT_LOG", str(path))
    return path


def test_record_appends_one_json_line(log_path: Path) -> None:
    audit.record("ingest.files", actor="alice", files=["a.pdf"], chunks=7)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "ingest.files"
    assert entry["status"] == "success"
    assert entry["actor"] == "alice"
    assert entry["files"] == ["a.pdf"]
    assert entry["chunks"] == 7


def test_explicit_failed_status_overrides_the_default(log_path: Path) -> None:
    audit.record(
        "ingest.files",
        actor="alice",
        **audit.failure_fields(RuntimeError("向量服务不可用"), "vector_write"),
    )

    entry = audit.read_entries()[0]
    assert entry["status"] == "failed"
    assert entry["error_type"] == "RuntimeError"
    assert entry["error"] == "向量服务不可用"
    assert entry["failure_stage"] == "vector_write"


def test_safe_error_redacts_credentials_and_signed_url_queries() -> None:
    raw = (
        "上传失败 https://oss.example/upload?X-Amz-Signature=very-secret&token=query-secret "
        "Authorization=Bearer bearer-secret API Key=sk-secret token=plain-secret"
    )

    cleaned = audit.safe_error(RuntimeError(raw))

    for secret in ("very-secret", "query-secret", "bearer-secret", "sk-secret", "plain-secret"):
        assert secret not in cleaned
    assert "https://oss.example/upload?[REDACTED]" in cleaned
    assert "[REDACTED]" in cleaned
    assert len(cleaned) <= audit.MAX_ERROR_CHARS


def test_safe_error_collapses_whitespace_and_limits_length() -> None:
    cleaned = audit.safe_error("第一行\n  第二行 " + "x" * 1000)

    assert cleaned.startswith("第一行 第二行 ")
    assert "\n" not in cleaned
    assert len(cleaned) == audit.MAX_ERROR_CHARS


def test_timestamp_is_iso_with_an_offset(log_path: Path) -> None:
    audit.record("delete.documents", actor="bob")

    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    # e.g. 2026-09-12T15:30:00.123456+08:00 — sortable and unambiguous.
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+?\d{2}:\d{2}$",
        entry["timestamp"],
    )


def test_record_keeps_appending_rather_than_overwriting(log_path: Path) -> None:
    audit.record("ingest.directory", actor="cli")
    audit.record("ingest.files", actor="alice")

    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


def test_record_defaults_a_missing_actor(log_path: Path) -> None:
    audit.record("ingest.directory")

    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["actor"] == "unknown"


def test_read_entries_returns_newest_first(log_path: Path) -> None:
    audit.record("first", actor="a")
    audit.record("second", actor="b")
    audit.record("third", actor="c")

    assert [entry["event"] for entry in audit.read_entries()] == [
        "third",
        "second",
        "first",
    ]


def test_read_entries_respects_the_limit(log_path: Path) -> None:
    for index in range(5):
        audit.record("event", actor="a", index=index)

    entries = audit.read_entries(limit=2)

    assert len(entries) == 2
    assert [entry["index"] for entry in entries] == [4, 3]


def test_read_entries_without_a_limit_scans_the_whole_trail(log_path: Path) -> None:
    """入库归属要能查到很久以前的记录，不能被默认的 100 条窗口截断。"""

    for index in range(150):
        audit.record("ingest.files", actor="a", index=index)

    assert len(audit.read_entries(limit=None)) == 150
    assert len(audit.read_entries(limit=100)) == 100
    assert len(audit.read_entries(limit=None)) > len(audit.read_entries())


def test_read_entries_skips_a_torn_line(log_path: Path) -> None:
    audit.record("good", actor="a")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "truncated"\n')

    assert [entry["event"] for entry in audit.read_entries()] == ["good"]


def test_read_entries_on_a_missing_file_is_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_AUDIT_LOG", str(tmp_path / "nope.jsonl"))

    assert audit.read_entries() == []


def test_record_never_raises_on_an_unwritable_path(monkeypatch) -> None:
    # A directory where the log file should be: opening it always fails.
    monkeypatch.setenv("VECTOR_AUDIT_LOG", "/proc/definitely/not/writable.jsonl")

    audit.record("ingest.files", actor="alice")  # must not propagate
