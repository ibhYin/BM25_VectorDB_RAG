from __future__ import annotations

import time
from threading import Event

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.service import FileIngestResult, ingest_files
from app.uploads import UploadJobStore
from rag import markdown_cache


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _detail(index: int, name: str, status: str, **extra) -> dict:
    detail = {
        "index": index,
        "name": name,
        "size_bytes": 10,
        "status": status,
        "cache_hit": False,
        "documents": 0,
        "chunks": 0,
        "duration_ms": 5,
        "stage": "load",
        "error": "",
    }
    detail.update(extra)
    return detail


def test_ingest_files_continues_after_one_failure(monkeypatch, tmp_path) -> None:
    chunks = [
        Document(
            page_content="chunk",
            metadata={"doc_id": "a", "chunk_id": "a#1"},
        )
    ]

    def fake_load(path, **kwargs):
        if path.name == "b.md":
            raise RuntimeError("boom")
        return [Document(page_content="page", metadata={"source": str(path)})]

    replaced = []
    monkeypatch.setattr("app.service.load_document_file", fake_load)
    monkeypatch.setattr("app.service.split_documents", lambda *a, **k: list(chunks))
    monkeypatch.setattr(
        "app.service.replace_documents", lambda **kwargs: replaced.append(kwargs)
    )
    monkeypatch.setattr("app.service.write_index_manifest", lambda *a, **k: None)
    monkeypatch.setattr("app.service.rebuild_bm25_index", lambda *a, **k: None)

    events = []

    def progress(stage, done, total, current_file, detail):
        events.append((stage, done, total, current_file, detail and detail["status"]))

    result = ingest_files(
        _settings(chroma_persist_dir=tmp_path / "chroma"),
        [tmp_path / "a.md", tmp_path / "b.md"],
        progress=progress,
        continue_on_error=True,
    )

    assert result.files == 1
    assert result.chunks == 1
    assert result.failed_files == ("b.md",)
    assert result.cancelled is False
    assert replaced
    assert any(item[-1] == "done" for item in events)
    assert any(item[-1] == "failed" for item in events)


def test_ingest_files_stops_between_files_when_cancelled(monkeypatch, tmp_path) -> None:
    chunk = Document(
        page_content="chunk",
        metadata={"doc_id": "a", "chunk_id": "a#1"},
    )
    monkeypatch.setattr(
        "app.service.load_document_file",
        lambda path, **kwargs: [
            Document(page_content="page", metadata={"source": str(path)})
        ],
    )
    monkeypatch.setattr("app.service.split_documents", lambda *a, **k: [chunk])
    replaced = []
    monkeypatch.setattr(
        "app.service.replace_documents", lambda **kwargs: replaced.append(kwargs)
    )
    monkeypatch.setattr("app.service.write_index_manifest", lambda *a, **k: None)
    monkeypatch.setattr("app.service.rebuild_bm25_index", lambda *a, **k: None)

    checks = {"count": 0}

    def should_cancel() -> bool:
        checks["count"] += 1
        return checks["count"] > 1

    events = []

    def progress(stage, done, total, current_file, detail):
        events.append((stage, detail and detail["status"]))

    result = ingest_files(
        _settings(chroma_persist_dir=tmp_path / "chroma"),
        [tmp_path / "a.md", tmp_path / "b.md"],
        progress=progress,
        should_cancel=should_cancel,
        continue_on_error=True,
    )

    assert result.cancelled is True
    assert result.files == 1
    assert result.failed_files == ()
    assert replaced
    assert any(status == "skipped" for _, status in events)


def test_ingest_files_all_failed_returns_failed_result(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "app.service.load_document_file",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    replaced = []
    monkeypatch.setattr(
        "app.service.replace_documents", lambda **kwargs: replaced.append(kwargs)
    )

    result = ingest_files(
        _settings(chroma_persist_dir=tmp_path / "chroma"),
        [tmp_path / "a.md", tmp_path / "b.md"],
        continue_on_error=True,
    )

    assert result.files == 0
    assert result.chunks == 0
    assert result.failed_files == ("a.md", "b.md")
    assert replaced == []


def _wait_for_terminal(store: UploadJobStore, job_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    snapshot = store.snapshot_job(job_id)
    while time.monotonic() < deadline and snapshot["status"] in {
        "queued",
        "indexing",
    }:
        time.sleep(0.01)
        snapshot = store.snapshot_job(job_id)
    return snapshot


def test_upload_job_records_file_details_and_partial_status(
    monkeypatch, tmp_path
) -> None:
    finished = Event()

    def fake_ingest(
        settings,
        paths,
        structurize=False,
        actor="",
        progress=None,
        should_cancel=None,
        continue_on_error=False,
    ):
        progress("load", 0, 2, "a.md", _detail(0, "a.md", "loading"))
        progress(
            "load",
            1,
            2,
            "a.md",
            _detail(0, "a.md", "done", documents=1, chunks=2),
        )
        progress("load", 1, 2, "b.md", _detail(1, "b.md", "failed", error="boom"))
        finished.set()
        return FileIngestResult(1, 1, 2, ("b.md",), False)

    monkeypatch.setattr("app.service.ingest_files", fake_ingest)
    store = UploadJobStore(
        _settings(max_concurrent_answers=1), tmp_path / "jobs.json"
    )

    created = store.create(
        ["a.md", "b.md"], [tmp_path / "a.md", tmp_path / "b.md"]
    )
    assert finished.wait(timeout=2)
    snapshot = _wait_for_terminal(store, created["id"])

    assert snapshot["status"] == "partial"
    assert snapshot["files"] == 1
    assert snapshot["chunks"] == 2
    assert snapshot["failed_files"] == ["b.md"]
    assert [item["status"] for item in snapshot["file_details"]] == [
        "done",
        "failed",
    ]
    assert snapshot["runtime"]["embedding_model"]
    assert snapshot["queue_position"] == 0


def test_upload_job_cancel_covers_queued_jobs(monkeypatch, tmp_path) -> None:
    started = Event()
    release = Event()

    def fake_ingest(
        settings,
        paths,
        structurize=False,
        actor="",
        progress=None,
        should_cancel=None,
        continue_on_error=False,
    ):
        started.set()
        release.wait(timeout=2)
        return FileIngestResult(1, 1, 1, (), False)

    monkeypatch.setattr("app.service.ingest_files", fake_ingest)
    store = UploadJobStore(
        _settings(max_concurrent_answers=1), tmp_path / "jobs.json"
    )

    first = store.create(["a.md"], [tmp_path / "a.md"])
    assert started.wait(timeout=2)
    second = store.create(["b.md"], [tmp_path / "b.md"])

    cancelled = store.request_cancel(second["id"])
    assert cancelled["status"] == "cancelled"

    release.set()
    _wait_for_terminal(store, first["id"])


def test_upload_job_retry_failed_reuses_failed_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "app.service.ingest_files",
        lambda *a, **k: FileIngestResult(0, 0, 0, (), False),
    )
    store = UploadJobStore(
        _settings(max_concurrent_answers=1), tmp_path / "jobs.json"
    )
    failed_path = tmp_path / "b.md"
    failed_path.write_text("broken", encoding="utf-8")

    job = store._new_job(
        ["a.md", "b.md"], [tmp_path / "a.md", failed_path]
    )
    job["status"] = "partial"
    job["file_details"] = [
        _detail(0, "a.md", "done"),
        _detail(1, "b.md", "failed", error="boom"),
    ]
    store._jobs[job["id"]] = job

    retried = store.retry_failed(job["id"])

    assert retried["filenames"] == ["b.md"]
    assert retried["paths"] == [str(failed_path)]
    _wait_for_terminal(store, retried["id"])


def test_markdown_cache_has_entry(tmp_path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"content")

    assert markdown_cache.has_entry(source, "mineru", "vlm") is False
    markdown_cache.write(source, "mineru", "vlm", "# markdown")
    assert markdown_cache.has_entry(source, "mineru", "vlm") is True
    assert markdown_cache.has_entry(source, "mineru", "pipeline") is False
