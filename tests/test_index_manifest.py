import json

import pytest

from rag.index_manifest import (
    manifest_path,
    read_index_embedding_model,
    write_index_manifest,
)


def test_write_then_read_round_trip(tmp_path) -> None:
    write_index_manifest(tmp_path / "chroma", "bge-small-zh:q8_0")

    assert read_index_embedding_model(tmp_path / "chroma") == "bge-small-zh:q8_0"
    payload = json.loads(manifest_path(tmp_path / "chroma").read_text(encoding="utf-8"))
    assert payload["embedding_model"] == "bge-small-zh:q8_0"
    assert payload["updated_at"]


def test_missing_manifest_reads_empty(tmp_path) -> None:
    assert read_index_embedding_model(tmp_path / "does-not-exist") == ""


def test_corrupt_or_non_object_manifest_reads_empty(tmp_path) -> None:
    directory = tmp_path / "chroma"
    directory.mkdir()
    target = manifest_path(directory)

    target.write_text("{not json", encoding="utf-8")
    assert read_index_embedding_model(directory) == ""

    target.write_text('["not", "an", "object"]', encoding="utf-8")
    assert read_index_embedding_model(directory) == ""


def test_blank_model_is_not_written(tmp_path) -> None:
    write_index_manifest(tmp_path / "chroma", "")

    assert read_index_embedding_model(tmp_path / "chroma") == ""


def test_rewrite_replaces_the_previous_model(tmp_path) -> None:
    write_index_manifest(tmp_path / "chroma", "bge-small-zh:q8_0")
    write_index_manifest(tmp_path / "chroma", "bge-m3:q8_0")

    assert read_index_embedding_model(tmp_path / "chroma") == "bge-m3:q8_0"


def test_vector_store_rejects_mismatched_model_before_opening(tmp_path, monkeypatch):
    from rag.vector_store import get_vector_store

    write_index_manifest(tmp_path, "old-model")
    monkeypatch.setattr("rag.vector_store.get_embeddings", lambda *args: None)
    monkeypatch.setattr("rag.vector_store.Chroma", lambda **kwargs: pytest.fail("must not open index"))
    with pytest.raises(ValueError, match="old-model"):
        get_vector_store(tmp_path, "new-model")
    assert read_index_embedding_model(tmp_path) == "old-model"


def test_mismatched_upload_preserves_original_file(tmp_path, monkeypatch):
    from app.config import Settings
    from app.uploads import UploadJobStore

    persist_dir = tmp_path / "index"
    write_index_manifest(persist_dir, "old-model")
    original = tmp_path / "note.md"
    original.write_bytes(b"original")
    store = UploadJobStore(
        Settings(_env_file=None, chroma_persist_dir=persist_dir, embedding_model="new-model"),
        tmp_path / "jobs.json",
    )
    monkeypatch.setattr(store._executor, "submit", lambda *args: None)
    try:
        with pytest.raises(ValueError, match="old-model"):
            store.create_uploaded([("note.md", b"replacement")], tmp_path)
        assert original.read_bytes() == b"original"
        assert store.snapshot() == []
    finally:
        store._executor.shutdown(wait=True)


def test_mismatched_approval_does_not_consume_staged_files(tmp_path, monkeypatch):
    from app.config import Settings
    from app.service import approve_staged_upload

    write_index_manifest(tmp_path, "old-model")
    monkeypatch.setattr(
        "app.staging.get_staging_store",
        lambda: pytest.fail("must not consume staging files"),
    )
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path, embedding_model="new-model")
    with pytest.raises(ValueError, match="old-model"):
        approve_staged_upload(settings, "pending-item")
