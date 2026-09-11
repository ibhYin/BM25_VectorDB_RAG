import json

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
