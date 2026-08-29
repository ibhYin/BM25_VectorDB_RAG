from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.service import ingest_files, list_indexed_documents
from app.uploads import parse_multipart, safe_filename
from rag.constants import SUPPORTED_SUFFIXES
from rag.vector_store import replace_documents


def multipart(filename: str, content: bytes = b"hello") -> bytes:
    return (
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="files"; filename="'
        + filename.encode("utf-8")
        + b'"\r\nContent-Type: application/octet-stream\r\n\r\n'
        + content
        + b"\r\n--boundary--\r\n"
    )


def test_multipart_parser_sanitizes_paths_and_preserves_content() -> None:
    files = parse_multipart(
        "multipart/form-data; boundary=boundary",
        multipart("../../nested/note.md", b"# note"),
        supported_suffixes=SUPPORTED_SUFFIXES,
        max_files=5,
    )

    assert files == [("note.md", b"# note")]


def test_multipart_parser_rejects_unsupported_and_duplicate_files() -> None:
    with pytest.raises(ValueError, match="Unsupported document type"):
        parse_multipart(
            "multipart/form-data; boundary=boundary",
            multipart("image.png"),
            supported_suffixes=SUPPORTED_SUFFIXES,
            max_files=5,
        )

    body = multipart("note.md").replace(b"\r\n--boundary--\r\n", b"\r\n") + (
        b"\r\n--boundary\r\n"
        b'Content-Disposition: form-data; name="files"; filename="NOTE.md"\r\n\r\n'
        b"another\r\n--boundary--\r\n"
    )
    with pytest.raises(ValueError, match="Duplicate filename"):
        parse_multipart(
            "multipart/form-data; boundary=boundary",
            body,
            supported_suffixes=SUPPORTED_SUFFIXES,
            max_files=5,
        )


def test_safe_filename_rejects_traversal_only_names() -> None:
    assert safe_filename("../note.md") == "note.md"
    with pytest.raises(ValueError, match="Filename is required"):
        safe_filename("../")


def test_replace_documents_removes_only_stale_chunks_after_upsert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeVectorStore:
        def get(self, where, include):
            calls.append(("get", where, include))
            return {"ids": ["doc#chunk_001", "doc#chunk_002"]}

        def add_documents(self, documents, ids):
            calls.append(("add", ids))

        def delete(self, ids):
            calls.append(("delete", ids))

    monkeypatch.setattr(
        "rag.vector_store.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )
    documents = [
        Document(
            page_content="new content",
            metadata={"doc_id": "doc", "chunk_id": "doc#chunk_001"},
        )
    ]

    replace_documents(documents, tmp_path, "embedding-model")

    assert calls == [
        ("get", {"doc_id": {"$in": ["doc"]}}, []),
        ("add", ["doc#chunk_001"]),
        ("delete", ["doc#chunk_002"]),
    ]


def test_ingest_files_uses_shared_settings_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    document = Document(page_content="page", metadata={"source": str(tmp_path / "note.md")})
    chunk = Document(
        page_content="chunk",
        metadata={"doc_id": "note", "chunk_id": "note#chunk_001"},
    )

    monkeypatch.setattr(
        "app.service.load_document_file",
        lambda path, **kwargs: captured.update(path=path, **kwargs) or [document],
    )
    monkeypatch.setattr("app.service.split_documents", lambda docs, **kwargs: [chunk])
    monkeypatch.setattr(
        "app.service.replace_documents",
        lambda **kwargs: captured.update(replacement=kwargs),
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        chroma_persist_dir=tmp_path / "chroma",
        embedding_model="bge-small-zh:q8_0",
        embedding_api_key="ollama",
        embedding_base_url="http://127.0.0.1:11434/v1",
    )
    source = tmp_path / "note.md"

    result = ingest_files(settings, [source])

    assert (result.files, result.documents, result.chunks) == (1, 1, 1)
    assert captured["path"] == source
    assert captured["replacement"]["embedding_model"] == "bge-small-zh:q8_0"


def test_list_indexed_documents_aggregates_chunk_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVectorStore:
        def get(self, include):
            return {
                "metadatas": [
                    {"doc_id": "a", "title": "A", "source": "a.md"},
                    {"doc_id": "a", "title": "A", "source": "a.md"},
                    {"doc_id": "b", "title": "B", "source": "b.md"},
                ]
            }

    monkeypatch.setattr(
        "app.service.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    documents = list_indexed_documents(Settings(_env_file=None))

    assert documents == [
        {"doc_id": "a", "title": "A", "source": "a.md", "chunk_count": 2},
        {"doc_id": "b", "title": "B", "source": "b.md", "chunk_count": 1},
    ]
