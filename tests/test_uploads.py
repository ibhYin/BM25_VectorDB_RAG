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
    files, fields = parse_multipart(
        "multipart/form-data; boundary=boundary",
        multipart("../../nested/note.md", b"# note"),
        supported_suffixes=SUPPORTED_SUFFIXES,
        max_files=5,
    )

    assert files == [("note.md", b"# note")]
    assert fields == {}


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
                    {"doc_id": "a", "title": "A", "source": "a.md", "document_access_group": "hr"},
                    {"doc_id": "a", "title": "A", "source": "a.md", "document_access_group": "hr"},
                    {"doc_id": "b", "title": "B", "source": "b.md"},
                ]
            }

    monkeypatch.setattr(
        "app.service.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    documents = list_indexed_documents(Settings(_env_file=None))

    assert documents == [
        {
            "doc_id": "a",
            "title": "A",
            "source": "a.md",
            "access_group": "hr",
            "access_groups": ["hr"],
            "chunk_count": 2,
            # 没有 structured_by 的分片走的是固定长度切分。
            "chunk_strategy": "fixed",
            "chunk_strategy_provider": "",
            # 非管理员：入库人/时间不参与（它们来自仅管理员可见的审计日志）。
            "last_ingested_by": "",
            "last_ingested_at": "",
        },
        # 没写访问组的分片按 public 计。
        {
            "doc_id": "b",
            "title": "B",
            "source": "b.md",
            "access_group": "public",
            "access_groups": ["public"],
            "chunk_count": 1,
            "chunk_strategy": "fixed",
            "chunk_strategy_provider": "",
            "last_ingested_by": "",
            "last_ingested_at": "",
        },
    ]


def test_list_indexed_documents_reports_heading_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """structured_by 是切片时分流用的字段，正是「切片方式」的来源。"""

    class FakeVectorStore:
        def get(self, include):
            return {
                "metadatas": [
                    {"doc_id": "s", "title": "S", "source": "s.pdf", "structured_by": "mineru"},
                    {"doc_id": "s", "title": "S", "source": "s.pdf", "structured_by": "mineru"},
                    {"doc_id": "m", "title": "M", "source": "m.docx", "structured_by": "markitdown"},
                ]
            }

    monkeypatch.setattr(
        "app.service.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    by_id = {d["doc_id"]: d for d in list_indexed_documents(Settings(_env_file=None))}

    assert by_id["s"]["chunk_strategy"] == "heading"
    assert by_id["s"]["chunk_strategy_provider"] == "mineru"
    assert by_id["m"]["chunk_strategy_provider"] == "markitdown"


def test_list_indexed_documents_attributes_ingest_to_the_audit_trail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """入库人与时间只有审计日志记得；重复入库取最近一次。"""
    log = tmp_path / "vector_audit.jsonl"
    monkeypatch.setenv("VECTOR_AUDIT_LOG", str(log))
    log.write_text(
        "\n".join(
            [
                # 最早一条：a 第一次入库
                '{"timestamp": "2026-09-12T15:31:43.000000+08:00", "event": "ingest.files",'
                ' "actor": "alice", "doc_ids": ["a"]}',
                # 删文档：不该被当成入库
                '{"timestamp": "2026-09-13T09:00:00.000000+08:00", "event": "delete.documents",'
                ' "actor": "bob", "doc_ids": ["a"]}',
                # a 重新入库：应取这条
                '{"timestamp": "2026-09-14T15:39:23.933703+08:00", "event": "ingest.files",'
                ' "actor": "carol", "doc_ids": ["a"]}',
                # 之后 dave 又尝试入库，但失败了：不能覆盖最近一次成功归属
                '{"timestamp": "2026-09-15T08:00:00.000000+08:00", "event": "ingest.files",'
                ' "status": "failed", "actor": "dave", "doc_ids": ["a"],'
                ' "error": "向量写入失败"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeVectorStore:
        def get(self, include):
            return {
                "metadatas": [
                    {"doc_id": "a", "title": "A", "source": "a.md"},
                    # b 是审计日志建立之前入的库，没有任何记录
                    {"doc_id": "b", "title": "B", "source": "b.md"},
                ]
            }

    monkeypatch.setattr(
        "app.service.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    by_id = {
        d["doc_id"]: d
        for d in list_indexed_documents(Settings(_env_file=None), admin=True)
    }

    assert by_id["a"]["last_ingested_by"] == "carol"
    assert by_id["a"]["last_ingested_at"] == "2026-09-14T15:39:23.933703+08:00"
    # 没有记录的文档留空，不能借用别人的入库人。
    assert by_id["b"]["last_ingested_by"] == ""
    assert by_id["b"]["last_ingested_at"] == ""


def test_list_indexed_documents_hides_ingest_attribution_from_non_admins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "vector_audit.jsonl"
    monkeypatch.setenv("VECTOR_AUDIT_LOG", str(log))
    log.write_text(
        '{"timestamp": "2026-09-14T15:39:23.000000+08:00", "event": "ingest.files",'
        ' "actor": "alice", "doc_ids": ["a"]}\n',
        encoding="utf-8",
    )

    class FakeVectorStore:
        def get(self, include):
            return {"metadatas": [{"doc_id": "a", "title": "A", "source": "a.md"}]}

    monkeypatch.setattr(
        "app.service.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    documents = list_indexed_documents(Settings(_env_file=None))

    assert documents[0]["last_ingested_by"] == ""
    # 切片方式与身份无关，仍然返回。
    assert documents[0]["chunk_strategy"] == "fixed"


def test_ingest_files_records_an_audit_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import audit

    monkeypatch.setattr(
        "app.service.load_document_file",
        lambda path, **kwargs: [
            Document(page_content="page", metadata={"source": str(path)})
        ],
    )
    monkeypatch.setattr(
        "app.service.split_documents",
        lambda docs, **kwargs: [
            Document(
                page_content="chunk",
                metadata={"doc_id": "note", "chunk_id": "note#chunk_001"},
            )
        ],
    )
    monkeypatch.setattr("app.service.replace_documents", lambda **kwargs: None)
    monkeypatch.setattr("app.service.write_index_manifest", lambda *a, **k: None)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        chroma_persist_dir=tmp_path / "chroma",
        embedding_model="bge-small-zh:q8_0",
    )

    ingest_files(settings, [tmp_path / "note.md"], False, "alice")

    entries = audit.read_entries()
    assert len(entries) == 1
    assert entries[0]["event"] == "ingest.files"
    assert entries[0]["actor"] == "alice"
    assert entries[0]["doc_ids"] == ["note"]
    assert entries[0]["chunks"] == 1


def test_ingest_files_records_load_failure_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import audit
    from app.service import ingest_files

    def fail_load(*args, **kwargs):
        raise RuntimeError("MinerU 解析失败：服务暂不可用")

    monkeypatch.setattr("app.service.load_document_file", fail_load)
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")
    source = tmp_path / "broken.pdf"

    with pytest.raises(RuntimeError, match="服务暂不可用"):
        ingest_files(settings, [source], structurize=True, actor="alice")

    entry = audit.read_entries()[0]
    assert entry["event"] == "ingest.files"
    assert entry["status"] == "failed"
    assert entry["actor"] == "alice"
    assert entry["files"] == ["broken.pdf"]
    assert entry["failure_stage"] == "load"
    assert entry["error_type"] == "RuntimeError"
    assert entry["error"] == "MinerU 解析失败：服务暂不可用"
    assert entry["structurize"] is True


def test_ingest_files_records_vector_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import audit
    from app.service import ingest_files

    document = Document(page_content="page", metadata={"source": "note.md"})
    chunk = Document(
        page_content="chunk",
        metadata={"doc_id": "note", "chunk_id": "note#chunk_001"},
    )
    monkeypatch.setattr("app.service.load_document_file", lambda *a, **k: [document])
    monkeypatch.setattr("app.service.split_documents", lambda *a, **k: [chunk])
    monkeypatch.setattr(
        "app.service.replace_documents",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("向量服务不可用")),
    )
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")

    with pytest.raises(RuntimeError, match="向量服务不可用"):
        ingest_files(settings, [tmp_path / "note.md"], actor="alice")

    entry = audit.read_entries()[0]
    assert entry["status"] == "failed"
    assert entry["failure_stage"] == "vector_write"
    assert entry["doc_ids"] == ["note"]


def test_delete_records_failure_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import audit
    from app.service import delete_indexed_documents

    class FakeVectorStore:
        def get(self, where, include):
            return {
                "metadatas": [
                    {
                        "doc_id": "note",
                        "title": "会议纪要",
                        "source": "data/raw/note.md",
                    }
                ]
            }

    monkeypatch.setattr(
        "app.service.get_vector_store", lambda *a, **k: FakeVectorStore()
    )
    monkeypatch.setattr(
        "app.service.delete_documents",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Chroma 删除失败")),
    )
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")

    with pytest.raises(RuntimeError, match="Chroma 删除失败"):
        delete_indexed_documents(settings, ["note"], "alice")

    entry = audit.read_entries()[0]
    assert entry["event"] == "delete.documents"
    assert entry["status"] == "failed"
    assert entry["failure_stage"] == "vector_delete"
    assert entry["deleted_documents"][0]["title"] == "会议纪要"
    assert entry["error"] == "Chroma 删除失败"


def test_delete_not_found_is_a_failed_audit_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import audit
    from app.service import delete_indexed_documents

    class EmptyVectorStore:
        def get(self, where, include):
            return {"metadatas": []}

    monkeypatch.setattr(
        "app.service.get_vector_store", lambda *a, **k: EmptyVectorStore()
    )
    monkeypatch.setattr("app.service.delete_documents", lambda *a, **k: 0)
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")

    assert delete_indexed_documents(settings, ["missing"], "alice") == 0

    entry = audit.read_entries()[0]
    assert entry["status"] == "failed"
    assert entry["error_type"] == "NotFound"
    assert entry["error"] == "Document not found"
    assert entry["deleted_chunks"] == 0
    assert entry["doc_ids"] == ["missing"]


def test_delete_records_an_audit_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import audit
    from app.service import delete_indexed_documents

    class FakeVectorStore:
        def get(self, where, include):
            assert where == {
                "$and": [
                    {"doc_id": {"$in": ["note"]}},
                    {"knowledge_base_id": "default"},
                ]
            }
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {
                        "doc_id": "note",
                        "title": "会议纪要",
                        "source": "data/raw/note.md",
                    },
                    {
                        "doc_id": "note",
                        "title": "会议纪要",
                        "source": "data/raw/note.md",
                    },
                    {
                        "doc_id": "note",
                        "title": "会议纪要",
                        "source": "data/raw/note.md",
                    },
                ]
            }

    deleted_ids = []
    monkeypatch.setattr(
        "app.service.get_vector_store", lambda *a, **k: FakeVectorStore()
    )
    monkeypatch.setattr(
        "app.service.delete_documents",
        lambda doc_ids, *a, **k: deleted_ids.extend(doc_ids) or 3,
    )
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")

    deleted = delete_indexed_documents(
        settings, [" note ", "", "note"], "alice"
    )

    assert deleted == 3
    assert deleted_ids == ["note"]
    entries = audit.read_entries()
    assert entries[0]["event"] == "delete.documents"
    assert entries[0]["actor"] == "alice"
    assert entries[0]["doc_ids"] == ["note"]
    assert entries[0]["deleted_documents"] == [
        {
            "doc_id": "note",
            "title": "会议纪要",
            "source": "data/raw/note.md",
            "chunks": 3,
        }
    ]
    assert entries[0]["deleted_chunks"] == 3
