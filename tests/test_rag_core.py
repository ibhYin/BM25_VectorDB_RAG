from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from langchain_core.documents import Document

from app.config import Settings
from app.service import answer_question, ingest_knowledge_base
from rag.retriever import build_vector_hits
from rag.splitter import split_documents
from rag.vector_store import build_vector_id, ingest_documents


def test_split_documents_adds_stable_chunk_metadata() -> None:
    documents = [
        Document(
            page_content="第一段介绍数据安全委员会负责统筹管理。\n\n第二段介绍管理要求。",
            metadata={"source": "data/raw/policy.md"},
        )
    ]

    chunks = split_documents(documents, chunk_size=30, chunk_overlap=0)

    assert chunks[0].metadata["doc_id"].startswith("policy_")
    assert chunks[0].metadata["chunk_id"].endswith("#chunk_001")
    assert chunks[0].metadata["title"] == "policy"
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == len(chunks[0].page_content)


def test_split_documents_keeps_markdown_section_path() -> None:
    documents = [
        Document(
            page_content="## 数据安全职责\n\n数据安全委员会负责统筹管理。",
            metadata={"source": "data/raw/policy.md"},
        )
    ]

    chunks = split_documents(documents, chunk_size=80, chunk_overlap=0)

    assert chunks[0].metadata["section_path"] == "数据安全职责"


def test_ingest_service_passes_shared_settings(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    document = Document(page_content="source", metadata={"source": "note.md"})
    chunk = Document(page_content="chunk", metadata={"chunk_id": "note#chunk_001"})
    captured = {}

    def fake_build_vector_index(**kwargs):
        captured.update(kwargs)
        return [document], [chunk]

    monkeypatch.setattr("app.service.build_vector_index", fake_build_vector_index)
    settings = Settings(
        _env_file=None,
        data_dir=source_dir,
        chroma_persist_dir=tmp_path / "chroma",
        openai_base_url="https://example.test/v1",
        embedding_api_key="",
        embedding_base_url="http://127.0.0.1:11434/v1",
    )

    result = ingest_knowledge_base(settings)

    assert (result.documents, result.chunks) == (1, 1)
    assert captured["openai_base_url"] == "https://example.test/v1"
    assert captured["embedding_api_key"] == ""
    assert captured["embedding_base_url"] == "http://127.0.0.1:11434/v1"


def test_answer_question_allows_concurrent_reads(monkeypatch) -> None:
    barrier = Barrier(2, timeout=2)

    class FakeEmbeddings:
        def embed_query(self, question):
            barrier.wait()
            return [0.1, 0.2]

    class FakeVectorStore:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            return []

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: FakeEmbeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: FakeVectorStore())
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kwargs: kwargs["question"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                answer_question, f"问题 {index}", Settings(),
                retrieval_method="vector",
            )
            for index in range(2)
        ]
        answers = [future.result() for future in futures]

    assert [answer["answer"] for answer in answers] == ["问题 0", "问题 1"]


def test_answer_uses_dedicated_embedding_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeEmbeddings:
        def embed_query(self, question):
            return [0.1, 0.2]

    class FakeVectorStore:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            return []

    monkeypatch.setattr(
        "app.service.get_embeddings",
        lambda model, api_key, base_url: captured.update(
            embedding_model=model,
            embedding_api_key=api_key,
            embedding_base_url=base_url,
        )
        or FakeEmbeddings(),
    )
    monkeypatch.setattr(
        "app.service.get_vector_store",
        lambda persist_dir, model, api_key, base_url: captured.update(
            persist_dir=persist_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        or FakeVectorStore(),
    )
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kwargs: "answer")
    settings = Settings(
        _env_file=None,
        chroma_persist_dir=Path("chroma-test"),
        embedding_model="bge-m3:q8_0",
        embedding_api_key="ollama",
        embedding_base_url="http://127.0.0.1:11434/v1",
    )

    result = answer_question("测试问题", settings, retrieval_method="vector")

    assert result["answer"] == "answer"
    assert captured == {
        "persist_dir": Path("chroma-test"),
        "model": "bge-m3:q8_0",
        "api_key": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "embedding_model": "bge-m3:q8_0",
        "embedding_api_key": "ollama",
        "embedding_base_url": "http://127.0.0.1:11434/v1",
    }


def test_vector_ingest_uses_stable_chunk_ids(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            captured["ids"] = ids

    monkeypatch.setattr(
        "rag.vector_store.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )
    documents = [
        Document(page_content="chunk", metadata={"chunk_id": "doc#chunk_001"})
    ]

    ingest_documents(documents, tmp_path, "embedding-model")

    assert captured["ids"] == ["doc#chunk_001"]


def test_vector_id_is_stable_without_chunk_metadata() -> None:
    document = Document(
        page_content="same content",
        metadata={"source": "book.pdf", "page": 3},
    )

    assert build_vector_id(document) == build_vector_id(document)
    assert build_vector_id(document).startswith("document_")


def test_vector_hits_keep_original_text_and_metadata() -> None:
    document = Document(
        page_content="原始切片文本",
        metadata={
            "chunk_id": "doc#chunk_001",
            "doc_id": "doc",
            "title": "文档",
            "source": "doc.pdf",
            "page": 2,
            "page_label": "3",
            "section_path": "第一章",
        },
    )

    hits = build_vector_hits([(document, 0.125)])

    assert hits[0]["text"] == "原始切片文本"
    assert hits[0]["chunk_id"] == "doc#chunk_001"
    assert hits[0]["distance"] == 0.125


def test_rebuild_deletes_stale_vectors_after_upsert(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class FakeVectorStore:
        def get(self, include):
            return {"ids": ["old-random-id", "doc#chunk_001"]}

        def add_documents(self, documents, ids):
            calls.append(("add", ids))

        def delete(self, ids):
            calls.append(("delete", ids))

    monkeypatch.setattr(
        "rag.vector_store.get_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    ingest_documents(
        [Document(page_content="chunk", metadata={"chunk_id": "doc#chunk_001"})],
        tmp_path,
        "embedding-model",
        rebuild=True,
    )

    assert calls == [
        ("add", ["doc#chunk_001"]),
        ("delete", ["old-random-id"]),
    ]
