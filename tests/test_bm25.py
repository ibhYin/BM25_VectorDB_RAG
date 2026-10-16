import json
from threading import Thread
from time import monotonic, sleep
from urllib.request import Request, urlopen

import chromadb
import pytest
from langchain_core.documents import Document

from app.answer_jobs import AnswerJobStore
from app.config import Settings
from app.service import answer_question
from app.web import RagWebServer, RequestError, WebRequestHandler
from rag.retriever import search_bm25
from rag.vector_store import COLLECTION_NAME


def test_bm25_ranks_chinese_and_reads_updated_chunks(tmp_path):
    assert search_bm25("attention", tmp_path, 20) == []
    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        COLLECTION_NAME, embedding_function=None,
    )
    assert search_bm25("attention", tmp_path, 20) == []
    collection.add(
        ids=["a", "b", "c"],
        documents=["注意力 机制 Attention", "注意力 " + "语言模型 " * 20, "数据库 索引"],
        metadatas=[
            {"chunk_id": "doc#chunk_001", "doc_id": "doc", "source": "doc.pdf", "page": 3},
            {"chunk_id": "doc#chunk_002"},
            {"chunk_id": "other#chunk_001"},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    hits = search_bm25("注意力", tmp_path, 20)
    assert len(hits) == 2
    assert [hit["rank"] for hit in hits] == [1, 2]
    assert hits[0]["bm25_score"] > hits[1]["bm25_score"] > 0
    assert hits[0]["chunk_id"] == "doc#chunk_001"
    assert hits[0]["source"] == "doc.pdf" and hits[0]["page"] == 3
    assert hits[0]["text"] == "注意力 机制 Attention"
    assert hits[0]["retrieval_method"] == "bm25"
    assert "distance" not in hits[0] and "similarity" not in hits[0]
    assert search_bm25("注意力", tmp_path, 1) == hits[:1]
    assert search_bm25('ATTENTION" ) OR (', tmp_path, 20)[0]["chunk_id"] == "doc#chunk_001"
    assert search_bm25("不存在的稀有关键词xyz", tmp_path, 20) == []
    assert search_bm25(' !!! " () ', tmp_path, 20) == []

    collection.upsert(ids=["a"], documents=["新版关键词"], embeddings=[[1.0, 0.0]])
    assert search_bm25("attention", tmp_path, 20) == []
    assert search_bm25("新版关键词", tmp_path, 20)[0]["text"] == "新版关键词"
    collection.delete(ids=["a", "b", "c"])
    assert search_bm25("注意力", tmp_path, 20) == []
    collection.add(ids=["single"], documents=["注意力"], embeddings=[[1.0, 0.0]])
    assert search_bm25("注意力", tmp_path, 20)[0]["bm25_score"] > 0


def test_bm25_http_sync_and_async_skip_embeddings(tmp_path, monkeypatch):
    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        COLLECTION_NAME, embedding_function=None,
    )
    collection.add(
        ids=["chunk"], documents=["关键词检索使用 BM25"],
        metadatas=[{"chunk_id": "doc#chunk_001", "doc_id": "doc"}],
        embeddings=[[1.0, 0.0]],
    )
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path)

    def no_embeddings(*args, **kwargs):
        pytest.fail("BM25 must not construct or call embedding clients")

    def fake_answer(**kwargs):
        assert "doc#chunk_001" in kwargs["vector_context"][0]
        return "关键词检索 [source: doc#chunk_001]"

    monkeypatch.setattr("app.service.get_embeddings", no_embeddings)
    monkeypatch.setattr("app.service.get_vector_store", no_embeddings)
    monkeypatch.setattr("chains.answer_chain.generate_answer", fake_answer)
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    store = AnswerJobStore(settings)
    monkeypatch.setattr("app.answer_jobs.get_answer_store", lambda: store)
    server = RagWebServer(("127.0.0.1", 0), WebRequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        with urlopen(Request(base_url + path, data=body, headers={
            "Content-Type": "application/json",
        }), timeout=5) as response:
            return json.load(response)

    try:
        payload = {"question": "BM25", "retrieval_method": "bm25", "top_k": 1,
                   "similarity_threshold": 1.0}
        sync = request("/api/ask", payload)
        assert sync["retrieval_method"] == "bm25"
        assert len(sync["vector_hits"]) == 1
        job = request("/api/answer-jobs", payload)
        deadline = monotonic() + 5
        while job["status"] not in {"completed", "failed"} and monotonic() < deadline:
            sleep(0.02)
            job = request(f"/api/answer-jobs/{job['id']}")
        assert job["status"] == "completed", job
        assert job["answer"] == sync["answer"]
        assert job["vector_hits"] == sync["vector_hits"]
        assert job["retrieval_method"] == "bm25" and job["top_k"] == 1
        assert job["stages"]["encode"]["status"] == "skipped"
        assert job["stages"]["encode"]["finished_at"]
        detail = job["stages"]["vector_search"]["detail"]
        assert detail["retrieval_method"] == "bm25"
        assert detail["similarity_threshold"] is None
        assert request("/api/status")["retrieval_method"] == "hybrid"

        settings.retrieval_method = "bm25"
        assert answer_question("BM25", settings)["vector_hits"] == sync["vector_hits"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store._executor.shutdown(wait=True)


@pytest.mark.parametrize("method", [None, "", "semantic", 1, False, [], {}])
def test_api_rejects_invalid_retrieval_method(method):
    with pytest.raises(RequestError) as error:
        WebRequestHandler._read_retrieval_params({"retrieval_method": method})
    assert error.value.status == 400


def test_retrieval_defaults_and_validation():
    assert Settings(_env_file=None).retrieval_method == "hybrid"
    with pytest.raises(ValueError):
        Settings(_env_file=None, retrieval_method="semantic")
    with pytest.raises(ValueError, match="retrieval_method"):
        answer_question("test", Settings(_env_file=None), retrieval_method="semantic")
    assert WebRequestHandler._read_retrieval_params({}) == {
        "top_k": None,
        "similarity_threshold": None,
        "retrieval_method": None,
        "candidate_pool": None,
        "fusion_method": None,
        "fusion_k": None,
        "fusion_alpha": None,
    }
    assert WebRequestHandler._read_retrieval_params({
        "retrieval_method": "bm25", "top_k": 20, "similarity_threshold": 0,
    }) == {
        "top_k": 20,
        "similarity_threshold": 0,
        "retrieval_method": "bm25",
        "candidate_pool": None,
        "fusion_method": None,
        "fusion_k": None,
        "fusion_alpha": None,
    }


def test_vector_mode_retains_top_k_and_similarity_filter(monkeypatch):
    class Embeddings:
        def embed_query(self, question):
            return [1.0, 0.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            assert embedding == [1.0, 0.0] and k == 20
            return [(Document(page_content="close"), 0.1),
                    (Document(page_content="far"), 1.0)]

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kwargs: "answer")
    settings = Settings(_env_file=None, retrieval_method="bm25")
    stages = {}
    result = answer_question("test", settings, top_k=20, similarity_threshold=0.35,
                             retrieval_method="vector",
                             progress=lambda stage, status, detail: (
                                 stages.update({stage: detail})
                                 if status == "completed" else None
                             ))
    assert result["retrieval_method"] == "vector"
    assert len(result["vector_hits"]) == 1
    assert stages["vector_search"]["recall"] == 2
    assert stages["vector_search"]["matches"] == 1
    assert result["vector_hits"][0]["distance"] == 0.1
    assert result["vector_hits"][0]["similarity"] >= 0.35
    assert "bm25_score" not in result["vector_hits"][0]
    result = answer_question("test", settings, similarity_threshold=0,
                             retrieval_method="vector")
    assert len(result["vector_hits"]) == 2


def _make_collection(tmp_path, docs):
    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        COLLECTION_NAME, embedding_function=None,
    )
    collection.add(
        ids=[f"id{i}" for i in range(len(docs))],
        documents=[text for text, _ in docs],
        metadatas=[meta for _, meta in docs],
        embeddings=[[1.0, 0.0]] * len(docs),
    )
    return collection


def test_persisted_index_matches_in_memory_results(tmp_path):
    docs = [
        ("注意力 机制 Attention", {"chunk_id": "doc#chunk_001", "doc_id": "doc"}),
        ("注意力 " + "语言模型 " * 20, {"chunk_id": "doc#chunk_002", "doc_id": "doc"}),
        ("数据库 索引", {"chunk_id": "other#chunk_001", "doc_id": "other"}),
    ]
    _make_collection(tmp_path, docs)

    # 无索引 → 回退内存路径
    fallback = search_bm25("注意力", tmp_path, 20)
    assert len(fallback) == 2

    from rag.bm25_index import rebuild_bm25_index

    rebuild_bm25_index(tmp_path, embedding_model="test")
    assert (tmp_path / "bm25.sqlite3").is_file()

    # 有索引 → 快路径
    indexed = search_bm25("注意力", tmp_path, 20)
    assert [hit["chunk_id"] for hit in indexed] == [
        hit["chunk_id"] for hit in fallback
    ]
    assert indexed[0]["bm25_score"] > indexed[1]["bm25_score"] > 0
    assert indexed[0]["text"] == "注意力 机制 Attention"
    assert indexed[0]["retrieval_method"] == "bm25"


def test_persisted_index_follows_upsert_and_delete(tmp_path):
    _make_collection(tmp_path, [("注意力", {"chunk_id": "doc#chunk_001"})])
    from rag.bm25_index import rebuild_bm25_index

    rebuild_bm25_index(tmp_path, embedding_model="test")
    assert search_bm25("注意力", tmp_path, 20)[0]["chunk_id"] == "doc#chunk_001"

    # upsert 后不重建 → 旧索引不返回新词
    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection(
        COLLECTION_NAME, embedding_function=None,
    )
    collection.upsert(ids=["id0"], documents=["新版关键词"], embeddings=[[1.0, 0.0]])
    assert search_bm25("新版关键词", tmp_path, 20) == []

    # 重建后 → 跟随新内容
    rebuild_bm25_index(tmp_path, embedding_model="test")
    assert search_bm25("新版关键词", tmp_path, 20)[0]["text"] == "新版关键词"

    # delete 后重建 → 不再返回
    collection.delete(ids=["id0"])
    rebuild_bm25_index(tmp_path, embedding_model="test")
    assert search_bm25("注意力", tmp_path, 20) == []


def test_persisted_index_respects_access_filter(tmp_path):
    _make_collection(
        tmp_path,
        [
            ("公开内容", {"chunk_id": "pub#chunk_001", "groups": ["public"]}),
            ("秘密内容", {"chunk_id": "sec#chunk_001", "groups": ["secret"]}),
        ],
    )
    from rag.bm25_index import rebuild_bm25_index
    from app.access import AccessPolicy

    rebuild_bm25_index(tmp_path, embedding_model="test")

    class Allowed:
        def can_access(self, metadata, user, knowledge_base_id):
            return "secret" not in (metadata.get("groups") or [])

    policy = Allowed()
    hits = search_bm25("内容", tmp_path, 20, "default", user="bob", access_policy=policy)
    assert [hit["chunk_id"] for hit in hits] == ["pub#chunk_001"]
