import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.service import answer_question
from app.web import RequestError, WebRequestHandler
from rag.retriever import rerank_bm25_scores, rrf_fuse


def test_rerank_scores_only_matching_texts():
    scores = rerank_bm25_scores(
        ["注意力 机制 Attention", "数据库 索引", "注意力 注意力 注意力"],
        "注意力",
    )

    assert scores[1] == 0.0
    assert scores[0] > 0 and scores[2] > scores[0]


def test_rrf_fuse_prefers_double_matches_over_vector_rank():
    hits = [
        {"rank": 1, "text": "a"},
        {"rank": 2, "text": "b"},
        {"rank": 3, "text": "c"},
    ]

    fused = rrf_fuse(hits, [0.0, 5.0, 3.0])

    # b: 1/62 + 1/61, c: 1/63 + 1/62, a: 1/61.
    assert [hit["text"] for hit in fused] == ["b", "c", "a"]
    assert fused[0]["retrieval_method"] == "hybrid"
    assert fused[0]["bm25_score"] == 5.0
    assert fused[1]["bm25_score"] == 3.0
    assert fused[2]["bm25_score"] == 0.0
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"] > fused[2]["rrf_score"]
    with pytest.raises(ValueError):
        rrf_fuse(hits[:2], [1.0, 2.0, 3.0])


def test_hybrid_fuses_candidates_and_ignores_threshold(monkeypatch):
    captured = {}

    class Embeddings:
        def embed_query(self, question):
            return [1.0, 0.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            captured["k"] = k
            documents = [
                Document(page_content="数据库 索引 优化",
                         metadata={"chunk_id": "a#chunk_001"}),
                Document(page_content="注意力 注意力 注意力",
                         metadata={"chunk_id": "b#chunk_002"}),
                Document(page_content="注意力 权重",
                         metadata={"chunk_id": "c#chunk_003"}),
            ]
            return [(documents[0], 0.1), (documents[1], 0.5), (documents[2], 1.5)]

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("chains.answer_chain.generate_answer",
                        lambda **kwargs: "answer")
    settings = Settings(_env_file=None, similarity_threshold=0.99)

    result = answer_question(
        "注意力", settings, top_k=2, candidate_pool=50,
        retrieval_method="hybrid",
    )

    assert captured["k"] == 50
    assert result["retrieval_method"] == "hybrid"
    hits = result["vector_hits"]
    # b wins with vector + BM25 support; vector-only a is pushed below c.
    assert [hit["chunk_id"] for hit in hits] == ["b#chunk_002", "c#chunk_003"]
    assert hits[0]["similarity"] > 0 and hits[0]["bm25_score"] > 0
    assert hits[1]["similarity"] == 0 and hits[1]["bm25_score"] > 0


def test_hybrid_pool_is_clamped_to_top_k(monkeypatch):
    captured = {}

    class Embeddings:
        def embed_query(self, question):
            return [1.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            captured["k"] = k
            return []

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("chains.answer_chain.generate_answer",
                        lambda **kwargs: "answer")

    answer_question(
        "test", Settings(_env_file=None), top_k=100, candidate_pool=20,
        retrieval_method="hybrid",
    )

    assert captured["k"] == 100


def test_settings_candidate_pool_validation():
    assert Settings(_env_file=None).candidate_pool_size == 100
    with pytest.raises(ValueError, match="CANDIDATE_POOL_SIZE"):
        Settings(_env_file=None, candidate_pool_size=10)
    with pytest.raises(ValueError, match="CANDIDATE_POOL_SIZE"):
        Settings(_env_file=None, candidate_pool_size=501)


def test_api_candidate_pool_validation():
    assert WebRequestHandler._read_retrieval_params({"candidate_pool": 50}) == (
        None, None, None, 50,
    )
    for bad in (10, 501, "50", True, 1.5):
        with pytest.raises(RequestError):
            WebRequestHandler._read_retrieval_params({"candidate_pool": bad})
