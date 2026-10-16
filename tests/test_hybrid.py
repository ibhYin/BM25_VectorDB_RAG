import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.service import answer_question
from app.web import RequestError, WebRequestHandler
from rag.retriever import rerank_bm25_scores, rrf_fuse, weighted_fuse


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
    assert WebRequestHandler._read_retrieval_params({"candidate_pool": 50})[
        "candidate_pool"
    ] == 50
    for bad in (10, 501, "50", True, 1.5):
        with pytest.raises(RequestError):
            WebRequestHandler._read_retrieval_params({"candidate_pool": bad})


# --- weighted fusion ------------------------------------------------------


def _hit(text, similarity):
    return {"rank": 1, "text": text, "similarity": similarity}


def test_weighted_fuse_alpha_one_follows_vector_order():
    hits = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.1)]

    fused = weighted_fuse(hits, [0.0, 5.0, 3.0], alpha=1.0)

    assert [hit["text"] for hit in fused] == ["a", "b", "c"]
    assert fused[0]["retrieval_method"] == "hybrid"
    assert fused[0]["weighted_score"] == pytest.approx(0.9)
    assert [hit["rank"] for hit in fused] == [1, 2, 3]


def test_weighted_fuse_alpha_zero_follows_bm25_order():
    hits = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.1)]

    fused = weighted_fuse(hits, [0.0, 5.0, 3.0], alpha=0.0)

    # BM25 归一化后 b=1.0、c=0.6、a=0.0
    assert [hit["text"] for hit in fused] == ["b", "c", "a"]
    assert fused[0]["weighted_score"] == pytest.approx(1.0)
    assert fused[1]["weighted_score"] == pytest.approx(0.6)
    assert fused[0]["bm25_score"] == 5.0


def test_weighted_fuse_balances_both_signals():
    hits = [_hit("vector-only", 1.0), _hit("both", 0.6), _hit("bm25-only", 0.0)]

    fused = weighted_fuse(hits, [0.0, 10.0, 10.0], alpha=0.5)

    # both: 0.5*0.6 + 0.5*1.0 = 0.8 > vector-only 0.5 > bm25-only 0.5
    assert [hit["text"] for hit in fused] == ["both", "vector-only", "bm25-only"]
    assert fused[0]["weighted_score"] == pytest.approx(0.8)


def test_weighted_fuse_handles_all_zero_bm25():
    hits = [_hit("a", 0.9), _hit("b", 0.4)]

    fused = weighted_fuse(hits, [0.0, 0.0], alpha=0.5)

    assert [hit["text"] for hit in fused] == ["a", "b"]
    assert fused[0]["weighted_score"] == pytest.approx(0.45)


def test_weighted_fuse_length_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_fuse([_hit("a", 1.0)], [1.0, 2.0])


def test_hybrid_weighted_fusion_used_when_selected(monkeypatch):
    class Embeddings:
        def embed_query(self, question):
            return [1.0, 0.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
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

    result = answer_question(
        "注意力",
        Settings(_env_file=None),
        top_k=2,
        candidate_pool=50,
        retrieval_method="hybrid",
        fusion_method="weighted",
        fusion_alpha=0.5,
    )

    assert result["fusion_method"] == "weighted"
    hits = result["vector_hits"]
    assert all("weighted_score" in hit for hit in hits)
    assert all("rrf_score" not in hit for hit in hits)


def test_hybrid_rrf_is_default_and_reports_fusion_method(monkeypatch):
    class Embeddings:
        def embed_query(self, question):
            return [1.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            return [(Document(page_content="注意力", metadata={"chunk_id": "a#1"}), 0.2)]

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("chains.answer_chain.generate_answer",
                        lambda **kwargs: "answer")

    result = answer_question(
        "注意力", Settings(_env_file=None), retrieval_method="hybrid",
    )

    assert result["fusion_method"] == "rrf"
    assert "rrf_score" in result["vector_hits"][0]


def test_settings_fusion_defaults_and_validation():
    settings = Settings(_env_file=None)
    assert settings.fusion_method == "rrf"
    assert settings.fusion_k == 60
    assert settings.fusion_alpha == 0.5
    # fusion_method 是 Literal，非法值由 pydantic 拦下（ValidationError 是 ValueError 子类）
    with pytest.raises(ValueError):
        Settings(_env_file=None, fusion_method="bogus")
    with pytest.raises(ValueError, match="FUSION_K"):
        Settings(_env_file=None, fusion_k=0)
    with pytest.raises(ValueError, match="FUSION_ALPHA"):
        Settings(_env_file=None, fusion_alpha=1.5)


def test_api_fusion_param_validation():
    params = WebRequestHandler._read_retrieval_params(
        {"fusion_method": "weighted", "fusion_k": 30, "fusion_alpha": 0.7}
    )
    assert params["fusion_method"] == "weighted"
    assert params["fusion_k"] == 30
    assert params["fusion_alpha"] == 0.7

    for bad in ({"fusion_method": "bogus"}, {"fusion_k": 0}, {"fusion_k": 201},
                {"fusion_k": "30"}, {"fusion_alpha": -0.1}, {"fusion_alpha": 1.1},
                {"fusion_alpha": "0.5"}):
        with pytest.raises(RequestError):
            WebRequestHandler._read_retrieval_params(bad)
