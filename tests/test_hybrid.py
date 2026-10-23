import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.service import answer_question
from app.web import RequestError, WebRequestHandler
from rag.retriever import (
    build_vector_hits,
    rerank_bm25_scores,
    rrf_fuse_paths,
    weighted_fuse_paths,
)


def test_rerank_scores_only_matching_texts():
    scores = rerank_bm25_scores(
        ["注意力 机制 Attention", "数据库 索引", "注意力 注意力 注意力"],
        "注意力",
    )

    assert scores[1] == 0.0
    assert scores[0] > 0 and scores[2] > scores[0]


def _vector_hit(chunk_id, similarity):
    return {"rank": 0, "chunk_id": chunk_id, "text": chunk_id, "similarity": similarity}


def _bm25_hit(chunk_id, score):
    return {
        "rank": 0,
        "chunk_id": chunk_id,
        "text": chunk_id,
        "retrieval_method": "bm25",
        "bm25_score": score,
    }


def test_rrf_fuse_paths_uses_real_ranks_from_both_paths():
    vector = [_vector_hit("a", 0.9), _vector_hit("b", 0.8), _vector_hit("c", 0.7)]
    bm25 = [_bm25_hit("b", 5.0), _bm25_hit("c", 3.0)]

    fused = rrf_fuse_paths(vector, bm25, k=60)

    # b: 1/62 + 1/61, c: 1/63 + 1/62, a: 1/61.
    assert [hit["chunk_id"] for hit in fused] == ["b", "c", "a"]
    assert fused[0]["retrieval_method"] == "hybrid"
    assert fused[0]["vector_rank"] == 2 and fused[0]["bm25_rank"] == 1
    assert fused[0]["sources"] == ["vector", "bm25"]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"] > fused[2]["rrf_score"]
    assert fused[2]["sources"] == ["vector"] and fused[2]["bm25_rank"] is None


def test_rrf_fuse_paths_keeps_bm25_only_candidates():
    fused = rrf_fuse_paths([_vector_hit("v", 0.9)], [_bm25_hit("k", 5.0)])

    assert [hit["chunk_id"] for hit in fused] == ["v", "k"]
    assert fused[1]["sources"] == ["bm25"]
    assert fused[1]["vector_rank"] is None
    assert "similarity" not in fused[1]
    assert fused[1]["bm25_rank"] == 1


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
    monkeypatch.setattr("app.service.search_bm25", lambda *args, **kwargs: build_vector_hits(
        [
            (Document(page_content="注意力 注意力 注意力",
                      metadata={"chunk_id": "b#chunk_002"}), 5.0),
            (Document(page_content="注意力 权重",
                      metadata={"chunk_id": "c#chunk_003"}), 3.0),
        ],
        retrieval_method="bm25",
    ))
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
    assert hits[0]["sources"] == ["vector", "bm25"]
    assert hits[0]["vector_rank"] == 2 and hits[0]["bm25_rank"] == 1


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


def test_weighted_fuse_paths_alpha_one_follows_vector_order():
    vector = [_vector_hit("a", 0.9), _vector_hit("b", 0.5), _vector_hit("c", 0.1)]

    fused = weighted_fuse_paths(vector, [], alpha=1.0)

    assert [hit["chunk_id"] for hit in fused] == ["a", "b", "c"]
    assert fused[0]["retrieval_method"] == "hybrid"
    assert fused[0]["weighted_score"] == pytest.approx(0.9)
    assert [hit["rank"] for hit in fused] == [1, 2, 3]


def test_weighted_fuse_paths_alpha_zero_follows_bm25_order():
    vector = [_vector_hit("a", 0.9), _vector_hit("b", 0.5), _vector_hit("c", 0.1)]
    bm25 = [_bm25_hit("b", 5.0), _bm25_hit("c", 3.0)]

    fused = weighted_fuse_paths(vector, bm25, alpha=0.0)

    # BM25 归一化后 b=1.0、c=0.6、a 缺席=0.0
    assert [hit["chunk_id"] for hit in fused] == ["b", "c", "a"]
    assert fused[0]["weighted_score"] == pytest.approx(1.0)
    assert fused[1]["weighted_score"] == pytest.approx(0.6)
    assert fused[0]["bm25_score"] == 5.0


def test_weighted_fuse_paths_balances_both_signals():
    vector = [_vector_hit("vector-only", 1.0), _vector_hit("both", 0.6)]
    bm25 = [_bm25_hit("both", 10.0), _bm25_hit("bm25-only", 10.0)]

    fused = weighted_fuse_paths(vector, bm25, alpha=0.5)

    # both: 0.5*0.6 + 0.5*1.0 = 0.8 > vector-only 0.5 = bm25-only 0.5
    assert [hit["chunk_id"] for hit in fused] == ["both", "vector-only", "bm25-only"]
    assert fused[0]["weighted_score"] == pytest.approx(0.8)


def test_weighted_fuse_paths_treats_absent_path_as_zero():
    vector = [_vector_hit("a", 0.9), _vector_hit("b", 0.4)]

    fused = weighted_fuse_paths(vector, [], alpha=0.5)

    assert [hit["chunk_id"] for hit in fused] == ["a", "b"]
    assert fused[0]["weighted_score"] == pytest.approx(0.45)


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
    monkeypatch.setattr("app.service.search_bm25", lambda *args, **kwargs: build_vector_hits(
        [
            (Document(page_content="注意力 注意力 注意力",
                      metadata={"chunk_id": "b#chunk_002"}), 5.0),
            (Document(page_content="注意力 权重",
                      metadata={"chunk_id": "c#chunk_003"}), 3.0),
        ],
        retrieval_method="bm25",
    ))
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


def test_hybrid_keeps_keyword_only_hits(monkeypatch):
    class Embeddings:
        def embed_query(self, question):
            return [1.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            return [(Document(page_content="设备的日常保养",
                              metadata={"chunk_id": "vector#chunk_001"}), 0.1)]

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("app.service.search_bm25", lambda *args, **kwargs: build_vector_hits(
        [(Document(page_content="ZXQ7429 故障复位步骤",
                   metadata={"chunk_id": "keyword#chunk_001"}), 4.0)],
        retrieval_method="bm25",
    ))
    monkeypatch.setattr("chains.answer_chain.generate_answer",
                        lambda **kwargs: "answer")

    result = answer_question("ZXQ7429", Settings(_env_file=None), top_k=2,
                             candidate_pool=20, retrieval_method="hybrid")

    assert [hit["chunk_id"] for hit in result["vector_hits"]] == [
        "vector#chunk_001", "keyword#chunk_001",
    ]
    assert result["vector_hits"][1]["sources"] == ["bm25"]


def test_hybrid_reports_recall_trace(monkeypatch):
    class Embeddings:
        def embed_query(self, question):
            return [1.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            return [(Document(page_content="设备的日常保养",
                              metadata={"chunk_id": "vector#chunk_001"}), 0.1)]

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("app.service.search_bm25", lambda *args, **kwargs: build_vector_hits(
        [(Document(page_content="ZXQ7429 故障复位步骤",
                   metadata={"chunk_id": "keyword#chunk_001"}), 4.0)],
        retrieval_method="bm25",
    ))
    monkeypatch.setattr("chains.answer_chain.generate_answer",
                        lambda **kwargs: "answer")
    stages = {}

    def progress(stage, status, detail):
        if stage == "vector_search" and status == "completed":
            stages.update(detail)

    answer_question("ZXQ7429", Settings(_env_file=None), top_k=2,
                    candidate_pool=20, retrieval_method="hybrid",
                    progress=progress)

    trace = stages["retrieval_trace"]
    assert trace["vector_count"] == 1
    assert trace["bm25_count"] == 1
    assert trace["union_count"] == 2
    assert trace["final_count"] == 2
    by_id = {item["chunk_id"]: item for item in trace["candidates"]}
    assert by_id["keyword#chunk_001"]["sources"] == ["bm25"]
    assert by_id["keyword#chunk_001"]["vector_rank"] is None
    assert by_id["keyword#chunk_001"]["final_rank"] == 2
    assert by_id["vector#chunk_001"]["final_rank"] == 1


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
