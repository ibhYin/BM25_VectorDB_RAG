from contextlib import contextmanager

import pytest

from app.config import Settings
from app.service import answer_question, retrieve_question
from tests.test_reranker import Response


@pytest.mark.parametrize("method", ["vector", "bm25", "hybrid"])
def test_shared_pipeline_expands_candidates_reranks_outside_lock_and_uses_standalone_query(monkeypatch, method):
    held = []
    queries = []
    @contextmanager
    def read():
        held.append(True)
        try: yield
        finally: held.pop()
    def recall(*args):
        assert held
        question, settings, actual_method, limit = args[:4]
        assert limit == 50 and actual_method == method
        queries.append(question)
        hits = [{"chunk_id": f"doc#chunk_{i}", "rank": i + 1, "text": f"candidate-{i}",
                 "title": "doc", "doc_id": "doc", "source": "doc.md", "retrieval_method": method} for i in range(limit)]
        args[-1]("vector_search", "completed", vector_hits=hits, matches=len(hits))
        return hits
    def post(*a, **kwargs):
        assert not held, "remote reranking must never hold the knowledge-base read lock"
        assert kwargs["json"]["query"] == "改写后的独立问题"
        return Response({"results": [{"index": i, "relevance_score": i / 100} for i in range(50)]})
    monkeypatch.setattr("app.service.knowledge_base_lock.read", read)
    monkeypatch.setattr("app.service._access_context", lambda *a: (None, None))
    monkeypatch.setattr("app.service._retrieve_locked", recall)
    monkeypatch.setattr("rag.reranker.requests.post", post)
    monkeypatch.setattr("chains.answer_chain.condense_question", lambda *a, **kw: "改写后的独立问题")
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kw: "mock answer")
    settings = Settings(_env_file=None, retrieval_method=method, rerank_enabled=True, rerank_api_key="fake")
    answer = answer_question("详细解释这个概念", settings, history=[{"question": "previous", "answer": "context"}])
    hits = retrieve_question("改写后的独立问题", settings)
    assert hits == answer["vector_hits"]
    assert len(hits) == 20 and hits[0]["chunk_id"] == "doc#chunk_49"
    assert queries == ["改写后的独立问题"] * 2


def test_aggregate_explicitly_disables_rerank(monkeypatch):
    from app.aggregate import run_aggregate
    def retrieve(question, settings, **kwargs):
        assert settings.rerank_enabled is False
        return []
    monkeypatch.setattr("app.aggregate.retrieve_question", retrieve)
    run_aggregate("theme", Settings(_env_file=None, rerank_enabled=True), sample_size=200, top_categories=3)


def test_all_filtered_candidates_skip_answer_generation(monkeypatch):
    monkeypatch.setattr("app.service._access_context", lambda *a: (None, None))
    monkeypatch.setattr("app.service._retrieve_ranked", lambda *a: ([], {"status": "completed"}))
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kw: pytest.fail("no evidence, no generation"))
    settings = Settings(_env_file=None, rerank_enabled=True, rerank_score_filter_enabled=True, rerank_score_threshold=.9)
    result = answer_question("解释注意力机制", settings)
    assert result["vector_hits"] == [] and "没有达到" in result["answer"]
