import json

import pytest
import requests

from app.config import Settings
from rag.reranker import rerank_hits


HITS = [{"chunk_id": f"doc#chunk_{i}", "rank": i + 1, "text": f"private-document-{i}",
         "retrieval_method": "hybrid"} for i in range(3)]


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
        self.headers = {"x-siliconcloud-trace-id": "provider-trace"}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def json(self): return self.body


def settings(**kwargs):
    return Settings(_env_file=None, rerank_enabled=True, rerank_api_key="fake-private-key", **kwargs)


def test_reranker_maps_indices_preserves_source_ranks_and_never_changes_original(monkeypatch):
    def post(url, **kwargs):
        assert kwargs["json"]["documents"] == [hit["text"] for hit in HITS]
        assert kwargs["json"]["top_n"] == 3 and not kwargs["json"]["return_documents"]
        assert kwargs["timeout"] == 10 and kwargs["allow_redirects"] is False
        assert kwargs["headers"]["X-Trace-Id"]
        return Response({"results": [{"index": 2, "relevance_score": .9},
            {"index": 0, "relevance_score": .2}, {"index": 1, "relevance_score": .5}]})
    monkeypatch.setattr("rag.reranker.requests.post", post)
    rows, detail = rerank_hits("private-query", HITS, settings(), 2)
    assert [row["chunk_id"] for row in rows] == ["doc#chunk_2", "doc#chunk_1"]
    assert rows[0]["rank"] == 1 and rows[0]["fusion_rank"] == 3
    assert "rerank_rank" not in HITS[0]
    assert detail["status"] == "completed" and detail["output_count"] == 2
    assert "private-document" not in json.dumps(detail)


@pytest.mark.parametrize("results", [[], [{"index": 0, "relevance_score": .4}] * 3,
    [{"index": 99, "relevance_score": .5}] * 3,
    [{"index": i, "relevance_score": float("nan")} for i in range(3)],
    [{"index": i, "relevance_score": True} for i in range(3)]])
def test_bad_response_falls_back_atomically(monkeypatch, results):
    monkeypatch.setattr("rag.reranker.requests.post", lambda *a, **kw: Response({"results": results}))
    rows, detail = rerank_hits("query", HITS, settings(), 2)
    assert rows == HITS[:2]
    assert detail["status"] == "degraded" and detail["error_type"] == "invalid_response"


def test_timeout_is_not_retried_and_diagnostics_do_not_leak(monkeypatch, caplog):
    calls = []
    def post(*a, **kw):
        calls.append(1)
        raise requests.Timeout("fake-private-key private-query private-document-0")
    monkeypatch.setattr("rag.reranker.requests.post", post)
    rows, detail = rerank_hits("private-query", HITS, settings(), 2)
    assert len(calls) == 1 and rows == HITS[:2]
    assert detail["error_type"] == "timeout"
    assert not any(value in caplog.text for value in ("fake-private-key", "private-query", "private-document"))


def test_rate_limit_keeps_provider_code_without_raw_error_body(monkeypatch):
    monkeypatch.setattr("rag.reranker.requests.post", lambda *a, **kw: Response({"code": 50505,
        "message": "fake-private-key private-document-0"}, 429))
    _, detail = rerank_hits("query", HITS, settings(), 2)
    assert detail["http_status"] == 429 and detail["error_code"] == "50505"
    assert detail["provider_request_id"] == "provider-trace"
    assert "fake-private-key" not in json.dumps(detail)


def test_score_filter_can_return_empty_without_fallback(monkeypatch):
    monkeypatch.setattr("rag.reranker.requests.post", lambda *a, **kw: Response({"results": [
        {"index": i, "relevance_score": .1} for i in range(3)]}))
    rows, detail = rerank_hits("query", HITS, settings(rerank_score_filter_enabled=True,
                                rerank_score_threshold=.5), 2)
    assert rows == [] and detail["status"] == "completed" and detail["filtered_count"] == 3


def test_disabled_empty_and_single_candidate_skip_without_api(monkeypatch):
    monkeypatch.setattr("rag.reranker.requests.post", lambda *a, **kw: pytest.fail("must skip"))
    assert rerank_hits("q", HITS, Settings(_env_file=None), 2)[1]["reason"] == "disabled"
    assert rerank_hits("q", [], settings(), 2)[1]["reason"] == "no_candidates"
    assert rerank_hits("q", HITS[:1], settings(), 2)[1]["reason"] == "single_candidate"
