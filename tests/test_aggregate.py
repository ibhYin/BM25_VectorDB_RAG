from __future__ import annotations

import time
from threading import Event

import pytest

from app.access import UserContext
from app.aggregate import count_by_doc, greedy_cluster, run_aggregate
from app.aggregate_jobs import AggregateJobStore
from app.config import Settings
from app.web import RequestError, WebRequestHandler
from chains.aggregate_chain import parse_extraction_json


def test_parse_extraction_json_tolerates_fences_and_prose() -> None:
    text = (
        '结果如下：\n```json\n{"items": [{"chunk_id": "c1", "relevant": true, '
        '"category": "逾期付款", "evidence": "原文"}]}\n```'
    )

    assert parse_extraction_json(text) == [
        {
            "chunk_id": "c1",
            "relevant": True,
            "category": "逾期付款",
            "evidence": "原文",
        }
    ]


def test_parse_extraction_json_drops_broken_records() -> None:
    assert parse_extraction_json("not json") == []
    text = (
        '{"items": [{"relevant": true, "category": "缺 chunk_id"}, '
        '{"chunk_id": "c2", "relevant": true, "category": ""}]}'
    )

    assert parse_extraction_json(text) == []


def test_greedy_cluster_groups_close_vectors() -> None:
    clusters = greedy_cluster([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], threshold=0.85)

    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


def test_count_by_doc_deduplicates_documents() -> None:
    items = [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": ""}]

    assert count_by_doc(items) == 2


HITS = [
    {
        "chunk_id": "c1",
        "doc_id": "doc-a",
        "title": "合同A",
        "source": "a.pdf",
        "text": "买方逾期付款超过30日",
    },
    {
        "chunk_id": "c2",
        "doc_id": "doc-b",
        "title": "合同B",
        "source": "b.pdf",
        "text": "买方未按期支付货款",
    },
    {
        "chunk_id": "c3",
        "doc_id": "doc-a",
        "title": "合同A",
        "source": "a.pdf",
        "text": "标的物质量不符合约定",
    },
    {
        "chunk_id": "c4",
        "doc_id": "doc-c",
        "title": "合同C",
        "source": "c.pdf",
        "text": "承包人擅自转包",
    },
]

CATEGORIES = {
    "c1": "买方逾期付款",
    "c2": "逾期支付货款",
    "c3": "质量不符合约定",
    "c4": "擅自转包",
}

VECTORS = {
    "买方逾期付款": [1.0, 0.0, 0.0],
    "逾期支付货款": [0.98, 0.2, 0.0],
    "质量不符合约定": [0.0, 1.0, 0.0],
    "擅自转包": [0.0, 0.0, 1.0],
}


def fake_extract(question, chunks, *, model_name, openai_api_key="", openai_base_url=""):
    return [
        {
            "chunk_id": hit["chunk_id"],
            "relevant": True,
            "category": CATEGORIES[hit["chunk_id"]],
            "evidence": hit["text"],
        }
        for hit in chunks
    ]


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [VECTORS[text] for text in texts]


def test_run_aggregate_counts_distinct_documents(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.aggregate.retrieve_question", lambda *args, **kwargs: list(HITS)
    )
    monkeypatch.setattr("app.aggregate.extract_categories", fake_extract)
    monkeypatch.setattr(
        "app.aggregate.get_embeddings", lambda *args, **kwargs: FakeEmbeddings()
    )
    settings = Settings(
        _env_file=None,
        aggregate_batch_size=2,
        aggregate_max_workers=2,
        aggregate_similarity_threshold=0.85,
        max_concurrent_answers=2,
    )

    result = run_aggregate(
        "违约条款有哪些情形", settings, sample_size=4, top_categories=3
    )

    assert result["sample"] == {"chunks": 4, "docs": 3}
    top = result["categories"][0]
    assert top["label"] in {"买方逾期付款", "逾期支付货款"}
    # 同一份文档的两个切片只算一篇；同义类别合并后跨两篇文档。
    assert top["doc_count"] == 2
    assert top["chunk_count"] == 2
    assert {item["doc_id"] for item in top["evidence"]} == {"doc-a", "doc-b"}
    assert result["meta"]["relevant_chunks"] == 4
    assert any("不代表全库" in warning for warning in result["meta"]["warnings"])


def test_run_aggregate_returns_empty_result_without_hits(monkeypatch) -> None:
    monkeypatch.setattr("app.aggregate.retrieve_question", lambda *args, **kwargs: [])
    settings = Settings(_env_file=None)

    result = run_aggregate("没有结果", settings, sample_size=10, top_categories=3)

    assert result["sample"] == {"chunks": 0, "docs": 0}
    assert result["categories"] == []
    assert any("召回结果为空" in warning for warning in result["meta"]["warnings"])


def test_run_aggregate_surfaces_degraded_recall(monkeypatch) -> None:
    def degraded_retrieve(*args, **kwargs):
        kwargs["progress"]("vector_search", "degraded", {
            "recall_detail": {
                "status": "degraded", "succeeded_paths": ["bm25"], "failed_paths": ["vector"],
            },
        })
        return []

    monkeypatch.setattr("app.aggregate.retrieve_question", degraded_retrieve)
    result = run_aggregate("降级样本", Settings(_env_file=None), sample_size=10, top_categories=3)

    assert result["meta"]["recall"]["status"] == "degraded"
    assert any("本次仅使用 BM25" in warning for warning in result["meta"]["warnings"])


def _wait_for_status(store, job_id, statuses, timeout=2.0):
    deadline = time.monotonic() + timeout
    result = store.get_for_user(job_id)
    while time.monotonic() < deadline and result["status"] not in statuses:
        time.sleep(0.01)
        result = store.get_for_user(job_id)
    return result


def test_aggregate_job_store_tracks_progress(monkeypatch) -> None:
    finished = Event()

    def fake_run(question, settings, *, progress, **kwargs):
        progress("retrieve", {"done": 1, "total": 1})
        progress("extract", {"done": 2, "total": 2})
        finished.set()
        return {
            "question": question,
            "sample": {"chunks": 1, "docs": 1},
            "categories": [],
            "meta": {"warnings": []},
        }

    monkeypatch.setattr("app.aggregate.run_aggregate", fake_run)
    store = AggregateJobStore(Settings(_env_file=None, max_concurrent_answers=1))

    created = store.create("测试问题", {"sample_size": 10, "top_categories": 3})
    assert finished.wait(timeout=2)
    result = _wait_for_status(store, created["id"], {"completed", "failed"})

    assert result["status"] == "completed"
    assert result["stage"] == "done"
    assert result["result"]["question"] == "测试问题"
    assert "user" not in result


def test_aggregate_job_is_scoped_to_owner(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.aggregate.run_aggregate",
        lambda question, settings, **kwargs: {
            "question": question,
            "sample": {"chunks": 0, "docs": 0},
            "categories": [],
            "meta": {"warnings": []},
        },
    )
    store = AggregateJobStore(Settings(_env_file=None, max_concurrent_answers=1))
    alice = UserContext("alice", frozenset())
    bob = UserContext("bob", frozenset())

    created = store.create("测试问题", {"sample_size": 10, "top_categories": 3}, alice)

    with pytest.raises(PermissionError):
        store.get_for_user(created["id"], bob)


def test_read_aggregate_params_rejects_out_of_range_values() -> None:
    for payload in ({"sample_size": 5}, {"top_categories": 0}, {"top_categories": 11}):
        with pytest.raises(RequestError) as error:
            WebRequestHandler._read_aggregate_params(payload)
        assert error.value.status == 400
