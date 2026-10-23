from threading import Event

import pytest

from app.answer_jobs import AnswerJobStore
from app.config import Settings


def test_answer_job_exposes_stage_progress_and_vector_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = Event()

    def fake_answer(question, settings, progress, **kwargs):
        progress("encode", "running", {"model": "embedding"})
        progress(
            "encode",
            "completed",
            {
                "model": "embedding",
                "dimensions": 2,
                "duration_ms": 4,
                "vector_preview": [0.1, 0.2],
            },
        )
        hits = [
            {
                "rank": 1,
                "distance": 0.25,
                "text": "原始切片文本",
                "chunk_id": "doc#chunk_001",
                "doc_id": "doc",
                "title": "文档",
                "source": "doc.md",
                "page": None,
                "page_label": "",
                "section_path": "",
            }
        ]
        progress("vector_search", "running", {"top_k": 4})
        progress(
            "vector_search",
            "completed",
            {
                "top_k": 4,
                "matches": 1,
                "duration_ms": 2,
                "vector_hits": hits,
            },
        )
        progress("answer", "running", {"model": "llm"})
        progress("answer", "completed", {"model": "llm", "duration_ms": 8})
        finished.set()
        return {
            "answer": "答案",
            "vector_hits": hits,
        }

    monkeypatch.setattr("app.service.answer_question", fake_answer)
    store = AnswerJobStore(Settings(_env_file=None, max_concurrent_answers=1))

    created = store.create("测试问题")
    assert finished.wait(timeout=2)
    result = store.get(created["id"])

    assert result["status"] == "completed"
    assert result["answer"] == "答案"
    assert result["vector_hits"][0]["text"] == "原始切片文本"
    assert result["stages"]["encode"]["detail"]["dimensions"] == 2
    assert result["stages"]["vector_search"]["detail"]["matches"] == 1
    assert "vector_hits" not in result["stages"]["vector_search"]["detail"]


def test_answer_job_persists_light_retrieval_summary(monkeypatch, tmp_path) -> None:
    from app.conversations import ConversationStore

    finished = Event()

    def fake_answer(question, settings, progress, **kwargs):
        progress("vector_search", "completed", {
            "retrieval_method": "hybrid",
            "matches": 1,
            "vector_candidates": 2,
            "bm25_candidates": 1,
            "union_candidates": 3,
            "retrieval_trace": {"candidates": [{"chunk_id": "x"}]},
            "vector_hits": [{"chunk_id": "x", "text": "正文"}],
        })
        finished.set()
        return {"answer": "答案", "vector_hits": [{"chunk_id": "x"}]}

    conversations = ConversationStore(tmp_path / "conversations.json")
    conversation = conversations.create("open-mode")
    monkeypatch.setattr("app.service.answer_question", fake_answer)
    monkeypatch.setattr(
        "app.conversations.get_conversation_store", lambda: conversations
    )
    store = AnswerJobStore(Settings(_env_file=None, max_concurrent_answers=1))

    store.create("测试问题", conversation_id=conversation["id"])
    store._executor.shutdown(wait=True)

    assert finished.is_set()
    turn = conversations.get(conversation["id"], "open-mode")["turns"][0]
    assert turn["retrieval"]["union_candidates"] == 3
    assert turn["retrieval"]["bm25_candidates"] == 1
    assert "retrieval_trace" not in turn["retrieval"]
    assert "vector_hits" not in turn["retrieval"]


def test_answer_job_streams_partial_answer_into_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = Event()
    mid = Event()
    proceed = Event()

    def fake_answer(question, settings, progress, **kwargs):
        on_token = kwargs.get("on_token")
        on_token("你")
        on_token("好")
        on_token("世界")
        progress("answer", "running", {"model": "llm"})
        mid.set()
        proceed.wait(timeout=2)
        progress("answer", "completed", {"model": "llm", "duration_ms": 8})
        finished.set()
        return {"answer": "你好世界", "vector_hits": []}

    monkeypatch.setattr("app.service.answer_question", fake_answer)
    store = AnswerJobStore(Settings(_env_file=None, max_concurrent_answers=1))
    created = store.create("测试问题")

    # 生成中途：部分答案已可被轮询看到。
    assert mid.wait(timeout=2)
    mid_state = store.get(created["id"])
    assert mid_state["answer_streaming"] is True
    assert mid_state["answer"] == "你好世界"

    proceed.set()
    assert finished.wait(timeout=2)
    final = store.get(created["id"])
    assert final["status"] == "completed"
    assert final["answer"] == "你好世界"
    assert final["answer_streaming"] is False
