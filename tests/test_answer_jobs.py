from threading import Event

import pytest

from app.answer_jobs import AnswerJobStore
from app.config import Settings


def test_answer_job_exposes_stage_progress_and_vector_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = Event()

    def fake_answer(question, settings, progress):
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
