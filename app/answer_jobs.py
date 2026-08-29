from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.config import Settings


STAGES = ("encode", "vector_search", "graph_search", "answer")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnswerJobStore:
    """Process-local answer jobs used to expose real RAG stage progress."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_answers,
            thread_name_prefix="rag-answer",
        )

    def create(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question is required")

        with self._lock:
            job_id = uuid.uuid4().hex
            now = utc_now()
            job: dict[str, Any] = {
                "id": job_id,
                "question": question,
                "status": "queued",
                "error": "",
                "answer": "",
                "vector_hits": [],
                "graph_context": [],
                "created_at": now,
                "started_at": "",
                "finished_at": "",
                "stages": {
                    stage: {
                        "status": "pending",
                        "started_at": "",
                        "finished_at": "",
                        "detail": {},
                    }
                    for stage in STAGES
                },
            }
            self._jobs[job_id] = job
            self._prune_locked()

        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Answer job not found")
            return self._snapshot(job)

    def _run(self, job_id: str) -> None:
        from app.service import answer_question

        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = utc_now()

        def progress(stage: str, status: str, detail: dict[str, object]) -> None:
            with self._lock:
                job = self._jobs[job_id]
                detail = dict(detail)
                if stage == "vector_search" and "vector_hits" in detail:
                    job["vector_hits"] = list(detail.pop("vector_hits"))
                stage_state = job["stages"][stage]
                stage_state["status"] = status
                stage_state["detail"] = detail
                if status == "running" and not stage_state["started_at"]:
                    stage_state["started_at"] = utc_now()
                if status in {"completed", "degraded", "failed"}:
                    stage_state["finished_at"] = utc_now()

        try:
            result = answer_question(
                job["question"],
                self._settings,
                progress=progress,
            )
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = utc_now()
                for stage_state in job["stages"].values():
                    if stage_state["status"] == "running":
                        stage_state["status"] = "failed"
                        stage_state["finished_at"] = job["finished_at"]
            return

        with self._lock:
            job = self._jobs[job_id]
            job.update(
                status="completed",
                answer=result.get("answer", ""),
                vector_hits=result.get("vector_hits", []),
                graph_context=result.get("graph_context", []),
                finished_at=utc_now(),
            )

    @staticmethod
    def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(job)
        snapshot["stages"] = {
            name: {
                **state,
                "detail": dict(state["detail"]),
            }
            for name, state in job["stages"].items()
        }
        snapshot["vector_hits"] = [dict(hit) for hit in job["vector_hits"]]
        snapshot["graph_context"] = list(job["graph_context"])
        return snapshot

    def _prune_locked(self) -> None:
        finished = sorted(
            (
                job
                for job in self._jobs.values()
                if job["status"] in {"completed", "failed"}
            ),
            key=lambda job: job["finished_at"],
            reverse=True,
        )
        for job in finished[100:]:
            self._jobs.pop(job["id"], None)


_answer_store: AnswerJobStore | None = None
_answer_store_lock = threading.Lock()


def get_answer_store() -> AnswerJobStore:
    global _answer_store
    with _answer_store_lock:
        if _answer_store is None:
            from app.config import get_settings

            _answer_store = AnswerJobStore(get_settings())
        return _answer_store
