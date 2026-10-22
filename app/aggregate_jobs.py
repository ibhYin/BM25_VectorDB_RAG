from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.access import UserContext
from app.config import Settings


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AggregateJobStore:
    """进程内聚合任务。每次请求都重新计算，任务只负责异步执行与进度。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_answers),
            thread_name_prefix="rag-aggregate-job",
        )

    def create(
        self,
        question: str,
        params: dict[str, Any],
        user: UserContext | None = None,
    ) -> dict[str, Any]:
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
                "stage": "queued",
                "progress": {},
                "error": "",
                "result": None,
                "params": dict(params),
                "user": user,
                "created_at": now,
                "started_at": "",
                "finished_at": "",
            }
            self._jobs[job_id] = job
            self._prune_locked()

        self._executor.submit(self._run, job_id)
        return self.get_for_user(job_id, user)

    def get_for_user(
        self, job_id: str, user: UserContext | None = None
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Aggregate job not found")
            owner = job.get("user")
            if owner is not None and (user is None or owner.user_id != user.user_id):
                raise PermissionError("Aggregate job is not available")
            return self._snapshot(job)

    def _run(self, job_id: str) -> None:
        from app.aggregate import run_aggregate

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["stage"] = "retrieve"
            job["started_at"] = utc_now()
            question = job["question"]
            params = dict(job["params"])
            user = job.get("user")

        def progress(stage: str, detail: dict[str, object]) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is None:
                    return
                current["stage"] = stage
                current["progress"] = dict(detail)

        try:
            result = run_aggregate(
                question,
                self._settings,
                user=user,
                progress=progress,
                **params,
            )
        except Exception as exc:
            logger.warning("Aggregate job %s failed", job_id, exc_info=True)
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current["status"] = "failed"
                    current["stage"] = "failed"
                    current["error"] = str(exc)
                    current["finished_at"] = utc_now()
            return

        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                current.update(
                    status="completed",
                    stage="done",
                    result=result,
                    finished_at=utc_now(),
                )

    @staticmethod
    def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(job)
        snapshot.pop("user", None)
        snapshot["params"] = dict(job.get("params") or {})
        snapshot["progress"] = dict(job.get("progress") or {})
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


_aggregate_store: AggregateJobStore | None = None
_aggregate_store_lock = threading.Lock()


def get_aggregate_store() -> AggregateJobStore:
    global _aggregate_store
    with _aggregate_store_lock:
        if _aggregate_store is None:
            from app.config import get_settings

            _aggregate_store = AggregateJobStore(get_settings())
        return _aggregate_store
