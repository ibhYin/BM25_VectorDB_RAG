from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.access import UserContext


logger = logging.getLogger(__name__)


# "condense" 排在最前：dict 顺序即前端渲染顺序。没有历史时后端会报 skipped，
# 前端对 skipped 已有「已跳过 / -」的展示，不需要动态增删阶段。
# "system" 是系统问答专用通道（格式/模型/版本/知识库清单），命中时其余阶段报 skipped。
STAGES = ("condense", "system", "encode", "vector_search", "rerank", "answer")


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

    def create(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        retrieval_method: str | None = None,
        candidate_pool: int | None = None,
        fusion_method: str | None = None,
        fusion_k: int | None = None,
        fusion_alpha: float | None = None,
        user: UserContext | None = None,
        conversation_id: str | None = None,
        llm_model: str | None = None,
        rerank_options: dict | None = None,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question is required")
        method = self._settings.retrieval_method if retrieval_method is None else retrieval_method
        if method not in ("vector", "bm25", "hybrid"):
            raise ValueError("retrieval_method must be vector, bm25 or hybrid")
        if fusion_method is not None and fusion_method not in ("rrf", "weighted"):
            raise ValueError("fusion_method must be rrf or weighted")
        run_settings = self._settings.model_copy(deep=True)
        for key, value in (rerank_options or {}).items():
            if key.startswith("rerank_") and key in type(run_settings).model_fields:
                setattr(run_settings, key, value)
        if llm_model is not None:
            run_settings.llm_model = llm_model

        with self._lock:
            job_id = uuid.uuid4().hex
            now = utc_now()
            job: dict[str, Any] = {
                "id": job_id,
                "question": question,
                "llm_model": run_settings.llm_model,
                "status": "queued",
                "error": "",
                "answer": "",
                "answer_streaming": False,
                "vector_hits": [],
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "retrieval_method": method,
                "candidate_pool": candidate_pool,
                "fusion_method": fusion_method,
                "fusion_k": fusion_k,
                "fusion_alpha": fusion_alpha,
                "user": user,
                "conversation_id": conversation_id or "",
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

        self._executor.submit(self._run, job_id, run_settings)
        return self.get_for_user(job_id, user)

    def get(self, job_id: str) -> dict[str, Any]:
        return self.get_for_user(job_id)

    def get_for_user(self, job_id: str, user: UserContext | None = None) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Answer job not found")
            owner = job.get("user")
            if owner is not None and (user is None or owner.user_id != user.user_id):
                raise PermissionError("Answer job is not available")
            return self._snapshot(job)

    def _run(self, job_id: str, settings: Settings) -> None:
        from app.conversations import get_conversation_store, owner_of
        from app.service import answer_question

        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = utc_now()
            question = job["question"]
            owner = owner_of(job.get("user"))
            conversation_id = job.get("conversation_id") or ""
            retrieval_method = job["retrieval_method"]

        # 读历史放在任务锁之外，避免和 500ms 一次的状态轮询相互串行；
        # 对话已被删除时 history_for 返回空，按「无历史」处理。
        history = (
            get_conversation_store().history_for(conversation_id, owner)
            if conversation_id
            else []
        )

        def progress(stage: str, status: str, detail: dict[str, object]) -> None:
            with self._lock:
                job = self._jobs[job_id]
                detail = dict(detail)
                if stage in ("vector_search", "rerank") and "vector_hits" in detail:
                    job["vector_hits"] = list(detail.pop("vector_hits"))
                if stage == "rerank":
                    detail["task_id"] = job_id
                stage_state = job["stages"][stage]
                stage_state["status"] = status
                stage_state["detail"] = detail
                if status == "running" and not stage_state["started_at"]:
                    stage_state["started_at"] = utc_now()
                if status in {"completed", "degraded", "failed", "skipped"}:
                    stage_state["finished_at"] = utc_now()

        # 流式增量：把生成的片段不断写进 job["answer"]，前端轮询就能边生成边显示。
        def on_token(delta: str) -> None:
            if not delta:
                return
            with self._lock:
                self._jobs[job_id]["answer"] += delta
                self._jobs[job_id]["answer_streaming"] = True

        try:
            result = answer_question(
                question,
                settings,
                progress=progress,
                top_k=job.get("top_k"),
                similarity_threshold=job.get("similarity_threshold"),
                retrieval_method=retrieval_method,
                candidate_pool=job.get("candidate_pool"),
                fusion_method=job.get("fusion_method"),
                fusion_k=job.get("fusion_k"),
                fusion_alpha=job.get("fusion_alpha"),
                user=job.get("user"),
                history=history,
                on_token=on_token,
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
                retrieval_method=result.get("retrieval_method", retrieval_method),
                fusion_method=result.get("fusion_method"),
                answer=result.get("answer", ""),
                answer_streaming=False,
                vector_hits=result.get("vector_hits", []),
                finished_at=utc_now(),
            )
            # 对话记录只存轻量摘要；完整候选轨迹留在 job 里给当前页面用。
            retrieval_detail = dict(
                (job["stages"].get("vector_search") or {}).get("detail") or {}
            )
            retrieval_detail["llm_model"] = settings.llm_model
            rerank_detail = dict(job["stages"]["rerank"]["detail"])
            rerank_detail["status"] = job["stages"]["rerank"]["status"]
            rerank_detail.pop("candidates", None)
            retrieval_detail["rerank"] = rerank_detail

        # 写对话放在任务锁之外：append 会全量重写文件，占着锁会阻塞状态轮询。
        # 失败只记日志——异常在这里抛出会被 executor 的 future 吞掉，那一轮就静默丢了。
        if conversation_id:
            try:
                get_conversation_store().append_turn(
                    conversation_id,
                    owner,
                    question,
                    result.get("answer", ""),
                    retrieval_method=result.get("retrieval_method", retrieval_method),
                    hits=result.get("vector_hits"),
                    retrieval=retrieval_detail,
                )
            except Exception:
                logger.warning("Could not append turn to conversation", exc_info=True)

    @staticmethod
    def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(job)
        snapshot.pop("user", None)
        snapshot["stages"] = {
            name: {
                **state,
                "detail": dict(state["detail"]),
            }
            for name, state in job["stages"].items()
        }
        snapshot["vector_hits"] = [dict(hit) for hit in job["vector_hits"]]
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
