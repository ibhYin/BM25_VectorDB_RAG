from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as default_email_policy
from pathlib import Path
from typing import Any

from app.config import Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    filename = Path(value.replace("\x00", "")).name.strip()
    if not filename or filename in {".", ".."}:
        raise ValueError("Filename is required")
    if len(filename.encode("utf-8")) > 200:
        raise ValueError("Filename is too long")
    return filename


def parse_multipart(
    content_type: str,
    body: bytes,
    *,
    supported_suffixes: frozenset[str],
    max_files: int,
) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """Parse multipart/form-data upload body.

    Returns (files, fields) where files is [(filename, bytes)] and fields is
    {name: value} for non-file form fields.
    """
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type != "multipart/form-data":
        raise ValueError("Expected multipart/form-data")

    # ponytail: email.parser handles browser multipart bodies without a new
    # dependency; uploads are bounded by MAX_UPLOAD_BYTES before parsing.
    raw_message = (
        b"Content-Type: "
        + content_type.encode("utf-8", errors="strict")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    message = BytesParser(policy=default_email_policy).parsebytes(raw_message)
    if not message.is_multipart():
        raise ValueError("Invalid multipart body")

    files: list[tuple[str, bytes]] = []
    fields: dict[str, str] = {}
    seen_names: set[str] = set()
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if field_name == "files":
            raw_filename = part.get_filename()
            if raw_filename is None:
                raise ValueError("A file part is missing its filename")
            filename = safe_filename(raw_filename)
            key = filename.casefold()
            if key in seen_names:
                raise ValueError(f"Duplicate filename: {filename}")
            if Path(filename).suffix.lower() not in supported_suffixes:
                raise ValueError(f"Unsupported document type: {filename}")
            content = part.get_payload(decode=True)
            if content is None:
                raise ValueError(f"Empty file body: {filename}")
            if not content:
                raise ValueError(f"Empty file: {filename}")
            seen_names.add(key)
            files.append((filename, content))
        elif field_name and part.get_filename() is None:
            # Non-file form field
            raw_value = part.get_payload(decode=True)
            if raw_value is not None:
                fields[field_name] = raw_value.decode("utf-8", errors="replace").strip()

    if not files:
        raise ValueError("At least one supported file is required")
    if len(files) > max_files:
        raise ValueError(f"At most {max_files} files per upload")
    return files, fields


def save_upload_files(
    files: list[tuple[str, bytes]],
    data_dir: Path,
) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for filename, content in files:
        target = data_dir / filename
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        saved_paths.append(target)
    return saved_paths


class UploadJobStore:
    """Persistent single-worker queue for browser file ingestion."""

    def __init__(self, settings: Settings, state_path: Path) -> None:
        self._settings = settings
        self._state_path = state_path
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rag-upload",
        )
        self._load()

    def _new_job(
        self,
        filenames: list[str],
        paths: list[Path],
        structurize: bool = False,
        actor: str = "",
    ) -> dict[str, Any]:
        return {
            "id": uuid.uuid4().hex,
            "filenames": list(filenames),
            "paths": [str(path) for path in paths],
            "structurize": structurize,
            "actor": actor,
            "status": "queued",
            "stage": "queued",
            "error": "",
            "files": 0,
            "documents": 0,
            "chunks": 0,
            "failed_files": [],
            "progress_done": 0,
            "progress_total": len(paths),
            "current_file": "",
            "file_details": [],
            "continue_on_error": True,
            "cancel_requested": False,
            "runtime": {
                "embedding_model": self._settings.embedding_model,
                "knowledge_base_id": self._settings.knowledge_base_id,
                "structurize_provider": "mineru" if structurize else "",
                "mineru_model_version": (
                    self._settings.mineru_model_version if structurize else ""
                ),
            },
            "created_at": utc_now(),
            "started_at": "",
            "finished_at": "",
        }

    def create(
        self,
        filenames: list[str],
        paths: list[Path],
        structurize: bool = False,
        actor: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if job["status"] in {"queued", "indexing"}
            ]
            if len(active) >= self._settings.max_upload_jobs:
                raise RuntimeError("Upload queue is full")
            queued_paths = {
                str(Path(path).resolve())
                for job in active
                for path in job["paths"]
            }
            for path in paths:
                if str(path.resolve()) in queued_paths:
                    raise ValueError(f"File is already queued: {path.name}")

            job = self._new_job(filenames, paths, structurize, actor)
            self._jobs[job["id"]] = job
            self._save_locked()

        self._executor.submit(self._run, job["id"])
        return self.snapshot_job(job["id"])

    def create_uploaded(
        self,
        files: list[tuple[str, bytes]],
        data_dir: Path,
        structurize: bool = False,
        actor: str = "",
    ) -> dict[str, Any]:
        """Reserve the queue slot before replacing any existing file.

        When structurize is True, uploaded files are converted to Markdown via
        MinerU before chunking. ``actor`` is recorded in the audit trail.
        """

        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if job["status"] in {"queued", "indexing"}
            ]
            if len(active) >= self._settings.max_upload_jobs:
                raise RuntimeError("Upload queue is full")
            queued_names = {
                Path(path).name.casefold()
                for job in active
                for path in job["paths"]
            }
            duplicate = next(
                (
                    filename
                    for filename, _ in files
                    if filename.casefold() in queued_names
                ),
                None,
            )
            if duplicate:
                raise ValueError(f"File is already queued: {duplicate}")

            paths: list[Path] = []
            try:
                paths = save_upload_files(files, data_dir)
            except Exception:
                for path in paths:
                    path.unlink(missing_ok=True)
                raise
            job = self._new_job(
                [filename for filename, _ in files], paths, structurize, actor
            )
            self._jobs[job["id"]] = job
            self._save_locked()

        self._executor.submit(self._run, job["id"])
        return self.snapshot_job(job["id"])

    def snapshot_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Upload job not found")
            return self._snapshot(job)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._snapshot(job)
                for job in sorted(
                    self._jobs.values(),
                    key=lambda job: job["created_at"],
                    reverse=True,
                )
            ]

    def _snapshot(self, job: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(job)
        snapshot["file_details"] = [
            dict(item) for item in job.get("file_details", [])
        ]
        snapshot["runtime"] = dict(job.get("runtime") or {})
        snapshot["failed_files"] = list(job.get("failed_files") or [])
        snapshot["queue_position"] = self._queue_position_locked(job)
        return snapshot

    def _queue_position_locked(self, job: dict[str, Any]) -> int:
        """How many active jobs were created before this one (0 while running)."""

        if job.get("status") != "queued":
            return 0
        return sum(
            1
            for other in self._jobs.values()
            if other.get("status") in {"queued", "indexing"}
            and other.get("created_at", "") < job.get("created_at", "")
        )

    def recover(self) -> None:
        with self._lock:
            pending = [
                job_id
                for job_id, job in self._jobs.items()
                if job["status"] in {"queued", "indexing"}
            ]
            for job_id in pending:
                job = self._jobs[job_id]
                job["status"] = "queued"
                job["stage"] = "queued"
                job["started_at"] = ""
                job["error"] = ""
                job["progress_done"] = 0
                job["progress_total"] = len(job.get("paths", []))
                job["current_file"] = ""
                job["file_details"] = []
                job["failed_files"] = []
                job["cancel_requested"] = False
            if pending:
                self._save_locked()
        for job_id in pending:
            self._executor.submit(self._run, job_id)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Upload job not found")
            if job.get("status") == "queued":
                job["status"] = "cancelled"
                job["stage"] = "cancelled"
                job["cancel_requested"] = True
                job["finished_at"] = utc_now()
                self._prune_locked()
                self._save_locked()
            elif job.get("status") == "indexing":
                job["cancel_requested"] = True
                self._save_locked()
            return self._snapshot(job)

    def retry_failed(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Upload job not found")
            failed = [
                detail
                for detail in job.get("file_details", [])
                if detail.get("status") == "failed"
            ]
            if not failed:
                raise ValueError("No failed files to retry")
            paths: list[Path] = []
            for detail in failed:
                index = int(detail.get("index", -1))
                if index < 0 or index >= len(job.get("paths", [])):
                    raise ValueError("Failed file path is no longer available")
                path = Path(job["paths"][index])
                if not path.is_file():
                    raise ValueError(f"File no longer exists: {path.name}")
                paths.append(path)
            filenames = [path.name for path in paths]
            structurize = bool(job.get("structurize", False))
            actor = str(job.get("actor", ""))

        return self.create(filenames, paths, structurize, actor)

    def _run(self, job_id: str) -> None:
        from app.service import ingest_files

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.get("cancel_requested") or job.get("status") == "cancelled":
                job["status"] = "cancelled"
                job["stage"] = "cancelled"
                job["finished_at"] = utc_now()
                self._save_locked()
                return
            job["status"] = "indexing"
            job["stage"] = "load"
            job["started_at"] = utc_now()
            self._save_locked()
            paths = [Path(path) for path in job["paths"]]
            structurize = job.get("structurize", False)
            actor = job.get("actor", "")
            continue_on_error = bool(job.get("continue_on_error", True))

        def should_cancel() -> bool:
            with self._lock:
                current = self._jobs.get(job_id)
                return bool(current and current.get("cancel_requested"))

        def progress(
            stage: str,
            done: int,
            total: int,
            current_file: str,
            file_detail: dict[str, object] | None,
        ) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is None:
                    return
                current["stage"] = stage
                current["progress_done"] = done
                current["progress_total"] = total
                current["current_file"] = current_file
                if file_detail is not None:
                    details = current.setdefault("file_details", [])
                    index = int(file_detail.get("index", len(details)))
                    while len(details) <= index:
                        details.append({})
                    details[index] = dict(file_detail)
                self._save_locked()

        try:
            result = ingest_files(
                self._settings,
                paths,
                structurize,
                actor,
                progress=progress,
                should_cancel=should_cancel,
                continue_on_error=continue_on_error,
            )
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["stage"] = "failed"
                    job["error"] = str(exc)
                    job["finished_at"] = utc_now()
                    self._save_locked()
            return

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if result.cancelled:
                status = "cancelled"
            elif result.failed_files and result.files:
                status = "partial"
            elif result.failed_files:
                status = "failed"
            else:
                status = "completed"
            job.update(
                status=status,
                stage=(
                    "cancelled"
                    if result.cancelled
                    else ("failed" if status == "failed" else "done")
                ),
                files=result.files,
                documents=result.documents,
                chunks=result.chunks,
                failed_files=list(result.failed_files),
                progress_done=len(job.get("file_details", [])),
                progress_total=len(paths),
                current_file="",
                finished_at=utc_now(),
            )
            if status == "failed" and not job.get("error"):
                job["error"] = "所有文件处理失败"
            self._prune_locked()
            self._save_locked()

    def _load(self) -> None:
        if not self._state_path.is_file():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self._jobs = {}
                for job in payload:
                    if not isinstance(job, dict) or not job.get("id"):
                        continue
                    job.setdefault(
                        "stage",
                        "queued" if job.get("status") == "queued" else "done",
                    )
                    job.setdefault("progress_done", 0)
                    job.setdefault("progress_total", len(job.get("paths", [])))
                    job.setdefault("current_file", "")
                    job.setdefault("file_details", [])
                    job.setdefault("failed_files", [])
                    job.setdefault("continue_on_error", True)
                    job.setdefault("cancel_requested", False)
                    job.setdefault("runtime", {})
                    self._jobs[str(job["id"])] = job
        except (OSError, json.JSONDecodeError):
            # A corrupt UI queue must not prevent document ingestion.
            self._jobs = {}

    def _save_locked(self) -> None:
        temporary = self._state_path.with_suffix(".tmp")
        jobs = [dict(job) for job in self._jobs.values()]
        temporary.write_text(
            json.dumps(jobs, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self._state_path)

    def _prune_locked(self) -> None:
        finished = sorted(
            (
                job
                for job in self._jobs.values()
                if job["status"] in {"completed", "failed", "partial", "cancelled"}
            ),
            key=lambda job: job["finished_at"],
            reverse=True,
        )
        keep_ids = {job["id"] for job in finished[:100]}
        for job in finished[100:]:
            self._jobs.pop(job["id"], None)


_upload_store: UploadJobStore | None = None
_upload_store_lock = threading.Lock()


def get_upload_store() -> UploadJobStore:
    global _upload_store
    with _upload_store_lock:
        if _upload_store is None:
            from app.config import get_settings

            settings = get_settings()
            _upload_store = UploadJobStore(
                settings,
                Path(".web_upload_jobs.json"),
            )
        return _upload_store
