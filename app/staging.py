"""Pending upload queue for users who cannot write to the vector store directly.

Non-admin uploads land here instead of the index: the bytes sit in a staging
directory and an entry is recorded, but nothing is chunked or embedded until an
admin approves it. Approval hands the bytes to the normal upload queue, so the
indexing path (and its audit trail) stays a single implementation.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.uploads import safe_filename


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StagingStore:
    """File-backed queue of uploads awaiting an admin decision."""

    def __init__(self, root: Path, state_path: Path) -> None:
        self._root = root
        self._state_path = state_path
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def add(
        self,
        files: list[tuple[str, bytes]],
        uploader: str,
        structurize: bool = False,
        note: str = "",
    ) -> dict[str, Any]:
        """Stage one submission (possibly several files) for review."""

        if not files:
            raise ValueError("At least one file is required")

        item_id = uuid.uuid4().hex
        directory = self._root / item_id
        directory.mkdir(parents=True, exist_ok=True)

        filenames: list[str] = []
        try:
            for raw_name, content in files:
                filename = safe_filename(raw_name)
                target = directory / filename
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                with temporary.open("wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                filenames.append(filename)
        except Exception:
            # Never leave a half-staged directory behind.
            for path in sorted(directory.glob("*")):
                path.unlink(missing_ok=True)
            directory.rmdir()
            raise

        item: dict[str, Any] = {
            "id": item_id,
            "filenames": filenames,
            "size_bytes": sum(len(content) for _, content in files),
            "uploader": uploader or "unknown",
            "structurize": structurize,
            "note": note,
            "status": "pending",
            "created_at": utc_now(),
            "decided_at": "",
            "decided_by": "",
            "documents": 0,
            "chunks": 0,
        }
        with self._lock:
            self._items[item_id] = item
            self._save_locked()
        return dict(item)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in sorted(
                    self._items.values(),
                    key=lambda item: item["created_at"],
                    reverse=True,
                )
            ]

    def get(self, item_id: str) -> dict[str, Any]:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise KeyError("Staged upload not found")
            return dict(item)

    def take(self, item_id: str) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
        """Read a staged submission and remove its files.

        The status is left alone so the caller can record the decision (and any
        resulting job id) once the hand-off to the index queue actually happened.
        """

        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise KeyError("Staged upload not found")
            if item["status"] != "pending":
                raise ValueError(f"Staged upload is already {item['status']}")
            payload = [
                (name, (self._root / item_id / name).read_bytes())
                for name in item["filenames"]
            ]

        self._remove_files(item_id)
        return dict(item), payload

    def discard(self, item_id: str, actor: str = "") -> dict[str, Any]:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise KeyError("Staged upload not found")
            if item["status"] != "pending":
                raise ValueError(f"Staged upload is already {item['status']}")

        self._remove_files(item_id)
        return self.decide(item_id, "rejected", actor)

    def decide(
        self, item_id: str, status: str, actor: str = "", **extra: Any
    ) -> dict[str, Any]:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise KeyError("Staged upload not found")
            item["status"] = status
            item["decided_at"] = utc_now()
            item["decided_by"] = actor
            item.update(extra)
            self._save_locked()
            return dict(item)

    def _remove_files(self, item_id: str) -> None:
        directory = self._root / item_id
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*")):
            path.unlink(missing_ok=True)
        directory.rmdir()

    def _load(self) -> None:
        if not self._state_path.is_file():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._items = {}
            return
        if isinstance(payload, list):
            self._items = {
                str(item.get("id")): item
                for item in payload
                if isinstance(item, dict) and item.get("id")
            }

    def _save_locked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(list(self._items.values()), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self._state_path)


_store: StagingStore | None = None
_store_lock = threading.Lock()


def get_staging_store() -> StagingStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = StagingStore(
                root=Path("data/pending"),
                state_path=Path(".web_staging.json"),
            )
        return _store
