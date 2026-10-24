"""Account-scoped retrieval preferences; service credentials stay in config."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import Settings


class RetrievalPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    rerank_enabled: bool = False
    rerank_pool_size: int = Field(default=50, ge=1, le=500)
    rerank_score_filter_enabled: bool = False
    rerank_score_threshold: float | None = Field(default=None, ge=0, le=1)
    top_k: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def require_threshold(self):
        if self.rerank_score_filter_enabled and self.rerank_score_threshold is None:
            raise ValueError("开启分数过滤时必须填写阈值")
        return self


_lock = Lock()


def _path() -> Path:
    return Path(os.environ.get("USER_PREFERENCES_FILE", "config/user_preferences.json"))


def _load() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    # Never replace an unreadable store with defaults: that would erase other users.
    if not isinstance(data, dict) or any(not isinstance(value, dict) for value in data.values()):
        raise ValueError("偏好文件格式无效")
    return data


def _preferences(settings: Settings, saved: dict) -> dict:
    defaults = {key: getattr(settings, "retrieval_top_k" if key == "top_k" else key)
                for key in RetrievalPreferences.model_fields}
    return RetrievalPreferences.model_validate({**defaults, **saved}).model_dump()


def read_preferences(owner: str, settings: Settings) -> dict:
    with _lock:
        return _preferences(settings, _load().get(owner, {}))


def save_preferences(owner: str, updates: dict, settings: Settings) -> dict:
    # ponytail: one process serializes this small JSON file; use SQLite for multiple workers.
    with _lock:
        data = _load()
        saved = _preferences(settings, {**data.get(owner, {}), **updates})
        data[owner] = saved
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                json.dump(data, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return saved
