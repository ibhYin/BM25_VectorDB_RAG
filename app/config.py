import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    ocr_model: str = "gpt-4o-mini"
    ocr_max_pages: int | None = None
    # MinerU online parsing; a token turns structurization on for real documents.
    mineru_api_token: str = ""
    mineru_model_version: Literal["pipeline", "vlm"] = "vlm"
    # Which converter runs when an ingest asks for structured extraction.
    structurize_provider: Literal["markitdown", "mineru"] = "markitdown"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 20
    retrieval_method: Literal["vector", "bm25", "hybrid"] = "hybrid"
    candidate_pool_size: int = 100
    similarity_threshold: float = 0.35
    max_concurrent_answers: int = 8
    max_request_bytes: int = 1_048_576
    max_upload_bytes: int = 100 * 1024 * 1024
    max_upload_files: int = 10
    max_upload_jobs: int = 5
    knowledge_base_id: str = "default"
    access_policy_file: Path = Path("config/access_policy.json")

    data_dir: Path = Path("data/raw")
    chroma_persist_dir: Path = Path(".chroma")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "llm_model",
        "embedding_model",
        "embedding_api_key",
        "embedding_base_url",
        "ocr_model",
        "mineru_api_token",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_ranges(self) -> "Settings":
        if not self.embedding_base_url:
            self.embedding_base_url = self.openai_base_url
            self.embedding_api_key = self.embedding_api_key or self.openai_api_key
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be between zero and CHUNK_SIZE")
        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be greater than zero")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        if not 20 <= self.candidate_pool_size <= 500:
            raise ValueError("CANDIDATE_POOL_SIZE must be between 20 and 500")
        if self.max_concurrent_answers <= 0:
            raise ValueError("MAX_CONCURRENT_ANSWERS must be greater than zero")
        if self.ocr_max_pages is not None and self.ocr_max_pages <= 0:
            raise ValueError("OCR_MAX_PAGES must be greater than zero")
        if self.max_request_bytes <= 0:
            raise ValueError("MAX_REQUEST_BYTES must be greater than zero")
        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be greater than zero")
        if self.max_upload_files <= 0 or self.max_upload_jobs <= 0:
            raise ValueError("upload limits must be greater than zero")
        return self


# Runtime overrides let the Web UI change model endpoints, models and keys
# without editing .env. Only these fields are accepted; everything else stays
# env-only.
RUNTIME_CONFIG_FIELDS = (
    "openai_base_url",
    "openai_api_key",
    "llm_model",
    "embedding_base_url",
    "embedding_api_key",
    "embedding_model",
    "mineru_api_token",
    "mineru_model_version",
    "structurize_provider",
)
# Empty means "keep what is already there": the UI never echoes secrets back,
# and a blank model picker must not silently erase the configured model.
RUNTIME_CONFIG_KEEP_WHEN_EMPTY = (
    "openai_api_key",
    "embedding_api_key",
    "llm_model",
    "embedding_model",
    "mineru_api_token",
    "mineru_model_version",
    "structurize_provider",
)


def runtime_config_path() -> Path:
    return Path(os.environ.get("RUNTIME_CONFIG_FILE", "config/runtime_config.json"))


def _load_runtime_overrides() -> dict[str, str]:
    """Read the persisted UI overrides; missing or corrupt files mean "no overrides"."""

    try:
        raw = json.loads(runtime_config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    overrides: dict[str, str] = {}
    for key, value in raw.items():
        if key not in RUNTIME_CONFIG_FIELDS or not isinstance(value, str):
            continue
        value = value.strip()
        # Empty base URLs are meaningful (reuse/default); empty secrets and model
        # names are not.
        if key in RUNTIME_CONFIG_KEEP_WHEN_EMPTY and not value:
            continue
        overrides[key] = value
    return overrides


_RUNTIME_OVERRIDES: dict[str, str] = _load_runtime_overrides()
_runtime_config_lock = Lock()


def _apply_embedding_fallback(settings: Settings) -> None:
    """Mirror the chat endpoint onto Embedding when it has none of its own."""

    if not settings.embedding_base_url:
        settings.embedding_base_url = settings.openai_base_url
        settings.embedding_api_key = settings.embedding_api_key or settings.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings(**_RUNTIME_OVERRIDES)


def update_runtime_config(overrides: dict[str, str]) -> Settings:
    """Apply UI overrides to the shared Settings instance and persist them."""

    unknown = set(overrides) - set(RUNTIME_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")

    with _runtime_config_lock:
        settings = get_settings()
        previous_base_url = settings.openai_base_url
        previous_api_key = settings.openai_api_key
        for key, value in overrides.items():
            setattr(settings, key, value)

        # A chat endpoint change must drag Embedding along when Embedding was
        # only mirroring it (no endpoint of its own in .env or the UI).
        if (
            "openai_base_url" in overrides
            and "embedding_base_url" not in overrides
            and settings.embedding_base_url == previous_base_url
        ):
            settings.embedding_base_url = overrides["openai_base_url"]
        if (
            "openai_api_key" in overrides
            and "embedding_api_key" not in overrides
            and settings.embedding_api_key == previous_api_key
        ):
            settings.embedding_api_key = overrides["openai_api_key"]

        _apply_embedding_fallback(settings)
        _RUNTIME_OVERRIDES.update(overrides)
        _persist_runtime_overrides()
    return settings


def _persist_runtime_overrides() -> None:
    path = runtime_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_RUNTIME_OVERRIDES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mask_secret(value: str) -> str:
    if len(value) >= 8:
        return "••••" + value[-4:]
    return "••••" if value else ""


def runtime_config_view() -> dict[str, object]:
    """Non-secret snapshot of the effective model endpoints for the UI."""

    settings = get_settings()
    return {
        "openai_base_url": settings.openai_base_url,
        "openai_api_key_set": bool(settings.openai_api_key),
        "openai_api_key_hint": _mask_secret(settings.openai_api_key),
        "llm_model": settings.llm_model,
        "embedding_base_url": settings.embedding_base_url,
        "embedding_api_key_set": bool(settings.embedding_api_key),
        "embedding_api_key_hint": _mask_secret(settings.embedding_api_key),
        "embedding_model": settings.embedding_model,
        "mineru_api_token_set": bool(settings.mineru_api_token),
        "mineru_api_token_hint": _mask_secret(settings.mineru_api_token),
        "mineru_model_version": settings.mineru_model_version,
        "structurize_provider": settings.structurize_provider,
        "runtime_config_file": str(runtime_config_path()),
    }


def endpoint_credentials(target: str) -> tuple[str, str]:
    """Effective (base_url, api_key) for the "llm" or "embedding" endpoint."""

    if target not in ("llm", "embedding"):
        raise ValueError("target must be llm or embedding")
    settings = get_settings()
    if target == "llm":
        return settings.openai_base_url, settings.openai_api_key
    return settings.embedding_base_url, settings.embedding_api_key
