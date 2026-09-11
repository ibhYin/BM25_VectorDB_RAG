from functools import lru_cache
from pathlib import Path
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
