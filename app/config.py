from functools import lru_cache
from pathlib import Path

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
    retrieval_top_k: int = 4
    graph_result_limit: int = 8
    max_concurrent_answers: int = 8
    max_request_bytes: int = 1_048_576
    max_upload_bytes: int = 100 * 1024 * 1024
    max_upload_files: int = 10
    max_upload_jobs: int = 5

    data_dir: Path = Path("data/raw")
    chroma_persist_dir: Path = Path(".chroma")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

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
        "neo4j_uri",
        "neo4j_username",
        "neo4j_password",
        "neo4j_database",
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
        if self.retrieval_top_k <= 0 or self.graph_result_limit <= 0:
            raise ValueError("retrieval limits must be greater than zero")
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
