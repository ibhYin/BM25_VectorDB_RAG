import pytest

from app.config import Settings


def test_settings_reject_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_settings_reject_non_positive_ocr_page_limit() -> None:
    with pytest.raises(ValueError, match="OCR_MAX_PAGES"):
        Settings(ocr_max_pages=0)


def test_settings_reject_non_positive_answer_concurrency() -> None:
    with pytest.raises(ValueError, match="MAX_CONCURRENT_ANSWERS"):
        Settings(max_concurrent_answers=0)


def test_embedding_credentials_fall_back_to_openai_until_base_url_is_set() -> None:
    shared = Settings(
        _env_file=None,
        openai_api_key="chat-key",
        openai_base_url="https://chat.example.test/v1",
    )
    assert (shared.embedding_api_key, shared.embedding_base_url) == (
        "chat-key",
        "https://chat.example.test/v1",
    )

    local = Settings(
        _env_file=None,
        openai_api_key="chat-key",
        openai_base_url="https://chat.example.test/v1",
        embedding_api_key="",
        embedding_base_url="http://127.0.0.1:11434/v1",
    )
    assert (local.embedding_api_key, local.embedding_base_url) == (
        "",
        "http://127.0.0.1:11434/v1",
    )


def test_settings_reject_invalid_upload_limits() -> None:
    with pytest.raises(ValueError, match="MAX_UPLOAD_BYTES"):
        Settings(_env_file=None, max_upload_bytes=0)

    with pytest.raises(ValueError, match="upload limits"):
        Settings(_env_file=None, max_upload_files=0)
