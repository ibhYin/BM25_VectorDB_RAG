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


def test_audio_transcribe_credentials_fall_back_to_openai() -> None:
    shared = Settings(
        _env_file=None,
        openai_api_key="chat-key",
        openai_base_url="https://chat.example.test/v1",
    )
    assert (shared.audio_transcribe_api_key, shared.audio_transcribe_base_url) == (
        "chat-key",
        "https://chat.example.test/v1",
    )
    assert shared.audio_transcribe_model == ""


def test_audio_transcribe_can_override_openai() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="chat-key",
        openai_base_url="https://chat.example.test/v1",
        audio_transcribe_api_key="asr-key",
        audio_transcribe_base_url="https://asr.example.test/v1",
        audio_transcribe_model="doubao-seed-asr-1",
    )
    assert (settings.audio_transcribe_api_key, settings.audio_transcribe_base_url) == (
        "asr-key",
        "https://asr.example.test/v1",
    )
    assert settings.audio_transcribe_model == "doubao-seed-asr-1"


def test_asr_provider_defaults_to_openai_and_aliyun_fields_empty() -> None:
    settings = Settings(_env_file=None)
    assert settings.audio_transcribe_provider == "openai"
    assert settings.aliyun_region == "cn-shanghai"
    assert settings.aliyun_access_key_id == ""
    assert settings.aliyun_oss_bucket == ""


def test_asr_provider_can_switch_to_aliyun() -> None:
    settings = Settings(
        _env_file=None,
        audio_transcribe_provider="aliyun",
        aliyun_access_key_id="LTAI-test",
        aliyun_oss_bucket="my-bucket",
    )
    assert settings.audio_transcribe_provider == "aliyun"
    assert settings.aliyun_access_key_id == "LTAI-test"
    assert settings.aliyun_oss_bucket == "my-bucket"
