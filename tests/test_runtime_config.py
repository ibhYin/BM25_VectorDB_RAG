import json

import pytest

from app import config


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Isolate the process-wide override dict and the cached Settings instance."""

    monkeypatch.setenv("RUNTIME_CONFIG_FILE", str(tmp_path / "runtime_config.json"))
    monkeypatch.setattr(config, "_RUNTIME_OVERRIDES", {})
    config.get_settings.cache_clear()
    yield config
    config.get_settings.cache_clear()


def test_update_applies_to_shared_settings_and_persists(runtime, tmp_path):
    settings = runtime.update_runtime_config(
        {
            "openai_base_url": "https://gateway.example.test/v1",
            "openai_api_key": "sk-live-key-1234",
        }
    )

    assert settings is runtime.get_settings()
    assert runtime.get_settings().openai_base_url == "https://gateway.example.test/v1"
    assert runtime.get_settings().openai_api_key == "sk-live-key-1234"

    persisted = json.loads((tmp_path / "runtime_config.json").read_text(encoding="utf-8"))
    assert persisted == {
        "openai_base_url": "https://gateway.example.test/v1",
        "openai_api_key": "sk-live-key-1234",
    }


def test_chat_endpoint_change_moves_mirrored_embedding(runtime, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example.test/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    runtime.get_settings.cache_clear()
    assert runtime.get_settings().embedding_base_url == "https://old.example.test/v1"

    runtime.update_runtime_config({"openai_base_url": "https://new.example.test/v1"})

    assert runtime.get_settings().embedding_base_url == "https://new.example.test/v1"


def test_explicit_embedding_endpoint_is_left_alone(runtime, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example.test/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    runtime.get_settings.cache_clear()

    runtime.update_runtime_config({"openai_base_url": "https://new.example.test/v1"})

    assert runtime.get_settings().embedding_base_url == "http://127.0.0.1:11434/v1"


def test_update_rejects_unknown_fields(runtime):
    with pytest.raises(ValueError, match="Unknown config fields"):
        runtime.update_runtime_config({"retrieval_top_k": "5"})


def test_config_view_masks_secrets(runtime):
    runtime.update_runtime_config({"openai_api_key": "sk-abcdefgh1234"})

    view = runtime.runtime_config_view()

    assert view["openai_api_key_set"] is True
    assert view["openai_api_key_hint"].endswith("1234")
    assert "sk-abcdefgh1234" not in json.dumps(view)


def test_load_overrides_filters_empty_secrets_and_unknown_fields(tmp_path, monkeypatch):
    path = tmp_path / "runtime_config.json"
    path.write_text(
        json.dumps(
            {
                "openai_api_key": "",
                "embedding_api_key": "keep-me",
                "openai_base_url": "",
                "retrieval_top_k": "5",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_CONFIG_FILE", str(path))

    assert config._load_runtime_overrides() == {
        "embedding_api_key": "keep-me",
        "openai_base_url": "",
    }


def test_model_names_are_overridable_and_persisted(runtime, tmp_path):
    runtime.update_runtime_config({"llm_model": "glm-4.5", "embedding_model": "bge-m3:q8_0"})

    assert runtime.get_settings().llm_model == "glm-4.5"
    assert runtime.get_settings().embedding_model == "bge-m3:q8_0"
    persisted = json.loads((tmp_path / "runtime_config.json").read_text(encoding="utf-8"))
    assert persisted == {"llm_model": "glm-4.5", "embedding_model": "bge-m3:q8_0"}


def test_load_overrides_keeps_base_url_but_drops_empty_model(tmp_path, monkeypatch):
    path = tmp_path / "runtime_config.json"
    path.write_text(
        json.dumps({"llm_model": "", "embedding_model": "bge-m3:q8_0", "openai_base_url": ""}),
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_CONFIG_FILE", str(path))

    assert config._load_runtime_overrides() == {
        "embedding_model": "bge-m3:q8_0",
        "openai_base_url": "",
    }


def test_config_view_exposes_models(runtime):
    runtime.update_runtime_config({"llm_model": "glm-4.5", "embedding_model": "bge-m3:q8_0"})

    view = runtime.runtime_config_view()

    assert view["llm_model"] == "glm-4.5"
    assert view["embedding_model"] == "bge-m3:q8_0"


def test_endpoint_credentials_resolve_per_target(runtime):
    runtime.update_runtime_config(
        {
            "openai_base_url": "https://chat.example.test/v1",
            "openai_api_key": "chat-key",
            "embedding_base_url": "http://127.0.0.1:11434/v1",
            "embedding_api_key": "",
        }
    )

    assert runtime.endpoint_credentials("llm") == (
        "https://chat.example.test/v1",
        "chat-key",
    )
    assert runtime.endpoint_credentials("embedding")[0] == "http://127.0.0.1:11434/v1"
    with pytest.raises(ValueError):
        runtime.endpoint_credentials("ocr")


def test_failed_persistence_keeps_live_and_saved_config(runtime, monkeypatch):
    original = runtime.get_settings().llm_model
    before = dict(runtime._RUNTIME_OVERRIDES)

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(runtime, "_persist_runtime_overrides", fail)
    with pytest.raises(OSError, match="disk full"):
        runtime.update_runtime_config({"llm_model": "new-model"})
    assert runtime.get_settings().llm_model == original
    assert runtime._RUNTIME_OVERRIDES == before


def test_llm_section_save_keeps_other_saved_connections_after_reload(runtime, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.test/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("AUDIO_TRANSCRIBE_BASE_URL", "")
    runtime.get_settings.cache_clear()
    original = runtime.runtime_config_view()
    runtime.update_runtime_config({"openai_base_url": "https://new.test/v1"}, section="llm")
    assert runtime.get_settings().embedding_base_url == original["embedding_base_url"]
    assert runtime.get_settings().audio_transcribe_base_url == original["audio_transcribe_base_url"]
    runtime.get_settings.cache_clear()
    restored = runtime.Settings(**runtime._load_runtime_overrides())
    assert restored.embedding_base_url == original["embedding_base_url"]
    assert restored.audio_transcribe_base_url == original["audio_transcribe_base_url"]


def test_invalid_split_dependency_is_not_persisted(runtime):
    before = runtime.get_settings().chunk_overlap
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        runtime.update_runtime_config({"chunk_size": "10", "chunk_overlap": "10"})
    assert runtime.get_settings().chunk_overlap == before
    assert not runtime.runtime_config_path().exists()


def test_explicit_mineru_clear_survives_restart(runtime, monkeypatch):
    monkeypatch.setenv("MINERU_API_TOKEN", "original-env-token")
    runtime.get_settings.cache_clear()
    assert runtime.get_settings().mineru_api_token == "original-env-token"
    runtime.update_runtime_config({"mineru_api_token": ""})
    overrides = runtime._load_runtime_overrides()
    assert overrides["mineru_api_token"] == ""
    assert runtime.Settings(**overrides).mineru_api_token == ""
