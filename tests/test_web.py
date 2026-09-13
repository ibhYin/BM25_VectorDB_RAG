import io
import json
from email.message import Message
from http import HTTPStatus

import pytest

from app.config import Settings
from app.web import RequestError, WebRequestHandler


def make_handler(payload: object) -> WebRequestHandler:
    body = json.dumps(payload).encode("utf-8")
    handler = object.__new__(WebRequestHandler)
    handler.headers = Message()
    handler.headers["Content-Type"] = "application/json"
    handler.headers["Content-Length"] = str(len(body))
    handler.rfile = io.BytesIO(body)
    return handler


def test_read_json_rejects_non_object_payload() -> None:
    handler = make_handler(["not", "an", "object"])

    with pytest.raises(RequestError) as error:
        handler._read_json()

    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_read_bool_does_not_treat_strings_as_true() -> None:
    with pytest.raises(RequestError) as error:
        WebRequestHandler._read_bool({"rebuild": "false"}, "rebuild")

    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_read_json_rejects_missing_or_wrong_content_type() -> None:
    for content_type in (None, "text/plain"):
        handler = make_handler({"question": "test"})
        if content_type is None:
            del handler.headers["Content-Type"]
        else:
            handler.headers.replace_header("Content-Type", content_type)

        with pytest.raises(RequestError) as error:
            handler._read_json()

        assert error.value.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_read_json_rejects_oversized_body(monkeypatch) -> None:
    handler = make_handler({"question": "too large"})
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(max_request_bytes=4),
    )

    with pytest.raises(RequestError) as error:
        handler._read_json()

    assert error.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_run_request_hides_unhandled_error_details(monkeypatch) -> None:
    handler = object.__new__(WebRequestHandler)
    handler.path = "/api/ask"
    handler.client_address = ("127.0.0.1", 1234)
    captured = {}
    monkeypatch.setattr(
        handler,
        "_send_error",
        lambda status, message: captured.update(status=status, message=message),
    )

    handler._run_request(lambda: (_ for _ in ()).throw(RuntimeError("secret detail")))

    assert captured == {
        "status": HTTPStatus.INTERNAL_SERVER_ERROR,
        "message": "Internal server error",
    }


def test_static_path_traversal_is_not_served(monkeypatch, tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    handler = object.__new__(WebRequestHandler)
    captured = {}
    monkeypatch.setattr(
        handler,
        "_send_error",
        lambda status, message: captured.update(status=status, message=message),
    )

    handler._send_static_file(secret, "text/plain")

    assert captured["status"] == HTTPStatus.NOT_FOUND


def test_status_lists_every_supported_document(tmp_path, monkeypatch) -> None:
    for name in ("book.pdf", "notes.md", "plain.txt", "ignored.csv"):
        (tmp_path / name).touch()
    settings = Settings(data_dir=tmp_path, chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler = object.__new__(WebRequestHandler)

    status = handler._status_payload()

    assert {path.rsplit("/", 1)[-1] for path in status["supported_files"]} == {
        "book.pdf",
        "notes.md",
        "plain.txt",
    }


def test_read_config_updates_rejects_non_url_base() -> None:
    with pytest.raises(RequestError) as error:
        WebRequestHandler._read_config_updates({"openai_base_url": "ftp://gateway"})

    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_read_config_updates_keeps_empty_base_url_but_drops_empty_keys() -> None:
    updates = WebRequestHandler._read_config_updates(
        {
            "openai_base_url": "https://gateway.example/v1",
            "embedding_base_url": "",
            "openai_api_key": "",
            "embedding_api_key": "sk-embed",
        }
    )

    assert updates == {
        "openai_base_url": "https://gateway.example/v1",
        "embedding_base_url": "",
        "embedding_api_key": "sk-embed",
    }


def test_read_config_updates_accepts_model_names() -> None:
    updates = WebRequestHandler._read_config_updates(
        {"llm_model": "glm-4.5", "embedding_model": "bge-m3:q8_0", "openai_base_url": ""}
    )

    assert updates == {
        "llm_model": "glm-4.5",
        "embedding_model": "bge-m3:q8_0",
        "openai_base_url": "",
    }


def test_read_config_updates_rejects_model_name_with_spaces() -> None:
    with pytest.raises(RequestError) as error:
        WebRequestHandler._read_config_updates({"llm_model": "glm 4.5"})

    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_read_config_updates_drops_blank_model_name() -> None:
    assert WebRequestHandler._read_config_updates({"llm_model": "   "}) == {}


def test_index_model_info_flags_a_mismatch(tmp_path) -> None:
    from rag.index_manifest import write_index_manifest

    chroma = tmp_path / "chroma"
    write_index_manifest(chroma, "bge-small-zh:q8_0")
    settings = Settings(
        _env_file=None,
        chroma_persist_dir=chroma,
        embedding_model="bge-m3:q8_0",
    )
    handler = object.__new__(WebRequestHandler)

    assert handler._index_model_info(settings) == {
        "index_embedding_model": "bge-small-zh:q8_0",
        "embedding_model_mismatch": True,
    }


def test_index_model_info_is_quiet_without_a_manifest(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        chroma_persist_dir=tmp_path / "chroma",
        embedding_model="bge-m3:q8_0",
    )
    handler = object.__new__(WebRequestHandler)

    assert handler._index_model_info(settings) == {
        "index_embedding_model": "",
        "embedding_model_mismatch": False,
    }


def test_index_model_info_is_quiet_when_models_match(tmp_path) -> None:
    from rag.index_manifest import write_index_manifest

    chroma = tmp_path / "chroma"
    write_index_manifest(chroma, "bge-small-zh:q8_0")
    settings = Settings(
        _env_file=None,
        chroma_persist_dir=chroma,
        embedding_model="bge-small-zh:q8_0",
    )
    handler = object.__new__(WebRequestHandler)

    assert handler._index_model_info(settings) == {
        "index_embedding_model": "bge-small-zh:q8_0",
        "embedding_model_mismatch": False,
    }


def test_read_config_updates_accepts_mineru_token_and_version() -> None:
    updates = WebRequestHandler._read_config_updates(
        {"mineru_api_token": "  secret-token  ", "mineru_model_version": "pipeline"}
    )

    assert updates == {
        "mineru_api_token": "secret-token",
        "mineru_model_version": "pipeline",
    }


def test_read_config_updates_drops_blank_mineru_token() -> None:
    assert WebRequestHandler._read_config_updates({"mineru_api_token": "   "}) == {}


def test_read_config_updates_rejects_unknown_mineru_version() -> None:
    with pytest.raises(RequestError) as error:
        WebRequestHandler._read_config_updates({"mineru_model_version": "turbo"})

    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_read_config_updates_accepts_structurize_provider() -> None:
    updates = WebRequestHandler._read_config_updates(
        {"structurize_provider": "mineru"}
    )

    assert updates == {"structurize_provider": "mineru"}


def test_read_config_updates_rejects_unknown_structurize_provider() -> None:
    with pytest.raises(RequestError) as error:
        WebRequestHandler._read_config_updates({"structurize_provider": "unknown"})

    assert error.value.status == HTTPStatus.BAD_REQUEST


def make_api_handler(*, user=None, payload=None):
    """A handler with identity/IO stubbed, for exercising one endpoint method."""

    handler = object.__new__(WebRequestHandler)
    captured: dict = {}
    handler._current_user = lambda *args, **kwargs: user
    handler._read_json = lambda: payload if payload is not None else {}
    handler._send_json = lambda body, status=HTTPStatus.OK: captured.update(
        body=body, status=int(status)
    )
    return handler, captured


@pytest.fixture
def conversation_store(tmp_path, monkeypatch):
    from app import conversations

    monkeypatch.setenv("CONVERSATIONS_FILE", str(tmp_path / "conversations.json"))
    monkeypatch.setattr(conversations, "_store", None)
    yield conversations.get_conversation_store()
    monkeypatch.setattr(conversations, "_store", None)


def _user(user_id: str):
    from app.access import UserContext

    return UserContext(user_id, frozenset({"public"}))


def test_conversation_list_only_returns_the_callers_own(conversation_store) -> None:
    conversation_store.create("alice")
    conversation_store.create("bob")

    handler, captured = make_api_handler(user=_user("bob"))
    handler._handle_conversations()

    assert captured["status"] == HTTPStatus.OK
    assert [c["title"] for c in captured["body"]["conversations"]] == ["新对话"]
    assert len(captured["body"]["conversations"]) == 1


def test_reading_another_users_conversation_is_forbidden(conversation_store) -> None:
    alice = conversation_store.create("alice")

    handler, _ = make_api_handler(user=_user("bob"))
    with pytest.raises(RequestError) as error:
        handler._handle_conversation_detail(alice["id"])

    assert error.value.status == HTTPStatus.FORBIDDEN


def test_reading_an_unknown_conversation_is_404(conversation_store) -> None:
    handler, _ = make_api_handler(user=_user("alice"))

    with pytest.raises(RequestError) as error:
        handler._handle_conversation_detail("does-not-exist")

    assert error.value.status == HTTPStatus.NOT_FOUND


def test_deleting_another_users_conversation_is_forbidden(conversation_store) -> None:
    alice = conversation_store.create("alice")

    handler, _ = make_api_handler(user=_user("bob"))
    with pytest.raises(RequestError) as error:
        handler._handle_conversation_delete(alice["id"])

    assert error.value.status == HTTPStatus.FORBIDDEN


def test_creating_an_answer_job_for_another_users_conversation_is_forbidden(
    conversation_store, monkeypatch
) -> None:
    alice = conversation_store.create("alice")
    monkeypatch.setattr(
        "app.answer_jobs.get_answer_store",
        lambda: pytest.fail("归属校验失败时不应创建任务"),
    )

    handler, _ = make_api_handler(
        user=_user("bob"),
        payload={"question": "试探", "conversation_id": alice["id"]},
    )

    with pytest.raises(RequestError) as error:
        handler._handle_create_answer_job()

    assert error.value.status == HTTPStatus.FORBIDDEN


def test_creating_an_answer_job_with_a_bad_conversation_id_type_is_rejected(
    conversation_store,
) -> None:
    handler, _ = make_api_handler(
        user=_user("alice"), payload={"question": "q", "conversation_id": 42}
    )

    with pytest.raises(RequestError) as error:
        handler._handle_create_answer_job()

    assert error.value.status == HTTPStatus.BAD_REQUEST
