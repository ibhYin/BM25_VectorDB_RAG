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
