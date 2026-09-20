import json
from email.message import Message
from http import HTTPStatus

import pytest

from app.config import Settings
from app.web import RequestError, WebRequestHandler


def make_me_handler(path: str, user_id: str = "", secret: str = "test-secret") -> WebRequestHandler:
    from app.web_auth import create_session

    handler = object.__new__(WebRequestHandler)
    headers = Message()
    if user_id:
        headers["Cookie"] = f"rag_session={create_session(user_id, secret)}"
    handler.headers = headers
    handler.path = path
    handler.captured = {}
    handler._send_json = lambda payload, status=HTTPStatus.OK: handler.captured.update(
        payload=payload, status=status
    )
    return handler


def test_me_returns_policy_identity_and_admin_state(tmp_path, monkeypatch) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "admins": ["alice"],
                "users": {
                    "alice": {"groups": ["engineering"]},
                    "bob": {"groups": ["finance"]},
                },
            }
        ),
        encoding="utf-8",
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"secret": "test-secret"}', encoding="utf-8")
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(
            _env_file=None, access_policy_file=policy_path, web_auth_file=auth_path
        ),
    )
    handler = make_me_handler("/api/me", "bob")
    WebRequestHandler._handle_me(handler)

    assert handler.captured["status"] == HTTPStatus.OK
    assert handler.captured["payload"] == {
        "user": "bob",
        "users": [{"id": "bob", "groups": ["finance", "public"]}],
        "is_admin": False,
    }


def test_me_rejects_user_missing_from_policy(tmp_path, monkeypatch) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        '{"users": {"alice": {"groups": ["engineering"]}}}', encoding="utf-8"
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"secret": "test-secret"}', encoding="utf-8")
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(
            _env_file=None, access_policy_file=policy_path, web_auth_file=auth_path
        ),
    )
    handler = make_me_handler("/api/me", "carol")

    try:
        WebRequestHandler._handle_me(handler)
    except RequestError as error:
        assert error.status == HTTPStatus.UNAUTHORIZED
    else:
        raise AssertionError("expected unauthorized")


def test_login_locks_after_repeated_failures(tmp_path, monkeypatch) -> None:
    import app.web as web_module
    from app.web import WebRequestHandler

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        '{"users": {"locked": {"groups": ["public"]}}}', encoding="utf-8"
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        '{"users": {"locked": {"password_hash": "encoded"}}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(
            _env_file=None, access_policy_file=policy_path, web_auth_file=auth_path
        ),
    )
    monkeypatch.setattr(
        "app.web.load_auth_config", lambda path: {"users": {"locked": {"password_hash": "encoded"}}}
    )
    monkeypatch.setattr("app.web.verify_password", lambda password, encoded: False)

    handler = object.__new__(WebRequestHandler)
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = Message()
    handler._read_json = lambda: {"username": "locked", "password": "wrong"}
    handler._send_json = lambda payload, status=HTTPStatus.OK: None

    web_module._login_failures.clear()
    for _ in range(8):
        with pytest.raises(RequestError) as error:
            handler._handle_login()
        assert error.value.status == HTTPStatus.UNAUTHORIZED

    with pytest.raises(RequestError) as error:
        handler._handle_login()
    assert error.value.status == HTTPStatus.TOO_MANY_REQUESTS
