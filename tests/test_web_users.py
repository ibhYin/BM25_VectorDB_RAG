import json
from email.message import Message
from http import HTTPStatus

import pytest

from app.config import Settings
from app.web import RequestError, WebRequestHandler


def make_handler(method: str, path: str, user_id: str, body: dict | None = None):
    from app.web_auth import create_session

    handler = object.__new__(WebRequestHandler)
    headers = Message()
    headers["Cookie"] = f"rag_session={create_session(user_id, 'test-secret')}"
    handler.headers = headers
    handler.path = path
    handler.captured = {}
    handler._send_json = lambda payload, status=HTTPStatus.OK: handler.captured.update(
        payload=payload, status=status
    )
    handler._read_json = lambda: body or {}
    handler._dispatch = method
    return handler


def install_settings(tmp_path, monkeypatch):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "admins": ["demo-admin", "demo-reader"],
                "users": {
                    "demo-admin": {"groups": ["engineering"]},
                    "demo-reader": {"groups": []},
                    "bob": {"groups": ["finance"]},
                },
                "knowledge_bases": {"default": {"group": "public"}},
            }
        ),
        encoding="utf-8",
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"secret": "test-secret", "users": {"demo-admin": {"password_hash": "h"}}}),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None, access_policy_file=policy_path, web_auth_file=auth_path
    )
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    return settings


def test_users_list_requires_admin(tmp_path, monkeypatch) -> None:
    install_settings(tmp_path, monkeypatch)

    handler = make_handler("GET", "/api/users", "demo-admin")
    WebRequestHandler._handle_users_list(handler)
    assert handler.captured["status"] == HTTPStatus.OK
    names = {user["username"] for user in handler.captured["payload"]["users"]}
    assert names == {"demo-admin", "demo-reader", "bob"}

    handler = make_handler("GET", "/api/users", "bob")
    with pytest.raises(RequestError) as error:
        WebRequestHandler._handle_users_list(handler)
    assert error.value.status == HTTPStatus.FORBIDDEN


def test_users_create_via_handler(tmp_path, monkeypatch) -> None:
    settings = install_settings(tmp_path, monkeypatch)

    handler = make_handler(
        "POST",
        "/api/users",
        "demo-admin",
        body={"username": "carol", "password": "secret123", "groups": ["finance"], "is_admin": False},
    )
    WebRequestHandler._handle_users_create(handler)
    assert handler.captured["payload"]["username"] == "carol"
    assert handler.captured["payload"]["groups"] == ["finance"]

    policy = json.loads(settings.access_policy_file.read_text(encoding="utf-8"))
    auth = json.loads(settings.web_auth_file.read_text(encoding="utf-8"))
    assert "carol" in policy["users"]
    assert "carol" in auth["users"]

    # 非管理员创建被拒
    handler = make_handler(
        "POST", "/api/users", "bob", body={"username": "dave", "password": "secret123", "groups": []}
    )
    with pytest.raises(RequestError) as error:
        WebRequestHandler._handle_users_create(handler)
    assert error.value.status == HTTPStatus.FORBIDDEN


def test_users_update_via_handler(tmp_path, monkeypatch) -> None:
    settings = install_settings(tmp_path, monkeypatch)

    handler = make_handler(
        "POST",
        "/api/users",
        "demo-admin",
        body={"username": "carol", "password": "secret123", "groups": ["finance"]},
    )
    WebRequestHandler._handle_users_create(handler)

    handler = make_handler(
        "POST",
        "/api/users/carol",
        "demo-admin",
        body={"groups": ["engineering"], "is_admin": True, "password": "newpass1"},
    )
    WebRequestHandler._handle_users_update(handler, "carol")
    assert handler.captured["payload"]["is_admin"] is True
    assert handler.captured["payload"]["groups"] == ["engineering"]

    policy = json.loads(settings.access_policy_file.read_text(encoding="utf-8"))
    assert policy["users"]["carol"]["groups"] == ["engineering"]
    assert "carol" in policy["admins"]


def test_users_delete_via_handler(tmp_path, monkeypatch) -> None:
    settings = install_settings(tmp_path, monkeypatch)
    handler = make_handler(
        "POST",
        "/api/users",
        "demo-admin",
        body={"username": "carol", "password": "secret123", "groups": []},
    )
    WebRequestHandler._handle_users_create(handler)

    handler = make_handler("DELETE", "/api/users/carol", "demo-admin")
    WebRequestHandler._handle_users_delete(handler, "carol")
    assert handler.captured["payload"] == {"deleted": "carol"}

    policy = json.loads(settings.access_policy_file.read_text(encoding="utf-8"))
    auth = json.loads(settings.web_auth_file.read_text(encoding="utf-8"))
    assert "carol" not in policy["users"]
    assert "carol" not in auth["users"]


def test_users_delete_self_rejected(tmp_path, monkeypatch) -> None:
    install_settings(tmp_path, monkeypatch)
    handler = make_handler("DELETE", "/api/users/demo-admin", "demo-admin")
    with pytest.raises(RequestError) as error:
        WebRequestHandler._handle_users_delete(handler, "demo-admin")
    assert error.value.status == HTTPStatus.BAD_REQUEST
