import json
from email.message import Message
from http import HTTPStatus

from app.config import Settings
from app.web import RequestError, WebRequestHandler


def make_me_handler(path: str, user_id: str = "") -> WebRequestHandler:
    handler = object.__new__(WebRequestHandler)
    headers = Message()
    if user_id:
        headers["X-User-ID"] = user_id
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
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(_env_file=None, access_policy_file=policy_path),
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
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(_env_file=None, access_policy_file=policy_path),
    )
    handler = make_me_handler("/api/me", "carol")

    try:
        WebRequestHandler._handle_me(handler)
    except RequestError as error:
        assert error.status == HTTPStatus.UNAUTHORIZED
    else:
        raise AssertionError("expected unauthorized")
