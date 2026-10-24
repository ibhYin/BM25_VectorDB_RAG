import json

import pytest

from app.config import Settings
from app.preferences import read_preferences, save_preferences
from app.web import WebRequestHandler


def test_preferences_defaults_and_account_persistence(tmp_path, monkeypatch):
    path = tmp_path / "preferences.json"
    monkeypatch.setenv("USER_PREFERENCES_FILE", str(path))
    settings = Settings(_env_file=None)
    assert read_preferences("alice", settings)["rerank_enabled"] is False
    saved = save_preferences("alice", {"rerank_enabled": True, "rerank_pool_size": 70,
                             "rerank_score_threshold": .2}, settings)
    assert read_preferences("alice", settings) == saved
    assert read_preferences("bob", settings)["rerank_pool_size"] == 50
    assert "api_key" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("updates", [{"owner": "bob"}, {"rerank_api_key": "bad"},
    {"rerank_enabled": "true"}, {"rerank_pool_size": True}, {"rerank_pool_size": 501},
    {"rerank_score_filter_enabled": True}, {"rerank_score_threshold": float("nan")},
    {"rerank_score_threshold": 1.1}])
def test_preferences_reject_invalid_updates(updates):
    with pytest.raises(ValueError):
        save_preferences("alice", updates, Settings(_env_file=None))


def test_failed_preference_write_preserves_existing_data(tmp_path, monkeypatch):
    path = tmp_path / "preferences.json"
    monkeypatch.setenv("USER_PREFERENCES_FILE", str(path))
    settings = Settings(_env_file=None)
    save_preferences("alice", {"rerank_enabled": True}, settings)
    before = path.read_bytes()
    monkeypatch.setattr("app.preferences.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        save_preferences("bob", {"rerank_pool_size": 80}, settings)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_preferences_are_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "preferences.json"
    monkeypatch.setenv("USER_PREFERENCES_FILE", str(path))
    path.write_text("broken")
    with pytest.raises(ValueError):
        save_preferences("alice", {}, Settings(_env_file=None))
    assert path.read_text() == "broken"


def test_rerank_config_validation_and_secret_view(monkeypatch):
    from app.config import runtime_config_view
    settings = Settings(_env_file=None, rerank_api_key="fake-private-secret")
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    view = runtime_config_view()
    assert view["rerank_api_key_set"] is True
    assert "fake-private-secret" not in json.dumps(view)
    assert WebRequestHandler._read_config_updates({"rerank_api_key": ""}) == {}
    assert WebRequestHandler._read_config_updates({"rerank_enabled": False}) == {"rerank_enabled": "false"}
    with pytest.raises(ValueError):
        Settings(_env_file=None, rerank_score_filter_enabled=True)
    with pytest.raises(ValueError):
        Settings(_env_file=None, rerank_url="https://user:secret@example.test/rerank")


def test_preferences_api_scopes_by_authenticated_identity(monkeypatch):
    from app.access import UserContext
    settings = Settings(_env_file=None)
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler = object.__new__(WebRequestHandler)
    result = {}
    handler._current_user = lambda *a: UserContext("alice", frozenset({"public"}))
    handler._read_json = lambda: {"rerank_pool_size": 90}
    handler._send_json = result.update
    handler._handle_preferences(update=True)
    assert result["rerank_pool_size"] == 90
    handler._current_user = lambda *a: UserContext("bob", frozenset({"public"}))
    handler._handle_preferences()
    assert result["rerank_pool_size"] == 50
