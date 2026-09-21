import json
import os
from pathlib import Path

import pytest

from app.accounts import create_user, delete_user, list_users, update_user
from app.config import Settings
from app.web_auth import verify_password


def make_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        access_policy_file=tmp_path / "policy.json",
        web_auth_file=tmp_path / "auth.json",
    )


def seed_policy(settings, admins=("demo-admin",), users=None, kb_groups=("public", "hr", "engineering")):
    payload = {
        "admins": list(admins),
        "users": users or {"demo-admin": {"groups": []}},
        "knowledge_bases": {group: {"group": group} for group in kb_groups},
    }
    settings.access_policy_file.write_text(json.dumps(payload), encoding="utf-8")


def read_policy(settings):
    return json.loads(settings.access_policy_file.read_text(encoding="utf-8"))


def read_auth(settings):
    return json.loads(settings.web_auth_file.read_text(encoding="utf-8"))


def test_list_users_unions_policy_admins_and_auth(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.access_policy_file.write_text(
        json.dumps(
            {
                "admins": ["demo-admin"],
                "users": {
                    "demo-admin": {"groups": ["hr"]},
                    "alice": {"groups": ["engineering"]},
                },
            }
        ),
        encoding="utf-8",
    )
    settings.web_auth_file.write_text(
        json.dumps({"secret": "s", "users": {"alice": {"password_hash": "h"}}}),
        encoding="utf-8",
    )

    by_name = {user["username"]: user for user in list_users(settings)}
    assert set(by_name) == {"demo-admin", "alice"}
    assert by_name["demo-admin"] == {
        "username": "demo-admin",
        "groups": ["hr"],
        "is_admin": True,
        "has_password": False,
        "in_policy": True,
    }
    assert by_name["alice"]["is_admin"] is False
    assert by_name["alice"]["has_password"] is True


def test_create_user_writes_both_files_and_generates_secret(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)

    created = create_user(
        settings,
        "carol",
        groups=["hr", "public"],
        is_admin=False,
        password="secret123",
        actor="demo-admin",
    )
    assert created["username"] == "carol"
    assert created["groups"] == ["hr", "public"]
    assert created["is_admin"] is False
    assert created["has_password"] is True

    auth = read_auth(settings)
    assert auth["secret"]
    assert "carol" in auth["users"]
    assert verify_password("secret123", auth["users"]["carol"]["password_hash"])

    policy = read_policy(settings)
    assert policy["users"]["carol"]["groups"] == ["hr", "public"]
    assert "carol" not in policy["admins"]


def test_create_user_as_admin_writes_admin_flag(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)
    create_user(
        settings, "dave", groups=[], is_admin=True, password="secret123", actor="demo-admin"
    )
    policy = read_policy(settings)
    assert policy["admins"] == ["demo-admin", "dave"]


def test_create_user_validates_and_leaves_no_files_behind(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings, users={"demo-admin": {"groups": []}, "alice": {"groups": ["hr"]}})

    with pytest.raises(ValueError, match="已存在"):
        create_user(settings, "alice", groups=[], is_admin=False, password="secret123", actor="demo-admin")
    with pytest.raises(ValueError, match="密码至少"):
        create_user(settings, "dave", groups=[], is_admin=False, password="123", actor="demo-admin")
    with pytest.raises(ValueError, match="未知访问组"):
        create_user(settings, "dave", groups=["nope"], is_admin=False, password="secret123", actor="demo-admin")
    with pytest.raises(ValueError, match="非法用户名"):
        create_user(settings, "a/b", groups=[], is_admin=False, password="secret123", actor="demo-admin")
    with pytest.raises(ValueError, match="非法用户名"):
        create_user(settings, "..", groups=[], is_admin=False, password="secret123", actor="demo-admin")

    # 失败的创建不应留下任何文件（包括不应顺手生成 secret）。
    assert not settings.web_auth_file.is_file()
    policy = read_policy(settings)
    assert "dave" not in policy["users"]


def test_update_user_changes_groups_admin_and_password(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)
    create_user(
        settings, "alice", groups=["hr"], is_admin=True, password="aaa111", actor="demo-admin"
    )

    updated = update_user(
        settings,
        "alice",
        groups=["engineering"],
        is_admin=False,
        password="bbb222",
        actor="demo-admin",
    )
    assert updated["groups"] == ["engineering"]
    assert updated["is_admin"] is False

    policy = read_policy(settings)
    assert policy["users"]["alice"]["groups"] == ["engineering"]
    assert "alice" not in policy["admins"]
    auth = read_auth(settings)
    assert verify_password("bbb222", auth["users"]["alice"]["password_hash"])


def test_update_user_clear_password_blocks_login(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)
    create_user(
        settings, "alice", groups=["hr"], is_admin=False, password="aaa111", actor="demo-admin"
    )
    update_user(settings, "alice", clear_password=True, actor="demo-admin")
    auth = read_auth(settings)
    assert "alice" not in auth["users"]
    alice_record = next(
        user for user in list_users(settings) if user["username"] == "alice"
    )
    assert alice_record["has_password"] is False


def test_update_user_cannot_demote_self(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.access_policy_file.write_text(
        json.dumps(
            {
                "admins": ["demo-admin", "demo-reader"],
                "users": {"demo-admin": {"groups": []}, "demo-reader": {"groups": []}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不能取消自己的管理员权限"):
        update_user(settings, "demo-admin", is_admin=False, actor="demo-admin")


def test_update_user_cannot_demote_last_admin(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)
    with pytest.raises(ValueError, match="至少保留一位管理员"):
        update_user(settings, "demo-admin", is_admin=False, actor="demo-reader")


def test_delete_user_removes_from_both_files(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)
    create_user(
        settings, "alice", groups=["hr"], is_admin=False, password="aaa111", actor="demo-admin"
    )

    delete_user(settings, "alice", actor="demo-admin")
    policy = read_policy(settings)
    auth = read_auth(settings)
    assert "alice" not in policy["users"]
    assert "alice" not in auth["users"]
    assert all(user["username"] != "alice" for user in list_users(settings))


def test_delete_user_guards_self_and_last_admin(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)

    with pytest.raises(ValueError, match="不能删除自己的账号"):
        delete_user(settings, "demo-admin", actor="demo-admin")
    with pytest.raises(ValueError, match="至少保留一位管理员"):
        delete_user(settings, "demo-admin", actor="demo-reader")


def test_delete_unknown_user_raises(tmp_path) -> None:
    settings = make_settings(tmp_path)
    seed_policy(settings)
    with pytest.raises(ValueError, match="用户不存在"):
        delete_user(settings, "ghost", actor="demo-admin")


def test_second_write_failure_rolls_back_first_file(tmp_path, monkeypatch) -> None:
    import app.accounts as accounts

    settings = make_settings(tmp_path)
    seed_policy(settings)
    before = settings.access_policy_file.read_text(encoding="utf-8")

    real_replace = os.replace

    def flaky_replace(src, dst):
        if Path(dst).name == "auth.json":
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(accounts.os, "replace", flaky_replace)

    with pytest.raises(OSError):
        create_user(
            settings, "carol", groups=[], is_admin=False, password="secret123", actor="demo-admin"
        )

    # 第一个文件被回滚到原样，第二个文件保持不存在。
    assert settings.access_policy_file.read_text(encoding="utf-8") == before
    assert not settings.web_auth_file.is_file()
