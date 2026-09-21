"""File-backed user account management for web admins.

A user spans two JSON files, both reloaded on every request:

- access_policy.json — who the user is: groups and the admin flag
- web_auth.json — whether they can log in: password hash, session secret

Every write updates both files through a module lock and tmp+os.replace.
If the second replace fails the first is rolled back. A crash between the two
leaves a half-updated pair, and that direction is *safe*: login requires the
username in both files, so a partial state only means "cannot log in" — never
"logs in with wrong rights".
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from app.access import known_groups, load_access_policy
from app.audit import record as audit_record
from app.web_auth import hash_password, load_auth_config

MIN_PASSWORD_LENGTH = 6
DEFAULT_SESSION_TTL = 12 * 60 * 60

_write_lock = threading.Lock()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _commit_two(path_a: Path, text_a: str, path_b: Path, text_b: str) -> None:
    """Replace two files as close to atomically as we can.

    The second replace failing rolls the first back to the bytes it had before,
    so the pair never drifts further apart than it already was.
    """

    before_a = path_a.read_bytes() if path_a.is_file() else None
    tmp_a = path_a.with_name(path_a.name + ".tmp")
    tmp_b = path_b.with_name(path_b.name + ".tmp")
    tmp_a.write_text(text_a, encoding="utf-8")
    tmp_b.write_text(text_b, encoding="utf-8")
    os.replace(tmp_a, path_a)
    try:
        os.replace(tmp_b, path_b)
    except OSError:
        if before_a is None:
            path_a.unlink(missing_ok=True)
        else:
            rollback = path_a.with_name(path_a.name + ".rollback.tmp")
            rollback.write_bytes(before_a)
            os.replace(rollback, path_a)
        raise


def _auth_payload(path: Path) -> dict[str, Any]:
    """Auth payload, synthesizing a session secret if the file is missing.

    Does not write: callers persist through _commit_two so a validation failure
    never leaves a partially created file behind.
    """

    auth = _load_json(path)
    if not auth.get("secret"):
        auth["secret"] = secrets.token_urlsafe(32)
        auth.setdefault("session_ttl_seconds", DEFAULT_SESSION_TTL)
    return auth


def _validated_username(username: str) -> str:
    username = (username or "").strip()
    # "/" would collide with the /api/users/{name} route, ".." with traversal.
    if not username or "/" in username or username == ".." or ".." in username:
        raise ValueError("非法用户名")
    return username


def _validated_password(password: object) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"密码至少 {MIN_PASSWORD_LENGTH} 位")
    return password


def _validated_groups(groups: object, policy_path: Path) -> list[str]:
    if not isinstance(groups, list) or not all(
        isinstance(group, str) and group.strip() for group in groups
    ):
        raise ValueError("groups 必须是字符串数组")
    values = sorted({group.strip() for group in groups})
    invalid = set(values) - known_groups(policy_path)
    if invalid:
        raise ValueError(f"未知访问组: {', '.join(sorted(invalid))}")
    return values


def _admin_count(policy_payload: dict[str, Any]) -> int:
    admins = policy_payload.get("admins")
    return len(admins) if isinstance(admins, list) else 0


def list_users(settings) -> list[dict[str, Any]]:
    """Union of policy users/admins and auth users, sorted by name."""

    policy = load_access_policy(settings.access_policy_file)
    auth = load_auth_config(Path(settings.web_auth_file))
    auth_users = auth.get("users") if isinstance(auth.get("users"), dict) else {}

    policy_users: dict[str, frozenset[str]] = {}
    admins: set[str] = set()
    if policy is not None:
        policy_users = policy.users
        admins = set(policy.admins)

    names: set[str] = set(policy_users)
    names.update(admins)
    names.update(auth_users)

    users: list[dict[str, Any]] = []
    for name in sorted(names):
        record = auth_users.get(name)
        users.append(
            {
                "username": name,
                "groups": sorted(policy_users.get(name, frozenset())),
                "is_admin": name in admins,
                "has_password": bool(
                    isinstance(record, dict) and record.get("password_hash")
                ),
                "in_policy": name in policy_users,
            }
        )
    return users


def create_user(
    settings,
    username: str,
    *,
    groups: list[str],
    is_admin: bool,
    password: str,
    actor: str,
) -> dict[str, Any]:
    username = _validated_username(username)
    _validated_password(password)

    policy_path = Path(settings.access_policy_file)
    auth_path = Path(settings.web_auth_file)

    with _write_lock:
        policy_payload = _load_json(policy_path)
        auth_payload = _auth_payload(auth_path)

        policy_users = policy_payload.setdefault("users", {})
        if not isinstance(policy_users, dict):
            raise ValueError("策略文件 users 必须是对象")
        auth_users = auth_payload.setdefault("users", {})
        if not isinstance(auth_users, dict):
            raise ValueError("认证文件 users 必须是对象")
        if username in policy_users or username in auth_users:
            raise ValueError("用户已存在")
        policy_users[username] = {"groups": _validated_groups(groups, policy_path)}

        if is_admin:
            admins = policy_payload.setdefault("admins", [])
            if not isinstance(admins, list):
                raise ValueError("策略文件 admins 必须是数组")
            if username not in admins:
                admins.append(username)

        auth_users[username] = {"password_hash": hash_password(password)}

        _commit_two(
            policy_path, _dump(policy_payload), auth_path, _dump(auth_payload)
        )
        audit_record(
            "user.created",
            actor=actor,
            username=username,
            groups=",".join(sorted(groups)),
            is_admin=bool(is_admin),
        )
    return _snapshot(username, policy_payload, auth_payload)


def update_user(
    settings,
    username: str,
    *,
    groups: list[str] | None = None,
    is_admin: bool | None = None,
    password: str | None = None,
    clear_password: bool = False,
    actor: str,
) -> dict[str, Any]:
    """Partially update a user; only the keys passed (not None) are touched."""

    username = _validated_username(username)
    policy_path = Path(settings.access_policy_file)
    auth_path = Path(settings.web_auth_file)

    with _write_lock:
        policy_payload = _load_json(policy_path)
        auth_payload = _load_json(auth_path)

        policy_users = policy_payload.setdefault("users", {})
        admins = policy_payload.setdefault("admins", [])
        if not isinstance(policy_users, dict) or not isinstance(admins, list):
            raise ValueError("策略文件格式错误")
        if username not in policy_users and username not in admins:
            raise ValueError("用户不存在")

        changed: list[str] = []

        if groups is not None:
            policy_users.setdefault(username, {})["groups"] = _validated_groups(
                groups, policy_path
            )
            changed.append("groups")

        if is_admin is not None:
            was_admin = username in admins
            if was_admin and not is_admin:
                if len(admins) == 1:
                    raise ValueError("至少保留一位管理员")
                if actor == username:
                    raise ValueError("不能取消自己的管理员权限")
                admins.remove(username)
                changed.append("is_admin")
            elif not was_admin and is_admin:
                admins.append(username)
                changed.append("is_admin")

        if password is not None:
            _validated_password(password)
            auth_users = auth_payload.setdefault("users", {})
            if not isinstance(auth_users, dict):
                raise ValueError("认证文件 users 必须是对象")
            auth_users[username] = {"password_hash": hash_password(password)}
            changed.append("password")
        elif clear_password:
            auth_users = auth_payload.get("users")
            if isinstance(auth_users, dict):
                auth_users.pop(username, None)
            changed.append("password")

        _commit_two(
            policy_path, _dump(policy_payload), auth_path, _dump(auth_payload)
        )
        audit_record(
            "user.updated",
            actor=actor,
            username=username,
            changes=",".join(changed),
        )
    return _snapshot(username, policy_payload, auth_payload)


def delete_user(settings, username: str, actor: str) -> None:
    username = _validated_username(username)
    if username == actor:
        raise ValueError("不能删除自己的账号")

    policy_path = Path(settings.access_policy_file)
    auth_path = Path(settings.web_auth_file)

    with _write_lock:
        policy_payload = _load_json(policy_path)
        auth_payload = _load_json(auth_path)

        admins = policy_payload.get("admins")
        policy_users = policy_payload.get("users")
        auth_users = auth_payload.get("users")
        in_policy = isinstance(policy_users, dict) and (
            username in policy_users or (isinstance(admins, list) and username in admins)
        )
        in_auth = isinstance(auth_users, dict) and username in auth_users
        if not (in_policy or in_auth):
            raise ValueError("用户不存在")
        if isinstance(admins, list) and username in admins and len(admins) == 1:
            raise ValueError("至少保留一位管理员")

        if isinstance(policy_users, dict):
            policy_users.pop(username, None)
        if isinstance(admins, list):
            admins[:] = [admin for admin in admins if admin != username]
        if isinstance(auth_users, dict):
            auth_users.pop(username, None)

        _commit_two(
            policy_path, _dump(policy_payload), auth_path, _dump(auth_payload)
        )
        audit_record("user.deleted", actor=actor, username=username)


def _snapshot(username: str, policy_payload: dict[str, Any], auth_payload: dict[str, Any]) -> dict[str, Any]:
    policy_users = policy_payload.get("users", {})
    admins = policy_payload.get("admins", [])
    auth_users = auth_payload.get("users", {})
    record = auth_users.get(username) if isinstance(auth_users, dict) else None
    return {
        "username": username,
        "groups": sorted(policy_users.get(username, {}).get("groups", [])) if isinstance(policy_users, dict) else [],
        "is_admin": username in (admins if isinstance(admins, list) else []),
        "has_password": bool(isinstance(record, dict) and record.get("password_hash")),
        "in_policy": isinstance(policy_users, dict) and username in policy_users,
    }
