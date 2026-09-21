"""`GET /api/knowledge-bases`：当前知识库、可访问性、以及策略里的其它库。

这里最要紧的一条是**不要把策略里的库说成「可用知识库」**：`knowledge_base_id` 是
进程级设置、检索时当 where 过滤，一个实例只服务一个库。策略里的其它 kb 定义为
「定义了但本实例不服务」才诚实。
"""

import json
from email.message import Message
from http import HTTPStatus
from pathlib import Path

import pytest

from app.access import UserContext
from app.config import Settings
from app.web import RequestError, WebRequestHandler


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "access_policy.json"
    path.write_text(
        json.dumps(
            {
                "admins": ["alice"],
                "users": {"alice": {"groups": ["hr"]}, "bob": {"groups": []}},
                "knowledge_bases": {
                    "default": {"group": "public"},
                    "engineering": {"group": "hr"},
                },
                "documents": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def make_handler(*, user, settings: Settings, monkeypatch):
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler = object.__new__(WebRequestHandler)
    handler.headers = Message()
    captured: dict = {}
    if user is not Ellipsis:
        handler._current_user = lambda *args, **kwargs: user
    handler._send_json = lambda body, status=HTTPStatus.OK: captured.update(
        body=body, status=int(status)
    )
    return handler, captured


def test_open_mode_reports_only_the_current_base(tmp_path, monkeypatch) -> None:
    """没有策略文件时没有「访问组」这个概念，只有一个当前库。"""

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(tmp_path / "absent.json"))
    handler, captured = make_handler(
        user=None, settings=Settings(_env_file=None, knowledge_base_id="default"),
        monkeypatch=monkeypatch,
    )

    handler._handle_knowledge_bases()

    assert captured["body"] == {
        "current": "default",
        "group": "public",
        "accessible": True,
        "policy_enabled": False,
        "other_knowledge_bases": [],
    }


def test_policy_lists_other_bases_as_not_served(policy_file, monkeypatch) -> None:
    """策略里的其它库要如实标成「本实例不服务」，而不是「可用」。"""

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    handler, captured = make_handler(
        user=UserContext("alice", frozenset({"hr", "public"})),
        settings=Settings(_env_file=None, knowledge_base_id="default"),
        monkeypatch=monkeypatch,
    )

    handler._handle_knowledge_bases()

    body = captured["body"]
    assert body["current"] == "default"
    assert body["group"] == "public"
    assert body["accessible"] is True
    assert body["policy_enabled"] is True
    assert body["other_knowledge_bases"] == ["engineering"]


def test_current_base_is_listed_only_once(policy_file, monkeypatch) -> None:
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    handler, captured = make_handler(
        user=UserContext("alice", frozenset({"hr", "public"})),
        settings=Settings(_env_file=None, knowledge_base_id="engineering"),
        monkeypatch=monkeypatch,
    )

    handler._handle_knowledge_bases()

    assert captured["body"]["current"] == "engineering"
    assert captured["body"]["other_knowledge_bases"] == ["default"]


def test_inaccessible_current_base_is_reported(policy_file, monkeypatch) -> None:
    """当前库的组不在调用者的组里 → 检索会返回空，得说出来。

    否则表现出来就是「莫名其妙什么都搜不到」。
    """

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    handler, captured = make_handler(
        user=UserContext("bob", frozenset({"public"})),
        settings=Settings(_env_file=None, knowledge_base_id="engineering"),
        monkeypatch=monkeypatch,
    )

    handler._handle_knowledge_bases()

    assert captured["body"]["accessible"] is False
    assert captured["body"]["group"] == "hr"


def test_a_base_missing_from_the_policy_falls_back_to_public(
    policy_file, monkeypatch
) -> None:
    """配置里的 kb 没写进策略时按 public 算（`knowledge_base_group` 的默认），
    而 public 会并进每个用户的组，所以仍算可访问。"""

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    handler, captured = make_handler(
        user=UserContext("bob", frozenset({"public"})),
        settings=Settings(_env_file=None, knowledge_base_id="未记录的库"),
        monkeypatch=monkeypatch,
    )

    handler._handle_knowledge_bases()

    assert captured["body"]["group"] == "public"
    assert captured["body"]["accessible"] is True


def test_anonymous_caller_is_rejected_in_policy_mode(policy_file, monkeypatch) -> None:
    """列表随身份而变，所以启用策略后必须带身份（这里不桩 _current_user）。"""

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    handler, _ = make_handler(
        user=Ellipsis,  # 用真的 _current_user
        settings=Settings(_env_file=None),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(RequestError) as error:
        handler._handle_knowledge_bases()

    assert error.value.status == HTTPStatus.UNAUTHORIZED
