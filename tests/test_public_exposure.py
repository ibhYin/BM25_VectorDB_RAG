"""公网暴露前的加固：失败即关闭、状态字段分级、可信代理密钥。

这三条都是「本机自用」时无所谓、一旦放到公网就会真实造成危害的问题，所以各配
一个回归测试，防止以后被改回去。
"""

import json
from email.message import Message
from http import HTTPStatus
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.access import UserContext
from app.config import Settings
from app.main import app as cli_app
from app.web import RequestError, WebRequestHandler


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    """一份最小可用策略：alice 是管理员，bob 不是。"""

    path = tmp_path / "access_policy.json"
    path.write_text(
        json.dumps(
            {
                "admins": ["alice"],
                "users": {"alice": {"groups": ["hr"]}, "bob": {"groups": []}},
                "knowledge_bases": {"default": {"group": "public"}},
                "documents": {},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def no_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把策略指向一个不存在的文件——也就是「开放模式」。"""

    missing = tmp_path / "absent.json"
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(missing))
    return missing


def make_handler(*, user_id: str | None = None, secret: str | None = None):
    """只带请求头的最小 handler，够 _current_user / _require_trusted_proxy 用。"""

    handler = object.__new__(WebRequestHandler)
    handler.headers = Message()
    if user_id is not None:
        handler.headers["X-User-ID"] = user_id
    if secret is not None:
        handler.headers["X-Proxy-Secret"] = secret
    return handler


# --- 失败即开放 → 失败即关闭 ------------------------------------------------


def test_missing_policy_denies_service_when_required(no_policy: Path) -> None:
    """要求策略却没有策略 → 503，而不是放行。

    这是「失败即开放」的核心回归：以前这里返回 None，于是 _require_admin 的
    `policy is not None` 判断整条短路，管理员端点对匿名者开放。
    """

    settings = Settings(require_access_policy=True)
    handler = make_handler()  # 连 X-User-ID 都不带

    with pytest.raises(RequestError) as error:
        handler._current_user(settings)

    assert error.value.status == HTTPStatus.SERVICE_UNAVAILABLE


def test_missing_policy_is_allowed_when_not_required(no_policy: Path) -> None:
    """本地单机默认不受影响：没有策略就是不校验身份。"""

    settings = Settings()
    assert settings.require_access_policy is False
    assert make_handler()._current_user(settings) is None


def test_admin_check_fails_closed_when_policy_is_required(
    no_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.web.get_settings", lambda: Settings(require_access_policy=True)
    )

    with pytest.raises(RequestError) as error:
        make_handler()._require_admin()

    assert error.value.status == HTTPStatus.SERVICE_UNAVAILABLE


def test_upload_cannot_index_anonymously_when_policy_is_required(
    no_policy: Path,
) -> None:
    """开放模式那条「直接入库」的分支，在被要求策略时走不到。"""

    with pytest.raises(RequestError) as error:
        make_handler()._upload_permission(Settings(require_access_policy=True))

    assert error.value.status == HTTPStatus.SERVICE_UNAVAILABLE


def test_unknown_user_id_is_still_rejected(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))

    with pytest.raises(RequestError) as error:
        make_handler(user_id="mallory")._current_user(Settings())

    assert error.value.status == HTTPStatus.UNAUTHORIZED


def test_web_refuses_to_start_when_required_policy_is_absent(
    no_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """启动就失败，别让服务静静地以全开放姿态起来。"""

    import app.web as web_module

    monkeypatch.setenv("REQUIRE_ACCESS_POLICY", "1")
    monkeypatch.setattr("app.main.get_settings", lambda: Settings(require_access_policy=True))
    started: list = []
    # 钉住它：万一守卫失效，测试要干净地失败，而不是真的把服务器起起来挂着。
    monkeypatch.setattr(web_module, "run_web_server", lambda *a, **k: started.append(a))

    result = CliRunner().invoke(cli_app, ["web"])

    assert result.exit_code == 1
    assert "REQUIRE_ACCESS_POLICY" in result.output
    assert not started


def test_web_starts_when_the_required_policy_is_present(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.web as web_module

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    monkeypatch.setattr("app.main.get_settings", lambda: Settings(require_access_policy=True))
    started: list = []
    monkeypatch.setattr(web_module, "run_web_server", lambda *a, **k: started.append(a))

    result = CliRunner().invoke(cli_app, ["web"])

    assert result.exit_code == 0
    assert started


# --- /api/status 字段分级 ---------------------------------------------------


def test_status_hides_paths_and_file_list_from_non_admins(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(data_dir=tmp_path, chroma_persist_dir=tmp_path / "chroma"),
    )
    (tmp_path / "book.pdf").touch()
    handler = object.__new__(WebRequestHandler)

    as_admin = handler._status_payload(UserContext("alice", frozenset({"hr"})))
    as_user = handler._status_payload(UserContext("bob", frozenset()))

    assert as_admin["data_dir"] == str(tmp_path)
    assert as_admin["supported_files"]
    assert as_admin["markdown_cache"]

    assert as_user["data_dir"] is None
    assert as_user["chroma_persist_dir"] is None
    assert as_user["openai_base_url"] is None
    assert as_user["embedding_base_url"] is None
    assert as_user["supported_files"] == []
    assert as_user["markdown_cache"] == {}

    # 问答页与知识库页要用的字段对所有人都在。
    assert as_user["retrieval_top_k"] == as_admin["retrieval_top_k"]
    assert as_user["max_upload_files"] == as_admin["max_upload_files"]
    assert as_user["embedding_model"] == as_admin["embedding_model"]


def test_status_without_a_policy_keeps_the_local_behaviour(
    no_policy: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """没有策略文件时视为管理员——本地单机要保持原样。"""

    monkeypatch.setattr(
        "app.web.get_settings",
        lambda: Settings(data_dir=tmp_path, chroma_persist_dir=tmp_path / "chroma"),
    )
    (tmp_path / "book.pdf").touch()

    status = object.__new__(WebRequestHandler)._status_payload()

    assert status["data_dir"] == str(tmp_path)
    assert status["supported_files"]


# --- 可信代理共享密钥 -------------------------------------------------------


def test_trusted_proxy_secret_is_enforced_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.web.get_settings", lambda: Settings(trusted_proxy_secret="s3cret")
    )

    for handler in (make_handler(), make_handler(secret="wrong"), make_handler(secret="")):
        with pytest.raises(RequestError) as error:
            handler._require_trusted_proxy()
        assert error.value.status == HTTPStatus.UNAUTHORIZED

    make_handler(secret="s3cret")._require_trusted_proxy()  # 不抛即通过


def test_trusted_proxy_secret_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没设密钥时放行，本地开发完全无感。"""

    monkeypatch.setattr("app.web.get_settings", lambda: Settings())

    make_handler()._require_trusted_proxy()


# --- /api/me ----------------------------------------------------------------


def make_me_handler(user_id, monkeypatch, settings):
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler = make_handler(user_id=user_id)
    captured: dict = {}
    handler._send_json = lambda body, status=HTTPStatus.OK: captured.update(
        body=body, status=int(status)
    )
    return handler, captured


def test_me_reports_the_authenticated_user(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))

    handler, captured = make_me_handler("alice", monkeypatch, Settings())
    handler._handle_me()

    assert captured["body"]["user"] == "alice"
    assert captured["body"]["is_admin"] is True


def test_me_marks_non_admins_and_never_leaks_the_account_list(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`users` 必须是空的。

    前端拿这个列表渲染的是「切换身份（开发模式）」菜单，点下去跳 `/login?user=`
    ——那个路由只有 scripts/dev_proxy.py 有。生产环境漏出账号名，既是信息泄露，
    又等于给出一个「变成别人」的入口。dev_proxy 自己那份带列表的实现只用于本地。
    """

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))

    handler, captured = make_me_handler("bob", monkeypatch, Settings())
    handler._handle_me()

    body = captured["body"]
    assert body["user"] == "bob"
    assert body["is_admin"] is False
    assert body["users"] == []
    # 策略里明明有 alice / bob 两个账号，一个都不能漏出来。
    assert "alice" not in str(body)


def test_me_treats_open_mode_as_admin(
    no_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有策略文件 = 本地单机。后端本来就什么都允许，前端也该显示管理员入口。"""

    handler, captured = make_me_handler(None, monkeypatch, Settings())
    handler._handle_me()

    assert captured["body"] == {"user": None, "users": [], "is_admin": True}


def test_me_gates_settings_navigation_for_non_admins(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回归：以前应用里没有 /api/me，生产环境 404 → 前端 user 为 null →
    canConfigure() 的 `!state.identity.user` 为真 → 非管理员也看到「设置/审计」。"""

    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))

    handler, captured = make_me_handler("bob", monkeypatch, Settings())
    handler._handle_me()

    body = captured["body"]
    # 复刻 core.js:191 的 canConfigure()
    can_configure = not body["user"] or body["is_admin"]
    assert can_configure is False
