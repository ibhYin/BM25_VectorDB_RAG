"""评估端点的行为：权限、参数校验、以及「取不到的 id 要报出来」。

端点都不返回召回正文（只有 run 返回 id），所以这里主要盯住参数与写入侧。
"""

import json
from email.message import Message
from http import HTTPStatus
from pathlib import Path

import pytest

from app.config import Settings
from app.eval_store import read_questions
from app.web import RequestError, WebRequestHandler, _read_k_list, _string_list


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "access_policy.json"
    path.write_text(
        json.dumps(
            {
                "admins": ["alice"],
                "users": {"alice": {"groups": []}, "bob": {"groups": []}},
                "knowledge_bases": {"default": {"group": "public"}},
                "documents": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def make_handler(*, user_id="alice", payload=None, monkeypatch=None, settings=None):
    if monkeypatch is not None:
        monkeypatch.setattr(
            "app.web.get_settings", lambda: settings or Settings(_env_file=None)
        )
    handler = object.__new__(WebRequestHandler)
    captured: dict = {}
    handler.headers = Message()
    if user_id is not None:
        handler.headers["X-User-ID"] = user_id
    handler._read_json = lambda: payload if payload is not None else {}
    handler._send_json = lambda body, status=HTTPStatus.OK: captured.update(
        body=body, status=int(status)
    )
    return handler, captured


# --- 解析辅助 ---------------------------------------------------------------


def test_read_k_list_defaults_and_sorts() -> None:
    assert _read_k_list(None) == [1, 3, 5]
    assert _read_k_list([5, 1, 3, 3]) == [1, 3, 5]


@pytest.mark.parametrize("bad", ["abc", [0], [-1], [1.5], [True], {"k": 1}, []])
def test_read_k_list_rejects_junk(bad) -> None:
    """非法 K 要报 400，不能悄悄退回默认——两次跑批 cutoff 不同，快照对比就没意义了。"""

    if bad == []:
        # 空列表退回默认值，不算非法
        assert _read_k_list(bad) == [1, 3, 5]
        return
    with pytest.raises(RequestError) as error:
        _read_k_list(bad)
    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_string_list_ignores_non_lists_and_blanks() -> None:
    assert _string_list(None) == []
    assert _string_list("not-a-list") == []
    assert _string_list(["a", "  ", "b"]) == ["a", "b"]


# --- 权限 -------------------------------------------------------------------


@pytest.mark.parametrize(
    "action", ["import", "label", "retrieve", "run"]
)
def test_eval_actions_require_admin(
    action: str, policy_file: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(policy_file))
    handler, _ = make_handler(user_id="bob", monkeypatch=monkeypatch)

    with pytest.raises(RequestError) as error:
        handler._handle_eval_action(action)

    assert error.value.status == HTTPStatus.FORBIDDEN


def test_unknown_eval_action_is_not_found() -> None:
    handler, _ = make_handler()

    with pytest.raises(RequestError) as error:
        handler._handle_eval_action("nope")

    assert error.value.status == HTTPStatus.NOT_FOUND


# --- 标注 -------------------------------------------------------------------


def resolved(*chunk_ids: str) -> dict[str, dict]:
    return {
        chunk_id: {
            "chunk_id": chunk_id,
            "text": f"{chunk_id} 正文",
            "doc_id": chunk_id.split("#")[0],
            "section_path": "",
            "page": None,
        }
        for chunk_id in chunk_ids
    }


def test_label_saves_and_reports_unresolved_ids(monkeypatch) -> None:
    """取不到的 id 必须回给前端——否则标注静悄悄地少一块，没人会发现。"""

    monkeypatch.setattr(
        "app.service.resolve_chunks",
        lambda settings, ids: resolved("good#chunk_001"),
    )
    handler, captured = make_handler(
        payload={
            "question": "问题",
            "relevant_chunk_ids": ["good#chunk_001", "typo#chunk_999"],
        },
        monkeypatch=monkeypatch,
    )

    handler._eval_label()

    assert captured["body"]["relevant_chunk_ids"] == ["good#chunk_001"]
    assert captured["body"]["unresolved"] == ["typo#chunk_999"]
    assert read_questions()[0]["relevant_chunk_ids"] == ["good#chunk_001"]


def test_label_requires_a_question(monkeypatch) -> None:
    handler, _ = make_handler(payload={"relevant_chunk_ids": ["a#chunk_001"]}, monkeypatch=monkeypatch)

    with pytest.raises(RequestError) as error:
        handler._eval_label()

    assert error.value.status == HTTPStatus.BAD_REQUEST


# --- 检索 -------------------------------------------------------------------


def test_retrieve_forwards_the_retrieval_params(monkeypatch) -> None:
    captured: dict = {}

    def fake_retrieve(question, settings, **kwargs):
        captured.update(question=question, **kwargs)
        return []

    monkeypatch.setattr("app.service.retrieve_question", fake_retrieve)
    handler, sent = make_handler(
        payload={
            "question": "问题",
            "retrieval_method": "bm25",
            "top_k": 7,
            "candidate_pool": 50,
            "similarity_threshold": 0.5,
        },
        monkeypatch=monkeypatch,
    )

    handler._eval_retrieve()

    assert captured["question"] == "问题"
    assert captured["retrieval_method"] == "bm25"
    assert captured["top_k"] == 7
    assert captured["candidate_pool"] == 50
    assert captured["similarity_threshold"] == 0.5
    assert sent["body"] == {"hits": []}


def test_retrieve_rejects_an_unknown_method(monkeypatch) -> None:
    handler, _ = make_handler(
        payload={"question": "问题", "retrieval_method": "semantic"}, monkeypatch=monkeypatch
    )

    with pytest.raises(RequestError) as error:
        handler._eval_retrieve()

    assert error.value.status == HTTPStatus.BAD_REQUEST


# --- 跑批 -------------------------------------------------------------------


def test_run_omits_hit_text_and_passes_ks(monkeypatch) -> None:
    """跑批响应不带 hit 正文——几十道题的正文字段会让响应白白变大。"""

    captured: dict = {}

    def fake_run(settings, questions, **kwargs):
        captured.update(questions=questions, **kwargs)
        return {
            "rows": [
                {
                    "id": "q1",
                    "question": "问题",
                    "retrieved": ["a#chunk_001"],
                    "relevant": ["a#chunk_001"],
                    "stale": [],
                    "elapsed": 0.1234,
                    "hits": [{"chunk_id": "a#chunk_001", "text": "很长的正文" * 50}],
                }
            ],
            "summary": {"hit@1": 1.0, "mrr": 1.0},
            "labelled": 1,
            "total": 1,
            "stale_total": 0,
        }

    monkeypatch.setattr("app.eval_runner.run_batch", fake_run)
    monkeypatch.setattr("app.service.anchor_eval_labels", lambda settings, questions: questions)
    handler, sent = make_handler(payload={"ks": [1, 3]}, monkeypatch=monkeypatch)

    handler._eval_run()

    assert captured["ks"] == [1, 3]
    body = sent["body"]
    assert body["labelled"] == 1
    assert body["questions"][0]["retrieved"] == ["a#chunk_001"]
    assert body["questions"][0]["elapsed_ms"] == 123.4
    assert "hits" not in body["questions"][0]
    assert "很长的正文" not in json.dumps(body, ensure_ascii=False)


def test_run_uses_the_default_ks_when_absent(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(settings, questions, **kwargs):
        captured.update(kwargs)
        return {"rows": [], "summary": {}, "labelled": 0, "total": 0, "stale_total": 0}

    monkeypatch.setattr("app.eval_runner.run_batch", fake_run)
    monkeypatch.setattr("app.service.anchor_eval_labels", lambda settings, questions: questions)
    handler, _ = make_handler(payload={}, monkeypatch=monkeypatch)

    handler._eval_run()

    assert captured["ks"] == [1, 3, 5]
