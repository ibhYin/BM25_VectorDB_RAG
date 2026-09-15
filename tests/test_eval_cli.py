"""CLI 评估命令的行为：导入的幂等性、标注的指纹、空集与坏参数的提示。

重点盯住一条容易犯的错：**重复导入不能把已有标注清掉**。导入是追加式的
`upsert_question`，而它会写一条新的（空标注）记录——如果不先判重，第二次运行
就会把辛苦标的题全部清空，而且从输出上看不出来。
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.eval_store import fingerprint_text, read_questions, upsert_question
from app.main import app

runner = CliRunner()


def write_conversations(path: Path, turns: list[list[str]]) -> None:
    """一个会话，`turns` 是 [(问题, 答案), ...]。"""

    path.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "owner": "alice",
                    "title": "t",
                    "created_at": "2026-09-15T10:00:00+08:00",
                    "updated_at": "2026-09-15T10:00:00+08:00",
                    "turns": [
                        {
                            "question": question,
                            "answer": answer,
                            "retrieval_method": "hybrid",
                            "hits": [],
                            "created_at": "2026-09-15T10:00:00+08:00",
                        }
                        for question, answer in turns
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# --- import-conversations ---------------------------------------------------


def test_import_takes_only_first_turns(tmp_path: Path, monkeypatch) -> None:
    """追问不能进问题集。

    追问依赖对话上下文，而改写后的独立问题只存在于内存里的 job detail、不落盘，
    原样导入等于拿一个断章问题去评估。
    """

    conversations = tmp_path / "conversations.json"
    write_conversations(conversations, [("第一轮问题", "答案"), ("那它的优势呢？", "答案")])
    monkeypatch.setenv("CONVERSATIONS_FILE", str(conversations))

    result = runner.invoke(app, ["eval", "import-conversations"])

    assert result.exit_code == 0
    assert [entry["question"] for entry in read_questions()] == ["第一轮问题"]


def test_import_is_idempotent_and_never_clobbers_labels(
    tmp_path: Path, monkeypatch
) -> None:
    """重复导入是安全的：不重复加题，更不会把已有标注清掉。"""

    conversations = tmp_path / "conversations.json"
    write_conversations(conversations, [("第一轮问题", "答案")])
    monkeypatch.setenv("CONVERSATIONS_FILE", str(conversations))

    runner.invoke(app, ["eval", "import-conversations"])
    # 标上一条
    upsert_question("第一轮问题", relevant_chunk_ids=["a#chunk_001"])
    # 再导入一次
    runner.invoke(app, ["eval", "import-conversations"])

    questions = read_questions()
    assert len(questions) == 1
    assert questions[0]["relevant_chunk_ids"] == ["a#chunk_001"]


# --- label ------------------------------------------------------------------


def resolved(*chunk_ids: str) -> dict[str, dict]:
    return {
        chunk_id: {
            "chunk_id": chunk_id,
            "text": f"{chunk_id} 的正文",
            "doc_id": chunk_id.split("#")[0],
            "section_path": "1 引言",
            "page": 2,
        }
        for chunk_id in chunk_ids
    }


def test_label_records_fingerprints_from_the_index(monkeypatch) -> None:
    """指纹由索引里的正文现算，标注的人不必手算哈希。"""

    monkeypatch.setattr("app.service.resolve_chunks", lambda settings, ids: resolved("a#chunk_001"))

    result = runner.invoke(app, ["eval", "label", "问题", "--relevant", "a#chunk_001"])

    assert result.exit_code == 0
    entry = read_questions()[0]
    assert entry["relevant_chunk_ids"] == ["a#chunk_001"]
    assert entry["fingerprints"]["a#chunk_001"]["text_sha1"] == fingerprint_text(
        "a#chunk_001 的正文"
    )
    assert entry["fingerprints"]["a#chunk_001"]["doc_id"] == "a"


def test_label_reports_ids_it_cannot_resolve(monkeypatch) -> None:
    """拼错的 id 要当场报出来，而不是悄悄丢掉让标注少一块。"""

    monkeypatch.setattr("app.service.resolve_chunks", lambda settings, ids: resolved("good#chunk_001"))

    result = runner.invoke(
        app, ["eval", "label", "问题", "--relevant", "good#chunk_001,typo#chunk_999"]
    )

    assert "typo#chunk_999" in result.output
    assert read_questions()[0]["relevant_chunk_ids"] == ["good#chunk_001"]


def test_label_requires_a_selection(monkeypatch) -> None:
    monkeypatch.setattr("app.service.resolve_chunks", lambda settings, ids: {})

    result = runner.invoke(app, ["eval", "label", "问题", "--relevant", " , "])

    assert result.exit_code == 2


def test_relabelling_replaces_the_previous_labels() -> None:
    upsert_question("问题", relevant_chunk_ids=["wrong#chunk_001"])
    upsert_question("问题", relevant_chunk_ids=["right#chunk_001"])

    assert read_questions()[0]["relevant_chunk_ids"] == ["right#chunk_001"]


# --- run --------------------------------------------------------------------


def stub_anchoring(monkeypatch) -> None:
    """把重新锚定打桩成「原样通过」。

    `eval run` 会先拿真实索引去锚定标签，那是另一个测试文件（test_eval_anchor.py）
    的事；这里只测 CLI 的报告行为，不该依赖索引。
    """

    monkeypatch.setattr(
        "app.service.anchor_eval_labels",
        lambda settings, questions: [{**entry, "stale_chunk_ids": []} for entry in questions],
    )


def test_run_without_questions_points_at_the_onboarding_step() -> None:
    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 1
    assert "import-conversations" in result.output


@pytest.mark.parametrize("bad_k", ["abc", "1,,x", "0", "-3"])
def test_run_rejects_a_bad_k(bad_k: str) -> None:
    result = runner.invoke(app, ["eval", "run", "--k", bad_k])

    assert result.exit_code == 2


def test_run_reports_no_numbers_when_nothing_is_labelled(monkeypatch) -> None:
    """一道都没标注时不能报出数字——那会让人以为检索很糟。"""

    stub_anchoring(monkeypatch)
    upsert_question("问题一")
    monkeypatch.setattr(
        "app.service.retrieve_question",
        lambda *args, **kwargs: [{"chunk_id": "a#chunk_001", "retrieval_method": "vector"}],
    )

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 0
    assert "已标注 0 / 1 道题" in result.output
    assert "没有可报告的数字" in result.output


def test_run_scores_labelled_questions(monkeypatch) -> None:
    stub_anchoring(monkeypatch)
    upsert_question("问题一", relevant_chunk_ids=["b#chunk_002"])
    monkeypatch.setattr(
        "app.service.retrieve_question",
        lambda *args, **kwargs: [
            {"chunk_id": "a#chunk_001", "retrieval_method": "vector"},
            {"chunk_id": "b#chunk_002", "retrieval_method": "vector"},
        ],
    )

    result = runner.invoke(app, ["eval", "run", "--k", "1,3"])

    assert result.exit_code == 0
    # 相关切片排第 2：Hit@1 = 0，Hit@3 = 1，MRR = 1/2
    assert "已标注 1 / 1 道题" in result.output
    assert "0.500" in result.output


def test_show_hits_prints_the_chunks(monkeypatch) -> None:
    stub_anchoring(monkeypatch)
    upsert_question("问题一")
    monkeypatch.setattr(
        "app.service.retrieve_question",
        lambda *args, **kwargs: [
            {
                "rank": 1,
                "chunk_id": "a#chunk_001",
                "retrieval_method": "vector",
                "similarity": 0.5,
                "text": "正文片段",
                "title": "文档",
            }
        ],
    )

    result = runner.invoke(app, ["eval", "run", "--show-hits"])

    assert "a#chunk_001" in result.output
    assert "正文片段" in result.output
