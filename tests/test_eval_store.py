"""评估集的存储语义：同 id 最新一条生效、正文指纹、损坏行容错。"""

import json
from pathlib import Path

import pytest

from app import eval_store


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "eval" / "retrieval_eval.jsonl"
    monkeypatch.setenv(eval_store.EVAL_LOG_ENV, str(path))
    return path


# --- 派生 id 与指纹 ---------------------------------------------------------


def test_question_id_ignores_whitespace_differences() -> None:
    """同一道题换行/空格不同仍是同一题，重复导入才不会造出重复条目。"""

    assert eval_store.question_id("注意力 机制是什么") == eval_store.question_id(
        "  注意力\n机制是什么  "
    )


def test_question_id_differs_for_different_questions() -> None:
    assert eval_store.question_id("注意力机制") != eval_store.question_id("数据库索引")


def test_fingerprint_normalises_whitespace() -> None:
    assert eval_store.fingerprint_text("一段  文字\n换行") == eval_store.fingerprint_text(
        "一段 文字 换行"
    )
    assert eval_store.fingerprint_text("一段文字") != eval_store.fingerprint_text("两段文字")


def test_fingerprints_for_keeps_only_the_selected_chunks() -> None:
    hits = [
        {
            "chunk_id": "doc#chunk_001",
            "doc_id": "doc",
            "text": "注意力 机制",
            "section_path": "1 引言",
            "page": 3,
        },
        {"chunk_id": "doc#chunk_002", "doc_id": "doc", "text": "别的内容"},
    ]

    fingerprints = eval_store.fingerprints_for(hits, ["doc#chunk_001", "没召回到的片"])

    assert set(fingerprints) == {"doc#chunk_001"}
    assert fingerprints["doc#chunk_001"] == {
        "text_sha1": eval_store.fingerprint_text("注意力 机制"),
        "doc_id": "doc",
        "section_path": "1 引言",
        "page": 3,
    }


# --- 追加与读取 -------------------------------------------------------------


def test_upsert_then_read_round_trips(log_path: Path) -> None:
    eval_store.upsert_question(
        "注意力机制是什么",
        relevant_chunk_ids=["doc#chunk_001"],
        fingerprints={"doc#chunk_001": {"text_sha1": "abc"}},
        note="题眼在第一章",
    )

    questions = eval_store.read_questions()

    assert len(questions) == 1
    entry = questions[0]
    assert entry["question"] == "注意力机制是什么"
    assert entry["relevant_chunk_ids"] == ["doc#chunk_001"]
    assert entry["note"] == "题眼在第一章"
    assert entry["source"] == "manual"
    assert entry["id"] == eval_store.question_id("注意力机制是什么")


def test_reupserting_the_same_question_replaces_the_labels(log_path: Path) -> None:
    """改错标签只要再追加一条修正，不必重写整个文件。"""

    eval_store.upsert_question("注意力机制是什么", relevant_chunk_ids=["wrong#chunk_001"])
    eval_store.upsert_question("注意力机制是什么", relevant_chunk_ids=["right#chunk_001"])

    questions = eval_store.read_questions()

    assert len(questions) == 1
    assert questions[0]["relevant_chunk_ids"] == ["right#chunk_001"]


def test_read_questions_keeps_first_appearance_order(log_path: Path) -> None:
    """第一道题被改过之后，仍然排在第一道的位置上，不跳到末尾。"""

    eval_store.upsert_question("题目一", relevant_chunk_ids=["a"])
    eval_store.upsert_question("题目二", relevant_chunk_ids=["b"])
    eval_store.upsert_question("题目一", relevant_chunk_ids=["a", "c"])

    questions = eval_store.read_questions()

    assert [entry["question"] for entry in questions] == ["题目一", "题目二"]
    assert questions[0]["relevant_chunk_ids"] == ["a", "c"]


def test_reading_skips_torn_and_unparsable_lines(log_path: Path) -> None:
    eval_store.upsert_question("题目一", relevant_chunk_ids=["a"])
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "torn\n')  # 崩溃留下的半行
        handle.write("[]\n")  # 合法 JSON 但不是对象
        handle.write("\n")
    eval_store.upsert_question("题目二", relevant_chunk_ids=["b"])

    assert [entry["question"] for entry in eval_store.read_questions()] == [
        "题目一",
        "题目二",
    ]


def test_reading_a_missing_file_is_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(eval_store.EVAL_LOG_ENV, str(tmp_path / "absent.jsonl"))

    assert eval_store.read_questions() == []


def test_upsert_rejects_a_blank_question(log_path: Path) -> None:
    with pytest.raises(ValueError, match="Question is required"):
        eval_store.upsert_question("   ", relevant_chunk_ids=["a"])


def test_upsert_never_raises_on_an_unwritable_path(tmp_path: Path, monkeypatch) -> None:
    """丢一条标注很糟，但让正在编辑的界面崩掉更糟——和审计日志同样的取舍。"""

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(eval_store.EVAL_LOG_ENV, str(blocker / "eval.jsonl"))

    entry = eval_store.upsert_question("题目", relevant_chunk_ids=["a"])

    assert entry["question"] == "题目"


def test_raw_log_is_append_only_and_keeps_the_label_history(log_path: Path) -> None:
    """标注变更史是刻意留下的：指标突然动了，能回头查是不是有人改了标签。"""

    eval_store.upsert_question("题目", relevant_chunk_ids=["a"])
    eval_store.upsert_question("题目", relevant_chunk_ids=["b"])

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]

    assert len(lines) == 2
    assert json.loads(lines[0])["relevant_chunk_ids"] == ["a"]
    assert json.loads(lines[1])["relevant_chunk_ids"] == ["b"]
