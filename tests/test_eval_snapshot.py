"""快照与对比。

重点在**可比性**：两份快照如果是在不同语料或不同标注上跑的，指标差异就跟你要验
的那个改动毫无关系。`compare_snapshots` 的第一件事就是拦这种情况。另一条是
「语料指纹不能只看 chunk_id」——重建索引后编号可能一模一样但内容换了。
"""

import json
from pathlib import Path

import pytest

from app.eval_snapshot import (
    build_snapshot,
    compare_snapshots,
    corpus_fingerprint,
    label_fingerprint,
    load_snapshot,
    save_snapshot,
)


def labelled(question_id: str, relevant: list[str]) -> dict:
    return {"id": question_id, "question": question_id, "relevant_chunk_ids": relevant}


# --- 指纹 -------------------------------------------------------------------


def test_corpus_fingerprint_catches_a_text_change_under_the_same_ids() -> None:
    """编号没变、内容换了 —— 必须被认出来。

    这正是重建索引后会发生的事：`#chunk_003` 还是 `#chunk_003`，但指向的正文
    已经不是原来那段。只看 id 的话，两份快照会被误判成「同一份语料」。
    """

    before = corpus_fingerprint([("a#chunk_001", "原来的正文")], ["a"])
    after = corpus_fingerprint([("a#chunk_001", "换成了别的正文")], ["a"])

    assert before["fingerprint"] != after["fingerprint"]


def test_corpus_fingerprint_ignores_chunk_order() -> None:
    one = corpus_fingerprint([("a#chunk_001", "甲"), ("a#chunk_002", "乙")], ["a"])
    other = corpus_fingerprint([("a#chunk_002", "乙"), ("a#chunk_001", "甲")], ["a"])

    assert one["fingerprint"] == other["fingerprint"]


def test_corpus_fingerprint_records_the_document_set() -> None:
    """文档集合要单独记：语料变了而文档集合没变，说明变的是切片方式。"""

    corpus = corpus_fingerprint([("a#chunk_001", "甲")], ["a", "a", "b"])

    assert corpus["doc_ids"] == ["a", "b"]
    assert corpus["documents"] == 2
    assert corpus["chunks"] == 1


def test_label_fingerprint_tracks_the_labelled_chunks() -> None:
    before = label_fingerprint([labelled("q1", ["a#chunk_001"])])
    after = label_fingerprint([labelled("q1", ["a#chunk_002"])])

    assert before["fingerprint"] != after["fingerprint"]
    assert before["questions"] == 1 and before["labelled"] == 1


def test_label_fingerprint_counts_unlabelled_questions_too() -> None:
    fingerprint = label_fingerprint([labelled("q1", ["a#chunk_001"]), labelled("q2", [])])

    assert fingerprint["questions"] == 2
    assert fingerprint["labelled"] == 1


# --- 构建与存取 -------------------------------------------------------------


def snapshot(chunks: list[tuple[str, str]], docs: list[str], rows: list[dict]) -> dict:
    """注意两种形状：存储条目用 `relevant_chunk_ids`，快照行用 `relevant`。

    `label_fingerprint` 吃的是前者（CLI 传的是重新锚定后的条目），`build_snapshot`
    吃的是后者。这里显式转一次——如果传错形状，指纹两边都会是空的、**看起来还一致**，
    对比就会误判成「可比」。
    """

    labels = label_fingerprint(
        [{"id": row["id"], "relevant_chunk_ids": row["relevant"]} for row in rows]
    )
    return build_snapshot(
        corpus=corpus_fingerprint(chunks, docs),
        labels=labels,
        config={"method": "hybrid"},
        ks=[1, 3],
        rows=rows,
    )


def row(question_id: str, retrieved: list[str], relevant: list[str]) -> dict:
    return {
        "id": question_id,
        "question": question_id,
        "retrieved": retrieved,
        "retrieved_docs": [chunk.split("#")[0] for chunk in retrieved],
        "relevant": relevant,
        "relevant_docs": [chunk.split("#")[0] for chunk in relevant],
        "stale": [],
        "elapsed": 0.25,
    }


def test_build_snapshot_keeps_the_raw_ids_not_computed_metrics() -> None:
    """只存原始结果：这样比较时能用双方共同的 K 集合重算。"""

    built = snapshot([("a#chunk_001", "甲")], ["a"], [row("q1", ["a#chunk_001"], [])])

    assert built["questions"][0]["retrieved"] == ["a#chunk_001"]
    assert built["questions"][0]["elapsed_ms"] == 250.0
    assert "hit@1" not in built["questions"][0]


def test_snapshot_round_trips_through_a_file(tmp_path: Path) -> None:
    path = tmp_path / "snap.json"
    built = snapshot([("a#chunk_001", "甲")], ["a"], [row("q1", ["a#chunk_001"], [])])

    save_snapshot(path, built)

    assert load_snapshot(path)["corpus"]["chunks"] == 1


def test_loading_a_non_snapshot_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(ValueError, match="不像是一份评估快照"):
        load_snapshot(path)


# --- 可比性 -----------------------------------------------------------------


def test_identical_fingerprints_have_no_blockers() -> None:
    questions = [row("q1", ["a#chunk_001"], ["a#chunk_001"])]
    before = snapshot([("a#chunk_001", "甲")], ["a"], questions)
    after = snapshot([("a#chunk_001", "甲")], ["a"], questions)

    assert compare_snapshots(before, after)["blockers"] == []


def test_a_text_change_under_the_same_ids_blocks_the_comparison() -> None:
    questions = [row("q1", ["a#chunk_001"], ["a#chunk_001"])]
    before = snapshot([("a#chunk_001", "甲")], ["a"], questions)
    after = snapshot([("a#chunk_001", "乙")], ["a"], questions)

    blockers = compare_snapshots(before, after)["blockers"]

    assert len(blockers) == 1
    # 文档集合没变 → 提示这是切片方式变了，还能看文档级
    assert "切片方式" in blockers[0]


def test_a_different_document_set_blocks_with_its_own_reason() -> None:
    questions = [row("q1", ["a#chunk_001"], ["a#chunk_001"])]
    before = snapshot([("a#chunk_001", "甲")], ["a"], questions)
    after = snapshot([("a#chunk_001", "甲"), ("b#chunk_001", "乙")], ["a", "b"], questions)

    assert "不是同一批文档" in compare_snapshots(before, after)["blockers"][0]


def test_changed_labels_block_the_comparison() -> None:
    """标注变了，指标差异可能来自标注——这种事必须先说清楚。"""

    before = snapshot([("a#chunk_001", "甲")], ["a"], [row("q1", ["a#chunk_001"], [])])
    after = snapshot([("a#chunk_001", "甲")], ["a"], [row("q1", ["a#chunk_001"], ["a#chunk_001"])])

    assert any("标注不同" in reason for reason in compare_snapshots(before, after)["blockers"])


# --- 指标与逐题变化 ---------------------------------------------------------


def test_compare_reports_improved_regressed_and_unchanged() -> None:
    corpus = ([("a#chunk_001", "甲"), ("a#chunk_002", "乙")], ["a"])
    questions = [row("q1", ["a#chunk_001"], ["a#chunk_001"])]
    before = snapshot(*corpus, questions)
    # 同一道题：相关切片从第 1 位掉到第 2 位 → 变差
    after = snapshot(*corpus, [row("q1", ["a#chunk_002", "a#chunk_001"], ["a#chunk_001"])])

    report = compare_snapshots(before, after)

    assert report["counts"] == {
        "improved": 0,
        "regressed": 1,
        "unchanged": 0,
        "only_in_before": 0,
    }
    assert report["before"]["mrr"] == 1.0
    assert report["after"]["mrr"] == 0.5
    assert report["questions"][0]["before_rank"] == 1
    assert report["questions"][0]["after_rank"] == 2


def test_compare_ignores_unlabelled_questions() -> None:
    corpus = ([("a#chunk_001", "甲")], ["a"])
    questions = [row("q1", ["a#chunk_001"], [])]
    report = compare_snapshots(snapshot(*corpus, questions), snapshot(*corpus, questions))

    assert report["counts"]["unchanged"] == 0
    assert report["before"]["mrr"] is None


def test_compare_uses_only_the_k_values_both_snapshots_share() -> None:
    """K 集合不同就直接用交集，免得拿不同 cutoff 的数字互比。"""

    corpus = ([("a#chunk_001", "甲")], ["a"])
    questions = [row("q1", ["a#chunk_001"], ["a#chunk_001"])]
    labels = label_fingerprint([{"id": "q1", "relevant_chunk_ids": ["a#chunk_001"]}])
    before = build_snapshot(
        corpus=corpus_fingerprint(*corpus),
        labels=labels,
        config={},
        ks=[1, 3, 5],
        rows=questions,
    )
    after = build_snapshot(
        corpus=corpus_fingerprint(*corpus),
        labels=labels,
        config={},
        ks=[3, 10],
        rows=questions,
    )

    assert compare_snapshots(before, after)["ks"] == [3]


def test_doc_level_metrics_survive_a_chunk_level_mismatch() -> None:
    """切片编号对不上时，文档级还能看出「有没有找对文档」。

    这里 retrieved 是同一个文档的**另一片**：切片级算漏，文档级算命中。
    """

    corpus = ([("a#chunk_001", "甲"), ("a#chunk_009", "壬")], ["a"])
    before = snapshot(*corpus, [row("q1", ["a#chunk_001"], ["a#chunk_001"])])
    after = snapshot(*corpus, [row("q1", ["a#chunk_009"], ["a#chunk_001"])])

    report = compare_snapshots(before, after)

    assert report["before"]["hit@1"] == 1.0
    # 切片级：相关的那片掉出第 1 位了
    assert report["after"]["hit@1"] == 0.0
    # 文档级仍然命中：检索到的那片属于同一个文档
    assert report["doc_level"]["after"]["hit@1"] == 1.0
