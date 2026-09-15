"""检索评估指标的边界与语义。

重点不是算术，而是两件事：**未标注要返回 None 而不是 0.0**（否则「没标」会被当成
「一条都没召回对」，汇总时把分数压低），以及 **rank-based**（不看分数）。
"""

import pytest

from app.eval_metrics import hit_at_k, mean_over_labeled, mrr, recall_at_k


RETRIEVED = ["a#chunk_001", "b#chunk_002", "c#chunk_003", "d#chunk_004"]


def test_hit_at_k_looks_only_at_the_first_k() -> None:
    # 相关切片排在第 3 位：k=2 时还没出现
    assert hit_at_k(RETRIEVED, {"c#chunk_003"}, 2) == 0.0
    assert hit_at_k(RETRIEVED, {"c#chunk_003"}, 3) == 1.0


def test_hit_at_k_hits_on_any_relevant_chunk() -> None:
    assert hit_at_k(RETRIEVED, {"d#chunk_004", "a#chunk_001"}, 1) == 1.0


def test_recall_at_k_is_the_fraction_covered() -> None:
    # 两个相关切片，前 2 条只覆盖了 1 个
    assert recall_at_k(RETRIEVED, {"a#chunk_001", "d#chunk_004"}, 2) == 0.5
    assert recall_at_k(RETRIEVED, {"a#chunk_001", "d#chunk_004"}, 4) == 1.0


def test_mrr_uses_the_first_relevant_rank() -> None:
    assert mrr(RETRIEVED, {"a#chunk_001"}) == 1.0
    assert mrr(RETRIEVED, {"c#chunk_003"}) == pytest.approx(1 / 3)
    # 多个相关时取最早出现的那个
    assert mrr(RETRIEVED, {"b#chunk_002", "d#chunk_004"}) == 0.5


def test_a_complete_miss_scores_zero_not_none() -> None:
    """召回全错是 0.0（真实结果），与「没标注」必须区分开。"""

    assert mrr(RETRIEVED, {"zzz#chunk_999"}) == 0.0
    assert hit_at_k(RETRIEVED, {"zzz#chunk_999"}, 4) == 0.0
    assert recall_at_k(RETRIEVED, {"zzz#chunk_999"}, 4) == 0.0


@pytest.mark.parametrize(
    "metric",
    [
        lambda: hit_at_k(RETRIEVED, set(), 3),
        lambda: recall_at_k(RETRIEVED, [], 3),
        lambda: mrr(RETRIEVED, set()),
    ],
)
def test_an_unlabelled_question_returns_none(metric) -> None:
    """未标注 ≠ 召回失败。返回 None 才能在汇总时被剔除。

    这是整个模块最容易写错的一条：如果这里返回 0.0，没标注的题会把平均分往下拽，
    看起来像检索变差了。
    """

    assert metric() is None


def test_empty_retrieval_is_scored_not_skipped() -> None:
    """一条都没召回、但题已标注 —— 这是货真价实的失败，要算 0。"""

    assert hit_at_k([], {"a#chunk_001"}, 5) == 0.0
    assert recall_at_k([], {"a#chunk_001"}, 5) == 0.0
    assert mrr([], {"a#chunk_001"}) == 0.0


def test_k_beyond_the_retrieved_length_is_fine() -> None:
    assert hit_at_k(RETRIEVED, {"d#chunk_004"}, 999) == 1.0
    assert recall_at_k(RETRIEVED, {"d#chunk_004"}, 999) == 1.0


def test_non_positive_k_scores_zero() -> None:
    assert hit_at_k(RETRIEVED, {"a#chunk_001"}, 0) == 0.0
    assert recall_at_k(RETRIEVED, {"a#chunk_001"}, 0) == 0.0


def test_mean_over_labeled_skips_unlabelled() -> None:
    assert mean_over_labeled([1.0, None, 0.0]) == 0.5


def test_mean_over_labeled_with_nothing_labelled_is_none() -> None:
    """全是未标注时不能返回 0.0 —— 那是「检索一无是处」的假象。"""

    assert mean_over_labeled([None, None]) is None
    assert mean_over_labeled([]) is None
