"""检索评估指标。全部是 rank-based 的纯函数，不碰索引、不碰配置。

**为什么不看分数**：三种检索方式的分数根本不可比（向量是 L2 距离换算的相似度、
BM25 是无上界的匹配分、hybrid 是 RRF 融合分），而且 hybrid 分支压根忽略
`similarity_threshold`。所以指标一律只看**排名**。

**为什么以 Hit@K 与 MRR 为主、Recall@K 只作辅助**：标注是人工做的，必然标不全。
只从一套系统的结果里标注（pooling）会让 Recall 系统性偏低——另一套系统召回了标注者
没见过的好切片，却会被记成漏，**对它系统性不利**。Hit@K 与 MRR 只要求「至少一个判
相关的出现」，对不完整的相关集天然鲁棒。要做公平对比，标注池必须取两套系统结果的
并集。

**为什么没有 nDCG**：二值标签加上标注不完整，nDCG 相对 MRR 几乎没有增量。

`relevant` 为空表示这道题还没标注——这种情况下所有指标返回 `None`（而不是 0.0），
因为「没标注」和「一条都没召回对」是两回事，前者必须从汇总里剔除。
"""

from collections.abc import Iterable


def hit_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float | None:
    """前 k 条里有没有任一条判相关。1.0 / 0.0，未标注返回 None。"""

    relevant_set = set(relevant)
    if not relevant_set:
        return None
    if k <= 0:
        return 0.0
    return 1.0 if relevant_set & set(retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float | None:
    """前 k 条覆盖了多少个判相关的切片。

    人工标注下这是**下界**：漏标的那些会被算作没召回。所以它可以用来发现退化，
    但不适合单独拿来做「A 比 B 好」的结论。
    """

    relevant_set = set(relevant)
    if not relevant_set:
        return None
    if k <= 0:
        return 0.0
    return len(relevant_set & set(retrieved[:k])) / len(relevant_set)


def mrr(retrieved: list[str], relevant: Iterable[str]) -> float | None:
    """第一个判相关切片的倒数排名。一个都没召回对就是 0.0。

    没有 k 参数：传进来的列表本身已经被 top_k 截断，所以排名天然带着那个上限。
    """

    relevant_set = set(relevant)
    if not relevant_set:
        return None
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / position
    return 0.0


def mean_over_labeled(values: Iterable[float | None]) -> float | None:
    """对已标注的题取平均；一道都没标就返回 None。

    单独抽出来是为了让「None 表示未标注、要剔除」这条语义只有一个实现，
    不然汇总逻辑一散开，很容易退化成把未标注当成 0 分。
    """

    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)
