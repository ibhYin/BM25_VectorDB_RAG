"""跑批：对问题集逐个检索、算出汇总指标。

CLI 与网页共用这一份。不把这套逻辑写在 CLI 里，是因为网页也要跑——两份实现迟早
分叉，跟检索本身是同一个道理（见 `service._retrieve_locked` 的说明）。

传进来的 `questions` 应当是**已经重新锚定过**的（见 `service.anchor_eval_labels`）：
失效的标注那时已被剔除，这里只管跑与算。
"""

from __future__ import annotations

from time import perf_counter
from math import ceil

from app.eval_metrics import hit_at_k, mean_over_labeled, mrr, recall_at_k


def run_batch(
    settings,
    questions: list[dict],
    *,
    ks: list[int],
    method: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    pool: int | None = None,
    fusion_method: str | None = None,
    fusion_k: int | None = None,
    fusion_alpha: float | None = None,
    user=None,
) -> dict:
    """逐题检索并汇总。返回 `{rows, summary, labelled, total, stale_total}`。"""

    from app.service import retrieve_question

    rows: list[dict] = []
    for entry in questions:
        started = perf_counter()
        rerank_detail = {}
        def progress(stage, status, detail):
            if stage == "rerank":
                rerank_detail.update(status=status, **{key: value for key, value in detail.items()
                                                     if key not in ("vector_hits", "candidates")})
        hits = retrieve_question(
            str(entry.get("question") or ""),
            settings,
            retrieval_method=method,
            top_k=top_k,
            similarity_threshold=threshold,
            candidate_pool=pool,
            fusion_method=fusion_method,
            fusion_k=fusion_k,
            fusion_alpha=fusion_alpha,
            user=user,
            progress=progress,
        )
        rows.append(
            {
                "id": entry.get("id"),
                "question": str(entry.get("question") or ""),
                "retrieved": [str(hit.get("chunk_id") or "") for hit in hits],
                # 文档级也留一份：切片编号不可比时，还能看「有没有找对文档」
                "retrieved_docs": [str(hit.get("doc_id") or "") for hit in hits],
                "relevant": list(entry.get("relevant_chunk_ids") or []),
                "relevant_docs": list(entry.get("relevant_doc_ids") or []),
                "stale": list(entry.get("stale_chunk_ids") or []),
                "elapsed": perf_counter() - started,
                "hits": hits,
                "rerank": rerank_detail,
            }
        )

    labelled = [row for row in rows if row["relevant"]]
    summary: dict[str, float | None] = {}
    for value in ks:
        summary[f"hit@{value}"] = mean_over_labeled(
            hit_at_k(row["retrieved"], row["relevant"], value) for row in labelled
        )
    summary["mrr"] = mean_over_labeled(
        mrr(row["retrieved"], row["relevant"]) for row in labelled
    )
    summary[f"recall@{ks[-1]}"] = mean_over_labeled(
        recall_at_k(row["retrieved"], row["relevant"], ks[-1]) for row in labelled
    )
    latencies = sorted(row["elapsed"] for row in rows)

    return {
        "performance": {
            "p95_latency_ms": latencies[ceil(len(latencies) * .95) - 1] * 1000 if latencies else None,
            "rerank_fallbacks": sum(row["rerank"].get("status") == "degraded" for row in rows),
        },
        "rows": rows,
        "summary": summary,
        "labelled": len(labelled),
        "total": len(rows),
        "stale_total": sum(len(row["stale"]) for row in rows),
    }
