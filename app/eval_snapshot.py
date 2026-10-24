"""跑批快照与两份快照的对比。

为什么用快照而不是运行时开关：要验的那类改动（比如「把章节路径拼进 embedding
文本」）是**代码改动**，不是参数。「跑一次 → 改代码 → 再跑 → 对比」是天然流程，
而且不必为每个实验加一套运行时开关。

**一份快照必须能自证可比。** 否则对比就是在骗自己：两次跑之间索引重建过、或有人
改了标注，指标差异就跟你要验的那个改动毫无关系。所以每份快照都带上语料指纹、
标注指纹、配置指纹与 git 状态，`compare` 第一件事就是比这些。
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from app.eval_metrics import hit_at_k, mean_over_labeled, mrr, recall_at_k


logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1


# --- 指纹 -------------------------------------------------------------------


def corpus_fingerprint(
    chunks: list[tuple[str, str]], doc_ids: list[str]
) -> dict[str, object]:
    """语料的身份：每个切片的 (chunk_id, 正文指纹) 排序后取哈希。

    `chunks` 是 `[(chunk_id, 正文), ...]`。**不能只记 chunk_id**：重新入库后编号
    可能一模一样，但指向的正文已经换了——那样两份快照看起来「语料相同」，实际
    不是，对比就失去意义。

    同时记下文档集合：语料变了但文档集合没变，说明变的是**切片方式**，那种情况下
    chunk 级指标不可比、但文档级还可以看。
    """

    from app.eval_store import fingerprint_text

    pairs = sorted(f"{chunk_id}\x1f{fingerprint_text(text)}" for chunk_id, text in chunks)
    digest = hashlib.sha1("\n".join(pairs).encode("utf-8")).hexdigest()[:16]
    return {
        "fingerprint": digest,
        "chunks": len(pairs),
        "doc_ids": sorted(set(doc_ids)),
        "documents": len(set(doc_ids)),
    }


def label_fingerprint(questions: list[dict]) -> dict[str, object]:
    """标注的身份：题目 id + **生效的**相关切片（重新锚定之后的）。

    用锚定后的 id，因为那才是真正参与算指标的东西。
    """

    lines = []
    for entry in questions:
        relevant = sorted(str(c) for c in (entry.get("relevant_chunk_ids") or []))
        lines.append(f"{entry.get('id')}\x1f{'|'.join(relevant)}")
    lines.sort()
    digest = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]
    return {
        "fingerprint": digest,
        "questions": len(lines),
        "labelled": sum(1 for entry in questions if entry.get("relevant_chunk_ids")),
    }


def _git_state() -> dict[str, object]:
    """快照是哪个版本的代码跑出来的。git 不可用时留空，不影响快照本身。"""

    def run(args: list[str]) -> str:
        try:
            done = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path(__file__).resolve().parents[1],
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    return {"commit": run(["rev-parse", "HEAD"])[:12], "dirty": bool(run(["status", "--porcelain"]))}


# --- 构建 -------------------------------------------------------------------


def build_snapshot(
    *,
    corpus: dict[str, object],
    labels: dict[str, object],
    config: dict[str, object],
    ks: list[int],
    rows: list[dict],
) -> dict[str, object]:
    """纯函数：把手上的东西组装成一份可对比的快照。

    只存**原始检索结果**（排好序的 chunk_id 与 doc_id），不存算好的指标——
    `compare` 用双方共同的 K 集合重算，这样即使两次跑的 K 不同也能比。
    """

    return {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "git": _git_state(),
        "corpus": corpus,
        "labels": labels,
        "config": config,
        "ks": sorted(ks),
        "questions": [
            {
                "id": row.get("id"),
                "question": str(row.get("question") or ""),
                "rerank": row.get("rerank") or {},
                "recall": row.get("recall") or {},
                "elapsed": row.get("elapsed"),
                "retrieved": list(row.get("retrieved") or []),
                "retrieved_docs": list(row.get("retrieved_docs") or []),
                "relevant": list(row.get("relevant") or []),
                "relevant_docs": list(row.get("relevant_docs") or []),
                "stale": list(row.get("stale") or []),
                "elapsed_ms": round(float(row.get("elapsed") or 0.0) * 1000, 1),
            }
            for row in rows
        ],
    }


def save_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_snapshot(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "questions" not in payload:
        raise ValueError(f"不像是一份评估快照：{path}")
    return payload


# --- 对比 -------------------------------------------------------------------


def _comparable(before: dict, after: dict) -> list[str]:
    """两次快照可比吗？返回阻止对比的理由（空表示可比）。"""

    blockers: list[str] = []
    before_corpus = before.get("corpus") or {}
    after_corpus = after.get("corpus") or {}
    if before_corpus.get("fingerprint") != after_corpus.get("fingerprint"):
        if before_corpus.get("doc_ids") == after_corpus.get("doc_ids"):
            blockers.append(
                "语料变了（文档集合没变，说明切片方式或内容变了）。"
                "chunk 编号已经对不上，只能看文档级指标。"
            )
        else:
            blockers.append("语料不同：两次快照跑的不是同一批文档。")

    before_labels = before.get("labels") or {}
    after_labels = after.get("labels") or {}
    if before_labels.get("fingerprint") != after_labels.get("fingerprint"):
        blockers.append(
            "标注不同：指标差异可能来自标注改动，而不是你要验的那个改动。"
        )
    return blockers


def _means(questions: list[dict], ks: list[int], level: str) -> dict[str, object]:
    def key(row: dict) -> str:
        return "relevant" if level == "chunk" else "relevant_docs"

    def retrieved(row: dict) -> list[str]:
        return list(row["retrieved"] if level == "chunk" else row["retrieved_docs"])

    summary: dict[str, object] = {}
    for value in ks:
        summary[f"hit@{value}"] = mean_over_labeled(
            hit_at_k(retrieved(row), row[key(row)] or [], value) for row in questions
        )
    summary["mrr"] = mean_over_labeled(
        mrr(retrieved(row), row[key(row)] or []) for row in questions
    )
    summary[f"recall@{ks[-1]}"] = mean_over_labeled(
        recall_at_k(retrieved(row), row[key(row)] or [], ks[-1]) for row in questions
    )
    return summary


def compare_snapshots(before: dict, after: dict) -> dict[str, object]:
    """对比两份快照。

    头条是**变好 / 变差 / 不变的题数**，不是均值：题目少的时候，一个均值会被单题
    带飞，看不出真实分布。

    逐题以 MRR 的变化判定（比 Hit@K 细，并列更少）。
    """

    ks = sorted(set(before.get("ks") or []) & set(after.get("ks") or []))
    if not ks:
        ks = [1, 3, 5]

    after_by_id = {
        str(row.get("id") or row.get("question") or ""): row
        for row in after.get("questions") or []
    }

    per_question: list[dict[str, object]] = []
    improved = regressed = unchanged = missing = 0
    for row in before.get("questions") or []:
        key = str(row.get("id") or row.get("question") or "")
        other = after_by_id.get(key)
        if other is None:
            missing += 1
            continue
        if not (row.get("relevant") or []):
            continue  # 没标注的题不参与判定
        was = mrr(list(row["retrieved"]), row["relevant"] or [])
        now = mrr(list(other["retrieved"]), other["relevant"] or [])
        if was is None or now is None:
            continue
        delta = (now or 0.0) - (was or 0.0)
        if delta > 1e-9:
            improved += 1
        elif delta < -1e-9:
            regressed += 1
        else:
            unchanged += 1
        per_question.append(
            {
                "question": row.get("question"),
                "before_mrr": was,
                "after_mrr": now,
                "delta": delta,
                "before_rank": _first_relevant_rank(list(row["retrieved"]), row["relevant"] or []),
                "after_rank": _first_relevant_rank(list(other["retrieved"]), other["relevant"] or []),
            }
        )

    return {
        "blockers": _comparable(before, after),
        "ks": ks,
        "before": _means(before.get("questions") or [], ks, "chunk"),
        "after": _means(after.get("questions") or [], ks, "chunk"),
        "doc_level": {
            "before": _means(before.get("questions") or [], ks, "doc"),
            "after": _means(after.get("questions") or [], ks, "doc"),
        },
        "counts": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "only_in_before": missing,
        },
        "questions": per_question,
        "config_before": before.get("config") or {},
        "config_after": after.get("config") or {},
    }


def _first_relevant_rank(retrieved: list[str], relevant: list[str]) -> int | None:
    wanted = set(relevant)
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in wanted:
            return position
    return None
