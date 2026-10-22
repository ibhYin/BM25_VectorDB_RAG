from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Callable

from app.access import UserContext
from app.config import Settings
from app.service import answer_limiter, retrieve_question
from chains.aggregate_chain import MAX_EVIDENCE_CHARS, extract_categories
from rag.vector_store import get_embeddings


ProgressCallback = Callable[[str, dict[str, object]], None]
MAX_EVIDENCE_PER_CATEGORY = 3
SAMPLE_NOTE = "基于召回样本估计，非全库精确计数。"


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return []
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def greedy_cluster(
    vectors: list[list[float]], threshold: float = 0.85
) -> list[list[int]]:
    """按与簇质心的余弦相似度做贪心聚类，返回每簇的成员下标。"""

    clusters: list[dict[str, Any]] = []
    for index, vector in enumerate(vectors):
        unit = _unit(vector)
        if not unit:
            clusters.append({"members": [index], "sum": [], "count": 1})
            continue
        best: dict[str, Any] | None = None
        best_similarity = threshold
        for cluster in clusters:
            if not cluster["sum"]:
                continue
            similarity = _dot(unit, _unit(cluster["sum"]))
            if similarity >= best_similarity:
                best_similarity = similarity
                best = cluster
        if best is None:
            clusters.append({"members": [index], "sum": unit, "count": 1})
        else:
            best["members"].append(index)
            best["sum"] = [a + b for a, b in zip(best["sum"], unit)]
            best["count"] += 1
    return [list(cluster["members"]) for cluster in clusters]


def count_by_doc(items: list[dict[str, Any]]) -> int:
    return len({str(item.get("doc_id") or "") for item in items if item.get("doc_id")})


def _medoid_label(
    members: list[int], labels: list[str], vectors: list[list[float]]
) -> str:
    """簇内与其他成员平均相似度最高的短语，作为该簇的代表名。"""

    if len(members) == 1:
        return labels[members[0]]
    units = {index: _unit(vectors[index]) for index in members}
    best_index = members[0]
    best_score = float("-inf")
    for index in members:
        score = sum(
            _dot(units[index], units[other]) for other in members if other != index
        )
        if score > best_score:
            best_score = score
            best_index = index
    return labels[best_index]


def _extract_batch(
    question: str, batch: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    with answer_limiter:
        return extract_categories(
            question,
            batch,
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
        )


def _group_relevant(
    relevant: list[dict[str, Any]],
    settings: Settings,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """把抽取出的类别做语义归一，返回 [{label, items}]。"""

    unique_labels = sorted({item["category"] for item in relevant})
    embeddings: list[list[float]] | None = None
    if unique_labels:
        try:
            embeddings = get_embeddings(
                settings.embedding_model,
                settings.embedding_api_key,
                settings.embedding_base_url,
            ).embed_documents(unique_labels)
        except Exception as exc:
            warnings.append(f"类别向量化失败，退化为字面归类：{exc}")

    if embeddings and len(embeddings) == len(unique_labels):
        groups: list[dict[str, Any]] = []
        category_to_group: dict[str, dict[str, Any]] = {}
        for members in greedy_cluster(
            embeddings, settings.aggregate_similarity_threshold
        ):
            group = {
                "label": _medoid_label(members, unique_labels, embeddings),
                "items": [],
            }
            groups.append(group)
            for member in members:
                category_to_group[unique_labels[member]] = group
        for item in relevant:
            category_to_group[item["category"]]["items"].append(item)
        return groups

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in relevant:
        grouped.setdefault(item["category"], []).append(item)
    return [{"label": label, "items": items} for label, items in grouped.items()]


def _category_result(
    group: dict[str, Any], sample_doc_count: int
) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = group["items"]
    doc_count = count_by_doc(items)
    if not doc_count:
        return None

    evidence: list[dict[str, str]] = []
    seen_docs: set[str] = set()
    for item in items:
        doc_id = str(item.get("doc_id") or "")
        if not doc_id or doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        evidence.append(
            {
                "doc_id": doc_id,
                "chunk_id": str(item.get("chunk_id") or ""),
                "title": str(item.get("title") or ""),
                "source": str(item.get("source") or ""),
                "snippet": str(item.get("evidence") or ""),
            }
        )
        if len(evidence) >= MAX_EVIDENCE_PER_CATEGORY:
            break

    return {
        "label": group["label"],
        "doc_count": doc_count,
        "chunk_count": len(items),
        "ratio": round(doc_count / sample_doc_count, 4) if sample_doc_count else 0.0,
        "evidence": evidence,
    }


def run_aggregate(
    question: str,
    settings: Settings,
    *,
    sample_size: int,
    top_categories: int,
    retrieval_method: str | None = None,
    similarity_threshold: float | None = None,
    candidate_pool: int | None = None,
    fusion_method: str | None = None,
    fusion_k: int | None = None,
    fusion_alpha: float | None = None,
    user: UserContext | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """召回样本 → LLM 语义抽取 → 向量聚类归一 → 按文档计数。"""

    def report(stage: str, **detail: object) -> None:
        if progress is not None:
            progress(stage, detail)

    started = perf_counter()
    warnings: list[str] = []

    report("retrieve", done=0, total=sample_size)
    hits = retrieve_question(
        question,
        settings,
        retrieval_method=retrieval_method,
        top_k=sample_size,
        similarity_threshold=similarity_threshold,
        candidate_pool=candidate_pool,
        fusion_method=fusion_method,
        fusion_k=fusion_k,
        fusion_alpha=fusion_alpha,
        user=user,
    )
    sample_docs = {str(hit.get("doc_id")) for hit in hits if hit.get("doc_id")}
    sample = {"chunks": len(hits), "docs": len(sample_docs)}
    report("retrieve", done=len(hits), total=len(hits))

    if not hits:
        warnings.append("召回结果为空，无法聚合。")
        return {
            "question": question,
            "sample": sample,
            "categories": [],
            "meta": {
                "model": settings.llm_model,
                "sample_size": sample_size,
                "elapsed_ms": round((perf_counter() - started) * 1000),
                "warnings": warnings + [SAMPLE_NOTE],
            },
        }

    hit_by_chunk = {
        str(hit.get("chunk_id")): hit
        for hit in hits
        if str(hit.get("chunk_id") or "").strip()
    }
    batch_size = max(1, settings.aggregate_batch_size)
    batches = [
        hits[start : start + batch_size] for start in range(0, len(hits), batch_size)
    ]

    extracted: list[dict[str, Any]] = []
    workers = max(1, min(settings.aggregate_max_workers, settings.max_concurrent_answers))
    report("extract", done=0, total=len(batches))
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="rag-aggregate"
    ) as pool:
        futures = {
            pool.submit(_extract_batch, question, batch, settings): index
            for index, batch in enumerate(batches)
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                extracted.extend(future.result())
            except Exception as exc:
                warnings.append(f"第 {futures[future] + 1} 批抽取失败：{exc}")
            report("extract", done=done, total=len(batches))

    relevant: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for item in extracted:
        if not item.get("relevant"):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen_chunks:
            continue
        hit = hit_by_chunk.get(chunk_id)
        if hit is None:
            # 模型编造的 chunk_id 一律丢弃。
            continue
        seen_chunks.add(chunk_id)
        relevant.append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(hit.get("doc_id") or ""),
                "title": str(hit.get("title") or ""),
                "source": str(hit.get("source") or ""),
                "category": str(item.get("category") or "").strip(),
                "evidence": str(item.get("evidence") or "").strip()
                or str(hit.get("text") or "")[:MAX_EVIDENCE_CHARS],
            }
        )

    if not relevant:
        warnings.append("召回片段中没有识别出与主题相关的情形。")
        categories: list[dict[str, Any]] = []
    else:
        report("cluster", done=0, total=len(relevant))
        groups = _group_relevant(relevant, settings, warnings)
        report("cluster", done=len(relevant), total=len(relevant))

        report("count", done=0, total=len(groups))
        categories = [
            result
            for group in groups
            if (result := _category_result(group, len(sample_docs))) is not None
        ]
        categories.sort(
            key=lambda item: (-item["doc_count"], -item["chunk_count"], item["label"])
        )
        categories = categories[:top_categories]
        report("count", done=len(categories), total=len(categories))

    return {
        "question": question,
        "sample": sample,
        "categories": categories,
        "meta": {
            "model": settings.llm_model,
            "sample_size": sample_size,
            "similarity_threshold": settings.aggregate_similarity_threshold,
            "relevant_chunks": len(relevant),
            "elapsed_ms": round((perf_counter() - started) * 1000),
            "warnings": warnings + [SAMPLE_NOTE],
        },
    }
