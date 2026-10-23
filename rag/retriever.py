import logging
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.documents import Document

from app.access import AccessPolicy, UserContext


logger = logging.getLogger(__name__)


def format_documents(documents: list[Document]) -> list[str]:
    formatted: list[str] = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        doc_id = document.metadata.get("doc_id", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        title = document.metadata.get("title", "")
        formatted.append(
            f"[Source Chunk {index}]\n"
            f"chunk_id: {chunk_id}\n"
            f"doc_id: {doc_id}\n"
            f"title: {title}\n"
            f"source: {source}\n"
            f"text: {document.page_content}"
        )

    return formatted


def build_vector_hits(
    results: list[tuple[Document, float]],
    retrieval_method: str = "vector",
) -> list[dict[str, object]]:
    """Keep the existing hit payload and citations for both retrieval methods."""

    hits: list[dict[str, object]] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        scores = (
            {"bm25_score": float(score)}
            if retrieval_method == "bm25"
            else {
                "distance": float(score),
                "similarity": max(0.0, min(1.0, 1.0 - float(score) / math.sqrt(2))),
            }
        )
        hits.append(
            {
                "rank": rank,
                "retrieval_method": retrieval_method,
                **scores,
                "text": document.page_content,
                "chunk_id": str(metadata.get("chunk_id", "unknown")),
                "doc_id": str(metadata.get("doc_id", "unknown")),
                "title": str(metadata.get("title", "")),
                "source": str(metadata.get("source", "unknown")),
                "page": metadata.get("page"),
                "page_label": str(metadata.get("page_label", "")),
                "section_path": str(metadata.get("section_path", "")),
            }
        )
    return hits


def search_bm25(
    question: str,
    persist_dir: Path,
    top_k: int,
    knowledge_base_id: str = "default",
    user: UserContext | None = None,
    access_policy: AccessPolicy | None = None,
) -> list[dict[str, object]]:
    """Rank current Chroma chunk text using jieba and SQLite FTS5 BM25.

    优先查持久化索引（写入时由 `rebuild_bm25_index` 全量重建）；索引缺失/损坏时
    回退到内存全量重建，行为与过去完全一致。
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not (persist_dir / "chroma.sqlite3").is_file():
        return []

    from rag.bm25_index import bm25_index_path

    if bm25_index_path(persist_dir).is_file():
        try:
            return _search_bm25_index(
                question, persist_dir, top_k, knowledge_base_id, user, access_policy
            )
        except sqlite3.DatabaseError:
            # 索引损坏（含旧版本留下的半写文件）时退回内存重建，别让检索整个失败。
            logger.warning("BM25 索引无法读取，回退到内存重建", exc_info=True)

    return _search_bm25_memory(
        question, persist_dir, top_k, knowledge_base_id, user, access_policy
    )


def _search_bm25_memory(
    question: str,
    persist_dir: Path,
    top_k: int,
    knowledge_base_id: str,
    user: UserContext | None,
    access_policy: AccessPolicy | None,
) -> list[dict[str, object]]:
    """回退路径：没有持久化索引（如测试的裸 chroma 目录、首次启动）时走内存全量重建。"""

    import chromadb

    from rag.vector_store import COLLECTION_NAME, metadata_client

    with metadata_client(persist_dir) as client:
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        except chromadb.errors.NotFoundError:
            return []
        records = collection.get(include=["documents", "metadatas"])
    texts = records["documents"] or []
    metadatas = records["metadatas"] or []
    if access_policy is not None and user is not None:
        allowed = [
            index
            for index, metadata in enumerate(metadatas)
            if access_policy.can_access(metadata or {}, user, knowledge_base_id)
        ]
        texts = [texts[index] for index in allowed]
        metadatas = [metadatas[index] for index in allowed]
    if not texts:
        return []
    scores = _bm25_scores(texts, question)
    ranked = sorted(
        (index for index, score in enumerate(scores) if score > 0),
        key=lambda index: (-scores[index], index),
    )[:top_k]
    return build_vector_hits(
        [
            (
                Document(
                    page_content=texts[i],
                    metadata=(metadatas[i] or {}),
                ),
                scores[i],
            )
            for i in ranked
        ],
        retrieval_method="bm25",
    )


def _search_bm25_index(
    question: str,
    persist_dir: Path,
    top_k: int,
    knowledge_base_id: str,
    user: UserContext | None,
    access_policy: AccessPolicy | None,
) -> list[dict[str, object]]:
    """持久化索引快路径：索引给排名，正文/元数据仍从 Chroma 按 chunk_id 定向取。

    有权限过滤时不能只取全局 top_k——无权限结果会占满配额；索引按页读，
    补足到 top_k 条允许结果为止。
    """

    from rag.bm25_index import iter_bm25_matches

    import chromadb

    from rag.vector_store import COLLECTION_NAME, metadata_client

    needs_filter = access_policy is not None and user is not None
    page_size = max(top_k, 100) if needs_filter else top_k

    with metadata_client(persist_dir) as client:
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        except chromadb.errors.NotFoundError:
            return []

        documents: list[tuple[Document, float]] = []
        matches = iter_bm25_matches(persist_dir, question, page_size=page_size)
        try:
            for page in matches:
                records = collection.get(
                    where={"chunk_id": {"$in": [chunk_id for chunk_id, _ in page]}},
                    include=["documents", "metadatas"],
                )
                by_id = {
                    str(metadata.get("chunk_id", "")): (text or "", metadata or {})
                    for text, metadata in zip(
                        records["documents"] or [], records["metadatas"] or []
                    )
                }
                for chunk_id, score in page:
                    if chunk_id not in by_id:
                        continue
                    text, metadata = by_id[chunk_id]
                    if needs_filter and not access_policy.can_access(
                        metadata, user, knowledge_base_id
                    ):
                        continue
                    documents.append(
                        (Document(page_content=text, metadata=metadata), score)
                    )
                    if len(documents) >= top_k:
                        break
                if len(documents) >= top_k:
                    break
        finally:
            matches.close()

    return build_vector_hits(documents, retrieval_method="bm25")


def _tokenize(text: str) -> list[str]:
    from rag.bm25_index import tokenize

    return tokenize(text)


def _bm25_scores(texts: list[str], question: str) -> list[float]:
    """Score every text with SQLite FTS5 BM25; zero means no query term matched.

    仅供 hybrid 重排（候选池小）与无持久化索引时的回退使用；单条 BM25 检索优先
    走 `search_bm25` 里的持久化索引快路径。
    """

    terms = list(dict.fromkeys(_tokenize(question)))
    scores = [0.0] * len(texts)
    if not terms or not texts:
        return scores

    # ponytail: rebuild from the given texts on each call, O(total text). This
    # avoids stale caches after any writer; large corpora need a persisted FTS index.
    with closing(sqlite3.connect(":memory:")) as index:
        index.execute("CREATE VIRTUAL TABLE chunks USING fts5(text)")
        index.executemany(
            "INSERT INTO chunks(rowid, text) VALUES (?, ?)",
            ((i, " ".join(_tokenize(text or ""))) for i, text in enumerate(texts)),
        )
        matches = index.execute(
            "SELECT rowid, bm25(chunks) FROM chunks WHERE chunks MATCH ?",
            (' OR '.join(f'"{term}"' for term in terms),),
        ).fetchall()

    # FTS5 negates BM25 for ascending SQL ranking; expose higher-is-better scores.
    for rowid, score in matches:
        scores[rowid] = -score
    return scores


def rerank_bm25_scores(texts: list[str], question: str) -> list[float]:
    """BM25-score already-retrieved candidates; aligned with the input order."""

    return _bm25_scores(texts, question)


def _chunk_key(hit: dict[str, object]) -> str:
    return str(hit.get("chunk_id", ""))


def _merge_paths(
    vector_hits: list[dict[str, object]],
    bm25_hits: list[dict[str, object]],
) -> list[dict[str, object]]:
    """按 chunk_id 合并两路召回，保留各自的名次与分数。

    某一路没有命中某条时，该路的 rank 为 None、分数字段缺失——不补 0，也不伪造
    名次。顺序是「向量路在前，BM25 独有结果按 BM25 名次追加」，融合排序的并列项
    按这个顺序稳定排列。
    """

    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []

    for rank, hit in enumerate(vector_hits, start=1):
        key = _chunk_key(hit)
        if not key or key in merged:
            continue
        item = dict(hit)
        item.pop("rank", None)  # 最终名次由融合重新编号
        item["retrieval_method"] = "hybrid"
        item["vector_rank"] = rank
        item["bm25_rank"] = None
        item["sources"] = ["vector"]
        merged[key] = item
        order.append(key)

    for rank, hit in enumerate(bm25_hits, start=1):
        key = _chunk_key(hit)
        if not key:
            continue
        item = merged.get(key)
        if item is None:
            item = dict(hit)
            item.pop("rank", None)
            item["retrieval_method"] = "hybrid"
            item["vector_rank"] = None
            item["sources"] = ["bm25"]
            merged[key] = item
            order.append(key)
        else:
            item["sources"] = list(item["sources"]) + ["bm25"]
        item["bm25_rank"] = rank
        item["bm25_score"] = float(hit.get("bm25_score") or 0.0)

    return [merged[key] for key in order]


def _assign_fusion_ranks(
    items: list[dict[str, object]], score_key: str
) -> list[dict[str, object]]:
    ranked = sorted(
        enumerate(items),
        key=lambda pair: (-float(pair[1][score_key]), pair[0]),
    )
    return [
        {**item, "rank": position}
        for position, (_, item) in enumerate(ranked, start=1)
    ]


def rrf_fuse_paths(
    vector_hits: list[dict[str, object]],
    bm25_hits: list[dict[str, object]],
    k: int = 60,
) -> list[dict[str, object]]:
    """对两路独立召回做 RRF；某一路缺席时该路贡献为零。"""

    items = _merge_paths(vector_hits, bm25_hits)
    for item in items:
        score = 0.0
        if item["vector_rank"] is not None:
            score += 1.0 / (k + int(item["vector_rank"]))
        if item["bm25_rank"] is not None:
            score += 1.0 / (k + int(item["bm25_rank"]))
        item["rrf_score"] = score
    return _assign_fusion_ranks(items, "rrf_score")


def weighted_fuse_paths(
    vector_hits: list[dict[str, object]],
    bm25_hits: list[dict[str, object]],
    alpha: float = 0.5,
) -> list[dict[str, object]]:
    """按归一化分数融合两路独立召回；某一路缺席时该路贡献为零。

    约定：向量侧用 `build_vector_hits` 产出的 similarity（由 L2 距离换算、已
    clamp 到 [0,1]）；BM25 侧用该路最高分归一化（`bm25_score / max`）。相似度
    本身未校准，因此这是近似排序；默认仍用 RRF。
    """

    items = _merge_paths(vector_hits, bm25_hits)
    bm25_max = max(
        (float(hit.get("bm25_score") or 0.0) for hit in bm25_hits), default=0.0
    )
    for item in items:
        similarity = float(item.get("similarity") or 0.0)
        bm25_score = float(item.get("bm25_score") or 0.0)
        normalized = bm25_score / bm25_max if bm25_max > 0 else 0.0
        item["weighted_score"] = alpha * similarity + (1.0 - alpha) * normalized
    return _assign_fusion_ranks(items, "weighted_score")


def format_vector_hits(hits: list[dict[str, object]]) -> list[str]:
    return [
        (
            f"[Source Chunk {hit['rank']}]\n"
            f"chunk_id: {hit['chunk_id']}\n"
            f"doc_id: {hit['doc_id']}\n"
            f"title: {hit['title']}\n"
            f"source: {hit['source']}\n"
            f"text: {hit['text']}"
        )
        for hit in hits
    ]
