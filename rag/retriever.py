import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.documents import Document

from app.access import AccessPolicy, UserContext


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
    """Rank current Chroma chunk text using jieba and SQLite FTS5 BM25."""

    import chromadb

    from rag.vector_store import COLLECTION_NAME

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not (persist_dir / "chroma.sqlite3").is_file():
        return []
    client = chromadb.PersistentClient(path=str(persist_dir))
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


def _tokenize(text: str) -> list[str]:
    import jieba

    return [
        term
        for word in jieba.cut_for_search(text.casefold())
        for term in re.findall(r"[^\W_]+", word)
    ]


def _bm25_scores(texts: list[str], question: str) -> list[float]:
    """Score every text with SQLite FTS5 BM25; zero means no query term matched."""

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


def rrf_fuse(
    vector_hits: list[dict[str, object]],
    bm25_scores: list[float],
    k: int = 60,
) -> list[dict[str, object]]:
    """Reciprocal Rank Fusion of the vector order with candidate BM25 scores."""

    if len(vector_hits) != len(bm25_scores):
        raise ValueError("vector_hits and bm25_scores must have the same length")

    bm25_order = sorted(
        (index for index, score in enumerate(bm25_scores) if score > 0),
        key=lambda index: (-bm25_scores[index], index),
    )
    bm25_rank = {index: rank + 1 for rank, index in enumerate(bm25_order)}

    fused: list[tuple[float, int, dict[str, object]]] = []
    for index, hit in enumerate(vector_hits):
        score = 1.0 / (k + index + 1)
        if index in bm25_rank:
            score += 1.0 / (k + bm25_rank[index])
        merged = dict(hit)
        merged["retrieval_method"] = "hybrid"
        merged["bm25_score"] = float(bm25_scores[index])
        merged["rrf_score"] = score
        fused.append((score, index, merged))

    fused.sort(key=lambda item: (-item[0], item[1]))
    # 重新编号。排序变了，而 `build_vector_hits` 给的 `rank` 还是**向量候选里的**
    # 名次——直接用会让界面上第一张卡片写着「#4」。评估页正是靠这个名次判断检索
    # 好坏的，糊弄不得。
    return [
        {**hit, "rank": position}
        for position, (_, _, hit) in enumerate(fused, start=1)
    ]


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
