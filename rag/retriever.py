import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.documents import Document


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
    question: str, persist_dir: Path, top_k: int,
) -> list[dict[str, object]]:
    """Rank current Chroma chunk text using jieba and SQLite FTS5 BM25."""

    import chromadb
    import jieba

    from rag.vector_store import COLLECTION_NAME

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    def tokenize(text: str) -> list[str]:
        return [
            term
            for word in jieba.cut_for_search(text.casefold())
            for term in re.findall(r"[^\W_]+", word)
        ]

    terms = list(dict.fromkeys(tokenize(question)))
    if not terms or not (persist_dir / "chroma.sqlite3").is_file():
        return []
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except chromadb.errors.NotFoundError:
        return []
    records = collection.get(include=["documents", "metadatas"])
    texts = records["documents"] or []
    if not texts:
        return []

    # ponytail: rebuild from current chunks on each query, O(total text). This
    # avoids stale caches after any writer; large corpora need a persisted FTS index.
    with closing(sqlite3.connect(":memory:")) as index:
        index.execute("CREATE VIRTUAL TABLE chunks USING fts5(text)")
        index.executemany(
            "INSERT INTO chunks(rowid, text) VALUES (?, ?)",
            ((i, " ".join(tokenize(text or ""))) for i, text in enumerate(texts)),
        )
        matches = index.execute(
            "SELECT rowid, bm25(chunks) FROM chunks "
            "WHERE chunks MATCH ? ORDER BY bm25(chunks), rowid LIMIT ?",
            (' OR '.join(f'"{term}"' for term in terms), top_k),
        ).fetchall()

    # FTS5 negates BM25 for ascending SQL ranking; expose higher-is-better scores.
    return build_vector_hits(
        [
            (
                Document(
                    page_content=texts[i],
                    metadata=(records["metadatas"][i] or {}),
                ),
                -score,
            )
            for i, score in matches
        ],
        retrieval_method="bm25",
    )


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
