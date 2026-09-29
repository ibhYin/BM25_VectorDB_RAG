"""持久化 BM25 索引：写入时全量重建，查询时只读 FTS5 文件。

只存排名索引（chunk_id → 分词），不存正文——正文/元数据仍是 Chroma 单一数据源。
写入低频（入库/删档），重建 O(总语料) 只发生在写入时；查询不再全量拉库、不再每次
建内存 FTS。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import chromadb

from rag.vector_store import COLLECTION_NAME

INDEX_FILENAME = "bm25.sqlite3"


def bm25_index_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / INDEX_FILENAME


def tokenize(text: str) -> list[str]:
    """jieba 搜索模式分词，剔除标点/下划线；与查询侧完全一致。"""

    import jieba

    return [
        term
        for word in jieba.cut_for_search(text.casefold())
        for term in re.findall(r"[^\W_]+", word)
    ]


def _create_index(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE VIRTUAL TABLE bm25 USING fts5(terms, chunk_id UNINDEXED)")


def rebuild_bm25_index(
    persist_dir: Path,
    embedding_model: str = "",
    embedding_api_key: str = "",
    embedding_base_url: str = "",
) -> None:
    """从 Chroma 全量重建 BM25 索引；原子写，读方永不看到半写文件。"""

    target = bm25_index_path(persist_dir)
    temporary = target.with_name(target.name + ".tmp")

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except chromadb.errors.NotFoundError:
        collection = None

    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(temporary))
    try:
        _create_index(connection)
        if collection is not None:
            records = collection.get(include=["documents", "metadatas"])
            texts = records["documents"] or []
            metadatas = records["metadatas"] or []
            rows = [
                (
                    " ".join(tokenize(text or "")),
                    str(metadatas[index].get("chunk_id", "")),
                )
                for index, text in enumerate(texts)
                if metadatas[index] and metadatas[index].get("chunk_id")
            ]
            connection.executemany(
                "INSERT INTO bm25(terms, chunk_id) VALUES (?, ?)", rows
            )
        connection.commit()
    finally:
        connection.close()
    # 原子替换：查询侧要么读到旧索引，要么读到新索引，绝无半写状态。
    temporary.replace(target)


def query_bm25_index(
    persist_dir: Path, question: str, top_k: int
) -> list[tuple[str, float]]:
    """按 BM25 分数从持久化索引取 top_k 个 (chunk_id, score)，已按分数降序。"""

    terms = list(dict.fromkeys(tokenize(question)))
    if not terms:
        return []
    connection = sqlite3.connect(
        f"file:{bm25_index_path(persist_dir)}?mode=ro", uri=True
    )
    try:
        matches = connection.execute(
            "SELECT chunk_id, bm25(bm25) FROM bm25 WHERE bm25 MATCH ?",
            (' OR '.join(f'"{term}"' for term in terms),),
        ).fetchall()
    finally:
        connection.close()

    # FTS5 的 bm25() 越低越优，取负转成"越高越好"（与内存路径 _bm25_scores 一致）。
    ranked = sorted(
        ((chunk_id, -score) for chunk_id, score in matches if score is not None),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:top_k]
