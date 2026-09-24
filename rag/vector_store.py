import hashlib
from pathlib import Path
from threading import Lock

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


COLLECTION_NAME = "knowledge_base"

# chromadb 用进程级引用计数共享一个 System（按 persist_directory 索引）。每次调用都
# 新建 Chroma 意味着每次创建都 +2、每次回收都可能 -2 归零，于是 System 被反复
# stop()；而「创建客户端」这一步要打开 sqlite 校验 tenant，和并发写入撞上就会
# ValueError: Could not connect to tenant，紧接着 chromadb 的清理路径再用一次
# 二次 stop() 把它掩盖成 AttributeError: bindings。缓存住客户端让计数稳定不变，
# 创建只发生一次。
_store_cache: dict[tuple, Chroma] = {}
_store_cache_lock = Lock()


def build_vector_id(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return chunk_id
    identity = "\u241f".join(
        [
            str(document.metadata.get("source", "unknown")),
            str(document.metadata.get("page", "")),
            document.page_content,
        ]
    )
    return "document_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def get_embeddings(
    model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
) -> OpenAIEmbeddings:
    kwargs = {"api_key": embedding_api_key} if embedding_api_key else {}
    kwargs["request_timeout"] = 60
    kwargs["max_retries"] = 2
    if embedding_base_url:
        # ponytail: custom gateways enforce their own context limits.
        kwargs["base_url"] = embedding_base_url
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 10
    return OpenAIEmbeddings(model=model, **kwargs)


def get_vector_store(
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """One long-lived Chroma client per configuration.

    每次请求都新建客户端在 chromadb 里代价很高也更脆：见 `_store_cache` 的注释。
    同配置复用同一个客户端，并发读写由 chromadb 自己处理，写入的互斥由调用方的
    knowledge_base_lock 负责。
    """

    key = (
        str(persist_dir),
        embedding_model,
        embedding_api_key,
        embedding_base_url,
        collection_name,
    )
    with _store_cache_lock:
        store = _store_cache.get(key)
        if store is None:
            store = Chroma(
                collection_name=collection_name,
                embedding_function=get_embeddings(
                    embedding_model,
                    embedding_api_key,
                    embedding_base_url,
                ),
                persist_directory=str(persist_dir),
            )
            _store_cache[key] = store
        return store


def ingest_documents(
    documents: list[Document],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    rebuild: bool = False,
    knowledge_base_id: str | None = None,
) -> Chroma:
    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
    )

    document_ids = [build_vector_id(document) for document in documents]
    existing_ids: set[str] = set()
    if rebuild:
        kwargs: dict[str, object] = {"include": []}
        if knowledge_base_id:
            kwargs["where"] = {"knowledge_base_id": knowledge_base_id}
        existing_ids = set(vector_store.get(**kwargs).get("ids", []))
    if documents:
        vector_store.add_documents(documents, ids=document_ids)
    stale_ids = existing_ids.difference(document_ids)
    if stale_ids:
        # Delete stale records only after all new embeddings were written successfully.
        vector_store.delete(ids=sorted(stale_ids))

    return vector_store


def replace_documents(
    documents: list[Document],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    knowledge_base_id: str | None = None,
) -> Chroma:
    """Upsert documents and remove stale chunks after the new vectors are stored."""

    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
    )
    document_ids = [build_vector_id(document) for document in documents]
    doc_ids = sorted(
        {
            str(document.metadata.get("doc_id", ""))
            for document in documents
            if document.metadata.get("doc_id")
        }
    )
    old_ids: set[str] = set()
    if doc_ids:
        where: dict[str, object] = {"doc_id": {"$in": doc_ids}}
        if knowledge_base_id:
            where = {
                "$and": [where, {"knowledge_base_id": knowledge_base_id}],
            }
        existing = vector_store.get(
            where=where,
            include=[],
        )
        old_ids = set(existing.get("ids", []))

    if documents:
        vector_store.add_documents(documents, ids=document_ids)
    stale_ids = old_ids.difference(document_ids)
    if stale_ids:
        vector_store.delete(ids=sorted(stale_ids))

    return vector_store


def delete_documents(
    doc_ids: list[str],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    knowledge_base_id: str | None = None,
) -> int:
    """Remove every chunk of the given documents, scoped to one knowledge base."""

    unique_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique_ids:
        return 0
    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
    )
    where: dict[str, object] = {"doc_id": {"$in": unique_ids}}
    if knowledge_base_id:
        where = {"$and": [where, {"knowledge_base_id": knowledge_base_id}]}
    existing = vector_store.get(where=where, include=[])
    stale_ids = sorted(existing.get("ids", []))
    if stale_ids:
        vector_store.delete(ids=stale_ids)
    return len(stale_ids)


def set_document_access_group(
    doc_ids: list[str],
    groups: list[str],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    knowledge_base_id: str | None = None,
    all_groups: list[str] | None = None,
) -> int:
    """Rewrite the access-group metadata on the stored chunks of these documents.

    The groups are normally baked in at split time, so changing them afterwards
    would mean re-indexing the document. Updating only the metadata keeps the
    vectors untouched — no embedding calls, and no risk to what the index already
    recalls.

    Each chunk carries three representations: ``document_access_groups`` (the
    full list, for display and Python-side filtering), ``document_access_group``
    (the primary group — a scalar kept for backward compatibility), and one
    boolean flag ``access_<group>`` per known group (chromadb can't filter on
    list values, so retrieval matches on the flags).
    """

    unique_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique_ids:
        return 0
    clean_groups = sorted({str(group).strip() for group in groups if str(group).strip()})
    if not clean_groups:
        raise ValueError("at least one group is required")
    primary = clean_groups[0]

    vector_store = get_vector_store(
        persist_dir, embedding_model, embedding_api_key, embedding_base_url
    )
    where: dict[str, object] = {"doc_id": {"$in": unique_ids}}
    if knowledge_base_id:
        where = {"$and": [where, {"knowledge_base_id": knowledge_base_id}]}

    existing = vector_store.get(where=where, include=["metadatas"])
    ids = list(existing.get("ids", []) or [])
    metadatas = list(existing.get("metadatas", []) or [])
    if not ids:
        return 0

    updated = []
    for metadata in metadatas:
        merged = dict(metadata or {})
        merged["document_access_group"] = primary
        merged["document_access_groups"] = clean_groups
        # 非成员组写 False：清除改组前遗留的陈旧 True flag。
        for group in all_groups or []:
            merged["access_" + group] = group in clean_groups
        updated.append(merged)

    # ponytail: langchain_chroma exposes no public metadata-only update, so reach
    # through to the underlying chromadb collection.
    vector_store._collection.update(ids=ids, metadatas=updated)
    return len(ids)
