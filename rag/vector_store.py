import hashlib
import atexit
import fcntl
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag.index_manifest import ensure_index_embedding_model, read_index_embedding_model


COLLECTION_NAME = "knowledge_base"

# Chroma 的 embeddings queue 单次最多接受 max_batch_size 条（本机 5461）。一次
# 入库可能产生上万切片，所以所有写操作都按这个保守批大小切分。
CHROMA_WRITE_BATCH_SIZE = 1000


def _batches(values: list, size: int | None = None):
    batch_size = max(1, size or CHROMA_WRITE_BATCH_SIZE)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _add_documents(
    vector_store: Chroma, documents: list[Document], ids: list[str]
) -> None:
    batch_size = max(1, CHROMA_WRITE_BATCH_SIZE)
    for start in range(0, len(documents), batch_size):
        vector_store.add_documents(
            documents[start : start + batch_size],
            ids=ids[start : start + batch_size],
        )


def _delete_ids(vector_store: Chroma, ids: list[str]) -> None:
    for batch in _batches(ids):
        vector_store.delete(ids=batch)

# chromadb 用进程级引用计数共享一个 System（按 persist_directory 索引）。每次调用都
# 新建 Chroma 意味着每次创建都 +2、每次回收都可能 -2 归零，于是 System 被反复
# stop()；而「创建客户端」这一步要打开 sqlite 校验 tenant，和并发写入撞上就会
# ValueError: Could not connect to tenant，紧接着 chromadb 的清理路径再用一次
# 二次 stop() 把它掩盖成 AttributeError: bindings。缓存住客户端让计数稳定不变，
# 创建只发生一次。
_store_cache: dict[tuple, Chroma] = {}
_store_cache_lock = Lock()
INDEX_ROOT = Path("data/indexes")
_directory_leases = {}
_closed_directories: set[str] = set()
_failed_closes: set[int] = set()


@contextmanager
def index_directory_lock(directory: Path, *, exclusive: bool = False, retire: bool = False):
    """Cooperating web/CLI processes share leases; deletion requires exclusivity.

    ponytail: local POSIX file locks cover this single-machine demo; external
    programs bypassing these helpers must be stopped before maintenance.
    Lock files stay outside generations and must never be unlinked while live.
    """
    directory = Path(directory).resolve()
    lock_dir = INDEX_ROOT / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / hashlib.sha256(str(directory).encode()).hexdigest()
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("索引目录仍被其他服务或 CLI 使用，稍后重试清理") from exc
        try:
            handle.seek(0)
            if not exclusive and handle.read().strip() == "retired":
                raise RuntimeError("索引目录已退役，请重新加载当前配置")
            yield
            if retire:
                if not exclusive:
                    raise ValueError("退役需要独占目录锁")
                handle.seek(0)
                handle.truncate()
                handle.write("retired")
                handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def close_vector_stores(directory: Path) -> None:
    """Retire every config/metadata client for a directory, using public close().

    The caller holds knowledge_base_lock.write() after active readers finish;
    candidates are owned by their rebuild worker. Failed closes remain cached
    and blocked so cleanup can retry rather than delete an open database.
    """
    normalized = str(Path(directory).resolve())
    with _store_cache_lock:
        _closed_directories.add(normalized)
        keys = [key for key in _store_cache if str(Path(key[0]).resolve()) == normalized]
        clients = {id(_store_cache[key]._client): _store_cache[key]._client for key in keys}
        errors = []
        for client in clients.values():
            try:
                if id(client) in _failed_closes and getattr(client, "_closed", False):
                    raise RuntimeError("客户端上次关闭异常，需重启服务后重试清理")
                client.close()
            except Exception as exc:
                _failed_closes.add(id(client))
                errors.append(exc)
            else:
                _failed_closes.discard(id(client))
        if errors:
            raise errors[0]
        for key in keys:
            del _store_cache[key]
        lease = _directory_leases.pop(normalized, None)
        if lease is not None:
            lease.__exit__(None, None, None)


def close_all_vector_stores() -> None:
    for directory in {key[0] for key in list(_store_cache)}:
        try:
            close_vector_stores(Path(directory))
        except Exception:
            pass  # Process exit leaves durable cleanup pending for next startup.


atexit.register(close_all_vector_stores)


@contextmanager
def metadata_client(directory: Path):
    """Short-lived non-embedding reads share the same cross-process lease."""
    import chromadb

    normalized = str(Path(directory).resolve())
    with index_directory_lock(directory):
        with _store_cache_lock:
            if normalized in _closed_directories:
                raise RuntimeError("索引目录已关闭，请刷新后使用当前索引")
            client = chromadb.PersistentClient(path=normalized)
        try:
            yield client
        finally:
            client.close()


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
    batch_size: int = 10,
) -> OpenAIEmbeddings:
    kwargs = {"api_key": embedding_api_key} if embedding_api_key else {}
    kwargs["chunk_size"] = batch_size
    kwargs["request_timeout"] = 60
    kwargs["max_retries"] = 2
    if embedding_base_url:
        # ponytail: custom gateways enforce their own context limits.
        kwargs["base_url"] = embedding_base_url
        kwargs["check_embedding_ctx_length"] = False
    return OpenAIEmbeddings(model=model, **kwargs)


def get_vector_store(
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    collection_name: str = COLLECTION_NAME,
    *,
    metadata_only: bool = False,
    embedding_batch_size: int = 10,
) -> Chroma:
    """One long-lived Chroma client per configuration.

    每次请求都新建客户端在 chromadb 里代价很高也更脆：见 `_store_cache` 的注释。
    同配置复用同一个客户端，并发读写由 chromadb 自己处理，写入的互斥由调用方的
    knowledge_base_lock 负责。
    """

    persist_dir = Path(persist_dir).resolve()
    ensure_index_embedding_model(persist_dir, embedding_model)
    key = (
        str(persist_dir),
        embedding_model,
        embedding_api_key,
        embedding_base_url,
        collection_name,
        metadata_only,
        embedding_batch_size,
    )
    with _store_cache_lock:
        if str(persist_dir) in _closed_directories:
            raise RuntimeError("索引目录已关闭，请刷新后使用当前索引")
        store = _store_cache.get(key)
        if store is None:
            lease = None
            if str(persist_dir) not in _directory_leases:
                lease = index_directory_lock(persist_dir)
                lease.__enter__()
            try:
                store = Chroma(
                    collection_name=collection_name,
                    embedding_function=get_embeddings(
                        embedding_model,
                        embedding_api_key,
                        embedding_base_url,
                        embedding_batch_size,
                    ) if not metadata_only else None,
                    persist_directory=str(persist_dir),
                )
            except BaseException:
                if lease is not None:
                    lease.__exit__(None, None, None)
                raise
            if lease is not None:
                _directory_leases[str(persist_dir)] = lease
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
    embedding_batch_size: int = 10,
) -> Chroma:
    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
        embedding_batch_size=embedding_batch_size,
    )

    document_ids = [build_vector_id(document) for document in documents]
    existing_ids: set[str] = set()
    if rebuild:
        kwargs: dict[str, object] = {"include": []}
        if knowledge_base_id:
            kwargs["where"] = {"knowledge_base_id": knowledge_base_id}
        existing_ids = set(vector_store.get(**kwargs).get("ids", []))
    if documents:
        _add_documents(vector_store, documents, document_ids)
    stale_ids = existing_ids.difference(document_ids)
    if stale_ids:
        # Delete stale records only after all new embeddings were written successfully.
        _delete_ids(vector_store, sorted(stale_ids))

    return vector_store


def replace_documents(
    documents: list[Document],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    knowledge_base_id: str | None = None,
    embedding_batch_size: int = 10,
) -> Chroma:
    """Upsert documents and remove stale chunks after the new vectors are stored."""

    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
        embedding_batch_size=embedding_batch_size,
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
        _add_documents(vector_store, documents, document_ids)
    stale_ids = old_ids.difference(document_ids)
    if stale_ids:
        _delete_ids(vector_store, sorted(stale_ids))

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
        read_index_embedding_model(persist_dir) or embedding_model,
        embedding_api_key,
        embedding_base_url,
        metadata_only=True,
    )
    where: dict[str, object] = {"doc_id": {"$in": unique_ids}}
    if knowledge_base_id:
        where = {"$and": [where, {"knowledge_base_id": knowledge_base_id}]}
    existing = vector_store.get(where=where, include=[])
    stale_ids = sorted(existing.get("ids", []))
    if stale_ids:
        _delete_ids(vector_store, stale_ids)
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
    batch_size = max(1, CHROMA_WRITE_BATCH_SIZE)
    for start in range(0, len(ids), batch_size):
        vector_store._collection.update(
            ids=ids[start : start + batch_size],
            metadatas=updated[start : start + batch_size],
        )
    return len(ids)
