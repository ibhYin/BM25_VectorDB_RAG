import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from threading import BoundedSemaphore, Condition
from typing import TypedDict

from app.config import Settings, get_settings
from app.access import (
    PUBLIC_GROUP,
    AccessPolicy,
    UserContext,
    known_groups,
    load_access_policy,
    set_document_groups,
)
from app import audit
from rag.bm25_index import rebuild_bm25_index
from rag import markdown_cache
from rag.indexer import build_vector_index
from rag.index_manifest import ensure_index_embedding_model, write_index_manifest
from rag.loaders import load_document_file
from rag.retriever import (
    build_vector_hits,
    format_vector_hits,
    rrf_fuse_paths,
    search_bm25,
    weighted_fuse_paths,
)
from rag.splitter import split_documents
from rag.structurizer import MINERU
from rag.vector_store import (
    delete_documents,
    get_embeddings,
    get_vector_store,
    replace_documents,
    set_document_access_group,
)


logger = logging.getLogger(__name__)


class KnowledgeBaseLock:
    """Process-local readers/writer lock for the shared knowledge-base state."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._mutations = 0
        self._rebuilding = False

    @contextmanager
    def mutation(self):
        # Reserve briefly; remote calls never hold this condition.
        with self._condition:
            if self._rebuilding:
                raise RuntimeError("索引正在重建，暂时不能上传、删除或修改索引/配置")
            self._mutations += 1
        try:
            yield
        finally:
            with self._condition:
                self._mutations -= 1

    def begin_rebuild(self):
        with self._condition:
            if self._rebuilding or self._mutations or self._writer or self._waiting_writers:
                raise RuntimeError("已有索引写入或重建任务，请等待结束后重试")
            self._rebuilding = True

    def end_rebuild(self):
        with self._condition:
            self._rebuilding = False

    @contextmanager
    def read(self, wait: bool = True):
        with self._condition:
            while self._writer or self._waiting_writers:
                if not wait:
                    raise RuntimeError("索引正在写入，请等待任务完成后重试")
                self._condition.wait()
            self._readers += 1

        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


# Readers (ask) run concurrently; a writer (ingest) gets exclusive access so
# Chroma is never read while this process is rebuilding it. Waiting writers
# block new readers to prevent a stream of questions from starving ingest.
knowledge_base_lock = KnowledgeBaseLock()
answer_limiter = BoundedSemaphore(get_settings().max_concurrent_answers)


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int
    rebuild: bool
    data_dir: Path


FileProgressCallback = Callable[
    [str, int, int, str, dict[str, object] | None], None
]


@dataclass(frozen=True)
class FileIngestResult:
    files: int
    documents: int
    chunks: int
    failed_files: tuple[str, ...] = ()
    cancelled: bool = False


@knowledge_base_lock.mutation()
def ingest_knowledge_base(
    settings: Settings,
    data_dir: Path | None = None,
    rebuild: bool = False,
    structurize: bool = False,
    actor: str = "cli",
) -> IngestResult:
    """Build the vector index from every supported document in one directory.

    When structurize is True, documents are converted to Markdown via MinerU
    before chunking, preserving headings, lists, and tables.
    """

    with knowledge_base_lock.write():
        source_dir = data_dir or settings.data_dir
        documents, chunks = build_vector_index(
            data_dir=source_dir,
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            ocr_model=settings.ocr_model,
            ocr_max_pages=settings.ocr_max_pages,
            embedding_max_input_bytes=settings.embedding_max_input_bytes,
            embedding_batch_size=settings.embedding_batch_size,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            rebuild=rebuild,
            knowledge_base_id=settings.knowledge_base_id,
            access_policy_file=settings.access_policy_file,
            structurize=structurize,
            mineru_token=settings.mineru_api_token,
            mineru_model_version=settings.mineru_model_version,
            audio_transcribe_base_url=settings.audio_transcribe_base_url,
            audio_transcribe_api_key=settings.audio_transcribe_api_key,
            audio_transcribe_model=settings.audio_transcribe_model,
            audio_transcribe_provider=settings.audio_transcribe_provider,
            aliyun_access_key_id=settings.aliyun_access_key_id,
            aliyun_access_key_secret=settings.aliyun_access_key_secret,
            aliyun_appkey=settings.aliyun_appkey,
            aliyun_region=settings.aliyun_region,
            aliyun_oss_bucket=settings.aliyun_oss_bucket,
            aliyun_oss_endpoint=settings.aliyun_oss_endpoint,
            aliyun_oss_prefix=settings.aliyun_oss_prefix,
        )
        rebuild_bm25_index(
            settings.chroma_persist_dir,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        )

    result = IngestResult(len(documents), len(chunks), rebuild, source_dir)
    audit.record(
        "ingest.directory",
        actor=actor,
        data_dir=str(source_dir),
        knowledge_base_id=settings.knowledge_base_id,
        documents=result.documents,
        chunks=result.chunks,
        rebuild=rebuild,
        structurize=structurize,
        structurize_provider=MINERU if structurize else "",
        embedding_model=settings.embedding_model,
    )
    return result


@knowledge_base_lock.mutation()
def ingest_files(
    settings: Settings,
    paths: list[Path],
    structurize: bool = False,
    actor: str = "",
    progress: FileProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    continue_on_error: bool = False,
) -> FileIngestResult:
    """Incrementally index uploaded files without rebuilding the whole directory.

    When structurize is True, files are converted to Markdown via MinerU
    before chunking. ``progress`` reports per-file state, ``should_cancel`` is
    polled between files, and ``continue_on_error`` keeps the batch going after
    one file fails.
    """

    if not paths:
        raise ValueError("At least one file is required")

    total = len(paths)
    filenames = [Path(path).name for path in paths]
    documents: list = []
    chunks: list = []
    failed_files: list[str] = []
    file_details: list[dict[str, object]] = []
    cancelled = False
    processed = 0
    current_file = ""
    stage = "load"

    def notify(detail: dict[str, object] | None = None) -> None:
        if progress is not None:
            progress(stage, processed, total, current_file, detail)

    try:
        with knowledge_base_lock.write():
            ensure_index_embedding_model(
                settings.chroma_persist_dir, settings.embedding_model
            )
            for index, path in enumerate(paths):
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break

                stage = "load"
                current_file = path.name
                started = perf_counter()
                detail: dict[str, object] = {
                    "index": index,
                    "name": path.name,
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                    "status": "loading",
                    "cache_hit": False,
                    "documents": 0,
                    "chunks": 0,
                    "duration_ms": 0,
                    "stage": "load",
                    "error": "",
                }
                file_details.append(detail)
                notify(detail)

                try:
                    if structurize:
                        detail["cache_hit"] = markdown_cache.has_entry(
                            path, MINERU, settings.mineru_model_version
                        )
                    loaded = load_document_file(
                        path,
                        openai_api_key=settings.openai_api_key,
                        openai_base_url=settings.openai_base_url,
                        ocr_model=settings.ocr_model,
                        ocr_max_pages=settings.ocr_max_pages,
                        structurize=structurize,
                        mineru_token=settings.mineru_api_token,
                        mineru_model_version=settings.mineru_model_version,
                        audio_transcribe_base_url=settings.audio_transcribe_base_url,
                        audio_transcribe_api_key=settings.audio_transcribe_api_key,
                        audio_transcribe_model=settings.audio_transcribe_model,
                        audio_transcribe_provider=settings.audio_transcribe_provider,
                        aliyun_access_key_id=settings.aliyun_access_key_id,
                        aliyun_access_key_secret=settings.aliyun_access_key_secret,
                        aliyun_appkey=settings.aliyun_appkey,
                        aliyun_region=settings.aliyun_region,
                        aliyun_oss_bucket=settings.aliyun_oss_bucket,
                        aliyun_oss_endpoint=settings.aliyun_oss_endpoint,
                        aliyun_oss_prefix=settings.aliyun_oss_prefix,
                    )
                    if not loaded:
                        raise ValueError(f"No text was extracted from {path.name}")
                    if any(
                        document.metadata.get("ocr_required")
                        for document in loaded
                    ):
                        raise ValueError(
                            f"OCR is required for uploaded file: {path.name}"
                        )
                    ocr_errors = [
                        str(document.metadata["ocr_error"])
                        for document in loaded
                        if document.metadata.get("ocr_error")
                    ]
                    if ocr_errors:
                        raise RuntimeError(f"OCR failed: {ocr_errors[0]}")

                    detail["documents"] = len(loaded)
                    detail["stage"] = "split"
                    stage = "split"
                    file_chunks = split_documents(
                        loaded,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        knowledge_base_id=settings.knowledge_base_id,
                        access_policy=load_access_policy(settings.access_policy_file),
                        max_input_bytes=settings.embedding_max_input_bytes,
                    )
                    if not file_chunks:
                        raise ValueError(
                            f"Uploaded document contained no indexable text: "
                            f"{path.name}"
                        )
                    detail["chunks"] = len(file_chunks)
                    detail["status"] = "done"
                    detail["stage"] = "done"
                    detail["duration_ms"] = round((perf_counter() - started) * 1000)
                    documents.extend(loaded)
                    chunks.extend(file_chunks)
                except Exception as exc:
                    detail["status"] = "failed"
                    detail["stage"] = (
                        "split" if detail["stage"] == "split" else "load"
                    )
                    detail["duration_ms"] = round((perf_counter() - started) * 1000)
                    from app.rebuild import safe_error

                    detail["error"] = safe_error(exc, settings)
                    failed_files.append(path.name)
                    processed += 1
                    notify(detail)
                    if not continue_on_error:
                        raise
                else:
                    processed += 1
                    notify(detail)

            if cancelled:
                stage = "cancelled"
                for skipped_index in range(processed, total):
                    skipped_path = Path(paths[skipped_index])
                    current_file = skipped_path.name
                    skipped: dict[str, object] = {
                        "index": skipped_index,
                        "name": skipped_path.name,
                        "size_bytes": 0,
                        "status": "skipped",
                        "cache_hit": False,
                        "documents": 0,
                        "chunks": 0,
                        "duration_ms": 0,
                        "stage": "cancelled",
                        "error": "",
                    }
                    file_details.append(skipped)
                    notify(skipped)

            if chunks:
                stage = "vector_write"
                current_file = ""
                notify()
                replace_documents(
                    documents=chunks,
                    persist_dir=settings.chroma_persist_dir,
                    embedding_model=settings.embedding_model,
                    embedding_api_key=settings.embedding_api_key,
                    embedding_base_url=settings.embedding_base_url,
                    knowledge_base_id=settings.knowledge_base_id,
                    embedding_batch_size=settings.embedding_batch_size,
                )
                markdown_cache.write_documents(settings.chroma_persist_dir, documents)
                write_index_manifest(
                    settings.chroma_persist_dir, settings.embedding_model,
                    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
                )
                rebuild_bm25_index(
                    settings.chroma_persist_dir,
                    settings.embedding_model,
                    settings.embedding_api_key,
                    settings.embedding_base_url,
                )
            elif continue_on_error and (failed_files or cancelled):
                stage = "cancelled" if cancelled else "failed"
                current_file = ""
                notify()
            else:
                raise ValueError("Uploaded documents contained no indexable text")
    except Exception as exc:
        # 原异常仍原样抛给上传任务（任务页继续显示失败）；审计只写清洗后的原因，
        # 不含 traceback / Token / 预签名 URL。
        doc_ids = sorted(
            {
                str(chunk.metadata.get("doc_id", ""))
                for chunk in chunks
                if chunk.metadata.get("doc_id")
            }
        )
        audit.record(
            "ingest.files",
            actor=actor,
            knowledge_base_id=settings.knowledge_base_id,
            files=filenames,
            doc_ids=doc_ids,
            structurize=structurize,
            structurize_provider=MINERU if structurize else "",
            embedding_model=settings.embedding_model,
            **audit.failure_fields(exc, stage),
        )
        raise

    successful_files = sum(
        1 for detail in file_details if detail["status"] == "done"
    )
    result = FileIngestResult(
        files=successful_files,
        documents=len(documents),
        chunks=len(chunks),
        failed_files=tuple(failed_files),
        cancelled=cancelled,
    )
    # Chunks carry the doc_id that split_documents assigned; the log uses it to
    # correlate an ingest with the access policy.
    doc_ids = sorted(
        {
            str(chunk.metadata.get("doc_id", ""))
            for chunk in chunks
            if chunk.metadata.get("doc_id")
        }
    )
    audit.record(
        "ingest.files",
        actor=actor,
        knowledge_base_id=settings.knowledge_base_id,
        files=filenames,
        failed_files=sorted(failed_files),
        cancelled=cancelled,
        status=(
            "cancelled"
            if cancelled
            else (
                "partial"
                if failed_files and successful_files
                else ("failed" if failed_files else "success")
            )
        ),
        doc_ids=doc_ids,
        documents=result.documents,
        chunks=result.chunks,
        structurize=structurize,
        structurize_provider=MINERU if structurize else "",
        embedding_model=settings.embedding_model,
    )
    return result


def index_management_store(settings: Settings):
    from rag.index_manifest import read_index_embedding_model

    return get_vector_store(
        settings.chroma_persist_dir,
        read_index_embedding_model(settings.chroma_persist_dir) or settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
        metadata_only=True,
    )


@knowledge_base_lock.mutation()
def delete_indexed_documents(
    settings: Settings, doc_ids: list[str], actor: str = ""
) -> int:
    """Delete every chunk of the given documents from the configured knowledge base."""

    clean_doc_ids = list(
        dict.fromkeys(doc_id.strip() for doc_id in doc_ids if doc_id.strip())
    )
    if not clean_doc_ids:
        raise ValueError("At least one doc_id is required")

    summaries: dict[str, dict[str, object]] = {}
    stage = "metadata_lookup"
    try:
        with knowledge_base_lock.write():
            # 标题/来源必须在删除前快照：删完后向量库里只剩 doc_id 的审计记录，已经
            # 没地方再还原「到底删了哪篇文档」。这份快照与删除放在同一把写锁里，
            # 避免查询完元数据、真正删除之前又有一次同名文档替换进来。
            metadata_result = index_management_store(settings).get(
                where={
                    "$and": [
                        {"doc_id": {"$in": clean_doc_ids}},
                        {"knowledge_base_id": settings.knowledge_base_id},
                    ]
                },
                include=["metadatas"],
            )
            for raw_metadata in metadata_result.get("metadatas", []) or []:
                metadata = raw_metadata or {}
                doc_id = str(metadata.get("doc_id") or "")
                if not doc_id:
                    continue
                source = str(metadata.get("source") or "")
                summary = summaries.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "title": str(
                            metadata.get("title") or Path(source).stem or doc_id
                        ),
                        "source": source,
                        "chunks": 0,
                    },
                )
                summary["chunks"] = int(summary["chunks"]) + 1

            stage = "vector_delete"
            deleted = delete_documents(
                clean_doc_ids,
                persist_dir=settings.chroma_persist_dir,
                embedding_model=settings.embedding_model,
                embedding_api_key=settings.embedding_api_key,
                embedding_base_url=settings.embedding_base_url,
                knowledge_base_id=settings.knowledge_base_id,
            )
            rebuild_bm25_index(
                settings.chroma_persist_dir,
                settings.embedding_model,
                settings.embedding_api_key,
                settings.embedding_base_url,
            )
    except Exception as exc:
        audit.record(
            "delete.documents",
            actor=actor,
            knowledge_base_id=settings.knowledge_base_id,
            doc_ids=clean_doc_ids,
            deleted_documents=[
                summaries[doc_id] for doc_id in clean_doc_ids if doc_id in summaries
            ],
            embedding_model=settings.embedding_model,
            **audit.failure_fields(exc, stage),
        )
        raise

    if not deleted:
        # 底层「删了 0 片」不是异常，但对用户来说就是删除失败（文档不存在）。
        # 只记失败，不再先记一条看似成功的 0 切片记录。
        audit.record(
            "delete.documents",
            actor=actor,
            knowledge_base_id=settings.knowledge_base_id,
            doc_ids=clean_doc_ids,
            deleted_documents=[
                summaries[doc_id] for doc_id in clean_doc_ids if doc_id in summaries
            ],
            deleted_chunks=0,
            embedding_model=settings.embedding_model,
            status="failed",
            error_type="NotFound",
            error="Document not found",
            failure_stage="vector_delete",
        )
        return 0

    audit.record(
        "delete.documents",
        actor=actor,
        knowledge_base_id=settings.knowledge_base_id,
        doc_ids=clean_doc_ids,
        # 新记录给审计页人能看懂的标题；旧记录没有这个字段，前端回退到 doc_ids。
        deleted_documents=[
            summaries[doc_id] for doc_id in clean_doc_ids if doc_id in summaries
        ],
        deleted_chunks=deleted,
        embedding_model=settings.embedding_model,
    )
    return deleted


@knowledge_base_lock.mutation()
def delete_indexed_documents_batch(settings: Settings, doc_ids: list[str], actor: str = "") -> dict:
    """One shared delete and BM25 refresh; report truth after partial failures."""
    if not isinstance(doc_ids, list) or not doc_ids or len(doc_ids) > 1000:
        raise ValueError("请选择 1–1000 份文档")
    if any(not isinstance(i, str) or not i.strip() or len(i) > 300 for i in doc_ids):
        raise ValueError("文档 ID 无效")
    ids = list(dict.fromkeys(i.strip() for i in doc_ids))

    def counts():
        with knowledge_base_lock.read():
            result = index_management_store(settings).get(where={"$and": [
                {"knowledge_base_id": settings.knowledge_base_id}, {"doc_id": {"$in": ids}},
            ]}, include=["metadatas"])
        totals = {}
        for meta in result.get("metadatas") or []:
            key = str((meta or {}).get("doc_id") or "")
            totals[key] = totals.get(key, 0) + 1
        return totals

    before = counts()
    known = [i for i in ids if i in before]
    error = ""
    if known:
        try:
            delete_indexed_documents(settings, known, actor)
        except Exception as exc:
            from app.rebuild import safe_error
            error = safe_error(exc, settings)
            # A delete can fail after some chunks were removed. Refresh BM25
            # once from the surviving records so it does not retain those IDs.
            try:
                with knowledge_base_lock.write():
                    rebuild_bm25_index(settings.chroma_persist_dir)
            except Exception as refresh_exc:
                error += "；BM25 更新失败：" + safe_error(refresh_exc, settings)
    try:
        after = counts()
    except Exception as exc:
        from app.rebuild import safe_error
        return {"deleted": [], "not_deleted": [i for i in ids if i not in before],
                "failed": known, "deleted_chunks": None, "error": safe_error(exc, settings),
                "verified": False}
    return {"deleted": [i for i in known if i not in after],
            "not_deleted": [i for i in ids if i not in before],
            "failed": [i for i in known if i in after],
            "deleted_chunks": sum(before.values()) - sum(after.values()),
            "error": error, "verified": True}


@knowledge_base_lock.mutation()
def set_document_access(
    settings: Settings,
    doc_id: str,
    groups: list[str],
    actor: str = "",
) -> int:
    """Point one indexed document at a set of access groups. Admin action.

    Updates both halves of the truth: the policy file (what future indexing will
    use) and the stored chunk metadata (what retrieval filters on right now).
    Only metadata is rewritten, so this costs no embedding calls.
    """

    doc_id = doc_id.strip()
    clean_groups = sorted({str(group).strip() for group in groups if str(group).strip()})
    if not doc_id:
        raise ValueError("doc_id is required")
    if not clean_groups:
        raise ValueError("at least one group is required")

    policy_path = settings.access_policy_file
    if not policy_path.is_file():
        raise ValueError("未启用访问控制：缺少访问策略文件")
    # 只允许已存在的组，避免拼错把文档变成谁都看不见。
    known = known_groups(policy_path)
    unknown = sorted(set(clean_groups) - known)
    if unknown:
        raise ValueError(f"未知的访问组：{'、'.join(unknown)}")

    with knowledge_base_lock.write():
        updated = set_document_access_group(
            [doc_id],
            clean_groups,
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            knowledge_base_id=settings.knowledge_base_id,
            all_groups=sorted(known),
        )
        if not updated:
            raise KeyError("Document not found")
        # 先确认文档存在再落策略，避免给不存在的 doc_id 留下垃圾条目。
        set_document_groups(policy_path, doc_id, clean_groups)

    audit.record(
        "access.change",
        actor=actor,
        knowledge_base_id=settings.knowledge_base_id,
        doc_ids=[doc_id],
        access_groups=clean_groups,
        chunks=updated,
    )
    return updated


@knowledge_base_lock.mutation()
def approve_staged_upload(
    settings: Settings,
    item_id: str,
    group: str = "",
    actor: str = "",
) -> dict[str, object]:
    """Index a staged upload, optionally pinning it to an access group.

    The group has to be written into the policy *before* indexing, because chunk
    metadata bakes in ``document_access_group`` at split time.
    """

    from app.staging import get_staging_store
    from app.uploads import get_upload_store
    from rag.splitter import build_doc_id

    ensure_index_embedding_model(settings.chroma_persist_dir, settings.embedding_model)
    store = get_staging_store()
    item, payload = store.take(item_id)

    doc_ids: list[str] = []
    policy_file = settings.access_policy_file
    if group and policy_file.is_file():
        for filename, _ in payload:
            doc_id = build_doc_id(str(settings.data_dir / filename))
            set_document_groups(policy_file, doc_id, [group])
            doc_ids.append(doc_id)

    job = get_upload_store().create_uploaded(
        payload, settings.data_dir, item["structurize"], actor
    )
    audit.record(
        "staging.approved",
        actor=actor,
        uploader=item["uploader"],
        item_id=item_id,
        files=item["filenames"],
        access_group=group,
        doc_ids=doc_ids,
        job_id=job["id"],
    )
    return store.decide(item_id, "approved", actor, job_id=job["id"])


def reject_staged_upload(
    item_id: str, actor: str = "", reason: str = ""
) -> dict[str, object]:
    """Discard a staged upload without touching the vector store."""

    from app.staging import get_staging_store

    item = get_staging_store().discard(item_id, actor)
    audit.record(
        "staging.rejected",
        actor=actor,
        uploader=item["uploader"],
        item_id=item_id,
        files=item["filenames"],
        reason=reason,
    )
    return item


def _chunk_order(metadata: dict[str, object]) -> int:
    """Chunk ids are "<doc_id>#chunk_007", assigned in load order."""
    chunk_id = str(metadata.get("chunk_id", ""))
    suffix = chunk_id.rsplit("_", 1)[-1] if "_" in chunk_id else ""
    return int(suffix) if suffix.isdigit() else 0


def get_indexed_document(
    settings: Settings,
    doc_id: str,
    user: UserContext | None = None,
    offset: int = 0,
    limit: int = 50,
    focus_chunk_id: str | None = None,
) -> dict[str, object]:
    """Return one indexed document's chunks in index order, paginated.

    ``focus_chunk_id`` recentres the page on a cited chunk so the UI can jump
    straight to it from a citation.
    """

    doc_id = doc_id.strip()
    if not doc_id:
        raise ValueError("doc_id is required")
    access_policy = load_access_policy(settings.access_policy_file)
    if access_policy is not None and user is None:
        raise PermissionError("Authenticated user context is required")

    with knowledge_base_lock.read():
        result = index_management_store(settings).get(
            where={
                "$and": [
                    {"doc_id": doc_id},
                    {"knowledge_base_id": settings.knowledge_base_id},
                ]
            },
            include=["documents", "metadatas"],
        )

    entries: list[tuple[dict[str, object], str]] = []
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    for text, raw_metadata in zip(documents, metadatas):
        metadata = raw_metadata or {}
        if access_policy is not None and not access_policy.can_access(
            metadata, user, settings.knowledge_base_id
        ):
            continue
        entries.append((metadata, text or ""))
    if not entries:
        raise KeyError("Document not found")

    entries.sort(key=lambda entry: _chunk_order(entry[0]))
    if focus_chunk_id:
        position = next(
            (
                index
                for index, (chunk_metadata, _) in enumerate(entries)
                if str(chunk_metadata.get("chunk_id", "")) == focus_chunk_id
            ),
            None,
        )
        if position is not None:
            # Land a couple of chunks early so the target is not the very first card.
            offset = max(0, position - 2)
    page = entries[offset : offset + limit]
    metadata = entries[0][0]
    return {
        "doc_id": doc_id,
        "title": str(metadata.get("title") or doc_id),
        "source": str(metadata.get("source") or ""),
        "chunk_count": len(entries),
        "offset": offset,
        "limit": limit,
        "chunks": [
            {
                "chunk_id": str(chunk_metadata.get("chunk_id", "")),
                "text": text,
                "page": chunk_metadata.get("page"),
                "page_label": str(chunk_metadata.get("page_label", "")),
                "section_path": str(chunk_metadata.get("section_path", "")),
                "start_char": int(chunk_metadata.get("start_char", 0)),
                "end_char": int(chunk_metadata.get("end_char", 0)),
            }
            for chunk_metadata, text in page
        ],
    }


def _ingest_attribution(
    doc_ids: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Map doc_id → who *most recently* ingested that document, and when.

    The audit trail is the only record of this. Chunk metadata carries a
    ``created_at`` key, but the splitter only ever sets it to an empty string,
    so it cannot answer the question.

    Documents indexed before the trail existed have no entry. They must render
    as unknown — never inherit a neighbouring document's actor.

    ``doc_ids`` 传入当前索引里的文档集合时，从日志尾部反向扫、每篇找到最近一次
    入库就提前停，不必把整份日志读完。
    """

    remaining = set(doc_ids) if doc_ids else None

    def stop_when(entry: object) -> bool:
        return remaining is not None and not remaining

    attribution: dict[str, dict[str, str]] = {}
    for entry in audit.iter_entries_newest_first(stop_when):
        if not str(entry.get("event", "")).startswith("ingest."):
            continue
        # 失败的入库尝试也用 ingest.* 事件名；它不能把「最近一次成功入库」覆盖掉。
        # 旧日志没有 status，按 success 兼容。
        if str(entry.get("status") or "success") != "success":
            continue
        for doc_id in entry.get("doc_ids") or []:
            doc_id = str(doc_id)
            # 尾部逆扫 newest first，所以重新入库会覆盖更早的那条归属。
            if doc_id not in attribution:
                attribution[doc_id] = {
                    "actor": str(entry.get("actor") or ""),
                    "timestamp": str(entry.get("timestamp") or ""),
                }
                if remaining is not None:
                    remaining.discard(doc_id)
    return attribution


def list_indexed_documents(
    settings: Settings,
    user: UserContext | None = None,
    admin: bool = False,
) -> list[dict[str, object]]:
    """Aggregate indexed documents from Chroma metadata.

    ``admin`` skips the per-document group filter: this listing doubles as the
    admin's access-management view, and hiding documents you are meant to
    administer would make it unusable. Content access is unchanged — retrieval
    and the document viewer still filter by group.

    读取必须在读锁内：此前这里没加锁，于是文档列表会和后台入库的写入同时访问
    sqlite，撞出 tenant 校验失败。`get_indexed_document` 一直是加锁的。

    最近入库人与时间只回给管理员：它们来自审计日志，而审计页本身就是仅管理员可见。
    切片方式对所有人可见——那是文档自身的属性，不涉及谁做了什么。
    """

    with knowledge_base_lock.read():
        result = index_management_store(settings).get(include=["metadatas"])
    documents: dict[str, dict[str, object]] = {}
    access_policy = load_access_policy(settings.access_policy_file)
    # 管理员视图才需要"最近入库"归属；传入当前文档集合让归属从日志尾部逆扫、
    # 每篇找到最近一次即停，避免整份日志读完。
    current_doc_ids = {
        str(m.get("doc_id", ""))
        for m in (result.get("metadatas") or [])
        if m and m.get("doc_id")
    }
    attribution = _ingest_attribution(current_doc_ids) if admin else {}
    for raw_metadata in result.get("metadatas", []):
        metadata = raw_metadata or {}
        if str(metadata.get("knowledge_base_id") or "default") != settings.knowledge_base_id:
            continue
        if access_policy is not None and not admin:
            if user is None or not access_policy.can_access(metadata, user, settings.knowledge_base_id):
                continue
        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id:
            continue
        ingested = attribution.get(doc_id) or {}
        # structured_by is written per chunk at split time; its presence is what
        # routes a document to heading-aligned chunking in the first place.
        provider = str(metadata.get("structured_by") or "")
        raw_groups = metadata.get("document_access_groups")
        if isinstance(raw_groups, list):
            access_groups = sorted(str(group) for group in raw_groups if group)
        else:
            access_groups = [str(metadata.get("document_access_group") or PUBLIC_GROUP)]
        document = documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": str(metadata.get("title") or "未命名文档"),
                "source": str(metadata.get("source") or ""),
                "access_group": access_groups[0] if access_groups else PUBLIC_GROUP,
                "access_groups": access_groups,
                "chunk_count": 0,
                "chunk_strategy": "heading" if provider else "fixed",
                "chunk_strategy_provider": provider,
                "last_ingested_by": ingested.get("actor", ""),
                "last_ingested_at": ingested.get("timestamp", ""),
            },
        )
        document["chunk_count"] = int(document["chunk_count"]) + 1

    return sorted(
        documents.values(),
        key=lambda document: str(document["title"]).lower(),
    )


class AnswerResult(TypedDict):
    question: str
    retrieval_method: str
    fusion_method: str | None
    vector_context: list[str]
    vector_hits: list[dict[str, object]]
    answer: str


ProgressCallback = Callable[[str, str, dict[str, object]], None]


def _retrieval_params(
    settings: Settings,
    retrieval_method: str | None,
    top_k: int | None,
    similarity_threshold: float | None,
    fusion_method: str | None = None,
    fusion_k: int | None = None,
    fusion_alpha: float | None = None,
) -> tuple[str, int, float, str, int, float]:
    """解析并校验检索参数。多个入口共用，免得默认值逻辑分叉。"""

    method = settings.retrieval_method if retrieval_method is None else retrieval_method
    if method not in ("vector", "bm25", "hybrid"):
        raise ValueError("retrieval_method must be vector, bm25 or hybrid")
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.similarity_threshold
    )
    effective_fusion = (
        settings.fusion_method if fusion_method is None else fusion_method
    )
    if effective_fusion not in ("rrf", "weighted"):
        raise ValueError("fusion_method must be rrf or weighted")
    effective_k = fusion_k if fusion_k is not None else settings.fusion_k
    if effective_k <= 0:
        raise ValueError("fusion_k must be greater than zero")
    effective_alpha = (
        fusion_alpha if fusion_alpha is not None else settings.fusion_alpha
    )
    if not 0.0 <= effective_alpha <= 1.0:
        raise ValueError("fusion_alpha must be between 0 and 1")
    return (
        method,
        effective_top_k,
        effective_threshold,
        effective_fusion,
        effective_k,
        effective_alpha,
    )


def _access_context(
    settings: Settings, user: UserContext | None
) -> tuple[AccessPolicy | None, dict[str, object] | None]:
    """策略与该用户的可检索过滤器。启用策略时没有身份就直接拒绝。"""

    access_policy = load_access_policy(settings.access_policy_file)
    if access_policy is not None and user is None:
        raise PermissionError("Authenticated user context is required")
    access_filter = (
        access_policy.chroma_filter(settings.knowledge_base_id, user)
        if access_policy is not None and user is not None
        else None
    )
    return access_policy, access_filter


def _safe_diagnostic_id(value: object) -> str:
    text = str(value or "")
    return text if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", text) else ""


def _recall_failure(path: str, phase: str, exc: Exception, settings: Settings) -> dict[str, object]:
    from app.rebuild import safe_error

    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None
    headers = getattr(response, "headers", {}) or {}
    provider_request_id = _safe_diagnostic_id(
        getattr(exc, "request_id", "")
        or headers.get("x-request-id", "")
        or headers.get("request-id", "")
        or headers.get("x-siliconcloud-trace-id", "")
    )
    message = safe_error(exc, settings)
    lowered = f"{exc.__class__.__name__} {message}".lower()
    if status_code in (401, 403):
        hint = "检查 Embedding API Key、模型权限和请求地址"
    elif status_code == 429:
        hint = "检查服务额度或限流，稍后重试"
    elif status_code is not None and status_code >= 500:
        hint = "Embedding 服务暂时不可用，稍后重试"
    elif "timeout" in lowered or "timed out" in lowered or "connection" in lowered:
        hint = "检查网络、Embedding 服务地址和超时设置"
    elif "索引由" in message or "embedding model" in lowered:
        hint = "核对索引向量模型；必要时切回原模型或重建索引"
    elif path == "bm25":
        hint = "检查索引目录与 SQLite/Chroma 文件状态，必要时重建索引"
    else:
        hint = "检查索引状态和服务日志后重试"
    return {
        "status": "failed",
        "stage": phase,
        "error_type": exc.__class__.__name__,
        "error_message": message,
        "error_code": _safe_diagnostic_id(getattr(exc, "code", "")),
        "http_status": status_code,
        "provider_request_id": provider_request_id,
        "retryable": bool(status_code in (408, 409, 429) or (status_code or 0) >= 500
                          or "timeout" in lowered or "connection" in lowered),
        "hint": hint,
    }


def _retrieve_locked(
    question: str,
    settings: Settings,
    method: str,
    top_k: int,
    threshold: float,
    candidate_pool: int | None,
    fusion_method: str,
    fusion_k: int,
    fusion_alpha: float,
    user: UserContext | None,
    access_policy: AccessPolicy | None,
    access_filter: dict[str, object] | None,
    report: ProgressCallback,
) -> list[dict[str, object]]:
    """三种检索方式的唯一实现。**假定读锁已持有**，自己不取锁。

    单拆一层是为了让 answer_question 与 retrieve_question 共用同一份逻辑。复制
    一份的话两边语义迟早分叉，而这里有几处细节最容易搬错：

    - `threshold` **只在 vector 分支生效**；hybrid 走 RRF 融合后取 top_k，阈值被忽略
    - bm25 不用 embedding、不用 threshold、不用 pool，但要 user + 策略做过滤
    - hybrid 的 `pool = max(candidate_pool, top_k)`，是向量粗筛的 k

    不要在被持读锁时调用公开入口 `retrieve_question` 来「复用」这段逻辑：
    KnowledgeBaseLock 是写者优先的，嵌套读锁一旦撞上等待中的写者就会自死锁。
    """

    recall: int | None = None
    effective_pool: int | None = None
    bm25_hits: list[dict[str, object]] = []
    fused: list[dict[str, object]] = []
    trace: dict[str, object] | None = None
    vector_ms: int | None = None
    bm25_ms: int | None = None
    vector_candidate_count: int | None = None
    bm25_candidate_count: int | None = None
    recall_detail: dict[str, object] | None = None

    if method == "bm25":
        report("encode", "skipped", retrieval_method=method)
        report(
            "vector_search",
            "running",
            top_k=top_k,
            retrieval_method=method,
        )
        started = perf_counter()
        hits = search_bm25(
            question,
            settings.chroma_persist_dir,
            top_k,
            settings.knowledge_base_id,
            user,
            access_policy,
        )
    elif method == "hybrid":
        effective_pool = max(
            (
                candidate_pool
                if candidate_pool is not None
                else settings.candidate_pool_size
            ),
            top_k,
        )
        report("encode", "running", model=settings.embedding_model)
        report(
            "vector_search",
            "running",
            top_k=top_k,
            retrieval_method=method,
            pool=effective_pool,
            parallel=True,
        )
        recall_id = uuid.uuid4().hex
        started = perf_counter()

        def vector_recall() -> dict[str, object]:
            path_started = perf_counter()
            phase = "embedding_setup"
            encode_detail = None
            query_ms = None
            try:
                embeddings = get_embeddings(
                    settings.embedding_model,
                    settings.embedding_api_key,
                    settings.embedding_base_url,
                )
                vector_store = get_vector_store(
                    settings.chroma_persist_dir,
                    settings.embedding_model,
                    settings.embedding_api_key,
                    settings.embedding_base_url,
                )
                phase = "embedding"
                encode_started = perf_counter()
                query_embedding = embeddings.embed_query(question)
                encode_detail = {
                    "model": settings.embedding_model,
                    "dimensions": len(query_embedding),
                    "duration_ms": round((perf_counter() - encode_started) * 1000),
                    "vector_preview": [round(value, 6) for value in query_embedding[:8]],
                }
                phase = "vector_query"
                query_started = perf_counter()
                if access_filter is None:
                    matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                        query_embedding, k=effective_pool
                    )
                else:
                    matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                        query_embedding, k=effective_pool, filter=access_filter
                    )
                path_hits = build_vector_hits(matches)
                query_ms = round((perf_counter() - query_started) * 1000)
                return {
                    "status": "completed", "hits": path_hits,
                    "elapsed_ms": round((perf_counter() - path_started) * 1000),
                    "vector_ms": query_ms, "encode_detail": encode_detail,
                }
            except Exception as exc:
                return {
                    "status": "failed", "hits": [],
                    "elapsed_ms": round((perf_counter() - path_started) * 1000),
                    "vector_ms": query_ms, "encode_detail": encode_detail,
                    "error": _recall_failure("vector", phase, exc, settings),
                }

        def bm25_recall() -> dict[str, object]:
            path_started = perf_counter()
            try:
                path_hits = search_bm25(
                    question,
                    settings.chroma_persist_dir,
                    effective_pool,
                    settings.knowledge_base_id,
                    user,
                    access_policy,
                )
                return {
                    "status": "completed", "hits": path_hits,
                    "elapsed_ms": round((perf_counter() - path_started) * 1000),
                }
            except Exception as exc:
                return {
                    "status": "failed", "hits": [],
                    "elapsed_ms": round((perf_counter() - path_started) * 1000),
                    "error": _recall_failure("bm25", "bm25_index", exc, settings),
                }

        results: dict[str, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-recall") as executor:
            futures = {
                executor.submit(vector_recall): "vector",
                executor.submit(bm25_recall): "bm25",
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        wall_ms = round((perf_counter() - started) * 1000)
        vector_result = results["vector"]
        bm25_result = results["bm25"]
        vector_hits = list(vector_result.get("hits") or [])
        bm25_hits = list(bm25_result.get("hits") or [])
        vector_ms = vector_result.get("vector_ms")
        bm25_ms = bm25_result["elapsed_ms"]

        encode_detail = vector_result.get("encode_detail")
        if encode_detail:
            report("encode", "completed", **encode_detail)
        else:
            error = vector_result.get("error") or {}
            error_detail = {key: value for key, value in error.items() if key not in {"status", "stage"}}
            if error.get("stage"):
                error_detail["failure_stage"] = error["stage"]
            report("encode", "degraded", model=settings.embedding_model,
                   duration_ms=vector_result["elapsed_ms"], **error_detail)

        succeeded_paths = [path for path in ("vector", "bm25") if results[path]["status"] == "completed"]
        failed_paths = [path for path in ("vector", "bm25") if results[path]["status"] == "failed"]
        recall_status = "failed" if not succeeded_paths else "degraded" if failed_paths else "completed"
        routes = {}
        for path, result in results.items():
            route = {
                "status": result["status"],
                "elapsed_ms": result["elapsed_ms"],
                "count": len(result.get("hits") or []),
            }
            if result.get("error"):
                route["error"] = result["error"]
            routes[path] = route
        recall_detail = {
            "recall_id": recall_id,
            "status": recall_status,
            "parallel": True,
            "wall_ms": wall_ms,
            "succeeded_paths": succeeded_paths,
            "failed_paths": failed_paths,
            "routes": routes,
        }

        vector_candidate_count = len(vector_hits) if vector_result["status"] == "completed" else None
        bm25_candidate_count = len(bm25_hits) if bm25_result["status"] == "completed" else None

        if recall_status == "failed":
            report(
                "vector_search", "failed", top_k=top_k, retrieval_method=method,
                matches=0, pool=effective_pool, duration_ms=wall_ms,
                vector_candidates=None, bm25_candidates=None,
                recall_detail=recall_detail, vector_hits=[],
            )
            failures = "；".join(
                f"{name}：{results[path]['error']['error_message']}"
                for path, name in (("vector", "向量"), ("bm25", "BM25"))
                if results[path]["status"] == "failed"
            )
            logger.warning("Hybrid recall failed: %s", recall_detail)
            raise RuntimeError(f"双路召回均失败；{failures}")
        if recall_status == "degraded":
            logger.warning("Hybrid recall degraded: %s", recall_detail)

        if fusion_method == "weighted":
            fused = weighted_fuse_paths(vector_hits, bm25_hits, alpha=fusion_alpha)
        else:
            fused = rrf_fuse_paths(vector_hits, bm25_hits, k=fusion_k)
        hits = fused[:top_k]
        recall = len(vector_hits)

        final_ranks = {str(hit["chunk_id"]): int(hit["rank"]) for hit in hits}
        trace = {
            "method": "hybrid",
            "parallel": True,
            "recall_status": recall_status,
            "fusion": {
                "method": fusion_method,
                "k": fusion_k if fusion_method == "rrf" else None,
                "alpha": fusion_alpha if fusion_method == "weighted" else None,
            },
            "pool": effective_pool,
            "vector_count": len(vector_hits),
            "bm25_count": len(bm25_hits),
            "union_count": len(fused),
            "final_count": len(hits),
            "candidates": [
                {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "title": item["title"],
                    "page_label": item["page_label"],
                    "section_path": item["section_path"],
                    "vector_rank": item["vector_rank"],
                    "similarity": item.get("similarity"),
                    "bm25_rank": item["bm25_rank"],
                    "bm25_score": item.get("bm25_score"),
                    "sources": item["sources"],
                    "fusion_score": item.get("rrf_score", item.get("weighted_score")),
                    "final_rank": final_ranks.get(str(item["chunk_id"])),
                }
                for item in fused
            ],
        }
    else:
        embeddings = get_embeddings(
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        )
        vector_store = get_vector_store(
            settings.chroma_persist_dir,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        )

        report("encode", "running", model=settings.embedding_model)
        started = perf_counter()
        query_embedding = embeddings.embed_query(question)
        report(
            "encode",
            "completed",
            model=settings.embedding_model,
            dimensions=len(query_embedding),
            duration_ms=round((perf_counter() - started) * 1000),
            vector_preview=[round(value, 6) for value in query_embedding[:8]],
        )

        report("vector_search", "running", top_k=top_k, retrieval_method=method)
        started = perf_counter()
        if access_filter is None:
            matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=top_k
            )
        else:
            matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=top_k, filter=access_filter
            )
        raw_hits = build_vector_hits(matches)
        recall = len(raw_hits)
        hits = [hit for hit in raw_hits if hit["similarity"] >= threshold]

    report(
        "vector_search",
        recall_detail["status"] if recall_detail else "completed",
        top_k=top_k,
        retrieval_method=method,
        matches=len(hits),
        recall=recall,
        duration_ms=round((perf_counter() - started) * 1000),
        similarity_threshold=threshold if method == "vector" else None,
        pool=effective_pool if method == "hybrid" else None,
        fusion_method=fusion_method if method == "hybrid" else None,
        fusion_k=fusion_k if method == "hybrid" else None,
        fusion_alpha=fusion_alpha if method == "hybrid" else None,
        vector_candidates=vector_candidate_count if method == "hybrid" else recall,
        bm25_candidates=(
            bm25_candidate_count if method == "hybrid"
            else len(hits) if method == "bm25"
            else None
        ),
        union_candidates=len(fused) if method == "hybrid" else None,
        vector_ms=vector_ms,
        bm25_ms=bm25_ms,
        recall_detail=recall_detail,
        retrieval_trace=trace,
        vector_hits=hits,
    )
    return hits


def _retrieve_ranked(question, settings, method, top_k, threshold, candidate_pool,
                     fusion_method, fusion_k, fusion_alpha, user, access_policy, access_filter, report):
    """One lock boundary for every caller: recall inside, remote rerank outside."""
    from rag.reranker import rerank_hits

    limit = max(top_k, settings.rerank_pool_size) if settings.rerank_enabled else top_k
    def recall_progress(stage, status, **detail):
        if stage == "vector_search":
            detail["top_k"] = top_k
        report(stage, status, **detail)

    with knowledge_base_lock.read():
        hits = _retrieve_locked(question, settings, method, limit, threshold, candidate_pool,
                                fusion_method, fusion_k, fusion_alpha, user, access_policy,
                                access_filter, recall_progress)
    if settings.rerank_enabled and hits:
        report("rerank", "running", model=settings.rerank_model, input_count=len(hits))
    output, detail = rerank_hits(question, hits, settings, top_k)
    report("rerank", detail["status"], **{key: value for key, value in detail.items() if key != "status"},
           vector_hits=output)
    return output, detail


def retrieve_question(
    question: str,
    settings: Settings,
    retrieval_method: str | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    candidate_pool: int | None = None,
    fusion_method: str | None = None,
    fusion_k: int | None = None,
    fusion_alpha: float | None = None,
    user: UserContext | None = None,
    progress: Callable | None = None,
) -> list[dict[str, object]]:
    """只要检索结果，不生成答案。

    给评估与调试用：同一批问题换不同参数跑，就能看出差异，而不必付一次答案生成
    的钱。返回的 hit 带全部分数与元数据（rank / similarity / bm25_score /
    rrf_score / weighted_score / section_path / chunk_id …）——注意喂给 LLM 的
    `format_vector_hits` 会把分数丢掉，这里不丢。

    通过共享链路取读锁，锁外执行可选重排。启用重排时会调用模型 API。
    **不要在已持锁时调用**，理由见 `_retrieve_locked`。
    """

    question = question.strip()
    if not question:
        raise ValueError("Question is required")
    (
        method,
        effective_top_k,
        effective_threshold,
        effective_fusion,
        effective_k,
        effective_alpha,
    ) = _retrieval_params(
        settings, retrieval_method, top_k, similarity_threshold,
        fusion_method, fusion_k, fusion_alpha,
    )
    access_policy, access_filter = _access_context(settings, user)

    def report(stage, status, **detail):
        if progress is not None:
            progress(stage, status, detail)
    hits, _ = _retrieve_ranked(
            question,
            settings,
            method,
            effective_top_k,
            effective_threshold,
            candidate_pool,
            effective_fusion,
            effective_k,
            effective_alpha,
            user,
            access_policy,
            access_filter,
            report,
        )
    return hits


def resolve_chunks(
    settings: Settings, chunk_ids: list[str]
) -> dict[str, dict[str, object]]:
    """按 `chunk_id` 取回当前的切片（正文 + 定位元数据）。查不到的 id 不在结果里。

    标注时用它打指纹，重新入库之后用它重新锚定标签——两处都要回答「这个 id 现在
    到底是什么」。返回的字段与 `retrieve_question` 的 hit 同形，好让
    `eval_store.fingerprints_for` 直接吃下去。

    调用方据「查不到」判定标签失效，所以这里**不替调用方猜**：没有就是没有。

    不做权限过滤：这是标注/维护动作（网页上仅管理员可达，CLI 本机运行），不是对
    外提供内容的路径。
    """

    wanted = [str(chunk_id) for chunk_id in chunk_ids if str(chunk_id).strip()]
    if not wanted:
        return {}

    with knowledge_base_lock.read():
        result = index_management_store(settings).get(
            where={
                "$and": [
                    {"chunk_id": {"$in": wanted}},
                    {"knowledge_base_id": settings.knowledge_base_id},
                ]
            },
            include=["documents", "metadatas"],
        )

    resolved: dict[str, dict[str, object]] = {}
    for text, raw_metadata in zip(
        result.get("documents") or [], result.get("metadatas") or []
    ):
        metadata = raw_metadata or {}
        chunk_id = str(metadata.get("chunk_id") or "")
        if not chunk_id:
            continue
        resolved[chunk_id] = {
            "chunk_id": chunk_id,
            "text": str(text or ""),
            "doc_id": str(metadata.get("doc_id") or ""),
            "section_path": str(metadata.get("section_path") or ""),
            "page": metadata.get("page"),
        }
    return resolved


def corpus_chunks(settings: Settings) -> tuple[list[tuple[str, str]], list[str]]:
    """把索引里全部切片读成 `[(chunk_id, 正文)]`，外加文档 id 列表。

    给评估快照算语料指纹用：只有比得出「两次跑的语料是不是同一份」，两份快照的
    对比才有意义。
    """

    with knowledge_base_lock.read():
        result = get_vector_store(
            settings.chroma_persist_dir,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        ).get(
            where={"knowledge_base_id": settings.knowledge_base_id},
            include=["documents", "metadatas"],
        )

    chunks: list[tuple[str, str]] = []
    doc_ids: list[str] = []
    for text, raw_metadata in zip(
        result.get("documents") or [], result.get("metadatas") or []
    ):
        metadata = raw_metadata or {}
        chunk_id = str(metadata.get("chunk_id") or "")
        doc_id = str(metadata.get("doc_id") or "")
        if chunk_id:
            chunks.append((chunk_id, str(text or "")))
        if doc_id:
            doc_ids.append(doc_id)
    return chunks, doc_ids


def anchor_eval_labels(
    settings: Settings, questions: list[dict]
) -> list[dict]:
    """把评估标注重新锚定到当前索引上，逐题返回有效与失效的 id。

    每条切片级标注独立走三步：

    1. `chunk_id` 还在、且正文指纹一致 → 直接有效
    2. `chunk_id` 找不到、或指纹对不上（编号还在但内容换了）→ 用
       `(doc_id, 正文指纹)` 在该文档里重新找
    3. 还找不到 → 记进 `stale_chunk_ids`，**调用方必须从指标里剔除**

    第 3 条最要紧：`chunk_id` 是按顺序编的号、`doc_id` 由路径派生，所以重新入库
    或重命名源文件之后，同一个 id 会指向别的正文。把失效标签当成「没召回」会让
    指标凭空变差，而完全看不出是索引变了——这正是评估集最容易骗人的地方。
    """

    from app.eval_store import fingerprint_text

    doc_ids = sorted(
        {
            str((entry.get("fingerprints") or {}).get(chunk_id, {}).get("doc_id") or "")
            for entry in questions
            for chunk_id in (entry.get("relevant_chunk_ids") or [])
        }
        - {""}
    )

    by_doc: dict[str, dict[str, str]] = {}
    if doc_ids:
        with knowledge_base_lock.read():
            result = index_management_store(settings).get(
                where={
                    "$and": [
                        {"doc_id": {"$in": doc_ids}},
                        {"knowledge_base_id": settings.knowledge_base_id},
                    ]
                },
                include=["documents", "metadatas"],
            )
        for text, raw_metadata in zip(
            result.get("documents") or [], result.get("metadatas") or []
        ):
            metadata = raw_metadata or {}
            chunk_id = str(metadata.get("chunk_id") or "")
            doc_id = str(metadata.get("doc_id") or "")
            if chunk_id and doc_id:
                # 指纹 → 当前 chunk_id，用于按内容重新定位
                by_doc.setdefault(doc_id, {})[fingerprint_text(str(text or ""))] = chunk_id

    current = resolve_chunks(
        settings,
        [
            str(chunk_id)
            for entry in questions
            for chunk_id in (entry.get("relevant_chunk_ids") or [])
        ],
    )

    anchored: list[dict] = []
    for entry in questions:
        fingerprints = entry.get("fingerprints") or {}
        resolved: list[str] = []
        stale: list[str] = []
        for raw_chunk_id in entry.get("relevant_chunk_ids") or []:
            chunk_id = str(raw_chunk_id)
            expected = str((fingerprints.get(chunk_id) or {}).get("text_sha1") or "")
            doc_id = str((fingerprints.get(chunk_id) or {}).get("doc_id") or "")

            live = current.get(chunk_id)
            if live is not None and (
                not expected or fingerprint_text(str(live.get("text") or "")) == expected
            ):
                resolved.append(chunk_id)
                continue

            relocated = by_doc.get(doc_id, {}).get(expected) if expected else None
            if relocated:
                resolved.append(relocated)
            else:
                stale.append(chunk_id)

        anchored.append(
            {
                "id": entry.get("id"),
                "question": str(entry.get("question") or ""),
                "relevant_chunk_ids": resolved,
                "relevant_doc_ids": [str(d) for d in (entry.get("relevant_doc_ids") or [])],
                "stale_chunk_ids": stale,
                "source": entry.get("source"),
            }
        )
    return anchored


def answer_question(
    question: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    retrieval_method: str | None = None,
    candidate_pool: int | None = None,
    fusion_method: str | None = None,
    fusion_k: int | None = None,
    fusion_alpha: float | None = None,
    user: UserContext | None = None,
    history: list[dict] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> AnswerResult:
    """Encode, retrieve and answer through the shared CLI/Web RAG pipeline.

    ``history`` is the prior turns of the conversation. When present the question
    is first rewritten into a standalone query — a follow-up like「它的原理是什么」
    retrieves nothing useful on its own — but the *original* question is what the
    answer model sees, alongside the history.
    """

    question = question.strip()
    if not question:
        raise ValueError("Question is required")
    (
        method,
        effective_top_k,
        effective_threshold,
        effective_fusion,
        effective_k,
        effective_alpha,
    ) = _retrieval_params(
        settings, retrieval_method, top_k, similarity_threshold,
        fusion_method, fusion_k, fusion_alpha,
    )
    access_policy, access_filter = _access_context(settings, user)

    def report(stage: str, status: str, **detail: object) -> None:
        if progress is not None:
            progress(stage, status, detail)

    # 系统本身的问题（支持什么格式/当前模型/版本/知识库清单）答案在实时状态里，
    # 走向量检索会答错。识别命中就直接拼答案短路，不占 LLM 信号量、不占读锁。
    from app.system_qa import answer_system_intent, detect_system_intent

    intent = detect_system_intent(question)
    if intent is not None:
        report("system", "running", intent=intent)
        started = perf_counter()
        answer = answer_system_intent(intent, settings, user)
        report(
            "system",
            "completed",
            intent=intent,
            duration_ms=round((perf_counter() - started) * 1000),
        )
        for stage in ("condense", "encode", "vector_search", "rerank", "answer"):
            report(stage, "skipped")
        return {
            "question": question,
            "retrieval_method": "system",
            "fusion_method": None,
            "vector_context": [],
            "vector_hits": [],
            "answer": answer,
        }

    with answer_limiter:
        # 改写要在并发闸之内（它就是用来限制并发 LLM 调用的），但在读锁之外——
        # 占着读锁等一次慢的 LLM 调用会挡住等待中的写者。
        retrieval_question = question
        if history:
            from chains.answer_chain import condense_question

            report("condense", "running", model=settings.llm_model)
            started = perf_counter()
            retrieval_question = condense_question(
                history,
                question,
                model_name=settings.llm_model,
                openai_api_key=settings.openai_api_key,
                openai_base_url=settings.openai_base_url,
            )
            report(
                "condense",
                "completed",
                original=question,
                standalone=retrieval_question,
                duration_ms=round((perf_counter() - started) * 1000),
            )
        else:
            report("condense", "skipped")

        hits, rerank_detail = _retrieve_ranked(
                retrieval_question,
                settings,
                method,
                effective_top_k,
                effective_threshold,
                candidate_pool,
                effective_fusion,
                effective_k,
                effective_alpha,
                user,
                access_policy,
                access_filter,
                report,
            )
        # 检索必须留在锁内（要读向量库），但**生成答案不能**：那是远程 LLM
        # 调用，一次几十秒。读锁是「有写者在等就阻塞新读者」的写者优先实现，
        # 所以占着读锁等 LLM 会让一次入库把全站问答一起堵住——和上面改写那步
        # 同样的道理，这里此前漏了。hits 已经在锁内取出来，释放锁后再用是安全的。
        vector_context = format_vector_hits(hits)

        if rerank_detail["status"] == "completed" and not hits and settings.rerank_score_filter_enabled:
            report("answer", "skipped", reason="score_filter_empty")
            return {"question": question, "retrieval_method": method,
                    "fusion_method": effective_fusion if method == "hybrid" else None,
                    "vector_context": [], "vector_hits": [],
                    "answer": "没有达到重排分数阈值的证据，请降低阈值或关闭分数过滤后重试。"}

        from chains.answer_chain import generate_answer

        report("answer", "running", model=settings.llm_model)
        started = perf_counter()
        answer = generate_answer(
            question=question,
            vector_context=vector_context,
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            history=history,
            on_token=on_token,
        )
        report(
            "answer",
            "completed",
            model=settings.llm_model,
            duration_ms=round((perf_counter() - started) * 1000),
        )

        return {
            "question": question,
            "retrieval_method": method,
            "fusion_method": effective_fusion if method == "hybrid" else None,
            "vector_context": vector_context,
            "vector_hits": hits,
            "answer": answer,
        }
