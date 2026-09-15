from collections.abc import Callable
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
    set_document_group,
)
from app import audit
from rag.indexer import build_vector_index
from rag.index_manifest import write_index_manifest
from rag.loaders import load_document_file
from rag.retriever import (
    build_vector_hits,
    format_vector_hits,
    rrf_fuse,
    rerank_bm25_scores,
    search_bm25,
)
from rag.splitter import split_documents
from rag.vector_store import (
    delete_documents,
    get_embeddings,
    get_vector_store,
    replace_documents,
    set_document_access_group,
)


class KnowledgeBaseLock:
    """Process-local readers/writer lock for the shared knowledge-base state."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self):
        with self._condition:
            while self._writer or self._waiting_writers:
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


@dataclass(frozen=True)
class FileIngestResult:
    files: int
    documents: int
    chunks: int


def ingest_knowledge_base(
    settings: Settings,
    data_dir: Path | None = None,
    rebuild: bool = False,
    structurize: bool = False,
    actor: str = "cli",
) -> IngestResult:
    """Build the vector index from every supported document in one directory.

    When structurize is True, documents are converted to Markdown via markitdown
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
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            rebuild=rebuild,
            knowledge_base_id=settings.knowledge_base_id,
            access_policy_file=settings.access_policy_file,
            structurize=structurize,
            structurize_provider=settings.structurize_provider,
            mineru_token=settings.mineru_api_token,
            mineru_model_version=settings.mineru_model_version,
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
        structurize_provider=settings.structurize_provider if structurize else "",
        embedding_model=settings.embedding_model,
    )
    return result


def ingest_files(
    settings: Settings,
    paths: list[Path],
    structurize: bool = False,
    actor: str = "",
) -> FileIngestResult:
    """Incrementally index uploaded files without rebuilding the whole directory.

    When structurize is True, files are converted to Markdown via markitdown
    before chunking.
    """

    if not paths:
        raise ValueError("At least one file is required")

    with knowledge_base_lock.write():
        documents: list = []
        for path in paths:
            documents.extend(
                load_document_file(
                    path,
                    openai_api_key=settings.openai_api_key,
                    openai_base_url=settings.openai_base_url,
                    ocr_model=settings.ocr_model,
                    ocr_max_pages=settings.ocr_max_pages,
                    structurize=structurize,
                    structurize_provider=settings.structurize_provider,
                    mineru_token=settings.mineru_api_token,
                    mineru_model_version=settings.mineru_model_version,
                )
            )

        ocr_required = {
            str(document.metadata.get("source", "unknown"))
            for document in documents
            if document.metadata.get("ocr_required")
        }
        if ocr_required:
            raise ValueError(
                "OCR is required for uploaded files: "
                + ", ".join(sorted(ocr_required))
            )
        ocr_errors = {
            str(document.metadata.get("source", "unknown")): str(
                document.metadata["ocr_error"]
            )
            for document in documents
            if document.metadata.get("ocr_error")
        }
        if ocr_errors:
            details = "; ".join(
                f"{source}: {error}" for source, error in ocr_errors.items()
            )
            raise RuntimeError(f"OCR failed: {details}")

        chunks = split_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            knowledge_base_id=settings.knowledge_base_id,
            access_policy=load_access_policy(settings.access_policy_file),
        )
        if not chunks:
            raise ValueError("Uploaded documents contained no indexable text")

        replace_documents(
            documents=chunks,
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            knowledge_base_id=settings.knowledge_base_id,
        )
        write_index_manifest(settings.chroma_persist_dir, settings.embedding_model)

    result = FileIngestResult(len(paths), len(documents), len(chunks))
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
        files=[Path(path).name for path in paths],
        doc_ids=doc_ids,
        documents=result.documents,
        chunks=result.chunks,
        structurize=structurize,
        structurize_provider=settings.structurize_provider if structurize else "",
        embedding_model=settings.embedding_model,
    )
    return result


def delete_indexed_documents(
    settings: Settings, doc_ids: list[str], actor: str = ""
) -> int:
    """Delete every chunk of the given documents from the configured knowledge base."""

    if not any(doc_id.strip() for doc_id in doc_ids):
        raise ValueError("At least one doc_id is required")

    with knowledge_base_lock.write():
        deleted = delete_documents(
            doc_ids,
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            knowledge_base_id=settings.knowledge_base_id,
        )

    audit.record(
        "delete.documents",
        actor=actor,
        knowledge_base_id=settings.knowledge_base_id,
        doc_ids=[doc_id for doc_id in doc_ids if doc_id.strip()],
        deleted_chunks=deleted,
        embedding_model=settings.embedding_model,
    )
    return deleted


def set_document_access(
    settings: Settings,
    doc_id: str,
    group: str,
    actor: str = "",
) -> int:
    """Point one indexed document at an access group. Admin action.

    Updates both halves of the truth: the policy file (what future indexing will
    use) and the stored chunk metadata (what retrieval filters on right now).
    Only metadata is rewritten, so this costs no embedding calls.
    """

    doc_id = doc_id.strip()
    group = group.strip()
    if not doc_id:
        raise ValueError("doc_id is required")
    if not group:
        raise ValueError("group is required")

    policy_path = settings.access_policy_file
    if not policy_path.is_file():
        raise ValueError("未启用访问控制：缺少访问策略文件")
    # 只允许已存在的组，避免拼错把文档变成谁都看不见。
    if group not in known_groups(policy_path):
        raise ValueError(f"未知的访问组：{group}")

    with knowledge_base_lock.write():
        updated = set_document_access_group(
            [doc_id],
            group,
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            knowledge_base_id=settings.knowledge_base_id,
        )
        if not updated:
            raise KeyError("Document not found")
        # 先确认文档存在再落策略，避免给不存在的 doc_id 留下垃圾条目。
        set_document_group(policy_path, doc_id, group)

    audit.record(
        "access.change",
        actor=actor,
        knowledge_base_id=settings.knowledge_base_id,
        doc_ids=[doc_id],
        access_group=group,
        chunks=updated,
    )
    return updated


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

    store = get_staging_store()
    item, payload = store.take(item_id)

    doc_ids: list[str] = []
    policy_file = settings.access_policy_file
    if group and policy_file.is_file():
        for filename, _ in payload:
            doc_id = build_doc_id(str(settings.data_dir / filename))
            set_document_group(policy_file, doc_id, group)
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
        result = get_vector_store(
            settings.chroma_persist_dir,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        ).get(
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


def _ingest_attribution() -> dict[str, dict[str, str]]:
    """Map doc_id → who *most recently* ingested that document, and when.

    The audit trail is the only record of this. Chunk metadata carries a
    ``created_at`` key, but the splitter only ever sets it to an empty string,
    so it cannot answer the question.

    Documents indexed before the trail existed have no entry. They must render
    as unknown — never inherit a neighbouring document's actor.
    """

    attribution: dict[str, dict[str, str]] = {}
    for entry in audit.read_entries(limit=None):
        if not str(entry.get("event", "")).startswith("ingest."):
            continue
        for doc_id in entry.get("doc_ids") or []:
            # read_entries yields newest first, so re-ingesting a document
            # overwrites the earlier attribution rather than keeping it.
            attribution.setdefault(
                str(doc_id),
                {
                    "actor": str(entry.get("actor") or ""),
                    "timestamp": str(entry.get("timestamp") or ""),
                },
            )
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
        result = get_vector_store(
            settings.chroma_persist_dir,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        ).get(include=["metadatas"])
    documents: dict[str, dict[str, object]] = {}
    access_policy = load_access_policy(settings.access_policy_file)
    attribution = _ingest_attribution() if admin else {}
    for raw_metadata in result.get("metadatas", []):
        metadata = raw_metadata or {}
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
        document = documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": str(metadata.get("title") or "未命名文档"),
                "source": str(metadata.get("source") or ""),
                "access_group": str(
                    metadata.get("document_access_group") or PUBLIC_GROUP
                ),
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
    vector_context: list[str]
    vector_hits: list[dict[str, object]]
    answer: str


ProgressCallback = Callable[[str, str, dict[str, object]], None]


def _retrieval_params(
    settings: Settings,
    retrieval_method: str | None,
    top_k: int | None,
    similarity_threshold: float | None,
) -> tuple[str, int, float]:
    """解析并校验检索参数。两个入口共用，免得默认值逻辑分叉。"""

    method = settings.retrieval_method if retrieval_method is None else retrieval_method
    if method not in ("vector", "bm25", "hybrid"):
        raise ValueError("retrieval_method must be vector, bm25 or hybrid")
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.similarity_threshold
    )
    return method, effective_top_k, effective_threshold


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


def _retrieve_locked(
    question: str,
    settings: Settings,
    method: str,
    top_k: int,
    threshold: float,
    candidate_pool: int | None,
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

        report(
            "vector_search",
            "running",
            top_k=top_k,
            retrieval_method=method,
            pool=effective_pool,
        )
        started = perf_counter()
        if access_filter is None:
            matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=effective_pool
            )
        else:
            matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=effective_pool, filter=access_filter
            )
        vector_hits = build_vector_hits(matches)
        bm25_scores = rerank_bm25_scores(
            [match[0].page_content for match in matches],
            question,
        )
        hits = rrf_fuse(vector_hits, bm25_scores)[:top_k]
        recall = len(matches)
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
        "completed",
        top_k=top_k,
        retrieval_method=method,
        matches=len(hits),
        recall=recall,
        duration_ms=round((perf_counter() - started) * 1000),
        similarity_threshold=threshold if method == "vector" else None,
        pool=effective_pool if method == "hybrid" else None,
        vector_hits=hits,
    )
    return hits


def retrieve_question(
    question: str,
    settings: Settings,
    retrieval_method: str | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    candidate_pool: int | None = None,
    user: UserContext | None = None,
) -> list[dict[str, object]]:
    """只要检索结果，不生成答案。

    给评估与调试用：同一批问题换不同参数跑，就能看出差异，而不必付一次答案生成
    的钱。返回的 hit 带全部分数与元数据（rank / similarity / bm25_score /
    rrf_score / section_path / chunk_id …）——注意喂给 LLM 的 `format_vector_hits`
    会把分数丢掉，这里不丢。

    自己取读锁。**不要在已持锁时调用**，理由见 `_retrieve_locked`。
    """

    question = question.strip()
    if not question:
        raise ValueError("Question is required")
    method, effective_top_k, effective_threshold = _retrieval_params(
        settings, retrieval_method, top_k, similarity_threshold
    )
    access_policy, access_filter = _access_context(settings, user)

    with knowledge_base_lock.read():
        return _retrieve_locked(
            question,
            settings,
            method,
            effective_top_k,
            effective_threshold,
            candidate_pool,
            user,
            access_policy,
            access_filter,
            lambda *args, **kwargs: None,
        )


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
        result = get_vector_store(
            settings.chroma_persist_dir,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_base_url,
        ).get(
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
            result = get_vector_store(
                settings.chroma_persist_dir,
                settings.embedding_model,
                settings.embedding_api_key,
                settings.embedding_base_url,
            ).get(
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
    user: UserContext | None = None,
    history: list[dict] | None = None,
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
    method, effective_top_k, effective_threshold = _retrieval_params(
        settings, retrieval_method, top_k, similarity_threshold
    )
    access_policy, access_filter = _access_context(settings, user)

    def report(stage: str, status: str, **detail: object) -> None:
        if progress is not None:
            progress(stage, status, detail)

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

        with knowledge_base_lock.read():
            hits = _retrieve_locked(
                retrieval_question,
                settings,
                method,
                effective_top_k,
                effective_threshold,
                candidate_pool,
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
            "vector_context": vector_context,
            "vector_hits": hits,
            "answer": answer,
        }
