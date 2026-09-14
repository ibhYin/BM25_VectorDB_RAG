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
    for raw_metadata in result.get("metadatas", []):
        metadata = raw_metadata or {}
        if access_policy is not None and not admin:
            if user is None or not access_policy.can_access(metadata, user, settings.knowledge_base_id):
                continue
        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id:
            continue
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
    method = settings.retrieval_method if retrieval_method is None else retrieval_method
    if method not in ("vector", "bm25", "hybrid"):
        raise ValueError("retrieval_method must be vector, bm25 or hybrid")
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.similarity_threshold
    )
    access_policy = load_access_policy(settings.access_policy_file)
    if access_policy is not None and user is None:
        raise PermissionError("Authenticated user context is required")
    access_filter = (
        access_policy.chroma_filter(settings.knowledge_base_id, user)
        if access_policy is not None and user is not None
        else None
    )

    def report(stage: str, status: str, **detail: object) -> None:
        if progress is not None:
            progress(stage, status, detail)

    recall = None
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
            if method == "bm25":
                report("encode", "skipped", retrieval_method=method)
                report(
                    "vector_search",
                    "running",
                    top_k=effective_top_k,
                    retrieval_method=method,
                )
                started = perf_counter()
                hits = search_bm25(
                    retrieval_question,
                    settings.chroma_persist_dir,
                    effective_top_k,
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
                    effective_top_k,
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
                query_embedding = embeddings.embed_query(retrieval_question)
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
                    top_k=effective_top_k,
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
                    retrieval_question,
                )
                hits = rrf_fuse(vector_hits, bm25_scores)[:effective_top_k]
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
                query_embedding = embeddings.embed_query(retrieval_question)
                report(
                    "encode",
                    "completed",
                    model=settings.embedding_model,
                    dimensions=len(query_embedding),
                    duration_ms=round((perf_counter() - started) * 1000),
                    vector_preview=[round(value, 6) for value in query_embedding[:8]],
                )

                report("vector_search", "running", top_k=effective_top_k, retrieval_method=method)
                started = perf_counter()
                if access_filter is None:
                    matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                        query_embedding, k=effective_top_k
                    )
                else:
                    matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                        query_embedding, k=effective_top_k, filter=access_filter
                    )
                raw_hits = build_vector_hits(matches)
                recall = len(raw_hits)
                hits = [
                    hit for hit in raw_hits
                    if hit["similarity"] >= effective_threshold
                ]
            report(
                "vector_search",
                "completed",
                top_k=effective_top_k,
                retrieval_method=method,
                matches=len(hits),
                recall=recall,
                duration_ms=round((perf_counter() - started) * 1000),
                similarity_threshold=effective_threshold if method == "vector" else None,
                pool=effective_pool if method == "hybrid" else None,
                vector_hits=hits,
            )
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
