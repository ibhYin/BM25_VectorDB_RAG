from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from threading import BoundedSemaphore, Condition
from typing import TypedDict

from app.config import Settings, get_settings
from rag.indexer import build_vector_index
from rag.loaders import load_document_file
from rag.retriever import build_vector_hits, format_vector_hits
from rag.splitter import split_documents
from rag.vector_store import get_embeddings, get_vector_store, replace_documents


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
) -> IngestResult:
    """Build the vector index from every supported document in one directory."""

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
        )

    return IngestResult(len(documents), len(chunks), rebuild, source_dir)


def ingest_files(settings: Settings, paths: list[Path]) -> FileIngestResult:
    """Incrementally index uploaded files without rebuilding the whole directory."""

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
        )
        if not chunks:
            raise ValueError("Uploaded documents contained no indexable text")

        replace_documents(
            documents=chunks,
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
        )

    return FileIngestResult(len(paths), len(documents), len(chunks))


def list_indexed_documents(settings: Settings) -> list[dict[str, object]]:
    """Aggregate indexed documents from Chroma metadata."""

    result = get_vector_store(
        settings.chroma_persist_dir,
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
    ).get(include=["metadatas"])
    documents: dict[str, dict[str, object]] = {}
    for metadata in result.get("metadatas", []):
        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id:
            continue
        document = documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": str(metadata.get("title") or "未命名文档"),
                "source": str(metadata.get("source") or ""),
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
) -> AnswerResult:
    """Encode, retrieve and answer through the shared CLI/Web RAG pipeline."""

    question = question.strip()
    if not question:
        raise ValueError("Question is required")
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.similarity_threshold
    )

    def report(stage: str, status: str, **detail: object) -> None:
        if progress is not None:
            progress(stage, status, detail)

    with answer_limiter:
        with knowledge_base_lock.read():
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

            report("vector_search", "running", top_k=effective_top_k)
            started = perf_counter()
            matches = vector_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=effective_top_k,
            )
            hits = build_vector_hits(matches)
            hits = [hit for hit in hits if hit["similarity"] >= effective_threshold]
            report(
                "vector_search",
                "completed",
                top_k=effective_top_k,
                matches=len(hits),
                duration_ms=round((perf_counter() - started) * 1000),
                similarity_threshold=effective_threshold,
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
            )
            report(
                "answer",
                "completed",
                model=settings.llm_model,
                duration_ms=round((perf_counter() - started) * 1000),
            )

            return {
                "question": question,
                "vector_context": vector_context,
                "vector_hits": hits,
                "answer": answer,
            }
