from pathlib import Path

from langchain_core.documents import Document

from app.access import load_access_policy
from rag.index_manifest import write_index_manifest
from rag.loaders import load_documents
from rag.splitter import split_documents
from rag.vector_store import ingest_documents


def build_vector_index(
    data_dir: Path,
    persist_dir: Path,
    embedding_model: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
    ocr_model: str = "gpt-4o-mini",
    ocr_max_pages: int | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    rebuild: bool = False,
    knowledge_base_id: str = "default",
    access_policy_file: Path | None = None,
) -> tuple[list[Document], list[Document]]:
    """Load, split and index one directory through the shared vector pipeline."""

    # None means unset; an explicitly empty value allows keyless local servers.
    embedding_api_key = openai_api_key if embedding_api_key is None else embedding_api_key
    embedding_base_url = openai_base_url if embedding_base_url is None else embedding_base_url

    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    documents = load_documents(
        data_dir,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ocr_model=ocr_model,
        ocr_max_pages=ocr_max_pages,
    )
    if not documents:
        raise ValueError(f"No supported documents found in: {data_dir}")

    ocr_required = {
        str(document.metadata.get("source", "unknown"))
        for document in documents
        if document.metadata.get("ocr_required")
    }
    ocr_errors = {
        str(document.metadata.get("source", "unknown")): str(
            document.metadata["ocr_error"]
        )
        for document in documents
        if document.metadata.get("ocr_error")
    }
    if ocr_required:
        raise ValueError(
            "OCR is required but OPENAI_API_KEY is missing for: "
            + ", ".join(sorted(ocr_required))
        )
    if ocr_errors:
        details = "; ".join(f"{source}: {error}" for source, error in ocr_errors.items())
        raise RuntimeError(f"OCR failed: {details}")

    access_policy = load_access_policy(access_policy_file) if access_policy_file else None
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        knowledge_base_id=knowledge_base_id,
        access_policy=access_policy,
    )
    if not chunks:
        raise ValueError(f"Documents contained no indexable text: {data_dir}")

    ingest_documents(
        documents=chunks,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        rebuild=rebuild,
        knowledge_base_id=knowledge_base_id,
    )
    write_index_manifest(persist_dir, embedding_model)
    return documents, chunks
