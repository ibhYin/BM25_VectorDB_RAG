from pathlib import Path

from langchain_core.documents import Document

from app.access import load_access_policy
from rag.index_manifest import ensure_index_embedding_model, write_index_manifest
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
    structurize: bool = False,
    structurize_provider: str = "mineru",
    mineru_token: str = "",
    mineru_model_version: str = "vlm",
    audio_transcribe_base_url: str = "",
    audio_transcribe_api_key: str = "",
    audio_transcribe_model: str = "",
    audio_transcribe_provider: str = "openai",
    aliyun_access_key_id: str = "",
    aliyun_access_key_secret: str = "",
    aliyun_appkey: str = "",
    aliyun_region: str = "cn-shanghai",
    aliyun_oss_bucket: str = "",
    aliyun_oss_endpoint: str = "",
    aliyun_oss_prefix: str = "",
    embedding_max_input_bytes: int | None = None,
    embedding_batch_size: int = 10,
) -> tuple[list[Document], list[Document]]:
    """Load, split and index one directory through the shared vector pipeline.

    When structurize is True, documents are converted to Markdown via MinerU
    before chunking, preserving headings, lists, and tables.
    """

    # None means unset; an explicitly empty value allows keyless local servers.
    embedding_api_key = openai_api_key if embedding_api_key is None else embedding_api_key
    embedding_base_url = openai_base_url if embedding_base_url is None else embedding_base_url

    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    ensure_index_embedding_model(persist_dir, embedding_model)
    documents = load_documents(
        data_dir,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ocr_model=ocr_model,
        ocr_max_pages=ocr_max_pages,
        structurize=structurize,
        structurize_provider=structurize_provider,
        mineru_token=mineru_token,
        mineru_model_version=mineru_model_version,
        audio_transcribe_base_url=audio_transcribe_base_url,
        audio_transcribe_api_key=audio_transcribe_api_key,
        audio_transcribe_model=audio_transcribe_model,
        audio_transcribe_provider=audio_transcribe_provider,
        aliyun_access_key_id=aliyun_access_key_id,
        aliyun_access_key_secret=aliyun_access_key_secret,
        aliyun_appkey=aliyun_appkey,
        aliyun_region=aliyun_region,
        aliyun_oss_bucket=aliyun_oss_bucket,
        aliyun_oss_endpoint=aliyun_oss_endpoint,
        aliyun_oss_prefix=aliyun_oss_prefix,
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
        max_input_bytes=embedding_max_input_bytes,
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
        embedding_batch_size=embedding_batch_size,
    )
    from rag.markdown_cache import write_documents

    write_documents(persist_dir, documents)
    write_index_manifest(persist_dir, embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return documents, chunks
