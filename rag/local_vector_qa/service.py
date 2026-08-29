from pathlib import Path

from langchain_core.documents import Document

from rag.constants import DEFAULT_OCR_MODEL
from rag.indexer import build_vector_index
from rag.retriever import format_documents
from rag.vector_store import get_vector_store


def build_local_vector_index(
    data_dir: Path,
    persist_dir: Path,
    embedding_model: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
    ocr_model: str = DEFAULT_OCR_MODEL,
    ocr_max_pages: int | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    rebuild: bool = False,
) -> tuple[int, int]:
    documents, chunks = build_vector_index(
        data_dir=data_dir,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        ocr_model=ocr_model,
        ocr_max_pages=ocr_max_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rebuild=rebuild,
    )
    return len(documents), len(chunks)


def retrieve_local_chunks(
    question: str,
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    top_k: int = 4,
) -> list[Document]:
    question = question.strip()
    if not question:
        raise ValueError("Question is required")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    vector_store = get_vector_store(
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
    return vector_store.similarity_search(question, k=top_k)


def answer_with_local_vector_store(
    question: str,
    persist_dir: Path,
    embedding_model: str,
    llm_model: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    top_k: int = 4,
) -> dict[str, object]:
    chunks = retrieve_local_chunks(
        question=question,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        top_k=top_k,
    )
    context = "\n\n".join(format_documents(chunks)) or "无"

    from chains.models import get_chat_model
    from rag.local_vector_qa.prompts import get_local_vector_qa_prompt

    llm = get_chat_model(llm_model, openai_api_key, openai_base_url)
    response = (get_local_vector_qa_prompt() | llm).invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "answer": response.text,
        "chunks": chunks,
        "context": context,
    }
