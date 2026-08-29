from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer
from rich.console import Console

from app.config import get_settings


app = typer.Typer(help="Local vector-only RAG CLI")
console = Console()


@app.command()
def ingest(
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help="Directory containing documents.",
    ),
    persist_dir: Path | None = typer.Option(
        None,
        "--persist-dir",
        help="Chroma persist directory.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Remove stale Chroma chunks after the new index is written.",
    ),
) -> None:
    settings = get_settings()
    source_dir = data_dir or settings.data_dir
    chroma_dir = persist_dir or settings.chroma_persist_dir

    from rag.local_vector_qa.service import build_local_vector_index

    document_count, chunk_count = build_local_vector_index(
        data_dir=source_dir,
        persist_dir=chroma_dir,
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

    console.print(f"Loaded {document_count} documents.")
    console.print(f"Ingested {chunk_count} chunks into {chroma_dir}.")


@app.command()
def retrieve(
    question: str,
    top_k: int = typer.Option(
        4,
        "--top-k",
        min=1,
        help="Number of chunks to retrieve.",
    ),
    persist_dir: Path | None = typer.Option(
        None,
        "--persist-dir",
        help="Chroma persist directory.",
    ),
) -> None:
    settings = get_settings()
    chroma_dir = persist_dir or settings.chroma_persist_dir

    from rag.local_vector_qa.service import retrieve_local_chunks
    from rag.retriever import format_documents

    chunks = retrieve_local_chunks(
        question=question,
        persist_dir=chroma_dir,
        embedding_model=settings.embedding_model,
        embedding_api_key=settings.embedding_api_key,
        embedding_base_url=settings.embedding_base_url,
        top_k=top_k,
    )
    console.print("\n\n".join(format_documents(chunks)) or "No chunks found.")


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(
        4,
        "--top-k",
        min=1,
        help="Number of chunks to retrieve.",
    ),
    persist_dir: Path | None = typer.Option(
        None,
        "--persist-dir",
        help="Chroma persist directory.",
    ),
) -> None:
    settings = get_settings()
    chroma_dir = persist_dir or settings.chroma_persist_dir

    from rag.local_vector_qa.service import answer_with_local_vector_store

    result = answer_with_local_vector_store(
        question=question,
        persist_dir=chroma_dir,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        embedding_api_key=settings.embedding_api_key,
        embedding_base_url=settings.embedding_base_url,
        top_k=top_k,
    )
    console.print(result["answer"])


@app.command()
def status(
    persist_dir: Path | None = typer.Option(
        None,
        "--persist-dir",
        help="Chroma persist directory.",
    ),
) -> None:
    settings = get_settings()
    chroma_dir = persist_dir or settings.chroma_persist_dir
    sqlite_path = chroma_dir / "chroma.sqlite3"

    console.print(f"data_dir: {settings.data_dir}")
    console.print(f"persist_dir: {chroma_dir}")
    console.print(f"embedding_model: {settings.embedding_model}")
    console.print(f"embedding_base_url: {settings.embedding_base_url or settings.openai_base_url}")
    console.print(f"llm_model: {settings.llm_model}")
    console.print(f"chroma_exists: {sqlite_path.exists()}")


if __name__ == "__main__":
    app()
