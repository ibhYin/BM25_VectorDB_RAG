from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from app.config import get_settings
from rich.console import Console


app = typer.Typer(help="RAG + LangGraph + Knowledge Graph CLI")
console = Console()


@app.command()
def ingest(
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help="Directory containing raw documents.",
    ),
    build_kg: bool = typer.Option(
        False,
        "--kg",
        help="Extract entities and relationships into Neo4j.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Remove stale Chroma chunks after the new index is written.",
    ),
) -> None:
    from app.service import ingest_knowledge_base

    settings = get_settings()
    try:
        result = ingest_knowledge_base(settings, data_dir, build_kg, rebuild)
    except Exception as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Loaded {result.documents} documents.")
    console.print(f"Ingested {result.chunks} chunks into Chroma.")
    if result.build_kg:
        console.print("Knowledge graph ingestion completed.")


@app.command()
def ask(question: str) -> None:
    from app.service import answer_question

    settings = get_settings()
    try:
        result = answer_question(question, settings)
    except Exception as exc:
        console.print(f"[red]Question failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(result["answer"])


@app.command("import-kg")
def import_kg(
    input_path: Path = typer.Option(
        Path("data/triples/triples.jsonl"),
        "--input",
        help="Canonical JSONL triples produced by scripts/kg/export.py.",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Delete the existing Neo4j graph before importing.",
    ),
    batch_size: int = typer.Option(
        1_000,
        "--batch-size",
        min=1,
        help="Triples per import batch.",
    ),
) -> None:
    from kg.neo4j_store import KnowledgeGraphStore

    try:
        store = KnowledgeGraphStore(get_settings())
        store.setup_schema()
        imported = store.import_triples(
            input_path,
            batch_size=batch_size,
            clear=clear,
        )
    except Exception as exc:
        console.print(f"[red]Graph import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Imported {imported} triples into Neo4j.")


@app.command()
def web(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host for the local web server.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port for the local web server.",
    ),
) -> None:
    from app.web import run_web_server

    console.print(f"Web UI running at http://{host}:{port}")
    run_web_server(host, port)


if __name__ == "__main__":
    app()
