from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from app.config import get_settings
from rich.console import Console


app = typer.Typer(help="Local RAG knowledge workspace CLI")
console = Console()


@app.command()
def ingest(
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help="Directory containing raw documents.",
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
        result = ingest_knowledge_base(settings, data_dir, rebuild)
    except Exception as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Loaded {result.documents} documents.")
    console.print(f"Ingested {result.chunks} chunks into Chroma.")


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
