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
    structurize: bool = typer.Option(
        False,
        "--structurize",
        help="Convert documents to Markdown via markitdown (preserves headings/lists/tables).",
    ),
) -> None:
    from app.service import ingest_knowledge_base

    settings = get_settings()
    try:
        result = ingest_knowledge_base(settings, data_dir, rebuild, structurize)
    except Exception as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Loaded {result.documents} documents.")
    console.print(f"Ingested {result.chunks} chunks into Chroma.")


@app.command()
def ask(
    question: str,
    user_id: str | None = typer.Option(None, "--user-id"),
) -> None:
    from app.service import answer_question
    from app.access import load_access_policy

    settings = get_settings()
    try:
        policy = load_access_policy(settings.access_policy_file)
        user = policy.user_context(user_id) if policy and user_id else None
        result = answer_question(question, settings, user=user)
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
    from app.access import load_access_policy
    from app.web import run_web_server

    settings = get_settings()
    # 要求策略却读不到策略，必须当场失败。否则服务会静静地以「全开放」姿态起来：
    # 管理员端点、上传鉴权全都不生效，而且从外面看不出任何异常。
    # 请求期还有一道 503 兜底（策略文件可能在运行中被删），这里是为了让人在
    # 启动日志里一眼看到原因，而不是等别人访问时才发现。
    if settings.require_access_policy and load_access_policy(settings.access_policy_file) is None:
        console.print(
            f"[red]REQUIRE_ACCESS_POLICY 已开启，但访问策略文件不存在或不可读：[/red]"
            f"{settings.access_policy_file}\n"
            "请先创建该文件（可参考 config/access_policy.example.json），"
            "或关闭 REQUIRE_ACCESS_POLICY。"
        )
        raise typer.Exit(code=1)

    console.print(f"Web UI running at http://{host}:{port}")
    run_web_server(host, port)


if __name__ == "__main__":
    app()
