from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from app.config import get_settings
from rich.console import Console
from rich.table import Table


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


def _fmt_number(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "—"


def _score_text(hit: dict) -> str:
    """按检索方式挑该显示的分数。

    三种方式的分数互不可比（向量是距离换算的相似度、BM25 无上界、hybrid 是 RRF
    融合分），所以不合并成一列含义模糊的「分数」。
    """

    method = str(hit.get("retrieval_method") or "")
    if method == "bm25":
        return f"bm25 {_fmt_number(hit.get('bm25_score'))}"
    if method == "hybrid":
        return (
            f"rrf {_fmt_number(hit.get('rrf_score'), 4)}"
            f" · sim {_fmt_number(hit.get('similarity'))}"
        )
    return f"sim {_fmt_number(hit.get('similarity'))}"


def _print_hits(hits: list[dict]) -> None:
    """一条命中打印成一个多行块。

    不用 rich 表格：表格会按终端宽度自动裁列，管道输出时 `chunk_id` 和位置常被
    丢掉——而 `chunk_id` 正是要粘去 `eval label` 的东西，不能被裁。
    """

    for hit in hits:
        console.print(
            f"[dim]#{hit.get('rank')}[/dim]  {_score_text(hit)}  "
            f"[bold]{hit.get('chunk_id') or ''}[/bold]"
        )
        location = [str(hit.get("title") or "")]
        if hit.get("section_path"):
            location.append(str(hit["section_path"]))
        page = hit.get("page_label") or hit.get("page")
        if page:
            location.append(f"p{page}")
        console.print(f"    [dim]{' · '.join(part for part in location if part)}[/dim]")
        snippet = " ".join(str(hit.get("text") or "").split())
        console.print(f"    {snippet[:160]}")


@app.command()
def retrieve(
    question: str,
    method: str | None = typer.Option(
        None, "--method", help="vector / bm25 / hybrid，默认取配置。"
    ),
    top_k: int | None = typer.Option(None, "--top-k"),
    pool: int | None = typer.Option(
        None, "--pool", help="仅 hybrid：向量粗筛的候选池大小。"
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="只对 vector 生效；hybrid 与 bm25 会忽略它。"
    ),
    user_id: str | None = typer.Option(None, "--user-id"),
) -> None:
    """只看检索结果，**不生成答案**。

    调参时用它：同一个问题换 `--method` / `--top-k` 跑两次就能看出差异，而不必付
    一次答案生成的钱。要看一整批就用 `eval run`。
    """

    from app.access import load_access_policy
    from app.service import retrieve_question

    settings = get_settings()
    try:
        policy = load_access_policy(settings.access_policy_file)
        user = policy.user_context(user_id) if policy and user_id else None
        hits = retrieve_question(
            question,
            settings,
            retrieval_method=method,
            top_k=top_k,
            similarity_threshold=threshold,
            candidate_pool=pool,
            user=user,
        )
    except Exception as exc:
        console.print(f"[red]Retrieval failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not hits:
        console.print("[yellow]一条都没召回。[/yellow]")
        return
    console.print(f"[bold]{len(hits)}[/bold] 条命中 · {hits[0]['retrieval_method']}")
    _print_hits(hits)


eval_app = typer.Typer(help="检索评估：对问题集跑召回、看指标。")
app.add_typer(eval_app, name="eval")


@eval_app.command("import-conversations")
def eval_import_conversations(
    owner: str = typer.Option("", "--owner", help="只导入这个身份的会话；默认全部。"),
) -> None:
    """把对话记录里的**首轮**问题导进问题集（标注留空，之后自己标）。

    只取首轮：追问依赖对话上下文，而改写后的独立问题不落盘，原样导入等于拿一个
    断章问题去评估。

    已存在的题目会跳过，**不会覆盖已有标注**——重复导入是安全的。
    """

    from app.conversations import get_conversation_store
    from app.eval_store import question_id, read_questions, upsert_question

    existing = {str(entry.get("id")) for entry in read_questions()}
    imported = 0
    for item in get_conversation_store().first_turn_questions():
        if owner and item["owner"] != owner:
            continue
        if question_id(item["question"]) in existing:
            continue
        upsert_question(item["question"], source="conversation")
        imported += 1

    console.print(
        f"新增 [bold]{imported}[/bold] 道题（已存在的不重复添加、不覆盖标注）。"
        f"当前问题集 {len(read_questions())} 道。"
    )


@eval_app.command("label")
def eval_label(
    question: str,
    relevant: str = typer.Option(
        ..., "--relevant", help="逗号分隔的 chunk_id（切片级标注）。"
    ),
    docs: str = typer.Option(
        "", "--docs", help="逗号分隔的 doc_id（文档级标注，粗一些但省事）。"
    ),
    note: str = typer.Option("", "--note"),
) -> None:
    """写一条标注。追加式，同一问题重复写以最后一次为准。

    先跑 `retrieve "问题"` 看召回，把判相关的 chunk_id 粘过来就行——正文指纹会
    自动从索引里取，不用手算。取不到的 id 会当场报出来，免得拼错了还不知道。
    """

    from app.eval_store import fingerprints_for, upsert_question
    from app.service import resolve_chunks

    chunk_ids = [part.strip() for part in relevant.split(",") if part.strip()]
    doc_ids = [part.strip() for part in docs.split(",") if part.strip()]
    if not chunk_ids and not doc_ids:
        console.print("[red]至少要给 --relevant 或 --docs 之一。[/red]")
        raise typer.Exit(code=2)

    resolved = resolve_chunks(get_settings(), chunk_ids)
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in resolved]
    if missing:
        console.print(
            "[yellow]这些 chunk_id 在索引里找不到，已跳过"
            "（拼错了？还是索引重建过、编号变了？）：[/yellow]"
        )
        for chunk_id in missing:
            console.print(f"  {chunk_id}")

    kept = list(resolved)
    entry = upsert_question(
        question,
        relevant_chunk_ids=kept,
        relevant_doc_ids=doc_ids,
        fingerprints=fingerprints_for(list(resolved.values()), kept),
        note=note,
    )
    console.print(
        f"已记录 [bold]{entry['id']}[/bold]：切片 {len(kept)} 个、文档 {len(doc_ids)} 个。"
    )


@eval_app.command("run")
def eval_run(
    k: str = typer.Option("1,3,5", "--k", help="逗号分隔的 K 列表。"),
    method: str | None = typer.Option(None, "--method"),
    top_k: int | None = typer.Option(None, "--top-k"),
    pool: int | None = typer.Option(None, "--pool"),
    threshold: float | None = typer.Option(None, "--threshold"),
    user_id: str | None = typer.Option(None, "--user-id"),
    show_hits: bool = typer.Option(False, "--show-hits", help="逐题打印召回的切片。"),
) -> None:
    """对问题集跑一遍检索，报告 Hit@K / MRR / Recall@K。

    **未标注的题会被剔除**，不是当成 0 分：「没标」和「一条都没召回对」是两回事。
    同理 `Recall@K` 是**下界**——人工标注必然标不全——所以判断好坏以 `Hit@K` 与
    `MRR` 为准。
    """

    from app.access import load_access_policy
    from app.eval_metrics import hit_at_k, mean_over_labeled, mrr, recall_at_k
    from app.eval_store import read_questions
    from app.service import anchor_eval_labels, retrieve_question

    try:
        ks = sorted({int(part) for part in k.split(",") if part.strip()})
    except ValueError as exc:
        console.print(f"[red]--k 需要逗号分隔的整数，收到：[/red]{k}")
        raise typer.Exit(code=2) from exc
    if not ks or any(value <= 0 for value in ks):
        console.print("[red]--k 只能是正整数。[/red]")
        raise typer.Exit(code=2)

    questions = read_questions()
    if not questions:
        console.print(
            "问题集是空的。先 `eval import-conversations`，或用 `eval label` 加题。"
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    policy = load_access_policy(settings.access_policy_file)
    user = policy.user_context(user_id) if policy and user_id else None

    # 先重新锚定：索引变了之后，同一个 chunk_id 可能已经指向别的正文，靠标签里
    # 存的正文指纹重新定位。锚不上的会被标为 stale 并**从指标里剔除**——绝不当成
    # 「没召回」，否则索引一变指标就凭空变差，还看不出原因。
    anchored = anchor_eval_labels(settings, questions)

    rows: list[dict] = []
    for entry in anchored:
        started = perf_counter()
        try:
            hits = retrieve_question(
                str(entry.get("question") or ""),
                settings,
                retrieval_method=method,
                top_k=top_k,
                similarity_threshold=threshold,
                candidate_pool=pool,
                user=user,
            )
        except Exception as exc:
            console.print(f"[red]检索失败：[/red]{entry.get('question')} — {exc}")
            raise typer.Exit(code=1) from exc
        rows.append(
            {
                "question": str(entry.get("question") or ""),
                "retrieved": [str(hit.get("chunk_id") or "") for hit in hits],
                "relevant": list(entry.get("relevant_chunk_ids") or []),
                "stale": list(entry.get("stale_chunk_ids") or []),
                "elapsed": perf_counter() - started,
                "hits": hits,
            }
        )

    last_k = ks[-1]
    table = Table(box=None, pad_edge=False)
    table.add_column("问题", overflow="fold")
    for value in ks:
        table.add_column(f"Hit@{value}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"Recall@{last_k}", justify="right")
    table.add_column("耗时", justify="right", style="dim")
    for row in rows:
        # 有失效标注的题打个记号：它的数字是基于「剩下的有效标注」算的
        marker = "⚠ " if row["stale"] else ""
        table.add_row(
            marker + row["question"][:44],
            *[
                _fmt_number(hit_at_k(row["retrieved"], row["relevant"], value), 2)
                for value in ks
            ],
            _fmt_number(mrr(row["retrieved"], row["relevant"]), 2),
            _fmt_number(recall_at_k(row["retrieved"], row["relevant"], last_k), 2),
            f"{row['elapsed'] * 1000:.0f}ms",
        )
    console.print(table)

    # 先给召回明细：还没标注时最需要看的正是「到底召回了什么」，不能被下面的
    # 「没标注就没数字」提前 return 掉。
    if show_hits:
        for row in rows:
            console.print(f"\n[bold]{row['question']}[/bold]")
            if row["hits"]:
                _print_hits(row["hits"])
            else:
                console.print("[yellow]  一条都没召回[/yellow]")

    labelled = [row for row in rows if row["relevant"]]
    console.print(
        f"\n已标注 [bold]{len(labelled)}[/bold] / {len(rows)} 道题"
        f"（未标注的不计入指标）。"
    )

    stale_total = sum(len(row["stale"]) for row in rows)
    if stale_total:
        console.print(
            f"[yellow]⚠ {stale_total} 条标注找不到对应切片，已从指标里剔除"
            "（不是算作没召回）。重建索引或源文件改名都会让 chunk 编号指向别的正文——"
            "用 `eval label` 把这几道题重标一下。[/yellow]"
        )
        for row in rows:
            if row["stale"]:
                console.print(f"  [dim]{row['question'][:40]}[/dim] → {', '.join(row['stale'])}")

    if not labelled:
        console.print(
            "[yellow]一道都没标注，没有可报告的数字。[/yellow]"
            '先 `retrieve "问题"` 看召回，再用 `eval label` 标上。'
        )
        return

    summary = Table(box=None, pad_edge=False, show_header=False)
    summary.add_column()
    summary.add_column(justify="right")
    for value in ks:
        summary.add_row(
            f"Hit@{value}",
            _fmt_number(
                mean_over_labeled(
                    hit_at_k(row["retrieved"], row["relevant"], value) for row in labelled
                ),
                3,
            ),
        )
    summary.add_row(
        "MRR",
        _fmt_number(
            mean_over_labeled(mrr(row["retrieved"], row["relevant"]) for row in labelled),
            3,
        ),
    )
    summary.add_row(
        f"Recall@{last_k}（下界）",
        _fmt_number(
            mean_over_labeled(
                recall_at_k(row["retrieved"], row["relevant"], last_k) for row in labelled
            ),
            3,
        ),
    )
    console.print(summary)


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
