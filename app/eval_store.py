"""评估问题集与人工标注的存储。

只追加的 JSONL，**同一个问题 id 以最后一条记录为准**。路径可用
`RETRIEVAL_EVAL_FILE` 覆盖，默认 `data/retrieval_eval.jsonl`。

为什么是追加式而不是一个可编辑的 JSON 对象：改错标签只需要再追加一条修正，不必
重写整个文件；而且天然留下一份「标签怎么变的」历史——指标突然动了，能回头查是不是
有人改了标注。

**每个相关切片都记一份正文指纹**（`text_sha1`），这是本模块存在的主要理由：
`chunk_id` 是按顺序编号的（`doc_id#chunk_003` 的意思是「第 3 片」，不是「这段
文字」），而 `doc_id` 由源文件路径派生。所以重新入库（改 chunk_size/overlap）或
只是重命名源文件之后，同一个 `chunk_id` 就指向别的正文了——标签会**静静地失效，
指标却照跑**。有指纹才能重新锚定，锚不上就标 stale 剔除，绝不当作「没召回」。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from threading import Lock


DEFAULT_EVAL_LOG = Path("data/retrieval_eval.jsonl")
EVAL_LOG_ENV = "RETRIEVAL_EVAL_FILE"

_append_lock = Lock()
logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def eval_log_path() -> Path:
    return Path(os.environ.get(EVAL_LOG_ENV, str(DEFAULT_EVAL_LOG)))


def _normalise(text: str) -> str:
    """指纹与 id 都要先归一化：换行/空格数量不该影响「这是同一段文字」的判断。"""

    return _WHITESPACE.sub(" ", text or "").strip()


def question_id(question: str) -> str:
    """由问题正文派生的稳定 id。

    这样重复导入同一批会话不会产生重复题目；改问题正文则视为新题（旧标注留在
    日志里，不会被悄悄带走）。
    """

    return hashlib.sha1(_normalise(question).encode("utf-8")).hexdigest()[:12]


def fingerprint_text(text: str) -> str:
    """切片正文的指纹，用于重新锚定。归一化后再哈希，避免只因空白差异失效。"""

    return hashlib.sha1(_normalise(text).encode("utf-8")).hexdigest()[:16]


def fingerprints_for(hits: list[dict], chunk_ids: list[str]) -> dict[str, dict]:
    """从召回结果里给选中的切片生成指纹。

    标注的人不该手算哈希——把召回结果和勾选结果交给它，指纹自动带上。
    """

    wanted = set(chunk_ids)
    fingerprints: dict[str, dict] = {}
    for hit in hits:
        chunk_id = str(hit.get("chunk_id") or "")
        if chunk_id in wanted:
            fingerprints[chunk_id] = {
                "text_sha1": fingerprint_text(str(hit.get("text") or "")),
                "doc_id": str(hit.get("doc_id") or ""),
                "section_path": str(hit.get("section_path") or ""),
                "page": hit.get("page"),
            }
    return fingerprints


def upsert_question(
    question: str,
    relevant_chunk_ids: list[str] | None = None,
    relevant_doc_ids: list[str] | None = None,
    fingerprints: dict[str, dict] | None = None,
    note: str = "",
    source: str = "manual",
) -> dict:
    """写入一道题（追加一条同名 id 的记录，旧的自动失效）。

    写失败只记 warning，绝不抛——和审计日志同样的取舍：丢一条标注很糟，但让
    正在编辑的界面崩掉更糟。
    """

    question = question.strip()
    if not question:
        raise ValueError("Question is required")

    entry = {
        "id": question_id(question),
        "question": question,
        "relevant_chunk_ids": list(relevant_chunk_ids or []),
        "relevant_doc_ids": list(relevant_doc_ids or []),
        "fingerprints": dict(fingerprints or {}),
        "note": note,
        "source": source,
        "created_at": datetime.now().astimezone().isoformat(),
    }

    try:
        path = eval_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _append_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        logger.warning("Could not write eval entry %s: %s", entry["id"], exc)

    return entry


def read_questions() -> list[dict]:
    """当前生效的问题集，按写入顺序返回（同 id 取最后一条）。

    跳过空行与解析失败的行：崩溃留下的半行不该让整份标注读不出来。
    """

    path = eval_log_path()
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    latest: dict[str, dict] = {}
    order: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        entry_id = str(payload.get("id") or "")
        if not entry_id:
            continue
        if entry_id not in latest:
            order.append(entry_id)
        latest[entry_id] = payload

    return [latest[entry_id] for entry_id in order]


def import_first_turn_questions(owner: str = "") -> int:
    """把对话记录里的**首轮**问题导进问题集，返回新增数量。

    只取首轮：追问依赖对话上下文，而改写后的独立问题不落盘，原样导入等于拿一个
    断章问题去评估。

    **幂等**：已存在的题跳过。这条不只是省事——`upsert_question` 会追加一条新记录
    并在读时覆盖旧记录，不判重的话第二次导入就把辛苦标的题全清空了，而且从输出上
    看不出异常。
    """

    from app.conversations import get_conversation_store

    existing = {str(entry.get("id")) for entry in read_questions()}
    imported = 0
    for item in get_conversation_store().first_turn_questions():
        if owner and item["owner"] != owner:
            continue
        if question_id(item["question"]) in existing:
            continue
        upsert_question(item["question"], source="conversation")
        imported += 1
    return imported
