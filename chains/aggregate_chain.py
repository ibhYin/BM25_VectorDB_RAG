from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from chains.models import get_chat_model


# 条款关键情形通常出现在片段前部；再长的部分只是增加 token 成本。
MAX_CHUNK_CHARS = 1200
MAX_CATEGORY_CHARS = 80
MAX_EVIDENCE_CHARS = 200


AGGREGATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个文档信息抽取器。只输出一个 JSON 对象，不要输出解释、Markdown 或代码围栏。"
            "你的任务是从每个片段中判断是否包含与用户主题相关的具体情形，并抽取成简短类别。",
        ),
        (
            "human",
            """
聚合主题：{question}

片段列表：
{chunks}

请对上面每一个片段输出一条记录，严格按以下 JSON 结构：
{{"items": [{{"chunk_id": "...", "relevant": true, "category": "...", "evidence": "..."}}]}}

规则：
1. relevant 只能是 true 或 false。
2. 只有片段确实描述了与主题相关的具体情形时才为 true；不相关、泛泛而谈或无法判断时为 false。
3. category 是不超过 20 字的中文名词短语，概括该片段中的情形；不同表述应尽量收敛到同一类说法。
4. evidence 必须从片段原文逐字摘录，不超过 80 字；不得改写或编造。
5. 输出必须包含所有片段的记录，顺序与输入一致。

{reminder}
""".strip(),
        ),
    ]
)


def _message_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    # 部分 OpenAI 兼容网关把 content 返回成分段列表。
    return "".join(
        str(part.get("text", "")) if isinstance(part, dict) else str(part)
        for part in content or []
    )


def parse_extraction_json(text: str) -> list[dict[str, Any]]:
    """把模型回复解析成规范化条目；格式不合法时返回空列表。"""

    if not isinstance(text, str) or not text.strip():
        return []
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        relevant = item.get("relevant")
        if isinstance(relevant, str):
            relevant = relevant.strip().lower() in {"true", "yes", "1"}
        if not isinstance(relevant, bool):
            relevant = False
        category = str(item.get("category", "")).strip()
        if relevant and not category:
            # 没有类别的相关片段无法参与计数，直接丢弃。
            continue
        normalized.append(
            {
                "chunk_id": chunk_id,
                "relevant": relevant,
                "category": category[:MAX_CATEGORY_CHARS],
                "evidence": str(item.get("evidence", "")).strip()[:MAX_EVIDENCE_CHARS],
            }
        )
    return normalized


def extract_categories(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> list[dict[str, Any]]:
    """对一批检索片段做一次结构化抽取。"""

    payload_chunks = [
        (
            str(hit.get("chunk_id") or ""),
            str(hit.get("text") or "")[:MAX_CHUNK_CHARS],
        )
        for hit in chunks
        if str(hit.get("chunk_id") or "").strip() and str(hit.get("text") or "").strip()
    ]
    if not payload_chunks:
        return []

    chunk_text = "\n\n".join(
        f"[chunk_id={chunk_id}]\n{text}" for chunk_id, text in payload_chunks
    )
    llm = get_chat_model(model_name, openai_api_key, openai_base_url)
    chain = AGGREGATE_PROMPT | llm

    inputs = {"question": question, "chunks": chunk_text, "reminder": ""}
    items = parse_extraction_json(_message_text(chain.invoke(inputs)))
    if items:
        return items

    # 个别网关会把 JSON 包在解释文字里；再要一次，失败就交给调用方降级。
    retry_inputs = {
        **inputs,
        "reminder": "再次强调：只输出 JSON 对象，不要任何解释或代码围栏。",
    }
    return parse_extraction_json(_message_text(chain.invoke(retry_inputs)))
