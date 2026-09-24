from pathlib import Path
import sys
from collections.abc import Callable

# 允许直接执行本文件时仍然能从项目根目录导入 chains 包。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage

from chains.prompts import ANSWER_PROMPT, CONDENSE_PROMPT
from chains.models import get_chat_model


# 历史窗口本身已由调用方限制轮数，这里再限制每轮长度，避免长答案把 prompt 撑爆。
MAX_ANSWER_CHARS_IN_HISTORY = 2000
MAX_ANSWER_CHARS_IN_CONDENSE = 500


def _history_messages(history: list[dict] | None) -> list:
    """Prior turns as real chat messages, oldest first."""

    messages: list = []
    for turn in history or []:
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append(HumanMessage(content=question))
        if answer:
            messages.append(AIMessage(content=answer[:MAX_ANSWER_CHARS_IN_HISTORY]))
    return messages


def _history_text(history: list[dict] | None) -> str:
    """Prior turns as plain text, for the condense prompt."""

    lines: list[str] = []
    for turn in history or []:
        question = str(turn.get("question") or "").strip()
        if not question:
            continue
        answer = str(turn.get("answer") or "").strip()
        lines.append(f"用户：{question}")
        if answer:
            lines.append(f"助手：{answer[:MAX_ANSWER_CHARS_IN_CONDENSE]}")
    return "\n".join(lines) or "（无）"


def condense_question(
    history: list[dict] | None,
    question: str,
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> str:
    """Rewrite a follow-up into a standalone retrieval query.

    Falls back to the original question when the model returns nothing, so a
    flaky rewrite degrades to today's behaviour instead of breaking retrieval.
    """

    llm = get_chat_model(model_name, openai_api_key, openai_base_url)
    chain = CONDENSE_PROMPT | llm
    response = chain.invoke(
        {"history": _history_text(history), "question": question}
    )
    condensed = (response.text or "").strip()
    return condensed or question


def generate_answer(
    question: str,
    vector_context: list[str],
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    history: list[dict] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """根据向量检索上下文生成最终中文回答。

    ``on_token`` 提供时走流式生成：每个内容块会回调一次；返回仍是完整文本，
    与旧行为一致（不传 on_token 时等价于一次 invoke）。
    """

    llm = get_chat_model(model_name, openai_api_key, openai_base_url)

    # 将回答 Prompt 和聊天模型串成 LangChain Runnable。
    chain = ANSWER_PROMPT | llm

    # 把检索到的上下文和历史注入 Prompt；没有上下文时显式传入“无”。
    inputs = {
        "question": question,
        "vector_context": "\n\n".join(vector_context) or "无",
        "history": _history_messages(history),
    }

    if on_token is None:
        return chain.invoke(inputs).text

    parts: list[str] = []
    for chunk in chain.stream(inputs):
        text = getattr(chunk, "content", "") or ""
        if not text:
            continue
        parts.append(text)
        on_token(text)
    return "".join(parts)
