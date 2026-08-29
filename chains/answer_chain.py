from pathlib import Path
import sys

# 允许直接执行本文件时仍然能从项目根目录导入 chains 包。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chains.prompts import ANSWER_PROMPT
from chains.models import get_chat_model


def generate_answer(
    question: str,
    vector_context: list[str],
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> str:
    """根据向量检索上下文生成最终中文回答。"""

    llm = get_chat_model(model_name, openai_api_key, openai_base_url)

    # 将回答 Prompt 和聊天模型串成 LangChain Runnable。
    chain = ANSWER_PROMPT | llm

    # 把检索到的上下文注入 Prompt；没有上下文时显式传入“无”。
    response = chain.invoke(
        {
            "question": question,
            "vector_context": "\n\n".join(vector_context) or "无",
        }
    )

    # ChatOpenAI 返回 AIMessage，这里只取文本内容给上层调用。
    return response.text
