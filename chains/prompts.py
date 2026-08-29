from langchain_core.prompts import ChatPromptTemplate


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的知识库问答助手。请只根据提供的上下文回答。"
            "如果上下文不足，请明确说明无法确定，并指出还缺少什么信息。"
            "每个关键结论都必须引用 doc_id 或 chunk_id。",
        ),
        (
            "human",
            """
问题：
{question}

Source Chunks：
{vector_context}

请给出中文答案。引用格式优先使用 chunk_id，例如：[source: policy_2024_001#chunk_003]。
""".strip(),
        ),
    ]
)
