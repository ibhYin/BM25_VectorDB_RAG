from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的知识库问答助手。请只根据提供的上下文回答。"
            "如果上下文不足，请明确说明无法确定，并指出还缺少什么信息。"
            "每个关键结论都必须引用 doc_id 或 chunk_id。",
        ),
        # 历史作为真实对话轮次注入，模型才能理解追问里的指代。
        MessagesPlaceholder("history"),
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


# 追问（「它的原理是什么」）直接用原句检索召不回东西，所以先改写成独立问题。
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个检索查询改写助手。根据对话历史，把用户的新问题改写成一个"
            "不依赖上下文、可以独立检索的完整问题。保留原语言和专有名词，不要回答问题本身。"
            "只输出改写后的问题，不要解释、不要加引号或前缀。",
        ),
        (
            "human",
            """
对话历史：
{history}

新问题：{question}

改写后的问题：
""".strip(),
        ),
    ]
)
