def get_local_vector_qa_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个只基于本地向量数据库上下文回答的知识库助手。"
                "只能使用给定 Source Chunks 中的信息。"
                "如果上下文不足，请明确说明无法确定。"
                "每个关键结论都要引用 chunk_id。",
            ),
            (
                "human",
                """
问题：
{question}

Source Chunks：
{context}

请用中文回答，并使用 [source: chunk_id] 标注来源。
""".strip(),
            ),
        ]
    )
