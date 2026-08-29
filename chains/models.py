from langchain_openai import ChatOpenAI


def get_chat_model(
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> ChatOpenAI:
    """Create the project's deterministic OpenAI-compatible chat model."""

    kwargs = {"api_key": openai_api_key} if openai_api_key else {}
    if openai_base_url:
        kwargs["base_url"] = openai_base_url
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        request_timeout=60,
        max_retries=2,
        **kwargs,
    )
