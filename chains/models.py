import uuid
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI


CLIENT_USER_AGENT = "rag-langchain-workspace/1.0"
# OpenCode Go routes by conversation and asks clients to identify themselves, so
# it wants one stable session id rather than a fresh one per request. Other
# OpenAI-compatible gateways ignore both headers, so they are only sent there.
_SESSION_ID = uuid.uuid4().hex


def _default_headers(base_url: str) -> dict[str, str]:
    host = (urlparse(base_url).hostname or "").lower()
    if host == "opencode.ai" or host.endswith(".opencode.ai"):
        return {
            "User-Agent": CLIENT_USER_AGENT,
            "x-opencode-session": _SESSION_ID,
        }
    return {}


def get_chat_model(
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> ChatOpenAI:
    """Create the project's deterministic OpenAI-compatible chat model."""

    kwargs = {"api_key": openai_api_key} if openai_api_key else {}
    if openai_base_url:
        kwargs["base_url"] = openai_base_url
    headers = _default_headers(openai_base_url)
    if headers:
        kwargs["default_headers"] = headers
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        request_timeout=60,
        max_retries=2,
        **kwargs,
    )
