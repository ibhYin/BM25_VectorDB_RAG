from chains.models import CLIENT_USER_AGENT, get_chat_model, _default_headers


def test_opencode_endpoint_gets_user_agent_and_session_headers() -> None:
    headers = _default_headers("https://opencode.ai/zen/go/v1")

    assert headers["User-Agent"] == CLIENT_USER_AGENT
    assert headers["x-opencode-session"]


def test_session_id_is_stable_across_requests() -> None:
    first = _default_headers("https://opencode.ai/zen/go/v1")
    second = _default_headers("https://opencode.ai/zen/go/v1")

    assert first["x-opencode-session"] == second["x-opencode-session"]


def test_other_gateways_get_no_extra_headers() -> None:
    for base_url in (
        "http://192.168.150.199:8090/v1",
        "https://api.example.com/v1",
        "http://127.0.0.1:11434/v1",
        "",
    ):
        assert _default_headers(base_url) == {}


def test_lookalike_host_is_not_treated_as_opencode() -> None:
    assert _default_headers("https://opencode.ai.example.com/v1") == {}


def test_chat_model_sends_headers_only_for_opencode() -> None:
    opencode = get_chat_model("glm-5.3", "sk-test", "https://opencode.ai/zen/go/v1")
    other = get_chat_model("glm-5.3", "sk-test", "https://api.example.com/v1")

    assert opencode.default_headers["x-opencode-session"]
    assert not other.default_headers
