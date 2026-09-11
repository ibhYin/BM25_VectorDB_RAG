import pytest

from app import model_catalog


def test_loopback_hosts_are_local() -> None:
    for base_url in (
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434/v1",
        "http://127.0.0.5:8000/v1",
        "http://ollama.local:11434/v1",
    ):
        assert model_catalog.is_local_endpoint(base_url) is True


def test_remote_hosts_are_not_local() -> None:
    for base_url in (
        "http://192.168.150.199:8090/v1",
        "https://api.example.com/v1",
        "",
        "not-a-url",
    ):
        assert model_catalog.is_local_endpoint(base_url) is False


def test_local_endpoint_uses_ollama_tags_and_reports_sizes(monkeypatch) -> None:
    requested: list[str] = []

    def fake_get_json(url: str, api_key: str = ""):
        requested.append(url)
        return {
            "models": [
                {
                    "name": "bge-small-zh:q8_0",
                    "size": 26472789,
                    "details": {"parameter_size": "23.69M"},
                },
                {"name": "bge-m3:q8_0", "size": 634553910, "details": {}},
            ]
        }

    monkeypatch.setattr(model_catalog, "_get_json", fake_get_json)

    result = model_catalog.list_models("http://127.0.0.1:11434/v1", "ollama")

    assert requested == ["http://127.0.0.1:11434/api/tags"]
    assert result["scope"] == "local"
    assert result["error"] == ""
    assert [model["id"] for model in result["models"]] == [
        "bge-small-zh:q8_0",
        "bge-m3:q8_0",
    ]
    assert result["models"][0]["size_label"] == "25.2 MB"
    assert result["models"][1]["size_label"] == "605.2 MB"
    assert result["models"][0]["parameter_size"] == "23.69M"


def test_local_endpoint_falls_back_to_openai_listing(monkeypatch) -> None:
    requested: list[str] = []

    def fake_get_json(url: str, api_key: str = ""):
        requested.append(url)
        if url.endswith("/api/tags"):
            raise ValueError("not an ollama server")
        return {"data": [{"id": "llama3.2"}]}

    monkeypatch.setattr(model_catalog, "_get_json", fake_get_json)

    result = model_catalog.list_models("http://127.0.0.1:11434/v1")

    assert requested == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/v1/models",
    ]
    assert result["scope"] == "local"
    assert [model["id"] for model in result["models"]] == ["llama3.2"]


def test_remote_endpoint_lists_openai_models(monkeypatch) -> None:
    def fake_get_json(url: str, api_key: str = ""):
        assert url == "https://gateway.example.test/v1/models"
        assert api_key == "sk-test"
        return {"data": [{"id": "glm-5.3"}, {"id": "gpt-4o"}]}

    monkeypatch.setattr(model_catalog, "_get_json", fake_get_json)

    result = model_catalog.list_models("https://gateway.example.test/v1", "sk-test")

    assert result["scope"] == "cloud"
    assert result["error"] == ""
    assert [model["id"] for model in result["models"]] == ["glm-5.3", "gpt-4o"]
    assert result["models"][0]["size_label"] == ""


def test_unreachable_endpoint_reports_error(monkeypatch) -> None:
    def fake_get_json(url: str, api_key: str = ""):
        raise OSError("connection refused")

    monkeypatch.setattr(model_catalog, "_get_json", fake_get_json)

    result = model_catalog.list_models("https://gateway.example.test/v1")

    assert result["models"] == []
    assert result["error"]


def test_missing_or_invalid_base_url_is_reported_without_network() -> None:
    assert model_catalog.list_models("") == {
        "scope": "cloud",
        "models": [],
        "error": "未配置请求地址",
    }

    result = model_catalog.list_models("ftp://gateway.example.test")

    assert result["models"] == []
    assert "http://" in str(result["error"])


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, ""),
        (None, ""),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (26472789, "25.2 MB"),
    ],
)
def test_human_size(size, expected) -> None:
    assert model_catalog._human_size(size) == expected


def test_get_json_identifies_the_client(monkeypatch) -> None:
    from chains.models import CLIENT_USER_AGENT

    captured: dict[str, dict[str, str]] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.headers)
        return FakeResponse()

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    model_catalog._get_json("https://gateway.example.test/v1/models", "sk-test")

    user_agent = next(
        value
        for key, value in captured["headers"].items()
        if key.lower() == "user-agent"
    )
    assert user_agent == CLIENT_USER_AGENT
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
