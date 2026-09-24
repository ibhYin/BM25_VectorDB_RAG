from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from chains import answer_chain
from chains.answer_chain import generate_answer


def _fake_model(monkeypatch, chunks):
    def gen(inputs):
        for piece in chunks:
            yield AIMessageChunk(content=piece)

    monkeypatch.setattr(
        answer_chain,
        "get_chat_model",
        lambda *args, **kwargs: RunnableGenerator(gen),
    )


def test_generate_answer_streams_tokens(monkeypatch) -> None:
    received: list[str] = []
    _fake_model(monkeypatch, ["你好", "，", "世界"])

    text = generate_answer("问题", [], "model", on_token=received.append)

    assert text == "你好，世界"
    assert received == ["你好", "，", "世界"]


def test_generate_answer_without_on_token_returns_full_text(monkeypatch) -> None:
    _fake_model(monkeypatch, ["你好", "世界"])

    text = generate_answer("问题", [], "model")

    assert text == "你好世界"


def test_generate_answer_skips_empty_chunks(monkeypatch) -> None:
    received: list[str] = []
    _fake_model(monkeypatch, ["你好", "", "世界"])

    text = generate_answer("问题", [], "model", on_token=received.append)

    assert text == "你好世界"
    assert received == ["你好", "世界"]
