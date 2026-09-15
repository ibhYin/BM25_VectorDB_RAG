"""问答与入库并发时的锁行为。

一次入库不该把全站问答堵住——这个问题只在多人并发时才会暴露，所以单独放一个
文件，跟「公网暴露」那组加固测试分开。
"""

import threading
from http import HTTPStatus

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.service import answer_question, knowledge_base_lock


def test_answer_generation_does_not_hold_the_knowledge_base_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生成答案时读锁必须已释放。

    读锁是写者优先的实现：一旦有入库在等，后续问答会被全部挡住。而
    generate_answer 是远程 LLM 调用（几十秒），占着读锁等它，等于一次入库就让
    全站问答停摆。同一个坑项目在「改写」那步已经绕开了，这里盯住「生成」这步。
    """

    class Embeddings:
        def embed_query(self, question):
            return [1.0, 0.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k):
            return [
                (
                    Document(page_content="内容", metadata={"chunk_id": "a#chunk_001"}),
                    0.1,
                )
            ]

    entering = threading.Event()
    release = threading.Event()

    def blocking_generate_answer(**_kwargs):
        entering.set()  # 此刻锁应当已经放开
        release.wait(timeout=5)
        return "答案"

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr(
        "chains.answer_chain.generate_answer", blocking_generate_answer
    )

    answering = threading.Thread(
        target=lambda: answer_question(
            "问题", Settings(_env_file=None), retrieval_method="vector"
        )
    )
    answering.start()
    assert entering.wait(timeout=10), "answer_question 没有走到生成这步"

    # 判据是行为而不是内部字段：生成答案期间，写者必须能拿到写锁。
    wrote = threading.Event()

    def take_write_lock() -> None:
        with knowledge_base_lock.write():
            wrote.set()

    writer = threading.Thread(target=take_write_lock)
    writer.start()
    blocked = not wrote.wait(timeout=3)

    release.set()
    answering.join(timeout=10)
    writer.join(timeout=10)

    assert not blocked, "生成答案时仍持有读锁——一次入库就能把全站问答堵死"
    assert not answering.is_alive() and not writer.is_alive()
