"""`retrieve_question` 与 `answer_question` 的等价性。

重构后 `answer_question` 在锁内只调 `_retrieve_locked`，而 `retrieve_question` 是
它外面包一层读锁的公开版本。评估工具依赖后者，所以「两个入口对同一问题给出逐字段
相同的 hits」必须钉死——尤其是 threshold 只在 vector 分支生效、hybrid 直接忽略它
这两处最容易搬错的语义。

这里跑的是**真 chroma**（向量检索与 BM25 都走真家伙），只把「查询向量」与「答案
生成」打桩，这样测的是真链路而不是假对象之间的约定。
"""

import chromadb
import pytest

from app.config import Settings
from app.service import answer_question, retrieve_question
from rag.vector_store import COLLECTION_NAME


DOCUMENTS = [
    "注意力 机制 Attention",
    "注意力 " + "语言模型 " * 20,
    "数据库 索引 优化",
]
METADATAS = [
    {
        "chunk_id": "doc#chunk_001",
        "doc_id": "doc",
        "title": "T",
        "source": "doc.pdf",
        "page": 1,
        "page_label": "1",
        "section_path": "1 引言",
    },
    {"chunk_id": "doc#chunk_002", "doc_id": "doc", "title": "T", "source": "doc.pdf"},
    {
        "chunk_id": "other#chunk_001",
        "doc_id": "other",
        "title": "O",
        "source": "other.pdf",
    },
]


@pytest.fixture
def indexed(tmp_path):
    chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        COLLECTION_NAME, embedding_function=None
    ).add(
        ids=["a", "b", "c"],
        documents=DOCUMENTS,
        metadatas=METADATAS,
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    return tmp_path


class FakeEmbeddings:
    def embed_query(self, question):
        return [1.0, 0.0]


@pytest.fixture
def stubs(monkeypatch):
    # 两处都要打桩：app.service 用的是自己那份引用，而 rag.vector_store 在构造
    # Chroma 的 embedding_function 时还会再调一次它自己那份。
    for target in ("app.service.get_embeddings", "rag.vector_store.get_embeddings"):
        monkeypatch.setattr(target, lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kwargs: "答案")


def settings_for(persist_dir, **overrides) -> Settings:
    return Settings(_env_file=None, chroma_persist_dir=persist_dir, **overrides)


@pytest.mark.parametrize("method", ["vector", "bm25", "hybrid"])
def test_retrieve_question_matches_answer_question(indexed, stubs, method) -> None:
    settings = settings_for(indexed, retrieval_method=method, retrieval_top_k=3)

    via_answer = answer_question("注意力", settings)["vector_hits"]
    via_retrieve = retrieve_question("注意力", settings)

    assert via_retrieve == via_answer
    # 空结果会让这条等价性测试变成空转
    assert via_retrieve


@pytest.mark.parametrize("method", ["bm25", "hybrid"])
def test_threshold_does_not_touch_bm25_or_hybrid(indexed, stubs, method) -> None:
    """threshold 只在 vector 分支生效。

    hybrid 把向量粗筛的结果交给 RRF 融合后取 top_k，阈值完全不参与；bm25 连
    embedding 都不用。改动检索代码时很容易顺手把阈值也套上去，这条盯住它。
    """

    loose = retrieve_question(
        "注意力", settings_for(indexed, similarity_threshold=0.0), retrieval_method=method
    )
    strict = retrieve_question(
        "注意力", settings_for(indexed, similarity_threshold=0.99), retrieval_method=method
    )

    assert loose == strict


def test_vector_threshold_filters_by_similarity(indexed, stubs) -> None:
    loose = retrieve_question(
        "注意力", settings_for(indexed, similarity_threshold=0.0), retrieval_method="vector"
    )
    strict = retrieve_question(
        "注意力", settings_for(indexed, similarity_threshold=0.99), retrieval_method="vector"
    )

    assert [hit["chunk_id"] for hit in strict] == [
        hit["chunk_id"] for hit in loose if hit["similarity"] >= 0.99
    ]


def test_retrieve_question_returns_scores_unlike_the_llm_context(indexed, stubs) -> None:
    """评估要用分数，而喂给 LLM 的 `format_vector_hits` 会把分数丢掉。"""

    hits = retrieve_question("注意力", settings_for(indexed), retrieval_method="vector")

    assert "similarity" in hits[0] and "distance" in hits[0]
    assert {"chunk_id", "doc_id", "section_path", "text"} <= set(hits[0])


def test_hybrid_hits_are_numbered_by_their_returned_order(indexed, stubs) -> None:
    """hybrid 按融合分重排之后，名次必须跟着重排。

    `build_vector_hits` 给的 `rank` 是**向量候选里的**名次，融合排序之后就对不上了
    ——界面上会出现「第一张卡片写着 #4」。评估页正是靠这个名次判断检索好坏的。
    """

    hits = retrieve_question("注意力", settings_for(indexed), retrieval_method="hybrid")

    assert [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1))
    scores = [hit["rrf_score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_question_rejects_a_blank_question(indexed) -> None:
    with pytest.raises(ValueError, match="Question is required"):
        retrieve_question("   ", settings_for(indexed))


def test_retrieve_question_rejects_an_unknown_method(indexed) -> None:
    with pytest.raises(ValueError, match="retrieval_method must be"):
        retrieve_question("注意力", settings_for(indexed), retrieval_method="semantic")
