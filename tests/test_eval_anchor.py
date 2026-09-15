"""标注的重新锚定。

`chunk_id` 是按顺序编的号，`doc_id` 由源文件路径派生。所以重建索引之后，同一个
`chunk_id` 可能已经指向别的正文——标签会**静静失效、指标却照跑**。这里把三种
重新锚定路径都钉住，尤其是「id 还在但内容换了」这种最容易被信任 id 骗过去的情形。
"""

import pytest

from app.config import Settings
from app.eval_store import fingerprint_text
from app.service import anchor_eval_labels


def question_with(chunk_ids: list[str], **overrides) -> dict:
    fingerprints = {
        chunk_id: {
            "text_sha1": fingerprint_text(f"{chunk_id} 的正文"),
            "doc_id": chunk_id.split("#")[0],
        }
        for chunk_id in chunk_ids
    }
    entry = {
        "id": "q1",
        "question": "问题",
        "relevant_chunk_ids": list(chunk_ids),
        "relevant_doc_ids": [],
        "fingerprints": fingerprints,
        "source": "manual",
    }
    entry.update(overrides)
    return entry


class FakeStore:
    """只实现锚定用到的那次按 doc_id 的取数。"""

    def __init__(self, chunks: list[tuple[str, dict]]):
        self._chunks = chunks

    def get(self, where, include):
        return {
            "documents": [text for text, _ in self._chunks],
            "metadatas": [metadata for _, metadata in self._chunks],
        }


@pytest.fixture
def patch_index(monkeypatch):
    """让测试分别控制「按 chunk_id 查」与「按 doc_id 全量查」的结果。"""

    def apply(current: dict[str, dict], doc_chunks: list[tuple[str, dict]]):
        monkeypatch.setattr("app.service.resolve_chunks", lambda settings, ids: current)
        monkeypatch.setattr(
            "app.service.get_vector_store", lambda *args, **kwargs: FakeStore(doc_chunks)
        )

    return apply


def test_an_unchanged_chunk_stays_anchored(patch_index) -> None:
    text = "doc#chunk_001 的正文"
    patch_index(
        {"doc#chunk_001": {"chunk_id": "doc#chunk_001", "text": text}},
        [(text, {"chunk_id": "doc#chunk_001", "doc_id": "doc"})],
    )

    [anchored] = anchor_eval_labels(Settings(_env_file=None), [question_with(["doc#chunk_001"])])

    assert anchored["relevant_chunk_ids"] == ["doc#chunk_001"]
    assert anchored["stale_chunk_ids"] == []


def test_a_renumbered_chunk_is_relocated_by_text(patch_index) -> None:
    """重建索引后编号变了：靠正文指纹把它找回来。"""

    text = "doc#chunk_005 的正文"  # 标签记的编号
    patch_index(
        {},  # doc#chunk_005 已经不存在
        [(text, {"chunk_id": "doc#chunk_003", "doc_id": "doc"})],  # 内容挪到了 003
    )

    [anchored] = anchor_eval_labels(Settings(_env_file=None), [question_with(["doc#chunk_005"])])

    assert anchored["relevant_chunk_ids"] == ["doc#chunk_003"]
    assert anchored["stale_chunk_ids"] == []


def test_a_reused_id_is_not_trusted_when_the_text_changed(patch_index) -> None:
    """最阴的一种：编号还在，但内容已经换成别的正文了。

    这时**不能相信 id**——必须按指纹重新定位，否则指标衡量的是另一段文字。
    """

    new_text = "doc#chunk_001 换成了别的正文"
    old_text = "doc#chunk_001 的正文"  # 标签当初记下的那段
    patch_index(
        {"doc#chunk_001": {"chunk_id": "doc#chunk_001", "text": new_text}},
        [
            (new_text, {"chunk_id": "doc#chunk_001", "doc_id": "doc"}),
            (old_text, {"chunk_id": "doc#chunk_007", "doc_id": "doc"}),
        ],
    )

    [anchored] = anchor_eval_labels(Settings(_env_file=None), [question_with(["doc#chunk_001"])])

    assert anchored["relevant_chunk_ids"] == ["doc#chunk_007"]
    assert anchored["stale_chunk_ids"] == []


def test_a_label_whose_text_is_gone_is_reported_stale(patch_index) -> None:
    """锚不上就报 stale，交给调用方从指标里剔除——绝不能算成「没召回」。"""

    patch_index({}, [("毫不相干的正文", {"chunk_id": "doc#chunk_001", "doc_id": "doc"})])

    [anchored] = anchor_eval_labels(Settings(_env_file=None), [question_with(["doc#chunk_005"])])

    assert anchored["relevant_chunk_ids"] == []
    assert anchored["stale_chunk_ids"] == ["doc#chunk_005"]


def test_unresolvable_ids_do_not_block_the_others(patch_index) -> None:
    """一条失效不该把同一道题的其他标注一起废掉。"""

    good = "doc#chunk_001 的正文"
    patch_index(
        {"doc#chunk_001": {"chunk_id": "doc#chunk_001", "text": good}},
        [(good, {"chunk_id": "doc#chunk_001", "doc_id": "doc"})],
    )

    [anchored] = anchor_eval_labels(
        Settings(_env_file=None), [question_with(["doc#chunk_001", "doc#chunk_099"])]
    )

    assert anchored["relevant_chunk_ids"] == ["doc#chunk_001"]
    assert anchored["stale_chunk_ids"] == ["doc#chunk_099"]


def test_a_label_without_fingerprints_still_works(patch_index) -> None:
    """早期只有 chunk_id、没存指纹的标注不该被当成失效。"""

    patch_index(
        {"doc#chunk_001": {"chunk_id": "doc#chunk_001", "text": "任意正文"}},
        [],
    )
    entry = question_with(["doc#chunk_001"], fingerprints={})

    [anchored] = anchor_eval_labels(Settings(_env_file=None), [entry])

    assert anchored["relevant_chunk_ids"] == ["doc#chunk_001"]
    assert anchored["stale_chunk_ids"] == []


def test_document_level_only_labels_are_left_alone(patch_index) -> None:
    patch_index({}, [])

    [anchored] = anchor_eval_labels(
        Settings(_env_file=None), [question_with([], relevant_doc_ids=["doc"])]
    )

    assert anchored["relevant_chunk_ids"] == []
    assert anchored["stale_chunk_ids"] == []
    assert anchored["relevant_doc_ids"] == ["doc"]
