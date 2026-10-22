from pathlib import Path

import pytest
from langchain_core.documents import Document

import rag.vector_store as vector_store_module
from rag.vector_store import (
    delete_documents,
    ingest_documents,
    replace_documents,
    set_document_access_group,
)


class FakeCollection:
    def __init__(self) -> None:
        self.updates: list[tuple[list[str], list[dict]]] = []

    def update(self, ids, metadatas) -> None:
        self.updates.append((list(ids), list(metadatas)))


class FakeStore:
    def __init__(self) -> None:
        self.added_batches: list[tuple[int, int]] = []
        self.deleted_batches: list[int] = []
        self.existing_ids: list[str] = []
        self.existing_metadatas: list[dict] = []
        self._collection = FakeCollection()

    def get(self, **kwargs):
        return {
            "ids": list(self.existing_ids),
            "metadatas": list(self.existing_metadatas),
        }

    def add_documents(self, documents, ids=None) -> None:
        self.added_batches.append((len(documents), len(ids)))

    def delete(self, ids=None) -> None:
        self.deleted_batches.append(len(ids))


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(
        vector_store_module, "get_vector_store", lambda *args, **kwargs: fake
    )
    monkeypatch.setattr(vector_store_module, "CHROMA_WRITE_BATCH_SIZE", 2)
    return fake


def _documents(count: int, prefix: str = "new") -> list[Document]:
    return [
        Document(
            page_content=f"text {index}",
            metadata={"doc_id": prefix, "chunk_id": f"{prefix}#{index}"},
        )
        for index in range(count)
    ]


def test_replace_documents_batches_adds(store) -> None:
    replace_documents(_documents(5), Path("/tmp/chroma"), "model")

    assert store.added_batches == [(2, 2), (2, 2), (1, 1)]


def test_replace_documents_batches_stale_deletes(store) -> None:
    store.existing_ids = [f"old#{index}" for index in range(5)]

    replace_documents(_documents(1), Path("/tmp/chroma"), "model")

    assert store.deleted_batches == [2, 2, 1]


def test_ingest_documents_batches_writes_and_deletes(store) -> None:
    store.existing_ids = [f"old#{index}" for index in range(3)]

    ingest_documents(
        _documents(5),
        Path("/tmp/chroma"),
        "model",
        rebuild=True,
    )

    assert store.added_batches == [(2, 2), (2, 2), (1, 1)]
    assert store.deleted_batches == [2, 1]


def test_delete_documents_batches_deletes(store) -> None:
    store.existing_ids = [f"doc#{index}" for index in range(5)]

    removed = delete_documents(["doc"], Path("/tmp/chroma"), "model")

    assert removed == 5
    assert store.deleted_batches == [2, 2, 1]


def test_set_document_access_group_batches_updates(store) -> None:
    store.existing_ids = [f"doc#{index}" for index in range(5)]
    store.existing_metadatas = [{"chunk_id": f"doc#{index}"} for index in range(5)]

    updated = set_document_access_group(
        ["doc"],
        ["hr"],
        Path("/tmp/chroma"),
        "model",
        all_groups=["hr", "finance"],
    )

    assert updated == 5
    sizes = [len(ids) for ids, _ in store._collection.updates]
    assert sizes == [2, 2, 1]
    assert all(
        len(metadatas) == len(ids) for ids, metadatas in store._collection.updates
    )
