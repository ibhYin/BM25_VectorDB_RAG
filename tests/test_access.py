from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.access import AccessPolicy
from app.config import Settings
from app.service import answer_question, get_indexed_document, list_indexed_documents
from rag.retriever import search_bm25
from rag.splitter import split_documents


def policy() -> AccessPolicy:
    return AccessPolicy.from_mapping(
        {
            "admins": ["alice"],
            "users": {
                "alice": {"groups": ["engineering"]},
                "bob": {"groups": ["finance"]},
            },
            "knowledge_bases": {
                "engineering": {"group": "engineering"},
                "finance": {"group": "finance"},
            },
            "documents": {"secret": {"group": "engineering"}},
        }
    )


def test_access_is_intersection_of_kb_and_document_groups() -> None:
    access = policy()
    bob = access.user_context("bob")  # finance，非管理员

    assert access.can_access(
        {
            "knowledge_base_id": "finance",
            "knowledge_base_access_group": "finance",
            "document_access_group": "finance",
        },
        bob,
        "finance",
    )
    # 文档组不匹配 → 拒（知识库组与文档组取交集）
    assert not access.can_access(
        {
            "knowledge_base_id": "finance",
            "knowledge_base_access_group": "finance",
            "document_access_group": "engineering",
        },
        bob,
        "finance",
    )
    assert access.chroma_filter("finance", bob)["$and"][-1] == {
        "document_access_group": {"$in": ["finance", "public"]}
    }


def test_admin_bypasses_group_filters_but_keeps_kb_constraint() -> None:
    access = policy()
    alice = access.user_context("alice")  # admin

    # can_access：管理员只看 knowledge_base_id，跨组文档也放行
    assert access.can_access(
        {
            "knowledge_base_id": "engineering",
            "knowledge_base_access_group": "engineering",
            "document_access_group": "finance",
        },
        alice,
        "engineering",
    )
    # 知识库约束仍然生效：别库的文档不放行
    assert not access.can_access(
        {
            "knowledge_base_id": "engineering",
            "knowledge_base_access_group": "engineering",
            "document_access_group": "engineering",
        },
        alice,
        "finance",
    )
    # chroma_filter：管理员只保留 kb 约束，不再夹组 $in 过滤
    assert access.chroma_filter("engineering", alice) == {
        "knowledge_base_id": "engineering"
    }


def test_non_admin_still_filtered_by_groups() -> None:
    access = policy()
    bob = access.user_context("bob")  # finance，非管理员

    assert not access.can_access(
        {
            "knowledge_base_id": "engineering",
            "knowledge_base_access_group": "engineering",
            "document_access_group": "finance",
        },
        bob,
        "engineering",
    )
    assert access.chroma_filter("engineering", bob)["$and"][-1] == {
        "document_access_group": {"$in": ["finance", "public"]}
    }


def test_split_documents_copies_acl_metadata_to_every_chunk() -> None:
    chunks = split_documents(
        [Document(page_content="机密内容", metadata={"source": "secret.md", "doc_id": "secret"})],
        chunk_size=100,
        chunk_overlap=0,
        knowledge_base_id="engineering",
        access_policy=policy(),
    )

    assert chunks[0].metadata["knowledge_base_id"] == "engineering"
    assert chunks[0].metadata["knowledge_base_access_group"] == "engineering"
    assert chunks[0].metadata["document_access_group"] == "engineering"


def test_bm25_filters_unauthorized_chunks(tmp_path: Path) -> None:
    import chromadb

    from rag.vector_store import COLLECTION_NAME

    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        COLLECTION_NAME, embedding_function=None,
    )
    collection.add(
        ids=["allowed", "denied"],
        documents=["财务秘密A", "财务秘密B"],
        metadatas=[
            {"knowledge_base_id": "finance", "knowledge_base_access_group": "finance", "document_access_group": "finance", "chunk_id": "allowed"},
            {"knowledge_base_id": "finance", "knowledge_base_access_group": "finance", "document_access_group": "engineering", "chunk_id": "denied"},
        ],
        embeddings=[[1.0], [1.0]],
    )

    # bob（finance 组）能召回 allowed，denied 的文档组不在其组内。
    hits = search_bm25("秘密", tmp_path, 20, "finance", policy().user_context("bob"), policy())
    assert [hit["chunk_id"] for hit in hits] == ["allowed"]


def test_vector_search_receives_acl_filter(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class Embeddings:
        def embed_query(self, question):
            return [1.0]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, embedding, k, filter):
            captured["filter"] = filter
            return []

    monkeypatch.setattr("app.service.get_embeddings", lambda *args: Embeddings())
    monkeypatch.setattr("app.service.get_vector_store", lambda *args: Store())
    monkeypatch.setattr("chains.answer_chain.generate_answer", lambda **kwargs: "answer")
    policy_path = tmp_path / "access.json"
    settings = Settings(_env_file=None, access_policy_file=policy_path)
    policy_path.write_text(
        '{"users": {"alice": {"groups": ["engineering"]}}, '
        '"knowledge_bases": {"engineering": {"group": "engineering"}}}',
        encoding="utf-8",
    )
    from app.access import load_access_policy

    answer_question(
        "问题", settings, user=load_access_policy(policy_path).user_context("alice")
    )

    assert captured["filter"]["$and"][0] == {"knowledge_base_id": "default"}


def test_answer_requires_user_when_policy_is_enabled(tmp_path: Path) -> None:
    policy_path = tmp_path / "access.json"
    policy_path.write_text('{"users": {"alice": {"groups": ["public"]}}}', encoding="utf-8")
    with pytest.raises(PermissionError, match="Authenticated user"):
        answer_question(
            "问题",
            Settings(_env_file=None, chroma_persist_dir=tmp_path, access_policy_file=policy_path),
        )


def test_document_listing_filters_unauthorized_metadata(tmp_path: Path, monkeypatch) -> None:
    class Store:
        def get(self, include):
            return {
                "metadatas": [
                    {"doc_id": "ok", "title": "ok", "knowledge_base_id": "engineering", "knowledge_base_access_group": "engineering", "document_access_group": "engineering"},
                    {"doc_id": "no", "title": "no", "knowledge_base_id": "engineering", "knowledge_base_access_group": "engineering", "document_access_group": "finance"},
                ]
            }

    policy_path = tmp_path / "access.json"
    policy_path.write_text('{"users": {"alice": {"groups": ["engineering"]}}}', encoding="utf-8")
    monkeypatch.setattr("app.service.get_vector_store", lambda *args, **kwargs: Store())
    settings = Settings(
        _env_file=None, access_policy_file=policy_path, knowledge_base_id="engineering",
    )

    assert [doc["doc_id"] for doc in list_indexed_documents(settings, policy().user_context("alice"))] == ["ok"]


def test_document_detail_hides_unauthorized_chunks(tmp_path: Path, monkeypatch) -> None:
    class Store:
        def get(self, where, include):
            return {
                "documents": ["财务机密"],
                "metadatas": [
                    {
                        "doc_id": "secret",
                        "chunk_id": "secret#chunk_001",
                        "knowledge_base_id": "engineering",
                        "knowledge_base_access_group": "engineering",
                        "document_access_group": "finance",
                    }
                ],
            }

    policy_path = tmp_path / "access.json"
    policy_path.write_text(
        '{"users": {"alice": {"groups": ["engineering"]}}}', encoding="utf-8"
    )
    monkeypatch.setattr("app.service.get_vector_store", lambda *args, **kwargs: Store())
    settings = Settings(
        _env_file=None,
        access_policy_file=policy_path,
        knowledge_base_id="engineering",
        chroma_persist_dir=tmp_path,
    )

    with pytest.raises(KeyError):
        get_indexed_document(settings, "secret", policy().user_context("alice"))
