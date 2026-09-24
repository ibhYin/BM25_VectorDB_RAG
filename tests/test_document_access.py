"""Per-document access management: metadata rewrite, policy write, endpoint."""

import json
from pathlib import Path

import pytest
from http import HTTPStatus

from app.access import known_groups
from app.config import Settings
from app.service import set_document_access


POLICY = {
    "admins": ["alice"],
    "users": {
        "alice": {"groups": ["hr", "engineering"]},
        "bob": {"groups": ["engineering"]},
    },
    "knowledge_bases": {"default": {"group": "public"}},
    "documents": {"doc": {"group": "public"}},
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(json.dumps(POLICY), encoding="utf-8")
    return Settings(
        _env_file=None,
        chroma_persist_dir=tmp_path / "chroma",
        knowledge_base_id="default",
        access_policy_file=policy_file,
    )


class FakeCollection:
    def __init__(self) -> None:
        self.updates: list[tuple[list[str], list[dict]]] = []

    def update(self, ids, metadatas):
        self.updates.append((list(ids), [dict(m) for m in metadatas]))


class FakeVectorStore:
    """Only the surface set_document_access_group touches."""

    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._collection = FakeCollection()

    def get(self, where, include):
        return {
            "ids": [f"doc#chunk_{index:03d}" for index in range(len(self._chunks))],
            "metadatas": [dict(chunk) for chunk in self._chunks],
        }


def _patch_store(monkeypatch, chunks: list[dict]) -> FakeVectorStore:
    store = FakeVectorStore(chunks)
    monkeypatch.setattr("rag.vector_store.get_vector_store", lambda *a, **k: store)
    return store


def test_group_change_rewrites_metadata_and_leaves_the_body_alone(
    settings, monkeypatch
) -> None:
    store = _patch_store(
        monkeypatch,
        [
            {"doc_id": "doc", "chunk_id": "doc#chunk_000", "document_access_group": "public", "page": 1},
            {"doc_id": "doc", "chunk_id": "doc#chunk_001", "document_access_group": "public", "page": 2},
        ],
    )

    updated = set_document_access(settings, "doc", ["engineering"], "alice")

    assert updated == 2
    ids, metadatas = store._collection.updates[0]
    assert ids == ["doc#chunk_000", "doc#chunk_001"]
    assert {m["document_access_group"] for m in metadatas} == {"engineering"}
    # 多组表示：完整列表 + 主组标量 + 每个已知组的布尔 flag。
    assert {tuple(m["document_access_groups"]) for m in metadatas} == {
        ("engineering",)
    }
    assert {m["access_engineering"] for m in metadatas} == {True}
    # 不在新组里的已知组 flag 被清掉，避免改组前遗留的陈旧 True。
    assert {m["access_hr"] for m in metadatas} == {False}
    assert {m["access_public"] for m in metadatas} == {False}
    # Everything else in the metadata survives the rewrite.
    assert [m["page"] for m in metadatas] == [1, 2]
    assert [m["chunk_id"] for m in metadatas] == ["doc#chunk_000", "doc#chunk_001"]


def test_group_change_to_multiple_groups_clears_old_flags(settings, monkeypatch) -> None:
    store = _patch_store(
        monkeypatch,
        [
            {
                "doc_id": "doc",
                "chunk_id": "doc#chunk_000",
                "document_access_group": "hr",
                "document_access_groups": ["hr"],
                "access_hr": True,
                "access_engineering": False,
                "access_public": False,
            }
        ],
    )

    updated = set_document_access(settings, "doc", ["hr", "engineering"], "alice")

    assert updated == 1
    _, metadatas = store._collection.updates[0]
    metadata = metadatas[0]
    assert metadata["document_access_groups"] == ["engineering", "hr"]
    assert metadata["document_access_group"] == "engineering"
    assert metadata["access_hr"] is True
    assert metadata["access_engineering"] is True
    assert metadata["access_public"] is False


def test_group_change_also_updates_the_policy_file(settings, monkeypatch) -> None:
    _patch_store(monkeypatch, [{"doc_id": "doc", "document_access_group": "public"}])

    set_document_access(settings, "doc", ["engineering"], "alice")

    saved = json.loads(settings.access_policy_file.read_text(encoding="utf-8"))
    assert saved["documents"]["doc"] == {"groups": ["engineering"]}


def test_unknown_group_is_rejected(settings, monkeypatch) -> None:
    _patch_store(monkeypatch, [{"doc_id": "doc", "document_access_group": "public"}])

    with pytest.raises(ValueError, match="未知的访问组"):
        set_document_access(settings, "doc", ["legal"], "alice")

    # Nothing was written.
    saved = json.loads(settings.access_policy_file.read_text(encoding="utf-8"))
    assert saved["documents"]["doc"] == {"group": "public"}


def test_missing_policy_file_is_rejected(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        chroma_persist_dir=tmp_path / "chroma",
        access_policy_file=tmp_path / "missing.json",
    )
    _patch_store(monkeypatch, [{"doc_id": "doc", "document_access_group": "public"}])

    with pytest.raises(ValueError, match="未启用访问控制"):
        set_document_access(settings, "doc", ["engineering"], "alice")


def test_unknown_document_is_rejected(settings, monkeypatch) -> None:
    _patch_store(monkeypatch, [])

    with pytest.raises(KeyError):
        set_document_access(settings, "nope", ["engineering"], "alice")

    # A bogus doc_id must not leave a policy entry behind.
    saved = json.loads(settings.access_policy_file.read_text(encoding="utf-8"))
    assert "nope" not in saved["documents"]


def test_the_change_is_audited(settings, monkeypatch) -> None:
    from app import audit

    _patch_store(monkeypatch, [{"doc_id": "doc", "document_access_group": "public"}])

    set_document_access(settings, "doc", ["engineering"], "alice")

    entry = audit.read_entries()[0]
    assert entry["event"] == "access.change"
    assert entry["actor"] == "alice"
    assert entry["doc_ids"] == ["doc"]
    assert entry["access_groups"] == ["engineering"]
    assert entry["chunks"] == 1


# --- admin ids must not be offered as groups ------------------------------


def test_admin_ids_are_not_offered_as_groups(tmp_path: Path) -> None:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(json.dumps(POLICY), encoding="utf-8")

    groups = known_groups(policy_file)

    assert groups == {"public", "hr", "engineering"}
    # "alice" is an identity. Assigning a document to it would hide the document
    # from everyone, because user_context never adds a user's own id to groups.
    assert "alice" not in groups


# --- endpoint -------------------------------------------------------------


def _handler(user):
    from app.web import WebRequestHandler

    handler = object.__new__(WebRequestHandler)
    captured: dict = {}
    handler._require_admin = lambda: user
    handler._read_json = lambda: {"groups": ["engineering"]}
    handler._send_json = lambda body, status=HTTPStatus.OK: captured.update(
        body=body, status=int(status)
    )
    return handler, captured


def test_endpoint_reports_the_updated_chunk_count(settings, monkeypatch) -> None:
    from app.access import UserContext

    _patch_store(
        monkeypatch,
        [
            {"doc_id": "doc", "document_access_group": "public"},
            {"doc_id": "doc", "document_access_group": "public"},
        ],
    )
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler, captured = _handler(UserContext("alice", frozenset({"hr"})))

    handler._handle_document_access("doc")

    assert captured["status"] == HTTPStatus.OK
    assert captured["body"]["updated_chunks"] == 2
    assert captured["body"]["access_groups"] == ["engineering"]


def test_endpoint_accepts_legacy_single_group(settings, monkeypatch) -> None:
    from app.access import UserContext

    _patch_store(monkeypatch, [{"doc_id": "doc", "document_access_group": "public"}])
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler, captured = _handler(UserContext("alice", frozenset({"hr"})))
    handler._read_json = lambda: {"group": "engineering"}

    handler._handle_document_access("doc")

    assert captured["body"]["access_groups"] == ["engineering"]


def test_endpoint_maps_an_unknown_group_to_400(settings, monkeypatch) -> None:
    from app.access import UserContext
    from app.web import RequestError

    _patch_store(monkeypatch, [{"doc_id": "doc", "document_access_group": "public"}])
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler, _ = _handler(UserContext("alice", frozenset({"hr"})))
    handler._read_json = lambda: {"groups": ["legal"]}

    with pytest.raises(RequestError) as error:
        handler._handle_document_access("doc")

    assert error.value.status == HTTPStatus.BAD_REQUEST


def test_endpoint_maps_a_missing_document_to_404(settings, monkeypatch) -> None:
    from app.access import UserContext
    from app.web import RequestError

    _patch_store(monkeypatch, [])
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    handler, _ = _handler(UserContext("alice", frozenset({"hr"})))

    with pytest.raises(RequestError) as error:
        handler._handle_document_access("nope")

    assert error.value.status == HTTPStatus.NOT_FOUND
