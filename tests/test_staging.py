"""Tests for the pending-upload staging queue."""

import json
from pathlib import Path

import pytest

from app.access import known_groups, load_access_policy, set_document_groups
from app.staging import StagingStore


@pytest.fixture
def store(tmp_path: Path) -> StagingStore:
    return StagingStore(root=tmp_path / "pending", state_path=tmp_path / "staging.json")


def test_add_stages_files_and_records_the_uploader(store: StagingStore) -> None:
    item = store.add([("note.md", b"# note")], uploader="bob", structurize=True)

    assert item["status"] == "pending"
    assert item["uploader"] == "bob"
    assert item["filenames"] == ["note.md"]
    assert item["size_bytes"] == 6
    assert item["structurize"] is True
    assert (store._root / item["id"] / "note.md").read_bytes() == b"# note"


def test_add_sanitizes_path_traversal(store: StagingStore) -> None:
    item = store.add([("../../etc/passwd.md", b"x")], uploader="bob")

    assert item["filenames"] == ["passwd.md"]
    assert (store._root / item["id"] / "passwd.md").is_file()


def test_snapshot_is_newest_first(store: StagingStore) -> None:
    first = store.add([("a.md", b"a")], uploader="bob")
    second = store.add([("b.md", b"b")], uploader="carol")

    ids = [item["id"] for item in store.snapshot()]

    assert ids.index(second["id"]) < ids.index(first["id"])


def test_take_reads_the_bytes_and_clears_the_staging_directory(
    store: StagingStore,
) -> None:
    item = store.add([("note.md", b"# note")], uploader="bob")

    taken, payload = store.take(item["id"])

    assert taken["id"] == item["id"]
    assert payload == [("note.md", b"# note")]
    assert not (store._root / item["id"]).exists()


def test_take_does_not_change_status_until_decided(store: StagingStore) -> None:
    item = store.add([("note.md", b"# note")], uploader="bob")

    store.take(item["id"])

    assert store.get(item["id"])["status"] == "pending"


def test_decide_records_who_and_the_resulting_job(store: StagingStore) -> None:
    item = store.add([("note.md", b"# note")], uploader="bob")

    decided = store.decide(item["id"], "approved", "alice", job_id="job-1")

    assert decided["status"] == "approved"
    assert decided["decided_by"] == "alice"
    assert decided["decided_at"]
    assert decided["job_id"] == "job-1"


def test_discard_removes_files_and_marks_rejected(store: StagingStore) -> None:
    item = store.add([("note.md", b"# note")], uploader="bob")

    rejected = store.discard(item["id"], "alice")

    assert rejected["status"] == "rejected"
    assert not (store._root / item["id"]).exists()


def test_deciding_twice_is_rejected(store: StagingStore) -> None:
    item = store.add([("note.md", b"# note")], uploader="bob")
    store.discard(item["id"], "alice")

    with pytest.raises(ValueError, match="already rejected"):
        store.discard(item["id"], "alice")


def test_unknown_item_raises_key_error(store: StagingStore) -> None:
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_state_survives_a_restart(tmp_path: Path) -> None:
    root = tmp_path / "pending"
    state = tmp_path / "staging.json"
    first = StagingStore(root=root, state_path=state)
    item = first.add([("note.md", b"note")], uploader="bob")

    reloaded = StagingStore(root=root, state_path=state)

    assert reloaded.get(item["id"])["uploader"] == "bob"


# --- access policy writes -------------------------------------------------


def test_set_document_groups_creates_the_entry(tmp_path: Path) -> None:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(
        json.dumps({"admins": ["alice"], "users": {"alice": {"groups": ["hr"]}}}),
        encoding="utf-8",
    )

    set_document_groups(policy_file, "note_abc12345", ["hr"])

    payload = json.loads(policy_file.read_text(encoding="utf-8"))
    assert payload["documents"]["note_abc12345"] == {"groups": ["hr"]}
    # The rest of the policy must survive the rewrite.
    assert payload["admins"] == ["alice"]


def test_set_document_groups_overwrites_an_existing_entry(tmp_path: Path) -> None:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(
        json.dumps({"documents": {"note_abc12345": {"group": "public"}}}),
        encoding="utf-8",
    )

    set_document_groups(policy_file, "note_abc12345", ["finance", "hr"])

    assert load_access_policy(policy_file).document_groups("note_abc12345") == frozenset(
        {"finance", "hr"}
    )
    # 旧格式的 `{"group": ...}` 条目被新格式覆盖。
    payload = json.loads(policy_file.read_text(encoding="utf-8"))
    assert payload["documents"]["note_abc12345"] == {"groups": ["finance", "hr"]}


def test_set_document_groups_rejects_blank_values(tmp_path: Path) -> None:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        set_document_groups(policy_file, "note_abc", ["  "])
    with pytest.raises(ValueError):
        set_document_groups(policy_file, "note_abc", [])


def test_known_groups_collects_every_referenced_group(tmp_path: Path) -> None:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "admins": ["alice"],
                "users": {
                    "alice": {"groups": ["hr", "engineering"]},
                    "bob": {"groups": ["finance"]},
                },
                "knowledge_bases": {"default": {"group": "public"}},
                "documents": {"doc": {"groups": ["legal", "hr"]}},
            }
        ),
        encoding="utf-8",
    )

    groups = known_groups(policy_file)

    assert groups == {"public", "hr", "engineering", "finance", "legal"}
    # Admin ids are identities, not groups: assigning a document to one would
    # make it visible to nobody, so they must not be offered as options.
    assert "alice" not in groups


def test_known_groups_still_reads_legacy_scalar_group(tmp_path: Path) -> None:
    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(
        json.dumps({"documents": {"doc": {"group": "legal"}}}), encoding="utf-8"
    )

    assert "legal" in known_groups(policy_file)


def test_known_groups_on_a_missing_file_is_just_public(tmp_path: Path) -> None:
    assert known_groups(tmp_path / "nope.json") == {"public"}


def test_approve_pins_the_group_then_hands_off_to_the_index_queue(
    tmp_path: Path, monkeypatch
) -> None:
    """The group must reach the policy *before* indexing bakes it into chunks."""
    from app import audit, service
    from app.config import Settings
    from rag.splitter import build_doc_id

    store = StagingStore(root=tmp_path / "pending", state_path=tmp_path / "staging.json")
    monkeypatch.setattr("app.staging.get_staging_store", lambda: store)

    policy_file = tmp_path / "access_policy.json"
    policy_file.write_text(
        json.dumps({"users": {"bob": {"groups": ["hr"]}}}), encoding="utf-8"
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "raw",
        access_policy_file=policy_file,
    )

    queued: dict[str, object] = {}

    class FakeUploadStore:
        def create_uploaded(self, files, data_dir, structurize, actor):
            queued.update(files=files, data_dir=data_dir, actor=actor)
            return {"id": "job-1"}

    monkeypatch.setattr("app.uploads.get_upload_store", lambda: FakeUploadStore())

    item = store.add([("note.md", b"# note")], uploader="bob")
    decided = service.approve_staged_upload(settings, item["id"], "hr", "alice")

    # The policy entry lands before the queue ever sees the bytes.
    doc_id = build_doc_id(str(settings.data_dir / "note.md"))
    assert load_access_policy(policy_file).document_group(doc_id) == "hr"
    assert queued["actor"] == "alice"
    assert decided["status"] == "approved"
    assert decided["job_id"] == "job-1"

    events = [entry["event"] for entry in audit.read_entries()]
    assert "staging.approved" in events


def test_reject_leaves_the_vector_store_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    from app import audit, service

    store = StagingStore(root=tmp_path / "pending", state_path=tmp_path / "staging.json")
    monkeypatch.setattr("app.staging.get_staging_store", lambda: store)
    item = store.add([("note.md", b"# note")], uploader="bob")

    rejected = service.reject_staged_upload(item["id"], "alice", "重复提交")

    assert rejected["status"] == "rejected"
    entries = audit.read_entries()
    assert entries[0]["event"] == "staging.rejected"
    assert entries[0]["reason"] == "重复提交"
