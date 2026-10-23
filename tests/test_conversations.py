"""Tests for the server-side conversation store."""

import json
from pathlib import Path

import pytest

from app import conversations
from app.conversations import (
    MAX_ANSWER_CHARS,
    MAX_HISTORY_TURNS,
    ConversationStore,
    owner_of,
)


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(tmp_path / "conversations.json")


def test_create_starts_empty_with_a_placeholder_title(store) -> None:
    conversation = store.create("alice")

    assert conversation["title"] == "新对话"
    assert conversation["turn_count"] == 0
    assert store.get(conversation["id"], "alice")["turns"] == []


def test_append_fills_the_title_from_the_first_question(store) -> None:
    conversation = store.create("alice")

    store.append_turn(conversation["id"], "alice", "开放问答为什么要先检索？", "因为……")
    store.append_turn(conversation["id"], "alice", "第二个问题", "答案")

    record = store.get(conversation["id"], "alice")
    assert record["title"] == "开放问答为什么要先检索？"
    assert len(record["turns"]) == 2


def test_long_title_is_truncated(store) -> None:
    conversation = store.create("alice")
    question = "很" * 80

    store.append_turn(conversation["id"], "alice", question, "答")

    title = store.get(conversation["id"], "alice")["title"]
    assert len(title) <= 41
    assert title.endswith("…")


def test_hits_keep_references_but_drop_the_body(store) -> None:
    conversation = store.create("alice")

    store.append_turn(
        conversation["id"],
        "alice",
        "q",
        "a",
        retrieval_method="hybrid",
        hits=[
            {
                "rank": 1,
                "chunk_id": "doc#chunk_001",
                "title": "文档",
                "similarity": 0.7,
                "vector_rank": 2,
                "bm25_rank": 1,
                "sources": ["vector", "bm25"],
                "rrf_score": 0.03,
                "text": "这一整段正文不应该被存进对话文件",
                "unknown_field": "也不该出现",
            }
        ],
    )

    hit = store.get(conversation["id"], "alice")["turns"][0]["hits"][0]
    assert hit["chunk_id"] == "doc#chunk_001"
    assert hit["rank"] == 1
    assert hit["vector_rank"] == 2 and hit["bm25_rank"] == 1
    assert hit["sources"] == ["vector", "bm25"]
    assert hit["rrf_score"] == 0.03
    assert "text" not in hit
    assert "unknown_field" not in hit


def test_turn_keeps_light_retrieval_summary(store) -> None:
    conversation = store.create("alice")

    store.append_turn(
        conversation["id"],
        "alice",
        "q",
        "a",
        retrieval_method="hybrid",
        hits=[{"chunk_id": "doc#chunk_001", "vector_rank": 2, "bm25_rank": 1}],
        retrieval={
            "retrieval_method": "hybrid",
            "vector_candidates": 100,
            "bm25_candidates": 87,
            "union_candidates": 152,
            "matches": 20,
            "retrieval_trace": {"candidates": [{"chunk_id": "doc#chunk_001"}]},
            "vector_hits": [{"chunk_id": "doc#chunk_001", "text": "正文"}],
            "unknown_field": "不该出现",
        },
    )

    turn = store.get(conversation["id"], "alice")["turns"][0]
    assert turn["retrieval"] == {
        "retrieval_method": "hybrid",
        "vector_candidates": 100,
        "bm25_candidates": 87,
        "union_candidates": 152,
        "matches": 20,
    }


def test_long_answers_are_truncated(store) -> None:
    conversation = store.create("alice")

    store.append_turn(conversation["id"], "alice", "q", "答" * (MAX_ANSWER_CHARS + 500))

    stored = store.get(conversation["id"], "alice")["turns"][0]["answer"]
    assert len(stored) == MAX_ANSWER_CHARS


def test_conversations_are_isolated_by_owner(store) -> None:
    alice = store.create("alice")
    bob = store.create("bob")

    # Each owner sees exactly their own conversation.
    assert [c["id"] for c in store.list_for("alice")] == [alice["id"]]
    assert [c["id"] for c in store.list_for("bob")] == [bob["id"]]
    assert alice["id"] != bob["id"]


def test_reading_another_owners_conversation_is_rejected(store) -> None:
    alice = store.create("alice")

    with pytest.raises(PermissionError):
        store.get(alice["id"], "bob")
    with pytest.raises(PermissionError):
        store.delete(alice["id"], "bob")
    assert store.owns(alice["id"], "alice") is True
    assert store.owns(alice["id"], "bob") is False


def test_deleting_removes_it_from_the_list(store) -> None:
    conversation = store.create("alice")

    store.delete(conversation["id"], "alice")

    assert store.list_for("alice") == []
    with pytest.raises(KeyError):
        store.get(conversation["id"], "alice")


def test_appending_to_a_deleted_conversation_is_a_no_op(store) -> None:
    conversation = store.create("alice")
    store.delete(conversation["id"], "alice")

    # The answer worker must not blow up when its conversation vanished mid-flight.
    assert store.append_turn(conversation["id"], "alice", "q", "a") is False


def test_appending_with_the_wrong_owner_is_refused(store) -> None:
    conversation = store.create("alice")

    assert store.append_turn(conversation["id"], "bob", "q", "a") is False
    assert store.get(conversation["id"], "alice")["turns"] == []


def test_history_is_capped_and_oldest_first(store) -> None:
    conversation = store.create("alice")
    for index in range(MAX_HISTORY_TURNS + 4):
        store.append_turn(conversation["id"], "alice", f"问题{index}", f"答案{index}")

    history = store.history_for(conversation["id"], "alice")

    assert len(history) == MAX_HISTORY_TURNS
    assert history[0]["question"] == "问题4"
    assert history[-1]["question"] == f"问题{MAX_HISTORY_TURNS + 3}"


def test_history_for_unknown_conversation_is_empty(store) -> None:
    assert store.history_for("nope", "alice") == []
    assert store.history_for("nope", "alice") == []


def test_history_for_another_owner_is_empty(store) -> None:
    conversation = store.create("alice")
    store.append_turn(conversation["id"], "alice", "q", "a")

    assert store.history_for(conversation["id"], "bob") == []


def test_turns_are_capped(store, monkeypatch) -> None:
    monkeypatch.setattr(conversations, "MAX_TURNS_PER_CONVERSATION", 3)
    conversation = store.create("alice")

    for index in range(5):
        store.append_turn(conversation["id"], "alice", f"q{index}", "a")

    turns = store.get(conversation["id"], "alice")["turns"]
    assert [t["question"] for t in turns] == ["q2", "q3", "q4"]


def test_state_survives_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "conversations.json"
    first = ConversationStore(path)
    conversation = first.create("alice")
    first.append_turn(conversation["id"], "alice", "q", "a")

    second = ConversationStore(path)

    assert second.get(conversation["id"], "alice")["turns"][0]["question"] == "q"


def test_a_corrupt_state_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "conversations.json"
    path.write_text("{not json", encoding="utf-8")

    store = ConversationStore(path)

    assert store.list_for("alice") == []
    # And it recovers by writing a valid file on the next change.
    conversation = store.create("alice")
    assert json.loads(path.read_text(encoding="utf-8"))[0]["id"] == conversation["id"]


def test_open_mode_uses_one_shared_bucket() -> None:
    assert owner_of(None) == "open-mode"

    class FakeUser:
        user_id = "alice"

    assert owner_of(FakeUser()) == "alice"


def test_get_conversation_store_honours_the_env_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONVERSATIONS_FILE", str(tmp_path / "from-env.json"))
    monkeypatch.setattr(conversations, "_store", None)

    store = conversations.get_conversation_store()

    assert conversations.conversation_store_path() == tmp_path / "from-env.json"
    store.create("alice")
    assert (tmp_path / "from-env.json").is_file()
    monkeypatch.setattr(conversations, "_store", None)
