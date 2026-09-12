"""Server-side conversation history, one owner per conversation.

Turns are appended by the answer-job worker after a question completes, so the
store must tolerate a conversation being deleted mid-flight (a no-op, not an
error). Only the chunk *references* are kept — not the chunk bodies — so the
state file stays small; the full text lives in the vector store.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

MAX_CONVERSATIONS_PER_OWNER = 100
MAX_TURNS_PER_CONVERSATION = 200
MAX_ANSWER_CHARS = 8000
# How many prior turns are handed to the model — independent of the storage cap,
# so a long conversation does not blow up context or cost.
MAX_HISTORY_TURNS = 8
DEFAULT_TITLE = "新对话"

# Fields kept per retrieved chunk; the body ("text") is deliberately dropped.
HIT_FIELDS = (
    "rank",
    "doc_id",
    "title",
    "chunk_id",
    "source",
    "page",
    "page_label",
    "section_path",
    "distance",
    "similarity",
    "bm25_score",
    "retrieval_method",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def owner_of(user: Any) -> str:
    """Owner key for a UserContext; open mode (no policy) shares one bucket."""

    return user.user_id if user is not None else "open-mode"


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title[:40] + "…" if len(title) > 40 else title or DEFAULT_TITLE


def _slim_hits(hits: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        slim.append({key: hit[key] for key in HIT_FIELDS if key in hit})
    return slim


class ConversationStore:
    """File-backed conversation history, scoped by owner."""

    def __init__(self, state_path: Path) -> None:
        self._state_path = state_path
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    # --- writes -----------------------------------------------------------

    def create(self, owner: str) -> dict[str, Any]:
        conversation = {
            "id": uuid.uuid4().hex,
            "owner": owner or "open-mode",
            "title": DEFAULT_TITLE,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "turns": [],
        }
        with self._lock:
            self._items[conversation["id"]] = conversation
            self._prune_locked(conversation["owner"])
            self._save_locked()
        return self._summary(conversation)

    def append_turn(
        self,
        conversation_id: str,
        owner: str,
        question: str,
        answer: str,
        retrieval_method: str = "",
        hits: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Record one completed turn. Returns False if it no longer applies."""

        with self._lock:
            conversation = self._items.get(conversation_id)
            if conversation is None or conversation["owner"] != (owner or "open-mode"):
                # Deleted or reassigned while the answer was in flight.
                return False

            turns = conversation["turns"]
            turns.append(
                {
                    "question": question,
                    "answer": answer[:MAX_ANSWER_CHARS],
                    "retrieval_method": retrieval_method,
                    "hits": _slim_hits(hits),
                    "created_at": utc_now(),
                }
            )
            del turns[:-MAX_TURNS_PER_CONVERSATION]
            if conversation["title"] == DEFAULT_TITLE:
                conversation["title"] = _title_from(question)
            conversation["updated_at"] = utc_now()
            self._save_locked()
        return True

    def delete(self, conversation_id: str, owner: str) -> None:
        with self._lock:
            conversation = self._items.get(conversation_id)
            if conversation is None:
                raise KeyError("Conversation not found")
            if conversation["owner"] != (owner or "open-mode"):
                raise PermissionError("Conversation belongs to another user")
            del self._items[conversation_id]
            self._save_locked()

    # --- reads ------------------------------------------------------------

    def list_for(self, owner: str) -> list[dict[str, Any]]:
        owner = owner or "open-mode"
        with self._lock:
            mine = [c for c in self._items.values() if c["owner"] == owner]
            mine.sort(key=lambda c: c["updated_at"], reverse=True)
            return [self._summary(c) for c in mine]

    def get(self, conversation_id: str, owner: str) -> dict[str, Any]:
        with self._lock:
            conversation = self._items.get(conversation_id)
            if conversation is None:
                raise KeyError("Conversation not found")
            if conversation["owner"] != (owner or "open-mode"):
                raise PermissionError("Conversation belongs to another user")
            return json.loads(json.dumps(conversation))

    def owns(self, conversation_id: str, owner: str) -> bool:
        with self._lock:
            conversation = self._items.get(conversation_id)
            return conversation is not None and conversation["owner"] == (
                owner or "open-mode"
            )

    def history_for(
        self, conversation_id: str, owner: str, limit: int = MAX_HISTORY_TURNS
    ) -> list[dict[str, str]]:
        """Prior turns for the model, oldest first. Empty when unavailable."""

        with self._lock:
            conversation = self._items.get(conversation_id)
            if conversation is None or conversation["owner"] != (owner or "open-mode"):
                return []
            turns = conversation["turns"][-limit:]
            return [
                {"question": str(turn.get("question", "")), "answer": str(turn.get("answer", ""))}
                for turn in turns
            ]

    # --- internals --------------------------------------------------------

    @staticmethod
    def _summary(conversation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "turn_count": len(conversation["turns"]),
        }

    def _prune_locked(self, owner: str) -> None:
        mine = sorted(
            (c for c in self._items.values() if c["owner"] == owner),
            key=lambda c: c["updated_at"],
            reverse=True,
        )
        for conversation in mine[MAX_CONVERSATIONS_PER_OWNER:]:
            self._items.pop(conversation["id"], None)

    def _load(self) -> None:
        if not self._state_path.is_file():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable conversation store: %s", self._state_path)
            return
        if not isinstance(payload, list):
            return
        self._items = {
            str(item.get("id")): item
            for item in payload
            if isinstance(item, dict) and item.get("id") and isinstance(item.get("turns"), list)
        }

    def _save_locked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(list(self._items.values()), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self._state_path)


_store: ConversationStore | None = None
_store_lock = threading.Lock()


def get_conversation_store() -> ConversationStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ConversationStore(Path(".web_conversations.json"))
        return _store
