"""Small, file-backed authorization model for the local RAG service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PUBLIC_GROUP = "public"


@dataclass(frozen=True)
class UserContext:
    user_id: str
    groups: frozenset[str]


@dataclass(frozen=True)
class AccessPolicy:
    """Users belong to groups; each KB/document is visible to one group."""

    users: dict[str, frozenset[str]]
    knowledge_bases: dict[str, str]
    documents: dict[str, str]
    admins: frozenset[str]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AccessPolicy":
        users = {
            str(user_id): frozenset(str(group) for group in value.get("groups", []))
            for user_id, value in payload.get("users", {}).items()
            if isinstance(value, dict)
        }
        knowledge_bases = {
            str(kb_id): str(value.get("group", PUBLIC_GROUP))
            for kb_id, value in payload.get("knowledge_bases", {}).items()
            if isinstance(value, dict)
        }
        documents = {
            str(doc_id): str(value.get("group", PUBLIC_GROUP))
            for doc_id, value in payload.get("documents", {}).items()
            if isinstance(value, dict)
        }
        admins = frozenset(str(user_id) for user_id in payload.get("admins", []))
        return cls(users, knowledge_bases, documents, admins)

    def user_context(self, user_id: str) -> UserContext:
        user_id = user_id.strip()
        if not user_id or user_id not in self.users:
            raise PermissionError("Unknown user")
        # Public documents are visible to every authenticated user.
        return UserContext(user_id, self.users[user_id] | {PUBLIC_GROUP})

    def is_admin(self, user: UserContext) -> bool:
        return user.user_id in self.admins

    def knowledge_base_group(self, knowledge_base_id: str) -> str:
        return self.knowledge_bases.get(knowledge_base_id, PUBLIC_GROUP)

    def document_group(self, doc_id: str) -> str:
        return self.documents.get(doc_id, PUBLIC_GROUP)

    def can_access(self, metadata: dict[str, Any], user: UserContext, knowledge_base_id: str) -> bool:
        return (
            metadata.get("knowledge_base_id") == knowledge_base_id
            and metadata.get("knowledge_base_access_group") in user.groups
            and metadata.get("document_access_group") in user.groups
        )

    def chroma_filter(self, knowledge_base_id: str, user: UserContext) -> dict[str, Any] | None:
        if not user.groups:
            return None
        groups = sorted(user.groups)
        return {
            "$and": [
                {"knowledge_base_id": knowledge_base_id},
                {"knowledge_base_access_group": {"$in": groups}},
                {"document_access_group": {"$in": groups}},
            ]
        }


def load_access_policy(path: Path) -> AccessPolicy | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid access policy: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Access policy must be a JSON object: {path}")
    return AccessPolicy.from_mapping(payload)


def document_access_metadata(
    metadata: dict[str, Any],
    knowledge_base_id: str,
    policy: AccessPolicy | None,
) -> None:
    metadata["knowledge_base_id"] = knowledge_base_id
    if policy is None:
        metadata.setdefault("knowledge_base_access_group", PUBLIC_GROUP)
        metadata.setdefault("document_access_group", PUBLIC_GROUP)
        return
    metadata["knowledge_base_access_group"] = policy.knowledge_base_group(knowledge_base_id)
    metadata["document_access_group"] = policy.document_group(str(metadata["doc_id"]))
