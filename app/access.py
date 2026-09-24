"""Small, file-backed authorization model for the local RAG service."""

from __future__ import annotations

import json
import os
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
    """Users belong to groups; each KB/document is visible to one or more groups."""

    users: dict[str, frozenset[str]]
    knowledge_bases: dict[str, str]
    documents: dict[str, frozenset[str]]
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
        documents: dict[str, frozenset[str]] = {}
        for doc_id, value in payload.get("documents", {}).items():
            if not isinstance(value, dict):
                continue
            raw_groups = value.get("groups")
            if isinstance(raw_groups, list):
                # 新格式：一个文档可属于多个组。
                documents[str(doc_id)] = frozenset(
                    str(group) for group in raw_groups if str(group).strip()
                )
            else:
                # 旧格式：{"group": "hr"} 包成单元素集合。
                documents[str(doc_id)] = frozenset(
                    {str(value.get("group", PUBLIC_GROUP))}
                )
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

    def document_groups(self, doc_id: str) -> frozenset[str]:
        return self.documents.get(doc_id, frozenset({PUBLIC_GROUP}))

    def document_group(self, doc_id: str) -> str:
        """主组（排序后第一个），保留给向后兼容的旧字段。"""
        groups = sorted(self.document_groups(doc_id))
        return groups[0] if groups else ""

    def all_groups(self) -> frozenset[str]:
        """策略里出现过的所有组（users + 知识库 + 文档），供切片时写 flag。"""
        groups = {PUBLIC_GROUP}
        for user_groups in self.users.values():
            groups.update(user_groups)
        groups.update(self.knowledge_bases.values())
        for doc_groups in self.documents.values():
            groups.update(doc_groups)
        return frozenset(groups)

    def can_access(self, metadata: dict[str, Any], user: UserContext, knowledge_base_id: str) -> bool:
        if self.is_admin(user):
            return metadata.get("knowledge_base_id") == knowledge_base_id
        return (
            metadata.get("knowledge_base_id") == knowledge_base_id
            and metadata.get("knowledge_base_access_group") in user.groups
            and bool(_metadata_doc_groups(metadata) & user.groups)
        )

    def chroma_filter(self, knowledge_base_id: str, user: UserContext) -> dict[str, Any] | None:
        # 管理员绕过组过滤，但保留知识库约束——返回 None 是「全放行」语义，会丢掉
        # knowledge_base_id 过滤，所以这里显式返回只含 kb 约束的过滤器。
        if self.is_admin(user):
            return {"knowledge_base_id": knowledge_base_id}
        if not user.groups:
            return None
        groups = sorted(user.groups)
        # 多组文档：切片元数据里为每个组存了 `access_<组>` 布尔 flag，
        # chromadb 对 list 型元数据的 $in 过滤不生效，所以用 flag 的 $or 命中；
        # 旧标量分支兼容从未改组的存量切片。
        return {
            "$and": [
                {"knowledge_base_id": knowledge_base_id},
                {"knowledge_base_access_group": {"$in": groups}},
                {
                    "$or": [
                        {"document_access_group": {"$in": groups}},
                        *[{"access_" + group: True} for group in groups],
                    ]
                },
            ]
        }


def _metadata_doc_groups(metadata: dict[str, Any]) -> frozenset[str]:
    """文档在切片元数据里的组集合：新格式是 list，旧格式是单个标量。"""
    raw = metadata.get("document_access_groups")
    if isinstance(raw, list):
        return frozenset(str(group) for group in raw if group)
    scalar = metadata.get("document_access_group")
    if scalar:
        return frozenset({str(scalar)})
    return frozenset()


def load_access_policy(path: Path) -> AccessPolicy | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid access policy: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Access policy must be a JSON object: %s" % path)
    return AccessPolicy.from_mapping(payload)


def known_groups(path: Path) -> set[str]:
    """Every group referenced anywhere in the policy, plus the public group.

    Admin user ids are deliberately *not* included: they are identities, not
    groups, and ``user_context`` never adds a user's own id to their groups — so
    a document assigned to one would be visible to nobody.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {PUBLIC_GROUP}
    if not isinstance(payload, dict):
        return {PUBLIC_GROUP}

    groups = {PUBLIC_GROUP}
    for user in (payload.get("users") or {}).values():
        if isinstance(user, dict):
            groups.update(str(group) for group in user.get("groups", []))
    for section in ("knowledge_bases", "documents"):
        for entry in (payload.get(section) or {}).values():
            if not isinstance(entry, dict):
                continue
            raw_groups = entry.get("groups")
            if isinstance(raw_groups, list):
                groups.update(str(group) for group in raw_groups)
            elif entry.get("group"):
                groups.add(str(entry["group"]))
    return groups


def set_document_groups(path: Path, doc_id: str, groups: list[str]) -> None:
    """把文档指派给一个或多个访问组，创建条目（覆盖旧组集合）时写入策略。

    Chunk metadata bakes in the groups at split time, so this must run *before*
    the document is indexed.
    """

    doc_id = doc_id.strip()
    clean = sorted({str(group).strip() for group in groups if str(group).strip()})
    if not doc_id:
        raise ValueError("doc_id is required")
    if not clean:
        raise ValueError("at least one group is required")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid access policy: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Access policy must be a JSON object: {path}")

    documents = payload.setdefault("documents", {})
    if not isinstance(documents, dict):
        raise ValueError(f"Policy 'documents' must be an object: {path}")
    documents[doc_id] = {"groups": clean}

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def document_access_metadata(
    metadata: dict[str, Any],
    knowledge_base_id: str,
    policy: AccessPolicy | None,
) -> None:
    metadata["knowledge_base_id"] = knowledge_base_id
    if policy is None:
        metadata.setdefault("knowledge_base_access_group", PUBLIC_GROUP)
        metadata.setdefault("document_access_group", PUBLIC_GROUP)
        metadata.setdefault("document_access_groups", [PUBLIC_GROUP])
        metadata.setdefault("access_" + PUBLIC_GROUP, True)
        return
    metadata["knowledge_base_access_group"] = policy.knowledge_base_group(knowledge_base_id)
    groups = policy.document_groups(str(metadata["doc_id"]))
    primary = sorted(groups)[0] if groups else ""
    metadata["document_access_group"] = primary
    metadata["document_access_groups"] = sorted(groups)
    # 全量 flag 一起写：重新索引会整包替换 chunk 元数据，写 False 保证之前
    # 属于、现在不属于的组不会留下陈旧 True。
    for group in policy.all_groups():
        metadata["access_" + group] = group in groups
