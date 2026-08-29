"""Dependency-free records used by offline graph processing and bulk import."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEntity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GraphRelationship:
    source: str
    target: str
    type: str
    evidence: str = ""
    source_chunk_id: str = ""
    confidence: float = 0.0
    source_type: str = ""
    target_type: str = ""


@dataclass
class GraphTriple:
    triple_id: str
    subject: str
    predicate: str
    object: str
    predicate_label: str
    subject_type: str
    object_type: str
    evidence: str
    source_chunk_id: str
    confidence: float


def entity_from_dict(raw: dict[str, Any]) -> GraphEntity:
    return GraphEntity(
        name=str(raw.get("name") or ""),
        type=str(raw.get("type") or "Entity"),
        aliases=list(raw.get("aliases") or []),
        description=str(raw.get("description") or ""),
        source_chunk_ids=list(raw.get("source_chunk_ids") or []),
    )


def relationship_from_dict(raw: dict[str, Any]) -> GraphRelationship:
    return GraphRelationship(
        source=str(raw.get("source") or ""),
        target=str(raw.get("target") or ""),
        type=str(raw.get("type") or "RELATED_TO"),
        evidence=str(raw.get("evidence") or ""),
        source_chunk_id=str(raw.get("source_chunk_id") or ""),
        confidence=float(raw.get("confidence") or 0.0),
        source_type=str(raw.get("source_type") or ""),
        target_type=str(raw.get("target_type") or ""),
    )


def triple_from_dict(raw: dict[str, Any]) -> GraphTriple:
    return GraphTriple(
        triple_id=str(raw.get("triple_id") or ""),
        subject=str(raw.get("subject") or ""),
        predicate=str(raw.get("predicate") or "RELATED_TO"),
        object=str(raw.get("object") or ""),
        predicate_label=str(raw.get("predicate_label") or ""),
        subject_type=str(raw.get("subject_type") or "Entity"),
        object_type=str(raw.get("object_type") or "Entity"),
        evidence=str(raw.get("evidence") or ""),
        source_chunk_id=str(raw.get("source_chunk_id") or ""),
        confidence=float(raw.get("confidence") or 0.0),
    )
