import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from app.config import Settings
from kg.extractor import KnowledgeGraphExtraction
from kg.records import GraphTriple, triple_from_dict
from kg.schema import ENTITY_TYPES, RELATION_TYPES
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph


logger = logging.getLogger(__name__)


def normalize_relationship_type(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value).upper().strip("_")
    return normalized or "RELATED_TO"


def normalize_entity_id(entity_type: str, name: str) -> str:
    normalized_type = re.sub(r"[^a-z0-9]+", "_", entity_type.lower()).strip("_")
    normalized_name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", name).strip("_").lower()
    return f"{normalized_type or 'entity'}:{normalized_name or 'unknown'}"


def extract_search_terms(question: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", question.lower())
    return [term for term in terms if len(term.strip()) > 1][:8]


class KnowledgeGraphStore:
    """Small Neo4j adapter for storing and retrieving knowledge graph context."""

    def __init__(self, settings: Settings) -> None:
        self.graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )

    def setup_schema(self) -> None:
        self.graph.query(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (entity:Entity) REQUIRE entity.id IS UNIQUE
            """
        )
        self.graph.query(
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (chunk:Chunk) REQUIRE chunk.id IS UNIQUE
            """
        )
        self.graph.query(
            """
            CREATE CONSTRAINT document_id_unique IF NOT EXISTS
            FOR (document:Document) REQUIRE document.id IS UNIQUE
            """
        )

    def upsert_chunk(self, chunk: Document) -> None:
        metadata = chunk.metadata
        self.graph.query(
            """
            MERGE (document:Document {id: $doc_id})
            SET document.title = $title,
                document.source = $source,
                document.created_at = $created_at
            MERGE (chunk:Chunk {id: $chunk_id})
            SET chunk.doc_id = $doc_id,
                chunk.text = $text,
                chunk.start_char = $start_char,
                chunk.end_char = $end_char,
                chunk.source = $source
            MERGE (document)-[:HAS_CHUNK]->(chunk)
            """,
            {
                "doc_id": metadata.get("doc_id", ""),
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "created_at": metadata.get("created_at", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "text": chunk.page_content,
                "start_char": metadata.get("start_char", 0),
                "end_char": metadata.get("end_char", 0),
            },
        )

    def upsert_extraction(
        self,
        extraction: KnowledgeGraphExtraction,
        chunk: Document | None = None,
    ) -> None:
        if chunk is not None:
            self.upsert_chunk(chunk)

        entities_by_name: dict[str, list] = defaultdict(list)
        for entity in extraction.entities:
            entities_by_name[entity.name].append(entity)
        for entity in extraction.entities:
            entity_id = normalize_entity_id(entity.type, entity.name)
            self.graph.query(
                """
                MERGE (entity:Entity {id: $id})
                SET entity.type = $type,
                    entity.name = $name,
                    entity.aliases = reduce(
                        aliases = [],
                        alias IN coalesce(entity.aliases, []) + $aliases |
                        CASE WHEN alias IN aliases THEN aliases ELSE aliases + alias END
                    ),
                    entity.description = CASE
                        WHEN $description <> "" THEN $description
                        ELSE coalesce(entity.description, "")
                    END,
                    entity.source_chunk_ids = reduce(
                        chunks = [],
                        chunk_id IN coalesce(entity.source_chunk_ids, []) + $source_chunk_ids |
                        CASE WHEN chunk_id IN chunks THEN chunks ELSE chunks + chunk_id END
                    )
                """,
                {
                    "id": entity_id,
                    "name": entity.name,
                    "type": entity.type,
                    "aliases": entity.aliases,
                    "description": entity.description,
                    "source_chunk_ids": entity.source_chunk_ids,
                },
            )
            for chunk_id in entity.source_chunk_ids:
                self.graph.query(
                    """
                    MATCH (entity:Entity {id: $entity_id})
                    MERGE (chunk:Chunk {id: $chunk_id})
                    MERGE (entity)-[:MENTIONED_IN]->(chunk)
                    """,
                    {"entity_id": entity_id, "chunk_id": chunk_id},
                )

        for relationship in extraction.relationships:
            relationship_type = normalize_relationship_type(relationship.type)
            source_candidates = entities_by_name.get(relationship.source, [])
            target_candidates = entities_by_name.get(relationship.target, [])
            if relationship.source_type:
                source_candidates = [
                    entity
                    for entity in source_candidates
                    if entity.type == relationship.source_type
                ]
            if relationship.target_type:
                target_candidates = [
                    entity
                    for entity in target_candidates
                    if entity.type == relationship.target_type
                ]
            if len(source_candidates) != 1 or len(target_candidates) != 1:
                logger.warning(
                    "Skipping ambiguous KG relationship %s -> %s",
                    relationship.source,
                    relationship.target,
                )
                continue
            source_entity = source_candidates[0]
            target_entity = target_candidates[0]
            source_id = normalize_entity_id(source_entity.type, source_entity.name)
            target_id = normalize_entity_id(target_entity.type, target_entity.name)
            self.graph.query(
                f"""
                MERGE (source:Entity {{id: $source_id}})
                MERGE (target:Entity {{id: $target_id}})
                MERGE (source)-[rel:{relationship_type} {{source_chunk_id: $source_chunk_id}}]->(target)
                SET rel.evidence = $evidence,
                    rel.confidence = $confidence
                """,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "evidence": relationship.evidence,
                    "source_chunk_id": relationship.source_chunk_id,
                    "confidence": relationship.confidence,
                },
            )

    def clear(self) -> None:
        """Delete the application graph. Call only for an explicit rebuild."""

        self.graph.query("MATCH (node) DETACH DELETE node")

    def import_triples(
        self,
        path: Path,
        batch_size: int = 1_000,
        clear: bool = False,
    ) -> int:
        """Bulk import the canonical JSONL triple export into Neo4j."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not path.is_file():
            raise ValueError(f"Triple file does not exist: {path}")

        triples: list[GraphTriple] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("expected a JSON object")
                    triple = triple_from_dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid triple at line {line_number}: {exc}") from exc
                self._validate_triple(triple, line_number)
                triples.append(triple)

        if clear:
            self.clear()
        for start in range(0, len(triples), batch_size):
            self._upsert_triple_batch(triples[start : start + batch_size])
        return len(triples)

    def _validate_triple(self, triple: GraphTriple, line_number: int) -> None:
        if not triple.subject.strip() or not triple.object.strip():
            raise ValueError(f"Invalid triple at line {line_number}: empty endpoint")
        if not triple.subject_type.strip() or not triple.object_type.strip():
            raise ValueError(f"Invalid triple at line {line_number}: empty endpoint type")
        if triple.subject_type not in ENTITY_TYPES or triple.object_type not in ENTITY_TYPES:
            raise ValueError(f"Invalid triple at line {line_number}: unsupported endpoint type")
        if normalize_relationship_type(triple.predicate) not in RELATION_TYPES:
            raise ValueError(
                f"Invalid triple at line {line_number}: unsupported predicate "
                f"{triple.predicate!r}"
            )
        if not 0 <= triple.confidence <= 1:
            raise ValueError(f"Invalid triple at line {line_number}: confidence outside 0..1")

    def _upsert_triple_batch(self, triples: list[GraphTriple]) -> None:
        by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
        for triple in triples:
            relationship_type = normalize_relationship_type(triple.predicate)
            by_type[relationship_type].append(
                {
                    "triple_id": triple.triple_id,
                    "source_id": normalize_entity_id(triple.subject_type, triple.subject),
                    "source_name": triple.subject,
                    "source_type": triple.subject_type,
                    "target_id": normalize_entity_id(triple.object_type, triple.object),
                    "target_name": triple.object,
                    "target_type": triple.object_type,
                    "evidence": triple.evidence,
                    "source_chunk_id": triple.source_chunk_id,
                    "confidence": triple.confidence,
                }
            )

        for relationship_type, rows in by_type.items():
            self.graph.query(
                f"""
                UNWIND $rows AS row
                MERGE (source:Entity {{id: row.source_id}})
                SET source.name = row.source_name,
                    source.type = row.source_type
                MERGE (target:Entity {{id: row.target_id}})
                SET target.name = row.target_name,
                    target.type = row.target_type
                MERGE (source)-[rel:{relationship_type} {{source_chunk_id: row.source_chunk_id}}]->(target)
                SET rel.triple_id = row.triple_id,
                    rel.evidence = row.evidence,
                    rel.confidence = row.confidence
                """,
                {"rows": rows},
            )

    def search_context(self, question: str, limit: int = 8) -> list[str]:
        terms = extract_search_terms(question)

        if not terms:
            return []

        rows = self.graph.query(
            """
            MATCH (source:Entity)-[rel]->(target:Entity)
            WHERE any(
                term IN $terms
                WHERE toLower(source.name) CONTAINS term
                   OR toLower(target.name) CONTAINS term
                   OR term CONTAINS toLower(source.name)
                   OR term CONTAINS toLower(target.name)
                   OR toLower(coalesce(source.description, "")) CONTAINS term
                   OR toLower(coalesce(target.description, "")) CONTAINS term
            )
            RETURN source.name AS source,
                   type(rel) AS relationship,
                   target.name AS target,
                   coalesce(rel.evidence, "") AS evidence,
                   coalesce(rel.source_chunk_id, "") AS source_chunk_id,
                   coalesce(rel.confidence, 0.0) AS confidence
            LIMIT $limit
            """,
            {"terms": terms, "limit": limit},
        )

        return [
            (
                f"[Graph Fact {index}] "
                f"{row['source']} -[{row['relationship']}]-> {row['target']}"
                f"\nevidence: {row['evidence']}"
                f"\nsource: {row['source_chunk_id']}"
                f"\nconfidence: {row['confidence']}"
            )
            for index, row in enumerate(rows, start=1)
        ]
