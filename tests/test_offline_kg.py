import pytest

from kg.records import GraphEntity, GraphRelationship
from kg.extractor import Entity, KnowledgeGraphExtraction, Relationship
from scripts.kg.align import align_graph, resolve_relationship_endpoint
from scripts.kg.export import create_triples, normalize_text
from scripts.kg.extract import (
    SimpleDocument,
    extract_rule_based_from_chunk,
    extract_with_llm_from_chunk,
)
from scripts.kg.io import staged_output_dir


def test_offline_pipeline_keeps_shared_records_compatible() -> None:
    chunk = SimpleDocument(
        page_content="Transformer 使用注意力机制解决文本生成任务。",
        metadata={
            "source": "transformer.md",
            "title": "Transformer 教程",
            "chunk_id": "transformer#chunk_001",
        },
    )

    extraction = extract_rule_based_from_chunk(chunk)
    aligned = align_graph(
        extraction.entities,
        extraction.relationships,
        max_name_chars=60,
        fuzzy_threshold=0.96,
        min_confidence=0.65,
    )
    triples, dropped = create_triples(
        aligned.entities,
        aligned.relationships,
        min_confidence=0.65,
        include_relation_types=set(),
        exclude_relation_types=set(),
        max_endpoint_chars=80,
        strict_endpoints=True,
    )

    assert any(entity.name == "Transformer" for entity in aligned.entities)
    assert triples
    assert all(triple.source_chunk_id == "transformer#chunk_001" for triple in triples)
    assert not dropped


def test_offline_llm_mode_reuses_the_canonical_extractor(monkeypatch) -> None:
    captured = {}

    def fake_extract_knowledge_graph(**kwargs):
        captured.update(kwargs)
        return KnowledgeGraphExtraction(
            entities=[
                Entity(
                    name="Transformer",
                    type="Model",
                    source_chunk_ids=[kwargs["chunk_id"]],
                )
            ],
            relationships=[
                Relationship(
                    source="Transformer",
                    target="Transformer",
                    type="RELATED_TO",
                    source_type="Model",
                    target_type="Model",
                    evidence="Transformer",
                    source_chunk_id=kwargs["chunk_id"],
                    confidence=0.8,
                )
            ],
        )

    monkeypatch.setattr("kg.extractor.extract_knowledge_graph", fake_extract_knowledge_graph)
    chunk = SimpleDocument(
        page_content="Transformer",
        metadata={
            "chunk_id": "book#chunk_001",
            "title": "Book",
            "section_path": "Chapter",
        },
    )

    result = extract_with_llm_from_chunk(chunk, "test-model")

    assert captured["book"] == "Book"
    assert captured["section"] == "Chapter"
    assert result.entities == [
        GraphEntity(
            name="Transformer",
            type="Model",
            source_chunk_ids=["book#chunk_001"],
        )
    ]
    assert result.relationships[0].source_type == "Model"


def test_alignment_does_not_guess_ambiguous_endpoint_types() -> None:
    entities = {
        "注意力机制": [
            GraphEntity(name="注意力机制", type="Concept"),
            GraphEntity(name="注意力机制", type="Method"),
        ]
    }
    mapping = {
        ("Concept", "注意力机制"): ("注意力机制", "Concept"),
        ("Method", "注意力机制"): ("注意力机制", "Method"),
    }

    assert resolve_relationship_endpoint("注意力机制", entities, mapping) == ("", "")


def test_export_drops_relationships_with_ambiguous_endpoint_types() -> None:
    entities = [
        GraphEntity(name="注意力机制", type="Concept"),
        GraphEntity(name="注意力机制", type="Method"),
        GraphEntity(name="Transformer", type="Model"),
    ]
    relationships = [
        GraphRelationship(
            source="Transformer",
            target="注意力机制",
            type="USES",
            source_chunk_id="chunk-1",
            confidence=0.9,
        )
    ]

    triples, dropped = create_triples(
        entities,
        relationships,
        min_confidence=0.65,
        include_relation_types=set(),
        exclude_relation_types=set(),
        max_endpoint_chars=80,
        strict_endpoints=True,
    )

    assert triples == []
    assert dropped[0]["reason"] == "endpoint_not_in_entity_index"


def test_export_removes_control_characters_that_break_csv() -> None:
    assert normalize_text("公式\x00内容\x01结束") == "公式内容结束"


def test_staged_output_keeps_previous_files_when_generation_fails(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "result.json"
    existing.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with staged_output_dir(output_dir) as staged:
            (staged / "result.json").write_text("new", encoding="utf-8")
            raise RuntimeError("generation failed")

    assert existing.read_text(encoding="utf-8") == "old"
