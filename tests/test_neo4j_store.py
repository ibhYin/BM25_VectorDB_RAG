import json

import pytest

from kg.neo4j_store import extract_search_terms, normalize_entity_id, normalize_relationship_type
from kg.neo4j_store import KnowledgeGraphStore


def test_normalize_relationship_type_keeps_safe_uppercase_format() -> None:
    assert normalize_relationship_type("part of") == "PART_OF"
    assert normalize_relationship_type("causes!") == "CAUSES"
    assert normalize_relationship_type("") == "RELATED_TO"


def test_normalize_entity_id_uses_type_and_name() -> None:
    assert normalize_entity_id("Company", "Open AI 公司") == "company:open_ai_公司"


def test_extract_search_terms_keeps_chinese_phrases_for_entity_matching() -> None:
    assert extract_search_terms("Transformer 为什么使用注意力机制？") == [
        "transformer",
        "为什么使用注意力机制",
    ]


def test_import_triples_batches_by_safe_relationship_type(tmp_path) -> None:
    triple_path = tmp_path / "triples.jsonl"
    triple_path.write_text(
        json.dumps(
            {
                "triple_id": "triple_1",
                "subject": "Transformer",
                "predicate": "USES",
                "object": "注意力机制",
                "predicate_label": "使用",
                "subject_type": "Model",
                "object_type": "Method",
                "evidence": "Transformer 使用注意力机制",
                "source_chunk_id": "book#chunk_001",
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        def __init__(self):
            self.calls = []

        def query(self, query, params=None):
            self.calls.append((query, params))
            return []

    store = object.__new__(KnowledgeGraphStore)
    store.graph = FakeGraph()

    imported = store.import_triples(triple_path)

    assert imported == 1
    query, params = store.graph.calls[0]
    assert "[rel:USES" in query
    assert params["rows"][0]["source_id"] == "model:transformer"


def test_clear_happens_only_after_the_entire_import_file_is_valid(tmp_path) -> None:
    triple_path = tmp_path / "invalid.jsonl"
    triple_path.write_text("{}\nnot-json\n", encoding="utf-8")

    class FakeGraph:
        def __init__(self):
            self.calls = []

        def query(self, query, params=None):
            self.calls.append((query, params))

    store = object.__new__(KnowledgeGraphStore)
    store.graph = FakeGraph()

    with pytest.raises(ValueError):
        store.import_triples(triple_path, clear=True)

    assert store.graph.calls == []
