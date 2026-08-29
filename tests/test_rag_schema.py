from langchain_core.documents import Document

from kg.extractor import (
    Entity,
    KnowledgeGraphExtraction,
    Relationship,
    validate_extraction,
)
from rag.splitter import split_documents


def test_split_documents_adds_stable_chunk_metadata() -> None:
    documents = [
        Document(
            page_content="第一段介绍数据安全委员会负责统筹管理。\n\n第二段介绍管理要求。",
            metadata={"source": "data/raw/policy.md"},
        )
    ]

    chunks = split_documents(documents, chunk_size=30, chunk_overlap=0)

    assert chunks[0].metadata["doc_id"].startswith("policy_")
    assert chunks[0].metadata["chunk_id"].endswith("#chunk_001")
    assert chunks[0].metadata["title"] == "policy"
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == len(chunks[0].page_content)


def test_split_documents_keeps_markdown_section_path() -> None:
    documents = [
        Document(
            page_content="## 数据安全职责\n\n数据安全委员会负责统筹管理。",
            metadata={"source": "data/raw/policy.md"},
        )
    ]

    chunks = split_documents(documents, chunk_size=80, chunk_overlap=0)

    assert chunks[0].metadata["section_path"] == "数据安全职责"


def test_validate_extraction_keeps_only_supported_evidence() -> None:
    text = "数据安全委员会负责数据安全统筹管理。"
    extraction = KnowledgeGraphExtraction(
        entities=[
            Entity(name="数据安全委员会", type="Department"),
            Entity(name="数据安全统筹管理", type="Concept"),
        ],
        relationships=[
            Relationship(
                source="数据安全委员会",
                target="数据安全统筹管理",
                type="RESPONSIBLE_FOR",
                evidence="数据安全委员会负责数据安全统筹管理",
                source_chunk_id="policy_2024_001#chunk_003",
                confidence=0.91,
            ),
            Relationship(
                source="数据安全委员会",
                target="数据安全统筹管理",
                type="RESPONSIBLE_FOR",
                evidence="文本中不存在的证据",
                source_chunk_id="policy_2024_001#chunk_003",
                confidence=0.95,
            ),
            Relationship(
                source="数据安全委员会",
                target="数据安全统筹管理",
                type="RESPONSIBLE_FOR",
                evidence="数据安全委员会负责数据安全统筹管理",
                source_chunk_id="policy_2024_001#chunk_003",
                confidence=0.2,
            ),
        ],
    )

    validated = validate_extraction(
        extraction,
        text=text,
        chunk_id="policy_2024_001#chunk_003",
    )

    assert len(validated.relationships) == 1
    assert validated.relationships[0].evidence == "数据安全委员会负责数据安全统筹管理"
    assert validated.entities[0].source_chunk_ids == ["policy_2024_001#chunk_003"]


def test_validate_extraction_drops_empty_and_duplicate_entities() -> None:
    extraction = KnowledgeGraphExtraction(
        entities=[
            Entity(name="  Transformer ", type="Model"),
            Entity(name="Transformer", type="Model"),
            Entity(name=" ", type="Concept"),
        ]
    )

    validated = validate_extraction(extraction, text="Transformer", chunk_id="chunk-1")

    assert [entity.name for entity in validated.entities] == ["Transformer"]


def test_validate_extraction_normalizes_provenance_and_deduplicates_relations() -> None:
    text = "Transformer 使用注意力机制。"
    extraction = KnowledgeGraphExtraction(
        entities=[
            Entity(
                name=" Transformer ",
                type="Model",
                aliases=[" Transformer ", "模型", "模型"],
                source_chunk_ids=["hallucinated-chunk"],
            ),
            Entity(name="注意力机制", type="Method"),
        ],
        relationships=[
            Relationship(
                source=" Transformer ",
                target="注意力机制 ",
                type="USES",
                evidence=" Transformer 使用注意力机制。 ",
                confidence=0.8,
            ),
            Relationship(
                source="Transformer",
                target="注意力机制",
                type="USES",
                evidence=text,
                confidence=0.9,
            ),
        ],
    )

    validated = validate_extraction(extraction, text=text, chunk_id="chunk-1")

    assert validated.entities[0].aliases == ["模型"]
    assert all(entity.source_chunk_ids == ["chunk-1"] for entity in validated.entities)
    assert len(validated.relationships) == 1
    assert validated.relationships[0].confidence == 0.9
    assert validated.relationships[0].source_type == "Model"
    assert validated.relationships[0].target_type == "Method"
