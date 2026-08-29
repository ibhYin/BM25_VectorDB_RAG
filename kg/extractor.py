from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

from chains.models import get_chat_model
from kg.schema import ENTITY_TYPES, RELATION_TYPES


class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type from the configured schema")
    aliases: list[str] = Field(default_factory=list, description="Entity aliases")
    description: str = Field(default="", description="Short description")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids that mention this entity",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {value}")
        return value


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type from the configured schema")
    evidence: str = Field(default="", description="Evidence from the source text")
    source_chunk_id: str = Field(
        default="",
        description="Chunk id that supports this relation",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type: str = Field(default="", description="Source entity type")
    target_type: str = Field(default="", description="Target entity type")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in RELATION_TYPES:
            raise ValueError(f"Unsupported relationship type: {value}")
        return value

    @field_validator("source_type", "target_type")
    @classmethod
    def validate_entity_type(cls, value: str) -> str:
        if value and value not in ENTITY_TYPES:
            raise ValueError(f"Unsupported endpoint entity type: {value}")
        return value


class KnowledgeGraphExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是面向技术文档的知识图谱信息抽取系统。只从给定文本中抽取明确事实，不要推测。"
            f"实体类型只能使用：{', '.join(sorted(ENTITY_TYPES))}。"
            f"关系类型只能使用：{', '.join(sorted(RELATION_TYPES))}。"
            "重点抽取书籍、章节、概念、算法、模型、任务、方法、数据结构、库、API、指标、数学对象和应用场景。"
            "每条关系必须填写 source_type、target_type、原文 evidence、source_chunk_id "
            "和 0 到 1 的 confidence。"
            "evidence 必须是原文中连续出现的短句。没有明确事实时返回空数组。",
        ),
        (
            "human",
            "chunk_id: {chunk_id}\nbook: {book}\nsection: {section}"
            "\n\n请从下面文本中抽取实体和关系：\n\n{text}",
        ),
    ]
)


def extract_knowledge_graph(
    text: str,
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    chunk_id: str = "",
    confidence_threshold: float = 0.7,
    book: str = "",
    section: str = "",
) -> KnowledgeGraphExtraction:
    llm = get_chat_model(model_name, openai_api_key, openai_base_url)
    chain = EXTRACTION_PROMPT | llm.with_structured_output(KnowledgeGraphExtraction)
    extraction = chain.invoke(
        {
            "text": text,
            "chunk_id": chunk_id,
            "book": book,
            "section": section,
        }
    )
    return validate_extraction(
        extraction,
        text=text,
        chunk_id=chunk_id,
        confidence_threshold=confidence_threshold,
    )


def extract_knowledge_graph_from_chunk(
    chunk: Document,
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    confidence_threshold: float = 0.7,
) -> KnowledgeGraphExtraction:
    return extract_knowledge_graph(
        text=chunk.page_content,
        model_name=model_name,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        chunk_id=str(chunk.metadata.get("chunk_id", "")),
        confidence_threshold=confidence_threshold,
        book=str(chunk.metadata.get("title", "")),
        section=str(chunk.metadata.get("section_path", "")),
    )


def validate_extraction(
    extraction: KnowledgeGraphExtraction,
    text: str,
    chunk_id: str,
    confidence_threshold: float = 0.7,
) -> KnowledgeGraphExtraction:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    chunk_id = chunk_id.strip()
    valid_entities: list[Entity] = []
    entity_keys: set[tuple[str, str]] = set()
    entity_names: set[str] = set()
    for entity in extraction.entities:
        entity.name = entity.name.strip()
        entity_key = (entity.name, entity.type)
        if not entity.name or entity_key in entity_keys:
            continue
        entity.aliases = list(
            dict.fromkeys(
                alias.strip()
                for alias in entity.aliases
                if alias.strip() and alias.strip() != entity.name
            )
        )
        entity.description = entity.description.strip()
        entity.source_chunk_ids = (
            [chunk_id]
            if chunk_id
            else list(
                dict.fromkeys(
                    source_id.strip()
                    for source_id in entity.source_chunk_ids
                    if source_id.strip()
                )
            )
        )
        entity_names.add(entity.name)
        entity_keys.add(entity_key)
        valid_entities.append(entity)
    extraction.entities = valid_entities
    valid_relationships: dict[tuple[str, str, str, str, str, str], Relationship] = {}
    entity_types_by_name: dict[str, set[str]] = {}
    for entity in extraction.entities:
        entity_types_by_name.setdefault(entity.name, set()).add(entity.type)

    for relationship in extraction.relationships:
        relationship.source = relationship.source.strip()
        relationship.target = relationship.target.strip()
        relationship.evidence = relationship.evidence.strip()
        relationship.source_chunk_id = relationship.source_chunk_id.strip() or chunk_id
        if relationship.confidence < confidence_threshold:
            continue
        if relationship.source not in entity_names or relationship.target not in entity_names:
            continue
        source_types = entity_types_by_name[relationship.source]
        target_types = entity_types_by_name[relationship.target]
        if relationship.source_type and relationship.source_type not in source_types:
            continue
        if relationship.target_type and relationship.target_type not in target_types:
            continue
        if not relationship.source_type and len(source_types) == 1:
            relationship.source_type = next(iter(source_types))
        if not relationship.target_type and len(target_types) == 1:
            relationship.target_type = next(iter(target_types))
        if not relationship.source_type or not relationship.target_type:
            continue
        if chunk_id and relationship.source_chunk_id != chunk_id:
            continue
        if (
            relationship.source == relationship.target
            and relationship.source_type == relationship.target_type
        ):
            continue
        if not relationship.evidence or relationship.evidence not in text:
            continue
        key = (
            relationship.source,
            relationship.source_type,
            relationship.type,
            relationship.target,
            relationship.target_type,
            relationship.source_chunk_id,
        )
        current = valid_relationships.get(key)
        if current is None or relationship.confidence > current.confidence:
            valid_relationships[key] = relationship

    extraction.relationships = sorted(
        valid_relationships.values(),
        key=lambda item: (item.source, item.type, item.target, item.source_chunk_id),
    )
    return extraction
