from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg.records import GraphEntity as Entity
from kg.records import GraphRelationship as Relationship
from kg.schema import ENTITY_TYPES, RELATION_TYPES
from scripts.kg.io import staged_output_dir


@dataclass
class TripleExtraction:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TermSpec:
    name: str
    type: str
    aliases: tuple[str, ...] = ()
    description: str = ""


# 规则抽取的领域词表：先覆盖大模型、NLP、NumPy/Pandas、算法图解中的高频实体。
TERM_SPECS = [
    TermSpec("大语言模型", "Concept", ("大型语言模型", "LLM", "Large Language Model")),
    TermSpec("Transformer", "Model", ("Transformer模型", "Transformer 架构")),
    TermSpec("注意力机制", "Method", ("Attention", "自注意力", "Self-Attention")),
    TermSpec("预训练", "Method", ("Pre-training", "预训练语言模型")),
    TermSpec("微调", "Method", ("Fine-tuning", "监督微调", "SFT")),
    TermSpec("指令微调", "Method", ("Instruction Tuning",)),
    TermSpec("强化学习", "Method", ("RL", "Reinforcement Learning")),
    TermSpec("人类反馈强化学习", "Method", ("RLHF",)),
    TermSpec("提示工程", "Method", ("Prompt Engineering", "Prompt")),
    TermSpec("检索增强生成", "Method", ("RAG", "Retrieval-Augmented Generation")),
    TermSpec("词嵌入", "Concept", ("Embedding", "词向量", "Word Embedding")),
    TermSpec("Token", "Concept", ("标记", "词元", "Tokenization")),
    TermSpec("BERT", "Model"),
    TermSpec("GPT", "Model"),
    TermSpec("Seq2Seq", "Model", ("序列到序列",)),
    TermSpec("Word2Vec", "Model"),
    TermSpec("语言模型", "Model"),
    TermSpec("文本分类", "Task"),
    TermSpec("机器翻译", "Task"),
    TermSpec("问答系统", "Application", ("QA", "Question Answering")),
    TermSpec("信息抽取", "Task"),
    TermSpec("文本生成", "Task"),
    TermSpec("命名实体识别", "Task", ("NER",)),
    TermSpec("词性标注", "Task"),
    TermSpec("句法分析", "Task"),
    TermSpec("分词", "Task"),
    TermSpec("NumPy", "Library", ("numpy",)),
    TermSpec("Pandas", "Library", ("pandas",)),
    TermSpec("ndarray", "DataStructure"),
    TermSpec("Series", "DataStructure"),
    TermSpec("DataFrame", "DataStructure"),
    TermSpec("loc", "API"),
    TermSpec("iloc", "API"),
    TermSpec("groupby", "API"),
    TermSpec("merge", "API"),
    TermSpec("concat", "API"),
    TermSpec("read_csv", "API"),
    TermSpec("数组", "DataStructure"),
    TermSpec("矩阵", "MathObject"),
    TermSpec("向量", "MathObject"),
    TermSpec("概率", "MathObject"),
    TermSpec("梯度", "MathObject"),
    TermSpec("损失函数", "Metric"),
    TermSpec("准确率", "Metric", ("Accuracy",)),
    TermSpec("召回率", "Metric", ("Recall",)),
    TermSpec("精确率", "Metric", ("Precision",)),
    TermSpec("F1", "Metric", ("F1值", "F1-score")),
    TermSpec("困惑度", "Metric", ("Perplexity",)),
    TermSpec("余弦相似度", "Metric"),
    TermSpec("时间复杂度", "Metric"),
    TermSpec("空间复杂度", "Metric"),
    TermSpec("二分查找", "Algorithm", ("Binary Search",)),
    TermSpec("递归", "Method", ("Recursion",)),
    TermSpec("快速排序", "Algorithm", ("Quicksort", "快排")),
    TermSpec("归并排序", "Algorithm", ("Merge Sort",)),
    TermSpec("广度优先搜索", "Algorithm", ("BFS",)),
    TermSpec("深度优先搜索", "Algorithm", ("DFS",)),
    TermSpec("Dijkstra算法", "Algorithm", ("Dijkstra", "狄克斯特拉算法")),
    TermSpec("动态规划", "Algorithm", ("DP",)),
    TermSpec("贪心算法", "Algorithm", ("Greedy Algorithm",)),
    TermSpec("K近邻算法", "Algorithm", ("KNN", "k近邻")),
    TermSpec("线性回归", "Algorithm"),
    TermSpec("逻辑回归", "Algorithm"),
    TermSpec("决策树", "Algorithm"),
    TermSpec("聚类", "Task", ("Clustering",)),
    TermSpec("分类", "Task"),
    TermSpec("回归", "Task"),
    TermSpec("最短路径问题", "Task"),
    TermSpec("搜索问题", "Task"),
    TermSpec("推荐系统", "Application"),
    TermSpec("搜索引擎", "Application"),
    TermSpec("聊天机器人", "Application"),
    TermSpec("代码生成", "Application"),
    TermSpec("列表", "DataStructure"),
    TermSpec("字典", "DataStructure"),
    TermSpec("栈", "DataStructure"),
    TermSpec("队列", "DataStructure"),
    TermSpec("散列表", "DataStructure", ("哈希表", "Hash Table")),
    TermSpec("图", "DataStructure"),
    TermSpec("树", "DataStructure"),
]


# 关系触发词表：当两个实体出现在同一句话里时，用中间动词粗略判断关系。
METHOD_RELATION_KEYWORDS = (
    ("使用", "USES"),
    ("利用", "USES"),
    ("采用", "USES"),
    ("基于", "BASED_ON"),
    ("依赖", "REQUIRES"),
    ("解决", "SOLVES"),
    ("用于", "APPLIES_TO"),
    ("适用于", "APPLIES_TO"),
    ("评估", "EVALUATED_BY"),
    ("衡量", "EVALUATED_BY"),
    ("实现", "IMPLEMENTS"),
    ("提供", "PROVIDES"),
)


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def entity_key(entity: Entity) -> tuple[str, str]:
    return entity.type, normalize_name(entity.name).lower()


def relationship_key(
    relationship: Relationship,
) -> tuple[str, str, str, str, str, str]:
    return (
        normalize_name(relationship.source).lower(),
        relationship.source_type,
        normalize_name(relationship.target).lower(),
        relationship.target_type,
        relationship.type,
        relationship.source_chunk_id,
    )


def annotate_relationship_types(
    entities: Iterable[Entity],
    relationships: Iterable[Relationship],
) -> None:
    types_by_name: dict[str, set[str]] = {}
    for entity in entities:
        types_by_name.setdefault(normalize_name(entity.name).lower(), set()).add(entity.type)
    for relationship in relationships:
        source_types = types_by_name.get(normalize_name(relationship.source).lower(), set())
        target_types = types_by_name.get(normalize_name(relationship.target).lower(), set())
        if not relationship.source_type and len(source_types) == 1:
            relationship.source_type = next(iter(source_types))
        if not relationship.target_type and len(target_types) == 1:
            relationship.target_type = next(iter(target_types))


def as_jsonable(extraction: TripleExtraction) -> dict[str, object]:
    return {
        "entities": [asdict(entity) for entity in extraction.entities],
        "relationships": [asdict(relationship) for relationship in extraction.relationships],
    }


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    # 英文/API 名称按词边界匹配，避免把 loc 匹配到 block 这类单词内部。
    if re.search(r"[A-Za-z0-9_]", term):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return term in text


def find_evidence_sentence(text: str, names: Iterable[str]) -> str:
    # 关系 evidence 优先返回同时包含所有实体名的短句，方便人工回溯来源。
    compact_text = re.sub(r"\s+", " ", text).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?；;])\s*|\n+", compact_text)
        if sentence.strip()
    ]
    names = [name for name in names if name]
    for sentence in sentences:
        if all(contains_term(sentence, name) for name in names):
            return sentence[:240]
    for sentence in sentences:
        if any(contains_term(sentence, name) for name in names):
            return sentence[:240]
    return compact_text[:240]


def add_entity(
    entities: dict[tuple[str, str], Entity],
    name: str,
    entity_type: str,
    chunk_id: str,
    aliases: Iterable[str] = (),
    description: str = "",
) -> Entity:
    # 同一个 chunk 内按 type + name 去重，同时合并别名和来源 chunk。
    name = normalize_name(name)
    if not name or entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unsupported entity: {name!r} / {entity_type!r}")

    key = (entity_type, name.lower())
    entity = entities.get(key)
    if entity is None:
        entity = Entity(name=name, type=entity_type, description=description)
        entities[key] = entity

    for alias in aliases:
        alias = normalize_name(alias)
        if alias and alias != entity.name and alias not in entity.aliases:
            entity.aliases.append(alias)
    if chunk_id and chunk_id not in entity.source_chunk_ids:
        entity.source_chunk_ids.append(chunk_id)
    if description and not entity.description:
        entity.description = description
    return entity


def add_relationship(
    relationships: dict[tuple[str, str, str, str, str, str], Relationship],
    source: str,
    target: str,
    relation_type: str,
    chunk_id: str,
    evidence: str,
    confidence: float,
) -> None:
    # 关系也去重，并把置信度限制在 0 到 1，保持导出 JSON 可直接消费。
    if not source or not target or source == target or relation_type not in RELATION_TYPES:
        return
    relationship = Relationship(
        source=normalize_name(source),
        target=normalize_name(target),
        type=relation_type,
        evidence=normalize_name(evidence),
        source_chunk_id=chunk_id,
        confidence=max(0.0, min(1.0, confidence)),
    )
    relationships.setdefault(relationship_key(relationship), relationship)


def get_chunk_id(chunk) -> str:
    return str(chunk.metadata.get("chunk_id") or chunk.metadata.get("source") or "chunk")


def get_book_title(chunk) -> str:
    source = str(chunk.metadata.get("source") or "")
    title = str(chunk.metadata.get("title") or "")
    if title:
        return title
    return Path(source).stem if source else "未知文档"


def get_section_names(chunk) -> list[str]:
    # 优先使用 splitter 推断出的章节路径；没有时再从 Markdown 标题兜底提取。
    section_path = str(chunk.metadata.get("section_path") or "")
    sections = [normalize_name(section) for section in section_path.split(">")]
    sections = [section for section in sections if section]
    if sections:
        return sections

    headings = [
        normalize_name(line.lstrip("#"))
        for line in chunk.page_content.splitlines()
        if line.lstrip().startswith("#")
    ]
    return [heading for heading in headings if heading]


def extract_defined_terms(text: str) -> list[str]:
    # 从“X 是/指/定义为 Y”这类句式里发现词表之外的新概念。
    patterns = [
        r"([A-Za-z][A-Za-z0-9_\- ]{1,40}|[\u4e00-\u9fffA-Za-z0-9_\-]{2,30})\s*[，,]?\s*(?:是|指|表示|定义为)",
        r"(?:所谓|其中)\s*([A-Za-z][A-Za-z0-9_\- ]{1,40}|[\u4e00-\u9fffA-Za-z0-9_\-]{2,30})\s*(?:是|指)",
        r"([A-Za-z][A-Za-z0-9_\- ]{1,40}|[\u4e00-\u9fffA-Za-z0-9_\-]{2,30})\s*[（(][A-Za-z][A-Za-z0-9_\- ]{1,50}[）)]",
    ]
    terms: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            term = normalize_name(match.group(1))
            if 2 <= len(term) <= 40 and not re.fullmatch(r"\d+", term):
                terms.append(term)
    return list(dict.fromkeys(terms))


def classify_defined_term(term: str) -> str:
    if contains_term(term, "算法") or contains_term(term, "搜索") or contains_term(term, "排序"):
        return "Algorithm"
    if contains_term(term, "模型") or term.upper() in {"BERT", "GPT"}:
        return "Model"
    if contains_term(term, "任务") or contains_term(term, "问题"):
        return "Task"
    if contains_term(term, "率") or contains_term(term, "复杂度") or contains_term(term, "函数"):
        return "Metric"
    return "Concept"


def extract_rule_based_from_chunk(chunk) -> TripleExtraction:
    # 离线抽取主流程：文档结构关系 + 领域词表命中 + 定义句式 + 同句关系推断。
    text = chunk.page_content or ""
    chunk_id = get_chunk_id(chunk)
    book_title = get_book_title(chunk)
    section_names = get_section_names(chunk)
    entities: dict[tuple[str, str], Entity] = {}
    relationships: dict[tuple[str, str, str, str, str, str], Relationship] = {}

    add_entity(entities, book_title, "Book", chunk_id)

    # 建立 Book/Chapter/Section 的包含关系，让图谱能回答“在哪本书哪一节”。
    previous_section = ""
    for section in section_names:
        section_type = "Chapter" if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节篇部分]", section) else "Section"
        add_entity(entities, section, section_type, chunk_id)
        add_relationship(
            relationships,
            previous_section or book_title,
            section,
            "CONTAINS",
            chunk_id,
            section,
            0.95,
        )
        previous_section = section

    anchor = section_names[-1] if section_names else book_title
    matched_entities: list[Entity] = []
    # 先用人工维护的领域词表做高精度实体识别。
    for spec in TERM_SPECS:
        names = (spec.name, *spec.aliases)
        if any(contains_term(text, name) for name in names):
            entity = add_entity(
                entities,
                spec.name,
                spec.type,
                chunk_id,
                aliases=spec.aliases,
                description=spec.description,
            )
            matched_entities.append(entity)
            add_relationship(
                relationships,
                anchor,
                entity.name,
                "MENTIONS",
                chunk_id,
                find_evidence_sentence(text, [entity.name]),
                0.72,
            )

    known_names = {entity.name.lower() for entity in entities.values()}
    # 再从定义句里补充词表没有列出的教材概念。
    for term in extract_defined_terms(text):
        if term.lower() in known_names:
            continue
        entity = add_entity(entities, term, classify_defined_term(term), chunk_id)
        matched_entities.append(entity)
        add_relationship(
            relationships,
            anchor,
            entity.name,
            "DEFINES",
            chunk_id,
            find_evidence_sentence(text, [entity.name]),
            0.66,
        )

    # 最后只在同一句中推断实体间关系，减少跨段落误连。
    infer_rule_based_relationships(text, matched_entities, relationships, chunk_id)

    sorted_entities = sorted(entities.values(), key=lambda item: (item.type, item.name))
    sorted_relationships = sorted(
        relationships.values(),
        key=lambda item: (item.source, item.type, item.target),
    )
    annotate_relationship_types(sorted_entities, sorted_relationships)
    return TripleExtraction(entities=sorted_entities, relationships=sorted_relationships)


def infer_rule_based_relationships(
    text: str,
    entities: list[Entity],
    relationships: dict[tuple[str, str, str, str, str, str], Relationship],
    chunk_id: str,
) -> None:
    # 规则关系抽取比较保守：两个实体必须同句出现，再根据触发词或实体类型配对。
    if len(entities) < 2:
        return

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
        if sentence.strip()
    ]
    for sentence in sentences:
        mentioned = [
            entity
            for entity in entities
            if contains_term(sentence, entity.name)
            or any(contains_term(sentence, alias) for alias in entity.aliases)
        ]
        if len(mentioned) < 2:
            continue
        for source in mentioned:
            for target in mentioned:
                if source.name == target.name:
                    continue
                relation_type = infer_relation_type(sentence, source, target)
                if relation_type:
                    add_relationship(
                        relationships,
                        source.name,
                        target.name,
                        relation_type,
                        chunk_id,
                        sentence[:240],
                        0.7,
                    )


def infer_relation_type(sentence: str, source: Entity, target: Entity) -> str:
    # 有明确触发词时优先使用触发词；否则根据实体类型给出弱关系。
    source_index = find_first_name_index(sentence, source)
    target_index = find_first_name_index(sentence, target)
    if source_index < 0 or target_index < 0 or source_index == target_index:
        return ""

    left_index, right_index = sorted([source_index, target_index])
    between = sentence[left_index:right_index]
    for keyword, relation_type in METHOD_RELATION_KEYWORDS:
        if keyword in between:
            return relation_type

    if source.type == "Library" and target.type == "API":
        return "PROVIDES"
    if source.type == "API" and target.type in {"DataStructure", "MathObject"}:
        return "OPERATES_ON"
    if source.type in {"Algorithm", "Method", "Model"} and target.type == "Task":
        return "SOLVES"
    if source.type in {"Algorithm", "Method", "Model"} and target.type in {"DataStructure", "MathObject"}:
        return "USES"
    if source.type in {"Concept", "Method", "Model"} and target.type in {"Concept", "Method", "Model"}:
        return "RELATED_TO"
    return ""


def find_first_name_index(sentence: str, entity: Entity) -> int:
    candidates = (entity.name, *entity.aliases)
    indexes = [sentence.lower().find(candidate.lower()) for candidate in candidates if candidate]
    indexes = [index for index in indexes if index >= 0]
    return min(indexes) if indexes else -1


def merge_extractions(extractions: Iterable[TripleExtraction]) -> TripleExtraction:
    # hybrid 模式和全局导出都走这里，避免同一实体在多个 chunk 中重复散落。
    entities: dict[tuple[str, str], Entity] = {}
    relationships: dict[tuple[str, str, str, str, str, str], Relationship] = {}

    for extraction in extractions:
        for entity in extraction.entities:
            merged = add_entity(
                entities,
                entity.name,
                entity.type,
                "",
                aliases=entity.aliases,
                description=entity.description,
            )
            for chunk_id in entity.source_chunk_ids:
                if chunk_id and chunk_id not in merged.source_chunk_ids:
                    merged.source_chunk_ids.append(chunk_id)
        for relationship in extraction.relationships:
            relationships.setdefault(relationship_key(relationship), relationship)

    sorted_entities = sorted(entities.values(), key=lambda item: (item.type, item.name))
    sorted_relationships = sorted(
        relationships.values(),
        key=lambda item: (item.source, item.type, item.target, item.source_chunk_id),
    )
    annotate_relationship_types(sorted_entities, sorted_relationships)
    return TripleExtraction(entities=sorted_entities, relationships=sorted_relationships)


def extract_with_llm_from_chunk(
    chunk,
    model_name: str,
    openai_api_key: str = "",
    openai_base_url: str = "",
    confidence_threshold: float = 0.65,
) -> TripleExtraction:
    # 在线和离线 LLM 模式复用同一 Prompt、schema 和证据校验。
    from kg.extractor import extract_knowledge_graph

    chunk_id = get_chunk_id(chunk)
    response = extract_knowledge_graph(
        text=chunk.page_content,
        model_name=model_name,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        chunk_id=chunk_id,
        confidence_threshold=confidence_threshold,
        book=get_book_title(chunk),
        section=" > ".join(get_section_names(chunk)),
    )
    return TripleExtraction(
        entities=[Entity(**entity.model_dump()) for entity in response.entities],
        relationships=[
            Relationship(**relationship.model_dump())
            for relationship in response.relationships
        ],
    )


def load_and_split_documents(
    data_dir: Path,
    openai_api_key: str = "",
    openai_base_url: str = "",
    ocr_model: str = "gpt-4o-mini",
    ocr_max_pages: int | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    # 依赖齐全时优先复用项目已有 loader/splitter，保证和 RAG 索引的 chunk_id 一致。
    try:
        from rag.loaders import load_documents
        from rag.splitter import split_documents

        documents = load_documents(
            data_dir,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            ocr_model=ocr_model,
            ocr_max_pages=ocr_max_pages,
        )
        return split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ModuleNotFoundError as exc:
        # 当前环境缺少 PyMuPDF/langchain-community 时，退回 pdftotext 简易加载器。
        print(
            f"Project document loader dependency is missing ({exc.name}); "
            "using the built-in fallback loader.",
            file=sys.stderr,
        )
        return fallback_load_and_split_documents(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def fallback_load_and_split_documents(
    data_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[SimpleDocument]:
    # 备用加载器只保留最关键的 metadata，足够支撑实体关系抽取和来源追踪。
    documents: list[SimpleDocument] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(fallback_load_pdf(path))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                SimpleDocument(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": 0,
                        "page_label": "1",
                        "title": path.stem,
                    },
                )
            )
    return fallback_split_documents(documents, chunk_size, chunk_overlap)


def fallback_load_pdf(path: Path) -> list[SimpleDocument]:
    # pdftotext 已在很多 macOS/Linux 环境中可用，可以让脚本在未安装依赖时先跑通。
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "PDF extraction requires either project dependencies "
            "(PyMuPDF/langchain-community) or the pdftotext command."
        ) from exc

    pages = result.stdout.split("\f")
    documents: list[SimpleDocument] = []
    # pdftotext 用换页符分隔页面，这里把每页先作为一个 SimpleDocument。
    for page_index, page_text in enumerate(pages):
        page_text = normalize_pdf_text(page_text)
        if not page_text.strip():
            continue
        documents.append(
            SimpleDocument(
                page_content=page_text,
                metadata={
                    "source": str(path),
                    "page": page_index,
                    "page_label": str(page_index + 1),
                    "title": path.stem,
                    "loader": "pdftotext_fallback",
                },
            )
        )
    return documents


def normalize_pdf_text(text: str) -> str:
    # 清理 PDF 抽取产生的行尾空格和多余空行，保留基本段落边界。
    lines = [line.rstrip() for line in text.splitlines()]
    normalized: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if normalized and normalized[-1]:
                normalized.append("")
            continue
        normalized.append(stripped)
    return "\n".join(normalized).strip()


def fallback_split_documents(
    documents: list[SimpleDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[SimpleDocument]:
    # 简单滑窗切分：chunk_overlap 让相邻块保留一点上下文，降低实体被切断的概率。
    chunks: list[SimpleDocument] = []
    counts_by_doc_id: dict[str, int] = {}
    step = max(1, chunk_size - chunk_overlap)

    for document in documents:
        source = str(document.metadata.get("source") or "unknown")
        doc_id = build_doc_id(source)
        title = str(document.metadata.get("title") or Path(source).stem or source)
        text = document.page_content.strip()
        if not text:
            continue

        starts = list(range(0, len(text), step)) or [0]
        for start in starts:
            chunk_text = text[start : start + chunk_size].strip()
            if not chunk_text:
                continue
            counts_by_doc_id[doc_id] = counts_by_doc_id.get(doc_id, 0) + 1
            chunk_id = f"{doc_id}#chunk_{counts_by_doc_id[doc_id]:03d}"
            # metadata 字段尽量对齐 rag.splitter，方便后续替换回项目正式加载器。
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "source": source,
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": start + len(chunk_text),
                    "section_path": infer_section_path_from_text(chunk_text),
                }
            )
            chunks.append(SimpleDocument(page_content=chunk_text, metadata=metadata))
            if start + chunk_size >= len(text):
                break
    return chunks


def build_doc_id(source: str) -> str:
    # doc_id 使用文件名 + 路径 hash，既可读又能避免同名文件冲突。
    stem = Path(source).stem or "document"
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", stem).strip("_").lower()
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'document'}_{digest}"


def infer_section_path_from_text(text: str) -> str:
    # 备用 splitter 没有版面信息，只能从 Markdown/章节编号样式粗略推断章节路径。
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
        elif re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节篇部分]", stripped):
            headings.append(stripped)
        elif re.match(r"^[0-9]+([.．][0-9]+)*[、.．\s]", stripped) and len(stripped) <= 80:
            headings.append(stripped)
    return " > ".join(headings[:3])


def write_json_outputs(
    output_dir: Path,
    chunk_extractions: list[tuple[str, TripleExtraction]],
    merged: TripleExtraction,
) -> None:
    # 同时输出按 chunk 的 JSONL 和全局合并后的 JSON，分别服务调试和图谱入库。
    with staged_output_dir(output_dir) as staged:
        with (staged / "triples_by_chunk.jsonl").open("w", encoding="utf-8") as file:
            for chunk_id, extraction in chunk_extractions:
                row = {"chunk_id": chunk_id, **as_jsonable(extraction)}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        with (staged / "knowledge_graph.json").open("w", encoding="utf-8") as file:
            json.dump(as_jsonable(merged), file, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    # CLI 默认使用 rules 模式，不需要 API Key；需要更高质量时可切到 llm/hybrid。
    parser = argparse.ArgumentParser(
        description="Extract entities and relationships from local technical books."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/kg"))
    parser.add_argument(
        "--mode",
        choices=("rules", "llm", "hybrid"),
        default="rules",
        help="rules is offline; llm requires an OpenAI-compatible API; hybrid merges both.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--ocr-model", default="gpt-4o-mini")
    parser.add_argument("--ocr-max-pages", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N chunks.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("chunk size must be positive and overlap smaller than chunk size")
    if args.max_chunks < 0 or args.progress_every <= 0:
        parser.error("max-chunks must be non-negative and progress-every must be positive")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("confidence-threshold must be between 0 and 1")
    if args.ocr_max_pages is not None and args.ocr_max_pages <= 0:
        parser.error("ocr-max-pages must be positive")

    # LLM/hybrid 模式没有显式传参时，沿用项目 .env 中的模型和 API 配置。
    if args.mode in {"llm", "hybrid"}:
        from app.config import get_settings

        settings = get_settings()
        args.model = args.model or settings.llm_model
        args.openai_api_key = args.openai_api_key or settings.openai_api_key
        args.openai_base_url = args.openai_base_url or settings.openai_base_url
        if not args.openai_api_key:
            parser.error("LLM/hybrid mode requires an OpenAI-compatible API key")

    chunks = load_and_split_documents(
        data_dir=args.data_dir,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        ocr_model=args.ocr_model,
        ocr_max_pages=args.ocr_max_pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        parser.error(f"no indexable documents found in {args.data_dir}")

    chunk_extractions: list[tuple[str, TripleExtraction]] = []
    # 每个 chunk 独立抽取，失败或质量问题更容易定位到具体来源块。
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = get_chunk_id(chunk)
        extractions: list[TripleExtraction] = []
        if args.mode in {"rules", "hybrid"}:
            extractions.append(extract_rule_based_from_chunk(chunk))
        if args.mode in {"llm", "hybrid"}:
            extractions.append(
                extract_with_llm_from_chunk(
                    chunk,
                    model_name=args.model,
                    openai_api_key=args.openai_api_key,
                    openai_base_url=args.openai_base_url,
                    confidence_threshold=args.confidence_threshold,
                )
            )
        extraction = merge_extractions(extractions)
        chunk_extractions.append((chunk_id, extraction))
        if index == 1 or index == len(chunks) or index % args.progress_every == 0:
            print(
                f"[{index}/{len(chunks)}] {chunk_id}: "
                f"{len(extraction.entities)} entities, "
                f"{len(extraction.relationships)} relationships"
            )

    merged = merge_extractions(extraction for _, extraction in chunk_extractions)
    write_json_outputs(args.output_dir, chunk_extractions, merged)
    print(f"Wrote {args.output_dir / 'triples_by_chunk.jsonl'}")
    print(f"Wrote {args.output_dir / 'knowledge_graph.json'}")
    print(
        f"Total: {len(merged.entities)} entities, "
        f"{len(merged.relationships)} relationships"
    )


if __name__ == "__main__":
    main()
