from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg.records import GraphEntity as Entity
from kg.records import GraphRelationship as Relationship
from kg.records import entity_from_dict, relationship_from_dict
from scripts.kg.io import staged_output_dir

# 可互相对齐的实体类型。默认不跨大类合并，避免把“图”(数据结构)和“图”(图片/章节)误并。
COMPATIBLE_TYPE_GROUPS = [
    {"Concept", "Method", "Model"},
    {"Algorithm"},
    {"Task", "Application"},
    {"DataStructure", "MathObject"},
    {"Library"},
    {"API"},
    {"Metric"},
    {"Book"},
    {"Chapter", "Section"},
    {"Person"},
]


# 人工维护的同义词词典。key 会先标准化，所以英文大小写和空格差异不会影响匹配。
MANUAL_ALIASES = {
    "llm": "大语言模型",
    "large language model": "大语言模型",
    "大型语言模型": "大语言模型",
    "大规模语言模型": "大语言模型",
    "transformer模型": "Transformer",
    "transformer 架构": "Transformer",
    "attention": "注意力机制",
    "self-attention": "注意力机制",
    "self attention": "注意力机制",
    "自注意力": "注意力机制",
    "pre-training": "预训练",
    "pre training": "预训练",
    "预训练语言模型": "预训练",
    "fine-tuning": "微调",
    "fine tuning": "微调",
    "监督微调": "微调",
    "sft": "微调",
    "instruction tuning": "指令微调",
    "rl": "强化学习",
    "reinforcement learning": "强化学习",
    "rlhf": "人类反馈强化学习",
    "prompt engineering": "提示工程",
    "prompt": "提示工程",
    "提示词": "提示工程",
    "的提示词": "提示工程",
    "rag": "检索增强生成",
    "retrieval-augmented generation": "检索增强生成",
    "retrieval augmented generation": "检索增强生成",
    "embedding": "词嵌入",
    "word embedding": "词嵌入",
    "词向量": "词嵌入",
    "tokenization": "Token",
    "标记": "Token",
    "词元": "Token",
    "qa": "问答系统",
    "question answering": "问答系统",
    "ner": "命名实体识别",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "data frame": "DataFrame",
    "dataframe": "DataFrame",
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1值": "F1",
    "f1-score": "F1",
    "perplexity": "困惑度",
    "binary search": "二分查找",
    "recursion": "递归",
    "quicksort": "快速排序",
    "快排": "快速排序",
    "merge sort": "归并排序",
    "bfs": "广度优先搜索",
    "dfs": "深度优先搜索",
    "dijkstra": "Dijkstra算法",
    "狄克斯特拉算法": "Dijkstra算法",
    "dp": "动态规划",
    "greedy algorithm": "贪心算法",
    "knn": "K近邻算法",
    "knn算法": "K近邻算法",
    "k近邻": "K近邻算法",
    "k近邻算法": "K近邻算法",
    "k最近邻算法": "K近邻算法",
    "clustering": "聚类",
    "hash table": "散列表",
    "哈希表": "散列表",
}


FUNCTION_WORD_PREFIXES = (
    "的",
    "该",
    "此",
    "这些",
    "那些",
    "并",
    "和",
    "或",
    "以及",
    "因此",
    "所以",
)


FRAGMENT_SUFFIXES = (
    "的",
    "就",
    "还",
    "是否",
    "真的",
    "通常",
)


SENTENCE_FRAGMENT_MARKERS = (
    "可以",
    "必须",
    "需要",
    "找到",
    "判断",
    "采用",
    "得到",
    "如下",
    "各种",
    "上述",
    "下面",
    "真的",
    "是否",
    "效果评测",
)


@dataclass
class AlignmentDecision:
    original_name: str
    original_type: str
    canonical_name: str
    canonical_type: str
    action: str
    reason: str
    score: float = 1.0


@dataclass
class AlignmentResult:
    entities: list[Entity]
    relationships: list[Relationship]
    alignment_map: list[AlignmentDecision]
    removed_entities: list[dict[str, Any]]
    ambiguity_report: dict[str, Any]


def normalize_text(value: str) -> str:
    value = value or ""
    value = value.replace("（", "(").replace("）", ")")
    value = value.replace("，", ",").replace("：", ":")
    value = re.sub(r"\[[0-9,\-\s]+\]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(" \t\n\r,，.;；:：、()（）[]【】")


def normalize_key(value: str) -> str:
    normalized = normalize_text(value).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[\u3000_]+", " ", normalized)
    normalized = normalized.replace("－", "-").replace("–", "-")
    return normalized.strip()


def load_graph(path: Path) -> tuple[list[Entity], list[Relationship]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entities = [entity_from_dict(item) for item in data.get("entities", [])]
    relationships = [
        relationship_from_dict(item) for item in data.get("relationships", [])
    ]
    return entities, relationships


def get_type_group(entity_type: str) -> str:
    for index, group in enumerate(COMPATIBLE_TYPE_GROUPS):
        if entity_type in group:
            return f"group_{index}"
    return entity_type


def invalid_entity_reason(entity: Entity, max_name_chars: int) -> str:
    name = normalize_text(entity.name)
    compact = re.sub(r"\s+", "", name)
    if not compact:
        return "empty_name"
    if len(name) > max_name_chars and entity.type not in {"Book"}:
        return "name_too_long"
    if entity.type in {"Chapter", "Section"} and len(name) > 80:
        return "section_like_sentence"
    if name.startswith(FUNCTION_WORD_PREFIXES) and len(name) < 12:
        return "fragment_starts_with_function_word"
    if entity.type in {"Algorithm", "Concept", "Method", "Task"} and name.endswith(FRAGMENT_SUFFIXES):
        return "fragment_ends_with_function_word"
    if re.match(r"^[0-9]+[a-zA-Z\u4e00-\u9fff]", name) and entity.type != "Book":
        return "likely_page_or_table_fragment"
    if re.search(r"[。！？!?；;]", name):
        return "sentence_punctuation_in_name"
    if len(compact) >= 18 and re.search(r"(可以|必须|需要|用于|通过|基于|进行|得到|提出|表示|包含)", compact):
        return "sentence_like_name"
    if entity.type in {"Algorithm", "Concept", "Method", "Task"}:
        marker_count = sum(marker in compact for marker in SENTENCE_FRAGMENT_MARKERS)
        if marker_count >= 2:
            return "likely_sentence_fragment"
    if entity.type in {"Algorithm", "Concept", "Method", "Task"} and len(compact) <= 1:
        return "too_short"
    return ""


def canonical_name_for(entity: Entity) -> tuple[str, str]:
    name = normalize_text(entity.name)
    key = normalize_key(name)

    # Book 实体不使用概念同义词表，否则《大语言模型》这本书会和“大语言模型”概念混在一起。
    if entity.type != "Book" and key in MANUAL_ALIASES:
        return MANUAL_ALIASES[key], "manual_alias"

    for alias in entity.aliases:
        alias_key = normalize_key(alias)
        if entity.type != "Book" and alias_key in MANUAL_ALIASES:
            return MANUAL_ALIASES[alias_key], "manual_alias_from_entity_alias"

    return name, "normalized_exact"


def merge_entity(target: Entity, source: Entity) -> None:
    for alias in [source.name, *source.aliases]:
        alias = normalize_text(alias)
        if alias and alias != target.name and alias not in target.aliases:
            target.aliases.append(alias)
    for chunk_id in source.source_chunk_ids:
        if chunk_id and chunk_id not in target.source_chunk_ids:
            target.source_chunk_ids.append(chunk_id)
    if source.description and not target.description:
        target.description = source.description


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_key(left), normalize_key(right)).ratio()


def align_entities(
    entities: list[Entity],
    max_name_chars: int,
    fuzzy_threshold: float,
) -> tuple[list[Entity], list[AlignmentDecision], list[dict[str, Any]]]:
    canonical_entities: dict[tuple[str, str], Entity] = {}
    alignment_map: list[AlignmentDecision] = []
    removed_entities: list[dict[str, Any]] = []

    for entity in entities:
        reason = invalid_entity_reason(entity, max_name_chars=max_name_chars)
        if reason:
            removed_entities.append({"entity": asdict(entity), "reason": reason})
            alignment_map.append(
                AlignmentDecision(
                    original_name=entity.name,
                    original_type=entity.type,
                    canonical_name="",
                    canonical_type="",
                    action="drop",
                    reason=reason,
                    score=0.0,
                )
            )
            continue

        canonical_name, reason = canonical_name_for(entity)
        key = (entity.type, normalize_key(canonical_name))
        existing = canonical_entities.get(key)
        if existing is None:
            existing = Entity(
                name=canonical_name,
                type=entity.type,
                aliases=[],
                description=entity.description,
                source_chunk_ids=[],
            )
            canonical_entities[key] = existing
        merge_entity(existing, entity)
        alignment_map.append(
            AlignmentDecision(
                original_name=entity.name,
                original_type=entity.type,
                canonical_name=existing.name,
                canonical_type=existing.type,
                action="merge" if entity.name != existing.name else "keep",
                reason=reason,
                score=1.0,
            )
        )

    fuzzy_merge_entities(canonical_entities, alignment_map, fuzzy_threshold)
    return (
        sorted(canonical_entities.values(), key=lambda item: (item.type, item.name)),
        alignment_map,
        removed_entities,
    )


def fuzzy_merge_entities(
    canonical_entities: dict[tuple[str, str], Entity],
    alignment_map: list[AlignmentDecision],
    fuzzy_threshold: float,
) -> None:
    by_group: dict[str, list[tuple[tuple[str, str], Entity]]] = defaultdict(list)
    for key, entity in canonical_entities.items():
        by_group[get_type_group(entity.type)].append((key, entity))

    for items in by_group.values():
        # 高阈值模糊合并只比较同类型、首字符相同、长度接近的候选，避免全量 O(n^2)。
        buckets: dict[tuple[str, str], list[tuple[tuple[str, str], Entity]]] = defaultdict(list)
        for key, entity in items:
            entity_key = normalize_key(entity.name)
            first_char = entity_key[:1]
            buckets[(entity.type, first_char)].append((key, entity))

        removed_keys: set[tuple[str, str]] = set()

        for bucket_items in buckets.values():
            bucket_items.sort(key=lambda item: (len(normalize_key(item[1].name)), item[1].name))
            for index, (left_key, left_entity) in enumerate(bucket_items):
                if left_key in removed_keys:
                    continue
                left_len = len(normalize_key(left_entity.name))
                for right_key, right_entity in bucket_items[index + 1 :]:
                    if right_key in removed_keys:
                        continue
                    right_len = len(normalize_key(right_entity.name))
                    if right_len - left_len > max_length_delta(left_len, fuzzy_threshold):
                        break
                    score = similarity(left_entity.name, right_entity.name)
                    if score < fuzzy_threshold:
                        continue
                    if should_skip_fuzzy_merge(left_entity.name, right_entity.name):
                        continue
                    merge_entity(left_entity, right_entity)
                    removed_keys.add(right_key)
                    canonical_entities.pop(right_key, None)
                    alignment_map.append(
                        AlignmentDecision(
                            original_name=right_entity.name,
                            original_type=right_entity.type,
                            canonical_name=left_entity.name,
                            canonical_type=left_entity.type,
                            action="merge",
                            reason="fuzzy_same_type",
                            score=round(score, 4),
                        )
                    )


def max_length_delta(length: int, fuzzy_threshold: float) -> int:
    # SequenceMatcher 在长度差很大时不可能达到高相似度，提前剪枝。
    return max(1, int(length * max(0.02, 1.0 - fuzzy_threshold + 0.02)))


def should_skip_fuzzy_merge(left: str, right: str) -> bool:
    left_key = normalize_key(left)
    right_key = normalize_key(right)
    if left_key == right_key:
        return False
    # “语言模型”和“大语言模型”这种包含关系不一定是同义词，交给人工报告处理。
    if left_key in right_key or right_key in left_key:
        return True
    return False


def build_name_type_to_canonical(
    decisions: list[AlignmentDecision],
) -> dict[tuple[str, str], tuple[str, str]]:
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    for decision in decisions:
        if decision.action == "drop":
            continue
        mapping[(decision.original_type, normalize_key(decision.original_name))] = (
            decision.canonical_name,
            decision.canonical_type,
        )
        mapping[(decision.canonical_type, normalize_key(decision.canonical_name))] = (
            decision.canonical_name,
            decision.canonical_type,
        )
    return mapping


def resolve_relationship_endpoint(
    name: str,
    original_entities_by_name: dict[str, list[Entity]],
    mapping: dict[tuple[str, str], tuple[str, str]],
    entity_type: str = "",
) -> tuple[str, str]:
    candidates = original_entities_by_name.get(normalize_key(name), [])
    if entity_type:
        candidates = [entity for entity in candidates if entity.type == entity_type]
    resolved: set[tuple[str, str]] = set()
    for entity in candidates:
        key = (entity.type, normalize_key(name))
        if key in mapping:
            resolved.add(mapping[key])
    if len(resolved) == 1:
        return resolved.pop()
    # ponytail: name-only relationships cannot disambiguate cross-type homonyms;
    # drop them now and upgrade the extraction schema with endpoint types if needed.
    return "", ""


def align_relationships(
    relationships: list[Relationship],
    original_entities: list[Entity],
    decisions: list[AlignmentDecision],
    min_confidence: float,
) -> tuple[list[Relationship], list[dict[str, Any]]]:
    mapping = build_name_type_to_canonical(decisions)
    original_entities_by_name: dict[str, list[Entity]] = defaultdict(list)
    for entity in original_entities:
        original_entities_by_name[normalize_key(entity.name)].append(entity)

    aligned: dict[tuple[str, str, str, str, str, str], Relationship] = {}
    dropped: list[dict[str, Any]] = []

    for relationship in relationships:
        if relationship.confidence < min_confidence:
            dropped.append({"relationship": asdict(relationship), "reason": "low_confidence"})
            continue
        target, target_type = resolve_relationship_endpoint(
            relationship.target,
            original_entities_by_name,
            mapping,
            relationship.target_type,
        )
        source, source_type = resolve_relationship_endpoint(
            relationship.source,
            original_entities_by_name,
            mapping,
            relationship.source_type,
        )
        if not source or not target:
            dropped.append(
                {"relationship": asdict(relationship), "reason": "endpoint_removed_or_unknown"}
            )
            continue
        if source == target:
            dropped.append({"relationship": asdict(relationship), "reason": "self_loop_after_alignment"})
            continue

        key = (
            source,
            source_type,
            relationship.type,
            target,
            target_type,
            relationship.source_chunk_id,
        )
        current = aligned.get(key)
        candidate = Relationship(
            source=source,
            target=target,
            type=relationship.type,
            evidence=relationship.evidence,
            source_chunk_id=relationship.source_chunk_id,
            confidence=relationship.confidence,
            source_type=source_type,
            target_type=target_type,
        )
        if current is None or candidate.confidence > current.confidence:
            aligned[key] = candidate

    return (
        sorted(aligned.values(), key=lambda item: (item.source, item.type, item.target)),
        dropped,
    )


def build_ambiguity_report(
    entities: list[Entity],
    aligned_entities: list[Entity],
    decisions: list[AlignmentDecision],
    removed_entities: list[dict[str, Any]],
    dropped_relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    surface_to_types: dict[str, set[str]] = defaultdict(set)
    for entity in aligned_entities:
        surface_to_types[normalize_key(entity.name)].add(entity.type)
        for alias in entity.aliases:
            surface_to_types[normalize_key(alias)].add(entity.type)

    ambiguous_surfaces = [
        {"surface": surface, "types": sorted(types)}
        for surface, types in surface_to_types.items()
        if surface and len(types) > 1
    ]

    merge_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        if decision.action == "drop":
            continue
        merge_groups[(decision.canonical_type, decision.canonical_name)].append(asdict(decision))

    merged_groups = [
        {
            "canonical_type": canonical_type,
            "canonical_name": canonical_name,
            "members": members,
        }
        for (canonical_type, canonical_name), members in merge_groups.items()
        if len({member["original_name"] for member in members}) > 1
    ]

    return {
        "summary": {
            "input_entities": len(entities),
            "aligned_entities": len(aligned_entities),
            "removed_entities": len(removed_entities),
            "merged_groups": len(merged_groups),
            "ambiguous_surface_names": len(ambiguous_surfaces),
            "dropped_relationships": len(dropped_relationships),
        },
        "ambiguous_surface_names": sorted(ambiguous_surfaces, key=lambda item: item["surface"]),
        "merged_groups": sorted(
            merged_groups,
            key=lambda item: (item["canonical_type"], item["canonical_name"]),
        ),
        "removed_entities": removed_entities,
        "dropped_relationships": dropped_relationships[:500],
    }


def align_graph(
    entities: list[Entity],
    relationships: list[Relationship],
    max_name_chars: int,
    fuzzy_threshold: float,
    min_confidence: float,
) -> AlignmentResult:
    aligned_entities, decisions, removed_entities = align_entities(
        entities,
        max_name_chars=max_name_chars,
        fuzzy_threshold=fuzzy_threshold,
    )
    aligned_relationships, dropped_relationships = align_relationships(
        relationships,
        original_entities=entities,
        decisions=decisions,
        min_confidence=min_confidence,
    )
    report = build_ambiguity_report(
        entities=entities,
        aligned_entities=aligned_entities,
        decisions=decisions,
        removed_entities=removed_entities,
        dropped_relationships=dropped_relationships,
    )
    return AlignmentResult(
        entities=aligned_entities,
        relationships=aligned_relationships,
        alignment_map=decisions,
        removed_entities=removed_entities,
        ambiguity_report=report,
    )


def write_outputs(result: AlignmentResult, output_dir: Path) -> None:
    with staged_output_dir(output_dir) as staged:
        aligned_graph = {
            "entities": [asdict(entity) for entity in result.entities],
            "relationships": [asdict(relationship) for relationship in result.relationships],
        }
        (staged / "aligned_knowledge_graph.json").write_text(
            json.dumps(aligned_graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (staged / "entity_alignment_map.json").write_text(
            json.dumps(
                [asdict(item) for item in result.alignment_map],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (staged / "ambiguity_report.json").write_text(
            json.dumps(result.ambiguity_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disambiguate and align entities extracted from the KG triple stage."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/kg/knowledge_graph.json"),
        help="Input knowledge_graph.json from 01_triple.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kg_aligned"),
        help="Directory for aligned graph, alignment map, and ambiguity report.",
    )
    parser.add_argument(
        "--max-name-chars",
        type=int,
        default=60,
        help="Drop non-book entities whose names are longer than this threshold.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.96,
        help="Same-type fuzzy merge threshold. Keep high to avoid semantic over-merge.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Drop relationships below this confidence after endpoint alignment.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if args.max_name_chars <= 0:
        parser.error("max-name-chars must be positive")
    if not 0 <= args.fuzzy_threshold <= 1 or not 0 <= args.min_confidence <= 1:
        parser.error("thresholds must be between 0 and 1")
    entities, relationships = load_graph(args.input)
    result = align_graph(
        entities=entities,
        relationships=relationships,
        max_name_chars=args.max_name_chars,
        fuzzy_threshold=args.fuzzy_threshold,
        min_confidence=args.min_confidence,
    )
    write_outputs(result, args.output_dir)

    summary = result.ambiguity_report["summary"]
    print(f"Input entities: {summary['input_entities']}")
    print(f"Aligned entities: {summary['aligned_entities']}")
    print(f"Removed entities: {summary['removed_entities']}")
    print(f"Merged groups: {summary['merged_groups']}")
    print(f"Ambiguous surface names: {summary['ambiguous_surface_names']}")
    print(f"Dropped relationships: {summary['dropped_relationships']}")
    print(f"Wrote {args.output_dir / 'aligned_knowledge_graph.json'}")
    print(f"Wrote {args.output_dir / 'entity_alignment_map.json'}")
    print(f"Wrote {args.output_dir / 'ambiguity_report.json'}")


if __name__ == "__main__":
    main()
