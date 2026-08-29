from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg.records import GraphEntity as Entity
from kg.records import GraphRelationship as Relationship
from kg.records import GraphTriple as Triple
from kg.records import entity_from_dict, relationship_from_dict
from kg.schema import PREDICATE_LABELS
from scripts.kg.io import staged_output_dir


CODE_FRAGMENT_PATTERNS = (
    r"\bdef\b",
    r"\bclass\b",
    r"\breturn\b",
    r"\bself\b",
    r"\bfor\b",
    r"\bif\b",
    r"\belse\b",
    r"\bimport\b",
    r"\bcontinue\b",
    r"\bfunction\b",
    r"\bpredict\b",
    r"\bsample\b",
    r"\bmask\b",
    r"\bbreak\b",
    r"\brange\b",
    r"\blen\b",
    r"\.py\b",
    r"__name__",
    r"input_data",
    r"tokenizer",
    r"model\(",
    r"logging\.",
    r"opt\.",
)


LOW_VALUE_CODE_NAMES = {
    "len",
    "range",
    "get_tokenizer",
}


def normalize_text(value: str) -> str:
    value = value or ""
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^#+\s*", "", value)
    return value.strip(" \t\n\r,，.;；:：、")


def normalize_key(value: str) -> str:
    return normalize_text(value).lower()


def load_aligned_graph(path: Path) -> tuple[list[Entity], list[Relationship]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entities = [entity_from_dict(item) for item in data.get("entities", [])]
    relationships = [
        relationship_from_dict(item) for item in data.get("relationships", [])
    ]
    return entities, relationships


def build_entity_index(entities: Iterable[Entity]) -> dict[str, Entity]:
    index: dict[str, Entity] = {}
    ambiguous: set[str] = set()
    for entity in entities:
        name = normalize_text(entity.name)
        if not name:
            continue
        key = normalize_key(name)
        if key in index and index[key].type != entity.type:
            ambiguous.add(key)
            index.pop(key, None)
        elif key not in ambiguous:
            index[key] = entity
    return index


def parse_relation_type_filter(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    relation_types: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip().upper()
            if item:
                relation_types.add(item)
    return relation_types


def should_drop_endpoint_name(name: str, max_endpoint_chars: int) -> bool:
    name = normalize_text(name)
    if not name:
        return True
    compact = re.sub(r"\s+", "", name)
    if len(compact) <= 1:
        return True
    if re.fullmatch(r"[#*.\-_=+~`|\\/]+", compact):
        return True
    if re.fullmatch(r"#?[A-Za-z]?", compact):
        return True
    if re.fullmatch(r"#?[\d.]+", compact):
        return True
    if name.startswith("#") and len(compact) <= 4:
        return True
    if normalize_key(name) in LOW_VALUE_CODE_NAMES:
        return True
    if re.search(r"[{}]", name):
        return True
    if "_" in name and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return True
    if is_mostly_english_phrase(name):
        return True
    if is_code_fragment_name(name):
        return True
    if len(name) > max_endpoint_chars:
        return True
    # 公式碎片、整句碎片不适合作为最终三元组的主语或宾语。
    if re.search(r"[。！？!?；;]", name):
        return True
    if re.search(r"[=＋+×*/÷]|Precision\)|Recall", name, flags=re.IGNORECASE):
        return True
    return False


def is_code_fragment_name(name: str) -> bool:
    hits = sum(
        re.search(pattern, name, flags=re.IGNORECASE) is not None
        for pattern in CODE_FRAGMENT_PATTERNS
    )
    if hits >= 2:
        return True
    if name.startswith("#") and hits >= 1:
        return True
    if re.match(r"^(def|class|for|if|return)\b", name, flags=re.IGNORECASE):
        return True
    return False


def is_mostly_english_phrase(name: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", name):
        return False
    words = re.findall(r"[A-Za-z]+", name)
    if len(words) < 4:
        return False
    ascii_chars = sum(1 for char in name if ord(char) < 128)
    return ascii_chars / max(1, len(name)) > 0.85


def make_triple_id(
    subject: str,
    predicate: str,
    object_: str,
    source_chunk_id: str,
) -> str:
    raw = "\u241f".join([subject, predicate, object_, source_chunk_id])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"triple_{digest}"


def create_triples(
    entities: list[Entity],
    relationships: list[Relationship],
    min_confidence: float,
    include_relation_types: set[str],
    exclude_relation_types: set[str],
    max_endpoint_chars: int,
    strict_endpoints: bool,
) -> tuple[list[Triple], list[dict[str, Any]]]:
    entity_index = build_entity_index(entities)
    typed_entity_index = {
        (entity.type, normalize_key(entity.name)): entity for entity in entities
    }
    triples: dict[tuple[str, str, str, str, str, str], Triple] = {}
    dropped: list[dict[str, Any]] = []

    for relationship in relationships:
        relation_type = normalize_text(relationship.type).upper()
        if include_relation_types and relation_type not in include_relation_types:
            dropped.append({"relationship": asdict(relationship), "reason": "not_in_include_filter"})
            continue
        if relation_type in exclude_relation_types:
            dropped.append({"relationship": asdict(relationship), "reason": "excluded_relation_type"})
            continue
        if relationship.confidence < min_confidence:
            dropped.append({"relationship": asdict(relationship), "reason": "low_confidence"})
            continue

        subject = normalize_text(relationship.source)
        object_ = normalize_text(relationship.target)
        if should_drop_endpoint_name(subject, max_endpoint_chars) or should_drop_endpoint_name(
            object_, max_endpoint_chars
        ):
            dropped.append({"relationship": asdict(relationship), "reason": "bad_endpoint_name"})
            continue
        if subject == object_:
            dropped.append({"relationship": asdict(relationship), "reason": "self_loop"})
            continue

        subject_entity = (
            typed_entity_index.get((relationship.source_type, normalize_key(subject)))
            if relationship.source_type
            else entity_index.get(normalize_key(subject))
        )
        object_entity = (
            typed_entity_index.get((relationship.target_type, normalize_key(object_)))
            if relationship.target_type
            else entity_index.get(normalize_key(object_))
        )
        if strict_endpoints and (subject_entity is None or object_entity is None):
            dropped.append({"relationship": asdict(relationship), "reason": "endpoint_not_in_entity_index"})
            continue

        subject_type = subject_entity.type if subject_entity else ""
        object_type = object_entity.type if object_entity else ""
        key = (
            subject,
            subject_type,
            relation_type,
            object_,
            object_type,
            relationship.source_chunk_id,
        )
        triple = Triple(
            triple_id=make_triple_id(
                subject=subject,
                predicate=relation_type,
                object_=object_,
                source_chunk_id=relationship.source_chunk_id,
            ),
            subject=subject,
            predicate=relation_type,
            object=object_,
            predicate_label=PREDICATE_LABELS.get(relation_type, relation_type),
            subject_type=subject_type,
            object_type=object_type,
            evidence=normalize_text(relationship.evidence),
            source_chunk_id=relationship.source_chunk_id,
            confidence=relationship.confidence,
        )

        current = triples.get(key)
        if current is None or triple.confidence > current.confidence:
            triples[key] = triple

    return (
        sorted(triples.values(), key=lambda item: (item.subject, item.predicate, item.object)),
        dropped,
    )


def write_jsonl(path: Path, triples: Iterable[Triple]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for triple in triples:
            file.write(json.dumps(asdict(triple), ensure_ascii=False) + "\n")


def write_csv(path: Path, triples: list[Triple]) -> None:
    fieldnames = [
        "triple_id",
        "subject",
        "predicate",
        "object",
        "predicate_label",
        "subject_type",
        "object_type",
        "evidence",
        "source_chunk_id",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for triple in triples:
            writer.writerow(asdict(triple))


def write_txt(path: Path, triples: Iterable[Triple]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for triple in triples:
            file.write(f"({triple.subject}, {triple.predicate_label}, {triple.object})\n")


def write_summary(
    path: Path,
    input_entities: int,
    input_relationships: int,
    triples: list[Triple],
    dropped: list[dict[str, Any]],
) -> None:
    by_predicate: dict[str, int] = {}
    by_subject_type: dict[str, int] = {}
    for triple in triples:
        by_predicate[triple.predicate] = by_predicate.get(triple.predicate, 0) + 1
        by_subject_type[triple.subject_type] = by_subject_type.get(triple.subject_type, 0) + 1

    summary = {
        "input_entities": input_entities,
        "input_relationships": input_relationships,
        "created_triples": len(triples),
        "dropped_relationships": len(dropped),
        "triples_by_predicate": dict(sorted(by_predicate.items())),
        "triples_by_subject_type": dict(sorted(by_subject_type.items())),
        "dropped_samples": dropped[:300],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    triples: list[Triple],
    dropped: list[dict[str, Any]],
    input_entities: int,
    input_relationships: int,
) -> None:
    with staged_output_dir(output_dir) as staged:
        write_jsonl(staged / "triples.jsonl", triples)
        write_csv(staged / "triples.csv", triples)
        write_txt(staged / "triples.txt", triples)
        write_summary(
            staged / "triple_summary.json",
            input_entities=input_entities,
            input_relationships=input_relationships,
            triples=triples,
            dropped=dropped,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create final subject-predicate-object triples from an aligned knowledge graph."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/kg_aligned/aligned_knowledge_graph.json"),
        help="Input aligned_knowledge_graph.json from 02_duiqi.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/triples"),
        help="Directory for triples.jsonl, triples.csv, triples.txt, and summary.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Drop relationships below this confidence.",
    )
    parser.add_argument(
        "--include-relation-type",
        action="append",
        default=[],
        help="Only keep these relation types. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--exclude-relation-type",
        action="append",
        default=[],
        help="Drop these relation types. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--max-endpoint-chars",
        type=int,
        default=80,
        help="Drop triples whose subject/object names exceed this length.",
    )
    parser.add_argument(
        "--no-strict-endpoints",
        action="store_true",
        help="Allow triples whose subject/object cannot be found in the entity list.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if not 0 <= args.min_confidence <= 1:
        parser.error("min-confidence must be between 0 and 1")
    if args.max_endpoint_chars <= 0:
        parser.error("max-endpoint-chars must be positive")
    entities, relationships = load_aligned_graph(args.input)
    include_relation_types = parse_relation_type_filter(args.include_relation_type)
    exclude_relation_types = parse_relation_type_filter(args.exclude_relation_type)

    triples, dropped = create_triples(
        entities=entities,
        relationships=relationships,
        min_confidence=args.min_confidence,
        include_relation_types=include_relation_types,
        exclude_relation_types=exclude_relation_types,
        max_endpoint_chars=args.max_endpoint_chars,
        strict_endpoints=not args.no_strict_endpoints,
    )
    write_outputs(
        output_dir=args.output_dir,
        triples=triples,
        dropped=dropped,
        input_entities=len(entities),
        input_relationships=len(relationships),
    )

    print(f"Input entities: {len(entities)}")
    print(f"Input relationships: {len(relationships)}")
    print(f"Created triples: {len(triples)}")
    print(f"Dropped relationships: {len(dropped)}")
    print(f"Wrote {args.output_dir / 'triples.jsonl'}")
    print(f"Wrote {args.output_dir / 'triples.csv'}")
    print(f"Wrote {args.output_dir / 'triples.txt'}")
    print(f"Wrote {args.output_dir / 'triple_summary.json'}")


if __name__ == "__main__":
    main()
