"""Records which embedding model produced the vectors in a Chroma directory.

Vectors from different embedding models are not comparable, so pointing the app
at a new model silently breaks retrieval. This manifest is written next to the
Chroma data at index time and checked before opening or appending to an index.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_NAME = "index_manifest.json"


def ensure_index_embedding_model(persist_dir: Path, embedding_model: str) -> None:
    recorded = read_index_embedding_model(persist_dir)
    # ponytail: legacy indexes have no model identity; rebuild them in a new
    # directory to enable verification rather than guessing from dimensions.
    if recorded and recorded != embedding_model:
        raise ValueError(
            f"索引由 {recorded} 建立，当前向量模型为 {embedding_model}。"
            "请切回原模型，或使用新的索引目录重新入库；禁止混用不同模型的向量。"
        )


def manifest_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / MANIFEST_NAME


def write_index_manifest(
    persist_dir: Path, embedding_model: str, *,
    chunk_size: int | None = None, chunk_overlap: int | None = None,
    strict: bool = False,
) -> None:
    """Record model/split identity; strict publication fails on a missing manifest."""

    if not embedding_model:
        return
    directory = Path(persist_dir)
    target = manifest_path(directory)
    temporary = target.with_name(target.name + ".tmp")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                {
                    "embedding_model": embedding_model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        if strict:
            raise


def read_index_embedding_model(persist_dir: Path) -> str:
    """The model recorded for this index, or "" when it was never recorded."""

    try:
        payload = json.loads(manifest_path(persist_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("embedding_model") or "").strip()
