"""Records which embedding model produced the vectors in a Chroma directory.

Vectors from different embedding models are not comparable, so pointing the app
at a new model silently breaks retrieval. This manifest is written next to the
Chroma data at index time and read back to warn about a mismatch.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_NAME = "index_manifest.json"


def manifest_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / MANIFEST_NAME


def write_index_manifest(persist_dir: Path, embedding_model: str) -> None:
    """Record the embedding model behind this index. Never blocks ingestion."""

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
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        # A missing manifest only costs the mismatch warning, never the ingest.
        pass


def read_index_embedding_model(persist_dir: Path) -> str:
    """The model recorded for this index, or "" when it was never recorded."""

    try:
        payload = json.loads(manifest_path(persist_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("embedding_model") or "").strip()
