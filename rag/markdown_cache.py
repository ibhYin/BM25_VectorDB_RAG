"""Content-addressed cache for converted Markdown.

Structured extraction is the expensive step — MinerU is a remote, metered API and
MinerU re-parses whole PDFs. The converted Markdown is keyed by the *file
contents* plus the converter identity, so:

* editing a file invalidates its entry automatically (no mtime bookkeeping)
* switching pipeline/vlm re-parses, because the output genuinely differs
* renaming a file still hits the cache

Not covered: rotating the MinerU token, or MinerU upgrading its model server.
Those keep the same key, so delete the cache directory to force a re-parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_CACHE_DIR = Path(".markdown-cache")
CACHE_DIR_ENV = "MARKDOWN_CACHE_DIR"
CACHE_DISABLED_ENV = "MARKDOWN_CACHE_DISABLED"

logger = logging.getLogger(__name__)


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_DIR_ENV, str(DEFAULT_CACHE_DIR)))


def _disabled() -> bool:
    return os.environ.get(CACHE_DISABLED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def cache_key(file_path: Path, provider: str, model_version: str = "") -> str:
    """Key on file contents, not path — renames must still hit the cache."""

    digest = hashlib.sha256()
    digest.update(Path(file_path).read_bytes())
    # Length-prefix each part so "ab"+"c" and "a"+"bc" cannot collide.
    for part in (provider, model_version):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _entry_path(file_path: Path, provider: str, model_version: str) -> Path:
    return cache_dir() / f"{cache_key(file_path, provider, model_version)}.json"


def read(file_path: Path, provider: str, model_version: str = "") -> str | None:
    """The cached Markdown, or None when there is no usable entry."""

    if _disabled():
        return None
    try:
        payload = json.loads(
            _entry_path(file_path, provider, model_version).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        # Missing, unreadable or corrupt all mean the same thing: a miss.
        return None
    if not isinstance(payload, dict):
        return None
    markdown = payload.get("markdown")
    return markdown if isinstance(markdown, str) and markdown else None


def has_entry(file_path: Path, provider: str, model_version: str = "") -> bool:
    """Whether a usable cached conversion exists, without reading it back."""

    if _disabled() or not Path(file_path).is_file():
        return False
    try:
        return _entry_path(file_path, provider, model_version).is_file()
    except OSError:
        return False


def write(
    file_path: Path,
    provider: str,
    model_version: str,
    markdown: str,
    source: str = "",
) -> None:
    """Store one conversion. Never raises: a cache miss must not fail an ingest."""

    if _disabled() or not markdown:
        return
    try:
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        target = _entry_path(file_path, provider, model_version)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "source": source or str(file_path),
                    "provider": provider,
                    "model_version": model_version,
                    "chars": len(markdown),
                    "cached_at": datetime.now().astimezone().isoformat(),
                    "markdown": markdown,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError as exc:
        logger.warning("Could not write markdown cache entry: %s", exc)


def info() -> dict[str, object]:
    """Cache location and entry count, for the settings page."""

    directory = cache_dir()
    entries = 0
    size = 0
    if directory.is_dir():
        for path in directory.glob("*.json"):
            entries += 1
            try:
                size += path.stat().st_size
            except OSError:
                continue
    return {"dir": str(directory), "entries": entries, "bytes": size}


def _parsed_path(persist_dir: Path, doc_id: str) -> Path:
    return Path(persist_dir) / "parsed" / (hashlib.sha256(doc_id.encode()).hexdigest() + ".json")


def _parsed_archive(persist_dir: Path) -> Path:
    return cache_dir() / "index-parsed" / hashlib.sha256(str(Path(persist_dir).resolve()).encode()).hexdigest()


def preserve_parsed_documents(persist_dir: Path) -> None:
    """Index retirement must not delete whole-parser caches, including deleted docs.

    Archives remain namespaced by index identity; never merge overlapping chunks
    or substitute another version's parser output for a live document.
    """
    parsed = Path(persist_dir) / "parsed"
    if not parsed.exists():
        return
    if parsed.is_symlink() or any(p.is_symlink() for p in parsed.rglob("*")):
        raise ValueError("解析缓存包含符号链接，保留索引目录待核对")
    shutil.copytree(parsed, _parsed_archive(persist_dir), dirs_exist_ok=True)


def write_documents(persist_dir: Path, documents: list, *, strict: bool = False) -> None:
    """Save whole parser outputs, never reconstructed overlapping chunks."""
    try:
        grouped: dict[str, list] = {}
        for doc in documents:
            doc_id = str(doc.metadata.get("doc_id") or "")
            if doc_id:
                grouped.setdefault(doc_id, []).append({"text": doc.page_content, "metadata": doc.metadata})
        for doc_id, entries in grouped.items():
            target = _parsed_path(persist_dir, doc_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, target)
    except OSError:
        if strict:
            raise
        logger.warning("Could not preserve complete parsed documents; original-file fallback remains available")


def read_documents(persist_dir: Path, doc_id: str) -> list:
    from langchain_core.documents import Document

    try:
        path = _parsed_path(persist_dir, doc_id)
        if not path.exists():
            path = _parsed_archive(persist_dir) / path.name
        entries = json.loads(path.read_text(encoding="utf-8"))
        return [Document(page_content=e["text"], metadata=e["metadata"]) for e in entries]
    except (OSError, ValueError, TypeError, KeyError):
        return []


def find_markdown(source: str, chunks: list) -> str | None:
    """Locate an existing complete conversion, including when the raw file is missing.

    Verify that every indexed chunk belongs to it; source names alone do not
    prove a cached conversion matches the indexed document version.
    """
    # ponytail: legacy entries lack a doc_id lookup (O(cache entries)); new
    # parsed snapshots use doc_id-addressed paths for subsequent rebuilds.
    for path in sorted(cache_dir().glob("*.json"), reverse=True):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict):
                continue
            if Path(str(entry.get("source") or "")).resolve() != Path(source).resolve():
                continue
            text = entry.get("markdown")
            if isinstance(text, str) and text and all(c.page_content in text for c in chunks):
                return text
        except (OSError, ValueError, TypeError):
            continue
    return None
