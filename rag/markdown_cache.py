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
