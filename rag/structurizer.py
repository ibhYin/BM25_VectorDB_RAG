"""Turn a source document into Markdown before chunking.

Two backends, in order of preference:

* **MinerU online** — real layout analysis, so headings/lists/tables survive.
  Needs an API token. This is the only backend that structures PDFs usefully.
* **markitdown** — local, free, good for formats that already carry structure
  (docx/pptx/xlsx/html). For PDFs it is weaker than the existing PyMuPDF path,
  which is why it is only a fallback.
"""

from pathlib import Path

from markitdown import MarkItDown

from rag.mineru_client import MineruError, convert_file as convert_with_mineru


_converter = MarkItDown()


def convert_to_markdown(
    file_path: Path,
    mineru_token: str = "",
    mineru_model_version: str = "vlm",
) -> str:
    """Convert one file to Markdown, or "" when no backend produced anything.

    With a MinerU token the online service is authoritative and its failures
    propagate: falling back silently would feed the index noticeably worse text
    than the caller asked for.
    """

    if mineru_token:
        return convert_with_mineru(file_path, mineru_token, mineru_model_version)
    return convert_with_markitdown(file_path)


def convert_with_markitdown(file_path: Path) -> str:
    """Local markitdown conversion; "" on failure so callers can fall back."""

    try:
        result = _converter.convert(str(file_path))
        return result.text_content or ""
    except Exception:
        # Conversion failed; caller should fall back to raw extraction.
        return ""


__all__ = [
    "MineruError",
    "convert_to_markdown",
    "convert_with_markitdown",
    "convert_with_mineru",
]
