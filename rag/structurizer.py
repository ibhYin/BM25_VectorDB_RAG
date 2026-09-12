"""Document structurization via markitdown."""

from pathlib import Path

from markitdown import MarkItDown


_converter = MarkItDown()


def convert_to_markdown(file_path: Path) -> str:
    """Convert PDF, DOCX, images, or other supported formats to Markdown.

    Uses Microsoft's markitdown library to extract structure (headings, lists,
    tables) from documents. Returns empty string on conversion failure rather
    than raising, so the caller can fall back to raw text extraction.
    """
    try:
        result = _converter.convert(str(file_path))
        return result.text_content or ""
    except Exception:
        # Conversion failed; caller should fall back to raw extraction.
        return ""
