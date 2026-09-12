"""Tests for document structurization via markitdown."""

from pathlib import Path
import pytest

from rag.structurizer import convert_to_markdown


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF for testing."""
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "# Heading\n\nParagraph text.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_convert_pdf_to_markdown(sample_pdf: Path) -> None:
    markdown = convert_to_markdown(sample_pdf)
    assert markdown
    assert isinstance(markdown, str)
    # Markitdown extracts text; exact format varies, but it should not be empty.
    assert len(markdown.strip()) > 0


def test_convert_returns_empty_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pdf"
    result = convert_to_markdown(missing)
    # Conversion fails silently; returns empty string so caller can fall back.
    assert result == ""


def test_convert_handles_unsupported_format(tmp_path: Path) -> None:
    unsupported = tmp_path / "binary.dat"
    unsupported.write_bytes(b"\x00\x01\x02\x03")
    result = convert_to_markdown(unsupported)
    # Markitdown may fail or return empty; either way, no crash.
    assert isinstance(result, str)
