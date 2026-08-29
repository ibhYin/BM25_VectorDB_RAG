from pathlib import Path

from langchain_core.documents import Document

from rag import loaders
from rag.loaders import load_documents, load_pdf_documents, pdf_page_to_markdown


def make_text_pdf(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        + stream
        + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def test_load_documents_supports_text_markdown_and_pdf(tmp_path: Path) -> None:
    pdf_text = "pdf document text " * 20
    (tmp_path / "note.txt").write_text("plain text document", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# markdown document", encoding="utf-8")
    (tmp_path / "sample.pdf").write_bytes(make_text_pdf(pdf_text))
    (tmp_path / "ignored.csv").write_text("ignored", encoding="utf-8")

    documents = load_documents(tmp_path)

    contents = "\n".join(document.page_content for document in documents)
    sources = {Path(document.metadata["source"]).suffix for document in documents}

    assert "plain text document" in contents
    assert "markdown document" in contents
    assert "pdf document text" in contents
    assert sources == {".txt", ".md", ".pdf"}


def test_load_pdf_documents_uses_fast_path_for_text_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "text.pdf"
    pdf_path.write_bytes(make_text_pdf("text pdf content " * 20))

    called = False

    def fake_ocr_pdf_with_openai(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(loaders, "ocr_pdf_with_openai", fake_ocr_pdf_with_openai)

    documents = load_pdf_documents(pdf_path, openai_api_key="test-key")

    assert called is False
    assert "text pdf content" in documents[0].page_content
    assert documents[0].metadata["loader"] == "pymupdf_markdown"
    assert documents[0].metadata["format"] == "markdown"


def test_pdf_page_to_markdown_preserves_heading_and_paragraph_boundaries(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "structured.pdf"
    pdf_path.write_bytes(make_text_pdf("1. Overview"))

    import fitz

    with fitz.open(pdf_path) as pdf:
        markdown = pdf_page_to_markdown(pdf[0])

    assert "## 1. Overview" in markdown


def test_load_pdf_documents_uses_ocr_for_scanned_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(make_text_pdf(""))

    def fake_ocr_pdf_with_openai(**kwargs):
        return [
            Document(
                page_content="ocr extracted content",
                metadata={"source": str(pdf_path), "loader": "openai_ocr"},
            )
        ]

    monkeypatch.setattr(loaders, "ocr_pdf_with_openai", fake_ocr_pdf_with_openai)

    documents = load_pdf_documents(pdf_path, openai_api_key="test-key")

    assert documents[0].page_content == "ocr extracted content"
    assert documents[0].metadata["loader"] == "openai_ocr"


def test_load_pdf_documents_keeps_empty_pdf_without_api_key(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(make_text_pdf(""))

    documents = load_pdf_documents(pdf_path)

    assert len(documents) == 1
    assert documents[0].page_content.strip() == ""
    assert documents[0].metadata["ocr_required"] is True


def test_load_pdf_documents_keeps_pdf_when_ocr_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(make_text_pdf(""))

    def fake_ocr_pdf_with_openai(**kwargs):
        raise RuntimeError("ocr failed")

    monkeypatch.setattr(loaders, "ocr_pdf_with_openai", fake_ocr_pdf_with_openai)

    documents = load_pdf_documents(pdf_path, openai_api_key="test-key")

    assert len(documents) == 1
    assert documents[0].page_content.strip() == ""
    assert documents[0].metadata["ocr_error"] == "ocr failed"
