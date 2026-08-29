import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ensure_document_metadata(documents: list[Document]) -> list[Document]:
    """Attach stable document metadata required by indexing and retrieval."""

    for document in documents:
        source = str(document.metadata.get("source") or "unknown")
        document.metadata.setdefault("doc_id", build_doc_id(source))
        document.metadata.setdefault("title", build_title(source))
        document.metadata.setdefault("created_at", "")
        document.metadata.setdefault("source", source)

    return documents


def build_doc_id(source: str) -> str:
    source_path = Path(source)
    stem = source_path.stem if source_path.stem else "document"
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", stem).strip("_").lower()
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'document'}_{digest}"


def build_title(source: str) -> str:
    source_path = Path(source)
    return source_path.stem or source


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split documents into chunks suitable for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
        add_start_index=True,
    )
    chunks = splitter.split_documents(ensure_document_metadata(documents))

    chunk_counts_by_doc_id: dict[str, int] = {}
    for chunk in chunks:
        doc_id = str(chunk.metadata["doc_id"])
        chunk_counts_by_doc_id[doc_id] = chunk_counts_by_doc_id.get(doc_id, 0) + 1

        start_char = int(chunk.metadata.pop("start_index", 0))
        chunk.metadata["chunk_id"] = (
            f"{doc_id}#chunk_{chunk_counts_by_doc_id[doc_id]:03d}"
        )
        chunk.metadata["start_char"] = start_char
        chunk.metadata["end_char"] = start_char + len(chunk.page_content)
        chunk.metadata["section_path"] = infer_section_path(chunk.page_content)

    return chunks


def infer_section_path(text: str) -> str:
    headings = [
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.startswith("#")
    ]
    return " > ".join(headings)
