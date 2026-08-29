from langchain_core.documents import Document


def format_documents(documents: list[Document]) -> list[str]:
    formatted: list[str] = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        doc_id = document.metadata.get("doc_id", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        title = document.metadata.get("title", "")
        formatted.append(
            f"[Source Chunk {index}]\n"
            f"chunk_id: {chunk_id}\n"
            f"doc_id: {doc_id}\n"
            f"title: {title}\n"
            f"source: {source}\n"
            f"text: {document.page_content}"
        )

    return formatted
def build_vector_hits(
    results: list[tuple[Document, float]],
) -> list[dict[str, object]]:
    """Return serializable Chroma hits while preserving the original chunk text."""

    hits: list[dict[str, object]] = []
    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        hits.append(
            {
                "rank": rank,
                "distance": float(distance),
                "text": document.page_content,
                "chunk_id": str(metadata.get("chunk_id", "unknown")),
                "doc_id": str(metadata.get("doc_id", "unknown")),
                "title": str(metadata.get("title", "")),
                "source": str(metadata.get("source", "unknown")),
                "page": metadata.get("page"),
                "page_label": str(metadata.get("page_label", "")),
                "section_path": str(metadata.get("section_path", "")),
            }
        )
    return hits


def format_vector_hits(hits: list[dict[str, object]]) -> list[str]:
    return [
        (
            f"[Source Chunk {hit['rank']}]\n"
            f"chunk_id: {hit['chunk_id']}\n"
            f"doc_id: {hit['doc_id']}\n"
            f"title: {hit['title']}\n"
            f"source: {hit['source']}\n"
            f"text: {hit['text']}"
        )
        for hit in hits
    ]
