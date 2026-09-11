import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


COLLECTION_NAME = "knowledge_base"


def build_vector_id(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return chunk_id
    identity = "\u241f".join(
        [
            str(document.metadata.get("source", "unknown")),
            str(document.metadata.get("page", "")),
            document.page_content,
        ]
    )
    return "document_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def get_embeddings(
    model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
) -> OpenAIEmbeddings:
    kwargs = {"api_key": embedding_api_key} if embedding_api_key else {}
    kwargs["request_timeout"] = 60
    kwargs["max_retries"] = 2
    if embedding_base_url:
        # ponytail: custom gateways enforce their own context limits.
        kwargs["base_url"] = embedding_base_url
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 10
    return OpenAIEmbeddings(model=model, **kwargs)


def get_vector_store(
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(
            embedding_model,
            embedding_api_key,
            embedding_base_url,
        ),
        persist_directory=str(persist_dir),
    )


def ingest_documents(
    documents: list[Document],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    rebuild: bool = False,
    knowledge_base_id: str | None = None,
) -> Chroma:
    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
    )

    document_ids = [build_vector_id(document) for document in documents]
    existing_ids: set[str] = set()
    if rebuild:
        kwargs: dict[str, object] = {"include": []}
        if knowledge_base_id:
            kwargs["where"] = {"knowledge_base_id": knowledge_base_id}
        existing_ids = set(vector_store.get(**kwargs).get("ids", []))
    if documents:
        vector_store.add_documents(documents, ids=document_ids)
    stale_ids = existing_ids.difference(document_ids)
    if stale_ids:
        # Delete stale records only after all new embeddings were written successfully.
        vector_store.delete(ids=sorted(stale_ids))

    return vector_store


def replace_documents(
    documents: list[Document],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    knowledge_base_id: str | None = None,
) -> Chroma:
    """Upsert documents and remove stale chunks after the new vectors are stored."""

    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
    )
    document_ids = [build_vector_id(document) for document in documents]
    doc_ids = sorted(
        {
            str(document.metadata.get("doc_id", ""))
            for document in documents
            if document.metadata.get("doc_id")
        }
    )
    old_ids: set[str] = set()
    if doc_ids:
        where: dict[str, object] = {"doc_id": {"$in": doc_ids}}
        if knowledge_base_id:
            where = {
                "$and": [where, {"knowledge_base_id": knowledge_base_id}],
            }
        existing = vector_store.get(
            where=where,
            include=[],
        )
        old_ids = set(existing.get("ids", []))

    if documents:
        vector_store.add_documents(documents, ids=document_ids)
    stale_ids = old_ids.difference(document_ids)
    if stale_ids:
        vector_store.delete(ids=sorted(stale_ids))

    return vector_store


def delete_documents(
    doc_ids: list[str],
    persist_dir: Path,
    embedding_model: str,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    knowledge_base_id: str | None = None,
) -> int:
    """Remove every chunk of the given documents, scoped to one knowledge base."""

    unique_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique_ids:
        return 0
    vector_store = get_vector_store(
        persist_dir,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
    )
    where: dict[str, object] = {"doc_id": {"$in": unique_ids}}
    if knowledge_base_id:
        where = {"$and": [where, {"knowledge_base_id": knowledge_base_id}]}
    existing = vector_store.get(where=where, include=[])
    stale_ids = sorted(existing.get("ids", []))
    if stale_ids:
        vector_store.delete(ids=stale_ids)
    return len(stale_ids)
