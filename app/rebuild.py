"""Rebuild only live documents, then publish a complete independent generation."""
from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from pathlib import Path

from langchain_core.documents import Document

from app import audit
from app.config import Settings
from rag import markdown_cache
from rag.bm25_index import rebuild_bm25_index
from rag.index_manifest import manifest_path, read_index_embedding_model, write_index_manifest
from rag.loaders import load_document_file
from rag.splitter import split_documents
from rag.vector_store import _add_documents, build_vector_id, get_vector_store


def safe_error(exc, settings: Settings) -> str:
    # Providers sometimes echo a bare key, outside audit.safe_error's patterns.
    message = str(exc)
    for name, value in settings.model_dump().items():
        if any(word in name for word in ("api_key", "token", "secret", "access_key", "appkey")) and value:
            message = message.replace(str(value), "[REDACTED]")
    return audit.safe_error(message)


def prepare(settings: Settings, force_reparse: bool = False) -> dict:
    from app.service import index_management_store, knowledge_base_lock

    with knowledge_base_lock.read(wait=False):
        records = index_management_store(settings).get(include=["documents", "metadatas"])
    grouped: dict[str, list[Document]] = {}
    for record_id, text, metadata in zip(records.get("ids") or [], records.get("documents") or [], records.get("metadatas") or []):
        metadata = dict(metadata or {})
        if str(metadata.get("knowledge_base_id") or "default") != settings.knowledge_base_id:
            raise ValueError("当前索引目录包含其他知识库；请先隔离目录，再重建当前知识库")
        doc_id = str(metadata.get("doc_id") or "")
        if not doc_id or not text:
            raise ValueError("索引含缺少文档身份或正文的记录，无法安全重建")
        metadata.setdefault("chunk_id", record_id)
        grouped.setdefault(doc_id, []).append(Document(page_content=text, metadata=metadata))
    if not grouped:
        raise ValueError("当前知识库没有已入库文档")
    try:
        manifest = json.loads(manifest_path(settings.chroma_persist_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    # ponytail: legacy manifests did not record splitting config. Re-split from
    # complete parser output rather than guessing; new generations record it.
    items = []
    for doc_id, chunks in sorted(grouped.items()):
        chunks.sort(key=lambda c: str(c.metadata["chunk_id"]))
        meta = chunks[0].metadata
        source = str(meta.get("source") or "")
        parsed = []
        action = "reuse_chunks"
        matching = (meta.get("index_chunk_size", manifest.get("chunk_size")), meta.get("index_chunk_overlap", manifest.get("chunk_overlap"))) == (settings.chunk_size, settings.chunk_overlap)
        if force_reparse or not matching or any(len(c.page_content.encode("utf-8")) > settings.embedding_max_input_bytes for c in chunks):
            action = "parse" if force_reparse else "resplit"
            if not force_reparse:
                parsed = markdown_cache.read_documents(settings.chroma_persist_dir, doc_id)
                if parsed and not all(any(c.page_content in d.page_content for d in parsed) for c in chunks):
                    parsed = []
                if not parsed:
                    markdown = markdown_cache.find_markdown(source, chunks)
                    if markdown:
                        parsed = [Document(page_content=markdown, metadata={"source": source, **({"structured_by": meta["structured_by"]} if meta.get("structured_by") else {})})]
            if not parsed:
                action = "parse"
                if not source or not Path(source).is_file():
                    raise ValueError(f"{meta.get('title') or doc_id}：需要完整解析文本，但缓存和原文件均不可用；旧切片不会被拼接或静默遗漏")
        items.append({"doc_id": doc_id, "source": source, "title": str(meta.get("title") or doc_id), "chunks": chunks, "parsed": parsed, "action": action})
    fingerprint = hashlib.sha256(json.dumps({
        "settings": settings.model_dump(mode="json"), "force": force_reparse,
        "records": sorted((c.metadata["chunk_id"], c.page_content, json.dumps(c.metadata, sort_keys=True)) for chunks in grouped.values() for c in chunks),
    }, sort_keys=True).encode()).hexdigest()
    return {"items": items, "plan_id": fingerprint, "force_reparse": force_reparse,
            "knowledge_base_id": settings.knowledge_base_id, "documents": len(items),
            "original_model": read_index_embedding_model(settings.chroma_persist_dir) or "未记录",
            "target_model": settings.embedding_model,
            "reuse_chunks": sum(i["action"] == "reuse_chunks" for i in items),
            "resplit": sum(i["action"] == "resplit" for i in items),
            "parse": sum(i["action"] == "parse" for i in items)}


def preview(plan: dict) -> dict:
    return {key: value for key, value in plan.items() if key != "items"}


def execute(settings: Settings, plan: dict, progress, publish, *, candidate_dir: Path | None = None) -> dict:
    from app.indexes import generation_path

    old_dir = settings.chroma_persist_dir
    target_dir = candidate_dir or generation_path(settings.knowledge_base_id, uuid.uuid4().hex)
    target = settings.model_copy(update={"chroma_persist_dir": target_dir})
    store = get_vector_store(target_dir, target.embedding_model, target.embedding_api_key,
                             target.embedding_base_url, embedding_batch_size=target.embedding_batch_size)
    total = len(plan["items"])
    chunks_written = 0
    for index, item in enumerate(plan["items"]):
        detail = {"index": index, "name": item["title"], "status": "loading", "stage": item["action"], "chunks": 0, "error": ""}
        progress(item["action"], index, total, item["title"], detail)
        try:
            chunks = item["chunks"]
            parsed = item["parsed"]
            if item["action"] != "reuse_chunks":
                if not parsed:
                    params = inspect.signature(load_document_file).parameters
                    options = {k: v for k, v in target.model_dump().items() if k in params}
                    parsed = load_document_file(Path(item["source"]), **options,
                        structurize=bool(chunks[0].metadata.get("structured_by")),
                        mineru_token=target.mineru_api_token, force_reparse=plan["force_reparse"])
                    if not parsed or any(d.metadata.get("ocr_required") or d.metadata.get("ocr_error") for d in parsed):
                        raise ValueError("解析未完成或 OCR 失败，未切换索引")
                original = chunks[0].metadata
                identity = {k: v for k, v in original.items() if k in {"doc_id", "source", "title", "created_at", "knowledge_base_id", "knowledge_base_access_group", "document_access_group", "document_access_groups"} or k.startswith("access_")}
                for doc in parsed:
                    doc.metadata.update(identity)
                chunks = split_documents(parsed, target.chunk_size, target.chunk_overlap,
                    knowledge_base_id=target.knowledge_base_id, max_input_bytes=target.embedding_max_input_bytes)
            if not chunks or any(len(c.page_content.encode("utf-8")) > target.embedding_max_input_bytes for c in chunks):
                raise ValueError("切片正文为空或仍超出目标向量输入预算")
            progress("vector_write", index, total, item["title"], detail)
            # The API batch budget is independent of upload/Chroma write limits.
            for start in range(0, len(chunks), target.embedding_batch_size):
                batch = chunks[start:start + target.embedding_batch_size]
                _add_documents(store, batch, [build_vector_id(c) for c in batch])
            if parsed:
                markdown_cache.write_documents(target_dir, parsed, strict=True)
            else:
                whole = markdown_cache.read_documents(old_dir, item["doc_id"])
                if whole:
                    markdown_cache.write_documents(target_dir, whole, strict=True)
            chunks_written += len(chunks)
            detail.update(status="done", stage="done", chunks=len(chunks), documents=len(parsed) or 1)
            progress("vector_write", index + 1, total, item["title"], detail)
        except Exception as exc:
            detail.update(status="failed", error=safe_error(exc, target))
            progress("failed", index, total, item["title"], detail)
            raise
    progress("bm25", total, total, "", None)
    rebuild_bm25_index(target_dir)
    write_index_manifest(target_dir, target.embedding_model, chunk_size=target.chunk_size,
                         chunk_overlap=target.chunk_overlap, strict=True)
    if store._collection.count() != chunks_written:
        raise ValueError("新索引切片数量校验失败，未切换")
    # Existing fingerprint anchoring detects and reports invalidated labels.
    from app.eval_store import read_questions
    from app.service import anchor_eval_labels
    anchored = anchor_eval_labels(target, read_questions())
    stale = sum(len(q["stale_chunk_ids"]) for q in anchored)
    progress("switch", total, total, "", None)
    publish(target)
    return {"files": total, "documents": total, "chunks": chunks_written,
            "stale_labels": stale, "index_path": str(target_dir.absolute()), "old_index_path": str(old_dir.absolute())}
