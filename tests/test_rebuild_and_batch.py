"""Temporary Chroma + deterministic local embeddings; never touch real indexes/APIs."""
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app import config, rebuild, service
from app.config import Settings
from app.uploads import UploadJobStore
from app.web import RequestError, WebRequestHandler
from rag import markdown_cache, vector_store
from rag.index_manifest import read_index_embedding_model, write_index_manifest
from rag.splitter import split_documents


class Embeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 2.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "get_embeddings", lambda *args: Embeddings())
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "old", data_dir=tmp_path / "raw",
                        embedding_model="old-model", embedding_api_key="test-secret-must-not-leak",
                        chunk_size=80, chunk_overlap=10, embedding_max_input_bytes=200)
    settings.data_dir.mkdir()
    docs = []
    for name in ("live", "deleted", "not-indexed"):
        path = settings.data_dir / f"{name}.txt"
        path.write_text("A full original document about " + name + ". " * 50)
        if name != "not-indexed":
            docs.append(Document(page_content=path.read_text(), metadata={"source": str(path), "doc_id": name, "document_access_groups": ["private"], "document_access_group": "private", "access_private": True}))
    chunks = split_documents(docs, 80, 10)
    old = vector_store.ingest_documents(chunks, settings.chroma_persist_dir, "old-model")
    old.delete(ids=[c.metadata["chunk_id"] for c in chunks if c.metadata["doc_id"] == "deleted"])
    write_index_manifest(settings.chroma_persist_dir, "old-model", chunk_size=80, chunk_overlap=10)
    markdown_cache.write_documents(settings.chroma_persist_dir, docs)
    settings.embedding_model = "new-model"
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(config, "_RUNTIME_OVERRIDES", {})
    return settings, old


def test_mismatch_management_and_successful_rebuild_only_live_docs(corpus, tmp_path):
    settings, old = corpus
    original_ids = old.get()["ids"]
    with pytest.raises(ValueError, match="禁止混用"):
        vector_store.get_vector_store(settings.chroma_persist_dir, "new-model")
    assert [d["doc_id"] for d in service.list_indexed_documents(settings, admin=True)] == ["live"]
    plan = rebuild.prepare(settings)
    assert plan["documents"] == 1 and plan["reuse_chunks"] == 1
    old_dir = settings.chroma_persist_dir
    jobs = UploadJobStore(settings, tmp_path / "jobs.json")
    job = jobs.create_rebuild(plan["plan_id"], actor="admin")
    jobs._executor.shutdown(wait=True)
    result = jobs.snapshot_job(job["id"])
    assert result["status"] == "completed", result
    assert settings.chroma_persist_dir != old_dir
    assert read_index_embedding_model(settings.chroma_persist_dir) == settings.embedding_model
    records = service.index_management_store(settings).get(include=["documents", "metadatas"])
    assert {m["doc_id"] for m in records["metadatas"]} == {"live"}
    assert all(m["access_private"] and m["document_access_groups"] == ["private"] for m in records["metadatas"])
    assert records["ids"] == original_ids
    assert old._client._closed
    # The retained snapshot is unchanged on disk; its retired client is closed.
    import chromadb
    with chromadb.PersistentClient(path=str(old_dir.resolve())) as client:
        assert client.get_collection(vector_store.COLLECTION_NAME).get()["ids"] == original_ids
    saved = json.loads(config.runtime_config_path().read_text())
    assert Path(saved["chroma_persist_dir"]) == settings.chroma_persist_dir
    # Reloading the task file keeps finished results without re-running anything.
    reloaded = UploadJobStore(settings, tmp_path / "jobs.json")
    assert reloaded.snapshot_job(job["id"])["status"] == "completed"
    reloaded._executor.shutdown()


@pytest.mark.parametrize("fail_stage", ["embedding", "bm25", "cache", "persist"])
def test_failure_retains_old_index_and_redacts_credentials(corpus, tmp_path, monkeypatch, fail_stage):
    settings, old = corpus
    old_dir = settings.chroma_persist_dir
    old_ids = old.get()["ids"]
    plan = rebuild.prepare(settings)

    def fail(*args, **kwargs):
        raise RuntimeError("remote failed " + settings.embedding_api_key)

    if fail_stage == "embedding":
        monkeypatch.setattr(rebuild, "_add_documents", fail)
    elif fail_stage == "bm25":
        monkeypatch.setattr(rebuild, "rebuild_bm25_index", fail)
    elif fail_stage == "cache":
        monkeypatch.setattr(markdown_cache, "write_documents", fail)
    else:
        monkeypatch.setattr(config, "_persist_runtime_overrides", fail)
    jobs = UploadJobStore(settings, tmp_path / "jobs.json")
    job = jobs.create_rebuild(plan["plan_id"])
    jobs._executor.shutdown(wait=True)
    result = jobs.snapshot_job(job["id"])
    assert result["status"] == "failed"
    assert settings.embedding_api_key not in json.dumps(result)
    assert "[REDACTED]" in result["error"]
    assert settings.chroma_persist_dir == old_dir
    assert read_index_embedding_model(old_dir) == "old-model"
    assert old.get()["ids"] == old_ids
    assert "chroma_persist_dir" not in config._RUNTIME_OVERRIDES
    assert result["cleanup_status"] == "cleaned"
    assert not Path(result["candidate_index_path"]).exists()
    # Failed task released the mutation/rebuild gate.
    with service.knowledge_base_lock.mutation():
        pass


def test_missing_original_uses_chunks_or_complete_parse_but_never_concatenates(corpus):
    settings, _ = corpus
    (settings.data_dir / "live.txt").unlink()
    assert rebuild.prepare(settings)["reuse_chunks"] == 1
    settings.chunk_size = 40
    plan = rebuild.prepare(settings)
    assert plan["resplit"] == 1
    markdown_cache._parsed_path(settings.chroma_persist_dir, "live").unlink()
    with pytest.raises(ValueError, match="缓存和原文件均不可用"):
        rebuild.prepare(settings)
    with pytest.raises(ValueError):
        rebuild.prepare(settings, force_reparse=True)


def test_resplit_preserves_identity_and_obeys_byte_budget(corpus):
    settings, _ = corpus
    settings.embedding_max_input_bytes = 32
    plan = rebuild.prepare(settings)
    assert plan["resplit"] == 1
    published = []
    result = rebuild.execute(settings, plan, lambda *args: None, published.append)
    target = published[0]
    chunks = service.index_management_store(target).get(include=["documents", "metadatas"])
    assert chunks["documents"] and all(len(t.encode()) <= 32 for t in chunks["documents"])
    assert {m["doc_id"] for m in chunks["metadatas"]} == {"live"}
    assert result["files"] == 1


def test_rebuild_blocks_mutations_and_duplicate_submission(corpus, tmp_path):
    settings, _ = corpus
    jobs = UploadJobStore(settings, tmp_path / "jobs.json")
    service.knowledge_base_lock.begin_rebuild()
    try:
        with pytest.raises(RuntimeError, match="已有"):
            jobs.create_rebuild("a" * 64)
        # Duplicate failure must not release somebody else's reservation.
        with pytest.raises(RuntimeError, match="重建"):
            jobs.create_uploaded([("a.txt", b"data")], settings.data_dir)
        with pytest.raises(RuntimeError, match="重建"):
            service.delete_indexed_documents(settings, ["live"])
    finally:
        service.knowledge_base_lock.end_rebuild()
        jobs._executor.shutdown()


def test_batch_delete_scopes_ids_and_updates_bm25_once(corpus, monkeypatch):
    settings, _ = corpus
    calls = []
    monkeypatch.setattr(service, "rebuild_bm25_index", lambda *args: calls.append(args))
    result = service.delete_indexed_documents_batch(settings, ["live", "not-indexed", "live"], actor="admin")
    assert result["deleted"] == ["live"]
    assert result["not_deleted"] == ["not-indexed"]
    assert not result["failed"] and len(calls) == 1
    assert (settings.data_dir / "live.txt").is_file()
    assert markdown_cache.read_documents(settings.chroma_persist_dir, "live")
    with pytest.raises(ValueError, match="没有已入库"):
        rebuild.prepare(settings)


def test_batch_delete_reports_partial_failure_and_foreign_kb(corpus, monkeypatch):
    settings, old = corpus
    foreign = Document(page_content="foreign", metadata={"doc_id": "foreign", "chunk_id": "foreign#1", "knowledge_base_id": "other"})
    old.add_documents([foreign], ids=["foreign#1"])
    original = vector_store._delete_ids

    def partial(store, ids):
        original(store, ids[:1])
        raise RuntimeError("write failed")

    monkeypatch.setattr(vector_store, "_delete_ids", partial)
    result = service.delete_indexed_documents_batch(settings, ["live", "foreign"])
    assert result["deleted"] == [] and result["failed"] == ["live"]
    assert result["not_deleted"] == ["foreign"] and result["deleted_chunks"] == 1
    assert "write failed" in result["error"]
    assert "foreign#1" in old.get()["ids"]


def handler(payload):
    h = object.__new__(WebRequestHandler)
    h._read_json = lambda: dict(payload)
    h._require_admin = lambda: None
    h._send_json = lambda result, **kwargs: result
    return h


def test_batch_route_checks_admin_and_kb_before_delete(corpus, monkeypatch):
    settings, _ = corpus
    monkeypatch.setattr("app.web.get_settings", lambda: settings)
    h = handler({"knowledge_base_id": "other", "doc_ids": ["live"]})
    with pytest.raises(RequestError, match="知识库已变化"):
        h._handle_document_delete_batch()
    h._require_admin = lambda: (_ for _ in ()).throw(RequestError(403, "admin required"))
    with pytest.raises(RequestError, match="admin required"):
        h._handle_document_delete_batch()


def test_section_config_rejects_cross_section_and_masked_secret():
    h = handler({"section": "mineru", "embedding_model": "new"})
    with pytest.raises(RequestError, match="本节"):
        h._handle_update_config()
    with pytest.raises(RequestError, match="占位符"):
        WebRequestHandler._read_config_updates({"embedding_api_key": "••••1234"})


def test_force_reparse_bypasses_complete_cached_output(corpus, monkeypatch):
    settings, _ = corpus
    calls = []

    def parser(path, **kwargs):
        calls.append(kwargs)
        return [Document(page_content="fresh parser result", metadata={"source": str(path)})]

    monkeypatch.setattr(rebuild, "load_document_file", parser)
    plan = rebuild.prepare(settings, force_reparse=True)
    assert plan["parse"] == 1
    published = []
    rebuild.execute(settings, plan, lambda *args: None, published.append)
    assert calls[0]["force_reparse"] is True
    records = service.index_management_store(published[0]).get(include=["documents"])
    assert records["documents"] == ["fresh parser result"]


def test_byte_budget_batch_limit_and_stale_eval_labels(corpus, monkeypatch):
    from app.eval_store import fingerprint_text, upsert_question

    settings, old = corpus
    records = old.get(include=["documents", "metadatas"])
    upsert_question("example", relevant_chunk_ids=[records["ids"][0]], fingerprints={
        records["ids"][0]: {"doc_id": "live", "text_sha1": fingerprint_text(records["documents"][0])},
    })
    whole = Document(page_content="中文证据，包含完整的财务指标与经营说明。" * 20,
                     metadata={"doc_id": "live", "source": str(settings.data_dir / "live.txt")})
    original_chunks = split_documents([whole], 80, 10)
    old.delete(ids=records["ids"])
    old.add_documents(original_chunks, ids=[c.metadata["chunk_id"] for c in original_chunks])
    upsert_question("example", relevant_chunk_ids=[original_chunks[0].metadata["chunk_id"]], fingerprints={
        original_chunks[0].metadata["chunk_id"]: {"doc_id": "live", "text_sha1": fingerprint_text(original_chunks[0].page_content)},
    })
    markdown_cache.write_documents(settings.chroma_persist_dir, [whole])
    settings.embedding_max_input_bytes = 24
    settings.embedding_batch_size = 2
    sizes = []

    class LimitedEmbedding(Embeddings):
        def embed_documents(self, texts):
            sizes.append(len(texts))
            assert len(texts) <= 2
            return super().embed_documents(texts)

    monkeypatch.setattr(vector_store, "get_embeddings", lambda *args: LimitedEmbedding())
    published = []
    result = rebuild.execute(settings, rebuild.prepare(settings), lambda *args: None, published.append)
    records = service.index_management_store(published[0]).get(include=["documents"])
    assert max(map(lambda t: len(t.encode()), records["documents"])) <= 24
    detailed = service.index_management_store(published[0]).get(include=["documents", "metadatas"])
    for text, meta in zip(detailed["documents"], detailed["metadatas"]):
        assert whole.page_content[meta["start_char"]:meta["end_char"]] == text
    assert sizes and max(sizes) <= 2
    assert result["stale_labels"] == 1


def test_confirmation_fingerprint_rejects_changed_configuration(corpus, tmp_path):
    settings, _ = corpus
    plan = rebuild.prepare(settings)
    settings.chunk_size = 70
    jobs = UploadJobStore(settings, tmp_path / "jobs.json")
    with pytest.raises(ValueError, match="已变化"):
        jobs.create_rebuild(plan["plan_id"])
    with service.knowledge_base_lock.mutation():
        pass
    assert jobs.snapshot() == []
    jobs._executor.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_interrupted_task_recovery_uses_atomic_config_pointer(corpus, tmp_path, committed):
    settings, _ = corpus
    jobs = UploadJobStore(settings, tmp_path / "jobs.json")
    candidate = tmp_path / "candidate"
    write_index_manifest(candidate, "new-model")
    job = jobs._new_job(["live"], [])
    job.update(kind="rebuild", status="indexing", stage="switch", candidate_index_path=str(candidate))
    jobs._jobs[job["id"]] = job
    jobs._save_locked()
    if committed:
        settings.chroma_persist_dir = candidate
    jobs.recover()
    assert jobs.snapshot_job(job["id"])["status"] == ("completed" if committed else "failed")
    jobs._executor.shutdown()


def test_stale_complete_parse_cache_is_not_silently_used(corpus, monkeypatch):
    settings, _ = corpus
    settings.chunk_size = 40
    markdown_cache.write_documents(settings.chroma_persist_dir, [Document(
        page_content="obsolete unrelated text", metadata={"doc_id": "live"})])
    plan = rebuild.prepare(settings)
    assert plan["parse"] == 1
    assert not plan["items"][0]["parsed"]


def test_relative_legacy_path_keeps_generation_ownership_consistent(corpus, tmp_path, monkeypatch):
    settings, _ = corpus
    monkeypatch.chdir(tmp_path)
    settings.chroma_persist_dir = Path('old')
    store = UploadJobStore(settings, tmp_path / 'relative-jobs.json')
    versions = []
    try:
        for _ in range(3):
            plan = rebuild.prepare(settings)
            # Run deterministically on this test's worker without submitting a paid model.
            store._executor.submit = lambda function, *args: function(*args)
            job = store.create_rebuild(plan['plan_id'])
            assert job['status'] == 'completed', job
            assert 'cleanup_error' not in job or not job['cleanup_error']
            versions.append(Path(job['candidate_index_path']))
        assert not versions[0].exists()
        assert versions[1].exists() and versions[2].exists()
        assert not any(j.get('cleanup_status') == 'pending' for j in store.snapshot())
    finally:
        store._executor.shutdown(wait=True)
