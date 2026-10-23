"""Temporary indexes and local fake services only; no real model calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import vector_store


def test_close_normalizes_directory_and_deduplicates_clients(tmp_path, monkeypatch):
    calls = []
    client = SimpleNamespace(close=lambda: calls.append('closed'))
    a, b = SimpleNamespace(_client=client), SimpleNamespace(_client=client)
    directory = tmp_path / 'index'
    key = str(directory.resolve())
    monkeypatch.setattr(vector_store, '_store_cache', {(key, 'a'): a, (key, 'b'): b})
    vector_store.close_vector_stores(directory / '..' / 'index')
    assert calls == ['closed']
    assert not vector_store._store_cache


def test_close_failure_stays_blocked_and_can_retry(tmp_path, monkeypatch):
    calls = []
    def close():
        calls.append(1)
        if len(calls) == 1:
            raise OSError('close failed')
    directory = tmp_path / 'index'
    monkeypatch.setattr(vector_store, '_store_cache', {(str(directory), 'a'): SimpleNamespace(_client=SimpleNamespace(close=close))})
    with pytest.raises(OSError):
        vector_store.close_vector_stores(directory)
    with pytest.raises(RuntimeError, match='关闭'):
        vector_store.get_vector_store(directory, 'model')
    vector_store.close_vector_stores(directory)
    assert not vector_store._store_cache

from app import indexes
from app.config import Settings
from app.uploads import UploadJobStore
from rag.index_manifest import write_index_manifest


def version(settings, n, *, old=None, status='completed'):
    gid = f'{n:032x}'
    path = indexes.generation_path(settings.knowledge_base_id, gid).absolute()
    job = {'id': gid, 'generation_id': gid, 'kind': 'rebuild', 'candidate_index_path': str(path),
           'old_index_path': str(old or path), 'runtime': {'knowledge_base_id': settings.knowledge_base_id, 'embedding_model': settings.embedding_model},
           'status': status, 'created_at': f'2026-10-23T{n:02}:00:00', 'error': 'original failure' if status == 'failed' else '',
           'filenames': [], 'paths': [], 'finished_at': ''}
    indexes.write_owner(job)
    (path / 'payload').write_text('retained data')
    return job, path


def settings_for(tmp_path):
    return Settings(_env_file=None, chroma_persist_dir=tmp_path / 'legacy', embedding_model='local')


def test_retention_keeps_current_previous_and_running_and_cleans_failed(tmp_path):
    settings = settings_for(tmp_path)
    a, pa = version(settings, 1)
    b, pb = version(settings, 2, old=pa)
    c, pc = version(settings, 3, old=pb)
    failed, pf = version(settings, 4, old=pb, status='failed')
    running, pr = version(settings, 5, old=pc, status='indexing')
    settings.chroma_persist_dir = pc
    jobs = {j['id']: j for j in (a, b, c, failed, running)}
    indexes.maintain(settings, jobs, lambda: None)
    assert not pa.exists() and not pf.exists()
    assert pc.exists() and pb.exists() and pr.exists()
    assert failed['error'] == 'original failure' and failed['cleanup_status'] == 'cleaned'
    d, pd = version(settings, 6, old=pc)
    jobs[d['id']] = d
    settings.chroma_persist_dir = pd
    indexes.maintain(settings, jobs, lambda: None)
    assert pd.exists() and pc.exists() and not pb.exists()


@pytest.mark.parametrize('operation', ['close', 'delete'])
def test_cleanup_failure_is_retryable_and_keeps_original_error(tmp_path, monkeypatch, operation):
    settings = settings_for(tmp_path)
    active, pa = version(settings, 1)
    failed, pf = version(settings, 2, old=pa, status='failed')
    settings.chroma_persist_dir = pa
    jobs = {j['id']: j for j in (active, failed)}
    obj, name = (vector_store, 'close_vector_stores') if operation == 'close' else (indexes.shutil, 'rmtree')
    original = getattr(obj, name)
    monkeypatch.setattr(obj, name, lambda *args, **kw: (_ for _ in ()).throw(OSError('maintenance failed')))
    indexes.maintain(settings, jobs, lambda: None)
    assert pf.exists() and pa.exists() and failed['cleanup_status'] == 'pending'
    assert failed['error'] == 'original failure' and 'maintenance failed' in failed['cleanup_error']
    monkeypatch.setattr(obj, name, original)
    indexes.maintain(settings, jobs, lambda: None)
    assert not pf.exists() and pa.exists() and failed['cleanup_status'] == 'cleaned'


def test_ownership_bounds_symlinks_and_other_kb_are_never_deleted(tmp_path):
    settings = settings_for(tmp_path)
    active, pa = version(settings, 1)
    settings.chroma_persist_dir = pa
    missing_owner, pm = version(settings, 2, status='failed')
    (pm / indexes.MARKER).unlink()
    outside, po = version(settings, 3, status='failed')
    outside['candidate_index_path'] = str(tmp_path)
    linked, pl = version(settings, 4, status='failed')
    indexes.shutil.rmtree(pl)
    pl.symlink_to(pm, target_is_directory=True)
    foreign_settings = settings.model_copy(update={'knowledge_base_id': 'other'})
    foreign, pforeign = version(foreign_settings, 5, status='failed')
    jobs = {j['id']: j for j in (active, missing_owner, outside, linked, foreign)}
    indexes.maintain(settings, jobs, lambda: None)
    assert pa.exists() and pm.exists() and tmp_path.exists() and pl.is_symlink() and pforeign.exists()
    assert all(j['cleanup_status'] == 'pending' for j in (missing_owner, outside, linked))
    assert 'cleanup_status' not in foreign


def test_directory_lease_defers_deletion_and_retired_path_cannot_reopen(tmp_path):
    settings = settings_for(tmp_path)
    active, pa = version(settings, 1)
    failed, pf = version(settings, 2, status='failed')
    settings.chroma_persist_dir = pa
    jobs = {j['id']: j for j in (active, failed)}
    with vector_store.index_directory_lock(pf):
        indexes.maintain(settings, jobs, lambda: None)
        assert pf.exists() and failed['cleanup_status'] == 'pending'
    indexes.maintain(settings, jobs, lambda: None)
    assert not pf.exists()
    with pytest.raises(RuntimeError, match='退役'):
        with vector_store.index_directory_lock(pf):
            pytest.fail('stale process must not recreate deleted generation')


def test_open_external_file_prevents_cleanup(tmp_path):
    with (tmp_path / 'open-file').open('w'):
        with pytest.raises(RuntimeError, match='进程使用'):
            indexes.assert_not_open(tmp_path)
    indexes.assert_not_open(tmp_path)


@pytest.mark.parametrize('committed', [False, True])
def test_crash_recovery_closes_orphan_without_remote_retry(tmp_path, committed):
    settings = settings_for(tmp_path)
    active, pa = version(settings, 1)
    orphan, po = version(settings, 2, old=pa, status='indexing')
    write_index_manifest(po, settings.embedding_model)
    settings.chroma_persist_dir = po if committed else pa
    store = UploadJobStore(settings, tmp_path / 'jobs.json')
    store._jobs = {j['id']: j for j in (active, orphan)}
    store._save_locked()
    try:
        store.recover()
        assert store.snapshot_job(orphan['id'])['status'] == ('completed' if committed else 'failed')
        assert po.exists() is committed
        assert pa.exists()
    finally:
        store._executor.shutdown()


def test_migration_config_failure_rolls_back_and_success_preserves_payload(tmp_path, monkeypatch):
    from app import config
    monkeypatch.chdir(tmp_path)
    settings = settings_for(tmp_path)
    settings.chroma_persist_dir = tmp_path / '.chroma-old'
    settings.chroma_persist_dir.mkdir()
    (settings.chroma_persist_dir / 'payload').write_text('source identities and parser cache')
    write_index_manifest(settings.chroma_persist_dir, 'old-model')
    jobs = {}
    monkeypatch.setattr(config, 'update_runtime_config', lambda *args: (_ for _ in ()).throw(OSError('config failed')))
    with pytest.raises(OSError, match='config failed'):
        indexes.migrate_active(settings, jobs, lambda: None)
    assert (settings.chroma_persist_dir / 'payload').read_text() == 'source identities and parser cache'
    assert not (vector_store.INDEX_ROOT / 'migration.json').exists()
    def commit(values):
        settings.chroma_persist_dir = Path(values['chroma_persist_dir'])
    monkeypatch.setattr(config, 'update_runtime_config', commit)
    result = indexes.migrate_active(settings, jobs, lambda: None)
    assert not Path(result['source']).exists()
    assert (settings.chroma_persist_dir / 'payload').read_text() == 'source identities and parser cache'
    baseline = next(j for j in jobs.values() if j['status'] == 'completed')
    assert indexes.owned_path(baseline) == settings.chroma_persist_dir


@pytest.mark.parametrize('committed', [False, True])
def test_migration_journal_recovery(tmp_path, monkeypatch, committed):
    monkeypatch.chdir(tmp_path)
    settings = settings_for(tmp_path)
    source = tmp_path / '.chroma-old'
    target = indexes.generation_path('default', 'f' * 32).absolute()
    target.mkdir(parents=True)
    (target / 'payload').write_text('original')
    journal = vector_store.INDEX_ROOT / 'migration.json'
    journal.write_text(json.dumps({'source': str(source), 'target': str(target), 'knowledge_base_id': 'default', 'generation_id': 'f' * 32}))
    settings.chroma_persist_dir = target if committed else source
    jobs = {'f' * 32: {'kind': 'migration', 'candidate_index_path': str(target), 'runtime': {'knowledge_base_id': 'default'}}}
    indexes.recover_migration(settings, jobs)
    assert not journal.exists()
    assert (target if committed else source).joinpath('payload').read_text() == 'original'


def test_retirement_preserves_deleted_documents_parse_cache(tmp_path):
    from langchain_core.documents import Document
    from rag import markdown_cache
    settings = settings_for(tmp_path)
    active, pa = version(settings, 1)
    obsolete, po = version(settings, 2, status='failed')
    settings.chroma_persist_dir = pa
    documents = [Document(page_content='whole parser output', metadata={'doc_id': 'deleted-document', 'source': '/original/report.pdf'})]
    markdown_cache.write_documents(po, documents)
    indexes.maintain(settings, {j['id']: j for j in (active, obsolete)}, lambda: None)
    assert not po.exists()
    archived = markdown_cache.read_documents(po, 'deleted-document')
    assert archived[0].page_content == documents[0].page_content
    assert archived[0].metadata == documents[0].metadata


def test_closing_waits_for_existing_readers(tmp_path):
    from threading import Event, Thread
    from app.service import knowledge_base_lock
    settings = settings_for(tmp_path)
    active, pa = version(settings, 1)
    failed, pf = version(settings, 2, status='failed')
    settings.chroma_persist_dir = pa
    started, finished = Event(), Event()
    def cleanup():
        started.set()
        indexes.maintain(settings, {j['id']: j for j in (active, failed)}, lambda: None)
        finished.set()
    with knowledge_base_lock.read():
        worker = Thread(target=cleanup)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.05)
        assert pf.exists()
    worker.join(timeout=5)
    assert finished.is_set() and not pf.exists()


def test_real_clients_for_same_directory_all_close(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, 'get_embeddings', lambda *args: None)
    directory = tmp_path / 'multiple-clients'
    a = vector_store.get_vector_store(directory, 'model', metadata_only=True)
    b = vector_store.get_vector_store(directory / '..' / directory.name, 'model', metadata_only=False)
    assert a._client is not b._client
    vector_store.close_vector_stores(directory)
    assert a._client._closed and b._client._closed
    assert not any(Path(key[0]).resolve() == directory.resolve() for key in vector_store._store_cache)
    indexes.assert_not_open(directory)


def test_candidate_ownership_is_durable_before_database_open(tmp_path, monkeypatch):
    from app import rebuild
    settings = settings_for(tmp_path)
    settings.chroma_persist_dir.mkdir()
    store = UploadJobStore(settings, tmp_path / 'jobs.json')
    plan = {'items': [{'title': 'report', 'doc_id': 'report'}], 'plan_id': 'confirmed', 'documents': 1, 'original_model': 'old'}
    monkeypatch.setattr(rebuild, 'prepare', lambda *args: plan)
    def opening(*args, **kwargs):
        persisted = json.loads((tmp_path / 'jobs.json').read_text())[0]
        assert persisted['generation_id'] and persisted['old_index_path']
        assert indexes.owned_path(persisted).is_dir()
        raise RuntimeError('controlled open failure')
    monkeypatch.setattr(rebuild, 'get_vector_store', opening)
    job = store.create_rebuild('confirmed')
    store._executor.shutdown(wait=True)
    result = store.snapshot_job(job['id'])
    assert result['status'] == 'failed' and 'controlled open failure' in result['error']
    assert result['cleanup_status'] == 'cleaned'


def test_explicit_legacy_adoption_verifies_job_and_contents(tmp_path, monkeypatch):
    import chromadb
    from datetime import datetime, timezone
    monkeypatch.chdir(tmp_path)
    settings = settings_for(tmp_path)
    settings.chroma_persist_dir = tmp_path / '.chroma-original'
    source = tmp_path / ('.chroma-generation-' + 'e' * 32)
    for path in (settings.chroma_persist_dir, source):
        with chromadb.PersistentClient(path=str(path)) as client:
            collection = client.create_collection(vector_store.COLLECTION_NAME)
            collection.add(ids=['one'], embeddings=[[1.0, 2.0, 3.0]], documents=['original body'],
                           metadatas=[{'doc_id': 'same-doc', 'knowledge_base_id': 'default'}])
    now = datetime.now(timezone.utc).isoformat()
    job = {'id': 'old-job', 'kind': 'rebuild', 'status': 'failed', 'runtime': {'knowledge_base_id': 'default'},
           'started_at': now, 'created_at': now, 'error': 'original provider failure'}
    jobs = {job['id']: job}
    # Neither a different task nor an unrelated KB can authorize adoption.
    job['runtime']['knowledge_base_id'] = 'foreign'
    with pytest.raises(ValueError):
        indexes.adopt_failed_candidate(settings, jobs, lambda: None, job['id'], source)
    assert source.exists()
    job['runtime']['knowledge_base_id'] = 'default'
    result = indexes.adopt_failed_candidate(settings, jobs, lambda: None, job['id'], source)
    assert not source.exists() and indexes.owned_path(job).is_dir() and result['chunks'] == 1
    indexes.maintain(settings, jobs, lambda: None)
    assert not Path(result['path']).exists() and settings.chroma_persist_dir.exists()
    assert job['error'] == 'original provider failure'


@pytest.mark.parametrize('command', ['maintain-indexes', 'web'])
def test_maintenance_and_startup_errors_do_not_print_keys(tmp_path, monkeypatch, command):
    from typer.testing import CliRunner
    from app import main, web
    settings = settings_for(tmp_path)
    settings.embedding_api_key = 'private-key-never-print'
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    def fail(*args, **kwargs):
        raise RuntimeError('controlled failure ' + settings.embedding_api_key)
    if command == 'web':
        monkeypatch.setattr(web, 'run_web_server', fail)
    else:
        monkeypatch.setattr(vector_store, 'index_directory_lock', fail)
    result = CliRunner().invoke(main.app, [command])
    assert result.exit_code == 1
    assert settings.embedding_api_key not in result.output
    assert '[REDACTED]' in result.output
