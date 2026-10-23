"""Local index generations: job ownership, retirement and conservative cleanup.

The runtime config remains the only active pointer. Job records plus an on-disk
ownership marker authorize deletion; directory names alone never do.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from rag import vector_store

MARKER = "generation.json"


def assert_not_open(path: Path) -> None:
    """Also catch pre-upgrade/external clients that do not take project leases."""
    if not path.exists():
        return
    result = subprocess.run(["lsof", "-t", "+D", str(path)], capture_output=True, text=True, timeout=10)
    if result.stdout.strip():
        raise RuntimeError("索引文件仍被进程使用，请结束相关服务/CLI 后重试维护")
    if result.returncode not in {0, 1} or result.stderr.strip():
        raise RuntimeError("无法确认索引文件已释放，已保留目录待下次维护")


def generation_path(knowledge_base_id: str, generation_id: str) -> Path:
    if not re.fullmatch(r"[\w-]{1,100}", knowledge_base_id) or not re.fullmatch(r"[a-f0-9]{32}", generation_id):
        raise ValueError("知识库或索引版本 ID 不合法")
    return vector_store.INDEX_ROOT / knowledge_base_id / "generations" / generation_id


def validate_path(path: Path, knowledge_base_id: str, generation_id: str) -> Path:
    expected = generation_path(knowledge_base_id, generation_id).absolute()
    path = Path(path).absolute()
    if path != expected or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("拒绝清理越界路径或符号链接")
    if path.resolve() != expected:
        raise ValueError("索引路径不规范")
    return path


def write_owner(job: dict) -> None:
    path = validate_path(Path(job["candidate_index_path"]), job["runtime"]["knowledge_base_id"], job["generation_id"])
    path.mkdir(parents=True, exist_ok=True)
    owner = {key: job[key] for key in ("id", "generation_id", "old_index_path")}
    owner["knowledge_base_id"] = job["runtime"]["knowledge_base_id"]
    temporary = path / (MARKER + ".tmp")
    temporary.write_text(json.dumps(owner), encoding="utf-8")
    os.replace(temporary, path / MARKER)


def owned_path(job: dict) -> Path:
    path = validate_path(Path(job["candidate_index_path"]), job["runtime"]["knowledge_base_id"], job["generation_id"])
    if path.exists():
        owner = json.loads((path / MARKER).read_text(encoding="utf-8"))
        if any(owner.get(key) != job[key] for key in ("id", "generation_id", "old_index_path")) or owner.get("knowledge_base_id") != job["runtime"]["knowledge_base_id"]:
            raise ValueError("索引版本归属与任务不一致，已保留待核对")
    return path


def maintain(settings, jobs: dict, save) -> None:
    """Caller owns the task worker or startup, no concurrent task mutations.

    Existing KB writer lock drains readers; no model call occurs while held.
    A path lease also prevents deletion while another project CLI uses it.
    """
    from app.service import knowledge_base_lock
    from app.rebuild import safe_error

    active = settings.chroma_persist_dir.resolve()
    versions = [j for j in jobs.values() if j.get("generation_id") and j.get("candidate_index_path")
                and j.get("runtime", {}).get("knowledge_base_id") == settings.knowledge_base_id]
    active_job = next((j for j in versions if Path(j["candidate_index_path"]).resolve() == active), None)
    previous = Path(active_job["old_index_path"]).resolve() if active_job else None
    if not active_job:
        completed = sorted((j for j in versions if j["status"] == "completed" and j.get("cleanup_status") != "cleaned"), key=lambda j: j["created_at"], reverse=True)
        previous = Path(completed[0]["candidate_index_path"]).resolve() if completed else None
    protected = {active, previous} | {Path(j["candidate_index_path"]).resolve() for j in versions if j["status"] in {"queued", "indexing"}}
    # Keep the previous snapshot on disk, but release its clients after old reads.
    if previous and previous != active:
        try:
            with knowledge_base_lock.write():
                vector_store.close_vector_stores(previous)
            if active_job and active_job.get("retirement_warning"):
                active_job.pop("retirement_warning", None)
                save()
        except Exception as exc:
            if active_job:
                active_job["retirement_warning"] = safe_error(exc, settings)
            save()
    for job in versions:
        raw = Path(job["candidate_index_path"])
        if raw.resolve() in protected:
            continue
        if job.get("cleanup_status") == "cleaned" and not raw.exists():
            continue
        try:
            path = owned_path(job)
            with knowledge_base_lock.write():
                # Re-check under the same lock as publication and management reads.
                if path.resolve() == settings.chroma_persist_dir.resolve():
                    continue
                vector_store.close_vector_stores(path)
                with vector_store.index_directory_lock(path, exclusive=True, retire=True):
                    if path.exists():
                        assert_not_open(path)
                        from rag.markdown_cache import preserve_parsed_documents
                        preserve_parsed_documents(path)
                        shutil.rmtree(path)
            job.update(cleanup_status="cleaned", cleanup_error="")
            from app import audit
            audit.record("index.cleanup", actor="maintenance", job_id=job["id"],
                         knowledge_base_id=settings.knowledge_base_id, index_path=str(path))
        except Exception as exc:
            job.update(cleanup_status="pending", cleanup_error=safe_error(exc, settings))
        save()


def migrate_active(settings, jobs: dict, save) -> dict:
    """Offline rename with a durable rollback journal. No parsing/embedding.

    Caller holds the service maintenance lease; directory leases exclude CLI.
    Old absolute document sources and all index/parsed files are kept verbatim.
    """
    from app.config import update_runtime_config
    from app.uploads import utc_now
    from rag.index_manifest import read_index_embedding_model

    source = settings.chroma_persist_dir.absolute()
    root = vector_store.INDEX_ROOT.absolute()
    if source.is_relative_to(root):
        return {"source": str(source), "target": str(source), "already_managed": True}
    if source.parent != Path.cwd() or not source.name.startswith('.chroma') or any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("只迁移当前项目根目录中经过核对的旧索引，拒绝越界/符号链接")
    generation_id = uuid.uuid4().hex
    target = validate_path(generation_path(settings.knowledge_base_id, generation_id), settings.knowledge_base_id, generation_id)
    journal = root / "migration.json"
    if journal.exists():
        raise ValueError("存在未完成迁移，请先恢复")
    vector_store.close_vector_stores(source)
    assert_not_open(source)
    root.mkdir(parents=True, exist_ok=True)
    record = {"source": str(source), "target": str(target), "generation_id": generation_id, "knowledge_base_id": settings.knowledge_base_id}
    job = {"id": generation_id, "kind": "migration", "generation_id": generation_id,
           "candidate_index_path": str(target), "old_index_path": str(target),
           "runtime": {"knowledge_base_id": settings.knowledge_base_id, "embedding_model": read_index_embedding_model(source)},
           "status": "indexing", "stage": "migration", "created_at": utc_now(),
           "label": "索引目录迁移", "filenames": ["保留原索引与解析缓存"], "paths": [], "error": "",
           "files": 0, "documents": 0, "chunks": 0}
    jobs[job['id']] = job
    save()
    with vector_store.index_directory_lock(source, exclusive=True, retire=True), vector_store.index_directory_lock(target, exclusive=True):
        assert_not_open(source)
        _write_journal(journal, record)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(target)
            write_owner(job)
            update_runtime_config({"chroma_persist_dir": str(target)})
        except BaseException:
            if target.exists() and not source.exists():
                target.rename(source)
            job.update(status='failed', stage='failed', error='迁移未提交，已恢复原目录', finished_at=utc_now())
            save()
            journal.unlink()
            raise
        job.update(status='completed', stage='done', finished_at=utc_now())
        save()
        journal.unlink()
    from app import audit
    audit.record("index.migrate", actor="maintenance", **record)
    return record


def _write_journal(path: Path, record: dict):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record), encoding='utf-8')
    os.replace(temporary, path)


def recover_migration(settings, jobs: dict) -> None:
    journal = vector_store.INDEX_ROOT / "migration.json"
    if not journal.exists():
        return
    record = json.loads(journal.read_text(encoding='utf-8'))
    target = validate_path(Path(record['target']), record['knowledge_base_id'], record['generation_id'])
    job = jobs.get(record['generation_id'], {})
    if (job.get('kind') != 'migration' or job.get('candidate_index_path') != str(target)
            or job.get('runtime', {}).get('knowledge_base_id') != record['knowledge_base_id']):
        raise ValueError("迁移记录与持久化任务归属不一致，保留目录待核对")
    source = Path(record['source']).absolute()
    if source.parent != Path.cwd() or not source.name.startswith('.chroma') or any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("迁移记录源目录不安全，需要人工核对")
    with vector_store.index_directory_lock(source, exclusive=True), vector_store.index_directory_lock(target, exclusive=True):
        active = settings.chroma_persist_dir.resolve()
        if active == target.resolve() and target.is_dir() and not source.exists():
            journal.unlink()
        elif active == source.resolve():
            if target.exists() and not source.exists():
                assert_not_open(target)
                target.rename(source)
            if source.is_dir() and not target.exists():
                journal.unlink()
            else:
                raise ValueError("迁移状态不明确，保留两侧目录待人工核对")
        else:
            raise ValueError("生效配置与迁移记录不一致，保留目录待人工核对")


def adopt_failed_candidate(settings, jobs: dict, save, job_id: str, source: Path) -> dict:
    """Explicit offline reconciliation only; startup never guesses legacy ownership."""
    from datetime import datetime
    import chromadb
    from rag.index_manifest import read_index_embedding_model

    job = jobs[job_id]
    source = Path(source).absolute()
    if (job.get('kind') != 'rebuild' or job.get('status') != 'failed'
            or job.get('runtime', {}).get('knowledge_base_id') != settings.knowledge_base_id
            or source.parent != Path.cwd() or not re.fullmatch(r'\.chroma-generation-[a-f0-9]{32}', source.name)
            or any(p.is_symlink() for p in (source, *source.parents))
            or source.resolve() == settings.chroma_persist_dir.resolve()):
        raise ValueError('只接受明确指定、同知识库、已失败的旧候选目录')
    if any(j['status'] in {'queued', 'indexing'} for j in jobs.values()):
        raise ValueError('有运行中任务，不能迁移或核对旧候选')
    started = datetime.fromisoformat(job.get('started_at') or job['created_at']).timestamp()
    # Birth time is evidence in addition to explicit operator mapping and contents.
    born = getattr(source.stat(), 'st_birthtime', source.stat().st_ctime)
    if abs(born - started) > 5:
        raise ValueError('候选目录创建时间与指定任务不匹配，保留待人工核对')
    assert_not_open(source)
    assert_not_open(settings.chroma_persist_dir)
    with vector_store.metadata_client(settings.chroma_persist_dir) as client:
        live = client.get_collection(vector_store.COLLECTION_NAME).get(include=['metadatas'])
        live_ids = {m.get('doc_id') for m in live['metadatas'] or []}
    with vector_store.metadata_client(source) as client:
        try:
            collection = client.get_collection(vector_store.COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            records = {'ids': [], 'metadatas': []}
        else:
            records = collection.get(include=['metadatas'])
        if any(m.get('doc_id') not in live_ids or str(m.get('knowledge_base_id') or 'default') != settings.knowledge_base_id for m in records['metadatas'] or []):
            raise ValueError('旧候选含无法与当前文档对应的记录，禁止清理')
    generation_id = source.name.rsplit('-', 1)[1]
    target = validate_path(generation_path(settings.knowledge_base_id, generation_id), settings.knowledge_base_id, generation_id)
    job.update(generation_id=generation_id, candidate_index_path=str(target),
               old_index_path=str(settings.chroma_persist_dir.absolute()), legacy_index_path=str(source),
               legacy_reconciled=True, cleanup_status='pending')
    save()
    vector_store.close_vector_stores(source)
    with vector_store.index_directory_lock(source, exclusive=True, retire=True), vector_store.index_directory_lock(target, exclusive=True):
        assert_not_open(source)
        if target.exists():
            raise ValueError('目标版本目录已存在，保留两侧目录待核对')
        owner = {key: job[key] for key in ('id', 'generation_id', 'old_index_path')}
        owner['knowledge_base_id'] = settings.knowledge_base_id
        (source / MARKER).write_text(json.dumps(owner), encoding='utf-8')
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    return {'job_id': job_id, 'path': str(target), 'chunks': len(records['ids']),
            'model': read_index_embedding_model(target)}
