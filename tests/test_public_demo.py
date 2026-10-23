"""README quick start in a fresh workspace, with deterministic local fake models."""
import json
import re
import shutil
from pathlib import Path

from typer.testing import CliRunner

from app import config, main, service, uploads
from rag import vector_store

ROOT = Path(__file__).resolve().parents[1]


class LocalEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 2.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_readme_first_ingest_migrate_and_question(tmp_path, monkeypatch):
    from chains import answer_chain

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CHROMA_PERSIST_DIR', '.chroma')
    monkeypatch.setenv('ANONYMIZED_TELEMETRY', 'False')
    monkeypatch.setattr(vector_store, 'INDEX_ROOT', tmp_path / 'data/indexes')
    monkeypatch.setattr(config, '_RUNTIME_OVERRIDES', {})
    monkeypatch.setattr(uploads, '_upload_store', None)
    shutil.copy(ROOT / '.env.example', '.env')
    shutil.copytree(ROOT / 'examples/documents', tmp_path / 'data/raw')
    config.get_settings.cache_clear()
    settings = config.get_settings()
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr(vector_store, 'get_embeddings', lambda *args: LocalEmbeddings())
    monkeypatch.setattr(service, 'get_embeddings', lambda *args: LocalEmbeddings())

    def answer(**kwargs):
        context = '\n'.join(kwargs['vector_context'])
        assert '14 天' in context
        chunk_id = re.search(r'chunk_id: ([^\n]+)', context).group(1)
        return f'星河项目试用期为 14 天。[source: {chunk_id}]'

    monkeypatch.setattr(answer_chain, 'generate_answer', answer)
    runner = CliRunner()
    try:
        for arguments in (['ingest'], ['maintain-indexes', '--migrate-legacy'],
                          ['ask', '星河项目的试用期是多少天？']):
            result = runner.invoke(main.app, arguments)
            assert result.exit_code == 0, (arguments, result.output, result.exception)
        assert '14 天' in result.output and '[source:' in result.output
        assert not (tmp_path / '.chroma').exists()
        assert settings.chroma_persist_dir.is_relative_to(tmp_path / 'data/indexes')
        saved = json.loads(config.runtime_config_path().read_text())
        assert Path(saved['chroma_persist_dir']) == settings.chroma_persist_dir
        records = service.index_management_store(settings).get(include=['documents', 'metadatas'])
        assert len({m['doc_id'] for m in records['metadatas']}) == 2
        # Public references must be exact evidence from the self-authored documents.
        for case in json.loads((ROOT / 'examples/evaluation.json').read_text()):
            assert case['evidence'] in (ROOT / 'examples/documents' / case['document']).read_text()
    finally:
        vector_store.close_vector_stores(settings.chroma_persist_dir)
        config.get_settings.cache_clear()


def test_mineru_diagnostic_redacts_auth_and_signed_urls(monkeypatch, capsys):
    import io
    import runpy
    import sys
    import urllib.error
    import urllib.request
    import pytest

    token = 'test-secret-never-print-this-value'
    settings = config.Settings(_env_file=None, mineru_api_token=token)
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    def fail(request, **kwargs):
        assert request.get_header('Authorization') == 'Bearer ' + token
        body = json.dumps({'token': token, 'url': 'https://example.invalid/upload?signature=private-signed-query'}).encode()
        raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, io.BytesIO(body))
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / 'scripts/check_mineru_token.py'), run_name='__main__')
    assert stopped.value.code == 1
    output = capsys.readouterr().out
    assert token not in output and token[:15] not in output and token[-10:] not in output
    assert 'private-signed-query' not in output
    assert '403' in output
