"""上传请求在入队前失败时也要进向量库审计。

入队后的解析/切片/向量写入失败由 `service.ingest_files` 记录；这里专门盯住请求大小、
multipart 解析和队列满——这些失败发生在 job 创建之前，后台永远看不到，必须在 Web 边界记。
"""

import io
from email.message import Message
from http import HTTPStatus

import pytest

from app.access import UserContext
from app.config import Settings
from app.web import RequestError, WebRequestHandler


def make_upload_handler(
    monkeypatch,
    settings: Settings,
    *,
    content_length: int = 10,
    body: bytes = b"upload-body",
):
    monkeypatch.setattr("app.web.get_settings", lambda: settings)

    handler = object.__new__(WebRequestHandler)
    handler.headers = Message()
    handler.headers["Content-Type"] = "multipart/form-data; boundary=x"
    handler.headers["Content-Length"] = str(content_length)
    handler.rfile = io.BytesIO(body)
    handler._upload_permission = lambda settings: (
        UserContext("alice", frozenset({"public"})),
        True,
    )
    handler._send_json = lambda *args, **kwargs: None
    return handler, settings


def test_queue_full_is_a_failed_upload_audit(tmp_path, monkeypatch) -> None:
    from app import audit

    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_bytes=100)
    monkeypatch.setattr(
        "app.web.parse_multipart",
        lambda *args, **kwargs: ([('a.pdf', b'x')], {"structurize": "on"}),
    )

    class FullQueue:
        def create_uploaded(self, *args, **kwargs):
            raise RuntimeError("Upload queue is full")

    monkeypatch.setattr("app.uploads.get_upload_store", lambda: FullQueue())
    handler, _ = make_upload_handler(monkeypatch, settings)
    with pytest.raises(RequestError) as error:
        handler._handle_document_upload()

    assert error.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    entry = audit.read_entries()[0]
    assert entry["event"] == "ingest.files"
    assert entry["status"] == "failed"
    assert entry["actor"] == "alice"
    assert entry["files"] == ["a.pdf"]
    assert entry["failure_stage"] == "queue"
    assert entry["error"] == "Upload queue is full"
    assert entry["structurize"] is True


def test_oversized_request_is_a_failed_upload_audit(tmp_path, monkeypatch) -> None:
    from app import audit

    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_bytes=5)
    handler, _ = make_upload_handler(monkeypatch, settings, content_length=10)
    with pytest.raises(RequestError) as error:
        handler._handle_document_upload()

    assert error.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    entry = audit.read_entries()[0]
    assert entry["status"] == "failed"
    assert entry["failure_stage"] == "request"
    assert entry["error"] == "Upload body is too large"
    assert entry["files"] == []


def test_bad_multipart_is_a_failed_upload_audit(tmp_path, monkeypatch) -> None:
    from app import audit

    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_bytes=100)
    monkeypatch.setattr(
        "app.web.parse_multipart",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Unsupported document type: .mp4")
        ),
    )
    handler, _ = make_upload_handler(monkeypatch, settings)
    with pytest.raises(RequestError) as error:
        handler._handle_document_upload()

    assert error.value.status == HTTPStatus.BAD_REQUEST
    entry = audit.read_entries()[0]
    assert entry["status"] == "failed"
    assert entry["failure_stage"] == "multipart_parse"
    assert entry["error"] == "Unsupported document type: .mp4"


def test_unexpected_disk_error_is_audited_and_reraised(tmp_path, monkeypatch) -> None:
    from app import audit

    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_bytes=100)
    monkeypatch.setattr(
        "app.web.parse_multipart", lambda *args, **kwargs: ([('a.pdf', b'x')], {})
    )

    class BrokenQueue:
        def create_uploaded(self, *args, **kwargs):
            raise OSError("disk full")

    monkeypatch.setattr("app.uploads.get_upload_store", lambda: BrokenQueue())
    handler, _ = make_upload_handler(monkeypatch, settings)
    with pytest.raises(OSError, match="disk full"):
        handler._handle_document_upload()

    entry = audit.read_entries()[0]
    assert entry["status"] == "failed"
    assert entry["failure_stage"] == "queue"
    assert entry["error_type"] == "OSError"
