"""Tests for the MinerU online parsing client."""

import io
import zipfile
from pathlib import Path

import pytest
import requests

from rag import mineru_client
from rag.mineru_client import MineruError, convert_file, find_markdown_entry


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


class FakeResponse:
    """Stand-in for a requests.Response."""

    def __init__(self, payload=None, status_code: int = 200, content: bytes = b""):
        self._payload = payload
        self.status_code = status_code
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_convert_file_requires_a_token(sample_pdf: Path) -> None:
    with pytest.raises(MineruError, match="Token"):
        convert_file(sample_pdf, "")


def test_convert_file_runs_upload_poll_download_in_order(
    sample_pdf: Path, monkeypatch
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        mineru_client,
        "_request_upload_url",
        lambda path, token, version: (calls.append("url"), ("batch-1", "https://up"))[1],
    )
    monkeypatch.setattr(
        mineru_client, "_upload_file", lambda url, path: calls.append("upload")
    )
    monkeypatch.setattr(
        mineru_client,
        "_wait_for_result",
        lambda batch_id, token, max_wait: (calls.append("poll"), "https://zip")[1],
    )
    monkeypatch.setattr(
        mineru_client,
        "_download_markdown",
        lambda url: (calls.append("download"), "# markdown")[1],
    )

    assert convert_file(sample_pdf, "token", "vlm") == "# markdown"
    assert calls == ["url", "upload", "poll", "download"]


def test_request_upload_url_returns_batch_and_first_url(monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests,
        "post",
        lambda *a, **k: FakeResponse(
            {"code": 0, "data": {"batch_id": "b1", "file_urls": ["https://up/1"]}}
        ),
    )

    assert mineru_client._request_upload_url(Path("a.pdf"), "t", "vlm") == (
        "b1",
        "https://up/1",
    )


def test_request_upload_url_sends_bearer_token(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json, headers=headers)
        return FakeResponse(
            {"code": 0, "data": {"batch_id": "b1", "file_urls": ["https://up/1"]}}
        )

    monkeypatch.setattr(mineru_client.requests, "post", fake_post)

    mineru_client._request_upload_url(Path("a.pdf"), "secret", "vlm")

    assert captured["url"] == "https://mineru.net/api/v4/file-urls/batch"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["payload"] == {
        "files": [{"name": "a.pdf", "data_id": "a"}],
        "model_version": "vlm",
    }


def test_request_upload_url_surfaces_api_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests,
        "post",
        lambda *a, **k: FakeResponse({"code": 40001, "msg": "token 无效"}),
    )

    with pytest.raises(MineruError, match="token 无效"):
        mineru_client._request_upload_url(Path("a.pdf"), "t", "vlm")


def test_request_upload_url_rejects_incomplete_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests,
        "post",
        lambda *a, **k: FakeResponse({"code": 0, "data": {"batch_id": "b1"}}),
    )

    with pytest.raises(MineruError, match="不完整"):
        mineru_client._request_upload_url(Path("a.pdf"), "t", "vlm")


def test_upload_file_streams_the_handle_with_no_extra_headers(
    sample_pdf: Path, monkeypatch
) -> None:
    """The OSS pre-signed URL rejects extra headers, and streaming avoids
    loading a 200MB file into memory."""
    captured: dict[str, object] = {}

    def fake_put(url, data=None, headers=None, timeout=None):
        captured.update(url=url, data=data, headers=headers)
        return FakeResponse(status_code=200)

    monkeypatch.setattr(mineru_client.requests, "put", fake_put)

    mineru_client._upload_file("https://oss/put", sample_pdf)

    assert captured["url"] == "https://oss/put"
    assert captured["headers"] == {}
    # A file handle, not bytes: requests can then stream the body.
    assert hasattr(captured["data"], "read")


def test_upload_file_raises_on_http_error(sample_pdf: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests, "put", lambda *a, **k: FakeResponse(status_code=403)
    )

    with pytest.raises(MineruError, match="403"):
        mineru_client._upload_file("https://oss/put", sample_pdf)


def test_wait_for_result_returns_zip_url(monkeypatch) -> None:
    # One file per batch, so the API returns a single extract_result entry.
    monkeypatch.setattr(
        mineru_client.requests,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "code": 0,
                "data": {
                    "extract_result": [
                        {
                            "file_name": "a.pdf",
                            "state": "done",
                            "full_zip_url": "https://zip",
                        }
                    ]
                },
            }
        ),
    )

    assert mineru_client._wait_for_result("b1", "t", 5) == "https://zip"


def test_wait_for_result_raises_on_failed_state(monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "code": 0,
                "data": {"extract_result": [{"state": "failed", "err_msg": "页数超限"}]},
            }
        ),
    )

    with pytest.raises(MineruError, match="页数超限"):
        mineru_client._wait_for_result("b1", "t", 5)


def test_wait_for_result_times_out(monkeypatch) -> None:
    calls: list[int] = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(
            {"code": 0, "data": {"extract_result": [{"state": "pending"}]}}
        )

    monkeypatch.setattr(mineru_client.requests, "get", fake_get)

    with pytest.raises(MineruError, match="超时"):
        mineru_client._wait_for_result("b1", "t", 0)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["full.md"], "full.md"),
        (["images/x.png", "out/full.md"], "out/full.md"),
        (["content_list.json", "layout.json"], None),
    ],
)
def test_find_markdown_entry(names: list[str], expected: str | None) -> None:
    assert find_markdown_entry(names) == expected


def _zip_bytes(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in entries.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_download_markdown_extracts_full_md(monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests,
        "get",
        lambda *a, **k: FakeResponse(content=_zip_bytes({"out/full.md": "# 标题\n\n正文"})),
    )

    assert mineru_client._download_markdown("https://zip").startswith("# 标题")


def test_download_markdown_rejects_archive_without_full_md(monkeypatch) -> None:
    monkeypatch.setattr(
        mineru_client.requests,
        "get",
        lambda *a, **k: FakeResponse(content=_zip_bytes({"layout.json": "{}"})),
    )

    with pytest.raises(MineruError, match="full.md"):
        mineru_client._download_markdown("https://zip")


def test_convert_file_splits_oversized_pdf_and_merges(tmp_path, monkeypatch) -> None:
    """超过 200 页的 PDF 切成 ≤150 页的多卷，逐卷解析后带页码标记拼回。"""
    import tempfile

    import fitz

    pdf = tmp_path / "big.pdf"
    with fitz.open() as doc:
        for _ in range(300):
            doc.new_page(width=595, height=842)
        doc.save(pdf)

    # 让切出来的临时卷落在 tmp_path，方便断言清理。
    real_mkstemp = tempfile.mkstemp

    def fake_mkstemp(suffix="", **kwargs):
        fd, name = real_mkstemp(dir=str(tmp_path), suffix=suffix, **kwargs)
        return fd, name

    monkeypatch.setattr(mineru_client.tempfile, "mkstemp", fake_mkstemp)

    calls: list[int] = []

    def fake_convert_one(path, token, model_version, max_wait_seconds):
        with fitz.open(path) as doc:
            calls.append(doc.page_count)
        return f"# 第 {len(calls)} 卷内容"

    monkeypatch.setattr(mineru_client, "_convert_one", fake_convert_one)

    result = convert_file(pdf, "token", "vlm")

    # 300 页 → 2 卷，各 150 页
    assert calls == [150, 150]
    assert "分卷：第 1-150 页" in result
    assert "分卷：第 151-300 页" in result
    assert result.count("分卷") == 2
    # 临时卷全部清理，只剩原文件
    assert sorted(p.name for p in tmp_path.iterdir()) == ["big.pdf"]


def test_convert_file_leaves_small_pdf_unsplit(tmp_path, monkeypatch) -> None:
    import fitz

    pdf = tmp_path / "small.pdf"
    with fitz.open() as doc:
        for _ in range(50):
            doc.new_page(width=595, height=842)
        doc.save(pdf)

    captured: dict = {}
    monkeypatch.setattr(
        mineru_client,
        "_convert_one",
        lambda path, token, model_version, max_wait_seconds: (
            captured.update(path=path),
            "# markdown",
        )[1],
    )

    assert convert_file(pdf, "token", "vlm") == "# markdown"
    assert captured["path"] == pdf  # 没有分卷，直接传原文件
