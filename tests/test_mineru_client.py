"""Tests for the MinerU online parsing client."""

import io
import zipfile
from pathlib import Path

import pytest

from rag import mineru_client
from rag.mineru_client import MineruError, convert_file, find_markdown_entry


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


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
        mineru_client,
        "_upload_file",
        lambda url, path: calls.append("upload"),
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
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "code": 0,
                "data": {"batch_id": "b1", "file_urls": ["https://up/1"]},
            }

    monkeypatch.setattr(
        "requests.post", lambda *args, **kwargs: FakeResponse()
    )

    assert mineru_client._request_upload_url(Path("a.pdf"), "t", "vlm") == (
        "b1",
        "https://up/1",
    )


def test_request_upload_url_surfaces_api_errors(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"code": 40001, "msg": "token 无效"}

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(MineruError, match="token 无效"):
        mineru_client._request_upload_url(Path("a.pdf"), "t", "vlm")


def test_request_upload_url_rejects_incomplete_payload(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"code": 0, "data": {"batch_id": "b1"}}

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(MineruError, match="不完整"):
        mineru_client._request_upload_url(Path("a.pdf"), "t", "vlm")


def test_wait_for_result_returns_zip_url(monkeypatch) -> None:
    # One file per batch, so the API returns a single extract_result entry.
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "code": 0,
                "data": {
                    "extract_result": [
                        {"file_name": "a.pdf", "state": "done", "full_zip_url": "https://zip"}
                    ]
                },
            }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    assert mineru_client._wait_for_result("b1", "t", 5) == "https://zip"


def test_wait_for_result_raises_on_failed_state(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "code": 0,
                "data": {"extract_result": [{"state": "failed", "err_msg": "页数超限"}]},
            }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(MineruError, match="页数超限"):
        mineru_client._wait_for_result("b1", "t", 5)


def test_wait_for_result_times_out(monkeypatch) -> None:
    calls: list[int] = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return {"code": 0, "data": {"extract_result": [{"state": "pending"}]}}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

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


def test_download_markdown_extracts_full_md(monkeypatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("out/full.md", "# 标题\n\n正文")

    class FakeResponse:
        def raise_for_status(self):
            pass

        @property
        def content(self) -> bytes:
            return buffer.getvalue()

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    assert mineru_client._download_markdown("https://zip").startswith("# 标题")


def test_download_markdown_rejects_archive_without_full_md(monkeypatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("layout.json", "{}")

    class FakeResponse:
        def raise_for_status(self):
            pass

        @property
        def content(self) -> bytes:
            return buffer.getvalue()

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(MineruError, match="full.md"):
        mineru_client._download_markdown("https://zip")
