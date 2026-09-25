"""MinerU online parsing: upload a local file, poll, and fetch the Markdown.

Flow (https://mineru.net/apiManage/docs):
  1. POST /file-urls/batch            -> batch_id + pre-signed upload URL
  2. PUT  <upload URL>                -> raw file bytes
  3. GET  /extract-results/batch/{id} -> poll until state is done
  4. GET  <full_zip_url>              -> zip whose full.md is the Markdown

The free Agent API (no token) is capped at 10MB / 20 pages, which real books
exceed, so this client targets the token-authenticated precise API.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from pathlib import Path

import requests


API_BASE = "https://mineru.net/api/v4"
REQUEST_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 5.0
MAX_WAIT_SECONDS = 1800.0
MARKDOWN_NAME = "full.md"
# MinerU 单文件有页数上限；超过就按页切分多卷解析后拼回一份。
MINERU_MAX_PAGES = 200
MINERU_PART_PAGES = 150


class MineruError(RuntimeError):
    """MinerU could not parse the document."""


def convert_file(
    path: Path,
    token: str,
    model_version: str = "vlm",
    max_wait_seconds: float = MAX_WAIT_SECONDS,
) -> str:
    """Parse one local file through MinerU and return its Markdown.

    超过 200 页的 PDF 会被 MinerU 拒绝，这里先按页切分成 ≤150 页的小卷、逐卷
    解析、再带页码标记拼回一份 Markdown。
    """

    if not token:
        raise MineruError("未配置 MinerU API Token")

    page_count = _pdf_page_count(path) if path.suffix.lower() == ".pdf" else 0
    if page_count > MINERU_MAX_PAGES:
        return _convert_split_pdf(
            path, page_count, token, model_version, max_wait_seconds
        )
    return _convert_one(path, token, model_version, max_wait_seconds)


def _pdf_page_count(path: Path) -> int:
    try:
        import fitz

        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        # 打不开就当小文件处理，由 MinerU 兜底报错。
        return 0


def _convert_split_pdf(
    path: Path,
    page_count: int,
    token: str,
    model_version: str,
    max_wait_seconds: float,
) -> str:
    """把超长 PDF 切卷、逐卷 MinerU 解析、按页码标记拼回。"""

    parts: list[tuple[Path, int, int]] = []
    try:
        for start in range(0, page_count, MINERU_PART_PAGES):
            end = min(start + MINERU_PART_PAGES, page_count)
            parts.append((_slice_pdf_pages(path, start, end), start + 1, end))
        chunks: list[str] = []
        for part_path, start_page, end_page in parts:
            markdown = _convert_one(
                part_path, token, model_version, max_wait_seconds
            )
            chunks.append(
                f"<!-- MinerU 分卷：第 {start_page}-{end_page} 页 -->\n\n{markdown}"
            )
        return "\n\n".join(chunks)
    finally:
        for part_path, _, _ in parts:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass


def _slice_pdf_pages(path: Path, start: int, end: int) -> Path:
    """把 PDF 的第 start..end-1 页（0 基）拷成一份临时 PDF。"""

    import fitz

    fd, name = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    part_path = Path(name)
    try:
        with fitz.open(path) as source, fitz.open() as out:
            out.insert_pdf(source, from_page=start, to_page=end - 1)
            out.save(part_path, garbage=3, deflate=True)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    return part_path


def _convert_one(
    path: Path,
    token: str,
    model_version: str = "vlm",
    max_wait_seconds: float = MAX_WAIT_SECONDS,
) -> str:
    """单文件 MinerU 解析（原有逻辑）。"""

    batch_id, upload_url = _request_upload_url(path, token, model_version)
    _upload_file(upload_url, path)
    zip_url = _wait_for_result(batch_id, token, max_wait_seconds)
    return _download_markdown(zip_url)


def _request_upload_url(
    path: Path, token: str, model_version: str
) -> tuple[str, str]:
    payload = {
        "files": [{"name": path.name, "data_id": path.stem[:64]}],
        "model_version": model_version,
    }
    try:
        response = requests.post(
            f"{API_BASE}/file-urls/batch",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        raise MineruError(f"申请上传地址失败：{exc}") from exc

    if body.get("code") != 0:
        raise MineruError(f"申请上传地址失败：{body.get('msg') or body}")

    info = body.get("data") or {}
    batch_id = str(info.get("batch_id") or "")
    urls = info.get("file_urls") or []
    if not batch_id or not urls:
        raise MineruError("申请上传地址的响应不完整")
    return batch_id, str(urls[0])


def _upload_file(upload_url: str, path: Path) -> None:
    """Upload raw file bytes to the pre-signed URL.

    OSS pre-signed URLs fail with urllib (403) but succeed with requests when
    using minimal headers. The signature is sensitive to User-Agent and other
    headers urllib adds by default.
    """
    try:
        with open(path, "rb") as f:
            response = requests.put(
                upload_url, data=f, headers={}, timeout=REQUEST_TIMEOUT_SECONDS
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MineruError(f"文件上传失败：{exc}") from exc


def _wait_for_result(batch_id: str, token: str, max_wait_seconds: float) -> str:
    deadline = time.monotonic() + max_wait_seconds
    while True:
        try:
            response = requests.get(
                f"{API_BASE}/extract-results/batch/{batch_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise MineruError(f"查询解析结果失败：{exc}") from exc

        if body.get("code") != 0:
            raise MineruError(f"查询解析结果失败：{body.get('msg') or body}")

        results = (body.get("data") or {}).get("extract_result") or []
        if results:
            item = results[0]
            state = str(item.get("state") or "")
            if state == "done":
                zip_url = str(item.get("full_zip_url") or "")
                if not zip_url:
                    raise MineruError("解析完成但没有返回结果地址")
                return zip_url
            if state == "failed":
                reason = str(item.get("err_msg") or "").strip()
                raise MineruError(f"MinerU 解析失败：{reason or '未提供原因'}")

        if time.monotonic() >= deadline:
            raise MineruError(f"MinerU 解析超时（超过 {int(max_wait_seconds)} 秒）")
        time.sleep(POLL_INTERVAL_SECONDS)


def _download_markdown(zip_url: str) -> str:
    try:
        response = requests.get(zip_url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.content
    except requests.RequestException as exc:
        raise MineruError(f"下载解析结果失败：{exc}") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = find_markdown_entry(archive.namelist())
            if name is None:
                raise MineruError(f"结果压缩包里没有 {MARKDOWN_NAME}")
            return archive.read(name).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise MineruError("下载到的解析结果不是有效的压缩包") from exc


def find_markdown_entry(names: list[str]) -> str | None:
    """Locate full.md in the archive, tolerating an enclosing folder."""

    for name in names:
        if name == MARKDOWN_NAME or name.endswith(f"/{MARKDOWN_NAME}"):
            return name
    return None
