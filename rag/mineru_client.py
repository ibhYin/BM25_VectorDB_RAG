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
import json
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://mineru.net/api/v4"
REQUEST_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 5.0
MAX_WAIT_SECONDS = 1800.0
MARKDOWN_NAME = "full.md"
# MinerU is not a coding-agent gateway, but generic HTTP-library user agents are
# blocked by plenty of WAFs; identify the client the same way the rest of the app does.
CLIENT_USER_AGENT = "rag-langchain-workspace/1.0"


class MineruError(RuntimeError):
    """MinerU could not parse the document."""


def convert_file(
    path: Path,
    token: str,
    model_version: str = "vlm",
    max_wait_seconds: float = MAX_WAIT_SECONDS,
) -> str:
    """Parse one local file through MinerU and return its Markdown."""

    if not token:
        raise MineruError("未配置 MinerU API Token")

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
    body = _post_json(f"{API_BASE}/file-urls/batch", payload, token)
    if body.get("code") != 0:
        raise MineruError(f"申请上传地址失败：{body.get('msg') or body}")

    info = body.get("data") or {}
    batch_id = str(info.get("batch_id") or "")
    urls = info.get("file_urls") or []
    if not batch_id or not urls:
        raise MineruError("申请上传地址的响应不完整")
    return batch_id, str(urls[0])


def _upload_file(upload_url: str, path: Path) -> None:
    request = Request(
        upload_url,
        data=path.read_bytes(),
        method="PUT",
        headers={
            "Content-Type": "application/octet-stream",
            "User-Agent": CLIENT_USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status >= 300:
                raise MineruError(f"文件上传失败：HTTP {response.status}")
    except HTTPError as exc:
        raise MineruError(f"文件上传失败：HTTP {exc.code}") from exc
    except URLError as exc:
        raise MineruError("上传文件时无法连接 MinerU") from exc


def _wait_for_result(batch_id: str, token: str, max_wait_seconds: float) -> str:
    deadline = time.monotonic() + max_wait_seconds
    while True:
        body = _get_json(f"{API_BASE}/extract-results/batch/{batch_id}", token)
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
    request = Request(
        zip_url, method="GET", headers={"User-Agent": CLIENT_USER_AGENT}
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (HTTPError, URLError, OSError) as exc:
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


def _post_json(url: str, payload: dict[str, object], token: str) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": CLIENT_USER_AGENT,
        },
    )
    return _read_json(request)


def _get_json(url: str, token: str) -> dict[str, object]:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": CLIENT_USER_AGENT,
        },
    )
    return _read_json(request)


def _read_json(request: Request) -> dict[str, object]:
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise MineruError(f"MinerU 拒绝了 Token（HTTP {exc.code}）") from exc
        raise MineruError(f"MinerU 返回 HTTP {exc.code}") from exc
    except URLError as exc:
        raise MineruError("无法连接 MinerU 服务") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise MineruError(f"MinerU 响应无法解析：{exc}") from exc

    if not isinstance(body, dict):
        raise MineruError("MinerU 返回了非预期的响应格式")
    return body
