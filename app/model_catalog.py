"""Discover the models a configured endpoint actually serves.

Local endpoints (loopback hosts, i.e. this machine's Ollama) are queried
through Ollama's native ``/api/tags`` because it is the only catalogue that
reports on-disk sizes. Everything else falls back to the OpenAI-compatible
``/models`` listing.
"""

from __future__ import annotations

import ipaddress
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REQUEST_TIMEOUT_SECONDS = 6.0
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_endpoint(base_url: str) -> bool:
    """Loopback means "this machine's own model server", not a remote gateway."""

    host = (urlparse(base_url).hostname or "").strip().lower()
    if not host:
        return False
    if host in LOOPBACK_HOSTS or host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_scheme(base_url: str) -> None:
    if urlparse(base_url).scheme.lower() not in ("http", "https"):
        raise ValueError("请求地址必须以 http:// 或 https:// 开头")


def _get_json(url: str, api_key: str = "") -> object:
    # Some gateways (OpenCode Zen) reject generic HTTP-library user agents, so
    # identify the client the same way the chat model does.
    from chains.models import CLIENT_USER_AGENT

    headers = {"Accept": "application/json", "User-Agent": CLIENT_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _human_size(size: object) -> str:
    if not isinstance(size, (int, float)) or size <= 0:
        return ""
    value = float(size)
    units = ("B", "KB", "MB", "GB", "TB")
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{int(value)} B" if index == 0 else f"{value:.1f} {units[index]}"


def _ollama_models(base_url: str) -> list[dict[str, object]]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    payload = _get_json(f"{root}/api/tags")
    if not isinstance(payload, dict):
        raise ValueError("Ollama 返回了非预期的响应")

    models: list[dict[str, object]] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("name") or item.get("model") or "").strip()
        if not model_id:
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        models.append(
            {
                "id": model_id,
                "size_bytes": item.get("size") if isinstance(item.get("size"), int) else None,
                "size_label": _human_size(item.get("size")),
                "parameter_size": str(details.get("parameter_size") or ""),
            }
        )
    return models


def _openai_models(base_url: str, api_key: str) -> list[dict[str, object]]:
    payload = _get_json(base_url.rstrip("/") + "/models", api_key)
    if not isinstance(payload, dict):
        raise ValueError("模型服务返回了非预期的响应")

    items = payload.get("data")
    if items is None:
        items = payload.get("models")
    models: list[dict[str, object]] = []
    for item in items or []:
        if isinstance(item, dict):
            model_id = str(item.get("id") or item.get("name") or "").strip()
        elif isinstance(item, str):
            model_id = item.strip()
        else:
            model_id = ""
        if model_id:
            models.append(
                {
                    "id": model_id,
                    "size_bytes": None,
                    "size_label": "",
                    "parameter_size": "",
                }
            )
    return models


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code in (401, 403):
            return (
                f"API Key 被拒绝（HTTP {exc.code}）"
                "；若刚改过请求地址，请先填入该地址对应的 Key 再刷新"
            )
        return f"模型服务返回 HTTP {exc.code}"
    if isinstance(exc, URLError):
        return "无法连接模型服务"
    return str(exc)


def list_models(base_url: str, api_key: str = "") -> dict[str, object]:
    """Return ``{scope, models, error}`` for one endpoint; never raises."""

    base_url = (base_url or "").strip()
    scope = "local" if is_local_endpoint(base_url) else "cloud"
    if not base_url:
        return {"scope": scope, "models": [], "error": "未配置请求地址"}
    try:
        _check_scheme(base_url)
    except ValueError as exc:
        return {"scope": scope, "models": [], "error": str(exc)}

    errors: list[str] = []
    if scope == "local":
        try:
            return {"scope": "local", "models": _ollama_models(base_url), "error": ""}
        except (OSError, ValueError) as exc:
            errors.append(_error_text(exc))

    try:
        return {"scope": scope, "models": _openai_models(base_url, api_key), "error": ""}
    except (OSError, ValueError) as exc:
        errors.append(_error_text(exc))

    return {"scope": scope, "models": [], "error": " · ".join(errors)}
