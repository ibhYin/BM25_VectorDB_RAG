"""极简服务器资源快照：读 /proc 与 statvfs，供网页头部状态条展示。

只做「看板」，不做历史存储；每次调用返回当前快照。非 Linux（没有 /proc）
时相关字段返回 None，调用方按「未知」展示，绝不因读取失败抛异常。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock
from urllib.request import urlopen

_STATS_TTL_SECONDS = 2.0
_EMBED_PROBE_TTL_SECONDS = 15.0

_stats_lock = Lock()
_stats_cache: tuple[float, dict] | None = None
_embed_lock = Lock()
_embed_cache: tuple[float, bool | None] | None = None


def _read_float_lines(path: str, key: str | None = None) -> list[float]:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    if key is None:
        return [float(item) for item in lines[0].split() if _is_float(item)]
    for line in lines:
        if line.startswith(key + ":"):
            parts = line.split()
            if len(parts) >= 2 and _is_float(parts[1]):
                return [float(parts[1])]
    return []


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _cpu_load() -> float | None:
    # /proc/loadavg: "0.30 0.31 0.28 1/351 9377"
    fields = _read_float_lines("/proc/loadavg")
    return fields[0] if fields else None


def _memory() -> dict | None:
    # kB -> bytes
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if not line or ":" not in line:
                continue
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                number = rest.strip().split()
                if number and number[0].isdigit():
                    info[key] = int(number[0]) * 1024
    except OSError:
        return None
    if not info.get("MemTotal"):
        return None
    total = info["MemTotal"]
    available = info.get("MemAvailable", 0)
    used = max(0, total - available)
    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    return {
        "total": total,
        "available": available,
        "used": used,
        "percent": round(used / total * 100, 1),
        "swap_total": swap_total,
        "swap_used": max(0, swap_total - swap_free),
    }


def _disk_usage(directory: str) -> dict | None:
    try:
        stat = os.statvfs(directory)
    except OSError:
        return None
    total = stat.f_blocks * stat.f_frsize
    free = stat.f_bavail * stat.f_frsize
    used = total - free
    if total <= 0:
        return None
    return {
        "total": total,
        "free": free,
        "used": used,
        "percent": round(used / total * 100, 1),
    }


def _process_rss() -> int | None:
    # /proc/self/statm: size resident shared ...（页数）
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _probe_embedding(base_url: str) -> bool | None:
    """向量服务是否在线；读不到/超时/无配置都返回 None（未知）。"""
    if not base_url:
        return None
    probe = base_url.rstrip("/") + "/models"
    try:
        with urlopen(probe, timeout=2) as response:  # noqa: S310 - 内网地址
            return 200 <= response.status < 300
    except OSError:
        return False


def embedding_online(base_url: str) -> bool | None:
    """带 TTL 缓存的向量服务探活，避免每个浏览器轮询都发一次请求。"""
    global _embed_cache
    now = time.monotonic()
    with _embed_lock:
        if _embed_cache is not None and now - _embed_cache[0] < _EMBED_PROBE_TTL_SECONDS:
            return _embed_cache[1]
    result = _probe_embedding(base_url)
    with _embed_lock:
        _embed_cache = (now, result)
    return result


def collect_system_stats(
    embedding_base_url: str = "", data_dir: str | None = None
) -> dict:
    """整份快照带 2 秒缓存，多个标签页轮询时只读一次 /proc。"""
    global _stats_cache
    now = time.monotonic()
    with _stats_lock:
        if _stats_cache is not None and now - _stats_cache[0] < _STATS_TTL_SECONDS:
            return _stats_cache[1]
    directory = data_dir or str(Path.cwd() / "data")
    snapshot = {
        "cpu_load": _cpu_load(),
        "cpu_cores": os.cpu_count(),
        "memory": _memory(),
        "disk": _disk_usage(directory),
        "process_rss": _process_rss(),
        "embedding_online": embedding_online(embedding_base_url),
        "collected_at": time.time(),
    }
    with _stats_lock:
        _stats_cache = (now, snapshot)
    return snapshot
