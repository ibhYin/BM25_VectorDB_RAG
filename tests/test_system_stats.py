import app.system_stats as stats
from app.web import WebRequestHandler


class _FakePath:
    """把 /proc/... 重定向到 tmp_path 下的同名相对路径。"""

    root = None

    def __init__(self, value):
        self.value = str(value)

    def read_text(self):
        relative = self.value.lstrip("/")
        return (type(self).root / relative).read_text()


def test_parse_proc_snapshot(monkeypatch, tmp_path) -> None:
    # _FakePath 把 /proc/... 映射到 tmp_path/proc/...，所以文件要按这个结构放。
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "self").mkdir()
    (proc / "loadavg").write_text("0.42 0.31 0.28 1/351 9377\n")
    (proc / "meminfo").write_text(
        "MemTotal:        1780812 kB\n"
        "MemAvailable:     590340 kB\n"
        "SwapTotal:       2097152 kB\n"
        "SwapFree:        1800000 kB\n"
    )
    (proc / "self" / "statm").write_text("100 50 30 10 0 0 0\n")
    _FakePath.root = tmp_path
    monkeypatch.setattr(stats, "Path", _FakePath)
    monkeypatch.setattr(stats, "_stats_cache", None)
    monkeypatch.setattr(stats, "_embed_cache", (0.0, None))

    snap = stats.collect_system_stats(data_dir=str(tmp_path))

    assert snap["cpu_load"] == 0.42
    assert snap["cpu_cores"] is not None
    memory = snap["memory"]
    assert memory["total"] == 1780812 * 1024
    assert memory["available"] == 590340 * 1024
    assert memory["used"] == (1780812 - 590340) * 1024
    assert memory["swap_total"] == 2097152 * 1024
    assert snap["process_rss"] == 50 * _page_size()
    assert snap["embedding_online"] is None  # 没配 base_url，不探活


def _page_size():
    import os

    return os.sysconf("SC_PAGE_SIZE")


def test_collect_degrades_without_proc(monkeypatch, tmp_path) -> None:
    # 显式模拟「读不到 /proc」：不依赖跑测试的机器有没有 /proc。
    monkeypatch.setattr(stats, "_cpu_load", lambda: None)
    monkeypatch.setattr(stats, "_memory", lambda: None)
    monkeypatch.setattr(stats, "_process_rss", lambda: None)
    stats._stats_cache = None
    stats._embed_cache = None
    snap = stats.collect_system_stats(data_dir=str(tmp_path))
    # 缺失的字段应为 None，磁盘用 statvfs 仍可取，整体不抛异常。
    assert snap["cpu_load"] is None
    assert snap["memory"] is None
    assert snap["process_rss"] is None
    assert snap["embedding_online"] is None
    assert "cpu_cores" in snap and "disk" in snap and "collected_at" in snap


def test_embedding_online_cached(monkeypatch) -> None:
    calls: list[str] = []

    def fake_probe(url):
        calls.append(url)
        return True

    monkeypatch.setattr(stats, "_probe_embedding", fake_probe)
    monkeypatch.setattr(stats, "_embed_cache", None)
    assert stats.embedding_online("http://127.0.0.1:11434/v1") is True
    assert stats.embedding_online("http://127.0.0.1:11434/v1") is True
    assert calls == ["http://127.0.0.1:11434/v1"]  # 缓存：原始地址只探一次


def test_embedding_probe_url_construction(monkeypatch) -> None:
    # "/models" 在 _probe_embedding 内部拼接，这里验证真实探活路径。
    seen: dict = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(url, timeout=2):
        seen["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(stats, "urlopen", fake_urlopen)
    assert stats._probe_embedding("http://127.0.0.1:11434/v1") is True
    assert seen["url"] == "http://127.0.0.1:11434/v1/models"


def test_embedding_probe_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        stats, "_probe_embedding", lambda url: False if url else None
    )
    monkeypatch.setattr(stats, "_embed_cache", None)
    assert stats.embedding_online("http://127.0.0.1:1/v1") is False


def test_system_stats_endpoint() -> None:
    stats._stats_cache = None
    stats._embed_cache = None
    handler = object.__new__(WebRequestHandler)
    handler.captured = {}
    handler._send_json = lambda payload, status=200: handler.captured.update(
        payload=payload, status=status
    )
    WebRequestHandler._handle_system_stats(handler)
    assert handler.captured["status"] == 200
    payload = handler.captured["payload"]
    assert "cpu_load" in payload
    assert "memory" in payload
    assert "disk" in payload
    assert "embedding_online" in payload
    assert "collected_at" in payload
