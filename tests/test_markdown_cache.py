"""The content-addressed cache for converted Markdown."""

import json
from pathlib import Path

import pytest

from rag import markdown_cache


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4 original")
    return path


def test_write_then_read_round_trip(pdf: Path) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# 标题\n\n正文", source=str(pdf))

    assert markdown_cache.read(pdf, "mineru", "vlm") == "# 标题\n\n正文"


def test_a_miss_on_an_unknown_key_is_none(pdf: Path) -> None:
    assert markdown_cache.read(pdf, "mineru", "vlm") is None


def test_the_key_follows_contents_not_the_path(pdf: Path, tmp_path: Path) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# 内容")
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(pdf.read_bytes())

    # Same bytes under a new name still hits — renames must not cost an API call.
    assert markdown_cache.read(renamed, "mineru", "vlm") == "# 内容"


def test_editing_the_file_invalidates_the_entry(pdf: Path) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# 旧内容")
    pdf.write_bytes(b"%PDF-1.4 edited")

    assert markdown_cache.read(pdf, "mineru", "vlm") is None


def test_provider_and_version_are_part_of_the_key(pdf: Path) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# mineru vlm")

    assert markdown_cache.read(pdf, "markitdown", "vlm") is None
    assert markdown_cache.read(pdf, "mineru", "pipeline") is None


def test_provider_boundaries_cannot_collide(pdf: Path) -> None:
    # Length-prefixing means ("ab", "c") and ("a", "bc") must differ.
    assert markdown_cache.cache_key(pdf, "ab", "c") != markdown_cache.cache_key(
        pdf, "a", "bc"
    )


def test_empty_markdown_is_not_cached(pdf: Path) -> None:
    markdown_cache.write(pdf, "markitdown", "", "")

    # Caching a blank result would block a retry that might succeed.
    assert markdown_cache.read(pdf, "markitdown", "") is None


def test_a_corrupt_entry_reads_as_a_miss(pdf: Path, tmp_path: Path) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# 内容")
    entry = next(markdown_cache.cache_dir().glob("*.json"))
    entry.write_text("{not json", encoding="utf-8")

    assert markdown_cache.read(pdf, "mineru", "vlm") is None


def test_the_entry_is_self_describing(pdf: Path) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# 内容", source="data/raw/paper.pdf")

    payload = json.loads(next(markdown_cache.cache_dir().glob("*.json")).read_text("utf-8"))
    assert payload["source"] == "data/raw/paper.pdf"
    assert payload["provider"] == "mineru"
    assert payload["model_version"] == "vlm"
    assert payload["chars"] == len("# 内容")
    assert payload["cached_at"]
    assert payload["markdown"] == "# 内容"


def test_disabling_skips_the_cache_entirely(pdf: Path, monkeypatch) -> None:
    markdown_cache.write(pdf, "mineru", "vlm", "# 内容")
    monkeypatch.setenv("MARKDOWN_CACHE_DISABLED", "1")

    assert markdown_cache.read(pdf, "mineru", "vlm") is None


def test_read_and_write_never_raise_on_a_bad_location(pdf: Path, monkeypatch) -> None:
    monkeypatch.setenv("MARKDOWN_CACHE_DIR", "/proc/definitely/not/writable")

    # Neither call may break an ingest.
    assert markdown_cache.read(pdf, "mineru", "vlm") is None
    markdown_cache.write(pdf, "mineru", "vlm", "# 内容")


def test_info_counts_entries(pdf: Path) -> None:
    assert markdown_cache.info()["entries"] == 0

    markdown_cache.write(pdf, "mineru", "vlm", "# 内容")

    info = markdown_cache.info()
    assert info["entries"] == 1
    assert int(info["bytes"]) > 0
    assert info["dir"]


# --- integration with the structurizer ------------------------------------


def test_structurizer_serves_the_second_call_from_cache(pdf: Path, monkeypatch) -> None:
    from rag import structurizer

    calls = []

    def fake_markitdown(path):
        calls.append(path)
        return "# 解析结果"

    monkeypatch.setattr(structurizer, "convert_with_markitdown", fake_markitdown)

    first = structurizer.convert_to_markdown(pdf, "markitdown")
    second = structurizer.convert_to_markdown(pdf, "markitdown")

    assert first == second == "# 解析结果"
    # 第二次没有再跑转换 —— 这正是重新入库时省下 MinerU 额度的地方。
    assert len(calls) == 1


def test_structurizer_does_not_cache_a_miss_that_would_need_retrying(
    pdf: Path, monkeypatch
) -> None:
    from rag import structurizer

    calls = []

    def fake_markitdown(path):
        calls.append(path)
        return ""

    monkeypatch.setattr(structurizer, "convert_with_markitdown", fake_markitdown)

    assert structurizer.convert_to_markdown(pdf, "markitdown") == ""
    structurizer.convert_to_markdown(pdf, "markitdown")

    # 空结果不缓存，所以第二次仍然真的去转换了。
    assert len(calls) == 2
