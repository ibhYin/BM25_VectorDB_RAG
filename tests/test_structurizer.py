"""Tests for document structurization via MinerU."""

from pathlib import Path
import pytest

from rag import structurizer
from rag.mineru_client import MineruError
from rag.structurizer import MINERU, convert_to_markdown


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF for testing."""
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "# Heading\n\nParagraph text.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_convert_uses_mineru_and_cleans_output(
    sample_pdf: Path, monkeypatch
) -> None:
    calls: list = []

    def fake_mineru(path, token, version):
        calls.append((path, token, version))
        return r"""# Heading

$$ \frac{1}{T} $$

正文"""

    monkeypatch.setattr(structurizer, "convert_with_mineru", fake_mineru)

    result = convert_to_markdown(sample_pdf, MINERU, mineru_token="token")

    assert calls == [(sample_pdf, "token", "vlm")]
    assert "\\frac" not in result
    assert "【公式：" in result


def test_convert_requires_mineru_token(sample_pdf: Path) -> None:
    with pytest.raises(MineruError):
        convert_to_markdown(sample_pdf, MINERU, mineru_token="")


def test_convert_rejects_unknown_provider(sample_pdf: Path) -> None:
    with pytest.raises(ValueError):
        convert_to_markdown(sample_pdf, provider="markitdown", mineru_token="t")


def test_convert_propagates_mineru_failure(
    sample_pdf: Path, monkeypatch
) -> None:
    def boom(path, token, version):
        raise RuntimeError("mineru down")

    monkeypatch.setattr(structurizer, "convert_with_mineru", boom)

    with pytest.raises(RuntimeError, match="mineru down"):
        convert_to_markdown(sample_pdf, MINERU, mineru_token="t")


def test_convert_empty_result_is_not_cached(
    sample_pdf: Path, monkeypatch
) -> None:
    calls: list = []

    def empty_mineru(path, token, version):
        calls.append(path)
        return ""

    monkeypatch.setattr(structurizer, "convert_with_mineru", empty_mineru)

    assert convert_to_markdown(sample_pdf, MINERU, mineru_token="t") == ""
    convert_to_markdown(sample_pdf, MINERU, mineru_token="t")
    # 空结果不缓存，第二次仍会真的调用 MinerU。
    assert len(calls) == 2


def test_clean_markdown_removes_dangling_image_refs() -> None:
    from rag.structurizer import clean_markdown

    md = "正文\n\n![](images/abc.jpg)\n\n继续"
    assert "![](images" not in clean_markdown(md)
    assert "正文" in clean_markdown(md)


def test_clean_markdown_normalizes_latex_math() -> None:
    from rag.structurizer import clean_markdown

    md = r"概率是 $\frac{1}{T}\sum_{t=1}^{T} x_t$，记作 $v_w$。"
    cleaned = clean_markdown(md)
    assert "\\frac" not in cleaned
    assert "\\sum" not in cleaned
    assert "1/T" in cleaned
    assert "Σ" in cleaned
    assert "【公式：" in cleaned


def test_clean_markdown_keeps_plain_dollar_text() -> None:
    from rag.structurizer import clean_markdown

    md = "营收达到 $5.6 亿美元，$2.3 亿来自海外。"
    cleaned = clean_markdown(md)
    assert "$5.6 亿美元" in cleaned  # 不含 LaTeX 特征的 $ 原样保留
    assert "【公式" not in cleaned


def test_clean_markdown_empty_and_blocks() -> None:
    from rag.structurizer import clean_markdown

    assert clean_markdown("") == ""
    block = "$$\n\\frac{1}{T}\n$$"
    cleaned = clean_markdown(block)
    assert "1/T" in cleaned
    assert "$$" not in cleaned


def test_clean_markdown_handles_latex_escapes_and_html_tags() -> None:
    from rag.structurizer import clean_markdown

    md = "COD为48.48\\~69.35 mg/L，氮氧化物58.42mg/m<sup>3</sup>。"
    cleaned = clean_markdown(md)
    assert "\\~" not in cleaned and "~" in cleaned
    assert "<sup>" not in cleaned and "³" in cleaned


def test_clean_markdown_collapses_spaced_out_formula() -> None:
    from rag.structurizer import clean_markdown

    md = r"烟尘 $< 2 0 m g / m ^ { 3 }$。"
    cleaned = clean_markdown(md)
    assert "【公式：" in cleaned
    assert "2 0 m g" not in cleaned
    assert "³" in cleaned
