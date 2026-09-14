"""Turn a source document into Markdown before chunking.

Two backends, chosen explicitly by the caller (``structurize_provider``):

* **mineru** — MinerU's online precise API. Real layout analysis, so headings,
  lists and tables survive. This is the only backend that structures PDFs
  usefully, and it needs an API token.
* **markitdown** — local and free, good for formats that already carry
  structure (docx/pptx/xlsx/html). For PDFs it is weaker than the built-in
  PyMuPDF path, so it is not a good default for PDF-heavy corpora.
"""

from pathlib import Path

from markitdown import MarkItDown

from rag import markdown_cache
from rag.mineru_client import MineruError, convert_file as convert_with_mineru


MARKITDOWN = "markitdown"
MINERU = "mineru"
STRUCTURIZE_PROVIDERS = (MARKITDOWN, MINERU)

_converter = MarkItDown()


def convert_to_markdown(
    file_path: Path,
    provider: str = MARKITDOWN,
    mineru_token: str = "",
    mineru_model_version: str = "vlm",
) -> str:
    """Convert one file to Markdown using the requested backend.

    Conversions are cached by file contents + converter identity, so re-indexing
    a corpus does not re-call a metered parser for files that have not changed.

    Failures propagate rather than falling back to the other backend: silently
    switching engines would feed the index text the caller did not ask for.
    """

    cached = markdown_cache.read(file_path, provider, mineru_model_version)
    if cached is not None:
        return cached

    if provider == MINERU:
        if not mineru_token:
            raise MineruError("已选择 MinerU 解析，但尚未配置 MinerU API Token")
        markdown = convert_with_mineru(file_path, mineru_token, mineru_model_version)
    elif provider == MARKITDOWN:
        markdown = convert_with_markitdown(file_path)
    else:
        raise ValueError(f"未知的结构化引擎：{provider}")

    # 空结果不缓存——那多半是失败，缓存住会挡住重试。
    if markdown:
        markdown_cache.write(
            file_path, provider, mineru_model_version, markdown, str(file_path)
        )
    return markdown


def convert_with_markitdown(file_path: Path) -> str:
    """Local markitdown conversion; "" on failure so callers can fall back."""

    try:
        result = _converter.convert(str(file_path))
        return result.text_content or ""
    except Exception:
        # Conversion failed; caller should fall back to raw extraction.
        return ""


__all__ = [
    "MARKITDOWN",
    "MINERU",
    "STRUCTURIZE_PROVIDERS",
    "MineruError",
    "convert_to_markdown",
    "convert_with_markitdown",
    "convert_with_mineru",
]
