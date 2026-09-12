from pathlib import Path
import re
import sys

# 允许直接执行本文件时仍然能从项目根目录导入 rag 包。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz
from langchain_core.documents import Document
from rag.ocr import ocr_pdf_with_openai
from rag.constants import DEFAULT_OCR_MODEL, SUPPORTED_SUFFIXES, TEXT_SUFFIXES


# 文件类型与 PDF 扫描件检测阈值配置。
PDF_MIN_NON_EMPTY_PAGE_RATIO = 0.2
PDF_MIN_AVERAGE_CHARS = 80
PDF_MIN_TOTAL_CHARS_FOR_TEXT = 40
PDF_DETECTION_MAX_PAGES = 3
HEADING_MAX_CHARS = 80


def load_documents(
    data_dir: Path,
    openai_api_key: str = "",
    openai_base_url: str = "",
    ocr_model: str = DEFAULT_OCR_MODEL,
    ocr_max_pages: int | None = None,
    structurize: bool = False,
) -> list[Document]:
    """Load local text documents from the knowledge base directory.

    When structurize is True, PDF and plain-text files are converted to Markdown
    via markitdown before chunking, preserving headings, lists, and tables.
    """

    documents: list[Document] = []

    # 遍历知识库目录，只处理当前支持的文本和 PDF 文件。
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        documents.extend(
            load_document_file(
                path,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                ocr_model=ocr_model,
                ocr_max_pages=ocr_max_pages,
                structurize=structurize,
            )
        )

    return documents


def load_document_file(
    path: Path,
    openai_api_key: str = "",
    openai_base_url: str = "",
    ocr_model: str = DEFAULT_OCR_MODEL,
    ocr_max_pages: int | None = None,
    structurize: bool = False,
) -> list[Document]:
    """Load one supported file through the same path as directory ingestion.

    When structurize is True, uses markitdown to convert the file to structured
    Markdown before splitting. Falls back to raw text extraction on failure.
    """

    if not path.is_file():
        raise ValueError(f"Document does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {path.suffix or '(none)'}")

    if structurize:
        from rag.structurizer import convert_to_markdown

        markdown_text = convert_to_markdown(path)
        if markdown_text:
            # Structurization succeeded; return as a single document with .md metadata.
            return [
                Document(
                    page_content=markdown_text,
                    metadata={
                        "source": str(path),
                        "structurized": True,
                    },
                )
            ]
        # Conversion failed; fall through to raw extraction below.

    if path.suffix.lower() in TEXT_SUFFIXES:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()

    return load_pdf_documents(
        path,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ocr_model=ocr_model,
        ocr_max_pages=ocr_max_pages,
    )


def load_pdf_documents(
    path: Path,
    openai_api_key: str = "",
    openai_base_url: str = "",
    ocr_model: str = DEFAULT_OCR_MODEL,
    ocr_max_pages: int | None = None,
) -> list[Document]:
    # 先抽取前几页文本，用于判断 PDF 是可直接抽文本，还是需要 OCR。
    detection_documents = load_pdf_preview_documents(path)

    # 可抽文本 PDF 转成 Markdown，尽量保留章节、段落等上下文边界。
    if not should_use_ocr(detection_documents):
        return load_pdf_as_markdown_documents(path)

    # 如果疑似扫描件但没有 API Key，只能返回预览结果，避免静默丢文档。
    if not openai_api_key:
        for document in detection_documents:
            document.metadata["ocr_required"] = True
        return detection_documents

    # 扫描件走视觉模型 OCR；失败时保留预览文档并记录错误。
    try:
        return ocr_pdf_with_openai(
            path=path,
            api_key=openai_api_key,
            model=ocr_model,
            base_url=openai_base_url,
            max_pages=ocr_max_pages,
        )
    except Exception as exc:
        for document in detection_documents:
            document.metadata["ocr_error"] = str(exc)
        return detection_documents


def load_pdf_preview_documents(
    path: Path,
    max_pages: int = PDF_DETECTION_MAX_PAGES,
) -> list[Document]:
    # 优先用 PyMuPDF 做轻量预览，速度快，适合扫描件检测。
    documents = load_pdf_preview_documents_with_pymupdf(path, max_pages)

    if any(document.page_content.strip() for document in documents):
        return documents

    # 极少数 PDF 在 PyMuPDF 下抽不到文本，兜底尝试 LangChain 的 PyPDFLoader。
    from langchain_community.document_loaders import PyPDFLoader

    fallback_documents = PyPDFLoader(str(path)).load()
    return fallback_documents[:max_pages]


def load_pdf_preview_documents_with_pymupdf(
    path: Path,
    max_pages: int = PDF_DETECTION_MAX_PAGES,
) -> list[Document]:
    documents: list[Document] = []
    with fitz.open(path) as pdf:
        total_pages = pdf.page_count
        page_count = min(total_pages, max_pages)

        # 只读取前 max_pages 页，降低扫描件检测成本。
        for page_index in range(page_count):
            text = pdf[page_index].get_text().strip()
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_index,
                        "page_label": str(page_index + 1),
                        "total_pages": total_pages,
                        "loader": "pymupdf_preview",
                    },
                )
            )

    return documents


def load_pdf_as_markdown_documents(path: Path) -> list[Document]:
    """Extract text PDF pages as Markdown while preserving coarse layout boundaries."""

    documents: list[Document] = []
    with fitz.open(path) as pdf:
        total_pages = pdf.page_count

        # 每页生成一个 Markdown Document，保留页码，后续 chunk 可追溯到页面。
        for page_index in range(total_pages):
            markdown = pdf_page_to_markdown(pdf[page_index])
            documents.append(
                Document(
                    page_content=markdown,
                    metadata={
                        "source": str(path),
                        "page": page_index,
                        "page_label": str(page_index + 1),
                        "total_pages": total_pages,
                        "loader": "pymupdf_markdown",
                        "format": "markdown",
                    },
                )
            )

    return documents


def pdf_page_to_markdown(page: fitz.Page) -> str:
    # PyMuPDF 的 blocks 大致对应版面块，比纯文本更能保住段落边界。
    blocks = page.get_text("blocks", sort=True)
    paragraphs: list[str] = []

    # 每个 block 先清洗行，再按标题/正文格式输出为 Markdown 段落。
    for block in blocks:
        text = normalize_pdf_block_text(block[4])
        if not text:
            continue
        paragraphs.append(format_pdf_block_as_markdown(text))

    return "\n\n".join(paragraphs).strip()


def normalize_pdf_block_text(text: str) -> str:
    # 去掉空行和行首尾空白，减少 PDF 抽取带来的碎片噪声。
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    # 对 PDF 自动换行进行保守合并，避免把同一句话切碎。
    normalized_lines: list[str] = []
    for line in lines:
        if normalized_lines and should_merge_pdf_line(normalized_lines[-1], line):
            normalized_lines[-1] = normalized_lines[-1] + line
        else:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def should_merge_pdf_line(previous: str, current: str) -> bool:
    # 断词换行优先合并，尽量还原原句。
    if previous.endswith(("-", "－")):
        return True
    # 已有明确句末标点时保留边界，避免破坏完整语义。
    if re.search(r"[。！？；：.!?;:]$", previous):
        return False
    # 疑似标题不和正文互相吞并，保留章节边界。
    if is_probable_heading(previous) or is_probable_heading(current):
        return False
    return True


def format_pdf_block_as_markdown(text: str) -> str:
    # 疑似章节标题统一转成二级 Markdown 标题，供 splitter 优先切分。
    if is_probable_heading(text):
        heading = re.sub(r"\s+", " ", text).strip()
        return f"## {heading}"
    return text


def is_probable_heading(text: str) -> bool:
    # 标题判断保持保守：短文本、章节编号、数字编号等才升级为标题。
    compact = re.sub(r"\s+", "", text)
    if not compact or len(compact) > HEADING_MAX_CHARS:
        return False
    if "\n" in text and len(text.splitlines()) > 2:
        return False
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节篇部分]", compact):
        return True
    if re.match(r"^[0-9]+([.．][0-9]+)*[、.．\s]", text):
        return True
    if compact.startswith(("#", "（", "(")) and len(compact) <= HEADING_MAX_CHARS:
        return True
    return False


def should_use_ocr(documents: list[Document]) -> bool:
    # 没有预览文档时不贸然 OCR，让上层保留原始处理结果。
    if not documents:
        return False

    # 根据前几页的非空比例和平均字符数判断是否疑似扫描件。
    lengths = [len((document.page_content or "").strip()) for document in documents]
    total_chars = sum(lengths)
    non_empty_ratio = sum(length > 0 for length in lengths) / len(lengths)
    average_chars = sum(lengths) / len(lengths)

    # 只要已经抽到足够文本，就认为是文本 PDF，避免误走昂贵 OCR。
    if total_chars >= PDF_MIN_TOTAL_CHARS_FOR_TEXT:
        return False

    return (
        non_empty_ratio < PDF_MIN_NON_EMPTY_PAGE_RATIO
        or average_chars < PDF_MIN_AVERAGE_CHARS
    )
