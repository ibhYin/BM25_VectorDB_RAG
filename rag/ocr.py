from __future__ import annotations

import base64
from pathlib import Path

import fitz
from langchain_core.documents import Document
from openai import OpenAI


OCR_PROMPT = (
    "请对这张 PDF 页面图片进行 OCR，只输出页面中的正文文字。"
    "保留自然段、标题、列表顺序；不要解释、不要总结、不要添加图片中不存在的内容。"
    "如果页面没有可识别文字，请输出空字符串。"
)


def ocr_pdf_with_openai(
    path: Path,
    api_key: str,
    model: str,
    base_url: str = "",
    max_pages: int | None = None,
    dpi: int = 160,
) -> list[Document]:
    """Render a PDF into page images and OCR each page with OpenAI."""

    documents: list[Document] = []
    kwargs = {"base_url": base_url} if base_url else {}
    client = OpenAI(api_key=api_key, timeout=60, max_retries=2, **kwargs)

    with fitz.open(path) as pdf:
        total_pages = pdf.page_count
        page_count = min(total_pages, max_pages) if max_pages else total_pages

        for page_index in range(page_count):
            image_data = render_pdf_page_png(pdf[page_index], dpi)
            text = ocr_image_with_openai(client, image_data, model)
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": str(path),
                        "page": page_index,
                        "page_label": str(page_index + 1),
                        "total_pages": total_pages,
                        "loader": "openai_ocr",
                    },
                )
            )

    return documents


def render_pdf_page_png(page: fitz.Page, dpi: int) -> bytes:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def ocr_image_with_openai(client: OpenAI, image_data: bytes, model: str) -> str:
    data_url = f"data:image/png;base64,{base64.b64encode(image_data).decode('ascii')}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""
