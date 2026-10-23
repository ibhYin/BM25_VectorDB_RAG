import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.access import AccessPolicy, document_access_metadata


# 结构化文档（MinerU 等）带真实标题层级，按它切分能让切片对齐章节边界。
# 不要用在未结构化的文档上：那些 "## " 是 PyMuPDF 用长度启发式造的
# （日期、正文句子、公式都会被标成标题），按它切只会把句子从中间劈开。
HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

PLAIN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
]


def ensure_document_metadata(
    documents: list[Document],
    knowledge_base_id: str = "default",
    access_policy: AccessPolicy | None = None,
) -> list[Document]:
    """Attach stable document metadata required by indexing and retrieval."""

    for document in documents:
        source = str(document.metadata.get("source") or "unknown")
        document.metadata.setdefault("doc_id", build_doc_id(source))
        document.metadata.setdefault("title", build_title(source))
        document.metadata.setdefault("created_at", "")
        document.metadata.setdefault("source", source)
        document_access_metadata(document.metadata, knowledge_base_id, access_policy)

    return documents


def build_doc_id(source: str) -> str:
    source_path = Path(source)
    stem = source_path.stem if source_path.stem else "document"
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", stem).strip("_").lower()
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'document'}_{digest}"


def build_title(source: str) -> str:
    source_path = Path(source)
    return source_path.stem or source


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    knowledge_base_id: str = "default",
    access_policy: AccessPolicy | None = None,
    max_input_bytes: int | None = None,
) -> list[Document]:
    """Split documents into chunks suitable for retrieval.

    Documents that went through structured extraction are split on their heading
    hierarchy (so a chunk never straddles two sections, and every chunk knows its
    real section path); everything else falls back to size-based splitting.
    """

    ensure_document_metadata(documents, knowledge_base_id, access_policy)

    chunks: list[Document] = []
    for document in documents:
        size = chunk_size
        if max_input_bytes:
            # Keep the native character splitter and accurate source offsets.
            # Worst character width gives a conservative bound without assuming
            # a provider tokenizer or mixing byte overlap with character offsets.
            width = max((len(char.encode("utf-8")) for char in document.page_content), default=1)
            size = min(size, max_input_bytes // width)
        overlap = chunk_overlap * size // chunk_size
        if document.metadata.get("structured_by"):
            chunks.extend(_split_by_headers(document, size, overlap))
        else:
            chunks.extend(_split_by_size([document], size, overlap))

    chunk_counts_by_doc_id: dict[str, int] = {}
    for chunk in chunks:
        doc_id = str(chunk.metadata["doc_id"])
        chunk_counts_by_doc_id[doc_id] = chunk_counts_by_doc_id.get(doc_id, 0) + 1

        chunk.metadata["index_chunk_size"] = chunk_size
        chunk.metadata["index_chunk_overlap"] = chunk_overlap
        chunk.metadata["chunk_id"] = (
            f"{doc_id}#chunk_{chunk_counts_by_doc_id[doc_id]:03d}"
        )
        # 结构化切片在 _split_by_headers 里已经算好了绝对偏移。
        if "start_char" not in chunk.metadata:
            start_char = int(chunk.metadata.pop("start_index", 0))
            chunk.metadata["start_char"] = start_char
            chunk.metadata["end_char"] = start_char + len(chunk.page_content)
        else:
            chunk.metadata.pop("start_index", None)
        chunk.metadata.setdefault(
            "section_path", infer_section_path(chunk.page_content)
        )

    return chunks


def _split_by_size(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=PLAIN_SEPARATORS,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _split_by_headers(
    document: Document, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split on the Markdown heading hierarchy, then by size within a section.

    自己按标题行切片，而不是用 MarkdownHeaderTextSplitter：后者会规范化正文
    （`\\n\\n` 变成行尾双空格），导致切片文本在原文档里找不到，偏移量就废了——
    而「连续阅读」视图要靠偏移裁重叠。自己切还能让偏移天然精确。
    """

    text = document.page_content
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=PLAIN_SEPARATORS,
        add_start_index=True,
    )

    chunks: list[Document] = []
    title = str(document.metadata.get("title") or "")
    for start, end, section_path in _heading_sections(text):
        section_path = _drop_title_root(section_path, title)
        for piece in body_splitter.create_documents([text[start:end]]):
            offset = start + int(piece.metadata.pop("start_index", 0))
            chunks.append(
                Document(
                    page_content=piece.page_content,
                    metadata={
                        **document.metadata,
                        "section_path": section_path,
                        "start_char": offset,
                        "end_char": offset + len(piece.page_content),
                    },
                )
            )
    return chunks


def _drop_title_root(section_path: str, title: str) -> str:
    """去掉路径开头与文档标题重复的那一级。

    文档标题几乎总是 h1，而证据卡片是右截断显示的——让它占掉每行开头，
    真正有用的「2.2 Negative Sampling」就被挤没了。
    """

    if not section_path or not title:
        return section_path
    root, _, rest = section_path.partition(" > ")
    return rest if _normalise(root) == _normalise(title) else section_path


def _normalise(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def _heading_sections(text: str) -> list[tuple[int, int, str]]:
    """每个标题到下一个同级/更高级标题之间算一节。

    返回 (起, 止, 路径)，路径形如 "论文标题 > 2 方法 > 2.1 负采样"。
    """

    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [(0, len(text), "")]

    sections: list[tuple[int, int, str]] = []
    # 第一个标题之前的内容也要保留，否则会丢正文。
    if matches[0].start() > 0:
        sections.append((0, matches[0].start(), ""))

    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(
            (
                match.start(),
                matches[index + 1].start() if index + 1 < len(matches) else len(text),
                " > ".join(heading for _, heading in stack),
            )
        )
    return sections


def infer_section_path(text: str) -> str:
    headings = [
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.startswith("#")
    ]
    return " > ".join(headings)
