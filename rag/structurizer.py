"""Turn a source document into Markdown before chunking.

The only backend is **mineru** — MinerU's online precise API. Real layout
analysis, so headings, lists and tables survive. It needs an API token.
"""

import re
from pathlib import Path

from rag import markdown_cache
from rag.mineru_client import MineruError, convert_file as convert_with_mineru


MINERU = "mineru"

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S
)
_LATEX_CMD = {
    "sum": "Σ", "prod": "Π", "int": "∫", "oint": "∮",
    "leq": "≤", "geq": "≥", "neq": "≠", "times": "×", "pm": "±", "infty": "∞",
    "log": "log", "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos",
    "prime": "′", "ldots": "…", "cdots": "⋯", "dots": "…",
    "begin": "", "end": "", "substack": "", "text": "",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "lambda": "λ",
    "mu": "μ", "sigma": "σ", "theta": "θ", "omega": "ω",
}

_SUP_TRANS = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
_SUB_TRANS = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.S)
_SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.S)
# MinerU 在普通文本里也常把 LaTeX 转义字符写出来（如 `\~` 表示"到"）。
_TEXT_ESCAPE_RE = re.compile(r"\\([~\-%&_#{}\$])")

# markdown 管道表的单元格分隔（转义 `\|` 用占位符保护，避免被当成列分隔）。
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEP_CELL_RE = re.compile(r":?-{3,}:?")


def _clean_math(expr: str) -> str:
    """把一段 LaTeX 公式压成可读文本：去反斜杠/花括号/多余空格、保留上下标结构。"""

    text = re.sub(r"\\frac\s*\{\s*([^{}]*?)\s*\}\s*\{\s*([^{}]*?)\s*\}", r"\1/\2", expr)
    text = re.sub(
        r"\\([a-zA-Z]+)", lambda m: _LATEX_CMD.get(m.group(1), ""), text
    )
    text = re.sub(r"[{}]", "", text)
    # 公式里的空格大多是 MinerU 逐字符排版造成的（`2 0 m g`），去掉；
    # 数字上下标再转成 Unicode（`^ 3` → ³、`_ 1` → ₁）。
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\^(\d)", lambda m: m.group(1).translate(_SUP_TRANS), text)
    text = re.sub(r"_(\d)", lambda m: m.group(1).translate(_SUB_TRANS), text)
    text = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]", "", text)
    return text.strip()


def _split_table_cells(line: str) -> list[str]:
    """把一行管道表拆成单元格；转义的 `\|` 不当作列分隔。"""
    line = line.strip()
    if not line.startswith("|"):
        return [line]
    inner = line[1:-1] if line.endswith("|") else line[1:]
    protected = inner.replace(r"\|", "\x00")
    return [cell.strip().replace("\x00", "|") for cell in protected.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(
        _TABLE_SEP_CELL_RE.fullmatch(cell.strip()) for cell in cells if cell.strip()
    )


def serialize_pipe_tables(markdown: str) -> str:
    """把 markdown 管道表序列化成每行自描述的「列名：值」文本。

    MinerU vlm 会把表格输出成 `| 表头 | … |` 管道表。通用文本切分可能把长表从
    中间切断，后半段的数值脱离表头（"53.53" 不知道是营收还是净利润）。这里把
    每一行串成「列名：值；…」，任何一行被独立检索到都自带列名语义。
    """

    lines = markdown.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _TABLE_ROW_RE.match(line):
            out.append(line)
            index += 1
            continue

        # 收集连续的表行
        block = [line]
        index += 1
        while index < len(lines) and _TABLE_ROW_RE.match(lines[index]):
            block.append(lines[index])
            index += 1

        # 至少要有 表头 + 分隔行 + 数据行 才值得处理；第二行必须是 `| --- |`。
        if len(block) < 3 or not _is_separator_row(_split_table_cells(block[1])):
            out.extend(block)
            continue

        header = _split_table_cells(block[0])
        for data_line in block[2:]:
            cells = _split_table_cells(data_line)
            parts: list[str] = []
            for column, cell in enumerate(cells):
                if not cell:
                    continue
                col_name = (
                    header[column].strip()
                    if column < len(header)
                    else f"列{column + 1}"
                )
                if col_name:
                    parts.append(f"{col_name}：{cell}")
                else:
                    # 左上角无表头时（如表格第一列是行标签），值原样保留
                    parts.append(cell)
            out.append("；".join(parts) if parts else data_line)
    return "\n".join(out)


def clean_markdown(markdown: str) -> str:
    """清洗 MinerU/转换产物里的多余符号。

    - 去掉悬空的图片引用（图片不会入库，`![](…)` 在知识库里没意义）
    - 把 `$…$` / `$$…$$` 等 LaTeX 公式压成可读文本并打上【公式：…】标记，
      只有真带 LaTeX 特征的（含 `\` `_` `^` `{` `}`）才处理，普通文本里的
      `$`（如金额）原样保留
    - 还原文本层的 LaTeX 转义字符（`\~` → `~` 等）
    - HTML 上下标转 Unicode（`<sup>3</sup>` → ³），并清掉残留的 HTML 标签
    - 管道表序列化成每行自描述的「列名：值」，保证切片不丢表头语义
    """

    if not markdown:
        return markdown
    text = _IMG_RE.sub("", markdown)
    text = _TEXT_ESCAPE_RE.sub(r"\1", text)
    text = _SUP_RE.sub(lambda m: m.group(1).translate(_SUP_TRANS), text)
    text = _SUB_RE.sub(lambda m: m.group(1).translate(_SUB_TRANS), text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")

    def math_repl(match: re.Match) -> str:
        expr = (
            match.group(1) or match.group(2) or match.group(3) or match.group(4) or ""
        )
        if not re.search(r"[\\_^{}]", expr):
            return match.group(0)
        cleaned = _clean_math(expr)
        return f"【公式：{cleaned}】" if cleaned else ""

    text = _MATH_RE.sub(math_repl, text)
    text = serialize_pipe_tables(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_to_markdown(
    file_path: Path,
    provider: str = MINERU,
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

    if provider != MINERU:
        raise ValueError(f"未知的结构化引擎：{provider}")
    if not mineru_token:
        raise MineruError("已选择 MinerU 解析，但尚未配置 MinerU API Token")
    markdown = convert_with_mineru(file_path, mineru_token, mineru_model_version)
    # MinerU 产物里常有 LaTeX 公式与悬空图片引用，入库前清洗成可读文本。
    markdown = clean_markdown(markdown)

    # 空结果不缓存——那多半是失败，缓存住会挡住重试。
    if markdown:
        markdown_cache.write(
            file_path, provider, mineru_model_version, markdown, str(file_path)
        )
    return markdown


__all__ = [
    "MINERU",
    "MineruError",
    "clean_markdown",
    "convert_to_markdown",
    "convert_with_mineru",
    "serialize_pipe_tables",
]
