"""系统问答专用通道。

关于系统本身的问题（支持什么格式、当前接入什么模型、运行版本、知识库里有哪些
内容）的答案在系统实时状态里，不在知识库文档里——走向量检索会答错或答「无法
确定」。这里用规则识别这类意图，命中就从实时状态拼答案、直接短路，不占用
LLM 信号量与读锁。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from rag.constants import MEDIA_SUFFIXES, SUPPORTED_SUFFIXES

# 运行中的代码版本：部署目录里有 DEPLOYED_COMMIT 就优先用它，否则取 git 短哈希。
# 进程启动时算一次并缓存——它标的是「这个进程跑的是哪版代码」。
def _code_version() -> str:
    try:
        marker = Path("DEPLOYED_COMMIT")
        if marker.is_file():
            text = marker.read_text(encoding="utf-8").strip()
            if text:
                return text
    except OSError:
        pass
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


CODE_VERSION = _code_version()

# --- 意图识别 ----------------------------------------------------------------
# 保守：必须在「问系统属性」才命中（models 要求 什么/哪个/接入 之类靠近 模型，
# 所以「Transformer 模型为什么好」不会命中）。命中即短路，未命中走正常检索。

_FORMAT_PATTERNS = [
    r"支持.{0,8}(什么|哪些|的)?(文件)?(格式|类型|后缀)",
    r"(哪些|什么|支持|有).{0,4}格式",
    r"格式.{0,4}(有哪些|是什么|支持哪些)",
    r"(能不能|可以|能否|能).{0,8}(上传|入库|加).{0,6}(视频|音频|文件|mp4|mp3|pdf)",
]
_MODEL_PATTERNS = [
    # 「接入/配置」是模型专属动词；或必须真的问到"模型"这个词。
    r"(接入|配置|配的|配).{0,8}(什么|哪个|哪款)(模型)?",
    r"(用的|用什么|用的是|用).{0,4}(什么|哪个).{0,2}(模型|llm|embedding|向量模型|对话模型)",
    r"(模型|llm|embedding|向量模型|对话模型).{0,6}(是什么|是哪个|用的什么|用什么|有哪些)",
    r"什么模型",
    r"(what model|which model)\??$",
]
_VERSION_PATTERNS = [
    r"(运行|当前|部署).{0,8}(版本|commit)",
    r"版本.{0,4}(是|多少|什么|几)",
    r"(what version|which version)\??$",
]
_KB_PATTERNS = [
    r"知识库.{0,12}(有哪些|什么内容|多少|几个|清单|列表)",
    r"库里.{0,8}(有哪些|什么内容|多少|几个|清单|列表)",
    r"(有哪些|什么内容|什么文档|多少篇|多少文档|几篇|几个文档|文档清单|文档列表)",
    r"(what.{0,8}(documents|docs|content)|how many documents)\??$",
]

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("formats", _FORMAT_PATTERNS),
    ("models", _MODEL_PATTERNS),
    ("version", _VERSION_PATTERNS),
    ("kb", _KB_PATTERNS),
]


def detect_system_intent(question: str) -> str | None:
    """返回命中的系统意图，未命中返回 None（走正常检索）。"""

    q = (question or "").strip()
    if not q:
        return None
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return intent
    return None


# --- 答案拼装 ----------------------------------------------------------------

_DOC_SUFFIXES = sorted(SUPPORTED_SUFFIXES - MEDIA_SUFFIXES)
_VIDEO_SUFFIXES = sorted(
    suffix for suffix in MEDIA_SUFFIXES if suffix in {".mp4", ".mkv", ".mov", ".webm", ".avi"}
)
_AUDIO_SUFFIXES = sorted(MEDIA_SUFFIXES - set(_VIDEO_SUFFIXES))


def _fmt_suffixes(suffixes: list[str]) -> str:
    return "、".join(suffix.lstrip(".") for suffix in suffixes)


def _asr_description(settings: Any) -> str:
    if settings.audio_transcribe_provider == "aliyun":
        if settings.aliyun_appkey and settings.aliyun_oss_bucket:
            return (
                f"阿里云录音文件识别（region {settings.aliyun_region or 'cn-shanghai'}，"
                f"bucket {settings.aliyun_oss_bucket}）"
            )
        return "阿里云录音文件识别（未配置完整）"
    if settings.audio_transcribe_model:
        return f"OpenAI 兼容 audio/transcriptions（{settings.audio_transcribe_model}）"
    return "未配置"


def answer_system_intent(
    intent: str, settings: Any, user: Any = None
) -> str | None:
    """从实时系统状态拼答案；未知 intent 返回 None。"""

    if intent == "formats":
        return (
            "支持入库的文件格式：\n"
            f"- 文档：{_fmt_suffixes(_DOC_SUFFIXES)}\n"
            f"- 视频：{_fmt_suffixes(_VIDEO_SUFFIXES)}（自动提取语音转写）\n"
            f"- 音频：{_fmt_suffixes(_AUDIO_SUFFIXES)}（自动转写语音）"
        )

    if intent == "models":
        return (
            "当前接入的模型：\n"
            f"- 对话模型（LLM）：{settings.llm_model}（{settings.openai_base_url or '默认端点'}）\n"
            f"- 向量模型（Embedding）：{settings.embedding_model}"
            f"（{settings.embedding_base_url or '默认端点'}）\n"
            f"- 语音转写：{_asr_description(settings)}"
        )

    if intent == "version":
        return f"当前运行版本：{CODE_VERSION}"

    if intent == "kb":
        from app.access import load_access_policy
        from app.service import list_indexed_documents

        policy = load_access_policy(settings.access_policy_file)
        is_admin = policy is not None and user is not None and policy.is_admin(user)
        documents = list_indexed_documents(settings, user=user, admin=is_admin)
        if not documents:
            return "知识库里目前还没有入库任何文档。"
        lines = "\n".join(
            f"- {doc.get('title') or doc.get('doc_id')}（{doc.get('chunk_count', 0)} 切片）"
            for doc in documents
        )
        suffix = (
            "\n（以上是你有权限查看的文档）"
            if policy is not None and not is_admin
            else ""
        )
        return f"知识库里共有 {len(documents)} 篇文档：\n{lines}{suffix}"

    return None
