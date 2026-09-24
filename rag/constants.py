SUPPORTED_SUFFIXES = frozenset({".md", ".pdf", ".txt"})
TEXT_SUFFIXES = frozenset({".md", ".txt"})
# 视频/音频文件没有可直接入库的文本，统一先抽音频、ASR 转写后再切片。
MEDIA_SUFFIXES = frozenset(
    {
        # 视频
        ".mp4",
        ".mkv",
        ".mov",
        ".webm",
        ".avi",
        # 音频
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
    }
)
SUPPORTED_SUFFIXES = SUPPORTED_SUFFIXES | MEDIA_SUFFIXES
DEFAULT_OCR_MODEL = "gpt-4o-mini"
