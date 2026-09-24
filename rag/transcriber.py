"""Video/audio → text for ingestion.

Media files carry no ingestable text, so they go through ffmpeg (audio
extraction) plus an OpenAI-compatible ``audio/transcriptions`` endpoint (ASR),
then get split into timestamped segments. The ASR endpoint is configurable;
the project bundles no transcription provider.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

# 单次 ASR 请求的音频切片时长（秒）。16k 单声道 PCM 每 10 分钟约 19MB，
# 压在主流网关 25MB 的单请求上限之下。
TRANSCRIBE_SLICE_SECONDS = 600
# 转写片段聚合成一个 Document 的目标时长（秒）或字符数。
DOCUMENT_SEGMENT_SECONDS = 30
DOCUMENT_SEGMENT_CHARS = 400
FFMPEG_TIMEOUT_SECONDS = 1800


class TranscriptionError(RuntimeError):
    """Audio extraction or ASR failure surfaced to the ingest pipeline."""


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("需要 ffmpeg 才能提取音频，请先安装 ffmpeg")


def _run_ffmpeg(args: list[str], description: str) -> None:
    _check_ffmpeg()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"ffmpeg {description} 超时") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-5:])
        raise TranscriptionError(f"ffmpeg {description} 失败：{tail or '未知错误'}")


def _extract_audio(path: Path) -> Path:
    """Extract a 16kHz mono WAV from a video/audio file to a temp path."""

    fd, wav_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav_path = Path(wav_name)
    try:
        _run_ffmpeg(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", "-f", "wav", str(wav_path),
            ],
            "提取音频",
        )
    except Exception:
        wav_path.unlink(missing_ok=True)
        raise
    return wav_path


def _slice_wav(wav_path: Path, start: float, duration: float) -> Path:
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sliced = Path(name)
    try:
        _run_ffmpeg(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                "-f", "wav", str(sliced),
            ],
            "切片音频",
        )
    except Exception:
        sliced.unlink(missing_ok=True)
        raise
    return sliced


def _wav_duration(wav_path: Path) -> float:
    """PCM 16k mono 的时长可从文件大小推算（44 字节 wav 头）。"""

    size = max(0, wav_path.stat().st_size - 44)
    return size / 32000.0


def _transcribe_slice(
    wav_path: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or "not-needed",
        base_url=base_url or None,
        timeout=300,
        max_retries=2,
    )
    with wav_path.open("rb") as handle:
        response = client.audio.transcriptions.create(
            model=model,
            file=handle,
            response_format="verbose_json",
        )

    segments = getattr(response, "segments", None)
    if segments:
        out: list[dict[str, Any]] = []
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "start": float(getattr(segment, "start", 0) or 0),
                    "end": float(getattr(segment, "end", 0) or 0),
                    "text": text,
                }
            )
        if out:
            return out

    # 网关不支持 verbose_json 分段时，退化成整段文本（时间范围取整片音频）。
    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": _wav_duration(wav_path), "text": text}]


def transcribe_audio(
    wav_path: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> list[dict[str, Any]]:
    """Transcribe a WAV, slicing long files so each request stays small.

    Slice-local timestamps are offset by the slice start so the returned
    segments carry absolute times within the whole audio.
    """

    if not model:
        raise TranscriptionError("语音识别未配置：请设置 audio_transcribe_model")
    if not wav_path.is_file():
        raise TranscriptionError(f"音频文件不存在：{wav_path}")

    duration = _wav_duration(wav_path)
    slice_count = max(1, math.ceil(duration / TRANSCRIBE_SLICE_SECONDS))

    segments: list[dict[str, Any]] = []
    for index in range(slice_count):
        start = index * TRANSCRIBE_SLICE_SECONDS
        end = min(duration, start + TRANSCRIBE_SLICE_SECONDS)
        sliced = _slice_wav(wav_path, start, end - start)
        try:
            parts = _transcribe_slice(sliced, base_url=base_url, api_key=api_key, model=model)
        finally:
            sliced.unlink(missing_ok=True)
        for segment in parts:
            segments.append(
                {
                    "start": segment["start"] + start,
                    "end": segment["end"] + start,
                    "text": segment["text"],
                }
            )
    return segments


def _fmt_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _make_document(texts: list[str], start: float, end: float, source: str) -> Document:
    return Document(
        page_content="\n".join(texts).strip(),
        metadata={
            "source": source,
            "start_time": start,
            "end_time": end,
            "section_path": f"{_fmt_time(start)}~{_fmt_time(end)}",
        },
    )


def group_segments(segments: list[dict[str, Any]], *, source: str) -> list[Document]:
    """Group short ASR segments into Documents of roughly target duration/size.

    Each Document keeps the absolute start/end so a retrieved chunk can point
    back to a time range in the source media.
    """

    documents: list[Document] = []
    bucket: list[str] = []
    bucket_start: float | None = None
    bucket_end = 0.0
    bucket_chars = 0

    for segment in segments:
        if bucket_start is None:
            bucket_start = float(segment["start"])
        bucket_end = float(segment["end"])
        bucket_chars += len(str(segment["text"]))
        bucket.append(str(segment["text"]))
        if bucket_end - bucket_start >= DOCUMENT_SEGMENT_SECONDS or bucket_chars >= DOCUMENT_SEGMENT_CHARS:
            documents.append(_make_document(bucket, bucket_start, bucket_end, source))
            bucket = []
            bucket_start = None
            bucket_chars = 0

    if bucket and bucket_start is not None:
        documents.append(_make_document(bucket, bucket_start, bucket_end, source))
    return documents


def transcribe_media(
    path: Path,
    *,
    provider: str = "openai",
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    aliyun_access_key_id: str = "",
    aliyun_access_key_secret: str = "",
    aliyun_appkey: str = "",
    aliyun_region: str = "cn-shanghai",
    aliyun_oss_bucket: str = "",
    aliyun_oss_endpoint: str = "",
    aliyun_oss_prefix: str = "",
) -> list[Document]:
    """Full media → timestamped Documents pipeline for the loaders."""

    wav_path = _extract_audio(path)
    try:
        if provider == "aliyun":
            from rag.aliyun_transcriber import transcribe_aliyun

            segments = transcribe_aliyun(
                wav_path,
                access_key_id=aliyun_access_key_id,
                access_key_secret=aliyun_access_key_secret,
                appkey=aliyun_appkey,
                region=aliyun_region,
                oss_bucket=aliyun_oss_bucket,
                oss_endpoint=aliyun_oss_endpoint,
                oss_prefix=aliyun_oss_prefix,
            )
        else:
            segments = transcribe_audio(
                wav_path, base_url=base_url, api_key=api_key, model=model
            )
    finally:
        wav_path.unlink(missing_ok=True)
    return group_segments(segments, source=str(path))
