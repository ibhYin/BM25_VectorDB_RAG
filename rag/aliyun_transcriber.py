"""阿里云录音文件识别后端（provider="aliyun"）。

录音文件识别要求音频走公网 URL，所以先把音频上传到 OSS（私有 bucket 用签名
URL），再提交识别任务并按 TaskId 轮询结果。结果带句子时间戳，直接复用到现有
转写分段管线。全程只依赖 oss2 与 aliyun-python-sdk-core（签名由 SDK 处理）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from rag.transcriber import TranscriptionError, _run_ffmpeg

logger = logging.getLogger(__name__)

# 录音文件识别（经典 RPC 接口，2018-08-17 版）：提交用 SubmitTask（Task 参数带
# 任务 JSON），查询用 GetTaskResult。版本/动作名已用真实 AK 实测确认。
_ALIYUN = {
    "submit": "SubmitTask",
    "result": "GetTaskResult",
    "version": "2018-08-17",
}
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 1800
# 签名 URL 有效期：给 filetrans 拉取音频留足余量，防止 401。
SIGN_URL_EXPIRES = 3600 * 12
# 12 小时的 16k 单声道 wav 约 1.38GB，超音频 512MB 上限；压成 64k mp3 约 345MB。
MP3_BITRATE = "64k"
# 识别结果的成功状态码。
_SUCCESS_CODE = 21050000


def _validate_config(**kwargs: str) -> None:
    missing = [name for name, value in kwargs.items() if not str(value or "").strip()]
    if missing:
        raise TranscriptionError(
            "阿里云录音文件识别未配置完整：缺少 " + "、".join(missing)
        )


def _encode_upload_audio(wav_path: Path) -> Path:
    """把 16k 单声道 wav 压成 64kbps mp3，控制 OSS 单文件体积。"""

    fd, name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    mp3_path = Path(name)
    try:
        _run_ffmpeg(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
                "-ac", "1", "-ar", "16000", str(mp3_path),
            ],
            "转码音频",
        )
    except Exception:
        mp3_path.unlink(missing_ok=True)
        raise
    return mp3_path


def _upload_to_oss(
    mp3_path: Path,
    *,
    access_key_id: str,
    access_key_secret: str,
    region: str,
    oss_bucket: str,
    oss_endpoint: str,
    oss_prefix: str,
) -> tuple[Any, str, str]:
    import oss2

    endpoint = oss_endpoint or f"https://oss-{region}.aliyuncs.com"
    bucket = oss2.Bucket(
        oss2.Auth(access_key_id, access_key_secret), endpoint, oss_bucket
    )
    key = f"{oss_prefix}{uuid.uuid4().hex}.mp3".lstrip("/")
    with mp3_path.open("rb") as handle:
        bucket.put_object(key, handle)
    url = bucket.sign_url("GET", key, SIGN_URL_EXPIRES)
    return bucket, key, url


def _rpc_client(access_key_id: str, access_key_secret: str, region: str) -> Any:
    from aliyunsdkcore.client import AcsClient

    return AcsClient(access_key_id, access_key_secret, region)


def _submit_task(
    client: Any, *, appkey: str, file_link: str, region: str
) -> str:
    from aliyunsdkcore.request import CommonRequest

    request = CommonRequest()
    request.set_domain(f"filetrans.{region}.aliyuncs.com")
    request.set_version(_ALIYUN["version"])
    request.set_action_name(_ALIYUN["submit"])
    request.set_method("POST")
    request.set_protocol_type("https")
    request.set_accept_format("json")
    task = {
        "appkey": appkey,
        "file_link": file_link,
        "version": "4.0",
        "enable_words": False,
    }
    request.add_query_param("Task", json.dumps(task, ensure_ascii=False))
    response = json.loads(client.do_action_with_exception(request).decode("utf-8"))
    task_id = response.get("TaskId") or response.get("Taskid")
    if not task_id:
        raise TranscriptionError(f"提交录音文件识别任务失败：{response}")
    return str(task_id)


def _get_task_result(client: Any, task_id: str, region: str) -> dict[str, Any]:
    from aliyunsdkcore.request import CommonRequest

    request = CommonRequest()
    request.set_domain(f"filetrans.{region}.aliyuncs.com")
    request.set_version(_ALIYUN["version"])
    request.set_action_name(_ALIYUN["result"])
    request.set_method("GET")
    request.set_protocol_type("https")
    request.set_accept_format("json")
    request.add_query_param("TaskId", task_id)
    return json.loads(client.do_action_with_exception(request).decode("utf-8"))


def _poll_result(
    client: Any,
    task_id: str,
    region: str,
    poll_interval: int,
    poll_timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + poll_timeout
    while True:
        try:
            response = _get_task_result(client, task_id, region)
        except Exception as exc:
            # HTTP/签名层错误与任务状态无关，退避重试；直到死线才判超时。
            if time.monotonic() >= deadline:
                raise TranscriptionError(
                    f"录音文件识别轮询超时：{exc}"
                ) from exc
            time.sleep(min(poll_interval, 5))
            continue

        status = str(response.get("StatusText", "")).upper()
        code = response.get("StatusCode")
        if status == "SUCCESS" or code == _SUCCESS_CODE:
            return response
        if status in {"FAILED", "ERROR"}:
            detail = str(response.get("Result") or "")
            raise TranscriptionError(
                f"录音文件识别任务失败：{status} ({code}) {detail[:200]}".strip()
            )
        # 其余（RUNNING/QUEUEING/WAIT 或未知状态）继续轮询，死线兜底。
        if time.monotonic() >= deadline:
            raise TranscriptionError("录音文件识别轮询超时")
        time.sleep(poll_interval)


def _parse_result(response: dict[str, Any]) -> list[dict[str, Any]]:
    result = response.get("Result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            result = None
    result = result if isinstance(result, dict) else {}

    segments: list[dict[str, Any]] = []
    for sentence in result.get("Sentences") or []:
        text = str(sentence.get("Text") or sentence.get("text") or "").strip()
        if not text:
            continue
        begin = float(sentence.get("BeginTime") or 0) / 1000.0
        end = float(sentence.get("EndTime") or begin) / 1000.0
        segments.append({"start": begin, "end": end, "text": text})

    # 无句子时间戳时退化成整段文本（时间范围未知，靠上层字符阈值分桶）。
    if not segments:
        text = str(result.get("Result") or "").strip()
        if text:
            segments.append({"start": 0.0, "end": 0.0, "text": text})
    return segments


def transcribe_aliyun(
    wav_path: Path,
    *,
    access_key_id: str,
    access_key_secret: str,
    appkey: str,
    region: str,
    oss_bucket: str,
    oss_endpoint: str = "",
    oss_prefix: str = "",
    poll_interval: int = POLL_INTERVAL_SECONDS,
    poll_timeout: int = POLL_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """完整阿里链路：压 mp3 → 传 OSS → 提交 → 轮询 → 解析句子。

    返回与 OpenAI 路径相同的 segments 列表。OSS 临时对象与本地 mp3 都在 finally
    里清理，删除失败只记日志、不掩盖真实错误。
    """

    _validate_config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        appkey=appkey,
        region=region,
        oss_bucket=oss_bucket,
    )
    if not wav_path.is_file():
        raise TranscriptionError(f"音频文件不存在：{wav_path}")

    mp3_path = _encode_upload_audio(wav_path)
    bucket: Any = None
    key = ""
    try:
        bucket, key, url = _upload_to_oss(
            mp3_path,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region=region,
            oss_bucket=oss_bucket,
            oss_endpoint=oss_endpoint,
            oss_prefix=oss_prefix,
        )
        client = _rpc_client(access_key_id, access_key_secret, region)
        task_id = _submit_task(client, appkey=appkey, file_link=url, region=region)
        response = _poll_result(client, task_id, region, poll_interval, poll_timeout)
        return _parse_result(response)
    finally:
        if bucket is not None and key:
            try:
                bucket.delete_object(key)
            except Exception as exc:  # 清理失败不能掩盖识别错误
                logger.warning("删除 OSS 临时音频失败 %s: %s", key, exc)
        mp3_path.unlink(missing_ok=True)
