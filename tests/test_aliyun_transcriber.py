import json
import sys
import types

import pytest

from rag import aliyun_transcriber as aliyun
from rag.aliyun_transcriber import TranscriptionError, transcribe_aliyun


class FakeCommonRequest:
    """替代 aliyunsdkcore.request.CommonRequest：只记录参数供 fake client 读。"""

    def __init__(self) -> None:
        self._action = ""

    def set_domain(self, value):  # noqa: ARG002
        return None

    def set_version(self, value):  # noqa: ARG002
        return None

    def set_action_name(self, value: str) -> None:
        self._action = value

    def set_method(self, value):  # noqa: ARG002
        return None

    def set_protocol_type(self, value):  # noqa: ARG002
        return None

    def set_accept_format(self, value):  # noqa: ARG002
        return None

    def add_query_param(self, key, value):  # noqa: ARG002
        return None

    def get_action_name(self) -> str:
        return self._action


def _fake_aliyun_sdk(monkeypatch) -> None:
    """让函数内 `from aliyunsdkcore.request import CommonRequest` 走假实现。"""

    package = types.ModuleType("aliyunsdkcore")
    request = types.ModuleType("aliyunsdkcore.request")
    request.CommonRequest = FakeCommonRequest
    monkeypatch.setitem(sys.modules, "aliyunsdkcore", package)
    monkeypatch.setitem(sys.modules, "aliyunsdkcore.request", request)


class FakeBucket:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def put_object(self, key, handle):  # noqa: ARG002
        return type("R", (), {})()

    def sign_url(self, method, key, expires):  # noqa: ARG002
        return f"https://signed/{key}"

    def delete_object(self, key: str) -> None:
        self.deleted.append(key)


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.actions: list[str] = []

    def do_action_with_exception(self, request) -> bytes:
        self.actions.append(request.get_action_name())
        action = request.get_action_name()
        if action == aliyun._ALIYUN["submit"]:
            return json.dumps({"TaskId": "task-1"}).encode("utf-8")
        return json.dumps(self.responses.pop(0)).encode("utf-8")


def _install_fakes(monkeypatch, responses=None, bucket=None):
    _fake_aliyun_sdk(monkeypatch)
    responses = responses if responses is not None else []
    bucket = bucket or FakeBucket()
    client = FakeClient(responses)
    monkeypatch.setattr(aliyun, "_encode_upload_audio", lambda path: path.with_suffix(".mp3"))
    monkeypatch.setattr(
        aliyun,
        "_upload_to_oss",
        lambda mp3, **kwargs: (bucket, "k.mp3", "https://signed/k.mp3"),
    )
    monkeypatch.setattr(aliyun, "_rpc_client", lambda a, b, c: client)
    return bucket, client


def test_transcribe_aliyun_requires_complete_config(tmp_path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(TranscriptionError, match="缺少 .*oss_bucket"):
        transcribe_aliyun(
            wav,
            access_key_id="ak",
            access_key_secret="sk",
            appkey="app",
            region="cn-shanghai",
            oss_bucket="",
        )


def test_transcribe_aliyun_full_flow(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    bucket, client = _install_fakes(
        monkeypatch,
        responses=[
            {"StatusText": "RUNNING"},
            {
                "StatusText": "SUCCESS",
                "StatusCode": 21050000,
                "Result": json.dumps(
                    {
                        "Sentences": [
                            {"BeginTime": 0, "EndTime": 1200, "Text": "你好"},
                            {"BeginTime": 1200, "EndTime": 2500, "Text": "世界"},
                        ]
                    }
                ),
            },
        ],
    )

    segments = transcribe_aliyun(
        wav,
        access_key_id="ak",
        access_key_secret="sk",
        appkey="app",
        region="cn-shanghai",
        oss_bucket="b",
        poll_interval=0,
    )

    assert segments == [
        {"start": 0.0, "end": 1.2, "text": "你好"},
        {"start": 1.2, "end": 2.5, "text": "世界"},
    ]
    # 先提交后查结果；OSS 临时对象被清理。
    assert client.actions == ["SubmitTask", "GetTaskResult", "GetTaskResult"]
    assert bucket.deleted == ["k.mp3"]


def test_transcribe_aliyun_task_failure_fails_fast(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    bucket, _ = _install_fakes(
        monkeypatch,
        responses=[{"StatusText": "FAILED", "StatusCode": 41050103, "Result": "时长超限"}],
    )

    with pytest.raises(TranscriptionError, match="录音文件识别任务失败"):
        transcribe_aliyun(
            wav,
            access_key_id="ak",
            access_key_secret="sk",
            appkey="app",
            region="cn-shanghai",
            oss_bucket="b",
            poll_interval=0,
        )
    assert bucket.deleted == ["k.mp3"]


def test_transcribe_aliyun_poll_timeout(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    _install_fakes(monkeypatch, responses=[{"StatusText": "RUNNING"}] * 3)

    with pytest.raises(TranscriptionError, match="轮询超时"):
        transcribe_aliyun(
            wav,
            access_key_id="ak",
            access_key_secret="sk",
            appkey="app",
            region="cn-shanghai",
            oss_bucket="b",
            poll_interval=0,
            poll_timeout=0.05,
        )


def test_parse_result_falls_back_to_plain_text() -> None:
    segments = aliyun._parse_result(
        {"StatusText": "SUCCESS", "Result": json.dumps({"Result": "整段文本"})}
    )
    assert segments == [{"start": 0.0, "end": 0.0, "text": "整段文本"}]


def test_upload_to_oss_names_key_and_signs_url(tmp_path, monkeypatch) -> None:
    import sys

    captured: dict = {}
    fake_bucket = FakeBucket()

    class FakeOss2:
        @staticmethod
        def Auth(ak, sk):
            return (ak, sk)

        @staticmethod
        def Bucket(auth, endpoint, bucket_name):
            captured["auth"] = auth
            captured["endpoint"] = endpoint
            captured["bucket"] = bucket_name
            return fake_bucket

    monkeypatch.setitem(sys.modules, "oss2", FakeOss2)

    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"fake mp3")
    bucket, key, url = aliyun._upload_to_oss(
        mp3,
        access_key_id="ak",
        access_key_secret="sk",
        region="cn-shanghai",
        oss_bucket="my-bucket",
        oss_endpoint="",
        oss_prefix="audio/",
    )
    assert bucket is fake_bucket
    assert captured["endpoint"] == "https://oss-cn-shanghai.aliyuncs.com"
    assert key.startswith("audio/") and key.endswith(".mp3")
    assert url == f"https://signed/{key}"
