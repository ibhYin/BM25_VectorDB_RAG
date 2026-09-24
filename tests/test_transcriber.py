import pytest

from rag import transcriber
from rag.transcriber import (
    TranscriptionError,
    _extract_audio,
    _fmt_time,
    _transcribe_slice,
    group_segments,
    transcribe_audio,
)


def test_fmt_time_formats_hms() -> None:
    assert _fmt_time(0) == "00:00"
    assert _fmt_time(65) == "01:05"
    assert _fmt_time(3725) == "1:02:05"


def test_group_segments_buckets_by_time_and_keeps_timestamps() -> None:
    segments = [
        {"start": 0.0, "end": 8.0, "text": "第一段"},
        {"start": 8.0, "end": 30.0, "text": "第二段"},
        {"start": 31.0, "end": 65.0, "text": "第三段"},  # 触发 30s 分桶
    ]
    docs = group_segments(segments, source="/x/a.mp4")

    assert len(docs) == 2
    first, second = docs
    assert first.metadata["start_time"] == 0.0
    assert first.metadata["end_time"] == 30.0
    assert first.metadata["section_path"] == "00:00~00:30"
    assert "第一段" in first.page_content and "第二段" in first.page_content
    assert second.metadata["start_time"] == 31.0
    assert second.metadata["end_time"] == 65.0
    assert second.metadata["section_path"] == "00:31~01:05"
    # 不带 structured_by，走固定长度切分；source 透传。
    assert "structured_by" not in first.metadata
    assert first.metadata["source"] == "/x/a.mp4"


def test_group_segments_buckets_by_char_count() -> None:
    long_text = "很长的句子。" * 40  # 240 字，三段的累计字符触发 400 字阈值
    segments = [
        {"start": 0.0, "end": 2.0, "text": long_text},
        {"start": 2.0, "end": 4.0, "text": long_text},
        {"start": 4.0, "end": 6.0, "text": long_text},
    ]
    docs = group_segments(segments, source="/x/a.mp3")
    assert len(docs) == 2
    assert docs[0].metadata["start_time"] == 0.0
    assert docs[0].metadata["end_time"] == 4.0
    assert docs[1].metadata["start_time"] == 4.0


def test_transcribe_audio_requires_model(tmp_path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00" * 1000)
    with pytest.raises(TranscriptionError, match="语音识别未配置"):
        transcribe_audio(wav, base_url="", api_key="", model="")


def test_transcribe_audio_slices_long_audio_and_offsets_timestamps(
    tmp_path, monkeypatch
) -> None:
    wav = tmp_path / "long.wav"
    wav.write_bytes(b"\x00" * 1000)
    monkeypatch.setattr(transcriber, "_wav_duration", lambda path: 2 * transcriber.TRANSCRIBE_SLICE_SECONDS + 5)
    monkeypatch.setattr(
        transcriber,
        "_slice_wav",
        lambda wav_path, start, duration: tmp_path / f"slice-{int(start)}.wav",
    )

    calls: list[tuple[float, float]] = []
    real = transcriber._transcribe_slice

    def fake_slice(wav_path, **kwargs):
        start = float(wav_path.stem.split("-")[1])
        calls.append((start, kwargs["model"]))
        return [{"start": 1.0, "end": 3.0, "text": f"切片{int(start)}"}]

    monkeypatch.setattr(transcriber, "_transcribe_slice", fake_slice)

    segments = transcribe_audio(wav, base_url="http://x/v1", api_key="k", model="m")
    assert len(calls) == 3
    # 每个切片的本地时间都加上了切片起点偏移。
    assert segments == [
        {"start": 1.0, "end": 3.0, "text": "切片0"},
        {"start": 1.0 + transcriber.TRANSCRIBE_SLICE_SECONDS,
         "end": 3.0 + transcriber.TRANSCRIBE_SLICE_SECONDS, "text": f"切片{transcriber.TRANSCRIBE_SLICE_SECONDS}"},
        {"start": 1.0 + 2 * transcriber.TRANSCRIBE_SLICE_SECONDS,
         "end": 3.0 + 2 * transcriber.TRANSCRIBE_SLICE_SECONDS,
         "text": f"切片{2 * transcriber.TRANSCRIBE_SLICE_SECONDS}"},
    ]


def test_extract_audio_requires_ffmpeg(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: None)
    with pytest.raises(TranscriptionError, match="ffmpeg"):
        _extract_audio(tmp_path / "x.mp4")


def test_transcribe_slice_uses_segments_when_present(tmp_path, monkeypatch) -> None:
    class FakeSeg:
        start = 1.0
        end = 2.5
        text = "你好"

    captured = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            response = type("Resp", (), {})()
            response.segments = [FakeSeg()]
            return response

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        audio = FakeAudio()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    wav = tmp_path / "s.wav"
    wav.write_bytes(b"\x00" * 1000)
    out = _transcribe_slice(wav, base_url="http://x/v1", api_key="k", model="m")
    assert out == [{"start": 1.0, "end": 2.5, "text": "你好"}]
    assert captured["kwargs"]["base_url"] == "http://x/v1"
    assert captured["create"]["model"] == "m"
    assert captured["create"]["response_format"] == "verbose_json"


def test_transcribe_slice_falls_back_to_text_without_segments(tmp_path, monkeypatch) -> None:
    captured = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            response = type("Resp", (), {})()
            response.segments = None
            response.text = "只有整段文本"
            return response

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        audio = FakeAudio()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr(transcriber, "_wav_duration", lambda path: 42.0)

    wav = tmp_path / "s.wav"
    wav.write_bytes(b"\x00" * 1000)
    out = _transcribe_slice(wav, base_url="", api_key="", model="m")
    assert out == [{"start": 0.0, "end": 42.0, "text": "只有整段文本"}]
