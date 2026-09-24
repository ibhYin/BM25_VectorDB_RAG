import pytest

from app.config import Settings
from app.system_qa import (
    CODE_VERSION,
    answer_system_intent,
    detect_system_intent,
)


def test_detect_hits_system_intents() -> None:
    assert detect_system_intent("支持mp4格式的文档入库吗？") == "formats"
    assert detect_system_intent("知识库支持哪些文件格式？") == "formats"
    assert detect_system_intent("现在接入的是什么模型？") == "models"
    assert detect_system_intent("当前用的是什么向量模型？") == "models"
    assert detect_system_intent("运行版本是什么？") == "version"
    assert detect_system_intent("现在部署的是哪个版本？") == "version"
    assert detect_system_intent("现在的知识库里有哪些内容？") == "kb"
    assert detect_system_intent("知识库里有多少篇文档？") == "kb"


def test_detect_does_not_hit_domain_questions() -> None:
    # 领域问题不应被当成系统问答。
    assert detect_system_intent("Transformer 模型为什么好？") is None
    assert detect_system_intent("什么是注意力机制？") is None
    assert detect_system_intent("知识库里有关 RAG 的文档吗？") is None
    assert detect_system_intent("pdf 文档的切片策略是什么？") is None


def test_answer_formats_lists_supported_suffixes() -> None:
    answer = answer_system_intent("formats", Settings(_env_file=None))
    assert "md" in answer and "pdf" in answer and "txt" in answer
    assert "mp4" in answer and "mkv" in answer and "webm" in answer
    assert "mp3" in answer and "wav" in answer and "ogg" in answer
    assert "转写" in answer


def test_answer_models_reflects_settings() -> None:
    settings = Settings(
        _env_file=None,
        llm_model="deepseek-test",
        openai_base_url="https://chat.example.test/v1",
        embedding_model="bge-small-zh",
        embedding_base_url="https://emb.example.test/v1",
    )
    answer = answer_system_intent("models", settings)
    assert "deepseek-test" in answer
    assert "bge-small-zh" in answer
    assert "未配置" in answer  # ASR 未配置


def test_answer_models_reports_asr_when_configured() -> None:
    settings = Settings(
        _env_file=None,
        audio_transcribe_provider="aliyun",
        aliyun_appkey="app",
        aliyun_oss_bucket="bucket",
        aliyun_region="cn-shanghai",
    )
    answer = answer_system_intent("models", settings)
    assert "阿里云录音文件识别" in answer
    assert "bucket" in answer


def test_answer_version() -> None:
    answer = answer_system_intent("version", Settings(_env_file=None))
    assert answer == f"当前运行版本：{CODE_VERSION}"


def test_answer_kb_empty(monkeypatch) -> None:
    import app.service as service

    monkeypatch.setattr(service, "list_indexed_documents", lambda *a, **k: [])
    answer = answer_system_intent("kb", Settings(_env_file=None))
    assert "还没有入库任何文档" in answer


def test_answer_kb_lists_documents(monkeypatch) -> None:
    import app.service as service

    fake = [
        {"title": "文档A", "doc_id": "a", "chunk_count": 3},
        {"title": "文档B", "doc_id": "b", "chunk_count": 5},
    ]
    monkeypatch.setattr(service, "list_indexed_documents", lambda *a, **k: fake)
    answer = answer_system_intent("kb", Settings(_env_file=None))
    assert "2 篇文档" in answer
    assert "文档A" in answer and "3 切片" in answer
    assert "文档B" in answer and "5 切片" in answer


def test_unknown_intent_returns_none() -> None:
    assert answer_system_intent("whatever", Settings(_env_file=None)) is None
