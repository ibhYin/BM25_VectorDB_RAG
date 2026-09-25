"""Minimal OpenAI-compatible embedding endpoint backed by a local ONNX model."""

from __future__ import annotations

import os
from pathlib import Path

# 2 核服务器：在导入 numpy/onnxruntime 前限制线程池，避免过度订阅 + 额外内存。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer


MODEL_DIR = Path(os.environ.get("EMBEDDING_MODEL_DIR", "models/bge-small-zh-v1.5"))
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "bge-small-zh:q8_0")
MODEL_FILE = os.environ.get("EMBEDDING_MODEL_FILE", "model.onnx")
MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_LENGTH", "512"))
THREADS = int(os.environ.get("EMBEDDING_THREADS", "2"))

# 关掉 CPU arena 预分配并限制线程：bge-small 模型本身很小（~90MB），500MB 的
# RSS 主要是 onnxruntime 的 arena 预留，关掉能明显压低常驻内存。
_session_options = ort.SessionOptions()
_session_options.intra_op_num_threads = THREADS
_session_options.inter_op_num_threads = 1
_session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
try:
    _session_options.enable_cpu_mem_arena = False
except AttributeError:
    pass

_session = ort.InferenceSession(
    str(MODEL_DIR / MODEL_FILE),
    sess_options=_session_options,
    providers=["CPUExecutionProvider"],
)
_tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
_tokenizer.enable_truncation(max_length=MAX_LENGTH)
_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

app = FastAPI(title="Local BGE Embedding API")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


def _embed(texts: list[str]) -> tuple[list[list[float]], int]:
    # The Ollama BERT tokenizer lowercases input; match it to preserve vectors.
    encodings = _tokenizer.encode_batch([text.lower() for text in texts])
    input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.asarray(
        [encoding.attention_mask for encoding in encodings], dtype=np.int64
    )
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if any(item.name == "token_type_ids" for item in _session.get_inputs()):
        feeds["token_type_ids"] = np.asarray(
            [encoding.type_ids for encoding in encodings], dtype=np.int64
        )

    hidden = _session.run(["last_hidden_state"], feeds)[0]
    vectors = hidden[:, 0]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    return vectors.tolist(), int(attention_mask.sum())


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/embeddings")
def create_embeddings(payload: EmbeddingRequest) -> dict[str, object]:
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    if not texts or any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="input must contain non-empty text")

    vectors, token_count = _embed(texts)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "model": payload.model or MODEL_NAME,
        "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
    }
