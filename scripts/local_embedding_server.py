"""Minimal OpenAI-compatible embedding endpoint backed by a local ONNX model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer


MODEL_DIR = Path(os.environ.get("EMBEDDING_MODEL_DIR", "models/bge-small-zh-v1.5"))
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "bge-small-zh:q8_0")
MODEL_FILE = os.environ.get("EMBEDDING_MODEL_FILE", "model.onnx")
MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_LENGTH", "512"))

_session = ort.InferenceSession(
    str(MODEL_DIR / MODEL_FILE),
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
