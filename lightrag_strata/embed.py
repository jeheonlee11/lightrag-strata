"""Local embeddings via fastembed (ONNX, no torch, no API key)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
from lightrag.utils import EmbeddingFunc

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def make_embedding_func(model_name: str = DEFAULT_EMBED_MODEL) -> EmbeddingFunc:
    # The Hugging Face Xet transfer backend can stall on some networks; plain HTTP is reliable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from fastembed import TextEmbedding

    cache_dir = os.environ.get("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
    model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    dim = _DIMS.get(model_name) or len(next(iter(model.embed(["dim probe"]))))

    async def embed(texts: list[str], **kwargs) -> np.ndarray:
        # fastembed is CPU-bound and synchronous; keep the event loop free.
        vectors = await asyncio.to_thread(lambda: list(model.embed(texts)))
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=512, func=embed, model_name=model_name)
