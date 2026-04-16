from __future__ import annotations

import hashlib
from typing import Protocol

import httpx
import numpy as np

from lib.bary_vec import normalize
from lib.config import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalized rows
        ...


class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.dim = settings.embed_dim
        self._url = settings.ollama_url.rstrip("/") + "/api/embed"
        self._model = settings.embed_model
        self._client = httpx.Client(timeout=settings.embed_timeout_seconds)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.post(self._url, json={"model": self._model, "input": texts})
        resp.raise_for_status()
        data = resp.json()
        arr = np.asarray(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms


class FakeEmbedder:
    """Deterministic, offline embedder for CI and unit tests."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little")
            rng = np.random.default_rng(seed)
            out[i] = normalize(rng.standard_normal(self.dim).astype(np.float32))
        return out


def get_embedder(settings: Settings) -> Embedder:
    if settings.fake_embed:
        return FakeEmbedder(dim=settings.embed_dim)
    return OllamaEmbedder(settings)
