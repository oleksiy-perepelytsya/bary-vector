"""Unit tests for CachedEmbedder and OllamaEmbedder retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from lib.config import Settings
from lib.embed import CachedEmbedder, FakeEmbedder, OllamaEmbedder


class CountingEmbedder:
    """Wraps FakeEmbedder and counts how many texts actually reach it."""

    def __init__(self, dim: int = 8):
        self.inner = FakeEmbedder(dim=dim)
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += len(texts)
        return self.inner.embed(texts)


@pytest.fixture()
def cache_path(tmp_path):
    return tmp_path / "embed_cache.jsonl"


def test_dedup_within_and_across_calls(cache_path):
    inner = CountingEmbedder()
    emb = CachedEmbedder(inner, cache_path)

    v1 = emb.embed(["hello world", "second text"])
    v2 = emb.embed(["hello world", "third text"])  # one hit, two misses overall
    assert inner.calls == 3  # only unique texts embedded
    assert v1.shape == (2, 8) and v2.shape == (2, 8)
    np.testing.assert_array_equal(v1[0], v2[0])  # identical text -> identical vector
    assert len(emb) == 3


def test_persists_across_restarts(cache_path):
    emb1 = CachedEmbedder(CountingEmbedder(), cache_path)
    ref = emb1.embed(["persist me"])
    emb1.close()

    # Fresh instance must serve hits from disk without embedding anything.
    fresh_inner = CountingEmbedder()
    emb2 = CachedEmbedder(fresh_inner, cache_path)
    got = emb2.embed(["persist me"])
    assert fresh_inner.calls == 0
    np.testing.assert_array_equal(got[0], ref[0])


def test_tolerates_truncated_tail_line(cache_path):
    emb = CachedEmbedder(CountingEmbedder(), cache_path)
    emb.embed(["survives"])
    emb.close()

    with open(cache_path, "ab") as f:  # simulate crash mid-flush
        f.write(b'{"h": "deadbeef", "v": [1.')

    emb2 = CachedEmbedder(CountingEmbedder(), cache_path)
    assert len(emb2) == 1  # truncated line dropped, earlier entries intact


def test_empty_batch(cache_path):
    emb = CachedEmbedder(CountingEmbedder(), cache_path)
    out = emb.embed([])
    assert out.shape == (0, 8)


def test_vectors_match_uncached_embedder(cache_path):
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, cache_path)
    texts = ["alpha beta gamma", "", "delta"]
    np.testing.assert_array_equal(cached.embed(texts), inner.inner.embed(texts))


# --- OllamaEmbedder retry tests ---


class _FakeResponse:
    def __init__(self, vecs):
        self._vecs = vecs
    def raise_for_status(self):
        pass
    def json(self):
        return {"embeddings": self._vecs}


def _make_settings():
    return Settings(
        ollama_url="http://x:1234",
        embed_model="test",
        embed_dim=8,
        fake_embed=True,  # skip real network call in constructor
        embed_timeout_seconds=5,
    )


def test_retry_succeeds_after_transient_failure():
    embedder = OllamaEmbedder(_make_settings())
    call_n = [0]
    fake_client = MagicMock()
    def side_effect(*_args, **_kwargs):
        call_n[0] += 1
        if call_n[0] <= 2:
            raise httpx.ReadTimeout("boom")
        return _FakeResponse([list(np.zeros(8, dtype=np.float32))])

    fake_client.post = MagicMock(side_effect=side_effect)
    with patch.object(embedder, "_client", return_value=fake_client), \
         patch("lib.embed.time.sleep"):
        out = embedder.embed(["hello"])
    assert call_n[0] == 3
    assert out.shape == (1, 8)


def test_retry_exhausts_and_raises():
    embedder = OllamaEmbedder(_make_settings())
    fake_client = MagicMock()
    fake_client.post = MagicMock(side_effect=httpx.ReadTimeout("persistent fail"))
    with patch.object(embedder, "_client", return_value=fake_client), \
         patch("lib.embed.time.sleep"):
        with pytest.raises(httpx.ReadTimeout):
            embedder.embed(["will fail"])


class FailNTimesEmbedder:
    """Fails the first ``n`` embed calls then succeeds."""

    def __init__(self, fail_n: int, dim: int = 8):
        self.dim = dim
        self._inner = FakeEmbedder(dim=dim)
        self._fail_n = fail_n
        self._calls = 0

    def embed(self, texts):
        self._calls += 1
        if self._calls <= self._fail_n:
            raise httpx.ReadTimeout("transient")
        return self._inner.embed(texts)
