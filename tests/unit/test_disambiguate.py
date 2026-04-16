from __future__ import annotations

import numpy as np

from lib.bary_vec import normalize
from lib.disambiguate import assign_sense, parse_dis1
from lib.embed import FakeEmbedder


def test_parse_dis1():
    assert parse_dis1("0 3 0") == [0, 3, 0]
    assert parse_dis1("") == []
    assert parse_dis1(None) == []
    assert parse_dis1("1 x 2") == [1, 0, 2]


def test_dis1_nonzero_wins():
    emb = FakeEmbedder(dim=8)
    sv = np.zeros((3, 8), dtype=np.float32)
    assert assign_sense({"_dis1": "0 5 0", "word": "foo"}, sv, emb) == 1


def test_cosine_fallback_above_threshold():
    emb = FakeEmbedder(dim=768)
    target = emb.embed(["river"])[0]
    other = normalize(np.random.default_rng(0).standard_normal(768).astype(np.float32))
    sv = np.stack([other, target])  # sense 1 == exact match → cos 1.0
    assert assign_sense({"_dis1": "0 0", "word": "river"}, sv, emb, threshold=0.72) == 1


def test_cosine_fallback_below_threshold():
    emb = FakeEmbedder(dim=768)
    rng = np.random.default_rng(42)
    sv = np.stack([normalize(rng.standard_normal(768).astype(np.float32)) for _ in range(3)])
    # random 768-dim vectors → cosine ≈ 0, well below 0.72
    assert assign_sense({"_dis1": "0 0 0", "word": "river"}, sv, emb, threshold=0.72) is None


def test_missing_dis1_uses_cosine():
    emb = FakeEmbedder(dim=768)
    target = emb.embed(["bank"])[0]
    sv = np.stack([target])
    assert assign_sense({"word": "bank"}, sv, emb, threshold=0.5) == 0


def test_no_word_returns_none():
    emb = FakeEmbedder(dim=8)
    assert assign_sense({"_dis1": "0 0"}, np.zeros((2, 8), dtype=np.float32), emb) is None
