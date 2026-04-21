from __future__ import annotations

import numpy as np

from lib.bary_vec import normalize
from lib.match import FERMION_TIERS, greedy_unique_match, nearest_row, top_k_pairs


def _vecs():
    # 0,1 nearly identical; 2,3 nearly identical; 0/1 far from 2/3.
    a = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    b = normalize(np.array([0.99, 0.01, 0.0], dtype=np.float32))
    c = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    d = normalize(np.array([0.0, 0.99, 0.01], dtype=np.float32))
    return np.stack([a, b, c, d])


def test_top_k_pairs_sorted_desc():
    pairs = list(top_k_pairs(_vecs()))
    scores = [s for *_, s in pairs]
    assert scores == sorted(scores, reverse=True)
    assert pairs[0][:2] in [(0, 1), (2, 3)]


def test_greedy_unique_parent():
    out = greedy_unique_match(top_k_pairs(_vecs()), threshold=0.5)
    # Exactly two pairs, every index used at most once.
    assert len(out) == 2
    used = [i for p in out for i in p[:2]]
    assert len(used) == len(set(used)) == 4


def test_greedy_threshold_excludes():
    # 0↔2, 0↔3, 1↔2, 1↔3 all have cos ≈ 0; threshold 0.5 keeps only the
    # two near-identical pairs, threshold >1 keeps none.
    assert len(greedy_unique_match(top_k_pairs(_vecs()), threshold=0.5)) == 2
    assert greedy_unique_match(top_k_pairs(_vecs()), threshold=1.01) == []


def test_polysemy_floor_applied():
    # Two orthogonal vectors → cos 0. With floor 0.5 and threshold 0.4 they pair.
    V = np.stack([normalize(np.array([1.0, 0.0])), normalize(np.array([0.0, 1.0]))])
    out = greedy_unique_match(
        top_k_pairs(V), threshold=0.4,
        same_word={frozenset((0, 1))}, polysemy_floor=0.5,
    )
    assert out == [(0, 1, 0.5)]


def test_nearest_row():
    V = _vecs()
    idx, score = nearest_row(V[0], V[1:])
    assert idx == 0  # row 0 of V[1:] is original index 1
    assert score > 0.99


def test_fermion_tiers_ordered_and_keyed():
    assert [t.priority for t in FERMION_TIERS] == [1, 2, 3, 4, 5, 6]
    assert FERMION_TIERS[0].kaikki_fields == ("antonyms",)
    assert FERMION_TIERS[-1].q_seed_key == "synonyms"
    # Tiers 5 and 6 share edge_type but differ in q_seed_key.
    assert FERMION_TIERS[4].edge_type == FERMION_TIERS[5].edge_type == "same_phenomenon"
    assert FERMION_TIERS[4].q_seed_key != FERMION_TIERS[5].q_seed_key


def test_fermion_q_seed_keys_match_settings():
    from lib.config import Settings
    s = Settings.load()
    for t in FERMION_TIERS:
        assert t.q_seed_key in s.q_seeds
