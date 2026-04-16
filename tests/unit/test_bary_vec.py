from __future__ import annotations

import math

import numpy as np
import pytest

from lib.bary_vec import (
    TYPE_SENTENCES,
    build_l15_type_text,
    compute_bary_vec,
    compute_metabary_vec,
    cosine,
    normalize,
    word_vector,
)


def _rand(dim=768, seed=0):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_normalize_unit_norm():
    v = normalize(_rand())
    assert math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-5)


def test_normalize_zero_vector():
    z = normalize(np.zeros(10))
    assert float(np.linalg.norm(z)) == 0.0


def test_cosine_self():
    v = _rand()
    assert math.isclose(cosine(v, v), 1.0, rel_tol=1e-5)


def test_bary_vec_q1_is_cm_mean_direction():
    v1, v2, vt = _rand(seed=1), _rand(seed=2), _rand(seed=3)
    out = compute_bary_vec(v1, v2, vt, q=1.0)
    expected = normalize(v1 + v2)
    assert np.allclose(out, expected, atol=1e-5)


def test_bary_vec_q0_is_type():
    v1, v2, vt = _rand(seed=1), _rand(seed=2), _rand(seed=3)
    out = compute_bary_vec(v1, v2, vt, q=0.0)
    assert np.allclose(out, normalize(vt), atol=1e-5)


def test_bary_vec_unit_norm():
    out = compute_bary_vec(_rand(seed=1), _rand(seed=2), _rand(seed=3), q=0.6)
    assert math.isclose(float(np.linalg.norm(out)), 1.0, rel_tol=1e-5)


def test_metabary_qmb_formula():
    _, q_mb = compute_metabary_vec(_rand(seed=1), _rand(seed=2), _rand(seed=3), 0.3, 0.4, 0.5)
    expected = 0.5 / math.sqrt(0.3**2 + 0.4**2 + 0.5**2)
    assert math.isclose(q_mb, expected, rel_tol=1e-9)


def test_word_vector_single_be():
    be = normalize(_rand(seed=7))
    out = word_vector([be], [])
    assert np.allclose(out, be, atol=1e-5)


def test_word_vector_empty_raises():
    with pytest.raises(ValueError):
        word_vector([], [])


def test_type_sentences_keys():
    assert set(TYPE_SENTENCES) == {
        "same_phenomenon",
        "contradicts",
        "extends",
        "applies_to",
        "is_instance_of",
    }


def test_build_l15_type_text():
    s = build_l15_type_text("hot", ["cold"], ["warm", "boiling"], "warm", [], ["tepid"])
    assert "hot" in s and "cold" in s and "tepid" in s
    assert s.count(";") >= 1
