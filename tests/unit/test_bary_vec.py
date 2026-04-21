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
    expected = 0.5**2 / math.sqrt(0.3**4 + 0.4**4 + 0.5**4)
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


def test_count_syllables():
    from lib.bary_vec import count_syllables
    assert count_syllables("cat") == 1
    assert count_syllables("dictionary") >= 3
    assert count_syllables("xyz") == 1  # floor at 1


def test_word_length_feature_shape_and_determinism():
    from lib.bary_vec import word_length_feature
    a = word_length_feature("cat", ["cats"])
    b = word_length_feature("cat", ["cats"])
    assert a.shape == (768,)
    assert math.isclose(float(np.linalg.norm(a)), 1.0, rel_tol=1e-5)
    assert np.allclose(a, b)


def test_word_length_feature_separates_short_long():
    from lib.bary_vec import cosine, word_length_feature
    short = word_length_feature("cat", [])
    long = word_length_feature("antidisestablishmentarianism", [])
    # Same projection direction (3 positive features through fixed matrix) but
    # the *raw* feature distance should be non-trivial; here we just assert
    # they are not identical.
    assert cosine(short, long) < 0.9999


def test_word_vector_with_length_feature():
    be = normalize(_rand(seed=1))
    phi = normalize(_rand(seed=2))
    v0 = word_vector([be], [], length_feat=phi, lam=0.0)
    v1 = word_vector([be], [], length_feat=phi, lam=0.5)
    assert np.allclose(v0, be, atol=1e-5)
    assert not np.allclose(v1, be, atol=1e-3)
    assert math.isclose(float(np.linalg.norm(v1)), 1.0, rel_tol=1e-5)
