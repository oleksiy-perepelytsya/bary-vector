from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

EMBED_DIM = 768

# L14 edge-type sentences — embedded once, used as v(type). See CLAUDE.md.
TYPE_SENTENCES: dict[str, str] = {
    "same_phenomenon": "these two words describe the same concept",
    "contradicts": "these two words have opposite meanings",
    "extends": "one word is derived from or extends the other",
    "applies_to": "these two words share a common origin or root",
    "is_instance_of": "this relationship is a specific instance of the broader relationship",
}


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def compute_bary_vec(
    v_cm1: np.ndarray, v_cm2: np.ndarray, v_type: np.ndarray, q: float
) -> np.ndarray:
    """bary_vec = normalize( q·v_cm1 + q·v_cm2 + (1−q)·v_type )  — L14/L15."""
    v_cm1 = np.asarray(v_cm1, dtype=np.float32)
    v_cm2 = np.asarray(v_cm2, dtype=np.float32)
    v_type = np.asarray(v_type, dtype=np.float32)
    raw = q * v_cm1 + q * v_cm2 + (1.0 - q) * v_type
    return normalize(raw)


def compute_metabary_vec(
    v_be1: np.ndarray,
    v_be2: np.ndarray,
    v_bridge: np.ndarray,
    q1: float,
    q2: float,
    q3: float,
) -> tuple[np.ndarray, float]:
    """MetaBary (L13+): q_mb = q3² / sqrt(q1⁴ + q2⁴ + q3⁴); then bary_vec with q=q_mb."""
    denom = float(np.sqrt(q1**4 + q2**4 + q3**4))
    q_mb = (q3 * q3) / denom if denom > 0.0 else 0.0
    vec = compute_bary_vec(v_be1, v_be2, v_bridge, q_mb)
    return vec, q_mb


def word_vector(
    be_vecs: Iterable[np.ndarray], orphan_sense_vecs: Iterable[np.ndarray]
) -> np.ndarray:
    """v(word) = normalize( Σ v(BE_i) + Σ v(orphan_sense_j) ). No embedding call."""
    parts = [np.asarray(v, dtype=np.float32) for v in be_vecs]
    parts += [np.asarray(v, dtype=np.float32) for v in orphan_sense_vecs]
    if not parts:
        raise ValueError("word_vector requires at least one component vector")
    return normalize(np.sum(np.stack(parts, axis=0), axis=0))


def build_l15_type_text(
    word_a: str,
    ant_a: Sequence[str],
    syn_a: Sequence[str],
    word_b: str,
    ant_b: Sequence[str],
    syn_b: Sequence[str],
) -> str:
    """Per-pair lexical-neighborhood string used as v(type) source at L15."""

    def part(w: str, ants: Sequence[str], syns: Sequence[str]) -> str:
        segs = []
        if ants:
            segs.append("antonyms: " + ", ".join(ants))
        if syns:
            segs.append("synonyms: " + ", ".join(syns))
        inner = "; ".join(segs) if segs else ""
        return f"{w} ({inner})" if inner else w

    return f"{part(word_a, ant_a, syn_a)}; {part(word_b, ant_b, syn_b)}"
