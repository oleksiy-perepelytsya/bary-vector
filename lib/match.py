"""Pair-matching primitives shared by stages 04, 06, 07, 08.

The greedy matcher here is brute-force O(n²) on a numpy cosine matrix,
which is fine for the fixture / dev slices. For the full ~2.5M-sense
corpus, swap :func:`top_k_pairs` for a FAISS/hnswlib ANN backend (see
v0.4 §17.1) — the rest of the pipeline only consumes the ranked-pair
iterable and is backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FermionTier:
    """One row of the L14 fermion-order matching table (v0.4 §7.1)."""

    priority: int
    edge_type: str
    kaikki_fields: tuple[str, ...]
    q_seed_key: str


FERMION_TIERS: tuple[FermionTier, ...] = (
    FermionTier(1, "contradicts", ("antonyms",), "contradicts"),
    FermionTier(2, "applies_to", ("meronyms", "holonyms"), "applies_to"),
    FermionTier(3, "is_instance_of", ("hypernyms", "hyponyms"), "is_instance_of"),
    FermionTier(4, "extends", ("derived", "related"), "extends"),
    FermionTier(5, "same_phenomenon", ("coordinate_terms",), "coordinate_terms"),
    FermionTier(6, "same_phenomenon", ("synonyms",), "synonyms"),
)


def top_k_pairs(vectors: np.ndarray, k: int | None = None) -> Iterable[tuple[int, int, float]]:
    """Yield (i, j, cos) with i<j, sorted by descending cosine.

    Brute-force upper-triangle scan. Rows of ``vectors`` are assumed
    L2-normalized.
    """
    n = vectors.shape[0]
    if n < 2:
        return
    sims = vectors @ vectors.T
    iu = np.triu_indices(n, k=1)
    scores = sims[iu]
    order = np.argsort(-scores)
    if k is not None:
        order = order[:k]
    for idx in order:
        yield int(iu[0][idx]), int(iu[1][idx]), float(scores[idx])


def greedy_unique_match(
    pairs: Iterable[tuple[int, int, float]],
    threshold: float,
    *,
    same_word: set[frozenset[int]] | None = None,
    polysemy_floor: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Greedy highest-first selection honoring the unique-parent constraint.

    ``same_word`` lets the caller mark same-headword sense pairs whose
    effective q is floored at ``polysemy_floor`` (v0.4 §7.3); they are
    still subject to ``threshold`` after flooring.
    """
    same_word = same_word or set()
    taken: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for i, j, q in pairs:
        if i in taken or j in taken:
            continue
        eff_q = max(q, polysemy_floor) if frozenset((i, j)) in same_word else q
        if eff_q < threshold:
            continue
        taken.add(i)
        taken.add(j)
        out.append((i, j, eff_q))
    return out


def nearest_row(query: np.ndarray, matrix: np.ndarray) -> tuple[int, float]:
    """Index + cosine of the nearest row in ``matrix`` to ``query``."""
    sims = matrix @ np.asarray(query, dtype=np.float32)
    best = int(np.argmax(sims))
    return best, float(sims[best])
