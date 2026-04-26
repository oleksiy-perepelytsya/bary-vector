"""Pair-matching primitives shared by stages 04, 06, 07, 08.

``top_k_pairs`` uses brute-force O(n²) for small inputs and switches to an
hnswlib HNSW ANN index above ANN_THRESHOLD. The rest of the pipeline only
consumes the ranked-pair iterable and is backend-agnostic.

ANN parameters are tunable via environment variables:
  ANN_THRESHOLD   switch point (default 20 000)
  ANN_K           neighbours queried per vector (default 50)
  ANN_EF          hnswlib ef query parameter (default 100)
  ANN_M           hnswlib M graph parameter (default 16)
"""

from __future__ import annotations

import os
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

# Above this many vectors, use hnswlib ANN instead of brute-force O(n²).
ANN_THRESHOLD: int = int(os.environ.get("ANN_THRESHOLD", 20_000))
ANN_K: int = int(os.environ.get("ANN_K", 50))
ANN_EF: int = int(os.environ.get("ANN_EF", 100))
ANN_M: int = int(os.environ.get("ANN_M", 16))
ANN_EF_CONSTRUCTION: int = int(os.environ.get("ANN_EF_CONSTRUCTION", 100))


def top_k_pairs(vectors: np.ndarray, k: int | None = None) -> Iterable[tuple[int, int, float]]:
    """Yield (i, j, cos) with i<j, sorted by descending cosine.

    Rows of ``vectors`` must be L2-normalised. Uses brute-force for n ≤
    ANN_THRESHOLD and hnswlib HNSW otherwise.
    """
    n = vectors.shape[0]
    if n < 2:
        return
    if n <= ANN_THRESHOLD:
        yield from _brute_force_pairs(vectors, k)
    else:
        yield from _ann_pairs(vectors, k)


def _brute_force_pairs(
    vectors: np.ndarray, k: int | None
) -> Iterable[tuple[int, int, float]]:
    sims = vectors @ vectors.T
    iu = np.triu_indices(vectors.shape[0], k=1)
    scores = sims[iu]
    order = np.argsort(-scores)
    if k is not None:
        order = order[:k]
    for idx in order:
        yield int(iu[0][idx]), int(iu[1][idx]), float(scores[idx])


def _ann_pairs(vectors: np.ndarray, k: int | None) -> Iterable[tuple[int, int, float]]:
    """ANN pair search via hnswlib HNSW index.

    hnswlib cosine space uses distance = 1 − cosine for normalised vectors,
    so cosine_similarity = 1 − distance.
    """
    import hnswlib  # optional at import time; always present in production

    n, dim = vectors.shape
    nn_k = min(n - 1, ANN_K)
    ef = max(ANN_EF, nn_k)

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ANN_EF_CONSTRUCTION, M=ANN_M)
    index.add_items(vectors, list(range(n)))
    index.set_ef(ef)

    labels, distances = index.knn_query(vectors, k=nn_k + 1)  # +1: self is returned

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j, dist in zip(labels[i], distances[i], strict=True):
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b, 1.0 - float(dist)))

    pairs.sort(key=lambda x: -x[2])
    if k is not None:
        pairs = pairs[:k]
    yield from pairs


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
