from __future__ import annotations

import numpy as np

from lib.embed import Embedder


def parse_dis1(s: str | None) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for tok in s.split():
        try:
            out.append(int(tok))
        except ValueError:
            out.append(0)
    return out


def assign_sense(
    item: dict,
    sense_vectors: np.ndarray,
    embedder: Embedder,
    threshold: float = 0.72,
) -> int | None:
    """
    Resolve a kaikki relation/translation item to a sense index.

    Priority:
      1. `_dis1` weights: if any > 0, return argmax.
      2. Cosine fallback: embed item['word'], pick best sense if ≥ threshold.
      3. Otherwise None (assign at word level / L14).
    """
    weights = parse_dis1(item.get("_dis1"))
    if weights and max(weights) > 0:
        return int(np.argmax(weights))

    word = item.get("word")
    if not word or sense_vectors.size == 0:
        return None

    target = embedder.embed([word])[0]
    sims = sense_vectors @ target  # both L2-normalized → cosine
    best = int(np.argmax(sims))
    if float(sims[best]) >= threshold:
        return best
    return None
