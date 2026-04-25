"""Extract synonym pairs from the ingested graph for recall evaluation.

For a smoke test (small corpus), run post-ingestion with --fraction 1.0 to
use all available synonym pairs. For a proper held-out eval on the full
corpus, run with --fraction 0.1 BEFORE s06_l14_edges so those synonym
links are excluded from edge formation — that requires s06 to filter against
holdout.json, which is not yet implemented (deferred to full-corpus run).

Output: data/holdout.json — list of {word_a, gloss_a, word_b} objects.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lib.config import Settings
from lib.db import get_collection
from lib.log import get_logger, setup_logging

HOLDOUT_PATH = Path("data/holdout.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate holdout synonym pairs")
    p.add_argument(
        "--fraction", type=float, default=1.0,
        help="fraction of pairs to include (default: 1.0 for smoke test, 0.1 for proper eval)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(HOLDOUT_PATH))
    return p.parse_args()


def run() -> None:
    args = _parse_args()
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("eval.holdout")

    coll = get_collection(settings)

    word_docs = list(coll.find(
        {"doc_type": "node", "node_type": "word", "level": 14},
        {"properties.word": 1, "properties.relations": 1},
    ))
    log.info("loaded %d word nodes", len(word_docs))

    known_words: set[str] = {d["properties"]["word"] for d in word_docs}

    # Collect canonical synonym pairs (alphabetical order to avoid duplicates).
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for d in word_docs:
        word_a = d["properties"]["word"]
        for r in (d["properties"].get("relations") or []):
            if r["kind"] != "synonyms":
                continue
            word_b = r["word"]
            if word_b not in known_words or word_b == word_a:
                continue
            key = (min(word_a, word_b), max(word_a, word_b))
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    log.info("found %d unique synonym pairs", len(pairs))
    if not pairs:
        log.warning("no synonym pairs found — graph may be too sparse or relations missing")
        return

    # Sample.
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = max(1, round(len(pairs) * args.fraction))
    sampled = pairs[:n]
    log.info("sampled %d pairs (fraction=%.2f)", len(sampled), args.fraction)

    # Fetch first sense gloss for each word_a (used as query text at eval time).
    word_a_set = {w for w, _ in sampled}
    gloss_map: dict[str, str] = {}
    for s in coll.find(
        {"doc_type": "node", "node_type": "sense", "level": 15,
         "properties.word": {"$in": list(word_a_set)}},
        {"properties.word": 1, "properties.gloss": 1},
    ):
        w = s["properties"]["word"]
        if w not in gloss_map:
            gloss_map[w] = s["properties"]["gloss"]

    output = [
        {
            "word_a": w_a,
            "gloss_a": gloss_map.get(w_a, w_a),
            "word_b": w_b,
        }
        for w_a, w_b in sampled
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("wrote %d pairs → %s", len(output), out_path)


if __name__ == "__main__":
    run()
