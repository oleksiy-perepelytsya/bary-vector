"""Primary eval: BaryGraph vs flat synonym recall@K.

BaryGraph retrieval:  query baryedge docs, check word_b via CM lineage traversal.
Flat baseline:        query node docs (L14/L15), check word_b in properties.word.

CM lineage rule (v0.5 §11.1):
  - L14/L15 BEs have node CMs → word found directly in properties.word.
  - L13+ MBs have BE/MB CMs → BFS through cm1_id/cm2_id until reaching nodes.

Run after s10_index and after eval/holdout.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lib.config import Settings
from lib.db import any_cm_has_word, get_collection, vector_search
from lib.embed import get_embedder
from lib.log import get_logger, setup_logging

HOLDOUT_PATH = Path("data/holdout.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BaryGraph vs flat recall@K evaluation")
    p.add_argument("--recall-k", type=int, default=20, dest="recall_k")
    p.add_argument("--num-candidates", type=int, default=200, dest="num_candidates")
    p.add_argument("--holdout", type=str, default=str(HOLDOUT_PATH))
    p.add_argument(
        "--max-pairs", type=int, default=None, dest="max_pairs",
        help="cap number of pairs evaluated (useful for quick smoke tests)",
    )
    return p.parse_args()


def _embed_in_batches(embedder, texts: list[str], batch_size: int) -> np.ndarray:
    parts = []
    for i in range(0, len(texts), batch_size):
        parts.append(embedder.embed(texts[i : i + batch_size]))
    return np.vstack(parts) if parts else np.empty((0, embedder.dim), dtype=np.float32)


def run() -> None:
    args = _parse_args()
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("eval.recall")

    holdout_path = Path(args.holdout)
    if not holdout_path.exists():
        raise FileNotFoundError(f"{holdout_path} missing — run `make eval-holdout` first")

    pairs: list[dict] = json.loads(holdout_path.read_text())
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    log.info("evaluating %d pairs  recall@%d  num_candidates=%d",
             len(pairs), args.recall_k, args.num_candidates)

    coll = get_collection(settings)
    embedder = get_embedder(settings)

    log.info("embedding %d query glosses …", len(pairs))
    query_vecs = _embed_in_batches(
        embedder,
        [p["gloss_a"] for p in pairs],
        batch_size=settings.embed_batch_size,
    )

    bary_hits = flat_hits = 0

    for i, (pair, qv) in enumerate(zip(pairs, query_vecs)):
        word_b = pair["word_b"]
        qv_list = qv.tolist()

        # --- BaryGraph retrieval: search all baryedge docs ---
        be_docs = vector_search(
            coll, qv_list,
            limit=args.recall_k,
            num_candidates=args.num_candidates,
            filter={"doc_type": "baryedge"},
        )
        be_ids = [d["_id"] for d in be_docs]
        if any_cm_has_word(coll, be_ids, word_b):
            bary_hits += 1

        # --- Flat baseline: search L14/L15 nodes ---
        node_docs = vector_search(
            coll, qv_list,
            limit=args.recall_k,
            num_candidates=args.num_candidates,
            filter={"doc_type": "node", "level": {"$in": [14, 15]}},
        )
        if any(d.get("properties", {}).get("word") == word_b for d in node_docs):
            flat_hits += 1

        n_done = i + 1
        if n_done % 20 == 0 or n_done == len(pairs):
            log.info(
                "progress %d/%d  bary=%.3f  flat=%.3f",
                n_done, len(pairs),
                bary_hits / n_done,
                flat_hits / n_done,
            )

    n = len(pairs)
    bary_r = bary_hits / n
    flat_r = flat_hits / n
    delta = bary_r - flat_r
    log.info(
        "recall@%d  bary=%.4f (%d/%d)  flat=%.4f (%d/%d)  delta=%+.4f",
        args.recall_k, bary_r, bary_hits, n, flat_r, flat_hits, n, delta,
    )
    if delta > 0:
        log.info("BaryGraph beats flat by %.1f%%", delta * 100)
    elif delta < 0:
        log.info("flat beats BaryGraph by %.1f%%", -delta * 100)
    else:
        log.info("no difference between BaryGraph and flat")


if __name__ == "__main__":
    run()
