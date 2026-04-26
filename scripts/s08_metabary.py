"""Form L13 triads and recurse upward toward L1.

For each unparented bridge BE at level L-1, find two unparented BEs at
level L with mutual cos > ``meta_bary_cos_threshold``. Form a MetaBary at
L-2 using the Born-rule q_MB and set ``parent_edge_id`` on all three.
Stop when a pass produces zero new triads.

Safeguard: refuses to run if any L≤13 baryedge already exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib.bary_vec import compute_metabary_vec, level_factor
from lib.db import get_collection
from lib.docs import metabary
from lib.match import greedy_unique_match, top_k_pairs
from scripts._base import bootstrap, finish

STAGE = "08_metabary"


def _load_unparented_bes(coll, level: int) -> list[dict]:
    return list(
        coll.find(
            {"doc_type": "baryedge", "level": level, "parent_edge_id": None},
            {"_id": 1, "vector": 1, "accumulated_weight": 1},
        )
    )


def _form_level(coll, child_level: int, bridge_level: int, threshold: float,
                alpha: float, dry_run: bool) -> int:
    """Form MetaBary at ``child_level - 2`` from children@L and bridges@L-1."""
    children = _load_unparented_bes(coll, child_level)
    bridges = _load_unparented_bes(coll, bridge_level)
    if len(children) < 2 or not bridges:
        return 0

    CV = np.asarray([c["vector"] for c in children], dtype=np.float32)
    # 1. Greedy mutual-cosine matching among children (similarity is between
    #    the two children, NOT bridge↔child — see CLAUDE.md Stage 7).
    pairs = greedy_unique_match(top_k_pairs(CV), threshold=threshold)
    if not pairs:
        return 0

    # 2. Assign each pair the nearest unused bridge (by pair-centroid cosine).
    BV = np.asarray([b["vector"] for b in bridges], dtype=np.float32)
    bridge_taken: set[int] = set()
    triads: list[tuple[int, int, int, float]] = []  # (ci, cj, bi, q_pair)
    for ci, cj, q_pair in pairs:
        centroid = CV[ci] + CV[cj]
        n = float(np.linalg.norm(centroid))
        centroid = centroid / n if n else centroid
        sims = BV @ centroid
        order = np.argsort(-sims)
        for bi in order:
            if int(bi) not in bridge_taken:
                bridge_taken.add(int(bi))
                triads.append((ci, cj, int(bi), q_pair))
                break

    if not triads or dry_run:
        return len(triads)

    mb_level = child_level - 2
    docs = []
    for ci, cj, bi, _ in triads:
        w1 = float(children[ci]["accumulated_weight"])
        w2 = float(children[cj]["accumulated_weight"])
        w3 = float(bridges[bi]["accumulated_weight"])
        vec, q_mb_raw = compute_metabary_vec(CV[ci], CV[cj], BV[bi], w1, w2, w3)
        acc_w = q_mb_raw * level_factor(mb_level, alpha)
        docs.append(metabary(children[ci]["_id"], children[cj]["_id"], mb_level, vec,
                             q_mb_raw, acc_w))
    res = coll.insert_many(docs)
    now = datetime.now(timezone.utc)
    ups: list[UpdateOne] = []
    for (ci, cj, bi, _), eid in zip(triads, res.inserted_ids, strict=True):
        for cm_id in (children[ci]["_id"], children[cj]["_id"], bridges[bi]["_id"]):
            ups.append(UpdateOne({"_id": cm_id},
                                 {"$set": {"parent_edge_id": eid, "updated_at": now}}))
    coll.bulk_write(ups, ordered=False)
    return len(triads)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha
    log.info("start processed=%d dry_run=%s cos_threshold=%.2f alpha=%.2f",
             cp.processed, args.dry_run, thr, alpha)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": {"$lte": 13}}, limit=1):
            raise RuntimeError(
                "MetaBary docs (level ≤13) already present — re-running would "
                "double-parent BEs. Drop them and use --reset, or pass --force."
            )

    total = 0
    child_level = 15
    while child_level - 2 >= 1:
        bridge_level = child_level - 1
        n = _form_level(coll, child_level, bridge_level, thr, alpha, args.dry_run)
        log.info("L%d MetaBary: children@L%d bridges@L%d → %d triads",
                 child_level - 2, child_level, bridge_level, n)
        total += n
        if n == 0:
            break
        child_level -= 1

    cp.processed = total
    cp.total = total
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
