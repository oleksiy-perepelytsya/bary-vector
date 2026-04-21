"""Absorb orphan L14 word CMs into nearest existing L14 BE.

L15 orphan re-entry happens earlier, inside s04_l15_edges (it must
precede s05_word_vectors). This stage handles L14 only.

Each orphan word is paired with its nearest existing L14 BE; the new BE
inherits ``edge_type`` / ``type_vector`` / ``q`` from that partner (no new
embedding call) — see CLAUDE.md Stage 6.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib.bary_vec import compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from lib.match import nearest_row
from scripts._base import bootstrap, finish

STAGE = "07_orphan_reentry"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start processed=%d dry_run=%s", cp.processed, args.dry_run)

    orphans = list(
        coll.find(
            {"doc_type": "node", "node_type": "word", "level": 14,
             "parent_edge_id": None, "vector": {"$ne": None}},
            {"_id": 1, "vector": 1},
        )
    )
    bes = list(
        coll.find(
            {"doc_type": "baryedge", "level": 14},
            {"_id": 1, "vector": 1, "edge_type": 1, "type_vector": 1, "q": 1},
        )
    )
    log.info("L14 orphans=%d existing L14 BEs=%d", len(orphans), len(bes))

    if not orphans or not bes:
        cp.total = len(orphans)
        if not args.dry_run:
            finish(cp, settings, log)
        return

    BEV = np.asarray([b["vector"] for b in bes], dtype=np.float32)
    edge_docs = []
    parent_ups: list[UpdateOne] = []
    for o in orphans:
        ov = np.asarray(o["vector"], dtype=np.float32)
        bi, _ = nearest_row(ov, BEV)
        partner = bes[bi]
        tv = np.asarray(partner["type_vector"], dtype=np.float32)
        q = float(partner["q"])
        bv = compute_bary_vec(ov, BEV[bi], tv, q)
        edge_docs.append(
            baryedge(o["_id"], partner["_id"], 14, bv, q,
                     edge_type=partner.get("edge_type"), type_vector=tv,
                     source="inferred", confidence=q)
        )
    if not args.dry_run:
        res = coll.insert_many(edge_docs)
        now = datetime.now(timezone.utc)
        for o, eid in zip(orphans, res.inserted_ids, strict=True):
            parent_ups.append(
                UpdateOne({"_id": o["_id"]}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
            )
        coll.bulk_write(parent_ups, ordered=False)

    cp.processed = len(edge_docs)
    cp.total = len(edge_docs)
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
