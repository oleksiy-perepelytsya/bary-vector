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

    orphan_ids: list = []
    OV = np.empty((500_000, settings.embed_dim), dtype=np.float32)
    for i, doc in enumerate(coll.find(
        {"doc_type": "node", "node_type": "word", "level": 14,
         "parent_edge_id": None, "vector": {"$ne": None}},
        {"_id": 1, "vector": 1},
    )):
        orphan_ids.append(doc["_id"])
        OV[i] = doc["vector"]
    n_orphans = len(orphan_ids)
    OV = OV[:n_orphans]

    be_ids: list = []
    be_meta: list[dict] = []  # edge_type, type_vector, q, accumulated_weight
    BEV = np.empty((500_000, settings.embed_dim), dtype=np.float32)
    for i, doc in enumerate(coll.find(
        {"doc_type": "baryedge", "level": 14},
        {"_id": 1, "vector": 1, "edge_type": 1, "type_vector": 1, "q": 1,
         "accumulated_weight": 1},
    )):
        be_ids.append(doc["_id"])
        be_meta.append(doc)
        BEV[i] = doc["vector"]
    n_bes = len(be_ids)
    BEV = BEV[:n_bes]

    log.info("L14 orphans=%d existing L14 BEs=%d", n_orphans, n_bes)

    if not n_orphans or not n_bes:
        cp.total = n_orphans
        if not args.dry_run:
            finish(cp, settings, log)
        return

    # Chunked nearest-BE search: full OV@BEV.T would be ~(300K×150K×4 B)=180 GB.
    # Process orphans in chunks so the intermediate sims matrix stays small.
    CHUNK = 1024
    best_bi = np.empty(n_orphans, dtype=np.int64)
    for start in range(0, n_orphans, CHUNK):
        end = min(start + CHUNK, n_orphans)
        best_bi[start:end] = np.argmax(OV[start:end] @ BEV.T, axis=1)

    batch_n = args.batch_size or settings.batch_size
    edge_docs = []
    parent_ups: list[UpdateOne] = []
    for idx, (oid, bi) in enumerate(zip(orphan_ids, best_bi.tolist(), strict=True)):
        partner = be_meta[bi]
        tv = np.asarray(partner["type_vector"], dtype=np.float32)
        q = float(partner["q"])
        acc_w = float(partner.get("accumulated_weight", q))
        bv = compute_bary_vec(OV[idx], BEV[bi], tv, q)
        edge_docs.append(
            baryedge(oid, be_ids[bi], 14, bv, q,
                     accumulated_weight=acc_w,
                     edge_type=partner.get("edge_type"), type_vector=tv,
                     source="inferred", confidence=q)
        )
    if not args.dry_run:
        now = datetime.now(timezone.utc)
        all_eids: list = []
        for start in range(0, len(edge_docs), batch_n):
            res = coll.insert_many(edge_docs[start : start + batch_n])
            all_eids.extend(res.inserted_ids)
        for oid, eid in zip(orphan_ids, all_eids, strict=True):
            parent_ups.append(
                UpdateOne({"_id": oid}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
            )
        coll.bulk_write(parent_ups, ordered=False)

    cp.processed = len(edge_docs)
    cp.total = len(edge_docs)
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
