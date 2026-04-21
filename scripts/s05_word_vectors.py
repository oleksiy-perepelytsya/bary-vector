"""Compute L14 word vectors from BE-centroids + orphan senses + λ·φ(W).

    v(W) = normalize( Σ v(BE_i) + Σ v(orphan_sense_j) + λ·φ(W) )

No embedding call. Strict stage boundary: depends on the *finalized* L15
BE set from stage 04 (including orphan re-entry).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib import checkpoint as cp_mod
from lib.bary_vec import word_length_feature, word_vector
from lib.db import get_collection
from scripts._base import bootstrap, finish

STAGE = "05_word_vectors"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    lam = settings.word_length_lambda
    log.info("start processed=%d dry_run=%s λ=%.2f", cp.processed, args.dry_run, lam)

    q = {"doc_type": "node", "node_type": "word", "level": 14}
    if cp.last_id:
        from bson import ObjectId
        q["_id"] = {"$gt": ObjectId(cp.last_id)}
    total = coll.count_documents({"doc_type": "node", "node_type": "word", "level": 14})

    batch_n = args.batch_size or settings.batch_size
    ops: list[UpdateOne] = []
    n = cp.processed

    cur = coll.find(q, {"_id": 1, "properties": 1}).sort("_id", 1)
    for w in cur:
        if args.limit and n - cp.processed >= args.limit:
            break
        props = w["properties"]
        word, pos = props["word"], props["pos"]

        # Senses of W and their parent BEs.
        sense_docs = list(
            coll.find(
                {
                    "doc_type": "node",
                    "node_type": "sense",
                    "properties.word": word,
                    "properties.pos": pos,
                },
                {"_id": 1, "vector": 1, "parent_edge_id": 1},
            )
        )
        be_ids = {s["parent_edge_id"] for s in sense_docs if s.get("parent_edge_id")}
        be_vecs = [
            np.asarray(be["vector"], dtype=np.float32)
            for be in coll.find({"_id": {"$in": list(be_ids)}}, {"vector": 1})
        ]
        orphan_vecs = [
            np.asarray(s["vector"], dtype=np.float32)
            for s in sense_docs
            if not s.get("parent_edge_id")
        ]
        if not be_vecs and not orphan_vecs:
            continue  # word with zero senses (shouldn't happen post-stage-03)

        phi = word_length_feature(word, props.get("forms") or [])
        vec = word_vector(be_vecs, orphan_vecs, length_feat=phi, lam=lam)

        ops.append(
            UpdateOne(
                {"_id": w["_id"]},
                {"$set": {"vector": vec.tolist(), "updated_at": datetime.now(timezone.utc)}},
            )
        )
        n += 1
        cp.last_id = str(w["_id"])
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
            cp.processed = n
            cp_mod.save(cp, settings)
            log.info("… %d/%d word vectors", n, total)

    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)

    cp.processed = n
    cp.total = total
    log.info("computed %d/%d L14 word vectors", n, total)
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
