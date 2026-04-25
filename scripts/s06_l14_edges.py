"""Kaikki-relation-driven L14 BaryEdge formation (fermion order, v0.4 §7.1).

Iterates the six fermion tiers in priority order. Within each tier every
word's kaikki relations of the tier's kind(s) are considered; words that
already carry a ``parent_edge_id`` are skipped (unique-parent invariant).

``v(type) = embed(TYPE_SENTENCES[edge_type])`` — embedded once per
edge_type, not per pair.

Safeguard: refuses to run if any L14 BaryEdge already exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib.bary_vec import TYPE_SENTENCES, compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from lib.embed import get_embedder
from lib.match import FERMION_TIERS
from scripts._base import bootstrap, finish

STAGE = "06_l14_edges"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start processed=%d dry_run=%s", cp.processed, args.dry_run)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": 14}, limit=1):
            raise RuntimeError(
                "L14 baryedges already present — re-running would double-parent words. "
                "Drop them and use --reset, or pass --force."
            )

    # Load all L14 word nodes that have a vector (stage 05 must have run).
    cur = coll.find(
        {"doc_type": "node", "node_type": "word", "level": 14, "vector": {"$ne": None}},
        {"_id": 1, "vector": 1, "properties.word": 1, "properties.pos": 1,
         "properties.relations": 1},
    )
    words = list(cur)
    if args.limit:
        words = words[: args.limit]
    log.info("loaded %d L14 word nodes with vectors", len(words))

    by_key: dict[tuple[str, str], int] = {}
    by_word: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        p = w["properties"]
        by_key[(p["word"], p["pos"])] = i
        by_word.setdefault(p["word"], []).append(i)
    V = np.asarray([w["vector"] for w in words], dtype=np.float32)
    ids = [w["_id"] for w in words]

    # Embed TYPE_SENTENCES once.
    embedder = get_embedder(settings)
    et_keys = list(TYPE_SENTENCES)
    et_vecs = embedder.embed([TYPE_SENTENCES[k] for k in et_keys])
    type_vec: dict[str, np.ndarray] = dict(zip(et_keys, et_vecs, strict=True))

    paired: set[int] = set()
    n_edges = 0

    for tier in FERMION_TIERS:
        q_seed = settings.q_seeds[tier.q_seed_key]
        tv = type_vec[tier.edge_type]
        edge_docs = []
        pair_idxs: list[tuple[int, int]] = []
        for i, w in enumerate(words):
            if i in paired:
                continue
            for rel in w["properties"].get("relations", []):
                if rel["kind"] not in tier.kaikki_fields:
                    continue
                # Target may exist under any pos; prefer same pos, else first match.
                target = rel["word"]
                cand = by_key.get((target, w["properties"]["pos"]))
                if cand is None:
                    cands = by_word.get(target, [])
                    cand = cands[0] if cands else None
                if cand is None or cand == i or cand in paired or i in paired:
                    continue
                bv = compute_bary_vec(V[i], V[cand], tv, q_seed)
                edge_docs.append(
                    baryedge(ids[i], ids[cand], 14, bv, q_seed,
                             accumulated_weight=q_seed,
                             edge_type=tier.edge_type,
                             type_vector=tv, source="ingested", confidence=1.0)
                )
                pair_idxs.append((i, cand))
                paired.add(i)
                paired.add(cand)
                break  # one parent per word per tier — move to next word
        if edge_docs and not args.dry_run:
            res = coll.insert_many(edge_docs)
            now = datetime.now(timezone.utc)
            ups = []
            for (a, b), eid in zip(pair_idxs, res.inserted_ids, strict=True):
                ups.append(UpdateOne({"_id": ids[a]},
                                     {"$set": {"parent_edge_id": eid, "updated_at": now}}))
                ups.append(UpdateOne({"_id": ids[b]},
                                     {"$set": {"parent_edge_id": eid, "updated_at": now}}))
            coll.bulk_write(ups, ordered=False)
        n_edges += len(edge_docs)
        log.info("tier %d (%s/%s): %d edges, %d words now paired",
                 tier.priority, tier.edge_type, tier.q_seed_key, len(edge_docs), len(paired))

    cp.processed = n_edges
    cp.total = n_edges
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
