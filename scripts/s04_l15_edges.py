"""Cosine-driven greedy L15 BaryEdge formation + L15 orphan re-entry.

L15 orphan re-entry MUST complete here (before s05_word_vectors) — see
v0.4 §2.4: word vectors depend on the finalized set of L15 BEs.
"""

from __future__ import annotations

from collections.abc import Sequence

from lib.db import get_collection
from scripts._base import bootstrap

STAGE = "04_l15_edges"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start last_id=%s processed=%d dry_run=%s", cp.last_id, cp.processed, args.dry_run)
    _ = coll  # connection acquired; core logic follows in a later PR
    raise NotImplementedError(f"{STAGE}: implement in follow-up PR")


if __name__ == "__main__":
    run()
