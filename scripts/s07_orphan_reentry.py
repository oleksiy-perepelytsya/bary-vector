"""Absorb orphan L14 word CMs into nearest existing L14 BE.

L15 orphan re-entry happens earlier, inside s04_l15_edges (it must
precede s05_word_vectors). This stage handles L14 only.
"""

from __future__ import annotations

from collections.abc import Sequence

from lib.db import get_collection
from scripts._base import bootstrap

STAGE = "07_orphan_reentry"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start last_id=%s processed=%d dry_run=%s", cp.last_id, cp.processed, args.dry_run)
    _ = coll  # connection acquired; core logic follows in a later PR
    raise NotImplementedError(f"{STAGE}: implement in follow-up PR")


if __name__ == "__main__":
    run()
