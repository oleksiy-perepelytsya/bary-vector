from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Sequence

from lib import checkpoint as cp_mod
from lib.checkpoint import Checkpoint
from lib.config import Settings
from lib.log import get_logger, setup_logging

# stage name → module path (run() entrypoint)
STAGES: dict[str, str] = {
    "01_parse": "scripts.s01_parse",
    "02_embed": "scripts.s02_embed",
    "03_insert_nodes": "scripts.s03_insert_nodes",
    "04_l15_edges": "scripts.s04_l15_edges",
    "05_word_vectors": "scripts.s05_word_vectors",
    "06_l14_edges": "scripts.s06_l14_edges",
    "07_orphan_reentry": "scripts.s07_orphan_reentry",
    "08_metabary": "scripts.s08_metabary",
    "10_index": "scripts.s10_index",
}


def make_parser(stage: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"bary {stage}", description=f"BaryGraph pipeline stage {stage}"
    )
    p.add_argument("--reset", action="store_true", help="discard checkpoint and start over")
    p.add_argument("--limit", type=int, default=None, help="process at most N items (dev)")
    p.add_argument("--dry-run", action="store_true", help="do not write to MongoDB")
    p.add_argument("--batch-size", type=int, default=None, help="override BATCH_SIZE")
    p.add_argument("--kaikki-path", type=str, default=None, help="override KAIKKI_PATH")
    return p


def bootstrap(
    stage: str, argv: Sequence[str] | None = None
) -> tuple[Settings, argparse.Namespace, logging.Logger, Checkpoint]:
    """Common entrypoint plumbing for every pipeline stage."""
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger(stage)

    args = make_parser(stage).parse_args(argv)

    if args.reset:
        cp_mod.reset(stage, settings)
        log.info("checkpoint reset")

    cp = cp_mod.load(stage, settings) or Checkpoint(stage=stage)
    cp_mod.save(cp, settings)  # write initial / touched checkpoint

    log.info(
        "settings: db=%s coll=%s kaikki=%s fake_embed=%s",
        settings.mongo_db,
        settings.mongo_collection,
        args.kaikki_path or settings.kaikki_path,
        settings.fake_embed,
    )
    return settings, args, log, cp


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: bary <stage> [options]\n\nstages:")
        for name in STAGES:
            print(f"  {name}")
        return 0 if argv else 1
    stage, rest = argv[0], argv[1:]
    if stage not in STAGES:
        print(f"unknown stage: {stage}", file=sys.stderr)
        return 2
    mod = importlib.import_module(STAGES[stage])
    mod.run(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
