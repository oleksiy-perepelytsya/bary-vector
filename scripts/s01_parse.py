"""Parse kaikki JSONL → intermediate sense/word records.

Reads ``Settings.kaikki_path`` line-by-line, runs each record through
:func:`lib.parse.parse_entry`, and writes ``words.jsonl`` / ``senses.jsonl``
under ``Settings.parsed_dir``. No MongoDB contact.

Output is written to ``*.tmp`` and atomically renamed on success so a
crash mid-run never leaves a half-written file for stage 02 to consume.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import orjson

from lib import checkpoint as cp_mod
from lib.parse import parse_entry
from lib.schema import SENSES_FILENAME, WORDS_FILENAME
from scripts._base import bootstrap, finish

STAGE = "01_parse"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    src = Path(args.kaikki_path or settings.kaikki_path)
    log.info("start src=%s processed=%d dry_run=%s", src, cp.processed, args.dry_run)

    if not src.exists():
        raise FileNotFoundError(f"kaikki dump not found: {src} — run `make fetch-kaikki`")

    out_dir = Path(settings.parsed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    words_tmp = out_dir / (WORDS_FILENAME + ".tmp")
    senses_tmp = out_dir / (SENSES_FILENAME + ".tmp")

    n_lines = n_words = n_senses = 0
    skip = cp.processed  # resume: skip lines already consumed

    with src.open("rb") as f, words_tmp.open(
        "wb" if skip == 0 else "ab"
    ) as wf, senses_tmp.open("wb" if skip == 0 else "ab") as sf:
        for line in f:
            n_lines += 1
            if n_lines <= skip:
                continue
            if args.limit and (n_lines - skip) > args.limit:
                break
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            parsed = parse_entry(obj)
            if parsed is None:
                continue
            pw, senses = parsed
            if not args.dry_run:
                wf.write(orjson.dumps(pw.to_dict()) + b"\n")
                for s in senses:
                    sf.write(orjson.dumps(s.to_dict()) + b"\n")
            n_words += 1
            n_senses += len(senses)

            cp.processed = n_lines
            if n_lines % 10000 == 0:
                cp_mod.save(cp, settings)
                log.info("… %d lines → %d words, %d senses", n_lines, n_words, n_senses)

    cp.processed = n_lines
    cp.total = n_lines
    log.info("parsed %d lines → %d words, %d senses", n_lines, n_words, n_senses)

    if not args.dry_run:
        os.replace(words_tmp, out_dir / WORDS_FILENAME)
        os.replace(senses_tmp, out_dir / SENSES_FILENAME)
        finish(cp, settings, log)
    else:
        log.info("dry-run: outputs not committed")


if __name__ == "__main__":
    run()
