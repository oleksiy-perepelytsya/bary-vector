"""Embed sense glosses (nomic-embed-text via ollama, or FakeEmbedder).

Reads ``senses.jsonl`` from stage 01, batch-embeds the ``embed_text`` field,
and writes ``senses_embedded.jsonl`` with the ``vector`` field populated.
Atomic-rename on completion; resumable by line count.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import orjson

from lib import checkpoint as cp_mod
from lib.embed import get_embedder
from lib.schema import SENSES_EMBEDDED_FILENAME, SENSES_FILENAME
from scripts._base import bootstrap, finish

STAGE = "02_embed"


def _batched(it, n):
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    src = Path(settings.parsed_dir) / SENSES_FILENAME
    log.info("start src=%s processed=%d dry_run=%s", src, cp.processed, args.dry_run)

    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run stage 01_parse first")

    embedder = get_embedder(settings)
    batch_n = args.batch_size or settings.embed_batch_size
    out_tmp = Path(settings.parsed_dir) / (SENSES_EMBEDDED_FILENAME + ".tmp")

    skip = cp.processed
    n = 0

    def _lines():
        with src.open("rb") as f:
            for line in f:
                yield orjson.loads(line)

    with out_tmp.open("wb" if skip == 0 else "ab") as out:
        for batch in _batched(_lines(), batch_n):
            if n + len(batch) <= skip:
                n += len(batch)
                continue
            if args.limit and n - skip >= args.limit:
                break
            texts = [rec["embed_text"] for rec in batch]
            vecs = embedder.embed(texts)
            for rec, v in zip(batch, vecs, strict=True):
                n += 1
                if n <= skip:
                    continue
                rec["vector"] = v.tolist()
                if not args.dry_run:
                    out.write(orjson.dumps(rec) + b"\n")
            cp.processed = n
            cp_mod.save(cp, settings)
            if n % (batch_n * 50) == 0:
                log.info("… embedded %d senses", n)

    cp.processed = n
    cp.total = n
    log.info("embedded %d senses (dim=%d)", n, embedder.dim)

    if not args.dry_run:
        os.replace(out_tmp, Path(settings.parsed_dir) / SENSES_EMBEDDED_FILENAME)
        finish(cp, settings, log)
    else:
        log.info("dry-run: output not committed")


if __name__ == "__main__":
    run()
