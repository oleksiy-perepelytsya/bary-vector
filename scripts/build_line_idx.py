"""Build a line-index for a large JSONL file for O(1) seek by line number.

Primarily used by ``s03_insert_nodes`` to resume mid-file: the pipeline reads
``senses_embedded.jsonl`` sequentially, so a crash requires re-reading from the
top. This builder emits a compact array of byte offsets (one uint64 per line)
plus a ``failed_lines.txt`` listing any line that fails to ``orjson.loads``
(so we can skip + investigate rather than silently lose records).

Outputs (next to the source file, resolved via Settings):
  <src>.idx          uint64 offsets, little-endian, offsets[i] = byte offset
                     at which line i (0-indexed) starts. Final entry = file size.
  failed_lines.txt   "<1-indexed line>\\t<byte offset>\\n" for each unparseable line.

Usage:
  python3 -m scripts.build_line_idx            # reads Settings-parsed senses_embedded.jsonl
  python3 -m scripts.build_line_idx data/parsed_all/senses_embedded.jsonl
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import orjson

from lib.config import Settings
from lib.schema import SENSES_EMBEDDED_FILENAME

IDX_SUFFIX = ".idx"
FAILED_SUFFIX = "failed_lines.txt"


def _resolve_path(path_arg: str | None, settings: Settings) -> Path:
    if path_arg:
        return Path(path_arg)
    return Path(settings.parsed_dir) / SENSES_EMBEDDED_FILENAME


def build(src: Path, log=None) -> None:
    idx_path = src.with_suffix(src.suffix + IDX_SUFFIX)
    failed_path = src.with_name(FAILED_SUFFIX)
    n_lines = 0
    n_failed = 0
    total_bytes = 0
    offsets: list[int] = []
    failed: list[tuple[int, int]] = []

    with src.open("rb") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(start)
            try:
                orjson.loads(line)
            except orjson.JSONDecodeError:
                failed.append((n_lines + 1, start))
                n_failed += 1
            n_lines += 1
            total_bytes += len(line)

    offsets.append(total_bytes)  # sentinel: end-of-file offset

    # Write binary idx: flat uint64 little-endian.
    buf = struct.pack(f"<{len(offsets)}Q", *offsets)
    idx_path.write_bytes(buf)

    # Write failed-lines report (1-indexed line + byte offset).
    failed_path.write_text(
        "".join(f"{ln}\t{off}\n" for ln, off in failed),
        encoding="utf-8",
    )

    if log is not None:
        log.info(
            "line index built: %d lines, %d bad, %d bytes -> %s (%.1f MB)",
            n_lines, n_failed, total_bytes, idx_path, len(buf) / 1e6,
        )
        if n_failed:
            log.warning("failed lines recorded in %s: %s", failed_path, [ln for ln, _ in failed])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default=None, help="JSONL path (default: settings senses_embedded.jsonl)")
    args = p.parse_args(argv)

    from lib.log import get_logger
    log = get_logger("build_line_idx")

    settings = Settings.load()
    src = _resolve_path(args.path, settings)
    if not src.exists():
        log.error("source not found: %s", src)
        return 1

    log.info("building line index for %s", src)
    build(src, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
