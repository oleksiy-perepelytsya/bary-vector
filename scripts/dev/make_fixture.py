"""Extract a small closed synonym/antonym cluster from the full kaikki dump
into tests/fixtures/kaikki-sample.jsonl. Run via `make fixture`."""

from __future__ import annotations

import sys
from pathlib import Path

import orjson

from lib.config import Settings

WORDS = {
    "happy", "glad", "joyful", "cheerful", "content", "merry",
    "sad", "unhappy", "miserable", "gloomy", "sorrowful",
}
OUT = Path("tests/fixtures/kaikki-sample.jsonl")


def main() -> int:
    settings = Settings.load()
    src = settings.kaikki_path
    if not src.exists():
        print(f"source dump not found: {src}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("rb") as f, OUT.open("wb") as out:
        for line in f:
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if obj.get("lang_code") == "en" and obj.get("word") in WORDS:
                out.write(line)
                n += 1
    print(f"wrote {n} entries → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
