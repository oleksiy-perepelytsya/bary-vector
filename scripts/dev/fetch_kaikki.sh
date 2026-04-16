#!/usr/bin/env bash
# Reproducible, resumable download of the English kaikki.org dump.
# Idempotent: skips if the target file already exists and is non-trivial.

set -euo pipefail

URL="${KAIKKI_URL:-https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl}"
DEST="${KAIKKI_PATH:-data/kaikki-en.jsonl}"
MIN_BYTES=$((100 * 1024 * 1024))

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  sz=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST")
  if (( sz > MIN_BYTES )); then
    echo "kaikki: already present ($sz bytes) — skipping download"
    exit 0
  fi
  echo "kaikki: partial file ($sz bytes), resuming"
fi

# curl -C - supports resume if the server sends Accept-Ranges.
curl -L --fail --retry 3 --retry-delay 5 -C - -o "$DEST" "$URL"

sz=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST")
if (( sz < MIN_BYTES )); then
  echo "kaikki: download too small ($sz bytes) — aborting" >&2
  exit 1
fi
echo "kaikki: $DEST  ($sz bytes)"
