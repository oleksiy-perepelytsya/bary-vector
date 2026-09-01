#!/bin/bash
URL="https://kaikki.org/dictionary/All%20languages%20combined/kaikki.org-dictionary-all.jsonl"
OUT="/workspace/bary-vector/data/kaikki.org-dictionary-all.jsonl"
EXPECTED=27143647418

mkdir -p "$(dirname "$OUT")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for attempt in $(seq 1 500); do
  log "attempt $attempt starting (have $(stat -c%s "$OUT" 2>/dev/null || echo 0) / $EXPECTED bytes)"
  curl -fL -C - \
    --speed-limit 10240 --speed-time 60 \
    --retry 5 --retry-delay 10 --retry-all-errors \
    --connect-timeout 30 --max-time 0 \
    -o "$OUT" "$URL"
  rc=$?
  actual=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$actual" -ge "$EXPECTED" ]; then
    log "DONE: $actual / $EXPECTED bytes (curl rc=$rc)"
    exit 0
  fi
  log "stalled/failed rc=$rc at $actual / $EXPECTED bytes; resuming in 15s"
  sleep 15
done

log "GAVE UP after 500 attempts"
exit 1
