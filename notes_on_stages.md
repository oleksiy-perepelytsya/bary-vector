# Stage notes — accepted findings (not yet fixed/addressed)

## s03 / s04 — kaikki duplicate entries are kept, not deduped
Exact-duplicate sense entries (same word, pos, lang, gloss from separate kaikki
entries) remain separate nodes; s04 pairs them as L15 "merge" edges (q=1.0),
consuming 2 sense nodes each that could otherwise pair cross-word.
- Measured: ~19.2% of top-of-stream L15 edges are same-lang same-gloss dups;
  ~9.7% same-word cross-lang (legit homograph links, not dups); 0 polysemous
  same-word merges (polysemy_floor working).
- All L15 edges are untyped (edge_type=None); typed only at L14 (s06).
- Acceptance: keep for now. Follow-up: dedupe (word,pos,lang,gloss) at s03
  ingest, merging etymology/alternates → free ~19% pairing capacity. Requires
  re-run of affected stages; not applicable mid-s04.

## s04 — effective embed rate ~15-17/s (below the ~28/s measured)
L15 type_text batch ~34s per 512 → ETA ~3.5-4 days (not 2-3). Accepted.
Follow-up: investigate batch-size ceiling / type_text+nb() overhead.

## s04 — checkpoint JSON only saved at finish()
cp.processed updates in memory per batch; the file stays processed:0 until
finish(). Live progress must be read via Mongo count of
{doc_type:baryedge, level:15}. Accepted; follow-up: periodic flush / metrics
feed (Babylon Bridge).

## lib/db.py — global socketTimeoutMS=120000
Long reads (e.g. full-collection $sample over 12.7M) still time out. The s04
crash was fixed via covering index {doc_type,node_type,level} + cap fallback
(count 12.7M in ~4s). Avoid heavy ad-hoc aggregations during pipeline runs.

## lib/match.py — ANN match uses Gaussian 4096->1024 projection
Reduces HNSW index ~210GB->55GB so pairing fits RAM; ranking cosine measured
in reduced space (approximate, equivalent recall on synthetic). Accepted
design decision.

## Global — pre-existing test failures (unrelated to pipeline)
test_sense_node_schema + test_assoc_search fail since float32 storage commit
fca4f80 (tests not updated). Out of scope.