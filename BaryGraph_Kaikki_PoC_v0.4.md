# BaryGraph PoC: English Kaikki Dictionary
## A Local-First Proof of Concept
**Version 0.4 · April 2026**

> **v0.4 changes from v0.3:**
> - Forest structure with unique-parent constraint (soft). Every CM has
>   at most one `parent_edge_id`. Orphans allowed.
> - L15 `v(type)` is now per-pair: embed of both parent words' lexical
>   neighborhoods (antonyms + synonyms). Much richer than generic sentence.
> - L14 word vectors = BE-centroid + orphan senses, not raw sense centroid.
> - Triadic recursion above L14: L13 MetaBary uses L14 BE as bridge,
>   L12+ is pure geometry (cosine > 0.9).
> - Removed: `is_metabary`, `hierarchy_direction`, `common_ancestor_id`,
>   lateral MetaBary, `edge_type` above L14.
> - Added: `parent_edge_id` on all nodes and BaryEdges.
> - TYPE_SENTENCES used only at L14.
> - Fermion order for L14 matching (antonyms first, synonyms last).
> - Removed: `strength`, `registry`, `summary_vector`, LLM summary stage.
>   The PoC is now embedding-only; `bary_vec` is the sole retrieval signal.

> **Level orientation:** L1 = top (most abstract), L15 = bottom (most
> concrete sense). MetaBary climbs from L to L-2 using L-1 as bridge.

---

## 1. Objective

Validate the core BaryGraph hypothesis — that relationship-aware vector
retrieval outperforms flat nearest-neighbour search — using the English
machine-readable dictionary from kaikki.org as corpus. The PoC runs
entirely on local hardware: MongoDB Community Edition with mongot for
storage and vector search, and nomic-embed-text for embeddings.

### 1.1 Why This Corpus

- **Pre-labeled relations.** Synonyms, derived forms, hypernyms, and
  etymology are explicit. BaryEdge `edge_type` comes from the corpus.
- **Built-in ground truth.** Hold out 10% of synonym links; measure
  recall against BaryEdge retrieval with zero human annotation.
- **Rich polysemy.** Words like "bank", "crane", "bark" have senses so
  distant in meaning that no embedding makes them neighbours — yet they
  are deeply related. BaryGraph surfaces this via MetaBary triads.
- **Bounded scale.** ~800K headwords, ~2–2.5M senses.
- **Translations as future cross-language bridges.** Each translation
  entry carries a `sense` gloss string that disambiguates which sense
  it corresponds to — enabling precise cross-language BaryEdge
  construction without cross-dump word matching.

### 1.2 What This PoC Validates

| BaryGraph Claim | How Kaikki Tests It |
|---|---|
| BaryEdge retrieval > flat retrieval | Hold-out recall on synonym links |
| `bary_vec` formula is useful | NN search on `bary_vec` retrieves correct CM pair |
| Forest-structure MetaBary encodes polysemy | Triad paths vs. WordNet sense clusters |
| Cross-level hierarchy via single `$graphLookup` | Forest traversal from L15 to root |
| Sense disambiguation from `_dis1` weights | Precision of sense-level edge assignment |
| Fermion-ordered matching preserves rare signals | Antonym edges survive synonym flood |

### 1.3 What This PoC Does Not Cover

- Antonym `contradicts` edges as primary eval signal (low noun coverage).
- Cross-language bridges (deferred to multi-language expansion).
- Production deployment, sharding, or cloud migration.

---

## 2. Core Equations

### 2.1 BaryEdge Vector (L14, L15)

```
bary_vec = normalize( q·v(CM₁) + q·v(CM₂) + (1−q)·v(type) )
```

where `q` is connection quality (0–1) and `v(type)` is level-dependent
(see §2.3).

### 2.2 MetaBary Vector (L13 and above)

```
meta_bary = normalize( q_MB·v(BE₁ᴸ) + q_MB·v(BE₂ᴸ) + (1−q_MB)·v(BEᴸ⁻¹) )
q_MB      = q₃² / √(q₁⁴ + q₂⁴ + q₃⁴)
```

At L13, `BE₁` and `BE₂` are L15 BaryEdges and `BEᴸ⁻¹` is the L14
BaryEdge acting as bridge. Above L13, all three components are
BaryEdges/MetaBarys from the level below.

### 2.3 v(type) Construction

**L15 — per-pair lexical neighborhood:**

```
type_text = "W_A (antonyms: a₁, a₂, …; synonyms: s₁, s₂, …); W_B (antonyms: b₁, b₂, …; synonyms: t₁, t₂, …)"
v(type)   = embed(type_text)
```

This anchors every L15 BE in the lexical neighborhood of both parent
words. The antonyms inject polarity contrast; the synonyms inject
cluster membership. The result is a type vector that captures the
relational context around the pairing, not just the pairing itself.

For same-headword sense pairs (polysemy), the type text degenerates to
one word's neighborhood. Still valid but less informative — q will
typically be lower for these pairs anyway.

For words with empty synonym/antonym sets (rare words), falls back to
`embed("W_A; W_B")`.

**L14 — TYPE_SENTENCES (fixed per edge type):**

```js
const TYPE_SENTENCES = {
  same_phenomenon: 'these two words describe the same concept',
  contradicts:     'these two words have opposite meanings',
  extends:         'one word is derived from or extends the other',
  applies_to:      'these two words share a common origin or root',
  is_instance_of:  'this relationship is a specific instance of the broader relationship',
};
v(type) = embed(TYPE_SENTENCES[edge_type])
```

L14 is the only level where TYPE_SENTENCES is used.

**L13+ — bridge vector (no embedding call):**

Above L14, the bridge BaryEdge's vector serves directly as the third
component. No v(type) computation, no TYPE_SENTENCES, no embedding call.
The bridge already encodes relational information from the levels below.

### 2.4 Word Vector (L14)

```
v(word_W) = normalize( Σᵢ v(BE_i) + Σⱼ v(sense_j) )
```

where `BE_i` are L15 BaryEdges in which one of W's senses participates,
and `sense_j` are orphan senses of W that found no partner at L15.
Unweighted sum, then normalize.

**Why BE-centroid, not sense-centroid:**
- Each L15 BE vector already encodes three signals — both senses plus
  the word-pair-context type. So the word vector absorbs relational
  information from every pairing its senses participated in.
- A word whose senses paired with diverse partners gets a vector that
  sits where its senses' relational neighborhoods overlap — exactly the
  property needed to bridge L15 BEs at L13 triad formation.
- Orphan senses still contribute their raw embedding. No information
  loss, but paired senses dominate direction (more information per
  component). This weights the word vector toward its most "connected"
  meanings — desirable behavior.

**Dependency:** L14 word vectors cannot be computed until all L15 BEs
are finalized (including orphan re-entry). Strict stage boundary.

---

## 3. Invariants

1. **Unique parent (soft):** every CM has at most one `parent_edge_id`.
   Orphans allowed. `parent_edge_id` always references a `baryedge`
   document — a node (word, sense, etc.) is a CM, never a parent edge.
2. **Triadic recursion only above L14.** No lateral edges, no cross-level
   BEs outside triads.
3. **Forest structure** — single `$graphLookup` climbs to root.
4. **BE and MetaBary interchangeable above L15** — same doc type, same
   role.
5. **Algebraically closed** — everything is
   `normalize(q·a + q·b + (1−q)·c)`, recursive.

---

## 4. Kaikki Data Structure

One JSONL line = one `(word, pos)` entry. Example: `dictionary (noun)`.

### 4.1 Top-Level Fields

```
word            "dictionary"
pos             "noun"
lang_code       "en"
forms[]         [{form: "dictionaries", tags: ["plural"]}, ...]
etymology_text  full etymology string
sounds[]        [{ipa: "/ˈdɪk.ʃə.nə.ɹi/", tags: ["Received-Pronunciation"]}, ...]
senses[]        array of sense objects (see §4.2)
translations[]  array of translation objects (see §4.3)

— Relations (word-level, NOT sense-level) —
synonyms[]         [{word: "dict", _dis1: "0 0 0 0 0 0"}, ...]
antonyms[]         (sparse — mainly adjectives/verbs, rare for nouns)
hypernyms[]        [{word: "catalog", _dis1: "0 0 0 0 0 0"}, ...]
hyponyms[]         [{word: "bilingual dictionary", _dis1: "0 0 0 0 0 0"}, ...]
meronyms[]         [{word: "vocabularium", _dis1: "0 0 0 0 0 0"}, ...]
derived[]          [{word: "dictionarial", _dis1: "0 0 0 0 0 0"}, ...]
related[]          [{word: "lexicon", _dis1: "0 0 0 0 0 0"}, ...]
coordinate_terms[] [{word: "thesaurus", _dis1: "0 0 0 0 0 0"}, ...]
```

**Critical:** All semantic relations live at the word entry level, not
inside individual senses. The `_dis1` field carries sense distribution
weights for disambiguation (see §4.4).

### 4.2 Sense-Level Fields

```
senses[i]:
  id          "en-dictionary-en-noun-en:Q23622"  (stable unique ID)
  glosses[]   ["A reference work listing words..."]
  examples[]  [{text: "...", type: "example"|"quotation"}]
  tags[]      ["broadly", "figuratively", "derogatory", ...]
  topics[]    ["computing", "engineering", "mathematics"]  (sparse)
  categories[]{name: "Computing", kind: "other", ...}
  wikidata[]  ["Q23622"]  (only on some senses)

  — Sense-level relations (sparse, minority of senses only) —
  hypernyms[]        (e.g. sense[0]: [{word: "wordbook"}])
  coordinate_terms[] (e.g. sense[0]: [{word: "thesaurus"}])
  hyponyms[]         (e.g. sense[5]: [{word: "hash table"}])
```

When a sense carries its own relation fields, these take priority over
word-level relations for that sense — no disambiguation needed.

### 4.3 Translation Structure

```
translations[i]:
  lang       "Abkhaz"
  lang_code  "ab"
  sense      "publication that explains the meanings of an ordered list of words"
  word       "ажәар"
  _dis1      "24 8 2 28 25 12"   ← non-zero: dominant sense = index 3
```

Translations carry a `sense` gloss string and non-zero `_dis1` weights —
the most precisely sense-disambiguated cross-language signal in the corpus.
Used in multi-language expansion for direct sense-level BaryEdge creation.

### 4.4 `_dis1` Sense Disambiguation

`_dis1` is a space-separated string of integers, one per sense. Each
integer is a weight indicating how strongly the relation maps to that
sense index.

```python
def assign_sense(item, sense_vectors, threshold=0.72):
    weights = [int(x) for x in item['_dis1'].split()]
    if max(weights) > 0:
        return weights.index(max(weights))          # use _dis1 directly → L15
    else:
        target_vec = embed(item['word'])
        sims = [cosine(target_vec, sv) for sv in sense_vectors]
        if max(sims) > threshold:
            return sims.index(max(sims))            # cosine fallback → L15
        else:
            return None                             # assign to word level (L14)
```

---

## 5. Hierarchy Mapping

| Level | Scale | Kaikki Source | PoC Status |
|---|---|---|---|
| 1–3 | Language family / Paradigm | Fixed: "English", "Germanic", "Indo-European" | Static scaffolding |
| 4–6 | Register / Period | `tags[]`: formal, archaic, slang, technical | Sparse — collapse to L7 if <5% coverage |
| 7–9 | Semantic field / POS cluster | `topics[]`, `pos`, sense `categories[]` | Active |
| 10–12 | Concept cluster (synset) | Clustered senses sharing hypernyms | Active |
| 13 | MetaBary (polysemy bridge) | L15 BE pairs bridged by L14 BE | Active |
| 14 | Word entry | One node per `(word, pos)` — kaikki relation matching | Active |
| 15 | Individual sense / Gloss | Each `senses[]` entry — cosine-matched BEs | Active |

---

## 6. Data Schema

Single collection `barygraph`. Two document types: `node`, `baryedge`.

### 6.1 Node

```js
{
  _id:            ObjectId(),
  doc_type:       'node',
  node_type:      'sense' | 'word' | 'synset' | 'field' | 'register' | 'stub',
  level:          Number,       // 1–15
  label:          String,
  vector:         [Number],     // 768-dim
  surface:        Number,
  rotation:       0.0,
  parent_edge_id: ObjectId() | null,   // ≤1 parent BE; null = orphan
  properties:     Object,       // see node_type table below
  created_at:     Date,
  updated_at:     Date
}
```

| node_type | level | key properties | vector source |
|---|---|---|---|
| `sense` | 15 | word, pos, sense_id, sense_idx, gloss, examples, tags, topics, wikidata | embed(gloss + examples[:2]) |
| `word` | 14 | word, pos, etymology, forms, ipa | BE-centroid + orphan senses |
| `synset` | 10–12 | hypernym, member_count | cluster centroid |
| `field` | 7–9 | name, pos_group | cluster centroid |
| `register` | 4–6 | name, tag — **may collapse to L7** | cluster centroid |
| `stub` | any | word, reason — no vector | none |

### 6.2 BaryEdge

```js
{
  _id:              ObjectId(),
  doc_type:         'baryedge',
  cm1_id:           ObjectId(),   // → node (L14/L15) or baryedge (L≤13)
  cm2_id:           ObjectId(),   // → node (L14/L15) or baryedge (L≤13)
  level:            Number,       // same as CMs at L14/L15;
                                  //   = cm1.level - 2 for MetaBary (L≤13)
  vector:           [Number],     // bary_vec — algebraic
  parent_edge_id:   ObjectId() | null,   // ≤1 parent; ALWAYS → baryedge doc; null = orphan
  connection_strength: Number,    // q (base) or q_MB (rescaled)

  // L14/L15 ONLY:
  edge_type:        String | null,  // kaikki relation (L14) or null (L15 cosine-matched)
  type_vector:      [Number],       // v(type) — per-pair embed (L15) or TYPE_SENTENCES (L14)
  q:                Number,         // 0–1
  source:           'ingested' | 'inferred' | 'manual' | 'placeholder',
  confidence:       Number,

  // DROPPED above L14:
  // edge_type, type_vector, q, source, confidence

  created_at:     Date,
  updated_at:     Date
}
```

**What's dropped above L14 and why:**

| Removed | Why |
|---|---|
| `edge_type` | Type is implicit in bridge BE vector |
| `is_metabary` flag | Everything above L14 is MB by construction |
| `hierarchy_direction` | Always upward; no lateral edges |
| `common_ancestor_id` | Forest structure makes traversal trivial |
| `TYPE_SENTENCES` above L14 | Only used at L14 for kaikki-based matching |
| Lateral MetaBary | Eliminated by unique-parent constraint |

---

## 7. Edge Types (L14/L15 only)

### 7.1 Fermion Order (L14 matching priority)

L14 BaryEdge matching follows fermion order — rarer, more informative
relations are matched first. Once a word has `parent_edge_id` set, it is
skipped at lower-priority tiers.

| Priority | edge_type | kaikki field | q_seed key | q_seed |
|---|---|---|---|---|
| 1 | `contradicts` | `antonyms[]` | `contradicts` | 0.85 |
| 2 | `applies_to` | `meronyms[]`, `holonyms[]` | `applies_to` | 0.55 |
| 3 | `is_instance_of` | `hypernyms[]`, `hyponyms[]` | `is_instance_of` | 0.65 |
| 4 | `extends` | `derived[]`, `related[]` | `extends` | 0.60 |
| 5 | `same_phenomenon` | `coordinate_terms[]` | `coordinate_terms` | 0.70 |
| 6 | `same_phenomenon` | `synonyms[]` | `synonyms` | 0.90 |

`q_seed key` is the lookup key into `Settings.q_seeds` — one per
priority tier (tiers 5/6 share `edge_type` but differ in q).

### 7.2 L15 Cosine-Matched Edges

L15 BaryEdges are formed by greedy highest-cosine matching across all
sense pairs. No edge_type — the relationship is captured entirely by
`v(type)` (lexical neighborhood embed).

| Parameter | Value |
|---|---|
| Matching method | Greedy highest-cosine first |
| q | `cos(s_A, s_B)` directly |
| Threshold (q_min) | 0.72 |
| v(type) | embed(word neighborhood text) |
| Orphan v(type) | embed(orphan's word neighborhood only) |

### 7.3 Polysemy Edges (L15, same headword)

Same-headword sense pairs are seeded into the L15 greedy-match
candidate pool with a q floor:

```
q = max(0.40, cosine(sense_vec_i, sense_vec_j))
```

Greedy matching still selects at most one parent BE per sense
(unique-parent invariant); the floor only ensures polysemous senses
remain eligible to pair with each other even when their raw cosine is
low. Any resulting BEs feed into L14 word vectors.

---

## 8. Construction Pipeline

### Stage 1 — Sense Nodes (L15)

- Parse kaikki JSONL → extract senses
- `v(sense) = embed(gloss + examples[:2])`
- Store as node: `node_type: 'sense'`, `level: 15`, `parent_edge_id: null`

### Stage 2 — Embed

- Batch embed all sense glosses via nomic-embed-text (~10 min GPU)
- One embedding call per sense

### Stage 3 — Insert Nodes

- Insert L15 sense nodes into MongoDB
- Insert L14 word nodes with placeholder vectors (computed in Stage 5)

### Stage 4 — L15 BE Formation (cosine-driven)

```
4a. Pairwise cosine among all L15 sense vectors (ANN-accelerated)
4b. Greedy match: highest cosine first, skip already-paired
4c. For each pair: build type_text (parent words + antonyms + synonyms),
    batch-embed → v(type)
4d. Compute bary_vec, set parent_edge_id on both senses
4e. Orphan re-entry: unpaired senses match with existing L15 BEs
```

**Scale note:** ~300K senses → brute-force is 90B pairs. Use ANN
(FAISS/hnswlib locally, or MongoDB vector search). Top-k neighbors per
sense, then greedy match from ranked pairs.

**Embedding cost:** ~1–2M v(type) embedding calls. Batchable at 1K per
request → ~1–2K batches.

### Stage 5 — L14 Word Vectors

```
For each word W:
  v(W) = normalize(Σ v(BE_i) for BEs holding W's senses
                 + Σ v(sense_j) for orphan senses of W)
```

Computed after L15 BE formation. No embedding call. Update the
placeholder L14 word node vectors in place.

### Stage 6 — L14 BE Formation (kaikki-driven, fermion order)

```
6a. Iterate kaikki relations in fermion order (§7.1)
6b. Skip words already paired at this priority tier
6c. embed(TYPE_SENTENCES[edge_type]) → v(type)
6d. Compute bary_vec, set parent_edge_id
```

L14 orphan re-entry runs as the next stage (`s07_orphan_reentry.py`):
each unpaired word matches the nearest existing L14 BE; the new BE
inherits that partner's `edge_type`, `type_vector`, and `q` (no new
embedding call).

### Stage 7 — L13 MetaBary (polysemy bridge)

For each L14 BE (the bridge), search L15 for two unparented BEs with
mutual `cos > 0.9`:

```
v(L13_MB) = normalize( q_MB·v(L15_BE₁) + q_MB·v(L15_BE₂) + (1−q_MB)·v(L14_BE) )
q_MB      = q_L14² / √(q_L15₁⁴ + q_L15₂⁴ + q_L14⁴)
```

This creates natural polysemy triads:

```
word "bank" + word "flow"  → L14 BE (via kaikki: related)
    ↑ bridge
sense "bank/financial" + sense "credit/loan"  → L15 BE₁
sense "bank/riverbank" + sense "flow/stream"  → L15 BE₂
    → L13 MB: financial↔river polysemy bridged by flow concept
```

### Stage 8 — L12→L1 Recursive

```
For each L-1 BE/MB:
  find two unparented L BEs/MBs with cos > 0.9
  form L-2 MetaBary using standard formula
  set both children's parent_edge_id
Orphan re-entry at each level.
Stop when no new triads form.
```

Pure geometry — no kaikki, no TYPE_SENTENCES.

### Stage 9 — Index

- Build mongot vector indexes (~4–8 hours)

### Pipeline Timing

| Stage | Duration | Blocking |
|---|---|---|
| 1–2. Parse + Embed | ~20 min | Yes |
| 3. Insert nodes | ~30 min | Yes |
| 4. L15 BE formation | ~45 min | Yes |
| 5. L14 word vectors | ~5 min | Yes |
| 6. L14 BE formation | ~30 min | Yes |
| 7. L13 MetaBary | ~20 min | Yes |
| 8. L12→L1 recursive | ~1–2 hours | Yes |
| 9. Index | ~4–8 hours | Yes |

**Queryable: ~7–12 hours.**

### Resumability

```json
// pipeline_state/{stage_name}.json
{ "last_id": "ObjectId(...)", "processed": 1240000, "total": 2500000 }
```

---

## 9. MongoDB Infrastructure

Database: `barygraph_poc`
Collection: `barygraph`

### 9.1 Standard Indexes

```js
db.barygraph.createIndex({ doc_type: 1, level: 1 })
db.barygraph.createIndex({ cm1_id: 1 })
db.barygraph.createIndex({ cm2_id: 1 })
db.barygraph.createIndex({ node_type: 1 })
db.barygraph.createIndex({ edge_type: 1, level: 1 })
db.barygraph.createIndex({ parent_edge_id: 1 })           // NEW in v0.4
db.barygraph.createIndex({ 'properties.word': 1, 'properties.pos': 1 })
db.barygraph.createIndex({ 'properties.sense_id': 1 })
```

### 9.2 Vector Index

```json
// bary_vec (all doc_types)
{
  "fields": [
    { "type": "vector", "path": "vector", "numDimensions": 768, "similarity": "cosine" },
    { "type": "filter", "path": "doc_type" },
    { "type": "filter", "path": "level" },
    { "type": "filter", "path": "edge_type" },
    { "type": "filter", "path": "node_type" }
  ]
}
```

---

## 10. Query Patterns

### Baseline (flat)
```js
filter: { doc_type: 'node', level: { $in: [14, 15] } }
```

### BaryGraph retrieval
```js
filter: { doc_type: { $in: ['node', 'baryedge'] }, level: { $in: [14, 15] } }
// + $lookup cm1_id, cm2_id
```

### Hierarchy traversal (forest walk via parent_edge_id)
```js
db.barygraph.aggregate([
  { $match: { _id: startNodeId }},
  { $graphLookup: {
      from: 'barygraph',
      startWith: '$parent_edge_id',
      connectFromField: 'parent_edge_id',
      connectToField: '_id',
      as: 'upward_chain',
      maxDepth: 15
  }},
  { $project: {
      chain: { $sortArray: { input: '$upward_chain', sortBy: { level: 1 } } }
  }}
])
```

Forest structure means no cycle handling, no `restrictSearchWithMatch`
needed. Single `$graphLookup` walks from any node to root.

---

## 11. Evaluation

### 11.1 Primary: Held-Out Synonym Recall@20

1. Hold out 10% of `synonyms[]` links → `data/holdout.json`
   (**synonyms only** — antonyms too sparse for noun-heavy corpus)
2. Ingest 90%
3. Query `embed(word_A gloss)`, filter `doc_type: 'baryedge'`, top-20
4. **Success:** word_B in CM lineage of any returned BaryEdge
5. BaryGraph recall@20 vs. flat recall@20

### 11.2 Secondary Metrics

| Metric | Method |
|---|---|
| `bary_vec` precision | BaryEdge 5-NN should include cm1 and cm2 |
| Disambiguation accuracy | Sample 100 word-level relations; manually verify sense assignment |
| `_dis1` vs cosine | Precision when `_dis1 > 0` vs cosine fallback |
| Forest coherence | Manual top-50 triad inspection at L13 |
| Hierarchy correctness | Forest paths vs. `topics[]` / `categories[]` |
| Fermion order impact | Recall with/without antonym-first priority |
| Orphan rate per level | Track % of CMs with parent_edge_id = null |

---

## 12. Defaults and Tuning Parameters

| ID | Parameter | Default | Rationale |
|---|---|---|---|
| R1 | L15 matching order | Greedy highest-cosine first | Sense-level kaikki too sparse to drive ordering |
| R2 | L15 q value | `cos(s_A, s_B)` directly | Natural quality signal |
| R3 | L15 orphan v(type) | embed(orphan's word neighborhood only) | One-sided but consistent |
| R4 | Antonym/synonym _dis1 filtering | No filtering initially | Simpler; revisit if eval shows noise |
| R5 | L15 matching threshold | 0.72 (matches disambiguation threshold) | Below this, sense remains orphan |

---

## 13. Resource Budget

| Stage | Duration | Blocking? |
|---|---|---|
| Parse + Embed | ~20 min | Yes |
| Insert nodes | ~30 min | Yes |
| L15 BE formation | ~45 min | Yes |
| L14 word vectors | ~5 min | Yes |
| L14 BE formation | ~30 min | Yes |
| L13+ MetaBary | ~1–2 hours | Yes |
| Build indexes | ~4–8 hours | Yes |
| **Queryable** | **~7–12 hours** | |

Hardware: 8–16 GB GPU VRAM, 32–64 GB RAM, 150–200 GB disk, 8+ cores.
Cost: zero (all open-source).

---

## 14. Deviations From Parent Spec (BaryGraph v1.1)

| v1.1 Section | Deviation | Reason |
|---|---|---|
| §4.2 — pairwise cosine scan all levels | Explicit-relation seeding at L14/L15 | O(n²) infeasible |
| §7 — MongoDB Atlas | Community + mongot | Local-first PoC |
| Embedding dimensions | 768-dim (not 1536) | Glosses are short |
| `v(type)` as edge label | Per-pair neighborhood embed (L15), TYPE_SENTENCES (L14 only) | Bare labels embed poorly; per-pair is richer |
| `registry.summary` / `summary_vector` | Dropped | PoC is embedding-only |
| All relations at sense level | Word-level with `_dis1`/cosine disambiguation | Actual kaikki structure |
| Lateral MetaBary | Eliminated | Forest structure with unique-parent |
| `is_metabary` flag | Eliminated | Redundant — level > 14 implies MetaBary |
| Word vectors = sense centroid | BE-centroid + orphan senses | Carries relational info from L15 pairings |

---

## 15. What v0.4 Gains Over v0.3

- **Algebraically closed model** — everything is
  `normalize(q·a + q·b + (1−q)·c)`, recursive.
- **Forest structure** — trivial hierarchy queries, no cycle handling.
- **Embedding-only pipeline** — no LLM dependency, no async stage.
- **Context-rich v(type) at L15** without LLM calls in the hot path.
- **Bootstrap chain:** sense → L15 BE → word vector → L14 BE → L13 MB →
  … each stage feeds the next with increasingly structured signal.
- **Fermion order** preserves rare, high-value signals (antonyms)
  that would otherwise be lost in synonym flood.

## 16. What v0.4 Costs vs v0.3

- **Sparser coverage.** Unique-parent + soft-orphan means many kaikki
  relations produce no BE. Recall@20 eval under pressure.
- **Second embedding pass at L15** for per-pair v(type) (~1–2M extra
  embeddings).
- **Sequential matching** — can't parallelize fermion order within a
  level (must respect "already paired" checks). Within a tier, parallel
  is fine if conflicts resolved deterministically.

---

## 17. Potential Issues

1. **L15 cosine matching at scale.** ~300K senses → brute-force 90B
   pairs. Need ANN (FAISS/hnswlib). Manageable but must be designed
   upfront.

2. **v(type) embedding calls at L15.** One call per formed BE. ~100–150K
   calls. Batchable at 1K/request → ~150 batches.

3. **Orphan re-entry asymmetry.** Orphan sense paired with existing BE
   creates structurally asymmetric children. Algebraically fine, but the
   new v(type) needs a fresh embed call.

4. **L13 candidate search ambiguity.** "Find two unparented L15 BEs with
   cos > 0.9" — this is cos(BE₁, BE₂) > 0.9, not cos(bridge, child).
   Children must be near each other; bridge initiates search but doesn't
   constrain their mutual similarity.

5. **Sparsity above L12.** With cos > 0.9 threshold and unique-parent,
   each level roughly halves the count. Graph may top out before L1.

---

## 18. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| mongot HNSW OOM | Medium | Baryedge subset first; fp16; Qdrant fallback |
| `_dis1` all-zero → poor disambiguation | Medium | Cosine fallback; track accuracy |
| `antonyms[]` too sparse | **Known** | Synonyms only for primary eval |
| `bary_vec` averages to mush | **This is the test** | Tune q / v(type); falsify if no lift |
| Targets not in dump | Medium | Stub nodes; exclude from eval |
| Unique-parent too sparse | Medium | Track orphan rate; relax if >60% orphan |
| L15 ANN quality | Low | Verify recall vs brute-force on 10K sample |

---

## 19. Open Questions

1. **Synset clustering** — agglomerative vs. Leiden
2. **Sparse L4–6** — collapse to L7 if <5% tag coverage
3. **Polysemy q floor** — start 0.40, tune after MetaBary formation
4. **Disambiguation threshold** — 0.72 default; tune via secondary eval
5. **MetaBary stopping criterion** — "no triads form" (natural) or hard cap?
6. **Stub promotion** — inline in stage 4 or separate pass?
7. **Antonym/synonym _dis1 filtering for v(type)** — revisit if L15 eval noisy

---

## 20. Expansion Path

1. **Multi-language** — translations carry `sense` gloss + non-zero `_dis1`,
   enabling direct sense-level cross-language BaryEdges. MetaBary encodes
   same metaphor patterns across languages.
2. **Atlas migration** — identical schema
3. **Live update loop** — `q` decay + incremental refresh
4. **RAG integration** — relationship structures as retrieval context

---

*BaryGraph Kaikki PoC v0.4 · CM Theory Project · April 2026*
