# CLAUDE.md — BaryGraph Kaikki PoC

Development guide for the BaryGraph proof-of-concept using the English
kaikki.org dictionary corpus. Read this before writing any code.

---

## What We Are Building

A knowledge graph in which every relationship is a first-class stored
object (a **BaryEdge**) with its own embedding vector, sitting between
two nodes exactly as a barycenter sits between two centers of mass.

**The hypothesis under test:** BaryEdge retrieval outperforms flat
nearest-neighbor search on synonym/antonym recall. If including BaryEdge
documents in vector search returns held-out synonym pairs more reliably
than searching nodes alone, the architecture is validated.

**Reference documents:**
- `VISION-KAIKKI-POC.md` — full spec (schema, pipeline, queries, eval)
- `BaryGraph_v1.0.md` — parent architecture spec

---

## Stack

| Component | Version | Purpose |
|---|---|---|
| MongoDB Community | 8.x | Storage, graph traversal, aggregation |
| mongot | bundled | Vector search (HNSW, cosine) |
| nomic-embed-text-v1.5 | GGUF | Embeddings — 768-dim |
| Llama 4 Scout | Q4_K_M (~27 GB) | Selective registry.summary generation |
| llama.cpp / ollama | latest | LLM + embedding runtime |
| Node.js or Python | 18+ / 3.11+ | Ingestion scripts |

Everything runs locally. Zero cloud dependencies. Zero cost.

---

## Repository Layout

```
barygraph-kaikki/
├── CLAUDE.md                  # this file
├── data/
│   └── kaikki-en.jsonl        # ~2–3 GB, download separately
├── scripts/
│   ├── 01_parse.js            # parse kaikki JSONL → intermediate JSON
│   ├── 02_embed.js            # embed sense glosses via nomic-embed-text
│   ├── 03_insert_nodes.js     # insert sense + word nodes into MongoDB
│   ├── 04_seed_edges.js       # seed BaryEdges from explicit relations
│   ├── 05_cluster_synsets.js  # build L10–12 concept cluster nodes
│   ├── 06_infer_edges.js      # cosine-threshold scan at L10–12, L7–9
│   ├── 07_summarize.js        # async LLM summary generation
│   ├── 08_metabary.js         # MetaBary formation at L8+
│   ├── 09_index.js            # build mongot vector indexes
│   └── eval/
│       ├── holdout.js         # hold out 10% of synonym/antonym links
│       ├── recall.js          # measure recall@20 flat vs BaryGraph
│       └── ab_summary.js      # A/B: bary_vec vs summary_vector signal
├── lib/
│   ├── embed.js               # nomic-embed-text wrapper
│   ├── llm.js                 # Llama Scout wrapper (summary generation)
│   ├── bary_vec.js            # bary_vec computation formula
│   └── db.js                  # MongoDB connection + collection helpers
├── pipeline_state/            # resumability checkpoints (gitignored)
└── indexes/
    └── vector_index.json      # mongot index definition
```

---

## MongoDB Collection

Single collection: `barygraph`  
Single database: `barygraph_poc`

All document types live together. `doc_type` field distinguishes them.

### Standard indexes (create before ingestion)

```js
db.barygraph.createIndex({ doc_type: 1, level: 1 })
db.barygraph.createIndex({ cm1_id: 1 })
db.barygraph.createIndex({ cm2_id: 1 })
db.barygraph.createIndex({ child_id: 1 })
db.barygraph.createIndex({ parent_id: 1 })
db.barygraph.createIndex({ node_type: 1 })
db.barygraph.createIndex({ edge_type: 1, level: 1 })
db.barygraph.createIndex({ is_metabary: 1 })
```

### Vector indexes (create after ingestion, stage 9)

```json
// Primary — bary_vec on all doc_types
{
  "fields": [
    { "type": "vector", "path": "vector", "numDimensions": 768, "similarity": "cosine" },
    { "type": "filter", "path": "doc_type" },
    { "type": "filter", "path": "level" },
    { "type": "filter", "path": "edge_type" },
    { "type": "filter", "path": "is_metabary" },
    { "type": "filter", "path": "node_type" }
  ]
}

// Secondary — summary_vector on BaryEdges only
// NOTE: test whether mongot supports two vector fields in one index.
// If not, create a second collection: barygraph_summaries { baryedge_id, summary_vector }
{
  "fields": [
    { "type": "vector", "path": "summary_vector", "numDimensions": 768, "similarity": "cosine" },
    { "type": "filter", "path": "edge_type" },
    { "type": "filter", "path": "level" }
  ]
}
```

---

## Document Schemas

### Node (doc_type: "node")

```js
{
  _id:        ObjectId(),
  doc_type:   'node',
  node_type:  'sense' | 'word' | 'synset' | 'field' | 'register' | 'stub',
  level:      Number,          // 1–15, see hierarchy table below
  label:      String,
  vector:     [Number],        // 768-dim, nomic-embed-text
  surface:    Number,          // max BaryEdges this node sustains
  rotation:   0.0,             // static corpus
  properties: Object,          // type-specific, see below
  created_at: Date,
  updated_at: Date
}
```

**node_type → level mapping:**

| node_type | level | properties keys |
|---|---|---|
| `sense` | 15 | word, pos, sense_idx, gloss, examples, tags, topics, wikidata, ipa |
| `word` | 14 | word, pos, etymology, forms |
| `synset` | 10–12 | hypernym, member_count, cluster_algorithm |
| `field` | 7–9 | name, pos_group, topic_root |
| `register` | 4–6 | name, tag (formal/archaic/slang/technical) — **sparse, may collapse to L7** |
| `stub` | any | word, reason — no vector, used when relation target not in dump |

### BaryEdge (doc_type: "baryedge")

```js
{
  _id:            ObjectId(),
  doc_type:       'baryedge',
  cm1_id:         ObjectId(),    // ref to Node or BaryEdge (level ≥ 8)
  cm2_id:         ObjectId(),    // ref to Node or BaryEdge (level ≥ 8)
  level:          Number,        // must equal cm1.level == cm2.level
  edge_type:      String,        // see edge type table below
  q:              Number,        // connection quality 0–1
  strength:       Number,        // raw cosine similarity or explicit weight
  vector:         [Number],      // bary_vec — computed algebraically, see formula
  summary_vector: [Number],      // 768-dim embedding of registry.summary (separate signal)
  registry: {
    shared_concepts: [String],
    divergence:      [String],
    summary:         String      // LLM-generated or template string
  },
  is_metabary:    Boolean,       // true when cm1 and cm2 are both BaryEdges
  common_ancestor_id: ObjectId(),// required when is_metabary = true
  source:         'ingested' | 'inferred' | 'manual',
  confidence:     Number,
  created_at:     Date,
  updated_at:     Date
}
```

### HierarchyLink (doc_type: "hierarchy_link")

```js
{
  _id:              ObjectId(),
  doc_type:         'hierarchy_link',
  child_id:         ObjectId(),
  parent_id:        ObjectId(),
  child_level:      Number,
  parent_level:     Number,      // must equal child_level − 1 exactly
  abstraction_loss: Number       // 0–1, detail lost going upward
}
```

**Rule:** `parent_level = child_level - 1` always. When a kaikki hypernym
jumps multiple levels (e.g. sparrow → animal), insert synthetic
intermediate synset nodes at each level to maintain the one-step rule.

---

## Edge Types

| kaikki field | edge_type | level | q seed | notes |
|---|---|---|---|---|
| `synonyms[]` | `same_phenomenon` | 15 | 0.90 | Core retrieval signal |
| `antonyms[]` | `contradicts` | 15 | 0.85 | divergence carries the contrast |
| `derived[]` | `extends` | 14 | 0.60 | word-level |
| `related[]` | `extends` | 15 | 0.60 | |
| `coordinate_terms[]` | `same_phenomenon` | 15 | 0.70 | siblings under same hypernym |
| `etymology_templates` (shared root) | `applies_to` | 14 | 0.70 | shared etymon |
| senses of same headword | `is_instance_of` | 15 | max(0.40, cosine) | polysemy edges — see q floor below |
| `hypernyms[]` / `hyponyms[]` | — | cross-level | — | → HierarchyLink only |
| `holonyms[]` / `meronyms[]` | — | cross-level | — | → HierarchyLink only |

**Polysemy q floor:** `q = max(0.40, cosine(s1, s2))` for all
`is_instance_of` edges. Low cosine between distant senses (bank:river vs
bank:finance) is correct, but `q` must be ≥ 0.40 so the MetaBary scanner
at L8+ can still find them. Tune: raise 0.40 if MetaBary recall is too
low; lower if spurious MetaBarys appear.

---

## Key Formulas

### bary_vec

```js
// lib/bary_vec.js
function computeBaryVec(v_cm1, v_cm2, v_type, q) {
  // v(type) = embedding of a fixed natural-language sentence per edge_type
  // NOT the bare label ("same_phenomenon" embeds poorly)
  // NOT centroid of edges (circular dependency: need bary_vec to build edges)
  const combined = v_cm1.map((x, i) => q * x + q * v_cm2[i] + (1 - q) * v_type[i]);
  const norm = Math.sqrt(combined.reduce((s, x) => s + x * x, 0));
  return combined.map(x => x / norm);
}

// Fixed v(type) sentences per edge_type:
const TYPE_SENTENCES = {
  same_phenomenon: 'these two words describe the same concept',
  contradicts:     'these two words have opposite meanings',
  extends:         'one word is derived from or extends the other',
  applies_to:      'these two words share a common origin or root',
  is_instance_of:  'these two senses belong to the same word with different meanings',
};
```

### Connection quality q decay (for live update loop, not PoC)

```
q(t) = q_seed * exp(-λ * Δt_days)
λ per node_type: sense=0.001, word=0.0005, neologism/slang (from tags[])=0.01
```

---

## Ingestion Pipeline

Run stages in order. Each stage is resumable via `pipeline_state/`.

```
Stage 1  parse          ~10 min    kaikki-en.jsonl → structured objects
Stage 2  embed          ~10 min    sense glosses + word labels → 768-dim vectors
Stage 3  insert_nodes   ~30 min    sense (L15) + word (L14) nodes → MongoDB
Stage 4  seed_edges     ~30 min    explicit relations → BaryEdges + HierarchyLinks
Stage 5  cluster        ~20 min    L15 BaryEdges → L10–12 synset nodes
Stage 6  infer_edges    ~1 hour    cosine scan at L10–12 (~50K) and L7–9 (~5K)
Stage 7  summarize      ~3 days    selective LLM summary generation (async)
Stage 8  metabary       ~2 hours   MetaBary formation at L8+ (after stage 7)
Stage 9  index          ~4–8 hours build mongot vector indexes
```

**First queryable state: ~6–10 hours** (after stage 9, before summaries).  
**Full enrichment: ~4 days** (after stage 7 completes asynchronously).

### Stage 7: When to generate vs. use template

```
generate_summary = (
    edge_type == 'contradicts'
 OR edge_type == 'is_instance_of'
 OR (edge_type == 'same_phenomenon' AND cosine(cm1, cm2) < 0.85)
 OR (edge_type == 'extends'         AND cosine(cm1, cm2) < 0.85)
 OR (edge_type == 'applies_to'      AND cosine(cm1, cm2) < 0.80)
 OR (edge_type == 'same_phenomenon' AND coordinate_term == true)
)
```

When false, use template (still embed, zero LLM cost):

| condition | template |
|---|---|
| `same_phenomenon`, cos ≥ 0.85 | `"{w1}" and "{w2}" are synonyms: {g1} / {g2}` |
| `same_phenomenon`, coordinate | `"{w1}" and "{w2}" are coordinate terms under {hypernym}` |
| `extends`, cos ≥ 0.85 | `"{w1}" is derived from "{w2}"` |
| `applies_to`, cos ≥ 0.80 | `"{w1}" and "{w2}" share the etymon {root}` |

LLM prompt (keep both glosses verbatim — model should not invent):

```
Given two dictionary senses connected by the relationship "{edge_type}":

Sense 1: {word1} ({pos1}): {gloss1}
Sense 2: {word2} ({pos2}): {gloss2}

Write one sentence describing what these senses share and how they differ.
Respond with only the sentence, nothing else.
```

Reject LLM output if: longer than 60 tokens, or contains neither `word1`
nor `word2`.

### Resumability

Every stage writes progress to `pipeline_state/{stage_name}.json`:
```json
{ "last_id": "ObjectId(...)", "processed": 1240000, "total": 2500000 }
```
On restart, each stage reads its marker and skips already-processed IDs.
Stage 7 is append-only and idempotent — safe to restart at any point.

---

## Query Patterns

### 7.1 Baseline (flat retrieval — the comparison target)

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector', path: 'vector',
      queryVector: embed('financial institution that lends money'),
      numCandidates: 200, limit: 20,
      filter: { doc_type: 'node', level: 15 }
  }},
  { $project: { label: 1, score: { $meta: 'vectorSearchScore' } } }
])
```

### 7.2 BaryGraph retrieval (nodes + edges together)

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector', path: 'vector',
      queryVector: embed('financial institution that lends money'),
      numCandidates: 200, limit: 20,
      filter: { doc_type: { $in: ['node', 'baryedge'] }, level: 15 }
  }},
  { $lookup: { from: 'barygraph', localField: 'cm1_id', foreignField: '_id', as: 'cm1' }},
  { $lookup: { from: 'barygraph', localField: 'cm2_id', foreignField: '_id', as: 'cm2' }}
])
// Each BaryEdge result implies two parent CMs — effective context is 2–3× top-k
```

### 7.3 Summary-vector retrieval (natural-language signal)

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_summary_vector', path: 'summary_vector',
      queryVector: embed('words that describe the same action differently'),
      numCandidates: 100, limit: 10,
      filter: { doc_type: 'baryedge' }
  }},
  { $lookup: { from: 'barygraph', localField: 'cm1_id', foreignField: '_id', as: 'cm1' }},
  { $lookup: { from: 'barygraph', localField: 'cm2_id', foreignField: '_id', as: 'cm2' }},
  { $project: {
      cm1_label: { $first: '$cm1.label' }, cm2_label: { $first: '$cm2.label' },
      summary: '$registry.summary', score: { $meta: 'vectorSearchScore' }
  }}
])
```

### 7.4 Cross-domain bridge (same_phenomenon only)

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector', path: 'vector',
      queryVector: embed('container for holding things'),
      numCandidates: 100, limit: 10,
      filter: { edge_type: 'same_phenomenon' }
  }},
  { $lookup: { from: 'barygraph', localField: 'cm1_id', foreignField: '_id', as: 'cm1' }},
  { $lookup: { from: 'barygraph', localField: 'cm2_id', foreignField: '_id', as: 'cm2' }},
  { $project: {
      cm1_label: { $first: '$cm1.label' }, cm2_label: { $first: '$cm2.label' },
      summary: '$registry.summary', score: { $meta: 'vectorSearchScore' }
  }}
])
```

### 7.5 MetaBary (polysemy patterns — "crane" ↔ "mouse")

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector', path: 'vector',
      queryVector: embed('word meaning both animal and machine'),
      numCandidates: 50, limit: 5,
      filter: { is_metabary: true }
  }},
  { $lookup: { from: 'barygraph', localField: 'cm1_id', foreignField: '_id', as: 'edge1' }},
  { $lookup: { from: 'barygraph', localField: 'cm2_id', foreignField: '_id', as: 'edge2' }}
])
```

### 7.6 Hierarchy traversal (three-step — correct approach)

```js
// Step 1: find hierarchy_link whose child is the starting node
// Step 2: walk parent_id chains upward
// Step 3: resolve ancestor node IDs
db.barygraph.aggregate([
  { $match: { doc_type: 'hierarchy_link', child_id: senseNodeId }},
  { $graphLookup: {
      from: 'barygraph',
      startWith: '$parent_id',
      connectFromField: 'parent_id',
      connectToField: 'child_id',
      as: 'ancestor_links',
      maxDepth: 8,
      restrictSearchWithMatch: { doc_type: 'hierarchy_link' }
  }},
  { $project: {
      ancestor_ids: { $concatArrays: [['$parent_id'], '$ancestor_links.parent_id'] }
  }},
  { $unwind: '$ancestor_ids' },
  { $lookup: { from: 'barygraph', localField: 'ancestor_ids', foreignField: '_id', as: 'ancestor_node' }},
  { $unwind: '$ancestor_node' },
  { $replaceRoot: { newRoot: '$ancestor_node' }},
  { $sort: { level: 1 }}
])
// NOTE: start from hierarchy_link layer, not the node itself
```

---

## Evaluation

### Primary: held-out relation recall

1. Before ingestion: hold out 10% of `synonyms[]` and `antonyms[]` links → `data/holdout.json`
2. Ingest remaining 90%
3. For each held-out pair (word_A, word_B):
   - Query: `embed(word_A.gloss)`, filter `doc_type: 'baryedge'`, top-20
   - **Success:** word_B appears in CM lineage of any returned BaryEdge
4. Compare recall@20: BaryGraph (nodes + baryedges) vs. flat (nodes only)

**BaryGraph must beat flat retrieval to justify the architecture.**

### Secondary metrics

| metric | how |
|---|---|
| bary_vec precision | for each BaryEdge, its 5-NN should include its own cm1 and cm2 |
| summary_vector lift A | bary_vec recall vs. summary_vector recall — which signal is stronger? |
| summary_vector lift B | merged vs. separate — resolves CLAUDE.md §9.3 open question |
| MetaBary coherence | manual inspection of top-50 MetaBary connections |
| hierarchy correctness | BaryGraph paths vs. kaikki `topics[]` taxonomy |

---

## Known Risks

| risk | likelihood | mitigation |
|---|---|---|
| mongot HNSW OOM at ~9M vectors | Medium | Index baryedge subset first; fp16 quantization; last resort: Qdrant for vectors, Mongo for graph |
| Llama Scout summaries hallucinate | Low–Med | Spot-check 1K before full run; reject if >60 tokens or neither word present |
| kaikki hypernym chains skip levels | High | Stage 5 inserts synthetic intermediate nodes |
| bary_vec averages to semantic mush | **This is the test** | If no lift, use summary_vector as primary BaryEdge vector instead |
| relation targets not in English dump | Medium | Create stub nodes (no vector); exclude from eval |
| 3-day LLM stage interrupted | High | Checkpoint every 10K edges; stage 7 is idempotent |
| mongot no multi-vector index support | Medium | Fall back to `barygraph_summaries` collection |

---

## Open Questions (decide during development)

1. **mongot multi-vector index** — does Community 8.x support two vector
   fields in one index? Test in stage 9 before committing to schema.

2. **Synset clustering algorithm** — agglomerative on sense vectors, or
   Leiden community detection on the synonym BaryEdge graph? Leiden is
   likely better (uses graph structure directly) but needs graph to exist
   first (stage 4).

3. **Sparse L4–6** — if fewer than 5% of senses carry register/period
   tags, collapse levels 4–6 into L7 for the PoC. Reinstate if
   multi-language expansion needs them.

4. **Polysemy q floor** — start at 0.40. Tune after MetaBary formation:
   too few MetaBarys → raise floor; too many spurious → lower it.

5. **v(type) bootstrap** — TYPE_SENTENCES are pre-defined in `lib/bary_vec.js`.
   Do not use centroids of existing edges (circular dependency).

---

## Hardware Requirements

| resource | minimum | recommended |
|---|---|---|
| GPU VRAM | 32 GB | 32 GB (Llama Scout Q4) |
| System RAM | 32 GB | 64 GB (mongot HNSW mmap) |
| Disk | 150 GB | 200 GB |
| CPU | 8 cores | 16 cores |

---

## Expansion Path (post-PoC)

1. **Multi-language** — add French, German, Japanese kaikki dumps;
   translation BaryEdges become cross-language bridges; MetaBary encodes
   same metaphor patterns across languages (the original motivation)
2. **Atlas migration** — schema identical; `mongodump` / `mongorestore`
3. **Live update loop** — q decay + incremental BaryEdge refresh on new dumps
4. **RAG integration** — BaryGraph as retrieval backend; returns relationship
   structures, not just similar words

---

*BaryGraph Kaikki PoC v0.2 · CM Theory Project · April 2026*
