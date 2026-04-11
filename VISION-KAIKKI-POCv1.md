# BaryGraph PoC: English Kaikki Dictionary
## A Local-First Proof of Concept
Version 0.1 · April 2026

---

## 1. Objective

Validate the core BaryGraph hypothesis — that relationship-aware vector retrieval outperforms flat nearest-neighbor search — using the English machine-readable dictionary from kaikki.org as corpus. The PoC runs entirely on local hardware: MongoDB Community Edition with mongot for storage and vector search, Llama Scout (Q4, 32 GB) for selective summary generation, and nomic-embed-text for embeddings.

### 1.1 Why This Corpus

The English kaikki.org dump (~2–3 GB JSONL) carries properties that make it an ideal first testbed for BaryGraph:

- **Pre-labeled relations.** Synonyms, antonyms, derived forms, translations, hypernyms, and etymology are explicit in the data. BaryEdge `edge_type` comes from the corpus, not a classifier — removing an entire failure mode from the pipeline.
- **Built-in ground truth.** Holding out 10% of explicit synonym/antonym links and measuring recall against BaryEdge retrieval gives a precision/recall evaluation with zero human annotation.
- **Rich polysemy.** English has extensive sense disambiguation ("bank", "crane", "bark"), producing natural MetaBary candidates at level ≥ 8 where disambiguation patterns across word families can be surfaced.
- **Bounded scale.** ~800K headwords, ~2–2.5M senses — large enough to stress the architecture, small enough to ingest on a single machine in days, not weeks.
- **Structured hierarchy.** kaikki.org entries carry `topics[]`, `tags[]`, `pos`, and categorized senses that map directly onto the BaryGraph level hierarchy without manual assignment.

### 1.2 What This PoC Validates

| BaryGraph Claim | How Kaikki Tests It |
|---|---|
| BaryEdge retrieval > flat retrieval | Hold-out recall on synonym/antonym links |
| `bary_vec` formula is useful | NN search on `bary_vec` retrieves correct CM pair |
| MetaBary encodes cross-paradigm insight | Polysemy-pattern MetaBarys vs. WordNet sense clusters |
| Hierarchy traversal finds shared ancestors | kaikki.org category taxonomy as ground truth |
| `registry.summary` improves retrieval | A/B: edges with vs. without LLM summary |

### 1.3 What This PoC Does Not Cover

- Anomaly detection (deferred to vessel-tracking PoC, CLAUDE.md §6.1).
- Cross-language bridges (deferred to multi-language kaikki expansion).
- Production deployment, sharding, or cloud migration.

---

## 2. Hierarchy Mapping

All nodes and BaryEdges carry a `level` property (1–15) per CLAUDE.md §2.2. The mapping to lexical structure:

| Level | Scale | Kaikki Source |
|---|---|---|
| 1–3 | Language family / Paradigm | Fixed: "English", "Germanic", "Indo-European" |
| 4–6 | Register / Period | `tags[]`: formal, informal, archaic, slang, technical |
| 7–9 | Semantic field / POS cluster | `topics[]`, `pos`, hypernym roots |
| 10–12 | Concept cluster (synset) | Clustered `senses[]` sharing hypernyms |
| 13–14 | Word entry | One node per `(word, pos)` pair |
| 15 | Individual sense / Gloss | Each `senses[]` entry with its gloss and examples |

For this English-only PoC, levels 1–3 are effectively static scaffolding (three fixed nodes). The active hierarchy is levels 4–15.

---

## 3. Data Schema

The schema follows CLAUDE.md §3 exactly. All documents live in a single MongoDB collection (`barygraph`). The `doc_type` field distinguishes entity types.

### 3.1 Sense Node (Level 15)

```js
{
  _id:        ObjectId(),
  doc_type:   'node',
  node_type:  'sense',
  level:      15,
  label:      'bank (noun, sense 2): financial institution',
  vector:     [/* 768-dim */],           // nomic-embed-text of gloss + examples
  surface:    47,                        // count of synonyms + antonyms + related
  rotation:   0.0,                       // static corpus — no live updates
  properties: {
    word:       'bank',
    pos:        'noun',
    sense_idx:  1,
    gloss:      'A financial institution that accepts deposits...',
    examples:   ['She deposited the check at the bank.'],
    tags:       [],
    topics:     ['finance', 'banking'],
    wikidata:   'Q22687',
    ipa:        '/bæŋk/'
  },
  created_at: new Date(),
  updated_at: new Date()
}
```

### 3.2 Word Node (Level 13–14)

```js
{
  _id:        ObjectId(),
  doc_type:   'node',
  node_type:  'word',
  level:      14,
  label:      'bank (noun)',
  vector:     [/* 768-dim */],           // centroid of child sense vectors, q-weighted
  surface:    3,                         // number of senses
  rotation:   0.0,
  properties: {
    word:       'bank',
    pos:        'noun',
    etymology:  'From Middle English banke, from Old French banc...',
    forms:      ['banks', 'banked', 'banking']
  },
  created_at: new Date(),
  updated_at: new Date()
}
```

### 3.3 BaryEdge — Unchanged From CLAUDE.md §3.2

The BaryEdge schema is identical to the parent spec. The only difference is how `edge_type` is assigned: from explicit kaikki relations rather than classifier inference.

### 3.4 HierarchyLink — Unchanged From CLAUDE.md §3.3

`parent_level = child_level - 1` strictly. When a sense's hypernym jumps multiple levels (e.g., *sparrow* → *animal*), intermediate concept-cluster nodes are inserted at each level to maintain the one-step rule.

---

## 4. Relation-to-Edge Mapping

Kaikki relations map to BaryGraph edge types as follows:

| Kaikki Field | `edge_type` | Level | `q` Seed | Notes |
|---|---|---|---|---|
| `synonyms[]` | `same_phenomenon` | 15 | 0.90 | Core retrieval signal |
| `antonyms[]` | `contradicts` | 15 | 0.85 | `registry.divergence` carries the contrast |
| `derived[]` | `extends` | 14 | 0.60 | Word-level, not sense-level |
| `related[]` | `extends` | 15 | 0.60 | |
| `coordinate_terms[]` | `same_phenomenon` | 15 | 0.70 | Sibling concepts under shared hypernym |
| `etymology_templates` (shared root) | `applies_to` | 14 | 0.70 | Shared etymon = structural link |
| Senses of same headword | `is_instance_of` | 15 | cosine(g₁, g₂) | Polysemy edges |
| `hypernyms[]` / `hyponyms[]` | — | cross-level | — | → HierarchyLink, not BaryEdge |
| `holonyms[]` / `meronyms[]` | — | cross-level | — | → HierarchyLink, not BaryEdge |

**Key deviation from CLAUDE.md §4.2:** The parent spec computes cosine similarity across all node pairs at each level. At 2.5M sense nodes, pairwise comparison is O(n²) ≈ 3 × 10¹² — infeasible. Instead:

- **Level 15 BaryEdges are seeded directly from explicit relation arrays.** No pairwise scan.
- **Level 10–12 (concept clusters, ~50K nodes):** pairwise cosine scan per CLAUDE.md §4.2 becomes feasible and runs as specified.
- **Level 7–9:** cosine scan across semantic-field nodes (~5K), fully per spec.

---

## 5. Infrastructure

### 5.1 MongoDB Community Edition + mongot

Single-process architecture. One database, one query language, no service mesh.

```
┌──────────────────────────────────────────────┐
│  MongoDB Community 8.x                       │
│                                              │
│  Collection: barygraph                       │
│  ┌────────────────────────────────────────┐  │
│  │  mongot search engine                  │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │  Vector index (768-dim, cosine)  │  │  │
│  │  │  Filter: doc_type, level,        │  │  │
│  │  │          edge_type, is_metabary  │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Standard indexes:                           │
│  - {doc_type: 1, level: 1}                   │
│  - {cm1_id: 1}, {cm2_id: 1}                 │
│  - {child_id: 1}, {parent_id: 1}            │
└──────────────────────────────────────────────┘
```

**mongot vector search index definition:**

```js
// Primary index — covers bary_vec on all doc_types
{
  fields: [{
    type:          'vector',
    path:          'vector',
    numDimensions: 768,
    similarity:    'cosine'
  }, {
    type: 'filter',
    path: 'doc_type'
  }, {
    type: 'filter',
    path: 'level'
  }, {
    type: 'filter',
    path: 'edge_type'
  }, {
    type: 'filter',
    path: 'is_metabary'
  }, {
    type: 'filter',
    path: 'node_type'
  }]
}

// Secondary index — covers registry.summary embeddings on BaryEdges
// (only if mongot supports multiple vector indexes; see Open Question #5)
{
  fields: [{
    type:          'vector',
    path:          'summary_vector',
    numDimensions: 768,
    similarity:    'cosine'
  }, {
    type: 'filter',
    path: 'edge_type'
  }, {
    type: 'filter',
    path: 'level'
  }]
}

```

**Storage estimate:**

| Component | Size |
|---|---|
| Vectors: 15M × 768 × 4 bytes | ~46 GB on disk |
| Metadata (labels, properties, registry) | ~3–5 GB |
| mongot vector index (HNSW) | ~40–60 GB mmap |
| Standard B-tree indexes | ~2 GB |
| **Total disk** | **~95–115 GB** |

mongot memory-maps the HNSW index. With 32 GB RAM, hot portions stay in memory; cold segments page from disk. Acceptable for a PoC where queries filter by `level` (narrowing the search space to a fraction of the index).

**Why not Qdrant:** Qdrant would offer faster vector search and better memory efficiency for this workload. But it introduces a second service, a second query language, and glue code to join vector results with MongoDB metadata. For a PoC where the goal is validating BaryGraph semantics (not search latency), the simplicity of one system outweighs the performance difference. If the PoC succeeds and query latency becomes a bottleneck, Qdrant or Atlas migration is straightforward — the schema is the same.

### 5.2 Embedding Model: nomic-embed-text

| Property | Value |
|---|---|
| Model | `nomic-embed-text-v1.5` (GGUF) |
| Parameters | 137M |
| Dimensions | 768 |
| Runtime | llama.cpp / llamafile |
| Throughput (CPU) | ~2,000 docs/s |
| Throughput (GPU) | ~8,000 docs/s |
| Memory | ~600 MB |

Glosses are short (10–40 tokens). 768-dim is appropriate — 1536-dim adds storage cost without retrieval benefit for text this short.

**Embedding time for full corpus:**

| What | Count | Time (GPU) |
|---|---|---|
| Sense nodes (L15) | ~2.5M | ~5 min |
| Word nodes (L14) | ~800K | ~2 min |
| Concept cluster centroids (L10–12) | ~50K | <1 min |
| `registry.summary` strings | ~500K–1M | ~2 min |
| **Total embedding time** | | **~10 min** |

BaryEdge vectors are computed algebraically (`bary_vec = normalize(q·v(cm1) + q·v(cm2) + (1−q)·v(edge_type))`), not embedded. The 5–8M BaryEdges cost zero embedding calls.

**`registry.summary` vector — PoC decision (resolves CLAUDE.md §9 open question):**
Store the `registry.summary` embedding as a **separate field** (`summary_vector`), not averaged into `bary_vec`. Rationale: averaging conflates two different signals (structural position from `bary_vec` vs. natural-language description from summary). Keeping them separate allows the evaluation (§8.2) to measure each signal's retrieval contribution independently. If the eval shows averaging outperforms, merge in v0.2. The cost is one extra 768-dim vector per BaryEdge that has a summary (~1M edges × 768 × 4B ≈ 3 GB) — acceptable.

### 5.3 LLM: Llama Scout Q4 (32 GB)

| Property | Value |
|---|---|
| Model | Llama 4 Scout 17B-active (MoE, 109B total) |
| Quantization | Q4_K_M (~25–28 GB model weight) |
| Runtime | llama.cpp / ollama |
| KV cache budget | ~4–7 GB → ~8K effective context |
| Generation speed | ~30–50 tok/s (single), ~100–150 tok/s (batched) |

The 8K effective context is sufficient — summary prompts are ~200 tokens in, ~40 tokens out.

---

## 6. Ingestion Pipeline

### 6.1 Overview

```
kaikki-en.jsonl (2–3 GB)
       │
       ▼
 ┌─────────────┐
 │  1. Parse    │  Extract senses, word forms, relations
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  2. Embed    │  nomic-embed-text: sense glosses, word labels (~10 min)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  3. Insert   │  Sense nodes (L15), word nodes (L14) → MongoDB
 │     Nodes    │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  4. Seed     │  Explicit relations → BaryEdges (algebraic bary_vec)
 │   BaryEdges  │  Hypernyms/hyponyms → HierarchyLinks
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  5. Cluster  │  Cluster L15 BaryEdges → L10–12 concept nodes
 │   Synsets    │  Centroid vectors, q-weighted
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  6. Infer    │  Cosine-threshold scan at L10–12, L7–9
 │   Edges      │  per CLAUDE.md §4.2 (feasible at ~50K nodes)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  7. Summary  │  Llama Scout: selective registry.summary (Strategy A)
 │   Generation │  ~500K–1M calls → 1–2 days batched
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  8. MetaBary │  Cosine pairs at L8+ with common ancestor
 │   Formation  │  ~10K candidates
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  9. Index    │  Build mongot vector search index
 └─────────────┘
```

### 6.2 Stage 7 Detail: Selective Summary Generation (Strategy A)

Not all BaryEdges need an LLM-generated `registry.summary`. The summary's value is highest when the relationship is non-obvious — when `bary_vec` alone is insufficient for retrieval.

**Generate summary (Llama Scout call):**

| Condition | Rationale |
|---|---|
| `edge_type = 'same_phenomenon'` | Cross-domain bridges — the core BaryGraph value proposition |
| `edge_type = 'contradicts'` | Divergence is the payload; needs natural-language description |
| `edge_type = 'extends'` AND cosine(cm1, cm2) < 0.85 | Distant derivations where the connection is non-obvious |
| `edge_type = 'applies_to'` AND cosine(cm1, cm2) < 0.80 | Shared etymology with semantic drift — non-obvious structural link |
| Polysemy edges (same headword, different senses) | Disambiguation patterns feed MetaBary formation |

**Skip summary (use template):**

| Condition | Template |
|---|---|
| `edge_type = 'same_phenomenon'` (synonym) AND cosine > 0.90 | `"{word1}" and "{word2}" are synonyms: {gloss1} / {gloss2}` |
| `edge_type = 'extends'` (derived) AND cosine ≥ 0.85 | `"{word1}" is derived from "{word2}"` |
| `edge_type = 'applies_to'` AND cosine ≥ 0.80 | `"{word1}" and "{word2}" share the etymological root {root}` |
| `edge_type = 'same_phenomenon'` (coordinate) AND cosine > 0.85 | `"{word1}" and "{word2}" are coordinate terms under {hypernym}` |

Template strings are still embedded via nomic-embed-text and stored in `registry.summary`. They cost zero LLM calls but remain searchable — a query landing near the template embedding still surfaces the BaryEdge.

**Estimated LLM calls:** 500K–1M out of 5–8M total BaryEdges.

**Prompt template for `registry.summary` generation:**

```
Given two dictionary senses connected by the relationship "{edge_type}":

Sense 1: {word1} ({pos1}): {gloss1}
Sense 2: {word2} ({pos2}): {gloss2}

Write one sentence describing what these senses share and how they differ.
Respond with only the sentence, nothing else.
```

**Throughput with llama.cpp continuous batching (batch size 4–8):**

| | Value |
|---|---|
| Effective throughput | ~100–150 tok/s |
| Tokens per call (output) | ~40 |
| Calls per second | ~3 |
| 750K calls | ~70 hours ≈ **3 days** |

This is the ingestion bottleneck. Everything else completes in under an hour.

### 6.3 Resumability

The pipeline must be resumable — a 3-day LLM stage will encounter interruptions. Each stage writes a progress marker (last processed `_id`) to a `pipeline_state` collection. On restart, each stage resumes from its marker. BaryEdges without `registry.summary` are queryable immediately (the `bary_vec` is computed algebraically at stage 4); summary generation enriches them asynchronously.

---

## 7. Query Patterns

All queries use MongoDB aggregation pipelines with `$vectorSearch` (mongot).

### 7.1 Standard Retrieval

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector',
      path: 'vector',
      queryVector: embed('financial institution that lends money'),
      numCandidates: 200,
      limit: 20,
      filter: { doc_type: 'node', level: 15 }
  }},
  { $project: { label: 1, score: { $meta: 'vectorSearchScore' } } }
])
```

### 7.2 Relationship-Aware Retrieval (the BaryGraph differentiator)

```js
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector',
      path: 'vector',
      queryVector: embed('financial institution that lends money'),
      numCandidates: 200,
      limit: 20,
      filter: { doc_type: { $in: ['node', 'baryedge'] }, level: 15 }
  }},
  // For each BaryEdge result, pull in its two parent CMs
  { $lookup: {
      from: 'barygraph',
      localField: 'cm1_id',
      foreignField: '_id',
      as: 'cm1'
  }},
  { $lookup: {
      from: 'barygraph',
      localField: 'cm2_id',
      foreignField: '_id',
      as: 'cm2'
  }}
])
```

A single query returns nodes AND the relationships connecting them. Each BaryEdge result implies two parent CMs — effective context is 2–3× the raw top-k count.

### 7.3 Cross-Domain Bridge Query

```js
// Find words from different semantic fields connected by "same_phenomenon"
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector',
      path: 'vector',
      queryVector: embed('container for holding things'),
      numCandidates: 100,
      limit: 10,
      filter: { edge_type: 'same_phenomenon' }
  }},
  { $lookup: { from: 'barygraph', localField: 'cm1_id', foreignField: '_id', as: 'cm1' } },
  { $lookup: { from: 'barygraph', localField: 'cm2_id', foreignField: '_id', as: 'cm2' } },
  { $project: {
      cm1_label: { $first: '$cm1.label' },
      cm2_label: { $first: '$cm2.label' },
      summary:   '$registry.summary',
      score:     { $meta: 'vectorSearchScore' }
  }}
])
```

### 7.4 Polysemy Pattern Query (MetaBary)

```js
// Find MetaBary connections: disambiguation patterns across word families
db.barygraph.aggregate([
  { $vectorSearch: {
      index: 'barygraph_vector',
      path: 'vector',
      queryVector: embed('word meaning both animal and machine'),
      numCandidates: 50,
      limit: 5,
      filter: { is_metabary: true }
  }},
  { $lookup: { from: 'barygraph', localField: 'cm1_id', foreignField: '_id', as: 'edge1' } },
  { $lookup: { from: 'barygraph', localField: 'cm2_id', foreignField: '_id', as: 'edge2' } }
])
```

This retrieves connections like: *"crane" (bird ↔ machine)* → MetaBary → *"mouse" (rodent ↔ device)* — both are animal-to-tool polysemy patterns, encoded as a queryable first-class object.

### 7.5 Hierarchy Traversal

`$graphLookup` operates on a single collection, so climbing the hierarchy is a two-step process: first collect the HierarchyLink chain, then resolve the ancestor nodes.

```js
// From a sense, climb to its concept cluster via HierarchyLinks
db.barygraph.aggregate([
  // Step 1: Find the hierarchy_link whose child is our starting node
  { $match: { doc_type: 'hierarchy_link', child_id: senseNodeId } },

  // Step 2: Walk upward — each link's parent_id becomes the next child_id
  { $graphLookup: {
      from: 'barygraph',
      startWith: '$parent_id',
      connectFromField: 'parent_id',
      connectToField: 'child_id',
      as: 'ancestor_links',
      maxDepth: 8,
      restrictSearchWithMatch: { doc_type: 'hierarchy_link' }
  }},

  // Step 3: Collect all ancestor node IDs and resolve them
  { $project: {
      ancestor_ids: {
        $concatArrays: [['$parent_id'], '$ancestor_links.parent_id']
      }
  }},
  { $unwind: '$ancestor_ids' },
  { $lookup: {
      from: 'barygraph',
      localField: 'ancestor_ids',
      foreignField: '_id',
      as: 'ancestor_node'
  }},
  { $unwind: '$ancestor_node' },
  { $replaceRoot: { newRoot: '$ancestor_node' } },
  { $sort: { level: 1 } }
])
```

---

## 8. Evaluation Plan

### 8.1 Primary Metric: Held-Out Relation Recall

1. Before ingestion, randomly hold out 10% of explicit `synonyms[]` and `antonyms[]` links.
2. Ingest the remaining 90%.
3. For each held-out pair (word_A, word_B):
   - Query: embed(word_A gloss), filter `doc_type: 'baryedge'`, top-20.
   - Success: word_B appears in the CM lineage of any returned BaryEdge.
4. Compare:
   - **BaryGraph retrieval** (query includes BaryEdges): recall @ 20
   - **Flat retrieval** (query on nodes only, no BaryEdges): recall @ 20

This directly tests whether BaryEdge vectors add retrieval value beyond node similarity.

### 8.2 Secondary Metrics

| Metric | Method |
|---|---|
| `bary_vec` precision | For each BaryEdge, check whether its 5-NN in vector space include its own cm1/cm2 |
| Summary retrieval lift | A/B: query recall with vs. without `registry.summary` embedding averaged into `bary_vec` |
| MetaBary coherence | Manual inspection of top-50 MetaBary connections for semantic validity |
| Hierarchy path correctness | Compare BaryGraph hierarchy paths against kaikki.org `topics[]` taxonomy |

### 8.3 Baseline Comparison

The simplest baseline: embed all glosses, run flat cosine NN, measure synonym/antonym recall. This is what a standard vector database gives you. BaryGraph must beat this to justify its complexity.

---

## 9. Resource Budget

### 9.1 Hardware Requirements

| Resource | Requirement |
|---|---|
| GPU VRAM | 32 GB (Llama Scout Q4) |
| System RAM | 32 GB minimum, 64 GB recommended (mongot HNSW mmap) |
| Disk | 150 GB free (DB + indexes + model weights) |
| CPU | 8+ cores (embedding throughput, MongoDB) |

### 9.2 Time Budget

| Stage | Duration | Blocking? |
|---|---|---|
| Parse kaikki JSONL | ~10 min | Yes |
| Embed all nodes (nomic-embed-text, GPU) | ~10 min | Yes |
| Insert nodes + seed BaryEdges | ~30 min | Yes |
| Cluster synsets (L10–12) | ~20 min | Yes |
| Cosine-threshold scan (L10–12, L7–9) | ~1 hour | Yes |
| **Selective summary generation (Llama Scout)** | **~3 days** | **No — async enrichment** |
| MetaBary formation | ~2 hours | After summaries |
| Build mongot vector index | ~1 hour | Yes |
| **Total to first queryable state** | **~3 hours** | |
| **Total to full enrichment** | **~4 days** | |

The system is queryable after ~3 hours. Summary generation runs asynchronously — BaryEdges are searchable via `bary_vec` immediately; `registry.summary` enriches them over the following days.

### 9.3 Cost

Zero. All components are local and open-source:
- MongoDB Community Edition (SSPL)
- mongot (bundled with Community 8.x)
- llama.cpp (MIT)
- nomic-embed-text (Apache 2.0)
- Llama 4 Scout (Llama license)

---

## 10. Deviations From Parent Spec (CLAUDE.md)

| CLAUDE.md Section | Deviation | Reason |
|---|---|---|
| §4.2 — pairwise cosine scan at all levels | Replaced by explicit-relation seeding at L15 | O(n²) at 2.5M nodes is infeasible |
| §7.1 — MongoDB Atlas | MongoDB Community + mongot | Local-first PoC, no cloud dependency |
| §7.2 — 1536-dim embeddings | 768-dim (nomic-embed-text) | Glosses are short; 1536 is storage waste |
| §8 — q decay rate λ per domain | λ per-node based on `tags[]` | Static corpus; neologism/slang tags signal volatility better than domain |
| §2.3 — `v(type)` is embedding of edge type label | `v(type)` = centroid of all edges with that label | Single-word labels ("cites") embed poorly; centroid is more stable |

---

## 11. Expansion Path

If the PoC validates the BaryGraph retrieval hypothesis:

1. **Multi-language kaikki.** Add French, German, Japanese dumps. Translation BaryEdges become cross-language bridges. MetaBary encodes "same metaphor pattern across languages" (bread→money in English/French/Russian).
2. **Atlas migration.** Move from local MongoDB + mongot to Atlas for production scale, shared access, and managed vector search. Schema is identical — migration is a `mongodump`/`mongorestore`.
3. **Live update loop.** kaikki.org dumps are periodic. Implement `q` decay (§8) and incremental BaryEdge refresh for new/changed entries.
4. **RAG integration.** Use BaryGraph as the retrieval backend for an LLM that answers questions about word relationships, etymology, and semantic structure — returning not just similar words but the *relationship structures* connecting them.

---

## 12. Open Questions (PoC-Scoped)

1. **mongot HNSW performance at 15M vectors / 768-dim.** No public benchmarks at this scale for Community Edition. Need to test early and measure query latency under filtered search.
2. **Synset clustering algorithm.** Agglomerative clustering on sense vectors? Leiden community detection on the synonym BaryEdge graph? The choice affects L10–12 node quality.
3. **`v(type)` bootstrap.** The centroid-of-all-edges-with-that-label approach (§10 deviation) requires a first pass to compute. Should this be iterative (recompute after BaryEdge formation)?
4. **Polysemy edge threshold.** Senses of the same headword are always connected, but at what `q`? Low cosine between "bank" (river) and "bank" (finance) is correct — they're distant. But `q` should still be high enough for the MetaBary scanner to pick them up. **Starting position:** set `q = max(0.40, cosine(s₁, s₂))` with a floor of 0.40 for all same-headword pairs. This keeps distant senses connected (the `is_instance_of` edge exists with low but nonzero quality) while letting the MetaBary scanner at L8+ find them. The 0.40 floor is tunable — if MetaBary recall is too low, raise it; if too many spurious MetaBarys appear, lower it.
5. **`summary_vector` index.** The separate `summary_vector` field (§5.2) needs its own mongot vector index or a second vector path in the existing index. mongot may or may not support multiple vector fields in a single index definition — test early. Fallback: a second collection `barygraph_summaries` with `{baryedge_id, summary_vector}` and its own index.

---

BaryGraph Kaikki PoC v0.1 · CM Theory Project · April 2026
