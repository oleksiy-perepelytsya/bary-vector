# BaryGraph PoC: English Kaikki Dictionary
## A Local-First Proof of Concept
**Version 0.3 · April 2026**

> **v0.3 change:** HierarchyLinks removed. All cross-level relationships
> are cross-level MetaBary connections (`hierarchy_direction: "up"`).
> Two document types only: `node` and `baryedge`.

> **Kaikki structure corrections (from dataset inspection):**
> Relations (`synonyms[]`, `hypernyms[]` etc.) live at the **word level**,
> not the sense level. Sense disambiguation uses `_dis1` weights where
> non-zero, cosine similarity as fallback. `antonyms[]` coverage is low
> for nouns — synonyms are the primary eval signal. `translations[]`
> carries a `sense` gloss string usable for cross-language sense matching.

---

## 1. Objective

Validate the core BaryGraph hypothesis — that relationship-aware vector
retrieval outperforms flat nearest-neighbour search — using the English
machine-readable dictionary from kaikki.org as corpus. The PoC runs
entirely on local hardware: MongoDB Community Edition with mongot for
storage and vector search, Llama 4 Scout (Q4, 32 GB) for selective
summary generation, and nomic-embed-text for embeddings.

### 1.1 Why This Corpus

- **Pre-labeled relations.** Synonyms, derived forms, hypernyms, and
  etymology are explicit. BaryEdge `edge_type` comes from the corpus.
- **Built-in ground truth.** Hold out 10% of synonym links; measure
  recall against BaryEdge retrieval with zero human annotation.
- **Rich polysemy.** Words like "bank", "crane", "bark" have senses so
  distant in meaning that no embedding makes them neighbours — yet they
  are deeply related. BaryGraph surfaces this via MetaBary connections.
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
| MetaBary encodes cross-paradigm insight | Polysemy-pattern MetaBarys vs. WordNet sense clusters |
| Cross-level MetaBary replaces HierarchyLinks | Hierarchy traversal via single `$graphLookup` |
| `registry.summary` improves retrieval | A/B: edges with vs. without LLM summary |
| `summary_vector` separate from `bary_vec` | A/B: merged vs. separate signal |
| Sense disambiguation from `_dis1` weights | Precision of sense-level edge assignment |

### 1.3 What This PoC Does Not Cover

- Antonym `contradicts` edges as primary eval signal (low noun coverage).
- Cross-language bridges (deferred to multi-language expansion).
- Production deployment, sharding, or cloud migration.

---

## 2. Kaikki Data Structure

One JSONL line = one `(word, pos)` entry. Example: `dictionary (noun)`.

### 2.1 Top-Level Fields

```
word            "dictionary"
pos             "noun"
lang_code       "en"
forms[]         [{form: "dictionaries", tags: ["plural"]}, ...]
etymology_text  full etymology string
sounds[]        [{ipa: "/ˈdɪk.ʃə.nə.ɹi/", tags: ["Received-Pronunciation"]}, ...]
senses[]        array of sense objects (see §2.2)
translations[]  array of translation objects (see §2.3)

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
weights for disambiguation (see §2.4).

### 2.2 Sense-Level Fields

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

### 2.3 Translation Structure

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

### 2.4 `_dis1` Sense Disambiguation

`_dis1` is a space-separated string of integers, one per sense. Each
integer is a weight indicating how strongly the relation maps to that
sense index.

```
synonyms[0]._dis1    = "0 0 0 0 0 0"     → undisambiguated
translations[0]._dis1 = "24 8 2 28 25 12" → dominant sense = index 3
```

**Disambiguation strategy:**

```python
def assign_sense(item, sense_vectors, threshold=0.72):
    weights = [int(x) for x in item['_dis1'].split()]
    if max(weights) > 0:
        # Non-zero weights: use dominant sense index directly
        return weights.index(max(weights))  # → level 15
    else:
        # All zero: cosine similarity against sense gloss vectors
        target_vec = embed(item['word'])
        sims = [cosine(target_vec, sv) for sv in sense_vectors]
        if max(sims) > threshold:
            return sims.index(max(sims))    # → level 15
        else:
            return None                     # → word level (level 14)
```

---

## 3. Hierarchy Mapping

| Level | Scale | Kaikki Source | PoC Status |
|---|---|---|---|
| 1–3 | Language family / Paradigm | Fixed: "English", "Germanic", "Indo-European" | Static scaffolding |
| 4–6 | Register / Period | `tags[]`: formal, archaic, slang, technical | Sparse — collapse to L7 if <5% coverage |
| 7–9 | Semantic field / POS cluster | `topics[]`, `pos`, sense `categories[]` | Active |
| 10–12 | Concept cluster (synset) | Clustered senses sharing hypernyms | Active |
| 13–14 | Word entry | One node per `(word, pos)` — **primary relation level** | Active |
| 15 | Individual sense / Gloss | Each `senses[]` entry — sense-disambiguated edges land here | Active |

---

## 4. Data Schema

Single collection `barygraph`. Two document types: `node`, `baryedge`.

### 4.1 Sense Node (Level 15)

```js
{
  _id:        ObjectId(),
  doc_type:   'node',
  node_type:  'sense',
  level:      15,
  label:      'dictionary (noun, sense 0): A reference work listing words...',
  vector:     [/* 768-dim: embed(gloss + examples[:2]) */],
  surface:    3,
  rotation:   0.0,
  properties: {
    word:      'dictionary',
    pos:       'noun',
    sense_id:  'en-dictionary-en-noun-en:Q23622',
    sense_idx: 0,
    gloss:     'A reference work listing words or names...',
    examples:  ['If you want to know the meaning of a word...'],
    tags:      [],
    topics:    [],
    categories:['English entries with etymology trees'],
    wikidata:  'Q23622'
  },
  created_at: new Date(), updated_at: new Date()
}
```

### 4.2 Word Node (Level 14)

```js
{
  _id:        ObjectId(),
  doc_type:   'node',
  node_type:  'word',
  level:      14,
  label:      'dictionary (noun)',
  vector:     [/* centroid of child sense vectors, q-weighted */],
  surface:    6,
  rotation:   0.0,
  properties: {
    word:      'dictionary',
    pos:       'noun',
    etymology: 'From Middle English dixionare...',
    forms:     ['dictionaries'],
    ipa:       '/ˈdɪk.ʃə.nə.ɹi/'
  },
  created_at: new Date(), updated_at: new Date()
}
```

### 4.3 BaryEdge (same-level, horizontal)

```js
{
  _id:                ObjectId(),
  doc_type:           'baryedge',
  cm1_id:             ObjectId(),
  cm2_id:             ObjectId(),
  level:              14,          // 14 = undisambiguated; 15 = sense-level
  edge_type:          'same_phenomenon',
  q:                  0.81,
  strength:           0.74,
  vector:             [/* bary_vec */],
  summary_vector:     [/* embed(registry.summary) */],
  registry: {
    shared_concepts:  ['reference work', 'word lookup'],
    divergence:       [],
    summary:          'Both words refer to reference works that list and define words.'
  },
  is_metabary:          false,
  hierarchy_direction:  null,
  common_ancestor_id:   null,
  source:               'ingested',
  confidence:           0.90,
  created_at: new Date(), updated_at: new Date()
}
```

### 4.4 MetaBary — Cross-Level (upward)

```js
{
  _id:                ObjectId(),
  doc_type:           'baryedge',
  cm1_id:             ObjectId(),   // BaryEdge at L15 (specific)
  cm2_id:             ObjectId(),   // BaryEdge at L10 (abstract)
  level:              15,           // store cm1 level
  edge_type:          'is_instance_of',
  q:                  0.68,
  strength:           0.71,
  vector:             [/* bary_vec */],
  summary_vector:     [/* embed(registry.summary) */],
  registry: {
    shared_concepts:  ['reference work membership'],
    divergence:       ['L15: specific word pair', 'L10: synset cluster pair'],
    summary:          'The sense-level synonym relationship is a specific instance of the synset-level concept grouping.'
  },
  is_metabary:          true,
  hierarchy_direction:  'up',
  common_ancestor_id:   null,
  source:               'inferred',
  confidence:           0.68,
  created_at: new Date(), updated_at: new Date()
}
```

### 4.5 MetaBary — Same-Level (lateral)

```js
{
  _id:                ObjectId(),
  doc_type:           'baryedge',
  cm1_id:             ObjectId(),   // BaryEdge at L10
  cm2_id:             ObjectId(),   // BaryEdge at L10
  level:              10,
  edge_type:          'same_phenomenon',
  q:                  0.76,
  strength:           0.81,
  vector:             [/* bary_vec */],
  summary_vector:     [/* embed(registry.summary) */],
  registry: {
    shared_concepts:  ['animal-to-tool polysemy'],
    divergence:       ['crane: avian vs mechanical', 'mouse: rodent vs device'],
    summary:          'Both pairs share the pattern of mapping an animal name to a mechanical tool.'
  },
  is_metabary:          true,
  hierarchy_direction:  'lateral',
  common_ancestor_id:   ObjectId(),
  source:               'inferred',
  confidence:           0.72,
  created_at: new Date(), updated_at: new Date()
}
```

---

## 5. Relation-to-Edge Mapping

### 5.1 Word-Level Relations (primary source)

| Kaikki Field | `edge_type` | Default Level | `q` Seed | Notes |
|---|---|---|---|---|
| `synonyms[]` | `same_phenomenon` | 14 → 15 if disambiguated | 0.90 | Primary eval signal |
| `antonyms[]` | `contradicts` | 14 → 15 if disambiguated | 0.85 | Sparse for nouns — supplementary only |
| `derived[]` | `extends` | 14 (keep word-level) | 0.60 | |
| `related[]` | `extends` | 14 → 15 if disambiguated | 0.60 | |
| `coordinate_terms[]` | `same_phenomenon` | 14 → 15 if disambiguated | 0.70 | |
| `etymology_templates` | `applies_to` | 14 (keep word-level) | 0.70 | |
| `hypernyms[]` | `is_instance_of` | cross-level MetaBary `"up"` | 0.65 | |
| `hyponyms[]` | `is_instance_of` | cross-level MetaBary `"up"` (reversed) | 0.65 | |
| `meronyms[]` | `applies_to` | cross-level MetaBary `"up"` | 0.55 | |
| `holonyms[]` | `applies_to` | cross-level MetaBary `"up"` (reversed) | 0.55 | |

### 5.2 Sense-Level Relations (override where present)

| Field | Behaviour |
|---|---|
| `senses[i].hypernyms[]` | Direct L15 cross-level MetaBary — no disambiguation needed |
| `senses[i].coordinate_terms[]` | Direct L15 BaryEdge |
| `senses[i].hyponyms[]` | Direct L15 cross-level MetaBary |

### 5.3 Polysemy Edges

Between all sense pairs of the same headword:

```python
q = max(0.40, cosine(sense_vec_i, sense_vec_j))
edge_type = 'is_instance_of'
level = 15
```

Floor of 0.40 keeps distant senses connected for MetaBary scanner.
Tune: raise if MetaBary recall too low; lower if spurious MetaBarys appear.

---

## 6. Infrastructure

### 6.1 MongoDB + mongot

```
Standard indexes:
  {doc_type: 1, level: 1}
  {cm1_id: 1}, {cm2_id: 1}
  {node_type: 1}
  {edge_type: 1, level: 1}
  {is_metabary: 1, hierarchy_direction: 1}
  {'properties.word': 1, 'properties.pos': 1}   ← ingestion lookup
  {'properties.sense_id': 1}                     ← ingestion lookup
```

Vector index definitions:

```js
// Primary — bary_vec
{
  fields: [
    { type: 'vector', path: 'vector', numDimensions: 768, similarity: 'cosine' },
    { type: 'filter', path: 'doc_type' },
    { type: 'filter', path: 'level' },
    { type: 'filter', path: 'edge_type' },
    { type: 'filter', path: 'is_metabary' },
    { type: 'filter', path: 'hierarchy_direction' },
    { type: 'filter', path: 'node_type' }
  ]
}

// Secondary — summary_vector (test multi-vector support; fallback: barygraph_summaries)
{
  fields: [
    { type: 'vector', path: 'summary_vector', numDimensions: 768, similarity: 'cosine' },
    { type: 'filter', path: 'edge_type' },
    { type: 'filter', path: 'level' },
    { type: 'filter', path: 'hierarchy_direction' }
  ]
}
```

| Component | Size |
|---|---|
| Vectors: 15M × 768 × 4 bytes | ~46 GB |
| `summary_vector`: ~1M × 768 × 4 bytes | ~3 GB |
| Metadata | ~3–5 GB |
| mongot HNSW (primary) | ~40–60 GB mmap |
| mongot HNSW (secondary) | ~4–6 GB mmap |
| Standard indexes | ~2 GB |
| **Total disk** | **~98–122 GB** |

### 6.2 Embed string construction

```python
# Sense node: gloss + first 2 example texts
embed_text = gloss + ' ' + ' '.join(ex['text'] for ex in examples[:2])

# Word node: centroid of sense vectors (algebraic, not embedded)

# BaryEdge bary_vec: algebraic (zero embedding calls)
# BaryEdge summary_vector: embed(registry.summary)
```

---

## 7. Ingestion Pipeline

```
kaikki-en.jsonl
      │
      ▼
 1. Parse        word entries, senses, relations, IPA, etymology
 2. Embed        sense gloss+examples → 768-dim (~10 min GPU)
 3. Insert nodes sense nodes (L15), word nodes (L14, centroid vectors)
 4. Seed edges
      4a. Sense-level relations → L15 BaryEdges directly
      4b. Word-level relations:
            – disambiguate via _dis1 weights or cosine fallback
            – same-level → L15 or L14 BaryEdge
            – hypernyms/hyponyms/meronyms → cross-level MetaBary ('up')
      4c. Polysemy edges (all sense pairs of same headword)
      4d. Stub promotion pass
 5. Cluster      L14/L15 BaryEdge clusters → L10–12 synset nodes
 6. Infer edges  cosine scan at L10–12 (~50K), L7–9 (~5K)
 7. Summarize    LLM registry.summary (~3 days, async)
 8. MetaBary     lateral MetaBary at L8+ with common ancestor
 9. Index        mongot vector indexes (~4–8 hours)
```

**First queryable: ~6–10 hours. Full enrichment: ~4 days.**

### 7.1 Stage 4b: Word-Level Relation Seeding

```python
SAME_LEVEL_FIELDS = {
    'synonyms':        ('same_phenomenon', 0.90),
    'antonyms':        ('contradicts',     0.85),
    'related':         ('extends',         0.60),
    'coordinate_terms':('same_phenomenon', 0.70),
}
WORD_LEVEL_ONLY = {
    'derived':         ('extends',         0.60),
    'etymology':       ('applies_to',      0.70),
}
CROSS_LEVEL_FIELDS = {
    'hypernyms':  ('is_instance_of', 0.65),
    'hyponyms':   ('is_instance_of', 0.65),
    'meronyms':   ('applies_to',     0.55),
    'holonyms':   ('applies_to',     0.55),
}

for entry in kaikki_entries:
    word_node   = lookup_word_node(entry['word'], entry['pos'])
    sense_nodes = lookup_sense_nodes(entry['word'], entry['pos'])
    sense_vecs  = [n['vector'] for n in sense_nodes]

    for field, (edge_type, q_seed) in SAME_LEVEL_FIELDS.items():
        for item in entry.get(field, []):
            idx = assign_sense(item, sense_vecs)          # §2.4
            cm1 = sense_nodes[idx] if idx is not None else word_node
            lvl = 15 if idx is not None else 14
            cm2 = lookup_or_stub(item['word'], entry['pos'])
            create_baryedge(cm1, cm2, lvl, edge_type, q_seed)

    for field, (edge_type, q_seed) in CROSS_LEVEL_FIELDS.items():
        for item in entry.get(field, []):
            # Find or create the cross-level MetaBary once BaryEdges exist
            # Use placeholder if BaryEdge for this sense not yet created
            schedule_cross_level_metabary(word_node, item['word'], edge_type, q_seed)
```

### 7.2 Stage 7: Summary Generation

```python
def needs_llm(edge):
    return (
        edge['edge_type'] == 'contradicts'
        or edge['edge_type'] == 'is_instance_of'
        or edge['is_metabary'] == True
        or (edge['edge_type'] == 'same_phenomenon' and edge['strength'] < 0.85)
        or (edge['edge_type'] == 'extends'         and edge['strength'] < 0.85)
        or (edge['edge_type'] == 'applies_to'      and edge['strength'] < 0.80)
    )
```

LLM prompt:
```
Given two dictionary senses connected by "{edge_type}":
Sense 1: {word1} ({pos1}): {gloss1}
Sense 2: {word2} ({pos2}): {gloss2}
Write one sentence describing what these senses share and how they differ.
Respond with only the sentence, nothing else.
```
Reject if: >60 tokens, or contains neither word1 nor word2.

### 7.3 Resumability

```json
{ "last_id": "ObjectId(...)", "processed": 1240000, "total": 2500000 }
```

Stage 7 is append-only and idempotent.

---

## 8. Query Patterns

### Baseline (flat)
```js
filter: { doc_type: 'node', level: { $in: [14, 15] } }
```

### BaryGraph retrieval
```js
filter: { doc_type: { $in: ['node', 'baryedge'] }, level: { $in: [14, 15] } }
// + $lookup cm1_id, cm2_id
```

### Summary-vector retrieval
```js
index: 'barygraph_summary_vector', path: 'summary_vector'
filter: { doc_type: 'baryedge' }
```

### Polysemy MetaBary
```js
filter: { is_metabary: true, hierarchy_direction: 'lateral' }
// "crane" (bird↔machine) → MetaBary → "mouse" (rodent↔device)
```

### Hierarchy traversal (single `$graphLookup` — no workaround)
```js
db.barygraph.aggregate([
  { $match: { _id: baryEdgeId }},
  { $graphLookup: {
      from: 'barygraph',
      startWith: '$_id',
      connectFromField: '_id',
      connectToField: 'cm1_id',
      as: 'hierarchy_chain',
      maxDepth: 10,
      restrictSearchWithMatch: { is_metabary: true, hierarchy_direction: 'up' }
  }},
  { $project: {
      chain: { $sortArray: { input: '$hierarchy_chain', sortBy: { level: 1 } } }
  }}
])
```

---

## 9. Evaluation

### 9.1 Primary: Held-Out Synonym Recall@20

1. Hold out 10% of `synonyms[]` links → `data/holdout.json`
   (**synonyms only** — antonyms too sparse for noun-heavy corpus)
2. Ingest 90%
3. Query `embed(word_A gloss)`, filter `doc_type: 'baryedge'`, top-20
4. **Success:** word_B in CM lineage of any returned BaryEdge
5. BaryGraph recall@20 vs. flat recall@20

### 9.2 Secondary Metrics

| Metric | Method |
|---|---|
| `bary_vec` precision | BaryEdge 5-NN should include cm1 and cm2 |
| Disambiguation accuracy | Sample 100 word-level relations; manually verify sense assignment |
| `_dis1` vs cosine | Precision when `_dis1 > 0` vs cosine fallback |
| `summary_vector` lift A | `bary_vec` recall vs. `summary_vector` recall |
| `summary_vector` lift B | Merged vs. separate signals |
| MetaBary coherence | Manual top-50 lateral MetaBary inspection |
| Hierarchy correctness | Cross-level MetaBary paths vs. `topics[]` / `categories[]` |

---

## 10. Resource Budget

| Stage | Duration | Blocking? |
|---|---|---|
| Parse | ~10 min | Yes |
| Embed | ~10 min | Yes |
| Insert nodes | ~30 min | Yes |
| Seed BaryEdges (4a–4d) | ~45 min | Yes |
| Cluster synsets | ~20 min | Yes |
| Cosine scan | ~1 hour | Yes |
| Summary generation | ~3 days | No — async |
| MetaBary formation | ~2 hours | After summaries |
| Build indexes | ~4–8 hours | Yes |
| **First queryable** | **~6–10 hours** | |
| **Full enrichment** | **~4 days** | |

Hardware: 32 GB GPU VRAM, 32–64 GB RAM, 150–200 GB disk, 8+ cores.
Cost: zero (all open-source).

---

## 11. Deviations From Parent Spec (BaryGraph v1.1)

| v1.1 Section | Deviation | Reason |
|---|---|---|
| §4.2 — pairwise cosine scan all levels | Explicit-relation seeding at L14/L15 | O(n²) infeasible |
| §7 — MongoDB Atlas | Community + mongot | Local-first PoC |
| Embedding dimensions | 768-dim | Glosses are short |
| `v(type)` as edge label | Fixed natural-language sentence | Bare labels embed poorly |
| `summary_vector` merged | Stored separately | A/B test first |
| All relations at sense level | Word-level with `_dis1`/cosine disambiguation | Actual kaikki structure |

---

## 12. Expansion Path

1. **Multi-language** — translations carry `sense` gloss + non-zero `_dis1`,
   enabling direct sense-level cross-language BaryEdges. MetaBary encodes
   same metaphor patterns across languages.
2. **Atlas migration** — identical schema
3. **Live update loop** — `q` decay + incremental refresh
4. **RAG integration** — relationship structures as retrieval context

---

## 13. Open Questions

1. **mongot multi-vector index** — test before committing
2. **Synset clustering** — agglomerative vs. Leiden
3. **Sparse L4–6** — collapse to L7 if <5% tag coverage
4. **Polysemy q floor** — start 0.40, tune after MetaBary formation
5. **Disambiguation threshold** — 0.72 default; tune via §9.2 sample eval
6. **Cross-level MetaBary threshold** — same as lateral (0.80) or lower?
7. **Stub promotion timing** — inline in stage 4d or separate pass?

---

## 14. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| mongot HNSW OOM | Medium | Baryedge subset first; fp16; Qdrant fallback |
| LLM summaries hallucinate | Low–Med | Spot-check 1K; reject >60 tokens |
| `_dis1` all-zero → poor disambiguation | Medium | Cosine fallback; track accuracy in §9.2 |
| `antonyms[]` too sparse | **Known** | Synonyms only for primary eval |
| `bary_vec` averages to mush | **This is the test** | Fall back to `summary_vector` as primary |
| Targets not in dump | Medium | Stub nodes; exclude from eval |
| LLM stage interrupted | High | Checkpoint every 10K; idempotent |
| mongot no multi-vector | Medium | `barygraph_summaries` fallback |
| Stub unpromoted | Medium | Post-stage-4d validation pass |

---

*BaryGraph Kaikki PoC v0.3 · CM Theory Project · April 2026*
