# BaryGraph PoC: English Kaikki Dictionary
## A Local-First Proof of Concept
**Version 0.4 · April 2026**

> **v0.4 changes from v0.3:**
> - Forest structure with unique-parent constraint (soft). Every CM has
>   at most one `parent_edge_id`. Orphans allowed.
> - L15 `v(type)` is now per-pair: embed of both parent words' lexical
>   neighborhoods (antonyms + synonyms). Much richer than generic sentence.
> - L14 word vectors = BE-centroid + orphan senses + word-length signal.
> - Triadic recursion above L14: L13 MetaBary uses L14 BE as bridge,
>   L12+ is pure geometry (cosine > 0.9).
> - Removed: `is_metabary`, `hierarchy_direction`, `common_ancestor_id`,
>   lateral MetaBary, `edge_type` above L14.
> - Added: `parent_edge_id` on all nodes and BaryEdges.
> - Added: word-length feature in L14 word node vector.
> - TYPE_SENTENCES used only at L14.
> - Fermion order for L14 matching (antonyms first, synonyms last).
> - Removed: `strength`, `registry`, `summary_vector`, LLM summary stage.
>   The PoC is embedding-only; `bary_vec` is the sole retrieval signal.
> - **Language: Python 3.11+ throughout.**

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
| Word-length feature improves word-level retrieval | Recall split by short vs. long words |

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

**Born rule interpretation:** `q` is amplitude, `q²` is connection
probability. The MetaBary quality is the bridge probability measured
against the combined probability mass of all three inputs. The L-1
bridge (q₃) dominates — a weak bridge weakens the MetaBary regardless
of L-level strength.

At L13, `BE₁` and `BE₂` are L15 BaryEdges and `BEᴸ⁻¹` is the L14
BaryEdge acting as bridge. Above L13, all three are BaryEdges/MetaBarys
from the level below.

### 2.3 v(type) Construction

**L15 — per-pair lexical neighborhood:**

```
type_text = "W_A (antonyms: a₁, a₂, …; synonyms: s₁, s₂, …); W_B (antonyms: b₁, b₂, …; synonyms: t₁, t₂, …)"
v(type)   = embed(type_text)
```

This anchors every L15 BE in the lexical neighborhood of both parent
words. The antonyms inject polarity contrast; the synonyms inject
cluster membership. The result captures relational context around the
pairing, not just the pairing itself.

For same-headword sense pairs (polysemy), the type text degenerates to
one word's neighborhood. Still valid but less informative — q will
typically be lower for these pairs anyway.

For words with empty synonym/antonym sets (rare words), falls back to
`embed("W_A; W_B")`.

**L14 — TYPE_SENTENCES (fixed per edge type):**

```python
TYPE_SENTENCES = {
    'same_phenomenon': 'these two words describe the same concept',
    'contradicts':     'these two words have opposite meanings',
    'extends':         'one word is derived from or extends the other',
    'applies_to':      'these two words share a common origin or root',
    'is_instance_of':  'this relationship is a specific instance of the broader relationship',
}
v_type = embed(TYPE_SENTENCES[edge_type])
```

L14 is the only level where TYPE_SENTENCES is used.

**L13+ — bridge vector (no embedding call):**

Above L14, the bridge BaryEdge's vector serves directly as the third
component. No embedding call. The bridge already encodes relational
information from the levels below.

### 2.4 Word Vector (L14) with Length Feature

```
v(word_W) = normalize( Σᵢ v(BE_i) + Σⱼ v(sense_j) + λ · φ(W) )
```

where:
- `BE_i` are L15 BaryEdges in which one of W's senses participates
- `sense_j` are orphan senses of W that found no partner at L15
- `φ(W)` is the word-length feature vector (see §2.5)
- `λ` is a tunable blend weight (default: 0.15)

**Why BE-centroid, not sense-centroid:**
Each L15 BE vector already encodes three signals — both senses plus the
word-pair-context type. The word vector absorbs relational information
from every pairing its senses participated in. A word whose senses
paired with diverse partners gets a vector that sits where its senses'
relational neighborhoods overlap — exactly the property needed to bridge
L15 BEs at L13 triad formation.

**Dependency:** L14 word vectors cannot be computed until all L15 BEs
are finalized (including orphan re-entry). Strict stage boundary.

### 2.5 Word-Length Feature φ(W)

Word length is a genuine psycholinguistic property: reading difficulty,
processing time, recall, and Zipf's law familiarity all correlate with
length. Short words tend to be more common, more abstract, and more
semantically central. Long words tend to be domain-specific, rarer, and
more precisely defined.

The feature vector `φ(W)` encodes three length signals:

```python
import numpy as np

def word_length_feature(word: str, forms: list[str]) -> np.ndarray:
    char_len    = len(word)                          # character count
    syllable_ct = count_syllables(word)              # estimated syllable count
    avg_form_len = np.mean([len(f) for f in forms])  # avg inflected form length

    # Normalize to [0, 1] using empirical English ranges
    # char_len: typical 1–20, cap at 25
    # syllable_ct: typical 1–8, cap at 10
    # avg_form_len: similar to char_len
    feat = np.array([
        min(char_len / 25.0, 1.0),
        min(syllable_ct / 10.0, 1.0),
        min(avg_form_len / 25.0, 1.0),
    ], dtype=np.float32)

    # Embed as a 768-dim vector by projecting through a fixed random matrix
    # seeded once at startup — same projection used across all words
    return LENGTH_PROJECTION @ feat   # shape (768,), pre-normalized

def count_syllables(word: str) -> int:
    """Simple vowel-group count. Replace with pyphen for accuracy."""
    import re
    return max(1, len(re.findall(r'[aeiouAEIOU]+', word)))
```

`LENGTH_PROJECTION` is a fixed (768, 3) random normal matrix, normalized
per row, initialized once with `np.random.seed(42)`. It maps the 3-dim
feature into the same 768-dim space as the embedding vectors without
requiring an additional embedding call.

**What this encodes:**
- `"cat"` (3 chars, 1 syllable) → short-word region
- `"dictionary"` (10 chars, 4 syllables) → medium-word region
- `"antidisestablishmentarianism"` (28 chars, 12 syllables) → long-word region

The length feature ensures two words with near-identical sense semantics
but very different lengths (e.g. "use" vs. "utilization") are not
collapsed to the same vector at L14, preserving the register difference
that length encodes.

**λ tuning:** start at 0.15. If eval shows length signal hurts retrieval,
set λ = 0. If it helps, tune upward. Tracked as parameter R6 in §12.

---

## 3. Invariants

1. **Unique parent (soft):** every CM has at most one `parent_edge_id`.
   Orphans allowed. `parent_edge_id` always references a `baryedge`
   document — nodes are CMs, never parent edges.
2. **Triadic recursion only above L14.** No lateral edges, no cross-level
   BEs outside triads.
3. **Forest structure** — single `$graphLookup` climbs to root.
4. **BE and MetaBary interchangeable above L14** — same doc type, same role.
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

**Critical:** All semantic relations live at the word entry level.
The `_dis1` field carries sense distribution weights for disambiguation.

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

### 4.4 `_dis1` Sense Disambiguation

`_dis1` is a space-separated string of integers, one per sense.

```python
def assign_sense(item: dict, sense_vectors: list, threshold: float = 0.72) -> int | None:
    weights = [int(x) for x in item['_dis1'].split()]
    if max(weights) > 0:
        return weights.index(max(weights))   # use _dis1 directly → L15
    target_vec = embed(item['word'])
    sims = [cosine(target_vec, sv) for sv in sense_vectors]
    if max(sims) > threshold:
        return sims.index(max(sims))         # cosine fallback → L15
    return None                              # assign to word level (L14)
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

```python
{
    '_id':            ObjectId(),
    'doc_type':       'node',
    'node_type':      'sense' | 'word' | 'synset' | 'field' | 'register' | 'stub',
    'level':          int,          # 1–15
    'label':          str,
    'vector':         list[float],  # 768-dim
    'surface':        int,
    'rotation':       0.0,
    'parent_edge_id': ObjectId() | None,  # ≤1 parent BE; None = orphan
    'properties':     dict,         # see node_type table below
    'created_at':     datetime,
    'updated_at':     datetime,
}
```

| node_type | level | key properties | vector source |
|---|---|---|---|
| `sense` | 15 | word, pos, sense_id, sense_idx, gloss, examples, tags, topics, wikidata | embed(gloss + examples[:2]) |
| `word` | 14 | word, pos, char_len, syllable_ct, etymology, forms, ipa | BE-centroid + orphan senses + λ·φ(W) |
| `synset` | 10–12 | hypernym, member_count | cluster centroid |
| `field` | 7–9 | name, pos_group | cluster centroid |
| `register` | 4–6 | name, tag — **may collapse to L7** | cluster centroid |
| `stub` | any | word, reason — no vector | none |

**Word node `properties` includes length fields:**

```python
'properties': {
    'word':        'dictionary',
    'pos':         'noun',
    'char_len':    10,              # len(word)
    'syllable_ct': 4,               # count_syllables(word)
    'etymology':   'From Middle English dixionare...',
    'forms':       ['dictionaries'],
    'ipa':         '/ˈdɪk.ʃə.nə.ɹi/',
}
```

### 6.2 BaryEdge

```python
{
    '_id':               ObjectId(),
    'doc_type':          'baryedge',
    'cm1_id':            ObjectId(),   # → node (L14/L15) or baryedge (L≤13)
    'cm2_id':            ObjectId(),   # → node (L14/L15) or baryedge (L≤13)
    'level':             int,          # same as CMs at L14/L15;
                                       # = cm1.level - 2 for MetaBary (L≤13)
    'vector':            list[float],  # bary_vec — algebraic
    'parent_edge_id':    ObjectId() | None,  # ≤1 parent; always → baryedge; None = orphan
    'connection_strength': float,      # q (base) or q_MB (rescaled)

    # L14/L15 ONLY:
    'edge_type':         str | None,   # kaikki relation (L14) or None (L15 cosine-matched)
    'type_vector':       list[float],  # v(type) — per-pair embed (L15) or TYPE_SENTENCES (L14)
    'q':                 float,        # 0–1
    'source':            str,          # 'ingested' | 'inferred' | 'manual' | 'placeholder'
    'confidence':        float,

    # ABSENT above L14:
    # edge_type, type_vector, q, source, confidence

    'created_at':  datetime,
    'updated_at':  datetime,
}
```

**What's dropped above L14:**

| Removed | Why |
|---|---|
| `edge_type` | Type is implicit in bridge BE vector |
| `is_metabary` flag | Everything above L14 is MB by construction |
| `hierarchy_direction` | Always upward; no lateral edges |
| `common_ancestor_id` | Forest structure makes traversal trivial |
| Lateral MetaBary | Eliminated by unique-parent constraint |

---

## 7. Edge Types (L14/L15 only)

### 7.1 Fermion Order (L14 matching priority)

L14 BaryEdge matching follows fermion order — rarer, more informative
relations matched first. Once a word has `parent_edge_id` set, it is
skipped at lower-priority tiers.

| Priority | edge_type | kaikki field | q_seed |
|---|---|---|---|
| 1 | `contradicts` | `antonyms[]` | 0.85 |
| 2 | `applies_to` | `meronyms[]`, `holonyms[]` | 0.55 |
| 3 | `is_instance_of` | `hypernyms[]`, `hyponyms[]` | 0.65 |
| 4 | `extends` | `derived[]`, `related[]` | 0.60 |
| 5 | `same_phenomenon` | `coordinate_terms[]` | 0.70 |
| 6 | `same_phenomenon` | `synonyms[]` | 0.90 |

Tiers 5 and 6 share `edge_type` but differ in q_seed — keyed separately
in `Settings.q_seeds` by kaikki field name.

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

Same-headword sense pairs enter the L15 greedy-match candidate pool
with a q floor:

```
q = max(0.40, cosine(sense_vec_i, sense_vec_j))
```

Greedy matching still selects at most one parent BE per sense.
The floor ensures polysemous senses remain eligible even when their
raw cosine is low.

---

## 8. Construction Pipeline

### Stage 1 — Sense Nodes (L15)

- Parse kaikki JSONL → extract senses
- `v(sense) = embed(gloss + examples[:2])`
- Store as node: `node_type: 'sense'`, `level: 15`, `parent_edge_id: None`

### Stage 2 — Embed

- Batch embed all sense glosses via nomic-embed-text (~10 min GPU)
- One embedding call per sense

### Stage 3 — Insert Nodes

- Insert L15 sense nodes into MongoDB
- Insert L14 word nodes with placeholder vectors (updated in Stage 5)
- Compute and store `char_len`, `syllable_ct` on word nodes

### Stage 4 — L15 BE Formation (cosine-driven)

```
4a. Pairwise cosine among all L15 sense vectors (ANN-accelerated)
4b. Greedy match: highest cosine first, skip already-paired
4c. For each pair: build type_text (parent words + antonyms + synonyms),
    batch-embed → v(type)
4d. Compute bary_vec, set parent_edge_id on both senses
4e. Orphan re-entry: unpaired senses match with existing L15 BEs
```

**Scale:** ~300K senses → use ANN (FAISS or hnswlib). Top-k neighbors
per sense, then greedy match from ranked pairs.

**Embedding cost:** ~1–2M v(type) calls, batchable at 1K → ~1–2K batches.

### Stage 5 — L14 Word Vectors

```python
for word_node in word_nodes:
    be_vecs = [be['vector'] for be in get_baryedges_for_word(word_node)]
    orphan_vecs = [s['vector'] for s in get_orphan_senses(word_node)]
    length_feat = word_length_feature(word_node['properties']['word'],
                                      word_node['properties']['forms'])
    raw = sum(be_vecs + orphan_vecs) + LAMBDA * length_feat
    word_node['vector'] = normalize(raw)
```

No embedding call. Strict stage boundary — runs after L15 BE formation.

### Stage 6 — L14 BE Formation (kaikki-driven, fermion order)

```
6a. Iterate kaikki relations in fermion order (§7.1)
6b. Skip words already paired at this priority tier
6c. v(type) = embed(TYPE_SENTENCES[edge_type])
6d. Compute bary_vec, set parent_edge_id
```

L14 orphan re-entry (`s07_orphan_reentry.py`): each unpaired word
matches the nearest existing L14 BE; new BE inherits that partner's
`edge_type`, `type_vector`, and `q` (no new embedding call).

### Stage 7 — L13 MetaBary (polysemy bridge)

For each L14 BE (bridge), find two unparented L15 BEs with mutual
`cos > 0.9`:

```python
def q_mb(q1: float, q2: float, q3: float) -> float:
    """Born rule MetaBary quality. q3 = bridge (L14), dominates."""
    return q3**2 / (q1**4 + q2**4 + q3**4) ** 0.5

v_l13 = normalize(q * v_be1 + q * v_be2 + (1 - q) * v_bridge)
# where q = q_mb(q_be1, q_be2, q_bridge)
```

Example polysemy triad:

```
word "bank" + word "flow"  → L14 BE (via kaikki: related)
    ↑ bridge
sense "bank/financial" + sense "credit/loan"  → L15 BE₁
sense "bank/riverbank" + sense "flow/stream"  → L15 BE₂
    → L13 MB: financial↔river polysemy bridged by flow concept
```

### Stage 8 — L12→L1 Recursive

```python
while True:
    new_triads = 0
    for bridge in get_unparented_bes(level=current_level - 1):
        candidates = find_unparented_bes_near(bridge, level=current_level, threshold=0.9)
        if len(candidates) >= 2:
            be1, be2 = candidates[:2]
            create_metabary(be1, be2, bridge, level=current_level - 2)
            new_triads += 1
    if new_triads == 0:
        break
    current_level -= 2
```

Pure geometry — no kaikki, no TYPE_SENTENCES.

### Stage 9 — Index

- Build mongot vector index (~4–8 hours)

### Pipeline Timing

| Stage | Script | Duration | Blocking |
|---|---|---|---|
| 1–2. Parse + Embed | `s01_parse_embed.py` | ~20 min | Yes |
| 3. Insert nodes | `s02_insert_nodes.py` | ~30 min | Yes |
| 4. L15 BE formation | `s03_l15_edges.py` | ~45 min | Yes |
| 5. L14 word vectors | `s04_word_vectors.py` | ~5 min | Yes |
| 6. L14 BE formation | `s05_l14_edges.py` | ~30 min | Yes |
| 7. L14 orphan re-entry | `s06_orphan_reentry.py` | ~10 min | Yes |
| 8. L13 MetaBary | `s07_metabary_l13.py` | ~20 min | Yes |
| 9. L12→L1 recursive | `s08_metabary_recursive.py` | ~1–2 hours | Yes |
| 10. Index | `s09_index.py` | ~4–8 hours | Yes |

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

```python
# lib/db.py — run once at setup
db.barygraph.create_index([('doc_type', 1), ('level', 1)])
db.barygraph.create_index([('cm1_id', 1)])
db.barygraph.create_index([('cm2_id', 1)])
db.barygraph.create_index([('node_type', 1)])
db.barygraph.create_index([('edge_type', 1), ('level', 1)])
db.barygraph.create_index([('parent_edge_id', 1)])
db.barygraph.create_index([('properties.word', 1), ('properties.pos', 1)])
db.barygraph.create_index([('properties.sense_id', 1)])
```

### 9.2 Vector Index

```json
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

MongoDB aggregation syntax is language-agnostic — shown as query
structures, executed via `pymongo`'s `aggregate()`.

### Baseline (flat)
```
filter: { doc_type: 'node', level: { $in: [14, 15] } }
```

### BaryGraph retrieval
```
filter: { doc_type: { $in: ['node', 'baryedge'] }, level: { $in: [14, 15] } }
# + $lookup on cm1_id, cm2_id
```

### Hierarchy traversal (forest walk via parent_edge_id)
```
$graphLookup:
  from: 'barygraph'
  startWith: '$parent_edge_id'
  connectFromField: 'parent_edge_id'
  connectToField: '_id'
  as: 'upward_chain'
  maxDepth: 15
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
| Orphan rate per level | Track % of CMs with parent_edge_id = None |
| Word-length signal | Recall@20 split by short (≤4 chars) vs long (≥9 chars) words |

---

## 12. Defaults and Tuning Parameters

| ID | Parameter | Default | Rationale |
|---|---|---|---|
| R1 | L15 matching order | Greedy highest-cosine first | Sense-level kaikki too sparse to drive ordering |
| R2 | L15 q value | `cos(s_A, s_B)` directly | Natural quality signal |
| R3 | L15 orphan v(type) | embed(orphan's word neighborhood only) | One-sided but consistent |
| R4 | Antonym/synonym _dis1 filtering | No filtering initially | Simpler; revisit if eval shows noise |
| R5 | L15 matching threshold | 0.72 | Below this, sense remains orphan |
| R6 | Word-length blend weight λ | 0.15 | Small contribution; set to 0 if eval shows no benefit |

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
| `v(type)` as edge label | Per-pair neighborhood embed (L15), TYPE_SENTENCES (L14 only) | Bare labels embed poorly |
| All relations at sense level | Word-level with `_dis1`/cosine disambiguation | Actual kaikki structure |
| Lateral MetaBary | Eliminated | Forest structure with unique-parent |
| `is_metabary` flag | Eliminated | Redundant — level > 14 implies MetaBary |
| Word vectors = sense centroid | BE-centroid + orphan senses + length feature | Carries relational + psycholinguistic info |
| `registry.summary` / `summary_vector` | Dropped | PoC is embedding-only |

---

## 15. What v0.4 Gains Over v0.3

- **Algebraically closed model** — everything is `normalize(q·a + q·b + (1−q)·c)`, recursive.
- **Forest structure** — trivial hierarchy queries, no cycle handling.
- **Embedding-only pipeline** — no LLM dependency, fully deterministic.
- **Context-rich v(type) at L15** without LLM calls in the hot path.
- **Bootstrap chain:** sense → L15 BE → word vector → L14 BE → L13 MB → … each stage feeds the next with increasingly structured signal.
- **Fermion order** preserves rare, high-value signals (antonyms) that would otherwise be lost in synonym flood.
- **Word-length feature** encodes psycholinguistic register difference at L14.

## 16. What v0.4 Costs vs v0.3

- **Sparser coverage.** Unique-parent + soft-orphan means many kaikki relations produce no BE.
- **Second embedding pass at L15** for per-pair v(type) (~1–2M extra embeddings).
- **Sequential matching** — fermion order within a level cannot be fully parallelized.

---

## 17. Potential Issues

1. **L15 cosine matching at scale.** ~300K senses → 90B pairs brute-force. Use FAISS/hnswlib for ANN.
2. **v(type) embedding calls at L15.** ~100–150K calls. Batchable at 1K → ~150 batches.
3. **Orphan re-entry asymmetry.** Orphan sense paired with existing BE creates structurally asymmetric children. Algebraically fine, but needs a fresh v(type) embed call.
4. **L13 candidate search ambiguity.** Children must be near each other (cos > 0.9 mutual); bridge initiates search but doesn't constrain their mutual similarity.
5. **Sparsity above L12.** Each level roughly halves the node count with cos > 0.9 + unique-parent. Graph may top out before L1.
6. **Word-length feature dimensionality.** φ(W) is projected into 768-dim via fixed random matrix — verify projection preserves relative distances on a sample before ingestion.

---

## 18. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| mongot HNSW OOM | Medium | Baryedge subset first; fp16; Qdrant fallback |
| `_dis1` all-zero → poor disambiguation | Medium | Cosine fallback; track accuracy in eval |
| `antonyms[]` too sparse | **Known** | Synonyms only for primary eval |
| `bary_vec` averages to mush | **This is the test** | Tune q / v(type); falsify if no lift |
| Targets not in dump | Medium | Stub nodes; exclude from eval |
| Unique-parent too sparse | Medium | Track orphan rate; relax if >60% orphan |
| L15 ANN quality | Low | Verify recall vs brute-force on 10K sample |
| Word-length feature hurts retrieval | Low | λ = 0 to disable; A/B in secondary eval |

---

## 19. Open Questions

1. **Synset clustering** — agglomerative vs. Leiden on synonym BaryEdge graph
2. **Sparse L4–6** — collapse to L7 if <5% tag coverage
3. **Polysemy q floor** — start 0.40, tune after MetaBary formation
4. **Disambiguation threshold** — 0.72 default; tune via secondary eval
5. **MetaBary stopping criterion** — "no triads form" (natural) or hard cap?
6. **Stub promotion** — inline in stage 4 or separate pass?
7. **Antonym/synonym _dis1 filtering for v(type)** — revisit if L15 eval noisy
8. **Word-length syllable counter** — simple vowel-group heuristic sufficient, or use pyphen?
9. **λ tuning** — 0.15 default; tune from secondary eval (R6)

---

## 20. Expansion Path

1. **Multi-language** — translations carry `sense` gloss + non-zero `_dis1`, enabling direct sense-level cross-language BaryEdges.
2. **Atlas migration** — identical schema; `mongodump` / `mongorestore`
3. **Live update loop** — `q` decay + incremental refresh
4. **RAG integration** — relationship structures as retrieval context

---

*BaryGraph Kaikki PoC v0.4 · CM Theory Project · April 2026*
