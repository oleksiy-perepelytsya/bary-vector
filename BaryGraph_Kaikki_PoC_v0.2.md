**BaryGraph PoC: Kaikki Dictionary**

*A Local-First Proof of Concept*

Version 0.2 · April 2026

**1. Objective**

Validate the core BaryGraph hypothesis --- that relationship-aware
vector retrieval outperforms flat nearest-neighbor search --- using the
English machine-readable dictionary from kaikki.org as corpus. The PoC
runs entirely on local hardware: MongoDB Community Edition with mongot
for storage and vector search, Llama 4 Scout (Q4, 32 GB) for selective
summary generation, and nomic-embed-text for embeddings.

**1.1 Why This Corpus**

-   **Pre-labeled relations.** Synonyms, antonyms, derived forms,
    translations, hypernyms, and etymology are explicit in the data.
    BaryEdge edge_type comes from the corpus, not a classifier ---
    removing an entire failure mode from the pipeline.

-   **Built-in ground truth.** Holding out 10% of explicit
    synonym/antonym links and measuring recall against BaryEdge
    retrieval gives a precision/recall evaluation with zero human
    annotation.

-   **Rich polysemy.** English has extensive sense disambiguation
    (\"bank\", \"crane\", \"bark\"), producing natural MetaBary
    candidates at level ≥ 8 where disambiguation patterns across word
    families can be surfaced.

-   **Bounded scale.** \~800K headwords, \~2--2.5M senses --- large
    enough to stress the architecture, small enough to ingest on a
    single machine in days, not weeks.

-   **Structured hierarchy.** kaikki.org entries carry topics\[\],
    tags\[\], pos, and categorized senses that map directly onto the
    BaryGraph level hierarchy without manual assignment.

**1.2 What This PoC Validates**

  ------------------------------- ---------------------------------------
  **BaryGraph Claim**             **How Kaikki Tests It**

  BaryEdge retrieval \> flat      Hold-out recall on synonym/antonym
  retrieval                       links

  bary_vec formula is useful      NN search on bary_vec retrieves correct
                                  CM pair

  MetaBary encodes cross-paradigm Polysemy-pattern MetaBarys vs. WordNet
  insight                         sense clusters

  Hierarchy traversal finds       kaikki.org category taxonomy as ground
  shared ancestors                truth

  registry.summary improves       A/B: edges with vs. without LLM summary
  retrieval                       embedding

  summary_vector separate from    A/B: merged vs. separate signal ---
  bary_vec                        resolves CLAUDE.md §9.3
  ------------------------------- ---------------------------------------

*Table 1 --- BaryGraph claims and how this corpus tests each*

**1.3 What This PoC Does Not Cover**

-   Anomaly detection (deferred to vessel-tracking PoC, CLAUDE.md §6.1).

-   Cross-language bridges (deferred to multi-language kaikki
    expansion).

-   Production deployment, sharding, or cloud migration.

**2. Hierarchy Mapping**

All nodes and BaryEdges carry a level property (1--15) per CLAUDE.md
§2.2. The mapping to lexical structure:

  ----------- ------------------- --------------------- --------------------
  **Level**   **Scale**           **Kaikki Source**     **PoC Status**

  1--3        Language family /   Fixed: \"English\",   Static scaffolding
              Paradigm            \"Germanic\",         
                                  \"Indo-European\"     

  4--6        Register / Period   tags\[\]: formal,     Sparse --- may
                                  informal, archaic,    collapse to L7 (see
                                  slang, technical      §12.3)

  7--9        Semantic field /    topics\[\], pos,      Active
              POS cluster         hypernym roots        

  10--12      Concept cluster     Clustered senses\[\]  Active
              (synset)            sharing hypernyms     

  13--14      Word entry          One node per (word,   Active
                                  pos) pair             

  15          Individual sense /  Each senses\[\] entry Active
              Gloss               with its gloss and    
                                  examples              
  ----------- ------------------- --------------------- --------------------

*Table 2 --- Hierarchy levels mapped to kaikki.org lexical structure*

For this English-only PoC, levels 1--3 are effectively static
scaffolding. The active hierarchy is levels 4--15.

**3. Data Schema**

All documents live in a single MongoDB collection (barygraph). The
doc_type field distinguishes entity types.

**3.1 Sense Node (Level 15)**

+-----------------------------------------------------------------------+
| {                                                                     |
|                                                                       |
| \_id: ObjectId(),                                                     |
|                                                                       |
| doc_type: \'node\',                                                   |
|                                                                       |
| node_type: \'sense\',                                                 |
|                                                                       |
| level: 15,                                                            |
|                                                                       |
| label: \'bank (noun, sense 2): financial institution\',               |
|                                                                       |
| vector: \[/\* 768-dim nomic-embed-text of gloss + examples \*/\],     |
|                                                                       |
| surface: 47, // count of synonyms + antonyms + related                |
|                                                                       |
| rotation: 0.0, // static corpus --- no live updates                   |
|                                                                       |
| properties: {                                                         |
|                                                                       |
| word: \'bank\', pos: \'noun\', sense_idx: 1,                          |
|                                                                       |
| gloss: \'A financial institution that accepts deposits\...\',         |
|                                                                       |
| examples: \[\'She deposited the check at the bank.\'\],               |
|                                                                       |
| tags: \[\], topics: \[\'finance\', \'banking\'\],                     |
|                                                                       |
| wikidata: \'Q22687\', ipa: \'/bæŋk/\'                                 |
|                                                                       |
| },                                                                    |
|                                                                       |
| created_at: new Date(), updated_at: new Date()                        |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

*Sense Node document --- level 15*

**3.2 Word Node (Level 13--14)**

+-----------------------------------------------------------------------+
| {                                                                     |
|                                                                       |
| \_id: ObjectId(),                                                     |
|                                                                       |
| doc_type: \'node\',                                                   |
|                                                                       |
| node_type: \'word\',                                                  |
|                                                                       |
| level: 14,                                                            |
|                                                                       |
| label: \'bank (noun)\',                                               |
|                                                                       |
| vector: \[/\* centroid of child sense vectors, q-weighted \*/\],      |
|                                                                       |
| surface: 3, // number of senses                                       |
|                                                                       |
| rotation: 0.0,                                                        |
|                                                                       |
| properties: {                                                         |
|                                                                       |
| word: \'bank\', pos: \'noun\',                                        |
|                                                                       |
| etymology: \'From Middle English banke, from Old French banc\...\',   |
|                                                                       |
| forms: \[\'banks\', \'banked\', \'banking\'\]                         |
|                                                                       |
| },                                                                    |
|                                                                       |
| created_at: new Date(), updated_at: new Date()                        |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

*Word Node document --- level 14*

**3.3 BaryEdge**

The BaryEdge schema follows CLAUDE.md §3.2 exactly, with one addition:
the summary_vector field (see §5.2). Edge_type is assigned from explicit
kaikki relations rather than classifier inference.

+-----------------------------------------------------------------------+
| {                                                                     |
|                                                                       |
| \_id: ObjectId(),                                                     |
|                                                                       |
| doc_type: \'baryedge\',                                               |
|                                                                       |
| cm1_id: ObjectId(\'\...\'),                                           |
|                                                                       |
| cm2_id: ObjectId(\'\...\'),                                           |
|                                                                       |
| level: 15,                                                            |
|                                                                       |
| edge_type: \'same_phenomenon\',                                       |
|                                                                       |
| q: 0.81,                                                              |
|                                                                       |
| strength: 0.74,                                                       |
|                                                                       |
| vector: \[/\* bary_vec: normalize(q·v(cm1) + q·v(cm2) +               |
| (1−q)·v(type)) \*/\],                                                 |
|                                                                       |
| summary_vector:\[/\* separate 768-dim embedding of registry.summary   |
| --- see §5.2 \*/\],                                                   |
|                                                                       |
| registry: {                                                           |
|                                                                       |
| shared_concepts: \[\'financial institution\', \'holds assets\'\],     |
|                                                                       |
| divergence: \[\],                                                     |
|                                                                       |
| summary: \'Both senses describe entities that hold and manage         |
| money.\'                                                              |
|                                                                       |
| },                                                                    |
|                                                                       |
| is_metabary: false,                                                   |
|                                                                       |
| source: \'ingested\',                                                 |
|                                                                       |
| confidence: 0.90,                                                     |
|                                                                       |
| created_at: new Date(), updated_at: new Date()                        |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

*BaryEdge document --- note separate summary_vector field*

**3.4 HierarchyLink**

HierarchyLinks connect a node to its parent level and are not BaryEdges
--- they cross levels. When a sense\'s hypernym jumps multiple levels
(e.g., sparrow → animal), intermediate concept-cluster nodes are
inserted at each level to maintain the one-step rule.

+-----------------------------------------------------------------------+
| {                                                                     |
|                                                                       |
| \_id: ObjectId(),                                                     |
|                                                                       |
| doc_type: \'hierarchy_link\',                                         |
|                                                                       |
| child_id: ObjectId(\'\...\'), // level-15 sense node                  |
|                                                                       |
| parent_id: ObjectId(\'\...\'), // level-14 word node                  |
|                                                                       |
| child_level: 15,                                                      |
|                                                                       |
| parent_level: 14, // must equal child_level − 1                       |
|                                                                       |
| abstraction_loss: 0.18                                                |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

*HierarchyLink document --- one-step-at-a-time strictly*

**4. Relation-to-Edge Mapping**

Kaikki relations map to BaryGraph edge types as follows:

  ---------------------- ----------------- ------------- ----------- -----------------------
  **Kaikki Field**       **edge_type**     **Level**     **q Seed**  **Notes**

  synonyms\[\]           same_phenomenon   15            0.90        Core retrieval signal

  antonyms\[\]           contradicts       15            0.85        registry.divergence
                                                                     carries the contrast

  derived\[\]            extends           14            0.60        Word-level, not
                                                                     sense-level

  related\[\]            extends           15            0.60        

  coordinate_terms\[\]   same_phenomenon   15            0.70        Siblings under same
                                                                     hypernym

  etymology_templates    applies_to        14            0.70        Shared etymon =
  (shared root)                                                      structural link

  Senses of same         is_instance_of    15            max(0.40,   Polysemy edges --- see
  headword                                               cosine)     §12.4

  hypernyms\[\] /        ---               cross-level   ---         → HierarchyLink, not
  hyponyms\[\]                                                       BaryEdge

  holonyms\[\] /         ---               cross-level   ---         → HierarchyLink, not
  meronyms\[\]                                                       BaryEdge
  ---------------------- ----------------- ------------- ----------- -----------------------

*Table 3 --- Kaikki relation fields mapped to BaryGraph edge types*

Key deviation from CLAUDE.md §4.2: pairwise O(n²) scan at 2.5M nodes is
infeasible. Instead:

-   Level 15 BaryEdges are seeded directly from explicit relation
    arrays. No pairwise scan.

-   Level 10--12 (concept clusters, \~50K nodes): pairwise cosine scan
    is feasible and runs per spec.

-   Level 7--9 (semantic-field nodes, \~5K): cosine scan fully per spec.

**5. Infrastructure**

**5.1 MongoDB Community Edition + mongot**

Single-process architecture. One database, one query language, no
service mesh.

+-----------------------------------------------------------------------+
| ┌──────────────────────────────────────────────────┐                  |
|                                                                       |
| │ MongoDB Community 8.x │                                             |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ Collection: barygraph │                                             |
|                                                                       |
| │ ┌────────────────────────────────────────────┐ │                    |
|                                                                       |
| │ │ mongot search engine │ │                                          |
|                                                                       |
| │ │ ┌──────────────────────────────────────┐ │ │                      |
|                                                                       |
| │ │ │ Vector index 1 (bary_vec, 768-dim) │ │ │                        |
|                                                                       |
| │ │ │ Vector index 2 (summary_vector) │ │ │                           |
|                                                                       |
| │ │ │ Filters: doc_type, level, edge_type,│ │ │                       |
|                                                                       |
| │ │ │ is_metabary, node_type │ │ │                                    |
|                                                                       |
| │ │ └──────────────────────────────────────┘ │ │                      |
|                                                                       |
| │ └────────────────────────────────────────────┘ │                    |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ Standard indexes: │                                                 |
|                                                                       |
| │ - {doc_type: 1, level: 1} │                                         |
|                                                                       |
| │ - {cm1_id: 1}, {cm2_id: 1} │                                        |
|                                                                       |
| │ - {child_id: 1}, {parent_id: 1} │                                   |
|                                                                       |
| └──────────────────────────────────────────────────┘                  |
+-----------------------------------------------------------------------+

*Single-process architecture --- one collection, two vector indexes*

mongot vector search index definitions:

+-----------------------------------------------------------------------+
| // Primary index --- bary_vec on all doc_types                        |
|                                                                       |
| {                                                                     |
|                                                                       |
| fields: \[                                                            |
|                                                                       |
| { type: \'vector\', path: \'vector\', numDimensions: 768, similarity: |
| \'cosine\' },                                                         |
|                                                                       |
| { type: \'filter\', path: \'doc_type\' },                             |
|                                                                       |
| { type: \'filter\', path: \'level\' },                                |
|                                                                       |
| { type: \'filter\', path: \'edge_type\' },                            |
|                                                                       |
| { type: \'filter\', path: \'is_metabary\' },                          |
|                                                                       |
| { type: \'filter\', path: \'node_type\' }                             |
|                                                                       |
| \]                                                                    |
|                                                                       |
| }                                                                     |
|                                                                       |
| // Secondary index --- summary_vector on BaryEdges                    |
|                                                                       |
| // (only if mongot supports multiple vector indexes; see §12.5)       |
|                                                                       |
| {                                                                     |
|                                                                       |
| fields: \[                                                            |
|                                                                       |
| { type: \'vector\', path: \'summary_vector\', numDimensions: 768,     |
| similarity: \'cosine\' },                                             |
|                                                                       |
| { type: \'filter\', path: \'edge_type\' },                            |
|                                                                       |
| { type: \'filter\', path: \'level\' }                                 |
|                                                                       |
| \]                                                                    |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

*mongot vector index definitions --- two separate indexes for bary_vec
and summary_vector*

  --------------------------- -------------------------------------------
  **Component**               **Size**

  Vectors: 15M × 768 × 4      \~46 GB on disk
  bytes                       

  summary_vector: \~1M × 768  \~3 GB additional
  × 4 bytes                   

  Metadata (labels,           \~3--5 GB
  properties, registry)       

  mongot HNSW index           \~40--60 GB mmap
  (bary_vec)                  

  mongot HNSW index           \~4--6 GB mmap
  (summary_vector)            

  Standard B-tree indexes     \~2 GB

  Total disk                  \~98--122 GB
  --------------------------- -------------------------------------------

*Table 4 --- Storage estimate including summary_vector field*

mongot memory-maps the HNSW index. With 32 GB RAM, hot portions stay in
memory; cold segments page from disk. Acceptable for a PoC where queries
filter by level, narrowing the search space to a fraction of the index.

**Why not Qdrant:** Qdrant would offer faster vector search and better
memory efficiency. But it introduces a second service, a second query
language, and glue code to join vector results with MongoDB metadata.
For a PoC validating BaryGraph semantics (not search latency),
simplicity outweighs performance. If the PoC succeeds and latency
becomes a bottleneck, Qdrant or Atlas migration is straightforward ---
the schema is identical.

**5.2 Embedding Model: nomic-embed-text**

  --------------------- -------------------------------------------------
  **Property**          **Value**

  Model                 nomic-embed-text-v1.5 (GGUF)

  Parameters            137M

  Dimensions            768

  Runtime               llama.cpp / llamafile

  Throughput (CPU)      \~2,000 docs/s

  Throughput (GPU)      \~8,000 docs/s

  Memory                \~600 MB
  --------------------- -------------------------------------------------

*Table 5 --- nomic-embed-text properties*

Glosses are short (10--40 tokens). 768-dim is appropriate --- 1536-dim
adds storage cost without retrieval benefit for text this short.

  --------------------------- ------------------ ------------------------
  **What**                    **Count**          **Time (GPU)**

  Sense nodes (L15)           \~2.5M             \~5 min

  Word nodes (L14)            \~800K             \~2 min

  Concept cluster centroids   \~50K              \<1 min
  (L10--12)                                      

  registry.summary strings    \~500K--1M         \~2 min

  Total embedding time                           \~10 min
  --------------------------- ------------------ ------------------------

*Table 6 --- Embedding time by document type*

BaryEdge bary_vec values are computed algebraically --- not embedded.
The 5--8M BaryEdges cost zero embedding calls.

**summary_vector --- PoC decision (resolves CLAUDE.md §9.3 open
question):** Store the registry.summary embedding as a separate field
(summary_vector), not averaged into bary_vec. Rationale: averaging
conflates two different signals --- structural position from bary_vec
vs. natural-language description from summary. Keeping them separate
allows §8.2 to measure each signal\'s retrieval contribution
independently. If the eval shows averaging outperforms, merge in v0.3.
Cost: \~1M edges × 768 × 4B ≈ 3 GB extra --- acceptable.

**5.3 LLM: Llama 4 Scout Q4 (32 GB)**

  --------------------- -------------------------------------------------
  **Property**          **Value**

  Model                 Llama 4 Scout 17B-active (MoE, 109B total)

  Quantization          Q4_K_M (\~25--28 GB model weight)

  Runtime               llama.cpp / ollama

  KV cache budget       \~4--7 GB → \~8K effective context

  Generation speed      \~30--50 tok/s (single), \~100--150 tok/s
                        (batched)
  --------------------- -------------------------------------------------

*Table 7 --- Llama 4 Scout Q4 properties*

The 8K effective context is sufficient --- summary prompts are \~200
tokens in, \~40 tokens out.

**6. Ingestion Pipeline**

**6.1 Overview**

+-----------------------------------------------------------------------+
| kaikki-en.jsonl (2--3 GB)                                             |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 1. Parse │ Extract senses, word forms, relations                    |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 2. Embed │ nomic-embed-text: sense glosses, word labels (\~10 min)  |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 3. Insert │ Sense nodes (L15), word nodes (L14) → MongoDB           |
|                                                                       |
| │ Nodes │                                                             |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 4. Seed │ Explicit relations → BaryEdges (algebraic bary_vec)       |
|                                                                       |
| │ BaryEdges │ Hypernyms/hyponyms → HierarchyLinks                     |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 5. Cluster │ Cluster L15 BaryEdges → L10--12 concept nodes          |
|                                                                       |
| │ Synsets │ Centroid vectors, q-weighted                              |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 6. Infer │ Cosine-threshold scan at L10--12, L7--9                  |
|                                                                       |
| │ Edges │ per CLAUDE.md §4.2 (feasible at \~50K nodes)                |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 7. Summary │ Llama Scout: selective registry.summary (§6.2)         |
|                                                                       |
| │ Generation │ \~500K--1M calls → \~3 days batched (async)            |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 8. MetaBary │ Cosine pairs at L8+ with common ancestor              |
|                                                                       |
| │ Formation │ \~10K candidates                                        |
|                                                                       |
| └──────┬──────┘                                                       |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| ┌─────────────┐                                                       |
|                                                                       |
| │ 9. Index │ Build mongot vector search indexes (4--8 hours)          |
|                                                                       |
| └─────────────┘                                                       |
+-----------------------------------------------------------------------+

*Ingestion pipeline --- 9 stages, queryable after \~6--10 hours, fully
enriched after \~4 days*

**6.2 Stage 7: Selective Summary Generation**

Not all BaryEdges need an LLM-generated registry.summary. The summary
adds most value when the relationship is non-obvious --- when bary_vec
alone is insufficient for retrieval.

Decision rule (evaluated per edge):

+-----------------------------------------------------------------------+
| generate_summary =                                                    |
|                                                                       |
| edge_type == \'contradicts\'                                          |
|                                                                       |
| OR edge_type == \'is_instance_of\' // polysemy → feeds MetaBary       |
|                                                                       |
| OR (edge_type == \'same_phenomenon\' AND cos(cm1,cm2) \< 0.85)        |
|                                                                       |
| OR (edge_type == \'extends\' AND cos(cm1,cm2) \< 0.85)                |
|                                                                       |
| OR (edge_type == \'applies_to\' AND cos(cm1,cm2) \< 0.80)             |
|                                                                       |
| OR (edge_type == \'same_phenomenon\' AND coordinate_term == true) //  |
| always                                                                |
+-----------------------------------------------------------------------+

*Summary generation decision rule --- \~500K--1M edges qualify out of
5--8M total*

When generate_summary is false, use a template (still embedded, zero LLM
cost):

  ------------------------ ----------------------------------------------
  **Edge type /            **Template**
  condition**              

  same_phenomenon, cos ≥   \"\[word1\]\" and \"\[word2\]\" are synonyms:
  0.85 (synonym)           {gloss1} / {gloss2}

  same_phenomenon,         \"\[word1\]\" and \"\[word2\]\" are coordinate
  coordinate_term          terms under {hypernym}

  extends, cos ≥ 0.85      \"\[word1\]\" is derived from \"\[word2\]\"
  (derived)                

  applies_to, cos ≥ 0.80   \"\[word1\]\" and \"\[word2\]\" share the
                           etymon {root}
  ------------------------ ----------------------------------------------

*Table 8 --- Template strings for low-cost edges (embedded, zero LLM
calls)*

Template strings are embedded via nomic-embed-text and stored in both
registry.summary and summary_vector. They cost zero LLM calls but remain
fully searchable.

Prompt template for LLM-generated summaries:

+-----------------------------------------------------------------------+
| Given two dictionary senses connected by the relationship             |
| \"{edge_type}\":                                                      |
|                                                                       |
| Sense 1: {word1} ({pos1}): {gloss1}                                   |
|                                                                       |
| Sense 2: {word2} ({pos2}): {gloss2}                                   |
|                                                                       |
| Write one sentence describing what these senses share and how they    |
| differ.                                                               |
|                                                                       |
| Respond with only the sentence, nothing else.                         |
+-----------------------------------------------------------------------+

*LLM prompt template --- both glosses verbatim, no need for model to
invent*

  ---------------------- ------------- ----------------------------------
  **Throughput measure** **Value**     

  Effective throughput   \~100--150    
  (batched)              tok/s         

  Tokens per call        \~40          
  (output)                             

  Calls per second       \~3           

  750K calls total       \~70 hours ≈  This is the ingestion bottleneck
                         3 days        
  ---------------------- ------------- ----------------------------------

*Table 9 --- LLM summary generation throughput*

**6.3 Resumability**

The pipeline must be resumable --- a 3-day LLM stage will encounter
interruptions. Each stage writes a progress marker (last processed \_id)
to a pipeline_state collection. On restart, each stage resumes from its
marker. BaryEdges without registry.summary are queryable immediately via
bary_vec; summary generation enriches them asynchronously.

**7. Query Patterns**

All queries use MongoDB aggregation pipelines with \$vectorSearch
(mongot).

**7.1 Standard Retrieval (Baseline)**

+-----------------------------------------------------------------------+
| db.barygraph.aggregate(\[                                             |
|                                                                       |
| { \$vectorSearch: {                                                   |
|                                                                       |
| index: \'barygraph_vector\',                                          |
|                                                                       |
| path: \'vector\',                                                     |
|                                                                       |
| queryVector: embed(\'financial institution that lends money\'),       |
|                                                                       |
| numCandidates: 200,                                                   |
|                                                                       |
| limit: 20,                                                            |
|                                                                       |
| filter: { doc_type: \'node\', level: 15 }                             |
|                                                                       |
| }},                                                                   |
|                                                                       |
| { \$project: { label: 1, score: { \$meta: \'vectorSearchScore\' } } } |
|                                                                       |
| \])                                                                   |
+-----------------------------------------------------------------------+

**7.2 Relationship-Aware Retrieval (BaryGraph Differentiator)**

+-----------------------------------------------------------------------+
| db.barygraph.aggregate(\[                                             |
|                                                                       |
| { \$vectorSearch: {                                                   |
|                                                                       |
| index: \'barygraph_vector\',                                          |
|                                                                       |
| path: \'vector\',                                                     |
|                                                                       |
| queryVector: embed(\'financial institution that lends money\'),       |
|                                                                       |
| numCandidates: 200,                                                   |
|                                                                       |
| limit: 20,                                                            |
|                                                                       |
| filter: { doc_type: { \$in: \[\'node\', \'baryedge\'\] }, level: 15 } |
|                                                                       |
| }},                                                                   |
|                                                                       |
| // For each BaryEdge result, pull in its two parent CMs               |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm1_id\',            |
| foreignField: \'\_id\', as: \'cm1\' }},                               |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm2_id\',            |
| foreignField: \'\_id\', as: \'cm2\' }}                                |
|                                                                       |
| \])                                                                   |
+-----------------------------------------------------------------------+

Each BaryEdge result implies two parent CMs --- effective context is
2--3× the raw top-k count.

**7.3 Summary-Vector Retrieval**

+-----------------------------------------------------------------------+
| // Query against registry.summary embeddings --- the natural-language |
| signal                                                                |
|                                                                       |
| db.barygraph.aggregate(\[                                             |
|                                                                       |
| { \$vectorSearch: {                                                   |
|                                                                       |
| index: \'barygraph_summary_vector\', // secondary index               |
|                                                                       |
| path: \'summary_vector\',                                             |
|                                                                       |
| queryVector: embed(\'words that describe the same action              |
| differently\'),                                                       |
|                                                                       |
| numCandidates: 100,                                                   |
|                                                                       |
| limit: 10,                                                            |
|                                                                       |
| filter: { doc_type: \'baryedge\' }                                    |
|                                                                       |
| }},                                                                   |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm1_id\',            |
| foreignField: \'\_id\', as: \'cm1\' }},                               |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm2_id\',            |
| foreignField: \'\_id\', as: \'cm2\' }},                               |
|                                                                       |
| { \$project: {                                                        |
|                                                                       |
| cm1_label: { \$first: \'\$cm1.label\' },                              |
|                                                                       |
| cm2_label: { \$first: \'\$cm2.label\' },                              |
|                                                                       |
| summary: \'\$registry.summary\',                                      |
|                                                                       |
| score: { \$meta: \'vectorSearchScore\' }                              |
|                                                                       |
| }}                                                                    |
|                                                                       |
| \])                                                                   |
+-----------------------------------------------------------------------+

Evaluation §8.2 runs both 7.2 and 7.3 queries against the same test set
to measure relative contribution of each signal.

**7.4 Cross-Domain Bridge Query**

+-----------------------------------------------------------------------+
| // Find words from different semantic fields connected by             |
| same_phenomenon                                                       |
|                                                                       |
| db.barygraph.aggregate(\[                                             |
|                                                                       |
| { \$vectorSearch: {                                                   |
|                                                                       |
| index: \'barygraph_vector\',                                          |
|                                                                       |
| path: \'vector\',                                                     |
|                                                                       |
| queryVector: embed(\'container for holding things\'),                 |
|                                                                       |
| numCandidates: 100,                                                   |
|                                                                       |
| limit: 10,                                                            |
|                                                                       |
| filter: { edge_type: \'same_phenomenon\' }                            |
|                                                                       |
| }},                                                                   |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm1_id\',            |
| foreignField: \'\_id\', as: \'cm1\' }},                               |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm2_id\',            |
| foreignField: \'\_id\', as: \'cm2\' }},                               |
|                                                                       |
| { \$project: {                                                        |
|                                                                       |
| cm1_label: { \$first: \'\$cm1.label\' },                              |
|                                                                       |
| cm2_label: { \$first: \'\$cm2.label\' },                              |
|                                                                       |
| summary: \'\$registry.summary\',                                      |
|                                                                       |
| score: { \$meta: \'vectorSearchScore\' }                              |
|                                                                       |
| }}                                                                    |
|                                                                       |
| \])                                                                   |
+-----------------------------------------------------------------------+

**7.5 Polysemy Pattern Query (MetaBary)**

+-----------------------------------------------------------------------+
| // Find MetaBary connections: disambiguation patterns across word     |
| families                                                              |
|                                                                       |
| db.barygraph.aggregate(\[                                             |
|                                                                       |
| { \$vectorSearch: {                                                   |
|                                                                       |
| index: \'barygraph_vector\',                                          |
|                                                                       |
| path: \'vector\',                                                     |
|                                                                       |
| queryVector: embed(\'word meaning both animal and machine\'),         |
|                                                                       |
| numCandidates: 50,                                                    |
|                                                                       |
| limit: 5,                                                             |
|                                                                       |
| filter: { is_metabary: true }                                         |
|                                                                       |
| }},                                                                   |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm1_id\',            |
| foreignField: \'\_id\', as: \'edge1\' }},                             |
|                                                                       |
| { \$lookup: { from: \'barygraph\', localField: \'cm2_id\',            |
| foreignField: \'\_id\', as: \'edge2\' }}                              |
|                                                                       |
| \])                                                                   |
+-----------------------------------------------------------------------+

This retrieves connections like: \"crane\" (bird ↔ machine) → MetaBary →
\"mouse\" (rodent ↔ device) --- both are animal-to-tool polysemy
patterns, encoded as a queryable first-class object.

**7.6 Hierarchy Traversal**

\$graphLookup operates on a single collection, so climbing the hierarchy
is a three-step process: find the HierarchyLink chain starting from the
node, walk upward, then resolve ancestor nodes.

+-----------------------------------------------------------------------+
| // From a sense node, climb to its concept cluster via HierarchyLinks |
|                                                                       |
| db.barygraph.aggregate(\[                                             |
|                                                                       |
| // Step 1: Find the hierarchy_link whose child is our starting node   |
|                                                                       |
| { \$match: { doc_type: \'hierarchy_link\', child_id: senseNodeId } }, |
|                                                                       |
| // Step 2: Walk upward --- each link\'s parent_id becomes the next    |
| child_id                                                              |
|                                                                       |
| { \$graphLookup: {                                                    |
|                                                                       |
| from: \'barygraph\',                                                  |
|                                                                       |
| startWith: \'\$parent_id\',                                           |
|                                                                       |
| connectFromField: \'parent_id\',                                      |
|                                                                       |
| connectToField: \'child_id\',                                         |
|                                                                       |
| as: \'ancestor_links\',                                               |
|                                                                       |
| maxDepth: 8,                                                          |
|                                                                       |
| restrictSearchWithMatch: { doc_type: \'hierarchy_link\' }             |
|                                                                       |
| }},                                                                   |
|                                                                       |
| // Step 3: Collect all ancestor node IDs and resolve them             |
|                                                                       |
| { \$project: {                                                        |
|                                                                       |
| ancestor_ids: {                                                       |
|                                                                       |
| \$concatArrays: \[\[\'\$parent_id\'\],                                |
| \'\$ancestor_links.parent_id\'\]                                      |
|                                                                       |
| }                                                                     |
|                                                                       |
| }},                                                                   |
|                                                                       |
| { \$unwind: \'\$ancestor_ids\' },                                     |
|                                                                       |
| { \$lookup: {                                                         |
|                                                                       |
| from: \'barygraph\',                                                  |
|                                                                       |
| localField: \'ancestor_ids\',                                         |
|                                                                       |
| foreignField: \'\_id\',                                               |
|                                                                       |
| as: \'ancestor_node\'                                                 |
|                                                                       |
| }},                                                                   |
|                                                                       |
| { \$unwind: \'\$ancestor_node\' },                                    |
|                                                                       |
| { \$replaceRoot: { newRoot: \'\$ancestor_node\' } },                  |
|                                                                       |
| { \$sort: { level: 1 } }                                              |
|                                                                       |
| \])                                                                   |
+-----------------------------------------------------------------------+

Start from the HierarchyLink layer (not the node), walk parent_id
chains, then resolve. Returns ancestor nodes sorted from highest (most
abstract) to lowest.

**8. Evaluation Plan**

**8.1 Primary Metric: Held-Out Relation Recall**

-   Before ingestion, randomly hold out 10% of explicit synonyms\[\] and
    antonyms\[\] links.

-   Ingest the remaining 90%.

-   For each held-out pair (word_A, word_B): query embed(word_A gloss),
    filter doc_type: \"baryedge\", top-20. Success: word_B appears in
    the CM lineage of any returned BaryEdge.

-   Compare: BaryGraph retrieval (query includes BaryEdges) vs. flat
    retrieval (nodes only). Measure recall @ 20 for both.

This directly tests whether BaryEdge vectors add retrieval value beyond
node similarity.

**8.2 Secondary Metrics**

  --------------------------- -------------------------------------------
  **Metric**                  **Method**

  bary_vec precision          For each BaryEdge, check whether its 5-NN
                              include its own cm1/cm2

  summary_vector lift (A)     Query recall: bary_vec only vs.
                              summary_vector only --- which signal is
                              stronger?

  summary_vector lift (B)     Query recall: bary_vec + summary_vector
                              merged vs. separate --- resolves CLAUDE.md
                              §9.3

  MetaBary coherence          Manual inspection of top-50 MetaBary
                              connections for semantic validity

  Hierarchy path correctness  Compare BaryGraph hierarchy paths against
                              kaikki.org topics\[\] taxonomy
  --------------------------- -------------------------------------------

*Table 10 --- Secondary evaluation metrics*

**8.3 Baseline Comparison**

The simplest baseline: embed all glosses, run flat cosine NN, measure
synonym/antonym recall. This is what a standard vector database gives
you. BaryGraph must beat this to justify its complexity.

**9. Resource Budget**

**9.1 Hardware Requirements**

  --------------------- -------------------------------------------------
  **Resource**          **Requirement**

  GPU VRAM              32 GB (Llama Scout Q4)

  System RAM            32 GB minimum, 64 GB recommended (mongot HNSW
                        mmap)

  Disk                  150 GB free (DB + indexes + model weights +
                        summary_vector overhead)

  CPU                   8+ cores (embedding throughput, MongoDB)
  --------------------- -------------------------------------------------

*Table 11 --- Hardware requirements*

**9.2 Time Budget**

  ------------------------- --------------- ------------------------------
  **Stage**                 **Duration**    **Blocking?**

  Parse kaikki JSONL        \~10 min        Yes

  Embed all nodes           \~10 min        Yes
  (nomic-embed-text, GPU)                   

  Insert nodes + seed       \~30 min        Yes
  BaryEdges                                 

  Cluster synsets (L10--12) \~20 min        Yes

  Cosine-threshold scan     \~1 hour        Yes
  (L10--12, L7--9)                          

  Selective summary         \~3 days        No --- async enrichment
  generation (Llama Scout)                  

  MetaBary formation        \~2 hours       After summaries

  Build mongot vector       \~4--8 hours    Yes
  indexes (×2)                              

  Total to first queryable  \~6--10 hours   
  state                                     

  Total to full enrichment  \~4 days        
  ------------------------- --------------- ------------------------------

*Table 12 --- Time budget by stage*

The system is queryable after \~6--10 hours. Summary generation runs
asynchronously --- BaryEdges are searchable via bary_vec immediately;
registry.summary enriches them over the following days.

**9.3 Cost**

Zero. All components are local and open-source: MongoDB Community
Edition (SSPL), mongot (bundled with Community 8.x), llama.cpp (MIT),
nomic-embed-text (Apache 2.0), Llama 4 Scout (Llama license).

**10. Deviations From Parent Spec (CLAUDE.md)**

  --------------------- -------------------- ------------------------------
  **CLAUDE.md Section** **Deviation**        **Reason**

  §4.2 --- pairwise     Replaced by          O(n²) at 2.5M nodes is
  cosine scan at all    explicit-relation    infeasible
  levels                seeding at L15       

  §7.1 --- MongoDB      MongoDB Community +  Local-first PoC, no cloud
  Atlas                 mongot               dependency

  §7.2 --- 1536-dim     768-dim              Glosses are short; 1536 adds
  embeddings            (nomic-embed-text)   storage cost without retrieval
                                             benefit

  §8 --- q decay rate λ λ per-node based on  Static corpus; neologism/slang
  per domain            tags\[\]             tags signal volatility better
                                             than domain

  §2.3 --- v(type) as   Fixed                Bare labels embed poorly;
  edge type label       natural-language     fixed sentences avoid circular
  embedding             sentence per type:   dependency (can\'t compute
                        same_phenomenon →    edge centroids before
                        \"these two words    bary_vec, which needs v(type))
                        describe the same    
                        concept\"            

  §9.3 ---              Stored as separate   Two signals should be measured
  summary_vector        field; A/B test      independently before deciding
  averaged into         determines whether   to conflate them
  bary_vec              to merge             
  --------------------- -------------------- ------------------------------

*Table 13 --- Deviations from parent BaryGraph spec with rationale*

**11. Expansion Path**

-   Multi-language kaikki. Add French, German, Japanese dumps.
    Translation BaryEdges become cross-language bridges. MetaBary
    encodes \"same metaphor pattern across languages\" (bread→money in
    English/French/Russian). This is the original BaryGraph motivation
    --- the PoC validates the architecture that makes it possible.

-   Atlas migration. Move from local MongoDB + mongot to Atlas for
    production scale and managed vector search. Schema is identical ---
    migration is a mongodump/mongorestore.

-   Live update loop. kaikki.org dumps are periodic. Implement q decay
    (§8) and incremental BaryEdge refresh for new/changed entries.

-   RAG integration. Use BaryGraph as the retrieval backend for an LLM
    answering questions about word relationships, etymology, and
    semantic structure --- returning not just similar words but the
    relationship structures connecting them.

**12. Open Questions (PoC-Scoped)**

-   **mongot HNSW performance at 15M vectors / 768-dim.** No public
    benchmarks at this scale for Community Edition. Build index on
    doc_type: \"baryedge\" subset first to validate latency under
    filtered search before committing to full ingestion.

-   **Synset clustering algorithm.** Agglomerative clustering on sense
    vectors? Leiden community detection on the synonym BaryEdge graph?
    The choice affects L10--12 node quality significantly.

-   **Sparse L4--6.** Register/period tags (archaic, slang, technical)
    are absent on most kaikki senses. Levels 4--6 may be too thin to be
    useful --- consider collapsing into L7 for the PoC and reinstating
    only if multi-language expansion needs them.

-   **Polysemy edge q floor.** Set q = max(0.40, cosine(s₁, s₂)) with a
    floor of 0.40 for all same-headword pairs. This keeps distant senses
    connected (is_instance_of edge exists with low but nonzero quality)
    while letting the MetaBary scanner find them at L8+. The 0.40 floor
    is tunable --- raise if MetaBary recall is too low, lower if
    spurious MetaBarys appear.

-   **summary_vector index.** The separate summary_vector field needs
    its own mongot vector index or a second vector path in the existing
    index. mongot may not support multiple vector fields in a single
    index definition --- test early. Fallback: a second collection
    barygraph_summaries with {baryedge_id, summary_vector} and its own
    index.

**13. Risks & Mitigations**

  ------------------------ ---------------- ------------------------------------
  **Risk**                 **Likelihood**   **Mitigation**

  mongot HNSW build        Medium           Build index on baryedge subset
  exhausts RAM at \~9M                      first; fall back to fp16 vector
  vectors                                   quantization (halves footprint);
                                            last resort: offload vector field to
                                            Qdrant keyed by \_id, keep graph in
                                            Mongo

  Llama Scout Q4 summaries Low--Med         Spot-check first 1K before full run;
  are low-quality or                        prompt includes both glosses
  hallucinate                               verbatim so model has no need to
                                            invent; reject outputs \> 60 tokens
                                            or containing neither CM word

  kaikki hypernym chains   High             Stage 5 inserts synthetic
  skip levels → broken                      intermediate synset nodes where
  one-step traversal                        parent_level ≠ child_level − 1

  bary_vec averages to     This is the      If §8.1 shows no lift, re-run eval
  semantic mush (near-zero hypothesis under with summary_vector as the primary
  retrieval lift)          test             BaryEdge vector --- this decides
                                            whether the natural-language
                                            description signal outperforms the
                                            structural position signal

  Relation targets in      Medium           Create lightweight stub nodes
  kaikki point to words                     (node_type: \"stub\", no vector) so
  not in the English dump                   edges resolve; exclude stubs from
                                            eval metrics

  3-day LLM stage          High             pipeline_state checkpoint every 10K
  interrupted                               edges (§6.3); stage 7 is append-only
                                            and idempotent

  secondary summary_vector Medium           Fall back to separate
  index unsupported by                      barygraph_summaries collection with
  mongot                                    its own vector index; join by
                                            baryedge_id in query pipeline
  ------------------------ ---------------- ------------------------------------

*Table 14 --- Risks with likelihood estimates and concrete mitigations*

BaryGraph Kaikki PoC v0.2 · CM Theory Project · April 2026
