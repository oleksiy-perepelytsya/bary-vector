"""BaryGraph MCP server — exposes the barygraph collection as Claude tools.

Provides six tools:
  find_word        — look up word nodes (all POS variants)
  word_senses      — list all L15 sense glosses for a word
  word_edges       — L14 BaryEdges where the word is a CM
  edge_info        — details + CM leaf-words for any BE/MB by id
  semantic_search  — $vectorSearch (requires mongot index from s10_index)
  graph_stats      — document counts by level / type

Run via stdio transport (Claude Code + Claude Desktop both use stdio):
    python -m scripts.mcp_server
"""

from __future__ import annotations

import json
from typing import Any

from bson import ObjectId
from mcp.server.fastmcp import FastMCP
from pymongo.errors import OperationFailure

from lib.config import Settings
from lib.db import any_cm_has_word, cm_leaf_words, get_collection, vector_search
from lib.embed import get_embedder
from lib.log import setup_logging

_settings = Settings.load()
setup_logging(_settings.log_level)
_coll = get_collection(_settings)

mcp = FastMCP(
    "barygraph",
    instructions=(
        "BaryGraph knowledge graph built from the kaikki.org English dictionary. "
        "L14 nodes are words (node_type='word'), L15 nodes are individual senses "
        "(node_type='sense'). Relationships are BaryEdge docs (doc_type='baryedge'). "
        "L13+ are MetaBary triads connecting BEs. Use semantic_search to find "
        "related concepts, then edge_info or word_edges to explore connections. "
        "graph_stats shows how much data has been ingested."
    ),
)


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


@mcp.tool()
def find_word(word: str) -> str:
    """Find a word in the graph. Returns all POS variants with edge counts and etymology."""
    docs = list(_coll.find(
        {"doc_type": "node", "node_type": "word", "properties.word": word},
        {"properties": 1, "parent_edge_id": 1},
    ))
    if not docs:
        return f"Word '{word}' not found. Try graph_stats to check if the graph is populated."

    results = []
    for d in docs:
        p = d["properties"]
        edge_count = _coll.count_documents({
            "doc_type": "baryedge",
            "$or": [{"cm1_id": d["_id"]}, {"cm2_id": d["_id"]}],
        })
        results.append({
            "id": str(d["_id"]),
            "word": p["word"],
            "pos": p["pos"],
            "ipa": p.get("ipa"),
            "etymology": (p.get("etymology") or "")[:150] or None,
            "forms": (p.get("forms") or [])[:6],
            "sense_count": len(p.get("sense_ids") or []),
            "baryedge_count": edge_count,
            "has_parent_edge": d.get("parent_edge_id") is not None,
        })
    return _fmt(results)


@mcp.tool()
def word_senses(word: str) -> str:
    """List all L15 sense nodes for a word — glosses, tags, and whether each sense is paired."""
    docs = list(_coll.find(
        {"doc_type": "node", "node_type": "sense", "properties.word": word},
        {"properties.sense_idx": 1, "properties.pos": 1, "properties.gloss": 1,
         "properties.tags": 1, "properties.topics": 1, "parent_edge_id": 1},
    ).sort("properties.sense_idx", 1))
    if not docs:
        return f"No senses found for '{word}' (word may not be in the graph)."
    return _fmt([
        {
            "id": str(d["_id"]),
            "sense_idx": d["properties"].get("sense_idx"),
            "pos": d["properties"].get("pos"),
            "gloss": d["properties"].get("gloss", ""),
            "tags": d["properties"].get("tags", []),
            "topics": d["properties"].get("topics", []),
            "paired": d.get("parent_edge_id") is not None,
        }
        for d in docs
    ])


@mcp.tool()
def word_edges(word: str, pos: str = "") -> str:
    """Get L14 BaryEdges where this word is a CM (direct kaikki relations).

    Optionally filter by POS (noun, verb, adj, …).
    Returns edge_type, partner word, q, and accumulated_weight.
    """
    query: dict[str, Any] = {
        "doc_type": "node", "node_type": "word", "properties.word": word,
    }
    if pos:
        query["properties.pos"] = pos

    word_docs = list(_coll.find(query, {"_id": 1, "properties.pos": 1}))
    if not word_docs:
        return f"Word '{word}'" + (f" ({pos})" if pos else "") + " not found."

    word_ids = [d["_id"] for d in word_docs]
    edges = list(_coll.find(
        {"doc_type": "baryedge", "level": 14,
         "$or": [{"cm1_id": {"$in": word_ids}}, {"cm2_id": {"$in": word_ids}}]},
        {"cm1_id": 1, "cm2_id": 1, "edge_type": 1, "q": 1, "accumulated_weight": 1},
    ))
    if not edges:
        return f"No L14 edges found for '{word}'. It may be an orphan — check word_senses."

    all_cm_ids = list({e["cm1_id"] for e in edges} | {e["cm2_id"] for e in edges})
    id_to_label: dict[Any, str] = {}
    for d in _coll.find({"_id": {"$in": all_cm_ids}},
                        {"properties.word": 1, "properties.pos": 1}):
        id_to_label[d["_id"]] = f"{d['properties']['word']} ({d['properties']['pos']})"

    return _fmt([
        {
            "edge_id": str(e["_id"]),
            "edge_type": e.get("edge_type"),
            "cm1": id_to_label.get(e["cm1_id"], str(e["cm1_id"])),
            "cm2": id_to_label.get(e["cm2_id"], str(e["cm2_id"])),
            "q": e.get("q"),
            "accumulated_weight": e.get("accumulated_weight"),
        }
        for e in edges
    ])


@mcp.tool()
def edge_info(edge_id: str) -> str:
    """Get full details about a BaryEdge or MetaBary including all CM leaf words.

    edge_id is the MongoDB ObjectId string from other tool results.
    CM leaf words are the actual words (L14/L15 nodes) reachable through the
    cm1_id/cm2_id chain — for L14/L15 BEs these are direct; for L13+ MBs
    the chain is traversed recursively.
    """
    try:
        oid = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}' — must be a 24-char hex ObjectId string."

    doc = _coll.find_one({"_id": oid})
    if not doc:
        return f"No document with id {edge_id}."

    words = cm_leaf_words(_coll, oid)
    return _fmt({
        "id": edge_id,
        "level": doc.get("level"),
        "edge_type": doc.get("edge_type"),
        "q": doc.get("q"),
        "connection_strength": doc.get("connection_strength"),
        "accumulated_weight": doc.get("accumulated_weight"),
        "cm_leaf_words": sorted(words),
        "cm1_id": str(doc.get("cm1_id")),
        "cm2_id": str(doc.get("cm2_id")),
        "has_parent": doc.get("parent_edge_id") is not None,
        "parent_id": str(doc["parent_edge_id"]) if doc.get("parent_edge_id") else None,
    })


@mcp.tool()
def semantic_search(query: str, doc_type: str = "baryedge", top_k: int = 10) -> str:
    """Semantic similarity search against the BaryGraph vector index (mongot).

    doc_type: 'baryedge' searches relationship vectors (default);
              'node' searches word/sense vectors.
    Requires `make pipeline-dev` to have run and s10_index to have completed.
    The HNSW index may take several minutes to build after creation.
    """
    try:
        embedder = get_embedder(_settings)
        qv = embedder.embed([query])[0].tolist()
    except Exception as e:
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    try:
        docs = vector_search(
            _coll, qv,
            limit=top_k,
            num_candidates=max(top_k * 10, 200),
            filter={"doc_type": doc_type},
        )
    except OperationFailure as e:
        return (
            "Vector search unavailable — the mongot index may still be building.\n"
            "Run `make pipeline-dev` then wait a few minutes for the HNSW index.\n"
            f"Error: {e}"
        )

    if not docs:
        return "No results returned. Index may still be building or corpus is empty."

    results = []
    for d in docs:
        r: dict[str, Any] = {
            "id": str(d["_id"]),
            "score": round(float(d.get("_score", 0)), 4),
            "level": d.get("level"),
        }
        if d["doc_type"] == "node":
            r["node_type"] = d.get("node_type")
            r["word"] = d.get("properties", {}).get("word")
            r["gloss"] = (d.get("properties", {}).get("gloss") or "")[:100]
        else:
            r["edge_type"] = d.get("edge_type")
            r["cm_words"] = sorted(cm_leaf_words(_coll, d["_id"]))
            r["accumulated_weight"] = d.get("accumulated_weight")
        results.append(r)

    return _fmt(results)


@mcp.tool()
def graph_stats() -> str:
    """Return document counts broken down by doc_type, level, node_type, and edge_type.

    Use this to check how much data has been ingested and what stages have run.
    """
    pipeline = [
        {"$group": {
            "_id": {
                "doc_type": "$doc_type",
                "level": "$level",
                "node_type": "$node_type",
                "edge_type": "$edge_type",
            },
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id.doc_type": 1, "_id.level": -1}},
    ]
    rows = list(_coll.aggregate(pipeline))
    total = _coll.count_documents({})
    return _fmt({
        "total_documents": total,
        "breakdown": [
            {k: v for k, v in r["_id"].items() if v is not None} | {"count": r["count"]}
            for r in rows
        ],
    })


if __name__ == "__main__":
    mcp.run()
