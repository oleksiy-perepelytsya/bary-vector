from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bson import ObjectId

from lib.assoc_search import run_search

pytestmark = pytest.mark.integration


def _norm(v: list[float]) -> list[float]:
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


VEC_M = _norm([1.0, 0, 0, 0, 0, 0, 0, 0])
VEC_B = _norm([0.9, 0.3, 0, 0, 0, 0, 0, 0])
VEC_D = _norm([0.0, 0, 1, 0, 0, 0, 0, 0])
VEC_F = _norm([0.0, 0, 0.9, 0.3, 0, 0, 0, 0])
QV = np.asarray(_norm([1.0, 0, 0.9, 0.3, 0, 0, 0, 0]), dtype=np.float32)


def _pair_vec(u: list[float], v: list[float]) -> list[float]:
    x = np.asarray(u, dtype=np.float32) + np.asarray(v, dtype=np.float32)
    return (x / np.linalg.norm(x)).tolist()


@pytest.fixture
def graph_coll(mongo_test_db):
    """Seed a thin two-route graph into the real Mongo test DB.

    memory -> be_a(L15) -> L13_a -+-> L12 (target) -> L11
    documentary -> be_b(L15) -> L13_b -'
    Two independent routes converge on the same L12 coordinate.
    """
    coll = mongo_test_db
    ids = {n: ObjectId() for n in (
        "s_m", "s_d", "be_a", "be_b", "L13_a", "L13_b", "L12", "L11")}

    docs: list[dict[str, Any]] = []
    be_of = {"s_m": "be_a", "s_d": "be_b"}
    for name, word, vec in (("s_m", "memory", VEC_M), ("s_d", "documentary", VEC_D)):
        docs.append({
            "_id": ids[name], "doc_type": "node", "node_type": "sense", "level": 15,
            "vector": vec, "parent_edge_id": ids[be_of[name]],
            "properties": {"word": word, "lang": "en"},
        })
    l13_of = {"be_a": "L13_a", "be_b": "L13_b"}
    for name, u, v in (("be_a", VEC_M, VEC_B), ("be_b", VEC_D, VEC_F)):
        docs.append({
            "_id": ids[name], "doc_type": "baryedge", "level": 15,
            "edge_type": "synonyms", "vector": _pair_vec(u, v),
            "parent_edge_id": ids[l13_of[name]], "connection_strength": 0.9,
            "cm1_id": ids["s_m"], "cm2_id": ids["s_d"],
        })
    docs.extend([
        {"_id": ids["L13_a"], "doc_type": "baryedge", "level": 13,
         "vector": _pair_vec(VEC_M, VEC_D), "parent_edge_id": ids["L12"],
         "cm1_id": ids["be_a"], "cm2_id": ids["be_b"]},
        {"_id": ids["L13_b"], "doc_type": "baryedge", "level": 13,
         "vector": _pair_vec(VEC_B, VEC_F), "parent_edge_id": ids["L12"],
         "cm1_id": ids["be_b"], "cm2_id": ids["be_a"]},
        {"_id": ids["L12"], "doc_type": "baryedge", "level": 12,
         "vector": _pair_vec(_pair_vec(VEC_M, VEC_D),
                             _pair_vec(VEC_B, VEC_F)),
         "parent_edge_id": ids["L11"],
         "cm1_id": ids["L13_a"], "cm2_id": ids["L13_b"]},
        {"_id": ids["L11"], "doc_type": "baryedge", "level": 11,
         "vector": _pair_vec(_pair_vec(VEC_M, VEC_B),
                             _norm([0, 0, 0, 0, 1.0, 0, 0, 0])),
         "parent_edge_id": None,
         "cm1_id": ids["L12"], "cm2_id": ids["L12"]},
    ])
    coll.insert_many(docs)
    return coll, ids


@pytest.fixture
def doi_coll(mongo_test_db):
    from lib.config import Settings
    from lib.db import get_client

    s = Settings.load()
    return get_client(s)[s.mongo_db][s.mongo_doi_bridges_collection]


def _clear_collection(coll):
    coll.delete_many({})


def test_assoc_search_convergence_against_real_mongo(graph_coll):
    coll, ids = graph_coll
    from lib.assoc_search import AssocConfig

    payload = run_search(
        coll, None, "", AssocConfig(max_hops=4), qv=QV,
        seed_ids=[ids["s_m"], ids["s_d"]],
    )
    assert payload["status"] == "ok"
    assert payload["seed_count"] == 2
    assert payload["highest_reached_level"] == 11
    top = payload["results"][0]
    assert top["level"] == 12
    assert top["id"] == str(ids["L12"])
    assert top["support_paths"] == 2          # memory-route + documentary-route
    assert set(top["words"]) >= {"memory", "documentary"}
    assert top["trigger"]["id"] in {str(ids["be_a"]), str(ids["be_b"])}
    assert set(top["trigger"]["words"]) <= {"memory", "blank", "documentary", "film"}
    assert top["why"]["trigger"]
    assert top["why"]["support"]
    assert payload["results"][1]["level"] == 11
    assert "ASSOCIATIVE COORDINATE" in payload["summary"]


def test_assoc_search_no_seed_real_mongo(mongo_test_db):
    from lib.assoc_search import AssocConfig

    payload = run_search(
        mongo_test_db, None, "", AssocConfig(), qv=QV,
        seed_ids=[ObjectId()],
    )
    assert payload["status"] == "no_seed"
    assert payload["seed_count"] == 0


def test_assoc_search_include_dois_real_mongo(graph_coll, doi_coll):
    from lib import doi_bridge
    from lib.assoc_search import AssocConfig

    coll, ids = graph_coll
    doi_bridge.register(doi_coll, ["10.9999/memory"], ids["s_m"])
    doi_bridge.register(doi_coll, ["10.9999/blank"], ids["s_m"])
    # Pipeline convention: DOIs propagate upward with every structure write.
    doi_bridge.propagate_up_chain(doi_coll, coll, ids["s_m"], ["10.9999/memory"])

    payload = run_search(
        coll, None, "", AssocConfig(include_dois=True), qv=QV,
        seed_ids=[ids["s_m"], ids["s_d"]], bridge_coll=doi_coll,
    )
    assert payload["status"] == "ok"
    top = payload["results"][0]
    assert "10.9999/memory" in top["dois"]

    _clear_collection(doi_coll)


def test_assoc_search_no_target_when_deeper_than_hops(graph_coll):
    from lib.assoc_search import AssocConfig

    coll, ids = graph_coll
    payload = run_search(
        coll, None, "", AssocConfig(max_hops=2, target_levels=(11, 10)),
        qv=QV, seed_ids=[ids["s_m"]],
    )
    assert payload["status"] == "no_target"
    assert payload["highest_reached_level"] == 13
