from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import mongomock
import numpy as np
import pytest

import lib.assoc_search as A
from lib.assoc_search import (
    AssocConfig,
    Candidate,
    Session,
    progressive,
    run_search,
)

# ---------------------------------------------------------------------------
# Test graph (deterministic vectors, no mongot needed)
#
#   s_m(memory)      -> be_a(L15 mem/blank, q=.9) -> L13_a --+--> L12_1 (target,
#   s_b(blank)       -> be_a                               |      2 independent
#   s_d(documentary) -> be_b(L15 doc/film,  q=.9) -> L13_b -+      routes)
#   s_f(film)        -> be_b
#   s_x(annal)       -> be_c(L15 annal/gamebook, q=.9) -> L13_c -> L12_2
#   s_g(gamebook)    -> be_c
#   L12_1 -> L11_1 -> L10_1 ;  L12_2 -> L12_1
#   w_memory = word node (for exact-headword seed matching)
# ---------------------------------------------------------------------------


def _norm(v: list[float]) -> list[float]:
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


VEC_M = _norm([1.0, 0, 0, 0, 0, 0, 0, 0])       # memory
VEC_B = _norm([0.9, 0.3, 0, 0, 0, 0, 0, 0])     # blank
VEC_D = _norm([0.0, 0, 1, 0, 0, 0, 0, 0])       # documentary
VEC_F = _norm([0.0, 0, 0.9, 0.3, 0, 0, 0, 0])   # film
VEC_X = _norm([0.0, 0, 0, 0, 1, 0, 0, 0])       # annal
VEC_G = _norm([0.0, 0, 0, 0, 0.9, 0.3, 0, 0])   # gamebook
VEC_G1 = _norm([0, 0, 0, 0, 0, 0, 1.0, 0])      # far anchor (L11)
VEC_G2 = _norm([0, 0, 0, 0, 0, 0, 0.5, 0.8])    # far anchor (L10)
NODE_NAMES = ("s_m", "s_b", "s_d", "s_f", "s_x", "s_g",
              "w_memory", "be_a", "be_b", "be_c",
              "L13_a", "L13_b", "L13_c", "L12_1", "L12_2",
              "L11_1", "L10_1")


def _pair_vec(u: list[float], v: list[float]) -> list[float]:
    x = np.asarray(u, dtype=np.float32) + np.asarray(v, dtype=np.float32)
    return (x / np.linalg.norm(x)).tolist()


def build_graph() -> tuple[Any, dict[str, Any]]:
    """Insert the test graph; returns (coll, name->_id lookup)."""
    coll = mongomock.MongoClient().db["barygraph"]
    names = {n: mongomock.ObjectId() for n in NODE_NAMES}
    docs: list[dict[str, Any]] = []

    def _sense(word: str, vec: list[float], name: str, parent_be: str) -> None:
        docs.append({
            "_id": names[name], "doc_type": "node", "node_type": "sense",
            "level": 15, "vector": vec, "parent_edge_id": names[parent_be],
            "properties": {"word": word, "lang": "en"},
        })

    def _be(name: str, u: list[float], v: list[float], cm1: str, cm2: str,
            parent: str) -> None:
        docs.append({
            "_id": names[name], "doc_type": "baryedge", "level": 15,
            "edge_type": "synonyms", "vector": _pair_vec(u, v),
            "parent_edge_id": names[parent], "connection_strength": 0.9,
            "cm1_id": names[cm1], "cm2_id": names[cm2],
        })

    def _mb(name: str, level: int, vec: list[float], cm1: str, cm2: str,
            parent: str | None, bridge: str | None = None) -> None:
        doc: dict[str, Any] = {
            "_id": names[name], "doc_type": "baryedge", "level": level,
            "vector": vec, "parent_edge_id": names[parent] if parent else None,
            "cm1_id": names[cm1], "cm2_id": names[cm2],
        }
        if bridge is not None:
            doc["bridge_id"] = names[bridge]
        docs.append(doc)

    class n:
        """Dotted namespaces for readable paren references."""
        s_m, s_b = "s_m", "s_b"
        s_d, s_f = "s_d", "s_f"
        s_x, s_g = "s_x", "s_g"
        w_memory = "w_memory"
        be_a, be_b, be_c = "be_a", "be_b", "be_c"
        L13_a, L13_b, L13_c = "L13_a", "L13_b", "L13_c"
        L12_1, L12_2 = "L12_1", "L12_2"
        L11_1, L10_1 = "L11_1", "L10_1"

    # word node for exact-headword seeding
    docs.append({
        "_id": names[n.w_memory], "doc_type": "node", "node_type": "word",
        "level": 14, "vector": _pair_vec(VEC_M, VEC_B), "parent_edge_id": None,
        "properties": {"word": "memory", "lang": "en"},
    })

    # senses (parent = the BE they are paired into)
    _sense("memory", VEC_M, n.s_m, "be_a")
    _sense("blank", VEC_B, n.s_b, "be_a")
    _sense("documentary", VEC_D, n.s_d, "be_b")
    _sense("film", VEC_F, n.s_f, "be_b")
    _sense("annal", VEC_X, n.s_x, "be_c")
    _sense("gamebook", VEC_G, n.s_g, "be_c")

    # level-15 BaryEdges (pair senses into structures)
    be_a = _pair_vec(VEC_M, VEC_B)
    be_b = _pair_vec(VEC_D, VEC_F)
    be_c = _pair_vec(VEC_X, VEC_G)
    _be("be_a", VEC_M, VEC_B, n.s_m, n.s_b, n.L13_a)
    _be("be_b", VEC_D, VEC_F, n.s_d, n.s_f, n.L13_b)
    _be("be_c", VEC_X, VEC_G, n.s_x, n.s_g, n.L13_c)

    # MetaBarys
    l13_a = _pair_vec(be_a, VEC_M)
    l13_b = _pair_vec(be_b, VEC_D)
    l13_c = _pair_vec(be_c, VEC_X)
    l12_1 = _pair_vec(l13_a, l13_b)
    l12_2 = _pair_vec(l13_b, l13_c)
    l11_1 = _pair_vec(l12_1, VEC_G1)
    l10_1 = _pair_vec(l11_1, VEC_G2)
    _mb("L13_a", 13, l13_a, n.be_a, n.be_b, n.L12_1, bridge="be_a")
    _mb("L13_b", 13, l13_b, n.be_b, n.be_c, n.L12_1, bridge="be_b")
    _mb("L13_c", 13, l13_c, n.be_c, n.be_a, n.L12_2, bridge="be_c")
    _mb("L12_1", 12, l12_1, n.L13_a, n.L13_b, n.L11_1)
    _mb("L12_2", 12, l12_2, n.L13_b, n.L13_c, n.L12_1)
    _mb("L11_1", 11, l11_1, n.L12_1, n.L12_2, n.L10_1)
    _mb("L10_1", 10, l10_1, n.L11_1, n.L12_1, None)

    coll.insert_many(docs)
    return coll, names


class _RaisesEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError(f"boom: {texts}")


def _cfg(**kwargs) -> AssocConfig:
    kwargs.setdefault("result_top_k", 5)
    return AssocConfig(**kwargs)


def _qv() -> np.ndarray:
    return np.asarray(_pair_vec(VEC_M, VEC_D), dtype=np.float32)


@pytest.fixture
def graph() -> tuple[Any, dict[str, Any]]:
    """Fresh mongomock graph per test (avoids cross-test cache interference)."""
    return build_graph()


@pytest.fixture(autouse=True)
def _clean_modules_state():
    A._SESSIONS.clear()
    A._WORD_CACHE.clear()
    yield
    A._SESSIONS.clear()
    A._WORD_CACHE.clear()


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_config_from_kwargs_defaults_and_validation():
    cfg = A._config_from_kwargs()
    assert cfg.target_levels == (12, 11, 10)
    assert cfg.bridge_top_k == 12
    with pytest.raises(ValueError, match="target_levels"):
        A._config_from_kwargs(target_levels=())
    with pytest.raises(ValueError, match="target_levels"):
        A._config_from_kwargs(target_levels=(0, 11))
    with pytest.raises(ValueError, match="target_levels"):
        A._config_from_kwargs(target_levels=(14,))
    # Non-level knobs are clamped, not raised, so tool params never crash.
    assert A._config_from_kwargs(min_convergence=0).min_convergence == 1
    assert A._config_from_kwargs(max_hops=0).max_hops == 1
    assert A._config_from_kwargs(bridge_top_k=0).bridge_top_k == 1


# ---------------------------------------------------------------------------
# seed resolution
# ---------------------------------------------------------------------------


def test_resolve_seeds_exact_word(graph):
    coll, _ = graph
    seeds = A.resolve_seeds(coll, _qv(), AssocConfig(), query="memory",
                            seed_ids=[])
    assert any(s.origin_word == "memory" for s in seeds)
    assert all(s.level == 14 for s in seeds if s.origin_word == "memory")


def test_resolve_seeds_by_ids_respects_top_n(graph):
    coll, names = graph
    ids = [names["s_m"], names["s_d"], names["s_x"], names["s_g"]]
    seeds = A.resolve_seeds(coll, _qv(), AssocConfig(seed_top_k=2),
                           query="", seed_ids=ids)
    assert len(seeds) == 2
    assert {s.node_id for s in seeds} == {names["s_m"], names["s_d"]}


def test_resolve_seeds_empty_ids_no_seed(graph):
    coll, _ = graph
    seeds = A.resolve_seeds(coll, _qv(), AssocConfig(), query="",
                            seed_ids=[mongomock.ObjectId()])
    assert seeds == []


# ---------------------------------------------------------------------------
# upward propagation mechanics
# ---------------------------------------------------------------------------


def test_expand_upward_decays_energy_and_marks_dead_ends(graph):
    coll, names = graph
    qv = _qv()
    config = AssocConfig()
    s_m = A._seed_from_doc(coll, coll.find_one({"_id": names["s_m"]}), qv)
    assert s_m is not None
    assert s_m.level == 15 and s_m.energy == 1.0

    expanded = A.expand_upward(coll, [s_m], config, [s_m.vector], qv)
    assert len(expanded) == 1
    hop1 = expanded[0]
    assert hop1.node_id == names["be_a"]
    assert hop1.conn_q == pytest.approx(0.9)
    assert hop1.energy == pytest.approx(1.0 * 0.9 * config.beam_decay)
    assert hop1.first_baryedge_id == names["be_a"]

    # Up 3 more hops: be_a -> L13_a -> L12_1 -> L11_1.
    reach = [hop1]
    for _ in range(3):
        reach = A.expand_upward(coll, reach, config, [s_m.vector], qv)
    assert len(reach) == 1
    assert reach[0].node_id == names["L11_1"]

    # A candidate whose parent is missing dies after one attempt.
    orphan = A._seed_from_doc(coll, coll.find_one({"_id": names["s_x"]}), qv)
    assert orphan is not None
    orphan.parent_id = mongomock.ObjectId()
    out = A.expand_upward(coll, [orphan], config, [], qv)
    assert out == []
    assert orphan.dead


# ---------------------------------------------------------------------------
# beam retention (three channels, spec §5)
# ---------------------------------------------------------------------------


def test_beam_select_strong_divergent_convergent_channels():
    config = AssocConfig(bridge_top_k=3)
    qv = _qv()

    def cand(node_id: Any, vec: list[float], novelty: float, groups: int,
             energy: float, local: float = 0.5, conn: float = 0.5) -> Candidate:
        return Candidate(
            node_id=node_id, level=12,
            vector=np.asarray(vec, dtype=np.float32), parent_id=None,
            conn_q=conn, origin_word="w", origin_id=node_id, origin_lang="en",
            first_baryedge_id=None, path_ids=[node_id], path_steps=[],
            local_score=local, energy=energy, branch_query=qv.copy(),
            arrival_groups=groups, novelty=novelty,
        )

    m1 = cand(1, _norm([0.9, 0.1, 0, 0, 0, 0, 0, 0]),
              novelty=0.1, groups=1, energy=0.7, local=0.95)   # strong
    m2 = cand(2, _norm([0.0, 0, 0, 0, 0.8, 0.2, 0, 0]),
              novelty=0.8, groups=1, energy=0.4, local=0.3)    # divergent
    m3 = cand(3, _norm([0.1, 0.1, 0.9, 0, 0, 0, 0, 0]),
              novelty=0.2, groups=4, energy=0.15, local=0.2)   # convergent-low
    m4 = cand(4, _norm([0, 0, 0, 0.7, 0.7, 0, 0, 0]),
              novelty=0.5, groups=1, energy=0.6)

    picked = A._beam_select([m1, m1, m2, m3, m4], config)
    assert len(picked) == 3
    picked_ids = {c.node_id for c in picked}
    assert picked_ids == {1, 2, 3}


# ---------------------------------------------------------------------------
# convergence + ranking (spec §6, §7)
# ---------------------------------------------------------------------------


def test_rank_targets_convergence_filter_and_novelty():
    qv = _qv()

    def tgt(node_id: Any, vec: list[float], energy: float) -> Candidate:
        return Candidate(
            node_id=node_id, level=12,
            vector=np.asarray(vec, dtype=np.float32), parent_id=None,
            conn_q=0.7, origin_word="w", origin_id=node_id, origin_lang="en",
            first_baryedge_id=node_id, path_ids=[node_id], path_steps=[],
            local_score=0.5, energy=energy, branch_query=qv.copy(),
        )

    seed = Candidate(
        node_id=mongomock.ObjectId(), level=15,
        vector=np.asarray(VEC_M, dtype=np.float32), parent_id=None, conn_q=1.0,
        origin_word="memory", origin_id=0, origin_lang="en",
        first_baryedge_id=None, path_ids=[], path_steps=[], local_score=1.0,
        energy=1.0, branch_query=qv.copy(),
    )
    near = tgt(1, _pair_vec(VEC_M, VEC_B), energy=0.5)      # close to seed
    far = tgt(2, _norm([0, 0, 0, 0, 0, 0, 0.9, 0.1]), energy=0.5)

    # Two independent routes (different origin words) converge on `near`.
    support: dict[Any, dict[tuple[str, str, str], Candidate]] = {
        near.node_id: {
            ("memory", str(near.node_id), "en"): near,
            ("documentary", str(near.node_id), "en"): tgt(1, near.vector, 0.5),
        },
        far.node_id: {("annal", str(far.node_id), "en"): far},
    }
    ranked = A._rank_targets(qv, [seed], support, AssocConfig(min_convergence=1))
    by_id = {int(r["target"].node_id): r for r in ranked}
    assert set(by_id) == {1, 2}
    assert by_id[1]["support_paths"] == 2
    assert by_id[2]["support_paths"] == 1
    assert by_id[1]["convergence"] == pytest.approx(1.0 - 0.25)
    assert by_id[1]["novelty"] < by_id[2]["novelty"]
    assert ranked[0]["target"].node_id == near.node_id

    # min_convergence=2 drops the single-route target.
    ranked2 = A._rank_targets(qv, [], support, AssocConfig(min_convergence=2))
    assert {int(r["target"].node_id) for r in ranked2} == {1}


# ---------------------------------------------------------------------------
# status contract (spec §13)
# ---------------------------------------------------------------------------


def test_run_search_no_seed(graph):
    coll, _ = graph
    payload = run_search(coll, None, "", _cfg(), qv=_qv(),
                         seed_ids=[mongomock.ObjectId()])
    assert payload["status"] == "no_seed"
    assert payload["seed_count"] == 0


def test_run_search_embedding_error():
    coll = mongomock.MongoClient().db["barygraph"]
    payload = run_search(coll, _RaisesEmbedder(), "memory", _cfg())
    assert payload["status"] == "error"
    assert "Embedding failed" in payload["message"]


def test_run_search_no_target_when_hops_insufficient(graph):
    coll, names = graph
    payload = run_search(
        coll, None, "", _cfg(max_hops=4, target_levels=(10,)), qv=_qv(),
        seed_ids=[names["s_m"]],
    )
    assert payload["status"] == "no_target"
    assert payload["highest_reached_level"] == 11
    assert "requested target levels" in payload["message"]


# ---------------------------------------------------------------------------
# one-shot end-to-end (spec §8)
# ---------------------------------------------------------------------------


def test_run_search_reaches_convergent_coordinate(graph):
    coll, names = graph
    payload = run_search(
        coll, None, "", _cfg(), qv=_qv(),
        seed_ids=[names["s_m"], names["s_d"]],
    )
    assert payload["status"] == "ok"
    assert payload["seed_count"] == 2
    assert payload["highest_reached_level"] == 11
    top = payload["results"][0]
    assert top["level"] == 12
    assert top["support_paths"] == 2            # memory-route + documentary-route
    assert set(top["words"]) >= {"memory", "blank", "documentary", "film"}
    assert top["trigger"]["level"] == 15
    assert top["trigger"]["edge_type"] == "synonyms"
    assert top["why"]["trigger"]
    assert top["why"]["support"]
    assert payload["results"][1]["level"] == 11
    assert "ASSOCIATIVE COORDINATE" in payload["summary"]


def test_run_search_trigger_is_first_baryedge(graph):
    coll, names = graph
    payload = run_search(coll, None, "", _cfg(), qv=_qv(),
                         seed_ids=[names["s_m"]])
    top = payload["results"][0]
    assert top["trigger"]["id"] == str(names["be_a"])
    assert set(top["trigger"]["words"]) == {"memory", "blank"}


def test_run_search_no_paths_when_return_paths_false(graph):
    coll, names = graph
    payload = run_search(
        coll, None, "", _cfg(return_paths=False), qv=_qv(),
        seed_ids=[names["s_m"], names["s_d"]],
    )
    assert payload["status"] == "ok"
    assert "paths" not in payload["results"][0]
    assert "path_steps" not in payload["results"][0]


# ---------------------------------------------------------------------------
# progressive sessions (spec §9)
# ---------------------------------------------------------------------------


def test_progressive_discover_expand_compare_propose(monkeypatch, graph):
    coll, names = graph
    qv = _qv()
    cfg = _cfg()

    def fake_hits(c, qv_, budget_node, budget_be):
        return [c.find_one({"_id": names[k]}) for k in ("s_m", "s_d", "s_x")]

    monkeypatch.setattr(A, "_vector_seed_hits", fake_hits)

    discover = progressive(coll, None, "sess-1", "discover", query="qu",
                           config=cfg, qv=qv)
    assert discover["status"] == "ok"
    assert discover["stage"] == "discover"
    cands = discover["candidates"]
    assert {c["level"] for c in cands} >= {12, 11}
    ids12 = [c["id"] for c in cands if c["level"] == 12]
    assert len(ids12) == 2

    # discover without a query -> error
    bad = progressive(coll, None, "sess-x", "discover", config=cfg, qv=qv)
    assert bad["status"] == "error"

    # non-discover stage without a session -> error
    wrong = progressive(coll, None, "nope", "expand", selected_ids=["x"],
                        config=cfg)
    assert wrong["status"] == "error"
    assert "discover first" in wrong["message"]

    # expand on the two level-12 coordinates
    exp = progressive(coll, None, "sess-1", "expand", selected_ids=ids12,
                      config=cfg)
    assert exp["status"] == "ok"
    assert len(exp["expanded"]) == 2
    for e in exp["expanded"]:
        assert e["level"] == 12
        assert set(e["triad"]) == {"child1", "child2", "bridge"}
        assert e["words"]
        assert e["strongest_leaf_support"]

    # compare the two coordinates
    cmp = progressive(coll, None, "sess-1", "compare", selected_ids=ids12,
                      config=cfg)
    assert cmp["status"] == "ok"
    assert len(cmp["pairs"]) == 1
    pair = cmp["pairs"][0]
    assert pair["a_level"] == pair["b_level"] == 12
    assert 0.0 <= pair["cosine"] <= 1.0
    assert {"word_overlap", "shared_origins", "shared_trigger"} <= set(pair)

    # propose an SMB packet for the two coordinates (not created)
    prop = progressive(coll, None, "sess-1", "propose", selected_ids=ids12,
                       config=cfg)
    assert prop["status"] == "ok"
    packet = prop["packet"]
    assert packet["child_level"] == 12
    assert packet["smb_level"] == 10
    assert {packet["cm1_id"], packet["cm2_id"]} == {
        str(names["L12_1"]), str(names["L12_2"]),
    }
    assert packet["expected_child_cosine"] > 0.4
    assert packet["cm1_words"] and packet["cm2_words"]
    assert "create_structure_meta_bary" in packet["note"]

    # unknown stage
    unknown = progressive(coll, None, "sess-1", "frobnicate",
                          selected_ids=ids12, config=cfg)
    assert unknown["status"] == "error"
    assert "unknown stage" in unknown["message"]


def test_progressive_needs_same_level_for_propose(monkeypatch, graph):
    coll, names = graph
    qv = _qv()
    cfg = _cfg()

    def fake_hits(c, qv_, budget_node, budget_be):
        return [c.find_one({"_id": names[k]}) for k in ("s_m", "s_d")]

    monkeypatch.setattr(A, "_vector_seed_hits", fake_hits)
    discover = progressive(coll, None, "sess-2", "discover", "qu2",
                           config=cfg, qv=qv)
    assert discover["status"] == "ok"
    lvl_12 = next(c["id"] for c in discover["candidates"] if c["level"] == 12)
    lvl_11 = next(c["id"] for c in discover["candidates"] if c["level"] == 11)
    prop = progressive(coll, None, "sess-2", "propose",
                       selected_ids=[lvl_12, lvl_11], config=cfg)
    assert prop["status"] == "error"
    assert "share a level" in prop["message"]


def test_propose_with_bridge_coll_and_candidates_does_not_overflow(monkeypatch, graph):
    """Regression: the bridge loop used `for b in bridges:`, rebinding the
    propose tuple `b` to the last bridge dict → KeyError('convergence') on the
    live server (which always passes bridge_coll). Only reproducible with a
    non-empty bridges list + a bridge_coll."""
    coll, names = graph
    qv = _qv()
    cfg = _cfg()
    bridge_coll = mongomock.MongoClient().db["doi_bridges"]

    def fake_hits(c, qv_, budget_node, budget_be):
        return [c.find_one({"_id": names[k]}) for k in ("s_m", "s_d", "s_x")]

    monkeypatch.setattr(A, "_vector_seed_hits", fake_hits)
    # bridge probe returns a real MB (id resolvable by _step_words)
    monkeypatch.setattr(A, "vector_search",
                        lambda *a, **k: [coll.find_one({"_id": names["L13_a"]})])
    discover = progressive(coll, None, "sess-p", "discover", "qu", config=cfg, qv=qv)
    ids12 = [c["id"] for c in discover["candidates"] if c["level"] == 12][:2]
    assert len(ids12) == 2
    prop = progressive(coll, None, "sess-p", "propose",
                       selected_ids=ids12, config=cfg, bridge_coll=bridge_coll)
    assert prop["status"] == "ok"
    packet = prop["packet"]
    assert packet["bridge_candidates"]  # the shadowing bug crashed here
    assert {"cm1_id", "cm2_id", "expected_child_cosine", "prior_smbs"} <= set(packet)


def test_prior_smbs_near_is_bounded_and_filters(monkeypatch, graph):
    coll, names = graph
    qv = _qv()
    fake_hits = [
        {"_id": names["L12_1"], "source": "structural", "vector": qv.tolist()},
        {"_id": names["L12_2"], "source": None, "vector": qv.tolist()},
        {"_id": names["L12_1"], "source": "structural", "vector": qv.tolist()},
    ]
    monkeypatch.setattr(A, "vector_search", lambda *a, **k: fake_hits)
    monkeypatch.setattr(A, "_clamp_sim", lambda x: x)
    region = set(A._step_words(coll, names["L12_1"], 20))
    out = A._prior_smbs_near(coll, qv, region)
    assert len(out) == 2 and all(m["cosine"] >= 0.5 for m in out)
    for i in range(A._MAX_SESSIONS + 4):
        A._session_put(Session(
            session_id=f"s{i}", query="q", config=AssocConfig(), qv=_qv(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
    assert len(A._SESSIONS) <= A._MAX_SESSIONS
    assert "s0" not in A._SESSIONS  # oldest evicted first
    assert f"s{A._MAX_SESSIONS + 3}" in A._SESSIONS
