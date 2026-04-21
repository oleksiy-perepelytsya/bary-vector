from __future__ import annotations

import pytest

from lib import checkpoint as cp_mod
from lib.checkpoint import Checkpoint
from scripts._base import (
    STAGE_ORDER,
    StageOrderError,
    _enforce_order,
    _prev_stage,
    bootstrap,
    invalidate_downstream,
)


def test_stage_order_matches_dict():
    assert STAGE_ORDER[0] == "01_parse"
    assert STAGE_ORDER[-1] == "10_index"
    assert _prev_stage("01_parse") is None
    assert _prev_stage("05_word_vectors") == "04_l15_edges"


def test_enforce_order_blocks_without_prereq(settings):
    import logging
    log = logging.getLogger("t")
    with pytest.raises(StageOrderError):
        _enforce_order("02_embed", settings, force=False, log=log)


def test_enforce_order_passes_after_prereq_done(settings):
    import logging
    cp_mod.save(Checkpoint(stage="01_parse").mark_done(), settings)
    _enforce_order("02_embed", settings, force=False, log=logging.getLogger("t"))


def test_enforce_order_force_bypasses(settings):
    import logging
    _enforce_order("08_metabary", settings, force=True, log=logging.getLogger("t"))


def test_bootstrap_refuses_rerun_of_done_stage(settings, monkeypatch):
    cp_mod.save(Checkpoint(stage="01_parse").mark_done(), settings)
    with pytest.raises(StageOrderError, match="already completed"):
        bootstrap("01_parse", argv=[])


def test_bootstrap_reset_clears_done_and_downstream(settings):
    cp_mod.save(Checkpoint(stage="01_parse").mark_done(), settings)
    cp_mod.save(Checkpoint(stage="02_embed").mark_done(), settings)
    cp_mod.save(Checkpoint(stage="03_insert_nodes").mark_done(), settings)
    # Reset stage 02 → its own + 03's checkpoints cleared; 01 untouched.
    _settings, args, _log, cp = bootstrap("02_embed", argv=["--reset"])
    assert cp.done is False
    assert cp_mod.is_done("01_parse", settings)
    assert not cp_mod.is_done("02_embed", settings)
    assert cp_mod.load("03_insert_nodes", settings) is None


def test_invalidate_downstream(settings):
    import logging
    for s in STAGE_ORDER:
        cp_mod.save(Checkpoint(stage=s).mark_done(), settings)
    invalidate_downstream("04_l15_edges", settings, logging.getLogger("t"))
    assert cp_mod.is_done("04_l15_edges", settings)
    assert cp_mod.load("05_word_vectors", settings) is None
    assert cp_mod.load("10_index", settings) is None
