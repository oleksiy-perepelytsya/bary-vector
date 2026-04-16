from __future__ import annotations

from lib import checkpoint as cp_mod
from lib.checkpoint import Checkpoint


def test_roundtrip(settings):
    cp = Checkpoint(stage="04_l15_edges", last_id="abc", processed=10, total=100)
    cp_mod.save(cp, settings)
    loaded = cp_mod.load("04_l15_edges", settings)
    assert loaded is not None
    assert loaded.last_id == "abc"
    assert loaded.processed == 10
    assert loaded.updated_at  # touched


def test_atomic_no_tmp_left(settings, tmp_state_dir):
    cp_mod.save(Checkpoint(stage="x"), settings)
    leftovers = list(tmp_state_dir.glob("*.tmp"))
    assert leftovers == []


def test_reset(settings):
    cp_mod.save(Checkpoint(stage="y"), settings)
    assert cp_mod.load("y", settings) is not None
    cp_mod.reset("y", settings)
    assert cp_mod.load("y", settings) is None


def test_load_missing(settings):
    assert cp_mod.load("nope", settings) is None
