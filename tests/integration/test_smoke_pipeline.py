from __future__ import annotations

import importlib

import pytest

from lib.config import Settings
from lib.db import STANDARD_INDEXES, ensure_indexes, ping
from scripts._base import STAGES

pytestmark = pytest.mark.integration

FIXTURE = "tests/fixtures/kaikki-sample.jsonl"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_state_dir, mongo_test_db):
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    monkeypatch.setenv("KAIKKI_PATH", FIXTURE)
    yield


def test_mongo_reachable():
    assert ping(Settings.load()) is True


def test_test_db_carries_project_prefix(mongo_test_db):
    """Isolation invariant: the integration DB name must be project-owned."""
    s = Settings.load()
    assert s.mongo_db.startswith(s.mongo_test_db_prefix)
    assert mongo_test_db.database.name == s.mongo_db


def test_ensure_indexes(mongo_test_db):
    names = ensure_indexes(mongo_test_db)
    assert len(names) == len(STANDARD_INDEXES)
    info = mongo_test_db.index_information()
    # _id_ + the standard ones
    assert len(info) == len(STANDARD_INDEXES) + 1


@pytest.mark.parametrize("stage", list(STAGES))
def test_stage_bootstraps_then_not_implemented(stage, tmp_state_dir):
    """Harness: each stage must connect, write a checkpoint, then raise.

    Future stage PRs replace the `pytest.raises` with real assertions.
    """
    mod = importlib.import_module(STAGES[stage])
    with pytest.raises(NotImplementedError):
        mod.run([])
    assert (tmp_state_dir / f"{stage}.json").exists()
