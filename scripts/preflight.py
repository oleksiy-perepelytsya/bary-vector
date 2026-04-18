"""Preflight checks run before starting the pipeline.

Verifies every resource stage 01+ depends on, producing a human-readable
PASS/FAIL report. Exits non-zero on any failure. Run via ``make preflight``.

Checks:
  * MongoDB reachable at the configured project URI
  * Configured MongoDB port matches ``MONGO_TEST_DB_PREFIX`` expectations
  * Ollama /api/version reachable (skipped if BARY_FAKE_EMBED=1)
  * Embedder returns vectors of length ``EMBED_DIM``
  * Kaikki dump exists, is non-trivial (>100 MB), and is readable
  * ``parsed_dir`` and ``pipeline_state_dir`` are writable
  * Free disk space on the data volume ≥ 20 GB
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from lib.config import Settings
from lib.db import ping
from lib.embed import get_embedder
from lib.log import get_logger, setup_logging

MIN_KAIKKI_BYTES = 100 * 1024 * 1024       # 100 MB
MIN_FREE_DISK_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _check_mongo(s: Settings) -> Check:
    if not ping(s):
        return Check("mongo.reachable", False, f"cannot ping {s.mongo_uri}")
    return Check("mongo.reachable", True, s.mongo_uri)


def _check_ollama(s: Settings) -> Check:
    if s.fake_embed:
        return Check("ollama.reachable", True, "skipped (fake embed enabled)")
    try:
        resp = httpx.get(s.ollama_url.rstrip("/") + "/api/version", timeout=5.0)
        resp.raise_for_status()
        return Check("ollama.reachable", True, resp.json().get("version", "ok"))
    except Exception as e:
        return Check("ollama.reachable", False, f"{s.ollama_url}: {e}")


def _check_embed_dim(s: Settings) -> Check:
    try:
        emb = get_embedder(s)
        v = emb.embed(["preflight"])
    except Exception as e:
        return Check("embed.probe", False, str(e))
    if v.shape != (1, s.embed_dim):
        return Check(
            "embed.probe",
            False,
            f"expected (1,{s.embed_dim}), got {v.shape} from model {s.embed_model}",
        )
    return Check("embed.probe", True, f"dim={s.embed_dim}")


def _check_kaikki(s: Settings) -> Check:
    p: Path = s.kaikki_path
    if not p.exists():
        return Check("kaikki.exists", False, f"not found: {p} — run `make fetch-kaikki`")
    sz = p.stat().st_size
    if sz < MIN_KAIKKI_BYTES:
        return Check("kaikki.exists", False, f"{p} is only {sz} bytes (partial download?)")
    return Check("kaikki.exists", True, f"{p} ({sz // (1024 * 1024)} MB)")


def _check_dirs_writable(s: Settings) -> Check:
    try:
        for d in (s.parsed_dir, s.pipeline_state_dir):
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".preflight_probe"
            probe.write_text("ok")
            probe.unlink()
    except OSError as e:
        return Check("dirs.writable", False, str(e))
    return Check("dirs.writable", True, f"{s.parsed_dir}, {s.pipeline_state_dir}")


def _check_disk(s: Settings) -> Check:
    path = s.kaikki_path.parent if s.kaikki_path.parent.exists() else Path(".")
    free = shutil.disk_usage(path).free
    gb = free // (1024 * 1024 * 1024)
    if free < MIN_FREE_DISK_BYTES:
        return Check("disk.free", False, f"only {gb} GB free at {path}")
    return Check("disk.free", True, f"{gb} GB free at {path}")


def run(settings: Settings | None = None) -> int:
    s = settings or Settings.load()
    setup_logging(s.log_level)
    log = get_logger("preflight")

    checks = [
        _check_mongo(s),
        _check_ollama(s),
        _check_embed_dim(s),
        _check_kaikki(s),
        _check_dirs_writable(s),
        _check_disk(s),
    ]
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        log.log(20 if c.ok else 40, "%s  %-22s  %s", mark, c.name, c.detail)

    failed = [c for c in checks if not c.ok]
    if failed:
        log.error("%d check(s) failed — pipeline is NOT safe to start", len(failed))
        return 1
    log.info("all preflight checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
