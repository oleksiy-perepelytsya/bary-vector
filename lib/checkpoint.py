from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import orjson

from lib.config import Settings


@dataclass
class Checkpoint:
    stage: str
    last_id: str | None = None
    processed: int = 0
    total: int = 0
    done: bool = False
    completed_at: str | None = None
    updated_at: str = ""

    def touch(self) -> Checkpoint:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self

    def mark_done(self) -> Checkpoint:
        self.done = True
        self.completed_at = datetime.now(timezone.utc).isoformat()
        return self


def is_done(stage: str, settings: Settings) -> bool:
    cp = load(stage, settings)
    return cp is not None and cp.done


def _path(stage: str, settings: Settings) -> Path:
    return Path(settings.pipeline_state_dir) / f"{stage}.json"


def load(stage: str, settings: Settings) -> Checkpoint | None:
    p = _path(stage, settings)
    if not p.exists():
        return None
    data = orjson.loads(p.read_bytes())
    return Checkpoint(**data)


def save(cp: Checkpoint, settings: Settings) -> None:
    cp.touch()
    p = _path(cp.stage, settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(asdict(cp), option=orjson.OPT_INDENT_2))
    os.replace(tmp, p)


def reset(stage: str, settings: Settings) -> None:
    p = _path(stage, settings)
    if p.exists():
        p.unlink()
