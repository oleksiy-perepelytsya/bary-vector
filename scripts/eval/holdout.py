"""Generate 10% held-out synonym pairs → data/holdout.json."""

from __future__ import annotations

from lib.config import Settings
from lib.log import get_logger, setup_logging


def run() -> None:
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("eval.holdout")
    log.info("settings: db=%s", settings.mongo_db)
    raise NotImplementedError("eval/holdout: implement in follow-up PR")


if __name__ == "__main__":
    run()
