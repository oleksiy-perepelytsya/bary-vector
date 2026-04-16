"""A/B: bary_vec vs summary_vector retrieval lift."""

from __future__ import annotations

from lib.config import Settings
from lib.log import get_logger, setup_logging


def run() -> None:
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("eval.ab_summary")
    log.info("settings: db=%s", settings.mongo_db)
    raise NotImplementedError("eval/ab_summary: implement in follow-up PR")


if __name__ == "__main__":
    run()
