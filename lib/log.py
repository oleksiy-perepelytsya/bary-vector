from __future__ import annotations

import logging

_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_FMT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
