"""Structured logging with daily rotation, 7-day retention."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def get_logger(name: str, log_dir: Path = Path("logs/"), level: str = "INFO") -> logging.Logger:
    """Return a configured logger with console + rotating file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler — daily, keep 7 days
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "pipeline.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
