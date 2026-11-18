"""Logging. The library only creates loggers; the application (the CLI, or your program) decides where they go.

Every module logs to ``logging.getLogger(__name__)``, under the ``autonomous_quant`` namespace, and the package adds a
``NullHandler`` so that importing it prints nothing. Call ``configure`` from an application to see the messages:

* INFO: what the research loop decided and why: each tick, study, verdict, enrolment and settlement.
* WARNING: what it refused or assumed: dropped model answers, data gaps, assumed availability, invalidated tests.
* ERROR: a study or settlement that failed, with its traceback; the loop records it and continues.
* DEBUG: per-try detail.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

NAMESPACE = "autonomous_quant"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure(level: str | int = "INFO", file: str | Path | None = None) -> logging.Logger:
    """Send the package's log records to stderr, and to ``file`` if given. Safe to call more than once."""
    logger = logging.getLogger(NAMESPACE)
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")
    logger.setLevel(numeric)
    for handler in list(logger.handlers):
        if getattr(handler, "_aq", False):
            logger.removeHandler(handler)
            handler.close()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if file is not None:
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter(FORMAT))
        handler._aq = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger
