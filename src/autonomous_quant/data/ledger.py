"""The trial ledger (Chapters 13 and 21, Algorithm 13.1).

A try is registered, in its own committed transaction, before it is evaluated. Its family's count therefore includes
every try that was started, including those that later failed, and the count a judge uses cannot be lowered by
forgetting the tries that did not work. Registering the same configuration twice returns the existing try: a
configuration is counted once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..errors import ConfigurationError, IntegrityError

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id            INTEGER PRIMARY KEY,
    family        TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    specification TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('registered', 'completed', 'failed')),
    result        TEXT,
    UNIQUE (family, fingerprint)
);
"""


def fingerprint(specification: Mapping[str, Any]) -> str:
    """The identity of an experiment: a hash of its executed fields in canonical form (Chapter 11)."""
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path: str | Path | sqlite3.Connection = ":memory:"):
        # A connection can be shared, so that the ledger and the loop's outbox commit to the same database.
        self.db = path if isinstance(path, sqlite3.Connection) else sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)

    def register(self, family: str, specification: Mapping[str, Any]) -> int:
        """Record the try before it runs. Returns its id; a configuration already in the family keeps its id."""
        if not family:
            raise ConfigurationError("a try needs a family")
        fp = fingerprint(specification)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO trials (family, fingerprint, specification, registered_at, status) VALUES (?, ?, ?, ?, 'registered')",
                (family, fp, json.dumps(specification, sort_keys=True, default=str), pd.Timestamp.now().isoformat()),
            )
        trial = self.db.execute("SELECT id FROM trials WHERE family = ? AND fingerprint = ?", (family, fp)).fetchone()[0]
        log.debug("registered try %d in %s", trial, family)
        return trial

    def _finish(self, trial: int, status: str, result: Mapping[str, Any]) -> None:
        """Record a try's end once. Recording the same end again is a no-op (a resumed study repeats it); a different
        end is refused, because the same try on the same frozen inputs must give the same result."""
        payload = json.dumps(result, sort_keys=True, default=float)
        with self.db:
            row = self.db.execute("SELECT status, result FROM trials WHERE id = ?", (trial,)).fetchone()
            if row is None:
                raise IntegrityError(f"no try {trial} in the ledger")
            if row[0] == "registered":
                self.db.execute("UPDATE trials SET status = ?, result = ? WHERE id = ?", (status, payload, trial))
                return
        if (row[0], row[1]) != (status, payload):
            raise IntegrityError(f"try {trial} already ended as {row[0]} with a different result; the evaluation is not deterministic")
        log.debug("try %d already recorded as %s", trial, status)

    def complete(self, trial: int, result: Mapping[str, Any]) -> None:
        self._finish(trial, "completed", result)

    def fail(self, trial: int, error: str) -> None:
        log.warning("try %d failed: %s", trial, error)
        self._finish(trial, "failed", {"error": error})

    def family_size(self, family: str) -> int:
        """Every try started in the family, whatever happened to it: the N the judge must use."""
        return self.db.execute("SELECT COUNT(*) FROM trials WHERE family = ?", (family,)).fetchone()[0]

    def results(self, family: str) -> list[dict]:
        rows = self.db.execute("SELECT id, specification, status, result FROM trials WHERE family = ? ORDER BY id", (family,)).fetchall()
        return [{"id": r[0], "specification": json.loads(r[1]), "status": r[2], "result": json.loads(r[3]) if r[3] else None} for r in rows]
