"""An append-only store of observations with an as-of read (Chapter 3, Algorithm 3.1).

Records are never updated or deleted: the database refuses. A revision is a new record with the same (key, observed)
and a later availability time. Reading "as of t" returns, for each (key, observed), the latest record that had arrived
by t, so the past viewed as of t cannot be changed by anything that arrives later.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..errors import DataError, UnknownAvailability
from ..interfaces import Observation

__all__ = ["Store", "UnknownAvailability"]

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    key        TEXT NOT NULL,
    observed   TEXT NOT NULL,
    value      REAL NOT NULL,
    available  TEXT,            -- NULL: the source could not say when the value became known
    revision   INTEGER NOT NULL,
    PRIMARY KEY (key, observed, revision)
);
CREATE TRIGGER IF NOT EXISTS observations_no_update BEFORE UPDATE ON observations
BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_delete BEFORE DELETE ON observations
BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
"""


def _iso(t: pd.Timestamp) -> str:
    """One text form for every timestamp, so that comparisons in SQL order correctly."""
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.isoformat()


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(_SCHEMA)

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def append(self, observations: Iterable[Observation]) -> int:
        """Add records in one transaction: all of them, or none if any is a duplicate of a stored record."""
        rows = [(o.key, _iso(o.observed), float(o.value), _iso(o.available) if o.available is not None else None, int(o.revision)) for o in observations]
        try:
            with self.db:
                self.db.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?)", rows)
        except sqlite3.IntegrityError as exc:
            raise DataError(f"a record with the same key, observation time and revision is already stored ({exc}); "
                            "a correction is a new revision, not a replacement") from exc
        unknown = sum(1 for r in rows if r[3] is None)
        log.info("stored %d observations (%d with unknown availability)", len(rows), unknown)
        return len(rows)

    def as_of(self, key: str, t: pd.Timestamp, *, assume_available_at_observation: bool = False) -> pd.Series:
        """The series for ``key`` as it stood at ``t``: for each observation time, the latest record available by t.

        Records with unknown availability are refused, unless the caller assumes each was available at its own
        observation time. That assumption is the caller's, and a result built on it cannot be shown to be adapted.
        """
        unknown = self.db.execute("SELECT COUNT(*) FROM observations WHERE key = ? AND available IS NULL", (key,)).fetchone()[0]
        if unknown and not assume_available_at_observation:
            raise UnknownAvailability(
                f"{unknown} records of {key!r} have no availability time; pass assume_available_at_observation=True to "
                "treat each as known at its observation time (an assumption, not a fact)"
            )
        if unknown:
            log.warning("%s: assuming %d records were available at their observation times", key, unknown)
        available = "COALESCE(available, observed)" if assume_available_at_observation else "available"
        rows = self.db.execute(
            f"""
            SELECT observed, value FROM (
                SELECT observed, value, ROW_NUMBER() OVER (
                    PARTITION BY observed ORDER BY {available} DESC, revision DESC) AS latest
                FROM observations WHERE key = ? AND {available} <= ?
            ) WHERE latest = 1 ORDER BY observed
            """,
            (key, _iso(t)),
        ).fetchall()
        if not rows:
            return pd.Series(dtype=float, name=key)
        index = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
        return pd.Series([r[1] for r in rows], index=index, name=key)

    def keys(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT key FROM observations ORDER BY key")]
