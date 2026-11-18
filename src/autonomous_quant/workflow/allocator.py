"""The allocator (Chapter 17, Algorithm 17.1): a seeded Thompson bandit over sources, with delayed rewards.

Each source has a Beta posterior over its mean reward. A study's slot goes to the source with the highest draw from
its posterior. The draws are seeded and each choice is recorded with its probability, estimated by repeated draws,
so that the allocation can be replayed and evaluated later (inverse-propensity weighting needs the probabilities).

Forward rewards arrive late: a study's reward is known only when its forward test settles. A rejected study is known
at once, and with ``rejections_as_zero`` it counts as a reward of zero immediately, which is most of the information
the allocator can have early (Chapter 17, on delayed and censored rewards).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass

import numpy as np

from ..errors import ConfigurationError, DataError

log = logging.getLogger(__name__)


@dataclass
class Allocator:
    db: sqlite3.Connection
    sources: tuple[str, ...]
    rejections_as_zero: bool = True
    seed: int = 0

    def __post_init__(self):
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ConfigurationError("the allocator needs distinct sources")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS posteriors (source TEXT PRIMARY KEY, a REAL NOT NULL, b REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS allocations (day TEXT, slot INTEGER, source TEXT, probability REAL, posterior TEXT NOT NULL, PRIMARY KEY (day, slot));
            CREATE TABLE IF NOT EXISTS rewards (key TEXT PRIMARY KEY, source TEXT, reward REAL, kind TEXT);
            """
        )
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO posteriors VALUES (?, 1, 1)", [(s,) for s in self.sources])

    def posterior(self) -> dict[str, tuple[float, float]]:
        return {s: (a, b) for s, a, b in self.db.execute("SELECT source, a, b FROM posteriors")}

    def choose(self, day: str, slots: int, available: set[str]) -> list[str]:
        """Pick a source for each slot. Idempotent: a day's choices are recorded once and replayed after a restart."""
        if slots < 0:
            raise ConfigurationError("slots must be non-negative")
        done = self.db.execute("SELECT source FROM allocations WHERE day = ? ORDER BY slot", (day,)).fetchall()
        if done:
            log.debug("allocation for %s already made; replaying it", day)
            return [r[0] for r in done]
        # A stable seed per day (not Python's hash(), which changes between processes), so a restart replays.
        rng = np.random.default_rng([self.seed, int(hashlib.sha256(day.encode()).hexdigest()[:8], 16)])
        post = {s: ab for s, ab in self.posterior().items() if s in available}
        if not post or slots == 0:
            log.info("%s: no study slots allocated (%d sources available)", day, len(post))
            return []
        names = sorted(post)
        a = np.array([post[s][0] for s in names])
        b = np.array([post[s][1] for s in names])
        probability = np.bincount(rng.beta(a, b, (2000, len(names))).argmax(axis=1), minlength=len(names)) / 2000
        picks = [names[int(rng.beta(a, b).argmax())] for _ in range(slots)]
        seen = json.dumps({s: list(post[s]) for s in names}, sort_keys=True)  # the posterior this decision saw
        with self.db:
            self.db.executemany("INSERT INTO allocations VALUES (?, ?, ?, ?, ?)",
                                [(day, i, s, float(probability[names.index(s)]), seen) for i, s in enumerate(picks)])
        log.info("%s: allocated %s (selection probabilities %s)", day, picks, dict(zip(names, np.round(probability, 3).tolist())))
        return picks

    def reward(self, key: str, source: str, value: float, kind: str = "forward") -> None:
        """Add a reward in [0, 1] once per key: the posterior gains ``value`` successes and ``1 - value`` failures."""
        if not 0.0 <= value <= 1.0:
            raise DataError(f"a reward is a number in [0, 1], got {value!r}")
        with self.db:
            if source not in self.sources:
                raise ConfigurationError(f"unknown source {source!r}")
            if self.db.execute("INSERT OR IGNORE INTO rewards VALUES (?, ?, ?, ?)", (key, source, value, kind)).rowcount:
                self.db.execute("UPDATE posteriors SET a = a + ?, b = b + ? WHERE source = ?", (value, 1.0 - value, source))
                log.info("reward %.3f (%s) to %s for %s", value, kind, source, key)
            else:
                log.debug("reward for %s already counted", key)

    def rejected(self, key: str, source: str) -> None:
        if self.rejections_as_zero:
            self.reward(key, source, 0.0, kind="rejection")
