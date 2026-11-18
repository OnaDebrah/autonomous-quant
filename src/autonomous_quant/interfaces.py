"""The interfaces that carry everything specific to a domain (Appendix C, "Extending it to another domain").

The store, ledger, judge, allocator, forward tests and loop do not depend on what is being researched. Markets enter
only through these: a source of stamped observations, instruments that turn them into readings, a laboratory that
turns a hypothesis and data into outcomes, and a reward that turns a forward window into a number in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class Observation:
    """One immutable record (Chapter 3): what is observed, when it refers to, its value, when it arrived, its revision.

    ``available`` is None when the source cannot say when the value became known. Such a record is kept, but an
    as-of read refuses it unless the caller explicitly assumes availability (Chapter 3, "unknown availability").
    """

    key: str  # what is observed, e.g. "SIM0:close"
    observed: pd.Timestamp  # s: the time the value refers to
    value: float
    available: pd.Timestamp | None  # a: when the system received it
    revision: int = 0  # j: successive reports of the same (key, observed)

    def __post_init__(self):
        from .errors import DataError

        if not isinstance(self.key, str) or not self.key:
            raise DataError(f"an observation needs a non-empty key, got {self.key!r}")
        if not isinstance(self.observed, pd.Timestamp) or (self.available is not None and not isinstance(self.available, pd.Timestamp)):
            raise DataError(f"{self.key}: observed and available must be pandas Timestamps")
        if not math.isfinite(float(self.value)):
            raise DataError(f"{self.key} at {self.observed}: value {self.value!r} is not finite; a missing value is no record, not a number")
        if self.available is not None and self.available < self.observed:
            raise DataError(f"{self.key}: available at {self.available}, before the time it refers to ({self.observed})")
        if int(self.revision) < 0:
            raise DataError(f"{self.key}: revision must be non-negative")


class DataSource(Protocol):
    """Produces observations. Every observation carries its availability time, or None if the source cannot know it."""

    name: str

    def observations(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> list[Observation]: ...


@dataclass(frozen=True)
class Reading:
    """An instrument's output at one decision time: what it read, its version, and the value."""

    instrument: str
    version: str
    asof: pd.Timestamp
    values: Mapping[str, float]  # e.g. a score per asset
    evidence: Mapping[str, Any] = field(default_factory=dict)


class Instrument(Protocol):
    """A pure, versioned function of the data available at a decision time (Chapters 5 and 10)."""

    name: str
    version: str

    def read(self, prices: pd.DataFrame, asof: pd.Timestamp) -> Reading: ...


@dataclass(frozen=True)
class Outcome:
    """What the laboratory measured for one hypothesis over one window."""

    returns: pd.Series  # the strategy's daily excess returns
    control_returns: pd.Series  # the control's, on the same days
    round_trips: int
    turnover: float
    metrics: Mapping[str, float] = field(default_factory=dict)


class Laboratory(Protocol):
    """Turns a rule, its parameters and data into a series of outcomes (Chapter 4)."""

    def run(self, rule: "Rule", params: Mapping[str, float], prices: pd.DataFrame, cash_rate: pd.Series | None = None) -> Outcome: ...


class Rule(Protocol):
    """A decision rule: maps prices up to and including each day to a target position in [-1, 1] for the next day."""

    name: str
    version: str
    region: Mapping[str, Sequence[float]]  # the parameter region a study may search (Chapter 11)

    def positions(self, prices: pd.Series, params: Mapping[str, float]) -> pd.Series: ...


class Reward(Protocol):
    """Turns a forward window into a number in [0, 1] (Chapter 22)."""

    def __call__(self, outcome: Outcome) -> float: ...


class ModelCaller(Protocol):
    """The three model calls of Chapter 18: match, tweak and write. The default implementation is rule-based."""

    def match(self, readings: Sequence[Reading], catalogue: Sequence[str]) -> str | None: ...

    def tweak(self, source: str, change: str) -> str | None: ...

    def write(self, specification: str) -> str | None: ...
