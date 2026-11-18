"""Predicates (Chapter 11): what a hypothesis claimed, fixed before its forward test, checked by code afterwards.

A predicate compares one metric from a fixed list with a threshold. A hypothesis registers three kinds:

* necessary conditions: without them the test says nothing (for example, enough observations);
* a mechanism test: what should hold if the idea is right;
* a failure condition: what, if true, counts against the idea.

The prediction is *supported* only if every necessary condition holds, the mechanism test holds and the failure
condition does not. If a necessary condition fails, the result is *not evaluable*: not a falsification. If the
conditions hold and the failure condition is true, it is *contradicted*; if neither holds, *not supported*. A contract
whose mechanism test and failure condition could both hold is inconsistent, and is refused before any test (``check``).
A predicate is not the judge: it fixes in advance what was claimed, and luck is the judge's business.
"""

from __future__ import annotations

import logging
import math
import operator
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from ..errors import ConfigurationError

log = logging.getLogger(__name__)

METRICS = ("observations", "strategy_return", "control_return", "incremental_return", "max_drawdown")
OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
KINDS = ("necessary", "mechanism", "failure")


@dataclass(frozen=True)
class Predicate:
    kind: str
    metric: str
    op: str
    threshold: float

    def __post_init__(self):
        if self.kind not in KINDS or self.metric not in METRICS or self.op not in OPERATORS:
            raise ConfigurationError(f"invalid predicate {self}: kind in {KINDS}, metric in {METRICS}, operator in {sorted(OPERATORS)}")
        if not math.isfinite(float(self.threshold)):
            raise ConfigurationError("a predicate's threshold must be finite")

    def holds(self, metrics: Mapping[str, float]) -> bool | None:
        """True or False, or None when the metric was not measured."""
        value = metrics.get(self.metric)
        if value is None or not math.isfinite(float(value)):
            return None
        return bool(OPERATORS[self.op](float(value), float(self.threshold)))

    def as_dict(self) -> dict:
        return asdict(self)


def standard(horizon: int) -> tuple[Predicate, ...]:
    """The predicates the generator registers: enough observations; the rule made money; the rule lost money.

    The mechanism test asks for an absolute gain over the window, and the failure condition is its mirror, so the two
    can never hold together. Beating a falling control while losing money is recorded in the metrics
    (``incremental_return``) but contradicts the prediction."""
    return (
        Predicate("necessary", "observations", ">=", round(0.9 * horizon)),
        Predicate("mechanism", "strategy_return", ">", 0.0),
        Predicate("failure", "strategy_return", "<=", 0.0),
    )


def _interval(p: Predicate) -> tuple[float, float, bool, bool]:
    """The values that satisfy ``p``, as (low, high, low included, high included)."""
    t = float(p.threshold)
    return {">": (t, math.inf, False, False), ">=": (t, math.inf, True, False),
            "<": (-math.inf, t, False, False), "<=": (-math.inf, t, False, True)}[p.op]


def check(predicates: Sequence[Predicate]) -> tuple[Predicate, ...]:
    """Refuse an inconsistent contract: exactly one mechanism test, at most one failure condition, and no value of the
    metrics at which both could hold. Predicates on different metrics could always both hold, so the failure
    condition must be on the mechanism's metric with a range disjoint from it."""
    predicates = tuple(predicates)
    kinds = {k: [p for p in predicates if p.kind == k] for k in KINDS}
    if len(kinds["mechanism"]) != 1 or len(kinds["failure"]) > 1:
        raise ConfigurationError("a prediction has exactly one mechanism test and at most one failure condition")
    if kinds["failure"]:
        m, f = kinds["mechanism"][0], kinds["failure"][0]
        if m.metric != f.metric:
            raise ConfigurationError(f"inconsistent contract: mechanism on {m.metric} and failure on {f.metric} could both hold")
        lo1, hi1, li1, hi1_in = _interval(m)
        lo2, hi2, li2, hi2_in = _interval(f)
        lo, hi = max(lo1, lo2), min(hi1, hi2)
        lo_in = li1 if lo1 > lo2 else li2 if lo2 > lo1 else li1 and li2
        hi_in = hi1_in if hi1 < hi2 else hi2_in if hi2 < hi1 else hi1_in and hi2_in
        if lo < hi or (lo == hi and lo_in and hi_in):
            raise ConfigurationError(f"inconsistent contract: {m.as_dict()} and {f.as_dict()} could both hold")
    return predicates


def evaluate(predicates: Sequence[Predicate], metrics: Mapping[str, float]) -> dict:
    kinds = {k: [p for p in check(predicates) if p.kind == k] for k in KINDS}
    necessary = [p.holds(metrics) for p in kinds["necessary"]]
    mechanism = kinds["mechanism"][0].holds(metrics)
    failure = kinds["failure"][0].holds(metrics) if kinds["failure"] else False
    if not all(n is True for n in necessary) or mechanism is None or failure is None:
        verdict = "not evaluable"
    elif mechanism and not failure:
        verdict = "supported"
    elif failure:
        verdict = "contradicted"
    else:
        verdict = "not supported"
    log.debug("predicates %s on %s: %s", [p.as_dict() for p in predicates], dict(metrics), verdict)
    return {"verdict": verdict, "necessary": necessary, "mechanism": mechanism, "failure": failure}


def from_dicts(items: Sequence[Mapping]) -> tuple[Predicate, ...]:
    return check(Predicate(**dict(i)) for i in items)
