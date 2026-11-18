"""From readings to hypotheses (Chapter 11, Algorithm 11.1).

A hypothesis states everything an experiment executes: the rule and its parameter region, the asset and interval, the
control, the predicted direction, the horizon, the predicates and the stage. Its identity is a fingerprint of those
fields only. The source (which reading suggested it) is provenance: two sources suggesting the same experiment
produce one hypothesis, tested once and counted once, with both sources kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..interfaces import Reading
from ..data.ledger import fingerprint
from .models import validated_match
from .predicates import Predicate, standard
from ..backtest.rules import CATALOGUE

log = logging.getLogger(__name__)
HORIZON = 63  # trading days of forward test


@dataclass(frozen=True)
class Hypothesis:
    rule: str
    version: str
    region: Mapping[str, tuple]
    asset: str
    control: str = "buy_and_hold"
    direction: str = "beats_control"
    horizon: int = HORIZON
    predicates: tuple[Predicate, ...] = field(default_factory=lambda: standard(HORIZON))
    stage: str = "study"
    interval: str = "1d"
    sources: tuple = field(default=(), compare=False)  # provenance, not identity

    def specification(self) -> dict:
        """The executed fields: what the identity is made of."""
        return {"rule": self.rule, "version": self.version, "region": {k: list(v) for k, v in self.region.items()}, "asset": self.asset,
                "control": self.control, "direction": self.direction, "horizon": self.horizon, "predicates": [p.as_dict() for p in self.predicates],
                "stage": self.stage, "interval": self.interval}

    @property
    def identity(self) -> str:
        return fingerprint(self.specification())


# The templates are the sources the allocator learns about (Chapter 17).
SOURCES = ("leaders", "laggards")


def generate(readings: Sequence[Reading], model, top: int = 2) -> list[tuple[str, Hypothesis]]:
    """Turn one day's readings into (source, hypothesis) pairs, merged by identity."""
    score = next((r for r in readings if r.instrument == "window_score"), None)
    if score is None or len(score.values) < 2 * top:
        log.info("no hypotheses: the window score covers %d assets, %d needed", len(score.values) if score else 0, 2 * top)
        return []
    rule_name = validated_match(model, readings, list(CATALOGUE))
    if rule_name is None:
        log.info("no hypotheses: no valid rule matched today's readings")
        return []
    rule = CATALOGUE[rule_name]
    ranked = sorted(score.values, key=score.values.get, reverse=True)
    picks = {"leaders": ranked[:top], "laggards": ranked[-top:]}
    merged: dict[str, tuple[str, Hypothesis]] = {}
    for source, assets in picks.items():
        for asset in assets:
            h = Hypothesis(rule.name, rule.version, {k: tuple(v) for k, v in rule.region.items()}, asset,
                           sources=(f"{source}:{score.instrument}@{score.version}",))
            if h.identity in merged:  # same experiment from another source: one hypothesis, both sources kept
                first_source, kept = merged[h.identity]
                merged[h.identity] = (first_source, Hypothesis(**{**kept.__dict__, "sources": kept.sources + h.sources}))
                log.info("merged %s on %s from %s into one hypothesis", rule.name, asset, kept.sources + h.sources)
            else:
                merged[h.identity] = (source, h)
    return list(merged.values())
