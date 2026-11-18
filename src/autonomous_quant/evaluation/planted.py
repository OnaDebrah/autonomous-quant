"""Planted truth (Chapter 12): worlds in which the answer is known, and the scorecard of a judge.

Each family has N tries of daily returns over Y years. With probability ``share_real`` one try in the family has a
true annual Sharpe ratio ``edge``; every other try has none. A judge passes one try of a family or none, and is scored
by the three rates of Chapter 12, with the family as the unit:

* false-pass rate: families in which the try passed has no edge, over all families. A false pass can happen in an
  edged family too, when the judge picks one of its null tries; such a family counts as a false pass and as a miss.
* power: edged families in which the edged try is passed, over edged families;
* false-discovery share: families in which the try passed has no edge, over families in which a try is passed.

Report the false-pass rate and power together, and the false-discovery share only with the mix that produced it.
A judge's quality cannot be read from real results, where the truth is unknown.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from ..errors import ConfigurationError
from .judge import deflated_threshold

PERIODS = 252
log = logging.getLogger(__name__)


@dataclass
class World:
    tries: int = 20
    years: float = 2.0
    edge: float = 1.5  # annual Sharpe ratio of the edged try
    share_real: float = 0.25
    daily_vol: float = 0.01
    seed: int = 0

    def __post_init__(self):
        if self.tries < 1 or not self.years > 0 or not 0 <= self.share_real <= 1 or not self.daily_vol > 0:
            raise ConfigurationError("need tries >= 1, years > 0, 0 <= share_real <= 1 and daily_vol > 0")
        if round(self.years * PERIODS) < 3:
            raise ConfigurationError("a family needs at least three days of returns")


Judge = Callable[[np.ndarray, World], tuple[bool, int]]  # (annual Sharpe of each try, world) -> (passed?, which)


def naive(sharpes: np.ndarray, world: World) -> tuple[bool, int]:
    """Pass the best try if its Sharpe ratio is above 1."""
    best = int(np.argmax(sharpes))
    return bool(sharpes[best] > 1.0), best


def single_test(sharpes: np.ndarray, world: World) -> tuple[bool, int]:
    """Pass the best try if it would pass a 5% test on its own, ignoring the search."""
    best = int(np.argmax(sharpes))
    return bool(sharpes[best] * math.sqrt(world.years) > norm.ppf(0.95)), best


def deflated(sharpes: np.ndarray, world: World) -> tuple[bool, int]:
    """Pass the best try only if it clears the bar for a family of this size (Chapter 13)."""
    best = int(np.argmax(sharpes))
    return bool(sharpes[best] * math.sqrt(world.years) > deflated_threshold(world.tries)), best


JUDGES: dict[str, Judge] = {"naive": naive, "single_test": single_test, "deflated": deflated}


def simulate(world: World, families: int) -> tuple[np.ndarray, np.ndarray]:
    """Annual Sharpe ratios of every try (families x tries), and which try is edged (-1 when none)."""
    rng = np.random.default_rng(world.seed)
    days = int(round(world.years * PERIODS))
    edged = np.where(rng.random(families) < world.share_real, rng.integers(0, world.tries, families), -1)
    sharpes = np.empty((families, world.tries))
    mean = world.edge / math.sqrt(PERIODS) * world.daily_vol
    for f in range(families):  # daily returns, so the estimate has its real sampling error
        r = rng.standard_normal((days, world.tries)) * world.daily_vol
        if edged[f] >= 0:
            r[:, edged[f]] += mean
        sharpes[f] = r.mean(axis=0) / r.std(axis=0, ddof=1) * math.sqrt(PERIODS)
    return sharpes, edged


def scorecard(world: World, families: int = 2000, judges: dict[str, Judge] | None = None) -> dict[str, dict]:
    if families < 1:
        raise ConfigurationError("families must be at least 1")
    sharpes, edged = simulate(world, families)
    real = edged >= 0
    out = {}
    for name, judge in (judges or JUDGES).items():
        decisions = [judge(s, world) for s in sharpes]
        passed = np.array([p for p, _ in decisions])
        passed_edge = np.array([p and b == e for (p, b), e in zip(decisions, edged)])
        false_pass = passed & ~passed_edge  # the try passed has no edge, in any family
        out[name] = {
            "false_pass_rate": float(false_pass.mean()),
            "power": float(passed_edge[real].mean()) if real.any() else float("nan"),
            "false_discovery_share": float(false_pass.sum() / passed.sum()) if passed.any() else float("nan"),
            "families": int(families),
            "edged_families": int(real.sum()),
        }
        log.info("planted truth, %s: false-pass rate %.4f, power %.3f over %d families", name, out[name]["false_pass_rate"], out[name]["power"], families)
    return out
