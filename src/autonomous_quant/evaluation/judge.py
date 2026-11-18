"""The judge (Chapters 13 to 15): the deflated rule, a contract that fails closed, and the sign-flip test.

Sharpe ratios are annual. With Y years of daily data a try's standardized estimate is ``SR * sqrt(Y)``, which under
no skill is approximately standard normal (Proposition 13.4). The best of N such tries is expected to reach ``m_N``
(Equation 13.1), so the deflated rule at level alpha passes the best try only if it exceeds ``m_N + z_(1 - alpha)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew

from ..errors import ConfigurationError, DataError

log = logging.getLogger(__name__)

GAMMA = 0.5772156649015329  # Euler-Mascheroni


def expected_best(n: int) -> float:
    """m_N: the expected best of n independent standard normals (Equation 13.1)."""
    if n <= 1:
        return 0.0
    return (1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * math.e))


def deflated_threshold(n: int, alpha: float = 0.05) -> float:
    """The cutoff c for the best of n tries, in standardized units."""
    if n < 1:
        raise ConfigurationError("a family has at least one try")
    if not 0 < alpha < 1:
        raise ConfigurationError("alpha must be between 0 and 1")
    return expected_best(n) + norm.ppf(1 - alpha)


def probabilistic_sharpe(returns, benchmark: float = 0.0) -> float:
    """PSR (the probabilistic Sharpe ratio of Chapter 4): Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)), per period.

    g3 is the skewness and g4 the (non-excess) kurtosis of the returns. It is an approximate one-sided confidence
    score, not a posterior probability. With fewer than three returns, or none that vary, it is not defined.
    """
    x = np.asarray(returns, dtype=float)
    if x.size < 3 or not np.isfinite(x).all():
        raise DataError("PSR needs at least three finite returns")
    sd = x.std(ddof=1)
    if sd == 0:
        raise DataError("PSR is not defined for returns that do not vary")
    sr = x.mean() / sd
    g3, g4 = float(skew(x)), float(kurtosis(x, fisher=False))
    denominator = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    if denominator <= 0:
        raise DataError("PSR's variance term is not positive for these returns")
    return float(norm.cdf((sr - benchmark) * math.sqrt(x.size - 1) / math.sqrt(denominator)))


def deflated_rule(sharpes: Sequence[float], years: float, family_size: int | None = None, alpha: float = 0.05) -> dict:
    """Pass the best try only if its standardized estimate clears the deflated bar for the whole family.

    ``family_size`` is the ledger's count of tries started, which may exceed the number of results supplied (failed
    tries count). Using fewer would lower the bar.
    """
    if len(sharpes) == 0:
        raise DataError("no results to judge: every try in the family failed")
    if not np.isfinite(np.asarray(sharpes, dtype=float)).all():
        raise DataError("Sharpe ratios must be finite")
    if not years > 0:
        raise ConfigurationError("years must be positive")
    n = family_size if family_size is not None else len(sharpes)
    if n < len(sharpes):
        raise ConfigurationError("family_size cannot be smaller than the number of results")
    best = int(np.argmax(sharpes))
    z = float(sharpes[best]) * math.sqrt(years)
    c = deflated_threshold(n, alpha)
    return {"passed": z > c, "best": best, "standardized": z, "threshold": c, "family_size": n}


def sign_flip_test(returns: np.ndarray, draws: int = 999, seed: int = 0) -> dict:
    """Algorithm 14.1: is the best Sharpe ratio in a family larger than sign-flipped families produce?

    ``returns`` has one column per try. Each draw multiplies every try's return on a day by the same random sign,
    which keeps the correlation between tries and removes any edge. Ties count as exceedances.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        raise DataError("the sign-flip test needs a (days x tries) matrix with at least three days")
    if not np.isfinite(x).all():
        raise DataError("returns must be finite; align the tries on common dates first")
    if draws < 1:
        raise ConfigurationError("draws must be at least 1")
    rng = np.random.default_rng(seed)

    def best(m: np.ndarray) -> float:
        sd = m.std(axis=0, ddof=1)
        return float(np.max(np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)))

    observed = best(x)
    exceed = sum(best(x * rng.choice((-1.0, 1.0), size=(x.shape[0], 1))) >= observed for _ in range(draws))
    return {"best_per_period": observed, "p_value": (1 + exceed) / (draws + 1), "draws": draws}


@dataclass(frozen=True)
class Clause:
    """One clause of a contract. ``check`` returns True (passes), False (fails) or None (cannot be evaluated)."""

    name: str
    check: Callable[[Mapping[str, float]], bool | None]


def judge_contract(clauses: Sequence[Clause], evidence: Mapping[str, float]) -> dict:
    """A contract fails closed (Chapter 15): any clause that fails, or cannot be evaluated, rejects the candidate."""
    verdicts = {}
    for clause in clauses:
        try:
            verdicts[clause.name] = clause.check(evidence)
        except (KeyError, TypeError, ValueError) as exc:
            verdicts[clause.name] = None  # missing or malformed evidence is not a pass
            log.warning("contract clause %r cannot be evaluated (%s: %s): the candidate fails", clause.name, type(exc).__name__, exc)
    failed = [k for k, v in verdicts.items() if v is not True]
    log.debug("contract: %s", verdicts)
    return {"passed": not failed, "clauses": verdicts, "reasons": failed}


def default_contract(min_round_trips: int = 10, max_drawdown: float = 0.35) -> list[Clause]:
    return [
        Clause("deflated", lambda e: bool(e["deflated_passed"])),
        Clause("enough_trades", lambda e: e["round_trips"] >= min_round_trips),
        Clause("drawdown", lambda e: e["max_drawdown"] <= max_drawdown),
        Clause("beats_control", lambda e: e["sharpe"] > e["control_sharpe"]),
    ]
