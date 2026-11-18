"""The laboratory: one reference simulator (Chapter 4).

It keeps what a brokerage account keeps, units and cash. A rule reads closes up to and including day t and states a
target position; the order is filled at the open of day t+1 (never at the close the rule has already read). Costs are
charged on the value traded. Cash earns the cash rate, and results are measured in excess of it, so a rule that stays
in cash earns exactly zero. ``check_invariants`` states these properties as tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..errors import ConfigurationError, DataError
from ..interfaces import Outcome, Rule

log = logging.getLogger(__name__)


def check_prices(prices: pd.DataFrame) -> None:
    """The laboratory's input contract: open and close, positive and finite, on sorted unique days."""
    missing = {"open", "close"} - set(prices.columns)
    if missing:
        raise DataError(f"prices need columns 'open' and 'close'; missing {sorted(missing)}")
    if len(prices) < 2:
        raise DataError("prices need at least two days")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise DataError("prices must be on sorted, unique days")
    values = prices[["open", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataError("prices must be positive and finite; drop or repair missing days before the laboratory")


def check_positions(positions: pd.Series, index: pd.Index, rule: str) -> pd.Series:
    """A rule's output contract: one finite position in [-1, 1] for each day. A violation is a bug in the rule."""
    if not isinstance(positions, pd.Series) or not positions.index.equals(index):
        raise ConfigurationError(f"rule {rule!r} must return one position for each day of its input")
    x = positions.to_numpy(float)
    if not np.isfinite(x).all():
        raise ConfigurationError(f"rule {rule!r} returned non-finite positions; return 0 while warming up")
    if (np.abs(x) > 1 + 1e-12).any():
        raise ConfigurationError(f"rule {rule!r} returned positions outside [-1, 1]")
    return positions


@dataclass
class Simulator:
    cost_bps: float = 5.0  # per unit of value traded, commission and spread together
    capital: float = 100_000.0

    def __post_init__(self):
        if not 0 <= self.cost_bps < 10_000 or not self.capital > 0:
            raise ConfigurationError("need 0 <= cost_bps < 10000 and capital > 0")

    def account(self, target: pd.Series, opens: pd.Series, closes: pd.Series, cash_rate: pd.Series) -> pd.DataFrame:
        """Units, cash and equity on each day for a target position decided at each close."""
        days = closes.index
        w = target.reindex(days).to_numpy(float)
        rate = cash_rate.reindex(days).fillna(0.0).to_numpy(float)
        o, c = opens.reindex(days).to_numpy(float), closes.to_numpy(float)
        n = len(days)
        units, cash, equity, traded = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        u, cash_now = 0.0, self.capital
        for t in range(n):
            if t > 0:
                cash_now *= 1 + rate[t]  # interest over the night and the day
                equity_open = cash_now + u * o[t]
                new_u = w[t - 1] * equity_open / o[t]  # decided at yesterday's close, filled at today's open
                traded[t] = abs(new_u - u) * o[t]
                cash_now -= (new_u - u) * o[t] + traded[t] * self.cost_bps / 1e4
                u = new_u
            units[t], cash[t], equity[t] = u, cash_now, cash_now + u * c[t]
        return pd.DataFrame({"units": units, "cash": cash, "equity": equity, "traded": traded}, index=days)

    def run(self, rule: Rule, params: Mapping[str, float], prices: pd.DataFrame, cash_rate: pd.Series | None = None) -> Outcome:
        """``prices`` has columns ``open`` and ``close`` for one asset. The control is buy and hold."""
        check_prices(prices)
        rate = cash_rate if cash_rate is not None else pd.Series(0.0, index=prices.index)
        if not np.isfinite(rate.reindex(prices.index).fillna(0.0).to_numpy(float)).all():
            raise DataError("the cash rate must be finite")
        target = check_positions(rule.positions(prices["close"], params), prices.index, rule.name)
        book = self.account(target, prices["open"], prices["close"], rate)
        hold = self.account(pd.Series(1.0, index=prices.index), prices["open"], prices["close"], rate)
        def excess(b: pd.DataFrame) -> pd.Series:
            return (b["equity"].pct_change() - rate.reindex(b.index).fillna(0.0)).iloc[1:]

        if (book["equity"] <= 0).any():
            raise DataError(f"rule {rule.name!r} {dict(params)}: the account's equity fell to zero; the result is not a return series")
        held = (book["units"] != 0).astype(int)
        round_trips = int(((held.diff() == -1).sum()) + (held.iloc[-1] == 1))  # completed and still-open holding periods
        return Outcome(
            returns=excess(book),
            control_returns=excess(hold),
            round_trips=round_trips,
            turnover=float(book["traded"].sum() / self.capital),
            metrics={"final_equity": float(book["equity"].iloc[-1])},
        )


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    x = np.asarray(returns, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def truncation_test(rule: Rule, params: Mapping[str, float], closes: pd.Series, cutoffs: int = 5, seed: int = 0) -> bool:
    """Algorithm 2.2: the position decided at t must not change when data after t is removed."""
    full = rule.positions(closes, params)
    rng = np.random.default_rng(seed)
    for i in rng.integers(len(closes) // 4, len(closes) - 1, cutoffs):
        cut = rule.positions(closes.iloc[: i + 1], params)
        if not np.isclose(full.iloc[i], cut.iloc[-1], equal_nan=True):
            return False
    return True
