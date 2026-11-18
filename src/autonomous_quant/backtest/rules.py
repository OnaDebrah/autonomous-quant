"""Toy, beginner rules for the loop to test. They are here so that the loop has something to search; no edge is claimed.

Each rule maps closes up to and including day t to a target position for day t+1, has a version, and states the
parameter region a study may search (Chapter 11). Both are long-or-flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from ..errors import ConfigurationError


def _int(params: Mapping[str, float], name: str, low: int) -> int:
    try:
        value = params[name]
    except KeyError:
        raise ConfigurationError(f"missing parameter {name!r}") from None
    if float(value) != int(value) or int(value) < low:
        raise ConfigurationError(f"{name} must be an integer of at least {low}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class TrendRule:
    """Long while the fast moving average is above the slow one."""

    name: str = "trend"
    version: str = "1"
    region: Mapping[str, Sequence[float]] = field(default_factory=lambda: {"fast": (5, 10, 20), "slow": (50, 100, 200)})

    def positions(self, prices: pd.Series, params: Mapping[str, float]) -> pd.Series:
        f, s = _int(params, "fast", 1), _int(params, "slow", 2)
        if f >= s:
            raise ConfigurationError(f"fast ({f}) must be shorter than slow ({s})")
        fast = prices.rolling(f, min_periods=f).mean()
        slow = prices.rolling(s, min_periods=s).mean()
        return (fast > slow).astype(float).where(slow.notna(), 0.0)


@dataclass(frozen=True)
class MeanReversionRule:
    """Long after the price falls ``entry`` standard deviations below its moving average; flat once it is back above."""

    name: str = "mean_reversion"
    version: str = "1"
    region: Mapping[str, Sequence[float]] = field(default_factory=lambda: {"lookback": (10, 20, 40), "entry": (1.0, 1.5, 2.0)})

    def positions(self, prices: pd.Series, params: Mapping[str, float]) -> pd.Series:
        n = _int(params, "lookback", 2)
        if "entry" not in params or not float(params["entry"]) > 0:
            raise ConfigurationError("entry must be a positive number of standard deviations")
        mean, sd = prices.rolling(n, min_periods=n).mean(), prices.rolling(n, min_periods=n).std()
        z = (prices - mean) / sd
        held, out = 0.0, []
        for value in z:  # stateful, but each position uses only data up to its own day
            if value != value:  # NaN: not enough history yet
                held = 0.0
            elif held == 0.0 and value < -params["entry"]:
                held = 1.0
            elif held == 1.0 and value > 0:
                held = 0.0
            out.append(held)
        return pd.Series(out, index=prices.index)


CATALOGUE = {r.name: r for r in (TrendRule(), MeanReversionRule())}
