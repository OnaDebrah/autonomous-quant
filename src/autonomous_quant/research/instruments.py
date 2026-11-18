"""Instruments: pure, versioned functions of the data available at a decision time (Chapters 5, 6 and 10).

* ``WindowScore`` (Chapter 5): each asset's log return over a window divided by the window's own volatility, so that
  assets of different volatility are comparable. Ranked, it is a scanner.
* ``VolatilityState`` (Chapter 6): each asset's recent volatility as a percentile of its own trailing history. It is
  the simple baseline the book recommends trying before any hidden-state model.

Each reads only rows up to ``asof``; ``validate`` (Chapter 10) gives an instrument a verdict from its own record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..errors import ConfigurationError, DataError
from ..interfaces import Reading

log = logging.getLogger(__name__)


def _closes(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise DataError("an instrument needs a non-empty frame of closes, one column per asset")
    if not prices.index.is_monotonic_increasing:
        raise DataError("closes must be in date order")
    p = prices.loc[:asof].dropna(how="all")
    if (p <= 0).any().any():
        raise DataError("closes must be positive")
    return p


@dataclass(frozen=True)
class WindowScore:
    name: str = "window_score"
    version: str = "1"
    window: int = 126

    def read(self, prices: pd.DataFrame, asof: pd.Timestamp) -> Reading:
        """``prices``: one column of closes per asset."""
        p = _closes(prices, asof)
        logret = np.log(p).diff().iloc[-self.window :]
        enough = logret.count() >= self.window
        score = logret.sum() / (logret.std(ddof=1) * np.sqrt(self.window))
        values = {a: float(score[a]) for a in p.columns if enough[a] and np.isfinite(score[a])}
        if len(values) < len(p.columns):
            log.debug("%s: no score for %s (too little history)", self.name, sorted(set(p.columns) - set(values)))
        return Reading(self.name, self.version, asof, values, {"window": self.window, "missing": sorted(set(p.columns) - set(values))})


@dataclass(frozen=True)
class VolatilityState:
    name: str = "volatility_state"
    version: str = "1"
    lookback: int = 20
    history: int = 252
    stormy_above: float = 0.8

    def read(self, prices: pd.DataFrame, asof: pd.Timestamp) -> Reading:
        p = _closes(prices, asof)
        vol = np.log(p).diff().rolling(self.lookback, min_periods=self.lookback).std()
        recent = vol.iloc[-self.history :]
        values = {}
        for a in p.columns:
            v = recent[a].dropna()
            if len(v) >= self.history // 2:
                values[a] = float((v <= v.iloc[-1]).mean())  # percentile of today's volatility in its own history
        stormy = sorted(a for a, q in values.items() if q > self.stormy_above)
        return Reading(self.name, self.version, asof, values, {"stormy": stormy, "market_stormy": float(np.mean(list(values.values()) or [0])) > self.stormy_above})


def validate(instrument, prices: pd.DataFrame, dates: list[pd.Timestamp], horizon: int = 21, min_periods: int = 24) -> dict:
    """A verdict for a cross-sectional score (Chapter 10, Table 10.2): does it rank the next ``horizon`` days' returns?

    Uses non-overlapping periods only, so the periods are close to independent. The baseline is no information (a
    rank correlation of zero). Below ``min_periods`` the verdict is 'insufficient data', whatever the numbers say.
    """
    if horizon < 1 or min_periods < 2:
        raise ConfigurationError("need horizon >= 1 and min_periods >= 2")
    closes = prices.sort_index()
    ics, last = [], None
    for d in dates:
        if last is not None and closes.index.get_indexer([d], method="pad")[0] - closes.index.get_indexer([last], method="pad")[0] < horizon:
            continue  # overlapping with the previous period
        i = closes.index.get_indexer([d], method="pad")[0]
        if i < 0 or i + horizon >= len(closes):
            continue
        reading = instrument.read(closes, closes.index[i])
        assets = list(reading.values)
        if len(assets) < 3:
            continue
        future = np.log(closes[assets].iloc[i + horizon] / closes[assets].iloc[i])
        ic = spearmanr([reading.values[a] for a in assets], future.to_numpy()).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
            last = d
    n = len(ics)
    if n < min_periods:
        log.info("%s v%s: insufficient data (%d of %d periods)", instrument.name, instrument.version, n, min_periods)
        return {"verdict": "insufficient data", "periods": n, "min_periods": min_periods}
    mean, sd = float(np.mean(ics)), float(np.std(ics, ddof=1))
    t = mean / sd * np.sqrt(n) if sd > 0 else 0.0
    verdict = "skill" if t > 2 else "wrong way" if t < -2 else "no skill shown"
    log.info("%s v%s: %s (mean IC %.3f, t %.2f over %d periods)", instrument.name, instrument.version, verdict, mean, t, n)
    return {"verdict": verdict, "periods": n, "mean_ic": mean, "t": float(t), "instrument": instrument.name, "version": instrument.version}
