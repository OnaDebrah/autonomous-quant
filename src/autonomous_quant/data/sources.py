"""Data sources. Each returns observations stamped with when they became available, or None when that is unknown.

* ``SimulatedSource``: a market of several assets whose drift switches between hidden up, flat and down regimes
  (the regime market of Chapters 6 and 20). The open of day s is available at the open, the close at the close. It is
  seeded, so a run can be reproduced exactly.
* ``YahooSource``: daily bars from Yahoo Finance, an optional extra (``pip install autonomous-quant[yahoo]``). A
  history downloaded today says when we received it, not when each value was first knowable, and adjusted prices are
  restated after every split and dividend. Its records therefore carry no availability time, and the store refuses
  to read them "as of" a past date unless the caller states the assumption that each was known at its observation
  time. That refusal is the lesson of Chapter 3.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..errors import ConfigurationError, DataError, SourceError
from ..interfaces import Observation

log = logging.getLogger(__name__)

CLOSE_HOUR, OPEN_HOUR = 16, 9  # observation and availability clocks, in exchange time


def _at(day: pd.Timestamp, hour: int) -> pd.Timestamp:
    return pd.Timestamp(day.date()) + pd.Timedelta(hours=hour)


def _check_window(symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> None:
    if not symbols or any(not isinstance(s, str) or not s for s in symbols):
        raise ConfigurationError("symbols must be a non-empty list of non-empty strings")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ConfigurationError(f"start {start} is after end {end}")


@dataclass
class SimulatedSource:
    """Daily open and close prices under hidden drift regimes.

    ``drift`` is the daily drift in the up regime (minus it in the down regime); the daily volatility is ``vol``
    whatever the regime; ``stay`` is the probability that a regime lasts another day; ``overnight_share`` is the share
    of each day's variance that arrives between the close and the next open.
    """

    assets: int = 5
    vol: float = 0.012
    drift: float = 0.0
    stay: float = 39 / 40
    overnight_share: float = 0.3
    seed: int = 0
    name: str = "simulated"

    def __post_init__(self):
        if self.assets < 1:
            raise ConfigurationError("assets must be at least 1")
        if not self.vol > 0 or not 0 < self.stay < 1 or not 0 < self.overnight_share < 1:
            raise ConfigurationError("need vol > 0, 0 < stay < 1 and 0 < overnight_share < 1")
        if self.drift**2 * (2 / 3) >= self.vol**2:
            raise ConfigurationError("drift is too large for the volatility: the regime means alone would exceed it")

    def returns(self, days: int) -> tuple[np.ndarray, np.ndarray]:
        """Overnight and intraday log returns, each of shape (days, assets)."""
        rng = np.random.default_rng(self.seed)
        state = rng.integers(0, 3, self.assets)
        noise = math.sqrt(self.vol**2 - (2 / 3) * self.drift**2)  # total daily variance stays vol**2
        w = self.overnight_share
        overnight, intraday = np.empty((days, self.assets)), np.empty((days, self.assets))
        for t in range(days):
            move = rng.random(self.assets) >= self.stay
            state = np.where(move, (state + rng.integers(1, 3, self.assets)) % 3, state)
            mean = self.drift * np.array([1.0, 0.0, -1.0])[state]
            overnight[t] = w * mean + noise * math.sqrt(w) * rng.standard_normal(self.assets)
            intraday[t] = (1 - w) * mean + noise * math.sqrt(1 - w) * rng.standard_normal(self.assets)
        return overnight, intraday

    def observations(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> list[Observation]:
        _check_window(symbols, start, end)
        if len(symbols) > self.assets:
            raise ConfigurationError(f"{len(symbols)} symbols requested from a market of {self.assets} assets")
        days = pd.bdate_range(start, end)
        overnight, intraday = self.returns(len(days))
        out: list[Observation] = []
        for k, symbol in enumerate(symbols):
            close = 100.0
            for t, day in enumerate(days):
                open_ = close * math.exp(overnight[t, k])
                close = open_ * math.exp(intraday[t, k])
                out.append(Observation(f"{symbol}:open", _at(day, OPEN_HOUR), open_, _at(day, OPEN_HOUR)))
                out.append(Observation(f"{symbol}:close", _at(day, CLOSE_HOUR), close, _at(day, CLOSE_HOUR)))
        log.info("simulated %d days for %d assets (seed %d)", len(days), len(symbols), self.seed)
        return out


@dataclass
class YahooSource:
    """Daily bars from Yahoo Finance. Records carry no availability time (see the module's docstring)."""

    name: str = "yahoo"
    adjusted: bool = True
    attempts: int = 3
    backoff_seconds: float = 2.0

    def _download(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ConfigurationError("YahooSource needs the optional extra: pip install 'autonomous-quant[yahoo]'") from exc
        last: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                frame = yf.download(list(symbols), start=start, end=end + pd.Timedelta(days=1), auto_adjust=self.adjusted,
                                    progress=False, group_by="ticker", threads=False)
                if frame is not None and not frame.empty:
                    return frame
                last = SourceError("Yahoo Finance returned no data")
            except Exception as exc:  # network and provider errors: retried, then reported
                last = exc
            log.warning("Yahoo download attempt %d of %d failed: %s", attempt, self.attempts, last)
            if attempt < self.attempts:
                time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
        raise SourceError(f"Yahoo Finance gave no data for {list(symbols)} from {start.date()} to {end.date()} "
                          f"after {self.attempts} attempts: {last}")

    def observations(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> list[Observation]:
        _check_window(symbols, start, end)
        frame = self._download(symbols, pd.Timestamp(start), pd.Timestamp(end))
        out: list[Observation] = []
        for symbol in symbols:
            try:
                bars = frame[symbol] if isinstance(frame.columns, pd.MultiIndex) else frame
            except KeyError:
                log.warning("Yahoo Finance returned nothing for %s", symbol)
                continue
            bars = bars.dropna(subset=["Open", "Close"])
            bad = (bars["Open"] <= 0) | (bars["Close"] <= 0)
            if bad.any():
                log.warning("%s: dropped %d bars with non-positive prices", symbol, int(bad.sum()))
                bars = bars[~bad]
            for day, row in bars.iterrows():
                out.append(Observation(f"{symbol}:open", _at(day, OPEN_HOUR), float(row["Open"]), None))
                out.append(Observation(f"{symbol}:close", _at(day, CLOSE_HOUR), float(row["Close"]), None))
        if not out:
            raise DataError(f"no usable bars for {list(symbols)}")
        log.info("fetched %d observations from Yahoo Finance for %s (availability unknown)", len(out), list(symbols))
        return out
