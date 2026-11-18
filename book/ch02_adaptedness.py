"""Chapter 2's empirical section. Needs only numpy.

1. Look-ahead through a full-sample statistic: "long below the mean price, short above it" on a driftless random walk,
   with the mean taken over the whole sample (not adapted) and over the past only (adapted).
2. Look-ahead through smoothing: a trend rule on a centred moving average (not adapted) and a trailing one (adapted).
3. Selection on the past: the best of N rules on the first two years, followed over the next two.
Run: python book/ch02_adaptedness.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261112
BARS = 504
PERIODS = 252
DAILY_VOL = 0.012
RUNS = 2_000
WINDOW = 21  # bars in the moving averages
TRIES = 1_000
SELECTION_RUNS = 1_000


def sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * math.sqrt(PERIODS)) if sd > 0 else 0.0


def held(decided: np.ndarray) -> np.ndarray:
    """A position decided at each close is held over the next bar."""
    return np.concatenate([[0.0], decided[:-1]])


def summary(values: list[float]) -> dict:
    v = np.array(values)
    return {
        "mean": float(v.mean()),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "share_above_1": float((v > 1).mean()),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED, "bars": BARS, "runs": RUNS, "window": WINDOW}
    rows = {
        "mean_full_sample": [],
        "mean_past_only": [],
        "trend_centred": [],
        "trend_trailing": [],
    }
    half = WINDOW // 2
    for _ in range(RUNS):
        r = rng.normal(0.0, DAILY_VOL, BARS)
        log_price = np.cumsum(r)
        past_mean = np.cumsum(log_price) / np.arange(1, BARS + 1)
        rows["mean_full_sample"].append(
            sharpe(held(np.sign(log_price.mean() - log_price)) * r)
        )
        rows["mean_past_only"].append(sharpe(held(np.sign(past_mean - log_price)) * r))
        padded = np.concatenate([np.full(WINDOW - 1, log_price[0]), log_price])
        trailing = np.convolve(padded, np.ones(WINDOW) / WINDOW, mode="valid")
        centred = np.concatenate(
            [trailing[half:], np.full(half, trailing[-1])]
        )  # the average of bars t-10 to t+10
        rows["trend_centred"].append(
            sharpe(held(np.sign(np.diff(centred, prepend=centred[0]))) * r)
        )
        rows["trend_trailing"].append(
            sharpe(held(np.sign(np.diff(trailing, prepend=trailing[0]))) * r)
        )
    out["look_ahead"] = {name: summary(values) for name, values in rows.items()}

    in_sample, forward = [], []
    for _ in range(SELECTION_RUNS):
        r = rng.normal(0.0, DAILY_VOL, 2 * BARS)
        positions = (
            rng.integers(0, 2, (TRIES, 2 * BARS), dtype=np.int8) * 2 - 1
        )  # each rule is fixed before any data
        returns = positions * r
        first = returns[:, :BARS]
        scores = first.mean(axis=1) / first.std(axis=1, ddof=1)
        best = int(scores.argmax())
        in_sample.append(sharpe(first[best]))
        forward.append(sharpe(returns[best, BARS:]))
    out["selection"] = {
        "tries": TRIES,
        "runs": SELECTION_RUNS,
        "in_sample": summary(in_sample),
        "forward": summary(forward),
    }
    out["selection"]["forward_standard_error"] = float(
        np.std(forward, ddof=1) / math.sqrt(SELECTION_RUNS)
    )

    Path(__file__).with_name("ch02_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
