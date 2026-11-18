"""Chapter 1's empirical section: a naive researcher on a driftless market. Needs only numpy and scipy.

Each try holds +1 or -1 at random each day on one market with no drift, so no try has any skill.
The researcher keeps the try with the highest Sharpe ratio. Run: python book/ch01_naive_researcher.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261111
BARS = 504  # two years of daily bars
PERIODS = 252
DAILY_VOL = 0.012
TRIES = (10, 100, 1_000, 10_000)
REPEATS = {10: 2_000, 100: 2_000, 1_000: 1_000, 10_000: 200}
EULER = 0.5772156649015329


def expected_max(n: int, variance: float) -> float:
    """Bailey and Lopez de Prado's approximation to the expected largest of n null Sharpe ratios."""
    return math.sqrt(variance) * (
        (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def best_sharpes(
    rng: np.random.Generator, tries: int, repeats: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best, above_one, above_two = np.empty(repeats), np.empty(repeats), np.empty(repeats)
    for k in range(repeats):
        market = rng.normal(0.0, DAILY_VOL, BARS)
        positions = rng.integers(0, 2, (tries, BARS), dtype=np.int8) * 2 - 1
        returns = positions * market
        sharpe = returns.mean(axis=1) / returns.std(axis=1, ddof=1) * math.sqrt(PERIODS)
        best[k], above_one[k], above_two[k] = (
            sharpe.max(),
            (sharpe > 1).sum(),
            (sharpe > 2).sum(),
        )
    return best, above_one, above_two


def false_discovery_share(
    threshold: float, true_sharpe: float, share_true: float, variance: float
) -> dict:
    sd = math.sqrt(variance)
    alpha = float(norm.sf(threshold / sd))
    power = float(norm.sf((threshold - true_sharpe) / sd))
    false = (1 - share_true) * alpha
    return {
        "alpha": alpha,
        "power": power,
        "false_discovery_share": false / (false + share_true * power),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    variance = PERIODS / BARS  # of an annualized null Sharpe ratio, to first order
    rows = []
    for tries in TRIES:
        best, one, two = best_sharpes(rng, tries, REPEATS[tries])
        rows.append(
            {
                "tries": tries,
                "repeats": REPEATS[tries],
                "theory_expected_best": expected_max(tries, variance),
                "mean_best": float(best.mean()),
                "p5_best": float(np.percentile(best, 5)),
                "p95_best": float(np.percentile(best, 95)),
                "share_of_runs_best_above_2": float((best > 2).mean()),
                "mean_tries_above_1": float(one.mean()),
                "mean_tries_above_2": float(two.mean()),
            }
        )
    out = {
        "seed": SEED,
        "bars": BARS,
        "null_sharpe_sd": math.sqrt(variance),
        "rows": rows,
        "worked_example": false_discovery_share(
            threshold=1.5, true_sharpe=1.0, share_true=0.01, variance=variance
        ),
    }
    path = Path(__file__).with_name("ch01_results.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
