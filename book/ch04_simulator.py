"""Chapter 4's simulations. Needs only numpy.

1. Trading costs of a sign rule: measured turnover and Sharpe ratio against the closed forms.
2. A position below full size: rebalanced to a constant share every bar for free, against units and cash.
3. The Sharpe ratio's standard error when returns are skewed.
4. The expected maximum drawdown of a strategy with no skill.
Run: python book/ch04_simulator.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261114
PERIODS = 252
BARS = 504
DAILY_VOL = 0.012
COST = 0.0015  # of the value traded


def sharpe(returns: np.ndarray, axis=-1) -> np.ndarray:
    return returns.mean(axis=axis) / returns.std(axis=axis, ddof=1) * math.sqrt(PERIODS)


def costs(rng: np.random.Generator) -> dict:
    """Sign of an AR(1) signal with no skill, long or short one unit, paying COST on the value traded."""
    out = {}
    runs = 2_000
    for phi in (0.0, 0.5, 0.9, 0.99):
        shocks = rng.standard_normal((runs, BARS))
        signal = np.empty_like(shocks)
        signal[:, 0] = shocks[:, 0]
        for t in range(1, BARS):
            signal[:, t] = phi * signal[:, t - 1] + math.sqrt(1 - phi**2) * shocks[:, t]
        position = np.sign(signal)
        market = rng.normal(0.0, DAILY_VOL, (runs, BARS))
        traded = np.abs(np.diff(position, axis=1))
        net = position[:, :-1] * market[:, 1:] - COST * traded
        flip = math.acos(phi) / math.pi
        out[str(phi)] = {
            "theory_traded_per_day": 2 * flip,
            "traded_per_day": float(traded.mean()),
            "theory_sharpe": -COST * 2 * flip / DAILY_VOL * math.sqrt(PERIODS),
            "mean_sharpe": float(sharpe(net).mean()),
        }
    return out


def constant_share(rng: np.random.Generator) -> dict:
    """Half of equity in the asset over five years: rebalanced every bar at no cost, against units and cash."""
    runs, bars, weight = 5_000, 5 * PERIODS, 0.5
    r = rng.normal(0.0, DAILY_VOL, (runs, bars))
    rebalanced = np.prod(1 + weight * r, axis=1)
    units = 1 + weight * (np.prod(1 + r, axis=1) - 1)
    gap = (rebalanced - units) * 100
    log_gap = np.log(rebalanced) - weight * np.log(np.prod(1 + r, axis=1))
    return {
        "weight": weight,
        "years": 5,
        "mean_absolute_gap_pct": float(np.abs(gap).mean()),
        "p95_absolute_gap_pct": float(np.percentile(np.abs(gap), 95)),
        "mean_gap_pct": float(gap.mean()),
        "mean_log_gap_against_geometric": float(log_gap.mean()),
        "theory_log_gap_against_geometric": 0.5
        * weight
        * (1 - weight)
        * bars
        * DAILY_VOL**2,
    }


def skewed_sharpe(rng: np.random.Generator) -> dict:
    """Returns mu + s (1 - Exp(1)): skewness -2, kurtosis 9. True annual Sharpe ratio 1.5."""
    runs = 20_000
    true = 1.5 / math.sqrt(PERIODS)
    x = true * DAILY_VOL + DAILY_VOL * (1 - rng.exponential(1.0, (runs, BARS)))
    estimates = sharpe(x)
    general = math.sqrt(PERIODS * (1 - (-2) * true + (9 - 1) / 4 * true**2) / BARS)
    normal = math.sqrt(PERIODS * (1 + true**2 / 2) / BARS)
    return {
        "true_sharpe": 1.5,
        "skewness": -2,
        "kurtosis": 9,
        "simulated_sd": float(estimates.std(ddof=1)),
        "formula_sd": general,
        "normal_formula_sd": normal,
    }


def drawdown(rng: np.random.Generator) -> dict:
    runs = 20_000
    log_equity = np.cumsum(rng.normal(0.0, DAILY_VOL, (runs, BARS)), axis=1)
    peak = np.maximum.accumulate(
        np.concatenate([np.zeros((runs, 1)), log_equity], axis=1), axis=1
    )[:, 1:]
    worst = (peak - log_equity).max(axis=1)
    return {
        "theory_mean_log_drawdown": math.sqrt(math.pi / 2)
        * DAILY_VOL
        * math.sqrt(BARS),
        "mean_log_drawdown": float(worst.mean()),
        "share_beyond_25pct": float((1 - np.exp(-worst) > 0.25).mean()),
        "median_pct": float(np.median(1 - np.exp(-worst)) * 100),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "costs": costs(rng),
        "constant_share": constant_share(rng),
        "skewed_sharpe": skewed_sharpe(rng),
        "drawdown": drawdown(rng),
    }
    Path(__file__).with_name("ch04_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
