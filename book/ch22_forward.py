"""Chapter 22's calculations and simulations. Needs numpy and scipy.

1. What one forward window says: the reward's distribution with and without an edge, and the information in it.
2. An expected band centred on the backtest: how often a paper record falls below it, by how much of the backtest
   was real, and when volatility has changed.
3. Evidence that decays with age: how much the system can ever know at once.
4. Updating a Beta posterior with rewards between 0 and 1.
Run: python book/ch22_forward.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

SEED = 20261203
Z05 = float(norm.ppf(0.05))


def one_window() -> dict:
    out = {}
    for days in (60, 180, 365, 730):
        row = {}
        for sharpe in (0.0, 1.0, 1.5, 2.0):
            shift = sharpe * math.sqrt(days / 365)
            row[str(sharpe)] = {
                "mean_reward": float(
                    norm.cdf(shift / math.sqrt(2))
                ),  # E[Phi(Z)], Z ~ N(shift, 1)
                "reward_above_0.95": float(norm.sf(norm.ppf(0.95) - shift)),
                "information_nats": shift**2 / 2,
                "windows_for_20_to_1": (math.log(20) / (shift**2 / 2))
                if shift
                else None,
            }
        out[str(days)] = row
    return out


def band() -> dict:
    """Backtest annual Sharpe 2. The paper's true Sharpe is a fraction of it; realized volatility is a multiple of the backtest's."""
    out, backtest = {}, 2.0
    for days in (30, 90, 180, 365):
        row = {}
        for real_share in (1.0, 0.5, 0.0):
            for vol_ratio in (1.0, 1.5):
                gap = (
                    (1 - real_share) * backtest * math.sqrt(days / 365)
                )  # shortfall of the mean, in the band's standard errors
                row[f"real_{real_share}_vol_{vol_ratio}"] = float(
                    norm.cdf(
                        (Z05 - gap + (1 - 1 / vol_ratio) * 0) / vol_ratio
                        + (gap - gap / vol_ratio) * 0
                        - 0
                    )
                )
                # Paper mean = real_share * backtest mean; paper sd = vol_ratio * backtest sd. Standardize by the band's sd:
                mean_paper = (
                    real_share * backtest * math.sqrt(days / 365) * vol_ratio
                )  # a Sharpe ratio held at higher volatility earns more
                mean_band = backtest * math.sqrt(days / 365)
                row[f"real_{real_share}_vol_{vol_ratio}"] = float(
                    norm.cdf((mean_band + Z05 - mean_paper) / vol_ratio)
                )
        out[str(days)] = row
    return out


def decay() -> dict:
    out = {}
    for half_life in (90, 180, 365):
        for per_week in (0.25, 1.0, 4.0):
            n_eff = (per_week / 7) * half_life / math.log(2)
            out[f"half_life_{half_life}_outcomes_per_week_{per_week}"] = {
                "steady_state_effective_outcomes": n_eff,
                "posterior_sd_at_rate_0.5": math.sqrt(0.25 / (n_eff + 2)),
                "smallest_difference_resolved": 2
                * 1.96
                * math.sqrt(0.25 / (n_eff + 2))
                * math.sqrt(0.5),
            }
    return out


def fractional_beta(rng) -> dict:
    """Rewards are Phi(Z) with Z ~ N(shift, 1); the posterior adds r to successes and 1 - r to failures."""
    out, runs = {}, 20_000
    for shift, n in ((0.0, 10), (0.0, 40), (0.6, 10), (0.6, 40)):
        r = norm.cdf(shift + rng.standard_normal((runs, n)))
        s = r.sum(axis=1)
        a, b = 1 + s, 1 + n - s
        truth = float(norm.cdf(shift / math.sqrt(2)))
        lo, hi = beta_dist.ppf(0.05, a, b), beta_dist.ppf(0.95, a, b)
        out[f"shift_{shift}_n_{n}"] = {
            "true_mean_reward": truth,
            "interval_90_covers": float(((lo <= truth) & (truth <= hi)).mean()),
            "posterior_sd": float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))).mean()),
            "actual_sd_of_estimate": float((s / n).std()),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "one_window": one_window(),
        "band_share_below_p5": band(),
        "decay": decay(),
        "fractional_beta": fractional_beta(rng),
    }
    Path(__file__).with_name("ch22_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    def r(x):
        return (
            {k: r(v) for k, v in x.items()}
            if isinstance(x, dict)
            else (round(x, 4) if isinstance(x, float) else x)
        )

    print(json.dumps(r(out)).replace("}, ", "},\n"))


if __name__ == "__main__":
    main()
