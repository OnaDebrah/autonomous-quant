"""Chapter 8's simulations. Needs numpy and scipy.

1. Choosing the perpetual with the highest trailing funding: what it pays afterwards.
2. Annualizing funding by an assumed three payments a day when instruments settle on different schedules.
3. Missing payments: at random, and concentrated where funding is highest.
4. The chance that the short leg of a cash-and-carry position touches its liquidation price.
Run: python book/ch08_crypto.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261119
PER_DAY = 3  # funding settlements a day on the common schedule
MEAN_RATE = 0.0001  # per settlement: 10.95% a year
RATE_SD = 0.0003
PHI = 0.98  # persistence from one settlement to the next


def ar_rates(rng, paths: int, steps: int) -> np.ndarray:
    x = np.empty((paths, steps))
    x[:, 0] = rng.standard_normal(paths)
    shocks = rng.standard_normal((paths, steps)) * math.sqrt(1 - PHI**2)
    for t in range(1, steps):
        x[:, t] = PHI * x[:, t - 1] + shocks[:, t]
    return MEAN_RATE + RATE_SD * x


def selection(rng) -> dict:
    """Thirty perpetuals; each week take those whose trailing 7-day funding is 10 points above an assumed 4% cash yield."""
    runs, coins, week, month = 4_000, 30, 7 * PER_DAY, 30 * PER_DAY
    rates = ar_rates(rng, runs * coins, week + month).reshape(runs, coins, -1)
    trailing = rates[:, :, :week].sum(axis=2) * 365 / 7
    following = rates[:, :, week:].sum(axis=2) * 365 / 30
    chosen = trailing >= 0.14
    top = trailing.argmax(axis=1)
    rows = np.arange(runs)
    slope = np.polyfit(trailing.ravel(), following.ravel(), 1)[0]
    return {
        "mean_apr_all": float(trailing.mean()),
        "share_chosen": float(chosen.mean()),
        "chosen_trailing_apr": float(trailing[chosen].mean()),
        "chosen_following_apr": float(following[chosen].mean()),
        "top_of_30_trailing_apr": float(trailing[rows, top].mean()),
        "top_of_30_following_apr": float(following[rows, top].mean()),
        "regression_slope_following_on_trailing": float(slope),
        "chosen_following_below_cash_share": float((following[chosen] < 0.04).mean()),
    }


def cadence(rng) -> dict:
    """Mean rate per settlement scales with the interval, so every schedule pays the same per year."""
    out = {}
    for per_day in (3, 6, 24):
        rates = (MEAN_RATE * 3 / per_day) + (
            RATE_SD * 3 / per_day
        ) * rng.standard_normal((20_000, 7 * per_day))
        true_apr = rates.sum(axis=1) * 365 / 7
        assumed = rates.mean(axis=1) * 3 * 365  # "mean rate x 3 x 365"
        out[str(per_day)] = {
            "true_apr": float(true_apr.mean()),
            "three_a_day_apr": float(assumed.mean()),
            "ratio": float(assumed.mean() / true_apr.mean()),
        }
    return out


def missing(rng) -> dict:
    paths, steps = 40_000, 7 * PER_DAY
    rates = ar_rates(rng, paths, steps)
    true = rates.sum(axis=1) * 365 / 7
    random_gap = rng.random((paths, steps)) < 0.10
    # Gaps where funding is extreme: the top tenth of each path's settlements by absolute rate.
    cut = np.quantile(np.abs(rates - MEAN_RATE), 0.90, axis=1, keepdims=True)
    stress_gap = np.abs(rates - MEAN_RATE) >= cut
    high = true > np.quantile(true, 0.9)
    out = {
        "true_apr": float(true.mean()),
        "true_apr_top_decile": float(true[high].mean()),
    }
    for name, gap in (("random", random_gap), ("at_extremes", stress_gap)):
        kept = np.where(gap, 0.0, rates)
        summed = kept.sum(axis=1) * 365 / 7  # missing treated as zero
        averaged = np.where(gap, np.nan, rates)
        scaled = (
            np.nanmean(averaged, axis=1) * PER_DAY * 365
        )  # missing ignored, mean scaled up
        out[name] = {
            "sum_as_if_complete": float(summed.mean()),
            "sum_as_if_complete_top_decile": float(summed[high].mean()),
            "mean_of_observed_scaled": float(scaled.mean()),
            "mean_of_observed_scaled_top_decile": float(scaled[high].mean()),
        }
    return out


def liquidation(rng) -> dict:
    out = {}
    for leverage, vol, days in (
        (2, 0.7, 30),
        (3, 0.7, 30),
        (5, 0.7, 30),
        (3, 0.7, 7),
        (3, 1.0, 30),
    ):
        maintenance = 0.01
        up = (
            1 / leverage - maintenance
        )  # adverse rise that exhausts the short leg's margin
        b = math.log(1 + up)
        sd = vol * math.sqrt(days / 365)
        steps = days * 24
        r = rng.normal(0, vol / math.sqrt(365 * 24), (40_000, steps))
        touched = np.cumsum(r, axis=1).max(axis=1) >= b
        out[f"{leverage}x_{vol}_{days}d"] = {
            "adverse_move": up,
            "theory": float(2 * norm.cdf(-b / sd)),
            "simulated_hourly": float(touched.mean()),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "selection": selection(rng),
        "cadence": cadence(rng),
        "missing": missing(rng),
        "liquidation": liquidation(rng),
    }
    Path(__file__).with_name("ch08_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    def r(x):
        return (
            {k: r(v) for k, v in x.items()}
            if isinstance(x, dict)
            else (round(x, 4) if isinstance(x, float) else x)
        )

    print(json.dumps(r(out), indent=1))


if __name__ == "__main__":
    main()
