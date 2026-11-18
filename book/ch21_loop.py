"""Chapter 21's calculations and simulations. Needs numpy and scipy.

1. Retries: chance a step completes, and attempts spent.
2. Leases: how often a live worker's lease expires, by lease length.
3. A forward test registered late: the window backdated to the true start, or begun at registration.
4. Keeping an outage's tries in the ledger: what it does to the bar.
Run: python book/ch21_loop.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261202
EULER = 0.5772156649015329


def m(n: int) -> float:
    return (
        0.0
        if n <= 1
        else (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def retries() -> dict:
    out = {}
    for p in (0.05, 0.3, 0.6):
        out[str(p)] = {
            str(k): {"completes": 1 - p**k, "expected_attempts": (1 - p**k) / (1 - p)}
            for k in (1, 2, 4, 8)
        }
    return out


def leases(rng) -> dict:
    """Task durations are log-normal with a median of 10 minutes and a long tail."""
    d = rng.lognormal(math.log(10), 0.8, 2_000_000)
    return {
        "median_minutes": float(np.median(d)),
        "p99_minutes": float(np.percentile(d, 99)),
        **{
            f"lease_{L}_min_expires_on_a_live_worker": float((d > L).mean())
            for L in (15, 30, 60, 120)
        },
    }


def late_registration(rng) -> dict:
    """A strategy runs unregistered for 20 days of a 60-day window, with a 10% loss limit; then it is found and enrolled
    if it is still running."""
    runs, days, found, limit, vol = 400_000, 60, 20, 0.10, 0.20 / math.sqrt(252)
    out = {}
    for sharpe in (0.0, 1.0):
        r = rng.normal(sharpe * 0.20 / 252, vol, (runs, days))
        path = np.cumsum(r, axis=1)
        alive = path[:, :found].min(axis=1) > -limit
        full = path[:, -1]
        full = np.where(
            path.min(axis=1) <= -limit, -limit, full
        )  # stopped strategies lose the limit
        fresh = path[:, -1] - path[:, found - 1]
        out[str(sharpe)] = {
            "share_still_running_when_found": float(alive.mean()),
            "true_share_positive_over_window": float((full > 0).mean()),
            "backdated_among_those_found": float((full[alive] > 0).mean()),
            "true_mean_return": float(full.mean()),
            "backdated_mean_return": float(full[alive].mean()),
            "fresh_window_mean_daily_return_ratio_to_truth": float(
                fresh[alive].mean() / (days - found) / (sharpe * 0.20 / 252)
            )
            if sharpe
            else None,
            "fresh_window_mean_return": float(fresh[alive].mean()),
        }
    return out


def outage() -> dict:
    out, annual = {}, math.sqrt(252 / 504)
    for before, after in ((30, 61), (10, 41), (60, 91), (2, 33)):
        out[f"{before}_to_{after}"] = {
            "bar_before_annual": m(before) * annual,
            "bar_after_annual": m(after) * annual,
            "rise": (m(after) - m(before)) * annual,
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "retries": retries(),
        "leases": leases(rng),
        "late_registration": late_registration(rng),
        "outage_bar": outage(),
    }
    Path(__file__).with_name("ch21_results.json").write_text(
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
