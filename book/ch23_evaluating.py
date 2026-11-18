"""Chapter 23's calculations. Needs numpy and scipy.

1. What a run of rejections can and cannot exclude about the share of ideas that are real.
2. The power of a replay through real history to separate two policies.
3. Estimating the share of promoted strategies that are real from their forward rewards.
4. Selection decay to expect from the best of a few settings.
5. Several holdouts clearing a level by chance.
Run: python book/ch23_evaluating.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

SEED = 20261204
EULER = 0.5772156649015329


def m(n: int) -> float:
    return (
        0.0
        if n <= 1
        else (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def nulls() -> dict:
    out = {}
    for tested in (20, 99, 124, 500):
        row = {}
        for power in (0.02, 0.22, 0.45):
            row[f"power_{power}"] = {
                "upper_bound_on_share_real_95": min(
                    1.0, -math.log(0.05) / (tested * power)
                ),
                "chance_of_no_pass_if_5pct_real": float((1 - 0.05 * power) ** tested),
                "chance_of_no_pass_if_20pct_real": float((1 - 0.20 * power) ** tested),
            }
        out[str(tested)] = row
    return out


def replay_power() -> dict:
    out = {}
    for years in (7.5, 15, 30):
        for difference in (0.02, 0.05):
            for tracking in (0.05, 0.10):
                t = difference * math.sqrt(years) / tracking
                out[f"years_{years}_difference_{difference}_tracking_{tracking}"] = {
                    "expected_t": t,
                    "power_at_5pct_one_sided": float(norm.sf(1.645 - t)),
                }
    return out


def share_real(rng) -> dict:
    """Promoted strategies: a share is real with annual Sharpe 1.5; the rest have none. Sixty-day rewards."""
    shift = 1.5 * math.sqrt(60 / 365)
    gain = float(norm.cdf(shift / math.sqrt(2)) - 0.5)
    out = {"mean_reward_gain_if_real": gain}
    for n in (10, 40, 170, 700):
        for share in (0.2, 0.6):
            runs = 20_000
            real = rng.random((runs, n)) < share
            r = norm.cdf(rng.standard_normal((runs, n)) + shift * real)
            estimate = (r.mean(axis=1) - 0.5) / gain
            out[f"n_{n}_share_{share}"] = {
                "mean_estimate": float(estimate.mean()),
                "sd_of_estimate": float(estimate.std()),
                "interval_90": [
                    float(np.percentile(estimate, 5)),
                    float(np.percentile(estimate, 95)),
                ],
            }
    return out


def decay() -> dict:
    out, annual = {}, math.sqrt(252 / 504)
    for settings in (5, 12, 30):
        out[str(settings)] = {
            "expected_best_sharpe_with_no_skill": m(settings) * annual,
            "expected_decay_with_no_skill": m(settings) * annual,
        }
    return out


def by_chance() -> dict:
    return {
        f"at_least_{k}_of_22_at_5pct": float(binom.sf(k - 1, 22, 0.05))
        for k in (1, 2, 3)
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "nulls": nulls(),
        "replay_power": replay_power(),
        "share_real": share_real(rng),
        "decay": decay(),
        "holdouts_by_chance": by_chance(),
    }
    Path(__file__).with_name("ch23_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    def r(x):
        if isinstance(x, dict):
            return {k: r(v) for k, v in x.items()}
        if isinstance(x, list):
            return [r(v) for v in x]
        return round(x, 3) if isinstance(x, float) else x

    print(json.dumps(r(out)).replace("}, ", "},\n"))


if __name__ == "__main__":
    main()
