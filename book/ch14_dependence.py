"""Chapter 14's simulations. Needs numpy and scipy.

1. The best of N equicorrelated null tries: its mean, its upper quantile, and what three rules let through.
2. Four judges on families of correlated tries, in four layouts: one idea or five, with and without an edge.
3. How many independent observations a set of correlated forward outcomes is worth.
Run: python book/ch14_dependence.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261125
EULER = 0.5772156649015329
Z95 = float(norm.ppf(0.95))
PERIODS, BARS, VOL = 252, 504, 0.012
GRID = np.linspace(-8, 8, 4001)
WEIGHT = norm.pdf(GRID) * (GRID[1] - GRID[0])


def m(n: float) -> float:
    return (
        0.0
        if n <= 1
        else (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def tail(c: float, n: int, rho: float) -> float:
    """P(max of n equicorrelated standard normals > c)."""
    if rho >= 1:
        return float(norm.sf(c))
    inner = norm.cdf((c - math.sqrt(rho) * GRID) / math.sqrt(1 - rho)) ** n
    return float(1 - (inner * WEIGHT).sum())


def quantile(n: int, rho: float, level: float = 0.95) -> float:
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid, n, rho) > 1 - level else (lo, mid)
    return (lo + hi) / 2


def correlated_maximum() -> dict:
    out = {}
    for n in (20, 100):
        for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
            eigen_n = n / (1 + (n - 1) * rho**2)
            linear_n = rho + (1 - rho) * n
            out[f"{n}_{rho}"] = {
                "expected_max": math.sqrt(1 - rho) * m(n),
                "threshold_family_5pct": quantile(n, rho),
                "false_pass_independent_bar": tail(
                    m(n) + Z95, n, rho
                ),  # bar from 1/T and N
                "false_pass_correlation_aware_bar": tail(
                    math.sqrt(1 - rho) * m(n) + Z95, n, rho
                ),  # bar from the spread with correlation kept
                "false_pass_linear_effective_n": tail(m(linear_n) + Z95, n, rho),
                "false_pass_eigen_effective_n": tail(m(eigen_n) + Z95, n, rho),
            }
    return out


def sharpe(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1) / x.std(axis=-1, ddof=1)


def judges(rng) -> dict:
    trials, runs, flips = 20, 1_500, 200
    mu = 1.5 / math.sqrt(PERIODS) * VOL
    v0 = 1 / BARS
    layouts = {
        "one_idea_no_edge": (1, 0),
        "one_idea_edge": (1, 1),
        "five_ideas_none_edged": (5, 0),
        "five_ideas_one_edged": (5, 1),
    }
    out = {}
    for rho in (0.3, 0.9):
        for name, (ideas, edged) in layouts.items():
            per = trials // ideas
            tally = {
                k: 0
                for k in (
                    "no_skill_spread",
                    "measured_spread",
                    "sign_flip_spread",
                    "sign_flip_max_5pct",
                    "sign_flip_max_1pct",
                )
            }
            bars = {"measured_spread": [], "sign_flip_spread": []}
            for _ in range(runs):
                common = rng.standard_normal((ideas, 1, BARS))
                own = rng.standard_normal((ideas, per, BARS))
                x = VOL * (math.sqrt(rho) * common + math.sqrt(1 - rho) * own)
                x[:edged] += mu
                x = x.reshape(trials, BARS)
                sr = sharpe(x)
                best = int(sr.argmax())
                real = best < edged * per if edged else False
                signs = rng.choice((-1.0, 1.0), (flips, 1, BARS))
                flipped = sharpe(x[None] * signs)  # flips x trials
                v_flip = float(flipped[:20].var(axis=1, ddof=1).mean())
                v_meas = float(sr.var(ddof=1))
                se = math.sqrt(1 / (BARS - 1))
                ok = (
                    real if edged else True
                )  # with an edge planted, count only passes of a real try
                tally["no_skill_spread"] += (
                    ok and (sr[best] - math.sqrt(v0) * m(trials)) / se > Z95
                )
                tally["measured_spread"] += (
                    ok and (sr[best] - math.sqrt(v_meas) * m(trials)) / se > Z95
                )
                tally["sign_flip_spread"] += (
                    ok and (sr[best] - math.sqrt(v_flip) * m(trials)) / se > Z95
                )
                p = (flipped.max(axis=1) >= sr[best]).mean()
                tally["sign_flip_max_5pct"] += ok and p < 0.05
                tally["sign_flip_max_1pct"] += ok and p < 0.01
                bars["measured_spread"].append(math.sqrt(v_meas / v0))
                bars["sign_flip_spread"].append(math.sqrt(v_flip / v0))
            out[f"{rho}_{name}"] = {
                **{k: v / runs for k, v in tally.items()},
                "measured_bar_ratio": float(np.mean(bars["measured_spread"])),
                "sign_flip_bar_ratio": float(np.mean(bars["sign_flip_spread"])),
                "theory_null_bar_ratio": math.sqrt(1 - rho) if ideas == 1 else None,
            }
    return out


def forward_evidence() -> dict:
    return {
        f"{k}_{rho}": k / (1 + (k - 1) * rho)
        for rho in (0.1, 0.25, 0.5, 0.9)
        for k in (2, 5, 20, 100, 1000)
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "correlated_maximum": correlated_maximum(),
        "judges": judges(rng),
        "forward_effective_observations": forward_evidence(),
    }
    Path(__file__).with_name("ch14_results.json").write_text(
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
