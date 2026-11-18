"""Chapter 12: a small planted-truth benchmark, and the first rows of the book's scorecard. Needs numpy and scipy.

A panel is 20 families of 20 tries on 504 daily bars. Five families have three tries with a planted one-bar edge;
every other try is a random signal. Three judges are scored on the same panels, with and without trading costs.
Run: python book/ch12_planted_truth.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, skew, kurtosis

SEED = 20261123
PERIODS, BARS, VOL = 252, 504, 0.012
FAMILIES, TRIES, EDGED_FAMILIES, EDGES = 20, 20, 5, 3
WINDOWS = (1, 5, 10, 20, 40)  # smoothing of the random signals: 5 windows x 4 draws
PANELS = 600
EULER = 0.5772156649015329


def edge_sharpe(k: float, cost: float) -> float:
    """Annual Sharpe ratio of 'long when next return + k*sigma*noise is positive, else flat', in closed form."""
    mean = (
        VOL / math.sqrt(2 * math.pi * (1 + k * k)) - cost / 2
    )  # the position changes on half the days
    var = VOL**2 / 2 - (VOL / math.sqrt(2 * math.pi * (1 + k * k))) ** 2 + cost**2 / 4
    return mean / math.sqrt(var) * math.sqrt(PERIODS)


def calibrate(target: float, cost: float) -> float:
    return brentq(lambda k: edge_sharpe(k, cost) - target, 0.01, 50.0)


def returns_of(signal: np.ndarray, market: np.ndarray, cost: float) -> np.ndarray:
    position = np.zeros(signal.shape)
    position[..., 1:] = signal[..., :-1] > 0
    change = np.abs(np.diff(position, axis=-1, prepend=0.0))
    return position * market - cost * change


def family(rng, edged: bool, k: float, cost: float):
    market = rng.normal(0.0, VOL, BARS)
    signals, is_edge = [], []
    n_edge = EDGES if edged else 0
    nxt = np.concatenate([market[1:], [0.0]])
    for _ in range(n_edge):
        signals.append(nxt + k * VOL * rng.standard_normal(BARS))
        is_edge.append(True)
    j = 0
    while len(signals) < TRIES:
        w = WINDOWS[j % len(WINDOWS)]
        noise = rng.standard_normal(BARS + w - 1)
        signals.append(np.convolve(noise, np.ones(w) / w, mode="valid"))
        is_edge.append(False)
        j += 1
    order = rng.permutation(TRIES)
    r = returns_of(np.array(signals)[order], market, cost)
    return r, np.array(is_edge)[order]


def judge(r: np.ndarray) -> dict:
    """Each judge selects the try with the highest Sharpe ratio; they differ in what it must clear."""
    sd = r.std(axis=1, ddof=1)
    sr = np.where(sd > 0, r.mean(axis=1) / np.where(sd > 0, sd, 1), -np.inf)
    best = int(sr.argmax())
    x, s = r[best], sr[best]
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    se = math.sqrt(max(1e-12, 1 - g3 * s + (g4 - 1) / 4 * s * s) / (BARS - 1))
    bar = math.sqrt(1 / BARS) * (
        (1 - EULER) * norm.ppf(1 - 1 / TRIES)
        + EULER * norm.ppf(1 - 1 / (TRIES * math.e))
    )
    return {
        "best": best,
        "naive_above_1": s * math.sqrt(PERIODS) > 1.0,
        "single_test_95": s / se > norm.ppf(0.95),
        "deflated_95": (s - bar) / se > norm.ppf(0.95),
    }


def world(rng, cost: float, target: float = 1.5) -> dict:
    k = calibrate(target, cost)
    judges = ("naive_above_1", "single_test_95", "deflated_95")
    tally = {j: {"false": 0, "true": 0} for j in judges}
    panel_false = {j: [] for j in judges}
    check = []
    for _ in range(PANELS):
        counts = {j: 0 for j in judges}
        for f in range(FAMILIES):
            r, is_edge = family(rng, f < EDGED_FAMILIES, k, cost)
            if f < EDGED_FAMILIES:
                e = r[is_edge][0]
                check.append(e.mean() / e.std(ddof=1) * math.sqrt(PERIODS))
            v = judge(r)
            for j in judges:
                if v[j]:
                    kind = "true" if is_edge[v["best"]] else "false"
                    tally[j][kind] += 1
                    counts[j] += kind == "false"
        for j in judges:
            panel_false[j].append(counts[j] / FAMILIES)
    families, edged = PANELS * FAMILIES, PANELS * EDGED_FAMILIES
    out = {
        "cost": cost,
        "strength_k": k,
        "closed_form_sharpe": edge_sharpe(k, cost),
        "simulated_edge_sharpe_504_bars": float(np.mean(check)),
        "families": families,
    }
    for j in judges:
        f, t = tally[j]["false"], tally[j]["true"]
        out[j] = {
            "false_pass_rate": f / families,
            "false_pass_rate_se_over_panels": float(
                np.std(panel_false[j], ddof=1) / math.sqrt(PANELS)
            ),
            "power": t / edged,
            "false_discovery_share": f / (f + t) if f + t else None,
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "panels": PANELS,
        "gross_sharpe_of_the_costed_edge": edge_sharpe(calibrate(1.5, 0.0015), 0.0),
        "with_costs": world(rng, 0.0015),
        "without_costs": world(rng, 0.0),
    }
    Path(__file__).with_name("ch12_results.json").write_text(
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
