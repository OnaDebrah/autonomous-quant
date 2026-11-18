"""Chapter 17's simulations. Needs numpy.

1. Allocating a daily budget across four sources of ideas when a discovery needs a pass and then a delayed forward result.
   Two ways of scoring an idea that did not pass: no information, or an immediate zero.
2. Forward outcomes that are dropped when a loss limit is hit: what the surviving outcomes average.
3. Expected information gain of one more observation, and what one failure says about an idea.
Run: python book/ch17_allocation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261128
PASS = np.array(
    [0.02, 0.06, 0.10, 0.03]
)  # chance a study passes the contract, by source
FORWARD = np.array([0.60, 0.50, 0.15, 0.30])  # chance a passed idea succeeds forward
YIELD = PASS * FORWARD
BUDGET, DELAY = 6, 60


def bandit(rng, policy: str, days: int, runs: int) -> dict:
    arms = len(PASS)
    discoveries = np.zeros(runs)
    best_share = np.zeros(runs)
    best = int(YIELD.argmax())
    for k in range(runs):
        s, f = np.zeros(arms), np.zeros(arms)
        pending = []  # (day due, arm, success)
        pulls = np.zeros(arms)
        for day in range(days):
            while pending and pending[0][0] <= day:
                _, a, ok = pending.pop(0)
                s[a] += ok
                f[a] += 1 - ok
            for _ in range(BUDGET):
                if policy == "uniform":
                    a = int(rng.integers(arms))
                else:
                    a = int(rng.beta(1 + s, 1 + f).argmax())
                pulls[a] += 1
                if rng.random() < PASS[a]:
                    ok = float(rng.random() < FORWARD[a])
                    discoveries[k] += ok
                    pending.append((day + DELAY, a, ok))
                elif policy == "thompson_yield":
                    f[a] += 1  # a rejection is a zero, known at once
        best_share[k] = pulls[best] / pulls.sum()
    return {
        "discoveries": float(discoveries.mean()),
        "share_of_budget_to_best_source": float(best_share.mean()),
    }


def allocation(rng) -> dict:
    out = {
        "yield_per_study": [float(y) for y in YIELD],
        "quality_given_pass": [float(x) for x in FORWARD],
    }
    for days in (252, 756):
        row = {
            "all_to_best_source": float(BUDGET * days * YIELD.max()),
            "all_to_highest_quality_source": float(
                BUDGET * days * YIELD[FORWARD.argmax()]
            ),
        }
        for policy in ("uniform", "thompson_quality", "thompson_yield"):
            row[policy] = bandit(rng, policy, days, 300)
        out[str(days)] = row
    return out


def censoring(rng) -> dict:
    """Sixty-day forward outcomes; an outcome is dropped if the strategy's loss from its start ever reaches the limit."""
    runs, days, out = 200_000, 60, {}
    for sharpe in (0.0, 1.0):
        vol = 0.20 / math.sqrt(252)
        r = rng.normal(sharpe * 0.20 / 252, vol, (runs, days))
        path = np.cumsum(r, axis=1)
        total = path[:, -1]
        for limit in (0.05, 0.10):
            kept = path.min(axis=1) > -limit
            out[f"sharpe_{sharpe}_limit_{limit}"] = {
                "true_mean_return": float(total.mean()),
                "share_dropped": float(1 - kept.mean()),
                "mean_of_kept": float(total[kept].mean()),
                "share_positive_all": float((total > 0).mean()),
                "share_positive_kept": float((total[kept] > 0).mean()),
                "bounds_on_share_positive": [
                    float((total[kept] > 0).mean() * kept.mean()),
                    float((total[kept] > 0).mean() * kept.mean() + 1 - kept.mean()),
                ],
            }
    return out


def information() -> dict:
    gain = {}
    for a, b in ((1, 1), (2, 1), (2, 2), (3, 3), (6, 6), (2, 10), (11, 11)):
        total = a + b
        reduction = a * b / (total**2 * (total + 1) ** 2)
        gain[f"beta_{a}_{b}"] = {
            "variance": a * b / (total**2 * (total + 1)),
            "expected_reduction": reduction,
            "relative_to_uniform_prior": reduction * 36,
        }
    failure = {}
    for power, size in ((0.22, 0.01), (0.45, 0.005), (0.80, 0.05)):
        lr = (1 - power) / (1 - size)
        failure[f"power_{power}"] = {
            "likelihood_ratio_of_one_failure": lr,
            "after_3": lr**3,
            "after_5": lr**5,
            "posterior_after_5_from_prior_10pct": 0.1 * lr**5 / (0.1 * lr**5 + 0.9),
        }
    return {"information_gain": gain, "failures": failure}


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "allocation": allocation(rng),
        "censoring": censoring(rng),
        **information(),
    }
    Path(__file__).with_name("ch17_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    def r(x):
        if isinstance(x, dict):
            return {k: r(v) for k, v in x.items()}
        if isinstance(x, list):
            return [r(v) for v in x]
        return round(x, 4) if isinstance(x, float) else x

    print(json.dumps(r(out)).replace("}, ", "},\n"))


if __name__ == "__main__":
    main()
