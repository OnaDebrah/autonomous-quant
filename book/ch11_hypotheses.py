"""Chapter 11's simulations. Needs numpy and scipy.

1. The same strategy judged against five controls: how often a strategy with no skill "beats" each, and how well
   each control separates a strategy with skill from one without.
2. The expected best result of a year of proposals, counted as one search.
3. Families judged one at a time: the chance that at least one passes with nothing there.
Run: python book/ch11_hypotheses.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261122
PERIODS = 252
TRAIN, HOLDOUT = 404, 100
COST = 0.0015
CASH = 0.04 / PERIODS
EULER = 0.5772156649015329


def controls(rng) -> dict:
    runs, days = 40_000, TRAIN + HOLDOUT
    market = rng.normal(0.06 / PERIODS, 0.16 / math.sqrt(PERIODS), (runs, days))
    sector_own = rng.normal(0, 0.08 / math.sqrt(PERIODS), (runs, days))
    own = rng.normal(0, 0.20 / math.sqrt(PERIODS), (runs, days))
    sector = market + sector_own
    out = {}
    for name, ic in (("no_skill", 0.0), ("skill", 0.06)):
        # A persistent signal; with skill it also knows a little about the stock's own next return.
        noise = np.empty((runs, days))
        noise[:, 0] = rng.standard_normal(runs)
        shocks = rng.standard_normal((runs, days)) * math.sqrt(1 - 0.9**2)
        for t in range(1, days):
            noise[:, t] = 0.9 * noise[:, t - 1] + shocks[:, t]
        nxt = np.concatenate([own[:, 1:], np.zeros((runs, 1))], axis=1) / (
            0.20 / math.sqrt(PERIODS)
        )
        signal = ic * nxt + math.sqrt(1 - ic**2) * noise
        stock = 1.2 * market + 0.5 * sector_own + own
        held = np.concatenate(
            [np.zeros((runs, 1)), (signal[:, :-1] > -0.5).astype(float)], axis=1
        )
        traded = np.abs(np.diff(held, axis=1, prepend=0.0))
        strategy = held * stock + (1 - held) * CASH - COST * traded
        h = slice(TRAIN, days)

        def total(x):
            return np.prod(1 + x[:, h], axis=1) - 1

        # Beta from the last 60 training days only.
        m_train, s_train = market[:, TRAIN - 60 : TRAIN], stock[:, TRAIN - 60 : TRAIN]
        mc = m_train - m_train.mean(axis=1, keepdims=True)
        beta = (mc * (s_train - s_train.mean(axis=1, keepdims=True))).sum(axis=1) / (
            mc**2
        ).sum(axis=1)
        exposure = held[:, h].mean(axis=1, keepdims=True)
        control_returns = {
            "cash": np.full((runs, days), CASH),
            "buy_and_hold": stock,
            "sector_only": sector,
            "beta_matched": beta[:, None] * market + (1 - beta[:, None]) * CASH,
            "exposure_matched": exposure * stock + (1 - exposure) * CASH,
        }
        row = {
            "strategy_mean_holdout_return": float(total(strategy).mean()),
            "mean_share_of_days_held": float(exposure.mean()),
            "beta_sd": float(beta.std()),
        }
        for control, r in control_returns.items():
            inc = total(strategy) - total(r)
            row[control] = {
                "beats_control": float((inc > 0).mean()),
                "mean_incremental": float(inc.mean()),
                "sd_incremental": float(inc.std(ddof=1)),
            }
        out[name] = row
    for control in (
        "cash",
        "buy_and_hold",
        "sector_only",
        "beta_matched",
        "exposure_matched",
    ):
        a, b = out["skill"][control], out["no_skill"][control]
        out.setdefault("separation", {})[control] = (
            a["mean_incremental"] - b["mean_incremental"]
        ) / b["sd_incremental"]
    return out


def expected_best(n: int, variance: float) -> float:
    return math.sqrt(variance) * (
        (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def search_size() -> dict:
    out = {}
    for label, per_day, trials in (
        ("three_screens_a_day", 3, 5),
        ("three_confirmations_a_day", 3, 30),
        ("ten_screens_a_day", 10, 5),
    ):
        n = per_day * trials * PERIODS
        out[label] = {
            "trials_a_year": n,
            "expected_best_sharpe_no_skill": expected_best(n, 0.5),
        }
    return out


def families() -> dict:
    return {str(f): 1 - 0.95**f for f in (1, 10, 50, 250, 750)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "controls": controls(rng),
        "search_size": search_size(),
        "families_any_pass_at_5pct": families(),
    }
    Path(__file__).with_name("ch11_results.json").write_text(
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
