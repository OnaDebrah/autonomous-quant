"""Chapter 9's simulations. Needs numpy and scipy.

1. What a candidate adds to a portfolio it is correlated with: the closed form against a simulated optimum.
2. A correlation gate on short records: how often it blocks and passes wrongly.
3. Which volatility is consistent with a stated drawdown tolerance.
4. Watchlists formed by attention: what the watched assets' past and future returns look like.
5. Estimating each user's success rate from a few outcomes: alone, pooled, and partly pooled.
Run: python book/ch09_investor.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261120
PERIODS = 252


def marginal_value(rng) -> dict:
    out = {}
    days = 2_000_000
    for rho in (0.0, 0.5, 0.7, 0.9):
        s = 1.0 / math.sqrt(PERIODS)  # both have an annual Sharpe ratio of 1
        z = rng.standard_normal((2, days))
        p = s + z[0]
        c = s + rho * z[0] + math.sqrt(1 - rho**2) * z[1]
        cov = np.cov(np.vstack([p, c]))
        w = np.linalg.solve(cov, np.array([p.mean(), c.mean()]))
        best = w @ np.vstack([p, c])
        combined = best.mean() / best.std() * math.sqrt(PERIODS)
        theory_ir = math.sqrt((1 - rho) / (1 + rho))
        out[str(rho)] = {
            "theory_added_ir": theory_ir,
            "theory_combined_sharpe": math.sqrt(1 + theory_ir**2),
            "simulated_combined_sharpe": float(combined),
        }
    return out


def correlation_gate(rng) -> dict:
    out, runs, threshold = {}, 40_000, 0.70
    for n in (5, 20, 60, 250):
        row = {}
        for rho in (0.3, 0.5, 0.7, 0.9):
            a = rng.standard_normal((runs, n))
            b = rho * a + math.sqrt(1 - rho**2) * rng.standard_normal((runs, n))
            a -= a.mean(axis=1, keepdims=True)
            b -= b.mean(axis=1, keepdims=True)
            r = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
            row[str(rho)] = {
                "sd_of_estimate": float(r.std()),
                "theory_sd": (1 - rho**2) / math.sqrt(n - 1),
                "share_blocked": float((np.abs(r) >= threshold).mean()),
            }
        out[str(n)] = row
    return out


def drawdown_tolerance(rng) -> dict:
    """Share of two-year paths whose maximum drawdown exceeds the tolerance, by annual volatility and Sharpe ratio."""
    runs, days, out = 20_000, 504, {}
    z = rng.standard_normal((runs, days))
    for sharpe in (0.0, 1.0):
        for vol in (0.05, 0.10, 0.20, 0.30, 0.40):
            daily = vol / math.sqrt(PERIODS)
            log_equity = np.cumsum(
                sharpe * vol / PERIODS - daily**2 / 2 + daily * z, axis=1
            )
            peak = np.maximum.accumulate(
                np.concatenate([np.zeros((runs, 1)), log_equity], axis=1), axis=1
            )[:, 1:]
            worst = 1 - np.exp(-(peak - log_equity).max(axis=1))
            out[f"sharpe_{sharpe}_vol_{vol}"] = {
                f"beyond_{int(d * 100)}pct": float((worst > d).mean())
                for d in (0.10, 0.20, 0.30)
            } | {"median": float(np.median(worst))}
    return out


def attention(rng) -> dict:
    """Assets are added to a watchlist with a chance that rises with the size of last month's move. Nothing is predictable."""
    assets, runs = 500, 2_000
    past = rng.standard_normal(
        (runs, assets)
    )  # last month's return, in units of its own standard deviation
    future = rng.standard_normal((runs, assets))
    chance = (
        0.02
        + 0.10 * np.clip(np.abs(past) - 1, 0, None)
        + 0.10 * np.clip(past - 1, 0, None)
    )  # big moves, and more so big rises
    watched = rng.random((runs, assets)) < chance
    return {
        "share_watched": float(watched.mean()),
        "mean_past_return_watched": float(past[watched].mean()),
        "mean_abs_past_return_watched": float(np.abs(past[watched]).mean()),
        "mean_abs_past_return_all": float(np.abs(past).mean()),
        "share_of_watched_with_past_above_1sd": float((past[watched] > 1).mean()),
        "share_of_all_with_past_above_1sd": float((past > 1).mean()),
        "mean_future_return_watched": float(future[watched].mean()),
    }


def pooling(rng) -> dict:
    users = 200_000
    theta = rng.beta(
        2, 6, users
    )  # each user's true chance that a promoted idea succeeds forward
    n = rng.geometric(0.25, users)  # matured outcomes per user: few
    s = rng.binomial(n, theta)
    alone = s / n
    pooled_mean = s.sum() / n.sum()
    kappa = 8.0  # prior strength: the Beta(2, 6) the users were drawn from
    partial = (s + kappa * pooled_mean) / (n + kappa)

    def mse(est):
        return float(((est - theta) ** 2).mean())

    few = n <= 2
    return {
        "mean_outcomes_per_user": float(n.mean()),
        "mse_each_user_alone": mse(alone),
        "mse_one_pooled_rate": mse(np.full(users, pooled_mean)),
        "mse_partial_pooling": mse(partial),
        "mse_alone_users_with_at_most_2": float(
            ((alone[few] - theta[few]) ** 2).mean()
        ),
        "mse_partial_users_with_at_most_2": float(
            ((partial[few] - theta[few]) ** 2).mean()
        ),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "marginal_value": marginal_value(rng),
        "correlation_gate": correlation_gate(rng),
        "drawdown_tolerance": drawdown_tolerance(rng),
        "attention": attention(rng),
        "pooling": pooling(rng),
    }
    Path(__file__).with_name("ch09_results.json").write_text(
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
