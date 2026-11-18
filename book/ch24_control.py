"""Chapter 24's calculations. Needs numpy and scipy.

1. A loss limit as a stopping rule: how often it stops a strategy that is working.
2. A human approval step: what it adds, by how well the person separates good from bad.
3. An edge that fades: the most that can ever be learned about it.
4. Capacity under square-root impact.
Run: python book/ch24_control.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import norm


def stopped(sharpe: float, vol: float, limit: float, years: float) -> float:
    """P(log equity falls by `limit` from its start within `years`), drift from the Sharpe ratio."""
    b = -math.log(1 - limit)
    nu = sharpe * vol - vol * vol / 2
    sd = vol * math.sqrt(years)
    return float(
        norm.cdf((-b - nu * years) / sd)
        + math.exp(-2 * nu * b / vol**2) * norm.cdf((-b + nu * years) / sd)
    )


def loss_limits() -> dict:
    out = {}
    for vol in (0.10, 0.20, 0.40):
        for limit in (0.05, 0.10, 0.20):
            out[f"vol_{vol}_limit_{limit}"] = {
                f"sharpe_{s}": stopped(s, vol, limit, 1.0) for s in (0.0, 1.0, 2.0)
            }
    return out


def human_gate() -> dict:
    """Of items reaching the gate, a share `prior` are good. The person approves good ones with probability a and bad ones with a_bad."""
    out = {}
    for prior in (0.5, 0.9):
        for a, a_bad in ((0.99, 0.99), (0.95, 0.80), (0.90, 0.40), (0.80, 0.10)):
            approved = prior * a + (1 - prior) * a_bad
            out[f"prior_{prior}_approves_{a}_of_good_{a_bad}_of_bad"] = {
                "share_approved": approved,
                "bad_among_approved": (1 - prior) * a_bad / approved,
                "bad_without_gate": 1 - prior,
                "good_lost": 1 - a,
            }
    return out


def fading_edge() -> dict:
    out = {}
    for s0 in (1.0, 1.5, 2.0, 3.0):
        row = {"years_of_life_needed_for_20_to_1": 4 * math.log(20) / s0**2}
        for tau in (0.5, 1.0, 2.0, 5.0):
            total = s0**2 * tau / 4
            row[f"half_life_{tau}y"] = {
                "total_information_nats": total,
                "best_odds_ever": math.exp(total),
            }
        out[str(s0)] = row
    return out


def capacity() -> dict:
    """Net annual return = gross - impact, impact = Y * daily_vol * sqrt(participation) * annual turnover."""
    out, y, daily_vol = {}, 1.0, 0.012
    for gross, turnover in ((0.10, 20), (0.10, 100), (0.05, 20)):
        row = {}
        for participation in (0.0001, 0.001, 0.01, 0.05):
            row[str(participation)] = (
                gross - y * daily_vol * math.sqrt(participation) * turnover
            )
        row["participation_at_zero_net"] = (gross / (y * daily_vol * turnover)) ** 2
        out[f"gross_{gross}_turnover_{turnover}"] = row
    return out


def main() -> None:
    out = {
        "loss_limits_share_stopped_within_a_year": loss_limits(),
        "human_gate": human_gate(),
        "fading_edge": fading_edge(),
        "capacity_net_return": capacity(),
    }
    Path(__file__).with_name("ch24_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    def r(x):
        return (
            {k: r(v) for k, v in x.items()}
            if isinstance(x, dict)
            else (round(x, 3) if isinstance(x, float) else x)
        )

    print(json.dumps(r(out)).replace("}, ", "},\n"))


if __name__ == "__main__":
    main()
