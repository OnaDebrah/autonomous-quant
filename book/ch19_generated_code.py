"""Chapter 19's simulations. Needs numpy and scipy.

1. Catching code that reads ahead by re-running it on truncated data: how many truncation points are needed.
2. A tournament of candidate fixes with one reserved confirmation window, with and without letting a runner-up try.
3. The probability of backtest overfitting computed on two or three entries.
4. Judging a fix against the code it replaces on the same bars, against judging it alone.
5. A memory that stops proposing an idea after a number of failed variations.
Run: python book/ch19_generated_code.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261130
PERIODS = 252


def prefix_detection() -> dict:
    out = {}
    for share in (
        0.001,
        0.01,
        0.05,
        0.20,
    ):  # share of bars on which the code's output depends on later data
        out[str(share)] = {str(n): 1 - (1 - share) ** n for n in (5, 20, 100, 500)}
    return out


def tournament(rng) -> dict:
    """Candidates with no real improvement. Selection picks the best on one window; confirmation is a fresh window."""
    out, runs, level = {}, 200_000, 0.05
    z = norm.ppf(1 - level)
    for k in (2, 3, 6):
        selection = rng.standard_normal((runs, k))
        confirmation = rng.standard_normal((runs, k))
        order = np.argsort(-selection, axis=1)
        winner = np.take_along_axis(confirmation, order[:, :1], axis=1)[:, 0] > z
        any_in_turn = (confirmation > z).any(
            axis=1
        )  # try the winner, then each runner-up, on the same window
        out[str(k)] = {
            "winner_only": float(winner.mean()),
            "runner_ups_allowed": float(any_in_turn.mean()),
            "theory_runner_ups": 1 - (1 - level) ** k,
        }
    return out


def pbo_small(rng) -> dict:
    """Entries with no skill; 16 blocks, every way of using half as the selection sample."""
    out, runs, blocks, per = {}, 400, 16, 30
    splits = list(itertools.combinations(range(blocks), blocks // 2))
    splits = [splits[i] for i in rng.choice(len(splits), 600, replace=False)]
    masks = np.zeros((len(splits), blocks), bool)
    for i, s in enumerate(splits):
        masks[i, list(s)] = True
    for entries in (2, 3, 20):
        values = []
        for _ in range(runs):
            x = rng.standard_normal((entries, blocks, per)).mean(axis=2)  # block means
            ins = (masks[:, None, :] * x[None]).sum(axis=2)
            oos = ((~masks)[:, None, :] * x[None]).sum(axis=2)
            best = ins.argmax(axis=1)
            rank = (oos < oos[np.arange(len(splits)), best][:, None]).sum(axis=1) / (
                entries - 1
            )  # 0 worst, 1 best
            values.append((rank <= 0.5).mean() if entries > 2 else (rank < 0.5).mean())
        v = np.array(values)
        out[str(entries)] = {
            "mean_pbo": float(v.mean()),
            "sd_across_runs": float(v.std()),
            "share_of_runs_below_0.5": float((v < 0.5).mean()),
        }
    return out


def paired_fix(rng) -> dict:
    """A fix that adds a small edge to a strategy; daily returns of old and new share most of their variation."""
    out, runs, bars = {}, 20_000, 252
    vol = 0.01
    gain = (
        0.5 / PERIODS * 0.16
    )  # the fix adds half a unit of annual Sharpe ratio at 16% volatility
    for rho in (0.5, 0.9, 0.98):
        common = rng.standard_normal((runs, bars))
        old = vol * common
        new = (
            vol
            * (rho * common + math.sqrt(1 - rho**2) * rng.standard_normal((runs, bars)))
            + gain
        )
        diff = new - old
        t_paired = diff.mean(axis=1) / (diff.std(axis=1, ddof=1) / math.sqrt(bars))
        t_alone = new.mean(axis=1) / (new.std(axis=1, ddof=1) / math.sqrt(bars))
        out[str(rho)] = {
            "sd_of_difference_over_sd_of_strategy": math.sqrt(2 * (1 - rho)),
            "power_paired": float((t_paired > 1.645).mean()),
            "power_judged_alone": float((t_alone > 1.645).mean()),
        }
    return out


def memory_ban() -> dict:
    out = {}
    for power in (0.17, 0.45):
        out[str(power)] = {
            f"stopped_after_{k}_failures": (1 - power) ** k for k in (1, 2, 3, 5)
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "prefix_detection": prefix_detection(),
        "tournament": tournament(rng),
        "pbo_small": pbo_small(rng),
        "paired_fix": paired_fix(rng),
        "real_ideas_stopped_by_memory": memory_ban(),
    }
    Path(__file__).with_name("ch19_results.json").write_text(
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
