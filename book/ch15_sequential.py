"""Chapter 15's simulations. Needs numpy and scipy.

1. A holdout check after a significance test: how much it removes when the test saw the holdout and when it did not.
2. A walk-forward consistency check on results that already passed, with and without a regime.
3. Looking at a running test every day.
4. A sequential test of a candidate against an incumbent, with the error budget split across candidates.
5. How long a paper record must be before a vanished edge shows.
Run: python book/ch15_sequential.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261126
EULER = 0.5772156649015329
Z95 = float(norm.ppf(0.95))
BARS, PERIODS, TRIES = 504, 252, 20
HOLD = 0.2


def m(n: int) -> float:
    return (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))


def z_of(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1) / x.std(axis=-1, ddof=1) * math.sqrt(x.shape[-1])


def holdout_and_walk_forward(rng) -> dict:
    """Families of 20 independent tries; none, or three with an annual Sharpe ratio of 1.5."""
    out = {}
    cut = 400  # the last 104 bars, about a fifth, are the holdout; the first 400 make five windows of 80
    for name, edged, runs in (("null", 0, 60_000), ("edged", 3, 6_000)):
        mu = 1.5 / math.sqrt(PERIODS)
        tallies = {
            k: 0
            for k in (
                "sig_all",
                "sig_all_and_holdout",
                "sig_all_and_wf",
                "sig_train",
                "sig_train_and_holdout",
            )
        }
        for _ in range(runs // 500):
            x = rng.standard_normal((500, TRIES, BARS))
            x[:, :edged] += mu
            # (a) significance on every bar, then a check on the last fifth
            z_all = z_of(x)
            best = z_all.argmax(axis=1)
            rows = np.arange(500)
            chosen = x[rows, best]
            ok = (best < edged) if edged else np.ones(500, bool)
            sig = (z_all[rows, best] > m(TRIES) + Z95) & ok
            hold = chosen[:, cut:].mean(axis=1) > 0
            windows = chosen[:, :cut].reshape(500, 5, -1).mean(axis=2) > 0
            wf = windows.mean(axis=1) >= 0.6
            tallies["sig_all"] += sig.sum()
            tallies["sig_all_and_holdout"] += (sig & hold).sum()
            tallies["sig_all_and_wf"] += (sig & wf).sum()
            # (b) select and test on the first four fifths only
            z_tr = z_of(x[:, :, :cut])
            best_t = z_tr.argmax(axis=1)
            chosen_t = x[rows, best_t]
            ok_t = (best_t < edged) if edged else np.ones(500, bool)
            sig_t = (z_tr[rows, best_t] > m(TRIES) + Z95) & ok_t
            tallies["sig_train"] += sig_t.sum()
            tallies["sig_train_and_holdout"] += (
                sig_t & (chosen_t[:, cut:].mean(axis=1) > 0)
            ).sum()
        out[name] = {k: v / runs for k, v in tallies.items()}
        out[name]["holdout_removes_share_when_test_saw_it"] = 1 - tallies[
            "sig_all_and_holdout"
        ] / max(tallies["sig_all"], 1)
        out[name]["holdout_removes_share_when_test_did_not"] = 1 - tallies[
            "sig_train_and_holdout"
        ] / max(tallies["sig_train"], 1)
        out[name]["walk_forward_removes_share"] = 1 - tallies["sig_all_and_wf"] / max(
            tallies["sig_all"], 1
        )
    # A result whose whole edge sits in one quarter of the sample.
    runs = 20_000
    x = rng.standard_normal((runs, BARS))
    x[:, :100] += 0.35  # an enormous edge in the first 100 bars, nothing after
    sig = z_of(x) > m(TRIES) + Z95
    wf = (x[:, :400].reshape(runs, 5, -1).mean(axis=2) > 0).mean(axis=1) >= 0.6
    hold = x[:, 400:].mean(axis=1) > 0
    out["one_regime"] = {
        "passes_significance": float(sig.mean()),
        "of_those_fail_walk_forward": float((sig & ~wf).sum() / sig.sum()),
        "of_those_fail_holdout": float((sig & ~hold).sum() / sig.sum()),
    }
    return out


def repeated_looks(rng) -> dict:
    runs, out = 40_000, {}
    steps = rng.standard_normal((runs, 500))
    walk = np.cumsum(steps, axis=1) / np.sqrt(np.arange(1, 501))
    for looks in (1, 5, 20, 60, 250, 500):
        idx = np.unique(np.linspace(500 / looks, 500, looks).astype(int)) - 1
        out[str(looks)] = float((walk[:, idx] > Z95).any(axis=1).mean())
    return out


def sprt(rng) -> dict:
    """Daily paired differences, unit variance; the improvement worth detecting is an annual information ratio of 2."""
    delta, alpha, beta, max_days, runs = 2.0 / math.sqrt(365), 0.05, 0.20, 1500, 20_000
    info = delta**2 / 2
    out = {
        "information_per_day": info,
        "fixed_sample_days": ((Z95 + norm.ppf(1 - beta)) / delta) ** 2,
    }
    for k in (1, 2, 5):
        a_k = alpha / (k * (k + 1))
        upper, lower = math.log((1 - beta) / a_k), math.log(beta / (1 - a_k))
        row = {
            "alpha_for_this_candidate": a_k,
            "theory_days_if_better": ((1 - beta) * upper + beta * lower) / info,
            "theory_days_if_not": (a_k * upper + (1 - a_k) * lower) / (-info),
        }
        for label, mean in (("better", delta), ("no_better", 0.0)):
            d = mean + rng.standard_normal((runs, max_days))
            llr = np.cumsum(delta * d - delta**2 / 2, axis=1)
            up, down = llr >= upper, llr <= lower
            first_up = np.where(up.any(axis=1), up.argmax(axis=1), max_days)
            first_down = np.where(down.any(axis=1), down.argmax(axis=1), max_days)
            accepted = first_up < first_down
            stop = np.minimum(first_up, first_down) + 1
            row[label] = {
                "accepted": float(accepted.mean()),
                "undecided": float(
                    (np.minimum(first_up, first_down) == max_days).mean()
                ),
                "mean_days": float(np.minimum(stop, max_days).mean()),
            }
        out[f"candidate_{k}"] = row
    return out


def paper_length() -> dict:
    out = {
        "sharpe_standard_error": {str(n): math.sqrt(365 / n) for n in (14, 30, 90, 365)}
    }
    out["days_until_a_vanished_edge_is_below_the_band_half_the_time"] = {
        str(s): 365 * (Z95 / s) ** 2 for s in (1.0, 1.5, 2.0, 3.0)
    }
    return out


def conjunction() -> dict:
    return {
        f"{clauses}_clauses_each_passing_{int(b * 100)}pct_of_true_edges": b**clauses
        for clauses in (3, 6, 9)
        for b in (0.99, 0.95, 0.90)
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "holdout_and_walk_forward": holdout_and_walk_forward(rng),
        "repeated_looks": repeated_looks(rng),
        "sprt": sprt(rng),
        "paper_length": paper_length(),
        "conjunction": conjunction(),
    }
    Path(__file__).with_name("ch15_results.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n"
    )

    def r(x):
        return (
            {k: r(v) for k, v in x.items()}
            if isinstance(x, dict)
            else (round(float(x), 4) if isinstance(x, (float, np.floating)) else x)
        )

    print(json.dumps(r(out), default=float).replace("}, ", "},\n"))


if __name__ == "__main__":
    main()
