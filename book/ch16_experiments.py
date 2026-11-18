"""Chapter 16's calculations and one simulation. Needs numpy and scipy.

1. One-at-a-time ablation of two controls that overlap: what each arm shows.
2. The price of each control in a published ablation: real discoveries given up per false discovery avoided,
   and how that price changes with the share of ideas that are real.
3. How often a keep / not-worth-keeping / inconclusive rule reaches each verdict, by size of study.
4. Pairing: the variance of a difference in pass rates from the families on which two arms disagree.
5. Analysis choices made after seeing the data.
Run: python book/ch16_experiments.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261127


def overlapping_controls(rng) -> dict:
    """Null results reach two checks. Each check alone stops half of them; they agree with correlation r."""
    out, n = {}, 2_000_000
    for r in (0.0, 0.5, 0.9, 1.0):
        z = rng.standard_normal((2, n))
        a = z[0] > 0
        b = (r * z[0] + math.sqrt(1 - r * r) * z[1]) > 0 if r < 1 else a.copy()
        both, only_a, only_b, none = (~a & ~b).mean(), (~a).mean(), (~b).mean(), 1.0
        out[str(r)] = {
            "passes_with_both": float(both),
            "passes_without_a": float(only_b),
            "passes_without_b": float(only_a),
            "passes_with_neither": none,
            "one_at_a_time_effect_of_a": float(only_b - both),
            "sum_of_one_at_a_time_effects": float((only_b - both) + (only_a - both)),
            "effect_of_removing_both": float(none - both),
        }
    return out


def exchange_rates() -> dict:
    # Pooled over eight edged cells, 8,000 edged families per arm: (false, true) discoveries.
    arms = {
        "full": (188, 1794),
        "batch_only_count": (228, 2071),
        "no_holdout": (312, 2314),
        "no_walk_forward": (188, 1794),
        "no_deflation": (2470, 4881),
        "all_off": (3342, 5496),
    }
    full_false, full_true = arms["full"]
    out = {}
    for name, (false, true) in arms.items():
        if name == "full":
            continue
        avoided, lost = false - full_false, true - full_true
        price = lost / avoided if avoided else None
        row = {
            "false_avoided": avoided,
            "true_lost": lost,
            "true_lost_per_false_avoided": price,
        }
        if price is not None:
            # The study's mix is a quarter of families edged. Rescale to other shares of real ideas.
            base = 0.25 / 0.75
            row["price_at_share_real"] = {
                str(s): price * (s / (1 - s)) / base for s in (0.25, 0.10, 0.02, 0.01)
            }
        out[name] = row
    return out


def verdict_probabilities() -> dict:
    """An estimator of a relative cut with standard error se300 at 300 units; keep if the lower bound is at least 25,
    not worth keeping if the upper bound is below 25."""
    out, threshold = {}, 25.0
    for true_cut, se300 in ((22.9, 10.1), (40.0, 10.1), (60.0, 10.1), (10.0, 10.1)):
        row = {}
        for n in (100, 300, 1000, 3000):
            se = se300 * math.sqrt(300 / n)
            keep = norm.sf((threshold - true_cut) / se + 1.96)
            drop = norm.cdf((threshold - true_cut) / se - 1.96)
            row[str(n)] = {
                "keep": float(keep),
                "not_worth_keeping": float(drop),
                "inconclusive": float(1 - keep - drop),
            }
        out[str(true_cut)] = row
    return out


def pairing(rng) -> dict:
    """Two arms on the same 4,000 families: pass rates 0.224 and 0.289, the second passing everything the first does."""
    n, p_low, p_high, runs = 4000, 0.224, 0.289, 20_000
    u = rng.random((runs, n))
    low, high = u < p_low, u < p_high  # nested: same families, common random numbers
    paired = (high.mean(axis=1) - low.mean(axis=1)).std()
    separate = (
        rng.binomial(n, p_high, runs) / n - rng.binomial(n, p_low, runs) / n
    ).std()
    discordant = p_high - p_low
    return {
        "sd_of_difference_paired": float(paired),
        "theory_paired": math.sqrt(discordant * (1 - discordant) / n),
        "sd_of_difference_separate_samples": float(separate),
        "theory_separate": math.sqrt((p_low * (1 - p_low) + p_high * (1 - p_high)) / n),
        "families_needed_separately_for_the_paired_precision": float(
            n * (separate / paired) ** 2
        ),
    }


def forking_paths() -> dict:
    return {
        str(j): {
            "analyses": 2**j,
            "chance_one_is_significant_if_independent": 1 - 0.95 ** (2**j),
        }
        for j in (1, 2, 3, 4, 5)
    }


def sample_size() -> dict:
    out = {}
    for p1, p2 in ((0.20, 0.35), (0.224, 0.289), (0.003, 0.006)):
        n = (
            (norm.ppf(0.975) + norm.ppf(0.8)) ** 2
            * (p1 * (1 - p1) + p2 * (1 - p2))
            / (p2 - p1) ** 2
        )
        out[f"{p1}_to_{p2}"] = float(n)
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "overlapping_controls": overlapping_controls(rng),
        "exchange_rates": exchange_rates(),
        "verdict_probabilities": verdict_probabilities(),
        "pairing": pairing(rng),
        "forking_paths": forking_paths(),
        "units_per_arm_for_80pct_power_unpaired": sample_size(),
    }
    Path(__file__).with_name("ch16_results.json").write_text(
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
