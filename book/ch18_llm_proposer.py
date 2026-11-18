"""Chapter 18's simulations. Needs numpy and scipy.

1. A proposer that remembers, imprecisely, how ideas performed on the history it is tested on.
   The ledger counts the ideas it proposed; the search it drew on is larger.
2. The same proposer confined to a closed list, with the bar set by the size of the list.
3. The same proposals on data the proposer has not seen.
4. A proposer that is merely better at choosing real ideas: what that is worth.
Run: python book/ch18_llm_proposer.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261129
EULER = 0.5772156649015329
Z95 = float(norm.ppf(0.95))
IDEAS = 6  # proposals per round


def m(n: int) -> float:
    return (
        0.0
        if n <= 1
        else (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def contaminated(rng) -> dict:
    out, runs = {}, 40_000
    for pool in (6, 50, 1000):
        for blur in (0.5, 1.0, 3.0, None):
            z = rng.standard_normal(
                (runs, pool)
            )  # each idea's standardized result on the history: no idea is real
            memory = (
                z + blur * rng.standard_normal((runs, pool))
                if blur is not None
                else rng.standard_normal((runs, pool))
            )
            chosen = (
                np.argpartition(-memory, IDEAS - 1, axis=1)[:, :IDEAS]
                if pool > IDEAS
                else np.tile(np.arange(pool), (runs, 1))
            )
            best = np.take_along_axis(z, chosen, axis=1).max(axis=1)
            fresh = rng.standard_normal((runs, IDEAS)).max(
                axis=1
            )  # the same six ideas on data the proposer never saw
            out[f"pool_{pool}_blur_{blur}"] = {
                "mean_best_of_the_six": float(best.mean()),
                "passes_counting_six": float((best > m(IDEAS) + Z95).mean()),
                "passes_family_5pct_counting_six": float(
                    (best > norm.ppf(0.95 ** (1 / IDEAS))).mean()
                ),
                "passes_counting_the_pool": float(
                    (best > norm.ppf(0.95 ** (1 / pool))).mean()
                ),
                "passes_on_unseen_data_counting_six": float(
                    (fresh > norm.ppf(0.95 ** (1 / IDEAS))).mean()
                ),
            }
    return out


def better_proposer() -> dict:
    """Studies are judged at a false-pass rate a and power b; a share pi of proposals is real."""
    out = {}
    for a, b in ((0.005, 0.45), (0.0005, 0.17)):
        for pi in (0.005, 0.01, 0.02, 0.05, 0.10):
            true, false = pi * b, (1 - pi) * a
            out[f"a_{a}_b_{b}_pi_{pi}"] = {
                "real_discoveries_per_1000_studies": 1000 * true,
                "false_per_1000": 1000 * false,
                "false_discovery_share": false / (true + false),
            }
    return out


def adaptive_holdout() -> dict:
    return {str(rounds): 1 - 0.5**rounds for rounds in (1, 2, 3, 5, 10)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "contaminated": contaminated(rng),
        "better_proposer": better_proposer(),
        "chance_a_null_idea_clears_a_positive_holdout_within_rounds": adaptive_holdout(),
    }
    Path(__file__).with_name("ch18_results.json").write_text(
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
