"""Chapter 13's simulations. Needs numpy and scipy, and ch12_planted_truth.py beside it.

1. What "deflated Sharpe ratio of at least 0.95" does to a family with nothing in it, by number of tries.
2. The scorecard extended: the deflated rule against a test on the maximum at chosen family-wise levels.
3. Estimating the spread of luck from the tries themselves when some of them are real.
4. A ledger that misses tries.
5. How long a record a real edge needs.
Run: python book/ch13_cost_of_search.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import ch12_planted_truth as w  # noqa: E402

SEED = 20261124
EULER = 0.5772156649015329
Z95 = float(norm.ppf(0.95))


def bar(n: int) -> float:
    """Expected best of n null tries, in units of the standard deviation of one try's Sharpe ratio."""
    return (
        0.0
        if n <= 1
        else (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e))
    )


def family_size_of_rule(rng) -> dict:
    out = {}
    for n in (1, 5, 10, 20, 50, 100, 1000):
        threshold = bar(n) + Z95
        exact = 1 - norm.cdf(threshold) ** n
        sim = float(
            (
                rng.standard_normal((200_000 if n <= 100 else 20_000, n)).max(axis=1)
                > threshold
            ).mean()
        )
        out[str(n)] = {
            "bar": bar(n),
            "threshold": threshold,
            "chance_a_null_family_passes": exact,
            "simulated": sim,
            "threshold_for_5pct_family": float(norm.ppf(0.95 ** (1 / n))),
            "threshold_for_1pct_family": float(norm.ppf(0.99 ** (1 / n))),
        }
    return out


def scorecard(rng) -> dict:
    k = w.calibrate(1.5, 0.0015)
    n = w.TRIES
    thresholds = {
        "single_test_5pct": Z95,
        "max_test_5pct": float(norm.ppf(0.95 ** (1 / n))),
        "max_test_1pct": float(norm.ppf(0.99 ** (1 / n))),
        "deflated_0.95": bar(n) + Z95,
    }
    tally = {j: [0, 0] for j in thresholds}
    for _ in range(w.PANELS):
        for f in range(w.FAMILIES):
            r, is_edge = w.family(rng, f < w.EDGED_FAMILIES, k, 0.0015)
            sd = r.std(axis=1, ddof=1)
            z = np.where(
                sd > 0, r.mean(axis=1) / np.where(sd > 0, sd, 1), -np.inf
            ) * math.sqrt(w.BARS)
            best = int(z.argmax())
            for j, c in thresholds.items():
                if z[best] > c:
                    tally[j][1 if is_edge[best] else 0] += 1
    families, edged = w.PANELS * w.FAMILIES, w.PANELS * w.EDGED_FAMILIES
    return {
        j: {
            "threshold": thresholds[j],
            "false_pass_rate": f / families,
            "power": t / edged,
            "false_discovery_share": f / (f + t) if f + t else None,
        }
        for j, (f, t) in tally.items()
    }


def measured_spread(rng) -> dict:
    """Twenty tries, a share of them with a real annual Sharpe ratio of 1.5; no costs."""
    k = w.calibrate(1.5, 0.0)
    mu = 1.5 / math.sqrt(w.PERIODS)
    v0 = 1 / w.BARS
    out, runs, n = {}, 3_000, 20
    for share in (0.0, 0.25, 0.5, 0.75):
        real = int(round(share * n))
        ratios, pass_measured, pass_analytic = [], 0, 0
        for _ in range(runs):
            market = rng.normal(0.0, w.VOL, w.BARS)
            nxt = np.concatenate([market[1:], [0.0]])
            signals = [
                nxt + k * w.VOL * rng.standard_normal(w.BARS) for _ in range(real)
            ] + [rng.standard_normal(w.BARS) for _ in range(n - real)]
            r = w.returns_of(np.array(signals), market, 0.0)
            sr = r.mean(axis=1) / r.std(axis=1, ddof=1)
            v = sr.var(ddof=1)
            ratios.append(math.sqrt(v / v0))
            best = int(sr.argmax())
            if best < real:
                se = math.sqrt(1 / (w.BARS - 1))
                pass_measured += (sr[best] - math.sqrt(v) * bar(n)) / se > Z95
                pass_analytic += (sr[best] - math.sqrt(v0) * bar(n)) / se > Z95
        out[str(share)] = {
            "theory_bar_ratio": math.sqrt(1 + share * (1 - share) * mu**2 / v0),
            "mean_bar_ratio": float(np.mean(ratios)),
            "passes_a_real_try_measured_spread": pass_measured / runs,
            "passes_a_real_try_no_skill_spread": pass_analytic / runs,
        }
    return out


def missed_tries() -> dict:
    out, annual = {}, math.sqrt(w.PERIODS / w.BARS)
    for counted, actual in ((20, 20), (20, 40), (20, 200), (20, 2000)):
        threshold = bar(counted) + Z95
        out[f"{counted}_of_{actual}"] = {
            "bar_used_annual": bar(counted) * annual,
            "bar_needed_annual": bar(actual) * annual,
            "chance_a_null_search_passes": 1 - norm.cdf(threshold) ** actual,
        }
    return out


def record_length() -> dict:
    out = {}
    for tries in (1, 20, 100, 1000):
        row = {}
        for sharpe in (1.0, 1.5, 2.0):
            # one real try among the rest; pass if z > bar + 1.645, z ~ N(sharpe * sqrt(T/252), 1)
            years_half = ((bar(tries) + Z95) / sharpe) ** 2
            power_2y = float(norm.sf(bar(tries) + Z95 - sharpe * math.sqrt(2)))
            row[str(sharpe)] = {
                "years_for_half_power": years_half,
                "power_on_two_years": power_2y,
            }
        out[str(tries)] = row
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "family_size_of_rule": family_size_of_rule(rng),
        "scorecard": scorecard(rng),
        "measured_spread": measured_spread(rng),
        "missed_tries": missed_tries(),
        "record_length": record_length(),
    }
    Path(__file__).with_name("ch13_results.json").write_text(
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
