"""Chapter 3's simulations. Needs only numpy.

1. Revisions: a signal read from revised values against the same signal read from first releases,
   when the revision is news that also moves the next return.
2. Survivorship: buying assets that have halved, on a market with nothing to find, measured on the
   point-in-time universe and on the universe of assets that survived to the end.
Run: python book/ch03_time_and_data.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261113
PERIODS = 252

# Revisions
BARS = 504
DAILY_VOL = 0.012
REVISION_SD = 0.5  # the revision's standard deviation, in units of the first release's
NEWS_CORRELATION = 0.10  # correlation between the revision and the next return
REVISION_RUNS = 2_000

# Survivorship
ASSETS = 200
YEARS = 5
ASSET_DAILY_VOL = 0.03
DELIST_LEVEL = 0.20  # an asset is delisted when its price first closes at or below this share of its starting price
DIP_LEVEL = (
    0.50  # the rule holds assets priced at or below this share of their starting price
)
SURVIVOR_RUNS = 500


def sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * math.sqrt(PERIODS)) if sd > 0 else 0.0


def summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "standard_error": float(v.std(ddof=1) / math.sqrt(len(v))),
    }


def revisions(rng: np.random.Generator) -> dict:
    revised, first = [], []
    for _ in range(REVISION_RUNS):
        release = rng.standard_normal(BARS)
        news = rng.standard_normal(BARS)
        other = rng.standard_normal(BARS)
        returns = DAILY_VOL * (
            NEWS_CORRELATION * news + math.sqrt(1 - NEWS_CORRELATION**2) * other
        )
        final = release + REVISION_SD * news
        revised.append(
            sharpe(np.sign(final) * returns)
        )  # the value on file today, placed on its observation date
        first.append(
            sharpe(np.sign(release) * returns)
        )  # the value that had arrived when the position was taken
    signal_correlation = NEWS_CORRELATION * REVISION_SD / math.sqrt(1 + REVISION_SD**2)
    return {
        "runs": REVISION_RUNS,
        "revision_sd": REVISION_SD,
        "news_correlation": NEWS_CORRELATION,
        "signal_correlation": signal_correlation,
        "theory_sharpe_revised": math.sqrt(PERIODS)
        * signal_correlation
        * math.sqrt(2 / math.pi)
        / math.sqrt(1 - (2 / math.pi) * signal_correlation**2),
        "revised": summary(revised),
        "first_release": summary(first),
    }


def survivorship(rng: np.random.Generator) -> dict:
    days = YEARS * PERIODS
    rows = {
        "dip_point_in_time": [],
        "dip_survivors": [],
        "hold_point_in_time": [],
        "hold_survivors": [],
    }
    delisted_share = []
    for _ in range(SURVIVOR_RUNS):
        r = rng.normal(0.0, ASSET_DAILY_VOL, (ASSETS, days))
        price = np.cumprod(1 + r, axis=1)
        hit = price <= DELIST_LEVEL
        gone = np.cumsum(hit, axis=1) > 0  # delisted at this close or an earlier one
        alive_before = np.concatenate(
            [np.ones((ASSETS, 1), bool), ~gone[:, :-1]], axis=1
        )  # tradable over this day
        r = np.where(alive_before, r, 0.0)
        price_before = np.concatenate([np.ones((ASSETS, 1)), price[:, :-1]], axis=1)
        dip = alive_before & (price_before <= DIP_LEVEL)
        survivor = ~gone[:, -1]
        delisted_share.append(1 - survivor.mean())

        def portfolio(mask: np.ndarray) -> np.ndarray:
            count = mask.sum(axis=0)
            return np.where(
                count > 0, (r * mask).sum(axis=0) / np.maximum(count, 1), 0.0
            )

        for name, mask in (
            ("dip_point_in_time", dip),
            ("dip_survivors", dip & survivor[:, None]),
            ("hold_point_in_time", alive_before),
            ("hold_survivors", alive_before & survivor[:, None]),
        ):
            rows[name].append(
                (portfolio(mask).mean() * PERIODS * 100, sharpe(portfolio(mask)))
            )
    out = {
        "runs": SURVIVOR_RUNS,
        "assets": ASSETS,
        "years": YEARS,
        "delisted_share": float(np.mean(delisted_share)),
    }
    for name, values in rows.items():
        v = np.array(values)
        out[name] = {"annual_return_pct": summary(v[:, 0]), "sharpe": summary(v[:, 1])}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "revisions": revisions(rng), "survivorship": survivorship(rng)}
    Path(__file__).with_name("ch03_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
