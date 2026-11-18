"""Chapter 5's simulations. Needs only numpy and scipy.

1. The sector score (window return over window volatility) on sectors with no differences between them.
2. Rank stability from overlapping windows alone.
3. The return and information ratio of a top-quantile portfolio against the closed forms.
4. An average of percentiles over the signals a stock has: who reaches the top when coverage differs.
5. How many months a rank correlation needs before skill is visible, with 11 assets and with 500.
Run: python book/ch05_scanning.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

SEED = 20261115
SECTORS = 11
WINDOW = 126
DAILY_VOL = 0.012


def sector_score(prices: np.ndarray) -> np.ndarray:
    """Window return over window volatility, per row of prices (the scanner's score)."""
    returns = prices[:, 1:] / prices[:, :-1] - 1
    return (prices[:, -1] / prices[:, 0] - 1) / (
        returns.std(axis=1, ddof=1) * math.sqrt(returns.shape[1])
    )


def null_scores(rng: np.random.Generator) -> dict:
    out = {}
    runs = 20_000
    for rho in (0.0, 0.6):
        common = rng.normal(0, 1, (runs, 1, WINDOW))
        own = rng.normal(0, 1, (runs, SECTORS, WINDOW))
        vols = DAILY_VOL * rng.uniform(
            0.6, 1.8, (runs, SECTORS, 1)
        )  # sectors differ in volatility
        r = vols * (math.sqrt(rho) * common + math.sqrt(1 - rho) * own)
        prices = np.concatenate(
            [np.ones((runs, SECTORS, 1)), np.cumprod(1 + r, axis=2)], axis=2
        )
        scores = np.stack([sector_score(prices[k]) for k in range(runs)])
        out[str(rho)] = {
            "score_sd": float(scores.std()),
            "mean_top": float(scores.max(axis=1).mean()),
            "mean_top_minus_bottom": float(
                (scores.max(axis=1) - scores.min(axis=1)).mean()
            ),
            "mean_top_minus_average": float(
                (scores.max(axis=1) - scores.mean(axis=1)).mean()
            ),
        }
    draws = rng.standard_normal((400_000, SECTORS))
    out["expected_max_of_11_normals"] = float(draws.max(axis=1).mean())
    return out


def overlap(rng: np.random.Generator) -> dict:
    """Scores on windows that overlap, with no persistence in returns."""
    runs, out = 4_000, {}
    for step in (1, 5, 21, 63, 126):
        r = rng.normal(0, DAILY_VOL, (runs, SECTORS, WINDOW + step))
        first = r[:, :, :WINDOW].sum(axis=2) / (
            r[:, :, :WINDOW].std(axis=2, ddof=1) * math.sqrt(WINDOW)
        )
        second = r[:, :, step:].sum(axis=2) / (
            r[:, :, step:].std(axis=2, ddof=1) * math.sqrt(WINDOW)
        )
        spearman = [
            np.corrcoef(rankdata(a), rankdata(b))[0, 1] for a, b in zip(first, second)
        ]
        kept = [
            len(set(np.argsort(-a)[:3]) & set(np.argsort(-b)[:3])) / 3
            for a, b in zip(first, second)
        ]
        out[str(step)] = {
            "theory_score_correlation": 1 - step / WINDOW,
            "mean_rank_correlation": float(np.mean(spearman)),
            "top3_kept": float(np.mean(kept)),
        }
    return out


def quantile_portfolio(rng: np.random.Generator) -> dict:
    """Top fifth, equal weights, against the equal-weight universe: monthly, independent residuals."""
    months, q, dispersion, out = 6_000, 0.2, 0.08, {}
    h = norm.pdf(norm.ppf(1 - q)) / q
    for names, ic in ((500, 0.02), (500, 0.05), (100, 0.05)):
        z = rng.standard_normal((months, names))
        ret = dispersion * (
            ic * z + math.sqrt(1 - ic**2) * rng.standard_normal((months, names))
        )
        k = int(names * q)
        top = np.argpartition(-z, k, axis=1)[:, :k]
        active = np.take_along_axis(ret, top, axis=1).mean(axis=1) - ret.mean(axis=1)
        out[f"{names}_{ic}"] = {
            "theory_mean_active": ic * dispersion * h,
            "mean_active": float(active.mean()),
            "theory_annual_ir": ic * h * math.sqrt(q * names / (1 - q)) * math.sqrt(12),
            "annual_ir": float(active.mean() / active.std(ddof=1) * math.sqrt(12)),
        }
    out["h"] = h
    return out


def coverage(rng: np.random.Generator) -> dict:
    """Composite = mean percentile over the signals a stock has. No signal means anything."""
    runs, names, signals, thin_share, thin_signals = 2_000, 500, 6, 0.3, 2
    thin = int(names * thin_share)
    top_thin, top_thin_scaled = [], []
    for _ in range(runs):
        pct = rng.uniform(0, 1, (names, signals))
        have = np.ones((names, signals), bool)
        have[:thin, thin_signals:] = (
            False  # the first `thin` stocks have only two signals
        )
        count = have.sum(axis=1)
        composite = np.where(have, pct, 0).sum(axis=1) / count
        scaled = (composite - 0.5) * np.sqrt(
            12 * count
        )  # the same composite in standard-error units
        k = names // 10
        top_thin.append((np.argsort(-composite)[:k] < thin).mean())
        top_thin_scaled.append((np.argsort(-scaled)[:k] < thin).mean())
    return {
        "thin_share_of_universe": thin_share,
        "signals": signals,
        "thin_signals": thin_signals,
        "thin_share_of_top_decile": float(np.mean(top_thin)),
        "thin_share_of_top_decile_scaled": float(np.mean(top_thin_scaled)),
    }


def ic_power(rng: np.random.Generator) -> dict:
    out, runs = {}, 4_000
    for names, ic, months in (
        (11, 0.05, 120),
        (11, 0.10, 120),
        (500, 0.02, 120),
        (500, 0.05, 60),
    ):
        z = rng.standard_normal((runs, months, names))
        ret = ic * z + math.sqrt(1 - ic**2) * rng.standard_normal((runs, months, names))
        zr = np.apply_along_axis(rankdata, 2, z)
        rr = np.apply_along_axis(rankdata, 2, ret)
        zr -= zr.mean(axis=2, keepdims=True)
        rr -= rr.mean(axis=2, keepdims=True)
        ics = (zr * rr).sum(axis=2) / np.sqrt((zr**2).sum(axis=2) * (rr**2).sum(axis=2))
        t = ics.mean(axis=1) / (ics.std(axis=1, ddof=1) / math.sqrt(months))
        out[f"{names}_{ic}_{months}"] = {
            "ic_sd": float(ics.std()),
            "theory_ic_sd": 1 / math.sqrt(names - 1),
            "theory_t": ic * math.sqrt(months * (names - 1)),
            "mean_t": float(t.mean()),
            "share_t_above_2": float((t >= 2).mean()),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "null_scores": null_scores(rng),
        "overlap": overlap(rng),
        "quantile_portfolio": quantile_portfolio(rng),
        "coverage": coverage(rng),
        "ic_power": ic_power(rng),
    }
    Path(__file__).with_name("ch05_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
