"""Chapter 7's simulations. Needs numpy and scipy.

1. The chance of a 10% fall within 63 sessions at constant volatility: closed form against simulation.
2. A market with volatility clustering and no drift: how the chance of a fall varies with trailing volatility, and a
   logistic model on log volatility, fitted on one half and scored on the other.
3. Blending a signal that works with one that points the wrong way.
4. The sampling error of an AUC when forecasts overlap: across histories, by the independent-sample formula, by blocks.
Run: python book/ch07_crash_risk.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

SEED = 20261118
PERIODS = 252
HORIZON = 63
FALL = 0.10
STEP = 5
WINDOW = 21
BARRIER = -math.log(1 - FALL)


def first_passage(annual_vol: float, horizon: int = HORIZON) -> float:
    """P(a driftless log price falls by BARRIER within the horizon), observed continuously."""
    return 2 * norm.cdf(-BARRIER / (annual_vol * math.sqrt(horizon / PERIODS)))


def auc(scores: np.ndarray, events: np.ndarray) -> float:
    pos, neg = scores[events], scores[~events]
    ranks = rankdata(np.concatenate([pos, neg]))
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def constant_volatility(rng) -> dict:
    out = {}
    for vol in (0.10, 0.20, 0.40):
        r = rng.normal(0, vol / math.sqrt(PERIODS), (200_000, HORIZON))
        out[str(vol)] = {
            "theory": first_passage(vol),
            "simulated_daily_closes": float(
                (np.cumsum(r, axis=1).min(axis=1) <= -BARRIER).mean()
            ),
        }
    return out


def garch_paths(
    rng, paths: int, days: int, alpha=0.08, beta=0.90, annual_vol=0.16
) -> np.ndarray:
    """Log returns from a GARCH(1,1) with no drift and the given unconditional volatility."""
    uncond = annual_vol**2 / PERIODS
    omega = uncond * (1 - alpha - beta)
    var = np.full(paths, uncond)
    out = np.empty((paths, days))
    for t in range(days):
        r = np.sqrt(var) * rng.standard_normal(paths)
        out[:, t] = r
        var = omega + alpha * r**2 + beta * var
    return out


def weekly_points(r: np.ndarray):
    """Per path: log trailing volatility at each weekly decision day, and whether a 10% fall followed within the horizon."""
    paths, days = r.shape
    log_price = np.cumsum(r, axis=1)
    idx = np.arange(WINDOW, days - HORIZON, STEP)
    x = np.empty((paths, len(idx)))
    fell = np.empty((paths, len(idx)), bool)
    for j, i in enumerate(idx):
        x[:, j] = np.log(
            r[:, i - WINDOW + 1 : i + 1].std(axis=1, ddof=1) * math.sqrt(PERIODS)
        )
        fell[:, j] = (
            log_price[:, i + 1 : i + 1 + HORIZON].min(axis=1) - log_price[:, i]
        ) <= -BARRIER
    return x, fell


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    a, b = 0.0, 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a + b * x)))
        w = p * (1 - p)
        g = np.array([(y - p).sum(), ((y - p) * x).sum()])
        h = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
        step = np.linalg.solve(h, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def clustering(rng) -> dict:
    r = garch_paths(rng, 400, 25_200)
    x, fell = weekly_points(r)
    half = x.shape[0] // 2
    xt, yt, xs, ys = (
        x[:half].ravel(),
        fell[:half].ravel(),
        x[half:].ravel(),
        fell[half:].ravel(),
    )
    a, b = logistic_fit(xt, yt.astype(float))
    p = 1 / (1 + np.exp(-(a + b * xs)))
    base = yt.mean()
    brier, brier_base = float(((p - ys) ** 2).mean()), float(((base - ys) ** 2).mean())
    buckets = []
    edges = np.quantile(xs, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xs >= lo) & (xs <= hi)
        vol = float(np.exp(np.median(xs[m])))
        buckets.append(
            {
                "median_trailing_vol": vol,
                "event_rate": float(ys[m].mean()),
                "model_probability": float(p[m].mean()),
                "constant_vol_formula": first_passage(vol),
            }
        )
    bands = []
    for lo, hi in zip((0, 0.05, 0.1, 0.2, 0.3, 0.5), (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)):
        m = (p >= lo) & (p < hi)
        if m.sum() > 200:
            bands.append(
                {
                    "from": lo,
                    "to": hi,
                    "share_of_points": float(m.mean()),
                    "predicted": float(p[m].mean()),
                    "event_rate": float(ys[m].mean()),
                }
            )
    reliability = sum(
        b_["share_of_points"] * (b_["predicted"] - b_["event_rate"]) ** 2
        for b_ in bands
    )
    return {
        "base_rate": float(ys.mean()),
        "intercept": a,
        "slope_on_log_vol": b,
        "auc": auc(xs, ys),
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill": 1 - brier / brier_base,
        "reliability_term": reliability,
        "by_trailing_vol_quintile": buckets,
        "calibration": bands,
    }


def blend(rng) -> dict:
    n, base = 400_000, 0.15
    events = rng.random(n) < base
    d_good = math.sqrt(2) * norm.ppf(0.67)
    good = rng.standard_normal(n) + d_good * events
    wrong = rng.standard_normal(n) - d_good * events
    mixed = 0.6 * wrong + 0.4 * good
    d_mix = (0.4 * d_good - 0.6 * d_good) / math.sqrt(0.6**2 + 0.4**2)
    return {
        "auc_good": auc(good, events),
        "auc_wrong_way": auc(wrong, events),
        "auc_blend_60_wrong_40_good": auc(mixed, events),
        "theory_blend": float(norm.cdf(d_mix / math.sqrt(2))),
        "auc_equal_weight_of_two_good": auc(
            good + (rng.standard_normal(n) + d_good * events), events
        ),
        "theory_two_good": float(norm.cdf(d_good * math.sqrt(2) / math.sqrt(2))),
    }


def overlap(rng) -> dict:
    """Thirty-year histories; the score is log trailing volatility."""
    r = garch_paths(rng, 300, 30 * PERIODS)
    x, fell = weekly_points(r)
    keep = [k for k in range(x.shape[0]) if 30 <= fell[k].sum() <= fell.shape[1] - 30]
    aucs = np.array([auc(x[k], fell[k]) for k in keep])
    naive = []
    for k in keep:  # Hanley and McNeil's standard error for independent points
        a, n1, n0 = auc(x[k], fell[k]), fell[k].sum(), (~fell[k]).sum()
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        naive.append(
            math.sqrt(
                (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a))
                / (n1 * n0)
            )
        )
    block, blocked = HORIZON // STEP, []
    for k in keep[:60]:
        m = x.shape[1]
        starts = np.arange(0, m - block + 1)
        vals = []
        for _ in range(200):
            idx = np.concatenate(
                [
                    np.arange(s, s + block)
                    for s in rng.choice(starts, size=math.ceil(m / block))
                ]
            )[:m]
            if 0 < fell[k][idx].sum() < m:
                vals.append(auc(x[k][idx], fell[k][idx]))
        blocked.append(np.std(vals, ddof=1))
    return {
        "histories": len(keep),
        "points_per_history": int(x.shape[1]),
        "mean_auc": float(aucs.mean()),
        "sd_across_histories": float(aucs.std(ddof=1)),
        "mean_independent_sample_se": float(np.mean(naive)),
        "mean_block_bootstrap_se": float(np.mean(blocked)),
        "block_points": block,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "constant_volatility": constant_volatility(rng),
        "clustering": clustering(rng),
        "blend": blend(rng),
        "overlap": overlap(rng),
    }
    Path(__file__).with_name("ch07_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
