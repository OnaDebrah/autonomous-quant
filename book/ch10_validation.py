"""Chapter 10's simulations. Needs numpy and scipy.

1. Scoring volatility forecasts against a noisy proxy: which loss functions still prefer the true forecast.
2. Many instruments tested at once: a fixed threshold against the Benjamini-Hochberg procedure.
3. A calibration tolerance per band: how often a perfectly calibrated forecaster fails it.
4. The population stability index when nothing has drifted.
5. Pooling forecasts across symbols on the same dates: the usual standard error against the truth.
6. How long it takes to notice that an instrument has lost its skill.
Run: python book/ch10_validation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2, norm

SEED = 20261121


def proxy_losses(rng) -> dict:
    n = 2_000_000
    true_var = np.exp(rng.normal(0, 0.5, n))  # the variance changes from day to day
    proxy = (
        true_var * rng.standard_normal(n) ** 2
    )  # squared return: unbiased, very noisy
    forecasts = {
        "true_variance": true_var,
        "half_the_true_variance": 0.5 * true_var,
        "one_and_a_half_times": 1.5 * true_var,
    }
    out = {}
    for name, h in forecasts.items():
        ratio = proxy / h
        out[name] = {
            "qlike": float((ratio - np.log(np.maximum(ratio, 1e-300)) - 1).mean()),
            "mse_variance": float(((proxy - h) ** 2).mean()),
            "mae_variance": float(np.abs(proxy - h).mean()),
            "mse_volatility": float(((np.sqrt(proxy) - np.sqrt(h)) ** 2).mean()),
        }
    return out


def many_instruments(rng) -> dict:
    runs, m, real, effect, alpha = 20_000, 100, 10, 3.0, 0.05
    t = rng.standard_normal((runs, m))
    t[:, :real] += effect
    p = norm.sf(t)
    fixed = p < alpha
    order = np.argsort(p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    passed = sorted_p <= alpha * np.arange(1, m + 1) / m
    k = np.where(passed.any(axis=1), m - np.argmax(passed[:, ::-1], axis=1), 0)
    bh = np.zeros_like(fixed)
    for i in range(runs):
        bh[i, order[i, : k[i]]] = True

    def summary(sel):
        false = sel[:, real:].sum(axis=1)
        total = sel.sum(axis=1)
        return {
            "mean_discoveries": float(total.mean()),
            "mean_false": float(false.mean()),
            "false_discovery_rate": float((false / np.maximum(total, 1)).mean()),
            "power": float(sel[:, :real].mean()),
        }

    none_real = norm.sf(rng.standard_normal((runs, m)))
    return {
        "fixed_threshold": summary(fixed),
        "benjamini_hochberg": summary(bh),
        "fixed_threshold_any_pass_when_none_real": float(
            (none_real < alpha).any(axis=1).mean()
        ),
    }


def calibration_tolerance() -> dict:
    out = {}
    probs = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    for n in (30, 100, 300):
        fails = []
        for p in probs:
            lo, hi = math.ceil((p - 0.10) * n - 1e-9), math.floor((p + 0.10) * n + 1e-9)
            fails.append(
                1 - (binom.cdf(hi, n, p) - (binom.cdf(lo - 1, n, p) if lo > 0 else 0))
            )
        out[str(n)] = {
            "per_band": {str(p): float(f) for p, f in zip(probs, fails)},
            "any_of_six_bands": float(1 - np.prod([1 - f for f in fails])),
        }
    return out


def calibration_rule() -> dict:
    """The two-part rule: a band fails only if it is off by more than 10 points and by more than z standard errors,
    with z = 2.64 for six bands. Exact binomial probabilities that at least one of six bands fails, for a calibrated
    forecaster and for two overconfident ones whose true probabilities lie a share s of the way from the base rate
    (the mean forecast) to the forecast: s = 0.5, and s = 0, a forecaster with no resolution at all."""
    probs = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    z = float(norm.ppf(1 - 0.05 / 12))
    base = float(np.mean(probs))

    def fail_any(n, true):
        ok = 1.0
        for p, q in zip(probs, true):
            ks = np.arange(n + 1)
            dev = np.abs(ks / n - p)
            fail = (dev > 0.10 + 1e-12) & (dev > z * math.sqrt(p * (1 - p) / n) + 1e-12)
            ok *= 1 - binom.pmf(ks, n, q)[fail].sum()
        return float(1 - ok)

    out = {"z": z}
    for n in (30, 100):
        out[str(n)] = {
            "calibrated": fail_any(n, probs),
            "halfway_to_base_rate": fail_any(n, [base + 0.5 * (p - base) for p in probs]),
            "no_resolution": fail_any(n, [base] * len(probs)),
        }
    return out


def stability_index(rng) -> dict:
    out, bins, runs = {}, 10, 20_000
    edges = norm.ppf(np.linspace(0, 1, bins + 1)[1:-1])
    for n in (100, 500, 5000):
        ref = (
            np.stack(
                [
                    np.bincount(
                        np.searchsorted(edges, rng.standard_normal(n)), minlength=bins
                    )
                    for _ in range(runs)
                ]
            )
            / n
        )
        new = (
            np.stack(
                [
                    np.bincount(
                        np.searchsorted(edges, rng.standard_normal(n)), minlength=bins
                    )
                    for _ in range(runs)
                ]
            )
            / n
        )
        ref, new = np.maximum(ref, 0.5 / n), np.maximum(new, 0.5 / n)
        psi = ((new - ref) * np.log(new / ref)).sum(axis=1)
        out[str(n)] = {
            "mean": float(psi.mean()),
            "theory_mean": (bins - 1) * 2 / n,
            "p95": float(np.percentile(psi, 95)),
            "theory_p95": float(chi2.ppf(0.95, bins - 1) * 2 / n),
            "share_above_0.10": float((psi > 0.10).mean()),
            "share_above_0.25": float((psi > 0.25).mean()),
        }
    return out


def pooled(rng) -> dict:
    runs, dates, symbols, rho = 4_000, 120, 50, 0.3
    common = rng.standard_normal((runs, dates, 1))
    d = math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.standard_normal(
        (runs, dates, symbols)
    )
    means = d.mean(axis=(1, 2))
    naive = d.reshape(runs, -1).std(axis=1, ddof=1) / math.sqrt(dates * symbols)
    by_date = d.mean(axis=2)
    clustered = by_date.std(axis=1, ddof=1) / math.sqrt(dates)
    return {
        "true_sd_of_pooled_mean": float(means.std(ddof=1)),
        "mean_naive_se": float(naive.mean()),
        "mean_clustered_se": float(clustered.mean()),
        "naive_rejects_at_5pct_when_nothing_is_there": float(
            (np.abs(means / naive) > 1.96).mean()
        ),
        "clustered_rejects": float((np.abs(means / clustered) > 1.98).mean()),
        "effective_symbols": symbols / (1 + (symbols - 1) * rho),
    }


def losing_skill(rng) -> dict:
    """Loss difference (baseline minus instrument) has mean delta while the instrument works and zero after."""
    out, target = {}, 240.0  # forecasts between false alarms
    for delta in (0.2, 0.3, 0.5):

        def run(mean, threshold, runs, horizon=20_000):
            g = np.zeros(runs)
            first = np.full(runs, float(horizon))
            live = np.ones(runs, bool)
            for t in range(1, horizon + 1):
                x = mean + rng.standard_normal(runs)
                g = np.maximum(
                    0.0, g + (-delta * x + delta**2 / 2)
                )  # log likelihood ratio of N(0,1) against N(delta,1)
                hit = live & (g >= threshold)
                first[hit] = t
                live &= ~hit
                if not live.any():
                    break
            return first

        low, high = 0.2, 10.0
        for _ in range(16):
            mid = (low + high) / 2
            low, high = (
                (mid, high) if run(delta, mid, 800).mean() < target else (low, mid)
            )
        threshold = (low + high) / 2
        delay = run(0.0, threshold, 4_000)
        out[str(delta)] = {
            "t_statistic_after_120_forecasts": delta * math.sqrt(120),
            "mean_forecasts_to_notice": float(delay.mean()),
            "median": float(np.median(delay)),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "proxy_losses": proxy_losses(rng),
        "many_instruments": many_instruments(rng),
        "calibration_tolerance": calibration_tolerance(),
        "calibration_rule": calibration_rule(),
        "stability_index": stability_index(rng),
        "pooled": pooled(rng),
        "losing_skill": losing_skill(rng),
    }
    Path(__file__).with_name("ch10_results.json").write_text(
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
