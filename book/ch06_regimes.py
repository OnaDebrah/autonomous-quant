"""Chapter 6's simulations. Needs only numpy.

1. Filtered against smoothed state probabilities, with the model's parameters known: how often each names the true
   state, and what a rule that trades on each earns. A drift-regime market and a volatility-regime market.
2. Detection delay of a CUSUM at a fixed false-alarm rate: a change in volatility against a change in drift.
3. Waiting time of a rule that needs m consecutive closes.
Run: python book/ch06_regimes.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261116
PERIODS = 252
BARS = 504
DAILY_VOL = 0.012
STAY = 39 / 40
RUNS = 2_000


def markov_path(rng, runs: int, bars: int, states: int, stay: float) -> np.ndarray:
    z = np.empty((runs, bars), dtype=int)
    z[:, 0] = rng.integers(0, states, runs)
    for t in range(1, bars):
        move = rng.random(runs) >= stay
        step = rng.integers(1, states, runs)
        z[:, t] = np.where(move, (z[:, t - 1] + step) % states, z[:, t - 1])
    return z


def forward_backward(r: np.ndarray, means: np.ndarray, sds: np.ndarray, stay: float):
    """Filtered P(Z_t | r_1..t) and smoothed P(Z_t | r_1..T) for each run, parameters known."""
    runs, bars = r.shape
    k = len(means)
    transition = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(transition, stay)
    density = np.exp(-0.5 * ((r[:, :, None] - means) / sds) ** 2) / sds
    filtered = np.empty((runs, bars, k))
    predicted = np.empty((runs, bars, k))
    prior = np.full((runs, k), 1 / k)
    for t in range(bars):
        predicted[:, t] = prior
        post = prior * density[:, t]
        post /= post.sum(axis=1, keepdims=True)
        filtered[:, t] = post
        prior = post @ transition
    smoothed = np.empty_like(filtered)
    smoothed[:, -1] = filtered[:, -1]
    for t in range(bars - 2, -1, -1):
        ratio = smoothed[:, t + 1] / predicted[:, t + 1]
        smoothed[:, t] = filtered[:, t] * (ratio @ transition.T)
    return filtered, smoothed


def sharpe(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    return np.where(
        sd > 0, x.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(PERIODS), 0.0
    )


def filter_against_smoother(rng) -> dict:
    out = {}
    # Drift regimes: up, flat, down; one volatility.
    drift = 0.003
    noise = math.sqrt(DAILY_VOL**2 - (2 / 3) * drift**2)
    means, sds = np.array([drift, 0.0, -drift]), np.full(3, noise)
    z = markov_path(rng, RUNS, BARS, 3, STAY)
    r = means[z] + noise * rng.standard_normal((RUNS, BARS))
    filtered, smoothed = forward_backward(r, means, sds, STAY)
    row = {}
    for name, p in (("filtered", filtered), ("smoothed", smoothed)):
        long_next = (p.argmax(axis=2) == 0)[:, :-1]  # decided at t, held over t+1
        row[name] = {
            "names_true_state": float((p.argmax(axis=2) == z).mean()),
            "mean_sharpe": float(sharpe(long_next * r[:, 1:]).mean()),
        }
    row["true_state_known"] = {
        "names_true_state": 1.0,
        "mean_sharpe": float(sharpe((z == 0)[:, :-1] * r[:, 1:]).mean()),
    }
    row["buy_and_hold"] = {"mean_sharpe": float(sharpe(r[:, 1:]).mean())}
    out["drift_regimes"] = row

    # Volatility regimes: calm and stormy; no drift.
    sds = np.array([0.008, 0.020])
    means = np.zeros(2)
    z = markov_path(rng, RUNS, BARS, 2, STAY)
    r = sds[z] * rng.standard_normal((RUNS, BARS))
    filtered, smoothed = forward_backward(r, means, sds, STAY)
    out["volatility_regimes"] = {
        name: {"names_true_state": float((p.argmax(axis=2) == z).mean())}
        for name, p in (("filtered", filtered), ("smoothed", smoothed))
    }
    return out


def cusum_run_length(
    rng, increments, threshold: float, runs: int, horizon: int
) -> np.ndarray:
    """First time the CUSUM of the given log-likelihood-ratio increments reaches the threshold (horizon if never)."""
    g = np.zeros(runs)
    first = np.full(runs, horizon, dtype=float)
    live = np.ones(runs, bool)
    for t in range(1, horizon + 1):
        g = np.maximum(0.0, g + increments(rng, runs))
        hit = live & (g >= threshold)
        first[hit] = t
        live &= ~hit
        if not live.any():
            break
    return first


def detection_delay(rng) -> dict:
    target = 500.0  # sessions between false alarms
    out = {}
    cases = {
        # log likelihood ratio of N(0, (2s)^2) against N(0, s^2), for x in units of s
        "volatility_doubles": (
            lambda g, n: -math.log(2) + 0.375 * g.standard_normal(n) ** 2,
            lambda g, n: -math.log(2) + 0.375 * (2 * g.standard_normal(n)) ** 2,
            0.5 * (4 - 1 - math.log(4)),
        ),
        # log likelihood ratio of N(d, 1) against N(0, 1), d = 0.25 (0.3% a day against 1.2% volatility)
        "drift_turns_up": (
            lambda g, n: 0.25 * g.standard_normal(n) - 0.25**2 / 2,
            lambda g, n: 0.25 * (0.25 + g.standard_normal(n)) - 0.25**2 / 2,
            0.25**2 / 2,
        ),
    }
    for name, (before, after, information) in cases.items():
        low, high = 0.5, 12.0
        for _ in range(18):  # bisect the threshold to the target false-alarm spacing
            mid = (low + high) / 2
            spacing = cusum_run_length(rng, before, mid, 1_500, 20_000).mean()
            low, high = (mid, high) if spacing < target else (low, mid)
        threshold = (low + high) / 2
        delay = cusum_run_length(rng, after, threshold, 4_000, 20_000)
        out[name] = {
            "information_per_day": information,
            "threshold": threshold,
            "false_alarm_spacing": float(
                cusum_run_length(rng, before, threshold, 3_000, 20_000).mean()
            ),
            "mean_delay_days": float(delay.mean()),
            "median_delay_days": float(np.median(delay)),
            "first_order_theory_days": math.log(target) / information,
        }
    return out


def consecutive(rng) -> dict:
    out = {}
    for prob, need in ((0.9, 3), (0.8, 3), (0.9, 2), (0.6, 3)):
        runs, wait = 20_000, []
        draws = rng.random((runs, 400)) < prob
        for row in draws:
            streak = 0
            for t, ok in enumerate(row, 1):
                streak = streak + 1 if ok else 0
                if streak == need:
                    wait.append(t)
                    break
        out[f"{prob}_{need}"] = {
            "theory_days": (prob**-need - 1) / (1 - prob),
            "mean_days": float(np.mean(wait)),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "filter_against_smoother": filter_against_smoother(rng),
        "detection_delay": detection_delay(rng),
        "consecutive": consecutive(rng),
    }
    Path(__file__).with_name("ch06_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
