"""Chapter 20's simulations. Needs numpy and scipy.

1. Volatility-targeted size under three kinds of edge, with and without a cap at full size.
2. Capped pyramiding (enter at a third, add a third after each rise of one daily standard deviation) on rules with
   no skill and with skill, in a market without trends and one with them, against a constant size at the same exposure.
3. Growth rate against bet size when the edge is estimated.
Run: python book/ch20_sizing.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import skew

SEED = 20261201
PERIODS = 252
VOL = 0.012


def sr(x: np.ndarray) -> float:
    return float(x.mean() / x.std() * math.sqrt(PERIODS))


def volatility_targeting(rng) -> dict:
    n, s, phi = 2_000_000, 0.4, 0.98
    z = np.empty(n)
    z[0] = rng.standard_normal()
    shocks = rng.standard_normal(n) * math.sqrt(1 - phi**2)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + shocks[t]
    sigma = VOL * np.exp(
        s * z - s * s / 2
    )  # true daily volatility, known a day ahead in the "known" rows
    e = rng.standard_normal(n)
    market = sigma * e
    trailing = np.sqrt(
        np.convolve(market**2, np.ones(21) / 21, mode="full")[:n]
    )  # 21-day estimate, from past days only
    trailing = np.concatenate([[VOL], trailing[:-1]])
    trailing[:21] = VOL
    target = VOL  # aim at the market's typical volatility
    lam = 1.0 / math.sqrt(PERIODS)
    k = 2.94
    noise = rng.standard_normal(n)
    edges = {
        "constant_sharpe": lam * sigma
        + market,  # expected return proportional to volatility
        "constant_mean": lam * VOL
        + market,  # the same expected return whatever the volatility
        "constant_noise": (market + k * VOL * noise > 0)
        * market,  # sees the day's return through noise of fixed size
    }
    out = {
        "theory_gain_constant_sharpe": math.exp(s * s / 2),
        "theory_gain_constant_mean": math.exp(1.5 * s * s),
    }
    for name, x in edges.items():
        row = {"fixed": sr(x)}
        for label, est in (("known_volatility", sigma), ("trailing_21_day", trailing)):
            w = target / est
            row[f"{label}_uncapped"] = sr(w * x)
            capped = np.minimum(1.0, w)
            row[f"{label}_capped_at_full"] = sr(capped * x)
            row[f"{label}_mean_capped_size"] = float(capped.mean())
        out[name] = row
    return out


def pyramid(
    close: np.ndarray, long_: np.ndarray, steps: int = 3, lookback: int = 20
) -> np.ndarray:
    """Size decided at each close, per path: a third on entry, one more third after each rise of one daily sigma."""
    paths, n = close.shape
    logret = np.diff(np.log(close), axis=1, prepend=np.log(close[:, :1]))
    size = np.zeros((paths, n))
    held = np.zeros(paths, int)
    ref = np.zeros(paths)
    for t in range(n):
        on = long_[:, t]
        enter = on & (held == 0)
        held[enter], ref[enter] = 1, close[enter, t]
        if t >= lookback:
            sigma = logret[:, t - lookback + 1 : t + 1].std(axis=1, ddof=1)
            add = on & ~enter & (held < steps) & (close[:, t] - ref >= sigma * ref)
            held[add] += 1
            ref[add] = close[add, t]
        held[~on] = 0
        size[:, t] = held / steps
    return size


def regime_market(
    rng, paths: int, n: int, drift: float = 0.003, stay: float = 39 / 40
) -> np.ndarray:
    state = rng.integers(0, 3, paths)
    noise_sd = math.sqrt(VOL**2 - (2 / 3) * drift**2)
    r = np.empty((paths, n))
    for t in range(n):
        move = rng.random(paths) >= stay
        state = np.where(move, (state + rng.integers(1, 3, paths)) % 3, state)
        r[:, t] = drift * np.array([1, 0, -1])[state] + noise_sd * rng.standard_normal(
            paths
        )
    return r


def pyramiding(rng) -> dict:
    paths, n, out = 6_000, 504, {}
    markets = {
        "no_trends": rng.normal(0, VOL, (paths, n)),
        "trends": regime_market(rng, paths, n),
    }
    for mname, r in markets.items():
        close = 100 * np.cumprod(1 + r, axis=1)
        nxt = np.concatenate([r[:, 1:], np.zeros((paths, 1))], axis=1)
        smooth = np.apply_along_axis(
            lambda v: np.convolve(v, np.ones(20) / 20, mode="same"),
            1,
            rng.standard_normal((paths, n)),
        )
        rules = {
            "no_skill": smooth > 0,
            "one_bar_edge": (nxt + 2.94 * VOL * rng.standard_normal((paths, n))) > 0,
        }
        for rname, on in rules.items():
            sizes = {"fixed": on.astype(float), "pyramided": pyramid(close, on)}
            avg = np.where(
                on.sum(axis=1) > 0,
                (sizes["pyramided"] * on).sum(axis=1) / np.maximum(on.sum(axis=1), 1),
                0.0,
            )
            sizes["matched"] = on * avg[:, None]
            row = {"mean_pyramid_size_while_long": float(avg.mean())}
            for arm, w in sizes.items():
                x = w[:, :-1] * r[:, 1:]
                sd = x.std(axis=1, ddof=1)
                ok = sd > 0
                sharpe = x[ok].mean(axis=1) / sd[ok] * math.sqrt(PERIODS)
                month = (
                    x[ok][:, : (x.shape[1] // 21) * 21]
                    .reshape(ok.sum(), -1, 21)
                    .sum(axis=2)
                )
                row[arm] = {
                    "mean_sharpe": float(sharpe.mean()),
                    "median_daily_skew": float(np.median(skew(x[ok], axis=1))),
                    "median_monthly_skew": float(np.median(skew(month, axis=1))),
                }
            out[f"{mname}_{rname}"] = row
    return out


def kelly() -> dict:
    """Growth rate, relative to the best possible, of betting a multiple of the Kelly size computed from an estimate."""
    out = {}
    for multiple in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
        out[str(multiple)] = 2 * multiple - multiple**2
    out["estimated_sharpe_over_true"] = {
        str(ratio): 2 * ratio - ratio**2 for ratio in (1.0, 1.5, 2.0, 3.0)
    }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "volatility_targeting": volatility_targeting(rng),
        "pyramiding": pyramiding(rng),
        "kelly_relative_growth": kelly(),
    }
    Path(__file__).with_name("ch20_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    def r(x):
        return (
            {k: r(v) for k, v in x.items()}
            if isinstance(x, dict)
            else (round(x, 3) if isinstance(x, float) else x)
        )

    print(json.dumps(r(out)).replace("}, ", "},\n"))


if __name__ == "__main__":
    main()
