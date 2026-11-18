"""Data for the book's plotted figures, written to figdata/ as whitespace-separated tables for pgfplots.

Each table comes from a chapter's own results file, its stated formulas, or a rerun of its own script with the
same seed and the same order of random draws. A rerun first reproduces the chapter's saved summary from those
draws and stops if it differs, so a plotted path belongs to the experiment behind the book's numbers.

Run from this folder:  python figures_data.py
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).parent
OUT = HERE / "figdata"
GAMMA = 0.5772156649015329  # Euler-Mascheroni


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def results(chapter: str) -> dict:
    return json.loads((HERE / f"ch{chapter}_results.json").read_text())


def write(name: str, header: list[str], rows) -> None:
    lines = [" ".join(header)] + [" ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) for row in rows]
    (OUT / f"{name}.dat").write_text("\n".join(lines) + "\n")


def same(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise SystemExit(f"rerun does not reproduce the chapter's {what}: {a} != {b}")


def m_n(n: float) -> float:
    """Expected best of n independent standard normals, the approximation of Equation (13.1)."""
    if n <= 1:
        return 0.0
    return (1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * math.e))


# P1. Best annualized Sharpe ratio of a search with nothing there (Chapter 1, Table 1.1).
def best_of_search() -> None:
    rows = results("01")["rows"]
    write("p1_best_of_search", ["tries", "p5", "mean", "p95", "theory"],
          [(r["tries"], r["p5_best"], r["mean_best"], r["p95_best"], r["theory_expected_best"]) for r in rows])


# P6. Filter against smoother: run 1 of Chapter 6's calm/stormy experiment.
def filter_against_smoother() -> None:
    ch6 = load("ch06_regimes")
    rng = np.random.default_rng(ch6.SEED)
    # The drift experiment comes first in the chapter's script; consume its draws in the same order.
    z = ch6.markov_path(rng, ch6.RUNS, ch6.BARS, 3, ch6.STAY)
    rng.standard_normal((ch6.RUNS, ch6.BARS))
    # The volatility experiment.
    sds, means = np.array([0.008, 0.020]), np.zeros(2)
    z = ch6.markov_path(rng, ch6.RUNS, ch6.BARS, 2, ch6.STAY)
    r = sds[z] * rng.standard_normal((ch6.RUNS, ch6.BARS))
    filtered, smoothed = ch6.forward_backward(r, means, sds, ch6.STAY)
    saved = results("06")["filter_against_smoother"]["volatility_regimes"]
    same(float((filtered.argmax(axis=2) == z).mean()), saved["filtered"]["names_true_state"], "filtered accuracy")
    same(float((smoothed.argmax(axis=2) == z).mean()), saved["smoothed"]["names_true_state"], "smoothed accuracy")
    write("p6_filter_smoother", ["day", "stormy", "filtered", "smoothed", "ret"],
          [(t, int(z[0, t]), float(filtered[0, t, 1]), float(smoothed[0, t, 1]), float(r[0, t])) for t in range(ch6.BARS)])


# P11. The scorecard plane (Chapter 12).
def scorecard() -> None:
    res = results("12")
    rows = []
    for setting in ("with_costs", "without_costs"):
        for judge in ("naive_above_1", "single_test_95", "deflated_95"):
            row = res[setting][judge]
            rows.append((setting, judge, 100 * row["false_pass_rate"], 100 * row["power"]))
    write("p11_scorecard", ["setting", "judge", "false_pass_pct", "power_pct"], rows)


# P12. The bar against the number of tries, and power beneath it (Chapter 13, Proposition 13.4).
def bar_and_power() -> None:
    z95, sharpe = norm.ppf(0.95), 1.5
    rows = []
    for n in np.unique(np.round(np.logspace(0, 4, 81)).astype(int)):
        m = m_n(n)
        c = m + z95
        rows.append((int(n), m / math.sqrt(2), m / math.sqrt(5), m / math.sqrt(10),
                     100 * norm.cdf(sharpe * math.sqrt(2) - c), 100 * norm.cdf(sharpe * math.sqrt(5) - c),
                     100 * norm.cdf(sharpe * math.sqrt(10) - c)))
    # Check against Table 13.3: power on two years for twenty tries at S = 1.5.
    same(round(100 * norm.cdf(1.5 * math.sqrt(2) - (m_n(20) + z95)), 1), 7.7, "power at N = 20, two years", tol=0.05)
    write("p12_bar_power", ["tries", "bar2", "bar5", "bar10", "power2", "power5", "power10"], rows)


# P13. Correlated tries move the bar both ways (Chapter 14, Table 14.3).
def correlated_bars() -> None:
    write("p13_theory", ["rho", "ratio"], [(rho, math.sqrt(1 - rho)) for rho in np.linspace(0, 0.95, 39)])
    judges = results("14")["judges"]
    rows = []
    for key, v in judges.items():
        rho, layout = key.split("_", 1)
        rows.append((float(rho), layout, v["measured_bar_ratio"], v["sign_flip_bar_ratio"]))
    write("p13_judges", ["rho", "layout", "measured", "signflip"], rows)


# P14. Looking early (Chapter 15).
def looking_early() -> None:
    looks = results("15")["repeated_looks"]
    write("p14_looks", ["looks", "false_positive_pct"], [(int(k), 100 * v) for k, v in looks.items()])


# P15. What each control buys and costs (Chapter 16, Table 16.4).
def control_prices() -> None:
    rates = results("16")["exchange_rates"]
    names = {"batch_only_count": "family-wide count", "no_holdout": "holdout", "no_deflation": "deflation", "all_off": "all controls"}
    write("p15_prices", ["control", "false_avoided", "real_given_up"],
          [(names[k].replace(" ", "_"), rates[k]["false_avoided"], rates[k]["true_lost"]) for k in names])


# P16. Allocation with delayed feedback: Chapter 17's own bandit, with each day's cumulative share recorded.
def allocation_over_time() -> None:
    ch17 = load("ch17_allocation")
    traj: dict[tuple[str, int], np.ndarray] = {}

    def bandit(rng, policy: str, days: int, runs: int) -> dict:  # ch17.bandit, plus a record of the share by day
        arms = len(ch17.PASS)
        discoveries, best_share = np.zeros(runs), np.zeros(runs)
        best = int(ch17.YIELD.argmax())
        share_by_day = np.zeros(days)
        for k in range(runs):
            s, f = np.zeros(arms), np.zeros(arms)
            pending, pulls = [], np.zeros(arms)
            for day in range(days):
                while pending and pending[0][0] <= day:
                    _, a, ok = pending.pop(0)
                    s[a] += ok
                    f[a] += 1 - ok
                for _ in range(ch17.BUDGET):
                    a = int(rng.integers(arms)) if policy == "uniform" else int(rng.beta(1 + s, 1 + f).argmax())
                    pulls[a] += 1
                    if rng.random() < ch17.PASS[a]:
                        ok = float(rng.random() < ch17.FORWARD[a])
                        discoveries[k] += ok
                        pending.append((day + ch17.DELAY, a, ok))
                    elif policy == "thompson_yield":
                        f[a] += 1
                share_by_day[day] += pulls[best] / pulls.sum()
            best_share[k] = pulls[best] / pulls.sum()
        traj[(policy, days)] = share_by_day / runs
        return {"discoveries": float(discoveries.mean()), "share_of_budget_to_best_source": float(best_share.mean())}

    ch17.bandit = bandit
    out = ch17.allocation(np.random.default_rng(ch17.SEED))
    saved = results("17")["allocation"]
    for days in ("252", "756"):
        for policy in ("uniform", "thompson_quality", "thompson_yield"):
            same(out[days][policy]["share_of_budget_to_best_source"], saved[days][policy]["share_of_budget_to_best_source"], f"{policy} share at {days} days")
    rows = [(d + 1, *(float(traj[(p, 756)][d]) for p in ("uniform", "thompson_quality", "thompson_yield"))) for d in range(756)]
    write("p16_allocation", ["day", "uniform", "thompson_quality", "thompson_yield"], rows[::3])


# P19. A pyramid and its exposure-matched twin: the first path of Chapter 20's trending market, slow rule.
def pyramid_path() -> None:
    ch20 = load("ch20_sizing")
    rng = np.random.default_rng(ch20.SEED)
    ch20.volatility_targeting(rng)  # first in the chapter's script; consume its draws
    paths, n = 6_000, 504
    markets = {"no_trends": rng.normal(0, ch20.VOL, (paths, n)), "trends": ch20.regime_market(rng, paths, n)}
    kept = {}
    for mname, r in markets.items():
        smooth = np.apply_along_axis(lambda v: np.convolve(v, np.ones(20) / 20, mode="same"), 1, rng.standard_normal((paths, n)))
        rng.standard_normal((paths, n))  # the one-bar-edge rule's noise, drawn next in the chapter's script
        kept[mname] = (r, smooth > 0)
    r, on = kept["trends"]
    close = 100 * np.cumprod(1 + r, axis=1)
    size = ch20.pyramid(close, on)
    avg = np.where(on.sum(axis=1) > 0, (size * on).sum(axis=1) / np.maximum(on.sum(axis=1), 1), 0.0)
    same(float(avg.mean()), results("20")["pyramiding"]["trends_no_skill"]["mean_pyramid_size_while_long"], "mean pyramid size")
    write("p19_pyramid", ["day", "close", "pyramid", "matched"],
          [(t, float(close[0, t]), float(size[0, t]), float(on[0, t] * avg[0])) for t in range(n)])


# P20. The expected band (Chapter 22, Proposition 22.4): Gaussian band for a backtest Sharpe ratio of 2 and one
# paper record whose true Sharpe ratio is half of it, in units of annual volatility.
def expected_band() -> None:
    backtest, real_share, days = 2.0, 0.5, 365
    rng = np.random.default_rng(load("ch22_forward").SEED)
    t = np.arange(1, days + 1) / 365
    record = np.cumsum(real_share * backtest / 365 + rng.standard_normal(days) / math.sqrt(365))
    mid, half = backtest * t, -norm.ppf(0.05) * np.sqrt(t)
    position = norm.cdf((record - mid) / np.sqrt(t))
    write("p20_band", ["day", "lo", "mid", "hi", "record", "position"],
          [(d + 1, float(mid[d] - half[d]), float(mid[d]), float(mid[d] + half[d]), float(record[d]), float(position[d])) for d in range(days)])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for make in (best_of_search, filter_against_smoother, scorecard, bar_and_power, correlated_bars, looking_early,
                 control_prices, allocation_over_time, pyramid_path, expected_band):
        make()
        print("wrote", make.__name__)
