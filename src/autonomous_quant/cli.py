"""Command line: ``aq benchmark`` scores the judges on planted truth; ``aq demo`` runs one search end to end."""

from __future__ import annotations

import argparse
import itertools
import logging
import sys

import numpy as np
import pandas as pd

from . import log as aqlog
from .evaluation import planted
from .errors import AutonomousQuantError, ConfigurationError, DataError
from .evaluation.judge import default_contract, deflated_rule, judge_contract, sign_flip_test
from .backtest.lab import Simulator, sharpe
from .data.ledger import Ledger
from .backtest.rules import CATALOGUE
from .data.sources import SimulatedSource, YahooSource
from .data.store import Store, UnknownAvailability

log = logging.getLogger("autonomous_quant.cli")


def benchmark(args: argparse.Namespace) -> None:
    world = planted.World(tries=args.tries, years=args.years, edge=args.edge, share_real=args.share_real, seed=args.seed)
    print(f"Planted truth: {args.families} families of {world.tries} tries, {world.years:g} years, "
          f"one try in {1 / world.share_real:g} families has an annual Sharpe ratio of {world.edge:g}\n")
    print(f"{'judge':<14}{'false-pass rate':>18}{'power':>10}")
    for name, row in planted.scorecard(world, args.families).items():
        print(f"{name:<14}{row['false_pass_rate']:>17.1%} {row['power']:>9.1%}")


def _prices(store: Store, symbol: str, asof: pd.Timestamp, assume: bool) -> pd.DataFrame:
    frame = pd.concat({col: store.as_of(f"{symbol}:{col}", asof, assume_available_at_observation=assume) for col in ("open", "close")}, axis=1)
    frame.index = frame.index.normalize()
    return frame.groupby(level=0).last().dropna()


def demo(args: argparse.Namespace) -> None:
    if args.years < 1:
        raise ConfigurationError("--years must be at least 1: a shorter history cannot judge a family")
    store, ledger, lab = Store(), Ledger(args.ledger), Simulator(cost_bps=args.cost_bps)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)
    source = YahooSource() if args.source == "yahoo" else SimulatedSource(assets=1, drift=args.drift, seed=args.seed)
    symbol = args.symbol if args.source == "yahoo" else "SIM0"
    store.append(source.observations([symbol], start, end))
    asof = end + pd.Timedelta(hours=23)
    try:
        prices = _prices(store, symbol, asof, assume=False)
    except UnknownAvailability as exc:
        print(f"As-of read refused: {exc}.\nContinuing with the assumption that each bar was available at its close, "
              "stated here as Chapter 3 requires.\n")
        prices = _prices(store, symbol, asof, assume=True)

    family, years = f"demo:{symbol}:{start.date()}:{end.date()}", len(prices) / 252
    rows, returns = [], []
    for rule in CATALOGUE.values():
        for values in itertools.product(*rule.region.values()):
            params = dict(zip(rule.region, values))
            spec = {"rule": rule.name, "version": rule.version, "params": params, "symbol": symbol, "interval": "1d",
                    "window": [str(prices.index[0].date()), str(prices.index[-1].date())], "control": "buy_and_hold", "cost_bps": args.cost_bps}
            trial = ledger.register(family, spec)  # before the evaluation, in its own transaction
            try:
                outcome = lab.run(rule, params, prices)
            except AutonomousQuantError as exc:  # a failed try still counts in the family
                ledger.fail(trial, f"{type(exc).__name__}: {exc}")
                continue
            equity = (1 + outcome.returns).cumprod()
            result = {"sharpe": sharpe(outcome.returns), "control_sharpe": sharpe(outcome.control_returns),
                      "round_trips": outcome.round_trips, "max_drawdown": float(1 - (equity / equity.cummax()).min())}
            ledger.complete(trial, result)
            rows.append((rule.name, params, result))
            returns.append(outcome.returns.to_numpy())

    n = ledger.family_size(family)
    if not rows:
        raise DataError(f"every one of the {n} tries in family {family} failed; see the log")
    verdict = deflated_rule([r[2]["sharpe"] for r in rows], years, family_size=n)
    name, params, best = rows[verdict["best"]]
    contract = judge_contract(default_contract(), {**best, "deflated_passed": verdict["passed"]})
    flip = sign_flip_test(np.column_stack(returns), draws=args.draws, seed=args.seed)

    print(f"{len(rows)} tries registered in family {family} (ledger count {n}), {years:.1f} years of {symbol}\n")
    print(f"{'rule':<16}{'parameters':<28}{'Sharpe':>8}{'control':>9}{'trips':>7}")
    for rule_name, p, r in sorted(rows, key=lambda x: -x[2]["sharpe"])[:8]:
        print(f"{rule_name:<16}{str(p):<28}{r['sharpe']:>8.2f}{r['control_sharpe']:>9.2f}{r['round_trips']:>7}")
    print(f"\nBest: {name} {params}, standardized {verdict['standardized']:.2f} against the deflated bar {verdict['threshold']:.2f} "
          f"for {n} tries: {'passes' if verdict['passed'] else 'does not pass'}.")
    print(f"Contract: {'passes' if contract['passed'] else 'fails on ' + ', '.join(contract['reasons'])}.")
    print(f"Sign-flip test of the family's best: p = {flip['p_value']:.3f} ({flip['draws']} draws).")


def _market(assets: int, days: int, drift: float, seed: int) -> tuple[Store, list[str], pd.DatetimeIndex]:
    names = [f"SIM{i}" for i in range(assets)]
    start = pd.Timestamp("2020-01-01")
    calendar = pd.bdate_range(start, periods=days)
    store = Store()
    store.append(SimulatedSource(assets=assets, drift=drift, seed=seed).observations(names, calendar[0], calendar[-1]))
    return store, names, calendar


def run(args: argparse.Namespace) -> None:
    from .workflow.loop import Loop
    from .evaluation.scorecard import maturity

    store, names, calendar = _market(args.assets, args.days, args.drift, args.seed)
    loop = Loop(store, names, path=args.db, slots=args.slots, seed=args.seed, research_enabled=not args.no_research)
    days = calendar[args.warmup :: args.every]
    print(f"{len(names)} simulated assets, {len(calendar)} days; ticking every {args.every} days from day {args.warmup}\n")
    for day in days:
        out = loop.tick(day)
        for s in out.get("studies", []):
            z = f"z={s['standardized']:5.2f} bar={s['threshold']:.2f}" if s["standardized"] is not None else "no completed tries"
            print(f"{out['day']}  study  {s['source']:<9}{s['rule']:<15}{s['asset']:<6} {z} "
                  f"{'PASS -> forward test' if s['passed'] else 'rejected: ' + ', '.join(s['reasons'])}")
        for f in out.get("settled", []):
            if f["status"] == "settled":
                print(f"{out['day']}  settle {f['source']:<9}{f['identity']:<34} reward={f['reward']:.2f} prediction {f['prediction']}")
            elif f["status"] != "observing":
                print(f"{out['day']}  {f['status']:<10}{f['identity']:<34} {f.get('cause', '')}")
    print("\nMaturity scorecard (Chapter 23)")
    for k, v in maturity(loop.db, years=loop.study_years).items():
        print(f"  {k:<38} {v:.3f}" if isinstance(v, float) else f"  {k:<38} {v}")
    print("\nAllocator posteriors (mean reward):", {s: round(a / (a + b), 3) for s, (a, b) in loop.allocator.posterior().items()})


def crash_test(args: argparse.Namespace) -> None:
    import tempfile
    from pathlib import Path

    from .workflow.loop import Crash, Loop

    store, names, calendar = _market(args.assets, args.days, args.drift, args.seed)
    days = list(calendar[args.warmup :: args.every])
    with tempfile.TemporaryDirectory() as tmp:
        clean = Loop(store, names, path=Path(tmp) / "clean.db", seed=args.seed)
        for d in days:
            clean.tick(d)
        crashed, resumes = Loop(store, names, path=Path(tmp) / "crash.db", seed=args.seed, crash_after=args.crash_after, lease_seconds=0), 0
        i = 0
        while i < len(days):
            try:
                crashed.tick(days[i])
                i += 1
            except Crash:  # the process died; a new one starts on the same database
                resumes += 1
                crashed = Loop(store, names, path=Path(tmp) / "crash.db", seed=args.seed, crash_after=args.crash_after, lease_seconds=0)
        same = clean.snapshot() == crashed.snapshot()
    print(f"{len(days)} ticks; the crashed run was killed and restarted {resumes} times.")
    print("Final state identical to the uninterrupted run:", "yes" if same else "NO")
    if not same:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aq", description="The seed code of The Autonomous Quant.")
    parser.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="what to log to stderr (default WARNING; INFO shows each decision)")
    parser.add_argument("--log-file", help="also write the log to this file, at the same level")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark", help="score the judges on planted truth (Chapter 12)")
    b.add_argument("--families", type=int, default=2000)
    b.add_argument("--tries", type=int, default=20)
    b.add_argument("--years", type=float, default=2.0)
    b.add_argument("--edge", type=float, default=1.5)
    b.add_argument("--share-real", type=float, default=0.25)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=benchmark)
    d = sub.add_parser("demo", help="search the toy rules on one asset, register every try, and judge the family")
    d.add_argument("--source", choices=("simulated", "yahoo"), default="simulated")
    d.add_argument("--symbol", default="SPY", help="with --source yahoo")
    d.add_argument("--years", type=int, default=5)
    d.add_argument("--end", default="2025-12-31")
    d.add_argument("--drift", type=float, default=0.0, help="simulated up/down regime drift per day; 0 is a market with no trends")
    d.add_argument("--cost-bps", type=float, default=5.0)
    d.add_argument("--draws", type=int, default=499)
    d.add_argument("--ledger", default=":memory:", help="a file keeps the ledger between runs")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=demo)
    for name, func, help_ in (("run", run, "run the research loop on a simulated market (Chapter 21)"),
                              ("crash-test", crash_test, "kill the loop repeatedly and check it ends where an uninterrupted run ends")):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--assets", type=int, default=8)
        c.add_argument("--days", type=int, default=900 if name == "run" else 700)
        c.add_argument("--drift", type=float, default=0.004)
        c.add_argument("--warmup", type=int, default=400)
        c.add_argument("--every", type=int, default=15 if name == "run" else 25)
        c.add_argument("--seed", type=int, default=0)
        if name == "run":
            c.add_argument("--slots", type=int, default=2)
            c.add_argument("--db", default=":memory:")
            c.add_argument("--no-research", action="store_true", help="recover and settle only; start no new studies")
        else:
            c.add_argument("--crash-after", type=int, default=3)
        c.set_defaults(func=func)
    args = parser.parse_args(argv)
    aqlog.configure(args.log_level, args.log_file)
    try:
        args.func(args)
    except AutonomousQuantError as exc:  # an expected failure: say what and why, without a traceback
        log.debug("command failed", exc_info=True)
        print(f"aq {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"aq {args.command}: interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception:  # a bug: keep the traceback in the log
        log.exception("aq %s failed unexpectedly", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
