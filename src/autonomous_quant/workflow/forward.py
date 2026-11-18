"""Forward tests (Chapter 22, Algorithms 22.1 and 22.2).

Enrolment writes an immutable binding: the rule, its version and selected parameters, the asset, the predicates, and
a window fixed from the decision (it starts the next trading day and lasts the hypothesis's horizon). The binding's
fingerprint is stored with it, and the database refuses any change to either.

Settlement checks the conditions of Section 22.6 before it uses an outcome:

* the binding is intact (its fingerprint matches, and the rule's version is still the bound one), else *invalidated*;
* the data cover the window: if the window's last day has not arrived yet the test stays observing (a recoverable gap);
  if days are missing beyond the allowance, or the data never arrive within the grace period, it is *incomplete*;
* the window opened where the test did, and the benchmark (the control) is complete over the same days.

Only then is it settled, once, with an availability time no earlier than the window's end. The reward is the
probabilistic Sharpe ratio against zero of the strategy's excess returns over the window, a number in [0, 1]
(Chapter 22's definition), and the registered predicates are evaluated (Chapter 11).

The seed values the paper record from stored closes through the laboratory. A live system records the account's own
valuations during the window, and that record, not prices filled in afterwards, is what a forward test is scored on.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import pandas as pd

from ..errors import ConfigurationError, DataError
from ..evaluation.judge import probabilistic_sharpe
from ..backtest.lab import Simulator
from ..data.ledger import fingerprint
from ..research.predicates import evaluate, from_dicts
from ..backtest.rules import CATALOGUE

log = logging.getLogger(__name__)

MISSING_ALLOWANCE = 0.05  # share of the window's trading days that may be missing
GRACE_DAYS = 10  # business days after the window's end to wait for its data before calling it incomplete

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bindings (
    id          INTEGER PRIMARY KEY,
    identity    TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL,
    binding     TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    starts      TEXT NOT NULL,
    ends        TEXT NOT NULL,
    settled_at  TEXT,
    status      TEXT,
    outcome     TEXT
);
CREATE TRIGGER IF NOT EXISTS bindings_frozen BEFORE UPDATE OF identity, source, binding, fingerprint, starts, ends ON bindings
BEGIN SELECT RAISE(ABORT, 'a forward binding is immutable'); END;
CREATE TRIGGER IF NOT EXISTS bindings_settle_once BEFORE UPDATE OF settled_at, status, outcome ON bindings WHEN OLD.settled_at IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'a forward test is settled once'); END;
"""


def init(db: sqlite3.Connection) -> None:
    db.executescript(_SCHEMA)


def enroll(db: sqlite3.Connection, identity: str, source: str, binding: dict, decided: pd.Timestamp, horizon: int) -> bool:
    """Write the binding and fix its window. Returns False if this identity was already enrolled (a repeat is a no-op)."""
    if horizon < 2:
        raise ConfigurationError("a forward window needs at least two trading days")
    for field in ("rule", "version", "params", "asset", "predicates"):
        if field not in binding:
            raise ConfigurationError(f"a binding needs {field!r}")
    from_dicts(binding["predicates"])  # refuse malformed predicates at enrolment, not at settlement
    starts = (decided + pd.offsets.BDay(1)).normalize()
    ends = (decided + pd.offsets.BDay(horizon)).normalize()
    with db:
        written = db.execute(
            "INSERT OR IGNORE INTO bindings (identity, source, binding, fingerprint, starts, ends) VALUES (?, ?, ?, ?, ?, ?)",
            (identity, source, json.dumps(binding, sort_keys=True), fingerprint(binding), starts.isoformat(), ends.isoformat()),
        ).rowcount
    if written:
        log.info("enrolled %s in a forward test from %s to %s", identity, starts.date(), ends.date())
    return bool(written)


def due(db: sqlite3.Connection, today: pd.Timestamp) -> list[tuple]:
    return db.execute(
        "SELECT id, identity, source, binding, fingerprint, starts, ends FROM bindings WHERE settled_at IS NULL AND ends <= ? ORDER BY id",
        (today.normalize().isoformat(),),
    ).fetchall()


def _record(db: sqlite3.Connection, bid: int, today: pd.Timestamp, status: str, outcome: dict) -> None:
    with db:
        db.execute("UPDATE bindings SET settled_at = ?, status = ?, outcome = ? WHERE id = ?",
                   (today.isoformat(), status, json.dumps(outcome, sort_keys=True), bid))


def settle(db: sqlite3.Connection, row: tuple, prices: pd.DataFrame, lab: Simulator, today: pd.Timestamp) -> dict:
    """Settle one due test, or leave it observing. Returns what happened, with ``status`` one of
    'observing', 'settled', 'invalidated' or 'incomplete'."""
    bid, identity, source, binding, stored_fp, starts, ends = row
    b = json.loads(binding)
    starts, ends = pd.Timestamp(starts), pd.Timestamp(ends)
    result = {"identity": identity, "source": source}

    rule = CATALOGUE.get(b["rule"])
    if fingerprint(b) != stored_fp or rule is None or rule.version != b["version"]:
        cause = "binding changed" if fingerprint(b) != stored_fp else "rule no longer available at the bound version"
        _record(db, bid, today, "invalidated", {"cause": cause})
        log.warning("forward test %s invalidated: %s", identity, cause)
        return {**result, "status": "invalidated", "cause": cause}

    expected = pd.bdate_range(starts, ends)
    have = prices.index[(prices.index >= starts) & (prices.index <= ends)] if len(prices) else pd.DatetimeIndex([])
    if len(prices) == 0 or prices.index.max() < ends:
        if today < ends + pd.offsets.BDay(GRACE_DAYS):
            log.info("forward test %s: data to %s not yet available; still observing", identity, ends.date())
            return {**result, "status": "observing"}
        cause = f"data for the window had not arrived {GRACE_DAYS} business days after its end"
        _record(db, bid, today, "incomplete", {"cause": cause})
        log.warning("forward test %s incomplete: %s", identity, cause)
        return {**result, "status": "incomplete", "cause": cause}
    missing = 1 - len(have) / len(expected)
    if missing > MISSING_ALLOWANCE or len(have) == 0 or have[0] != expected[0]:
        cause = f"{missing:.0%} of the window's days are missing" if missing > MISSING_ALLOWANCE else "the window did not open where the test did"
        _record(db, bid, today, "incomplete", {"cause": cause})
        log.warning("forward test %s incomplete: %s", identity, cause)
        return {**result, "status": "incomplete", "cause": cause}

    try:
        out = lab.run(rule, b["params"], prices.loc[:ends])  # history before the window warms the rule up
        inside = out.returns.index >= starts
        strategy, control = out.returns[inside], out.control_returns[inside]
        if strategy.isna().any() or control.isna().any() or len(control) != len(strategy):
            raise DataError("the benchmark is incomplete over the window")
        equity = (1 + strategy).cumprod()
        metrics = {"observations": int(len(strategy)), "strategy_return": float(strategy.sum()), "control_return": float(control.sum()),
                   "incremental_return": float(strategy.sum() - control.sum()), "max_drawdown": float(1 - (equity / equity.cummax()).min())}
        try:
            reward = probabilistic_sharpe(strategy.to_numpy(), 0.0)
        except DataError as exc:  # e.g. the rule stayed flat all window: no return to score
            reward = 0.5
            log.info("forward test %s: %s; reward set to 0.5 (no evidence either way)", identity, exc)
        prediction = evaluate(from_dicts(b["predicates"]), metrics)
    except DataError as exc:
        _record(db, bid, today, "incomplete", {"cause": str(exc)})
        log.warning("forward test %s incomplete: %s", identity, exc)
        return {**result, "status": "incomplete", "cause": str(exc)}

    outcome = {"reward": reward, "metrics": metrics, "prediction": prediction["verdict"], "available_from": today.isoformat()}
    _record(db, bid, today, "settled", outcome)
    log.info("forward test %s settled: reward %.3f, prediction %s", identity, reward, prediction["verdict"])
    return {**result, "status": "settled", **outcome}
