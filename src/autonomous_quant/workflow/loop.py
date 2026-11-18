"""The loop (Chapter 21, Algorithms 21.1 to 21.3): an idempotent tick with an outbox and leases.

One tick, for one decision day, in the order of Algorithm 21.1:

1. Recover, unconditionally: deliver any study that was decided and not finished, and settle every forward test that
   is due. This runs even when new research is switched off: a forward test in progress is a registered experiment,
   and pausing research must not leave it unscored.
2. If research is switched off, stop.
3. Read the instruments as of the day, generate hypotheses, and let the allocator choose which to study, recording
   each choice with its probability and the posterior it saw.
4. In one transaction, write the chosen studies to the outbox: they are now owed, exactly once in effect.
5. Work the outbox: lease a study, register every try in the ledger before running it, judge the family, enrol a pass
   in a forward test or record a rejection with the allocator, and complete the study under its own lease.

Every step is keyed so that repeating it changes nothing, which is what makes the loop safe to kill and restart. A
study that raises an error is retried on later ticks and, after ``max_attempts``, recorded as failed with its error;
the loop continues. ``tests/test_loop.py`` kills the loop part-way and checks that it ends where an uninterrupted run
ends.
"""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import forward
from .allocator import Allocator
from ..errors import AutonomousQuantError, ConfigurationError, DataError
from ..research.generator import SOURCES, Hypothesis, generate
from ..research.instruments import VolatilityState, WindowScore
from ..evaluation.judge import default_contract, deflated_rule, judge_contract
from ..backtest.lab import Simulator, sharpe
from ..data.ledger import Ledger
from ..research.models import RuleBased
from ..backtest.rules import CATALOGUE
from ..data.store import Store

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (day TEXT PRIMARY KEY, finished_at TEXT);
CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,      -- day + hypothesis identity: a study is owed once
    day         TEXT NOT NULL,
    source      TEXT NOT NULL,
    hypothesis  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'leased', 'done', 'failed')),
    lease_token TEXT,
    lease_until REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT
);
"""


class Crash(BaseException):
    """Raised on purpose by the crash test to stop a tick part-way. A BaseException, so that no handler in the loop
    mistakes it for a failed study: it must behave like the process dying."""


@dataclass
class Loop:
    store: Store
    assets: list[str]
    path: str | Path = ":memory:"
    slots: int = 2  # studies per tick
    study_years: float = 1.5
    lease_seconds: float = 60.0
    max_attempts: int = 3
    cost_bps: float = 5.0
    seed: int = 0
    research_enabled: bool = True
    model: object = field(default_factory=RuleBased)
    crash_after: int | None = None  # for the crash test: stop after this many studies have been worked

    def __post_init__(self):
        if not self.assets:
            raise ConfigurationError("the loop needs at least one asset")
        if self.slots < 0 or not self.study_years > 0 or self.lease_seconds < 0 or self.max_attempts < 1:
            raise ConfigurationError("need slots >= 0, study_years > 0, lease_seconds >= 0 and max_attempts >= 1")
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(_SCHEMA)
        forward.init(self.db)
        self.ledger = Ledger(self.db)
        self.allocator = Allocator(self.db, SOURCES, seed=self.seed)
        self.lab = Simulator(cost_bps=self.cost_bps)
        self.instruments = (WindowScore(), VolatilityState())
        self._worked = 0

    def close(self) -> None:
        self.db.close()

    # Data -------------------------------------------------------------------------------------------------------
    def prices(self, asset: str, asof: pd.Timestamp) -> pd.DataFrame:
        frame = pd.concat({c: self.store.as_of(f"{asset}:{c}", asof) for c in ("open", "close")}, axis=1)
        frame.index = frame.index.normalize()
        return frame.groupby(level=0).last().dropna()

    def closes(self, asof: pd.Timestamp) -> pd.DataFrame:
        return pd.concat({a: self.prices(a, asof)["close"] for a in self.assets}, axis=1)

    @staticmethod
    def _asof(day: pd.Timestamp) -> pd.Timestamp:
        return day.normalize() + pd.Timedelta(hours=23)  # after the day's close has arrived

    # One tick ---------------------------------------------------------------------------------------------------
    def tick(self, day: pd.Timestamp) -> dict:
        key = day.date().isoformat()
        if self.db.execute("SELECT 1 FROM ticks WHERE day = ? AND finished_at IS NOT NULL", (key,)).fetchone():
            log.debug("tick %s already finished", key)
            return {"day": key, "skipped": "already finished"}
        log.info("tick %s", key)
        recovered = self.work_outbox(day)  # 1. recover: studies already owed
        settled = self.settle(day)  #          ... and forward tests already due
        studies: list[dict] = []
        if self.research_enabled:  # 2.
            self.decide(day, key)  # 3 and 4
            studies = self.work_outbox(day)  # 5
        else:
            log.info("tick %s: research switched off; recovered and settled only", key)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO ticks VALUES (?, ?)", (key, pd.Timestamp.now().isoformat()))
        return {"day": key, "studies": recovered + studies, "settled": settled}

    def decide(self, day: pd.Timestamp, key: str) -> None:
        asof = self._asof(day)
        try:
            closes = self.closes(asof)
            readings = [i.read(closes, asof) for i in self.instruments]
        except DataError:
            log.exception("tick %s: instruments could not be read; no new studies today", key)
            return
        by_source: dict[str, list[Hypothesis]] = {}
        for source, h in generate(readings, self.model):
            by_source.setdefault(source, []).append(h)
        chosen = self.allocator.choose(key, self.slots, set(by_source))
        with self.db:  # the day's studies become owed, all at once
            for slot, source in enumerate(chosen):
                h = by_source[source][slot % len(by_source[source])]
                self.db.execute("INSERT OR IGNORE INTO outbox (key, day, source, hypothesis, status) VALUES (?, ?, ?, ?, 'pending')",
                                (f"{key}:{h.identity}", key, source, json.dumps({**h.specification(), "sources": list(h.sources)})))
        log.info("tick %s: %d studies owed", key, len(chosen))

    # The outbox -------------------------------------------------------------------------------------------------
    def _lease(self, skip: tuple[int, ...] = ()) -> tuple | None:
        token, now = uuid.uuid4().hex, time.time()
        with self.db:
            row = self.db.execute(
                "SELECT id FROM outbox WHERE (status = 'pending' OR (status = 'leased' AND lease_until < ?))"
                f" AND id NOT IN ({','.join('?' * len(skip))}) ORDER BY id LIMIT 1", (now, *skip)
            ).fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE outbox SET status = 'leased', lease_token = ?, lease_until = ?, attempts = attempts + 1 WHERE id = ?",
                            (token, now + self.lease_seconds, row[0]))
        return (row[0], token)

    def work_outbox(self, day: pd.Timestamp) -> list[dict]:
        done, deferred = [], []  # a study that failed this tick is retried on a later tick, not in a tight loop
        while (lease := self._lease(tuple(deferred))) is not None:
            job_id, token = lease
            source, spec, attempts = self.db.execute("SELECT source, hypothesis, attempts FROM outbox WHERE id = ?", (job_id,)).fetchone()
            try:
                result = self.study(json.loads(spec), source, day)
            except Exception as exc:  # recorded and retried; the loop goes on (Crash is not an Exception)
                final = attempts >= self.max_attempts
                status = "failed" if final else "pending"
                log.error("study %d (%s) failed on attempt %d of %d%s", job_id, source, attempts, self.max_attempts,
                          "; giving up" if final else "; will retry", exc_info=not isinstance(exc, AutonomousQuantError))
                with self.db:
                    self.db.execute("UPDATE outbox SET status = ?, error = ?, lease_token = NULL WHERE id = ? AND lease_token = ?",
                                    (status, "".join(traceback.format_exception_only(type(exc), exc)).strip(), job_id, token))
                if not final:
                    deferred.append(job_id)
                continue
            self._worked += 1
            if self.crash_after is not None and self._worked > self.crash_after:
                raise Crash(f"stopped after {self.crash_after} studies")  # the study ran; its completion was not recorded
            with self.db:  # complete only under our own lease (a fencing token)
                completed = self.db.execute("UPDATE outbox SET status = 'done', result = ? WHERE id = ? AND lease_token = ?",
                                            (json.dumps(result), job_id, token)).rowcount
            if not completed:
                log.warning("study %d: lease lost before completion; another worker owns it", job_id)
            done.append(result)
        return done

    # A study ----------------------------------------------------------------------------------------------------
    def study(self, spec: dict, source: str, day: pd.Timestamp) -> dict:
        """Search the rule's region on the asset's history, register every try first, and judge the family."""
        rule = CATALOGUE.get(spec["rule"])
        if rule is None or rule.version != spec["version"]:
            raise ConfigurationError(f"rule {spec['rule']} version {spec['version']} is not in the catalogue")
        prices = self.prices(spec["asset"], self._asof(day)).iloc[-int(self.study_years * 252):]
        if len(prices) < 0.9 * self.study_years * 252:
            raise DataError(f"{spec['asset']}: {len(prices)} days of history, {int(self.study_years * 252)} needed")
        window = [str(prices.index[0].date()), str(prices.index[-1].date())]
        family = f"{spec['rule']}|{spec['asset']}|{window[0]}|{window[1]}"
        results = []
        for values in itertools.product(*spec["region"].values()):
            params = dict(zip(spec["region"], values))
            trial = self.ledger.register(family, {**{k: v for k, v in spec.items() if k != "sources"}, "params": params, "window": window})
            try:
                out = self.lab.run(rule, params, prices)
            except AutonomousQuantError as exc:  # a failed try still counts in the family
                self.ledger.fail(trial, f"{type(exc).__name__}: {exc}")
                continue
            equity = (1 + out.returns).cumprod()
            r = {"params": params, "sharpe": sharpe(out.returns), "control_sharpe": sharpe(out.control_returns),
                 "round_trips": out.round_trips, "max_drawdown": float(1 - (equity / equity.cummax()).min())}
            self.ledger.complete(trial, r)
            results.append(r)
        identity = f"{day.date()}|{spec['rule']}|{spec['asset']}"
        if not results:  # every try failed: the family is counted, and the study is a rejection
            self.allocator.rejected(identity, source)
            log.warning("study %s %s on %s: every try failed; rejected", source, spec["rule"], spec["asset"])
            return {"source": source, "rule": spec["rule"], "asset": spec["asset"], "family_size": self.ledger.family_size(family),
                    "standardized": None, "threshold": None, "passed": False, "reasons": ["every try failed"]}
        verdict = deflated_rule([r["sharpe"] for r in results], len(prices) / 252, family_size=self.ledger.family_size(family))
        best = results[verdict["best"]]
        contract = judge_contract(default_contract(min_round_trips=4), {**best, "deflated_passed": verdict["passed"]})
        if contract["passed"]:
            forward.enroll(self.db, identity, source, {"rule": rule.name, "version": rule.version, "params": best["params"],
                                                       "asset": spec["asset"], "predicates": spec["predicates"]}, day, spec["horizon"])
        else:
            self.allocator.rejected(identity, source)
        log.info("study %s %s on %s: z %.2f against %.2f for %d tries: %s", source, spec["rule"], spec["asset"], verdict["standardized"],
                 verdict["threshold"], verdict["family_size"], "passed" if contract["passed"] else "rejected (" + ", ".join(contract["reasons"]) + ")")
        return {"source": source, "rule": spec["rule"], "asset": spec["asset"], "family_size": verdict["family_size"],
                "standardized": verdict["standardized"], "threshold": verdict["threshold"], "passed": contract["passed"],
                "reasons": contract["reasons"]}

    # Forward tests ----------------------------------------------------------------------------------------------
    def settle(self, day: pd.Timestamp) -> list[dict]:
        out = []
        for row in forward.due(self.db, day):
            identity = row[1]
            try:
                asset = json.loads(row[3])["asset"]
                result = forward.settle(self.db, row, self.prices(asset, self._asof(day)), self.lab, day)
            except Exception:  # one bad settlement must not stop the others; it is retried on the next tick
                log.exception("forward test %s could not be settled this tick", identity)
                continue
            if result["status"] == "settled":
                self.allocator.reward(identity, result["source"], result["reward"])
            out.append(result)
        return out

    # A snapshot for tests and reports ---------------------------------------------------------------------------
    def snapshot(self) -> dict:
        def q(sql: str) -> list:
            return self.db.execute(sql).fetchall()

        return {
            "trials": q("SELECT family, fingerprint, status, result FROM trials ORDER BY family, fingerprint"),
            "outbox": q("SELECT key, status FROM outbox ORDER BY key"),
            "bindings": q("SELECT identity, binding, starts, ends, status, outcome FROM bindings ORDER BY identity"),
            "posteriors": q("SELECT source, round(a, 9), round(b, 9) FROM posteriors ORDER BY source"),
            "allocations": q("SELECT day, slot, source, posterior FROM allocations ORDER BY day, slot"),
        }
