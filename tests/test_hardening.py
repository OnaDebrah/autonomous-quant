"""Errors, logging and the book's definitions: what the seed refuses, what it records, and what it reports."""

import json
import logging
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm, skew

from autonomous_quant.workflow import forward
from autonomous_quant.cli import _market, main
from autonomous_quant.errors import AutonomousQuantError, ConfigurationError, DataError, IntegrityError
from autonomous_quant.interfaces import Observation
from autonomous_quant.evaluation.judge import deflated_rule, probabilistic_sharpe
from autonomous_quant.backtest.lab import Simulator
from autonomous_quant.data.ledger import Ledger
from autonomous_quant.workflow.loop import Loop
from autonomous_quant.research.predicates import Predicate, check, evaluate, standard
from autonomous_quant.backtest.rules import CATALOGUE
from autonomous_quant.data.store import Store, UnknownAvailability


def prices(days=200, seed=0, start="2021-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=days)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, days)))
    return pd.DataFrame({"open": close * (1 + rng.normal(0, 0.002, days)), "close": close}, index=idx)


# Errors ---------------------------------------------------------------------------------------------------------

def test_the_errors_form_one_hierarchy():
    for e in (ConfigurationError, DataError, IntegrityError, UnknownAvailability):
        assert issubclass(e, AutonomousQuantError)


@pytest.mark.parametrize("kwargs", [dict(value=float("nan")), dict(key=""), dict(revision=-1),
                                    dict(available=pd.Timestamp("2020-12-31"))])
def test_an_observation_refuses_impossible_values(kwargs):
    base = dict(key="A:close", observed=pd.Timestamp("2021-01-04"), value=1.0, available=pd.Timestamp("2021-01-04 17:00"), revision=0)
    with pytest.raises(DataError):
        Observation(**{**base, **kwargs})


def test_an_as_of_read_refuses_unknown_availability_unless_the_assumption_is_stated(caplog):
    store = Store()
    store.append([Observation("A:close", pd.Timestamp("2021-01-04"), 1.0, None, 0)])
    with pytest.raises(UnknownAvailability):
        store.as_of("A:close", pd.Timestamp("2021-01-05"))
    with caplog.at_level(logging.WARNING, logger="autonomous_quant"):
        assert len(store.as_of("A:close", pd.Timestamp("2021-01-05"), assume_available_at_observation=True)) == 1
    assert any("assum" in r.message for r in caplog.records)


def test_the_laboratory_refuses_bad_prices_and_bad_positions():
    lab, rule = Simulator(), CATALOGUE["trend"]
    bad = prices()
    bad.iloc[5, 1] = -1.0
    with pytest.raises(DataError):
        lab.run(rule, {"fast": 5, "slow": 50}, bad)
    with pytest.raises(ConfigurationError):
        lab.run(rule, {"fast": 50, "slow": 5}, prices())  # fast must be faster than slow
    with pytest.raises(ConfigurationError):
        Simulator(cost_bps=-1)


def test_a_finished_try_cannot_be_rewritten_but_a_repeat_is_harmless():
    ledger = Ledger()
    t = ledger.register("f", {"rule": "trend", "params": {"fast": 5}})
    ledger.complete(t, {"sharpe": 1.0})
    ledger.complete(t, {"sharpe": 1.0})  # the same result again: a no-op
    with pytest.raises(IntegrityError):
        ledger.complete(t, {"sharpe": 2.0})


def test_a_family_whose_tries_all_failed_has_no_verdict():
    with pytest.raises(DataError):
        deflated_rule([], years=2, family_size=5)


# The book's definitions -----------------------------------------------------------------------------------------

def test_the_probabilistic_sharpe_ratio_matches_chapter_4():
    r = np.random.default_rng(3).normal(0.001, 0.01, 250)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    expected = norm.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    assert probabilistic_sharpe(r) == pytest.approx(expected, rel=1e-9)
    with pytest.raises(DataError):
        probabilistic_sharpe(np.zeros(10))  # no variation, no ratio


def test_predicates_separate_not_supported_from_not_evaluable():
    p = standard(60)
    good = {"observations": 60, "strategy_return": 0.02, "incremental_return": 0.01, "max_drawdown": 0.05}
    assert evaluate(p, good)["verdict"] == "supported"
    assert evaluate(p, {**good, "strategy_return": -0.01, "incremental_return": 0.03})["verdict"] == "contradicted"  # beat a falling control, lost money
    assert evaluate(p, {**good, "observations": 10})["verdict"] == "not evaluable"  # not a falsification
    banded = (p[0], Predicate("mechanism", "strategy_return", ">", 0.05), Predicate("failure", "strategy_return", "<", 0.0))
    assert evaluate(banded, {**good, "strategy_return": 0.01})["verdict"] == "not supported"  # neither held
    with pytest.raises(ConfigurationError):
        Predicate("mechanism", "luck", ">", 0)


@pytest.mark.parametrize("failure", [Predicate("failure", "max_drawdown", ">", 0.25),  # another metric: both could hold
                                     Predicate("failure", "strategy_return", "<", 0.01),  # overlaps (0, 0.01)
                                     Predicate("failure", "strategy_return", ">=", 0.0)])
def test_an_inconsistent_contract_is_refused_before_any_test(failure):
    with pytest.raises(ConfigurationError):
        check((Predicate("mechanism", "strategy_return", ">", 0.0), failure))
    check((Predicate("mechanism", "strategy_return", ">", 0.0), Predicate("failure", "strategy_return", "<=", 0.0)))


# Forward tests --------------------------------------------------------------------------------------------------

def _enrolled(horizon=20):
    db = sqlite3.connect(":memory:")
    forward.init(db)
    binding = {"rule": "trend", "version": CATALOGUE["trend"].version, "params": {"fast": 5, "slow": 50}, "asset": "A",
               "predicates": [p.as_dict() for p in standard(horizon)]}
    decided = pd.Timestamp("2021-06-01")
    forward.enroll(db, "h1", "leaders", binding, decided, horizon)
    return db, decided


def test_a_forward_test_waits_for_its_data_then_settles_once_with_a_psr_reward():
    db, decided = _enrolled()
    p = prices(260)
    (row,) = forward.due(db, pd.Timestamp("2021-07-01"))
    early = forward.settle(db, row, p.loc[: decided + pd.offsets.BDay(10)], Simulator(), pd.Timestamp("2021-07-01"))
    assert early["status"] == "observing" and forward.due(db, pd.Timestamp("2021-07-01"))  # a recoverable gap: still due
    done = forward.settle(db, row, p, Simulator(), pd.Timestamp("2021-07-02"))
    assert done["status"] == "settled" and 0 <= done["reward"] <= 1
    assert done["prediction"] in ("supported", "contradicted", "not evaluable")
    assert json.loads(db.execute("SELECT outcome FROM bindings").fetchone()[0])["available_from"] >= "2021-06-29"
    assert not forward.due(db, pd.Timestamp("2021-08-01"))


def test_a_forward_test_with_missing_days_or_a_changed_binding_is_not_scored():
    db, decided = _enrolled()
    p = prices(260)
    window = p.index[(p.index > decided) & (p.index <= decided + pd.offsets.BDay(20))]
    (row,) = forward.due(db, pd.Timestamp("2021-07-01"))
    assert forward.settle(db, row, p.drop(window[3:8]), Simulator(), pd.Timestamp("2021-07-01"))["status"] == "incomplete"

    db, _ = _enrolled()
    (row,) = forward.due(db, pd.Timestamp("2021-07-01"))
    tampered = (*row[:3], row[3].replace('"fast": 5', '"fast": 6'), *row[4:])
    out = forward.settle(db, tampered, p, Simulator(), pd.Timestamp("2021-07-01"))
    assert out["status"] == "invalidated" and out["cause"] == "binding changed"


def test_a_forward_test_whose_data_never_arrive_becomes_incomplete():
    db, decided = _enrolled()
    (row,) = forward.due(db, pd.Timestamp("2022-01-01"))
    out = forward.settle(db, row, prices(100).loc[:decided], Simulator(), pd.Timestamp("2022-01-01"))
    assert out["status"] == "incomplete"


def test_a_binding_without_predicates_is_refused():
    db = sqlite3.connect(":memory:")
    forward.init(db)
    with pytest.raises(ConfigurationError):
        forward.enroll(db, "h", "s", {"rule": "trend", "version": "1", "params": {}, "asset": "A"}, pd.Timestamp("2021-01-04"), 20)


# The loop -------------------------------------------------------------------------------------------------------

def test_a_failing_study_is_retried_then_recorded_and_the_loop_goes_on(tmp_path, monkeypatch, caplog):
    store, names, cal = _market(4, 520, 0.004, 1)
    loop = Loop(store, names, path=tmp_path / "a.db", seed=1, max_attempts=2)
    monkeypatch.setattr(loop, "study", lambda *a: (_ for _ in ()).throw(DataError("no history")))
    with caplog.at_level(logging.ERROR, logger="autonomous_quant"):
        loop.tick(cal[450])
        assert {s for _, s in loop.snapshot()["outbox"]} == {"pending"}  # one attempt each this tick
        loop.tick(cal[451])
    statuses = loop.db.execute("SELECT status, attempts, error FROM outbox WHERE day = ?", (str(cal[450].date()),)).fetchall()
    assert statuses and all(s == "failed" and a == 2 and "no history" in e for s, a, e in statuses)
    assert any("giving up" in r.message for r in caplog.records)


def test_switching_research_off_still_settles_due_forward_tests(tmp_path):
    store, names, cal = _market(4, 700, 0.004, 1)
    loop = Loop(store, names, path=tmp_path / "a.db", seed=1)
    binding = {"rule": "trend", "version": CATALOGUE["trend"].version, "params": {"fast": 5, "slow": 50}, "asset": names[0],
               "predicates": [p.as_dict() for p in standard(20)]}
    forward.enroll(loop.db, "h1", "leaders", binding, cal[450], 20)
    loop.research_enabled = False
    out = loop.tick(cal[480])
    assert out["studies"] == [] and [s["status"] for s in out["settled"]] == ["settled"]
    assert loop.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    assert loop.db.execute("SELECT COUNT(*) FROM rewards WHERE kind = 'forward'").fetchone()[0] == 1


def test_each_allocation_records_the_posterior_it_saw(tmp_path):
    store, names, cal = _market(4, 520, 0.004, 1)
    loop = Loop(store, names, path=tmp_path / "a.db", seed=1)
    loop.tick(cal[450])
    rows = loop.snapshot()["allocations"]
    assert rows and all(set(json.loads(posterior)) == {"leaders", "laggards"} for *_, posterior in rows)


# The command line -----------------------------------------------------------------------------------------------

def test_the_cli_reports_an_expected_failure_without_a_traceback(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["--log-level", "ERROR", "demo", "--years", "0"])
    assert exit_.value.code == 2
    err = capsys.readouterr().err
    assert "aq demo:" in err and "Traceback" not in err
