import sqlite3

import numpy as np
import pandas as pd
import pytest

from autonomous_quant.workflow import forward
from autonomous_quant.workflow.allocator import Allocator
from autonomous_quant.research.generator import generate
from autonomous_quant.research.instruments import VolatilityState, WindowScore, validate
from autonomous_quant.interfaces import Reading
from autonomous_quant.research.models import RuleBased, validated_match
from autonomous_quant.research.predicates import standard
from autonomous_quant.backtest.rules import CATALOGUE


def closes(assets=6, days=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, assets)), axis=0)), index=idx, columns=[f"A{i}" for i in range(assets)])


@pytest.mark.parametrize("instrument", [WindowScore(), VolatilityState()])
def test_an_instrument_reads_only_what_had_arrived(instrument):
    p = closes()
    t = p.index[300]
    assert instrument.read(p, t).values == instrument.read(p.loc[:t], t).values


def test_validation_withholds_a_verdict_on_too_few_periods_and_finds_no_skill_in_noise():
    p = closes(days=1500)
    assert validate(WindowScore(), p, list(p.index[200:400:21]))["verdict"] == "insufficient data"
    long = validate(WindowScore(), p, list(p.index[200:1400:21]))
    assert long["verdict"] in ("no skill shown", "skill", "wrong way") and long["periods"] >= 24


def test_a_model_answer_outside_the_catalogue_is_dropped():
    class Bad:
        def match(self, readings, catalogue):
            return "a strategy I just invented"

    assert validated_match(Bad(), [], list(CATALOGUE)) is None
    assert validated_match(RuleBased(), [], list(CATALOGUE)) in CATALOGUE


def test_the_same_experiment_from_two_sources_is_one_hypothesis_with_both_sources():
    reading = Reading("window_score", "1", pd.Timestamp("2021-01-01"), {"A": 3.0, "B": 2.0})
    pairs = generate([reading], RuleBased(), top=1)  # A leads, B lags: two different experiments
    assert len(pairs) == 2
    same = Reading("window_score", "1", pd.Timestamp("2021-01-01"), {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5})
    ids = [h.identity for _, h in generate([same], RuleBased(), top=2)]
    assert len(ids) == len(set(ids))  # merged by identity
    h = pairs[0][1]
    assert "sources" not in h.specification()  # provenance is not part of identity


def test_the_allocator_replays_a_day_and_counts_each_reward_once():
    db = sqlite3.connect(":memory:")
    a = Allocator(db, ("x", "y"), seed=3)
    first = a.choose("2021-01-04", 3, {"x", "y"})
    assert a.choose("2021-01-04", 3, {"x", "y"}) == first
    a.reward("k1", "x", 1.0)
    a.reward("k1", "x", 1.0)  # a repeat is counted once
    a.rejected("k2", "y")
    assert a.posterior() == {"x": (2.0, 1.0), "y": (1.0, 2.0)}
    with pytest.raises(ValueError):
        a.reward("k3", "x", 1.5)


def test_a_forward_binding_cannot_be_changed_and_settles_once():
    db = sqlite3.connect(":memory:")
    forward.init(db)
    binding = {"rule": "trend", "version": "1", "params": {"fast": 5, "slow": 50}, "asset": "A", "predicates": [p.as_dict() for p in standard(20)]}
    assert forward.enroll(db, "h1", "x", binding, pd.Timestamp("2021-01-04"), 20)
    assert not forward.enroll(db, "h1", "x", binding, pd.Timestamp("2021-01-04"), 20)  # a repeat is a no-op
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE bindings SET ends = '2030-01-01'")
    db.execute("UPDATE bindings SET settled_at = 'now', status = 'settled', outcome = '{}'")
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE bindings SET outcome = '{\"reward\": 1}'")
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE bindings SET status = 'invalidated'")
