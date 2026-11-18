import numpy as np
import pytest

from autonomous_quant.evaluation.judge import Clause, deflated_rule, deflated_threshold, expected_best, judge_contract, sign_flip_test
from autonomous_quant.data.ledger import Ledger


def test_every_started_try_counts_and_a_configuration_counts_once():
    ledger = Ledger()
    a = ledger.register("f", {"rule": "trend", "params": {"fast": 5}})
    b = ledger.register("f", {"rule": "trend", "params": {"fast": 10}})
    ledger.fail(b, "crashed")
    assert ledger.register("f", {"params": {"fast": 5}, "rule": "trend"}) == a  # same configuration, any key order
    assert ledger.family_size("f") == 2  # the failed try still counts


def test_the_deflated_bar_matches_the_book():
    assert expected_best(20) == pytest.approx(1.9007, abs=1e-4)  # Chapter 14: 1.90 for twenty independent tries
    assert deflated_threshold(20) == pytest.approx(3.5456, abs=1e-3)


def test_failed_tries_raise_the_bar():
    sharpes = [2.4, 0.3, -0.1]
    assert deflated_rule(sharpes, years=2)["passed"]
    assert not deflated_rule(sharpes, years=2, family_size=50)["passed"]
    with pytest.raises(ValueError):
        deflated_rule(sharpes, years=2, family_size=2)


def test_a_contract_fails_closed():
    clauses = [Clause("trades", lambda e: e["round_trips"] >= 10)]
    assert judge_contract(clauses, {"round_trips": 12})["passed"]
    missing = judge_contract(clauses, {})
    assert not missing["passed"] and missing["clauses"]["trades"] is None


def test_the_sign_flip_test_finds_a_real_edge_and_not_noise():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((504, 20)) * 0.01
    edged = noise.copy()
    edged[:, 3] += 0.003
    assert sign_flip_test(noise, draws=199)["p_value"] > 0.05
    assert sign_flip_test(edged, draws=199)["p_value"] < 0.05
