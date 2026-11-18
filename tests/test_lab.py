import numpy as np
import pandas as pd

from autonomous_quant.backtest.lab import Simulator, truncation_test
from autonomous_quant.backtest.rules import MeanReversionRule, TrendRule
from autonomous_quant.data.sources import SimulatedSource


def prices(days=600, seed=1):
    src = SimulatedSource(assets=1, drift=0.002, seed=seed)
    obs = src.observations(["A"], pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01") + pd.offsets.BDay(days - 1))
    frame = pd.DataFrame([(o.observed.normalize(), o.key.split(":")[1], o.value) for o in obs], columns=["day", "col", "v"])
    return frame.pivot(index="day", columns="col", values="v")


class Constant:
    name, version, region = "constant", "1", {}

    def __init__(self, level):
        self.level = level

    def positions(self, closes, params):
        return pd.Series(self.level, index=closes.index)


def test_cash_earns_the_cash_rate_so_a_flat_rule_earns_exactly_zero():
    p = prices()
    rate = pd.Series(0.0002, index=p.index)
    out = Simulator().run(Constant(0.0), {}, p, rate)
    assert np.allclose(out.returns, 0.0) and out.round_trips == 0


def test_the_account_is_self_financing_and_costs_only_lower_equity():
    p = prices()
    target = TrendRule().positions(p["close"], {"fast": 10, "slow": 50})
    free = Simulator(cost_bps=0).account(target, p["open"], p["close"], pd.Series(0.0, index=p.index))
    costly = Simulator(cost_bps=20).account(target, p["open"], p["close"], pd.Series(0.0, index=p.index))
    assert np.allclose(free["equity"], free["cash"] + free["units"] * p["close"])
    assert (costly["equity"] <= free["equity"] + 1e-6).all()


def test_a_position_decided_at_a_close_is_filled_at_the_next_open():
    p = prices(50)
    target = pd.Series(0.0, index=p.index)
    target.iloc[20] = 1.0  # decided at the close of day 20
    book = Simulator().account(target, p["open"], p["close"], pd.Series(0.0, index=p.index))
    assert book["units"].iloc[20] == 0 and book["units"].iloc[21] > 0


def test_the_toy_rules_pass_the_truncation_test_and_a_rule_that_looks_ahead_fails_it():
    closes = prices()["close"]
    assert truncation_test(TrendRule(), {"fast": 10, "slow": 50}, closes)
    assert truncation_test(MeanReversionRule(), {"lookback": 20, "entry": 1.5}, closes)

    class Peeking:
        name, version, region = "peek", "1", {}

        def positions(self, c, params):
            return (c.shift(-1) > c).astype(float)  # uses tomorrow's close

    assert not truncation_test(Peeking(), {}, closes)
