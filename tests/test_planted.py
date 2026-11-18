"""The book's central numbers, reproduced on planted truth (Chapters 12 and 13)."""

from autonomous_quant.evaluation import planted


def test_the_deflated_rule_is_honest_and_nearly_blind_on_two_years():
    card = planted.scorecard(planted.World(tries=20, years=2.0, edge=1.5, share_real=0.25, seed=7), families=3000)
    assert card["deflated"]["false_pass_rate"] < 0.01  # Chapter 13: about 0.4% for a null family of twenty
    assert 0.03 < card["deflated"]["power"] < 0.14  # Table 13.3: 7.7%
    assert card["naive"]["false_pass_rate"] > 0.10  # the naive judge passes noise often
    assert card["naive"]["power"] > card["deflated"]["power"]


def test_more_years_buy_power_without_more_false_passes():
    short = planted.scorecard(planted.World(years=2.0, seed=3), families=2000)["deflated"]
    long = planted.scorecard(planted.World(years=10.0, seed=3), families=2000)["deflated"]
    assert long["power"] > 4 * short["power"] and long["false_pass_rate"] < 0.01
