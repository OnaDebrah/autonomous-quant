"""The loop's crash test (Algorithm 21.2): kill it part-way, run it again, and compare with an uninterrupted run."""

from autonomous_quant.cli import _market
from autonomous_quant.workflow.loop import Crash, Loop


def test_a_tick_run_twice_changes_nothing(tmp_path):
    store, names, cal = _market(4, 520, 0.004, 1)
    loop = Loop(store, names, path=tmp_path / "a.db", seed=1)
    loop.tick(cal[450])
    before = loop.snapshot()
    assert loop.tick(cal[450]) == {"day": str(cal[450].date()), "skipped": "already finished"}
    assert loop.snapshot() == before and before["trials"]


def test_a_crashed_and_restarted_loop_ends_where_an_uninterrupted_one_ends(tmp_path):
    store, names, cal = _market(4, 560, 0.004, 2)
    days = list(cal[420::40])
    clean = Loop(store, names, path=tmp_path / "clean.db", seed=2)
    for d in days:
        clean.tick(d)
    def make():
        return Loop(store, names, path=tmp_path / "crash.db", seed=2, crash_after=1, lease_seconds=0)

    loop, i, crashes = make(), 0, 0
    while i < len(days):
        try:
            loop.tick(days[i])
            i += 1
        except Crash:
            crashes += 1
            loop = make()
    assert crashes > 0
    assert loop.snapshot() == clean.snapshot()
