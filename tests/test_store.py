import sqlite3

import pandas as pd
import pytest

from autonomous_quant.interfaces import Observation
from autonomous_quant.data.store import Store, UnknownAvailability

T = pd.Timestamp


def test_the_past_as_of_t_is_not_changed_by_a_later_revision():
    store = Store()
    store.append([Observation("gdp", T("2026-03-31"), 1.0, T("2026-04-30")), Observation("gdp", T("2026-03-31"), 1.4, T("2026-06-30"), revision=1)])
    assert store.as_of("gdp", T("2026-04-01")).empty  # not yet released
    assert store.as_of("gdp", T("2026-05-01")).iloc[0] == 1.0  # the first release
    assert store.as_of("gdp", T("2026-07-01")).iloc[0] == 1.4  # the revision


def test_the_store_is_append_only():
    store = Store()
    store.append([Observation("x", T("2026-01-01"), 1.0, T("2026-01-01"))])
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("UPDATE observations SET value = 2")
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("DELETE FROM observations")


def test_unknown_availability_is_refused_unless_assumed():
    store = Store()
    store.append([Observation("x", T("2026-01-01"), 1.0, None)])
    with pytest.raises(UnknownAvailability):
        store.as_of("x", T("2026-02-01"))
    assert store.as_of("x", T("2026-02-01"), assume_available_at_observation=True).iloc[0] == 1.0
