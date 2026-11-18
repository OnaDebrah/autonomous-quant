"""The maturity scorecard (Chapter 23, Algorithm 23.2): what the researcher's own record can and cannot yet say.

A young researcher looks weak whatever its ideas are worth, because on short records estimates of weak and strong
rules alike are dominated by noise. The scorecard states that arithmetic for the record the system actually has: the
length of history its studies use, the bar its families face, the power that bar leaves for an edge of a given size,
and the share of its passes that would be false if one idea in ten were real. Then it adds the forward record.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3

import numpy as np
from scipy.stats import norm

from ..errors import ConfigurationError
from .judge import deflated_threshold

log = logging.getLogger(__name__)


def maturity(db: sqlite3.Connection, years: float, edge: float = 1.5, share_real: float = 0.1, alpha: float = 0.05) -> dict:
    if not years > 0 or not edge > 0 or not 0 < share_real < 1 or not 0 < alpha < 1:
        raise ConfigurationError("need years > 0, edge > 0, and share_real and alpha in (0, 1)")
    families = db.execute("SELECT family, COUNT(*) FROM trials GROUP BY family").fetchall()
    sizes = [n for _, n in families] or [1]
    n = int(np.median(sizes))
    c = deflated_threshold(n, alpha)
    power = float(norm.cdf(edge * math.sqrt(years) - c))
    false_pass = float(1 - norm.cdf(c))  # per try with no edge, under the Gaussian approximation
    false_share = (1 - share_real) * false_pass / ((1 - share_real) * false_pass + share_real * power) if power + false_pass > 0 else float("nan")
    statuses = dict(db.execute("SELECT COALESCE(status, 'observing'), COUNT(*) FROM bindings GROUP BY 1").fetchall())
    outcomes = [json.loads(r[0]) for r in db.execute("SELECT outcome FROM bindings WHERE status = 'settled'")]
    rewards = [o["reward"] for o in outcomes]
    supported = sum(o.get("prediction") == "supported" for o in outcomes)
    contradicted = sum(o.get("prediction") == "contradicted" for o in outcomes)
    if not families:
        log.info("maturity scorecard on an empty ledger: the bar shown is for a family of one")
    studies = db.execute("SELECT COUNT(*), SUM(json_extract(result, '$.passed')) FROM outbox WHERE status = 'done'").fetchone()
    failed = db.execute("SELECT COUNT(*) FROM outbox WHERE status = 'failed'").fetchone()[0]
    return {
        "years_per_study": years,
        "families": len(families),
        "median_family_size": n,
        "deflated_bar": c,
        f"power_at_sharpe_{edge}": power,
        f"false_share_of_passes_if_{share_real:g}_real": false_share,
        "studies": int(studies[0] or 0),
        "passes": int(studies[1] or 0),
        "studies_failed": int(failed),
        "forward_observing": statuses.get("observing", 0),
        "forward_settled": len(rewards),
        "forward_invalidated": statuses.get("invalidated", 0),
        "forward_incomplete": statuses.get("incomplete", 0),
        "forward_predictions_supported": supported,
        "forward_predictions_contradicted": contradicted,
        "forward_mean_reward": float(np.mean(rewards)) if rewards else None,
        "forward_reward_se": float(np.std(rewards, ddof=1) / math.sqrt(len(rewards))) if len(rewards) > 1 else None,
        "years_for_half_power": (c / edge) ** 2,
    }
