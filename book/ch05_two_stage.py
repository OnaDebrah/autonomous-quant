"""Chapter 5, two-stage scanning: rank sectors, then rank the assets inside the chosen sector. Needs numpy and scipy.

1. No differences anywhere: the score of the best asset in the best sector.
2. Skill at both stages: where the pick's return comes from, and what concentrating in one sector costs in risk.
Run: python book/ch05_two_stage.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20261117
SECTORS = 11
MEMBERS = 40


def expected_order_statistics(n: int, rng, draws: int = 400_000) -> np.ndarray:
    return np.sort(rng.standard_normal((draws, n)), axis=1)[:, ::-1].mean(axis=0)


def null_drill_down(rng) -> dict:
    """Asset score = sqrt(rho) * its sector's common part + sqrt(1 - rho) * its own part; nothing differs."""
    runs, rho = 40_000, 0.3
    sector = rng.standard_normal((runs, SECTORS))
    own = rng.standard_normal((runs, SECTORS, MEMBERS))
    score = math.sqrt(rho) * sector[:, :, None] + math.sqrt(1 - rho) * own
    best_sector = sector.argmax(axis=1)
    inside = score[np.arange(runs), best_sector]
    m_sector = expected_order_statistics(SECTORS, rng)[0]
    m_member = expected_order_statistics(MEMBERS, rng)[0]
    return {
        "rho": rho,
        "theory_top_asset_in_top_sector": math.sqrt(rho) * m_sector
        + math.sqrt(1 - rho) * m_member,
        "top_asset_in_top_sector": float(inside.max(axis=1).mean()),
        "mean_asset_in_top_sector": float(inside.mean()),
        "theory_mean_asset_in_top_sector": math.sqrt(rho) * m_sector,
        "top_asset_overall": float(score.reshape(runs, -1).max(axis=1).mean()),
        "top_overall_is_in_top_sector": float(
            (score.reshape(runs, -1).argmax(axis=1) // MEMBERS == best_sector).mean()
        ),
    }


def skilled_drill_down(rng) -> dict:
    """Monthly returns = sector part + own part; the sector score predicts the first, the asset score the second."""
    months, q = 20_000, 0.25
    sector_sd, own_sd, ic_sector, ic_own = 0.03, 0.08, 0.05, 0.03
    zs = rng.standard_normal((months, SECTORS))
    zo = rng.standard_normal((months, SECTORS, MEMBERS))
    sector_ret = sector_sd * (
        ic_sector * zs
        + math.sqrt(1 - ic_sector**2) * rng.standard_normal((months, SECTORS))
    )
    own_ret = own_sd * (
        ic_own * zo
        + math.sqrt(1 - ic_own**2) * rng.standard_normal((months, SECTORS, MEMBERS))
    )
    ret = sector_ret[:, :, None] + own_ret
    universe = ret.mean(axis=(1, 2))
    k = int(MEMBERS * q)
    order = np.argsort(-zs, axis=1)
    order_stats = expected_order_statistics(SECTORS, rng)
    h = norm.pdf(norm.ppf(1 - q)) / q
    out = {"h": h, "sector_order_statistics_top3": [float(x) for x in order_stats[:3]]}
    for name, top in (("one_sector", 1), ("three_sectors", 3)):
        picks, sector_only = [], []
        for j in range(top):
            s = order[:, j]
            members_z = zo[np.arange(months), s]
            members_r = ret[np.arange(months), s]
            chosen = np.argpartition(-members_z, k, axis=1)[:, :k]
            picks.append(np.take_along_axis(members_r, chosen, axis=1).mean(axis=1))
            sector_only.append(members_r.mean(axis=1))
        active = np.mean(picks, axis=0) - universe
        sector_active = np.mean(sector_only, axis=0) - universe
        intensity = float(order_stats[:top].mean())
        out[name] = {
            "theory_from_sector_stage": ic_sector * sector_sd * intensity,
            "from_sector_stage": float(sector_active.mean()),
            "theory_from_asset_stage": ic_own * own_sd * h,
            "from_asset_stage": float((active - sector_active).mean()),
            "mean_active_monthly": float(active.mean()),
            "active_risk_monthly": float(active.std(ddof=1)),
            "annual_information_ratio": float(
                active.mean() / active.std(ddof=1) * math.sqrt(12)
            ),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "seed": SEED,
        "sectors": SECTORS,
        "members": MEMBERS,
        "null_drill_down": null_drill_down(rng),
        "skilled_drill_down": skilled_drill_down(rng),
    }
    Path(__file__).with_name("ch05_two_stage_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
