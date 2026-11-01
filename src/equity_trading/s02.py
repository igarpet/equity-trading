"""S02 gates and summaries (kept out of the notebook so they are testable)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .coint import rate_diff_ci

S02_VERSION = "0.1.1"
ALPHA = 0.05
G1_MIN_RATIO = 2.0      # null-group pass rate multiple
G2_MIN_LIFT = 0.10      # lift in OOS band-hold share vs null-group passers


def g1(res: pd.DataFrame, group: str, null_group: str, **kw) -> dict:
    """Formation pass rate in `group` vs its empirical null group."""
    hit = (res["p"] <= ALPHA).to_numpy()
    r = rate_diff_ci(res["window"].to_numpy(), hit, (res["group"] == group).to_numpy(),
                     hit, (res["group"] == null_group).to_numpy(), **kw)
    r["pass"] = bool(r["diff_lo"] > 0 and r["ratio"] >= G1_MIN_RATIO)
    return r


def g2(res: pd.DataFrame, group: str, null_group: str, **kw) -> dict:
    """Persistence (amended pre-data, see S02_README): among formation passers, mean OOS band-hold
    share in `group` minus the same in `null_group` (whose passes are mostly spurious)."""
    d = res[res["band_hold"].notna() & (res["p"] <= ALPHA)]
    val = d["band_hold"].to_numpy()
    r = rate_diff_ci(d["window"].to_numpy(), val, (d["group"] == group).to_numpy(),
                     val, (d["group"] == null_group).to_numpy(), **kw)
    r["pass"] = bool(r["diff"] >= G2_MIN_LIFT and r["diff_lo"] > 0)
    return r


def adf_lift(res: pd.DataFrame, group: str, **kw) -> dict:
    """Descriptive (the original G2): OOS ADF-stationary rate, formation passers vs failers."""
    d = res[(res["group"] == group) & res["p_oos"].notna()]
    hit = (d["p_oos"] <= ALPHA).to_numpy()
    passed = (d["p"] <= ALPHA).to_numpy()
    return rate_diff_ci(d["window"].to_numpy(), hit, passed, hit, ~passed, **kw)


def verdict(g1r: dict, g2r: dict) -> str:
    if g1r["pass"] and g2r["pass"]:
        return "PERSISTENT_COINTEGRATION"
    if g1r["pass"]:
        return "SIGNAL_NOT_PERSISTENT"
    return "NO_COINTEGRATION_SIGNAL"


def describe(res: pd.DataFrame) -> pd.DataFrame:
    """Descriptive table per group: pass rate, OOS stationarity, drift and crossings by pass status."""
    rows = []
    for g, d in res.groupby("group"):
        ps = d["p"] <= ALPHA
        rows.append({"group": g, "pair_windows": len(d), "pass_rate": ps.mean(),
                     "oos_stat_rate_passers": (d.loc[ps, "p_oos"] <= ALPHA).mean(),
                     "oos_stat_rate_others": (d.loc[~ps, "p_oos"] <= ALPHA).mean(),
                     "band_hold_passers": d.loc[ps, "band_hold"].mean(),
                     "band_hold_others": d.loc[~ps, "band_hold"].mean(),
                     "median_drift_sd_passers": d.loc[ps, "drift_sd"].median(),
                     "median_drift_sd_others": d.loc[~ps, "drift_sd"].median(),
                     "median_half_life_bars_passers": d.loc[ps, "half_life_bars"].median(),
                     "crossings_per_day_passers": d.loc[ps, "crossings_per_day"].median()})
    return pd.DataFrame(rows).set_index("group")


def pass_rate_over_time(res: pd.DataFrame) -> pd.DataFrame:
    return (res.assign(hit=res["p"] <= ALPHA)
               .groupby(["form_start", "group"])["hit"].mean().unstack("group"))
