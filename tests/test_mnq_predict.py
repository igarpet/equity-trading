import datetime as dt
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import mnq_predict as P  # noqa: E402

D = dt.date
CFG = replace(P.MRPConfig(), pairs=("EURUSD", "USDJPY"))


def _m1(days=(D(2024, 3, 5), D(2024, 3, 6)), seed=0, drop=None, switch=None):
    """1-minute bars for whole CME days (17:00-16:00 Chicago)."""
    rng = np.random.default_rng(seed)
    parts = []
    for d in days:
        open_ = pd.Timestamp(d - dt.timedelta(days=1), tz=P.CHICAGO) + pd.Timedelta(hours=17)
        ts = (open_ + pd.to_timedelta(np.arange(23 * 60), unit="min")).tz_convert("UTC")
        parts.append(pd.DataFrame({"close": 18000 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(ts)))),
                                   "volume": 5, "instrument_id": 1}, index=ts))
    m = pd.concat(parts)
    if switch is not None:
        m.loc[m.index >= switch, "instrument_id"] = 2
        m.loc[m.index >= switch, "close"] += 80
    if drop is not None:
        m = m[~((m.index >= drop[0]) & (m.index < drop[1]))]
    m.index.name = "ts"
    return m


def test_guard():
    with pytest.raises(P.HoldoutError):
        P.guard(D(2026, 1, 1))


def test_grid_shape_slots_and_first_bar():
    g = P.grid_from_1m(_m1(), CFG)
    assert len(g) == 2 * 276
    assert g["slot"].min() == 0 and g["slot"].max() == 275
    first = g[g["slot"] == 0]
    assert first["ret"].isna().all()                                  # no return across the daily break
    loc = first.index.tz_convert(P.CHICAGO)
    assert (loc.hour == 17).all() and (loc.minute == 0).all()
    assert g["ret"].iloc[1:276].notna().all()
    # bar return = log close ratio of consecutive bars
    c = g["close"].to_numpy()
    assert np.isclose(g["ret"].iloc[5], np.log(c[5] / c[4]))


def test_grid_gap_and_roll():
    d = D(2024, 3, 5)
    t0 = pd.Timestamp("2024-03-05 03:00", tz="UTC")
    g = P.grid_from_1m(_m1(days=(d,), drop=(t0, t0 + pd.Timedelta(minutes=20))), CFG)
    gap = g.loc[t0:t0 + pd.Timedelta(minutes=20)]
    assert gap["ret"].isna().all() and gap["gap"].iloc[:4].all()      # 4 empty bars + the bar after
    g2 = P.grid_from_1m(_m1(days=(d,), drop=(t0, t0 + pd.Timedelta(minutes=10))), CFG)
    short = g2.loc[t0:t0 + pd.Timedelta(minutes=5)]
    assert (short["ret"] == 0).all()                                  # 2 empty bars: carried close, return 0
    sw = pd.Timestamp("2024-03-05 00:00", tz="UTC")
    g3 = P.grid_from_1m(_m1(days=(d,), switch=sw), CFG)
    assert g3.loc[sw, "roll"] and np.isnan(g3.loc[sw, "ret"])
    assert g3["ret"].abs().max() < 0.01                               # basis jump never a return


def test_sessions_local_time_dst():
    # 2024-03-12: New York on daylight time, London not yet (mismatch week)
    ts = pd.DatetimeIndex(pd.to_datetime(["2024-03-12 08:00", "2024-03-12 13:25", "2024-03-12 13:30",
                                          "2024-03-12 20:00", "2024-03-12 20:55", "2024-03-12 22:00"], utc=True))
    assert list(P.session_of(ts)) == ["london", "london", "ny", "post", "post", "asia"]
    ts2 = pd.DatetimeIndex(pd.to_datetime(["2024-01-10 07:55", "2024-01-10 08:00", "2024-01-10 14:30",
                                           "2024-01-10 21:00"], utc=True))
    assert list(P.session_of(ts2)) == ["asia", "london", "ny", "post"]


def test_fx_on_grid_alignment():
    idx = pd.date_range("2024-03-05 10:00", periods=30, freq="1min", tz="UTC")
    mid = pd.Series(np.exp(np.arange(30) * 1e-4), index=idx)
    g = pd.DatetimeIndex([pd.Timestamp("2024-03-05 10:10", tz="UTC")])
    out = P.fx_on_grid(mid, g)
    # bar [10:10, 10:15): close of minute 10:14 vs close of 10:09 -> 5 steps
    assert np.isclose(out["ret"].iloc[0], 5e-4)


def _dataset(seed=0, mutate_after=None):
    days = tuple(D(2024, 3, 4) + dt.timedelta(days=i) for i in range(10) if (D(2024, 3, 4) + dt.timedelta(days=i)).weekday() < 5)
    m1 = _m1(days=days, seed=seed)
    if mutate_after is not None:
        m1.loc[m1.index >= mutate_after, "close"] *= 1.01
    g = P.grid_from_1m(m1, CFG)
    s2 = pd.Series(1e-6, index=g.index)
    fx, fx2 = {}, {}
    for i, p in enumerate(CFG.pairs):
        rng = np.random.default_rng(seed + 10 + i)
        fx[p] = pd.DataFrame({"ret": rng.normal(0, 1e-4, len(g)), "rv": 1e-8}, index=g.index)
        fx2[p] = pd.Series(1e-8, index=g.index)
    return g, P.build_dataset(g, s2, fx, fx2, CFG)


def test_dataset_columns_and_exclusions():
    g, (X, y, meta) = _dataset()
    cols = set(sum(P.groups(CFG).values(), [])) | set(P.BASE)
    assert cols == set(X.columns)
    assert (meta["slot"] >= CFG.drop_first).all()
    # target is the bar's own z; lag 1 is the previous bar's z (same day)
    t = X.index[100]
    pos = g.index.get_loc(t)
    assert np.isclose(X.loc[t, "z_l1"], g["ret"].iloc[pos - 1] / 1e-3)
    assert np.isclose(y.loc[t], g["ret"].iloc[pos] / 1e-3)


def test_no_lookahead():
    _, (X, _, _) = _dataset()
    cut = pd.Timestamp("2024-03-07 15:00", tz="UTC")
    _, (X2, _, _) = _dataset(mutate_after=cut)
    before = X.index[X.index <= cut]
    pd.testing.assert_frame_equal(X.loc[before], X2.loc[before])


def test_session_sums_completed_only():
    g, (X, _, meta) = _dataset()
    # during NY on day k, sess_ny must be the previous day's NY session, sess_london today's
    ny_rows = meta[meta["session"] == "ny"]
    d0 = ny_rows["trade_date"].unique()[1]
    a = X.loc[ny_rows.index[ny_rows["trade_date"] == d0], "sess_ny"]
    assert a.nunique() == 1                                          # constant inside the session (yesterday's)
    b = X.loc[ny_rows.index[ny_rows["trade_date"] == d0], "sess_london"]
    assert b.nunique() == 1
    lon = meta.index[(meta["session"] == "london") & (meta["trade_date"] == d0)]
    assert X.loc[lon, "sess_london"].iloc[0] != b.iloc[0]            # London value changes once London completes


def test_cluster_cov_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n = 2000
    g = np.repeat(np.arange(100), 20)
    X = pd.DataFrame({"const": 1.0, "a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(0.3 * X["a"] + rng.normal(size=n) + np.repeat(rng.normal(size=100), 20))
    r = P.ols(X, y, clusters=g)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert np.allclose(r.beta, ref.params.to_numpy())
    assert np.allclose(np.sqrt(np.diag(r.cov)), ref.bse.to_numpy(), rtol=1e-6)
    w = P.wald(r, ["a", "b"])
    assert w["p"] < 1e-6


def test_holm_and_clark_west():
    h = P.holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert list(h.index) == ["a", "b", "c"] and np.allclose(h["p_holm"], [0.03, 0.04, 0.5])
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    y = 0.3 * x + rng.normal(size=5000)
    day = np.repeat(np.arange(100), 50)
    assert P.clark_west(y, np.zeros(5000), 0.3 * x, day)["p_one_sided"] < 1e-6


def test_vol_fit_beats_tod_on_clustered_vol():
    days = tuple(D(2024, 1, 1) + dt.timedelta(days=i) for i in range(90) if (D(2024, 1, 1) + dt.timedelta(days=i)).weekday() < 5)
    rng = np.random.default_rng(3)
    parts, lv = [], 0.0
    for d in days:
        lv = 0.9 * lv + 0.4 * rng.normal()
        open_ = pd.Timestamp(d - dt.timedelta(days=1), tz=P.CHICAGO) + pd.Timedelta(hours=17)
        ts = (open_ + pd.to_timedelta(np.arange(23 * 60), unit="min")).tz_convert("UTC")
        parts.append(pd.DataFrame({"close": 18000 * np.exp(np.cumsum(rng.normal(0, 2e-4 * np.exp(lv), len(ts)))),
                                   "volume": 5, "instrument_id": 1}, index=ts))
    g = P.grid_from_1m(pd.concat(parts), CFG)
    pred, _ = P.walk_forward_vol(g, days[0], (days[40],), days[-1], CFG)
    gate = P.vol_gate(g, pred, CFG)
    assert gate["qlike_full"] < gate["qlike_tod"] and gate["V1"]


def test_one_bar_pnl_costs():
    fc = pd.DataFrame({"mu_M3": [5.0, -5.0, 0.0], "sig2": [1e-6] * 3, "close_prev": [20000.0] * 3,
                       "close": [20010.0, 20010.0, 20010.0], "trade_date": [D(2024, 1, 2)] * 3})
    t = P.one_bar_pnl(fc, "M3", CFG)
    assert list(t["pos"]) == [1, -1, 0]
    cost = CFG.tick * CFG.usd_per_point + 2 * CFG.commission_per_side
    assert np.allclose(t["net_usd"], [20 - cost, -20 - cost, 0])
