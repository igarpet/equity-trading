import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import burst_entry as E

RC = E.BEConfig()


def _m1(n=200, start="2024-03-05 15:00", gaps=()):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    keep = np.ones(n, bool)
    keep[list(gaps)] = False
    c = 18000 + np.arange(n) * 0.25
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1, "instrument_id": 1},
                        index=ts)[keep]


def test_grid_returns_only_across_consecutive_minutes():
    g = E.grid_1m(_m1(gaps=(10,)))
    i = g.index.get_loc(pd.Timestamp("2024-03-05 15:11", tz="UTC"))
    assert np.isnan(g["ret"].iat[i]) and np.isfinite(g["ret"].iat[i + 1])
    assert np.isnan(g["ret"].iat[0])


def _series(close, high=None, low=None, start="2024-03-05 15:00"):
    n = len(close)
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = np.asarray(close, float)
    s = pd.DataFrame({"trade_date": dt.date(2024, 3, 5), "slot": 100, "z": 0.0, "zl": 0.0, "obs": True,
                      "sig2": (2 / 18000) ** 2, "ret": np.r_[np.nan, np.diff(np.log(close))], "open": np.r_[close[0], close[:-1]],
                      "high": close + 0.25 if high is None else high, "low": close - 0.25 if low is None else low,
                      "close": close, "iid": 1}, index=ts)
    return s


def test_entry_with_and_against_and_time_exit():
    close = np.r_[np.full(10, 18000.0), 18010.0, 18012 + np.arange(30) * 0.5]
    s = _series(close)
    pb = np.zeros(len(s))
    pb[10:14] = 0.9
    ev = E.burst_starts(s, pb, np.full(len(s), 6), RC)
    assert list(ev["row"]) == [10] and ev["candle_sign"].iat[0] == 1
    flat = E.flat_index(s, RC)
    w = E.simulate(s, flat, ev, "none", ev["candle_sign"].to_numpy(), RC)
    # entry 18010 + 0.5 tick, exit at close of row 16 (18014.5) - 0.5 tick: 4.5 points = 18 ticks - 1
    assert w["reason"].iat[0] == "time" and w["gross_ticks"].iat[0] == pytest.approx(17.0)
    a = E.simulate(s, flat, ev, "none", -ev["candle_sign"].to_numpy(), RC)
    assert a["gross_ticks"].iat[0] == pytest.approx(-19.0)


def test_extreme_stop_and_flat_time():
    close = np.r_[np.full(10, 18000.0), 18010.0, np.full(10, 18000.0)]
    low = close - 0.25
    low[10] = 18005.0
    s = _series(close, low=low)
    pb = np.zeros(len(s)); pb[10] = 0.9
    ev = E.burst_starts(s, pb, np.full(len(s), 6), RC)
    flat = E.flat_index(s, RC)
    tr = E.simulate(s, flat, ev, "extreme", np.array([1]), RC)
    assert tr["reason"].iat[0] == "stop" and tr["stop_ticks"].iat[0] == pytest.approx(round((18010.125 - 18004.5) / 0.25))
    # same pattern with the entry minute at 15:54 Chicago (21:54 UTC on 2024-03-05, CST): no room before 15:55
    s2 = _series(close, start="2024-03-05 21:44")
    ev2 = E.burst_starts(s2, pb, np.full(len(s2), 6), RC)
    tr2 = E.simulate(s2, E.flat_index(s2, RC), ev2, "none", np.array([1]), RC)
    assert len(tr2) == 0 and tr2.attrs["skipped"] == 1


def test_vol_model_runs_and_is_ex_ante():
    rng = np.random.default_rng(0)
    n = 1380 * 8
    ts = pd.date_range("2024-01-01 23:00", periods=n, freq="1min", tz="UTC")
    c = 18000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    g = E.grid_1m(pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1, "instrument_id": 1}, index=ts))
    f = E.fit_vol_1m(g, g["trade_date"].min(), g["trade_date"].max(), RC)
    v = E.predict_vol_1m(g, f, RC)
    g2 = g.copy()
    g2.iloc[-100:, g2.columns.get_loc("ret")] *= 50
    v2 = E.predict_vol_1m(g2, f, RC)
    assert np.allclose(v[:-100], v2[:-100], equal_nan=True)
