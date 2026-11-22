import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import burst_reversion as B
from equity_trading import trail_strategy as T

RC = B.BRConfig()


def _setup(path_points, n_bars=40, burst=(10, 12)):
    """One CME day of 5-minute bars at 15:00 UTC onwards; minute closes interpolate the 5-minute path."""
    idx = pd.date_range("2024-03-05 14:30", periods=n_bars, freq="5min", tz="UTC")
    close5 = np.asarray(path_points, float)
    s = pd.DataFrame({"close": close5, "trade_date": pd.Timestamp("2024-03-05").date(), "slot": 100, "obs": True,
                      "sig2": (5 / 18000) ** 2, "iid": 1}, index=idx)
    pb = np.zeros(n_bars)
    pb[burst[0]:burst[1]] = 0.9
    mts = pd.date_range(idx[0], periods=n_bars * 5, freq="1min", tz="UTC")
    c1 = np.repeat(close5, 5)
    m1 = pd.DataFrame({"open": c1, "high": c1, "low": c1, "close": c1, "volume": 1, "instrument_id": 1}, index=mts)
    M = T.minutes_from_1m(m1, T.TrailConfig())
    return s, pb, M


def test_episode_direction_and_time_exit():
    path = np.full(40, 18000.0)
    path[10:12] = [17980, 17960]                 # burst down 40 points
    path[12:] = 17960 + np.arange(28) * 1.0      # slow recovery, +1 point per bar
    s, pb, M = _setup(path)
    ev = B.episodes(s, pb, np.full(40, 10), RC)
    assert len(ev) == 1 and ev.loc[0, "side"] == 1 and ev.loc[0, "switch_row"] == 12
    tr = B.simulate(M, ev, s.index, "none", RC)
    # entry at close of bar 12 (17960) + 0.5 tick; exit at close of bar 22 (17970) - 0.5 tick: 40 - 1 ticks gross
    assert tr.loc[0, "reason"] == "time"
    assert tr.loc[0, "gross_ticks"] == pytest.approx(39.0)
    assert tr.loc[0, "net_ticks"] == pytest.approx(37.0)       # minus 2 x 1-tick commission


def test_stops():
    path = np.full(40, 18000.0)
    path[10:12] = [17980, 17960]
    path[12:20] = [17960, 17955, 17945, 17930, 17920, 17910, 17900, 17890]
    path[20:] = 17890.0
    s, pb, M = _setup(path)
    ev = B.episodes(s, pb, np.full(40, 10), RC)
    fixed = B.simulate(M, ev, s.index, "fixed40", RC).iloc[0]
    assert fixed["reason"] == "stop" and fixed["gross_ticks"] < -40    # 10 points + gap past the stop + slippage
    ext = B.simulate(M, ev, s.index, "extreme", RC).iloc[0]
    assert ext["stop_ticks"] == pytest.approx(round((17960.125 - (17960 - 0.5)) / 0.25))   # low 17960 - 2 ticks
    none = B.simulate(M, ev, s.index, "none", RC).iloc[0]
    assert none["reason"] == "time" and none["gross_ticks"] < fixed["gross_ticks"]


def test_no_overlap_and_mirror():
    path = np.full(60, 18000.0)
    path[10:12] = [17980, 17960]
    path[14:16] = [17990, 18020]                 # second burst while the first trade is open
    s, pb, M = _setup(path, n_bars=60)
    pb[14:16] = 0.9
    ev = B.episodes(s, pb, np.full(60, 10), RC)
    assert len(ev) == 2
    tr = B.simulate(M, ev, s.index, "none", RC)
    assert len(tr) == 1
    mir = B.simulate(M, ev, s.index, "none", RC, sides=-ev["side"].to_numpy())
    assert mir.loc[0, "side"] == -1


def test_no_trade_across_the_daily_break():
    idx = pd.date_range("2024-03-05 21:25", periods=7, freq="5min", tz="UTC")       # 15:25-15:55 Chicago
    idx2 = pd.date_range("2024-03-05 23:00", periods=20, freq="5min", tz="UTC")     # next CME day from 17:00
    ix = idx.append(idx2)
    close = np.r_[18000, 18000, 18000, 18000, 17980, 17960, 17960, np.full(20, 18100.0)]
    td = [pd.Timestamp("2024-03-05").date()] * 7 + [pd.Timestamp("2024-03-06").date()] * 20
    s = pd.DataFrame({"close": close, "trade_date": td, "slot": [268 + i for i in range(7)] + list(range(20)),
                      "obs": True, "sig2": (5 / 18000) ** 2, "iid": 1}, index=ix)
    pb = np.zeros(len(s))
    pb[4:6] = 0.9                                   # switch on the 15:55-16:00 bar
    mts = ix.repeat(5) + pd.to_timedelta(np.tile(np.arange(5), len(ix)), unit="min")
    c1 = np.repeat(close, 5)
    m1 = pd.DataFrame({"open": c1, "high": c1, "low": c1, "close": c1, "volume": 1, "instrument_id": 1}, index=mts)
    M = T.minutes_from_1m(m1, T.TrailConfig())
    ev = B.episodes(s, pb, np.full(len(s), 10), RC)
    tr = B.simulate(M, ev, s.index, "none", RC)
    assert len(ev) == 1 and len(tr) == 0 and tr.attrs["skipped_session_end"] == 1
