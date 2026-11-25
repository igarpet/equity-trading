"""HMK 01: hierarchical Markov regimes on MNQ (discussions/MNQ_Hierarchical_Markov/001)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import hmk as H
from equity_trading import micro_bars as MB
from equity_trading.futures import CHICAGO

D = dt.date


def _sim(T=20000, seed=0):
    """2-state child chain whose transition matrix depends on a known 2-state parent (one-hot)."""
    rng = np.random.default_rng(seed)
    A = np.array([[[0.95, 0.05], [0.30, 0.70]], [[0.70, 0.30], [0.05, 0.95]]])
    mu, sd = np.array([-5.0, 5.0]), np.array([4.0, 4.0])
    parent = (np.arange(T) // 500) % 2
    s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=A[parent[t], s[t - 1]])
    y = rng.normal(mu[s], sd[s])
    W = np.zeros((T, 2)); W[np.arange(T), parent] = 1
    return y, W, A, mu


def test_em_recovers_parent_dependent_transitions():
    y, W, A, mu = _sim()
    m = H.fit(y, np.ones(len(y), bool), W, H.HMKConfig(K=2, n_starts=3))
    assert np.allclose(m.mu, mu, atol=0.3)
    assert np.allclose(m.A, A, atol=0.04)


def test_filter_is_causal():
    y, W, _, _ = _sim(3000)
    m = H.fit(y, np.ones(len(y), bool), W, H.HMKConfig(K=2, n_starts=1))
    obs = np.ones(len(y), bool)
    _, p1 = H.filter_run(m, y, obs, W)
    y2 = y.copy(); y2[2000:] = 100.0
    _, p2 = H.filter_run(m, y2, obs, W)
    assert np.allclose(p1[:2001], p2[:2001])             # P(s_t | F_{t-1}) for t <= 2000 ignores y_2000 onwards
    assert np.allclose(p1.sum(1), 1)


def _day_minutes(day, iid=1, price=10000.0):
    open_ = pd.Timestamp(day - dt.timedelta(days=1), tz=CHICAGO) + pd.Timedelta(hours=17)
    ts = (open_ + pd.to_timedelta(np.arange(H.SESSION_MIN), unit="min")).tz_convert("UTC")
    c = price + 0.25 * np.arange(H.SESSION_MIN)
    return pd.DataFrame({"close": c, "volume": 1, "instrument_id": iid}, index=pd.DatetimeIndex(ts, name="ts"))


def test_session_bars_grid_and_roll():
    m1 = pd.concat([_day_minutes(D(2021, 3, 8)), _day_minutes(D(2021, 3, 9), iid=2, price=11000.0)])
    b = H.session_bars(m1)
    assert len(b["1h"]) == 46 and len(b["15m"]) == 184 and len(b["1d"]) == 2
    h = b["1h"]
    assert not h.obs[h["first"]].any()                   # first bar of each day never observed
    assert h.loc[1, "ret_bp"] == pytest.approx(1e4 * np.log((10000 + 0.25 * 119) / (10000 + 0.25 * 59)))
    assert not b["1d"].obs.iloc[1]                       # contract switch between the two days
    assert h.ts.iloc[0] == pd.Timestamp("2021-03-07 23:00", tz="UTC")


def test_positions_costs():
    cfg = H.HMKConfig()
    df = pd.DataFrame({"trade_date": [D(2021, 1, 4)] * 4, "k": [0, 1, 2, 3], "close": [100, 101, 100, 102.0],
                       "e_bp": [0, 1.0, -1.0, -1.0], "first": [True, False, False, False],
                       "obs": [False, True, True, True]})
    p = H.daily_pnl(df, cfg).iloc[0]
    # long bar 1 (+4 ticks), short bar 2 (+4), short bar 3 (-8): gross 0; units: 1 entry + 2 flip + 1 exit = 4
    assert p.gross == 0 and p.units == 4
    assert p.net_market == pytest.approx(-4 * cfg.half_rt_market)
    assert p.long_gross == pytest.approx(8.0)


def test_guard_2026():
    with pytest.raises(RuntimeError):
        H.guard(D(2026, 1, 1))


SYN = H.HMKConfig(n_starts=2, refits_discovery=(D(2020, 7, 1),), discovery=(D(2020, 7, 1), D(2020, 12, 31)))


def _syn_eval(tmp_path, effect):
    H.write_synthetic(tmp_path, D(2019, 11, 1), D(2020, 12, 31), effect=effect)
    bars = H.session_bars(MB.load_1m("MNQ", D(2019, 11, 1), D(2021, 1, 1)), SYN)
    wf = H.walk_forward(bars, SYN, SYN.refits_discovery, SYN.discovery[1])
    return H.evaluate(wf["pred"]["1h_H3"], wf["pred"]["1h_U"], SYN, "1h_H3")


def test_planted_structure_is_found(tmp_path):
    r = _syn_eval(tmp_path, 2.0)
    assert r["bss"] > 0 and r["g2_p"] < 0.05 and r["net_ticks_day"] > 0


def test_null_has_no_skill(tmp_path):
    r = _syn_eval(tmp_path, 0.0)
    assert r["g2_p"] > 0.05
