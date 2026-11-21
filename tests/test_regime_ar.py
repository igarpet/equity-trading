import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import mnq_predict as P
from equity_trading import regime_ar as R

RC = R.RSWConfig(n_starts=4, max_iter=300)


def _series(z, day=None, extras=None):
    T = len(z)
    day = np.repeat(np.arange(T // 200 + 1), 200)[:T] if day is None else day
    same = np.r_[False, day[1:] == day[:-1]]
    s = pd.DataFrame({"trade_date": day, "z": z, "zl": np.where(same, np.r_[0.0, z[:-1]], 0.0),
                      "obs": np.ones(T, bool), "imb": 0.0, "vsurp": 0.0})
    if extras is not None:
        s["imb"], s["vsurp"] = extras
    return s


def test_em_recovers_planted_regimes():
    z, st = R.simulate_msar(40_000, phis=(-0.3, 0.3), p_stay=0.98, seed=1)
    s = _series(z, day=np.zeros(len(z), int))
    m = R.fit(s, 2, False, np.ones(len(s), bool), RC)
    assert m.phi[0] == pytest.approx(-0.3, abs=0.05)
    assert m.phi[1] == pytest.approx(0.3, abs=0.05)
    assert np.diag(m.A) == pytest.approx([0.98, 0.98], abs=0.02)
    g = R.smooth(s, m)
    assert np.mean(g.argmax(1) == st) > 0.75
    assert R.phi_gap(m)["t_gap"] > 10


def test_one_state_is_ols_ar1():
    rng = np.random.default_rng(0)
    z = np.empty(5000)
    z[0] = 0
    for t in range(1, len(z)):
        z[t] = 0.1 * z[t - 1] + rng.normal()
    s = _series(z, day=np.zeros(len(z), int))
    m = R.fit(s, 1, False, np.ones(len(s), bool), RC)
    X = np.column_stack([np.ones(len(z) - 1), z[:-1]])
    b = np.linalg.lstsq(X, z[1:], rcond=None)[0]
    assert m.phi[0] == pytest.approx(b[1], abs=2e-3)


def test_filter_is_causal():
    z, _ = R.simulate_msar(3000, seed=2)
    s = _series(z)
    m = R.fit(s, 2, False, np.ones(len(s), bool), RC)
    f1 = R.filter_run(s, m)
    s2 = s.copy()
    s2.loc[2000:, "z"] = 0.0
    s2.loc[2000:, "zl"] = 0.0
    f2 = R.filter_run(s2, m)
    pd.testing.assert_frame_equal(f1.iloc[:2000], f2.iloc[:2000])


def test_unobserved_rows_only_propagate():
    z, _ = R.simulate_msar(2000, seed=3)
    s = _series(z)
    s.loc[500:520, "obs"] = False
    m = R.fit(s, 2, False, np.ones(len(s), bool), RC)
    f = R.filter_run(s, m)
    assert f.loc[500:520, "lpz"].isna().all()
    p = f[["p0", "p1"]].to_numpy()
    # without observations the filtered probabilities move toward the stationary distribution
    assert np.abs(p[520] - p[499] @ np.linalg.matrix_power(m.A, 21)).max() < 1e-10


def test_tvtp_recovers_flow_driven_switching():
    rng = np.random.default_rng(4)
    T = 30_000
    x = rng.normal(0, 1, T)                                    # vsurp of bar t-1 drives the move into bar t
    st = np.zeros(T, int)
    z = np.zeros(T)
    phis = (-0.3, 0.3)
    for t in range(1, T):
        p_leave = 1 / (1 + np.exp(-(-3.5 + 2.0 * x[t - 1])))
        st[t] = 1 - st[t - 1] if rng.random() < p_leave else st[t - 1]
        z[t] = phis[st[t]] * z[t - 1] + rng.normal()
    s = _series(z, day=np.zeros(T, int), extras=(np.zeros(T), x))
    m = R.fit(s, 2, True, np.ones(T, bool), RC)
    tab = m.table()
    assert tab["leave_logit_vsurp"].to_numpy() == pytest.approx([2.0, 2.0], abs=0.6)
    assert m.phi == pytest.approx(phis, abs=0.05)
    m0 = R.fit(s, 2, False, np.ones(T, bool), RC)
    assert m.loglik > m0.loglik + 50
    f = R.filter_run(s, m)
    assert f[["p0", "p1"]].sum(axis=1).to_numpy() == pytest.approx(np.ones(T))


def test_bvc_sign_and_bounds():
    ts = pd.date_range("2024-03-05 14:30", periods=400, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1e-4, len(ts))
    r[300:305] = 5e-4                                          # one strongly rising 5-minute slot
    close = 18000 * np.exp(np.cumsum(r))
    m1 = pd.DataFrame({"close": close, "volume": 10, "instrument_id": 1}, index=ts)
    b = R.bvc_imbalance(m1)
    assert b["imb"].dropna().between(-1, 1).all()
    slot = P._slot(ts[300:301])[0]
    assert b.loc[b["slot"] == slot, "imb"].iloc[0] > 0.9


def test_volume_surprise_is_ex_ante():
    days = [dt.date(2024, 1, d) for d in range(1, 31)]
    g = pd.DataFrame({"trade_date": np.repeat(days, 2), "slot": np.tile([0, 1], len(days)),
                      "volume": np.r_[np.full(58, 100.0), 1000.0, 100.0], "empty": False})
    v = R.volume_surprise(g, R.RSWConfig(vol_days=5, vol_min_days=3))
    assert v[58] == pytest.approx(np.log1p(1000) - np.log1p(100))   # the spike is not in its own baseline
    assert v[59] == pytest.approx(0.0)
    assert np.isnan(v[:4]).all()


def test_holm_and_trades():
    h = R.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert h["a"] == pytest.approx(0.03) and h["b"] == pytest.approx(0.08) and h["c"] == pytest.approx(0.5)
    idx = pd.date_range("2024-01-02", periods=4, freq="5min", tz="UTC")
    s = pd.DataFrame({"trade_date": [dt.date(2024, 1, 2)] * 4, "session": "ny", "close": [100, 101, 100, 102.0],
                      "sig2": 1e-4, "obs": True, "ret": 0.0}, index=idx)
    tr = R.one_bar_trades(s, np.array([np.nan, 1.0, -1.0, 0.01]), threshold_ticks=3)
    assert list(tr["gross_ticks"]) == [4.0, 4.0]               # +1 point long, -1 point short = +4 ticks each
