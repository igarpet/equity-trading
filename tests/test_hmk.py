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


def test_rolling_states_rule():
    r = np.r_[np.full(20, 30.0), np.full(20, -30.0)]          # +30 bp x 20 = +6% -> bull; then down
    st, ok = H.rolling_states(r)
    assert not ok[:19].any() and ok[19:].all()
    assert st[19] == 2 and st[-1] == 0
    assert st[30] == 1                                         # mixed window: +30 x 9 - 30 x 11 = -0.6%


def test_random_walk_persistence_is_mechanical():
    """On iid returns the 20-bar rule looks sticky; the shuffle null matches it and non-overlapping blocks lose it."""
    r = np.random.default_rng(1).normal(3, 120, 4000)
    rep = H.chain_report(r, n_null=30)
    assert rep["diag"]["bull"] > 0.7
    assert abs(rep["diag"]["bull"] - rep["null_shuffle_diag_mean"]["bull"]) < 0.05
    assert rep["P_nonoverlap"].loc["bull", "bull"] < 0.4
    assert np.allclose(rep["P"].sum(1), 1)


def test_band_scaled():
    assert H.band_scaled("1d") == pytest.approx(0.05)
    assert H.band_scaled("1h") == pytest.approx(0.05 * np.sqrt(60 / 1380))


def test_band_for_window_matches_daily_rule():
    assert H.band_for_window("1d", 20) == pytest.approx(0.05)
    assert H.band_for_window("1h", 40) == pytest.approx(0.05 * np.sqrt(40 * 60 / (20 * 1380)))


def test_hmm_report_runs_and_is_causal():
    rng = np.random.default_rng(2)
    r = np.r_[rng.normal(4, 30, 1500), rng.normal(-4, 30, 1500), rng.normal(0, 30, 1500)]
    mask = np.ones(len(r), bool)
    rep = H.hmm_report(r, 40, mask, H.HMKConfig(n_starts=2), n_null=2)
    assert np.allclose(rep["P"].sum(1), 1)
    assert rep["state_mean_pct"].is_monotonic_increasing
    st1, _, m = H.hmm_states(r, 40, mask, H.HMKConfig(n_starts=1))
    r2 = r.copy(); r2[3000:] = 0
    x1, ok1 = H.rolling_sum_bp(r, 40); x2, ok2 = H.rolling_sum_bp(r2, 40)
    W = np.ones((len(r), 1))
    a1, _ = H.filter_run(m, np.where(ok1, x1, 0), ok1, W); a2, _ = H.filter_run(m, np.where(ok2, x2, 0), ok2, W)
    assert np.allclose(a1[:3000], a2[:3000])


def test_tier_dependence_on_random_walk(tmp_path):
    """iid returns: the tiers look dependent against a circular shift (shared returns), but no more than the
    shuffled-within-slot reference."""
    H.write_synthetic(tmp_path, D(2019, 11, 1), D(2020, 9, 30), effect=0.0)
    bars = H.session_bars(MB.load_1m("MNQ", D(2019, 11, 1), D(2020, 10, 1)))
    T = H.tiers_from_15m(bars)
    r15 = T["q"].ret_bp.to_numpy(float)
    R = H.tier_returns_from(r15, T)
    assert np.allclose(np.nan_to_num(R["1h"]), np.nan_to_num(bars["1h"].ret_bp.to_numpy(float)))
    a = D(2020, 1, 1)
    masks = {"15m": (pd.Series(T["q"].trade_date).dt.date >= a).to_numpy(),
             "1h": np.array([d >= a for d in T["hour_dates"]]), "1d": np.array([d >= a for d in T["day_dates"]])}
    specs = {"1d": H.TierSpec("1d", 20, "band", 0.05), "1h": H.TierSpec("1h", 40, "band", H.band_for_window("1h", 40)),
             "15m": H.TierSpec("15m", 80, "band", H.band_for_window("15m", 80))}
    real = H.dependence(r15, T, specs, masks)
    rng = np.random.default_rng(0)
    sh = [H.dependence(H.shuffle_within_slot(r15, T["q"].slot.to_numpy(), rng), T, specs, masks)["15m|1h"]["mi"]
          for _ in range(20)]
    cs = H.circular_shift_null(r15, T, specs, masks, 20, rng)["15m|1h"]["mi"]
    assert real["15m|1h"]["mi"] > np.max(cs)                       # 'dependent' by shared returns
    assert np.mean(np.array(sh) >= real["15m|1h"]["mi"]) > 0.05     # but not beyond the shuffled reference
    P = H.cond_tables(real["15m|1d"]["counts"])
    assert np.allclose(P.sum(1), 1)
