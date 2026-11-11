import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import fx_seasonality as F  # noqa: E402

D = dt.date


@pytest.fixture
def market_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    monkeypatch.delenv("DUKASCOPY_DIR", raising=False)
    return tmp_path


def _downloader():
    spec = importlib.util.spec_from_file_location("dc", REPO / "tools" / "dukascopy" / "dukascopy_candles.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_candle_format_check():
    dc = _downloader()
    n = 200
    t = np.arange(n, dtype=np.uint32) * 60
    rng = np.random.default_rng(0)
    path = 110_000 + np.cumsum(rng.integers(-3, 4, n * 4)).reshape(n, 4)
    o, c = path[:, 0], path[:, 3]
    h, l = path.max(1) + 1, path.min(1) - 1
    good = np.zeros(n, dc.REC)
    good["t"], good["o"], good["c"], good["l"], good["h"], good["v"] = t, o, c, l, h, 1.0
    assert dc.check_format(good) is None
    swapped = good.copy(); swapped["o"], swapped["c"] = c, o
    assert dc.check_format(swapped) is None                     # advisory only, never stops the run
    assert not dc.swap_suspect(swapped)                         # 199 pairs: too few to judge
    big = np.zeros(2000, dc.REC)
    pth = 110_000 + np.cumsum(rng.integers(-3, 4, 2000 * 4)).reshape(2000, 4)
    big["t"] = np.arange(2000) % 1440 * 60
    big["o"], big["c"], big["l"], big["h"], big["v"] = pth[:, 0], pth[:, 3], pth.min(1) - 1, pth.max(1) + 1, 1.0
    assert not dc.swap_suspect(big)
    big_sw = big.copy(); big_sw["o"], big_sw["c"] = big["c"], big["o"]
    assert dc.swap_suspect(big_sw)
    wrong = good.copy(); wrong["c"], wrong["h"] = h, c          # (t, o, h, l, c) read as (t, o, c, l, h)
    assert dc.check_format(wrong) is not None
    bad_t = good.copy(); bad_t["t"] = t + 7
    assert "time" in dc.check_format(bad_t)


def test_guards():
    with pytest.raises(F.WindowError):
        F._guard(D(2016, 1, 1), D(2021, 1, 4), "discovery")
    with pytest.raises(F.HoldoutError):
        F._guard(D(2021, 1, 1), D(2026, 1, 2), "confirmation")
    F._guard(D(2021, 1, 1), D(2025, 12, 31), "confirmation")


def test_bh_and_nw():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej, q = F.bh_reject(p, 0.05)
    assert rej.tolist() == [True, True] + [False] * 8
    x = np.random.default_rng(1).normal(0.1, 1, (4000, 1))
    m, se, t = F.nw_tstats(x, 0)
    assert se[0] == pytest.approx(x.std() / np.sqrt(4000), rel=1e-3)


def test_overlap_across_clocks():
    # NY 08:00-09:00 is slot 900-960 -> UTC 12:00-13:00 (summer) or 13:00-14:00 (winter)
    ny = F._utc_segments("NY", 900, 960)
    assert F._overlap(ny, F._utc_segments("UTC", 12 * 60 + 30, 13 * 60 + 30))
    assert not F._overlap(ny, F._utc_segments("UTC", 10 * 60, 11 * 60))
    assert F.slot_label("NY", 900) == "NY 08:00" and F.slot_label("UTC", 90) == "UTC 01:30"


def test_panel_boundaries_and_trades(market_dir):
    base = market_dir / "Dukascopy"
    F.write_synthetic_candles(base, "EURUSD", D(2016, 3, 1), D(2016, 3, 31), seed=3)
    m = F.load_minutes("EURUSD", D(2016, 3, 2), D(2016, 3, 30), "discovery")
    p = F.build_panel(m, "EURUSD", "NY", D(2016, 3, 2), D(2016, 3, 30), 0.8)
    assert p.bid.shape[1] == F.SLOTS + 1 and len(p.dates) >= 18
    assert all(pd.Timestamp(x).weekday() < 5 for x in p.dates)
    # boundary 0 of a NY day = close of the 16:59 NY minute; boundary k+1 = close of minute k
    d0 = pd.Timestamp(p.dates[3])
    t_start = (d0 - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize(F.NY).tz_convert("UTC")
    assert p.bid[3, 0] == pytest.approx(m.loc[t_start - pd.Timedelta(minutes=1), "bid"])
    assert p.bid[3, 61] == pytest.approx(m.loc[t_start + pd.Timedelta(minutes=60), "bid"])
    tr = F.candidate_trades(p, 0, 60, +1, F.FXSConfig())
    assert tr["gross_bps"].iloc[3] == pytest.approx(np.log(p.bid[3, 60] / p.ask[3, 0]) * 1e4)
    assert (tr["net_bps"] == tr["gross_bps"] - 0.7).all()


def test_synthetic_drift_found_and_placebo_clean(market_dir):
    base = market_dir / "Dukascopy"
    F.write_synthetic_candles(base, "EURUSD", D(2016, 1, 1), D(2020, 12, 31), seed=5, drift_bps=4.0)
    F.write_synthetic_candles(base, "USDJPY", D(2016, 1, 1), D(2020, 12, 31), seed=6, price=110.0)
    cfg = F.FXSConfig(n_placebo=2, max_candidates=5)
    panels = F.load_panels(["EURUSD", "USDJPY"], *F.DISCOVERY, "discovery", cfg)
    scan = pd.concat([F.scan_panel(p, cfg) for p in panels.values()], ignore_index=True)
    s, cand = F.select_candidates(scan, cfg)
    assert len(cand) >= 1 and cand.iloc[0]["pair"] == "EURUSD" and cand.iloc[0]["direction"] == 1
    assert cand.iloc[0]["id"] == "EURUSD-01"
    top = cand.iloc[0]
    assert top["start_time"] >= "NY 07:30" and top["end_time"] <= "NY 09:30" or top["clock"] == "UTC"
    assert (cand["pair"] == "USDJPY").sum() == 0
    real = int((s["D1"] & s["D2"] & s["D3"]).sum())
    pl = F.placebo_counts(panels, cfg)
    # rotated days keep each day's total drift (here an extreme +4 bps/day), so a few long placebo
    # intervals can pass; the time-of-day structure must still dominate
    assert pl["D1_D2_D3"].max() * 5 < real


def test_cache_not_reused_while_downloading(market_dir):
    base = market_dir / "Dukascopy"
    F.write_synthetic_candles(base, "EURUSD", D(2016, 1, 1), D(2016, 1, 20), seed=1)
    a = F.load_minutes("EURUSD", D(2016, 1, 4), D(2016, 2, 26), "discovery")
    F.write_synthetic_candles(base, "EURUSD", D(2016, 1, 21), D(2016, 2, 26), seed=2)
    b = F.load_minutes("EURUSD", D(2016, 1, 4), D(2016, 2, 26), "discovery")
    assert b.index[-1] > a.index[-1]
    cov = F.download_coverage("EURUSD", D(2016, 1, 1), D(2016, 12, 31))
    assert cov.loc[2016, "present"] == 41 and cov.loc[2016, "expected"] == 261


def test_bonferroni_and_block_consistency(market_dir):
    assert F.bonferroni_t(F.pooled_n_tests(F.FXSConfig())) == pytest.approx(4.84, abs=0.02)
    base = market_dir / "Dukascopy"
    F.write_synthetic_candles(base, "EURUSD", D(2016, 1, 1), D(2017, 12, 31), seed=4, drift_bps=4.0)
    m = F.load_minutes("EURUSD", D(2016, 1, 4), D(2017, 12, 29), "discovery")
    p = F.build_panel(m, "EURUSD", "NY", D(2016, 1, 4), D(2017, 12, 29), 0.8)
    bc = F.block_consistency(p)
    top = bc.loc[bc["t"].idxmax()]
    assert top["time"] >= "NY 08:00" and top["time"] < "NY 09:00" and top["years_same_sign"] == 2
    yp = F.yearly_profiles(p)
    assert list(yp.columns) == [2016, 2017]


def test_incomplete_download_refused(market_dir):
    base = market_dir / "Dukascopy"
    F.write_synthetic_candles(base, "EURUSD", D(2015, 12, 28), D(2016, 6, 30), seed=1)
    cfg = F.FXSConfig()
    with pytest.raises(F.IncompleteDataError):
        F.load_panels(["EURUSD"], *F.DISCOVERY, "discovery", cfg)
    cov = F.file_coverage("EURUSD", *F.DISCOVERY)
    assert cov["present"] < cov["expected"] and D(2018, 1, 2) in cov["missing"]
    assert F.file_coverage("EURUSD", D(2016, 1, 4), D(2016, 6, 30))["share"] == 1.0
    F.load_panels(["EURUSD"], D(2016, 1, 4), D(2016, 6, 30), "discovery", cfg)


def test_frozen_hash_tolerates_line_endings(tmp_path):
    f = tmp_path / "c.csv"
    f.write_bytes(b"id,pair\r\nEURUSD-01,EURUSD\r\n")
    h = F.file_sha256(f)
    f.write_bytes(b"id,pair\nEURUSD-01,EURUSD\n")
    assert F.frozen_hash_ok(f, h)
    f.write_bytes(b"id,pair\nEURUSD-01,GBPUSD\n")
    assert not F.frozen_hash_ok(f, h)


def test_account_growth_compounds_and_kelly_drops_losers():
    idx = pd.bdate_range("2021-01-01", periods=500)
    rng = np.random.default_rng(0)
    R = pd.DataFrame({"good": rng.normal(2.0, 10, 500), "bad": rng.normal(-2.0, 10, 500)}, index=idx)
    w = F.kelly_weights(R, 0.5)
    assert w["bad"] == 0 and w["good"] > 0
    capped = F.kelly_weights(R, 0.5, cap_total=10)
    assert capped.sum() <= 10 + 1e-9
    flat = pd.DataFrame({"a": np.full(10, 100.0)}, index=idx[:10])            # +1% a day at 1x
    acc = F.simulate_account(flat, pd.Series({"a": 1.0}), 100_000, min_units=1)
    assert acc["equity"].iloc[-1] == pytest.approx(100_000 * 1.01 ** 10)       # size grows with equity
    small = F.simulate_account(flat, pd.Series({"a": 1.0}), 500)               # below one 1,000-unit lot
    assert (small["equity"] == 500).all()
    st = F.account_stats(acc, 100_000)
    assert st["max_drawdown"] == 0 and not st["ruined"]
    cand = pd.DataFrame({"clock": ["UTC", "UTC", "NY"], "start": [0, 30, 0], "end": [60, 90, 60]})
    assert F.max_concurrent(cand) == 2


def test_prop_costs_and_rules():
    assert F.commission_bps_per_lot("USDJPY", 3.0) == pytest.approx(0.30)
    assert F.commission_bps_per_lot("EURUSD", 3.0) == pytest.approx(0.30 / 1.10)
    r = F.PropRules(daily_loss=0.05, max_loss=0.10, target=0.08, min_days=1)
    G = np.full((100, 1), 10.0); M = np.zeros((100, 1)); c = np.zeros(1)
    o, d, e = F.run_prop_path(G, M, c, np.array([10.0]), 10_000, r, 100)       # +1% a day at 10x
    assert o == "pass" and d == 8
    M2 = M.copy(); M2[3, 0] = -60.0                                              # -6% open loss on day 4
    o, d, _ = F.run_prop_path(G, M2, c, np.array([10.0]), 10_000, r, 100)
    assert o == "daily_breach" and d == 4
    G3 = np.full((100, 1), -4.0)
    o, d, _ = F.run_prop_path(G3, np.full((100, 1), -4.0), c, np.array([10.0]), 10_000, r, 100)
    assert o == "max_breach"


def test_pips_and_pip_value():
    assert F.bps_to_pips(1.0, 1.10, "EURUSD") == pytest.approx(1.10)
    assert F.bps_to_pips(1.0, 150.0, "USDJPY") == pytest.approx(1.50)
    assert F.usd_per_pip_per_lot("EURUSD", 1.10) == pytest.approx(10.0)
    assert F.usd_per_pip_per_lot("USDJPY", 150.0) == pytest.approx(1000 / 150)
    assert F.usd_per_pip_per_lot("EURJPY", 160.0) == pytest.approx(1000 / 160 * 1.10)


def test_holdout_stage_only_in_2026():
    F._guard(D(2026, 1, 1), D(2026, 11, 6), "holdout")
    with pytest.raises(F.WindowError):
        F._guard(D(2025, 12, 1), D(2026, 11, 6), "holdout")
    with pytest.raises(F.HoldoutError):
        F._guard(D(2025, 1, 1), D(2026, 1, 5), "confirmation")


def test_spread_factor_scales_cost(market_dir):
    base = market_dir / "Dukascopy"
    F.write_synthetic_candles(base, "EURUSD", D(2016, 3, 1), D(2016, 3, 31), seed=3, spread_bps=1.0)
    cfg = F.FXSConfig()
    cand = pd.DataFrame([{"id": "X", "pair": "EURUSD", "clock": "NY", "start": 0, "end": 60, "direction": 1}])
    g1, _ = F.structure_detail(cand, D(2016, 3, 2), D(2016, 3, 30), "discovery", cfg)
    g9, _ = F.structure_detail(cand, D(2016, 3, 2), D(2016, 3, 30), "discovery", cfg, spread_factor=0.9)
    assert (g9["X"] - g1["X"]).mean() == pytest.approx(0.1, abs=0.01)     # 10% of a 1 bp spread saved


def _arrays(mid, half=0.00005):
    mid = np.asarray(mid, float)
    A = {}
    for side, s in (("bid", -half), ("ask", half)):
        for k in ("o", "h", "l", "c"):
            A[f"{side}_{k}"] = mid + s
    return A


def test_trailing_stop_rules():
    pip = 0.0001
    # long from 1.10000 (ask 1.10005): rises 10 pips, then falls back -> trailing stop at +5 - spread effects
    path = 1.1 + pip * np.r_[0, 0, np.arange(1, 11), np.arange(9, -15, -1)]
    A = _arrays(path)
    x, g, why, worst, _ = F.simulate_trade(A, 1, None, +1, pip, F.TrailRule())
    # best bid excursion = 1.1010 - 0.00005 - 1.10005 = +9 pips -> stop at floor(9) - 5 = +4
    assert why == "trail" and g == pytest.approx(4.0)
    # never reaches +8: initial stop at -12
    A2 = _arrays(1.1 + pip * np.r_[0, 0, np.arange(1, 6), np.arange(4, -20, -1)])
    x, g, why, _, _ = F.simulate_trade(A2, 1, None, +1, pip, F.TrailRule())
    assert why == "stop" and g == pytest.approx(-12.0)
    # with a time exit before any stop: closes at the last minute's close
    x, g, why, _, _ = F.simulate_trade(A2, 1, 5, +1, pip, F.TrailRule())
    assert why == "time" and g == pytest.approx((A2["bid_c"][4] - A2["ask_c"][0]) / pip)
    # gap through the stop: filled at the open
    A3 = _arrays(1.1 + pip * np.r_[0, 0, -3, -30, -31])
    x, g, why, _, _ = F.simulate_trade(A3, 1, None, +1, pip, F.TrailRule())
    assert why == "stop" and g == pytest.approx((A3["bid_o"][3] - A3["ask_c"][0]) / pip)
    # short mirror: falls 10 pips then rises
    A4 = _arrays(1.1 - pip * np.r_[0, 0, np.arange(1, 11), np.arange(9, -15, -1)])
    x, g, why, _, _ = F.simulate_trade(A4, 1, None, -1, pip, F.TrailRule())
    assert why == "trail" and g == pytest.approx(4.0)
    # no-stop rule with a time exit equals the plain close-to-close trade
    x, g, why, _, _ = F.simulate_trade(A, 1, 20, +1, pip, F.NO_STOP)
    assert why == "time" and g == pytest.approx((A["bid_c"][19] - A["ask_c"][0]) / pip)


def test_take_profit_only():
    pip = 0.0001
    tp = F.take_profit_only(15.0)
    # long rises 20 pips: closed at +15 exactly
    A = _arrays(1.1 + pip * np.r_[0, 0, np.arange(1, 21)])
    x, g, why, worst, best = F.simulate_trade(A, 1, None, +1, pip, tp)
    assert why == "tp" and g == pytest.approx(15.0) and best >= 15.0 - 1e-6
    # falls 40 pips then recovers to -5 at the time exit: no stop loss, closed at the time exit
    A2 = _arrays(1.1 + pip * np.r_[0, 0, -np.arange(1, 41), -np.arange(39, 4, -1)])
    x, g, why, worst, best = F.simulate_trade(A2, 1, len(A2["bid_c"]), +1, pip, tp)
    assert why == "time" and worst < -39 and g == pytest.approx((A2["bid_c"][-1] - A2["ask_c"][0]) / pip)
    # gap above the take-profit: filled at the (better) open
    A3 = _arrays(1.1 + pip * np.r_[0, 0, 3, 30, 31])
    x, g, why, _, _ = F.simulate_trade(A3, 1, None, +1, pip, tp)
    assert why == "tp" and g == pytest.approx((A3["bid_o"][3] - A3["ask_c"][0]) / pip)


def test_bh_ignores_nan():
    p = np.array([0.001, np.nan, 0.04, 0.5])
    rej, q = F.bh_reject(p, 0.05)
    assert np.isnan(q[1]) and not rej[1]
    ref_rej, ref_q = F.bh_reject(np.array([0.001, 0.04, 0.5]), 0.05)
    assert np.allclose(q[[0, 2, 3]], ref_q) and (rej[[0, 2, 3]] == ref_rej).all()
