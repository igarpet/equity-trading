import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import s04, s05  # noqa: E402
from equity_trading.universe import SECTOR_ETF_01, UNIVERSE_01  # noqa: E402


def test_estimate_day_recovers_beta_and_ou():
    rng = np.random.default_rng(0)
    W, N = 520, 3
    e = rng.standard_normal((W, N)) * 1e-3
    u = np.zeros((W, N))
    for t in range(1, W):
        u[t] = 0.9 * u[t - 1] + rng.standard_normal(N) * 5e-4
    r = 1.2 * e + np.diff(np.vstack([np.zeros((1, N)), u]), axis=0)   # residual level u is AR(1)
    est = s05.estimate_day(r, e, s05.S05Config(center_m=False))
    np.testing.assert_allclose(est["beta"], 1.2, atol=0.1)
    assert est["valid"].all()
    assert np.all((est["tau"] > 5) & (est["tau"] < 20))              # 1/(-ln 0.9) = 9.5 bars


def test_estimate_day_rejects_random_walk_residual():
    rng = np.random.default_rng(1)
    W = 520
    e = rng.standard_normal((W, 1)) * 1e-3
    r = 0.8 * e + rng.standard_normal((W, 1)) * 1e-3                  # residual level is a random walk
    est = s05.estimate_day(r, e, s05.S05Config(max_tau_frac=0.05))
    assert not est["valid"][0]


def test_random_direction_test():
    g = np.array([1.0] * 50 + [-0.5] * 50)
    res = s05.random_direction_test(g, n=2000)
    assert res["p_upper"] < 0.05
    assert s05.random_direction_test(-g, n=2000)["p_upper"] > 0.5


@pytest.fixture(scope="module")
def run():
    cfg = s05.S05Config()
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    sc = np.array([panel.symbols.index(s) for s in stocks])
    sect = {s: k for k, v in UNIVERSE_01.items() for s in v}
    ec = np.array([panel.symbols.index(SECTOR_ETF_01[sect[s]]) for s in stocks])
    P = s05.estimate_all(panel, sc, ec, cfg, s05.first_decision_day(cfg))
    bt = s05.backtest(panel, P, sc, ec, cfg)
    return cfg, panel, sc, ec, P, bt


def test_no_lookahead_in_estimates(run):
    cfg, panel, sc, ec, P, bt = run
    assert np.isnan(P["beta"][: s05.first_decision_day(cfg)]).all()
    assert np.isnan(P["beta"][-1]).all()                               # last day decides nothing
    tr = bt["trades"]
    first_bar = panel.day_start[s05.first_decision_day(cfg) + 1]
    assert (tr["entry_bar"] >= first_bar).all()


def test_accounting_identity_and_caps(run):
    cfg, panel, sc, ec, P, bt = run
    tr = bt["trades"]
    assert len(tr) > 100
    assert bt["final_equity"] - cfg.initial_equity == pytest.approx(tr["gross_pnl"].sum() - bt["total_cost"], rel=1e-9)
    assert bt["n_open"].max() <= cfg.max_positions
    eq = bt["bar_equity"].to_numpy()
    frac = tr["notional"].to_numpy() / eq[tr["entry_bar"].to_numpy() - 1]
    assert np.all(frac <= cfg.size_per_stock * 1.02)


def test_sides_follow_s_score(run):
    tr = run[5]["trades"]
    assert (np.sign(tr["s_entry"]) == -tr["side"]).all()               # contrarian
    assert (tr["s_entry"].abs() >= 1.25).all()


def test_costs_and_placebo(run):
    cfg, panel, sc, ec, P, bt = run
    hi = s05.backtest(panel, P, sc, ec, replace(cfg, cost_mult=3.0), record=False)
    assert hi["final_equity"] < bt["final_equity"]
    a = s05.run_placebo(panel, P, sc, ec, cfg, bt["entry_rate"], 7, bt["first_day"])
    b = s05.run_placebo(panel, P, sc, ec, cfg, bt["entry_rate"], 7, bt["first_day"])
    assert a == b


def test_sector_map_covers_constituents():
    import pandas as pd
    c = pd.read_csv(Path(__file__).resolve().parents[1] / "reference" / "sp500_constituents_20260913.csv")
    assert set(c["GICS Sector"].unique()) <= set(s05.SECTOR_ETF)


# ------------------------------------------------------------------ S05b: measured costs
import pandas as pd  # noqa: E402

from equity_trading import alpaca, s05b  # noqa: E402


class _QResp:
    def __init__(self, js):
        self.status_code, self._js, self.text = 200, js, ""

    def json(self):
        return self._js


class _QSession:
    def __init__(self):
        self.calls = []

    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        if "page_token" not in params:
            return _QResp({"quotes": {"AAA": [{"t": "2024-01-02T15:00:00Z", "bp": 99.99, "ap": 100.01},
                                              {"t": "2024-01-02T15:00:00.001234567Z", "bp": 99.99, "ap": 100.01}]},
                           "next_page_token": "x"})
        return _QResp({"quotes": {"AAA": [{"t": "2024-01-02T15:00:01Z", "bp": 99.98, "ap": 100.02},
                                          {"t": "2024-01-02T15:00:02Z", "bp": 0, "ap": 100.0}],
                                  "BBB": [{"t": "2024-01-02T15:00:01Z", "bp": 50.0, "ap": 49.0}]},
                       "next_page_token": None})


def test_fetch_and_summarize_quotes(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setattr(alpaca, "_limiter", alpaca._RateLimiter(10 ** 6))
    sess = _QSession()
    q = alpaca.fetch_quotes(["AAA", "BBB"], "2024-01-02T15:00:00Z", "2024-01-02T15:00:03Z", session=sess)
    assert len(sess.calls) == 2 and sess.calls[0]["feed"] == "sip"
    s = alpaca.summarize_quotes(q).set_index("symbol")
    assert s.loc["AAA", "n_quotes"] == 3                                  # zero bid dropped; mixed stamp precision parsed
    assert s.loc["AAA", "half_spread"] == pytest.approx(1e-4, rel=1e-3)
    assert s.loc["BBB", "n_quotes"] == 0 and np.isnan(s.loc["BBB", "half_spread"])   # crossed quote dropped


def test_cost_model_and_array():
    rows = []
    for d in ["2024-03-01", "2025-03-03"]:
        for slot in range(4):
            prof = 2.0 if slot == 0 else 1.0
            rows += [("AAA", 1e-4 * prof, 10, slot, pd.Timestamp(d).date()),
                     ("BBB", 3e-4 * prof, 10, slot, pd.Timestamp(d).date())]
    m = s05b.cost_model(pd.DataFrame(rows, columns=["symbol", "half_spread", "n_quotes", "slot", "date"]))
    assert m["profile"][0] == pytest.approx(2.0) and m["profile"][1] == pytest.approx(1.0)
    idx = pd.DatetimeIndex(pd.date_range("2024-03-01 09:30", periods=4, freq="15min", tz="America/New_York"))
    H = s05b.cost_array(idx, np.array([0, 4]), ["AAA", "BBB", "CCC"], m)
    assert H.shape == (4, 3)
    assert H[1, 0] == pytest.approx(1e-4, rel=1e-5) and H[0, 0] == pytest.approx(2e-4, rel=1e-5)
    assert H[1, 2] == pytest.approx(m["year_median"][2024], rel=1e-5)     # unknown symbol: year median


def test_array_costs_match_flat_costs(run):
    cfg, panel, sc, ec, P, bt = run
    H = np.full(panel.c.shape, cfg.half_spread_bps * 1e-4, dtype=np.float32)
    bt2 = s05.backtest(panel, P, sc, ec, cfg, half_spread=H)
    assert bt2["final_equity"] == pytest.approx(bt["final_equity"], rel=1e-6)
    bt3 = s05.backtest(panel, P, sc, ec, cfg, half_spread=H * 0.5)
    assert bt3["final_equity"] > bt["final_equity"]


def test_sample_sessions_deterministic():
    a = s05b.sample_sessions([2024, 2025], 4, seed=1)
    assert a == s05b.sample_sessions([2024, 2025], 4, seed=1) and len(a) == 8
    assert len(s05b.slot_starts([2024], a[0])) in (14, 26)


# ------------------------------------------------------------------ S06: passive execution
from equity_trading import s06  # noqa: E402


@pytest.fixture(scope="module")
def run6():
    cfg = s06.S06Config()
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    sc = np.array([panel.symbols.index(s) for s in stocks])
    sect = {s: k for k, v in UNIVERSE_01.items() for s in v}
    ec = np.array([panel.symbols.index(SECTOR_ETF_01[sect[s]]) for s in stocks])
    P = s05.estimate_all(panel, sc, ec, cfg.s05(), s06.first_decision_day(cfg))
    H = np.full(panel.c.shape, 3e-4)
    return cfg, panel, sc, ec, P, H


@pytest.mark.parametrize("ht", ["fill_bar_mid", "fill_bar_open", "next_open"])
def test_s06_accounting_identity(run6, ht):
    cfg, panel, sc, ec, P, H = run6
    bt = s06.backtest(panel, P, sc, ec, H, replace(cfg, hedge_timing=ht))
    tr = bt["trades"]
    assert len(tr) > 50 and bt["n_open"].max() <= cfg.max_positions
    assert bt["final_equity"] - cfg.initial_equity == pytest.approx(
        tr["signal_pnl"].sum() + bt["total_capture"] - bt["total_cost"], abs=1e-4)
    assert bt["total_entry_fills"] == len(tr)
    assert (tr["notional"] > 0).all()


def test_s06_fill_rules_and_stress(run6):
    cfg, panel, sc, ec, P, H = run6
    th = s06.backtest(panel, P, sc, ec, H, cfg, record=False)
    to = s06.backtest(panel, P, sc, ec, H, replace(cfg, fill_rule="touch"), record=False)
    assert to["total_entry_fills"] / to["total_entry_orders"] > th["total_entry_fills"] / th["total_entry_orders"]
    st = s06.backtest(panel, P, sc, ec, H, replace(cfg, capture_mult=0.5, cost_mult=2.0), record=False)
    assert st["total_capture"] < th["total_capture"] and st["total_cost"] > th["total_cost"]


def test_s06_no_fill_without_traded_bars(run6):
    cfg, panel, sc, ec, P, H = run6
    import copy
    p2 = copy.copy(panel)
    p2.h = np.full_like(panel.h, np.nan)
    p2.l = np.full_like(panel.l, np.nan)
    bt = s06.backtest(p2, P, sc, ec, H, cfg, record=False)
    assert bt["total_entry_fills"] == 0 and bt["final_equity"] == pytest.approx(cfg.initial_equity)


def test_s06_placebo_seeded(run6):
    cfg, panel, sc, ec, P, H = run6
    a = s06.run_placebo(panel, P, sc, ec, H, cfg, 0.05, 3, pd.Timestamp("2025-02-01"))
    assert a == s06.run_placebo(panel, P, sc, ec, H, cfg, 0.05, 3, pd.Timestamp("2025-02-01"))


# ------------------------------------------------------------------ S07: continuous weights with a no-trade band
from equity_trading import s07  # noqa: E402


def test_s07_target_weights():
    cfg = s07.S07Config(w_cap=1.0)
    s = np.array([2.0, -1.0, 0.5, 5.0, np.nan])
    valid = np.array([True, True, True, True, True])
    w = s07.target_weights(s, valid, cfg)
    assert np.abs(w).sum() == pytest.approx(cfg.gross)
    assert np.all(np.sign(w[:4]) == -np.sign(s[:4])) and w[4] == 0
    assert w[3] == pytest.approx(w[0] * 3 / 2)                           # saturates at s_cap = 3
    capped = s07.target_weights(s, valid, s07.S07Config(w_cap=0.1))
    assert np.abs(capped).max() <= 0.1 + 1e-12
    assert np.all(s07.target_weights(s, ~valid, cfg) == 0)


@pytest.fixture(scope="module")
def run7():
    cfg = s07.S07Config(band_mult=0.5)
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    sc = np.array([panel.symbols.index(s) for s in stocks])
    sect = {s: k for k, v in UNIVERSE_01.items() for s in v}
    ec = np.array([panel.symbols.index(SECTOR_ETF_01[sect[s]]) for s in stocks])
    P = s05.estimate_all(panel, sc, ec, cfg.s05(), s07.first_decision_day(cfg))
    H = np.full(panel.c.shape, 3e-4)
    return cfg, panel, sc, ec, P, H


def test_s07_band_cuts_turnover_and_costs_scale(run7):
    cfg, panel, sc, ec, P, H = run7
    b0 = s07.backtest(panel, P, sc, ec, H, replace(cfg, band_mult=0.0), record=False)
    b5 = s07.backtest(panel, P, sc, ec, H, cfg, record=False)
    assert b5["total_traded"] < 0.5 * b0["total_traded"]
    free = s07.backtest(panel, P, sc, ec, H, replace(cfg, cost_mult=0.0), record=False)
    dbl = s07.backtest(panel, P, sc, ec, H, replace(cfg, cost_mult=2.0), record=False)
    assert free["total_cost"] == 0 and dbl["final_equity"] < b5["final_equity"] < free["final_equity"]


def test_s07_hedge_tracks_stock_book(run7):
    cfg, panel, sc, ec, P, H = run7
    b = s07.backtest(panel, P, sc, ec, H, cfg)
    assert b["gross_exposure"].max() > 0
    assert b["total_cost"] / b["total_traded"] == pytest.approx(3e-4, rel=1e-6)   # every traded dollar pays hs


def test_s07_placebo_seeded(run7):
    cfg, panel, sc, ec, P, H = run7
    a = s07.run_placebo(panel, P, sc, ec, H, cfg, 5, pd.Timestamp("2025-02-01"))
    assert a == s07.run_placebo(panel, P, sc, ec, H, cfg, 5, pd.Timestamp("2025-02-01"))


def test_s07_gates():
    boot = {10: (1e-4, 1e-5, 2e-4), 20: (1e-4, 1e-5, 2e-4), 40: (1e-4, 1e-5, 2e-4)}
    sh = {"total_return": {"p_upper": 0.02}, "sharpe": {"p_upper": 0.03}}
    g = s07.gates(boot, sh, {"total_return": 0.1, "sharpe": 0.4}, {"total_return": 0.2})
    assert not g["G2_vs_shuffled_signal"] and g["verdict"] == "NO_EDGE"            # 0.03 > 0.025


def test_s05_exit_levels_change_holding(run):
    cfg, panel, sc, ec, P, bt = run
    far = s05.backtest(panel, P, sc, ec, replace(cfg, close_long=1.0, close_short=-1.0))
    assert far["trades"]["bars"].mean() > bt["trades"]["bars"].mean()
    assert (np.sign(far["trades"]["s_entry"]) == -far["trades"]["side"]).all()


def test_s07_skip_open_bars(run7):
    cfg, panel, sc, ec, P, H = run7
    none = s07.backtest(panel, P, sc, ec, H, replace(cfg, skip_open_bars=30), record=False)
    assert none["total_traded"] == 0 and none["final_equity"] == pytest.approx(cfg.initial_equity)
    sk = s07.backtest(panel, P, sc, ec, H, replace(cfg, skip_open_bars=2), record=False)
    assert 0 < sk["total_traded"]
