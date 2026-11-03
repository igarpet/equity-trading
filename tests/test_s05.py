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
