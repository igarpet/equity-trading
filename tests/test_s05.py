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
