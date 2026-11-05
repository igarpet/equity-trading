import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p5, s04, s05  # noqa: E402
from equity_trading.universe import SECTOR_ETF_01, UNIVERSE_01  # noqa: E402


def test_gp_rate_table():
    for q, c, tau in ((12, 3, 0.25), (90, 9, 0.1), (600, 24, 0.04)):
        cc, tt = p5.gp_rate(q)
        assert abs(cc - c) < 1e-9 and abs(tt - tau) < 1e-9


def test_aim_persistence_weighting_shifts_to_slow_signal():
    rng = np.random.default_rng(0)
    sA, sB = rng.standard_normal(100), rng.standard_normal(100)
    sz, ok = np.full(100, 1e-3), np.ones(100, bool)
    lo = p5.aim_weights(sA, sB, sz, ok, 1, 1, 0.1, 0.005, 3, 1.0, 1.0)
    hi = p5.aim_weights(sA, sB, sz, ok, 1, 1, 0.1, 0.005, 24, 1.0, 1.0)
    assert abs(np.abs(hi).sum() - 1) < 1e-9
    corrB = lambda w: np.corrcoef(w, -sB)[0, 1]  # noqa: E731
    assert corrB(hi) > corrB(lo)
    assert np.all(p5.aim_weights(sA, sB, sz, np.zeros(100, bool), 1, 1, 0.1, 0.005, 3, 1.0, 1.0) == 0)


@pytest.fixture(scope="module")
def setup():
    cfg = s05.S05Config()
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    sc = np.array([panel.symbols.index(s) for s in stocks])
    sect = {s: k for k, v in UNIVERSE_01.items() for s in v}
    ec = np.array([panel.symbols.index(SECTOR_ETF_01[sect[s]]) for s in stocks])
    P = s05.estimate_all(panel, sc, ec, cfg, s05.first_decision_day(cfg))
    sig = p5.signals(panel, P, sc, ec, 130)
    return panel, P, sc, ec, sig


def test_signal_A_matches_s05(setup):
    panel, P, sc, ec, sig = setup
    t = 40
    a0, a1 = panel.day_start[t], panel.day_start[t + 1]
    r = s05.log_returns(panel)
    e = r[a0:a1][:, sc] - P["alpha"][t - 1] - P["beta"][t - 1] * r[a0:a1][:, ec]
    s = (np.cumsum(np.nan_to_num(e), 0) - P["m"][t - 1]) / P["sig"][t - 1]
    v = P["valid"][t - 1]
    np.testing.assert_allclose(sig["sA"][a0:a1][:, v], s[:, v], rtol=1e-4)


def test_estimates_reversal_sign_and_decay(setup):
    panel, P, sc, ec, sig = setup
    est = p5.estimates(p5.day_stats(sig, panel.n_days), 20)
    e = est.dropna()
    assert len(e) > 100
    assert e["B_A"].median() > 0                     # residual reverts in the synthetic panel
    assert (e["phi_A"] > e["phi_B"]).mean() > 0.9    # fast signal decays faster


def test_backtest_runs_and_rate_reduces_turnover(setup):
    panel, P, sc, ec, sig = setup
    est = p5.estimates(p5.day_stats(sig, panel.n_days), 20)
    H = np.full(panel.c.shape, 2e-4, np.float32)
    cfg = p5.P5Config()
    fast = p5.backtest(panel, P, sig, est, sc, ec, H, cfg, 12.0)
    slow = p5.backtest(panel, P, sig, est, sc, ec, H, cfg, 600.0)
    assert slow["annual_turnover_x_equity"] < fast["annual_turnover_x_equity"]
    pl = p5.backtest(panel, P, sig, est, sc, ec, H, cfg, 90.0, shuffle=True, seed=1)
    assert np.isfinite(pl["daily_equity"]).all()
    blk = np.ones((panel.n_days, len(sc)), bool)
    flat = p5.backtest(panel, P, sig, est, sc, ec, H, cfg, 90.0, block=blk)
    assert flat["total_traded"] == 0
