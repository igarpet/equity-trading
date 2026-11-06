import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p8, s04, s05  # noqa: E402
from equity_trading.universe import SECTOR_ETF_01, UNIVERSE_01  # noqa: E402


def test_bertram_cycle_time_matches_simulation():
    from scipy.special import erfi
    rng = np.random.default_rng(0)
    m, dt, n = 0.8, 0.002, 2000
    b = np.exp(-dt); sd = np.sqrt(1 - b * b)
    times = []
    for _ in range(n):
        s, t, state = -m, 0.0, "up"
        while True:
            s = b * s + sd * rng.standard_normal(); t += dt
            if state == "up" and s >= m:
                state = "down"
            elif state == "down" and s <= -m:
                break
        times.append(t)
    assert abs(np.mean(times) / (2 * np.pi * erfi(m / np.sqrt(2))) - 1) < 0.08


def test_bertram_mstar_rises_with_cost():
    ms = [p8.bertram_mstar(c) for c in (0.05, 0.2, 1.0, 3.0)]
    assert all(np.diff(ms) > 0) and ms[0] < 0.5 and ms[-1] > 1.0
    assert p8.bertram_rate(ms[1], 0.2) >= p8.bertram_rate(ms[1] * 1.2, 0.2)


def test_lp_table_and_choice():
    cfg = p8.P8Config(sim_paths=500, s0_grid=(1.25, 3.0), c_grid=(0.01, 2.0))
    tab = p8.lp_table(cfg)
    T = cfg.lp_horizon_halflives * np.log(2)
    assert np.all(tab["E_tau"] <= T + 1e-9) and np.all(tab["E_tau"] > 0)
    # deeper entries revert more on average with a take-profit at the mean and no stop
    j, l = list(cfg.tp_grid).index(0.0), len(cfg.sl_grid) - 1
    assert tab["E_ds"][1, j, l] > tab["E_ds"][0, j, l] > 0
    ch = p8.lp_choice(tab, cfg.c_grid)
    assert ch["tp"].shape == (2, 2) and np.all(np.isin(ch["tp"], cfg.tp_grid))


@pytest.fixture(scope="module")
def setup():
    cfg = s05.S05Config(close_long=1.0, close_short=-1.0)
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    sc = np.array([panel.symbols.index(s) for s in stocks])
    sect = {s: k for k, v in UNIVERSE_01.items() for s in v}
    ec = np.array([panel.symbols.index(SECTOR_ETF_01[sect[s]]) for s in stocks])
    P = s05.estimate_all(panel, sc, ec, cfg, s05.first_decision_day(cfg))
    H = np.full(panel.c.shape, 2e-4, np.float32)
    return cfg, panel, sc, ec, P, H


def test_lp_reduces_to_s05_x10(setup):
    cfg, panel, sc, ec, P, H = setup
    pc = p8.P8Config(lp_horizon_halflives=1e9)
    ch = {"c": np.array([1.0]), "s0": np.array([1.25]), "tp": np.array([[1.0]]), "sl": np.array([[np.inf]])}
    ct = np.ones((panel.n_days, len(sc)))
    a = p8.backtest(panel, P, sc, ec, cfg, "LP", H, ct, pc, lp_ch=ch)
    b = s05.backtest(panel, P, sc, ec, cfg, half_spread=H)
    assert len(a["trades"]) == len(b["trades"])
    np.testing.assert_allclose(a["daily_equity"].to_numpy(), b["daily_equity"].to_numpy(), rtol=1e-9)


def test_bt_and_lp_run_with_stops(setup):
    cfg, panel, sc, ec, P, H = setup
    pc = p8.P8Config(sim_paths=300)
    ct = p8.cost_sigma_units(P, H, panel, sc, ec)
    bt = p8.backtest(panel, P, sc, ec, cfg, "BT", H, ct, pc, bt_tab=p8.bertram_table(pc.c_grid))
    ch = p8.lp_choice(p8.lp_table(pc), pc.c_grid)
    lp = p8.backtest(panel, P, sc, ec, cfg, "LP", H, ct, pc, lp_ch=ch)
    assert len(bt["trades"]) > 0 and len(lp["trades"]) > 0
    assert set(lp["trades"]["reason"]) <= {"take_profit", "stop_loss", "time_limit", "invalid", "end"}
