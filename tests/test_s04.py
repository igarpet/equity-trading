import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import s04  # noqa: E402


def test_abdi_ranaldo_recovers_known_spread():
    rng = np.random.default_rng(0)
    days, bars, sub = 60, 26, 15
    hs = 5e-4
    mid = np.cumsum(rng.standard_normal(days * bars * sub) * 2e-4).reshape(days * bars, sub)
    trade = mid + hs * rng.choice([-1, 1], size=mid.shape)
    c, h, l = trade[:, -1], trade.max(axis=1), trade.min(axis=1)
    session = np.repeat(np.arange(days), bars)
    s2 = s04.abdi_ranaldo_s2(c, h, l, session, days)
    est = np.sqrt(np.clip(np.nanmean(s2), 0, None)) / 2
    assert 0.5 * hs < est < 1.5 * hs


def test_bh_adjust_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    adj = s04.bh_adjust(p)
    assert np.isnan(adj[-1])
    np.testing.assert_allclose(adj[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_ou_tp_probability_bounds_and_monotone():
    z = np.linspace(0.5, 3.0, 11)
    p = s04.ou_tp_probability(z)
    assert p[0] == pytest.approx(1.0) and p[-1] == pytest.approx(0.0)
    assert np.all(np.diff(p) < 0)
    assert 0.5 < float(s04.ou_tp_probability(1.5)) < 1.0


def _tests_frame(rows):
    df = pd.DataFrame(rows, columns=["day", "passed", "failed"])
    df["y"], df["x"] = 0, 1
    return df


def test_hysteresis_add_after_three_passes_remove_after_three_fails():
    cfg = s04.S04Config()
    seq = [(0, 1, 0), (1, 1, 0), (2, 1, 0),           # third pass at day 2 -> tradable from day 3
           (3, 0, 1), (4, 0, 0), (5, 0, 1), (6, 0, 1), (7, 0, 1),   # neutral resets; fails 5,6,7 -> end 8
           (8, 1, 0), (9, 1, 0)]
    sp = s04.hysteresis(_tests_frame(seq), cfg, n_days=20)
    assert sp.to_dict("records") == [{"y": 0, "x": 1, "start": 3, "end": 8}]


def test_hysteresis_missing_days_count_as_fails():
    cfg = s04.S04Config()
    seq = [(0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0), (9, 1, 0)]  # days 4.. missing -> removed after day 6
    sp = s04.hysteresis(_tests_frame(seq), cfg, n_days=20)
    assert sp.iloc[0].to_dict() == {"y": 0, "x": 1, "start": 3, "end": 7}


def test_hysteresis_open_spell_runs_to_end():
    cfg = s04.S04Config()
    seq = [(d, 1, 0) for d in range(10, 19)]
    sp = s04.hysteresis(_tests_frame(seq), cfg, n_days=20)
    assert sp.iloc[0].to_dict() == {"y": 0, "x": 1, "start": 13, "end": 20}


@pytest.fixture(scope="module")
def synthetic_run():
    cfg = s04.S04Config(window_rule="fixed", max_window_sessions=40, fixed_window_sessions=40)
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    idx = np.array([panel.symbols.index(s) for s in stocks])
    pools = s04.candidate_schedule(panel, idx, cfg, s04.first_decision_day(cfg))
    tests = s04.run_tests(panel, pools, cfg, n_jobs=1, verbose=False)
    tests, spells = s04.tradable_spells(panel, tests, cfg, n_jobs=1, verbose=False)
    hs = panel.half_spread(cfg)
    params = s04._Params(tests, len(panel.symbols))
    bt = s04.backtest(panel, params, spells, hs, cfg)
    return cfg, panel, tests, spells, hs, params, bt


def test_pools_are_alphabetical_and_no_lookahead(synthetic_run):
    cfg, panel, tests, spells, *_ = synthetic_run
    sym = np.array(panel.symbols)
    assert (sym[tests["y"]] < sym[tests["x"]]).all()
    assert len(spells) > 0
    first_pass = tests[tests.passed].groupby(["y", "x"])["day"].min()
    for r in spells.itertuples(index=False):
        assert r.start > first_pass.loc[(r.y, r.x)]       # trading starts the day after a decision


def test_backtest_accounting_identity_and_limits(synthetic_run):
    cfg, panel, tests, spells, hs, params, bt = synthetic_run
    tr = bt["trades"]
    assert len(tr) > 0
    assert bt["final_equity"] - cfg.initial_equity == pytest.approx(tr["pnl"].sum(), rel=1e-9, abs=1e-6)
    assert bt["n_open"].max() <= cfg.max_pairs
    assert (tr["cost"] > 0).all()
    assert set(tr["reason"]) <= {"tp", "stop", "time", "break", "end"}
    # every trade's notional is about 4% of equity at entry (integer shares round it down)
    eq = bt["bar_equity"].to_numpy()
    frac = tr["notional"].to_numpy() / eq[tr["entry_bar"].to_numpy() - 1]
    assert np.all(frac <= cfg.gross_per_pair * 1.02) and np.median(frac) > cfg.gross_per_pair * 0.8


def test_backtest_costs_reduce_equity(synthetic_run):
    cfg, panel, tests, spells, hs, params, bt = synthetic_run
    hi = s04.backtest(panel, params, spells, panel.half_spread(replace(cfg, cost_mult=3.0)), cfg, record=False)
    assert hi["final_equity"] < bt["final_equity"]


def test_placebos_run_and_are_seeded(synthetic_run):
    cfg, panel, tests, spells, hs, params, bt = synthetic_run
    a = s04.backtest(panel, params, spells, hs, cfg, mode="placebo_pairs", seed=7, record=False)
    b = s04.backtest(panel, params, spells, hs, cfg, mode="placebo_pairs", seed=7, record=False)
    c = s04.backtest(panel, params, spells, hs, cfg, mode="placebo_timing", entry_prob=bt["entry_rate"], seed=7,
                     record=False)
    assert a["final_equity"] == b["final_equity"]
    assert np.isfinite(c["final_equity"])


def test_gates_logic():
    boot = {10: (1e-4, 1e-5, 2e-4), 20: (1e-4, 2e-6, 2e-4), 40: (1e-4, -1e-6, 2e-4)}
    good = {"total_return": {"p_upper": 0.01}, "sharpe": {"p_upper": 0.02}}
    g = s04.gates(boot, good, good, {"total_return": 0.1, "sharpe": 0.5}, {"total_return": 0.2})
    assert not g["G1_bootstrap"] and g["G2_vs_random_pairs"] and g["verdict"] == "NO_EDGE"


def test_bar_grid_matches_exchange_calendar():
    g = s04.bar_grid([2024], 15)
    days = pd.Series(g.date)
    assert days.nunique() == 252                                # NYSE sessions in 2024
    assert g[0].year == 2024 and (g[0].hour, g[0].minute) == (9, 30)
    per_day = days.value_counts()
    assert per_day.max() == 26 and per_day.min() == 14          # early closes end at 13:00
    assert g.is_monotonic_increasing and g.is_unique


def test_load_panel_places_bars_on_grid(monkeypatch):
    g = s04.bar_grid([2024], 15)
    bars = pd.DataFrame({"t": g.tz_convert("UTC")[:520], "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5,
                         "v": 1, "n": 1, "vw": 100.0})
    monkeypatch.setattr("equity_trading.alpaca.load_bars", lambda s, tf, years: bars)
    with pytest.raises(RuntimeError):                            # 520 of 6,516 bars filled: rejected
        s04.load_panel(["AAA"], "15Min", [2024])
    full = bars.iloc[:0].reindex(range(len(g)))
    full["t"] = g.tz_convert("UTC"); full[["o", "h", "l", "c"]] = 100.0
    monkeypatch.setattr("equity_trading.alpaca.load_bars", lambda s, tf, years: full)
    p = s04.load_panel(["AAA"], "15Min", [2024])
    assert p.n_days == 252 and np.isfinite(p.c).all()
