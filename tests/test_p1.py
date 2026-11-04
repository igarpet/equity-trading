import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p1, s04, s05  # noqa: E402
from equity_trading.universe import SECTOR_ETF_01, UNIVERSE_01  # noqa: E402

CFG = p1.P1Config()


def test_abnormal_volume_has_no_lookahead():
    rng = np.random.default_rng(0)
    v = pd.DataFrame(rng.uniform(1, 2, (400, 3)))
    a = p1.abnormal_volume(v, CFG)
    v2 = v.copy(); v2.iloc[300:] *= 50
    b = p1.abnormal_volume(v2, CFG)
    pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])
    assert a.iloc[:CFG.av_min_long - 1].isna().all().all()


def test_group_labels_balanced():
    x = pd.DataFrame(np.random.default_rng(1).standard_normal((5, 30)))
    lab = p1.group_labels(x, 3)
    for _, row in lab.iterrows():
        assert sorted(np.bincount(row.to_numpy())) == [10, 10, 10]


def test_reversal_weights_sum():
    rng = np.random.default_rng(2)
    sig = pd.DataFrame(rng.standard_normal((4, 60)))
    lab = pd.DataFrame(np.zeros((4, 60), np.int8))
    w = p1.reversal_weights(sig, lab, 0, 5)
    np.testing.assert_allclose(w.clip(lower=0).sum(axis=1), 1.0)
    np.testing.assert_allclose(w.clip(upper=0).sum(axis=1), -1.0)
    # long the lowest signal, short the highest
    for t in range(4):
        assert w.iloc[t, sig.iloc[t].idxmin()] > 0 and w.iloc[t, sig.iloc[t].idxmax()] < 0


def test_ls_returns_timing_and_cost():
    idx = pd.RangeIndex(6)
    tgt = pd.DataFrame({"a": [0, 1.0, 0, 0, 0, 0]}, index=idx)          # formed at close of day 1
    ret = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}, index=idx)
    cost = pd.DataFrame({"a": [1e-3] * 6}, index=idx)
    out = p1.ls_returns(tgt, ret, cost, skip=1)
    # traded at the close of day 2, held through day 3 only (target back to 0 at day 2 -> traded out day 3)
    np.testing.assert_allclose(out["gross"].to_numpy(), [0, 0, 0, 0.04, 0, 0])
    np.testing.assert_allclose(out["cost"].to_numpy(), [0, 0, 1e-3, 1e-3, 0, 0])


def test_arm_d_detects_planted_effect():
    c, o, v = p1.synthetic_daily(n_stocks=150, effect=0.3)
    av = p1.abnormal_volume(v, CFG)
    sig = np.log(c / c.shift(1))
    lab = p1.group_labels(av, 3, eligible=sig.notna())
    win = c.index[c.index >= "2020-01-01"]
    res = p1.arm_d_horizon(sig, lab, c.pct_change(), None, 1, CFG, win)
    low, high = res[0]["gross"].mean(), res[2]["gross"].mean()
    assert low > 0 > high


def test_label_shuffle_keeps_group_sizes():
    lab = pd.DataFrame(np.array([[0, 1, 2, -1, 0, 1]] * 3, np.int8))
    sh = p1.shuffle_labels(lab, np.random.default_rng(0))
    assert (sh.to_numpy() == -1).sum() == 3
    for r in sh.to_numpy():
        assert sorted(r[r >= 0]) == [0, 0, 1, 1, 2]


def test_trade_label_rows():
    ds = np.array([0, 26, 52, 78])
    np.testing.assert_array_equal(p1.trade_label_rows(np.array([27, 26, 52, 60]), ds), [0, -1, 0, 1])


def test_cluster_bootstrap_diff():
    rng = np.random.default_rng(3)
    n = 4000
    g = rng.integers(0, 2, n)
    x = np.where(g == 0, 0.002, 0.0) + rng.standard_normal(n) * 0.01
    obs, lo, hi = p1.cluster_bootstrap_diff(x, g, rng.integers(0, 200, n), 0, 1, 2000, 0)
    assert lo > 0 and lo < obs < hi


@pytest.fixture(scope="module")
def s05run():
    cfg = s05.S05Config()
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    sc = np.array([panel.symbols.index(s) for s in stocks])
    sect = {s: k for k, v in UNIVERSE_01.items() for s in v}
    ec = np.array([panel.symbols.index(SECTOR_ETF_01[sect[s]]) for s in stocks])
    P = s05.estimate_all(panel, sc, ec, cfg, s05.first_decision_day(cfg))
    return cfg, panel, sc, ec, P


def test_entry_mask_all_true_is_unchanged(s05run):
    cfg, panel, sc, ec, P = s05run
    a = s05.backtest(panel, P, sc, ec, cfg)
    b = s05.backtest(panel, P, sc, ec, cfg, entry_mask=np.ones((panel.n_days, len(sc)), bool))
    assert a["final_equity"] == b["final_equity"] and len(a["trades"]) == len(b["trades"])


def test_entry_mask_restricts_entries(s05run):
    cfg, panel, sc, ec, P = s05run
    m = np.zeros((panel.n_days, len(sc)), bool)
    assert len(s05.backtest(panel, P, sc, ec, cfg, entry_mask=m)["trades"]) == 0
    m[:, :5] = True
    tr = s05.backtest(panel, P, sc, ec, cfg, entry_mask=m)["trades"]
    assert len(tr) > 0 and tr["i"].max() < 5
    rows = p1.trade_label_rows(tr["entry_bar"].to_numpy(), panel.day_start)
    assert np.all(m[rows, tr["i"].to_numpy()])
