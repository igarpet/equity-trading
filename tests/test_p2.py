import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p1, p2  # noqa: E402

CFG = p2.P2Config()


def test_regime_has_no_lookahead():
    idx = pd.bdate_range("2016-01-01", periods=400)
    v = p2.synthetic_vix(idx)
    a = p2.vix_regime(v, CFG)
    v2 = v.copy(); v2.iloc[300:] *= 3
    b = p2.vix_regime(v2, CFG)
    pd.testing.assert_series_equal(a.iloc[:300], b.iloc[:300])
    assert a.iloc[:CFG.vix_min - 1].isna().all()


def test_contrarian_weights_scaled_and_signed():
    r = pd.DataFrame(np.random.default_rng(0).standard_normal((5, 40)))
    w = p2.contrarian_weights(r)
    np.testing.assert_allclose(w.clip(lower=0).sum(axis=1), 1.0)
    np.testing.assert_allclose(w.clip(upper=0).sum(axis=1), -1.0)
    assert (np.sign(w.to_numpy()) == -np.sign(r.sub(r.mean(axis=1), axis=0).to_numpy())).all()


def test_gate_and_alignment():
    idx = pd.RangeIndex(5)
    tgt = pd.DataFrame({"a": [1.0] * 5}, index=idx)
    reg = pd.Series([True, False, True, False, True], index=idx)
    np.testing.assert_allclose(p2.gate_by_regime(tgt, reg)["a"], [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(p2.aligned_signal(pd.Series(np.arange(5.0), index=idx), idx, 1).to_numpy()[2:], [0, 1, 2])


def test_h1_detects_planted_vix_effect():
    c, v, vix = p2.synthetic_daily_vix(n_stocks=120, slope=0.3)
    ret = c.pct_change()
    w = p2.contrarian_weights(ret)
    g = p1.ls_returns(w, ret, None, CFG.skip)["gross"]
    reg = p2.vix_regime(vix, CFG)
    win = g.index[g.index >= "2016-12-01"]
    res = p2.hac_slope(g.loc[win], p2.aligned_signal(reg.astype(float), g.index, CFG.skip).loc[win], 5)
    assert res["slope"] > 0 and res["p_one_sided"] < 0.01


def test_shifted_regime_keeps_count_and_clustering():
    rng = np.random.default_rng(0)
    a = np.repeat([True, False] * 10, 30)
    valid = np.ones(len(a), bool); valid[:50] = False
    b = p2.shifted_regime(a, rng, 60, valid)
    assert b.sum() == a.sum() and (b[:50] == a[:50]).all()
    assert np.sum(b[1:] != b[:-1]) <= np.sum(a[1:] != a[:-1]) + 2


def test_regime_mask_shape():
    m = p2.regime_mask(np.array([True, False, True]), 4)
    assert m.shape == (3, 4) and m[0].all() and not m[1].any()
