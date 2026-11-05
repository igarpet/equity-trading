import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p6  # noqa: E402

CFG = p6.P6Config()


def test_pca_residuals_remove_factors_and_no_lookahead():
    R = p6.synthetic_returns(n_days=400, n_stocks=60, rev=0.0)
    pr = p6.pca_residuals(R, 252, 5, verbose=False)
    t = 300
    ok = np.isfinite(pr["resid"][t])
    # residual of day t uses a window ending at t-1: changing day t+1 leaves it unchanged
    R2 = R.copy(); R2.iloc[t + 1:] *= 3
    pr2 = p6.pca_residuals(R2, 252, 5, verbose=False)
    np.testing.assert_allclose(pr["resid"][t], pr2["resid"][t])
    raw = R.iloc[t].to_numpy()
    assert np.nanstd(pr["resid"][t][ok]) < np.std(raw[ok])
    assert np.isnan(pr["resid"][:252]).all()


def test_inputs_shape_and_scaling():
    rng = np.random.default_rng(0)
    res = rng.standard_normal((40, 20)).astype(np.float32)
    x = p6.inputs(res, 30)
    assert x.shape == (40, 20, 30) and np.isnan(x[:29]).all()
    np.testing.assert_allclose(np.std(x[35, :, -1]), 1.0, rtol=1e-5)
    np.testing.assert_allclose(x[35, 3] * np.std(np.cumsum(res[6:36], 0)[-1]), np.cumsum(res[6:36, 3]), rtol=1e-4)


def test_neutralise():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 5)); w = rng.standard_normal(50); ok = np.ones(50, bool); ok[:5] = False
    out = p6.neutralise(w, X, ok)
    assert abs(np.abs(out).sum() - 1) < 1e-12 and np.allclose(X[ok].T @ out[ok], 0, atol=1e-12) and (out[:5] == 0).all()


def test_evaluate_timing():
    W = np.zeros((6, 1)); W[1] = 1.0
    R = np.arange(6, dtype=float)[:, None] / 100
    C = np.full((6, 1), 1e-3)
    e = p6.evaluate(W, R, C, skip=1)
    np.testing.assert_allclose(e["gross"], [0, 0, 0, 0.03, 0, 0])      # formed d=1, traded d=2, earns d=3
    np.testing.assert_allclose(e["cost"], [0, 0, 1e-3, 1e-3, 0, 0])


def test_dataset_alignment_and_benchmarks_detect_reversal():
    R = p6.synthetic_returns(n_days=700, n_stocks=80, rev=0.3)
    cost = pd.DataFrame(1e-4, index=R.index, columns=R.columns)
    d = p6.build_dataset(R, None, cost, CFG, verbose=False)
    t = 500
    np.testing.assert_allclose(d["ret"][t], R.to_numpy(np.float32)[t + 2])
    rows = np.arange(300, 690)
    for sc in (p6.linear_scores(d["x"]), p6.ou_scores(d["x"])):
        W = p6.weights_from_scores(np.where(d["ok"], sc, np.nan), np.where(d["ok"], 1.0, np.nan),
                                   d["X"], d["ok"])
        g = np.nansum(W[rows] * d["ret"][rows], 1)
        assert g.mean() > 0


def test_fit_schedule_no_overlap():
    idx = pd.bdate_range("2012-01-02", "2018-12-31")
    cfg = p6.P6Config(first_test="2017-01-03", end="2018-12-31")
    s = p6.fit_schedule(idx, cfg)
    for f in s:
        assert f["valid"][1] + cfg.skip + 1 <= f["predict"][0]
        assert f["train"][1] == f["valid"][0] and f["valid"][1] - f["train"][0] == cfg.train_days
    assert s[0]["predict"][0] == idx.get_loc(pd.Timestamp("2017-01-03"))


def test_placebo_gross_shape():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((20, 10)); R = rng.standard_normal((20, 10)) * 0.01; ok = np.ones((20, 10), bool)
    pg = p6.placebo_gross(W, R, ok, np.arange(5, 15), 50, 0)
    np.testing.assert_allclose(np.sort(pg[:, 0] * 0 + (W[5] @ R[5])), np.full(50, W[5] @ R[5]))
    assert pg.shape == (50, 10) and np.isfinite(pg).all()


def test_model_trains_on_planted_reversal():
    torch = pytest.importorskip("torch")
    from equity_trading import p6_model
    R = p6.synthetic_returns(n_days=700, n_stocks=60, rev=0.4)
    cost = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    d = p6.build_dataset(R, None, cost, CFG, verbose=False)
    rows = np.flatnonzero(d["ok"].sum(1) > 30)
    tr, va, te = rows[:200], rows[200:300], rows[300:]
    model, info = p6_model.fit(d, tr, va, seed=0, epochs=8, patience=8, chunk=50, device="cpu")
    s = p6_model.predict_scores(model, d, te, device="cpu")
    W = p6.weights_from_scores(s, np.ones_like(s), d["X"][te], d["ok"][te])
    assert np.nansum(W * d["ret"][te], 1).mean() > 0


def test_evaluate_rows_matches_evaluate():
    R = p6.synthetic_returns(n_days=400, n_stocks=40, rev=0.3)
    cost = pd.DataFrame(2e-4, index=R.index, columns=R.columns)
    d = p6.build_dataset(R, None, cost, CFG, verbose=False)
    rows = np.arange(300, 390)
    W = p6.weights_from_scores(np.where(d["ok"], p6.linear_scores(d["x"]), np.nan), np.ones(d["ok"].shape), d["X"], d["ok"])
    a = p6.evaluate_rows(W, d, rows)
    Wfull = np.zeros_like(W); Wfull[rows] = W[rows]; Wfull[rows[0] - 1] = W[rows[0] - 1]
    b = p6.evaluate(Wfull, R.to_numpy(), cost.to_numpy(), skip=1)
    np.testing.assert_allclose(a["gross"].to_numpy(), b["gross"].to_numpy()[rows + 2], atol=1e-6)
