import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p3, s04, s05b  # noqa: E402

PD = p3.PER_DAY


def test_to_30m_aggregates_full_days():
    panel, stocks = s04.synthetic_ohlc_panel([2025])
    cols = np.array([panel.symbols.index(s) for s in stocks])
    out = p3.to_30m(panel, cols)
    assert out["R"].shape == (len(out["days"]) * PD, len(cols))
    b = out["entry_bar"][0]
    np.testing.assert_allclose(out["R"][0], np.log(panel.c[b + 1, cols] / panel.o[b, cols]), rtol=1e-5)


def test_fm_slopes_detect_same_slot_persistence():
    R = p3.synthetic_intervals(n_days=200, n_stocks=200, effect=1.5)
    lags = np.arange(1, 3 * PD + 1)
    S = p3.fm_slopes(R, lags, min_n=100)
    m = np.nanmean(S, 0)
    assert m[PD - 1] > 0.02 and m[2 * PD - 1] > 0.02          # lags 13, 26
    assert abs(m[0]) < m[PD - 1] / 3                           # lag 1 near zero
    D, N = p3.periodicity_series(S, lags)
    assert p3.nw_mean(D - N, 13)["p_one_sided"] < 0.01


def test_fm_slopes_null_is_flat():
    R = p3.synthetic_intervals(n_days=150, n_stocks=200, effect=0.0)
    S = p3.fm_slopes(R, np.array([1, 13, 26]), min_n=100)
    assert np.all(np.abs(np.nanmean(S, 0)) < 0.01)


def test_same_slot_signal_uses_only_past_sessions():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((10 * PD, 3)).astype(np.float32)
    sig = p3.same_slot_signal(R, 2)
    R3 = R.reshape(10, PD, 3)
    np.testing.assert_allclose(sig.reshape(10, PD, 3)[5, 4], R3[3:5, 4].mean(0), rtol=1e-5)
    assert np.isnan(sig[: 2 * PD]).all()
    R2 = R.copy(); R2[6 * PD:] += 100
    np.testing.assert_allclose(p3.same_slot_signal(R2, 2)[: 7 * PD], sig[: 7 * PD])


def test_decile_weights_and_strategy_costs():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((2 * PD, 100)).astype(np.float32)
    W = p3.decile_weights(sig, 0.1, 50)
    np.testing.assert_allclose(W.clip(min=0).sum(1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(W.clip(max=0).sum(1), -1.0, rtol=1e-6)
    # constant weights within a day: cost = entry (2 units) + unwind (2 units) per day
    Wc = np.repeat(W[:1], 2 * PD, 0)
    H = np.full(Wc.shape, 1e-4, np.float32)
    s = p3.strategy(Wc, np.zeros_like(Wc), H)
    np.testing.assert_allclose(s["cost"].sum(), 2 * 4 * 1e-4, rtol=1e-6)


def test_strategy_earns_planted_effect():
    R = p3.synthetic_intervals(n_days=150, n_stocks=200, effect=1.5)
    W = p3.decile_weights(p3.same_slot_signal(R, 20))
    s = p3.strategy(W, R, None)
    assert s["gross"].iloc[20 * PD:].mean() > 0
