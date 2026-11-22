import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import hsmm as S
from equity_trading import regime_ar as R

RC = S.HSMMConfig(D=40, max_iter=200)


def _frame(z, day=None):
    day = np.zeros(len(z), int) if day is None else day
    same = np.r_[False, day[1:] == day[:-1]]
    return pd.DataFrame({"trade_date": day, "z": z, "zl": np.where(same, np.r_[0.0, z[:-1]], 0.0), "obs": True,
                         "imb": 0.0, "vsurp": 0.0})


def test_ages():
    e = np.array([0, 1, 0, 0, 1, 1, 0], float)
    assert list(S.ages(e)) == [0, 0, 1, 2, 3, 1, 1]


def test_iid_events_have_flat_hazard():
    rng = np.random.default_rng(0)
    e = (rng.random(400_000) < 0.04).astype(float)
    hc = S.hazard_curve(S.hazard_counts(e, np.ones(len(e), bool), np.zeros(len(e), np.int64), S.HSMMConfig().age_edges))
    assert abs(S.hazard_slope(hc, (13, 480), 200)) < 0.05


def test_geometric_hsmm_equals_hmm_likelihood():
    z, _ = R.simulate_msar(20_000, phis=(-0.2, 0.3), p_stay=0.95, seed=1)
    s = _frame(z)
    m = R.fit(s, 2, False, np.ones(len(s), bool), R.RSWConfig(n_starts=2))
    h = S.init_from_hmm(m, "WB", False, 40, 1.0)
    zz, zl, obs, ld = S._arrays(s, 40)
    ll = S._forward(zz, zl, obs, ld, h.c, h.phi, h.b, h.s2, h.pi, h.hazards(), h.W)[3]
    y, G, ob, At, lB = R._pass(s, m)
    ll_hmm = R._forward(lB, m.pi, At)[4]
    assert ll == pytest.approx(ll_hmm, abs=1e-6)


def test_em_recovers_rising_calm_hazard_and_beats_hmm():
    z, st, _ = S.simulate_hsmm(60_000, lam=(0.01, 0.6), beta=(1.8, 1.0), seed=2)
    s = _frame(z)
    mask = np.ones(len(s), bool)
    m = R.fit(s, 2, False, mask, R.RSWConfig(n_starts=2))
    h = S.fit(s, mask, m, "WB", False, RC)
    tab = h.table()
    assert tab.loc[0, "shape"] == pytest.approx(1.8, abs=0.3)          # calm state (lower phi)
    assert tab.loc[0, "t_shape_vs_1"] > 3
    assert h.loglik > m.loglik + 20
    hn = S.fit(s, mask, m, "NB", False, RC)                             # NB agrees: shape r > 1 (rising)
    assert hn.table().loc[0, "shape"] > 1.3 and hn.table().loc[0, "t_shape_vs_1"] > 3


def test_filter_is_causal_and_age_drift_recovered():
    z, _, _ = S.simulate_hsmm(40_000, lam=(0.05, 0.6), beta=(1.0, 1.0), b=(0.4, 0.0), seed=3)
    s = _frame(z)
    mask = np.ones(len(s), bool)
    m = R.fit(s, 2, False, mask, R.RSWConfig(n_starts=2))
    hb = S.fit(s, mask, m, "WB", False, RC)
    hc = S.fit(s, mask, m, "WB", True, RC, warm=hb)
    assert hc.b[0] == pytest.approx(0.4, abs=0.1) and abs(hc.b[0] / hc.b_se[0]) > 3
    f1 = S.filter_run(s, hc, 2.0)
    s2 = s.copy()
    s2.loc[30_000:, ["z", "zl"]] = 0.0
    f2 = S.filter_run(s2, hc, 2.0)
    pd.testing.assert_frame_equal(f1.iloc[:30_000], f2.iloc[:30_000])
    assert np.allclose(f1[["p0", "p1"]].sum(axis=1), 1.0)
