import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import hawkes as H
from equity_trading import regime_ar as R

HC = H.HWKConfig(n_starts=2, n_boot=200)


def test_geometric_excitation_is_causal_and_normalized():
    x = np.zeros(400)
    x[10] = 1.0
    r = H._geo_excitation(x, 0.9)
    assert r[10] == 0 and r[11] == pytest.approx(0.1)
    assert r[11:].sum() == pytest.approx(1 - 0.9 ** 389)           # kernel weights sum to 1


def test_power_law_excitation_matches_direct_sum():
    rng = np.random.default_rng(0)
    x = (rng.random(3000) < 0.05).astype(float)
    g = H.pl_kernel(2.0, 0.7, 200)
    exc = H._pl_excitation(x, g)
    t = 1500
    direct = sum(g[k - 1] * x[t - k] for k in range(1, 201))
    assert exc[t] == pytest.approx(direct, abs=1e-10)
    assert exc[0] == 0 and g.sum() == pytest.approx(1.0)


def test_hawkes_mle_recovers_branching_ratio():
    e = H.simulate_hawkes(150_000, mu=0.02, n=0.5, beta=0.9, seed=1)
    obs = np.ones(len(e), bool)
    f = H.fit_hawkes("HX", e, np.zeros(len(e)), obs, 0.0, HC)
    assert f.n == pytest.approx(0.5, abs=0.06)
    assert f.beta == pytest.approx(0.9, abs=0.03)
    assert f.mu == pytest.approx(0.02, abs=0.004)


def test_hawkes_beats_chain_and_decay_exceeds_chain_on_hawkes_data():
    e = H.simulate_hawkes(100_000, mu=0.02, n=0.6, beta=0.97, seed=2)
    obs = np.ones(len(e), bool)
    cnt = H.decay_counts(e, obs, np.repeat(np.arange(len(e) // 276 + 1), 276)[: len(e)], 120)
    cv = H.curves(cnt)
    assert H.excess(cv["C"], cv["C_chain"], (6, 60)) > 0.2          # slow excitation: chain decays too fast
    f = H.fit_hawkes("HX", e, np.zeros(len(e)), obs, 0.0, HC)
    lam, _ = H.intensity(f, e, np.zeros(len(e)))
    pc, _ = H.prob_chain(e, obs, obs)
    assert H.bernoulli_score(1 - np.exp(-lam), e).sum() > H.bernoulli_score(pc, e).sum() + 100


def test_markov_data_has_no_excess_over_its_hmm_curve():
    z, _ = R.simulate_msar(200_000, phis=(-0.05, 0.2), p_stay=0.9, seed=3)
    day = np.repeat(np.arange(len(z) // 276 + 1), 276)[: len(z)]
    same = np.r_[False, day[1:] == day[:-1]]
    s = pd.DataFrame({"trade_date": day, "z": z, "zl": np.where(same, np.r_[0.0, z[:-1]], 0.0), "obs": True,
                      "imb": 0.0, "vsurp": 0.0})
    m = R.fit(s, 2, False, np.ones(len(s), bool), R.RSWConfig(n_starts=2))
    u = H.threshold(s, np.ones(len(s), bool), 0.96)
    C_hmm = H.hmm_null_curve(m, u, 1_000_000, 120, seed=4)
    d = H.decay_test(s, u, {"M1": C_hmm}, HC)
    assert not d["H1"]
    assert abs(d["excess_vs_M1"]) < 0.05


def test_marks_and_power_law_fit_run():
    rng = np.random.default_rng(5)
    e = H.simulate_hawkes(40_000, mu=0.03, n=0.4, beta=0.8, seed=5)
    y = np.where(e > 0, 2 + rng.exponential(0.5, len(e)), 0.0)
    obs = np.ones(len(e), bool)
    fm = H.fit_hawkes("HXM", e, y, obs, 2.0, HC)
    fp = H.fit_hawkes("HP", e, y, obs, 2.0, H.HWKConfig(n_starts=1, pl_kmax=500))
    assert 0.2 < fm.n < 0.6 and 0.1 < fp.n < 0.8
    g = H.gpd_fit(y[e > 0] - 2.0, np.ones(int(e.sum())) * 1.5)
    assert g["xi"] == pytest.approx(0.0, abs=0.1) and g["sigma"] == pytest.approx(0.5, abs=0.05)
    assert g["p_kappa0"] > 0.01 or np.isnan(g["p_kappa0"])


def test_prob_msar_matches_simulation_rate():
    z, _ = R.simulate_msar(100_000, phis=(-0.1, 0.3), p_stay=0.95, seed=6)
    s = pd.DataFrame({"trade_date": 0, "z": z, "zl": np.r_[0.0, z[:-1]], "obs": True, "imb": 0.0, "vsurp": 0.0})
    m = R.fit(s, 2, False, np.ones(len(s), bool), R.RSWConfig(n_starts=2))
    p = H.prob_msar(s, m, 2.0)
    assert p.mean() == pytest.approx((np.abs(z) >= 2.0).mean(), abs=0.003)
