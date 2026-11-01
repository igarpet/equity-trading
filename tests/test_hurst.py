import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fou_eq.data import regular_session, synthetic_closes  # noqa: E402
from fou_eq.hurst import (ar1_fit, arma11_fit, fbm, hurst_cof, noise_sd,  # noqa: E402
                          simulate_arma11, simulate_ou_noise)
from fou_eq.spreads import bar_slot, deseasonalise, ols_spread  # noqa: E402


@pytest.mark.parametrize("h", [0.3, 0.5, 0.7])
def test_cof_recovers_h_on_fbm(h):
    rng = np.random.default_rng(1)
    paths = fbm(1500, h, 8, rng)
    est = np.mean([hurst_cof(p) for p in paths])
    assert abs(est - h) < 0.04


def test_cof_biased_low_on_fast_ou():
    """Mean reversion alone pulls the estimate below 0.5 (modestly: about -0.02 at phi=0.8)."""
    rng = np.random.default_rng(2)
    paths = simulate_ou_noise(600, phi=0.8, sd=1.0, eta=0.0, n_paths=200, rng=rng)
    assert np.mean([hurst_cof(p) for p in paths]) < 0.49


def test_noise_pulls_estimate_lower():
    rng = np.random.default_rng(3)
    clean = simulate_ou_noise(600, 0.95, 1.0, 0.0, 50, rng)
    noisy = simulate_ou_noise(600, 0.95, 1.0, 1.0, 50, rng)
    assert np.mean([hurst_cof(p) for p in noisy]) < np.mean([hurst_cof(p) for p in clean]) - 0.05


def test_ar1_and_noise_recovery():
    rng = np.random.default_rng(4)
    x = simulate_ou_noise(20000, 0.9, 1.0, 0.0, 1, rng)[0]
    assert abs(ar1_fit(x)["phi"] - 0.9) < 0.01
    y = simulate_ou_noise(20000, 0.98, 0.3, 1.0, 1, rng)[0]
    assert 0.8 < noise_sd(y) < 1.2


def test_arma_null_matches_ou_plus_noise():
    rng = np.random.default_rng(5)
    x = simulate_ou_noise(3000, 0.95, 0.5, 0.7, 1, rng)[0]
    fit = arma11_fit(x)
    assert fit is not None and fit["theta"] < 0
    sims = simulate_arma11(3000, fit["phi"], fit["theta"], fit["sd"], 20, rng)
    assert abs(np.mean([hurst_cof(s) for s in sims]) - hurst_cof(x)) < 0.06


def test_deseasonalise_flattens_profile():
    px = synthetic_closes(["A", "B"], n_days=30, bars_per_day=13, seed=0)
    x, _, _ = ols_spread(np.log(px["A"]), np.log(px["B"]))
    z, prof = deseasonalise(x)
    slots = pd.Series(bar_slot(z.index), index=z.index)
    sd = z.diff().groupby(slots).std()
    assert sd.max() / sd.min() < 1.3
    assert prof.max() / prof.min() > 1.5


def test_regular_session_and_slots():
    px = synthetic_closes(["A"], n_days=3, bars_per_day=13)
    assert len(regular_session(px)) == len(px)
    s = bar_slot(px.index)
    assert s[0] == 0 and s[13] == 0 and s.max() == 12
