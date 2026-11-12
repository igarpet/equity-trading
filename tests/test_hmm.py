import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from equity_trading.hmm import GaussianHMM  # noqa: E402


def _simulate(T=4000, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    mu = np.array([[0.0, -1.0], [0.0, 1.0]])
    sd = np.array([[0.5, 0.3], [1.5, 0.3]])
    s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    X = mu[s] + sd[s] * rng.standard_normal((T, 2))
    return X, s, A


def test_recovers_two_states():
    X, s, A = _simulate()
    h = GaussianHMM(2).fit(X, n_restarts=3)
    assert np.allclose(h.mu[:, 1], [-1, 1], atol=0.1)
    assert np.allclose(np.diag(h.A), np.diag(A), atol=0.02)
    acc = ((h.filter(X).argmax(1)) == s).mean()
    assert acc > 0.95


def test_filter_is_causal():
    X, _, _ = _simulate(T=600, seed=1)
    h = GaussianHMM(2).fit(X, n_restarts=2)
    full = h.filter(X)
    part = h.filter(X[:300])
    assert np.allclose(full[:300], part)
    nxt = h.predict_next(X[:300])
    assert np.allclose(nxt[-1], full[299] @ h.A)
