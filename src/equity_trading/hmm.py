"""Gaussian hidden Markov model with diagonal covariances (Baum-Welch), and causal filtering.

Written for FXS 13 (no extra dependency). `filter` uses only observations up to each day, so filtered and
one-step-ahead state probabilities are usable for trading; `smooth` uses the whole sample (look-ahead: description only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOG2PI = np.log(2 * np.pi)


def _emission_logp(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """(T, K) log density of each row of X under each state's diagonal Gaussian."""
    d = X[:, None, :] - mu[None, :, :]
    return -0.5 * (LOG2PI * X.shape[1] + np.log(var).sum(1)[None, :] + (d * d / var[None, :, :]).sum(2))


def _forward(logB: np.ndarray, pi: np.ndarray, A: np.ndarray):
    """Scaled forward pass. Returns filtered probabilities alpha (T, K) and log-likelihood."""
    T, K = logB.shape
    alpha = np.empty((T, K))
    m = logB.max(1, keepdims=True)
    B = np.exp(logB - m)
    ll = 0.0
    a = pi * B[0]
    for t in range(T):
        if t:
            a = (alpha[t - 1] @ A) * B[t]
        s = a.sum()
        alpha[t] = a / s
        ll += np.log(s) + m[t, 0]
    return alpha, ll


@dataclass
class GaussianHMM:
    n_states: int
    var_floor: float = 1e-3
    pi: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)
    mu: np.ndarray = field(default=None, repr=False)
    var: np.ndarray = field(default=None, repr=False)
    loglik: float = -np.inf

    def _init(self, X: np.ndarray, rng: np.random.Generator):
        K = self.n_states
        idx = rng.choice(len(X), K, replace=False)
        self.mu = X[idx].copy()
        self.var = np.tile(X.var(0), (K, 1))
        self.pi = np.full(K, 1.0 / K)
        self.A = np.full((K, K), 0.05 / max(K - 1, 1)) + np.eye(K) * (0.95 - 0.05 / max(K - 1, 1))

    def _em(self, X: np.ndarray, n_iter: int, tol: float):
        K, T = self.n_states, len(X)
        prev = -np.inf
        for _ in range(n_iter):
            logB = _emission_logp(X, self.mu, self.var)
            m = logB.max(1, keepdims=True)
            B = np.exp(logB - m)
            alpha, ll = _forward(logB, self.pi, self.A)
            beta = np.empty((T, K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                b = self.A @ (B[t + 1] * beta[t + 1])
                beta[t] = b / b.sum()
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            x = alpha[:-1, :, None] * self.A[None] * (B[1:] * beta[1:])[:, None, :]
            xi = (x / x.sum((1, 2), keepdims=True)).sum(0)
            self.pi = gamma[0] + 1e-12
            self.pi /= self.pi.sum()
            self.A = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mu = (gamma.T @ X) / w[:, None]
            self.var = np.maximum((gamma.T @ (X * X)) / w[:, None] - self.mu ** 2, self.var_floor)
            self.loglik = ll
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        return self

    def fit(self, X: np.ndarray, n_restarts: int = 5, n_iter: int = 300, tol: float = 1e-7, seed: int = 0,
            order_col: int = -1) -> "GaussianHMM":
        """Best of n_restarts EM runs. States are then ordered by the mean of column `order_col` (ascending),
        so state labels mean the same thing across refits."""
        X = np.asarray(X, float)
        best = None
        for r in range(n_restarts):
            h = GaussianHMM(self.n_states, self.var_floor)
            h._init(X, np.random.default_rng(seed + r))
            h._em(X, n_iter, tol)
            if best is None or h.loglik > best.loglik:
                best = h
        o = np.argsort(best.mu[:, order_col])
        self.pi, self.A = best.pi[o], best.A[np.ix_(o, o)]
        self.mu, self.var, self.loglik = best.mu[o], best.var[o], best.loglik
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..x_t): causal."""
        return _forward(_emission_logp(np.asarray(X, float), self.mu, self.var), self.pi, self.A)[0]

    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """P(state_{t+1} | x_1..x_t): row t is known at the end of day t."""
        return self.filter(X) @ self.A

    def bic(self, X: np.ndarray) -> float:
        K, d = self.n_states, X.shape[1]
        k = (K - 1) + K * (K - 1) + 2 * K * d
        return -2 * self.loglik + k * np.log(len(X))
