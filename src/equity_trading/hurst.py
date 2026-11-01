"""Hurst estimation, fOU moments, and simulation helpers."""
from __future__ import annotations

from math import gamma

import numpy as np

HURST_VERSION = "0.1.0"


def hurst_cof(x: np.ndarray) -> float:
    """Change-of-frequency estimator from the thesis (Kubilius & Skorniakov 2017).

    D1_t = X_t - 2X_{t-1} + X_{t-2},  D2_t = X_t - 2X_{t-2} + X_{t-4},
    H = 0.5 * log2( sum D2^2 / sum D1^2 ), summed over t = 4..n-1, clipped to [0.01, 0.99].
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return np.nan
    t = np.arange(4, x.size)
    d1 = x[t] - 2 * x[t - 1] + x[t - 2]
    d2 = x[t] - 2 * x[t - 2] + x[t - 4]
    s1, s2 = np.sum(d1 ** 2), np.sum(d2 ** 2)
    if s1 <= 0 or s2 <= 0:
        return np.nan
    return float(np.clip(0.5 * np.log2(s2 / s1), 0.01, 0.99))


def hurst_cof_step(x: np.ndarray, m: int = 2) -> float:
    """Change-of-frequency estimator using second differences at steps m and 2m
    (m=1 is the thesis estimator). Larger m reduces the weight of bar-level noise."""
    x = np.asarray(x, dtype=float)
    if x.size < 8 * m:
        return np.nan
    t = np.arange(4 * m, x.size)
    d1 = x[t] - 2 * x[t - m] + x[t - 2 * m]
    d2 = x[t] - 2 * x[t - 2 * m] + x[t - 4 * m]
    s1, s2 = np.sum(d1 ** 2), np.sum(d2 ** 2)
    if s1 <= 0 or s2 <= 0:
        return np.nan
    return float(np.clip(0.5 * np.log2(s2 / s1), 0.01, 0.99))


def increment_acf_sum(x: np.ndarray, lags: range = range(2, 11)) -> float:
    """Sum of increment autocorrelations at lags 2..10.

    i.i.d. measurement noise only touches lag 1 of the increments; OU increments have
    small, geometrically decaying autocorrelation; fGn with H < 0.5 has negative
    power-law autocorrelation at every lag. Lag 1 is excluded on purpose.
    """
    dx = np.diff(np.asarray(x, dtype=float))
    dx = dx - dx.mean()
    v = np.dot(dx, dx)
    if v <= 0:
        return np.nan
    return float(sum(np.dot(dx[k:], dx[:-k]) / v for k in lags))


def fou_moments(x: np.ndarray, h: float, dt: float = 1.0) -> dict:
    """Thesis fOU moment estimates: mu, sigma (second differences), kappa (stationary variance)."""
    x = np.asarray(x, dtype=float)
    mu, var = x.mean(), x.var(ddof=1)
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    sigma2 = np.mean(d2 ** 2) / ((4 - 2 ** (2 * h)) * dt ** (2 * h))
    kappa = (sigma2 * gamma(2 * h + 1) / (2 * var)) ** (1 / (2 * h)) if var > 0 else np.nan
    return {"mu": mu, "sigma": float(np.sqrt(sigma2)), "kappa": float(kappa),
            "half_life": float(np.log(2) / kappa) if kappa and kappa > 0 else np.nan}


def ar1_fit(x: np.ndarray) -> dict:
    """Discrete OU fit X_{t+1} - mu = phi (X_t - mu) + e. Returns phi, residual s.d., mu,
    kappa = -ln(phi), half-life in bars."""
    x = np.asarray(x, dtype=float)
    a, b = x[:-1], x[1:]
    am, bm = a.mean(), b.mean()
    phi = np.sum((a - am) * (b - bm)) / np.sum((a - am) ** 2)
    c = bm - phi * am
    resid = b - c - phi * a
    ok = 0 < phi < 1
    mu = c / (1 - phi) if ok else np.nan
    kappa = -np.log(phi) if ok else np.nan
    return {"phi": float(phi), "sd": float(resid.std(ddof=2)), "mu": float(mu),
            "kappa": float(kappa), "half_life": float(np.log(2) / kappa) if ok else np.nan}


def noise_sd(x: np.ndarray) -> float:
    """i.i.d. measurement noise s.d. implied by negative lag-1 autocovariance of increments.

    For observed = latent + eta, cov(dX_t, dX_{t-1}) = latent part - var(eta).
    Zero when the autocovariance is non-negative.
    """
    dx = np.diff(np.asarray(x, dtype=float))
    dx = dx - dx.mean()
    acov1 = np.mean(dx[1:] * dx[:-1])
    return float(np.sqrt(-acov1)) if acov1 < 0 else 0.0


def simulate_ou_noise(n: int, phi: float, sd: float, eta: float, n_paths: int,
                      rng: np.random.Generator) -> np.ndarray:
    """OU (exact AR(1) discretisation, stationary start) plus i.i.d. noise. Shape (n_paths, n)."""
    e = rng.standard_normal((n_paths, n)) * sd
    x = np.empty((n_paths, n))
    x[:, 0] = rng.standard_normal(n_paths) * sd / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    if eta > 0:
        x = x + rng.standard_normal((n_paths, n)) * eta
    return x


def arma11_fit(x: np.ndarray) -> dict | None:
    """ARMA(1,1) with constant: the discrete form of OU + i.i.d. measurement noise.

    Returns phi, theta (MA), innovation s.d.; None if the fit fails or is non-stationary.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        res = ARIMA(np.asarray(x, dtype=float), order=(1, 0, 1), trend="c").fit()
        phi, theta = float(res.arparams[0]), float(res.maparams[0])
        sd = float(np.sqrt(res.params[-1]))
    except Exception:
        return None
    if not (0 < phi < 1) or not np.isfinite(sd):
        return None
    return {"phi": phi, "theta": theta, "sd": sd}


def simulate_arma11(n: int, phi: float, theta: float, sd: float, n_paths: int,
                    rng: np.random.Generator, burn: int = 500) -> np.ndarray:
    """x_t = phi x_{t-1} + e_t + theta e_{t-1}. Shape (n_paths, n)."""
    from scipy.signal import lfilter
    e = rng.standard_normal((n_paths, n + burn)) * sd
    return lfilter([1.0, theta], [1.0, -phi], e, axis=1)[:, burn:]


def fbm(n: int, h: float, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Brownian motion on 1..n via Cholesky of the fGn covariance (tests only)."""
    k = np.arange(n)
    g = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    cov = g[np.abs(k[:, None] - k[None, :])]
    lchol = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    return np.cumsum(rng.standard_normal((n_paths, n)) @ lchol.T, axis=1)
