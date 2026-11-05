"""P3: Heston, Korajczyk & Sadka (2010) - intraday periodicity in the cross-section of returns.

Protocol: discussions/Intraday_Periodicity/001_claude_p3_intraday_periodicity_preregistration.md

30-minute intervals from the 15-minute panel (open of the first bar, close of the second), full days only
(13 intervals), interval return log(close / open) (overnight gap excluded).
Notebook 01: Fama-MacBeth slopes of r_t on r_{t-k}, k = 1..520; gates on the daily-multiple lags.
Notebook 02: same-slot signal (mean of the stock's returns in this slot over the previous n sessions),
decile long-short held one interval, turnover costs, flat overnight.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

P3_VERSION = "0.1.0"
PER_DAY = 13


@dataclass(frozen=True)
class P3Config:
    max_lag_days: int = 40
    min_names: int = 100
    nw_lags: int = 13
    alpha: float = 0.05
    lookback: int = 20                 # judged same-slot lookback (sessions)
    lookbacks_reported: tuple = (1, 5, 20, 40)
    decile: float = 0.10
    min_names_portfolio: int = 50
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== data
def to_30m(panel, cols: np.ndarray, half_spread: np.ndarray | None = None) -> dict:
    """30-minute intervals on full sessions (26 fifteen-minute bars). Returns R (T30, N) log returns,
    days (D,), entry_bar (T30,) index of the opening 15-minute bar, H (T30, N) one-way cost at that bar."""
    ds = panel.day_start
    full = [d for d in range(panel.n_days) if ds[d + 1] - ds[d] == 2 * PER_DAY]
    first = np.concatenate([ds[d] + 2 * np.arange(PER_DAY) for d in full]) if full else np.array([], int)
    o = panel.o[first][:, cols]
    c = panel.c[first + 1][:, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.log(c / o).astype(np.float32)
    R[~np.isfinite(R)] = np.nan
    out = {"R": R, "days": np.asarray(panel.days)[full], "entry_bar": first}
    if half_spread is not None:
        out["H"] = half_spread[first][:, cols].astype(np.float32)
    return out


def synthetic_intervals(n_days: int = 300, n_stocks: int = 200, effect: float = 1.0, seed: int = 3) -> np.ndarray:
    """(n_days x 13, n_stocks) returns with a persistent stock-by-slot mean (planted periodicity)."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((PER_DAY, n_stocks)) * 4e-4 * effect
    R = mu[None] + rng.standard_normal((n_days, PER_DAY, n_stocks)) * 3e-3
    return R.reshape(-1, n_stocks).astype(np.float32)


# =================================================================== notebook 01
def fm_slopes(R: np.ndarray, lags, min_n: int = 100) -> np.ndarray:
    """(T, L) cross-sectional OLS slope of r_t on r_{t-k} for each interval t and lag k (NaN if < min_n names)."""
    T = len(R)
    out = np.full((T, len(lags)), np.nan, np.float32)
    for j, k in enumerate(lags):
        X, Y = R[:-k].astype(np.float64), R[k:].astype(np.float64)
        m = np.isfinite(X) & np.isfinite(Y)
        n = m.sum(1)
        X0, Y0 = np.where(m, X, 0.0), np.where(m, Y, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my = X0.sum(1) / n, Y0.sum(1) / n
            sxy = (X0 * Y0).sum(1) - n * mx * my
            sxx = (X0 * X0).sum(1) - n * mx * mx
            b = sxy / sxx
        b[(n < min_n) | ~np.isfinite(b)] = np.nan
        out[k:, j] = b
    return out


def nw_mean(x: np.ndarray, lags: int) -> dict:
    """Mean of x with a Newey-West standard error; one-sided p for mean > 0."""
    from scipy.stats import norm
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(v / n)
    t = x.mean() / se if se > 0 else np.nan
    return {"mean": float(x.mean()), "t": float(t), "p_one_sided": float(norm.sf(t)), "n": int(n)}


def periodicity_series(S: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D_t = mean slope over daily-multiple lags; N_t = mean slope over non-multiple lags >= 14."""
    lags = np.asarray(lags)
    mult = (lags % PER_DAY == 0)
    nonm = (~mult) & (lags >= PER_DAY + 1)
    with np.errstate(invalid="ignore"):
        D = np.nanmean(np.where(np.isfinite(S[:, mult]), S[:, mult], np.nan), axis=1)
        N = np.nanmean(np.where(np.isfinite(S[:, nonm]), S[:, nonm], np.nan), axis=1)
    return D, N


def gates_01(s1: dict, s2: dict, alpha: float) -> dict:
    g1 = s1["mean"] > 0 and s1["p_one_sided"] <= alpha
    g2 = s2["mean"] > 0 and s2["p_one_sided"] <= alpha
    return {"S1_persistence": g1, "S2_specificity": g2, "pass": g1 and g2}


# =================================================================== notebook 02
def same_slot_signal(R: np.ndarray, n: int) -> np.ndarray:
    """(T, N): for each interval, the mean of the stock's returns in the same slot over the previous n
    sessions (at least half of them observed); NaN in the first n sessions."""
    T, N = R.shape
    D = T // PER_DAY
    R3 = R.reshape(D, PER_DAY, N).astype(np.float64)
    ok = np.isfinite(R3)
    cs = np.concatenate([np.zeros((1, PER_DAY, N)), np.cumsum(np.where(ok, R3, 0.0), 0)])
    cn = np.concatenate([np.zeros((1, PER_DAY, N)), np.cumsum(ok, 0)])
    out = np.full((D, PER_DAY, N), np.nan)
    s, c = cs[n:D] - cs[:D - n], cn[n:D] - cn[:D - n]          # sessions d-n .. d-1 for d = n .. D-1
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n:] = np.where(c >= max(1, n // 2), s / c, np.nan)
    return out.reshape(T, N).astype(np.float32)


def decile_weights(sig: np.ndarray, q: float = 0.10, min_names: int = 50) -> np.ndarray:
    """(T, N): equal-weight +1 over the top q of each row's signal, -1 over the bottom q."""
    pct = pd.DataFrame(sig).rank(axis=1, pct=True, method="first").to_numpy()
    n = np.isfinite(sig).sum(1)
    top, bot = pct > 1 - q, pct <= q
    with np.errstate(invalid="ignore", divide="ignore"):
        w = top / top.sum(1, keepdims=True) - bot / bot.sum(1, keepdims=True)
    w = np.where(np.isfinite(w), w, 0.0)
    w[n < min_names] = 0.0
    return w.astype(np.float32)


def strategy(W: np.ndarray, R: np.ndarray, H: np.ndarray | None, cost_mult: float = 1.0) -> pd.DataFrame:
    """Per-interval gross and cost. Weights change at each interval open (cost on |change| at that interval's
    cost); the book is unwound at the session close (cost of the last interval). Returns (T, 2) frame."""
    T, N = W.shape
    D = T // PER_DAY
    ret = np.where(np.isfinite(R), np.expm1(R.astype(np.float64)), 0.0)
    gross = (W * ret).sum(1)
    W3 = W.reshape(D, PER_DAY, N).astype(np.float64)
    prev = np.concatenate([np.zeros((D, 1, N)), W3[:, :-1]], 1)
    dw = np.abs(W3 - prev)
    dw[:, -1] += np.abs(W3[:, -1])                       # unwind at the close
    if H is None:
        cost = np.zeros(T)
    else:
        h = np.where(np.isfinite(H), H, np.nanmedian(H)).reshape(D, PER_DAY, N)
        cost = (dw * h).sum(2).reshape(-1) * cost_mult
    return pd.DataFrame({"gross": gross, "cost": cost})


def daily(per_interval: pd.DataFrame, days) -> pd.DataFrame:
    g = per_interval.groupby(np.arange(len(per_interval)) // PER_DAY).sum()
    g.index = pd.to_datetime(list(days)[:len(g)])
    g["net"] = g["gross"] - g["cost"]
    return g


def gates_02(boot_gross: dict, boot_net: dict, net2x_mean: float) -> dict:
    e1 = all(v[1] > 0 for v in boot_gross.values())
    e2 = all(v[1] > 0 for v in boot_net.values()) and net2x_mean > 0
    return {"E1_gross": e1, "E2_net": e2, "pass": e1 and e2}
