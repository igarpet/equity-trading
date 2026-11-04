"""S09: calibrated HAR, a two-state switching residual filter, and overnight-gap regimes (diagnostic, no trading).

Inputs are the S08 paths (s08.compute): residual increments `eps`, hedged returns `hedged`, s-scores `s0`, bar slot.
Training = decision days before TEST_START; every fitted quantity uses training days only.

A. Calibrated HAR (Corsi 2009, log form, pooled across stocks) for the next session's realized variance of
   residual increments: log RV_{d+1} = c + b1 log RV_d + b5 log RV5_d + b20 log RV20_d + u.
   V4 s-score = s0 / lambda, lambda^2 = forecast RV_{d+1} (smearing-corrected) / RV20_d, frozen for the day.
B. Two-state switching model of the s-score path within a session, pooled across stocks, Gaussian with a common
   variance so the states differ only in drift:
     state R (reverting):  ds_t = -k s_{t-1} + sigma e_t
     state S (shifted):    ds_t =             sigma e_t
   Fitted by EM (Hamilton filter / forward-backward) on training stock-sessions; increments from bar 1 on
   ("after the first bar"); filtered P(R) at each bar uses only that session's bars so far.
C. Overnight-gap regime: standardized residual gap of the first bar, g = eps_0 / sigma_bar (sigma_bar from the
   trailing 20 sessions), and the forward hedged return from the close of the first bar, faded (-sign(g)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

S09_VERSION = "0.1.0"


# ------------------------------------------------------------------ A. HAR
def daily_rv(eps: np.ndarray, day_start: np.ndarray, min_share: float = 0.9) -> np.ndarray:
    D = len(day_start) - 1
    out = np.full((D, eps.shape[1]), np.nan)
    for d in range(D):
        x = eps[day_start[d]:day_start[d + 1]].astype(float)
        ok = np.isfinite(x)
        share = ok.mean(0)
        v = np.where(ok, x * x, 0.0).sum(0)
        out[d] = np.where((share >= min_share) & (v > 0), v / np.maximum(share, 1e-9), np.nan)
    return out


def har_features(rv: np.ndarray) -> dict:
    lr = pd.DataFrame(rv)
    f5 = lr.rolling(5, min_periods=4).mean().to_numpy()
    f20 = lr.rolling(20, min_periods=16).mean().to_numpy()
    nxt = np.vstack([rv[1:], np.full((1, rv.shape[1]), np.nan)])
    return {"rv1": rv, "rv5": f5, "rv20": f20, "target": nxt}


def fit_har(F: dict, train_days: np.ndarray) -> dict:
    cols = [np.log(F[k][train_days]).ravel() for k in ("rv1", "rv5", "rv20")]
    y = np.log(F["target"][train_days]).ravel()
    X = np.column_stack([np.ones_like(y)] + cols)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid = y[ok] - X[ok] @ coef
    return {"coef": coef, "smear": float(np.mean(np.exp(resid))), "n": int(ok.sum()),
            "r2": float(1 - resid.var() / y[ok].var())}


def har_lambda(F: dict, fit: dict, clip=(0.25, 4.0)) -> np.ndarray:
    """(D, N) lambda for the day AFTER each decision day d (row d+1 holds the value used on day d+1)."""
    c = fit["coef"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.exp(c[0] + c[1] * np.log(F["rv1"]) + c[2] * np.log(F["rv5"]) + c[3] * np.log(F["rv20"])) * fit["smear"]
        lam = np.sqrt(pred / F["rv20"])
    out = np.full_like(lam, np.nan)
    out[1:] = np.clip(lam[:-1], *clip)
    return out


def apply_daily(s: np.ndarray, lam_day: np.ndarray, day_start: np.ndarray) -> np.ndarray:
    out = np.full_like(s, np.nan)
    for d in range(len(day_start) - 1):
        a, b = day_start[d], day_start[d + 1]
        out[a:b] = s[a:b] / lam_day[d][None, :]
    return out


# ------------------------------------------------------------------ B. switching filter
def _norm_pdf(x, sd):
    return np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def session_sequences(s: np.ndarray, day_start: np.ndarray, days: np.ndarray, L: int = 26,
                      clip_ds: float = 5.0, clip_s: float = 8.0):
    """Stack full-length sessions (L bars) as (days x stocks) sequences: lagged s and increments for bars 1..L-1."""
    seqs = []
    for d in days:
        a, b = day_start[d], day_start[d + 1]
        if b - a != L:
            continue
        x = s[a:b].T.astype(float)                        # (N, L)
        keep = np.isfinite(x[:, 0]) & (np.isfinite(x).mean(1) > 0.9)
        seqs.append(x[keep])
    Y = np.vstack(seqs) if seqs else np.empty((0, L))
    ylag = np.clip(Y[:, :-1], -clip_s, clip_s)
    dy = np.clip(Y[:, 1:] - Y[:, :-1], -clip_ds, clip_ds)
    return ylag, dy


def _emissions(ylag, dy, k, sd):
    m = np.isfinite(ylag) & np.isfinite(dy)
    eR = np.where(m, _norm_pdf(np.where(m, dy + k * ylag, 0.0), sd), 1.0)
    eS = np.where(m, _norm_pdf(np.where(m, dy, 0.0), sd), 1.0)
    return np.stack([eR, eS], -1), m                     # (n, T, 2)


def fit_switching(ylag: np.ndarray, dy: np.ndarray, n_iter: int = 60, tol: float = 1e-6, seed: int = 0) -> dict:
    """EM for the two-state model. States: 0 = R (reverting), 1 = S (shifted / no pull)."""
    m0 = np.isfinite(ylag) & np.isfinite(dy)
    k = 0.02
    sd = float(np.nanstd(dy[m0]))
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    pi = np.array([0.5, 0.5])
    ll_old = -np.inf
    n, T = dy.shape
    for it in range(n_iter):
        E, m = _emissions(ylag, dy, k, sd)
        alpha = np.zeros((n, T, 2)); c = np.zeros((n, T))
        a = pi[None, :] * E[:, 0]
        c[:, 0] = a.sum(1); alpha[:, 0] = a / c[:, :1]
        for t in range(1, T):
            a = (alpha[:, t - 1] @ P) * E[:, t]
            c[:, t] = a.sum(1); alpha[:, t] = a / c[:, t:t + 1]
        beta = np.ones((n, T, 2))
        for t in range(T - 2, -1, -1):
            b = (E[:, t + 1] * beta[:, t + 1]) @ P.T
            beta[:, t] = b / c[:, t + 1:t + 2]
        g = alpha * beta
        g /= g.sum(-1, keepdims=True)
        xi = (alpha[:, :-1, :, None] * P[None, None] * (E[:, 1:] * beta[:, 1:])[:, :, None, :]) / c[:, 1:, None, None]
        ll = float(np.log(c).sum())
        # M-step
        gR, gS = np.where(m, g[..., 0], 0.0), np.where(m, g[..., 1], 0.0)
        yl, d_ = np.where(m, ylag, 0.0), np.where(m, dy, 0.0)
        k = float(-(gR * d_ * yl).sum() / max((gR * yl * yl).sum(), 1e-12))
        sd = float(np.sqrt(((gR * (d_ + k * yl) ** 2) + gS * d_ ** 2).sum() / max((gR + gS).sum(), 1e-12)))
        P = xi.sum((0, 1)); P /= P.sum(1, keepdims=True)
        pi = g[:, 0].mean(0)
        if abs(ll - ll_old) < tol * abs(ll):
            break
        ll_old = ll
    stat = np.array([P[1, 0], P[0, 1]]) / (P[1, 0] + P[0, 1])
    return {"k": k, "sd": sd, "P": P, "pi": pi, "stationary": stat, "loglik": ll, "iterations": it + 1,
            "half_life_bars_R": float(np.log(2) / k) if 0 < k < 1 else np.nan}


def filter_prob_R(s: np.ndarray, day_start: np.ndarray, fit: dict, clip_ds: float = 5.0, clip_s: float = 8.0) -> np.ndarray:
    """(T, N) filtered P(state R) at each bar close, using only the session's bars so far (bar 0: stationary prior)."""
    T_, N = s.shape
    out = np.full((T_, N), np.nan, np.float32)
    P, k, sd = fit["P"], fit["k"], fit["sd"]
    prior = fit["stationary"]
    for d in range(len(day_start) - 1):
        a, b = day_start[d], day_start[d + 1]
        x = s[a:b].astype(float)
        p = np.tile(prior, (N, 1))
        out[a] = np.where(np.isfinite(x[0]), prior[0], np.nan)
        for t in range(1, b - a):
            yl = np.clip(x[t - 1], -clip_s, clip_s)
            dy = np.clip(x[t] - x[t - 1], -clip_ds, clip_ds)
            m = np.isfinite(yl) & np.isfinite(dy)
            pr = p @ P
            eR = np.where(m, _norm_pdf(np.where(m, dy + k * yl, 0.0), sd), 1.0)
            eS = np.where(m, _norm_pdf(np.where(m, dy, 0.0), sd), 1.0)
            num = np.column_stack([pr[:, 0] * eR, pr[:, 1] * eS])
            p = num / num.sum(1, keepdims=True)
            out[a + t] = np.where(np.isfinite(x[t]), p[:, 0], np.nan)
    return out


def conditional_table(s: np.ndarray, pR: np.ndarray, fwd: np.ndarray, rows: np.ndarray, s_min: float = 2.0,
                      edges=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0)) -> pd.DataFrame:
    x, p, y = s[rows].ravel(), pR[rows].ravel(), fwd[rows].ravel()
    ok = np.isfinite(x) & np.isfinite(p) & np.isfinite(y) & (np.abs(x) >= s_min)
    df = pd.DataFrame({"pR": pd.cut(p[ok], edges, include_lowest=True), "contrarian_bps": -1e4 * np.sign(x[ok]) * y[ok]})
    return df.groupby("pR", observed=True).agg(n=("contrarian_bps", "size"), contrarian_bps=("contrarian_bps", "mean"),
                                              se_bps=("contrarian_bps", lambda v: v.std() / np.sqrt(max(len(v), 1))))


def qualifying_counts(s: np.ndarray, pR: np.ndarray, rows: np.ndarray, s_min: float = 2.0, p_min: float = 0.7) -> np.ndarray:
    return ((np.abs(s[rows]) >= s_min) & (pR[rows] > p_min)).sum(1)


# ------------------------------------------------------------------ C. gap regime
def gap_table(eps: np.ndarray, rv: np.ndarray, day_start: np.ndarray, fwd: dict, days: np.ndarray,
              edges=(0, 1, 2, 3, 5, np.inf)) -> pd.DataFrame:
    """Standardized first-bar residual gap g = eps_0 / sigma_bar (sigma_bar^2 = trailing 20-session mean RV / bars)
    and the faded forward hedged return from the first bar's close, by |g| bucket."""
    rv20 = pd.DataFrame(rv).rolling(20, min_periods=16).mean().shift(1).to_numpy()
    recs = []
    for d in days:
        a, b = day_start[d], day_start[d + 1]
        sig_bar = np.sqrt(rv20[d] / (b - a))
        g = eps[a] / sig_bar
        for h, f in fwd.items():
            y = f[a]
            ok = np.isfinite(g) & np.isfinite(y)
            recs.append(pd.DataFrame({"h": h, "abs_g": np.abs(g[ok]), "fade_bps": -1e4 * np.sign(g[ok]) * y[ok]}))
    df = pd.concat(recs, ignore_index=True)
    df["bucket"] = pd.cut(df["abs_g"], edges, right=False)
    return df.groupby(["h", "bucket"], observed=True).agg(n=("fade_bps", "size"), fade_bps=("fade_bps", "mean"),
                                                         se_bps=("fade_bps", lambda v: v.std() / np.sqrt(max(len(v), 1))))
