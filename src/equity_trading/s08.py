"""S08: diagnostic of the S05 s-score (no trading).

Three s-score variants, nested, all from the same daily sector-ETF regression (S05, 20-session window):
  V0  S05 as traded: X = cumulative residual, AR(1) on X over the window, s = (X - m) / sigma_eq.
  V1  overnight gap excluded: the first bar of each session (09:30 bar, which carries the overnight move)
      contributes no residual increment, in the window fit and in the live residual. (A time-of-day
      deseasonalisation of the increments was tried first and rejected: rescaling the increments of a
      mean-reverting level and re-cumulating them distorts the reversion; see S08_README.)
  V2  V1 / lambda_t, lambda_t^2 = HAR-style variance ratio (mean of the last 1-, 5- and 20-session realized
      variance of V1 residual increments, divided by the 20-session one), using bars up to t.
  V3  V0 / lambda_t (same lambda): the volatility scaling alone, without the overnight choice.
V1 is only valid if overnight residual moves are permanent: removing increments that do revert leaves a
random walk in the level. Real data decide which (on the synthetic panel, where overnight moves revert by
construction, V1 is worse than V0).
Outputs per bar and stock (float32): residual increments, s for V0/V1/V2, plus per-day parameters and
innovation statistics. Everything uses only data up to the decision time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .s04 import Panel
from .s05 import log_returns

S08_VERSION = "0.1.1"


@dataclass(frozen=True)
class S08Config:
    window_sessions: int = 20
    min_bar_share: float = 0.95
    max_tau_frac: float = 0.5
    rv_short_sessions: int = 1
    rv_mid_sessions: int = 5
    lambda_clip: tuple = (0.25, 4.0)
    kurt_sample_per_day: int = 5          # stock-days per day whose standardized innovations are kept (t fits)
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def _ar1(X: np.ndarray, valid_in: np.ndarray, W: int, max_tau_frac: float):
    """AR(1) X_{t+1} = a + b X_t + z on each column. Returns b, m, sig_eq, z (W-1, N), valid."""
    x0, x1 = X[:-1], X[1:]
    m0, m1 = x0.mean(0), x1.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = ((x0 - m0) * (x1 - m1)).sum(0) / ((x0 - m0) ** 2).sum(0)
        a = m1 - b * m0
        z = x1 - a - b * x0
        vz = z.var(axis=0, ddof=1)
        m = a / (1 - b)
        sig = np.sqrt(vz / (1 - b * b))
        tau = -1.0 / np.log(b)
    valid = valid_in & (b > 0) & (b < 1) & np.isfinite(sig) & (sig > 0) & (tau < max_tau_frac * W)
    return b, m, sig, tau, z, valid


def _excess_kurtosis(z: np.ndarray) -> np.ndarray:
    zc = z - z.mean(0)
    v = (zc ** 2).mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (zc ** 4).mean(0) / v ** 2 - 3.0


def compute(panel: Panel, stock_cols: np.ndarray, etf_cols: np.ndarray, cfg: S08Config, verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    r = log_returns(panel)
    ds = panel.day_start
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    slot = np.zeros(T, dtype=np.int16)
    for a_, b_ in zip(ds[:-1], ds[1:]):
        slot[a_:b_] = np.arange(b_ - a_)
    f32 = np.float32
    eps_all = np.full((T, N), np.nan, f32)          # residual increment with the day's frozen alpha, beta
    hed_all = np.full((T, N), np.nan, f32)          # beta-hedged return r - beta r_etf (no alpha): what a position earns
    eds_all = np.full((T, N), np.nan, f32)          # V1 increment (overnight bar zeroed)
    S = {k: np.full((T, N), np.nan, f32) for k in ("s0", "s1", "s2", "s3")}
    m_over_sig0 = np.full((T, N), np.nan, f32)      # -m / sigma_eq component of s0 (constant within a day)
    day = {k: np.full((D, N), np.nan, f32) for k in ("tau0", "tau1", "kurt0", "kurt1", "beta")}
    day_valid = np.zeros((D, N), bool)
    z_samples = []
    first = cfg.window_sessions
    for e in range(first - 1, D - 1):
        a, b = ds[e - cfg.window_sessions + 1], ds[e + 1]
        W = b - a
        R, E = r[a:b][:, stock_cols], r[a:b][:, etf_cols]
        ok = np.isfinite(R) & np.isfinite(E)
        share = ok.mean(0)
        R0, E0 = np.where(ok, R, 0.0), np.where(ok, E, 0.0)
        n = ok.sum(0).clip(min=1)
        rm, em = R0.sum(0) / n, E0.sum(0) / n
        dr, de = np.where(ok, R0 - rm, 0.0), np.where(ok, E0 - em, 0.0)
        vx = (de * de).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (dr * de).sum(0) / vx
            alpha = rm - beta * em
        eps = np.where(ok, R0 - alpha - beta * E0, 0.0)
        base_ok = (share >= cfg.min_bar_share) & (vx > 0) & np.isfinite(beta)
        # V0
        b0, m0, sg0, tau0, z0, v0 = _ar1(np.cumsum(eps, 0), base_ok, W, cfg.max_tau_frac)
        # V1: overnight bar contributes nothing
        sw = slot[a:b]
        eds = np.where((sw == 0)[:, None], 0.0, eps)
        b1, m1, sg1, tau1, z1, v1 = _ar1(np.cumsum(eds, 0), base_ok, W, cfg.max_tau_frac)
        valid = v0 & v1
        mc0 = m0 - np.nanmean(m0[valid]) if valid.any() else m0
        mc1 = m1 - np.nanmean(m1[valid]) if valid.any() else m1
        # window history of V1 increments, for the variance ratio
        intr = ok & (sw != 0)[:, None]
        if np.all(np.isnan(eds_all[a:b, 0])):
            eds_all[a:b] = np.where(intr, eds, np.nan)
        rv20 = np.nanmean(np.where(intr, eds, np.nan) ** 2, axis=0)
        # next day t = e + 1
        t0, t1 = ds[e + 1], ds[e + 2]
        Rt, Et = r[t0:t1][:, stock_cols], r[t0:t1][:, etf_cols]
        epst = Rt - alpha - beta * Et
        epst0 = np.where(np.isfinite(epst), epst, 0.0)
        st = slot[t0:t1]
        edst = np.where((st == 0)[:, None], np.nan, epst)          # overnight bar: no increment, not in RV
        edst0 = np.where(np.isfinite(edst), edst, 0.0)
        X0, X1 = np.cumsum(epst0, 0), np.cumsum(edst0, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s0 = (X0 - mc0) / sg0
            s1 = (X1 - mc1) / sg1
        eps_all[t0:t1] = epst
        hed_all[t0:t1] = Rt - beta * Et
        eds_all[t0:t1] = edst
        # HAR-style variance ratio from bars up to and including each bar of day t
        bpd = int(np.median(np.diff(ds)))
        hist = eds_all[max(0, t1 - (cfg.rv_mid_sessions + 1) * bpd - 1):t1].astype(float)
        hsq = np.where(np.isfinite(hist), hist ** 2, 0.0)
        hcnt = np.isfinite(hist).astype(float)
        csq, ccnt = np.cumsum(hsq, 0), np.cumsum(hcnt, 0)
        off = len(hist) - (t1 - t0)
        lam2 = np.full((t1 - t0, N), np.nan)
        for j in range(t1 - t0):
            i = off + j
            def mean_last(k):
                lo = i - k
                s_ = csq[i] - (csq[lo] if lo >= 0 else 0.0)
                c_ = ccnt[i] - (ccnt[lo] if lo >= 0 else 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return s_ / c_
            rv1 = mean_last(cfg.rv_short_sessions * bpd)
            rv5 = mean_last(cfg.rv_mid_sessions * bpd)
            with np.errstate(invalid="ignore", divide="ignore"):
                lam2[j] = (rv1 + rv5 + rv20) / (3 * rv20)
        lam = np.clip(np.sqrt(lam2), *cfg.lambda_clip)
        s2 = s1 / lam
        vmask = valid[None, :]
        S["s0"][t0:t1] = np.where(vmask, s0, np.nan)
        S["s1"][t0:t1] = np.where(vmask, s1, np.nan)
        S["s2"][t0:t1] = np.where(vmask, s2, np.nan)
        S["s3"][t0:t1] = np.where(vmask, s0 / lam, np.nan)
        m_over_sig0[t0:t1] = np.where(vmask, np.broadcast_to(-mc0 / sg0, s0.shape), np.nan)
        day_valid[e + 1] = valid
        day["tau0"][e + 1], day["tau1"][e + 1] = tau0, tau1
        day["kurt0"][e + 1] = np.where(valid, _excess_kurtosis(z0), np.nan)
        day["kurt1"][e + 1] = np.where(valid, _excess_kurtosis(z1), np.nan)
        day["beta"][e + 1] = beta
        vi = np.where(valid)[0]
        if len(vi):
            for j in rng.choice(vi, size=min(cfg.kurt_sample_per_day, len(vi)), replace=False):
                z = z1[:, j]
                z_samples.append((e + 1, int(j), ((z - z.mean()) / z.std(ddof=1)).astype(f32)))
        if verbose and (e - first) % 250 == 0:
            print(f"  day {e + 1}/{D}", flush=True)
    return {"eps": eps_all, "hedged": hed_all, **S, "m_over_sig0": m_over_sig0, "slot": slot, "day_valid": day_valid, **day,
            "z_samples": z_samples}


# ------------------------------------------------------------------ summaries
def exceedance(s: np.ndarray, levels=(1.25, 2.0, 3.0, 4.0)) -> dict:
    a = np.abs(s[np.isfinite(s)])
    return {f">{l:g}": float(np.mean(a > l)) for l in levels}


def normal_exceedance(levels=(1.25, 2.0, 3.0, 4.0)) -> dict:
    from scipy.stats import norm
    return {f">{l:g}": float(2 * norm.sf(l)) for l in levels}


def forward_sum(eps: np.ndarray, h: int) -> np.ndarray:
    """Sum of residual increments over the next h bars (t+1..t+h); NaN where any is missing."""
    x = np.where(np.isfinite(eps), eps, 0.0).astype(np.float64)
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    nan = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(~np.isfinite(eps), 0)])
    T = len(eps)
    out = np.full(eps.shape, np.nan, np.float32)
    idx = np.arange(T - h)
    s = c[idx + 1 + h] - c[idx + 1]
    bad = (nan[idx + 1 + h] - nan[idx + 1]) > 0
    out[idx] = np.where(bad, np.nan, s)
    return out


def rank_ic(signal: np.ndarray, fwd: np.ndarray, rows: np.ndarray, min_n: int = 50) -> np.ndarray:
    """Cross-sectional Spearman correlation of signal and fwd on each sampled row."""
    out = np.full(len(rows), np.nan)
    for k, t in enumerate(rows):
        x, y = signal[t], fwd[t]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_n:
            continue
        rx = pd.Series(x[ok]).rank().to_numpy()
        ry = pd.Series(y[ok]).rank().to_numpy()
        out[k] = np.corrcoef(rx, ry)[0, 1]
    return out


def bucket_returns(s: np.ndarray, fwd: np.ndarray, rows: np.ndarray,
                   edges=(-np.inf, -4, -3, -2, -1.25, -0.5, 0.5, 1.25, 2, 3, 4, np.inf)) -> pd.DataFrame:
    """Mean forward residual return (bps), signed contrarian (-sign(s) x fwd), by s bucket."""
    x, y = s[rows].ravel(), fwd[rows].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    b = pd.cut(x, edges)
    df = pd.DataFrame({"bucket": b, "fwd_bps": 1e4 * y, "contrarian_bps": -1e4 * np.sign(x) * y})
    return df.groupby("bucket", observed=True).agg(n=("fwd_bps", "size"), mean_fwd_bps=("fwd_bps", "mean"),
                                                  contrarian_bps=("contrarian_bps", "mean"))


def ou_hit_zero_prob(s0, stop: float = 4.0) -> np.ndarray:
    """OU (stationary, z units): P(reach 0 before |s| = stop | |s0|), scale-function formula."""
    from scipy.special import erfi
    Sf = lambda z: np.sqrt(np.pi / 2) * erfi(np.asarray(z, float) / np.sqrt(2))  # noqa: E731
    z = np.clip(np.abs(np.asarray(s0, float)), 0.0, stop)
    return (Sf(stop) - Sf(z)) / Sf(stop)


def within_day_hits(s: np.ndarray, day_start: np.ndarray, rows: np.ndarray, stop: float = 4.0) -> pd.DataFrame:
    """For sampled bars: does s reach 0 (sign change or 0) before |s| >= stop, later the same session?
    Returns per observation: |s0|, outcome (1 hit zero, 0 hit stop, NaN unresolved by the close)."""
    day_of = np.searchsorted(day_start, rows, side="right") - 1
    recs = []
    for t, d in zip(rows, day_of):
        end = day_start[d + 1]
        if t + 1 >= end:
            continue
        x0 = s[t]
        path = s[t + 1:end]
        for j in np.where(np.isfinite(x0) & (np.abs(x0) >= 1.25))[0]:
            p = path[:, j] * np.sign(x0[j])
            p = p[np.isfinite(p)]
            hz = np.flatnonzero(p <= 0)
            hs = np.flatnonzero(p >= stop)
            if len(hz) == 0 and len(hs) == 0:
                out = np.nan
            elif len(hs) == 0 or (len(hz) and hz[0] < hs[0]):
                out = 1.0
            else:
                out = 0.0
            recs.append((abs(float(x0[j])), out))
    return pd.DataFrame(recs, columns=["abs_s0", "hit_zero"])
