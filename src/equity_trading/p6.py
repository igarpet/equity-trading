"""P6: Guijarro-Ordonez, Pelger & Zanotti - deep learning statistical arbitrage (data, residuals, evaluation).

Protocol: discussions/Deep_Statistical_Arbitrage/001_claude_p6_deep_stat_arb_preregistration.md
The network and its training live in p6_model.py (needs PyTorch); everything here is numpy / pandas.

Residuals: PCA with K factors on the correlation of the previous `window` sessions (complete histories only),
re-estimated daily and applied to the next session: e = sigma (z - V V' z), z = (r - mu) / sigma.
Inputs at the close of d: the cumulative residual over sessions d-L+1..d (L = 30), divided by that day's
cross-sectional s.d. of the final value. Weights are neutralised to X = sigma * V and scaled to sum|w| = 1;
they are traded at the close of d + 1 (skip 1) and earn the returns of d + 2.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

P6_VERSION = "0.1.0"


@dataclass(frozen=True)
class P6Config:
    pca_window: int = 252
    n_factors: int = 5
    lookback: int = 30
    train_days: int = 1000
    valid_days: int = 250
    refit_every: int = 125
    first_test: str = "2017-01-03"
    end: str = "2025-12-31"
    yahoo_start: str = "2005-01-01"
    splice: str = "2016-01-04"
    min_price: float = 5.0
    skip: int = 1
    seeds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    chunk_days: int = 50
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    n_placebo: int = 1000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== data
def load_yahoo_daily(tickers: list[str], start: str, end: str, batch: int = 50, verbose: bool = True) -> pd.DataFrame:
    """Adjusted daily closes from Yahoo (yfinance), cached per ticker under <data>/Yahoo/daily/."""
    import yfinance as yf
    from .data import vendor_dir
    d = vendor_dir("Yahoo") / "daily"
    d.mkdir(parents=True, exist_ok=True)
    ysym = {t: t.replace(".", "-") for t in tickers}
    todo = [t for t in tickers if not (d / f"{ysym[t]}.parquet").exists()]
    if verbose:
        print(f"Yahoo daily: {len(tickers) - len(todo)}/{len(tickers)} cached; downloading {len(todo)}", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        df = yf.download([ysym[t] for t in chunk], start=start, end=end, auto_adjust=True, progress=False,
                         group_by="column", threads=True)
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": ysym[chunk[0]]})
        for t in chunk:
            s = close[ysym[t]] if ysym[t] in close.columns else pd.Series(dtype=float)
            out = pd.DataFrame({"close": s.astype(float)})
            out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
            tmp = d / f"{ysym[t]}.parquet.tmp"; out.to_parquet(tmp); os.replace(tmp, d / f"{ysym[t]}.parquet")
        if verbose:
            print(f"  {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        time.sleep(1.0)
    cols = {}
    for t in tickers:
        p = d / f"{ysym[t]}.parquet"
        if p.exists():
            s = pd.read_parquet(p)["close"]
            if len(s):
                cols[t] = s
    return pd.DataFrame(cols).sort_index()


def splice_returns(yahoo: pd.DataFrame, alpaca: pd.DataFrame, sessions: pd.DatetimeIndex, cut: str) -> pd.DataFrame:
    """Daily simple returns on `sessions`: Yahoo before `cut` and on `cut` itself, Alpaca after."""
    cut = pd.Timestamp(cut)
    ry = yahoo.reindex(sessions).pct_change(fill_method=None)
    ra = alpaca.reindex(sessions).pct_change(fill_method=None)
    cols = sorted(set(ry.columns) | set(ra.columns))
    ry, ra = ry.reindex(columns=cols), ra.reindex(columns=cols)
    return ry.where(ry.index <= cut, ra)


def synthetic_returns(n_days: int = 1500, n_stocks: int = 120, k: int = 5, rev: float = 0.25, seed: int = 0,
                      start: str = "2012-01-02") -> pd.DataFrame:
    """Factor returns + idiosyncratic noise whose shocks partly reverse two sessions later (tradable at skip 1)."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n_days, k)) * 0.01
    L = rng.standard_normal((n_stocks, k)) * 0.8
    u = rng.standard_normal((n_days, n_stocks)) * 0.015
    e = u.copy()
    e[2:] -= rev * u[:-2]
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(F @ L.T + e, index=idx, columns=[f"S{i:03d}" for i in range(n_stocks)])


# =================================================================== residuals
def pca_residuals(R: pd.DataFrame, window: int = 252, k: int = 5, start_row: int | None = None,
                  verbose: bool = True) -> dict:
    """resid (D, N), sigma (D, N) and V (D, N, k) from the window ending at t-1, applied to row t."""
    X = R.to_numpy(float)
    D, N = X.shape
    resid = np.full((D, N), np.nan, np.float32)
    sig = np.full((D, N), np.nan, np.float32)
    V = np.zeros((D, N, k), np.float32)
    t0 = max(window, start_row or window)
    tic = time.time()
    for t in range(t0, D):
        W = X[t - window:t]
        ok = np.isfinite(W).all(0) & np.isfinite(X[t])
        if ok.sum() < 3 * k:
            continue
        Wk = W[:, ok]
        mu, sd = Wk.mean(0), Wk.std(0, ddof=1)
        good = sd > 0
        idx = np.flatnonzero(ok)[good]
        Z = (Wk[:, good] - mu[good]) / sd[good]
        C = Z.T @ Z / (window - 1)
        w, vec = np.linalg.eigh(C)
        Vk = vec[:, -k:]
        z = (X[t, idx] - mu[good]) / sd[good]
        ez = z - Vk @ (Vk.T @ z)
        resid[t, idx] = sd[good] * ez
        sig[t, idx] = sd[good]
        V[t, idx] = Vk
        if verbose and (t - t0) % 500 == 0:
            print(f"  PCA residuals {t - t0}/{D - t0} days, {time.time() - tic:.0f}s", flush=True)
    return {"resid": resid, "sigma": sig, "V": V}


def inputs(resid: np.ndarray, L: int = 30) -> np.ndarray:
    """(D, N, L) cumulative residuals over d-L+1..d (NaN if any missing), divided by the day's cross-sectional
    s.d. of the final cumulative value."""
    D, N = resid.shape
    out = np.full((D, N, L), np.nan, np.float32)
    for t in range(L - 1, D):
        win = resid[t - L + 1:t + 1]
        c = np.cumsum(win, 0).T                                    # (N, L)
        ok = np.isfinite(c).all(1)
        if ok.sum() < 10:
            continue
        s = np.std(c[ok, -1])
        if s > 0:
            out[t, ok] = c[ok] / s
    return out


def neutralise(w: np.ndarray, X: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """One day: demean w over ok, remove exposure to the columns of X (N, k), scale to sum|w| = 1."""
    out = np.zeros_like(w, dtype=float)
    if ok.sum() < X.shape[1] + 2:
        return out
    ww = w[ok] - w[ok].mean()
    Xo = X[ok].astype(float)
    beta = np.linalg.lstsq(Xo, ww, rcond=None)[0]
    ww = ww - Xo @ beta
    s = np.abs(ww).sum()
    if s > 0:
        out[ok] = ww / s
    return out


def weights_from_scores(scores: np.ndarray, sig: np.ndarray, V: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(D, N) neutralised weights from (D, N) scores, with exposures X = sigma * V of the same day."""
    W = np.zeros(scores.shape)
    for t in range(len(scores)):
        m = ok[t] & np.isfinite(scores[t]) & np.isfinite(sig[t])
        W[t] = neutralise(np.nan_to_num(scores[t]), sig[t][:, None] * V[t], m)
    return W


# =================================================================== benchmarks
def ou_scores(Xin: np.ndarray) -> np.ndarray:
    """-s from an AR(1) on each 30-day cumulative residual path (Avellaneda-Lee on the window)."""
    x0, x1 = Xin[..., :-1], Xin[..., 1:]
    m0, m1 = x0.mean(-1, keepdims=True), x1.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = ((x0 - m0) * (x1 - m1)).sum(-1) / ((x0 - m0) ** 2).sum(-1)
        a = m1[..., 0] - b * m0[..., 0]
        z = x1 - a[..., None] - b[..., None] * x0
        m = a / (1 - b)
        se = np.sqrt(z.var(-1) / (1 - b * b))
        s = (Xin[..., -1] - m) / se
    s = np.where((b > 0) & (b < 1) & np.isfinite(s), s, np.nan)
    return -s


def linear_scores(Xin: np.ndarray, k: int = 5) -> np.ndarray:
    return -(Xin[..., -1] - Xin[..., -1 - k])


# =================================================================== evaluation
def evaluate(W: np.ndarray, R: np.ndarray, C: np.ndarray, skip: int = 1, cost_mult: float = 1.0) -> pd.DataFrame:
    """Daily gross / cost / net (rows aligned with R): weights formed at d, traded at d + skip, earning d + skip + 1."""
    D = len(W)
    held = np.vstack([np.zeros((skip + 1, W.shape[1])), W[:D - skip - 1]])
    gross = np.nansum(held * np.nan_to_num(R), 1)
    traded = np.vstack([np.zeros((skip, W.shape[1])), W[:D - skip]])
    dw = np.abs(traded - np.vstack([np.zeros((1, W.shape[1])), traded[:-1]]))
    cost = np.nansum(dw * np.nan_to_num(C), 1) * cost_mult
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost})


def evaluate_rows(W: np.ndarray, d: dict, rows: np.ndarray, cost_mult: float = 1.0) -> pd.DataFrame:
    """Daily gross / cost / net for decision rows (weights W[d] earn d['ret'][d], i.e. the return of d + skip + 1;
    cost on |W[d] - W[d-1]| at d['cost'][d], the trade day). Indexed by the return date."""
    Wr = W[rows]
    prev = np.vstack([W[rows[0] - 1][None] if rows[0] > 0 else np.zeros((1, W.shape[1])), Wr[:-1]])
    gross = np.nansum(Wr * np.nan_to_num(d["ret"][rows]), 1)
    cost = np.nansum(np.abs(Wr - prev) * np.nan_to_num(d["cost"][rows]), 1) * cost_mult
    h = 2  # skip + 1 sessions from decision to return day
    idx = d["index"]
    when = [idx[min(r + h, len(idx) - 1)] for r in rows]
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost}, index=pd.DatetimeIndex(when))


def placebo_gross(W: np.ndarray, ret: np.ndarray, ok: np.ndarray, rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    """(n, len(rows)) daily gross of permuted weights: on each decision row the day's weights are permuted among
    the day's stocks; ret is aligned to decision rows (d['ret'])."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, len(rows)))
    Rn = np.nan_to_num(ret)
    for j, d in enumerate(rows):
        idx = np.flatnonzero(ok[d] & (W[d] != 0))
        if len(idx) < 2:
            continue
        perm = rng.permuted(np.tile(W[d, idx], (n, 1)), axis=1)
        out[:, j] = perm @ Rn[d, idx]
    return out


def gates(boot: dict, ranks: dict, net2: dict) -> dict:
    g1 = all(v[1] > 0 for v in boot.values())
    g2 = ranks["total_return"]["p_upper"] <= 0.05 and ranks["sharpe"]["p_upper"] <= 0.05
    g3 = net2["total_return"] > 0 and net2["sharpe"] > 0
    return {"G1_bootstrap": g1, "G2_vs_permuted_gross": g2, "G3_double_cost": g3, "pass": g1 and g2 and g3}


def series_metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    return {"total_return": float(eq.iloc[-1] - 1) if len(eq) else 0.0,
            "sharpe": float(np.sqrt(252) * r.mean() / sd) if sd > 0 else 0.0,
            "max_drawdown": float(-(eq / eq.cummax() - 1).min()) if len(eq) else 0.0, "days": int(len(r))}


def fit_schedule(index: pd.DatetimeIndex, cfg: P6Config) -> list[dict]:
    """Rolling fits: each predicts `refit_every` decision days; trains on the `train_days` decision days ending
    skip + 2 sessions before its first prediction day (targets never overlap predictions); the last
    `valid_days` of them choose the epoch."""
    first = int(np.searchsorted(index.values, pd.Timestamp(cfg.first_test).to_datetime64()))
    last = int(np.searchsorted(index.values, pd.Timestamp(cfg.end).to_datetime64(), side="right"))
    out = []
    for p0 in range(first, last, cfg.refit_every):
        p1 = min(p0 + cfg.refit_every, last)
        t1 = p0 - (cfg.skip + 1)
        t0 = t1 - cfg.train_days
        if t0 < 0:
            raise ValueError("not enough history before the first test day")
        out.append({"train": (t0, t1 - cfg.valid_days), "valid": (t1 - cfg.valid_days, t1), "predict": (p0, p1),
                    "first_day": str(index[p0].date())})
    return out


# =================================================================== dataset
def build_dataset(R: pd.DataFrame, price: pd.DataFrame | None, cost: pd.DataFrame, cfg: P6Config,
                  start_row: int | None = None, verbose: bool = True) -> dict:
    """Day-level arrays for decision rows d: x (D, N, L) inputs, X (D, N, K) exposures, ret (D, N) = return of
    d + skip + 1, cost (D, N) = one-way cost on the trade day d + skip, ok (D, N) usable."""
    pr = pca_residuals(R, cfg.pca_window, cfg.n_factors, start_row, verbose)
    x = inputs(pr["resid"], cfg.lookback)
    D, N = R.shape
    h = cfg.skip + 1
    ret = np.full((D, N), np.nan, np.float32)
    ret[:D - h] = R.to_numpy(np.float32)[h:]
    c = np.full((D, N), np.nan, np.float32)
    c[:D - cfg.skip] = cost.reindex(index=R.index, columns=R.columns).to_numpy(np.float32)[cfg.skip:]
    X = pr["sigma"][..., None] * pr["V"]
    ok = np.isfinite(x).all(-1) & np.isfinite(pr["sigma"]) & np.isfinite(ret) & np.isfinite(c)
    if price is not None:
        p = price.reindex(index=R.index, columns=R.columns).to_numpy()
        ok &= ~(np.isfinite(p) & (p < cfg.min_price))
    return {"x": x, "X": X.astype(np.float32), "ret": ret, "cost": c, "ok": ok, "resid": pr["resid"],
            "index": R.index, "columns": list(R.columns)}


def save_dataset(d: dict, path) -> None:
    np.savez(path, x=d["x"], X=d["X"], ret=d["ret"], cost=d["cost"], ok=d["ok"], resid=d["resid"],
             index=d["index"].values, columns=np.array(d["columns"]))


def load_dataset(path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {"x": z["x"], "X": z["X"], "ret": z["ret"], "cost": z["cost"], "ok": z["ok"], "resid": z["resid"],
            "index": pd.DatetimeIndex(z["index"]), "columns": list(z["columns"])}
