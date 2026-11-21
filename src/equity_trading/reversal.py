"""Conditional reversal after large MNQ 5-minute bars (MRP 02; discussions/MNQ_Return_Prediction/003).

Event: |z_t| >= k (z = r / ex-ante sigma from the MRP 001 volatility model). Trade against the bar: enter at the
close of bar t + d, exit at the close of bar t + d + h, one contract, no overlap per configuration. Everything is
measured in ticks (0.25 points, $0.50 per contract).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import mnq_predict as P
from .fx_seasonality import bh_reject

REV_VERSION = "0.1.0"
TICK = 0.25
USD_PER_TICK = 0.50


@dataclass(frozen=True)
class RevConfig:
    ks: tuple = (1.5, 2.0, 2.5, 3.0, 4.0)
    delays: tuple = (0, 1)
    holds: tuple = (1, 2, 3, 6)
    sessions: tuple = ("all", "asia", "london", "ny", "post")
    commission_per_side: float = 0.50          # USD per contract per side (Lucid; Iván confirms)
    realized_cost_ticks: float = 1.6           # Iván's stated realized round trip ($0.80)
    min_trades_discovery: int = 150
    min_trades_confirm: int = 75
    q: float = 0.05
    size_bins: tuple = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, np.inf)

    @property
    def market_cost_ticks(self) -> float:
        return 1.0 + 2 * self.commission_per_side / USD_PER_TICK


def configs(rc: RevConfig = RevConfig()) -> pd.DataFrame:
    rows = [{"k": k, "d": d, "h": h, "session": s} for k, d, h, s in
            itertools.product(rc.ks, rc.delays, rc.holds, rc.sessions)]
    df = pd.DataFrame(rows)
    df["config"] = [f"k{r.k:g}_d{r.d}_h{r.h}_{r.session}" for r in df.itertuples()]
    return df


# ----------------------------------------------------------------------------- volatility for event sizes

def event_sigma(grid: pd.DataFrame, first_fit: tuple, refits: tuple, end: dt.date, cfg: P.MRPConfig) -> pd.Series:
    """Ex-ante sig2 per bar. Trade dates before refits[0] use a fit on ``first_fit`` (in sample, discovery only);
    later dates use walk-forward fits made before each refit (expanding from first_fit[0])."""
    td = grid["trade_date"].to_numpy()
    parts = []
    if first_fit is not None:
        f0 = P.fit_vol(grid, first_fit[0], first_fit[1], cfg)
        p0 = P.predict_vol(grid, f0, cfg)["sig2"]
        parts.append(p0[(td >= first_fit[0]) & (td < refits[0])])
    start = first_fit[0] if first_fit is not None else cfg.train[0]
    wf, _ = P.walk_forward_vol(grid, start, refits, end, cfg)
    parts.append(wf["sig2"])
    return pd.concat(parts).sort_index().reindex(grid.index)


# ----------------------------------------------------------------------------- events and trades

def bar_table(grid: pd.DataFrame, sig2: pd.Series, cfg: P.MRPConfig) -> pd.DataFrame:
    """Per bar: z, sign, session, eligibility, close; positional arrays used by ``trades``."""
    r = grid["ret"].to_numpy(float)
    z = r / np.sqrt(sig2.reindex(grid.index).to_numpy(float))
    elig = np.isfinite(z) & (grid["slot"].to_numpy() >= cfg.drop_first) & \
        (grid["n_bars_day"].to_numpy() >= cfg.min_day_bars)
    return pd.DataFrame({"trade_date": grid["trade_date"].to_numpy(), "slot": grid["slot"].to_numpy(),
                         "session": P.session_of(grid.index), "z": z, "ret": r, "eligible": elig,
                         "close": grid["close"].to_numpy(float)}, index=grid.index)


def _clean_run(bt: pd.DataFrame) -> np.ndarray:
    """clean[i] = number of consecutive later bars (i+1, i+2, ...) in the same CME day that have a return."""
    day = bt["trade_date"].to_numpy()
    ok = np.isfinite(bt["ret"].to_numpy())
    n = len(bt)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n - 2, -1, -1):
        if day[i + 1] == day[i] and ok[i + 1]:
            out[i] = out[i + 1] + 1
    return out


def trades(bt: pd.DataFrame, k: float, d: int, h: int, session: str, clean: np.ndarray | None = None,
           direction: int = -1) -> pd.DataFrame:
    """Non-overlapping trades of one configuration. direction -1 = against the large bar, +1 = with it."""
    clean = _clean_run(bt) if clean is None else clean
    z = bt["z"].to_numpy()
    ev = bt["eligible"].to_numpy() & (np.abs(np.nan_to_num(z)) >= k) & (clean >= d + h)
    if session != "all":
        ev &= bt["session"].to_numpy() == session
    idx = np.flatnonzero(ev)
    close = bt["close"].to_numpy()
    keep, free_from = [], -1
    for i in idx:
        if i <= free_from:
            continue
        keep.append(i)
        free_from = i + d + h - 1          # in a position until the exit bar; an event on the exit bar is allowed
    keep = np.asarray(keep, dtype=np.int64)
    if not len(keep):
        return pd.DataFrame(columns=["ts", "trade_date", "z", "side", "gross_ticks"])
    side = direction * np.sign(z[keep])
    gross = side * (close[keep + d + h] - close[keep + d]) / TICK
    return pd.DataFrame({"ts": bt.index[keep], "trade_date": bt["trade_date"].to_numpy()[keep], "z": z[keep],
                         "side": side, "gross_ticks": gross})


# ----------------------------------------------------------------------------- statistics

def day_cluster_t(x: np.ndarray, day: np.ndarray) -> tuple[float, float]:
    """Mean of x, t-statistic with errors clustered by day."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    m = x.mean()
    g = pd.Series(x - m).groupby(np.asarray(day)).sum().to_numpy()
    G = len(g)
    if G < 3:
        return float(m), float("nan")
    se = np.sqrt((g ** 2).sum() * G / (G - 1)) / n
    return float(m), float(m / se) if se > 0 else float("nan")


def summarize(tr: pd.DataFrame, cost_ticks: float, periods: dict | None = None) -> dict:
    g = tr["gross_ticks"].to_numpy(float)
    net = g - cost_ticks
    day = tr["trade_date"].to_numpy()
    m, t = day_cluster_t(net, day) if len(net) else (np.nan, np.nan)
    out = {"n": int(len(g)), "days": int(pd.Series(day).nunique()), "gross": float(np.mean(g)) if len(g) else np.nan,
           "net": m, "t": t, "p": float(stats.norm.sf(t)) if np.isfinite(t) else np.nan,
           "win_rate": float(np.mean(net > 0)) if len(g) else np.nan,
           "mean_win": float(net[net > 0].mean()) if (net > 0).any() else np.nan,
           "mean_loss": float(net[net <= 0].mean()) if (net <= 0).any() else np.nan,
           "q05": float(np.quantile(net, 0.05)) if len(g) else np.nan,
           "q01": float(np.quantile(net, 0.01)) if len(g) else np.nan,
           "worst": float(net.min()) if len(g) else np.nan,
           "max_dd": float((np.maximum.accumulate(np.cumsum(net)) - np.cumsum(net)).max()) if len(g) else np.nan,
           "total": float(net.sum())}
    if periods:
        for name, (a, b) in periods.items():
            sel = (day >= a) & (day <= b)
            out[f"net_{name}"] = float(net[sel].mean()) if sel.any() else np.nan
            out[f"n_{name}"] = int(sel.sum())
    return out


def scan(bt: pd.DataFrame, rc: RevConfig, periods: dict) -> tuple[pd.DataFrame, dict]:
    """All configurations: reversal at market and realized cost, both-sides check; gates D1-D4."""
    clean = _clean_run(bt)
    rows, store = [], {}
    for c in configs(rc).itertuples():
        tr = trades(bt, c.k, c.d, c.h, c.session, clean)
        store[c.config] = tr
        s = summarize(tr, rc.market_cost_ticks, periods)
        sr = summarize(tr, rc.realized_cost_ticks)
        side = tr["side"].to_numpy() if len(tr) else np.array([])
        g = tr["gross_ticks"].to_numpy(float) if len(tr) else np.array([])
        row = {"config": c.config, "k": c.k, "d": c.d, "h": c.h, "session": c.session, **s,
               "net_realized": sr["net"], "t_realized": sr["t"],
               "gross_long": float(g[side > 0].mean()) if (side > 0).any() else np.nan,     # after down bars
               "gross_short": float(g[side < 0].mean()) if (side < 0).any() else np.nan}    # after up bars
        rows.append(row)
    df = pd.DataFrame(rows)
    rej, qv = bh_reject(df["p"].to_numpy(), rc.q)
    df["q_bh"] = qv
    df["D1"] = df["n"] >= rc.min_trades_discovery
    df["D2"] = (df["net"] > 0) & rej
    pcols = [f"net_{p}" for p in periods]
    df["periods_positive"] = (df[pcols] > 0).sum(axis=1)
    df["D3"] = df["periods_positive"] >= min(3, len(periods))
    df["D4"] = (df["gross_long"] > 0) & (df["gross_short"] > 0)     # not a one-sided drift
    df["candidate"] = df["D1"] & df["D2"] & df["D3"] & df["D4"]
    df["realized_only"] = (~df["candidate"]) & df["D1"] & (df["net_realized"] > 0) & df["D4"]
    return df, store


def size_curve(bt: pd.DataFrame, rc: RevConfig, d: int, h: int = 1) -> pd.DataFrame:
    """Mean reversal ticks (no overlap rule: every eligible bar) by |z| bin and session."""
    clean = _clean_run(bt)
    z = bt["z"].to_numpy()
    ok = bt["eligible"].to_numpy() & (clean >= d + h) & np.isfinite(z)
    i = np.flatnonzero(ok)
    close = bt["close"].to_numpy()
    rev = -np.sign(z[i]) * (close[i + d + h] - close[i + d]) / TICK
    df = pd.DataFrame({"absz": np.abs(z[i]), "session": bt["session"].to_numpy()[i], "rev": rev,
                       "day": bt["trade_date"].to_numpy()[i]})
    df["bin"] = pd.cut(df["absz"], list(rc.size_bins), right=False)
    df = df.dropna(subset=["bin"])
    out = []
    for keys, g in list(df.groupby("bin", observed=True)) + list(df.groupby(["bin", "session"], observed=True)):
        b, s = (keys, "all") if not isinstance(keys, tuple) else keys
        m, t = day_cluster_t(g["rev"].to_numpy(), g["day"].to_numpy())
        out.append({"bin": str(b), "session": s, "n": len(g), "mean_rev_ticks": m, "t": t})
    return pd.DataFrame(out)


def freeze(cands: pd.DataFrame, path: Path) -> str:
    cands.to_csv(path, index=False)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_ok(path: Path, sha: str) -> bool:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha


def day_block_lower(tr: pd.DataFrame, cost_ticks: float, n_boot: int = 5000, seed: int = 0) -> dict:
    daily = (tr["gross_ticks"] - cost_ticks).groupby(tr["trade_date"]).sum().to_numpy()
    return P.day_block_bootstrap(daily, n_boot=n_boot, seed=seed, block=5)
