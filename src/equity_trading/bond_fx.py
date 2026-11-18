"""Bond-FX pairs on 5-minute bars (branch Bond_FX_Pairs, BFX 001).

Four pairings of an FX pair with the government bonds of its two currencies (Dukascopy CFDs):

=========  ===============================  ==========================================
pairing    bonds                            economic link
=========  ===============================  ==========================================
EURUSD     Bund, US T-Bond                  EUR-USD long-rate differential
GBPUSD     Gilt, US T-Bond                  GBP-USD long-rate differential
USDJPY     US T-Bond                        US yields (no JGB CFD available)
EURGBP     Bund, Gilt                       EUR-GBP long-rate differential
=========  ===============================  ==========================================

Three methods, each tested on the same data and windows as BDS 001 (H1 2018-2021, H2 2022-2025, FULL):

* ``leadlag``  bonds move first: after a large bond-implied FX move over the last L bars, hold the FX pair
  in that direction for H bars. FX leg only.
* ``residual`` Avellaneda-Lee style: rolling regression of FX returns on bond returns, cumulative residual,
  rolling AR(1)/OU fit, s-score; fade |s| > s_open until |s| < s_close or a maximum holding time. FX leg only.
* ``coint``    levels: each day an Engle-Granger regression of log FX on log bond prices over the previous
  20 or 60 days (hourly samples), traded only if the residual is cointegrated (p < 0.05); fade |z| > z_open
  until the spread crosses its mean, |z| reaches 4 or 5 days pass. Both legs.

Bond prices are made continuous across the quarterly contract switches (BDS 002 signature rule), days with
1-minute price spikes and days with missing files are excluded. A 5-minute bar is used only if every leg was
quoted in it. Costs: FX at Dukascopy's quoted bid/ask + 0.70 bp commission; bond legs one futures tick +
0.70 bp per unit of hedge ratio. 2026 is sealed.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import bond_seasonality as B
from . import fx_seasonality as F

BFX_VERSION = "0.1.1"   # 0.1.1: zero/negative quotes treated as missing; frames drop non-finite rows
PAIRINGS = {"EURUSD": ("BUNDTREUR", "USTBONDTRUSD"),
            "GBPUSD": ("UKGILTTRGBP", "USTBONDTRUSD"),
            "USDJPY": ("USTBONDTRUSD",),
            "EURGBP": ("BUNDTREUR", "UKGILTTRGBP")}
WINDOWS = B.WINDOWS
METHODS = ("leadlag", "residual", "coint")
BOND_SHORT = {s: v["short"] for s, v in B.SYMBOLS.items()}


@dataclass(frozen=True)
class BFXConfig:
    """Pre-registered specification (discussions/Bond_FX_Pairs/001_claude_bfx01_preregistration.md)."""
    bar_minutes: int = 5
    min_file_share: float = 0.95
    # lead-lag
    ll_beta_days: int = 20
    ll_L: tuple = (1, 3, 6)
    ll_H: tuple = (1, 3, 6, 12)
    ll_k: tuple = (1.5, 2.5)
    # residual reversion
    rr_W: tuple = (1, 5)
    rr_open: tuple = (1.5, 2.0, 2.5)
    rr_close: float = 0.5
    rr_hold: tuple = (12, 48)
    # cointegration of levels
    ci_form: tuple = (20, 60)
    ci_open: tuple = (1.5, 2.0, 2.5)
    ci_stop: float = 4.0
    ci_hold_days: int = 5
    ci_p: float = 0.05
    ci_sample: int = 12                 # formation uses every 12th bar (hourly)
    ci_adf_lags: int = 2                # Engle-Granger ADF with a fixed 2 lags (hourly samples)
    # costs and gates
    commission_bps_rt: float = 0.70
    min_trades: int = 100
    fdr_q: float = 0.05
    min_year_days: int = 120
    year_share_d: float = 0.75
    year_share_k: float = 0.60
    nw_lags: int = 5
    n_placebo: int = 5
    placebo_min_shift_days: int = 40
    conf_fdr_q: float = 0.05
    alpha: float = 0.025
    blocks: tuple = (10, 20)
    n_boot: int = 10_000
    seed: int = 42
    bds: B.BDSConfig = field(default_factory=B.BDSConfig)


def guard(start: dt.date, end: dt.date) -> None:
    B.guard(start, end)


# ----------------------------------------------------------------------------- data
def _bars(m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Bar-end-labelled bars: last bid/ask (forward-filled quotes) and whether any quote fell in the bar."""
    key = m.index.floor(f"{minutes}min")
    g = m.groupby(key)
    out = pd.DataFrame({"bid": g["bid"].last(), "ask": g["ask"].last(), "fresh": g["tick"].any()})
    out.index = out.index + pd.Timedelta(minutes=minutes)
    return out


def _missing_days(symbol: str, start: dt.date, end: dt.date) -> set:
    present = {dt.date.fromisoformat(f.stem.split("_")[-1]) for f in F.day_files(symbol, start, end)}
    first = min(present) if present else start
    return {first + dt.timedelta(n) for n in range((end - first).days + 1)
            if (first + dt.timedelta(n)).weekday() < 5 and (first + dt.timedelta(n)) not in present}


@dataclass
class Leg:
    symbol: str
    bars: pd.DataFrame          # bid, ask, fresh, mid, lmid (continuous for bonds), excluded
    rolls: pd.DataFrame
    spikes: pd.DataFrame
    missing: list
    file_share: float
    bad_quotes: int = 0         # quoted minutes with a zero / negative / non-finite price (treated as missing)


def load_leg(symbol: str, start: dt.date, end: dt.date, cfg: BFXConfig) -> Leg:
    guard(start, end)
    m = F.load_minutes(symbol, start, end, "confirmation").copy()
    bad = ~((m["bid"] > 0) & (m["ask"] > 0) & np.isfinite(m["bid"]) & np.isfinite(m["ask"]))
    n_bad = int((bad & m["tick"]).sum())
    if bad.any():                                        # a zero price is a missing quote, not a price
        m.loc[bad, ["bid", "ask"]] = np.nan
        m[["bid", "ask"]] = m[["bid", "ask"]].ffill()
        m["tick"] = m["tick"] & ~bad
    is_bond = symbol in B.SYMBOLS
    spikes = B.screen_spikes(m, cfg.bds)
    rolls = B.detect_rolls(m, start, end, cfg.bds, symbol) if is_bond else pd.DataFrame()
    lmid = np.log(0.5 * (m["bid"].to_numpy() + m["ask"].to_numpy()))
    if is_bond and len(rolls):
        adj = np.zeros(len(m))
        idx = m.index
        for _, r in rolls.iterrows():
            adj[idx >= r["after"]] += r["gap_bps"] / 1e4          # remove the switch jump from all later prices
        lmid = lmid - adj
    m = m.assign(lmid=lmid)
    bars = _bars(m, cfg.bar_minutes)
    key = m.index.floor(f"{cfg.bar_minutes}min")
    lm = m.groupby(key)["lmid"].last()
    lm.index = lm.index + pd.Timedelta(minutes=cfg.bar_minutes)
    bars["lmid"] = lm
    bars["mid"] = 0.5 * (bars["bid"] + bars["ask"])
    missing = sorted(_missing_days(symbol, start, end))
    bad = set(missing) | (set(spikes["ts"].dt.date) if len(spikes) else set())
    day = (bars.index - pd.Timedelta(minutes=1)).date              # UTC day of the bar's minutes
    bars["excluded"] = pd.Series(day, index=bars.index).isin(bad).to_numpy()
    exp = sum(1 for n in range((end - start).days + 1) if (start + dt.timedelta(n)).weekday() < 5)
    share = 1 - len([d for d in missing if start <= d <= end]) / max(exp, 1)
    return Leg(symbol, bars, rolls, spikes, missing, share, n_bad)


@dataclass
class PairFrame:
    pairing: str
    bonds: tuple
    window: str
    t: pd.DatetimeIndex        # valid bar end times (UTC)
    y: np.ndarray              # log FX mid
    bid: np.ndarray
    ask: np.ndarray
    X: np.ndarray              # (n, k) continuous log bond mids
    bmid: np.ndarray           # (n, k) actual bond mids (for tick costs)
    seg: np.ndarray            # segment id: consecutive bars 5 minutes apart
    day: np.ndarray            # UTC date (datetime64[D]) of each bar
    legs: dict
    bars_per_day: float

    @property
    def n(self) -> int:
        return len(self.y)

    def returns(self, X: np.ndarray | None = None):
        X = self.X if X is None else X
        same = np.r_[False, self.seg[1:] == self.seg[:-1]]
        ry = np.where(same, np.r_[np.nan, np.diff(self.y)], np.nan)
        rX = np.where(same[:, None], np.vstack([np.full((1, X.shape[1]), np.nan), np.diff(X, axis=0)]), np.nan)
        return ry, rX

    @property
    def years(self) -> np.ndarray:
        return self.day.astype("datetime64[Y]").astype(int) + 1970


def frame(pairing: str, window: str, cfg: BFXConfig, legs: dict | None = None) -> PairFrame:
    start, end = WINDOWS[window]
    guard(start, end)
    bonds = PAIRINGS[pairing]
    legs = legs if legs is not None else {}
    for s in (pairing, *bonds):
        if s not in legs:
            legs[s] = load_leg(s, start, end, cfg)
    fx = legs[pairing].bars
    df = fx[["bid", "ask", "lmid", "fresh", "excluded"]].rename(columns={"lmid": "y"})
    ok = df["fresh"] & ~df["excluded"]
    for j, b in enumerate(bonds):
        bb = legs[b].bars.reindex(df.index)
        df[f"x{j}"], df[f"m{j}"] = bb["lmid"], bb["mid"]
        ok &= bb["fresh"].fillna(False).astype(bool) & ~bb["excluded"].fillna(True).astype(bool)
    d = df.index.tz_convert("UTC")
    ok &= np.asarray(d.weekday < 5) | np.asarray((d.weekday == 6))           # Sunday-evening FX/UST bars allowed
    ok &= np.asarray((d - pd.Timedelta(minutes=1)).date >= start) & np.asarray((d - pd.Timedelta(minutes=1)).date <= end)
    v = df[ok.to_numpy()].replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "bid", "ask"] + [f"x{j}" for j in range(len(bonds))])
    t = v.index
    step = np.r_[np.inf, np.diff(t.to_numpy()).astype("timedelta64[m]").astype(np.int64)]
    seg = np.cumsum(step != cfg.bar_minutes)
    day = np.asarray((t - pd.Timedelta(minutes=1)).date, dtype="datetime64[D]")
    bpd = float(pd.Series(day).value_counts().median()) if len(day) else 0.0
    X = v[[f"x{j}" for j in range(len(bonds))]].to_numpy()
    M = v[[f"m{j}" for j in range(len(bonds))]].to_numpy()
    return PairFrame(pairing, bonds, window, t, v["y"].to_numpy(), v["bid"].to_numpy(), v["ask"].to_numpy(),
                     X, M, seg, day, {s: legs[s] for s in (pairing, *bonds)}, bpd)


def monthly_diagnostics(fr: PairFrame) -> pd.DataFrame:
    """Per month: valid bars, days, bars with a usable 5-minute return (previous bar 5 minutes earlier), and the
    share of those returns where each bond did not move (stale prices marked as quoted show up here)."""
    ry, rX = fr.returns()
    ok = np.isfinite(ry) & np.isfinite(rX).all(axis=1)
    mon = pd.PeriodIndex(pd.DatetimeIndex(fr.day.astype("datetime64[ns]")), freq="M")
    df = pd.DataFrame({"month": mon, "day": fr.day, "ret": ok})
    for j, b in enumerate(fr.bonds):
        df[f"{BOND_SHORT[b]} zero"] = np.where(ok, rX[:, j] == 0, np.nan)
    df["FX zero"] = np.where(ok, ry == 0, np.nan)
    g = df.groupby("month")
    out = pd.DataFrame({"bars": g.size(), "days": g["day"].nunique(), "returns": g["ret"].sum()})
    for c in [c for c in df.columns if c.endswith(" zero")]:
        out[f"share {c}"] = g[c].mean()
    return out


def coverage_table(fr: PairFrame) -> pd.DataFrame:
    s = pd.Series(1, index=pd.Index(fr.day)).groupby(fr.years).agg(["size"])
    s["days"] = pd.Series(fr.day).groupby(fr.years).nunique().to_numpy()
    s["bars_per_day"] = s["size"] / s["days"]
    return s.rename(columns={"size": "bars"})


# ----------------------------------------------------------------------------- rolling helpers
def _csum(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0)
    return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])


def _trail(c: np.ndarray, W: int, upto_excl: bool) -> np.ndarray:
    """Sum over the trailing W rows ending at i-1 (upto_excl) or at i, from a cumsum with a leading zero."""
    n = len(c) - 1
    i = np.arange(n)
    hi = i if upto_excl else i + 1
    lo = np.maximum(hi - W, 0)
    return c[hi] - c[lo]


def rolling_beta(ry: np.ndarray, rX: np.ndarray, W: int) -> np.ndarray:
    """No-intercept OLS of FX returns on bond returns over the trailing W bars ending at i-1 (no look-ahead)."""
    ok = np.isfinite(ry) & np.isfinite(rX).all(axis=1)
    y = np.where(ok, ry, 0.0)
    X = np.where(ok[:, None], rX, 0.0)
    k = X.shape[1]
    XX = _trail(_csum(X[:, :, None] * X[:, None, :]), W, True)
    Xy = _trail(_csum(X * y[:, None]), W, True)
    cnt = _trail(_csum(ok.astype(float)), W, True)
    beta = np.full((len(y), k), np.nan)
    good = cnt >= max(30, W // 4)
    if k == 1:
        den = XX[:, 0, 0]
        g = good & (den > 0)
        beta[g, 0] = Xy[g, 0] / den[g]
    else:
        det = np.linalg.det(XX)
        g = good & (np.abs(det) > 1e-30)
        beta[g] = np.linalg.solve(XX[g], Xy[g][:, :, None])[:, :, 0]
    return beta


def rolling_std(x: np.ndarray, W: int) -> np.ndarray:
    ok = np.isfinite(x)
    a = np.where(ok, x, 0.0)
    n = _trail(_csum(ok.astype(float)), W, True)
    s1 = _trail(_csum(a), W, True)
    s2 = _trail(_csum(a * a), W, True)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    return np.where(n >= 30, np.sqrt(np.maximum(var, 0)), np.nan)


def lagdiff(a: np.ndarray, seg: np.ndarray, L: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    if L >= len(a):
        return out
    same = seg[L:] == seg[:-L]
    d = a[L:] - a[:-L]
    out[L:] = np.where(same if a.ndim == 1 else same[:, None], d, np.nan)
    return out


def seg_last(seg: np.ndarray) -> np.ndarray:
    """Index of the last row of each row's segment."""
    n = len(seg)
    end = np.r_[np.flatnonzero(seg[1:] != seg[:-1]), n - 1]
    return end[np.searchsorted(end, np.arange(n))]


# ----------------------------------------------------------------------------- trades
TRADE_COLS = ["entry", "exit", "dir", "entry_time", "exit_time", "exit_day", "gross_bps", "net_bps", "net2_bps", "mid_bps",
              "hedged_mid_bps"]


def _fx_trade(fr: PairFrame, i: int, e: int, d: int, cfg: BFXConfig):
    if d > 0:
        gross = math.log(fr.bid[e] / fr.ask[i]) * 1e4
    else:
        gross = math.log(fr.bid[i] / fr.ask[e]) * 1e4
    return gross, gross - cfg.commission_bps_rt, gross - 2 * cfg.commission_bps_rt, d * (fr.y[e] - fr.y[i]) * 1e4


def _greedy(entries: np.ndarray, exits: np.ndarray) -> list:
    """Non-overlapping trades: take an entry only after the previous trade has exited."""
    out, last = [], -1
    for i, e in zip(entries, exits):
        if i > last and e > i:
            out.append((int(i), int(e)))
            last = e
    return out


def _trades_frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRADE_COLS)


def leadlag_signal(fr: PairFrame, cfg: BFXConfig, L: int, X: np.ndarray | None = None):
    """Bond-implied FX move over the last L bars (beta from the trailing ll_beta_days, no look-ahead), its z-score
    against its own trailing ll_beta_days standard deviation, and the FX move over the same L bars."""
    X = fr.X if X is None else X
    ry, rX = fr.returns(X)
    W = int(cfg.ll_beta_days * fr.bars_per_day)
    beta = rolling_beta(ry, rX, W)
    dX = lagdiff(X, fr.seg, L)
    implied = np.nansum(beta * dX, axis=1)
    implied[~(np.isfinite(beta).all(axis=1) & np.isfinite(dX).all(axis=1))] = np.nan
    z = implied / rolling_std(implied, W)
    return implied, z, lagdiff(fr.y, fr.seg, L), beta


def run_leadlag(fr: PairFrame, cfg: BFXConfig, L: int, H: int, k: float, X: np.ndarray | None = None, sig=None) -> pd.DataFrame:
    implied, z, _, beta = sig if sig is not None else leadlag_signal(fr, cfg, L, X)
    last = seg_last(fr.seg)
    ent = np.flatnonzero(np.abs(z) > k)
    ex = np.minimum(ent + H, last[ent])
    rows = []
    for i, e in _greedy(ent, ex):
        d = int(np.sign(implied[i]))
        g, n, n2, mid = _fx_trade(fr, i, e, d, cfg)
        rows.append((i, e, d, fr.t[i], fr.t[e], fr.day[e], g, n, n2, mid, np.nan))
    return _trades_frame(rows)


def residual_signal(fr: PairFrame, cfg: BFXConfig, W_days: int, X: np.ndarray | None = None):
    """Avellaneda-Lee s-score: residual of FX returns on bond returns (beta over the trailing window, ending at
    the previous bar), cumulated; AR(1) fitted on the cumulated residual over the trailing window (including the
    current bar); s = (X - m) / sigma_eq, NaN unless 0 < b and the half-life is at most half the window."""
    X = fr.X if X is None else X
    ry, rX = fr.returns(X)
    W = int(W_days * fr.bars_per_day)
    beta = rolling_beta(ry, rX, W)
    eps = ry - np.nansum(beta * rX, axis=1)
    eps[~(np.isfinite(ry) & np.isfinite(beta).all(axis=1) & np.isfinite(rX).all(axis=1))] = np.nan
    Xc = np.cumsum(np.nan_to_num(eps, nan=0.0))
    x, yv = np.r_[np.nan, Xc[:-1]], Xc
    ok = np.isfinite(x)
    x0, y0 = np.where(ok, x, 0.0), np.where(ok, yv, 0.0)
    n = _trail(_csum(ok.astype(float)), W, False)
    Sx, Sy = _trail(_csum(x0), W, False), _trail(_csum(y0), W, False)
    Sxx, Sxy, Syy = _trail(_csum(x0 * x0), W, False), _trail(_csum(x0 * y0), W, False), _trail(_csum(y0 * y0), W, False)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx)
        a = (Sy - b * Sx) / n
        sse = Syy - 2 * a * Sy - 2 * b * Sxy + n * a * a + 2 * a * b * Sx + b * b * Sxx
        var = sse / n
        m = a / (1 - b)
        sig = np.sqrt(np.maximum(var, 0) / (1 - b * b))
        s = (Xc - m) / sig
    bmax = 2.0 ** (-2.0 / max(W, 2))                     # half-life <= W / 2
    s[~((n >= max(30, W // 4)) & (b > 0) & (b < bmax) & (sig > 0))] = np.nan
    return s, beta


def run_residual(fr: PairFrame, cfg: BFXConfig, W_days: int, s_open: float, hold: int, X: np.ndarray | None = None,
                 sig=None) -> pd.DataFrame:
    s, beta = sig if sig is not None else residual_signal(fr, cfg, W_days, X)
    X = fr.X if X is None else X
    last = seg_last(fr.seg)
    ent = np.flatnonzero(np.abs(s) > s_open)
    rows, lastx = [], -1
    for i in ent:
        if i <= lastx:
            continue
        d = -int(np.sign(s[i]))                          # fade: s high -> FX rich -> short
        stop = min(i + hold, last[i])
        if stop <= i:
            continue
        w = s[i + 1:stop + 1]
        hit = np.flatnonzero((d < 0) & (w < cfg.rr_close) | (d > 0) & (w > -cfg.rr_close))
        e = i + 1 + int(hit[0]) if len(hit) else stop
        g, nn, n2, mid = _fx_trade(fr, i, e, d, cfg)
        b = beta[i] if np.isfinite(beta[i]).all() else np.zeros(X.shape[1])
        hedged = d * ((fr.y[e] - fr.y[i]) - float(b @ (X[e] - X[i]))) * 1e4
        rows.append((i, e, d, fr.t[i], fr.t[e], fr.day[e], g, nn, n2, mid, hedged))
        lastx = e
    return _trades_frame(rows)


def coint_formations(fr: PairFrame, cfg: BFXConfig, form_days: int, X: np.ndarray | None = None) -> pd.DataFrame:
    """For each trading day: Engle-Granger regression of log FX on log bonds (with constant) over the previous
    form_days days, sampled every ci_sample bars; residual sd and statsmodels coint p-value."""
    from statsmodels.tsa.stattools import coint
    X = fr.X if X is None else X
    days, first = np.unique(fr.day, return_index=True)
    bounds = np.r_[first, fr.n]
    rows = []
    for j in range(form_days, len(days)):
        lo, hi = bounds[j - form_days], bounds[j]
        idx = np.arange(lo, hi, cfg.ci_sample)
        if len(idx) < 50:
            continue
        yy, XX = fr.y[idx], X[idx]
        A = np.column_stack([np.ones(len(idx)), XX])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        u = yy - A @ coef
        try:
            p = float(coint(yy, XX, trend="c", maxlag=cfg.ci_adf_lags, autolag=None)[1])
        except Exception:
            p = np.nan
        rows.append({"day": days[j], "lo": int(bounds[j]), "hi": int(bounds[j + 1]), "alpha": coef[0],
                     **{f"beta{q}": coef[1 + q] for q in range(XX.shape[1])}, "sd": float(u.std(ddof=1)), "p": p})
    return pd.DataFrame(rows)


def run_coint(fr: PairFrame, cfg: BFXConfig, form: pd.DataFrame, z_open: float, X: np.ndarray | None = None) -> pd.DataFrame:
    X = fr.X if X is None else X
    k = X.shape[1]
    hold = int(cfg.ci_hold_days * fr.bars_per_day)
    rows, lastx = [], -1
    for _, f in form.iterrows():
        if not (f["p"] < cfg.ci_p) or f["sd"] <= 0:
            continue
        beta = np.array([f[f"beta{q}"] for q in range(k)])
        lo, hi = int(f["lo"]), int(f["hi"])
        z = (fr.y[lo:hi] - f["alpha"] - X[lo:hi] @ beta) / f["sd"]
        for off in np.flatnonzero((np.abs(z) > z_open) & (np.abs(z) < cfg.ci_stop)):
            i = lo + int(off)
            if i <= lastx:
                continue
            d = -int(np.sign(z[off]))
            stop = min(i + hold, fr.n - 1)
            if stop <= i:
                break
            zz = (fr.y[i + 1:stop + 1] - f["alpha"] - X[i + 1:stop + 1] @ beta) / f["sd"]
            hit = np.flatnonzero((d * zz >= 0) | (np.abs(zz) >= cfg.ci_stop))
            e = i + 1 + int(hit[0]) if len(hit) else stop
            g, nn, n2, mid = _fx_trade(fr, i, e, d, cfg)
            bond_leg = -d * float(beta @ (X[e] - X[i])) * 1e4
            tick = float(np.sum(np.abs(beta) * np.array([B.tick_bps(b, fr.bmid[i, q]) for q, b in enumerate(fr.bonds)])))
            comm_b = float(np.sum(np.abs(beta))) * cfg.commission_bps_rt
            rows.append((i, e, d, fr.t[i], fr.t[e], fr.day[e], g + bond_leg - tick, nn + bond_leg - tick - comm_b,
                         n2 + bond_leg - tick - 2 * comm_b, mid, mid + bond_leg))
            lastx = e
            break                                         # one trade per formation day at most
    return _trades_frame(rows)


# ----------------------------------------------------------------------------- configurations
def configs(method: str, cfg: BFXConfig) -> list[dict]:
    if method == "leadlag":
        return [{"id": f"LL-L{L}-H{H}-k{k}", "L": L, "H": H, "k": k} for L in cfg.ll_L for H in cfg.ll_H for k in cfg.ll_k]
    if method == "residual":
        return [{"id": f"RR-W{W}-s{s}-h{h}", "W": W, "s_open": s, "hold": h}
                for W in cfg.rr_W for s in cfg.rr_open for h in cfg.rr_hold]
    if method == "coint":
        return [{"id": f"CI-F{f}-z{z}", "form": f, "z_open": z} for f in cfg.ci_form for z in cfg.ci_open]
    raise ValueError(method)


def run_config(method: str, fr: PairFrame, cfg: BFXConfig, c: dict, X: np.ndarray | None = None, cache: dict | None = None):
    cache = {} if cache is None else cache
    if method == "leadlag":
        key = ("ll", c["L"])
        if key not in cache:
            cache[key] = leadlag_signal(fr, cfg, c["L"], X)
        return run_leadlag(fr, cfg, c["L"], c["H"], c["k"], X, cache[key])
    if method == "residual":
        key = ("rr", c["W"])
        if key not in cache:
            cache[key] = residual_signal(fr, cfg, c["W"], X)
        return run_residual(fr, cfg, c["W"], c["s_open"], c["hold"], X, cache[key])
    key = ("ci", c["form"])
    if key not in cache:
        cache[key] = coint_formations(fr, cfg, c["form"], X)
    return run_coint(fr, cfg, cache[key], c["z_open"], X)


# ----------------------------------------------------------------------------- evaluation
def counted_years(fr: PairFrame, cfg: BFXConfig) -> list[int]:
    yd = pd.Series(fr.day).groupby(fr.years).nunique()
    return [int(y) for y, n in yd.items() if n >= cfg.min_year_days]


def daily(tr: pd.DataFrame, fr: PairFrame) -> pd.DataFrame:
    """One row per trading day of the frame (days without an exit = 0): gross, net, net2 summed by exit day."""
    days = np.unique(fr.day)
    g = tr.groupby("exit_day")[["gross_bps", "net_bps", "net2_bps"]].sum() if len(tr) else pd.DataFrame(
        columns=["gross_bps", "net_bps", "net2_bps"])
    g.index = np.asarray(g.index, dtype="datetime64[D]")
    out = g.reindex(days).astype(float).fillna(0.0)
    out["year"] = days.astype("datetime64[Y]").astype(int) + 1970
    out["date"] = days
    out["gross_pips"] = 0.0
    return out.reset_index(drop=True)


def evaluate(tr: pd.DataFrame, fr: PairFrame, cfg: BFXConfig) -> dict:
    from scipy.stats import norm
    dd = daily(tr, fr)
    m, se, t = F.nw_tstats(dd["net_bps"].to_numpy()[:, None], cfg.nw_lags)
    yrs = counted_years(fr, cfg)
    ym = dd[dd["year"].isin(yrs)].groupby("year")["net_bps"].mean()
    return {"n_trades": int(len(tr)), "trades_per_day": len(tr) / max(len(dd), 1),
            "mean_trade_net_bps": float(tr["net_bps"].mean()) if len(tr) else np.nan,
            "mean_trade_mid_bps": float(tr["mid_bps"].mean()) if len(tr) else np.nan,
            "mean_trade_hedged_bps": float(tr["hedged_mid_bps"].mean()) if len(tr) else np.nan,
            "win_rate": float((tr["net_bps"] > 0).mean()) if len(tr) else np.nan,
            "daily_net_bps": float(m[0]), "t": float(t[0]) if np.isfinite(t[0]) else np.nan,
            "p_one_sided": float(norm.sf(t[0])) if np.isfinite(t[0]) else np.nan,
            "years_positive": int((ym > 0).sum()), "years_counted": len(yrs)}


def scan(method: str, frames: dict, cfg: BFXConfig, X_override: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Every pairing x configuration of one method: metrics and trades."""
    rows, trades = [], {}
    for p, fr in frames.items():
        X = None if X_override is None else X_override.get(p)
        cache = {}
        for c in configs(method, cfg):
            tr = run_config(method, fr, cfg, c, X, cache)
            trades[(p, c["id"])] = tr
            rows.append({"pairing": p, "config": c["id"], **{k: v for k, v in c.items() if k != "id"}, **evaluate(tr, fr, cfg)})
    return pd.DataFrame(rows), trades


def select(res: pd.DataFrame, cfg: BFXConfig, method: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E1 >= min_trades; E2 BH (one-sided, daily net > 0) over every pairing x configuration, q <= fdr_q;
    E3 daily net > 0 in >= year_share_d of counted years. Candidate: per pairing, the passing configuration
    with the largest t."""
    res = res.copy()
    res["E1"] = res["n_trades"] >= cfg.min_trades
    p = np.where(res["E1"], res["p_one_sided"], np.nan)
    res["E2"], res["q"] = F.bh_reject(p, cfg.fdr_q)
    need = res["years_counted"].apply(lambda n: B.years_needed(n, cfg.year_share_d))
    res["E3"] = res["years_positive"] >= need
    res["E123"] = res["E1"] & res["E2"] & res["E3"]
    surv = res[res["E123"]].sort_values("t", ascending=False)
    cand = surv.groupby("pairing", sort=False).head(1).reset_index(drop=True)
    short = {"leadlag": "LL", "residual": "RR", "coint": "CI"}[method]
    cand.insert(0, "id", [f"{short}-{window}-{r['pairing']}" for _, r in cand.iterrows()])
    cand["method"], cand["window"] = method, window
    return res, cand


def placebo(method: str, frames: dict, cfg: BFXConfig, window: str) -> pd.DataFrame:
    """Bond series shifted by a random number of bars (at least placebo_min_shift_days days, circular) against
    the FX series: the bond link is destroyed, each series' own dynamics (including the FX pair's own mean
    reversion) are kept. Full pipeline per placebo; one row per run and pairing: best t and E1-E3 count."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for k in range(cfg.n_placebo):
        Xo = {}
        for p, fr in frames.items():
            lo = int(cfg.placebo_min_shift_days * fr.bars_per_day)
            Xo[p] = np.roll(fr.X, int(rng.integers(lo, max(lo + 1, fr.n - lo))), axis=0)
        res, _ = scan(method, frames, cfg, Xo)
        res, _ = select(res, cfg, method, window)
        for p, g in res.groupby("pairing"):
            rows.append({"run": k, "pairing": p, "best_t": float(g["t"].max()), "E1_E2_E3": int(g["E123"].sum())})
    return pd.DataFrame(rows)


def d0_table(res: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    """E4 / D0 per pairing: the real best t must exceed the best t of every placebo run (the bonds add something
    beyond the FX pair's own dynamics)."""
    real = res.groupby("pairing").agg(best_t=("t", "max"), E1_E2_E3=("E123", "sum"))
    plc = pl.groupby("pairing").agg(placebo_max_best_t=("best_t", "max"), placebo_max_E1_E2_E3=("E1_E2_E3", "max"))
    out = real.join(plc)
    out["E4"] = out["best_t"] > out["placebo_max_best_t"]
    out["D0"] = np.where(out["E4"], "STRUCTURE_FOUND", "NO_STRUCTURE")
    return out.reset_index()


def confirm(cand: pd.DataFrame, frames: dict, cfg: BFXConfig) -> tuple[pd.DataFrame, dict]:
    """Frozen candidates on another window's frames: K1 BH one-sided on daily mean after spread (before
    commission), K2 block-bootstrap lower bound of daily net > 0, K3 daily net at 2x commission > 0, K4 daily
    net > 0 in >= year_share_k of counted years."""
    if cand.empty:
        return pd.DataFrame(), {}
    daily_tr, trades = {}, {}
    for _, r in cand.iterrows():
        fr = frames[r["pairing"]]
        c = next(c for c in configs(r["method"], cfg) if c["id"] == r["config"])
        tr = run_config(r["method"], fr, cfg, c)
        trades[r["id"]] = tr
        daily_tr[r["id"]] = daily(tr, fr)
    fxc = F.FXSConfig(nw_lags=cfg.nw_lags, conf_fdr_q=cfg.conf_fdr_q, alpha=cfg.alpha, blocks=cfg.blocks,
                      n_boot=cfg.n_boot, seed=cfg.seed, commission_bps_rt=cfg.commission_bps_rt)
    out = F.confirm(daily_tr, fxc)
    ks = []
    for _, r in cand.iterrows():
        fr = frames[r["pairing"]]
        yrs = counted_years(fr, cfg)
        dd = daily_tr[r["id"]]
        ym = dd[dd["year"].isin(yrs)].groupby("year")["net_bps"].mean()
        ks.append({"id": r["id"], "n_trades": len(trades[r["id"]]), "years_counted": len(yrs),
                   "years_needed": B.years_needed(len(yrs), cfg.year_share_k), "years_net_positive": int((ym > 0).sum()),
                   "mean_trade_net_bps": float(trades[r["id"]]["net_bps"].mean()) if len(trades[r["id"]]) else np.nan})
    ks = pd.DataFrame(ks)
    out = out.drop(columns=["years_net_positive", "years", "gross_pips"]).merge(ks, on="id")
    out["K4"] = out["years_net_positive"] >= out["years_needed"]
    out["verdict"] = np.where(out[["K1", "K2", "K3", "K4"]].all(axis=1), "CONFIRMED", "NOT_CONFIRMED")
    return out, trades


CAND_COLS = ["id", "method", "window", "pairing", "config", "n_trades", "mean_trade_net_bps", "daily_net_bps", "t", "q",
             "years_positive", "years_counted"]


def load_frozen(folder: Path) -> tuple[pd.DataFrame, dict]:
    import json
    f, s = Path(folder) / "candidates_frozen.csv", Path(folder) / "summary.json"
    summ = json.loads(s.read_text())
    if not F.frozen_hash_ok(f, summ["candidates_sha256"]):
        raise RuntimeError(f"{f} changed after discovery")
    return pd.read_csv(f), summ


# ----------------------------------------------------------------------------- descriptive
def hourly_correlation(fr: PairFrame) -> pd.DataFrame:
    """Correlation of 5-minute FX returns with each bond's returns, by UTC hour."""
    ry, rX = fr.returns()
    h = np.asarray(fr.t.hour)
    rows = []
    for hr in range(24):
        sel = (h == hr) & np.isfinite(ry) & np.isfinite(rX).all(axis=1)
        if sel.sum() < 200:
            continue
        rows.append({"hour_utc": hr, "bars": int(sel.sum()),
                     **{BOND_SHORT[b]: float(np.corrcoef(ry[sel], rX[sel, j])[0, 1]) for j, b in enumerate(fr.bonds)}})
    return pd.DataFrame(rows)


def predictive_map(fr: PairFrame, cfg: BFXConfig, Ls=(1, 3, 6, 12), Hs=(1, 3, 6, 12)) -> pd.DataFrame:
    """t-statistic (Newey-West, H lags) of b in r_fx(t, t+H) = a + b implied_L(t) + c r_fx(t-L, t), on
    non-overlapping-safe HAC errors."""
    rows = []
    for L in Ls:
        implied, z, fxm, _ = leadlag_signal(fr, cfg, L)
        for H in Hs:
            fwd = -lagdiff(fr.y[::-1], fr.seg[::-1], H)[::-1]
            ok = np.isfinite(implied) & np.isfinite(fxm) & np.isfinite(fwd)
            if ok.sum() < 500:
                continue
            A = np.column_stack([np.ones(ok.sum()), implied[ok], fxm[ok]])
            yv = fwd[ok]
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            e = yv - A @ coef
            Xe = A * e[:, None]
            S = Xe.T @ Xe
            for l in range(1, H + 1):
                w = 1 - l / (H + 1)
                G = Xe[l:].T @ Xe[:-l]
                S += w * (G + G.T)
            Ainv = np.linalg.inv(A.T @ A)
            se = np.sqrt(np.diag(Ainv @ S @ Ainv))
            rows.append({"L": L, "H": H, "b": coef[1], "t_b": coef[1] / se[1], "c": coef[2], "t_c": coef[2] / se[2],
                         "n": int(ok.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- synthetic market
def write_synthetic_market(base: Path, start: dt.date, end: dt.date, seed: int, relations: bool = True,
                           spike: str | None = "2019-05-15 10:00") -> None:
    """Synthetic 1-minute candles for the three bond CFDs and EURUSD, GBPUSD, USDJPY, EURGBP.

    Bonds: random walks during their sessions (``B._open``), quotes stop 20 minutes before the close on switch
    days and the CFD jumps -60 bp at the next open (the economic price does not). With ``relations``:
    USDJPY minute returns = -0.4 x the T-Bond's minute return now and -0.8 x 5 minutes earlier + noise
    (bonds lead FX);
    EURUSD adjusts towards c - 0.3 log Bund + 0.3 log T-Bond with a half-life of 1 day (cointegrated);
    log GBPUSD = c - 0.4 log Gilt + OU (half-life 30 minutes, sd 4 bp), with the Gilt three times as volatile
    as the other bonds, so that GBPUSD on its own is dominated by the Gilt's random walk and the reversion is
    only visible against the Gilt (residual reversion);
    EURGBP: independent random walk. Shifting the bonds in time (the placebo) removes each relation."""
    rng = np.random.default_rng(seed)
    days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() != 5]
    ts = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d), periods=F.SLOTS, freq="1min").to_numpy()
                                          for d in days])).tz_localize("UTC")
    N = len(ts)
    ny = ts.tz_convert(F.NY)
    nwd, nmod = np.asarray(ny.weekday), np.asarray(ny.hour * 60 + ny.minute)
    fx_open = ~(((nwd == 4) & (nmod >= 17 * 60)) | (nwd == 5) | ((nwd == 6) & (nmod < 17 * 60)))
    rolls = set(B.roll_days(start, end))
    econ, cfd, quoted = {}, {}, {}
    px = {"USTBONDTRUSD": 120.0, "BUNDTREUR": 135.0, "UKGILTTRGBP": 99.0}
    for j, s in enumerate(B.SYMBOLS):
        op = B._open(s, ts)
        loc, mod = B._local(s, ts)
        ld = np.asarray(loc.date)
        early = op & np.isin(ld, list(rolls)) & (mod >= B.SYMBOLS[s]["close"] - 20) & (mod < B.SYMBOLS[s]["close"])
        op = op & ~early
        r = rng.normal(0, (1.5e-4 if s == "UKGILTTRGBP" else 0.5e-4), N) * op
        e = np.log(px[s]) + np.cumsum(r)
        jump = np.zeros(N)
        ends = np.flatnonzero(early[:-1] & ~early[1:])
        for i in ends:
            nxt = np.flatnonzero(op[i + 1:])
            if len(nxt):
                jump[i + 1 + nxt[0]] = -60e-4
        econ[s], cfd[s] = e, e + np.cumsum(jump)
        quoted[s] = op & (rng.random(N) < 0.9)
    eps = lambda sd: rng.normal(0, sd, N) * fx_open
    ust, bund, gilt = econ["USTBONDTRUSD"], econ["BUNDTREUR"], econ["UKGILTTRGBP"]
    fx = {}
    if relations:
        r_ust = np.r_[0.0, np.diff(ust)]
        lag = np.r_[np.zeros(5), r_ust[:-5]]
        fx["USDJPY"] = np.log(110.0) + np.cumsum((-0.4 * r_ust - 0.8 * lag) * fx_open + eps(0.3e-4))

        def adjust(target, half_life, noise_sd):
            a = 2 ** (-1 / half_life)
            e = rng.normal(0, noise_sd, N) * fx_open
            out = np.empty(N); out[0] = target[0]
            for i in range(1, N):
                out[i] = out[i - 1] + ((1 - a) * (target[i] - out[i - 1]) + e[i] if fx_open[i] else 0.0)
            return out
        fx["EURUSD"] = adjust(np.log(1.10) - 0.3 * (bund - bund[0]) + 0.3 * (ust - ust[0]), 1440, 0.2e-4)
        d = np.zeros(N); a = 2 ** (-1 / 30); z = rng.normal(0, 4e-4 * math.sqrt(1 - a * a), N)
        for i in range(1, N):
            d[i] = a * d[i - 1] + z[i] if fx_open[i] else d[i - 1]
        fx["GBPUSD"] = np.log(1.30) - 0.4 * (gilt - gilt[0]) + d
    else:
        fx["USDJPY"] = np.log(110.0) + np.cumsum(eps(0.4e-4))
        fx["EURUSD"] = np.log(1.10) + np.cumsum(eps(0.4e-4))
        fx["GBPUSD"] = np.log(1.30) + np.cumsum(eps(0.4e-4))
    fx["EURGBP"] = np.log(0.85) + np.cumsum(eps(0.3e-4))
    series = {**{s: (cfd[s], quoted[s], 0.05, 3) for s in B.SYMBOLS},
              **{p: (fx[p], fx_open & (rng.random(N) < 0.95), 0.2 * F.pip_size(p), 3 if p.endswith("JPY") else 5) for p in fx}}
    if spike:
        k = int(np.searchsorted(ts, pd.Timestamp(spike, tz="UTC")))
        s0 = "BUNDTREUR"
        lv = series[s0][0].copy(); lv[k] += 0.03
        q = series[s0][1].copy(); q[k] = q[k + 1] = True
        series[s0] = (lv, q, 0.05, 3)
    for sym, (lv, q, spread, dig) in series.items():
        mid = np.exp(lv)
        bid, ask = np.round(mid - spread / 2, dig), np.round(mid + spread / 2, dig)
        v = q.astype(np.float32)
        root = Path(base) / sym / "candles_m1"
        for n, d in enumerate(days):
            sl = slice(n * F.SLOTS, (n + 1) * F.SLOTS)
            df = pd.DataFrame({"ts": ts[sl], "bid_o": bid[sl], "bid_h": bid[sl], "bid_l": bid[sl], "bid_c": bid[sl],
                               "bid_v": v[sl], "ask_o": ask[sl], "ask_h": ask[sl], "ask_l": ask[sl], "ask_c": ask[sl],
                               "ask_v": v[sl]})
            f = root / f"{d:%Y}" / f"{sym}_{d:%Y-%m-%d}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(f, index=False)
