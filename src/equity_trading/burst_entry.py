"""Burst entry on 1-minute MNQ bars (RSW 05; discussions/MNQ_Regime_Switching/009).

Rules (Iván, 2026-11-22):
* Regime models refitted on 1-MINUTE bars: 2-state HMM (M1) and 2-state Weibull HSMM (WB2), on z_t = r_t / (sigma_t L_t)
  from a 1-minute volatility model (time-of-day profile x HAR on squared 1-minute returns; L = rms of z over the
  previous 1,380 minutes, one CME day). Burst = the state with the larger sd.
* Entry: at the close of the first 1-minute candle with P(burst | F_t) > 0.5 after a minute with P <= 0.5.
  Direction: WITH that candle (continuation) or AGAINST it (reversion).
* Time stop: round(mean burst duration in minutes, from the frozen fit) + 5 minutes.
* Stops: none; extreme = 2 ticks beyond the entry candle's extreme against the position (long: below its low;
  short: above its high); m x 5-minute sigma (sqrt(5) x next-minute sigma_hat, m = 1.5, 3; same size definition as
  RSW 04); fixed 40 ticks.
* One position at a time; flat by 15:55 Chicago, never across the daily break or a contract switch.
Fills: market and stop fills pay 0.5 tick; a stop gapped through fills at the minute's open; $0.50 commission per side.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import mnq_predict as P
from .futures import CHICAGO, NY, cme_trade_date

try:
    from numba import njit
except ImportError:                                  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

BE_VERSION = "0.1.0"
TICK = 0.25
USD_PER_TICK = 0.50
MIN_PER_DAY = 1380                                   # 17:00 -> 16:00 Chicago


@dataclass(frozen=True)
class BEConfig:
    drop_first: int = 3                  # first minutes of each CME day are not observations
    min_day_minutes: int = 900           # CME days with fewer traded minutes are not observations
    har: tuple = (("Qh", 60), ("Qd", 1380), ("Qw", 5 * 1380), ("Qm", 22 * 1380))
    h_floor: float = 0.1
    local_bars: int = 1380
    hsmm_D: int = 30                     # HSMM ages 1..30 minutes, 30 = tail with constant hazard
    p_burst: float = 0.5
    extra_bars: int = 5
    stops: tuple = ("none", "extreme", "vol1.5", "vol3", "fixed40")
    directions: tuple = ("with", "against")
    extreme_buffer: int = 2
    fixed_ticks: int = 40
    commission_per_side: float = 0.50
    slip: float = 0.5
    flat_ct: tuple = (15, 55)
    min_trades: int = 150
    n_placebo: int = 20
    alpha: float = 0.05

    @property
    def comm_ticks(self) -> float:
        return self.commission_per_side / USD_PER_TICK


# ============================================================================= 1. 1-minute grid

def slot_1m(ts: pd.DatetimeIndex) -> np.ndarray:
    loc = ts.tz_convert(CHICAGO)
    return (((loc.hour * 60 + loc.minute) - 17 * 60) % 1440).to_numpy()


def grid_1m(m1: pd.DataFrame, rc: BEConfig = BEConfig()) -> pd.DataFrame:
    """Traded 1-minute bars with trade_date, slot, ret (log, from the previous minute only if that minute traded in
    the same CME day and contract; NaN otherwise), n_day (traded minutes in the CME day). No empty rows: a minute
    without trades has no row (the chain simply moves on)."""
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index)
    td = np.asarray(cme_trade_date(pd.Series(ts)).to_numpy())
    slot = slot_1m(ts)
    keep = slot < MIN_PER_DAY
    m1, ts, td, slot = m1[keep], ts[keep], td[keep], slot[keep]
    lc = np.log(m1["close"].to_numpy(float))
    iid = m1["instrument_id"].to_numpy()
    consec = np.r_[False, (np.diff(ts.values.astype("datetime64[s]").astype(np.int64)) == 60) & (td[1:] == td[:-1]) & (iid[1:] == iid[:-1])]
    ret = np.where(consec, np.r_[np.nan, np.diff(lc)], np.nan)
    g = pd.DataFrame({"trade_date": td, "slot": slot, "open": m1["open"].to_numpy(float),
                      "high": m1["high"].to_numpy(float), "low": m1["low"].to_numpy(float),
                      "close": m1["close"].to_numpy(float), "volume": m1["volume"].to_numpy(), "iid": iid,
                      "ret": ret}, index=ts)
    g["n_day"] = g.groupby("trade_date")["close"].transform("size")
    return g


def coverage(g: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month: CME days, traded minutes, minutes per day (about 1,000-1,380 for MNQ in normal months)."""
    d = pd.to_datetime(pd.Series(g["trade_date"].to_numpy()))
    m = d.dt.to_period("M")
    out = pd.DataFrame({"days": d.groupby(m).nunique(), "minutes": d.groupby(m).size()})
    out["minutes_per_day"] = out["minutes"] / out["days"]
    return out


# ============================================================================= 2. 1-minute volatility model

@dataclass
class Vol1m:
    prof: np.ndarray            # mean r^2 per slot
    beta: np.ndarray            # HAR on [1, Qh, Qd, Qw, Qm] (deseasonalized r^2)
    fit_end: dt.date


def _har(q: np.ndarray, rc: BEConfig) -> np.ndarray:
    qs = pd.Series(q)
    return np.column_stack([qs.rolling(w, min_periods=w // 2).mean().to_numpy() for _, w in rc.har])


def fit_vol_1m(g: pd.DataFrame, fit_start: dt.date, fit_end_excl: dt.date, rc: BEConfig = BEConfig()) -> Vol1m:
    td = g["trade_date"].to_numpy()
    slot = g["slot"].to_numpy()
    r2 = g["ret"].to_numpy(float) ** 2
    win = (td >= fit_start) & (td < fit_end_excl) & np.isfinite(r2)
    s = pd.Series(r2[win]).groupby(slot[win]).mean().reindex(range(MIN_PER_DAY))
    s = s.rolling(5, center=True, min_periods=1).mean()                   # light smoothing across minutes
    overall = float(np.nanmean(r2[win]))
    prof = np.maximum(s.fillna(overall).to_numpy(float), overall * 1e-3)
    q = r2 / prof[slot]
    Q = _har(np.where(np.isfinite(q), q, np.nan), rc)
    y = np.r_[q[1:], np.nan]
    X = np.column_stack([np.ones(len(Q)), Q])
    ok = np.r_[win[1:], False] & np.isfinite(y) & np.isfinite(X).all(1)
    beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return Vol1m(prof, beta, fit_end_excl)


def predict_vol_1m(g: pd.DataFrame, f: Vol1m, rc: BEConfig = BEConfig()) -> np.ndarray:
    """Ex-ante variance of each minute (forecast at the previous row)."""
    slot = g["slot"].to_numpy()
    r2 = g["ret"].to_numpy(float) ** 2
    q = r2 / f.prof[slot]
    Q = _har(np.where(np.isfinite(q), q, np.nan), rc)
    h = np.maximum(np.column_stack([np.ones(len(Q)), Q]) @ f.beta, rc.h_floor)
    return f.prof[slot] * np.r_[np.nan, h[:-1]]


def vol_walk_forward(g: pd.DataFrame, first_fit: tuple, refits: tuple, end: dt.date, rc: BEConfig = BEConfig()) -> np.ndarray:
    """2020 (before refits[0]) in sample on ``first_fit``; later dates from fits made before each refit."""
    td = g["trade_date"].to_numpy()
    out = np.full(len(g), np.nan)
    f0 = fit_vol_1m(g, first_fit[0], first_fit[1], rc)
    p0 = predict_vol_1m(g, f0, rc)
    sel = td < refits[0]
    out[sel] = p0[sel]
    edges = list(refits) + [end + dt.timedelta(days=1)]
    for a, b in zip(edges[:-1], edges[1:]):
        f = fit_vol_1m(g, first_fit[0], a, rc)
        p = predict_vol_1m(g, f, rc)
        sel = (td >= a) & (td < b)
        out[sel] = p[sel]
    return out


def build_series_1m(g: pd.DataFrame, sig2: np.ndarray, rc: BEConfig = BEConfig()) -> pd.DataFrame:
    """RSW-style series on 1-minute rows: z, zl, has_lag, obs, sig2 (effective), close/high/low, iid, imb/vsurp = 0."""
    r = g["ret"].to_numpy(float)
    z0 = r / np.sqrt(sig2)
    day = g["trade_date"].to_numpy()
    elig = np.isfinite(z0) & (g["slot"].to_numpy() >= rc.drop_first) & (g["n_day"].to_numpy() >= rc.min_day_minutes)
    L2 = pd.Series(np.where(elig, z0 ** 2, np.nan)).rolling(rc.local_bars, min_periods=rc.local_bars // 2).mean().shift(1).to_numpy()
    s2 = sig2 * L2
    z = r / np.sqrt(s2)
    elig &= np.isfinite(z)
    consec = np.r_[False, np.diff(g.index.values.astype("datetime64[s]").astype(np.int64)) == 60]
    prev_ok = consec & np.r_[False, (day[1:] == day[:-1]) & np.isfinite(z[:-1])]
    zl = np.where(prev_ok, np.r_[np.nan, z[:-1]], 0.0)
    return pd.DataFrame({"trade_date": day, "slot": g["slot"].to_numpy(), "z": z, "zl": zl, "has_lag": prev_ok,
                         "obs": elig, "sig2": s2, "ret": r, "open": g["open"].to_numpy(), "high": g["high"].to_numpy(),
                         "low": g["low"].to_numpy(), "close": g["close"].to_numpy(), "iid": g["iid"].to_numpy(),
                         "imb": 0.0, "vsurp": 0.0}, index=g.index)


# ============================================================================= 3. Events and simulation

def flat_index(s: pd.DataFrame, rc: BEConfig = BEConfig()) -> np.ndarray:
    """For each row: first row index at which the position must be flat (15:55 Chicago, CME day end, contract
    switch). Rows at/after 15:55 point to themselves."""
    loc = s.index.tz_convert(CHICAGO)
    mins = np.asarray(loc.hour * 60 + loc.minute)
    late = (mins >= rc.flat_ct[0] * 60 + rc.flat_ct[1]) & (mins < 17 * 60)
    day = s["trade_date"].to_numpy()
    iid = s["iid"].to_numpy()
    n = len(s)
    out = np.empty(n, dtype=np.int64)
    nxt = n
    for i in range(n - 1, -1, -1):
        if i == n - 1 or day[i + 1] != day[i] or iid[i + 1] != iid[i]:
            nxt = i + 1
        if late[i]:
            nxt = i
        out[i] = nxt
    return out


def burst_starts(s: pd.DataFrame, p_burst: np.ndarray, hold: np.ndarray, rc: BEConfig = BEConfig()) -> pd.DataFrame:
    """First minute with P(burst) > p after a minute with P <= p, on an observed row with a non-zero return."""
    pb = np.nan_to_num(np.asarray(p_burst, float), nan=-1.0)
    inb = pb > rc.p_burst
    start = np.flatnonzero(inb & np.r_[False, (pb[:-1] >= 0) & ~inb[:-1]])
    r = s["ret"].to_numpy(float)
    ok = s["obs"].to_numpy(bool)[start] & np.isfinite(r[start]) & (r[start] != 0)
    j = start[ok]
    return pd.DataFrame({"row": j, "candle_sign": np.sign(r[j]).astype(int), "hold": hold[j].astype(int),
                         "trade_date": s["trade_date"].to_numpy()[j], "entry_ts": s.index[j]})


@njit(cache=True)
def _sim(o, h, l, c, flat, rows, sides, stop_dist, holds, slip, comm, tick):
    n = len(c)
    K = len(rows)
    out_i = np.full(K, -1, np.int64)
    out_exit = np.full(K, -1, np.int64)
    out_gross = np.zeros(K)
    out_reason = np.zeros(K, np.int64)        # 1 stop, 2 time, 3 flat
    free = -1
    for k in range(K):
        j = rows[k]
        i0 = j + 1
        if j <= free or i0 >= n or flat[j] <= i0 or flat[i0] <= i0:
            continue                          # no room before the flat time / day end / roll
        side = sides[k]
        entry = c[j] + side * slip * tick
        stop = entry - side * stop_dist[k] * tick
        end = min(j + 1 + holds[k], flat[i0])
        px = np.nan
        reason = 2
        i = i0
        while i < end:
            hit = (l[i] <= stop) if side > 0 else (h[i] >= stop)
            if hit:
                gap = (o[i] <= stop) if side > 0 else (o[i] >= stop)
                px = (o[i] if gap else stop) - side * slip * tick
                reason = 1
                break
            i += 1
        if reason != 1:
            i = end - 1
            if end < j + 1 + holds[k]:
                reason = 3
            px = c[i] - side * slip * tick
        out_i[k] = j
        out_exit[k] = i
        out_gross[k] = side * (px - entry) / tick
        out_reason[k] = reason
        free = i
    return out_i, out_exit, out_gross, out_reason


def stop_distance(kind: str, s: pd.DataFrame, rows: np.ndarray, sides: np.ndarray, rc: BEConfig = BEConfig()) -> np.ndarray:
    close = s["close"].to_numpy(float)[rows]
    entry = close + sides * rc.slip * TICK
    if kind == "none":
        return np.full(len(rows), 1e9)
    if kind == "fixed40":
        return np.full(len(rows), float(rc.fixed_ticks))
    if kind.startswith("vol"):
        m = float(kind[3:])
        sig_next = np.sqrt(s["sig2"].to_numpy(float)[np.minimum(rows + 1, len(s) - 1)])
        return np.maximum(np.round(m * np.sqrt(5.0) * sig_next * close / TICK), 1.0)
    if kind == "extreme":
        lvl = np.where(sides > 0, s["low"].to_numpy(float)[rows] - rc.extreme_buffer * TICK,
                       s["high"].to_numpy(float)[rows] + rc.extreme_buffer * TICK)
        return np.maximum(np.round(sides * (entry - lvl) / TICK), 1.0)
    raise ValueError(kind)


def simulate(s: pd.DataFrame, flat: np.ndarray, ev: pd.DataFrame, kind: str, sides: np.ndarray,
             rc: BEConfig = BEConfig()) -> pd.DataFrame:
    rows = ev["row"].to_numpy(np.int64)
    sides = np.asarray(sides, np.int64)
    sd = stop_distance(kind, s, rows, sides, rc)
    sd = np.where(np.isfinite(sd), sd, 1e9)
    i, ex, gross, reason = _sim(s["open"].to_numpy(float), s["high"].to_numpy(float), s["low"].to_numpy(float),
                                s["close"].to_numpy(float), flat, rows, sides, sd, ev["hold"].to_numpy(np.int64),
                                rc.slip, rc.comm_ticks, TICK)
    took = i >= 0
    out = pd.DataFrame({"entry_ts": ev["entry_ts"].to_numpy()[took], "trade_date": ev["trade_date"].to_numpy()[took],
                        "side": sides[took], "candle_sign": ev["candle_sign"].to_numpy()[took], "stop_ticks": sd[took],
                        "hold": ev["hold"].to_numpy()[took], "minutes_held": (ex - i + 1)[took] if took.any() else [],
                        "reason": np.array(["", "stop", "time", "flat"])[reason[took]],
                        "gross_ticks": gross[took], "net_ticks": gross[took] - 2 * rc.comm_ticks})
    out.attrs["skipped"] = int((~took).sum())
    return out


# ============================================================================= 4. Synthetic (tests, smoke run)

def write_synthetic_1m(base, start: dt.date, end: dt.date, cfg, seed: int = 51, cont: float = 0.8,
                       burst_mult: float = 3.0, p_enter: float = 1 / 60, p_leave: float = 0.2) -> None:
    """MNQ ohlcv-1m month files with a 2-state minute regime: calm sd = base, bursts burst_mult x base, entered with
    probability p_enter per minute, lasting 1 / p_leave minutes on average. Planted continuation: every burst minute
    after the first carries a drift of ``cont`` x the base sd in the direction of the burst's first minute
    (cont = 0: null). Daily log-vol AR(1) and a U-shaped intraday profile, quarterly rolls, 3% empty minutes.
    High / low come from a 12-step path inside each minute (first version used open/close plus a small wiggle,
    which understated intrabar excursions and made tight stops look profitable on null data)."""
    import os
    from pathlib import Path
    from . import micro_bars as MB
    base_p = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base_p)
    rng = np.random.default_rng(seed)
    price, logvol, iid, state, first = 9000.0, 0.0, 1, 0, 0.0
    rows = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            open_ = pd.Timestamp(d - dt.timedelta(days=1), tz=CHICAGO) + pd.Timedelta(hours=17)
            ts = (open_ + pd.to_timedelta(np.arange(MIN_PER_DAY), unit="min")).tz_convert("UTC")
            if d.day == 10 and d.month in (3, 6, 9, 12):
                iid += 1
                price += 50.0
            logvol = 0.95 * logvol + 0.25 * rng.normal()
            loc = ts.tz_convert(NY)
            mins = np.asarray(loc.hour * 60 + loc.minute, float)
            prof = 1.0 + 2.5 * np.exp(-((mins - 575) / 25.0) ** 2) + 1.2 * np.exp(-((mins - 955) / 20.0) ** 2)
            sd = 2.0e-4 * np.exp(logvol) * prof
            r = np.empty(MIN_PER_DAY)
            mult = np.ones(MIN_PER_DAY)
            for t in range(MIN_PER_DAY):
                if state == 0 and rng.random() < p_enter:
                    state, first = 1, 0.0
                elif state == 1 and rng.random() < p_leave:
                    state = 0
                if state == 1:
                    mult[t] = burst_mult
                    e = burst_mult * rng.normal()
                    if first == 0.0:
                        first = np.sign(e)
                        r[t] = e * sd[t]
                    else:
                        r[t] = (e + cont * first) * sd[t]
                else:
                    r[t] = rng.normal() * sd[t]
            # within-minute path: 12 steps per minute summing to r (a Brownian bridge), so high / low are the
            # path's real extremes (a stop sees the true intrabar excursion)
            ns = 12
            sub = rng.normal(0, 1, (MIN_PER_DAY, ns))
            sub = (sub - sub.mean(1, keepdims=True)) * (sd * mult)[:, None] / np.sqrt(ns)   # burst minutes: wider
            sub = sub + (r / ns)[:, None]
            lp0 = np.log(price) + np.r_[0.0, np.cumsum(r)[:-1]]               # log open of each minute
            path = lp0[:, None] + np.cumsum(sub, axis=1)
            close = np.exp(path[:, -1])
            opn = np.exp(lp0)
            high = np.maximum(opn, np.exp(path.max(1)))
            low = np.minimum(opn, np.exp(path.min(1)))
            price = close[-1]
            keep = rng.random(MIN_PER_DAY) > 0.03
            rows.append(pd.DataFrame({"ts": ts[keep], "open": opn[keep], "high": high[keep], "low": low[keep],
                                      "close": close[keep], "volume": rng.poisson(30, keep.sum()) + 1,
                                      "instrument_id": iid}))
        d += dt.timedelta(days=1)
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for key, part in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        a = dt.date(int(key[:4]), int(key[5:]), 1)
        f = MB.month_path(cfg.root, a)
        f.parent.mkdir(parents=True, exist_ok=True)
        part[MB.COLUMNS].reset_index(drop=True).to_parquet(f, index=False)
