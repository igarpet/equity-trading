"""Burst reversion on MNQ (RSW 04; discussions/MNQ_Regime_Switching/007).

Rules (Iván, 2026-11-22):
* Regime: causal P(burst | F_t) from the walk-forward RSW fits (2-state HMM M1, or the 2-state Weibull HSMM WB2).
  The burst state is the state with the larger sd.
* Burst episode: consecutive 5-minute rows with P(burst) > 0.5. The switch is the first row after it with P(burst)
  <= 0.5; entry at the close of that bar (market).
* Direction: AGAINST the episode's net move (close of the row before the episode -> close of its last row).
* Time stop: the frozen fit's mean calm duration + 5 bars (exit at the close of bar entry + H).
* Stop loss, one of: none; beyond the burst extreme (+2 ticks); m x sigma_hat of the next bar (m = 1.5, 3);
  fixed 40 ticks.
* One position at a time; flat by 15:55 Chicago and before a contract switch; a new burst does not close the trade.
  A switch whose entry would fall at or after 15:55 Chicago (or whose next minute is in the next CME day) is not
  traded: every trade opens and closes inside one CME session, never across the daily break or a weekend.

Fills on 1-minute bars (MRP 04 conventions): market and stop fills pay 0.5 tick; a stop gapped through fills at the
minute's open; commission per side. Everything in ticks per contract (1 tick = 0.25 points = $0.50).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import trail_strategy as T

BR_VERSION = "0.1.1"   # 0.1.1: no entry when the next minute is past 15:55 Chicago or in another CME day (0.1.0 carried those across the daily break)
TICK = 0.25
USD_PER_TICK = 0.50


@dataclass(frozen=True)
class BRConfig:
    p_burst: float = 0.5
    extra_bars: int = 5
    stops: tuple = ("none", "extreme", "vol1.5", "vol3", "fixed40")
    extreme_buffer: int = 2           # ticks beyond the burst's low (long) / high (short)
    fixed_ticks: int = 40
    commission_per_side: float = 0.50
    slip: float = 0.5
    flat_ct: tuple = (15, 55)
    min_trades: int = 150
    n_placebo: int = 20
    alpha: float = 0.05
    drop_first: int = 3

    @property
    def comm_ticks(self) -> float:
        return self.commission_per_side / USD_PER_TICK

    @property
    def round_trip_ticks(self) -> float:
        return 2 * self.slip + 2 * self.comm_ticks


# ============================================================================= signals

def episodes(s: pd.DataFrame, p_burst: np.ndarray, hold: np.ndarray, rc: BRConfig = BRConfig()) -> pd.DataFrame:
    """One row per burst episode that ends in a tradable switch.

    ``s``: RSW series rows in time order with close, trade_date, slot, obs, sig2, iid; ``p_burst`` aligned to s
    (NaN outside the evaluated window); ``hold`` bars per row (mean calm duration + extra, from the fit in force)."""
    pb = np.asarray(p_burst, float)
    inb = np.nan_to_num(pb, nan=0.0) > rc.p_burst
    close = s["close"].to_numpy(float)
    day = s["trade_date"].to_numpy()
    iid = s["iid"].to_numpy()
    obs = s["obs"].to_numpy(bool)
    slot = s["slot"].to_numpy()
    rows = []
    n = len(s)
    i = 1
    while i < n:
        if inb[i] and not inb[i - 1]:
            s0 = i
            while i < n and inb[i]:
                i += 1
            j = i                                            # switch row
            if j >= n or not np.isfinite(pb[j]):
                break
            pre = s0 - 1
            same = (day[pre] == day[j]) and (iid[pre] == iid[j])
            move = close[j - 1] - close[pre]
            if same and obs[j] and slot[j] >= rc.drop_first and move != 0 and np.isfinite(s["sig2"].iat[min(j + 1, n - 1)]):
                rows.append({"start_row": s0, "switch_row": j, "burst_bars": j - s0, "burst_move_ticks": move / TICK,
                             "side": -int(np.sign(move)), "entry_ts": s.index[j], "close5": close[j],
                             "sig_next": float(np.sqrt(s["sig2"].iat[j + 1])) if j + 1 < n else np.nan,
                             "hold_bars": int(hold[j]), "trade_date": day[j],
                             "ts_start": s.index[s0], "ts_last_burst": s.index[j - 1]})
        else:
            i += 1
    return pd.DataFrame(rows)


# ============================================================================= simulation

def _first_min(M: T.Minutes, ts: pd.Timestamp) -> int:
    """Index of the first minute starting at or after ``ts``."""
    return int(M.ts.searchsorted(ts, side="left"))


def burst_range(M: T.Minutes, ts_start: pd.Timestamp, ts_last: pd.Timestamp) -> tuple[float, float]:
    a = _first_min(M, ts_start)
    b = _first_min(M, ts_last + pd.Timedelta(minutes=5))
    if b <= a:
        return np.nan, np.nan
    return float(M.l[a:b].min()), float(M.h[a:b].max())


def stop_ticks(kind: str, ev: pd.Series, entry: float, lo: float, hi: float, side: int, rc: BRConfig) -> float:
    if kind == "none":
        return np.inf
    if kind == "fixed40":
        return float(rc.fixed_ticks)
    if kind.startswith("vol"):
        m = float(kind[3:])
        return float(max(np.round(m * ev["sig_next"] * ev["close5"] / TICK), 1))
    if kind == "extreme":
        lvl = (lo - rc.extreme_buffer * TICK) if side > 0 else (hi + rc.extreme_buffer * TICK)
        return float(max(np.round(side * (entry - lvl) / TICK), 1))
    raise ValueError(kind)


def simulate(M: T.Minutes, ev: pd.DataFrame, grid_index: pd.DatetimeIndex, kind: str, rc: BRConfig = BRConfig(),
             sides: np.ndarray | None = None) -> pd.DataFrame:
    """Non-overlapping trades for one stop rule. ``sides`` overrides the direction (placebo / mirror)."""
    out = []
    free_from = -1
    skipped = 0
    n_rows = len(grid_index)
    for k, e in enumerate(ev.itertuples(index=False)):
        side = int(sides[k]) if sides is not None else int(e.side)
        i0 = _first_min(M, e.entry_ts + pd.Timedelta(minutes=5))      # first minute after the entry bar
        if i0 <= free_from or i0 >= len(M.ts):
            continue
        if M.day[i0] != e.trade_date or M.stop_at[i0] <= i0:
            skipped += 1                                                 # entry at/after the flat time: no trade
            continue
        entry = e.close5 + side * rc.slip * TICK
        lo, hi = burst_range(M, e.ts_start, e.ts_last_burst)
        st = stop_ticks(kind, pd.Series(e._asdict()), entry, lo, hi, side, rc)
        stop = entry - side * st * TICK if np.isfinite(st) else (-np.inf if side > 0 else np.inf)
        exit_row = min(e.switch_row + e.hold_bars, n_rows - 1)
        i_time = _first_min(M, grid_index[exit_row] + pd.Timedelta(minutes=5))   # exclusive: up to that bar's close
        end = min(i_time, int(M.stop_at[i0]))
        reason, i = "time", i0
        px = None
        while i < end:
            if M.day[i] != M.day[i0] or M.iid[i] != M.iid[i0]:
                break
            hit = (M.l[i] <= stop) if side > 0 else (M.h[i] >= stop)
            if hit:
                gap = (M.o[i] <= stop) if side > 0 else (M.o[i] >= stop)
                px = (M.o[i] if gap else stop) - side * rc.slip * TICK
                reason = "stop"
                break
            i += 1
        if px is None:
            last = max(min(i, end) - 1, i0)
            if end < i_time:
                reason = "flat"
            px = M.c[last] - side * rc.slip * TICK
            i = last
        gross = side * (px - entry) / TICK
        net = gross - 2 * rc.comm_ticks
        out.append({"entry_ts": e.entry_ts, "trade_date": e.trade_date, "side": side, "burst_side": -int(e.side),
                    "burst_bars": e.burst_bars, "burst_move_ticks": e.burst_move_ticks, "stop_ticks": st,
                    "hold_bars": e.hold_bars, "exit_ts": M.ts[i], "reason": reason, "gross_ticks": gross,
                    "net_ticks": net, "minutes_held": i - i0 + 1})
        free_from = i
    df = pd.DataFrame(out)
    df.attrs["skipped_session_end"] = skipped
    return df


def stats_for(tr: pd.DataFrame, periods: dict) -> dict:
    from .reversal import day_cluster_t
    from scipy import stats
    if len(tr) < 3:
        return {"trades": len(tr), "net": np.nan, "t": np.nan, "p": 1.0}
    x = tr["net_ticks"].to_numpy(float)
    m, t = day_cluster_t(x, tr["trade_date"].to_numpy())
    w = x > 0
    cum = np.cumsum(x)
    row = {"trades": len(x), "gross": float(tr["gross_ticks"].mean()), "net": m, "t": t, "p": float(stats.norm.sf(t)),
           "win_rate": float(w.mean()), "mean_win": float(x[w].mean()) if w.any() else np.nan,
           "mean_loss": float(x[~w].mean()) if (~w).any() else np.nan,
           "total_ticks": float(x.sum()), "total_usd_1c": float(x.sum() * USD_PER_TICK),
           "max_dd_ticks": float((np.maximum.accumulate(cum) - cum).max()), "worst": float(x.min()),
           "q05": float(np.quantile(x, 0.05)), "exit_stop": float((tr["reason"] == "stop").mean()),
           "exit_time": float((tr["reason"] == "time").mean()), "exit_flat": float((tr["reason"] == "flat").mean()),
           "mean_minutes": float(tr["minutes_held"].mean()),
           "net_long": float(tr.loc[tr["side"] > 0, "net_ticks"].mean()) if (tr["side"] > 0).any() else np.nan,
           "net_short": float(tr.loc[tr["side"] < 0, "net_ticks"].mean()) if (tr["side"] < 0).any() else np.nan,
           "n_long": int((tr["side"] > 0).sum()), "n_short": int((tr["side"] < 0).sum())}
    d = tr["trade_date"].to_numpy()
    for p, (a, b) in periods.items():
        sel = (d >= a) & (d <= b)
        row[f"net_{p}"] = float(x[sel].mean()) if sel.any() else np.nan
    return row


# ============================================================================= synthetic (smoke run)

def write_synthetic_burst_reversion(base, start: dt.date, end: dt.date, cfg, seed: int = 41,
                                    revert_frac: float = 0.6, revert_bars: int = 15) -> None:
    """MNQ ohlcv-1m month files: 2-state calm/burst z (calm sd 0.85, phi -0.05, burst sd 2.5, phi 0.2; burst lasts
    about 2 bars, calm about 15). After each burst ends, the next ``revert_bars`` calm bars carry a drift that takes
    back ``revert_frac`` of the burst's net move in z. revert_frac = 0: no planted reversion (null smoke)."""
    import os
    from pathlib import Path
    from . import micro_bars as MB
    from .futures import CHICAGO, NY
    base = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base)
    rng = np.random.default_rng(seed)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    T_ = len(days) * 276
    z = np.empty(T_)
    s, zp, cum, drift, left = 0, 0.0, 0.0, 0.0, 0
    for t in range(T_):
        if t:
            if s == 0 and rng.random() < 1 / 15:
                s, cum = 1, 0.0
            elif s == 1 and rng.random() < 0.5:
                s = 0
                drift, left = -revert_frac * cum / revert_bars, revert_bars
        if s == 1:
            z[t] = 0.2 * zp + 2.5 * rng.normal()
            cum += z[t]
        else:
            z[t] = -0.05 * zp + 0.85 * rng.normal() + (drift if left > 0 else 0.0)
            left -= 1
        zp = z[t]
    price, logvol, iid = 9000.0, 0.0, 1
    rows = []
    for i, d in enumerate(days):
        open_ = pd.Timestamp(d - dt.timedelta(days=1), tz=CHICAGO) + pd.Timedelta(hours=17)
        ts = (open_ + pd.to_timedelta(np.arange(23 * 60), unit="min")).tz_convert("UTC")
        if d.day == 10 and d.month in (3, 6, 9, 12):
            iid += 1
            price += 50.0
        logvol = 0.95 * logvol + 0.25 * rng.normal()
        loc = ts.tz_convert(NY)
        mins = np.asarray(loc.hour * 60 + loc.minute, float)
        prof = 1.0 + 2.5 * np.exp(-((mins - 575) / 25.0) ** 2) + 1.2 * np.exp(-((mins - 955) / 20.0) ** 2)
        sd = (2.0e-4 * np.exp(logvol) * prof).reshape(-1, 5)
        bsd = np.sqrt((sd ** 2).sum(1))
        zb = z[i * 276:(i + 1) * 276]
        e = rng.normal(0, 1, (276, 5)) * sd
        e = e - e.mean(1, keepdims=True)
        r = (e + (zb * bsd)[:, None] / 5).ravel()
        close = np.exp(np.log(price) + np.cumsum(r))
        price = close[-1]
        keep = rng.random(len(ts)) > 0.01
        rows.append(pd.DataFrame({"ts": ts[keep], "open": close[keep], "high": close[keep] + 0.25,
                                  "low": close[keep] - 0.25, "close": close[keep],
                                  "volume": rng.poisson(30, keep.sum()) + 1, "instrument_id": iid}))
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for key, part in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        a = dt.date(int(key[:4]), int(key[5:]), 1)
        f = MB.month_path(cfg.root, a)
        f.parent.mkdir(parents=True, exist_ok=True)
        part[MB.COLUMNS].reset_index(drop=True).to_parquet(f, index=False)
