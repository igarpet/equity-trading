"""MRP 04: continuation entry with a volatility stop, a 50/30 trailing stop and one reverse re-entry (MRP 005).

Simulated on 1-minute MNQ bars (Databento ohlcv-1m). Everything in ticks per contract (0.25 points, $0.50).

Rules (Iván, 2026-11-21):
* Signal: 5-minute bar t with |z_t| >= k (MRP 001 volatility model). Enter WITH the bar at its close (market).
* Initial stop: m x sigma_{t+1} (5-minute volatility forecast of the next bar) from the entry, rounded to ticks.
* Trailing: once the trade is +50 ticks, the stop goes to +30 ticks and then follows the best price 20 ticks behind,
  tick by tick. The stop never moves back.
* Re-entry: if the INITIAL stop is hit, a limit order in the REVERSE direction is placed 2 ticks beyond the stop
  (long stopped at S -> sell limit at S + 2 ticks), working 30 minutes. The re-entry trade follows the same stop and
  trailing rules. One re-entry per signal.
* Flat at 15:55 Chicago at the latest (and before any contract switch). One signal chain at a time.

Fill model (conservative): market and stop fills pay 0.5 tick slippage; a stop gapped through fills at the bar open;
a limit fills only when the bar trades through it by 1 tick, at the limit price; commission per fill. Inside a minute
the adverse extreme is checked before the favourable one, and a stop raised in a minute applies from the next minute.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .futures import CHICAGO, cme_trade_date

TRAIL_VERSION = "0.1.0"
TICK = 0.25
USD_PER_TICK = 0.50


@dataclass(frozen=True)
class TrailConfig:
    k: float = 3.0
    stop_mults: tuple = (1.0, 1.5, 2.0, 3.0)
    trail_activate: int = 50          # ticks in profit before the trailing stop starts
    trail_lock: int = 30              # stop moves to this profit at activation
    reentry_offset: int = 2           # ticks beyond the stop price for the reverse limit
    reentry_expiry_min: int = 30
    max_reentries: int = 1
    flat_ct: tuple = (15, 55)         # Chicago time: flat by then
    commission_per_side: float = 0.50
    slip_market: float = 0.5          # ticks per market / stop fill
    min_chains: int = 150
    n_placebo: int = 20
    alpha: float = 0.05

    @property
    def trail_dist(self) -> int:
        return self.trail_activate - self.trail_lock

    @property
    def comm_ticks(self) -> float:
        return self.commission_per_side / USD_PER_TICK


# ----------------------------------------------------------------------------- minute arrays

@dataclass
class Minutes:
    ts: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    iid: np.ndarray
    day: np.ndarray
    stop_at: np.ndarray        # for each minute: first index at which the position must be flat (exclusive end)


def minutes_from_1m(m1: pd.DataFrame, cfg: TrailConfig = TrailConfig()) -> Minutes:
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index)
    day = np.asarray(cme_trade_date(pd.Series(ts)).to_numpy())
    loc = ts.tz_convert(CHICAGO)
    mins = (loc.hour * 60 + loc.minute).to_numpy()
    late = (mins >= cfg.flat_ct[0] * 60 + cfg.flat_ct[1]) & (mins < 17 * 60)
    iid = m1["instrument_id"].to_numpy()
    n = len(ts)
    stop_at = np.empty(n, dtype=np.int64)
    nxt = n
    for i in range(n - 1, -1, -1):
        if i == n - 1 or day[i + 1] != day[i] or iid[i + 1] != iid[i]:
            nxt = i + 1
        if late[i]:
            nxt = i                     # this minute itself is past the flat time
        stop_at[i] = nxt
    return Minutes(ts, m1["open"].to_numpy(float), m1["high"].to_numpy(float), m1["low"].to_numpy(float),
                   m1["close"].to_numpy(float), iid, day, stop_at)


# ----------------------------------------------------------------------------- one trade

def run_trade(M: Minutes, i0: int, side: int, entry: float, stop_ticks: float, cfg: TrailConfig,
              fill_bar: bool = False) -> dict:
    """Trade from minute i0 (first bar that can hit the stop). Returns exit index, price, reason, P&L in ticks.

    fill_bar=True: i0 is the minute the entry filled in; its adverse extreme is checked (conservative) but its
    favourable extreme is not credited (it may have happened before the fill)."""
    end = M.stop_at[i0] if i0 < len(M.ts) else i0
    stop = entry - side * stop_ticks * TICK
    best = entry
    active = False
    i = i0
    while i < end:
        o, hi, lo = M.o[i], M.h[i], M.l[i]
        adverse_hit = (lo <= stop) if side > 0 else (hi >= stop)
        if adverse_hit:
            gap = (o <= stop) if side > 0 else (o >= stop)
            px = (o if gap else stop) - side * cfg.slip_market * TICK
            return _close(entry, px, side, i, "trail" if active else "stop", best, cfg)
        fav = hi if side > 0 else lo
        if fill_bar and i == i0:
            fav = best
        if (fav - best) * side > 0:
            best = fav
        if not active and (best - entry) * side >= cfg.trail_activate * TICK:
            active = True
        if active:
            new = best - side * cfg.trail_dist * TICK
            if (new - stop) * side > 0:
                stop = new
        i += 1
    last = max(min(end, len(M.ts)) - 1, i0)
    px = M.c[last] - side * cfg.slip_market * TICK
    return _close(entry, px, side, last, "flat", best, cfg)


def _close(entry, px, side, i, reason, best, cfg):
    gross = side * (px - entry) / TICK
    return {"exit_i": i, "exit_px": px, "reason": reason, "gross_ticks": gross,
            "mfe_ticks": side * (best - entry) / TICK}


def wait_limit(M: Minutes, i0: int, side: int, limit: float, cfg: TrailConfig) -> int | None:
    """Index of the minute in which a limit order (side = direction of the new position) fills, or None."""
    t_end = M.ts[min(i0, len(M.ts) - 1)] + pd.Timedelta(minutes=cfg.reentry_expiry_min)
    end = M.stop_at[i0] if i0 < len(M.ts) else i0
    i = i0
    while i < end and M.ts[i] < t_end:
        if side < 0 and M.h[i] >= limit + TICK:          # sell limit above the market
            return i
        if side > 0 and M.l[i] <= limit - TICK:          # buy limit below the market
            return i
        i += 1
    return None


# ----------------------------------------------------------------------------- signal chains

def signals(bt: pd.DataFrame, sig2: pd.Series, k: float) -> pd.DataFrame:
    """Continuation signals from the 5-minute bar table (reversal.bar_table): close time, direction, sigma of t+1."""
    z = bt["z"].to_numpy()
    ev = bt["eligible"].to_numpy() & (np.abs(np.nan_to_num(z)) >= k)
    s = bt[ev]
    close_t = s.index + pd.Timedelta(minutes=5)
    sig_next = sig2.reindex(close_t).to_numpy(float)
    return pd.DataFrame({"t_close": close_t, "side": np.sign(s["z"].to_numpy()).astype(int),
                         "close5": s["close"].to_numpy(float), "sig_next": sig_next,
                         "trade_date": s["trade_date"].to_numpy()}, index=s.index)


def run_chains(M: Minutes, sig: pd.DataFrame, sig2: pd.Series, mult: float, cfg: TrailConfig,
               sides: np.ndarray | None = None) -> pd.DataFrame:
    """Simulate every signal chain (entry + at most ``max_reentries`` reverse re-entries), one at a time."""
    rows = []
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    sig2_idx = sig2.dropna()
    for j, s in enumerate(sig.itertuples()):
        if s.t_close <= busy_until or not np.isfinite(s.sig_next):
            continue
        i0 = int(M.ts.searchsorted(s.t_close))           # first minute after the signal bar closed
        if i0 >= len(M.ts) or M.day[i0] != s.trade_date or M.stop_at[i0] <= i0:
            continue
        side = int(sides[j]) if sides is not None else s.side
        entry = s.close5 + side * cfg.slip_market * TICK
        stop_ticks = max(1.0, round(mult * np.sqrt(s.sig_next) * s.close5 / TICK))
        legs = []
        tr = run_trade(M, i0, side, entry, stop_ticks, cfg)
        legs.append({**tr, "leg": 0, "side": side, "entry_px": entry, "stop_ticks": stop_ticks,
                     "entry_ts": M.ts[i0], "fills": 2, "market_fills": 2})
        n_re = 0
        while legs[-1]["reason"] == "stop" and n_re < cfg.max_reentries:
            last = legs[-1]
            stop_px = last["entry_px"] - last["side"] * last["stop_ticks"] * TICK
            new_side = -last["side"]
            limit = stop_px - new_side * cfg.reentry_offset * TICK   # short re-entry: above the stop
            k0 = last["exit_i"] + 1
            if k0 >= len(M.ts) or M.day[k0] != s.trade_date:
                break
            f = wait_limit(M, k0, new_side, limit, cfg)
            if f is None:
                break
            tsf = M.ts[f]
            s2 = sig2_idx.asof(tsf.floor("5min")) if len(sig2_idx) else np.nan
            st = max(1.0, round(mult * np.sqrt(s2) * limit / TICK)) if np.isfinite(s2) else last["stop_ticks"]
            tr = run_trade(M, f, new_side, limit, st, cfg, fill_bar=True)
            legs.append({**tr, "leg": n_re + 1, "side": new_side, "entry_px": limit, "stop_ticks": st,
                         "entry_ts": tsf, "fills": 2, "market_fills": 1})
            n_re += 1
        for lg in legs:
            cost = lg["fills"] * cfg.comm_ticks          # slippage already in the fill prices
            rows.append({"signal_ts": s.Index, "trade_date": s.trade_date, "leg": lg["leg"], "side": lg["side"],
                         "entry_ts": lg["entry_ts"], "exit_ts": M.ts[lg["exit_i"]], "reason": lg["reason"],
                         "stop_ticks": lg["stop_ticks"], "gross_ticks": lg["gross_ticks"],
                         "net_ticks": lg["gross_ticks"] - cost, "mfe_ticks": lg["mfe_ticks"]})
        busy_until = M.ts[legs[-1]["exit_i"]]
    return pd.DataFrame(rows)


def chain_table(legs: pd.DataFrame) -> pd.DataFrame:
    if legs.empty:
        return pd.DataFrame(columns=["signal_ts", "trade_date", "net_ticks", "legs", "initial_stop_ticks", "R"])
    g = legs.groupby("signal_ts", sort=True)
    out = pd.DataFrame({"trade_date": g["trade_date"].first(), "net_ticks": g["net_ticks"].sum(),
                        "legs": g.size(), "initial_stop_ticks": g["stop_ticks"].first()})
    out["R"] = out["net_ticks"] / out["initial_stop_ticks"]
    return out.reset_index()


def sized_usd(chains: pd.DataFrame, risk_usd: float) -> pd.Series:
    """P&L in USD when each chain risks ``risk_usd`` on its initial stop (at least 1 contract)."""
    n = np.maximum(1, np.floor(risk_usd / (chains["initial_stop_ticks"] * USD_PER_TICK)))
    return chains["net_ticks"] * USD_PER_TICK * n
