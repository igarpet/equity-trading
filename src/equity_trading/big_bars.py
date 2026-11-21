"""When do large MNQ 5-minute bars happen? Descriptive (MRP 006). No rule is selected or tested here.

Two definitions of "large":
* relative: |z| >= k, z = r / sigma-hat (MRP 001 volatility model). The time-of-day profile is part of sigma-hat, so
  a bar at a busy time (08:30 New York) must move more to count.
* absolute: the bar's move in ticks is in the top q of all eligible bars (e.g. 1%), whatever the time.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .futures import CHICAGO, NY

BB_VERSION = "0.1.0"
TICK = 0.25

# FOMC policy-statement days (federalreserve.gov, fetched 2026-11-21). Statements at 14:00 New York.
# 2020 includes the unscheduled 2020-03-03 (10:00 NY) and 2020-03-15 (Sunday, 17:00 NY) cuts; notation votes and the
# 2020-08-27 strategy statement and 2025-08-22 notation vote are left out.
FOMC = [
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]
FOMC_DATES = {dt.date.fromisoformat(d) for d in FOMC}

# Clock times (New York) of scheduled US releases and session events, for labelling 5-minute bars by open time.
MARKS = {"08:30 releases (CPI, NFP, claims, PPI, retail sales, GDP)": "08:30",
         "09:30 cash open": "09:30", "10:00 releases (ISM, confidence, JOLTS)": "10:00",
         "14:00 FOMC statement (FOMC days)": "14:00", "14:30 FOMC press conference (FOMC days)": "14:30",
         "16:00 cash close": "16:00"}


def bar_events(bt: pd.DataFrame, k: float = 3.0, top_q: float = 0.01) -> pd.DataFrame:
    """Eligible bars with clock labels and both 'large' flags. bt: reversal.bar_table output."""
    e = bt[bt["eligible"]].copy()
    ny = e.index.tz_convert(NY)
    e["ny_time"] = ny.strftime("%H:%M")
    e["ny_hour"] = ny.hour + ny.minute / 60
    e["weekday"] = ny.day_name()
    e["ny_date"] = ny.date
    e["move_ticks"] = np.abs(e["ret"]) * e["close"] / TICK
    e["big_z"] = e["z"].abs() >= k
    thr = e["move_ticks"].quantile(1 - top_q)
    e["big_abs"] = e["move_ticks"] >= thr
    e.attrs["abs_threshold_ticks"] = float(thr)
    e["fomc_day"] = [d in FOMC_DATES for d in e["ny_date"]]
    d = pd.to_datetime(pd.Series(e["ny_date"].to_numpy()))
    e["first_friday"] = ((d.dt.weekday == 4) & (d.dt.day <= 7)).to_numpy()     # proxy for payrolls days
    e["year"] = d.dt.year.to_numpy()
    return e


def slot_rates(e: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Per New York clock slot: bars, large bars, rate, and lift (rate / overall rate)."""
    g = e.groupby("ny_time")[flag].agg(bars="size", big="sum")
    g["rate"] = g["big"] / g["bars"]
    g["lift"] = g["rate"] / (e[flag].mean() or np.nan)
    g["share_of_all_big"] = g["big"] / max(e[flag].sum(), 1)
    return g


def mark_table(e: pd.DataFrame, flag: str) -> pd.DataFrame:
    rows = []
    n_big = max(int(e[flag].sum()), 1)
    for name, t in MARKS.items():
        sel = e["ny_time"] == t
        if "FOMC" in name:
            sel &= e["fomc_day"]
        rows.append({"slot": name, "bars": int(sel.sum()), "big": int(e.loc[sel, flag].sum()),
                     "rate": float(e.loc[sel, flag].mean()) if sel.any() else np.nan,
                     "share_of_bars": float(sel.mean()), "share_of_big": float(e.loc[sel, flag].sum() / n_big)})
    out = pd.DataFrame(rows)
    out["lift"] = out["rate"] / e[flag].mean()
    return out


def weekday_hour(e: pd.DataFrame, flag: str) -> pd.DataFrame:
    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    p = e.assign(h=np.floor(e["ny_hour"]).astype(int)).pivot_table(index="weekday", columns="h", values=flag,
                                                                    aggfunc="mean")
    return p.reindex([d for d in order if d in p.index]) * 100


def clustering(e: pd.DataFrame, flag: str, horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    """P(another large bar within the next n bars of the same CME day | large bar now) vs the unconditional rate."""
    rows = []
    for _, g in [("all", e)]:
        x = g[flag].to_numpy()
        day = g["trade_date"].to_numpy()
        for n in horizons:
            nxt = np.zeros(len(x), bool)
            for j in range(1, n + 1):
                s = np.zeros(len(x), bool)
                s[:-j] = x[j:] & (day[j:] == day[:-j])
                nxt |= s
            base = nxt.mean()
            cond = nxt[x].mean() if x.any() else np.nan
            rows.append({"within_bars": n, "p_given_big": cond, "p_unconditional": base, "ratio": cond / base})
    return pd.DataFrame(rows)


def per_day(e: pd.DataFrame, flag: str) -> pd.DataFrame:
    c = e.groupby("trade_date")[flag].sum()
    return pd.DataFrame({"days": [len(c)], "days_with_any": [int((c > 0).sum())],
                         "share_days_with_any": [float((c > 0).mean())], "mean_per_day": [float(c.mean())],
                         "top10pct_days_share_of_big": [float(c.sort_values(ascending=False)
                                                              .head(max(1, len(c) // 10)).sum() / max(c.sum(), 1))]})


def load_calendar(path: Path) -> pd.DataFrame | None:
    """Optional user calendar: CSV with columns date (YYYY-MM-DD), time_ny (HH:MM), event."""
    path = Path(path)
    if not path.exists():
        return None
    c = pd.read_csv(path, dtype=str)
    c["ts"] = pd.to_datetime(c["date"] + " " + c["time_ny"]).dt.tz_localize(NY).dt.tz_convert("UTC")
    return c


def calendar_hits(e: pd.DataFrame, cal: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Large-bar rate in the 5-minute bar that contains each calendar release, by event type."""
    idx = e.index
    rows = []
    for ev, g in cal.groupby("event"):
        bars = g["ts"].dt.floor("5min")
        hit = e.reindex(bars)
        rows.append({"event": ev, "releases": len(g), "bars_found": int(hit[flag].notna().sum()),
                     "big": int(hit[flag].fillna(False).sum()), "rate": float(hit[flag].dropna().mean())
                     if hit[flag].notna().any() else np.nan,
                     "median_move_ticks": float(hit["move_ticks"].median())})
    out = pd.DataFrame(rows)
    out["lift"] = out["rate"] / e[flag].mean()
    return out
