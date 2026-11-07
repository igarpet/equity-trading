"""USD/JPY gotobi-day Tokyo-fix study (branch Gotobi_Tokyo_Fix, GTF 001).

Source idea: Ito, T. & Yamada, M. (2017), Journal of International Money and Finance — USD/JPY
rises into the 9:55 JST Tokyo fix on gotobi days (settlement days ending in 5 or 0). The trading
rule tested (from a YouTube backtest, 2026): on a gotobi day sell USD/JPY at 9:55 JST, stop
12 pips above the fill, buy back at 10:30 JST.

Data: Dukascopy bid/ask ticks written by ``tools/dukascopy/dukascopy_download.py``, one Parquet
file per UTC day. Japan has no DST, so 9:55 JST = 00:55 UTC on the same calendar date.

Discipline enforced here, not only in the notebook:
* no date on or after ``HOLDOUT_START`` (2026-01-01) is read unless ``allow_holdout=True``;
* fills use the quoted side (sell at bid, buy at ask); no mid-price fills.
"""
from __future__ import annotations

import calendar
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import vendor_dir, work_dir

GOTOBI_VERSION = "0.2.0"   # 0.2.0: correction C1 (pre-00:00 quote, commission at fill) + GTF 02 long leg
SYMBOL = "USDJPY"
PIP = 0.01
HOLDOUT_START = dt.date(2026, 1, 1)
MS_MIN = 60_000
MORNING_END_MS = 2 * 60 * MS_MIN          # keep ticks 00:00-02:00 UTC (09:00-11:00 JST)
PREV_HOUR_MS = 60 * MS_MIN                # plus the last hour of the previous UTC day (C1)
BANK_YEAR_END = {(12, 31), (1, 1), (1, 2), (1, 3)}

# Yen interventions by Japan's Ministry of Finance (robustness exclusion only).
# Dates from MoF intervention records as recalled; VERIFY against the MoF
# "Foreign Exchange Intervention Operations" tables before relying on them.
INTERVENTION_DAYS = [
    "2010-09-15", "2011-03-18", "2011-08-04", "2011-10-31", "2011-11-01", "2011-11-02",
    "2011-11-03", "2011-11-04", "2022-09-22", "2022-10-21", "2022-10-24",
    "2024-04-29", "2024-05-01", "2024-07-11", "2024-07-12",
]


class HoldoutError(RuntimeError):
    """Raised when code would read data from the sealed holdout (2026 onward)."""


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class GTFConfig:
    """Pre-registered specification (discussions/Gotobi_Tokyo_Fix/001_claude_gtf01_preregistration.md)."""
    entry_min: int = 55                    # 00:55 UTC = 09:55 JST (fix)
    exit_min: int = 90                     # 01:30 UTC = 10:30 JST
    stop_pips: float | None = 12.0
    max_lag_ms: int = 30_000               # fill must occur within 30 s of the scheduled time
    commission_usd_per_m: float = 35.0     # Dukascopy Europe, net deposit < $5k, per side
    extra_slip_pips: float = 0.0           # per side, on top of the quoted spread
    alpha: float = 0.025
    hac_lags: int = 5
    blocks: tuple = (10, 20)
    n_boot: int = 10_000
    n_random_calendars: int = 10_000
    seed: int = 42
    # robustness grid (reported, not gated)
    grid_entry: tuple = (54, 55, 56)
    grid_exit: tuple = (75, 90, 105)
    grid_stop: tuple = (8.0, 12.0, 20.0, None)
    path_minutes: tuple = field(default_factory=lambda: tuple(range(0, 106)))
    path_max_age_ms: int = 120_000         # mid path: last quote at most 2 minutes old (C1)
    # GTF 02: long into the fix, then reverse into the GTF 01 short
    long_entry_min: int = 25               # 00:25 UTC = 09:25 JST
    long_grid_entry: tuple = (0, 15, 25, 35, 45)


WINDOWS = {
    "INDEPENDENT": (dt.date(2010, 1, 1), dt.date(2015, 12, 31)),   # binding: before the video's sample
    "REPLICATION": (dt.date(2016, 1, 1), dt.date(2025, 12, 31)),   # the video's sample, 2026 excluded
}


# ----------------------------------------------------------------------------- calendar
def is_jp_bank_day(d: dt.date) -> bool:
    """Japanese bank business day: Mon-Fri, not a national holiday, not Dec 31 - Jan 3."""
    import jpholiday
    return d.weekday() < 5 and (d.month, d.day) not in BANK_YEAR_END and not jpholiday.is_holiday(d)


def previous_bank_day(d: dt.date) -> dt.date:
    while not is_jp_bank_day(d):
        d -= dt.timedelta(days=1)
    return d


def _months(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def gotobi_days(start: dt.date, end: dt.date, month_end: bool = False) -> list[dt.date]:
    """Gotobi settlement days in [start, end].

    Video definition (``month_end=False``): calendar days 5, 10, 15, 20, 25, 30 of each month
    (February has no 30th), each moved to the previous bank day when it is not one.
    ``month_end=True`` adds the last calendar day of every month, rolled back the same way
    (a common market convention; sensitivity only).
    """
    out: set[dt.date] = set()
    lo = dt.date(start.year, start.month, 1) - dt.timedelta(days=1)
    for y, m in _months(lo, end + dt.timedelta(days=31)):
        last = calendar.monthrange(y, m)[1]
        days = [d for d in (5, 10, 15, 20, 25, 30) if d <= last]
        if month_end:
            days.append(last)
        for d in days:
            out.add(previous_bank_day(dt.date(y, m, d)))
    return sorted(d for d in out if start <= d <= end)


def day_table(start: dt.date, end: dt.date) -> pd.DataFrame:
    """One row per weekday with calendar labels."""
    days = [d.date() for d in pd.bdate_range(start, end)]
    g = set(gotobi_days(start, end))
    gm = set(gotobi_days(start, end, month_end=True))
    rows = []
    for d in days:
        bank = is_jp_bank_day(d)
        last = calendar.monthrange(d.year, d.month)[1]
        is_g = d in g
        rows.append({
            "date": d, "jp_bank_day": bank, "gotobi": is_g, "gotobi_m": d in gm,
            # gotobi that settles the 30th / month end (the roll of day 30 or the last day)
            "gotobi_month_end": is_g and (previous_bank_day(dt.date(d.year, d.month, min(30, last))) == d),
            "weekday": d.weekday(),
            "intervention": d.isoformat() in INTERVENTION_DAYS,
        })
    t = pd.DataFrame(rows)
    t["day_type"] = np.where(~t["jp_bank_day"], "jp_holiday", np.where(t["gotobi"], "gotobi", "control"))
    return t


def shifted_calendar(table: pd.DataFrame, k: int) -> pd.Series:
    """Gotobi labels moved k JP bank days (k<0 earlier). Returns a boolean Series on table.index."""
    bank = table.index[table["jp_bank_day"].to_numpy()]
    pos = {ix: i for i, ix in enumerate(bank)}
    lab = pd.Series(False, index=table.index)
    for ix in table.index[table["gotobi"].to_numpy()]:
        j = pos[ix] + k
        if 0 <= j < len(bank):
            lab[bank[j]] = True
    return lab


# ----------------------------------------------------------------------------- data access
def dukascopy_root(symbol: str = SYMBOL) -> Path:
    """Folder holding daily Parquet ticks whose hours include 00 and 01 UTC.

    Base: env ``DUKASCOPY_DIR`` if set (the downloader's --out), else <EQUITY_TRADING_DIR>/Dukascopy.
    Prefers the folder with the most day files.
    """
    base = Path(os.environ["DUKASCOPY_DIR"]) if os.environ.get("DUKASCOPY_DIR") else vendor_dir("Dukascopy")
    best, best_n = None, 0
    for p in sorted((base / symbol).glob("hours_*")):
        tag = p.name[len("hours_"):]
        hours = set(range(24)) if tag == "all" else {int(h) for h in tag.split("-")}
        if {0, 1} <= hours:
            n = sum(1 for _ in p.rglob(f"{symbol}_*.parquet"))
            if n > best_n:
                best, best_n = p, n
    if best is None:
        raise FileNotFoundError(f"No Dukascopy {symbol} folder with hours 00-01 UTC under {base / symbol}. "
                                "Set DUKASCOPY_DIR to the downloader's --out folder.")
    return best


def day_file(root: Path, day: dt.date, symbol: str = SYMBOL) -> Path:
    return root / f"{day:%Y}" / f"{symbol}_{day:%Y-%m-%d}.parquet"


def _read_ticks(f: Path, day: dt.date):
    df = pd.read_parquet(f, columns=["ts", "bid", "ask"])
    ts = pd.to_datetime(df["ts"], utc=True)
    t = ((ts - pd.Timestamp(day, tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(np.int64)
    return t, df["bid"].to_numpy(float), df["ask"].to_numpy(float)


def load_morning(root: Path, day: dt.date, symbol: str = SYMBOL, allow_holdout: bool = False):
    """Ticks from 23:00 UTC of the previous day to 02:00 UTC as (t_ms relative to 00:00 UTC of
    `day`, bid, ask); t < 0 for the previous day's last hour. None if the day has no file.

    The previous hour (C1) supplies the quote standing at 00:00 UTC (09:00 JST); without it the
    09:00 mid is missing on almost every day.
    """
    if day >= HOLDOUT_START and not allow_holdout:
        raise HoldoutError(f"{day} is in the sealed holdout (>= {HOLDOUT_START}).")
    f = day_file(root, day, symbol)
    if not f.exists():
        return None
    t, b, a = _read_ticks(f, day)
    keep = (t >= 0) & (t < MORNING_END_MS)
    t, b, a = t[keep], b[keep], a[keep]
    prev = day - dt.timedelta(days=1)
    fp = day_file(root, prev, symbol)
    if fp.exists():
        tp, bp, ap = _read_ticks(fp, day)          # times relative to `day`, so negative
        kp = (tp >= -PREV_HOUR_MS) & (tp < 0)
        t, b, a = np.concatenate([tp[kp], t]), np.concatenate([bp[kp], b]), np.concatenate([ap[kp], a])
    return t, b, a


# ----------------------------------------------------------------------------- trade simulation
def short_trade(t, bid, ask, entry_ms: int, exit_ms: int, stop_pips: float | None, max_lag_ms: int):
    """Sell at the first bid at/after entry_ms, buy back at the first ask at/after exit_ms,
    or at the first ask >= fill + stop. Returns (pnl_pips, reason)."""
    i = int(np.searchsorted(t, entry_ms, "left"))
    if i >= len(t) or t[i] - entry_ms > max_lag_ms:
        return np.nan, "no_entry"
    j = int(np.searchsorted(t, exit_ms, "left"))
    if j >= len(t) or t[j] - exit_ms > max_lag_ms:
        return np.nan, "no_exit"
    fill = bid[i]
    if stop_pips is not None:
        hit = np.flatnonzero(ask[i + 1:j] >= fill + stop_pips * PIP - 1e-9)
        if hit.size:
            return (fill - ask[i + 1 + hit[0]]) / PIP, "stop"
    return (fill - ask[j]) / PIP, "time"


def long_trade(t, bid, ask, entry_ms: int, exit_ms: int, max_lag_ms: int):
    """Buy at the first ask at/after entry_ms, sell at the first bid at/after exit_ms (no stop)."""
    i = int(np.searchsorted(t, entry_ms, "left"))
    j = int(np.searchsorted(t, exit_ms, "left"))
    if i >= len(t) or t[i] - entry_ms > max_lag_ms or j >= len(t) or t[j] - exit_ms > max_lag_ms:
        return np.nan
    return (bid[j] - ask[i]) / PIP


def mid_at(t, bid, ask, mark_ms: int, max_age_ms: int = 60_000) -> float:
    """Last mid at or before mark_ms (nan if older than max_age_ms)."""
    k = int(np.searchsorted(t, mark_ms, "right")) - 1
    if k < 0 or mark_ms - t[k] > max_age_ms:
        return np.nan
    return 0.5 * (bid[k] + ask[k])


def grid_key(e: int, x: int, s) -> str:
    return f"pnl_e{e}_x{x}_s{'none' if s is None else int(s)}"


def day_outcome(t, bid, ask, cfg: GTFConfig) -> dict:
    """All per-day quantities: grid of short trades, pre-fix long, mid path, spread and price at fix."""
    rec = {"n_ticks": int((t >= 0).sum())}
    if len(t) == 0:
        return rec
    fix = cfg.entry_min * MS_MIN
    i = int(np.searchsorted(t, fix, "left"))
    fill_ok = i < len(t) and t[i] - fix <= cfg.max_lag_ms
    rec["spread_fix_pips"] = (ask[i] - bid[i]) / PIP if fill_ok else np.nan
    rec["entry_price"] = bid[i] if fill_ok else np.nan          # short fill at 09:55 JST; prices commission (C1)
    rec["mid_fix"] = mid_at(t, bid, ask, fix, cfg.path_max_age_ms)
    for m in cfg.path_minutes:
        rec[f"mid_{m:03d}"] = mid_at(t, bid, ask, m * MS_MIN, cfg.path_max_age_ms)
    seen = set()
    for e in (*cfg.grid_entry, cfg.entry_min):
        for x in (*cfg.grid_exit, cfg.exit_min):
            for s in (*cfg.grid_stop, cfg.stop_pips):
                k = grid_key(e, x, s)
                if k in seen:
                    continue
                seen.add(k)
                pnl, why = short_trade(t, bid, ask, e * MS_MIN, x * MS_MIN, s, cfg.max_lag_ms)
                rec[k] = pnl
                if (e, x, s) == (cfg.entry_min, cfg.exit_min, cfg.stop_pips):
                    rec["base_reason"] = why
    rec["prefix_long_pips"] = long_trade(t, bid, ask, 0, fix, cfg.max_lag_ms)
    # GTF 02: long from e to the fix (buy ask, sell bid at the same tick the short is sold)
    for e in sorted({*cfg.long_grid_entry, cfg.long_entry_min}):
        rec[f"long_e{e}"] = long_trade(t, bid, ask, e * MS_MIN, fix, cfg.max_lag_ms)
        i0 = int(np.searchsorted(t, e * MS_MIN, "left"))
        rec[f"long_price_e{e}"] = ask[i0] if i0 < len(t) and t[i0] - e * MS_MIN <= cfg.max_lag_ms else np.nan
    return rec


def _one_day(root, day, cfg, allow_holdout=False):
    data = load_morning(root, day, allow_holdout=allow_holdout)
    if data is None:
        return {"date": day, "has_file": False}
    return {"date": day, "has_file": True, **day_outcome(*data, cfg)}


def build_day_outcomes(start: dt.date, end: dt.date, cfg: GTFConfig, root: Path | None = None,
                       n_jobs: int = 4, refresh: bool = False, allow_holdout: bool = False) -> pd.DataFrame:
    """Day table joined with per-day outcomes; cached under <EQUITY_TRADING_DIR>/EquityTrading/Gotobi/."""
    if end >= HOLDOUT_START and not allow_holdout:
        raise HoldoutError(f"end {end} reaches the sealed holdout.")
    root = root or dukascopy_root()
    cache = work_dir() / "Gotobi" / f"day_outcomes_{GOTOBI_VERSION}_{root.name}_{start}_{end}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    table = day_table(start, end)
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=n_jobs)(delayed(_one_day)(root, d, cfg, allow_holdout) for d in table["date"])
    out = table.merge(pd.DataFrame(recs), on="date", how="left")
    out["year"] = pd.to_datetime(out["date"]).dt.year
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


# ----------------------------------------------------------------------------- costs
def commission_pips(price: np.ndarray | float, usd_per_m_side: float) -> np.ndarray | float:
    """Round-trip commission in pips for a USD-notional USD/JPY trade.

    One pip on $1M is 10,000 JPY = 10,000 / price USD, so 2c / (10,000 / price) = 2c * price / 10,000.
    """
    return 2.0 * usd_per_m_side * np.asarray(price, float) / 10_000.0


def net_pips(gross: np.ndarray, price: np.ndarray, usd_per_m_side: float, extra_slip_pips: float = 0.0):
    return np.asarray(gross, float) - commission_pips(price, usd_per_m_side) - 2.0 * extra_slip_pips


# ----------------------------------------------------------------------------- statistics
def hac_mean(x, lags: int) -> dict:
    """Mean with Newey-West standard error; one-sided p for mean > 0."""
    import statsmodels.api as sm
    from scipy.stats import norm
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {"n": int(len(x)), "mean": float(np.mean(x)) if len(x) else np.nan, "t": np.nan, "p_one_sided": np.nan}
    fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(fit.tvalues[0])
    return {"n": int(len(x)), "mean": float(x.mean()), "se": float(fit.bse[0]), "t": t, "p_one_sided": float(norm.sf(t))}


def block_bootstrap_lower(x, block: int, n_boot: int, alpha: float, seed: int) -> dict:
    """Moving-block bootstrap of the mean; lower bound at quantile alpha (one-sided 1-alpha)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return {"mean": float(np.mean(x)) if n else np.nan, "lower": np.nan, "block": block}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, (n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    m = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "lower": float(np.quantile(m, alpha)), "block": block}


def welch_diff(a, b) -> dict:
    """Mean(a) - mean(b), Welch t, one-sided p for a > b."""
    from scipy.stats import ttest_ind
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    r = ttest_ind(a, b, equal_var=False, alternative="greater")
    return {"diff": float(a.mean() - b.mean()), "t": float(r.statistic), "p_one_sided": float(r.pvalue),
            "n_a": int(len(a)), "n_b": int(len(b))}


def random_calendar_means(pnl: np.ndarray, month: np.ndarray, is_gotobi: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Mean P&L of random calendars: in every month, draw as many JP bank days (with a valid trade)
    as that month has gotobi days, uniformly without replacement. Same count per month as the real calendar."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, float)
    ok = np.isfinite(pnl)
    sums = np.zeros(n)
    total = 0
    for mo in np.unique(month[ok]):
        pool = np.flatnonzero(ok & (month == mo))
        k = int((is_gotobi[pool]).sum())
        if k == 0:
            continue
        keys = rng.random((n, len(pool)))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
        sums += pnl[pool][pick].sum(axis=1)
        total += k
    return sums / max(total, 1)


def placebo_rank(value: float, dist: np.ndarray) -> dict:
    dist = np.asarray(dist, float)
    return {"value": float(value), "percentile": float(100 * np.mean(dist < value)),
            "p_upper": float((1 + np.sum(dist >= value)) / (len(dist) + 1)),
            "placebo_max": float(dist.max()), "placebo_p95": float(np.percentile(dist, 95))}


def equity_curve(net: pd.Series, risk: float = 0.005, stop_pips: float = 12.0) -> pd.DataFrame:
    """Compounded equity risking `risk` of equity per trade with a `stop_pips` stop (return = risk * pnl / stop)."""
    r = risk * net.astype(float) / stop_pips
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return pd.DataFrame({"ret": r, "equity": eq, "drawdown": dd})


def gates(window: str, g1_hac: dict, g1_boot: list[dict], g2_diff: dict, g2_rand: dict,
          g3_boot: list[dict], g3_mean_2x: float, alpha: float) -> dict:
    """Pre-registered gates G1-G3 for one window (GTF 001 §4)."""
    g1 = (g1_hac["mean"] > 0 and g1_hac["p_one_sided"] <= alpha and all(b["lower"] > 0 for b in g1_boot))
    g2 = g2_diff["diff"] > 0 and g2_diff["p_one_sided"] <= alpha and g2_rand["p_upper"] <= alpha
    g3 = all(b["lower"] > 0 for b in g3_boot) and g3_mean_2x > 0
    return {"window": window, "G1_statistical": bool(g1), "G2_calendar": bool(g2), "G3_economic": bool(g3)}


def combined_pnl(days: pd.DataFrame, long_col: str, short_col: str) -> pd.Series:
    """GTF 02 per-day P&L after spread: long into the fix plus the short after it. A day counts
    only if both legs fill (the reversal is one decision)."""
    return days[long_col] + days[short_col]


def combined_net(days: pd.DataFrame, long_col: str, short_col: str, long_price_col: str,
                 usd_per_m_side: float, extra_slip_pips: float = 0.0) -> pd.Series:
    """Combined P&L minus both round trips' commission (each priced at its own entry fill) and slippage."""
    return (combined_pnl(days, long_col, short_col)
            - commission_pips(days[long_price_col], usd_per_m_side)
            - commission_pips(days["entry_price"], usd_per_m_side)
            - 4.0 * extra_slip_pips)


def holdout_gates(hac: dict, diff: dict, net_mean: float, alpha: float = 0.05) -> dict:
    """GTF 002 §4 holdout gates (low power, alpha 0.05)."""
    hg1 = hac["mean"] > 0 and hac["p_one_sided"] <= alpha
    hg2 = diff["diff"] > 0 and diff["p_one_sided"] <= alpha
    hg3 = net_mean > 0
    return {"HG1_statistical": bool(hg1), "HG2_calendar": bool(hg2), "HG3_economic": bool(hg3)}


def verdict(g: dict[str, dict]) -> dict:
    ind, rep = g["INDEPENDENT"], g["REPLICATION"]
    rep_v = "REPLICATES" if rep["G1_statistical"] and rep["G2_calendar"] else "NOT_REPLICATED"
    ind_v = "CONFIRMED" if ind["G1_statistical"] and ind["G2_calendar"] else "NOT_CONFIRMED"
    econ = ind["G3_economic"] and rep["G3_economic"]
    final = "EDGE_CANDIDATE" if ind_v == "CONFIRMED" and econ else "NO_EDGE"
    return {"replication_2016_2025": rep_v, "independent_2010_2015": ind_v,
            "economic_both_windows": bool(econ), "verdict": final}


# ----------------------------------------------------------------------------- synthetic data (tests / smoke run)
def synthetic_ticks(day: dt.date, gotobi: bool, seed: int, effect_pre: float = 2.0, effect_post: float = -3.0,
                    spread_pips: float = 0.3, every_ms: int = 1_000, price: float = 110.0,
                    step_pips: float = 0.03) -> pd.DataFrame:
    """Random-walk mid 00:00-02:00 UTC; on gotobi days a +effect_pre pip drift into 00:55 and
    effect_post pips from 00:55 to 01:30."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, MORNING_END_MS, every_ms, dtype=np.int64)
    steps = rng.normal(0, step_pips, len(t)) * PIP
    mid = price + np.cumsum(steps)
    if gotobi:
        fix, out = 55 * MS_MIN, 90 * MS_MIN
        drift = np.where(t <= fix, effect_pre * t / fix,
                         effect_pre + effect_post * np.clip((t - fix) / (out - fix), 0, 1))
        mid = mid + drift * PIP
    half = spread_pips * PIP / 2
    ts = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(t, unit="ms")
    return pd.DataFrame({"ts": ts.astype("datetime64[ms, UTC]"), "bid": np.round(mid - half, 3),
                         "ask": np.round(mid + half, 3), "bid_vol": np.float32(1.0), "ask_vol": np.float32(1.0)})


def write_synthetic(base: Path, start: dt.date, end: dt.date, seed: int = 7, symbol: str = SYMBOL) -> Path:
    """Write synthetic day files in the downloader layout under base/<symbol>/hours_00-01-02-22-23/."""
    root = Path(base) / symbol / "hours_00-01-02-22-23"
    g = set(gotobi_days(start, end))
    for n, d in enumerate(pd.bdate_range(start, end)):
        d = d.date()
        f = day_file(root, d, symbol)
        f.parent.mkdir(parents=True, exist_ok=True)
        synthetic_ticks(d, d in g and is_jp_bank_day(d), seed + n).to_parquet(f, index=False)
    return root


# ----------------------------------------------------------------------------- small-account leverage (GTF 02 sizing, reported only)
def adverse_excursions(days: pd.DataFrame, long_entry_min: int, fix_min: int, exit_min: int, stop_pips: float) -> pd.DataFrame:
    """Approximate worst open loss of each leg, in pips, from the minute mid path plus half the spread at the fix.

    Long: entry ask minus the lowest mid between entry and the fix. Short: highest mid between the fix and
    the exit minus the entry bid, capped at the stop. Minute marks miss intra-minute spikes, so these are
    lower bounds of the true excursion."""
    half = days["spread_fix_pips"].fillna(0) / 2
    lc = [f"mid_{m:03d}" for m in range(long_entry_min, fix_min + 1)]
    sc = [f"mid_{m:03d}" for m in range(fix_min, exit_min + 1)]
    mae_long = (days[f"long_price_e{long_entry_min}"] - days[lc].min(axis=1)) / PIP + half
    mae_short = ((days[sc].max(axis=1) - days["entry_price"]) / PIP + half).clip(upper=stop_pips)
    return pd.DataFrame({"mae_long": mae_long.clip(lower=0), "mae_short": mae_short.clip(lower=0)}, index=days.index)


def simulate_account(trades: pd.DataFrame, equity0: float, leverage: float, mode: str = "full",
                     usd_per_m_side: float = 35.0, stopout: float = 0.5, min_units: int = 1_000,
                     floor_usd: float = 0.0) -> pd.DataFrame:
    """Day-by-day account for the GTF 02 long+short trade.

    trades: rows in date order with columns long, short (pips after spread), long_px, short_px
    (entry prices), mae_long, mae_short (pips).
    mode 'full': notional = equity x leverage each day (rounded down to min_units);
    mode 'fixed': notional = equity0 x leverage throughout (reduced if margin no longer allows it).
    Margin = notional / leverage. If the open loss of a leg takes equity below stopout x margin, the broker
    liquidates there: the day's loss is equity - stopout x margin and the second leg is not traded.
    The account stops trading when it cannot open min_units or equity <= floor_usd.
    """
    eq_path, units_path, pnl_path, ev = _simulate_arrays(
        trades[["long", "short"]].to_numpy(float), trades[["long_px", "short_px"]].to_numpy(float),
        trades[["mae_long", "mae_short"]].to_numpy(float), equity0, leverage, mode, usd_per_m_side,
        stopout, min_units, floor_usd)
    names = np.array(["ok", "stopout_long", "stopout_short", "cannot_trade"])
    out = pd.DataFrame({"equity": eq_path, "units": units_path, "pnl": pnl_path, "event": names[ev]},
                       index=trades.index)
    out.index.name = "date"
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1
    return out


def _simulate_arrays(pnl, px, mae, equity0, leverage, mode, usd_per_m_side, stopout, min_units, floor_usd):
    n = len(pnl)
    eq_path, units_path, pnl_path = np.empty(n), np.zeros(n), np.zeros(n)
    ev = np.zeros(n, dtype=int)
    eq = float(equity0)
    for k in range(n):
        units = 0.0
        if eq > floor_usd:
            units = eq * leverage if mode == "full" else min(equity0 * leverage, eq * leverage)
            units = np.floor(units / min_units) * min_units
        if units < min_units:
            eq_path[k], ev[k] = eq, 3
            continue
        margin = units / leverage
        day = 0.0
        for leg in (0, 1):
            pip_usd = units * PIP / px[k, leg]
            comm = 2 * usd_per_m_side * units / 1e6
            if eq + day - mae[k, leg] * pip_usd - comm < stopout * margin:
                day = stopout * margin - eq
                ev[k] = 1 + leg
                break
            day += pnl[k, leg] * pip_usd - comm
        eq = max(eq + day, 0.0)
        eq_path[k], units_path[k], pnl_path[k] = eq, units, day
    return eq_path, units_path, pnl_path, ev


def simulate_final_equity(trades: pd.DataFrame, equity0: float, leverage: float, n_sims: int, n_trades: int,
                          seed: int, mode: str = "full", usd_per_m_side: float = 35.0, stopout: float = 0.5) -> np.ndarray:
    """Final equity of n_sims sequences of n_trades days drawn with replacement from `trades`."""
    rng = np.random.default_rng(seed)
    pnl = trades[["long", "short"]].to_numpy(float)
    px = trades[["long_px", "short_px"]].to_numpy(float)
    mae = trades[["mae_long", "mae_short"]].to_numpy(float)
    out = np.empty(n_sims)
    for i in range(n_sims):
        ix = rng.integers(0, len(pnl), n_trades)
        out[i] = _simulate_arrays(pnl[ix], px[ix], mae[ix], equity0, leverage, mode, usd_per_m_side,
                                  stopout, 1_000, 0.0)[0][-1]
    return out


def longest_losing_streak(pnl: pd.Series) -> int:
    best = cur = 0
    for x in pnl.dropna():
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best
