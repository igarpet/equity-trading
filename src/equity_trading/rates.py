"""US short-rate futures from Databento (bbo-1m) for the rate-FX study (RFX 001).

Contracts (CME Globex, GLBX.MDP3), front contract rolled by volume (continuous symbology ``<ROOT>.v.0``):

====  ==========================  ==========================  ==============
root  contract                    tick (price points)         used years
====  ==========================  ==========================  ==============
ZT    2-year T-Note               1/8 of 1/32 = 0.00390625    2016-2025
ZF    5-year T-Note               1/4 of 1/32 = 0.0078125     2016-2025
SR3   3-month SOFR                0.0025 (front) / 0.005      2022-2025 (downloaded from 2022)
====  ==========================  ==========================  ==============

Layout under <EQUITY_TRADING_DIR>/Databento:

* ``raw/bbo-1m/<ROOT>/<YYYY-MM>.dbn.zst``                raw monthly downloads (kept: re-normalising is free)
* ``bbo-1m/<ROOT>/<YYYY>/<ROOT>_<YYYY-MM-DD>.parquet``   one file per UTC day: ts (minute start, UTC), bid_c,
  ask_c, bid_v, ask_v (sizes), instrument_id
* ``ledger.csv``                                          shared with ``futures.py``: every paid request

A bbo-1m record is the best bid and offer sampled once per minute; its minute is the one ending at ts_recv
(``floor(ts_recv - 1 ns)``). The continuous symbol switches contract (instrument_id) at the volume roll; the
price jump between contracts is removed downstream, as for the bond CFDs (``detect_rolls``).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .data import vendor_dir

RATES_VERSION = "0.1.2"   # 0.1.2: SR3 from 2022; months with no tradable symbol skipped; 0.1.1: retries on Databento server errors (504 gateway timeouts)
DATASET = "GLBX.MDP3"
SCHEMA = "bbo-1m"
HOLDOUT_START = dt.date(2026, 1, 1)
ROOTS = {"ZT": {"name": "2-year T-Note", "tick": 1 / 256, "start": 2016},
         "ZF": {"name": "5-year T-Note", "tick": 1 / 128, "start": 2016},
         "SR3": {"name": "3-month SOFR", "tick": 0.005, "start": 2022}}   # listed May 2018; the study uses 2022-2025
COLUMNS = ["ts", "bid_c", "ask_c", "bid_v", "ask_v", "instrument_id"]


class HoldoutError(RuntimeError):
    """Raised when code would download or read 2026 or later."""


def guard(end: dt.date) -> None:
    if end >= HOLDOUT_START:
        raise HoldoutError(f"{end} reaches the sealed holdout (>= {HOLDOUT_START}).")


def root_dir(root: str) -> Path:
    return vendor_dir("Databento") / SCHEMA / root


def raw_dir(root: str) -> Path:
    return vendor_dir("Databento") / "raw" / SCHEMA / root


def day_path(root: str, d: dt.date) -> Path:
    return root_dir(root) / f"{d:%Y}" / f"{root}_{d:%Y-%m-%d}.parquet"


def day_files(root: str, start: dt.date, end: dt.date) -> list[Path]:
    out, d = [], start
    while d <= end:
        f = day_path(root, d)
        if f.exists():
            out.append(f)
        d += dt.timedelta(days=1)
    return out


# ----------------------------------------------------------------------------- normalising
def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Databento bbo-1m ``to_df()`` frame -> COLUMNS, one row per minute (last record of the minute)."""
    ts = pd.to_datetime(raw.index if "ts_recv" not in raw.columns else raw["ts_recv"], utc=True)
    minute = (pd.DatetimeIndex(ts) - pd.Timedelta(1, "ns")).floor("1min")
    df = pd.DataFrame({"ts": minute,
                       "bid_c": raw["bid_px_00"].to_numpy(float), "ask_c": raw["ask_px_00"].to_numpy(float),
                       "bid_v": raw["bid_sz_00"].to_numpy(float), "ask_v": raw["ask_sz_00"].to_numpy(float),
                       "instrument_id": raw["instrument_id"].to_numpy(np.int64)})
    ok = (df["bid_c"] > 0) & (df["ask_c"] >= df["bid_c"]) & np.isfinite(df["bid_c"]) & np.isfinite(df["ask_c"])
    return df[ok].drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


def write_days(df: pd.DataFrame, root: str) -> int:
    if df.empty:
        return 0
    day = df["ts"].dt.date
    n = 0
    for d, part in df.groupby(day):
        if d >= HOLDOUT_START:
            continue
        f = day_path(root, d)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        part.reset_index(drop=True).to_parquet(tmp, index=False)
        tmp.replace(f)
        n += 1
    return n


def normalise_raw_file(path: Path, root: str) -> int:
    import databento as db
    raw = db.DBNStore.from_file(path).to_df()
    return write_days(normalise(raw), root)


# ----------------------------------------------------------------------------- downloading
def months(start_year: int, end_year: int) -> list[tuple[dt.date, dt.date]]:
    out = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            a = dt.date(y, m, 1)
            b = dt.date(y + (m == 12), m % 12 + 1, 1)
            out.append((a, b))
    return out


def retry(fn, *args, tries: int = 6, wait: float = 5.0, log=print, **kw):
    """Call fn, retrying server-side errors (5xx, timeouts) with growing waits. Client errors are raised at once."""
    import time
    for k in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            msg = str(e)
            transient = any(c in msg for c in ("500", "502", "503", "504", "timed out", "Timeout", "Connection"))
            if not transient or k == tries - 1:
                raise
            log(f"  Databento: {msg[:60]} -> retry {k + 1}/{tries - 1} in {wait * (k + 1):.0f} s")
            time.sleep(wait * (k + 1))


def query(root: str, a: dt.date, b: dt.date) -> dict:
    return dict(dataset=DATASET, symbols=[f"{root}.v.0"], stype_in="continuous", schema=SCHEMA,
                start=a.isoformat(), end=b.isoformat())


def download(cl, roots: list[str], start_year: int, end_year: int, budget: float, log=print) -> pd.DataFrame:
    """Buy and normalise month by month. A month whose raw file exists is only re-normalised (free). The quoted
    cost is checked against the shared ledger before each request and logged before the transfer, so an
    interrupted run never under-counts spend or pays twice."""
    from .futures import ledger_path, spent_usd
    guard(dt.date(end_year, 12, 31))
    rows = []
    for root in roots:
        y0 = max(start_year, ROOTS[root]["start"])
        for a, b in months(y0, end_year):
            key = f"{SCHEMA}:{root}:{a:%Y-%m}"
            raw = raw_dir(root) / f"{a:%Y-%m}.dbn.zst"
            if raw.exists():
                n = normalise_raw_file(raw, root)
                rows.append({"key": key, "status": "normalised from raw", "cost_usd": 0.0, "days": n})
                continue
            q = query(root, a, b)
            try:
                cost = float(retry(cl.metadata.get_cost, log=log, **q))
            except Exception as e:
                if "symbology_invalid_request" in str(e) or "could not be resolved" in str(e):
                    rows.append({"key": key, "status": "no symbol that month (not listed yet)", "cost_usd": 0.0, "days": 0})
                    log(f"{key}: no symbol that month, skipped")
                    continue
                raise
            spent = spent_usd()
            if spent + cost > budget:
                rows.append({"key": key, "status": f"stopped: {spent:.2f} + {cost:.2f} > budget {budget:.2f}",
                             "cost_usd": cost, "days": 0})
                log(rows[-1]["status"])
                return pd.DataFrame(rows)
            pd.DataFrame([{"utc": pd.Timestamp.now(tz="UTC").isoformat(), "key": key, "cost_usd": cost,
                           "records": None}]).to_csv(ledger_path(), mode="a", header=not ledger_path().exists(),
                                                      index=False)
            raw.parent.mkdir(parents=True, exist_ok=True)
            tmp = raw.with_suffix(".part")
            try:
                retry(cl.timeseries.get_range, log=log, **q, path=tmp)
            except Exception as e:                       # e.g. no data for that month
                rows.append({"key": key, "status": f"error: {str(e)[:80]}", "cost_usd": cost, "days": 0})
                log(f"{key}: {rows[-1]['status']}")
                continue
            tmp.replace(raw)
            n = normalise_raw_file(raw, root)
            rows.append({"key": key, "status": "downloaded", "cost_usd": cost, "days": n})
            log(f"{key}: ${cost:.3f}, {n} days | ledger total ${spent + cost:.2f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- loading
def load_minutes(root: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """UTC minute series from start-1 to end: bid, ask (forward-filled), tick (a quote in that minute) and
    instrument_id (forward-filled). Same shape as ``fx_seasonality.load_minutes`` plus instrument_id."""
    guard(end)
    files = day_files(root, start - dt.timedelta(days=1), end)
    if not files:
        raise FileNotFoundError(f"No {SCHEMA} files for {root} under {root_dir(root)}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    tick = pd.Series(True, index=df.index).reindex(full, fill_value=False)
    df = df.reindex(full)
    return pd.DataFrame({"bid": df["bid_c"].ffill(), "ask": df["ask_c"].ffill(), "tick": tick.to_numpy(),
                         "instrument_id": df["instrument_id"].ffill()}, index=full)


def detect_rolls(m: pd.DataFrame) -> pd.DataFrame:
    """Contract switches of the continuous symbol: consecutive quoted minutes with a different instrument_id.
    Same columns as the bond CFD switches (gap_bps = mid move across the switch, removed downstream)."""
    q = m[m["tick"]]
    iid = q["instrument_id"].to_numpy()
    i = np.flatnonzero(iid[1:] != iid[:-1]) + 1
    lm = np.log(0.5 * (q["bid"].to_numpy() + q["ask"].to_numpy()))
    rows = [{"quarter": f"{q.index[k]:%Y-%m}", "before": q.index[k - 1], "after": q.index[k],
             "break_minutes": int((q.index[k] - q.index[k - 1]) / pd.Timedelta(minutes=1)),
             "gap_bps": float((lm[k] - lm[k - 1]) * 1e4), "ratio": np.nan, "extra": False,
             "method": "instrument_change", "signature": ""} for k in i]
    return pd.DataFrame(rows, columns=["quarter", "before", "after", "break_minutes", "gap_bps", "ratio", "extra",
                                       "method", "signature"])
