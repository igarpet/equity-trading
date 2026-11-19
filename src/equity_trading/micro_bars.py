"""CME micro index futures 1-minute trade bars from Databento (ohlcv-1m) for the MNQ return-prediction study.

Micro contracts only (MNQ, MES, MYM; FOF 002). Front contract rolled by volume (continuous ``<ROOT>.v.0``).
Databento has no 5-minute schema: 1-minute bars are bought and resampled here (``to_bars``).

Layout under <EQUITY_TRADING_DIR>/Databento:

* ``raw/ohlcv-1m/<ROOT>/<YYYY-MM>.dbn.zst``   raw monthly downloads (kept: re-normalising is free)
* ``ohlcv-1m/<ROOT>/<ROOT>_<YYYY-MM>.parquet`` one file per UTC month: ts, open, high, low, close, volume,
  instrument_id
* ``ledger.csv``                               shared with ``futures.py`` and ``rates.py``: every paid request

Conventions:

* ``ts`` is the bar OPEN time in UTC (Databento ``ts_event``); the bar covers [ts, ts + 1 min).
* A minute with no trade has no bar (no forward fill here).
* Prices are raw, not back-adjusted. The continuous symbol switches ``instrument_id`` at the roll; ``to_bars``
  never builds a bar across two contracts and flags the first bar after a switch (``roll``), whose return
  must be dropped.
* Nothing from 2026 is downloaded or loaded (sealed holdout).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .data import vendor_dir
from .futures import assert_micro
from .rates import months, retry

MICRO_BARS_VERSION = "0.1.0"
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
HOLDOUT_START = dt.date(2026, 1, 1)
LISTED = {"MNQ": 2019, "MES": 2019, "MYM": 2019}      # micro index futures listed 2019-05-06
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "instrument_id"]


class HoldoutError(RuntimeError):
    """Raised when code would download or read 2026 or later."""


def guard(end: dt.date) -> None:
    """``end`` is exclusive."""
    if end > HOLDOUT_START:
        raise HoldoutError(f"{end} reaches the sealed holdout (>= {HOLDOUT_START}).")


def root_dir(root: str) -> Path:
    assert_micro(root)
    return vendor_dir("Databento") / SCHEMA / root


def raw_dir(root: str) -> Path:
    assert_micro(root)
    return vendor_dir("Databento") / "raw" / SCHEMA / root


def month_path(root: str, a: dt.date) -> Path:
    return root_dir(root) / f"{root}_{a:%Y-%m}.parquet"


def query(root: str, a: dt.date, b: dt.date) -> dict:
    assert_micro(root)
    guard(b)
    return dict(dataset=DATASET, symbols=[f"{root}.v.0"], stype_in="continuous", schema=SCHEMA,
                start=a.isoformat(), end=b.isoformat())


# ----------------------------------------------------------------------------- normalising

def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Databento ohlcv-1m ``to_df()`` frame -> COLUMNS (bar open time, UTC)."""
    ts = raw["ts_event"] if "ts_event" in raw.columns else raw.index
    df = pd.DataFrame({"ts": pd.to_datetime(np.asarray(ts), utc=True),
                       "open": raw["open"].to_numpy(float), "high": raw["high"].to_numpy(float),
                       "low": raw["low"].to_numpy(float), "close": raw["close"].to_numpy(float),
                       "volume": raw["volume"].to_numpy(np.int64),
                       "instrument_id": raw["instrument_id"].to_numpy(np.int64)})
    ok = (df["close"] > 0) & np.isfinite(df["close"]) & (df["high"] >= df["low"])
    return df[ok].drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


def normalise_raw_file(path: Path, root: str, a: dt.date) -> int:
    import databento as db
    df = normalise(db.DBNStore.from_file(path).to_df())
    lo, hi = pd.Timestamp(a, tz="UTC"), pd.Timestamp(a, tz="UTC") + pd.offsets.MonthBegin(1)
    df = df[(df["ts"] >= lo) & (df["ts"] < hi)].reset_index(drop=True)
    f = month_path(root, a)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(f)
    return len(df)


# ----------------------------------------------------------------------------- quoting and downloading

def _no_symbol(e: Exception) -> bool:
    s = str(e)
    return "symbology_invalid_request" in s or "could not be resolved" in s


def quote(cl, roots: list[str], start_year: int, end_year: int, log=print) -> pd.DataFrame:
    """Quoted cost and billable size per root and year, for months not yet downloaded. Free calls."""
    guard(dt.date(end_year + 1, 1, 1))
    rows = []
    for root in roots:
        for y in range(max(start_year, LISTED[root]), end_year + 1):
            todo = [(a, b) for a, b in months(y, y) if not (raw_dir(root) / f"{a:%Y-%m}.dbn.zst").exists()]
            if y == 2019:                                  # listed in May: earlier months have no symbol
                todo = [(a, b) for a, b in todo if a >= dt.date(2019, 5, 1)]
            if not todo:
                continue
            q = query(root, todo[0][0], todo[-1][1])
            try:
                usd = float(retry(cl.metadata.get_cost, log=log, **q))
                gb = int(retry(cl.metadata.get_billable_size, log=log, **q)) / 1e9
            except Exception as e:
                if _no_symbol(e):
                    log(f"  {root} {y}: no symbol, skipped")
                    continue
                raise
            rows.append({"root": root, "year": y, "months": len(todo), "gb": gb, "usd": usd})
            log(f"  {root} {y}: {len(todo)} months, {gb * 1e3:.1f} MB, ${usd:.3f}")
    return pd.DataFrame(rows, columns=["root", "year", "months", "gb", "usd"])


def download(cl, roots: list[str], start_year: int, end_year: int, budget: float, log=print) -> pd.DataFrame:
    """Buy and normalise month by month (same discipline as ``rates.download``): a month whose raw file exists is
    only re-normalised (free); the quoted cost is checked against the shared ledger and logged before the
    transfer, so an interrupted run never under-counts spend or pays twice."""
    from .futures import ledger_path, spent_usd
    guard(dt.date(end_year + 1, 1, 1))
    rows = []
    for root in roots:
        for a, b in months(max(start_year, LISTED[root]), end_year):
            if a < dt.date(2019, 5, 1):
                continue
            key = f"{SCHEMA}:{root}:{a:%Y-%m}"
            raw = raw_dir(root) / f"{a:%Y-%m}.dbn.zst"
            if raw.exists():
                n = normalise_raw_file(raw, root, a)
                rows.append({"key": key, "status": "normalised from raw", "cost_usd": 0.0, "bars": n})
                continue
            q = query(root, a, b)
            try:
                cost = float(retry(cl.metadata.get_cost, log=log, **q))
            except Exception as e:
                if _no_symbol(e):
                    rows.append({"key": key, "status": "no symbol that month", "cost_usd": 0.0, "bars": 0})
                    log(f"{key}: no symbol that month, skipped")
                    continue
                raise
            spent = spent_usd()
            if spent + cost > budget:
                rows.append({"key": key, "status": f"stopped: {spent:.2f} + {cost:.2f} > budget {budget:.2f}",
                             "cost_usd": cost, "bars": 0})
                log(rows[-1]["status"])
                return pd.DataFrame(rows)
            pd.DataFrame([{"utc": pd.Timestamp.now(tz="UTC").isoformat(), "key": key, "cost_usd": cost,
                           "records": None}]).to_csv(ledger_path(), mode="a", header=not ledger_path().exists(),
                                                      index=False)
            raw.parent.mkdir(parents=True, exist_ok=True)
            tmp = raw.with_suffix(".part")
            try:
                retry(cl.timeseries.get_range, log=log, **q, path=tmp)
            except Exception as e:
                rows.append({"key": key, "status": f"error: {str(e)[:80]}", "cost_usd": cost, "bars": 0})
                log(f"{key}: {rows[-1]['status']}")
                continue
            tmp.replace(raw)
            n = normalise_raw_file(raw, root, a)
            rows.append({"key": key, "status": "downloaded", "cost_usd": cost, "bars": n})
            log(f"{key}: ${cost:.3f}, {n} bars | ledger total ${spent + cost:.2f}")
    return pd.DataFrame(rows, columns=["key", "status", "cost_usd", "bars"])


# ----------------------------------------------------------------------------- loading

def load_1m(root: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """1-minute bars in [start, end) (UTC dates, end exclusive), indexed by bar open time."""
    guard(end)
    parts = []
    for a, _ in months(start.year, (end - dt.timedelta(days=1)).year):
        f = month_path(root, a)
        if f.exists():
            parts.append(pd.read_parquet(f))
    if not parts:
        raise FileNotFoundError(f"No {SCHEMA} files for {root} under {root_dir(root)}")
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return df[(df["ts"] >= lo) & (df["ts"] < hi)].set_index("ts").sort_index()


def to_bars(m1: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Resample 1-minute bars to ``minutes`` bars on the UTC clock grid, never across a contract switch.

    Index = bar open time. ``log_ret`` = log(close / previous bar close) within the same contract; NaN on the
    first bar after a switch (``roll`` = True) and on the first bar overall. ``minutes_traded`` = number of
    1-minute bars inside (use it to drop thin bars, e.g. around the CME daily break).
    """
    if m1.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "minutes_traded", "instrument_id",
                                     "roll", "log_ret"])
    rule = f"{minutes}min"
    key = [pd.Grouper(freq=rule, label="left", closed="left"), "instrument_id"]
    g = m1.reset_index().set_index("ts").groupby(key)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(), "minutes_traded": g["close"].size()})
    b = b.reset_index().sort_values(["ts", "volume"])
    # a bucket that straddles the switch has two rows: keep the contract that traded more in it
    b = b.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    iid = b["instrument_id"].to_numpy()
    roll = np.r_[False, iid[1:] != iid[:-1]]
    ret = np.log(b["close"]).diff().to_numpy(copy=True)
    ret[roll] = np.nan
    b["roll"] = roll
    b["log_ret"] = ret
    return b
