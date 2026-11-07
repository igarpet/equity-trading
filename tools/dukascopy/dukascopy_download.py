#!/usr/bin/env python3
"""
Dukascopy historical tick downloader (bid/ask) -> daily Parquet files.

Source: https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
  - MM0 is the ZERO-based month (00 = January).
  - Each .bi5 is LZMA-compressed; records are 20 bytes big-endian:
      uint32 ms_since_hour_start, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol
  - Prices are integers scaled by the instrument point (USDJPY: /1000).
  - Volumes are in millions of base currency (Dukascopy SWFX liquidity, not market-wide).
  - All timestamps are UTC. 9:55 JST == 00:55 UTC (Japan has no DST).

Output layout:
  <out>/<SYMBOL>/<hours_tag>/<YYYY>/<SYMBOL>_<YYYY-MM-DD>.parquet
  <out>/<SYMBOL>/<hours_tag>/_gaps.csv   (hours that returned 404 / corrupt payloads)

Behaviour:
  - Resumable: existing day files are skipped.
  - Days with network errors are NOT written, so a rerun retries them.
  - 404 / undecodable hours are permanent gaps: the day is written without them and
    the gap is logged in _gaps.csv. Treat gap days as missing in the backtest.
  - Saturdays (UTC) are skipped; the FX market is closed.

Examples:
  # Full days, 2010-2025 (2026 left out as holdout)
  python dukascopy_download.py --symbol USDJPY --start 2010-01-01 --end 2025-12-31

  # Only the Tokyo-morning window (22:00-02:59 UTC), much smaller
  python dukascopy_download.py --symbol USDJPY --start 2010-01-01 --end 2025-12-31 --hours 22-2

  # Test one month first to measure size and speed
  python dukascopy_download.py --symbol USDJPY --start 2024-01-01 --end 2024-01-31

Requirements: Python 3.10+, numpy, pandas, pyarrow, requests
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import lzma
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"

# Price divisor = 1 / point size. Add symbols as needed.
DIVISOR = {
    "USDJPY": 1_000, "EURJPY": 1_000, "GBPJPY": 1_000, "AUDJPY": 1_000,
    "CADJPY": 1_000, "CHFJPY": 1_000, "NZDJPY": 1_000,
    "EURUSD": 100_000, "GBPUSD": 100_000, "AUDUSD": 100_000, "NZDUSD": 100_000,
    "USDCHF": 100_000, "USDCAD": 100_000,
}

REC = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                ("ask_vol", ">f4"), ("bid_vol", ">f4")])

EMPTY = pd.DataFrame({
    "ts": pd.Series([], dtype="datetime64[ms, UTC]"),
    "bid": pd.Series([], dtype="float64"),
    "ask": pd.Series([], dtype="float64"),
    "bid_vol": pd.Series([], dtype="float32"),
    "ask_vol": pd.Series([], dtype="float32"),
})

_local = threading.local()


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (research tick downloader)"
        _local.s = s
    return s


def parse_hours(spec: str) -> list[int]:
    """'all' | '0-2' | '22-2' (wraps midnight) | '22,23,0,1,2'."""
    if spec == "all":
        return list(range(24))
    hours: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            hours += list(range(a, b + 1)) if a <= b else list(range(a, 24)) + list(range(0, b + 1))
        else:
            hours.append(int(part))
    if any(not 0 <= h <= 23 for h in hours):
        raise ValueError(f"bad hours spec: {spec}")
    return sorted(set(hours))


def hours_tag(hours: list[int]) -> str:
    return "hours_all" if len(hours) == 24 else "hours_" + "-".join(f"{h:02d}" for h in hours)


def fetch_hour(sym: str, day: dt.date, hour: int, retries: int) -> tuple[str, np.ndarray | None]:
    """Return (status, records). status: ok | empty | missing | corrupt | error."""
    url = URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day, h=hour)
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=30)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code == 404:
            return "missing", None
        if r.status_code != 200:          # 429/5xx: back off and retry
            time.sleep(min(2 ** attempt, 30))
            continue
        if not r.content:
            return "empty", None          # no ticks in that hour (normal around weekends)
        try:
            raw = lzma.decompress(r.content)
        except lzma.LZMAError:
            return "corrupt", None
        if len(raw) % REC.itemsize:
            return "corrupt", None
        return "ok", np.frombuffer(raw, dtype=REC)
    return "error", None


def to_frame(arr: np.ndarray, day: dt.date, hour: int, divisor: int) -> pd.DataFrame:
    base = np.datetime64(dt.datetime(day.year, day.month, day.day, hour), "ms")
    ts = base + arr["ms"].astype("int64").astype("timedelta64[ms]")
    return pd.DataFrame({
        "ts": pd.DatetimeIndex(ts).tz_localize("UTC"),
        "bid": arr["bid"].astype(np.float64) / divisor,
        "ask": arr["ask"].astype(np.float64) / divisor,
        "bid_vol": arr["bid_vol"].astype(np.float32),
        "ask_vol": arr["ask_vol"].astype(np.float32),
    })


def day_path(root: Path, sym: str, day: dt.date) -> Path:
    return root / f"{day:%Y}" / f"{sym}_{day:%Y-%m-%d}.parquet"


def download_day(sym: str, day: dt.date, hours: list[int], root: Path,
                 divisor: int, retries: int) -> tuple[dt.date, str, int, list[tuple[int, str]]]:
    path = day_path(root, sym, day)
    if path.exists():
        return day, "skipped", 0, []

    frames, problems = [], []
    for h in hours:
        status, arr = fetch_hour(sym, day, h, retries)
        if status == "ok":
            frames.append(to_frame(arr, day, h, divisor))
        elif status != "empty":
            problems.append((h, status))

    if any(s == "error" for _, s in problems):
        return day, "retry", 0, problems  # do not write; rerun will retry this day

    df = pd.concat(frames, ignore_index=True) if frames else EMPTY.copy()
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
    return day, "done", len(df), problems


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*.parquet"))


def main() -> int:
    p = argparse.ArgumentParser(description="Download Dukascopy tick data to daily Parquet.")
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--start", required=True, type=dt.date.fromisoformat)
    p.add_argument("--end", required=True, type=dt.date.fromisoformat)
    p.add_argument("--out", default=os.environ.get("DUKASCOPY_DIR", r"D:\MarketData\Dukascopy"))
    p.add_argument("--hours", default="all", help="'all', '0-2', '22-2', or '22,23,0,1,2' (UTC)")
    p.add_argument("--workers", type=int, default=4, help="parallel days; keep low to avoid throttling")
    p.add_argument("--retries", type=int, default=6)
    a = p.parse_args()

    sym = a.symbol.upper()
    if sym not in DIVISOR:
        sys.exit(f"Unknown point size for {sym}; add it to DIVISOR.")
    if a.end < a.start:
        sys.exit("--end before --start")

    hours = parse_hours(a.hours)
    root = Path(a.out) / sym / hours_tag(hours)
    root.mkdir(parents=True, exist_ok=True)
    gaps_file = root / "_gaps.csv"
    new_gaps_file = not gaps_file.exists()

    days = [a.start + dt.timedelta(n) for n in range((a.end - a.start).days + 1)]
    days = [d for d in days if d.weekday() != 5]  # skip Saturdays

    print(f"{sym}  {a.start} -> {a.end}  {len(days)} days  hours={a.hours}  -> {root}")
    t0, counts, ticks = time.time(), {"done": 0, "skipped": 0, "retry": 0}, 0

    with open(gaps_file, "a", newline="") as gf, ThreadPoolExecutor(a.workers) as ex:
        gw = csv.writer(gf)
        if new_gaps_file:
            gw.writerow(["date", "hour_utc", "status"])
        futs = [ex.submit(download_day, sym, d, hours, root, DIVISOR[sym], a.retries) for d in days]
        for i, f in enumerate(as_completed(futs), 1):
            day, status, n, problems = f.result()
            counts[status] += 1
            ticks += n
            if status == "done":
                for h, s in problems:
                    gw.writerow([day.isoformat(), h, s])
            elif status == "retry":
                print(f"  network errors on {day}, will retry on next run: {problems}")
            if i % 50 == 0 or i == len(futs):
                gf.flush()
                el = time.time() - t0
                print(f"  {i}/{len(futs)} days | done {counts['done']} skipped {counts['skipped']} "
                      f"retry {counts['retry']} | {ticks:,} ticks | {el/60:.1f} min")

    size = dir_size(root)
    print(f"\nFinished. New ticks: {ticks:,}. Folder size: {size/1e9:.2f} GB")
    if counts["retry"]:
        print(f"{counts['retry']} days had network errors -- rerun the same command to fill them.")
    print(f"Permanent gaps logged in {gaps_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
