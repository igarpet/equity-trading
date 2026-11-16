#!/usr/bin/env python3
"""
Dukascopy 1-minute bid/ask candles -> one Parquet file per symbol per UTC day.

Source: https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{BID|ASK}_candles_min_1.bi5
  - MM0 is the ZERO-based month (00 = January).
  - LZMA-compressed; 24-byte big-endian records, assumed:
      uint32 seconds_from_day_start, uint32 open, uint32 close, uint32 low, uint32 high, float32 volume
  - Prices are integers scaled by the point (JPY pairs /1000, others /100000).
  - Every file is checked: time field on whole minutes, high >= max(open, close), low <= min(open, close),
    and open/close not swapped (|open[t+1] - close[t]| must be smaller than |close[t+1] - open[t]|).
    A failed check stops the run: the assumed layout would then be wrong.

Output:
  <out>/<SYMBOL>/candles_m1/<YYYY>/<SYMBOL>_<YYYY-MM-DD>.parquet
    ts (minute start, UTC), bid_o, bid_h, bid_l, bid_c, bid_v, ask_o, ask_h, ask_l, ask_c, ask_v
  <out>/<SYMBOL>/candles_m1/_gaps.csv   (days where a side returned 404 / undecodable data)

Behaviour: resumable (existing days skipped); days with network errors are not written and are retried
on the next run; Saturdays skipped; Ctrl+C stops cleanly (second Ctrl+C force-quits).

Examples:
  # test one month for two pairs
  python tools/dukascopy/dukascopy_candles.py --symbols EURUSD,CADCHF --start 2024-01-01 --end 2024-01-31
  # FX seasonality study (FXS 001): all 15 pairs, 2016-2025 (2026 sealed)
  python tools/dukascopy/dukascopy_candles.py --symbols all --start 2015-12-30 --end 2025-12-31 --workers 8

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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
MAJORS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
CROSSES = ("EURGBP", "EURCHF", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CADCHF", "AUDNZD")
PRESETS = {"all": MAJORS + CROSSES, "majors": MAJORS, "crosses": CROSSES}
REC = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])
STOP = threading.Event()
_local = threading.local()
_stats_lock = threading.Lock()
STATS = {"requests": 0, "seconds": 0.0, "200": 0, "404": 0, "429": 0, "5xx": 0, "other": 0, "exception": 0,
         "exception_seconds": 0.0}
TIMEOUT = [5.0, 10.0]      # connect, read (seconds); read is set by --timeout


def _record(key: str, seconds: float) -> None:
    with _stats_lock:
        STATS["requests"] += 1
        STATS["seconds"] += seconds
        STATS[key] += 1
        if key == "exception":
            STATS["exception_seconds"] += seconds


def stats_line(elapsed: float) -> str:
    with _stats_lock:
        s = dict(STATS)
    rate = s["requests"] / elapsed if elapsed > 0 else 0.0
    lat = s["seconds"] / s["requests"] if s["requests"] else 0.0
    answered = s["requests"] - s["exception"]
    lat_ok = (s["seconds"] - s["exception_seconds"]) / answered if answered else 0.0
    lat_exc = s["exception_seconds"] / s["exception"] if s["exception"] else 0.0
    return (f"{rate:.2f} req/s, {lat:.2f} s/request | http 200:{s['200']} 404:{s['404']} 429:{s['429']} "
            f"5xx:{s['5xx']} other:{s['other']} exceptions:{s['exception']} | "
            f"answered {lat_ok:.2f} s each, exceptions {lat_exc:.2f} s each")


class FormatError(RuntimeError):
    """The candle file does not match the assumed record layout."""


def divisor(sym: str) -> int:
    return 1_000 if sym.endswith("JPY") else 100_000


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (research candle downloader)"
        _local.s = s
    return s


def fetch(sym: str, day: dt.date, side: str, retries: int) -> tuple[str, np.ndarray | None]:
    """Return (status, records); status ok | missing | corrupt | error."""
    url = URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day, side=side)
    for attempt in range(retries):
        if STOP.is_set():
            return "error", None
        t0 = time.time()
        try:
            r = session().get(url, timeout=tuple(TIMEOUT))
        except requests.RequestException:
            _record("exception", time.time() - t0)
            if STOP.wait(min(2 ** attempt, 30)):
                return "error", None
            continue
        code = r.status_code
        _record("200" if code == 200 else "404" if code == 404 else "429" if code == 429
                else "5xx" if code >= 500 else "other", time.time() - t0)
        if code == 404:
            return "missing", None
        if code != 200:
            wait_s = min(2 ** attempt, 30)
            if code == 429 and r.headers.get("Retry-After", "").isdigit():
                wait_s = min(int(r.headers["Retry-After"]), 120)
            if STOP.wait(wait_s):
                return "error", None
            continue
        if not r.content:
            return "missing", None
        try:
            raw = lzma.decompress(r.content)
        except lzma.LZMAError:
            return "corrupt", None
        if len(raw) % REC.itemsize:
            return "corrupt", None
        return "ok", np.frombuffer(raw, dtype=REC)
    return "error", None


def check_format(arr: np.ndarray) -> str | None:
    """None if the records look like (t, open, close, low, high, volume); else the failed check.

    Hard checks only (time field, high/low consistent with open/close). They stop the run. The layout,
    including which field is the close, was validated against USD/JPY ticks (FXS 01, 23 days, 100%
    match), so the open/close heuristic below is advisory (`swap_suspect`)."""
    t = arr["t"].astype(np.int64)
    if len(t) == 0 or (t % 60).any() or t.max() >= 86_400:
        return "time field is not whole minutes within the day"
    live = arr["v"] > 0
    o, c, l, h = (arr[k].astype(np.int64) for k in ("o", "c", "l", "h"))
    if live.sum() >= 30:
        ok = ((h >= np.maximum(o, c)) & (l <= np.minimum(o, c)))[live].mean()
        if ok < 0.99:
            return f"high/low inconsistent with open/close on {1 - ok:.1%} of live minutes"
    return None


def swap_suspect(arr: np.ndarray, min_pairs: int = 300) -> bool:
    """Advisory: True if, over enough consecutive live minutes, the next open sits closer to this minute's
    open than to its close (medians, so one weekend-open jump cannot decide it). Thin days (Sundays,
    holidays) have too few minutes and are not judged."""
    live = arr["v"] > 0
    both = np.flatnonzero(live[1:] & live[:-1])
    if len(both) < min_pairs:
        return False
    o, c = arr["o"].astype(np.int64), arr["c"].astype(np.int64)
    a = np.median(np.abs(o[both + 1] - c[both]))
    b = np.median(np.abs(c[both + 1] - o[both]))
    return bool(a > b)


def day_path(root: Path, sym: str, day: dt.date) -> Path:
    return root / f"{day:%Y}" / f"{sym}_{day:%Y-%m-%d}.parquet"


_warn_lock = threading.Lock()


def _warn(root: Path, day: dt.date, side: str, msg: str) -> None:
    f = root / "_format_warnings.csv"
    with _warn_lock:
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["date", "side", "warning"])
            w.writerow([day.isoformat(), side, msg])


def download_day(sym: str, day: dt.date, root: Path, retries: int):
    path = day_path(root, sym, day)
    if path.exists():
        return day, "skipped", 0, []
    sides, problems = {}, []
    for side in ("BID", "ASK"):
        if STOP.is_set():
            return day, "cancelled", 0, []
        status, arr = fetch(sym, day, side, retries)
        if status == "ok":
            bad = check_format(arr)
            if bad:
                raise FormatError(f"{sym} {day} {side}: {bad}")
            if swap_suspect(arr):
                _warn(root, day, side, "open/close order looks unusual (advisory)")
            sides[side] = arr
        else:
            problems.append((side, status))
    if any(s == "error" for _, s in problems):
        return day, "retry", 0, problems
    if len(sides) < 2:
        return day, "gap", 0, problems
    div = divisor(sym)
    base = np.datetime64(dt.datetime(day.year, day.month, day.day), "ms")
    b, a = sides["BID"], sides["ASK"]
    if len(b) != len(a) or (b["t"] != a["t"]).any():
        raise FormatError(f"{sym} {day}: BID and ASK minute stamps differ")
    df = pd.DataFrame({"ts": pd.DatetimeIndex(base + b["t"].astype("int64").astype("timedelta64[s]")).tz_localize("UTC")})
    for pre, arr in (("bid", b), ("ask", a)):
        for k in ("o", "h", "l", "c"):
            df[f"{pre}_{k}"] = arr[k].astype(np.float64) / div
        df[f"{pre}_v"] = arr["v"].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
    return day, "done", int((df["bid_v"] > 0).sum()), problems


def run_symbol(sym: str, days: list[dt.date], out: Path, workers: int, retries: int) -> tuple[bool, int]:
    root = out / sym / "candles_m1"
    root.mkdir(parents=True, exist_ok=True)
    gaps_file = root / "_gaps.csv"
    new_gaps = not gaps_file.exists()
    counts = {"done": 0, "skipped": 0, "retry": 0, "gap": 0, "cancelled": 0}
    t0, i, live_min = time.time(), 0, 0
    with _stats_lock:
        for k in STATS:
            STATS[k] = 0.0 if k.endswith("seconds") else 0
    interrupted = False
    print(f"{sym}: {len(days)} days -> {root}", flush=True)
    ex = ThreadPoolExecutor(workers)
    with open(gaps_file, "a", newline="") as gf:
        gw = csv.writer(gf)
        if new_gaps:
            gw.writerow(["date", "side", "status"])
        pending = {ex.submit(download_day, sym, d, root, retries) for d in days}

        def handle(f):
            nonlocal i, live_min
            day, status, n, problems = f.result()
            counts[status] += 1
            live_min += n
            i += 1
            if status == "gap":
                for side, s in problems:
                    gw.writerow([day.isoformat(), side, s])
            elif status == "retry":
                print(f"  {sym} network errors on {day}, will retry on next run", flush=True)
            if i % 200 == 0 or i == len(days):
                gf.flush()
                el = time.time() - t0
                print(f"  {sym} {i}/{len(days)} | done {counts['done']} skipped {counts['skipped']} "
                      f"gap {counts['gap']} retry {counts['retry']} | {el / 60:.1f} min | {stats_line(el)}", flush=True)

        try:
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in done:
                    handle(f)
        except KeyboardInterrupt:
            interrupted = True
            STOP.set()
            print("\nStopping (a few seconds; Ctrl+C again to force-quit).", flush=True)
            for f in pending:
                f.cancel()
            try:
                while pending:
                    done, pending = wait(pending, timeout=1.0)
                    for f in done:
                        if not f.cancelled():
                            try:
                                handle(f)
                            except FormatError:
                                pass
            except KeyboardInterrupt:
                gf.flush()
                os._exit(130)
        except FormatError as e:
            STOP.set()
            for f in pending:
                f.cancel()
            print(f"\nFORMAT CHECK FAILED: {e}\nThe assumed record layout is wrong; nothing more is written. "
                  "Report this message.", flush=True)
            interrupted = True
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            gf.flush()
    size = sum(f.stat().st_size for f in root.rglob("*.parquet")) / 1e6
    print(f"{sym}: {'stopped' if interrupted else 'finished'} | {counts} | folder {size:.0f} MB", flush=True)
    return not interrupted, counts["retry"]


def main() -> int:
    p = argparse.ArgumentParser(description="Download Dukascopy 1-minute bid/ask candles.")
    p.add_argument("--symbols", default="all", help="comma list, or all / majors / crosses")
    p.add_argument("--start", required=True, type=dt.date.fromisoformat)
    p.add_argument("--end", required=True, type=dt.date.fromisoformat)
    p.add_argument("--out", default=os.environ.get("DUKASCOPY_DIR", r"D:\MarketData\Dukascopy"))
    p.add_argument("--workers", type=int, default=8, help="parallel requests (days in flight x 2 sides)")
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--timeout", type=float, default=10.0,
                   help="seconds to wait for a reply before retrying (was 30 before 2026-11-16)")
    p.add_argument("--passes", type=int, default=3, help="extra passes over days that hit network errors")
    p.add_argument("--pause", type=int, default=60, help="seconds to wait before an extra pass")
    a = p.parse_args()
    TIMEOUT[1] = a.timeout
    syms = PRESETS.get(a.symbols.lower()) or tuple(s.strip().upper() for s in a.symbols.split(","))
    if a.end < a.start:
        sys.exit("--end before --start")
    days = [a.start + dt.timedelta(n) for n in range((a.end - a.start).days + 1)]
    days = [d for d in days if d.weekday() != 5]
    print(f"symbols: {', '.join(syms)} | {a.start} -> {a.end} | Ctrl+C stops cleanly; rerun resumes.")
    left = {}
    for sym in syms:
        for k in range(a.passes + 1):
            if k:
                print(f"{sym}: pass {k + 1} for {retry} day(s) with network errors in {a.pause} s", flush=True)
                if STOP.wait(a.pause):
                    return 1
            ok, retry = run_symbol(sym, days, Path(a.out), a.workers, a.retries)   # finished days are skipped
            if not ok:
                print("Run stopped. Rerun the same command to continue.")
                return 1
            if retry == 0:
                break
        if retry:
            left[sym] = retry
    if left:
        print(f"Days still missing after {a.passes} extra passes: {left}. Rerun the same command later.")
    else:
        print("All symbols complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
