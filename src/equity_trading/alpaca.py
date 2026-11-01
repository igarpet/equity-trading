"""Alpaca market-data client: historical stock bars (SIP), cached to Parquet.

Credentials come from .env (APCA_API_KEY_ID, APCA_API_SECRET_KEY). On the free
plan SIP queries must end at least 15 minutes in the past; this module only
requests whole past calendar years or explicit past end dates.

Cache layout: <EQUITY_TRADING_DIR>/alpaca/bars/<timeframe>/<SYMBOL>/<YEAR>.parquet
(all bars Alpaca returns, extended hours included, UTC bar-start timestamps).
A year file is written only after the full year was downloaded, so an
interrupted run resumes cleanly. The current year is never cached as complete.
"""
from __future__ import annotations

import os
import time
from datetime import date

import numpy as np
import pandas as pd

from .data import NY, data_dir

ALPACA_VERSION = "0.1.0"
BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"
MAX_CALLS_PER_MIN = 180  # free plan limit is 200


def _credentials() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (check .env in the repo root).")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


class _RateLimiter:
    def __init__(self, per_min: int):
        self.gap = 60.0 / per_min
        self.last = 0.0

    def wait(self):
        dt = time.monotonic() - self.last
        if dt < self.gap:
            time.sleep(self.gap - dt)
        self.last = time.monotonic()


_limiter = _RateLimiter(MAX_CALLS_PER_MIN)


def fetch_bars(symbols: list[str], timeframe: str, start: str, end: str,
               adjustment: str = "all", feed: str = "sip", session=None,
               max_retries: int = 6) -> dict[str, pd.DataFrame]:
    """Download bars for several symbols between start and end (RFC-3339 or YYYY-MM-DD).

    Returns {symbol: DataFrame[t (UTC), o, h, l, c, v, n, vw]}.
    """
    import requests
    session = session or requests.Session()
    headers = _credentials()
    params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
              "adjustment": adjustment, "feed": feed, "limit": 10000, "sort": "asc"}
    out: dict[str, list] = {s: [] for s in symbols}
    token = None
    while True:
        if token:
            params["page_token"] = token
        for attempt in range(max_retries):
            _limiter.wait()
            r = session.get(BASE_URL, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:300]}")
        else:
            raise RuntimeError(f"Alpaca: gave up after {max_retries} retries ({r.status_code}).")
        js = r.json()
        for sym, rows in (js.get("bars") or {}).items():
            out.setdefault(sym, []).extend(rows)
        token = js.get("next_page_token")
        if not token:
            break
    res = {}
    for sym, rows in out.items():
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "n", "vw"])
        df["t"] = pd.to_datetime(df["t"], utc=True)
        res[sym] = df
    return res


def _cache_dir(timeframe: str) -> "os.PathLike":
    d = data_dir() / "alpaca" / "bars" / timeframe
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_years(symbols: list[str], timeframe: str, years: list[int], batch: int = 10,
                   verbose: bool = True) -> None:
    """Fill the cache for complete past years. Skips symbol-years already cached."""
    this_year = date.today().year
    for y in years:
        if y >= this_year:
            raise ValueError(f"{y} is not a complete past year; download_years only caches complete years.")
        todo = [s for s in symbols if not (_cache_dir(timeframe) / s / f"{y}.parquet").exists()]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            t0 = time.time()
            got = fetch_bars(chunk, timeframe, f"{y}-01-01", f"{y + 1}-01-01")
            for s in chunk:
                p = _cache_dir(timeframe) / s
                p.mkdir(parents=True, exist_ok=True)
                got.get(s, pd.DataFrame(columns=["t"])).to_parquet(p / f"{y}.parquet", index=False)
            if verbose:
                n = sum(len(got.get(s, [])) for s in chunk)
                print(f"{timeframe} {y} {chunk[0]}..{chunk[-1]}: {n:,} bars in {time.time() - t0:.0f}s")


def load_bars(symbol: str, timeframe: str, years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = _cache_dir(timeframe) / symbol / f"{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run download_years first.")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df


def regular_session_panel(symbols: list[str], timeframe: str, years: list[int],
                          field: str = "c", ffill_limit: int = 2) -> pd.DataFrame:
    """Wide panel (New York time index, one column per symbol) of regular-session bars.

    A bar is regular-session if it starts at or after 09:30 and before 16:00 New York
    time. Missing bars are forward-filled within the same session up to ffill_limit
    bars (no fill across days); remaining gaps stay NaN.
    """
    cols = {}
    for s in symbols:
        b = load_bars(s, timeframe, years)
        if b.empty:
            continue
        t = b["t"].dt.tz_convert(NY)
        m = t.dt.hour * 60 + t.dt.minute
        keep = (m >= 570) & (m < 960)
        cols[s] = pd.Series(b.loc[keep, field].to_numpy(dtype=float), index=pd.DatetimeIndex(t[keep]))
    panel = pd.DataFrame(cols).sort_index()
    if ffill_limit:
        day = panel.index.date
        panel = panel.groupby(day, group_keys=False).apply(lambda g: g.ffill(limit=ffill_limit))
    return panel


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: first/last bar, sessions, share of panel bars that are NaN."""
    days = pd.Series(panel.index.date, index=panel.index)
    return pd.DataFrame({
        "first": panel.apply(lambda c: c.first_valid_index()),
        "last": panel.apply(lambda c: c.last_valid_index()),
        "sessions": panel.notna().groupby(days.values).any().sum(),
        "nan_share": panel.isna().mean().round(4),
    })


def synthetic_panel(universe: dict[str, list[str]], etfs: dict[str, str], years: list[int],
                    bars_per_day: int = 13, seed: int = 0) -> pd.DataFrame:
    """Offline stand-in for regular_session_panel.

    Each sector has a random-walk factor; its ETF tracks the factor; each stock is
    beta * factor + AR(1) idiosyncratic term, so same-sector stock pairs and
    stock-ETF pairs are cointegrated by construction and cross-sector pairs are not.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    step = 390 // bars_per_day
    offs = pd.to_timedelta(570 + step * np.arange(bars_per_day), unit="min")
    idx = pd.DatetimeIndex([d.tz_localize(NY) + o for d in days for o in offs])
    n = len(idx)
    tod = np.tile(np.arange(bars_per_day), len(days))
    u = 1.0 + 1.0 * ((tod - (bars_per_day - 1) / 2) / ((bars_per_day - 1) / 2)) ** 2
    u[tod == 0] *= 2.5
    from scipy.signal import lfilter
    out = {}
    for sector, names in universe.items():
        f = np.cumsum(rng.standard_normal(n) * 0.0025 * u)
        out[etfs[sector]] = 100 * np.exp(f + rng.standard_normal(n) * 1e-4)
        for s in names:
            phi = rng.uniform(0.97, 0.995)
            idio = lfilter([1.0], [1.0, -phi], rng.standard_normal(n) * 0.0015 * u)
            out[s] = 100 * np.exp(rng.uniform(0.7, 1.3) * f + idio + rng.standard_normal(n) * 2e-4)
    return pd.DataFrame(out, index=idx)
