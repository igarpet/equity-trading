"""Alpaca market-data client: historical stock bars (SIP), cached to Parquet.

Credentials come from .env (APCA_API_KEY_ID, APCA_API_SECRET_KEY). On the free
plan SIP queries must end at least 15 minutes in the past; this module only
requests whole past calendar years or explicit past end dates.

Cache layout: <EQUITY_TRADING_DIR>/Alpaca/bars/<timeframe>/<SYMBOL>/<YEAR>.parquet
(Windows-reserved symbols such as CON use CON_/; see data.safe_name)
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

from .data import NY, safe_name, vendor_dir

ALPACA_VERSION = "0.4.0"
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
    """Thread-safe: spaces request starts at least 60/per_min seconds apart across all threads."""

    def __init__(self, per_min: int):
        import threading
        self.gap = 60.0 / per_min
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.gap
        if start > now:
            time.sleep(start - now)


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


def _cache_dir(timeframe: str, adjustment: str = "all") -> "os.PathLike":
    """bars/<timeframe>/ for split+dividend adjusted bars (default), bars/<timeframe>_<adjustment>/ otherwise
    (e.g. 1Day_raw: unadjusted, for market capitalisation = price x shares reported at the time)."""
    d = vendor_dir("Alpaca") / "bars" / (timeframe if adjustment == "all" else f"{timeframe}_{adjustment}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_chunk(chunk: list[str], timeframe: str, y: int, adjustment: str = "all") -> int:
    import requests
    got = fetch_bars(chunk, timeframe, f"{y}-01-01", f"{y + 1}-01-01", adjustment=adjustment,
                     session=requests.Session())
    n = 0
    for s in chunk:
        p = _cache_dir(timeframe, adjustment) / safe_name(s)
        p.mkdir(parents=True, exist_ok=True)
        df = got.get(s, pd.DataFrame(columns=["t"]))
        tmp = p / f"{y}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p / f"{y}.parquet")  # atomic: a file exists only once fully written
        n += len(df)
    return n


def download_years(symbols: list[str], timeframe: str, years: list[int], batch: int = 10,
                   workers: int = 8, verbose: bool = True, adjustment: str = "all") -> None:
    """Fill the cache for complete past years with `workers` parallel requests (shared rate
    limit, MAX_CALLS_PER_MIN). Skips symbol-years already cached; safe to interrupt and re-run."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    this_year = date.today().year
    bad = [y for y in years if y >= this_year]
    if bad:
        raise ValueError(f"{bad} not complete past years; download_years only caches complete years.")
    tasks = []
    for y in years:
        todo = [s for s in symbols if not (_cache_dir(timeframe, adjustment) / safe_name(s) / f"{y}.parquet").exists()]
        tasks += [(todo[i:i + batch], y) for i in range(0, len(todo), batch)]
    if verbose:
        print(f"{timeframe} ({adjustment}): {len(tasks)} batches to download ({sum(len(c) for c, _ in tasks)} symbol-years), "
              f"{workers} parallel requests")
    if not tasks:
        return
    t0, done, bars = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_chunk, c, timeframe, y, adjustment) for c, y in tasks]
        for f in as_completed(futs):
            bars += f.result()
            done += 1
            if verbose and (done % 10 == 0 or done == len(tasks)):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} batches, {bars:,} bars, {el / 60:.1f} min elapsed, "
                      f"~{el / done * (len(tasks) - done) / 60:.0f} min left", flush=True)


def load_bars(symbol: str, timeframe: str, years: list[int], adjustment: str = "all") -> pd.DataFrame:
    parts = []
    for y in years:
        p = _cache_dir(timeframe, adjustment) / safe_name(symbol) / f"{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run download_years first.")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df


def session_schedule(years: list[int]) -> pd.DataFrame:
    """NYSE sessions (index: session date) with market_open / market_close in UTC."""
    import pandas_market_calendars as mcal
    sc = mcal.get_calendar("XNYS").schedule(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    sc.index = pd.DatetimeIndex(sc.index).date
    return sc[["market_open", "market_close"]]


def in_regular_session(t_utc: pd.Series, sched: pd.DataFrame) -> np.ndarray:
    """True where the bar start lies in [open, close) of its NYSE session."""
    t = pd.to_datetime(t_utc, utc=True)
    d = t.dt.tz_convert(NY).dt.date
    op = d.map(sched["market_open"]); cl = d.map(sched["market_close"])
    ok = op.notna() & cl.notna()
    res = np.zeros(len(t), dtype=bool)
    res[ok.to_numpy()] = ((t[ok] >= pd.to_datetime(op[ok], utc=True)) & (t[ok] < pd.to_datetime(cl[ok], utc=True))).to_numpy()
    return res


def regular_session_panel(symbols: list[str], timeframe: str, years: list[int],
                          field: str = "c", ffill_limit: int = 2) -> pd.DataFrame:
    """Wide panel (New York time index, one column per symbol) of regular-session bars.

    A bar is regular-session if it starts at or after the NYSE open and before the NYSE
    close of its trading day (exchange calendar, so early-close days end at 13:00 and
    non-trading days are dropped). Missing bars are forward-filled within the same session
    up to ffill_limit bars (no fill across days); remaining gaps stay NaN.
    """
    sched = session_schedule(years)
    cols = {}
    for s in symbols:
        b = load_bars(s, timeframe, years)
        if b.empty:
            continue
        keep = in_regular_session(b["t"], sched)
        t = b["t"].dt.tz_convert(NY)
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


# ------------------------------------------------------------------ quotes (S05b cost sampling)
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"


def fetch_quotes(symbols: list[str], start: str, end: str, feed: str = "sip", session=None,
                 max_retries: int = 6) -> dict[str, pd.DataFrame]:
    """Historical NBBO quotes for several symbols in [start, end) (RFC-3339).

    Returns {symbol: DataFrame[t (UTC), bp, ap]} (bid, ask). Paginates like fetch_bars."""
    import requests
    session = session or requests.Session()
    headers = _credentials()
    params = {"symbols": ",".join(symbols), "start": start, "end": end, "feed": feed, "limit": 10000, "sort": "asc"}
    out: dict[str, list] = {s: [] for s in symbols}
    token = None
    while True:
        if token:
            params["page_token"] = token
        for attempt in range(max_retries):
            _limiter.wait()
            r = session.get(QUOTES_URL, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            raise RuntimeError(f"Alpaca quotes {r.status_code}: {r.text[:300]}")
        else:
            raise RuntimeError(f"Alpaca quotes: gave up after {max_retries} retries ({r.status_code}).")
        js = r.json()
        for sym, rows in (js.get("quotes") or {}).items():
            out.setdefault(sym, []).extend((q.get("t"), q.get("bp"), q.get("ap")) for q in rows)
        token = js.get("next_page_token")
        if not token:
            break
    res = {}
    for sym, rows in out.items():
        df = pd.DataFrame(rows, columns=["t", "bp", "ap"])
        df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")  # quote stamps mix whole and fractional seconds
        res[sym] = df
    return res


def summarize_quotes(q: dict[str, pd.DataFrame], max_rel_spread: float = 0.02) -> pd.DataFrame:
    """Per symbol: median relative half-spread (ap - bp) / (ap + bp) over valid quotes in the window
    (0 < bp < ap, relative spread below max_rel_spread), and the number of valid quotes."""
    rows = []
    for sym, df in q.items():
        if df.empty:
            rows.append((sym, np.nan, 0))
            continue
        bp, ap = df["bp"].to_numpy(float), df["ap"].to_numpy(float)
        ok = (bp > 0) & (ap > bp)
        hs = (ap - bp) / (ap + bp)
        ok &= 2 * hs < max_rel_spread
        rows.append((sym, float(np.median(hs[ok])) if ok.any() else np.nan, int(ok.sum())))
    return pd.DataFrame(rows, columns=["symbol", "half_spread", "n_quotes"])


def sample_quote_spreads(symbols: list[str], session_date, slot_starts_utc: list, window_s: float = 3.0,
                         chunk: int = 100, cache_dir=None, verbose: bool = False) -> pd.DataFrame:
    """Quoted half-spreads for every symbol in the first `window_s` seconds after each bar start of
    one session. Cached per session as <cache_dir>/<YYYY-MM-DD>.parquet (written only when complete)."""
    import requests
    p = None
    if cache_dir is not None:
        p = cache_dir / f"{pd.Timestamp(session_date).date()}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    sess = requests.Session()
    parts = []
    for slot, t0 in enumerate(slot_starts_utc):
        t0 = pd.Timestamp(t0).tz_convert("UTC")
        t1 = t0 + pd.Timedelta(seconds=window_s)
        for i in range(0, len(symbols), chunk):
            q = fetch_quotes(symbols[i:i + chunk], t0.isoformat().replace("+00:00", "Z"),
                             t1.isoformat().replace("+00:00", "Z"), session=sess)
            s = summarize_quotes(q)
            s["slot"], s["date"] = slot, pd.Timestamp(session_date).date()
            parts.append(s)
        if verbose:
            print(f"  {pd.Timestamp(session_date).date()} slot {slot + 1}/{len(slot_starts_utc)}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    if p is not None:
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
    return df
