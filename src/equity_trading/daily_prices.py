"""Daily stock prices for the Fair_Value branch (FV 01): Yahoo (yfinance) and Alpaca, cached per symbol.

Yahoo keeps live symbols only: delisted firms usually vanish and reused symbols return the *new*
company's history without warning. Alpaca daily bars (2016 ->) are downloaded for the same symbol list
as a second source. Neither is trusted blindly: the FV 01 notebook matches each EDGAR (CIK, symbol,
window) to the price series and flags mismatches.

Yahoo is stored UNADJUSTED (auto_adjust=False) with its Dividends and Stock Splits columns, so total
returns are computed by us and do not move when Yahoo revises its adjusted close.

Layout: <EQUITY_TRADING_DIR>/Yahoo/daily/<SYMBOL>.parquet, <EQUITY_TRADING_DIR>/Yahoo/daily/_status.csv
Alpaca daily bars use the existing cache, <EQUITY_TRADING_DIR>/Alpaca/bars/1Day/<SYMBOL>/<YEAR>.parquet.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .data import vendor_dir

DAILY_PRICES_VERSION = "0.1.0"
YAHOO_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
STATUS_COLUMNS = ["symbol", "status", "rows", "first", "last", "checked"]


def yahoo_dir() -> Path:
    d = vendor_dir("Yahoo") / "daily"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _status_path() -> Path:
    return yahoo_dir() / "_status.csv"


def load_status() -> pd.DataFrame:
    p = _status_path()
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=STATUS_COLUMNS)


def _append_status(rows: list[dict]) -> None:
    if not rows:
        return
    p = _status_path()
    pd.DataFrame(rows, columns=STATUS_COLUMNS).to_csv(p, mode="a", header=not p.exists(), index=False)


def split_download(d: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """yf.download(group_by='ticker') frame -> {symbol: frame with YAHOO_COLUMNS}, empty series dropped."""
    out = {}
    if d is None or d.empty:
        return out
    if isinstance(d.columns, pd.MultiIndex):
        lv0 = set(d.columns.get_level_values(0))
        for s in symbols:
            if s not in lv0:
                continue
            x = d[s].copy()
            x = x.dropna(how="all", subset=[c for c in ("Open", "High", "Low", "Close") if c in x.columns])
            if len(x):
                out[s] = x.reindex(columns=YAHOO_COLUMNS)
    elif len(symbols) == 1:
        x = d.dropna(how="all", subset=[c for c in ("Open", "High", "Low", "Close") if c in d.columns])
        if len(x):
            out[symbols[0]] = x.reindex(columns=YAHOO_COLUMNS)
    return out


def download_yahoo(symbols: list[str], start: str = "2008-01-01", end: str = "2026-01-01", batch: int = 50,
                   pause: float = 2.0, retry_missing: bool = False, refresh: bool = False) -> pd.DataFrame:
    """Download unadjusted daily OHLCV + actions for symbols into Yahoo/daily/<SYMBOL>.parquet.

    end is exclusive (default stops at 2025-12-31; 2026 sealed). Resumable through _status.csv:
    symbols already 'ok' are skipped unless refresh; symbols recorded 'missing' are skipped unless
    retry_missing. Returns the status rows of this run.
    """
    import yfinance as yf

    st = load_status()
    done = set(st.loc[st["status"] == "ok", "symbol"]) if not refresh else set()
    missing = set(st.loc[st["status"] == "missing", "symbol"]) if not retry_missing else set()
    todo = [s for s in dict.fromkeys(symbols) if s not in done and s not in missing]
    print(f"{len(todo):,} symbols to download ({len(done):,} ok, {len(missing):,} known missing skipped)")
    log = []
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=False, actions=True,
                                group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # noqa: BLE001 — Yahoo throttling: back off
                print(f"  batch {k // batch}: {type(e).__name__}: {e}; retry")
                time.sleep(30 * (attempt + 1))
        else:
            continue
        got = split_download(d, chunk)
        now = pd.Timestamp.now().strftime("%Y-%m-%d")
        rows = []
        for s in chunk:
            if s in got:
                x = got[s]
                x.index = pd.DatetimeIndex(x.index).tz_localize(None)
                x.index.name = "date"
                x.to_parquet(yahoo_dir() / f"{s}.parquet")
                rows.append({"symbol": s, "status": "ok", "rows": len(x), "first": x.index.min().date(),
                             "last": x.index.max().date(), "checked": now})
            else:
                rows.append({"symbol": s, "status": "missing", "rows": 0, "first": None, "last": None,
                             "checked": now})
        _append_status(rows)
        log.extend(rows)
        n_ok = sum(r["status"] == "ok" for r in rows)
        print(f"  {min(k + batch, len(todo)):,}/{len(todo):,}: {n_ok}/{len(chunk)} returned data")
        time.sleep(pause)
    return pd.DataFrame(log, columns=STATUS_COLUMNS)


def load_yahoo(symbol: str) -> pd.DataFrame | None:
    p = yahoo_dir() / f"{symbol}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def yahoo_coverage() -> pd.DataFrame:
    """Latest status row per symbol."""
    st = load_status()
    return st.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def download_alpaca_daily(symbols: list[str], years: list[int], batch: int = 50) -> None:
    """Alpaca 1Day bars (SIP, 2016 ->) for symbols, via the existing alpaca cache (resumable per year)."""
    from . import alpaca
    years = [y for y in years if 2016 <= y <= 2025]
    alpaca.download_years(list(dict.fromkeys(symbols)), "1Day", years, batch=batch)


def alpaca_coverage(symbols: list[str]) -> pd.DataFrame:
    """Per symbol: years cached, first and last bar date in the Alpaca 1Day cache."""
    root = vendor_dir("Alpaca") / "bars" / "1Day"
    rows = []
    for s in symbols:
        d = root / s
        files = sorted(d.glob("*.parquet")) if d.exists() else []
        first = last = None
        n = 0
        for f in files:
            x = pd.read_parquet(f)
            if len(x):
                t = pd.to_datetime(x["t"] if "t" in x.columns else x.index)
                first = t.min() if first is None else min(first, t.min())
                last = t.max() if last is None else max(last, t.max())
                n += len(x)
        rows.append({"symbol": s, "years_cached": len(files), "rows": n, "first": first, "last": last})
    return pd.DataFrame(rows)
