"""Price loading.

Yahoo Finance (via yfinance) limits intraday history: 15m/30m bars cover only
the last 60 calendar days. Use it for exploratory studies only; Alpaca SIP is
the research-grade source.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_VERSION = "0.1.0"
NY = "America/New_York"


def data_dir() -> Path:
    """Data root from EQUITY_TRADING_DIR (read from .env if present), else ./data."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    d = Path(os.environ.get("EQUITY_TRADING_DIR", Path.cwd() / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_yahoo_closes(tickers: list[str], interval: str = "30m", period: str = "60d",
                      refresh: bool = False) -> pd.DataFrame:
    """Regular-session close prices, index in New York time, one column per ticker.

    Cached to <EQUITY_TRADING_DIR>/raw/yahoo_<interval>_<period>.parquet. Prices are
    split/dividend adjusted (auto_adjust=True).
    """
    raw = data_dir() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"yahoo_{interval}_{period}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df[tickers]
    import yfinance as yf  # imported lazily so tests do not need it

    d = yf.download(tickers, interval=interval, period=period, auto_adjust=True,
                    prepost=False, progress=False, group_by="column", threads=True)
    if d.empty:
        raise RuntimeError("Yahoo returned no data; check network access and tickers.")
    close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.DatetimeIndex(close.index).tz_convert(NY)
    close = regular_session(close)
    close.to_parquet(path)
    return close[tickers]


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars stamped 09:30 <= t < 16:00 New York time."""
    t = df.index.tz_convert(NY)
    mins = t.hour * 60 + t.minute
    return df[(mins >= 9 * 60 + 30) & (mins < 16 * 60)]


def synthetic_closes(tickers: list[str], n_days: int = 40, bars_per_day: int = 13,
                     seed: int = 0) -> pd.DataFrame:
    """Synthetic regular-session prices for smoke tests.

    Each ticker = sector factor + idiosyncratic AR(1) component, with a U-shaped
    intraday volatility profile and an overnight gap bar. Sectors are assigned by
    position in blocks of 10.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n_days)
    step = 390 // bars_per_day
    stamps = [pd.Timestamp(d).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=step * k)
              for d in days for k in range(bars_per_day)]
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    tod = np.tile(np.arange(bars_per_day), n_days)
    u = 1.0 + 1.2 * ((tod - (bars_per_day - 1) / 2) / ((bars_per_day - 1) / 2)) ** 2
    u[tod == 0] *= 2.0  # overnight gap
    n_sect = int(np.ceil(len(tickers) / 10))
    factors = np.cumsum(rng.standard_normal((n, n_sect)) * 0.002 * u[:, None], axis=0)
    out = {}
    for i, tk in enumerate(tickers):
        phi = rng.uniform(0.85, 0.97)
        e = rng.standard_normal(n) * 0.0015 * u
        idio = np.zeros(n)
        for t in range(1, n):
            idio[t] = phi * idio[t - 1] + e[t]
        beta = rng.uniform(0.7, 1.3)
        out[tk] = 100 * np.exp(beta * factors[:, i // 10] + idio + rng.standard_normal(n) * 0.0002)
    return pd.DataFrame(out, index=idx)
