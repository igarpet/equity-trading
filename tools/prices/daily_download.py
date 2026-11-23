#!/usr/bin/env python3
"""Daily prices for every EDGAR symbol candidate (Fair_Value branch, FV 01).

Run after tools/edgar/edgar_download.py (needs EDGAR/ticker_history.parquet and filings.parquet).
Symbol list = every candidate symbol of every CIK that filed a 10-K/10-Q/20-F/40-F since --since,
live and delisted. No size filter here: enterprise-value filters are applied point-in-time in the studies.

  python tools/prices/daily_download.py --source yahoo            # unadjusted OHLCV + actions, 2008 -> 2025
  python tools/prices/daily_download.py --source alpaca           # Alpaca 1Day SIP bars 2016 -> 2025 (adjusted)
  python tools/prices/daily_download.py --source yahoo --retry-missing   # re-ask symbols recorded missing
  Rate limits: rate-limited symbols are not recorded; the run sleeps 15 -> 60 min and retries the batch.
  python tools/prices/daily_download.py --list-only               # write the symbol list and stop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from equity_trading import daily_prices as P  # noqa: E402
from equity_trading import edgar as E  # noqa: E402


def symbol_list(since: str) -> pd.DataFrame:
    e = E.edgar_dir()
    h = pd.read_parquet(e / "ticker_history.parquet")
    st = E.company_status()
    keep = st.loc[st["last_periodic"] >= pd.Timestamp(since), "cik"]
    h = h[h["cik"].isin(keep) & (h["last_filed"] >= pd.Timestamp(since))]
    syms = (h.groupby("symbol")["cik"].nunique().rename("n_ciks").reset_index())
    syms["yahoo"] = syms["symbol"].map(E.yahoo_symbol)
    syms["alpaca"] = syms["symbol"].str.replace("-", ".", regex=False)
    syms.to_csv(e / "price_symbols.csv", index=False)
    return syms


def main() -> int:
    p = argparse.ArgumentParser(description="Download daily prices for EDGAR symbol candidates.")
    p.add_argument("--source", choices=["yahoo", "alpaca"], default="yahoo")
    p.add_argument("--since", default="2009-01-01", help="CIKs with a periodic filing on/after this date")
    p.add_argument("--start", default="2008-01-01")
    p.add_argument("--retry-missing", action="store_true")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--batch", type=int, default=20, help="Yahoo symbols per request batch")
    p.add_argument("--pause", type=float, default=5.0, help="seconds between Yahoo batches")
    a = p.parse_args()
    syms = symbol_list(a.since)
    print(f"{len(syms):,} symbols ({(syms['n_ciks'] > 1).sum():,} shared by more than one CIK over time)")
    if a.list_only:
        return 0
    if a.source == "yahoo":
        log = P.download_yahoo(syms["yahoo"].tolist(), start=a.start, end="2026-01-01",
                               batch=a.batch, pause=a.pause, retry_missing=a.retry_missing)
        if len(log):
            print(log["status"].value_counts().to_string())
    else:
        P.download_alpaca_daily(syms["alpaca"].tolist(), list(range(2016, 2026)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
