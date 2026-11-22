#!/usr/bin/env python3
"""Download and parse SEC EDGAR fundamentals for the Fair_Value branch (FV 01). Free, no key.

Needs SEC_USER_AGENT in .env ("Your Name you@mail.com"). Everything goes to <EQUITY_TRADING_DIR>/EDGAR/.
Every step is resumable; re-running skips finished work.

  python tools/edgar/edgar_download.py                       # all steps
  python tools/edgar/edgar_download.py --steps bulk,fsds     # downloads only
  python tools/edgar/edgar_download.py --steps facts         # (re)parse companyfacts only

Steps, in order:
  bulk         companyfacts.zip (~1-2 GB), submissions.zip (~1-2 GB), company_tickers_exchange.json (dated)
  fsds         Financial Statement Data Sets 2009q1 -> last published quarter; keeps sub.txt only (~5 GB transfer)
  submissions  companies.parquet, filings.parquet (periodic reports, 8-K, delisting, merger forms)
  facts        facts/part-NNNNN.parquet: every us-gaap / dei / ifrs-full fact from 10-K/10-Q/20-F/40-F (+ amendments)
  tickers      fsds_sub.parquet, ticker_history.parquet (symbol candidates per CIK incl. delisted)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from equity_trading import edgar as E  # noqa: E402

STEPS = ["bulk", "fsds", "submissions", "facts", "tickers"]


def main() -> int:
    p = argparse.ArgumentParser(description="Download / parse SEC EDGAR bulk fundamentals.")
    p.add_argument("--steps", default=",".join(STEPS))
    p.add_argument("--force", action="store_true", help="re-download today's bulk snapshot")
    p.add_argument("--fsds-first", default="2009q1")
    p.add_argument("--overwrite-facts", action="store_true", help="reparse facts from a newer companyfacts snapshot")
    a = p.parse_args()
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"unknown steps {bad}; choose from {STEPS}")
    E.user_agent()  # fail early if not set
    print(f"EDGAR folder: {E.edgar_dir()} (engine {E.EDGAR_VERSION})")
    for s in STEPS:
        if s not in steps:
            continue
        t0 = time.time()
        print(f"\n== {s}")
        if s == "bulk":
            E.download_bulk(force=a.force)
        elif s == "fsds":
            E.download_fsds(E.fsds_quarters(a.fsds_first))
        elif s == "submissions":
            c, f = E.parse_submissions()
            print(f"{len(c):,} entities, {len(f):,} filings kept")
        elif s == "facts":
            E.parse_companyfacts(overwrite=a.overwrite_facts)
        elif s == "tickers":
            sub = E.combine_fsds()
            h = E.ticker_history(fsds_sub=sub)
            print(f"{len(sub):,} FSDS submissions; {h['cik'].nunique():,} CIKs with {len(h):,} symbol candidates")
        print(f"   {s} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
