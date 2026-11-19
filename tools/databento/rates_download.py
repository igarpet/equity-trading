#!/usr/bin/env python3
"""Download Databento bbo-1m for the US short-rate futures of the rate-FX study (RFX 001).

  ZT  (2-year T-Note)  2016-2025
  ZF  (5-year T-Note)  2016-2025
  SR3 (3-month SOFR)   2018-2025 (the study uses 2022-2025; earlier years are thin and cost cents)

Front contract rolled by volume (continuous ``<ROOT>.v.0``), month by month. Before each request the quoted cost
is checked against the shared ledger (<EQUITY_TRADING_DIR>/Databento/ledger.csv, also used by futures.py) and the
--budget; the request is logged before the transfer. Resumable: months whose raw file exists are only
re-normalised (free). 2026 is refused.

  python tools/databento/rates_download.py --dry-run          # quote only, buys nothing
  python tools/databento/rates_download.py                    # buy (about $12 at the 2026-11-19 quote)
  python tools/databento/rates_download.py --roots ZT --start 2016 --end 2016
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from equity_trading import rates as R  # noqa: E402
from equity_trading.futures import client, spent_usd  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Download Databento bbo-1m for ZT, ZF, SR3.")
    p.add_argument("--roots", default="ZT,ZF,SR3")
    p.add_argument("--start", type=int, default=2016)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--budget", type=float, default=115.0, help="stop before the ledger total would exceed this (USD)")
    p.add_argument("--dry-run", action="store_true", help="print the quoted cost of what is not yet downloaded")
    a = p.parse_args()
    if a.end >= 2026:
        sys.exit("2026 is sealed: use --end 2025 or earlier")
    roots = [r.strip().upper() for r in a.roots.split(",")]
    cl = client()
    print(f"ledger total so far: ${spent_usd():.2f} | budget ${a.budget:.2f}")
    if a.dry_run:
        rows = []
        for root in roots:
            for m0, m1 in R.months(max(a.start, R.ROOTS[root]["start"]), a.end):
                if (R.raw_dir(root) / f"{m0:%Y-%m}.dbn.zst").exists():
                    continue
                rows.append({"root": root, "year": m0.year, "usd": float(cl.metadata.get_cost(**R.query(root, m0, m1)))})
        df = pd.DataFrame(rows)
        if df.empty:
            print("everything already downloaded")
        else:
            print(df.groupby(["root", "year"])["usd"].sum().round(3).to_string())
            print(f"total to buy: ${df['usd'].sum():.2f}")
        return 0
    log = R.download(cl, roots, a.start, a.end, a.budget)
    print(log.groupby("status")["cost_usd"].agg(["count", "sum"]).round(3).to_string() if len(log) else "nothing to do")
    print(f"ledger total now: ${spent_usd():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
