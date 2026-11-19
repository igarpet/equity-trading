#!/usr/bin/env python3
"""Quote and download Databento ohlcv-1m for the micro index futures (MNQ return-prediction study).

  MNQ (Micro E-mini Nasdaq-100) by default; MES, MYM optional. Micro contracts only.
  Front contract rolled by volume (continuous ``<ROOT>.v.0``), month by month, from listing (2019-05) to 2025.
  Databento has no 5-minute schema: 1-minute bars are bought; ``equity_trading.micro_bars.to_bars`` resamples.

Before each request the quoted cost is checked against the shared ledger (<EQUITY_TRADING_DIR>/Databento/ledger.csv,
also used by futures.py and rates.py) and --budget; the request is logged before the transfer. Resumable: months
whose raw file exists are only re-normalised (free). 2026 is refused.

  python tools/databento/micro_bars_download.py --dry-run                 # quote only, buys nothing
  python tools/databento/micro_bars_download.py                           # buy MNQ 2019-05 .. 2025-12
  python tools/databento/micro_bars_download.py --start 2023 --end 2025   # shorter window
  python tools/databento/micro_bars_download.py --roots MNQ,MES --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from equity_trading import micro_bars as M  # noqa: E402
from equity_trading.futures import MICROS, client, spent_usd  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Quote / download Databento ohlcv-1m for micro index futures.")
    p.add_argument("--roots", default="MNQ")
    p.add_argument("--start", type=int, default=2019)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--budget", type=float, default=115.0, help="stop before the ledger total would exceed this (USD)")
    p.add_argument("--dry-run", action="store_true", help="print the quoted cost of what is not yet downloaded")
    a = p.parse_args()
    if a.end >= 2026:
        sys.exit("2026 is sealed: use --end 2025 or earlier")
    roots = [r.strip().upper() for r in a.roots.split(",")]
    bad = [r for r in roots if r not in MICROS]
    if bad:
        sys.exit(f"{bad}: micro contracts only {MICROS}")
    cl = client()
    print(f"ledger total so far: ${spent_usd():.2f} | budget ${a.budget:.2f}")
    if a.dry_run:
        q = M.quote(cl, roots, a.start, a.end)
        if q.empty:
            print("everything already downloaded")
            return 0
        print(q.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        total = q["usd"].sum()
        print(f"total to buy: ${total:.2f} ({q['gb'].sum() * 1e3:.0f} MB) | "
              f"ledger after: ${spent_usd() + total:.2f} of ${a.budget:.2f}")
        return 0
    log = M.download(cl, roots, a.start, a.end, a.budget)
    print(log.groupby("status")["cost_usd"].agg(["count", "sum"]).round(3).to_string() if len(log) else "nothing to do")
    print(f"ledger total now: ${spent_usd():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
