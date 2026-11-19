#!/usr/bin/env python3
"""Quote the Databento cost of short-rate futures for the bond-FX study. Downloads nothing.

metadata.get_cost / get_billable_size are free calls. The API key is read from DATABENTO_API_KEY (.env),
as in equity_trading.futures.

Contracts (CME Globex, GLBX.MDP3), front contract rolled by volume (continuous symbology "<ROOT>.v.0"):
  SR3  3-month SOFR futures   (expected short rates: the T-bill end of the curve)
  ZT   2-year T-Note futures
  Z3N  3-year T-Note futures
  ZF   5-year T-Note futures

Schemas quoted:
  ohlcv-1m  1-minute trade bars (what the bond-FX engine needs)
  bbo-1m    best bid/offer sampled every minute (only if real futures spreads are wanted for costs)

Years 2016-2025 only; 2026 stays sealed.

  python tools/databento/rates_cost.py
  python tools/databento/rates_cost.py --roots ZT,ZF --schemas ohlcv-1m --start 2016 --end 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from equity_trading.futures import client  # noqa: E402

ROOTS = {"SR3": "3-month SOFR", "ZT": "2-year T-Note", "Z3N": "3-year T-Note", "ZF": "5-year T-Note"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--roots", default=",".join(ROOTS))
    p.add_argument("--schemas", default="ohlcv-1m,bbo-1m")
    p.add_argument("--start", type=int, default=2016)
    p.add_argument("--end", type=int, default=2025)
    a = p.parse_args()
    if a.end >= 2026:
        sys.exit("2026 is sealed: use --end 2025 or earlier")
    cl = client()
    rows = []
    for schema in a.schemas.split(","):
        for root in a.roots.split(","):
            for y in range(a.start, a.end + 1):
                q = dict(dataset="GLBX.MDP3", symbols=[f"{root}.v.0"], stype_in="continuous", schema=schema,
                         start=f"{y}-01-01", end=f"{y + 1}-01-01")
                try:
                    cost = float(cl.metadata.get_cost(**q))
                    size = int(cl.metadata.get_billable_size(**q))
                    err = ""
                except Exception as e:                       # e.g. symbol not available that year
                    cost, size, err = float("nan"), 0, str(e)[:80]
                rows.append({"schema": schema, "root": root, "year": y, "usd": cost, "mb": size / 1e6, "error": err})
                print(f"{schema:9s} {root:4s} {y}: ${cost:8.2f}  {size / 1e6:8.1f} MB {err}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nPer schema and contract, all years:")
    print(df.groupby(["schema", "root"])[["usd", "mb"]].sum().round(2).to_string())
    print("\nTotal per schema:")
    print(df.groupby("schema")[["usd", "mb"]].sum().round(2).to_string())
    out = Path(__file__).with_name("rates_cost_quote.csv")
    df.to_csv(out, index=False)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
