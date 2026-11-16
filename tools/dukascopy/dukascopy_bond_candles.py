#!/usr/bin/env python3
"""
Dukascopy 1-minute bid/ask candles for the three government-bond CFDs.

Thin wrapper around dukascopy_candles.py (must sit in the same folder, tools/dukascopy/). It reuses that
downloader unchanged, so the files have exactly the FX layout:

  <out>/<SYMBOL>/candles_m1/<YYYY>/<SYMBOL>_<YYYY-MM-DD>.parquet
    ts (minute start, UTC), bid_o, bid_h, bid_l, bid_c, bid_v, ask_o, ask_h, ask_l, ask_c, ask_v
  <out>/<SYMBOL>/candles_m1/_gaps.csv

Symbols (Dukascopy feed codes):
  USTBONDTRUSD  US T-Bond CFD      (USTBOND.TR/USD)
  BUNDTREUR     Euro Bund CFD      (BUND.TR/EUR)
  UKGILTTRGBP   UK Long Gilt CFD   (UKGILT.TR/GBP)

Only two things differ from FX:
  - price scaling: integers / 1000 (assumed; FX uses 100000, JPY pairs 1000);
  - after the run, the median bid of the newest file is printed per symbol and checked against a
    plausible range (50-250). A value outside it means the /1000 scaling is wrong: report it.

These are CFDs that follow the front futures contract: prices jump at each quarterly roll, the spread is
Dukascopy's own, and volume is not exchange volume. Days before a symbol's history starts are logged as
gaps (status "missing"); that is expected.

Examples:
  # test one month first
  python tools/dukascopy/dukascopy_bond_candles.py --start 2024-01-01 --end 2024-01-31
  # full history, 2026 sealed
  python tools/dukascopy/dukascopy_bond_candles.py --start 2017-12-29 --end 2025-12-31 --workers 8
  # one symbol
  python tools/dukascopy/dukascopy_bond_candles.py --symbols BUNDTREUR --start 2024-01-01 --end 2024-01-31

All other options (--out, --workers, --retries) are those of dukascopy_candles.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dukascopy_candles as dc  # noqa: E402

BONDS = ("USTBONDTRUSD", "BUNDTREUR", "UKGILTTRGBP")
BOND_DIVISOR = 1_000
PLAUSIBLE = (50.0, 250.0)

_fx_divisor = dc.divisor


def divisor(sym: str) -> int:
    return BOND_DIVISOR if sym in BONDS else _fx_divisor(sym)


dc.divisor = divisor
dc.PRESETS["bonds"] = BONDS


def price_check(out: Path, syms: tuple[str, ...]) -> bool:
    import pandas as pd
    ok = True
    for sym in syms:
        files = sorted((out / sym / "candles_m1").rglob(f"{sym}_*.parquet"))
        if not files:
            print(f"{sym}: no files written (no data in this date range?)")
            continue
        df = pd.read_parquet(files[-1], columns=["bid_c", "ask_c", "bid_v"])
        live = df[df["bid_v"] > 0]
        if live.empty:
            print(f"{sym}: {files[-1].name} has no live minutes; price check skipped")
            continue
        bid = float(live["bid_c"].median())
        spread = float((live["ask_c"] - live["bid_c"]).median())
        good = PLAUSIBLE[0] <= bid <= PLAUSIBLE[1]
        ok &= good
        print(f"{sym}: {files[-1].name} median bid {bid:.3f}, median spread {spread:.3f}, "
              f"{len(live)} live minutes -> {'OK' if good else 'IMPLAUSIBLE PRICE: scaling is wrong, report this'}")
    return ok


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--symbols", default=None)
    pre.add_argument("--out", default=os.environ.get("DUKASCOPY_DIR", r"D:\MarketData\Dukascopy"))
    known, _ = pre.parse_known_args()
    if known.symbols is None:
        sys.argv += ["--symbols", "bonds"]
        syms = BONDS
    else:
        syms = dc.PRESETS.get(known.symbols.lower()) or tuple(s.strip().upper() for s in known.symbols.split(","))
    rc = dc.main()
    print("\nPrice check (newest file per symbol):")
    if not price_check(Path(known.out), tuple(s for s in syms if s in BONDS)):
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
