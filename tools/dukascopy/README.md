# Dukascopy tick downloader

Free historical bid/ask ticks from Dukascopy's public datafeed (no account needed),
saved as one Parquet file per UTC day. Used for the USD/JPY gotobi / Tokyo-fix test
(Ito & Yamada, 2017).

```
pip install numpy pandas pyarrow requests
# test one month first (prints the folder size at the end)
python tools/dukascopy/dukascopy_download.py --symbol USDJPY --start 2024-01-01 --end 2024-01-31
# full history; 2026 stays sealed as holdout
python tools/dukascopy/dukascopy_download.py --symbol USDJPY --start 2010-01-01 --end 2025-12-31
```

- Output: `D:\MarketData\Dukascopy\<SYMBOL>\<hours_tag>\<YYYY>\<SYMBOL>_<YYYY-MM-DD>.parquet`
  (override with `--out` or env var `DUKASCOPY_DIR`).
- Columns: `ts` (UTC, ms), `bid`, `ask`, `bid_vol`, `ask_vol` (millions, Dukascopy SWFX only).
- Resumable; days with network errors are not written and are retried on rerun.
- Permanent gaps (404 / corrupt hours) are logged in `_gaps.csv`; treat those days as missing.
- `--hours 22-2` restricts to a UTC hour window. 9:55 JST = 00:55 UTC (no DST in Japan).
- Expected size (estimate): ~2-5 GB for full days 2010-2025 in Parquet.

## 1-minute candles (FX seasonality, FXS 001)

`dukascopy_candles.py` downloads Dukascopy's daily 1-minute **bid and ask** candle files: 2 requests per day instead of 24 hourly tick files.

```
# test one month first
python tools/dukascopy/dukascopy_candles.py --symbols EURUSD,CADCHF --start 2024-01-01 --end 2024-01-31
# all 7 majors + 8 crosses, 2016-2025 (2026 sealed)
python tools/dukascopy/dukascopy_candles.py --symbols all --start 2015-12-30 --end 2025-12-31 --workers 8
```

- Output: `D:\MarketData\Dukascopy\<SYMBOL>\candles_m1\<YYYY>\<SYMBOL>_<YYYY-MM-DD>.parquet`; columns `ts` (minute start, UTC), `bid_o/h/l/c/v`, `ask_o/h/l/c/v`.
- The record layout `(seconds, open, close, low, high, volume)` is checked on every file (whole-minute times, high/low consistent with open/close, open and close not swapped). A failed check stops the run with a message: report it.
- `--symbols all | majors | crosses | EURUSD,GBPUSD,...`. Resumable; Ctrl+C stops cleanly. Missing sides are logged in `_gaps.csv`.
