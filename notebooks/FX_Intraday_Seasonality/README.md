# FX Intraday Seasonality — stacked 1-minute days

**Question.** Lay every trading day of 1-minute candles on top of the others by minute of day, with no calendar labels: is there an interval whose mean return differs from zero, consistently across years, by more than it costs to trade, and does it hold out of sample? 15 pairs (7 majors, 8 crosses), two clocks (New York 17:00→17:00, UTC 00:00→24:00).

| Notebook | Description | Status |
|---|---|---|
| `01_discovery_scan.ipynb` | 2016–2020 only: candle check against USD/JPY ticks, stacked-day profiles, activity and spread by minute, hourly t-maps, scan of 40,560 intervals (D1 BH, D2 years, D3 cost), de-duplication, frozen candidate list, rotated-day placebos (D0) | Code ready, not run |
| `02_confirmation.ipynb` | 2021–2025: frozen candidates traded at quoted prices, gates K1–K4, by year, equity, discovery-vs-confirmation shrinkage | Run after `01/` is committed |

Protocol: [`001_claude_fxs01_preregistration.md`](../../discussions/FX_Intraday_Seasonality/001_claude_fxs01_preregistration.md).

**Data.** `tools/dukascopy/dukascopy_candles.py` (Dukascopy 1-minute bid/ask candles, one file per UTC day; layout checked on every file). Expected ~1–2 GB for 15 pairs × 10 years.

**Code.** `src/equity_trading/fx_seasonality.py`, tests in `tests/test_fx_seasonality.py`. Discovery code refuses 2021+ data; nothing reads 2026.

**Smoke run.** `FXS_SYNTHETIC=1` runs both notebooks offline on three synthetic pairs with one injected effect (EURUSD +4 bp, 08:00–09:00 New York); it must return one `CONFIRMED` candidate in that window. Set `FXS_SYNTH_DIR` to the same folder for both notebooks.
