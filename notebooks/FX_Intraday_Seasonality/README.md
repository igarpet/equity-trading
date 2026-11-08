# FX Intraday Seasonality — stacked 1-minute days

**Question.** Lay every trading day of 1-minute candles on top of the others by minute of day, with no calendar labels: is there an interval whose mean return differs from zero, consistently across years, by more than it costs to trade, and does it hold out of sample? 15 pairs (7 majors, 8 crosses), two clocks (New York 17:00→17:00, UTC 00:00→24:00).

One folder per pair (FXS 002), each with two notebooks that are identical across pairs except for the pair:

```
FX_Intraday_Seasonality/
  EURUSD/
    01_discovery.ipynb      2016-2020 scan; writes EURUSD/01/ (candidates_frozen.csv, summary.json, figures)
    02_confirmation.ipynb   2021-2025 test of the frozen candidates; writes EURUSD/02/
  GBPUSD/ ...  (EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURGBP EURCHF EURJPY GBPJPY AUDJPY CADJPY CADCHF AUDNZD)
```

| Notebook | Description | Status |
|---|---|---|
| `<PAIR>/01_discovery` | download-complete check, candle close check vs USD/JPY ticks, stacked day per clock and per year, activity and spread by minute, 15-minute block t and year consistency, 2,704-interval scan (D1 BH within the pair, D2 years, D3 cost) with t-map, frozen candidates, 20 rotated-day placebos (D0) | EURUSD `5e5a28d`: 45 intervals pass D1–D3, placebos 0 → `STRUCTURE_FOUND`, 2 candidates. GBPUSD `08b8717`: 0 pass D1 → `NO_STRUCTURE`. Other pairs not run |
| `<PAIR>/02_confirmation` | frozen candidates traded at quoted prices on 2021–2025, gates K1–K4, by year, cumulative net, shrinkage | EURUSD `08b8717`: both `NOT_CONFIRMED` (EURUSD-01 net −0.55 bp, negative every year; EURUSD-02 net +0.96 bp, fails K2 by 0.02–0.04 bp, 2024–2025 negative). GBPUSD: nothing to confirm |

Protocols: [`001`](../../discussions/FX_Intraday_Seasonality/001_claude_fxs01_preregistration.md) (design), [`002`](../../discussions/FX_Intraday_Seasonality/002_claude_fxs01_amendment_per_pair.md) (one study per pair).

**Data.** `tools/dukascopy/dukascopy_candles.py` (Dukascopy 1-minute bid/ask candles, one file per UTC day; layout checked on every file). Expected ~1–2 GB for 15 pairs × 10 years.

**Code.** `src/equity_trading/fx_seasonality.py` 0.2.0, tests in `tests/test_fx_seasonality.py`. Discovery code refuses 2021+ data; nothing reads 2026; a pair is refused until ≥ 99% of its day files exist, so it can be analysed while other pairs download.

**Smoke run.** `FXS_SYNTHETIC=1` runs a pair's notebooks offline on synthetic candles; EURUSD gets one injected effect (+4 bp, 08:00–09:00 New York) and must return one `CONFIRMED` candidate there, every other pair must return none. Set `FXS_SYNTH_DIR` to the same folder for both notebooks of a pair.

**Layout change (2026-11-08).** After the EURUSD discovery run, files moved from `01_discovery_EURUSD.ipynb` and `01/EURUSD/` to `EURUSD/01_discovery.ipynb` and `EURUSD/01/`, unchanged except for paths in the notebook code. The frozen-file check accepts LF or CRLF line endings (Git on Windows converts them; the content is what is verified).
