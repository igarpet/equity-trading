# FXS 001 — Stacked 1-minute days: intraday structure in 15 FX pairs (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-08 |
| **Branch** | FX_Intraday_Seasonality (new) |
| **Responds to** | Iván, 2026-11-08: stack every day's 1-minute candles by time of day, without calendar labels, and look for structure that repeats; all majors and crosses; discovery 2016–2020, confirmation 2021–2025 |
| **Status** | Pre-registration. Written before any data exists. Gates are not changed after results exist. |

## 1. Question

If every trading day of 1-minute candles is laid on top of the others by minute of day, is there any interval whose average return differs from zero, consistently across years, by more than it costs to trade, and does it hold out of sample?

No calendar conditioning (gotobi, month-end, weekday, data days). That is a later study, conditional on this one.

## 2. Data

| Item | Value |
|---|---|
| Pairs | Majors EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD; crosses EURGBP, EURCHF, EURJPY, GBPJPY, AUDJPY, CADJPY, CADCHF, AUDNZD |
| Source | Dukascopy 1-minute bid and ask candles (`tools/dukascopy/dukascopy_candles.py`), one file per UTC day. Every file is checked for the record layout; the first run is validated against the USD/JPY ticks already downloaded (minute close = last tick of the minute). |
| **DISCOVERY** | trading days 2016-01-01 → 2020-12-31 |
| **CONFIRMATION** | trading days 2021-01-01 → 2025-12-31 |
| Sealed | 2026 onward (code refuses it). USD/JPY's 2026 Tokyo-morning data has been looked at in GTF 02, so a later 2026 holdout is weaker for that pair. |

Venue caveat: Dukascopy SWFX quotes, not interbank. Costs are measured from these quotes.

## 3. Construction

- **Two clocks**, each scanned separately:
  - `NY`: day = 17:00 → 17:00 New York (the FX rollover), labelled by the date it ends; follows US daylight saving.
  - `UTC`: day = 00:00 → 24:00 UTC (Tokyo time minus 9 h; no daylight saving).
  - London's clock differs from New York's only for 2–3 weeks a year, so `NY` stands in for it.
- **Days:** Monday–Friday labels; a day counts if ≥ 80% of its minutes have quotes. Gaps are forward-filled.
- **Price at a boundary:** the mid (bid + ask)/2 at the close of the minute ending there. 1,441 boundaries per day.
- **Interval return:** log(mid at end / mid at start) in basis points.
- **Intervals:** starts every 5 minutes, lengths 15, 30, 60, 120, 240 minutes, inside one day: 1,352 intervals per pair and clock, **40,560 tests** in total.
- **Profiles (descriptive):** average cumulative return path with its standard error, mean absolute 1-minute return, median spread per boundary, hourly means by year. Discovery window only.

## 4. Discovery (2016–2020)

For each interval: mean, Newey–West standard error (5 lags, days in order), two-sided p.

| Filter | Condition |
|---|---|
| **D1** | Benjamini–Hochberg over all 40,560 tests, q ≤ 0.05 |
| **D2** | yearly mean has the same sign as the overall mean in ≥ 4 of 5 years |
| **D3** | \|mean\| > cost = (median spread at start + median spread at end)/2 + 0.70 bp commission ($35/M per side) |

**Candidates:** survivors sorted by \|t\|; per pair, an interval is dropped if it overlaps an already kept one in UTC time (New York intervals under both DST offsets, so the same effect is not kept twice across clocks). At most **40** candidates. Direction = sign of the discovery mean. Different pairs are not de-duplicated, so one USD effect can appear in several USD pairs; confirmation tests each separately.

The candidate list is written to `notebooks/FX_Intraday_Seasonality/01/candidates_frozen.csv`, its SHA-256 recorded, and **committed before notebook 02 is run**.

**Global check D0 (reported):** the same pipeline on 20 placebos in which each day's 1-minute returns are rotated by an independent random offset. This keeps each day's returns and total move and destroys time-of-day alignment. `STRUCTURE_FOUND` if the real number of D1–D3 survivors exceeds the 95th percentile of the placebo counts; otherwise `NO_STRUCTURE`. Confirmation runs either way.

## 5. Confirmation (2021–2025)

Each frozen candidate is traded every confirmation day, at quoted prices: long buys the ask at the start and sells the bid at the end; short sells the bid and buys the ask. Net = after spread − 0.70 bp; stress = − 1.40 bp.

| Gate | Condition |
|---|---|
| **K1** | mean after spread > 0: Newey–West one-sided p, Benjamini–Hochberg across the candidates, q ≤ 0.05 |
| **K2** | net: moving-block bootstrap lower bound (quantile 0.025) > 0 at blocks 10 and 20 days |
| **K3** | mean at 2× commission > 0 |
| **K4** | yearly mean net > 0 in ≥ 3 of 5 years |

Candidate verdict: `CONFIRMED` if K1–K4, else `NOT_CONFIRMED`.

## 6. Readings fixed in advance

- **CONFIRMED candidates:** frozen as they are; a 2026 holdout look in a separate pre-registration, then calendar conditioning (FXS 002) on those intervals only.
- **NO_STRUCTURE and nothing confirmed:** unconditional intraday seasonality in these pairs does not pay Dukascopy costs.
- **STRUCTURE_FOUND but nothing confirmed:** time-of-day structure exists in 2016–2020 but is unstable or below cost after 2020.
- No candidate is modified (times, direction, pair) after the frozen file is committed. Changes are new hypotheses for a new pre-registration.

## 7. Not tested

Calendar conditioning, volatility or spread structure as trade signals, intervals across the day boundary, overnight carry (rollover swap), position sizing, combining candidates into a portfolio.

## 8. Run order

1. `python -m pytest -q tests/test_fx_seasonality.py`
2. Download (resumable; test one month first):
   `python tools/dukascopy/dukascopy_candles.py --symbols all --start 2015-12-30 --end 2025-12-31 --workers 8`
3. `notebooks/FX_Intraday_Seasonality/01_discovery_scan.ipynb`; commit `01/` including `candidates_frozen.csv`.
4. `notebooks/FX_Intraday_Seasonality/02_confirmation.ipynb`; commit `02/`.

`FXS_SYNTHETIC=1` runs both notebooks offline on synthetic candles with one injected effect (outputs to `NN_synthetic/`, ignored by git).
