# FXS 003 — EURUSD-02 on the sealed 2026 data: one look (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-09 |
| **Study** | FXS 05 |
| **Responds to** | Iván, 2026-11-09: test EURUSD-02 on 2026 before building an MT5 bot for prop-firm accounts; news days are **not** skipped |
| **Status** | Pre-registration. Written before any 2026 EURUSD candle has been downloaded or read. One look; nothing is changed after it. |

## 1. Why this test

EURUSD-02 was found in 2016–2020, held up after spread in 2021–2025 (+1.83 pips per trade, t 3.21), and clears every FXS 001 gate only at the prop-firm cost of $3 per lot. That cost model was chosen after the 2021–2025 results, and EURUSD-02 was picked from ten structures because of those results. 2026 is the only EURUSD data that has not been used for anything.

## 2. Rule (frozen; from `notebooks/FX_Intraday_Seasonality/EURUSD/01/candidates_frozen.csv`)

Every trading day (New York 17:00 → 17:00 days, Monday–Friday labels), news days included:

- **Sell** EURUSD at the bid at **04:00 New York**.
- **Buy back** at the ask at **08:00 New York**.
- No stop, no filter, no position sizing inside the test.
- **Cost:** $3 per 100,000-unit lot round trip (≈ 0.30 pip), on top of the quoted spread.

## 3. Data

Dukascopy 1-minute bid/ask candles, trading days 2026-01-01 → 2026-11-06 (download `--start 2025-12-31 --end 2026-11-06`). At least 99% of the expected day files must exist. The code reads 2026 only through the explicit `holdout` stage.

## 4. Power, stated before the data

About 215 trading days. The 2021–2025 per-trade standard deviation is ≈ 21 pips, so the standard error of the 2026 mean is ≈ 1.45 pips. If the true edge is +1.53 pips, a one-sided 5% test of "mean > 0" passes only about **27%** of the time. So the decision rests on two questions: is the 2026 mean positive, and is it clearly *below* what 2021–2025 promised?

## 5. Tests (net pips per trade, Newey–West standard error, 5 lags)

| Test | Condition |
|---|---|
| **H1** | 2026 mean net > 0 |
| **H2** | 2026 mean net significantly **below +1.53 pips** (one-sided p ≤ 0.05) |
| H3 (reported) | 2026 mean net > 0 with one-sided p ≤ 0.05 |

**Verdict**
- `HOLDS`: H1 true and H2 false.
- `INCONCLUSIVE`: H1 false (mean ≤ 0) and H2 false.
- `FAILS`: H2 true.

## 6. Decisions fixed in advance

- **HOLDS:** build the MT5 EA (fixed lots on the initial balance; 0.3–0.5 lots per $6,000; server-time conversion; hard daily stop inside the 5% limit) and forward-test it on the free trial before paying for a challenge.
- **INCONCLUSIVE:** no paid challenge. Run the rule forward on the trial or demo account from today; re-evaluate after 6 months of new trades.
- **FAILS:** drop EURUSD-02.

## 7. Reported, not decisive

Monthly net pips; 2024, 2025 and 2026 side by side; worst open loss in pips; the 2026 path of a $6,000 prop account at 2× and 5× (fixed size, 5% daily / 10% total limits). Not modelled: FundedNext's funded-account rule that profitable trades placed within 5 minutes of a high-impact event count at 40%.
