# FXS 004 — The positive structures on the sealed 2026 data: one look (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-09 |
| **Study** | FXS 06 (extends FXS 05 / discussion 003 from EURUSD-02 to the whole set) |
| **Responds to** | Iván, 2026-11-09: test the whole profitable-structure portfolio on 2026, not only EURUSD-02 |
| **Status** | Pre-registration. Written before any 2026 EURUSD, USDJPY or EURJPY candle has been downloaded. One look; nothing is changed after it. |

## 1. The set

Every frozen structure whose 2021–2025 net was positive at $3 per lot (FXS 04, `Portfolio/04/04_structures_pips.csv`). Chosen **after** seeing 2021–2025, which is why 2026 is the test.

| Structure | Rule (every trading day, news days included) | 2021–25 net pips/trade | 2021–25 USD per 1-lot trade |
|---|---|---|---|
| EURUSD-02 | sell 04:00 → buy 08:00 New York | +1.53 | +15.30 |
| USDJPY-01 | sell 00:55 → buy 01:55 UTC (09:55 → 10:55 JST) | +0.92 | +6.36 |
| USDJPY-02 | buy 00:25 → sell 00:55 UTC (09:25 → 09:55 JST) | +0.74 | +5.72 |
| EURJPY-02 | buy 12:35 → sell 16:35 New York | +0.83 | +5.91 |

Fills at the quoted bid/ask; $3 per 100,000-unit lot round trip.

**Disclosed contamination.** GTF 02 (2026-11-08) already looked at USDJPY's 2026 Tokyo mornings (09:25 → 10:30 JST, all bank days, Tokyo-morning ticks). USDJPY-01 and -02 overlap those minutes, so their 2026 result is weaker evidence. EURUSD and EURJPY 2026 data are unseen.

## 2. Tests (per structure and for the portfolio; Newey–West, 5 lags)

Per structure, in net pips per trade, against its own 2021–2025 figure ("promised"):

| Test | Condition |
|---|---|
| H1 | 2026 mean net > 0 |
| H2 | 2026 mean net significantly below the promised figure (one-sided p ≤ 0.05) |
| H3 (reported) | 2026 mean net > 0 with one-sided p ≤ 0.05 |

Verdict: `HOLDS` = H1 and not H2; `INCONCLUSIVE` = not H1 and not H2; `FAILS` = H2.

**Portfolio:** 1 lot of each structure every day, daily P&L in USD (promised +$33.29 per day = 15.30 + 6.36 + 5.72 + 5.91). Same three tests on the daily USD series.

EURUSD-02's own verdict is already pre-registered in discussion 003 with the same rule and tests; both notebooks compute it from the same data and must agree.

## 3. Decisions fixed in advance

- **Portfolio HOLDS:** the EA trades the structures whose own verdict is `HOLDS`; forward-test on the free trial before any paid challenge.
- **Portfolio INCONCLUSIVE:** no paid challenge; forward-record from today and re-evaluate after 6 months.
- **Portfolio FAILS:** drop the portfolio; individual structures that still `HOLD` are re-examined only with new data (2027+).
- A structure that `FAILS` is removed and not re-tested on this data.

## 4. Reported, not decisive

Monthly results; 2024–2026 side by side; worst open loss in pips; correlation of the structures in 2026; a $6,000 prop account through 2026 at 2× and 5× per structure (fixed size, 5% daily / 10% total limits).

## 5. Data

Dukascopy 1-minute candles, trading days 2026-01-01 → 2026-11-06, for EURUSD, USDJPY and EURJPY (download `--start 2025-12-31 --end 2026-11-06`); ≥ 99% of the expected day files per pair.
