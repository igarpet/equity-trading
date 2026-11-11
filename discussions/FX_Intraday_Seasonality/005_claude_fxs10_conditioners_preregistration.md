# FXS 005 — When does each signal work? Five pre-entry conditioners (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-12 |
| **Study** | FXS 10 |
| **Responds to** | Iván, 2026-11-12: test the five conditioners on 2020–2026 only; the 06/08 structures did not exist before 2021 |
| **Status** | Written before any conditioned result has been computed. Nothing below is changed after the run. |

## 1. Signals

The 12 signals of FXS 09 (PREVIOUS, NEW from 06, NEW-08), same times, every trading day, time exits, news days included. Fills at Dukascopy bid/ask × 0.9, plus $3 per lot round trip.

## 2. Conditioners (all known before entry)

| # | Conditioner | Buckets |
|---|---|---|
| C1 | Gotobi: the Tokyo date of entry is a gotobi day (5, 10, …, 30, rolled to the previous Japanese bank day; GTF definition) | gotobi / other |
| C2 | Realized volatility: st.dev. of the last 20 daily (17:00 NY) log returns, through the previous close | above / below the training-period median |
| C3 | 60-day trend: log return of the last 60 closes, through the previous close, times trade direction | with trend / against |
| C4 | Move earlier the same day: mid from the signal's day start (00:00 of its clock) to entry, times trade direction | continuation / reversal |
| C5a | Day of week | Mon … Fri |
| C5b | Month-end: last two trading days of the calendar month | month-end / other |

C1 is applied to every signal but has a stated reason only for the Tokyo-fix signals (USDJPY-01, USDJPY-02, N-UJ-1).

## 3. Tests

**A. Descriptive, 2020–2026 pooled (in sample).** Per signal × conditioner: mean net pips per bucket. Binary conditioners: Welch t of the bucket difference. Day of week: one-way ANOVA. Benjamini–Hochberg over all 72 tests (12 × 6), q ≤ 0.05.

**B. Walk-forward, 2021–2026 (the decisive test).** For each test year Y, using only trades from 2020 to Y−1:
- C2's threshold is the median of the training values.
- Trade a bucket in Y if its training mean net is > 0; a bucket with fewer than 30 training trades is traded.
- **C0 (baseline):** trade the signal in Y only if its own training mean net is > 0, with no conditioner. A conditioner counts only if it beats C0 as well as the unfiltered signal.
- Measure: net pips per calendar day (a skipped day = 0) against the unfiltered signal; paired Newey–West t (5 lags). BH over the 72 signal × conditioner comparisons with the unfiltered signal.

**C. Portfolio.** All 12 signals on a $6,000 compounding account at 1, 2, 3, 5 and 10× per signal, 2021–2026 and 2026 alone: unfiltered, C0, and each conditioner applied to every signal through its walk-forward rule.

## 4. Power, stated before the data

Per-trade st.dev. is 10–24 pips. With about 1,700 trades per signal in 2020–2026 and buckets near half of them, a bucket difference must be about 2–3 pips to reach t ≈ 2. Smaller conditional effects cannot be told apart from noise here.

## 5. What counts as a result

A conditioner is **usable** for a signal only if, in B:
- it passes BH against the unfiltered signal;
- it also beats C0;
- its 2026 net pips per day are > 0.

Anything else is descriptive. The walk-forward has a one-year first training window (2020), which is short; this choice is Iván's, made because the 06/08 structures are believed absent before 2021.
