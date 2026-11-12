# FXS 006 — Structures that separate winning from losing days (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-12 |
| **Study** | FXS 10b |
| **Responds to** | Iván, 2026-11-12: scan for recurring structures on the winning days of the 12 signals; 2020–2025 for all signals; winner = mid-price drift; all 12 signals |
| **Status** | Written before the scan. Candidates are frozen before 2026 is read. |

## 1. Labels

For each of the 12 FXS 09 signals and each trading day of 2020–2025, the mid-price log change from entry to exit, in the trade direction:
- **winner** if > 0;
- **loser** if < 0;
- days with exactly 0 (no quotes) are dropped.

## 2. What is scanned

The signal's own pair, on the signal's own clock, at times relative to entry: starts every 5 minutes, lengths 15, 30, 60, 120 and 240 minutes.

| Zone | Intervals | Use if found |
|---|---|---|
| **pre** | inside the 24 h before entry (end ≤ entry) | filter: trade the signal only when the pre-entry move points to a winner |
| **post** | inside the 24 h after exit (start ≥ exit) | second trade, opened after exit once the first trade's result is known |
| during the trade | **not scanned** | winners rise there by definition (circular) |

That is 1,352 intervals per zone, so 2,704 per signal. Prices at minutes when the market is closed (Fri 17:00 → Sun 17:00 New York) are missing; intervals with a missing endpoint are dropped for that day.

## 3. Tests, per signal

- **E1:** winner mean − loser mean of the interval's mid log return; Welch t; Benjamini–Hochberg over the signal's 2,704 tests, q ≤ 0.05.
- **E2:** the difference has the same sign in at least 5 of the 6 years.
- **E0 (placebo):** winner/loser labels shuffled within each year, 20 times. The signal has structure only if its E1 + E2 count exceeds the placebos' 95th percentile.
- **Candidates:** E1 + E2 survivors by |t|, non-overlapping within a zone, at most 10 per signal.

## 4. Trading measures (reported for every candidate; 0.9 × Dukascopy spread, $3 per lot)

- **pre:** the signal's net pips on days the pre-entry move agrees with the winner pattern, against days it disagrees, and the win rate in each group.
- **post:** the follow-on trade's net pips. Trade the interval in the sign of the winner−loser difference after a winning day, and in the opposite sign after a losing day.

## 5. 2026, one look

The candidates are written to `10b/10b_candidates_frozen.csv` (SHA-256 recorded) before any 2026 minute is loaded. Then, for each candidate on 2026:
- the E1 difference and its sign;
- the trading measure of §4.

A candidate **holds** in 2026 if the difference has the same sign and the trading measure is positive. Over about 215 days, nothing smaller than roughly 2–3 pips can be told from noise.
