# BDS 001 — Stacked 1-minute days in three government-bond CFDs (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-18 |
| **Branch** | Bond_Intraday_Seasonality (new) |
| **Responds to** | Iván, 2026-11-18: same analysis as the FX pairs on the three Dukascopy bond CFDs, but discover on the first half, on the second half and on the whole period, because FX structure also turned up in the later years; later possibly bond–FX pair trading |
| **Engine** | `bond_seasonality.py` 0.1.0 on top of `fx_seasonality.py` 0.2.0 |
| **Status** | Written before any bond data has been analysed. Gates are not changed after results exist. |

## 1. Question

Laying every trading day of 1-minute candles on top of the others by minute of day, without calendar labels: does any interval of the US T-Bond, Euro Bund or UK Long Gilt have a mean return that differs from zero, consistently across years, by more than a futures trade costs, and does it hold in the other half of the sample?

## 2. Data

| Item | Value |
|---|---|
| Symbols | `USTBONDTRUSD` (US T-Bond, future ZB, tick 1/32), `BUNDTREUR` (Euro Bund, FGBL, tick 0.01), `UKGILTTRGBP` (UK Long Gilt, tick 0.01) |
| Source | Dukascopy 1-minute bid/ask candles, `tools/dukascopy/dukascopy_bond_candles.py`, one file per UTC day |
| History | From each symbol's first file (the Bund and Gilt CFDs launched in August 2018) to 2025-12-31 |
| Sealed | 2026 onward; the code refuses it |

There are 8 years, not 10: H1 and H2 are 4 years each, and for the Bund and Gilt H1 is about 3⅓ years.

## 3. Windows and how they are used

| Window | Dates | Role |
|---|---|---|
| **H1** | 2018 → 2021 | discovery; candidates frozen, then confirmed on H2 (**forward**) |
| **H2** | 2022 → 2025 | discovery; candidates frozen, then confirmed on H1 (**backward**) |
| **FULL** | 2018 → 2025 | discovery on everything; in-sample, no confirmation window left |

Discovering on all three windows leaves no untouched data inside 2018–2025 for a FULL candidate, and backward confirmation tests whether a recent structure also existed earlier, not whether it will continue. Only 2026, kept sealed, can test a candidate on data that played no part in selecting it. A later pre-registration will name the candidates for that one look.

Both H1 and H2 frozen candidate files are committed before the cross-confirmation notebook is run.

## 4. Differences from the FX study (FXS 001/002)

1. **Contract switches.** The CFD follows the front futures contract and is switched to the next one during the daily break of the adjustment day (Dukascopy CFD adjustment calendar). The price jumps by the calendar spread, which is not a market move and keeps the same sign for long periods, so it would appear as time-of-day structure at the break. **Rule:** for each quarterly expiry month (Mar, Jun, Sep, Dec), among the breaks (≥ 30 minutes without quotes) between the 10th of the previous month and the end of the expiry month, the one with the largest |price gap| is the switch; any other break in that window larger than both half of it and 8× the symbol's median |gap| is also treated as a switch. Every trading day, in each clock, that contains the last quote before or the first quote after a switch is dropped. Each notebook lists the switches with their gap and its ratio to the median gap.
2. **Trading hours.** A day counts if its quoted minutes are at least **50%** of the window's median day (FX: 80% of all 1,440 minutes; bonds are not quoted around the clock).
3. **Missing files.** A day is dropped if any UTC day file it spans is missing (New York day d spans UTC days d−1 and d). Sundays count as expected files only for a symbol that is quoted on Sundays. Discovery requires ≥ **95%** of weekday files from the symbol's first file.
4. **Costs.** D3 and the confirmation net use a **futures-equivalent cost**: one tick of the matching future at the window's median price (UST about 2.6 bp, Bund about 0.75 bp, Gilt about 1 bp) plus the FX commission of 0.70 bp. Dukascopy's own CFD spread is about 0.05 points, 4–5 times wider; passing D3 at that cost is reported as `D3_cfd`, and the confirmation reports `cfd_net_bps`. This is a deliberate change from FX: trading these structures would go through futures, and gating on a CFD spread that will not be paid would reject structures for the wrong reason.
5. **Years.** Only calendar years with ≥ 120 days in the window count. D2: same sign in ≥ 75% of counted years (H1/H2: 3 of 4, or 3 of 3 for Bund/Gilt H1; FULL: 6 of 8, or 6 of 7). K4: net > 0 in ≥ 60% of counted years (3 of 4; 2 of 3).

Unchanged from FXS 001/002: both clocks (`NY` 17:00→17:00 New York, `UTC` 00:00→24:00), the price at a boundary is the mid at the close of the minute ending there, Monday–Friday labels, intervals starting every 5 minutes and lasting 15, 30, 60, 120 or 240 minutes (**2,704 tests per symbol and window**), Newey–West standard errors with 5 lags, de-duplication across clocks in UTC time, at most 10 candidates per symbol and window, 20 rotated-day placebos.

## 5. Discovery (each window separately)

| Filter | Condition |
|---|---|
| **D1** | Benjamini–Hochberg over the symbol's 2,704 tests, q ≤ 0.05 (intervals entirely in closed hours have no variance and are not tests) |
| **D2** | same sign in ≥ 75% of counted years |
| **D3** | \|mean\| > futures-equivalent cost |
| **D0** | `STRUCTURE_FOUND` if the D1–D3 count exceeds the 95th percentile of 20 rotated-day placebos |
| reported | Bonferroni over the 3 symbols' 8,112 tests (\|t\| ≥ 4.52); `D3_cfd`; trimmed mean (1% each tail), median and share of positive days |

Candidate ids: `UST-H1-01`, `BUND-H2-03`, `GILT-FULL-02`, …

## 6. Cross-confirmation

H1 candidates are traded every H2 day, and H2 candidates every H1 day, at the mid, one futures tick and 0.70 bp commission:

| Gate | Condition |
|---|---|
| **K1** | mean after the tick > 0: Newey–West one-sided p, Benjamini–Hochberg across that window's candidates, q ≤ 0.05 |
| **K2** | net: moving-block bootstrap lower bound (0.025) > 0 at blocks of 10 and 20 days |
| **K3** | mean at 2× commission > 0 |
| **K4** | yearly mean net > 0 in ≥ 60% of counted years |

Verdicts: `CONFIRMED_FORWARD` (H1 → H2), `CONFIRMED_BACKWARD` (H2 → H1), otherwise `NOT_CONFIRMED`. Also reported: whether an H1 candidate overlaps an H2 candidate in the same direction (found in both halves independently).

## 7. FULL window

Same filters as §5 with D2 = 6 of 8 (or 6 of 7). Each FULL candidate is matched to overlapping H1 and H2 candidates and to their cross-confirmation verdicts, and its mean is shown for each year. Descriptive: a FULL candidate that was not found or not confirmed in either half is not evidence of a structure.

## 8. Readings fixed in advance

- A structure **confirmed in at least one direction and found in FULL** is the strongest result this data can give, and goes to a 2026 holdout pre-registration.
- A structure confirmed only backward (found in H2, present in H1) is weaker: it may be a recent structure, but it existed before as well, so it is not explained by a regime change in 2022.
- A structure found only in H2 or only in FULL, and not confirmed: not evidence.
- `D3` but not `D3_cfd`: tradable only through futures, not through these CFDs.
- No candidate is modified (times, direction, symbol) after its frozen file is committed.

## 9. Not tested here

Calendar conditioning (auction days, data release days, month end), bond–FX pairs or lead-lag, position sizing, trading the CFDs themselves.

## 10. Run order (per symbol)

1. `python -m pytest -q tests/test_bond_seasonality.py`
2. `<SYMBOL>/01_discovery_H1.ipynb` and `<SYMBOL>/02_discovery_H2.ipynb`; commit `01/` and `02/` (with `candidates_frozen.csv`)
3. `<SYMBOL>/03_cross_confirmation.ipynb`; commit `03/`
4. `<SYMBOL>/04_full_window.ipynb`; commit `04/`

`BDS_SYNTHETIC=1` runs the notebooks offline on synthetic candles (UST: +4 bp per day injected 08:30–09:30 New York and a −60 bp contract switch every quarter; Bund and Gilt: switches only).
