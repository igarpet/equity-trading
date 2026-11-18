# BFX 002 — Amendment to BFX 001: windows re-split on 2020–2025

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-18 |
| **Amends** | BFX 001 §3 (windows) and the year rules that depend on them |
| **Responds to** | Iván, 2026-11-18: re-split on 2020–2025 after the 00 data notebook |
| **Engine** | `bond_fx.py` 0.2.0 |
| **Status** | Decided after `00_data` (descriptive) only. No discovery, scan or placebo has been run on real data. |

## Why

The month-by-month diagnostic in `00_data` (run `c8bfb8a`) showed that Dukascopy's bond CFD quotes are unusable before 2020:

| | Before 2020 | From February 2020 |
|---|---|---|
| US T-Bond | no data July–December 2018; thin January–February 2019; 5-minute returns with no price change: 1–3% until June 2019, then **23–43%** July–December 2019 | 2–3% |
| Bund | no price change in 5–17% of returns | about 1% |
| Gilt | no price change in 8–19% of returns | about 1% |

The change is abrupt in January–February 2020 for all three bonds, consistent with a change in the data feed rather than in the markets. Stale bond prices weaken the measured FX–bond link and create spurious lead–lag (a stale bond price catching up looks like a bond move that FX follows).

## Change

| Window | BFX 001 | BFX 002 |
|---|---|---|
| H1 | 2018–2021 | **2020–2022** |
| H2 | 2022–2025 | **2023–2025** |
| FULL | 2018–2025 | **2020–2025** |

2026 stays sealed. Everything else in BFX 001 is unchanged. With 3-year halves the year rules become: E3 (≥ 75% of counted years) = 3 of 3 in a half, 5 of 6 in FULL; K4 (≥ 60%) = 2 of 3.

## Side note on BDS 001/002

The bond seasonality study (BDS) used 2018–2021 as its first half, so its first-half scans were partly on the same stale data. Its second half (2022–2025) and full window found no structure either, so its conclusion does not change.
