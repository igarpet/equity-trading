# FXS 002 — Amendment to FXS 001: one study per pair

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-08 |
| **Amends** | FXS 001 (§4 discovery, §5 confirmation) |
| **Responds to** | Iván, 2026-11-08: analyse each pair separately, starting with EURUSD while the other pairs download; structure is likely pair-specific |
| **Engine** | `fx_seasonality.py` 0.2.0 |
| **Status** | Written before any real candle data has been analysed. No result exists. |

## Changes

| | FXS 001 | FXS 002 |
|---|---|---|
| D1 family | Benjamini–Hochberg over all 40,560 tests (15 pairs × 2 clocks × 1,352 intervals) | BH over **one pair's 2,704 tests** (2 clocks × 1,352), q ≤ 0.05 |
| Candidates | ≤ 40 over all pairs | ≤ **10 per pair** |
| D0 placebo | pooled | **per pair**, 20 rotated-day placebos of that pair |
| Frozen file | one | **one per pair**: `notebooks/FX_Intraday_Seasonality/01/<PAIR>/candidates_frozen.csv`, committed before that pair's confirmation |
| K1 family | BH across all candidates | BH across **that pair's** candidates |
| Order | all pairs together | any order; a pair is analysed only once its files are complete |

D2, D3, K2–K4, windows, clocks, interval grid and costs are unchanged.

## Cost of the change, stated in advance

Per-pair families are less strict than one pooled family. BH still controls the false-discovery proportion within each pair at 5%, but over 15 pairs some pairs will produce false candidates. Two guards:

1. Each candidate also reports whether it passes Bonferroni over the **pooled** 40,560 tests (|t| ≥ 4.84). This is reported, not a gate.
2. Confirmation on 2021–2025 remains the binding test for every candidate.

The final readout counts confirmed candidates across all pairs and compares that count with what per-pair false discoveries alone would produce.

## Completeness

Discovery for a pair runs only when ≥ 99% of its expected UTC day files for 2015-12-31 → 2020-12-31 exist (Saturdays excluded); otherwise the code stops with the count. Confirmation applies the same rule to 2020-12-31 → 2025-12-31. Missing days are listed, and marked when the downloader logged them as gaps. Minute caches are keyed on the file set, so a cache built during a download is never reused.

## Notebooks

`01_discovery_<PAIR>.ipynb` and `02_confirmation_<PAIR>.ipynb` for each of the 15 pairs. They are identical except for the pair. The pooled notebooks from FXS 001 are removed; they were never run.
