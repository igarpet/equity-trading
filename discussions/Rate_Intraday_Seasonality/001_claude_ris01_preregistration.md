# RIS 001 — Stacked 1-minute days in US rate futures (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-19 |
| **Branch** | Rate_Intraday_Seasonality (new) |
| **Responds to** | Iván, 2026-11-19: no structure in the notes–FX pairs (RFX 001); check the individual notes for structure |
| **Engine** | `bond_seasonality.py` 0.3.0 (BDS design), `rates.py` 0.2.0 |
| **Status** | Written before any rate-futures seasonality scan. Gates are not changed after results exist. |

## 1. Question

The BDS question (is there an interval of the day whose mean return is non-zero, consistent across years, larger than its cost, and present in the other half?) asked of the US short-rate futures themselves rather than of FX.

## 2. Instruments

| Future | Data | H1 | H2 | FULL |
|---|---|---|---|---|
| ZT (2-year T-Note) | Databento `bbo-1m`, front contract by volume (`ZT.v.0`), as downloaded for RFX | 2016–2020 | 2021–2025 | 2016–2025 |
| ZF (5-year T-Note) | idem (`ZF.v.0`) | 2016–2020 | 2021–2025 | 2016–2025 |
| SR3C3 (SOFR, 4th quarterly contract, about 9–12 months ahead) | `SR3.c.3`, **new download** | 2022–2023 | 2024–2025 | 2022–2025 |

**Why SR3C3 and not the RFX SOFR series.** The RFX download used `SR3.v.0` (front contract by volume). The 00 run showed **217 contract changes in 2022–2023**: the most traded SOFR contract is usually not the front one and the lead changes between maturities, so that series hops along the curve. The RFX SOFR results are therefore not interpretable and are not used. `SR3.c.3` holds one maturity point (rolling quarterly at expiry) and is among the most traded SOFR contracts. Download: `python tools/databento/rates_download.py --roots SR3C3` (about $2–3, same ledger and budget).

2026 is sealed.

## 3. Design

As BDS 001 (with the BDS 002 spike screen), except:

- **Rolls:** found from the instrument id of the continuous symbol (no inference); the days containing a roll are dropped in each clock.
- **Costs:** these are the futures' own quotes, so D3 and the confirmation trades use the quoted bid/ask (median spread at the interval's start and end) + 0.70 bp commission, not an assumed tick.
- **Year rules:** D2 same sign in ≥ 75% of counted years (ZT, ZF: 4 of 5 per half, 8 of 10 FULL; SR3C3: 2 of 2 per half, 3 of 4 FULL); K4 net > 0 in ≥ 60% (ZT, ZF: 3 of 5; SR3C3: 2 of 2).
- **Pooled report:** Bonferroni over the 3 futures' 8,112 tests (|t| ≥ 4.52), reported, not a gate.

Both clocks (New York 17:00 → 17:00, UTC), 2,704 intervals per future and window (start every 5 minutes, 15–240 minutes), D1 BH q ≤ 0.05 within the future, D2, D3, D0 rotated-day placebos, cross-confirmation H1 ↔ H2 with K1–K4, FULL matched to the halves.

## 4. Note on the data (from the RFX 00 runs)

In 5-minute bars, ZT's mid does not move in about 74% of returns and ZF's in about 42%. This is not staleness: ZT moves less than one tick (1/256 point, about 0.37 bp) in most 5-minute intervals. Intervals of 15–240 minutes are long enough for the scan, but very small drifts will be swallowed by the spread, which the D3 gate measures directly.

## 5. Run order

1. `python -m pytest -q tests/test_bond_seasonality.py tests/test_rates.py`
2. `python tools/databento/rates_download.py --roots SR3C3 --dry-run`, then without `--dry-run`
3. Per future (`ZT/`, `ZF/`, `SR3C3/`): `01_discovery_H1`, `02_discovery_H2`, commit `01/` and `02/`, then `03_cross_confirmation`, `04_full_window`.

`RIS_SYNTHETIC=1` runs offline on synthetic quotes (ZT +3 bp/day 08:30–09:30 New York; quarterly −30 bp rolls).
