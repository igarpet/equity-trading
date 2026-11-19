# RFX 001 — US short-rate futures vs FX on 5-minute bars (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-19 |
| **Branch** | Rate_FX_Pairs (new) |
| **Responds to** | Iván, 2026-11-19: long bonds are driven by different factors than short-term FX; test the short end (T-bills to 5-year notes), using the Databento free credit |
| **Engine** | `bond_fx.py` 0.3.0 (BFX methods unchanged) with `rates.py` 0.1.0 |
| **Status** | Written before any rate-futures data has been downloaded. Gates are not changed after results exist. |

## 1. Question

Do US short-rate futures carry tradable intraday information about EURUSD, GBPUSD and USDJPY? Same three methods as BFX 001 (lead–lag, residual reversion, cointegration), with the long-bond CFDs replaced by the instruments that price expected policy rates.

## 2. Instruments and data

T-bills do not trade on an exchange intraday and Dukascopy has no short-maturity bond instruments, so the short end is taken from CME futures (Databento GLBX.MDP3, `bbo-1m`: best bid and offer once a minute; front contract rolled by volume, continuous symbol `<ROOT>.v.0`).

| Study | Futures | Why | Windows |
|---|---|---|---|
| **Notes** (`RFX`) | ZT (2-year T-Note), ZF (5-year T-Note) | 2-year yields track expected policy rates; complete 2016–2025 | H1 **2016–2020**, H2 **2021–2025**, FULL 2016–2025 |
| **SOFR** (`SOFR`) | SR3 (3-month SOFR futures) | the T-bill end; replaced Eurodollar futures only in 2022 (2018–2021 volume is negligible) | H1 **2022–2023**, H2 **2024–2025**, FULL 2022–2025 |

Z3N (3-year) was quoted but is not used: almost no data before 2020 and it sits between ZT and ZF.

- Quoted cost (2026-11-19): `bbo-1m` ZT $4.63, ZF $4.76, SR3 2018–2025 $2.68; about $12 in total, paid from the Databento free credit through the shared budget ledger.
- FX: Dukascopy 1-minute candles (as in FXS and BFX). The pairs are tested against the USD side only; no euro or sterling short-rate history is long enough (Eurex Schatz on Databento starts in March 2025).
- A futures price rises when US rates fall, which weakens the dollar: the expected signs are + for EURUSD and GBPUSD, − for USDJPY. They are not imposed.
- Contract rolls (instrument changes of the continuous symbol) are removed from the price series, as the CFD switches were in BFX. Days with a 1-minute price spike (BDS 002 screen) or a missing file are excluded. 2026 is sealed: nothing from 2026 is downloaded or read.

## 3. Methods, grids, costs, filters and gates

Exactly as BFX 001 §4–§7 (24 lead–lag, 12 residual, 6 cointegration configurations; FX at Dukascopy's quoted bid/ask + 0.70 bp; futures legs one tick + 0.70 bp per unit of hedge ratio; E1 ≥ 100 trades, E2 one-sided BH q ≤ 0.05 over the method's pairing × configuration tests, E3 positive in ≥ 75% of counted years, E4 best t > 0 and above every one of 5 shifted-rate placebos; K1–K4 cross-confirmation).

Ticks: ZT 1/256, ZF 1/128, SR3 0.005 (the front SR3 tick is 0.0025; the larger tick is the conservative choice).

Year rules: Notes halves (5 years) E3 4 of 5, K4 3 of 5, FULL 8 of 10; SOFR halves (2 years) E3 2 of 2, K4 2 of 2, FULL 3 of 4.

Test families: Notes 72 lead–lag, 36 residual and 18 cointegration tests per window (3 pairings); SOFR the same.

## 4. Readings fixed in advance

- The BFX result (FX follows long bonds by 5–10 minutes, about 5% of the bond-implied move, below costs) is the benchmark. If short rates matter more, the lead–lag follow-through and the per-trade mid move should be clearly larger than BFX's 0.1–0.6 bp.
- A candidate confirmed in at least one direction and also passing in FULL goes to a 2026 holdout pre-registration.
- Notes and SOFR are separate studies; a result in one is not evidence for the other.

## 5. Run order

1. `python -m pytest -q tests/test_rates.py tests/test_bond_fx.py`
2. `python tools/databento/rates_download.py --dry-run`, then `python tools/databento/rates_download.py`
3. For each study folder (`Notes/`, `SOFR/`): `00_data`, then per method `01_discovery_H1`, `02_discovery_H2`, commit `01/` and `02/`, then `03_cross_confirmation`, `04_full_window`.

`RFX_SYNTHETIC=1` runs the notebooks offline on synthetic data (Notes: USDJPY lead–lag on ZT, GBPUSD residual reversion against ZF, EURUSD cointegrated with ZT and ZF; SR3 unrelated to every pair).
