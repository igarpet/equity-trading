# MRP 004 — Continuation after large MNQ bars on the test window (MRP 03, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-21 |
| **Branch** | MNQ_Return_Prediction |
| **Responds to** | Iván, 2026-11-21: MRP 02 discovery (run `548783c`) showed no reversal after large bars, but a continuation; test it |
| **Engine** | `reversal.py` 0.2.0 (`continuation_test`), `mnq_predict.py` 0.1.1 |
| **Status** | Written after MRP 02 discovery, before any test-window data is read. This uses up the 2023-07 – 2025-12 window; 2026 stays sealed. |

## 1. Origin and prior

The hypothesis comes from the training window, so that window is not evidence for it. In MRP 02 discovery, re-scored as continuation (trade with the large bar, 3-tick market cost), the best configurations had:

- net +4 to +20 ticks per trade, t = 1.6–2.1;
- no configuration with BH q below 0.82;
- the gain concentrated in 2022, and negative in 2023H1 for every top configuration.

The prior for confirmation is low. The test is run because it is cheap and fully specified in advance.

## 2. Window and data

- Test window 2023-07-01 – 2025-12-31.
- MRP 001 grid and exclusions.
- Bar size z from the MRP 001 volatility model, with walk-forward fits made before 2023-07-01, 2024-01-01 and 2025-01-01, expanding from 2020-01-01.

## 3. Configurations (fixed: 8)

**Event**: bar t with |z_t| ≥ k, any session.

**Trade**:
- direction **+sign(r_t)** (with the large bar);
- entry at the close of bar t + d;
- exit at the close of bar t + d + h;
- one contract;
- no overlap, and no trade across a CME day, gap or roll.

| k | d | h |
|---|---|---|
| 3, 4 | 0, 1 | 3, 6 |

All results are in ticks per contract (1 tick = $0.50).

**Cost**: market = 1 tick + 2 × Lucid commission per side (Iván confirms; $0.50 → 3 ticks). Iván's realized 1.6 ticks is reported, not used for the verdict.

## 4. Gates (per configuration)

- **C1**: at least 75 trades.
- **C2**: mean net ticks > 0, and the day-block bootstrap (5-day blocks, 5,000 draws) 95% lower bound of total net > 0.
- **C3**: mean net > 0 in at least 2 of the 3 periods (2023H2, 2024, 2025).
- **C4**: one-sided p of mean net (t clustered by CME day), Holm-adjusted across the 8 configurations, ≤ 0.05.

A configuration passing C1–C4 is **confirmed** and goes to a 2026-holdout pre-registration (one configuration, sizing under Lucid drawdown rules). The configurations share events, so a pass in one or several is one finding, not several.

Also reported:
- longs and shorts separately;
- win rate, mean win and loss;
- 5% and 1% trade quantiles, worst trade, maximum drawdown in ticks;
- per-period net.

## 5. Readings fixed in advance

- If nothing passes, the continuation seen in discovery is treated as the 2022 trend regime and not pursued as a standalone rule. A regime-conditional version would need its own pre-registration, with discovery on data other than this window.
- A pass at 1.6 ticks but not at the market cost is not a pass.

## 6. Run

`notebooks/MNQ_Return_Prediction/06_continuation_test.ipynb`, once. It refuses to run if `06/summary.json` already exists or the commission is not confirmed.
