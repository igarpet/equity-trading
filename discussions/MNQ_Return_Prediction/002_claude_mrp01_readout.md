# MRP 002 — Readout of MRP 001 (runs `10f6a0b`, `0e0c3df`)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-21 |
| **Protocol** | [`001_claude_mrp01_preregistration.md`](001_claude_mrp01_preregistration.md) |
| **Verdict** | Gate V **FAIL** (V2) → primary scale raw · Gate S **FAIL** · test window not opened |

## Data (00)

- 2020-01-01 – 2023-06-30: 884 CME days, 243,984 bars. 237,183 targets remain after every exclusion: first 3 bars of each day, gaps and rolls, and 5 short days.
- Empty-bar share 1.9% in 2020, falling to 0.8% in 2023. FX is present on every bar.
- **To check:** 2022 has 237 CME days against 259 in 2020 and 2021, so about 20 days appear to be missing (download gap?). Not material to the verdict; to be checked before the test window is used.
- Timestamp alignment: the 1-minute MNQ–FX correlation peaks at an offset of 0 for EURUSD and USDJPY.

## Volatility (01)

- **V1 passes.** QLIKE −13.83 for the full model, against −13.21 for time of day and −12.84 for a constant. Diebold–Mariano against time of day: t = 14.5 over 622 days.
- **V3 passes.** The mean ACF of z² at lags 1–12 is 0.006, against 0.056 for r².
- **V2 fails.** var(z) was 0.77 (2021H1), 0.65 (2021H2), 1.06 and 1.05 (2022) and 0.85 (2023H1). The level is too high in calm periods: the first fit saw only 2020, and the expanding window keeps that level. The volatility ranking is strong; the level is biased. Any fix is post hoc and belongs to a new pre-registration.
- HAR weights are stable across the three fits: about 0.68 on the last hour, 0.09 on the last day, 0.17 on the last week.

## Mean model (02, raw scale)

- **No group survives Holm.** A own lags p = 0.026 (Holm 0.18); F FX p = 0.032 (0.19); every other group p ≥ 0.13. In the z version A has p = 0.0008, B 0.017, C 0.020, but z is the secondary scale.
- **Hypotheses:**
  - B (large-move continuation > 0): not supported, coefficient +0.009, p = 0.31, near zero in the first half and under Huber.
  - D (previous CME day strongly ≠ 0): rejected, coefficient −0.0004, p = 0.28, sign unstable.
  - F (FX = 0): supported.
- **Only stable coefficient:** lag 2 (the bar before last), −0.0175, t = −3.0, same sign under Huber, in both halves and in z. In ticks, at MNQ ≈ 13,000, that is an expected reversal of ≈ 0.9 ticks after a 1σ bar (≈ 52 ticks) and ≈ 2.7 ticks after a 3σ bar, against a round trip of 1.6 ticks (Iván's realized cost) to 3 ticks (1 tick plus $0.50 commission per side).
- **Walk-forward 2021-01 – 2023-06:**
  - R²_OOS against zero is negative for every model and every period: M1 −0.08%, M2 −0.12%, M3 −0.22%.
  - Adding the FX block lowers R² by a further 0.10%.
  - Clark–West p ≥ 0.13, and every Brier score is above 0.25.
- **Correction:** the interval coverage printed by 02 (3.6% / 12.6%) is wrong. In raw scale it assumed unit variance. Fixed in engine 0.1.1; gate S, P(up) and Brier are not affected.

## Readings

- Linear next-bar prediction of MNQ from its own history and the FX majors has no out-of-sample value at 5 minutes. This closes the variants that only add lags or regressors.
- The volatility model forecasts magnitude well; candidate use: sizing and stops.
- The lag-2 reversal is real but averages below cost. Whether it is concentrated after large bars, by session or with a delayed entry is tested in MRP 02 ([`003`](003_claude_mrp02_conditional_reversal_preregistration.md)), with discovery on the training window and one look at the test window.
