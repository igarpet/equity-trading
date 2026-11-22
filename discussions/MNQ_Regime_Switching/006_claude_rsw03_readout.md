# RSW 006 — RSW 03 readout: regimes get stickier with age; no timing or direction gain

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Run** | `c3a9f58` (Iván), engine `hsmm.py` 0.1.0 |
| **Protocol** | [`005`](./005_claude_rsw03_duration_preregistration.md) |

## Result

| Part | Gates | Verdict |
|---|---|---|
| A (hazard) | HA fails. The hazard **falls** with age: slope −0.145 [−0.184, −0.109], against null slopes of −0.04 to 0.00 | `FALLING_HAZARD` (not identified) |
| B (HSMM vs HMM) | HB1 passes for WB2, NB2, WB3, NB3 (Holm p 0.007–0.02; about 0.0004 nats per bar). **HB2 fails** (Holm p 0.14–0.55). HB3 passes: every shape is < 1 | `DURATION_DEPENDENCE_IN_DENSITY_ONLY` |
| C (direction by age) | HC1 passes (calm b ≈ −0.01 to −0.06 per log-bar, t −7 to −19). HC2 fails (R²_OOS < 0 against zero and against B). HC3 fails (net −2.0 to −3.4 ticks per trade, negative in every period) | `NO_DIRECTION_BY_AGE` |

## Readings

- **Durations show negative dependence: the longer a regime lasts, the less likely it is to end.**
  - Calm (WB2): the chance of leaving is 0.19–0.27 at age 1 and 0.07–0.08 at age 10 (β = 0.62–0.75).
  - Burst: 0.69–0.80 at age 1. A burst that survives its first bar lasts longer (0.24–0.49 at age 10).
  - Counting bars in calm therefore does not point to a coming switch. After a long calm a switch is less likely, not more.
- **Negative duration dependence is what heterogeneity produces** (calm periods of different lengths across low- and high-volatility days). The Markov smoke run passed HB3 and nearly HB1. The small density gain does not show that duration itself matters.
- **Age drift (C):** early calm has a positive drift that fades with age, i.e. a rebound after selloff bursts. It is strong in sample but forecasts nothing out of sample.
- **The 3-state fits are unstable.** WB3 and NB3 did not converge in 300 iterations in the last fit, and the calm mean duration fell from about 11 to 3.5 bars. The 2-state fits are consistent across the three fits.
- **Detection limit** (Iván's question about slow calm trends): with drift c and calm noise σ ≈ 0.8 (z), the sign is known at t = 2 only after about (2σ/c)² bars. For the observed c ≈ 0.03 that is about 2,800 bars (10 days), against calm spells of 10–13 bars.

## Next

Iván (2026-11-22): trade against the burst at the switch back to calm, with a time stop at the mean calm duration + 5 bars and several stop losses. RSW 04, [`007`](./007_claude_rsw04_burst_reversion_preregistration.md).
