# P2 002 — Readout (run e9dedda)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-04 |
| Protocol | `001_claude_p2_liquidity_provision_preregistration.md` |
| Verdicts | 01 (H1 daily): `NO_REPLICATION`, `NOT_PROFITABLE_AFTER_COSTS`. 02 (H2, 2017–2019): `NOT_CONFIRMED`. 03 (H1 at 15m): **`EDGE_CANDIDATE`** (V1, V2, V3 all pass); prospective verdict pending. |

## 1. H1 daily (notebook 01): no replication

The Lehmann contrarian portfolio's gross return does not rise with lagged VIX: slope t = 0.11 (one-sided p = 0.46). The high-regime dummy gives t = 1.23 (p = 0.11). By VIX quintile, mean gross is 1.3, 0.0, −3.6, 3.6 and 3.9 bps/day: not monotone. Net of costs the portfolio loses 6.2 bps/day ungated and 1.8 gated. The daily one-session-skip reversal is too small in 2016–2025 for the paper's effect to show.

## 2. H2 (notebook 02): P1's observation was noise

On 2017–2019, T3 − T1 = −0.84 bps/day (interval [−4.6, +2.8]); the shuffled-label upper-tail p = 0.80. High abnormal volume does **not** reverse more out of sample, and the order is reversed again (T1 3.5, T2 3.3, T3 2.6 bps/day gross). The P1 observation is dropped.

## 3. H1 at 15 minutes (notebook 03): edge candidate

| Arm | Net 1× | Net 2× | Gross 0× | Sharpe net | Gross / trade | Cost / trade | Avg positions |
|---|---|---|---|---|---|---|---|
| ALL | +9.9% | −9.9% | +33.4% | 0.36 | 20.2 bps | 13.0 bps | 100 |
| VIX | **+22.7%** | **+7.0%** | +40.6% | **0.82** | 35.9 bps | 13.8 bps | 61 |

- **V1.** In the ungated run, trades entered after a high-VIX close earn 42.0 bps gross; after a low close, −0.7 bps. Difference 42.8 bps, 95% [22.6, 64.7]. The whole S05 gross edge sits in high-VIX regimes.
- **V2.** The VIX arm beats 199 of 200 circularly shifted regimes on net return and on Sharpe (p = 0.010 on both).
- **V3.** Positive at 1× and 2× cost. The bootstrap lower bounds are positive but thin: 0.10, 0.19 and 0.38 bps/day at blocks 10, 20 and 40.

## 4. Robustness (descriptive, after the result)

| Year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| ALL net | +10.2% | +0.6% | +2.1% | −4.3% | −0.4% | +2.6% |
| VIX net | +12.6% | +0.5% | 0.0% | +0.3% | +3.9% | +4.9% |
| VIX gross / trade (bps) | 64.4 | 8.0 | 15.4 | 18.0 | 38.0 | 45.1 |

- **2020 dependence.** Excluding February–June 2020, VIX is +11.4% (Sharpe 0.58) against ALL +0.7% (Sharpe 0.05). The result does not rest on the COVID crash alone.
- **Year by year.** VIX is not negative in any year. It beats ALL in four of six years; 2022 is the weak year.

## 5. Caveats

1. **Thin return.** About 3.5% a year net on roughly 60% average stock exposure. The 2× cost margin is small.
2. **Selected exit rule.** `x10` exits were chosen after S05c.
3. **Multiple testing.** This is one of about fifteen judged arms since S04. A placebo p of 0.01 is strong for one test, not for a family of that size.
4. **Mechanism.** Nagel's daily effect did not replicate (notebook 01). What we found is narrower: the 15-minute sector-residual reversal pays only when VIX is above its trailing median.

## 6. Next

This is the first candidate, so it is confirmed before anything is built on it:

1. **Freeze the specification.** S05 0.1.3, `x10` exits, VIX regime with a 250-session trailing median, measured costs.
2. **Independent replication on 2017–2019 at 15 minutes.** These are data no 15-minute study has touched. Measured quote samples for those years, same gates.
3. **The 2026 holdout.** About ten months are available now, used once.
