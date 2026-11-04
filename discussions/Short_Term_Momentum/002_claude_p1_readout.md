# P1 002 — Readout: reversal by abnormal turnover (run a7d07f0)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-04 |
| Protocol | `001_claude_p1_turnover_reversal_preregistration.md` |
| Verdicts | Arm D: `NO_REPLICATION`, `NOT_PROFITABLE_AFTER_COSTS` at k = 21, 5, 1. Arm I: `NO_EDGE` (I1, I2, I3 all false). |

## 1. Pre-registered result

The paper's claim does not hold in our sample with abnormal volume (AV) as the turnover proxy.

**Arm D (daily, 501 stocks, 2020–2025).** Reversal gross, bps per day per $1 long / $1 short:

| k | T1 low AV | T2 | T3 high AV | T1 − T3 | Shuffled-label percentile |
|---|---|---|---|---|---|
| 21 (primary) | 0.04 | 0.60 | 0.69 | −0.64 | 5.4 |
| 5 | 1.16 | 2.56 | 5.39 | −4.23 | 0.0 |
| 1 | 2.35 | 2.00 | 0.18 | +2.17 | 86.8 |

No horizon passes D1. T3 never shows momentum (D2). T1 is net negative everywhere (D3).

**Arm I (15 minutes, S05 engine, `x10` exits).**

| Arm | Net 1× | Net 2× | Gross 0× | Gross per trade (bps) | Cost per trade (bps) |
|---|---|---|---|---|---|
| ALL | +9.9% | −9.9% | +33.4% | 20.2 | 13.0 |
| T1 | −3.8% | −23.0% | +18.9% | 11.5 | 13.3 |
| T2 | +8.0% | −11.7% | +33.7% | 17.9 | 12.7 |
| T3 | +16.5% | −5.4% | +44.3% | 23.5 | 13.5 |

I1: T1 − T3 per trade in the ALL run is −4.8 bps, 95% [−28.6, +18.3]. I2: T1 sits below all 200 random-membership placebos (percentile 0). I3: T1 is net negative.

## 2. Observation (not pre-registered)

At both horizons the ordering is the **opposite** of the paper: high-AV stocks reverse more, low-AV stocks less.
- Daily k = 5: the observed T1 − T3 is below all 500 shuffled-label placebos (lower-tail p = 0.002). At k = 21 the lower-tail p is 0.056; at k = 1 the sign flips back and nothing is significant.
- 15 minutes: masked to one tercile, gross per trade rises from T1 (11.5 bps) to T3 (23.5 bps). Against the T1-sized random-membership placebos, T3 ranks at the 87.5th percentile on net return (p = 0.13) and the 80th on Sharpe. Not significant.

This was found after the result, so it is a hypothesis, not evidence. It fits Nagel (2012), paper P2: reversal pays for providing liquidity, and a volume shock marks liquidity demand. AV also differs from the paper's variable: terciles are barely ordered by dollar volume (medians 19.1 / 19.3 / 19.4 log $) and do not persist (16% of T1 names are still in T1 21 sessions later, below the one-third of random relabelling). AV measures transient volume shocks, not a stock's turnover level, so "no replication" here is about AV, as the protocol states.

## 3. Economics

Nothing changes the cost picture. The best arm (T3 at 15 minutes) earns 23.5 bps gross per trade against 13.5 bps cost, gross/cost 1.69, and loses money at 2× cost. The daily k = 5 T3 portfolio earns 3.7 bps per day net (Sharpe 0.43) and 1.9 at 2×, but its interval includes zero.

## 4. Consequence

- P1 closes as pre-registered: no replication, no edge.
- The high-AV reading goes into P2 (Nagel) as a pre-registered condition: liquidity-demand shocks (AV, and market volatility) predicting larger reversal, judged against random-membership placebos of the same size, at daily k = 5 and 15 minutes.
- Housekeeping: 85 ALL-run trades had no label (first days before AV coverage); excluded from I1, not material.
