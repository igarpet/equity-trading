# P8 002 — Readout (run d63819a)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-07 |
| Protocol | `001_claude_p8_ou_optimal_thresholds_preregistration.md` |
| Verdicts | `BT`: `NO_IMPROVEMENT`, `NO_EDGE`. `LP`: `NO_IMPROVEMENT`, `NO_EDGE`. No gate passes in either window. |

## 1. Result

| Window | Arm | Net 1× | Net 2× | Gross 0× (Sharpe) | Trades / day | Median bars | Gross / trade | Cost / trade |
|---|---|---|---|---|---|---|---|---|
| A 2020–2025 | Incumbent (`x10` + earnings filter) | +21.4% | −0.4% | +49.6% (1.31) | 11.4 | 208 | 25.9 bps | 13.1 bps |
| | `BT` | +18.0% | −9.6% | +58.0% (1.43) | 15.4 | 143 | 21.6 | 13.2 |
| | `LP` | **−52.3%** | −82.0% | **+68.2% (1.45)** | 70.6 | 26 | 5.2 | 12.5 |
| B 2017–2019 | Incumbent | −1.0% | −8.7% | +7.7% (0.83) | 11.4 | 207 | 8.9 | 10.1 |
| | `BT` | −3.8% | −13.7% | +7.3% (0.77) | 15.5 | 142 | 6.6 | 10.0 |
| | `LP` | **−26.1%** | −55.4% | **+24.8% (2.09)** | 75.2 | 26 | 4.2 | 9.9 |

- **G2 (improvement over the incumbent):** `BT` is −0.2 (A) and −0.4 (B) bps/day, with intervals spanning zero. `LP` is −6.3 and −4.0 bps/day, with intervals below zero.
- **G1 and G3:** fail for both arms in both windows.

## 2. Reading

**Bertram's band is the incumbent with tighter exits.**
- The round-trip cost in σ units is small: median c̃ = 0.05. So $m^* \approx 0.43$.
- Because the book is always full, the entry threshold rarely binds (as in S05d). The real change is the exit at ±0.43 instead of ±1.0.
- That gives 35% more trades, shorter holds and lower gross per trade, for a net slightly below the incumbent's.
- The OU model's optimum adds nothing over the hand-set rule.

**LP finds the fastest part of the reversion, but it is too small to pay for.**
- For small c̃ the table chooses to exit **before** the mean (take-profit at s = −1 for a long), with a 3-half-life limit.
- Its gross is the highest of any S05 variant in both windows. The gross Sharpe of 2.09 in 2017–2019 is notable, because the window has no gross edge under the slower rules.
- But it trades 70–75 times a day at 4–5 bps gross against 10–12.5 bps cost.
- This agrees with P3: short-horizon reversal is real in both windows, and its size per trade is below the spread.

**Both rules assume the OU model's own cost-to-edge ratio.**
- A c̃ of 0.05 says one σ_eq of reversion is worth about 20 round trips.
- The realised gross per trade (5–26 bps), against σ_eq of a few hundred bps, says the model overstates reversion by an order of magnitude. This is the S08 finding (heavy tails, weak information coefficient).
- Thresholds optimised inside a mis-specified model cannot fix the model.

## 3. Consequence

P8 closes. The incumbent (S05 `x10` with the earnings filter) remains the best S05 rule. It still has no edge in 2017–2019.

**Open lead, not tested:** `LP`'s fast-exit gross (Sharpe 1.45 / 2.09 in both windows) is the only S05 variant with consistent gross in both windows. It would need execution costs below about 4 bps per round trip. S06 showed that passive limit orders lose about 25 bps per trade to adverse selection on this signal, so no available execution route reaches that.
