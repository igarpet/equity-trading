# HMK 005 — Readout of HMK 02 (run `c50a7b9`): longer windows and the HMM make the chains stickier, not more informative

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Responds to** | Iván: HMK 02 run `c50a7b9` (2020-01 – 2023-06) |
| **Status** | Descriptive readout (HMK 004) |

## 1. Levels the HMM chose (state means of the window return; thresholds = midpoints)

| Case | bear mean (sd) | side mean (sd) | bull mean (sd) | thresholds | set band |
|---|---|---|---|---|---|
| 1h W40 | −2.36% (1.52) | +0.21% (0.62) | +2.42% (1.37) | −1.08 / +1.32% | ±1.47% |
| 1h W60 | −2.27% (1.76) | +0.69% (0.68) | +3.20% (1.48) | −0.79 / +1.95% | ±1.81% |
| 15m W60 | −0.65% (2.19) | −0.11% (0.34) | +0.90% (0.41) | −0.38 / +0.40% | ±0.90% |
| 15m W80 | −0.74% (2.55) | −0.19% (0.39) | +1.02% (0.47) | −0.46 / +0.41% | ±1.04% |

The 1 h HMM lands close to the set bands. On 15 m the "bear" state has a far larger sd (2.2–2.6%) than the others: it is partly a high-volatility state (2020 crash, 2022), not just a down-trend.

## 2. Persistence

| Case | P(stay) bear / side / bull | shuffled reference | non-overlapping bear / side / bull (shares) |
|---|---|---|---|
| 1d 20 ±5% | 0.85 / 0.83 / 0.82 | 0.83 / 0.85 / 0.87 | 0.29 / 0.44 / 0.21 (0.18 / 0.49 / 0.33) |
| 1h W40 band | 0.90 / 0.93 / 0.90 | 0.90 / 0.91 / 0.91 | 0.22 / 0.61 / 0.19 (0.19 / 0.58 / 0.24) |
| 1h W40 HMM | 0.96 / 0.95 / 0.96 | 0.96 / 0.94 / 0.96 | 0.36 / 0.53 / 0.26 (0.27 / 0.44 / 0.29) |
| 1h W60 band | 0.91 / 0.93 / 0.91 | 0.92 / 0.93 / 0.93 | 0.20 / 0.66 / 0.23 (0.18 / 0.58 / 0.24) |
| 1h W60 HMM | 0.98 / 0.96 / 0.97 | 0.97 / 0.95 / 0.97 | 0.45 / 0.56 / 0.18 (0.37 / 0.39 / 0.24) |
| 15m W60 band | 0.92 / 0.95 / 0.91 | 0.92 / 0.93 / 0.92 | 0.26 / 0.64 / 0.22 (0.18 / 0.61 / 0.21) |
| 15m W60 HMM | 0.97 / 0.96 / 0.96 | 0.97 / 0.95 / 0.97 | 0.41 / 0.45 / 0.33 (0.29 / 0.40 / 0.31) |
| 15m W80 band | 0.93 / 0.95 / 0.93 | 0.93 / 0.94 / 0.94 | 0.23 / 0.66 / 0.21 (0.19 / 0.59 / 0.22) |
| 15m W80 HMM | 0.97 / 0.97 / 0.97 | 0.98 / 0.96 / 0.98 | 0.37 / 0.42 / 0.34 (0.28 / 0.38 / 0.34) |

- Longer windows push the diagonal to 0.90–0.98, and the shuffled reference rises with it: **bull and bear persistence is the window overlap** in every case (bull is at or below the shuffled value in all nine).
- Sideways persists slightly more than shuffled in every intraday case (by 0.01–0.02) and on non-overlapping blocks: calm periods cluster.
- Non-overlapping blocks: bull → bull is at or **below** the bull share everywhere (strongest: 1h W60 HMM, 0.18 vs 0.24; bull → bear 0.45). Bear → bear is above its share with the HMM (0.37–0.45 vs 0.27–0.37) and in 15m W60 band (0.26 vs 0.18) — consistent with the HMM bear state being the high-volatility state.

## 3. Next bar up, by state (descriptive)

| Case | bear | side | bull | all bars |
|---|---|---|---|---|
| 1h (all four) | 51.4–52.0% | 52.4–52.9% | 51.4–52.6% | 52.3% |
| 15m (all four) | 50.8–51.6% | 51.6–52.0% | 51.1–52.1% | 51.6% |

Every state is within about 1 pp of the base rate (SE 0.2–0.8 pp). The bull state is never above the base rate by more than 0.3 pp. Both halves show the same ordering-free picture: H1 (2020–21) is higher in every state because of the drift, H2 lower in every state.

## 4. Conclusion

Longer windows and the HMM do what Iván intended — they find trend levels that look sensible (1 h bull ≈ +2.4–3.2%, bear ≈ −2.3% over 40–60 hours) and states that last a day or more — but the persistence is still what overlapping windows produce from shuffled returns, and the state still says nothing about the next bar's direction on 1 h or 15 m. The only structure beyond the overlap is volatility: calm periods and the high-volatility "bear" state cluster.
