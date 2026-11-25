# HMK 003 — Readout of HMK 01 (run `6bb2ce5`): the matrices' persistence is the 20-bar overlap; states carry no next-bar direction

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Responds to** | Iván: HMK 01 run `6bb2ce5` (2020-01 – 2023-06) |
| **Status** | Descriptive readout (HMK 002); no gates |

## 1. Daily (±5%, 882 days)

| | bear | side | bull |
|---|---|---|---|
| share of days | 17.5% | 49.3% | 33.2% |
| P(stay) | 0.851 | 0.829 | 0.823 |
| P(stay), shuffled returns (mean / 95%) | 0.825 / 0.882 | 0.851 / 0.879 | 0.866 / 0.902 |
| P(stay), non-overlapping 20-day blocks | 0.29 | 0.44 | 0.21 |
| next day closes up | 52.3% (n 151) | 53.8% (424) | 55.7% (291) |

- The video-style matrix is sticky (≈ 0.82–0.85 on the diagonal), but **shuffled returns with the same rule are just as sticky** (0.83–0.87). The persistence comes from the 19 shared days, not from the market.
- On non-overlapping blocks the diagonal falls to the states' shares or below (bull → bull 0.21 vs 33% of days): no month-to-month persistence; if anything mild reversal (≈ 44 blocks, too few to say).
- Next-day up share: 52–56% in every state against 54.2% overall (SE 2.4–4.1 pp). The state does not separate up days from down days. Halves agree (p_stay 0.79–0.87 in both).

## 2. 1 hour and 15 minutes

| | 1 h ±5% | 1 h ±1.04% (scaled) | 15 m ±5% | 15 m ±0.52% (scaled) |
|---|---|---|---|---|
| share sideways | 98.9% | 59.2% | 99.9% | 67.4% |
| P(stay) bear / side / bull | 0.82 / 1.00 / 0.71 | 0.857 / 0.899 / 0.847 | 0.70 / 1.00 / 0.79 | 0.847 / 0.925 / 0.841 |
| shuffled bear / side / bull | 0.74 / 1.00 / 0.81 | 0.856 / 0.877 / 0.862 | 0.60 / 1.00 / 0.59 | 0.859 / 0.884 / 0.861 |
| non-overlapping bear / side / bull | — | 0.25 / 0.64 / 0.21 | — | 0.24 / 0.72 / 0.22 |
| next bar up bear / side / bull | — | 51.2 / 52.9 / 51.7% | — | 51.7 / 51.6 / 51.4% |

- **Literal ±5% is unusable intraday:** 99% of hours and 99.9% of quarter-hours are "sideways" (as expected in HMK 002 §3).
- **Scaled band:** bull and bear persistence equals the shuffled reference (0.85 vs 0.86). Only **sideways** persists more than shuffled (1 h 0.899 vs 95% band 0.882; 15 m 0.925 vs 0.886) and more on non-overlapping blocks (0.64 vs 59%, 0.72 vs 67%). That is calm periods clustering — volatility clustering — not trend.
- Next-bar up share is 51–53% in every state, the same as the base rate (52.3% 1 h, 51.6% 15 m; SE ≈ 0.5–0.8 pp). In the bull state the next hour is up 51.7% of the time, *less* than in the sideways state (52.9%).

## 3. Answer to the question

How persistent is it? As persistent as the rule forces it to be and no more, on every tier. The only structure beyond the overlap is that quiet stretches last longer than chance (volatility clustering), which is the same thing RSW 01–03 and the HMK 001 run found (the daily parent improved the density forecast, not direction).

The matrix does not give P(next bar up) > 50% beyond the drift. "P(bull tomorrow) = 0.82" says the 20-bar sum will most likely still be above the threshold tomorrow, because 19 of its 20 bars are already known — not that tomorrow's bar will rise.

## 4. Options for Iván

1. **Close the direction question for price-only Markov states on MNQ** (HMK 01 here, HMK 001 run, RSW 01–05, MRP 01 all agree).
2. If the volatility clustering is of interest: it is real and persistent across tiers, but it is a volatility signal (sizing, stop width, when not to trade, options), not a direction signal.
3. An HMM in place of the ±5% thresholds (the video's step 10) was effectively tested in the HMK 001 run: no direction skill at 1 h or 15 m.
