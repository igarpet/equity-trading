# P4 002 — Readout (run 0d41706)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-05 |
| Protocol | `001_claude_p4_news_filtered_reversal_preregistration.md` |
| Verdicts | 01 (daily): `NO_REPLICATION`, `NOT_PROFITABLE_AFTER_COSTS` at k = 21 and 5. 02 (15m): `NO_EDGE`. **Q1 and Q2 pass in both windows; Q3 fails in both.** |

Data:
- **EDGAR:** 19,921 earnings 8-Ks for 498 stocks, 4.2 per stock-year in every year 2016–2025.
- **Flagged stock-days:** 4.7%.
- **Alpaca news:** 546,082 articles; 2.3% of stock-days are news shocks, and 53% of those fall inside earnings windows.

## 1. Daily (notebook 01): no replication

Gross, bps/day:

| k | Window | R | I | IN | IN − I | Placebo percentile | I − R |
|---|---|---|---|---|---|---|---|
| 21 | A 2020–2025 | 0.21 | 0.23 | 1.04 | +0.81 | 100 | +0.02 |
| 21 | B 2017–2019 | 3.94 | 1.90 | 2.57 | +0.67 | 100 | −2.03 |
| 5 | A 2020–2025 | 2.87 | 2.30 | 2.43 | +0.13 | 84.5 | −0.57 |
| 5 | B 2017–2019 | 3.30 | 2.46 | 3.26 | +0.80 | 100 | −0.84 |

- **F1 (news filter).** The sign is the same in all four cells, and at k = 21 the filter beats all 200 random exclusions in both windows (p = 0.005). But the bootstrap lower bound is below zero everywhere except k = 5 in B. The filter's effect is real relative to removing random names, but too small against the daily noise.
- **F2 (industry adjustment).** Fails everywhere and is *negative* in B: subtracting the sector ETF lowers the reversal. This is the opposite of the paper.
- **F3 (economic).** Fails everywhere.
- **Secondary arms.** Excluding Alpaca news shocks adds nothing: IA ≈ I, and INA ≈ IN.

## 2. 15 minutes (notebook 02): the earnings filter works in both windows, but not enough

| Window | ALL net | EX net | EX net 2× | EX gross | Gross / trade inside windows | Gross / trade outside |
|---|---|---|---|---|---|---|
| A 2020–2025 | +9.9% (Sharpe 0.36) | **+21.4%** (0.66) | −0.4% | +49.6% | **−13.0 bps** (3,717 trades) | +30.1 bps (12,404) |
| B 2017–2019 | −6.1% (−0.72) | **−1.0%** (−0.11) | −8.7% | +7.7% | **−17.7 bps** (1,973) | +8.2 bps (5,802) |

- **Q1 passes in both windows.** Outside minus inside is 43 bps per trade, 95% [21, 66], in A, and 26 bps [1.2, 49.5] in B.
- **Q2 passes in both windows.** EX beats 100 of 100 random exclusions of the same size on net return and Sharpe (p = 0.0099).
- **Q3 fails in both windows.** In A, the bootstrap lower bound is negative at blocks 10 and 20 (positive at 40), and 2× cost loses 0.4%. In B, the net is negative.
- **By year, EX vs ALL.** EX is better in 7 of 9 years; the exceptions are 2020 and 2021.

**Mechanism.** Earnings windows are 4.9% of stock-days but hold 23–25% of S05's trades. The book takes the largest |s| first, and earnings moves produce the largest residuals. Those moves do not revert, consistent with post-earnings drift, so fading them loses money. This is Da, Liu & Schaumburg's point in reverse: our reversal signal was partly news, and the news part does not reverse.

## 3. Consequences

1. **P4 closes `NO_EDGE`.** The S05 residual still does not clear costs in 2017–2019 even with the filter; outside-window trades there earn 8.2 bps against about 10 bps of cost.
2. **The earnings-window exclusion is the first S05 modification that improves results, as pre-registered, in both windows.** It becomes a **standard component** of any later reversal study, alongside P3's time-of-day control. It is stated in each protocol, not re-tuned.
3. No combination with the VIX gate is tested on these data: it would be post hoc, and the VIX gate did not replicate.
