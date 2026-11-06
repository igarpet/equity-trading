# P7 003 — Readout (run f0b016d, correction C1)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Protocol | `001_…preregistration.md` + correction C1 (`002_…`) |
| Data gate | Rank-return s.d. / tradable s.d. = 0.71; correlation of rank vs tradable return of the same rank = 0.615; \|rank return\| > 25% on 0.0004% of rank-days. 343 filings dropped by the C1 cleaning; 4 tickers rescaled (BRK.B, CTVA, PCG, VMRK). |
| Verdict | **`NO_EDGE`** for both arms. No gate passes in either window. |

## 1. Rank-space returns are large; tradable returns are not

| Window | Strategy | Rank gross (not tradable), bps/day | Rank Sharpe | Tradable gross, bps/day | Tradable gross Sharpe | Net total |
|---|---|---|---|---|---|---|
| B 2017–2019 | `RS-paper` | +11.96 | **7.33** | +0.42 | 0.24 | −9.0% |
| | `RS-net` | +2.97 | 3.33 | +1.04 | 1.07 | −1.5% |
| | OU (rank) | +7.08 | **11.64** | +0.01 | 0.02 | −9.6% |
| A 2020–2025 | `RS-paper` | +12.42 | **4.50** | **−1.91** | −0.66 | −44.3% |
| | `RS-net` | +2.83 | 1.97 | −0.25 | −0.16 | −28.7% |
| | OU (rank) | +8.19 | **9.26** | −0.41 | −0.41 | −31.5% |

**The paper's effect, in its own metric, is reproduced.** Rank-space portfolios earn Sharpe 4.5–11.6, the order of the paper's no-cost Sharpe of 9.

**It does not survive becoming a stock book.** Holding the stocks that occupy those ranks earns roughly zero gross, or negative gross: `RS-paper` in A falls below 99% of permuted-weight placebos. This happens **before costs**, so the cost level (2 bps vs measured) is not what kills it.

## 2. Gates

- **G1** fails everywhere. The tradable net intervals are below zero in A for both arms.
- **G2** fails everywhere. The closest is `RS-net` in B: gross return at the 98th percentile of 1,000 permutations (p = 0.020), but Sharpe p = 0.062 > α = 0.025.
- **G3** fails everywhere.

All arms are net negative in 7–9 of 9 years.

## 3. Why

Rank returns mean-revert by construction. The k-th largest cap is bounded by its neighbours, and the reflections when ranks cross (the local-time terms in Fernholz's decomposition) belong to no portfolio.

A check run outside the repo: 450 independent random-walk stocks with no predictability, sorted into 400 ranks.
- Rank returns have lag-1 autocorrelation −0.24.
- A rank-space reversal earns 169% a year at Sharpe 52.
- The same weights on the stocks holding those ranks earn −0.8% a year, before costs.

The real data shows the same pattern, at a smaller scale.

**The two arms learned different things:**
- `RS-paper` learned the sorting artefact. Its validation objective was stable at 0.2–0.5 across seeds and years, because the artefact is arithmetic and does not decay.
- `RS-net`, trained on the tradable book net of costs, scored below zero in validation for most fits. It found nothing it could trade.

## 4. What was not tested

The paper re-maps intraday every 225 minutes; we re-map daily.
- Finer re-mapping captures more of the reflection term, but at more trades. The paper's own rank-space OU loses heavily at 2 bps with intraday re-mapping.
- Our tradable gross is ≤ 0 before costs at daily frequency.
- An intraday test using the 15-minute panels and intraday caps is possible but has a weak prior. It is not planned.

**The five extreme rank-return days** (2019-02-27 at 61.5%; 2016-01-04 at the Yahoo/Alpaca splice; three early-2016 days) are not material to a result that is negative before costs in both windows.

## 5. Consequence

P7 closes `NO_EDGE`. Rank-space statistical arbitrage, as specified, measures a non-tradable sorting effect.
