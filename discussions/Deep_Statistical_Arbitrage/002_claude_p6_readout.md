# P6 002 — Readout (run ccb77b2)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Protocol | `001_claude_p6_deep_stat_arb_preregistration.md` |
| Verdict | **`NO_EDGE`**. G1 and G3 fail in both windows. G2 passes in A and fails in B (Sharpe). |
| Run | 19 rolling fits × 5 seeds on the GTX 1070 (PyTorch 2.14 cu126), about 1–1.5 min per seed |

## 1. Out-of-sample result (every day predicted by a model trained only on earlier data)

| Window | Strategy | Gross / day | Cost / day | Net / day | Net total | Net Sharpe | Gross Sharpe |
|---|---|---|---|---|---|---|---|
| B 2017–2019 | **Network** | 0.63 bps | 0.51 | +0.12 | +0.8% | 0.12 | 0.63 |
| | OU | −0.65 | 1.34 | −1.99 | −13.9% | −2.76 | −0.90 |
| | Linear 5-day | 0.82 | 1.33 | −0.51 | −3.9% | −0.37 | 0.60 |
| A 2020–2025 | **Network** | 0.96 | 0.75 | +0.21 | +2.7% | 0.13 | 0.57 |
| | OU | −0.26 | 1.91 | −2.17 | −28.2% | −1.83 | −0.22 |
| | Linear 5-day | 1.08 | 1.77 | −0.69 | −10.7% | −0.30 | 0.48 |

- **G1** fails. The bootstrap intervals of mean daily net are [−1.1, +1.4] bps (B) and [−1.3, +1.8] bps (A).
- **G2:**
  - **A passes.** Gross return beats all 1,000 weight permutations (p = 0.001), and gross Sharpe p = 0.031.
  - **B fails.** Gross return p = 0.035, but gross Sharpe p = 0.16.
- **G3** fails: the network is net negative at 2× cost in both windows.

## 2. What the network learned

1. **To trade less.** Its daily turnover is 0.26 (Σ|Δw| at Σ|w| = 1), against 0.72 for OU and 0.63 for the linear signal. The cost-aware objective did what it should: costs fell from about 1.8 to about 0.6 bps a day. This is the only reason its net is positive while both benchmarks lose.
2. **No more gross than a 5-day reversal.** Its gross (0.63 / 0.96 bps a day) is at or below the linear benchmark's (0.82 / 1.08). The CNN + Transformer found a lower-turnover version of short-term residual reversal, not new structure.
3. **The signal decays over time.** The mean validation Sharpe across seeds is 0.4–1.2 for fits through 2020 and −0.4 to −0.9 for the last four fits (2024–2025). Net by year is −4.4% in 2024 and 0.0% in 2025.
4. **Fits are unstable.** Validation Sharpes across seeds span up to 5 Sharpe units within a fit (fit 10: −3.3 to +2.0).

## 3. Against the paper

The paper reports a gross Sharpe of 3.4 for its PCA-5 CNN + Transformer (2002–2016, point-in-time largest 550 stocks, trading at the signal's own close). Ours is 0.6. This study cannot separate the causes:
- the one-day execution skip (the paper notes that only about half its Sharpe survives a one-week holding);
- the later period;
- the survivorship-biased universe;
- PCA residuals from a 252-day projection rather than 60-day regressions.

A same-close (skip 0) run would measure the first. That is a diagnostic only, since it cannot be traded.

## 4. Consequence

P6 closes `NO_EDGE`. The useful finding is methodological: training on **net** returns with measured costs gives a book that trades far less and is the only daily residual strategy here with non-negative net in both windows. The underlying daily residual reversal is too small, and fading in 2024–2025, for that to become an edge.
