# Rank Space — paper P7

**Paper.** Li, Y.-F. & Papanicolaou, G. (2024). Statistical Arbitrage in Rank Space. arXiv 2410.06568.

**Claim to test.** Residuals of capitalization-rank portfolios mean-revert more reliably than name-space residuals; reported results assume 2 bps cost.

**Our question.** Does rank-space mean reversion survive our measured costs (≈ 13 bps round trip in S05b), not 2 bps?

| Notebook | Description | Status |
|---|---|---|
| `01_ranks_and_training.ipynb` | Point-in-time caps (SEC shares × split-only prices, split basis, unit check), ranks 1–400, rank returns, PCA-1 residuals; CNN+Transformer fits for `RS-paper` (gross mean-variance on rank returns) and `RS-net` (net Sharpe of the tradable rank-mapped book); 19 fits × 3 seeds each, cached | Built; awaiting run |
| `02_evaluation.ipynb` | Tradable book (re-mapped every close, measured costs) vs the paper's non-tradable rank-space gross; turnover split; rank OU benchmark; gates per arm and window | Built; awaiting run (after 01) |

Protocol: [`001_claude_p7_rank_space_preregistration.md`](../../discussions/Rank_Space/001_claude_p7_rank_space_preregistration.md). Reuses the P6 pipeline; needs `SEC_USER_AGENT` (as P4) and the P6 PyTorch install.

Run: `python -m pytest -q tests/test_p7.py`, then 01 (downloads SEC shares, Yahoo splits and split-only prices, Alpaca split-adjusted 1Day; minutes. Training ~2–2.5 h on the GPU, resumable) and 02 (minutes). `P7_SYNTHETIC=1` runs both offline. Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
