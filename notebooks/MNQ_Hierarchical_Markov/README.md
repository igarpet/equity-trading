# MNQ hierarchical Markov regimes — daily → 1 hour → 15 minutes (HMK)

**Question (HMK 01, descriptive).** What do the video's bull/sideways/bear transition matrices look like for MNQ on daily, 1-hour and 15-minute bars, and how much of their persistence is real rather than built into the 20-bar rule? No trading test.

Protocol: [`HMK 002`](../../discussions/MNQ_Hierarchical_Markov/002_claude_hmk01_transition_matrices.md) (supersedes the withdrawn HMK 001).

| Notebook | Content | Output |
|---|---|---|
| `01_transition_matrices` | 20-bar sum, ±5% (intraday also √time-scaled ±1.04% / ±0.52%); counts, P, persistence vs shuffled returns and non-overlapping blocks, P^2/3/5/20, stationary distribution, run lengths, next-bar up share by state; 2020-01 – 2023-06 and halves | `01/` |

**Data.** MNQ `ohlcv-1m` already on disk; nothing from 2023-07 onwards or 2026 is read.

**Run.** `python -m pytest -q tests/test_hmk.py`, then run `01` (a few minutes). Smoke run: `HMK_SYNTHETIC=1` (random walk; shows the rule's built-in persistence).

**Code.** `src/equity_trading/hmk.py`. The hierarchical HMM engine from HMK 001 is kept there but unused.

Status: code ready, not run.
