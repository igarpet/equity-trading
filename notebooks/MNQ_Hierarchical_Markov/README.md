# MNQ hierarchical Markov regimes — daily → 1 hour → 15 minutes (HMK)

**Question (HMK 01, descriptive).** What do the video's bull/sideways/bear transition matrices look like for MNQ on daily, 1-hour and 15-minute bars, and how much of their persistence is real rather than built into the 20-bar rule? No trading test.

Protocol: [`HMK 002`](../../discussions/MNQ_Hierarchical_Markov/002_claude_hmk01_transition_matrices.md) (supersedes the withdrawn HMK 001).

| Notebook | Content | Output |
|---|---|---|
| `01_transition_matrices` | 20-bar sum, ±5% (intraday also √time-scaled ±1.04% / ±0.52%); counts, P, persistence vs shuffled returns and non-overlapping blocks, P^2/3/5/20, stationary distribution, run lengths, next-bar up share by state; 2020-01 – 2023-06 and halves | `01/` |
| `02_longer_windows` | 1 h on 40 / 60 bars, 15 m on 60 / 80 bars (daily 20 / ±5% reference); (a) set √time-scaled bands, (b) 3-state HMM on the window return (state means = expected bull / bear returns); same matrices and references as 01 ([`004`](../../discussions/MNQ_Hierarchical_Markov/004_claude_hmk02_longer_windows.md)) | `02/` |

**Data.** MNQ `ohlcv-1m` already on disk; nothing from 2023-07 onwards or 2026 is read.

**Run.** `python -m pytest -q tests/test_hmk.py`, then run `01` (a few minutes). Smoke run: `HMK_SYNTHETIC=1` (random walk; shows the rule's built-in persistence).

**Code.** `src/equity_trading/hmk.py`. The hierarchical HMM engine from HMK 001 is kept there.

**Record.** `withdrawn_hmk001/`: Iván's run `85560a6` of the superseded HMK 001 notebook and its outputs, unchanged (summary in HMK 002 §7). It will not rerun as is: it expects to sit one folder higher.

**Result HMK 01 (run `6bb2ce5`, readout [`003`](../../discussions/MNQ_Hierarchical_Markov/003_claude_hmk01_readout.md)).** The matrices' persistence equals what shuffled returns give under the same 20-bar rule; only sideways (calm) persists beyond it; next-bar up share is the base rate in every state, on every tier.

**Result HMK 02 (run `c50a7b9`, readout [`005`](../../discussions/MNQ_Hierarchical_Markov/005_claude_hmk02_readout.md)).** Longer windows and the HMM raise persistence to 0.90–0.98, all of it matched by shuffled returns for bull/bear; next-bar up share stays within ~1 pp of the base rate in every state.
