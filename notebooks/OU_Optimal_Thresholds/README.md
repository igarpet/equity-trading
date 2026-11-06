# OU Optimal Thresholds — paper P8

**Paper.** Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading. *Physica A* 389(11), 2234–2243. Lipton, A. & López de Prado, M. (2020). A closed-form solution for optimal Ornstein–Uhlenbeck driven trading strategies.

**Claim to test.** Closed-form optimal entry/exit thresholds for an OU process with costs (Bertram) and with a finite horizon and stop-loss (Lipton & López de Prado).

**Our question.** Do model-optimal thresholds beat the S05c/S05d rules out of sample? Expected to change timing, not edge size (low priority).

| Notebook | Description | Status |
|---|---|---|
| `01_ou_optimal_thresholds.ipynb` | Bertram per-stock-day band m*(c̃) and Lipton–López de Prado take-profit / stop-loss / 3-half-life limit (simulated first-passage table) on the S05 residual with the earnings filter, measured costs, both windows; vs the incumbent S05 `x10` + earnings filter | Run d63819a: `NO_IMPROVEMENT`, `NO_EDGE` for both; BT ≈ incumbent with tighter exits (m* ≈ 0.43); LP fast exits give the highest gross (Sharpe 1.45 / 2.09) but 70–75 trades/day at 4–5 bps vs 10–13 bps cost |

Readout: [`002_claude_p8_readout.md`](../../discussions/OU_Optimal_Thresholds/002_claude_p8_readout.md). Protocol: [`001_claude_p8_ou_optimal_thresholds_preregistration.md`](../../discussions/OU_Optimal_Thresholds/001_claude_p8_ou_optimal_thresholds_preregistration.md). Both windows judged; earnings filter built in.

Run: `python -m pytest -q tests/test_p8.py`, then notebook 01 (~1 h, no downloads; restart the kernel after pulling). `P8_SYNTHETIC=1` runs it offline. Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
