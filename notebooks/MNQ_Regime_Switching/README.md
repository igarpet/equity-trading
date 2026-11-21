# MNQ regime switching — reversal vs continuation regimes in 5-minute returns (RSW)

**Question.** Do MNQ's standardized 5-minute returns switch between a reversal regime (φ < 0) and a continuation regime (φ > 0)? If so, does the causal regime probability forecast the next bar well enough to pay a round trip?

The model is a regime-switching OU discretized on returns: a Markov-switching AR(1) on z. Optionally, the previous bar's order flow (BVC imbalance and volume surprise from the 1-minute bars) drives the transition probabilities. Micro contracts only. Ticks per contract.

Protocol: [`001_claude_rsw01_preregistration.md`](../../discussions/MNQ_Regime_Switching/001_claude_rsw01_preregistration.md).

Discovery runs on the training window 2020-01 – 2023-06, with walk-forward out-of-sample evaluation 2021-01 – 2023-06. The 2023-07 – 2025-12 window was used by MRP 03. A candidate is confirmed only on the sealed 2026 holdout.

| Notebook | Content | Output |
|---|---|---|
| `01_msar_discovery` | M0 (AR(1)), M1/M2 (2/3 states), M3/M4 (2/3 states, flow-driven transitions); expanding fits before 2021, 2022 and 2023; gates R1–R4 and E; regimes by session, year and σ̂ quintile; sample-week plot | `01/` |

**Data.** Only the MNQ `ohlcv-1m` already downloaded for MRP (`tools/databento/micro_bars_download.py`); nothing to buy.

**Code.**
- `src/equity_trading/regime_ar.py`: EM with numba kernels, TVTP M-step, causal filter, gates, synthetic data.
- Reuses the MRP grid and volatility engine: `mnq_predict.py`, and `reversal.event_sigma` / `day_cluster_t`. Only the code is reused; no MRP results, thresholds or fitted models.
- Tests: `tests/test_regime_ar.py`.

**Run order.**
1. `pip install -r requirements.txt` (adds `numba`).
2. `python -m pytest -q tests/test_regime_ar.py`
3. Run `01`. Fits are cached in `01/fits_<version>.pkl`; delete that file to refit. The flow models (M3/M4) are the slow part: expect roughly 30–60 minutes on the research machine.

**Smoke runs.**
- `RSW_SYNTHETIC=1`: planted φ = ±0.3 with flow-driven switching. All models are candidates, and the vsurp leave-logit is recovered.
- `RSW_SYNTHETIC=1 RSW_SYNTH_NULL=1` (separate `RSW_SYNTH_DIR`): no regimes. R1 passes but R2, R4 and E fail, so R1 alone is not evidence.
