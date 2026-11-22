# MNQ regime switching — reversal vs continuation regimes in 5-minute returns (RSW)

**Question.** Do MNQ's standardized 5-minute returns switch between a reversal regime (φ < 0) and a continuation regime (φ > 0)? If so, does the causal regime probability forecast the next bar well enough to pay a round trip?

The model is a regime-switching OU discretized on returns: a Markov-switching AR(1) on z. Optionally, the previous bar's order flow (BVC imbalance and volume surprise from the 1-minute bars) drives the transition probabilities. Micro contracts only. Ticks per contract.

Protocols: RSW 01 [`001`](../../discussions/MNQ_Regime_Switching/001_claude_rsw01_preregistration.md) (readout [`002`](../../discussions/MNQ_Regime_Switching/002_claude_rsw01_readout.md)); RSW 02 [`003`](../../discussions/MNQ_Regime_Switching/003_claude_rsw02_hawkes_pot_preregistration.md).

Discovery runs on the training window 2020-01 – 2023-06, with walk-forward out-of-sample evaluation 2021-01 – 2023-06. The 2023-07 – 2025-12 window was used by MRP 03. A candidate is confirmed only on the sealed 2026 holdout.

| Notebook | Content | Output |
|---|---|---|
| `01_msar_discovery` | M0 (AR(1)), M1/M2 (2/3 states), M3/M4 (2/3 states, flow-driven transitions); expanding fits before 2021, 2022 and 2023; gates R1–R4 and E; regimes by session, year and σ̂ quintile; sample-week plot | `01/` |
| `02_hawkes_pot` | RSW 02: Hawkes vs Markov for \|z\| exceedances (u = 0.96 quantile): decay of clustering against the binary chain and simulated M1/M2 (H1); walk-forward next-bar exceedance forecast, Hawkes (geometric, marked, power-law kernels) vs MC/M1/M2 (H2); branching-ratio stability (H3); GPD sizes vs intensity | `02/` |

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

## Result RSW 01 (run `0d7b2af`): regimes exist, no tradable one-bar direction

Readout: [`002`](../../discussions/MNQ_Regime_Switching/002_claude_rsw01_readout.md).
- **Calm regime**: about 95% of bars, φ ≈ −0.05, lasts 12–20 bars.
- **Burst regime**: about 4% of bars, z sd ≈ 2–3, φ ≈ +0.12 to +0.29, c ≈ −0.2, lasts 1.4–1.9 bars (3–4 with flow).
- R1 and R2 pass for all models; R3 passes only for M3; R4 and E fail for all.
- Trading every bar: gross +0.33 ticks per bar, about a tenth of the cost.
- The notebook's verdict labels for M1, M2 and M4 are wrong: they fail R3, not R2.

**RSW 02** (Hawkes vs Markov, [`003`](../../discussions/MNQ_Regime_Switching/003_claude_rsw02_hawkes_pot_preregistration.md)): code ready, not run.
- Run `python -m pytest -q tests/test_hawkes.py`, then `02`.
- Fits are cached in `02/fits_<version>.pkl`.
- Expect about 20–40 minutes: each threshold refits M1/M2 three times.
