# MNQ regime switching — reversal vs continuation regimes in 5-minute returns (RSW)

**Question.** Do MNQ's standardized 5-minute returns switch between a reversal regime (φ < 0) and a continuation regime (φ > 0)? If so, does the causal regime probability forecast the next bar well enough to pay a round trip?

The model is a regime-switching OU discretized on returns: a Markov-switching AR(1) on z. Optionally, the previous bar's order flow (BVC imbalance and volume surprise from the 1-minute bars) drives the transition probabilities. Micro contracts only. Ticks per contract.

Protocols: RSW 01 [`001`](../../discussions/MNQ_Regime_Switching/001_claude_rsw01_preregistration.md) (readout [`002`](../../discussions/MNQ_Regime_Switching/002_claude_rsw01_readout.md)); RSW 02 [`003`](../../discussions/MNQ_Regime_Switching/003_claude_rsw02_hawkes_pot_preregistration.md) (readout [`004`](../../discussions/MNQ_Regime_Switching/004_claude_rsw02_readout.md)); RSW 03 [`005`](../../discussions/MNQ_Regime_Switching/005_claude_rsw03_duration_preregistration.md) (readout [`006`](../../discussions/MNQ_Regime_Switching/006_claude_rsw03_readout.md)); RSW 04 [`007`](../../discussions/MNQ_Regime_Switching/007_claude_rsw04_burst_reversion_preregistration.md) (readout [`008`](../../discussions/MNQ_Regime_Switching/008_claude_rsw04_readout.md)); RSW 05 [`009`](../../discussions/MNQ_Regime_Switching/009_claude_rsw05_burst_entry_preregistration.md).

Discovery runs on the training window 2020-01 – 2023-06, with walk-forward out-of-sample evaluation 2021-01 – 2023-06. The 2023-07 – 2025-12 window was used by MRP 03. A candidate is confirmed only on the sealed 2026 holdout.

| Notebook | Content | Output |
|---|---|---|
| `01_msar_discovery` | M0 (AR(1)), M1/M2 (2/3 states), M3/M4 (2/3 states, flow-driven transitions); expanding fits before 2021, 2022 and 2023; gates R1–R4 and E; regimes by session, year and σ̂ quintile; sample-week plot | `01/` |
| `02_hawkes_pot` | RSW 02: Hawkes vs Markov for \|z\| exceedances (u = 0.96 quantile): decay of clustering against the binary chain and simulated M1/M2 (H1); walk-forward next-bar exceedance forecast, Hawkes (geometric, marked, power-law kernels) vs MC/M1/M2 (H2); branching-ratio stability (H3); GPD sizes vs intensity | `02/` |
| `03_duration` | RSW 03: does regime age matter? (A) model-free exceedance hazard by bars since the last one vs chain/M1/M2; (B) HSMM (Weibull / negative-binomial durations) vs HMM, out-of-sample density and exceedance forecast; (C) age-drift HSMM, direction R4 and one-bar trade E | `03/` |
| `04_burst_reversion` | RSW 04: enter against the burst at the switch back to calm (M1 HMM, WB2 HSMM), time stop at mean calm duration + 5 bars, 5 stop losses; 1-minute fills, random-direction placebo, mirror | `04/` |
| `04b_burst_reversion_full` | RSW 04 frozen rules (engine 0.1.1) extended to 2021-01 – 2025-12, yearly refits; gates on the original window, on 2023-07 – 2025-12 and on all (a check, not a new discovery) | `04b/` |
| `05_burst_entry` | RSW 05: volatility and regime models (M1, WB2) refitted on 1-minute bars; enter at the first burst minute, with or against it; time stop mean burst duration + 5 minutes; 5 stops; 20 strategies; monthly data coverage | `05/` |

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

## Result RSW 02 (run `b808aef`): Markov regimes suffice

Readout: [`004`](../../discussions/MNQ_Regime_Switching/004_claude_rsw02_readout.md).
- H1 passes at u(0.96) only.
- H2 fails: Hawkes beats the chain but loses to M1 and M2 (t −3.7 to −7.9).
- Branching ratio n ≈ 0.11–0.23, half-life about 2 bars.
- Sizes are heavy-tailed (ξ ≈ 0.2–0.28) and grow with intensity (κ ≈ 0.4).
- Clustered extremes keep their sign 59.5% of the time (descriptive).

## Result RSW 03 (run `c3a9f58`): regimes get stickier with age; no timing or direction gain

Readout: [`006`](../../discussions/MNQ_Regime_Switching/006_claude_rsw03_readout.md).
- A: the hazard falls with age.
- B: the HSMM improves the density only (HB2 fails); every duration shape is below 1.
- C: the age drift is real in sample, but R4 and E fail.

## Result RSW 04 (run `2e702f3`): no edge

Readout: [`008`](../../discussions/MNQ_Regime_Switching/008_claude_rsw04_readout.md).
- Gross about 0 for reversion, mirror and random direction alike; net −1 to −3 ticks per trade.
- Engine 0.1.1 fixes trades carried across the daily break; a rerun is pending.
- Trades stop from mid-May to early July 2022: check for missing data.

**RSW 05** (1-minute burst entry, [`009`](../../discussions/MNQ_Regime_Switching/009_claude_rsw05_burst_entry_preregistration.md)): code ready, not run.
- Run `python -m pytest -q tests/test_burst_entry.py`, then `05`.
