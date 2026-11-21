# RSW 001 — Regime-switching AR(1) on MNQ 5-minute returns (RSW 01, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-21 |
| **Branch** | MNQ_Regime_Switching |
| **Responds to** | Iván, 2026-11-21: model mean reversion vs continuation (not direction) with a regime-switching OU; tick-level order flow is too expensive (Databento `tbbo` 2020–2025 ≈ $1,965, `trades` ≈ $1,179, `ohlcv-1s` ≈ $336), so use the 1-minute bars already owned |
| **Engine** | `regime_ar.py` 0.1.0, `mnq_predict.py` 0.1.1 (grid, volatility model), `reversal.py` 0.2.0 (`event_sigma`, `day_cluster_t`) |
| **Status** | Discovery only, written before any fit. The test window (2023-07 – 2025-12) was used by MRP 03, so this runs on the training window 2020-01 – 2023-06. That window has been seen in MRP 01–04, so a pass is a candidate, confirmed only on the sealed 2026 holdout. |

## 1. Question

Do MNQ's standardized 5-minute returns have hidden regimes with different autocorrelation, i.e. a reversal regime (φ < 0) and a continuation regime (φ > 0)?

If they do, there are two follow-up questions:
1. Does the causal (filtered) regime probability forecast the next bar better than a one-regime AR(1)?
2. Is that forecast worth a round trip?

**Prior.** MRP 01 found no forecasting value in 12 own lags. Its only stable coefficient was a weak lag-2 reversal, and MRP 02–04 found no tradable reversal or continuation. A linear AR averages over regimes, so φ_s of opposite sign could cancel. That is the only reason this test is not redundant. The prior for an economic pass is low.

## 2. Model

This is a regime-switching OU, discretized on returns rather than on the price level. An OU on the non-stationary price level would give θ ≈ 0, or a spurious θ from finite-sample bias.

Inside regime s (hidden, first-order Markov, transition matrix A):

z_t = c_s + φ_s z_{t−1} + σ_s ε_t

- z_t = r_t / σ̂_t, from the MRP 001 time-of-day × HAR volatility model.
- 2020 uses the in-sample fit. Later dates use the walk-forward fits made before 2021-01-01, 2022-01-01 and 2023-01-01. This is the same as MRP 04.
- **Local rescaling**: z is then divided by L_t, the rms of r / σ̂ over the eligible bars among the previous 276 bars (one CME day, ex ante). MRP 01 gate V2 failed on the volatility level, and the synthetic smoke run showed that a 2-state model then splits on the level error (σ_s) instead of on φ_s. L_t removes the slow level error, so a state split has to come from short-run variance or autocorrelation. σ̂_t L_t is the volatility used to convert forecasts into ticks.
- The AR term is switched off when bar t−1 is in another CME day or has no return.

**Order flow in the transition probabilities** (models M3, M4; TVTP, Diebold, Lee & Weinbach 1994):

A_t[i, j] = softmax_j(W_ij · [1, x_{t−1}]), with staying as the reference. x_{t−1} is the previous bar's flow, standardized on the fit window:
- **vsurp**: log(1 + volume) minus its mean in the same slot over the previous 20 CME days.
- **|imb|** and **imb × sign(z)**: imb is bulk volume classification (Easley, López de Prado & O'Hara 2012) on the 1-minute bars inside the bar. The buy share of each minute is Φ(r_m / σ_m), where σ_m is the sd of the previous 60 one-minute returns. imb = Σ V(2Φ − 1) / Σ V ∈ [−1, 1]. imb × sign(z) measures whether flow agreed with the price move.

Flow can move the chain between regimes but is not modelled as an emission. In the smoke run, joint Gaussian emissions for imb and vsurp made the states split on the shape of the flow distribution and lose a planted φ split. That design was rejected before any real data was read.

| Model | States | Order flow in transitions |
|---|---|---|
| M0 (benchmark) | 1 (plain AR(1)) | no |
| M1 | 2 | no |
| M2 | 3 | no |
| M3 | 2 | yes |
| M4 | 3 | yes |

**Estimation.** EM (Baum-Welch with an AR(1) emission per state). M0–M2 use 5 random starts. M3/M4 start from the best M1/M2 fit plus one random start. The best likelihood is kept, and states are ordered by φ.

**Common observation mask** for every model: eligible bars as in MRP 001 (slot ≥ 3, CME day ≥ 220 traded bars, finite z) with vsurp and imb defined. Other bars only propagate the chain. M-step for W: weighted multinomial logit with a N(0, 1) prior on the standardized slopes.

**Walk-forward.**
- Expanding fits on 2020-01-01 → before 2021-01-01, 2022-01-01 and 2023-01-01.
- Each fit is frozen and filters its year causally.
- Out-of-sample window: 2021-01-01 – 2023-06-30.
- Periods: 2021, 2022, 2023H1.

## 3. Gates (per model M1–M4)

- **R1 (regimes improve the density forecast)**: the out-of-sample daily mean log predictive score of z_t (z equation only) beats M0. One-sided Newey-West p (5 lags), Holm-adjusted across M1–M4, ≤ 0.05.
- **R2 (regimes differ in autocorrelation)**: in every one of the three fits, φ_max − φ_min ≥ 0.05 and t_gap ≥ 3. The weighted-LS standard errors ignore regime uncertainty and are therefore too small, hence the threshold of 3 rather than 2.
- **R3 (persistent regimes)**: in every fit, the min-φ and max-φ states have p_stay ≥ 2/3, i.e. an expected duration of at least 3 bars. A regime that changes every bar cannot be acted on.
- **R4 (the mean forecast has value)**: the one-step forecast E[z_t | F_{t−1}] = Σ_s P(s_t | F_{t−1})(c_s + φ_s z_{t−1}) must satisfy all of:
  - R²_OOS > 0 against zero;
  - R²_OOS > 0 against M0's forecast;
  - Clark–West against M0, one-sided p Holm-adjusted across M1–M4, ≤ 0.05.
- **E (worth a round trip)**:
  - Trade bar t in the sign of the forecast when |E[z_t]| × σ̂_t × price ≥ 3 ticks (the cost).
  - Entry at the close of t−1, exit at the close of t, one contract; never across a CME day.
  - Net = gross − 3 ticks (1 tick spread + 2 × $0.50 commission).
  - Pass requires: at least 150 trades; mean net > 0 with one-sided p (t clustered by CME day) Holm-adjusted across M1–M4 ≤ 0.05; mean net > 0 in at least 2 of the 3 periods.

**Verdicts.**

| Gates passed | Verdict |
|---|---|
| R1–R4 and E | **Candidate**: goes to a 2026-holdout pre-registration (one model, fixed parameters re-fitted on 2020 – 2025-12 only if specified there) |
| R1–R3 only | **Regimes exist but carry no tradable direction at one bar.** The filtered probability may be tested later as a filter for another strategy, only under a new pre-registration |
| R1 fails | No regime structure beyond what the volatility model already removes |

## 4. Also reported (descriptive, no gates)

- Parameters, standard errors, expected durations and BIC of every fit.
- Each regime's share by session (Asia / London / NY / post) and by year. This checks that regimes are not just the clock.
- Each regime's share by σ̂ quintile. This checks that regimes are not just volatility-model error: gate V2 failed in MRP 01 because the z variance was too high in 2021.
- Correlations of imb and vsurp with z and |z|.
- Out-of-sample hit rate, and trades at 1.6 ticks (Iván's realized round trip) and at 0 (gross).
- A filtered-probability plot for one sample week, and filtered vs smoothed agreement.

## 5. Readings fixed in advance

- In M3/M4, p_stay in R3 is the mean of A_t over the fit window.
- BVC on 1-minute bars is a re-expression of price moves weighted by volume, not true aggressor flow. If M3/M4 add nothing, that says nothing about real signed flow. If they add something, buying `trades` or `ohlcv-1s` for one year becomes a justified next step.
- A regime split mainly in σ_s with no φ gap (R1 passes, R2 fails) is volatility-model residue, not mean reversion vs continuation. The null smoke run (no planted regimes) passes R1 for all four models and fails R2, R4 and E. R1 on its own therefore means nothing.
- Adding states, emissions or lags after seeing these results would be fitting. Any variant needs its own pre-registration.

## 6. Run

1. `python -m pytest -q tests/test_regime_ar.py`
2. `notebooks/MNQ_Regime_Switching/01_msar_discovery.ipynb`

Smoke runs offline:
- `RSW_SYNTHETIC=1` It plants a two-state MS-AR(1) with φ = ±0.3 and baseline p_stay = 0.98, and the previous bar's volume shock drives switching (+1.5 on the leave logit per sd). Expected and obtained: all four models are candidates, with leave-logit on vsurp ≈ 1.4.
- `RSW_SYNTHETIC=1 RSW_SYNTH_NULL=1` (use a separate `RSW_SYNTH_DIR`): no regimes and no flow effect. Obtained: R1 passes, R2, R4 and E fail, and no candidate.
