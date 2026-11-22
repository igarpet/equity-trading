# RSW 005 — Does the time spent in a regime matter? Hazard, HSMM vs HMM, direction by age (RSW 03, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Responds to** | Iván, 2026-11-22: an HMM has no memory by construction, but we do. Counting the bars spent in a regime should give the switch probabilities, and perhaps direction. Test an HSMM against the HMM. He approved all three parts (A, B, C) in one study |
| **Engine** | `hsmm.py` 0.1.0, `hawkes.py` 0.1.0, `regime_ar.py` 0.1.0, `mnq_predict.py` 0.1.1 |
| **Status** | Discovery on the training window 2020-01 – 2023-06, which has been seen before (RSW 01–02, MRP). A candidate in part C goes to a 2026-holdout pre-registration; nothing else does. |

## 1. Question and the logic it tests

In a Markov chain:

P(stay | stayed n bars) = p^{n+1} / p^n = p.

The regime's age carries no information by construction: durations are geometric and the hazard is flat. Counting bars only helps if the true durations are **not** geometric. This study tests that three ways.

- **A.** Directly on the data, with no hidden state.
- **B.** Inside a hidden semi-Markov model (HSMM) with the RSW 01 regimes. The regime is never observed, so the count has to be filtered: the HSMM tracks P(regime, age | data).
- **C.** Direction: whether the regime's drift changes with its age.

## 2. Data

- RSW 01 series: same grid, same volatility model and day rescale, same observation mask.
- Exceedance: |z| ≥ u, with u = the 0.96 quantile of |z| on the fit window, as in RSW 02.

## 3. Part A — model-free hazard (gate HA)

**Hazard.** h(n) = P(exceedance at t | the last exceedance was n rows before t).
- Bins: ages 1–12 individually, then 16 log-spaced bins up to 480.
- Bins with fewer than 200 observed rows are dropped.

**Statistic.** The weighted least-squares slope of log h on log n over ages **13–480**. This skips the first hour, where RSW 02 already showed short clustering.

**Nulls.**
- The binary chain: flat after n = 1, so slope 0.
- M1 and M2 refitted on the whole window, simulated (2,000,000 bars each) and binned the same way.

**HA (rising) passes when** the observed slope minus each null's slope has a 95% lower bound > 0. This uses a day-block bootstrap (5-day blocks, 2,000 draws) with the nulls held fixed.

**Fixed readings.**
- A rising hazard means the longer the market has been calm, the likelier an extreme. It cannot be produced by heterogeneity, because mixing calm and volatile periods only pushes the hazard down.
- A falling hazard is not identified: volatility persistence left in z produces it. The Markov smoke run shows this (slope −0.23 against null slopes ≥ −0.06).

## 4. Part B — HSMM vs HMM (gates HB1–HB3)

**Model.** The same AR(1) regimes as RSW 01, with 2 or 3 states and constant parameters per state. Each state has its own duration distribution:
- **WB**, discrete Weibull: h(d) = 1 − exp(−λ(d^β − (d−1)^β)). β = 1 is geometric, i.e. the HMM.
- **NB**, shifted negative binomial: r = 1 is geometric.

On leaving state s, the chain moves to j ≠ s with probability W[s, j]. It is run as an HMM on (state, age) pairs with ages 1–60, where age 60 is a tail with constant hazard (Langrock & Zucchini 2011).

**Estimation.**
- EM, starting from the HMM fit of the same K with shape 1. At that start the HSMM likelihood equals the HMM's exactly (tested).
- Up to 300 iterations.
- Expanding walk-forward fits before 2021-01-01, 2022-01-01 and 2023-01-01. Each is frozen and filters its own segment causally. Out of sample: 2021-01 – 2023-06.
- Four models: WB2, NB2, WB3, NB3. Each is compared to the HMM with the same K (RSW 01 M1 and M2, refitted).

**Gates.**
- **HB1**: the out-of-sample log predictive density of z beats the HMM's. Daily Newey-West t, one-sided, Holm across 4, ≤ 0.05.
- **HB2**: the out-of-sample next-bar exceedance score (Bernoulli, as in RSW 02 H2) beats the HMM's, under the same test.
- **HB3**: some state's duration shape differs from 1, with |t| ≥ 3 and the same sign, in all three fits. The t comes from the Hessian of the hazard M-step given the expected counts, so it is too small; hence the threshold of 3.

**Verdicts for B.**

| Gates passed | Verdict |
|---|---|
| HB2 and HB3 | `DURATION_DEPENDENCE_FORECASTS_TIMING` |
| HB1 and HB3, without HB2 | `DURATION_DEPENDENCE_IN_DENSITY_ONLY` |
| otherwise | `HMM_SUFFICES` |

**HB3 alone means nothing.** The Markov smoke run passes HB3, because misspecified states absorb volatility persistence. It also fails HB1 and HB2 there.

## 5. Part C — direction by age (gates HC1–HC3)

**Model.** WB HSMM with 2 and 3 states, with an age drift: z_t = c_s + φ_s z_{t−1} + **b_s log d_t** + σ_s ε_t. It starts from the part B WB fit with b = 0. The forecast is

E[z_t | F_{t−1}] = Σ_{s,d} P(s, d | F_{t−1})(c_s + φ_s z_{t−1} + b_s log d).

**Gates.**
- **HC1**: some state's b_s has |t| ≥ 3 with the same sign in all three fits.
- **HC2 (= RSW 01 R4)**: all of
  - R²_OOS > 0 against zero;
  - R²_OOS > 0 against the part B WB model of the same K;
  - Clark–West against that model, Holm across 2, ≤ 0.05.
- **HC3 (= RSW 01 E)**: a one-bar trade when the forecast is ≥ 3 ticks:
  - 3-tick cost;
  - at least 150 trades;
  - net > 0 with Holm across 2 ≤ 0.05;
  - net > 0 in at least 2 of 3 periods.

**Verdict.** HC1, HC2 and HC3 together are a **candidate** and go to a 2026-holdout pre-registration. HC3 without HC2 is not a pass: the edge would come from the AR regimes, not from age. The HSMM smoke run shows exactly this: HC3 passes there, because the planted φ regimes are tradable, but HC2 fails.

## 6. Also reported

- The hazard table and plot with both nulls.
- Every fit's parameters: c, φ, σ, b, shape and its t, mean duration, h(1) and h(10).
- The fitted hazards against the HMM's constant hazard, and the quantiles of the filtered expected age.
- Trades at 1.6 ticks and gross on every bar.

## 7. Smoke runs (obtained before any real data was read)

- **`RSW_SYNTHETIC=1`** (Markov regimes plus daily log-volatility AR(1)):
  - A: falling hazard, not identified;
  - B: `HMM_SUFFICES` (HB1/HB2 Holm p ≥ 0.06; HB3 passes, which is the false alarm above);
  - C: no.
- **`RSW_SYNTHETIC=1 RSW_SYNTH_HSMM=1`** (calm duration Weibull β = 1.8, mean about 17 bars):
  - **A: not detected** (slope +0.02 against null slopes 0.006–0.013, interval includes 0). A has low power for latent duration dependence, because exceedances inside the calm regime dilute it. A failure of A on real data is therefore weak evidence.
  - **B: `DURATION_DEPENDENCE_FORECASTS_TIMING` for all four models** (HB2 t = 2.8–3.4; calm shape 1.64–1.71 against 1.8 planted).
  - C: no (HC2 fails, as it should; no age drift was planted).

## 8. Run

1. `python -m pytest -q tests/test_hsmm.py tests/test_hawkes.py tests/test_regime_ar.py`
2. `notebooks/MNQ_Regime_Switching/03_duration.ipynb`

Fits are cached one by one in `03/fits_<version>.pkl`, so an interrupted run resumes. Expect about 45–90 minutes on the research machine.
