# RSW 003 — Hawkes vs Markov for MNQ 5-minute extremes: decay and next-bar forecast (RSW 02, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Responds to** | Iván, 2026-11-22: after RSW 01 ([`002`](./002_claude_rsw01_readout.md)), the burst regime looks like self-exciting jumps (Hawkes), modelled with extreme value theory. He chose to test (1) Markov vs Hawkes decay and (2) a self-exciting peaks-over-threshold forecast. Not tested: what the forecast would be used for (risk filter or direction) |
| **Engine** | `hawkes.py` 0.1.0, `regime_ar.py` 0.1.0, `mnq_predict.py` 0.1.1 |
| **Status** | Statistical study, no economic gate. Training window 2020-01 – 2023-06 only (seen before, in RSW 01 and MRP). A use of the forecast needs its own pre-registration, and a candidate needs the 2026 holdout. |

## 1. Question

Is the clustering of large MNQ 5-minute moves Markov or Hawkes?
- **Markov**: the chance of a new burst returns to its baseline geometrically, at the speed of a hidden two-state chain.
- **Hawkes**: every burst raises the chance of later bursts, and that excitation fades slowly.

If it is Hawkes, does a self-exciting model forecast the next exceedance better than the Markov models?

Two points are fixed in advance:
- The first burst of each Hawkes cluster comes from the baseline rate and is unpredictable by construction. The branching ratio n is the largest share of bursts a Hawkes model could forecast.
- z is already standardized by the MRP 001 HAR model, which uses the last hour's realized variance, and rescaled by the previous day's rms. Any clustering found here is beyond what the volatility model already captures.

## 2. Data and events

- RSW 01 series: same grid, same volatility fits (2020 in sample, walk-forward later) and the same z.
- Same observation mask: observed bars only.
- **Event**: e_t = 1 when bar t is observed and |z_t| ≥ u.
- **Threshold**: u = the 0.96 quantile of |z| on the fit window, close to the RSW 01 burst share. Sensitivity: 0.98 and 0.99.
- Time runs over grid rows (276 per CME day). Unobserved rows carry no event and no likelihood term.

## 3. Test 1 — decay (gate H1)

**Curve.** On 2020-01 – 2023-06, at u(0.96):

C(k) = P(e_{t+k} = 1 | e_t = 1), for k = 1…120.

**Nulls.**
- (a) Binary first-order chain on e: C(k) = p + (1 − p) λ^k, with λ = p11 − p01.
- (b) RSW M1 (2 states) and (c) RSW M2 (3 states), refitted on the same window and simulated (2,000,000 bars each). These are hidden Markov models, so their curves also decay geometrically, at their latent eigenvalues. M2's extra state can absorb part of the slow volatility persistence.

**Statistic.** X = Σ_{k=6}^{60} [C(k) − C_null(k)], the excess clustering from half an hour to five hours after an event.

**H1 passes when** the 95% lower bound of X is > 0 against all three nulls:
- moving-block bootstrap by CME day (5-day blocks, 2,000 draws);
- the chain is refitted in every draw; the M1 and M2 curves are held fixed.

**H1 is necessary, not sufficient.** Self-excitation and volatility persistence left in z by the HAR model both give slowly decaying clustering. The Markov smoke run shows this: Markov regimes with a daily log-volatility AR(1) pass H1 against all three nulls (excess over M2 = 0.24, lower bound 0.07). The decay alone does not identify the mechanism, so H2 is the deciding gate.

Also reported:
- C(k) with both nulls and the base rate (plot);
- u(0.98) and u(0.99);
- z without the day rescale (z0 = r / σ̂): the rescale uses the previous 276 bars and could hide excitation that lasts longer than a day.

## 4. Test 2 — next-bar forecast of an exceedance (gates H2, H3)

Each model gives P(e_t = 1 | F_{t−1}).

| Model | Definition |
|---|---|
| B0 | constant rate |
| MC | binary first-order chain |
| M1 | RSW 01 2-state MSAR: Σ_s P(s_t \| F_{t−1}) P(\|N(c_s + φ_s z_{t−1}, σ_s²)\| ≥ u) |
| M2 | the same, 3 states |
| **HX** | discrete Hawkes, λ_t = μ + n Σ_{k≥1} (1 − β) β^{k−1} e_{t−k}, P = 1 − e^{−λ_t} |
| **HXM** | HX with marks m = exp(δ(\|z\| − u)), normalized to mean 1 on the fit exceedances (larger extremes excite more) |
| **HP** | power-law kernel (k + c)^{−(1+γ)}, truncated at 2,760 bars (10 days) |

**Fitting.**
- Maximum likelihood (Bernoulli), 4 starts.
- Expanding fits before 2021-01-01, 2022-01-01 and 2023-01-01, as in RSW 01.
- u is re-estimated in each fit.
- Each fit is frozen for its segment.
- Out of sample: 2021-01-01 – 2023-06-30.

**Gates.**
- **H2**: a Hawkes model beats MC, M1 and M2 on the out-of-sample Bernoulli log score:
  - daily mean differences, Newey-West t (5 lags), one-sided;
  - Holm-adjusted across the 9 tests (3 Hawkes models × 3 benchmarks), ≤ 0.05 against all three.
  - Passing H2 means past exceedances forecast the next one better than the regime filters do. That is the operational claim. It still does not prove self-excitation over an unmodelled persistent level.
  - If several models pass, the winner is the one with the highest score.
- **H3**: the winner's branching ratio n is in (0.05, 0.95) in every fit, with a range of at most 0.15 across fits.

**Sizes** (the extreme value theory part, reported, no gate): (|z| − u) ~ GPD(ξ, σ_t), with σ_t = σ0 (λ_t / μ)^κ. Likelihood-ratio test of κ = 0 (sizes do not grow with excitation), per fit, using the winning Hawkes model, or HX if none wins.

**Also reported:**
- parameters per fit: μ, n, β or (c, γ), δ, half-life or median lag of the kernel;
- out-of-sample calibration in deciles;
- lift: event rate in the top 5% of predicted probability vs the base rate, and the share of events caught there;
- H2 scores at u(0.98) and u(0.99);
- sign persistence of consecutive exceedances within 12 bars (descriptive only; direction is not tested).

## 5. Verdicts

| Gates | Verdict |
|---|---|
| H1, H2, H3 | **Hawkes**: burst timing is forecastable beyond the Markov regimes. Next is a use pre-registration (risk filter or direction), then the 2026 holdout |
| H1 only | Slow clustering (Hawkes, or volatility persistence left in z: not identified) with no forecast gain over the Markov regimes |
| H2 without H1 | The Hawkes form helps short-run forecasting; the long-memory claim is not supported |
| neither | Markov regimes (RSW 01) are enough; Hawkes adds nothing |

## 6. Readings fixed in advance

- A Hawkes pass says when a large bar is more likely. It says nothing about its sign. RSW 01 and MRP 02–03 found no tradable direction.
- A small n (for example 0.2) caps the forecastable share of bursts even if H2 passes.
- Adding kernels, covariates or thresholds after seeing the results would be fitting.

## 7. Run

1. `python -m pytest -q tests/test_hawkes.py tests/test_regime_ar.py`
2. `notebooks/MNQ_Regime_Switching/02_hawkes_pot.ipynb`

Smoke runs:
- `RSW_SYNTHETIC=1`: the RSW 01 synthetic data (Markov regimes plus daily log-volatility AR(1)). Obtained: H1 passes (the false positive above); H2 fails, with M1 and M2 beating every Hawkes model. Verdict: slow clustering, no forecast gain.
- `RSW_SYNTHETIC=1 RSW_SYNTH_HAWKES=1` (separate `RSW_SYNTH_DIR`): Hawkes-timed bursts (μ = 0.02, n = 0.5, β = 0.97).
  - H1 passes.
  - Hawkes beats MC and M1 (Holm p < 0.001) and beats M2 at t ≈ 1.9 (Holm p 0.07). With only a 6-month out-of-sample window, H2 is not reached at u(0.96); it passes at u(0.99). The real out-of-sample window is 2.5 years, about 5 times as long.
  - n is estimated at 0.23–0.31 against 0.5 planted: the HAR model (last hour's realized variance) and the day rescale absorb part of the excitation. Expect n to be understated on real data for the same reason.
