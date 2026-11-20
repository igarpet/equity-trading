# MRP 001 — Next-bar MNQ return from own history and FX on 5-minute bars (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-21 |
| **Branch** | MNQ_Return_Prediction (new) |
| **Responds to** | Iván, 2026-11-19 → 2026-11-21: a t+1 return model for MNQ 5-minute bars (direction probability and magnitude), regressors from MNQ's own history and the 7 USD FX majors; specification agreed in chat on 2026-11-21 |
| **Engine** | `mnq_predict.py` 0.1.0 with `micro_bars.py` 0.1.0 (Databento ohlcv-1m) and the Dukascopy 1-minute FX candles |
| **Status** | Written after the MNQ download, before any MNQ return or regression has been looked at. Gates are not changed after results exist. |

## 1. Question

Given everything known at the close of 5-minute bar t, can we forecast the MNQ log return of bar t+1 better than zero and better than MNQ's own recent returns alone? The model gives a mean μ̂ and a volatility σ̂, so both P(up) and the size of the move come from it.

GBM is not assumed: under GBM the return is unpredictable by construction. The model is a conditional mean (linear regression) plus a conditional variance (time-of-day × HAR) with Student-t errors.

## 2. Data

- **MNQ**: Databento GLBX.MDP3 `ohlcv-1m`, continuous `MNQ.v.0` (volume roll), resampled to 5-minute bars on the UTC clock, labelled by bar open time. Raw prices; a 5-minute bar never mixes two contracts, and the first bar after a contract switch has no return.
- **FX**: Dukascopy 1-minute candles, mid = (bid + ask) / 2, for EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD. The FX return of bar t is the mid log return over exactly the same 5 minutes as MNQ bar t (close of the minute starting at t + 4 min against the close of the minute starting at t − 1 min). No crosses: they are linear combinations of the majors.
- **Windows**: training **2020-01-01 – 2023-06-30**, test **2023-07-01 – 2025-12-31** (looked at once). 2026 is sealed. December 2019 is loaded only as warm-up for lags and the volatility model; no target or fit uses it.

## 3. Bar grid, sessions and exclusions

- **CME day**: 17:00 → 16:00 Chicago time (276 five-minute bars), labelled by the date it ends. Every time window below is defined in local exchange time, so daylight-saving changes in Chicago, London and New York are handled.
- **Grid**: every 5-minute slot of each CME day that has at least one trade. Empty bars inside a session carry the last close (return 0). A run of 3 or more consecutive empty bars (≥ 15 minutes) counts as a gap: those bars and the first bar after them have no return.
- **Returns** are within one CME day and one contract: the first bar of a day, the first bar after a gap and the first bar after a roll have no return.
- **Excluded as targets**:
  - the first 3 bars of every CME day (17:00–17:15 Chicago, Iván);
  - bars without a return (gaps, rolls);
  - CME days with fewer than 220 of the 276 bars (holidays, early closes).
- **No winsorizing** (Iván): extreme bars stay in. A Huber regression is reported as a robustness check (§6).

## 4. Volatility model and standardization

All returns are log returns. Each series (MNQ and each FX pair) is standardized by its own ex-ante volatility forecast: z_t = r_t / σ̂_t, with σ̂_t forecast at the close of bar t − 1.

- **Time of day**: s²(slot) = mean r² per CME slot on the fit window.
- **HAR** on deseasonalized realized variance, where q = (sum of squared 1-minute returns in the bar) / its slot mean. Components known at the close of bar t:
  - Qh: mean q over the last 12 bars;
  - Qd: last 276 bars;
  - Qw: last 5 × 276 bars;
  - Qm: last 22 × 276 bars.
- **Fit**: OLS of (r_{t+1}/s)² on [1, Qh, Qd, Qw, Qm]. Forecast h_{t+1}, floored at 0.1, and σ̂²_{t+1} = s²(slot_{t+1}) · h_{t+1}.
- **Walk-forward**: yearly expanding refits. Fits use data before 2021-01-01, 2022-01-01 and 2023-01-01, and predict up to the next refit. The test window uses refits at 2023-07-01, 2024-01-01 and 2025-01-01.

**Gate V (Iván: standardize only if the volatility model has low error).** Judged on MNQ, walk-forward out of sample 2021-01-01 – 2023-06-30. All three must hold:

- **V1**: QLIKE (log σ̂² + r²/σ̂²) of the full model is below both benchmarks, a constant variance and the time-of-day profile alone. A Diebold–Mariano test on daily mean loss differences (Newey–West, 5 days) against the time-of-day profile gives one-sided p < 0.05.
- **V2**: the variance of z lies in [0.8, 1.25] in every calendar half-year.
- **V3**: the mean autocorrelation of z² over lags 1–12 is below 0.03, and below one third of that of the raw r².

If V passes, the mean model is estimated on standardized z (primary). If V fails, the primary model is OLS on raw log returns (scaled ×10⁴) with the same regressors, using raw returns, and the z version is reported as secondary. FX series are standardized by the same procedure; no gate applies to them because they are regressors.

## 5. Mean model

The target is z_{t+1}. All regressors are known at the close of bar t. Lags that reach before the start of the current CME day, or that have no return, are set to 0.

| Group | Regressors | n | Hypothesis (who) |
|---|---|---|---|
| **A** own lags | z_t, z_{t−1}, …, z_{t−11} | 12 | lag 1 ≤ 0 and small (Claude); further lags ≈ 0 (Iván) |
| **B** large-move continuation | M = S3 · 1{\|S3\| > 2}, with S3 = (z_t + z_{t−1} + z_{t−2}) / √3 | 1 | **> 0** (Iván: big pushes continue) |
| **C** completed sessions | standardized return of the last completed Asia, London and NY session | 3 | exploratory, two-sided |
| **D** previous CME day | standardized return of the previous CME day | 1 | **≠ 0, strongly** (Iván); no sign given |
| **E** same slot yesterday | z of the same CME slot on the previous CME day | 1 | exploratory (HKS periodicity, P3), two-sided |
| **F** FX | z of the 7 USD majors at lags 1, 2, 3 (bars t, t−1, t−2) | 21 | **= 0** (Iván) |
| **G** session interactions | 1{t+1 in NY} × z_t, and 1{t+1 in NY} × each FX lag 1 | 8 | exploratory, two-sided |
| — | intercept, dummies for London, NY and post-close sessions (Asia baseline) | 4 | not tested |

**Sessions** (bar open time, local clocks, non-overlapping):
- **Asia**: CME open (17:00 Chicago) → 08:00 London.
- **London**: 08:00 London → 09:30 New York.
- **NY**: 09:30 → 16:00 New York.
- **Post-close**: 16:00 New York → 16:00 Chicago. The post-close hour has no session sum, but it is part of the previous-CME-day return.

A session's standardized return is Σ r / √(Σ σ̂²) over its bars. "Last completed" means its end is at or before the close of bar t, so a session in progress never enters C.

The previous CME day and the three sessions are only partly collinear (during Asia, the last three completed sessions are all from the previous day). Variance inflation factors are reported per regressor.

## 6. Inference (training window, in sample)

- One regression on 2020-01-01 – 2023-06-30. The z come from the volatility model fitted on the whole training window (in-sample weights; the forecasts in §7 are fully out of sample).
- Standard errors clustered by CME day (about 870 clusters). Newey–West (12 bars) reported alongside.
- **One Wald (F) test per group A–G**, with Holm correction across the 7 groups at α = 0.05. Individual t-statistics are descriptive.
- Directional hypotheses (B > 0; F = 0) are read from the sign and the group test. D is two-sided.
- **Robustness**:
  - Huber regression (statsmodels RLM, HuberT) with the same regressors;
  - raw-return OLS with HAC errors;
  - each half of the training window (2020-01 – 2021-09, 2021-10 – 2023-06).

  A group's verdict counts as robust only if its coefficients with |t| ≥ 2 in the main fit (the largest |t| if none reaches 2) keep their sign under Huber and in both halves.

## 7. Forecasting (out of sample)

Nested models, all re-estimated at each refit date (§4), with the same refit dates as the volatility model:

| Model | Regressors |
|---|---|
| M0 | zero forecast (μ̂ = 0) |
| M1 | intercept + session dummies + A |
| M2 | M1 + B + C + D + E |
| M3 | M2 + F + G (full) |

- **R²_OOS** = 1 − Σ(z − ẑ)² / Σ(z − ẑ_bench)², against M0, and incremental: M2 vs M1 (own history) and M3 vs M2 (FX). Clark–West test for each nested pair, with errors clustered by day. Reported per year.
- **P(up)** = T_ν(μ̂ / scale), where ν is the Student-t degrees of freedom of the fit window's residuals and scale = s · √((ν − 2)/ν), with s their standard deviation (about 1 for z). Brier score against 0.5 and against the fit window's up-share, with calibration in deciles. Bars with a zero return are left out of the Brier score.
- **Magnitude**: the predictive distribution's |r| quantiles against realized values (coverage of the 50% and 90% intervals).

## 8. Gates

Statistical, economic and holdout verdicts are reported separately.

- **S (statistical edge), training walk-forward 2021-01 – 2023-06.** The best of M1–M3 has R²_OOS > 0 against M0 with Clark–West p < 0.05, and R²_OOS > 0 in at least 2 of the 3 periods (2021, 2022, 2023H1). For FX specifically: M3 vs M2 with CW p < 0.05.
- **T (test), 2023-07-01 – 2025-12-31, one look.** The model passing S keeps R²_OOS > 0 against M0 with CW p < 0.05 and Brier below 0.5's.
- **E (economic), only if S passes.** On the test window, trade one MNQ contract in the sign of μ̂ when the expected move |μ̂ · σ̂| in points exceeds the round-trip cost. The cost is 1 tick ($0.50) plus Lucid commissions per side (Iván confirms the rate before notebook 03 runs; the default parameter is $0.50 per side). Hold one bar. Pass: net P&L > 0 with a day-block bootstrap 95% lower bound > 0.
- A model passing S, T and E goes to a 2026-holdout pre-registration. Nothing else does.

## 9. Readings fixed in advance

- If V fails, the conclusions rest on the raw-return model; a "significant" z-model group then needs to hold there too.
- A significant A at lag 1 with a negative sign of about one tick of bounce (|β| ≲ 0.02 in z) is microstructure, not an edge, unless E passes.
- D is constant within a CME day: its evidence is about 870 daily observations in training, which the day clustering accounts for.
- A group that is significant in sample but fails S has no forecasting value.

## 10. Run order

1. `python -m pytest -q tests/test_micro_bars.py tests/test_mnq_predict.py`
2. `notebooks/MNQ_Return_Prediction/00_data.ipynb`: coverage, liquidity 2020–2023H1, gaps, exclusions, FX/MNQ alignment (training window only).
3. `01_volatility.ipynb`: gate V. Commit `01/`.
4. `02_mean_model_train.ipynb`: §6 inference and §7 walk-forward, gate S. Commit `02/`.
5. `03_test.ipynb`: gate T and E. One run, after 02 is committed; it refuses to run without `02/summary.json`.

`MRP_SYNTHETIC=1` runs the notebooks offline on synthetic data. The synthetic MNQ has GARCH-like volatility, a time-of-day profile and a planted lag-1 FX dependence on USDJPY; the other pairs are noise.
