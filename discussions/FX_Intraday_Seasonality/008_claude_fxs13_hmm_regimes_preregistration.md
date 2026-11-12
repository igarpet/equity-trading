# FXS 008 — Hidden Markov regimes: do they tell us when the signals work? (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-12 |
| **Study** | FXS 13 |
| **Responds to** | Iván, 2026-11-12: "time for an HMM on returns, we know there is something but don't know exactly what" |
| **Status** | Written before any HMM result. |

## 1. Model

For each of the six pairs (EURUSD, USDJPY, EURJPY, AUDJPY, CADJPY, GBPJPY), one observation per New York day (17:00 → 17:00):
- the day's mid log return;
- the log of realized volatility, √Σ of the 5-minute squared mid log returns.

Both features are standardized with the training sample. The model is a Gaussian HMM with diagonal covariances, K = 2 and K = 3 states, fitted by Baum–Welch (best of 3 starts). States are ordered by mean volatility (state 0 = calmest).

**Causality.** The state used for day t is the one-step prediction P(s_t | data through the close of t−1). It is computed with the forward filter only; smoothed states use the future and are shown for description only.

## 2. Walk-forward

For each test year Y = 2021 … 2026:
- the HMM is fitted on 2016 → Y−1;
- the signal-conditioning statistics use **2020 → Y−1** (as in FXS 10);
- day t is assigned its most probable predicted state.

## 3. Tests

**H1 (direct).** Per pair: the mean next-day return (pips) in each predicted state, 2021–2026; one-way ANOVA across states.

**H2 (signals).** The 12 FXS 09 signals, net pips ($3/lot, 0.9 × spread). In year Y, a signal trades in the states whose 2020 → Y−1 mean net is > 0; a state with fewer than 30 training trades is traded. The comparisons:
- against the unfiltered signal;
- against C0, which trades the signal only if its own past mean is positive.

The measure is net pips per calendar day, with a paired Newey–West t (5 lags), two-sided against the unfiltered signal. BH is applied over the 24 tests (12 signals × 2 values of K). The null is symmetric here, because a filter can raise or lower pips.

**Usable** means:
- BH q ≤ 0.05 with a positive difference;
- it beats C0;
- 2026 net pips per day > 0.

**H3 (portfolio).** The 12 signals with each HMM filter, the unfiltered signals and C0, on $6,000 at 1, 2, 3, 5 and 10×; 2021–2026 and 2026 alone.

## 4. What this can and cannot show

An HMM always finds K states: in FX daily data these are almost surely volatility regimes. Finding states is not a result. The result is whether the **predicted** state, known before the trade, changes the outcome **out of sample**. With about 250 trades per year and per-trade standard deviations of 6–24 pips, a state effect must be around 2 pips or more to be detected.
