# S08: why is |s| > 3 routine? A diagnostic of the S05 s-score

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-04, written after S05d (9cd8551) and before any S08 result |
| **Notebook** | `studies/S08_sscore_diagnostic.ipynb` |
| **Engine** | `src/equity_trading/s08.py` (tests in `tests/test_s05.py`) |
| **Data** | Shared 15Min panel 2020–2025 (`PANEL_VERSION` 0.2.0). 2026 not used. No trading, no costs, no placebos. |

## 0. Question

In S05d the 100-slot book stayed full even at \|s\| ≥ 3. At any time more than 100 of about 500 stocks had \|s\| > 3, whereas a correctly scaled OU score would exceed 3 only about 0.27% of the time. Four candidate causes, each with a different fix:

| Cause | Fix it would point to |
|---|---|
| Noise in m = a/(1 − b) when b ≈ 1 | robust or shrunk m |
| Overnight gaps (earnings, news) entering the residual | exclude or treat the overnight bar separately |
| Volatility clustering: σ_eq is a 20-day average | volatility forecast in the denominator (Iván's `deribit-market-data` HAR engine) |
| Fat-tailed innovations | Student-t likelihood (Iván's question) |

Plus whether mean reversion itself depends on the market volatility state, which would justify a regime model (HSMM) for b.

**Prior evidence from `deribit-market-data`:**
- HSMM turns 001–016: no development evidence for discrete volatility regimes, and a fitted GARCH with very short memory (α + β = 0.24), with the persistence carried by the HAR components.
- So S08 uses a HAR-style variance ratio as the volatility engine and a simple market-volatility split as the regime test, before any HSMM is built.

## 1. Variants (fixed)

All use the S05 daily sector-ETF regression (20 sessions, α and β frozen for the next day) and the AR(1) fit of the cumulative residual.

| Variant | s-score |
|---|---|
| V0 | S05 as traded: s = (X − m_c)/σ_eq |
| V1 | **Overnight bar excluded:** the 09:30 bar contributes no residual increment, in the window fit and in the live residual |
| V2 | V1 / λ_t, where λ_t² = (RV_1 + RV_5 + RV_20)/(3·RV_20) is a HAR-style ratio of the mean squared V1 increment over the last 1, 5 and 20 sessions, using bars up to t; λ clipped to [0.25, 4] |
| V3 | V0 / λ_t (the same λ): volatility scaling alone |

**V1 caveat:** removing increments from a mean-reverting level is only valid if those moves are **permanent**. If overnight residual moves revert, V1 adds a random walk and does worse. On the synthetic panel, where overnight moves revert by construction, V1 is much worse than V0 (rank IC 0.08 vs 0.25). Real data decide.

**Rejected while building:** dividing increments by a time-of-day volatility profile and re-cumulating them. This rescales the reversion drift along with the noise and inflated extreme \|s\| on synthetic data. Time-of-day volatility mostly averages out over a reversion time of 1–3 sessions.

**Target for all variants:** the forward sum of S05 residual increments (V0 parameters, overnight included) over 1 hour (4 bars), 1 day (26) and 5 days (130). This is the residual return a position actually earns.

## 2. Measurements

1. **Exceedance:** share of stock-bars with \|s\| > 1.25, 2, 3 and 4 per variant, against a normal OU; share with \|s\| > 3 by bar of the session.
2. **Mean decomposition (V0):** the share of \|−m/σ\| beyond 1, 2 and 3; among \|s0\| > 3, the share with \|m/σ\| > 2.
3. **Innovation tails:** median excess kurtosis of the AR(1) innovations per stock-day (V0 and V1); Student-t ν fitted on 800 sampled stock-days of standardized V1 innovations.
4. **Signal quality:** cross-sectional rank IC of −s against the forward residual return, sampled every 4th bar. Means of daily IC series with Newey–West t-statistics, and by year.
5. **Economics:** mean contrarian residual return (bps) by s bucket at 1 and 5 days, against the S05b measured round trip of about 13 bps.
6. **Calibration:** within the session, the empirical probability of s reaching 0 before \|s\| = 4, by \|s\| bucket, against OU's prediction (unresolved paths reported).
7. **Regimes:** terciles of trailing 20-session SPY intraday realized variance, known before each day. Per tercile: median reversion time 1/κ, 1-day IC (V0 and V3), and V0 contrarian bps for \|s\| in [2, 3].

## 3. Pre-stated readings (what each result triggers)

| Reading | Condition | Next step if true |
|---|---|---|
| R1 | V3 IC ≥ 1.10 × V0 IC at both 1d and 5d | Port the volatility engine into the s-score |
| R2 | V1 IC ≥ 1.10 × V0 IC at both 1d and 5d | Adopt overnight exclusion |
| R3 | V0 1-day IC in high-vol vs low-vol tercile differs by ≥ 25% | Regime model (HSMM) for b / gating |
| R4 | V1 innovations' median excess kurtosis > 3 | Test Student-t likelihood estimation |
| R5 | Among \|s0\| > 3, more than 50% have \|m/σ\| > 2 | Robust or shrunk m |
| R6 | Any bucket's 5-day contrarian return > 13 bps | A longer-horizon version can clear costs; turnover, not signal, was the binding constraint |

These are descriptive thresholds that decide what to build next. They are not significance tests.

## 4. Limitations

- **Development data:** 2020–2025 was used in S02–S05d.
- **IC on overlapping targets:** forward returns at 1d and 5d overlap across sampled bars; Newey–West lags are 1 and 5 days on daily IC means.
- **Regime terciles are full-sample quantiles** of a causal variable. This is fine for a descriptive split, not for trading.
- **Within-session calibration is truncated at the close;** the unresolved share is reported.

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 studies/S08_sscore_diagnostic.ipynb
```

Uses the shared panel cache, so nothing is downloaded. The s-score paths take a few minutes. IC and buckets on about 9,700 cross-sections take a few minutes more. Memory: about 1 GB for the float32 paths.

## Amendment A1 (2026-11-04, after the first run a261864): tradable target

- **The flaw:** the original target, forward residual increments r − α − β·r_ETF, subtracts each stock's 20-session drift α. A β-hedged position does not earn −α. Over 5 days (130 bars), α × 130 can be hundreds of bps, so the first run's contrarian bps (R6) and possibly its IC levels did not measure tradable returns.
- **The evidence:** S05c's `x10` earned about 20 bps gross per trade over a median of 232 bars, while the residual-target buckets showed 90–180 bps at 5 days.
- **Primary target is now the hedged return** r − β·r_ETF (the day's β, no α), forward over the same horizons. It is used for every reading. The residual target is still reported for comparison.
- Regime table: V1 added (1-day IC; 5-day contrarian bps at \|s\| in [2, 3]).
- **Readings and thresholds are unchanged.**

**First-run results (residual target; superseded for R1, R2, R3 and R6 by the re-run):**
- V1 1-day IC 0.101 vs V0 0.021 (Newey–West t = 32 vs 6); 5-day IC 0.224 vs 0.102.
- Volatility scaling: no gain (V3 ≤ V0, V2 ≈ V1).
- τ is flat across volatility terciles (2.44–2.53 sessions); V0 IC is 0.030 (low volatility) vs 0.014 (high).
- V1 innovations: median excess kurtosis 3.0, Student-t ν ≈ 4.0.
- Among \|s0\| > 3, 18% have \|m/σ\| > 2.

## Result of the re-run (80d72d3, hedged target, primary)

| Rank IC of −s vs hedged return | 1h | 1d | 5d |
|---|---|---|---|
| V0 | 0.008 (t 6.4) | 0.014 (t 4.1) | 0.015 (t 2.7) |
| V1 | 0.003 | 0.005 | 0.006 |
| V3 | 0.008 | 0.013 | 0.014 |

- **V1's advantage in the first run was an artifact of the −α term.** On tradable returns, excluding the overnight bar makes the signal *worse*. R2 is false.
- **Readings:** R1, R2, R3 and R5 false; R4 true (heavy-tailed innovations); R6 true, but only for the extreme V0 buckets.
- **Contrarian hedged return of V0 at 5 days:** 10–12 bps for \|s\| in [2, 3], 15–19 bps for \|s\| in [3, 4], 18–30 bps beyond 4, against a 13 bps round trip. This is consistent with S05c `x10`, which entered at a median \|s\| of 3.5, earned about 20 bps gross per trade and had a gross/cost ratio of 1.47.
- **Regimes:** reversion time is flat across terciles. V0's 5-day contrarian bps at \|s\| in [2, 3] by tercile is 7.4 (low), −1.0 (mid) and 25.8 (high volatility): not monotonic, and noted only as a possible lead.
- **Conclusion:** the tradable edge of the sector-ETF residual reversal at 15m is about 1–2× the market-order round trip, and only for extreme deviations held for a week or more. None of the tested model changes (overnight exclusion, HAR volatility scaling) improves it.
