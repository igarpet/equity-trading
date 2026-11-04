# S09: calibrated HAR, a reverting-vs-shifted switching filter, and overnight-gap regimes

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-04, written after S08 (80d72d3) and before any S09 result |
| **Notebook** | `studies/S09_har_switching_gaps.ipynb` |
| **Engine** | `src/equity_trading/s09.py` on the S08 paths (`s08.compute`); tests in `tests/test_s05.py` |
| **Split** | **Fit on decision days before 2023-01-01; evaluate on 2023–2025.** 2026 not used. This is the first train/test split in the series. |
| **Target** | Forward β-hedged return r − β·r_ETF (no α), over 1 and 5 days, as in S08 amendment A1 |

## 0. Questions

1. **HAR (Iván):** S08's volatility ratio was an equal-weight, uncalibrated stand-in, not the `deribit-market-data` HAR (VD18, fitted on crypto 5m). Does an HAR **calibrated on equity residuals** improve the s-score?
2. **Regime switching after the first bar (Iván):** can a per-stock filter tell a residual that will revert (state R) from one whose mean has shifted (state S), using the bars after the first one?
3. **Regime defined on the first bar's gap itself (Iván):** do large overnight residual gaps get faded, or do they continue?

## 1. Methods (fixed)

**A. Calibrated HAR**
- **Model:** log-HAR (Corsi 2009), pooled over stocks, for next-session realized variance of S05 residual increments: log RV_{d+1} = c + b1 log RV_d + b5 log RV5_d + b20 log RV20_d + u. RV_d is the sum of squared 15m residual increments in the session, overnight bar included, with ≥ 90% of bars present.
- **Fit:** OLS on training days, with a smearing correction.
- **New variant:** V4 = s0/λ, λ² = forecast RV_{d+1}/RV20_d, frozen for the session and clipped to [0.25, 4].
- **Compared with:** V0 and S08's V3 on test-period IC.

**B. Switching filter**
- **Model**, on the s-score path within a session:
  - state R: Δs_t = −k·s_{t−1} + σ·e_t;
  - state S: Δs_t = σ·e_t.
  - Gaussian errors with a **common σ**, so the states differ only in pull, not in variance. Otherwise the model would split on volatility.
  - Increments from bar 1 on; s clipped at ±8 and Δs at ±5.
- **Fit:** EM (forward–backward) on 60,000 randomly sampled full-length training stock-sessions; parameters shared by all stocks.
- **Filtered P(R)** at each bar uses only that session's bars so far. At bar 0 it is the stationary probability.
- **Test-period measurements, at \|s0\| ≥ 2:**
  - contrarian hedged return by P(R) bucket (0–0.3, 0.3–0.5, 0.5–0.7, 0.7–0.9, 0.9–1) at 1 and 5 days, with standard errors;
  - IC of −s0·P(R) vs −s0;
  - number of names per cross-section with \|s0\| ≥ 2 and P(R) > 0.7.

**C. Overnight-gap regime**
- **Gap measure:** g = residual increment of the first bar / σ_bar, with σ_bar² = trailing 20-session mean RV / bars per session.
- **Target:** faded forward hedged return from the close of the first bar, −sign(g) × fwd, by \|g\| bucket [0, 1), [1, 2), [2, 3), [3, 5), ≥ 5, reported for both training and test periods. Nothing is fitted.

## 2. Pre-stated readings (test period)

| Reading | Condition | Next step if true |
|---|---|---|
| H1 | V4 IC ≥ 1.10 × V0 IC at 1d and 5d | Port the calibrated HAR into the s-score |
| M1 | At \|s\| ≥ 2 and P(R) > 0.7: 5-day contrarian ≥ 25 bps (about 2× the round trip), ≥ 10 bps above the P(R) ≤ 0.3 group, and median qualifying names ≥ 25 per cross-section | Backtest S05c `x10` with a P(R) filter |
| G1 | \|g\| ≥ 3: 5-day faded return ≥ 25 bps | Backtest a gap-fade strategy |
| G2 | \|g\| ≥ 3: 5-day faded return ≤ −13 bps (continuation beyond the round trip) | Backtest a gap-continuation strategy |

**Note on M1's count:** it was first proposed as ≥ 100 names. S08 showed \|s\| > 2 occurs for about 11% of stock-bars, about 55 names at a time, so 100 is impossible by construction. Set to 25 before any S09 result existed.

## 3. Limitations

- **The test period is not untouched:** 2023–2025 was used in S02–S08. The split guards against fitting the HAR and switching parameters to the evaluation data, not against earlier design choices.
- **One shared switching model,** with Gaussian errors despite fat tails (S08 R4) and states defined only by pull.
- **Gap regime has no news or earnings data;** large gaps are a proxy.
- **Bucket returns overlap in time;** the standard errors treat observations as independent and are optimistic.

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 studies/S09_har_switching_gaps.ipynb
```

Uses the shared panel cache. The S08 paths take a few minutes, the EM fit about a minute, and filtering every session a few minutes.
