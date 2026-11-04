# S01: Is intraday anti-persistence real, or estimator bias?

| | |
|---|---|
| **Status** | **On hold: failed its own power check before any real data was run (2026-11-01).** See "Power check" below. |
| **Board card** | "Evidence test: is H ≠ 0.5 real, and does fOU beat OU?" (part 1 only) |
| **Data** | Yahoo Finance via yfinance, last 60 days, regular session, split/dividend-adjusted closes |
| **Intervals** | 30m (primary), 15m (secondary) |

## Question

Within-sector spreads at 15–30 minutes often show $\hat H < 0.5$ under the thesis change-of-frequency estimator.

Three things push $\hat H$ down even when $H = 0.5$ exactly:
- fast mean reversion,
- bid-ask noise,
- the U-shaped intraday volatility pattern.

S01 asks whether real spreads sit **below what an OU process with measurement noise, matched to each pair, already produces**.

## Design (fixed)

1. **Universe** (`equity_trading.universe.UNIVERSE_01`): 4 sectors × 10 liquid S&P 500 names (tech, banks, energy, utilities). All 45 within-sector pairs per sector, 180 in total. Orientation is alphabetical, as in the thesis.
2. **Spread:** $X_t = \log P_y - \alpha - \beta \log P_x$, with OLS fitted on the full window. This is descriptive, not a trading test.
3. **Deseasonalisation:** divide each increment by the s.d. of increments in the same intraday slot, then re-cumulate. The first bar of each day, which carries the overnight gap, is its own slot.
4. **Estimate:** $\hat H$ by the change-of-frequency estimator, on the deseasonalised spread.
5. **Primary null:** OU plus i.i.d. measurement noise, i.e. ARMA(1,1) fitted to the deseasonalised spread. Simulate 200 paths of the same length and re-estimate $H$ on each with the same estimator.
   - One-sided p-values: $p_{low} = (1+\#\{H_{null} \le \hat H\})/201$, and $p_{high}$ symmetrically.
6. **Secondary null:** clean OU (AR(1), no noise). Reported only, to show how much of any gap is noise.
7. **Validity flag:** AR(1) half-life ≥ 4 bars. Below that, the scaling window is too short to say anything about $H$.

## Gates (primary: 30m, OU + noise null, all stationary pairs)

| Verdict | Condition |
|---|---|
| `H_BELOW_NULL` | ≥ 20% of pairs have $p_{low} \le 0.05$ (four times the 5% expected under the null) **and** median $(\hat H - \bar H_{null}) \le -0.03$ |
| `H_ABOVE_NULL` | Same thresholds mirrored on the upper tail |
| `NO_EVIDENCE` | Otherwise |

**Reported but not used for the verdict:**
- 15m results;
- the clean-OU null;
- the valid-window subset;
- the Engle–Granger p ≤ 0.05 subset;
- results by sector.

## What each verdict means

- **`H_BELOW_NULL`:** anti-persistence beyond estimator bias. This supports the fOU arm and unlocks the out-of-sample fOU vs OU test.
- **`NO_EVIDENCE`:** $\hat H < 0.5$ is explained by mean reversion plus noise, so OU is sufficient at this frequency. fOU must then justify itself through out-of-sample forecasting alone.

## Limitations (known in advance)

- **One 60-day window.** No out-of-sample component.
- **Shared legs.** Pairs that share a stock are not independent, so the 20% share is a screen, not a formal family-wise test.
- **In-window estimation.** $\beta$ and the seasonal profile are estimated on the same window they are applied to.
- **Yahoo data quality.** Bars may contain gaps or errors.

## Power check (synthetic, before real data)

**Setup:** spreads simulated as true fOU ($\phi = 0.97$ per bar), with and without measurement noise, 20–25 pairs per cell, 200 null paths. The table shows the share of pairs rejected at 5% against the OU + noise (ARMA(1,1)) null.

| True H | n bars | Thesis COF (step 1) | COF step 2 | Increment ACF, lags 2–10 |
|---|---|---|---|---|
| 0.3 | 520 (≈ 60 days of 30m bars) | 0–5% | 10% | 5–10% |
| 0.3 | 2,600 (≈ 1 year of 30m bars) | 0% | 15–30% | 30% |
| 0.4 | 2,600 | 0% | 10–15% | 15% |
| 0.5 | 520 / 2,600 | 0% | 0–10% | 0–5% |

**Conclusions**

1. **The thesis estimator has no power against an OU + noise null.** The fitted ARMA(1,1) absorbs anti-persistence as measurement noise, because both produce negative lag-1 autocorrelation in increments.
2. **With 60 days of Yahoo data, no statistic reaches the 20% gate even when H = 0.3 is true.** A `NO_EVIDENCE` result would therefore be uninformative.
3. **Power grows with sample length.** Separating fOU from OU + noise needs years of bars and a likelihood-based test that uses all lags: Whittle or exact likelihood, fOU + noise vs OU + noise, with a parametric-bootstrap likelihood-ratio test.

S01 is redesigned as S01b on Alpaca data. The Yahoo pipeline remains usable as a data and plumbing smoke test.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python notebooks/Pairs_Cointegration/01_hurst_null.py            # downloads Yahoo data into ./data, writes notebooks/Pairs_Cointegration/01/
S01_SYNTHETIC=1 python notebooks/Pairs_Cointegration/01_hurst_null.py   # offline smoke test
```

**Outputs:** `notebooks/Pairs_Cointegration/01/<interval>/pairs.csv`, `summary.json`, `manifest.json`. Commit the `notebooks/Pairs_Cointegration/01/` folder after the run.
