# S03: Thesis pair discovery on intraday bars, plus Johansen, rolling cointegration and economic-distance clustering

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-01, written before any data for this universe was downloaded |
| **Notebook** | `studies/S03_thesis_pair_discovery_intraday.ipynb` |
| **Method source** | García Petrov (2026) thesis, `igarpet/Thesis` (`01_Stock_Data`, `02_Pair_Selection`, `src/cointegration.py`). The port in `src/equity_trading/pairs.py` reproduces the thesis p-values exactly on the thesis's own formation data (max \|Δp\| = 4e-14). |
| **Data** | Alpaca SIP, `adjustment=all`, regular session, 2020-01-01 → 2025-12-31. 2026 sealed. |
| **Bars** | `TIMEFRAME` = 30Min or 15Min, run once each. Outputs go to `results/s03/<TIMEFRAME>/`. |

## 1. Thesis pipeline, unchanged except for bar size

| Step | Thesis (daily) | S03 (intraday) |
|---|---|---|
| Universe | S&P 500 snapshot of 2026-09-13 (503 symbols) | Same file: `reference/sp500_constituents_20260913.csv` |
| Prices | Yahoo adjusted close | Alpaca SIP close, `adjustment=all`, regular-session bars |
| Split | First 70% of observations = formation | Same, on bars |
| Availability | ≤ 5% formation missing, valid first formation price, complete OOS, formation forward-filled | Same. **The only adaptation:** before these rules, a missing bar inside a session is forward-filled for up to 2 bars, because a bar with no trade is an unchanged price, not missing data. |
| Pre-screen | Pearson correlation of formation log returns; each stock's 10 highest signed correlations | Same, on bar log returns |
| Integration screen | ADF levels not rejected at 5% **and** ADF differences rejected at 5%, AIC lags | Same |
| Cointegration | Fixed orientation (alphabetical y), OLS log prices with intercept, `coint(trend="c", autolag="aic")`, p ≤ 1%, β > 0 | Same |

## 2. Additions (Iván, 2026-11-01)

1. **Johansen trace test**
   - Bivariate VECM of log prices; `det_order = 0`; lag by AIC (`select_order`, max 10).
   - Selected if trace(r = 0) > 99% critical value, trace(r ≤ 1) ≤ 95% critical value, I(1) screen passed, Johansen β > 0. These are the same conventions as the thesis rule, and the test needs no orientation choice.
2. **Rolling (dynamic) cointegration**
   - Inside the formation sample only: windows of 63 sessions, stepping 21 sessions.
   - Engle–Granger with a fixed 1-lag ADF and MacKinnon p-value. AIC lag search over every window would be impractical.
   - Per pair: stability = share of windows with p ≤ 5%, β coefficient of variation, and median rolling half-life.
   - **Rolling-stable** = stability ≥ 0.5.
3. **Clustering by economic distance**, as alternative candidate generators to the correlation pre-screen:
   - **GICS:** all pairs within the same GICS Sub-Industry.
   - **Correlation clusters:** hierarchical clustering (average linkage) on the Mantegna distance $d_{ij}=\sqrt{2(1-\rho_{ij})}$ of formation log returns. Number of clusters = number of GICS sub-industries among retained stocks, so both methods have the same granularity. Clusters with more than 40 members are skipped and reported.
   - Fundamental-distance clustering is deferred, because no fundamentals source is connected yet.
4. **Empirical null:** 3,000 random pairs outside every candidate set, passed through every rule.

**Rules** applied to each candidate set:

| Rule | Definition |
|---|---|
| `EG` | Thesis rule |
| `JO` | Johansen |
| `EG+JO` | Both |
| `EG+ROLL` | Thesis rule and rolling-stable |

## 3. Evaluation (pre-registered)

The out-of-sample sample is the last 30% of bars, with frozen formation α, β, mean and sd, as in the thesis.

**Per selected pair:**
- `band_hold`: share of OOS bars with |z| ≤ 2.
- `drift_sd`: |mean OOS z|.
- `reversion_rate`: share of OOS excursions to |z| ≥ 1.5 that return to z = 0 before |z| ≥ 3 or the end of the sample.
- `adf_oos_p`.

**Gates for each candidate set × rule, against null pairs selected by the same rule:**

| Gate | Condition |
|---|---|
| **G1 (selection excess)** | selection rate ≥ 2 × null selection rate |
| **G2 (out-of-sample)** | mean `reversion_rate` − null mean ≥ 0.10 **and** mean `band_hold` − null mean ≥ 0.10. Both 95% bootstrap CIs (resampling pairs) must be above 0. If the rule selects fewer than 10 null pairs, the reference becomes all 3,000 null pairs, each fitted with formation OLS. This fallback is flagged in `gates.csv`. |

**Verdicts:**

| Verdict | Condition |
|---|---|
| `DISCOVERS_PERSISTENT_PAIRS` | G1 and G2 pass |
| `SELECTS_BUT_NOT_PERSISTENT` | G1 passes only |
| `NO_EXCESS` | G1 fails |

**Primary:** the thesis method (correlation pre-screen + `EG`) at 30Min. Everything else is a comparison arm.

**Reported only:**
- the estimated false-discovery share, null rate × candidates / selected;
- EG vs Johansen agreement;
- overlap with the thesis's 98 daily pairs.

## 4. Known limitations

- **Current-snapshot universe:** survivorship and selection bias, as in the thesis.
- **Shared legs:** pairs that share a stock are not independent, so the pair bootstrap CIs are optimistic.
- **Span:** 2020–2025 is shorter than the thesis's 2016–2025. More bars do not add power against slow mean reversion (Shiller & Perron 1985).

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 studies/S03_thesis_pair_discovery_intraday.ipynb
```

**Download:** about 500 symbols × 6 years. Roughly 4 h for 30Min and 7 h for 15Min at the free-plan rate limit. It resumes if interrupted, and the 44 S02 symbols are already cached.

**Tests:** about 10,000 pairs. With 8 cores, roughly 30 min at 30Min and 1.5 h at 15Min. Results are cached in `<EQUITY_TRADING_DIR>/EquityTrading/results/s03/`, so a re-run skips finished steps.

**To run 15Min:** set `TIMEFRAME = "15Min"` in the first code cell and run again.
