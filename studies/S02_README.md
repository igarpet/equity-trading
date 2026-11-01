# S02: Cointegration base rate of intraday pairs

| | |
|---|---|
| **Status** | Pre-registration, written 2026-11-01 before any Alpaca data was downloaded. |
| **Notebook** | `studies/S02_cointegration_base_rate.ipynb` |
| **Board card** | "Cointegration base rate: are intraday same-sector pairs mean-reverting at all?" |
| **Data** | Alpaca SIP bars, `adjustment=all`, regular session (09:30–16:00 New York) |
| **Period** | 2020-01-01 → 2025-12-31 |
| **Sealed** | 2026-01-01 onward is **not downloaded or used**; it is reserved for strategy confirmation |
| **Bars** | 30Min (primary), 15Min (secondary) |

## Question

Same-sector stock pairs are the base material of the thesis strategy. S02 asks two questions about them:

1. **Rate:** are they cointegrated more often than an empirical null, the cross-sector pairs?
2. **Persistence:** when a pair passes in a formation window, does the frozen-β spread stay stationary in the next window? This is the property a trading strategy actually needs.

The stock-vs-sector-ETF residual (the Avellaneda & Lee leg) is tested the same way, against stock-vs-other-sector-ETF as its null.

S01 found 6 of 180 pairs passing at 5% over 60 days, about the chance level. That is why this is a gate before any strategy work.

## Design (fixed)

- **Universe:** `UNIVERSE_01` (40 stocks, 4 sectors) plus the sector ETFs XLK, XLF, XLE and XLU. Pair counts: 180 `sector`, 600 `cross`, 40 `etf`, 120 `etf_cross`.
  - *Known survivorship bias:* these are current names, chosen in 2026. Exploratory.
- **Formation windows:** L ∈ {20, 60, 120} trading days, stepping 20 trading days. Out-of-sample (OOS) window: the next 20 trading days.
- **Formation test:** Engle–Granger on log prices.
  - OLS with intercept; alphabetical orientation for stock pairs; stock = y for ETF pairs.
  - ADF with constant and a fixed 1 lag.
  - MacKinnon p-value (N = 2). Pass = p ≤ 0.05.
- **OOS measures:** α, β, mean and sd frozen from formation.
  - **Band-hold share:** the fraction of OOS bars with the spread within mean ± 2 sd. This is the gated measure.
  - **ADF** (constant, 1 lag) on the OOS spread, with a MacKinnon N = 1 p-value, since nothing is re-estimated. Descriptive only.
- **Descriptives** (no gates):
  - OOS drift, |mean_OOS − mean_formation| / sd_formation;
  - OOS crossings of the formation mean per day;
  - AR(1) half-life in formation.
- **Coverage rule:** a pair-window is used only if both legs have ≥ 90% of bars. Up to 2 missing bars are forward-filled within a session, never across sessions.
- **Inference:** moving-block bootstrap over formation windows (block = 3, 2,000 reps). This keeps the dependence between pairs tested in the same window.

## Gates

**Primary:** 30Min, L = 60. The groups are `sector` vs `cross`.

| Gate | Condition |
|---|---|
| **G1 (rate)** | pass rate(sector) − pass rate(cross): 95% CI lower bound > 0, **and** ratio ≥ 2 |
| **G2 (persistence)** | Among formation passers: mean OOS band-hold share in `sector` − mean OOS band-hold share in `cross` ≥ 0.10, **and** 95% CI lower bound > 0. Band-hold share = fraction of OOS bars with the frozen spread within formation mean ± 2 formation sd. |

### Amendment A1 (2026-11-01, before any real data)

The original G2 compared OOS ADF stationarity between formation passers and failers within `sector`. A synthetic run failed it at the primary configuration (lift 3.4 pp, CI [2.3, 4.6]), even though every synthetic same-sector pair is cointegrated by construction. Two reasons:

- **Low power:** an ADF test on 20 days of a spread with a multi-day half-life rarely rejects.
- **Wrong contrast:** formation failers are largely false negatives, so comparing passers with failers understates persistence.

The amended G2 compares group passers with null-group passers, which are mostly spurious. It uses band-hold, which is closer to what a trade needs: the formation equilibrium holding. On the same synthetic data, sector passers hold 0.80 vs 0.61 for cross passers (lift 0.19, CI [0.15, 0.25]).

The original statistic is still reported as `adf_oos_lift`, descriptive only.

| Verdict | Condition |
|---|---|
| `PERSISTENT_COINTEGRATION` | G1 and G2 pass |
| `SIGNAL_NOT_PERSISTENT` | G1 passes only |
| `NO_COINTEGRATION_SIGNAL` | G1 fails |

**Secondary:** the same gates and verdict for `etf` vs `etf_cross`.

**Reported only:** all other window lengths, the 15Min results, pass rates over time, and results by sector.

## Consequences

| Verdict | Next step |
|---|---|
| `PERSISTENT_COINTEGRATION` | The thesis pair mechanism has a base to build on. Proceed to the feature-based selection model, labelled on OOS outcomes. |
| `SIGNAL_NOT_PERSISTENT` | Pairs exist but do not persist. Selection must predict persistence, or the strategy must use the ETF residual instead. |
| `NO_COINTEGRATION_SIGNAL` | Same-sector intraday pairs are not a viable base at this frequency. Reconsider the frequency (daily) or the spread definition (ETF / PCA residuals). |

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace studies/S02_cointegration_base_rate.ipynb
```

The first run downloads about 6 years × 44 symbols × 2 bar sizes into `<EQUITY_TRADING_DIR>/Alpaca/` (around 10–20 minutes). Later runs use the cache. Set `S02_SYNTHETIC=1` for an offline run on synthetic data.

**Outputs:**
- `results/s02/summary.json`
- `results/s02/describe_*.csv`
- `results/s02/pass_rate_over_time_*.csv`

Full pair-window rows go to `<EQUITY_TRADING_DIR>/EquityTrading/results/s02/`, outside git. Commit the executed notebook and `results/s02/`.
