# S04: Dynamic OU pairs backtest on 15-minute bars (stocks, long/short)

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-02, written before any S04 result exists |
| **Notebook** | `studies/S04_dynamic_ou_pairs_backtest.ipynb` |
| **Engine** | `src/equity_trading/s04.py` (tests: `tests/test_s04.py`) |
| **Data** | Alpaca SIP bars, `adjustment=all`, regular session, 15Min, 2020-01-01 → 2025-12-31 (same window as S03). 2026 sealed as the second holdout. |
| **Universe** | `reference/sp500_constituents_20260913.csv`; SPY is downloaded as the benchmark only (never traded). |
| **Arms** | `ARM = "hl5"` and `ARM = "fixed60"`, run once each. Outputs go to `results/s04/<ARM>/`. |

## 0. Prior evidence, stated before the run

- **S02 and S03:** no cointegration selection excess over a random-pair null at intraday bars, with frozen formation parameters.
- **S04 asks a different question:** can daily re-estimation with a dynamic pair list extract a tradable signal anyway?
- **Window power.** On the synthetic panel, where same-sector pairs are cointegrated by construction with half-lives of 1–5 sessions, the share of pairs reaching Engle–Granger p ≤ 1% is:

  | Window (sessions) | Cointegrated pairs | Cross-sector pairs (null) |
  |---|---|---|
  | 5 | 1.6% | 1.0% |
  | 10 | 1.4% | 1.5% |
  | 20 | 4.5% | 1.2% |
  | 40 | 22% | 3.5% |
  | 60 | 52% | 3.2% |

  A window of 5 × half-life is about 5–25 sessions for these pairs, so `hl5` is expected to select close to the null rate. `fixed60` is the arm with power. This was found while building the engine and is recorded here before any real-data run.

## 1. Pipeline

All decisions use data up to the close of decision day *e* and take effect on *e + 1*.

| Step | Rule |
|---|---|
| Panel | Exchange-calendar 15m grid. Close forward-filled within a session for up to 2 bars. A missing open is filled with the previous close of the same session. No full-sample availability filter: a pair is tested on a day only if both legs have ≥ 95% of bars in the window. |
| Costs | One-way cost on day *d* = ½ × the Abdi–Ranaldo (2017) spread, from 15m OHLC over sessions *d*−20 … *d*−1, floored at 1 bp. Applied to every fill. No commissions, borrow fees or dividends. Every stock is assumed shortable. |
| Candidates | Every 5 sessions: each stock's top-10 signed correlations of 15m log returns over the trailing 20 sessions, excluding the overnight bar. Orientation is alphabetical *y*. The pool on day *e* is the union of the last 4 refreshes. |
| Daily test | Half-life from an AR(1) fit on an OLS spread over a 20-session reference window. Engle–Granger with a fixed 1-lag ADF and MacKinnon p-value (N = 2) over the test window. `hl5`: clamp(5 × HL, 5, 60) sessions. `fixed60`: 60 sessions. BH-FDR across each day's batch. |
| Pass / fail | **Pass:** BH-adjusted p ≤ 1%, β > 0, valid half-life, and the window holds ≥ 5 half-lives. **Fail:** raw p > 10%, β ≤ 0, invalid half-life, or window < 5 half-lives. Anything else is neutral. |
| Hysteresis | **Tradable** after 3 consecutive passes. **Removed** after 3 consecutive fails. A day with no test counts as a fail. Neutral resets both streaks. Tradable pairs keep being re-tested after they leave the correlation pool. |
| Entry | Signal at a bar close, fill at the next bar's open. Requires: pair tradable and flat; 1.5 ≤ \|z\| < 3; and (1.5 − 0.5)·sd ≥ 2 × round-trip cost. z uses that day's parameters (estimated at the previous close). |
| Sizing | 4% of equity gross per pair, split by β (notional on *x* = β̂ × notional on *y*), integer shares. At most 15 open pairs. When more pairs signal than there are free slots, shortest half-life first. |
| Exit | First of: \|z\| ≤ 0.5 (take profit), \|z\| ≥ 3 (stop), 3 × half-life bars held (time), pair removed (break). Entry parameters are frozen for the life of the position. Exits also fill at the next open. Liquidation at the last close of 2025. |
| Capital | $100,000. No leverage beyond about 60% gross. Cash earns nothing. |

## 2. Evaluation (pre-registered)

**Runs:**
- Strategy at 1× cost (primary).
- Strategy at 2× cost and at 0× cost (gross).

**Placebos (1,000 each, same engine, costs and caps):**
- **P-pairs:** every tradable spell keeps its calendar, but its pair is replaced by a random pool pair with a valid OU shape that day (β > 0, valid half-life), without the significance or hysteresis requirement.
- **P-timing:** same pairs and spells. Contrarian entries at random flat bars with 0.5 < \|z\| < 3, at the strategy's own entry rate.

**Gates:**

| Gate | Condition |
|---|---|
| **G1 bootstrap** | Moving-block bootstrap (10,000 draws) 95% CI of mean daily net return > 0 for blocks of 10, 20 and 40 days |
| **G2 vs random pairs** | Total return and Sharpe each have upper-tail p ≤ 0.05 against P-pairs |
| **G3 vs random timing** | Total return and Sharpe each have upper-tail p ≤ 0.05 against P-timing |
| **G4 double cost** | Total return > 0 and Sharpe > 0 at 2× cost |

**Verdicts, reported separately:**

| Kind | Verdict |
|---|---|
| Statistical | `SIGNAL` if G1, G2 and G3 pass; `PARTIAL` if some pass; `NO_SIGNAL` if none |
| Economic | `PROFITABLE_AFTER_COSTS` if net total return > 0 and G4 passes |
| Prospective | `PENDING_2026_HOLDOUT`, always for this study |
| Overall | `EDGE_CANDIDATE` only if statistical = `SIGNAL` and economic = profitable; otherwise `NO_EDGE` |

**Reported only:**
- Market beta and alpha vs SPY: daily OLS with 5-lag HAC standard errors, on raw returns since no risk-free series is loaded.
- Max drawdown, exit-reason shares, holding time, cost share of gross P&L.
- Tradable-pair count over time, number of spells and their durations.
- **Calibration:** OU probability of reaching \|z\| = 0.5 before 3 from the entry z (scale-function formula, independent of κ) vs the realized take-profit share, on trades that ended at tp or stop. Trades ended by time, break or end are censored and their share is reported.

## 3. Known limitations

- **Survivorship:** current-snapshot universe, as in the thesis and S03.
- **Short side:** shorting is free and unconstrained. This is optimistic.
- **Costs are an estimate:** a high/low-based spread estimator with no quote data, and no market impact. At 4% gross per pair, impact is small for S&P 500 names.
- **Not untouched data:** 2020–2025 was already used in S02/S03 for pair discovery. This study is development; the 2026 holdout is the confirmation.
- **Optimistic CIs:** shared legs across pairs make the bootstrap CIs optimistic. The placebo gates are the main control.
- **Sticky re-testing:** once tradable, a pair is tested every day, while other pairs are tested only while in the pool. P-pairs uses the same spell calendar, so this does not favour the strategy against it.

## Run

```bash
python -m pytest -q
# arm 1 (set ARM = "hl5" in the first code cell), then arm 2 (ARM = "fixed60")
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 studies/S04_dynamic_ou_pairs_backtest.ipynb
```

**Download:** S03 cached 30Min for the full universe. If the 15Min cache only covers the 44 S02 symbols, the first cell downloads the rest. That takes roughly 7 h at the free-plan rate limit and resumes if interrupted.

**Compute:**
- Daily tests: about 1,500 days × about 5–10k pool pairs, roughly 20–60 min on 8 cores. Cached per 20-day block in `<EQUITY_TRADING_DIR>/EquityTrading/results/s04/<ARM>/`.
- One backtest takes seconds. 2 × 1,000 placebos take roughly 1–2 h on 8 cores. Set `N_PLACEBO` lower for a first pass; the gates require 1,000.
