# S05a: Avellaneda–Lee residual mean reversion, sector-ETF factor, 15-minute bars

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-03, written before any S05a result exists |
| **Notebook** | `notebooks/Residual_Reversion/05a_avellaneda_lee_etf.ipynb` |
| **Engine** | `src/equity_trading/s05.py` (tests: `tests/test_s05.py`) |
| **Method source** | Avellaneda, M. & Lee, J.-H. (2010), *Statistical arbitrage in the US equities market*, Quantitative Finance 10(7), ETF-factor variant |
| **Data** | Alpaca SIP 15Min, `adjustment=all`, regular session, 2020-01-01 → 2025-12-31. 2026 is not used. |
| **Universe** | `src/equity_trading/reference/sp500_constituents_20260913.csv`, each stock against its GICS-sector Select Sector SPDR (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC). SPY is the benchmark only. |

## 0. Why this study

S04 traded pairs selected by daily cointegration tests. The selection step passed only about 0.09% of pair-days, so the strategy traded rarely. Over 2021–2025 it did no better than random pairs or random timing.

S05a removes pair selection altogether. Every stock is traded on its residual against its sector factor. The design goal is to be in the market continuously, with high turnover.

The PCA-factor variant of the paper is deferred to S05b. Hedging with eigenportfolios needs a portfolio-level netting engine that does not exist yet.

## 1. Rules

All decisions use data up to the close of day *e* and apply to day *e* + 1.

| Step | Rule |
|---|---|
| Panel | Same as S04: exchange-calendar 15m grid, closes forward-filled within a session for up to 2 bars, missing open = previous close in the same session. Returns are bar log returns; the first bar of each session carries the overnight move. |
| Factor regression | At each close, for each stock over the trailing **20 sessions** (520 bars): OLS of its returns on its sector ETF's returns, r = α + β r_ETF + ε. Requires ≥ 95% of bars present for both. |
| OU fit | X = cumulative ε over the window (it ends at 0). AR(1): X(t+1) = a + b X(t) + ζ. κ = −ln b per bar, m = a/(1−b), σ_eq = √(var ζ / (1−b²)). |
| Tradable | 0 < b < 1 and mean-reversion time 1/κ < ½ × window (paper). m is centred by subtracting the mean m of tradable stocks that day (paper). |
| s-score | During day *e* + 1, X keeps accumulating bar by bar with that day's frozen α and β. s = (X − m_c) / σ_eq. |
| Signals (paper) | Open long if s < −1.25. Open short if s > +1.25. Close long if s > −0.50. Close short if s < +0.75. No stop-loss, no time stop. A position is also closed if its stock is not tradable on a later day. |
| Execution | Signal at a bar close, fill at the next bar's open. Costs: a flat 2 bps per fill on every stock and ETF trade (S04 amendment A1). |
| Position | Stock leg = 1% of equity (integer shares). ETF leg = −β × the stock notional, fixed at entry. ETF hedges are netted per ETF, so only the net ETF change is traded and charged. Exits use each day's re-estimated parameters, as in the paper. |
| Capacity | At most 100 open positions. When more stocks signal than there are free slots, largest \|s\| first. |
| Capital | $100,000. Up to about 200% gross (stock legs plus hedges). Cash earns nothing. Shorting is free and unconstrained. |

## 2. Evaluation (pre-registered)

**Runs:** strategy at 1× cost (primary), at 2× cost (4 bps per fill), and at 0× (gross).

**Placebos:**
- **Random direction:** the strategy's own trades, with a random sign for each trade (10,000 draws). It asks whether the direction carries information, given the same timing and holding periods.
- **Random timing:** 1,000 full backtests on the same engine. Contrarian entries at random flat, tradable bars where \|s\| is beyond the exit band, at the strategy's own entry rate, with the same exits.

**Gates:**

| Gate | Condition |
|---|---|
| **G1 bootstrap** | Moving-block bootstrap 95% CI of mean daily net return > 0 for blocks of 10, 20 and 40 days |
| **G2 random direction** | Upper-tail p ≤ 0.05 for the summed gross P&L |
| **G3 random timing** | Total return and Sharpe each have upper-tail p ≤ 0.05 |
| **G4 double cost** | Total return > 0 and Sharpe > 0 at 4 bps per fill |

**Verdicts:**
- **Statistical:** `SIGNAL` if G1, G2 and G3 all pass; `PARTIAL` if some pass; `NO_SIGNAL` if none.
- **Economic:** `PROFITABLE_AFTER_COSTS` if net total return > 0 and G4 passes.
- **Overall:** `EDGE_CANDIDATE` only if both hold.

**Reported only:**
- trades per day, mean gross P&L per trade in bps, holding time, average open positions and gross exposure;
- cost share of gross P&L, results by year and by entry \|s\|;
- beta and alpha vs SPY.

## 3. Known limitations

- **Not untouched data:** 2020–2025 was already used by S02–S04. This is development evidence, and no holdout confirmation is planned.
- **Survivorship:** current-snapshot universe.
- **Costs and shorting:** flat 2 bps per fill with no market impact; free and unlimited shorting. ETF hedges are fixed at entry, not rebalanced as β changes.
- **Overlapping trades:** the random-direction test treats trades as exchangeable, and overlapping trades in the same sector are correlated, so its p-value is optimistic. G3 is the stricter test.
- **Overnight:** positions are held overnight, so earnings gaps enter both the residuals and the P&L.

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/Residual_Reversion/05a_avellaneda_lee_etf.ipynb
```

**Data:** the first run downloads 15Min bars for the 11 sector ETFs (about 5 min; stocks and SPY are already cached) and builds a new panel cache.

**Compute:** estimation takes a few minutes. One backtest takes roughly 10–30 s, so 1,000 timing placebos take roughly 1–2 h on 4 cores. Lower `N_PLACEBO` for a first look; the gates need 1,000.

## Correction C1 (2026-11-03): panel field-order bug, all earlier real-data results void

- **The bug:** `s04.load_panel` passed bar fields to the panel builder as (open, high, low, close), but the builder expected (open, close, high, low). On real data the panel's close was the bar **high**, its high was the **low**, and its low was the **close**. Opens were correct. The synthetic panel was built correctly, so tests did not catch it.
- **Scope:** every real-data run of S04, S05a, S05b and S06 before this correction (fixed in the commit that adds this note). Signals, estimates and marks used bar highs. The S04 Abdi–Ranaldo estimate (36 bps, the reason for amendment A1) was computed from scrambled fields. S02 and S03 used a different, correct loader and are not affected. The S05b quote samples are not affected.
- **Fix:** fields passed by name. A regression test checks the OHLC fields stay distinct. One shared panel cache is now keyed on `PANEL_VERSION` 0.2.0, and every derived cache is keyed on it too, so no stale panel or result can be reused.
- **Status:** the pre-registered rules are unchanged; only the data were wrong. The earlier results in `results/` stay in git history and must not be cited. Re-runs replace them.
