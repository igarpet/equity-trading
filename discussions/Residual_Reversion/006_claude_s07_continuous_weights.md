# S07: S05a residual signal as continuous weights with a no-trade band

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-03, written after the corrected S05a/S05b/S06 runs (c4d67fd) and before any S07 result |
| **Notebook** | `notebooks/Residual_Reversion/07_continuous_weights.ipynb` |
| **Engine** | `src/equity_trading/s07.py` (tests in `tests/test_s05.py`) |
| **Data** | Alpaca SIP 15Min 2020–2025 (shared panel, `PANEL_VERSION` 0.2.0), S05b quote samples for costs. 2026 not used. |

## 0. Why

Corrected S05a/S05b/S06 results (c4d67fd) on the sector-ETF residual signal:
- **The direction carries the value:** gross +54.6% over 2020–2025, Sharpe 1.22, every year positive gross, random-direction p = 0.0001.
- **Timing adds nothing:** the threshold rules beat random contrarian timing only at the 83rd percentile (p = 0.17).
- **Turnover costs too much:** market orders pay 13 bps per trade against 8 bps of gross edge (S05b −25%). Passive orders lose about 25 bps per trade to adverse selection (S06 −44%).

If the value lies in direction, not timing, the same exposure can be held with less trading. S07 holds a continuous contrarian book and trades only when targets move far enough to justify the cost.

## 1. Rules

| Step | Rule |
|---|---|
| Signal | Unchanged from S05a: sector-ETF residual over 20 sessions, re-estimated daily, s-score every bar, tradable if 1/κ < ½ window. |
| Target weights | At each bar close, for tradable stocks: raw = −clip(s, −3, 3); w = 1.0 × raw / Σ\|raw\| (stock-leg gross = 100% of equity), then \|w\| ≤ 2%. Non-tradable stocks: target 0. |
| Stock trading | Market orders at the next bar open. **No-trade band:** if \|target − current\| > band, trade to the **band edge** (target ∓ band), the optimal policy shape under proportional costs (Davis & Norman 1990). band = `band_mult` × 100% / n_tradable, i.e. a multiple of the average position size, in equity. |
| Hedge | After the stock trades, each sector ETF is set to −Σ β_i × (actual stock notional_i) of its stocks whenever it is off by more than 0.1% of equity. |
| Costs | S05b measured half-spread per symbol, year and intraday slot, on every traded dollar. |
| Capital | $100,000. Stock gross ≤ 100% plus the ETF hedge. Free shorting. |

**Arms (two, fixed):** `band05` (band_mult = 0.5) and `band10` (band_mult = 1.0).

**Reference (reported only):** `band00`, which trades to target every bar.

## 2. Evaluation (pre-registered)

**Runs:** each arm at 1×, 2× and 0× measured cost.

**Placebo:** 500 backtests per arm. Each day, every tradable stock receives the s-score path of a randomly permuted tradable stock. This keeps the signal's distribution, dynamics and turnover, and removes the link between a stock's signal and its own residual.

**Gates, per arm** (α = 0.05 / 2 arms = **0.025**):

| Gate | Condition |
|---|---|
| **G1 bootstrap** | Block-bootstrap 95% CI of mean daily net return > 0, for blocks of 10, 20 and 40 days |
| **G2 shuffled signal** | Total return and Sharpe each have upper-tail p ≤ 0.025 |
| **G3 double cost** | Total return > 0 and Sharpe > 0 at 2× measured cost |

**Verdicts, per arm:**
- **Statistical:** `SIGNAL` if G1 and G2 pass; `PARTIAL` if one passes; `NO_SIGNAL` if neither.
- **Economic:** `PROFITABLE_AFTER_COSTS` if net return > 0 and G3 passes.
- **Overall:** `EDGE_CANDIDATE` if both hold.

**Reported:**
- annual turnover (× equity) and annual cost (% of equity);
- stock trades per day and average gross exposure;
- results by year, beta and alpha vs SPY;
- the `band00` reference;
- comparison with S05a's gross Sharpe of 1.22.

**Pre-stated reading:**
- If gross Sharpe falls far below 1.2, the band destroys the signal.
- If gross Sharpe holds but net is negative, turnover is still too high for this horizon.

**Synthetic check (engine only):** band 0.5 cuts turnover from about 1,760× to 500× a year and costs from 53% to 15% of equity. Band 1.0 cuts them to 230× and 7%.

## 3. Known limitations

- **Development data:** 2020–2025 was used in S02–S06, and S07 was designed after seeing S05a's direction-vs-timing result.
- **Costs:** quoted half-spreads, no market impact. 100% gross spread across about 500 names keeps orders small.
- **Hedge:** sector ETF only, with β frozen per day. The ETF hedge pays its own measured spread.
- **Overnight:** positions are held overnight (earnings gaps included). Shorting is free and unconstrained. The universe is the current snapshot.

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/Residual_Reversion/07_continuous_weights.ipynb
```

The panel and quote samples come from the shared caches, so nothing is downloaded. There are 8 backtests (a few minutes), then 2 × 500 placebos. At G2's α of 0.025, `N_PLACEBO` must be at least 40 for a pass to be possible at all; 500 is the pre-registered number.
