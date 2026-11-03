# S06: S05a residual signal with passive (limit-order) stock execution

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-03, written after S05b (825b58d) and before any S06 result |
| **Notebook** | `studies/S06_passive_execution.ipynb` |
| **Engine** | `src/equity_trading/s06.py` (tests in `tests/test_s05.py`) |
| **Data** | As S05b: Alpaca SIP 15Min OHLC 2020–2025, plus the S05b quote samples (reused from cache). 2026 not used. |

## 0. Why

S05a/S05b established three things about the sector-ETF residual signal:
- the contrarian direction carries information (random-direction p ≈ 0.0002);
- it earns about **5.7 bps gross per trade**;
- market orders pay about **12.7 bps per trade**, so it loses heavily (−34.6%).

Passive execution flips the spread from a cost to a credit. The question is whether that credit survives **adverse selection**: a limit order fills when price moves through it, which tends to happen when price is moving against the position.

## 1. Signal (unchanged)

The S05a signal, parameters and position limits are unchanged: sector-ETF residual, 20-session window, s-score thresholds ±1.25, exits −0.50 / +0.75, 1% of equity per stock, at most 100 positions. Costs are the S05b measured half-spreads by symbol, year and intraday slot.

## 2. Execution model

| Element | Rule |
|---|---|
| Stock orders | Limit orders at the touch implied by the bar open as mid: buy L = o(1 − hs), sell L = o(1 + hs). hs = measured half-spread for that symbol, year and slot. |
| Fill rule (primary) | **Trade-through by one quoted spread:** a buy fills if the bar low ≤ L − 2·hs·o, a sell if the high ≥ L + 2·hs·o. A quoted spread is at least one tick, so this requires trading at least one tick through the limit price. This is a standard conservative stand-in for queue position. All-or-nothing fills; bars without trades never fill. |
| Entry orders | Live for one bar. A stock still signalling at the next close is re-posted. Slots: open positions count against the 100-position cap; ranking by largest \|s\|, as in S05a. |
| Exit orders | Re-posted every bar. After **4** unfilled bars, a market exit at the next open (pays hs). |
| ETF hedge | Market order, netted per ETF, paying the ETF's measured hs. The **hedge price inside the fill bar is unobservable on 15m bars**, so it is bracketed: **primary = ETF bar mid (o + c)/2**, optimistic = ETF bar open, conservative = next bar's ETF open. |
| Fees and rebates | None. Alpaca charges no commission and pays no maker rebate. SEC/FINRA fees are ignored (< 0.03 bps). |
| Accounting | final − initial = signal P&L (all legs at reference prices) + spread captured on passive fills − market-order costs. Verified to machine precision in tests. |

## 3. Evaluation (pre-registered)

**Runs:**
- **primary:** mid hedge, trade-through, full capture, 1× market costs;
- **stress:** half capture, 2× market costs;
- **optimistic:** hedge at the ETF bar open;
- **conservative:** hedge at the next ETF open;
- **touch:** fill when the bar low touches the limit;
- **gross:** 0× market costs.

**Placebos (on primary):**
- **random direction:** the signal P&L of the primary trades with a random sign per trade, 10,000 draws;
- **random timing:** 1,000 backtests with contrarian entry orders posted at random flat tradable bars, at the strategy's entry rate, through the same execution model.

**Gates:**

| Gate | Condition |
|---|---|
| **G1 bootstrap** | Block-bootstrap 95% CI of mean daily net return > 0 (blocks of 10, 20 and 40 days), primary |
| **G2 random direction** | p ≤ 0.05 |
| **G3 random timing** | Total return and Sharpe each have p ≤ 0.05 |
| **G4 stress** | Stress run total return > 0 and Sharpe > 0 |

**Verdicts:**
- **Statistical:** `SIGNAL` if G1, G2 and G3 all pass; `PARTIAL` if some pass; `NO_SIGNAL` if none.
- **Economic:** `PROFITABLE_AFTER_COSTS` if primary net return > 0 and G4 passes.
- **Overall:** `EDGE_CANDIDATE` if both hold.
- **`EXECUTION_ROBUST`:** reported flag; the conservative run is profitable (return > 0 and Sharpe > 0).

**Interpretation, fixed in advance:**

| Outcome | Reading |
|---|---|
| `EDGE_CANDIDATE` and `EXECUTION_ROBUST` | Worth paper trading |
| `EDGE_CANDIDATE` without `EXECUTION_ROBUST` | The result depends on hedge timing within 15 minutes and cannot be settled with 15m bars. It needs 1-minute bars around fills, or paper trading, before any conclusion. |
| Optimistic bound unprofitable | Passive execution does not rescue this signal |

**Reported:** fill rate, passive-exit share, capture and market cost per trade, signal P&L per trade, results by year, the bracket of equity curves, and the S05b market-order reference.

**Synthetic check (engine only):** on the synthetic panel, hedge timing alone moves the result from +22% (open) through 0% (mid) to −17% (next open). The sector factor moves far more per bar than the spread, so hedge timing dominates. Synthetic intrabar paths are not realistic; this only shows that the bracket matters.

## 4. Known limitations

- **No intrabar path:** the fill decision uses only the bar's high and low. Fill timing inside the bar, and therefore the hedge price, is bracketed rather than known.
- **Bar open as mid:** the open is a trade price, at the bid or the ask. This adds noise to L and to the captured spread.
- **Queue position and partial fills** are not modelled beyond the trade-through rule. Orders of about $1,000 make partial fills unlikely.
- **Other S05 limitations still apply:** free shorting, current-snapshot universe, overnight holding, and development data (2020–2025 used in S02–S05b).

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 studies/S06_passive_execution.ipynb
```

**First run:** rebuilds the panel with raw high/low from the cached bars (about 4 min, new cache file). Quote samples come from the S05b cache, so nothing is downloaded.

**Compute:** six backtests take a few minutes. Placebos take longer per run than in S05b, because each fill is evaluated individually; lower `N_PLACEBO` for a first look.
