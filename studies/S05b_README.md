# S05b: S05a with measured transaction costs

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-03, written after S05a's result (35b2dc3) and before any quote data was downloaded |
| **Notebook** | `studies/S05b_measured_costs.ipynb` |
| **Engine** | `src/equity_trading/s05.py` (unchanged rules; per-bar, per-symbol cost array added), `src/equity_trading/s05b.py`, `alpaca.fetch_quotes` / `sample_quote_spreads` |
| **Data** | As S05a, plus Alpaca SIP NBBO quote samples. 2020–2025 only. |

## 0. Why

S05a captured a gross signal: +37.9% over 2020–2025, Sharpe 0.93, and the random-direction test was significant. After an assumed flat 2 bps per fill it lost money (−7.8%). The breakeven cost was **1.56 bps per fill**.

The verdict therefore depends on a number that was assumed, not measured. S05b measures it and changes nothing else.

## 1. Cost measurement (fixed before sampling)

| Step | Rule |
|---|---|
| Sessions | 4 NYSE sessions per year, 2020–2025 (24 total), drawn from the exchange calendar with seed 42. No price, return or spread data is used to choose them. |
| Moments | Every 15m bar start of each sampled session (26, or 14 on early-close days). This is when S05 orders fill. |
| Quotes | Alpaca SIP NBBO quotes in the first **3 seconds** after each bar start, for every traded stock and sector ETF. Valid quote: 0 < bid < ask, and relative spread < 2%. |
| Sample value | Median of (ask − bid)/(ask + bid) over the valid quotes in the window, i.e. the quoted half-spread as a fraction of mid. |
| Cost model | half_spread(symbol, year, slot) = base(symbol, year) × profile(slot). `base` = the symbol's median over that year's samples. `profile` = the cross-sectional median of sample / base by slot (the intraday U-shape). Fallbacks: the symbol's all-year median, then that year's cross-sectional median. No floor. |
| Applied | Every fill pays its symbol's half-spread for that year and bar slot. ETF hedges are netted as in S05a. |

A market order pays roughly half the quoted spread. Price improvement would lower the true cost and market impact would raise it; at about $1,000 per fill in S&P 500 names, impact is negligible.

## 2. Strategy and evaluation

Strategy, parameters, placebos, gates and verdicts are **identical to S05a** (`studies/S05a_README.md`). The only differences:
- every run uses the measured cost array;
- G4 runs at **2× the measured costs**.

**Reported in addition:**
- cost distribution by symbol and year, and the intraday profile;
- realized cost per trade in bps;
- **breakeven cost multiple** = gross P&L / measured cost (above 1 means profitable before the 2× stress).

## 3. Known limitations

- **Sampling:** 24 sessions × 26 moments is a sample. Spreads on unsampled days can differ, especially on stressed days.
- **Quoted ≠ effective spread:** no price improvement, no hidden liquidity, no queue position.
- **Development data:** the result remains development evidence on data already used (S02–S05a). Only the cost input changed, and it was measured without reference to returns.
- **Other S05a limitations still apply:** free shorting, survivorship, overnight holding.

## Run

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 studies/S05b_measured_costs.ipynb
```

**Quote download:** at least about 24 × 26 × 6 ≈ 3,700 requests, plus extra pages for active symbols. That is roughly 30–60 min at the free-plan rate limit. Each session is cached in `<EQUITY_TRADING_DIR>/Alpaca/quote_samples/` once complete, so an interrupted run resumes.

**If Alpaca refuses the quotes endpoint on your plan,** the cell stops with the API's error message.

**Placebos:** the same cost as in S05a.

## Correction C1 (2026-11-03): panel field-order bug, all earlier real-data results void

- **The bug:** `s04.load_panel` passed bar fields to the panel builder as (open, high, low, close), but the builder expected (open, close, high, low). On real data the panel's close was the bar **high**, its high was the **low**, and its low was the **close**. Opens were correct. The synthetic panel was built correctly, so tests did not catch it.
- **Scope:** every real-data run of S04, S05a, S05b and S06 before this correction (fixed in the commit that adds this note). Signals, estimates and marks used bar highs. The S04 Abdi–Ranaldo estimate (36 bps, the reason for amendment A1) was computed from scrambled fields. S02 and S03 used a different, correct loader and are not affected. The S05b quote samples are not affected.
- **Fix:** fields passed by name. A regression test checks the OHLC fields stay distinct. One shared panel cache is now keyed on `PANEL_VERSION` 0.2.0, and every derived cache is keyed on it too, so no stale panel or result can be reused.
- **Status:** the pre-registered rules are unchanged; only the data were wrong. The earlier results in `results/` stay in git history and must not be cited. Re-runs replace them.
