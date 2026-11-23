# FV 001 — Fundamentals and prices for the fair-value research: free data plan and coverage check (FV 01)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | Fair_Value |
| **Responds to** | Iván, 2026-11-22: move from short-term signals to long-term fair value. Automated DCF on stocks, fair value updated at every filing, trade when price is ≥ 20% below fair value (stock, and synthetic options to test feasibility). Own the pipeline, free data only: SEC EDGAR for fundamentals, Yahoo for prices. No index-constituent universe: a broad universe filtered by enterprise value, excluding the mega caps. |
| **Engine** | `edgar.py` 0.1.2, `daily_prices.py` 0.1.0 |
| **Status** | Data plan and coverage check. No signal, no returns. |

## 1. Window

EDGAR XBRL facts start in 2009 (phase-in; smaller filers from mid-2011). The branch therefore uses **2009–2025**, an exception to the 2020–2025 rule of the other branches. **2026 stays sealed** as the holdout. EDGAR bulk files contain 2026 filings, so every loader cuts at `filed <= 2025-12-31` by default (`edgar.load_facts`). Prices stop at 2025-12-31.

## 2. Fundamentals: SEC EDGAR (free, official)

| File | Use |
|---|---|
| `companyfacts.zip` | Every XBRL numeric fact (us-gaap, dei, ifrs-full) from 10-K, 10-Q, 20-F, 40-F and their amendments. Each fact carries the filing that reported it (`accn`, `form`, `filed`). |
| `submissions.zip` | Per-CIK metadata (SIC, current tickers, former names) and the filing list with **acceptance timestamps**. It also includes 8-K filings (events), delisting and deregistration forms (25, 15) and merger forms (DEFM14A, SC TO-T, S-4) for deal prices. |
| FSDS `sub.txt` 2009q1 → | The XBRL instance file name (`aapl-20130928.xml`) has a ticker-like prefix. This is the only free source of historical symbols for delisted CIKs. |

**Point in time.** A restated value comes back as a new row under the later filing. The facts table keeps every row, so the state of knowledge at date t is `filed <= t`. Signals are timed from the acceptance timestamp, with trading at the next day's open at the earliest (FV 02+). Amendments (10-K/A, 10-Q/A) are new information on their own filing dates.

**Survivorship.** Facts are keyed by CIK, so dead companies keep their filings.

## 3. Prices: Yahoo + Alpaca (free)

**Yahoo.** Yahoo keeps live symbols only. Delisted firms usually disappear, and a reused symbol returns the new company's history without warning. Yahoo data is stored **unadjusted**, with dividends and splits, so total returns are computed by us.

**Alpaca.** Alpaca 1Day SIP bars (2016–2025, split and dividend adjusted) are downloaded for the same symbols as a second source for firms delisted since 2016.

**Symbols.** The symbol list is every candidate symbol of every CIK with a periodic filing since 2009: FSDS prefixes, inline-XBRL primary document prefixes and current SEC tickers.

**No size filter at download.** Enterprise value is applied point in time when the signal is formed. Filtering on today's EV would select on the outcome.

## 4. Coverage check (`notebooks/Fair_Value/01_data_coverage.ipynb`)

1. Periodic filers per year; CIKs that stop filing; delisting and deregistration forms per year; SPAC (SIC 6770) count.
2. Share of 10-K filers reporting each DCF item per year (revenue, operating income, net income, CFO, capex, D&A, cash, debt, equity, shares). Share of facts later restated.
3. For each CIK, does a symbol candidate's price series cover ≥ 80% of the period it filed in?
   - Each CIK is classed as match / partial / late / missing, for active and dead CIKs, for Yahoo and Alpaca.
   - **late** means the series starts more than a year after the filing window: a reused symbol.
   - Dead-CIK match rate is reported by exit year.
4. Spot check: SIVB, FRC, BBBY, ATVI, TWTR, CELG, AAPL.

**What the readout decides:**
- Which delisted CIKs enter FV 02 with real prices.
- Which need the delisting-return approximation:
  - acquisitions: deal price from DEFM14A / SC TO-T;
  - performance delistings: −30% (Shumway 1997), stress-tested at −100%.
- Whether 2009–2015 exits need another source.

## 5. Next (not in this note)

**FV 02, panel builder.**
- Tag mapping per DCF item.
- Flows: Q4 = 10-K annual figure − 9-month YTD; quarterly cash flows by differencing YTD values; TTM sums.
- Point-in-time panel keyed by acceptance time, plus EV and market cap from prices × shares.

**FV 03, fair value and the gap.**
- Reverse DCF / residual income.
- Half-life of log(P/V).

**FV 04, trading tests.**
- Stock only, plus synthetic BSM options (6, 12 and 24 months; IV = k × RV, with k = 1.0 / 1.2 / 1.4; spread costs).
- Entry threshold grid 10/20/30/40%.
- Baselines: random stocks, the most overvalued quintile, and stock vs options.
