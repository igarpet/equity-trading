# Fair value — automated DCF on US stocks from free EDGAR data (FV)

**Question.** Does a fair value computed automatically from point-in-time SEC filings predict long-horizon returns? If it does, is a large gap (price ≥ 20% below fair value) tradable in the stock or in options?

The fair value is updated at every filing. The window is 2009–2025, because EDGAR XBRL starts in 2009; 2026 is sealed.

Protocols: FV 01 [`001`](../../discussions/Fair_Value/001_claude_fv01_data_plan.md); FV 02 [`002`](../../discussions/Fair_Value/002_claude_fv02_panel.md).

Window: signals and returns **2016–2025**, because dead companies have prices almost only from 2016 (FV 01 readout). 2009–2015 is DCF history only.

| Notebook | Content | Output |
|---|---|---|
| `01_data_coverage` | FV 01: filers and exits per year, DCF-item tag coverage, restatement share, price-series match per CIK (live and dead, Yahoo vs Alpaca), known-delisting spot check. Run 767f00f: see `002` §1 | `01/` |
| `02_panel` | FV 02: point-in-time filing snapshots (TTM, 5-year history, balance sheet, shares), Alpaca month-end raw/adjusted prices, symbol→CIK windows, monthly panel 2016–2025 with market cap and EV, exits; checks A–F (Apple, universe by EV bucket, item coverage, Form 25 coverage, mcap/float, exits) | `02/` |

**Data** (free; all under `EQUITY_TRADING_DIR`):
- `EDGAR/`: `companyfacts.zip` → `facts/part-*.parquet`; `submissions.zip` → `companies.parquet` and `filings.parquet`; FSDS `sub.txt` → `fsds_sub.parquet`; `ticker_history.parquet`.
- `Yahoo/daily/<SYMBOL>.parquet`: unadjusted, with dividends and splits, plus `_status.csv`.
- `Alpaca/bars/1Day/`: from 2016.

**Code.** `src/equity_trading/edgar.py`, `src/equity_trading/daily_prices.py`, `src/equity_trading/fv_panel.py`; tests in `tests/test_edgar.py`, `tests/test_fv_panel.py`.

**Run order.**
1. Add `SEC_USER_AGENT="Your Name you@mail.com"` to `.env`. The SEC requires a name and an e-mail address.
2. `python -m pytest -q tests/test_edgar.py`
3. `python tools/edgar/edgar_download.py`
   - Roughly 3–5 GB of downloads, then parsing. The facts step is the slow part: expect 30–90 minutes on the research machine.
   - It is resumable: re-run after any interruption.
4. `python tools/prices/daily_download.py --source yahoo`
   - Several thousand symbols in batches of 50.
   - If Yahoo throttles, stop and re-run; finished symbols are skipped.
5. `python tools/prices/daily_download.py --source alpaca`
6. Run `01_data_coverage`.
7. `python tools/prices/daily_download.py --source alpaca --adjustment raw` (unadjusted bars for market cap, about 30 min).
8. Run `02_panel`. The first run builds the snapshots, roughly 20–60 min with 3 workers, cached in `EquityTrading/fair_value/`.
