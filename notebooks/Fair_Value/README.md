# Fair value — automated DCF on US stocks from free EDGAR data (FV)

**Question.** Does a fair value computed automatically from point-in-time SEC filings predict long-horizon returns? If it does, is a large gap (price ≥ 20% below fair value) tradable in the stock or in options?

The fair value is updated at every filing. The window is 2009–2025, because EDGAR XBRL starts in 2009; 2026 is sealed.

Protocols: FV 01 [`001`](../../discussions/Fair_Value/001_claude_fv01_data_plan.md).

| Notebook | Content | Output |
|---|---|---|
| `01_data_coverage` | FV 01: filers and exits per year, DCF-item tag coverage, restatement share, price-series match per CIK (live and dead, Yahoo vs Alpaca), known-delisting spot check | `01/` |

**Data** (free; all under `EQUITY_TRADING_DIR`):
- `EDGAR/`: `companyfacts.zip` → `facts/part-*.parquet`; `submissions.zip` → `companies.parquet` and `filings.parquet`; FSDS `sub.txt` → `fsds_sub.parquet`; `ticker_history.parquet`.
- `Yahoo/daily/<SYMBOL>.parquet`: unadjusted, with dividends and splits, plus `_status.csv`.
- `Alpaca/bars/1Day/`: from 2016.

**Code.** `src/equity_trading/edgar.py`, `src/equity_trading/daily_prices.py`; tests in `tests/test_edgar.py`.

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
