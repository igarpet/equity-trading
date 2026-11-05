# News Filtered Reversal — paper P4

**Paper.** Da, Z., Liu, Q. & Schaumburg, E. (2014). A Closer Look at the Short-Term Return Reversal. *Management Science* 60(3), 658–674. NY Fed SR 513.

**Claim to test.** Reversal is much stronger once industry returns and cash-flow news are removed from the past return.

**Our question.** Does removing news-driven moves (earnings 8-K windows from SEC EDGAR; Alpaca / Benzinga news shocks as a secondary arm) leave a stronger reversal?

| Notebook | Description | Status |
|---|---|---|
| `01_daily_news_filtered_reversal.ipynb` | Daily quintile reversal: raw (R), industry-adjusted (I), I without earnings-8-K windows (IN); secondary Alpaca news-shock arms (IA, INA); k = 21 / 5; both windows; 200 random-exclusion placebos | Built; awaiting run |
| `02_intraday_earnings_mask.ipynb` | S05 residual (`x10`) with no entries inside earnings windows; both windows; 100 random-exclusion placebos each | Built; awaiting run (after 01) |

Protocol: [`001_claude_p4_news_filtered_reversal_preregistration.md`](../../discussions/News_Filtered_Reversal/001_claude_p4_news_filtered_reversal_preregistration.md). Both windows (2017–2019, 2020–2025) are judged.

Run: set `SEC_USER_AGENT="Your Name you@example.com"` in `.env`; `python -m pytest -q tests/test_p4.py`; notebook 01 (EDGAR: minutes; Alpaca news: resumable, possibly hours — `RUN_NEWS = False` runs the primary arms first); notebook 02 (~2–4 h). `P4_SYNTHETIC=1` runs both offline. ForexFactory and FinancialJuice were checked and not used (protocol section 3). Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
