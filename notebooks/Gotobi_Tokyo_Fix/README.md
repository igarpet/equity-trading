# Gotobi Tokyo Fix — USD/JPY on Japanese settlement days

**Question.** Does USD/JPY rise into the 09:55 JST Tokyo fix on gotobi days (Ito & Yamada 2017), and does the YouTube rule — sell at 09:55 JST, 12-pip stop, buy back at 10:30 JST — pay after Dukascopy spread and commission? The video claims +2.35 pips per trade after spread on 2016–2026. Here it is replicated on 2016–2025 and tested on 2010–2015, which the video did not use.

| Notebook | Description | Status |
|---|---|---|
| `01_gotobi_fix_backtest.ipynb` | Calendar, per-day tick outcomes, mid path, H1 (pre-fix), H2 gates G1–G3 per window, by-year, robustness grid, costs, equity, verdict | Code ready, not run on real data |

Protocol: [`001_claude_gtf01_preregistration.md`](../../discussions/Gotobi_Tokyo_Fix/001_claude_gtf01_preregistration.md).

**Windows.** INDEPENDENT 2010–2015 (binding), REPLICATION 2016–2025. 2026 sealed (refused in code).

**Setup.**
- `pip install -r requirements.txt` (adds `jpholiday`).
- Ticks from `tools/dukascopy/dukascopy_download.py`. The loader looks in `<EQUITY_TRADING_DIR>/Dukascopy/USDJPY/hours_*` (any folder with hours 00–01 UTC); set `DUKASCOPY_DIR` if the downloader's `--out` differs.
- `python -m pytest -q tests/test_gotobi.py`

**Code.** `src/equity_trading/gotobi.py`, tests in `tests/test_gotobi.py`. Fills are on the quoted side (sell bid, buy ask); a fill more than 30 s late voids the day.

**Smoke run.** `GTF_SYNTHETIC=1` runs the notebook offline on synthetic ticks with an injected effect; it must return `EDGE_CANDIDATE`. Outputs go to `01_synthetic/` (ignored by git).
