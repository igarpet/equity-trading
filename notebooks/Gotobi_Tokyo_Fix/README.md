# Gotobi Tokyo Fix — USD/JPY on Japanese settlement days

**Question.** Does USD/JPY rise into the 09:55 JST Tokyo fix on gotobi days (Ito & Yamada 2017), and does the YouTube rule — sell at 09:55 JST, 12-pip stop, buy back at 10:30 JST — pay after Dukascopy spread and commission? The video claims +2.35 pips per trade after spread on 2016–2026. Here it is replicated on 2016–2025 and tested on 2010–2015, which the video did not use.

| Notebook | Description | Status |
|---|---|---|
| `01_gotobi_fix_backtest.ipynb` | Calendar, per-day tick outcomes, mid path, H1 (pre-fix), H2 gates G1–G3 per window, by-year, robustness grid, costs, equity, verdict | Run `610d790`: REPLICATES 2016–2025, NOT_CONFIRMED 2010–2015 → `NO_EDGE`. Correction C1 (engine 0.2.0): rerun pending |
| `02_long_into_fix_reversal.ipynb` | GTF 02: long 09:25→09:55 JST, reverse into the GTF 01 short. Exploratory gates on 2010–2015 / 2016–2025, long-leg check L, entry-time grid, costs, equity; binding 2026 holdout behind `OPEN_HOLDOUT` | Run `ab27207`: G1–G3 pass on 2016–2025 (exploratory), G2/G3 fail on 2010–2015; long leg not gotobi-specific (diff +0.62, p = 0.12) → `PENDING_HOLDOUT` |
| `03_small_account_leverage.ipynb` | Sizing only (no gates): $180 account at 10–500:1, Kelly leverage, equity curves with broker stop-out, losses-to-wipe, one-year resampling | Code ready, not run |

Protocols: [`001`](../../discussions/Gotobi_Tokyo_Fix/001_claude_gtf01_preregistration.md) (GTF 01), [`002`](../../discussions/Gotobi_Tokyo_Fix/002_claude_gtf01_correction_c1.md) (correction C1), [`003`](../../discussions/Gotobi_Tokyo_Fix/003_claude_gtf02_preregistration.md) (GTF 02).

**Windows.** GTF 01: INDEPENDENT 2010–2015 (binding), REPLICATION 2016–2025. GTF 02: both exploratory; the 2026 holdout is binding (one look, `allow_holdout=True` only in notebook 02's holdout section).

**Setup.**
- `pip install -r requirements.txt` (adds `jpholiday`).
- Ticks from `tools/dukascopy/dukascopy_download.py`. The loader looks in `<EQUITY_TRADING_DIR>/Dukascopy/USDJPY/hours_*` (any folder with hours 00–01 UTC); set `DUKASCOPY_DIR` if the downloader's `--out` differs.
- `python -m pytest -q tests/test_gotobi.py`

**Code.** `src/equity_trading/gotobi.py`, tests in `tests/test_gotobi.py`. Fills are on the quoted side (sell bid, buy ask); a fill more than 30 s late voids the day.

**Smoke run.** `GTF_SYNTHETIC=1` runs the notebook offline on synthetic ticks with an injected effect; it must return `EDGE_CANDIDATE`. Outputs go to `01_synthetic/` (ignored by git).
