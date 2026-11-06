# Futures Order Flow — CME index futures for Lucid prop accounts

**Question.** Does signed taker flow in ES/NQ predict the next few minutes well enough to pay a micro-contract round trip (1 tick + commission, about 0.3–0.5 bp of notional)? The design reference is deribit-market-data Candle_Streaks CS03c: on crypto perps the effect was statistically real but about 20× smaller than crypto costs. This branch re-specifies it for CME and inherits no thresholds or conclusions.

| Notebook | Description | Status |
|---|---|---|
| `01_data_cost_and_coverage.ipynb` | Databento prices and budget-capped download (2023–2025 `tbbo`), NinjaTrader export and recorder ingestion, data checks C1–C5; holdouts as day counts only | Code ready; run 1 with `DOWNLOAD = False` (nothing billed) |
| `02_…` | Strategy pre-registration FOF 002 | After C1–C5 pass |

Protocol: [`001_claude_fof01_data_plan.md`](../../discussions/Futures_Order_Flow/001_claude_fof01_data_plan.md).

**Windows.** DEV 2023–2024, TEST 2025 (Databento). Holdout A: NinjaTrader export from 2026-01-01 (sealed). Holdout B: forward recording. Nov–Dec 2025 is the bridge where NinjaTrader and Databento overlap; it is used only for data checks.

**Setup.**
- `pip install -r requirements.txt` (adds `databento`).
- `DATABENTO_API_KEY=` in `.env`.
- NinjaTrader exports go in `<EQUITY_TRADING_DIR>/NinjaTrader/export/<ROOT>/`, recorder files in `<EQUITY_TRADING_DIR>/NinjaTrader/recorder/`. See [`tools/ninjatrader/README.md`](../../tools/ninjatrader/README.md).

**Code.** `src/equity_trading/futures.py`, tests in `tests/test_futures.py`. The holdout guard and the $115 budget cap are enforced in the module, not only in the notebook.

**Smoke run.** `FOF_SYNTHETIC=1` runs the notebook offline in a temporary folder. C1 fails there by design: stitching drops the first synthetic day, and the check reports it.
