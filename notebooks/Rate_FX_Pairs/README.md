# Rate–FX pairs on 5-minute bars (RFX 001)

**Question.** BFX found FX following long-bond moves by 5–10 minutes, but only about 5% of the bond-implied move, below costs. Long bonds carry term premium and inflation risk; FX moves with expected policy rates. Do the **short-rate** futures do better? Same three methods as BFX, against the USD side of EURUSD, GBPUSD and USDJPY.

| Study folder | Futures (Databento `bbo-1m`) | H1 | H2 | FULL |
|---|---|---|---|---|
| `Notes/` | ZT (2-year), ZF (5-year) | 2016–2020 | 2021–2025 | 2016–2025 |
| `SOFR/` | SR3 (3-month SOFR, the T-bill end) | 2022–2023 | 2024–2025 | 2022–2025 |

Each study folder: `00_data` (H1, descriptive), and `LeadLag/`, `Residual/`, `Coint/`, each with `01_discovery_H1`, `02_discovery_H2`, `03_cross_confirmation`, `04_full_window`. Status: code ready, not run; data not yet downloaded.

Protocol: [`001_claude_rfx01_preregistration.md`](../../discussions/Rate_FX_Pairs/001_claude_rfx01_preregistration.md). Methods, filters and gates: [`BFX 001`](../../discussions/Bond_FX_Pairs/001_claude_bfx01_preregistration.md).

**Data.** `python tools/databento/rates_download.py --dry-run`, then without `--dry-run` (about $12; shared budget ledger with the futures code; resumable; 2026 refused). Files: `<EQUITY_TRADING_DIR>/Databento/bbo-1m/<ROOT>/<YYYY>/<ROOT>_<date>.parquet`.

**Code.** `src/equity_trading/rates.py` (download, normalise, load, roll detection), `src/equity_trading/bond_fx.py` 0.3.0 (studies `RFX`, `SOFR`). Tests: `tests/test_rates.py`, `tests/test_bond_fx.py`.

**Smoke run.** `RFX_SYNTHETIC=1` (one `RFX_SYNTH_DIR` for all notebooks): Notes must find USDJPY lead–lag on ZT (and the GBPUSD / EURUSD relations under residual and cointegration); SOFR must find nothing.
