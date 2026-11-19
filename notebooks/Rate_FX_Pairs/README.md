# Rate–FX pairs on 5-minute bars (RFX 001)

**Question.** BFX found FX following long-bond moves by 5–10 minutes, but only about 5% of the bond-implied move, below costs. Long bonds carry term premium and inflation risk; FX moves with expected policy rates. Do the **short-rate** futures do better? Same three methods as BFX, against the USD side of EURUSD, GBPUSD and USDJPY.

| Study folder | Futures (Databento `bbo-1m`) | H1 | H2 | FULL |
|---|---|---|---|---|
| `Notes/` | ZT (2-year), ZF (5-year) | 2016–2020 | 2021–2025 | 2016–2025 |
| `SOFR/` | SR3 (3-month SOFR, the T-bill end) | 2022–2023 | 2024–2025 | 2022–2025 |

Each study folder: `00_data` (H1, descriptive), and `LeadLag/`, `Residual/`, `Coint/`, each with `01_discovery_H1`, `02_discovery_H2`, `03_cross_confirmation`, `04_full_window`. Status: run (`bc42324`), see Result.

Protocol: [`001_claude_rfx01_preregistration.md`](../../discussions/Rate_FX_Pairs/001_claude_rfx01_preregistration.md). Methods, filters and gates: [`BFX 001`](../../discussions/Bond_FX_Pairs/001_claude_bfx01_preregistration.md).

**Data.** `python tools/databento/rates_download.py --dry-run`, then without `--dry-run` (about $12; shared budget ledger with the futures code; resumable; 2026 refused). Files: `<EQUITY_TRADING_DIR>/Databento/bbo-1m/<ROOT>/<YYYY>/<ROOT>_<date>.parquet`.

**Code.** `src/equity_trading/rates.py` (download, normalise, load, roll detection), `src/equity_trading/bond_fx.py` 0.3.0 (studies `RFX`, `SOFR`). Tests: `tests/test_rates.py`, `tests/test_bond_fx.py`.

**Smoke run.** `RFX_SYNTHETIC=1` (one `RFX_SYNTH_DIR` for all notebooks): Notes must find USDJPY lead–lag on ZT (and the GBPUSD / EURUSD relations under residual and cointegration); SOFR must find nothing.

## Result (run `bc42324`)

No candidate in any method, half or study.

- **Notes (ZT + ZF, 2016–2025).** FX follows rate moves for 5–60 minutes: predictive t up to 5.8 (USDJPY, 2016–2020) and 3–4 for EURUSD and GBPUSD in 2021–2025, but the follow-through is 3–10% of the rate-implied move, about the same as with the long bonds (BFX). Per trade that is 0.2–0.7 bp at the mid against 1.1–1.4 bp of FX costs; every configuration loses net. Residual reversion: few valid signals, nothing near significance. Cointegration: at most 20 trades per configuration.
- **SOFR (SR3.v.0, 2022–2025): not interpretable.** The volume-rolled continuous symbol changed contract 217 times in 2022–2023 (the most traded SOFR contract is usually not the front one), so the series hops between maturities. Replaced by `SR3.c.3` in RIS 001; not rerun here.
- **Data note.** ZT's mid is unchanged in about 74% of 5-minute returns and ZF's in about 42%: moves smaller than one tick, not stale quotes.
