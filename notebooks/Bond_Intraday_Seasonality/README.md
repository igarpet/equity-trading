# Bond Intraday Seasonality — stacked 1-minute days (BDS 001)

**Question.** The FX stacked-day scan applied to Dukascopy's three government-bond CFDs: is there an interval whose mean return differs from zero, consistently across years, by more than a futures trade costs, and does it hold in the other half of the sample?

| Symbol | Bond | Matching future (tick) |
|---|---|---|
| `USTBONDTRUSD` | US T-Bond | ZB (1/32) |
| `BUNDTREUR` | Euro Bund | FGBL (0.01) |
| `UKGILTTRGBP` | UK Long Gilt | ICE Long Gilt (0.01) |

One folder per symbol, four notebooks identical across symbols except for the symbol:

| Notebook | Window | Description | Status |
|---|---|---|---|
| `01_discovery_H1` | 2018–2021 | download check, contract switches found and removed, stacked day, activity and spread, 15-minute blocks, 2,704-interval scan (D1 BH, D2 years, D3 futures cost), robustness, frozen candidates, placebos | Code ready, not run |
| `02_discovery_H2` | 2022–2025 | same | Code ready, not run |
| `03_cross_confirmation` | — | H1 candidates on H2 (forward), H2 candidates on H1 (backward), gates K1–K4; structures found in both halves | Run after `01/` and `02/` are committed |
| `04_full_window` | 2018–2025 | same scan on everything (in-sample), each candidate matched to the H1/H2 candidates and their verdicts, H1 vs H2 stacked day | Run last |

Protocol: [`001_claude_bds01_preregistration.md`](../../discussions/Bond_Intraday_Seasonality/001_claude_bds01_preregistration.md).

**Differences from FX.** Quarterly contract switches (the CFD jumps to the next future's price at the daily break) are detected and the days containing them dropped; a day counts if it has at least half the window's median quoted minutes; days spanning a missing file are dropped; costs are one futures tick + 0.70 bp (Dukascopy's CFD spread, about 0.05 points, is reported alongside); D2/K4 use the share of years with ≥ 120 days (75% / 60%).

**Data.** `tools/dukascopy/dukascopy_bond_candles.py`. Discovery needs ≥ 95% of weekday files from the symbol's first file.

**Code.** `src/equity_trading/bond_seasonality.py` (on top of `fx_seasonality.py`), tests in `tests/test_bond_seasonality.py`. Nothing reads 2026.

**Smoke run.** `BDS_SYNTHETIC=1` (with the same `BDS_SYNTH_DIR` for all notebooks) runs everything offline: UST must return one candidate, 08:30–09:30 New York long, confirmed forward and backward and matched by the FULL candidate; Bund and Gilt must return nothing, with every injected contract switch detected.

**Next, depending on results:** bond–FX pairs (e.g. Bund vs EURUSD, Gilt vs GBPUSD, T-Bond vs USDJPY), as a separate study.
