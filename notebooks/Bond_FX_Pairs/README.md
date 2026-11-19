# Bond–FX pairs on 5-minute bars (BFX 001)

**Question.** Does the bond market of a currency pair's two countries carry tradable intraday information about the pair? Three approaches, same data and windows:

| Folder | Method | Legs traded | Configurations |
|---|---|---|---|
| `LeadLag/` | after a large bond-implied FX move over the last L bars, hold FX in that direction for H bars | FX | 24 |
| `Residual/` | Avellaneda–Lee: rolling regression of FX returns on bond returns, cumulated residual, AR(1) s-score, fade extremes | FX | 12 |
| `Coint/` | daily Engle–Granger regression of log FX on log bonds over 20/60 days, traded only if cointegrated (p < 0.05), fade the spread | FX + bonds | 6 |

| Pairing | Bonds (Dukascopy CFDs) |
|---|---|
| EURUSD | Bund, US T-Bond |
| GBPUSD | Gilt, US T-Bond |
| USDJPY | US T-Bond |
| EURGBP | Bund, Gilt |

Each method folder:

| Notebook | Window | Status |
|---|---|---|
| `01_discovery_H1` | 2020–2022 | Run `86ccb72`: no candidates in any method |
| `02_discovery_H2` | 2023–2025 | Run `86ccb72`: no candidates in any method |
| `03_cross_confirmation` | H1 → H2 (forward), H2 → H1 (backward), gates K1–K4 | Nothing to confirm |
| `04_full_window` | 2020–2025, matched to the halves | Run last |

`00_data.ipynb` (descriptive, H1 only; adds month-by-month data quality): bars per pairing and year, correlation of FX and bond returns by hour, rolling hedge ratios.

Protocol: [`001_claude_bfx01_preregistration.md`](../../discussions/Bond_FX_Pairs/001_claude_bfx01_preregistration.md); windows re-split to 2020–2022 / 2023–2025 by [`002_claude_bfx02_amendment_windows.md`](../../discussions/Bond_FX_Pairs/002_claude_bfx02_amendment_windows.md) after the 00 run showed the bond quotes missing or stale before 2020 (run `c8bfb8a`, outputs replaced by the rerun).

**Data.** 5-minute bars where every leg was quoted; bonds made continuous across contract switches (BDS 002 rule); spike days and missing-file days excluded. FX costs at Dukascopy's quoted bid/ask + 0.70 bp; bond legs one futures tick + 0.70 bp per unit of hedge ratio.

**Code.** `src/equity_trading/bond_fx.py`, tests in `tests/test_bond_fx.py`. Nothing reads 2026.

**Smoke run.** `BFX_SYNTHETIC=1` with one `BFX_SYNTH_DIR` for all notebooks: USDJPY lead–lag, GBPUSD residual reversion and EURUSD cointegration are injected, EURGBP is independent. Each method must find its injected pairing (and may find another pairing whose synthetic relation also shows up under it) and never EURGBP.

## Result (run `86ccb72`, 2020–2025)

No candidate in any method or half; nothing to cross-confirm.

- **Lead–lag:** FX does follow bond moves for one or two bars (predictive t about 3 for every pairing in 2020–2022; in 2023–2025 only EURGBP, t 3–4, and USDJPY, t 2.3), but the follow-through is about 5% of the bond-implied move: 0.1–0.6 bp per trade at the mid against about 1.1–1.4 bp of spread plus commission. Every configuration loses net.
- **Residual reversion:** the AR(1) s-score is rarely valid (the residual mostly behaves like a random walk at 1–5 days), so trades are few. EURUSD earns about +1.2 bp net per trade in both halves, but on 68 and 187 trades (t 0.7–0.8), and the shifted-bond placebo does as well in 2020–2022.
- **Cointegration:** formations are cointegrated (p < 0.05) on 5–7% of days for most pairings and windows, about the test's false-positive rate (10–14% only for USDJPY and GBPUSD with 60-day formations), so there is no evidence of a level relationship; at most 25 trades per configuration.
- `D0` in the run's summaries could read `STRUCTURE_FOUND` with a negative best t; from engine 0.2.1, E4 also needs best t > 0. No candidate was affected (none passed E1–E3).
