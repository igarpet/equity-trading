# MNQ next-bar return model on 5-minute bars (MRP 001)

**Question.** Given everything known at the close of a 5-minute bar, can MNQ's next-bar log return be forecast better than zero, and better than MNQ's own recent returns, using its own history (lags, large-move continuation, completed Asia/London/NY sessions, the previous CME day, the same slot yesterday) and the 7 USD FX majors? The model gives a mean and a volatility, so both P(up) and the size of the move.

Protocol: [`001_claude_mrp01_preregistration.md`](../../discussions/MNQ_Return_Prediction/001_claude_mrp01_preregistration.md). Training 2020-01-01 – 2023-06-30; test 2023-07-01 – 2025-12-31 (one look); 2026 sealed.

| Notebook | Content | Output |
|---|---|---|
| `00_data` | coverage, liquidity, gaps, rolls, exclusions, FX coverage and timestamp alignment (training window) | `00/` |
| `01_volatility` | time-of-day × HAR volatility model, walk-forward 2021–2023H1, **gate V** → primary scale (z or raw) | `01/` |
| `02_mean_model_train` | in-sample inference (groups A–G, Holm, Huber, halves), walk-forward M0–M3, **gate S** | `02/` |
| `03_test` | test window, **gate T** and **gate E** (one-bar trade after costs); refuses without `02/summary.json` | `03/` |
| `04_reversal_discovery` | MRP 02: reversal after large bars (|z| ≥ k), 200 configurations, ticks per contract, gates D1–D4, frozen candidates | `04/` |
| `05_reversal_confirmation` | MRP 02: frozen candidates on the test window (one look), gates C1–C3 | `05/` |

**Data.** MNQ: `python tools/databento/micro_bars_download.py` (Databento `ohlcv-1m`, `MNQ.v.0`) → `<EQUITY_TRADING_DIR>/Databento/ohlcv-1m/MNQ/`. FX: the Dukascopy 1-minute candles already used by FXS / BFX.

**Code.** `src/equity_trading/mnq_predict.py` (grid, FX alignment, volatility model, dataset, inference, walk-forward, economic test, synthetic data) with `micro_bars.py`. Tests: `tests/test_mnq_predict.py`, `tests/test_micro_bars.py`.

**Run order.** `python -m pytest -q tests/test_mnq_predict.py tests/test_micro_bars.py`, then 00 → 01 (commit `01/`) → 02 (commit `02/`) → set the Lucid commission in 03 → 03.

**Smoke run.** `MRP_SYNTHETIC=1` (one `MRP_SYNTH_DIR` for all four notebooks) runs everything offline on a compressed calendar (train 2020, test 2021H1). The synthetic MNQ has clustered volatility, an intraday profile, quarterly rolls and a planted lag-1 USDJPY effect (0.15 in z): gate V passes, F is the only significant group, M3 beats M0 and M2, gates T and E pass.

## Result MRP 01 (runs `10f6a0b`, `0e0c3df`): no forecasting value

Readout: [`002`](../../discussions/MNQ_Return_Prediction/002_claude_mrp01_readout.md).
- Gate V failed on V2 only: the variance level is too high in 2021. The ranking is strong (DM t = 14.5), so the primary scale is raw.
- No regressor group survives Holm. R²_OOS is negative for M1–M3 in every period. Gate S failed, and 03 was not run.
- The only stable coefficient is the lag-2 reversal (≈ 0.9 ticks after a 1σ bar).

**MRP 02** (conditional reversal, [`003`](../../discussions/MNQ_Return_Prediction/003_claude_mrp02_conditional_reversal_preregistration.md)): code ready, not run. Smoke: `MRP_SYNTHETIC=1` (with its own `MRP_REV_SYNTH_DIR`) plants a 24-tick reversal after 3-sd bars; 04 finds it and 05 confirms it.
