# MNQ hierarchical Markov regimes — daily → 1 hour → 15 minutes (HMK)

**Question.** Does conditioning MNQ's intraday Markov regimes on the higher timeframe's state (P(intraday state | daily state)) give a calibrated P(up) for the next 1-hour bar that pays its costs? Price only. Micro contracts only. Ticks per contract.

Protocol: HMK 01 [`001`](../../discussions/MNQ_Hierarchical_Markov/001_claude_hmk01_preregistration.md).

| Notebook | Content | Output |
|---|---|---|
| `01_hierarchical_markov` | 3-state HMMs on 1d, 1h and 15m returns; child transitions depend on the parent's one-step forecast (`H3`), vs no parent (`U`) and the video's 20-day ±5% state (`RULE`); walk-forward 2021-01 – 2023-06; gates G1–G4; conditional matrices, calibration, always-in sign(E[r]) vs always long | `01/` (`01c/` with `HMK_CONFIRM=1`) |

**Data.** MNQ `ohlcv-1m` already on disk (`micro_bars`); nothing to buy. Nothing from 2026.

**Code.** `src/equity_trading/hmk.py` (numba EM with parent-dependent transitions, causal filter, bars, gates, synthetic data). Reuses only `micro_bars.load_1m` and `futures.cme_trade_date`; no other branch's models or thresholds. Tests: `tests/test_hmk.py`.

**Run order.**
1. `python -m pytest -q tests/test_hmk.py`
2. Run `01` (fits cached in `01/wf_0.1.0.pkl`; delete to refit).
3. Confirmation only after the discovery readout: `set HMK_CONFIRM=1`, rerun `01` (writes `01c/`).

**Smoke runs.** `HMK_SYNTHETIC=1` (planted: G2–G4 pass, G1 fails by design, see HMK 001 §8); add `HMK_SYNTH_NULL=1` with a separate `HMK_SYNTH_DIR` (no structure: all gates fail).

Status: code ready, not run.
