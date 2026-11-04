# Short Term Momentum — paper P1

**Paper.** Medhat, M. & Schmeling, M. (2022). Short-term Momentum. *Review of Financial Studies* 35(3), 1480–1526. CEPR DP15857.

**Claim to test.** Short-term reversal is concentrated in low-turnover stocks; the highest-turnover stocks show short-term momentum instead (monthly sorts on prior-month return × prior-month turnover).

**Our question.** Does the S05 residual reversal depend on turnover? Split the universe by trailing relative turnover: is reversal stronger (and above cost) in low-turnover names, and do high-turnover names continue rather than revert?

| Notebook | Description | Status |
|---|---|---|
| `01_daily_turnover_reversal.ipynb` | Arm D (faithful): daily reversal (k = 21, 5, 1) within abnormal-volume terciles, conditional sorts, one-session skip, measured close-slot costs; bootstrap and shuffled-label placebo | Run a7d07f0: `NO_REPLICATION`, not profitable (all k); high-AV terciles reverse *more* at k = 5 (post hoc) |
| `02_intraday_turnover_residual.ipynb` | Arm I (applied): S05 residual signal, `x10` exits, entries masked to one AV tercile (T1/T2/T3/ALL); trade-level T1 − T3, random-membership placebos | Run a7d07f0: `NO_EDGE`; T1 gross 11.5 bps/trade vs T3 23.5 (opposite to the paper; T3 vs placebos p = 0.13, post hoc) |

Readout: [`002_claude_p1_readout.md`](../../discussions/Short_Term_Momentum/002_claude_p1_readout.md). Protocol: [`001_claude_p1_turnover_reversal_preregistration.md`](../../discussions/Short_Term_Momentum/001_claude_p1_turnover_reversal_preregistration.md).

Run: `python -m pytest -q tests/test_p1.py tests/test_s05.py`, then notebook 01 (downloads `1Day` bars 2019–2025, a few minutes), then 02 (~1–3 h with 200 placebos). `P1_SYNTHETIC=1` runs both offline. Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
