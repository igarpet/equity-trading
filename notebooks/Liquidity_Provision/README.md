# Liquidity Provision — paper P2

**Paper.** Nagel, S. (2012). Evaporating Liquidity. *Review of Financial Studies* 25(7), 2005–2039. NBER w17653.

**Claim to test.** Returns of short-term reversal strategies proxy for the returns to liquidity provision, and their expected value rises with market volatility (VIX).

**Our question.** Is the residual reversal edge larger in high-volatility regimes, enough to clear costs? Gate S05 entries by a market-volatility state known at the time.

| Notebook | Description | Status |
|---|---|---|
| `01_daily_vix_reversal.ipynb` | H1 faithful: Lehmann contrarian portfolio, daily 2016-12 → 2025; gross return on lagged VIX and on the high-VIX regime (HAC); regime-gated portfolio net of measured costs 2020–2025 | Run e9dedda: `NO_REPLICATION` (t = 0.11 / 1.23), net negative |
| `02_high_volume_independent_sample.ipynb` | H2: P1's post-hoc observation (high abnormal volume reverses more) re-tested on 2017–2019, P1 arm-D construction at k = 5, 500 shuffled-label placebos | Run e9dedda: `NOT_CONFIRMED` (T3 − T1 = −0.84 bps/day, p = 0.80); P1 observation dropped |
| `03_intraday_vix_gated_residual.ipynb` | H1 applied: S05 residual (`x10` exits) with entries only after high-VIX closes, 2020–2025; 200 circularly shifted regime placebos | Run e9dedda: **`EDGE_CANDIDATE`**: net +22.7% (Sharpe 0.82) vs ungated +9.9%; +7.0% at 2× cost; beats 199/200 shifted regimes; high-regime trades 42.0 bps gross vs −0.7 low |
| `04_vix_gated_replication_2017_2019.ipynb` | Frozen VIX-gated S05 (spec in protocol 003), independent replication on 15m 2017–2019 with new quote samples; same gates; `REPLICATED` → 2026 holdout | Built; awaiting run |

Readout: [`002_claude_p2_readout.md`](../../discussions/Liquidity_Provision/002_claude_p2_readout.md). Protocol: [`001_claude_p2_liquidity_provision_preregistration.md`](../../discussions/Liquidity_Provision/001_claude_p2_liquidity_provision_preregistration.md).

Run: `python -m pytest -q tests/test_p2.py`, then 01 (downloads `1Day` 2016–2018 and `^VIX` from Yahoo), 02, 03 (~1–3 h). `P2_SYNTHETIC=1` runs all three offline. Daily data now spans 2016–2025: 2016 is warm-up, 2017–2019 is the independent sample for H2, economic gates stay on 2020–2025 (measured costs). Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
