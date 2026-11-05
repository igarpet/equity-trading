# Intraday Periodicity — paper P3

**Paper.** Heston, S. L., Korajczyk, R. A. & Sadka, R. (2010). Intraday Patterns in the Cross-Section of Stock Returns. *Journal of Finance* 65(4), 1369–1407. arXiv 1005.3535.

**Claim to test.** A stock's return in a given half-hour interval predicts its return in the same interval on subsequent days, for weeks; within-day returns show reversal.

**Our question.** Does same-slot periodicity exist on our 15m/30m SIP bars 2020–2025, and does it interact with (or contaminate) the residual reversal signal, especially at the open?

| Notebook | Description | Status |
|---|---|---|
| `01_periodicity_regressions.ipynb` | Faithful: 30-minute intervals (full days), Fama–MacBeth slopes at lags 1–520, NW t; S1 daily-multiple slopes > 0, S2 above other lags; windows A 2020–2025 and B 2017–2019 | Built; awaiting run |
| `02_same_slot_strategy.ipynb` | Economic: same-slot 20-session mean signal, decile long-short per interval, flat overnight, measured costs; E1 gross, E2 net, both windows | Built; awaiting run |

Protocol: [`001_claude_p3_intraday_periodicity_preregistration.md`](../../discussions/Intraday_Periodicity/001_claude_p3_intraday_periodicity_preregistration.md). Both windows are judged (2020–2025 and the 2017–2019 replication).

Run: `python -m pytest -q tests/test_p3.py`, then 01 and 02. Nothing is downloaded: bars and quote samples come from the S05b / P2-04 caches. `P3_SYNTHETIC=1` runs both offline. Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
