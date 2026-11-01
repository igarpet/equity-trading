# fOU equity statistical arbitrage

Research rebuild of *Fractional Ornstein-Uhlenbeck Statistical Arbitrage with Synthetic Equity Options* (García Petrov, UNED MSc, 2026), on US equities at 15–30 minute bars.

## Rules

- Studies are Jupyter notebooks in `studies/`; each has a pre-registration (`studies/SNN_README.md`). Its gates are written before results exist and are not changed afterwards.
- Statistical, economic-after-costs and prospective verdicts are reported separately.
- Reusable logic lives in `src/equity_trading/`. Study scripts import it.
- Every study writes compact CSV/JSON results plus a manifest to `results/sNN/`.
- Raw data stays out of git. Set `EQUITY_TRADING_DIR`, or the default `./data/` is used.

## Studies

| ID | Question | Data | Status |
|---|---|---|---|
| S01 | Is intraday H < 0.5 real, or estimator bias? | Yahoo, 60 days, 30m/15m | Run 9ab45bb: `NO_EVIDENCE`, uninformative (no power) |
| S02 | Are same-sector / stock-ETF pairs cointegrated beyond a null, and does it persist? | Alpaca SIP 2020–2025, 30m/15m | Pre-registered; notebook ready, awaiting run |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```
