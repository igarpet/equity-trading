# fOU equity statistical arbitrage

Research rebuild of *Fractional Ornstein-Uhlenbeck Statistical Arbitrage with Synthetic Equity Options* (García Petrov, UNED MSc, 2026), on US equities at 15–30 minute bars.

## Rules

- Studies are Jupyter notebooks in `studies/`; each has a pre-registration (`studies/SNN_README.md`). Its gates are written before results exist and are not changed afterwards.
- Statistical, economic-after-costs and prospective verdicts are reported separately.
- Reusable logic lives in `src/equity_trading/`. Study scripts import it.
- Every study writes compact CSV/JSON results plus a manifest to `results/sNN/`.
- Data stays out of git. `EQUITY_TRADING_DIR` is the shared market-data folder (alongside `Deribit/` and `Binance/`):
  vendor data goes to `Alpaca/` and `Yahoo/`, large derived outputs to `EquityTrading/`.

## Studies

| ID | Question | Data | Status |
|---|---|---|---|
| S01 | Is intraday H < 0.5 real, or estimator bias? | Yahoo, 60 days, 30m/15m | Run 9ab45bb: `NO_EVIDENCE`, uninformative (no power) |
| S02 | Are same-sector / stock-ETF pairs cointegrated beyond a null, and does it persist? | Alpaca SIP 2020–2025, 30m/15m | Run 26c6437: `NO_COINTEGRATION_SIGNAL` (sector and ETF) |
| S03 | Thesis pair discovery (S&P 500, top-10 corr, I(1), EG 1%) on intraday bars + Johansen, rolling, GICS / correlation clustering | Alpaca SIP 2020–2025, 30m/15m | Run d30e3be (30Min): `NO_EXCESS` (primary) |
| S04 | Dynamic OU pairs backtest: daily re-estimated pair list (BH-FDR, 3-day hysteresis), z-score entries/exits, 4% per pair, ≤ 15 pairs; random-pair and random-timing placebos | Alpaca SIP 2020–2025, 15m | Runs 6bb6e40 (`hl5`), fec1b00 (`fixed60`): `NO_EDGE`; `fixed120` pending |
| S05a | Avellaneda–Lee residual mean reversion: every S&P 500 stock vs its sector ETF, OU s-score rules, 1% per stock, ≤ 100 positions; random-direction and random-timing placebos | Alpaca SIP 2020–2025, 15m | Pre-registered; notebook ready, awaiting run |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```
