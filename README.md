# equity_trading

Research rebuild of *Fractional Ornstein-Uhlenbeck Statistical Arbitrage with Synthetic Equity Options* (García Petrov, UNED MSc, 2026), on US equities at 15–30 minute bars.

## Layout

Mirrors `igarpet/deribit-market-data`:

| Path | Content |
|---|---|
| `discussions/<Branch>/` | Pre-registrations, readouts and corrections, `NNN_author_topic.md` |
| `notebooks/<Branch>/` | Study notebooks `NN_name.ipynb`, outputs in `NN/`, branch README |
| `notebooks/README.md` | Research rules, branch index, old→new mapping |
| `src/equity_trading/` | Reusable engines; `reference/` holds constituent lists and thesis pairs |
| `tests/` | `pytest` suite |

Data stays out of git. `EQUITY_TRADING_DIR` is the shared market-data folder (alongside `Deribit/` and `Binance/`): vendor data goes to `Alpaca/` and `Yahoo/`, large derived outputs to `EquityTrading/`.

See [notebooks/README.md](notebooks/README.md) for the research rules and the study index.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```
