# Pairs Cointegration — thesis pair discovery and OU pairs trading on intraday bars

S01–S04: is intraday anti-persistence real (S01), are same-sector / stock–ETF pairs cointegrated beyond a null (S02), does the thesis pair-discovery pipeline select more than chance on 15–30 minute bars (S03), and does a dynamic OU pairs book earn after costs (S04).

| Notebook | Question | Data | Status | Protocol |
|---|---|---|---|---|
| `01_hurst_null.py` (S01) | Is intraday H < 0.5 real, or estimator bias? | Yahoo, 60 days, 30m/15m | Run 9ab45bb: `NO_EVIDENCE`, uninformative (no power) | [001](../../discussions/Pairs_Cointegration/001_claude_s01_hurst_null.md) |
| `02_cointegration_base_rate.ipynb` (S02) | Are same-sector / stock-ETF pairs cointegrated beyond a null, and does it persist? | Alpaca SIP 2020–2025, 30m/15m | Run 26c6437: `NO_COINTEGRATION_SIGNAL` (sector and ETF) | [002](../../discussions/Pairs_Cointegration/002_claude_s02_cointegration_base_rate.md) |
| `03_thesis_pair_discovery_intraday.ipynb` (S03) | Thesis pair discovery (S&P 500, top-10 corr, I(1), EG 1%) on intraday bars + Johansen, rolling, GICS / correlation clustering | Alpaca SIP 2020–2025, 30m/15m | Run d30e3be (30Min): `NO_EXCESS` (primary) | [003](../../discussions/Pairs_Cointegration/003_claude_s03_thesis_pair_discovery_intraday.md) |
| `04_dynamic_ou_pairs_backtest.ipynb` (S04) | Dynamic OU pairs backtest: daily re-estimated pair list (BH-FDR, 3-day hysteresis), z-score entries/exits, 4% per pair, ≤ 15 pairs; random-pair and random-timing placebos | Alpaca SIP 2020–2025, 15m | Void (correction C1); re-run pending | [004](../../discussions/Pairs_Cointegration/004_claude_s04_dynamic_ou_pairs_backtest.md) |

Each notebook writes compact CSV/JSON results plus a manifest into the subfolder named after its ID (`03_thesis_pair_discovery_intraday.ipynb` → `03/`). Large intermediates go to `<EQUITY_TRADING_DIR>/EquityTrading/` (cache paths unchanged). Synthetic smoke runs write `NN_synthetic/`, which git ignores. `01_hurst_null.py` is a script: `python notebooks/Pairs_Cointegration/01_hurst_null.py` (`S01_SYNTHETIC=1` for the offline smoke test).
