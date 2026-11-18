# Notebooks

One folder per research branch, matching `discussions/`. Notebook file names keep the study ID (`S05b` → `05b_measured_costs.ipynb`); each notebook writes its results into the subfolder named after that ID (`05b/`), so the branch folder holds only notebooks, their output folders and a README.

## Rules

- Each study has a pre-registration in `discussions/<Branch>/NNN_author_sID_topic.md`. Its gates are written before results exist and are not changed afterwards.
- Statistical, economic-after-costs and prospective verdicts are reported separately.
- Reusable logic lives in `src/equity_trading/`; notebooks import it. Reference inputs (constituent lists, thesis pairs) are in `src/equity_trading/reference/`.
- 2026 data is sealed as the second holdout; studies use 2020–2025 only.
- Research branches are independent: a branch does not silently reuse another branch's outputs, fitted models or thresholds as assumptions. Any cross-branch import needs Ivan's approval and must be documented.

| Folder | Content |
|---|---|
| [Pairs_Cointegration](./Pairs_Cointegration/) | S01–S04: intraday Hurst null, cointegration base rate, thesis pair discovery (EG, Johansen, rolling, clustering), dynamic OU pairs backtest. |
| [Residual_Reversion](./Residual_Reversion/) | S05a–S09: Avellaneda–Lee stock-vs-sector-ETF residuals: measured costs, exit/entry rules, passive execution, continuous weights, s-score diagnostics. |
| [Short_Term_Momentum](./Short_Term_Momentum/) | Paper P1: Medhat, M. & Schmeling, M. — Does the S05 residual reversal depend on turnover? Run a7d07f0: no replication, no edge; reversal is stronger in high abnormal volume (post hoc). |
| [Liquidity_Provision](./Liquidity_Provision/) | Paper P2: Nagel, S. — Is the residual reversal edge larger in high-volatility regimes, enough to clear costs? Run e9dedda: daily no replication; 15m VIX-gated S05 `EDGE_CANDIDATE` on 2020–2025, `NOT_REPLICATED` on 2017–2019 (run 8aba3d3); closed, holdout sealed. |
| [Intraday_Periodicity](./Intraday_Periodicity/) | Paper P3: Heston, S. L., Korajczyk, R. A. & Sadka, R. — Does same-slot periodicity exist on our 15m/30m SIP bars 2020–2025, and does it interact with (or contaminate) the residual reversal signal, especially at the open? Run 48a18b5: periodicity replicates in 2020–2025 and 2017–2019; untradable alone (cost ≈ 25× gross). Kept as a control variable. |
| [News_Filtered_Reversal](./News_Filtered_Reversal/) | Paper P4: Da, Z., Liu, Q. & Schaumburg, E. — Does removing news-driven residual moves (earnings dates, news flow) leave a stronger reversal? Run 0d41706: no replication, no edge; excluding earnings windows improves S05 in both windows (kept as a standard filter). |
| [Dynamic_Trading_Costs](./Dynamic_Trading_Costs/) | Paper P5: Gârleanu, N. & Pedersen, L. H. — Successor to S07: does the optimal partial-adjustment rule, with our measured S05b costs, turn the gross residual signal into a net positive? Run d9955d8: no edge; partial adjustment cuts costs but leaves no gross; slow 5-session residual has no forecasting value. |
| [Deep_Statistical_Arbitrage](./Deep_Statistical_Arbitrage/) | Paper P6: Guijarro-Ordonez, J., Pelger, M. & Zanotti, G. — Do richer factor residuals and a learned policy produce a gross edge larger than the sector-ETF residual, after measured costs? Run ccb77b2: no edge; network learns a low-turnover 5-day residual reversal, net ≈ 0 in both windows, fading in 2024–2025. |
| [Rank_Space](./Rank_Space/) | Paper P7: Li, Y.-F. & Papanicolaou, G. — Does rank-space mean reversion survive our measured costs (≈ 13 bps round trip in S05b), not 2 bps? Run f0b016d (after correction C1): no edge; rank-space Sharpe 4.5–11.6 is a non-tradable sorting effect, tradable gross ≈ 0 before costs. |
| [OU_Optimal_Thresholds](./OU_Optimal_Thresholds/) | Paper P8: Bertram, W. K. — Do model-optimal thresholds beat the S05c/S05d rules out of sample? Run d63819a: no improvement, no edge; OU-optimal rules do not fix a mis-specified OU model; fast exits have gross in both windows but far below cost. |
| [Futures_Order_Flow](./Futures_Order_Flow/) | CME micro index futures only (MNQ/MES/MYM, Lucid prop accounts): does MNQ's own signed taker flow pay its round trip? FOF 01–02: data plan (micros only), Databento (2023–2025) + NinjaTrader export (sealed 2026) + forward recorder, checks C1–C5. Code ready, not run. |
| [Gotobi_Tokyo_Fix](./Gotobi_Tokyo_Fix/) | GTF 01: USD/JPY gotobi days (Ito & Yamada 2017) and a YouTube rule (sell 09:55 JST, 12-pip stop, exit 10:30). Dukascopy ticks. Run 610d790: replicates 2016–2025 (+2.6 pips/trade after spread), not confirmed 2010–2015 → `NO_EDGE`; C1 rerun pending. GTF 02 (long 09:25→09:55 then short): code ready, binding test on the 2026 holdout. |
| [FX_Intraday_Seasonality](./FX_Intraday_Seasonality/) | FXS 01–02: stack every day's 1-minute candles by time of day (no calendar labels), 15 FX pairs, NY and UTC clocks, one study per pair; scan 2,704 intervals per pair on 2016–2020 (BH, year consistency, cost), confirm frozen candidates on 2021–2025. Code ready, not run. |
| [Bond_Intraday_Seasonality](./Bond_Intraday_Seasonality/) | BDS 01–02: the FX stacked-day scan on the US T-Bond, Bund and Gilt CFDs; discovery on 2018–2021, 2022–2025 and 2018–2025. Run: no structure (Bund, Gilt at noise level; T-Bond NY 15:30–16:00 drift ~1.1 bp/day, below the 2.9 bp futures cost). |
| [Bond_FX_Pairs](./Bond_FX_Pairs/) | BFX 01: EURUSD, GBPUSD, USDJPY, EURGBP against the bonds of their two currencies on 5-minute bars; lead–lag, residual reversion (Avellaneda–Lee) and cointegration of levels; discovery on 2020–2022 and 2023–2025 (bond quotes missing or stale before 2020). Run: no candidates; lead–lag real but 0.1–0.6 bp per trade at the mid, below costs; no cointegration beyond chance. |

Paper-testing order (Iván, 2026-11-04): P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8. Each paper is its own branch; it may use the shared S05 engine and the S05b cost model (documented in its protocol), but not other branches' thresholds or conclusions.

## Mapping from the old layout (restructured 2026-11-04)

| Study | Old | New (notebooks relative to `notebooks/`) |
|---|---|---|
| S01 | `studies/s01_hurst_null.py`, `results/s01/`, `studies/S01_README.md` | `Pairs_Cointegration/01_hurst_null.py` + `01/`, `discussions/Pairs_Cointegration/001_claude_s01_hurst_null.md` |
| S02 | `studies/S02_*.ipynb`, `results/s02/`, `studies/S02_README.md` | `Pairs_Cointegration/02_cointegration_base_rate.ipynb` + `02/`, `discussions/Pairs_Cointegration/002_claude_s02_cointegration_base_rate.md` |
| S03 | `studies/S03_*.ipynb`, `results/s03/`, `studies/S03_README.md` | `Pairs_Cointegration/03_thesis_pair_discovery_intraday.ipynb` + `03/`, `discussions/Pairs_Cointegration/003_claude_s03_thesis_pair_discovery_intraday.md` |
| S04 | `studies/S04_*.ipynb`, `results/s04/`, `studies/S04_README.md` | `Pairs_Cointegration/04_dynamic_ou_pairs_backtest.ipynb` + `04/`, `discussions/Pairs_Cointegration/004_claude_s04_dynamic_ou_pairs_backtest.md` |
| S05a | `studies/S05a_*.ipynb`, `results/s05a/`, `studies/S05a_README.md` | `Residual_Reversion/05a_avellaneda_lee_etf.ipynb` + `05a/`, `discussions/Residual_Reversion/001_claude_s05a_avellaneda_lee_etf.md` |
| S05b | `studies/S05b_*.ipynb`, `results/s05b/`, `studies/S05b_README.md` | `Residual_Reversion/05b_measured_costs.ipynb` + `05b/`, `discussions/Residual_Reversion/002_claude_s05b_measured_costs.md` |
| S05c | `studies/S05c_*.ipynb`, `results/s05c/`, `studies/S05c_README.md` | `Residual_Reversion/05c_exit_levels.ipynb` + `05c/`, `discussions/Residual_Reversion/003_claude_s05c_exit_levels.md` |
| S05d | `studies/S05d_*.ipynb`, `results/s05d/`, `studies/S05d_README.md` | `Residual_Reversion/05d_entry_rules.ipynb` + `05d/`, `discussions/Residual_Reversion/004_claude_s05d_entry_rules.md` |
| S06 | `studies/S06_*.ipynb`, `results/s06/`, `studies/S06_README.md` | `Residual_Reversion/06_passive_execution.ipynb` + `06/`, `discussions/Residual_Reversion/005_claude_s06_passive_execution.md` |
| S07 | `studies/S07_*.ipynb`, `results/s07/`, `studies/S07_README.md` | `Residual_Reversion/07_continuous_weights.ipynb` + `07/`, `discussions/Residual_Reversion/006_claude_s07_continuous_weights.md` |
| S07b | `studies/S07b_*.ipynb`, `results/s07b/`, `studies/S07b_README.md` | `Residual_Reversion/07b_wide_bands.ipynb` + `07b/`, `discussions/Residual_Reversion/007_claude_s07b_wide_bands.md` |
| S08 | `studies/S08_*.ipynb`, `results/s08/`, `studies/S08_README.md` | `Residual_Reversion/08_sscore_diagnostic.ipynb` + `08/`, `discussions/Residual_Reversion/008_claude_s08_sscore_diagnostic.md` |
| S09 | `studies/S09_*.ipynb`, `results/s09/`, `studies/S09_README.md` | `Residual_Reversion/09_har_switching_gaps.ipynb` + `09/`, `discussions/Residual_Reversion/009_claude_s09_har_switching_gaps.md` |
| — | `reference/` | `src/equity_trading/reference/` |

Only file locations and the paths in notebook code and docs changed. Saved cell outputs, commit messages and `summary.json` contents still show the old paths. Cache paths under `<EQUITY_TRADING_DIR>/EquityTrading/results/sNN/` are unchanged, so existing caches remain valid.
