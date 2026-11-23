# FXSeasonalityEA (MT5)

MetaTrader 5 Expert Advisor for the FX intraday seasonality portfolio (FXS 09/10). It trades the 12 time-of-day signals with a time entry and a time exit every trading day, sized at L × balance per signal.

## Which study it reproduces

These are the FX studies with the highest 10× CAGR that a live EA can trade:

| Study | Variant | L=10 CAGR | Max DD | Period | Status |
|---|---|---|---|---|---|
| 03h | optimal exit | 123.5% | – | 2020–2025 | **Lookahead.** Not tradable, excluded |
| 13b | HMM tilt 1.5/0.5 | 330.2% | −92.1% | 2021–2026 | **In-sample** sizing scheme. Not in the EA |
| **10 / 13** | **12 signals, no filter (`SET_ALL12`)** | **259.0%** | **−77.5%** | **2021–2026** | **Default EA setup.** 2026 alone: +11.6%, DD −51.6% |
| 10 | C1 gotobi, walk-forward | 181.6% | −76.3% | 2021–2026 | Best 2026: +82.8%, DD −48.8%. Use `InpGotobiRules` (see below) |
| 09 | ALL 12 | 137.4% | −93.5% | 2020–2025 | Same signals, different window |
| 07 | PREVIOUS + NEW, 8 signals (`SET_PREV_NEW`) | 117.8% | −86.8% | 2020–2025 | |

How to read these numbers:

- **The edge is the same at every L.** Sharpe is 1.65 at L=1 and at L=10; leverage only scales the edge.
- **L=10 is past the growth optimum.** In 2026 it returned less than L=5 (+11.6% vs +15.1%), with a −51.6% drawdown.
- **The 259% is partly in-sample.** 9 of the 12 signals were found on 2021–2025 data. Only 2026 is a clean holdout.
- **Prop-firm rules.** At L=10, the 5% daily-loss rule is breached within months (FXS 04).

## Signals (from `Portfolio/09/summary.json`)

| id | Pair | Clock | Window | Side |
|---|---|---|---|---|
| EURUSD-02 | EURUSD | NY | 04:00–08:00 NY | sell |
| USDJPY-01 | USDJPY | UTC | 00:55–01:55 UTC | sell |
| USDJPY-02 | USDJPY | UTC | 00:25–00:55 UTC | buy |
| N-EU-1 | EURUSD | NY | 07:45–08:00 NY | sell |
| N-UJ-1 | USDJPY | UTC | 00:55–01:10 UTC | sell |
| N-UJ-3 | USDJPY | NY | 15:20–16:20 NY | buy |
| N-EJ-1 | EURJPY | NY | 14:55–16:55 NY | buy |
| N-EJ-2 | EURJPY | NY | 06:00–08:00 NY | sell |
| AUDJPY-N1 | AUDJPY | NY | 12:55–16:55 NY | buy |
| AUDJPY-N2 | AUDJPY | UTC | 01:00–03:00 UTC | buy |
| CADJPY-N1 | CADJPY | NY | 10:50–14:50 NY | buy |
| GBPJPY-N1 | GBPJPY | NY | 12:55–16:55 NY | buy |

**NY clock.** Minutes are counted from 17:00 New York, so New York time = clock − 7 h. A NY-clock day label runs from 17:00 NY of the previous day to 17:00 NY; labels Mon–Fri are traded.

**UTC clock.** UTC calendar days, Mon–Fri.

## Mapping to the study

| Study | EA |
|---|---|
| Notional per signal = floor(equity × L / 1,000) × 1,000 USD, re-marked daily | Lots = balance × L × multiplier / (notional per lot), floored to the volume step. Notional per lot = price × tick value / tick size. Entries are capped at `InpMarginUse` of free margin. |
| Fill at the quoted bid/ask at the boundary minute | Market order at the scheduled second. Skipped if it cannot be sent within `InpMaxEntryDelaySec`. |
| Time exit | Market close at entry + (end − start) minutes |
| Days with < 80% minute coverage dropped | Dec 25 and Jan 1 labels skipped (`InpSkipHolidays`) |
| Costs: Dukascopy spread × 0.9 + $3/lot RT | The broker's real spread and commission |

**Account type.** The EA needs a **hedging** account, because USDJPY-01 and N-UJ-1 hold the same symbol at the same time. Each signal has its own magic number (`InpMagicBase + index`), and positions are closed per magic.

**Not in the EA:** HMM sizing (13/13b). It needs daily Python-fitted states, and the best HMM result (13b) is in-sample. `InpMultipliers` takes fixed per-signal multipliers.

## Server clock

`SERVER_NY_PLUS_7` (default) assumes server time = New York + 7 h: GMT+2 in US winter and GMT+3 in US summer. This is the usual FX broker clock.

- On a live or demo chart, the EA compares the real server–GMT offset with this assumption at start-up. A mismatch prints a `WARNING` in the Experts log.
- If the broker uses a fixed offset, set `SERVER_FIXED_GMT` and `InpServerGmtOffset`.
- **Check this before backtesting.** The tester cannot verify it, and a wrong clock shifts every trade by an hour or more.

The clock and gotobi logic were cross-checked in Python against `zoneinfo` and `gotobi.py`, 2016–2027, with no mismatches. The MQL5 file itself has **not been compiled here**; compile it in MetaEditor (F7) and report any errors.

## Gotobi rules (FXS 10, C1)

`InpGotobiRules` takes one rule per signal: `G` trades only on gotobi Tokyo dates, `O` trades only on other dates, `B` trades both. For example: `USDJPY-01=G;N-UJ-1=O`.

- **The rules are static.** In the study, the rule for year Y was learned from 2020 to Y−1. A static rule applied over 2021–2026 is therefore in-sample for the earlier years.
- **The Japanese holiday table covers 2015–2027.**

To get the rules for 2026 (trained on 2020–2025), run this in `10_conditioners.ipynb` after the walk-forward cell:

```python
rules = []
for i in SIGNALS:
    x = pips[i]; tr = (idx.year >= 2020) & (idx.year < 2026)
    b, _ = buckets(i, "C1 gotobi")
    ok = [k for k in (0, 1) if (lambda xs: len(xs) < MIN_BUCKET or xs.mean() > 0)(x[tr & (b == k)].dropna())]
    rules.append(f"{i}={'B' if ok == [0, 1] else 'G' if ok == [1] else 'O' if ok == [0] else 'X'}")
print(";".join(r for r in rules if not r.endswith("=X")))
print("disable:", ",".join(r.split("=")[0] for r in rules if r.endswith("=X")))
```

Paste the first line into `InpGotobiRules` and the second into `InpDisable`.

## Install and test (FundedNext MT5)

1. **Install.** Copy `FXSeasonalityEA.mq5` to `MQL5\Experts\` (File → Open Data Folder) and compile it in MetaEditor.
2. **Symbols.** In Market Watch, show EURUSD, USDJPY, EURJPY, AUDJPY, CADJPY and GBPJPY. If the broker adds a suffix, set `InpSymbolSuffix`.
3. **Strategy Tester settings:**
   - Expert `FXSeasonalityEA`, on any of the six symbols, M1.
   - Modelling: **Every tick based on real ticks**.
   - Deposit 6000 USD, leverage 1:100.
   - Dates: 2026.01.01 – today (clean holdout), then 2021.01.01 – 2025.12.31.
   - Inputs: `InpSet = SET_ALL12`, `InpLeverage = 10`. Also run `InpLeverage = 2` to compare with the study's L=2 rows: 43.3% CAGR, −19.7% DD for 2021–2026; +7.2% for 2026.
4. **Check commission.** The Journal must show a non-zero commission on deals. If it is 0, the tester is not charging the $3/lot, and the results are too high by that amount.
5. **Outputs:**
   - The trade journal is written to `Terminal\Common\Files\FXS_EA_trades.csv`, one row per trade: pips, profit, commission, swap, net.
   - `OnDeinit` prints net pips per trade per signal. Compare these with the study's net pips (`09/summary.json`, signals table).
   - The custom optimisation criterion (`OnTester`) is CAGR.

Expected gaps against the study:

- Broker spreads differ from Dukascopy × 0.9.
- Tester tick history depth may be shorter than 2021.
- Entries fill at the first tick after the scheduled second.
