# RSW 008 — RSW 04 readout: burst reversion at the switch to calm has no edge

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Run** | `2e702f3` (Iván), engine `burst_reversion.py` 0.1.0. Superseded by engine 0.1.1 (see below); rerun pending |
| **Protocol** | [`007`](./007_claude_rsw04_burst_reversion_preregistration.md) |

## Result (run `2e702f3`)

| | M1 | WB2 |
|---|---|---|
| Trades | 3,483–5,072 | 3,801–5,609 |
| Gross per trade | −0.8 to +0.9 ticks | −1.1 to +1.0 ticks |
| Net per trade (3 ticks) | −1.1 to −2.8 | −1.0 to −3.1 |
| Best t | −0.53 | −0.33 |
| Mirror (with the burst), net | −3.7 to +1.5 | −4.9 to −0.5 |
| Max drawdown, 1 contract | 12,100–20,200 ticks = 3,000–5,000 points = $6,100–10,100 | 11,400–18,700 ticks = $5,700–9,300 |

**Verdict:** `NO_EDGE` for all 10 strategies.
- No strategy passes G2 or G4.
- 2023H1 is negative for every strategy (−4 to −15 ticks per trade).
- Reversion, the mirror and the random direction all come out at about zero gross: the switch out of a burst carries no information about the next 15–20 bars.
- The net by burst-size quartile and side is noise: about 350 trades per cell, SE about ±11 ticks, and the signs flip between neighbouring quartiles.

## Bug found after the run (engine 0.1.1)

A switch on the 15:55–16:00 Chicago bar entered at that bar's close. The simulation then started from the next existing minute, i.e. the next session's 17:00 open, or Sunday. Those trades carried across the daily break with the gap.

From 0.1.1 such switches are skipped and counted. On synthetic data this affects about 0.4% of trades. The verdict cannot flip from these few trades, but the run must be repeated for exact numbers.

## Data gap to check

Every strategy is flat from about mid-May to early July 2022: no trades for 6–7 weeks. `micro_bars.load_1m` skips missing month files silently, so the most likely cause is a missing or partial `MNQ_2022-05` / `MNQ_2022-06` parquet. Notebook 05 prints a monthly coverage table that will show it. If data is missing, every MRP and RSW study lacked those weeks. That is not a bias, but those weeks should be re-downloaded.

## Prop-account reading (Iván, 2026-11-22)

"Better than a coin flip" means positive expectancy after costs, not a win rate above 50%.
- The no-stop variant wins 51% of trades but loses about 1–2 ticks per trade.
- A negative-expectancy strategy passes an evaluation only by luck.
- The drawdowns are far beyond prop trailing-drawdown limits.

## Next

Iván (2026-11-22): enter at the **start** of the burst on 1-minute bars, with or against the first burst minute. RSW 05, [`009`](./009_claude_rsw05_burst_entry_preregistration.md).
