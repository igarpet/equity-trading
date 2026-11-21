# MRP 005 — Continuation entry with a volatility stop, 50/30 trailing stop and reverse re-entry (MRP 04, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-21 |
| **Branch** | MNQ_Return_Prediction |
| **Responds to** | Iván, 2026-11-21: after MRP 03 (continuation with fixed holds failed on the test window, win rate ≈ 0.46, average win ≈ 120 ticks), use the volatility engine for the stop, a trailing stop and a reverse re-entry |
| **Engine** | `trail_strategy.py` 0.1.0, `reversal.py` 0.2.0, `mnq_predict.py` 0.1.1 |
| **Status** | Discovery only. The test window (2023-07 – 2025-12) was used by MRP 03, so this runs on the training window 2020-01 – 2023-06. The training window has been seen (MRP 01–02), so a pass here is a candidate, confirmed only on the sealed 2026 holdout. |

## 1. Question

The same entries have negative expectancy with fixed holding times (MRP 03). Does this exit and re-entry logic turn them positive?

Exits alone cannot create an edge on a random walk. This test therefore asks whether the price path after a large bar has structure that these exits capture. Hence the random-direction placebo (G4).

## 2. Rules (Iván, 2026-11-21; answers to Claude's questions)

- **Signal**: 5-minute bar t with |z_t| ≥ 3 (MRP 001 volatility model; the 2020 fit is in sample, later dates use the walk-forward fits made before 2021-01-01, 2022-01-01 and 2023-01-01).
- **Entry**: with the bar (+sign r_t), at market at the close of bar t.
- **Initial stop**: m × σ̂_{t+1} × price, rounded to ticks, where σ̂_{t+1} is the forecast 5-minute volatility of the next bar. m ∈ **{1, 1.5, 2, 3}**: 4 strategies.
- **Trailing stop**: once the best price is ≥ 50 ticks in profit, the stop moves to +30 ticks and then follows the best price 20 ticks behind, tick by tick. It never moves back.
- **Re-entry**: only if the *initial* stop is hit. A **reverse-direction** limit order is placed 2 ticks beyond the stop (long stopped at S → sell limit at S + 2 ticks; short stopped at S → buy limit at S − 2 ticks). It works for 30 minutes; at most **one** re-entry per signal. The re-entry trade has its own stop (m × σ̂ of the 5-minute bar it fills in) and the same trailing rule.
- **Time limit**: flat by 15:55 Chicago, and before any contract switch.
- **One chain at a time**: signals arriving while a chain is open or a re-entry order is working are skipped.
- **"A little above / below"** is fixed at 2 ticks (Claude's choice; Iván did not specify).

## 3. Fill and cost model (on 1-minute bars)

- Market entry: the 5-minute close + 0.5 tick.
- Stop exit: the stop price − 0.5 tick, or the minute's open − 0.5 tick if the open is already beyond the stop.
- Flat exit: the close of the last minute before 15:55 Chicago − 0.5 tick.
- Limit re-entry: fills only when the minute trades 1 tick through the limit, at the limit price.
  - In the fill minute, the adverse extreme can stop the re-entry out.
  - The favourable extreme of the fill minute is not credited.
- Inside a minute the adverse extreme is checked first; a trailing stop raised in a minute applies from the next minute.
- Commission per fill: Lucid rate per side (Iván confirms; default $0.50 = 1 tick).
- All results are in ticks per contract (1 tick = $0.50).

## 4. Gates (discovery, per stop multiple m)

The unit is the **chain**: an entry plus its re-entry, if any.

- **G1**: at least 150 chains.
- **G2**: mean net ticks per chain > 0, with a one-sided t clustered by CME day, Holm-adjusted across the 4 values of m, ≤ 0.05.
- **G3**: mean net > 0 in at least 3 of 4 periods (2020, 2021, 2022, 2023H1).
- **G4 (placebo)**: the mean net must beat the 95th percentile of the same rules run with a **random direction** at the same signal times (20 seeds). This separates "the exits capture continuation" from "the exits shape a random walk".

A strategy passing G1–G4 goes to a 2026-holdout pre-registration (one m, Lucid account rules). Nothing else does.

Also reported:
- exit reasons (initial stop, trailing, flat);
- re-entry fill rate;
- win rate, mean win and loss, payoff ratio;
- R multiples (net / initial stop);
- maximum drawdown in ticks;
- per period;
- the same results with sizing at a fixed $100 risk per chain on the initial stop (at least 1 contract). Sizing changes the weights, not the sign of each chain.

## 5. Readings fixed in advance

- Discovery reuses data on which the entry was already seen to lose in 2023H1. A pass is weak evidence until 2026.
- If no m passes, the strategy is closed. Variants of the trailing distances (50/30/20) found by looking at these results would be fitting.

## 6. Run

`notebooks/MNQ_Return_Prediction/07_trailing_strategy.ipynb` (after `pytest tests/test_trail_strategy.py`).
