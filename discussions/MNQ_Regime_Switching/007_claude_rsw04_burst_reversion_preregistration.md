# RSW 007 — Burst reversion: enter against the burst when the regime switches back to calm (RSW 04, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Responds to** | Iván, 2026-11-22: when the high-volatility state ends, go long after a negative burst and short after a positive one, betting on a slow reversion. No take profit; time stop at the mean calm duration + 5 bars; test several stop losses. He chose all four stop options and both regime models |
| **Engine** | `burst_reversion.py` 0.1.0, `hsmm.py` 0.1.0, `regime_ar.py` 0.1.0, `trail_strategy.py` 0.1.0 (minute arrays), `mnq_predict.py` 0.1.1 |
| **Status** | Discovery on 2021-01 – 2023-06 (regime fits out of sample; data seen before). A pass is a candidate, confirmed only on the sealed 2026 holdout. The 2023-07 – 2025-12 window was used by MRP 03. |

## 1. Prior evidence (fixed before the run)

**Against reversion:**
- MRP 02: no reversal after large bars, and the move looked like continuation.
- RSW 02: extremes within 12 bars keep the same sign 59.5% of the time.
- RSW 01: the calm regime's φ ≈ −0.05 gives back only about 5% of the last bar.

**For it:** RSW 03 C found a positive drift early in calm periods. Bursts are mostly selloffs, so this is a rebound. It failed as a 1-bar trade; this test holds 15–20 bars.

## 2. Rules

**Regime.** The causal P(burst | F_t) from the walk-forward 2-state fits:
- **M1**: HMM, constant transitions.
- **WB2**: Weibull HSMM.

Fits are made before 2021-01-01, 2022-01-01 and 2023-01-01, expanding from 2020-01-01, and each is frozen for its year. The burst state is the state with the larger sd.

**Episode.** Consecutive rows with P(burst) > 0.5.

**Switch.** The first row after an episode with P(burst) ≤ 0.5, on an observed bar with slot ≥ 3. Entry is at that bar's close, at market.

**Direction.** Against the episode's net move, measured from the close of the row before the episode to the close of its last row. Both rows must be on the same CME day and contract.

**Time stop.** Exit at the close of bar switch + H, where H = round(the frozen fit's mean calm duration) + 5. That is about 15–21 bars.

**Stop losses (5):**
- none;
- **extreme**: 2 ticks beyond the burst's 1-minute low (for a long) or high (for a short);
- **vol1.5** and **vol3**: m × σ̂ of the next bar × price;
- **fixed40**: 40 ticks (10 points).

**Other rules.**
- One position at a time.
- A new burst during a trade does not close it.
- Flat by 15:55 Chicago and before any contract switch.
- **Amendment 2026-11-22 (engine 0.1.1, bug fix after run `2e702f3`).** A switch on the last bar of the CME day (15:55–16:00 Chicago) entered at that bar's close. Engine 0.1.0 then simulated from the next existing minute, the next session's 17:00 open (or Sunday), so those trades carried across the daily break with the gap. From 0.1.1 a switch whose next minute is at or after 15:55 Chicago, or in another CME day, is not traded. The notebook prints how many were skipped. Run `2e702f3` is superseded; the gates are unchanged.

**Fills** (1-minute bars, MRP 04 conventions):
- market and stop fills pay 0.5 tick;
- a stop gapped through fills at the minute's open;
- commission $0.50 per side (Lucid; confirm);
- round trip 3 ticks.

All results are in ticks per contract (1 tick = $0.50) and in dollars per contract.

## 3. Gates (per strategy: 2 models × 5 stops = 10)

- **G1**: at least 150 trades.
- **G2**: mean net > 0, with one-sided p (t clustered by CME day) Holm-adjusted across the 10 strategies ≤ 0.05.
- **G3**: mean net > 0 in at least 2 of the 3 periods (2021, 2022, 2023H1).
- **G4**: mean net above the 95th percentile of the same rules with a **random direction** at the same entry times (20 seeds). This separates reversion from what the exits do on their own.

A strategy passing G1–G4 goes to a 2026-holdout pre-registration (one strategy, Lucid account rules). The strategies share entries, so several passes are one finding.

**Also reported:**
- longs (after down bursts) and shorts separately;
- the mirror direction (with the burst);
- exit reasons, win rate, mean win and loss;
- maximum drawdown, worst trade and 5% quantile;
- net by burst-size quartile;
- cumulative net plot.

## 4. Readings fixed in advance

- Variants found by looking at these results (other hold times, burst-size filters, long-only) would be fitting. They would need the 2026 holdout as their first test.
- A pass driven only by longs after down bursts is the RSW 03 C rebound. The equity-index upward drift contributes to it; the placebo does not remove that, because it is direction-symmetric. Compare to the long side of the mirror and to the unconditional drift (about 0.1 tick per bar over 2020–2023H1).

## 5. Smoke runs (obtained before any real data was read)

- **`RSW_SYNTHETIC=1`** (planted: 60% of each burst's move reverts over the next 15 calm bars): all 10 strategies pass G1–G4. Net +4.9 to +14.8 ticks per trade, placebo mean −1.2 to −7.1, mirror −12 to −21.
- **`RSW_SYNTHETIC=1 RSW_SYNTH_NULL=1`** (no planted reversion): `NO_EDGE`. Net −3.8 to +1.0 ticks, every Holm p = 1.

## 6. Run

1. `python -m pytest -q tests/test_burst_reversion.py`
2. `notebooks/MNQ_Regime_Switching/04_burst_reversion.ipynb`

Fits are cached in `04/fits_<version>.pkl`.

## 7. Extension to 2025-12 (Iván, 2026-11-22; written before the extended run)

The rules are frozen (engine 0.1.1, with the daily-break fix). Run them on **2021-01 – 2025-12**:
- regime fits (M1, WB2) refitted before every year from 2021 to 2025, expanding from 2020-01-01;
- volatility walk-forward with the same refit dates.

Results are reported on three windows:
- **original**: 2021-01 – 2023-06;
- **new**: 2023-07 – 2025-12, the MRP test window, used before only by MRP 03;
- **all**.

Each window gets net, t and Holm across the 10 strategies.

**Status.** This is a check. Discovery already failed, so a pass on the new window would not make a strategy a candidate on its own merits. It would need its own pre-registration and the 2026 holdout. The original window will differ slightly from run `2e702f3` because of the fix, and because 2023's regime fits now come from the same yearly schedule.

**Data.** June 2022 is almost empty (1 day, 120 minutes; RSW 05 coverage table) until it is re-downloaded.

**Run:** `notebooks/MNQ_Regime_Switching/04b_burst_reversion_full.ipynb`, output `04b/`.
