# RSW 009 — Burst entry on 1-minute bars: with or against the first burst minute (RSW 05, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Responds to** | Iván, 2026-11-22: enter just after the first 1-minute candle of the high-volatility regime, (a) continuing its direction and (b) against it. Models M1 and WB2 with the five stop settings, 2 × 2 × 5 = 20 backtests. His answers: **M1** (2-state HMM) and WB2; a **new 1-minute regime model**; time stop = **mean burst duration + 5 bars** |
| **Engine** | `burst_entry.py` 0.1.0, `regime_ar.py` 0.1.0, `hsmm.py` 0.1.0, `burst_reversion.py` 0.1.1 (`stats_for`) |
| **Status** | Discovery on 2021-01 – 2023-06 (regime fits out of sample). A pass is a candidate, confirmed only on the sealed 2026 holdout. |

## 1. Why a 1-minute model

The RSW 01–04 regimes run on 5-minute bars, so the earliest a burst can be known is the close of a 5-minute bar. Entering on "the first 1-minute candle" of a burst found on 5-minute bars would use the bar's close, which is in the future. A model run on 1-minute bars makes the first burst minute observable causally.

## 2. Data and models

**Grid.** 1-minute ohlcv (`MNQ.v.0`), CME days 17:00–16:00 Chicago.
- Only traded minutes are rows.
- r_t = log close change from the previous minute, only if that minute traded in the same CME day and contract. Otherwise there is no observation.
- Observed rows: slot ≥ 3 and a CME day with ≥ 900 traded minutes.

**Volatility.** Time-of-day profile of r² (1,380 slots, 5-minute smoothing) × HAR on deseasonalized r², with windows of 60 minutes, 1 day, 5 days and 22 days.
- 2020 uses an in-sample fit. Later dates use fits made before 2021-01-01, 2022-01-01 and 2023-01-01 (expanding).
- Then z = r / (σ̂ L), with L = rms of z over the previous 1,380 minutes, as in RSW 01.

**Regimes.**
- **M1**: 2-state AR(1) HMM.
- **WB2**: 2-state Weibull HSMM, ages 1–30 minutes plus a tail.
- Walk-forward fits as above; each is frozen and filters its own year causally.
- Burst = the state with the larger sd.

**Data coverage.** The notebook prints traded minutes per month and flags months with fewer than 700 per CME day. RSW 04's trades stop from mid-May to early July 2022, which may be missing data.

## 3. Rules

**Entry.** At the close of the first minute with P(burst | F_t) > 0.5, after a minute with P ≤ 0.5. The row must be observed and the return non-zero. Market order.
- **with**: in the direction of that candle;
- **against**: the opposite direction.

**Time stop.** Exit at the close of the row entry + H, with H = round(the frozen fit's mean burst duration, in minutes) + 5. "Bars" are 1-minute rows (traded minutes), because the model runs on 1-minute bars.

**Stops (5).**
- none;
- **extreme**: 2 ticks beyond the entry candle's extreme against the position (long: below its low; short: above its high);
- **vol1.5** and **vol3**: m × √5 × the next minute's σ̂ × price. This is the same 5-minute size definition as RSW 04;
- **fixed40**.

**Other rules.**
- One position at a time.
- No entry without at least one minute before 15:55 Chicago.
- Flat by 15:55 Chicago, never across the daily break or a contract switch.

**Fills.** Market and stop fills pay 0.5 tick; a stop gapped through fills at the minute's open; $0.50 commission per side. Round trip 3 ticks.

## 4. Gates (20 strategies: 2 models × 2 directions × 5 stops)

- **G1**: at least 150 trades.
- **G2**: net > 0, with a one-sided day-clustered p Holm-adjusted across the 20 ≤ 0.05.
- **G3**: net > 0 in at least 2 of the 3 periods (2021, 2022, 2023H1).
- **G4**: net above the 95th percentile of the random-direction placebo (20 seeds) at the same entries.

The two directions are each other's mirror. A pass on one side is one finding, as are several stops on the same side. A pass goes to a 2026-holdout pre-registration (one strategy, Lucid rules).

**Also reported:**
- equity curves;
- net per month for every strategy (where the drawdowns happen; descriptive only);
- burst starts per month;
- the fit parameters (mean burst and calm duration in minutes, φ, sd, drift).

## 5. Prior evidence

- **For continuation:**
  - RSW 01 burst φ ≈ +0.12 to +0.29 on 5-minute bars;
  - RSW 02: clustered extremes keep their sign 59.5% of the time.
- **Against:**
  - MRP 03 continuation after large 5-minute bars failed out of sample;
  - RSW 01 gate E failed.

On 1-minute bars the entry is earlier, which is what is new here.

## 6. Smoke runs (obtained before any real data was read)

**Synthetic data.** Minute regime with bursts at 3× sd, lasting about 5 minutes. Highs and lows come from a 12-step path inside each minute.

The first version used open/close ± a small random wiggle. That understated intrabar excursions, and on null data "against / extreme" falsely passed, because tight stops were hit too rarely. It was rebuilt before any real data was read.

**Results.**
- **`RSW_SYNTHETIC=1`** (planted continuation: drift 0.8 sd per minute in the direction of the first burst minute): all 10 "with" strategies pass (+3.3 to +4.0 ticks per trade); all "against" strategies fail.
- **`RSW_SYNTHETIC=1 RSW_SYNTH_NULL=1`**: `NO_EDGE`; every strategy nets between −0.7 and −3.4 ticks.

**Reading for real data:** tight stops are sensitive to intrabar paths. Real 1-minute OHLC carries the true extremes, but within a minute the simulation cannot know whether the high or the low came first. It checks the adverse side first, which is conservative.

## 7. Run

1. `python -m pytest -q tests/test_burst_entry.py`
2. `notebooks/MNQ_Regime_Switching/05_burst_entry.ipynb`

Fits are cached in `05/fits_<version>.pkl`. The 1-minute grid has about 1.2 million rows: expect about 20–60 minutes, mostly the WB2 fits.
