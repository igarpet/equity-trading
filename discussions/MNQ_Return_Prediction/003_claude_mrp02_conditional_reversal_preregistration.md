# MRP 003 — Conditional reversal after large MNQ bars (MRP 02, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-21 |
| **Branch** | MNQ_Return_Prediction |
| **Responds to** | Iván, 2026-11-21: the lag-2 reversal of MRP 01 averages below cost, but he sees reversals after moves repeatedly and wants small, consistent wins. Is the reversal concentrated after large bars, in a session, or with a delayed entry, enough to clear the round trip? |
| **Engine** | `reversal.py` 0.1.0 with `mnq_predict.py` 0.1.1 (grid, volatility model) |
| **Status** | Written after MRP 01. Discovery reuses the MRP 01 training window, which has been seen only through the linear regression; no conditional or event statistic has been computed. Gates are not changed after results exist. |

## 1. Question

After a 5-minute MNQ bar that is large relative to its expected volatility, does a trade against that bar, entered at market and held a few bars, earn more than its cost in ticks, consistently across years?

All moves, costs and results are in **ticks** (0.25 points, $0.50 per contract) and **dollars per contract**.

## 2. Data and bars

- MRP 001 grid and exclusions: 5-minute CME-day grid; no target in the first 3 bars of a CME day; no return across gaps or rolls; CME days with fewer than 220 bars are excluded.
- **Size** of bar t: z_t = r_t / σ̂_t, where σ̂_t is the MRP 001 volatility forecast (time of day × HAR) made at the close of bar t − 1.
  - Discovery: 2020 uses a fit on 2020 itself (in sample; it only sets which bars count as large). 2021-01 – 2023-06 uses the walk-forward fits made before 2021-01-01, 2022-01-01 and 2023-01-01.
  - Confirmation: fits made before 2023-07-01, 2024-01-01 and 2025-01-01.
- MRP 001 found the level of σ̂ biased upward in calm periods (V2), so a fixed k selects fewer bars in calm years. That is accepted and reported (events per year).

## 3. Rule and grid

**Event**: bar t with |z_t| ≥ k, on an eligible bar (it has a return, slot ≥ 3, normal day).

**Trade**:
- direction −sign(r_t);
- entry at the close of bar t + d;
- exit at the close of bar t + d + h;
- one contract.

The holding window must stay inside the same CME day and contract, with no gap. While a trade of a configuration is open, new events for that configuration are skipped (no overlap).

| Parameter | Values |
|---|---|
| k (size threshold, in σ̂) | 1.5, 2, 2.5, 3, 4 |
| d (entry delay, bars) | 0 (at the close of the large bar), 1 (one bar later: the MRP 01 lag-2 reversal) |
| h (holding, bars) | 1, 2, 3, 6 |
| session of the event bar | all, Asia, London, NY, post-close (MRP 001 §5 definitions, local clocks) |

That is 5 × 2 × 4 × 5 = **200 configurations**.

**Gross move per trade** = −sign(r_t) × (close at exit − close at entry) / 0.25, in ticks.

**Costs per round trip** (one contract):
- **market (verdict)**: 1 tick + 2 × commission; commission = Lucid MNQ rate per side (Iván confirms; the default parameter is $0.50, giving 3 ticks);
- **realized (reported)**: 1.6 ticks ($0.80, Iván's stated experience on small moves, which includes slippage).

## 4. Discovery (2020-01-01 – 2023-06-30)

Per configuration:

- **D1**: at least 150 trades.
- **D2**: mean net ticks at market cost > 0, with one-sided p from a t-statistic clustered by CME day; Benjamini–Hochberg q ≤ 0.05 across the 200 configurations.
- **D3**: mean net ticks at market cost > 0 in at least 3 of the 4 periods (2020, 2021, 2022, 2023H1).
- **D4 (both sides)**: mean gross ticks > 0 separately for longs (after down bars) and shorts (after up bars). This guards against a trend or drift that favours one side.

Candidates = configurations passing D1–D4. They are frozen to `04/candidates.csv` with its SHA-256 recorded in `04/summary.json` before the confirmation notebook is opened.

Also reported, descriptively and without selection:
- **Size curve**: mean gross reversal ticks for d ∈ {0, 1}, h = 1, by |z| bin (1–1.5, 1.5–2, 2–2.5, 2.5–3, 3–4, ≥ 4) and by session.
- Per configuration: win rate, mean win, mean loss, 5% and 1% trade quantiles, worst trade and maximum drawdown in ticks per contract. Iván wants consistent small wins; the tail shows what they cost.

## 5. Confirmation (2023-07-01 – 2025-12-31, one look)

For each frozen candidate:

- **C1**: at least 75 trades.
- **C2**: mean net ticks at market cost > 0, with a day-block bootstrap (5-day blocks, 5,000 draws) 95% lower bound of total net > 0.
- **C3**: net > 0 in at least 2 of the 3 periods (2023H2, 2024, 2025).

A candidate passing C1–C3 goes to a 2026-holdout pre-registration (prop-account sizing, Lucid drawdown rules). If no candidate passes D, the confirmation notebook does not open the test window.

## 6. Readings fixed in advance

- MRP 01's linear reversal implies about 0.9 ticks after a 1σ bar and 2.7 ticks after 3σ (h = 1, d = 1). Below 3 ticks at market cost, no linear extrapolation clears the cost. A candidate needs the reversal after large bars to be larger than linear.
- A configuration that clears 1.6 ticks but not the market cost is reported as "only with passive or better-than-market execution". That is a reason for a quote-level (Databento `tbbo`) execution study, not a pass.
- Many configurations share events (nested k, overlapping h), so passing configurations are not independent evidence. The size curve is the primary description.

## 7. Run order

1. `python -m pytest -q tests/test_reversal.py`
2. `04_reversal_discovery.ipynb` → commit `04/`.
3. Iván confirms the Lucid commission in `05_reversal_confirmation.ipynb`, then runs it once.

`MRP_SYNTHETIC=1` runs both notebooks on synthetic data with a planted reversal after bars beyond 3 true sd (24 ticks over 2 bars, entry at d = 1).
