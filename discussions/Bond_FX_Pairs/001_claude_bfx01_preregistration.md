# BFX 001 — Bond–FX pairs on 5-minute bars: lead–lag, residual reversion, cointegration (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-18 |
| **Branch** | Bond_FX_Pairs (new) |
| **Responds to** | Iván, 2026-11-18: no structure in the bonds on their own; build all three bond–FX approaches (lead–lag, residual reversion, cointegration of levels) on 5-minute bars |
| **Engine** | `bond_fx.py` 0.1.0 (on `bond_seasonality.py` 0.2.0 and `fx_seasonality.py` 0.2.0) |
| **Status** | Written before any real bond–FX data has been analysed. Gates are not changed after results exist. |

## 1. Question

Does the bond market of a currency pair's two countries carry tradable information about the pair at intraday horizons, either because bonds move first (lead–lag), because the pair temporarily deviates from its bond-implied move and comes back (residual reversion), or because the pair and the bonds share a long-run level (cointegration)?

## 2. Pairings and data

| Pairing | Bonds | Expected sign of the FX–bond link |
|---|---|---|
| EURUSD | Bund, US T-Bond | Bund up (EUR yields down) → EURUSD down; T-Bond up → EURUSD up |
| GBPUSD | Gilt, US T-Bond | Gilt up → GBPUSD down; T-Bond up → GBPUSD up |
| USDJPY | US T-Bond | T-Bond up (US yields down) → USDJPY down (no JGB CFD exists) |
| EURGBP | Bund, Gilt | Bund up → EURGBP down; Gilt up → EURGBP up |

Signs are not imposed: every hedge ratio is estimated.

- Dukascopy 1-minute bid/ask candles; the FX pairs from the FXS download, the bonds from the bond download.
- **5-minute bars**, labelled by their end time, at the last quote. A bar is used only if **every leg was quoted in it**, no leg's UTC day is a missing file or a spike day (BDS 002 screen), and it is not Saturday. Consecutive used bars 5 minutes apart form a *segment*; FX-only methods never hold a position across a segment end.
- Bond prices are made **continuous** across the quarterly contract switches (BDS 002 signature rule): each switch gap is removed from all later prices, so levels and returns are those of one rolling futures position.
- 2026 is sealed.

## 3. Windows

As BDS 001: **H1** 2018–2021, **H2** 2022–2025, **FULL** 2018–2025. Each method is scanned on each window; H1 candidates are confirmed on H2 (forward), H2 candidates on H1 (backward); FULL is in-sample and only matched to the halves. Rolling estimates use only data before the bar they are applied to; the first weeks of each window are therefore spent warming up.

## 4. Methods and configuration grids

All trades are non-overlapping per pairing and configuration (a new entry only after the previous exit).

### 4.1 Lead–lag (`leadlag`, FX leg only) — 24 configurations

- β: no-intercept OLS of 5-minute FX returns on the bond returns over the trailing 20 days of bars, ending at the previous bar.
- Bond-implied FX move over the last L bars: `implied = β · ΔX(L)`; z = implied / its trailing 20-day standard deviation.
- Entry when |z| > k, in the direction of `implied`; exit after H bars (or at the segment end).
- Grid: L ∈ {1, 3, 6}, H ∈ {1, 3, 6, 12}, k ∈ {1.5, 2.5}.
- Descriptive: predictive regression of the next-H-bar FX return on `implied` and on the FX pair's own last-L move (Newey–West, H lags), L, H ∈ {1, 3, 6, 12}.

### 4.2 Residual reversion (`residual`, FX leg only) — 12 configurations

Avellaneda–Lee: residual ε = FX return − β · bond returns (β over the trailing W days, ending at the previous bar), cumulated into X; AR(1) of X on the trailing W days; m = a/(1−b), σ_eq = √(Var(η)/(1−b²)), s = (X − m)/σ_eq. Valid only if 0 < b and the half-life ≤ W/2.

- s > s_open → short FX; s < −s_open → long. Exit when |s| < 0.5, after `hold` bars, or at the segment end.
- Grid: W ∈ {1, 5} days, s_open ∈ {1.5, 2.0, 2.5}, hold ∈ {12, 48} bars (1 or 4 hours).
- Reported, not gated: the hedged P&L (FX minus β × bonds, at the mid).

### 4.3 Cointegration of levels (`coint`, both legs) — 6 configurations

- Each trading day: OLS of log FX on log bond prices (with constant) over the previous F days, sampled every 12th bar (hourly); statsmodels `coint` (Engle–Granger with MacKinnon p-values, ADF with a fixed 2 lags; AIC lag selection was 20 times slower). The day is traded only if p < 0.05.
- z = (log FX − α − β · log bonds) / residual sd. Entry when z_open < |z| < 4: z > 0 → short FX, long β bonds; z < 0 → the opposite. Parameters fixed at entry. Exit when z crosses 0, |z| ≥ 4 (stop), or after 5 days. At most one entry per day.
- Grid: F ∈ {20, 60} days, z_open ∈ {1.5, 2.0, 2.5}.

## 5. Costs

- FX leg: Dukascopy's quoted bid/ask (long buys the ask, sells the bid) + **0.70 bp** commission per round trip.
- Bond legs (cointegration only): per unit of |β|, **one futures tick** (T-Bond 1/32, Bund and Gilt 0.01) + 0.70 bp.
- P&L in bps of FX notional; daily P&L = sum of trades by exit day, 0 on days without an exit.

## 6. Discovery (each method × window; all pairings and configurations in one family)

| Filter | Condition |
|---|---|
| **E1** | at least 100 trades |
| **E2** | Benjamini–Hochberg over the method's pairing × configuration tests (96, 48 or 24), one-sided Newey–West p (5 lags) of the daily net mean > 0, q ≤ 0.05 |
| **E3** | daily net mean > 0 in ≥ 75% of counted years (years with ≥ 120 days) |
| **E4 / D0** | per pairing: the best t over the pairing's configurations exceeds the best t of every one of 5 placebos (`STRUCTURE_FOUND`). Placebo: the pairing's bond series shifted circularly by a random amount (≥ 40 days) against the FX series, full pipeline rerun |
| **Candidate** | per pairing passing E4, the configuration passing E1–E3 with the largest t (at most 4 per method and window) |

Selecting the best configuration per pairing after BH inflates its t; the confirmation on the other half is the binding test.

## 7. Cross-confirmation

The frozen candidate (pairing, configuration) is run unchanged on the other half:

| Gate | Condition |
|---|---|
| **K1** | daily mean after spread (before commission) > 0: one-sided Newey–West p, BH across the window's candidates of that method, q ≤ 0.05 |
| **K2** | daily net: moving-block bootstrap lower bound (0.025) > 0 at blocks of 10 and 20 days |
| **K3** | daily net at 2× commission > 0 |
| **K4** | daily net > 0 in ≥ 60% of counted years |

Verdicts: `CONFIRMED_FORWARD`, `CONFIRMED_BACKWARD`, `NOT_CONFIRMED`.

## 8. Readings fixed in advance

- Strongest result: a candidate confirmed in at least one direction whose pairing and configuration also pass in FULL. It goes to a 2026 holdout pre-registration.
- Lead–lag or residual candidates that only work at the mid (`mean_trade_mid_bps` > 0, net < 0) are real but not tradable at Dukascopy's FX spread; they would need prop-firm FX costs, recomputed separately.
- A cointegration candidate depends on bond execution at one tick, which is not available on the FX prop accounts.
- No candidate is modified after its frozen file is committed.

## 9. Not tested here

Other bond tenors (2-year yields drive FX more than long bonds), yield-curve slope factors, conditioning on data releases, position sizing, overnight swap costs on multi-day cointegration trades.

## 10. Run order

1. `python -m pytest -q tests/test_bond_fx.py`
2. `00_data.ipynb` (coverage, correlations by hour, rolling hedge ratios, predictive map): descriptive.
3. For each method folder (`LeadLag/`, `Residual/`, `Coint/`): `01_discovery_H1`, `02_discovery_H2`, commit `01/` and `02/`; then `03_cross_confirmation`; then `04_full_window`.

`BFX_SYNTHETIC=1` runs everything offline on synthetic data with one injected relation per method (USDJPY lead–lag, GBPUSD residual reversion, EURUSD cointegration; EURGBP independent). Each synthetic relation needs the bonds to be seen, so it must disappear in the placebo.

**Why the placebo matters here.** Residual and cointegration trades can profit from the FX pair's own short-term mean reversion even when the bonds add nothing. Shifting the bonds in time keeps that own reversion and removes the bond link, so E4 measures what the bonds add. It is compared per pairing on the best t, not on a count of passing configurations: in the synthetic test a pairing's own reversion alone passed E1–E3 in every configuration (count saturated), while its best t fell from about 20 to about 9 without the bonds.
