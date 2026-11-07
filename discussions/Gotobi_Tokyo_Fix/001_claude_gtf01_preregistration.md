# GTF 001 — USD/JPY gotobi days and the Tokyo fix (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-08 |
| **Branch** | Gotobi_Tokyo_Fix (new) |
| **Source** | Ito, T. & Yamada, M. (2017), *Journal of International Money and Finance*: USD/JPY rises into the 9:55 JST Tokyo fix on gotobi days. Trading rule from a YouTube backtest (2026) claiming +2.35 pips per trade after spread, 775 trades 2016-01 → 2026-10, t = 5.5. |
| **Responds to** | Iván, 2026-11-07: test the gotobi rule independently |
| **Status** | Pre-registration. Written before any run on real data. Gates are not changed after results exist. |

## 1. Claims to test

**H1 (paper, pre-fix).** On gotobi days the USD/JPY mid rises from 09:00 to 09:55 JST more than on other Japanese bank days.

**H2 (video rule, post-fix).** On a gotobi day: sell USD/JPY at 09:55 JST, stop 12 pips above the fill, buy back at 10:30 JST. The trade earns a positive mean after the quoted spread, the effect is specific to the gotobi calendar, and it survives commission.

## 2. Data and windows

| Item | Value |
|---|---|
| Ticks | Dukascopy bid/ask (`tools/dukascopy/dukascopy_download.py`), Parquet per UTC day, 00:00–02:00 UTC used. 09:55 JST = 00:55 UTC (Japan has no DST). |
| **INDEPENDENT** | 2010-01-01 → 2015-12-31. **Binding.** Not used by the video. It may overlap the paper's own sample; if so, it is independent of the video's post-fix rule, not of the paper's pre-fix finding. |
| REPLICATION | 2016-01-01 → 2025-12-31. The video's sample without 2026. |
| Sealed | 2026-01-01 onward (code refuses it). One frozen-spec look later, in a separate turn. |

Venue caveat: these are Dukascopy SWFX quotes, not EBS/interbank. Fills and spreads at the fix are venue-specific.

## 3. Specification (frozen)

- **Bank day:** Mon–Fri, not a Japanese national holiday (`jpholiday`, includes substitute and special holidays), not Dec 31 – Jan 3.
- **Gotobi day (video definition):** calendar days 5, 10, 15, 20, 25, 30 (February has none on the 30th), each moved to the previous bank day if not one. Duplicates collapse.
- **Control days:** bank days that are not gotobi days. Japanese holidays on weekdays are reported separately (no fix is set).
- **Short trade:** sell at the first **bid** at or after 00:55:00 UTC; stop when the **ask** reaches fill + 12 pips (filled at that ask, gaps included); otherwise buy at the first **ask** at or after 01:30:00 UTC. A fill more than 30 s after its scheduled time voids the day (counted, not traded). 1 pip = 0.01 JPY.
- **Pre-fix (H1):** mid at 00:55 minus mid at 00:00 UTC, in pips (last quote ≤ 60 s old). A tradable long (buy ask 00:00, sell bid 00:55) is reported, not gated.
- **Costs:** the spread is in every fill. Commission, round trip, in pips = 2 · c · price / 10,000 with c in USD per $1M per side. Gate level **c = 35** (Dukascopy Europe, deposit < $5k), plus a 2× stress (c = 70).
- **Sizing (reported only):** risk 0.5% of equity per trade, return = 0.005 · net pips / 12.

## 4. Gates (per window; α = 0.025 one-sided)

| Gate | Condition |
|---|---|
| **G1 statistical** | Gotobi short-trade P&L after spread, before commission: mean > 0 with HAC (5 lags, trade order) one-sided p ≤ α, **and** moving-block bootstrap lower bound (quantile α) > 0 at blocks 10 and 20 trades |
| **G2 calendar** | Gotobi mean − control mean > 0 with Welch one-sided p ≤ α, **and** the gotobi mean exceeds 10,000 random calendars (same number of days per month, drawn from that month's bank days with a valid trade): upper-tail p ≤ α |
| **G3 economic** | Net at c = 35: bootstrap lower bound > 0 at blocks 10 and 20, **and** mean net at c = 70 > 0 |

H1 is tested with the G2 construction on the pre-fix mid move (Welch gotobi − control, p ≤ α) in each window and reported as `H1_REPLICATES` / `H1_NOT_REPLICATED`. It does not enter the trading verdict.

**Verdicts.**
- REPLICATION window: `REPLICATES` if G1 and G2, else `NOT_REPLICATED`.
- INDEPENDENT window: `CONFIRMED` if G1 and G2, else `NOT_CONFIRMED`.
- Final: **`EDGE_CANDIDATE`** if INDEPENDENT is `CONFIRMED` and G3 passes in **both** windows; otherwise `NO_EDGE`.

## 5. Reported, not gated

Per-year means and win rates; average mid path 09:00–10:45 JST (gotobi vs control); shifted calendars (±1, ±2, ±3 bank days); the month-end convention (`gotobi_m`); month-end gotobi vs the 5th–25th; weekday; MoF intervention days excluded (list in code, to be verified); grid of entry 09:54/55/56, exit 10:15/10:30/10:45, stop 8/12/20/none; extra slippage 0.2 and 0.5 pips per side; break-even commission; equity curve and max drawdown at 0.5% risk.

None of these can rescue a failed gate. A grid cell that beats the base rule is a hypothesis for a new pre-registration, tested on the 2026 holdout only.

## 6. Readings fixed in advance

- **EDGE_CANDIDATE:** freeze the spec; one look at 2026 (Dukascopy, Jan → latest) after Iván approves; then decide the execution venue (spot broker vs CME yen futures, which need their own cost check).
- **REPLICATES but NOT_CONFIRMED:** the effect is a post-2016 phenomenon or a sample artefact. No trading.
- **CONFIRMED but G3 fails:** the anomaly exists but does not pay retail costs at Dukascopy.
- **NOT_REPLICATED:** the video's result does not reproduce on Dukascopy quotes; check the venue caveat before drawing conclusions about the anomaly itself.

## 7. Run order

1. `pip install -r requirements.txt` (adds `jpholiday`), then `python -m pytest -q tests/test_gotobi.py`.
2. `notebooks/Gotobi_Tokyo_Fix/01_gotobi_fix_backtest.ipynb`. First run reads ~4,100 day files (minutes) and caches the per-day outcomes; 10,000 random calendars per window take seconds.

`GTF_SYNTHETIC=1` runs it offline on synthetic ticks with an injected effect (outputs to `01_synthetic/`, ignored by git).
