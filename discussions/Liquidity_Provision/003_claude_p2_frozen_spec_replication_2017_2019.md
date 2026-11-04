# P2 003 — VIX-gated S05: frozen specification and replication on 2017–2019 (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-04 |
| Follows | `002_claude_p2_readout.md` (notebook 03: `EDGE_CANDIDATE` on 2020–2025) |
| Ivan's choice | Replicate on 2017–2019 at 15 minutes before using the 2026 holdout |
| Status | Pre-registration. Written before any 2017–2019 15-minute data or quotes are downloaded. |

## 1. Frozen specification

Nothing below is changed for this test or for the 2026 holdout.

| Element | Value |
|---|---|
| Engine | `s05.py` 0.1.3, `S05Config` defaults except the exits |
| Signal | Each stock vs its GICS sector Select Sector SPDR; OLS on the trailing 20 sessions of 15-minute log returns; AR(1) residual s-score, cross-sectionally centred mean |
| Entries | $s < -1.25$ long, $s > +1.25$ short; 1% of equity per stock against β × that notional in the ETF; at most 100 positions, largest $\|s\|$ first |
| Exits | `x10`: close a long at $s > +1.0$, a short at $s < -1.0$; positions also close when the stock's estimate becomes invalid |
| Execution | Signal at a bar close, fill at the next bar open |
| Regime gate | New entries on session $t$ only if VIX (Yahoo `^VIX` close) at session $t-1$ exceeds its median over the 250 sessions ending at $t-1$ (at least 200) |
| Costs | S05b method: per-symbol, per-year NBBO half-spread × intraday slot profile, from SIP quote samples (4 sessions per year, seed 42, first 3 s after each bar start) |

## 2. Data for the replication

| Item | Value |
|---|---|
| Bars | Alpaca SIP `15Min`, `adjustment=all`, **2017–2019**. Kept in a separate panel cache, so the shared 2020–2025 cache is not touched. |
| Costs | **New quote samples for 2017–2019** (the sampled sessions are drawn from the calendar with seed 42 by the S05b procedure). |
| Universe | Current S&P 500 snapshot with current GICS sectors (survivorship bias larger than in 2020–2025) |
| Known gap | XLC starts in June 2018. Communication Services stocks have no hedge before then; they enter once XLC has 20 sessions. This follows from the frozen rule (invalid estimate, no trade), not from a choice. |

Trading starts after the first 20 sessions of 2017.

## 3. Gates

Same as notebook 03 ($\alpha = 0.05$):

| Gate | Condition |
|---|---|
| V1 | In the ungated run, high-regime minus low-regime mean gross per trade > 0 (bootstrap over entry days, 95% lower bound > 0) |
| V2 | The gated arm's net total return and net Sharpe beat 200 placebos with the regime series circularly shifted by at least 60 sessions (upper-tail p ≤ 0.05 on both) |
| V3 | Gated net total return > 0 at 1× and at 2× cost; bootstrap lower bound of the mean daily net return > 0 at blocks 10, 20 and 40 |

Verdict: **`REPLICATED`** if V1, V2 and V3 all pass; otherwise `NOT_REPLICATED`.

- `REPLICATED`: the frozen specification goes to the 2026 holdout (one use) unchanged.
- `NOT_REPLICATED`: the candidate is not taken to the holdout, which stays sealed. The 2020–2025 result is then read as specific to that period.

Reported, not judged: per-year results; ungated vs gated; gross per trade by regime.

## 4. Power

There are three years against six, and fewer high-VIX sessions in 2017 (a very calm year). A shorter sample can fail V3 for lack of power even if the effect is real. That outcome counts as `NOT_REPLICATED`; the gates are not relaxed for the shorter sample.

## 5. Run

Notebook `04_vix_gated_replication_2017_2019.ipynb`. The first run downloads 15-minute bars for 2017–2019 (about 520 symbols × 3 years, comparable to one third of the 2020–2025 download) and 12 quote-sample sessions. Then 2 × 3 backtests and 200 placebos. `P2_SYNTHETIC=1` smoke-tests it offline.
