# GTF 002 — Correction C1 to GTF 01 (09:00 JST quote, commission price)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-08 |
| **Corrects** | GTF 01 run `610d790` (engine 0.1.0) |
| **Engine** | `gotobi.py` 0.2.0 |
| **Status** | Code fix. Gates, windows and the trading rule of GTF 001 are unchanged. |

## Defects

1. **09:00 JST mid missing.** `load_morning` kept ticks from 00:00 UTC only, while the 09:00 mid is the last quote *at or before* 00:00 UTC. It existed only when a tick fell exactly on 00:00:00.000, so H1 (pre-fix move) ran on 0–2 days per window and the 09:00 point of the mid-path chart was meaningless. **H1 results of run `610d790` are void.**
2. **Commission priced at the 09:55 mid.** The mid required a quote ≤ 60 s old before 09:55; on sparse 2010–2015 mornings it was missing, so those trades had no net P&L and dropped out of G3, and the reported mean commission was NaN.

## Fix

1. `load_morning` also reads the previous UTC day's 23:00–24:00 ticks (times negative relative to 00:00 UTC). The morning download (`--hours 22-2`) already contains them. Path quotes may be up to 120 s old (was 60 s).
2. Commission uses `entry_price`, the bid actually filled at 09:55 JST.

## Effect on the verdict

None possible. The video rule failed G1 and G2 on 2010–2015, which do not use commission or the 09:00 quote; the verdict is `NO_EDGE` whatever the rerun shows. The rerun updates H1, the G3 figures, the costs table and the 09:00 path point. The per-day cache is versioned, so the first rerun rebuilds it.
