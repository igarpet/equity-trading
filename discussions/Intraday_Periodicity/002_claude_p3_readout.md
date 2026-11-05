# P3 002 — Readout (run 48a18b5)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-05 |
| Protocol | `001_claude_p3_intraday_periodicity_preregistration.md` |
| Verdicts | 01: **`REPLICATES`** (S1 and S2 pass in both windows). 02: `NO_EDGE` (net negative by two orders of magnitude). |

## 1. Notebook 01: the periodicity is real, in both windows

| | A 2020–2025 | B 2017–2019 |
|---|---|---|
| Stocks / full sessions | 501 / 1,496 | 480 / 746 |
| S1: mean slope at same-slot lags (13…520) | 0.00097, t = 4.9 | 0.00119, t = 5.6 |
| S2: same-slot minus other lags | 0.00088, t = 4.4 | 0.00139, t = 6.4 |
| Slope at lag 1 (next half-hour) | −0.0078 | −0.0117 |
| Slope at lag 13 / 26 | 0.0043 / 0.0034 | 0.0042 / 0.0081 |

As in the paper: half-hour returns reverse into the next half-hour (negative lag-1 slope), and continue into the same half-hour on later sessions. This is the first effect in the project that holds, as pre-registered, in both 2020–2025 and 2017–2019.

## 2. Notebook 02: untradable alone

| Window, lookback 20 | Gross / day | Cost / day | Net / day | Gross / interval |
|---|---|---|---|---|
| A 2020–2025 | 9.9 bps | 252 bps | −242 bps | 0.76 bps |
| B 2017–2019 | 6.3 bps | 222 bps | −216 bps | 0.48 bps |

- **E1.** Gross is positive in A (bootstrap lower bound 2.7–4.4 bps/day). In B the lower bound touches zero at blocks 20 and 40, so B fails.
- **E2.** Net fails by a factor of about 25. The book turns over about 44 units of weight a day: the decile membership changes almost completely each half-hour.
- **Lookbacks 1, 5 and 40** (reported, not judged) give the same picture: gross 4–12 bps/day, net −200 to −240.

**Where the gross sits (reported).** The gross is concentrated in the first (09:30) and last (15:30) intervals: 6.7 and 2.8 bps in A, 3.8 and 4.9 in B. The middle of the day is near zero. A post-hoc version that trades only those two slots would still pay a full round trip on both legs for about 3–7 bps gross. At 2–3 bps one-way it does not clear cost either; it is not pursued as a strategy.

## 3. Consequence (reading fixed in advance)

`REPLICATES` + `NO_EDGE`: the pattern is real but too small to trade alone, so it becomes a **control variable for other branches**, not a strategy. Concretely:
- Studies that rank stocks intraday (S05-type signals, future P5/P6) should check that their signal is not partly a same-slot or open/close-slot effect.
- The negative lag-1 slope is the 30-minute version of the reversal the S05 residual trades. It is real in both windows, but P2-04 showed that the sector-residual version of it did not pay before 2020.

P3 closes.
