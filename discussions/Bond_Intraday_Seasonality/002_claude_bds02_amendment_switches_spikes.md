# BDS 002 — Amendment to BDS 001: contract switches by signature, 1-minute spike screen

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-18 |
| **Amends** | BDS 001 §4.1 (contract switches); adds a spike screen |
| **Responds to** | Iván, 2026-11-18: fix the two data anomalies found in the BDS 001 run, then move on to bond–FX pairs |
| **Engine** | `bond_seasonality.py` 0.2.0 |
| **Scope** | Rerun for the US T-Bond (all four notebooks). Bund and Gilt keep their BDS 001 run: they showed no structure at or above noise, and the amendment only removes days. |

## Why

The BDS 001 run (commit `6b13a49`) found no candidate in any symbol or window. Two anomalies:

1. **Contract switches.** Taking the largest gap of each roll window found the switch cleanly in 2018–2019 (Bund and Gilt: a 1-hour midday break with gaps 20–35× the median). Later, several picks were only 3–6× an ordinary overnight or weekend gap, and some T-Bond picks fell on ordinary weekends (e.g. 2023-06-09, 2023-08-11). A missed switch leaves a jump of 20–70 bp at the break. Two of the four T-Bond full-window D1 survivors (UTC 21:40 → 22:40) straddle that break.
2. **Broken T-Bond quotes in 2018.** Gaps of 24–46% across breaks in February–March 2018 and two-week quote outages in June 2018 and February 2019. The coverage rule dropped most of these days, but a price spike inside a kept day would not have been caught.

## Changes

**1. Switch rule (`switch_rule="signature"`).** Dukascopy stops CFD trading 20 minutes early on the adjustment day and switches the price during the break (CFD adjustment calendar). Per roll window (10th of the month before an expiry month → end of the expiry month), the switch is the break with the largest |gap| among breaks with either signature:

| Signature | Condition |
|---|---|
| `early_stop` | the last quote before the break is 15–25 minutes before the regular close in exchange time (T-Bond 17:00 New York, Bund 22:00 Frankfurt, Gilt 18:00 London) |
| `midday` | a break of 45–90 minutes that starts after 06:00 and ends at least an hour before the close, on the same local day (the adjustment break used in 2018–2019) |

If a window has no such break, the BDS 001 rule applies (`method = fallback`). The rule for extra switch-sized breaks is unchanged. `switch_rule="largest"` reproduces BDS 001.

**2. Spike screen.** A day is dropped (in each clock, every day containing the minute) if, between consecutive quoted minutes inside a session (not across a break of ≥ 30 minutes), the mid moves

- at least **100 bps** in one minute, or
- at least **50 bps** and is reversed by at least **80%** within the next **5** quoted minutes.

Neither occurs in these futures outside data errors, except possibly in extreme events (e.g. the Gilt in September 2022). Every flagged minute is listed in §2 of each notebook, so a dropped real move is visible.

Nothing else changes: windows, gates, costs and candidate limits are those of BDS 001.

## Tests

On synthetic data with an ordinary +90 bp weekend gap in every roll window (larger than the −60 bp switch), the BDS 001 rule finds 0 of 8 switches and the signature rule 8 of 8, for all three symbols. A 300 bp one-minute spike is flagged and its day dropped.

## Reading

The amendment is decided after seeing BDS 001's results, but it only removes days (it cannot select a new interval) and is applied to the whole T-Bond scan unchanged. The previous T-Bond frozen candidate files were empty, so nothing frozen is replaced.
