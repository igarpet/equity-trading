# GTF 003 — Long into the fix, reverse into the short (GTF 02 pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-08 |
| **Branch** | Gotobi_Tokyo_Fix |
| **Responds to** | Iván, 2026-11-08: long for the 30 minutes before the fix, then switch to the GTF 01 short |
| **Builds on** | GTF 001 (rule, calendar, costs), correction C1 (engine 0.2.0) |
| **Status** | Pre-registration. Written before any GTF 02 run. Gates are not changed after results exist. |

## 1. Rule

On a gotobi day (GTF 001 definition):

1. **Long leg:** buy at the first **ask** at or after 09:25:00 JST (00:25 UTC); sell at the first **bid** at or after 09:55:00 JST. No stop.
2. **Short leg:** GTF 001 unchanged: sell at that same 09:55 bid tick, stop at fill + 12 pips on the ask, else buy back at the first ask at or after 10:30 JST.

In practice this is one order of two units at 09:55. A day counts only if both legs fill within 30 s of schedule. Each leg pays its own spread and its own round-trip commission (2 · c · price / 10,000 pips, priced at that leg's entry fill); reversing saves nothing.

## 2. What is already known (disclosed)

Both 2010–2015 and 2016–2025 have been seen. GTF 01's mid-path chart showed a gotobi rise from 09:30 to 09:55 of +1.44 pips (2010–2015) and +1.92 (2016–2025), against +0.69 and +1.27 on other bank days. This idea came partly from that chart, so **no window before 2026 can confirm it**. They are run with the gates below and labelled **EXPLORATORY**.

Pre-2010 data, the only unseen history, is not used: Iván's view (2026-11-08) is that the effect depends on payment practices of recent years. Note that GTF 01 already found the post-fix short absent in 2010–2015.

## 3. Windows

| Window | Dates | Role |
|---|---|---|
| EXPLORATORY-A | 2010–2015 | reported with gates |
| EXPLORATORY-B | 2016–2025 | reported with gates; part of the final verdict |
| **HOLDOUT** | 2026-01-01 → last downloaded day | **binding, one look** with the spec frozen here |

The holdout needs a separate download (`--start 2026-01-01 --hours 22-2`) and is opened in the notebook only when `OPEN_HOLDOUT = True`. About 60 gotobi days are available up to early November 2026.

**Power.** GTF 01 per-trade standard deviation was ~12.6 pips. With ~60 trades the standard error is ~1.6 pips, so the holdout detects only a mean of roughly 2.7 pips or more at α = 0.05. A holdout fail is weak evidence against; a pass is weak evidence for.

## 4. Gates

**Exploratory windows** (combined P&L; α = 0.025 one-sided), same construction as GTF 001:
- **G1:** after spread, HAC p ≤ α and block-bootstrap lower bound > 0 (blocks 10, 20).
- **G2:** gotobi − control (Welch) p ≤ α and random calendars p ≤ α.
- **G3:** net at c = 35 bootstrap lower bound > 0 (10, 20) and mean net at c = 70 > 0.

**Long-leg value (L), exploratory windows.** The long leg alone, net of its own commission at c = 35: mean > 0 with HAC p ≤ α. If L fails in 2016–2025, the combined trade is the GTF 01 short plus a cost and is not preferred.

**Holdout (α = 0.05, one-sided):**
- **HG1:** combined mean after spread > 0, HAC p ≤ 0.05.
- **HG2:** combined gotobi − control > 0, Welch p ≤ 0.05.
- **HG3:** combined mean net at c = 35 > 0.

**Verdict.** `EDGE_CANDIDATE` if G1–G3 pass on EXPLORATORY-B **and** HG1–HG3 pass on the holdout; `NO_EDGE` otherwise. EXPLORATORY-A is reported only (it already failed for the short in GTF 01). Separately: `LONG_ADDS_VALUE` if L passes on EXPLORATORY-B and the long leg's holdout mean net is > 0.

## 5. Reported, not gated

Long entry at 09:00 / 09:15 / 09:25 / 09:35 / 09:45 JST (exit always 09:55); by-year long, short and combined; break-even commission; extra slippage 0.2 / 0.5 pips per fill; equity at the GTF 01 sizing (same notional both legs). None of these can change a verdict.

## 6. Run order

1. `python -m pytest -q tests/test_gotobi.py`
2. Rerun `01_gotobi_fix_backtest.ipynb` (C1).
3. Run `02_long_into_fix_reversal.ipynb` with `OPEN_HOLDOUT = False`.
4. Download 2026: `python tools/dukascopy/dukascopy_download.py --symbol USDJPY --start 2026-01-01 --end <last full day> --hours 22-2`.
5. Set `OPEN_HOLDOUT = True` once, run the holdout section, commit the result. No further changes to the rule after that look.
