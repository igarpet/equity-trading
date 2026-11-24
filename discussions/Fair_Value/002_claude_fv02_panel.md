# FV 002 — Point-in-time fundamentals panel, monthly 2016–2025 (FV 02)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-24 |
| **Branch** | Fair_Value |
| **Responds to** | Iván, 2026-11-24: go ahead with FV 02 on the FV 01 readout. The FV 01 readout found that dead companies have prices almost only from 2016 (Alpaca). Decisions: backtest 2016–2025; Alpaca the main price source; unadjusted Alpaca bars for market cap; Form-25 coverage check. |
| **Engine** | `fv_panel.py` 0.1.1, `alpaca.py` 0.4.0, `daily_prices.py` 0.1.2, `edgar.py` 0.1.2 |
| **Status** | Data build and checks. No signal, no returns. |

## 1. FV 01 readout (run 767f00f)

- **Periodic filers.** 6,700–9,900 a year; 500–1,250 stop filing a year. 1,707 SPACs are among the filers.
- **DCF items.** Coverage near zero before 2012 (XBRL phase-in), then 84–89% for CFO, net income and shares. Revenue (60–70%), capex (~60%) and debt (~40%) are low because every filer is in the denominator and the first tag list was narrow.
- **Restatements.** 4–11% of 10-K values are later reported differently.
- **Prices, active CIKs.** 78% matched by Yahoo or Alpaca.
- **Prices, dead CIKs.** 28% matched in all. By exit year: 1–14% for 2009–2015 and 43–56% for 2016–2024, almost entirely from Alpaca. SIVB, FRC, ATVI, TWTR and CELG are in Alpaca up to delisting and missing from Yahoo. BBBY is a reused symbol (two CIKs).

**Consequences.**
- Signals and returns run 2016–2025. 2009–2015 is only DCF history; a 10-K carries three years of flows, so 5-year histories exist from about 2014.
- Alpaca is the price source for all CIKs.

## 2. Build

**Snapshots.** One row per periodic filing of a domestic us-gaap filer: what was known at its acceptance time.
- **Facts.** Facts accepted at or before the filing. A restated value replaces the old one only from its own acceptance onwards.
- **Tags.** Priority lists per item in `fv_panel.FLOW_ITEMS` and `STOCK_ITEMS`. These are wider than FV 01: revenue variants, debt split into noncurrent, current and short borrowings, leases kept apart.
- **TTM flows at the filing's period end.**
  - At a 10-K: the FY value.
  - At a 10-Q: FY_prev + YTD_cur − YTD_prev, with dates matched within ±10 days (52/53-week years).
  - This handles year-to-date 10-Q cash flows without deriving Q4.
- **History.** The last 5 fiscal years (fy0..fy4) of revenue, operating income, net income, CFO, capex and D&A; FCF = CFO − capex.
- **Balance sheet.** Items at the period end (±5 days).
- **Shares.** From the cover page (dei, dated after the period end), else us-gaap `CommonStockSharesOutstanding`.
- **Debt.** noncurrent + current + short borrowings; else `LongTermDebt` + short borrowings. No reported debt counts as 0 and is flagged.

**Prices.** Alpaca month ends, per symbol and month:
- last bar date;
- raw close, used for market cap;
- adjusted close, used later for returns;
- median daily dollar volume.

**Symbol → CIK.**
- **Windows.**
  - Prefix symbols (FSDS instance names, inline-XBRL documents): first to last filing ± 400 days.
  - Current SEC tickers: from the CIK's last use of a different prefix symbol (renames such as OSTK → BYON → BBBY) to the last filing + 400 days.
  - A Form 25 / 15 near the end closes the window 30 days after it.
- **Validity.** A bar within 7 days before the month end.
- **Conflicts.** A symbol claimed by two CIKs goes to the one whose actual filing span covers the date; otherwise the month is dropped and counted. A CIK with several symbols keeps the most traded one.

**Panel.** (CIK, month end) 2016-01 .. 2025-12.
- Uses the latest snapshot accepted strictly before the month-end date, so it is public at least one session earlier.
- Snapshots whose period ended more than 460 days earlier are not used.
- mcap = raw close × shares; EV = mcap + debt + preferred + minority interest − cash − short-term investments.
- **Flags.** Financial (SIC 6000–6799), blank check (6770), and `mcap_suspect`: mcap / public float outside [0.8, 50], meaning a wrong symbol, wrong share units or a multi-class company.

**Exits.** CIKs whose price stops before 2025-12, with the last price date, the Form 25 / 15 date, and the exit type:
- **acquired**: DEFM14A, DEFM14C, PREM14A, SC TO-T or SC 14D9 filed within 18 months before the last price;
- **delisted_other**: everything else.

## 3. Checks (`02_panel.ipynb`)

- **A.** Apple: FY2023 revenue 383.285 bn, FY2022 394.328 bn, TTM to Dec-2023 385.706 bn.
- **B.** Universe per December:
  - priced;
  - non-financial, non-SPAC;
  - mapping OK;
  - DCF-ready: TTM revenue, TTM FCF and fy4 FCF present, EV > 0;
  - EV buckets <300M, 300M–2B, 2–10B, 10–200B, >200B.
- **C.** Item coverage on the priced non-financial universe.
- **D.** Form 25 / 25-NSE filers 2016–2025 with periodic reports: share priced up to within 120 days before the form.
- **E.** mcap / public float distribution and the worst cases; share of shares from dei.
- **F.** Exits by year and type.

**Known limits of v0.1.**
- Multi-class companies (GOOG/GOOGL) take one symbol × cover shares, which can be one class only. Check E shows them.
- IFRS filers (20-F / 40-F) are left out.
- Tag choice is per (item, period) by priority. A switch of tag between years (Revenues → RevenueFromContract…, 2018) can mix definitions.

## 4. Readout of run 09a5e3f (fv_panel 0.1.0) and fixes in 0.1.1

**Build.**
- 696,581 snapshots from 14,755 CIKs.
- 10,818 symbols with raw bars.
- 533,004 CIK-months assigned to 8,670 CIKs; 14,444 ambiguous symbol-months dropped (2.7%).
- Panel: 449,110 rows, 7,203 CIKs.

**A. Apple.** All three checks match the reported figures exactly (383.285 / 394.328 / 385.706 bn).

**B. Universe per December.**
- Priced: 3,400–4,100.
- Non-financial, non-SPAC: 2,300–2,800.
- DCF-ready with 5 years: 1,250–1,800, except 934 in 2022. The 2022 dip is an artefact of check E (below).
- EV 10–200 bn: 260–400 companies; > 200 bn: 13–43.

**C. Coverage on the priced non-financial universe.**
- CFO and net income 94–96%; D&A ~89%; operating income ~88%; revenue ~84% (pre-revenue biotech is part of the gap); capex 80–88%; interest expense ~70%.
- fy4 FCF 73–85%.
- Shares 100%, but 9.3% of panel rows have shares "none": not priced into an mcap.

**D. Form 25 coverage.**
- 74% overall, but 84–92% in 2016–2020 and 53–61% in 2023–2025.
- The denominator included SPAC liquidations (2021–2023 wave), which never enter the universe. 0.1.1 reports operating companies separately.

**E. The mapping check was wrong, for two reasons.**
1. **Multi-class companies.** companyfacts strips the class dimension, so the last class read won. A class B with 0 shares gave mcap 0 (CRWD, DDOG, LYFT, TOST); a class with float 0 gave an infinite ratio (GOOG).
2. **Timing.** Public float is valued at the prior 30 June, so mcap / float falls below 0.8 whenever the price drops after that date. The whole of 2022 was flagged; the median ratio of suspects is 0.61.

Also seen: warrant symbols (ZCARW) chosen over the common stock.

**F. Exits.** Acquired: 118–208 a year. Delisted-other jumps to 398 / 441 / 980 in 2020–2022, consistent with SPAC liquidations. 0.1.1 splits exits by SPAC.

**Fixes in 0.1.1** (snapshots must be rebuilt; the version check enforces it):
- **Cover-page share classes.** Shares outstanding and public float are summed across classes within the same filing and date; exact duplicates are counted once.
- **Shares fallback.** dei → us-gaap `CommonStockSharesOutstanding` → latest-quarter weighted diluted (else basic) shares. The 0.1.0 TTM of diluted shares was meaningless and is removed.
- **Mapping check on share counts.** Shares outstanding ÷ (public float ÷ raw close at the float date), flagged outside [0.8, 50]. Float ≤ 0 (new IPOs) is not checked.
- **Warrants, units and rights** (…W, …WS, …U, …R, …RT) are dropped when the CIK's base symbol is also a candidate.
- **D and F** reported separately for operating companies and SPACs.

## 5. Readout of run 5b4be3e (fv_panel 0.1.1)

**A. Apple.** Unchanged, exact.

**B. DCF-ready universe per December.** 1,960–2,120 in 2016–2020, 1,610–1,880 in 2021–2025; the 2022 dip has gone.
- **By EV:** <300M 260–620; 300M–2B 400–650; 2–10B 480–560; 10–200B 315–440; >200B 15–45.

**C.** Unchanged: CFO and net income 87–97%, revenue ~84%, capex 77–88%, fy4 FCF 71–84%.

**D. Form 25 coverage, operating companies (1,992 filers).** 75.1% overall: 84–92% in 2016–2020, 74–76% in 2021–2022, 55–62% in 2023–2025. The recent decline is not SPACs (SIC 6770). New check G measures it on the companies that matter.

**E. Mapping.** 6.7% of rows flagged, down from 26%. Median shares / float shares 1.19. The remaining extremes (10⁻⁷ … 10⁹) are mostly:
- **wholly owned subsidiaries filing their own 10-K** under the parent's symbol: old Dow Chemical → DOW, EIDP → DD;
- **cover-page scale errors**, with shares or float tagged in the wrong unit (ManpowerGroup, Hexcel, Zimmer at about 10⁻⁶; Brookdale, AEHR at 10⁸–10⁹).

Flagged rows are excluded from the studies.

**F. Exits, operating companies.** Acquired: 119–206 a year. Delisted-other: 105–163 a year, rising to 394 / 432 / 806 in 2020–2022. That spike is probably SPACs filed without SIC 6770, which F cannot separate. Check G restricts exits to companies that were DCF-ready.

**Added (notebook only, no rebuild).**
- **G1.** Form 25 coverage among companies that had, within 24 months before the form, a filing with a 5-year FCF history, revenue > 0, non-financial, and public float ≥ $100M. Reported by year and by float bucket, with the uncovered companies and their Alpaca spans.
- **G2.** Exits among companies that were DCF-ready in the panel within 24 months before their last price.
