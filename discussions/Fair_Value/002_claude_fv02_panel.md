# FV 002 — Point-in-time fundamentals panel, monthly 2016–2025 (FV 02)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-24 |
| **Branch** | Fair_Value |
| **Responds to** | Iván, 2026-11-24: go ahead with FV 02 on the FV 01 readout. The FV 01 readout found that dead companies have prices almost only from 2016 (Alpaca). Decisions: backtest 2016–2025; Alpaca the main price source; unadjusted Alpaca bars for market cap; Form-25 coverage check. |
| **Engine** | `fv_panel.py` 0.1.0, `alpaca.py` 0.4.0, `daily_prices.py` 0.1.2, `edgar.py` 0.1.2 |
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
