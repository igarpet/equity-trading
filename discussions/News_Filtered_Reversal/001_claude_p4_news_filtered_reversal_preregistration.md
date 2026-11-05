# P4 001 — News-filtered reversal: is reversal stronger once industry moves and news are removed? (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-05 |
| Branch | News_Filtered_Reversal (paper P4) |
| Source | Da, Z., Liu, Q. & Schaumburg, E. (2014), *A Closer Look at the Short-Term Return Reversal*, Management Science 60(3), 658–674 (NY Fed SR 513) |
| Ivan's choices | News from SEC EDGAR earnings filings (primary) and Alpaca / Benzinga news (secondary); check ForexFactory and FinancialJuice (section 3) |
| Working rule | Both 2017–2019 and 2020–2025 are judged (P2 004) |
| Status | Pre-registration. Written before any EDGAR or news data is downloaded. |

## 1. Claim and what we can test

The paper decomposes the past return into an industry part, a cash-flow-news part (from I/B/E/S analyst forecast revisions) and a residual. It finds that the reversal is much stronger in the residual than in the raw return.

We have no analyst revisions. Cash-flow news is proxied by **event windows**: stock-days around earnings announcements and around firm-specific news shocks are removed from the sort. This is a proxy design, not a replication of the decomposition; the verdict is about the proxy.

## 2. Prior

- **P1/P2:** the daily reversal in this universe is weak (Lehmann portfolio gross ≈ 1.6 bps/day 2020–2025, net negative).
- **P2-04:** the 15-minute residual signal had no gross edge in 2017–2019.
- **P3:** the 30-minute lag-1 reversal is real in both windows, but small.

Removing earnings windows takes out only about 4 × 3 = 12 stock-days per stock per year (≈ 5%), so a large change in the averages is not expected. The test is whether the filtered names reverse *more*.

## 3. Sources checked at Ivan's request (not used)

| Source | What it is | Decision |
|---|---|---|
| ForexFactory | Macro calendar (CPI, NFP, FOMC, …), all countries, impact levels. The site refuses automated access, and there is no official export. An unofficial scrape (Hugging Face `Tropstan/Forex_Factory_Calendar`, MIT, 2007 → 2025-04-07) exists but stops nine months before the end of window A. | Not used. Macro events are market-wide; the sector-ETF and cross-sectional legs net them out, and the paper's news is firm-specific. A macro-announcement-day study can be a separate branch from official FOMC/BLS schedules. |
| FinancialJuice | Real-time macro and market headlines; RSS feed only | Not used: no historical archive found, so 2017–2025 cannot be reconstructed. |

## 4. Data

| Item | Value |
|---|---|
| Universe | `sp500_constituents_20260913.csv` |
| Daily bars | Alpaca SIP `1Day` 2016–2025 (cached from P2), plus the 11 Select Sector SPDRs |
| 15-minute bars | Shared 2020–2025 panel; the 2017–2019 panel from P2-04 |
| Costs | S05b method; quote samples exist for both windows (24 sessions 2020–2025, 12 sessions 2017–2019) |
| Earnings events (primary) | SEC EDGAR submissions API: every **8-K whose items include 2.02** (Results of Operations), filing dates 2016–2025, ticker → CIK from SEC `company_tickers.json`. Requests carry a User-Agent from the `.env` variable `SEC_USER_AGENT` (the SEC requires a contact). |
| News shocks (secondary) | Alpaca `v1beta1/news` (Benzinga), 2016–2025. Only articles tagging at most 3 symbols are counted (firm-specific; roundups dropped). An article belongs to the session whose 16:00 ET close it precedes. |
| Windows | **A 2020–2025** and **B 2017–2019**. 2016 is warm-up only. 2026 sealed. |

**Event windows.**
- **Earnings:** the session of the filing date (or the next session if it is not a trading day), plus one session before and one after: $\{f-1, f, f+1\}$. This covers both pre-open and after-close announcements without relying on acceptance times.
- **News shock:** a session on which the stock's firm-specific article count is at least 3 **and** at least 3 × its mean over the previous 60 sessions (at least 20 observed).

## 5. Notebook 01 — daily (faithful in structure)

**Signals**, past $k$-session return to the close of $d$, $k \in \{21$ (primary, the paper's month), $5\}$:
- **R**: raw stock log return.
- **I**: industry-adjusted, the stock's log return minus its sector ETF's over the same $k$ sessions.
- **IN**: I, with stocks removed from the sort if any session in $d-k+1 \ldots d$ is in an earnings window.

Only stocks with a defined industry return enter any arm, so R, I and IN draw from the same universe. Communication Services enters once XLC has $k$ sessions.

**Portfolio.** As in P1 arm D: quintile sort of the signal; long the bottom quintile, short the top, equal weight (+1 / −1); price ≥ $5; overlapping holding $h = k$; signal at the close of $d$, trade at the close of $d+1$. Close-slot measured costs.

**Gates** (per window and horizon; $\alpha = 0.025$ for the two horizons; block bootstrap at blocks 10, 20 and 40, two-sided $1-\alpha$ intervals):

| Gate | Condition |
|---|---|
| F1 news filter | Gross IN − I > 0: lower bound > 0 at all blocks, **and** upper-tail p ≤ α against 200 placebos that remove the same number of stocks per day at random (instead of the earnings-window stocks) |
| F2 industry adjustment | Gross I − R > 0: lower bound > 0 at all blocks |
| F3 economic | IN net at 1×: lower bound > 0 at all blocks, and mean net at 2× > 0 |

Verdicts per horizon:
- `REPLICATES`: F1 and F2 pass in both windows.
- `PARTIAL`: exactly one of F1 and F2 passes in both windows.
- `NO_REPLICATION`: otherwise.
- Economic: `PROFITABLE_AFTER_COSTS` if F3 passes in both windows.

**Secondary (reported, not judged):** INA, which is IN also excluding news-shock stocks over the formation window, and IA, which is I excluding news shocks only. Both get the same statistics as F1.

## 6. Notebook 02 — 15 minutes (applied to S05)

The S05 engine is unchanged (0.1.3, `x10` exits, measured costs). Arm `EX`: new entries on session $t$ are not allowed for stocks with $t$ in an earnings window (mask row $t-1$). Arm `ALL`: no mask. Both windows.

| Gate | Condition (α = 0.05, per window) |
|---|---|
| Q1 | In `ALL`, mean gross per trade of entries outside earnings windows minus entries inside > 0 (bootstrap over entry days, 95% lower bound > 0) |
| Q2 | `EX` net total return and net Sharpe beat 100 placebos that exclude the same number of stocks per session at random (upper-tail p ≤ 0.05 on both) |
| Q3 | `EX` net > 0 at 1× and 2×; bootstrap lower bound of the mean daily net > 0 at blocks 10, 20 and 40 |

Verdict: `EDGE_CANDIDATE` only if Q1–Q3 pass in **both** windows; otherwise `NO_EDGE`. Given P2-04, Q3 is expected to fail in window B.

## 7. Run

1. `python -m pytest -q tests/test_p4.py`
2. Set `SEC_USER_AGENT="Name email@domain"` in `.env`.
3. Notebook 01:
   - Downloads the EDGAR filings: about 500 requests at ≤ 8 per second, a few minutes.
   - Downloads the Alpaca news: resumable per batch-year; it may take hours. `RUN_NEWS = False` skips the secondary arm so the primary result comes first.
   - The remaining computations take minutes.
4. Notebook 02: two windows × 3 backtests + 2 × 100 placebos, about 2–4 hours.

`P4_SYNTHETIC=1` runs both offline.
