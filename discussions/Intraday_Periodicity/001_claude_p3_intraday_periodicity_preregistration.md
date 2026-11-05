# P3 001 — Intraday periodicity: does a stock's return in a half-hour slot predict the same slot on later days? (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-05 |
| Branch | Intraday_Periodicity (paper P3) |
| Source | Heston, S. L., Korajczyk, R. A. & Sadka, R. (2010), *Intraday Patterns in the Cross-Section of Stock Returns*, Journal of Finance 65(4), 1369–1407 (arXiv 1005.3535) |
| Working rule | Following P2 004: a result on 2020–2025 counts only if it also holds, unchanged, on 2017–2019. Both windows are pre-registered here. |
| Status | Pre-registration. Written before any P3 computation on real data. |

## 1. Claim

In cross-sectional regressions of half-hour returns on lagged half-hour returns, the slopes are positive at lags that are exact multiples of one trading day (13 half-hours), and stay positive for many days. At other lags they are near zero or negative. A stock's return in a given half-hour therefore predicts its return in the same half-hour on later days.

## 2. Prior

The paper's effect is a few basis points per half-hour at most, before costs. Trading it means rebuilding a long-short book every half-hour. Our measured one-way cost is about 2 bps, so a round trip costs about 4 bps per name per interval. **The economic gate is expected to fail.** The study is still useful for two things: whether the effect exists in recent data, and whether same-slot persistence is a structure later branches should control for (the opening bar in S05d is one such case).

## 3. Data

| Item | Value |
|---|---|
| Universe | S&P 500 stocks in `sp500_constituents_20260913.csv` (no ETFs traded) |
| Bars | Alpaca SIP `15Min`, aggregated to 30 minutes: open of the first 15-minute bar, close of the second. Regular session only. **Full days only** (13 intervals); early-close days are dropped, so lags in multiples of 13 always mean "same slot, j sessions earlier". |
| Interval return | $r = \log(\text{close}/\text{open})$ of the interval. The overnight gap is excluded; the first interval runs 09:30–10:00 from the open. |
| Windows | **A: 2020–2025** (shared 15-minute panel). **B: 2017–2019** (the P2/04 bars, own panel cache). |
| Costs | S05b method per window (quote samples 2020–2025: 24 sessions; 2017–2019: 12 sessions). One-way cost at the 15-minute slot where the 30-minute interval opens. |
| Sealed | 2026 |

## 4. Notebook 01 — regressions (faithful)

For each interval $t$ and lag $k = 1 \ldots 520$ (40 sessions), the cross-sectional OLS slope $\gamma_{t,k}$ of $r_{i,t}$ on $r_{i,t-k}$ is computed (at least 100 stocks). Fama–MacBeth means $\bar\gamma_k$ are reported with Newey–West t-statistics (13 lags).

| Gate | Condition, per window (one-sided, α = 0.05) |
|---|---|
| S1 persistence | $D_t$ = mean of $\gamma_{t,13j}$ over $j = 1 \ldots 40$. Its time-series mean is > 0 (NW t, p ≤ α). |
| S2 specificity | $D_t$ minus the mean of $\gamma_{t,k}$ over the non-multiple lags $14 \le k \le 520$ is > 0 (NW t, p ≤ α). |

Verdict: **`REPLICATES`** if S1 and S2 pass in **both** windows; `PARTIAL` if they pass in one window only; `NO_REPLICATION` otherwise.

## 5. Notebook 02 — same-slot strategy (economic)

**Signal.** For stock $i$ in slot $s$ of session $d$: the mean of its returns in slot $s$ over sessions $d-20 \ldots d-1$. Twenty sessions is our choice, fixed here. Lookbacks of 1, 5 and 40 sessions are reported, not judged.

**Portfolio.** Each interval: equal-weight long the top decile of the signal, short the bottom decile (+1 / −1). Enter at the interval's open, hold the interval. Weights carry over between consecutive intervals of the same session (only changes are traded). Everything is closed at the session close; nothing is held overnight.

**Costs.** One-way cost of the entry slot × |change in weight|; the final unwind uses the last slot's cost. Gross, 1× and 2× are reported.

| Gate | Condition, per window |
|---|---|
| E1 gross | Daily gross return (sum over the day's intervals): block-bootstrap 95% lower bound > 0 at blocks 10, 20 and 40 sessions |
| E2 net | Daily net at 1× cost: bootstrap lower bound > 0 at all three blocks, and mean daily net at 2× cost > 0 |

Verdict: `EDGE_CANDIDATE` only if E1 and E2 pass in both windows; otherwise `NO_EDGE`. Reported, not judged: gross by slot of day, the four lookbacks, and turnover.

## 6. Readings fixed in advance

| Notebook 01 | Notebook 02 | Reading |
|---|---|---|
| `REPLICATES` | `NO_EDGE` | The pattern is real but too small to trade alone. It becomes a candidate control variable (slot-of-day effect) for other branches, not a strategy. |
| `REPLICATES` | `EDGE_CANDIDATE` | Unexpected given the prior. Goes to the 2026 holdout unchanged. |
| `PARTIAL` or `NO_REPLICATION` | any | The pattern is not stable in our data; it is not used as a control. |

## 7. Run

`python -m pytest -q tests/test_p3.py`, then notebook 01 (lags 1–520 on two windows; a few minutes) and notebook 02 (minutes). Both read cached bars and quote samples; nothing new is downloaded. `P3_SYNTHETIC=1` runs both offline.
