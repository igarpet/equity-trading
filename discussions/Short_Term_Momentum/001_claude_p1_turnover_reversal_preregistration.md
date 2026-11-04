# P1 001 — Short-term momentum: does reversal depend on turnover? (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-04 |
| Branch | Short_Term_Momentum (paper P1) |
| Source | Medhat, M. & Schmeling, M. (2022), *Short-term Momentum*, Review of Financial Studies 35(3), 1480–1526 (CEPR DP15857) |
| Ivan's choices | Test both horizons: a daily version faithful to the paper and the 15-minute S05 residual signal split by turnover; abnormal volume as the turnover proxy |
| Status | Pre-registration. Written before any P1 data download or run. Gates below are not changed after results exist. |

## 1. Claim

The paper sorts US stocks on prior-month return and prior-month share turnover. The short-term reversal is concentrated in low-turnover stocks; among the highest-turnover stocks the prior-month return continues (short-term momentum).

## 2. Prior (our evidence)

- S05a–S09: the sector-ETF residual reversal at 15 minutes is real in gross terms (S05a gross Sharpe 1.22, beats random direction; S07 beats a shuffled signal, p = 0.002), but its gross edge (≈ 8 bps per trade) is below the measured round trip (≈ 13 bps, S05b).
- The S05 book is always full (S05c/S05d), so a filter that removes the worst third of names changes which trades are taken, not how many.
- Nothing yet conditions on volume.

If the paper's mechanism holds in our sample, part of our signal sits in high-turnover names where moves continue, and removing them should raise the gross edge per trade. That is the only route by which P1 can change our verdict.

## 3. Data

| Item | Value |
|---|---|
| Universe | `src/equity_trading/reference/sp500_constituents_20260913.csv` (current snapshot: survivorship bias, as in all earlier studies) |
| Daily bars | Alpaca SIP `1Day`, `adjustment=all`, 2019–2025. **2019 is used only as the volume baseline** (warm-up of the 250-session average); no signal, trade or statistic uses 2019 returns. |
| 15-minute bars | Alpaca SIP `15Min`, 2020–2025, shared panel (`PANEL_VERSION` 0.2.0), as S05 |
| Costs | S05b measured cost model (SIP NBBO samples, 4 sessions per year, per symbol-year × intraday slot profile) |
| Sealed | 2026 onward: not downloaded, not used |

## 4. Turnover proxy (difference from the paper)

The paper uses **share turnover** (volume / shares outstanding), a cross-sectional level. Alpaca provides no shares outstanding. We use **abnormal volume**:

$$AV_{i,d} = \frac{\text{mean volume}_{i}(d-19 \ldots d)}{\text{mean volume}_{i}(d-249 \ldots d)}, \quad \text{at least 200 sessions in the denominator.}$$

It is known at the close of day $d$. AV measures turnover relative to the stock's own norm, not the level across stocks. A null result therefore does not refute the paper: it says that *abnormal* turnover does not condition the reversal in this universe. A positive result is about AV, not share turnover.

Each day, stocks with a finite AV (and, in the daily arm, price ≥ $5 and a finite signal) are split into terciles: **T1** low AV, T2, **T3** high AV.

## 5. Arm D — daily, faithful (notebook `01_daily_turnover_reversal.ipynb`)

**Signal.** Past log return over $k$ sessions to the close of $d$. Horizons $k \in \{21, 5, 1\}$; $k = 21$ (one month) is the paper's horizon and is primary; 5 and 1 are secondary. All three are judged, at $\alpha = 0.05 / 3$.

**Portfolios.** Conditional sort: within each AV tercile, quintiles of the signal. Reversal portfolio of a tercile = equal-weight long the bottom quintile, short the top quintile (+1 long, −1 short). Formation every day; holding $h = k$ sessions with overlapping portfolios (the daily target is the mean of the last $h$ formations).

**Timing.** Signal at the close of $d$; trade at the close of $d+1$ (one-session skip, implementable with market-on-close orders); P&L from the close of $d+1$ on. Returns are per $1 long and $1 short.

**Costs.** One-way cost per symbol per day = S05b base half-spread (symbol, year) × the profile of the last 15-minute slot of the session (15:45), charged on $\sum_i |\Delta w_i|$ at execution. Weight drift between rebalances is ignored. Gross (0×), 1× and 2× are reported.

**Gates** (per horizon; block bootstrap of daily means, blocks 10, 20 and 40 sessions, 10,000 draws, two-sided intervals at $1 - \alpha$, $\alpha = 0.05/3$):

| Gate | Condition |
|---|---|
| D1 turnover dependence | Gross daily return of (T1 reversal − T3 reversal) > 0: interval lower bound > 0 at all three block lengths, **and** upper-tail p ≤ α against 500 placebos that shuffle the AV labels across stocks within each day |
| D2 high-turnover momentum | Gross T3 reversal < 0: interval upper bound < 0 at all three block lengths |
| D3 economic | Net T1 reversal at 1× cost: interval lower bound > 0 at all three block lengths, **and** net mean at 2× cost > 0 |

Replication verdict: `REPLICATES` (D1 and D2), `PARTIAL` (D1 only), `NO_REPLICATION` otherwise. Economic verdict: `PROFITABLE_AFTER_COSTS` if D3, else `NOT_PROFITABLE_AFTER_COSTS`.

## 6. Arm I — 15 minutes, applied (notebook `02_intraday_turnover_residual.ipynb`)

**Engine.** S05 (`s05.py`, version 0.1.3: an optional entry mask is added; with no mask the behaviour is unchanged and test-checked). Sector-ETF residual s-score, 20-session window, entries at $|s| > 1.25$, 1% of equity per stock, at most 100 positions, measured S05b costs.

**Exits.** `x10` (close a long at $s > +1.0$, a short at $s < -1.0$). Disclosed: `x10` was the best of three exit arms in S05c, chosen after that result. P1 compares terciles under the same exits, so the selection affects the level of all arms, not the comparison between them.

**Arms.** Entries allowed only for stocks in the AV tercile known at the close of the previous session: `T1`, `T2`, `T3`; plus `ALL` (no mask, = S05c `x10`).

**Gates** ($\alpha = 0.05$):

| Gate | Condition |
|---|---|
| I1 turnover dependence | In the `ALL` run, mean gross return per trade of T1 entries − T3 entries > 0 (bootstrap over entry days, 10,000 draws, 95% lower bound > 0). Tercile of a trade = the label in force when its signal was taken. |
| I2 beats random membership | `T1` net total return and net Sharpe each rank above 200 placebo runs whose entry mask is a random set of stocks of the same daily size as T1 (upper-tail p ≤ 0.05 on both) |
| I3 economic | `T1` net total return > 0 at 1× and at 2× cost, and the bootstrap interval of its mean daily net return has lower bound > 0 at blocks 10, 20 and 40 |

Verdict: `EDGE_CANDIDATE` if I1, I2 and I3; otherwise `NO_EDGE`. Reported, not judged: T3 mean gross per trade (< 0 would be the paper's momentum reading at 15 minutes), random-direction test on T1 trades, gross over cost per arm.

## 7. Readings fixed in advance

| D (daily) | I (15m) | Reading |
|---|---|---|
| REPLICATES or PARTIAL | EDGE_CANDIDATE | AV conditioning works at both horizons; T1 goes to a 2026 prospective test |
| REPLICATES or PARTIAL | NO_EDGE | The effect exists at daily horizons but does not carry to the 15-minute residual; a daily-horizon strategy becomes a separate study |
| NO_REPLICATION | EDGE_CANDIDATE | Unexpected; treated as a false-positive candidate until a 2026 prospective test |
| NO_REPLICATION | NO_EDGE | Abnormal turnover does not condition reversal in this universe; P1 closes |

## 8. Run order and outputs

1. `python -m pytest -q tests/test_p1.py tests/test_s05.py`
2. Notebook 01: downloads `1Day` bars 2019–2025 (≈ 300 requests, a few minutes); reuses the S05b quote samples in `<EQUITY_TRADING_DIR>/Alpaca/quote_samples/`. Outputs to `01/`.
3. Notebook 02: uses the shared 15-minute panel and the `1Day` bars from step 2. Backtests plus 200 placebos; expect ~1–3 hours with all cores. Outputs to `02/`.

`P1_SYNTHETIC=1` runs both on synthetic data (smoke test; outputs to `NN_synthetic/`, ignored by git).

## 9. Not tested

Share turnover itself; value-weighted portfolios; the paper's full sample period; size or price-level interactions; turnover measured from intraday volume.
