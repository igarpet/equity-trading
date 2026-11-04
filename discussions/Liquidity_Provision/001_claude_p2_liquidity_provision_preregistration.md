# P2 001 — Evaporating liquidity: is reversal larger when liquidity is scarce? (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-04 |
| Branch | Liquidity_Provision (paper P2) |
| Source | Nagel, S. (2012), *Evaporating Liquidity*, Review of Financial Studies 25(7), 2005–2039 (NBER w17653) |
| Carried in | P1 readout (`../Short_Term_Momentum/002_claude_p1_readout.md`): post-hoc observation that high abnormal-volume stocks reverse more |
| Status | Pre-registration. Written before any P2 run. Gates are not changed after results exist. |

## 1. Claims to test

**H1 (Nagel).** Returns of short-term reversal strategies are returns to liquidity provision. Their expected value rises with market volatility, so lagged VIX predicts them.

**H2 (from P1, post hoc).** Within a day's cross-section, stocks with a volume shock (high abnormal volume, AV) reverse more than stocks with low AV. Found in 2020–2025, so it is tested only on data not used in P1: **2017–2019**.

## 2. Data

| Item | Value |
|---|---|
| Universe | `sp500_constituents_20260913.csv`. Current snapshot: survivorship bias is larger the further back the sample goes, so 2016–2019 is more biased than 2020–2025. |
| Daily bars | Alpaca SIP `1Day`, `adjustment=all`, **2016–2025**. 2016 serves as warm-up for the 250-session AV and VIX baselines. |
| 15-minute bars | Alpaca SIP `15Min`, 2020–2025 (shared panel) |
| VIX | CBOE VIX daily close from Yahoo Finance (`^VIX`). Closes at 16:15 New York, after the stock close; always used one session or more before it is traded on. |
| Costs | S05b measured model. It is measured on 2020–2025 quotes only, so **economic gates use 2020–2025 only**. 2016–2019 results are gross. |
| Sealed | 2026 onward |

**Regime.** VIX is *high* at the close of $d$ when it exceeds its own trailing median over sessions $d-249 \ldots d$ (at least 200). The regime is ex ante and adapts to the level of the period.

## 3. Notebook 01 — daily, H1 (faithful)

**Portfolio.** Lehmann (1990) / Lo–MacKinlay (1990) contrarian weights, the reversal strategy used in the paper:
$w_{i,d} \propto -(r_{i,d} - \bar r_d)$, scaled to +1 long and −1 short. Here $r_{i,d}$ is the stock's return on session $d$ and $\bar r_d$ the equal-weight mean; price ≥ $5. Signal at the close of $d$, trade at the close of $d+1$ (one-session skip, as P1), held one session. Return per $1 long and $1 short.

**Gates** ($\alpha = 0.025$ per hypothesis family; HAC standard errors with 5 lags; one-sided tests):

| Gate | Window | Condition |
|---|---|---|
| N1 slope | 2016-12 → 2025 | Regression of the daily gross return on VIX at its signal close: slope > 0, p ≤ α |
| N2 regime | 2016-12 → 2025 | Same regression on the high-VIX dummy: slope > 0, p ≤ α |
| N3 economic | 2020 → 2025 | The VIX-gated portfolio (trades only on signals with a high regime, flat otherwise), net of measured costs: block-bootstrap interval of the mean daily net return, at $1-\alpha$, lower bound > 0 at blocks 10, 20 and 40, **and** mean net at 2× cost > 0 |

Verdicts: `REPLICATES` (N1 and N2), `PARTIAL` (one of them), `NO_REPLICATION`. Economic: `PROFITABLE_AFTER_COSTS` if N3.

## 4. Notebook 02 — daily, H2 on the independent sample

Same construction as P1 arm D at **k = 5** (the horizon where P1 saw the effect): AV terciles, quintile sort on the past 5-session return within each tercile, overlapping 5-session holding, one-session skip. Window **2017-01-01 → 2019-12-31**.

| Gate | Condition |
|---|---|
| A1 bootstrap | Gross T3 − T1 reversal > 0: block-bootstrap $1-\alpha$ lower bound > 0 at blocks 10, 20 and 40 |
| A2 placebo | Observed T3 − T1 larger than 500 shuffled-label placebos: upper-tail p ≤ α |

Verdict: `CONFIRMED` (A1 and A2), `NOT_CONFIRMED` otherwise. No economic gate (no measured costs before 2020); net figures at the 2020–2025 cost level are reported only. Reported for reference, not judged: the same statistics on 2020–2025.

## 5. Notebook 03 — 15 minutes, H1 applied to the S05 residual signal

S05 engine (0.1.3), sector-ETF residual, entries at $|s| > 1.25$, `x10` exits, 1% per stock, ≤ 100 positions, measured costs, 2020–2025. Arm `VIX`: new entries are allowed on session $t$ only if the regime at the close of $t-1$ is high (all stocks). Arm `ALL`: no gate (= P1 `ALL`).

| Gate | Condition ($\alpha = 0.05$) |
|---|---|
| V1 | In the `ALL` run: mean gross per trade of entries in a high regime minus entries in a low regime > 0 (bootstrap over entry days, 95% lower bound > 0) |
| V2 | `VIX` net total return and net Sharpe rank above 200 placebos whose regime series is the real one shifted circularly by a random offset of at least 60 sessions (same share and clustering of allowed days); upper-tail p ≤ 0.05 on both |
| V3 | `VIX` net total return > 0 at 1× and 2× cost; bootstrap interval of the mean daily net return lower bound > 0 at blocks 10, 20 and 40 |

Verdict: `EDGE_CANDIDATE` if V1, V2 and V3; otherwise `NO_EDGE`.

## 6. Readings fixed in advance

- **H1 replicates daily and V passes:** VIX gating becomes part of the S05 strategy and goes to the 2026 prospective test.
- **H1 replicates daily, V fails:** the liquidity premium exists at daily horizons but does not lift the 15-minute residual above cost.
- **H2 confirmed on 2017–2019:** high-AV conditioning is admitted as a hypothesis for a later strategy study (with H1 if both hold).
- **H2 not confirmed:** the P1 observation is treated as sample noise and dropped.

## 7. Run order

1. `python -m pytest -q tests/test_p2.py`
2. Notebook 01: downloads `1Day` bars 2016–2018 (the rest are cached from P1) and `^VIX` from Yahoo. Minutes.
3. Notebook 02: 500 label placebos. About 5–15 minutes.
4. Notebook 03: 2 × 3 backtests plus 200 placebos. About 1–3 hours.

`P2_SYNTHETIC=1` runs all three offline (outputs to `NN_synthetic/`, ignored by git).

## 8. Not tested

VIX futures or option-implied liquidity measures; the paper's other liquidity proxies; industry-adjusted reversal; value weighting; 15-minute bars before 2020.
