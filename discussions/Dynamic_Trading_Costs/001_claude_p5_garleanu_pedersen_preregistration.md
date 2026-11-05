# P5 001 — Dynamic trading with predictable returns and costs (Gârleanu & Pedersen): pre-registration

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-05 |
| Branch | Dynamic_Trading_Costs (paper P5) |
| Source | Gârleanu, N. & Pedersen, L. H. (2013), *Dynamic Trading with Predictable Returns and Transaction Costs*, Journal of Finance 68(6), 2309–2340 |
| Standard components | Earnings-window exclusion (P4 002); both windows judged (P2 004) |
| Status | Pre-registration. Written before any P5 run. |

## 1. The paper's rule

With return forecasts $f$ that decay at rates $\phi_k$, covariance $\Sigma$, risk aversion $\gamma$ and quadratic costs $\Lambda = \lambda\Sigma$, the optimal portfolio has two features:

1. **Partial adjustment.** Trade a fixed fraction toward an aim portfolio each period:
   $x_t = x_{t-1} + \tfrac{a}{\lambda}(\text{aim}_t - x_{t-1})$, where $a = \tfrac{-\gamma + \sqrt{\gamma^2 + 4\gamma\lambda}}{2}$ (no discounting).
2. **Persistence-weighted aim.** Each signal is weighted by how slowly it decays:
   $\text{aim}_t = (\gamma\Sigma)^{-1} \sum_k \frac{B_k f_{k,t}}{1 + \phi_k a/\gamma}$.
   Slow signals get relatively more weight as costs rise.

Only the ratio $q = \lambda/\gamma$ matters once the aim's gross exposure is fixed. It sets both the trade rate $\tau = a/\lambda$ and the persistence weights through $c = a/\gamma = (-1 + \sqrt{1+4q})/2$.

**Difference from the paper.** Our measured costs are proportional (half-spreads), not quadratic. Under proportional costs the exact optimum is a no-trade band (S07). P5 tests whether the paper's rule, with its persistence weighting, does better *in practice* than what we have. It is not a test of the paper's optimality claim.

## 2. Signals (both 15-minute, from the S05 estimation)

All S05 parameters are those estimated at the close of session $t-1$ and frozen for session $t$: α, β, the AR(1) $b$, the centred mean $m$ and $\sigma_{eq}$. The innovation s.d. is $\sigma_z = \sigma_{eq}\sqrt{1-b^2}$.

| Signal | Definition | Expected decay |
|---|---|---|
| **A, fast** | S05 s-score: the cumulative residual since the session open, minus $m$, over $\sigma_{eq}$ | Within the day |
| **B, slow** | Cumulative residual over the last 130 bars (5 sessions), over $\sigma_z\sqrt{130}$ | Several days |

**Estimated ex ante each session**, pooled over stocks and bars of the previous 20 sessions:
- $B_A, B_B$: OLS of the next bar's hedged return ($r - \beta r_{ETF}$) divided by $\sigma_z$, on $(-s_A, -s_B)$.
- $\phi_A, \phi_B$: one minus the pooled bar-to-bar AR(1) coefficient of each signal.

**Aim.** For valid stocks outside earnings windows:
$u_i = \left[\frac{B_A(-s_{A,i})}{1+\phi_A c} + \frac{B_B(-s_{B,i})}{1+\phi_B c}\right] / \sigma_{z,i}$.
Then aim $= u / \sum|u|$, so the aim's stock gross is 100% of equity, with at most 2% per stock. Stocks in an earnings window $\{f-1, f, f+1\}$, or with an invalid estimate, have $u = 0$.

**Trading.**
- **Stocks:** at each bar close, the new stock target is $x + \tau(\text{aim} - x)$, from current actual weights. It is executed at the next bar open with market orders at the measured half-spread.
- **Hedge:** sector-ETF hedges are netted and rebalanced when more than 0.1% of equity off target, as in S07.
- **Overnight:** positions are held overnight.

## 3. Arms

| Arm | $q = \lambda/\gamma$ | $c$ | Trade rate τ per bar | Weight on A vs B at $\phi_A = 0.1$, $\phi_B = 0.005$ (illustration) |
|---|---|---|---|---|
| `q12` | 12 | 3 | 0.25 | 0.77 : 0.99 |
| `q90` | 90 | 9 | 0.10 | 0.53 : 0.96 |
| `q600` | 600 | 24 | 0.04 | 0.29 : 0.89 |

**Reported, not judged (at `q90`):** `noPW` (both signals with equal persistence weights, same τ), `Aonly` and `Bonly`.

## 4. Data

| Window | 15-minute panel | Quote samples |
|---|---|---|
| A 2020–2025 | Shared panel | S05b sessions |
| B 2017–2019 | P3 / P2-04 panel | P2-04 sessions |

Earnings events come from the P4 EDGAR cache. 2026 is sealed. Nothing new is downloaded.

## 5. Gates

Per window and arm, with α = 0.05 / 3 arms:

| Gate | Condition |
|---|---|
| G1 | Mean daily net return at 1×: block-bootstrap two-sided $1-\alpha$ interval, lower bound > 0 at blocks 10, 20 and 40 |
| G2 | Net total return and net Sharpe beat 100 placebos in which, each session, the stocks' forecasts are randomly permuted among valid stocks (upper-tail p ≤ α on both) |
| G3 | Net total return > 0 and net Sharpe > 0 at 2× cost |

**Verdict.** `EDGE_CANDIDATE` if the **same arm** passes G1–G3 in **both** windows; otherwise `NO_EDGE`.

**Reported:** gross Sharpe, annual turnover, annual cost, average gross exposure, the estimated $B$ and $\phi$ over time, results by year, and the ablations.

## 6. Prior

The 15-minute residual has gross edge in 2020–2025 but none in 2017–2019 (P2-04, P4-02). Daily reversal earns about 2–3 bps/day gross in both windows (P1, P4). P5 can only pass if persistence weighting moves the book toward the slow signal and the slower turnover keeps that gross. A pass in window B is unlikely.

## 7. Run

`python -m pytest -q tests/test_p5.py`, then `01_garleanu_pedersen.ipynb`. Per window, signal construction takes minutes; then 15 backtests and 3 × 100 placebos. Expect roughly 1.5–2 hours per window with all cores (3–4 hours in total). Nothing is downloaded. `P5_SYNTHETIC=1` runs it offline.
