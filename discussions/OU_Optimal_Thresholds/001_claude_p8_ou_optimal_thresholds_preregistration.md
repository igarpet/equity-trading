# P8 001 — OU-optimal entry and exit rules (Bertram; Lipton & López de Prado): pre-registration

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Branch | OU_Optimal_Thresholds (paper P8) |
| Sources | Bertram, W. K. (2010), *Analytic solutions for optimal statistical arbitrage trading*, Physica A 389(11), 2234–2243. Lipton, A. & López de Prado, M. (2020), *A closed-form solution for optimal Ornstein–Uhlenbeck driven trading strategies* |
| Standard components | Earnings-window exclusion (P4); both windows judged (P2 004); S05b-method measured costs |
| Status | Pre-registration. Written before any P8 run. |

## 1. The two rules, in s-score units

Write the S05 residual as a stationary OU process in s units, $ds = -\kappa s\,dt + \sqrt{2\kappa}\,dW$. Here $s$ is measured in units of $\sigma_{eq}$, and κ is the per-bar mean-reversion rate from S05 (κ = 1/τ). A long residual position gains $\sigma_{eq}\,\Delta s$. The round-trip cost in s units is
$$\tilde c = 2\,(h_{stock} + |\beta|\,h_{ETF})/\sigma_{eq},$$
with $h$ the S05b measured one-way half-spreads (each symbol's median over the session's slots).

**Bertram (`BT`).**
- **Band:** a symmetric band $\pm m$. Go long at $s \le -m$, close at $s \ge +m$; shorts mirror this.
- **Cycle time:** the expected time of a full cycle is $E[T] = \frac{2\pi}{\kappa}\,\mathrm{Erfi}(m/\sqrt2)$.
- **Optimal band:** $m^*$ maximises the expected net return per unit time, $(2m - \tilde c)/E[T]$. κ cancels out, so $m^*$ depends only on $\tilde c$.
- **Per stock and day:** $m^*$ is recomputed from that day's $\tilde c$. If $m^* > 5$, the stock is not traded that day.

**Lipton & López de Prado (`LP`).**
- **Entry:** S05's rule, $|s| \ge 1.25$. The book takes the largest |s| first, up to 100 positions.
- **Exits:** each trade gets a take-profit level, a stop-loss and a time limit $T$ = 3 half-lives (3 ln 2 / κ bars).
- **Choice of exits:** they are chosen to maximise expected net P&L per unit time in the trade, $(E[\Delta s] - \tilde c)/E[\tau]$, given the entry level $|s_0|$ and $\tilde c$.
- **Candidate levels:**
  - take-profit at $s \in \{-1, -0.5, 0, 0.5, 1, 1.5\}$ on the mean side (long: exit at $s \ge$ level);
  - stop-loss at 0.5, 1, 1.5 or 2 beyond the entry, or none.
- **How it is solved:** the paper derives closed forms and optimises the Sharpe ratio. We solve the same first-passage problem by simulating the standard OU process (κ-time, step 0.01, 4,000 paths per $|s_0|$ on a grid of 1.25 … 5). The criterion is per unit time, as in Bertram, because the S05 book is always full and capital is reused immediately. This deviation is stated here.

## 2. Unchanged components

- **S05 estimation:** 0.1.3, 20-session window, validity rules.
- **Sizing:** 1% of equity per stock, β-hedged with the sector ETF.
- **Execution:** fills at the next bar open, measured costs.
- **Earnings filter:** no entries on sessions inside an earnings window $\{f-1, f, f+1\}$.

**Incumbent (reference):** S05 with `x10` exits plus the earnings filter, i.e. P4's `EX` arm.

## 3. Data

Window A is 2020–2025 (the shared 15-minute panel). Window B is 2017–2019 (the P3/P2-04 panel). Quote samples and EDGAR events are cached, so nothing is downloaded. 2026 is sealed.

## 4. Gates

Per window and arm, α = 0.05 / 2 arms:

| Gate | Condition |
|---|---|
| G1 edge | Mean daily net return: block-bootstrap two-sided $1-\alpha$ interval, lower bound > 0 at blocks 10, 20 and 40 |
| G2 improvement | Mean daily net (arm − incumbent): paired block-bootstrap lower bound > 0 at blocks 10, 20 and 40 |
| G3 cost robustness | Net total return > 0 and Sharpe > 0 at 2× cost |

**Verdicts per arm:**
- **Improvement:** `IMPROVES_ON_S05` if G2 passes in both windows.
- **Edge:** `EDGE_CANDIDATE` if G1 and G3 pass in both windows.
- Otherwise: `NO_IMPROVEMENT` / `NO_EDGE`.

**Reported:** trades, mean holding, exit reasons, gross and cost per trade, the distribution of $m^*$ and $\tilde c$, and results by year.

## 5. Prior

S08 found that the s-score's tails are far heavier than OU: |s| > 3 is routine, and only |s| > 3 earns more than the round trip. S05c found that exit levels change the trade count, not the gross per unit of time. Both rules assume the OU model holds.

- **`BT`:** with $\tilde c$ small relative to $\sigma_{eq}$, $m^*$ will be small. That means frequent, near-the-mean trading, which the data says does not pay. Expected to lose.
- **`LP`:** may shorten losing trades through stops and time limits. Expected to change timing, not the gross edge.

## 6. Run

`python -m pytest -q tests/test_p8.py`, then `01_ou_optimal_thresholds.ipynb`. The LP table takes about a minute. Then 2 windows × (2 arms × 3 cost levels + incumbent), about 1 hour. `P8_SYNTHETIC=1` runs it offline.
