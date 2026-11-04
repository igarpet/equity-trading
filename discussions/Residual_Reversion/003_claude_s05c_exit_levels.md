# S05c: exit level of the threshold strategy, at the mean and beyond it

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-03, written after S07 (8b38eef) and before any S05c result |
| **Notebook** | `notebooks/Residual_Reversion/05c_exit_levels.ipynb` |
| **Engine** | `src/equity_trading/s05.py` (S05b engine with measured costs; the placebo eligibility band is now a separate parameter, fixed at the paper's exit band) |
| **Data** | As S05b. 2020–2025 only. |

## 0. Question

The paper's exits (close a long at s > −0.50, a short at s < +0.75) leave each trade **before** the residual reaches its mean. In S05b the gross edge was 8.0 bps per trade against a cost of 13.0 bps.

**Hypotheses:**
- **Exiting at the mean (s = 0)** captures more of each reversion for the same number of round trips, raising gross per trade at an unchanged cost per trade.
- **Exiting beyond the mean (Iván's proposal)** bets that the residual overshoots. A pure OU process predicts no overshoot: once s crosses 0, the expected drift points back toward 0. This arm tests whether the data show overshoot beyond OU.

## 1. Arms (three, fixed)

| Arm | Close long when | Close short when |
|---|---|---|
| `x00` | s > 0 | s < 0 |
| `x05` | s > +0.5 | s < −0.5 |
| `x10` | s > +1.0 | s < −1.0 |
| `paper` (reference, not judged) | s > −0.50 | s < +0.75 |

Everything else is identical to S05b:
- entries at \|s\| ≥ 1.25;
- 1% of equity per stock, β-hedged in the sector ETF, at most 100 positions, ranked by \|s\|;
- exit when the stock becomes non-tradable;
- market orders at the next open with measured costs.

## 2. Evaluation

Gates per arm are as in S05b, with **α = 0.05 / 3 = 0.0167**:

| Gate | Condition |
|---|---|
| **G1 bootstrap** | Block-bootstrap 95% CI of mean daily net return > 0, for blocks of 10, 20 and 40 days |
| **G2 random direction** | Same trades with random signs, p ≤ α |
| **G3 random timing** | 300 backtests: contrarian entries at random flat tradable bars with s < −0.50 or s > +0.75 (eligibility fixed across arms), at the arm's entry rate, using the arm's own exits; total return and Sharpe each p ≤ α |
| **G4 double cost** | Net return > 0 and Sharpe > 0 at 2× measured cost |

**Verdicts:** `SIGNAL` if G1, G2 and G3 all pass; economic = net return > 0 and G4; `EDGE_CANDIDATE` if both.

**Reported:** gross per trade, cost per trade, gross/cost ratio, holding time and win rate per arm.

**Pre-stated reading:**
- If `x05` or `x10` beats `x00` on **gross** per trade, the residual overshoots beyond what OU implies.
- If not, the extra gross per trade comes from holding longer, and the cost/edge ratio decides.

## 3. Limitations

As S05b: development data, quoted spreads only, free shorting, survivorship, overnight holding. The exit levels were proposed after S05–S07 results were seen.
