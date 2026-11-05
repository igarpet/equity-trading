# P6 001 — Deep learning statistical arbitrage (Guijarro-Ordonez, Pelger & Zanotti): pre-registration

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Branch | Deep_Statistical_Arbitrage (paper P6) |
| Source | Guijarro-Ordonez, J., Pelger, M. & Zanotti, G., *Deep Learning Statistical Arbitrage* (arXiv 2106.04028) |
| Ivan's choices | Yahoo daily data to extend the training history; GPU: GTX 1070 (8 GB, Pascal) |
| Working rule | Both 2017–2019 and 2020–2025 judged, each fully out of sample (P2 004) |
| Status | Pre-registration. Written before any Yahoo download or model training. |

## 1. Paper's pipeline, and ours

| Step | Paper | P6 |
|---|---|---|
| Universe | Large US stocks, daily | Current S&P 500 snapshot, daily, price ≥ $5 |
| Residuals | Fama–French, PCA or IPCA factors on rolling windows | **PCA, 5 factors**, correlation matrix of the previous 252 sessions, re-estimated daily, applied to the next day's return (ex ante) |
| Signal input | Last 30 days of cumulative residual returns per stock | Same: 30 sessions, divided by that day's cross-sectional s.d. |
| Signal model | CNN + Transformer | Two 1-D convolution layers (8 filters, kernel 2, residual connection) + one Transformer encoder layer (4 heads, d = 8) + linear head → score per stock |
| Allocation | Network maps signals to weights, trained end-to-end to maximise the Sharpe ratio | Scores minus their mean, **neutralised to the 5 PCA factors**, scaled to Σ\|w\| = 1; end-to-end objective = **Sharpe of daily net returns** on the training set |
| Training | Rolling 1,000 days, refit every 125 days | Same. Within each 1,000 days: first 750 to train, last 250 to choose the epoch (best validation net Sharpe), up to 100 epochs, full batch, Adam (lr 1e-3). Ensemble of 5 seeds (mean score). |
| Costs | Fixed bps per unit of turnover | **Measured** close-slot half-spread per symbol-year (S05b method); in training years before 2017, each symbol's earliest measured value |

**Execution.** Weights are formed from data up to the close of $d$ and traded at the close of $d+1$; returns are earned from $d+1$ to $d+2$. This is the one-session skip of P1/P4, and training uses the same target. The paper trades at the close the signal uses; that is not implementable and is not tested.

**Factor neutralisation.** With standardised returns $z = r/\sigma$ and PCA eigenvectors $V$ (N × 5), a stock portfolio $w$ is exposed to the factors through $X = \sigma \odot V$. We set $w \leftarrow w - X(X^\top X)^{-1}X^\top w$, so the portfolio's return is its residual return. This runs inside the training graph, so the network learns weights *after* hedging.

## 2. Data

| Item | Value |
|---|---|
| Daily prices | **Yahoo** (yfinance, adjusted close) 2005-01 → 2015-12, then **Alpaca** SIP `1Day` 2016-01 → 2025-12. They are spliced on returns: Yahoo returns before 2016-01-04, Alpaca returns from then on. |
| First prediction | 2017-01-03. Model inputs need 252 + 30 sessions, and each fit needs 1,000 sessions of residuals. The first fit trains on about 2012–2016. |
| Test windows | **B 2017–2019** and **A 2020–2025**. Every test day is predicted by a model trained only on earlier data. |
| Sealed | 2026 |

Survivorship: the current constituent list is applied back to 2005, so training data has more survivorship bias than the test windows. This affects what the model learns, not the out-of-sample evaluation, which is biased as in all earlier studies.

## 3. Benchmarks (reported, not judged)

Same residuals, neutralisation, normalisation, skip and costs as the network:
- **OU:** the Avellaneda–Lee s-score from an AR(1) on the 30-day cumulative residual; weight ∝ −s.
- **Linear:** weight ∝ −(cumulative residual over the last 5 sessions).

## 4. Gates

Per window, α = 0.05:

| Gate | Condition |
|---|---|
| G1 | Mean daily net return: block-bootstrap 95% lower bound > 0 at blocks 10, 20 and 40 |
| G2 | **Gross** total return and gross Sharpe beat 1,000 placebos that permute the network's daily weights among that day's stocks (upper-tail p ≤ 0.05 on both). Gross, because a new permutation each day raises the placebo's turnover, so a net comparison would favour the model unfairly. |
| G3 | Net total return > 0 and net Sharpe > 0 at 2× cost |

Verdict: **`EDGE_CANDIDATE`** if G1–G3 pass in **both** windows; otherwise `NO_EDGE`.

Reported: gross results, turnover, cost, the benchmarks, results by year, and the in-sample (validation) Sharpe of each fit.

## 5. Prior

The paper reports large out-of-sample Sharpe ratios after costs. Our daily residual reversal is weak:
- the Lehmann reversal grosses about 1.6 bps/day (P2);
- quintile sorts gross 1–4 bps/day (P1, P4).

The network can only pass if it finds structure the linear and OU signals miss. With about 500 names, daily data and a one-session skip, a pass is not expected.

## 6. Software and run

- **PyTorch for the GTX 1070.** The card is Pascal (sm_61). Official PyTorch wheels still support it only in the CUDA 12.6 build, up to PyTorch 2.14:
  `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`
- **Notebook 01** downloads the Yahoo history (about 500 tickers, a few minutes). It builds PCA residuals (minutes) and runs 18 rolling fits × 5 seeds. At about 1–2 minutes per fit on the GPU, that is 2–3 hours. Every fit is cached, so a restart resumes.
- **Notebook 02** evaluates the predictions: backtests, placebos and gates, in minutes.
- **Fallbacks.** If CUDA is not available, the notebook runs on the CPU, about 5–10× slower. `P6_SYNTHETIC=1` smoke-tests both notebooks offline.
