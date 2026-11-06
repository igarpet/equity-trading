# P7 001 — Statistical arbitrage in rank space (Li & Papanicolaou): pre-registration

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Branch | Rank_Space (paper P7) |
| Source | Li, Y.-F. & Papanicolaou, G. (2024), *Statistical Arbitrage in Rank Space*, arXiv 2410.06568 |
| Reuses | P6 data and network (`p6.py`, `p6_model.py`) |
| Working rule | Both 2017–2019 and 2020–2025 judged, fully out of sample |
| Status | Pre-registration. Written before any shares-outstanding download or P7 training. |

## 1. Paper (as summarised from its abstract page)

- **Data and universe:** CRSP, 1990–2022; the top 500 US stocks by capitalisation, re-selected daily.
- **Rank space:** stocks are indexed by capitalisation **rank** instead of name. The rank-$k$ return follows "whichever stock holds rank $k$".
- **Residuals:** PCA on a 252-day window, with **1 factor in rank space** and 5 in name space.
- **Signals:** OU with thresholds 1.25 / 0.5; and a CNN–Transformer with a mean-variance objective (γ = 2), refit quarterly.
- **Trading:** rank-space returns are not directly tradable. The paper converts rank weights into stock positions with intraday rebalancing every 225 minutes.
- **Results with 2 bps costs, 2007–2022:**

| Space | Model | Return / yr | Sharpe |
|---|---|---|---|
| Rank | Network | 35.7% | 3.28 |
| Rank | Network, no costs | 206% | 9.04 |
| Rank | OU | −29.4% | −9.95 |
| Name | OU | +1.1% | 0.41 |
| Name | Network | −3.9% | −0.48 |

A no-cost Sharpe of 9 that falls to 3.3 at 2 bps says most of the gross comes from the rank re-mapping itself. **P7's question:** how much of the rank-space edge is tradable after the re-mapping trades, at measured costs?

## 2. Capitalisation and ranks (new data)

| Item | Value |
|---|---|
| Shares outstanding | SEC XBRL `companyconcept`, per CIK, in this order: `dei:EntityCommonStockSharesOutstanding`, then `us-gaap:CommonStockSharesOutstanding`, then `us-gaap:WeightedAverageNumberOfSharesOutstandingBasic`. **Point in time:** a value is used from its *filing* date until the next filing. Uses `SEC_USER_AGENT` from `.env`, as in P4. |
| Splits | Yahoo split history per ticker. Reported shares are converted to today's split basis: × the product of split ratios after the report date. |
| Price | Split-adjusted, **not** dividend-adjusted. Yahoo `Close` with `auto_adjust=False` before 2016-01-04; Alpaca `1Day` with `adjustment=split` from then on (new cache `bars_split/`). |
| Unit check | Each ticker's latest SEC share count is compared with Yahoo's current count. If the ratio is outside [0.5, 2] (share-class or unit mismatch, e.g. BRK.B), the whole SEC series is rescaled by it. Rescaled tickers are listed in the output. |
| Share classes | One ticker per CIK (the first alphabetically), so GOOG/GOOGL, FOX/FOXA and NWS/NWSA count once. |
| Cap and ranks | cap = price × split-basis shares. Each close, stocks with a cap, a price and a next-day return are ranked by cap. Ranks **1–400** are used, so the rank dimension is fixed. |

Survivorship: ranks are taken within the current S&P 500 list, not the full market (the paper's top 500).

## 3. Rank-space returns, residuals, network

- **Rank return** (the paper's object, not tradable): $r^{rank}_{k,t} = \text{cap}_{(k),t} / \text{cap}_{(k),t-1} - 1$, where cap$_{(k)}$ is the $k$-th largest cap.
- **Tradable return of rank $k$** for a position formed at the close of $d$ and traded at the close of $d+1$: the total return on $d+2$ of the stock holding rank $k$ at the close of $d+1$.
- **Residuals:** PCA with **1 factor** on the rank returns. The rest is P6's machinery: 252-session window, re-estimated daily, 30-session cumulative-residual inputs, one-session skip.
- **Network:** P6's CNN–Transformer, with weights neutralised to the rank-space factor and Σ|w| = 1.
- **Fits:** rolling 750 / 250 / 125, **3 seeds** (P6 used 5).
  - Deviations from the paper: refit every 125 sessions instead of quarterly, and 3 seeds instead of 5, to keep two arms within about 2.5 hours on the GTX 1070.

**Judged arms:**

| Arm | Training target | Objective |
|---|---|---|
| `RS-paper` | Rank returns (non-tradable), gross | Mean-variance, γ = 2 (as the paper) |
| `RS-net` | Tradable returns of the rank-mapped stock book, net of measured costs, including the re-mapping trades | Sharpe (as P6) |

**Evaluation (both arms):**
- **Book:** rank weights from the close of $d$ go to the stocks holding those ranks at the close of $d+1$ and are held to the close of $d+2$. The book is re-mapped every close.
- **Costs:** measured close-slot half-spread × |Δ stock weight|. The paper's intraday 225-minute re-mapping is not reproduced; a daily strategy re-maps at its daily trade.

## 4. Gates

Per window and arm, α = 0.05 / 2:

| Gate | Condition |
|---|---|
| G1 | Tradable net, mean daily: block-bootstrap two-sided $1-\alpha$ interval, lower bound > 0 at blocks 10, 20 and 40 |
| G2 | Tradable **gross** total return and Sharpe beat 1,000 placebos that permute the day's rank weights across ranks (upper-tail p ≤ α) |
| G3 | Tradable net total return > 0 and Sharpe > 0 at 2× cost |

Verdict: `EDGE_CANDIDATE` if the same arm passes in both windows; otherwise `NO_EDGE`.

**Reported:**
- the non-tradable rank-space gross (the paper's metric) next to the tradable gross; the gap is what re-mapping loses;
- turnover split into signal turnover and re-mapping turnover;
- a rank-space OU benchmark, using the continuous −s weight as in P6;
- P6's name-space network as the name-space comparison (not re-run).

## 5. Prior

The paper's own OU result shows that a strong-looking rank-space reversal disappears once re-mapping costs are charged. P6 found the name-space network close to zero after costs.
- **`RS-paper`:** expected to show a large gap between non-tradable and tradable gross.
- **`RS-net`:** can only pass if the network finds a rank-space signal that survives its own re-mapping trades.

## 6. Run

1. `python -m pytest -q tests/test_p7.py`
2. Notebook 01 downloads:
   - SEC shares: about 500 requests, a few minutes;
   - Yahoo splits and split-only prices for 2011–2015;
   - Alpaca `1Day` split-adjusted bars for 2016–2025: about 500 symbol-years per year, minutes.

   Then 2 arms × 19 fits × 3 seeds, about 2–2.5 hours on the GPU, cached.
3. Notebook 02 evaluates, in minutes.

`P7_SYNTHETIC=1` runs both offline.
