# HMK 001 — Hierarchical Markov regimes on MNQ: daily → 1 hour → 15 minutes (HMK 01, pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Branch** | MNQ_Hierarchical_Markov (new) |
| **Responds to** | Iván, 2026-11-25/26: a price-only Markov regime model in the spirit of a YouTube "hedge fund method" (bull/sideways/bear transition matrix), but with an HMM instead of the arbitrary ±5% 20-day rule, and hierarchical: identify the daily state, then condition intraday state probabilities on it, P(intraday state \| daily state). Instrument **MNQ**, lowest timeframe **15 m**, middle **1 h**. Goal: know with P > 50% whether the next 1-hour bar is up or down |
| **Engine** | `src/equity_trading/hmk.py` 0.1.0; tests `tests/test_hmk.py` |
| **Status** | Discovery 2021-01 – 2023-06 (all fits out of sample). Confirmation 2023-07 – 2025-12 only after the discovery readout. 2026 sealed |

## 1. What is tested, and what is not

The video's state is the bucket of the 20-day return (≥ +5% bull, ≤ −5% bear). Because consecutive 20-day windows share 19 days, its "stickiness" is largely mechanical and says little about the next return. It is kept only as a baseline (`RULE`). The tested model replaces it with HMMs, which learn states from the returns themselves.

The question that matters for trading is not whether states persist but **whether the next bar's return is predictable after costs**. A hit rate above 50% is not sufficient on its own: an always-in position that flips pays a round trip per flip, so it needs (2p − 1)·E|move| > cost per flip, and drift alone already gives MNQ more up than down bars (hence the always-long benchmark).

This branch is independent of MNQ_Return_Prediction and MNQ_Regime_Switching: it reuses only the 1-minute data loader (`micro_bars.load_1m`) and the CME trade-date convention (`futures.cme_trade_date`), no fitted model, threshold or result. Those branches found 5-minute regimes to be volatility states with no tradable direction (RSW 002); this study tests whether conditioning on a higher timeframe changes that.

## 2. Data and bars

- MNQ `ohlcv-1m` (Databento, continuous front contract), already on disk. Load from 2019-11-01; nothing from 2026.
- CME day 17:00–16:00 Chicago, labelled by its end date; days with < 900 traded minutes dropped.
- Bars aligned to the session open: **1d** (close-to-close), **1h** (23 per day), **15m** (92 per day). Returns in bp, within the same contract (NaN across a roll).
- The first intraday bar of each day carries the daily break / weekend: it is never an observation and never traded.

## 3. Model

Each tier is a K = 3 Gaussian HMM on its own bar returns (price only). States are sorted by emission mean (bear < side < bull). The child's transitions depend on the parent's **one-step-ahead** state forecast w_t (known at the close of bar t−1):

$$P(s_t=j \mid s_{t-1}=i,\ \text{parent}) = \sum_g w_t[g]\,A_g[i,j]$$

| Tier | Parent weight of bar t |
|---|---|
| 1d | none |
| 1h (CME day d) | P(D_d \| F_{d−1}) from the daily HMM |
| 15m (hour h) | P(H_h \| F_{h−1}) from the hierarchical 1h model |

EM treats the parent component as latent (responsibility w_g A_g[i,j] / A_t[i,j]), so the joint P(child j, parent g) is estimated directly — not a product of two independently estimated matrices, which would assume the tiers are independent. Forecast of bar t at the close of t−1: P(s_t \| F_{t−1}) → P(up) = Σ_j P(s_t = j) Φ(μ_j/σ_j), E[r] = Σ_j P(s_t = j) μ_j. Best of 8 EM starts.

| Variant | Tiers | Role |
|---|---|---|
| `H3` | 1h and 15m with parent | **primary** |
| `U` | 1h and 15m without parent (G = 1) | benchmark for G1; also judged on its own (G2–G4) |
| `RULE` | 1h with parent = one-hot video state (20-day ±5%) of the previous day | baseline, reported |

**Walk-forward.** Expanding fits from 2020-01-01, refitted 2021-01-01, 2022-01-01, 2023-01-01; each fit is frozen and filters causally to the next refit. Training base rate of up bars per refit for the Brier benchmark.

## 4. Trading rule (for G3)

Always in sign(E[r]) from the 2nd bar of each CME day, decided at the previous bar's close; flat at the day's close (no position across the daily break). One contract. Costs per contract traded: 0.5 tick slippage + $0.50 commission = **1.5 ticks** (3 per round trip) at market; Iván's realized 1.6-tick round trip reported. Benchmark: always long, same hours, same costs.

## 5. Windows

| Window | Use |
|---|---|
| 2020-01 – 2020-12 | first fit only |
| **Discovery** 2021-01 – 2023-06 | verdict; periods 2021, 2022, 2023H1 |
| Confirmation 2023-07 – 2025-12 | `HMK_CONFIRM=1`, refits 2023-07, 2024-01, 2025-01; periods 2023H2, 2024, 2025; only after the discovery readout |
| 2026 | sealed |

## 6. Gates (one-sided, α = 0.05, Newey-West 5 lags on daily series)

| Gate | Test |
|---|---|
| **G1** conditioning | daily sum of predictive log score, `H3` − `U` > 0 |
| **G2** direction skill | Brier score of P(up) below the training up-share's: daily mean of (B₀ − B) > 0 and BSS > 0 |
| **G3** economics | daily net ticks at market cost > 0 **and** net − always-long net > 0 |
| **G4** stability | net > 0 and BSS > 0 in ≥ 2 of 3 periods |

Primary: **1h `H3`**. Secondary: 15m `H3`. `U` and `RULE` get G2–G4 verdicts too.

## 7. Verdicts

| Verdict | Condition |
|---|---|
| `EDGE_CANDIDATE` | G1–G4 pass: the hierarchy adds information and the forecast pays |
| `EDGE_WITHOUT_HIERARCHY` | G2–G4 pass, G1 fails |
| `SKILL_BELOW_COST` | G2 passes, G3 fails |
| `CONDITIONING_ONLY` | G1 passes, nothing else |
| `NO_EDGE` | otherwise |

A candidate must pass the same gates on the confirmation window, then the sealed 2026 holdout, before any trading rule is built.

## 8. Smoke runs (synthetic, `HMK_SYNTHETIC=1`)

- **Planted** (daily regime chain; hourly drift ±24 bp/h, sd 30 bp, hourly transitions depending on the daily regime): G2–G4 pass for every model. G1 fails: an intraday HMM that runs through the day boundary already infers the daily regime from recent hours, so the parent adds no measurable information there. That is a property of the model class, not a bug, and the reason G1 is a separate gate.
- **Null** (`HMK_SYNTH_NULL=1`): every gate fails, BSS ≈ 0, hit ≈ 50%.
- Unit tests: EM recovers parent-dependent transition matrices; the filter is causal; bar grid, roll and cost accounting.

## 9. Run order (PC, Windows CMD)

```bat
cd /d D:\Projects\equity-trading
git pull
python -m pytest -q tests/test_hmk.py
cd notebooks\MNQ_Hierarchical_Markov
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 01_hierarchical_markov.ipynb
cd ..\..
git add notebooks\MNQ_Hierarchical_Markov
git commit -m "HMK 01 discovery run"
git push
```

Fits are cached in `01/wf_0.1.0.pkl` (delete it to refit). Expect a few minutes.
