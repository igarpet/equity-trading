# HMK 002 — HMK 01 redefined: descriptive Markov transition matrices (daily, 1 h, 15 m), video rule; HMK 001 withdrawn

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Branch** | MNQ_Hierarchical_Markov |
| **Responds to** | Iván, 2026-11-26: first discover the daily-return Markov probability matrix and how persistent it is, then the 1 h and 15 m matrices, with the video's rules (last 20 bars). No trading backtest |
| **Supersedes** | HMK 001 (hierarchical HMM + trading gates), withdrawn before any run; its notebook was removed. The engine code stays in `hmk.py` for a later study if Iván asks for it |
| **Engine** | `src/equity_trading/hmk.py` 0.1.0 (`rolling_states`, `transition_counts`, `chain_report`); tests `tests/test_hmk.py` |
| **Status** | Descriptive only: no gates, no verdict, no costs |

## 1. Rule (the video's)

State of bar t from the sum S_t of the last 20 bar log returns, bar t included: S_t ≥ +5% bull, S_t ≤ −5% bear, otherwise sideways. A bar can close red and still be bull if S_t ≥ 5%. Transitions are counted between consecutive bars, P[i, j] = count(i → j) / count(i → ·).

## 2. Data

MNQ `ohlcv-1m`, CME days 17:00–16:00 Chicago. Daily = CME-day close-to-close; 1 h (23/day) and 15 m (92/day) aligned to the session open, every bar in time order across days (the first bar of a day includes the break). Missing returns count as 0 in the sum; no return across a contract roll.

**Window:** 2020-01-01 – 2023-06-30, plus halves 2020-01 – 2021-09 and 2021-10 – 2023-06 for stability. The 20-bar sums use the bars before the window as run-in. 2023-07 onwards and 2026 are not read.

## 3. Bands

- Daily: ±5% (the video).
- 1 h and 15 m: the literal ±5% **and** the band scaled to the bar by √time, 5% × √(bar minutes / 1,380): ±1.04% (1 h), ±0.52% (15 m). Twenty hourly bars span less than one trading day, so ±5% will almost never be reached and nearly every bar will be "sideways"; the scaled band keeps the three states comparable across tiers. Both are reported.

## 4. Reported per tier and band

1. State shares, counts and the 3×3 matrix P.
2. Persistence: diagonal p_ii and expected duration 1 / (1 − p_ii); empirical run lengths.
3. P², P³, P⁵, P²⁰ (the video's "squaring the matrix") and the stationary distribution (the long-run share each state converges to).
4. Two references for "how persistent":
   - **shuffled returns** (200 permutations, same rule): the persistence produced by the 19 shared bars alone for returns with the same distribution and no order;
   - **non-overlapping 20-bar blocks**: the transition matrix of block t → block t + 1, which shares no bars.
5. Descriptive only: the share of next bars that close up, by state.

Persistence above the shuffled reference, or a non-overlapping diagonal above the states' shares, is persistence that is not built into the rule.

## 5. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\equity-trading
git pull
python -m pytest -q tests/test_hmk.py
cd notebooks\MNQ_Hierarchical_Markov
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 01_transition_matrices.ipynb
```

Outputs in `01/`: `01_summary.csv`, `01_matrices.csv`, `01_matrices.png`, `01_run.json`.

## 6. Next

After the readout: Iván decides the next step (e.g. an HMM to replace the ±5% thresholds, or conditioning the 1 h matrix on the daily state).
