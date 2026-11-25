# HMK 004 — HMK 02: longer intraday windows (1 h: 40 / 60 bars; 15 m: 60 / 80 bars), set bands vs HMM (descriptive)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Responds to** | Iván, 2026-11-26: ±5% is unrealistic intraday (black-swan size) but fine for daily. 20 hourly bars is less than a day: use 40 or 60 for 1 h, 60 to 80 for 15 m, to capture the trend. Either set the bull / bear % ourselves or let an HMM decide |
| **Engine** | `hmk.py` 0.1.0 (`band_for_window`, `chain_report`, `hmm_report`); tests `tests/test_hmk.py` |
| **Status** | Descriptive, no gates, no trading test |

## 1. Cases

| Tier | W (bars) | Span | Set band (a) |
|---|---|---|---|
| 1d | 20 | 20 days | ±5% (reference, as HMK 01) |
| 1h | 40 | ≈ 1.7 days | ±1.47% |
| 1h | 60 | ≈ 2.6 days | ±1.81% |
| 15m | 60 | 15 hours | ±0.90% |
| 15m | 80 | 20 hours | ±1.04% |

S_t = sum of the last W bar log returns, bar t included.

**(a) Set band:** the daily rule scaled by √time, 5% × √(W × bar minutes / (20 × 1,380)), so a W-bar window needs a move of the same size relative to its own typical range as 5% over 20 days. Iván can replace these numbers; they are one argument.

**(b) HMM:** 3-state Gaussian HMM on S_t, fitted on the window (in sample, descriptive). States are sorted by mean and taken as the argmax of the **causal filtered** probabilities (no look-ahead). Reported: each state's mean and sd of S_t in % (the bull / bear returns the HMM 'expects') and the implied thresholds (midpoints between adjacent means).

## 2. Reported for every case (as HMK 01)

State shares, counts, P, persistence (diagonal, expected duration, run lengths), P², P³, P⁵, P²⁰, stationary distribution, and the share of next bars that close up by state (descriptive). Two persistence references:
- **shuffled returns**, same W and rule (200 permutations) or HMM refitted on each permutation (20);
- **non-overlapping W-bar blocks**.

Longer windows share more bars between t and t + 1 (39 of 40, 79 of 80), so the overlap alone makes the chains stickier than with W = 20; the two references measure how much of the persistence is that.

**Window:** 2020-01 – 2023-06 and halves 2020-01 – 2021-09 / 2021-10 – 2023-06 (references on the full window only).

## 3. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\equity-trading
git pull
python -m pytest -q tests/test_hmk.py
cd notebooks\MNQ_Hierarchical_Markov
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 02_longer_windows.ipynb
```

Outputs in `02/`: `02_summary.csv`, `02_matrices.csv`, `02_hmm_states.json`, `02_matrices.png`, `02_run.json`. The HMM references refit 20 times per case: expect several minutes.
