# HMK 006 — HMK 03: are the daily, 1 h and 15 m state probabilities independent of each other? (descriptive)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Responds to** | Iván, 2026-11-26: with persistence, probabilities and HMM durations from HMK 02, move to the hierarchy. A daily bull run contains retracements that the smaller timeframes resolve; the aim is P(15 m state at t+1 \| current 1D state). First: check whether the tiers' probabilities are independent of each other. Direction is not the focus of this notebook |
| **Engine** | `hmk.py` 0.1.0 (`tiers_from_15m`, `dependence`, `circular_shift_null`, `shuffle_within_slot`); tests `tests/test_hmk.py` |
| **Status** | Descriptive, no gates |

## 1. States (as HMK 02)

Daily: 20 days, ±5%. 1 h: W = 40 or 60. 15 m: W = 60 or 80. Intraday tiers either (a) the √time set band or (b) the 3-state HMM on the window return, fitted once on the window and then used as a fixed classifier (causal filtered argmax). 2 methods × 2 × 2 = 8 cases.

All tiers are built from the same 15 m returns (an hour is the sum of its four 15 m returns, a day of its 92), so a shuffle of 15 m returns carries through to every tier.

## 2. What is tested

Parent of a bar = the parent's **last completed** bar (no look-ahead): 15 m in day d → daily state of d − 1; 15 m in hour h → 1 h state of h − 1; 1 h in day d → daily state of d − 1. Links: **15m \| 1d**, **1h \| 1d**, **15m \| 1h**.

1. **State independence:** P(child \| parent) = P(child)? Statistic: mutual information I(child; parent).
2. **Transition independence:** P(child_{t+1} \| child_t, parent) = P(child_{t+1} \| child_t)? Statistic: conditional mutual information I(child_{t+1}; parent \| child_t). This is the P(15 m state at t+1 \| current 1D state) question.

Tables reported: P(child \| parent) next to the child's overall shares, and the child's 3×3 transition matrix under each parent state next to the pooled matrix.

## 3. References

The rows are heavily serially dependent, so a χ² p-value would be meaningless. Two resampling references instead:

| Reference | How | Answers |
|---|---|---|
| Circular shift (500) | the parent's state series shifted by a random whole number of days (≥ 60); each tier keeps its own persistence and time of day | are the tiers independent at all? |
| Shuffled returns (200) | 15 m returns permuted across days within each time-of-day slot; every tier rebuilt from them; same classifiers | is there dependence beyond what shared returns produce? (the parent's window contains the child's bars, so some dependence is arithmetic) |

p = share of the reference at or above the real statistic. On a random walk (smoke run) the 15m \| 1h link is "dependent" against the circular shift but no more dependent than the shuffled returns, as expected.

**Window:** 2020-01 – 2023-06 (references); halves 2020-01 – 2021-09 / 2021-10 – 2023-06 (statistics only).

## 4. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\equity-trading
git pull
python -m pytest -q tests/test_hmk.py
cd notebooks\MNQ_Hierarchical_Markov
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 03_tier_independence.ipynb
```

Outputs in `03/`: `03_stats.csv`, `03_conditional_states.csv`, `03_conditional_transitions.csv`, `03_halves.csv`, `03_transitions_15m_by_daily.png`, `03_run.json`.
