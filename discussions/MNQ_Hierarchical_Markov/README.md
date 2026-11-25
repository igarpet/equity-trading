# MNQ hierarchical Markov discussions

Price-only Markov regimes on MNQ across timeframes: daily → 1 hour → 15 minutes. Notebooks: `../../notebooks/MNQ_Hierarchical_Markov/`.

| Turn | Content |
|---|---|
| `001_claude_hmk01_preregistration.md` | Hierarchical HMM with trading gates. Run `85560a6` (kept in `withdrawn_hmk001/`): 1 h and 15 m `CONDITIONING_ONLY` (daily parent improves the density, no direction skill, net ≈ 0 vs +21 ticks/day long); superseded by HMK 002 at Iván's request |
| `002_claude_hmk01_transition_matrices.md` | HMK 01 (descriptive): video rule (20-bar sum, ±5%; intraday also √time-scaled band) transition matrices for daily, 1 h, 15 m; persistence vs shuffled returns and non-overlapping blocks; P^n, stationary distribution; 2020-01 – 2023-06 |

Next turn: `003`.
