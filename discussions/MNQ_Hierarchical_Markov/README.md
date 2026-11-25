# MNQ hierarchical Markov discussions

Price-only Markov regimes on MNQ across timeframes: daily → 1 hour → 15 minutes. Notebooks: `../../notebooks/MNQ_Hierarchical_Markov/`.

| Turn | Content |
|---|---|
| `001_claude_hmk01_preregistration.md` | Hierarchical HMM with trading gates. Run `85560a6` (kept in `withdrawn_hmk001/`): 1 h and 15 m `CONDITIONING_ONLY` (daily parent improves the density, no direction skill, net ≈ 0 vs +21 ticks/day long); superseded by HMK 002 at Iván's request |
| `002_claude_hmk01_transition_matrices.md` | HMK 01 (descriptive): video rule (20-bar sum, ±5%; intraday also √time-scaled band) transition matrices for daily, 1 h, 15 m; persistence vs shuffled returns and non-overlapping blocks; P^n, stationary distribution; 2020-01 – 2023-06 |
| `003_claude_hmk01_readout.md` | HMK 01 (6bb2ce5): daily p_stay 0.82–0.85 = shuffled-return reference 0.83–0.87; non-overlapping diagonal ≤ shares; intraday ±5% gives 99% sideways; scaled bands: only sideways persists beyond the overlap (volatility clustering); next-bar up share 51–56% in every state = base rate |
| `004_claude_hmk02_longer_windows.md` | HMK 02 (descriptive): 1 h W = 40 / 60, 15 m W = 60 / 80; set √time-scaled bands (±1.47 / 1.81%, ±0.90 / 1.04%) vs a 3-state HMM on the window return; same matrices and persistence references as HMK 01 |

Next turn: `005`.
