# P5 002 — Readout (run d9955d8)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Protocol | `001_claude_p5_garleanu_pedersen_preregistration.md` |
| Verdict | **`NO_EDGE`**. No arm passes G1 or G3 in either window. G2 (beats shuffled forecasts) passes for q12 and q90 in both windows. |

## 1. Result

| Arm | τ / bar | A net | A gross | B net | B gross | Turnover / yr, A | Cost / yr, A |
|---|---|---|---|---|---|---|---|
| q12 | 0.25 | −32.0% | +0.9% | −16.9% | +4.6% | 235× | 7.1% |
| q90 | 0.10 | −13.9% | +1.0% | −6.4% | +3.1% | 97× | 2.5% |
| q600 | 0.04 | −8.1% | −3.4% | −2.6% | +0.7% | 27× | 0.7% |
| *Ablations at q90* | | | | | | | |
| noPW | 0.10 | −12.3% | +1.7% | −6.4% | +2.8% | 99× | 2.5% |
| Aonly | 0.10 | −3.5% | **+13.0%** | −5.2% | +4.2% | 108× | 2.8% |
| Bonly | 0.10 | −19.9% | −10.5% | −5.4% | +2.6% | 78× | 2.0% |

- Every arm is net negative in every year of both windows.
- The partial-adjustment rule works mechanically: turnover falls from 235× to 27× a year, and costs from 7.1% to 0.7% of equity. **There is no gross return left to keep.**

## 2. Why

1. **The slow signal predicts nothing.** The median ex-ante slope of the next-bar hedged return on the 5-session residual is 0.0002 (A) and −0.0012 (B), against 0.003 / 0.007 for the fast signal. Its decay is half as fast (φ_B ≈ 0.009 vs φ_A ≈ 0.020 per bar), so the persistence weighting moves the book *toward* a signal with no forecasting value. Bonly loses even gross in A (−10.5%), and adding B to A cuts A's gross from +13.0% to +1.7%.
2. **A continuous book dilutes the fast signal.** Aonly earns +13.0% gross over six years in A, against +33.4% for the S05 threshold rule and +49.6% with the earnings filter (P4). Holding every name in proportion to its s-score spreads the book over the many small s-scores that S08 showed carry no edge. Only |s| > 3 paid.
3. **Beating the shuffle is not enough.** Real forecasts beat permuted ones (q12 and q90 at the 100th percentile in both windows), so the signal is real, but it is worth about 1–5% gross over 3–6 years.

## 3. Consequences

- P5 closes `NO_EDGE`. Gârleanu–Pedersen, like S07's no-trade band before it, solves the cost problem only by trading less, and these signals have too little gross per unit of exposure to survive that.
- Across S05–S09 and P1–P5, the 15-minute sector-residual edge is confined to the largest residuals, in 2020–2025, outside earnings windows. No portfolio construction tested so far turns it into a net profit in both windows.
