# RSW 002 — RSW 01 readout: regimes exist, no tradable one-bar direction

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Run** | `0d7b2af` (Iván), engine `regime_ar.py` 0.1.0 |
| **Protocol** | [`001`](./001_claude_rsw01_preregistration.md) |

## Result

| Model | R1 | R2 | R3 | R4 | E | Verdict |
|---|---|---|---|---|---|---|
| M1 (2 states) | ✓ | ✓ | ✗ | ✗ | ✗ | regimes, burst too short (R3) |
| M2 (3 states) | ✓ | ✓ | ✗ | ✗ | ✗ | regimes, burst too short (R3) |
| M3 (2 states, flow TVTP) | ✓ | ✓ | ✓ | ✗ | ✗ | `REGIMES_NO_TRADABLE_DIRECTION` |
| M4 (3 states, flow TVTP) | ✓ | ✓ | ✗ | ✗ | ✗ | regimes, burst too short (R3) |

**Label correction.** The notebook labels M1, M2 and M4 "R1_ONLY (volatility residue unless R2)". That is wrong: they pass R1 and R2 and fail R3 (expected duration under 3 bars). The gate values are right; only the label text is wrong. It will be fixed in the next engine change. Run `0d7b2af` is not re-run for it.

## The two regimes (stable across the three fits)

- **Calm**:
  - about 95% of bars;
  - φ = −0.04 to −0.07, z sd 0.6–0.8, c ≈ +0.04;
  - lasts 12–20 bars (8–9 in M4).
- **Burst**:
  - about 4% of bars;
  - φ = +0.12 to +0.29, z sd 1.8–3.0, c ≈ −0.2 (mostly selling);
  - lasts 1.4–1.9 bars, or 3–4 bars with flow.
- In M3, flow mainly affects leaving the burst:
  - imb × sign(z): +1.0 to +1.9 on the leave logit;
  - |imb|: −0.8 to −1.2.
- Flow barely affects entering the burst: the calm-state leave logits are ≤ 0.2.
- M1/M3 state shares are flat across sessions and σ̂ quintiles. M2/M4 also split the calm state by session and volatility (Asia vs NY), which is volatility structure.
- Out-of-sample, the lag-1 autocorrelation of z inside each filtered state has the sign of its φ (M1: −0.06 / +0.16).

## Why there is no tradable direction

- R²_OOS against zero is −0.03% to −0.11%, and the hit rate is 0.50.
- Clark–West against M0 is positive (Holm p 0.012–0.084): there is a signal, smaller than its estimation noise.
- One-bar trade with |forecast| ≥ 3 ticks:
  - net −1.2 to −2.3 ticks at the 3-tick cost, negative in every period;
  - also negative at the 1.6-tick cost.
- Every bar, gross: +0.32 to +0.34 ticks (t ≈ 2.9), about a tenth of the cost.
- The calm-state reversal is about 5% of the last bar.
- The burst continues, but it lasts under 2 bars and the filter recognises it only after it has started.

## Open from this readout

Iván (2026-11-22) reads the burst state as self-exciting, i.e. a Hawkes process. The evidence for clustering is P(burst_t | burst_{t−1}) = 0.30–0.47 against an unconditional 4%.

These fits cannot separate Hawkes from Markov switching, because both give clustering.

RSW 02 ([`003`](./003_claude_rsw02_hawkes_pot_preregistration.md)) tests the decay of the clustering and the out-of-sample forecast of the next exceedance, against both Markov nulls.
