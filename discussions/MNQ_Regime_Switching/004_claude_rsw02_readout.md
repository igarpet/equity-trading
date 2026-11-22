# RSW 004 — RSW 02 readout: Markov regimes suffice; extremes cluster briefly, not as a long-memory Hawkes process

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-22 |
| **Branch** | MNQ_Regime_Switching |
| **Run** | `b808aef` (Iván), engine `hawkes.py` 0.1.0 |
| **Protocol** | [`003`](./003_claude_rsw02_hawkes_pot_preregistration.md) |

## Result

| Gate | Result |
|---|---|
| **H1** (clustering beyond the chain, M1, M2) | Passes at u(0.96): lower bounds 0.15 / 0.20 / 0.21. **Fails** at u(0.98) and u(0.99) against M1. On z0 it passes. |
| **H2** (Hawkes beats MC, M1, M2) | **Fails.** Each Hawkes model beats the chain (HX, HP: Holm p ≤ 0.005) but **loses** to M1 (t −3.7 to −4.6) and to M2 (t −6.2 to −7.9). Same at u(0.98) and u(0.99). |
| **H3** | Not reached. |

**Verdict:** `SLOW_CLUSTERING_NO_FORECAST_GAIN`. Whether the clustering is self-excitation or volatility persistence left in z is not identified.

## Readings

- **Excitation is weak and short.**
  - HX: branching ratio n = 0.11–0.13, half-life 1.6–2.0 bars.
  - HP: n = 0.18–0.23, median lag 6–7 bars.
  - Marks (HXM) add nothing.
  - About 10–20% of exceedances are offspring. This is the RSW 01 burst regime, not long-memory excitation.
- **Excess clustering is confined to moderate exceedances.** It disappears at the 0.99 quantile, which matches the Markov smoke run's false-positive pattern (leftover volatility persistence).
- **The regime filters forecast the next exceedance better than event counting.** They use the size of the last bar and the regime probability. Every model, M1/M2 included, puts about 2.1× the base rate in its top 5% of predictions and catches about 11% of exceedances there.
- **Sizes** (descriptive):
  - GPD ξ = 0.20–0.28: a heavy tail.
  - κ = 0.32–0.45: sizes grow with intensity (LR p ≤ 0.001 in every fit; out of sample κ = 0.45, p < 1e-10). An exceedance soon after another is larger.
- **Sign** (descriptive, not tested):
  - Exceedances within 12 bars of each other have the same sign 0.595 of the time, against 0.526 when further apart.
  - 54% of exceedances are negative.
  - This resembles the MRP 03 continuation, which failed out of sample. Every window 2020–2025 has now been seen.

## Open

Iván (2026-11-22) proposes that the time spent in a regime carries information that an HMM discards by construction.

RSW 03 ([`005`](./005_claude_rsw03_duration_preregistration.md)) tests this three ways:
- the model-free hazard;
- an HSMM against the HMM;
- direction by regime age.
