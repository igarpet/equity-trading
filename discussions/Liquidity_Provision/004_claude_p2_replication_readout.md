# P2 004 — Replication readout: VIX-gated S05 on 2017–2019 (run 8aba3d3)

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-05 |
| Protocol | `003_claude_p2_frozen_spec_replication_2017_2019.md` |
| Verdict | **`NOT_REPLICATED`** (V1, V2, V3 all false). The 2026 holdout stays sealed. |

## 1. Result

| Arm | Net 1× | Net 2× | Gross 0× | Gross / trade | Cost / trade |
|---|---|---|---|---|---|
| ALL | −6.1% | −13.0% | +1.2% | 1.6 bps | 10.0 bps |
| VIX | −4.8% | −9.8% | +0.8% | 1.5 bps | 10.6 bps |

- **V1.** High-regime entries earn 6.2 bps gross, low-regime entries −2.7 bps. The difference is 9.0 bps, 95% [−10.2, +28.1]. The sign is the same as in 2020–2025, but the size is about a fifth of the 42.8 bps seen there, and the interval includes zero.
- **V2.** The gated arm sits at the 40th percentile of the 200 shifted-regime placebos on return and the 44th on Sharpe.
- **V3.** Net negative at 1× and 2× cost. Net by year: 2017 −1.3%, 2018 +0.3%, 2019 −3.9% (ungated: −2.5%, −0.3%, −3.3%).

Data were complete: 480 traded names; 12 quote sessions, median half-spread 1.76 bps; the regime was high on 46% of sessions.

## 2. Reading

The gate did not fail on its own. **The signal it gates has no gross edge in 2017–2019.** Ungated S05 earned 1.6 bps gross per trade, against 20.2 in 2020–2025. Costs were *lower* in 2017–2019 (10.0 vs 13.0 bps per trade), so cost is not why it failed.

The sector-ETF residual reversal at 15 minutes (S05a–S09, P1, P2) is therefore a 2020–2025 phenomenon in our data, concentrated in its volatile stretches (2020, 2024–2025). The VIX gate picked the best part of a period-specific effect. The protocol fixed this reading in advance: the result is specific to 2020–2025, and the 2026 holdout is not spent on it.

Caveats that limit, but do not reverse, this reading:
- 2017 was an unusually calm year.
- The current-constituent universe has more survivorship bias further back.
- Communication Services had no hedge before XLC existed (from mid-2018).

None of these explains a gross edge near zero over three years.

## 3. Consequences

1. The VIX-gated S05 candidate is closed. No strategy built on the S05 residual goes to the holdout without new evidence from outside 2020–2025.
2. **Proposed working rule for P3–P8:** every study that finds something in 2020–2025 is re-run unchanged on 2017–2019 before it is called a candidate. This cost one evening here and saved the holdout.
3. P3 (intraday periodicity) tests a mechanism unrelated to the sector residual, so it is not compromised by this result.
