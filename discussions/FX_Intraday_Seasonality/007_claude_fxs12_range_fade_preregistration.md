# FXS 007 — Fading moves with OCO limit orders (pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-12 |
| **Study** | FXS 12 |
| **Responds to** | Iván, 2026-11-12: "sell volatility" with limit orders on both sides, cancelling the one not hit |
| **Status** | Written before any result. New hypothesis: no earlier FXS result supports it. |

## 1. Rule

At the top of a New York-clock hour h (h = 0 … 23, counted from 17:00 New York):
- Let M0 be the mid at that instant.
- Place a **sell limit** at M0 + D and a **buy limit** at M0 − D, OCO.

The first one filled cancels the other. After the fill:
- **take-profit:** a limit back at M0, which gains D;
- **stop:** S beyond the fill price;
- **time exit:** everything (unfilled orders and the open position) is closed at h + 2 h.

**Grid:** D ∈ {5, 10, 15, 20} pips, S ∈ {D, 2D}, h ∈ 0…23, pairs EURUSD, USDJPY and EURJPY. That is 576 tests.

## 2. Fills (1-minute bid/ask OHLC, Dukascopy, spread × 0.9, $3 per lot)

- A sell limit fills when the bid high reaches the level; a buy limit fills when the ask low reaches it. The fill is at the level.
- **Fill minute:** only the stop can trigger in that minute (conservative). If both limits are reached in the same minute, the side toward which the minute closed is taken.
- **Later minutes:** the stop is checked first (at the level, or at the open if the minute opened beyond it), then the take-profit (at the level).
- **Time exit:** at the closing quote you would exit at.
- **Strict-fill sensitivity:** the level must be traded through by 0.2 pip.
- Placements whose 2 h window crosses the Friday 17:00 close are skipped. News days are kept.

## 3. Windows

| Window | Use |
|---|---|
| 2020–2023 | discovery |
| 2024–2025 | confirmation |
| 2026 | one look, after the confirmed list is frozen |

## 4. Tests

**Discovery.** Per configuration:
- mean net pips per filled trade, Newey–West t (5 lags);
- Benjamini–Hochberg over all 576 tests, q ≤ 0.05;
- mean > 0 and the same sign in at least 3 of the 4 years.

Candidates: the best configuration per (pair, hour), at most 10 overall by t.

**Confirmation (2024–2025).** Per candidate:
- mean net > 0;
- one-sided Newey–West p, BH across candidates, q ≤ 0.05.

The strict-fill mean is reported.

**2026.** Frozen file with SHA-256. A confirmed configuration **holds** if its 2026 mean net is > 0.

**Diagnostic, not a gate.** For each configuration, the take-profit share among trades that reach either the take-profit or the stop (time exits left out), against the random-walk rate (S − spread) / (D + S). Real mean reversion shows up as a take-profit rate clearly above that benchmark.

## 5. Risk, reported for every candidate

- worst trade;
- worst day (sum over the pair's hours if several are traded);
- longest losing streak;
- share of trades stopped.

A strategy with frequent small wins and rare stops must show its stop rate next to its mean.
