# S07b: S07 with wider no-trade bands and no trading in the first 30 minutes

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-03, written after S07 (8b38eef) and before any S07b result |
| **Notebook** | `studies/S07b_wide_bands.ipynb` |
| **Engine** | `src/equity_trading/s07.py` (unchanged apart from the `skip_open_bars` option) |

## 0. Question

S07, after measured costs:

| Arm | Gross | Annual cost | Net |
|---|---|---|---|
| `band05` | ~5.2%/yr | 11.8% of equity | −34% |
| `band10` | ~4.3%/yr | 5.9% of equity | −10% |

Doubling the band halved costs but cut the gross return only by about a fifth. The signal beat the shuffled placebo at p = 0.002 in both arms.

S07b pushes the same lever further and avoids the widest spreads of the day. Both changes are chosen for cost reasons, not returns, but they follow from S07's numbers, so S07b is development evidence.

## 1. Changes from S07

| Change | Rule |
|---|---|
| Wider bands | Arms `band15` (1.5× the average position) and `band20` (2.0×) |
| No trading at the open | No fills at the 09:30 and 09:45 bar opens; orders wait for the 10:00 open (`skip_open_bars = 2`). The S05b intraday profile shows the widest spreads at the open. |
| Reference (not judged) | `band10skip`: S07's band 1.0 with the open skip, to separate the effect of the skip from the effect of the band |

**Why not 2× and 4× as first proposed:** with trade-to-edge, a stock only opens a position when its target exceeds the band. The largest target is s_cap / mean \|clip(s)\| times the average weight: about 3.7× if s is standard normal, and lower with the fatter tails seen in S05. **A 4× band would never trade.** A 2× band already limits the book to stocks with \|s\| above roughly 2, so gross exposure falls as the band widens. The table therefore reports annual return per unit of average gross exposure.

## 2. Evaluation

Identical to S07:
- gates G1 (bootstrap), G2 (shuffled signal, 500 placebos per arm) and G3 (2× measured cost);
- **α = 0.025** per arm.

**Pre-stated reading:**
- **Profitable arm:** wider bands keep enough of the signal at a sustainable turnover.
- **Gross Sharpe falls while costs fall:** the band trades away signal faster than cost, and the sector-ETF residual at this horizon is not tradable with market orders at retail spreads.

## 3. Limitations

As S07. Band widths were chosen after S07's result.
