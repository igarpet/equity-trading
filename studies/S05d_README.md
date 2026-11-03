# S05d: entry rules on the x10 exits

| | |
|---|---|
| **Status** | Pre-registration, 2026-11-04, written after S05c (1fcef86) and before any S05d result |
| **Notebook** | `studies/S05d_entry_rules.ipynb` |
| **Engine** | `src/equity_trading/s05.py` 0.1.2 (`entry_mode`, `peak_*`; default behaviour unchanged, test-checked) |
| **Data** | As S05b/S05c. 2020–2025 only. |

## 0. Question

In S05c, the 100-position book was always full in every arm (average 99.7–100 open positions). Each freed slot went to the largest \|s\| available: median entry \|s\| was 3.5 under `x10`, and 99.6% of entries had \|s\| > 2. So the ±1.25 threshold almost never decided anything. About 30% of `x10` entries were filled at 09:45, from the first bar's s, which contains the overnight gap. Random contrarian timing with the same exits did about as well (58th percentile).

S05d tests two entry rules (Iván's choice), separately and combined, on the `x10` exits:
- **A threshold above the capacity-binding level:** \|s\| ≥ 3.
- **A peak requirement:** enter only once the excursion has started to revert.

## 1. Arms (three, fixed; all with `x10` exits: close long at s > +1.0, short at s < −1.0)

| Arm | Entry, for a flat tradable stock at a bar close |
|---|---|
| `t30` | s ≤ −3.0 → long; s ≥ +3.0 → short |
| `pk` | **Peaked excursion**, see the definition below |
| `t30pk` | As `pk`, but the excursion's peak \|s\| must have reached ≥ 3.0 |
| `x10` (reference, not judged) | S05c: \|s\| ≥ 1.25 |

**Excursion, as implemented:**
- At each bar close, if \|s\| ≥ 1.25 with the same sign as at the previous close, the excursion continues and its peak is the running maximum of \|s\|.
- If \|s\| ≥ 1.25 with a new sign, a new excursion starts.
- If \|s\| < 1.25, there is no excursion.
- Excursions reset at the start of each session, because the parameters are re-estimated daily. A consequence is that the overnight bar alone can never trigger a peaked entry.

**Peaked entry:** the peak ≥ `peak_open` (1.25 for `pk`, 3.0 for `t30pk`), \|s\| ≤ peak − 0.25, and \|s\| ≥ 1.25. The position is contrarian to the sign of s.

**Unchanged:**
- at most 100 positions, freed slots to the largest current \|s\|;
- 1% per stock, β-hedged in the sector ETF;
- market orders at the next open, S05b measured costs;
- exit when the stock becomes non-tradable.

## 2. Evaluation

As S05c, with **α = 0.05 / 3 = 0.0167** per arm.

| Gate | Condition |
|---|---|
| **G1** | Block-bootstrap 95% CI of mean daily net return > 0 (blocks of 10, 20, 40 days) |
| **G2** | Random direction on the same trades, p ≤ α |
| **G3** | 300 random-timing placebos per arm: contrarian entries at random flat tradable bars with s < −0.50 or s > +0.75, at the arm's own entry rate, with `x10` exits; total return and Sharpe each p ≤ α |
| **G4** | Net return > 0 and Sharpe > 0 at 2× measured cost |

**Reported:**
- trades, gross and cost per trade, gross/cost ratio;
- median entry \|s\|, average open positions (does the arm leave capacity unused?);
- share of entries filled at 09:45.

**Pre-stated reading:**
- **G3 is the decisive gate.** An entry rule that adds value must beat random timing under the same exits, which no rule has done so far.
- **`t30` will leave slots empty** (fewer qualifying stocks), lowering exposure. Judge it on Sharpe and on gross per trade, not on total return alone.

## 3. Limitations

As S05c. The entry rules were proposed after S05a–S05c were seen, and the `x10` exits were chosen from S05c's results. This is development evidence on data used many times. A pass is a lead that would need forward confirmation.
