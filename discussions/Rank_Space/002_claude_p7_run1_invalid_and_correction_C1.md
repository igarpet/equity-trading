# P7 002 — Run 3bf6ff1 is invalid (corrupted rank returns); correction C1

| | |
|---|---|
| Author | Claude |
| Date | 2026-11-06 |
| Protocol | `001_claude_p7_rank_space_preregistration.md` |
| Status | The first run's verdict (`NO_EDGE` for both arms) **is void**: the rank-space returns it was trained and evaluated on are corrupted. Correction C1, below, is fixed before the re-run. |

## 1. Evidence

From notebook 01's own check: *rank return vs tradable return of the same rank: corr −0.002, s.d. 31.74 vs 0.0192.*

A rank return (the change in the k-th largest cap) should have a daily s.d. close to a stock return's, about 0.02. It was 31.7, so some days carried cap ratios in the hundreds or thousands. Consequences:
- **Non-tradable rank gross:** −448 to −496 bps/day in 2020–2025, Sharpe −0.4. It is meaningless.
- **`RS-paper` training:** it was trained on these returns, and its validation objective is flat at about 0.3 for every fit.
- **Inputs for both arms:** the residual inputs come from the same series, so both arms are affected.

## 2. Causes

1. **Shares coverage.** 0.1.0 kept only the first SEC tag that returned any data. For companies whose `dei` cover-page count is mostly dimensional (per share class), `companyconcept` returns only a few non-dimensional facts:
   - XOM: 2 filings, the first in 2026;
   - V and CMCSA: 2 filings each, in 2009.
   
   These stocks are missing for years or carry stale counts.
2. **Rank shifting.** 0.1.0 computed rank returns from each day's sorted caps over that day's valid stocks. When a large stock enters or leaves the data (a missing share count, a missing price or return), every rank below it shifts by one, and the rank-k "return" becomes the ratio of two different companies' caps.
3. **Unit errors.** For example, FOX's only `dei` fact is 1 share.

## 3. Correction C1 (`p7.py` 0.1.1)

1. **All three share tags are fetched and merged** by filing date, in priority order `dei` > `CommonStockSharesOutstanding` > weighted-average basic. They are cached in a new folder (`SEC/shares_all/`).
2. **Cleaning on the split basis:**
   - drop isolated spikes, meaning a filing off by more than 1.5× from both neighbours in the same direction;
   - drop values more than 3× from the symbol's median;
   - then apply the Yahoo unit check, unchanged.
3. **Common-set rank returns:** $r_{k,t}$ = k-th largest cap at $t$ over k-th largest cap at $t-1$, both over the stocks with a cap on **both** days.
4. **Data gate before training:** notebook 01 stops if the rank-return s.d. exceeds 3× the tradable-return s.d. It also prints the days with the largest |rank return|.
5. **New caches:** all caches move to `results/p7/v0.1.1/`, so nothing from the first run is reused.

The model, arms, evaluation, gates and α are unchanged. C1 changes only the data definitions, and it is fixed before any re-run result exists.

## 4. Not concluded from the void run

The first run's tradable figures are reported here only as a record, not as evidence:

| Arm | Net, B 2017–2019 | Net, A 2020–2025 |
|---|---|---|
| `RS-net` | +3.8% | +17.4% |
| `RS-paper` | −4.8% | −1.3% |

Neither beats permuted weights on gross (percentiles 3–34).
