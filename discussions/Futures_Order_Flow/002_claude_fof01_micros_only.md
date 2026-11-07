# FOF 002 — Amendment before any data: micro contracts only

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-07 |
| **Branch** | Futures_Order_Flow |
| **Responds to** | FOF 001; Ivan, 2026-11-07: "we are not going to touch anything that is not a micro contract" |
| **Status** | Amendment. No Databento data bought, no NinjaTrader data loaded; the first cost table (run without download) prompted it. |

## 1. Decision

The branch studies and trades **MNQ, MES and MYM only**. Full-size NQ, ES and YM are neither requested, loaded nor used as predictors.

- **Code.** `futures.py` 0.2.0 refuses non-micro roots in `plan_requests`, `cache_dir` and `load_nt_root`. `read_recorder` drops non-micro instruments.
- **Signal source.** The flow measured is MNQ's own taker flow, which is also the book Ivan executes on.
  - MNQ and NQ share the price path to within a tick.
  - Their flow differs: MNQ is more retail, with smaller trades and a thinner book.
  - The CS03c design reference was the perp's own flow, so this is the consistent choice.
- **Not tested as a consequence:** whether NQ flow leads MNQ. Recorded here so it is not later presented as tested.

## 2. Changes to FOF 001

| FOF 001 | Now |
|---|---|
| Signals from NQ/ES/YM flow, execution modelled on micros | Signals and execution on the same micro contract |
| Spend order NQ → ES → YM | **MNQ 2025 → MNQ 2024 → MNQ 2023**, then MES 2025, MYM 2025 if budget remains |
| NinjaTrader exports and recorder on NQ/ES charts | MNQ (and MES/MYM) charts; folders `export/MNQ/`, `export/MES/` |

## 3. Open before the first paid request

The first cost table (`tbbo`) priced about $16–21 per full-size contract-month, about $26/GB. That is far above the $0.50/GB Ivan read on Databento's site. The schema and the DEV/TEST windows are therefore **not fixed yet**. They are set in the next turn from:

- `metadata.list_unit_prices` for GLBX.MDP3;
- the MNQ month cost under `trades`, `tbbo`, `mbp-1` and `mbo`.

The pre-registered checks C1–C5 and the $115 cap are unchanged. The strategy pre-registration becomes **FOF 003**.

## Decisions needed from Ivan

1. Run the cost comparison and post the output. Schema and windows are chosen from it, before anything is bought.
