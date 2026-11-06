# FOF 001 — Futures order flow: data plan (Databento credit + NinjaTrader + forward recording)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-07 |
| **Branch** | Futures_Order_Flow (new) |
| **Responds to** | Ivan, 2026-11-07: trade micro index futures (MNQ/MES/MYM) through Lucid prop accounts; data "option 2" |
| **Status** | Data plan + code. No strategy result exists; no return statistic is computed in FOF 01. |

## 1. Why this branch

The repo scan (2026-11-07) left one directional candidate: **signed taker flow predicts the next 5 minutes** (deribit-market-data, Candle_Streaks CS03c/CS03e). On crypto perps the statistical gate passed on TEST (BTC, ETH), the taker gross was +0.45 bp per trade (BTC) and +0.21 bp (ETH), and costs were 12 bp round trip: about 20× short. Passive entry made it negative before fees (adverse selection).

On CME micros a round trip is 1 tick plus commissions, roughly 0.3–0.5 bp of notional. The cost hurdle falls by ~30×, so this is the only result in the stack where costs, not the signal, were binding. **Prior: ES/NQ are more efficient than crypto perps; expect a smaller gross effect and a likely fail.**

The strategy itself is pre-registered in FOF 002, after the data in this turn passes its checks. This branch does not inherit CS03c thresholds or conclusions; its design is the reference, re-specified for CME.

## 2. Sources and windows

| Window | Dates (UTC) | Source | Use |
|---|---|---|---|
| DEV | 2023-01-01 → 2024-12-31 | Databento GLBX.MDP3 `tbbo` | design, selection |
| TEST | 2025-01-01 → 2025-12-31 | Databento GLBX.MDP3 `tbbo` | binding verdict of FOF 002 |
| **Holdout A (sealed)** | 2026-01-01 → export date | NinjaTrader tick export (~1 year kept on NT servers) | one look, after the TEST verdict, frozen spec only |
| Holdout B (prospective) | recorder start → | NinjaScript recorder on the Lucid CQG feed | prospective confirmation |
| Bridge | ~2025-11 → 2025-12-31 | NinjaTrader **and** Databento | data validation only (§4), no return statistics |

- **Contracts.** Signals use full-size flow: NQ, ES (YM if the budget allows). Execution is modelled on MNQ/MES at the same price path. Micro top of book normally matches the full-size contract to the tick, and Lucid sizes (≤ 20–40 micros) do not need depth.
- **Databento request.** Continuous front month `NQ.v.0`, `ES.v.0`, `YM.v.0` (`stype_in="continuous"`, roll by previous-day volume), schema `tbbo` (every trade with the best bid/offer just before it). Monthly requests, one Parquet file per root-month.
- **Budget.** The $125 credit expires 6 months after sign-up. The code refuses any request that would take cumulative spend above `BUDGET_USD = 115` and any request that touches 2026.
  - Spend order: NQ 2025 → ES 2025 → NQ 2024 → ES 2024 → NQ 2023 → ES 2023 → YM.
  - If the credit does not cover DEV for both roots, DEV shrinks to what is covered. TEST stays 2025 in full.

## 3. Common tick format

Every source is normalised to one table per root: `ts` (UTC, ns), `price`, `size`, `bid`, `ask`, `sign` (+1 buyer-initiated, −1 seller-initiated, 0 unknown), `sign_method` (`exchange`, `quote`, `tick`, `none`), `contract`, `source`.

- **Databento:** `sign` comes from the exchange aggressor flag (`side`: `B` → +1, `A` → −1, `N` → 0). `bid`/`ask` are `bid_px_00`/`ask_px_00` (book before the trade). `ts` = `ts_event`.
- **NinjaTrader export:** no aggressor flag. Quote rule against the bid/ask carried with each tick (price ≥ ask → +1, ≤ bid → −1), tick rule for prices inside the spread, 0 if neither applies.
  - Accepted formats: `yyyyMMdd HHmmss fffffff;last;bid;ask;volume`, or separate Last / Bid / Ask exports merged as-of.
  - The export time zone must be declared; it is not assumed.
- **Recorder:** same as NinjaTrader; the recorder writes UTC directly.
- **Roll for NinjaTrader** (one export per contract month): for each CME trade date (17:00 Chicago boundary), use the contract with the highest volume on the previous trade date. This is the same rule as Databento's `v.0`. The first trade date of a file set is dropped.

## 4. Checks in FOF 01 (no return statistics)

| Check | Pass |
|---|---|
| C1 Coverage | every CME trading day in DEV/TEST present for each downloaded root; gaps listed |
| C2 Time zone | NinjaTrader vs Databento per-minute volume on the bridge: best lag = 0 minutes |
| C3 Signing | NinjaTrader quote/tick-rule sign agrees with the exchange aggressor flag on ≥ 95% of bridge volume (trades matched by minute, price and size) |
| C4 Unsigned share | ≤ 2% of volume with `sign = 0` in each source |
| C5 Roll | no day where the stitched NinjaTrader contract differs from Databento's `v.0` contract on the bridge |

If C2 or C3 fails, Holdout A cannot be scored against TEST without a correction, which must be written in a new turn before any holdout use. For the holdout itself, FOF 01 reports only day counts; prices, flows and returns from 2026 are not read.

## 5. What Ivan runs

1. `pip install -r requirements.txt` (adds `databento`), then put `DATABENTO_API_KEY=` in `.env`.
2. `notebooks/Futures_Order_Flow/01_data_cost_and_coverage.ipynb` with `DOWNLOAD = False`. It prints the price of every planned request and the cumulative total. **Nothing is billed.**
3. Decide the request list from that table, set `DOWNLOAD = True`, run again.
4. NinjaTrader → *Tools → Options → General*: set the time zone to **UTC** (avoids the DST hour; set it back afterwards if you prefer). Then *Historical Data* → load NQ and ES at **Tick** for each contract month since Nov 2025 → *Export*. Put the files in `<EQUITY_TRADING_DIR>/NinjaTrader/export/<ROOT>/`, one folder per root. Run section 3 of the notebook; C2 confirms the time zone against Databento.
5. Install `tools/ninjatrader/OrderFlowRecorder.cs` (NinjaScript Editor → Indicators), add it to an NQ and an ES chart, and leave NinjaTrader running during sessions. Files appear under `Documents/NinjaTrader 8/OrderFlowRecorder/`; copy them to `<EQUITY_TRADING_DIR>/NinjaTrader/recorder/` before running section 4.

## Decisions needed from Ivan

1. Approve the windows and gates above before the first paid request.
2. After the cost table: which roots/years to buy within the $115 cap.
