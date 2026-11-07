# NinjaTrader tools (Futures_Order_Flow)

## OrderFlowRecorder.cs — forward recording for Holdout B

1. NinjaTrader 8 → *New → NinjaScript Editor* → right-click *Indicators* → *New Indicator*, name it `OrderFlowRecorder`. Replace the generated code with this file and press F5 to compile.
2. Open a chart of the front **MNQ** contract (any bar type), *Indicators* → add `OrderFlowRecorder`. Repeat for MES (and MYM if wanted). Micro contracts only: files from other instruments are ignored by the loader.
3. Leave NinjaTrader connected during the sessions you want recorded (the Lucid CQG connection works; the indicator never trades).
4. Files are written to `Documents\NinjaTrader 8\OrderFlowRecorder\` as `<instrument>_<yyyy-MM-dd>.csv` (UTC days). Copy them to `<EQUITY_TRADING_DIR>\NinjaTrader\recorder\` before running section 4 of `notebooks/Futures_Order_Flow/01_data_cost_and_coverage.ipynb`.

**Roll weeks** (Mar/Jun/Sep/Dec, about 8 days before expiry): add the recorder to a chart of the next contract as well, until volume has moved. The loader keeps, per CME trade date, the contract with the highest volume on the previous trade date, so recording both contracts around the roll is required. Recording the old one too long does no harm.

Gaps (PC off, disconnects) are expected; the notebook lists the recorded trade dates. Days with partial coverage are flagged rather than silently used.

## Tick exports for Holdout A

*Tools → Options → General*: set the time zone to **UTC** first. Then *Tools → Historical Data*: load each MNQ (and MES) contract month at **Tick** (as far back as the servers keep, about one year), then *Export* each one as tick data. Keep NinjaTrader's file names (`MNQ 12-25.Last.txt`) and put them under `<EQUITY_TRADING_DIR>\NinjaTrader\export\MNQ\` and `...\MES\`.

If the Last export has only three fields (`time;price;volume`), also export Bid and Ask for the same contract. The loader merges them, taking the quote strictly before each trade.
