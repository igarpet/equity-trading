# Rate Intraday Seasonality — stacked 1-minute days in US rate futures (RIS 001)

The bond seasonality design (BDS) applied to US short-rate futures from Databento (best bid/offer once a minute).

| Folder | Future | H1 | H2 | Status |
|---|---|---|---|---|
| `ZT/` | 2-year T-Note | 2016–2020 | 2021–2025 | Code ready, not run |
| `ZF/` | 5-year T-Note | 2016–2020 | 2021–2025 | Code ready, not run |
| `SR3C3/` | SOFR, 4th quarterly contract (`SR3.c.3`) | 2022–2023 | 2024–2025 | Needs `python tools/databento/rates_download.py --roots SR3C3` |

Each folder: `01_discovery_H1`, `02_discovery_H2`, `03_cross_confirmation`, `04_full_window`, as in `Bond_Intraday_Seasonality/`. Costs are the futures' own quoted bid/ask + 0.70 bp; rolls come from the instrument id.

Protocol: [`001_claude_ris01_preregistration.md`](../../discussions/Rate_Intraday_Seasonality/001_claude_ris01_preregistration.md).

`RIS_SYNTHETIC=1` (one `RIS_SYNTH_DIR` for all notebooks): ZT must return one candidate, 08:30–09:30 New York long, confirmed both ways and matched in FULL; ZF and SR3C3 nothing.
