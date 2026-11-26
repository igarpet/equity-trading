# HMK 007 — Readout of HMK 03 (run `70084c7`): the tiers are not independent; the daily state mostly sets the intraday volatility regime

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-26 |
| **Responds to** | Iván: HMK 03 run `70084c7` (2020-01 – 2023-06) |
| **Status** | Descriptive readout (HMK 006) |

## 1. Independence tests (8 cases; p from 500 circular shifts / 200 shuffled-return sets; minimum p 0.002 / 0.005)

| Link | State dependence I(child; parent) | Transition dependence I(child_{t+1}; parent \| child_t) |
|---|---|---|
| **15m \| 1d** | dependent in all 8 (p_shift 0.002); **beyond shuffled returns in all 8** (p 0.005): MI 0.020–0.031 vs 0.001–0.005 | dependent and beyond shuffled in all 8: CMI 0.0017–0.0025 vs 0.0001–0.0002 |
| **1h \| 1d** | all 8 beyond shuffled (p 0.01–0.025): MI 0.029–0.056 vs 0.017–0.034 | all 8 beyond shuffled (p 0.005) |
| **15m \| 1h** | band: **no excess** (p 0.75–0.99; the dependence is the shared bars). HMM: beyond shuffled (p 0.005) | band: mixed (2 of 4 beyond); HMM: no excess (p 0.42–1.00) |

Both halves give the same statistics within a factor of about 1.5 (e.g. 15m \| 1d MI 0.022–0.033 in H1, 0.017–0.025 in H2).

So: **the probabilities are not independent.** The daily state changes both the share and the transitions of the 1 h and 15 m states by more than shared returns can explain. Between 1 h and 15 m (band), the dependence is the overlap of their windows.

## 2. What the dependence looks like (band, 1h W40 / 15m W80; the HMM cases look the same)

P(15 m state | previous day's state):

| daily state | 15m bear | 15m side | 15m bull |
|---|---|---|---|
| bear | **34.3%** | 36.9% | **28.9%** |
| side | 17.6% | 62.4% | 20.1% |
| bull | 12.4% | **66.5%** | 21.1% |
| overall | 18.8% | 59.2% | 22.0% |

- After a **daily bear** state, *both* extreme 15 m states are more frequent (bear 34% vs 19%, bull 29% vs 22%) and sideways much rarer (37% vs 59%): a daily bear state is a turbulent regime, with violent moves both ways, tilted down.
- After a **daily bull** state, 15 m is calmer than usual (66% sideways), bear 15 m windows are rarer (12% vs 19%) and bull ones are not more frequent (21% vs 22%).
- 1 h \| 1d shows the same pattern (bear 37% / bull 24% under a daily bear; bear 9% / bull 28% under a daily bull).

15 m transitions under each daily state (band case):

| daily state | stay bear | stay side | stay bull | side → bear | side → bull |
|---|---|---|---|---|---|
| bear | 0.942 | **0.891** | 0.930 | 0.052 | 0.057 |
| side | 0.927 | 0.956 | 0.926 | 0.021 | 0.024 |
| bull | 0.917 | 0.959 | 0.921 | 0.016 | 0.024 |
| pooled | 0.929 | 0.950 | 0.926 | 0.022 | 0.028 |

Under a daily bear state, 15 m sideways spells break about twice as often, to either side; extreme 15 m states last slightly longer. The differences are 1–6 pp on probabilities near 0.9–0.95 — real (beyond both references) but small.

## 3. Reading

- The hierarchy carries information: the daily state tells you what kind of intraday regime to expect.
- Most of that information is **volatility** (daily bear ↔ turbulent intraday; daily bull ↔ calm intraday), with a smaller directional tilt (fewer intraday bear windows in a daily bull; more bear than bull windows in a daily bear).
- The shuffled reference permutes returns across days, which also removes volatility clustering; part of the "beyond shuffled" dependence is therefore volatility clustering itself. Separating direction from volatility would need states built on volatility-standardized returns — a possible next notebook if Iván wants it.
