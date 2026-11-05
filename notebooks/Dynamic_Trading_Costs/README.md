# Dynamic Trading Costs — paper P5

**Paper.** Gârleanu, N. & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *Journal of Finance* 68(6), 2309–2340.

**Claim to test.** With quadratic costs, trade partially toward an aim portfolio; signals are weighted by their persistence (slower-decaying signals get more weight).

**Our question.** Successor to S07: does the optimal partial-adjustment rule, with our measured S05b costs, turn the gross residual signal into a net positive?

| Notebook | Description | Status |
|---|---|---|
| `01_garleanu_pedersen.ipynb` | GP rule on two 15m residual signals (fast S05 s-score, slow 5-session residual), ex-ante B and φ, persistence-weighted aim, partial adjustment τ = 0.25 / 0.10 / 0.04 per bar, earnings windows excluded; both windows; 100 shuffled-forecast placebos per arm; ablations noPW / Aonly / Bonly | Built; awaiting run |

Protocol: [`001_claude_p5_garleanu_pedersen_preregistration.md`](../../discussions/Dynamic_Trading_Costs/001_claude_p5_garleanu_pedersen_preregistration.md). Both windows judged; earnings-window exclusion (P4) built in.

Run: `python -m pytest -q tests/test_p5.py`, then notebook 01 (~3–4 h, no downloads). `P5_SYNTHETIC=1` runs it offline. Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
