# Deep Statistical Arbitrage — paper P6

**Paper.** Guijarro-Ordonez, J., Pelger, M. & Zanotti, G. Deep Learning Statistical Arbitrage. arXiv 2106.04028.

**Claim to test.** Residuals from factor models (incl. PCA/IPCA), with a neural network extracting time-series signals and learning the allocation.

**Our question.** Do richer factor residuals and a learned policy produce a gross edge larger than the sector-ETF residual, after measured costs?

| Notebook | Description | Status |
|---|---|---|
| `01_data_and_training.ipynb` | Yahoo 2005–2015 + Alpaca 2016–2025 daily returns; PCA-5 residuals (252-session window, daily); 30-session cumulative-residual inputs; rolling CNN+Transformer fits (750 train / 250 validation / 125 predict, 5 seeds), net-Sharpe objective with factor neutralisation and measured costs; all fits cached | Built; awaiting run |
| `02_evaluation.ipynb` | Out-of-sample 2017–2019 and 2020–2025: network vs OU and linear benchmarks; gates G1 bootstrap, G2 gross vs 1,000 weight permutations, G3 2× cost | Built; awaiting run (after 01) |

Protocol: [`001_claude_p6_deep_stat_arb_preregistration.md`](../../discussions/Deep_Statistical_Arbitrage/001_claude_p6_deep_stat_arb_preregistration.md).

**PyTorch for a GTX 1070 (Pascal, sm_61).** Only the CUDA 12.6 wheels still include Pascal, and PyTorch 2.14 is the last release with them:

```
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run: `python -m pytest -q tests/test_p6.py`, then 01 (Yahoo download minutes; training ~2–3 h on the GPU, resumable) and 02 (minutes). `P6_SYNTHETIC=1` runs both offline. Notebooks are numbered from `01`; outputs go to `NN/`. Data: Alpaca SIP 2020–2025; 2026 sealed. Costs: S05b measured cost model unless the protocol states otherwise.
