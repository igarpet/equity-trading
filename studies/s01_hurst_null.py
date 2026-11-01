"""Study 01: is intraday anti-persistence (H < 0.5) real, or estimator bias?

Pre-registration: see studies/S01_README.md. Gates are fixed there and must not be
changed after results are seen.

Run:  python studies/s01_hurst_null.py            (Yahoo, needs internet)
      S01_SYNTHETIC=1 python studies/s01_hurst_null.py   (offline smoke test)
Outputs: results/s01/<interval>/pairs.csv, summary.json, manifest.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fou_eq import __version__  # noqa: E402
from fou_eq.data import load_yahoo_closes, synthetic_closes  # noqa: E402
from fou_eq.hurst import (HURST_VERSION, ar1_fit, arma11_fit, fou_moments, hurst_cof,  # noqa: E402
                          noise_sd, simulate_arma11, simulate_ou_noise)
from fou_eq.spreads import SPREADS_VERSION, deseasonalise, ols_spread  # noqa: E402
from fou_eq.universe import UNIVERSE_01  # noqa: E402

# ---- fixed design (pre-registered) ----
INTERVALS = ["30m", "15m"]          # 30m primary, 15m secondary
N_NULL = 200                        # bootstrap paths per pair
SEED = 20261101
P_LEVEL = 0.05                      # one-sided, per pair
GATE_SHARE = 0.20                   # share of pairs rejecting (4x the 5% null rate)
GATE_MEDIAN_DH = -0.03              # median H_obs - mean H_null
MIN_HALF_LIFE_BARS = 4.0            # scaling-window validity flag
MIN_OBS = 200


def eg_pvalue(ly: pd.Series, lx: pd.Series) -> float:
    try:
        from statsmodels.tsa.stattools import coint
        return float(coint(ly, lx, trend="c", autolag="AIC")[1])
    except Exception:
        return np.nan


def analyse_pair(sector, ty, tx, logp, rng):
    ly, lx = logp[ty].dropna(), logp[tx].dropna()
    common = ly.index.intersection(lx.index)
    ly, lx = ly.loc[common], lx.loc[common]
    if len(common) < MIN_OBS:
        return None
    x, alpha, beta = ols_spread(ly, lx)
    z, _ = deseasonalise(x)
    zv = z.to_numpy()
    h_raw, h_des = hurst_cof(x.to_numpy()), hurst_cof(zv)
    ar = ar1_fit(zv)
    eta = noise_sd(zv)
    row = {"sector": sector, "y": ty, "x": tx, "n": len(zv), "beta": beta,
           "eg_p": eg_pvalue(ly, lx), "h_raw": h_raw, "h_des": h_des,
           "phi": ar["phi"], "half_life_bars": ar["half_life"], "noise_sd": eta,
           "noise_share": eta ** 2 / np.var(np.diff(zv)) if np.var(np.diff(zv)) > 0 else np.nan}
    if not (0 < ar["phi"] < 1):
        row.update(stationary=False)
        return row
    fm = fou_moments(zv, h_des)
    row.update(stationary=True, fou_kappa=fm["kappa"], fou_half_life=fm["half_life"])
    arma = arma11_fit(zv)
    row["arma_phi"] = arma["phi"] if arma else np.nan
    row["arma_theta"] = arma["theta"] if arma else np.nan
    for tag in ("noise", "clean"):
        if tag == "noise":
            if arma is None:
                row.update(stationary=False)
                return row
            # OU + measurement noise null: fitted ARMA(1,1)
            sims = simulate_arma11(len(zv), arma["phi"], arma["theta"], arma["sd"], N_NULL, rng)
        else:
            # clean OU null: fitted AR(1), no noise
            sims = simulate_ou_noise(len(zv), ar["phi"], ar["sd"], 0.0, N_NULL, rng)
        hn = np.array([hurst_cof(s) for s in sims])
        row[f"hnull_{tag}_mean"] = float(np.nanmean(hn))
        row[f"hnull_{tag}_sd"] = float(np.nanstd(hn))
        row[f"p_low_{tag}"] = float((1 + np.sum(hn <= h_des)) / (N_NULL + 1))
        row[f"p_high_{tag}"] = float((1 + np.sum(hn >= h_des)) / (N_NULL + 1))
    row["dh_noise"] = h_des - row["hnull_noise_mean"]
    row["valid_window"] = bool(ar["half_life"] >= MIN_HALF_LIFE_BARS)
    return row


def verdict(df: pd.DataFrame) -> dict:
    d = df[df["stationary"] == True]  # noqa: E712
    if d.empty:
        return {"verdict": "NO_DATA"}
    share_low = float((d["p_low_noise"] <= P_LEVEL).mean())
    share_high = float((d["p_high_noise"] <= P_LEVEL).mean())
    med_dh = float(d["dh_noise"].median())
    if share_low >= GATE_SHARE and med_dh <= GATE_MEDIAN_DH:
        v = "H_BELOW_NULL"
    elif share_high >= GATE_SHARE and med_dh >= -GATE_MEDIAN_DH:
        v = "H_ABOVE_NULL"
    else:
        v = "NO_EVIDENCE"
    out = {"verdict": v, "pairs": int(len(d)), "share_reject_low": share_low,
           "share_reject_high": share_high, "median_dh": med_dh,
           "median_h_raw": float(d["h_raw"].median()), "median_h_des": float(d["h_des"].median()),
           "median_hnull_noise": float(d["hnull_noise_mean"].median()),
           "median_hnull_clean": float(d["hnull_clean_mean"].median()),
           "share_reject_low_clean_null": float((d["p_low_clean"] <= P_LEVEL).mean()),
           "share_valid_window": float(d["valid_window"].mean()),
           "median_half_life_bars": float(d["half_life_bars"].median())}
    sub = d[d["valid_window"]]
    out["valid_window_only"] = {"pairs": int(len(sub)),
                                "share_reject_low": float((sub["p_low_noise"] <= P_LEVEL).mean()) if len(sub) else None,
                                "median_dh": float(sub["dh_noise"].median()) if len(sub) else None}
    eg = d[d["eg_p"] <= 0.05]
    out["eg_p05_only"] = {"pairs": int(len(eg)),
                          "share_reject_low": float((eg["p_low_noise"] <= P_LEVEL).mean()) if len(eg) else None,
                          "median_dh": float(eg["dh_noise"].median()) if len(eg) else None}
    out["by_sector"] = {s: {"pairs": int(len(g)), "share_reject_low": float((g["p_low_noise"] <= P_LEVEL).mean()),
                            "median_dh": float(g["dh_noise"].median())} for s, g in d.groupby("sector")}
    return out


def main():
    synthetic = os.environ.get("S01_SYNTHETIC") == "1"
    tickers = [t for v in UNIVERSE_01.values() for t in v]
    out_root = ROOT / ("results/s01_synthetic" if synthetic else "results/s01")
    for interval in INTERVALS:
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        if synthetic:
            closes = synthetic_closes(tickers, n_days=40, bars_per_day=13 if interval == "30m" else 26, seed=SEED)
        else:
            closes = load_yahoo_closes(tickers, interval=interval, period="60d")
        logp = np.log(closes)
        rows = []
        for sector, names in UNIVERSE_01.items():
            for ty, tx in itertools.combinations(sorted(names), 2):
                r = analyse_pair(sector, ty, tx, logp, rng)
                if r is not None:
                    rows.append(r)
        df = pd.DataFrame(rows)
        out = out_root / interval
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "pairs.csv", index=False, float_format="%.6g")
        summ = verdict(df)
        summ["interval"] = interval
        summ["primary"] = interval == "30m"
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        manifest = {"study": "S01", "synthetic": synthetic, "interval": interval,
                    "start": str(closes.index.min()), "end": str(closes.index.max()),
                    "bars": int(len(closes)), "tickers": tickers, "n_null": N_NULL, "seed": SEED,
                    "versions": {"package": __version__, "hurst": HURST_VERSION, "spreads": SPREADS_VERSION},
                    "runtime_s": round(time.time() - t0, 1)}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{interval}] {summ['verdict']}: pairs={summ.get('pairs')} share_low={summ.get('share_reject_low')} "
              f"median_dH={summ.get('median_dh')}  ({manifest['runtime_s']} s)")


if __name__ == "__main__":
    main()
