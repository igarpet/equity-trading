"""Fast Engle-Granger tests and the S02 rolling cointegration base-rate engine."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp

COINT_VERSION = "0.1.1"


def adf_tstat(u: np.ndarray, lags: int = 1) -> float:
    """ADF t-statistic with constant: du_t = a + g u_{t-1} + sum_{j=1..lags} c_j du_{t-j} + e."""
    u = np.asarray(u, dtype=float)
    du = np.diff(u)
    n = du.size - lags
    if n < 20:
        return np.nan
    cols = [np.ones(n), u[lags:-1]] + [du[lags - j:-j] for j in range(1, lags + 1)]
    X = np.column_stack(cols)
    y = du[lags:]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (n - X.shape[1])
    se = np.sqrt(s2 * XtX_inv[1, 1])
    return float(b[1] / se) if se > 0 else np.nan


def ols_ab(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    vx = np.sum((x - xm) ** 2)
    beta = np.sum((x - xm) * (y - ym)) / vx if vx > 0 else np.nan
    return float(ym - beta * xm), float(beta)


def eg_test(y: np.ndarray, x: np.ndarray, lags: int = 1) -> dict:
    """Engle-Granger two-step with fixed ADF lag; p-value from MacKinnon (N=2, constant)."""
    a, b = ols_ab(y, x)
    u = y - a - b * x
    t = adf_tstat(u, lags)
    p = float(mackinnonp(t, regression="c", N=2)) if np.isfinite(t) else np.nan
    return {"alpha": a, "beta": b, "t": t, "p": p, "mu": float(u.mean()), "sd": float(u.std(ddof=1))}


def ar1_half_life(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    a, b = u[:-1] - u[:-1].mean(), u[1:] - u[1:].mean()
    phi = np.sum(a * b) / np.sum(a * a)
    return float(np.log(2) / -np.log(phi)) if 0 < phi < 1 else np.nan


def build_pairs(universe: dict[str, list[str]], etfs: dict[str, str]) -> pd.DataFrame:
    """Pair list with groups.

    sector    : both stocks in the same sector (alphabetical orientation, y = first)
    cross     : stocks in different sectors (empirical null for 'sector')
    etf       : stock (y) vs its own sector ETF (x)
    etf_cross : stock (y) vs another sector's ETF (x) (empirical null for 'etf')
    """
    rows = []
    sect_of = {s: k for k, v in universe.items() for s in v}
    stocks = sorted(sect_of)
    for a, b in itertools.combinations(stocks, 2):
        rows.append({"y": a, "x": b, "group": "sector" if sect_of[a] == sect_of[b] else "cross",
                     "sector_y": sect_of[a], "sector_x": sect_of[b]})
    for s in stocks:
        for k, e in etfs.items():
            rows.append({"y": s, "x": e, "group": "etf" if sect_of[s] == k else "etf_cross",
                         "sector_y": sect_of[s], "sector_x": k})
    return pd.DataFrame(rows)


def session_dates(index: pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(index).tz_convert("America/New_York").normalize()
    return np.unique(d)


def rolling_windows(dates: np.ndarray, form_days: int, step_days: int, oos_days: int):
    """Yield (form_start, form_end, oos_end) as session-date indices (half-open)."""
    i = 0
    while i + form_days + oos_days <= len(dates):
        yield i, i + form_days, i + form_days + oos_days
        i += step_days


def run_rolling(logp: pd.DataFrame, pairs: pd.DataFrame, form_days: int, step_days: int = 20,
                oos_days: int = 20, lags: int = 1, min_obs_share: float = 0.9) -> pd.DataFrame:
    """For each formation window and pair: EG test in formation; then, with the formation
    alpha/beta frozen, on the next oos_days of spread: ADF (N=1 critical values, since nothing
    is re-estimated), band_hold (share of bars within formation mean +/- 2 formation sd),
    standardised drift of the OOS mean, and crossings of the formation mean per day.
    """
    day = pd.DatetimeIndex(logp.index).tz_convert("America/New_York").normalize()
    dates = np.unique(day)
    day_pos = np.searchsorted(dates, day)
    arr = {c: logp[c].to_numpy() for c in logp.columns}
    bars_per_day = len(logp) / max(len(dates), 1)
    rows = []
    for w, (i0, i1, i2) in enumerate(rolling_windows(dates, form_days, step_days, oos_days)):
        f_mask = (day_pos >= i0) & (day_pos < i1)
        o_mask = (day_pos >= i1) & (day_pos < i2)
        nf_full, no_full = f_mask.sum(), o_mask.sum()
        for r in pairs.itertuples(index=False):
            yf, xf = arr[r.y][f_mask], arr[r.x][f_mask]
            ok = np.isfinite(yf) & np.isfinite(xf)
            if ok.sum() < min_obs_share * nf_full:
                continue
            res = eg_test(yf[ok], xf[ok], lags)
            uf = yf[ok] - res["alpha"] - res["beta"] * xf[ok]
            yo, xo = arr[r.y][o_mask], arr[r.x][o_mask]
            oko = np.isfinite(yo) & np.isfinite(xo)
            row = {"window": w, "form_start": dates[i0], "oos_start": dates[i1], "y": r.y, "x": r.x,
                   "group": r.group, "p": res["p"], "beta": res["beta"],
                   "half_life_bars": ar1_half_life(uf)}
            if oko.sum() >= min_obs_share * no_full and res["sd"] > 0:
                uo = yo[oko] - res["alpha"] - res["beta"] * xo[oko]
                t_o = adf_tstat(uo, lags)
                row["p_oos"] = float(mackinnonp(t_o, regression="c", N=1)) if np.isfinite(t_o) else np.nan
                row["drift_sd"] = float(abs(uo.mean() - res["mu"]) / res["sd"])
                row["band_hold"] = float(np.mean(np.abs(uo - res["mu"]) <= 2 * res["sd"]))
                s = np.sign(uo - res["mu"])
                row["crossings_per_day"] = float(np.sum(s[1:] != s[:-1]) / (oko.sum() / bars_per_day))
            rows.append(row)
    return pd.DataFrame(rows)


def rate_diff_ci(windows: np.ndarray, a_hit: np.ndarray, a_in: np.ndarray, b_hit: np.ndarray,
                 b_in: np.ndarray, n_boot: int = 2000, block: int = 3, seed: int = 0) -> dict:
    """Rates r_a = sum(a_hit)/sum(a_in), r_b likewise (a_hit may be boolean or a value in [0, 1],
    giving a mean), their difference and ratio, with 95% CIs
    from a moving-block bootstrap over formation windows (fast: works on per-window counts).

    a_in/b_in select the rows in each group; a_hit/b_hit mark the 'successes' among them.
    """
    w = pd.Series(windows)
    agg = pd.DataFrame({"ah": np.where(a_in, a_hit, 0.0), "ai": a_in,
                        "bh": np.where(b_in, b_hit, 0.0), "bi": b_in}).astype(float)
    agg = agg.groupby(w.values).sum().sort_index().to_numpy()
    nw = len(agg)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nw / block))
    starts = rng.integers(0, max(nw - block + 1, 1), (n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :nw]
    s = agg[idx].sum(axis=1)  # (n_boot, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        ra, rb = s[:, 0] / s[:, 1], s[:, 2] / s[:, 3]
        tot = agg.sum(axis=0)
        est_a, est_b = tot[0] / tot[1], tot[2] / tot[3]
        out = {"rate_a": est_a, "rate_b": est_b, "diff": est_a - est_b,
               "ratio": est_a / est_b if est_b > 0 else np.inf,
               "n_a": int(tot[1]), "n_b": int(tot[3]), "windows": nw}
        for k, v in (("rate_a", ra), ("rate_b", rb), ("diff", ra - rb), ("ratio", ra / rb)):
            lo, hi = np.nanpercentile(v[np.isfinite(v)], [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
            out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def window_bootstrap(df: pd.DataFrame, stat, n_boot: int = 1000, block: int = 3,
                     seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% CI of stat(df), resampling formation windows in moving blocks
    (keeps the cross-sectional dependence of pairs observed in the same window)."""
    rng = np.random.default_rng(seed)
    wins = np.sort(df["window"].unique())
    groups = {w: g for w, g in df.groupby("window")}
    nb = int(np.ceil(len(wins) / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(len(wins) - block + 1, 1), nb)
        pick = np.concatenate([wins[s:s + block] for s in starts])[:len(wins)]
        vals.append(stat(pd.concat([groups[w] for w in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(stat(df)), float(lo), float(hi)
