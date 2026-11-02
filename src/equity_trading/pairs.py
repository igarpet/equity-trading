"""Pair discovery: the thesis pipeline (García Petrov 2026, igarpet/Thesis src/cointegration.py)
applied to intraday bars, plus Johansen, rolling cointegration and economic-distance clustering.

Thesis-faithful parts keep the thesis logic and defaults:
  - 70/30 chronological split by observation, <=5% formation missing, first formation price valid,
    complete out-of-sample prices, formation gaps forward-filled
  - Pearson correlation of formation log returns; each stock's 10 highest *signed* correlations
  - ADF I(1) screen per ticker (levels not rejected at 5%, first differences rejected), AIC lags
  - fixed orientation (alphabetically first ticker = y), OLS of log prices with intercept
  - statsmodels coint(trend="c", autolag="aic"); selected if p <= 1%, I(1) screen passed, beta > 0
The only intraday adaptation is applied before this module: missing bars inside a session are
forward-filled up to a small limit (a bar with no trade is an unchanged price, not missing data).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

PAIRS_VERSION = "0.1.1"


# ---------------------------------------------------------------- availability (thesis 01)
def thesis_split(prices: pd.DataFrame, train_fraction: float = 0.70, max_missing: float = 0.05):
    """Thesis 01_Stock_Data: split rows 70/30, apply availability rules.

    Returns (train_prices, test_prices, availability table). train_prices are forward-filled.
    """
    cut = int(len(prices) * train_fraction)
    formation, oos = prices.iloc[:cut], prices.iloc[cut:]
    miss = formation.isna().mean()
    ok = (miss <= max_missing) & formation.iloc[0].notna()
    complete = oos.notna().all()
    retained = ok & complete
    avail = pd.DataFrame({"formation_missing_fraction": miss, "first_price_valid": formation.iloc[0].notna(),
                          "complete_out_of_sample": complete, "retained": retained})
    keep = retained[retained].index
    return formation[keep].ffill(), oos[keep].copy(), avail


# ---------------------------------------------------------------- candidate generators
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def top_correlation_pairs(corr: pd.DataFrame, top_n: int = 10) -> list[tuple[str, str]]:
    """Thesis generate_candidate_pairs: each stock's top_n highest signed correlations."""
    pairs = set()
    for s in corr.columns:
        for o in corr[s].drop(s).nlargest(top_n).index:
            pairs.add(tuple(sorted((s, o))))
    return sorted(pairs)


def gics_pairs(tickers: list[str], constituents: pd.DataFrame, level: str = "GICS Sub-Industry") -> list[tuple[str, str]]:
    """Economic distance, categorical: all pairs that share the same GICS group."""
    g = constituents.set_index("Symbol")[level].reindex(tickers).dropna()
    out = []
    for _, names in g.groupby(g):
        out += [tuple(sorted(p)) for p in itertools.combinations(sorted(names.index), 2)]
    return sorted(set(out))


def correlation_clusters(corr: pd.DataFrame, n_clusters: int, method: str = "average") -> pd.Series:
    """Economic distance, return-based: hierarchical clustering on the Mantegna distance
    d_ij = sqrt(2 (1 - rho_ij)). Returns cluster label per ticker."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = np.sqrt(np.clip(2 * (1 - corr.to_numpy()), 0, None))
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method=method)
    return pd.Series(fcluster(z, t=n_clusters, criterion="maxclust"), index=corr.columns, name="cluster")


def cluster_pairs(labels: pd.Series, max_cluster_size: int | None = None) -> list[tuple[str, str]]:
    """All within-cluster pairs; clusters larger than max_cluster_size are skipped (reported)."""
    out = []
    for _, names in labels.groupby(labels):
        if max_cluster_size and len(names) > max_cluster_size:
            continue
        out += [tuple(sorted(p)) for p in itertools.combinations(sorted(names.index), 2)]
    return sorted(set(out))


def random_null_pairs(tickers: list[str], exclude: set, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Random pairs outside every candidate set: the empirical null for the selection rule.

    Samples without replacement from the pairs that actually exist; if fewer than n are
    available, returns all of them and warns (never loops forever).
    """
    import warnings
    t = sorted(tickers)
    pool = [p for p in itertools.combinations(t, 2) if p not in exclude]
    if len(pool) < n:
        warnings.warn(f"only {len(pool)} non-candidate pairs exist among {len(t)} tickers; "
                      f"null uses all of them instead of {n}")
        return pool
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), n, replace=False)
    return sorted(pool[i] for i in idx)


# ---------------------------------------------------------------- tests (thesis + extensions)
def _quiet():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning if hasattr(np, "exceptions") else Warning)


def _adf_pair(series: np.ndarray, integration_alpha: float) -> dict:
    _quiet()
    from statsmodels.tsa.stattools import adfuller
    lvl = adfuller(series, regression="c", autolag="AIC")[1]
    dif = adfuller(np.diff(series), regression="c", autolag="AIC")[1]
    return {"level_p": float(lvl), "diff_p": float(dif),
            "i1": bool(lvl >= integration_alpha and dif < integration_alpha)}


def integration_screen(log_prices: pd.DataFrame, tickers: list[str], integration_alpha: float = 0.05,
                       n_jobs: int = -1) -> pd.DataFrame:
    """Thesis I(1) screen per ticker (level ADF not rejected, difference ADF rejected; AIC lags)."""
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs)(delayed(_adf_pair)(log_prices[t].to_numpy(), integration_alpha) for t in tickers)
    return pd.DataFrame(res, index=tickers)


def _eg_one(y: np.ndarray, x: np.ndarray) -> dict:
    _quiet()
    from statsmodels.tsa.stattools import coint
    X = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    stat, p, crit = coint(y, x, trend="c", autolag="aic")
    return {"alpha": float(alpha), "beta": float(beta), "eg_stat": float(stat), "eg_p": float(p)}


def _johansen_one(y: np.ndarray, x: np.ndarray, max_lags: int, det_order: int) -> dict:
    _quiet()
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order
    data = np.column_stack([y, x])
    try:
        k = int(select_order(data, maxlags=max_lags, deterministic="ci").aic)
        r = coint_johansen(data, det_order=det_order, k_ar_diff=max(k, 1))
    except Exception:
        return {"jo_k": np.nan, "jo_trace0": np.nan, "jo_cv0_99": np.nan, "jo_cv0_95": np.nan,
                "jo_trace1": np.nan, "jo_cv1_95": np.nan, "jo_beta": np.nan}
    v = np.real(r.evec[:, 0])
    return {"jo_k": max(k, 1), "jo_trace0": float(r.lr1[0]), "jo_cv0_99": float(r.cvt[0, 2]),
            "jo_cv0_95": float(r.cvt[0, 1]), "jo_trace1": float(r.lr1[1]), "jo_cv1_95": float(r.cvt[1, 1]),
            "jo_beta": float(-v[1] / v[0]) if v[0] != 0 else np.nan}


def _pair_tests(y, x, do_johansen, max_lags, det_order):
    out = _eg_one(y, x)
    if do_johansen:
        out.update(_johansen_one(y, x, max_lags, det_order))
    return out


def screen_pairs(log_prices: pd.DataFrame, pairs: list[tuple[str, str]], roots: pd.DataFrame,
                 significance: float = 0.01, johansen: bool = True, johansen_max_lags: int = 10,
                 johansen_det_order: int = 0, n_jobs: int = -1, batch_size: int = 16) -> pd.DataFrame:
    """Thesis screen_cointegration for every pair, plus (optionally) the Johansen trace test.

    eg_selected  : thesis rule (EG p <= significance, both legs I(1), beta > 0)
    jo_selected  : Johansen trace rejects r = 0 at 1% (trace > 99% critical value), r <= 1 not
                   rejected at 5%, both legs I(1), Johansen beta > 0 (same conventions as the thesis)
    """
    from joblib import Parallel, delayed
    pairs = sorted(set(tuple(sorted(p)) for p in pairs))
    arr = {c: log_prices[c].to_numpy() for c in log_prices.columns}
    res = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        delayed(_pair_tests)(arr[a], arr[b], johansen, johansen_max_lags, johansen_det_order) for a, b in pairs)
    df = pd.DataFrame(res)
    df.insert(0, "dependent", [a for a, _ in pairs])
    df.insert(1, "independent", [b for _, b in pairs])
    df.insert(0, "pair", df["dependent"] + "-" + df["independent"])
    df["i1_screen"] = roots.loc[df["dependent"], "i1"].to_numpy() & roots.loc[df["independent"], "i1"].to_numpy()
    df["eg_selected"] = (df["eg_p"] <= significance) & df["i1_screen"] & (df["beta"] > 0)
    if johansen:
        df["jo_reject0_1pct"] = df["jo_trace0"] > df["jo_cv0_99"]
        df["jo_selected"] = (df["jo_reject0_1pct"] & (df["jo_trace1"] <= df["jo_cv1_95"])
                             & df["i1_screen"] & (df["jo_beta"] > 0))
    return df


# ---------------------------------------------------------------- rolling (dynamic) cointegration
def rolling_stability(log_prices: pd.DataFrame, pairs: list[tuple[str, str]], window: int, step: int,
                      alpha: float = 0.05, lags: int = 1, n_jobs: int = -1) -> pd.DataFrame:
    """Rolling Engle-Granger inside the formation sample (windows and steps in bars).

    Uses the fast fixed-lag EG (equity_trading.coint.eg_test, MacKinnon p-value) because AIC lag
    search over tens of thousands of windows is impractical. Per pair: share of windows with
    p <= alpha, coefficient of variation of beta, median rolling half-life (bars).
    """
    from joblib import Parallel, delayed
    starts = list(range(0, len(log_prices) - window + 1, step))
    arr = {c: log_prices[c].to_numpy() for c in log_prices.columns}

    def one(a, b):
        from .coint import ar1_half_life, eg_test
        ps, bs, hl = [], [], []
        for s in starts:
            y, x = arr[a][s:s + window], arr[b][s:s + window]
            r = eg_test(y, x, lags)
            ps.append(r["p"]); bs.append(r["beta"])
            hl.append(ar1_half_life(y - r["alpha"] - r["beta"] * x))
        ps, bs = np.array(ps), np.array(bs)
        return {"roll_windows": len(starts), "roll_pass_share": float(np.mean(ps <= alpha)),
                "roll_beta_mean": float(np.mean(bs)),
                "roll_beta_cv": float(np.std(bs) / abs(np.mean(bs))) if np.mean(bs) != 0 else np.nan,
                "roll_half_life_med": float(np.nanmedian(hl))}

    pairs = sorted(set(tuple(sorted(p)) for p in pairs))
    res = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(one)(a, b) for a, b in pairs)
    df = pd.DataFrame(res)
    df.insert(0, "pair", [f"{a}-{b}" for a, b in pairs])
    return df


# ---------------------------------------------------------------- out-of-sample behaviour
def oos_diagnostics(train_log: pd.DataFrame, test_log: pd.DataFrame, sel: pd.DataFrame,
                    entry_z: float = 1.5, stop_z: float = 3.0) -> pd.DataFrame:
    """Frozen formation alpha/beta/mean/sd, applied to the out-of-sample bars (thesis: no refit).

    band_hold      share of OOS bars with |z| <= 2
    drift_sd       |mean(OOS spread) - formation mean| / formation sd
    episodes       number of OOS excursions to |z| >= entry_z (a new one starts after z returns to 0)
    reversion_rate share of those excursions that reach z = 0 before |z| >= stop_z or the sample ends
    adf_oos_p      ADF p-value on the OOS frozen spread (MacKinnon, N = 1)
    """
    from statsmodels.tsa.adfvalues import mackinnonp

    from .coint import adf_tstat
    rows = []
    for r in sel.itertuples(index=False):
        uf = train_log[r.dependent] - r.alpha - r.beta * train_log[r.independent]
        mu, sd = uf.mean(), uf.std(ddof=1)
        uo = (test_log[r.dependent] - r.alpha - r.beta * test_log[r.independent]).to_numpy()
        z = (uo - mu) / sd
        ep, rev, i = 0, 0, 0
        while i < len(z):
            if abs(z[i]) >= entry_z:
                ep += 1
                s = np.sign(z[i]); j = i
                while j < len(z) and s * z[j] > 0 and abs(z[j]) < stop_z:
                    j += 1
                if j < len(z) and s * z[j] <= 0:
                    rev += 1
                while j < len(z) and s * z[j] > 0:  # wait until back through 0 before a new episode
                    j += 1
                i = j
            i += 1
        t = adf_tstat(uo, 1)
        rows.append({"pair": r.pair, "band_hold": float(np.mean(np.abs(z) <= 2)),
                     "drift_sd": float(abs(z.mean())), "episodes": ep,
                     "reversion_rate": rev / ep if ep else np.nan,
                     "adf_oos_p": float(mackinnonp(t, regression="c", N=1)) if np.isfinite(t) else np.nan})
    return pd.DataFrame(rows)


def mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
