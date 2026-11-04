"""P2: Nagel (2012), Evaporating Liquidity - is reversal larger when liquidity is scarce?

Protocol: discussions/Liquidity_Provision/001_claude_p2_liquidity_provision_preregistration.md

H1: lagged VIX predicts the return of a short-term reversal (liquidity-provision) strategy.
    Daily Lehmann / Lo-MacKinlay contrarian weights w ~ -(r_i - mean r), +1 long / -1 short.
    Regime: VIX at the close of d above its trailing 250-session median (>= 200), known at that close.
H2: (from P1) high abnormal-volume stocks reverse more; tested on 2017-2019 with the P1 arm-D machinery.
The 15-minute arm gates S05 entries by the regime; placebos shift the regime series circularly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

P2_VERSION = "0.1.0"


@dataclass(frozen=True)
class P2Config:
    vix_window: int = 250
    vix_min: int = 200
    skip: int = 1
    min_price: float = 5.0
    hac_lags: int = 5
    alpha: float = 0.025              # per hypothesis family (H1, H2) in the daily notebooks
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    h1_start: str = "2016-12-01"      # VIX baseline needs ~250 sessions of 2016
    econ_start: str = "2020-01-01"    # measured costs exist from 2020
    end: str = "2025-12-31"
    h2_start: str = "2017-01-01"      # independent sample for the P1 observation
    h2_end: str = "2019-12-31"
    h2_k: int = 5
    n_label_placebo: int = 500
    n_shift_placebo: int = 200
    min_shift: int = 60
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== VIX
def load_vix(start: str = "2015-01-01", end: str = "2026-01-01", refresh: bool = False) -> pd.Series:
    """CBOE VIX daily close from Yahoo (^VIX), index = session date (naive). Cached under <data>/Yahoo/."""
    from .data import vendor_dir
    path = vendor_dir("Yahoo") / f"VIX_1d_{start}_{end}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["VIX"]
    import yfinance as yf
    d = yf.download("^VIX", start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if d.empty:
        raise RuntimeError("Yahoo returned no ^VIX data; check network access.")
    close = d["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = pd.Series(close.to_numpy(float), index=pd.DatetimeIndex(close.index).tz_localize(None).normalize(), name="VIX")
    s = s[s.index < pd.Timestamp(end)]
    s.to_frame().to_parquet(path)
    return s


def vix_regime(vix: pd.Series, cfg: P2Config) -> pd.Series:
    """True where VIX at that close exceeds its trailing median (window includes the day; no look-ahead)."""
    med = vix.rolling(cfg.vix_window, min_periods=cfg.vix_min).median()
    return (vix > med).where(med.notna())


def synthetic_vix(index: pd.DatetimeIndex, seed: int = 11) -> pd.Series:
    """Persistent log-AR(1) volatility index around 18."""
    rng = np.random.default_rng(seed)
    x = np.zeros(len(index))
    for t in range(1, len(index)):
        x[t] = 0.97 * x[t - 1] + rng.standard_normal() * 0.08
    return pd.Series(18 * np.exp(x), index=index, name="VIX")


def synthetic_daily_vix(n_stocks: int = 150, years: list[int] | None = None, seed: int = 5, base: float = 0.0,
                        slope: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Offline daily panel whose reversal depends on lagged VIX: the idiosyncratic return at t adds
    -(base + slope x 1{VIX high at t-2}) x (the market-adjusted return at t-2), tradable at skip = 1."""
    years = years or [2016, 2017, 2018]
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    vix = synthetic_vix(days, seed)
    hi = vix_regime(vix, P2Config()).fillna(False).to_numpy().astype(float)
    D, N = len(days), n_stocks
    eps = rng.standard_normal((D, N)) * 0.02
    mkt = rng.standard_normal(D) * 0.01
    r = np.zeros((D, N))
    for t in range(D):
        adj = 0.0
        if t >= 2:
            prev = r[t - 2] - r[t - 2].mean()
            adj = -(base + slope * hi[t - 2]) * prev
        r[t] = mkt[t] + eps[t] + adj
    c = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), index=days, columns=[f"S{j:03d}" for j in range(N)])
    v = pd.DataFrame(rng.uniform(1e5, 1e7, (D, N)), index=days, columns=c.columns)
    return c, v, vix


# =================================================================== daily H1
def contrarian_weights(ret: pd.DataFrame, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lehmann / Lo-MacKinlay: w ~ -(r_i - mean r) over eligible names, scaled to +1 long / -1 short."""
    r = ret.where(eligible) if eligible is not None else ret
    dev = r.sub(r.mean(axis=1), axis=0)
    w = -dev
    scale = w.abs().sum(axis=1) / 2
    return w.div(scale.replace(0, np.nan), axis=0).fillna(0.0)


def gate_by_regime(target: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """Keep formation rows whose regime (at the formation close) is high; zero the rest."""
    on = regime.reindex(target.index).fillna(False).astype(bool)
    return target.mul(on.astype(float), axis=0)


def aligned_signal(x: pd.Series, index: pd.Index, skip: int) -> pd.Series:
    """Value at the formation close of the position earning return day t (formed at t - skip - 1)."""
    return x.reindex(index).shift(skip + 1)


def hac_slope(y: pd.Series, x: pd.Series, lags: int) -> dict:
    """OLS y = a + b x with HAC (Newey-West) errors; one-sided p for b > 0."""
    import statsmodels.api as sm
    from scipy.stats import norm
    d = pd.concat([y, x.astype(float)], axis=1, keys=["y", "x"]).dropna()
    fit = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(fit.tvalues["x"])
    return {"intercept": float(fit.params["const"]), "slope": float(fit.params["x"]), "t": t,
            "p_one_sided": float(norm.sf(t)), "n": int(len(d))}


def gates_h1(n1: dict, n2: dict, boot_net: dict, net2x_mean: float, alpha: float) -> dict:
    g1 = n1["slope"] > 0 and n1["p_one_sided"] <= alpha
    g2 = n2["slope"] > 0 and n2["p_one_sided"] <= alpha
    g3 = all(v[1] > 0 for v in boot_net.values()) and net2x_mean > 0
    rep = "REPLICATES" if (g1 and g2) else ("PARTIAL" if (g1 or g2) else "NO_REPLICATION")
    return {"N1_vix_slope": g1, "N2_vix_regime": g2, "N3_economic": g3, "replication_verdict": rep,
            "economic_verdict": "PROFITABLE_AFTER_COSTS" if g3 else "NOT_PROFITABLE_AFTER_COSTS"}


def gates_h2(boot_diff: dict, p_upper: float, alpha: float) -> dict:
    a1 = all(v[1] > 0 for v in boot_diff.values())
    a2 = p_upper <= alpha
    return {"A1_bootstrap": a1, "A2_label_placebo": a2, "verdict": "CONFIRMED" if (a1 and a2) else "NOT_CONFIRMED"}


# =================================================================== 15-minute arm
def regime_mask(regime_on_days: np.ndarray, n_stocks: int) -> np.ndarray:
    """(D, N) entry mask: row e is all True when the regime at the close of e is high."""
    return np.repeat(np.asarray(regime_on_days, bool)[:, None], n_stocks, axis=1)


def shifted_regime(regime_on_days: np.ndarray, rng: np.random.Generator, min_shift: int,
                   valid: np.ndarray | None = None) -> np.ndarray:
    """Circular shift of the regime series over its valid span by a random offset in [min_shift, n - min_shift]."""
    a = np.asarray(regime_on_days, bool).copy()
    idx = np.flatnonzero(valid) if valid is not None else np.arange(len(a))
    n = len(idx)
    k = int(rng.integers(min_shift, max(min_shift + 1, n - min_shift)))
    a[idx] = np.roll(a[idx], k)
    return a


def run_regime_backtest(panel, params, stock_cols, etf_cols, cfg, regime_on_days, seed: int, start, half_spread) -> dict:
    from .p1 import run_mask_backtest
    return run_mask_backtest(panel, params, stock_cols, etf_cols, cfg, regime_mask(regime_on_days, len(stock_cols)),
                             seed, start, half_spread)


def gates_15m(diff_ci: tuple, ranks: dict, net1: dict, net2: dict, boot_net: dict, alpha: float = 0.05) -> dict:
    v1 = diff_ci[1] > 0
    v2 = ranks["total_return"]["p_upper"] <= alpha and ranks["sharpe"]["p_upper"] <= alpha
    v3 = net1["total_return"] > 0 and net2["total_return"] > 0 and all(v[1] > 0 for v in boot_net.values())
    return {"V1_regime_dependence": v1, "V2_vs_shifted_regime": v2, "V3_economic": v3,
            "verdict": "EDGE_CANDIDATE" if (v1 and v2 and v3) else "NO_EDGE",
            "prospective_verdict": "PENDING_2026_HOLDOUT"}


# =================================================================== panels outside the shared cache
def panel_cache(symbols: list[str], timeframe: str, years: list[int], path, ffill_bars: int = 2):
    """Panel for other years than the shared 2020-2025 cache, stored at `path`.
    (s04.cached_panel keeps one file per timeframe and would overwrite the shared cache.)"""
    import os
    from pathlib import Path
    from .s04 import NY, Panel, load_panel
    path = Path(path)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["years"]) == list(years) and set(symbols) <= set(z["symbols"]) | set(z["absent"]):
            return Panel(index=pd.DatetimeIndex(z["index"]).tz_localize("UTC").tz_convert(NY), symbols=list(z["symbols"]),
                         o=z["o"], c=z["c"], day_start=z["day_start"], days=z["days"], s2_daily=z["s2"], h=z["h"], l=z["l"])
    panel = load_panel(sorted(symbols), timeframe, years, ffill_bars=ffill_bars)
    absent = sorted(set(symbols) - set(panel.symbols))
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp, index=panel.index.tz_convert("UTC").tz_localize(None).to_numpy(), symbols=np.array(panel.symbols),
             absent=np.array(absent, dtype=object), years=np.array(years), o=panel.o, c=panel.c, day_start=panel.day_start,
             days=panel.days, s2=panel.s2_daily, h=panel.h, l=panel.l)
    os.replace(tmp, path)
    return panel
