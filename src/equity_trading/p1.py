"""P1: Medhat & Schmeling (2022), short-term momentum - does reversal depend on turnover?

Protocol: discussions/Short_Term_Momentum/001_claude_p1_turnover_reversal_preregistration.md

Turnover proxy (no shares outstanding in Alpaca): abnormal volume
    AV[d] = mean volume over sessions d-19..d / mean volume over d-249..d (>= 200 sessions),
known at the close of d. Each day, eligible stocks are split into AV terciles (0 = low, 2 = high).

Arm D (daily): conditional sort - within each tercile, quintiles of the past k-session log return; reversal
portfolio = equal-weight long the bottom quintile, short the top (+1 / -1). Overlapping holding of h = k
sessions. Signal at the close of d, trade at the close of d + skip, P&L from then on. Cost: one-way
half-spread x |change in weight| on the execution day.

Arm I (15 minutes): the S05 engine with an entry mask built from the same terciles (labels_on_days, plus
trade attribution and placebo helpers here).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

P1_VERSION = "0.1.0"
NY = "America/New_York"


@dataclass(frozen=True)
class P1Config:
    av_short: int = 20
    av_long: int = 250
    av_min_long: int = 200
    n_groups: int = 3
    n_quantiles: int = 5
    horizons: tuple = (21, 5, 1)      # first = primary (the paper's one-month horizon)
    skip: int = 1                     # sessions from the signal close to the execution close
    min_price: float = 5.0
    trade_start: str = "2020-01-01"   # 2019 is volume warm-up only
    trade_end: str = "2025-12-31"
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    n_label_placebo: int = 500
    alpha: float = 0.05 / 3
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== data
def load_daily(symbols: list[str], years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Close, open and volume (index: NYSE session date, columns: symbol) from the Alpaca 1Day cache."""
    from .alpaca import load_bars, session_schedule
    sessions = pd.DatetimeIndex(pd.to_datetime(list(session_schedule(years).index)))
    cols = {"c": {}, "o": {}, "v": {}}
    for s in symbols:
        try:
            b = load_bars(s, "1Day", years)
        except FileNotFoundError:
            continue
        if b.empty:
            continue
        d = pd.DatetimeIndex(pd.to_datetime(b["t"], utc=True)).tz_convert(NY).normalize().tz_localize(None)
        for k in cols:
            cols[k][s] = pd.Series(b[k].to_numpy(float), index=d)
    out = [pd.DataFrame(cols[k]).reindex(sessions) for k in ("c", "o", "v")]
    return tuple(out)


def synthetic_daily(n_stocks: int = 150, years: list[int] | None = None, seed: int = 7,
                    effect: float = 0.15, lag: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Offline daily panel with a planted effect: a stock's return adds -effect x (its return `lag` sessions
    earlier) if its abnormal volume was low, +effect x if high, 0 in between. lag = 2 matches the one-session
    execution skip, so the effect is tradable at skip = 1."""
    years = years or [2019, 2020, 2021]
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    D, N = len(days), n_stocks
    lvl = rng.uniform(1e5, 1e7, N)
    shock = np.exp(np.cumsum(rng.standard_normal((D, N)) * 0.15, 0) * 0.2)     # slow volume regimes
    v = lvl * shock * np.exp(rng.standard_normal((D, N)) * 0.3)
    vv = pd.DataFrame(v)
    av = (vv.rolling(20).mean() / vv.rolling(250, min_periods=200).mean()).to_numpy()
    rk = pd.DataFrame(av).rank(axis=1, pct=True).to_numpy()
    coef = np.where(rk < 1 / 3, -effect, np.where(rk > 2 / 3, effect, 0.0))
    r = np.zeros((D, N))
    eps = rng.standard_normal((D, N)) * 0.02
    for t in range(lag, D):
        c = np.where(np.isfinite(coef[t - lag]), coef[t - lag], 0.0)
        r[t] = c * r[t - lag] + eps[t]
    c = 50 * np.exp(np.cumsum(r, 0))
    o = c * np.exp(rng.standard_normal((D, N)) * 0.003)
    cols = [f"S{j:03d}" for j in range(N)]
    mk = lambda a: pd.DataFrame(a, index=days, columns=cols)  # noqa: E731
    return mk(c), mk(o), mk(v)


# =================================================================== turnover and labels
def abnormal_volume(v: pd.DataFrame, cfg: P1Config) -> pd.DataFrame:
    vv = v.where(v > 0)
    num = vv.rolling(cfg.av_short, min_periods=cfg.av_short).mean()
    den = vv.rolling(cfg.av_long, min_periods=cfg.av_min_long).mean()
    return num / den


def group_labels(x: pd.DataFrame, n: int, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-row n-tile labels 0..n-1 of x among eligible finite entries; -1 elsewhere."""
    xe = x.where(eligible) if eligible is not None else x
    pct = xe.rank(axis=1, pct=True, method="first")
    lab = np.ceil(pct.to_numpy() * n) - 1
    lab = np.where(np.isfinite(lab), lab, -1).astype(np.int8)
    return pd.DataFrame(lab, index=x.index, columns=x.columns)


# =================================================================== arm D
def reversal_weights(signal: pd.DataFrame, labels: pd.DataFrame, group: int, nq: int) -> pd.DataFrame:
    """Daily formation weights of the reversal portfolio inside one group: +1 spread equally over the
    bottom signal quantile, -1 over the top. Rows with fewer than 2 x nq names get zero."""
    s = signal.where(labels == group)
    cnt = s.notna().sum(axis=1)
    pct = s.rank(axis=1, pct=True, method="first")
    q = np.ceil(pct * nq)
    lo, hi = (q == 1), (q == nq)
    nlo, nhi = lo.sum(axis=1), hi.sum(axis=1)
    w = lo.div(nlo.replace(0, np.nan), axis=0).fillna(0.0) - hi.div(nhi.replace(0, np.nan), axis=0).fillna(0.0)
    w.loc[cnt < 2 * nq, :] = 0.0
    return w


def overlap(w: pd.DataFrame, h: int) -> pd.DataFrame:
    """Overlapping portfolios: target = mean of the last h daily formations."""
    return w.rolling(h, min_periods=1).sum() / h


def ls_returns(target: pd.DataFrame, ret: pd.DataFrame, cost: pd.DataFrame | None, skip: int,
               cost_mult: float = 1.0) -> pd.DataFrame:
    """Daily gross, cost and net return of a weight path. target row d is formed at the close of d and traded
    at the close of d + skip; it earns the returns of d + skip + 1 onward."""
    held = target.shift(skip + 1)
    r = ret.reindex_like(target)
    gross = (held * r).sum(axis=1, min_count=1).fillna(0.0)
    traded = target.shift(skip)
    dw = (traded - traded.shift(1)).abs()
    if cost is None:
        c = pd.Series(0.0, index=target.index)
    else:
        c = (dw * cost.reindex_like(target)).sum(axis=1).fillna(0.0) * cost_mult
    return pd.DataFrame({"gross": gross, "cost": c, "net": gross - c})


def daily_cost(index: pd.DatetimeIndex, symbols: list[str], model: dict, slot: int | None = None) -> pd.DataFrame:
    """One-way cost per symbol and day: S05b base half-spread (symbol, year) x profile of one 15-minute slot
    (default: the last slot of the session)."""
    prof = model["profile"]
    pf = float(prof.loc[prof.index.max() if slot is None else slot])
    years = index.year
    out = np.empty((len(index), len(symbols)))
    for y in np.unique(years):
        ym = model["year_median"].get(y, model["year_median"].median())
        row = []
        for s in symbols:
            v = model["base"].at[s, y] if (s in model["base"].index and y in model["base"].columns) else np.nan
            if not np.isfinite(v):
                v = model["sym_median"].get(s, np.nan)
            row.append(v if np.isfinite(v) else ym)
        out[years == y] = np.array(row) * pf
    return pd.DataFrame(out, index=index, columns=symbols)


def block_bootstrap(x: np.ndarray, block: int, n_boot: int, seed: int, alpha: float) -> tuple[float, float, float]:
    """Mean and two-sided (1 - alpha) moving-block bootstrap interval."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, (n_boot, nb))
    m = np.empty(n_boot)
    for k in range(0, n_boot, 1000):                       # chunks keep memory small
        idx = (st[k:k + 1000, :, None] + np.arange(block)).reshape(len(st[k:k + 1000]), -1)[:, :n]
        m[k:k + 1000] = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def arm_d_horizon(signal: pd.DataFrame, labels: pd.DataFrame, ret: pd.DataFrame, cost: pd.DataFrame | None,
                  h: int, cfg: P1Config, window: pd.DatetimeIndex) -> dict[int, pd.DataFrame]:
    """{group: daily gross/cost/net (1x) of that group's reversal portfolio} over `window`."""
    out = {}
    for g in range(cfg.n_groups):
        tgt = overlap(reversal_weights(signal, labels, g, cfg.n_quantiles), h)
        out[g] = ls_returns(tgt, ret, cost, cfg.skip).loc[window]
    return out


def shuffle_labels(labels: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute the labels among the labelled stocks within each day (group sizes kept)."""
    a = labels.to_numpy().copy()
    for t in range(len(a)):
        m = np.flatnonzero(a[t] >= 0)
        if len(m) > 1:
            a[t, m] = a[t, m][rng.permutation(len(m))]
    return pd.DataFrame(a, index=labels.index, columns=labels.columns)


def label_placebo(signal, labels, ret, h, cfg: P1Config, window, n: int, seed: int) -> np.ndarray:
    """Mean daily gross of (low-group reversal - high-group reversal) under shuffled labels."""
    rng = np.random.default_rng(seed)
    hi = cfg.n_groups - 1
    out = np.empty(n)
    for k in range(n):
        lab = shuffle_labels(labels, rng)
        r0 = ls_returns(overlap(reversal_weights(signal, lab, 0, cfg.n_quantiles), h), ret, None, cfg.skip).loc[window]
        r2 = ls_returns(overlap(reversal_weights(signal, lab, hi, cfg.n_quantiles), h), ret, None, cfg.skip).loc[window]
        out[k] = float((r0["gross"] - r2["gross"]).mean())
    return out


def gates_d(boot_diff: dict, p_label: float, boot_high: dict, boot_low_net: dict, low_net_2x_mean: float,
            alpha: float) -> dict:
    d1 = all(v[1] > 0 for v in boot_diff.values()) and p_label <= alpha
    d2 = all(v[2] < 0 for v in boot_high.values())
    d3 = all(v[1] > 0 for v in boot_low_net.values()) and low_net_2x_mean > 0
    rep = "REPLICATES" if (d1 and d2) else ("PARTIAL" if d1 else "NO_REPLICATION")
    return {"D1_turnover_dependence": d1, "D2_high_turnover_momentum": d2, "D3_low_turnover_net": d3,
            "replication_verdict": rep,
            "economic_verdict": "PROFITABLE_AFTER_COSTS" if d3 else "NOT_PROFITABLE_AFTER_COSTS"}


# =================================================================== arm I
def labels_on_days(labels: pd.DataFrame, days, symbols: list[str]) -> np.ndarray:
    """(D, N) int8 labels aligned to the 15-minute panel's session dates and traded symbols (-1 if none)."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(days)))
    lab = labels.reindex(index=idx, columns=symbols)
    return lab.fillna(-1).to_numpy().astype(np.int8)


def random_masks(lab_days: np.ndarray, group: int, rng: np.random.Generator) -> np.ndarray:
    """(D, N) bool: each day, a random set of labelled stocks of the same size as `group`."""
    out = np.zeros(lab_days.shape, bool)
    for d in range(len(lab_days)):
        pool = np.flatnonzero(lab_days[d] >= 0)
        k = int((lab_days[d] == group).sum())
        if k:
            out[d, rng.choice(pool, size=k, replace=False)] = True
    return out


def trade_label_rows(entry_bar: np.ndarray, day_start: np.ndarray) -> np.ndarray:
    """Decision-day row whose mask governed each entry: entries fill at the open after the signal bar, so a fill
    on the first bar of day t came from a signal on day t - 1 (mask row t - 2); any other fill on day t came from
    a signal on day t (mask row t - 1)."""
    day = np.searchsorted(day_start, entry_bar, side="right") - 1
    first = entry_bar == day_start[day]
    return day - 1 - first.astype(int)


def cluster_bootstrap_diff(x: np.ndarray, grp: np.ndarray, cluster: np.ndarray, a: int, b: int,
                           n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """mean(x | grp == a) - mean(x | grp == b), resampling clusters (entry days) with replacement."""
    df = pd.DataFrame({"x": x, "g": grp, "c": cluster})
    df = df[df["g"].isin([a, b])]
    agg = df.groupby(["c", "g"])["x"].agg(["sum", "count"]).unstack("g", fill_value=0)
    S, C = agg["sum"].reindex(columns=[a, b], fill_value=0).to_numpy(), agg["count"].reindex(columns=[a, b], fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(S)
    pick = rng.integers(0, n, (n_boot, n))
    ss, cc = S[pick].sum(1), C[pick].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = ss[:, 0] / cc[:, 0] - ss[:, 1] / cc[:, 1]
    obs = S[:, 0].sum() / C[:, 0].sum() - S[:, 1].sum() / C[:, 1].sum()
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(obs), float(lo), float(hi)


def run_mask_backtest(panel, params, stock_cols, etf_cols, cfg, mask, seed: int, start, half_spread) -> dict:
    """One placebo-membership backtest -> metrics row (module level so joblib memory-maps the arrays)."""
    from .s04 import perf_metrics
    from .s05 import backtest
    r = backtest(panel, params, stock_cols, etf_cols, cfg, record=False, half_spread=half_spread, entry_mask=mask)
    return {"seed": seed, **perf_metrics(r["daily_equity"], cfg.initial_equity, start)}


def gates_i(diff_ci: tuple, ranks: dict, net1: dict, net2: dict, boot_net: dict, alpha: float = 0.05) -> dict:
    i1 = diff_ci[1] > 0
    i2 = ranks["total_return"]["p_upper"] <= alpha and ranks["sharpe"]["p_upper"] <= alpha
    i3 = net1["total_return"] > 0 and net2["total_return"] > 0 and all(v[1] > 0 for v in boot_net.values())
    return {"I1_turnover_dependence": i1, "I2_vs_random_membership": i2, "I3_economic": i3,
            "verdict": "EDGE_CANDIDATE" if (i1 and i2 and i3) else "NO_EDGE",
            "prospective_verdict": "PENDING_2026_HOLDOUT"}
