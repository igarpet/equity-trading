"""Hierarchical Markov regimes on MNQ (HMK 01; discussions/MNQ_Hierarchical_Markov/001).

Three tiers, each a Gaussian HMM on its own bar returns (price only):

    daily (CME day)  ->  1 hour  ->  15 minutes

The child's transition matrix depends on the parent's state (Iván, 2026-11-25/26): with G parent states,

    P(s_t = j | s_{t-1} = i, parent) = sum_g w_t[g] * A_g[i, j]

where w_t is the parent's ONE-STEP-AHEAD state distribution, known at the close of bar t-1:
    1h bar of CME day d    -> w = P(D_d | F_{d-1})          (daily forecast made at the previous day's close)
    15m bar in hour h      -> w = P(H_h | F_{h-1})          (hourly forecast made at the previous hour's close)
EM treats the parent component g as latent (responsibility w_g A_g[i,j] / A_t[i,j]), so the joint
P(child = j, parent = g) = w_g A_g[i, j] is estimated directly, not as a product of two independent matrices.

The forecast for bar t is made at the close of bar t-1 from the filtered state and the parent forecast:
    P(s_t | F_{t-1}) -> P(up_t) = sum_j P(s_t=j) Phi(mu_j / sd_j),  E[r_t] = sum_j P(s_t=j) mu_j.

States are sorted by emission mean (0 = most bearish). Returns are in basis points.
Conventions: 1-minute Databento bars (``micro_bars``), CME day 17:00-16:00 Chicago labelled by its end date,
bars aligned to the session open (23 hourly, 92 quarter-hour bars). The first bar of each CME day is not an
observation (it carries the daily break / weekend) and is never traded; positions are flat at each day's close.
Nothing from 2026 is read. Everything economic is in ticks per contract (1 tick = 0.25 pt = $0.50).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from .futures import CHICAGO, cme_trade_date

HMK_VERSION = "0.1.0"
TICK = 0.25
USD_PER_TICK = 0.5
HOLDOUT_START = dt.date(2026, 1, 1)
SESSION_MIN = 23 * 60
TIERS = {"1d": SESSION_MIN, "1h": 60, "15m": 15}
D = dt.date


@dataclass(frozen=True)
class HMKConfig:
    root: str = "MNQ"
    load_start: dt.date = D(2019, 11, 1)          # 20-day state needs a run-in
    fit_start: dt.date = D(2020, 1, 1)
    refits_discovery: tuple = (D(2021, 1, 1), D(2022, 1, 1), D(2023, 1, 1))
    discovery: tuple = (D(2021, 1, 1), D(2023, 6, 30))
    refits_confirm: tuple = (D(2023, 7, 1), D(2024, 1, 1), D(2025, 1, 1))
    confirm: tuple = (D(2023, 7, 1), D(2025, 12, 31))
    periods_discovery: tuple = (("2021", D(2021, 1, 1), D(2021, 12, 31)), ("2022", D(2022, 1, 1), D(2022, 12, 31)),
                                ("2023H1", D(2023, 1, 1), D(2023, 6, 30)))
    periods_confirm: tuple = (("2023H2", D(2023, 7, 1), D(2023, 12, 31)), ("2024", D(2024, 1, 1), D(2024, 12, 31)),
                              ("2025", D(2025, 1, 1), D(2025, 12, 31)))
    K: int = 3
    n_starts: int = 8
    max_iter: int = 300
    tol: float = 1e-7
    min_sd_bp: float = 0.5
    min_day_minutes: int = 900                    # CME days with fewer traded minutes are dropped
    rule_window: int = 20                         # video baseline: 20-day sum of daily returns
    rule_band: float = 0.05                       # +-5%
    commission_per_side: float = 0.50             # USD per contract per side (Lucid)
    slip_ticks: float = 0.5                       # per side at market
    realized_round_trip_ticks: float = 1.6        # Iván's stated realized round trip
    nw_lags: int = 5
    alpha: float = 0.05

    @property
    def half_rt_market(self) -> float:
        return self.slip_ticks + self.commission_per_side / USD_PER_TICK

    @property
    def round_trip_market(self) -> float:
        return 2 * self.half_rt_market


def guard(end: dt.date) -> None:
    if end >= HOLDOUT_START:
        raise RuntimeError("2026 is the sealed holdout: HMK reads nothing from 2026")


# ----------------------------------------------------------------------------- bars
def session_bars(m1: pd.DataFrame, cfg: HMKConfig = HMKConfig()) -> dict[str, pd.DataFrame]:
    """1-minute bars (index ts UTC: close, volume, instrument_id) -> {'1d','1h','15m'} grids.

    Columns: trade_date, k (bar in day), ts (bar open UTC), close (carried within day), minutes, iid,
    ret_bp (log return vs the previous ROW's close, same contract; NaN otherwise), obs, first.
    Intraday: obs = traded bar, previous row traded, same contract, k >= 1. Daily: close-to-close, same contract.
    """
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index)
    td = pd.to_datetime(cme_trade_date(pd.Series(ts)).to_numpy())
    loc = ts.tz_convert(CHICAGO).tz_localize(None)
    start = td - pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    mos = ((loc - start) / pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    ok = (mos >= 0) & (mos < SESSION_MIN)
    base = pd.DataFrame({"trade_date": td[ok], "mos": mos[ok], "close": m1["close"].to_numpy(float)[ok],
                         "iid": m1["instrument_id"].to_numpy()[ok]})
    dm = base.groupby("trade_date").size()
    good = dm.index[dm >= cfg.min_day_minutes]
    base = base[base.trade_date.isin(good)]
    out = {}
    for tier, width in TIERS.items():
        n = SESSION_MIN // width
        b = base.assign(k=base.mos // width)
        g = b.groupby(["trade_date", "k"], sort=True)
        agg = pd.DataFrame({"close": g.close.last(), "minutes": g.close.size(), "iid": g.iid.last()})
        full = pd.MultiIndex.from_product([good, np.arange(n)], names=["trade_date", "k"])
        agg = agg.reindex(full).reset_index()
        traded = agg.minutes.notna().to_numpy()
        agg["minutes"] = agg.minutes.fillna(0).astype(int)
        gd = agg.groupby("trade_date", sort=False)
        agg["close"] = gd.close.ffill()
        agg["iid"] = gd.iid.ffill()
        lc = np.log(agg.close.to_numpy(float))
        iid = agg.iid.to_numpy(float)
        prev_traded = np.r_[False, traded[:-1]]
        same = np.r_[False, (iid[1:] == iid[:-1]) & np.isfinite(iid[1:])]
        ret = np.r_[np.nan, np.diff(lc)] * 1e4
        first = agg.k.to_numpy() == 0
        if tier == "1d":
            obs = traded & same
        else:
            obs = traded & prev_traded & same & ~first
        agg["ret_bp"] = np.where(same, ret, np.nan)
        agg["obs"] = obs & np.isfinite(agg.ret_bp.to_numpy())
        agg["first"] = first
        open_ = (pd.to_datetime(agg.trade_date) - pd.Timedelta(days=1) + pd.Timedelta(hours=17))
        agg["ts"] = (open_.dt.tz_localize(CHICAGO, ambiguous="NaT", nonexistent="shift_forward")
                     + pd.to_timedelta(agg.k * width, unit="min")).dt.tz_convert("UTC")
        out[tier] = agg
    return out


def rule_states(daily: pd.DataFrame, cfg: HMKConfig = HMKConfig()) -> np.ndarray:
    """Video baseline, observable: state of day d from the 20-day sum of daily log returns ending at d.
    0 bear (<= -5%), 1 sideways, 2 bull (>= +5%). NaN-safe: missing returns count as 0, first 19 days -> sideways."""
    r = np.nan_to_num(daily.ret_bp.to_numpy(float)) / 1e4
    s = pd.Series(r).rolling(cfg.rule_window, min_periods=cfg.rule_window).sum().to_numpy()
    st = np.ones(len(r), np.int64)
    st[s >= cfg.rule_band] = 2
    st[s <= -cfg.rule_band] = 0
    return st


# ----------------------------------------------------------------------------- conditional HMM kernels
@njit(cache=True)
def _logb(y, obs, mu, sd):
    T, K = y.shape[0], mu.shape[0]
    out = np.zeros((T, K))
    for t in range(T):
        if obs[t]:
            for k in range(K):
                z = (y[t] - mu[k]) / sd[k]
                out[t, k] = -0.5 * z * z - math.log(sd[k]) - 0.9189385332046727
    return out


@njit(cache=True)
def _forward(logb, W, A, pi):
    T, K = logb.shape
    G = A.shape[0]
    alpha = np.zeros((T, K)); prior = np.zeros((T, K)); c = np.ones(T)
    ll = 0.0
    At = np.zeros((K, K))
    for t in range(T):
        if t == 0:
            for j in range(K):
                prior[0, j] = pi[j]
        else:
            for i in range(K):
                for j in range(K):
                    s = 0.0
                    for g in range(G):
                        s += W[t, g] * A[g, i, j]
                    At[i, j] = s
            for j in range(K):
                s = 0.0
                for i in range(K):
                    s += alpha[t - 1, i] * At[i, j]
                prior[t, j] = s
        m = logb[t, 0]
        for k in range(1, K):
            if logb[t, k] > m:
                m = logb[t, k]
        tot = 0.0
        for k in range(K):
            v = prior[t, k] * math.exp(logb[t, k] - m)
            alpha[t, k] = v
            tot += v
        if tot <= 0:
            tot = 1e-300
        for k in range(K):
            alpha[t, k] /= tot
        c[t] = tot
        ll += math.log(tot) + m
    return alpha, prior, c, ll


@njit(cache=True)
def _backward(logb, W, A, alpha, c):
    T, K = logb.shape
    G = A.shape[0]
    beta = np.ones((T, K))
    xi = np.zeros((G, K, K))
    gamma = np.zeros((T, K))
    At = np.zeros((K, K))
    bt = np.zeros(K)
    for t in range(T - 1, 0, -1):
        m = logb[t, 0]
        for k in range(1, K):
            if logb[t, k] > m:
                m = logb[t, k]
        for k in range(K):
            bt[k] = math.exp(logb[t, k] - m)
        for i in range(K):
            for j in range(K):
                s = 0.0
                for g in range(G):
                    s += W[t, g] * A[g, i, j]
                At[i, j] = s
        for i in range(K):
            s = 0.0
            for j in range(K):
                s += At[i, j] * bt[j] * beta[t, j]
            beta[t - 1, i] = s / c[t]
        for i in range(K):
            for j in range(K):
                x = alpha[t - 1, i] * At[i, j] * bt[j] * beta[t, j] / c[t]
                if At[i, j] > 0 and x > 0:
                    for g in range(G):
                        xi[g, i, j] += x * W[t, g] * A[g, i, j] / At[i, j]
    for t in range(T):
        s = 0.0
        for k in range(K):
            gamma[t, k] = alpha[t, k] * beta[t, k]
            s += gamma[t, k]
        for k in range(K):
            gamma[t, k] /= s
    return gamma, xi


@dataclass
class CondHMM:
    mu: np.ndarray
    sd: np.ndarray
    A: np.ndarray            # (G, K, K)
    pi: np.ndarray
    loglik: float = float("nan")
    n_iter: int = 0
    n_obs: int = 0

    @property
    def K(self):
        return len(self.mu)

    @property
    def G(self):
        return self.A.shape[0]

    def p_up_state(self) -> np.ndarray:
        from scipy.stats import norm
        return norm.cdf(self.mu / self.sd)

    def summary(self) -> dict:
        return {"mu_bp": self.mu.round(3).tolist(), "sd_bp": self.sd.round(3).tolist(),
                "p_up_state": self.p_up_state().round(4).tolist(), "A": self.A.round(4).tolist(),
                "loglik": self.loglik, "n_iter": self.n_iter, "n_obs": self.n_obs}


def _sort(m: CondHMM) -> CondHMM:
    o = np.argsort(m.mu)
    return CondHMM(m.mu[o], m.sd[o], m.A[:, o][:, :, o], m.pi[o], m.loglik, m.n_iter, m.n_obs)


def fit(y: np.ndarray, obs: np.ndarray, W: np.ndarray, cfg: HMKConfig = HMKConfig(), seed: int = 0) -> CondHMM:
    """EM, best of cfg.n_starts starts. y in bp; obs bool; W (T, G) parent weights (rows sum to 1)."""
    y = np.where(obs, y, 0.0).astype(float)
    obs = obs.astype(np.bool_)
    W = np.ascontiguousarray(W, float)
    K, G = cfg.K, W.shape[1]
    yo = y[obs]
    rng = np.random.default_rng(seed)
    best = None
    for s in range(cfg.n_starts):
        qs = np.sort(rng.uniform(0.15, 0.85, K))
        mu = np.quantile(yo, qs) * rng.uniform(0.3, 1.0)
        sd = np.std(yo) * rng.uniform(0.5, 1.5, K)
        sd = np.maximum(sd, cfg.min_sd_bp)
        A = np.zeros((G, K, K))
        for g in range(G):
            a = rng.dirichlet(np.ones(K), K) * 0.2 + np.eye(K) * 0.8
            A[g] = a / a.sum(1, keepdims=True)
        pi = np.ones(K) / K
        prev = -np.inf
        it = 0
        for it in range(1, cfg.max_iter + 1):
            lb = _logb(y, obs, mu, sd)
            alpha, prior, c, ll = _forward(lb, W, A, pi)
            gamma, xi = _backward(lb, W, A, alpha, c)
            go = gamma[obs]
            wsum = go.sum(0) + 1e-12
            mu = (go * yo[:, None]).sum(0) / wsum
            sd = np.sqrt((go * (yo[:, None] - mu) ** 2).sum(0) / wsum)
            sd = np.maximum(sd, cfg.min_sd_bp)
            A = (xi + 1e-6) / (xi + 1e-6).sum(2, keepdims=True)
            pi = gamma[0] / gamma[0].sum()
            if ll - prev < cfg.tol * abs(ll):
                break
            prev = ll
        lb = _logb(y, obs, mu, sd)
        ll = _forward(lb, W, A, pi)[3]
        m = CondHMM(mu, sd, A, pi, ll, it, int(obs.sum()))
        if best is None or ll > best.loglik:
            best = m
    return _sort(best)


def filter_run(m: CondHMM, y: np.ndarray, obs: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal pass with fixed parameters: (filtered P(s_t | F_t), forecast P(s_t | F_{t-1}))."""
    y = np.where(obs, y, 0.0).astype(float)
    lb = _logb(y, obs.astype(np.bool_), m.mu, m.sd)
    alpha, prior, _, _ = _forward(lb, np.ascontiguousarray(W, float), m.A, m.pi)
    return alpha, prior


def predictive(m: CondHMM, prior: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per row: P(up), E[r] (bp) and the predictive log density of the realized return (bp)."""
    from scipy.stats import norm
    pup = prior @ m.p_up_state()
    emu = prior @ m.mu
    dens = (prior * norm.pdf((np.nan_to_num(y)[:, None] - m.mu) / m.sd) / m.sd).sum(1)
    return pd.DataFrame({"p_up": pup, "e_bp": emu, "logscore": np.log(np.maximum(dens, 1e-300))})


# ----------------------------------------------------------------------------- the three-tier pipeline
def _onehot(s: np.ndarray, G: int) -> np.ndarray:
    W = np.zeros((len(s), G)); W[np.arange(len(s)), s] = 1.0
    return W


def daily_forecast_for_day(daily: pd.DataFrame, prior_daily: np.ndarray) -> pd.DataFrame:
    """P(D_d | F_{d-1}) per trade_date (the forward 'prior' of the daily row)."""
    return pd.DataFrame(prior_daily, index=pd.to_datetime(daily.trade_date))


def parent_weights(child: pd.DataFrame, parent_by_key: pd.DataFrame, key: pd.Series) -> np.ndarray:
    W = parent_by_key.reindex(key.to_numpy()).to_numpy(float).copy()
    bad = ~np.isfinite(W).all(1)
    W[bad] = 1.0 / W.shape[1]
    return W / W.sum(1, keepdims=True)


def hour_key(df: pd.DataFrame) -> pd.Series:
    """(trade_date, hour index) key of a 15m row, as a string to reindex the hourly table."""
    return df.trade_date.astype(str) + "|" + (df.k // 4).astype(str)


VARIANTS = ("H3", "U", "RULE")


def run_segment(bars: dict, cfg: HMKConfig, fit_end: dt.date, seg: tuple, seed: int = 0) -> dict:
    """Fit every tier on [fit_start, fit_end) and filter causally through seg[1]; return segment rows and fits."""
    guard(seg[1])
    d, h, q = (bars[t] for t in ("1d", "1h", "15m"))
    tdd, tdh, tdq = (pd.to_datetime(x.trade_date).dt.date for x in (d, h, q))
    fs = cfg.fit_start
    run_d, run_h, run_q = (tdd <= seg[1]).to_numpy(), (tdh <= seg[1]).to_numpy(), (tdq <= seg[1]).to_numpy()
    fit_d = ((tdd >= fs) & (tdd < fit_end)).to_numpy()[run_d]
    fit_h = ((tdh >= fs) & (tdh < fit_end)).to_numpy()[run_h]
    fit_q = ((tdq >= fs) & (tdq < fit_end)).to_numpy()[run_q]
    D_, H_, Q_ = d[run_d].reset_index(drop=True), h[run_h].reset_index(drop=True), q[run_q].reset_index(drop=True)
    fits, pred = {}, {}

    # daily tier (no parent)
    yd, od = D_.ret_bp.to_numpy(float), D_.obs.to_numpy(bool)
    W1 = np.ones((len(D_), 1))
    md = fit(yd[fit_d], od[fit_d], W1[fit_d], cfg, seed)
    _, prior_d = filter_run(md, yd, od, W1)
    fits["1d"] = md
    by_day = daily_forecast_for_day(D_, prior_d)
    rule = rule_states(D_, cfg)
    # the rule state of day d-1 is the parent of day d (known at d-1's close)
    rule_prev = np.r_[1, rule[:-1]]
    by_day_rule = pd.DataFrame(_onehot(rule_prev, 3), index=pd.to_datetime(D_.trade_date))

    yh, oh = H_.ret_bp.to_numpy(float), H_.obs.to_numpy(bool)
    keyh = pd.to_datetime(H_.trade_date)
    Wh = {"H3": parent_weights(H_, by_day, keyh), "U": np.ones((len(H_), 1)),
          "RULE": parent_weights(H_, by_day_rule, keyh)}
    hourly_prior = {}
    for v, W in Wh.items():
        m = fit(yh[fit_h], oh[fit_h], W[fit_h], cfg, seed)
        _, pr = filter_run(m, yh, oh, W)
        fits[f"1h_{v}"] = m
        hourly_prior[v] = pr
        pred[f"1h_{v}"] = pd.concat([H_[["trade_date", "k", "ts", "close", "ret_bp", "obs", "first"]],
                                     predictive(m, pr, yh), pd.DataFrame(W, columns=[f"w{g}" for g in range(W.shape[1])])],
                                    axis=1)
    # 15m tier: parent = hourly forecast of the hour that contains the bar (H3 chain), or no parent (U)
    yq, oq = Q_.ret_bp.to_numpy(float), Q_.obs.to_numpy(bool)
    hk = H_.trade_date.astype(str) + "|" + H_.k.astype(str)
    by_hour = pd.DataFrame(hourly_prior["H3"], index=hk.to_numpy())
    Wq = {"H3": parent_weights(Q_, by_hour, hour_key(Q_)), "U": np.ones((len(Q_), 1))}
    for v, W in Wq.items():
        m = fit(yq[fit_q], oq[fit_q], W[fit_q], cfg, seed)
        _, pr = filter_run(m, yq, oq, W)
        fits[f"15m_{v}"] = m
        pred[f"15m_{v}"] = pd.concat([Q_[["trade_date", "k", "ts", "close", "ret_bp", "obs", "first"]],
                                      predictive(m, pr, yq), pd.DataFrame(W, columns=[f"w{g}" for g in range(W.shape[1])])],
                                     axis=1)
    pred["1d"] = pd.concat([D_[["trade_date", "close", "ret_bp", "obs"]], predictive(md, prior_d, yd),
                            pd.DataFrame({"rule_state": rule})], axis=1)
    base_up = {t: float((x.ret_bp[x.obs & m_] > 0).mean()) for t, x, m_ in
               (("1h", H_, fit_h), ("15m", Q_, fit_q), ("1d", D_, fit_d))}
    lo, hi = seg
    for k_, df in pred.items():
        tdk = pd.to_datetime(df.trade_date).dt.date
        df = df[(tdk >= lo) & (tdk <= hi)].copy()
        df["refit"] = str(fit_end)
        df["base_up"] = base_up[k_.split("_")[0]]
        pred[k_] = df.reset_index(drop=True)
    return {"pred": pred, "fits": fits}


def walk_forward(bars: dict, cfg: HMKConfig, refits: tuple, end: dt.date, seed: int = 0, log=None) -> dict:
    """Expanding fits at each refit date; each fit forecasts [refit, next refit) causally."""
    out = {"pred": {}, "fits": {}}
    for i, r in enumerate(refits):
        seg_end = (refits[i + 1] - dt.timedelta(days=1)) if i + 1 < len(refits) else end
        res = run_segment(bars, cfg, r, (r, seg_end), seed)
        for k, v in res["pred"].items():
            out["pred"].setdefault(k, []).append(v)
        out["fits"][str(r)] = {k: m.summary() for k, m in res["fits"].items()}
        if log:
            log(f"refit {r}: segment {r} -> {seg_end} done")
    out["pred"] = {k: pd.concat(v, ignore_index=True) for k, v in out["pred"].items()}
    return out


# ----------------------------------------------------------------------------- evaluation
def _nw_t(x: np.ndarray, lags: int) -> tuple[float, float]:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    m = x.mean(); e = x - m
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(v, 1e-300) / n)
    return float(m), float(m / se)


def _p_one(t: float) -> float:
    from scipy.stats import norm
    return float(norm.sf(t)) if np.isfinite(t) else float("nan")


def eligible(df: pd.DataFrame) -> pd.Series:
    return df.obs & ~df["first"]


def positions(df: pd.DataFrame) -> pd.DataFrame:
    """Always-in sign(E[r]) for every bar k >= 1, decided at the previous bar's close; flat at the day's close.
    Gross ticks from the previous row's close; units traded per day = entry + flips + exit."""
    z = df.copy()
    z["dclose_ticks"] = z.groupby("trade_date").close.diff() / TICK
    z = z[~z["first"]].copy()
    z["pos"] = np.sign(z.e_bp).replace(0, np.nan)
    z["pos"] = z.groupby("trade_date").pos.ffill().fillna(0.0)
    z["gross"] = z.pos * z.dclose_ticks.fillna(0.0)
    prev = z.groupby("trade_date").pos.shift(1).fillna(0.0)
    z["units"] = np.abs(z.pos - prev)
    last = z.groupby("trade_date").cumcount(ascending=False) == 0
    z.loc[last, "units"] += np.abs(z.loc[last, "pos"])
    return z


def daily_pnl(df: pd.DataFrame, cfg: HMKConfig) -> pd.DataFrame:
    z = positions(df)
    g = z.groupby("trade_date")
    out = pd.DataFrame({"gross": g.gross.sum(), "units": g.units.sum(), "bars": g.size()})
    long = z.dclose_ticks.fillna(0.0).groupby(z.trade_date).sum()
    out["net_market"] = out.gross - out.units * cfg.half_rt_market
    out["net_realized"] = out.gross - out.units * cfg.realized_round_trip_ticks / 2
    out["long_gross"] = long
    out["long_net_market"] = long - 2 * cfg.half_rt_market
    return out


def evaluate(df: pd.DataFrame, bench: pd.DataFrame | None, cfg: HMKConfig, label: str) -> dict:
    """Gates of HMK 001 §6 for one tier/variant over the rows of ``df`` (one evaluation window)."""
    e = df[eligible(df)].copy()
    y = (e.ret_bp > 0).astype(float)
    nz = e.ret_bp != 0
    res = {"label": label, "bars": int(len(e)), "days": int(e.trade_date.nunique())}
    # G2: Brier vs the training base rate, daily mean of (B0 - B), NW
    e["brier"] = (e.p_up - y) ** 2
    e["brier0"] = (e.base_up - y) ** 2
    dd = (e.brier0 - e.brier).groupby(e.trade_date).mean()
    m, t = _nw_t(dd.to_numpy(), cfg.nw_lags)
    res.update(brier=float(e.brier.mean()), brier_base=float(e.brier0.mean()),
               bss=float(1 - e.brier.mean() / e.brier0.mean()), g2_t=t, g2_p=_p_one(t))
    res.update(hit=float((np.sign(e.e_bp[nz]) == np.sign(e.ret_bp[nz])).mean()),
               up_share=float((e.ret_bp[nz] > 0).mean()), mean_p_up=float(e.p_up.mean()),
               share_p_up_above_half=float((e.p_up > 0.5).mean()))
    # G1: predictive log score vs the benchmark (unconditional) model, daily sums, NW
    if bench is not None:
        b = bench[eligible(bench)][["trade_date", "k", "logscore"]].rename(columns={"logscore": "ls_b"})
        j = e[["trade_date", "k", "logscore"]].merge(b, on=["trade_date", "k"])
        diff = (j.logscore - j.ls_b).groupby(j.trade_date).sum()
        m1, t1 = _nw_t(diff.to_numpy(), cfg.nw_lags)
        res.update(g1_logscore_per_day=m1, g1_t=t1, g1_p=_p_one(t1))
    # G3: economics, always-in sign(E[r]) vs always long, daily net at market cost
    pnl = daily_pnl(df, cfg)
    m3, t3 = _nw_t(pnl.net_market.to_numpy(), cfg.nw_lags)
    md, td = _nw_t((pnl.net_market - pnl.long_net_market).to_numpy(), cfg.nw_lags)
    res.update(net_ticks_day=m3, g3_t=t3, g3_p=_p_one(t3), net_realized_day=float(pnl.net_realized.mean()),
               gross_ticks_day=float(pnl.gross.mean()), units_day=float(pnl.units.mean()),
               long_net_day=float(pnl.long_net_market.mean()), vs_long_ticks_day=md, vs_long_t=td,
               vs_long_p=_p_one(td), net_usd_day=m3 * USD_PER_TICK)
    # law-of-large-numbers check: a bar-by-bar flip pays one round trip per bar and earns (2p - 1) E|move|
    mean_abs = float((df.groupby("trade_date").close.diff().abs() / TICK)[eligible(df)].mean())
    res.update(mean_abs_move_ticks=mean_abs,
               breakeven_hit_if_every_bar_flips=float(0.5 + cfg.round_trip_market / (2 * mean_abs))
               if mean_abs > 0 else float("nan"))
    return res


def calibration(df: pd.DataFrame, bins=(0, .45, .48, .49, .5, .51, .52, .55, 1.0)) -> pd.DataFrame:
    e = df[eligible(df)]
    b = pd.cut(e.p_up, bins, include_lowest=True).rename("bin")
    return e.groupby(b, observed=True).agg(n=("p_up", "size"), p_up=("p_up", "mean"),
                                           realized_up=("ret_bp", lambda s: float((s > 0).mean())),
                                           mean_ret_bp=("ret_bp", "mean")).reset_index()


def conditional_matrices(fit_summary: dict, tier: str = "1h_H3") -> pd.DataFrame:
    """The fitted P(child state j | child state i, parent state g), long format, for one refit."""
    A = np.asarray(fit_summary[tier]["A"])
    rows = []
    names = {0: "bear", 1: "side", 2: "bull"} if A.shape[1] == 3 else {i: f"s{i}" for i in range(A.shape[1])}
    for g in range(A.shape[0]):
        for i in range(A.shape[1]):
            for j in range(A.shape[2]):
                rows.append({"parent": names.get(g, g) if A.shape[0] == 3 else g, "from": names[i], "to": names[j],
                             "p": A[g, i, j]})
    return pd.DataFrame(rows)


def verdict(r: dict, per: list[dict], cfg: HMKConfig) -> dict:
    """HMK 001 §7. G1 needs a benchmark (absent for the U models, which are judged on G2-G4 only)."""
    g1 = bool(r.get("g1_p", np.nan) < cfg.alpha)
    g2 = bool(r["g2_p"] < cfg.alpha and r["bss"] > 0)
    g3 = bool(r["g3_p"] < cfg.alpha and r["vs_long_p"] < cfg.alpha)
    need = int(np.ceil(2 / 3 * len(per)))
    g4 = bool(sum((p["net_ticks_day"] > 0) and (p["bss"] > 0) for p in per) >= need)
    tradable = g2 and g3 and g4
    if tradable and g1:
        v = "EDGE_CANDIDATE"                      # the hierarchy adds information and the forecast pays
    elif tradable:
        v = "EDGE_WITHOUT_HIERARCHY"              # the forecast pays, the parent adds nothing measurable
    elif g2 and not g3:
        v = "SKILL_BELOW_COST"
    elif g1:
        v = "CONDITIONING_ONLY"
    else:
        v = "NO_EDGE"
    return {"G1_conditioning": g1, "G2_direction_skill": g2, "G3_net_and_beats_long": g3, "G4_stable": g4,
            "verdict": v}


def in_window(df: pd.DataFrame, a: dt.date, b: dt.date) -> pd.DataFrame:
    t = pd.to_datetime(df.trade_date).dt.date
    return df[(t >= a) & (t <= b)]


# ----------------------------------------------------------------------------- synthetic data
def write_synthetic(base, start: dt.date, end: dt.date, effect: float = 1.0, seed: int = 3,
                    parent_only: bool = False) -> None:
    """MNQ ohlcv-1m month files under ``base``. Daily regime chain (bear/side/bull, stay 0.9) with daily drift;
    an hourly chain whose transitions depend on the daily regime (bull days favour the up state, bear days the
    down state) with hourly drift +-12 x effect bp per hour (sd 30 bp). effect = 0 -> no structure (null).
    parent_only: hourly states are drawn afresh each hour from a distribution set by the daily regime (no hourly
    persistence of their own), so only the parent carries information."""
    import os
    from pathlib import Path
    from . import micro_bars as MB
    base = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base)
    rng = np.random.default_rng(seed)
    Pd = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
    Ah = {0: np.array([[0.8, 0.15, 0.05], [0.4, 0.5, 0.1], [0.5, 0.3, 0.2]]),
          1: np.array([[0.6, 0.3, 0.1], [0.15, 0.7, 0.15], [0.1, 0.3, 0.6]]),
          2: np.array([[0.2, 0.3, 0.5], [0.1, 0.5, 0.4], [0.05, 0.15, 0.8]])}
    hmu = np.array([-12.0, 0.0, 12.0]) * effect           # bp per hour
    price, iid, ds, hs = 12000.0, 1, 1, 1
    rows = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            ds = rng.choice(3, p=Pd[ds])
            A = Ah[ds] if effect > 0 else Ah[1]
            if parent_only:
                row = {0: [0.6, 0.3, 0.1], 1: [0.2, 0.6, 0.2], 2: [0.1, 0.3, 0.6]}[ds] if effect > 0 else [0.2, 0.6, 0.2]
                A = np.array([row] * 3)
            open_ = pd.Timestamp(d - dt.timedelta(days=1), tz=CHICAGO) + pd.Timedelta(hours=17)
            ts = (open_ + pd.to_timedelta(np.arange(SESSION_MIN), unit="min")).tz_convert("UTC")
            if d.day == 10 and d.month in (3, 6, 9, 12):
                iid += 1; price += 40.0
            r = np.empty(SESSION_MIN)
            for hh in range(23):
                hs = rng.choice(3, p=A[hs])
                r[hh * 60:(hh + 1) * 60] = (hmu[hs] / 60 + rng.normal(0, 30 / np.sqrt(60), 60)) / 1e4
            c = price * np.exp(np.cumsum(r))
            price = float(c[-1])
            keep = rng.random(SESSION_MIN) > 0.01
            rows.append(pd.DataFrame({"ts": ts[keep], "open": c[keep], "high": c[keep] + 0.25, "low": c[keep] - 0.25,
                                      "close": np.round(c[keep] / TICK) * TICK, "volume": 10, "instrument_id": iid}))
        d += dt.timedelta(days=1)
    df = pd.concat(rows, ignore_index=True)
    df["m"] = df.ts.dt.strftime("%Y-%m")
    out = MB.root_dir("MNQ"); out.mkdir(parents=True, exist_ok=True)
    for m, g in df.groupby("m"):
        g.drop(columns="m").to_parquet(out / f"MNQ_{m}.parquet", index=False)


def registration(cfg: HMKConfig = HMKConfig()) -> dict:
    d = asdict(cfg)
    d.update(version=HMK_VERSION, half_rt_market=cfg.half_rt_market, round_trip_market=cfg.round_trip_market)
    return d


# ----------------------------------------------------------------------------- HMK 01: observable 20-bar chains (video rule)
STATE_NAMES = ("bear", "side", "bull")


def band_scaled(tier: str, band: float = 0.05) -> float:
    """The video's daily band scaled to a bar of ``tier`` by sqrt(time): 5% x sqrt(bar minutes / 1380)."""
    return band * math.sqrt(TIERS[tier] / SESSION_MIN)


def rolling_states(ret_bp: np.ndarray, window: int = 20, band: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Video rule on any bar series: sum of the last ``window`` bar log returns (this bar included) >= +band -> bull,
    <= -band -> bear, else sideways. Missing returns count as 0. Returns (state 0/1/2, valid) where valid needs a
    full window."""
    r = np.nan_to_num(np.asarray(ret_bp, float)) / 1e4
    s = pd.Series(r).rolling(window, min_periods=window).sum().to_numpy()
    st = np.ones(len(r), np.int64)
    st[s >= band] = 2
    st[s <= -band] = 0
    return st, np.isfinite(s)


def transition_counts(st: np.ndarray, valid: np.ndarray, step: int = 1) -> np.ndarray:
    """3x3 counts of (state at t, state at t + step) over pairs where both are valid."""
    a, b = st[:-step], st[step:]
    ok = valid[:-step] & valid[step:]
    C = np.zeros((3, 3))
    np.add.at(C, (a[ok], b[ok]), 1)
    return C


def to_probs(C: np.ndarray) -> np.ndarray:
    rs = C.sum(1, keepdims=True)
    return np.divide(C, rs, out=np.full_like(C, np.nan), where=rs > 0)


def stationary(P: np.ndarray) -> np.ndarray:
    Q = np.nan_to_num(P)
    w, v = np.linalg.eig(Q.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    return p / p.sum()


def run_lengths(st: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    """Mean / median length (bars) of uninterrupted runs in each state."""
    rows = []
    cur, n = None, 0
    for s, v in zip(st, valid):
        if not v:
            if cur is not None:
                rows.append((cur, n))
            cur, n = None, 0
            continue
        if s == cur:
            n += 1
        else:
            if cur is not None:
                rows.append((cur, n))
            cur, n = s, 1
    if cur is not None:
        rows.append((cur, n))
    d = pd.DataFrame(rows, columns=["state", "len"])
    return d.groupby("state").len.agg(["count", "mean", "median", "max"]).reindex(range(3)).set_axis(STATE_NAMES)


def chain_stats(st: np.ndarray, ok: np.ndarray, ret_bp: np.ndarray, window: int) -> dict:
    """Matrix, persistence, n-step, stationary, runs, non-overlapping blocks and next-bar up share for states ``st``
    on bars ``ok``."""
    r = np.asarray(ret_bp, float)
    C = transition_counts(st, ok)
    P = to_probs(C)
    idx = np.flatnonzero(ok)
    blk = np.zeros(len(st), bool)
    if len(idx):
        blk[idx[0]::window] = True
    blk &= ok
    sb = st[blk]
    Cb = np.zeros((3, 3))
    if len(sb) > 1:
        np.add.at(Cb, (sb[:-1], sb[1:]), 1)
    nxt = np.r_[r[1:], np.nan]
    m = ok & np.isfinite(nxt) & (nxt != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"window": window, "n_bars": int(ok.sum()),
               "share": pd.Series(np.bincount(st[ok], minlength=3) / max(ok.sum(), 1), index=STATE_NAMES),
               "counts": pd.DataFrame(C, index=STATE_NAMES, columns=STATE_NAMES),
               "P": pd.DataFrame(P, index=STATE_NAMES, columns=STATE_NAMES),
               "diag": pd.Series(np.diag(P), index=STATE_NAMES),
               "expected_duration": pd.Series(1 / (1 - np.diag(P)), index=STATE_NAMES),
               "P_nonoverlap": pd.DataFrame(to_probs(Cb), index=STATE_NAMES, columns=STATE_NAMES),
               "n_blocks": int(blk.sum()),
               "stationary": pd.Series(stationary(P), index=STATE_NAMES),
               "runs": run_lengths(st, ok)}
    out["P_steps"] = {n: pd.DataFrame(np.linalg.matrix_power(np.nan_to_num(P), n), index=STATE_NAMES,
                                      columns=STATE_NAMES) for n in (2, 3, 5, 20)}
    out["next_bar_up_share"] = pd.Series([float((nxt[m & (st == k)] > 0).mean()) if (m & (st == k)).any() else np.nan
                                          for k in range(3)], index=STATE_NAMES)
    out["next_bar_n"] = pd.Series([int((m & (st == k)).sum()) for k in range(3)], index=STATE_NAMES)
    return out


def _shuffled(r: np.ndarray, rng) -> np.ndarray:
    rr = r.copy()
    fin = np.isfinite(r)
    rr[fin] = rng.permutation(r[fin])
    return rr


def chain_report(ret_bp: np.ndarray, window: int = 20, band: float = 0.05, n_null: int = 200, seed: int = 0,
                 mask: np.ndarray | None = None) -> dict:
    """Observable rule (``band`` on the ``window``-bar return): chain_stats plus the shuffled-return null.

    Null A (shuffle): the same returns in random order, same rule -> the persistence that the window overlap alone
    produces for these returns. Null B (non-overlapping): states on consecutive non-overlapping blocks.
    ``mask``: bars to report (e.g. the study window); earlier bars still feed the sums (run-in)."""
    r = np.asarray(ret_bp, float)
    mask = np.ones(len(r), bool) if mask is None else np.asarray(mask, bool)
    st, ok = rolling_states(r, window, band)
    ok = ok & mask
    out = chain_stats(st, ok, r, window)
    rng = np.random.default_rng(seed)
    dn = []
    for _ in range(n_null):
        s2, o2 = rolling_states(_shuffled(r, rng), window, band)
        dn.append(np.diag(to_probs(transition_counts(s2, o2 & mask))))
    dn = np.array(dn)
    with np.errstate(invalid="ignore"):
        out["null_shuffle_diag_mean"] = pd.Series(np.nanmean(dn, 0), index=STATE_NAMES)
        out["null_shuffle_diag_q95"] = pd.Series(np.nanquantile(dn, 0.95, 0), index=STATE_NAMES)
    out["band"] = band
    return out


def band_for_window(tier: str, window: int, daily_band: float = 0.05, daily_window: int = 20) -> float:
    """The daily rule (+-5% over 20 days) scaled by sqrt(time) to ``window`` bars of ``tier``."""
    return daily_band * math.sqrt(window * TIERS[tier] / (daily_window * SESSION_MIN))


def rolling_sum_bp(ret_bp: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.nan_to_num(np.asarray(ret_bp, float))
    s = pd.Series(r).rolling(window, min_periods=window).sum().to_numpy()
    return s, np.isfinite(s)


def hmm_states(ret_bp: np.ndarray, window: int, mask: np.ndarray, cfg: HMKConfig = HMKConfig(), seed: int = 0):
    """3-state Gaussian HMM on the ``window``-bar return (bp), fitted on the masked bars; states = argmax of the
    causal filtered probabilities (no look-ahead). Returns (states, ok, fitted model)."""
    x, ok = rolling_sum_bp(ret_bp, window)
    ok = ok & mask
    W = np.ones((len(x), 1))
    m = fit(np.where(ok, x, 0.0), ok, W, cfg, seed)
    alpha, _ = filter_run(m, np.where(ok, x, 0.0), ok, W)
    return alpha.argmax(1).astype(np.int64), ok, m


def hmm_report(ret_bp: np.ndarray, window: int, mask: np.ndarray, cfg: HMKConfig = HMKConfig(), n_null: int = 20,
               seed: int = 0) -> dict:
    """HMM version of chain_report: the HMM chooses the bull/bear levels of the window return.
    Null: shuffled returns, same window, HMM refitted on each permutation."""
    r = np.asarray(ret_bp, float)
    st, ok, m = hmm_states(r, window, mask, cfg, seed)
    out = chain_stats(st, ok, r, window)
    out["state_mean_pct"] = pd.Series(m.mu / 100, index=STATE_NAMES)
    out["state_sd_pct"] = pd.Series(m.sd / 100, index=STATE_NAMES)
    out["boundaries_pct"] = ((m.mu[:-1] + m.mu[1:]) / 2 / 100).tolist()
    out["hmm_A"] = pd.DataFrame(m.A[0], index=STATE_NAMES, columns=STATE_NAMES)
    rng = np.random.default_rng(seed)
    cfg_null = HMKConfig(n_starts=2, max_iter=cfg.max_iter, tol=cfg.tol)
    dn = []
    for i in range(n_null):
        s2, o2, _ = hmm_states(_shuffled(r, rng), window, mask, cfg_null, seed + i + 1)
        dn.append(np.diag(to_probs(transition_counts(s2, o2))))
    dn = np.array(dn)
    with np.errstate(invalid="ignore"):
        out["null_shuffle_diag_mean"] = pd.Series(np.nanmean(dn, 0), index=STATE_NAMES)
        out["null_shuffle_diag_q95"] = pd.Series(np.nanquantile(dn, 0.95, 0), index=STATE_NAMES)
    return out


def tier_returns(bars: dict, tier: str, a: dt.date, b: dt.date) -> pd.DataFrame:
    """Bar returns of one tier in [a, b] (CME trade dates), every bar of the session in time order."""
    x = in_window(bars[tier], a, b).copy()
    return x[["trade_date", "k", "ts", "close", "ret_bp", "obs"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- HMK 03: are the tiers independent?
def tiers_from_15m(bars: dict) -> dict:
    """15m, 1h and daily return series all built from the same 15m bar returns (log returns add up), so a shuffle
    of the 15m returns can be carried to every tier. Index maps: 15m row -> its hour row and its day row."""
    q = bars["15m"][["trade_date", "k", "ret_bp"]].reset_index(drop=True).copy()
    q["slot"] = q.k
    q["hour"] = q.k // 4
    hk = q.groupby(["trade_date", "hour"], sort=True).ngroup().to_numpy()
    dk = q.groupby("trade_date", sort=True).ngroup().to_numpy()
    return {"q": q, "hour_row": hk, "day_row": dk, "n_hours": int(hk.max()) + 1, "n_days": int(dk.max()) + 1,
            "day_dates": pd.to_datetime(q.groupby("trade_date", sort=True).size().index).date,
            "hour_dates": pd.to_datetime(q.groupby(["trade_date", "hour"], sort=True).size().index
                                         .get_level_values(0)).date}


def _agg(r: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    """Sum of 15m returns per hour / day row; NaN if any 15m return in it is NaN (roll)."""
    s = np.bincount(rows, weights=np.nan_to_num(r), minlength=n)
    bad = np.bincount(rows, weights=~np.isfinite(r), minlength=n) > 0
    s[bad] = np.nan
    return s


def tier_returns_from(r15: np.ndarray, T: dict) -> dict:
    return {"15m": r15, "1h": _agg(r15, T["hour_row"], T["n_hours"]), "1d": _agg(r15, T["day_row"], T["n_days"])}


def shuffle_within_slot(r15: np.ndarray, slot: np.ndarray, rng) -> np.ndarray:
    """Permute 15m returns across days within each time-of-day slot (keeps the intraday volatility profile,
    destroys serial dependence and volatility clustering)."""
    out = r15.copy()
    for s in np.unique(slot):
        i = np.flatnonzero((slot == s) & np.isfinite(r15))
        out[i] = r15[rng.permutation(i)]
    return out


def hmm_classify(m: CondHMM, ret_bp: np.ndarray, window: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """States of a FIXED fitted HMM on the window return (causal filtered argmax)."""
    x, ok = rolling_sum_bp(ret_bp, window)
    ok = ok & mask
    W = np.ones((len(x), 1))
    alpha, _ = filter_run(m, np.where(ok, x, 0.0), ok, W)
    return alpha.argmax(1).astype(np.int64), ok


def mutual_info(x: np.ndarray, y: np.ndarray, ok: np.ndarray, kx: int = 3, ky: int = 3) -> tuple[float, np.ndarray]:
    """I(X; Y) in nats from the joint counts over rows ``ok``; returns (MI, counts)."""
    C = np.zeros((kx, ky))
    np.add.at(C, (x[ok], y[ok]), 1)
    n = C.sum()
    if n == 0:
        return float("nan"), C
    P = C / n
    px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P / (px * py)), 0.0)
    return float(t.sum()), C


def cond_mutual_info(nxt: np.ndarray, cur: np.ndarray, par: np.ndarray, ok: np.ndarray) -> tuple[float, np.ndarray]:
    """I(S_{t+1}; Parent | S_t) in nats: does the parent change the child's transition matrix?
    Returns (CMI, counts[parent, cur, nxt])."""
    C = np.zeros((3, 3, 3))
    np.add.at(C, (par[ok], cur[ok], nxt[ok]), 1)
    n = C.sum()
    tot = 0.0
    for i in range(3):
        Ci = C[:, i, :]
        ni = Ci.sum()
        if ni == 0:
            continue
        P = Ci / ni
        pp, pn = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(P > 0, P * np.log(P / (pp * pn)), 0.0)
        tot += ni / n * float(t.sum())
    return tot, C


@dataclass
class TierSpec:
    """How the states of one tier are made: 'band' (set +-band on the W-bar return) or 'hmm' (fitted model)."""
    tier: str
    window: int
    method: str = "band"
    band: float | None = None
    model: CondHMM | None = None

    def states(self, ret_bp: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.method == "band":
            st, ok = rolling_states(ret_bp, self.window, self.band)
            return st, ok & mask
        return hmm_classify(self.model, ret_bp, self.window, mask)


def parent_of(child_rows_parent: np.ndarray, parent_states: np.ndarray, parent_ok: np.ndarray):
    """Parent state LAST COMPLETED before the child bar: parent row index - 1 (causal)."""
    j = child_rows_parent - 1
    ok = j >= 0
    js = np.where(ok, j, 0)
    return parent_states[js], ok & parent_ok[js]


def dependence(r15: np.ndarray, T: dict, specs: dict, masks: dict) -> dict:
    """All dependence statistics for one realization of the 15m returns.

    Pairs (child | parent), parent = last completed parent bar:
      15m | 1d, 1h | 1d, 15m | 1h : marginal MI and P(child state | parent state);
      transitions: I(15m_{t+1}; 1d | 15m_t), I(15m_{t+1}; 1h | 15m_t), I(1h_{t+1}; 1d | 1h_t)."""
    R = tier_returns_from(r15, T)
    S = {t: specs[t].states(R[t], masks[t]) for t in ("1d", "1h", "15m")}
    hour_of_q, day_of_q = T["hour_row"], T["day_row"]
    # day row of each hour row
    day_of_h = np.zeros(T["n_hours"], np.int64)
    day_of_h[hour_of_q] = day_of_q
    links = {"15m|1d": ("15m", "1d", day_of_q), "1h|1d": ("1h", "1d", day_of_h), "15m|1h": ("15m", "1h", hour_of_q)}
    out = {}
    for name, (c, p, rows) in links.items():
        cs, cok = S[c]
        ps, pok = parent_of(rows, *S[p])
        ok = cok & pok
        mi, C = mutual_info(cs, ps, ok)
        # transition of the child, conditioned on the parent of bar t+1 (known at the close of t)
        nxt, cur = cs[1:], cs[:-1]
        okt = ok[1:] & cok[:-1]
        cmi, C3 = cond_mutual_info(nxt, cur, ps[1:], okt)
        out[name] = {"mi": mi, "counts": C, "cmi": cmi, "counts3": C3, "n": int(ok.sum())}
    return out


def circular_shift_null(r15: np.ndarray, T: dict, specs: dict, masks: dict, n: int, rng, min_days: int = 60) -> dict:
    """Null 1 (any dependence at all): shift the parent's state series by a random whole number of days
    (>= min_days), keeping both tiers' own persistence and time of day."""
    R = tier_returns_from(r15, T)
    S = {t: specs[t].states(R[t], masks[t]) for t in ("1d", "1h", "15m")}
    hour_of_q, day_of_q = T["hour_row"], T["day_row"]
    day_of_h = np.zeros(T["n_hours"], np.int64); day_of_h[hour_of_q] = day_of_q
    links = {"15m|1d": ("15m", "1d", day_of_q, 1), "1h|1d": ("1h", "1d", day_of_h, 1),
             "15m|1h": ("15m", "1h", hour_of_q, 23)}
    res = {k: {"mi": [], "cmi": []} for k in links}
    for _ in range(n):
        for name, (c, p, rows, per_day) in links.items():
            ps, pok = S[p]
            L = len(ps)
            k = int(rng.integers(min_days, max(min_days + 1, L // per_day - min_days))) * per_day
            ps2, pok2 = np.roll(ps, k), np.roll(pok, k)
            cs, cok = S[c]
            pp, ppok = parent_of(rows, ps2, pok2)
            ok = cok & ppok
            res[name]["mi"].append(mutual_info(cs, pp, ok)[0])
            okt = ok[1:] & cok[:-1]
            res[name]["cmi"].append(cond_mutual_info(cs[1:], cs[:-1], pp[1:], okt)[0])
    return res


def cond_tables(C: np.ndarray) -> pd.DataFrame:
    """P(child state | parent state) from counts[child, parent], long format."""
    P = C / C.sum(0, keepdims=True)
    return pd.DataFrame(P.T, index=[f"parent {s}" for s in STATE_NAMES], columns=STATE_NAMES)


def cond_transitions(C3: np.ndarray) -> dict:
    """P(child_{t+1} | child_t, parent) per parent state, plus pooled over parents."""
    out = {}
    for g, name in enumerate(STATE_NAMES):
        out[f"parent {name}"] = pd.DataFrame(to_probs(C3[g]), index=STATE_NAMES, columns=STATE_NAMES)
    out["pooled"] = pd.DataFrame(to_probs(C3.sum(0)), index=STATE_NAMES, columns=STATE_NAMES)
    return out
