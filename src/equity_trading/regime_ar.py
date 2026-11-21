"""Markov-switching AR(1) on standardized MNQ 5-minute returns (RSW 01; discussions/MNQ_Regime_Switching/001).

This is the regime-switching OU discretized on returns: inside regime s,

    z_t = c_s + phi_s * z_{t-1} + sqrt(s2_s) * eps_t,     z_t = r_t / sigma_hat_t (MRP 001 volatility model)

phi_s < 0 is a reversal regime, phi_s > 0 a continuation regime. The hidden state follows a first-order Markov
chain. In the order-flow models the transition probabilities vary with the previous bar's flow (TVTP, Diebold,
Lee & Weinbach 1994): A_t[i, j] = softmax_j(W[i, j] @ [1, x_{t-1}]), x = (volume surprise, |BVC imbalance|,
imbalance x sign(z)). Flow can move the chain between regimes; it is not an emission, so it cannot pull the states
toward clusters of the flow distribution (which the smoke run showed joint emissions do). Estimation is EM
(Baum-Welch; weighted multinomial logit M-step for W). ``filter_run`` is causal: the filtered P(s_t | F_t) and the
one-step forecast use only bars up to t, with parameters fitted before the segment starts.

Conventions: rows are 5-minute bars in time order. ``obs`` marks bars whose emissions enter the likelihood;
other bars only propagate the chain. ``zl`` is z_{t-1} when bar t-1 is in the same CME day and has a return,
otherwise 0 (then the AR term is off for that bar).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import mnq_predict as P
from .futures import CHICAGO, NY, cme_trade_date

try:
    from numba import njit
except ImportError:                                  # pragma: no cover - slow fallback
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

RSW_VERSION = "0.1.0"
TICK = 0.25
LOG2PI = float(np.log(2 * np.pi))
D = dt.date


@dataclass(frozen=True)
class RSWConfig:
    # (name, number of states, order flow in the transition probabilities); M0 = one-state AR(1), the benchmark
    models: tuple = (("M1", 2, False), ("M2", 3, False), ("M3", 2, True), ("M4", 3, True))
    n_starts: int = 5
    n_starts_tvtp: int = 2            # warm start from the constant-A fit + 1 random start
    max_iter: int = 400
    tol: float = 1e-7                 # relative change of the log-likelihood
    var_floor: float = 1e-3
    seed: int = 0
    bvc_window: int = 60              # 1-minute returns in the ex-ante sigma of BVC
    bvc_min: int = 20
    vol_days: int = 20                # volume baseline: same slot, previous CME days
    vol_min_days: int = 10
    local_bars: int = 276             # z is rescaled by the rms of eligible z over the previous 276 bars (one CME day)
    phi_gap: float = 0.05             # R2
    t_gap: float = 3.0                # R2
    min_stay: float = 2 / 3           # R3: expected duration >= 3 bars
    alpha: float = 0.05
    min_trades: int = 150
    cost_ticks: float = 3.0           # 1 tick spread + 2 x $0.50 commission (MRP 03 convention)
    realized_cost_ticks: float = 1.6  # Iván's realized round trip
    nw_lags: int = 5


# ============================================================================= 1. Inputs

def bvc_imbalance(m1: pd.DataFrame, rc: RSWConfig = RSWConfig()) -> pd.DataFrame:
    """Bulk volume classification (Easley, López de Prado & O'Hara 2012) on 1-minute bars, summed per 5-minute
    CME slot. Buy share of minute m = Phi(r_m / sigma_m), sigma_m = sd of the previous ``bvc_window`` 1-minute
    returns (ex ante). Returns trade_date, slot, imb = sum V(2 Phi - 1) / sum V over minutes with a return."""
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index)
    td = cme_trade_date(pd.Series(ts)).to_numpy()
    slot = P._slot(ts)
    lc = np.log(m1["close"].to_numpy(float))
    iid = m1["instrument_id"].to_numpy()
    same = np.r_[False, (td[1:] == td[:-1]) & (iid[1:] == iid[:-1])]
    r1 = np.where(same, np.r_[np.nan, np.diff(lc)], np.nan)
    sd = pd.Series(r1).rolling(rc.bvc_window, min_periods=rc.bvc_min).std().shift(1).to_numpy()
    ok = np.isfinite(r1) & np.isfinite(sd) & (sd > 0)
    v = m1["volume"].to_numpy(float)
    signed = np.where(ok, v * (2 * stats.norm.cdf(np.where(ok, r1 / np.where(ok, sd, 1), 0)) - 1), 0.0)
    df = pd.DataFrame({"trade_date": td, "slot": slot, "s": signed, "v": np.where(ok, v, 0.0)})
    df = df[df["slot"] < P.BARS_PER_DAY]
    g = df.groupby(["trade_date", "slot"])[["s", "v"]].sum()
    g["imb"] = np.where(g["v"] > 0, g["s"] / g["v"].where(g["v"] > 0, 1), np.nan)
    return g[["imb"]].reset_index()


def volume_surprise(grid: pd.DataFrame, rc: RSWConfig = RSWConfig()) -> np.ndarray:
    """log(1 + volume) minus its mean in the same slot over the previous ``vol_days`` CME days (ex ante)."""
    lv = np.log1p(grid["volume"].to_numpy(float))
    lv[grid["empty"].to_numpy()] = np.nan
    piv = pd.DataFrame({"d": grid["trade_date"].to_numpy(), "s": grid["slot"].to_numpy(), "lv": lv}) \
        .pivot_table(index="d", columns="s", values="lv", aggfunc="first").sort_index()
    base = piv.rolling(rc.vol_days, min_periods=rc.vol_min_days).mean().shift(1)
    b = base.stack(future_stack=True).rename("b")
    key = pd.MultiIndex.from_arrays([grid["trade_date"].to_numpy(), grid["slot"].to_numpy()])
    return lv - b.reindex(key).to_numpy(float)


def build_series(grid: pd.DataFrame, sig2: pd.Series, m1: pd.DataFrame | None, cfg: P.MRPConfig,
                 rc: RSWConfig = RSWConfig()) -> pd.DataFrame:
    """One row per 5-minute bar: z, zl, has_lag, obs, imb, vsurp, sig2, close, session, slot, trade_date.

    z = r / (sigma_hat x L): sigma_hat from the MRP 001 volatility model, L = rms of r / sigma_hat over the eligible bars among
    the previous ``local_bars`` bars (ex ante). L removes the volatility model's level error (MRP 01 gate V2), which would
    otherwise be the easiest thing for a regime to fit. ``sig2`` is the effective ex-ante variance sigma_hat^2 L^2.
    """
    r = grid["ret"].to_numpy(float)
    s2 = sig2.reindex(grid.index).to_numpy(float)
    z0 = r / np.sqrt(s2)
    day = grid["trade_date"].to_numpy()
    elig = np.isfinite(z0) & (grid["slot"].to_numpy() >= cfg.drop_first) & \
        (grid["n_bars_day"].to_numpy() >= cfg.min_day_bars)
    q = pd.Series(np.where(elig, z0 ** 2, np.nan))
    L2 = q.rolling(rc.local_bars, min_periods=rc.local_bars // 2).mean().shift(1).to_numpy()
    s2 = s2 * L2
    z = r / np.sqrt(s2)
    elig &= np.isfinite(z)
    prev_ok = np.r_[False, (day[1:] == day[:-1]) & np.isfinite(z[:-1])]
    zl = np.where(prev_ok, np.r_[np.nan, z[:-1]], 0.0)
    out = pd.DataFrame({"trade_date": day, "slot": grid["slot"].to_numpy(), "session": P.session_of(grid.index),
                        "z": z, "zl": zl, "has_lag": prev_ok, "sig2": s2, "close": grid["close"].to_numpy(float),
                        "ret": r, "eligible": elig}, index=grid.index)
    out["vsurp"] = volume_surprise(grid, rc)
    if m1 is not None:
        b = bvc_imbalance(m1, rc)
        key = pd.MultiIndex.from_arrays([day, grid["slot"].to_numpy()])
        out["imb"] = b.set_index(["trade_date", "slot"])["imb"].reindex(key).to_numpy(float)
    else:
        out["imb"] = np.nan
    # one common mask for every model, so their likelihoods are comparable
    out["obs"] = elig & np.isfinite(out["vsurp"].to_numpy()) & np.isfinite(out["imb"].to_numpy())
    return out


def equations(s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The z equation: y = z_t, design [1, zl_t]."""
    z = np.nan_to_num(s["z"].to_numpy(float))
    return z, np.column_stack([np.ones(len(z)), np.nan_to_num(s["zl"].to_numpy(float))])


FLOW_NAMES = ("vsurp", "abs_imb", "imb_x_sign_z")


def flow_raw(s: pd.DataFrame) -> np.ndarray:
    """(T, 3) transition covariates of row t, taken from bar t-1 (known when the chain moves into bar t):
    volume surprise, |BVC imbalance|, and imbalance x sign(z) (flow agreeing with the price move)."""
    imb = s["imb"].to_numpy(float)
    z = s["z"].to_numpy(float)
    x = np.column_stack([s["vsurp"].to_numpy(float), np.abs(imb), imb * np.sign(z)])
    return np.vstack([np.full((1, 3), np.nan), x[:-1]])


def flow_design(s: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(T, 4) [1, standardized flow]; missing values set to the mean (0)."""
    x = (flow_raw(s) - mu) / sd
    return np.column_stack([np.ones(len(s)), np.nan_to_num(x)])


# ============================================================================= 2. Model

@dataclass
class MSAR:
    K: int
    tvtp: bool                                          # transition probabilities driven by order flow
    pi: np.ndarray = field(repr=False, default=None)
    A: np.ndarray = field(repr=False, default=None)     # constant A, or the mean of A_t over the fit sample
    W: np.ndarray = field(repr=False, default=None)     # (K, K, 4) logits of A_t[i, j] vs staying (W[i, i] = 0)
    flow_mu: np.ndarray = field(repr=False, default=None)
    flow_sd: np.ndarray = field(repr=False, default=None)
    beta: np.ndarray = field(repr=False, default=None)  # (K, 2): c, phi
    s2: np.ndarray = field(repr=False, default=None)    # (K,)
    se: np.ndarray = field(repr=False, default=None)    # (K, 2) approximate se of [c, phi]
    loglik: float = -np.inf
    n_obs: int = 0
    n_iter: int = 0
    converged: bool = False

    @property
    def c(self) -> np.ndarray:
        return self.beta[:, 0]

    @property
    def phi(self) -> np.ndarray:
        return self.beta[:, 1]

    @property
    def n_params(self) -> int:
        trans = self.K * (self.K - 1) * (4 if self.tvtp else 1)
        return trans + (self.K - 1) + 3 * self.K

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(max(self.n_obs, 1))

    def table(self) -> pd.DataFrame:
        rows = []
        for k in range(self.K):
            row = {"state": k, "c": self.c[k], "phi": self.phi[k], "sd": np.sqrt(self.s2[k]),
                   "se_phi": self.se[k, 1], "p_stay": self.A[k, k],
                   "exp_duration_bars": 1 / max(1 - self.A[k, k], 1e-12)}
            if self.tvtp:   # effect of +1 sd of each flow variable on the logit of leaving state k
                leave = [j for j in range(self.K) if j != k]
                for i, n in enumerate(FLOW_NAMES, start=1):
                    row[f"leave_logit_{n}"] = float(np.mean([self.W[k, j, i] for j in leave]))
            rows.append(row)
        return pd.DataFrame(rows).set_index("state")

    def transitions(self, X: np.ndarray) -> np.ndarray:
        """(T, K, K) transition matrices; row t is the move from bar t-1 into bar t."""
        if not self.tvtp:
            return np.broadcast_to(self.A, (len(X), self.K, self.K)).copy()
        eta = np.einsum("tp,ijp->tij", X, self.W)
        eta -= eta.max(2, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(2, keepdims=True)


def _logB(y: np.ndarray, G: np.ndarray, obs: np.ndarray, beta: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """(T, K) log density of z_t in each state; 0 where not observed."""
    res = y[:, None] - G @ beta.T
    lp = -0.5 * (LOG2PI + np.log(s2)[None, :] + res * res / s2[None, :])
    lp[~obs] = 0.0
    return lp


@njit(cache=True)
def _forward(logB, pi, At):
    """Scaled forward pass with time-varying At (T, K, K): filtered alpha, prior P(s_t | F_{t-1}), scaled B,
    scales, loglik."""
    T, K = logB.shape
    alpha = np.empty((T, K))
    prior = np.empty((T, K))
    B = np.empty((T, K))
    cs = np.empty(T)
    ll = 0.0
    for t in range(T):
        m = logB[t, 0]
        for k in range(1, K):
            if logB[t, k] > m:
                m = logB[t, k]
        for k in range(K):
            B[t, k] = np.exp(logB[t, k] - m)
        if t == 0:
            for k in range(K):
                prior[0, k] = pi[k]
        else:
            for j in range(K):
                acc = 0.0
                for i in range(K):
                    acc += alpha[t - 1, i] * At[t, i, j]
                prior[t, j] = acc
        s = 0.0
        for k in range(K):
            alpha[t, k] = prior[t, k] * B[t, k]
            s += alpha[t, k]
        for k in range(K):
            alpha[t, k] /= s
        cs[t] = s
        ll += np.log(s) + m
    return alpha, prior, B, cs, ll


@njit(cache=True)
def _backward(alpha, B, cs, At):
    """Smoothed gamma (T, K) and xi (T, K, K): xi[t, i, j] = P(s_{t-1} = i, s_t = j | all), xi[0] = 0."""
    T, K = alpha.shape
    beta = np.ones((T, K))
    xi = np.zeros((T, K, K))
    tmp = np.empty(K)
    for t in range(T - 2, -1, -1):
        for j in range(K):
            tmp[j] = B[t + 1, j] * beta[t + 1, j]
        for i in range(K):
            acc = 0.0
            for j in range(K):
                acc += At[t + 1, i, j] * tmp[j]
            beta[t, i] = acc / cs[t + 1]
        tot = 0.0
        for i in range(K):
            for j in range(K):
                tot += alpha[t, i] * At[t + 1, i, j] * tmp[j]
        for i in range(K):
            for j in range(K):
                xi[t + 1, i, j] = alpha[t, i] * At[t + 1, i, j] * tmp[j] / tot
    gamma = alpha * beta
    for t in range(T):
        s = 0.0
        for k in range(K):
            s += gamma[t, k]
        for k in range(K):
            gamma[t, k] /= s
    return gamma, xi


def _wls(y: np.ndarray, G: np.ndarray, w: np.ndarray, floor: float) -> tuple[np.ndarray, float, np.ndarray]:
    Gw = G * w[:, None]
    XtX = G.T @ Gw + 1e-9 * np.eye(G.shape[1])
    b = np.linalg.solve(XtX, Gw.T @ y)
    r = y - G @ b
    v = max(float((w * r * r).sum() / max(w.sum(), 1e-12)), floor)
    return b, v, np.sqrt(np.diag(v * np.linalg.inv(XtX)))


def _mstep_W(W: np.ndarray, xi: np.ndarray, X: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Weighted multinomial logit per origin state i (reference = stay): maximize
    sum_t sum_j xi[t,i,j] log A_t[i,j] - ridge/2 ||slopes||^2 (a N(0, 1) prior on standardized slopes: negligible
    at our sample sizes, it only keeps W finite when a state is nearly empty)."""
    from scipy.optimize import minimize
    K, p = W.shape[0], X.shape[1]
    W = W.copy()
    for i in range(K):
        cols = [j for j in range(K) if j != i]
        q = xi[:, i, :]                                         # (T, K)
        w = q.sum(1)
        n = max(w.sum(), 1.0)

        def f(theta):
            Wi = np.zeros((K, p))
            Wi[cols] = theta.reshape(len(cols), p)
            eta = X @ Wi.T
            mx = eta.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(eta - mx).sum(1))
            val = -((q * eta).sum() - (w * lse).sum()) / n
            val += 0.5 * ridge * (Wi[cols][:, 1:] ** 2).sum() / n
            Pm = np.exp(eta - lse[:, None])
            g = -((q - w[:, None] * Pm).T @ X)[cols] / n
            g[:, 1:] += ridge * Wi[cols][:, 1:] / n
            return val, g.ravel()

        r = minimize(f, W[i, cols].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 50})
        W[i, cols] = r.x.reshape(len(cols), p)
    return W


def _init(y, G, obs, K, tvtp, rng, floor):
    vz = float(np.var(y[obs]))
    phi0 = np.sort(rng.uniform(-0.3, 0.3, K)) if K > 1 else np.array([0.0])
    beta = np.column_stack([rng.normal(0, 0.02, K), phi0])
    s2 = np.maximum(vz * rng.uniform(0.7, 1.3, K), floor)
    A = np.full((K, K), 0.05 / max(K - 1, 1)) + np.eye(K) * (0.95 - 0.05 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
    W = np.zeros((K, K, 4))
    if tvtp and K > 1:
        for i in range(K):
            for j in range(K):
                if j != i:
                    W[i, j, 0] = np.log(A[i, j] / A[i, i])
    return np.full(K, 1.0 / K), A, W, beta, s2


def _em(y, G, X, obs, K, tvtp, rng, rc: RSWConfig, warm: MSAR | None = None) -> MSAR:
    pi, A, W, beta, s2 = _init(y, G, obs, K, tvtp, rng, rc.var_floor)
    if warm is not None:                                # start from a constant-A fit with the same K
        pi, A, beta, s2 = warm.pi.copy(), warm.A.copy(), warm.beta.copy(), warm.s2.copy()
        W = np.zeros((K, K, 4))
        for i in range(K):
            for j in range(K):
                if j != i:
                    W[i, j, 0] = np.log(max(A[i, j], 1e-6) / A[i, i])
    m = MSAR(K, tvtp and K > 1, pi, A, W, beta=beta, s2=s2)
    prev, ll, it, conv = -np.inf, -np.inf, 0, False
    se = np.zeros((K, 2))
    for it in range(1, rc.max_iter + 1):
        At = m.transitions(X)
        alpha, _, B, cs, ll = _forward(_logB(y, G, obs, m.beta, m.s2), m.pi, At)
        gamma, xi = _backward(alpha, B, cs, At)
        m.pi = (gamma[0] + 1e-12) / (gamma[0] + 1e-12).sum()
        if m.tvtp:
            m.W = _mstep_W(m.W, xi[1:], X[1:])
        else:
            a = xi[1:].sum(0) + 1e-12
            m.A = a / a.sum(1, keepdims=True)
        g = gamma[obs]
        for k in range(K):
            b, v, s_ = _wls(y[obs], G[obs], g[:, k], rc.var_floor)
            m.beta[k], m.s2[k], se[k] = b, v, s_
        if np.isfinite(prev) and abs(ll - prev) < rc.tol * abs(ll):
            conv = True
            break
        prev = ll
    if m.tvtp:
        m.A = m.transitions(X).mean(0)
    m.loglik, m.n_obs, m.n_iter, m.converged, m.se = ll, int(obs.sum()), it, conv, se
    return _sort_states(m)


def _sort_states(m: MSAR) -> MSAR:
    o = np.argsort(m.phi)
    m.pi, m.A, m.W = m.pi[o], m.A[np.ix_(o, o)], m.W[np.ix_(o, o)]
    m.beta, m.s2, m.se = m.beta[o], m.s2[o], m.se[o]
    if m.tvtp:                                          # re-reference: logits relative to staying in the new order
        for i in range(m.K):
            m.W[i] = m.W[i] - m.W[i, i][None, :]
    return m


def _flow_stats(s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = flow_raw(s)[s["obs"].to_numpy(bool)]
    mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def fit(s: pd.DataFrame, K: int, tvtp: bool, mask: np.ndarray, rc: RSWConfig = RSWConfig()) -> MSAR:
    """EM fit on rows ``mask`` (a contiguous time window), best of ``n_starts`` random starts. A TVTP model is
    started from the best constant-A fit of the same K, plus ``n_starts_tvtp - 1`` random starts."""
    sub = s[mask]
    y, G = equations(sub)
    mu, sd = _flow_stats(sub)
    X = flow_design(sub, mu, sd)
    obs = sub["obs"].to_numpy(bool)
    rng = np.random.default_rng(rc.seed + 1000 * K + 100 * int(tvtp))
    best = None
    if tvtp and K > 1:
        warm = fit(s, K, False, mask, rc)
        cands = [_em(y, G, X, obs, K, True, rng, rc, warm)] + \
            [_em(y, G, X, obs, K, True, rng, rc) for _ in range(rc.n_starts_tvtp - 1)]
    else:
        cands = [_em(y, G, X, obs, K, False, rng, rc) for _ in range(rc.n_starts if K > 1 else 1)]
    for m in cands:
        if best is None or m.loglik > best.loglik:
            best = m
    best.flow_mu, best.flow_sd = mu, sd
    return best


def _pass(s: pd.DataFrame, m: MSAR):
    y, G = equations(s)
    obs = s["obs"].to_numpy(bool)
    At = m.transitions(flow_design(s, m.flow_mu, m.flow_sd))
    lB = _logB(y, G, obs, m.beta, m.s2)
    return y, G, obs, At, lB


def filter_run(s: pd.DataFrame, m: MSAR) -> pd.DataFrame:
    """Causal pass over all rows of ``s`` with fixed parameters.

    Columns: p0..p{K-1} = P(s_t | F_t); fc = E[z_t | F_{t-1}] (the one-step forecast of bar t made at t-1);
    lpz = log predictive density of z_t given F_{t-1}."""
    y, G, obs, At, lB = _pass(s, m)
    alpha, prior, _, _, _ = _forward(lB, m.pi, At)
    fc = (prior * (G @ m.beta.T)).sum(1)
    dens = (prior * np.exp(np.where(obs[:, None], lB, -np.inf))).sum(1)
    out = pd.DataFrame(alpha, index=s.index, columns=[f"p{k}" for k in range(m.K)])
    out["fc"] = fc
    out["lpz"] = np.where(obs, np.log(np.maximum(dens, 1e-300)), np.nan)
    return out


def smooth(s: pd.DataFrame, m: MSAR) -> np.ndarray:
    """Smoothed P(s_t | all data). Look-ahead: description only."""
    y, G, obs, At, lB = _pass(s, m)
    alpha, _, B, cs, _ = _forward(lB, m.pi, At)
    return _backward(alpha, B, cs, At)[0]


# ============================================================================= 3. Walk-forward

def walk_forward(s: pd.DataFrame, fit_start: dt.date, refits: tuple, end: dt.date, K: int, tvtp: bool,
                 rc: RSWConfig = RSWConfig()) -> tuple[pd.DataFrame, list[MSAR]]:
    """Expanding fits on [fit_start, refit) and causal filtering of [refit, next refit); end inclusive."""
    td = s["trade_date"].to_numpy()
    edges = list(refits) + [end + dt.timedelta(days=1)]
    parts, fits = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = fit(s, K, tvtp, (td >= fit_start) & (td < a), rc)
        fits.append(m)
        f = filter_run(s[td >= fit_start], m)                 # filter history from fit_start, fixed params
        ftd = s.loc[f.index, "trade_date"].to_numpy()
        f = f[(ftd >= a) & (ftd < b)]
        for k in range(m.K, 3):                               # pad so every model has p0..p2
            f[f"p{k}"] = np.nan
        parts.append(f)
    return pd.concat(parts).sort_index(), fits


# ============================================================================= 4. Gates

def _nw_t(x: np.ndarray, lags: int) -> tuple[float, float]:
    m, se, t = P.nw_mean_t(x, lags)
    return m, t


def predictive_dm(lp_model: np.ndarray, lp_bench: np.ndarray, day: np.ndarray, lags: int = 5) -> dict:
    """R1: daily mean log-score difference (model - benchmark), Newey-West t, one-sided p."""
    ok = np.isfinite(lp_model) & np.isfinite(lp_bench)
    d = pd.Series(lp_model[ok] - lp_bench[ok]).groupby(pd.Series(day[ok])).mean().to_numpy()
    m, t = _nw_t(d, lags)
    return {"mean_daily_diff": m, "t": t, "p": float(stats.norm.sf(t)), "days": len(d),
            "per_bar_diff": float(np.mean(lp_model[ok] - lp_bench[ok]))}


def phi_gap(m: MSAR) -> dict:
    """R2 on one fit: gap between the most reverting and the most continuing state."""
    lo, hi = 0, m.K - 1
    gap = float(m.phi[hi] - m.phi[lo])
    t = gap / float(np.hypot(m.se[hi, 1], m.se[lo, 1]))
    return {"phi_min": float(m.phi[lo]), "phi_max": float(m.phi[hi]), "gap": gap, "t_gap": t,
            "stay_min_state": float(m.A[lo, lo]), "stay_max_state": float(m.A[hi, hi])}


def forecast_scores(z: np.ndarray, fc: np.ndarray, fc_bench: np.ndarray, day: np.ndarray, lags: int = 5) -> dict:
    """R4: R2_OOS vs zero and vs the benchmark forecast; Clark-West vs the benchmark."""
    ok = np.isfinite(z) & np.isfinite(fc) & np.isfinite(fc_bench)
    y, f, b = z[ok], fc[ok], fc_bench[ok]
    r2_zero = 1 - np.sum((y - f) ** 2) / np.sum(y ** 2)
    r2_bench = 1 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2)
    adj = (y - b) ** 2 - ((y - f) ** 2 - (b - f) ** 2)
    d = pd.Series(adj).groupby(pd.Series(day[ok])).mean().to_numpy()
    m, t = _nw_t(d, lags)
    return {"r2_oos_vs_zero": float(r2_zero), "r2_oos_vs_M0": float(r2_bench), "cw_t": t,
            "cw_p": float(stats.norm.sf(t)), "n": int(ok.sum()),
            "hit_rate": float(np.mean(np.sign(f) == np.sign(y)))}


def one_bar_trades(s: pd.DataFrame, fc: np.ndarray, threshold_ticks: float) -> pd.DataFrame:
    """Trade bar t (entered at the close of t-1, exited at the close of t) in the sign of the forecast when the
    forecast move |fc| x sigma_hat x price is at least ``threshold_ticks``. Gross ticks per contract."""
    close = s["close"].to_numpy(float)
    day = s["trade_date"].to_numpy()
    prev_close = np.r_[np.nan, close[:-1]]
    same = np.r_[False, day[1:] == day[:-1]]
    ft = fc * np.sqrt(s["sig2"].to_numpy(float)) * prev_close / TICK
    ok = s["obs"].to_numpy(bool) & same & np.isfinite(ft) & np.isfinite(s["ret"].to_numpy(float)) & \
        (np.abs(ft) >= threshold_ticks) & (np.abs(ft) > 0)
    side = np.sign(ft[ok])
    gross = side * (close[ok] - prev_close[ok]) / TICK
    return pd.DataFrame({"trade_date": day[ok], "session": s["session"].to_numpy()[ok], "side": side,
                         "fc_ticks": ft[ok], "gross_ticks": gross}, index=s.index[ok])


def trade_stats(tr: pd.DataFrame, cost: float, periods: dict) -> dict:
    from .reversal import day_cluster_t
    net = tr["gross_ticks"].to_numpy(float) - cost
    if len(net) < 3:
        return {"trades": len(net), "gross": np.nan, "net": np.nan, "t": np.nan, "p": np.nan}
    m, t = day_cluster_t(net, tr["trade_date"].to_numpy())
    out = {"trades": len(net), "gross": float(tr["gross_ticks"].mean()), "net": m, "t": t,
           "p": float(stats.norm.sf(t)), "win_rate": float(np.mean(net > 0)),
           "trades_per_day": len(net) / max(tr["trade_date"].nunique(), 1),
           "net_usd_total_1c": float(net.sum() * TICK * 2)}
    d = tr["trade_date"].to_numpy()
    for p, (a, b) in periods.items():
        sel = (d >= a) & (d <= b)
        out[f"net_{p}"] = float(net[sel].mean()) if sel.any() else np.nan
    return out


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    n, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (n - i) * p[k]))
        out[k] = run
    return out


# ============================================================================= 5. Synthetic data (tests, smoke run)

def simulate_msar(T: int, phis=(-0.25, 0.25), p_stay: float = 0.98, sd: float = 1.0, seed: int = 0):
    """z path and states of a 2-or-more-state MS-AR(1) with zero intercepts."""
    rng = np.random.default_rng(seed)
    K = len(phis)
    A = np.full((K, K), (1 - p_stay) / max(K - 1, 1))
    np.fill_diagonal(A, p_stay)
    st = np.empty(T, dtype=np.int64)
    z = np.empty(T)
    st[0], z[0] = 0, rng.normal(0, sd)
    for t in range(1, T):
        st[t] = rng.choice(K, p=A[st[t - 1]])
        z[t] = phis[st[t]] * z[t - 1] + rng.normal(0, sd)
    return z, st


def write_synthetic_regimes(base, start: dt.date, end: dt.date, cfg: P.MRPConfig, seed: int = 11,
                            phis=(-0.3, 0.3), p_stay: float = 0.98, flow_effect: float = 1.5) -> None:
    """MNQ ohlcv-1m month files under ``base`` whose 5-minute returns follow a planted 2-state MS-AR(1) in z:
    per-bucket sd from a daily log-vol AR(1) x U-shaped intraday profile; bucket return = z_b x bucket sd, spread
    over its five minutes with extra intra-bar noise; quarterly contract switch; 1% of minutes missing.
    Volume: each bucket has a log-volume shock u_b ~ N(0, 0.5); the logit of leaving the current regime at bucket b
    is logit(1 - p_stay) + flow_effect x u_{b-1} / 0.5, so the previous bar's volume surprise drives switching."""
    import os
    from pathlib import Path
    from . import micro_bars as MB
    base = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base)
    rng = np.random.default_rng(seed)
    K = len(phis)
    a0 = np.log((1 - p_stay) / p_stay)
    price, logvol, iid, state, u_prev = 9000.0, 0.0, 1, 0, 0.0
    rows = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            open_ = pd.Timestamp(d - dt.timedelta(days=1), tz=CHICAGO) + pd.Timedelta(hours=17)
            ts = (open_ + pd.to_timedelta(np.arange(23 * 60), unit="min")).tz_convert("UTC")
            if d.day == 10 and d.month in (3, 6, 9, 12):
                iid += 1
                price += 50.0
            logvol = 0.95 * logvol + 0.25 * rng.normal()
            loc = ts.tz_convert(NY)
            mins = np.asarray(loc.hour * 60 + loc.minute, float)
            prof = 1.0 + 2.5 * np.exp(-((mins - 575) / 25.0) ** 2) + 1.2 * np.exp(-((mins - 955) / 20.0) ** 2)
            sd = (2.0e-4 * np.exp(logvol) * prof).reshape(-1, 5)
            bsd = np.sqrt((sd ** 2).sum(1))
            nb = len(bsd)
            zb = np.empty(nb)
            ub = rng.normal(0, 0.5, nb)
            zprev = 0.0
            for b in range(nb):
                p_leave = 1 / (1 + np.exp(-(a0 + flow_effect * u_prev / 0.5)))
                if rng.random() < p_leave:
                    state = (state + 1 + rng.integers(K - 1)) % K
                zb[b] = (phis[state] * zprev if b else 0.0) + rng.normal()
                zprev, u_prev = zb[b], ub[b]
            e = rng.normal(0, 1, (nb, 5)) * sd
            e = e - e.mean(1, keepdims=True)                   # intra-bar noise, zero sum per bucket
            r = (e + (zb * bsd)[:, None] / 5).ravel()
            close = np.exp(np.log(price) + np.cumsum(r))
            price = close[-1]
            keep = rng.random(len(ts)) > 0.01
            rows.append(pd.DataFrame({"ts": ts[keep], "open": close[keep], "high": close[keep] + 0.25,
                                      "low": close[keep] - 0.25, "close": close[keep],
                                      "volume": rng.poisson(30 * np.repeat(np.exp(ub), 5)[keep]) + 1,
                                      "instrument_id": iid}))
        d += dt.timedelta(days=1)
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for key, part in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        a = dt.date(int(key[:4]), int(key[5:]), 1)
        f = MB.month_path(cfg.root, a)
        f.parent.mkdir(parents=True, exist_ok=True)
        part[MB.COLUMNS].reset_index(drop=True).to_parquet(f, index=False)
