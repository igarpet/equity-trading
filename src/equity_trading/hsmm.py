"""Duration dependence in MNQ 5-minute regimes (RSW 03; discussions/MNQ_Regime_Switching/005).

A. Model-free hazard: h(n) = P(e_t = 1 | n rows since the last exceedance), e = |z| >= u, against the hazard of the
   binary chain (flat after n = 1) and of the RSW 01 HMMs (simulated).
B. Explicit-duration hidden semi-Markov model (HSMM) with the RSW 01 AR(1) regimes:
       z_t = c_s + phi_s z_{t-1} + b_s log(d_t) + sigma_s eps_t,     d_t = age of the current regime (1, 2, ...)
   (b_s = 0 in part B). Each regime's duration has its own distribution: discrete Weibull
   h(d) = 1 - exp(-lam (d^beta - (d-1)^beta)) (beta < 1 falling hazard, > 1 rising, = 1 geometric = HMM) or a shifted
   negative binomial (r = 1 geometric). On leaving s the chain moves to j != s with probability W[s, j].
C. The same with the age drift b_s free: direction by age.

The HSMM is run as an HMM on (state, age) pairs with ages 1..D; age D is a tail state with the constant hazard
h(D) (Langrock & Zucchini 2011), exact for durations up to D - 1. Estimation is EM: the forward-backward pass gives
expected end / continue counts per (state, age) (weighted binomial hazard M-step), switch flows (W) and weighted
least squares sufficient statistics (emissions), accumulated on the fly so only alpha (T, K, D) is stored.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import hawkes as H
from . import regime_ar as R

try:
    from numba import njit
except ImportError:                                  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

HSMM_VERSION = "0.1.0"
EPS = 1e-12


@dataclass(frozen=True)
class HSMMConfig:
    D: int = 60                          # ages 1..D, D = tail with constant hazard
    families: tuple = ("WB", "NB")
    max_iter: int = 300
    start_shapes: tuple = (1.0,)          # warm start from the HMM fit of the same K
    tol: float = 1e-7
    var_floor: float = 1e-3
    q: float = 0.96                      # exceedance threshold quantile (as RSW 02)
    # part A
    age_edges: tuple = tuple(range(1, 13)) + (13, 16, 20, 25, 32, 40, 50, 63, 80, 100, 126, 160, 200, 252, 318, 400, 481)
    slope_range: tuple = (13, 480)
    min_bin_obs: int = 200
    hmm_sim_bars: int = 2_000_000
    n_boot: int = 2000
    block_days: int = 5
    seed: int = 0
    # gates
    alpha: float = 0.05
    t_dur: float = 3.0                   # HB3 / HC1: |t| of the duration-shape / age-drift parameter
    nw_lags: int = 5


# ============================================================================= A. Model-free hazard

def ages(e: np.ndarray) -> np.ndarray:
    """Rows since the last event strictly before t (1 = an event at t-1); 0 before the first event."""
    t = np.arange(len(e))
    last = np.maximum.accumulate(np.where(e > 0, t, -1))
    before = np.r_[-1, last[:-1]]
    return np.where(before >= 0, t - before, 0)


def hazard_counts(e: np.ndarray, obs: np.ndarray, day_id: np.ndarray, edges: tuple) -> dict:
    """Per day and age bin [edges[i], edges[i+1]): events and observed rows (age >= 1 only)."""
    a = ages(e)
    b = np.searchsorted(np.asarray(edges), a, side="right") - 1
    ok = obs & (a >= edges[0]) & (a < edges[-1])
    nb, nd = len(edges) - 1, int(day_id.max()) + 1
    key = day_id[ok] * nb + b[ok]
    num = np.bincount(key, e[ok], minlength=nd * nb).reshape(nd, nb)
    den = np.bincount(key, minlength=nd * nb).reshape(nd, nb).astype(float)
    return {"num": num, "den": den, "edges": np.asarray(edges)}


def hazard_curve(cnt: dict, w: np.ndarray | None = None) -> pd.DataFrame:
    w = np.ones(cnt["num"].shape[0]) if w is None else w
    num, den = w @ cnt["num"], w @ cnt["den"]
    e = cnt["edges"]
    mid = np.sqrt(e[:-1] * np.maximum(e[1:] - 1, e[:-1]))
    return pd.DataFrame({"lo": e[:-1], "hi": e[1:] - 1, "mid": mid, "events": num, "rows": den,
                         "h": num / np.maximum(den, EPS)})


def hazard_slope(hc: pd.DataFrame, rng_: tuple, min_obs: int) -> float:
    """Weighted LS slope of log h on log(age) over bins inside ``rng_`` with enough rows (and h > 0)."""
    m = (hc["lo"] >= rng_[0]) & (hc["hi"] <= rng_[1]) & (hc["rows"] >= min_obs) & (hc["h"] > 0)
    if m.sum() < 3:
        return float("nan")
    x, y, w = np.log(hc.loc[m, "mid"]), np.log(hc.loc[m, "h"]), hc.loc[m, "rows"]
    X = np.column_stack([np.ones(m.sum()), x])
    b = np.linalg.solve(X.T @ (X * w.to_numpy()[:, None]), X.T @ (w.to_numpy() * y))
    return float(b[1])


def null_hazard(m: R.MSAR, u: float, rc: HSMMConfig) -> pd.DataFrame:
    w, v = np.linalg.eig(m.A.T)
    pi0 = np.real(v[:, np.argmin(np.abs(w - 1))])
    e = H._simulate_msar(rc.hmm_sim_bars, pi0 / pi0.sum(), m.A, m.c.astype(float), m.phi.astype(float),
                         np.sqrt(m.s2), u, rc.seed)
    return hazard_curve(hazard_counts(e, np.ones(len(e), bool), np.zeros(len(e), np.int64), rc.age_edges))


def hazard_test(s: pd.DataFrame, u: float, nulls: dict, rc: HSMMConfig = HSMMConfig()) -> dict:
    """HA: slope of the observed hazard over ``slope_range`` minus each null's slope; day-block bootstrap of the
    observed curve (nulls fixed; the chain's slope is 0 by construction). HA = every lower bound > 0 (rising)."""
    e, _ = H.events(s, u)
    obs = s["obs"].to_numpy(bool)
    day_id = pd.factorize(s["trade_date"])[0].astype(np.int64)
    cnt = hazard_counts(e, obs, day_id, rc.age_edges)
    hc = hazard_curve(cnt)
    slope = hazard_slope(hc, rc.slope_range, rc.min_bin_obs)
    ns = {"chain": 0.0, **{k: hazard_slope(v, rc.slope_range, rc.min_bin_obs) for k, v in nulls.items()}}
    rng = np.random.default_rng(rc.seed)
    nd = cnt["num"].shape[0]
    boot = np.array([hazard_slope(hazard_curve(cnt, H.block_weights(nd, rc.block_days, rng)), rc.slope_range,
                                  rc.min_bin_obs) for _ in range(rc.n_boot)])
    lo, hi = np.nanquantile(boot, [0.05, 0.95])
    out = {"u": u, "slope": slope, "slope_lo95": float(lo), "slope_hi95": float(hi), "curve": hc,
           "null_curves": nulls, "null_slopes": ns}
    out["HA_rising"] = bool(all(lo - v > 0 for v in ns.values()))
    out["falling_vs_all"] = bool(all(hi - v < 0 for v in ns.values()))
    return out


# ============================================================================= B/C. HSMM

def wb_hazard(lam: float, beta: float, D: int) -> np.ndarray:
    d = np.arange(1, D + 1, dtype=float)
    return np.clip(1 - np.exp(-lam * (d ** beta - (d - 1) ** beta)), 1e-9, 1 - 1e-9)


def nb_hazard(r: float, p: float, D: int) -> np.ndarray:
    k = np.arange(D)
    pmf = stats.nbinom.pmf(k, r, p)
    sf_incl = stats.nbinom.sf(k - 1, r, p)
    return np.clip(pmf / np.maximum(sf_incl, 1e-300), 1e-9, 1 - 1e-9)


def family_hazard(fam: str, th: np.ndarray, D: int) -> np.ndarray:
    if fam == "WB":
        return wb_hazard(np.exp(th[0]), np.exp(th[1]), D)
    if fam == "NB":
        return nb_hazard(np.exp(th[0]), 1 / (1 + np.exp(-th[1])), D)
    if fam == "GEO":
        return np.full(D, np.clip(1 / (1 + np.exp(-th[0])), 1e-9, 1 - 1e-9))
    raise ValueError(fam)


SHAPE_INDEX = {"WB": 1, "NB": 0}       # log beta (Weibull), log r (negative binomial)


def shape_param(fam: str, th: np.ndarray) -> float:
    """1 = geometric (memoryless): Weibull beta, NB r. Above 1: the hazard rises with age; below 1: it falls."""
    return float(np.exp(th[SHAPE_INDEX[fam]])) if fam in SHAPE_INDEX else 1.0


@njit(cache=True)
def _emis(z, zl, ld, c, phi, b, s2, t, s, d):
    m = c[s] + phi[s] * zl[t] + b[s] * ld[d]
    r = z[t] - m
    return -0.5 * (1.8378770664093453 + math.log(s2[s]) + r * r / s2[s])


@njit(cache=True)
def _forward(z, zl, obs, ld, c, phi, b, s2, pi, Hz, W):
    T = len(z)
    K, D = Hz.shape
    alpha = np.empty((T, K, D))
    prior = np.zeros((K, D))
    lb = np.zeros((K, D))
    mt = np.empty(T)
    cs = np.empty(T)
    ll = 0.0
    for t in range(T):
        prior[:, :] = 0.0
        if t == 0:
            for k in range(K):
                prior[k, 0] = pi[k]
        else:
            for s in range(K):
                out_s = 0.0
                for d in range(D):
                    a = alpha[t - 1, s, d]
                    h = Hz[s, d]
                    out_s += a * h
                    if d < D - 1:
                        prior[s, d + 1] += a * (1 - h)
                    else:
                        prior[s, d] += a * (1 - h)
                for j in range(K):
                    prior[j, 0] += out_s * W[s, j]
        m = 0.0
        if obs[t]:
            m = -1e300
            for s in range(K):
                if b[s] == 0.0:
                    v0 = _emis(z, zl, ld, c, phi, b, s2, t, s, 0)
                    for d in range(D):
                        lb[s, d] = v0
                    if v0 > m:
                        m = v0
                else:
                    for d in range(D):
                        lb[s, d] = _emis(z, zl, ld, c, phi, b, s2, t, s, d)
                        if lb[s, d] > m:
                            m = lb[s, d]
            for s in range(K):
                if b[s] == 0.0:
                    v0 = math.exp(lb[s, 0] - m)
                    for d in range(D):
                        lb[s, d] = v0
                else:
                    for d in range(D):
                        lb[s, d] = math.exp(lb[s, d] - m)
        tot = 0.0
        for s in range(K):
            for d in range(D):
                v = prior[s, d] * (lb[s, d] if obs[t] else 1.0)
                alpha[t, s, d] = v
                tot += v
        for s in range(K):
            for d in range(D):
                alpha[t, s, d] /= tot
        mt[t] = m
        cs[t] = tot
        ll += math.log(tot) + m
    return alpha, mt, cs, ll


@njit(cache=True)
def _backward(alpha, mt, cs, z, zl, obs, ld, c, phi, b, s2, Hz, W, P):
    """Expected end / continue counts (K, D), switch flows (K, K), gamma at t = 0 (K, D) and emission sufficient
    statistics per state: XtWX (K, P, P), XtWy (K, P), yWy (K), wsum (K), with X = [1, zl, log d][:P]."""
    T, K, D = alpha.shape
    beta = np.ones((K, D))
    nb = np.empty((K, D))
    Bn = np.empty((K, D))
    end = np.zeros((K, D))
    cont = np.zeros((K, D))
    flow = np.zeros((K, K))
    XtWX = np.zeros((K, P, P))
    XtWy = np.zeros((K, P))
    yWy = np.zeros(K)
    wsum = np.zeros(K)
    x = np.empty(3)
    for t in range(T - 1, -1, -1):
        # gamma_t = alpha_t * beta_t: emission statistics
        if obs[t]:
            g = 0.0
            for s in range(K):
                for d in range(D):
                    g += alpha[t, s, d] * beta[s, d]
            for s in range(K):
                for d in range(D):
                    w = alpha[t, s, d] * beta[s, d] / g
                    if w < 1e-300:
                        continue
                    x[0] = 1.0
                    x[1] = zl[t]
                    x[2] = ld[d]
                    for i in range(P):
                        XtWy[s, i] += w * x[i] * z[t]
                        for j in range(P):
                            XtWX[s, i, j] += w * x[i] * x[j]
                    yWy[s] += w * z[t] * z[t]
                    wsum[s] += w
        if t == 0:
            break
        # B_t (scaled) for the transition t-1 -> t
        for s in range(K):
            if not obs[t]:
                for d in range(D):
                    Bn[s, d] = 1.0
            elif b[s] == 0.0:
                v0 = math.exp(_emis(z, zl, ld, c, phi, b, s2, t, s, 0) - mt[t])
                for d in range(D):
                    Bn[s, d] = v0
            else:
                for d in range(D):
                    Bn[s, d] = math.exp(_emis(z, zl, ld, c, phi, b, s2, t, s, d) - mt[t])
        # beta_{t-1} and counts
        for s in range(K):
            sw = 0.0
            for j in range(K):
                sw += W[s, j] * Bn[j, 0] * beta[j, 0]
            for d in range(D):
                h = Hz[s, d]
                dn = d + 1 if d < D - 1 else d
                cv = (1 - h) * Bn[s, dn] * beta[s, dn]
                ev = h * sw
                nb[s, d] = (cv + ev) / cs[t]
                a = alpha[t - 1, s, d]
                cont[s, d] += a * cv / cs[t]
                end[s, d] += a * ev / cs[t]
                for j in range(K):
                    flow[s, j] += a * h * W[s, j] * Bn[j, 0] * beta[j, 0] / cs[t]
        for s in range(K):
            for d in range(D):
                beta[s, d] = nb[s, d]
    g0 = alpha[0] * beta
    g0 = g0 / g0.sum()
    return end, cont, flow, g0, XtWX, XtWy, yWy, wsum


@njit(cache=True)
def _filter(z, zl, obs, ld, c, phi, b, s2, pi, Hz, W, u):
    """Causal pass: filtered P(s_t | F_t) (T, K), mean filtered age, one-step forecast E[z_t | F_{t-1}], log
    predictive density of z_t, P(|z_t| >= u | F_{t-1})."""
    T = len(z)
    K, D = Hz.shape
    a_prev = np.zeros((K, D))
    prior = np.zeros((K, D))
    lb = np.zeros((K, D))
    filt = np.empty((T, K))
    age = np.empty(T)
    fc = np.empty(T)
    lpz = np.full(T, np.nan)
    pex = np.empty(T)
    sq2 = math.sqrt(2.0)
    for t in range(T):
        prior[:, :] = 0.0
        if t == 0:
            for k in range(K):
                prior[k, 0] = pi[k]
        else:
            for s in range(K):
                out_s = 0.0
                for d in range(D):
                    a = a_prev[s, d]
                    h = Hz[s, d]
                    out_s += a * h
                    if d < D - 1:
                        prior[s, d + 1] += a * (1 - h)
                    else:
                        prior[s, d] += a * (1 - h)
                for j in range(K):
                    prior[j, 0] += out_s * W[s, j]
        f = 0.0
        dens = 0.0
        pe = 0.0
        for s in range(K):
            sd = math.sqrt(s2[s])
            for d in range(D):
                m = c[s] + phi[s] * zl[t] + b[s] * ld[d]
                f += prior[s, d] * m
                pe += prior[s, d] * (0.5 * math.erfc((u - m) / (sd * sq2)) + 0.5 * math.erfc((u + m) / (sd * sq2)))
        fc[t] = f
        pex[t] = pe
        tot = 0.0
        if obs[t]:
            mx = -1e300
            for s in range(K):
                for d in range(D):
                    v = _emis(z, zl, ld, c, phi, b, s2, t, s, d) if (b[s] != 0.0 or d == 0) else lb[s, 0]
                    lb[s, d] = v
                    if v > mx:
                        mx = v
            for s in range(K):
                e0 = math.exp(lb[s, 0] - mx)
                for d in range(D):
                    v = prior[s, d] * (math.exp(lb[s, d] - mx) if b[s] != 0.0 else e0)
                    a_prev[s, d] = v
                    tot += v
            lpz[t] = math.log(tot) + mx
        else:
            for s in range(K):
                for d in range(D):
                    a_prev[s, d] = prior[s, d]
                    tot += prior[s, d]
        for s in range(K):
            acc = 0.0
            for d in range(D):
                a_prev[s, d] /= tot
                acc += a_prev[s, d]
            filt[t, s] = acc
        ag = 0.0
        for s in range(K):
            for d in range(D):
                ag += a_prev[s, d] * (d + 1)
        age[t] = ag
    return filt, age, fc, lpz, pex


@dataclass
class HSMM:
    K: int
    fam: str
    age_drift: bool
    D: int
    pi: np.ndarray = field(repr=False, default=None)
    th: np.ndarray = field(repr=False, default=None)      # (K, 2) duration parameters
    th_se: np.ndarray = field(repr=False, default=None)   # (K, 2) approximate se (Hessian of the hazard M-step)
    W: np.ndarray = field(repr=False, default=None)
    c: np.ndarray = field(repr=False, default=None)
    phi: np.ndarray = field(repr=False, default=None)
    b: np.ndarray = field(repr=False, default=None)
    s2: np.ndarray = field(repr=False, default=None)
    b_se: np.ndarray = field(repr=False, default=None)
    loglik: float = -np.inf
    n_obs: int = 0
    n_iter: int = 0
    converged: bool = False

    def hazards(self) -> np.ndarray:
        return np.vstack([family_hazard(self.fam, self.th[k], self.D) for k in range(self.K)])

    def mean_duration(self, k: int, dmax: int = 5000) -> float:
        h = family_hazard(self.fam, self.th[k], self.D)
        surv, m = 1.0, 0.0
        for d in range(1, dmax + 1):
            hd = h[min(d, self.D) - 1]
            m += d * surv * hd
            surv *= 1 - hd
        return float(m + dmax * surv)

    def shape_t(self, k: int) -> float:
        """t-statistic of log(shape) = 0 (geometric), from the hazard M-step Hessian."""
        if self.fam not in SHAPE_INDEX or self.th_se is None:
            return float("nan")
        i = SHAPE_INDEX[self.fam]
        return float(self.th[k, i] / self.th_se[k, i])

    def table(self) -> pd.DataFrame:
        rows = []
        for k in range(self.K):
            rows.append({"state": k, "c": self.c[k], "phi": self.phi[k], "sd": np.sqrt(self.s2[k]),
                         "b_age": self.b[k], "t_b_age": self.b[k] / self.b_se[k] if self.age_drift else np.nan,
                         "shape": shape_param(self.fam, self.th[k]), "t_shape_vs_1": self.shape_t(k),
                         "mean_duration": self.mean_duration(k), "h1": family_hazard(self.fam, self.th[k], self.D)[0],
                         "h10": family_hazard(self.fam, self.th[k], self.D)[9]})
        return pd.DataFrame(rows).set_index("state")


def _hazard_mstep(fam: str, th0: np.ndarray, end: np.ndarray, cont: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    def nll(th):
        h = family_hazard(fam, th, D)
        return -float(np.sum(end * np.log(h) + cont * np.log(1 - h)))
    r = optimize.minimize(nll, th0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    th = r.x
    se = np.full(len(th), np.nan)
    try:                                               # numerical Hessian (conditional on the expected counts)
        eps = 1e-4
        n = len(th)
        Hs = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                ei, ej = np.eye(n)[i] * eps, np.eye(n)[j] * eps
                Hs[i, j] = (nll(th + ei + ej) - nll(th + ei - ej) - nll(th - ei + ej) + nll(th - ei - ej)) / (4 * eps * eps)
        se = np.sqrt(np.diag(np.linalg.inv(Hs)))
    except Exception:
        pass
    return th, se


def _arrays(s: pd.DataFrame, D: int):
    z = np.nan_to_num(s["z"].to_numpy(float))
    zl = np.nan_to_num(s["zl"].to_numpy(float))
    obs = s["obs"].to_numpy(bool)
    ld = np.log(np.arange(1, D + 1, dtype=float))
    return z, zl, obs, ld


def init_from_hmm(m: R.MSAR, fam: str, age_drift: bool, D: int, shape: float = 1.0) -> HSMM:
    K = m.K
    th = np.zeros((K, 2))
    for k in range(K):
        stay = float(np.clip(m.A[k, k], 1e-4, 1 - 1e-4))
        if fam == "WB":
            th[k] = [np.log(-np.log(stay)), np.log(shape)]
        else:                                        # NB: r = shape, p chosen so the mean duration matches
            mean_d = 1 / (1 - stay)
            p = shape / (shape + mean_d - 1)
            th[k] = [np.log(shape), np.log(p / (1 - p))]
    W = m.A.copy()
    np.fill_diagonal(W, 0.0)
    W = W / np.maximum(W.sum(1, keepdims=True), EPS)
    if K == 2:
        W = np.array([[0.0, 1.0], [1.0, 0.0]])
    return HSMM(K, fam, age_drift, D, pi=m.pi.copy(), th=th, W=W, c=m.c.copy(), phi=m.phi.copy(),
                b=np.zeros(K), s2=m.s2.copy(), b_se=np.full(K, np.nan))


def em(s: pd.DataFrame, h0: HSMM, rc: HSMMConfig = HSMMConfig(), log=None) -> HSMM:
    """EM from the starting model ``h0`` on all rows of ``s``."""
    h = h0
    z, zl, obs, ld = _arrays(s, h.D)
    P = 3 if h.age_drift else 2
    prev = -np.inf
    th_se = np.full_like(h.th, np.nan)
    for it in range(1, rc.max_iter + 1):
        Hz = h.hazards()
        alpha, mt, cs, ll = _forward(z, zl, obs, ld, h.c, h.phi, h.b, h.s2, h.pi, Hz, h.W)
        end, cont, flow, g0, XtWX, XtWy, yWy, wsum = _backward(alpha, mt, cs, z, zl, obs, ld, h.c, h.phi, h.b,
                                                               h.s2, Hz, h.W, P)
        del alpha
        h.pi = g0[:, 0] / g0[:, 0].sum() if g0[:, 0].sum() > 0 else g0.sum(1) / g0.sum()
        for k in range(h.K):
            h.th[k], th_se[k] = _hazard_mstep(h.fam, h.th[k], end[k], cont[k], h.D)
        if h.K > 2:
            Wn = flow.copy()
            np.fill_diagonal(Wn, 0.0)
            h.W = Wn / np.maximum(Wn.sum(1, keepdims=True), EPS)
        b_se = np.full(h.K, np.nan)
        for k in range(h.K):
            A_ = XtWX[k] + 1e-9 * np.eye(P)
            beta = np.linalg.solve(A_, XtWy[k])
            rss = yWy[k] - 2 * beta @ XtWy[k] + beta @ A_ @ beta
            v = max(rss / max(wsum[k], EPS), rc.var_floor)
            h.c[k], h.phi[k], h.s2[k] = beta[0], beta[1], v
            if h.age_drift:
                h.b[k] = beta[2]
                b_se[k] = float(np.sqrt(v * np.linalg.inv(A_)[2, 2]))
        h.b_se = b_se
        h.loglik, h.n_iter = ll, it
        if log is not None and it % 25 == 0:
            log(f"    iter {it}: loglik {ll:.2f}")
        if np.isfinite(prev) and abs(ll - prev) < rc.tol * abs(ll):
            h.converged = True
            break
        prev = ll
    h.th_se = th_se
    h.n_obs = int(obs.sum())
    # sort states by phi (as RSW 01)
    o = np.argsort(h.phi)
    h.pi, h.th, h.th_se, h.W = h.pi[o], h.th[o], h.th_se[o], h.W[np.ix_(o, o)]
    h.c, h.phi, h.b, h.s2, h.b_se = h.c[o], h.phi[o], h.b[o], h.s2[o], h.b_se[o]
    return h


def fit(s: pd.DataFrame, mask: np.ndarray, hmm: R.MSAR, fam: str, age_drift: bool, rc: HSMMConfig = HSMMConfig(),
        warm: HSMM | None = None, log=None) -> HSMM:
    """Best of the starts on rows ``mask``: from the HMM with duration shape ``rc.start_shapes`` (1 = the HMM's
    geometric durations); or from ``warm`` (part C starts from the part B fit, with b = 0)."""
    import copy
    sub = s[mask]
    if warm is not None:
        starts = [copy.deepcopy(warm)]
        starts[0].age_drift = age_drift
        starts[0].b = np.zeros(warm.K)
    else:
        starts = [init_from_hmm(hmm, fam, age_drift, rc.D, sh) for sh in rc.start_shapes]
    best = None
    for h0 in starts:
        h = em(sub, h0, rc, log)
        if best is None or h.loglik > best.loglik:
            best = h
    return best


def filter_run(s: pd.DataFrame, h: HSMM, u: float) -> pd.DataFrame:
    z, zl, obs, ld = _arrays(s, h.D)
    filt, age, fc, lpz, pex = _filter(z, zl, obs, ld, h.c, h.phi, h.b, h.s2, h.pi, h.hazards(), h.W, u)
    out = pd.DataFrame(filt, index=s.index, columns=[f"p{k}" for k in range(h.K)])
    out["age"], out["fc"], out["lpz"], out["pexc"] = age, fc, lpz, pex
    return out


# ============================================================================= Synthetic (tests, smoke run)

def simulate_hsmm(T: int, phis=(-0.05, 0.25), sds=(0.8, 2.0), lam=(0.08, 0.6), beta=(1.6, 1.0),
                  b=(0.0, 0.0), seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-state AR(1) HSMM with discrete Weibull durations; returns z, state, age."""
    rng = np.random.default_rng(seed)
    z, st, ag = np.empty(T), np.empty(T, np.int64), np.empty(T, np.int64)
    s, d, zp = 0, 1, 0.0
    for t in range(T):
        if t:
            h = 1 - np.exp(-lam[s] * (d ** beta[s] - (d - 1) ** beta[s]))
            if rng.random() < h:
                s, d = 1 - s, 1
            else:
                d += 1
        z[t] = phis[s] * zp + b[s] * np.log(d) + sds[s] * rng.normal()
        zp, st[t], ag[t] = z[t], s, d
    return z, st, ag


def write_synthetic_hsmm(base, start: dt.date, end: dt.date, cfg, seed: int = 31, calm_beta: float = 1.8,
                         calm_lam: float = 0.005, burst_lam: float = 0.6, calm_b: float = 0.0) -> None:
    """MNQ ohlcv-1m month files with 5-minute z following ``simulate_hsmm`` (rising calm hazard by default: discrete
    Weibull beta = 1.8, lam = 0.005, mean calm duration about 17 bars as in RSW 01), on the regime_ar synthetic
    price/volume/roll layout."""
    import os
    from pathlib import Path
    from . import micro_bars as MB
    from .futures import CHICAGO, NY
    base = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base)
    rng = np.random.default_rng(seed)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    zall, _, _ = simulate_hsmm(len(days) * 276, phis=(-0.05, 0.25), sds=(0.85, 2.0), lam=(calm_lam, burst_lam),
                               beta=(calm_beta, 1.0), b=(calm_b, 0.0), seed=seed)
    zall = zall / np.std(zall)
    price, logvol, iid = 9000.0, 0.0, 1
    rows = []
    for i, d in enumerate(days):
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
        zb = zall[i * 276:(i + 1) * 276]
        e = rng.normal(0, 1, (276, 5)) * sd
        e = e - e.mean(1, keepdims=True)
        r = (e + (zb * bsd)[:, None] / 5).ravel()
        close = np.exp(np.log(price) + np.cumsum(r))
        price = close[-1]
        keep = rng.random(len(ts)) > 0.01
        rows.append(pd.DataFrame({"ts": ts[keep], "open": close[keep], "high": close[keep] + 0.25,
                                  "low": close[keep] - 0.25, "close": close[keep],
                                  "volume": rng.poisson(30, keep.sum()) + 1, "instrument_id": iid}))
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for key, part in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        a = dt.date(int(key[:4]), int(key[5:]), 1)
        f = MB.month_path(cfg.root, a)
        f.parent.mkdir(parents=True, exist_ok=True)
        part[MB.COLUMNS].reset_index(drop=True).to_parquet(f, index=False)
