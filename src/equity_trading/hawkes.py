"""Self-exciting peaks-over-threshold on MNQ 5-minute z (RSW 02; discussions/MNQ_Regime_Switching/003).

Events: e_t = 1 when bar t is observed and |z_t| >= u (u = a quantile of |z| on the fit window). z is the RSW 01
series (MRP 001 volatility model, rescaled by the previous day's rms).

Test 1, decay: C(k) = P(e_{t+k} = 1 | e_t = 1) against Markov nulls,
  * the binary first-order chain on e:      C(k) = p + (1 - p) lambda^k, lambda = p11 - p01
  * the RSW 01 models M1 (2 states) and M2 (3 states), simulated (hidden Markov: geometric decay at the latent
    eigenvalues).
  A Hawkes process keeps C(k) above them for longer. So does volatility persistence that the volatility model
  leaves in z (the Markov smoke run shows it), so H1 is necessary, not sufficient: the mechanism (self-excitation
  vs a persistent latent level) is not identified by the decay alone.

Test 2, forecast: discrete-time Hawkes, P(e_t = 1 | F_{t-1}) = 1 - exp(-lambda_t),
  lambda_t = mu + n * sum_{k >= 1} g(k) m_{t-k} e_{t-k},   sum_k g(k) = 1, so n is the branching ratio,
  kernels: geometric (HX), geometric with marks m = exp(delta (y - u)) normalized to mean 1 on the fit
  exceedances (HXM), power law g(k) ~ (k + c)^-(1 + gamma) truncated at ``pl_kmax`` bars (HP).
  Sizes: y - u ~ GPD(xi, sigma_t), sigma_t = sigma0 (lambda_t / mu)^kappa (kappa = 0: sizes independent of
  intensity).

Time runs over the 5-minute grid rows (276 per CME day; the daily break is not a row). Unobserved rows (first
slots, thin days, gaps) carry no event and no likelihood term; they still let the excitation decay.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

from . import mnq_predict as P
from . import regime_ar as R

try:
    from numba import njit
except ImportError:                                  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

HWK_VERSION = "0.1.0"
EPS = 1e-12


@dataclass(frozen=True)
class HWKConfig:
    q_main: float = 0.96                  # threshold quantile of |z| (about the RSW 01 burst share)
    q_sens: tuple = (0.98, 0.99)
    kmax: int = 120                       # decay curve lags (bars)
    excess_lags: tuple = (6, 60)          # H1 statistic: sum of C(k) - C_null(k) over these lags
    pl_kmax: int = 2760                   # power-law kernel truncation: 10 CME days
    n_boot: int = 2000
    block_days: int = 5
    hmm_sim_bars: int = 2_000_000
    n_starts: int = 4
    seed: int = 0
    alpha: float = 0.05
    nw_lags: int = 5
    stab_lo: float = 0.05                 # H3: branching ratio in (stab_lo, stab_hi) in every fit
    stab_hi: float = 0.95
    stab_range: float = 0.15              # H3: max - min across fits
    lift_top: float = 0.05


# ============================================================================= 1. Events

def threshold(s: pd.DataFrame, mask: np.ndarray, q: float, col: str = "z") -> float:
    a = np.abs(s[col].to_numpy(float))[mask & s["obs"].to_numpy(bool)]
    return float(np.quantile(a[np.isfinite(a)], q))


def events(s: pd.DataFrame, u: float, col: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """e (0/1 float) and y = |z| for every row; e = 0 on unobserved rows."""
    y = np.abs(s[col].to_numpy(float))
    obs = s["obs"].to_numpy(bool) & np.isfinite(y)
    e = (obs & (y >= u)).astype(float)
    return e, np.where(obs, y, 0.0)


# ============================================================================= 2. Test 1: decay of clustering

def decay_counts(e: np.ndarray, obs: np.ndarray, day_id: np.ndarray, kmax: int) -> dict:
    """Per-day sufficient statistics (day of the conditioning bar t):
    num[d, k] = sum e_t e_{t+k} obs_{t+k};  den[d, k] = sum e_t obs_{t+k};  ev[d], nob[d];
    n01[d] = sum (1 - e_t) obs_t e_{t+1}, d0[d] = sum (1 - e_t) obs_t obs_{t+1} (binary chain)."""
    nd = int(day_id.max()) + 1
    T = len(e)
    o = obs.astype(float)
    num = np.zeros((nd, kmax + 1))
    den = np.zeros((nd, kmax + 1))
    for k in range(1, kmax + 1):
        a = e[:T - k]
        num[:, k] = np.bincount(day_id[:T - k], a * e[k:] * o[k:], minlength=nd)
        den[:, k] = np.bincount(day_id[:T - k], a * o[k:], minlength=nd)
    c0 = (1 - e[:-1]) * o[:-1]
    return {"num": num, "den": den, "ev": np.bincount(day_id, e, minlength=nd),
            "nob": np.bincount(day_id, o, minlength=nd),
            "n01": np.bincount(day_id[:-1], c0 * e[1:], minlength=nd),
            "d0": np.bincount(day_id[:-1], c0 * o[1:], minlength=nd)}


def curves(cnt: dict, w: np.ndarray | None = None) -> dict:
    """C(k), base rate p and the binary-chain null from (optionally day-weighted) counts."""
    w = np.ones(len(cnt["ev"])) if w is None else w
    num, den = w @ cnt["num"], w @ cnt["den"]
    C = num / np.maximum(den, EPS)
    p = (w @ cnt["ev"]) / max(w @ cnt["nob"], EPS)
    p11 = C[1]
    p01 = (w @ cnt["n01"]) / max(w @ cnt["d0"], EPS)
    lam = p11 - p01
    k = np.arange(len(C))
    return {"C": C, "p": p, "p11": p11, "p01": p01, "lam": lam, "C_chain": p + (1 - p) * lam ** k}


def excess(C: np.ndarray, C_null: np.ndarray, lags: tuple) -> float:
    a, b = lags
    return float(np.sum(C[a:b + 1] - C_null[a:b + 1]))


def block_weights(n_days: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Day multiplicities of a moving-block bootstrap draw (blocks of consecutive days)."""
    nb = int(np.ceil(n_days / block))
    starts = rng.integers(0, max(n_days - block + 1, 1), nb)
    w = np.zeros(n_days)
    for s0 in starts:
        w[s0:s0 + block] += 1
    return w * n_days / w.sum()


@njit(cache=True)
def _simulate_msar(T, pi0, A, c, phi, sd, u, seed):
    np.random.seed(seed)
    K = len(c)
    e = np.zeros(T)
    s = 0
    r = np.random.random()
    acc = 0.0
    for k in range(K):
        acc += pi0[k]
        if r < acc:
            s = k
            break
    z = 0.0
    for t in range(T):
        if t:
            r = np.random.random()
            acc = 0.0
            for k in range(K):
                acc += A[s, k]
                if r < acc:
                    s = k
                    break
        z = c[s] + phi[s] * z + sd[s] * np.random.randn()
        if abs(z) >= u:
            e[t] = 1.0
    return e


def hmm_null_curve(m: R.MSAR, u: float, T: int, kmax: int, seed: int = 0) -> np.ndarray:
    """C(k) implied by an RSW MSAR fit (constant A), by simulation of one long continuous path."""
    w, v = np.linalg.eig(m.A.T)
    pi0 = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi0 = pi0 / pi0.sum()
    e = _simulate_msar(T, pi0, m.A, m.c.astype(float), m.phi.astype(float), np.sqrt(m.s2), u, seed)
    cnt = decay_counts(e, np.ones(T, bool), np.zeros(T, np.int64), kmax)
    return curves(cnt)["C"]


def decay_test(s: pd.DataFrame, u: float, nulls: dict, rc: HWKConfig = HWKConfig(), col: str = "z") -> dict:
    """H1 statistics with a day-block bootstrap. ``nulls``: name -> fixed C(k) curve (simulated hidden Markov
    fits); the binary chain ("chain") is refitted in every draw. H1 = every lower bound > 0."""
    e, _ = events(s, u, col)
    obs = s["obs"].to_numpy(bool)
    day_id = pd.factorize(s["trade_date"])[0].astype(np.int64)
    cnt = decay_counts(e, obs, day_id, rc.kmax)
    cv = curves(cnt)
    out = {"u": u, "p": cv["p"], "p11": cv["p11"], "p01": cv["p01"], "lambda_chain": cv["lam"],
           "C": cv["C"], "C_chain": cv["C_chain"], "events": int(e.sum()),
           "excess_vs_base": excess(cv["C"], np.full_like(cv["C"], cv["p"]), rc.excess_lags),
           "excess_vs_chain": excess(cv["C"], cv["C_chain"], rc.excess_lags)}
    for name, Cn in nulls.items():
        out[f"C_{name}"] = Cn
        out[f"excess_vs_{name}"] = excess(cv["C"], Cn, rc.excess_lags)
    rng = np.random.default_rng(rc.seed)
    boot = {name: np.empty(rc.n_boot) for name in ("chain",) + tuple(nulls)}
    nd = len(cnt["ev"])
    for i in range(rc.n_boot):
        cb = curves(cnt, block_weights(nd, rc.block_days, rng))
        boot["chain"][i] = excess(cb["C"], cb["C_chain"], rc.excess_lags)
        for name, Cn in nulls.items():
            boot[name][i] = excess(cb["C"], Cn, rc.excess_lags)
    for name, b in boot.items():
        out[f"lo95_vs_{name}"] = float(np.quantile(b, 0.05))
    out["H1"] = bool(all(out[f"lo95_vs_{name}"] > 0 for name in boot))
    return out


# ============================================================================= 3. Test 2: Hawkes models

@njit(cache=True)
def _geo_excitation(x, beta):
    """R_t = sum_{k>=1} (1 - beta) beta^(k-1) x_{t-k}."""
    T = len(x)
    out = np.zeros(T)
    r = 0.0
    for t in range(1, T):
        r = beta * r + (1 - beta) * x[t - 1]
        out[t] = r
    return out


def pl_kernel(c: float, gamma: float, kmax: int) -> np.ndarray:
    k = np.arange(1, kmax + 1, dtype=float)
    lg = -(1 + gamma) * np.log(k + c)                    # log space: no overflow for large c, gamma
    g = np.exp(lg - lg.max())
    return g / g.sum()


def _pl_excitation(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    full = signal.fftconvolve(x, g)[: len(x)]            # full[t] = sum_{j} g[j] x[t - j], j = 0 .. (lag j+1)
    return np.r_[0.0, full[:-1]]


@dataclass
class HawkesFit:
    kind: str                         # "HX", "HXM", "HP"
    mu: float
    n: float
    beta: float = np.nan              # HX, HXM
    delta: float = 0.0                # HXM
    mark_norm: float = 1.0            # HXM: mean exp(delta (y - u)) over fit exceedances
    c: float = np.nan                 # HP
    gamma: float = np.nan             # HP
    u: float = np.nan
    loglik: float = -np.inf
    n_obs: int = 0
    n_events: int = 0
    converged: bool = False
    pl_kmax: int = 2760

    @property
    def half_life_bars(self) -> float:
        if self.kind in ("HX", "HXM"):
            return float(np.log(0.5) / np.log(self.beta))
        g = pl_kernel(self.c, self.gamma, self.pl_kmax)
        return float(np.searchsorted(np.cumsum(g), 0.5) + 1)     # median lag of the kernel

    def row(self) -> dict:
        return {"kind": self.kind, "u": self.u, "mu": self.mu, "n_branching": self.n, "beta": self.beta,
                "delta": self.delta, "c": self.c, "gamma": self.gamma, "half_life_or_median_lag": self.half_life_bars,
                "loglik": self.loglik, "events": self.n_events, "n_obs": self.n_obs, "converged": self.converged}


def intensity(f: HawkesFit, e: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lambda_t, excitation part) for every row; uses events up to t-1 only."""
    if f.kind == "HP":
        x = e
        exc = _pl_excitation(x, pl_kernel(f.c, f.gamma, f.pl_kmax))
    else:
        x = e * (np.exp(f.delta * (y - f.u)) / f.mark_norm if f.kind == "HXM" else 1.0)
        exc = _geo_excitation(x, f.beta)
    return f.mu + f.n * exc, f.n * exc


def _bern_ll(lam: np.ndarray, e: np.ndarray, obs: np.ndarray) -> float:
    p = np.clip(1 - np.exp(-lam[obs]), EPS, 1 - EPS)
    ee = e[obs]
    return float(np.sum(ee * np.log(p) + (1 - ee) * np.log(1 - p)))


def _unpack(kind, th, e, y, u, kmax, mask_ev):
    mu, n = np.exp(th[0]), 1 / (1 + np.exp(-th[1]))
    f = HawkesFit(kind, float(mu), float(n), u=u, pl_kmax=kmax)
    if kind in ("HX", "HXM"):
        f.beta = float(1 / (1 + np.exp(-th[2])))
    if kind == "HXM":
        f.delta = float(th[3])
        f.mark_norm = float(np.mean(np.exp(f.delta * (y[mask_ev] - u)))) if mask_ev.any() else 1.0
    if kind == "HP":
        f.c, f.gamma = float(np.exp(th[2])), float(np.exp(th[3]))
    return f


def fit_hawkes(kind: str, e: np.ndarray, y: np.ndarray, obs: np.ndarray, u: float,
               rc: HWKConfig = HWKConfig()) -> HawkesFit:
    """MLE on the rows given (history starts at the first row). Best of ``n_starts`` starts."""
    p = e[obs].mean()
    mask_ev = e > 0
    rng = np.random.default_rng(rc.seed)

    def nll(th):
        f = _unpack(kind, th, e, y, u, rc.pl_kmax, mask_ev)
        lam, _ = intensity(f, e, y)
        return -_bern_ll(lam, e, obs)

    best = None
    for i in range(rc.n_starts):
        n0 = rng.uniform(0.1, 0.6)
        th = [np.log(p * (1 - n0)), np.log(n0 / (1 - n0))]
        if kind in ("HX", "HXM"):
            th.append(np.log(rng.uniform(0.5, 0.97) / (1 - 0.97)) if i else 2.0)
        if kind == "HXM":
            th.append(rng.normal(0, 0.3))
        if kind == "HP":
            th += [np.log(rng.uniform(0.5, 10)), np.log(rng.uniform(0.2, 1.5))]
        r = optimize.minimize(nll, np.asarray(th, float), method="Nelder-Mead" if kind == "HP" else "L-BFGS-B",
                              options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-4} if kind == "HP" else {"maxiter": 500})
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x, bool(r.success))
    f = _unpack(kind, best[1], e, y, u, rc.pl_kmax, mask_ev)
    f.loglik, f.converged, f.n_obs, f.n_events = -best[0], best[2], int(obs.sum()), int(e[obs].sum())
    return f


# --------------------------------------------------------------------------- benchmarks

def prob_constant(e: np.ndarray, obs: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    return np.full(len(e), e[obs & fit_mask].mean())


def prob_chain(e: np.ndarray, obs: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Binary first-order chain: P(e_t | e_{t-1}); an unobserved previous row counts as no event."""
    prev = np.r_[0.0, e[:-1]]
    m = obs & fit_mask
    p11 = e[m & (prev > 0)].mean()
    p01 = e[m & (prev == 0)].mean()
    return np.where(prev > 0, p11, p01), {"p01": float(p01), "p11": float(p11)}


def prob_msar(s: pd.DataFrame, m: R.MSAR, u: float, col: str = "z") -> np.ndarray:
    """P(|z_t| >= u | F_{t-1}) under an RSW MSAR fit: sum_s P(s_t | F_{t-1}) P(|N(c_s + phi_s zl_t, s2_s)| >= u).
    ``s`` must carry the column used by the fit as ``z``."""
    y, G, obs, At, lB = R._pass(s, m)
    _, prior, _, _, _ = R._forward(lB, m.pi, At)
    mean = G @ m.beta.T
    sd = np.sqrt(m.s2)[None, :]
    pk = stats.norm.cdf((-u - mean) / sd) + stats.norm.sf((u - mean) / sd)
    return (prior * pk).sum(1)


# ============================================================================= 4. Sizes (GPD)

def gpd_fit(exc: np.ndarray, lam_ratio: np.ndarray | None = None) -> dict:
    """y - u ~ GPD(xi, sigma), sigma = sigma0 * lam_ratio^kappa. Returns xi, sigma0, kappa, LR test of kappa = 0."""
    exc = np.asarray(exc, float)

    def nll(th, use_k):
        xi, ls, k = th[0], th[1], (th[2] if use_k else 0.0)
        sig = np.exp(ls) * (lam_ratio ** k if use_k else 1.0)
        return -np.sum(stats.genpareto.logpdf(exc, xi, loc=0, scale=sig))

    xi0, _, sc0 = stats.genpareto.fit(exc, floc=0)
    r0 = optimize.minimize(nll, [xi0, np.log(sc0)], args=(False,), method="Nelder-Mead")
    out = {"xi": float(r0.x[0]), "sigma": float(np.exp(r0.x[1])), "n": len(exc)}
    if lam_ratio is not None:
        lam_ratio = np.asarray(lam_ratio, float)
        r1 = optimize.minimize(nll, [r0.x[0], r0.x[1], 0.0], args=(True,), method="Nelder-Mead")
        lr = 2 * (r0.fun - r1.fun)
        out.update({"xi_k": float(r1.x[0]), "sigma0_k": float(np.exp(r1.x[1])), "kappa": float(r1.x[2]),
                    "lr": float(lr), "p_kappa0": float(stats.chi2.sf(max(lr, 0), 1))})
    return out


# ============================================================================= 5. Scores

def bernoulli_score(p: np.ndarray, e: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return e * np.log(p) + (1 - e) * np.log(1 - p)


def score_dm(sa: np.ndarray, sb: np.ndarray, day: np.ndarray, lags: int = 5) -> dict:
    """H1 of the test: model a has the higher log score. Daily mean differences (a - b), Newey-West."""
    ok = np.isfinite(sa) & np.isfinite(sb)
    d = pd.Series(sa[ok] - sb[ok]).groupby(pd.Series(day[ok])).mean().to_numpy()
    m, _, t = P.nw_mean_t(d, lags)
    return {"mean_daily_diff": m, "per_bar_diff": float(np.mean(sa[ok] - sb[ok])), "t": t,
            "p": float(stats.norm.sf(t)), "days": len(d)}


def calibration(p: np.ndarray, e: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"p": p, "e": e, "bin": q}).groupby("bin").agg(mean_p=("p", "mean"), rate=("e", "mean"),
                                                                       n=("e", "size"))


def lift(p: np.ndarray, e: np.ndarray, top: float) -> dict:
    k = max(int(np.ceil(top * len(p))), 1)
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind="stable")[:k]] = True          # exactly the top share (ties broken by order)
    return {"top_share": top, "rate_top": float(e[sel].mean()), "rate_all": float(e.mean()),
            "lift": float(e[sel].mean() / e.mean()), "share_of_events_caught": float(e[sel].sum() / e.sum())}


# ============================================================================= 6. Synthetic (tests)

def simulate_hawkes(T: int, mu: float, n: float, beta: float, seed: int = 0) -> np.ndarray:
    """Discrete-time geometric-kernel Hawkes events (Bernoulli with P = 1 - exp(-lambda))."""
    rng = np.random.default_rng(seed)
    e = np.zeros(T)
    r = 0.0
    for t in range(T):
        if t:
            r = beta * r + (1 - beta) * e[t - 1]
        e[t] = rng.random() < 1 - np.exp(-(mu + n * r))
    return e


def write_synthetic_bursts(base, start: dt.date, end: dt.date, cfg: P.MRPConfig, seed: int = 21,
                           mu: float = 0.02, n: float = 0.5, beta: float = 0.97, burst_mult: float = 2.5) -> None:
    """MNQ ohlcv-1m month files under ``base`` with Hawkes-timed bursts: bucket b is a burst with probability
    1 - exp(-(mu + n R_b)), R_b = geometric excitation of past bursts (continuous across days); a burst bucket has
    burst_mult x the normal sd. Same price, volume, roll and missing-minute layout as regime_ar's generator."""
    import os
    from pathlib import Path
    from . import micro_bars as MB
    from .futures import CHICAGO, NY
    base = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base)
    rng = np.random.default_rng(seed)
    price, logvol, iid, r_exc, prev = 9000.0, 0.0, 1, 0.0, 0.0
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
            nb = len(sd)
            mult = np.ones(nb)
            for b in range(nb):
                r_exc = beta * r_exc + (1 - beta) * prev
                prev = float(rng.random() < 1 - np.exp(-(mu + n * r_exc)))
                if prev:
                    mult[b] = burst_mult
            r = (rng.normal(0, 1, (nb, 5)) * sd * mult[:, None]).ravel()
            close = np.exp(np.log(price) + np.cumsum(r))
            price = close[-1]
            keep = rng.random(len(ts)) > 0.01
            rows.append(pd.DataFrame({"ts": ts[keep], "open": close[keep], "high": close[keep] + 0.25,
                                      "low": close[keep] - 0.25, "close": close[keep],
                                      "volume": rng.poisson(30, keep.sum()) + 1, "instrument_id": iid}))
        d += dt.timedelta(days=1)
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for key, part in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        a = dt.date(int(key[:4]), int(key[5:]), 1)
        f = MB.month_path(cfg.root, a)
        f.parent.mkdir(parents=True, exist_ok=True)
        part[MB.COLUMNS].reset_index(drop=True).to_parquet(f, index=False)
