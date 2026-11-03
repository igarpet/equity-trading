"""S05a: Avellaneda-Lee (2010) residual mean reversion on 15-minute bars, sector-ETF factor.

Each decision day e (close), for every stock i with its GICS sector ETF:
  1. OLS of the stock's 15m log returns on the ETF's over the trailing W sessions:
     r_i = alpha_i + beta_i r_etf + eps_i.
  2. X = cumulative eps over the window (ends at 0 by construction); AR(1) X_{t+1} = a + b X_t + z.
     kappa = -ln b (per bar), m = a / (1 - b), sigma_eq = sqrt(var(z) / (1 - b^2)).
     Tradable if 0 < b < 1 and the mean-reversion time 1/kappa < max_tau_frac x window.
  3. m is centred cross-sectionally (m - mean m over tradable stocks), as in the paper.
During day e+1 the residual keeps accumulating bar by bar with the frozen alpha, beta:
  s = (X_t - m_c) / sigma_eq.
Rules (paper): open long s < -1.25, open short s > 1.25, close long s > -0.50, close short s < 0.75.
A position is the stock (size_per_stock x equity) against beta x that notional in its ETF; ETF hedges
are netted per ETF, so only net ETF changes trade. Signals at a bar close, fills at the next open.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .s04 import Panel, block_bootstrap_mean, perf_metrics, placebo_rank  # noqa: F401  (re-exported for the notebook)

S05_VERSION = "0.1.0"

SECTOR_ETF = {"Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
              "Industrials": "XLI", "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Utilities": "XLU",
              "Materials": "XLB", "Real Estate": "XLRE", "Communication Services": "XLC"}


@dataclass(frozen=True)
class S05Config:
    timeframe: str = "15Min"
    ffill_bars: int = 2
    window_sessions: int = 20
    min_bar_share: float = 0.95
    max_tau_frac: float = 0.5          # 1/kappa < 0.5 x window (paper)
    center_m: bool = True
    open_long: float = -1.25
    open_short: float = 1.25
    close_long: float = -0.50
    close_short: float = 0.75
    size_per_stock: float = 0.01       # stock-leg notional, fraction of equity
    max_positions: int = 100
    initial_equity: float = 100_000.0
    half_spread_bps: float = 2.0       # one-way cost per fill (S04 amendment A1)
    cost_mult: float = 1.0
    n_placebo: int = 1000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== estimation
def log_returns(panel: Panel) -> np.ndarray:
    """(T, N) bar log returns; the first bar of a session carries the overnight move."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(panel.c)
    r = np.vstack([np.full((1, lc.shape[1]), np.nan), np.diff(lc, axis=0)])
    return r


def estimate_day(R: np.ndarray, E: np.ndarray, cfg: S05Config) -> dict:
    """Vectorised per-stock estimation on one window. R, E: (W, N) stock and matching-ETF returns."""
    ok = np.isfinite(R) & np.isfinite(E)
    share = ok.mean(axis=0)
    R0, E0 = np.where(ok, R, 0.0), np.where(ok, E, 0.0)
    n = ok.sum(axis=0).clip(min=1)
    rm, em = R0.sum(0) / n, E0.sum(0) / n
    dr, de = np.where(ok, R0 - rm, 0.0), np.where(ok, E0 - em, 0.0)
    vx = (de * de).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (dr * de).sum(0) / vx
        alpha = rm - beta * em
        eps = np.where(ok, R0 - alpha - beta * E0, 0.0)
        X = np.cumsum(eps, axis=0)
        x0, x1 = X[:-1], X[1:]
        m0, m1 = x0.mean(0), x1.mean(0)
        b = ((x0 - m0) * (x1 - m1)).sum(0) / ((x0 - m0) ** 2).sum(0)
        a = m1 - b * m0
        z = x1 - a - b * x0
        vz = z.var(axis=0, ddof=1)
        kappa = -np.log(b)
        tau = 1.0 / kappa
        m = a / (1 - b)
        sig = np.sqrt(vz / (1 - b * b))
    W = R.shape[0]
    valid = ((share >= cfg.min_bar_share) & (vx > 0) & np.isfinite(beta) & (b > 0) & (b < 1) & np.isfinite(sig)
             & (sig > 0) & (tau < cfg.max_tau_frac * W))
    mc = m - (np.nanmean(m[valid]) if cfg.center_m and valid.any() else 0.0)
    return {"alpha": alpha, "beta": beta, "m": mc, "sig": sig, "tau": tau, "valid": valid}


def estimate_all(panel: Panel, stock_cols: np.ndarray, etf_cols: np.ndarray, cfg: S05Config,
                 first_day: int) -> dict[str, np.ndarray]:
    """(D, N) parameter arrays; row e holds the estimate at the close of decision day e."""
    r = log_returns(panel)
    D, N = panel.n_days, len(stock_cols)
    out = {k: np.full((D, N), np.nan) for k in ["alpha", "beta", "m", "sig", "tau"]}
    out["valid"] = np.zeros((D, N), bool)
    ds = panel.day_start
    for e in range(first_day, D - 1):
        a, b = ds[e - cfg.window_sessions + 1], ds[e + 1]
        est = estimate_day(r[a:b][:, stock_cols], r[a:b][:, etf_cols], cfg)
        for k in out:
            out[k][e] = est[k]
    return out


def first_decision_day(cfg: S05Config) -> int:
    return cfg.window_sessions


# =================================================================== backtest
def backtest(panel: Panel, params: dict, stock_cols: np.ndarray, etf_cols: np.ndarray, cfg: S05Config,
             mode: str = "strategy", entry_prob: float | None = None, seed: int = 0, record: bool = True) -> dict:
    """Bar-by-bar simulation. mode: 'strategy' | 'placebo_timing' (contrarian entries at random flat
    tradable bars with |s| beyond the exit band, at probability entry_prob)."""
    rng = np.random.default_rng(seed)
    r = log_returns(panel)
    o, c = panel.o, panel.c
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    hs = cfg.half_spread_bps * 1e-4 * cfg.cost_mult
    ncol = c.shape[1]
    hold = np.zeros(ncol)                  # shares per panel column (stocks and ETFs)
    cash = cfg.initial_equity
    last_px = np.full(ncol, np.nan)
    side = np.zeros(N, np.int8)
    sh_s = np.zeros(N)
    sh_h = np.zeros(N)
    ent_bar = np.zeros(N, np.int64)
    ent_ps, ent_pe, ent_s = np.zeros(N), np.zeros(N), np.zeros(N)
    pend = np.zeros(N, np.int8)            # +1/-1 open long/short, 2 close
    pend_reason = np.zeros(N, np.int8)     # 1 rule, 2 invalid
    pend_s = np.zeros(N)
    trades = []
    eq_bar = np.full(T, np.nan)
    n_open = np.zeros(T, np.int16)
    gross = np.zeros(T)
    total_cost = 0.0
    elig_count = sig_count = 0
    eq_prev = cfg.initial_equity
    first = first_decision_day(cfg) + 1
    eq_bar[:panel.day_start[first]] = cfg.initial_equity
    beta_today = np.zeros(N)

    def execute(delta: np.ndarray, px: np.ndarray):
        nonlocal cash, total_cost
        m = delta != 0
        if m.any():
            d, p = delta[m], px[m]
            cash -= np.sum(d * p * (1 + np.sign(d) * hs))
            total_cost += np.sum(np.abs(d) * p * hs)
            hold[m] += d

    for t in range(first, D):
        e = t - 1
        al, be, mc, sg, vd = (params[k][e] for k in ("alpha", "beta", "m", "sig", "valid"))
        beta_today = np.where(np.isfinite(be), be, beta_today)
        X = np.zeros(N)
        a0, a1 = panel.day_start[t], panel.day_start[t + 1]
        for B in range(a0, a1):
            # ---------- fills at this bar's open
            if pend.any():
                px_open = np.where(np.isfinite(o[B]), o[B], last_px)
                ps, pe = px_open[stock_cols], px_open[etf_cols]
                delta = np.zeros(ncol)
                # closes
                cl = (pend == 2) & np.isfinite(ps) & np.isfinite(pe)
                if cl.any():
                    idx = np.where(cl)[0]
                    np.add.at(delta, stock_cols[idx], -sh_s[idx])
                    np.add.at(delta, etf_cols[idx], -sh_h[idx])
                    if record:
                        gp = sh_s[idx] * (ps[idx] - ent_ps[idx]) + sh_h[idx] * (pe[idx] - ent_pe[idx])
                        for k, i in enumerate(idx):
                            trades.append((int(i), int(side[i]), int(ent_bar[i]), B, float(ent_s[i]), float(gp[k]),
                                           float(abs(sh_s[i]) * ent_ps[i]), int(pend_reason[i])))
                    side[idx] = 0
                    sh_s[idx] = 0
                    sh_h[idx] = 0
                    pend[idx] = 0
                # opens
                op = ((pend == 1) | (pend == -1)) & np.isfinite(ps) & np.isfinite(pe) & (pe > 0)
                if op.any():
                    idx = np.where(op)[0]
                    n_sh = np.floor(cfg.size_per_stock * eq_prev / ps[idx])
                    good = n_sh >= 1
                    idx, n_sh = idx[good], n_sh[good]
                    dirs = pend[idx].astype(float)
                    s_sh = dirs * n_sh
                    h_sh = -np.round(beta_today[idx] * s_sh * ps[idx] / pe[idx])
                    np.add.at(delta, stock_cols[idx], s_sh)
                    np.add.at(delta, etf_cols[idx], h_sh)
                    side[idx] = dirs.astype(np.int8)
                    sh_s[idx], sh_h[idx] = s_sh, h_sh
                    ent_bar[idx], ent_ps[idx], ent_pe[idx], ent_s[idx] = B, ps[idx], pe[idx], pend_s[idx]
                pend[(pend == 1) | (pend == -1)] = 0      # unfilled opens are dropped
                execute(delta, px_open)
            # ---------- close of bar
            row = c[B]
            okp = np.isfinite(row)
            last_px[okp] = row[okp]
            rb = r[B]
            eps = rb[stock_cols] - al - be * rb[etf_cols]
            X = X + np.where(np.isfinite(eps), eps, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                s = (X - mc) / sg
            fin = np.isfinite(s)
            # exits
            ex = ((side > 0) & fin & (s > cfg.close_long)) | ((side < 0) & fin & (s < cfg.close_short))
            inv = (side != 0) & ~vd
            pend_reason[ex] = 1
            pend_reason[inv & ~ex] = 2
            pend[(ex | inv) & (pend != 2)] = 2
            # entries
            flat = (side == 0) & (pend == 0) & vd & fin
            if mode == "placebo_timing":
                elig = flat & ((s < cfg.close_long) | (s > cfg.close_short))
                sig_m = elig & (rng.random(N) < entry_prob)
            else:
                elig = flat & ((s < cfg.close_long) | (s > cfg.close_short))
                sig_m = flat & ((s < cfg.open_long) | (s > cfg.open_short))
            elig_count += int(elig.sum())
            sig_count += int(sig_m.sum())
            if sig_m.any():
                free = cfg.max_positions - int(((side != 0) & (pend != 2)).sum())
                if free > 0:
                    cand = np.where(sig_m)[0]
                    if len(cand) > free:
                        cand = cand[np.argsort(-np.abs(s[cand]), kind="stable")[:free]]
                    pend[cand] = np.where(s[cand] > 0, -1, 1)
                    pend_s[cand] = s[cand]
            # ---------- mark
            val = np.where(hold != 0, hold * last_px, 0.0)
            eq_prev = cash + val.sum()
            eq_bar[B] = eq_prev
            n_open[B] = int((side != 0).sum())
            gross[B] = np.abs(val).sum()
    # liquidate at the last close
    Bl = T - 1
    if np.any(hold != 0):
        idx = np.where(side != 0)[0]
        if record:
            ps, pe = last_px[stock_cols], last_px[etf_cols]
            gp = sh_s[idx] * (ps[idx] - ent_ps[idx]) + sh_h[idx] * (pe[idx] - ent_pe[idx])
            for k, i in enumerate(idx):
                trades.append((int(i), int(side[i]), int(ent_bar[i]), Bl, float(ent_s[i]), float(gp[k]),
                               float(abs(sh_s[i]) * ent_ps[i]), 3))
        execute(-hold.copy(), last_px)
        eq_bar[Bl] = cash
    day_eq = pd.Series(eq_bar[panel.day_start[1:] - 1], index=pd.to_datetime(panel.days)).ffill().fillna(cfg.initial_equity)
    out = {"daily_equity": day_eq, "final_equity": float(day_eq.iloc[-1]), "total_cost": total_cost,
           "entry_rate": sig_count / max(elig_count, 1), "first_day": pd.Timestamp(panel.days[min(first, D - 1)])}
    if record:
        tr = pd.DataFrame(trades, columns=["i", "side", "entry_bar", "exit_bar", "s_entry", "gross_pnl", "notional",
                                           "reason_code"])
        tr["reason"] = tr["reason_code"].map({1: "rule", 2: "invalid", 3: "end"})
        tr["bars"] = tr["exit_bar"] - tr["entry_bar"]
        tr["gross_ret"] = tr["gross_pnl"] / tr["notional"]
        tr["entry_time"] = panel.index[tr["entry_bar"].to_numpy()] if len(tr) else []
        out.update(trades=tr, bar_equity=pd.Series(eq_bar, index=panel.index), n_open=n_open, gross_exposure=gross)
    return out


# =================================================================== diagnostics
def random_direction_test(gross_pnl: np.ndarray, n: int = 10_000, seed: int = 0) -> dict:
    """Same trades, random sign per trade: is the summed gross P&L of the real directions in the upper tail?"""
    g = np.asarray(gross_pnl, float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(g)))
    dist = signs @ g
    obs = float(g.sum())
    return {"observed_gross": obs, "p_upper": float((1 + np.sum(dist >= obs)) / (n + 1)),
            "percentile": float(100 * np.mean(dist < obs)), "placebo_p95": float(np.percentile(dist, 95))}


def run_placebo(panel: Panel, params: dict, stock_cols, etf_cols, cfg: S05Config, entry_prob: float, seed: int, start) -> dict:
    r = backtest(panel, params, stock_cols, etf_cols, cfg, mode="placebo_timing", entry_prob=entry_prob, seed=seed,
                 record=False)
    return {"seed": seed, **perf_metrics(r["daily_equity"], cfg.initial_equity, start)}


def gates(boot: dict, rand_dir: dict, pl_timing: dict, cost2: dict, base: dict, alpha: float = 0.05) -> dict:
    """Pre-registered gates (studies/S05a_README.md)."""
    g1 = all(v[1] > 0 for v in boot.values())
    g2 = rand_dir["p_upper"] <= alpha
    g3 = pl_timing["total_return"]["p_upper"] <= alpha and pl_timing["sharpe"]["p_upper"] <= alpha
    g4 = cost2["total_return"] > 0 and cost2["sharpe"] > 0
    stat = g1 and g2 and g3
    econ = base["total_return"] > 0 and g4
    return {"G1_bootstrap": g1, "G2_random_direction": g2, "G3_vs_random_timing": g3, "G4_double_cost": g4,
            "statistical_verdict": "SIGNAL" if stat else ("PARTIAL" if (g1 or g2 or g3) else "NO_SIGNAL"),
            "economic_verdict": "PROFITABLE_AFTER_COSTS" if econ else "NOT_PROFITABLE_AFTER_COSTS",
            "verdict": "EDGE_CANDIDATE" if (stat and econ) else "NO_EDGE"}
