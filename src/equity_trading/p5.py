"""P5: Garleanu & Pedersen (2013) - dynamic trading with predictable returns and transaction costs.

Protocol: discussions/Dynamic_Trading_Costs/001_claude_p5_garleanu_pedersen_preregistration.md

Two 15-minute signals from the S05 estimation (parameters of session t-1 frozen for session t):
  A (fast) = S05 s-score (cumulative residual since the open, minus m, over sigma_eq);
  B (slow) = cumulative residual over the last L = 130 bars over sigma_z sqrt(L).
Each session, pooled over the previous 20 sessions: B_k from OLS of the next bar's hedged return / sigma_z on
(-s_A, -s_B); phi_k = 1 - pooled AR(1) of each signal. GP rule with q = lambda/gamma:
  c = a/gamma = (-1 + sqrt(1 + 4q)) / 2,  tau = c / q,
  u_i = [B_A(-s_A)/(1 + phi_A c) + B_B(-s_B)/(1 + phi_B c)] / sigma_z,  aim = u / sum|u| (|w| <= cap),
  x_next = x + tau (aim - x), traded at the next open; ETF hedge netted as in S07.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .s04 import Panel, perf_metrics
from .s05 import log_returns

P5_VERSION = "0.1.0"


@dataclass(frozen=True)
class P5Config:
    slow_bars: int = 130
    est_sessions: int = 20
    gross: float = 1.0
    w_cap: float = 0.02
    etf_band: float = 0.001
    initial_equity: float = 100_000.0
    cost_mult: float = 1.0
    arms: tuple = (12.0, 90.0, 600.0)
    alpha: float = 0.05 / 3
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    n_placebo: int = 100
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def gp_rate(q: float) -> tuple[float, float]:
    """(c = a/gamma, tau = a/lambda) for q = lambda/gamma, no discounting."""
    c = (-1.0 + np.sqrt(1.0 + 4.0 * q)) / 2.0
    return float(c), float(c / q)


# =================================================================== signals
def signals(panel: Panel, P: dict, stock_cols: np.ndarray, etf_cols: np.ndarray, slow_bars: int = 130) -> dict:
    """Bar-level arrays (float32, (T, N)): sA, sB, hedged return h, sigma_z; plus day index per bar.
    Row t of P holds the estimate at the close of session t (used for session t + 1)."""
    r = log_returns(panel)
    ds = panel.day_start
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    f32 = np.float32
    sA = np.full((T, N), np.nan, f32)
    eps = np.zeros((T, N))
    h = np.full((T, N), np.nan, f32)
    sz = np.full((T, N), np.nan, f32)
    valid = np.zeros((T, N), bool)
    day = np.zeros(T, np.int32)
    for t in range(1, D):
        a0, a1 = ds[t], ds[t + 1]
        day[a0:a1] = t
        al, be, mc, sg, tau, vd = (P[k][t - 1] for k in ("alpha", "beta", "m", "sig", "tau", "valid"))
        if not np.any(vd):
            continue
        R, E = r[a0:a1][:, stock_cols], r[a0:a1][:, etf_cols]
        e = R - al - be * E
        e0 = np.where(np.isfinite(e), e, 0.0)
        eps[a0:a1] = np.where(vd, e0, 0.0)
        X = np.cumsum(e0, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (X - mc) / sg
            b = np.exp(-1.0 / tau)
            z = sg * np.sqrt(np.clip(1 - b * b, 0, None))
        sA[a0:a1] = np.where(vd, s, np.nan)
        h[a0:a1] = np.where(vd, R - be * E, np.nan)
        sz[a0:a1] = np.where(vd, np.broadcast_to(z, s.shape), np.nan)
        valid[a0:a1] = vd
    cs = np.vstack([np.zeros((1, N)), np.cumsum(eps, 0)])
    L = slow_bars
    roll = np.full((T, N), np.nan)
    roll[L - 1:] = cs[L:] - cs[:T - L + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        sB = (roll / (sz * np.sqrt(L))).astype(f32)
    sB[~valid] = np.nan
    return {"sA": sA, "sB": sB, "h": h, "sz": sz, "valid": valid, "day": day}


def day_stats(sig: dict, n_days: int) -> dict:
    """Per-session sufficient statistics: OLS of h_{t+1}/sz_t on (-sA_t, -sB_t), and the bar-to-bar AR(1) of sA and sB
    over all consecutive bars (for sA this includes its reset at each open: the forecast's real persistence)."""
    sA, sB, h, sz, day = sig["sA"], sig["sB"], sig["h"], sig["sz"], sig["day"]
    T = len(day)
    xA, xB = -sA[:-1].astype(float), -sB[:-1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = h[1:].astype(float) / sz[:-1].astype(float)
    ok = np.isfinite(xA) & np.isfinite(xB) & np.isfinite(y)
    okA = np.isfinite(sA[:-1]) & np.isfinite(sA[1:])
    okB = np.isfinite(sB[:-1]) & np.isfinite(sB[1:])
    d = day[:-1]
    st = np.zeros((n_days, 13))
    def acc(col, v, m):
        np.add.at(st[:, col], d, np.where(m, v, 0.0).sum(1))
    acc(0, xA * xA, ok); acc(1, xA * xB, ok); acc(2, xB * xB, ok); acc(3, xA * y, ok); acc(4, xB * y, ok)
    a0, a1 = sA[:-1].astype(float), sA[1:].astype(float)
    b0, b1 = sB[:-1].astype(float), sB[1:].astype(float)
    acc(5, a0 * a1, okA); acc(6, a0 * a0, okA); acc(7, b0 * b1, okB); acc(8, b0 * b0, okB)
    acc(9, np.ones_like(xA), ok)
    return st


def estimates(st: np.ndarray, est_sessions: int) -> pd.DataFrame:
    """Row t: (B_A, B_B, phi_A, phi_B) estimated on sessions t - est_sessions .. t - 1 (for trading session t)."""
    D = len(st)
    cs = np.vstack([np.zeros((1, st.shape[1])), np.cumsum(st, 0)])
    out = np.full((D, 4), np.nan)
    for t in range(est_sessions, D):
        s = cs[t] - cs[t - est_sessions]
        XtX = np.array([[s[0], s[1]], [s[1], s[2]]])
        if s[9] < 1000 or np.linalg.det(XtX) <= 0:
            continue
        BA, BB = np.linalg.solve(XtX, s[3:5])
        rA = s[5] / s[6] if s[6] > 0 else np.nan
        rB = s[7] / s[8] if s[8] > 0 else np.nan
        out[t] = (BA, BB, 1 - rA, 1 - rB)
    return pd.DataFrame(out, columns=["B_A", "B_B", "phi_A", "phi_B"])


def aim_weights(sA, sB, sz, ok, BA, BB, phA, phB, c, gross, cap, mode: str = "gp") -> np.ndarray:
    """mode: 'gp' persistence-weighted; 'noPW' equal weights; 'A' fast only; 'B' slow only."""
    wA = 1.0 / (1.0 + max(phA, 0.0) * c)
    wB = 1.0 / (1.0 + max(phB, 0.0) * c)
    if mode == "noPW":
        wA = wB = 1.0
    gA = BA * wA if mode != "B" else 0.0
    gB = BB * wB if mode != "A" else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (gA * (-np.nan_to_num(sA)) + gB * (-np.nan_to_num(sB))) / sz
    u = np.where(ok & np.isfinite(u), u, 0.0)
    tot = np.abs(u).sum()
    if tot <= 0:
        return np.zeros_like(u)
    return np.clip(gross * u / tot, -cap, cap)


# =================================================================== backtest
def backtest(panel: Panel, P: dict, sig: dict, est: pd.DataFrame, stock_cols: np.ndarray, etf_cols: np.ndarray,
             half_spread: np.ndarray, cfg: P5Config, q: float, mode: str = "gp", block: np.ndarray | None = None,
             shuffle: bool = False, seed: int = 0) -> dict:
    """block: optional (D, N) bool, True = no aim on that session (earnings windows). shuffle: permute the stocks'
    aims among valid stocks each session (placebo)."""
    rng = np.random.default_rng(seed)
    c, tau = gp_rate(q)
    o, cl = panel.o, panel.c
    ds = panel.day_start
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    ncol = cl.shape[1]
    etfs = np.unique(etf_cols)
    etf_pos = np.searchsorted(etfs, etf_cols)
    hold = np.zeros(ncol)
    cash = cfg.initial_equity
    last_px = np.full(ncol, np.nan)
    tot_cost = tot_traded = 0.0
    pend = None
    eq_bar = np.full(T, np.nan)
    gross_bar = np.zeros(T)
    first = int(np.argmax(np.isfinite(est["B_A"].to_numpy()))) if np.isfinite(est["B_A"]).any() else D
    eq_bar[:ds[min(first, D)]] = cfg.initial_equity
    eq_prev = cfg.initial_equity
    be_today = np.zeros(N)
    for t in range(first, D):
        BA, BB, phA, phB = est.iloc[t].to_numpy()
        be = P["beta"][t - 1]
        be_today = np.where(np.isfinite(be), be, 0.0)
        okday = sig["valid"][ds[t]].copy() if ds[t] < T else np.zeros(N, bool)
        if block is not None:
            okday &= ~block[t]
        perm = np.arange(N)
        if shuffle:
            vi = np.flatnonzero(okday)
            perm[vi] = rng.permutation(vi)
        for B in range(ds[t], ds[t + 1]):
            if pend is not None:
                px = np.where(np.isfinite(o[B]), o[B], last_px)
                ps = px[stock_cols]
                okp = np.isfinite(ps) & (ps > 0)
                dev = pend - np.where(okp, hold[stock_cols] * ps, 0.0)
                dq = np.zeros(ncol)
                tr = okp & (np.abs(dev) > 0.5 * ps)
                dq[stock_cols[tr]] = np.round(dev[tr] / ps[tr])
                new_s = (hold[stock_cols] + dq[stock_cols]) * np.where(okp, ps, 0.0)
                pe = px[etfs]
                etf_tgt = -np.bincount(etf_pos, weights=be_today * new_s, minlength=len(etfs))
                oke = np.isfinite(pe) & (pe > 0)
                dev_e = etf_tgt - np.where(oke, hold[etfs] * pe, 0.0)
                tre = oke & (np.abs(dev_e) > cfg.etf_band * eq_prev)
                dq[etfs[tre]] = np.round(dev_e[tre] / pe[tre])
                m = dq != 0
                if m.any():
                    hs = half_spread[B].astype(float)[m] * cfg.cost_mult
                    p_ = px[m]
                    cash -= np.sum(dq[m] * p_ * (1 + np.sign(dq[m]) * hs))
                    tot_cost += float(np.sum(np.abs(dq[m]) * p_ * hs))
                    tot_traded += float(np.sum(np.abs(dq[m]) * p_))
                    hold[m] += dq[m]
            row = cl[B]
            okc = np.isfinite(row)
            last_px[okc] = row[okc]
            val = np.where(hold != 0, hold * last_px, 0.0)
            eq_prev = cash + val.sum()
            eq_bar[B] = eq_prev
            gross_bar[B] = np.abs(val[stock_cols]).sum() / max(eq_prev, 1e-9)
            ok = okday & np.isfinite(sig["sA"][B]) & np.isfinite(sig["sB"][B]) & np.isfinite(sig["sz"][B])
            if np.isfinite(BA):
                aim = aim_weights(sig["sA"][B][perm], sig["sB"][B][perm], sig["sz"][B][perm], ok[perm] & ok,
                                  BA, BB, phA, phB, c, cfg.gross, cfg.w_cap, mode)
            else:
                aim = np.zeros(N)
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.where(np.isfinite(last_px[stock_cols]), hold[stock_cols] * last_px[stock_cols], 0.0) / eq_prev
            pend = (x + tau * (aim - x)) * eq_prev
    Bl = T - 1
    m = hold != 0
    if m.any():
        hs = half_spread[Bl].astype(float)[m] * cfg.cost_mult
        p_ = last_px[m]
        cash += np.sum(hold[m] * p_ * (1 - np.sign(hold[m]) * hs))
        tot_cost += float(np.sum(np.abs(hold[m]) * p_ * hs))
        tot_traded += float(np.sum(np.abs(hold[m]) * p_))
    eq_bar[Bl] = cash
    day_eq = pd.Series(eq_bar[ds[1:] - 1], index=pd.to_datetime(panel.days)).ffill().fillna(cfg.initial_equity)
    start = pd.Timestamp(panel.days[min(first, D - 1)])
    eq = day_eq[day_eq.index >= start]
    years = max(len(eq) / 252, 1e-9)
    return {"daily_equity": day_eq, "first_day": start, "total_cost": tot_cost, "total_traded": tot_traded,
            "annual_turnover_x_equity": tot_traded / float(eq.mean()) / years,
            "annual_cost_pct_equity": 100 * tot_cost / float(eq.mean()) / years,
            "avg_stock_gross": float(np.mean(gross_bar[ds[min(first, D - 1)]:]))}


def run_placebo(panel, P, sig, est, stock_cols, etf_cols, half_spread, cfg, q, block, seed, start) -> dict:
    r = backtest(panel, P, sig, est, stock_cols, etf_cols, half_spread, cfg, q, block=block, shuffle=True, seed=seed)
    return {"seed": seed, **perf_metrics(r["daily_equity"], cfg.initial_equity, start)}


def gates(boot: dict, ranks: dict, net2: dict, alpha: float) -> dict:
    g1 = all(v[1] > 0 for v in boot.values())
    g2 = ranks["total_return"]["p_upper"] <= alpha and ranks["sharpe"]["p_upper"] <= alpha
    g3 = net2["total_return"] > 0 and net2["sharpe"] > 0
    return {"G1_bootstrap": g1, "G2_vs_shuffled": g2, "G3_double_cost": g3, "pass": g1 and g2 and g3}
