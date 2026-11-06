"""P8: OU-optimal entry and exit rules on the S05 residual (Bertram 2010; Lipton & Lopez de Prado 2020).

Protocol: discussions/OU_Optimal_Thresholds/001_claude_p8_ou_optimal_thresholds_preregistration.md

In s-score units (ds = -kappa s dt + sqrt(2 kappa) dW), with round-trip cost c in units of sigma_eq:
  BT: symmetric band +-m; m* maximises (2m - c) / E[T], E[T] = 2 pi Erfi(m / sqrt 2) / kappa (kappa cancels).
  LP: entry |s| >= 1.25; take-profit / stop-loss / time limit 3 half-lives chosen per trade to maximise
      (E[ds] - c) / E[tau] from a simulated first-passage table (kappa-time).
backtest() is the S05 engine (signals at bar closes, fills at the next open, 1% per stock, beta hedge in the sector
ETF, netted ETF trades, <= 100 positions, largest |s| first) with per-stock-day entry thresholds and per-position
take-profit, stop-loss and expiry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .s04 import Panel
from .s05 import S05Config, log_returns

P8_VERSION = "0.1.0"


@dataclass(frozen=True)
class P8Config:
    lp_entry: float = 1.25
    bt_cap: float = 5.0
    lp_horizon_halflives: float = 3.0
    tp_grid: tuple = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    sl_grid: tuple = (0.5, 1.0, 1.5, 2.0, np.inf)
    s0_grid: tuple = tuple(np.round(np.arange(1.25, 5.01, 0.25), 2))
    c_grid: tuple = tuple(np.round(np.geomspace(0.01, 3.0, 40), 5))
    sim_paths: int = 4000
    sim_dt: float = 0.01
    alpha: float = 0.025
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    seed: int = 42

    def as_dict(self) -> dict:
        d = asdict(self)
        d["sl_grid"] = [float(x) for x in self.sl_grid]
        return d


# =================================================================== Bertram
def bertram_rate(m, c):
    """(2m - c) / (2 pi Erfi(m / sqrt 2)): expected net s-units per unit kappa-time of the symmetric band."""
    from scipy.special import erfi
    m = np.asarray(m, float)
    return (2 * m - c) / (2 * np.pi * erfi(m / np.sqrt(2)))


def bertram_mstar(c: float) -> float:
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda m: -bertram_rate(m, c), bounds=(1e-4, 8.0), method="bounded", options={"xatol": 1e-5})
    return float(r.x)


def bertram_table(c_grid) -> pd.Series:
    return pd.Series([bertram_mstar(c) for c in c_grid], index=np.asarray(c_grid, float), name="m_star")


def lookup(table_index: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the nearest grid value (log scale for positive grids)."""
    g = np.log(table_index) if np.all(table_index > 0) else table_index
    xv = np.log(np.clip(x, table_index.min(), table_index.max())) if np.all(table_index > 0) else x
    return np.abs(xv[..., None] - g).argmin(-1)


# =================================================================== Lipton & Lopez de Prado (numerical)
def lp_table(cfg: P8Config) -> dict:
    """Simulated first-passage moments for a long trade entered at s0 = -|s0| (mirror for shorts), time in
    kappa units, horizon 3 half-lives = 3 ln 2. Returns E[ds] and E[tau] for every (s0, tp, sl)."""
    rng = np.random.default_rng(cfg.seed)
    T = cfg.lp_horizon_halflives * np.log(2)
    n = int(np.ceil(T / cfg.sim_dt))
    b = np.exp(-cfg.sim_dt)
    sd = np.sqrt(1 - b * b)
    S0, TP, SL = np.array(cfg.s0_grid), np.array(cfg.tp_grid), np.array(cfg.sl_grid)
    E_ds = np.zeros((len(S0), len(TP), len(SL)))
    E_t = np.zeros_like(E_ds)
    eps = rng.standard_normal((n, cfg.sim_paths))
    for i, s0 in enumerate(S0):
        path = np.empty((n + 1, cfg.sim_paths)); path[0] = -s0
        for k in range(n):
            path[k + 1] = b * path[k] + sd * eps[k]
        for j, tp in enumerate(TP):
            hit_tp = path >= tp
            for l, sl in enumerate(SL):
                hit = hit_tp | (path <= -s0 - sl) if np.isfinite(sl) else hit_tp
                hit[0] = False
                first = np.where(hit.any(0), hit.argmax(0), n)
                end = path[first, np.arange(cfg.sim_paths)]
                E_ds[i, j, l] = float(np.mean(end + s0))
                E_t[i, j, l] = float(np.mean(first * cfg.sim_dt))
    return {"E_ds": E_ds, "E_tau": E_t, "s0": S0, "tp": TP, "sl": SL}


def lp_choice(tab: dict, c_grid) -> dict:
    """For each (s0, c): the (tp, sl) maximising (E[ds] - c) / E[tau]; also that rate."""
    c = np.asarray(c_grid, float)
    rate = (tab["E_ds"][None] - c[:, None, None, None]) / np.maximum(tab["E_tau"][None], 1e-9)   # (C, S0, TP, SL)
    flat = rate.reshape(len(c), len(tab["s0"]), -1)
    best = flat.argmax(-1)
    tp_i, sl_i = np.unravel_index(best, (len(tab["tp"]), len(tab["sl"])))
    return {"tp": tab["tp"][tp_i], "sl": tab["sl"][sl_i], "rate": flat.max(-1), "c": c, "s0": tab["s0"]}


# =================================================================== costs in sigma units
def cost_sigma_units(P: dict, H: np.ndarray, panel: Panel, stock_cols, etf_cols) -> np.ndarray:
    """(D, N) round-trip cost of a residual trade in units of sigma_eq, row e = estimate at the close of e (used on e+1):
    2 (h_stock + |beta| h_etf) / sigma_eq, h = median one-way half-spread over session e+1's bars."""
    ds = panel.day_start
    D, N = panel.n_days, len(stock_cols)
    out = np.full((D, N), np.nan)
    for e in range(D - 1):
        a, b = ds[e + 1], ds[e + 2]
        hs = np.median(H[a:b], 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[e] = 2 * (hs[stock_cols] + np.abs(P["beta"][e]) * hs[etf_cols]) / P["sig"][e]
    return out


# =================================================================== backtest
def backtest(panel: Panel, P: dict, stock_cols, etf_cols, cfg: S05Config, rule: str, half_spread: np.ndarray,
             ctilde: np.ndarray, pcfg: P8Config, bt_tab: pd.Series | None = None, lp_ch: dict | None = None,
             entry_mask: np.ndarray | None = None, cost_mult: float = 1.0, record: bool = True) -> dict:
    """rule 'BT' or 'LP'. Same accounting as s05.backtest; exit reasons: 1 take-profit/band, 2 invalid, 3 end,
    4 stop-loss, 5 time limit."""
    r = log_returns(panel)
    o, c = panel.o, panel.c
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    ncol = c.shape[1]
    hold = np.zeros(ncol)
    cash = cfg.initial_equity
    last_px = np.full(ncol, np.nan)
    side = np.zeros(N, np.int8)
    sh_s, sh_h = np.zeros(N), np.zeros(N)
    ent_bar = np.zeros(N, np.int64)
    ent_ps, ent_pe, ent_s = np.zeros(N), np.zeros(N), np.zeros(N)
    tp_l, sl_l, exp_b = np.zeros(N), np.zeros(N), np.zeros(N, np.int64)
    pend = np.zeros(N, np.int8); pend_reason = np.zeros(N, np.int8); pend_s = np.zeros(N)
    pend_tp, pend_sl, pend_T = np.zeros(N), np.zeros(N), np.zeros(N, np.int64)
    trades = []
    eq_bar = np.full(T, np.nan)
    total_cost = 0.0
    first = cfg.window_sessions + 1
    eq_bar[:panel.day_start[first]] = cfg.initial_equity
    eq_prev = cfg.initial_equity
    beta_today = np.zeros(N)
    hs = 0.0

    def execute(delta, px):
        nonlocal cash, total_cost
        m = delta != 0
        if m.any():
            d_, p_ = delta[m], px[m]
            h = hs[m] if np.ndim(hs) else hs
            cash -= np.sum(d_ * p_ * (1 + np.sign(d_) * h))
            total_cost += np.sum(np.abs(d_) * p_ * h)
            hold[m] += d_

    for t in range(first, D):
        e = t - 1
        al, be, mc, sg, vd, tau = (P[k][e] for k in ("alpha", "beta", "m", "sig", "valid", "tau"))
        beta_today = np.where(np.isfinite(be), be, beta_today)
        ct = ctilde[e]
        if rule == "BT":
            mi = bt_tab.to_numpy()[lookup(bt_tab.index.to_numpy(), np.nan_to_num(ct, nan=1e9))]
            open_thr = np.where(np.isfinite(ct) & (mi <= pcfg.bt_cap), mi, np.inf)
        else:
            open_thr = np.full(N, pcfg.lp_entry)
        hl_bars = np.where(np.isfinite(tau), pcfg.lp_horizon_halflives * np.log(2) * tau, 1e9)
        X = np.zeros(N)
        for B in range(panel.day_start[t], panel.day_start[t + 1]):
            hs = half_spread[B].astype(float) * cost_mult
            if pend.any():
                px_open = np.where(np.isfinite(o[B]), o[B], last_px)
                ps, pe = px_open[stock_cols], px_open[etf_cols]
                delta = np.zeros(ncol)
                cl = (pend == 2) & np.isfinite(ps) & np.isfinite(pe)
                if cl.any():
                    idx = np.where(cl)[0]
                    np.add.at(delta, stock_cols[idx], -sh_s[idx]); np.add.at(delta, etf_cols[idx], -sh_h[idx])
                    if record:
                        gp = sh_s[idx] * (ps[idx] - ent_ps[idx]) + sh_h[idx] * (pe[idx] - ent_pe[idx])
                        for k, i in enumerate(idx):
                            trades.append((int(i), int(side[i]), int(ent_bar[i]), B, float(ent_s[i]), float(gp[k]),
                                           float(abs(sh_s[i]) * ent_ps[i]), int(pend_reason[i])))
                    side[idx] = 0; sh_s[idx] = 0; sh_h[idx] = 0; pend[idx] = 0
                op = ((pend == 1) | (pend == -1)) & np.isfinite(ps) & np.isfinite(pe) & (pe > 0)
                if op.any():
                    idx = np.where(op)[0]
                    n_sh = np.floor(cfg.size_per_stock * eq_prev / ps[idx])
                    good = n_sh >= 1
                    idx, n_sh = idx[good], n_sh[good]
                    dirs = pend[idx].astype(float)
                    s_sh = dirs * n_sh
                    h_sh = -np.round(beta_today[idx] * s_sh * ps[idx] / pe[idx])
                    np.add.at(delta, stock_cols[idx], s_sh); np.add.at(delta, etf_cols[idx], h_sh)
                    side[idx] = dirs.astype(np.int8); sh_s[idx], sh_h[idx] = s_sh, h_sh
                    ent_bar[idx], ent_ps[idx], ent_pe[idx], ent_s[idx] = B, ps[idx], pe[idx], pend_s[idx]
                    tp_l[idx], sl_l[idx], exp_b[idx] = pend_tp[idx], pend_sl[idx], B + pend_T[idx]
                pend[(pend == 1) | (pend == -1)] = 0
                execute(delta, px_open)
            row = c[B]
            okp = np.isfinite(row)
            last_px[okp] = row[okp]
            rb = r[B]
            eps = rb[stock_cols] - al - be * rb[etf_cols]
            X = X + np.where(np.isfinite(eps), eps, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                s = (X - mc) / sg
            fin = np.isfinite(s)
            lg, st = side > 0, side < 0
            tp_hit = fin & ((lg & (s >= tp_l)) | (st & (s <= tp_l)))
            sl_hit = fin & ((lg & (s <= sl_l)) | (st & (s >= sl_l))) & ~tp_hit
            tm_hit = (side != 0) & (B >= exp_b) & ~tp_hit & ~sl_hit
            inv = (side != 0) & ~vd
            for m_, code in ((tp_hit, 1), (sl_hit, 4), (tm_hit, 5), (inv & ~tp_hit & ~sl_hit & ~tm_hit, 2)):
                sel = m_ & (pend != 2)
                pend_reason[sel] = code
                pend[sel] = 2
            flat = (side == 0) & (pend == 0) & vd & fin
            if entry_mask is not None:
                flat &= entry_mask[e]
            sig_m = flat & (np.abs(s) >= open_thr)
            if sig_m.any():
                free = cfg.max_positions - int(((side != 0) & (pend != 2)).sum())
                if free > 0:
                    cand = np.where(sig_m)[0]
                    if len(cand) > free:
                        cand = cand[np.argsort(-np.abs(s[cand]), kind="stable")[:free]]
                    dirn = np.where(s[cand] > 0, -1, 1)
                    pend[cand] = dirn
                    pend_s[cand] = s[cand]
                    if rule == "BT":
                        pend_tp[cand] = dirn * open_thr[cand]                    # long exits at +m*, short at -m*
                        pend_sl[cand] = -dirn * np.inf
                        pend_T[cand] = 10 ** 9
                    else:
                        a0 = np.abs(s[cand])
                        ci = lookup(lp_ch["c"], np.nan_to_num(ct[cand], nan=lp_ch["c"].max()))
                        si = lookup(lp_ch["s0"], a0)
                        tp, sl = lp_ch["tp"][ci, si], lp_ch["sl"][ci, si]
                        pend_tp[cand] = dirn * tp                                # long: s >= tp; short: s <= -tp
                        pend_sl[cand] = s[cand] - dirn * sl                      # beyond the entry, away from the mean
                        pend_T[cand] = np.ceil(hl_bars[cand]).astype(np.int64)
            val = np.where(hold != 0, hold * last_px, 0.0)
            eq_prev = cash + val.sum()
            eq_bar[B] = eq_prev
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
    out = {"daily_equity": day_eq, "total_cost": total_cost, "first_day": pd.Timestamp(panel.days[min(first, D - 1)])}
    if record:
        tr = pd.DataFrame(trades, columns=["i", "side", "entry_bar", "exit_bar", "s_entry", "gross_pnl", "notional", "reason_code"])
        tr["reason"] = tr["reason_code"].map({1: "take_profit", 2: "invalid", 3: "end", 4: "stop_loss", 5: "time_limit"})
        tr["bars"] = tr["exit_bar"] - tr["entry_bar"]
        tr["gross_ret"] = tr["gross_pnl"] / tr["notional"]
        out["trades"] = tr
    return out


def paired_bootstrap(a: pd.Series, b: pd.Series, block: int, n_boot: int, seed: int, alpha: float):
    from .p1 import block_bootstrap
    d = (a - b).dropna().to_numpy()
    return block_bootstrap(d, block, n_boot, seed, alpha)


def gates(boot_net: dict, boot_diff: dict, net2: dict) -> dict:
    g1 = all(v[1] > 0 for v in boot_net.values())
    g2 = all(v[1] > 0 for v in boot_diff.values())
    g3 = net2["total_return"] > 0 and net2["sharpe"] > 0
    return {"G1_edge": g1, "G2_improvement": g2, "G3_double_cost": g3}
