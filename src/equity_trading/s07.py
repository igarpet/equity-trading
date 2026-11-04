"""S07: S05a residual signal traded as continuous weights with a no-trade band.

Signal: unchanged (sector-ETF residual, 20-session window, daily re-estimation, s-score updated every bar).
Portfolio at each bar close, over stocks tradable that day:
    raw_i = -clip(s_i, -s_cap, s_cap)                    (contrarian, saturating)
    w_i   = gross x raw_i / sum_j |raw_j|, then |w_i| <= w_cap      (fraction of equity, stock leg)
    ETF target (per ETF) = -sum over its stocks of beta_i x w_i       (sector hedge, netted)
Trading (market orders at the next bar open, S05b measured half-spreads):
    stocks: no-trade band, trade to the band edge (optimal policy shape under proportional costs,
    Davis & Norman 1990): if |target - current| > band, trade to target -/+ band; band =
    band_mult x gross / n_tradable (a multiple of the average position size), in equity terms.
    ETF hedges: after the stock trades, each ETF is set to -sum(beta_i x actual stock notional_i) of its
    stocks whenever it deviates by more than etf_band x equity (trade to target).
Stocks that stop being tradable have target 0 (subject to the same band).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .s04 import Panel, perf_metrics
from .s05 import log_returns

S07_VERSION = "0.1.1"


@dataclass(frozen=True)
class S07Config:
    timeframe: str = "15Min"
    ffill_bars: int = 2
    window_sessions: int = 20
    min_bar_share: float = 0.95
    max_tau_frac: float = 0.5
    center_m: bool = True
    s_cap: float = 3.0
    gross: float = 1.0                 # stock-leg gross exposure, fraction of equity
    w_cap: float = 0.02                # max |weight| per stock
    band_mult: float = 1.0             # band = band_mult x gross / n_tradable
    etf_band: float = 0.001            # ETF hedge rebalanced when off by > 0.1% of equity
    skip_open_bars: int = 0            # no trading at the opens of the first k bars of a session (S07b: 2 = 09:30-10:00)
    initial_equity: float = 100_000.0
    cost_mult: float = 1.0
    n_placebo: int = 1000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)

    def s05(self):
        from .s05 import S05Config
        return S05Config(window_sessions=self.window_sessions, min_bar_share=self.min_bar_share,
                         max_tau_frac=self.max_tau_frac, center_m=self.center_m)


def first_decision_day(cfg: S07Config) -> int:
    return cfg.window_sessions


def target_weights(s: np.ndarray, valid: np.ndarray, cfg: S07Config) -> np.ndarray:
    raw = np.where(valid & np.isfinite(s), -np.clip(s, -cfg.s_cap, cfg.s_cap), 0.0)
    tot = np.abs(raw).sum()
    if tot <= 0:
        return np.zeros_like(raw)
    return np.clip(cfg.gross * raw / tot, -cfg.w_cap, cfg.w_cap)


def backtest(panel: Panel, params: dict, stock_cols: np.ndarray, etf_cols: np.ndarray, half_spread: np.ndarray,
             cfg: S07Config, mode: str = "strategy", seed: int = 0, record: bool = True) -> dict:
    """mode 'strategy' | 'placebo_shuffle': each day, stocks receive the s-score path of a randomly permuted
    tradable stock (same distribution and dynamics of signals, link to the stock's own residual destroyed)."""
    rng = np.random.default_rng(seed)
    r = log_returns(panel)
    o, c = panel.o, panel.c
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    ncol = c.shape[1]
    etfs = np.unique(etf_cols)
    etf_pos = np.searchsorted(etfs, etf_cols)          # stock -> index into etfs
    hold = np.zeros(ncol)
    cash = cfg.initial_equity
    last_px = np.full(ncol, np.nan)
    tot_cost = tot_traded = 0.0
    pend = None                                        # stock target notionals for the next open
    eq_bar = np.full(T, np.nan)
    gross_bar = np.zeros(T)
    n_trades = 0
    first = first_decision_day(cfg) + 1
    eq_bar[:panel.day_start[first]] = cfg.initial_equity
    eq_prev = cfg.initial_equity
    stock_pnl_gross = 0.0
    for t in range(first, D):
        e = t - 1
        al, be, mc, sg, vd = (params[k][e] for k in ("alpha", "beta", "m", "sig", "valid"))
        perm = np.arange(N)
        if mode == "placebo_shuffle":
            vi = np.where(vd)[0]
            perm[vi] = rng.permutation(vi)
        bev = np.where(np.isfinite(be), be, 0.0)
        X = np.zeros(N)
        n_valid = max(int(vd.sum()), 1)
        band = cfg.band_mult * cfg.gross / n_valid
        for B in range(panel.day_start[t], panel.day_start[t + 1]):
            # ---------- fills at the open
            if pend is not None and B - panel.day_start[t] >= cfg.skip_open_bars:
                px = np.where(np.isfinite(o[B]), o[B], last_px)
                ps = px[stock_cols]
                okp_s = np.isfinite(ps) & (ps > 0)
                dev = pend - np.where(okp_s, hold[stock_cols] * ps, 0.0)
                bnd = band * eq_prev
                trade = okp_s & (np.abs(dev) > bnd)
                dq = np.zeros(ncol)
                dq[stock_cols[trade]] = np.round((dev[trade] - np.sign(dev[trade]) * bnd) / ps[trade])
                new_s = (hold[stock_cols] + dq[stock_cols]) * np.where(okp_s, ps, 0.0)
                pe = px[etfs]
                etf_tgt = -np.bincount(etf_pos, weights=bev * new_s, minlength=len(etfs))
                okp_e = np.isfinite(pe) & (pe > 0)
                dev_e = etf_tgt - np.where(okp_e, hold[etfs] * pe, 0.0)
                tr_e = okp_e & (np.abs(dev_e) > cfg.etf_band * eq_prev)
                dq[etfs[tr_e]] = np.round(dev_e[tr_e] / pe[tr_e])
                m = dq != 0
                if m.any():
                    hs = half_spread[B].astype(float)[m] * cfg.cost_mult
                    p_ = px[m]
                    cash -= np.sum(dq[m] * p_ * (1 + np.sign(dq[m]) * hs))
                    tot_cost += float(np.sum(np.abs(dq[m]) * p_ * hs))
                    tot_traded += float(np.sum(np.abs(dq[m]) * p_))
                    n_trades += int(m[stock_cols].sum())
                    hold[m] += dq[m]
            # ---------- close: signal and new targets
            row = c[B]
            okp = np.isfinite(row)
            last_px[okp] = row[okp]
            rb = r[B]
            eps = rb[stock_cols] - al - be * rb[etf_cols]
            X = X + np.where(np.isfinite(eps), eps, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                s = (X - mc) / sg
            s_used = s[perm]
            w = target_weights(s_used, vd, cfg)
            val = np.where(hold != 0, hold * last_px, 0.0)
            eq_prev = cash + val.sum()
            pend = w * eq_prev
            eq_bar[B] = eq_prev
            gross_bar[B] = np.abs(val).sum()
    # liquidate at the last close
    Bl = T - 1
    m = hold != 0
    if m.any():
        hs = half_spread[Bl].astype(float)[m] * cfg.cost_mult
        p_ = last_px[m]
        cash += np.sum(hold[m] * p_ * (1 - np.sign(hold[m]) * hs))
        tot_cost += float(np.sum(np.abs(hold[m]) * p_ * hs))
        tot_traded += float(np.sum(np.abs(hold[m]) * p_))
        hold[:] = 0
    eq_bar[Bl] = cash
    day_eq = pd.Series(eq_bar[panel.day_start[1:] - 1], index=pd.to_datetime(panel.days)).ffill().fillna(cfg.initial_equity)
    out = {"daily_equity": day_eq, "final_equity": float(day_eq.iloc[-1]), "total_cost": tot_cost,
           "total_traded": tot_traded, "stock_trades": n_trades,
           "first_day": pd.Timestamp(panel.days[min(first, D - 1)])}
    if record:
        out.update(bar_equity=pd.Series(eq_bar, index=panel.index), gross_exposure=gross_bar)
    return out


def turnover_stats(res: dict, start, initial: float) -> dict:
    eq = res["daily_equity"][res["daily_equity"].index >= pd.Timestamp(start)]
    years = max(len(eq) / 252, 1e-9)
    avg_eq = float(eq.mean())
    return {"annual_turnover_x_equity": res["total_traded"] / avg_eq / years,
            "annual_cost_pct_equity": 100 * res["total_cost"] / avg_eq / years,
            "stock_trades_per_day": res["stock_trades"] / max(len(eq), 1)}


def run_placebo(panel, params, stock_cols, etf_cols, half_spread, cfg: S07Config, seed: int, start) -> dict:
    r = backtest(panel, params, stock_cols, etf_cols, half_spread, cfg, mode="placebo_shuffle", seed=seed, record=False)
    return {"seed": seed, **perf_metrics(r["daily_equity"], cfg.initial_equity, start)}


def gates(boot: dict, shuffle: dict, cost2: dict, base: dict, alpha: float = 0.025) -> dict:
    """Pre-registered gates (discussions/Residual_Reversion/006_claude_s07_continuous_weights.md); alpha = 0.05 / 2 arms."""
    g1 = all(v[1] > 0 for v in boot.values())
    g2 = shuffle["total_return"]["p_upper"] <= alpha and shuffle["sharpe"]["p_upper"] <= alpha
    g3 = cost2["total_return"] > 0 and cost2["sharpe"] > 0
    stat = g1 and g2
    econ = base["total_return"] > 0 and g3
    return {"G1_bootstrap": g1, "G2_vs_shuffled_signal": g2, "G3_double_cost": g3,
            "statistical_verdict": "SIGNAL" if stat else ("PARTIAL" if (g1 or g2) else "NO_SIGNAL"),
            "economic_verdict": "PROFITABLE_AFTER_COSTS" if econ else "NOT_PROFITABLE_AFTER_COSTS",
            "verdict": "EDGE_CANDIDATE" if (stat and econ) else "NO_EDGE"}
