"""S06: S05a residual signal with passive (limit-order) execution on the stock leg.

Signal: unchanged from S05a (sector-ETF residual, 20-session window, s-score thresholds +/-1.25,
exits -0.50 / +0.75, 1% per stock, <= 100 positions). Costs: S05b measured half-spreads.

Execution model, per bar B (bar open o, high h, low l, measured half-spread hs of that symbol, year, slot):
  * Working stock orders are limit orders at the touch: buy L = o (1 - hs), sell L = o (1 + hs), i.e. the
    bid / ask implied by the bar open as mid.
  * Fill rule (trade-through): a buy fills only if the bar trades through the limit by one full quoted
    spread, l <= L - 2 hs o (sell: h >= L + 2 hs o). A quoted spread is at least one tick, so this means
    at least one tick through the price, a standard conservative stand-in for queue position. Bars with
    no trades (NaN high/low) never fill. All-or-nothing fills.
  * Entry orders live one bar; a still-signalling stock is re-posted at the next bar's open.
  * Exit orders are re-posted every bar; after `exit_patience` unfilled bars the stock is closed with a
    market order at the next open (pays hs).
  * The ETF hedge is a market order at the open after the stock fills (entry) or after the stock exit
    fills, so a position is unhedged for at most one bar. Hedges are netted per ETF.
Accounting: final - initial = signal P&L (all legs at bar-open reference prices) + spread captured on
passive fills - costs paid on market orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .s04 import Panel, perf_metrics
from .s05 import log_returns

S06_VERSION = "0.1.0"


@dataclass(frozen=True)
class S06Config:
    timeframe: str = "15Min"
    ffill_bars: int = 2
    window_sessions: int = 20
    min_bar_share: float = 0.95
    max_tau_frac: float = 0.5
    center_m: bool = True
    open_long: float = -1.25
    open_short: float = 1.25
    close_long: float = -0.50
    close_short: float = 0.75
    size_per_stock: float = 0.01
    max_positions: int = 100
    initial_equity: float = 100_000.0
    # execution
    fill_rule: str = "through"          # "through": trade through by one quoted spread | "touch": l <= L (optimistic)
    exit_patience: int = 4               # bars of unfilled passive exit before a market exit
    hedge_timing: str = "fill_bar_mid"   # ETF hedge price for a stock fill in bar B (fill time inside B is unknown):
                                         # "fill_bar_mid": (open + close) / 2 of the ETF in B (central, primary)
                                         # "fill_bar_open": ETF open of B (optimistic bound)
                                         # "next_open": ETF open of B + 1 (conservative bound)
    capture_mult: float = 1.0            # share of the half-spread booked on passive fills (stress: 0.5)
    cost_mult: float = 1.0               # multiplier on market-order costs (stress: 2)
    n_placebo: int = 1000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)

    def s05(self):
        """The matching S05Config for estimation (same signal parameters)."""
        from .s05 import S05Config
        return S05Config(window_sessions=self.window_sessions, min_bar_share=self.min_bar_share,
                         max_tau_frac=self.max_tau_frac, center_m=self.center_m)


def first_decision_day(cfg: S06Config) -> int:
    return cfg.window_sessions


def backtest(panel: Panel, params: dict, stock_cols: np.ndarray, etf_cols: np.ndarray, half_spread: np.ndarray,
             cfg: S06Config, mode: str = "strategy", entry_prob: float | None = None, seed: int = 0,
             record: bool = True) -> dict:
    """mode: 'strategy' | 'placebo_timing' (contrarian entry orders posted at random flat tradable bars
    with |s| beyond the exit band, at probability entry_prob)."""
    if panel.h is None or panel.l is None:
        raise ValueError("S06 needs a panel with raw high/low (rebuild it with s04.load_panel).")
    rng = np.random.default_rng(seed)
    r = log_returns(panel)
    o, c, hi, lo = panel.o, panel.c, panel.h, panel.l
    D, T, N = panel.n_days, len(panel.index), len(stock_cols)
    ncol = c.shape[1]
    hold = np.zeros(ncol)
    cash = cfg.initial_equity
    last_px = np.full(ncol, np.nan)
    # position state (per stock)
    side = np.zeros(N, np.int8)              # +1 long, -1 short, 0 flat
    sh_s = np.zeros(N)                       # stock shares held
    sh_h = np.zeros(N)                       # hedge shares currently held in the ETF for this position
    hedge_todo = np.zeros(N, np.int8)        # 1: place hedge at next open, -1: unwind hedge at next open
    hedge_beta = np.zeros(N)
    ref_s_in, ref_e_in = np.zeros(N), np.zeros(N)
    ent_bar = np.zeros(N, np.int64)
    ent_s = np.zeros(N)
    ent_notional = np.zeros(N)
    exiting = np.zeros(N, bool)
    exit_wait = np.zeros(N, np.int32)
    exit_reason = np.zeros(N, np.int8)       # 1 rule, 2 invalid
    stock_pnl = np.zeros(N)                  # realized stock-leg signal P&L of the current trade
    passive_exit = np.zeros(N, bool)
    want = np.zeros(N, np.int8)              # entry order to work this bar: +1 buy, -1 sell
    want_s = np.zeros(N)
    beta_today = np.zeros(N)
    trades = []
    eq_bar = np.full(T, np.nan)
    n_open = np.zeros(T, np.int16)
    gross = np.zeros(T)
    tot = {"cost": 0.0, "capture": 0.0, "entry_orders": 0, "entry_fills": 0, "exit_passive": 0, "exit_market": 0}
    elig_count = sig_count = 0
    eq_prev = cfg.initial_equity
    first = first_decision_day(cfg) + 1
    eq_bar[:panel.day_start[first]] = cfg.initial_equity
    cm, kap = cfg.cost_mult, cfg.capture_mult

    def market(delta: np.ndarray, px: np.ndarray, hs: np.ndarray):
        nonlocal cash
        m = (delta != 0) & np.isfinite(px)
        if m.any():
            d, p, h = delta[m], px[m], hs[m] * cm
            cash -= np.sum(d * p * (1 + np.sign(d) * h))
            tot["cost"] += float(np.sum(np.abs(d) * p * h))
            hold[m] += d

    def close_trade(i: int, B: int, hedge_pnl: float):
        if record:
            trades.append((i, int(side[i]), int(ent_bar[i]), B, float(ent_s[i]), float(stock_pnl[i] + hedge_pnl),
                           float(ent_notional[i]), int(exit_reason[i]), bool(passive_exit[i])))

    pending_close = {}                       # stock index -> stock-leg P&L awaiting hedge unwind
    for t in range(first, D):
        e = t - 1
        al, be, mc, sg, vd = (params[k][e] for k in ("alpha", "beta", "m", "sig", "valid"))
        beta_today = np.where(np.isfinite(be), be, beta_today)
        X = np.zeros(N)
        for B in range(panel.day_start[t], panel.day_start[t + 1]):
            hsB = half_spread[B].astype(float)
            px_open = np.where(np.isfinite(o[B]), o[B], last_px)
            ps, pe = px_open[stock_cols], px_open[etf_cols]
            hs_s = hsB[stock_cols]
            instant = cfg.hedge_timing != "next_open"
            if cfg.hedge_timing == "fill_bar_mid":
                ce = c[B][etf_cols]
                pe_h = np.where(np.isfinite(ce), 0.5 * (pe + ce), pe)
            else:
                pe_h = pe
            px_h = px_open.copy()
            delta = np.zeros(ncol)
            # ---------- at the open: hedge placements / unwinds, forced market exits
            if hedge_todo.any():
                put = np.where((hedge_todo == 1) & np.isfinite(pe) & (pe > 0))[0]
                if len(put):
                    hsh = -np.round(hedge_beta[put] * sh_s[put] * ref_s_in[put] / pe[put])
                    np.add.at(delta, etf_cols[put], hsh)
                    sh_h[put] = hsh
                    ref_e_in[put] = pe[put]
                    hedge_todo[put] = 0
                un = np.where((hedge_todo == -1) & np.isfinite(pe))[0]
                for i in un:
                    delta[etf_cols[i]] -= sh_h[i]
                    hp = sh_h[i] * (pe[i] - ref_e_in[i])
                    close_trade(i, B, hp)
                    sh_h[i] = 0
                    hedge_todo[i] = 0
                    side[i] = 0
            force = np.where(exiting & (exit_wait >= cfg.exit_patience) & np.isfinite(ps) & np.isfinite(pe))[0]
            for i in force:
                delta[stock_cols[i]] -= sh_s[i]
                delta[etf_cols[i]] -= sh_h[i]
                stock_pnl[i] = sh_s[i] * (ps[i] - ref_s_in[i])
                passive_exit[i] = False
                close_trade(i, B, sh_h[i] * (pe[i] - ref_e_in[i]) if hedge_todo[i] == 0 else 0.0)
                tot["exit_market"] += 1
                sh_s[i] = sh_h[i] = 0
                side[i] = 0
                exiting[i] = False
                hedge_todo[i] = 0
            market(delta, px_open, hsB)
            # ---------- during the bar: passive stock orders
            lB, hB = lo[B][stock_cols], hi[B][stock_cols]
            margin = 0.0 if cfg.fill_rule == "touch" else 1.0
            # exits
            ex = np.where(exiting & (side != 0) & np.isfinite(ps))[0]
            if len(ex):
                sell = sh_s[ex] > 0
                L = np.where(sell, ps[ex] * (1 + hs_s[ex]), ps[ex] * (1 - hs_s[ex]))
                thr = 2 * hs_s[ex] * ps[ex] * margin
                fill = np.where(sell, hB[ex] >= L + thr, lB[ex] <= L - thr) & np.isfinite(lB[ex]) & np.isfinite(hB[ex])
                for k in np.where(fill)[0]:
                    i = ex[k]
                    q = -sh_s[i]
                    cap = abs(q) * hs_s[i] * ps[i] * kap
                    cash -= q * ps[i] - cap           # booked at reference price minus captured spread
                    hold[stock_cols[i]] += q
                    tot["capture"] += cap
                    stock_pnl[i] = sh_s[i] * (ps[i] - ref_s_in[i])
                    passive_exit[i] = True
                    tot["exit_passive"] += 1
                    sh_s[i] = 0
                    exiting[i] = False
                    hedge_todo[i] = -1                # side stays set until the hedge is unwound
                    if instant and np.isfinite(pe_h[i]):
                        dlt = np.zeros(ncol); dlt[etf_cols[i]] = -sh_h[i]
                        px_h[etf_cols[i]] = pe_h[i]
                        market(dlt, px_h, hsB)
                        close_trade(i, B, sh_h[i] * (pe_h[i] - ref_e_in[i]))
                        sh_h[i], hedge_todo[i], side[i] = 0, 0, 0
                exit_wait[ex[~fill]] += 1
            # entries
            en = np.where((want != 0) & (side == 0) & np.isfinite(ps) & np.isfinite(pe) & (pe > 0))[0]
            tot["entry_orders"] += len(en)
            if len(en):
                buy = want[en] > 0
                L = np.where(buy, ps[en] * (1 - hs_s[en]), ps[en] * (1 + hs_s[en]))
                thr = 2 * hs_s[en] * ps[en] * margin
                fill = np.where(buy, lB[en] <= L - thr, hB[en] >= L + thr) & np.isfinite(lB[en]) & np.isfinite(hB[en])
                for k in np.where(fill)[0]:
                    i = en[k]
                    n_sh = np.floor(cfg.size_per_stock * eq_prev / ps[i])
                    if n_sh < 1:
                        continue
                    q = want[i] * n_sh
                    cap = n_sh * hs_s[i] * ps[i] * kap
                    cash -= q * ps[i] - cap
                    hold[stock_cols[i]] += q
                    tot["capture"] += cap
                    tot["entry_fills"] += 1
                    side[i], sh_s[i] = want[i], q
                    ref_s_in[i], ent_bar[i], ent_s[i], ent_notional[i] = ps[i], B, want_s[i], n_sh * ps[i]
                    hedge_beta[i], hedge_todo[i] = beta_today[i], 1
                    exiting[i], exit_wait[i], stock_pnl[i] = False, 0, 0.0
                    if instant and np.isfinite(pe_h[i]) and pe_h[i] > 0:
                        hsh = -np.round(hedge_beta[i] * sh_s[i] * ps[i] / pe_h[i])
                        dlt = np.zeros(ncol); dlt[etf_cols[i]] = hsh
                        px_h[etf_cols[i]] = pe_h[i]
                        market(dlt, px_h, hsB)
                        sh_h[i], ref_e_in[i], hedge_todo[i] = hsh, pe_h[i], 0
            want[:] = 0
            # ---------- close of bar: signals
            row = c[B]
            okp = np.isfinite(row)
            last_px[okp] = row[okp]
            rb = r[B]
            eps = rb[stock_cols] - al - be * rb[etf_cols]
            X = X + np.where(np.isfinite(eps), eps, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                s = (X - mc) / sg
            fin = np.isfinite(s)
            live = (side != 0) & (sh_s != 0) & ~exiting
            rule = live & fin & (((side > 0) & (s > cfg.close_long)) | ((side < 0) & (s < cfg.close_short)))
            inv = live & ~vd & ~rule
            exiting[rule | inv] = True
            exit_wait[rule | inv] = 0
            exit_reason[rule] = 1
            exit_reason[inv] = 2
            flat = (side == 0) & vd & fin
            elig = flat & ((s < cfg.close_long) | (s > cfg.close_short))
            if mode == "placebo_timing":
                sig_m = elig & (rng.random(N) < entry_prob)
            else:
                sig_m = flat & ((s < cfg.open_long) | (s > cfg.open_short))
            elig_count += int(elig.sum())
            sig_count += int(sig_m.sum())
            if sig_m.any():
                free = cfg.max_positions - int((side != 0).sum())
                if free > 0:
                    cand = np.where(sig_m)[0]
                    if len(cand) > free:
                        cand = cand[np.argsort(-np.abs(s[cand]), kind="stable")[:free]]
                    want[cand] = np.where(s[cand] > 0, -1, 1)
                    want_s[cand] = s[cand]
            # ---------- mark
            val = np.where(hold != 0, hold * last_px, 0.0)
            eq_prev = cash + val.sum()
            eq_bar[B] = eq_prev
            n_open[B] = int((side != 0).sum())
            gross[B] = np.abs(val).sum()
    # liquidate everything at the last close with market orders
    Bl = T - 1
    hsl = half_spread[Bl].astype(float)
    for i in np.where(side != 0)[0]:
        if sh_s[i] != 0:
            stock_pnl[i] = sh_s[i] * (last_px[stock_cols[i]] - ref_s_in[i])
        exit_reason[i] = 3
        passive_exit[i] = False
        close_trade(i, Bl, sh_h[i] * (last_px[etf_cols[i]] - ref_e_in[i]) if sh_h[i] != 0 else 0.0)
    market(-hold.copy(), last_px, hsl)
    eq_bar[Bl] = cash
    day_eq = pd.Series(eq_bar[panel.day_start[1:] - 1], index=pd.to_datetime(panel.days)).ffill().fillna(cfg.initial_equity)
    out = {"daily_equity": day_eq, "final_equity": float(day_eq.iloc[-1]), "entry_rate": sig_count / max(elig_count, 1),
           "first_day": pd.Timestamp(panel.days[min(first, D - 1)]), **{f"total_{k}": v for k, v in tot.items()}}
    if record:
        tr = pd.DataFrame(trades, columns=["i", "side", "entry_bar", "exit_bar", "s_entry", "signal_pnl", "notional",
                                           "reason_code", "passive_exit"])
        tr["reason"] = tr["reason_code"].map({1: "rule", 2: "invalid", 3: "end"})
        tr["bars"] = tr["exit_bar"] - tr["entry_bar"]
        tr["signal_ret"] = tr["signal_pnl"] / tr["notional"]
        tr["entry_time"] = panel.index[tr["entry_bar"].to_numpy()] if len(tr) else []
        out.update(trades=tr, bar_equity=pd.Series(eq_bar, index=panel.index), n_open=n_open, gross_exposure=gross)
    return out


def run_placebo(panel, params, stock_cols, etf_cols, half_spread, cfg: S06Config, entry_prob: float, seed: int, start) -> dict:
    r = backtest(panel, params, stock_cols, etf_cols, half_spread, cfg, mode="placebo_timing", entry_prob=entry_prob,
                 seed=seed, record=False)
    return {"seed": seed, **perf_metrics(r["daily_equity"], cfg.initial_equity, start)}
