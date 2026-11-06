"""P7: Li & Papanicolaou (2024) - statistical arbitrage in rank space (data, ranks, evaluation).

Protocol: discussions/Rank_Space/001_claude_p7_rank_space_preregistration.md

Capitalisation = split-adjusted (not dividend-adjusted) price x shares outstanding (SEC XBRL, point in time by
filing date, converted to today's split basis with Yahoo split history). Each close, the K largest caps define
ranks 1..K. Rank return r_rank[t, k] = cap_(k),t / cap_(k),t-1 - 1 (not tradable). A rank weight formed at the
close of d is held in the stock that holds that rank at the close of d + 1 (the trade day) and earns that stock's
total return on d + 2. The network and its training are in p7_model.py; residuals and inputs reuse p6.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

P7_VERSION = "0.1.0"
SHARE_TAGS = (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
              ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"))


@dataclass(frozen=True)
class P7Config:
    n_ranks: int = 400
    n_factors: int = 1
    pca_window: int = 252
    lookback: int = 30
    train_days: int = 1000
    valid_days: int = 250
    refit_every: int = 125
    first_test: str = "2017-01-03"
    end: str = "2025-12-31"
    data_start: str = "2011-01-01"
    splice: str = "2016-01-04"
    skip: int = 1
    seeds: tuple = (0, 1, 2)
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    chunk_days: int = 50
    gamma: float = 2.0
    unit_band: tuple = (0.5, 2.0)
    alpha: float = 0.025
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    n_placebo: int = 1000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== downloads
def _cik_map() -> dict:
    import requests
    from .data import vendor_dir
    from .p4 import SEC_TICKERS, _sec_get
    p = vendor_dir("SEC") / "company_tickers.json"
    if not p.exists():
        p.write_text(json.dumps(_sec_get(SEC_TICKERS, requests.Session())))
    return {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(p.read_text()).values()}


def sec_shares(symbols: list[str], verbose: bool = True) -> pd.DataFrame:
    """[symbol, cik, tag, end, val, filed] share counts from SEC XBRL company concepts (first tag with data).
    Cached per CIK under <data>/SEC/shares/."""
    import requests
    from .data import vendor_dir
    from .p4 import _sec_get, sec_symbol
    d = vendor_dir("SEC") / "shares"
    d.mkdir(parents=True, exist_ok=True)
    cmap, sess, out, missing, t0 = _cik_map(), requests.Session(), [], [], time.time()
    for k, s in enumerate(symbols, 1):
        cik = cmap.get(sec_symbol(s))
        if cik is None:
            missing.append(s)
            continue
        p = d / f"CIK{cik:010d}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
        else:
            df = pd.DataFrame(columns=["tag", "end", "val", "filed"])
            for tax, tag in SHARE_TAGS:
                js = _sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{tax}/{tag}.json", sess)
                rows = ((js or {}).get("units") or {}).get("shares") or []
                if rows:
                    df = pd.DataFrame([(tag, r.get("end"), r.get("val"), r.get("filed")) for r in rows],
                                      columns=["tag", "end", "val", "filed"])
                    break
            tmp = p.with_suffix(".parquet.tmp"); df.to_parquet(tmp, index=False); os.replace(tmp, p)
        out.append(df.assign(symbol=s, cik=cik))
        if verbose and (k % 50 == 0 or k == len(symbols)):
            print(f"  SEC shares {k}/{len(symbols)}, {time.time() - t0:.0f}s", flush=True)
    if verbose and missing:
        print(f"  no CIK for {len(missing)}: {missing[:15]}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "cik", "tag", "end", "val", "filed"])


def yahoo_splits_and_shares(tickers: list[str], verbose: bool = True) -> tuple[dict, dict]:
    """({ticker: Series of split ratios by date}, {ticker: current share count or NaN}), cached under <data>/Yahoo/."""
    import yfinance as yf
    from .data import vendor_dir
    d = vendor_dir("Yahoo") / "splits"
    d.mkdir(parents=True, exist_ok=True)
    cur_path = vendor_dir("Yahoo") / "current_shares.json"
    cur = json.loads(cur_path.read_text()) if cur_path.exists() else {}
    splits = {}
    for k, t in enumerate(tickers, 1):
        y = t.replace(".", "-")
        p = d / f"{y}.parquet"
        if not p.exists() or t not in cur:
            tk = yf.Ticker(y)
            try:
                s = tk.splits
                s = s[s > 0]
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            except Exception:
                s = pd.Series(dtype=float)
            pd.DataFrame({"ratio": s.astype(float)}).to_parquet(p)
            try:
                cur[t] = float(tk.fast_info["shares"])
            except Exception:
                cur[t] = float("nan")
            cur_path.write_text(json.dumps(cur))
            time.sleep(0.2)
            if verbose and k % 50 == 0:
                print(f"  Yahoo splits/shares {k}/{len(tickers)}", flush=True)
        splits[t] = pd.read_parquet(p)["ratio"]
    return splits, {t: cur.get(t, float("nan")) for t in tickers}


def yahoo_split_prices(tickers: list[str], start: str, end: str, batch: int = 50, verbose: bool = True) -> pd.DataFrame:
    """Split-adjusted, not dividend-adjusted, daily closes (yfinance auto_adjust=False 'Close'), cached per ticker."""
    import yfinance as yf
    from .data import vendor_dir
    d = vendor_dir("Yahoo") / "daily_split"
    d.mkdir(parents=True, exist_ok=True)
    ys = {t: t.replace(".", "-") for t in tickers}
    todo = [t for t in tickers if not (d / f"{ys[t]}.parquet").exists()]
    if verbose:
        print(f"Yahoo split-only prices: {len(tickers) - len(todo)}/{len(tickers)} cached; downloading {len(todo)}", flush=True)
    for i in range(0, len(todo), batch):
        ch = todo[i:i + batch]
        df = yf.download([ys[t] for t in ch], start=start, end=end, auto_adjust=False, progress=False, group_by="column", threads=True)
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": ys[ch[0]]})
        for t in ch:
            s = close[ys[t]] if ys[t] in close.columns else pd.Series(dtype=float)
            o = pd.DataFrame({"close": s.astype(float)})
            o.index = pd.DatetimeIndex(o.index).tz_localize(None).normalize()
            tmp = d / f"{ys[t]}.parquet.tmp"; o.to_parquet(tmp); os.replace(tmp, d / f"{ys[t]}.parquet")
        time.sleep(1.0)
    cols = {}
    for t in tickers:
        p = d / f"{ys[t]}.parquet"
        if p.exists():
            s = pd.read_parquet(p)["close"]
            if len(s):
                cols[t] = s
    return pd.DataFrame(cols).sort_index()


def alpaca_split_daily(symbols: list[str], years: list[int], batch: int = 50, verbose: bool = True) -> pd.DataFrame:
    """Alpaca SIP 1Day closes with adjustment='split' (session date index), cached under <data>/Alpaca/bars_split/1Day/."""
    import requests
    from .alpaca import fetch_bars
    from .data import vendor_dir
    NY = "America/New_York"
    d = vendor_dir("Alpaca") / "bars_split" / "1Day"
    d.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    for y in years:
        todo = [s for s in symbols if not (d / s / f"{y}.parquet").exists()]
        for i in range(0, len(todo), batch):
            ch = todo[i:i + batch]
            got = fetch_bars(ch, "1Day", f"{y}-01-01", f"{y + 1}-01-01", adjustment="split", session=sess)
            for s in ch:
                (d / s).mkdir(parents=True, exist_ok=True)
                df = got.get(s, pd.DataFrame(columns=["t", "c"]))
                tmp = d / s / f"{y}.parquet.tmp"; df.to_parquet(tmp, index=False); os.replace(tmp, d / s / f"{y}.parquet")
        if verbose:
            print(f"  Alpaca split-adjusted 1Day {y}: {len(todo)} symbols downloaded", flush=True)
    cols = {}
    for s in symbols:
        parts = [pd.read_parquet(d / s / f"{y}.parquet") for y in years if (d / s / f"{y}.parquet").exists()]
        parts = [p for p in parts if len(p)]
        if parts:
            b = pd.concat(parts)
            idx = pd.DatetimeIndex(pd.to_datetime(b["t"], utc=True)).tz_convert(NY).normalize().tz_localize(None)
            cols[s] = pd.Series(b["c"].to_numpy(float), index=idx)
    return pd.DataFrame(cols).sort_index()


# =================================================================== capitalisation and ranks
def shares_daily(records: pd.DataFrame, splits: dict, current: dict, sessions: pd.DatetimeIndex,
                 band: tuple = (0.5, 2.0)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sessions x symbols) point-in-time shares on today's split basis, and a per-symbol diagnostics frame.
    One symbol per CIK (first alphabetically). Rescaled by the Yahoo current count when the ratio is outside band."""
    out, diag = {}, []
    if not len(records):
        return pd.DataFrame(index=sessions), pd.DataFrame()
    keep = records.groupby("cik")["symbol"].min()
    recs = records[records["symbol"].isin(set(keep))].dropna(subset=["val", "filed", "end"])
    for s, g in recs.groupby("symbol"):
        g = g.copy()
        g["end"], g["filed"] = pd.to_datetime(g["end"]), pd.to_datetime(g["filed"])
        sp = splits.get(s, pd.Series(dtype=float))
        g["adj"] = g.apply(lambda r: float(r["val"]) * float(np.prod(sp[sp.index > r["end"]].to_numpy())) if len(sp) else float(r["val"]), axis=1)
        g = g.sort_values(["filed", "end"]).drop_duplicates("filed", keep="last")
        g = g[g["adj"] > 0]
        if g.empty:
            continue
        pos = np.searchsorted(g["filed"].values, sessions.values, side="right") - 1
        vals = np.where(pos >= 0, g["adj"].to_numpy()[np.clip(pos, 0, None)], np.nan)
        cur = current.get(s, np.nan)
        ratio = cur / g["adj"].iloc[-1] if np.isfinite(cur) and cur > 0 else np.nan
        scaled = bool(np.isfinite(ratio) and not (band[0] <= ratio <= band[1]))
        if scaled:
            vals = vals * ratio
        out[s] = vals
        diag.append({"symbol": s, "tag": g["tag"].iloc[-1], "filings": len(g), "first_filed": g["filed"].iloc[0].date(),
                     "latest_adj": g["adj"].iloc[-1], "yahoo_current": cur, "ratio": ratio, "rescaled": scaled})
    return pd.DataFrame(out, index=sessions), pd.DataFrame(diag)


def rank_arrays(cap: pd.DataFrame, valid: pd.DataFrame, K: int) -> dict:
    """stock_of_rank (D, K) column index (-1 if fewer than K valid that day) and sorted caps (D, K)."""
    C = cap.to_numpy(float)
    V = valid.to_numpy(bool) & np.isfinite(C) & (C > 0)
    D = len(C)
    sor = np.full((D, K), -1, np.int32)
    sc = np.full((D, K), np.nan)
    for t in range(D):
        idx = np.flatnonzero(V[t])
        if len(idx) < K:
            continue
        top = idx[np.argsort(-C[t, idx], kind="stable")[:K]]
        sor[t], sc[t] = top, C[t, top]
    return {"stock_of_rank": sor, "sorted_cap": sc}


def rank_returns(sorted_cap: np.ndarray) -> np.ndarray:
    """r_rank[t, k] = cap_(k),t / cap_(k),t-1 - 1."""
    r = np.full(sorted_cap.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = sorted_cap[1:] / sorted_cap[:-1] - 1
    return r


def build_rank_dataset(R: pd.DataFrame, cap: pd.DataFrame, cost: pd.DataFrame, cfg: P7Config,
                       start_row: int | None = None, verbose: bool = True) -> dict:
    """Decision-row arrays in rank coordinates. R: stock total returns (sessions x stocks), cap aligned to R."""
    from .p6 import inputs, pca_residuals
    D, N = R.shape
    Rn = R.to_numpy(float)
    nxt = np.vstack([Rn[1:], np.full((1, N), np.nan)])
    valid = pd.DataFrame(np.isfinite(Rn) & np.isfinite(nxt), index=R.index, columns=R.columns)
    ra = rank_arrays(cap.reindex(index=R.index, columns=R.columns), valid, cfg.n_ranks)
    rr = rank_returns(ra["sorted_cap"])
    RR = pd.DataFrame(rr, index=R.index)
    pr = pca_residuals(RR, cfg.pca_window, cfg.n_factors, start_row, verbose)
    x = inputs(pr["resid"], cfg.lookback)
    h, s = cfg.skip + 1, cfg.skip
    K = cfg.n_ranks
    target = np.full((D, K), np.nan, np.float32); target[:D - h] = rr[h:]
    smap = np.full((D, K), -1, np.int32); smap[:D - s] = ra["stock_of_rank"][s:]
    tret = np.full((D, K), np.nan, np.float32)
    for d in range(D - h):
        m = smap[d] >= 0
        tret[d, m] = Rn[d + h, smap[d, m]]
    c = np.full((D, N), np.nan, np.float32)
    c[:D - s] = cost.reindex(index=R.index, columns=R.columns).to_numpy(np.float32)[s:]
    X = (pr["sigma"][..., None] * pr["V"]).astype(np.float32)
    ok = np.isfinite(x).all(-1) & np.isfinite(pr["sigma"]) & np.isfinite(target) & np.isfinite(tret) & (smap >= 0)
    return {"x": x, "X": X, "ret": target, "tret": tret, "smap": smap, "cost_stock": c, "ok": ok, "index": R.index,
            "columns": list(R.columns), "rank_return": rr.astype(np.float32), "resid": pr["resid"]}


def save_dataset(d: dict, path) -> None:
    np.savez(path, **{k: (v.values if k == "index" else np.array(v) if k == "columns" else v) for k, v in d.items()})


def load_dataset(path) -> dict:
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["index"], d["columns"] = pd.DatetimeIndex(d["index"]), list(d["columns"])
    return d


# =================================================================== evaluation
def to_stock(W: np.ndarray, smap: np.ndarray, n_stocks: int) -> np.ndarray:
    """(D, N) stock weights: rank weights placed on the stocks holding those ranks on the trade day."""
    S = np.zeros((len(W), n_stocks))
    for d in range(len(W)):
        m = (smap[d] >= 0) & (W[d] != 0)
        np.add.at(S[d], smap[d, m], W[d, m])
    return S


def evaluate(W: np.ndarray, d: dict, rows: np.ndarray, cost_mult: float = 1.0) -> pd.DataFrame:
    """Daily tradable gross / cost / net, non-tradable rank gross, and turnovers, indexed by return date."""
    N = len(d["columns"])
    S = to_stock(W[rows[0] - 1:rows[-1] + 1], d["smap"][rows[0] - 1:rows[-1] + 1], N)
    Wr = W[rows]
    prevW = W[rows - 1]
    gross = np.nansum(Wr * np.nan_to_num(d["tret"][rows]), 1)
    rank_gross = np.nansum(Wr * np.nan_to_num(d["ret"][rows]), 1)
    dS = np.abs(S[1:] - S[:-1])
    cost = np.nansum(dS * np.nan_to_num(d["cost_stock"][rows]), 1) * cost_mult
    when = [d["index"][min(r + 2, len(d["index"]) - 1)] for r in rows]
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "rank_gross": rank_gross,
                         "stock_turnover": dS.sum(1), "rank_turnover": np.abs(Wr - prevW).sum(1)},
                        index=pd.DatetimeIndex(when))


def gates(boot: dict, ranks: dict, net2: dict, alpha: float) -> dict:
    g1 = all(v[1] > 0 for v in boot.values())
    g2 = ranks["total_return"]["p_upper"] <= alpha and ranks["sharpe"]["p_upper"] <= alpha
    g3 = net2["total_return"] > 0 and net2["sharpe"] > 0
    return {"G1_bootstrap": g1, "G2_vs_permuted_gross": g2, "G3_double_cost": g3, "pass": g1 and g2 and g3}


def synthetic_market(n_days: int = 1500, n_stocks: int = 120, seed: int = 0, start: str = "2011-01-03") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(returns, caps): a market factor plus idiosyncratic shocks that partly reverse two sessions later."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n_days) * 0.01
    u = rng.standard_normal((n_days, n_stocks)) * 0.015
    e = u.copy(); e[2:] -= 0.3 * u[:-2]
    r = f[:, None] * rng.uniform(0.7, 1.3, n_stocks) + e
    idx = pd.bdate_range(start, periods=n_days)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    R = pd.DataFrame(r, index=idx, columns=cols)
    cap = pd.DataFrame(np.exp(rng.uniform(22, 26, n_stocks)) * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    return R, cap
