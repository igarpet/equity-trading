"""Next-bar MNQ return model on 5-minute bars (MRP 001): own history + 7 USD FX majors.

Pipeline (see discussions/MNQ_Return_Prediction/001_claude_mrp01_preregistration.md):

1. ``load_grid``      MNQ 1-minute bars (micro_bars) -> 5-minute CME-day grid with returns, realized variance, slots.
2. ``fx_on_grid``     Dukascopy FX mid -> 5-minute returns / realized variance on the same timestamps.
3. ``fit_vol`` / ``predict_vol``   time-of-day profile x HAR forecast of each bar's variance (ex ante).
4. ``build_dataset``  standardized regressors (groups A-G, dummies) and target z_{t+1}.
5. ``ols`` + ``cluster_cov`` / ``nw_cov`` + ``wald``   inference; ``walk_forward``  out-of-sample forecasts.

Conventions: timestamps are 5-minute bar OPEN times in UTC; bar t closes at ts + 5 min. A CME day runs 17:00 -> 16:00
Chicago (276 slots) and is labelled by the date it ends. Nothing from 2026 is read.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import fx_seasonality as FXS
from . import micro_bars as MB
from .futures import CHICAGO, NY, cme_trade_date

MRP_VERSION = "0.1.0"
LONDON = "Europe/London"
HOLDOUT_START = dt.date(2026, 1, 1)
BARS_PER_DAY = 276
MAJORS = FXS.MAJORS
SESSIONS = ("asia", "london", "ny", "post")
D = dt.date


class HoldoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class MRPConfig:
    root: str = "MNQ"
    warmup_start: dt.date = D(2019, 12, 1)
    train: tuple = (D(2020, 1, 1), D(2023, 6, 30))
    test: tuple = (D(2023, 7, 1), D(2025, 12, 31))
    refits_train: tuple = (D(2021, 1, 1), D(2022, 1, 1), D(2023, 1, 1))
    refits_test: tuple = (D(2023, 7, 1), D(2024, 1, 1), D(2025, 1, 1))
    halves: tuple = ((D(2020, 1, 1), D(2021, 9, 30)), (D(2021, 10, 1), D(2023, 6, 30)))
    pairs: tuple = MAJORS
    drop_first: int = 3                 # first bars of each CME day excluded as targets (17:00-17:15 Chicago)
    min_day_bars: int = 220             # CME days with fewer traded bars are excluded as targets
    gap_bars: int = 3                   # >= this many consecutive empty bars = gap (no return)
    fx_stale_minutes: int = 30          # FX mid older than this is treated as missing
    har: tuple = (("Qh", 12), ("Qd", 276), ("Qw", 5 * 276), ("Qm", 22 * 276))
    h_floor: float = 0.1
    n_lags: int = 12
    fx_lags: int = 3
    mom_threshold: float = 2.0
    tick: float = 0.25
    usd_per_point: float = 2.0
    commission_per_side: float = 0.50   # Lucid rate: Iván confirms before 03 runs
    nw_lags: int = 12


def guard(end: dt.date) -> None:
    if end >= HOLDOUT_START:
        raise HoldoutError(f"{end} reaches the sealed holdout (>= {HOLDOUT_START}).")


# ============================================================================= 1. MNQ grid

def _slot(ts: pd.DatetimeIndex) -> np.ndarray:
    loc = ts.tz_convert(CHICAGO)
    return (((loc.hour * 60 + loc.minute) - 17 * 60) % 1440 // 5).to_numpy()


def grid_from_1m(m1: pd.DataFrame, cfg: MRPConfig = MRPConfig()) -> pd.DataFrame:
    """1-minute bars (index ts UTC; close, volume, instrument_id) -> 5-minute CME-day grid.

    Columns: trade_date, slot, close (carried), volume, minutes, instrument_id, empty, ret (log, within day and
    contract, NaN after gaps/rolls), rv (sum of squared 1-minute log returns in the bar), n_bars_day.
    """
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index)
    td = pd.to_datetime(cme_trade_date(pd.Series(ts)).to_numpy())
    slot = _slot(ts)
    keep = slot < BARS_PER_DAY                                   # nothing should trade 16:00-17:00 Chicago
    m1, ts, td, slot = m1[keep], ts[keep], td[keep], slot[keep]
    lc = np.log(m1["close"].to_numpy(float))
    iid = m1["instrument_id"].to_numpy()
    same = np.r_[False, (td[1:] == td[:-1]) & (iid[1:] == iid[:-1])]
    r1 = np.where(same, np.r_[np.nan, np.diff(lc)], np.nan)
    b = pd.DataFrame({"trade_date": td, "slot": slot, "close": m1["close"].to_numpy(float),
                      "volume": m1["volume"].to_numpy(), "instrument_id": iid, "r1sq": r1 ** 2})
    g = b.groupby(["trade_date", "slot"], sort=True)
    agg = pd.DataFrame({"close": g["close"].last(), "volume": g["volume"].sum(), "minutes": g["close"].size(),
                        "instrument_id": g["instrument_id"].last(), "rv": g["r1sq"].sum(min_count=0)})
    # full grid: every slot of every CME day that traded
    days = agg.index.get_level_values(0).unique()
    full = pd.MultiIndex.from_product([days, np.arange(BARS_PER_DAY)], names=["trade_date", "slot"])
    agg = agg.reindex(full)
    agg["empty"] = agg["minutes"].isna()
    agg["minutes"] = agg["minutes"].fillna(0).astype(int)
    agg["volume"] = agg["volume"].fillna(0)
    agg["rv"] = agg["rv"].where(~agg["empty"], 0.0)
    gd = agg.groupby(level=0, sort=False)
    agg["close"] = gd["close"].ffill()
    agg["instrument_id"] = gd["instrument_id"].ffill()
    agg = agg.reset_index()
    start = pd.to_datetime(agg["trade_date"]) - pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    agg["ts"] = (start.dt.tz_localize(CHICAGO) + pd.to_timedelta(agg["slot"] * 5, unit="min")).dt.tz_convert("UTC")
    # gaps: runs of >= gap_bars empty bars inside a day (leading empties included)
    e = agg["empty"].to_numpy()
    run_id = np.cumsum(np.r_[True, (e[1:] != e[:-1]) | (agg["slot"].to_numpy()[1:] == 0)])
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    gap = e & (run_len >= cfg.gap_bars)
    lc5 = np.log(agg["close"].to_numpy(float))
    iid5 = agg["instrument_id"].to_numpy(float)
    first = agg["slot"].to_numpy() == 0
    prev_gap = np.r_[False, gap[:-1]]
    roll = np.r_[False, (iid5[1:] != iid5[:-1]) & np.isfinite(iid5[1:]) & np.isfinite(iid5[:-1])] & ~first
    ret = np.r_[np.nan, np.diff(lc5)]
    ret[first | gap | prev_gap | roll] = np.nan
    agg["ret"] = ret
    agg["rv"] = agg["rv"].where(~(gap | prev_gap | first), np.nan)
    agg["roll"] = roll
    agg["gap"] = gap
    agg["n_bars_day"] = agg.groupby("trade_date")["empty"].transform(lambda s: int((~s).sum()))
    agg["trade_date"] = agg["trade_date"].dt.date
    return agg.set_index("ts")[["trade_date", "slot", "close", "volume", "minutes", "instrument_id", "empty", "gap",
                                "roll", "ret", "rv", "n_bars_day"]]


def load_grid(start: dt.date, end: dt.date, cfg: MRPConfig = MRPConfig()) -> pd.DataFrame:
    """Grid for CME trade dates start..end (inclusive)."""
    guard(end)
    m1 = MB.load_1m(cfg.root, start - dt.timedelta(days=3), end + dt.timedelta(days=1))
    g = grid_from_1m(m1, cfg)
    return g[(g["trade_date"] >= start) & (g["trade_date"] <= end)]


# ============================================================================= 2. FX on the grid

def load_fx_minutes(pair: str, start: dt.date, end: dt.date) -> pd.Series:
    """Dukascopy 1-minute mid (close of each minute), index = minute start UTC, only minutes with quotes."""
    guard(end)
    files = FXS.day_files(pair, start, end)
    if not files:
        raise FileNotFoundError(f"No candle files for {pair} under {FXS.candles_root(pair)}")
    df = pd.concat([pd.read_parquet(f, columns=["ts", "bid_c", "ask_c", "bid_v", "ask_v"]) for f in files],
                   ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    quoted = (df["bid_v"].fillna(0) > 0) | (df["ask_v"].fillna(0) > 0)
    mid = 0.5 * (df["bid_c"] + df["ask_c"])
    return mid[quoted & (mid > 0)].rename(pair)


def fx_on_grid(mid: pd.Series, grid_index: pd.DatetimeIndex, stale_minutes: int = 30) -> pd.DataFrame:
    """5-minute FX log return and realized variance over exactly the MNQ bar [ts, ts+5min).

    ret = log mid(close of minute ts+4) - log mid(close of minute ts-1); rv = sum of the 5 squared 1-minute returns.
    """
    full = pd.date_range(mid.index[0], mid.index[-1], freq="1min", tz="UTC")
    lm = np.log(mid).reindex(full).ffill(limit=stale_minutes)
    r1 = lm.diff()
    ret5 = r1.rolling(5, min_periods=5).sum()
    rv5 = (r1 ** 2).rolling(5, min_periods=5).sum()
    at = grid_index + pd.Timedelta(minutes=4)                    # rolling window ending at minute ts+4
    return pd.DataFrame({"ret": ret5.reindex(at).to_numpy(), "rv": rv5.reindex(at).to_numpy()}, index=grid_index)


# ============================================================================= 3. Volatility model

@dataclass
class VolFit:
    s2_r: np.ndarray            # mean r^2 per slot
    s2_rv: np.ndarray           # mean rv per slot
    beta: np.ndarray            # HAR coefficients on [1, Qh, Qd, Qw, Qm]
    const_var: float            # benchmark: constant variance
    fit_end: dt.date
    n: int = 0


def _profile(x: np.ndarray, slot: np.ndarray, ok: np.ndarray) -> np.ndarray:
    s = pd.Series(x[ok]).groupby(slot[ok]).mean().reindex(range(BARS_PER_DAY))
    cnt = pd.Series(x[ok]).groupby(slot[ok]).size().reindex(range(BARS_PER_DAY)).fillna(0)
    overall = np.nanmean(x[ok])
    s = s.where(cnt >= 20, overall).fillna(overall)
    return np.maximum(s.to_numpy(float), overall * 1e-3)


def har_components(series: pd.DataFrame, s2_rv: np.ndarray, cfg: MRPConfig) -> pd.DataFrame:
    """Deseasonalized realized-variance means known at the close of each bar (rows in time order)."""
    q = series["rv"].to_numpy(float) / s2_rv[series["slot"].to_numpy()]
    qs = pd.Series(q)
    out = {name: qs.rolling(w, min_periods=max(2, w // 2)).mean().to_numpy() for name, w in cfg.har}
    return pd.DataFrame(out, index=series.index)


def fit_vol(series: pd.DataFrame, fit_start: dt.date, fit_end_excl: dt.date, cfg: MRPConfig = MRPConfig()) -> VolFit:
    """Fit on targets (bar t+1) with trade_date in [fit_start, fit_end_excl). ``series``: ret, rv, slot, trade_date."""
    td = series["trade_date"].to_numpy()
    slot = series["slot"].to_numpy()
    r = series["ret"].to_numpy(float)
    rv = series["rv"].to_numpy(float)
    win = (td >= fit_start) & (td < fit_end_excl)
    s2_r = _profile(r ** 2, slot, win & np.isfinite(r))
    s2_rv = _profile(rv, slot, win & np.isfinite(rv))
    Q = har_components(series, s2_rv, cfg).to_numpy()
    u2_next = np.r_[(r ** 2 / s2_r[slot])[1:], np.nan]
    X = np.column_stack([np.ones(len(Q)), Q])
    ok = np.r_[win[1:], False] & np.isfinite(u2_next) & np.isfinite(X).all(axis=1)
    beta = np.linalg.lstsq(X[ok], u2_next[ok], rcond=None)[0]
    return VolFit(s2_r, s2_rv, beta, float(np.nanmean((r ** 2)[win])), fit_end_excl, int(ok.sum()))


def predict_vol(series: pd.DataFrame, fit: VolFit, cfg: MRPConfig = MRPConfig()) -> pd.DataFrame:
    """Ex-ante variance of each bar (forecast at the previous row): full model, time-of-day only, constant."""
    slot = series["slot"].to_numpy()
    Q = har_components(series, fit.s2_rv, cfg).to_numpy()
    h = np.column_stack([np.ones(len(Q)), Q]) @ fit.beta
    h = np.maximum(h, cfg.h_floor)
    h_for = np.r_[np.nan, h[:-1]]                                # forecast made at t for bar t+1
    tod = fit.s2_r[slot]
    return pd.DataFrame({"sig2": tod * h_for, "sig2_tod": tod, "sig2_const": fit.const_var}, index=series.index)


def qlike(r: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    return np.log(sig2) + r ** 2 / sig2


def nw_mean_t(x: np.ndarray, lags: int) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m, se, t = FXS.nw_tstats(x.reshape(-1, 1), lags)
    return float(m[0]), float(se[0]), float(t[0])


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, day: np.ndarray, lags: int = 5) -> dict:
    """H1: model a has lower loss than b. Daily mean loss differences (b - a), Newey-West."""
    d = pd.Series(loss_b - loss_a).groupby(pd.Series(day)).mean()
    m, se, t = nw_mean_t(d.to_numpy(), lags)
    return {"mean_diff": m, "t": t, "p_one_sided": float(stats.norm.sf(t)), "days": len(d)}


def walk_forward_vol(series: pd.DataFrame, fit_start: dt.date, refits: tuple, end: dt.date,
                     cfg: MRPConfig = MRPConfig()) -> tuple[pd.DataFrame, list[VolFit]]:
    """Out-of-sample variance forecasts for trade dates in [refits[0], end]; one fit per refit (expanding)."""
    parts, fits = [], []
    edges = list(refits) + [end + dt.timedelta(days=1)]
    td = series["trade_date"].to_numpy()
    for a, b in zip(edges[:-1], edges[1:]):
        f = fit_vol(series, fit_start, a, cfg)
        fits.append(f)
        p = predict_vol(series, f, cfg)
        parts.append(p[(td >= a) & (td < b)])
    return pd.concat(parts).sort_index(), fits


def vol_gate(series: pd.DataFrame, pred: pd.DataFrame, cfg: MRPConfig = MRPConfig()) -> dict:
    """Gate V (MRP 001 §4) on out-of-sample forecasts ``pred`` aligned to ``series``."""
    s = series.loc[pred.index]
    ok = (np.isfinite(s["ret"]) & np.isfinite(pred["sig2"]) & (s["slot"] >= cfg.drop_first)
          & (s["n_bars_day"] >= cfg.min_day_bars)).to_numpy()
    r = s["ret"].to_numpy(float)[ok]
    day = s["trade_date"].to_numpy()[ok]
    L = {k: qlike(r, pred[c].to_numpy(float)[ok]) for k, c in
         (("full", "sig2"), ("tod", "sig2_tod"), ("const", "sig2_const"))}
    dm = diebold_mariano(L["full"], L["tod"], day)
    z = r / np.sqrt(pred["sig2"].to_numpy(float)[ok])
    tdd = pd.to_datetime(pd.Series(day))
    half = pd.Series(z).groupby([tdd.dt.year.to_numpy(), (tdd.dt.month.to_numpy() > 6).astype(int) + 1]).var()

    def acf_mean(x, lags=12):
        x = (x - x.mean()) / x.std()
        return float(np.mean([np.mean(x[l:] * x[:-l]) for l in range(1, lags + 1)]))
    acf_z2, acf_r2 = acf_mean(z ** 2), acf_mean(r ** 2)
    v1 = (L["full"].mean() < L["tod"].mean()) and (L["full"].mean() < L["const"].mean()) and dm["p_one_sided"] < 0.05
    v2 = bool(((half >= 0.8) & (half <= 1.25)).all())
    v3 = (acf_z2 < 0.03) and (acf_z2 < acf_r2 / 3)
    return {"qlike_full": float(L["full"].mean()), "qlike_tod": float(L["tod"].mean()),
            "qlike_const": float(L["const"].mean()), "dm_vs_tod": dm,
            "z_var_by_half": {f"{y}H{h}": float(v) for (y, h), v in half.items()},
            "acf_z2_mean_1_12": acf_z2, "acf_r2_mean_1_12": acf_r2,
            "V1": bool(v1), "V2": v2, "V3": bool(v3), "pass": bool(v1 and v2 and v3), "n": int(ok.sum())}


# ============================================================================= 4. Dataset

def session_of(ts: pd.DatetimeIndex) -> np.ndarray:
    """Session of each bar (by open time): asia, london, ny, post (MRP 001 §5)."""
    ny = ts.tz_convert(NY)
    ld = ts.tz_convert(LONDON)
    ny_m = (ny.hour * 60 + ny.minute).to_numpy()
    ld_m = (ld.hour * 60 + ld.minute).to_numpy()
    out = np.full(len(ts), "asia", dtype=object)
    london = (ld_m >= 480) & (ny_m < 570)
    out[london] = "london"
    out[(ny_m >= 570) & (ny_m < 960)] = "ny"
    out[(ny_m >= 960) & (ny_m < 1020)] = "post"
    return out


def _shift_in_day(x: np.ndarray, day: np.ndarray, k: int) -> np.ndarray:
    """x shifted by k rows (k > 0: past) within the same CME day; NaN otherwise."""
    out = np.full(len(x), np.nan)
    if k == 0:
        return x.astype(float).copy()
    same = np.zeros(len(x), bool)
    same[k:] = day[k:] == day[:-k]
    out[k:] = x[:-k]
    out[~same] = np.nan
    return out


def _session_sums(ts: pd.DatetimeIndex, day: np.ndarray, sess: np.ndarray, ret: np.ndarray, sig2: np.ndarray,
                  scale: str) -> pd.DataFrame:
    """One row per (CME day, session) plus the whole day: end time (close of last slot) and standardized return."""
    ok = np.isfinite(ret) & np.isfinite(sig2)
    df = pd.DataFrame({"ts": ts, "day": day, "sess": sess, "r": np.where(ok, ret, 0.0),
                       "s2": np.where(ok, sig2, 0.0), "ok": ok})
    rows = []
    for key, g in (("sess", df[df["sess"] != "post"].groupby(["day", "sess"], sort=False)),
                   ("day", df.groupby(["day"], sort=False))):
        a = g.agg(end=("ts", "max"), r=("r", "sum"), s2=("s2", "sum"), ok=("ok", "mean")).reset_index()
        if key == "day":
            a["sess"] = "day"
        rows.append(a)
    s = pd.concat(rows, ignore_index=True)
    s["end"] = s["end"] + pd.Timedelta(minutes=5)
    val = s["r"] / np.sqrt(s["s2"].where(s["s2"] > 0)) if scale == "z" else s["r"] * 1e4
    s["value"] = val.where(s["ok"] >= 0.5)
    return s


def build_dataset(grid: pd.DataFrame, sig2: pd.Series, fx: dict, fx_sig2: dict, cfg: MRPConfig = MRPConfig(),
                  scale: str = "z") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Regressors (rows = target bar t+1, information through bar t), target and meta.

    grid: load_grid output. sig2: ex-ante MNQ variance per bar (same index). fx: {pair: fx_on_grid frame}.
    fx_sig2: {pair: ex-ante variance series}. scale='z' standardizes by the forecasts; 'raw' uses log returns x 1e4.
    Returns (X, y, meta) restricted to eligible targets (all dates; filter by meta['trade_date']).
    """
    ts = grid.index
    day = grid["trade_date"].to_numpy()
    slot = grid["slot"].to_numpy()
    r = grid["ret"].to_numpy(float)
    s2 = sig2.reindex(ts).to_numpy(float)
    zz = r / np.sqrt(s2) if scale == "z" else r * 1e4
    sess = session_of(ts)
    cols: dict[str, np.ndarray] = {"const": np.ones(len(ts))}
    for s in ("london", "ny", "post"):
        cols[f"d_{s}"] = (sess == s).astype(float)
    # A own lags (lag 1 = bar t, the bar before the target)
    for k in range(1, cfg.n_lags + 1):
        cols[f"z_l{k}"] = np.nan_to_num(_shift_in_day(zz, day, k))
    # B large-move continuation
    s3 = (cols["z_l1"] + cols["z_l2"] + cols["z_l3"]) / np.sqrt(3)
    thr = cfg.mom_threshold if scale == "z" else cfg.mom_threshold * np.nanstd(zz) * 1.0
    cols["mom3"] = s3 * (np.abs(s3) > thr)
    # C, D completed sessions and previous CME day (end <= open of the target bar = close of bar t)
    ss = _session_sums(ts, day, sess, r, s2, scale)
    tgt = pd.DataFrame({"ts": ts}).reset_index(drop=True)
    for name in ("asia", "london", "ny", "day"):
        sub = ss[ss["sess"] == name][["end", "value"]].sort_values("end").dropna(subset=["end"])
        m = pd.merge_asof(tgt, sub, left_on="ts", right_on="end", direction="backward", allow_exact_matches=True)
        cols["prev_day" if name == "day" else f"sess_{name}"] = np.nan_to_num(m["value"].to_numpy(float))
    # E same slot on the previous CME day
    zs = pd.Series(zz, index=pd.MultiIndex.from_arrays([day, slot]))
    days = np.array(sorted(set(day)))
    prev_of = dict(zip(days[1:], days[:-1]))
    prev_day = np.array([prev_of.get(d) for d in day], dtype=object)
    lookup = zs[~zs.index.duplicated()]
    key = pd.MultiIndex.from_arrays([prev_day, slot])
    cols["same_slot_yday"] = np.nan_to_num(lookup.reindex(key).to_numpy(float))
    # F FX lags
    for p in cfg.pairs:
        fr = fx[p]["ret"].reindex(ts).to_numpy(float)
        fz = fr / np.sqrt(fx_sig2[p].reindex(ts).to_numpy(float)) if scale == "z" else fr * 1e4
        for k in range(1, cfg.fx_lags + 1):
            cols[f"{p}_l{k}"] = np.nan_to_num(_shift_in_day(fz, day, k))
    # G NY interactions
    dny = cols["d_ny"]
    cols["ny_x_z_l1"] = dny * cols["z_l1"]
    for p in cfg.pairs:
        cols[f"ny_x_{p}_l1"] = dny * cols[f"{p}_l1"]
    X = pd.DataFrame(cols, index=ts)
    y = pd.Series(zz, index=ts, name="y")
    meta = pd.DataFrame({"trade_date": day, "slot": slot, "session": sess, "ret": r, "sig2": s2,
                         "close": grid["close"].to_numpy(float),
                         "close_prev": _shift_in_day(grid["close"].to_numpy(float), day, 1)}, index=ts)
    ok = (slot >= cfg.drop_first) & np.isfinite(zz) & (grid["n_bars_day"].to_numpy() >= cfg.min_day_bars) \
        & np.isfinite(X.to_numpy()).all(axis=1)
    return X[ok], y[ok], meta[ok]


def groups(cfg: MRPConfig = MRPConfig()) -> dict[str, list[str]]:
    return {"A_own_lags": [f"z_l{k}" for k in range(1, cfg.n_lags + 1)],
            "B_large_move": ["mom3"],
            "C_sessions": ["sess_asia", "sess_london", "sess_ny"],
            "D_prev_day": ["prev_day"],
            "E_same_slot_yday": ["same_slot_yday"],
            "F_fx": [f"{p}_l{k}" for p in cfg.pairs for k in range(1, cfg.fx_lags + 1)],
            "G_ny_interactions": ["ny_x_z_l1"] + [f"ny_x_{p}_l1" for p in cfg.pairs]}


BASE = ["const", "d_london", "d_ny", "d_post"]


def model_columns(cfg: MRPConfig = MRPConfig()) -> dict[str, list[str]]:
    g = groups(cfg)
    m1 = BASE + g["A_own_lags"]
    m2 = m1 + g["B_large_move"] + g["C_sessions"] + g["D_prev_day"] + g["E_same_slot_yday"]
    m3 = m2 + g["F_fx"] + g["G_ny_interactions"]
    return {"M1": m1, "M2": m2, "M3": m3}


# ============================================================================= 5. Regression and inference

@dataclass
class OLSResult:
    names: list
    beta: np.ndarray
    resid: np.ndarray
    cov: np.ndarray
    cov_type: str
    n: int
    n_clusters: int = 0
    extra: dict = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        se = np.sqrt(np.diag(self.cov))
        t = self.beta / se
        df = max(self.n_clusters - 1, 1) if self.cov_type == "cluster" else self.n - len(self.beta)
        return pd.DataFrame({"coef": self.beta, "se": se, "t": t, "p": 2 * stats.t.sf(np.abs(t), df)},
                            index=self.names)


def ols(X: pd.DataFrame, y: pd.Series, clusters=None, nw_lags: int | None = None) -> OLSResult:
    A = X.to_numpy(float)
    b = y.to_numpy(float)
    XtX_inv = np.linalg.pinv(A.T @ A)
    beta = XtX_inv @ (A.T @ b)
    u = b - A @ beta
    n, k = A.shape
    if clusters is not None:
        cov, G = cluster_cov(A, u, np.asarray(clusters), XtX_inv)
        return OLSResult(list(X.columns), beta, u, cov, "cluster", n, G)
    if nw_lags is not None:
        return OLSResult(list(X.columns), beta, u, nw_cov(A, u, nw_lags, XtX_inv), "nw", n)
    return OLSResult(list(X.columns), beta, u, XtX_inv * (u @ u) / (n - k), "iid", n)


def cluster_cov(A: np.ndarray, u: np.ndarray, groups_: np.ndarray, XtX_inv: np.ndarray | None = None):
    """CR1 cluster-robust covariance."""
    n, k = A.shape
    if XtX_inv is None:
        XtX_inv = np.linalg.pinv(A.T @ A)
    S = pd.DataFrame(A * u[:, None]).groupby(groups_).sum().to_numpy()
    G = S.shape[0]
    meat = S.T @ S
    c = G / (G - 1) * (n - 1) / (n - k)
    return c * XtX_inv @ meat @ XtX_inv, G


def nw_cov(A: np.ndarray, u: np.ndarray, lags: int, XtX_inv: np.ndarray | None = None) -> np.ndarray:
    if XtX_inv is None:
        XtX_inv = np.linalg.pinv(A.T @ A)
    s = A * u[:, None]
    meat = s.T @ s
    for l in range(1, lags + 1):
        g = s[l:].T @ s[:-l]
        meat += (1 - l / (lags + 1)) * (g + g.T)
    return XtX_inv @ meat @ XtX_inv


def wald(res: OLSResult, cols: list[str]) -> dict:
    """Joint test that the coefficients in ``cols`` are zero. F = W/q with (q, G-1) df for cluster covariance."""
    idx = [res.names.index(c) for c in cols]
    b = res.beta[idx]
    V = res.cov[np.ix_(idx, idx)]
    W = float(b @ np.linalg.pinv(V) @ b)
    q = len(idx)
    df2 = max(res.n_clusters - 1, 1) if res.cov_type == "cluster" else res.n - len(res.beta)
    F = W / q
    return {"q": q, "F": F, "p": float(stats.f.sf(F, q, df2))}


def holm(p: dict, alpha: float = 0.05) -> pd.DataFrame:
    s = pd.Series(p).sort_values()
    m = len(s)
    adj = np.maximum.accumulate([min(1.0, (m - i) * v) for i, v in enumerate(s.to_numpy())])
    return pd.DataFrame({"p": s, "p_holm": adj, "reject": adj <= alpha})


def vif(X: pd.DataFrame, cols: list[str] | None = None) -> pd.Series:
    """Variance inflation factors from the inverse correlation matrix (constant excluded)."""
    cols = cols or [c for c in X.columns if c != "const" and X[c].std() > 0]
    C = np.corrcoef(X[cols].to_numpy(float), rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(C)), index=cols, name="vif")


def huber(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    import statsmodels.api as sm
    r = sm.RLM(y.to_numpy(float), X.to_numpy(float), M=sm.robust.norms.HuberT()).fit()
    return pd.DataFrame({"coef": r.params, "se": r.bse, "t": r.tvalues}, index=X.columns)


# ============================================================================= 6. Forecasting

def student_nu(resid: np.ndarray, max_n: int = 200_000, seed: int = 0) -> float:
    x = resid[np.isfinite(resid)]
    if len(x) > max_n:
        x = np.random.default_rng(seed).choice(x, max_n, replace=False)
    nu, _, _ = stats.t.fit(x / x.std(), floc=0)
    return float(max(nu, 2.1))


def p_up(mu: np.ndarray, nu: float, s: float) -> np.ndarray:
    scale = s * np.sqrt((nu - 2) / nu)
    return stats.t.sf(-mu / scale, nu)


def walk_forward(datasets: dict, refits: tuple, end: dt.date, fit_start: dt.date, cfg: MRPConfig = MRPConfig()
                 ) -> pd.DataFrame:
    """Out-of-sample forecasts of M0-M3.

    datasets: {refit_date: (X, y, meta)} built with the volatility model fitted before that refit. For each refit,
    M1-M3 are fitted on targets with trade_date in [fit_start, refit) and predict [refit, next refit).
    Output per target: y, ret, sig2, mu_M0..M3, pup_M0..M3, trade_date, refit.
    """
    edges = list(refits) + [end + dt.timedelta(days=1)]
    cols = model_columns(cfg)
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        X, y, meta = datasets[a]
        td = meta["trade_date"].to_numpy()
        fit = (td >= fit_start) & (td < a)
        pred = (td >= a) & (td < b)
        out = pd.DataFrame({"y": y[pred], "ret": meta["ret"][pred], "sig2": meta["sig2"][pred],
                            "trade_date": td[pred], "close": meta["close"][pred],
                            "close_prev": meta["close_prev"][pred], "refit": a}, index=y.index[pred])
        out["mu_M0"] = 0.0
        yf = y[fit]
        nz = yf != 0
        out["up_share_fit"] = float((yf[nz] > 0).mean())
        s0 = float(yf.std())
        out["pup_M0"] = p_up(np.zeros(pred.sum()), student_nu(yf.to_numpy() - yf.mean()), s0)
        for m, c in cols.items():
            res = ols(X.loc[fit, c], yf)
            nu = student_nu(res.resid)
            mu = X.loc[pred, c].to_numpy(float) @ res.beta
            out[f"mu_{m}"] = mu
            out[f"pup_{m}"] = p_up(mu, nu, float(res.resid.std()))
            out[f"nu_{m}"] = nu
        parts.append(out)
    return pd.concat(parts).sort_index()


def r2_oos(y: np.ndarray, f: np.ndarray, f_bench: np.ndarray) -> float:
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - f_bench) ** 2))


def clark_west(y: np.ndarray, f_small: np.ndarray, f_big: np.ndarray, day: np.ndarray, lags: int = 5) -> dict:
    """Clark-West (2007) test that the larger (nesting) model forecasts better; daily sums, Newey-West."""
    adj = (y - f_small) ** 2 - ((y - f_big) ** 2 - (f_small - f_big) ** 2)
    d = pd.Series(adj).groupby(pd.Series(day)).sum()
    m, se, t = nw_mean_t(d.to_numpy(), lags)
    return {"t": t, "p_one_sided": float(stats.norm.sf(t)), "days": len(d)}


def brier(p: np.ndarray, ret: np.ndarray) -> float:
    nz = ret != 0
    return float(np.mean((p[nz] - (ret[nz] > 0)) ** 2))


def forecast_report(fc: pd.DataFrame, periods: dict | None = None) -> dict:
    """R2_OOS (vs M0 and incremental), Clark-West, Brier, per period."""
    periods = periods or {"all": np.ones(len(fc), bool)}
    out = {}
    for name, mask in periods.items():
        f = fc[np.asarray(mask)]
        y, day, ret = f["y"].to_numpy(), f["trade_date"].to_numpy(), f["ret"].to_numpy()
        row = {"n": len(f), "days": int(pd.Series(day).nunique())}
        for m in ("M1", "M2", "M3"):
            row[f"r2_{m}_vs_M0"] = r2_oos(y, f[f"mu_{m}"].to_numpy(), f["mu_M0"].to_numpy())
            row[f"cw_{m}_vs_M0"] = clark_west(y, f["mu_M0"].to_numpy(), f[f"mu_{m}"].to_numpy(), day)
            row[f"brier_{m}"] = brier(f[f"pup_{m}"].to_numpy(), ret)
        for small, big in (("M1", "M2"), ("M2", "M3")):
            row[f"r2_{big}_vs_{small}"] = r2_oos(y, f[f"mu_{big}"].to_numpy(), f[f"mu_{small}"].to_numpy())
            row[f"cw_{big}_vs_{small}"] = clark_west(y, f[f"mu_{small}"].to_numpy(), f[f"mu_{big}"].to_numpy(), day)
        row["brier_half"] = brier(np.full(len(f), 0.5), ret)
        row["brier_upshare"] = brier(f["up_share_fit"].to_numpy(), ret)
        out[name] = row
    return out


def calibration(p: np.ndarray, ret: np.ndarray, bins: int = 10) -> pd.DataFrame:
    nz = ret != 0
    p, up = p[nz], (ret[nz] > 0).astype(float)
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"p_mean": pd.Series(p).groupby(q, observed=True).mean().to_numpy(),
                         "up_rate": pd.Series(up).groupby(q, observed=True).mean().to_numpy(),
                         "n": pd.Series(up).groupby(q, observed=True).size().to_numpy()})


def interval_coverage(fc: pd.DataFrame, model: str, levels=(0.5, 0.9)) -> dict:
    nu = fc[f"nu_{model}"].to_numpy()
    mu = fc[f"mu_{model}"].to_numpy()
    y = fc["y"].to_numpy()
    sc = np.sqrt((nu - 2) / nu)
    out = {}
    for lv in levels:
        h = stats.t.ppf(0.5 + lv / 2, nu) * sc
        out[f"cover_{int(lv * 100)}"] = float(np.mean(np.abs(y - mu) <= h))
    return out


# ============================================================================= 7. Economic test (one bar)

def one_bar_pnl(fc: pd.DataFrame, model: str, cfg: MRPConfig = MRPConfig(), scale: str = "z") -> pd.DataFrame:
    """Trade 1 contract in the sign of mu when the expected move in points exceeds the round-trip cost.

    Entry at the close of bar t (close_prev of the target row), exit at the close of bar t+1; cost = 1 tick
    + 2 commissions. Expected move in points = |mu_r| x close, mu_r = mu x sigma (z) or mu / 1e4 (raw).
    """
    mu = fc[f"mu_{model}"].to_numpy()
    mu_r = mu * np.sqrt(fc["sig2"].to_numpy()) if scale == "z" else mu / 1e4
    exp_pts = np.abs(mu_r) * fc["close_prev"].to_numpy()
    cost_usd = cfg.tick * cfg.usd_per_point + 2 * cfg.commission_per_side
    cost_pts = cost_usd / cfg.usd_per_point
    pos = np.sign(mu_r) * (exp_pts > cost_pts)
    gross = pos * (fc["close"].to_numpy() - fc["close_prev"].to_numpy()) * cfg.usd_per_point
    net = gross - np.abs(pos) * cost_usd
    return pd.DataFrame({"trade_date": fc["trade_date"].to_numpy(), "pos": pos, "gross_usd": gross, "net_usd": net,
                         "exp_pts": exp_pts}, index=fc.index)


def day_block_bootstrap(daily: np.ndarray, n_boot: int = 5000, seed: int = 0, block: int = 5) -> dict:
    x = np.asarray(daily, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    sums = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        sums[i] = x[idx].sum()
    return {"total": float(x.sum()), "lower_95": float(np.quantile(sums, 0.05)), "upper_95": float(np.quantile(sums, 0.95))}


# ============================================================================= 7b. Notebook helpers

SYNTH_CFG = MRPConfig(warmup_start=D(2019, 12, 1), train=(D(2020, 1, 1), D(2020, 12, 31)),
                      refits_train=(D(2020, 7, 1), D(2020, 10, 1)),
                      halves=((D(2020, 1, 1), D(2020, 6, 30)), (D(2020, 7, 1), D(2020, 12, 31))),
                      test=(D(2021, 1, 1), D(2021, 6, 30)), refits_test=(D(2021, 1, 1), D(2021, 4, 1)))


def load_inputs(end: dt.date, cfg: MRPConfig = MRPConfig()) -> tuple[pd.DataFrame, dict]:
    """MNQ grid from the warm-up start to ``end`` and the FX pairs on the same bars."""
    guard(end)
    grid = load_grid(cfg.warmup_start, end, cfg)
    fx = {}
    for p in cfg.pairs:
        mid = load_fx_minutes(p, cfg.warmup_start - dt.timedelta(days=3), end + dt.timedelta(days=1))
        fx[p] = fx_on_grid(mid, grid.index, cfg.fx_stale_minutes)
    return grid, fx


def fx_series(fx: dict, grid: pd.DataFrame, pair: str) -> pd.DataFrame:
    return fx[pair].assign(slot=grid["slot"].to_numpy(), trade_date=grid["trade_date"].to_numpy())


def sigmas(grid: pd.DataFrame, fx: dict, fit_start: dt.date, fit_end_excl: dt.date, cfg: MRPConfig = MRPConfig()):
    """Volatility models fitted on [fit_start, fit_end_excl) for MNQ and each FX pair; ex-ante sig2 for every bar."""
    f = fit_vol(grid, fit_start, fit_end_excl, cfg)
    s2 = predict_vol(grid, f, cfg)["sig2"]
    fx2 = {}
    for p in cfg.pairs:
        fs = fx_series(fx, grid, p)
        fx2[p] = predict_vol(fs, fit_vol(fs, fit_start, fit_end_excl, cfg), cfg)["sig2"]
    return s2, fx2, f


def dataset_for(grid: pd.DataFrame, fx: dict, fit_start: dt.date, fit_end_excl: dt.date,
                cfg: MRPConfig = MRPConfig(), scale: str = "z"):
    s2, fx2, _ = sigmas(grid, fx, fit_start, fit_end_excl, cfg)
    return build_dataset(grid, s2, fx, fx2, cfg, scale)


def in_window(meta: pd.DataFrame, a: dt.date, b: dt.date) -> np.ndarray:
    td = meta["trade_date"].to_numpy()
    return (td >= a) & (td <= b)


# ============================================================================= 8. Synthetic data (tests, smoke runs)

def write_synthetic(base, start: dt.date, end: dt.date, cfg: MRPConfig = MRPConfig(), seed: int = 7,
                    beta_jpy: float = 0.15) -> None:
    """Synthetic Dukascopy FX candles for cfg.pairs and MNQ ohlcv-1m month files under ``base``.

    MNQ: log-vol AR(1) by day, U-shaped intraday profile, quarterly contract switch with a basis jump, 1% empty
    minutes. Planted signal: each minute of 5-minute bucket b gets beta_jpy x (USDJPY bucket b-1 return / its sd)
    x the MNQ per-minute sd, i.e. a lag-1 FX effect of about beta_jpy in z units. Other pairs are noise.
    """
    import os
    from pathlib import Path
    base = Path(base)
    os.environ["EQUITY_TRADING_DIR"] = str(base)
    os.environ.pop("DUKASCOPY_DIR", None)
    for i, p in enumerate(cfg.pairs):
        FXS.write_synthetic_candles(base / "Dukascopy", p, start - dt.timedelta(days=2), end + dt.timedelta(days=1), seed=seed + i,
                                    price=150.0 if p.endswith("JPY") else 1.10)
    jpy = load_fx_minutes("USDJPY", start - dt.timedelta(days=2), end + dt.timedelta(days=1))
    jpy_full = np.log(jpy).reindex(pd.date_range(jpy.index[0], jpy.index[-1], freq="1min", tz="UTC")).ffill()
    jr5 = jpy_full.diff().rolling(5).sum()                       # return over [ts-4, ts] minute closes
    jsd = float(np.nanstd(jr5.to_numpy()))
    rng = np.random.default_rng(seed)
    price, logvol, iid = 9000.0, 0.0, 1
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
            mins = loc.hour * 60 + loc.minute
            prof = 1.0 + 2.5 * np.exp(-((mins - 575) / 25.0) ** 2) + 1.2 * np.exp(-((mins - 955) / 20.0) ** 2)
            sd = 2.0e-4 * np.exp(logvol) * prof
            bucket = ts.floor("5min")
            prev_b_end = bucket - pd.Timedelta(minutes=1)        # last minute of the previous bucket
            sig = np.nan_to_num(jr5.reindex(prev_b_end).to_numpy()) / jsd
            r = rng.normal(0, 1, len(ts)) * sd + beta_jpy * sig * sd / np.sqrt(5)   # bucket effect = beta_jpy in z
            lp = np.log(price) + np.cumsum(r)
            close = np.exp(lp)
            price = close[-1]
            keep = rng.random(len(ts)) > 0.01
            rows.append(pd.DataFrame({"ts": ts[keep], "open": close[keep], "high": close[keep] + 0.25,
                                      "low": close[keep] - 0.25, "close": close[keep],
                                      "volume": rng.integers(1, 50, keep.sum()), "instrument_id": iid}))
        d += dt.timedelta(days=1)
    df = pd.concat(rows, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for key, part in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        a = dt.date(int(key[:4]), int(key[5:]), 1)
        f = MB.month_path(cfg.root, a)
        f.parent.mkdir(parents=True, exist_ok=True)
        part[MB.COLUMNS].reset_index(drop=True).to_parquet(f, index=False)
