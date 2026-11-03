"""S04: dynamic OU pairs backtest on 15-minute bars (stocks only, long/short).

Pipeline (all decisions use data up to the close of the decision day e; they take effect on e+1):
  1. Panel: regular-session open/close grid, closes forward-filled within a session (<= 2 bars),
     daily one-way cost = half the Abdi-Ranaldo (2017) spread estimated from past 15m OHLC.
  2. Candidates: every `corr_every` sessions, each stock's top-10 signed correlations of 15m log
     returns over the trailing 20 sessions (overnight bar excluded). Pool on day e = union of the
     last `pool_refreshes` candidate sets.
  3. Daily test of every pool pair: half-life on a 20-session reference window, then Engle-Granger
     (fixed 1-lag ADF, MacKinnon N=2) on a window of clamp(5 x HL, 5, 60) sessions. BH-FDR per day.
  4. Hysteresis: tradable after 3 consecutive passes (BH q <= 1%, beta > 0, valid HL, window not
     clamped), removed after 3 consecutive fails (raw p > 10%, beta <= 0, invalid HL, or clamped).
     A missing test day counts as a fail; tradable pairs are re-tested so they are not dropped
     merely for leaving the correlation pool.
  5. Backtest: entry at |z| >= 1.5 (and < 3) if (1.5 - 0.5) sd covers 2x the round-trip cost;
     4% of equity gross per pair split by beta, at most 15 pairs, ranked by shortest half-life;
     exits at |z| <= 0.5, |z| >= 3, 3 x HL bars, or removal (break). Orders fill at the next bar
     open; entry parameters are frozen for the life of the position.
  6. Placebos: random pairs over the same tradable spells; random contrarian entry timing.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .coint import ar1_half_life, eg_test, ols_ab

S04_VERSION = "0.2.0"
S04_TESTS_VERSION = "0.1.1"   # pair-test logic; the test cache is keyed on this, so cost-only changes reuse it
NY = "America/New_York"


@dataclass(frozen=True)
class S04Config:
    timeframe: str = "15Min"
    ffill_bars: int = 2
    # candidates
    corr_lookback_sessions: int = 20
    corr_top_n: int = 10
    corr_every: int = 5
    pool_refreshes: int = 4
    min_bar_share: float = 0.95
    # estimation
    ref_sessions: int = 20
    window_rule: str = "hl"          # "hl": clamp(window_hl_mult x HL, min, max) sessions; "fixed": fixed_window_sessions
    window_hl_mult: float = 5.0
    fixed_window_sessions: int = 60
    min_window_sessions: int = 5
    max_window_sessions: int = 60
    adf_lags: int = 1
    fdr_q: float = 0.01
    remove_p: float = 0.10
    streak: int = 3
    # trading
    entry_z: float = 1.5
    tp_z: float = 0.5
    stop_z: float = 3.0
    time_stop_hl: float = 3.0
    gross_per_pair: float = 0.04
    max_pairs: int = 15
    initial_equity: float = 100_000.0
    cost_filter_mult: float = 2.0
    # costs (amendment 2026-11-03: flat 2 bps replaces Abdi-Ranaldo, which gave ~36 bps on 15m bars)
    cost_model: str = "flat"         # "flat" | "abdi_ranaldo" (diagnostic)
    flat_half_spread_bps: float = 2.0
    spread_lookback_sessions: int = 20
    min_half_spread_bps: float = 1.0
    cost_mult: float = 1.0
    # validation
    n_placebo: int = 1000
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== panel
@dataclass
class Panel:
    index: pd.DatetimeIndex      # bar start, New York time
    symbols: list
    o: np.ndarray                # (T, N) open, NaN if no bar (filled with previous close in session)
    c: np.ndarray                # (T, N) close, forward-filled within session up to ffill_bars
    day_start: np.ndarray        # (D + 1,) bar offset of each session
    days: np.ndarray             # (D,) session dates
    s2_daily: np.ndarray         # (D, N) Abdi-Ranaldo squared-spread estimate per session (NaN if none)
    h: np.ndarray | None = None  # (T, N) raw high of real bars only (NaN where no bar; never forward-filled)
    l: np.ndarray | None = None  # (T, N) raw low, same convention

    @property
    def n_days(self) -> int:
        return len(self.days)

    def half_spread(self, cfg: S04Config) -> np.ndarray:
        """(D, N) one-way cost applied on day d. Flat model: cfg.flat_half_spread_bps everywhere.
        Abdi-Ranaldo model: from sessions d-L..d-1 only (no look-ahead)."""
        if cfg.cost_model == "flat":
            return np.full(self.s2_daily.shape, cfg.flat_half_spread_bps * 1e-4 * cfg.cost_mult)
        s2 = pd.DataFrame(self.s2_daily)
        m = s2.rolling(cfg.spread_lookback_sessions, min_periods=5).mean().shift(1).to_numpy()
        hs = 0.5 * np.sqrt(np.clip(m, 0.0, None))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(hs, axis=1, keepdims=True)  # symbol without history: that day's median
        hs = np.where(np.isfinite(hs), hs, med)
        hs = np.where(np.isfinite(hs), hs, 10e-4)  # no history at all: 10 bps
        return np.maximum(hs, cfg.min_half_spread_bps * 1e-4) * cfg.cost_mult


def abdi_ranaldo_s2(lc: np.ndarray, lh: np.ndarray, ll: np.ndarray, session: np.ndarray, n_days: int) -> np.ndarray:
    """Per-session mean of 4 (c_t - eta_t)(c_t - eta_{t+1}), eta = (log h + log l) / 2,
    over consecutive bars of the same session that both traded (Abdi & Ranaldo 2017).
    Inputs are (T,) log prices of real (not forward-filled) bars, NaN where no bar."""
    eta = 0.5 * (lh + ll)
    prod = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    ok = np.isfinite(prod) & (session[:-1] == session[1:])
    num = np.bincount(session[:-1][ok], weights=prod[ok], minlength=n_days)
    cnt = np.bincount(session[:-1][ok], minlength=n_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= 5, 4.0 * num / cnt, np.nan)


def _ffill_in_session(a: np.ndarray, session: np.ndarray, limit: int) -> np.ndarray:
    df = pd.DataFrame(a)
    return df.groupby(session).ffill(limit=limit).to_numpy() if limit else a


def _assemble(index: pd.DatetimeIndex, symbols, o, c, h, l, ffill_bars: int) -> Panel:
    dates = index.tz_convert(NY).date
    days, session = np.unique(dates, return_inverse=True)
    D = len(days)
    day_start = np.searchsorted(session, np.arange(D + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.column_stack([abdi_ranaldo_s2(np.log(c[:, j]), np.log(h[:, j]), np.log(l[:, j]), session, D)
                              for j in range(c.shape[1])])
    cf = _ffill_in_session(c, session, ffill_bars)
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), cf[:-1]])
    prev[day_start[:-1]] = np.nan                       # never carry yesterday's close into today's open
    of = np.where(np.isfinite(o), o, prev)
    of = np.where(np.isfinite(cf), of, np.nan)
    return Panel(index=index, symbols=list(symbols), o=of, c=cf, day_start=day_start, days=days, s2_daily=s2,
                 h=np.asarray(h, dtype=float), l=np.asarray(l, dtype=float))


def bar_grid(years: list[int], minutes: int) -> pd.DatetimeIndex:
    from .alpaca import session_schedule
    sc = session_schedule(years)
    stamps = [pd.date_range(op, cl, freq=f"{minutes}min", inclusive="left") for op, cl in zip(sc.market_open, sc.market_close)]
    # concatenate as datetimes, never as raw integers: pandas >= 3 may store microseconds, not nanoseconds
    grid = stamps[0].append(stamps[1:]) if stamps else pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(grid).tz_convert(NY)


def load_panel(symbols: list[str], timeframe: str, years: list[int], ffill_bars: int = 2) -> Panel:
    """Regular-session OHLC grid from the Alpaca cache (exchange calendar, early closes respected)."""
    from .alpaca import load_bars
    minutes = int(timeframe.replace("Min", ""))
    grid = bar_grid(years, minutes)
    T = len(grid)
    out = {k: np.full((T, len(symbols)), np.nan) for k in "ohlc"}
    kept = []
    for j, s in enumerate(symbols):
        try:
            b = load_bars(s, timeframe, years)
        except FileNotFoundError:
            continue
        if b.empty:
            continue
        t = pd.DatetimeIndex(pd.to_datetime(b["t"], utc=True)).tz_convert(NY)
        pos = grid.get_indexer(t)
        m = pos >= 0
        for k in "ohlc":
            out[k][pos[m], j] = b[k].to_numpy(dtype=float)[m]
        kept.append(j)
    kept = np.array(kept, dtype=int)
    filled = np.isfinite(out["c"][:, kept]).mean() if len(kept) else 0.0
    if filled < 0.5:
        raise RuntimeError(f"load_panel: only {filled:.1%} of grid cells have a close; bar timestamps do not match the grid.")
    return _assemble(grid, [symbols[j] for j in kept], *(out[k][:, kept] for k in "ohlc"), ffill_bars=ffill_bars)


def synthetic_ohlc_panel(years: list[int], bars_per_day: int = 26, half_spread_bps: float = 3.0,
                         seed: int = 3) -> tuple[Panel, list[str]]:
    """Offline panel: alpaca.synthetic_panel mid prices (same-sector names cointegrated by
    construction) plus a bid/ask bounce of known half-spread. Returns (panel, tradable symbols);
    the sector ETFs stay in the panel, 'XLK' doubles as the benchmark."""
    from .alpaca import synthetic_panel
    from .universe import SECTOR_ETF_01, UNIVERSE_01
    mid = synthetic_panel(UNIVERSE_01, SECTOR_ETF_01, years, bars_per_day=bars_per_day, seed=seed)
    rng = np.random.default_rng(seed)
    m = mid.to_numpy()
    hs = half_spread_bps * 1e-4
    sign = lambda: rng.choice([-1.0, 1.0], size=m.shape)  # noqa: E731
    c = m * (1 + hs * sign())
    prev_mid = np.vstack([m[:1], m[:-1]])
    o = prev_mid * (1 + hs * sign())
    wig = np.abs(rng.standard_normal(m.shape)) * 4e-4
    h = np.maximum(np.maximum(o, c), m * (1 + hs + wig))
    l = np.minimum(np.minimum(o, c), m * (1 - hs - wig))
    panel = _assemble(mid.index, list(mid.columns), o, c, h, l, ffill_bars=2)
    stocks = sorted(s for v in UNIVERSE_01.values() for s in v)
    return panel, stocks


# =================================================================== candidates and tests
def candidate_schedule(panel: Panel, tradable_idx: np.ndarray, cfg: S04Config, first_day: int) -> dict[int, np.ndarray]:
    """{decision day e: (P, 2) int array of (y, x) column indices}, pool = union of the last
    `pool_refreshes` correlation refreshes. y is the alphabetically first ticker (thesis rule)."""
    lc = np.log(panel.c[:, tradable_idx])
    r = np.diff(lc, axis=0, prepend=np.nan)
    r[panel.day_start[:-1]] = np.nan                    # drop the overnight bar
    sym = np.array(panel.symbols)
    refresh = {}
    D = panel.n_days
    for e in range(first_day, D - 1):
        if (e - first_day) % cfg.corr_every:
            continue
        a, b = panel.day_start[e - cfg.corr_lookback_sessions + 1], panel.day_start[e + 1]
        w = r[a:b]
        okc = np.isfinite(w).mean(axis=0) >= cfg.min_bar_share
        cols = np.where(okc)[0]
        corr = pd.DataFrame(w[:, cols]).corr().to_numpy().copy()
        np.fill_diagonal(corr, -np.inf)
        corr = np.where(np.isfinite(corr), corr, -np.inf)
        k = min(cfg.corr_top_n, len(cols) - 1)
        if k < 1:
            refresh[e] = np.empty((0, 2), dtype=int)
            continue
        top = np.argpartition(-corr, k - 1, axis=1)[:, :k]
        i = np.repeat(np.arange(len(cols)), k)
        j = top.ravel()
        good = np.isfinite(corr[i, j])
        gi, gj = tradable_idx[cols[i[good]]], tradable_idx[cols[j[good]]]
        y = np.where(sym[gi] < sym[gj], gi, gj)          # alphabetical orientation
        x = np.where(y == gi, gj, gi)
        refresh[e] = np.unique(np.column_stack([y, x]), axis=0)
    keys = sorted(refresh)
    pools = {}
    for e in range(first_day, D - 1):
        recent = [refresh[k] for k in keys if k <= e][-cfg.pool_refreshes:]
        pools[e] = np.unique(np.vstack(recent), axis=0) if recent else np.empty((0, 2), dtype=int)
    return pools


TEST_COLS = ["day", "y", "x", "alpha", "beta", "mu", "sd", "p", "hl_ref", "hl", "window", "slow"]


def test_pair(ly: np.ndarray, lx: np.ndarray, bars_ref: int, cfg: S04Config, bars_per_day: int) -> tuple | None:
    """Tests one pair on log closes ending at the decision bar. Returns the TEST_COLS[3:] tuple or None."""
    y, x = ly[-bars_ref:], lx[-bars_ref:]
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.mean() < cfg.min_bar_share:
        return None
    a, b = ols_ab(y[ok], x[ok])
    hl_ref = ar1_half_life(y[ok] - a - b * x[ok])
    nan = float("nan")
    if not np.isfinite(hl_ref):
        return (nan, b, nan, nan, nan, nan, nan, 0, True)
    lo, hi = cfg.min_window_sessions * bars_per_day, cfg.max_window_sessions * bars_per_day
    if cfg.window_rule == "fixed":
        need = cfg.fixed_window_sessions * bars_per_day
        slow = bool(cfg.window_hl_mult * hl_ref > need)   # window holds fewer than window_hl_mult half-lives
    else:
        need = cfg.window_hl_mult * hl_ref
        slow = bool(need > hi)
    W = int(min(max(math.ceil(need), lo), hi, len(ly)))
    y, x = ly[-W:], lx[-W:]
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.mean() < cfg.min_bar_share:
        return None
    res = eg_test(y[ok], x[ok], cfg.adf_lags)
    u = y[ok] - res["alpha"] - res["beta"] * x[ok]
    hl = ar1_half_life(u)
    return (res["alpha"], res["beta"], res["mu"], res["sd"], res["p"], hl_ref, hl, W, slow)


def _test_block(lc: np.ndarray, day_start: np.ndarray, jobs: list[tuple[int, np.ndarray]], cfg: S04Config,
                bars_per_day: int) -> list:
    rows = []
    max_bars = cfg.max_window_sessions * bars_per_day
    for e, pairs in jobs:
        end = day_start[e + 1]
        a = max(0, end - max_bars)
        bars_ref = end - day_start[e - cfg.ref_sessions + 1]
        seg = lc[a:end]
        for yi, xi in pairs:
            r = test_pair(seg[:, yi], seg[:, xi], bars_ref, cfg, bars_per_day)
            if r is not None:
                rows.append((e, int(yi), int(xi)) + r)
    return rows


def run_tests(panel: Panel, pools: dict[int, np.ndarray], cfg: S04Config, cache_dir=None, block_days: int = 20,
              n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
    """Daily tests of every pool pair. Cached per block of decision days (parquet) when cache_dir is set."""
    from joblib import Parallel, delayed
    import time
    lc = np.log(panel.c)
    bpd = int(round(np.median(np.diff(panel.day_start))))
    days = sorted(pools)
    blocks = [days[i:i + block_days] for i in range(0, len(days), block_days)]
    out, todo = [], []
    for blk in blocks:
        p = cache_dir / f"tests_{blk[0]:05d}_{blk[-1]:05d}.parquet" if cache_dir is not None else None
        if p is not None and p.exists():
            out.append(pd.read_parquet(p))
        else:
            todo.append((blk, p))
    if verbose:
        print(f"tests: {len(blocks) - len(todo)} cached blocks, {len(todo)} to run "
              f"({sum(len(pools[e]) for b, _ in todo for e in b):,} pair-days)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(_test_block)(lc, panel.day_start, [(e, pools[e]) for e in blk], cfg, bpd) for blk, _ in todo)
    for (blk, p), rows in zip(todo, res):
        df = _rows_frame(rows)
        if p is not None:
            df.to_parquet(p)
        out.append(df)
    if verbose and todo:
        print(f"tests done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return finalize_tests(pd.concat(out, ignore_index=True) if out else _rows_frame([]), cfg)


def _rows_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=TEST_COLS)
    return df.astype({"day": "int32", "y": "int32", "x": "int32", "window": "int32", "slow": bool})


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    o = np.argsort(q)
    adj = q[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def finalize_tests(df: pd.DataFrame, cfg: S04Config) -> pd.DataFrame:
    """Drop duplicates, add BH-adjusted p per decision day and pass/fail flags."""
    df = df.drop_duplicates(["day", "y", "x"]).sort_values(["day", "y", "x"]).reset_index(drop=True)
    df["padj"] = df.groupby("day")["p"].transform(lambda s: pd.Series(bh_adjust(s.to_numpy()), index=s.index))
    valid = np.isfinite(df["hl"]) & np.isfinite(df["hl_ref"]) & (df["hl"] > 0) & ~df["slow"]
    df["passed"] = (df["padj"] <= cfg.fdr_q) & (df["beta"] > 0) & valid
    df["failed"] = ~(df["p"] <= cfg.remove_p) | ~(df["beta"] > 0) | ~valid
    return df


# =================================================================== hysteresis
def hysteresis(tests: pd.DataFrame, cfg: S04Config, n_days: int) -> pd.DataFrame:
    """Tradable spells [start, end) in effective trading days (decision day + 1).

    Add after `streak` consecutive passes; remove after `streak` consecutive fails. Days on which a
    pair has no test row count as fails. Neutral days (neither pass nor fail) reset both streaks.
    """
    k = cfg.streak
    key = tests["y"].to_numpy().astype(np.int64) * 100_000 + tests["x"].to_numpy()
    npass = pd.Series(tests["passed"].to_numpy(int)).groupby(key).sum()
    keep = np.isin(key, npass.index[npass.to_numpy() >= k])
    sub = tests.loc[keep, ["day", "y", "x", "passed", "failed"]].assign(_k=key[keep]).sort_values(["_k", "day"])
    spells = []
    last_decision = n_days - 2
    kk = sub["_k"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1], True]) if len(kk) else np.array([0])
    D_, P_, F_, Y_, X_ = (sub[c].to_numpy() for c in ["day", "passed", "failed", "y", "x"])
    for g0, g1 in zip(bounds[:-1], bounds[1:]):
        yi, xi = int(Y_[g0]), int(X_[g0])
        days, pas, fai = D_[g0:g1], P_[g0:g1], F_[g0:g1]
        trad, add, rem, last, start = False, 0, 0, None, None
        for e, ps, fl in zip(days, pas, fai):
            gap = 0 if last is None else e - last - 1
            if gap > 0:
                add = 0
                if trad:
                    if rem + gap >= k:
                        f = last + (k - rem)
                        spells.append((yi, xi, start, f + 1))
                        trad, rem = False, 0
                    else:
                        rem += gap
            if ps:
                add, rem = add + 1, 0
            elif fl:
                add, rem = 0, rem + 1
            else:
                add, rem = 0, 0
            if not trad and add >= k:
                trad, start = True, e + 1
            elif trad and rem >= k:
                spells.append((yi, xi, start, e + 1))
                trad, rem = False, 0
            last = e
        if trad:
            f = last + (k - rem)
            end = f + 1 if f <= last_decision else n_days
            spells.append((yi, xi, start, min(end, n_days)))
    return pd.DataFrame(spells, columns=["y", "x", "start", "end"]).astype("int64")


def spell_gaps(tests: pd.DataFrame, spells: pd.DataFrame, n_days: int, streak: int = 3,
               buffer: int = 20) -> dict[int, list]:
    """Decision days with no test row inside each spell: {day: [(y, x), ...]}.

    Testing these keeps a tradable pair from being dropped only because it left the correlation
    pool. When a spell was ended by missing days, the next `buffer` days are requested as well, so
    the spell can run until a real removal (neutral days keep a pair tradable)."""
    have = set(zip(tests["day"].to_numpy().tolist(), tests["y"].to_numpy().tolist(), tests["x"].to_numpy().tolist()))
    need: dict[int, list] = {}
    last = n_days - 2
    for yi, xi, s, e in spells.itertuples(index=False):
        tail = range(max(s - 1, e - streak), min(e, last + 1))
        gap_ended = e < n_days and any((d, yi, xi) not in have for d in tail)
        hi = min(e + buffer if gap_ended else e, last + 1)
        for d in range(s - 1, hi):
            if (d, yi, xi) not in have:
                need.setdefault(d, []).append((yi, xi))
    return need


def tradable_spells(panel: Panel, tests: pd.DataFrame, cfg: S04Config, max_iter: int = 50,
                    n_jobs: int = -1, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hysteresis with sticky re-testing of tradable pairs until no spell has a missing test day."""
    from joblib import Parallel, delayed
    lc = np.log(panel.c)
    bpd = int(round(np.median(np.diff(panel.day_start))))
    for it in range(max_iter):
        spells = hysteresis(tests, cfg, panel.n_days)
        need = spell_gaps(tests, spells, panel.n_days, cfg.streak)
        if verbose:
            print(f"hysteresis pass {it + 1}: {len(spells):,} spells, {sum(map(len, need.values())):,} missing pair-days", flush=True)
        if not need:
            return tests, spells
        days = sorted(need)
        chunks = [days[i:i + 20] for i in range(0, len(days), 20)]
        res = Parallel(n_jobs=n_jobs)(delayed(_test_block)(lc, panel.day_start,
                                                          [(e, np.array(need[e])) for e in ch], cfg, bpd) for ch in chunks)
        new = _rows_frame([r for rows in res for r in rows])
        # pairs with no data on a day stay missing; mark them as explicit fails so the loop ends
        got = set(zip(new["day"], new["y"], new["x"]))
        miss = [(d, yi, xi) for d in days for yi, xi in need[d] if (d, yi, xi) not in got]
        if miss:
            filler = pd.DataFrame(miss, columns=["day", "y", "x"])
            for c in TEST_COLS[3:]:
                filler[c] = np.nan
            filler["window"], filler["slow"] = 0, True
            new = pd.concat([new, _rows_frame(filler.to_numpy().tolist())], ignore_index=True)
        base = tests[TEST_COLS]
        tests = finalize_tests(pd.concat([base, new], ignore_index=True), cfg)
    return tests, hysteresis(tests, cfg, panel.n_days)


# =================================================================== OU hitting probability
def ou_tp_probability(z0, tp: float = 0.5, stop: float = 3.0) -> np.ndarray:
    """P(|z| reaches tp before stop | |z| = |z0|) for a stationary OU in z units
    (dz = -k z dt + sqrt(2k) dW). Scale function S(z) = int_0^z exp(u^2/2) du; independent of k."""
    from scipy.special import erfi
    S = lambda z: np.sqrt(np.pi / 2) * erfi(np.asarray(z, float) / np.sqrt(2))  # noqa: E731
    z = np.clip(np.abs(np.asarray(z0, float)), tp, stop)
    return (S(stop) - S(z)) / (S(stop) - S(tp))


# =================================================================== backtest
class _Params:
    """Per-day parameter lookup from the tests table (decision day d -> row)."""

    def __init__(self, tests: pd.DataFrame, n_sym: int):
        t = tests.sort_values(["day", "y", "x"])
        self.n = n_sym
        self.day = t["day"].to_numpy()
        self.key = t["y"].to_numpy().astype(np.int64) * n_sym + t["x"].to_numpy()
        self.vals = t[["alpha", "beta", "mu", "sd", "hl"]].to_numpy(dtype=float)
        self.valid = (np.isfinite(self.vals).all(axis=1) & (t["beta"].to_numpy() > 0) & (t["hl"].to_numpy() > 0)
                      & ~t["slow"].to_numpy(dtype=bool) & (t["sd"].to_numpy() > 0))
        D = int(self.day.max()) + 2 if len(self.day) else 1
        self.off = np.searchsorted(self.day, np.arange(D + 1))

    def get(self, d: int, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows of decision day d for keys -> (values (K,5), found mask)."""
        if d < 0 or d + 1 >= len(self.off):
            return np.full((len(keys), 5), np.nan), np.zeros(len(keys), bool)
        a, b = self.off[d], self.off[d + 1]
        kk = self.key[a:b]
        pos = np.searchsorted(kk, keys)
        pos_c = np.clip(pos, 0, max(len(kk) - 1, 0))
        found = (len(kk) > 0) & (kk[pos_c] == keys) if len(kk) else np.zeros(len(keys), bool)
        out = np.full((len(keys), 5), np.nan)
        if len(kk):
            out[found] = self.vals[a + pos_c[found]]
            found = found & self.valid[a + pos_c]
        return out, found

    def valid_keys(self, d: int) -> np.ndarray:
        a, b = self.off[d], self.off[d + 1]
        return self.key[a:b][self.valid[a:b]]


def placebo_pair_spells(spells: pd.DataFrame, params: _Params, rng: np.random.Generator) -> pd.DataFrame:
    """Same spell calendar, each spell's pair replaced by a random pool pair with valid OU shape on
    the spell's decision day (not significant-tested, not tradable that day)."""
    out = []
    n = params.n
    real = {}
    for yi, xi, s, e in spells.itertuples(index=False):
        for d in range(s, e):
            real.setdefault(d, set()).add(yi * n + xi)
    for yi, xi, s, e in spells.itertuples(index=False):
        pool = params.valid_keys(s - 1)
        pool = pool[~np.isin(pool, list(real.get(s, ())))]
        if len(pool) == 0:
            continue
        k = int(rng.choice(pool))
        out.append((k // n, k % n, s, e))
    return pd.DataFrame(out, columns=["y", "x", "start", "end"]).astype("int64")


def backtest(panel: Panel, params: _Params, spells: pd.DataFrame, half_spread: np.ndarray, cfg: S04Config,
             mode: str = "strategy", entry_prob: float | None = None, seed: int = 0, record: bool = True) -> dict:
    """Bar-by-bar portfolio simulation. mode: 'strategy' | 'placebo_pairs' | 'placebo_timing'.

    Signals at bar close, fills at the next bar open (+/- half-spread). Equity is marked at closes.
    """
    rng = np.random.default_rng(seed)
    if mode == "placebo_pairs":
        spells = placebo_pair_spells(spells, params, rng)
    n = params.n
    D, T = panel.n_days, len(panel.index)
    o, c = panel.o, panel.c
    lc_all = np.log(c)
    starts = {}
    for yi, xi, s, e in spells.itertuples(index=False):
        starts.setdefault(int(s), []).append((int(yi) * n + int(xi), int(e)))
    active: dict[int, int] = {}                          # key -> spell end (exclusive)
    cash = cfg.initial_equity
    last_px = np.full(c.shape[1], np.nan)
    pos: dict[int, dict] = {}
    pend_exit: dict[int, str] = {}
    pend_entry: list[dict] = []
    trades = []
    eq_bar = np.full(T, np.nan)
    n_open = np.zeros(T, np.int16)
    gross_exp = np.zeros(T)
    elig_count = sig_count = 0
    eq_prev = cfg.initial_equity
    first = int(spells["start"].min()) if len(spells) else D
    if first < D:
        eq_bar[:panel.day_start[first]] = cfg.initial_equity

    def fill_px(j, b_abs):
        p = o[b_abs, j]
        return p if np.isfinite(p) else last_px[j]

    def do_exit(k, b_abs, reason, at_close=False):
        nonlocal cash
        p = pos[k]
        py = c[b_abs, p["y"]] if at_close else fill_px(p["y"], b_abs)
        px = c[b_abs, p["x"]] if at_close else fill_px(p["x"], b_abs)
        if at_close:
            py = py if np.isfinite(py) else last_px[p["y"]]
            px = px if np.isfinite(px) else last_px[p["x"]]
        if not (np.isfinite(py) and np.isfinite(px)):
            return False
        hs_y, hs_x = hs_today[p["y"]], hs_today[p["x"]]
        fy = py * (1 - np.sign(p["ny"]) * hs_y)          # selling a long gets bid, covering a short pays ask
        fx = px * (1 - np.sign(p["nx"]) * hs_x)
        cash += p["ny"] * fy + p["nx"] * fx
        cost = abs(p["ny"]) * py * hs_y + abs(p["nx"]) * px * hs_x
        pnl = p["ny"] * (fy - p["fy"]) + p["nx"] * (fx - p["fx"])
        if record:
            trades.append({"y": p["y"], "x": p["x"], "dir": p["dir"], "entry_bar": p["bar"], "exit_bar": b_abs,
                           "bars": b_abs - p["bar"], "reason": reason, "z_entry": p["z"], "hl": p["hl"],
                           "beta": p["beta"], "notional": p["notional"], "pnl": pnl,
                           "cost": cost + p["cost"], "ret": pnl / p["notional"]})
        del pos[k]
        return True

    for t in range(first, D):
        for k, e in starts.get(t, []):
            active[k] = e
        for k in [k for k, e in active.items() if e <= t]:
            del active[k]
        keys = np.fromiter(active.keys(), np.int64, len(active))
        vals, found = params.get(t - 1, keys)
        if len(keys):                                    # tradable but invalid/missing today: carry last valid params forward
            for i in np.where(~found)[0]:
                for back in range(2, 30):
                    v, f = params.get(t - back, keys[i:i + 1])
                    if f[0]:
                        vals[i], found[i] = v[0], True
                        break
        keys, vals = keys[found], vals[found]
        yi, xi = keys // n, keys % n
        al, be, mu, sd, hl = vals.T if len(vals) else (np.empty(0),) * 5
        hs_today = half_spread[t]
        cost_ok = (cfg.entry_z - cfg.tp_z) * sd >= cfg.cost_filter_mult * 2 * (hs_today[yi] + be * hs_today[xi])
        a, b_end = panel.day_start[t], panel.day_start[t + 1]
        lc = lc_all[a:b_end]
        Z = (lc[:, yi] - al - be * lc[:, xi] - mu) / sd if len(keys) else np.empty((b_end - a, 0))
        act_set = set(keys.tolist())
        for k in pos:                                    # break: pair no longer tradable
            if k not in act_set and k not in pend_exit:
                pend_exit[k] = "break"
        pend_entry = [q for q in pend_entry if q["key"] in act_set]
        hl_rank = np.argsort(hl, kind="stable") if len(keys) else np.empty(0, int)
        for bi in range(b_end - a):
            B = a + bi
            # ---- fills at this bar's open
            for k in list(pend_exit):
                if do_exit(k, B, pend_exit[k]):
                    del pend_exit[k]
            for q in pend_entry:
                if len(pos) >= cfg.max_pairs or q["key"] in pos:
                    continue
                py, px = fill_px(q["y"], B), fill_px(q["x"], B)
                if not (np.isfinite(py) and np.isfinite(px)):
                    continue
                ny_d = cfg.gross_per_pair * eq_prev / (1 + q["beta"])
                ny, nx = math.floor(ny_d / py), math.floor(q["beta"] * ny_d / px)
                if ny < 1 or nx < 1:
                    continue
                s = q["dir"]                               # +1: spread rich -> short y, long x
                sy, sx = -s * ny, s * nx
                hy, hx = hs_today[q["y"]], hs_today[q["x"]]
                fy, fx = py * (1 + np.sign(sy) * hy), px * (1 + np.sign(sx) * hx)
                cash -= sy * fy + sx * fx
                pos[q["key"]] = {"y": q["y"], "x": q["x"], "ny": sy, "nx": sx, "fy": fy, "fx": fx, "dir": s,
                                 "al": q["al"], "be": q["beta"], "mu": q["mu"], "sd": q["sd"], "hl": q["hl"],
                                 "beta": q["beta"], "z": q["z"], "bar": B,
                                 "max_bars": math.ceil(cfg.time_stop_hl * q["hl"]),
                                 "notional": ny * py + nx * px, "cost": ny * py * hy + nx * px * hx}
            pend_entry = []
            # ---- close of bar
            row = c[B]
            okp = np.isfinite(row)
            last_px[okp] = row[okp]
            for k, p in pos.items():
                if k in pend_exit:
                    continue
                z = (lc[bi, p["y"]] - p["al"] - p["be"] * lc[bi, p["x"]] - p["mu"]) / p["sd"]
                if not np.isfinite(z):
                    continue
                sz = p["dir"] * z
                if sz >= cfg.stop_z:
                    pend_exit[k] = "stop"
                elif sz <= cfg.tp_z:
                    pend_exit[k] = "tp"
                elif B - p["bar"] + 1 >= p["max_bars"]:
                    pend_exit[k] = "time"
            if len(keys):
                z = Z[bi]
                az = np.abs(z)
                flat = np.array([k not in pos for k in keys.tolist()])
                base = flat & np.isfinite(z) & cost_ok & (az < cfg.stop_z)
                if mode == "placebo_timing":
                    elig = base & (az > cfg.tp_z)
                    sig = elig & (rng.random(len(keys)) < entry_prob)
                else:
                    elig = base & (az > cfg.tp_z)
                    sig = base & (az >= cfg.entry_z)
                elig_count += int(elig.sum())
                sig_count += int(sig.sum())
                free = cfg.max_pairs - (len(pos) - len(pend_exit))
                if free > 0 and sig.any():
                    for i in hl_rank[sig[hl_rank]][:free]:
                        pend_entry.append({"key": int(keys[i]), "y": int(yi[i]), "x": int(xi[i]), "dir": int(np.sign(z[i])),
                                           "al": al[i], "beta": be[i], "mu": mu[i], "sd": sd[i], "hl": hl[i], "z": z[i]})
            mv = sum(p["ny"] * last_px[p["y"]] + p["nx"] * last_px[p["x"]] for p in pos.values())
            eq_prev = cash + mv
            eq_bar[B] = eq_prev
            n_open[B] = len(pos)
            gross_exp[B] = sum(abs(p["ny"]) * last_px[p["y"]] + abs(p["nx"]) * last_px[p["x"]] for p in pos.values())
    # liquidate at the last close
    if len(pos):
        Bl = T - 1
        hs_today = half_spread[D - 1]
        for k in list(pos):
            do_exit(k, Bl, "end", at_close=True)
        eq_bar[Bl] = cash
    day_eq = pd.Series(eq_bar[panel.day_start[1:] - 1], index=pd.to_datetime(panel.days))
    out = {"daily_equity": day_eq.ffill().fillna(cfg.initial_equity), "entry_rate": sig_count / max(elig_count, 1),
           "first_day": pd.Timestamp(panel.days[min(first, D - 1)]),
           "final_equity": float(day_eq.ffill().iloc[-1]) if len(day_eq) else cfg.initial_equity}
    if record:
        tr = pd.DataFrame(trades)
        if len(tr):
            tr["entry_time"] = panel.index[tr["entry_bar"].to_numpy()]
            tr["exit_time"] = panel.index[tr["exit_bar"].to_numpy()]
            tr["pair"] = [f"{panel.symbols[a]}-{panel.symbols[b]}" for a, b in zip(tr["y"], tr["x"])]
            tr["p_tp_model"] = ou_tp_probability(tr["z_entry"], cfg.tp_z, cfg.stop_z)
        out.update(trades=tr, bar_equity=pd.Series(eq_bar, index=panel.index), n_open=n_open, gross_exposure=gross_exp,
                   spells=spells)
    return out


# =================================================================== diagnostics
def perf_metrics(daily_equity: pd.Series, initial: float, start=None) -> dict:
    """Daily-return metrics from `start` (first trading day) on; Sharpe uses raw returns, sqrt(252)."""
    if start is not None:
        daily_equity = daily_equity[daily_equity.index >= pd.Timestamp(start)]
    r = daily_equity.pct_change().dropna()
    dd = daily_equity / daily_equity.cummax() - 1
    sd = r.std(ddof=1)
    return {"total_return": float(daily_equity.iloc[-1] / initial - 1),
            "sharpe": float(np.sqrt(252) * r.mean() / sd) if sd > 0 else 0.0,
            "max_drawdown": float(-dd.min()), "mean_daily": float(r.mean()), "days": int(len(r))}


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or not len(trades):
        return {"trades": 0}
    gross = trades["pnl"] + trades["cost"]
    out = {"trades": int(len(trades)), "win_rate": float((trades["pnl"] > 0).mean()),
           "mean_trade_ret": float(trades["ret"].mean()), "median_trade_ret": float(trades["ret"].median()),
           "mean_bars": float(trades["bars"].mean()), "total_pnl": float(trades["pnl"].sum()),
           "total_cost": float(trades["cost"].sum()), "gross_pnl": float(gross.sum()),
           "cost_share_of_gross": float(trades["cost"].sum() / gross.abs().sum()) if gross.abs().sum() > 0 else np.nan}
    for r, v in trades["reason"].value_counts(normalize=True).items():
        out[f"exit_{r}"] = float(v)
    return out


def calibration(trades: pd.DataFrame) -> dict:
    """Model P(tp before stop | z_entry) vs realized, on trades that ended at tp or stop."""
    if trades is None or not len(trades):
        return {}
    d = trades[trades["reason"].isin(["tp", "stop"])]
    if not len(d):
        return {"resolved": 0}
    hit = (d["reason"] == "tp").to_numpy(float)
    p = d["p_tp_model"].to_numpy()
    return {"resolved": int(len(d)), "censored_share": float(1 - len(d) / len(trades)),
            "mean_predicted": float(p.mean()), "realized": float(hit.mean()),
            "gap_pp": float(100 * (hit.mean() - p.mean())), "brier": float(np.mean((p - hit) ** 2))}


def block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, (n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def market_regression(strategy_daily: pd.Series, bench_close: pd.Series, lags: int = 5) -> dict:
    import statsmodels.api as sm
    r = strategy_daily.pct_change()
    m = bench_close.pct_change()
    d = pd.concat([r, m], axis=1, keys=["s", "m"]).dropna()
    d = d[d["s"] != 0] if (d["s"] != 0).any() else d
    if len(d) < 30:
        return {}
    fit = sm.OLS(d["s"], sm.add_constant(d["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_daily": float(fit.params["const"]), "alpha_p": float(fit.pvalues["const"]),
            "beta": float(fit.params["m"]), "r2": float(fit.rsquared), "n": int(len(d))}


def placebo_rank(value: float, dist: np.ndarray) -> dict:
    dist = np.asarray(dist, float)
    return {"value": float(value), "percentile": float(100 * np.mean(dist < value)),
            "p_upper": float((1 + np.sum(dist >= value)) / (len(dist) + 1)),
            "placebo_median": float(np.median(dist)), "placebo_p95": float(np.percentile(dist, 95))}


def gates(boot: dict, pl_pairs: dict, pl_timing: dict, cost2: dict, base: dict, alpha: float = 0.05) -> dict:
    """Pre-registered gates (studies/S04_README.md)."""
    g1 = all(v[1] > 0 for v in boot.values())
    g2 = pl_pairs["total_return"]["p_upper"] <= alpha and pl_pairs["sharpe"]["p_upper"] <= alpha
    g3 = pl_timing["total_return"]["p_upper"] <= alpha and pl_timing["sharpe"]["p_upper"] <= alpha
    g4 = cost2["total_return"] > 0 and cost2["sharpe"] > 0
    econ = base["total_return"] > 0 and g4
    stat = g1 and g2 and g3
    return {"G1_bootstrap": g1, "G2_vs_random_pairs": g2, "G3_vs_random_timing": g3, "G4_double_cost": g4,
            "statistical_verdict": "SIGNAL" if stat else ("PARTIAL" if (g1 or g2 or g3) else "NO_SIGNAL"),
            "economic_verdict": "PROFITABLE_AFTER_COSTS" if econ else "NOT_PROFITABLE_AFTER_COSTS",
            "prospective_verdict": "PENDING_2026_HOLDOUT",
            "verdict": "EDGE_CANDIDATE" if (stat and econ) else "NO_EDGE"}


def with_cost(cfg: S04Config, mult: float) -> S04Config:
    return replace(cfg, cost_mult=mult)


def first_decision_day(cfg: S04Config) -> int:
    """First session whose close has enough history for every window (0-based session index)."""
    return max(cfg.max_window_sessions, cfg.corr_lookback_sessions, cfg.ref_sessions) - 1


def run_placebo(panel: Panel, params: _Params, spells: pd.DataFrame, half_spread: np.ndarray, cfg: S04Config,
                mode: str, entry_prob: float, seed: int, start) -> dict:
    """One placebo backtest -> metrics row. Module-level so joblib memory-maps the large arrays
    instead of pickling notebook globals."""
    r = backtest(panel, params, spells, half_spread, cfg, mode=mode, entry_prob=entry_prob, seed=seed, record=False)
    return {"mode": mode, "seed": seed, **perf_metrics(r["daily_equity"], cfg.initial_equity, start)}
