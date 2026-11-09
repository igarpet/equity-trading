"""FX intraday seasonality scan (branch FX_Intraday_Seasonality, FXS 001).

Every trading day of 1-minute candles is stacked on top of the others by minute of day, without
calendar labels, and every interval (start every 5 minutes, 15-240 minutes long) is tested for a
non-zero mean return. Discovery 2016-2020, confirmation 2021-2025, 2026 sealed.

Data: Dukascopy 1-minute bid/ask candles from ``tools/dukascopy/dukascopy_candles.py``
(<Dukascopy>/<PAIR>/candles_m1/<YYYY>/<PAIR>_<YYYY-MM-DD>.parquet, one file per UTC day).

Clocks:
* ``NY``: trading day = 17:00 -> 17:00 New York (the FX rollover), labelled by the date it ends.
  Follows US daylight saving, so London/New York session effects stay on the same slot.
* ``UTC``: trading day = 00:00 -> 24:00 UTC (= Tokyo time minus 9 h; Japan has no DST).
Slot k (0..1440) is the boundary k minutes after the day start; the price at a boundary is the close of
the minute that ends there. Only Monday-Friday labels are used.

Discipline enforced here: discovery code refuses data on or after ``CONFIRMATION[0]``; nothing reads
on or after ``HOLDOUT_START``; confirmation reads the frozen candidate file written by discovery.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import vendor_dir, work_dir

FXS_VERSION = "0.2.0"   # 0.2.0: per-pair families (FXS 002), completeness check; holdout stage (FXS 005)
NY = "America/New_York"
MAJORS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
CROSSES = ("EURGBP", "EURCHF", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CADCHF", "AUDNZD")
PAIRS = MAJORS + CROSSES
CLOCKS = ("NY", "UTC")
SLOTS = 1440
DISCOVERY = (dt.date(2016, 1, 1), dt.date(2020, 12, 31))
CONFIRMATION = (dt.date(2021, 1, 1), dt.date(2025, 12, 31))
HOLDOUT_START = dt.date(2026, 1, 1)
NY_UTC_OFFSETS = (21 * 60, 22 * 60)      # 17:00 New York in UTC minutes: summer (EDT), winter (EST)


class HoldoutError(RuntimeError):
    """Raised when code would read the sealed 2026 holdout."""


class WindowError(RuntimeError):
    """Raised when discovery code would read confirmation data."""


class IncompleteDataError(RuntimeError):
    """Raised when a pair's day files for the requested window are not all downloaded yet."""


@dataclass(frozen=True)
class FXSConfig:
    """Pre-registered specification (discussions/FX_Intraday_Seasonality/001_claude_fxs01_preregistration.md)."""
    durations: tuple = (15, 30, 60, 120, 240)
    start_step: int = 5
    min_coverage: float = 0.80           # share of minutes with ticks for a day to count
    nw_lags: int = 5
    fdr_q: float = 0.05                  # discovery Benjamini-Hochberg, all pairs x clocks x intervals
    min_years_same_sign: int = 4         # of 5 discovery years
    commission_bps_rt: float = 0.70      # $35 per $1M per side
    max_candidates: int = 10             # per pair (FXS 002; was 40 over all pairs)
    min_file_share: float = 0.99         # share of expected day files that must exist
    n_placebo: int = 20
    conf_fdr_q: float = 0.05             # confirmation Benjamini-Hochberg across candidates
    alpha: float = 0.025
    blocks: tuple = (10, 20)
    n_boot: int = 10_000
    min_years_positive: int = 3          # of 5 confirmation years, net
    seed: int = 42


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def slot_label(clock: str, k: int) -> str:
    """Wall-clock time of boundary k in that clock."""
    m = (17 * 60 + k) % SLOTS if clock == "NY" else k % SLOTS
    return f"{clock} {m // 60:02d}:{m % 60:02d}"


# ----------------------------------------------------------------------------- data access
def candles_root(pair: str) -> Path:
    base = Path(os.environ["DUKASCOPY_DIR"]) if os.environ.get("DUKASCOPY_DIR") else vendor_dir("Dukascopy")
    return base / pair / "candles_m1"


def day_files(pair: str, start: dt.date, end: dt.date) -> list[Path]:
    """Existing candle files for UTC days start..end, in date order."""
    root, out, d = candles_root(pair), [], start
    while d <= end:
        f = root / f"{d:%Y}" / f"{pair}_{d:%Y-%m-%d}.parquet"
        if f.exists():
            out.append(f)
        d += dt.timedelta(days=1)
    return out


def download_coverage(pair: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Per year: Monday-Friday UTC days expected vs day files present."""
    files = {f.stem.split("_")[-1] for f in day_files(pair, start, end)}
    days = pd.bdate_range(start, end)
    df = pd.DataFrame({"year": days.year, "present": [d.strftime("%Y-%m-%d") in files for d in days]})
    out = df.groupby("year")["present"].agg(expected="size", present="sum")
    out["share"] = out["present"] / out["expected"]
    return out


def file_coverage(pair: str, start: dt.date, end: dt.date) -> dict:
    """Expected day files for a window (UTC days start-1 .. end, Saturdays excluded, as the downloader
    fetches them) and which are missing. Missing days are either still downloading or logged gaps."""
    exp, d = [], start - dt.timedelta(days=1)
    while d <= end:
        if d.weekday() != 5:
            exp.append(d)
        d += dt.timedelta(days=1)
    have = {f.stem.split("_")[-1] for f in day_files(pair, exp[0], exp[-1])}
    missing = [x for x in exp if x.isoformat() not in have]
    gaps = candles_root(pair) / "_gaps.csv"
    logged = set(pd.read_csv(gaps)["date"].astype(str)) if gaps.exists() and gaps.stat().st_size > 20 else set()
    return {"pair": pair, "expected": len(exp), "present": len(exp) - len(missing), "missing": missing,
            "missing_logged_as_gap": [x for x in missing if x.isoformat() in logged],
            "share": (len(exp) - len(missing)) / len(exp)}


def _guard(start: dt.date, end: dt.date, stage: str) -> None:
    if stage == "holdout":
        # one pre-registered look (FXS 005): the window must lie entirely in 2026
        if start < HOLDOUT_START:
            raise WindowError(f"holdout window must start on or after {HOLDOUT_START}; got {start}.")
        return
    if end >= HOLDOUT_START:
        raise HoldoutError(f"{end} reaches the sealed holdout (>= {HOLDOUT_START}).")
    if stage == "discovery" and end >= CONFIRMATION[0]:
        raise WindowError(f"discovery must end before {CONFIRMATION[0]}; got {end}.")


def load_minutes(pair: str, start: dt.date, end: dt.date, stage: str, refresh: bool = False,
                 min_file_share: float = 0.0) -> pd.DataFrame:
    """UTC minute series (index: minute start) with columns bid, ask (candle close, forward-filled) and
    tick (bool: the minute had quotes). Reads UTC day files start-1 .. end (enough for both clocks).

    With min_file_share > 0, refuses (IncompleteDataError) while fewer than that share of the expected
    day files exist, so a pair that is still downloading is not analysed on partial data."""
    _guard(start, end, stage)
    if min_file_share > 0:
        cov = file_coverage(pair, start, end)
        if cov["share"] < min_file_share:
            raise IncompleteDataError(
                f"{pair}: {cov['present']}/{cov['expected']} day files for {start}..{end} ({cov['share']:.1%}); "
                "wait for the download to finish, or rerun it to fill gaps.")
    files = day_files(pair, start - dt.timedelta(days=1), end)
    if not files:
        raise FileNotFoundError(f"No candle files for {pair} under {candles_root(pair)}")
    # the key includes the file count and last file, so a cache built during a running download is not reused
    last = files[-1].stem.split("_")[-1]
    cache = (work_dir() / "FXSeasonality" / "minutes"
             / f"{pair}_{start}_{end}_{FXS_VERSION}_n{len(files)}_{last}.parquet")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    frames = [pd.read_parquet(f, columns=["ts", "bid_c", "ask_c", "bid_v", "ask_v"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    df = df.reindex(full)
    tick = (df["bid_v"].fillna(0) > 0) | (df["ask_v"].fillna(0) > 0)
    out = pd.DataFrame({"bid": df["bid_c"].ffill(), "ask": df["ask_c"].ffill(), "tick": tick.to_numpy()}, index=full)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


@dataclass
class Panel:
    pair: str
    clock: str
    dates: np.ndarray          # day labels, datetime64[D], chronological
    bid: np.ndarray            # (days, 1441) bid at each boundary
    ask: np.ndarray            # (days, 1441)
    coverage: np.ndarray       # (days,) share of minutes with ticks

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid + self.ask)

    @property
    def logmid(self) -> np.ndarray:
        return np.log(self.mid)

    @property
    def years(self) -> np.ndarray:
        return self.dates.astype("datetime64[Y]").astype(int) + 1970

    def spread_bps(self) -> np.ndarray:
        return (self.ask - self.bid) / self.mid * 1e4


def build_panel(minutes: pd.DataFrame, pair: str, clock: str, start: dt.date, end: dt.date,
                min_coverage: float) -> Panel:
    """Stack days: one row per trading day, one column per boundary 0..1440."""
    ts = minutes.index
    if clock == "NY":
        loc = ts.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7)   # 17:00 NY -> 00:00 of the label
    elif clock == "UTC":
        loc = ts.tz_convert("UTC").tz_localize(None)
    else:
        raise ValueError(clock)
    label = loc.normalize()
    slot = ((loc - label) // pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    lab = label.to_numpy().astype("datetime64[D]")
    keep = ((pd.DatetimeIndex(label).weekday < 5) & (lab >= np.datetime64(start)) & (lab <= np.datetime64(end))
            & (slot >= 0) & (slot < SLOTS))
    # price at boundary k = close of the minute ending at k = close of the minute starting at k-1
    bid_c, ask_c = minutes["bid"].to_numpy(), minutes["ask"].to_numpy()
    bid_at, ask_at = np.r_[np.nan, bid_c[:-1]], np.r_[np.nan, ask_c[:-1]]
    tick = minutes["tick"].to_numpy()
    lab, slot = lab[keep], slot[keep]
    dates = np.unique(lab)
    r = np.searchsorted(dates, lab)
    n = len(dates)
    B = np.full((n, SLOTS + 1), np.nan)
    A = np.full((n, SLOTS + 1), np.nan)
    T = np.zeros((n, SLOTS), bool)
    B[r, slot], A[r, slot] = bid_at[keep], ask_at[keep]
    T[r, slot] = tick[keep]
    last = slot == SLOTS - 1                       # boundary 1440 = close of the day's last minute
    B[r[last], SLOTS], A[r[last], SLOTS] = bid_c[keep][last], ask_c[keep][last]
    cov = T.mean(axis=1)
    ok = (cov >= min_coverage) & np.isfinite(B).all(axis=1) & np.isfinite(A).all(axis=1) & (A >= B).all(axis=1)
    return Panel(pair, clock, dates[ok], B[ok], A[ok], cov[ok])


def load_panels(pairs, start: dt.date, end: dt.date, stage: str, cfg: FXSConfig, clocks=CLOCKS) -> dict:
    out = {}
    for p in pairs:
        m = load_minutes(p, start, end, stage, min_file_share=cfg.min_file_share)
        for c in clocks:
            out[(p, c)] = build_panel(m, p, c, start, end, cfg.min_coverage)
    return out


# ----------------------------------------------------------------------------- stacked profiles
def profile(panel: Panel) -> pd.DataFrame:
    """Average day: mean and standard error of the cumulative log-mid return from boundary 0, in bps;
    mean absolute 1-minute return and median spread per boundary."""
    lp = panel.logmid
    C = (lp - lp[:, :1]) * 1e4
    r = np.abs(np.diff(lp, axis=1)) * 1e4
    return pd.DataFrame({"cum_mean_bps": C.mean(0), "cum_se_bps": C.std(0, ddof=1) / np.sqrt(len(C)),
                         "abs_ret_bps": np.r_[np.nan, r.mean(0)], "spread_bps": np.median(panel.spread_bps(), axis=0)},
                        index=pd.RangeIndex(SLOTS + 1, name="boundary"))


def hourly_stats(panel: Panel) -> pd.DataFrame:
    """Mean return (bps) and t of each of the 24 one-hour blocks, overall and per year."""
    lp = panel.logmid
    R = (lp[:, 60::60] - lp[:, 0:SLOTS:60]) * 1e4
    rows = []
    for y in [None, *np.unique(panel.years)]:
        sel = slice(None) if y is None else panel.years == y
        x = R[sel]
        m, se, t = nw_tstats(x, 0)
        for h in range(24):
            rows.append({"year": "all" if y is None else int(y), "hour": h, "mean_bps": m[h], "t": t[h]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def intervals(cfg: FXSConfig) -> np.ndarray:
    return np.array([(s, s + d) for d in cfg.durations for s in range(0, SLOTS, cfg.start_step) if s + d <= SLOTS])


def nw_tstats(R: np.ndarray, lags: int):
    """Column means, Newey-West standard errors (Bartlett) and t-statistics; rows are days in order."""
    n = R.shape[0]
    m = R.mean(axis=0)
    x = R - m
    v = (x * x).sum(axis=0) / n
    for l in range(1, lags + 1):
        v = v + 2 * (1 - l / (lags + 1)) * (x[l:] * x[:-l]).sum(axis=0) / n
    se = np.sqrt(v / n)
    return m, se, m / se


def bh_reject(p: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: reject mask and adjusted q-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    qvals = np.empty(n)
    qvals[order] = np.minimum(qv, 1.0)
    return qvals <= q, qvals


def scan_panel(panel: Panel, cfg: FXSConfig, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Test every interval of one pair x clock. With rng, each day's 1-minute returns are first rotated by
    an independent random offset (placebo: same returns, time-of-day alignment destroyed)."""
    from scipy.stats import norm
    lp = panel.logmid
    if rng is not None:
        r = np.diff(lp, axis=1)
        k = rng.integers(0, SLOTS, len(r))
        idx = (np.arange(SLOTS)[None, :] - k[:, None]) % SLOTS
        r = r[np.arange(len(r))[:, None], idx]
        lp = np.concatenate([lp[:, :1], lp[:, :1] + np.cumsum(r, axis=1)], axis=1)
    iv = intervals(cfg)
    R = (lp[:, iv[:, 1]] - lp[:, iv[:, 0]]) * 1e4
    m, se, t = nw_tstats(R, cfg.nw_lags)
    yrs = panel.years
    sign = np.sign(m)
    same = np.zeros(len(iv), int)
    for y in np.unique(yrs):
        same += (np.sign(R[yrs == y].mean(axis=0)) == sign).astype(int)
    spr = np.median(panel.spread_bps(), axis=0)
    cost = (spr[iv[:, 0]] + spr[iv[:, 1]]) / 2 + cfg.commission_bps_rt
    return pd.DataFrame({"pair": panel.pair, "clock": panel.clock, "start": iv[:, 0], "end": iv[:, 1],
                         "minutes": iv[:, 1] - iv[:, 0], "n_days": len(R), "mean_bps": m, "se_bps": se, "t": t,
                         "p": 2 * norm.sf(np.abs(t)), "years_same_sign": same, "cost_bps": cost})


def _utc_segments(clock: str, s: int, e: int) -> list[tuple[int, int]]:
    """UTC minute-of-day segments covered by an interval (NY intervals under both DST offsets)."""
    offs = (0,) if clock == "UTC" else NY_UTC_OFFSETS
    segs = []
    for o in offs:
        a, b = (s + o) % SLOTS, (s + o) % SLOTS + (e - s)
        segs += [(a, b)] if b <= SLOTS else [(a, SLOTS), (0, b - SLOTS)]
    return segs


def _overlap(x: list, y: list) -> bool:
    return any(a1 < b2 and a2 < b1 for a1, b1 in x for a2, b2 in y)


def select_candidates(scan: pd.DataFrame, cfg: FXSConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply D1 (BH over every test in `scan`: one pair's 2,704 tests under FXS 002), D2 (year consistency),
    D3 (|mean| > median cost), then keep the strongest non-overlapping intervals per pair (across both
    clocks), at most cfg.max_candidates per pair."""
    scan = scan.copy()
    scan["D1"], scan["q"] = bh_reject(scan["p"].to_numpy(), cfg.fdr_q)
    scan["D2"] = scan["years_same_sign"] >= cfg.min_years_same_sign
    scan["D3"] = scan["mean_bps"].abs() > scan["cost_bps"]
    surv = scan[scan["D1"] & scan["D2"] & scan["D3"]].copy()
    surv = surv.reindex(surv["t"].abs().sort_values(ascending=False).index)
    kept, segs = [], {}
    for ix, r in surv.iterrows():
        sg = _utc_segments(r["clock"], int(r["start"]), int(r["end"]))
        have = segs.setdefault(r["pair"], [])
        if len(have) >= cfg.max_candidates or any(_overlap(sg, other) for other in have):
            continue
        have.append(sg)
        kept.append(ix)
    cand = surv.loc[kept].reset_index(drop=True)
    cand.insert(0, "id", [f"{p}-{i + 1:02d}" for p, i in zip(cand["pair"], cand.groupby("pair").cumcount())])
    cand["direction"] = np.sign(cand["mean_bps"]).astype(int)
    cand["start_time"] = [slot_label(c, s) for c, s in zip(cand["clock"], cand["start"])]
    cand["end_time"] = [slot_label(c, e) for c, e in zip(cand["clock"], cand["end"])]
    return scan, cand


def placebo_counts(panels: dict, cfg: FXSConfig) -> pd.DataFrame:
    """Number of survivors of the full discovery pipeline on rotated-day placebos."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for k in range(cfg.n_placebo):
        sc = pd.concat([scan_panel(p, cfg, rng) for p in panels.values()], ignore_index=True)
        s, c = select_candidates(sc, cfg)
        rows.append({"run": k, "D1": int(s["D1"].sum()), "D1_D2_D3": int((s["D1"] & s["D2"] & s["D3"]).sum()),
                     "candidates": len(c)})
    return pd.DataFrame(rows)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hash_ok(path: Path, expected: str) -> bool:
    """True if the file's SHA-256 matches `expected` with LF or CRLF line endings. Git on Windows stores
    text with LF and checks it out with CRLF, so the same content can carry either hash."""
    b = Path(path).read_bytes()
    lf = b.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(x).hexdigest() for x in (b, lf, lf.replace(b"\n", b"\r\n"))}


# ----------------------------------------------------------------------------- confirmation
def candidate_trades(panel: Panel, start: int, end: int, direction: int, cfg: FXSConfig) -> pd.DataFrame:
    """Daily trade on quoted prices: long buys the ask at `start` and sells the bid at `end`; short sells the
    bid at `start` and buys the ask at `end`. Returns gross (after spread) and net bps per day."""
    b, a = panel.bid, panel.ask
    if direction > 0:
        gross = np.log(b[:, end] / a[:, start]) * 1e4
    else:
        gross = np.log(b[:, start] / a[:, end]) * 1e4
    mid = 0.5 * (b[:, start] + a[:, start])
    return pd.DataFrame({"date": panel.dates, "year": panel.years, "gross_bps": gross,
                         "net_bps": gross - cfg.commission_bps_rt, "net2_bps": gross - 2 * cfg.commission_bps_rt,
                         "gross_pips": gross / 1e4 * mid / pip_size(panel.pair)})


def block_bootstrap_lower(x, block: int, n_boot: int, alpha: float, seed: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, (n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return float(np.quantile(x[idx].mean(axis=1), alpha))


def confirm(trades: dict[str, pd.DataFrame], cfg: FXSConfig) -> pd.DataFrame:
    """Gates K1-K4 for every frozen candidate (FXS 001 §5)."""
    from scipy.stats import norm
    rows = []
    for cid, tr in trades.items():
        g = tr["gross_bps"].to_numpy()[:, None]
        m, se, t = nw_tstats(g, cfg.nw_lags)
        yearly = tr.groupby("year")["net_bps"].mean()
        rows.append({"id": cid, "n_days": len(tr), "gross_bps": float(m[0]), "t": float(t[0]),
                     "p_one_sided": float(norm.sf(t[0])), "net_bps": float(tr["net_bps"].mean()),
                     "net2_bps": float(tr["net2_bps"].mean()), "gross_pips": float(tr["gross_pips"].mean()),
                     "win_rate": float((tr["net_bps"] > 0).mean()),
                     "boot_net_lower": [block_bootstrap_lower(tr["net_bps"], b, cfg.n_boot, cfg.alpha, cfg.seed)
                                        for b in cfg.blocks],
                     "years_net_positive": int((yearly > 0).sum()), "years": int(len(yearly))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["K1"], out["q"] = bh_reject(out["p_one_sided"].to_numpy(), cfg.conf_fdr_q)
    out["K1"] = out["K1"] & (out["gross_bps"] > 0)
    out["K2"] = out["boot_net_lower"].apply(lambda v: all(np.isfinite(v)) and min(v) > 0)
    out["K3"] = out["net2_bps"] > 0
    out["K4"] = out["years_net_positive"] >= cfg.min_years_positive
    out["verdict"] = np.where(out[["K1", "K2", "K3", "K4"]].all(axis=1), "CONFIRMED", "NOT_CONFIRMED")
    return out


# ----------------------------------------------------------------------------- validation and synthetic data
def validate_against_ticks(pair: str, candle_root: Path, tick_root: Path, days: list[dt.date]) -> pd.DataFrame:
    """Compare candle bid/ask close with the last tick of each minute, for the hours present in the ticks."""
    rows = []
    for d in days:
        fc = candle_root / f"{d:%Y}" / f"{pair}_{d:%Y-%m-%d}.parquet"
        ft = tick_root / f"{d:%Y}" / f"{pair}_{d:%Y-%m-%d}.parquet"
        if not (fc.exists() and ft.exists()):
            continue
        c = pd.read_parquet(fc).set_index("ts")
        t = pd.read_parquet(ft, columns=["ts", "bid", "ask"])
        t["minute"] = pd.to_datetime(t["ts"], utc=True).dt.floor("1min")
        last = t.groupby("minute")[["bid", "ask"]].last()
        j = last.join(c[["bid_c", "ask_c"]], how="inner")
        if j.empty:
            continue
        rows.append({"date": d, "minutes": len(j),
                     "bid_close_match": float(np.isclose(j["bid"], j["bid_c"], atol=1e-9).mean()),
                     "ask_close_match": float(np.isclose(j["ask"], j["ask_c"], atol=1e-9).mean())})
    return pd.DataFrame(rows)


def write_synthetic_candles(base: Path, pair: str, start: dt.date, end: dt.date, seed: int,
                            drift_bps: float = 0.0, drift_ny: tuple = (8, 9), spread_bps: float = 0.5,
                            price: float = 1.10) -> Path:
    """Synthetic 1-minute candles in the downloader layout: random-walk mid, closed from Friday 17:00 to
    Sunday 17:00 New York, and an added drift of `drift_bps` spread evenly over [drift_ny) New York hours."""
    rng = np.random.default_rng(seed)
    root = Path(base) / pair / "candles_m1"
    lp = np.log(price)
    d = start
    while d <= end:
        if d.weekday() != 5:
            ts = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=SLOTS, freq="1min")
            ny = ts.tz_convert(NY)
            open_ = ~(((ny.weekday == 4) & (ny.hour >= 17)) | (ny.weekday == 5) | ((ny.weekday == 6) & (ny.hour < 17)))
            r = rng.normal(0, 0.6e-4, SLOTS) * open_
            h0, h1 = drift_ny
            r = r + np.where(open_ & (ny.hour >= h0) & (ny.hour < h1), drift_bps * 1e-4 / ((h1 - h0) * 60), 0.0)
            path = lp + np.cumsum(r)
            lp = path[-1]
            mid = np.exp(path)
            half = mid * spread_bps * 1e-4 / 2
            dig = 3 if pair.endswith("JPY") else 5
            bid, ask = np.round(mid - half, dig), np.round(mid + half, dig)
            v = np.where(open_, 1.0, 0.0).astype(np.float32)
            df = pd.DataFrame({"ts": ts, "bid_o": bid, "bid_h": bid, "bid_l": bid, "bid_c": bid, "bid_v": v,
                               "ask_o": ask, "ask_h": ask, "ask_l": ask, "ask_c": ask, "ask_v": v})
            f = root / f"{d:%Y}" / f"{pair}_{d:%Y-%m-%d}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(f, index=False)
        d += dt.timedelta(days=1)
    return root


# ----------------------------------------------------------------------------- per-pair exploration (descriptive)
def yearly_profiles(panel: Panel) -> pd.DataFrame:
    """Cumulative mean log-mid return (bps) from boundary 0, one column per year."""
    lp = panel.logmid
    C = (lp - lp[:, :1]) * 1e4
    return pd.DataFrame({int(y): C[panel.years == y].mean(axis=0) for y in np.unique(panel.years)},
                        index=pd.RangeIndex(SLOTS + 1, name="boundary"))


def block_consistency(panel: Panel, block: int = 15) -> pd.DataFrame:
    """For each block of `block` minutes: mean (bps), t (no lags), share of days up, and number of years
    whose mean has the sign of the overall mean."""
    lp = panel.logmid
    s = np.arange(0, SLOTS, block)
    R = (lp[:, s + block] - lp[:, s]) * 1e4
    m, se, t = nw_tstats(R, 0)
    yrs = panel.years
    same = sum((np.sign(R[yrs == y].mean(axis=0)) == np.sign(m)).astype(int) for y in np.unique(yrs))
    return pd.DataFrame({"start": s, "time": [slot_label(panel.clock, k) for k in s], "mean_bps": m, "t": t,
                         "share_up": (R > 0).mean(axis=0), "years_same_sign": same, "n_years": len(np.unique(yrs))})


def bonferroni_t(n_tests: int, alpha: float = 0.05) -> float:
    """|t| above which a two-sided normal test is significant after Bonferroni over n_tests. Any interval
    beyond it also passes Benjamini-Hochberg over the same n_tests."""
    from scipy.stats import norm
    return float(norm.isf(alpha / (2 * n_tests)))


def pooled_n_tests(cfg: FXSConfig) -> int:
    return len(intervals(cfg)) * len(CLOCKS) * len(PAIRS)


# ----------------------------------------------------------------------------- account growth (FXS 03, sizing only, no gates)
def load_frozen_candidates(base: Path) -> pd.DataFrame:
    """All pairs' frozen candidate files under the branch folder, each verified against its summary."""
    import json
    rows = []
    for p in PAIRS:
        f, s = Path(base) / p / "01" / "candidates_frozen.csv", Path(base) / p / "01" / "summary.json"
        if not f.exists():
            continue
        sha = json.loads(s.read_text())["candidates_sha256"]
        if not frozen_hash_ok(f, sha):
            raise RuntimeError(f"{f} changed after discovery")
        c = pd.read_csv(f)
        if len(c):
            rows.append(c)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def structure_returns(cand: pd.DataFrame, start: dt.date, end: dt.date, stage: str, cfg: FXSConfig,
                      panels: dict | None = None) -> pd.DataFrame:
    """Daily net return (bps, after spread and commission) of every structure: rows = dates, columns = ids.
    A day on which a structure has no valid quotes is 0 (no trade)."""
    pairs = sorted(set(cand["pair"]))
    panels = panels or load_panels(pairs, start, end, stage, cfg)
    cols = {}
    for _, r in cand.iterrows():
        tr = candidate_trades(panels[(r["pair"], r["clock"])], int(r["start"]), int(r["end"]), int(r["direction"]), cfg)
        cols[r["id"]] = pd.Series(tr["net_bps"].to_numpy(), index=pd.to_datetime(tr["date"]))
    return pd.DataFrame(cols).sort_index().fillna(0.0)


def max_concurrent(cand: pd.DataFrame) -> int:
    """Largest number of structures open at the same UTC minute (New York intervals under both DST offsets)."""
    best = 0
    for o in range(len(NY_UTC_OFFSETS)):
        occ = np.zeros(SLOTS, int)
        for _, r in cand.iterrows():
            off = 0 if r["clock"] == "UTC" else NY_UTC_OFFSETS[o]
            m = (np.arange(int(r["start"]), int(r["end"])) + off) % SLOTS
            occ[m] += 1
        best = max(best, int(occ.max()))
    return best


def kelly_weights(R_bps: pd.DataFrame, fraction: float = 0.5, cap_total: float | None = None) -> pd.Series:
    """Fractional Kelly notional per structure (multiple of equity) from daily net returns, long-only:
    w = fraction * inv(Cov) * mean, structures with a negative weight dropped and the rest re-solved.
    Optionally scaled down so the sum of weights is at most cap_total."""
    R = R_bps / 1e4
    keep = list(R.columns[R.mean() > 0])
    while keep:
        mu, S = R[keep].mean().to_numpy(), np.cov(R[keep].to_numpy(), rowvar=False).reshape(len(keep), len(keep))
        w = fraction * np.linalg.solve(S + 1e-12 * np.eye(len(keep)), mu)
        if (w >= 0).all():
            break
        keep = [k for k, x in zip(keep, w) if x > 0]
    out = pd.Series(0.0, index=R.columns)
    if keep:
        out[keep] = w
    if cap_total is not None and out.sum() > cap_total:
        out *= cap_total / out.sum()
    return out


def simulate_account(R_bps: pd.DataFrame, weights: pd.Series, equity0: float, min_units: int = 1_000) -> pd.DataFrame:
    """Compounded account: each day, structure i trades floor(equity * w_i / min_units) * min_units units of
    notional and earns that times its net return. Equity is re-marked daily, so size grows with equity.
    An account at or below zero stops."""
    w = weights.reindex(R_bps.columns).fillna(0.0).to_numpy()
    R = R_bps.to_numpy() / 1e4
    eq = np.empty(len(R))
    e = float(equity0)
    for k in range(len(R)):
        if e > 0:
            units = np.floor(e * w / min_units) * min_units
            e = max(e + float(units @ R[k]), 0.0)
        eq[k] = e
    out = pd.DataFrame({"equity": eq}, index=R_bps.index)
    out["ret"] = out["equity"].pct_change().fillna(out["equity"].iloc[0] / equity0 - 1)
    out["drawdown"] = out["equity"] / out["equity"].cummax().clip(lower=equity0) - 1
    return out


def account_stats(acc: pd.DataFrame, equity0: float) -> dict:
    yrs = max((acc.index[-1] - acc.index[0]).days / 365.25, 1e-9)
    end = float(acc["equity"].iloc[-1])
    r = acc["ret"]
    yearly = acc["equity"].resample("YE").last()
    yr_ret = yearly.pct_change()
    yr_ret.iloc[0] = yearly.iloc[0] / equity0 - 1
    return {"final_equity": end, "cagr": (end / equity0) ** (1 / yrs) - 1 if end > 0 else -1.0,
            "ann_vol": float(r.std() * np.sqrt(252)), "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "max_drawdown": float(acc["drawdown"].min()), "worst_year": float(yr_ret.min()), "best_year": float(yr_ret.max()),
            "ruined": bool(end <= 0)}


# ----------------------------------------------------------------------------- prop-firm costs and rules (FXS 04, post hoc, no gates)
# Approximate USD value of one unit of each base currency over 2021-2025, to turn a per-lot commission into bps.
BASE_USD = {"EUR": 1.10, "GBP": 1.27, "USD": 1.0, "AUD": 0.70, "NZD": 0.63, "CAD": 0.75, "CHF": 1.10}


def commission_bps_per_lot(pair: str, usd_per_lot_round_trip: float, lot_units: int = 100_000) -> float:
    """Round-trip commission in bps of notional for a broker charging a fixed USD amount per lot (lot = 100,000
    units of the base currency)."""
    return usd_per_lot_round_trip / (lot_units * BASE_USD[pair[:3]]) * 1e4


def bps_to_pips(bps, price, pair: str):
    """Convert a return in bps of price to pips at that price."""
    return np.asarray(bps, float) * 1e-4 * np.asarray(price, float) / pip_size(pair)


def usd_per_pip_per_lot(pair: str, price, lot_units: int = 100_000):
    """USD value of one pip on one 100,000-unit lot: quote-currency pip value converted to USD
    (USD-quoted pairs: 10; USD-based pairs: 10 / price x pip scale; crosses via BASE_USD / price)."""
    price = np.asarray(price, float)
    quote_value = lot_units * pip_size(pair)                    # in quote currency
    if pair.endswith("USD"):
        return np.full_like(price, quote_value)
    if pair.startswith("USD"):
        return quote_value / price
    return quote_value / price * BASE_USD[pair[:3]]          # quote -> base at price, base -> USD


def structure_detail(cand: pd.DataFrame, start: dt.date, end: dt.date, stage: str, cfg: FXSConfig,
                     panels: dict | None = None, with_price: bool = False):
    """Per structure and day: gross return after spread (bps) and maximum adverse excursion while open (bps,
    <= 0, marked at minute closes on the quote you would exit at). Rows = dates, columns = ids; no trade = 0."""
    pairs = sorted(set(cand["pair"]))
    panels = panels or load_panels(pairs, start, end, stage, cfg)
    G, M, P = {}, {}, {}
    for _, r in cand.iterrows():
        p = panels[(r["pair"], r["clock"])]
        s, e, d = int(r["start"]), int(r["end"]), int(r["direction"])
        if d > 0:
            path = np.log(p.bid[:, s:e + 1] / p.ask[:, [s]]) * 1e4
        else:
            path = np.log(p.bid[:, [s]] / p.ask[:, s:e + 1]) * 1e4
        idx = pd.to_datetime(p.dates)
        G[r["id"]] = pd.Series(path[:, -1], index=idx)
        M[r["id"]] = pd.Series(np.minimum(path.min(axis=1), 0.0), index=idx)
        P[r["id"]] = pd.Series(0.5 * (p.bid[:, s] + p.ask[:, s]), index=idx)
    out = (pd.DataFrame(G).sort_index().fillna(0.0), pd.DataFrame(M).sort_index().fillna(0.0))
    return (*out, pd.DataFrame(P).sort_index().ffill().bfill()) if with_price else out


@dataclass(frozen=True)
class PropRules:
    """Stellar 2-Step style rules (FundedNext, as published Oct 2026; confirm on the firm's site)."""
    daily_loss: float = 0.05        # of initial balance, measured from the day's starting equity
    max_loss: float = 0.10          # of initial balance, static floor
    target: float = 0.08            # phase 1 profit target (0.05 in phase 2; None in funded stage)
    min_days: int = 5


def run_prop_path(G: np.ndarray, M: np.ndarray, comm_bps: np.ndarray, w: np.ndarray, equity0: float,
                  rules: PropRules, max_days: int, compound: bool = False) -> tuple[str, int, float]:
    """One path through rows of G (gross bps) and M (adverse excursion bps), trading notional w x equity per
    structure. Breach is checked on the worst case of the day (all structures at their worst excursion at once,
    a conservative bound). Limits are fractions of the initial balance, so on a prop account size is normally kept
    on the initial balance (compound=False): profits are withdrawn, not compounded. Returns (outcome, days,
    final equity); outcome pass | daily_breach | max_breach | open."""
    e = float(equity0)
    floor = equity0 * (1 - rules.max_loss)
    for k in range(min(max_days, len(G))):
        base = e if compound else equity0
        units = np.floor(base * w / 1_000) * 1_000
        worst = e + float(units @ (M[k] - comm_bps)) / 1e4
        if e - worst >= rules.daily_loss * equity0:
            return "daily_breach", k + 1, worst
        if worst <= floor:
            return "max_breach", k + 1, worst
        e += float(units @ (G[k] - comm_bps)) / 1e4
        if rules.target is not None and e >= equity0 * (1 + rules.target) and k + 1 >= rules.min_days:
            return "pass", k + 1, e
    return "open", min(max_days, len(G)), e


def prop_odds(G: pd.DataFrame, M: pd.DataFrame, comm_bps: pd.Series, w: pd.Series, equity0: float,
              rules: PropRules, max_days: int, n_sims: int, seed: int, compound: bool = False) -> dict:
    """Bootstrap whole trading days (all structures together) from the sample; odds of each outcome."""
    rng = np.random.default_rng(seed)
    Gv, Mv = G.to_numpy(), M.to_numpy()
    c, wv = comm_bps.reindex(G.columns).to_numpy(), w.reindex(G.columns).fillna(0.0).to_numpy()
    out, days, ends = [], [], []
    for _ in range(n_sims):
        ix = rng.integers(0, len(Gv), max_days)
        o, d, e = run_prop_path(Gv[ix], Mv[ix], c, wv, equity0, rules, max_days, compound)
        out.append(o); days.append(d); ends.append(e)
    out, days, ends = np.array(out), np.array(days), np.array(ends)
    res = {k: float((out == k).mean()) for k in ("pass", "daily_breach", "max_breach", "open")}
    res["median_days_to_pass"] = float(np.median(days[out == "pass"])) if (out == "pass").any() else np.nan
    res["median_end_equity"] = float(np.median(ends))
    return res
