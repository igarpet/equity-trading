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

FXS_VERSION = "0.2.0"   # 0.2.0: per-pair families (FXS 002), completeness check
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
