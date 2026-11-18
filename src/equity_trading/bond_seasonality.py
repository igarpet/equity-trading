"""Bond CFD intraday seasonality (branch Bond_Intraday_Seasonality, BDS 001).

The FX stacked-day scan (``fx_seasonality``) applied to Dukascopy's three government-bond CFDs:
US T-Bond (USTBONDTRUSD), Euro Bund (BUNDTREUR), UK Long Gilt (UKGILTTRGBP).

Three windows, each scanned on its own:

* ``H1``   2018-2021 (first half; Bund and Gilt start in 2018 when their CFDs were launched)
* ``H2``   2022-2025 (second half)
* ``FULL`` 2018-2025

H1 candidates are confirmed on H2 (forward) and H2 candidates on H1 (backward). FULL is in-sample.
2026 is sealed: nothing here reads it.

What differs from the FX engine, and why:

* **Contract switches.** The CFD follows the front futures contract and jumps to the next contract's
  price during the daily break of the adjustment day, once a quarter. The jump is the calendar spread,
  not a market move, and its sign is persistent, so it would show up as fake time-of-day structure.
  Each quarter, the daily break with the largest price gap inside the roll window is taken as the
  switch, and the trading days containing it are dropped (``detect_rolls``).
* **Trading hours.** Bonds are not quoted 24 hours a day, so a day counts if its quoted minutes are at
  least ``rel_coverage`` of the window's median day (FX used 80% of all 1,440 minutes).
* **Missing days.** A trading day is dropped if any UTC day file it spans is missing (otherwise the
  last price would be carried across the missing day).
* **Costs.** The D3 cost and the confirmation net use a futures-equivalent cost (one tick of the
  matching future + 0.70 bp commission). Dukascopy's own CFD spread, 4-5 times wider, is reported as
  ``D3_cfd`` / ``cfd_net_bps``. Trading would be done in futures, not in these CFDs.
* **Year rules.** Only years with at least ``min_year_days`` days count; D2 needs the same sign in
  >= 75% of them, K4 a positive net in >= 60%.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import fx_seasonality as F

BDS_VERSION = "0.1.0"
SYMBOLS = {
    "USTBONDTRUSD": {"short": "UST", "name": "US T-Bond", "tick": 1 / 32, "future": "ZB (CBOT T-Bond)"},
    "BUNDTREUR": {"short": "BUND", "name": "Euro Bund", "tick": 0.01, "future": "FGBL (Eurex Bund)"},
    "UKGILTTRGBP": {"short": "GILT", "name": "UK Long Gilt", "tick": 0.01, "future": "Long Gilt (ICE)"},
}
WINDOWS = {"H1": (dt.date(2018, 1, 1), dt.date(2021, 12, 31)),
           "H2": (dt.date(2022, 1, 1), dt.date(2025, 12, 31)),
           "FULL": (dt.date(2018, 1, 1), dt.date(2025, 12, 31))}
HOLDOUT_START = dt.date(2026, 1, 1)
CLOCKS = F.CLOCKS
ROLL_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class BDSConfig:
    """Pre-registered specification (discussions/Bond_Intraday_Seasonality/001_claude_bds01_preregistration.md)."""
    durations: tuple = (15, 30, 60, 120, 240)
    start_step: int = 5
    rel_coverage: float = 0.5          # quoted minutes >= this share of the window's median day
    min_file_share: float = 0.95       # weekday files present from the symbol's first file
    min_year_days: int = 120           # a year counts in D2/K4 with at least this many days
    year_share_d2: float = 0.75        # D2: same sign in >= 75% of counted years
    year_share_k4: float = 0.60        # K4: net > 0 in >= 60% of counted years
    nw_lags: int = 5
    fdr_q: float = 0.05
    commission_bps_rt: float = 0.70
    max_candidates: int = 10
    n_placebo: int = 20
    conf_fdr_q: float = 0.05
    alpha: float = 0.025
    blocks: tuple = (10, 20)
    n_boot: int = 10_000
    seed: int = 42
    break_minutes: int = 30            # a gap of at least this many unquoted minutes is a break
    roll_extra_ratio: float = 8.0      # other breaks in a roll window this many times the median gap are also dropped
    trim: float = 0.01                 # trimmed-mean robustness check (each tail)

    def fx(self, **kw) -> F.FXSConfig:
        base = F.FXSConfig(durations=self.durations, start_step=self.start_step, nw_lags=self.nw_lags,
                           fdr_q=self.fdr_q, commission_bps_rt=self.commission_bps_rt,
                           max_candidates=self.max_candidates, n_placebo=self.n_placebo,
                           conf_fdr_q=self.conf_fdr_q, alpha=self.alpha, blocks=self.blocks,
                           n_boot=self.n_boot, seed=self.seed)
        return replace(base, **kw)


def guard(start: dt.date, end: dt.date) -> None:
    if end >= HOLDOUT_START:
        raise F.HoldoutError(f"{end} reaches the sealed holdout (>= {HOLDOUT_START}).")


def years_needed(n_years: int, share: float) -> int:
    return max(1, math.ceil(share * n_years - 1e-9))


# ----------------------------------------------------------------------------- data
@dataclass
class BondData:
    symbol: str
    window: str
    start: dt.date
    end: dt.date
    first_file: dt.date
    minutes: pd.DataFrame
    files: dict                # expected / present / missing / share / sunday_session
    rolls: pd.DataFrame
    panels: dict               # clock -> Panel (after all drops)
    drops: pd.DataFrame        # per clock: days built, dropped by reason, kept


def _labels(ts: pd.Timestamp) -> dict:
    """Day label of a UTC timestamp in each clock (NY day = 17:00 -> 17:00 New York, labelled by its end)."""
    ny = (ts.tz_convert(F.NY).tz_localize(None) + pd.Timedelta(hours=7)).normalize()
    return {"NY": np.datetime64(ny.date(), "D"), "UTC": np.datetime64(ts.tz_convert("UTC").date(), "D")}


def break_gaps(minutes: pd.DataFrame, break_minutes: int) -> pd.DataFrame:
    """Every break (>= break_minutes without quotes between two quoted minutes) and the mid move across it."""
    q = minutes[minutes["tick"]]
    ts = q.index
    mid = np.log(0.5 * (q["bid"].to_numpy() + q["ask"].to_numpy()))
    gap = np.diff(ts.to_numpy()).astype("timedelta64[m]").astype(np.int64)   # index unit may be ms or ns
    i = np.flatnonzero(gap >= break_minutes) + 1
    return pd.DataFrame({"before": ts[i - 1], "after": ts[i], "minutes": gap[i - 1],
                         "gap_bps": (mid[i] - mid[i - 1]) * 1e4})


def detect_rolls(minutes: pd.DataFrame, start: dt.date, end: dt.date, cfg: BDSConfig) -> pd.DataFrame:
    """One contract switch per quarter: the break with the largest |gap| between the 10th of the month before
    a quarterly expiry month and the end of that month. Other breaks in the same window larger than both
    half that gap and `roll_extra_ratio` x the median |gap| are also dropped (flag `extra`)."""
    br = break_gaps(minutes, cfg.break_minutes)
    if br.empty:
        return br.assign(quarter=[], ratio=[], extra=[])
    med = float(br["gap_bps"].abs().median()) or 1e-9
    rows = []
    for y in range(start.year, end.year + 1):
        for m in ROLL_MONTHS:
            a = pd.Timestamp(dt.date(y, m - 1, 10), tz="UTC")
            b = pd.Timestamp(dt.date(y + (m == 12), m % 12 + 1, 1), tz="UTC")
            w = br[(br["after"] >= a) & (br["after"] < b) & (br["after"].dt.date >= start) & (br["after"].dt.date <= end)]
            if w.empty:
                continue
            top = w["gap_bps"].abs().idxmax()
            big = float(abs(w.loc[top, "gap_bps"]))
            for ix, r in w.iterrows():
                g = abs(r["gap_bps"])
                if ix == top or (g >= 0.5 * big and g >= cfg.roll_extra_ratio * med):
                    rows.append({"quarter": f"{y}-{m:02d}", "before": r["before"], "after": r["after"],
                                 "break_minutes": int(r["minutes"]), "gap_bps": float(r["gap_bps"]),
                                 "ratio": g / med, "extra": ix != top})
    return pd.DataFrame(rows)


def _subset(p: F.Panel, keep: np.ndarray) -> F.Panel:
    return F.Panel(p.pair, p.clock, p.dates[keep], p.bid[keep], p.ask[keep], p.coverage[keep])


def load(symbol: str, window: str, cfg: BDSConfig, drop_rolls: bool = True) -> BondData:
    start, end = WINDOWS[window]
    guard(start, end)
    files = F.day_files(symbol, start - dt.timedelta(days=1), end)
    if not files:
        raise FileNotFoundError(f"No candle files for {symbol} under {F.candles_root(symbol)}")
    present = {dt.date.fromisoformat(f.stem.split("_")[-1]) for f in files}
    first = min(d for d in present if d >= start) if any(d >= start for d in present) else None
    if first is None:
        raise FileNotFoundError(f"No candle files for {symbol} in {start}..{end}")
    minutes = F.load_minutes(symbol, first, end, "confirmation")
    live = minutes["tick"].groupby(minutes.index.tz_convert("UTC").date).sum()
    sundays = [d for d in present if d.weekday() == 6 and first <= d <= end]
    sunday_session = bool(sundays) and float(np.mean([live.get(d, 0) > 0 for d in sundays])) >= 0.5
    exp = [first + dt.timedelta(n) for n in range((end - first).days + 1)]
    weekdays = [d for d in exp if d.weekday() < 5]
    expected = set(weekdays) | ({d for d in exp if d.weekday() == 6} if sunday_session else set())
    missing = sorted(d for d in expected if d not in present)
    fs = {"first_file": first, "expected_weekdays": len(weekdays),
          "present_weekdays": sum(d in present for d in weekdays),
          "share": sum(d in present for d in weekdays) / len(weekdays), "sunday_session": sunday_session,
          "missing": missing}
    rolls = detect_rolls(minutes, first, end, cfg)
    roll_lab = {c: set() for c in CLOCKS}
    if drop_rolls:
        for _, r in rolls.iterrows():
            for t in (r["before"], r["after"]):
                for c, lab in _labels(t).items():
                    roll_lab[c].add(lab)
    miss = {np.datetime64(d, "D") for d in missing}
    panels, drops = {}, []
    for c in CLOCKS:
        p = F.build_panel(minutes, symbol, c, first, end, 0.0)
        span = [p.dates - np.timedelta64(1, "D"), p.dates] if c == "NY" else [p.dates]
        m_missing = np.zeros(len(p.dates), bool)
        for s in span:
            m_missing |= np.isin(s, list(miss)) if miss else False
        m_cov = p.coverage < cfg.rel_coverage * float(np.median(p.coverage))
        m_roll = np.isin(p.dates, list(roll_lab[c])) if roll_lab[c] else np.zeros(len(p.dates), bool)
        keep = ~(m_missing | m_cov | m_roll)
        panels[c] = _subset(p, keep)
        drops.append({"clock": c, "days built": len(p.dates), "missing file": int(m_missing.sum()),
                      "low coverage": int((m_cov & ~m_missing).sum()), "contract switch": int((m_roll & ~m_missing & ~m_cov).sum()),
                      "kept": int(keep.sum())})
    return BondData(symbol, window, start, end, first, minutes, fs, rolls, panels, pd.DataFrame(drops))


def counted_years(panel: F.Panel, cfg: BDSConfig) -> list[int]:
    y, n = np.unique(panel.years, return_counts=True)
    return [int(a) for a, b in zip(y, n) if b >= cfg.min_year_days]


def tick_bps(symbol: str, mid) -> np.ndarray:
    return SYMBOLS[symbol]["tick"] / np.asarray(mid, float) * 1e4


def cost_bps(panel: F.Panel, cfg: BDSConfig) -> float:
    """Futures-equivalent round trip: one tick at the window's median price + commission."""
    return float(tick_bps(panel.pair, np.median(panel.mid))) + cfg.commission_bps_rt


# ----------------------------------------------------------------------------- scan
def scan_panel(panel: F.Panel, cfg: BDSConfig, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """F.scan_panel with year consistency over counted years only, the futures-equivalent cost in `cost_bps`
    and Dukascopy's quoted cost in `cost_cfd_bps`. With rng: rotated-day placebo."""
    from scipy.stats import norm
    lp = panel.logmid
    if rng is not None:
        r = np.diff(lp, axis=1)
        k = rng.integers(0, F.SLOTS, len(r))
        idx = (np.arange(F.SLOTS)[None, :] - k[:, None]) % F.SLOTS
        r = r[np.arange(len(r))[:, None], idx]
        lp = np.concatenate([lp[:, :1], lp[:, :1] + np.cumsum(r, axis=1)], axis=1)
    iv = F.intervals(cfg.fx())
    R = (lp[:, iv[:, 1]] - lp[:, iv[:, 0]]) * 1e4
    with np.errstate(divide="ignore", invalid="ignore"):
        m, se, t = F.nw_tstats(R, cfg.nw_lags)
    t = np.where(se > 0, t, np.nan)                      # intervals in closed hours: no test
    yrs = panel.years
    same = np.zeros(len(iv), int)
    for y in counted_years(panel, cfg):
        same += (np.sign(R[yrs == y].mean(axis=0)) == np.sign(m)).astype(int)
    spr = np.median(panel.spread_bps(), axis=0)
    return pd.DataFrame({"pair": panel.pair, "clock": panel.clock, "start": iv[:, 0], "end": iv[:, 1],
                         "minutes": iv[:, 1] - iv[:, 0], "n_days": len(R), "mean_bps": m, "se_bps": se, "t": t,
                         "p": 2 * norm.sf(np.abs(t)), "years_same_sign": same,
                         "cost_bps": cost_bps(panel, cfg),
                         "cost_cfd_bps": (spr[iv[:, 0]] + spr[iv[:, 1]]) / 2 + cfg.commission_bps_rt})


def d2_needed(panels: dict, cfg: BDSConfig) -> tuple[int, int]:
    """(counted years, years needed for D2), using the clock with fewer counted years."""
    n = min(len(counted_years(p, cfg)) for p in panels.values())
    return n, years_needed(n, cfg.year_share_d2)


def select(scan: pd.DataFrame, cfg: BDSConfig, n_needed: int, short: str, window: str):
    s, c = F.select_candidates(scan, cfg.fx(min_years_same_sign=n_needed))
    s["D3_cfd"] = s["mean_bps"].abs() > s["cost_cfd_bps"]
    s["D123"] = s["D1"] & s["D2"] & s["D3"]
    if len(c):
        c["D3_cfd"] = c["mean_bps"].abs() > c["cost_cfd_bps"]
        c["id"] = [f"{short}-{window}-{i + 1:02d}" for i in range(len(c))]
        c["window"] = window
    return s, c


def placebo_counts(panels: dict, cfg: BDSConfig, n_needed: int) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for k in range(cfg.n_placebo):
        sc = pd.concat([scan_panel(p, cfg, rng) for p in panels.values()], ignore_index=True)
        s, c = select(sc, cfg, n_needed, "P", "P")
        rows.append({"run": k, "D1": int(s["D1"].sum()), "D1_D2_D3": int(s["D123"].sum()), "candidates": len(c)})
    return pd.DataFrame(rows)


def interval_returns(panel: F.Panel, start: int, end: int, direction: int) -> np.ndarray:
    return direction * (panel.logmid[:, end] - panel.logmid[:, start]) * 1e4


def robustness(panels: dict, cand: pd.DataFrame, cfg: BDSConfig) -> pd.DataFrame:
    """Trimmed mean (cfg.trim each tail) and median of each candidate's daily return, in its direction."""
    from scipy.stats import trim_mean
    rows = []
    for _, r in cand.iterrows():
        x = interval_returns(panels[r["clock"]], int(r["start"]), int(r["end"]), int(r["direction"]))
        rows.append({"id": r["id"], "mean_bps": float(x.mean()), "trimmed_mean_bps": float(trim_mean(x, cfg.trim)),
                     "median_bps": float(np.median(x)), "share_positive": float((x > 0).mean())})
    return pd.DataFrame(rows)


def yearly_means(panels: dict, cand: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for _, r in cand.iterrows():
        p = panels[r["clock"]]
        x = interval_returns(p, int(r["start"]), int(r["end"]), int(r["direction"]))
        out[r["id"]] = pd.Series(x).groupby(p.years).mean()
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- confirmation
def trades(panel: F.Panel, start: int, end: int, direction: int, cfg: BDSConfig) -> pd.DataFrame:
    """One trade per day. gross = mid move in the trade direction minus one futures tick; net = gross minus
    commission; net2 = gross minus 2x commission. cfd_net = at Dukascopy's quoted bid/ask minus commission.
    `gross_pips` holds the mid move in price points (name kept for F.confirm)."""
    mid = panel.mid
    move = interval_returns(panel, start, end, direction)
    gross = move - tick_bps(panel.pair, mid[:, start])
    cfd = F.candidate_trades(panel, start, end, direction, cfg.fx())
    return pd.DataFrame({"date": panel.dates, "year": panel.years, "mid_bps": move, "gross_bps": gross,
                         "net_bps": gross - cfg.commission_bps_rt, "net2_bps": gross - 2 * cfg.commission_bps_rt,
                         "gross_pips": direction * (mid[:, end] - mid[:, start]),
                         "cfd_net_bps": cfd["net_bps"].to_numpy()})


def confirm(tr: dict, panels: dict, cfg: BDSConfig) -> pd.DataFrame:
    """F.confirm gates K1-K3; K4 recomputed over counted years of the confirmation window."""
    if not tr:
        return pd.DataFrame()
    out = F.confirm(tr, cfg.fx())
    yrs = sorted(set.intersection(*[set(counted_years(p, cfg)) for p in panels.values()]))
    need = years_needed(len(yrs), cfg.year_share_k4)
    out["years_counted"] = len(yrs)
    out["years_needed"] = need
    out["years_net_positive"] = [int((t[t["year"].isin(yrs)].groupby("year")["net_bps"].mean() > 0).sum())
                                 for t in tr.values()]
    out["K4"] = out["years_net_positive"] >= need
    out["cfd_net_bps"] = [float(t["cfd_net_bps"].mean()) for t in tr.values()]
    out["verdict"] = np.where(out[["K1", "K2", "K3", "K4"]].all(axis=1), "CONFIRMED", "NOT_CONFIRMED")
    return out


def overlaps(a: pd.Series, b: pd.Series) -> bool:
    return F._overlap(F._utc_segments(a["clock"], int(a["start"]), int(a["end"])),
                      F._utc_segments(b["clock"], int(b["start"]), int(b["end"])))


CAND_COLS = ["id", "window", "pair", "clock", "start_time", "end_time", "minutes", "direction", "mean_bps", "t", "q",
             "years_same_sign", "cost_bps", "cost_cfd_bps", "D3_cfd", "n_days", "start", "end"]


def load_frozen(folder: Path) -> tuple[pd.DataFrame, dict]:
    """A window's frozen candidates, verified against its summary.json."""
    import json
    f, s = Path(folder) / "candidates_frozen.csv", Path(folder) / "summary.json"
    summ = json.loads(s.read_text())
    if not F.frozen_hash_ok(f, summ["candidates_sha256"]):
        raise RuntimeError(f"{f} changed after discovery")
    c = pd.read_csv(f)
    return c, summ


# ----------------------------------------------------------------------------- synthetic data
def _open(symbol: str, ts: pd.DatetimeIndex) -> np.ndarray:
    wd, mod = ts.weekday, ts.hour * 60 + ts.minute
    if symbol == "USTBONDTRUSD":                 # Sunday 22:00 -> Friday 21:00 UTC, daily break 21:00-22:00
        closed = (wd == 5) | ((wd == 6) & (mod < 22 * 60)) | ((wd == 4) & (mod >= 21 * 60)) | \
                 ((wd < 4) & (mod >= 21 * 60) & (mod < 22 * 60))
        return ~closed
    lo, hi = (7 * 60, 21 * 60) if symbol == "BUNDTREUR" else (8 * 60, 17 * 60)
    return (wd < 5) & (mod >= lo) & (mod < hi)


def roll_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Synthetic switch days: the business day on or before the 25th of the month before each expiry month."""
    out = []
    for y in range(start.year, end.year + 1):
        for m in ROLL_MONTHS:
            d = dt.date(y, m - 1, 25)
            while d.weekday() >= 5:
                d -= dt.timedelta(days=1)
            if start <= d <= end:
                out.append(d)
    return out


def write_synthetic_bond(base: Path, symbol: str, start: dt.date, end: dt.date, seed: int, drift_bps: float = 0.0,
                         drift_ny_min: tuple = (510, 570), roll_bps: float = -60.0, price: float = 120.0,
                         spread: float = 0.05, missing: int = 0) -> Path:
    """Synthetic 1-minute bid/ask candles in the downloader layout: random-walk mid during the symbol's
    session, a `roll_bps` jump at the reopening after each synthetic switch day, `drift_bps` per day spread
    over [drift_ny_min) New York minutes of the day, and `missing` random weekday files removed."""
    rng = np.random.default_rng(seed)
    root = Path(base) / symbol / "candles_m1"
    lp, pending = np.log(price), 0.0
    rolls = set(roll_days(start, end))
    d = start
    while d <= end:
        if d.weekday() != 5:
            ts = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=F.SLOTS, freq="1min")
            op = _open(symbol, ts)
            quoted = op & (rng.random(F.SLOTS) < 0.9)
            r = rng.normal(0, 0.4e-4, F.SLOTS) * op
            ny = ts.tz_convert(F.NY)
            nm = ny.hour * 60 + ny.minute
            a, b = drift_ny_min
            r = r + np.where(op & (nm >= a) & (nm < b) & (ny.weekday < 5), drift_bps * 1e-4 / (b - a), 0.0)
            if pending and op.any():
                r[np.flatnonzero(op)[0]] += pending
                pending = 0.0
            if d in rolls:
                if symbol == "USTBONDTRUSD":
                    r[22 * 60] += roll_bps * 1e-4          # reopening after the 21:00-22:00 UTC break
                else:
                    pending = roll_bps * 1e-4             # next session's first minute
            path = lp + np.cumsum(r)
            lp = path[-1]
            mid = np.exp(path)
            bid, ask = np.round(mid - spread / 2, 3), np.round(mid + spread / 2, 3)
            v = quoted.astype(np.float32)
            df = pd.DataFrame({"ts": ts, "bid_o": bid, "bid_h": bid, "bid_l": bid, "bid_c": bid, "bid_v": v,
                               "ask_o": ask, "ask_h": ask, "ask_l": ask, "ask_c": ask, "ask_v": v})
            f = root / f"{d:%Y}" / f"{symbol}_{d:%Y-%m-%d}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(f, index=False)
        d += dt.timedelta(days=1)
    if missing:
        wk = [f for f in root.rglob("*.parquet") if dt.date.fromisoformat(f.stem[-10:]).weekday() < 4]
        for f in rng.choice(wk, missing, replace=False):
            f.unlink()
    return root
