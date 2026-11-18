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

BDS_VERSION = "0.2.0"   # 0.2.0: switch detection by Dukascopy's adjustment-break signature, 1-minute spike screen (BDS 002)
SYMBOLS = {
    "USTBONDTRUSD": {"short": "UST", "name": "US T-Bond", "tick": 1 / 32, "future": "ZB (CBOT T-Bond)",
                     "tz": "America/New_York", "close": 17 * 60},
    "BUNDTREUR": {"short": "BUND", "name": "Euro Bund", "tick": 0.01, "future": "FGBL (Eurex Bund)",
                  "tz": "Europe/Berlin", "close": 22 * 60},
    "UKGILTTRGBP": {"short": "GILT", "name": "UK Long Gilt", "tick": 0.01, "future": "Long Gilt (ICE)",
                    "tz": "Europe/London", "close": 18 * 60},
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
    switch_rule: str = "signature"     # "signature" (BDS 002) or "largest" (BDS 001)
    early_stop: tuple = (15, 25)       # signature A: last quote 15-25 min before the regular close (Dukascopy stops 20 min early)
    midday_break: tuple = (45, 90)     # signature B: an in-session break of 45-90 min (adjustment break used in 2018-2019)
    spike_abs_bps: float = 100.0       # a 1-minute mid move at least this large flags the day
    spike_revert_bps: float = 50.0     # ... or a move at least this large
    spike_revert_share: float = 0.8    # ... reversed by at least this share
    spike_revert_minutes: int = 5      # ... within this many quoted minutes
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
    spikes: pd.DataFrame       # flagged 1-minute moves (screen_spikes)
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


def _signature(br: pd.DataFrame, symbol: str, cfg: BDSConfig) -> pd.Series:
    """Which breaks look like a Dukascopy CFD adjustment break:
    A ``early_stop``: the last quote before the break is 15-25 minutes before the regular close (local exchange
      time), because on the adjustment day CFD trading stops 20 minutes early;
    B ``midday``: a 45-90 minute break that starts and ends during the session (the adjustment break seen in
      2018-2019). Returns "early_stop", "midday" or "" per break."""
    info = SYMBOLS[symbol]
    loc = br["before"].dt.tz_convert(info["tz"])
    mod = loc.dt.hour * 60 + loc.dt.minute
    lead = info["close"] - 1 - mod                      # minutes between the last quote and the close's last minute
    a = (lead >= cfg.early_stop[0] - 1) & (lead <= cfg.early_stop[1])
    after = br["after"].dt.tz_convert(info["tz"])
    mod_a = after.dt.hour * 60 + after.dt.minute
    b = (br["minutes"] >= cfg.midday_break[0]) & (br["minutes"] <= cfg.midday_break[1]) & \
        (mod >= 6 * 60) & (mod_a <= info["close"] - 60) & (loc.dt.date == after.dt.date)
    return pd.Series(np.where(a, "early_stop", np.where(b, "midday", "")), index=br.index)


def detect_rolls(minutes: pd.DataFrame, start: dt.date, end: dt.date, cfg: BDSConfig, symbol: str | None = None) -> pd.DataFrame:
    """One contract switch per quarter, looked for between the 10th of the month before a quarterly expiry month
    and the end of that month.

    ``switch_rule="signature"`` (BDS 002): the break with the largest |gap| among breaks with an adjustment-break
    signature (see ``_signature``); if the window has none, the largest |gap| of all its breaks (``method``
    "fallback"). ``switch_rule="largest"`` (BDS 001): always the largest |gap|.
    Other breaks in the window larger than both half the chosen gap and `roll_extra_ratio` x the median |gap|
    are also dropped (``extra``)."""
    br = break_gaps(minutes, cfg.break_minutes)
    if br.empty:
        return br.assign(quarter=[], ratio=[], extra=[], method=[])
    use_sig = cfg.switch_rule == "signature" and symbol is not None
    br["signature"] = _signature(br, symbol, cfg) if use_sig else ""
    med = float(br["gap_bps"].abs().median()) or 1e-9
    rows = []
    for y in range(start.year, end.year + 1):
        for m in ROLL_MONTHS:
            a = pd.Timestamp(dt.date(y, m - 1, 10), tz="UTC")
            b = pd.Timestamp(dt.date(y + (m == 12), m % 12 + 1, 1), tz="UTC")
            w = br[(br["after"] >= a) & (br["after"] < b) & (br["after"].dt.date >= start) & (br["after"].dt.date <= end)]
            if w.empty:
                continue
            sig = w[w["signature"] != ""]
            pool = sig if use_sig and len(sig) else w
            method = ("signature" if len(sig) else "fallback") if use_sig else "largest"
            top = pool["gap_bps"].abs().idxmax()
            big = float(abs(w.loc[top, "gap_bps"]))
            for ix, r in w.iterrows():
                g = abs(r["gap_bps"])
                if ix == top or (g >= 0.5 * big and g >= cfg.roll_extra_ratio * med):
                    rows.append({"quarter": f"{y}-{m:02d}", "before": r["before"], "after": r["after"],
                                 "break_minutes": int(r["minutes"]), "gap_bps": float(r["gap_bps"]),
                                 "ratio": g / med, "extra": ix != top, "method": method if ix == top else "extra",
                                 "signature": r["signature"]})
    return pd.DataFrame(rows)


def screen_spikes(minutes: pd.DataFrame, cfg: BDSConfig) -> pd.DataFrame:
    """1-minute mid moves between consecutive quoted minutes (within a session, not across a break) that are
    either at least `spike_abs_bps`, or at least `spike_revert_bps` and reversed by `spike_revert_share` within
    `spike_revert_minutes` quoted minutes. Each row is a flagged minute; its days are dropped."""
    q = minutes[minutes["tick"]]
    ts = q.index
    lm = np.log(0.5 * (q["bid"].to_numpy() + q["ask"].to_numpy()))
    if len(lm) < 3:
        return pd.DataFrame(columns=["ts", "move_bps", "reverted_bps", "kind"])
    gap = np.diff(ts.to_numpy()).astype("timedelta64[m]").astype(np.int64)
    r = np.diff(lm) * 1e4
    r[gap >= cfg.break_minutes] = 0.0                    # moves across breaks are not 1-minute moves
    k = cfg.spike_revert_minutes
    cum = np.concatenate([[0.0], np.cumsum(r)])
    rows = []
    for i in np.flatnonzero(np.abs(r) >= min(cfg.spike_abs_bps, cfg.spike_revert_bps)):
        j = min(len(r), i + 1 + k)
        after = cum[j] - cum[i + 1]                         # move over the next k quoted minutes
        reverted = -after * np.sign(r[i])
        kind = "large" if abs(r[i]) >= cfg.spike_abs_bps else ""
        if reverted >= cfg.spike_revert_share * abs(r[i]):
            kind = "spike" if not kind else "large+spike"
        if kind:
            rows.append({"ts": ts[i + 1], "move_bps": float(r[i]), "reverted_bps": float(reverted), "kind": kind})
    return pd.DataFrame(rows, columns=["ts", "move_bps", "reverted_bps", "kind"])


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
    rolls = detect_rolls(minutes, first, end, cfg, symbol)
    spikes = screen_spikes(minutes, cfg)
    spike_lab = {c: set() for c in CLOCKS}
    for t in spikes["ts"]:
        for c, lab in _labels(t).items():
            spike_lab[c].add(lab)
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
        m_spike = np.isin(p.dates, list(spike_lab[c])) if spike_lab[c] else np.zeros(len(p.dates), bool)
        keep = ~(m_missing | m_cov | m_roll | m_spike)
        panels[c] = _subset(p, keep)
        drops.append({"clock": c, "days built": len(p.dates), "missing file": int(m_missing.sum()),
                      "low coverage": int((m_cov & ~m_missing).sum()), "contract switch": int((m_roll & ~m_missing & ~m_cov).sum()),
                      "price spike": int((m_spike & ~m_roll & ~m_missing & ~m_cov).sum()), "kept": int(keep.sum())})
    return BondData(symbol, window, start, end, first, minutes, fs, rolls, spikes, panels, pd.DataFrame(drops))


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
def _local(symbol: str, ts: pd.DatetimeIndex):
    loc = ts.tz_convert(SYMBOLS[symbol]["tz"])
    return loc, np.asarray(loc.hour * 60 + loc.minute)


def _open(symbol: str, ts: pd.DatetimeIndex) -> np.ndarray:
    """Synthetic sessions in local exchange time: UST Sunday 18:00 -> Friday 17:00 New York with a daily break
    17:00-18:00; Bund 08:00-22:00 Frankfurt; Gilt 08:00-18:00 London (Monday-Friday)."""
    loc, mod = _local(symbol, ts)
    wd = np.asarray(loc.weekday)
    if symbol == "USTBONDTRUSD":
        closed = (wd == 5) | ((wd == 6) & (mod < 18 * 60)) | ((wd == 4) & (mod >= 17 * 60)) | \
                 ((wd < 4) & (mod >= 17 * 60) & (mod < 18 * 60))
        return ~closed
    return (wd < 5) & (mod >= 8 * 60) & (mod < SYMBOLS[symbol]["close"])


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
                         spread: float = 0.05, missing: int = 0, weekend_gap_bps: float = 0.0,
                         spikes: tuple = ()) -> Path:
    """Synthetic 1-minute bid/ask candles in the downloader layout: random-walk mid during the symbol's session;
    on each synthetic switch day quotes stop 20 minutes before the close (as Dukascopy does) and the mid jumps
    `roll_bps` at the next open; `drift_bps` per day spread over [drift_ny_min) New York minutes;
    `weekend_gap_bps` at the first open after the weekend following the 15th of each roll window's expiry month
    (an ordinary gap larger than the switch); `spikes`: UTC timestamps where the mid jumps 300 bps for one
    minute and comes back; `missing` random weekday files removed."""
    rng = np.random.default_rng(seed)
    root = Path(base) / symbol / "candles_m1"
    lp, pending = np.log(price), 0.0
    rolls = set(roll_days(start, end))
    close = SYMBOLS[symbol]["close"]
    gap_days = set()
    for y in range(start.year, end.year + 1):
        for m in ROLL_MONTHS:
            d0 = dt.date(y, m, 15)
            while d0.weekday() != 0:
                d0 += dt.timedelta(days=1)
            gap_days.add(d0)                              # Monday after the weekend
    spike_ts = {pd.Timestamp(t, tz="UTC") for t in spikes}
    d = start
    while d <= end:
        if d.weekday() != 5:
            ts = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=F.SLOTS, freq="1min")
            op = _open(symbol, ts)
            loc, mod = _local(symbol, ts)
            early = np.zeros(F.SLOTS, bool)
            if d in rolls:
                early = op & (mod >= close - 20) & (mod < close) & (np.asarray(loc.date) == d)
                op = op & ~early
            quoted = op & (rng.random(F.SLOTS) < 0.9)
            r = rng.normal(0, 0.4e-4, F.SLOTS) * op
            ny = ts.tz_convert(F.NY)
            nm = np.asarray(ny.hour * 60 + ny.minute)
            a, b = drift_ny_min
            r = r + np.where(op & (nm >= a) & (nm < b) & (np.asarray(ny.weekday) < 5), drift_bps * 1e-4 / (b - a), 0.0)
            first_open = np.flatnonzero(op)
            if pending and len(first_open):
                r[first_open[0]] += pending
                pending = 0.0
            if weekend_gap_bps and len(first_open) and first_open[0] > 0:     # a session open inside this file
                ld = np.asarray(loc.date)
                if ld[first_open[0]] in gap_days or (symbol == "USTBONDTRUSD" and (ld[first_open[0]] + dt.timedelta(days=1)) in gap_days
                                                    and np.asarray(loc.weekday)[first_open[0]] == 6):
                    r[first_open[0]] += weekend_gap_bps * 1e-4
            if early.any():
                nxt = np.flatnonzero(op & (np.arange(F.SLOTS) > np.flatnonzero(early).max()))
                if len(nxt):
                    r[nxt[0]] += roll_bps * 1e-4
                else:
                    pending = roll_bps * 1e-4
            path = lp + np.cumsum(r)
            lp = path[-1]
            mid = np.exp(path)
            for t in spike_ts:
                if t.date() == d:
                    i = int((t - ts[0]) / pd.Timedelta(minutes=1))
                    mid[i] *= 1.03
                    quoted[i] = quoted[i + 1] = True
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
