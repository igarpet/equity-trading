"""S05b: measured transaction costs for the S05a strategy.

1. Sample sessions: `per_year` NYSE sessions per year, drawn with a fixed seed from the calendar only
   (no price or return data is used to choose them).
2. On each sampled session, for every bar start (15m grid), the quoted NBBO half-spread of every symbol
   in the first few seconds after the bar opens (Alpaca SIP quotes) - the moment S05 orders fill.
3. Cost model: half_spread(symbol, year, slot) = base(symbol, year) x profile(slot), where base is the
   symbol's median over that year's samples and profile is the cross-sectional median ratio of a slot's
   sample to its symbol-year base (the intraday U-shape). Fallbacks: the symbol's all-year median, then
   that year's cross-sectional median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

S05B_VERSION = "0.1.0"


def sample_sessions(years: list[int], per_year: int, seed: int = 42) -> list:
    from .alpaca import session_schedule
    sc = session_schedule(years)
    rng = np.random.default_rng(seed)
    out = []
    for y in years:
        days = [d for d in sc.index if d.year == y]
        out += sorted(rng.choice(np.array(days, dtype=object), size=min(per_year, len(days)), replace=False).tolist())
    return out


def slot_starts(years: list[int], session_date, minutes: int = 15) -> list:
    """UTC bar starts of one session (exchange calendar, early closes respected)."""
    from .alpaca import session_schedule
    sc = session_schedule(years)
    op, cl = sc.loc[session_date, "market_open"], sc.loc[session_date, "market_close"]
    return list(pd.date_range(op, cl, freq=f"{minutes}min", inclusive="left"))


def cost_model(samples: pd.DataFrame) -> dict:
    """samples: [symbol, half_spread, n_quotes, slot, date] -> {'base': DataFrame(symbol x year),
    'profile': Series(slot), 'year_median': Series(year), 'sym_median': Series(symbol)}."""
    d = samples.dropna(subset=["half_spread"]).copy()
    d = d[d["n_quotes"] > 0]
    d["year"] = pd.to_datetime(d["date"]).dt.year
    base = d.groupby(["symbol", "year"])["half_spread"].median().unstack()
    d = d.join(base.stack().rename("base"), on=["symbol", "year"])
    d["ratio"] = d["half_spread"] / d["base"]
    prof = d.groupby("slot")["ratio"].median()
    return {"base": base, "profile": prof, "year_median": d.groupby("year")["half_spread"].median(),
            "sym_median": d.groupby("symbol")["half_spread"].median()}


def cost_array(index: pd.DatetimeIndex, day_start: np.ndarray, symbols: list[str], model: dict,
               dtype=np.float32) -> np.ndarray:
    """(T, N) one-way cost per bar and panel column from the cost model."""
    T, N = len(index), len(symbols)
    years = index.year.to_numpy()
    slot = np.zeros(T, dtype=int)
    for a, b in zip(day_start[:-1], day_start[1:]):
        slot[a:b] = np.arange(b - a)
    base, prof = model["base"], model["profile"]
    uniq_years = np.unique(years)
    lvl = np.empty((len(uniq_years), N))
    for k, y in enumerate(uniq_years):
        ym = model["year_median"].get(y, model["year_median"].median())
        for j, s in enumerate(symbols):
            v = base.at[s, y] if (s in base.index and y in base.columns) else np.nan
            if not np.isfinite(v):
                v = model["sym_median"].get(s, np.nan)
            lvl[k, j] = v if np.isfinite(v) else ym
    pf = prof.reindex(range(slot.max() + 1)).ffill().bfill().to_numpy()
    yi = np.searchsorted(uniq_years, years)
    return (lvl[yi] * pf[slot][:, None]).astype(dtype)
