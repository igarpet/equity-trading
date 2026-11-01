"""Spread construction and intraday deseasonalisation."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPREADS_VERSION = "0.1.0"


def ols_spread(log_y: pd.Series, log_x: pd.Series) -> tuple[pd.Series, float, float]:
    """X_t = log P_y - alpha - beta log P_x (OLS with intercept, as in the thesis)."""
    df = pd.concat([log_y, log_x], axis=1).dropna()
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    return pd.Series(y - alpha - beta * x, index=df.index), float(alpha), float(beta)


def bar_slot(index: pd.DatetimeIndex) -> np.ndarray:
    """Intraday slot of each bar: 0 for the first bar of each session (which
    carries the overnight gap), then 1, 2, ... within the day."""
    dates = index.tz_convert("America/New_York").date
    slot = np.zeros(len(index), dtype=int)
    for i in range(1, len(index)):
        slot[i] = 0 if dates[i] != dates[i - 1] else slot[i - 1] + 1
    return slot


def deseasonalise(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Divide spread increments by the s.d. of increments in the same intraday slot,
    then re-cumulate. Slot 0 (overnight) is its own bucket.

    Returns (deseasonalised level, slot s.d. profile). The profile is estimated on
    the same window as it is applied (descriptive studies only; a trading study
    must estimate it on past data).
    """
    dx = x.diff()
    slot = pd.Series(bar_slot(x.index), index=x.index)
    prof = dx.groupby(slot).std()
    prof = prof / prof.median()
    scale = slot.map(prof).to_numpy()
    dz = (dx / scale).fillna(0.0)
    return pd.Series(np.cumsum(dz.to_numpy()), index=x.index), prof
