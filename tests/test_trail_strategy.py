import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import trail_strategy as T  # noqa: E402

CFG = T.TrailConfig()


def _minutes(path, start="2024-03-05 14:00", iid=None):
    """Minute bars from a list of (open, high, low, close)."""
    ts = pd.date_range(start, periods=len(path), freq="1min", tz="UTC")
    a = np.asarray(path, float)
    m1 = pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "volume": 1,
                       "instrument_id": 1 if iid is None else iid}, index=ts)
    return T.minutes_from_1m(m1, CFG)


def flat(p, n):
    return [(p, p, p, p)] * n


def test_initial_stop_with_slippage_and_gap():
    M = _minutes(flat(100, 2) + [(100, 100, 97, 98)] + flat(98, 3))
    tr = T.run_trade(M, 0, +1, 100.0, 8, CFG)                       # stop 8 ticks = 2 points -> 98
    assert tr["reason"] == "stop" and tr["exit_px"] == pytest.approx(98 - 0.125)
    M2 = _minutes(flat(100, 2) + [(96, 96, 95, 95.5)])
    tr2 = T.run_trade(M2, 0, +1, 100.0, 8, CFG)                     # gapped through: fill at the open
    assert tr2["exit_px"] == pytest.approx(96 - 0.125)


def test_trailing_activation_lock_and_follow():
    # +50 ticks = 12.5 points; lock at +30 ticks = 7.5 points; then 20 ticks (5 points) behind the best price
    up = [(100 + i, 100 + i + 0.5, 100 + i - 0.25, 100 + i + 0.5) for i in range(14)]   # best reaches 113.5
    down = [(113, 113, 107, 107)]
    M = _minutes(up + down)
    tr = T.run_trade(M, 0, +1, 100.0, 40, CFG)
    assert tr["reason"] == "trail"
    assert tr["exit_px"] == pytest.approx(113.5 - 5.0 - 0.125)     # stop at best - 20 ticks
    assert tr["gross_ticks"] == pytest.approx((108.5 - 0.125 - 100) / 0.25)


def test_trail_never_below_lock_and_flat_exit():
    path = [(100, 112.5, 100, 112.5)] + flat(112.5, 3) + [(112.5, 112.5, 107.5, 107.6)]
    M = _minutes(path)
    tr = T.run_trade(M, 0, +1, 100.0, 40, CFG)
    assert tr["reason"] == "trail" and tr["exit_px"] == pytest.approx(107.5 - 0.125)   # +30 ticks lock
    # flat at 15:55 Chicago: 2024-03-05 is CST (UTC-6) -> 21:55 UTC
    M2 = _minutes(flat(100, 10), start="2024-03-05 21:50")
    tr2 = T.run_trade(M2, 0, +1, 100.0, 40, CFG)
    assert tr2["reason"] == "flat" and M2.ts[tr2["exit_i"]] == pd.Timestamp("2024-03-05 21:54", tz="UTC")


def test_reverse_reentry_limit():
    # long from 100 stopped at 98; sell limit at 98.5 (2 ticks above); price rallies through 98.75 -> short at 98.5
    path = flat(100, 1) + [(100, 100, 97.75, 98)] + [(98, 98.75, 98, 98.5)] + flat(98.5, 5)
    M = _minutes(path)
    lim = T.wait_limit(M, 2, -1, 98.5, CFG)
    assert lim == 2
    assert T.wait_limit(M, 2, -1, 99.0, CFG) is None


def test_chain_end_to_end():
    ts5 = pd.date_range("2024-03-05 14:00", periods=1, freq="5min", tz="UTC")
    bt = pd.DataFrame({"trade_date": [dt.date(2024, 3, 5)], "z": [4.0], "eligible": [True], "close": [100.0]}, index=ts5)
    sig2 = pd.Series([(2 / 100) ** 2] * 3, index=pd.date_range("2024-03-05 14:00", periods=3, freq="5min", tz="UTC"))
    # entry 100.125 (slippage), stop 2 ticks -> 99.625; reverse sell limit at 100.125, filled when 100.375 trades
    path = flat(100, 5) + [(100, 100, 99.5, 99.5)] + [(99.5, 100.5, 99.5, 100.0)] + \
        [(100 - i, 100 - i, 100 - i - 0.5, 100 - i - 0.5) for i in range(20)]
    M = _minutes(path)
    sig = T.signals(bt, sig2, 3.0)
    legs = T.run_chains(M, sig, sig2, 0.25, CFG)                      # stop = 0.25 x 2 points = 2 ticks
    assert legs["leg"].max() == 1 and legs.iloc[1]["side"] == -1
    assert legs.iloc[0]["reason"] == "stop" and legs.iloc[1]["reason"] in ("stop", "trail", "flat")
    ch = T.chain_table(legs)
    assert len(ch) == 1 and ch["legs"].iloc[0] == 2
