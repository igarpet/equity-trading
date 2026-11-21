import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import reversal as V  # noqa: E402


def _bt(close, z, day=None, ret=None, session="ny"):
    n = len(close)
    idx = pd.date_range("2024-03-05 14:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"trade_date": day if day is not None else [dt.date(2024, 3, 5)] * n,
                         "slot": np.arange(n) + 10, "session": session, "z": np.asarray(z, float),
                         "ret": np.ones(n) * 1e-4 if ret is None else ret, "eligible": True,
                         "close": np.asarray(close, float)}, index=idx)


def test_trade_ticks_direction_delay_hold():
    close = [100, 110, 108, 105, 104, 104, 104, 104]
    z = [0, 4, 0, 0, 0, 0, 0, 0]
    bt = _bt(close, z)
    t = V.trades(bt, 3, 0, 1, "all")                       # short at 110, exit 108 -> +8 ticks
    assert len(t) == 1 and t["gross_ticks"].iloc[0] == 8
    t = V.trades(bt, 3, 1, 2, "all")                       # enter at 108 (d=1), exit 104 (h=2) -> +16 ticks
    assert t["gross_ticks"].iloc[0] == 16
    c = V.trades(bt, 3, 1, 2, "all", direction=+1)
    assert c["gross_ticks"].iloc[0] == -16


def test_no_overlap_and_day_boundary():
    close = np.arange(10, dtype=float) + 100
    z = [4, 4, 4, 0, 4, 0, 0, 0, 0, 4]
    bt = _bt(close, z)
    t = V.trades(bt, 3, 0, 2, "all")
    assert list(t["ts"]) == [bt.index[0], bt.index[2], bt.index[4]]   # bar 9 has no room for h=2
    day = [dt.date(2024, 3, 5)] * 5 + [dt.date(2024, 3, 6)] * 5
    bt2 = _bt(close, [0, 0, 0, 4, 0, 0, 0, 0, 0, 0], day=day)
    assert len(V.trades(bt2, 3, 0, 2, "all")) == 0                   # would cross into the next CME day
    ret = np.ones(10) * 1e-4
    ret[5] = np.nan                                                   # gap
    bt3 = _bt(close, [0, 0, 0, 4, 0, 0, 0, 0, 0, 0], ret=ret)
    assert len(V.trades(bt3, 3, 0, 2, "all")) == 0


def test_session_filter_and_costs():
    rc = V.RevConfig(commission_per_side=0.50)
    assert rc.market_cost_ticks == 3.0
    bt = _bt([100, 110, 108], [0, 4, 0], session="asia")
    assert len(V.trades(bt, 3, 0, 1, "ny")) == 0 and len(V.trades(bt, 3, 0, 1, "asia")) == 1
    assert len(V.configs()) == 200


def test_scan_finds_planted_reversal():
    rng = np.random.default_rng(0)
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    rows = []
    for d in days:
        n = 100
        z = rng.standard_t(4, n)
        steps = rng.normal(0, 4, n)
        big = np.flatnonzero(np.abs(z) >= 3)
        for b in big:
            steps[b] = np.sign(z[b]) * 40
            if b + 2 < n:
                steps[b + 2] -= np.sign(z[b]) * 10           # 10 points = 40 ticks against, one bar later
        close = 10000 + np.cumsum(steps)
        rows.append(pd.DataFrame({"trade_date": d, "slot": np.arange(n) + 5, "session": "ny", "z": z,
                                  "ret": 1e-4, "eligible": True, "close": close}))
    bt = pd.concat(rows, ignore_index=True)
    bt.index = pd.date_range("2024-01-01", periods=len(bt), freq="5min", tz="UTC")
    rc = V.RevConfig(ks=(2.0, 3.0), delays=(0, 1), holds=(1,), sessions=("all",), min_trades_discovery=50)
    per = {"a": (days[0], days[199]), "b": (days[200], days[-1])}
    df, _ = V.scan(bt, rc, per)
    best = df.set_index("config").loc["k3_d1_h1_all"]
    assert best["candidate"] and best["gross"] > 30
    assert not df.set_index("config").loc["k3_d0_h1_all", "candidate"]


def test_day_cluster_t():
    x = np.r_[np.ones(50), -np.ones(50)] + 0.5
    m, t = V.day_cluster_t(x, np.repeat(np.arange(20), 5))
    assert m == pytest.approx(0.5) and np.isfinite(t)


def test_freeze(tmp_path):
    df = pd.DataFrame({"config": ["a"], "k": [3.0]})
    sha = V.freeze(df, tmp_path / "c.csv")
    assert V.frozen_ok(tmp_path / "c.csv", sha)


def test_continuation_test_gates():
    rng = np.random.default_rng(1)
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(300)]
    rows = []
    for d in days:
        n = 100
        z = rng.standard_t(4, n)
        steps = rng.normal(0, 4, n)
        for b in np.flatnonzero(np.abs(z) >= 3):
            steps[b] = np.sign(z[b]) * 40
            for j in range(b + 1, min(b + 4, n)):
                steps[j] += np.sign(z[b]) * 5            # 20 ticks per bar with the move
        rows.append(pd.DataFrame({"trade_date": d, "slot": np.arange(n) + 5, "session": "ny", "z": z,
                                  "ret": 1e-4, "eligible": True, "close": 10000 + np.cumsum(steps)}))
    bt = pd.concat(rows, ignore_index=True)
    bt.index = pd.date_range("2024-01-01", periods=len(bt), freq="5min", tz="UTC")
    rc = V.RevConfig(min_trades_confirm=50)
    per = {"a": (days[0], days[99]), "b": (days[100], days[199]), "c": (days[200], days[-1])}
    df, _ = V.continuation_test(bt, rc, per)
    assert len(df) == 8
    assert df.set_index("config").loc["cont_k3_d0_h3_all", "confirmed"]
