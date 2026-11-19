import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import futures as F  # noqa: E402
from equity_trading import micro_bars as M  # noqa: E402

D = dt.date


@pytest.fixture
def market_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    return tmp_path


def _m1(n=30, start="2025-03-14 14:00", switch_at=None):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = 20000 + np.arange(n, dtype=float)
    iid = np.full(n, 1, dtype=np.int64)
    if switch_at is not None:
        iid[switch_at:] = 2
        close[switch_at:] += 100.0                      # contract basis
    return pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close,
                         "volume": np.full(n, 10, dtype=np.int64), "instrument_id": iid}, index=pd.Index(ts, name="ts"))


def test_guard_and_micros():
    with pytest.raises(M.HoldoutError):
        M.guard(D(2026, 1, 2))
    M.guard(D(2026, 1, 1))                              # exclusive end: December 2025 is allowed
    with pytest.raises(ValueError):
        M.query("NQ", D(2025, 1, 1), D(2025, 2, 1))
    assert M.query("MNQ", D(2025, 1, 1), D(2025, 2, 1))["symbols"] == ["MNQ.v.0"]


def test_normalise():
    raw = _m1(3).reset_index().rename(columns={"ts": "ts_event"})
    raw.loc[1, "close"] = 0.0                           # bad print dropped
    out = M.normalise(raw)
    assert list(out.columns) == M.COLUMNS and len(out) == 2


def test_to_bars_no_roll():
    b = M.to_bars(_m1(30), 5)
    assert len(b) == 6 and (b["minutes_traded"] == 5).all()
    assert b["close"].iloc[0] == 20004 and b["open"].iloc[0] == 20000 - 0.5
    assert np.isnan(b["log_ret"].iloc[0])
    assert np.isclose(b["log_ret"].iloc[1], np.log(20009 / 20004))
    assert not b["roll"].any()


def test_to_bars_roll_never_crosses_contracts():
    b = M.to_bars(_m1(30, switch_at=12), 5)            # switch inside the 10:00-15:00 bucket
    assert b["roll"].sum() == 1
    r = b.index[b["roll"]][0]
    assert np.isnan(b.loc[r, "log_ret"])
    assert (b["log_ret"].dropna().abs() < 0.001).all()  # the 100-point basis jump never shows as a return
    # straddling bucket keeps the contract with more volume in it (3 minutes new vs 2 old)
    assert b.loc[r, "instrument_id"] == 2 and b.loc[r, "minutes_traded"] == 3


class _Client:
    def __init__(self, cost=0.05):
        self.calls = []
        self.metadata = SimpleNamespace(get_cost=lambda **q: cost, get_billable_size=lambda **q: 1_000_000)
        self.timeseries = SimpleNamespace(get_range=self._get)

    def _get(self, path, **q):
        self.calls.append(q["start"])
        Path(path).write_bytes(b"x")


def test_download_ledger_budget_and_resume(market_dir, monkeypatch):
    monkeypatch.setattr(M, "normalise_raw_file", lambda path, root, a: 100)
    cl = _Client(cost=0.05)
    log = M.download(cl, ["MNQ"], 2019, 2019, budget=1.0, log=lambda *a: None)
    assert cl.calls[0] == "2019-05-01" and len(cl.calls) == 8          # May-Dec 2019
    assert np.isclose(F.spent_usd(), 0.40)
    log2 = M.download(cl, ["MNQ"], 2019, 2019, budget=1.0, log=lambda *a: None)
    assert (log2["status"] == "normalised from raw").all() and len(cl.calls) == 8
    stop = M.download(_Client(cost=5.0), ["MNQ"], 2020, 2020, budget=1.0, log=lambda *a: None)
    assert stop["status"].iloc[-1].startswith("stopped")


def test_quote(market_dir):
    q = M.quote(_Client(cost=0.1), ["MNQ"], 2019, 2020, log=lambda *a: None)
    assert list(q["months"]) == [8, 12] and np.isclose(q["usd"].sum(), 0.2)
