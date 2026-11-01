import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import alpaca  # noqa: E402
from equity_trading.coint import (adf_tstat, build_pairs, eg_test, run_rolling,  # noqa: E402
                                  window_bootstrap)
from equity_trading.universe import SECTOR_ETF_01, UNIVERSE_01  # noqa: E402


class _Resp:
    def __init__(self, js):
        self.status_code, self._js, self.text = 200, js, ""

    def json(self):
        return self._js


class _FakeSession:
    """Two pages: AAPL on page 1, AAPL + MSFT on page 2 (Alpaca sorts by symbol)."""
    def __init__(self):
        self.calls = []

    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        if "page_token" not in params:
            return _Resp({"bars": {"AAPL": [{"t": "2024-01-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1.0,
                                             "v": 10, "n": 1, "vw": 1}]}, "next_page_token": "abc"})
        return _Resp({"bars": {"AAPL": [{"t": "2024-01-02T15:00:00Z", "o": 1, "h": 1, "l": 1, "c": 2.0,
                                         "v": 10, "n": 1, "vw": 1}],
                               "MSFT": [{"t": "2024-01-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 3.0,
                                         "v": 10, "n": 1, "vw": 1}]}, "next_page_token": None})


def test_fetch_bars_paginates(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setattr(alpaca, "_limiter", alpaca._RateLimiter(10 ** 6))
    sess = _FakeSession()
    out = alpaca.fetch_bars(["AAPL", "MSFT"], "30Min", "2024-01-01", "2025-01-01", session=sess)
    assert len(sess.calls) == 2 and sess.calls[1]["page_token"] == "abc"
    assert sess.calls[0]["feed"] == "sip" and sess.calls[0]["adjustment"] == "all"
    assert out["AAPL"]["c"].tolist() == [1.0, 2.0] and len(out["MSFT"]) == 1


def test_regular_session_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    d = tmp_path / "alpaca" / "bars" / "30Min" / "AAA"
    d.mkdir(parents=True)
    t = pd.to_datetime(["2024-01-02T13:30Z", "2024-01-02T14:30Z", "2024-01-02T15:00Z",
                        "2024-01-02T16:00Z", "2024-01-02T21:00Z"], utc=True)  # 08:30, 09:30, 10:00, 11:00, 16:00 NY
    pd.DataFrame({"t": t, "c": [1.0, 2.0, 3.0, 5.0, 9.0]}).to_parquet(d / "2024.parquet", index=False)
    e = tmp_path / "alpaca" / "bars" / "30Min" / "BBB"
    e.mkdir(parents=True)
    pd.DataFrame({"t": t[[1, 2, 3]], "c": [1.0, 1.0, 1.0]}).to_parquet(e / "2024.parquet", index=False)
    p = alpaca.regular_session_panel(["AAA", "BBB"], "30Min", [2024])
    assert p["AAA"].tolist() == [2.0, 3.0, 5.0]  # pre-market and 16:00 bars dropped
    assert len(p) == 3


def test_eg_detects_cointegration_and_not_random_walks():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(1000))
    u = np.zeros(1000)
    for t in range(1, 1000):
        u[t] = 0.9 * u[t - 1] + rng.standard_normal()
    assert eg_test(1.5 * x + u, x)["p"] < 0.01
    ps = [eg_test(np.cumsum(rng.standard_normal(800)), np.cumsum(rng.standard_normal(800)))["p"] for _ in range(300)]
    assert 0.02 < np.mean(np.array(ps) <= 0.05) < 0.10  # size close to nominal 5%


def test_adf_matches_statsmodels():
    from statsmodels.tsa.stattools import adfuller
    rng = np.random.default_rng(1)
    u = np.cumsum(rng.standard_normal(500)) * 0.1 + rng.standard_normal(500)
    assert abs(adf_tstat(u, 1) - adfuller(u, maxlag=1, autolag=None, regression="c")[0]) < 1e-8


def test_pairs_and_rolling_on_synthetic():
    pairs = build_pairs(UNIVERSE_01, SECTOR_ETF_01)
    assert pairs.group.value_counts().to_dict() == {"cross": 600, "sector": 180, "etf_cross": 120, "etf": 40}
    panel = alpaca.synthetic_panel(UNIVERSE_01, SECTOR_ETF_01, [2024], seed=3)
    sub = pairs[pairs.y.isin(["AAPL", "JPM"]) & pairs.x.isin(["MSFT", "BAC", "XLK", "XLF"])]
    res = run_rolling(np.log(panel), sub, form_days=60, step_days=40)
    rate = res.groupby("group")["p"].apply(lambda s: (s <= 0.05).mean())
    assert rate["etf"] > rate["etf_cross"]
    est, lo, hi = window_bootstrap(res, lambda d: (d.p <= 0.05).mean(), n_boot=50)
    assert lo <= est <= hi


def test_g2_null_vs_null_has_no_lift():
    """False-positive check: cross vs etf_cross passers (both spurious) should show ~0 lift."""
    from equity_trading.s02 import g2
    panel = alpaca.synthetic_panel(UNIVERSE_01, SECTOR_ETF_01, [2023, 2024], seed=5)
    pairs = build_pairs(UNIVERSE_01, SECTOR_ETF_01)
    res = run_rolling(np.log(panel), pairs, form_days=60, step_days=20)
    r = g2(res, "cross", "etf_cross", n_boot=300)
    assert not r["pass"] and abs(r["diff"]) < 0.08
    assert g2(res, "sector", "cross", n_boot=300)["pass"]
