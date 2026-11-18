import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import bond_fx as X  # noqa: E402
from equity_trading import fx_seasonality as F  # noqa: E402

D = dt.date


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    old = {k: os.environ.get(k) for k in ("EQUITY_TRADING_DIR", "DUKASCOPY_DIR")}
    root = tmp_path_factory.mktemp("bfx")
    os.environ["EQUITY_TRADING_DIR"] = str(root)
    os.environ.pop("DUKASCOPY_DIR", None)
    X.write_synthetic_market(root / "Dukascopy", D(2017, 12, 28), D(2018, 12, 31), seed=3, spike="2018-05-15 10:00")
    X.WINDOWS["T"] = (D(2018, 1, 1), D(2018, 12, 31))
    cfg = X.BFXConfig(n_placebo=1, min_trades=40, min_year_days=100)   # one year of data: fewer trades
    legs = {}
    frames = {p: X.frame(p, "T", cfg, legs) for p in X.PAIRINGS}
    yield cfg, legs, frames
    X.WINDOWS.pop("T", None)
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_holdout_refused():
    with pytest.raises(F.HoldoutError):
        X.guard(D(2025, 6, 1), D(2026, 1, 5))


def test_switches_removed_and_spike_day_excluded(market):
    cfg, legs, frames = market
    ust = legs["USTBONDTRUSD"]
    assert (ust.rolls["method"] == "signature").sum() == 4
    fr = frames["USDJPY"]
    ry, rX = fr.returns()
    assert np.nanmax(np.abs(rX)) * 1e4 < 30          # the -60 bp CFD jumps are gone from the continuous series
    eu = frames["EURUSD"]
    assert np.datetime64("2018-05-15") not in set(eu.day)
    assert np.datetime64("2018-05-14") in set(eu.day)


def test_rolling_beta_has_no_lookahead(market):
    cfg, legs, frames = market
    fr = frames["USDJPY"]
    ry, rX = fr.returns()
    b0 = X.rolling_beta(ry, rX, 500)
    ry2 = ry.copy(); ry2[20_000] += 0.05
    b1 = X.rolling_beta(ry2, rX, 500)
    assert np.array_equal(np.nan_to_num(b0[:20_001]), np.nan_to_num(b1[:20_001]))   # beta at row i ignores row i
    assert not np.allclose(np.nan_to_num(b0[20_001:20_400]), np.nan_to_num(b1[20_001:20_400]))


def test_fx_trade_costs(market):
    cfg, legs, frames = market
    fr = frames["EURUSD"]
    g, n, n2, mid = X._fx_trade(fr, 100, 110, 1, cfg)
    assert g == pytest.approx(np.log(fr.bid[110] / fr.ask[100]) * 1e4)
    assert n == pytest.approx(g - 0.70) and n2 == pytest.approx(g - 1.40)
    g, *_ = X._fx_trade(fr, 100, 110, -1, cfg)
    assert g == pytest.approx(np.log(fr.bid[100] / fr.ask[110]) * 1e4)


@pytest.mark.parametrize("method,expected", [("leadlag", {"USDJPY"}), ("residual", {"GBPUSD"}), ("coint", {"EURUSD"})])
def test_injected_relations_found(market, method, expected):
    cfg, legs, frames = market
    res, tr = X.scan(method, frames, cfg)
    res, cand = X.select(res, cfg, method, "T")
    assert expected <= set(cand["pairing"])
    assert "EURGBP" not in set(cand["pairing"])


@pytest.mark.parametrize("method,expected", [("leadlag", "USDJPY"), ("residual", "GBPUSD")])
def test_placebo_separates_bond_link(market, method, expected):
    """Shifting the bonds removes the injected relation: the pairing's real best t beats every placebo (E4),
    and EURGBP (no relation) does not."""
    cfg, legs, frames = market
    res, _ = X.scan(method, frames, cfg)
    res, _ = X.select(res, cfg, method, "T")
    d0 = X.d0_table(res, X.placebo(method, frames, cfg, "T")).set_index("pairing")
    assert d0.loc[expected, "E4"]
    assert d0.loc["EURGBP", "E1_E2_E3"] == 0


def test_windows_resplit_2020():
    from equity_trading import bond_seasonality as B
    assert X.WINDOWS["H1"] == (D(2020, 1, 1), D(2022, 12, 31))
    assert X.WINDOWS["H2"] == (D(2023, 1, 1), D(2025, 12, 31))
    assert X.WINDOWS["FULL"] == (D(2020, 1, 1), D(2025, 12, 31))
    assert X.WINDOWS is not B.WINDOWS and B.WINDOWS["H1"][0] == D(2018, 1, 1)   # the bond study keeps its windows
