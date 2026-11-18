import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import bond_seasonality as B  # noqa: E402
from equity_trading import fx_seasonality as F  # noqa: E402

D = dt.date


@pytest.fixture
def market_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    monkeypatch.delenv("DUKASCOPY_DIR", raising=False)
    monkeypatch.setitem(B.WINDOWS, "T", (D(2018, 1, 1), D(2019, 12, 31)))
    return tmp_path


def test_guard_and_years():
    with pytest.raises(F.HoldoutError):
        B.guard(D(2025, 1, 1), D(2026, 1, 2))
    B.guard(D(2018, 1, 1), D(2025, 12, 31))
    assert [B.years_needed(n, 0.75) for n in (3, 4, 7, 8)] == [3, 3, 6, 6]
    assert [B.years_needed(n, 0.60) for n in (3, 4, 8)] == [2, 3, 5]


def test_rolls_detected_and_dropped(market_dir):
    base = market_dir / "Dukascopy"
    for s, seed in (("USTBONDTRUSD", 1), ("BUNDTREUR", 2)):
        B.write_synthetic_bond(base, s, D(2017, 12, 28), D(2019, 12, 31), seed=seed, price=120.0)
    cfg = B.BDSConfig()
    for s in ("USTBONDTRUSD", "BUNDTREUR"):
        bd = B.load(s, "T", cfg)
        main = bd.rolls[~bd.rolls["extra"]]
        assert len(main) == 8
        assert (main["gap_bps"] < -40).all() and (main["ratio"] > 10).all()
        assert set(main["before"].dt.date) == set(B.roll_days(D(2018, 1, 1), D(2019, 12, 31)))
        assert (bd.drops["contract switch"] >= 8).all()
    assert B.load("USTBONDTRUSD", "T", cfg).files["sunday_session"]
    assert not B.load("BUNDTREUR", "T", cfg).files["sunday_session"]


def test_roll_jump_biases_break_interval_unless_dropped(market_dir):
    base = market_dir / "Dukascopy"
    B.write_synthetic_bond(base, "USTBONDTRUSD", D(2017, 12, 28), D(2019, 12, 31), seed=3, roll_bps=-150.0)
    cfg = B.BDSConfig(min_year_days=100)
    mean = {}
    for drop in (False, True):
        p = B.load("USTBONDTRUSD", "T", cfg, drop_rolls=drop).panels["UTC"]
        mean[drop] = float(B.interval_returns(p, 21 * 60 - 15, 22 * 60 + 15, 1).mean())   # across the break
    assert mean[False] < -1.5 and abs(mean[True]) < 0.5


def test_injected_drift_found(market_dir):
    base = market_dir / "Dukascopy"
    B.write_synthetic_bond(base, "USTBONDTRUSD", D(2017, 12, 28), D(2019, 12, 31), seed=4, drift_bps=4.0)
    cfg = B.BDSConfig(min_year_days=100)
    bd = B.load("USTBONDTRUSD", "T", cfg)
    sc = pd.concat([B.scan_panel(bd.panels[c], cfg) for c in B.CLOCKS], ignore_index=True)
    n, need = B.d2_needed(bd.panels, cfg)
    s, c = B.select(sc, cfg, need, "UST", "T")
    assert len(c) >= 1
    top = c.iloc[0]
    assert top["direction"] == 1 and top["id"] == "UST-T-01"
    seg = F._utc_segments(top["clock"], int(top["start"]), int(top["end"]))
    assert F._overlap(seg, F._utc_segments("NY", 930, 990))        # 08:30-09:30 New York


def test_missing_file_drops_spanning_days(market_dir):
    base = market_dir / "Dukascopy"
    B.write_synthetic_bond(base, "BUNDTREUR", D(2017, 12, 28), D(2019, 12, 31), seed=5)
    gone = D(2019, 3, 13)
    (base / "BUNDTREUR" / "candles_m1" / "2019" / f"BUNDTREUR_{gone}.parquet").unlink()
    bd = B.load("BUNDTREUR", "T", B.BDSConfig())
    assert gone in bd.files["missing"]
    ny = set(bd.panels["NY"].dates.astype("datetime64[D]").astype(object))
    utc = set(bd.panels["UTC"].dates.astype("datetime64[D]").astype(object))
    assert gone not in ny and gone + dt.timedelta(days=1) not in ny and gone not in utc
    assert gone - dt.timedelta(days=1) in utc


def test_trade_costs(market_dir):
    base = market_dir / "Dukascopy"
    B.write_synthetic_bond(base, "BUNDTREUR", D(2017, 12, 28), D(2018, 12, 31), seed=6, price=135.0)
    cfg = B.BDSConfig()
    p = B.load("BUNDTREUR", "T", cfg).panels["UTC"]
    tr = B.trades(p, 600, 660, -1, cfg)
    tick = 0.01 / p.mid[:, 600] * 1e4
    assert np.allclose(tr["gross_bps"], -(p.logmid[:, 660] - p.logmid[:, 600]) * 1e4 - tick)
    assert np.allclose(tr["net_bps"], tr["gross_bps"] - 0.70)
    assert np.allclose(tr["gross_pips"], -(p.mid[:, 660] - p.mid[:, 600]))
    assert (tr["cfd_net_bps"] < tr["net_bps"]).mean() > 0.95      # 0.05-point CFD spread > one 0.01 tick
