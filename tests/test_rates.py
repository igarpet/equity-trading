import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import bond_fx as X  # noqa: E402
from equity_trading import rates as R  # noqa: E402

D = dt.date


@pytest.fixture
def market_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    monkeypatch.delenv("DUKASCOPY_DIR", raising=False)
    return tmp_path


def test_guard():
    with pytest.raises(R.HoldoutError):
        R.guard(D(2026, 1, 1))
    R.guard(D(2025, 12, 31))


def test_normalise_minute_labels_and_bad_quotes():
    idx = pd.to_datetime(["2020-03-02 14:31:00", "2020-03-02 14:31:00.500", "2020-03-02 14:32:00", "2020-03-02 14:33:00"], format="ISO8601",
                         utc=True)
    raw = pd.DataFrame({"bid_px_00": [110.0, 110.01, 0.0, 110.02], "ask_px_00": [110.01, 110.02, 110.0, 110.03],
                        "bid_sz_00": 5, "ask_sz_00": 6, "instrument_id": 7}, index=idx)
    df = R.normalise(raw)
    # a record at 14:31:00 closes the 14:30 minute; 14:31:00.5 belongs to the 14:31 minute; zero bid dropped
    assert list(df["ts"].dt.strftime("%H:%M")) == ["14:30", "14:31", "14:32"]
    assert df["bid_c"].tolist() == [110.0, 110.01, 110.02]


def test_budget_stops_before_buying(market_dir):
    calls = []
    cl = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **q: 5.0),
                         timeseries=SimpleNamespace(get_range=lambda **q: calls.append(q)))
    log = R.download(cl, ["ZT"], 2016, 2016, budget=4.0, log=lambda *_: None)
    assert log.iloc[-1]["status"].startswith("stopped") and not calls
    assert not (market_dir / "Databento" / "ledger.csv").exists()


def test_rate_legs_rolls_removed_and_relations_found(market_dir):
    X.write_synthetic_rates(market_dir, D(2017, 12, 28), D(2018, 12, 31), seed=4)
    X.STUDIES["T"] = X.Study("T", X.STUDIES["RFX"].pairings, {"T": (D(2018, 1, 1), D(2018, 12, 31))})
    try:
        cfg = X.BFXConfig(n_placebo=1, min_trades=40, min_year_days=100)
        legs = {}
        frames = {p: X.frame(p, "T", cfg, legs, study="T") for p in ("USDJPY", "GBPUSD")}
        assert len(legs["ZT"].rolls) == 4 and (legs["ZT"].rolls["gap_bps"] < -25).all()
        ry, rX = frames["USDJPY"].returns()
        assert np.nanmax(np.abs(rX)) * 1e4 < 25                      # the -30 bp roll jumps are removed
        res, _ = X.scan("leadlag", frames, cfg)
        res, cand = X.select(res, cfg, "leadlag", "T")
        assert "USDJPY" in set(cand["pairing"])
    finally:
        X.STUDIES.pop("T", None)


def test_study_windows():
    assert X.STUDIES["RFX"].windows["H1"] == (D(2016, 1, 1), D(2020, 12, 31))
    assert X.STUDIES["SOFR"].windows["H2"] == (D(2024, 1, 1), D(2025, 12, 31))
    assert X.STUDIES["BFX"].windows["H1"][0] == D(2020, 1, 1)
    assert X.tick_size("ZT") == 1 / 256 and X.tick_size("BUNDTREUR") == 0.01
