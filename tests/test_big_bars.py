import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from equity_trading import big_bars as B  # noqa: E402


def _bt():
    idx = pd.date_range("2024-01-31 13:00", periods=200, freq="5min", tz="UTC")      # 08:00 NY onward
    z = np.zeros(200)
    z[[6, 7, 72]] = 4.0                                                               # 08:30, 08:35, 14:00 NY
    return pd.DataFrame({"trade_date": [dt.date(2024, 1, 31)] * 200, "z": z, "ret": np.where(z > 0, 1e-3, 1e-5),
                         "close": 17000.0, "eligible": True}, index=idx)


def test_labels_and_marks():
    e = B.bar_events(_bt(), 3.0, 0.02)
    assert e.loc[e.big_z, "ny_time"].tolist() == ["08:30", "08:35", "14:00"]
    assert e.loc[e.ny_time == "14:00", "fomc_day"].iloc[0]                           # 2024-01-31 is an FOMC day
    mt = B.mark_table(e, "big_z").set_index("slot")
    assert mt.filter(like="08:30", axis=0)["big"].iloc[0] == 1
    assert mt.filter(like="14:00", axis=0)["big"].iloc[0] == 1


def test_clustering():
    e = B.bar_events(_bt(), 3.0, 0.02)
    c = B.clustering(e, "big_z").set_index("within_bars")
    assert c.loc[1, "p_given_big"] > c.loc[1, "p_unconditional"]
