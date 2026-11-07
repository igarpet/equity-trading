import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import gotobi as G  # noqa: E402

D = dt.date


@pytest.fixture
def market_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    monkeypatch.delenv("DUKASCOPY_DIR", raising=False)
    return tmp_path


def test_bank_days():
    assert G.is_jp_bank_day(D(2024, 1, 4))
    assert not G.is_jp_bank_day(D(2024, 1, 3))          # bank year-end closure
    assert not G.is_jp_bank_day(D(2023, 12, 31))
    assert not G.is_jp_bank_day(D(2024, 5, 3))          # Constitution Day
    assert not G.is_jp_bank_day(D(2024, 5, 4))          # Saturday


def test_gotobi_rollback_and_february():
    g = G.gotobi_days(D(2024, 2, 1), D(2024, 3, 31))
    # 2024-02-10 Sat -> 02-09; 02-25 Sun -> 02-22 (02-23 holiday); no 30th in February
    assert D(2024, 2, 9) in g and D(2024, 2, 10) not in g
    assert D(2024, 2, 22) in g and D(2024, 2, 23) not in g
    assert [d for d in g if d.month == 2] == [D(2024, 2, 5), D(2024, 2, 9), D(2024, 2, 15),
                                               D(2024, 2, 20), D(2024, 2, 22)]
    # March 2024: 30th is Saturday -> 29th
    assert D(2024, 3, 29) in g
    gm = G.gotobi_days(D(2024, 2, 1), D(2024, 2, 29), month_end=True)
    assert D(2024, 2, 29) in gm and D(2024, 2, 29) not in g


def test_day_table_labels():
    t = G.day_table(D(2024, 4, 1), D(2024, 5, 31))
    row = t.set_index("date")
    assert row.loc[D(2024, 5, 3), "day_type"] == "jp_holiday"
    assert row.loc[D(2024, 4, 30), "day_type"] == "gotobi" and row.loc[D(2024, 4, 30), "gotobi_month_end"]
    assert row.loc[D(2024, 5, 10), "day_type"] == "gotobi" and not row.loc[D(2024, 5, 10), "gotobi_month_end"]
    assert row.loc[D(2024, 5, 9), "day_type"] == "control"


def test_shifted_calendar_keeps_count():
    t = G.day_table(D(2024, 1, 1), D(2024, 12, 31))
    for k in (-2, -1, 1, 2):
        s = G.shifted_calendar(t, k)
        assert abs(int(s.sum()) - int(t["gotobi"].sum())) <= abs(k)
        assert not (s & ~t["jp_bank_day"]).any()


def test_short_trade_fills_on_quoted_side_and_stop():
    t = np.array([0, 55 * 60_000, 60 * 60_000, 90 * 60_000], dtype=np.int64)
    bid = np.array([110.00, 110.00, 109.95, 109.97])
    ask = np.array([110.01, 110.01, 109.96, 109.98])
    pnl, why = G.short_trade(t, bid, ask, 55 * 60_000, 90 * 60_000, 12, 30_000)
    assert why == "time" and pnl == pytest.approx((110.00 - 109.98) / 0.01)
    ask2 = ask.copy(); ask2[2] = 110.13
    pnl, why = G.short_trade(t, bid, ask2, 55 * 60_000, 90 * 60_000, 12, 30_000)
    assert why == "stop" and pnl == pytest.approx(-13.0)
    pnl, why = G.short_trade(t, bid, ask, 56 * 60_000, 90 * 60_000, 12, 30_000)   # no tick within 30 s
    assert why == "no_entry" and np.isnan(pnl)


def test_commission_pips():
    assert G.commission_pips(153.0, 35) == pytest.approx(1.071)
    assert G.commission_pips(80.0, 35) == pytest.approx(0.56)


def test_holdout_guard(market_dir):
    with pytest.raises(G.HoldoutError):
        G.load_morning(market_dir, D(2026, 1, 5))
    with pytest.raises(G.HoldoutError):
        G.build_day_outcomes(D(2025, 12, 1), D(2026, 1, 10), G.GTFConfig())


def test_random_calendar_same_count():
    pnl = np.arange(20, dtype=float)
    month = np.array([1] * 10 + [2] * 10)
    is_g = np.zeros(20, bool); is_g[[0, 1, 10]] = True
    m = G.random_calendar_means(pnl, month, is_g, 500, 0)
    assert m.shape == (500,) and m.min() >= 0 and m.max() <= 19


def test_synthetic_end_to_end(market_dir):
    base = market_dir / "Dukascopy"
    G.write_synthetic(base, D(2015, 1, 1), D(2015, 3, 31))
    root = G.dukascopy_root()
    assert root.name == "hours_00-01-02-22-23"
    cfg = G.GTFConfig()
    out = G.build_day_outcomes(D(2015, 1, 1), D(2015, 3, 31), cfg, root=root, n_jobs=1)
    k = G.grid_key(55, 90, 12.0)
    g = out.loc[out["day_type"] == "gotobi", k].mean()
    c = out.loc[out["day_type"] == "control", k].mean()
    assert g > c + 1.5                       # injected -3 pip post-fix move on gotobi days
    assert out.loc[out["has_file"], "spread_fix_pips"].round(6).eq(0.3).all()


def test_c1_previous_hour_supplies_0900_quote(tmp_path):
    root = tmp_path / "USDJPY" / "hours_00-01-02-22-23"
    prev, day = D(2024, 3, 4), D(2024, 3, 5)
    for d, secs, px in [(prev, [23 * 3600 + 3590], 150.000), (day, [30, 55 * 60, 90 * 60], 150.100)]:
        f = G.day_file(root, d); f.parent.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp(d, tz="UTC") + pd.to_timedelta(secs, unit="s")
        pd.DataFrame({"ts": ts, "bid": px, "ask": px + 0.003}).to_parquet(f, index=False)
    t, bid, ask = G.load_morning(root, day)
    assert t[0] == -10_000 and (t >= 0).sum() == 3
    assert G.mid_at(t, bid, ask, 0, 120_000) == pytest.approx(150.0015)
    out = G.day_outcome(t, bid, ask, G.GTFConfig())
    assert out["entry_price"] == pytest.approx(150.100) and out["n_ticks"] == 3


def test_combined_net_charges_two_round_trips():
    d = pd.DataFrame({"long_e25": [1.0], "short": [2.0], "long_price_e25": [150.0], "entry_price": [150.0]})
    net = G.combined_net(d, "long_e25", "short", "long_price_e25", 35.0, 0.1)
    assert net.iloc[0] == pytest.approx(3.0 - 2 * 1.05 - 0.4)


def test_synthetic_long_leg(market_dir):
    base = market_dir / "Dukascopy"
    G.write_synthetic(base, D(2015, 1, 1), D(2015, 6, 30))
    out = G.build_day_outcomes(D(2015, 1, 1), D(2015, 6, 30), G.GTFConfig(), root=G.dukascopy_root(), n_jobs=1)
    comb = G.combined_pnl(out, "long_e25", G.grid_key(55, 90, 12.0))
    assert comb[out["day_type"] == "gotobi"].mean() > comb[out["day_type"] == "control"].mean() + 2.0
