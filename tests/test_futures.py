import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import futures as F  # noqa: E402


@pytest.fixture
def market_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    return tmp_path


def test_cme_trade_date_rolls_at_17_chicago():
    ts = pd.Series(pd.to_datetime(["2025-11-04 16:59", "2025-11-04 17:00"]).tz_localize(F.CHICAGO).tz_convert("UTC"))
    td = F.cme_trade_date(ts)
    assert str(td.iloc[0]) == "2025-11-04" and str(td.iloc[1]) == "2025-11-05"


def test_plan_requests_refuse_holdout():
    reqs = F.plan_requests([("MNQ", 2025)])
    assert len(reqs) == 12 and reqs[-1].end == F.HOLDOUT_START
    with pytest.raises(F.HoldoutError):
        F.plan_requests([("MNQ", 2026)])


def test_load_ticks_holdout_guard(market_dir):
    df = F.synthetic_ticks("2026-01-05", days=2)
    F.write_months(df, "ninjatrader", "MNQ")
    with pytest.raises(F.HoldoutError):
        F.load_ticks("MNQ", "2025-12-01", "2026-02-01", "ninjatrader")
    assert len(F.load_ticks("MNQ", "2025-12-01", "2026-02-01", "ninjatrader", allow_holdout=True)) == len(df)
    counts = F.holdout_day_counts("MNQ", "ninjatrader")
    assert counts["ticks"].sum() == len(df) and set(counts.columns) == {"month", "trade_dates", "ticks"}


def test_sign_trades_quote_then_tick_rule():
    df = pd.DataFrame({"ts": pd.date_range("2025-01-02", periods=5, freq="s", tz="UTC"),
                       "price": [100.25, 100.0, 100.0, 100.125, 100.125],
                       "bid": [100.0, 100.0, 100.0, 100.0, np.nan], "ask": [100.25] * 4 + [np.nan],
                       "size": 1, "contract": "X"})
    out = F.sign_trades(df)
    assert out["sign"].tolist() == [1, -1, -1, 1, 1]
    assert out["sign_method"].tolist() == ["quote", "quote", "quote", "tick", "tick"]


def test_sign_trades_keeps_exchange_signs():
    df = F.synthetic_ticks(days=1)
    df.loc[df.index[:10], "price"] = df.loc[df.index[:10], "bid"]       # would be -1 by the quote rule
    out = F.sign_trades(df)
    pd.testing.assert_series_equal(out["sign"], df["sign"])


def test_stitch_front_follows_prior_day_volume():
    df = F.synthetic_ticks(days=10, roll_day=5)
    out = F.stitch_front(df)
    rolls = F.roll_dates(out)
    assert len(rolls) == 1 and rolls["from"].iloc[0] == "MNQ 12-25" and rolls["to"].iloc[0] == "MNQ 03-26"
    days = sorted(set(F.cme_trade_date(df["ts"])))
    assert rolls["trade_date"].iloc[0] == days[6]                          # leader from day 5 -> front from day 6
    assert days[0] not in set(F.cme_trade_date(out["ts"]))                 # first day has no prior day


def _write_nt(df, folder, tz="UTC", five=True):
    folder.mkdir(parents=True, exist_ok=True)
    for c, part in df.groupby("contract"):
        if five:
            (folder / f"{c}.Last.txt").write_text("\n".join(F.to_nt_lines(part, tz)) + "\n")
        else:
            (folder / f"{c}.Last.txt").write_text("\n".join(F.to_nt_lines(part, tz, five_fields=False)) + "\n")
            for side in ("bid", "ask"):
                q = part.assign(price=part[side], ts=part["ts"] - pd.Timedelta(milliseconds=1))
                (folder / f"{c}.{side.title()}.txt").write_text("\n".join(F.to_nt_lines(q, tz, False)) + "\n")


@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_nt_export_roundtrip(tmp_path, tz):
    df = F.synthetic_ticks(days=8, roll_day=4)
    _write_nt(df, tmp_path / "MNQ", tz)
    got = F.load_nt_root(tmp_path / "MNQ", tz)
    want = F.stitch_front(df)
    assert len(got) == len(want)
    assert (got["ts"].to_numpy() == want["ts"].to_numpy()).all()
    assert (got["sign"].to_numpy() == want["sign"].to_numpy()).all()       # quote rule recovers the true side
    assert set(got["sign_method"]) == {"quote"}


def test_nt_three_field_exports_merge_quotes(tmp_path):
    df = F.synthetic_ticks(days=3, roll_day=10)
    _write_nt(df, tmp_path / "MNQ", five=False)
    got = F.load_nt_root(tmp_path / "MNQ", "UTC")
    agree = F.sign_agreement(got, F.stitch_front(df))
    assert agree["match_rate"] > 0.99 and agree["agreement"] > 0.99


def test_parse_nt_contract_names():
    assert F.parse_nt_contract("MNQ 12-25.Last.txt") == ("MNQ", "MNQ 12-25", "Last")
    assert F.parse_nt_contract("MES 03-26.txt") == ("MES", "MES 03-26", "Last")
    with pytest.raises(ValueError):
        F.parse_nt_contract("NQ.csv")


def test_best_lag_detects_clock_shift():
    df = F.synthetic_ticks(days=4, roll_day=10)
    shifted = df.assign(ts=df["ts"] + pd.Timedelta(minutes=60))
    lag, corr = F.best_lag_minutes(shifted, df, max_lag=120)
    assert lag == 60 and corr > 0.5
    assert F.best_lag_minutes(df, df, max_lag=30)[0] == 0


def test_sign_agreement_flags_flipped_signs():
    df = F.synthetic_ticks(days=2, roll_day=10)
    flipped = df.assign(sign=np.where(np.arange(len(df)) % 4 == 0, -df["sign"], df["sign"]).astype(np.int8))
    a = F.sign_agreement(flipped, df)
    assert a["match_rate"] == pytest.approx(1.0) and 0.70 < a["agreement"] < 0.80


def test_normalise_databento_side_mapping():
    raw = pd.DataFrame({"ts_event": pd.date_range("2025-03-03", periods=3, freq="s", tz="UTC"),
                        "side": ["B", "A", "N"], "price": [100.25, 100.0, 100.0], "size": [2, 1, 3],
                        "bid_px_00": [100.0] * 3, "ask_px_00": [100.25] * 3, "instrument_id": [42] * 3,
                        "action": ["T"] * 3})
    out = F.normalise_databento(raw)
    assert out["sign"].tolist() == [1, -1, 0]
    assert out["sign_method"].tolist() == ["exchange", "exchange", "none"]
    assert list(out.columns) == F.TICK_COLUMNS and out["contract"].iloc[0] == "42"


def test_download_stops_at_budget_and_logs_before_paying(market_dir, monkeypatch):
    calls = []

    def get_range(**q):
        calls.append(q["start"])
        Path(q["path"]).write_bytes(b"x")

    cl = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **q: 50.0, get_billable_size=lambda **q: 1),
                         timeseries=SimpleNamespace(get_range=get_range))
    monkeypatch.setattr(F, "normalise_raw_file", lambda path, r: (
        F.synthetic_ticks(f"{r.start:%Y-%m}-05", days=1).to_parquet(
            F.cache_dir("databento", r.root) / f"{r.start:%Y-%m}.parquet"), 1)[1])
    reqs = F.plan_requests([("MNQ", 2025)])[:4]
    log = F.download(cl, reqs, budget=115.0)
    assert log["status"].tolist()[:2] == ["downloaded", "downloaded"]
    assert log["status"].iloc[2].startswith("stopped")
    assert len(calls) == 2 and F.spent_usd() == pytest.approx(100.0)
    again = F.download(cl, reqs[:2], budget=115.0)                         # cached months cost nothing
    assert again["status"].tolist() == ["cached", "cached"] and len(calls) == 2


def test_read_recorder_roundtrip(tmp_path):
    df = F.synthetic_ticks(days=3, roll_day=10)
    lines = ["ts_utc,instrument,price,size,bid,ask"] + [
        f"{t.strftime('%Y-%m-%dT%H:%M:%S.%f')}0Z,{c},{p},{s},{b},{a}"
        for t, c, p, s, b, a in zip(df["ts"], df["contract"], df["price"], df["size"], df["bid"], df["ask"])]
    p = tmp_path / "MNQ_12-25_x.csv"
    p.write_text("\n".join(lines) + "\n")
    out = F.read_recorder([p, p])                                          # duplicated file -> deduplicated
    got = out["MNQ"]
    want = F.stitch_front(df)
    assert len(got) == len(want) and (got["sign"].to_numpy() == want["sign"].to_numpy()).all()


def test_coverage_and_missing_days():
    df = F.synthetic_ticks("2025-11-03", days=4, roll_day=10)
    cov = F.coverage(df)
    assert len(cov) == 4 and (cov["unsigned_share"] == 0).all()
    expected = F.cme_trading_days("2025-11-03", "2025-11-10")
    assert [str(d) for d in expected] == ["2025-11-03", "2025-11-04", "2025-11-05", "2025-11-06", "2025-11-07"]
    assert [str(d) for d in F.missing_trade_dates(cov, expected)] == ["2025-11-07"]


def test_full_size_contracts_are_refused(tmp_path):
    for root in ("NQ", "ES", "YM"):
        with pytest.raises(ValueError):
            F.plan_requests([(root, 2025)])
    df = F.synthetic_ticks(days=2, contracts=("NQ 12-25", "NQ 03-26"))
    folder = tmp_path / "NQ"
    folder.mkdir()
    (folder / "NQ 12-25.Last.txt").write_text("\n".join(F.to_nt_lines(df[df["contract"] == "NQ 12-25"])) + "\n")
    with pytest.raises(ValueError):
        F.load_nt_root(folder, "UTC")
