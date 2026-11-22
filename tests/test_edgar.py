"""EDGAR parsers on synthetic bulk files (no network)."""
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from equity_trading import edgar as E


@pytest.fixture()
def edgar_root(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_TRADING_DIR", str(tmp_path))
    return tmp_path


def _cf_doc(cik, filings):
    """filings: list of (val, end, form, filed, accn, start)."""
    obs = [{"start": s, "end": e, "val": v, "accn": a, "fy": 2020, "fp": "FY", "form": f, "filed": fd}
           for v, e, f, fd, a, s in filings]
    return {"cik": cik, "entityName": f"Co {cik}", "facts": {
        "us-gaap": {"Revenues": {"units": {"USD": obs}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2021-02-01", "val": 1e6, "accn": "a1", "fy": 2020, "fp": "FY", "form": "10-K",
             "filed": "2021-02-15"}]}}},
        "srt": {"Ignored": {"units": {"USD": obs}}},
    }}


def test_companyfacts_keeps_restatements_and_filters_forms(edgar_root):
    z = E.raw_dir() / "companyfacts_20261122.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("CIK0000000001.json", json.dumps(_cf_doc(1, [
            (100.0, "2020-12-31", "10-K", "2021-02-15", "a1", "2020-01-01"),
            (90.0, "2020-12-31", "10-K/A", "2021-06-01", "a2", "2020-01-01"),   # restatement: new row
            (5.0, "2020-12-31", "8-K", "2021-01-20", "a0", "2020-01-01"),       # not a periodic form
        ])))
        zf.writestr("CIK0000000002.json", json.dumps(_cf_doc(2, [
            (7.0, "2026-03-31", "10-Q", "2026-05-01", "b1", "2026-01-01")])))
    s = E.parse_companyfacts(per_part=1)
    assert len(s) == 2
    f = E.load_facts(filed_max=None)
    rev = f[(f["tag"] == "Revenues") & (f["cik"] == 1)].sort_values("filed")
    assert rev["val"].tolist() == [100.0, 90.0]
    assert set(f["taxonomy"].astype(str)) == {"us-gaap", "dei"}
    # point in time: as of 2021-03-01 only the original value is known
    pit = rev[rev["filed"] <= "2021-03-01"]
    assert pit["val"].tolist() == [100.0]
    # default load seals 2026
    assert 2 not in set(E.load_facts(tags=["Revenues"])["cik"])
    # resumable: second call parses nothing
    assert len(E.parse_companyfacts(per_part=1)) == 0
    with pytest.raises(RuntimeError):
        E.parse_companyfacts(zip_path=z.rename(z.with_name("companyfacts_20261201.zip")), per_part=1)


def _sub_doc(cik, tickers, rows):
    cols = {k: [r[i] for r in rows] for i, k in enumerate(
        ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "isXBRL",
         "isInlineXBRL", "primaryDocument"])}
    return {"cik": f"{cik:010d}", "name": f"Co {cik}", "entityType": "operating", "sic": "3571",
            "tickers": tickers, "exchanges": ["Nasdaq"] * len(tickers), "formerNames": [],
            "filings": {"recent": cols, "files": []}}


def test_submissions_tickers_and_status(edgar_root):
    z = E.raw_dir() / "submissions_20261122.zip"
    live = [("x1", "2024-02-01", "2023-12-31", "2024-02-01T16:05:00.000Z", "10-K", 1, 1, "abc-20231231.htm"),
            ("x2", "2024-03-01", "", "2024-03-01T08:00:00.000Z", "4", 0, 0, "form4.xml")]
    dead_recent = [("y2", "2016-05-01", "2016-04-30", "2016-05-01T09:00:00.000Z", "25-NSE", 0, 0, "x.htm")]
    dead_old = {"accessionNumber": ["y1"], "filingDate": ["2015-03-01"], "reportDate": ["2014-12-31"],
                "acceptanceDateTime": ["2015-03-01T17:30:00.000Z"], "form": ["10-K"], "isXBRL": [1],
                "isInlineXBRL": [0], "primaryDocument": ["d123d10k.htm"]}
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("CIK0000000001.json", json.dumps(_sub_doc(1, ["ABC"], live)))
        zf.writestr("CIK0000000002.json", json.dumps(_sub_doc(2, [], dead_recent)))
        zf.writestr("CIK0000000002-submissions-001.json", json.dumps(dead_old))
    c, f = E.parse_submissions()
    assert len(c) == 2 and set(f["accn"]) == {"x1", "y1", "y2"}          # Form 4 dropped
    assert f.loc[f["accn"] == "x1", "accepted"].iloc[0] == pd.Timestamp("2024-02-01 16:05:00")
    fsds = pd.DataFrame({"cik": [2], "instance": ["dead-20141231.xml"], "filed": [pd.Timestamp("2015-03-01")]})
    h = E.ticker_history(fsds_sub=fsds)
    assert ("DEAD", "fsds") in set(zip(h.loc[h["cik"] == 2, "symbol"], h.loc[h["cik"] == 2, "source"]))
    assert {"ABC"} <= set(h.loc[h["cik"] == 1, "symbol"])
    st = E.company_status(as_of="2024-12-31").set_index("cik")
    assert bool(st.loc[1, "active"]) and not bool(st.loc[2, "active"])
    assert st.loc[2, "first_delist_form"] == pd.Timestamp("2016-05-01")


@pytest.mark.parametrize("name,exp", [("aapl-20230930.htm", "AAPL"), ("brkb-20231231_htm.xml", "BRKB"),
                                      ("form10-k.htm", None), ("d123456d10k.htm", None),
                                      ("abc-12345678.htm", None), (None, None)])
def test_file_prefix(name, exp):
    assert E.file_prefix(name) == exp


def test_fsds_quarters():
    q = E.fsds_quarters("2009q1", "2010q2")
    assert q == ["2009q1", "2009q2", "2009q3", "2009q4", "2010q1", "2010q2"]


def test_read_fsds_sub():
    txt = ("adsh\tcik\tname\tsic\tcountryba\tcountryinc\tafs\tfye\tform\tperiod\tfy\tfp\tfiled\taccepted\t"
           "prevrpt\tinstance\tnciks\n"
           "0000320193-14-000001\t320193\tAPPLE INC\t3571\tUS\tUS\t1-LAF\t0927\t10-K\t20130930\t2013\tFY\t"
           "20131030\t2013-10-29 20:38:00.0\t0\taapl-20130928.xml\t1\n")
    s = E.read_fsds_sub(io.BytesIO(txt.encode()))
    assert s.loc[0, "accn"] == "0000320193-14-000001" and s.loc[0, "cik"] == 320193
    assert s.loc[0, "filed"] == pd.Timestamp("2013-10-30") and not s.loc[0, "prevrpt"]
    assert E.file_prefix(s.loc[0, "instance"]) == "AAPL"


def test_yahoo_symbol():
    assert E.yahoo_symbol("BRK.B") == "BRK-B"


def test_split_download():
    from equity_trading.daily_prices import split_download
    idx = pd.date_range("2020-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]])
    d = pd.DataFrame(1.0, index=idx, columns=cols)
    d.loc[:, ("BBB", slice(None))] = float("nan")
    out = split_download(d, ["AAA", "BBB", "CCC"])
    assert list(out) == ["AAA"] and "Dividends" in out["AAA"].columns
