"""fv_panel on synthetic facts (no data files)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from equity_trading import fv_panel as F  # noqa: E402

T = pd.Timestamp


def _facts():
    r = []

    def add(tag, start, end, val, accn, acc):
        r.append({"cik": 1, "tag": tag, "start": T(start) if start else pd.NaT, "end": T(end), "val": float(val),
                  "accn": accn, "form": "10-K" if accn.startswith("K") else "10-Q", "filed": T(acc).normalize(),
                  "accepted": T(acc)})
    add("Revenues", "2021-01-01", "2021-12-31", 100, "K21", "2022-02-10 16:30")
    add("Revenues", "2022-01-01", "2022-12-31", 120, "K22", "2023-02-10 16:30")
    add("Revenues", "2021-01-01", "2021-12-31", 105, "K22", "2023-02-10 16:30")  # restated comparative
    add("Revenues", "2023-01-01", "2023-03-31", 35, "Q123", "2023-05-05 08:00")
    add("Revenues", "2022-01-01", "2022-03-31", 28, "Q123", "2023-05-05 08:00")
    add("Revenues", "2023-04-01", "2023-06-30", 35, "Q223", "2023-08-04 08:00")   # 3-month, ignored by TTM
    add("Revenues", "2023-01-01", "2023-06-30", 70, "Q223", "2023-08-04 08:00")
    add("Revenues", "2022-01-01", "2022-06-30", 58, "Q223", "2023-08-04 08:00")
    # cash flow: only YTD in 10-Qs
    add("NetCashProvidedByUsedInOperatingActivities", "2022-01-01", "2022-12-31", 40, "K22", "2023-02-10 16:30")
    add("NetCashProvidedByUsedInOperatingActivities", "2023-01-01", "2023-06-30", 25, "Q223", "2023-08-04 08:00")
    add("NetCashProvidedByUsedInOperatingActivities", "2022-01-01", "2022-06-30", 15, "Q223", "2023-08-04 08:00")
    add("PaymentsToAcquirePropertyPlantAndEquipment", "2023-01-01", "2023-06-30", 5, "Q223", "2023-08-04 08:00")
    add("PaymentsToAcquirePropertyPlantAndEquipment", "2022-01-01", "2022-06-30", 4, "Q223", "2023-08-04 08:00")
    add("PaymentsToAcquirePropertyPlantAndEquipment", "2022-01-01", "2022-12-31", 9, "K22", "2023-02-10 16:30")
    add("CashAndCashEquivalentsAtCarryingValue", None, "2023-06-30", 50, "Q223", "2023-08-04 08:00")
    add("LongTermDebtNoncurrent", None, "2023-06-30", 30, "Q223", "2023-08-04 08:00")
    add("LongTermDebtCurrent", None, "2023-06-30", 5, "Q223", "2023-08-04 08:00")
    add("EntityCommonStockSharesOutstanding", None, "2023-07-28", 10, "Q223", "2023-08-04 08:00")
    add("EntityPublicFloat", None, "2022-06-30", 400, "K22", "2023-02-10 16:30")
    f = pd.DataFrame(r)
    fil = (f.groupby("accn").agg(form=("form", "first"), accepted=("accepted", "first")).reset_index())
    fil["report_date"] = fil["accn"].map({"K21": T("2021-12-31"), "K22": T("2022-12-31"),
                                          "Q123": T("2023-03-31"), "Q223": T("2023-06-30")})
    return f, fil


def test_snapshots_ttm_history_and_point_in_time():
    f, fil = _facts()
    s = F.derive(pd.DataFrame(F.company_snapshots(f, fil))).set_index("accn")
    assert s.loc["K21", "ttm_revenue"] == 100 and s.loc["K21", "fy0_revenue"] == 100
    # K22 restates FY2021 to 105: visible from K22 onwards only
    assert s.loc["K22", "fy0_revenue"] == 120 and s.loc["K22", "fy1_revenue"] == 105
    assert s.loc["Q123", "ttm_revenue"] == 120 + 35 - 28
    assert s.loc["Q223", "ttm_revenue"] == 120 + 70 - 58
    assert s.loc["Q223", "ttm_cfo"] == 40 + 25 - 15 and s.loc["Q223", "ttm_capex"] == 9 + 5 - 4
    assert s.loc["Q223", "ttm_fcf"] == 50 - 10
    assert np.isnan(s.loc["Q123", "ttm_cfo"])           # no YTD CFO in Q1 filing
    assert s.loc["Q223", "debt"] == 35 and s.loc["Q223", "cash"] == 50
    assert s.loc["Q223", "shares"] == 10 and s.loc["Q223", "shares_source"] == "dei"
    assert np.isnan(s.loc["Q123", "shares"])            # cover shares only known from Q223
    assert s.loc["Q223", "public_float"] == 400


def test_monthly_panel_uses_filing_only_after_acceptance():
    f, fil = _facts()
    snap = F.derive(pd.DataFrame(F.company_snapshots(f, fil)))
    months = pd.to_datetime(["2023-07-31", "2023-08-04", "2023-08-31"])
    a = pd.DataFrame({"cik": 1, "month_end": months, "symbol": "XYZ", "close_raw": 100.0, "close_adj": 100.0,
                      "dollar_vol_med": 1e6, "last_date": months})
    comp = pd.DataFrame({"cik": [1], "sic": ["3571"], "name": ["X"]})
    p = F.monthly_panel(snap, a, comp).set_index("month_end")
    assert p.loc["2023-07-31", "accn"] == "Q123"
    assert p.loc["2023-08-04", "accn"] == "Q123"        # accepted 08:00 that day: not usable at that date
    assert p.loc["2023-08-31", "accn"] == "Q223"
    r = p.loc["2023-08-31"]
    assert r["mcap"] == 1000 and r["ev"] == 1000 + 35 - 50
    assert not r["is_financial"] and not r["mcap_suspect"]


def test_symbol_windows_rename_and_reuse():
    th = pd.DataFrame([
        {"cik": 2, "symbol": "OSTK", "source": "fsds", "first_filed": T("2012-03-01"), "last_filed": T("2023-08-01")},
        {"cik": 2, "symbol": "BBBY", "source": "current", "first_filed": T("2012-03-01"), "last_filed": T("2025-08-01")},
        {"cik": 3, "symbol": "BBBY", "source": "fsds", "first_filed": T("2012-04-01"), "last_filed": T("2023-01-01")},
    ])
    status = pd.DataFrame({"cik": [2, 3], "first_periodic": [T("2012-03-01"), T("2012-04-01")],
                           "last_periodic": [T("2025-08-01"), T("2023-01-01")]})
    w = F.symbol_windows(th, status)
    months = pd.date_range("2016-01-31", "2025-12-31", freq="ME")
    prices = pd.concat([pd.DataFrame({"symbol": s, "month_end": months, "last_date": months, "close_raw": 1.0,
                                      "close_adj": 1.0, "dollar_vol_med": 1.0}) for s in ("BBBY", "OSTK")])
    a = F.assign_symbols(w, prices).set_index(["symbol", "month_end"])["cik"]
    assert a.loc[("BBBY", T("2020-06-30"))] == 3
    assert a.loc[("BBBY", T("2025-06-30"))] == 2
    assert a.loc[("OSTK", T("2020-06-30"))] == 2
    # CIK 2 holds one symbol per month: OSTK before the rename
    a2 = F.assign_symbols(w, prices)
    assert not a2.duplicated(["cik", "month_end"]).any()


def test_stale_price_not_used():
    w = pd.DataFrame({"cik": [1], "symbol": ["AAA"], "w0": [T("2015-01-01")], "core0": [T("2015-01-01")],
                      "w1": [T("2026-01-01")], "source": ["prefix"]})
    pr = pd.DataFrame({"symbol": "AAA", "month_end": [T("2020-01-31"), T("2020-02-29")],
                       "last_date": [T("2020-01-31"), T("2020-02-03")], "close_raw": 1.0, "close_adj": 1.0,
                       "dollar_vol_med": 1.0})
    a = F.assign_symbols(w, pr)
    assert a["month_end"].tolist() == [T("2020-01-31")]


def test_delisting_form_caps_window():
    th = pd.DataFrame([{"cik": 7, "symbol": "OLD", "source": "fsds", "first_filed": T("2012-03-01"),
                        "last_filed": T("2023-02-15")}])
    status = pd.DataFrame({"cik": [7], "first_periodic": [T("2012-03-01")], "last_periodic": [T("2023-02-15")],
                           "first_delist_form": [T("2023-05-01")]})
    w = F.symbol_windows(th, status)
    assert w["w1"].iloc[0] == T("2023-05-31")


def test_share_classes_summed_and_float_check():
    f, fil = _facts()
    extra = f[f["tag"] == "EntityCommonStockSharesOutstanding"].copy()
    extra["val"] = 5.0                                   # class B, same filing and date
    zero_float = f[f["tag"] == "EntityPublicFloat"].copy()
    zero_float["val"] = 0.0                              # class B float (not traded)
    f2 = pd.concat([f, extra, zero_float, extra], ignore_index=True)   # exact duplicate counted once
    s = F.derive(pd.DataFrame(F.company_snapshots(f2, fil))).set_index("accn")
    assert s.loc["Q223", "shares"] == 15 and s.loc["Q223", "public_float"] == 400
    snap = s.reset_index()
    a = pd.DataFrame({"cik": 1, "month_end": [T("2023-08-31")], "symbol": "XYZ", "close_raw": 100.0,
                      "close_adj": 100.0, "dollar_vol_med": 1e6, "last_date": [T("2023-08-31")]})
    prices = pd.DataFrame({"symbol": "XYZ", "month_end": [T("2022-06-30")], "close_raw": [40.0]})
    comp = pd.DataFrame({"cik": [1], "sic": ["3571"], "name": ["X"]})
    r = F.monthly_panel(snap, a, comp, prices=prices).iloc[0]
    # float shares = 400 / 40 = 10; shares 15 -> ratio 1.5, fine even though mcap/float = 1500/400
    assert r["shares_to_float"] == 1.5 and not r["mcap_suspect"]


def test_diluted_shares_fallback_not_ttm():
    f, fil = _facts()
    f = f[f["tag"] != "EntityCommonStockSharesOutstanding"]
    add = pd.DataFrame([{"cik": 1, "tag": "WeightedAverageNumberOfDilutedSharesOutstanding", "start": T(st),
                         "end": T("2023-06-30"), "val": v, "accn": "Q223", "form": "10-Q",
                         "filed": T("2023-08-04"), "accepted": T("2023-08-04 08:00")}
                        for st, v in [("2023-04-01", 12.0), ("2023-01-01", 11.0)]])
    s = F.derive(pd.DataFrame(F.company_snapshots(pd.concat([f, add]), fil))).set_index("accn")
    assert s.loc["Q223", "shares"] == 12 and s.loc["Q223", "shares_source"] == "diluted_avg"


def test_warrants_and_units_dropped():
    th = pd.DataFrame({"cik": [1, 1, 1, 1, 2], "symbol": ["ZCAR", "ZCARW", "ZCAR.WS", "ZCARU", "ABCW"],
                       "source": "current", "first_filed": T("2020-01-01"), "last_filed": T("2024-01-01")})
    assert F.is_derivative_symbol(th).tolist() == [False, True, True, True, False]
