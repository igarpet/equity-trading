import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p4  # noqa: E402

CFG = p4.P4Config()
SESS = pd.bdate_range("2021-01-04", periods=30)


def test_parse_8k_202():
    block = {"form": ["8-K", "10-Q", "8-K", "8-K/A"], "filingDate": ["2021-01-05", "2021-01-06", "2021-02-01", "2021-03-01"],
             "acceptanceDateTime": ["", "", "", ""], "items": ["2.02,9.01", "", "5.02", "2.02"]}
    df = p4.parse_8k_202(block)
    assert df["filing_date"].tolist() == ["2021-01-05", "2021-03-01"]


def test_event_flags_window_and_holiday():
    ev = pd.DataFrame({"symbol": ["A", "B"], "filing_date": ["2021-01-06", "2021-01-09"]})   # 01-09 is a Saturday
    f = p4.event_flags(ev, SESS, ["A", "B", "C"], 1, 1)
    a = f.index[f["A"]].strftime("%m-%d").tolist()
    b = f.index[f["B"]].strftime("%m-%d").tolist()
    assert a == ["01-05", "01-06", "01-07"]
    assert b == ["01-08", "01-11", "01-12"]                    # next session is Monday 01-11
    assert not f["C"].any()


def test_any_in_window():
    fl = pd.DataFrame({"x": [False, True, False, False, False]})
    np.testing.assert_array_equal(p4.any_in_window(fl, 2)["x"].to_numpy(), [False, True, True, False, False])


def test_news_session_assignment():
    t = pd.Series(pd.to_datetime(["2021-01-05 14:00", "2021-01-05 21:30", "2021-01-09 15:00"]).tz_localize("UTC"))
    # 09:00 ET same day; 16:30 ET -> next session; Saturday -> Monday
    idx = p4.news_session(t, SESS)
    assert [SESS[i].strftime("%m-%d") for i in idx] == ["01-05", "01-06", "01-11"]


def test_news_counts_drop_roundups_and_shocks():
    t = pd.to_datetime(["2021-01-05 14:00"] * 3 + ["2021-01-05 15:00"]).tz_localize("UTC")
    art = pd.DataFrame({"id": [1, 2, 3, 4], "created_at": t, "symbols": ["A", "A|B", "A|B|C|D", "B"]})
    c = p4.news_counts(art, SESS, ["A", "B"], max_tags=3)
    assert c.loc["2021-01-05", "A"] == 2 and c.loc["2021-01-05", "B"] == 2
    cnt = pd.DataFrame({"A": [1] * 25 + [5, 1]}, index=pd.bdate_range("2021-01-04", periods=27))
    sh = p4.news_shocks(cnt, CFG)
    assert sh["A"].sum() == 1 and sh["A"].iloc[25]


def test_random_exclusion_sizes():
    rng = np.random.default_rng(0)
    pool = np.ones((4, 10), bool)
    ex = np.zeros((4, 10), bool); ex[0, :3] = True; ex[2, 5] = True
    r = p4.random_exclusion(ex, pool, rng)
    assert r.sum(1).tolist() == [3, 0, 1, 0]


def test_sort_returns_excludes_names():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-04", periods=60)
    cols = [f"S{i}" for i in range(50)]
    ret = pd.DataFrame(rng.standard_normal((60, 50)) * 0.01, index=idx, columns=cols)
    sig = ret.rolling(5).sum()
    el = pd.DataFrame(True, index=idx, columns=cols)
    a = p4.sort_returns(sig, el, ret, None, 5, 5, 1, idx[10:])
    el2 = el.copy(); el2.iloc[:, :25] = False
    b = p4.sort_returns(sig, el2, ret, None, 5, 5, 1, idx[10:])
    assert not np.allclose(a["gross"], b["gross"])


def test_daily_arms_detect_planted_effects():
    C, V, etf, ev = p4.synthetic_daily_news(n_per_sector=60, years=[2016, 2017, 2018, 2019], rev=0.3, jump=0.08)
    S = sorted(etf); ret = C[S].pct_change(); k = 5
    lr = np.log(C / C.shift(k)); sR = lr[S]; sI = sR - pd.DataFrame({s: lr[etf[s]] for s in S})
    base = sI.notna() & sR.notna()
    ex = p4.any_in_window(p4.event_flags(ev, C.index, S, 1, 1), k)
    w = C.index[C.index >= "2017-01-01"]
    g = lambda sig, el: p4.sort_returns(sig, el, ret, None, k, 5, 1, w)["gross"].mean()  # noqa: E731
    r, i, n = g(sR, base), g(sI, base), g(sI, base & ~ex)
    assert r < i < n
