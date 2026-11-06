import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_trading import p6, p7  # noqa: E402

CFG = p7.P7Config(n_ranks=50)


def test_shares_point_in_time_split_basis_and_rescale():
    sess = pd.bdate_range("2014-01-01", "2014-12-31")
    recs = pd.DataFrame({"symbol": ["A", "A", "B", "BB"], "cik": [1, 1, 2, 2], "tag": "EntityCommonStockSharesOutstanding",
                         "end": ["2013-12-31", "2014-06-30", "2014-03-31", "2014-03-31"],
                         "val": [100.0, 110.0, 5.0, 5.0], "filed": ["2014-02-15", "2014-08-01", "2014-04-20", "2014-04-20"]})
    splits = {"A": pd.Series([2.0], index=pd.DatetimeIndex(["2014-05-01"]))}
    sh, diag = p7.shares_daily(recs, splits, {"A": 110.0, "B": 5000.0}, sess)
    assert "BB" not in sh.columns                                  # one ticker per CIK
    assert np.isnan(sh.loc["2014-02-14", "A"])                     # before the first filing
    assert sh.loc["2014-03-03", "A"] == 200.0                      # 100 reported before a 2:1 split -> 200
    assert sh.loc["2014-09-02", "A"] == 110.0                      # reported after the split
    assert sh.loc["2014-05-01", "B"] == 5000.0                     # ratio 1000 outside band -> rescaled
    assert bool(diag.set_index("symbol").loc["B", "rescaled"])


def test_rank_arrays_and_rank_returns():
    cap = pd.DataFrame([[3.0, 2.0, 1.0], [1.5, 2.2, 1.1]], columns=list("abc"))
    ra = p7.rank_arrays(cap, pd.DataFrame(True, index=cap.index, columns=cap.columns), 2)
    np.testing.assert_array_equal(ra["stock_of_rank"], [[0, 1], [1, 0]])
    rr = p7.rank_returns(ra["sorted_cap"])
    np.testing.assert_allclose(rr[1], [2.2 / 3.0 - 1, 1.5 / 2.0 - 1])   # rank 1 is now b, but rank return uses rank caps


def test_dataset_alignment_and_tradable_vs_rank():
    R, cap = p7.synthetic_market(n_days=500, n_stocks=80)
    cost = pd.DataFrame(1e-4, index=R.index, columns=R.columns)
    d = p7.build_rank_dataset(R, cap, cost, CFG, verbose=False)
    t = 400
    j = d["smap"][t, 0]
    assert j == int(np.argmax(cap.iloc[t + 1].to_numpy()))                  # rank 1 at the trade day
    np.testing.assert_allclose(d["tret"][t, 0], R.iloc[t + 2, j], rtol=1e-6)
    np.testing.assert_allclose(d["ret"][t], d["rank_return"][t + 2], rtol=1e-6)


def test_to_stock_and_evaluate_costs_remapping():
    W = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    smap = np.array([[0, 1], [0, 1], [1, 0]])                               # ranks swap on the third day
    d = {"tret": np.zeros((3, 2)), "ret": np.zeros((3, 2)), "smap": smap, "cost_stock": np.full((3, 2), 1e-3),
         "columns": ["a", "b"], "index": pd.bdate_range("2020-01-01", periods=5)}
    e = p7.evaluate(W, d, np.array([1, 2]))
    np.testing.assert_allclose(e["rank_turnover"], [0, 0])
    np.testing.assert_allclose(e["stock_turnover"], [0, 2.0])               # the swap flips both positions
    np.testing.assert_allclose(e["cost"], [0, 2e-3])


def test_models_train_in_rank_space():
    pytest.importorskip("torch")
    from equity_trading import p7_model
    R, cap = p7.synthetic_market(n_days=700, n_stocks=80)
    cost = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    d = p7.build_rank_dataset(R, cap, cost, CFG, verbose=False)
    rows = np.flatnonzero(d["ok"].sum(1) > 20)
    for mode in ("paper", "net"):
        m, info = p7_model.fit(d, rows[:200], rows[200:300], seed=0, mode=mode, epochs=3, patience=3, device="cpu")
        s = p7_model.predict_scores(m, d, rows[300:])
        assert np.isfinite(s[d["ok"][rows[300:]]]).all() and np.isfinite(info["best_valid"])


def test_tags_merged_by_priority_and_spikes_dropped():
    sess = pd.bdate_range("2014-01-01", "2014-12-31")
    recs = pd.DataFrame({"symbol": "A", "cik": 1,
                         "tag": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding",
                                 "CommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
                         "end": ["2013-12-31", "2013-12-31", "2014-03-31", "2014-06-30", "2014-09-30"],
                         "val": [99.0, 100.0, 100000.0, 101.0, 102.0],
                         "filed": ["2014-02-01", "2014-02-01", "2014-05-01", "2014-08-01", "2014-11-01"]})
    sh, diag = p7.shares_daily(recs, {}, {}, sess)
    assert sh.loc["2014-03-03", "A"] == 100.0            # dei wins on the same filing date
    assert sh.loc["2014-06-02", "A"] == 100.0            # the 1000x spike is dropped (previous value carried)
    assert sh.loc["2014-12-01", "A"] == 102.0 and int(diag["dropped"].iloc[0]) == 1


def test_common_rank_returns_ignore_entry_of_a_large_stock():
    C = np.array([[np.nan, 5.0, 4.0, 3.0], [100.0, 5.5, 4.0, 3.3]])     # a mega cap appears on day 1
    r = p7.common_rank_returns(C, 2)
    np.testing.assert_allclose(r[1], [5.5 / 5 - 1, 4.0 / 4.0 - 1])
    old = p7.rank_returns(np.array([[5.0, 4.0], [100.0, 5.5]]))
    assert old[1, 0] > 10                                             # the 0.1.0 definition jumped
