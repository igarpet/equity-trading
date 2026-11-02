import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from equity_trading.pairs import (cluster_pairs, compute_returns, correlation_clusters, gics_pairs,  # noqa: E402
                                  integration_screen, oos_diagnostics, random_null_pairs,
                                  rolling_stability, screen_pairs, thesis_split, top_correlation_pairs)


def _panel(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(n) * 0.01)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.97 * u[t - 1] + rng.standard_normal() * 0.004
    p = pd.DataFrame({
        "AAA": f + u,                                   # cointegrated with BBB
        "BBB": 0.8 * f,
        "CCC": np.cumsum(rng.standard_normal(n) * 0.01),  # independent random walks
        "DDD": np.cumsum(rng.standard_normal(n) * 0.01),
    }, index=pd.date_range("2024-01-02 09:30", periods=n, freq="30min"))
    return np.exp(p + 4)


def test_thesis_split_rules():
    p = _panel()
    p.iloc[:200, 2] = np.nan            # CCC: 200/2100 formation missing (> 5%) and first price missing
    p.iloc[2500, 3] = np.nan            # DDD: one missing OOS bar
    tr, te, av = thesis_split(p)
    assert len(tr) == int(len(p) * 0.7) and list(tr.columns) == ["AAA", "BBB"]
    assert not av.loc["CCC", "retained"] and not av.loc["DDD", "complete_out_of_sample"]


def test_top_correlation_matches_thesis_logic():
    p = _panel()
    corr = compute_returns(p).corr()
    pairs = top_correlation_pairs(corr, top_n=1)
    assert ("AAA", "BBB") in pairs and all(a < b for a, b in pairs)


def test_screen_selects_cointegrated_pair_only():
    lp = np.log(_panel())
    roots = integration_screen(lp, list(lp.columns), n_jobs=1)
    res = screen_pairs(lp, [("AAA", "BBB"), ("CCC", "DDD"), ("AAA", "CCC")], roots, n_jobs=1).set_index("pair")
    assert res.loc["AAA-BBB", "eg_selected"] and res.loc["AAA-BBB", "jo_selected"]
    assert not res.loc["CCC-DDD", "eg_selected"] and not res.loc["CCC-DDD", "jo_selected"]
    assert abs(res.loc["AAA-BBB", "beta"] - 1.25) < 0.1 and abs(res.loc["AAA-BBB", "jo_beta"] - 1.25) < 0.1


def test_eg_matches_thesis_function_exactly():
    from statsmodels.tsa.stattools import coint
    lp = np.log(_panel())
    roots = integration_screen(lp, ["AAA", "BBB"], n_jobs=1)
    r = screen_pairs(lp, [("AAA", "BBB")], roots, johansen=False, n_jobs=1).iloc[0]
    assert r["eg_p"] == coint(lp["AAA"], lp["BBB"], trend="c", autolag="aic")[1]


def test_gics_and_clusters():
    cons = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC", "DDD"], "GICS Sub-Industry": ["x", "x", "y", "z"]})
    assert gics_pairs(["AAA", "BBB", "CCC", "DDD"], cons) == [("AAA", "BBB")]
    corr = compute_returns(_panel()).corr()
    lab = correlation_clusters(corr, n_clusters=3)
    assert lab["AAA"] == lab["BBB"]
    assert ("AAA", "BBB") in cluster_pairs(lab)
    assert len(random_null_pairs(list(corr.columns), {("AAA", "BBB")}, 3)) == 3


def test_rolling_and_oos():
    lp = np.log(_panel())
    tr, te = lp.iloc[:2100], lp.iloc[2100:]
    roll = rolling_stability(tr, [("AAA", "BBB"), ("CCC", "DDD")], window=500, step=250, n_jobs=1).set_index("pair")
    assert roll.loc["AAA-BBB", "roll_pass_share"] > roll.loc["CCC-DDD", "roll_pass_share"]
    roots = integration_screen(tr, list(tr.columns), n_jobs=1)
    sel = screen_pairs(tr, [("AAA", "BBB"), ("CCC", "DDD")], roots, johansen=False, n_jobs=1)
    d = oos_diagnostics(tr, te, sel).set_index("pair")
    assert d.loc["AAA-BBB", "band_hold"] > 0.8 and d.loc["AAA-BBB", "episodes"] > 0


def test_null_pairs_never_loops_when_pool_is_small():
    import warnings
    t = ["A", "B", "C", "D"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = random_null_pairs(t, {("A", "B")}, 3000)
    assert len(out) == 5 and ("A", "B") not in out and w
    assert len(random_null_pairs(t, set(), 3, seed=1)) == 3
