"""Point-in-time fundamentals panel for the Fair_Value branch (FV 02).

Three products, all written under <EQUITY_TRADING_DIR>/EquityTrading/fair_value/:

1. ``snapshots/part-NNNNN.parquet`` — one row per periodic filing (10-K, 10-Q, 10-KT, 10-QT and amendments)
   of a domestic us-gaap filer: what was known about the company at that filing's acceptance time.
   - TTM flows at the filing's period end: TTM = FY_prev + YTD_cur - YTD_prev (or the FY itself at a
     10-K). Works for income-statement and cash-flow items alike (10-Q cash flows are year-to-date).
   - Annual history: the last 5 fiscal years (fy0 = latest) for the DCF items.
   - Balance-sheet items at the period end; shares outstanding from the cover page (dei), else us-gaap.
   - "Known" = facts whose filing was accepted at or before this filing (restatements replace older
     values only from their own acceptance time onwards).
2. ``month_end_prices.parquet`` — Alpaca 1Day month-end raw and adjusted closes per symbol, the last bar
   date and the median daily dollar volume.
3. ``panel_monthly.parquet`` — one row per (CIK, month end) 2016-01 .. 2025-12: latest snapshot accepted
   strictly before the month-end date (so tradable at the earliest the next session), the assigned symbol,
   raw close, market cap, enterprise value and universe flags. 2026 is sealed.

Scope v0.1: domestic us-gaap filers only (20-F / 40-F IFRS filers are left out); financials (SIC 6000-6799)
and blank checks (SIC 6770) are kept but flagged, filtering happens in the studies.
"""
from __future__ import annotations

import bisect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import edgar as E
from .data import safe_name, vendor_dir, work_dir

FV_PANEL_VERSION = "0.1.1"

DOMESTIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A"})

# Tag priority lists (first = preferred). Flow items are duration facts, stock items are instant facts.
FLOW_ITEMS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense", "OperatingLeasesIncomeStatementLeaseRevenue"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
                         "InterestPaidNet", "InterestPaid"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization", "Depreciation"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
}
# Duration items that are not additive (no TTM): value of the shortest period ending at the period end.
PERIOD_ITEMS: dict[str, list[str]] = {
    "diluted_shares_q": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"],
}
STOCK_ITEMS: dict[str, list[str]] = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "Cash"],
    "short_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                          "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
                        "LongTermDebtAndFinanceLeaseObligationsNoncurrent" ],
    "debt_current": ["LongTermDebtCurrent", "DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "debt_total": ["LongTermDebt", "DebtInstrumentCarryingAmount", "DebtLongtermAndShorttermCombinedAmount"],
    "short_borrowings": ["ShortTermBorrowings", "CommercialPaper"],
    "operating_lease_liability": ["OperatingLeaseLiability"],
    "preferred": ["PreferredStockValue"],
    "minority_interest": ["MinorityInterest"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "shares_gaap": ["CommonStockSharesOutstanding"],
}
DEI_ITEMS = {"shares_cover": "EntityCommonStockSharesOutstanding", "public_float": "EntityPublicFloat"}
HISTORY_ITEMS = ["revenue", "operating_income", "net_income", "cfo", "capex", "d_and_a"]
N_HISTORY = 5

_TAG_ITEM: dict[str, tuple[str, int]] = {}
for _items in (FLOW_ITEMS, STOCK_ITEMS, PERIOD_ITEMS):
    for _it, _tags in _items.items():
        for _p, _t in enumerate(_tags):
            _TAG_ITEM[_t] = (_it, _p)
for _it, _t in DEI_ITEMS.items():
    _TAG_ITEM[_t] = (_it, 0)
ALL_TAGS = sorted(_TAG_ITEM)

DEI_TAGS = frozenset(DEI_ITEMS.values())
DAY = np.timedelta64(1, "D")


def fv_dir() -> Path:
    d = work_dir() / "fair_value"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================================ snapshots

class _ItemStore:
    """Latest known values of one item: (start, end) -> {priority: (accepted, value)}; ends kept sorted."""

    __slots__ = ("cells", "ends", "by_end")

    def __init__(self):
        self.cells: dict[tuple, dict[int, tuple]] = {}
        self.ends: list = []
        self.by_end: dict = {}

    def add(self, start, end, prio, accepted, val):
        key = (start, end)
        c = self.cells.get(key)
        if c is None:
            c = self.cells[key] = {}
            if end not in self.by_end:
                bisect.insort(self.ends, end)
                self.by_end[end] = set()
            self.by_end[end].add(start)
        old = c.get(prio)
        if old is None or accepted >= old[0]:
            c[prio] = (accepted, val)

    def value(self, start, end):
        c = self.cells.get((start, end))
        if not c:
            return np.nan
        return c[min(c)][1]

    def ends_near(self, target, tol_days: int):
        lo = bisect.bisect_left(self.ends, target - tol_days * DAY)
        hi = bisect.bisect_right(self.ends, target + tol_days * DAY)
        return self.ends[lo:hi]

    def durations_at(self, end):
        """[(days, start, value)] of duration facts ending at end, longest first."""
        out = []
        for st in self.by_end.get(end, ()):
            if st is None:
                continue
            v = self.value(st, end)
            if not np.isnan(v):
                out.append((int((end - st) / DAY) + 1, st, v))
        return sorted(out, key=lambda x: -x[0])

    def instant(self, end, tol_days: int = 5):
        best = None
        for e in self.ends_near(end, tol_days):
            if None in self.by_end[e]:
                d = abs(int((e - end) / DAY))
                if best is None or d < best[0]:
                    best = (d, self.value(None, e))
        return np.nan if best is None else best[1]

    def latest_instant(self, not_after):
        """Value at the latest instant end <= not_after (cover-page items such as shares outstanding)."""
        i = bisect.bisect_right(self.ends, not_after)
        while i > 0:
            e = self.ends[i - 1]
            if None in self.by_end[e]:
                return self.value(None, e), e
            i -= 1
        return np.nan, None


def _is_fy(days: int) -> bool:
    return 350 <= days <= 380


def ttm(store: _ItemStore, period_end, tol: int = 10) -> float:
    """Trailing twelve months ending at period_end, or NaN.

    10-K period: the FY value. 10-Q period: FY_prev + YTD_cur - YTD_prev, where YTD_cur is the longest
    duration < 350 days ending at period_end, FY_prev ends the day before YTD_cur starts, and YTD_prev has
    the same length (±tol days) ending one year earlier (±tol days).
    """
    if store is None:
        return np.nan
    durs = []
    for e in store.ends_near(period_end, 3):
        durs += store.durations_at(e)
    if not durs:
        return np.nan
    durs.sort(key=lambda x: -x[0])
    for days, st, v in durs:
        if _is_fy(days):
            return v
    days, st, ytd = next(((d, s, v) for d, s, v in durs if d < 350), (None, None, None))
    if days is None:
        return np.nan
    fy_end = st - DAY
    fy = None
    for e in store.ends_near(fy_end, tol):
        for d, s, v in store.durations_at(e):
            if _is_fy(d):
                fy = v
                break
        if fy is not None:
            break
    if fy is None:
        return np.nan
    prev = None
    for e in store.ends_near(period_end - 365 * DAY, tol):
        for d, s, v in store.durations_at(e):
            if abs(d - days) <= tol:
                prev = v
                break
        if prev is not None:
            break
    if prev is None:
        return np.nan
    return fy + ytd - prev


def period_value(store: _ItemStore, period_end) -> float:
    """Value of the shortest duration ending at period_end (±3 days): the latest quarter for 10-Qs."""
    if store is None:
        return np.nan
    durs = []
    for e in store.ends_near(period_end, 3):
        durs += store.durations_at(e)
    return min(durs, key=lambda x: x[0])[2] if durs else np.nan


def combine_share_classes(facts: pd.DataFrame) -> pd.DataFrame:
    """Sum cover-page dei values (shares outstanding, public float) reported per share class in one filing.

    companyfacts strips the class dimension: a dual-class company (CRWD, GOOG) has several
    EntityCommonStockSharesOutstanding rows with the same end date and accession. Without summing, the
    last class read wins (often a class B with 0 or few shares). Exact duplicate rows are counted once.
    """
    dei = facts["tag"].astype(str).isin(DEI_TAGS)
    if not dei.any():
        return facts
    d = facts[dei].drop_duplicates(["tag", "end", "accn", "val"])
    keys = ["tag", "end", "accn"]
    agg = d.groupby(keys, observed=True, as_index=False, dropna=False).agg(
        val=("val", "sum"), **{c: (c, "first") for c in facts.columns if c not in keys + ["val"]})
    return pd.concat([facts[~dei], agg[facts.columns]], ignore_index=True)


def fy_history(store: _ItemStore, not_after, n: int = N_HISTORY) -> list[tuple]:
    """Last n fiscal years [(end, value)] with end <= not_after, latest first (ends at least 300 days apart)."""
    out = []
    if store is None:
        return out
    i = bisect.bisect_right(store.ends, not_after + 3 * DAY)
    while i > 0 and len(out) < n:
        e = store.ends[i - 1]
        i -= 1
        if out and (out[-1][0] - e) / DAY < 300:
            continue
        for d, s, v in store.durations_at(e):
            if _is_fy(d):
                out.append((e, v))
                break
    return out


def company_snapshots(facts: pd.DataFrame, filings: pd.DataFrame) -> list[dict]:
    """Snapshots of one CIK. facts: rows of that CIK (tag, start, end, val, accn, form, filed, accepted);
    filings: that CIK's periodic domestic filings (accn, form, accepted, report_date)."""
    if facts.empty or filings.empty:
        return []
    facts = combine_share_classes(facts).sort_values("accepted", kind="stable")
    stores: dict[str, _ItemStore] = {}
    f_acc = facts["accepted"].to_numpy()
    f_tag = facts["tag"].astype(str).to_numpy()
    f_start = facts["start"].to_numpy()
    f_end = facts["end"].to_numpy()
    f_val = facts["val"].to_numpy(dtype=float)
    f_accn = facts["accn"].astype(str).to_numpy()
    max_end_by_accn = facts.groupby(facts["accn"].astype(str))["end"].max()
    out, j = [], 0
    filings = filings.sort_values("accepted", kind="stable")
    cik = int(facts["cik"].iloc[0])
    for row in filings.itertuples(index=False):
        acc = np.datetime64(row.accepted)
        while j < len(f_acc) and f_acc[j] <= acc:
            it, prio = _TAG_ITEM[f_tag[j]]
            st = None if pd.isna(f_start[j]) else f_start[j]
            if not np.isnan(f_val[j]) and not pd.isna(f_end[j]):
                stores.setdefault(it, _ItemStore()).add(st, f_end[j], prio, f_acc[j], f_val[j])
            j += 1
        pe = row.report_date
        if pd.isna(pe):
            pe = max_end_by_accn.get(str(row.accn))
            if pe is None or pd.isna(pe):
                continue
        pe = np.datetime64(pd.Timestamp(pe), "ns")
        snap = {"cik": cik, "accn": row.accn, "form": row.form, "accepted": row.accepted,
                "period_end": pd.Timestamp(pe)}
        for it in FLOW_ITEMS:
            snap[f"ttm_{it}"] = ttm(stores.get(it), pe)
        for it in HISTORY_ITEMS:
            h = fy_history(stores.get(it), pe)
            for k in range(N_HISTORY):
                snap[f"fy{k}_{it}"] = h[k][1] if k < len(h) else np.nan
            if it == "revenue":
                snap["fy0_end"] = pd.Timestamp(h[0][0]) if h else pd.NaT
        for it in STOCK_ITEMS:
            s = stores.get(it)
            snap[it] = s.instant(pe) if s is not None else np.nan
        for it in PERIOD_ITEMS:
            snap[it] = period_value(stores.get(it), pe)
        for it in DEI_ITEMS:
            s = stores.get(it)
            v, e = (s.latest_instant(acc) if s is not None else (np.nan, None))
            snap[it] = v
            snap[f"{it}_date"] = pd.Timestamp(e) if e is not None else pd.NaT
        out.append(snap)
    return out


def derive(snap: pd.DataFrame) -> pd.DataFrame:
    """Columns built from the raw snapshot items (no prices): debt, net debt pieces, shares, FCF."""
    s = snap
    nc = s["debt_noncurrent"]
    total_alt = s["debt_total"]
    cur = s["debt_current"].fillna(0)
    stb = s["short_borrowings"].fillna(0)
    debt = np.where(nc.notna(), nc + cur + stb, np.where(total_alt.notna(), total_alt + stb, cur + stb))
    s["debt"] = debt
    s["debt_reported"] = nc.notna() | total_alt.notna() | s["debt_current"].notna() | s["short_borrowings"].notna()
    dei, gaap, dil = s["shares_cover"], s["shares_gaap"], s["diluted_shares_q"]
    s["shares"] = np.where(dei > 0, dei, np.where(gaap > 0, gaap, np.where(dil > 0, dil, np.nan)))
    s["shares_source"] = np.where(dei > 0, "dei", np.where(gaap > 0, "gaap", np.where(dil > 0, "diluted_avg", "none")))
    s["ttm_fcf"] = s["ttm_cfo"] - s["ttm_capex"].fillna(0)
    for k in range(N_HISTORY):
        s[f"fy{k}_fcf"] = s[f"fy{k}_cfo"] - s[f"fy{k}_capex"].fillna(0)
    return s


def build_snapshots(overwrite: bool = False, workers: int = 3, facts_dir: Path | None = None) -> pd.DataFrame:
    """Snapshots for every facts part (resumable per part). Returns a per-part summary."""
    from joblib import Parallel, delayed
    e = E.edgar_dir()
    filings = pd.read_parquet(e / "filings.parquet", columns=["cik", "accn", "form", "accepted", "filing_date",
                                                               "report_date"])
    filings["form"] = filings["form"].astype(str)
    filings = filings[filings["form"].isin(DOMESTIC_FORMS)]
    filings["accepted"] = filings["accepted"].fillna(filings["filing_date"] + pd.Timedelta(hours=17, minutes=30))
    filings = filings[filings["filing_date"] <= "2025-12-31"]  # 2026 sealed
    parts = sorted(Path(facts_dir or e / "facts").glob("part-*.parquet"))
    out_dir = fv_dir() / "snapshots"
    out_dir.mkdir(exist_ok=True)
    man = out_dir / "manifest.json"
    meta = {"version": FV_PANEL_VERSION, "n_parts": len(parts)}
    if man.exists() and json.loads(man.read_text()).get("version") != FV_PANEL_VERSION:
        if not overwrite:
            raise RuntimeError(f"{out_dir} was built by another fv_panel version; pass overwrite=True")
        for p in out_dir.glob("part-*.parquet"):
            p.unlink()
    man.write_text(json.dumps(meta))
    todo = [p for p in parts if overwrite or not (out_dir / p.name).exists()]
    print(f"{len(todo)} of {len(parts)} facts parts to process, {workers} workers")
    res = Parallel(n_jobs=workers, verbose=5)(delayed(_snapshot_part)(p, filings, out_dir) for p in todo)
    return pd.DataFrame(res)


def _snapshot_part(part: Path, filings: pd.DataFrame, out_dir: Path) -> dict:
    t0 = time.time()
    f = pd.read_parquet(part, columns=["cik", "taxonomy", "tag", "unit", "start", "end", "val", "accn", "form",
                                       "filed"])
    f = f[f["tag"].astype(str).isin(ALL_TAGS) & f["form"].astype(str).isin(DOMESTIC_FORMS)
          & f["taxonomy"].astype(str).isin(["us-gaap", "dei"])
          & f["unit"].astype(str).isin(["USD", "shares"])
          & (f["filed"] <= "2025-12-31")]
    acc = filings.set_index("accn")["accepted"]
    f["accn"] = f["accn"].astype(str)
    f["accepted"] = f["accn"].map(acc)
    f["accepted"] = f["accepted"].fillna(f["filed"] + pd.Timedelta(hours=17, minutes=30))
    fil = filings[filings["cik"].isin(f["cik"].unique())]
    fil_by = dict(tuple(fil.groupby("cik")))
    rows = []
    for cik, g in f.groupby("cik", sort=False):
        if cik in fil_by:
            rows += company_snapshots(g, fil_by[cik])
    snap = derive(pd.DataFrame(rows)) if rows else pd.DataFrame()
    snap.to_parquet(out_dir / part.name, index=False)
    return {"part": part.name, "ciks": f["cik"].nunique(), "snapshots": len(snap),
            "seconds": round(time.time() - t0, 1)}


def load_snapshots() -> pd.DataFrame:
    parts = sorted((fv_dir() / "snapshots").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError("no snapshots: run build_snapshots()")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# ================================================================================ prices

def month_end_prices(symbols: list[str], years=range(2016, 2026), workers: int = 4) -> pd.DataFrame:
    """Month-end raw and adjusted closes from the Alpaca 1Day caches (bars/1Day_raw, bars/1Day).

    Per (symbol, month): last bar date in the month, raw close, adjusted close, median daily dollar volume
    (raw close x volume). Written to fair_value/month_end_prices.parquet.
    """
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=workers, verbose=1)(delayed(_symbol_months)(s, list(years)) for s in symbols)
    out = pd.concat([r for r in res if r is not None and len(r)], ignore_index=True)
    out.to_parquet(fv_dir() / "month_end_prices.parquet", index=False)
    return out


def _read_alpaca(symbol: str, years, sub: str) -> pd.DataFrame:
    root = vendor_dir("Alpaca") / "bars" / sub / safe_name(symbol)
    parts = [pd.read_parquet(root / f"{y}.parquet") for y in years if (root / f"{y}.parquet").exists()]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["t", "c", "v"])
    d = pd.concat(parts, ignore_index=True)
    d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return d[["t", "c", "v"]]


def _symbol_months(symbol: str, years) -> pd.DataFrame | None:
    raw = _read_alpaca(symbol, years, "1Day_raw")
    if raw.empty:
        return None
    adj = _read_alpaca(symbol, years, "1Day").rename(columns={"c": "c_adj"})[["t", "c_adj"]]
    d = raw.merge(adj, on="t", how="left").sort_values("t")
    d["dv"] = d["c"] * d["v"]
    d["month"] = d["t"].dt.to_period("M")
    g = d.groupby("month")
    m = pd.DataFrame({"last_date": g["t"].max(), "close_raw": g["c"].last(), "close_adj": g["c_adj"].last(),
                      "dollar_vol_med": g["dv"].median(), "n_days": g["t"].count()}).reset_index()
    m.insert(0, "symbol", symbol)
    m["month_end"] = m["month"].dt.to_timestamp(how="end").dt.normalize()
    return m.drop(columns="month")


# ================================================================================ symbol assignment

SOURCE_RANK = {"fsds": 0, "primary": 0, "current": 1}
_DERIV_SUFFIXES = ("WS", "W", "U", "R", "RT")


def is_derivative_symbol(th: pd.DataFrame) -> pd.Series:
    """True for warrant / unit / right symbols (ZCARW, ABC.WS, ABCU) whose base symbol is also a candidate of the
    same CIK. Only the common stock is kept (SEC current tickers list all of a company's listed securities)."""
    sym = th["symbol"].astype(str)
    have = set(zip(th["cik"], sym))
    out = np.zeros(len(th), dtype=bool)
    for i, (c, x) in enumerate(zip(th["cik"], sym)):
        for suf in _DERIV_SUFFIXES:
            for sep in (".", ""):
                tail = sep + suf
                if x.endswith(tail) and len(x) > len(tail) and (c, x[: -len(tail)]) in have:
                    out[i] = True
    return pd.Series(out, index=th.index)


def symbol_windows(th: pd.DataFrame, status: pd.DataFrame, pad_days: int = 400) -> pd.DataFrame:
    """Date windows in which each (CIK, symbol) candidate may be the CIK's trading symbol.

    fsds / primary candidates: [first_filed - pad, last_filed + pad]; 'core' = [first_filed, last_filed + pad].
    current tickers: from the last use of a *different* prefix symbol by the same CIK (a rename: Overstock
    OSTK -> BYON -> BBBY), else from the first periodic filing - pad, to the last periodic filing + pad.
    """
    th = th.copy()
    th["symbol"] = th["symbol"].str.replace("-", ".", regex=False)
    th = th[~is_derivative_symbol(th)]
    pref = th[th["source"] != "current"]
    w = pref.groupby(["cik", "symbol"]).agg(first=("first_filed", "min"), last=("last_filed", "max")).reset_index()
    w["w0"] = w["first"] - pd.Timedelta(days=pad_days)
    w["core0"] = w["first"]
    w["w1"] = w["last"] + pd.Timedelta(days=pad_days)
    w["source"] = "prefix"
    cur = th[th["source"] == "current"][["cik", "symbol"]].drop_duplicates()
    cur = cur.merge(w[["cik", "symbol"]].assign(_in=True), on=["cik", "symbol"], how="left")
    cur = cur[cur["_in"].isna()].drop(columns="_in")  # current symbols already seen as prefixes keep that window
    st = status.set_index("cik")
    other_last = (pref.merge(cur[["cik", "symbol"]].rename(columns={"symbol": "cur"}), on="cik")
                  .query("symbol != cur").groupby(["cik", "cur"])["last_filed"].max())
    rows = []
    for r in cur.itertuples(index=False):
        if r.cik not in st.index:
            continue
        ol = other_last.get((r.cik, r.symbol))
        w0 = ol if ol is not None and not pd.isna(ol) else st.at[r.cik, "first_periodic"] - pd.Timedelta(days=pad_days)
        rows.append({"cik": r.cik, "symbol": r.symbol, "w0": w0, "core0": w0,
                     "w1": st.at[r.cik, "last_periodic"] + pd.Timedelta(days=pad_days), "source": "current"})
    w = pd.concat([w[["cik", "symbol", "w0", "core0", "w1", "source"]], pd.DataFrame(rows)], ignore_index=True)
    # A Form 25 / 15 filed after (or up to a year before) the window's last filing ends the window 30 days
    # later: a symbol reused by another company inside the padding must not be attached to the dead CIK.
    if "first_delist_form" in status.columns:
        dl = w["cik"].map(status.set_index("cik")["first_delist_form"])
        cap = dl + pd.Timedelta(days=30)
        use = dl.notna() & (dl >= w["w1"] - pd.Timedelta(days=pad_days + 365)) & (cap < w["w1"])
        w.loc[use, "w1"] = cap[use]
    return w


def assign_symbols(windows: pd.DataFrame, prices: pd.DataFrame, max_stale_days: int = 7) -> pd.DataFrame:
    """(cik, month_end) -> symbol. A candidate is valid at month end d if d is inside its window and the
    symbol has a bar within max_stale_days before d. A symbol claimed by several CIKs goes to the CIK whose
    core window (first actual filing onwards) contains d; still ambiguous -> dropped (counted). A CIK with
    several valid symbols keeps the one with the largest median dollar volume.
    Returns cik, month_end, symbol, close_raw, close_adj, dollar_vol_med, last_date, ambiguous_dropped.
    """
    p = prices[["symbol", "month_end", "last_date", "close_raw", "close_adj", "dollar_vol_med"]]
    p = p[(p["month_end"] - p["last_date"]).dt.days <= max_stale_days]
    m = windows.merge(p, on="symbol")
    m = m[(m["month_end"] >= m["w0"]) & (m["month_end"] <= m["w1"])]
    m["in_core"] = m["month_end"] >= m["core0"]
    n = m.groupby(["symbol", "month_end"])["cik"].transform("nunique")
    multi = m[n > 1]
    keep_multi = multi[multi["in_core"]]
    k2 = keep_multi.groupby(["symbol", "month_end"])["cik"].transform("nunique")
    keep_multi = keep_multi[k2 == 1]
    dropped = multi.groupby(["symbol", "month_end"]).ngroups - keep_multi.groupby(["symbol", "month_end"]).ngroups
    m = pd.concat([m[n == 1], keep_multi], ignore_index=True)
    m = m.sort_values(["cik", "month_end", "dollar_vol_med"], ascending=[True, True, False])
    m = m.drop_duplicates(["cik", "month_end"], keep="first")
    out = m[["cik", "month_end", "symbol", "close_raw", "close_adj", "dollar_vol_med", "last_date"]].reset_index(drop=True)
    out.attrs["ambiguous_dropped"] = int(dropped)
    return out


# ================================================================================ monthly panel

def monthly_panel(snap: pd.DataFrame, assigned: pd.DataFrame, companies: pd.DataFrame,
                  months: pd.DatetimeIndex | None = None, max_age_days: int = 460,
                  prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join each (cik, month_end) price row to the latest snapshot accepted strictly before the month-end
    date (00:00 of that day, so the filing is public at least one session before), then compute market
    cap and EV. Snapshots older than max_age_days (by period end) are not used.
    """
    a = assigned.copy()
    if months is not None:
        a = a[a["month_end"].isin(months)]
    a = a[a["month_end"] <= "2025-12-31"]
    s = snap.sort_values("accepted")
    s = s[s["form"].astype(str).isin(DOMESTIC_FORMS)]
    a = a.sort_values("month_end")
    p = pd.merge_asof(a, s, left_on="month_end", right_on="accepted", by="cik", direction="backward",
                      allow_exact_matches=False)
    p = p[p["accepted"].notna()]
    p = p[(p["month_end"] - p["period_end"]).dt.days <= max_age_days]
    p["mcap"] = p["close_raw"] * p["shares"]
    claims = (p["debt"].fillna(0) + p["preferred"].fillna(0) + p["minority_interest"].fillna(0))
    p["ev"] = p["mcap"] + claims - p["cash"].fillna(0) - p["short_investments"].fillna(0)
    p["mcap_to_float"] = p["mcap"] / p["public_float"].where(p["public_float"] > 0)
    # Mapping / share-count check. Public float is valued at the float date (last day of Q2), not today, so
    # compare share COUNTS: implied float shares = public float / raw close at the float date.
    p["float_price"] = np.nan
    if prices is not None:
        fp = prices[["symbol", "month_end", "close_raw"]].rename(columns={"month_end": "_fm", "close_raw": "float_price"})
        p["_fm"] = p["public_float_date"] + pd.offsets.MonthEnd(0)
        p = p.drop(columns="float_price").merge(fp, on=["symbol", "_fm"], how="left").drop(columns="_fm")
    p["shares_to_float"] = p["shares"] / (p["public_float"].where(p["public_float"] > 0) / p["float_price"])
    c = companies.set_index("cik")
    p["sic"] = p["cik"].map(c["sic"])
    sic = pd.to_numeric(p["sic"], errors="coerce")
    p["is_financial"] = sic.between(6000, 6799)
    p["is_blank_check"] = sic.eq(6770)
    p["name"] = p["cik"].map(c["name"])
    # shares outstanding should be >= float shares (float excludes insiders) and not absurdly larger; outside
    # [0.8, 50] flags a wrong symbol, wrong share units or a share class missing.
    p["mcap_suspect"] = ~p["shares_to_float"].between(0.8, 50) & p["shares_to_float"].notna()
    p["age_days"] = (p["month_end"] - p["period_end"]).dt.days
    return p.reset_index(drop=True)


def exits(filings: pd.DataFrame, prices: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """Per CIK whose last assigned month is before 2025-12: last price date and the exit type
    ('acquired' if a merger / tender form (DEFM14A, DEFM14C, PREM14A, SC TO-T, SC 14D9) was filed within
    18 months before the last price, else 'delisted_other'), plus the Form 25 / 15 date if any."""
    last = assigned.groupby("cik").agg(last_month=("month_end", "max"), symbol=("symbol", "last"),
                                       last_date=("last_date", "max"))
    last = last[last["last_month"] < pd.Timestamp("2025-12-31")]
    f = filings[filings["cik"].isin(last.index)].copy()
    f["form"] = f["form"].astype(str)
    mna = {"DEFM14A", "DEFM14C", "PREM14A", "SC TO-T", "SC 14D9"}
    groups = dict(tuple(f.groupby("cik")))
    empty = f.iloc[:0]
    out = []
    for cik, r in last.iterrows():
        g = groups.get(cik, empty)
        win = g[(g["filing_date"] <= r["last_date"] + pd.Timedelta(days=60))
                & (g["filing_date"] >= r["last_date"] - pd.Timedelta(days=548))]
        m = win[win["form"].isin(mna)]
        d = g[g["form"].isin(E.DELIST_FORMS) & (g["filing_date"] >= r["last_date"] - pd.Timedelta(days=60))]
        out.append({"cik": cik, "symbol": r["symbol"], "last_date": r["last_date"],
                    "exit_type": "acquired" if len(m) else "delisted_other",
                    "mna_form": m["form"].iloc[-1] if len(m) else None,
                    "mna_accn": m["accn"].iloc[-1] if len(m) else None,
                    "delist_form_date": d["filing_date"].min() if len(d) else pd.NaT})
    return pd.DataFrame(out)
