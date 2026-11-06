"""CME index-futures tick data for the Futures_Order_Flow branch (FOF 001).

Three sources are normalised to one tick table per root (``TICK_COLUMNS``):

* Databento GLBX.MDP3 ``tbbo`` (DEV/TEST, 2023-2025): exchange aggressor flag.
* NinjaTrader 8 tick exports (sealed Holdout A, 2026 -, plus the Nov-Dec 2025 bridge).
* The NinjaScript recorder ``tools/ninjatrader/OrderFlowRecorder.cs`` (Holdout B, forward).

Discipline enforced in code, not left to the notebook:

* nothing from 2026 is downloaded from Databento;
* ``load_ticks`` refuses ranges that reach ``HOLDOUT_START`` unless ``allow_holdout=True``;
* paid Databento requests are logged in a ledger and stop at ``BUDGET_USD``.

Cache layout under <EQUITY_TRADING_DIR>:
  Databento/ledger.csv                                one row per paid request
  Databento/<schema>/<ROOT>/<YYYY-MM>.parquet         normalised ticks
  NinjaTrader/export/<ROOT>/*.txt                     raw NinjaTrader exports (Ivan copies them here)
  NinjaTrader/ticks/<ROOT>/<YYYY-MM>.parquet          normalised, stitched front month
  NinjaTrader/recorder/*.csv                          raw recorder files
  NinjaTrader/recorded/<ROOT>/<YYYY-MM>.parquet       normalised recorder ticks
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import vendor_dir

FUTURES_VERSION = "0.1.0"
DATASET = "GLBX.MDP3"
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
BUDGET_USD = 115.0
CHICAGO = "America/Chicago"
NY = "America/New_York"

# root: (tick size in index points, USD per tick per contract)
SPECS = {
    "ES": (0.25, 12.50), "NQ": (0.25, 5.00), "YM": (1.0, 5.00),
    "MES": (0.25, 1.25), "MNQ": (0.25, 0.50), "MYM": (1.0, 0.50),
}

TICK_COLUMNS = ["ts", "price", "size", "bid", "ask", "sign", "sign_method", "contract", "source"]
SOURCES = {"databento": ("Databento", "tbbo"), "ninjatrader": ("NinjaTrader", "ticks"),
           "recorder": ("NinjaTrader", "recorded")}


class HoldoutError(RuntimeError):
    """Raised when code would read or buy data from the sealed holdout (2026 onward)."""


class BudgetError(RuntimeError):
    """Raised when a Databento request would take cumulative spend above the budget."""


# ---------------------------------------------------------------------------------------------
# Holdout guard and cache reading
# ---------------------------------------------------------------------------------------------

def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def assert_pre_holdout(end, what: str = "request") -> None:
    """``end`` is exclusive. Anything ending after HOLDOUT_START touches the sealed holdout."""
    if _utc(end) > HOLDOUT_START:
        raise HoldoutError(f"{what} ends {_utc(end)} > {HOLDOUT_START}: 2026 is sealed (FOF 001 §2).")


def cache_dir(source: str, root: str) -> Path:
    vendor, sub = SOURCES[source]
    d = vendor_dir(vendor) / sub / root
    d.mkdir(parents=True, exist_ok=True)
    return d


def _months(start, end) -> list[pd.Timestamp]:
    """Month starts m with [m, next m) overlapping [start, end)."""
    s, e = _utc(start), _utc(end)
    m = s.tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    out = []
    while m < e:
        out.append(m)
        m = m + pd.offsets.MonthBegin(1)
    return out


def write_months(df: pd.DataFrame, source: str, root: str) -> list[Path]:
    """Write a normalised tick table into monthly Parquet files (replacing those months)."""
    if df.empty:
        return []
    out = []
    month = df["ts"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    for key, part in df.groupby(month, sort=True):
        p = cache_dir(source, root) / f"{key}.parquet"
        part.reset_index(drop=True).to_parquet(p, index=False)
        out.append(p)
    return out


def load_ticks(root: str, start, end, source: str = "databento", allow_holdout: bool = False,
               columns: list[str] | None = None) -> pd.DataFrame:
    """Normalised ticks for ``root`` in [start, end) from the monthly cache."""
    if not allow_holdout:
        assert_pre_holdout(end, f"load_ticks({root}, {source})")
    s, e = _utc(start), _utc(end)
    parts = []
    for m in _months(s, e):
        p = cache_dir(source, root) / f"{m:%Y-%m}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=columns))
    if not parts:
        return pd.DataFrame(columns=columns or TICK_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["ts"] >= s) & (df["ts"] < e)].reset_index(drop=True)


def holdout_day_counts(root: str, source: str) -> pd.DataFrame:
    """Trade dates and tick counts per month in the holdout. Reads only timestamps (no prices)."""
    rows = []
    for p in sorted(cache_dir(source, root).glob("*.parquet")):
        if pd.Timestamp(p.stem + "-01", tz="UTC") < HOLDOUT_START:
            continue
        ts = pd.read_parquet(p, columns=["ts"])["ts"]
        rows.append({"month": p.stem, "trade_dates": int(cme_trade_date(ts).nunique()), "ticks": len(ts)})
    return pd.DataFrame(rows, columns=["month", "trade_dates", "ticks"])


# ---------------------------------------------------------------------------------------------
# Calendar, signing, roll
# ---------------------------------------------------------------------------------------------

def cme_trade_date(ts: pd.Series) -> pd.Series:
    """CME Globex trade date: the session that opens at 17:00 Chicago belongs to the next day."""
    local = pd.DatetimeIndex(ts).tz_convert(CHICAGO)
    d = local.normalize().tz_localize(None) + pd.to_timedelta((local.hour >= 17).astype(int), unit="D")
    return pd.Series(d.date, index=getattr(ts, "index", None))


def regular_hours(ts: pd.Series) -> np.ndarray:
    """True for 09:30 <= New York time < 16:00 (cash session)."""
    local = pd.DatetimeIndex(ts).tz_convert(NY)
    mins = local.hour * 60 + local.minute
    return np.asarray((mins >= 570) & (mins < 960) & (local.dayofweek < 5))


def sign_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Quote rule against the carried bid/ask, tick rule inside the spread, 0 otherwise.

    Rows that already carry an exchange sign (``sign_method == 'exchange'``) are left alone.
    The tick rule uses the last price change within the same contract (no look-ahead).
    """
    df = df.copy()
    if "sign_method" not in df:
        df["sign_method"] = "none"
        df["sign"] = 0
    todo = df["sign_method"].ne("exchange").to_numpy()
    p, b, a = df["price"].to_numpy(float), df["bid"].to_numpy(float), df["ask"].to_numpy(float)
    valid_q = np.isfinite(b) & np.isfinite(a) & (a > b)
    buy, sell = valid_q & (p >= a), valid_q & (p <= b)
    chg = df.groupby("contract", sort=False)["price"].diff()
    last_dir = np.sign(chg.where(chg != 0)).groupby(df["contract"], sort=False).ffill().to_numpy()
    tick = np.nan_to_num(last_dir, nan=0.0)
    sign = np.where(buy, 1, np.where(sell, -1, tick)).astype(np.int8)
    method = np.where(buy | sell, "quote", np.where(tick != 0, "tick", "none"))
    df.loc[todo, "sign"] = sign[todo]
    df.loc[todo, "sign_method"] = method[todo]
    df["sign"] = df["sign"].astype(np.int8)
    return df


def daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Volume per CME trade date (rows) and contract (columns)."""
    td = cme_trade_date(df["ts"]).to_numpy()
    return df.assign(_td=td).pivot_table(index="_td", columns="contract", values="size", aggfunc="sum",
                                         fill_value=0).sort_index()


def front_by_prior_volume(df: pd.DataFrame) -> pd.Series:
    """Contract to trade on each trade date: highest volume on the previous trade date.

    Same rule as Databento's ``.v.0`` continuous symbol. The first trade date has no prior
    day and is dropped.
    """
    vol = daily_volume(df)
    prev = vol.shift(1)
    ok = prev.sum(axis=1) > 0
    return prev[ok].idxmax(axis=1).rename("contract")


def stitch_front(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ticks of the front contract chosen by ``front_by_prior_volume``."""
    if df.empty:
        return df
    front = front_by_prior_volume(df)
    td = cme_trade_date(df["ts"]).to_numpy()
    keep = pd.Series(td).map(front).to_numpy() == df["contract"].to_numpy()
    return df[keep].reset_index(drop=True)


def roll_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Trade dates on which the front contract of a stitched tick table changes."""
    td = cme_trade_date(df["ts"]).to_numpy()
    per_day = pd.Series(df["contract"].to_numpy(), index=td).groupby(level=0).first()
    change = per_day.ne(per_day.shift(1)) & per_day.shift(1).notna()
    return pd.DataFrame({"trade_date": per_day.index[change], "from": per_day.shift(1)[change].to_numpy(),
                         "to": per_day[change].to_numpy()})


def _finish(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df["source"] = source
    df["size"] = df["size"].astype(np.int64)
    df = df.sort_values(["ts"], kind="stable").reset_index(drop=True)
    return df[TICK_COLUMNS]


# ---------------------------------------------------------------------------------------------
# Databento
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    root: str
    start: pd.Timestamp       # inclusive, UTC month start
    end: pd.Timestamp         # exclusive
    schema: str = "tbbo"

    @property
    def symbol(self) -> str:
        return f"{self.root}.v.0"

    @property
    def key(self) -> str:
        return f"{self.schema}/{self.root}/{self.start:%Y-%m}"


def plan_requests(order: list[tuple[str, int]], schema: str = "tbbo") -> list[Request]:
    """Monthly requests in spend order, e.g. [("NQ", 2025), ("ES", 2025), ("NQ", 2024)]."""
    out = []
    for root, year in order:
        for m in _months(f"{year}-01-01", f"{year + 1}-01-01"):
            r = Request(root, m, m + pd.offsets.MonthBegin(1), schema)
            assert_pre_holdout(r.end, r.key)
            out.append(r)
    return out


def client(key: str | None = None):
    """Databento historical client; the key comes from DATABENTO_API_KEY (.env) by default."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import databento as db  # imported lazily so tests do not need it
    return db.Historical(key) if key else db.Historical()


def _query(r: Request) -> dict:
    return dict(dataset=DATASET, symbols=[r.symbol], schema=r.schema, stype_in="continuous",
                start=r.start.strftime("%Y-%m-%dT%H:%M:%S"), end=r.end.strftime("%Y-%m-%dT%H:%M:%S"))


def estimate_costs(cl, requests: list[Request]) -> pd.DataFrame:
    """Quoted USD cost and size of each request (free: metadata calls are not billed)."""
    rows = []
    for r in requests:
        assert_pre_holdout(r.end, r.key)
        q = _query(r)
        cost = float(cl.metadata.get_cost(**q))
        size = int(cl.metadata.get_billable_size(**q))
        rows.append({"key": r.key, "root": r.root, "month": f"{r.start:%Y-%m}", "schema": r.schema,
                     "cost_usd": cost, "gb": size / 1e9,
                     "cached": (cache_dir("databento", r.root) / f"{r.start:%Y-%m}.parquet").exists()})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cum_cost_usd"] = df["cost_usd"].where(~df["cached"], 0.0).cumsum()
    return df


def ledger_path() -> Path:
    return vendor_dir("Databento") / "ledger.csv"


def ledger() -> pd.DataFrame:
    p = ledger_path()
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["utc", "key", "cost_usd", "records"])


def spent_usd() -> float:
    return float(ledger()["cost_usd"].sum())


def normalise_databento(raw: pd.DataFrame) -> pd.DataFrame:
    """Databento tbbo/trades frame (``to_df()`` output) -> TICK_COLUMNS."""
    side = raw["side"].astype(str).to_numpy()
    sign = np.select([side == "B", side == "A"], [1, -1], 0).astype(np.int8)
    ts = raw["ts_event"] if "ts_event" in raw else pd.Series(raw.index, index=raw.index)
    df = pd.DataFrame({
        "ts": pd.to_datetime(ts, utc=True).to_numpy(),
        "price": raw["price"].to_numpy(float),
        "size": raw["size"].to_numpy(),
        "bid": raw["bid_px_00"].to_numpy(float) if "bid_px_00" in raw else np.nan,
        "ask": raw["ask_px_00"].to_numpy(float) if "ask_px_00" in raw else np.nan,
        "sign": sign,
        "sign_method": np.where(sign != 0, "exchange", "none"),
        "contract": raw["instrument_id"].astype(str).to_numpy(),
    })
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if "action" in raw:      # tbbo is trades only; keep the guard for other schemas
        df = df[raw["action"].astype(str).to_numpy() == "T"]
    return _finish(df, "databento")


def raw_path(r: Request) -> Path:
    d = vendor_dir("Databento") / "raw" / r.schema / r.root
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{r.start:%Y-%m}.dbn.zst"


def normalise_raw_file(path: Path | str, r: Request, chunk: int = 2_000_000) -> int:
    """Raw DBN file -> monthly Parquet, in chunks so a month of NQ fits in 16 GB of RAM."""
    import databento as db
    store = db.DBNStore.from_file(path)
    parts = []
    for raw in store.to_df(count=chunk):
        t = normalise_databento(raw)
        parts.append(t[(t["ts"] >= r.start) & (t["ts"] < r.end)])
    ticks = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TICK_COLUMNS)
    ticks.to_parquet(cache_dir("databento", r.root) / f"{r.start:%Y-%m}.parquet", index=False)
    return len(ticks)


def download(cl, requests: list[Request], budget: float = BUDGET_USD) -> pd.DataFrame:
    """Buy and cache requests in order. Stops before the first one that would exceed ``budget``.

    * A month with normalised Parquet is skipped (no charge).
    * A month whose raw DBN file exists is re-normalised from disk (no charge).
    * Otherwise the quoted cost is checked against the ledger total, the raw DBN file is
      streamed to ``Databento/raw/`` and the request is logged before normalising, so an
      interrupted run never under-counts spend or pays twice.
    """
    log = []
    for r in requests:
        assert_pre_holdout(r.end, r.key)
        target = cache_dir("databento", r.root) / f"{r.start:%Y-%m}.parquet"
        raw = raw_path(r)
        if target.exists():
            log.append({"key": r.key, "status": "cached", "cost_usd": 0.0, "records": None})
            continue
        if raw.exists():
            n = normalise_raw_file(raw, r)
            log.append({"key": r.key, "status": "normalised from raw", "cost_usd": 0.0, "records": n})
            continue
        q = _query(r)
        cost = float(cl.metadata.get_cost(**q))
        spent = spent_usd()
        if spent + cost > budget:
            log.append({"key": r.key, "status": f"stopped: {spent:.2f} + {cost:.2f} > budget {budget:.2f}",
                        "cost_usd": cost, "records": None})
            break
        # logged before the call: an interrupted transfer may still be billed, so count it
        row = pd.DataFrame([{"utc": pd.Timestamp.now(tz="UTC").isoformat(), "key": r.key,
                             "cost_usd": cost, "records": None}])
        row.to_csv(ledger_path(), mode="a", header=not ledger_path().exists(), index=False)
        tmp = raw.with_suffix(".part")
        cl.timeseries.get_range(**q, path=tmp)
        tmp.replace(raw)
        n = normalise_raw_file(raw, r)
        log.append({"key": r.key, "status": "downloaded", "cost_usd": cost, "records": n})
    return pd.DataFrame(log)


# ---------------------------------------------------------------------------------------------
# NinjaTrader exports and the recorder
# ---------------------------------------------------------------------------------------------

_NT_NAME = re.compile(r"^(?P<contract>(?P<root>[A-Z0-9]+) \d{2}-\d{2})(?:\.(?P<kind>Last|Bid|Ask))?\.txt$")


def parse_nt_contract(name: str) -> tuple[str, str, str]:
    """'NQ 12-25.Last.txt' -> ('NQ', 'NQ 12-25', 'Last'). Files without a kind are 'Last'."""
    m = _NT_NAME.match(Path(name).name)
    if not m:
        raise ValueError(f"not a NinjaTrader tick export name: {name!r} (expected e.g. 'NQ 12-25.Last.txt')")
    return m["root"], m["contract"], m["kind"] or "Last"


def _nt_timestamps(col: pd.Series, tz: str) -> pd.Series:
    """'yyyyMMdd HHmmss[ fffffff]' in ``tz`` -> UTC timestamps (ns)."""
    s = col.astype(str).str.strip()
    base = pd.to_datetime(s.str.slice(0, 15), format="%Y%m%d %H%M%S")
    frac = s.str.slice(16).str.strip()
    ns = pd.to_numeric(frac.where(frac.str.len() > 0, "0").str.ljust(9, "0").str.slice(0, 9))
    t = base + pd.to_timedelta(ns, unit="ns")
    if tz.upper() == "UTC":
        return t.dt.tz_localize("UTC")
    return t.dt.tz_localize(tz, ambiguous="infer", nonexistent="raise").dt.tz_convert("UTC")


def read_nt_export(path: Path | str, tz: str) -> pd.DataFrame:
    """One NinjaTrader 8 tick export.

    5 fields ``time;last;bid;ask;volume`` -> ts, price, bid, ask, size.
    3 fields ``time;price;volume`` -> ts, price, size (a separate Last, Bid or Ask export).
    ``tz`` is the time zone NinjaTrader was set to when exporting; it is required on purpose.
    """
    raw = pd.read_csv(path, sep=";", header=None, dtype={0: str}, engine="c")
    ts = _nt_timestamps(raw[0], tz)
    if raw.shape[1] == 5:
        df = pd.DataFrame({"ts": ts, "price": raw[1].astype(float), "bid": raw[2].astype(float),
                           "ask": raw[3].astype(float), "size": raw[4].astype(np.int64)})
    elif raw.shape[1] == 3:
        df = pd.DataFrame({"ts": ts, "price": raw[1].astype(float), "size": raw[2].astype(np.int64)})
    else:
        raise ValueError(f"{path}: {raw.shape[1]} fields per line; expected 5 (last;bid;ask) or 3")
    return df


def attach_quotes(last: pd.DataFrame, bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Separate Last/Bid/Ask exports -> trades carrying the last quote strictly before each trade."""
    out = last.sort_values("ts", kind="stable")
    for side, q in (("bid", bid), ("ask", ask)):
        q = q.sort_values("ts", kind="stable")[["ts", "price"]].rename(columns={"price": side})
        out = pd.merge_asof(out, q, on="ts", direction="backward", allow_exact_matches=False)
    return out


def load_nt_root(folder: Path | str, tz: str) -> pd.DataFrame:
    """All contract exports of one root in ``folder`` -> signed, stitched front-month ticks."""
    files: dict[str, dict[str, Path]] = {}
    root_seen = set()
    for p in sorted(Path(folder).glob("*.txt")):
        root, contract, kind = parse_nt_contract(p.name)
        root_seen.add(root)
        files.setdefault(contract, {})[kind] = p
    if len(root_seen) > 1:
        raise ValueError(f"{folder}: several roots {sorted(root_seen)}; keep one root per folder")
    parts = []
    for contract, kinds in files.items():
        last = read_nt_export(kinds["Last"], tz)
        if "bid" not in last:
            if not {"Bid", "Ask"} <= kinds.keys():
                raise ValueError(f"{contract}: Last export has no bid/ask and no Bid/Ask exports to merge")
            last = attach_quotes(last, read_nt_export(kinds["Bid"], tz), read_nt_export(kinds["Ask"], tz))
        last["contract"] = contract
        parts.append(last)
    if not parts:
        return pd.DataFrame(columns=TICK_COLUMNS)
    df = pd.concat(parts, ignore_index=True).sort_values(["ts"], kind="stable").reset_index(drop=True)
    df = sign_trades(df)
    return _finish(stitch_front(df), "ninjatrader")


def read_recorder(paths: list[Path | str]) -> dict[str, pd.DataFrame]:
    """Recorder CSVs (``ts_utc,instrument,price,size,bid,ask``) -> {root: signed stitched ticks}."""
    if not paths:
        return {}
    raw = pd.concat([pd.read_csv(p, dtype={"instrument": str}) for p in paths], ignore_index=True)
    raw = raw.drop_duplicates()
    df = pd.DataFrame({"ts": pd.to_datetime(raw["ts_utc"], utc=True, format="ISO8601"),
                       "price": raw["price"].astype(float), "size": raw["size"].astype(np.int64),
                       "bid": raw["bid"].astype(float), "ask": raw["ask"].astype(float),
                       "contract": raw["instrument"].str.strip()})
    df["root"] = df["contract"].str.split(" ").str[0]
    out = {}
    for root, part in df.groupby("root"):
        part = sign_trades(part.drop(columns="root").sort_values("ts", kind="stable").reset_index(drop=True))
        out[root] = _finish(stitch_front(part), "recorder")
    return out


# ---------------------------------------------------------------------------------------------
# FOF 01 checks (data only: no prices or returns are compared)
# ---------------------------------------------------------------------------------------------

def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Ticks, volume and unsigned share per CME trade date."""
    td = cme_trade_date(df["ts"]).to_numpy()
    g = df.assign(_td=td, _unsigned=(df["sign"] == 0) * df["size"]).groupby("_td")
    out = pd.DataFrame({"ticks": g.size(), "volume": g["size"].sum(), "unsigned_volume": g["_unsigned"].sum()})
    out["unsigned_share"] = out["unsigned_volume"] / out["volume"]
    out.index.name = "trade_date"
    return out


def cme_trading_days(start, end) -> list:
    """CME equity-index trading days in [start, end) (pandas_market_calendars 'CME_Equity')."""
    import pandas_market_calendars as mcal
    end_incl = pd.Timestamp(end).tz_localize(None) - pd.Timedelta(days=1)
    days = mcal.get_calendar("CME_Equity").valid_days(pd.Timestamp(start).tz_localize(None), end_incl)
    return [d.date() for d in days]


def missing_trade_dates(cov: pd.DataFrame, expected) -> list:
    """Expected trading days (e.g. ``cme_trading_days``) with no ticks in ``cov``."""
    have = set(cov.index)
    return [d for d in expected if d not in have]


def minute_volume(df: pd.DataFrame) -> pd.Series:
    return df.set_index("ts")["size"].resample("1min").sum()


def best_lag_minutes(a: pd.DataFrame, b: pd.DataFrame, max_lag: int = 720) -> tuple[int, float]:
    """Lag (minutes) that maximises the correlation of per-minute volume; > 0 means ``a`` is late."""
    va, vb = minute_volume(a), minute_volume(b)
    grid = pd.date_range(min(va.index[0], vb.index[0]), max(va.index[-1], vb.index[-1]), freq="1min")
    x = va.reindex(grid, fill_value=0).to_numpy(float)
    y = vb.reindex(grid, fill_value=0).to_numpy(float)
    x, y = (x - x.mean()) / (x.std() or 1), (y - y.mean()) / (y.std() or 1)
    n = len(x)
    best, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(x[lag:], y[:n - lag])) / max(n - lag, 1)
        else:
            c = float(np.dot(x[:n + lag], y[-lag:])) / max(n + lag, 1)
        if c > best_c:
            best, best_c = lag, c
    return best, best_c


def sign_agreement(inferred: pd.DataFrame, exchange: pd.DataFrame) -> dict:
    """Match trades on (second, price, size, occurrence) and compare signs.

    Returns the match rate (by volume of ``exchange``) and the volume-weighted share of matched
    trades where the inferred sign equals the exchange aggressor sign.
    """
    def keyed(d):
        k = d.assign(_sec=d["ts"].dt.floor("s"))
        k["_n"] = k.groupby(["_sec", "price", "size"]).cumcount()
        return k[["_sec", "price", "size", "_n", "sign"]]
    m = keyed(exchange).merge(keyed(inferred), on=["_sec", "price", "size", "_n"], suffixes=("_x", "_i"))
    vol_x = float(exchange["size"].sum())
    signed = m[m["sign_x"] != 0]
    agree = float(signed.loc[signed["sign_x"] == signed["sign_i"], "size"].sum())
    return {"exchange_volume": vol_x, "matched_volume": float(m["size"].sum()),
            "match_rate": float(m["size"].sum()) / vol_x if vol_x else np.nan,
            "agreement": agree / float(signed["size"].sum()) if len(signed) else np.nan,
            "matched_trades": len(m)}


# ---------------------------------------------------------------------------------------------
# Synthetic data for tests and smoke runs
# ---------------------------------------------------------------------------------------------

def synthetic_ticks(start: str = "2025-11-03", days: int = 10, per_day: int = 4000, seed: int = 0,
                    contracts: tuple[str, str] = ("NQ 12-25", "NQ 03-26"), roll_day: int = 5) -> pd.DataFrame:
    """Random trades on a 0.25 grid with a one-tick book and a volume handover at ``roll_day``.

    Both contracts trade every day; the second becomes the volume leader from ``roll_day`` on.
    Signs are the true aggressor sides (``sign_method='exchange'``).
    """
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp(start, tz=CHICAGO) - pd.Timedelta(hours=7)          # 17:00 Chicago, prior evening
    for d in range(days):
        session = t0 + pd.Timedelta(days=d)
        for ci, c in enumerate(contracts):
            lead = (ci == 0) == (d < roll_day)
            n = per_day if lead else per_day // 10
            secs = np.sort(rng.uniform(0, 23 * 3600, n))
            mid = 20000 + 50 * ci + np.cumsum(rng.choice([-0.25, 0, 0.25], n))
            bid = np.floor(mid * 4) / 4
            ask = bid + 0.25
            side = rng.choice([1, -1], n)
            price = np.where(side > 0, ask, bid)
            ts = (session.tz_convert("UTC") + pd.to_timedelta(secs, unit="s")).floor("us")
            rows.append(pd.DataFrame({"ts": ts,
                                      "price": price, "size": rng.integers(1, 6, n), "bid": bid, "ask": ask,
                                      "sign": side.astype(np.int8), "sign_method": "exchange", "contract": c}))
    df = pd.concat(rows, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    return _finish(df, "synthetic")


def to_nt_lines(df: pd.DataFrame, tz: str = "UTC", five_fields: bool = True) -> list[str]:
    """Render ticks in NinjaTrader 8 export format (used by tests and the synthetic smoke run)."""
    local = df["ts"].dt.tz_convert(tz)
    stamp = local.dt.strftime("%Y%m%d %H%M%S") + " " + (local.dt.microsecond * 10 + local.dt.nanosecond // 100
                                                         ).astype(str).str.zfill(7)
    if five_fields:
        return (stamp + ";" + df["price"].map("{:.12g}".format) + ";" + df["bid"].map("{:.12g}".format) + ";"
                + df["ask"].map("{:.12g}".format) + ";" + df["size"].astype(str)).tolist()
    return (stamp + ";" + df["price"].map("{:.12g}".format) + ";" + df["size"].astype(str)).tolist()
